"""Pregame shadow-run and exact confirmed-review orchestration."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from pydantic import BaseModel

from analysis.forecasting import LineupState, run_forecast
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.engine import (
    build_semantic_market_inputs,
    resolve_forecast_input,
)
from analysis.forecasting.identity import (
    build_forecast_run_id_for_input,
    canonical_json,
)
from analysis.forecasting.validation import (
    PregameRelationshipError,
    PregameTimingError,
    validate_pregame_input,
)
from analysis.ledger import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    IdempotencyConflict,
    LedgerActiveCaptureHeadChangedError,
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerMarketHeadChangedError,
    LedgerReferenceError,
    LedgerStore,
    MarketObservationCommand,
    MarketObservationRecord,
    OfficialResultCommand,
    OfficialResultRecord,
    PregameLookup,
    ReviewRecord,
    linked_shadow_comparison,
)
from analysis.pipeline import LEGACY_MODEL_VERSION
from analysis.review import build_and_record_review
from analysis.review.calibration import (
    build_calibration_report,
    select_primary_rows,
)
from analysis.shadow import (
    ShadowRunIdempotencyKeys,
    build_shadow_capture_command,
    canonical_legacy_decision_payload,
    compare_shadow_forecast,
    deserialize_forecast_context,
    deserialize_shadow_comparison,
    record_shadow_run,
    require_recommended_review_eligibility,
    validate_frozen_market_output_binding,
    validate_recommended_review_contract,
    validate_shadow_forecast_context,
)

from .bundles import (
    DEFAULT_BUNDLE_ROOT,
    ManagedBundleCopy,
    ReplayOnlyPregameBundle,
    intrinsic_capture_key,
    load_historical_managed_bundle,
    persist_managed_bundle,
)
from .config import load_forecast_config
from .errors import (
    BundleValidationError,
    OperatorError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPartialProgress,
    OperatorPreconditionError,
)
from .reporting import (
    CalibrationReportArtifacts,
    DEFAULT_REPORT_DIR,
    _StaleCalibrationPublicationError,
    build_report_mapping,
    write_calibration_reports,
)
from .schemas import (
    CURRENT_PREGAME_AUTHORITY,
    CURRENT_PREGAME_SCHEMA_VERSION,
    ConfirmedBundle,
    HistoricalRelaxedConfirmedBundleV1,
    HistoricalRelaxedProjectedBundleV1,
    HistoricalStrictConfirmedBundleV1,
    HistoricalStrictProjectedBundleV1,
    ProjectedBundle,
    ReviewBundle,
    is_current_pregame_payload,
)


DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ledger" / "shadow.jsonl"
)

_Bundle = ProjectedBundle | ConfirmedBundle
_ReplayBundle = (
    ProjectedBundle
    | ConfirmedBundle
    | HistoricalRelaxedProjectedBundleV1
    | HistoricalStrictProjectedBundleV1
    | HistoricalRelaxedConfirmedBundleV1
    | HistoricalStrictConfirmedBundleV1
)
_T = TypeVar("_T")
_HEAD_CAS_MAX_ATTEMPTS = 3
_REPORT_PUBLICATION_MAX_ATTEMPTS = 3
_LEDGER_ERRORS = (
    IdempotencyConflict,
    LedgerCorruptionError,
    LedgerReferenceError,
)


@dataclass(frozen=True)
class PregameRunResult:
    """Records and immutable artifact produced by one pregame stage."""

    stage: str
    game_pk: int
    lineup_state: str
    capture: CaptureRecord
    forecast: ForecastRecord
    link: CaptureForecastLink
    managed_bundle: ManagedBundleCopy
    capture_replayed: bool
    forecast_replayed: bool
    link_replayed: bool


@dataclass(frozen=True)
class ReviewRunResult:
    """Exact frozen records and replay state produced by one review stage."""

    stage: str
    game_pk: int
    capture: CaptureRecord
    forecast: ForecastRecord
    link: CaptureForecastLink
    closing_observations: tuple[MarketObservationRecord, ...]
    official_result: OfficialResultRecord
    reviews: tuple[ReviewRecord, ...]
    managed_bundle: ManagedBundleCopy
    closing_replay_count: int
    official_result_replayed: bool
    review_replay_count: int
    classification_counts: Mapping[str, int]
    calibration_json_path: Path
    calibration_markdown_path: Path
    calibration_generation_dir: Path
    calibration_current_generation_path: Path
    calibration_report_replayed: bool
    calibration_sample_size: int
    confirmed_game_count: int
    confirmed_decision_count: int
    confirmed_decision_counts: Mapping[str, int]


@dataclass(frozen=True)
class _ClosingPreflight:
    existing: MarketObservationRecord | None
    expected_latest_record_id: str | None


@dataclass(frozen=True)
class _LedgerSnapshot:
    records: tuple[object, ...]

    def iter_records(self):
        return iter(self.records)


def _partial_progress_records(
    store: LedgerStore,
) -> tuple[object, ...] | None:
    """Best-effort authoritative reread that never masks the original error."""

    try:
        return _review_ledger_records(store)
    except Exception:
        return None


def _single_record(
    records: tuple[object, ...],
    record_type: type[_T],
    predicate: Callable[[_T], bool],
) -> _T | None:
    matches = tuple(
        record
        for record in records
        if isinstance(record, record_type) and predicate(record)
    )
    return matches[0] if len(matches) == 1 else None


def _ledger_operation(label: str, operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except LedgerHeadChangedError:
        raise
    except _LEDGER_ERRORS as error:
        raise OperatorLedgerError(f"{label}: {error}") from error
    except OSError as error:
        raise OperatorIOError(f"{label}: {error}") from error


def _pregame_operation(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except PregameRelationshipError as error:
        raise BundleValidationError("invalid pregame relationship") from error
    except PregameTimingError as error:
        raise OperatorPreconditionError("invalid pregame timing") from error


def _open_ledger(path: str | Path) -> LedgerStore:
    return _ledger_operation(
        "unable to open shadow ledger",
        lambda: LedgerStore(path),
    )


def _require_stage(bundle: _Bundle, expected_stage: str) -> None:
    if not isinstance(bundle, (ProjectedBundle, ConfirmedBundle)):
        raise TypeError(
            "bundle must be a validated projected or confirmed bundle"
        )
    if bundle.stage != expected_stage:
        raise OperatorPreconditionError(
            f"bundle stage {bundle.stage!r} cannot run as {expected_stage!r}"
        )


def _validate_game(bundle: _ReplayBundle, expected_stage: str) -> None:
    game = bundle.payload.game_input
    _pregame_operation(lambda: validate_pregame_input(game))

    lineup_states = (game.away_lineup.state, game.home_lineup.state)
    if expected_stage == "projected":
        if LineupState.PROJECTED not in lineup_states:
            raise OperatorPreconditionError(
                "projected stage requires at least one projected lineup"
            )
    elif lineup_states != (LineupState.CONFIRMED, LineupState.CONFIRMED):
        raise OperatorPreconditionError(
            "confirmed stage requires both lineups to be confirmed"
        )


def _capture_command(bundle: _ReplayBundle) -> CaptureCommand:
    return _pregame_operation(
        lambda: build_shadow_capture_command(bundle.payload.game_input)
    )


def _captures(store: LedgerStore) -> tuple[CaptureRecord, ...]:
    records = _ledger_operation(
        "unable to read shadow captures",
        lambda: tuple(store.iter_records("capture")),
    )
    return tuple(record for record in records if isinstance(record, CaptureRecord))


def _existing_keyed_capture(
    captures: tuple[CaptureRecord, ...],
    idempotency_key: str,
) -> CaptureRecord | None:
    return next(
        (
            capture
            for capture in captures
            if capture.envelope.idempotency_key == idempotency_key
        ),
        None,
    )


def _active_game_captures(
    captures: tuple[CaptureRecord, ...],
    game_pk: int,
) -> tuple[CaptureRecord, ...]:
    game_captures = tuple(
        capture for capture in captures if capture.game_pk == game_pk
    )
    superseded_ids = frozenset(
        capture.supersedes_capture_id
        for capture in game_captures
        if capture.supersedes_capture_id is not None
    )
    return tuple(
        capture
        for capture in game_captures
        if capture.captured_at < capture.first_pitch
        if capture.capture_id not in superseded_ids
    )


def _latest_active_capture(
    captures: tuple[CaptureRecord, ...],
    game_pk: int,
) -> CaptureRecord | None:
    active = _active_game_captures(captures, game_pk)
    if not active:
        return None
    return max(
        active,
        key=lambda capture: (
            capture.captured_at,
            capture.envelope.appended_at,
            capture.envelope.record_id,
        ),
    )


def _resolve_supersession_target(
    *,
    stage: str,
    existing_capture: CaptureRecord | None,
    active_head: CaptureRecord | None,
    active_confirmed_capture: CaptureRecord | None,
) -> str | None:
    if existing_capture is not None:
        return existing_capture.supersedes_capture_id

    if (
        stage == "projected"
        and active_confirmed_capture is not None
    ):
        raise OperatorPreconditionError(
            "new projected capture is not allowed after an active confirmed capture"
        )
    if active_head is None:
        return None
    if stage == "projected" and active_head.lineup_state != "projected":
        return None
    return active_head.capture_id


def _require_chronological_new_capture(
    command: CaptureCommand,
    existing_capture: CaptureRecord | None,
    active_head: CaptureRecord | None,
) -> None:
    if existing_capture is not None or active_head is None:
        return
    if command.captured_at < active_head.captured_at:
        raise OperatorPreconditionError(
            "new capture captured_at cannot be earlier than the latest active "
            "capture"
        )


def _record_replay_state(
    store: LedgerStore,
    *,
    existing_capture: CaptureRecord | None,
    forecast_run_id: str,
) -> tuple[bool, bool]:
    forecasts = _ledger_operation(
        "unable to read shadow forecasts",
        lambda: tuple(store.iter_records("forecast")),
    )
    forecast_replayed = any(
        isinstance(record, ForecastRecord)
        and record.forecast_run_id == forecast_run_id
        for record in forecasts
    )
    if existing_capture is None:
        return forecast_replayed, False
    links = _ledger_operation(
        "unable to read shadow capture-forecast links",
        lambda: tuple(store.iter_records("capture_forecast_link")),
    )
    link_replayed = any(
        isinstance(record, CaptureForecastLink)
        and record.capture_id == existing_capture.capture_id
        and record.forecast_run_id == forecast_run_id
        for record in links
    )
    return forecast_replayed, link_replayed


def _simulation_config_hash(config: SimulationConfig) -> str:
    mapping = config.to_mapping()
    return hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest()


def _capture_matches_command(
    capture: CaptureRecord,
    command: CaptureCommand,
    *,
    capture_key: str,
    expected_stage: str,
) -> bool:
    return (
        capture.envelope.idempotency_key == capture_key
        and capture.game_pk == command.game_pk
        and capture.captured_at == command.captured_at
        and capture.first_pitch == command.first_pitch
        and capture.lineup_state == command.lineup_state
        and capture.lineup_state == expected_stage
        and capture.out_of_sample_eligible
        == command.out_of_sample_eligible
        and capture.normalized_input_hash
        == command.normalized_input_hash
        and capture.payload == command.payload
    )


def _forecast_outputs_match(
    forecast: ForecastRecord,
    link: CaptureForecastLink,
    bundle: _ReplayBundle,
    config: SimulationConfig,
    forecast_run_id: str,
) -> bool:
    game = bundle.payload.game_input
    resolved_game = resolve_forecast_input(game)
    expected_market_inputs = build_semantic_market_inputs(
        resolved_game.markets
    )
    try:
        context = deserialize_forecast_context(
            forecast.outputs,
            away_team=game.identity.away_team,
            home_team=game.identity.home_team,
        )
        validate_frozen_market_output_binding(
            context,
            away_team=game.identity.away_team,
            home_team=game.identity.home_team,
        )
        comparison = deserialize_shadow_comparison(
            linked_shadow_comparison(link, forecast)
        )
        validate_shadow_forecast_context(
            comparison,
            context.market_inputs,
            context.markets,
            context.component_distributions,
            away_team=game.identity.away_team,
            home_team=game.identity.home_team,
        )
    except (TypeError, ValueError):
        return False
    return (
        context.forecast_run_id == forecast_run_id
        and context.forecast_run_id == forecast.forecast_run_id
        and context.config_version == forecast.model_version
        and context.simulation_config == config
        and context.simulation_count == config.simulations
        and forecast.simulation_config_hash
        == _simulation_config_hash(context.simulation_config)
        and context.market_inputs == expected_market_inputs
        and comparison.forecast_run_id == forecast_run_id
        and canonical_json(
            canonical_legacy_decision_payload(comparison.legacy)
        )
        == canonical_json(
            canonical_legacy_decision_payload(
                bundle.payload.legacy_decision
            )
        )
    )


def _exact_pregame_replay(
    lookup: PregameLookup,
    bundle: _ReplayBundle,
    *,
    expected_stage: str,
    command: CaptureCommand,
    capture_key: str,
    forecast_run_id: str,
    config: SimulationConfig,
) -> tuple[CaptureRecord, ForecastRecord, CaptureForecastLink] | None:
    capture = lookup.capture
    forecast = lookup.forecast
    link = lookup.link
    if capture is None or forecast is None or link is None:
        return None
    if not _capture_matches_command(
        capture,
        command,
        capture_key=capture_key,
        expected_stage=expected_stage,
    ):
        return None
    if (
        forecast.forecast_run_id != forecast_run_id
        or forecast.model_version != config.version
        or forecast.simulation_config_hash != _simulation_config_hash(config)
        or forecast.normalized_input_hash != command.normalized_input_hash
        or link.capture_id != capture.capture_id
        or link.forecast_run_id != forecast.forecast_run_id
    ):
        return None
    if not _forecast_outputs_match(
        forecast,
        link,
        bundle,
        config,
        forecast_run_id,
    ):
        return None
    return capture, forecast, link


def _existing_exact_pregame_replay(
    ledger_path: str | Path,
    bundle: _ReplayBundle,
    *,
    expected_stage: str,
    command: CaptureCommand,
    capture_key: str,
    forecast_run_id: str,
    config: SimulationConfig,
) -> tuple[CaptureRecord, ForecastRecord, CaptureForecastLink] | None:
    lookup_existing = getattr(
        LedgerStore,
        "lookup_existing_shadow_pregame",
        None,
    )
    if not callable(lookup_existing):
        return None
    lookup = _ledger_operation(
        "unable to look up existing shadow pregame replay",
        lambda: lookup_existing(
            ledger_path,
            capture_key,
            forecast_run_id,
            bundle.game_pk,
        ),
    )
    if lookup is None:
        return None
    return _exact_pregame_replay(
        lookup,
        bundle,
        expected_stage=expected_stage,
        command=command,
        capture_key=capture_key,
        forecast_run_id=forecast_run_id,
        config=config,
    )


def _replayed_pregame_result(
    bundle: _Bundle,
    *,
    expected_stage: str,
    exact_replay: tuple[CaptureRecord, ForecastRecord, CaptureForecastLink],
    bundle_root: str | Path,
) -> PregameRunResult:
    capture, stored_forecast, link = exact_replay
    managed_bundle = persist_managed_bundle(bundle, root=bundle_root)
    return PregameRunResult(
        stage=expected_stage,
        game_pk=bundle.game_pk,
        lineup_state=capture.lineup_state,
        capture=capture,
        forecast=stored_forecast,
        link=link,
        managed_bundle=managed_bundle,
        capture_replayed=True,
        forecast_replayed=True,
        link_replayed=True,
    )


def _pregame_partial_progress(
    store: LedgerStore,
    bundle: _Bundle,
    *,
    error: OperatorError,
    capture_key: str,
    forecast_run_id: str,
    capture_replayed: bool | None,
    forecast_replayed: bool | None,
    link_replayed: bool | None,
) -> OperatorPartialProgress:
    recovered_records = _partial_progress_records(store)
    reread_succeeded = recovered_records is not None
    records = () if recovered_records is None else recovered_records
    capture = _single_record(
        records,
        CaptureRecord,
        lambda record: record.envelope.idempotency_key == capture_key,
    )
    committed_forecast = _single_record(
        records,
        ForecastRecord,
        lambda record: record.forecast_run_id == forecast_run_id,
    )
    link = None
    if capture is not None and committed_forecast is not None:
        link = _single_record(
            records,
            CaptureForecastLink,
            lambda record: (
                record.capture_id == capture.capture_id
                and record.forecast_run_id
                == committed_forecast.forecast_run_id
            ),
        )
    return OperatorPartialProgress(
        stage=bundle.stage,
        game_pk=bundle.game_pk,
        capture_id=None if capture is None else capture.capture_id,
        forecast_run_id=(
            None
            if committed_forecast is None
            else committed_forecast.forecast_run_id
        ),
        capture_forecast_link_record_id=(
            None if link is None else link.envelope.record_id
        ),
        capture_replayed=(
            capture_replayed if capture is not None else None
        ),
        forecast_replayed=(
            forecast_replayed if committed_forecast is not None else None
        ),
        link_replayed=link_replayed if link is not None else None,
        rerun_safe=(
            isinstance(error, OperatorIOError)
            and reread_succeeded
            and capture is not None
        ),
    )


def _run_pregame(
    bundle: _Bundle,
    *,
    expected_stage: str,
    ledger_path: str | Path,
    bundle_root: str | Path,
    registry_root: str | Path | None,
) -> PregameRunResult:
    _require_stage(bundle, expected_stage)
    _validate_game(bundle, expected_stage)
    config = load_forecast_config(
        bundle.payload.forecast_config_version,
        registry_root=registry_root,
    )
    command = _capture_command(bundle)
    capture_key = intrinsic_capture_key(
        bundle.schema_version,
        bundle.stage,
        command,
    )
    forecast_run_id = _pregame_operation(
        lambda: build_forecast_run_id_for_input(
            bundle.payload.game_input,
            config,
        )
    )

    exact_replay = _existing_exact_pregame_replay(
        ledger_path,
        bundle,
        expected_stage=expected_stage,
        command=command,
        capture_key=capture_key,
        forecast_run_id=forecast_run_id,
        config=config,
    )
    if exact_replay is not None:
        return _replayed_pregame_result(
            bundle,
            expected_stage=expected_stage,
            exact_replay=exact_replay,
            bundle_root=bundle_root,
        )

    received_legacy_model_version = (
        bundle.payload.legacy_decision.model_version
    )
    if received_legacy_model_version != LEGACY_MODEL_VERSION:
        raise OperatorPreconditionError(
            "fresh pregame legacy model_version must be "
            f"{LEGACY_MODEL_VERSION}; received "
            f"{received_legacy_model_version}"
        )

    try:
        validate_recommended_review_contract(
            bundle.payload.legacy_decision,
            bundle.payload.game_input.markets,
        )
    except ValueError as error:
        exact_replay = _existing_exact_pregame_replay(
            ledger_path,
            bundle,
            expected_stage=expected_stage,
            command=command,
            capture_key=capture_key,
            forecast_run_id=forecast_run_id,
            config=config,
        )
        if exact_replay is not None:
            return _replayed_pregame_result(
                bundle,
                expected_stage=expected_stage,
                exact_replay=exact_replay,
                bundle_root=bundle_root,
            )
        raise OperatorPreconditionError(str(error)) from error

    forecast = _pregame_operation(
        lambda: run_forecast(bundle.payload.game_input, config)
    )
    comparison = compare_shadow_forecast(
        bundle.payload.legacy_decision,
        forecast,
    )
    try:
        require_recommended_review_eligibility(comparison)
    except ValueError as error:
        exact_replay = _existing_exact_pregame_replay(
            ledger_path,
            bundle,
            expected_stage=expected_stage,
            command=command,
            capture_key=capture_key,
            forecast_run_id=forecast_run_id,
            config=config,
        )
        if exact_replay is not None:
            return _replayed_pregame_result(
                bundle,
                expected_stage=expected_stage,
                exact_replay=exact_replay,
                bundle_root=bundle_root,
            )
        raise OperatorPreconditionError(str(error)) from error

    store = _open_ledger(ledger_path)
    capture_replayed: bool | None = None
    forecast_replayed: bool | None = None
    link_replayed: bool | None = None
    managed_bundle: ManagedBundleCopy | None = None
    try:
        lookup = _ledger_operation(
            "unable to look up shadow pregame replay",
            lambda: store.lookup_shadow_pregame(
                capture_key,
                forecast_run_id,
                bundle.game_pk,
            ),
        )
        exact_replay = _exact_pregame_replay(
            lookup,
            bundle,
            expected_stage=expected_stage,
            command=command,
            capture_key=capture_key,
            forecast_run_id=forecast_run_id,
            config=config,
        )
        if exact_replay is not None:
            capture_replayed = True
            forecast_replayed = True
            link_replayed = True
            return _replayed_pregame_result(
                bundle,
                expected_stage=expected_stage,
                exact_replay=exact_replay,
                bundle_root=bundle_root,
            )
        for attempt in range(_HEAD_CAS_MAX_ATTEMPTS):
            if attempt:
                lookup = _ledger_operation(
                    "unable to refresh shadow pregame state",
                    lambda: store.lookup_shadow_pregame(
                        capture_key,
                        forecast_run_id,
                        bundle.game_pk,
                    ),
                )
            existing_capture = lookup.capture
            active_head = lookup.active_capture
            _require_chronological_new_capture(
                command,
                existing_capture,
                active_head,
            )
            supersedes_capture_id = _resolve_supersession_target(
                stage=expected_stage,
                existing_capture=existing_capture,
                active_head=active_head,
                active_confirmed_capture=(
                    lookup.active_confirmed_capture
                ),
            )
            forecast_replayed = lookup.forecast is not None
            link_replayed = lookup.link is not None
            capture_replayed = existing_capture is not None
            try:
                records = _pregame_operation(
                    lambda: _ledger_operation(
                        "unable to record shadow pregame run",
                        lambda: record_shadow_run(
                            store,
                            bundle.payload.game_input,
                            bundle.payload.legacy_decision,
                            forecast,
                            ShadowRunIdempotencyKeys(capture=capture_key),
                            comparison=comparison,
                            supersedes_capture_id=supersedes_capture_id,
                            expected_active_capture_id=(
                                None
                                if active_head is None
                                else active_head.capture_id
                            ),
                        ),
                    )
                )
            except LedgerHeadChangedError as error:
                if attempt + 1 == _HEAD_CAS_MAX_ATTEMPTS:
                    raise OperatorLedgerError(
                        "unable to record shadow pregame run: active capture "
                        f"head changed after {_HEAD_CAS_MAX_ATTEMPTS} attempts: "
                        f"{error}"
                    ) from error
                continue
            break
        else:
            raise RuntimeError(
                "pregame head CAS retry loop exited unexpectedly"
            )
        if records.capture_replayed is not None:
            capture_replayed = records.capture_replayed
        if records.forecast_replayed is not None:
            forecast_replayed = records.forecast_replayed
        if records.link_replayed is not None:
            link_replayed = records.link_replayed
        managed_bundle = persist_managed_bundle(bundle, root=bundle_root)
        return PregameRunResult(
            stage=expected_stage,
            game_pk=bundle.game_pk,
            lineup_state=records.capture.lineup_state,
            capture=records.capture,
            forecast=records.forecast,
            link=records.link,
            managed_bundle=managed_bundle,
            capture_replayed=bool(capture_replayed),
            forecast_replayed=bool(forecast_replayed),
            link_replayed=bool(link_replayed),
        )
    except OperatorError as error:
        try:
            progress = _pregame_partial_progress(
                store,
                bundle,
                error=error,
                capture_key=capture_key,
                forecast_run_id=forecast_run_id,
                capture_replayed=capture_replayed,
                forecast_replayed=forecast_replayed,
                link_replayed=link_replayed,
            )
        except Exception:
            progress = OperatorPartialProgress(
                stage=bundle.stage,
                game_pk=bundle.game_pk,
            )
        raise error.with_partial_progress(progress) from error


def run_projected(
    bundle: ProjectedBundle,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    registry_root: str | Path | None = None,
) -> PregameRunResult:
    """Append or replay one projected-lineup shadow forecast."""

    if (
        type(bundle) is not ProjectedBundle
        or bundle.schema_version != CURRENT_PREGAME_SCHEMA_VERSION
        or bundle.authority != CURRENT_PREGAME_AUTHORITY
        or not is_current_pregame_payload(bundle.payload)
    ):
        raise TypeError(
            "normal pregame runner requires a strict current bundle: "
            "ProjectedBundle"
        )
    return _run_pregame(
        bundle,
        expected_stage="projected",
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )


def run_confirmed(
    bundle: ConfirmedBundle,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    registry_root: str | Path | None = None,
) -> PregameRunResult:
    """Append or replay one confirmed-lineup shadow forecast."""

    if (
        type(bundle) is not ConfirmedBundle
        or bundle.schema_version != CURRENT_PREGAME_SCHEMA_VERSION
        or bundle.authority != CURRENT_PREGAME_AUTHORITY
        or not is_current_pregame_payload(bundle.payload)
    ):
        raise TypeError(
            "normal pregame runner requires a strict current bundle: "
            "ConfirmedBundle"
        )
    return _run_pregame(
        bundle,
        expected_stage="confirmed",
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )


def _replay_historical_pregame(
    path: str | Path,
    *,
    expected_stage: str,
    expected_types: tuple[type[BaseModel], ...],
    ledger_path: str | Path,
    bundle_root: str | Path,
    registry_root: str | Path | None,
) -> PregameRunResult:
    artifact = load_historical_managed_bundle(path, root=bundle_root)
    if not isinstance(artifact, ReplayOnlyPregameBundle):
        raise OperatorPreconditionError(
            "historical replay requires a schema-v1 managed bundle"
        )
    bundle = artifact.bundle
    if type(bundle) not in expected_types or bundle.stage != expected_stage:
        raise OperatorPreconditionError("historical replay stage/type mismatch")

    _validate_game(bundle, expected_stage)
    config = load_forecast_config(
        bundle.payload.forecast_config_version,
        registry_root=registry_root,
    )
    command = _capture_command(bundle)
    capture_key = intrinsic_capture_key(
        bundle.schema_version,
        bundle.stage,
        command,
    )
    forecast_run_id = _pregame_operation(
        lambda: build_forecast_run_id_for_input(
            bundle.payload.game_input,
            config,
        )
    )
    exact_replay = _existing_exact_pregame_replay(
        ledger_path,
        bundle,
        expected_stage=expected_stage,
        command=command,
        capture_key=capture_key,
        forecast_run_id=forecast_run_id,
        config=config,
    )
    if exact_replay is None:
        raise OperatorPreconditionError(
            "historical replay requires an exact existing replay"
        )
    capture, forecast, link = exact_replay
    return PregameRunResult(
        stage=expected_stage,
        game_pk=bundle.game_pk,
        lineup_state=capture.lineup_state,
        capture=capture,
        forecast=forecast,
        link=link,
        managed_bundle=artifact.managed_bundle,
        capture_replayed=True,
        forecast_replayed=True,
        link_replayed=True,
    )


def replay_historical_projected(
    path: str | Path,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    registry_root: str | Path | None = None,
) -> PregameRunResult:
    """Replay an authenticated pre-change projected managed bundle."""

    return _replay_historical_pregame(
        path,
        expected_stage="projected",
        expected_types=(
            HistoricalRelaxedProjectedBundleV1,
            HistoricalStrictProjectedBundleV1,
        ),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )


def replay_historical_confirmed(
    path: str | Path,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    registry_root: str | Path | None = None,
) -> PregameRunResult:
    """Replay an authenticated pre-change confirmed managed bundle."""

    return _replay_historical_pregame(
        path,
        expected_stage="confirmed",
        expected_types=(
            HistoricalRelaxedConfirmedBundleV1,
            HistoricalStrictConfirmedBundleV1,
        ),
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )


def _review_ledger_records(store: LedgerStore) -> tuple[object, ...]:
    return _ledger_operation(
        "unable to read shadow review records",
        lambda: tuple(store.iter_records()),
    )


def _exact_review_record(
    records: tuple[object, ...],
    record_type: type[_T],
    identifier: str,
    *,
    id_field: str,
    field_name: str | None = None,
) -> _T:
    label = id_field if field_name is None else field_name
    matches = tuple(
        record
        for record in records
        if isinstance(record, record_type)
        and _record_identifier(record, id_field) == identifier
    )
    if not matches:
        raise OperatorPreconditionError(
            f"{label} must reference an existing frozen {record_type.__name__}"
        )
    if len(matches) != 1:
        raise OperatorLedgerError(
            f"{label} resolves multiple frozen {record_type.__name__} records"
        )
    return matches[0]


def _resolve_review_records(
    bundle: ReviewBundle,
    records: tuple[object, ...],
) -> tuple[CaptureRecord, ForecastRecord, CaptureForecastLink]:
    capture = _exact_review_record(
        records,
        CaptureRecord,
        bundle.payload.capture_id,
        id_field="capture_id",
    )
    forecast = _exact_review_record(
        records,
        ForecastRecord,
        bundle.payload.forecast_run_id,
        id_field="forecast_run_id",
    )
    link = _exact_review_record(
        records,
        CaptureForecastLink,
        bundle.payload.capture_forecast_link_record_id,
        id_field="envelope.record_id",
        field_name="capture_forecast_link_record_id",
    )
    return capture, forecast, link


def _record_identifier(record: object, id_field: str) -> object:
    value = record
    for part in id_field.split("."):
        value = getattr(value, part)
    return value


def _require_frozen_review_relationships(
    bundle: ReviewBundle,
    records: tuple[object, ...],
    capture: CaptureRecord,
    forecast: ForecastRecord,
    link: CaptureForecastLink,
) -> None:
    if capture.game_pk != bundle.game_pk:
        raise OperatorPreconditionError(
            "frozen capture game_pk must match review bundle game_pk"
        )
    if capture.lineup_state != "confirmed":
        raise OperatorPreconditionError(
            "review requires a frozen confirmed capture"
        )
    if capture.captured_at >= capture.first_pitch:
        raise OperatorPreconditionError(
            "frozen capture captured_at must be before first_pitch"
        )
    if link.capture_id != capture.capture_id:
        raise OperatorPreconditionError(
            "frozen link capture_id must match the exact review capture"
        )
    if link.forecast_run_id != forecast.forecast_run_id:
        raise OperatorPreconditionError(
            "frozen link forecast_run_id must match the exact review forecast"
        )
    if any(
        isinstance(record, CaptureForecastLink)
        and record.supersedes_link_record_id == link.envelope.record_id
        for record in records
    ):
        raise OperatorPreconditionError(
            "review requires the latest corrected capture-forecast link"
        )
    if capture.normalized_input_hash != forecast.normalized_input_hash:
        raise OperatorPreconditionError(
            "frozen capture and forecast normalized_input_hash must match"
        )
    latest_active = _latest_active_capture(
        tuple(
            record for record in records if isinstance(record, CaptureRecord)
        ),
        capture.game_pk,
    )
    if latest_active is None or latest_active.capture_id != capture.capture_id:
        raise OperatorPreconditionError(
            "review requires the exact frozen capture to remain the latest active "
            "pregame capture"
        )


def _eligible_review_market_keys(
    forecast: ForecastRecord,
    link: CaptureForecastLink,
) -> tuple[str, ...]:
    try:
        comparison = deserialize_shadow_comparison(
            linked_shadow_comparison(link, forecast)
        )
    except (TypeError, ValueError) as error:
        raise OperatorLedgerError(
            f"invalid frozen shadow comparison: {error}"
        ) from error
    if comparison.forecast_run_id != forecast.forecast_run_id:
        raise OperatorPreconditionError(
            "frozen shadow comparison forecast_run_id must match review forecast"
        )
    return tuple(
        sorted(
            {
                market.market_key
                for market in comparison.market_comparisons
                if market.review_eligible
            }
        )
    )


def _require_result_preconditions(
    command: OfficialResultCommand,
    capture: CaptureRecord,
) -> None:
    if command.source_name != "MLB":
        raise OperatorPreconditionError(
            "official result source_name must be exactly MLB"
        )
    if command.game_pk != capture.game_pk:
        raise OperatorPreconditionError(
            "official result game_pk must match the frozen capture"
        )
    if command.finalized_at <= capture.first_pitch:
        raise OperatorPreconditionError(
            "official result finalized_at must be strictly after first_pitch"
        )


def _require_close_preconditions(
    command: MarketObservationCommand,
    capture: CaptureRecord,
) -> None:
    if command.game_pk != capture.game_pk:
        raise OperatorPreconditionError(
            "closing observation game_pk must match the frozen capture"
        )
    if command.observation_kind != "close":
        raise OperatorPreconditionError(
            "closing observation observation_kind must be exactly close"
        )
    for field_name in ("selected_decimal_price", "opposing_decimal_price"):
        value = getattr(command, field_name)
        if value is None or value <= 1.0:
            raise OperatorPreconditionError(
                f"closing observation {field_name} must be above 1"
            )
    if command.available_at is None:
        raise OperatorPreconditionError(
            "closing observation available_at must be present"
        )
    for field_name in ("observed_at", "available_at"):
        timestamp = getattr(command, field_name)
        if not capture.captured_at <= timestamp <= capture.first_pitch:
            raise OperatorPreconditionError(
                f"closing observation {field_name} must be within the frozen "
                "capture interval"
            )


def _require_unique_closing_streams(
    commands: tuple[MarketObservationCommand, ...],
) -> None:
    streams = tuple(
        (
            command.game_pk,
            command.source_name,
            command.source_record,
            command.observation_kind,
            command.market_key,
        )
        for command in commands
    )
    if len(set(streams)) != len(streams):
        raise OperatorPreconditionError(
            "review bundle contains a duplicate logical closing stream"
        )


def _record_with_key(
    records: tuple[object, ...],
    record_type: type[_T],
    idempotency_key: str,
) -> _T | None:
    matches = tuple(
        record
        for record in records
        if isinstance(record, record_type)
        and record.envelope.idempotency_key == idempotency_key
    )
    if len(matches) > 1:
        raise OperatorLedgerError(
            f"duplicate {record_type.__name__} idempotency key"
        )
    return None if not matches else matches[0]


def _record_matches_command(
    record: MarketObservationRecord | OfficialResultRecord,
    command: MarketObservationCommand | OfficialResultCommand,
) -> bool:
    return all(
        getattr(record, field.name) == getattr(command, field.name)
        for field in fields(command)
    )


def _market_observation_key(command: MarketObservationCommand) -> str:
    return (
        f"market:{command.source_name}:{command.source_record}:"
        f"{command.source_revision}:{command.observation_kind}:"
        f"{command.market_key}"
    )


def _official_result_key(command: OfficialResultCommand) -> str:
    return (
        f"result:{command.source_name}:{command.game_pk}:"
        f"{command.source_revision}"
    )


def _same_closing_stream(
    record: MarketObservationRecord,
    command: MarketObservationCommand,
) -> bool:
    return (
        record.game_pk == command.game_pk
        and record.source_name == command.source_name
        and record.source_record == command.source_record
        and record.observation_kind == command.observation_kind
        and record.market_key == command.market_key
    )


def _latest_closing_observation(
    records: tuple[object, ...],
    command: MarketObservationCommand,
) -> MarketObservationRecord | None:
    stream = tuple(
        record
        for record in records
        if isinstance(record, MarketObservationRecord)
        and _same_closing_stream(record, command)
    )
    superseded_ids = frozenset(
        record.supersedes_record_id
        for record in stream
        if record.supersedes_record_id is not None
    )
    latest = tuple(
        record
        for record in stream
        if record.envelope.record_id not in superseded_ids
    )
    if len(latest) > 1:
        raise OperatorLedgerError(
            "closing observation revisions do not have one current latest record"
        )
    return None if not latest else latest[0]


def _latest_result_record(
    records: tuple[object, ...],
    game_pk: int,
) -> OfficialResultRecord | None:
    game_results = tuple(
        record
        for record in records
        if isinstance(record, OfficialResultRecord)
        and record.game_pk == game_pk
    )
    superseded_ids = frozenset(
        record.supersedes_record_id
        for record in game_results
        if record.supersedes_record_id is not None
    )
    latest = tuple(
        record
        for record in game_results
        if record.envelope.record_id not in superseded_ids
    )
    if len(latest) > 1:
        raise OperatorLedgerError(
            "official result revisions do not have one current latest record"
        )
    return None if not latest else latest[0]


def _preflight_closing_lineage(
    records: tuple[object, ...],
    command: MarketObservationCommand,
) -> _ClosingPreflight:
    existing = _record_with_key(
        records,
        MarketObservationRecord,
        _market_observation_key(command),
    )
    if existing is not None:
        original_target = existing.supersedes_record_id
        if command.supersedes_record_id == existing.envelope.record_id:
            raise OperatorPreconditionError(
                "closing observation cannot supersede itself"
            )
        if original_target == existing.envelope.record_id:
            raise OperatorLedgerError(
                "stored closing observation supersedes itself"
            )
        latest = _latest_closing_observation(records, command)
        if (
            latest is None
            or latest.envelope.record_id != existing.envelope.record_id
        ):
            raise OperatorPreconditionError(
                "exact closing observation is no longer the current latest "
                "record"
            )
        return _ClosingPreflight(existing, original_target)

    latest = _latest_closing_observation(records, command)
    if latest is None:
        if command.supersedes_record_id is not None:
            raise OperatorPreconditionError(
                "initial closing observation supersession target must be null"
            )
        return _ClosingPreflight(None, None)
    if command.supersedes_record_id is None:
        raise OperatorPreconditionError(
            "closing observation correction must explicitly supersede the "
            "current latest record"
        )
    if command.supersedes_record_id != latest.envelope.record_id:
        raise OperatorPreconditionError(
            "closing observation correction supersession target must be the "
            "current latest record"
        )
    return _ClosingPreflight(None, latest.envelope.record_id)


def _preflight_result_lineage(
    records: tuple[object, ...],
    command: OfficialResultCommand,
) -> OfficialResultRecord | None:
    existing = _record_with_key(
        records,
        OfficialResultRecord,
        _official_result_key(command),
    )
    if existing is not None:
        original_target = existing.supersedes_record_id
        if command.supersedes_record_id == existing.envelope.record_id:
            raise OperatorPreconditionError(
                "official result cannot supersede itself"
            )
        if original_target == existing.envelope.record_id:
            raise OperatorLedgerError("stored official result supersedes itself")
        latest = _latest_result_record(records, command.game_pk)
        if (
            latest is None
            or latest.envelope.record_id != existing.envelope.record_id
        ):
            raise OperatorPreconditionError(
                "exact official result is no longer the current latest record"
            )
        return existing

    latest = _latest_result_record(records, command.game_pk)
    if latest is None:
        if command.supersedes_record_id is not None:
            raise OperatorPreconditionError(
                "initial official result supersession target must be null"
            )
        return None
    if command.supersedes_record_id is None:
        raise OperatorPreconditionError(
            "official result correction must explicitly supersede the current "
            "latest result"
        )
    if command.supersedes_record_id != latest.envelope.record_id:
        raise OperatorPreconditionError(
            "official result correction supersession target must be the current "
            "latest result"
        )
    return None


def _append_closing_observation(
    store: LedgerStore,
    command: MarketObservationCommand,
    preflight: _ClosingPreflight,
) -> tuple[MarketObservationRecord, bool]:
    current = preflight
    for attempt in range(_HEAD_CAS_MAX_ATTEMPTS):
        try:
            record = store.append_market_observation(
                command,
                expected_latest_record_id=current.expected_latest_record_id,
            )
            return record, current.existing is not None
        except LedgerMarketHeadChangedError as error:
            current = _preflight_closing_lineage(
                _review_ledger_records(store),
                command,
            )
            if attempt + 1 == _HEAD_CAS_MAX_ATTEMPTS:
                raise OperatorLedgerError(
                    "unable to record closing observation after repeated "
                    f"stream head changes: {error}"
                ) from error
        except _LEDGER_ERRORS as error:
            raise OperatorLedgerError(
                f"unable to record closing observation: {error}"
            ) from error
        except OSError as error:
            raise OperatorIOError(
                f"unable to record closing observation: {error}"
            ) from error
    raise AssertionError("closing observation CAS retry loop exhausted")


def _append_official_result(
    store: LedgerStore,
    command: OfficialResultCommand,
    required_active_capture_id: str,
) -> OfficialResultRecord:
    try:
        return store.append_official_result(
            command,
            required_active_capture_id=required_active_capture_id,
        )
    except LedgerActiveCaptureHeadChangedError as error:
        raise OperatorPreconditionError(
            "unable to record official result because the frozen active capture "
            f"changed: {error}"
        ) from error
    except _LEDGER_ERRORS as error:
        raise OperatorLedgerError(
            f"unable to record official result: {error}"
        ) from error
    except OSError as error:
        raise OperatorIOError(
            f"unable to record official result: {error}"
        ) from error


def _review_for_exact_lineage(
    records: tuple[object, ...],
    link: CaptureForecastLink,
    result: OfficialResultRecord,
    market_key: str,
) -> ReviewRecord | None:
    matches = tuple(
        record
        for record in records
        if isinstance(record, ReviewRecord)
        and record.capture_forecast_link_record_id
        == link.envelope.record_id
        and record.official_result_record_id == result.envelope.record_id
        and record.market_key == market_key
    )
    if len(matches) > 1:
        raise OperatorLedgerError(
            "review lineage has multiple immutable records"
        )
    return None if not matches else matches[0]


def _review_partial_progress(
    store: LedgerStore,
    bundle: ReviewBundle,
    *,
    error: OperatorError,
    capture: CaptureRecord,
    forecast: ForecastRecord,
    link: CaptureForecastLink,
    market_keys: tuple[str, ...],
    baseline_record_ids: frozenset[str],
) -> OperatorPartialProgress:
    recovered_records = _partial_progress_records(store)
    reread_succeeded = recovered_records is not None
    records = () if recovered_records is None else recovered_records
    closing_records: list[MarketObservationRecord] = []
    for command in bundle.payload.closing_observations:
        record = _single_record(
            records,
            MarketObservationRecord,
            lambda candidate, key=_market_observation_key(command): (
                candidate.envelope.idempotency_key == key
            ),
        )
        if record is not None:
            closing_records.append(record)

    result = _single_record(
        records,
        OfficialResultRecord,
        lambda record: (
            record.envelope.idempotency_key
            == _official_result_key(bundle.payload.official_result)
        ),
    )
    review_records: list[ReviewRecord] = []
    if result is not None:
        for market_key in market_keys:
            record = _review_for_exact_lineage(
                records,
                link,
                result,
                market_key,
            )
            if record is not None:
                review_records.append(record)

    latest_capture = (
        None
        if not reread_succeeded
        else _latest_active_capture(
            tuple(
                record
                for record in records
                if isinstance(record, CaptureRecord)
            ),
            bundle.game_pk,
        )
    )
    retry_lineage_safe = _review_retry_lineage_safe(
        records,
        bundle,
        capture,
    ) if reread_succeeded else False
    return OperatorPartialProgress(
        stage=bundle.stage,
        game_pk=bundle.game_pk,
        capture_id=capture.capture_id,
        forecast_run_id=forecast.forecast_run_id,
        capture_forecast_link_record_id=link.envelope.record_id,
        closing_observation_record_ids=tuple(
            record.envelope.record_id for record in closing_records
        ),
        official_result_record_id=(
            None if result is None else result.envelope.record_id
        ),
        review_record_ids=tuple(
            record.envelope.record_id for record in review_records
        ),
        closing_replay_count=(
            None
            if not reread_succeeded
            else sum(
                record.envelope.record_id in baseline_record_ids
                for record in closing_records
            )
        ),
        official_result_replayed=(
            None
            if not reread_succeeded or result is None
            else result.envelope.record_id in baseline_record_ids
        ),
        review_replay_count=(
            None
            if not reread_succeeded
            else sum(
                record.envelope.record_id in baseline_record_ids
                for record in review_records
            )
        ),
        rerun_safe=(
            isinstance(error, OperatorIOError)
            and latest_capture is not None
            and latest_capture.capture_id == capture.capture_id
            and retry_lineage_safe
        ),
    )


def _review_retry_lineage_safe(
    records: tuple[object, ...],
    bundle: ReviewBundle,
    capture: CaptureRecord,
) -> bool:
    """Read-only proof that an exact retry still targets current ledger heads."""

    try:
        latest_capture = _latest_active_capture(
            tuple(
                record
                for record in records
                if isinstance(record, CaptureRecord)
            ),
            bundle.game_pk,
        )
        if (
            latest_capture is None
            or latest_capture.capture_id != capture.capture_id
        ):
            return False
        for command in bundle.payload.closing_observations:
            preflight = _preflight_closing_lineage(records, command)
            if preflight.existing is not None and not _record_matches_command(
                preflight.existing,
                command,
            ):
                return False
        existing_result = _preflight_result_lineage(
            records,
            bundle.payload.official_result,
        )
        if existing_result is not None and not _record_matches_command(
            existing_result,
            bundle.payload.official_result,
        ):
            return False
        return True
    except Exception:
        return False


def _build_exact_review(
    store: LedgerStore,
    link: CaptureForecastLink,
    result: OfficialResultRecord,
    market_key: str,
    required_active_capture_id: str,
):
    try:
        return build_and_record_review(
            store,
            link.envelope.record_id,
            result.envelope.record_id,
            market_key,
            required_active_capture_id=required_active_capture_id,
        )
    except LedgerActiveCaptureHeadChangedError as error:
        raise OperatorPreconditionError(
            "unable to record exact review because the frozen active capture "
            f"changed: {error}"
        ) from error
    except _LEDGER_ERRORS as error:
        raise OperatorLedgerError(
            f"unable to record exact review for {market_key}: {error}"
        ) from error
    except OSError as error:
        raise OperatorIOError(
            f"unable to record exact review for {market_key}: {error}"
        ) from error
    except (TypeError, ValueError) as error:
        raise OperatorPreconditionError(
            f"unable to build exact review for {market_key}: {error}"
        ) from error


def _classification_count_mapping(
    reviews: tuple[ReviewRecord, ...],
) -> Mapping[str, int]:
    counts = {key: 0 for key in ("A", "B", "C", "D", "unclassified")}
    for review in reviews:
        key = "unclassified" if review.classification is None else review.classification
        counts[key] += 1
    return MappingProxyType(counts)


def _calibration_report_state(
    store: LedgerStore,
    ledger_path: str | Path,
) -> tuple[dict[str, object], int, int, Mapping[str, int]]:
    try:
        ledger_records = tuple(store.iter_records())
        rows = select_primary_rows(_LedgerSnapshot(ledger_records))
        report = build_calibration_report(rows)
        mapping = build_report_mapping(
            ledger_path=ledger_path,
            ledger_records=ledger_records,
            rows=rows,
            report=report,
        )
    except _LEDGER_ERRORS as error:
        raise OperatorLedgerError(
            f"unable to build calibration report: {error}"
        ) from error
    except OSError as error:
        raise OperatorIOError(
            f"unable to build calibration report: {error}"
        ) from error
    except (TypeError, ValueError) as error:
        raise OperatorLedgerError(
            f"unable to build calibration report: {error}"
        ) from error

    overall = mapping["overall"]
    migration = mapping["migration"]
    if not isinstance(overall, Mapping) or not isinstance(migration, Mapping):
        raise AssertionError("calibration report mapping has invalid sections")
    confirmed_games = migration["confirmed_games"]
    market_types = migration["market_types"]
    if not isinstance(confirmed_games, Mapping) or not isinstance(
        market_types, Mapping
    ):
        raise AssertionError("calibration migration mapping has invalid sections")
    decision_counts = {
        market_type: int(progress["count"])
        for market_type, progress in market_types.items()
        if isinstance(progress, Mapping)
    }
    return (
        mapping,
        int(overall["sample_size"]),
        int(confirmed_games["count"]),
        MappingProxyType(decision_counts),
    )


def _publish_calibration_report(
    store: LedgerStore,
    ledger_path: str | Path,
    report_dir: str | Path,
) -> tuple[CalibrationReportArtifacts, int, int, Mapping[str, int]]:
    stale_error: _StaleCalibrationPublicationError | None = None
    for _ in range(_REPORT_PUBLICATION_MAX_ATTEMPTS):
        (
            mapping,
            sample_size,
            confirmed_game_count,
            confirmed_decision_counts,
        ) = _calibration_report_state(store, ledger_path)
        try:
            artifacts = write_calibration_reports(
                mapping,
                report_dir=report_dir,
            )
        except _StaleCalibrationPublicationError as error:
            stale_error = error
            continue
        return (
            artifacts,
            sample_size,
            confirmed_game_count,
            confirmed_decision_counts,
        )
    raise OperatorLedgerError(
        "unable to publish calibration report after rebuilding the current "
        f"ledger {_REPORT_PUBLICATION_MAX_ATTEMPTS} times"
    ) from stale_error


def run_review(
    bundle: ReviewBundle,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> ReviewRunResult:
    """Append or replay one official result and its exact frozen reviews."""

    if not isinstance(bundle, ReviewBundle):
        raise TypeError("bundle must be a validated review bundle")
    if bundle.stage != "review":
        raise OperatorPreconditionError(
            f"bundle stage {bundle.stage!r} cannot run as 'review'"
        )

    store = _open_ledger(ledger_path)
    records = _review_ledger_records(store)
    capture, forecast, link = _resolve_review_records(bundle, records)
    _require_frozen_review_relationships(
        bundle,
        records,
        capture,
        forecast,
        link,
    )
    market_keys = _eligible_review_market_keys(forecast, link)
    result_command = bundle.payload.official_result
    close_commands = bundle.payload.closing_observations

    # Validate all leakage and source constraints before the first append/replay.
    _require_result_preconditions(result_command, capture)
    for command in close_commands:
        _require_close_preconditions(command, capture)
    _require_unique_closing_streams(close_commands)
    closing_preflights = tuple(
        _preflight_closing_lineage(records, command)
        for command in close_commands
    )
    existing_result = _preflight_result_lineage(records, result_command)
    baseline_record_ids = frozenset(
        record.envelope.record_id for record in records
    )
    managed_bundle: ManagedBundleCopy | None = None

    try:
        closing_outcomes = tuple(
            _append_closing_observation(store, command, preflight)
            for command, preflight in zip(
                close_commands,
                closing_preflights,
                strict=True,
            )
        )
        closing_records = tuple(record for record, _ in closing_outcomes)
        result_record = _append_official_result(
            store,
            result_command,
            capture.capture_id,
        )

        review_records: list[ReviewRecord] = []
        review_replay_count = 0
        for market_key in market_keys:
            current_records = _review_ledger_records(store)
            existing_review = _review_for_exact_lineage(
                current_records,
                link,
                result_record,
                market_key,
            )
            if existing_review is not None:
                review_records.append(existing_review)
                review_replay_count += 1
                continue
            recorded = _build_exact_review(
                store,
                link,
                result_record,
                market_key,
                capture.capture_id,
            )
            review_records.append(recorded.record)

        frozen_reviews = tuple(review_records)
        managed_bundle = persist_managed_bundle(bundle, root=bundle_root)
        (
            report_artifacts,
            calibration_sample_size,
            confirmed_game_count,
            confirmed_decision_counts,
        ) = _publish_calibration_report(store, ledger_path, report_dir)
        return ReviewRunResult(
            stage="review",
            game_pk=bundle.game_pk,
            capture=capture,
            forecast=forecast,
            link=link,
            closing_observations=closing_records,
            official_result=result_record,
            reviews=frozen_reviews,
            managed_bundle=managed_bundle,
            closing_replay_count=sum(
                replayed for _, replayed in closing_outcomes
            ),
            official_result_replayed=existing_result is not None,
            review_replay_count=review_replay_count,
            classification_counts=_classification_count_mapping(
                frozen_reviews
            ),
            calibration_json_path=report_artifacts.json_path,
            calibration_markdown_path=report_artifacts.markdown_path,
            calibration_generation_dir=report_artifacts.generation_dir,
            calibration_current_generation_path=(
                report_artifacts.current_generation_path
            ),
            calibration_report_replayed=report_artifacts.replayed,
            calibration_sample_size=calibration_sample_size,
            confirmed_game_count=confirmed_game_count,
            confirmed_decision_count=sum(
                confirmed_decision_counts.values()
            ),
            confirmed_decision_counts=confirmed_decision_counts,
        )
    except OperatorError as error:
        try:
            progress = _review_partial_progress(
                store,
                bundle,
                error=error,
                capture=capture,
                forecast=forecast,
                link=link,
                market_keys=market_keys,
                baseline_record_ids=baseline_record_ids,
            )
        except Exception:
            progress = OperatorPartialProgress(
                stage=bundle.stage,
                game_pk=bundle.game_pk,
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                capture_forecast_link_record_id=link.envelope.record_id,
            )
        raise error.with_partial_progress(progress) from error


__all__ = (
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_REPORT_DIR",
    "PregameRunResult",
    "ReviewRunResult",
    "replay_historical_confirmed",
    "replay_historical_projected",
    "run_confirmed",
    "run_projected",
    "run_review",
)
