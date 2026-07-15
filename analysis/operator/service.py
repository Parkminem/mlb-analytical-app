"""Pregame shadow-run and exact confirmed-review orchestration."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from analysis.forecasting import LineupState, run_forecast
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
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerReferenceError,
    LedgerStore,
    MarketObservationCommand,
    MarketObservationRecord,
    OfficialResultCommand,
    OfficialResultRecord,
    ReviewRecord,
)
from analysis.review import build_and_record_review
from analysis.shadow import (
    ShadowRunIdempotencyKeys,
    build_shadow_capture_command,
    deserialize_shadow_comparison,
    record_shadow_run,
)

from .bundles import (
    DEFAULT_BUNDLE_ROOT,
    ManagedBundleCopy,
    intrinsic_capture_key,
    persist_managed_bundle,
)
from .config import load_forecast_config
from .errors import (
    BundleValidationError,
    OperatorIOError,
    OperatorLedgerError,
    OperatorPreconditionError,
)
from .schemas import ConfirmedBundle, ProjectedBundle, ReviewBundle


DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ledger" / "shadow.jsonl"
)

_Bundle = ProjectedBundle | ConfirmedBundle
_T = TypeVar("_T")
_HEAD_CAS_MAX_ATTEMPTS = 3
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
        raise BundleValidationError(
            f"invalid pregame relationship: {error}"
        ) from error
    except PregameTimingError as error:
        raise OperatorPreconditionError(
            f"invalid pregame timing: {error}"
        ) from error


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


def _validate_game(bundle: _Bundle, expected_stage: str) -> None:
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


def _capture_command(bundle: _Bundle) -> CaptureCommand:
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
    captures: tuple[CaptureRecord, ...],
    existing_capture: CaptureRecord | None,
    game_pk: int,
) -> str | None:
    if existing_capture is not None:
        return existing_capture.supersedes_capture_id

    active = _active_game_captures(captures, game_pk)
    if stage == "projected" and any(
        capture.lineup_state == "confirmed" for capture in active
    ):
        raise OperatorPreconditionError(
            "new projected capture is not allowed after an active confirmed capture"
        )
    latest = _latest_active_capture(captures, game_pk)
    if latest is None:
        return None
    if stage == "projected" and latest.lineup_state != "projected":
        return None
    return latest.capture_id


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

    forecast = _pregame_operation(
        lambda: run_forecast(bundle.payload.game_input, config)
    )

    store = _open_ledger(ledger_path)
    for attempt in range(_HEAD_CAS_MAX_ATTEMPTS):
        captures = _captures(store)
        existing_capture = _existing_keyed_capture(captures, capture_key)
        active_head = _latest_active_capture(captures, bundle.game_pk)
        supersedes_capture_id = _resolve_supersession_target(
            stage=expected_stage,
            captures=captures,
            existing_capture=existing_capture,
            game_pk=bundle.game_pk,
        )
        forecast_replayed, link_replayed = _record_replay_state(
            store,
            existing_capture=existing_capture,
            forecast_run_id=forecast.forecast_run_id,
        )
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
        raise RuntimeError("pregame head CAS retry loop exited unexpectedly")
    managed_bundle = persist_managed_bundle(bundle, root=bundle_root)
    return PregameRunResult(
        stage=expected_stage,
        game_pk=bundle.game_pk,
        lineup_state=records.capture.lineup_state,
        capture=records.capture,
        forecast=records.forecast,
        link=records.link,
        managed_bundle=managed_bundle,
        capture_replayed=existing_capture is not None,
        forecast_replayed=forecast_replayed,
        link_replayed=link_replayed,
    )


def run_projected(
    bundle: ProjectedBundle,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
    registry_root: str | Path | None = None,
) -> PregameRunResult:
    """Append or replay one projected-lineup shadow forecast."""

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

    return _run_pregame(
        bundle,
        expected_stage="confirmed",
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
    if capture.normalized_input_hash != forecast.normalized_input_hash:
        raise OperatorPreconditionError(
            "frozen capture and forecast normalized_input_hash must match"
        )
    superseded_capture_ids = frozenset(
        record.supersedes_capture_id
        for record in records
        if isinstance(record, CaptureRecord)
        and record.game_pk == capture.game_pk
        and record.supersedes_capture_id is not None
    )
    if capture.capture_id in superseded_capture_ids:
        raise OperatorPreconditionError(
            "review requires the exact frozen capture to remain active"
        )


def _eligible_review_market_keys(forecast: ForecastRecord) -> tuple[str, ...]:
    try:
        comparison = deserialize_shadow_comparison(
            forecast.outputs.get("shadow_comparison")
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
            f"duplicate {record_type.__name__} idempotency key: {idempotency_key}"
        )
    return None if not matches else matches[0]


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
) -> MarketObservationRecord | None:
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
        return existing

    latest = _latest_closing_observation(records, command)
    if latest is None:
        if command.supersedes_record_id is not None:
            raise OperatorPreconditionError(
                "initial closing observation supersession target must be null"
            )
        return None
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
    return None


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
) -> MarketObservationRecord:
    return _ledger_operation(
        "unable to record closing observation",
        lambda: store.append_market_observation(command),
    )


def _append_official_result(
    store: LedgerStore,
    command: OfficialResultCommand,
) -> OfficialResultRecord:
    return _ledger_operation(
        "unable to record official result",
        lambda: store.append_official_result(command),
    )


def _review_idempotency_key(
    capture: CaptureRecord,
    forecast: ForecastRecord,
    result: OfficialResultRecord,
    market_key: str,
) -> str:
    return (
        f"review:{capture.capture_id}:{forecast.forecast_run_id}:"
        f"{result.envelope.record_id}:{market_key}"
    )


def _build_exact_review(
    store: LedgerStore,
    link: CaptureForecastLink,
    result: OfficialResultRecord,
    market_key: str,
):
    try:
        return build_and_record_review(
            store,
            link.envelope.record_id,
            result.envelope.record_id,
            market_key,
        )
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


def run_review(
    bundle: ReviewBundle,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    bundle_root: str | Path = DEFAULT_BUNDLE_ROOT,
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
    market_keys = _eligible_review_market_keys(forecast)
    result_command = bundle.payload.official_result
    close_commands = bundle.payload.closing_observations

    # Validate all leakage and source constraints before the first append/replay.
    _require_result_preconditions(result_command, capture)
    for command in close_commands:
        _require_close_preconditions(command, capture)
    _require_unique_closing_streams(close_commands)
    existing_closings = tuple(
        _preflight_closing_lineage(records, command)
        for command in close_commands
    )
    existing_result = _preflight_result_lineage(records, result_command)

    closing_records = tuple(
        _append_closing_observation(store, command)
        for command in close_commands
    )
    result_record = _append_official_result(store, result_command)

    review_records: list[ReviewRecord] = []
    review_replay_count = 0
    for market_key in market_keys:
        current_records = _review_ledger_records(store)
        existing_review = _record_with_key(
            current_records,
            ReviewRecord,
            _review_idempotency_key(
                capture,
                forecast,
                result_record,
                market_key,
            ),
        )
        recorded = _build_exact_review(
            store,
            link,
            result_record,
            market_key,
        )
        review_records.append(recorded.record)
        review_replay_count += existing_review is not None

    frozen_reviews = tuple(review_records)
    managed_bundle = persist_managed_bundle(bundle, root=bundle_root)
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
            record is not None for record in existing_closings
        ),
        official_result_replayed=existing_result is not None,
        review_replay_count=review_replay_count,
        classification_counts=_classification_count_mapping(frozen_reviews),
    )


__all__ = (
    "DEFAULT_LEDGER_PATH",
    "PregameRunResult",
    "ReviewRunResult",
    "run_confirmed",
    "run_projected",
    "run_review",
)
