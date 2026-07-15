"""Projected and confirmed pregame shadow-run orchestration."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    LedgerReferenceError,
    LedgerStore,
)
from analysis.shadow import (
    ShadowRunIdempotencyKeys,
    build_shadow_capture_command,
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
    OperatorLedgerError,
    OperatorPreconditionError,
)
from .schemas import ConfirmedBundle, ProjectedBundle


DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ledger" / "shadow.jsonl"
)

_Bundle = ProjectedBundle | ConfirmedBundle
_T = TypeVar("_T")
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


def _ledger_operation(label: str, operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except _LEDGER_ERRORS as error:
        raise OperatorLedgerError(f"{label}: {error}") from error


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
        capture
        for capture in captures
        if capture.game_pk == game_pk
        and capture.captured_at < capture.first_pitch
    )
    superseded_ids = frozenset(
        capture.supersedes_capture_id
        for capture in game_captures
        if capture.supersedes_capture_id is not None
    )
    return tuple(
        capture
        for capture in game_captures
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

    store = _open_ledger(ledger_path)
    captures = _captures(store)
    existing_capture = _existing_keyed_capture(captures, capture_key)
    supersedes_capture_id = _resolve_supersession_target(
        stage=expected_stage,
        captures=captures,
        existing_capture=existing_capture,
        game_pk=bundle.game_pk,
    )

    forecast = _pregame_operation(
        lambda: run_forecast(bundle.payload.game_input, config)
    )

    forecast_replayed, link_replayed = _record_replay_state(
        store,
        existing_capture=existing_capture,
        forecast_run_id=forecast.forecast_run_id,
    )
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
            ),
        )
    )
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


__all__ = (
    "DEFAULT_LEDGER_PATH",
    "PregameRunResult",
    "run_confirmed",
    "run_projected",
)
