from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from pathlib import Path

import pytest

from analysis.forecasting.identity import canonical_json
from analysis.ledger import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    LedgerHeadChangedError,
    LedgerStore,
)


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 27, 1, 30, tzinfo=UTC)
FIRST_PITCH = datetime(2026, 7, 27, 23, 10, tzinfo=UTC)


def capture_command(
    *,
    game_pk: int = 900001,
    captured_at: datetime = CAPTURED_AT,
    normalized_input_hash: str = "input-hash-1",
    supersedes_capture_id: str | None = None,
) -> CaptureCommand:
    return CaptureCommand(
        game_pk=game_pk,
        captured_at=captured_at,
        first_pitch=FIRST_PITCH,
        lineup_state="confirmed",
        out_of_sample_eligible=True,
        normalized_input_hash=normalized_input_hash,
        payload={"away_team": "SEA", "home_team": "MIN", "markets": []},
        supersedes_capture_id=supersedes_capture_id,
    )


def forecast_command(
    *,
    forecast_run_id: str = "forecast-run-1",
    normalized_input_hash: str = "input-hash-1",
) -> ForecastCommand:
    return ForecastCommand(
        forecast_run_id=forecast_run_id,
        model_version="model-v1",
        simulation_config_hash="simulation-hash-1",
        normalized_input_hash=normalized_input_hash,
        outputs={"market_inputs": {}, "markets": []},
    )


def payload_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def test_lookup_returns_exact_relationship_and_same_latest_active_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    exact = store.append_capture("capture:exact", capture_command())
    active = store.append_capture(
        "capture:active",
        capture_command(captured_at=CAPTURED_AT + timedelta(minutes=2)),
    )
    forecast = store.append_forecast(forecast_command())
    link = store.link_capture_forecast(
        exact.capture_id,
        forecast.forecast_run_id,
    )
    unrelated = store.append_forecast(
        forecast_command(forecast_run_id="unrelated-forecast")
    )
    store.pregame_checkpoint()

    import analysis.ledger.pregame as pregame_module

    real_deserialize = pregame_module._deserialize_record
    loaded: list[tuple[str, str]] = []

    def deserialize_spy(value: object):
        record = real_deserialize(value)
        loaded.append(
            (
                record.envelope.record_kind,
                getattr(record, "forecast_run_id", ""),
            )
        )
        return record

    monkeypatch.setattr(
        pregame_module,
        "_deserialize_record",
        deserialize_spy,
    )

    lookup = store.lookup_shadow_pregame(
        "capture:exact",
        forecast.forecast_run_id,
        exact.game_pk,
    )

    from analysis.ledger import PregameLookup

    assert isinstance(lookup, PregameLookup)
    assert lookup.capture == exact
    assert lookup.active_capture == active
    assert lookup.forecast == forecast
    assert lookup.link == link
    assert ("forecast", unrelated.forecast_run_id) not in loaded
    with pytest.raises(FrozenInstanceError):
        lookup.capture = active


def test_atomic_append_matches_legacy_record_contract_and_row_order(
    tmp_path: Path,
) -> None:
    capture = capture_command()
    forecast = forecast_command()
    legacy_store = LedgerStore(tmp_path / "legacy.jsonl")
    legacy_capture = legacy_store.append_capture("capture:new", capture)
    legacy_forecast = legacy_store.append_forecast(forecast)
    legacy_link = legacy_store.link_capture_forecast(
        legacy_capture.capture_id,
        legacy_forecast.forecast_run_id,
    )
    store = LedgerStore(tmp_path / "atomic.jsonl")

    records = store.append_shadow_run_atomic(
        "capture:new",
        capture,
        forecast,
        None,
    )

    assert isinstance(records.capture, CaptureRecord)
    assert isinstance(records.forecast, ForecastRecord)
    assert isinstance(records.link, CaptureForecastLink)
    assert records.capture.game_pk == legacy_capture.game_pk
    assert records.capture.captured_at == legacy_capture.captured_at
    assert records.capture.first_pitch == legacy_capture.first_pitch
    assert records.capture.lineup_state == legacy_capture.lineup_state
    assert records.capture.out_of_sample_eligible == (
        legacy_capture.out_of_sample_eligible
    )
    assert records.capture.normalized_input_hash == (
        legacy_capture.normalized_input_hash
    )
    assert records.capture.payload == legacy_capture.payload
    assert records.capture.supersedes_capture_id == (
        legacy_capture.supersedes_capture_id
    )
    assert records.capture.envelope.record_kind == "capture"
    assert records.capture.envelope.idempotency_key == "capture:new"
    assert records.capture.envelope.payload_hash == payload_hash(capture)
    assert records.forecast.model_version == legacy_forecast.model_version
    assert records.forecast.simulation_config_hash == (
        legacy_forecast.simulation_config_hash
    )
    assert records.forecast.normalized_input_hash == (
        legacy_forecast.normalized_input_hash
    )
    assert records.forecast.outputs == legacy_forecast.outputs
    assert records.forecast.envelope.record_kind == "forecast"
    assert records.forecast.envelope.idempotency_key == (
        f"forecast:{forecast.forecast_run_id}"
    )
    assert records.forecast.envelope.payload_hash == payload_hash(forecast)
    link_payload = {
        "capture_id": records.capture.capture_id,
        "forecast_run_id": records.forecast.forecast_run_id,
    }
    assert records.link.capture_id == records.capture.capture_id
    assert records.link.forecast_run_id == records.forecast.forecast_run_id
    assert records.link.envelope.record_kind == "capture_forecast_link"
    assert records.link.envelope.idempotency_key == (
        f"capture-forecast:{records.capture.capture_id}:"
        f"{records.forecast.forecast_run_id}"
    )
    assert records.link.envelope.payload_hash == payload_hash(link_payload)
    assert [
        record.envelope.record_kind for record in store.iter_records()
    ] == ["capture", "forecast", "capture_forecast_link"]
    assert legacy_link.envelope.record_kind == records.link.envelope.record_kind


def test_atomic_append_uses_one_lock_session_and_publishes_after_ledger_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    store.pregame_checkpoint()
    real_flock = fcntl.flock
    exclusive_locks = 0

    def flock_spy(descriptor: int, operation: int) -> None:
        nonlocal exclusive_locks
        if operation == fcntl.LOCK_EX:
            exclusive_locks += 1
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", flock_spy)
    import analysis.ledger.pregame as pregame_module

    events: list[str] = []
    real_fsync = pregame_module.os.fsync
    real_publish = pregame_module._publish_checkpoint

    def fsync_spy(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    def publish_spy(checkpoint) -> None:
        events.append("publish")
        assert len(store.path.read_bytes().splitlines()) == 3
        real_publish(checkpoint)

    monkeypatch.setattr(pregame_module.os, "fsync", fsync_spy)
    monkeypatch.setattr(
        pregame_module,
        "_publish_checkpoint",
        publish_spy,
    )

    store.append_shadow_run_atomic(
        "capture:new",
        capture_command(),
        forecast_command(),
        None,
    )

    assert exclusive_locks == 2
    assert events[0:2] == ["fsync", "publish"]


def test_atomic_append_recovers_when_checkpoint_publication_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    stale = store.pregame_checkpoint()

    def fail_publish(checkpoint) -> None:
        raise OSError("injected checkpoint publication interruption")

    monkeypatch.setattr(
        "analysis.ledger.pregame._publish_checkpoint",
        fail_publish,
    )

    appended = store.append_shadow_run_atomic(
        "capture:new",
        capture_command(),
        forecast_command(),
        None,
    )

    assert len(store.path.read_bytes().splitlines()) == 3
    assert stale.record_count == 0
    monkeypatch.undo()

    recovered_store = LedgerStore(store.path)
    lookup = recovered_store.lookup_shadow_pregame(
        "capture:new",
        appended.forecast.forecast_run_id,
        appended.capture.game_pk,
    )

    assert lookup.capture == appended.capture
    assert lookup.forecast == appended.forecast
    assert lookup.link == appended.link
    assert recovered_store.pregame_checkpoint().record_count == 3


def test_atomic_exact_replay_returns_same_records_without_appending(
    tmp_path: Path,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    capture = capture_command()
    forecast = forecast_command()
    first = store.append_shadow_run_atomic(
        "capture:replay",
        capture,
        forecast,
        None,
    )
    before = store.path.read_bytes()

    replay = store.append_shadow_run_atomic(
        "capture:replay",
        capture,
        forecast,
        first.capture.capture_id,
    )

    assert replay == first
    assert store.path.read_bytes() == before


def test_atomic_append_completes_partial_relationship_without_duplicates(
    tmp_path: Path,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    capture = store.append_capture("capture:partial", capture_command())
    forecast = store.append_forecast(forecast_command())
    before = tuple(store.iter_records())

    completed = store.append_shadow_run_atomic(
        "capture:partial",
        capture_command(),
        forecast_command(),
        capture.capture_id,
    )

    assert completed.capture == capture
    assert completed.forecast == forecast
    assert completed.link.capture_id == capture.capture_id
    assert len(before) == 2
    assert len(tuple(store.iter_records())) == 3


def test_atomic_append_rejects_changed_active_head_without_partial_write(
    tmp_path: Path,
) -> None:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    active = store.append_capture("capture:active", capture_command())
    before = store.path.read_bytes()

    with pytest.raises(LedgerHeadChangedError, match=active.capture_id):
        store.append_shadow_run_atomic(
            "capture:stale",
            capture_command(captured_at=CAPTURED_AT + timedelta(minutes=1)),
            forecast_command(),
            None,
        )

    assert store.path.read_bytes() == before
