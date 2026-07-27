from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pytest

from analysis.ledger import (
    CaptureCommand,
    ForecastCommand,
    LedgerCheckpoint,
    LedgerCorruptionError,
    LedgerStore,
)
from analysis.ledger.checkpoint import CheckpointError, build_checkpoint


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


def seeded_ledger_store(
    tmp_path: Path,
) -> tuple[LedgerStore, tuple[object, ...]]:
    store = LedgerStore(tmp_path / "ledger.jsonl")
    capture = store.append_capture("capture:seed", capture_command())
    forecast = store.append_forecast(forecast_command())
    store.link_capture_forecast(capture.capture_id, forecast.forecast_run_id)
    return store, tuple(store.iter_records())


def read_checkpoint_json(checkpoint: LedgerCheckpoint) -> dict[str, object]:
    value = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_checkpoint_json(
    checkpoint: LedgerCheckpoint,
    value: dict[str, object],
) -> None:
    checkpoint.path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def mutate_capture_game_pk(value: dict[str, object]) -> None:
    value["records"][0]["metadata"]["game_pk"] = 900002


def mutate_unique_idempotency_key(value: dict[str, object]) -> None:
    value["records"][0]["idempotency_key"] = "capture:forged"


def mutate_contiguous_offsets(value: dict[str, object]) -> None:
    first = value["records"][0]
    second = value["records"][1]
    first["byte_length"] += 1
    second["byte_offset"] += 1
    second["byte_length"] -= 1


def test_checkpoint_builds_from_strictly_validated_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    ledger_before = store.path.read_bytes()
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    checkpoint = store.pregame_checkpoint()

    assert calls == 1
    assert checkpoint.record_count == len(expected_records)
    assert checkpoint.fingerprint.size == store.path.stat().st_size
    assert checkpoint.fingerprint.sha256
    assert checkpoint.latest_appended_at == max(
        record.envelope.appended_at for record in expected_records
    )
    assert checkpoint.path.name == "ledger.jsonl.checkpoint-v1.json"
    assert checkpoint.path.is_file()
    assert store.path.read_bytes() == ledger_before


def test_warm_checkpoint_does_not_call_read_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    store.pregame_checkpoint()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("strict ledger rebuild should not run")

    monkeypatch.setattr(LedgerStore, "_read_records", fail)

    warm = LedgerStore(store.path).pregame_checkpoint()

    assert warm.record_count == len(expected_records)


@pytest.mark.parametrize(
    "mutate",
    (
        mutate_capture_game_pk,
        mutate_unique_idempotency_key,
        mutate_contiguous_offsets,
    ),
    ids=("capture-game-pk", "idempotency-key", "contiguous-offsets"),
)
def test_coherently_tampered_checkpoint_rebuilds_from_authoritative_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    value = read_checkpoint_json(checkpoint)
    mutate(value)
    write_checkpoint_json(checkpoint, value)
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    assert rebuilt.records[0].game_pk == expected_records[0].game_pk
    assert rebuilt.records[0].idempotency_key == "capture:seed"
    assert rebuilt.records[0].byte_length == len(
        store.path.read_bytes().splitlines(keepends=True)[0]
    )


def test_changed_ledger_rebuilds_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    original = store.pregame_checkpoint()
    store.append_capture("capture:new1", capture_command(game_pk=900002))
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    assert rebuilt.record_count == original.record_count + 1
    assert rebuilt.fingerprint.sha256 != original.fingerprint.sha256


def test_same_size_same_mtime_ledger_mutation_forces_strict_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    original = store.pregame_checkpoint()
    original_stat = store.path.stat()
    mutated = store.path.read_bytes().replace(
        b'"capture:seed"',
        b'"capture:edit"',
        1,
    )
    assert len(mutated) == original_stat.st_size
    store.path.write_bytes(mutated)
    os.utime(
        store.path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    assert rebuilt.fingerprint.device == original.fingerprint.device
    assert rebuilt.fingerprint.inode == original.fingerprint.inode
    assert rebuilt.fingerprint.size == original.fingerprint.size
    assert rebuilt.fingerprint.mtime_ns == original.fingerprint.mtime_ns
    assert rebuilt.fingerprint.sha256 != original.fingerprint.sha256
    assert rebuilt.record("capture", next(
        record.capture_id
        for record in store.iter_records("capture")
    )).idempotency_key == "capture:edit"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(schema_version=True),
        lambda value: value["records"][1].update(
            record_id=value["records"][0]["record_id"]
        ),
        lambda value: value["records"][1].update(
            byte_offset=value["records"][0]["byte_offset"]
        ),
        lambda value: value["records"][0].update(
            byte_length=value["fingerprint"]["size"] + 1
        ),
        lambda value: value["records"][2]["metadata"].update(
            forecast_run_id="missing-forecast"
        ),
    ],
    ids=(
        "schema-version",
        "schema-version-boolean",
        "duplicate-record-id",
        "overlapping-offset",
        "impossible-length",
        "contradictory-link",
    ),
)
def test_contradictory_checkpoint_is_rebuilt_from_strict_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    value = read_checkpoint_json(checkpoint)
    mutate(value)
    write_checkpoint_json(checkpoint, value)
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    assert rebuilt.record_count == len(expected_records)
    assert [entry.record_id for entry in rebuilt.records] == [
        record.envelope.record_id for record in expected_records
    ]


def test_duplicate_same_kind_idempotency_key_rebuilds_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    store.append_capture("capture:next", capture_command(game_pk=900002))
    checkpoint = store.pregame_checkpoint()
    value = read_checkpoint_json(checkpoint)
    captures = [
        entry
        for entry in value["records"]
        if entry["record_kind"] == "capture"
    ]
    captures[1]["idempotency_key"] = captures[0]["idempotency_key"]
    write_checkpoint_json(checkpoint, value)
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    rebuilt_keys = [
        entry.idempotency_key
        for entry in rebuilt.records
        if entry.record_kind == "capture"
    ]
    assert rebuilt_keys == ["capture:seed", "capture:next"]


def test_malformed_checkpoint_is_rebuilt_safely(
    tmp_path: Path,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    checkpoint.path.write_text("{", encoding="utf-8")

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert rebuilt.record_count == len(expected_records)
    assert read_checkpoint_json(rebuilt)["record_count"] == len(expected_records)


def test_symlinked_checkpoint_is_replaced_without_touching_target(
    tmp_path: Path,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    victim = tmp_path / "victim.json"
    victim.write_text("do not touch\n", encoding="utf-8")
    checkpoint.path.unlink()
    checkpoint.path.symlink_to(victim)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert rebuilt.record_count == len(expected_records)
    assert victim.read_text(encoding="utf-8") == "do not touch\n"
    assert rebuilt.path.is_file()
    assert not rebuilt.path.is_symlink()


def test_deleted_checkpoint_is_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    checkpoint.path.unlink()
    real_read_records = LedgerStore._read_records
    calls = 0

    def read_records_spy(
        instance: LedgerStore,
        descriptor: int | None = None,
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return real_read_records(instance, descriptor)

    monkeypatch.setattr(LedgerStore, "_read_records", read_records_spy)

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert calls == 1
    assert rebuilt.record_count == len(expected_records)
    assert rebuilt.path.is_file()


def test_deeply_nested_checkpoint_json_is_disposable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    checkpoint.path.write_text("[[[0]]]", encoding="utf-8")
    real_loads = json.loads

    def recurse_on_checkpoint(
        value: str | bytes | bytearray,
        *args: object,
        **kwargs: object,
    ) -> object:
        if value == "[[[0]]]":
            raise RecursionError("injected deeply nested checkpoint")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(
        "analysis.ledger.checkpoint.json.loads",
        recurse_on_checkpoint,
    )

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert rebuilt.record_count == len(expected_records)
    assert rebuilt.path.is_file()


def test_checkpoint_removal_during_post_read_validation_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    real_stat = os.stat
    checkpoint_stat_calls = 0

    def remove_before_post_read_stat(
        path: str | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal checkpoint_stat_calls
        if (
            path == checkpoint.path.name
            and kwargs.get("dir_fd") is not None
        ):
            checkpoint_stat_calls += 1
            if checkpoint_stat_calls == 2:
                os.unlink(path, dir_fd=kwargs["dir_fd"])
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        "analysis.ledger.checkpoint.os.stat",
        remove_before_post_read_stat,
    )

    rebuilt = LedgerStore(store.path).pregame_checkpoint()

    assert checkpoint_stat_calls >= 2
    assert rebuilt.record_count == len(expected_records)
    assert rebuilt.path.is_file()


def test_authoritative_row_read_failure_is_ledger_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    store.pregame_checkpoint()
    real_pread = os.pread
    calls = 0

    def fail_first_row_read(
        descriptor: int,
        length: int,
        offset: int,
    ) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected authoritative row read failure")
        return real_pread(descriptor, length, offset)

    monkeypatch.setattr(
        "analysis.ledger.checkpoint.os.pread",
        fail_first_row_read,
    )

    with pytest.raises(LedgerCorruptionError, match="authoritative ledger"):
        LedgerStore(store.path).pregame_checkpoint()


def test_invalid_ledger_never_uses_or_replaces_checkpoint(
    tmp_path: Path,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    checkpoint_before = checkpoint.path.read_bytes()
    store.path.write_bytes(store.path.read_bytes()[:-1])

    with pytest.raises(LedgerCorruptionError, match="newline"):
        LedgerStore(store.path).pregame_checkpoint()

    assert checkpoint.path.read_bytes() == checkpoint_before


def test_warm_load_rejects_ledger_mutation_after_fingerprinting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _ = seeded_ledger_store(tmp_path)
    store.pregame_checkpoint()
    real_load = LedgerCheckpoint.load

    def load_then_mutate(
        cls,
        path: Path,
        ledger_stat: os.stat_result,
        ledger_sha256: str,
        *,
        ledger_descriptor: int | None = None,
    ) -> LedgerCheckpoint:
        checkpoint = real_load(
            path,
            ledger_stat,
            ledger_sha256,
            ledger_descriptor=ledger_descriptor,
        )
        store.path.write_bytes(store.path.read_bytes() + b" ")
        return checkpoint

    monkeypatch.setattr(
        LedgerCheckpoint,
        "load",
        classmethod(load_then_mutate),
    )

    with pytest.raises(LedgerCorruptionError, match="changed"):
        LedgerStore(store.path).pregame_checkpoint()


def test_build_rejects_raw_rows_that_disagree_with_validated_records(
    tmp_path: Path,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    original = store.path.read_bytes()
    store.path.write_bytes(
        original.replace(b'"capture:seed"', b'"capture:edit"', 1)
    )

    with pytest.raises(CheckpointError, match="validated record"):
        build_checkpoint(store.path, expected_records)


def test_build_rejects_fabricated_record_with_matching_envelope(
    tmp_path: Path,
) -> None:
    store, expected_records = seeded_ledger_store(tmp_path)
    fabricated = (
        replace(expected_records[0], game_pk=900002),
        *expected_records[1:],
    )

    with pytest.raises(CheckpointError, match="authoritative ledger"):
        build_checkpoint(store.path, fabricated)


def test_checkpoint_indexes_relationships_and_values_are_frozen(
    tmp_path: Path,
) -> None:
    store, records = seeded_ledger_store(tmp_path)
    checkpoint = store.pregame_checkpoint()
    capture = records[0]
    forecast = records[1]
    link = records[2]

    capture_entry = checkpoint.record("capture", capture.capture_id)
    forecast_entry = checkpoint.record(
        "forecast",
        forecast.forecast_run_id,
    )
    link_entry = checkpoint.record(
        "capture_forecast_link",
        link.envelope.record_id,
    )

    assert capture_entry is not None
    assert capture_entry.game_pk == capture.game_pk
    assert capture_entry.captured_at == capture.captured_at
    assert capture_entry.first_pitch == capture.first_pitch
    assert capture_entry.normalized_input_hash == capture.normalized_input_hash
    assert forecast_entry is not None
    assert forecast_entry.normalized_input_hash == forecast.normalized_input_hash
    assert link_entry is not None
    assert link_entry.capture_id == capture.capture_id
    assert link_entry.forecast_run_id == forecast.forecast_run_id
    assert checkpoint.record("capture", "missing") is None
    with pytest.raises(FrozenInstanceError):
        checkpoint.record_count = 0
    with pytest.raises(FrozenInstanceError):
        capture_entry.byte_offset = 0
