from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import pytest

import analysis.operator.batch as batch_module
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.operator.bundles import bundle_sha256
from analysis.operator.batch import (
    PregameBatchFailure,
    PregameBatchResult,
    PregameBatchSuccess,
    audit_pregame_batch,
    load_pregame_batch_manifest,
    prepare_pregame_batch,
    run_pregame_batch,
)
from analysis.operator.errors import BundleValidationError, OperatorIOError
from analysis.operator.schemas import parse_confirmed_bundle_json
from analysis.shadow import canonical_legacy_decision_payload
from tests.test_operator_pregame import _confirmed_game, _ordered_legacy_decision
from tests.test_operator_schemas import confirmed_bundle_mapping


def _write_confirmed_bundle(
    directory: Path,
    game_pk: int,
    *,
    captured_at_offset_minutes: int = 0,
    forecast_config_version: str | None = None,
    legacy_model_version: str | None = None,
    name: str | None = None,
) -> Path:
    mapping = confirmed_bundle_mapping()
    game = _confirmed_game()
    if captured_at_offset_minutes:
        game = replace(
            game,
            captured_at=(
                game.captured_at
                + timedelta(minutes=captured_at_offset_minutes)
            ),
        )
    mapping["payload"]["game_input"] = capture_payload(game)
    mapping["payload"]["legacy_decision"] = (
        canonical_legacy_decision_payload(_ordered_legacy_decision())
    )
    if forecast_config_version is not None:
        mapping["payload"]["forecast_config_version"] = (
            forecast_config_version
        )
    if legacy_model_version is not None:
        mapping["payload"]["legacy_decision"]["model_version"] = (
            legacy_model_version
        )
    raw = canonical_json(mapping).replace("900001", str(game_pk))
    path = directory / (name or f"{game_pk}-confirmed.json")
    path.write_text(raw, encoding="utf-8")
    return path


def _write_test_registry(directory: Path) -> Path:
    registry = directory / "forecasting"
    registry.mkdir()
    config = SimulationConfig(simulations=12)
    (registry / f"{config.version}.json").write_text(
        canonical_json(config.to_mapping()),
        encoding="utf-8",
    )
    return registry


def _write_manifest(
    directory: Path,
    *,
    entries: tuple[tuple[int, str], ...],
    name: str = "manifest.json",
    stage: str = "confirmed",
    **extra: object,
) -> Path:
    mapping: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "entries": [
            {"game_pk": game_pk, "bundle": bundle}
            for game_pk, bundle in entries
        ],
    }
    mapping.update(extra)
    path = directory / name
    path.write_text(canonical_json(mapping), encoding="utf-8")
    return path


def _success(game_pk: int, bundle_path: Path) -> PregameBatchSuccess:
    suffix = f"{game_pk:032x}"[-32:]
    return PregameBatchSuccess(
        game_pk=game_pk,
        bundle_path=bundle_path,
        capture_id=suffix,
        forecast_run_id=(suffix * 2)[:64],
        capture_forecast_link_record_id=("f" * 31) + suffix[-1],
        managed_bundle_path=bundle_path.parent / f"managed-{game_pk}.json",
        managed_bundle_hash="a" * 64,
        managed_bundle_replayed=False,
        capture_replayed=False,
        forecast_replayed=False,
        link_replayed=False,
    )


def _failure(game_pk: int, bundle_path: Path) -> PregameBatchFailure:
    return PregameBatchFailure(
        game_pk=game_pk,
        bundle_path=bundle_path,
        error_type="OperatorPreconditionError",
        message="candidate is not review-eligible",
        exit_code=4,
    )


def test_prepare_batch_reads_and_canonicalizes_every_bundle_before_work(
    tmp_path: Path,
) -> None:
    first = _write_confirmed_bundle(tmp_path, 900001)
    second = _write_confirmed_bundle(tmp_path, 900002)
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, first.name), (900002, second.name)),
    )

    manifest = load_pregame_batch_manifest(
        manifest_path,
        expected_stage="confirmed",
    )
    prepared = prepare_pregame_batch(
        manifest,
        manifest_path=manifest_path,
    )

    assert tuple(item.game_pk for item in prepared) == (900001, 900002)
    assert all(item.bundle_path.is_absolute() for item in prepared)
    assert all(not item.canonical_bundle.endswith(b"\n") for item in prepared)
    assert json.loads(prepared[1].canonical_bundle)["game_pk"] == 900002


@pytest.mark.parametrize(
    ("entries", "extra", "message"),
    (
        (((900001, "one.json"), (900001, "two.json")), {}, "game_pk"),
        (((900001, "one.json"), (900002, "one.json")), {}, "bundle"),
        ((), {}, "entries"),
        (((900001, "one.json"),), {"unknown": True}, "extra"),
    ),
)
def test_manifest_rejects_incomplete_or_ambiguous_targets(
    tmp_path: Path,
    entries: tuple[tuple[int, str], ...],
    extra: dict[str, object],
    message: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, entries=entries, **extra)

    with pytest.raises(BundleValidationError, match=message):
        load_pregame_batch_manifest(
            manifest_path,
            expected_stage="confirmed",
        )


def test_manifest_rejects_command_stage_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, "one.json"),),
        stage="projected",
    )

    with pytest.raises(BundleValidationError, match="stage"):
        load_pregame_batch_manifest(
            manifest_path,
            expected_stage="confirmed",
        )


def test_prepare_batch_rejects_bundle_game_mismatch(tmp_path: Path) -> None:
    bundle = _write_confirmed_bundle(tmp_path, 900002)
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    manifest = load_pregame_batch_manifest(
        manifest_path,
        expected_stage="confirmed",
    )

    with pytest.raises(BundleValidationError, match="game_pk"):
        prepare_pregame_batch(manifest, manifest_path=manifest_path)


def test_complete_receipt_preserves_manifest_order_and_triplets(
    tmp_path: Path,
) -> None:
    first = _success(900001, tmp_path / "one.json")
    second = _success(900002, tmp_path / "two.json")

    result = PregameBatchResult(
        stage="confirmed",
        max_workers=3,
        target_game_pks=(900001, 900002),
        entries=(first, second),
    )

    assert result.complete is True
    assert result.target_count == 2
    assert result.success_count == 2
    assert result.failure_count == 0
    mapping = result.to_mapping()
    assert [entry["game_pk"] for entry in mapping["entries"]] == [
        900001,
        900002,
    ]
    assert mapping["entries"][0]["capture_id"] == first.capture_id
    assert mapping["entries"][0]["forecast_run_id"] == first.forecast_run_id
    assert (
        mapping["entries"][0]["capture_forecast_link_record_id"]
        == first.capture_forecast_link_record_id
    )


@pytest.mark.parametrize(
    ("targets", "entries", "message"),
    (
        (
            (900001, 900002),
            (_success(900001, Path("one.json")),),
            "coverage",
        ),
        (
            (900001, 900002),
            (
                _success(900002, Path("two.json")),
                _success(900001, Path("one.json")),
            ),
            "order",
        ),
        (
            (900001, 900001),
            (
                _success(900001, Path("one.json")),
                _failure(900001, Path("two.json")),
            ),
            "target_game_pks",
        ),
    ),
)
def test_receipt_rejects_missing_reordered_or_duplicate_results(
    targets: tuple[int, ...],
    entries: tuple[PregameBatchSuccess | PregameBatchFailure, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PregameBatchResult(
            stage="confirmed",
            max_workers=3,
            target_game_pks=targets,
            entries=entries,
        )


def test_partial_receipt_is_never_complete(tmp_path: Path) -> None:
    success = _success(900001, tmp_path / "one.json")
    failure = _failure(900002, tmp_path / "two.json")

    result = PregameBatchResult(
        stage="confirmed",
        max_workers=2,
        target_game_pks=(900001, 900002),
        entries=(success, failure),
    )

    assert result.complete is False
    assert result.success_count == 1
    assert result.failure_count == 1
    mapping = result.to_mapping()
    assert mapping["complete"] is False
    assert mapping["entries"][1]["status"] == "failure"
    assert mapping["entries"][1]["error_type"] == "OperatorPreconditionError"


def test_two_game_execution_writes_complete_audited_receipt(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    first = _write_confirmed_bundle(tmp_path, 900001)
    second = _write_confirmed_bundle(tmp_path, 900002)
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, first.name), (900002, second.name)),
    )
    ledger = tmp_path / "ledger" / "shadow.jsonl"
    bundle_root = tmp_path / "managed"
    receipt = tmp_path / "receipts" / "confirmed.json"

    result = run_pregame_batch(
        manifest_path,
        expected_stage="confirmed",
        ledger_path=ledger,
        bundle_root=bundle_root,
        receipt_path=receipt,
        registry_root=registry,
        max_workers=2,
    )

    assert result.complete is True
    assert result.success_count == 2
    assert result.failure_count == 0
    assert json.loads(receipt.read_text(encoding="utf-8")) == (
        result.to_mapping()
    )
    assert receipt.read_bytes().endswith(b"\n")
    assert {item.game_pk for item in result.successes} == {900001, 900002}


def test_exact_batch_replay_preserves_ids_and_reports_replay_flags(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    runtime = {
        "ledger_path": tmp_path / "ledger" / "shadow.jsonl",
        "bundle_root": tmp_path / "managed",
        "receipt_path": tmp_path / "receipt.json",
        "registry_root": registry,
        "max_workers": 1,
    }

    first = run_pregame_batch(
        manifest_path,
        expected_stage="confirmed",
        **runtime,
    )
    second = run_pregame_batch(
        manifest_path,
        expected_stage="confirmed",
        **runtime,
    )

    first_success = first.successes[0]
    second_success = second.successes[0]
    assert second_success.capture_id == first_success.capture_id
    assert second_success.forecast_run_id == first_success.forecast_run_id
    assert (
        second_success.capture_forecast_link_record_id
        == first_success.capture_forecast_link_record_id
    )
    assert second_success.capture_replayed is True
    assert second_success.forecast_replayed is True
    assert second_success.link_replayed is True


def test_exact_historical_batch_replay_survives_newer_superseding_capture(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    old_bundle = _write_confirmed_bundle(
        tmp_path,
        900001,
        name="old-confirmed.json",
    )
    old_manifest = _write_manifest(
        tmp_path,
        entries=((900001, old_bundle.name),),
        name="old-manifest.json",
    )
    newer_bundle = _write_confirmed_bundle(
        tmp_path,
        900001,
        captured_at_offset_minutes=5,
        name="newer-confirmed.json",
    )
    newer_manifest = _write_manifest(
        tmp_path,
        entries=((900001, newer_bundle.name),),
        name="newer-manifest.json",
    )
    runtime = {
        "ledger_path": tmp_path / "ledger" / "shadow.jsonl",
        "bundle_root": tmp_path / "managed",
        "receipt_path": tmp_path / "receipt.json",
        "registry_root": registry,
        "max_workers": 1,
    }

    first = run_pregame_batch(
        old_manifest,
        expected_stage="confirmed",
        **runtime,
    )
    newer = run_pregame_batch(
        newer_manifest,
        expected_stage="confirmed",
        **runtime,
    )
    replay = run_pregame_batch(
        old_manifest,
        expected_stage="confirmed",
        **runtime,
    )

    first_success = first.successes[0]
    newer_success = newer.successes[0]
    replay_success = replay.successes[0]
    assert newer_success.capture_id != first_success.capture_id
    assert replay.complete is True
    assert replay_success.capture_id == first_success.capture_id
    assert replay_success.forecast_run_id == first_success.forecast_run_id
    assert (
        replay_success.capture_forecast_link_record_id
        == first_success.capture_forecast_link_record_id
    )
    assert replay_success.capture_replayed is True
    assert replay_success.forecast_replayed is True
    assert replay_success.link_replayed is True


def test_one_service_failure_keeps_success_and_writes_partial_receipt(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    good = _write_confirmed_bundle(tmp_path, 900001)
    stale = _write_confirmed_bundle(
        tmp_path,
        900002,
        legacy_model_version="2026-08-04",
    )
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, good.name), (900002, stale.name)),
    )
    ledger = tmp_path / "ledger" / "shadow.jsonl"
    receipt = tmp_path / "receipt.json"

    result = run_pregame_batch(
        manifest_path,
        expected_stage="confirmed",
        ledger_path=ledger,
        bundle_root=tmp_path / "managed",
        receipt_path=receipt,
        registry_root=registry,
        max_workers=2,
    )

    assert result.complete is False
    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.successes[0].game_pk == 900001
    assert result.failures[0].game_pk == 900002
    assert result.failures[0].error_type == "OperatorPreconditionError"
    assert json.loads(receipt.read_text(encoding="utf-8"))["complete"] is False


def test_coverage_audit_converts_missing_triplet_to_failure(
    tmp_path: Path,
) -> None:
    success = _success(900001, tmp_path / "bundle.json")
    result = PregameBatchResult(
        stage="confirmed",
        max_workers=1,
        target_game_pks=(900001,),
        entries=(success,),
    )

    audited = audit_pregame_batch(
        result,
        ledger_path=tmp_path / "empty" / "shadow.jsonl",
    )

    assert audited.complete is False
    assert audited.failures[0].error_type == "coverage_audit"
    assert "capture" in audited.failures[0].message


def test_coverage_audit_rejects_mixed_capture_forecast_relationship(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    first_bundle = _write_confirmed_bundle(tmp_path, 900001)
    second_bundle = _write_confirmed_bundle(tmp_path, 900002)
    manifest = _write_manifest(
        tmp_path,
        entries=(
            (900001, first_bundle.name),
            (900002, second_bundle.name),
        ),
    )
    ledger = tmp_path / "ledger" / "shadow.jsonl"
    recorded = run_pregame_batch(
        manifest,
        expected_stage="confirmed",
        ledger_path=ledger,
        bundle_root=tmp_path / "managed",
        receipt_path=tmp_path / "receipt.json",
        registry_root=registry,
        max_workers=2,
    )
    first, second = recorded.successes
    mixed = replace(first, forecast_run_id=second.forecast_run_id)
    claimed = PregameBatchResult(
        stage="confirmed",
        max_workers=2,
        target_game_pks=(900001,),
        entries=(mixed,),
    )

    audited = audit_pregame_batch(claimed, ledger_path=ledger)

    assert audited.complete is False
    assert audited.failures[0].error_type == "coverage_audit"
    assert "different forecast" in audited.failures[0].message


def test_receipt_publish_failure_preserves_durable_batch_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )

    original_publish = batch_module._publish_batch_receipt
    publish_calls = 0

    def fail_publish(
        path: Path,
        result: PregameBatchResult,
        **kwargs,
    ) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise OperatorIOError("disk full")
        original_publish(path, result, **kwargs)

    monkeypatch.setattr(batch_module, "_publish_batch_receipt", fail_publish)

    with pytest.raises(
        batch_module.PregameBatchReceiptError,
        match="ledger.*durable.*rerun",
    ) as raised:
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=tmp_path / "ledger" / "shadow.jsonl",
            bundle_root=tmp_path / "managed",
            receipt_path=tmp_path / "receipt.json",
            registry_root=registry,
            max_workers=1,
        )

    assert raised.value.result.complete is True
    assert raised.value.result.success_count == 1
    assert raised.value.result.successes[0].capture_id
    assert raised.value.result.successes[0].forecast_run_id
    assert raised.value.result.successes[0].capture_forecast_link_record_id

    replay = run_pregame_batch(
        manifest,
        expected_stage="confirmed",
        ledger_path=tmp_path / "ledger" / "shadow.jsonl",
        bundle_root=tmp_path / "managed",
        receipt_path=tmp_path / "receipt.json",
        registry_root=registry,
        max_workers=1,
    )

    durable_success = raised.value.result.successes[0]
    replay_success = replay.successes[0]
    assert replay_success.capture_id == durable_success.capture_id
    assert replay_success.forecast_run_id == durable_success.forecast_run_id
    assert (
        replay_success.capture_forecast_link_record_id
        == durable_success.capture_forecast_link_record_id
    )
    assert replay_success.capture_replayed is True
    assert replay_success.forecast_replayed is True
    assert replay_success.link_replayed is True
    assert (tmp_path / "receipt.json").exists()


def test_atomic_receipt_replace_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    result = PregameBatchResult(
        stage="confirmed",
        max_workers=1,
        target_game_pks=(900001,),
        entries=(_success(900001, tmp_path / "bundle.json"),),
    )

    def fail_replace(*args, **kwargs) -> None:
        raise OSError("replace unavailable")

    monkeypatch.setattr(batch_module.os, "replace", fail_replace)

    with pytest.raises(OperatorIOError, match="unable to publish"):
        batch_module._publish_batch_receipt(receipt, result)

    assert not receipt.exists()
    assert list(tmp_path.glob(".receipt.json.*")) == []


@pytest.mark.parametrize("max_workers", (0, 5, True))
def test_invalid_worker_count_fails_before_receipt(
    tmp_path: Path,
    max_workers: object,
) -> None:
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest_path = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    receipt = tmp_path / "receipt.json"

    with pytest.raises(ValueError, match="max_workers"):
        run_pregame_batch(
            manifest_path,
            expected_stage="confirmed",
            ledger_path=tmp_path / "ledger" / "shadow.jsonl",
            bundle_root=tmp_path / "managed",
            receipt_path=receipt,
            max_workers=max_workers,
        )

    assert not receipt.exists()


@pytest.mark.parametrize(
    "collision",
    ("manifest", "bundle", "ledger", "checkpoint", "lock"),
)
def test_receipt_path_cannot_replace_authoritative_or_input_files(
    tmp_path: Path,
    collision: str,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    ledger = tmp_path / "ledger" / "shadow.jsonl"
    receipt_paths = {
        "manifest": manifest,
        "bundle": bundle,
        "ledger": ledger,
        "checkpoint": Path(f"{ledger}.checkpoint-v1.json"),
        "lock": Path(f"{ledger}.lock"),
    }
    manifest_before = manifest.read_bytes()
    bundle_before = bundle.read_bytes()

    with pytest.raises(OperatorIOError, match="receipt.*collides"):
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=ledger,
            bundle_root=tmp_path / "managed",
            receipt_path=receipt_paths[collision],
            registry_root=registry,
            max_workers=1,
        )

    assert manifest.read_bytes() == manifest_before
    assert bundle.read_bytes() == bundle_before
    assert not ledger.exists()


def test_receipt_path_cannot_replace_managed_bundle_artifact(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    parsed = parse_confirmed_bundle_json(bundle.read_bytes())
    digest = bundle_sha256(parsed)
    managed_target = (
        tmp_path
        / "managed"
        / "900001"
        / f"confirmed-v2-{digest[:16]}.json"
    )

    with pytest.raises(OperatorIOError, match="receipt.*collides"):
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=tmp_path / "ledger" / "shadow.jsonl",
            bundle_root=tmp_path / "managed",
            receipt_path=managed_target,
            registry_root=registry,
            max_workers=1,
        )

    assert not managed_target.exists()


def test_receipt_path_cannot_replace_forecast_registry_file(
    tmp_path: Path,
) -> None:
    registry = _write_test_registry(tmp_path)
    config_path = registry / f"{SimulationConfig(simulations=12).version}.json"
    config_before = config_path.read_bytes()
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )

    with pytest.raises(OperatorIOError, match="receipt.*collides"):
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=tmp_path / "ledger" / "shadow.jsonl",
            bundle_root=tmp_path / "managed",
            receipt_path=config_path,
            registry_root=registry,
            max_workers=1,
        )

    assert config_path.read_bytes() == config_before


@pytest.mark.parametrize("artifact_store", ("managed", "registry"))
def test_receipt_path_rejects_every_immutable_artifact_store_file(
    tmp_path: Path,
    artifact_store: str,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle_root = tmp_path / "managed"
    if artifact_store == "managed":
        unrelated = bundle_root / "999999" / "confirmed-v2-unrelated.json"
    else:
        unrelated = registry / "unrelated-version.json"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_bytes(b"immutable-artifact")
    before = unrelated.read_bytes()
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )

    with pytest.raises(OperatorIOError, match="receipt.*collides"):
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=tmp_path / "ledger" / "shadow.jsonl",
            bundle_root=bundle_root,
            receipt_path=unrelated,
            registry_root=registry,
            max_workers=1,
        )

    assert unrelated.read_bytes() == before


def test_receipt_parent_swap_cannot_redirect_publication_to_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _write_test_registry(tmp_path)
    bundle = _write_confirmed_bundle(tmp_path, 900001)
    manifest = _write_manifest(
        tmp_path,
        entries=((900001, bundle.name),),
    )
    ledger = tmp_path / "ledger" / "shadow.jsonl"
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()
    receipt = receipt_parent / ledger.name
    ledger_before_swap: list[bytes] = []
    original_execute = batch_module._execute_prepared_batch

    def execute_then_swap(*args, **kwargs):
        entries = original_execute(*args, **kwargs)
        ledger_before_swap.append(ledger.read_bytes())
        receipt_parent.rename(tmp_path / "receipts-original")
        receipt_parent.symlink_to(ledger.parent, target_is_directory=True)
        return entries

    monkeypatch.setattr(
        batch_module,
        "_execute_prepared_batch",
        execute_then_swap,
    )

    with pytest.raises(
        batch_module.PregameBatchReceiptError,
        match="receipt.*changed|publication failed",
    ):
        run_pregame_batch(
            manifest,
            expected_stage="confirmed",
            ledger_path=ledger,
            bundle_root=tmp_path / "managed",
            receipt_path=receipt,
            registry_root=registry,
            max_workers=1,
        )

    assert ledger.read_bytes() == ledger_before_swap[0]
