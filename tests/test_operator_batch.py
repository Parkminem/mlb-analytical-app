from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.operator.batch import (
    PregameBatchFailure,
    PregameBatchResult,
    PregameBatchSuccess,
    audit_pregame_batch,
    load_pregame_batch_manifest,
    prepare_pregame_batch,
    run_pregame_batch,
)
from analysis.operator.errors import BundleValidationError
from analysis.shadow import canonical_legacy_decision_payload
from tests.test_operator_pregame import _confirmed_game, _ordered_legacy_decision
from tests.test_operator_schemas import confirmed_bundle_mapping


def _write_confirmed_bundle(
    directory: Path,
    game_pk: int,
    *,
    forecast_config_version: str | None = None,
    legacy_model_version: str | None = None,
) -> Path:
    mapping = confirmed_bundle_mapping()
    mapping["payload"]["game_input"] = capture_payload(_confirmed_game())
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
    path = directory / f"{game_pk}-confirmed.json"
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
    path = directory / "manifest.json"
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
