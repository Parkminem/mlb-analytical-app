from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.forecasting.identity import canonical_json
from analysis.operator.batch import (
    PregameBatchFailure,
    PregameBatchResult,
    PregameBatchSuccess,
    load_pregame_batch_manifest,
    prepare_pregame_batch,
)
from analysis.operator.errors import BundleValidationError
from tests.test_operator_schemas import confirmed_bundle_mapping


def _write_confirmed_bundle(
    directory: Path,
    game_pk: int,
) -> Path:
    mapping = confirmed_bundle_mapping()
    raw = canonical_json(mapping).replace("900001", str(game_pk))
    path = directory / f"{game_pk}-confirmed.json"
    path.write_text(raw, encoding="utf-8")
    return path


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
