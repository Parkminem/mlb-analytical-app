from dataclasses import fields, replace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.operator.service as service_module
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType
from analysis.ledger import LedgerStore
from analysis.operator.errors import OperatorIOError
from analysis.operator.reporting import (
    CALIBRATION_JSON_FILENAME,
    CALIBRATION_MARKDOWN_FILENAME,
    CONFIRMED_GAME_THRESHOLD,
    DECISIONS_PER_MARKET_TYPE_THRESHOLD,
    REPORT_SCHEMA_VERSION,
    build_report_mapping,
    render_report_markdown,
    write_calibration_reports,
)
from analysis.operator.schemas import parse_review_bundle_json
from analysis.operator.service import run_review
from analysis.review.calibration import (
    CalibrationMetrics,
    GROUPING_DIMENSIONS,
    build_calibration_report,
)
from tests.test_calibration import calibration_row
from tests.test_forecasting_engine import _complete_game
from tests.test_operator_review import (
    _legacy_snapshot,
    _raw,
    _record_pregame,
    _review_mapping,
)


def _ledger_record(appended_at):
    return SimpleNamespace(
        envelope=SimpleNamespace(appended_at=appended_at),
    )


def _mapping(
    tmp_path: Path,
    rows,
    *,
    ledger_records=(),
):
    report = build_calibration_report(tuple(rows))
    return build_report_mapping(
        ledger_path=tmp_path / "ledger" / "shadow.jsonl",
        ledger_records=tuple(ledger_records),
        rows=tuple(rows),
        report=report,
    )


def _unique_row(index: int, **overrides):
    values = {
        "game_pk": 910000 + index,
        "capture_id": f"capture-{index}",
        "forecast_run_id": f"forecast-{index}",
        "market_key": f"game:{910000 + index}:ml:away",
    }
    values.update(overrides)
    return calibration_row(**values)


def _run_review_for_game(
    *,
    ledger_path: Path,
    bundle_root: Path,
    report_dir: Path,
    game_pk: int,
    day_offset: int = 0,
):
    base = _complete_game()
    game = replace(
        base,
        identity=replace(base.identity, game_pk=game_pk),
        captured_at=base.captured_at + timedelta(seconds=day_offset),
    )
    confirmed = _record_pregame(game, _legacy_snapshot(), ledger_path)
    bundle = parse_review_bundle_json(_raw(_review_mapping(confirmed)))
    return run_review(
        bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        report_dir=report_dir,
    )


def test_report_mapping_is_versioned_complete_and_uses_latest_ledger_append(
    tmp_path: Path,
) -> None:
    first = _unique_row(1)
    second = _unique_row(
        2,
        market_result="loss",
        selected_decimal_price=None,
        closing_selected_decimal_price=None,
        entry_no_vig_probability=None,
        closing_no_vig_probability=None,
    )
    early = _complete_game().captured_at
    latest = early + timedelta(hours=9)

    mapping = _mapping(
        tmp_path,
        (first, second),
        ledger_records=(_ledger_record(latest), _ledger_record(early)),
    )

    assert mapping["schema_version"] == REPORT_SCHEMA_VERSION == 1
    assert mapping["ledger_path"] == str(
        (tmp_path / "ledger" / "shadow.jsonl").resolve()
    )
    assert mapping["ledger_as_of"] == latest.isoformat().replace("+00:00", "Z")
    assert "generated_at" not in mapping
    assert mapping["group_by"] == list(GROUPING_DIMENSIONS)
    assert mapping["overall"]["sample_size"] == 2
    assert set(mapping["overall"]) == {
        field.name for field in fields(CalibrationMetrics)
    }
    assert mapping["groups"]
    assert all(
        list(group["dimensions"]) == list(GROUPING_DIMENSIONS)
        for group in mapping["groups"]
    )
    assert mapping["overall"]["priced_stake_count"] == 1


def test_empty_ledger_has_null_as_of_and_unavailable_metrics(tmp_path: Path) -> None:
    mapping = _mapping(tmp_path, ())

    assert mapping["ledger_as_of"] is None
    assert mapping["overall"]["sample_size"] == 0
    assert mapping["overall"]["roi"] is None
    assert mapping["overall"]["mean_probability_clv"] is None
    assert mapping["overall"]["mean_price_clv"] is None
    assert mapping["migration"]["confirmed_games"]["count"] == 0

    markdown = render_report_markdown(mapping)
    assert "사용 불가 (unavailable)" in markdown
    assert "ROI | 사용 불가 (unavailable)" in markdown
    assert "ROI | 0" not in markdown


def test_migration_counts_are_unique_confirmed_only_and_cover_all_market_types(
    tmp_path: Path,
) -> None:
    confirmed = _unique_row(1)
    duplicate_decision = replace(confirmed, capture_id="capture-duplicate")
    eligible_pass = _unique_row(
        2,
        market_type="full_total",
        market_key="game:910002:full_total:under",
        line=8.5,
        legacy_priority_score=60,
    )
    projected = _unique_row(3, lineup_state="projected")

    mapping = _mapping(
        tmp_path,
        (confirmed, duplicate_decision, eligible_pass, projected),
    )
    migration = mapping["migration"]

    assert migration["confirmed_games"]["count"] == 2
    assert migration["confirmed_games"]["threshold"] == (
        CONFIRMED_GAME_THRESHOLD
    )
    assert migration["market_types"]["ml"]["count"] == 1
    assert migration["market_types"]["full_total"]["count"] == 1
    assert tuple(migration["market_types"]) == tuple(
        market_type.value for market_type in MarketType
    )
    assert all(
        progress["threshold"] == DECISIONS_PER_MARKET_TYPE_THRESHOLD
        for progress in migration["market_types"].values()
    )


def test_migration_progress_is_deterministic_and_capped(tmp_path: Path) -> None:
    rows = tuple(
        _unique_row(
            index,
            game_pk=920000 + (index % 105),
            market_key=f"decision:{index}:ml",
        )
        for index in range(305)
    )

    migration = _mapping(tmp_path, rows)["migration"]
    games = migration["confirmed_games"]
    ml = migration["market_types"]["ml"]

    assert games["count"] == 105
    assert games["completed"] == 100
    assert games["remaining"] == 0
    assert games["progress"] == 1.0
    assert games["threshold_met"] is True
    assert ml["count"] == 305
    assert ml["completed"] == 300
    assert ml["remaining"] == 0
    assert ml["progress"] == 1.0
    assert ml["threshold_met"] is True


def test_json_and_markdown_are_byte_identical_for_unchanged_mapping(
    tmp_path: Path,
) -> None:
    mapping = _mapping(tmp_path, (_unique_row(1),))
    report_dir = tmp_path / "reports"

    first = write_calibration_reports(mapping, report_dir=report_dir)
    first_json = first.json_path.read_bytes()
    first_markdown = first.markdown_path.read_bytes()
    second = write_calibration_reports(mapping, report_dir=report_dir)

    assert first.json_path == report_dir.resolve() / CALIBRATION_JSON_FILENAME
    assert first.markdown_path == (
        report_dir.resolve() / CALIBRATION_MARKDOWN_FILENAME
    )
    assert second.json_path.read_bytes() == first_json
    assert second.markdown_path.read_bytes() == first_markdown
    assert first_json == (canonical_json(mapping) + "\n").encode("utf-8")
    assert first_json.endswith(b"\n")
    assert first_markdown.endswith(b"\n")
    assert json.loads(first_json) == mapping
    assert "generated_at" not in first_json.decode("utf-8")


def test_atomic_writer_fsyncs_files_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, (_unique_row(1),))
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr("analysis.operator.reporting.os.fsync", recording_fsync)

    write_calibration_reports(mapping, report_dir=tmp_path / "reports")

    assert len(calls) == 4


@pytest.mark.parametrize("failure", ("replace", "fsync"))
def test_atomic_write_failures_map_to_operator_io_error_and_clean_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    mapping = _mapping(tmp_path, (_unique_row(1),))
    report_dir = tmp_path / "reports"

    def fail(*args, **kwargs):
        raise OSError(f"{failure} failed")

    monkeypatch.setattr(f"analysis.operator.reporting.os.{failure}", fail)

    with pytest.raises(OperatorIOError, match=failure):
        write_calibration_reports(mapping, report_dir=report_dir)

    assert not tuple(report_dir.glob(".*.tmp"))


def test_report_writer_rejects_symlinked_directory_and_destination(
    tmp_path: Path,
) -> None:
    mapping = _mapping(tmp_path, (_unique_row(1),))
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(OperatorIOError, match="symbolic link"):
        write_calibration_reports(mapping, report_dir=linked_dir)

    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("outside", encoding="utf-8")
    (destination_dir / CALIBRATION_JSON_FILENAME).symlink_to(target)

    with pytest.raises(OperatorIOError, match="symbolic link"):
        write_calibration_reports(mapping, report_dir=destination_dir)
    assert target.read_text(encoding="utf-8") == "outside"


def test_run_review_rebuilds_full_ledger_and_returns_report_counts_and_paths(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger" / "shadow.jsonl"
    bundle_root = tmp_path / "bundles"
    report_dir = tmp_path / "custom-reports"

    first = _run_review_for_game(
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        report_dir=report_dir,
        game_pk=930001,
    )
    first_json = first.calibration_json_path.read_bytes()
    replay = _run_review_for_game(
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        report_dir=report_dir,
        game_pk=930001,
    )

    assert replay.calibration_json_path.read_bytes() == first_json
    assert replay.calibration_markdown_path.read_bytes() == (
        first.calibration_markdown_path.read_bytes()
    )
    assert first.calibration_sample_size == 2
    assert first.confirmed_game_count == 1
    assert first.confirmed_decision_count == 2
    assert first.confirmed_decision_counts["ml"] == 1
    assert first.confirmed_decision_counts["full_total"] == 1

    second = _run_review_for_game(
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        report_dir=report_dir,
        game_pk=930002,
        day_offset=1,
    )
    payload = json.loads(second.calibration_json_path.read_bytes())

    assert second.calibration_sample_size == 4
    assert second.confirmed_game_count == 2
    assert second.confirmed_decision_count == 4
    assert second.confirmed_decision_counts["ml"] == 2
    assert second.confirmed_decision_counts["full_total"] == 2
    assert payload["overall"]["sample_size"] == 4
    assert payload["migration"]["confirmed_games"]["count"] == 2
    assert payload["migration"]["market_types"]["ml"]["count"] == 2
    assert payload["migration"]["market_types"]["full_total"]["count"] == 2
    assert sum(second.classification_counts.values()) == 2
    assert len(tuple(LedgerStore(ledger_path).iter_records("review"))) == 4


def test_service_builds_rows_and_ledger_as_of_from_one_immutable_snapshot(
    tmp_path: Path,
) -> None:
    appended_after_selection = _complete_game().captured_at

    class ChangingStore:
        def __init__(self) -> None:
            self.calls = 0

        def iter_records(self):
            self.calls += 1
            if self.calls == 1:
                return iter(())
            return iter((_ledger_record(appended_after_selection),))

    store = ChangingStore()

    mapping, sample_size, game_count, decision_counts = (
        service_module._calibration_report_state(
            store,
            tmp_path / "ledger.jsonl",
        )
    )

    assert store.calls == 1
    assert mapping["ledger_as_of"] is None
    assert sample_size == 0
    assert game_count == 0
    assert sum(decision_counts.values()) == 0
