import copy
from dataclasses import fields, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import analysis.operator.reporting as reporting_module
import analysis.operator.service as service_module
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType
from analysis.ledger import (
    CaptureForecastLink,
    ForecastCommand,
    LedgerStore,
    RecordEnvelope,
    ReviewCommand,
)
from analysis.operator.errors import OperatorIOError, OperatorLedgerError
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
    select_primary_rows,
)
from analysis.shadow import deserialize_shadow_comparison
from tests.test_calibration import calibration_row
from tests.test_forecasting_engine import _complete_game
from tests.test_operator_review import (
    _legacy_snapshot,
    _raw,
    _record_pregame,
    _review_mapping,
)


def _ledger_record(appended_at, *, sequence: int = 0, record_kind: str = "review"):
    identity = hashlib.sha256(
        f"{record_kind}:{sequence}:{appended_at.isoformat()}".encode("utf-8")
    ).hexdigest()
    return SimpleNamespace(
        envelope=SimpleNamespace(
            record_kind=record_kind,
            record_id=identity[:32],
            idempotency_key=f"report-test:{identity[:16]}",
            payload_hash=identity,
            appended_at=appended_at,
        ),
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


def _frozen_forecast(tmp_path: Path):
    confirmed = _record_pregame(
        _complete_game(),
        _legacy_snapshot(),
        tmp_path / "frozen-comparison-ledger.jsonl",
    )
    return confirmed.forecast


def _forecast_with_ineligible_market(tmp_path: Path):
    forecast = _frozen_forecast(tmp_path)
    outputs = json.loads(canonical_json(forecast.outputs))
    shadow = outputs["shadow_comparison"]
    ineligible = next(
        comparison
        for comparison in shadow["market_comparisons"]
        if comparison["market_key"] == "sea_ml"
    )
    ineligible["principal_paths"] = []
    ineligible["review_eligible"] = False
    ineligible["review_ineligibility_flags"] = [
        "missing_component_distribution"
    ]
    frozen = replace(forecast, outputs=outputs)
    comparison = deserialize_shadow_comparison(
        frozen.outputs["shadow_comparison"]
    )
    assert comparison.forecast_run_id == frozen.forecast_run_id
    stored_ineligible = next(
        market
        for market in comparison.market_comparisons
        if market.market_key == "sea_ml"
    )
    assert stored_ineligible.review_eligible is False
    return frozen


def _clone_eligible_forecast_for_row(forecast, row, index: int):
    run_id = hashlib.sha256(f"migration-forecast-{index}".encode()).hexdigest()
    market_key = row.market_key
    shadow = json.loads(canonical_json(forecast.outputs["shadow_comparison"]))
    shadow["forecast_run_id"] = run_id
    comparison = next(
        market
        for market in shadow["market_comparisons"]
        if market["legacy_candidate_index"] == 0
    )
    comparison["market_key"] = market_key
    binding = next(
        item
        for item in shadow["legacy"]["review_bindings"]
        if item["legacy_candidate_index"] == 0
    )
    binding["market_key"] = market_key
    cloned = replace(
        forecast,
        envelope=replace(
            forecast.envelope,
            record_id=hashlib.sha256(f"migration-envelope-{index}".encode())
            .hexdigest()[:32],
            idempotency_key=f"migration-forecast-{index}",
            payload_hash=f"migration-payload-{index}",
        ),
        forecast_run_id=run_id,
        normalized_input_hash=f"migration-input-{index}",
        outputs={"shadow_comparison": shadow},
    )
    deserialize_shadow_comparison(cloned.outputs["shadow_comparison"])
    return replace(row, forecast_run_id=run_id), cloned


def _exact_migration_row(
    forecast,
    *,
    game_pk: int,
    market_key: str,
    market_type: str,
    legacy_priority_score: int,
):
    return calibration_row(
        game_pk=game_pk,
        capture_id=f"capture-{game_pk}-{market_key}",
        forecast_run_id=forecast.forecast_run_id,
        market_key=market_key,
        market_type=market_type,
        legacy_priority_score=legacy_priority_score,
    )


def _mapping_from_frozen_forecast(
    tmp_path: Path,
    forecast,
    rows,
):
    rows = tuple(rows)
    return build_report_mapping(
        ledger_path=tmp_path / "ledger.jsonl",
        ledger_records=(forecast,),
        rows=rows,
        report=build_calibration_report(rows),
    )


def test_eligible_legacy_pass_counts_for_migration(tmp_path: Path) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    comparison = deserialize_shadow_comparison(
        forecast.outputs["shadow_comparison"]
    )
    frozen_pass = next(
        market
        for market in comparison.market_comparisons
        if market.market_key == "full_under"
    )
    assert frozen_pass.legacy_rating == "pass"
    assert frozen_pass.review_eligible is True
    eligible_pass = _exact_migration_row(
        forecast,
        game_pk=940001,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )

    mapping = _mapping_from_frozen_forecast(
        tmp_path,
        forecast,
        (eligible_pass,),
    )

    assert mapping["overall"]["sample_size"] == 1
    assert mapping["migration"]["confirmed_games"]["count"] == 1
    assert mapping["migration"]["market_types"]["full_total"]["count"] == 1


def test_ineligible_exact_reviewed_market_does_not_count_for_migration(
    tmp_path: Path,
) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    eligible_pass = _exact_migration_row(
        forecast,
        game_pk=940001,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )
    ineligible = _exact_migration_row(
        forecast,
        game_pk=940002,
        market_key="sea_ml",
        market_type="ml",
        legacy_priority_score=78,
    )

    mapping = _mapping_from_frozen_forecast(
        tmp_path,
        forecast,
        (eligible_pass, ineligible),
    )

    assert mapping["overall"]["sample_size"] == 2
    assert mapping["migration"]["market_types"]["full_total"]["count"] == 1
    assert mapping["migration"]["market_types"]["ml"]["count"] == 0


def test_migration_uses_latest_corrected_link_comparison(
    tmp_path: Path,
) -> None:
    forecast = _frozen_forecast(tmp_path)
    outputs = json.loads(canonical_json(forecast.outputs))
    corrected_shadow = outputs["shadow_comparison"]
    corrected_market = next(
        market
        for market in corrected_shadow["market_comparisons"]
        if market["market_key"] == "full_under"
    )
    corrected_market["principal_paths"] = []
    corrected_market["review_eligible"] = False
    corrected_market["review_ineligibility_flags"] = [
        "missing_component_distribution"
    ]
    deserialize_shadow_comparison(corrected_shadow)
    capture_id = "a" * 32
    first_link = CaptureForecastLink(
        envelope=RecordEnvelope(
            record_id="b" * 32,
            record_kind="capture_forecast_link",
            idempotency_key="report-link:original",
            payload_hash="c" * 64,
            appended_at=forecast.envelope.appended_at,
        ),
        capture_id=capture_id,
        forecast_run_id=forecast.forecast_run_id,
    )
    corrected_link = CaptureForecastLink(
        envelope=RecordEnvelope(
            record_id="d" * 32,
            record_kind="capture_forecast_link",
            idempotency_key="report-link:corrected",
            payload_hash="e" * 64,
            appended_at=forecast.envelope.appended_at,
        ),
        capture_id=capture_id,
        forecast_run_id=forecast.forecast_run_id,
        shadow_comparison=corrected_shadow,
        supersedes_link_record_id=first_link.envelope.record_id,
    )
    row = _exact_migration_row(
        forecast,
        game_pk=940005,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )
    row = replace(row, capture_id=capture_id)

    mapping = build_report_mapping(
        ledger_path=tmp_path / "ledger.jsonl",
        ledger_records=(forecast, first_link, corrected_link),
        rows=(row,),
        report=build_calibration_report((row,)),
    )

    assert mapping["overall"]["sample_size"] == 1
    assert mapping["migration"]["confirmed_games"]["count"] == 0
    assert mapping["migration"]["market_types"]["full_total"]["count"] == 0


def test_game_with_only_ineligible_row_does_not_count_as_confirmed_game(
    tmp_path: Path,
) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    ineligible = _exact_migration_row(
        forecast,
        game_pk=940003,
        market_key="sea_ml",
        market_type="ml",
        legacy_priority_score=78,
    )

    mapping = _mapping_from_frozen_forecast(
        tmp_path,
        forecast,
        (ineligible,),
    )

    assert mapping["overall"]["sample_size"] == 1
    assert mapping["migration"]["confirmed_games"]["count"] == 0


def test_duplicate_eligible_rows_remain_one_migration_decision(
    tmp_path: Path,
) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    eligible_pass = _exact_migration_row(
        forecast,
        game_pk=940004,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )
    duplicate = replace(eligible_pass, capture_id="duplicate-capture")

    mapping = _mapping_from_frozen_forecast(
        tmp_path,
        forecast,
        (eligible_pass, duplicate),
    )

    assert mapping["overall"]["sample_size"] == 2
    assert mapping["migration"]["confirmed_games"]["count"] == 1
    assert mapping["migration"]["market_types"]["full_total"]["count"] == 1


def test_historical_ineligible_review_stays_in_metrics_but_not_migration(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "historical-ledger.jsonl"
    confirmed = _record_pregame(
        _complete_game(),
        _legacy_snapshot(),
        ledger_path,
    )
    store = LedgerStore(ledger_path)
    forecast_run_id = hashlib.sha256(b"historical-ineligible").hexdigest()
    outputs = json.loads(canonical_json(confirmed.forecast.outputs))
    outputs["forecast_run_id"] = forecast_run_id
    shadow = outputs["shadow_comparison"]
    shadow["forecast_run_id"] = forecast_run_id
    ineligible = next(
        comparison
        for comparison in shadow["market_comparisons"]
        if comparison["market_key"] == "sea_ml"
    )
    ineligible["principal_paths"] = []
    ineligible["review_eligible"] = False
    ineligible["review_ineligibility_flags"] = [
        "missing_component_distribution"
    ]
    deserialize_shadow_comparison(shadow)
    historical_forecast = store.append_forecast(
        ForecastCommand(
            forecast_run_id=forecast_run_id,
            model_version=confirmed.forecast.model_version,
            simulation_config_hash=(
                confirmed.forecast.simulation_config_hash
            ),
            normalized_input_hash=confirmed.capture.normalized_input_hash,
            outputs=outputs,
        )
    )
    historical_link = store.link_capture_forecast(
        confirmed.capture.capture_id,
        historical_forecast.forecast_run_id,
    )
    bundle = parse_review_bundle_json(_raw(_review_mapping(confirmed)))
    result = store.append_official_result(bundle.payload.official_result)
    store.append_review(
        ReviewCommand(
            capture_forecast_link_record_id=(
                historical_link.envelope.record_id
            ),
            capture_id=confirmed.capture.capture_id,
            forecast_run_id=historical_forecast.forecast_run_id,
            official_result_record_id=result.envelope.record_id,
            market_key="sea_ml",
            market_result="win",
            classification="A",
            inside_80_ratio=1.0,
            contradicted_paths=(),
            component_reviews=(
                {
                    "evidence_path": "starter_suppression",
                    "component_ids": (
                        "home_pitcher_chain_runs_allowed",
                    ),
                    "contradicted": False,
                    "inside_80_count": 1,
                    "component_count": 1,
                    "notes": (),
                },
                {
                    "evidence_path": "top_order_matchup",
                    "component_ids": (
                        "away_full_runs",
                        "away_full_traffic",
                        "away_full_extra_base_hits",
                    ),
                    "contradicted": False,
                    "inside_80_count": 3,
                    "component_count": 3,
                    "notes": (),
                },
            ),
        )
    )

    rows = select_primary_rows(store)
    mapping = build_report_mapping(
        ledger_path=ledger_path,
        ledger_records=tuple(store.iter_records()),
        rows=rows,
        report=build_calibration_report(rows),
    )

    assert tuple(row.market_key for row in rows) == ("sea_ml",)
    assert mapping["overall"]["sample_size"] == 1
    assert mapping["migration"]["confirmed_games"]["count"] == 0
    assert mapping["migration"]["market_types"]["ml"]["count"] == 0


@pytest.mark.parametrize(
    "comparison_state",
    ("missing_forecast", "missing_market", "invalid"),
)
def test_migration_rows_require_valid_exact_frozen_comparison(
    tmp_path: Path,
    comparison_state: str,
) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    row = _exact_migration_row(
        forecast,
        game_pk=940005,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )
    if comparison_state == "missing_forecast":
        records = (_ledger_record(forecast.envelope.appended_at),)
    elif comparison_state == "missing_market":
        records = (forecast,)
        row = replace(row, market_key="missing_exact_market")
    else:
        outputs = json.loads(canonical_json(forecast.outputs))
        outputs.pop("shadow_comparison")
        records = (replace(forecast, outputs=outputs),)

    with pytest.raises(ValueError, match="frozen.*comparison|forecast"):
        build_report_mapping(
            ledger_path=tmp_path / "ledger.jsonl",
            ledger_records=records,
            rows=(row,),
            report=build_calibration_report((row,)),
        )


def test_service_maps_invalid_migration_comparison_to_ledger_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecast = _forecast_with_ineligible_market(tmp_path)
    outputs = json.loads(canonical_json(forecast.outputs))
    outputs.pop("shadow_comparison")
    invalid_forecast = replace(forecast, outputs=outputs)
    row = _exact_migration_row(
        forecast,
        game_pk=940006,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )

    class SnapshotStore:
        def iter_records(self):
            return iter((invalid_forecast,))

    monkeypatch.setattr(
        service_module,
        "select_primary_rows",
        lambda snapshot: (row,),
    )

    with pytest.raises(
        OperatorLedgerError,
        match="invalid frozen shadow comparison",
    ):
        service_module._calibration_report_state(
            SnapshotStore(),
            tmp_path / "ledger.jsonl",
        )


def test_report_mapping_is_versioned_complete_and_uses_latest_ledger_append(
    tmp_path: Path,
) -> None:
    first = _unique_row(1, lineup_state="projected")
    second = _unique_row(
        2,
        lineup_state="projected",
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
    assert mapping["ledger_revision"]["record_count"] == 2
    assert len(mapping["ledger_revision"]["state_hash"]) == 64
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


def test_ledger_revision_hashes_ordered_immutable_envelope_identity(
    tmp_path: Path,
) -> None:
    appended_at = _complete_game().captured_at
    first = _ledger_record(appended_at, sequence=1)
    second = _ledger_record(appended_at, sequence=2)

    forward = _mapping(
        tmp_path,
        (),
        ledger_records=(first, second),
    )["ledger_revision"]
    replay = _mapping(
        tmp_path,
        (),
        ledger_records=(first, second),
    )["ledger_revision"]
    reversed_revision = _mapping(
        tmp_path,
        (),
        ledger_records=(second, first),
    )["ledger_revision"]

    assert forward == replay
    assert forward["record_count"] == 2
    assert forward["state_hash"] != reversed_revision["state_hash"]
    expected_identities = [
        {
            "record_kind": record.envelope.record_kind,
            "record_id": record.envelope.record_id,
            "idempotency_key": record.envelope.idempotency_key,
            "payload_hash": record.envelope.payload_hash,
            "appended_at": record.envelope.appended_at.isoformat().replace(
                "+00:00",
                "Z",
            ),
        }
        for record in (first, second)
    ]
    assert forward["state_hash"] == hashlib.sha256(
        canonical_json(expected_identities).encode("utf-8")
    ).hexdigest()


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
    forecast = _frozen_forecast(tmp_path)
    confirmed = _exact_migration_row(
        forecast,
        game_pk=910001,
        market_key="sea_ml",
        market_type="ml",
        legacy_priority_score=78,
    )
    duplicate_decision = replace(confirmed, capture_id="capture-duplicate")
    eligible_pass = _exact_migration_row(
        forecast,
        game_pk=910002,
        market_key="full_under",
        market_type="full_total",
        legacy_priority_score=60,
    )
    projected = _unique_row(3, lineup_state="projected")

    mapping = _mapping(
        tmp_path,
        (confirmed, duplicate_decision, eligible_pass, projected),
        ledger_records=(forecast,),
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
    forecast = _frozen_forecast(tmp_path)
    cloned = tuple(
        _clone_eligible_forecast_for_row(
            forecast,
            _unique_row(
                index,
                game_pk=920000 + (index % 105),
                market_key=f"decision:{index}:ml",
            ),
            index,
        )
        for index in range(305)
    )
    rows = tuple(row for row, _ in cloned)
    forecasts = tuple(record for _, record in cloned)

    migration = _mapping(
        tmp_path,
        rows,
        ledger_records=forecasts,
    )["migration"]
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
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
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
    assert first.generation_dir == second.generation_dir
    assert first.replayed is False
    assert second.replayed is True
    assert first.generation_json_path.read_bytes() == first_json
    assert first.generation_markdown_path.read_bytes() == first_markdown
    assert first_json == (canonical_json(mapping) + "\n").encode("utf-8")
    assert first_json.endswith(b"\n")
    assert first_markdown.endswith(b"\n")
    assert json.loads(first_json) == mapping
    assert "generated_at" not in first_json.decode("utf-8")


def test_one_pointer_swap_publishes_consistent_pair_and_compatibility_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
        ledger_records=(_ledger_record(appended_at),),
    )
    report_dir = tmp_path / "reports"
    pointer_swaps: list[Path] = []
    real_replace = os.replace

    def recording_replace(source, destination, **kwargs):
        if Path(destination).name == "current":
            assert (report_dir / CALIBRATION_JSON_FILENAME).is_symlink()
            assert (report_dir / CALIBRATION_MARKDOWN_FILENAME).is_symlink()
            pointer_swaps.append(Path(destination))
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(
        "analysis.operator.reporting.os.replace",
        recording_replace,
    )

    artifacts = write_calibration_reports(mapping, report_dir=report_dir)

    assert len(pointer_swaps) == 1
    assert artifacts.generation_dir.is_dir()
    assert artifacts.current_generation_path.is_symlink()
    assert artifacts.current_generation_path.resolve() == artifacts.generation_dir
    assert artifacts.json_path.is_symlink()
    assert artifacts.markdown_path.is_symlink()
    assert artifacts.json_path.resolve() == artifacts.generation_json_path
    assert artifacts.markdown_path.resolve() == artifacts.generation_markdown_path
    payload = json.loads(artifacts.generation_json_path.read_bytes())
    revision = payload["ledger_revision"]
    markdown = artifacts.generation_markdown_path.read_text(encoding="utf-8")
    assert str(revision["record_count"]) in markdown
    assert revision["state_hash"] in markdown


def test_partial_legacy_compatibility_conversion_never_exposes_mixed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy_json = b'{"legacy":"json"}\n'
    legacy_markdown = b"# legacy markdown\n"
    json_path = report_dir / CALIBRATION_JSON_FILENAME
    markdown_path = report_dir / CALIBRATION_MARKDOWN_FILENAME
    json_path.write_bytes(legacy_json)
    markdown_path.write_bytes(legacy_markdown)
    real_replace = os.replace
    failed = False

    def fail_markdown_conversion_once(source, destination, **kwargs):
        nonlocal failed
        if (
            Path(destination).name == markdown_path.name
            and not failed
        ):
            failed = True
            raise OSError("legacy markdown conversion failed")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(
        "analysis.operator.reporting.os.replace",
        fail_markdown_conversion_once,
    )

    with pytest.raises(OperatorIOError, match="conversion failed"):
        write_calibration_reports(mapping, report_dir=report_dir)

    current_generation = (report_dir / "current").resolve()
    assert json_path.is_symlink()
    assert markdown_path.is_file() and not markdown_path.is_symlink()
    assert json_path.read_bytes() == legacy_json
    assert markdown_path.read_bytes() == legacy_markdown
    assert (
        current_generation / CALIBRATION_JSON_FILENAME
    ).read_bytes() == legacy_json
    assert (
        current_generation / CALIBRATION_MARKDOWN_FILENAME
    ).read_bytes() == legacy_markdown

    artifacts = write_calibration_reports(mapping, report_dir=report_dir)

    assert artifacts.json_path.is_symlink()
    assert artifacts.markdown_path.is_symlink()
    assert artifacts.json_path.resolve() == artifacts.generation_json_path
    assert artifacts.markdown_path.resolve() == artifacts.generation_markdown_path
    assert artifacts.json_path.read_bytes() == (
        canonical_json(mapping) + "\n"
    ).encode("utf-8")


def test_equal_revision_rejects_divergent_json_bytes(tmp_path: Path) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(mapping, report_dir=report_dir)
    published_json = published.generation_json_path.read_bytes()
    published_markdown = published.generation_markdown_path.read_bytes()
    divergent = copy.deepcopy(mapping)
    divergent["unrendered_conflict"] = "different JSON bytes"
    assert render_report_markdown(divergent) == render_report_markdown(mapping)

    with pytest.raises(OperatorLedgerError, match="replay content conflict"):
        write_calibration_reports(divergent, report_dir=report_dir)

    assert published.current_generation_path.resolve() == published.generation_dir
    assert published.generation_json_path.read_bytes() == published_json
    assert published.generation_markdown_path.read_bytes() == published_markdown


def test_equal_revision_rejects_divergent_markdown_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(mapping, report_dir=report_dir)
    published_json = published.generation_json_path.read_bytes()
    published_markdown = published.generation_markdown_path.read_bytes()
    original_render = reporting_module.render_report_markdown

    def divergent_markdown(value):
        return original_render(value) + "\n충돌"

    monkeypatch.setattr(
        reporting_module,
        "render_report_markdown",
        divergent_markdown,
    )

    with pytest.raises(OperatorLedgerError, match="replay content conflict"):
        write_calibration_reports(mapping, report_dir=report_dir)

    assert published.current_generation_path.resolve() == published.generation_dir
    assert published.generation_json_path.read_bytes() == published_json
    assert published.generation_markdown_path.read_bytes() == published_markdown


def test_stale_writer_cannot_regress_current_generation(tmp_path: Path) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(
        appended_at + timedelta(seconds=1),
        sequence=2,
    )
    older = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
        ledger_records=(first_record,),
    )
    newer = _mapping(
        tmp_path,
        (
            _unique_row(1, lineup_state="projected"),
            _unique_row(2, lineup_state="projected"),
        ),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(newer, report_dir=report_dir)
    published_json = published.json_path.read_bytes()
    published_markdown = published.markdown_path.read_bytes()

    with pytest.raises(reporting_module._StaleCalibrationPublicationError):
        write_calibration_reports(older, report_dir=report_dir)

    assert published.current_generation_path.resolve() == published.generation_dir
    assert published.json_path.read_bytes() == published_json
    assert published.markdown_path.read_bytes() == published_markdown


def test_equal_count_different_revision_is_ledger_conflict(tmp_path: Path) -> None:
    appended_at = _complete_game().captured_at
    first = _mapping(
        tmp_path,
        (),
        ledger_records=(_ledger_record(appended_at, sequence=1),),
    )
    conflict = _mapping(
        tmp_path,
        (),
        ledger_records=(_ledger_record(appended_at, sequence=2),),
    )
    report_dir = tmp_path / "reports"
    published = write_calibration_reports(first, report_dir=report_dir)

    with pytest.raises(OperatorLedgerError, match="revision conflict"):
        write_calibration_reports(conflict, report_dir=report_dir)

    assert json.loads(published.json_path.read_bytes()) == first


def test_pointer_swap_failure_leaves_old_pair_current_and_cleans_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(
        appended_at + timedelta(seconds=1),
        sequence=2,
    )
    old_mapping = _mapping(tmp_path, (), ledger_records=(first_record,))
    new_mapping = _mapping(
        tmp_path,
        (),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    old = write_calibration_reports(old_mapping, report_dir=report_dir)
    old_json = old.json_path.read_bytes()
    old_markdown = old.markdown_path.read_bytes()
    real_replace = os.replace

    def fail_pointer_swap(source, destination, **kwargs):
        if Path(destination).name == "current":
            raise OSError("current pointer swap failed")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(
        "analysis.operator.reporting.os.replace",
        fail_pointer_swap,
    )

    with pytest.raises(OperatorIOError, match="pointer swap failed"):
        write_calibration_reports(new_mapping, report_dir=report_dir)

    assert old.current_generation_path.resolve() == old.generation_dir
    assert old.json_path.read_bytes() == old_json
    assert old.markdown_path.read_bytes() == old_markdown
    assert {path.name for path in (report_dir / "generations").iterdir()} == {
        old.generation_dir.name
    }
    assert not tuple(report_dir.rglob("*.tmp"))


def test_post_commit_interrupt_preserves_normal_generation_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    report_dir = tmp_path / "reports"
    real_publish = reporting_module._publish_current_pointer

    def publish_then_interrupt(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after current commit")

    monkeypatch.setattr(
        reporting_module,
        "_publish_current_pointer",
        publish_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="after current commit"):
        write_calibration_reports(mapping, report_dir=report_dir)

    current_path = report_dir / "current"
    generation_dir = report_dir / os.readlink(current_path)
    assert generation_dir.is_dir()
    assert (generation_dir / CALIBRATION_JSON_FILENAME).read_bytes() == (
        canonical_json(mapping) + "\n"
    ).encode("utf-8")
    assert (generation_dir / CALIBRATION_MARKDOWN_FILENAME).is_file()
    assert (report_dir / CALIBRATION_JSON_FILENAME).resolve() == (
        generation_dir / CALIBRATION_JSON_FILENAME
    )
    assert (report_dir / CALIBRATION_MARKDOWN_FILENAME).resolve() == (
        generation_dir / CALIBRATION_MARKDOWN_FILENAME
    )

    monkeypatch.setattr(
        reporting_module,
        "_publish_current_pointer",
        real_publish,
    )
    replay = write_calibration_reports(mapping, report_dir=report_dir)

    assert replay.replayed is True
    assert replay.generation_dir == generation_dir


def test_interrupt_after_pointer_durability_preserves_committed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(tmp_path, ())
    report_dir = tmp_path / "reports"
    real_verify = reporting_module._verify_report_directories
    verification_count = 0

    def verify_then_interrupt(*args, **kwargs):
        nonlocal verification_count
        result = real_verify(*args, **kwargs)
        verification_count += 1
        if verification_count == 4:
            raise KeyboardInterrupt("interrupted after durable pointer")
        return result

    monkeypatch.setattr(
        reporting_module,
        "_verify_report_directories",
        verify_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="after durable pointer"):
        write_calibration_reports(mapping, report_dir=report_dir)

    current_path = report_dir / "current"
    generation_dir = report_dir / os.readlink(current_path)
    assert generation_dir.is_dir()
    assert (generation_dir / CALIBRATION_JSON_FILENAME).is_file()
    assert (generation_dir / CALIBRATION_MARKDOWN_FILENAME).is_file()

    monkeypatch.setattr(
        reporting_module,
        "_verify_report_directories",
        real_verify,
    )
    replay = write_calibration_reports(mapping, report_dir=report_dir)

    assert replay.replayed is True
    assert replay.generation_dir == generation_dir


def test_post_commit_interrupt_preserves_legacy_migration_and_rerun_heals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy_json = b'{"legacy":"json"}\n'
    legacy_markdown = b"# legacy markdown\n"
    json_path = report_dir / CALIBRATION_JSON_FILENAME
    markdown_path = report_dir / CALIBRATION_MARKDOWN_FILENAME
    json_path.write_bytes(legacy_json)
    markdown_path.write_bytes(legacy_markdown)
    real_publish = reporting_module._publish_current_pointer

    def publish_then_interrupt(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after migration commit")

    monkeypatch.setattr(
        reporting_module,
        "_publish_current_pointer",
        publish_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="after migration commit"):
        write_calibration_reports(mapping, report_dir=report_dir)

    migration_dir = report_dir / os.readlink(report_dir / "current")
    assert migration_dir.is_dir()
    assert (migration_dir / CALIBRATION_JSON_FILENAME).read_bytes() == legacy_json
    assert (
        migration_dir / CALIBRATION_MARKDOWN_FILENAME
    ).read_bytes() == legacy_markdown
    assert json_path.read_bytes() == legacy_json
    assert markdown_path.read_bytes() == legacy_markdown

    monkeypatch.setattr(
        reporting_module,
        "_publish_current_pointer",
        real_publish,
    )
    healed = write_calibration_reports(mapping, report_dir=report_dir)
    replay = write_calibration_reports(mapping, report_dir=report_dir)

    assert healed.replayed is False
    assert replay.replayed is True
    assert replay.generation_dir == healed.generation_dir
    assert json_path.is_symlink()
    assert markdown_path.is_symlink()
    assert json_path.resolve() == healed.generation_json_path
    assert markdown_path.resolve() == healed.generation_markdown_path


def test_pre_commit_interrupt_keeps_old_current_and_removes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(
        appended_at + timedelta(seconds=1),
        sequence=2,
    )
    old_mapping = _mapping(tmp_path, (), ledger_records=(first_record,))
    new_mapping = _mapping(
        tmp_path,
        (),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    old = write_calibration_reports(old_mapping, report_dir=report_dir)
    old_target = os.readlink(old.current_generation_path)
    old_generation_names = {
        path.name for path in (report_dir / "generations").iterdir()
    }

    def interrupt_before_publish(*args, **kwargs):
        raise KeyboardInterrupt("interrupted before current commit")

    monkeypatch.setattr(
        reporting_module,
        "_publish_current_pointer",
        interrupt_before_publish,
    )

    with pytest.raises(KeyboardInterrupt, match="before current commit"):
        write_calibration_reports(new_mapping, report_dir=report_dir)

    assert os.readlink(old.current_generation_path) == old_target
    assert old.current_generation_path.resolve() == old.generation_dir
    assert {
        path.name for path in (report_dir / "generations").iterdir()
    } == old_generation_names


def test_generation_construction_failure_leaves_old_pair_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(
        appended_at + timedelta(seconds=1),
        sequence=2,
    )
    old_mapping = _mapping(tmp_path, (), ledger_records=(first_record,))
    new_mapping = _mapping(
        tmp_path,
        (),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    old = write_calibration_reports(old_mapping, report_dir=report_dir)
    old_json = old.json_path.read_bytes()
    old_markdown = old.markdown_path.read_bytes()
    real_write = reporting_module._write_generation_file
    calls = 0

    def fail_second_generation_file(descriptor, filename, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("generation file write failed")
        return real_write(descriptor, filename, content)

    monkeypatch.setattr(
        reporting_module,
        "_write_generation_file",
        fail_second_generation_file,
    )

    with pytest.raises(OperatorIOError, match="generation file write failed"):
        write_calibration_reports(new_mapping, report_dir=report_dir)

    assert old.current_generation_path.resolve() == old.generation_dir
    assert old.json_path.read_bytes() == old_json
    assert old.markdown_path.read_bytes() == old_markdown
    assert {path.name for path in (report_dir / "generations").iterdir()} == {
        old.generation_dir.name
    }
    assert not tuple(report_dir.rglob("*.tmp"))


def test_atomic_writer_fsyncs_files_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
    modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr("analysis.operator.reporting.os.fsync", recording_fsync)

    write_calibration_reports(mapping, report_dir=tmp_path / "reports")

    assert sum(stat.S_ISREG(mode) for mode in modes) == 2
    assert sum(stat.S_ISDIR(mode) for mode in modes) >= 3


@pytest.mark.parametrize("failure", ("replace", "fsync"))
def test_atomic_write_failures_map_to_operator_io_error_and_clean_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
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
    mapping = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
    )
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


def test_run_review_rebuilds_and_retries_after_stale_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended_at = _complete_game().captured_at
    first_record = _ledger_record(appended_at, sequence=1)
    second_record = _ledger_record(
        appended_at + timedelta(seconds=1),
        sequence=2,
    )
    stale = _mapping(
        tmp_path,
        (_unique_row(1, lineup_state="projected"),),
        ledger_records=(first_record,),
    )
    current = _mapping(
        tmp_path,
        (
            _unique_row(1, lineup_state="projected"),
            _unique_row(2, lineup_state="projected"),
        ),
        ledger_records=(first_record, second_record),
    )
    report_dir = tmp_path / "reports"
    write_calibration_reports(current, report_dir=report_dir)
    states = iter(
        (
            (stale, 1, 0, {market_type.value: 0 for market_type in MarketType}),
            (current, 2, 0, {market_type.value: 0 for market_type in MarketType}),
        )
    )
    calls = 0

    def changing_report_state(store, ledger_path):
        nonlocal calls
        calls += 1
        return next(states)

    monkeypatch.setattr(
        service_module,
        "_calibration_report_state",
        changing_report_state,
    )

    result = _run_review_for_game(
        ledger_path=tmp_path / "ledger.jsonl",
        bundle_root=tmp_path / "bundles",
        report_dir=report_dir,
        game_pk=930101,
    )

    assert calls == 2
    assert result.calibration_sample_size == 2
    assert result.calibration_generation_dir.resolve() == (
        result.calibration_current_generation_path.resolve()
    )
    assert json.loads(result.calibration_json_path.read_bytes()) == current


def test_report_publication_stale_retry_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def report_state(store, ledger_path):
        nonlocal calls
        calls += 1
        return ({}, 0, 0, {})

    def always_stale(mapping, *, report_dir):
        raise reporting_module._StaleCalibrationPublicationError("stale")

    monkeypatch.setattr(service_module, "_calibration_report_state", report_state)
    monkeypatch.setattr(service_module, "write_calibration_reports", always_stale)

    with pytest.raises(OperatorLedgerError, match="rebuilding the current ledger"):
        service_module._publish_calibration_report(
            object(),
            tmp_path / "ledger.jsonl",
            tmp_path / "reports",
        )

    assert calls == service_module._REPORT_PUBLICATION_MAX_ATTEMPTS == 3


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
