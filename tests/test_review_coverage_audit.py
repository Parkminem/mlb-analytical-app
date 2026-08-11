from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import analysis.operator.reporting as reporting_module
import analysis.review.coverage_audit as coverage_audit_module
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType
from analysis.ledger import (
    CaptureCommand,
    ForecastCommand,
    LedgerCorruptionError,
    LedgerStore,
    OfficialResultCommand,
    ReviewCommand,
)
from analysis.ledger.store import (
    _deserialize_record,
    _logical_payload,
    _payload_hash,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review.classification import (
    ContradictionRule,
    PrincipalPathBinding,
)
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    ShadowComparison,
    ShadowMarketComparison,
    _shadow_comparison_payload,
)

from analysis.review.coverage_audit import (
    build_coverage_audit,
    render_coverage_audit_markdown,
)


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 28, 12, tzinfo=UTC)
FIRST_PITCH = CAPTURED_AT + timedelta(hours=2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_metrics() -> dict[str, object]:
    return {
        "sample_size": 1,
        "wins": 1,
        "losses": 0,
        "pushes": 0,
        "binary_sample_size": 1,
        "multiclass_sample_size": 0,
        "priced_stake_count": 1,
        "probability_clv_sample_size": 0,
        "price_clv_sample_size": 0,
        "hit_rate": 1.0,
        "push_rate": 0.0,
        "binary_brier": 0.1,
        "multiclass_brier": None,
        "log_loss": 0.1,
        "profit_units": 0.91,
        "roi": 0.91,
        "mean_probability_clv": None,
        "mean_price_clv": None,
    }


def _write_calibration(path: Path, ledger_path: Path) -> None:
    ledger_revision, appended_at_values = reporting_module._ledger_revision_mapping(
        tuple(LedgerStore(ledger_path).iter_records())
    )
    path.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "ledger_as_of": (
                    None
                    if not appended_at_values
                    else reporting_module._utc_iso(max(appended_at_values))
                ),
                "ledger_path": str(ledger_path.expanduser().resolve(strict=False)),
                "ledger_revision": ledger_revision,
                "migration": {
                    "confirmed_games": _progress(0, 30),
                    "market_types": {
                        market_type.value: _progress(0, 25)
                        for market_type in MarketType
                    },
                },
                "overall": _official_metrics(),
                "group_by": [],
                "groups": [],
            }
        ),
        encoding="utf-8",
    )


def _progress(count: int, threshold: int) -> dict[str, object]:
    return {
        "count": count,
        "threshold": threshold,
        "completed": min(count, threshold),
        "remaining": max(threshold - count, 0),
        "progress": min(count, threshold) / threshold,
        "threshold_met": count >= threshold,
    }


def _comparison_payload(
    *,
    forecast_run_id: str,
    market_key: str,
    eligible: bool,
    rating: str = "play",
) -> dict[str, object]:
    starter_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=("home_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )
    lineup_path = PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.TOP_ORDER_MATCHUP,
        component_ids=("away_full_runs",),
        contradiction_rule=ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
    )
    candidate = PriorityMarketCandidate(
        market="full_total",
        score=80,
        decision_rating=rating,
        evidence_paths=frozenset(
            (
                PriorityEvidencePath.STARTER_SUPPRESSION,
                PriorityEvidencePath.TOP_ORDER_MATCHUP,
            )
        ),
    )
    binding = CandidateReviewBinding(
        market_key=market_key,
        legacy_candidate_index=0,
        legacy_market="full_total",
        principal_paths=(starter_path, lineup_path),
    )
    comparison = ShadowMarketComparison(
        market_key=market_key,
        legacy_candidate_index=0,
        legacy_market="full_total",
        legacy_score=80,
        legacy_rating=rating,
        model_win_probability=0.55,
        fair_decimal_price=1.82,
        market_no_vig_probability=0.52,
        edge=0.03,
        principal_paths=(starter_path, lineup_path) if eligible else (),
        review_eligible=eligible,
        review_ineligibility_flags=() if eligible else ("invalid_path_count",),
    )
    return _shadow_comparison_payload(
        ShadowComparison(
            authority="legacy",
            legacy=LegacyDecisionSnapshot(
                model_version="legacy-fixed-v1",
                candidates=(candidate,),
                final_priority=("full_total",),
                review_bindings=(binding,),
            ),
            forecast_run_id=forecast_run_id,
            market_comparisons=(comparison,),
        )
    )


def _append_market(
    store: LedgerStore,
    *,
    game_pk: int,
    suffix: str,
    eligible: bool,
    rating: str = "play",
) -> tuple[object, object, object, str]:
    market_key = f"game:{game_pk}:full_total:over"
    forecast_run_id = f"{game_pk:064x}"
    normalized_input_hash = f"input:{suffix}"
    capture = store.append_capture(
        f"capture:{suffix}",
        CaptureCommand(
            game_pk=game_pk,
            captured_at=CAPTURED_AT,
            first_pitch=FIRST_PITCH,
            lineup_state="confirmed",
            out_of_sample_eligible=True,
            normalized_input_hash=normalized_input_hash,
            payload={
                "markets": [
                    {
                        "market_key": market_key,
                        "market_type": "full_total",
                        "settlement_format": "two_way",
                        "selected_side": "over",
                        "line": 8.5,
                        "selected_decimal_price": 1.91,
                        "opposing_decimal_price": 1.91,
                    }
                ]
            },
        ),
    )
    forecast = store.append_forecast(
        ForecastCommand(
            forecast_run_id=forecast_run_id,
            model_version="shadow-forecast-v1",
            simulation_config_hash="simulation-v1",
            normalized_input_hash=normalized_input_hash,
            outputs={
                "market_inputs": {
                    market_key: {
                        "market_key": market_key,
                        "market_type": "full_total",
                        "settlement_format": "two_way",
                        "selected_side": "over",
                        "line": 8.5,
                        "selected_decimal_price": 1.91,
                        "opposing_decimal_price": 1.91,
                    }
                },
                "shadow_comparison": _comparison_payload(
                    forecast_run_id=forecast_run_id,
                    market_key=market_key,
                    eligible=eligible,
                    rating=rating,
                ),
            },
        )
    )
    link = store.link_capture_forecast(capture.capture_id, forecast.forecast_run_id)
    return capture, forecast, link, market_key


def _append_result(store: LedgerStore, game_pk: int, *, revision: str, runs: int):
    return store.append_official_result(
        OfficialResultCommand(
            game_pk=game_pk,
            source_name="MLB",
            source_record=f"game/{game_pk}/boxscore",
            source_revision=revision,
            finalized_at=FIRST_PITCH + timedelta(hours=int(revision)),
            away_f5_runs=runs,
            home_f5_runs=0,
            away_full_runs=runs,
            home_full_runs=1,
            component_values={},
            documented_events=(),
        )
    )


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "audit_id": "2026-07-28-review-coverage",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _rewrite_forecast_with_valid_payload_hash(
    ledger_path: Path,
    forecast_run_id: str,
    mutate,
) -> None:
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row.get("forecast_run_id") != forecast_run_id:
            continue
        mutate(row)
        record = _deserialize_record(row)
        row["envelope"]["payload_hash"] = _payload_hash(_logical_payload(record))
        break
    else:
        raise AssertionError("required forecast row was not found")
    ledger_path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    ledger_path = tmp_path / "shadow.jsonl"
    store = LedgerStore(ledger_path)
    eligible_capture, eligible_forecast, eligible_link, eligible_market = _append_market(
        store,
        game_pk=900001,
        suffix="eligible",
        eligible=True,
    )
    eligible_result = _append_result(store, eligible_capture.game_pk, revision="1", runs=9)
    store.append_review(
        ReviewCommand(
            capture_forecast_link_record_id=eligible_link.envelope.record_id,
            capture_id=eligible_capture.capture_id,
            forecast_run_id=eligible_forecast.forecast_run_id,
            official_result_record_id=eligible_result.envelope.record_id,
            market_key=eligible_market,
            market_result="win",
            classification="A",
            inside_80_ratio=1.0,
            contradicted_paths=(),
            component_reviews=(
                {
                    "evidence_path": "starter_suppression",
                    "component_ids": ("home_pitcher_chain_runs_allowed",),
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
    settled_capture, settled_forecast, settled_link, settled_market = _append_market(
        store,
        game_pk=900002,
        suffix="settled",
        eligible=False,
    )
    first_result = _append_result(store, settled_capture.game_pk, revision="1", runs=9)
    store.append_official_result(
        OfficialResultCommand(
            game_pk=settled_capture.game_pk,
            source_name="MLB",
            source_record=f"game/{settled_capture.game_pk}/boxscore",
            source_revision="2",
            finalized_at=FIRST_PITCH + timedelta(hours=3),
            away_f5_runs=4,
            home_f5_runs=3,
            away_full_runs=4,
            home_full_runs=3,
            component_values={},
            documented_events=(),
            supersedes_record_id=first_result.envelope.record_id,
        )
    )
    unresolved_capture, unresolved_forecast, unresolved_link, unresolved_market = _append_market(
        store,
        game_pk=900003,
        suffix="postponed",
        eligible=False,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "game_pk": settled_capture.game_pk,
                "capture_id": settled_capture.capture_id,
                "forecast_run_id": settled_forecast.forecast_run_id,
                "capture_forecast_link_record_id": settled_link.envelope.record_id,
                "market_key": settled_market,
            },
            {
                "game_pk": unresolved_capture.game_pk,
                "capture_id": unresolved_capture.capture_id,
                "forecast_run_id": unresolved_forecast.forecast_run_id,
                "capture_forecast_link_record_id": unresolved_link.envelope.record_id,
                "market_key": unresolved_market,
            },
        ],
    )
    calibration_path = tmp_path / "calibration-latest.json"
    _write_calibration(calibration_path, ledger_path)
    return ledger_path, calibration_path, manifest_path


def test_coverage_audit_resolves_only_recommended_ineligible_entries(tmp_path: Path) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    ledger_hash = _sha256(ledger_path)
    calibration_hash = _sha256(calibration_path)

    audit = build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert audit["official_calibration_unchanged"] is True
    assert audit["resolved_omission_count"] == 1
    assert audit["unresolved_omission_count"] == 1
    assert audit["audit_only_metrics"]["losses"] == (
        audit["official_metrics"]["losses"] + 1
    )
    assert audit["entries"][0]["ineligibility_flags"] == [
        "invalid_path_count"
    ]
    assert audit["entries"][0]["market_result"] == "loss"
    records = tuple(LedgerStore(ledger_path).iter_records())
    latest_result = next(
        record
        for record in records
        if getattr(record, "game_pk", None) == 900002
        and getattr(record, "supersedes_record_id", None) is not None
    )
    assert (
        audit["entries"][0]["official_result_record_id"]
        == latest_result.envelope.record_id
    )
    assert audit["entries"][1]["status"] == "unresolved"
    assert set(audit["audit_only_metrics"]) == {
        "sample_size",
        "wins",
        "losses",
        "pushes",
        "priced_stake_count",
        "profit_units",
        "roi",
    }
    assert _sha256(ledger_path) == ledger_hash
    assert _sha256(calibration_path) == calibration_hash


def test_coverage_audit_json_and_markdown_are_deterministic(tmp_path: Path) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)

    first = build_coverage_audit(ledger_path, calibration_path, manifest_path)
    second = build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert canonical_json(first).encode() == canonical_json(second).encode()
    assert render_coverage_audit_markdown(first).encode() == (
        render_coverage_audit_markdown(second).encode()
    )
    disclaimer = "이 조정치는 감사용이며 공식 calibration row가 아닙니다."
    assert disclaimer in render_coverage_audit_markdown(first)


def test_coverage_audit_accepts_calibration_schema_v2_watchlist(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["schema_version"] = 2
    calibration["watchlist"] = []
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")

    audit = build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert audit["official_metrics"] == _official_metrics()
    assert audit["official_calibration_unchanged"] is True


def test_coverage_audit_preserves_historical_calibration_schema_v1(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["schema_version"] == 1
    assert "watchlist" not in calibration
    assert build_coverage_audit(ledger_path, calibration_path, manifest_path)[
        "official_metrics"
    ] == _official_metrics()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda calibration: calibration.update({"schema_version": 2}),
            "calibration has invalid fields",
        ),
        (
            lambda calibration: calibration.update({"watchlist": []}),
            "calibration has invalid fields",
        ),
        (
            lambda calibration: calibration.update(
                {"schema_version": 2, "watchlist": {}}
            ),
            "watchlist must be a JSON array",
        ),
        (
            lambda calibration: calibration.update({"schema_version": 3}),
            "calibration schema_version must be 1 or 2",
        ),
    ),
)
def test_coverage_audit_requires_schema_specific_calibration_shape(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    mutation(calibration)
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_rejects_unknown_calibration_metric_fields(tmp_path: Path) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["overall"]["unexpected_metric"] = 1
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match="overall metrics has invalid fields"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["notes"] = "not allowed"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest entries\\[0\\] has invalid fields"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_rejects_a_mismatched_exact_link(tmp_path: Path) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["capture_forecast_link_record_id"] = (
        manifest["entries"][1]["capture_forecast_link_record_id"]
    )
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="link relationship does not resolve"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda row: row.update({"normalized_input_hash": "tampered-input"}),
            "capture and forecast normalized_input_hash must match",
        ),
        (
            lambda row: row["outputs"]["market_inputs"]
            [next(iter(row["outputs"]["market_inputs"]))].update({"line": 9.5}),
            "forecast market_inputs must exactly match canonical capture markets",
        ),
    ),
)
def test_coverage_audit_rejects_rehashed_required_forecast_cross_binding(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forecast_run_id = manifest["entries"][0]["forecast_run_id"]

    _rewrite_forecast_with_valid_payload_hash(
        ledger_path,
        forecast_run_id,
        mutate,
    )

    with pytest.raises(LedgerCorruptionError, match=message):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_rejects_tampered_unselected_forecast_payload_hash(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_forecast_ids = {
        entry["forecast_run_id"] for entry in manifest["entries"]
    }
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if (
            row.get("envelope", {}).get("record_kind") == "forecast"
            and row["forecast_run_id"] not in selected_forecast_ids
        ):
            market = next(iter(row["outputs"]["market_inputs"].values()))
            market["line"] = 9.5
            break
    else:
        raise AssertionError("unselected forecast row was not found")
    ledger_path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(LedgerCorruptionError, match="record payload hash mismatch"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("ledger_path", lambda calibration: calibration.update({"ledger_path": "unrelated-ledger.jsonl"})),
        (
            "record_count",
            lambda calibration: calibration["ledger_revision"].update(
                {"record_count": calibration["ledger_revision"]["record_count"] + 1}
            ),
        ),
        (
            "state_hash",
            lambda calibration: calibration["ledger_revision"].update(
                {"state_hash": "0" * 64}
            ),
        ),
        (
            "ledger_as_of",
            lambda calibration: calibration.update(
                {"ledger_as_of": "1999-01-01T00:00:00Z"}
            ),
        ),
    ),
)
def test_coverage_audit_cli_rejects_unrelated_calibration_revision_without_output(
    tmp_path: Path,
    field: str,
    mutation,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    mutation(calibration)
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")
    ledger_hash = _sha256(ledger_path)
    calibration_hash = _sha256(calibration_path)
    output_dir = tmp_path / "audit-output"

    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "analysis.review.coverage_audit",
            "--ledger",
            str(ledger_path),
            "--calibration",
            str(calibration_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"coverage-audit: calibration {field} does not match ledger\n"
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()
    assert _sha256(ledger_path) == ledger_hash
    assert _sha256(calibration_path) == calibration_hash


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda calibration: calibration.update({"migration": 7}),
            "migration must be a JSON object",
        ),
        (
            lambda calibration: calibration.update({"group_by": "market_type"}),
            "group_by must be a JSON array",
        ),
        (
            lambda calibration: calibration.update({"groups": {"bad": "shape"}}),
            "groups must be a JSON array",
        ),
        (
            lambda calibration: calibration["overall"].update({"losses": 1}),
            "outcome counts must sum to sample_size",
        ),
        (
            lambda calibration: calibration.update({"schema_version": True}),
            "calibration schema_version must be a positive integer",
        ),
    ),
)
def test_coverage_audit_rejects_invalid_calibration_schema_and_invariants(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    mutation(calibration)
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


@pytest.mark.parametrize(
    ("path_name", "contents"),
    (
        ("manifest.json", '{"schema_version":1,"schema_version":1}'),
        ("calibration-latest.json", '{"schema_version":NaN}'),
        ("calibration-latest.json", '{"schema_version":Infinity}'),
    ),
)
def test_coverage_audit_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    path_name: str,
    contents: str,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    path = {
        "manifest.json": manifest_path,
        "calibration-latest.json": calibration_path,
    }[path_name]
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="is not valid JSON"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_accepts_serialized_group_dimensions_out_of_order(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["group_by"] = ["model_version", "market_type"]
    calibration["groups"] = [
        {
            "dimensions": {
                "market_type": "full_total",
                "model_version": "shadow-forecast-v1",
            },
            "metrics": _official_metrics(),
        }
    ]
    calibration_path.write_text(canonical_json(calibration), encoding="utf-8")

    assert build_coverage_audit(ledger_path, calibration_path, manifest_path)[
        "official_metrics"
    ] == _official_metrics()


def test_coverage_audit_retains_only_exact_manifest_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    retained_sizes: list[tuple[int, int, int, int]] = []
    original = coverage_audit_module._audit_entry

    def inspect_retained_records(
        entry,
        captures,
        forecasts,
        links,
        latest_results,
    ):
        retained_sizes.append(
            (
                len(captures),
                len(forecasts),
                len(links),
                len(latest_results),
            )
        )
        return original(entry, captures, forecasts, links, latest_results)

    monkeypatch.setattr(
        coverage_audit_module,
        "_audit_entry",
        inspect_retained_records,
    )

    build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert retained_sizes == [(2, 2, 2, 1), (2, 2, 2, 1)]


def test_coverage_audit_does_not_use_ledger_store_iterator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)

    def fail_if_materialized(*args, **kwargs):
        raise AssertionError("coverage audit must stream the ledger directly")

    monkeypatch.setattr(LedgerStore, "iter_records", fail_if_materialized)

    assert build_coverage_audit(ledger_path, calibration_path, manifest_path)[
        "resolved_omission_count"
    ] == 1


def test_missing_ledger_does_not_create_parent_or_sidecars(tmp_path: Path) -> None:
    _, calibration_path, manifest_path = _audit_fixture(tmp_path)
    ledger_path = tmp_path / "missing" / "shadow.jsonl"

    with pytest.raises(ValueError, match="ledger does not exist"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert not ledger_path.parent.exists()


def test_invalid_ledger_does_not_create_lock_or_checkpoint(tmp_path: Path) -> None:
    _, calibration_path, manifest_path = _audit_fixture(tmp_path)
    ledger_path = tmp_path / "invalid" / "shadow.jsonl"
    ledger_path.parent.mkdir()
    ledger_path.write_bytes(b"{invalid-json}\n")

    with pytest.raises(LedgerCorruptionError, match="invalid ledger row"):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)

    assert tuple(ledger_path.parent.iterdir()) == (ledger_path,)


@pytest.mark.parametrize(
    ("eligible", "rating", "message"),
    (
        (True, "play", "named market is not review-ineligible"),
        (False, "pass", "named market is not recommended"),
    ),
)
def test_coverage_audit_requires_recommended_ineligible_comparison(
    tmp_path: Path,
    eligible: bool,
    rating: str,
    message: str,
) -> None:
    ledger_path = tmp_path / "shadow.jsonl"
    store = LedgerStore(ledger_path)
    capture, forecast, link, market_key = _append_market(
        store,
        game_pk=900004,
        suffix=f"{rating}-{eligible}",
        eligible=eligible,
        rating=rating,
    )
    _append_result(store, capture.game_pk, revision="1", runs=7)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "game_pk": capture.game_pk,
                "capture_id": capture.capture_id,
                "forecast_run_id": forecast.forecast_run_id,
                "capture_forecast_link_record_id": link.envelope.record_id,
                "market_key": market_key,
            }
        ],
    )
    calibration_path = tmp_path / "calibration-latest.json"
    _write_calibration(calibration_path, ledger_path)

    with pytest.raises(ValueError, match=message):
        build_coverage_audit(ledger_path, calibration_path, manifest_path)


def test_coverage_audit_module_cli_writes_exact_bytes_without_warning(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    output_dir = tmp_path / "audit-output"
    expected = build_coverage_audit(
        ledger_path,
        calibration_path,
        manifest_path,
    )

    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "analysis.review.coverage_audit",
            "--ledger",
            str(ledger_path),
            "--calibration",
            str(calibration_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert (output_dir / "audit.json").read_bytes() == (
        canonical_json(expected) + "\n"
    ).encode()
    assert (output_dir / "audit.md").read_bytes() == (
        render_coverage_audit_markdown(expected).encode()
    )


def test_coverage_audit_module_cli_reports_validation_error_without_output(
    tmp_path: Path,
) -> None:
    ledger_path, calibration_path, manifest_path = _audit_fixture(tmp_path)
    manifest_path.write_text("{invalid-json}", encoding="utf-8")
    output_dir = tmp_path / "audit-output"

    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "analysis.review.coverage_audit",
            "--ledger",
            str(ledger_path),
            "--calibration",
            str(calibration_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ),
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "coverage-audit: manifest is not valid JSON\n"
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()


def test_july_28_manifest_has_exact_frozen_relationships() -> None:
    manifest_path = Path("data/audits/2026-07-28-review-coverage-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest == {
        "schema_version": 1,
        "audit_id": "2026-07-28-review-coverage",
        "entries": [
            {
                "game_pk": 822705,
                "capture_id": "37542c98c77540b7a5e197e23f47dc1c",
                "forecast_run_id": "517cb5c1a300b14900afd5d48dcc018ed22ab51850fb23e5733ae542fd208d56",
                "capture_forecast_link_record_id": "a11b949640c643fea5dbee4346c561a7",
                "market_key": "game:822705:f5_side:home",
            },
            {
                "game_pk": 822705,
                "capture_id": "37542c98c77540b7a5e197e23f47dc1c",
                "forecast_run_id": "517cb5c1a300b14900afd5d48dcc018ed22ab51850fb23e5733ae542fd208d56",
                "capture_forecast_link_record_id": "a11b949640c643fea5dbee4346c561a7",
                "market_key": "game:822705:full_total:over",
            },
            {
                "game_pk": 822705,
                "capture_id": "37542c98c77540b7a5e197e23f47dc1c",
                "forecast_run_id": "517cb5c1a300b14900afd5d48dcc018ed22ab51850fb23e5733ae542fd208d56",
                "capture_forecast_link_record_id": "a11b949640c643fea5dbee4346c561a7",
                "market_key": "game:822705:f5_total:over",
            },
            {
                "game_pk": 823353,
                "capture_id": "c9654b6073c849578873dadd6fc80211",
                "forecast_run_id": "22efca40e1ca7284ff4ae7b108a524a25bfd82b3eaaafb1c12cdb1b93b4aafa4",
                "capture_forecast_link_record_id": "f7d8f2b45f0b4d3abf122aca7f12beeb",
                "market_key": "game:823353:f5_total:over",
            },
            {
                "game_pk": 823839,
                "capture_id": "7f68480efc27484e8c400858614e1e09",
                "forecast_run_id": "5961d07c05d50ff8b4df34fc53a07e1c46166242ea2a5516ffc42ce89e18fe33",
                "capture_forecast_link_record_id": "44c6b7cc156b4da4ba6bfb0b763871b9",
                "market_key": "game:823839:f5_total:under",
            },
            {
                "game_pk": 824490,
                "capture_id": "4dc190dc8af3455387d2509caf56ff93",
                "forecast_run_id": "b8f4e0431f7f6357070f62ba21a58d8207c5d04414debe71d7ac3887f012be9f",
                "capture_forecast_link_record_id": "c6b558e0edd743b2a34d530c94b229fb",
                "market_key": "game:824490:f5_total:over",
            },
        ],
    }
