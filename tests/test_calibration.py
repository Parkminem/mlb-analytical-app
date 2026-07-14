from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import unittest

import analysis.review.calibration as calibration_module
from analysis.ledger import (
    CaptureCommand,
    ForecastCommand,
    LedgerStore,
    MarketObservationCommand,
    OfficialResultCommand,
    ReviewCommand,
)
from analysis.review.calibration import (
    CalibrationGroup,
    CalibrationMetrics,
    CalibrationReport,
    CalibrationRow,
    CaptureCutoffPolicy,
    GROUPING_DIMENSIONS,
    LATEST_PREGAME,
    build_calibration_report,
    probability_bucket,
    priority_score_bucket,
    select_primary_rows,
)


UTC = timezone.utc
FIRST_PITCH = datetime(2026, 7, 14, 23, 0, tzinfo=UTC)
_USE_OBSERVED_AT = object()


def calibration_row(**overrides: object) -> CalibrationRow:
    values: dict[str, object] = {
        "game_pk": 900001,
        "capture_id": "capture-1",
        "forecast_run_id": "forecast-1",
        "market_key": "game:900001:ml:away",
        "model_version": "forecast-v1",
        "market_type": "ml",
        "settlement_format": "two_way",
        "line": None,
        "lineup_state": "confirmed",
        "starter_structure": "starter_vs_starter",
        "legacy_priority_score": 80,
        "p_win": 0.55,
        "p_push": 0.0,
        "p_loss": 0.45,
        "market_result": "win",
        "selected_decimal_price": 2.0,
        "closing_selected_decimal_price": 1.8,
        "entry_no_vig_probability": 0.52,
        "closing_no_vig_probability": 0.56,
    }
    values.update(overrides)
    if "line" not in overrides:
        default_lines = {
            "f5_side": 0.0,
            "f5_total": 8.0,
            "plus_1_5": 1.5,
            "full_total": 8.0,
        }
        values["line"] = default_lines.get(values["market_type"])
    return CalibrationRow(**values)


def market_key(game_pk: int, market_type: str = "ml") -> str:
    return f"game:{game_pk}:{market_type}:away"


def forecast_outputs(
    game_pk: int,
    *,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
    p_win: float = 0.55,
    p_push: float = 0.0,
    p_loss: float = 0.45,
    selected_price: float | None = 2.0,
    opposing_price: float | None = 1.9,
    entry_no_vig: float | None = 0.52,
    legacy_score: int | None = 80,
) -> dict[str, object]:
    key = market_key(game_pk, market_type)
    return {
        "forecast_output": {
            "markets": {
                key: {
                    "probabilities": {
                        "win": p_win,
                        "push": p_push,
                        "loss": p_loss,
                    },
                    "selected_decimal_price": selected_price,
                    "opposing_decimal_price": opposing_price,
                    "market_no_vig_probability": entry_no_vig,
                }
            }
        },
        "shadow_comparison": {
            "market_comparisons": [
                {
                    "market_key": key,
                    "legacy_score": legacy_score,
                }
            ]
        },
    }


def capture_payload(
    game_pk: int,
    *,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
    away_roles: tuple[str, ...] = ("starter",),
    home_roles: tuple[str, ...] = ("starter",),
) -> dict[str, object]:
    return {
        "away_pitching": {
            "pitchers": [{"role": role} for role in away_roles],
        },
        "home_pitching": {
            "pitchers": [{"role": role} for role in home_roles],
        },
        "markets": [
            {
                "market_key": market_key(game_pk, market_type),
                "market_type": market_type,
                "settlement_format": "two_way",
                "selected_side": selected_side,
                "line": line,
            }
        ],
    }


def append_capture_link(
    store: LedgerStore,
    *,
    game_pk: int,
    captured_at: datetime,
    lineup_state: str,
    forecast_run_id: str,
    id_suffix: str,
    out_of_sample_eligible: bool = True,
    supersedes_capture_id: str | None = None,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
    away_roles: tuple[str, ...] = ("starter",),
    home_roles: tuple[str, ...] = ("starter",),
    p_win: float = 0.55,
    p_push: float = 0.0,
    p_loss: float = 0.45,
    selected_price: float | None = 2.0,
    opposing_price: float | None = 1.9,
    entry_no_vig: float | None = 0.52,
    legacy_score: int | None = 80,
    forecast_outputs_override: dict[str, object] | None = None,
):
    normalized_hash = f"input:{forecast_run_id}"
    capture = store.append_capture(
        f"capture:{game_pk}:{id_suffix}",
        CaptureCommand(
            game_pk=game_pk,
            captured_at=captured_at,
            first_pitch=FIRST_PITCH,
            lineup_state=lineup_state,
            out_of_sample_eligible=out_of_sample_eligible,
            normalized_input_hash=normalized_hash,
            payload=capture_payload(
                game_pk,
                market_type=market_type,
                line=line,
                selected_side=selected_side,
                away_roles=away_roles,
                home_roles=home_roles,
            ),
            supersedes_capture_id=supersedes_capture_id,
        ),
    )
    forecast = store.append_forecast(
        ForecastCommand(
            forecast_run_id=forecast_run_id,
            model_version="forecast-v1",
            simulation_config_hash="simulation-v1",
            normalized_input_hash=normalized_hash,
            outputs=(
                forecast_outputs(
                    game_pk,
                    market_type=market_type,
                    line=line,
                    selected_side=selected_side,
                    p_win=p_win,
                    p_push=p_push,
                    p_loss=p_loss,
                    selected_price=selected_price,
                    opposing_price=opposing_price,
                    entry_no_vig=entry_no_vig,
                    legacy_score=legacy_score,
                )
                if forecast_outputs_override is None
                else forecast_outputs_override
            ),
        )
    )
    link = store.link_capture_forecast(capture.capture_id, forecast_run_id)
    return capture, forecast, link


def append_entry_market(
    store: LedgerStore,
    *,
    game_pk: int,
    id_suffix: str,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
) -> None:
    store.append_market_observation(
        MarketObservationCommand(
            game_pk=game_pk,
            source_name="book",
            source_record=f"entry:{game_pk}:{id_suffix}",
            source_revision="1",
            observation_kind="entry",
            market_key=market_key(game_pk, market_type),
            market_type=market_type,
            settlement_format="two_way",
            selected_side=selected_side,
            line=line,
            selected_decimal_price=2.0,
            opposing_decimal_price=1.9,
            observed_at=FIRST_PITCH - timedelta(hours=2),
            available_at=FIRST_PITCH - timedelta(hours=2),
        )
    )


def append_result(
    store: LedgerStore,
    *,
    game_pk: int,
    revision: str = "1",
    supersedes_record_id: str | None = None,
):
    return store.append_official_result(
        OfficialResultCommand(
            game_pk=game_pk,
            source_name="MLB",
            source_record=f"game/{game_pk}/boxscore",
            source_revision=revision,
            finalized_at=FIRST_PITCH + timedelta(hours=4, minutes=int(revision)),
            away_f5_runs=2,
            home_f5_runs=1,
            away_full_runs=4,
            home_full_runs=3,
            component_values={"away_full_runs": 4.0, "home_full_runs": 3.0},
            documented_events=(),
            supersedes_record_id=supersedes_record_id,
        )
    )


def append_review(
    store: LedgerStore,
    *,
    capture,
    forecast,
    link,
    result,
    market_result: str = "win",
    market_type: str = "ml",
    id_suffix: str = "1",
    classification: str | None = None,
    inside_80_ratio: float = 1.0,
    contradicted_paths: tuple[str, ...] = (),
    component_reviews: tuple[dict[str, object], ...] | None = None,
):
    resolved_classification = classification
    if resolved_classification is None:
        if contradicted_paths or inside_80_ratio < 0.60:
            resolved_classification = "C"
        elif market_result == "win" and inside_80_ratio >= 0.75:
            resolved_classification = "A"
        else:
            resolved_classification = "B"
    resolved_component_reviews = component_reviews
    if resolved_component_reviews is None:
        evidence_paths = contradicted_paths or ("away_lineup.power",)
        resolved_component_reviews = tuple(
            {
                "evidence_path": evidence_path,
                "component_ids": ("away_full_runs",),
                "contradicted": evidence_path in contradicted_paths,
                "inside_80_count": int(inside_80_ratio >= 0.50),
                "component_count": 1,
                "notes": (),
            }
            for evidence_path in evidence_paths
        )
    return store.append_review(
        ReviewCommand(
            capture_forecast_link_record_id=link.envelope.record_id,
            capture_id=capture.capture_id,
            forecast_run_id=forecast.forecast_run_id,
            official_result_record_id=result.envelope.record_id,
            market_key=market_key(capture.game_pk, market_type),
            market_result=market_result,
            classification=resolved_classification,
            inside_80_ratio=inside_80_ratio,
            contradicted_paths=contradicted_paths,
            component_reviews=resolved_component_reviews,
        ),
        idempotency_key=(
            f"review:{capture.capture_id}:{result.envelope.record_id}:"
            f"{market_type}:{id_suffix}"
        ),
    )


def append_close_market(
    store: LedgerStore,
    *,
    game_pk: int,
    id_suffix: str,
    observed_at: datetime,
    selected_price: float | None,
    opposing_price: float | None,
    available_at: datetime | None | object = _USE_OBSERVED_AT,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
    close_market_key: str | None = None,
    source_name: str = "book",
    source_record: str | None = None,
    supersedes_record_id: str | None = None,
):
    resolved_available_at = (
        observed_at if available_at is _USE_OBSERVED_AT else available_at
    )
    return store.append_market_observation(
        MarketObservationCommand(
            game_pk=game_pk,
            source_name=source_name,
            source_record=source_record or f"close:{game_pk}",
            source_revision=id_suffix,
            observation_kind="close",
            market_key=close_market_key or market_key(game_pk, market_type),
            market_type=market_type,
            settlement_format="two_way",
            selected_side=selected_side,
            line=line,
            selected_decimal_price=selected_price,
            opposing_decimal_price=opposing_price,
            observed_at=observed_at,
            available_at=resolved_available_at,
            supersedes_record_id=supersedes_record_id,
        )
    )


class ReorderedLedgerView:
    def __init__(self, store: LedgerStore) -> None:
        self._store = store

    def iter_records(self, kind: str | None = None):
        return iter(reversed(tuple(self._store.iter_records(kind))))

    def records_superseded_by(self, kind: str):
        return self._store.records_superseded_by(kind)

    def links_for_capture(self, capture_id: str):
        return tuple(reversed(self._store.links_for_capture(capture_id)))

    def forecast_for_link(self, link_record_id: str):
        return self._store.forecast_for_link(link_record_id)

    def latest_official_result(self, game_pk: int):
        return self._store.latest_official_result(game_pk)


class InstrumentedLedgerView:
    def __init__(self, store: LedgerStore) -> None:
        self._store = store
        self.calls = {
            "iter_records": 0,
            "records_superseded_by": 0,
            "latest_official_result": 0,
            "links_for_capture": 0,
            "forecast_for_link": 0,
        }

    def iter_records(self, kind: str | None = None):
        self.calls["iter_records"] += 1
        return self._store.iter_records(kind)

    def records_superseded_by(self, kind: str):
        self.calls["records_superseded_by"] += 1
        return self._store.records_superseded_by(kind)

    def latest_official_result(self, game_pk: int):
        self.calls["latest_official_result"] += 1
        return self._store.latest_official_result(game_pk)

    def links_for_capture(self, capture_id: str):
        self.calls["links_for_capture"] += 1
        return self._store.links_for_capture(capture_id)

    def forecast_for_link(self, link_record_id: str):
        self.calls["forecast_for_link"] += 1
        return self._store.forecast_for_link(link_record_id)


class CalibrationContractTests(unittest.TestCase):
    def test_cutoff_policy_is_frozen_and_validated(self):
        self.assertEqual(LATEST_PREGAME.name, "latest_pregame")
        with self.assertRaises(FrozenInstanceError):
            LATEST_PREGAME.name = "changed"
        with self.assertRaisesRegex(ValueError, "name"):
            CaptureCutoffPolicy(name="")
        with self.assertRaisesRegex(ValueError, "minimum_seconds"):
            CaptureCutoffPolicy(name="bad", minimum_seconds_before_first_pitch=-1)
        with self.assertRaisesRegex(ValueError, "required_lineup_state"):
            CaptureCutoffPolicy(name="bad", required_lineup_state="")
        with self.assertRaisesRegex(ValueError, "include_later"):
            CaptureCutoffPolicy(
                name="bad",
                include_later_superseded_captures=1,
            )

    def test_latest_pregame_name_is_reserved_for_the_headline_policy(self):
        invalid_overrides = (
            {"minimum_seconds_before_first_pitch": 1},
            {"required_lineup_state": "projected"},
            {"include_later_superseded_captures": True},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "latest_pregame"):
                    CaptureCutoffPolicy(name="latest_pregame", **overrides)

    def test_probability_and_priority_bucket_boundaries_are_fixed(self):
        self.assertEqual(probability_bucket(0.0), "0.00-0.45")
        self.assertEqual(probability_bucket(0.45), "0.45-0.50")
        self.assertEqual(probability_bucket(0.50), "0.50-0.55")
        self.assertEqual(probability_bucket(0.55), "0.55-0.60")
        self.assertEqual(probability_bucket(0.60), "0.60-1.00")
        self.assertEqual(probability_bucket(1.0), "0.60-1.00")
        self.assertEqual(priority_score_bucket(0), "0-69")
        self.assertEqual(priority_score_bucket(70), "70-74")
        self.assertEqual(priority_score_bucket(75), "75-79")
        self.assertEqual(priority_score_bucket(80), "80-84")
        self.assertEqual(priority_score_bucket(85), "85-100")
        self.assertEqual(priority_score_bucket(100), "85-100")

    def test_bucket_helpers_reject_values_outside_their_domains(self):
        for value in (-0.01, 1.01, float("nan"), True):
            with self.subTest(probability=value):
                with self.assertRaisesRegex(ValueError, "probability"):
                    probability_bucket(value)
        for value in (-1, 101, 80.0, True):
            with self.subTest(score=value):
                with self.assertRaisesRegex(ValueError, "priority score"):
                    priority_score_bucket(value)

    def test_calibration_row_is_frozen_and_validates_probabilities(self):
        row = calibration_row()
        with self.assertRaises(FrozenInstanceError):
            row.p_win = 0.1
        with self.assertRaisesRegex(ValueError, "sum to one"):
            calibration_row(p_win=0.6)
        with self.assertRaisesRegex(ValueError, "p_push"):
            calibration_row(p_win=0.0, p_push=float("nan"), p_loss=0.0)

    def test_calibration_row_rejects_exact_one_with_residual_probability(self):
        invalid_rows = (
            {"p_win": 1.0, "p_push": 0.0, "p_loss": 5e-13},
            {
                "p_win": 5e-13,
                "p_push": 1.0,
                "p_loss": 0.0,
                "market_result": "push",
            },
            {"p_win": 5e-13, "p_push": 0.0, "p_loss": 1.0},
        )
        for overrides in invalid_rows:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "exact-one"):
                    calibration_row(**overrides)

    def test_calibration_row_validates_identifiers_scores_results_and_prices(self):
        invalid_rows = (
            ({"game_pk": 0}, "game_pk"),
            ({"capture_id": ""}, "capture_id"),
            ({"legacy_priority_score": 101}, "legacy_priority_score"),
            ({"market_result": "void"}, "market_result"),
            ({"selected_decimal_price": 1.0}, "selected_decimal_price"),
            ({"closing_selected_decimal_price": float("inf")}, "closing_selected"),
            ({"entry_no_vig_probability": -0.01}, "entry_no_vig"),
            ({"closing_no_vig_probability": 1.01}, "closing_no_vig"),
        )
        for overrides, message in invalid_rows:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    calibration_row(**overrides)

    def test_calibration_row_derives_push_capability_from_settlement_terms(self):
        cases = (
            ("f5_side", "two_way", 0.0, True),
            ("f5_side", "two_way", -1.0, True),
            ("f5_side", "two_way", 0.5, False),
            ("f5_side", "two_way", -0.5, False),
            ("f5_side", "three_way", 0.0, False),
            ("f5_total", "two_way", 8.0, True),
            ("f5_total", "two_way", 8.5, False),
            ("full_total", "two_way", 9.0, True),
            ("full_total", "two_way", 8.5, False),
            ("ml", "two_way", None, False),
            ("plus_1_5", "two_way", 1.5, False),
        )
        for market_type, settlement_format, line, expected in cases:
            with self.subTest(
                market_type=market_type,
                settlement_format=settlement_format,
                line=line,
            ):
                row = calibration_row(
                    market_type=market_type,
                    settlement_format=settlement_format,
                    line=line,
                )
                self.assertEqual(row.settlement_format, settlement_format)
                self.assertEqual(row.line, line)
                self.assertIs(row.push_capable, expected)

        row = calibration_row(market_type="f5_side")
        with self.assertRaises(FrozenInstanceError):
            row.push_capable = False

    def test_calibration_row_validates_authoritative_market_terms(self):
        invalid_rows = (
            ({"market_type": "unknown", "line": None}, "market_type"),
            ({"settlement_format": "unknown"}, "settlement_format"),
            ({"line": True}, "line"),
            ({"line": float("inf")}, "line"),
            ({"market_type": "ml", "line": 0.0}, "ML line"),
            ({"market_type": "f5_side", "line": None}, "f5_side line"),
            ({"market_type": "f5_total", "line": None}, "f5_total line"),
            ({"market_type": "f5_total", "line": -0.5}, "total line"),
            ({"market_type": "plus_1_5", "line": 1.0}, "plus_1_5 line"),
            (
                {
                    "market_type": "full_total",
                    "settlement_format": "three_way",
                    "line": 8.0,
                },
                "settlement_format",
            ),
        )
        for overrides, message in invalid_rows:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    calibration_row(**overrides)

    def test_non_push_capable_rows_reject_push_probability_and_outcomes(self):
        invalid_rows = (
            {
                "market_type": "ml",
                "p_win": 0.5,
                "p_push": 0.1,
                "p_loss": 0.4,
            },
            {
                "market_type": "full_total",
                "line": 8.5,
                "p_win": 0.5,
                "p_push": 0.1,
                "p_loss": 0.4,
            },
            {
                "market_type": "plus_1_5",
                "line": 1.5,
                "market_result": "push",
            },
            {
                "market_type": "f5_side",
                "settlement_format": "three_way",
                "line": 0.0,
                "market_result": "push",
            },
            {
                "market_type": "f5_side",
                "settlement_format": "two_way",
                "line": 0.5,
                "p_win": 0.5,
                "p_push": 0.1,
                "p_loss": 0.4,
            },
            {
                "market_type": "f5_side",
                "settlement_format": "two_way",
                "line": -0.5,
                "market_result": "push",
            },
        )
        for overrides in invalid_rows:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "non-push-capable"):
                    calibration_row(**overrides)

    def test_metric_and_group_types_are_frozen_and_validated(self):
        metrics = CalibrationMetrics(
            sample_size=1,
            wins=1,
            losses=0,
            pushes=0,
            binary_sample_size=1,
            multiclass_sample_size=1,
            priced_stake_count=1,
            probability_clv_sample_size=1,
            price_clv_sample_size=1,
            hit_rate=1.0,
            push_rate=0.0,
            binary_brier=0.25,
            multiclass_brier=0.5,
            log_loss=math.log(2.0),
            profit_units=1.0,
            roi=1.0,
            mean_probability_clv=0.04,
            mean_price_clv=0.1,
        )
        group = CalibrationGroup(dimensions=(("market_type", "ml"),), metrics=metrics)
        report = CalibrationReport(
            overall=metrics,
            groups=(group,),
            group_by=("market_type",),
        )
        with self.assertRaises(FrozenInstanceError):
            report.groups = ()
        with self.assertRaisesRegex(ValueError, "counts"):
            CalibrationMetrics(
                sample_size=2,
                wins=1,
                losses=0,
                pushes=0,
                binary_sample_size=1,
                multiclass_sample_size=1,
                priced_stake_count=1,
                probability_clv_sample_size=1,
                price_clv_sample_size=1,
                hit_rate=1.0,
                push_rate=0.0,
                binary_brier=0.25,
                multiclass_brier=0.5,
                log_loss=math.log(2.0),
                profit_units=1.0,
                roi=1.0,
                mean_probability_clv=0.04,
                mean_price_clv=0.1,
            )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CalibrationGroup(
                dimensions=(("market_type", "ml"), ("market_type", "f5_side")),
                metrics=metrics,
            )

    def test_metric_availability_matches_outcome_sample(self):
        values: dict[str, object] = {
            "sample_size": 1,
            "wins": 1,
            "losses": 0,
            "pushes": 0,
            "binary_sample_size": 1,
            "multiclass_sample_size": 1,
            "priced_stake_count": 0,
            "probability_clv_sample_size": 0,
            "price_clv_sample_size": 0,
            "hit_rate": 1.0,
            "push_rate": 0.0,
            "binary_brier": 0.25,
            "multiclass_brier": 0.5,
            "log_loss": math.log(2.0),
            "profit_units": None,
            "roi": None,
            "mean_probability_clv": None,
            "mean_price_clv": None,
        }
        for field_name in ("binary_brier", "log_loss"):
            with self.subTest(field_name=field_name):
                invalid = dict(values)
                invalid[field_name] = None
                with self.assertRaisesRegex(ValueError, field_name):
                    CalibrationMetrics(**invalid)

        no_multiclass = dict(values)
        no_multiclass["multiclass_sample_size"] = 0
        no_multiclass["multiclass_brier"] = None
        self.assertIsNone(
            CalibrationMetrics(**no_multiclass).multiclass_brier
        )

        push_values = dict(values)
        push_values.update(
            wins=0,
            pushes=1,
            hit_rate=None,
            push_rate=1.0,
            binary_sample_size=0,
            binary_brier=0.25,
            log_loss=None,
        )
        with self.assertRaisesRegex(ValueError, "binary_brier"):
            CalibrationMetrics(**push_values)

    def test_push_rate_direct_invariants_match_outcome_counts(self):
        values: dict[str, object] = {
            "sample_size": 2,
            "wins": 1,
            "losses": 0,
            "pushes": 1,
            "binary_sample_size": 1,
            "multiclass_sample_size": 1,
            "priced_stake_count": 0,
            "probability_clv_sample_size": 0,
            "price_clv_sample_size": 0,
            "hit_rate": 1.0,
            "push_rate": 0.5,
            "binary_brier": 0.25,
            "multiclass_brier": 0.5,
            "log_loss": math.log(2.0),
            "profit_units": None,
            "roi": None,
            "mean_probability_clv": None,
            "mean_price_clv": None,
        }
        self.assertEqual(CalibrationMetrics(**values).push_rate, 0.5)
        for invalid_rate in (None, 0.25, -0.1, float("nan")):
            with self.subTest(push_rate=invalid_rate):
                invalid = dict(values)
                invalid["push_rate"] = invalid_rate
                with self.assertRaisesRegex(ValueError, "push_rate"):
                    CalibrationMetrics(**invalid)

        empty = dict(values)
        empty.update(
            sample_size=0,
            wins=0,
            pushes=0,
            binary_sample_size=0,
            multiclass_sample_size=0,
            hit_rate=None,
            push_rate=0.0,
            binary_brier=None,
            multiclass_brier=None,
            log_loss=None,
        )
        with self.assertRaisesRegex(ValueError, "push_rate"):
            CalibrationMetrics(**empty)

    def test_metric_denominators_validate_domains_and_metric_availability(self):
        values: dict[str, object] = {
            "sample_size": 1,
            "wins": 1,
            "losses": 0,
            "pushes": 0,
            "binary_sample_size": 1,
            "multiclass_sample_size": 0,
            "priced_stake_count": 0,
            "probability_clv_sample_size": 0,
            "price_clv_sample_size": 0,
            "hit_rate": 1.0,
            "push_rate": 0.0,
            "binary_brier": 0.25,
            "multiclass_brier": None,
            "log_loss": math.log(2.0),
            "profit_units": None,
            "roi": None,
            "mean_probability_clv": None,
            "mean_price_clv": None,
        }
        invalid_counts = (
            ("binary_sample_size", 0),
            ("multiclass_sample_size", 2),
            ("priced_stake_count", 2),
            ("probability_clv_sample_size", 2),
            ("price_clv_sample_size", 1),
        )
        for field_name, value in invalid_counts:
            with self.subTest(field_name=field_name, value=value):
                invalid = dict(values)
                invalid[field_name] = value
                with self.assertRaisesRegex(ValueError, field_name):
                    CalibrationMetrics(**invalid)

        unavailable_cases = (
            {"multiclass_brier": 0.5},
            {"profit_units": 0.0, "roi": 0.0},
            {"mean_probability_clv": 0.1},
            {"mean_price_clv": 0.1},
        )
        for overrides in unavailable_cases:
            with self.subTest(overrides=overrides):
                invalid = dict(values)
                invalid.update(overrides)
                with self.assertRaisesRegex(ValueError, next(iter(overrides))):
                    CalibrationMetrics(**invalid)

        required_cases = (
            ({"multiclass_sample_size": 1}, "multiclass_brier"),
            ({"priced_stake_count": 1}, "profit_units"),
            ({"probability_clv_sample_size": 1}, "mean_probability_clv"),
            (
                {
                    "priced_stake_count": 1,
                    "profit_units": 0.0,
                    "roi": 0.0,
                    "price_clv_sample_size": 1,
                },
                "mean_price_clv",
            ),
        )
        for overrides, message in required_cases:
            with self.subTest(overrides=overrides):
                invalid = dict(values)
                invalid.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    CalibrationMetrics(**invalid)

    def test_roi_must_use_priced_stake_count_as_its_denominator(self):
        with self.assertRaisesRegex(ValueError, "roi"):
            CalibrationMetrics(
                sample_size=2,
                wins=1,
                losses=1,
                pushes=0,
                binary_sample_size=2,
                multiclass_sample_size=0,
                priced_stake_count=2,
                probability_clv_sample_size=0,
                price_clv_sample_size=0,
                hit_rate=0.5,
                push_rate=0.0,
                binary_brier=0.25,
                multiclass_brier=None,
                log_loss=math.log(2.0),
                profit_units=0.5,
                roi=0.3,
                mean_probability_clv=None,
                mean_price_clv=None,
            )

    def test_group_and_report_validate_fixed_partition_contract(self):
        metrics = build_calibration_report(
            (calibration_row(),),
            group_by=(),
        ).overall
        with self.assertRaisesRegex(ValueError, "unknown grouping"):
            CalibrationGroup(dimensions=(("unknown", "value"),), metrics=metrics)
        with self.assertRaisesRegex(ValueError, "partition"):
            CalibrationReport(
                overall=metrics,
                groups=(),
                group_by=("market_type",),
            )


class CalibrationMetricTests(unittest.TestCase):
    def test_push_is_excluded_from_binary_scores_and_zero_for_priced_roi(self):
        report = build_calibration_report(
            (
                calibration_row(
                    market_type="f5_side",
                    p_win=0.2,
                    p_push=0.5,
                    p_loss=0.3,
                    market_result="push",
                ),
            ),
            group_by=("market_type",),
        )

        metrics = report.groups[0].metrics
        self.assertEqual(metrics.pushes, 1)
        self.assertIsNone(metrics.hit_rate)
        self.assertEqual(metrics.push_rate, 1.0)
        self.assertIsNone(metrics.binary_brier)
        self.assertIsNone(metrics.log_loss)
        self.assertAlmostEqual(metrics.multiclass_brier, 0.38)
        self.assertEqual(metrics.profit_units, 0.0)
        self.assertEqual(metrics.roi, 0.0)

    def test_binary_scores_use_conditional_non_push_probability(self):
        report = build_calibration_report(
            (
                calibration_row(
                    market_type="f5_side",
                    p_win=0.45,
                    p_push=0.10,
                    p_loss=0.45,
                ),
            ),
            group_by=(),
        )

        self.assertAlmostEqual(report.overall.binary_brier, (0.50 - 1.0) ** 2)
        self.assertAlmostEqual(report.overall.log_loss, -math.log(0.50))

    def test_conditional_probability_uses_explicit_win_and_loss_mass(self):
        p_win = 0.45
        p_loss = 0.4500000000005
        report = build_calibration_report(
            (
                calibration_row(
                    market_type="f5_side",
                    p_win=p_win,
                    p_push=0.10,
                    p_loss=p_loss,
                ),
            ),
            group_by=(),
        )

        expected_probability = p_win / math.fsum((p_win, p_loss))
        self.assertEqual(
            report.overall.binary_brier,
            (expected_probability - 1.0) ** 2,
        )

    def test_log_loss_clips_endpoint_probabilities(self):
        report = build_calibration_report(
            (calibration_row(p_win=0.0, p_push=0.0, p_loss=1.0),),
            group_by=(),
        )

        self.assertAlmostEqual(report.overall.log_loss, -math.log(1e-15))

    def test_multiclass_brier_averages_all_three_outcomes(self):
        rows = (
            calibration_row(
                forecast_run_id="forecast-win",
                market_type="f5_side",
                p_win=0.6,
                p_push=0.1,
                p_loss=0.3,
                market_result="win",
            ),
            calibration_row(
                forecast_run_id="forecast-loss",
                market_type="f5_side",
                p_win=0.4,
                p_push=0.1,
                p_loss=0.5,
                market_result="loss",
            ),
        )

        report = build_calibration_report(rows, group_by=())

        expected = ((0.6 - 1) ** 2 + 0.1**2 + 0.3**2)
        expected += (0.4**2 + 0.1**2 + (0.5 - 1) ** 2)
        self.assertAlmostEqual(report.overall.multiclass_brier, expected / 2)
        self.assertEqual(report.overall.hit_rate, 0.5)

    def test_multiclass_brier_includes_every_push_capable_row(self):
        rows = (
            calibration_row(
                forecast_run_id="two-way-f5-side",
                market_type="f5_side",
                p_win=0.7,
                p_push=0.0,
                p_loss=0.3,
                market_result="win",
            ),
            calibration_row(
                forecast_run_id="integral-f5-total",
                market_type="f5_total",
                line=8.0,
                p_win=0.4,
                p_push=0.0,
                p_loss=0.6,
                market_result="loss",
            ),
            calibration_row(
                forecast_run_id="moneyline",
                market_type="ml",
                p_win=0.9,
                p_push=0.0,
                p_loss=0.1,
                market_result="win",
            ),
            calibration_row(
                forecast_run_id="half-run-f5-side",
                market_type="f5_side",
                line=0.5,
                p_win=0.8,
                p_push=0.0,
                p_loss=0.2,
                market_result="win",
            ),
        )

        metrics = build_calibration_report(rows, group_by=()).overall

        f5_side_brier = (0.7 - 1.0) ** 2 + 0.0**2 + 0.3**2
        f5_total_brier = 0.4**2 + 0.0**2 + (0.6 - 1.0) ** 2
        self.assertEqual(metrics.multiclass_sample_size, 2)
        self.assertAlmostEqual(
            metrics.multiclass_brier,
            (f5_side_brier + f5_total_brier) / 2,
        )

    def test_multiclass_brier_excludes_zero_push_probability_rows(self):
        eligible = calibration_row(
            forecast_run_id="push-capable",
            market_type="f5_side",
            p_win=0.4,
            p_push=0.2,
            p_loss=0.4,
            market_result="win",
        )
        binary_only = calibration_row(
            forecast_run_id="binary-only",
            p_win=0.9,
            p_push=0.0,
            p_loss=0.1,
            market_result="loss",
        )

        report = build_calibration_report(
            (binary_only, eligible),
            group_by=(),
        )

        expected = (0.4 - 1.0) ** 2 + 0.2**2 + 0.4**2
        self.assertAlmostEqual(report.overall.multiclass_brier, expected)

    def test_multiclass_brier_penalizes_zero_push_forecast_that_pushes(self):
        report = build_calibration_report(
            (
                calibration_row(
                    market_type="f5_side",
                    p_win=1.0,
                    p_push=0.0,
                    p_loss=0.0,
                    market_result="push",
                ),
            ),
            group_by=(),
        )

        self.assertEqual(report.overall.multiclass_sample_size, 1)
        self.assertEqual(report.overall.multiclass_brier, 2.0)

    def test_multiclass_brier_is_none_without_push_probability(self):
        report = build_calibration_report(
            (calibration_row(p_win=0.6, p_push=0.0, p_loss=0.4),),
            group_by=(),
        )

        self.assertIsNone(report.overall.multiclass_brier)

    def test_push_rate_uses_all_outcomes(self):
        rows = (
            calibration_row(forecast_run_id="push-rate-win"),
            calibration_row(
                forecast_run_id="push-rate-loss",
                market_result="loss",
            ),
            calibration_row(
                forecast_run_id="push-rate-push",
                market_type="f5_side",
                p_win=0.2,
                p_push=0.5,
                p_loss=0.3,
                market_result="push",
            ),
        )

        report = build_calibration_report(rows, group_by=())

        self.assertAlmostEqual(report.overall.push_rate, 1.0 / 3.0)

    def test_profit_and_roi_use_only_rows_with_priced_stakes(self):
        rows = (
            calibration_row(forecast_run_id="win", selected_decimal_price=2.2),
            calibration_row(
                forecast_run_id="loss",
                market_result="loss",
                selected_decimal_price=1.5,
            ),
            calibration_row(
                forecast_run_id="unpriced",
                market_result="win",
                selected_decimal_price=None,
            ),
        )

        report = build_calibration_report(rows, group_by=())

        self.assertAlmostEqual(report.overall.profit_units, 0.2)
        self.assertAlmostEqual(report.overall.roi, 0.1)

    def test_metric_denominators_track_each_eligible_subset(self):
        rows = (
            calibration_row(forecast_run_id="priced-win"),
            calibration_row(
                forecast_run_id="unpriced-loss",
                market_result="loss",
                selected_decimal_price=None,
                closing_selected_decimal_price=None,
                entry_no_vig_probability=None,
                closing_no_vig_probability=None,
            ),
            calibration_row(
                forecast_run_id="zero-push-actual-push",
                market_type="f5_side",
                p_win=0.6,
                p_push=0.0,
                p_loss=0.4,
                market_result="push",
                closing_selected_decimal_price=None,
                entry_no_vig_probability=None,
                closing_no_vig_probability=None,
            ),
            calibration_row(
                forecast_run_id="push-capable-win",
                market_type="f5_side",
                p_win=0.5,
                p_push=0.2,
                p_loss=0.3,
                selected_decimal_price=None,
                closing_selected_decimal_price=None,
                entry_no_vig_probability=0.50,
                closing_no_vig_probability=0.55,
            ),
        )

        metrics = build_calibration_report(rows, group_by=()).overall

        self.assertEqual(metrics.binary_sample_size, 3)
        self.assertEqual(metrics.multiclass_sample_size, 2)
        self.assertEqual(metrics.priced_stake_count, 2)
        self.assertEqual(metrics.probability_clv_sample_size, 2)
        self.assertEqual(metrics.price_clv_sample_size, 1)

    def test_clv_uses_matching_available_pairs_and_selected_direction(self):
        rows = (
            calibration_row(
                forecast_run_id="with-close",
                selected_decimal_price=2.0,
                closing_selected_decimal_price=1.8,
                entry_no_vig_probability=0.52,
                closing_no_vig_probability=0.56,
            ),
            calibration_row(
                forecast_run_id="without-close",
                closing_selected_decimal_price=None,
                closing_no_vig_probability=None,
            ),
        )

        report = build_calibration_report(rows, group_by=())

        self.assertAlmostEqual(report.overall.mean_probability_clv, 0.04)
        self.assertAlmostEqual(
            report.overall.mean_price_clv,
            (2.0 / 1.8) - 1.0,
        )

    def test_missing_prices_and_closes_produce_explicit_none_metrics(self):
        row = calibration_row(
            selected_decimal_price=None,
            closing_selected_decimal_price=None,
            entry_no_vig_probability=None,
            closing_no_vig_probability=None,
        )

        report = build_calibration_report((row,), group_by=())

        self.assertIsNone(report.overall.profit_units)
        self.assertIsNone(report.overall.roi)
        self.assertIsNone(report.overall.mean_probability_clv)
        self.assertIsNone(report.overall.mean_price_clv)

    def test_empty_report_has_zero_counts_and_unavailable_metrics(self):
        report = build_calibration_report((), group_by=())

        self.assertEqual(report.overall.sample_size, 0)
        self.assertEqual(report.overall.wins, 0)
        self.assertEqual(report.overall.losses, 0)
        self.assertEqual(report.overall.pushes, 0)
        self.assertEqual(report.overall.binary_sample_size, 0)
        self.assertEqual(report.overall.multiclass_sample_size, 0)
        self.assertEqual(report.overall.priced_stake_count, 0)
        self.assertEqual(report.overall.probability_clv_sample_size, 0)
        self.assertEqual(report.overall.price_clv_sample_size, 0)
        self.assertIsNone(report.overall.hit_rate)
        self.assertIsNone(report.overall.push_rate)
        self.assertIsNone(report.overall.multiclass_brier)
        self.assertIsNone(report.overall.roi)


class CalibrationGroupingTests(unittest.TestCase):
    def _fully_observed_report(self) -> CalibrationReport:
        rows = (
            calibration_row(
                forecast_run_id="partition-ml-win",
                market_type="ml",
                p_win=0.6,
                p_push=0.0,
                p_loss=0.4,
                market_result="win",
                selected_decimal_price=2.0,
                closing_selected_decimal_price=1.8,
                entry_no_vig_probability=0.52,
                closing_no_vig_probability=0.56,
            ),
            calibration_row(
                forecast_run_id="partition-total-loss",
                market_type="full_total",
                p_win=0.4,
                p_push=0.2,
                p_loss=0.4,
                market_result="loss",
                selected_decimal_price=1.9,
                closing_selected_decimal_price=2.0,
                entry_no_vig_probability=0.55,
                closing_no_vig_probability=0.50,
            ),
            calibration_row(
                forecast_run_id="partition-total-push",
                market_type="full_total",
                p_win=0.5,
                p_push=0.0,
                p_loss=0.5,
                market_result="push",
                selected_decimal_price=2.1,
                closing_selected_decimal_price=2.0,
                entry_no_vig_probability=0.50,
                closing_no_vig_probability=0.50,
            ),
        )
        return build_calibration_report(rows, group_by=("market_type",))

    def test_all_fixed_grouping_dimensions_are_supported_and_deterministic(self):
        self.assertEqual(
            GROUPING_DIMENSIONS,
            (
                "model_version",
                "market_type",
                "lineup_state",
                "starter_structure",
                "probability_bucket",
                "priority_score_bucket",
            ),
        )
        rows = (
            calibration_row(
                forecast_run_id="z",
                model_version="v2",
                market_type="ml",
                lineup_state="confirmed",
                starter_structure="starter_vs_starter",
                legacy_priority_score=80,
                p_win=0.54,
                p_push=0.0,
                p_loss=0.46,
            ),
            calibration_row(
                forecast_run_id="a",
                model_version="v1",
                market_type="f5_total",
                lineup_state="projected",
                starter_structure="opener_bulk_vs_starter",
                legacy_priority_score=74,
                p_win=0.60,
                p_push=0.20,
                p_loss=0.20,
            ),
        )

        report = build_calibration_report(rows, group_by=GROUPING_DIMENSIONS)

        self.assertEqual(report.group_by, GROUPING_DIMENSIONS)
        self.assertEqual(
            tuple(group.dimensions for group in report.groups),
            (
                (
                    ("model_version", "v1"),
                    ("market_type", "f5_total"),
                    ("lineup_state", "projected"),
                    ("starter_structure", "opener_bulk_vs_starter"),
                    ("probability_bucket", "0.60-1.00"),
                    ("priority_score_bucket", "70-74"),
                ),
                (
                    ("model_version", "v2"),
                    ("market_type", "ml"),
                    ("lineup_state", "confirmed"),
                    ("starter_structure", "starter_vs_starter"),
                    ("probability_bucket", "0.50-0.55"),
                    ("priority_score_bucket", "80-84"),
                ),
            ),
        )

    def test_unavailable_bucket_values_are_explicit(self):
        row = calibration_row(
            market_type="f5_side",
            legacy_priority_score=None,
            p_win=0.0,
            p_push=1.0,
            p_loss=0.0,
            market_result="push",
        )

        report = build_calibration_report(
            (row,),
            group_by=("probability_bucket", "priority_score_bucket"),
        )

        self.assertEqual(
            report.groups[0].dimensions,
            (
                ("probability_bucket", "unavailable"),
                ("priority_score_bucket", "unavailable"),
            ),
        )

    def test_group_by_rejects_unknown_duplicate_or_non_tuple_dimensions(self):
        for group_by, message in (
            (("unknown",), "unknown"),
            (("market_type", "market_type"), "unique"),
            (["market_type"], "tuple"),
        ):
            with self.subTest(group_by=group_by):
                with self.assertRaisesRegex(ValueError, message):
                    build_calibration_report((calibration_row(),), group_by=group_by)

    def test_report_rejects_outcome_and_denominator_partition_mismatches(self):
        report = self._fully_observed_report()
        full_total_group = report.groups[0]
        self.assertEqual(
            full_total_group.dimensions,
            (("market_type", "full_total"),),
        )
        mismatches = (
            (
                "wins",
                {
                    "wins": 1,
                    "losses": 0,
                    "hit_rate": 1.0,
                },
            ),
            (
                "multiclass_sample_size",
                {"multiclass_sample_size": 1},
            ),
            (
                "priced_stake_count",
                {
                    "priced_stake_count": 1,
                    "price_clv_sample_size": 1,
                    "roi": -1.0,
                },
            ),
            (
                "probability_clv_sample_size",
                {"probability_clv_sample_size": 1},
            ),
            (
                "price_clv_sample_size",
                {"price_clv_sample_size": 1},
            ),
        )
        for field_name, overrides in mismatches:
            with self.subTest(field_name=field_name):
                invalid_group = replace(
                    full_total_group,
                    metrics=replace(full_total_group.metrics, **overrides),
                )
                with self.assertRaisesRegex(ValueError, field_name):
                    CalibrationReport(
                        overall=report.overall,
                        groups=(invalid_group, *report.groups[1:]),
                        group_by=report.group_by,
                    )

    def test_report_reconstructs_weighted_metrics_and_additive_profit(self):
        report = self._fully_observed_report()
        metric_overrides = (
            ("binary_brier", report.overall.binary_brier + 0.01),
            ("multiclass_brier", report.overall.multiclass_brier + 0.01),
            ("log_loss", report.overall.log_loss + 0.01),
            (
                "mean_probability_clv",
                report.overall.mean_probability_clv + 0.01,
            ),
            ("mean_price_clv", report.overall.mean_price_clv + 0.01),
        )
        for field_name, value in metric_overrides:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    CalibrationReport(
                        overall=replace(report.overall, **{field_name: value}),
                        groups=report.groups,
                        group_by=report.group_by,
                    )

        profit_units = report.overall.profit_units + 0.1
        with self.assertRaisesRegex(ValueError, "profit_units"):
            CalibrationReport(
                overall=replace(
                    report.overall,
                    profit_units=profit_units,
                    roi=profit_units / report.overall.priced_stake_count,
                ),
                groups=report.groups,
                group_by=report.group_by,
            )


class CalibrationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.store = LedgerStore(
            Path(self._temporary_directory.name) / "ledger.jsonl"
        )

    def _complete_reviews(
        self,
        game_pk: int,
        contexts: tuple[tuple[object, object, object], ...],
        *,
        market_results: tuple[str, ...] | None = None,
        market_type: str = "ml",
    ):
        append_entry_market(
            self.store,
            game_pk=game_pk,
            id_suffix="shared",
            market_type=market_type,
            line=8.5 if market_type == "full_total" else None,
            selected_side="over" if market_type == "full_total" else "away",
        )
        result = append_result(self.store, game_pk=game_pk)
        results = market_results or tuple("win" for _ in contexts)
        for index, ((capture, forecast, link), market_result) in enumerate(
            zip(contexts, results)
        ):
            append_review(
                self.store,
                capture=capture,
                forecast=forecast,
                link=link,
                result=result,
                market_result=market_result,
                market_type=market_type,
                id_suffix=str(index),
            )
        return result

    def test_selector_loads_ledger_once_without_loop_time_store_queries(self):
        for index in range(16):
            game_pk = 920000 + index
            context = append_capture_link(
                self.store,
                game_pk=game_pk,
                captured_at=FIRST_PITCH - timedelta(hours=1),
                lineup_state="confirmed",
                forecast_run_id=f"scaled-run-{index}",
                id_suffix="capture",
            )
            self._complete_reviews(game_pk, (context,))
        instrumented = InstrumentedLedgerView(self.store)

        rows = select_primary_rows(instrumented)

        self.assertEqual(len(rows), 16)
        self.assertEqual(
            instrumented.calls,
            {
                "iter_records": 1,
                "records_superseded_by": 0,
                "latest_official_result": 0,
                "links_for_capture": 0,
                "forecast_for_link": 0,
            },
        )

    def test_unchanged_repeated_capture_counts_once_after_primary_selection(self):
        first = append_capture_link(
            self.store,
            game_pk=910001,
            captured_at=FIRST_PITCH - timedelta(hours=3),
            lineup_state="confirmed",
            forecast_run_id="unchanged-run",
            id_suffix="first",
        )
        second = append_capture_link(
            self.store,
            game_pk=910001,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="unchanged-run",
            id_suffix="second",
        )
        self._complete_reviews(910001, (first, second))

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].capture_id, second[0].capture_id)
        self.assertEqual(rows[0].forecast_run_id, "unchanged-run")

    def test_latest_changed_pregame_capture_is_primary(self):
        projected = append_capture_link(
            self.store,
            game_pk=910002,
            captured_at=FIRST_PITCH - timedelta(hours=7),
            lineup_state="projected",
            forecast_run_id="projected-run",
            id_suffix="projected",
        )
        confirmed = append_capture_link(
            self.store,
            game_pk=910002,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="confirmed-run",
            id_suffix="confirmed",
            supersedes_capture_id=projected[0].capture_id,
        )
        self._complete_reviews(910002, (projected, confirmed))

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].lineup_state, "confirmed")
        self.assertEqual(rows[0].forecast_run_id, "confirmed-run")

    def test_named_time_slice_can_ignore_successor_after_boundary(self):
        projected = append_capture_link(
            self.store,
            game_pk=910003,
            captured_at=FIRST_PITCH - timedelta(hours=7),
            lineup_state="projected",
            forecast_run_id="projected-slice-run",
            id_suffix="projected",
        )
        confirmed = append_capture_link(
            self.store,
            game_pk=910003,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="confirmed-slice-run",
            id_suffix="confirmed",
            supersedes_capture_id=projected[0].capture_id,
        )
        self._complete_reviews(910003, (projected, confirmed))
        policy = CaptureCutoffPolicy(
            name="projected_lineup_6h_before",
            minimum_seconds_before_first_pitch=21_600,
            required_lineup_state="projected",
            include_later_superseded_captures=True,
        )

        rows = select_primary_rows(self.store, policy)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].lineup_state, "projected")
        self.assertEqual(rows[0].forecast_run_id, "projected-slice-run")

    def test_named_time_slice_rejects_successor_at_boundary_despite_lineup(self):
        projected = append_capture_link(
            self.store,
            game_pk=910013,
            captured_at=FIRST_PITCH - timedelta(hours=8),
            lineup_state="projected",
            forecast_run_id="stale-boundary-run",
            id_suffix="projected",
        )
        confirmed = append_capture_link(
            self.store,
            game_pk=910013,
            captured_at=FIRST_PITCH - timedelta(hours=6),
            lineup_state="confirmed",
            forecast_run_id="boundary-successor-run",
            id_suffix="confirmed",
            supersedes_capture_id=projected[0].capture_id,
        )
        self._complete_reviews(910013, (projected, confirmed))
        policy = CaptureCutoffPolicy(
            name="projected_lineup_6h_before",
            minimum_seconds_before_first_pitch=21_600,
            required_lineup_state="projected",
            include_later_superseded_captures=True,
        )

        self.assertEqual(select_primary_rows(self.store, policy), ())

    def test_named_time_slice_rejects_successor_before_boundary(self):
        projected = append_capture_link(
            self.store,
            game_pk=910015,
            captured_at=FIRST_PITCH - timedelta(hours=9),
            lineup_state="projected",
            forecast_run_id="stale-before-boundary-run",
            id_suffix="projected",
        )
        confirmed = append_capture_link(
            self.store,
            game_pk=910015,
            captured_at=FIRST_PITCH - timedelta(hours=7),
            lineup_state="confirmed",
            forecast_run_id="before-boundary-successor-run",
            id_suffix="confirmed",
            out_of_sample_eligible=False,
            supersedes_capture_id=projected[0].capture_id,
        )
        self._complete_reviews(910015, (projected, confirmed))
        policy = CaptureCutoffPolicy(
            name="projected_lineup_6h_before",
            minimum_seconds_before_first_pitch=21_600,
            required_lineup_state="projected",
            include_later_superseded_captures=True,
        )

        self.assertEqual(select_primary_rows(self.store, policy), ())

    def test_named_time_slice_checks_every_successor_in_chain(self):
        projected = append_capture_link(
            self.store,
            game_pk=910014,
            captured_at=FIRST_PITCH - timedelta(hours=9),
            lineup_state="projected",
            forecast_run_id="stale-chain-run",
            id_suffix="projected",
        )
        later_successor = append_capture_link(
            self.store,
            game_pk=910014,
            captured_at=FIRST_PITCH - timedelta(hours=5),
            lineup_state="confirmed",
            forecast_run_id="later-chain-run",
            id_suffix="later",
            supersedes_capture_id=projected[0].capture_id,
        )
        boundary_successor = append_capture_link(
            self.store,
            game_pk=910014,
            captured_at=FIRST_PITCH - timedelta(hours=6),
            lineup_state="confirmed",
            forecast_run_id="boundary-chain-run",
            id_suffix="boundary",
            supersedes_capture_id=later_successor[0].capture_id,
        )
        self._complete_reviews(
            910014,
            (projected, later_successor, boundary_successor),
        )
        policy = CaptureCutoffPolicy(
            name="projected_lineup_6h_before",
            minimum_seconds_before_first_pitch=21_600,
            required_lineup_state="projected",
            include_later_superseded_captures=True,
        )

        self.assertEqual(select_primary_rows(self.store, policy), ())

    def test_post_pitch_and_ineligible_successors_are_rejected(self):
        eligible = append_capture_link(
            self.store,
            game_pk=910004,
            captured_at=FIRST_PITCH - timedelta(hours=2),
            lineup_state="confirmed",
            forecast_run_id="eligible-run",
            id_suffix="eligible",
        )
        ineligible = append_capture_link(
            self.store,
            game_pk=910004,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="ineligible-run",
            id_suffix="ineligible",
            out_of_sample_eligible=False,
            supersedes_capture_id=eligible[0].capture_id,
        )
        post_pitch = append_capture_link(
            self.store,
            game_pk=910004,
            captured_at=FIRST_PITCH + timedelta(seconds=1),
            lineup_state="confirmed",
            forecast_run_id="post-pitch-run",
            id_suffix="post-pitch",
            supersedes_capture_id=ineligible[0].capture_id,
        )
        self._complete_reviews(910004, (eligible, ineligible, post_pitch))

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].forecast_run_id, "eligible-run")

    def test_first_pitch_boundary_is_ineligible(self):
        boundary = append_capture_link(
            self.store,
            game_pk=910005,
            captured_at=FIRST_PITCH,
            lineup_state="confirmed",
            forecast_run_id="boundary-run",
            id_suffix="boundary",
        )
        self._complete_reviews(910005, (boundary,))

        self.assertEqual(select_primary_rows(self.store), ())

    def test_equal_capture_timestamps_use_lexicographically_smallest_id(self):
        first = append_capture_link(
            self.store,
            game_pk=910006,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="tie-run-a",
            id_suffix="a",
        )
        second = append_capture_link(
            self.store,
            game_pk=910006,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="tie-run-b",
            id_suffix="b",
        )
        self._complete_reviews(910006, (first, second))

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].capture_id,
            min(first[0].capture_id, second[0].capture_id),
        )

    def test_exact_link_and_latest_official_result_revision_are_joined(self):
        context = append_capture_link(
            self.store,
            game_pk=910007,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="corrected-result-run",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910007, id_suffix="entry")
        first_result = append_result(self.store, game_pk=910007)
        append_review(
            self.store,
            capture=context[0],
            forecast=context[1],
            link=context[2],
            result=first_result,
            market_result="win",
            id_suffix="old",
        )
        corrected_result = append_result(
            self.store,
            game_pk=910007,
            revision="2",
            supersedes_record_id=first_result.envelope.record_id,
        )
        append_review(
            self.store,
            capture=context[0],
            forecast=context[1],
            link=context[2],
            result=corrected_result,
            market_result="loss",
            id_suffix="corrected",
        )

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].market_result, "loss")

    def test_review_for_superseded_result_is_not_used_as_latest(self):
        context = append_capture_link(
            self.store,
            game_pk=910008,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="missing-latest-review-run",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910008, id_suffix="entry")
        first_result = append_result(self.store, game_pk=910008)
        append_review(
            self.store,
            capture=context[0],
            forecast=context[1],
            link=context[2],
            result=first_result,
            id_suffix="old",
        )
        append_result(
            self.store,
            game_pk=910008,
            revision="2",
            supersedes_record_id=first_result.envelope.record_id,
        )

        self.assertEqual(select_primary_rows(self.store), ())

    def test_identical_forecast_market_across_all_roots_is_accepted(self):
        game_pk = 910028
        outputs = forecast_outputs(game_pk)
        nested_market = outputs["forecast_output"]["markets"][market_key(game_pk)]
        outputs["markets"] = {
            market_key(game_pk): dict(reversed(tuple(nested_market.items())))
        }
        outputs["forecast"] = {
            "markets": {market_key(game_pk): dict(nested_market)}
        }
        context = append_capture_link(
            self.store,
            game_pk=game_pk,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="identical-forecast-roots",
            id_suffix="capture",
            forecast_outputs_override=outputs,
        )
        self._complete_reviews(game_pk, (context,))

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].p_win, 0.55)

    def test_conflicting_forecast_market_in_later_root_raises(self):
        game_pk = 910029
        outputs = forecast_outputs(game_pk)
        nested_market = outputs["forecast_output"]["markets"][market_key(game_pk)]
        outputs["markets"] = {market_key(game_pk): dict(nested_market)}
        conflicting_market = dict(nested_market)
        conflicting_market["probabilities"] = {
            "win": 0.60,
            "push": 0.0,
            "loss": 0.40,
        }
        outputs["forecast"] = {
            "markets": {market_key(game_pk): conflicting_market}
        }
        context = append_capture_link(
            self.store,
            game_pk=game_pk,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-forecast-roots",
            id_suffix="capture",
            forecast_outputs_override=outputs,
        )
        self._complete_reviews(game_pk, (context,))

        with self.assertRaisesRegex(ValueError, "conflicting forecast market"):
            select_primary_rows(self.store)

    def test_present_null_forecast_market_cannot_fallback_to_another_root(self):
        game_pk = 910034
        outputs = forecast_outputs(game_pk)
        outputs["markets"] = {market_key(game_pk): None}
        context = append_capture_link(
            self.store,
            game_pk=game_pk,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="null-forecast-market",
            id_suffix="capture",
            forecast_outputs_override=outputs,
        )
        self._complete_reviews(game_pk, (context,))

        with self.assertRaisesRegex(ValueError, "forecast market.*mapping"):
            select_primary_rows(self.store)

    def test_exact_duplicate_reviews_are_deduplicated_semantically(self):
        context = append_capture_link(
            self.store,
            game_pk=910030,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="duplicate-review",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910030, id_suffix="entry")
        result = append_result(self.store, game_pk=910030)
        for id_suffix in ("a", "b"):
            append_review(
                self.store,
                capture=context[0],
                forecast=context[1],
                link=context[2],
                result=result,
                id_suffix=id_suffix,
            )

        normal = select_primary_rows(self.store)
        reordered = select_primary_rows(ReorderedLedgerView(self.store))

        self.assertEqual(normal, reordered)
        self.assertEqual(len(normal), 1)

    def test_duplicate_reviews_with_conflicting_results_raise(self):
        context = append_capture_link(
            self.store,
            game_pk=910031,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-review-result",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910031, id_suffix="entry")
        result = append_result(self.store, game_pk=910031)
        for id_suffix, market_result in (("win", "win"), ("loss", "loss")):
            append_review(
                self.store,
                capture=context[0],
                forecast=context[1],
                link=context[2],
                result=result,
                id_suffix=id_suffix,
                market_result=market_result,
            )

        with self.assertRaisesRegex(ValueError, "conflicting review records"):
            select_primary_rows(self.store)

    def test_duplicate_reviews_with_conflicting_classifications_raise(self):
        context = append_capture_link(
            self.store,
            game_pk=910032,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-review-classification",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910032, id_suffix="entry")
        result = append_result(self.store, game_pk=910032)
        for classification, inside_80_ratio in (("A", 0.75), ("B", 0.70)):
            append_review(
                self.store,
                capture=context[0],
                forecast=context[1],
                link=context[2],
                result=result,
                id_suffix=classification,
                classification=classification,
                inside_80_ratio=inside_80_ratio,
            )

        with self.assertRaisesRegex(ValueError, "conflicting review records"):
            select_primary_rows(self.store)

    def test_duplicate_reviews_with_conflicting_component_reviews_raise(self):
        context = append_capture_link(
            self.store,
            game_pk=910033,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-review-component",
            id_suffix="capture",
        )
        append_entry_market(self.store, game_pk=910033, id_suffix="entry")
        result = append_result(self.store, game_pk=910033)
        for classification in ("A", "B"):
            append_review(
                self.store,
                capture=context[0],
                forecast=context[1],
                link=context[2],
                result=result,
                id_suffix=classification,
                component_reviews=(
                    {
                        "evidence_path": "away_lineup.power",
                        "component_ids": ("away_full_runs",),
                        "contradicted": False,
                        "inside_80_count": 1,
                        "component_count": 1,
                        "notes": (classification,),
                    },
                ),
            )

        with self.assertRaisesRegex(ValueError, "conflicting review records"):
            select_primary_rows(self.store)

    def test_close_must_match_selected_market_contract(self):
        context = append_capture_link(
            self.store,
            game_pk=910009,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="close-match-run",
            id_suffix="capture",
            market_type="full_total",
            line=8.5,
            selected_side="over",
            away_roles=("opener", "bulk"),
        )
        self._complete_reviews(910009, (context,), market_type="full_total")
        append_close_market(
            self.store,
            game_pk=910009,
            id_suffix="matching",
            observed_at=FIRST_PITCH - timedelta(minutes=5),
            selected_price=1.8,
            opposing_price=2.1,
            market_type="full_total",
            line=8.5,
            selected_side="over",
        )
        append_close_market(
            self.store,
            game_pk=910009,
            id_suffix="wrong-line",
            observed_at=FIRST_PITCH - timedelta(minutes=1),
            selected_price=1.7,
            opposing_price=2.2,
            market_type="full_total",
            line=9.0,
            selected_side="over",
        )

        rows = select_primary_rows(self.store)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].market_type, "full_total")
        self.assertEqual(rows[0].starter_structure, "opener_bulk_vs_starter")
        self.assertEqual(rows[0].closing_selected_decimal_price, 1.8)
        expected_no_vig = (1.0 / 1.8) / ((1.0 / 1.8) + (1.0 / 2.1))
        self.assertAlmostEqual(rows[0].closing_no_vig_probability, expected_no_vig)

    def test_latest_non_superseded_matching_close_is_used(self):
        context = append_capture_link(
            self.store,
            game_pk=910010,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="close-revision-run",
            id_suffix="capture",
        )
        self._complete_reviews(910010, (context,))
        first_close = append_close_market(
            self.store,
            game_pk=910010,
            id_suffix="1",
            observed_at=FIRST_PITCH - timedelta(minutes=5),
            selected_price=1.9,
            opposing_price=2.0,
        )
        append_close_market(
            self.store,
            game_pk=910010,
            id_suffix="2",
            observed_at=FIRST_PITCH - timedelta(minutes=2),
            selected_price=1.85,
            opposing_price=2.05,
            supersedes_record_id=first_close.envelope.record_id,
        )

        rows = select_primary_rows(self.store)

        self.assertEqual(rows[0].closing_selected_decimal_price, 1.85)

    def test_close_requires_observation_and_availability_by_first_pitch(self):
        invalid_times = (
            (
                "future-observation",
                FIRST_PITCH + timedelta(minutes=1),
                FIRST_PITCH - timedelta(minutes=1),
            ),
            (
                "missing-availability",
                FIRST_PITCH - timedelta(minutes=1),
                None,
            ),
            (
                "future-availability",
                FIRST_PITCH - timedelta(minutes=1),
                FIRST_PITCH + timedelta(minutes=1),
            ),
        )
        for index, (name, observed_at, available_at) in enumerate(invalid_times):
            with self.subTest(name=name):
                game_pk = 910020 + index
                context = append_capture_link(
                    self.store,
                    game_pk=game_pk,
                    captured_at=FIRST_PITCH - timedelta(hours=1),
                    lineup_state="confirmed",
                    forecast_run_id=f"close-time-{index}",
                    id_suffix="capture",
                )
                self._complete_reviews(game_pk, (context,))
                append_close_market(
                    self.store,
                    game_pk=game_pk,
                    id_suffix="eligible",
                    observed_at=FIRST_PITCH - timedelta(minutes=10),
                    selected_price=1.95,
                    opposing_price=1.95,
                )
                append_close_market(
                    self.store,
                    game_pk=game_pk,
                    id_suffix="ineligible",
                    observed_at=observed_at,
                    available_at=available_at,
                    selected_price=1.70,
                    opposing_price=2.20,
                    source_record=f"invalid-close:{game_pk}",
                )

                row = next(
                    row
                    for row in select_primary_rows(self.store)
                    if row.game_pk == game_pk
                )
                self.assertEqual(row.closing_selected_decimal_price, 1.95)

    def test_close_chronology_starts_at_capture_timestamp(self):
        capture_time = FIRST_PITCH - timedelta(hours=1)
        invalid_times = (
            (
                "observation-before-capture",
                capture_time - timedelta(seconds=1),
                capture_time + timedelta(seconds=1),
            ),
            (
                "availability-before-capture",
                capture_time + timedelta(seconds=1),
                capture_time - timedelta(seconds=1),
            ),
        )
        for index, (name, observed_at, available_at) in enumerate(invalid_times):
            with self.subTest(name=name):
                game_pk = 910035 + index
                context = append_capture_link(
                    self.store,
                    game_pk=game_pk,
                    captured_at=capture_time,
                    lineup_state="confirmed",
                    forecast_run_id=f"pre-capture-close-{index}",
                    id_suffix="capture",
                )
                self._complete_reviews(game_pk, (context,))
                append_close_market(
                    self.store,
                    game_pk=game_pk,
                    id_suffix="ineligible",
                    observed_at=observed_at,
                    available_at=available_at,
                    selected_price=1.70,
                    opposing_price=2.20,
                )

                row = next(
                    row
                    for row in select_primary_rows(self.store)
                    if row.game_pk == game_pk
                )
                self.assertIsNone(row.closing_selected_decimal_price)

        boundary_game_pk = 910037
        boundary_context = append_capture_link(
            self.store,
            game_pk=boundary_game_pk,
            captured_at=capture_time,
            lineup_state="confirmed",
            forecast_run_id="capture-boundary-close",
            id_suffix="capture",
        )
        self._complete_reviews(boundary_game_pk, (boundary_context,))
        append_close_market(
            self.store,
            game_pk=boundary_game_pk,
            id_suffix="boundary",
            observed_at=capture_time,
            available_at=capture_time,
            selected_price=1.88,
            opposing_price=2.02,
        )

        boundary_row = next(
            row
            for row in select_primary_rows(self.store)
            if row.game_pk == boundary_game_pk
        )
        self.assertEqual(boundary_row.closing_selected_decimal_price, 1.88)

    def test_successor_helpers_validate_cycle_after_matching_successor(self):
        first = append_capture_link(
            self.store,
            game_pk=910038,
            captured_at=FIRST_PITCH - timedelta(hours=2),
            lineup_state="projected",
            forecast_run_id="cycle-first",
            id_suffix="first",
        )[0]
        second = append_capture_link(
            self.store,
            game_pk=910038,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="cycle-second",
            id_suffix="second",
        )[0]
        first_record_id = first.envelope.record_id
        second_record_id = second.envelope.record_id
        relationships = {
            first_record_id: second_record_id,
            second_record_id: first_record_id,
        }

        with self.assertRaisesRegex(ValueError, "cycles"):
            calibration_module._has_eligible_successor(
                first_record_id,
                relationships,
                frozenset((second_record_id,)),
            )
        with self.assertRaisesRegex(ValueError, "cycles"):
            calibration_module._has_successor_at_or_before_boundary(
                first_record_id,
                relationships,
                {
                    first_record_id: first,
                    second_record_id: second,
                },
                FIRST_PITCH,
            )

    def test_post_pitch_successor_does_not_hide_eligible_close(self):
        context = append_capture_link(
            self.store,
            game_pk=910023,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="close-future-revision",
            id_suffix="capture",
        )
        self._complete_reviews(910023, (context,))
        eligible = append_close_market(
            self.store,
            game_pk=910023,
            id_suffix="1",
            observed_at=FIRST_PITCH - timedelta(minutes=5),
            selected_price=1.90,
            opposing_price=2.00,
        )
        append_close_market(
            self.store,
            game_pk=910023,
            id_suffix="2",
            observed_at=FIRST_PITCH + timedelta(minutes=1),
            selected_price=1.70,
            opposing_price=2.20,
            supersedes_record_id=eligible.envelope.record_id,
        )

        rows = select_primary_rows(self.store)

        self.assertEqual(rows[0].closing_selected_decimal_price, 1.90)

    def test_close_matches_semantic_market_terms_without_identical_market_key(self):
        context = append_capture_link(
            self.store,
            game_pk=910024,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="semantic-close-key",
            id_suffix="capture",
            market_type="full_total",
            line=8.5,
            selected_side="over",
        )
        self._complete_reviews(910024, (context,), market_type="full_total")
        append_close_market(
            self.store,
            game_pk=910024,
            id_suffix="semantic",
            observed_at=FIRST_PITCH - timedelta(minutes=2),
            selected_price=1.82,
            opposing_price=2.08,
            market_type="full_total",
            line=8.5,
            selected_side="over",
            close_market_key="sportsbook:rotation:910024:total:over",
        )

        rows = select_primary_rows(self.store)

        self.assertEqual(rows[0].closing_selected_decimal_price, 1.82)

    def test_identical_latest_close_quotes_are_canonicalized(self):
        context = append_capture_link(
            self.store,
            game_pk=910025,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="duplicate-close-quotes",
            id_suffix="capture",
        )
        self._complete_reviews(910025, (context,))
        for source_name in ("z-book", "a-book"):
            append_close_market(
                self.store,
                game_pk=910025,
                id_suffix="1",
                observed_at=FIRST_PITCH - timedelta(minutes=2),
                selected_price=1.84,
                opposing_price=2.06,
                source_name=source_name,
            )

        normal = select_primary_rows(self.store)
        reordered = select_primary_rows(ReorderedLedgerView(self.store))

        self.assertEqual(normal, reordered)
        self.assertEqual(normal[0].closing_selected_decimal_price, 1.84)

    def test_conflicting_latest_close_selected_prices_raise(self):
        context = append_capture_link(
            self.store,
            game_pk=910026,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-close-price",
            id_suffix="capture",
        )
        self._complete_reviews(910026, (context,))
        for source_name, selected_price in (("a-book", 1.80), ("b-book", 1.90)):
            append_close_market(
                self.store,
                game_pk=910026,
                id_suffix="1",
                observed_at=FIRST_PITCH - timedelta(minutes=2),
                selected_price=selected_price,
                opposing_price=2.05,
                source_name=source_name,
            )

        with self.assertRaisesRegex(ValueError, "conflicting closing quotes"):
            select_primary_rows(self.store)

    def test_conflicting_latest_close_no_vig_inputs_raise(self):
        context = append_capture_link(
            self.store,
            game_pk=910027,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="conflicting-close-no-vig",
            id_suffix="capture",
        )
        self._complete_reviews(910027, (context,))
        for source_name, opposing_price in (("a-book", 2.00), ("b-book", 2.20)):
            append_close_market(
                self.store,
                game_pk=910027,
                id_suffix="1",
                observed_at=FIRST_PITCH - timedelta(minutes=2),
                selected_price=1.85,
                opposing_price=opposing_price,
                source_name=source_name,
            )

        with self.assertRaisesRegex(ValueError, "conflicting closing quotes"):
            select_primary_rows(self.store)

    def test_missing_matching_close_is_explicitly_unavailable(self):
        context = append_capture_link(
            self.store,
            game_pk=910011,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="no-close-run",
            id_suffix="capture",
        )
        self._complete_reviews(910011, (context,))

        rows = select_primary_rows(self.store)

        self.assertIsNone(rows[0].closing_selected_decimal_price)
        self.assertIsNone(rows[0].closing_no_vig_probability)

    def test_record_iteration_order_does_not_change_selection(self):
        earlier = append_capture_link(
            self.store,
            game_pk=910012,
            captured_at=FIRST_PITCH - timedelta(hours=3),
            lineup_state="projected",
            forecast_run_id="order-earlier-run",
            id_suffix="earlier",
        )
        later = append_capture_link(
            self.store,
            game_pk=910012,
            captured_at=FIRST_PITCH - timedelta(hours=1),
            lineup_state="confirmed",
            forecast_run_id="order-later-run",
            id_suffix="later",
            supersedes_capture_id=earlier[0].capture_id,
        )
        self._complete_reviews(910012, (earlier, later))

        normal = select_primary_rows(self.store)
        reordered = select_primary_rows(ReorderedLedgerView(self.store))

        self.assertEqual(reordered, normal)


if __name__ == "__main__":
    unittest.main()
