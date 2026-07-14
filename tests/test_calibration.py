from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import unittest

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


def calibration_row(**overrides: object) -> CalibrationRow:
    values: dict[str, object] = {
        "game_pk": 900001,
        "capture_id": "capture-1",
        "forecast_run_id": "forecast-1",
        "market_key": "game:900001:ml:away",
        "model_version": "forecast-v1",
        "market_type": "ml",
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
            outputs=forecast_outputs(
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
) -> None:
    store.append_review(
        ReviewCommand(
            capture_forecast_link_record_id=link.envelope.record_id,
            capture_id=capture.capture_id,
            forecast_run_id=forecast.forecast_run_id,
            official_result_record_id=result.envelope.record_id,
            market_key=market_key(capture.game_pk, market_type),
            market_result=market_result,
            classification="A",
            inside_80_ratio=1.0,
            contradicted_paths=(),
            component_reviews=(),
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
    selected_price: float,
    opposing_price: float,
    market_type: str = "ml",
    line: float | None = None,
    selected_side: str = "away",
    supersedes_record_id: str | None = None,
):
    return store.append_market_observation(
        MarketObservationCommand(
            game_pk=game_pk,
            source_name="book",
            source_record=f"close:{game_pk}",
            source_revision=id_suffix,
            observation_kind="close",
            market_key=market_key(game_pk, market_type),
            market_type=market_type,
            settlement_format="two_way",
            selected_side=selected_side,
            line=line,
            selected_decimal_price=selected_price,
            opposing_decimal_price=opposing_price,
            observed_at=observed_at,
            available_at=observed_at,
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

    def test_metric_and_group_types_are_frozen_and_validated(self):
        metrics = CalibrationMetrics(
            sample_size=1,
            wins=1,
            losses=0,
            pushes=0,
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
            hit_rate=None,
            push_rate=0.0,
            binary_brier=None,
            multiclass_brier=None,
            log_loss=None,
        )
        with self.assertRaisesRegex(ValueError, "push_rate"):
            CalibrationMetrics(**empty)

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
                    p_win=0.45,
                    p_push=0.10,
                    p_loss=0.45,
                ),
            ),
            group_by=(),
        )

        self.assertAlmostEqual(report.overall.binary_brier, (0.50 - 1.0) ** 2)
        self.assertAlmostEqual(report.overall.log_loss, -math.log(0.50))

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
                p_win=0.6,
                p_push=0.1,
                p_loss=0.3,
                market_result="win",
            ),
            calibration_row(
                forecast_run_id="forecast-loss",
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

    def test_multiclass_brier_excludes_zero_push_probability_rows(self):
        eligible = calibration_row(
            forecast_run_id="push-capable",
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
        self.assertIsNone(report.overall.hit_rate)
        self.assertIsNone(report.overall.push_rate)
        self.assertIsNone(report.overall.multiclass_brier)
        self.assertIsNone(report.overall.roi)


class CalibrationGroupingTests(unittest.TestCase):
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
