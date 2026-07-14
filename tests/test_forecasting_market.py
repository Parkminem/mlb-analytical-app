from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
import math
import sys
from types import MappingProxyType
import unittest

from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import (
    FeatureProvenance,
    MarketInput,
    MarketType,
    SettlementFormat,
)
from analysis.forecasting.simulation import SimulatedGame, SimulationResult


try:
    from analysis.forecasting.market import (
        MarketEvaluation,
        OutcomeProbabilities,
        evaluate_market,
        price_market,
    )
except ModuleNotFoundError:
    MarketEvaluation = None
    OutcomeProbabilities = None
    evaluate_market = None
    price_market = None


MARKET_API_AVAILABLE = all(
    value is not None
    for value in (
        MarketEvaluation,
        OutcomeProbabilities,
        evaluate_market,
        price_market,
    )
)
UTC = timezone.utc
PROVENANCE = FeatureProvenance(
    observation_ids=("obs:market",),
    latest_available_at=datetime(2026, 7, 14, 1, 0, tzinfo=UTC),
    lookback_end=datetime(2026, 7, 14, 1, 0, tzinfo=UTC),
)


def _market(
    market_type,
    selected_side,
    *,
    line,
    selected_price=1.91,
    opposing_price=1.91,
    settlement_format=SettlementFormat.TWO_WAY,
):
    return MarketInput(
        market_key=f"test:{market_type.value}:{selected_side}",
        market_type=market_type,
        settlement_format=settlement_format,
        selected_side=selected_side,
        line=line,
        selected_decimal_price=selected_price,
        opposing_decimal_price=opposing_price,
        provenance=PROVENANCE,
    )


def _game(path):
    away_f5, home_f5, away_full, home_full = path
    constructed_home_full = home_full if away_full != home_full else home_full + 1
    home_margin = constructed_home_full - away_full
    game = SimulatedGame(
        away_f5_runs=away_f5,
        home_f5_runs=home_f5,
        away_full_runs=away_full,
        home_full_runs=constructed_home_full,
        away_f5_traffic=0,
        home_f5_traffic=0,
        away_full_traffic=0,
        home_full_traffic=0,
        away_f5_extra_base_hits=0,
        home_f5_extra_base_hits=0,
        away_full_extra_base_hits=0,
        home_full_extra_base_hits=0,
        away_pitcher_chain_runs_allowed=constructed_home_full,
        home_pitcher_chain_runs_allowed=away_full,
        away_bullpen_runs_allowed=0,
        home_bullpen_runs_allowed=0,
        away_pitcher_chain_workload_outs=15,
        home_pitcher_chain_workload_outs=15,
        away_bullpen_segment_ids_exposed=(),
        home_bullpen_segment_ids_exposed=(),
        away_protected_bullpen_close_entry=False,
        home_protected_bullpen_close_entry=False,
        bottom_ninth_played=away_full > constructed_home_full or home_margin == 1,
        final_inning=9,
        max_inning_tiebreak_used=False,
    )
    if away_full == home_full:
        # The approved settlement fixture predates the simulator's winner-only
        # invariant. Preserve that exact fixed path only inside this unit test.
        object.__setattr__(game, "home_full_runs", home_full)
        object.__setattr__(game, "away_pitcher_chain_runs_allowed", home_full)
    return game


def _scores(paths):
    games = tuple(_game(path) for path in paths)
    simulation = object.__new__(SimulationResult)
    object.__setattr__(simulation, "forecast_run_id", "a" * 64)
    object.__setattr__(simulation, "simulation_count", len(games))
    object.__setattr__(simulation, "games", games)
    object.__setattr__(simulation, "component_distributions", MappingProxyType({}))
    return simulation


class MarketApiAvailabilityTests(unittest.TestCase):
    def test_market_api_is_available(self):
        self.assertTrue(MARKET_API_AVAILABLE, "forecasting market API is missing")


@unittest.skipUnless(MARKET_API_AVAILABLE, "forecasting market API is missing")
class OutcomeProbabilityTests(unittest.TestCase):
    def test_probabilities_are_frozen_and_canonical(self):
        probabilities = OutcomeProbabilities(win=0.5, push=0.1, loss=0.4)

        self.assertEqual(
            canonical_json(probabilities),
            '{"loss":0.4,"push":0.1,"win":0.5}',
        )
        with self.assertRaises(FrozenInstanceError):
            probabilities.win = 0.4

    def test_probability_sum_accepts_only_a_small_float_tolerance(self):
        accepted = OutcomeProbabilities(
            win=0.5,
            push=0.1,
            loss=0.4000000000005,
        )

        self.assertAlmostEqual(
            accepted.win + accepted.push + accepted.loss,
            1.0,
            delta=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "sum"):
            OutcomeProbabilities(win=0.5, push=0.1, loss=0.400000000002)

    def test_tolerated_sum_residual_is_canonical_and_safely_priceable(self):
        probabilities = OutcomeProbabilities(
            win=0.6,
            push=0.4000000000005,
            loss=0.0,
        )

        result = price_market(
            probabilities,
            selected_price=2.0,
            opposing_price=2.0,
        )

        self.assertEqual(
            math.fsum(
                (probabilities.win, probabilities.push, probabilities.loss)
            ),
            1.0,
        )
        self.assertEqual(result.model_conditional_win_probability, 1.0)

    def test_probabilities_reject_nonfinite_out_of_range_and_nonnumeric_values(self):
        invalid_values = (math.nan, math.inf, -0.01, 1.01, "0.5", True)

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    OutcomeProbabilities(win=invalid, push=0.0, loss=1.0)

    def test_exact_one_endpoint_rejects_any_nonzero_residual_mass(self):
        impossible = (
            {"win": 1.0, "push": 1e-20, "loss": 0.0},
            {"win": 1.0, "push": 0.0, "loss": 1e-20},
            {"win": 1e-20, "push": 1.0, "loss": 0.0},
            {"win": 0.0, "push": 1.0, "loss": 1e-20},
            {"win": 1e-20, "push": 0.0, "loss": 1.0},
            {"win": 0.0, "push": 1e-20, "loss": 1.0},
        )

        for probabilities in impossible:
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(ValueError, "exact-one"):
                    OutcomeProbabilities(**probabilities)

    def test_sum_canonicalization_cannot_create_an_impossible_endpoint(self):
        with self.assertRaisesRegex(ValueError, "exact-one"):
            OutcomeProbabilities(
                win=1e-20,
                push=0.9999999999995,
                loss=0.0,
            )


@unittest.skipUnless(MARKET_API_AVAILABLE, "forecasting market API is missing")
class MarketPricingTests(unittest.TestCase):
    def test_push_aware_fair_price_and_ev(self):
        probabilities = OutcomeProbabilities(win=0.50, push=0.10, loss=0.40)

        result = price_market(
            probabilities,
            selected_price=2.00,
            opposing_price=1.90,
        )

        self.assertAlmostEqual(result.fair_decimal_price, 1.80)
        self.assertAlmostEqual(result.expected_value, 0.10)

    def test_no_vig_compares_conditional_non_push_probabilities(self):
        result = price_market(
            OutcomeProbabilities(win=0.45, push=0.10, loss=0.45),
            selected_price=1.91,
            opposing_price=1.91,
        )

        self.assertAlmostEqual(result.model_conditional_win_probability, 0.50)
        self.assertAlmostEqual(result.market_no_vig_probability, 0.50)
        self.assertAlmostEqual(result.edge, 0.0)

    def test_conditional_probability_clamps_float_roundoff_at_one(self):
        probabilities = OutcomeProbabilities(win=0.1, push=0.9, loss=0.0)

        try:
            result = price_market(
                probabilities,
                selected_price=2.0,
                opposing_price=2.0,
            )
        except ValueError as error:
            self.fail(f"mathematical probability boundary was rejected: {error}")

        self.assertEqual(result.model_conditional_win_probability, 1.0)
        self.assertEqual(result.fair_decimal_price, 1.0)
        self.assertEqual(result.market_no_vig_probability, 0.5)
        self.assertEqual(result.edge, 0.5)
        self.assertEqual(result.expected_value, 0.1)

    def test_materially_invalid_derived_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"must be in \[0, 1\]"):
            MarketEvaluation(
                probabilities=OutcomeProbabilities(
                    win=0.1,
                    push=0.9,
                    loss=0.0,
                ),
                model_conditional_win_probability=1.000001,
                fair_decimal_price=1.0,
                market_no_vig_probability=0.5,
                edge=0.5,
                expected_value=0.1,
            )

    def test_market_evaluation_price_context_is_canonical_and_reconstructible(self):
        result = price_market(
            OutcomeProbabilities(win=0.5, push=0.1, loss=0.4),
            selected_price=2,
            opposing_price=1.9,
        )
        field_names = {field.name for field in fields(result)}

        self.assertIn("selected_decimal_price", field_names)
        self.assertIn("opposing_decimal_price", field_names)
        self.assertIsInstance(result.selected_decimal_price, float)
        self.assertIsInstance(result.opposing_decimal_price, float)
        reconstructed = MarketEvaluation(
            **{field.name: getattr(result, field.name) for field in fields(result)}
        )
        self.assertEqual(reconstructed, result)
        self.assertEqual(canonical_json(reconstructed), canonical_json(result))

    def test_direct_constructor_rejects_forged_pricing_fields(self):
        result = price_market(
            OutcomeProbabilities(win=0.5, push=0.1, loss=0.4),
            selected_price=2.0,
            opposing_price=1.9,
        )
        baseline = {
            field.name: getattr(result, field.name) for field in fields(result)
        }
        forged_values = {
            "selected_decimal_price": 2.1,
            "opposing_decimal_price": 2.1,
            "model_conditional_win_probability": 0.6,
            "fair_decimal_price": 2.1,
            "market_no_vig_probability": 0.4,
            "edge": 0.2,
            "expected_value": 0.2,
        }

        for field_name, forged_value in forged_values.items():
            values = dict(baseline)
            values[field_name] = forged_value
            with self.subTest(field_name=field_name):
                try:
                    MarketEvaluation(**values)
                except ValueError as error:
                    self.assertIn(field_name, str(error))
                except TypeError as error:
                    self.fail(f"primitive pricing context is missing: {error}")
                else:
                    self.fail(f"forged {field_name} was accepted")

    def test_direct_constructor_enforces_all_push_degenerates(self):
        result = price_market(
            OutcomeProbabilities(win=0.0, push=1.0, loss=0.0),
            selected_price=2.0,
            opposing_price=2.0,
        )
        baseline = {
            field.name: getattr(result, field.name) for field in fields(result)
        }
        forged_values = {
            "model_conditional_win_probability": 0.1,
            "fair_decimal_price": 1.0,
            "market_no_vig_probability": 0.4,
            "edge": 0.1,
            "expected_value": 0.1,
        }

        for field_name, forged_value in forged_values.items():
            values = dict(baseline)
            values[field_name] = forged_value
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    MarketEvaluation(**values)

    def test_unevaluated_market_rejects_primitive_pricing_context(self):
        result = evaluate_market(
            _scores([(3, 2, 5, 4)]),
            _market(
                MarketType.F5_SIDE,
                "away",
                line=0.0,
                settlement_format=SettlementFormat.THREE_WAY,
            ),
        )
        values = {
            field.name: getattr(result, field.name) for field in fields(result)
        }
        values["selected_decimal_price"] = 1.91

        try:
            MarketEvaluation(**values)
        except ValueError as error:
            self.assertIn("selected_decimal_price", str(error))
        except TypeError as error:
            self.fail(f"primitive pricing context is missing: {error}")
        else:
            self.fail("unevaluated market accepted primitive pricing context")

    def test_missing_entry_price_keeps_probability_and_flow_review(self):
        probabilities = OutcomeProbabilities(win=0.55, push=0.0, loss=0.45)

        result = price_market(
            probabilities,
            selected_price=None,
            opposing_price=None,
        )

        self.assertTrue(result.evaluated)
        self.assertIsNone(result.reason)
        self.assertIs(result.probabilities, probabilities)
        self.assertAlmostEqual(result.fair_decimal_price, 1 + 0.45 / 0.55)
        self.assertIsNone(result.expected_value)
        self.assertIsNone(result.market_no_vig_probability)
        self.assertIsNone(result.edge)

    def test_all_push_has_no_conditional_probability_or_fair_price(self):
        result = price_market(
            OutcomeProbabilities(win=0.0, push=1.0, loss=0.0),
            selected_price=1.91,
            opposing_price=1.91,
        )

        self.assertIsNone(result.model_conditional_win_probability)
        self.assertIsNone(result.fair_decimal_price)
        self.assertAlmostEqual(result.market_no_vig_probability, 0.5)
        self.assertIsNone(result.edge)
        self.assertEqual(result.expected_value, 0.0)

    def test_all_push_without_entry_price_has_no_ev(self):
        result = price_market(
            OutcomeProbabilities(win=0.0, push=1.0, loss=0.0),
            selected_price=None,
            opposing_price=1.91,
        )

        self.assertIsNone(result.expected_value)
        self.assertIsNone(result.market_no_vig_probability)

    def test_zero_win_probability_has_no_fair_price_and_loses_non_push_mass(self):
        result = price_market(
            OutcomeProbabilities(win=0.0, push=0.2, loss=0.8),
            selected_price=2.0,
            opposing_price=2.0,
        )

        self.assertEqual(result.model_conditional_win_probability, 0.0)
        self.assertIsNone(result.fair_decimal_price)
        self.assertAlmostEqual(result.edge, -0.5)
        self.assertEqual(result.expected_value, -0.8)

    def test_subnormal_win_caps_fair_price_at_max_finite_float(self):
        probabilities = OutcomeProbabilities(
            win=math.ulp(0.0),
            push=0.5,
            loss=0.5,
        )

        try:
            result = price_market(
                probabilities,
                selected_price=2.0,
                opposing_price=2.0,
            )
        except ValueError as error:
            self.fail(f"subnormal fair-price overflow was rejected: {error}")

        self.assertEqual(result.fair_decimal_price, sys.float_info.max)
        self.assertEqual(
            result.expected_value,
            probabilities.win * (2.0 - 1.0) - probabilities.loss,
        )
        self.assertTrue(math.isfinite(result.fair_decimal_price))
        canonical_json(result)

    def test_missing_opposing_price_keeps_model_price_and_ev_only(self):
        result = price_market(
            OutcomeProbabilities(win=0.6, push=0.0, loss=0.4),
            selected_price=1.9,
            opposing_price=None,
        )

        self.assertAlmostEqual(result.fair_decimal_price, 1 + 0.4 / 0.6)
        self.assertAlmostEqual(result.expected_value, 0.14)
        self.assertIsNone(result.market_no_vig_probability)
        self.assertIsNone(result.edge)

    def test_prices_must_be_finite_numeric_decimal_prices_above_one(self):
        invalid_prices = (1.0, 0.99, 0.0, -2.0, math.nan, math.inf, "1.91", True)

        for field_name in ("selected_price", "opposing_price"):
            for invalid in invalid_prices:
                prices = {"selected_price": 1.91, "opposing_price": 1.91}
                prices[field_name] = invalid
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        price_market(
                            OutcomeProbabilities(win=0.5, push=0.0, loss=0.5),
                            **prices,
                        )

    def test_market_evaluation_is_frozen_and_canonical(self):
        result = price_market(
            OutcomeProbabilities(win=0.5, push=0.0, loss=0.5),
            selected_price=2.0,
            opposing_price=2.0,
        )

        self.assertEqual(canonical_json(result), canonical_json(result))
        with self.assertRaises(FrozenInstanceError):
            result.edge = 1.0


@unittest.skipUnless(MARKET_API_AVAILABLE, "forecasting market API is missing")
class MarketSettlementTests(unittest.TestCase):
    def test_f5_tie_is_push_for_two_way_market(self):
        result = evaluate_market(
            _scores([(2, 2, 4, 3)]),
            _market(MarketType.F5_SIDE, "away", line=0, selected_price=1.91),
        )

        self.assertEqual(result.probabilities.push, 1.0)
        self.assertIsNone(result.model_conditional_win_probability)
        self.assertIsNone(result.fair_decimal_price)
        self.assertIsNone(result.edge)
        self.assertEqual(result.expected_value, 0.0)

    def test_integer_total_can_push(self):
        result = evaluate_market(
            _scores([(1, 1, 4, 4)]),
            _market(MarketType.FULL_TOTAL, "over", line=8.0),
        )

        self.assertEqual(result.probabilities.push, 1.0)

    def test_full_game_ml_has_no_push(self):
        result = evaluate_market(
            _scores([(2, 2, 5, 4)]),
            _market(MarketType.ML, "home", line=None),
        )

        self.assertEqual(result.probabilities.push, 0.0)
        self.assertEqual(result.probabilities.loss, 1.0)

    def test_every_market_type_settles_both_selected_sides(self):
        cases = (
            (MarketType.F5_SIDE, "away", 0.0, (3, 2, 5, 4), "win"),
            (MarketType.F5_SIDE, "home", 0.0, (3, 2, 5, 4), "loss"),
            (MarketType.F5_TOTAL, "over", 4.5, (3, 2, 5, 4), "win"),
            (MarketType.F5_TOTAL, "under", 4.5, (3, 2, 5, 4), "loss"),
            (MarketType.ML, "away", None, (3, 2, 5, 4), "win"),
            (MarketType.ML, "home", None, (3, 2, 5, 4), "loss"),
            (MarketType.PLUS_1_5, "away", 1.5, (2, 1, 3, 4), "win"),
            (MarketType.PLUS_1_5, "home", 1.5, (1, 2, 4, 3), "win"),
            (MarketType.FULL_TOTAL, "over", 8.5, (3, 2, 5, 4), "win"),
            (MarketType.FULL_TOTAL, "under", 8.5, (3, 2, 5, 4), "loss"),
        )

        for market_type, side, line, path, expected in cases:
            with self.subTest(market_type=market_type, side=side):
                result = evaluate_market(
                    _scores([path]),
                    _market(market_type, side, line=line),
                )
                self.assertEqual(getattr(result.probabilities, expected), 1.0)

    def test_side_and_total_boundaries_push_but_plus_one_and_a_half_does_not(self):
        cases = (
            (MarketType.F5_SIDE, "away", 1.0, (2, 3, 5, 4), 1.0),
            (MarketType.F5_TOTAL, "under", 5.0, (2, 3, 5, 4), 1.0),
            (MarketType.FULL_TOTAL, "over", 9.0, (2, 3, 5, 4), 1.0),
            (MarketType.PLUS_1_5, "away", 1.5, (2, 3, 3, 4), 0.0),
        )

        for market_type, side, line, path, expected_push in cases:
            with self.subTest(market_type=market_type):
                result = evaluate_market(
                    _scores([path]),
                    _market(market_type, side, line=line),
                )
                self.assertEqual(result.probabilities.push, expected_push)

    def test_probabilities_use_every_simulated_game(self):
        result = evaluate_market(
            _scores(
                [
                    (3, 2, 5, 4),
                    (2, 2, 4, 3),
                    (1, 2, 3, 4),
                    (4, 1, 6, 2),
                ]
            ),
            _market(MarketType.F5_SIDE, "away", line=0.0),
        )

        self.assertEqual(
            result.probabilities,
            OutcomeProbabilities(win=0.5, push=0.25, loss=0.25),
        )

    def test_unsupported_three_way_f5_is_preserved_as_unevaluated(self):
        result = evaluate_market(
            _scores([(3, 2, 5, 4)]),
            _market(
                MarketType.F5_SIDE,
                "away",
                line=0.0,
                settlement_format=SettlementFormat.THREE_WAY,
            ),
        )

        self.assertFalse(result.evaluated)
        self.assertIn("three-way F5", result.reason)
        self.assertIsNone(result.probabilities)
        self.assertIsNone(result.expected_value)

    def test_market_contract_requires_normalized_side_and_line_shapes(self):
        invalid_cases = (
            (MarketType.F5_SIDE, "SEA", 0.0),
            (MarketType.F5_SIDE, "away", None),
            (MarketType.F5_TOTAL, "away", 4.5),
            (MarketType.F5_TOTAL, "over", None),
            (MarketType.ML, "away", 0.0),
            (MarketType.PLUS_1_5, "away", None),
            (MarketType.PLUS_1_5, "away", -1.5),
            (MarketType.FULL_TOTAL, "home", 8.5),
            (MarketType.FULL_TOTAL, "under", None),
        )

        for market_type, side, line in invalid_cases:
            with self.subTest(market_type=market_type, side=side, line=line):
                with self.assertRaises(ValueError):
                    evaluate_market(
                        _scores([(3, 2, 5, 4)]),
                        _market(market_type, side, line=line),
                    )

    def test_evaluation_revalidates_market_prices_above_one(self):
        market = _market(
            MarketType.ML,
            "away",
            line=None,
            selected_price=1.0,
        )

        with self.assertRaisesRegex(ValueError, "selected_decimal_price"):
            evaluate_market(_scores([(3, 2, 5, 4)]), market)

    def test_non_f5_three_way_format_is_rejected(self):
        market = _market(
            MarketType.ML,
            "away",
            line=None,
            settlement_format=SettlementFormat.THREE_WAY,
        )

        with self.assertRaisesRegex(ValueError, "settlement_format"):
            evaluate_market(_scores([(3, 2, 5, 4)]), market)


if __name__ == "__main__":
    unittest.main()
