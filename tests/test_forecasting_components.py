import ast
from dataclasses import FrozenInstanceError, fields, replace
import json
import math
from pathlib import Path
import random
import sys
import unittest

import analysis.forecasting.config as config_module
import analysis.forecasting.randomness as randomness_module
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.randomness import (
    central_interval,
    draw_bounded_lognormal,
    draw_gamma_poisson,
    weighted_choice,
)


EXPECTED_DEFAULTS = {
    "version": "shadow-forecast-2026-07-14.1",
    "league_runs_per_team_game": 4.50,
    "league_woba": 0.320,
    "gamma_poisson_shape": 1.60,
    "simulations": 20_000,
    "extra_inning_multiplier": 1.80,
    "max_innings": 30,
    "offense_bounds": [0.65, 1.45],
    "matchup_bounds": [0.80, 1.20],
    "starter_ra9_bounds": [0.55, 1.75],
    "bullpen_ra9_bounds": [0.60, 1.90],
    "pa_weights": [1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87],
    "pitch_limit_high_bucket_transfer": 1.0,
    "pitch_limit_transfer_to_under_five": 0.40,
    "sample_shortage_log_se_penalty": 0.12,
    "unknown_bullpen_matchup_log_se_penalty": 0.08,
    "handoff_uncertainty_log_se_penalty": 0.10,
    "injury_return_log_se_penalty": 0.12,
    "recent_short_exit_log_se_penalty": 0.10,
    "limited_reliever_log_se_penalty": 0.10,
    "unscheduled_f5_relief_log_standard_error": 0.15,
}


class FixedGaussianRng:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def gauss(self, mean, standard_deviation):
        self.calls.append((mean, standard_deviation))
        return self.value


class FixedGammaRng:
    def __init__(self, latent, random_values=()):
        self.latent = latent
        self.random_values = iter(random_values)
        self.gamma_calls = []
        self.random_calls = 0

    def gammavariate(self, alpha, beta):
        self.gamma_calls.append((alpha, beta))
        return self.latent

    def random(self):
        self.random_calls += 1
        return next(self.random_values)


class FixedRandomRng:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


class SimulationConfigTests(unittest.TestCase):
    def test_defaults_are_exact_and_instance_is_frozen(self):
        config = SimulationConfig()

        self.assertEqual(config.to_mapping(), EXPECTED_DEFAULTS)
        self.assertEqual(
            config.offense_bounds,
            (0.65, 1.45),
        )
        self.assertEqual(
            config.pa_weights,
            (1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87),
        )
        with self.assertRaises(FrozenInstanceError):
            config.simulations = 1

    def test_to_mapping_is_complete_deterministic_and_round_trips(self):
        config = SimulationConfig()

        first = config.to_mapping()
        second = config.to_mapping()
        field_names = [field.name for field in fields(SimulationConfig)]

        self.assertIs(type(first), dict)
        self.assertEqual(list(first), field_names)
        self.assertEqual(first, second)
        self.assertEqual(SimulationConfig.from_mapping(first), config)
        self.assertEqual(
            SimulationConfig.from_mapping(json.loads(json.dumps(first))),
            config,
        )

    def test_to_mapping_returns_json_lists_without_mutating_config(self):
        weights = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        config = SimulationConfig(pa_weights=weights)

        mapping = config.to_mapping()
        mapping["pa_weights"][0] = 99.0
        mapping["offense_bounds"][0] = 0.99

        self.assertIs(config.pa_weights, weights)
        self.assertEqual(config.pa_weights[0], 1.0)
        self.assertEqual(config.offense_bounds, (0.65, 1.45))

    def test_from_mapping_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown.*unexpected"):
            SimulationConfig.from_mapping({"unexpected": 1})

    def test_from_mapping_normalizes_json_lists_to_tuples(self):
        mapping = json.loads(json.dumps(EXPECTED_DEFAULTS))

        config = SimulationConfig.from_mapping(mapping)

        for field_name in (
            "offense_bounds",
            "matchup_bounds",
            "starter_ra9_bounds",
            "bullpen_ra9_bounds",
            "pa_weights",
        ):
            with self.subTest(field_name=field_name):
                self.assertIsInstance(getattr(config, field_name), tuple)

    def test_from_mapping_accepts_documented_partial_operator_overrides(self):
        config = SimulationConfig.from_mapping(
            {
                "version": "operator-override-v2",
                "simulations": 1_000,
            }
        )

        self.assertEqual(config.version, "operator-override-v2")
        self.assertEqual(config.simulations, 1_000)
        self.assertEqual(config.league_woba, SimulationConfig().league_woba)

    def test_from_mapping_requires_a_mapping(self):
        for value in (None, [], (("simulations", 1_000),)):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    SimulationConfig.from_mapping(value)

    def test_version_must_be_nonempty(self):
        for value in ("", "   ", None, True, 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "version"):
                    replace(SimulationConfig(), version=value)

    def test_positive_rate_fields_reject_invalid_numeric_values(self):
        field_names = (
            "league_runs_per_team_game",
            "league_woba",
            "gamma_poisson_shape",
            "extra_inning_multiplier",
        )
        invalid_values = (0.0, -0.01, math.inf, -math.inf, math.nan, True)

        for field_name in field_names:
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        replace(SimulationConfig(), **{field_name: value})

    def test_integer_fields_reject_bools_nonintegers_and_invalid_ranges(self):
        cases = {
            "simulations": (0, -1, 1.5, math.inf, True),
            "max_innings": (0, 9, -1, 10.5, math.inf, True),
        }

        for field_name, invalid_values in cases.items():
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        replace(SimulationConfig(), **{field_name: value})

        self.assertEqual(SimulationConfig(max_innings=10).max_innings, 10)

    def test_every_bounds_pair_is_validated(self):
        field_names = (
            "offense_bounds",
            "matchup_bounds",
            "starter_ra9_bounds",
            "bullpen_ra9_bounds",
        )
        invalid_bounds = (
            (),
            (0.8,),
            (0.8, 1.2, 1.3),
            [0.8, 1.2],
            (0.0, 1.2),
            (-0.1, 1.2),
            (0.8, math.inf),
            (math.nan, 1.2),
            (True, 1.2),
            (1.01, 1.2),
            (0.8, 0.99),
            (1.2, 0.8),
        )

        for field_name in field_names:
            for value in invalid_bounds:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        replace(SimulationConfig(), **{field_name: value})

        self.assertEqual(
            SimulationConfig(offense_bounds=(1.0, 1.0)).offense_bounds,
            (1.0, 1.0),
        )

    def test_pa_weights_require_exactly_nine_positive_finite_values(self):
        invalid_weights = (
            (),
            (1.0,) * 8,
            (1.0,) * 10,
            [1.0] * 9,
            (0.0,) + (1.0,) * 8,
            (-1.0,) + (1.0,) * 8,
            (math.inf,) + (1.0,) * 8,
            (math.nan,) + (1.0,) * 8,
            (True,) + (1.0,) * 8,
        )

        for value in invalid_weights:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "pa_weights"):
                    replace(SimulationConfig(), pa_weights=value)

    def test_transfer_fields_are_closed_unit_intervals(self):
        field_names = (
            "pitch_limit_high_bucket_transfer",
            "pitch_limit_transfer_to_under_five",
        )
        invalid_values = (-0.01, 1.01, math.inf, -math.inf, math.nan, True)

        for field_name in field_names:
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        replace(SimulationConfig(), **{field_name: value})

        self.assertEqual(
            SimulationConfig(
                pitch_limit_high_bucket_transfer=0.0,
                pitch_limit_transfer_to_under_five=1.0,
            ).pitch_limit_transfer_to_under_five,
            1.0,
        )

    def test_log_standard_error_penalties_are_finite_and_nonnegative(self):
        field_names = (
            "sample_shortage_log_se_penalty",
            "unknown_bullpen_matchup_log_se_penalty",
            "handoff_uncertainty_log_se_penalty",
            "injury_return_log_se_penalty",
            "recent_short_exit_log_se_penalty",
            "limited_reliever_log_se_penalty",
            "unscheduled_f5_relief_log_standard_error",
        )
        invalid_values = (-0.01, math.inf, -math.inf, math.nan, True)

        for field_name in field_names:
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        replace(SimulationConfig(), **{field_name: value})

        overrides = {field_name: 0.0 for field_name in field_names}
        config = SimulationConfig(**overrides)
        self.assertTrue(all(getattr(config, name) == 0.0 for name in field_names))


class ForecastRandomnessTests(unittest.TestCase):
    def test_bounded_lognormal_uses_mean_preserving_latent_exactly(self):
        rng = FixedGaussianRng(0.75)

        result = draw_bounded_lognormal(rng, 1.2, 0.3, 0.5, 2.0)

        expected = math.exp(math.log(1.2) - 0.3 * 0.3 / 2 + 0.3 * 0.75)
        self.assertAlmostEqual(result, expected)
        self.assertEqual(rng.calls, [(0, 1)])

    def test_bounded_lognormal_empirical_mean_and_bounds(self):
        rng = random.Random(17)
        draws = [
            draw_bounded_lognormal(rng, 1.0, 0.15, 0.55, 1.75)
            for _ in range(50_000)
        ]

        self.assertTrue(all(0.55 <= value <= 1.75 for value in draws))
        self.assertAlmostEqual(sum(draws) / len(draws), 1.0, delta=0.015)

    def test_bounded_lognormal_sigma_zero_and_clipping_edges(self):
        self.assertEqual(
            draw_bounded_lognormal(FixedGaussianRng(100.0), 1.2, 0.0, 0.5, 2.0),
            1.2,
        )
        self.assertEqual(
            draw_bounded_lognormal(FixedGaussianRng(0.0), 0.4, 0.0, 0.5, 2.0),
            0.5,
        )
        self.assertEqual(
            draw_bounded_lognormal(FixedGaussianRng(0.0), 2.5, 0.0, 0.5, 2.0),
            2.0,
        )
        self.assertEqual(
            draw_bounded_lognormal(
                FixedGaussianRng(1_000.0),
                1.0,
                1.0,
                0.5,
                2.0,
            ),
            2.0,
        )
        self.assertEqual(
            draw_bounded_lognormal(
                FixedGaussianRng(-1_000.0),
                1.0,
                1.0,
                0.5,
                2.0,
            ),
            0.5,
        )

    def test_bounded_lognormal_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            draw_bounded_lognormal(object(), 1.0, 0.1, 0.5, 2.0)

        cases = (
            (0.0, 0.1, 0.5, 2.0),
            (-1.0, 0.1, 0.5, 2.0),
            (math.nan, 0.1, 0.5, 2.0),
            (math.inf, 0.1, 0.5, 2.0),
            (True, 0.1, 0.5, 2.0),
            (1.0, -0.1, 0.5, 2.0),
            (1.0, math.nan, 0.5, 2.0),
            (1.0, True, 0.5, 2.0),
            (1.0, 0.1, 0.0, 2.0),
            (1.0, 0.1, 0.5, math.inf),
            (1.0, 0.1, True, 2.0),
            (1.0, 0.1, 2.0, 0.5),
        )
        for point, sigma, lower, upper in cases:
            with self.subTest(
                point=point,
                sigma=sigma,
                lower=lower,
                upper=upper,
            ):
                with self.assertRaises(ValueError):
                    draw_bounded_lognormal(
                        FixedGaussianRng(0.0),
                        point,
                        sigma,
                        lower,
                        upper,
                    )

    def test_gamma_poisson_is_reproducible_with_seeded_rng(self):
        rng = random.Random(9)

        draws = [
            draw_gamma_poisson(rng, mean=4.5, shape=1.6)
            for _ in range(12)
        ]

        self.assertEqual(draws, [2, 3, 3, 3, 2, 2, 12, 0, 1, 2, 4, 1])

    def test_gamma_poisson_zero_mean_does_not_consume_randomness(self):
        rng = random.Random(9)
        state = rng.getstate()

        result = draw_gamma_poisson(rng, mean=0.0, shape=1.6)

        self.assertEqual(result, 0)
        self.assertEqual(rng.getstate(), state)

    def test_gamma_poisson_uses_gamma_latent_and_knuth_poisson(self):
        rng = FixedGammaRng(1.0, (0.8, 0.5, 0.5))

        result = draw_gamma_poisson(rng, mean=4.5, shape=1.6)

        self.assertEqual(result, 2)
        self.assertEqual(rng.gamma_calls, [(1.6, 4.5 / 1.6)])
        self.assertEqual(rng.random_calls, 3)

    def test_gamma_poisson_rejects_invalid_inputs_and_latents(self):
        with self.assertRaises(TypeError):
            draw_gamma_poisson(object(), mean=1.0, shape=1.0)

        for mean in (-0.1, math.nan, math.inf, -math.inf, True):
            with self.subTest(mean=mean):
                with self.assertRaises(ValueError):
                    draw_gamma_poisson(random.Random(1), mean=mean, shape=1.6)

        for shape in (0.0, -0.1, math.nan, math.inf, -math.inf, True):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    draw_gamma_poisson(random.Random(1), mean=1.0, shape=shape)

        for latent in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(latent=latent):
                with self.assertRaisesRegex(ValueError, "latent"):
                    draw_gamma_poisson(
                        FixedGammaRng(latent),
                        mean=1.0,
                        shape=1.0,
                    )

    def test_gamma_poisson_empirical_mean_and_overdispersion_are_reasonable(self):
        rng = random.Random(91)
        draws = [
            draw_gamma_poisson(rng, mean=4.5, shape=1.6)
            for _ in range(25_000)
        ]

        sample_mean = sum(draws) / len(draws)
        variance = sum((value - sample_mean) ** 2 for value in draws) / len(draws)

        self.assertAlmostEqual(sample_mean, 4.5, delta=0.15)
        self.assertGreater(variance, sample_mean * 2.0)
        self.assertLess(variance, sample_mean * 6.0)

    def test_weighted_choice_is_reproducible(self):
        rng = random.Random(9)

        choices = [
            weighted_choice(rng, ("a", "b", "c"), (1.0, 2.0, 3.0))
            for _ in range(12)
        ]

        self.assertEqual(
            choices,
            ["b", "b", "a", "c", "a", "c", "c", "a", "c", "c", "a", "b"],
        )

    def test_weighted_choice_preserves_values_and_skips_zero_weights(self):
        marker = object()

        result = weighted_choice(
            FixedRandomRng(0.0),
            ("zero-weight", marker),
            (0.0, 1.0),
        )

        self.assertIs(result, marker)

    def test_weighted_choice_returns_final_value_for_rounding_tail(self):
        final_value = ("preserved", "tuple")

        result = weighted_choice(
            FixedRandomRng(1.0),
            ("first", "second", final_value),
            (0.1, 0.2, 0.3),
        )

        self.assertIs(result, final_value)

    def test_weighted_choice_rounding_fallback_skips_trailing_zero_weights(self):
        last_positive = object()
        largest_sub_one = math.nextafter(1.0, 0.0)

        result = weighted_choice(
            FixedRandomRng(largest_sub_one),
            ("large", "small", last_positive, "zero-tail-1", "zero-tail-2"),
            (1e16, 1.0, 1.0, 0.0, 0.0),
        )

        self.assertIs(result, last_positive)

    def test_weighted_choice_all_mass_can_precede_trailing_zero_weights(self):
        only_positive = object()
        largest_sub_one = math.nextafter(1.0, 0.0)

        result = weighted_choice(
            FixedRandomRng(largest_sub_one),
            ("leading-zero", only_positive, "zero-tail-1", "zero-tail-2"),
            (0.0, 4.0, 0.0, 0.0),
        )

        self.assertIs(result, only_positive)

    def test_weighted_choice_rejects_invalid_inputs(self):
        rng = FixedRandomRng(0.5)

        with self.assertRaises(TypeError):
            weighted_choice(object(), ("a",), (1.0,))
        with self.assertRaises(ValueError):
            weighted_choice(rng, (), ())
        with self.assertRaises(ValueError):
            weighted_choice(rng, ("a", "b"), (1.0,))

        for weights in (
            (-1.0, 2.0),
            (math.nan, 1.0),
            (math.inf, 1.0),
            (-math.inf, 1.0),
            (True, 1.0),
            (0.0, 0.0),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    weighted_choice(rng, ("a", "b"), weights)

    def test_central_interval_uses_equal_tail_nearest_ranks(self):
        self.assertEqual(
            central_interval(range(1, 11)),
            (1.0, 9.0),
        )
        self.assertEqual(
            central_interval([1, 2, 3, 4], mass=0.50),
            (1.0, 3.0),
        )
        self.assertEqual(
            central_interval([7], mass=0.01),
            (7.0, 7.0),
        )

    def test_central_interval_is_order_invariant(self):
        ordered = central_interval([1, 2, 3, 4, 5], mass=0.60)
        reversed_values = central_interval([5, 4, 3, 2, 1], mass=0.60)

        self.assertEqual(ordered, (1.0, 4.0))
        self.assertEqual(reversed_values, ordered)

    def test_central_interval_rejects_invalid_values_and_mass(self):
        for values in ((), (1.0, math.nan), (1.0, math.inf), (1.0, True), ("1",)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    central_interval(values)

        for mass in (0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf, True):
            with self.subTest(mass=mass):
                with self.assertRaises(ValueError):
                    central_interval((1.0, 2.0), mass=mass)

    def test_modules_use_only_standard_library_dependencies(self):
        for module in (config_module, randomness_module):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            imported_roots = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported_roots.update(
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            )

            with self.subTest(module=module.__name__):
                self.assertEqual(
                    imported_roots.difference(sys.stdlib_module_names),
                    set(),
                )

    def test_helpers_do_not_consume_global_random_state(self):
        global_state = random.getstate()
        rng = random.Random(101)

        draw_bounded_lognormal(rng, 1.0, 0.1, 0.5, 2.0)
        draw_gamma_poisson(rng, mean=4.5, shape=1.6)
        weighted_choice(rng, ("a", "b"), (1.0, 1.0))

        self.assertEqual(random.getstate(), global_state)


if __name__ == "__main__":
    unittest.main()
