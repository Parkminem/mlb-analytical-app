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
import analysis.forecasting.starter as starter_module
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.inputs import (
    BooleanFeature,
    FeatureProvenance,
    IntegerFeature,
    NumericFeature,
    PitcherChainInput,
    PitcherInput,
    PitcherRole,
    StarterProjectionComponents,
    WorkloadHistory,
)
from analysis.forecasting.randomness import (
    central_interval,
    draw_bounded_lognormal,
    draw_gamma_poisson,
    weighted_choice,
)
from analysis.forecasting.starter import (
    PitcherChainForecast,
    PitcherForecast,
    PitcherPlanSegment,
    SampledPitcherPlan,
    WorkloadDistribution,
    forecast_pitcher,
    forecast_pitcher_chain,
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


def _starter_provenance():
    return FeatureProvenance(
        observation_ids=("obs:starter",),
        latest_available_at=None,
        lookback_end=None,
    )


def _starter_numeric(value):
    return NumericFeature(value=value, provenance=_starter_provenance())


def _starter_boolean(value=False):
    return BooleanFeature(value=value, provenance=_starter_provenance())


def _starter_integer(value):
    return IntegerFeature(value=value, provenance=_starter_provenance())


def _starter_workload(outs):
    return WorkloadHistory(
        recorded_outs=tuple(outs),
        provenance=_starter_provenance(),
    )


def _starter_projection_components():
    return StarterProjectionComponents(
        strikeout_minus_walk_rate=_starter_numeric(0.18),
        walk_rate=_starter_numeric(0.07),
        home_runs_per_contact=_starter_numeric(0.04),
        pitches_per_inning=_starter_numeric(16.0),
        recent_pitch_count=_starter_numeric(90.0),
        recent_workload_outs=_starter_numeric(16.0),
    )


def _starter_pitcher(
    pitcher_id="pitcher:starter",
    role=PitcherRole.STARTER,
    *,
    eligible_outs=(12, 15, 18),
    prior_outs=(9, 15, 18),
    pitch_limit=None,
    projected_ra9=4.5,
    log_standard_error=0.20,
    command_risk=False,
    home_run_risk=False,
    injury_return=False,
    recent_short_exit=False,
):
    return PitcherInput(
        pitcher_id=pitcher_id,
        pitcher_name=f"Pitcher {pitcher_id}",
        role=role,
        throws="R",
        projected_ra9=_starter_numeric(projected_ra9),
        projection_components=_starter_projection_components(),
        log_standard_error=_starter_numeric(log_standard_error),
        eligible_workload=_starter_workload(eligible_outs),
        league_prior_workload=_starter_workload(prior_outs),
        pitch_limit=(
            None if pitch_limit is None else _starter_integer(pitch_limit)
        ),
        pitch_mix=(),
        command_risk=_starter_boolean(command_risk),
        home_run_risk=_starter_boolean(home_run_risk),
        injury_return=_starter_boolean(injury_return),
        recent_short_exit=_starter_boolean(recent_short_exit),
    )


def _conventional_starter_chain(**pitcher_overrides):
    chain_pitcher = _starter_pitcher(**pitcher_overrides)
    return PitcherChainInput(
        pitchers=(chain_pitcher,),
        official_pitcher_id=chain_pitcher.pitcher_id,
        rotowire_pitcher_id=chain_pitcher.pitcher_id,
        matchup_pitcher_id=chain_pitcher.pitcher_id,
        starter_confirmed=_starter_boolean(True),
        starter_source_conflict=_starter_boolean(False),
        handoff_verified=_starter_boolean(True),
        timing_uncertain=_starter_boolean(False),
        eligible_combined_workload=None,
        league_prior_combined_workload=None,
        combined_pitch_limit=None,
    )


def _opener_chain(
    *,
    final_role=PitcherRole.BULK,
    opener_outs=(3, 6, 9),
    final_outs=(12, 15, 18),
    component_prior_outs=(9, 15, 18),
    combined_outs=(12, 16, 21),
    combined_prior_outs=(9, 15, 18),
    opener_pitch_limit=None,
    final_pitch_limit=None,
    combined_pitch_limit=None,
    handoff_verified=True,
    timing_uncertain=False,
):
    opener = _starter_pitcher(
        "pitcher:opener",
        PitcherRole.OPENER,
        eligible_outs=opener_outs,
        prior_outs=component_prior_outs,
        pitch_limit=opener_pitch_limit,
        projected_ra9=4.05,
        log_standard_error=0.25,
    )
    final = _starter_pitcher(
        f"pitcher:{final_role.value}",
        final_role,
        eligible_outs=final_outs,
        prior_outs=component_prior_outs,
        pitch_limit=final_pitch_limit,
        projected_ra9=4.95,
        log_standard_error=0.30,
    )
    return PitcherChainInput(
        pitchers=(opener, final),
        official_pitcher_id=opener.pitcher_id,
        rotowire_pitcher_id=final.pitcher_id,
        matchup_pitcher_id=final.pitcher_id,
        starter_confirmed=_starter_boolean(True),
        starter_source_conflict=_starter_boolean(False),
        handoff_verified=_starter_boolean(handoff_verified),
        timing_uncertain=_starter_boolean(timing_uncertain),
        eligible_combined_workload=_starter_workload(combined_outs),
        league_prior_combined_workload=_starter_workload(
            combined_prior_outs
        ),
        combined_pitch_limit=(
            None
            if combined_pitch_limit is None
            else _starter_integer(combined_pitch_limit)
        ),
    )


def _convolved_bucket_probabilities(first, second):
    probabilities = [0.0, 0.0, 0.0]
    for first_bucket_probability, first_support in zip(
        first.bucket_probabilities,
        first.supports,
    ):
        for first_outs in first_support:
            first_probability = first_bucket_probability / len(first_support)
            for second_bucket_probability, second_support in zip(
                second.bucket_probabilities,
                second.supports,
            ):
                for second_outs in second_support:
                    probability = (
                        first_probability
                        * second_bucket_probability
                        / len(second_support)
                    )
                    combined_outs = first_outs + second_outs
                    bucket = 0 if combined_outs < 15 else 1 if combined_outs < 18 else 2
                    probabilities[bucket] += probability
    return tuple(probabilities)


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
        for module in (config_module, randomness_module, starter_module):
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


class StarterForecastTests(unittest.TestCase):
    def test_bucket_probabilities_sum_to_one_and_are_positive_without_limit(self):
        forecast = forecast_pitcher(
            _starter_pitcher(
                eligible_outs=(12, 13, 15, 18, 21),
                prior_outs=(9, 12, 15, 16, 18, 24),
            ),
            SimulationConfig(),
        )

        self.assertEqual(forecast.workload.source, "pitcher_history")
        self.assertEqual(forecast.workload.bucket_probabilities, (3 / 8, 2 / 8, 3 / 8))
        self.assertAlmostEqual(sum(forecast.workload.bucket_probabilities), 1.0)
        self.assertGreater(forecast.workload.probability_under_five, 0.0)
        self.assertGreater(forecast.workload.probability_five, 0.0)
        self.assertGreater(forecast.workload.probability_six_plus, 0.0)

    def test_each_bucket_gets_one_median_low_prior_representative(self):
        forecast = forecast_pitcher(
            _starter_pitcher(
                eligible_outs=(10, 15, 18),
                prior_outs=(3, 7, 11, 14, 15, 16, 17, 17, 18, 20, 22, 24),
            ),
            SimulationConfig(),
        )

        self.assertEqual(
            forecast.workload.supports,
            ((10, 7), (15, 16), (18, 20)),
        )
        expected = ((10 + 7) / 2 + (15 + 16) / 2 + (18 + 20) / 2) / 3
        self.assertAlmostEqual(forecast.workload.expected_outs, expected)

    def test_empty_pitcher_history_uses_one_prior_value_per_bucket(self):
        forecast = forecast_pitcher(
            _starter_pitcher(
                eligible_outs=(),
                prior_outs=(3, 7, 11, 14, 15, 16, 17, 17, 18, 20, 22, 24),
            ),
            SimulationConfig(),
        )

        self.assertEqual(forecast.workload.supports, ((7,), (16,), (20,)))
        self.assertEqual(forecast.workload.bucket_probabilities, (1 / 3, 1 / 3, 1 / 3))

    def test_missing_required_league_prior_bucket_raises_clearly(self):
        with self.assertRaisesRegex(ValueError, "league_prior_workload.*six_plus"):
            forecast_pitcher(
                _starter_pitcher(
                    eligible_outs=(12, 15, 18),
                    prior_outs=(9, 15),
                ),
                SimulationConfig(),
            )

    def test_default_pitch_limit_transfers_all_high_mass_forty_sixty(self):
        forecast = forecast_pitcher(
            _starter_pitcher(pitch_limit=80),
            SimulationConfig(),
        )

        self.assertAlmostEqual(forecast.workload.probability_under_five, 7 / 15)
        self.assertAlmostEqual(forecast.workload.probability_five, 8 / 15)
        self.assertEqual(forecast.workload.probability_six_plus, 0.0)
        self.assertEqual(forecast.workload.supports[2], (18, 18))
        self.assertIn("pitch_limit=80", forecast.workload.notes[0])
        self.assertIn("40%", forecast.workload.notes[0])
        self.assertIn("60%", forecast.workload.notes[0])

    def test_custom_pitch_limit_transfer_stays_normalized_without_drift(self):
        config = SimulationConfig(
            pitch_limit_high_bucket_transfer=0.25,
            pitch_limit_transfer_to_under_five=0.20,
        )

        forecast = forecast_pitcher(
            _starter_pitcher(pitch_limit=75),
            config,
        )

        self.assertAlmostEqual(forecast.workload.probability_under_five, 0.35)
        self.assertAlmostEqual(forecast.workload.probability_five, 0.40)
        self.assertAlmostEqual(forecast.workload.probability_six_plus, 0.25)
        self.assertAlmostEqual(sum(forecast.workload.bucket_probabilities), 1.0)
        self.assertTrue(
            all(0.0 <= probability <= 1.0 for probability in forecast.workload.bucket_probabilities)
        )

    def test_sample_outs_is_seeded_supported_and_skips_zero_probability_high(self):
        workload = forecast_pitcher(
            _starter_pitcher(
                eligible_outs=(9, 12, 15, 17, 18, 21),
                prior_outs=(6, 12, 15, 16, 18, 24),
                pitch_limit=80,
            ),
            SimulationConfig(),
        ).workload

        self.assertEqual(
            workload.sample_outs(random.Random(31)),
            workload.sample_outs(random.Random(31)),
        )
        allowed = set(workload.supports[0] + workload.supports[1])
        draws = [workload.sample_outs(random.Random(seed)) for seed in range(100)]
        self.assertTrue(all(draw in allowed for draw in draws))
        self.assertTrue(all(draw < 18 for draw in draws))

        tail_rng = FixedRandomRng(1.0)
        self.assertIn(workload.sample_outs(tail_rng), workload.supports[1])
        self.assertEqual(tail_rng.calls, 2)

    def test_projected_ra9_maps_to_clipped_multiplier_without_projected_runs(self):
        cases = ((1.0, 0.55), (5.4, 1.20), (20.0, 1.75))

        for projected_ra9, expected in cases:
            with self.subTest(projected_ra9=projected_ra9):
                forecast = forecast_pitcher(
                    _starter_pitcher(projected_ra9=projected_ra9),
                    SimulationConfig(),
                )
                self.assertAlmostEqual(forecast.run_prevention_multiplier, expected)
                self.assertFalse(hasattr(forecast, "projected_runs"))

        for result_type in (
            WorkloadDistribution,
            PitcherForecast,
            PitcherChainForecast,
            PitcherPlanSegment,
            SampledPitcherPlan,
        ):
            self.assertNotIn("projected_runs", {field.name for field in fields(result_type)})

    def test_injury_and_short_exit_penalties_are_added_in_exact_quadrature(self):
        config = SimulationConfig()
        baseline = forecast_pitcher(
            _starter_pitcher(log_standard_error=0.20),
            config,
        )
        penalized = forecast_pitcher(
            _starter_pitcher(
                log_standard_error=0.20,
                injury_return=True,
                recent_short_exit=True,
            ),
            config,
        )

        self.assertEqual(baseline.log_standard_error, 0.20)
        self.assertAlmostEqual(
            penalized.log_standard_error,
            math.sqrt(0.20 ** 2 + 0.12 ** 2 + 0.10 ** 2),
        )

    def test_command_and_home_run_risk_are_diagnostics_only(self):
        clean = forecast_pitcher(_starter_pitcher(), SimulationConfig())
        risky = forecast_pitcher(
            _starter_pitcher(command_risk=True, home_run_risk=True),
            SimulationConfig(),
        )

        self.assertTrue(risky.command_risk)
        self.assertTrue(risky.home_run_risk)
        self.assertIn("command_risk", risky.diagnostics)
        self.assertIn("home_run_risk", risky.diagnostics)
        self.assertEqual(risky.workload, clean.workload)
        self.assertEqual(
            risky.run_prevention_multiplier,
            clean.run_prevention_multiplier,
        )
        self.assertEqual(risky.log_standard_error, clean.log_standard_error)

    def test_uncertain_handoff_only_widens_chain_sigma_and_diagnostics(self):
        config = SimulationConfig()
        chain = _opener_chain()
        verified = forecast_pitcher_chain(chain, config)
        unverified = forecast_pitcher_chain(
            replace(chain, handoff_verified=_starter_boolean(False)),
            config,
        )
        timing_uncertain = forecast_pitcher_chain(
            replace(chain, timing_uncertain=_starter_boolean(True)),
            config,
        )

        for uncertain in (unverified, timing_uncertain):
            with self.subTest(diagnostics=uncertain.diagnostics):
                self.assertAlmostEqual(
                    uncertain.log_standard_error,
                    math.hypot(
                        verified.log_standard_error,
                        config.handoff_uncertainty_log_se_penalty,
                    ),
                )
                self.assertEqual(
                    uncertain.combined_workload.bucket_probabilities,
                    verified.combined_workload.bucket_probabilities,
                )
                self.assertEqual(
                    tuple(p.run_prevention_multiplier for p in uncertain.pitchers),
                    tuple(p.run_prevention_multiplier for p in verified.pitchers),
                )

    def test_verified_opener_bulk_and_prim_plans_are_ordered_and_contiguous(self):
        for final_role in (PitcherRole.BULK, PitcherRole.PRIM):
            with self.subTest(final_role=final_role):
                forecast = forecast_pitcher_chain(
                    _opener_chain(final_role=final_role),
                    SimulationConfig(),
                )
                plan = forecast.sample_plan(random.Random(19))

                self.assertEqual(
                    [segment.role for segment in plan.segments],
                    ["opener", final_role.value],
                )
                cursor = 0
                for segment in plan.segments:
                    self.assertEqual(segment.start_out, cursor)
                    cursor = segment.end_out
                self.assertEqual(cursor, plan.combined_outs)
                self.assertEqual(
                    plan.combined_outs,
                    sum(segment.outs for segment in plan.segments),
                )
                self.assertLessEqual(plan.combined_outs, 27)

    def test_final_pitcher_gets_exact_remainder_when_opener_draw_exceeds_total(self):
        forecast = forecast_pitcher_chain(
            _opener_chain(
                opener_outs=(14,),
                combined_outs=(12,),
            ),
            SimulationConfig(),
        )

        plan = forecast.sample_plan(FixedRandomRng(0.0))

        self.assertEqual([segment.outs for segment in plan.segments], [12, 0])
        self.assertEqual(plan.segments[-1].start_out, 12)
        self.assertEqual(plan.segments[-1].end_out, 12)
        self.assertEqual(plan.combined_outs, 12)

    def test_combined_probabilities_come_from_history_not_component_convolution(self):
        forecast = forecast_pitcher_chain(
            _opener_chain(
                opener_outs=(3, 3, 6),
                final_outs=(15, 18, 18),
                combined_outs=(18, 21, 24),
            ),
            SimulationConfig(),
        )

        self.assertEqual(forecast.combined_workload.source, "combined_chain_history")
        self.assertEqual(
            forecast.combined_workload.bucket_probabilities,
            (1 / 6, 1 / 6, 4 / 6),
        )
        convolved = _convolved_bucket_probabilities(
            forecast.pitchers[0].workload,
            forecast.pitchers[1].workload,
        )
        self.assertNotEqual(forecast.combined_workload.bucket_probabilities, convolved)

    def test_only_combined_pitch_limit_changes_combined_distribution(self):
        config = SimulationConfig()
        baseline = forecast_pitcher_chain(_opener_chain(), config)
        component_limited = forecast_pitcher_chain(
            _opener_chain(opener_pitch_limit=50, final_pitch_limit=80),
            config,
        )
        combined_limited = forecast_pitcher_chain(
            _opener_chain(combined_pitch_limit=105),
            config,
        )

        self.assertEqual(
            component_limited.combined_workload,
            baseline.combined_workload,
        )
        self.assertTrue(
            all(
                pitcher.workload.probability_six_plus == 0.0
                for pitcher in component_limited.pitchers
            )
        )
        self.assertEqual(combined_limited.combined_workload.probability_six_plus, 0.0)
        self.assertGreater(baseline.combined_workload.probability_six_plus, 0.0)

    def test_single_starter_reuses_individual_workload_and_samples_one_segment(self):
        forecast = forecast_pitcher_chain(
            _conventional_starter_chain(),
            SimulationConfig(),
        )
        plan = forecast.sample_plan(random.Random(7))

        self.assertIs(forecast.combined_workload, forecast.pitchers[0].workload)
        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].role, "starter")
        self.assertEqual(plan.segments[0].start_out, 0)
        self.assertEqual(plan.segments[0].end_out, plan.combined_outs)

    def test_sample_plan_is_deterministic_for_the_same_seed(self):
        forecast = forecast_pitcher_chain(_opener_chain(), SimulationConfig())

        first = forecast.sample_plan(random.Random(73))
        second = forecast.sample_plan(random.Random(73))

        self.assertEqual(first, second)

    def test_all_result_dataclasses_are_frozen(self):
        pitcher = forecast_pitcher(_starter_pitcher(), SimulationConfig())
        chain = forecast_pitcher_chain(_opener_chain(), SimulationConfig())
        plan = chain.sample_plan(random.Random(11))
        values_and_fields = (
            (pitcher.workload, "source"),
            (pitcher, "pitcher_id"),
            (chain, "log_standard_error"),
            (plan.segments[0], "start_out"),
            (plan, "combined_outs"),
        )

        for value, field_name in values_and_fields:
            with self.subTest(result_type=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, field_name, getattr(value, field_name))

    def test_invalid_direct_workload_distributions_are_rejected(self):
        valid = WorkloadDistribution(
            probabilities=(1 / 3, 1 / 3, 1 / 3),
            supports=((12,), (15,), (18,)),
            source="pitcher_history",
            notes=(),
        )
        invalid_changes = (
            {"probabilities": (-0.1, 0.6, 0.5)},
            {"probabilities": (0.2, 0.2, 0.2)},
            {"probabilities": (math.nan, 0.5, 0.5)},
            {"probabilities": (True, 0.0, 0.0)},
            {"supports": ((), (15,), (18,))},
            {"supports": ((12.0,), (15,), (18,))},
            {"supports": ((-1,), (15,), (18,))},
            {"supports": ((12,), (15,), (28,))},
            {"source": "invented_history"},
            {"notes": []},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(valid, **changes)

    def test_result_dataclasses_validate_types_ranges_and_finite_values(self):
        pitcher = forecast_pitcher(_starter_pitcher(), SimulationConfig())
        chain = forecast_pitcher_chain(_opener_chain(), SimulationConfig())
        plan = chain.sample_plan(random.Random(5))
        segment = plan.segments[0]
        invalid_replacements = (
            (pitcher, {"run_prevention_multiplier": math.nan}),
            (pitcher, {"log_standard_error": -0.1}),
            (pitcher, {"command_risk": 1}),
            (chain, {"log_standard_error": math.inf}),
            (chain, {"pitchers": list(chain.pitchers)}),
            (segment, {"role": "reliever"}),
            (segment, {"start_out": -1}),
            (segment, {"end_out": 28}),
            (plan, {"combined_outs": plan.combined_outs + 1}),
            (plan, {"segments": list(plan.segments)}),
        )

        for value, changes in invalid_replacements:
            with self.subTest(result_type=type(value).__name__, changes=changes):
                with self.assertRaises(ValueError):
                    replace(value, **changes)


if __name__ == "__main__":
    unittest.main()
