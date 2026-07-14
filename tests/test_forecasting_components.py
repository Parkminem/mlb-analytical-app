import ast
from dataclasses import FrozenInstanceError, fields, replace
import json
import math
from pathlib import Path
import random
import sys
import unittest

import analysis.forecasting.bullpen as bullpen_module
import analysis.forecasting.config as config_module
import analysis.forecasting.offense as offense_module
import analysis.forecasting.randomness as randomness_module
import analysis.forecasting.starter as starter_module
from analysis.forecasting.bullpen import (
    BullpenForecast,
    BullpenSegmentForecast,
    forecast_bullpen,
    select_bullpen_segment,
)
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import (
    BooleanFeature,
    BullpenInput,
    BullpenRole,
    BullpenSegmentInput,
    FeatureProvenance,
    IntegerFeature,
    LineupInput,
    LineupSlot,
    LineupState,
    NumericFeature,
    PitcherChainInput,
    PitcherInput,
    PitcherRole,
    PitchUsage,
    PitchVelocityResponse,
    RelieverAppearanceInput,
    StarterProjectionComponents,
    WorkloadHistory,
)
from analysis.forecasting.randomness import (
    central_interval,
    draw_bounded_lognormal,
    draw_gamma_poisson,
    weighted_choice,
)
from analysis.forecasting.offense import (
    OffenseForecast,
    SlotMatchupForecast,
    forecast_offense,
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


class SequenceRandomRng:
    def __init__(self, values):
        self._values = iter(values)
        self.calls = []

    def random(self):
        value = next(self._values)
        self.calls.append(value)
        return value


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


def _offense_usage(
    pitcher_id,
    opposing_side="R",
    pitch_type="FF",
    usage=1.0,
    average_velocity_mph=95.0,
):
    return PitchUsage(
        pitcher_id=pitcher_id,
        opposing_side=opposing_side,
        pitch_type=pitch_type,
        usage=usage,
        average_velocity_mph=average_velocity_mph,
        provenance=_starter_provenance(),
    )


def _offense_response(
    order,
    pitcher_id="pitcher:starter",
    *,
    expected_side="R",
    pitch_type="FF",
    velocity_center_mph=95.0,
    band_width_mph=1.5,
    plate_appearances=30,
    adjusted_woba=0.320,
    sample_strength="strong",
    slot_label="중립",
):
    return PitchVelocityResponse(
        batter_id=f"batter:{order}",
        pitcher_id=pitcher_id,
        expected_side=expected_side,
        pitch_type=pitch_type,
        velocity_center_mph=velocity_center_mph,
        band_width_mph=band_width_mph,
        plate_appearances=plate_appearances,
        adjusted_woba=adjusted_woba,
        sample_strength=sample_strength,
        slot_label=slot_label,
        provenance=_starter_provenance(),
    )


def _offense_slot(order, expected_side="R", responses=()):
    return LineupSlot(
        order=order,
        batter_id=f"batter:{order}",
        batter_name=f"Batter {order}",
        bats=expected_side,
        expected_side=expected_side,
        responses=responses,
    )


def _offense_lineup(
    *,
    slots=None,
    sides=("R", "L", "R", "L", "R", "L", "R", "L", "R"),
    projected_runs_per_9=4.5,
    projected_traffic_per_9=11.5,
    projected_extra_base_hits_per_9=2.4,
    walk_rate=0.08,
    strikeout_rate=0.22,
    isolated_power=0.16,
    barrel_rate=0.08,
    recent_form_index=1.0,
    lower_order_continuity=1.0,
    log_standard_error=0.08,
    recent_form_included=True,
):
    resolved_slots = slots or tuple(
        _offense_slot(order, sides[order - 1])
        for order in range(1, 10)
    )
    return LineupInput(
        team="SEA",
        state=LineupState.CONFIRMED,
        state_provenance=_starter_provenance(),
        slots=resolved_slots,
        projected_runs_per_9=_starter_numeric(projected_runs_per_9),
        projected_traffic_per_9=_starter_numeric(projected_traffic_per_9),
        projected_extra_base_hits_per_9=_starter_numeric(
            projected_extra_base_hits_per_9
        ),
        walk_rate=_starter_numeric(walk_rate),
        strikeout_rate=_starter_numeric(strikeout_rate),
        isolated_power=_starter_numeric(isolated_power),
        barrel_rate=_starter_numeric(barrel_rate),
        recent_form_index=_starter_numeric(recent_form_index),
        lower_order_continuity=_starter_numeric(lower_order_continuity),
        log_standard_error=_starter_numeric(log_standard_error),
        recent_form_included=_starter_boolean(recent_form_included),
    )


def _offense_with_responses(lineup, responses_by_order):
    return replace(
        lineup,
        slots=tuple(
            replace(
                slot,
                responses=tuple(responses_by_order.get(slot.order, ())),
            )
            for slot in lineup.slots
        ),
    )


def _offense_profile(pitcher_id="pitcher:starter", pitch_mix=None):
    resolved_mix = pitch_mix
    if resolved_mix is None:
        resolved_mix = (
            _offense_usage(pitcher_id, "R", "FF", 1.0, 95.0),
            _offense_usage(pitcher_id, "L", "FF", 1.0, 95.0),
        )
    return replace(
        _starter_pitcher(pitcher_id=pitcher_id),
        pitch_mix=resolved_mix,
    )


def _offense_uniform_response_lineup(
    pitcher_id,
    *,
    adjusted_woba=0.320,
    slot_label="중립",
    **lineup_overrides,
):
    lineup = _offense_lineup(**lineup_overrides)
    return _offense_with_responses(
        lineup,
        {
            slot.order: (
                _offense_response(
                    slot.order,
                    pitcher_id,
                    expected_side=slot.expected_side,
                    adjusted_woba=adjusted_woba,
                    slot_label=slot_label,
                ),
            )
            for slot in lineup.slots
        },
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
    opener_log_standard_error=0.25,
    final_log_standard_error=0.30,
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
        log_standard_error=opener_log_standard_error,
    )
    final = _starter_pitcher(
        f"pitcher:{final_role.value}",
        final_role,
        eligible_outs=final_outs,
        prior_outs=component_prior_outs,
        pitch_limit=final_pitch_limit,
        projected_ra9=4.95,
        log_standard_error=final_log_standard_error,
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


_BULLPEN_ROLE_ORDER = (
    BullpenRole.BRIDGE,
    BullpenRole.MIDDLE,
    BullpenRole.HIGH_LEVERAGE,
    BullpenRole.CLOSER,
)
_BULLPEN_RA9_BY_ROLE = {
    BullpenRole.BRIDGE: 6.75,
    BullpenRole.MIDDLE: 4.50,
    BullpenRole.HIGH_LEVERAGE: 3.60,
    BullpenRole.CLOSER: 3.15,
}


def _bullpen_provenance():
    return FeatureProvenance(
        observation_ids=("obs:bullpen",),
        latest_available_at=None,
        lookback_end=None,
    )


def _bullpen_numeric(value):
    return NumericFeature(value=value, provenance=_bullpen_provenance())


def _bullpen_boolean(value=False):
    return BooleanFeature(value=value, provenance=_bullpen_provenance())


def _bullpen_appearance(pitcher_id, days_ago, pitches):
    return RelieverAppearanceInput(
        pitcher_id=pitcher_id,
        days_ago=days_ago,
        pitches=pitches,
        provenance=_bullpen_provenance(),
    )


def _bullpen_segment(
    role,
    *,
    segment_id=None,
    projected_ra9=None,
    log_standard_error=0.20,
    appearances=(),
    available=True,
    limited=False,
):
    return BullpenSegmentInput(
        segment_id=segment_id or f"bullpen:{role.value}",
        role=role,
        projected_ra9=_bullpen_numeric(
            _BULLPEN_RA9_BY_ROLE[role]
            if projected_ra9 is None
            else projected_ra9
        ),
        log_standard_error=_bullpen_numeric(log_standard_error),
        appearances=appearances,
        available=_bullpen_boolean(available),
        limited_performance_or_use_risk=_bullpen_boolean(limited),
    )


def _bullpen_input(role_changes=None, *, segments=None):
    if segments is not None:
        return BullpenInput(team="SEA", segments=segments)

    changes = role_changes or {}
    return BullpenInput(
        team="SEA",
        segments=tuple(
            replace(_bullpen_segment(role), **changes.get(role, {}))
            for role in _BULLPEN_ROLE_ORDER
        ),
    )


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
        for module in (
            bullpen_module,
            config_module,
            offense_module,
            randomness_module,
            starter_module,
        ):
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
        expected_probabilities = (
            1.0 - math.fsum((1 / 3, 1 / 3)),
            1 / 3,
            1 / 3,
        )
        self.assertEqual(
            forecast.workload.bucket_probabilities,
            expected_probabilities,
        )
        self.assertEqual(
            math.fsum(forecast.workload.bucket_probabilities),
            1.0,
        )

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

    def test_non_dyadic_transfer_assigns_largest_bucket_as_exact_complement(self):
        config = SimulationConfig(
            pitch_limit_high_bucket_transfer=1 / 3,
            pitch_limit_transfer_to_under_five=1 / 3,
        )

        probabilities = forecast_pitcher(
            _starter_pitcher(pitch_limit=80),
            config,
        ).workload.bucket_probabilities

        self.assertEqual(
            probabilities,
            (
                0.37037037037037035,
                0.40740740740740744,
                0.2222222222222222,
            ),
        )
        largest_index = max(range(3), key=probabilities.__getitem__)
        other_probabilities = tuple(
            probability
            for index, probability in enumerate(probabilities)
            if index != largest_index
        )
        self.assertEqual(
            probabilities[largest_index],
            1.0 - math.fsum(other_probabilities),
        )
        self.assertEqual(math.fsum(probabilities), 1.0)
        self.assertTrue(all(probability >= 0.0 for probability in probabilities))

    def test_transfer_corrects_one_ulp_residual_in_largest_bucket(self):
        config = SimulationConfig(
            pitch_limit_high_bucket_transfer=0.2699789638878243,
            pitch_limit_transfer_to_under_five=0.6548166999884969,
        )
        eligible_outs = (12,) * 70 + (16,) * 41 + (18,) * 13

        workload = forecast_pitcher(
            _starter_pitcher(
                eligible_outs=eligible_outs,
                prior_outs=(12, 16, 18),
                pitch_limit=80,
            ),
            config,
        ).workload

        self.assertEqual(
            tuple(len(support) for support in workload.supports),
            (71, 42, 14),
        )
        probabilities = workload.bucket_probabilities
        complement = 0.5785434195180372
        self.assertEqual(
            probabilities,
            (
                math.nextafter(complement, math.inf),
                0.34098182059558113,
                0.08047475988638157,
            ),
        )
        self.assertEqual(math.fsum(probabilities), 1.0)

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

    def test_projected_ra9_must_be_positive_before_clipping(self):
        for projected_ra9 in (0.0, -0.01):
            with self.subTest(projected_ra9=projected_ra9):
                with self.assertRaisesRegex(
                    ValueError,
                    "projected_ra9.value.*positive",
                ):
                    forecast_pitcher(
                        _starter_pitcher(projected_ra9=projected_ra9),
                        SimulationConfig(),
                    )

        forecast = forecast_pitcher(
            _starter_pitcher(projected_ra9=0.001),
            SimulationConfig(),
        )
        self.assertEqual(forecast.run_prevention_multiplier, 0.55)

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

    def test_chain_sigma_is_expected_outs_weighted_with_one_handoff_penalty(self):
        config = SimulationConfig()
        chain = _opener_chain(
            opener_outs=(3, 3, 6, 9, 12),
            final_outs=(18, 21, 24, 27, 27),
            opener_log_standard_error=0.08,
            final_log_standard_error=0.52,
        )
        verified = forecast_pitcher_chain(chain, config)
        expected_outs = tuple(
            pitcher.workload.expected_outs for pitcher in verified.pitchers
        )

        self.assertEqual(expected_outs, (9.375, 19.875))
        weighted_rms = math.sqrt(
            (
                expected_outs[0] * 0.08 ** 2
                + expected_outs[1] * 0.52 ** 2
            )
            / sum(expected_outs)
        )
        unweighted_rms = math.sqrt((0.08 ** 2 + 0.52 ** 2) / 2)
        self.assertNotAlmostEqual(weighted_rms, unweighted_rms, places=6)
        self.assertAlmostEqual(verified.log_standard_error, weighted_rms)

        uncertain_chains = (
            replace(chain, handoff_verified=_starter_boolean(False)),
            replace(chain, timing_uncertain=_starter_boolean(True)),
            replace(
                chain,
                handoff_verified=_starter_boolean(False),
                timing_uncertain=_starter_boolean(True),
            ),
        )
        one_penalty = math.hypot(
            weighted_rms,
            config.handoff_uncertainty_log_se_penalty,
        )
        double_penalty = math.sqrt(
            weighted_rms ** 2
            + 2 * config.handoff_uncertainty_log_se_penalty ** 2
        )

        for uncertain_chain in uncertain_chains:
            with self.subTest(
                handoff_verified=uncertain_chain.handoff_verified.value,
                timing_uncertain=uncertain_chain.timing_uncertain.value,
            ):
                uncertain = forecast_pitcher_chain(uncertain_chain, config)
                self.assertAlmostEqual(
                    uncertain.log_standard_error,
                    one_penalty,
                )
                self.assertNotAlmostEqual(
                    uncertain.log_standard_error,
                    double_penalty,
                    places=12,
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

    def test_sample_plan_draws_combined_total_before_component_workload(self):
        combined_workload = WorkloadDistribution(
            probabilities=(1.0, 0.0, 0.0),
            supports=((4, 12), (15,), (18,)),
            source="combined_chain_history",
            notes=(),
        )
        opener_workload = WorkloadDistribution(
            probabilities=(1.0, 0.0, 0.0),
            supports=((2, 10), (15,), (18,)),
            source="pitcher_history",
            notes=(),
        )
        opener = PitcherForecast(
            pitcher_id="pitcher:opener",
            role="opener",
            workload=opener_workload,
            run_prevention_multiplier=0.9,
            log_standard_error=0.2,
            command_risk=False,
            home_run_risk=False,
            diagnostics=(),
        )
        bulk = replace(
            opener,
            pitcher_id="pitcher:bulk",
            role="bulk",
        )
        forecast = PitcherChainForecast(
            pitchers=(opener, bulk),
            combined_workload=combined_workload,
            log_standard_error=0.2,
            diagnostics=(),
        )
        # Combined-first selects total 12 then opener 2; reversed order yields total 4.
        rng = SequenceRandomRng((0.0, 0.99, 0.0, 0.01))

        plan = forecast.sample_plan(rng)

        self.assertEqual(rng.calls, [0.0, 0.99, 0.0, 0.01])
        self.assertEqual(plan.combined_outs, 12)
        self.assertEqual([segment.outs for segment in plan.segments], [2, 10])
        self.assertEqual(
            [(segment.start_out, segment.end_out) for segment in plan.segments],
            [(0, 2), (2, 12)],
        )
        self.assertEqual(
            plan.segments[-1].outs,
            plan.combined_outs - plan.segments[0].outs,
        )
        self.assertEqual(
            sum(segment.outs for segment in plan.segments),
            plan.combined_outs,
        )

    def test_low_total_three_pitcher_plans_preserve_combined_first_allocation(self):
        component_workload = WorkloadDistribution(
            probabilities=(1.0, 0.0, 0.0),
            supports=((0, 1, 2), (15,), (18,)),
            source="pitcher_history",
            notes=(),
        )
        combined_workload = replace(
            component_workload,
            source="combined_chain_history",
        )
        opener = PitcherForecast(
            pitcher_id="pitcher:opener",
            role="opener",
            workload=component_workload,
            run_prevention_multiplier=1.0,
            log_standard_error=0.2,
            command_risk=False,
            home_run_risk=False,
            diagnostics=(),
        )
        bulk = replace(
            opener,
            pitcher_id="pitcher:bulk",
            role="bulk",
        )
        prim = replace(
            opener,
            pitcher_id="pitcher:prim",
            role="prim",
        )
        forecast = PitcherChainForecast(
            pitchers=(opener, bulk, prim),
            combined_workload=combined_workload,
            log_standard_error=0.2,
            diagnostics=(),
        )
        cases = (
            (0, (0.0, 0.0, 0.0, 0.99, 0.0, 0.99), (0, 0, 0)),
            (1, (0.0, 0.50, 0.0, 0.0, 0.0, 0.99), (0, 1, 0)),
            (2, (0.0, 0.99, 0.0, 0.50, 0.0, 0.0), (1, 0, 1)),
        )

        for total_outs, random_values, expected_segment_outs in cases:
            with self.subTest(total_outs=total_outs):
                rng = SequenceRandomRng(random_values)

                plan = forecast.sample_plan(rng)

                self.assertEqual(rng.calls, list(random_values))
                self.assertEqual(plan.combined_outs, total_outs)
                self.assertEqual(
                    tuple(segment.role for segment in plan.segments),
                    ("opener", "bulk", "prim"),
                )
                self.assertEqual(
                    tuple(segment.outs for segment in plan.segments),
                    expected_segment_outs,
                )
                cursor = 0
                for segment in plan.segments:
                    self.assertEqual(segment.start_out, cursor)
                    self.assertEqual(segment.end_out, cursor + segment.outs)
                    cursor = segment.end_out
                self.assertEqual(cursor, total_outs)
                self.assertEqual(
                    plan.segments[-1].outs,
                    total_outs - sum(expected_segment_outs[:-1]),
                )
                self.assertEqual(
                    sum(segment.outs for segment in plan.segments),
                    total_outs,
                )
                self.assertTrue(
                    any(segment.outs == 0 for segment in plan.segments)
                )

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
        expected_probabilities = (
            1 / 6,
            1 / 6,
            1.0 - math.fsum((1 / 6, 1 / 6)),
        )
        self.assertEqual(
            forecast.combined_workload.bucket_probabilities,
            expected_probabilities,
        )
        self.assertEqual(
            math.fsum(forecast.combined_workload.bucket_probabilities),
            1.0,
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

    def test_direct_result_constructors_accept_valid_workload_relationships(self):
        starter_workload = WorkloadDistribution(
            probabilities=(1 / 3, 1 / 3, 1 / 3),
            supports=((12,), (15,), (18,)),
            source="pitcher_history",
            notes=(),
        )
        starter = PitcherForecast(
            pitcher_id="pitcher:starter",
            role="starter",
            workload=starter_workload,
            run_prevention_multiplier=1.0,
            log_standard_error=0.2,
            command_risk=False,
            home_run_risk=False,
            diagnostics=(),
        )
        equal_workload = replace(starter_workload)

        single = PitcherChainForecast(
            pitchers=(starter,),
            combined_workload=equal_workload,
            log_standard_error=0.2,
            diagnostics=(),
        )

        self.assertIsNot(single.combined_workload, starter.workload)
        self.assertEqual(single.combined_workload, starter.workload)

        opener = replace(
            starter,
            pitcher_id="pitcher:opener",
            role="opener",
        )
        bulk = replace(
            starter,
            pitcher_id="pitcher:bulk",
            role="bulk",
        )
        prim = replace(
            starter,
            pitcher_id="pitcher:prim",
            role="prim",
        )
        combined_workload = replace(
            starter_workload,
            source="combined_chain_history",
        )

        multi = PitcherChainForecast(
            pitchers=(opener, bulk, prim),
            combined_workload=combined_workload,
            log_standard_error=0.2,
            diagnostics=(),
        )

        self.assertEqual(
            tuple(pitcher.role for pitcher in multi.pitchers),
            ("opener", "bulk", "prim"),
        )
        self.assertEqual(multi.combined_workload.source, "combined_chain_history")

    def test_pitcher_forecast_rejects_combined_chain_workload_source(self):
        combined_workload = WorkloadDistribution(
            probabilities=(1 / 3, 1 / 3, 1 / 3),
            supports=((12,), (15,), (18,)),
            source="combined_chain_history",
            notes=(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "workload.source.*pitcher_history",
        ):
            PitcherForecast(
                pitcher_id="pitcher:starter",
                role="starter",
                workload=combined_workload,
                run_prevention_multiplier=1.0,
                log_standard_error=0.2,
                command_risk=False,
                home_run_risk=False,
                diagnostics=(),
            )

    def test_pitcher_chain_forecast_rejects_invalid_direct_relationships(self):
        starter_workload = WorkloadDistribution(
            probabilities=(1 / 3, 1 / 3, 1 / 3),
            supports=((12,), (15,), (18,)),
            source="pitcher_history",
            notes=(),
        )
        different_pitcher_workload = replace(
            starter_workload,
            probabilities=(1.0, 0.0, 0.0),
        )
        combined_workload = replace(
            starter_workload,
            source="combined_chain_history",
        )
        starter = PitcherForecast(
            pitcher_id="pitcher:starter",
            role="starter",
            workload=starter_workload,
            run_prevention_multiplier=1.0,
            log_standard_error=0.2,
            command_risk=False,
            home_run_risk=False,
            diagnostics=(),
        )
        opener = replace(
            starter,
            pitcher_id="pitcher:opener",
            role="opener",
        )
        bulk = replace(
            starter,
            pitcher_id="pitcher:bulk",
            role="bulk",
        )
        second_bulk = replace(bulk, pitcher_id="pitcher:bulk-2")
        late_starter = replace(starter, pitcher_id="pitcher:late-starter")

        invalid_builders = (
            (
                "empty",
                "pitchers.*nonempty",
                lambda: PitcherChainForecast(
                    pitchers=(),
                    combined_workload=starter_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "single role",
                "single.*starter",
                lambda: PitcherChainForecast(
                    pitchers=(opener,),
                    combined_workload=starter_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "single workload mismatch",
                "single.*equal.*workload",
                lambda: PitcherChainForecast(
                    pitchers=(starter,),
                    combined_workload=different_pitcher_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "single source",
                "single.*pitcher_history",
                lambda: PitcherChainForecast(
                    pitchers=(starter,),
                    combined_workload=combined_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "multi first role",
                "multi.*opener",
                lambda: PitcherChainForecast(
                    pitchers=(bulk, opener),
                    combined_workload=combined_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "multi duplicate later role",
                "later.*distinct",
                lambda: PitcherChainForecast(
                    pitchers=(opener, bulk, second_bulk),
                    combined_workload=combined_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "multi later starter",
                "later.*bulk or prim",
                lambda: PitcherChainForecast(
                    pitchers=(opener, late_starter),
                    combined_workload=combined_workload,
                    log_standard_error=0.2,
                ),
            ),
            (
                "multi source",
                "multi.*combined_chain_history",
                lambda: PitcherChainForecast(
                    pitchers=(opener, bulk),
                    combined_workload=starter_workload,
                    log_standard_error=0.2,
                ),
            ),
        )

        for label, message, builder in invalid_builders:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    builder()

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


class OffenseForecastTests(unittest.TestCase):
    def test_all_slots_for_each_profile_are_sorted_by_order_then_profile(self):
        opener = _offense_profile("pitcher:opener")
        bulk = _offense_profile("pitcher:bulk")
        ascending = _offense_lineup()
        lineup = replace(ascending, slots=tuple(reversed(ascending.slots)))

        forecast = forecast_offense(lineup, (opener, bulk), SimulationConfig())

        expected_pairs = tuple(
            (order, pitcher_id)
            for order in range(1, 10)
            for pitcher_id in (opener.pitcher_id, bulk.pitcher_id)
        )
        self.assertEqual(len(forecast.slot_forecasts), 18)
        self.assertEqual(
            tuple(
                (slot.order, slot.pitcher_id)
                for slot in forecast.slot_forecasts
            ),
            expected_pairs,
        )
        expected_sides = {
            slot.order: slot.expected_side for slot in ascending.slots
        }
        self.assertEqual(
            tuple(slot.expected_side for slot in forecast.slot_forecasts),
            tuple(
                expected_sides[order]
                for order in range(1, 10)
                for _ in range(2)
            ),
        )
        for pitcher in (opener, bulk):
            diagnostics = forecast.slot_forecasts_for(pitcher.pitcher_id)
            self.assertEqual(len(diagnostics), 9)
            self.assertEqual(tuple(slot.order for slot in diagnostics), tuple(range(1, 10)))

    def test_only_expected_side_pitch_mix_is_evaluated(self):
        pitcher_id = "pitcher:lefty"
        profile = _offense_profile(
            pitcher_id,
            (
                _offense_usage(pitcher_id, "R", "FF", 0.6, 94.5),
                _offense_usage(pitcher_id, "R", "CH", 0.4, 86.0),
                _offense_usage(pitcher_id, "L", "SL", 0.7, 84.0),
                _offense_usage(pitcher_id, "L", "SI", 0.3, 93.0),
            ),
        )

        forecast = forecast_offense(
            _offense_lineup(),
            (profile,),
            SimulationConfig(),
        )
        by_order = {
            slot.order: slot for slot in forecast.slot_forecasts_for(pitcher_id)
        }

        self.assertEqual(by_order[1].expected_side, "R")
        self.assertEqual(by_order[1].evaluated_pitch_types, ("FF", "CH"))
        self.assertEqual(set(by_order[1].average_velocities_mph), {"FF", "CH"})
        self.assertEqual(by_order[2].expected_side, "L")
        self.assertEqual(by_order[2].evaluated_pitch_types, ("SL", "SI"))
        self.assertEqual(set(by_order[2].average_velocities_mph), {"SL", "SI"})

    def test_exact_band_precedes_fallback_regardless_of_pa_or_delta(self):
        pitcher_id = "pitcher:starter"
        lineup = _offense_with_responses(
            _offense_lineup(),
            {
                1: (
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=96.4,
                        band_width_mph=1.5,
                        plate_appearances=5,
                        adjusted_woba=0.341,
                        slot_label="플러스",
                    ),
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=95.0,
                        band_width_mph=3.0,
                        plate_appearances=999,
                        adjusted_woba=0.111,
                        slot_label="마이너스",
                    ),
                ),
            },
        )

        slot = forecast_offense(
            lineup,
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        ).slot_forecasts[0]

        self.assertEqual(slot.selected_band_widths["FF"], 1.5)
        self.assertAlmostEqual(slot.velocity_deltas_mph["FF"], 1.4)
        self.assertEqual(slot.plate_appearances["FF"], 5)
        self.assertEqual(slot.adjusted_wobas["FF"], 0.341)
        self.assertEqual(slot.labels, ("플러스",))

    def test_nominal_exact_outside_band_uses_valid_fallback(self):
        pitcher_id = "pitcher:starter"
        lineup = _offense_with_responses(
            _offense_lineup(),
            {
                1: (
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=96.500001,
                        band_width_mph=1.5,
                        adjusted_woba=0.600,
                    ),
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=97.8,
                        band_width_mph=3.0,
                        plate_appearances=12,
                        adjusted_woba=0.355,
                        slot_label="플러스",
                    ),
                ),
            },
        )

        slot = forecast_offense(
            lineup,
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        ).slot_forecasts[0]

        self.assertEqual(slot.selected_band_widths["FF"], 3.0)
        self.assertAlmostEqual(slot.velocity_deltas_mph["FF"], 2.8)
        self.assertEqual(slot.adjusted_wobas["FF"], 0.355)
        self.assertEqual(slot.labels, ("플러스",))

    def test_candidates_outside_both_bands_are_neutral_unknown(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig()
        lineup = _offense_with_responses(
            _offense_lineup(),
            {
                1: (
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=96.500001,
                        band_width_mph=1.5,
                        adjusted_woba=0.900,
                    ),
                    _offense_response(
                        1,
                        pitcher_id,
                        velocity_center_mph=98.000001,
                        band_width_mph=3.0,
                        adjusted_woba=0.900,
                    ),
                ),
            },
        )

        slot = forecast_offense(
            lineup,
            (_offense_profile(pitcher_id),),
            config,
        ).slot_forecasts[0]

        self.assertIsNone(slot.selected_band_widths["FF"])
        self.assertIsNone(slot.velocity_deltas_mph["FF"])
        self.assertEqual(slot.plate_appearances["FF"], 0)
        self.assertEqual(slot.adjusted_wobas["FF"], config.league_woba)
        self.assertEqual(slot.labels, ("표본 부족",))
        self.assertIsNone(slot.sample_strengths["FF"])
        self.assertEqual(slot.unknown_exposure, 1.0)

    def test_candidate_ties_use_delta_then_pa_then_stable_fields(self):
        pitcher_id = "pitcher:starter"
        profile = _offense_profile(
            pitcher_id,
            (
                _offense_usage(pitcher_id, "R", "FF", 1 / 3, 95.0),
                _offense_usage(pitcher_id, "R", "CH", 1 / 3, 85.0),
                _offense_usage(pitcher_id, "R", "SL", 1 / 3, 85.0),
            ),
        )
        responses = (
            _offense_response(1, pitcher_id, pitch_type="FF", velocity_center_mph=95.2, plate_appearances=4, adjusted_woba=0.331),
            _offense_response(1, pitcher_id, pitch_type="FF", velocity_center_mph=95.4, plate_appearances=100, adjusted_woba=0.399),
            _offense_response(1, pitcher_id, pitch_type="CH", velocity_center_mph=84.5, plate_appearances=5, adjusted_woba=0.300),
            _offense_response(1, pitcher_id, pitch_type="CH", velocity_center_mph=85.5, plate_appearances=20, adjusted_woba=0.350),
            _offense_response(1, pitcher_id, pitch_type="SL", velocity_center_mph=86.0, plate_appearances=10, adjusted_woba=0.370),
            _offense_response(1, pitcher_id, pitch_type="SL", velocity_center_mph=84.0, plate_appearances=10, adjusted_woba=0.290),
        )
        first_lineup = _offense_with_responses(_offense_lineup(), {1: responses})
        reversed_lineup = _offense_with_responses(
            _offense_lineup(),
            {1: tuple(reversed(responses))},
        )

        first = forecast_offense(first_lineup, (profile,), SimulationConfig()).slot_forecasts[0]
        reversed_result = forecast_offense(
            reversed_lineup,
            (profile,),
            SimulationConfig(),
        ).slot_forecasts[0]

        self.assertEqual(first.adjusted_wobas["FF"], 0.331)
        self.assertEqual(first.plate_appearances["CH"], 20)
        self.assertEqual(first.adjusted_wobas["CH"], 0.350)
        self.assertEqual(first.adjusted_wobas["SL"], 0.290)
        self.assertEqual(first, reversed_result)

    def test_usage_normalization_excludes_zero_and_zero_or_no_side_is_unknown(self):
        config = SimulationConfig()
        pitcher_id = "pitcher:normalized"
        normalized = _offense_profile(
            pitcher_id,
            (
                _offense_usage(pitcher_id, "R", "FF", 0.2, 95.0),
                _offense_usage(pitcher_id, "R", "CH", 0.3, 85.0),
                _offense_usage(pitcher_id, "R", "SL", 0.0, 84.0),
            ),
        )
        zero_id = "pitcher:zero"
        all_zero = _offense_profile(
            zero_id,
            (_offense_usage(zero_id, "R", "FF", 0.0, 95.0),),
        )
        no_side_id = "pitcher:no-side"
        no_side = _offense_profile(
            no_side_id,
            (_offense_usage(no_side_id, "L", "FF", 1.0, 95.0),),
        )
        lineup = _offense_with_responses(
            _offense_lineup(),
            {
                1: (
                    _offense_response(
                        1,
                        pitcher_id,
                        adjusted_woba=0.400,
                        slot_label="플러스",
                    ),
                ),
            },
        )

        forecast = forecast_offense(
            lineup,
            (normalized, all_zero, no_side),
            config,
        )
        normalized_slot = forecast.slot_forecasts_for(pitcher_id)[0]
        zero_slot = forecast.slot_forecasts_for(zero_id)[0]
        no_side_slot = forecast.slot_forecasts_for(no_side_id)[0]

        self.assertEqual(normalized_slot.evaluated_pitch_types, ("FF", "CH"))
        self.assertEqual(
            normalized_slot.normalized_pitch_usages,
            {"FF": 0.4, "CH": 0.6},
        )
        self.assertEqual(
            math.fsum(normalized_slot.normalized_pitch_usages.values()),
            1.0,
        )
        self.assertAlmostEqual(
            normalized_slot.exposure_weighted_adjusted_woba,
            0.4 * 0.400 + 0.6 * config.league_woba,
        )
        self.assertEqual(normalized_slot.unknown_exposure, 0.6)
        for neutral_slot in (zero_slot, no_side_slot):
            self.assertEqual(neutral_slot.evaluated_pitch_types, ())
            self.assertEqual(neutral_slot.normalized_pitch_usages, {})
            self.assertEqual(neutral_slot.exposure_weighted_adjusted_woba, config.league_woba)
            self.assertEqual(neutral_slot.unknown_exposure, 1.0)
            self.assertEqual(neutral_slot.labels, ("표본 부족",))

    def test_all_pa_weights_apply_by_batting_order_not_input_position(self):
        pitcher_id = "pitcher:starter"
        raw_weights = (9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
        config = SimulationConfig(
            pa_weights=raw_weights,
            matchup_bounds=(0.10, 2.0),
        )
        ascending = _offense_lineup()
        responses = {
            slot.order: (
                _offense_response(
                    slot.order,
                    pitcher_id,
                    expected_side=slot.expected_side,
                    adjusted_woba=0.25 + slot.order * 0.01,
                ),
            )
            for slot in ascending.slots
        }
        lineup = _offense_with_responses(ascending, responses)
        lineup = replace(lineup, slots=tuple(reversed(lineup.slots)))

        forecast = forecast_offense(
            lineup,
            (_offense_profile(pitcher_id),),
            config,
        )
        expected_woba = math.fsum(
            raw_weights[order - 1] * (0.25 + order * 0.01)
            for order in range(1, 10)
        ) / math.fsum(raw_weights)

        self.assertAlmostEqual(
            forecast.matchup_multipliers[pitcher_id],
            expected_woba / config.league_woba,
        )
        self.assertEqual(
            tuple(slot.order for slot in forecast.slot_forecasts),
            tuple(range(1, 10)),
        )

    def test_missing_sample_is_neutral_and_only_widens_sigma_exactly(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig()
        profile = _offense_profile(
            pitcher_id,
            tuple(
                usage
                for side in ("R", "L")
                for usage in (
                    _offense_usage(pitcher_id, side, "FF", 0.25, 95.0),
                    _offense_usage(pitcher_id, side, "CH", 0.75, 85.0),
                )
            ),
        )
        base = _offense_lineup(log_standard_error=0.09)
        missing_responses = {
            slot.order: (
                _offense_response(
                    slot.order,
                    pitcher_id,
                    expected_side=slot.expected_side,
                    pitch_type="FF",
                    adjusted_woba=config.league_woba,
                ),
            )
            for slot in base.slots
        }
        complete_responses = {
            slot.order: missing_responses[slot.order]
            + (
                _offense_response(
                    slot.order,
                    pitcher_id,
                    expected_side=slot.expected_side,
                    pitch_type="CH",
                    velocity_center_mph=85.0,
                    adjusted_woba=config.league_woba,
                ),
            )
            for slot in base.slots
        }

        missing = forecast_offense(
            _offense_with_responses(base, missing_responses),
            (profile,),
            config,
        )
        complete = forecast_offense(
            _offense_with_responses(base, complete_responses),
            (profile,),
            config,
        )

        self.assertEqual(missing.matchup_multipliers[pitcher_id], 1.0)
        self.assertEqual(complete.matchup_multipliers[pitcher_id], 1.0)
        self.assertEqual(missing.matchup_unknown_exposures[pitcher_id], 0.75)
        self.assertEqual(complete.matchup_unknown_exposures[pitcher_id], 0.0)
        self.assertAlmostEqual(
            missing.matchup_log_standard_errors[pitcher_id],
            math.hypot(0.09, 0.75 * config.sample_shortage_log_se_penalty),
        )
        self.assertEqual(complete.matchup_log_standard_errors[pitcher_id], 0.09)
        self.assertGreater(missing.log_standard_error, complete.log_standard_error)
        self.assertTrue(hasattr(missing, "lineup_log_standard_error"))
        self.assertEqual(missing.lineup_log_standard_error, 0.09)
        with self.assertRaises(FrozenInstanceError):
            missing.lineup_log_standard_error = 0.50

    def test_selected_values_and_fixed_labels_are_carried_without_reshrinking(self):
        pitcher_id = "pitcher:starter"
        pitches = ("FF", "CH", "SL", "SI")
        velocities = (95.0, 85.0, 84.0, 93.0)
        wobas = (0.111, 0.222, 0.333, 0.444)
        labels = ("플러스", "중립", "마이너스", "표본 부족")
        strengths = ("s1", "s2", "s3", "s4")
        profile = _offense_profile(
            pitcher_id,
            tuple(
                _offense_usage(pitcher_id, "R", pitch, 0.25, velocity)
                for pitch, velocity in zip(pitches, velocities)
            ),
        )
        lineup = _offense_with_responses(
            _offense_lineup(),
            {
                1: tuple(
                    _offense_response(
                        1,
                        pitcher_id,
                        pitch_type=pitch,
                        velocity_center_mph=velocity,
                        plate_appearances=index,
                        adjusted_woba=woba,
                        sample_strength=strength,
                        slot_label=label,
                    )
                    for index, (pitch, velocity, woba, strength, label) in enumerate(
                        zip(pitches, velocities, wobas, strengths, labels),
                        start=11,
                    )
                ),
            },
        )

        slot = forecast_offense(lineup, (profile,), SimulationConfig()).slot_forecasts[0]

        self.assertEqual(slot.adjusted_wobas, dict(zip(pitches, wobas)))
        self.assertEqual(slot.labels, labels)
        self.assertEqual(slot.sample_strengths, dict(zip(pitches, strengths)))
        self.assertEqual(slot.plate_appearances, dict(zip(pitches, range(11, 15))))
        self.assertAlmostEqual(
            slot.exposure_weighted_adjusted_woba,
            math.fsum(wobas) / len(wobas),
        )
        self.assertEqual(slot.unknown_exposure, 0.0)

    def test_base_and_matchup_multipliers_clip_high_and_low(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig()
        cases = (
            (0.001, 0.001, config.offense_bounds[0], config.matchup_bounds[0]),
            (100.0, 0.999, config.offense_bounds[1], config.matchup_bounds[1]),
        )

        for projected_runs, woba, expected_base, expected_matchup in cases:
            with self.subTest(projected_runs=projected_runs, woba=woba):
                lineup = _offense_uniform_response_lineup(
                    pitcher_id,
                    projected_runs_per_9=projected_runs,
                    adjusted_woba=woba,
                )
                forecast = forecast_offense(
                    lineup,
                    (_offense_profile(pitcher_id),),
                    config,
                )
                self.assertEqual(forecast.base_offense_multiplier, expected_base)
                self.assertEqual(
                    forecast.matchup_multipliers[pitcher_id],
                    expected_matchup,
                )

    def test_traffic_and_extra_base_hit_projections_are_carried_exactly(self):
        pitcher_id = "pitcher:starter"
        lineup = _offense_lineup(
            projected_traffic_per_9=13.75,
            projected_extra_base_hits_per_9=3.125,
        )

        forecast = forecast_offense(
            lineup,
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        )

        self.assertEqual(forecast.projected_traffic_per_9, 13.75)
        self.assertEqual(forecast.projected_extra_base_hits_per_9, 3.125)

    def test_forecast_rejects_wrong_types_and_invalid_numeric_domains(self):
        profile = _offense_profile()
        config = SimulationConfig()
        wrong_calls = (
            lambda: forecast_offense(None, (profile,), config),
            lambda: forecast_offense(_offense_lineup(), [profile], config),
            lambda: forecast_offense(_offense_lineup(), (object(),), config),
            lambda: forecast_offense(_offense_lineup(), (profile,), object()),
        )
        for call in wrong_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()
        with self.assertRaisesRegex(ValueError, "opposing_profiles.*nonempty"):
            forecast_offense(_offense_lineup(), (), config)

        invalid_lineup_values = (
            ("projected_runs_per_9", 0.0),
            ("projected_runs_per_9", -0.1),
            ("projected_traffic_per_9", -0.1),
            ("projected_extra_base_hits_per_9", -0.1),
            ("log_standard_error", -0.1),
            ("projected_runs_per_9", math.nan),
            ("projected_traffic_per_9", math.inf),
            ("projected_extra_base_hits_per_9", -math.inf),
            ("log_standard_error", math.nan),
        )
        for field_name, value in invalid_lineup_values:
            with self.subTest(field_name=field_name, value=value):
                lineup = _offense_lineup()
                object.__setattr__(getattr(lineup, field_name), "value", value)
                with self.assertRaisesRegex(ValueError, field_name):
                    forecast_offense(lineup, (profile,), config)

        for adjusted_woba in (-0.1, math.nan, math.inf):
            with self.subTest(adjusted_woba=adjusted_woba):
                response = _offense_response(1, profile.pitcher_id)
                object.__setattr__(response, "adjusted_woba", adjusted_woba)
                lineup = _offense_with_responses(
                    _offense_lineup(),
                    {1: (response,)},
                )
                with self.assertRaisesRegex(ValueError, "adjusted_woba"):
                    forecast_offense(lineup, (profile,), config)

    def test_multiple_profiles_keep_independent_points_sigmas_and_diagnostics(self):
        first_id = "pitcher:first"
        second_id = "pitcher:second"
        first = _offense_profile(first_id)
        second = _offense_profile(
            second_id,
            tuple(
                usage
                for side in ("R", "L")
                for usage in (
                    _offense_usage(second_id, side, "FF", 0.5, 95.0),
                    _offense_usage(second_id, side, "CH", 0.5, 85.0),
                )
            ),
        )
        base = _offense_lineup(log_standard_error=0.10)
        lineup = _offense_with_responses(
            base,
            {
                slot.order: (
                    _offense_response(
                        slot.order,
                        first_id,
                        expected_side=slot.expected_side,
                        adjusted_woba=0.400,
                        slot_label="플러스",
                    ),
                    _offense_response(
                        slot.order,
                        second_id,
                        expected_side=slot.expected_side,
                        adjusted_woba=0.320,
                    ),
                )
                for slot in base.slots
            },
        )
        config = SimulationConfig(matchup_bounds=(0.5, 1.5))

        forecast = forecast_offense(lineup, (first, second), config)

        self.assertEqual(forecast.matchup_multipliers[first_id], 1.25)
        self.assertEqual(forecast.matchup_multipliers[second_id], 1.0)
        self.assertEqual(forecast.matchup_unknown_exposures[first_id], 0.0)
        self.assertEqual(forecast.matchup_unknown_exposures[second_id], 0.5)
        self.assertEqual(forecast.matchup_log_standard_errors[first_id], 0.10)
        self.assertAlmostEqual(
            forecast.matchup_log_standard_errors[second_id],
            math.hypot(0.10, 0.5 * config.sample_shortage_log_se_penalty),
        )
        self.assertEqual(len(forecast.slot_forecasts_for(first_id)), 9)
        self.assertEqual(len(forecast.slot_forecasts_for(second_id)), 9)
        self.assertEqual(set(forecast.matchup_multipliers), {first_id, second_id})
        self.assertEqual(set(forecast.matchup_log_standard_errors), {first_id, second_id})

    def test_unknown_bullpen_is_neutral_and_adds_penalty_in_quadrature(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig()
        forecast = forecast_offense(
            _offense_uniform_response_lineup(
                pitcher_id,
                adjusted_woba=0.400,
                log_standard_error=0.11,
            ),
            (_offense_profile(pitcher_id),),
            config,
        )

        self.assertNotEqual(forecast.multiplier_for(pitcher_id), 1.0)
        self.assertEqual(forecast.multiplier_for("bullpen:unknown"), 1.0)
        self.assertEqual(
            forecast.log_standard_error_for(pitcher_id),
            forecast.matchup_log_standard_errors[pitcher_id],
        )
        self.assertAlmostEqual(
            forecast.log_standard_error_for("bullpen:unknown"),
            math.hypot(
                forecast.log_standard_error,
                config.unknown_bullpen_matchup_log_se_penalty,
            ),
        )
        self.assertEqual(forecast.slot_forecasts_for("bullpen:unknown"), ())

    def test_preblended_component_fields_are_not_multiplied_again(self):
        pitcher_id = "pitcher:starter"
        baseline = _offense_uniform_response_lineup(pitcher_id)
        altered = _offense_uniform_response_lineup(
            pitcher_id,
            walk_rate=0.99,
            strikeout_rate=0.01,
            isolated_power=0.88,
            barrel_rate=0.77,
            recent_form_index=4.0,
            lower_order_continuity=0.01,
            recent_form_included=False,
        )
        profile = _offense_profile(pitcher_id)

        first = forecast_offense(baseline, (profile,), SimulationConfig())
        second = forecast_offense(altered, (profile,), SimulationConfig())

        self.assertEqual(first, second)
        result_fields = {field.name for field in fields(OffenseForecast)}
        self.assertNotIn("recent_form_multiplier", result_fields)
        self.assertFalse(hasattr(first, "recent_form_multiplier"))

    def test_result_mappings_are_defensively_immutable_and_instances_frozen(self):
        pitcher_id = "pitcher:starter"
        result = forecast_offense(
            _offense_uniform_response_lineup(pitcher_id),
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        )
        slot = result.slot_forecasts[0]

        for field_name in (
            "normalized_pitch_usages",
            "average_velocities_mph",
            "selected_band_widths",
            "velocity_deltas_mph",
            "plate_appearances",
            "adjusted_wobas",
            "sample_strengths",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError):
                    getattr(slot, field_name)["FF"] = None
        for field_name in (
            "matchup_multipliers",
            "matchup_log_standard_errors",
            "matchup_unknown_exposures",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError):
                    getattr(result, field_name)[pitcher_id] = 99.0

        usages = dict(slot.normalized_pitch_usages)
        direct_slot = replace(slot, normalized_pitch_usages=usages)
        usages["FF"] = 0.5
        self.assertEqual(direct_slot.normalized_pitch_usages["FF"], 1.0)

        multipliers = dict(result.matchup_multipliers)
        direct_result = replace(result, matchup_multipliers=multipliers)
        multipliers[pitcher_id] = 99.0
        self.assertEqual(direct_result.matchup_multipliers[pitcher_id], 1.0)

        with self.assertRaises(FrozenInstanceError):
            direct_slot.order = 2
        with self.assertRaises(FrozenInstanceError):
            direct_result.base_offense_multiplier = 2.0

    def test_missing_pitch_response_requires_league_woba_point_value(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig(league_woba=0.333)
        result = forecast_offense(
            _offense_lineup(),
            (_offense_profile(pitcher_id),),
            config,
        )
        slot = result.slot_forecasts[0]
        forged_woba = 0.401

        self.assertIsNone(slot.selected_band_widths["FF"])
        self.assertEqual(slot.adjusted_wobas["FF"], config.league_woba)
        with self.assertRaisesRegex(ValueError, "missing.*league_woba"):
            replace(
                slot,
                adjusted_wobas={"FF": forged_woba},
                exposure_weighted_adjusted_woba=forged_woba,
            )

    def test_direct_offense_maps_must_match_slot_diagnostics(self):
        pitcher_id = "pitcher:starter"
        config = SimulationConfig(matchup_bounds=(0.5, 1.5))
        result = forecast_offense(
            _offense_uniform_response_lineup(
                pitcher_id,
                adjusted_woba=0.400,
                log_standard_error=0.10,
            ),
            (_offense_profile(pitcher_id),),
            config,
        )
        inconsistent_changes = (
            (
                "matchup_multipliers",
                {"matchup_multipliers": {pitcher_id: 0.5}},
            ),
            (
                "matchup_unknown_exposures",
                {"matchup_unknown_exposures": {pitcher_id: 0.5}},
            ),
            (
                "matchup_log_standard_errors",
                {
                    "matchup_log_standard_errors": {pitcher_id: 0.9},
                    "log_standard_error": 0.9,
                },
            ),
        )

        self.assertEqual(result.matchup_multipliers[pitcher_id], 1.25)
        self.assertEqual(result.matchup_unknown_exposures[pitcher_id], 0.0)
        self.assertEqual(result.matchup_log_standard_errors[pitcher_id], 0.10)
        for field_name, changes in inconsistent_changes:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name}.*slot diagnostics",
                ):
                    replace(result, **changes)

    def test_public_field_only_construction_requires_factory_context(self):
        pitcher_id = "pitcher:starter"
        result = forecast_offense(
            _offense_uniform_response_lineup(pitcher_id),
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        )
        values = (result.slot_forecasts[0], result)

        for value in values:
            with self.subTest(result_type=type(value).__name__):
                public_values = {
                    result_field.name: getattr(value, result_field.name)
                    for result_field in fields(value)
                    if not result_field.name.startswith("_")
                }
                with self.assertRaisesRegex(ValueError, "forecast_offense"):
                    type(value)(**public_values)

    def test_caller_cannot_forge_offense_invariant_context(self):
        pitcher_id = "pitcher:starter"
        result = forecast_offense(
            _offense_uniform_response_lineup(pitcher_id),
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        )
        context = result._context

        self.assertIs(replace(result)._context, context)
        self.assertIs(replace(result.slot_forecasts[0])._context, context)
        forged_context = offense_module._OffenseInvariantContext(
            league_woba=context.league_woba,
            normalized_pa_weights=context.normalized_pa_weights,
            matchup_bounds=context.matchup_bounds,
            lineup_log_standard_error=context.lineup_log_standard_error,
            sample_shortage_log_se_penalty=(
                context.sample_shortage_log_se_penalty
            ),
        )

        for value in (result.slot_forecasts[0], result):
            with self.subTest(result_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "forecast_offense"):
                    replace(value, _context=forged_context)

    def test_module_exposes_no_arbitrary_context_issuer(self):
        forbidden_names = {
            "_build_offense_context_factory",
            "_build_forecast_offense",
            "_create_offense_invariant_context",
        }

        self.assertEqual(
            forbidden_names.intersection(vars(offense_module)),
            set(),
        )

    def test_offense_forecast_canonical_json_is_deterministic(self):
        pitcher_id = "pitcher:starter"
        lineup = _offense_uniform_response_lineup(pitcher_id)
        profiles = (_offense_profile(pitcher_id),)
        config = SimulationConfig()
        first = forecast_offense(lineup, profiles, config)
        second = forecast_offense(lineup, profiles, config)

        def serialize(value):
            try:
                return canonical_json(value)
            except (TypeError, ValueError) as error:
                self.fail(f"canonical_json rejected OffenseForecast: {error}")

        first_json = serialize(first)
        self.assertEqual(first_json, serialize(second))
        self.assertNotIn(
            "_factory_capability",
            json.loads(first_json)["_context"],
        )

    def test_global_log_standard_error_must_equal_max_profile_sigma(self):
        first_id = "pitcher:opener"
        second_id = "pitcher:bulk"
        result = forecast_offense(
            _offense_uniform_response_lineup(
                first_id,
                log_standard_error=0.10,
            ),
            (
                _offense_profile(first_id),
                _offense_profile(second_id),
            ),
            SimulationConfig(),
        )
        expected = max(result.matchup_log_standard_errors.values())

        self.assertEqual(result.log_standard_error, expected)
        within_tolerance = expected + offense_module._FLOAT_TOLERANCE / 2.0
        self.assertEqual(
            replace(
                result,
                log_standard_error=within_tolerance,
            ).log_standard_error,
            within_tolerance,
        )
        with self.assertRaisesRegex(
            ValueError,
            "log_standard_error.*maximum matchup sigma",
        ):
            replace(result, log_standard_error=99.0)

    def test_direct_result_constructors_reject_invalid_invariants(self):
        pitcher_id = "pitcher:starter"
        result = forecast_offense(
            _offense_uniform_response_lineup(pitcher_id),
            (_offense_profile(pitcher_id),),
            SimulationConfig(),
        )
        slot = result.slot_forecasts[0]
        invalid_slot_changes = (
            {"order": 0},
            {"expected_side": "S"},
            {"evaluated_pitch_types": ("FF", "FF")},
            {"normalized_pitch_usages": {}},
            {"average_velocities_mph": {}},
            {"selected_band_widths": {"FF": 2.0}},
            {"velocity_deltas_mph": {"FF": -0.1}},
            {"plate_appearances": {"FF": -1}},
            {"adjusted_wobas": {"FF": -0.1}},
            {"adjusted_wobas": {"FF": math.nan}},
            {"labels": ()},
            {"labels": ("invented",)},
            {"sample_strengths": {"FF": ""}},
            {"exposure_weighted_adjusted_woba": math.inf},
            {"unknown_exposure": 1.1},
        )
        for changes in invalid_slot_changes:
            with self.subTest(slot_changes=changes):
                with self.assertRaises(ValueError):
                    replace(slot, **changes)

        duplicate_slots = list(result.slot_forecasts)
        duplicate_slots[1] = duplicate_slots[0]
        duplicate_batters = list(result.slot_forecasts)
        duplicate_batters[1] = replace(
            duplicate_batters[1],
            batter_id=duplicate_batters[0].batter_id,
        )
        invalid_result_changes = (
            {"base_offense_multiplier": 0.0},
            {"base_offense_multiplier": math.nan},
            {"matchup_multipliers": {}},
            {"matchup_multipliers": {pitcher_id: 0.0}},
            {"matchup_log_standard_errors": {}},
            {"matchup_log_standard_errors": {pitcher_id: -0.1}},
            {"matchup_unknown_exposures": {}},
            {"matchup_unknown_exposures": {pitcher_id: 1.1}},
            {"projected_traffic_per_9": -0.1},
            {"projected_extra_base_hits_per_9": math.inf},
            {"log_standard_error": -0.1},
            {"log_standard_error": 0.01},
            {"slot_forecasts": list(result.slot_forecasts)},
            {"slot_forecasts": tuple(duplicate_slots)},
            {"slot_forecasts": tuple(duplicate_batters)},
            {"slot_forecasts": result.slot_forecasts[:-1]},
            {"unknown_bullpen_matchup_log_se_penalty": -0.1},
        )
        for changes in invalid_result_changes:
            with self.subTest(result_changes=changes):
                with self.assertRaises(ValueError):
                    replace(result, **changes)

    def test_duplicate_profiles_and_duplicate_side_pitch_usage_are_rejected(self):
        profile = _offense_profile()
        with self.assertRaisesRegex(ValueError, "unique pitcher_id"):
            forecast_offense(
                _offense_lineup(),
                (profile, profile),
                SimulationConfig(),
            )

        pitcher_id = "pitcher:duplicate-mix"
        duplicate_mix = _offense_profile(
            pitcher_id,
            (
                _offense_usage(pitcher_id, "R", "FF", 0.5, 95.0),
                _offense_usage(pitcher_id, "R", "FF", 0.0, 96.0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate pitch usage"):
            forecast_offense(
                _offense_lineup(),
                (duplicate_mix,),
                SimulationConfig(),
            )

    def test_output_is_deterministic_and_profile_order_only_changes_diagnostics(self):
        first_id = "pitcher:first"
        second_id = "pitcher:second"
        profiles = (_offense_profile(first_id), _offense_profile(second_id))
        base = _offense_lineup()
        lineup = _offense_with_responses(
            base,
            {
                slot.order: (
                    _offense_response(
                        slot.order,
                        first_id,
                        expected_side=slot.expected_side,
                        adjusted_woba=0.340,
                    ),
                    _offense_response(
                        slot.order,
                        second_id,
                        expected_side=slot.expected_side,
                        adjusted_woba=0.300,
                    ),
                )
                for slot in base.slots
            },
        )
        config = SimulationConfig()

        first = forecast_offense(lineup, profiles, config)
        repeat = forecast_offense(lineup, profiles, config)
        reversed_profiles = forecast_offense(lineup, tuple(reversed(profiles)), config)

        self.assertEqual(first, repeat)
        self.assertEqual(first.matchup_multipliers, reversed_profiles.matchup_multipliers)
        self.assertEqual(
            first.matchup_log_standard_errors,
            reversed_profiles.matchup_log_standard_errors,
        )
        self.assertEqual(
            first.matchup_unknown_exposures,
            reversed_profiles.matchup_unknown_exposures,
        )
        self.assertEqual(
            tuple(
                (slot.order, slot.pitcher_id)
                for slot in reversed_profiles.slot_forecasts[:4]
            ),
            ((1, second_id), (1, first_id), (2, second_id), (2, first_id)),
        )
        for pitcher_id in (first_id, second_id):
            self.assertEqual(
                first.slot_forecasts_for(pitcher_id),
                reversed_profiles.slot_forecasts_for(pitcher_id),
            )

    def test_responses_for_other_identity_dimensions_do_not_leak(self):
        pitcher_id = "pitcher:starter"
        profile = _offense_profile(
            pitcher_id,
            (_offense_usage(pitcher_id, "R", "FF", 1.0, 95.0),),
        )
        base = _offense_lineup(sides=("R",) * 9)
        lineup = _offense_with_responses(
            base,
            {
                1: (
                    _offense_response(1, "pitcher:other", adjusted_woba=0.900),
                    _offense_response(
                        1,
                        pitcher_id,
                        expected_side="L",
                        adjusted_woba=0.900,
                    ),
                    _offense_response(
                        1,
                        pitcher_id,
                        pitch_type="CH",
                        adjusted_woba=0.900,
                    ),
                ),
                2: (
                    _offense_response(
                        2,
                        pitcher_id,
                        adjusted_woba=0.500,
                        slot_label="플러스",
                    ),
                ),
            },
        )

        diagnostics = forecast_offense(
            lineup,
            (profile,),
            SimulationConfig(),
        ).slot_forecasts_for(pitcher_id)

        self.assertEqual(diagnostics[0].adjusted_wobas["FF"], 0.320)
        self.assertEqual(diagnostics[0].unknown_exposure, 1.0)
        self.assertEqual(diagnostics[0].labels, ("표본 부족",))
        self.assertEqual(diagnostics[1].adjusted_wobas["FF"], 0.500)
        self.assertEqual(diagnostics[1].unknown_exposure, 0.0)


class BullpenForecastTests(unittest.TestCase):
    def test_roles_stay_separate_and_multipliers_use_configured_ra9_and_bounds(self):
        config = SimulationConfig(
            league_runs_per_team_game=5.0,
            bullpen_ra9_bounds=(0.60, 1.90),
        )
        bullpen = _bullpen_input(
            {
                BullpenRole.BRIDGE: {
                    "projected_ra9": _bullpen_numeric(2.0),
                },
                BullpenRole.MIDDLE: {
                    "projected_ra9": _bullpen_numeric(4.5),
                },
                BullpenRole.HIGH_LEVERAGE: {
                    "projected_ra9": _bullpen_numeric(6.0),
                },
                BullpenRole.CLOSER: {
                    "projected_ra9": _bullpen_numeric(12.0),
                },
            }
        )

        forecast = forecast_bullpen(bullpen, config)

        self.assertEqual(
            tuple(segment.role for segment in forecast.segments),
            _BULLPEN_ROLE_ORDER,
        )
        self.assertEqual(tuple(forecast.by_role), _BULLPEN_ROLE_ORDER)
        self.assertEqual(
            {
                role: forecast.by_role[role].run_prevention_multiplier
                for role in _BULLPEN_ROLE_ORDER
            },
            {
                BullpenRole.BRIDGE: 0.60,
                BullpenRole.MIDDLE: 0.90,
                BullpenRole.HIGH_LEVERAGE: 1.20,
                BullpenRole.CLOSER: 1.90,
            },
        )
        self.assertFalse(hasattr(forecast, "team_average_multiplier"))

    def test_projected_ra9_is_rejected_before_clipping_when_not_positive_finite(self):
        for invalid in (0.0, -0.1, math.nan, math.inf, -math.inf, True):
            with self.subTest(invalid=invalid):
                projected_ra9 = _bullpen_numeric(4.5)
                object.__setattr__(projected_ra9, "value", invalid)
                bullpen = _bullpen_input(
                    {
                        BullpenRole.BRIDGE: {
                            "projected_ra9": projected_ra9,
                        }
                    }
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "projected_ra9.*finite positive",
                ):
                    forecast_bullpen(bullpen, SimulationConfig())

    def test_base_log_standard_error_must_be_finite_and_nonnegative(self):
        for invalid in (-0.1, math.nan, math.inf, -math.inf, True):
            with self.subTest(invalid=invalid):
                log_standard_error = _bullpen_numeric(0.2)
                object.__setattr__(log_standard_error, "value", invalid)
                bullpen = _bullpen_input(
                    {
                        BullpenRole.MIDDLE: {
                            "log_standard_error": log_standard_error,
                        }
                    }
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "log_standard_error.*finite nonnegative",
                ):
                    forecast_bullpen(bullpen, SimulationConfig())

    def test_rested_high_leverage_is_distinct_from_used_lower_relief(self):
        tired_lower = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.BRIDGE: {
                        "limited_performance_or_use_risk": _bullpen_boolean(True),
                    }
                }
            ),
            SimulationConfig(),
        )
        protected_only = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "limited_performance_or_use_risk": _bullpen_boolean(True),
                    },
                    BullpenRole.CLOSER: {
                        "available": _bullpen_boolean(False),
                    },
                }
            ),
            SimulationConfig(),
        )
        unavailable_setup = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "available": _bullpen_boolean(False),
                    }
                }
            ),
            SimulationConfig(),
        )

        self.assertTrue(tired_lower.high_leverage_available)
        self.assertTrue(tired_lower.lower_relief_risk)
        self.assertTrue(protected_only.high_leverage_available)
        self.assertFalse(protected_only.lower_relief_risk)
        self.assertFalse(unavailable_setup.high_leverage_available)
        self.assertFalse(unavailable_setup.lower_relief_risk)

    def test_limited_arm_stays_available_and_adds_sigma_in_quadrature(self):
        config = SimulationConfig(limited_reliever_log_se_penalty=0.30)
        forecast = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "log_standard_error": _bullpen_numeric(0.20),
                        "limited_performance_or_use_risk": _bullpen_boolean(True),
                    }
                }
            ),
            config,
        )

        setup = forecast.by_role[BullpenRole.HIGH_LEVERAGE]
        selected = select_bullpen_segment(
            forecast,
            inning=8,
            margin=1,
            bridge_needed=False,
            verified_opener_bulk=False,
        )

        self.assertTrue(setup.available)
        self.assertTrue(setup.limited_performance_or_use_risk)
        self.assertEqual(setup.base_log_standard_error, 0.20)
        self.assertEqual(
            setup.log_standard_error,
            math.sqrt(0.20 ** 2 + 0.30 ** 2),
        )
        self.assertEqual(setup.notes, ("limited_performance_or_use_risk",))
        self.assertIs(selected, setup)

    def test_direct_segment_invariant_uses_approved_sigma_expression(self):
        base_sigma = 1e50
        penalty = 7e50
        forecast = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "log_standard_error": _bullpen_numeric(base_sigma),
                        "limited_performance_or_use_risk": _bullpen_boolean(True),
                    }
                }
            ),
            SimulationConfig(limited_reliever_log_se_penalty=penalty),
        )
        setup = forecast.by_role[BullpenRole.HIGH_LEVERAGE]
        approved_sigma = math.sqrt(base_sigma ** 2 + penalty ** 2)
        hypot_sigma = math.hypot(base_sigma, penalty)

        self.assertNotEqual(approved_sigma, hypot_sigma)
        try:
            rebuilt = replace(setup, log_standard_error=approved_sigma)
        except ValueError as error:
            self.fail(f"approved sigma expression was rejected: {error}")
        self.assertEqual(rebuilt.log_standard_error, approved_sigma)
        with self.assertRaisesRegex(ValueError, "log_standard_error"):
            replace(setup, log_standard_error=hypot_sigma)

    def test_limited_and_unavailable_normalized_flags_are_inconsistent(self):
        bullpen = _bullpen_input(
            {
                BullpenRole.HIGH_LEVERAGE: {
                    "available": _bullpen_boolean(False),
                    "limited_performance_or_use_risk": _bullpen_boolean(True),
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "limited.*available"):
            forecast_bullpen(bullpen, SimulationConfig())

    def test_recent_appearances_are_not_recalculated_into_normalized_flags(self):
        heavy_usage = (
            _bullpen_appearance("reliever:used", 0, 45),
            _bullpen_appearance("reliever:used", 1, 35),
            _bullpen_appearance("reliever:used", 2, 25),
        )
        used = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.BRIDGE: {"appearances": heavy_usage},
                    BullpenRole.HIGH_LEVERAGE: {"appearances": heavy_usage},
                }
            ),
            SimulationConfig(),
        )
        baseline = forecast_bullpen(_bullpen_input(), SimulationConfig())

        self.assertEqual(used, baseline)
        self.assertTrue(used.high_leverage_available)
        self.assertFalse(used.lower_relief_risk)
        self.assertEqual(
            used.by_role[BullpenRole.HIGH_LEVERAGE].log_standard_error,
            used.by_role[BullpenRole.HIGH_LEVERAGE].base_log_standard_error,
        )

    def test_unverified_early_relief_uses_neutral_f5_prior_not_team_relief(self):
        config = SimulationConfig(
            unscheduled_f5_relief_log_standard_error=0.37,
        )
        forecast = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.BRIDGE: {
                        "projected_ra9": _bullpen_numeric(40.0),
                        "available": _bullpen_boolean(False),
                    },
                    BullpenRole.MIDDLE: {
                        "projected_ra9": _bullpen_numeric(0.5),
                        "available": _bullpen_boolean(False),
                    },
                }
            ),
            config,
        )

        for inning in (1, 5):
            with self.subTest(inning=inning):
                segment = select_bullpen_segment(
                    forecast,
                    inning=inning,
                    margin=1,
                    bridge_needed=True,
                    verified_opener_bulk=False,
                )

                self.assertIs(segment, forecast.unscheduled_f5_relief)
                self.assertEqual(segment.segment_id, "league_unscheduled_f5_relief")
                self.assertEqual(segment.run_prevention_multiplier, 1.0)
                self.assertEqual(segment.base_log_standard_error, 0.37)
                self.assertEqual(segment.log_standard_error, 0.37)
                self.assertTrue(segment.synthetic)
                self.assertTrue(segment.available)
                self.assertNotEqual(
                    segment.run_prevention_multiplier,
                    forecast.by_role[BullpenRole.BRIDGE].run_prevention_multiplier,
                )

    def test_verified_early_opener_bulk_stays_in_initial_pitcher_chain(self):
        forecast = forecast_bullpen(_bullpen_input(), SimulationConfig())

        with self.assertRaisesRegex(ValueError, "initial pitcher chain"):
            select_bullpen_segment(
                forecast,
                inning=5,
                margin=1,
                bridge_needed=False,
                verified_opener_bulk=True,
            )

        self.assertEqual(
            select_bullpen_segment(
                forecast,
                inning=6,
                margin=1,
                bridge_needed=False,
                verified_opener_bulk=True,
            ).role,
            BullpenRole.MIDDLE,
        )

    def test_role_selection_follows_inning_margin_and_bridge_state(self):
        forecast = forecast_bullpen(_bullpen_input(), SimulationConfig())
        cases = (
            (6, 1, True, BullpenRole.BRIDGE),
            (6, 1, False, BullpenRole.MIDDLE),
            (7, 1, True, BullpenRole.HIGH_LEVERAGE),
            (7, 1, False, BullpenRole.HIGH_LEVERAGE),
            (7, 4, False, BullpenRole.MIDDLE),
            (8, -3, False, BullpenRole.HIGH_LEVERAGE),
            (8, -4, False, BullpenRole.MIDDLE),
            (8, 4, True, BullpenRole.BRIDGE),
            (9, 3, False, BullpenRole.CLOSER),
            (10, -3, False, BullpenRole.CLOSER),
            (10, 4, False, BullpenRole.MIDDLE),
        )

        for inning, margin, bridge_needed, expected_role in cases:
            with self.subTest(
                inning=inning,
                margin=margin,
                bridge_needed=bridge_needed,
            ):
                selected = select_bullpen_segment(
                    forecast,
                    inning=inning,
                    margin=margin,
                    bridge_needed=bridge_needed,
                    verified_opener_bulk=False,
                )
                self.assertEqual(selected.role, expected_role)

    def test_unavailable_high_leverage_falls_back_without_mutation(self):
        forecast = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "available": _bullpen_boolean(False),
                    }
                }
            ),
            SimulationConfig(),
        )
        before = canonical_json(forecast)

        segment = select_bullpen_segment(
            forecast,
            inning=8,
            margin=1,
            bridge_needed=False,
            verified_opener_bulk=False,
        )

        self.assertEqual(segment.role, BullpenRole.MIDDLE)
        self.assertFalse(forecast.by_role[BullpenRole.HIGH_LEVERAGE].available)
        self.assertEqual(canonical_json(forecast), before)

    def test_requested_roles_fall_back_in_deterministic_order(self):
        def with_unavailable(*roles):
            return forecast_bullpen(
                _bullpen_input(
                    {
                        role: {"available": _bullpen_boolean(False)}
                        for role in roles
                    }
                ),
                SimulationConfig(),
            )

        cases = (
            (
                with_unavailable(BullpenRole.BRIDGE),
                6,
                1,
                True,
                BullpenRole.MIDDLE,
            ),
            (
                with_unavailable(BullpenRole.MIDDLE),
                8,
                4,
                False,
                BullpenRole.BRIDGE,
            ),
            (
                with_unavailable(BullpenRole.CLOSER),
                9,
                1,
                False,
                BullpenRole.HIGH_LEVERAGE,
            ),
            (
                with_unavailable(
                    BullpenRole.CLOSER,
                    BullpenRole.HIGH_LEVERAGE,
                ),
                9,
                1,
                False,
                BullpenRole.MIDDLE,
            ),
        )

        for forecast, inning, margin, bridge_needed, expected_role in cases:
            with self.subTest(
                inning=inning,
                bridge_needed=bridge_needed,
                expected_role=expected_role,
            ):
                selected = select_bullpen_segment(
                    forecast,
                    inning=inning,
                    margin=margin,
                    bridge_needed=bridge_needed,
                    verified_opener_bulk=False,
                )
                self.assertEqual(selected.role, expected_role)

    def test_late_fallback_exhaustion_never_synthesizes_or_forces_protected_arms(self):
        lower_unavailable = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.BRIDGE: {
                        "available": _bullpen_boolean(False),
                    },
                    BullpenRole.MIDDLE: {
                        "available": _bullpen_boolean(False),
                    },
                }
            ),
            SimulationConfig(),
        )
        close_chain_unavailable = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.MIDDLE: {
                        "available": _bullpen_boolean(False),
                    },
                    BullpenRole.HIGH_LEVERAGE: {
                        "available": _bullpen_boolean(False),
                    },
                    BullpenRole.CLOSER: {
                        "available": _bullpen_boolean(False),
                    },
                }
            ),
            SimulationConfig(),
        )

        with self.assertRaisesRegex(ValueError, "no available team bullpen segment"):
            select_bullpen_segment(
                lower_unavailable,
                inning=8,
                margin=4,
                bridge_needed=False,
                verified_opener_bulk=False,
            )
        with self.assertRaisesRegex(ValueError, "no available team bullpen segment"):
            select_bullpen_segment(
                close_chain_unavailable,
                inning=9,
                margin=1,
                bridge_needed=False,
                verified_opener_bulk=False,
            )

    def test_duplicate_ids_duplicate_roles_and_missing_roles_are_rejected(self):
        baseline = _bullpen_input()
        duplicate_ids = list(baseline.segments)
        duplicate_ids[1] = replace(
            duplicate_ids[1],
            segment_id=duplicate_ids[0].segment_id,
        )
        duplicate_roles = list(baseline.segments)
        duplicate_roles[-1] = replace(
            duplicate_roles[-1],
            role=BullpenRole.HIGH_LEVERAGE,
        )
        invalid_inputs = (
            ("segment_id", _bullpen_input(segments=tuple(duplicate_ids))),
            ("role", _bullpen_input(segments=tuple(duplicate_roles))),
            ("roles", _bullpen_input(segments=baseline.segments[:-1])),
        )

        for message, bullpen in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    forecast_bullpen(bullpen, SimulationConfig())

    def test_public_functions_validate_types_ranges_and_booleans(self):
        config = SimulationConfig()
        forecast = forecast_bullpen(_bullpen_input(), config)

        with self.assertRaisesRegex(TypeError, "bullpen"):
            forecast_bullpen(None, config)
        with self.assertRaisesRegex(TypeError, "config"):
            forecast_bullpen(_bullpen_input(), None)
        with self.assertRaisesRegex(TypeError, "forecast"):
            select_bullpen_segment(
                None,
                inning=6,
                margin=1,
                bridge_needed=False,
                verified_opener_bulk=False,
            )

        for inning in (1.0, True, math.nan, math.inf, "6", None):
            with self.subTest(inning=inning):
                with self.assertRaisesRegex(TypeError, "inning"):
                    select_bullpen_segment(
                        forecast,
                        inning=inning,
                        margin=1,
                        bridge_needed=False,
                        verified_opener_bulk=False,
                    )
        for inning in (0, -1):
            with self.subTest(inning=inning):
                with self.assertRaisesRegex(ValueError, "inning"):
                    select_bullpen_segment(
                        forecast,
                        inning=inning,
                        margin=1,
                        bridge_needed=False,
                        verified_opener_bulk=False,
                    )
        for margin in (1.0, True, math.nan, math.inf, "1", None):
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(TypeError, "margin"):
                    select_bullpen_segment(
                        forecast,
                        inning=6,
                        margin=margin,
                        bridge_needed=False,
                        verified_opener_bulk=False,
                    )
        for field_name in ("bridge_needed", "verified_opener_bulk"):
            for invalid in (0, 1, None, "false"):
                with self.subTest(field_name=field_name, invalid=invalid):
                    arguments = {
                        "forecast": forecast,
                        "inning": 6,
                        "margin": 1,
                        "bridge_needed": False,
                        "verified_opener_bulk": False,
                    }
                    arguments[field_name] = invalid
                    with self.assertRaisesRegex(TypeError, field_name):
                        select_bullpen_segment(**arguments)

    def test_results_and_role_mapping_are_defensively_immutable(self):
        forecast = forecast_bullpen(_bullpen_input(), SimulationConfig())
        segment = forecast.by_role[BullpenRole.BRIDGE]

        with self.assertRaises(FrozenInstanceError):
            segment.available = False
        with self.assertRaises(FrozenInstanceError):
            forecast.lower_relief_risk = True
        with self.assertRaises(TypeError):
            forecast.by_role[BullpenRole.BRIDGE] = segment

        source_mapping = dict(forecast.by_role)
        direct = replace(forecast, by_role=source_mapping)
        source_mapping[BullpenRole.BRIDGE] = forecast.by_role[BullpenRole.MIDDLE]

        self.assertEqual(
            direct.by_role[BullpenRole.BRIDGE].role,
            BullpenRole.BRIDGE,
        )

    def test_direct_segment_construction_rejects_invalid_or_inconsistent_values(self):
        forecast = forecast_bullpen(_bullpen_input(), SimulationConfig())
        segment = forecast.by_role[BullpenRole.BRIDGE]
        invalid_changes = (
            {"segment_id": ""},
            {"role": "bridge"},
            {"projected_ra9": 0.0},
            {"projected_ra9": math.nan},
            {"projected_ra9": 4.5},
            {"run_prevention_multiplier": 0.0},
            {"run_prevention_multiplier": math.inf},
            {"run_prevention_multiplier": 1.1},
            {"base_log_standard_error": -0.1},
            {"base_log_standard_error": math.inf},
            {"base_log_standard_error": 0.4},
            {"log_standard_error": -0.1},
            {"log_standard_error": math.nan},
            {"available": 1},
            {"limited_performance_or_use_risk": 1},
            {"synthetic": 1},
            {"notes": []},
            {"notes": ("invented",)},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(segment, **changes)

        synthetic = forecast.unscheduled_f5_relief
        for changes in (
            {"segment_id": "bullpen:fake-prior"},
            {"synthetic": False},
            {"run_prevention_multiplier": 1.1},
        ):
            with self.subTest(synthetic_changes=changes):
                with self.assertRaises(ValueError):
                    replace(synthetic, **changes)

    def test_direct_forecast_construction_rejects_relationship_inconsistency(self):
        forecast = forecast_bullpen(_bullpen_input(), SimulationConfig())
        bridge = forecast.by_role[BullpenRole.BRIDGE]
        middle = forecast.by_role[BullpenRole.MIDDLE]
        duplicate_role = replace(middle, role=BullpenRole.BRIDGE)
        duplicate_id = replace(middle, segment_id=bridge.segment_id)
        duplicate_role_segments = list(forecast.segments)
        duplicate_role_segments[1] = duplicate_role
        duplicate_id_segments = list(forecast.segments)
        duplicate_id_segments[1] = duplicate_id
        wrong_mapping = dict(forecast.by_role)
        wrong_mapping[BullpenRole.BRIDGE] = middle
        invalid_changes = (
            {"team": ""},
            {"segments": list(forecast.segments)},
            {"segments": forecast.segments[:-1]},
            {"segments": tuple(duplicate_role_segments)},
            {"segments": tuple(duplicate_id_segments)},
            {"by_role": {}},
            {"by_role": wrong_mapping},
            {
                "high_leverage_available": (
                    not forecast.high_leverage_available
                )
            },
            {"lower_relief_risk": not forecast.lower_relief_risk},
            {"unscheduled_f5_relief": bridge},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(forecast, **changes)

        for value in (bridge, forecast):
            with self.subTest(result_type=type(value).__name__):
                public_values = {
                    result_field.name: getattr(value, result_field.name)
                    for result_field in fields(value)
                    if not result_field.name.startswith("_")
                }
                with self.assertRaisesRegex(ValueError, "forecast_bullpen"):
                    type(value)(**public_values)

    def test_canonical_serialization_is_deterministic_and_input_order_independent(self):
        baseline = _bullpen_input()
        first = forecast_bullpen(baseline, SimulationConfig())
        second = forecast_bullpen(
            BullpenInput(
                team=baseline.team,
                segments=tuple(reversed(baseline.segments)),
            ),
            SimulationConfig(),
        )

        first_json = canonical_json(first)
        payload = json.loads(first_json)

        self.assertEqual(first, second)
        self.assertEqual(first_json, canonical_json(second))
        self.assertEqual(
            [segment["role"] for segment in payload["segments"]],
            [role.value for role in _BULLPEN_ROLE_ORDER],
        )
        self.assertEqual(
            set(payload["by_role"]),
            {role.value for role in _BULLPEN_ROLE_ORDER},
        )


if __name__ == "__main__":
    unittest.main()
