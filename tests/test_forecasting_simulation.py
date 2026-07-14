from collections import Counter
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import math
import unittest
from unittest.mock import patch

import analysis.forecasting.simulation as simulation_module
from analysis.forecasting.bullpen import forecast_bullpen
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import BullpenRole
from analysis.forecasting.offense import forecast_offense
from analysis.forecasting.simulation import (
    ComponentDistribution,
    DistributionSummary,
    GameSimulationInput,
    SimulationResult,
    mean_runs_for_half_inning,
    simulate_game_distribution,
)
from analysis.forecasting.starter import (
    PitcherChainForecast,
    PitcherForecast,
    WorkloadDistribution,
)
from tests.test_forecasting_components import (
    _bullpen_boolean,
    _bullpen_input,
    _bullpen_numeric,
    _offense_lineup,
    _offense_profile,
    _offense_response,
    _offense_usage,
    _offense_with_responses,
)


def _fast_config(**overrides):
    return replace(SimulationConfig(simulations=500), **overrides)


def _fixed_pitcher(pitcher_id, role, outs, run_prevention_multiplier=1.0):
    bucket = 0 if outs < 15 else 1 if outs < 18 else 2
    probabilities = tuple(1.0 if index == bucket else 0.0 for index in range(3))
    supports = [(12,), (15,), (18,)]
    supports[bucket] = (outs,)
    workload = WorkloadDistribution(
        probabilities=probabilities,
        supports=tuple(supports),
        source="pitcher_history",
    )
    pitcher = PitcherForecast(
        pitcher_id=pitcher_id,
        role=role,
        workload=workload,
        run_prevention_multiplier=run_prevention_multiplier,
        log_standard_error=0.05,
        command_risk=False,
        home_run_risk=False,
    )
    return pitcher


def _fixed_starter_chain(pitcher_id, outs=13):
    pitcher = _fixed_pitcher(pitcher_id, "starter", outs)
    workload = pitcher.workload
    return PitcherChainForecast(
        pitchers=(pitcher,),
        combined_workload=workload,
        log_standard_error=0.05,
    )


def _offense_forecast_for_ids(opposing_pitcher_ids, config):
    lineup = _offense_lineup(log_standard_error=0.05)
    lineup = _offense_with_responses(
        lineup,
        {
            slot.order: tuple(
                _offense_response(
                    slot.order,
                    pitcher_id,
                    expected_side=slot.expected_side,
                )
                for pitcher_id in opposing_pitcher_ids
            )
            for slot in lineup.slots
        },
    )
    return forecast_offense(
        lineup,
        tuple(_offense_profile(pitcher_id) for pitcher_id in opposing_pitcher_ids),
        config,
    )


def _unverified_opener_bulk_chain():
    opener = _fixed_pitcher("pitcher:home-opener", "opener", 3)
    bulk = _fixed_pitcher(
        "pitcher:home-bulk",
        "bulk",
        12,
        run_prevention_multiplier=1.75,
    )
    combined_workload = WorkloadDistribution(
        probabilities=(0.0, 1.0, 0.0),
        supports=((12,), (15,), (18,)),
        source="combined_chain_history",
    )
    return PitcherChainForecast(
        pitchers=(opener, bulk),
        combined_workload=combined_workload,
        log_standard_error=0.05,
        diagnostics=("handoff_unverified",),
    )


def _game_sim_input(
    config,
    *,
    away_chain=None,
    home_chain=None,
    away_bullpen=None,
    home_bullpen=None,
):
    away_chain = away_chain or _fixed_starter_chain("pitcher:away")
    home_chain = home_chain or _fixed_starter_chain("pitcher:home")
    away_bullpen = away_bullpen or forecast_bullpen(_bullpen_input(), config)
    home_bullpen = home_bullpen or forecast_bullpen(_bullpen_input(), config)
    return GameSimulationInput(
        away_offense=_offense_forecast_for_ids(
            tuple(pitcher.pitcher_id for pitcher in home_chain.pitchers),
            config,
        ),
        home_offense=_offense_forecast_for_ids(
            tuple(pitcher.pitcher_id for pitcher in away_chain.pitchers),
            config,
        ),
        away_pitcher_chain=away_chain,
        home_pitcher_chain=home_chain,
        away_bullpen=away_bullpen,
        home_bullpen=home_bullpen,
    )


def _point_latent(_rng, point, _sigma, _lower, _upper):
    return point


def _simulate_scripted_game(runs):
    config = SimulationConfig(simulations=1)
    scripted_runs = iter(runs)
    with (
        patch.object(
            simulation_module,
            "draw_bounded_lognormal",
            side_effect=_point_latent,
        ),
        patch.object(
            simulation_module,
            "_draw_run_count",
            side_effect=lambda *_args, **_kwargs: next(scripted_runs),
        ) as run_draw,
        patch.object(
            simulation_module,
            "_draw_diagnostic_counts",
            return_value=(0, 0),
        ),
    ):
        result = simulate_game_distribution(
            _game_sim_input(config),
            "a" * 64,
            config,
        )
    return result.games[0], run_draw.call_count


class GameSimulationTests(unittest.TestCase):
    def test_forecast_component_domain_validator_is_public_and_strict(self):
        validator = getattr(
            simulation_module,
            "validate_component_distribution_domain",
            None,
        )
        self.assertTrue(callable(validator), "component domain validator missing")
        valid = ComponentDistribution(
            component_id="away_f5_runs",
            values=(0, 1, 2),
            summary=simulation_module._summarize((0, 1, 2)),
        )
        invalid = (
            ComponentDistribution(
                component_id="away_f5_runs",
                values=(-1,),
                summary=simulation_module._summarize((-1,)),
            ),
            ComponentDistribution(
                component_id="away_full_runs",
                values=(1.5,),
                summary=simulation_module._summarize((1.5,)),
            ),
            ComponentDistribution(
                component_id="home_protected_bullpen_close_entry",
                values=(2,),
                summary=simulation_module._summarize((2,)),
            ),
            ComponentDistribution(
                component_id="home_bullpen_segment_bridge_exposed",
                values=(2,),
                summary=simulation_module._summarize((2,)),
            ),
        )

        self.assertIsNone(validator(valid))
        for distribution in invalid:
            with self.subTest(component_id=distribution.component_id):
                with self.assertRaisesRegex(ValueError, "domain|binary|count"):
                    validator(distribution)

    def test_same_run_id_reproduces_exact_scores(self):
        config = _fast_config()
        simulation_input = _game_sim_input(config)

        first = simulate_game_distribution(simulation_input, "a" * 64, config)
        second = simulate_game_distribution(simulation_input, "a" * 64, config)

        self.assertEqual(first.games, second.games)

    def test_different_run_id_changes_seeded_paths(self):
        config = _fast_config()
        simulation_input = _game_sim_input(config)

        first = simulate_game_distribution(simulation_input, "a" * 64, config)
        second = simulate_game_distribution(simulation_input, "b" * 64, config)

        self.assertNotEqual(first.games, second.games)

    def test_batch_rngs_are_seeded_once_from_the_forecast_id_prefix(self):
        config = SimulationConfig(simulations=3)
        seed = int(("a" * 64)[:16], 16)
        constructed_seeds = []
        random_factory = simulation_module.random.Random

        def tracked_random_factory(value):
            constructed_seeds.append(value)
            return random_factory(value)

        with patch.object(
            simulation_module.random,
            "Random",
            side_effect=tracked_random_factory,
        ):
            simulate_game_distribution(
                _game_sim_input(config),
                "a" * 64,
                config,
            )

        self.assertEqual(constructed_seeds, [seed, seed])

    def test_larger_batch_preserves_the_exact_simulation_prefix(self):
        short_config = SimulationConfig(simulations=7)
        long_config = replace(short_config, simulations=11)
        simulation_input = _game_sim_input(short_config)

        short = simulate_game_distribution(
            simulation_input,
            "b" * 64,
            short_config,
        )
        long = simulate_game_distribution(
            simulation_input,
            "b" * 64,
            long_config,
        )

        self.assertEqual(short.games, long.games[: short_config.simulations])

    def test_home_lead_skips_bottom_ninth(self):
        game, run_draws = _simulate_scripted_game([0, 1] + [0] * 15)

        self.assertFalse(game.bottom_ninth_played)
        self.assertEqual(run_draws, 17)

    def test_walkoff_is_capped_at_runs_needed_to_win(self):
        game, run_draws = _simulate_scripted_game([0] * 18 + [1, 5])

        self.assertEqual(game.final_inning, 10)
        self.assertEqual(game.home_full_runs - game.away_full_runs, 1)
        self.assertEqual(run_draws, 20)

    def test_played_bottom_ninth_home_win_must_have_one_run_margin(self):
        game, _ = _simulate_scripted_game([0] * 18 + [1, 5])

        with self.assertRaisesRegex(ValueError, "walkoff"):
            replace(
                game,
                home_full_runs=game.home_full_runs + 2,
                away_bullpen_runs_allowed=(
                    game.away_bullpen_runs_allowed + 2
                ),
            )

    def test_skipped_bottom_ninth_home_win_may_have_larger_margin(self):
        game, _ = _simulate_scripted_game([0, 3] + [0] * 15)

        self.assertFalse(game.bottom_ninth_played)
        self.assertEqual(game.home_full_runs - game.away_full_runs, 3)

    def test_starter_and_bullpen_are_not_added_as_separate_run_totals(self):
        source = inspect.getsource(mean_runs_for_half_inning)
        run_draw_source = inspect.getsource(simulation_module._draw_run_count)

        self.assertNotIn("projected_runs", source)
        self.assertIn("run_prevention_multiplier", source)
        self.assertEqual(run_draw_source.count("draw_gamma_poisson"), 1)
        self.assertAlmostEqual(
            mean_runs_for_half_inning(
                league_inning_rate=0.5,
                offense_multiplier=1.2,
                matchup_multiplier=0.9,
                run_prevention_multiplier=1.1,
            ),
            0.5 * 1.2 * 0.9 * 1.1,
        )

    def test_simulation_emits_reviewable_component_distributions(self):
        config = _fast_config()
        result = simulate_game_distribution(
            _game_sim_input(config),
            "a" * 64,
            config,
        )
        required = {
            "away_f5_runs",
            "home_f5_runs",
            "away_full_runs",
            "home_full_runs",
            "away_full_traffic",
            "home_full_traffic",
            "away_full_extra_base_hits",
            "home_full_extra_base_hits",
            "away_pitcher_chain_runs_allowed",
            "home_pitcher_chain_runs_allowed",
            "away_bullpen_runs_allowed",
            "home_bullpen_runs_allowed",
            "away_pitcher_chain_workload_outs",
            "home_pitcher_chain_workload_outs",
            "away_protected_bullpen_close_entry",
            "home_protected_bullpen_close_entry",
        }

        self.assertTrue(required.issubset(result.component_distributions))
        self.assertEqual(len(result.games), 500)
        self.assertIsNotNone(
            result.component_distributions[
                "away_pitcher_chain_workload_outs"
            ].summary.p80
        )

    def test_unverified_opener_bulk_handoff_preserves_declared_bulk_in_f5(self):
        config = SimulationConfig(simulations=1)
        home_chain = _unverified_opener_bulk_chain()
        simulation_input = _game_sim_input(config, home_chain=home_chain)
        bulk = home_chain.pitchers[1]
        means = []
        scripted_runs = iter([0, 0, 1] + [0] * 15)

        def record_run_mean(_rng, *, mean, shape):
            del shape
            means.append(mean)
            return next(scripted_runs)

        with (
            patch.object(
                simulation_module,
                "draw_bounded_lognormal",
                side_effect=_point_latent,
            ),
            patch.object(
                simulation_module,
                "_draw_run_count",
                side_effect=record_run_mean,
            ),
            patch.object(
                simulation_module,
                "_draw_diagnostic_counts",
                return_value=(0, 0),
            ),
        ):
            result = simulate_game_distribution(
                simulation_input,
                "a" * 64,
                config,
            )

        self.assertAlmostEqual(
            means[2],
            mean_runs_for_half_inning(
                league_inning_rate=config.league_runs_per_team_game / 9,
                offense_multiplier=(
                    simulation_input.away_offense.base_offense_multiplier
                ),
                matchup_multiplier=(
                    simulation_input.away_offense.multiplier_for(bulk.pitcher_id)
                ),
                run_prevention_multiplier=bulk.run_prevention_multiplier,
            ),
        )
        self.assertEqual(result.games[0].home_pitcher_chain_runs_allowed, 1)
        self.assertEqual(result.games[0].home_bullpen_runs_allowed, 0)

    def test_mixed_exit_inning_conserves_every_realized_run(self):
        runs = [0] * 18
        runs[1] = 1
        runs[8] = 30

        game, run_draws = _simulate_scripted_game(runs)

        self.assertEqual(run_draws, 18)
        self.assertEqual(game.away_full_runs, 30)
        self.assertEqual(
            game.home_pitcher_chain_runs_allowed
            + game.home_bullpen_runs_allowed,
            game.away_full_runs,
        )
        self.assertGreater(game.home_pitcher_chain_runs_allowed, 0)
        self.assertGreater(game.home_bullpen_runs_allowed, 0)
        mixed_mean = mean_runs_for_half_inning(
            league_inning_rate=0.5,
            offense_multiplier=1.0,
            matchup_multiplier=0.8,
            run_prevention_multiplier=0.6,
            out_fraction=1 / 3,
        ) + mean_runs_for_half_inning(
            league_inning_rate=0.5,
            offense_multiplier=1.0,
            matchup_multiplier=1.2,
            run_prevention_multiplier=1.9,
            out_fraction=2 / 3,
        )
        self.assertAlmostEqual(
            mixed_mean,
            0.5 * ((1 / 3) * 0.8 * 0.6 + (2 / 3) * 1.2 * 1.9),
        )

    def test_binomial_quantile_is_stable_for_large_symmetric_trials(self):
        result = simulation_module._binomial_quantile(2000, 0.5, 0.5)

        self.assertLessEqual(abs(result - 1000), 1)

    def test_binomial_quantile_handles_extreme_probabilities_and_samples(self):
        self.assertEqual(
            simulation_module._binomial_quantile(2000, 1e-9, 0.5),
            0,
        )
        self.assertEqual(
            simulation_module._binomial_quantile(2000, 1 - 1e-9, 0.5),
            2000,
        )
        self.assertEqual(
            simulation_module._binomial_quantile(2000, 0.5, 0.0),
            0,
        )
        self.assertEqual(
            simulation_module._binomial_quantile(2000, 0.5, 1.0),
            2000,
        )

    def test_f5_paths_are_exactly_isolated_from_team_bullpen_quality(self):
        config = _fast_config()
        roles = (
            BullpenRole.BRIDGE,
            BullpenRole.MIDDLE,
            BullpenRole.HIGH_LEVERAGE,
            BullpenRole.CLOSER,
        )

        def bullpen_with_ra9(projected_ra9):
            return forecast_bullpen(
                _bullpen_input(
                    {
                        role: {
                            "projected_ra9": _bullpen_numeric(projected_ra9)
                        }
                        for role in roles
                    }
                ),
                config,
            )

        strong = simulate_game_distribution(
            _game_sim_input(
                config,
                home_bullpen=bullpen_with_ra9(0.5),
            ),
            "c" * 64,
            config,
        )
        weak = simulate_game_distribution(
            _game_sim_input(
                config,
                home_bullpen=bullpen_with_ra9(40.0),
            ),
            "c" * 64,
            config,
        )

        for component_id in (
            "away_f5_runs",
            "home_f5_runs",
            "away_f5_traffic",
            "home_f5_traffic",
            "away_f5_extra_base_hits",
            "home_f5_extra_base_hits",
        ):
            with self.subTest(component_id=component_id):
                self.assertEqual(
                    strong.component_distributions[component_id].values,
                    weak.component_distributions[component_id].values,
                )
        self.assertNotEqual(
            strong.component_distributions["away_full_runs"].values,
            weak.component_distributions["away_full_runs"].values,
        )

    def test_f5_score_paths_ignore_late_bullpen_availability_and_quality(self):
        config = _fast_config()
        baseline_bullpen = forecast_bullpen(_bullpen_input(), config)
        changed_bullpen = forecast_bullpen(
            _bullpen_input(
                {
                    BullpenRole.HIGH_LEVERAGE: {
                        "available": _bullpen_boolean(False),
                    },
                    BullpenRole.CLOSER: {
                        "projected_ra9": _bullpen_numeric(40.0),
                    },
                }
            ),
            config,
        )

        baseline = simulate_game_distribution(
            _game_sim_input(config, home_bullpen=baseline_bullpen),
            "4" * 64,
            config,
        )
        changed = simulate_game_distribution(
            _game_sim_input(config, home_bullpen=changed_bullpen),
            "4" * 64,
            config,
        )

        def score_path(game, period):
            return (
                getattr(game, f"away_{period}_runs"),
                getattr(game, f"home_{period}_runs"),
            )

        baseline_f5 = tuple(score_path(game, "f5") for game in baseline.games)
        changed_f5 = tuple(score_path(game, "f5") for game in changed.games)
        self.assertEqual(canonical_json(baseline_f5), canonical_json(changed_f5))
        self.assertNotEqual(
            tuple(score_path(game, "full") for game in baseline.games),
            tuple(score_path(game, "full") for game in changed.games),
        )

    def test_diagnostic_projections_never_change_run_paths(self):
        config = _fast_config(simulations=100)
        baseline_input = _game_sim_input(config)
        changed_input = replace(
            baseline_input,
            away_offense=replace(
                baseline_input.away_offense,
                projected_traffic_per_9=100.0,
                projected_extra_base_hits_per_9=80.0,
            ),
            home_offense=replace(
                baseline_input.home_offense,
                projected_traffic_per_9=0.0,
                projected_extra_base_hits_per_9=0.0,
            ),
        )

        baseline = simulate_game_distribution(
            baseline_input,
            "d" * 64,
            config,
        )
        changed = simulate_game_distribution(
            changed_input,
            "d" * 64,
            config,
        )

        def run_path(game):
            return (
                game.away_f5_runs,
                game.home_f5_runs,
                game.away_full_runs,
                game.home_full_runs,
                game.away_pitcher_chain_runs_allowed,
                game.home_pitcher_chain_runs_allowed,
                game.away_bullpen_runs_allowed,
                game.home_bullpen_runs_allowed,
                game.away_pitcher_chain_workload_outs,
                game.home_pitcher_chain_workload_outs,
                game.away_bullpen_segment_ids_exposed,
                game.home_bullpen_segment_ids_exposed,
            )

        self.assertEqual(
            tuple(map(run_path, baseline.games)),
            tuple(map(run_path, changed.games)),
        )

    def test_diagnostic_uncertainty_uses_raw_lineup_sigma(self):
        config = SimulationConfig(simulations=1)
        baseline_input = _game_sim_input(config)
        pitcher_id = baseline_input.home_pitcher_chain.pitchers[0].pitcher_id
        lineup = _offense_lineup(log_standard_error=0.09)
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
        missing_sample_offense = forecast_offense(
            _offense_with_responses(
                lineup,
                {
                    slot.order: (
                        _offense_response(
                            slot.order,
                            pitcher_id,
                            expected_side=slot.expected_side,
                            pitch_type="FF",
                        ),
                    )
                    for slot in lineup.slots
                },
            ),
            (profile,),
            config,
        )
        expected_matchup_sigma = math.hypot(
            0.09,
            0.75 * config.sample_shortage_log_se_penalty,
        )
        simulation_input = replace(
            baseline_input,
            away_offense=missing_sample_offense,
        )
        random_factory = simulation_module.random.Random
        random_instances = []
        latent_calls = []

        def tracked_random_factory(seed):
            instance = random_factory(seed)
            random_instances.append(instance)
            return instance

        def tracked_latent(rng, point, sigma, lower, upper):
            latent_calls.append((rng, point, sigma, lower, upper))
            return point

        with (
            patch.object(
                simulation_module.random,
                "Random",
                side_effect=tracked_random_factory,
            ),
            patch.object(
                simulation_module,
                "draw_bounded_lognormal",
                side_effect=tracked_latent,
            ),
        ):
            simulate_game_distribution(simulation_input, "6" * 64, config)

        score_rng, diagnostic_rng = random_instances
        diagnostic_sigmas = tuple(
            sigma for rng, _point, sigma, _lower, _upper in latent_calls
            if rng is diagnostic_rng
        )
        score_sigmas = tuple(
            sigma for rng, _point, sigma, _lower, _upper in latent_calls
            if rng is score_rng
        )
        self.assertEqual(diagnostic_sigmas[0], 0.09)
        self.assertIn(expected_matchup_sigma, score_sigmas)

    def test_component_distributions_are_deterministic_nearest_rank_values(self):
        config = _fast_config()
        simulation_input = _game_sim_input(config)
        first = simulate_game_distribution(simulation_input, "e" * 64, config)
        second = simulate_game_distribution(simulation_input, "e" * 64, config)
        expected_ids = {
            f"{side}_{suffix}"
            for side in ("away", "home")
            for suffix in (
                "f5_runs",
                "full_runs",
                "f5_traffic",
                "full_traffic",
                "f5_extra_base_hits",
                "full_extra_base_hits",
                "pitcher_chain_runs_allowed",
                "bullpen_runs_allowed",
                "pitcher_chain_workload_outs",
                "protected_bullpen_close_entry",
            )
        }
        for side in ("away", "home"):
            bullpen = getattr(simulation_input, f"{side}_bullpen")
            expected_ids.update(
                f"{side}_bullpen_segment_{segment.segment_id}_exposed"
                for segment in bullpen.available_segments
            )

        self.assertEqual(first.component_distributions, second.component_distributions)
        self.assertEqual(set(first.component_distributions), expected_ids)
        for component_id, distribution in first.component_distributions.items():
            with self.subTest(component_id=component_id):
                ordered = sorted(distribution.values)
                expected = tuple(
                    ordered[math.ceil(probability * len(ordered)) - 1]
                    for probability in (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)
                )
                self.assertEqual(
                    (
                        distribution.summary.p05,
                        distribution.summary.p10,
                        distribution.summary.p20,
                        distribution.summary.median,
                        distribution.summary.p80,
                        distribution.summary.p90,
                        distribution.summary.p95,
                    ),
                    expected,
                )

    def test_results_are_frozen_defensive_and_canonically_serializable(self):
        config = _fast_config(simulations=10)
        simulation_input = _game_sim_input(config)
        result = simulate_game_distribution(
            simulation_input,
            "f" * 64,
            config,
        )
        component = result.component_distributions["away_full_runs"]
        payload = json.loads(canonical_json(result))

        self.assertEqual(payload["simulation_count"], 10)
        self.assertEqual(len(payload["games"]), 10)
        self.assertTrue(
            all(
                component_id.startswith(("away_", "home_"))
                for component_id in result.component_distributions
            )
        )
        with self.assertRaises(FrozenInstanceError):
            simulation_input.away_offense = simulation_input.home_offense
        with self.assertRaises(FrozenInstanceError):
            result.simulation_count = 1
        with self.assertRaises(FrozenInstanceError):
            result.games[0].away_full_runs = 99
        with self.assertRaises(FrozenInstanceError):
            component.summary.p95 = 99
        with self.assertRaises(TypeError):
            result.component_distributions["new"] = component

        inconsistent = DistributionSummary(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "summary"):
            ComponentDistribution(
                component_id="away_test",
                values=(1, 2, 3),
                summary=inconsistent,
            )

    def test_simulation_result_cannot_be_constructed_outside_the_factory(self):
        config = SimulationConfig(simulations=1)
        valid = simulate_game_distribution(
            _game_sim_input(config),
            "7" * 64,
            config,
        )

        with self.assertRaisesRegex(ValueError, "simulation factory"):
            SimulationResult(
                forecast_run_id=valid.forecast_run_id,
                simulation_count=valid.simulation_count,
                games=valid.games,
                component_distributions=valid.component_distributions,
            )

    def test_module_exposes_no_result_context_issuer(self):
        forbidden_names = {
            "_build_simulation_result",
            "_make_simulation_result_factory",
            "_build_simulate_game_distribution",
            "_create_simulation_result_context",
        }

        self.assertEqual(
            forbidden_names.intersection(vars(simulation_module)),
            set(),
        )
        self.assertTrue(
            callable(simulation_module._require_simulation_result_context)
        )

    def test_result_rejects_noncanonical_or_nonderived_components(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        valid = simulate_game_distribution(
            simulation_input,
            "8" * 64,
            config,
        )
        baseline = dict(valid.component_distributions)
        exposure_id = next(
            component_id
            for component_id in baseline
            if "_bullpen_segment_" in component_id
        )

        missing_base = dict(baseline)
        missing_base.pop("away_f5_runs")
        missing_exposure = dict(baseline)
        missing_exposure.pop(exposure_id)
        fabricated = dict(baseline)
        fabricated_component = replace(
            baseline["away_full_runs"],
            component_id="away_fabricated",
        )
        fabricated[fabricated_component.component_id] = fabricated_component
        wrong_values = dict(baseline)
        full_runs = baseline["away_full_runs"].values[0] + 1
        wrong_values["away_full_runs"] = ComponentDistribution(
            component_id="away_full_runs",
            values=(full_runs,),
            summary=simulation_module._summarize((full_runs,)),
        )
        invalid_binary = dict(baseline)
        invalid_binary["away_protected_bullpen_close_entry"] = (
            ComponentDistribution(
                component_id="away_protected_bullpen_close_entry",
                values=(2,),
                summary=simulation_module._summarize((2,)),
            )
        )
        wrong_exposure = dict(baseline)
        exposure_value = 1 - baseline[exposure_id].values[0]
        wrong_exposure[exposure_id] = ComponentDistribution(
            component_id=exposure_id,
            values=(exposure_value,),
            summary=simulation_module._summarize((exposure_value,)),
        )
        wrong_summary = dict(baseline)
        altered_summary = replace(baseline["home_full_runs"])
        object.__setattr__(
            altered_summary,
            "summary",
            DistributionSummary(99, 99, 99, 99, 99, 99, 99),
        )
        wrong_summary["home_full_runs"] = altered_summary

        cases = {
            "missing_base": missing_base,
            "missing_exposure": missing_exposure,
            "fabricated": fabricated,
            "wrong_values": wrong_values,
            "invalid_binary": invalid_binary,
            "wrong_exposure": wrong_exposure,
            "wrong_summary": wrong_summary,
        }
        for label, components in cases.items():
            with self.subTest(label=label):
                with patch.object(
                    simulation_module,
                    "_component_distributions",
                    return_value=components,
                ):
                    with self.assertRaisesRegex(ValueError, "component"):
                        simulate_game_distribution(
                            simulation_input,
                            "8" * 64,
                            config,
                        )

    def test_full_game_extra_base_hits_cannot_fall_below_f5(self):
        game, _ = _simulate_scripted_game([0, 1] + [0] * 15)

        with self.assertRaisesRegex(ValueError, "full-game extra-base hits"):
            replace(
                game,
                away_f5_traffic=1,
                away_full_traffic=1,
                away_f5_extra_base_hits=1,
                away_full_extra_base_hits=0,
            )

    def test_exposed_bullpen_ids_must_be_canonical_sorted_unique(self):
        game, _ = _simulate_scripted_game([0, 1] + [0] * 15)

        with self.assertRaisesRegex(ValueError, "canonical sorted"):
            replace(
                game,
                away_bullpen_segment_ids_exposed=("bullpen:z", "bullpen:a"),
            )

    def test_game_state_flags_must_match_regulation_and_extra_innings(self):
        home_lead, _ = _simulate_scripted_game([0, 1] + [0] * 15)
        away_lead, _ = _simulate_scripted_game([1, 0] + [0] * 16)
        extra_game, _ = _simulate_scripted_game([0] * 18 + [1, 0])
        cases = (
            (home_lead, {"final_inning": 10}),
            (away_lead, {"bottom_ninth_played": False}),
            (extra_game, {"bottom_ninth_played": False}),
            (home_lead, {"max_inning_tiebreak_used": True}),
        )

        for game, changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "game state"):
                    replace(game, **changes)

    def test_result_rejects_exposure_ids_not_available_in_context(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        valid = simulate_game_distribution(
            simulation_input,
            "9" * 64,
            config,
        )
        invalid_game = replace(
            valid.games[0],
            away_bullpen_segment_ids_exposed=("bullpen:not-available",),
        )

        with patch.object(
            simulation_module,
            "_simulate_one_game",
            return_value=invalid_game,
        ):
            with self.assertRaisesRegex(ValueError, "available bullpen"):
                simulate_game_distribution(
                    simulation_input,
                    "9" * 64,
                    config,
                )

    def test_protected_entry_requires_exposed_high_leverage_or_closer(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        valid = simulate_game_distribution(
            simulation_input,
            "b" * 64,
            config,
        )
        bridge_id = next(
            segment.segment_id
            for segment in simulation_input.away_bullpen.available_segments
            if segment.role == BullpenRole.BRIDGE
        )
        invalid_game = replace(
            valid.games[0],
            away_bullpen_segment_ids_exposed=(bridge_id,),
            away_protected_bullpen_close_entry=True,
        )

        with patch.object(
            simulation_module,
            "_simulate_one_game",
            return_value=invalid_game,
        ):
            with self.assertRaisesRegex(ValueError, "protected bullpen"):
                simulate_game_distribution(
                    simulation_input,
                    "b" * 64,
                    config,
                )

    def test_unprotected_entry_may_expose_a_closer(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        valid = simulate_game_distribution(
            simulation_input,
            "c" * 64,
            config,
        )
        closer_id = next(
            segment.segment_id
            for segment in simulation_input.away_bullpen.available_segments
            if segment.role == BullpenRole.CLOSER
        )
        allowed_game = replace(
            valid.games[0],
            away_bullpen_segment_ids_exposed=(closer_id,),
            away_protected_bullpen_close_entry=False,
        )

        with patch.object(
            simulation_module,
            "_simulate_one_game",
            return_value=allowed_game,
        ):
            result = simulate_game_distribution(
                simulation_input,
                "c" * 64,
                config,
            )

        self.assertFalse(result.games[0].away_protected_bullpen_close_entry)
        self.assertEqual(
            result.games[0].away_bullpen_segment_ids_exposed,
            (closer_id,),
        )

    def test_synthetic_f5_relief_is_not_an_exposed_team_segment(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        game = simulation_module._simulate_one_game(
            simulation_input,
            simulation_module.random.Random(1),
            simulation_module.random.Random(1),
            config,
        )

        self.assertNotIn(
            simulation_input.away_bullpen.unscheduled_f5_relief.segment_id,
            game.away_bullpen_segment_ids_exposed,
        )
        self.assertNotIn(
            simulation_input.home_bullpen.unscheduled_f5_relief.segment_id,
            game.home_bullpen_segment_ids_exposed,
        )

    def test_result_binds_max_tiebreak_to_configured_max_inning(self):
        config = SimulationConfig(simulations=1, max_innings=30)
        extra_game, _ = _simulate_scripted_game([0] * 18 + [1, 0])
        invalid_game = replace(extra_game, max_inning_tiebreak_used=True)

        with patch.object(
            simulation_module,
            "_simulate_one_game",
            return_value=invalid_game,
        ):
            with self.assertRaisesRegex(ValueError, "max_innings"):
                simulate_game_distribution(
                    _game_sim_input(config),
                    "a" * 64,
                    config,
                )

    def test_invalid_ids_inputs_config_and_numeric_values_are_rejected(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        for invalid in ("", "a" * 63, "g" * 64, "A" * 64):
            with self.subTest(forecast_run_id=invalid):
                with self.assertRaises(ValueError):
                    simulate_game_distribution(simulation_input, invalid, config)
        with self.assertRaises(TypeError):
            simulate_game_distribution(simulation_input, 1, config)
        with self.assertRaises(TypeError):
            simulate_game_distribution(object(), "a" * 64, config)
        with self.assertRaises(TypeError):
            simulate_game_distribution(simulation_input, "a" * 64, object())

        malformed_config = SimulationConfig(simulations=1)
        object.__setattr__(malformed_config, "max_innings", 9)
        with self.assertRaisesRegex(ValueError, "config"):
            simulate_game_distribution(
                _game_sim_input(malformed_config),
                "a" * 64,
                malformed_config,
            )

        malformed_input = _game_sim_input(config)
        object.__setattr__(
            malformed_input.away_offense,
            "base_offense_multiplier",
            math.nan,
        )
        with self.assertRaises(ValueError):
            simulate_game_distribution(malformed_input, "a" * 64, config)
        with self.assertRaises(ValueError):
            mean_runs_for_half_inning(
                league_inning_rate=0.5,
                offense_multiplier=1.0,
                matchup_multiplier=1.0,
                run_prevention_multiplier=1.0,
                out_fraction=math.nan,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            mean_runs_for_half_inning(
                league_inning_rate=1e308,
                offense_multiplier=1e308,
                matchup_multiplier=1.0,
                run_prevention_multiplier=1.0,
            )

    def test_extra_base_hits_never_exceed_traffic(self):
        config = _fast_config()
        result = simulate_game_distribution(
            _game_sim_input(config),
            "1" * 64,
            config,
        )

        for game in result.games:
            for side in ("away", "home"):
                with self.subTest(side=side):
                    self.assertLessEqual(
                        getattr(game, f"{side}_f5_extra_base_hits"),
                        getattr(game, f"{side}_f5_traffic"),
                    )
                    self.assertLessEqual(
                        getattr(game, f"{side}_full_extra_base_hits"),
                        getattr(game, f"{side}_full_traffic"),
                    )

    def test_max_inning_tiebreak_awards_exactly_one_seeded_run(self):
        game, run_draws = _simulate_scripted_game([0] * 60)

        self.assertEqual(run_draws, 60)
        self.assertEqual(game.final_inning, 30)
        self.assertTrue(game.max_inning_tiebreak_used)
        self.assertEqual(game.away_full_runs + game.home_full_runs, 1)
        self.assertEqual(abs(game.away_full_runs - game.home_full_runs), 1)
        self.assertEqual(
            game.home_pitcher_chain_runs_allowed
            + game.home_bullpen_runs_allowed,
            game.away_full_runs,
        )
        self.assertEqual(
            game.away_pitcher_chain_runs_allowed
            + game.away_bullpen_runs_allowed,
            game.home_full_runs,
        )

    def test_extra_innings_apply_exact_multiplier_and_retain_runs(self):
        config = SimulationConfig(simulations=1)
        means = []
        scripted_runs = iter([0] * 18 + [1, 0])

        def record_run_mean(_rng, *, mean, shape):
            del shape
            means.append(mean)
            return next(scripted_runs)

        with (
            patch.object(
                simulation_module,
                "draw_bounded_lognormal",
                side_effect=_point_latent,
            ),
            patch.object(
                simulation_module,
                "_draw_run_count",
                side_effect=record_run_mean,
            ),
            patch.object(
                simulation_module,
                "_draw_diagnostic_counts",
                return_value=(0, 0),
            ),
        ):
            game = simulate_game_distribution(
                _game_sim_input(config),
                "2" * 64,
                config,
            ).games[0]

        self.assertAlmostEqual(means[18] / means[16], 1.80)
        self.assertEqual((game.away_full_runs, game.home_full_runs), (1, 0))
        self.assertEqual(game.final_inning, 10)

    def test_each_game_draws_latents_and_pitcher_plans_once(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        sampled_chain_ids = []
        original_sample_plan = PitcherChainForecast.sample_plan

        def tracked_sample_plan(chain, rng):
            sampled_chain_ids.append(id(chain))
            return original_sample_plan(chain, rng)

        with (
            patch.object(
                simulation_module,
                "draw_bounded_lognormal",
                side_effect=_point_latent,
            ) as latent_draw,
            patch.object(
                PitcherChainForecast,
                "sample_plan",
                new=tracked_sample_plan,
            ),
        ):
            simulate_game_distribution(
                simulation_input,
                "3" * 64,
                config,
            )

        self.assertEqual(latent_draw.call_count, 28)
        self.assertEqual(
            Counter(sampled_chain_ids),
            Counter(
                {
                    id(simulation_input.away_pitcher_chain): 1,
                    id(simulation_input.home_pitcher_chain): 1,
                }
            ),
        )

    def test_both_pitcher_plans_are_sampled_before_any_latent(self):
        config = SimulationConfig(simulations=1)
        simulation_input = _game_sim_input(config)
        events = []
        original_sample_plan = PitcherChainForecast.sample_plan
        original_draw_offense = simulation_module._draw_offense_latents
        original_draw_pitching = simulation_module._draw_pitching_latents

        def tracked_sample_plan(chain, rng):
            events.append("plan")
            return original_sample_plan(chain, rng)

        def tracked_draw_offense(*args, **kwargs):
            events.append("offense_latents")
            return original_draw_offense(*args, **kwargs)

        def tracked_draw_pitching(*args, **kwargs):
            events.append("pitching_latents")
            return original_draw_pitching(*args, **kwargs)

        with (
            patch.object(
                PitcherChainForecast,
                "sample_plan",
                new=tracked_sample_plan,
            ),
            patch.object(
                simulation_module,
                "_draw_offense_latents",
                side_effect=tracked_draw_offense,
            ),
            patch.object(
                simulation_module,
                "_draw_pitching_latents",
                side_effect=tracked_draw_pitching,
            ),
        ):
            simulate_game_distribution(
                simulation_input,
                "5" * 64,
                config,
            )

        self.assertEqual(events[:2], ["plan", "plan"])
        self.assertEqual(
            events[2:],
            [
                "offense_latents",
                "offense_latents",
                "pitching_latents",
                "pitching_latents",
            ],
        )

    def test_direct_game_rejects_out_of_range_sampled_workload(self):
        game, _ = _simulate_scripted_game([0, 1] + [0] * 15)

        with self.assertRaisesRegex(ValueError, "workload"):
            replace(game, away_pitcher_chain_workload_outs=28)


if __name__ == "__main__":
    unittest.main()
