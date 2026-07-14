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

    def test_unverified_bulk_handoff_uses_unscheduled_f5_relief(self):
        config = SimulationConfig(simulations=1)
        means = []
        scripted_runs = iter([0, 1] + [0] * 15)

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
            simulate_game_distribution(
                _game_sim_input(
                    config,
                    home_chain=_unverified_opener_bulk_chain(),
                ),
                "a" * 64,
                config,
            )

        self.assertAlmostEqual(
            means[2],
            config.league_runs_per_team_game / 9,
        )

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
