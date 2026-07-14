from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
import inspect
import json
import unittest
from unittest.mock import patch

from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import (
    BullpenInput,
    BullpenRole,
    BullpenSegmentInput,
    FeatureProvenance,
    MarketInput,
    MarketType,
    SettlementFormat,
)
from analysis.forecasting.market import (
    MarketEvaluation,
    OutcomeProbabilities,
    price_market,
)
from analysis.forecasting.simulation import (
    ComponentDistribution,
    DistributionSummary,
)
from tests.test_forecasting_inputs import (
    AVAILABLE_AT,
    CAPTURED_AT,
    boolean,
    conventional_chain,
    lineup_against,
    normalized_game,
    numeric,
    provenance,
    replace_all_provenance,
)


try:
    import analysis.forecasting.engine as forecast_engine_module
    from analysis.forecasting.engine import ForecastOutput, run_forecast
    resolve_forecast_input = getattr(
        forecast_engine_module,
        "resolve_forecast_input",
        None,
    )
except ModuleNotFoundError:
    forecast_engine_module = None
    ForecastOutput = None
    resolve_forecast_input = None
    run_forecast = None


ENGINE_API_AVAILABLE = all(
    value is not None
    for value in (forecast_engine_module, ForecastOutput, run_forecast)
)
UTC = timezone.utc
MARKET_KEYS = {
    "sea_f5",
    "f5_under",
    "sea_ml",
    "sea_plus_1_5",
    "full_under",
}
COMPONENT_KEYS = {
    "away_pitcher_chain",
    "home_pitcher_chain",
    "away_offense",
    "home_offense",
    "away_bullpen",
    "home_bullpen",
}


def _fast_config():
    return SimulationConfig(simulations=12)


def _chain(pitcher_id):
    chain = conventional_chain(pitcher_id)
    profile = chain.pitchers[0]
    right_usage = replace(
        profile.pitch_mix[0],
        opposing_side="R",
        usage=1.0,
    )
    left_usage = replace(right_usage, opposing_side="L")
    return replace(
        chain,
        pitchers=(replace(profile, pitch_mix=(right_usage, left_usage)),),
    )


def _bullpen(team):
    projected_ra9 = {
        BullpenRole.BRIDGE: 4.8,
        BullpenRole.MIDDLE: 4.5,
        BullpenRole.HIGH_LEVERAGE: 3.8,
        BullpenRole.CLOSER: 3.6,
    }
    return BullpenInput(
        team=team,
        segments=tuple(
            BullpenSegmentInput(
                segment_id=f"bullpen:{team}:{role.value}",
                role=role,
                projected_ra9=numeric(projected_ra9[role]),
                log_standard_error=numeric(0.15),
                appearances=(),
                available=boolean(True),
                limited_performance_or_use_risk=boolean(False),
            )
            for role in (
                BullpenRole.BRIDGE,
                BullpenRole.MIDDLE,
                BullpenRole.HIGH_LEVERAGE,
                BullpenRole.CLOSER,
            )
        ),
    )


def _market(market_key, market_type, selected_side, line):
    return MarketInput(
        market_key=market_key,
        market_type=market_type,
        settlement_format=SettlementFormat.TWO_WAY,
        selected_side=selected_side,
        line=line,
        selected_decimal_price=1.91,
        opposing_decimal_price=1.91,
        provenance=provenance(),
    )


def _markets():
    return (
        _market("sea_f5", MarketType.F5_SIDE, "away", 0.0),
        _market("f5_under", MarketType.F5_TOTAL, "under", 4.5),
        _market("sea_ml", MarketType.ML, "away", None),
        _market("sea_plus_1_5", MarketType.PLUS_1_5, "away", 1.5),
        _market("full_under", MarketType.FULL_TOTAL, "under", 8.5),
    )


def _complete_game(*, captured_at=CAPTURED_AT, markets=None):
    away_pitching = _chain("pitcher:away")
    home_pitching = _chain("pitcher:home")
    return normalized_game(
        captured_at=captured_at,
        away_pitching=away_pitching,
        home_pitching=home_pitching,
        away_lineup=lineup_against("SEA", home_pitching.matchup_pitcher_id),
        home_lineup=lineup_against("MIN", away_pitching.matchup_pitcher_id),
        away_bullpen=_bullpen("SEA"),
        home_bullpen=_bullpen("MIN"),
        markets=_markets() if markets is None else markets,
    )


class ForecastEngineApiAvailabilityTests(unittest.TestCase):
    def test_engine_api_is_available(self):
        self.assertTrue(ENGINE_API_AVAILABLE, "forecasting engine API is missing")

    @unittest.skipUnless(ENGINE_API_AVAILABLE, "forecasting engine API is missing")
    def test_package_exports_only_the_stable_market_and_engine_interfaces(self):
        import analysis.forecasting as forecasting_package

        expected = {
            "OutcomeProbabilities",
            "MarketEvaluation",
            "price_market",
            "evaluate_market",
            "ForecastOutput",
            "run_forecast",
        }
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, forecasting_package.__all__)
                self.assertTrue(hasattr(forecasting_package, name))


@unittest.skipUnless(ENGINE_API_AVAILABLE, "forecasting engine API is missing")
class ForecastEngineTests(unittest.TestCase):
    def test_public_input_resolver_canonicalizes_team_aliases_and_market_order(self):
        self.assertIsNotNone(
            resolve_forecast_input,
            "public forecast input resolver is missing",
        )
        team_market = _market("z:side", MarketType.ML, "SEA", None)
        total_market = _market(
            "a:total",
            MarketType.FULL_TOTAL,
            "under",
            8.5,
        )
        game = _complete_game(markets=(team_market, total_market))

        resolved = resolve_forecast_input(game)

        self.assertEqual(
            tuple(market.market_key for market in resolved.markets),
            ("a:total", "z:side"),
        )
        self.assertEqual(resolved.markets[1].selected_side, "away")
        self.assertEqual(game.markets[0].selected_side, "SEA")
        self.assertIn("resolve_forecast_input", forecast_engine_module.__all__)

    def test_engine_returns_every_verified_market_from_one_simulation(self):
        game = _complete_game()
        config = _fast_config()

        with (
            patch.object(
                forecast_engine_module,
                "forecast_pitcher_chain",
                wraps=forecast_engine_module.forecast_pitcher_chain,
            ) as pitcher_forecast,
            patch.object(
                forecast_engine_module,
                "forecast_offense",
                wraps=forecast_engine_module.forecast_offense,
            ) as offense_forecast,
            patch.object(
                forecast_engine_module,
                "forecast_bullpen",
                wraps=forecast_engine_module.forecast_bullpen,
            ) as bullpen_forecast,
            patch.object(
                forecast_engine_module,
                "simulate_game_distribution",
                wraps=forecast_engine_module.simulate_game_distribution,
            ) as simulate,
            patch.object(
                forecast_engine_module,
                "evaluate_market",
                wraps=forecast_engine_module.evaluate_market,
            ) as evaluate,
        ):
            output = run_forecast(game, config)

        self.assertEqual(set(output.markets), MARKET_KEYS)
        self.assertEqual(output.simulation_count, config.simulations)
        self.assertEqual(pitcher_forecast.call_count, 2)
        self.assertEqual(offense_forecast.call_count, 2)
        self.assertEqual(bullpen_forecast.call_count, 2)
        self.assertEqual(simulate.call_count, 1)
        self.assertEqual(evaluate.call_count, len(game.markets))
        shared_simulations = {id(call.args[0]) for call in evaluate.call_args_list}
        self.assertEqual(len(shared_simulations), 1)

    def test_offenses_receive_original_opposing_pitcher_profiles(self):
        game = _complete_game()

        with patch.object(
            forecast_engine_module,
            "forecast_offense",
            wraps=forecast_engine_module.forecast_offense,
        ) as offense_forecast:
            run_forecast(game, _fast_config())

        self.assertIs(
            offense_forecast.call_args_list[0].args[1],
            game.home_pitching.pitchers,
        )
        self.assertIs(
            offense_forecast.call_args_list[1].args[1],
            game.away_pitching.pitchers,
        )

    def test_team_coded_side_markets_resolve_to_canonical_sides(self):
        market_shapes = (
            (MarketType.F5_SIDE, 0.0),
            (MarketType.ML, None),
            (MarketType.PLUS_1_5, 1.5),
        )
        team_sides = (("SEA", "away"), ("MIN", "home"))

        for market_type, line in market_shapes:
            for team_id, canonical_side in team_sides:
                market = _market(
                    f"team:{market_type.value}:{team_id}",
                    market_type,
                    team_id,
                    line,
                )
                game = _complete_game(markets=(market,))
                with self.subTest(market_type=market_type, team_id=team_id):
                    with patch.object(
                        forecast_engine_module,
                        "evaluate_market",
                        wraps=forecast_engine_module.evaluate_market,
                    ) as evaluate:
                        try:
                            output = run_forecast(game, _fast_config())
                        except ValueError as error:
                            self.fail(f"team-coded side was not resolved: {error}")

                    resolved = evaluate.call_args.args[1]
                    self.assertEqual(resolved.selected_side, canonical_side)
                    self.assertEqual(resolved.market_key, market.market_key)
                    self.assertIs(resolved.provenance, market.provenance)
                    self.assertIn(market.market_key, output.markets)

    def test_team_side_aliases_share_canonical_identity_seed_and_output(self):
        team_market = _market("z:side", MarketType.ML, "SEA", None)
        canonical_market = replace(team_market, selected_side="away")
        total_market = _market(
            "a:total",
            MarketType.FULL_TOTAL,
            "under",
            8.5,
        )
        team_game = _complete_game(markets=(team_market, total_market))
        canonical_game = _complete_game(
            markets=(total_market, canonical_market)
        )

        with (
            patch.object(
                forecast_engine_module,
                "build_forecast_run_id",
                wraps=forecast_engine_module.build_forecast_run_id,
            ) as identity,
            patch.object(
                forecast_engine_module,
                "simulate_game_distribution",
                wraps=forecast_engine_module.simulate_game_distribution,
            ) as simulate,
        ):
            team_output = run_forecast(team_game, _fast_config())
            canonical_output = run_forecast(canonical_game, _fast_config())

        identity_games = [call.args[0] for call in identity.call_args_list]
        for identity_game in identity_games:
            self.assertEqual(
                tuple(market.market_key for market in identity_game.markets),
                ("a:total", "z:side"),
            )
            self.assertEqual(identity_game.markets[1].selected_side, "away")
        simulation_ids = [call.args[1] for call in simulate.call_args_list]
        simulation_seeds = [int(run_id[:16], 16) for run_id in simulation_ids]
        self.assertEqual(team_output.forecast_run_id, canonical_output.forecast_run_id)
        self.assertEqual(simulation_ids[0], simulation_ids[1])
        self.assertEqual(simulation_seeds[0], simulation_seeds[1])
        self.assertEqual(team_output, canonical_output)
        self.assertEqual(canonical_json(team_output), canonical_json(canonical_output))

    def test_unknown_team_side_is_rejected_before_simulation(self):
        market = _market("team:ml:unknown", MarketType.ML, "OAK", None)
        game = _complete_game(markets=(market,))

        with patch.object(
            forecast_engine_module,
            "simulate_game_distribution",
            wraps=forecast_engine_module.simulate_game_distribution,
        ) as simulate:
            with self.assertRaisesRegex(ValueError, "away or home team"):
                run_forecast(game, _fast_config())

        simulate.assert_not_called()

    def test_engine_is_reproducible_when_only_captured_at_changes(self):
        first = run_forecast(
            _complete_game(
                captured_at=datetime(2026, 7, 14, 1, 30, tzinfo=UTC)
            ),
            _fast_config(),
        )
        second = run_forecast(
            _complete_game(
                captured_at=datetime(2026, 7, 14, 1, 45, tzinfo=UTC)
            ),
            _fast_config(),
        )

        self.assertEqual(first.forecast_run_id, second.forecast_run_id)
        self.assertEqual(first, second)

    def test_market_order_is_semantically_stable(self):
        markets = _markets()

        first = run_forecast(_complete_game(markets=markets), _fast_config())
        second = run_forecast(
            _complete_game(markets=tuple(reversed(markets))),
            _fast_config(),
        )

        self.assertEqual(first.forecast_run_id, second.forecast_run_id)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first.markets), tuple(sorted(MARKET_KEYS)))

    def test_output_preserves_all_diagnostics_distributions_and_intervals(self):
        output = run_forecast(_complete_game(), _fast_config())

        self.assertEqual(set(output.component_diagnostics), COMPONENT_KEYS)
        self.assertIn("away_full_traffic", output.component_distributions)
        self.assertEqual(
            set(output.intervals),
            set(output.component_distributions),
        )
        for component_id, distribution in output.component_distributions.items():
            self.assertEqual(len(distribution.values), output.simulation_count)
            self.assertIs(output.intervals[component_id], distribution.summary)
        self.assertEqual(
            output.simulation_config["simulations"],
            output.simulation_count,
        )
        self.assertEqual(output.config_version, _fast_config().version)

    def test_output_rejects_component_values_outside_count_domains(self):
        output = run_forecast(_complete_game(), _fast_config())
        exposure_id = next(
            component_id
            for component_id in output.component_distributions
            if "_bullpen_segment_" in component_id
            and component_id.endswith("_exposed")
        )
        cases = (
            ("home_f5_runs", -1),
            ("home_full_runs", 1.5),
            ("home_protected_bullpen_close_entry", 2),
            (exposure_id, 2),
        )
        for component_id, value in cases:
            with self.subTest(component_id=component_id, value=value):
                values = (value,) * output.simulation_count
                distribution = ComponentDistribution(
                    component_id=component_id,
                    values=values,
                    summary=DistributionSummary(
                        value,
                        value,
                        value,
                        value,
                        value,
                        value,
                        value,
                    ),
                )
                distributions = dict(output.component_distributions)
                distributions[component_id] = distribution
                intervals = dict(output.intervals)
                intervals[component_id] = distribution.summary
                output_values = {
                    field.name: getattr(output, field.name)
                    for field in fields(output)
                }
                output_values["component_distributions"] = distributions
                output_values["intervals"] = intervals

                with self.assertRaisesRegex(
                    ValueError,
                    "domain|binary|count",
                ):
                    ForecastOutput(**output_values)

    def test_output_preserves_canonical_resolved_semantic_market_inputs(self):
        team_market = _market("z:side", MarketType.ML, "SEA", None)
        total_market = _market(
            "a:total",
            MarketType.FULL_TOTAL,
            "under",
            8.5,
        )

        output = run_forecast(
            _complete_game(markets=(team_market, total_market)),
            _fast_config(),
        )
        market_inputs = getattr(output, "market_inputs", None)

        self.assertIsNotNone(
            market_inputs,
            "ForecastOutput must preserve semantic market inputs",
        )
        self.assertEqual(tuple(market_inputs), ("a:total", "z:side"))
        self.assertEqual(tuple(market_inputs), tuple(output.markets))
        self.assertEqual(market_inputs["z:side"].market_key, "z:side")
        self.assertEqual(market_inputs["z:side"].selected_side, "away")
        self.assertEqual(market_inputs["z:side"].market_type, MarketType.ML)
        self.assertFalse(hasattr(market_inputs["z:side"], "provenance"))
        payload = json.loads(canonical_json(output))
        self.assertEqual(
            set(payload["market_inputs"]["z:side"]),
            {
                "line",
                "market_key",
                "market_type",
                "opposing_decimal_price",
                "selected_decimal_price",
                "selected_side",
                "settlement_format",
            },
        )

    def test_market_inputs_are_defensively_frozen(self):
        output = run_forecast(_complete_game(), _fast_config())
        market_inputs = getattr(output, "market_inputs", None)
        self.assertIsNotNone(
            market_inputs,
            "ForecastOutput must preserve semantic market inputs",
        )
        supplied = dict(market_inputs)
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        values["market_inputs"] = supplied

        reconstructed = ForecastOutput(**values)
        removed_key = next(iter(supplied))
        supplied.pop(removed_key)

        self.assertIn(removed_key, reconstructed.market_inputs)
        with self.assertRaises(TypeError):
            reconstructed.market_inputs["new"] = next(
                iter(reconstructed.market_inputs.values())
            )

    def test_market_input_keys_must_exactly_match_evaluated_markets(self):
        output = run_forecast(_complete_game(), _fast_config())
        market_inputs = getattr(output, "market_inputs", None)
        self.assertIsNotNone(
            market_inputs,
            "ForecastOutput must preserve semantic market inputs",
        )
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        mismatched = dict(market_inputs)
        mismatched.pop(next(iter(mismatched)))
        values["market_inputs"] = mismatched

        with self.assertRaisesRegex(ValueError, "market_inputs.*markets"):
            ForecastOutput(**values)

    def test_evaluated_market_prices_must_match_semantic_market_input(self):
        output = run_forecast(_complete_game(), _fast_config())
        market_key = "sea_ml"
        evaluation = output.markets[market_key]
        semantic_market = output.market_inputs[market_key]
        self.assertTrue(evaluation.evaluated)
        for field_name in (
            "selected_decimal_price",
            "opposing_decimal_price",
        ):
            with self.subTest(field_name=field_name):
                selected_price = semantic_market.selected_decimal_price
                opposing_price = semantic_market.opposing_decimal_price
                if field_name == "selected_decimal_price":
                    selected_price += 0.01
                else:
                    opposing_price += 0.01
                mismatched_markets = dict(output.markets)
                mismatched_markets[market_key] = price_market(
                    evaluation.probabilities,
                    selected_price,
                    opposing_price,
                )
                values = {
                    field.name: getattr(output, field.name)
                    for field in fields(output)
                }
                values["markets"] = mismatched_markets

                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name}.*market_inputs",
                ):
                    ForecastOutput(**values)

    def test_forecast_rejects_evaluation_bound_to_different_selected_side(self):
        output = run_forecast(_complete_game(), _fast_config())
        market_inputs = dict(output.market_inputs)
        market_inputs["sea_ml"] = replace(
            market_inputs["sea_ml"],
            selected_side="home",
        )
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        values["market_inputs"] = market_inputs

        with self.assertRaisesRegex(ValueError, "fingerprint.*market_inputs"):
            ForecastOutput(**values)

    def test_forecast_rejects_push_probability_for_non_push_market(self):
        output = run_forecast(_complete_game(), _fast_config())
        semantic = output.market_inputs["sea_ml"]
        evaluation = replace(
            price_market(
                OutcomeProbabilities(win=0.5, push=0.1, loss=0.4),
                semantic.selected_decimal_price,
                semantic.opposing_decimal_price,
            ),
            semantic_fingerprint=semantic.semantic_fingerprint,
        )
        markets = dict(output.markets)
        markets["sea_ml"] = evaluation
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        values["markets"] = markets

        with self.assertRaisesRegex(ValueError, "push"):
            ForecastOutput(**values)

    def test_forecast_rejects_unbound_unevaluated_ordinary_market(self):
        output = run_forecast(_complete_game(), _fast_config())
        markets = dict(output.markets)
        markets["sea_ml"] = MarketEvaluation(
            probabilities=None,
            model_conditional_win_probability=None,
            fair_decimal_price=None,
            market_no_vig_probability=None,
            edge=None,
            expected_value=None,
            evaluated=False,
            reason="arbitrary unevaluated replacement",
        )
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        values["markets"] = markets

        with self.assertRaisesRegex(ValueError, "fingerprint.*market_inputs"):
            ForecastOutput(**values)

    def test_forecast_rejects_nonprice_semantic_fingerprint_mismatches(self):
        output = run_forecast(_complete_game(), _fast_config())
        baseline = output.market_inputs["sea_ml"]
        mismatches = (
            ("market_type", {"market_type": MarketType.PLUS_1_5}),
            ("line", {"line": 1.5}),
            (
                "settlement_format",
                {"settlement_format": SettlementFormat.THREE_WAY},
            ),
        )

        for label, changes in mismatches:
            with self.subTest(label=label):
                market_inputs = dict(output.market_inputs)
                market_inputs["sea_ml"] = replace(baseline, **changes)
                values = {
                    field.name: getattr(output, field.name)
                    for field in fields(output)
                }
                values["market_inputs"] = market_inputs

                with self.assertRaisesRegex(
                    ValueError,
                    "fingerprint.*market_inputs",
                ):
                    ForecastOutput(**values)

    def test_unevaluated_market_keeps_semantic_prices_without_evaluation_prices(
        self,
    ):
        market = replace(
            _market("three-way-f5", MarketType.F5_SIDE, "away", 0.0),
            settlement_format=SettlementFormat.THREE_WAY,
        )

        output = run_forecast(
            _complete_game(markets=(market,)),
            _fast_config(),
        )
        evaluation = output.markets[market.market_key]
        reconstructed = ForecastOutput(
            **{field.name: getattr(output, field.name) for field in fields(output)}
        )

        self.assertFalse(evaluation.evaluated)
        self.assertEqual(
            output.market_inputs[market.market_key].selected_decimal_price,
            market.selected_decimal_price,
        )
        self.assertIsNone(evaluation.selected_decimal_price)
        self.assertIsNone(evaluation.opposing_decimal_price)
        self.assertEqual(reconstructed, output)

    def test_capture_availability_does_not_change_forecast_output(self):
        known_game = _complete_game()
        unknown = FeatureProvenance(
            observation_ids=("obs:1",),
            latest_available_at=None,
            lookback_end=AVAILABLE_AT,
        )
        unknown_game = replace_all_provenance(known_game, unknown)

        known_output = run_forecast(known_game, _fast_config())
        unknown_output = run_forecast(unknown_game, _fast_config())

        self.assertEqual(unknown_output, known_output)
        self.assertEqual(
            canonical_json(unknown_output),
            canonical_json(known_output),
        )

    def test_output_excludes_capture_validation_metadata(self):
        output = run_forecast(_complete_game(), _fast_config())
        field_names = {field.name for field in fields(output)}
        payload = json.loads(canonical_json(output))

        for capture_field in ("out_of_sample_eligible", "validation_flags"):
            with self.subTest(capture_field=capture_field):
                self.assertNotIn(capture_field, field_names)
                self.assertNotIn(capture_field, payload)

    def test_validation_failure_stops_before_identity_or_component_work(self):
        game = _complete_game()
        invalid = replace(game, captured_at=game.identity.first_pitch)

        with (
            patch.object(forecast_engine_module, "build_forecast_run_id") as identity,
            patch.object(forecast_engine_module, "forecast_pitcher_chain") as pitcher,
            patch.object(forecast_engine_module, "forecast_offense") as offense,
            patch.object(forecast_engine_module, "forecast_bullpen") as bullpen,
            patch.object(
                forecast_engine_module,
                "simulate_game_distribution",
            ) as simulate,
        ):
            with self.assertRaisesRegex(ValueError, "captured_at"):
                run_forecast(invalid, _fast_config())

        identity.assert_not_called()
        pitcher.assert_not_called()
        offense.assert_not_called()
        bullpen.assert_not_called()
        simulate.assert_not_called()

    def test_output_is_immutable_canonical_and_directly_reconstructible(self):
        output = run_forecast(_complete_game(), _fast_config())
        encoded = canonical_json(output)
        payload = json.loads(encoded)
        reconstructed = ForecastOutput(
            **{field.name: getattr(output, field.name) for field in fields(output)}
        )

        self.assertEqual(reconstructed, output)
        self.assertEqual(
            len(
                payload["component_distributions"]["away_full_traffic"][
                    "values"
                ]
            ),
            output.simulation_count,
        )
        self.assertTrue(
            all(
                isinstance(value, MarketEvaluation)
                for value in output.markets.values()
            )
        )
        with self.assertRaises(FrozenInstanceError):
            output.simulation_count = 1
        with self.assertRaises(TypeError):
            output.markets["new"] = next(iter(output.markets.values()))
        with self.assertRaises(TypeError):
            output.component_distributions["new"] = next(
                iter(output.component_distributions.values())
            )
        with self.assertRaises(TypeError):
            output.simulation_config["simulations"] = 1

    def test_direct_constructor_rejects_lossy_or_mismatched_intervals(self):
        output = run_forecast(_complete_game(), _fast_config())
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        intervals = dict(output.intervals)
        intervals.pop(next(iter(intervals)))
        values["intervals"] = intervals

        with self.assertRaisesRegex(ValueError, "interval"):
            ForecastOutput(**values)

    def test_direct_constructor_rejects_removed_bullpen_exposure_distribution(self):
        output = run_forecast(_complete_game(), _fast_config())
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        component_id = next(
            key
            for key in output.component_distributions
            if "_bullpen_segment_" in key
        )
        distributions = dict(output.component_distributions)
        intervals = dict(output.intervals)
        distributions.pop(component_id)
        intervals.pop(component_id)
        values["component_distributions"] = distributions
        values["intervals"] = intervals

        with self.assertRaisesRegex(ValueError, "component_distributions"):
            ForecastOutput(**values)

    def test_engine_does_not_import_legacy_or_ledger_decisions(self):
        source = inspect.getsource(forecast_engine_module)

        for forbidden in (
            "recommend_full_game",
            "evaluate_priority_admission",
            "from analysis.model import",
            "import analysis.model",
            "from analysis.ledger",
            "import analysis.ledger",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
