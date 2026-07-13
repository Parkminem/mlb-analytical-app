# Probability Forecasting Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert normalized starter, all-lineup pitch/velocity, bullpen, and market inputs into deterministic F5 and full-game score distributions and push-aware market probabilities.

**Architecture:** Build independent starter, offense, and bullpen forecast components that emit workload distributions or bounded multipliers, never independent run totals. A single seeded Gamma-Poisson simulator composes those components inning by inning, and a market evaluator settles F5 side, F5 O/U, ML, `+1.5`, and full-game O/U from the same simulated score paths.

**Tech Stack:** Python 3 frozen dataclasses, standard-library `random`/`math`/`statistics`, deterministic SHA-256 seeds, `unittest`

---

## Dependencies And Constraints

- Complete `2026-07-14-forecasting-ledger-foundation.md` first.
- Do not write to the ledger in this plan; return typed deterministic results for the later shadow adapter.
- Do not import or mutate legacy decision contexts in `analysis/model.py`.
- Use every lineup slot and the expected batter side against each explicit pitcher profile.
- Prefer the provided `±1.5 mph` response; use an already normalized `±3 mph` fallback only when exact-band evidence is absent.
- Treat missing pitch/velocity evidence as neutral with wider uncertainty, never as hitter weakness.
- Use one and only one run generator. Starter, offense, and bullpen modules cannot each add projected runs.
- Keep weather absent from this engine unless a future version adds a user-supplied normalized environment feature.

## File Map

- Create `analysis/forecasting/config.py`: immutable versioned simulation and multiplier constants.
- Create `analysis/forecasting/randomness.py`: bounded mean-preserving latent draws, Gamma-Poisson draws, weighted choice, and quantiles.
- Create `analysis/forecasting/starter.py`: starter/opener/bulk workload and run-prevention forecasts.
- Create `analysis/forecasting/offense.py`: all-nine-hitter pitch/velocity exposure and offense multipliers.
- Create `analysis/forecasting/bullpen.py`: role-segment availability, run-prevention, and fallback chain.
- Create `analysis/forecasting/simulation.py`: deterministic F5/regulation/extra-inning game simulation.
- Create `analysis/forecasting/market.py`: settlement, no-vig, fair-price, edge, and EV calculations.
- Create `analysis/forecasting/engine.py`: pure orchestration from normalized game input to one typed forecast output.
- Modify `analysis/forecasting/__init__.py`: export stable engine interfaces.
- Create `tests/test_forecasting_components.py`: starter, offense, and bullpen unit tests.
- Create `tests/test_forecasting_simulation.py`: deterministic simulation and baseball-state tests.
- Create `tests/test_forecasting_market.py`: settlement and market-value tests.
- Create `tests/test_forecasting_engine.py`: one-run-generator and end-to-end deterministic tests.

### Task 1: Versioned Configuration And Random Primitives

**Files:**
- Create: `analysis/forecasting/config.py`
- Create: `analysis/forecasting/randomness.py`
- Create: `tests/test_forecasting_components.py`

**Interfaces:**
- `SimulationConfig`
- `SimulationConfig.from_mapping(mapping)`
- `draw_bounded_lognormal(...)`
- `draw_gamma_poisson(...)`
- `weighted_choice(...)`
- `central_interval(...)`

- [ ] **Step 1: Write failing configuration and randomness tests**

```python
class ForecastRandomnessTests(unittest.TestCase):
    def test_default_configuration_is_frozen_and_versioned(self):
        config = SimulationConfig()
        self.assertEqual(config.league_runs_per_team_game, 4.50)
        self.assertEqual(config.gamma_poisson_shape, 1.60)
        self.assertEqual(config.simulations, 20_000)
        with self.assertRaises(FrozenInstanceError):
            config.simulations = 1

    def test_configuration_round_trips_through_canonical_mapping(self):
        original = SimulationConfig()
        rebuilt = SimulationConfig.from_mapping(original.to_mapping())
        self.assertEqual(rebuilt, original)

    def test_lognormal_uncertainty_does_not_raise_the_mean(self):
        rng = Random(17)
        draws = [
            draw_bounded_lognormal(rng, 1.0, 0.15, 0.55, 1.75)
            for _ in range(50_000)
        ]
        self.assertAlmostEqual(statistics.fmean(draws), 1.0, delta=0.015)

    def test_gamma_poisson_is_reproducible(self):
        first = draw_gamma_poisson(Random(9), mean=0.52, shape=1.60)
        second = draw_gamma_poisson(Random(9), mean=0.52, shape=1.60)
        self.assertEqual(first, second)
```

- [ ] **Step 2: Run primitive tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_components.ForecastRandomnessTests -v`

Expected: FAIL because configuration and random modules do not exist.

- [ ] **Step 3: Implement frozen configuration**

Use a frozen dataclass with every behavior-changing value included in canonical serialization:

```python
@dataclass(frozen=True)
class SimulationConfig:
    version: str = "shadow-forecast-2026-07-14.1"
    league_runs_per_team_game: float = 4.50
    league_woba: float = 0.320
    gamma_poisson_shape: float = 1.60
    simulations: int = 20_000
    extra_inning_multiplier: float = 1.80
    max_innings: int = 30
    offense_bounds: tuple[float, float] = (0.65, 1.45)
    matchup_bounds: tuple[float, float] = (0.80, 1.20)
    starter_ra9_bounds: tuple[float, float] = (0.55, 1.75)
    bullpen_ra9_bounds: tuple[float, float] = (0.60, 1.90)
    pa_weights: tuple[float, ...] = (
        1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87
    )
    pitch_limit_high_bucket_transfer: float = 1.0
    pitch_limit_transfer_to_under_five: float = 0.40
    sample_shortage_log_se_penalty: float = 0.12
    unknown_bullpen_matchup_log_se_penalty: float = 0.08
    handoff_uncertainty_log_se_penalty: float = 0.10
    injury_return_log_se_penalty: float = 0.12
    recent_short_exit_log_se_penalty: float = 0.10
    limited_reliever_log_se_penalty: float = 0.10
    unscheduled_f5_relief_log_standard_error: float = 0.15
```

Validate probabilities, bounds, positive simulation count, and exactly nine PA weights. Normalize PA weights to sum to nine inside the offense module, not by mutating configuration.

- [ ] **Step 4: Implement deterministic random helpers**

Use the mean-preserving draw from the approved design:

```python
latent = math.exp(math.log(point) - sigma * sigma / 2 + sigma * rng.gauss(0, 1))
return min(upper, max(lower, latent))
```

Implement Gamma-Poisson using `rng.gammavariate(shape, mean / shape)` and a documented Knuth Poisson draw. Add explicit guards for zero mean and invalid shape. Implement nearest-rank quantiles without NumPy.

- [ ] **Step 5: Run primitive tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_components.ForecastRandomnessTests -v`

Expected: all randomness tests PASS.

- [ ] **Step 6: Commit configuration and random primitives**

```bash
git add analysis/forecasting/config.py analysis/forecasting/randomness.py tests/test_forecasting_components.py
git commit -m "feat: add forecast simulation primitives"
```

### Task 2: Starter And Opener/Bulk Workload Forecasts

**Files:**
- Create: `analysis/forecasting/starter.py`
- Modify: `tests/test_forecasting_components.py`

**Interfaces:**
- `WorkloadDistribution`
- `PitcherForecast`
- `PitcherChainForecast`
- `forecast_pitcher_chain(chain, config)`

- [ ] **Step 1: Write failing workload and chain tests**

```python
class StarterForecastTests(unittest.TestCase):
    def test_workload_bucket_probabilities_sum_to_one(self):
        forecast = forecast_pitcher(sample_starter(eligible_outs=(12, 15, 17, 18, 21)), config())
        self.assertAlmostEqual(sum(forecast.workload.bucket_probabilities), 1.0)
        self.assertGreater(forecast.workload.probability_under_five, 0.0)
        self.assertGreater(forecast.workload.probability_six_plus, 0.0)

    def test_each_bucket_receives_one_league_prior_pseudo_start(self):
        forecast = forecast_pitcher(sample_starter(eligible_outs=(18, 21)), config())
        self.assertGreater(forecast.workload.probability_under_five, 0.0)
        self.assertGreater(forecast.workload.probability_five, 0.0)

    def test_documented_pitch_limit_moves_all_six_plus_mass(self):
        limited = forecast_pitcher(sample_starter(pitch_limit=80), config())
        self.assertEqual(limited.workload.probability_six_plus, 0.0)

    def test_ra9_is_a_multiplier_not_an_independent_run_total(self):
        forecast = forecast_pitcher(sample_starter(projected_ra9=5.40), config())
        self.assertAlmostEqual(forecast.run_prevention_multiplier, 1.20)
        self.assertFalse(hasattr(forecast, "projected_runs"))

    def test_unverified_bulk_handoff_widens_uncertainty_without_adding_runs(self):
        verified = forecast_pitcher_chain(sample_opener_bulk(handoff_verified=True), config())
        uncertain = forecast_pitcher_chain(sample_opener_bulk(handoff_verified=False), config())
        self.assertGreater(uncertain.log_standard_error, verified.log_standard_error)

    def test_opener_bulk_chain_samples_sequential_outs_and_combined_workload(self):
        chain = forecast_pitcher_chain(sample_opener_bulk(handoff_verified=True), config())
        plan = chain.sample_plan(Random(19))
        self.assertEqual([segment.role for segment in plan.segments], ["opener", "bulk"])
        self.assertEqual(plan.combined_outs, sum(s.outs for s in plan.segments))
        self.assertAlmostEqual(sum(chain.combined_workload.probabilities), 1.0)

    def test_combined_chain_buckets_use_combined_history_not_convolution(self):
        chain = sample_opener_bulk(
            opener_outs=(3, 3, 6),
            bulk_outs=(15, 18, 18),
            combined_outs=(18, 21, 24),
        )
        forecast = forecast_pitcher_chain(chain, config())
        self.assertEqual(forecast.combined_workload.source, "combined_chain_history")
```

- [ ] **Step 2: Run starter tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_components.StarterForecastTests -v`

Expected: FAIL because `analysis.forecasting.starter` does not exist.

- [ ] **Step 3: Implement empirical workload buckets**

Bucket recorded outs as `<15`, `15-17`, and `>=18`. Add exactly one pseudo-start per bucket using the median eligible league-prior out value in that bucket. Store both probabilities and within-bucket out supports.

When `pitch_limit` is documented, transfer `pitch_limit_high_bucket_transfer` of the `>=18` mass, allocating `40%` to `<15` and `60%` to `15-17`. Renormalize and record the transfer in notes. This is a versioned prior, not a learned value.

`sample_outs(rng)` first selects a bucket by probability, then selects one integer out value from its pitcher support plus one league-prior representative. Never parse baseball decimal innings.

- [ ] **Step 4: Implement run-prevention and chain forecasts**

```python
multiplier = clip(
    pitcher.projected_ra9.value / config.league_runs_per_team_game,
    *config.starter_ra9_bounds,
)
```

Carry command and HR flags as diagnostics. Compute each pitcher's uncertainty without inventing a point penalty:

```python
pitcher_sigma = math.sqrt(
    pitcher.log_standard_error.value ** 2
    + (config.injury_return_log_se_penalty ** 2 if pitcher.injury_return.value else 0)
    + (config.recent_short_exit_log_se_penalty ** 2 if pitcher.recent_short_exit.value else 0)
)
```

Preserve each opener/bulk/PRIM pitcher in declared chain order. A legal multi-pitcher chain must begin with `opener` and continue with `bulk` or `prim`; reject ambiguous duplicate roles. For a multi-pitcher chain, build the three workload buckets directly from `eligible_combined_workload.recorded_outs` and add one pseudo-start per bucket from `league_prior_combined_workload`; do not convolve starter-length per-pitcher histories. Apply only `combined_pitch_limit` to this combined distribution.

Sampling a chain first draws total combined outs from that combined distribution. For each non-final pitcher, draw role-specific outs from its individual distribution and cap them at remaining total outs. Assign all remaining outs to the final bulk/PRIM pitcher. Emit ordered `(pitcher_id, role, start_out, end_out)` segments whose outs sum exactly to the sampled combined total. A single conventional starter continues to use its individual workload distribution.

Compute chain uncertainty as the expected-outs-weighted root mean square of component sigmas, then add handoff uncertainty in quadrature:

```python
weighted_variance = sum(expected_outs[p] * sigma[p] ** 2 for p in pitchers) / sum(expected_outs.values())
chain_sigma = math.sqrt(
    weighted_variance
    + (
        config.handoff_uncertainty_log_se_penalty ** 2
        if not chain.handoff_verified.value or chain.timing_uncertain.value
        else 0
    )
)
```

An uncertain handoff still samples immediate opener-to-bulk sequencing; uncertainty changes only the latent distribution width and diagnostics, never an automatic run or over adjustment.

- [ ] **Step 5: Run starter tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_components.StarterForecastTests -v`

Expected: all starter tests PASS.

- [ ] **Step 6: Commit starter forecasts**

```bash
git add analysis/forecasting/starter.py tests/test_forecasting_components.py
git commit -m "feat: model starter workload distributions"
```

### Task 3: All-Lineup Pitch/Velocity Run Conversion

**Files:**
- Create: `analysis/forecasting/offense.py`
- Modify: `tests/test_forecasting_components.py`

**Interfaces:**
- `SlotMatchupForecast`
- `OffenseForecast(base_offense_multiplier, matchup_multipliers, projected_traffic_per_9, projected_extra_base_hits_per_9, log_standard_error, slot_forecasts)`
- `forecast_offense(lineup, opposing_profiles, config)`

- [ ] **Step 1: Write failing all-lineup tests**

```python
class OffenseForecastTests(unittest.TestCase):
    def test_all_nine_slots_are_scored_by_expected_side_and_pitch_usage(self):
        forecast = forecast_offense(full_lineup(), (left_handed_starter(),), config())
        self.assertEqual(len(forecast.slot_forecasts), 9)
        self.assertEqual(forecast.slot_forecasts[0].expected_side, "R")
        self.assertEqual(
            set(forecast.slot_forecasts[0].evaluated_pitch_types),
            {"four_seam", "changeup"},
        )

    def test_exact_velocity_band_is_selected_before_fallback(self):
        forecast = forecast_offense(lineup_with_both_bands(), (starter(),), config())
        self.assertEqual(forecast.slot_forecasts[0].selected_band_widths["slider"], 1.5)

    def test_nominal_exact_band_outside_actual_velocity_is_rejected(self):
        forecast = forecast_offense(
            lineup_with_response(pitch="slider", center=84.0, band=1.5),
            (starter(slider_velocity=87.0),),
            config(),
        )
        self.assertEqual(forecast.slot_forecasts[0].labels["slider"], "표본 부족")

    def test_missing_sample_is_neutral_and_only_widens_uncertainty(self):
        complete = forecast_offense(full_lineup(), (starter(),), config())
        missing = forecast_offense(lineup_with_missing_responses(), (starter(),), config())
        self.assertGreater(missing.log_standard_error, complete.log_standard_error)
        self.assertNotLess(missing.matchup_multipliers[starter().pitcher_id], 0.80)
        self.assertIn("표본 부족", missing.slot_forecasts[-1].labels)

    def test_recent_form_is_not_multiplied_twice(self):
        first = forecast_offense(full_lineup(recent_form_included=True), (starter(),), config())
        second = forecast_offense(full_lineup(recent_form_included=True), (starter(),), config())
        self.assertEqual(first.base_offense_multiplier, second.base_offense_multiplier)
        self.assertFalse(hasattr(first, "recent_form_multiplier"))
```

- [ ] **Step 2: Run offense tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_components.OffenseForecastTests -v`

Expected: FAIL because `analysis.forecasting.offense` does not exist.

- [ ] **Step 3: Implement batter-side and pitch-usage matching**

For every lineup slot and opposing pitcher:

1. Select only pitch usage matching that batter's expected side.
2. Match batter ID, pitcher ID, expected side, and pitch type. Compute `velocity_delta = abs(response.velocity_center_mph - usage.average_velocity_mph)`.
3. Select an exact response only when `response.band_width_mph == 1.5` and `velocity_delta <= 1.5`. If multiple remain, choose smallest delta, then largest PA.
4. Only when no exact response remains, select a fallback where `band_width_mph == 3.0` and `velocity_delta <= 3.0`, using the same tie breakers.
5. Use the already shrunk `adjusted_woba` point estimate.
6. If no response exists, use `league_woba`, mark `표본 부족`, and add that pitch usage to unknown exposure.
7. Weight pitch responses by that pitcher's normalized usage.
8. Weight all nine batter values by the normalized PA weights.

Compute:

```python
base = clip(
    lineup.projected_runs_per_9.value / config.league_runs_per_team_game,
    *config.offense_bounds,
)
matchup = clip(
    exposure_weighted_adjusted_woba / config.league_woba,
    *config.matchup_bounds,
)
sigma = math.sqrt(
    lineup.log_standard_error.value ** 2
    + (unknown_exposure * config.sample_shortage_log_se_penalty) ** 2
)
```

Return matchup multipliers keyed by every supplied pitcher ID. Do not infer weakness from handedness alone. Include slot-level pitch types, bands, PA, adjusted wOBA, labels, and unknown exposure for written review.

Carry `lineup.projected_traffic_per_9.value` and `lineup.projected_extra_base_hits_per_9.value` into the forecast as diagnostic distribution means. Walk, strikeout, ISO, barrel, recent-form, and lower-order inputs remain auditable component fields; they are already represented in `projected_runs_per_9` and may not be multiplied into run intensity again in this version.

- [ ] **Step 4: Add neutral bullpen fallback behavior**

`OffenseForecast.multiplier_for(pitcher_or_segment_id)` returns an explicit pitcher multiplier when available. For an unprofiled bullpen segment it returns point estimate `1.0` and increases the applicable log standard error by `unknown_bullpen_matchup_log_se_penalty`; it must not silently reuse the starter matchup.

- [ ] **Step 5: Run offense tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_components.OffenseForecastTests -v`

Expected: all offense tests PASS.

- [ ] **Step 6: Commit offense forecasts**

```bash
git add analysis/forecasting/offense.py tests/test_forecasting_components.py
git commit -m "feat: convert all-lineup matchup exposure"
```

### Task 4: Bullpen Role Chain And Availability

**Files:**
- Create: `analysis/forecasting/bullpen.py`
- Modify: `tests/test_forecasting_components.py`

**Interfaces:**
- `BullpenSegmentForecast`
- `BullpenForecast`
- `forecast_bullpen(bullpen, config)`
- `select_bullpen_segment(forecast, inning, margin, bridge_needed, verified_opener_bulk)`

- [ ] **Step 1: Write failing bullpen-chain tests**

```python
class BullpenForecastTests(unittest.TestCase):
    def test_rested_high_leverage_is_distinct_from_used_lower_relief(self):
        forecast = forecast_bullpen(bullpen_with_tired_lower_and_rested_high(), config())
        self.assertTrue(forecast.high_leverage_available)
        self.assertTrue(forecast.lower_relief_risk)

    def test_unverified_early_relief_uses_neutral_f5_prior_not_team_bridge(self):
        forecast = forecast_bullpen(standard_bullpen(), config())
        segment = select_bullpen_segment(
            forecast,
            inning=5,
            margin=1,
            bridge_needed=True,
            verified_opener_bulk=False,
        )
        self.assertEqual(segment.segment_id, "league_unscheduled_f5_relief")
        self.assertEqual(segment.run_prevention_multiplier, 1.0)

    def test_bridge_can_enter_from_sixth_after_short_starter(self):
        forecast = forecast_bullpen(standard_bullpen(), config())
        segment = select_bullpen_segment(
            forecast,
            inning=6,
            margin=1,
            bridge_needed=True,
            verified_opener_bulk=False,
        )
        self.assertEqual(segment.role, BullpenRole.BRIDGE)

    def test_unavailable_high_leverage_falls_back_without_mutation(self):
        forecast = forecast_bullpen(bullpen_without_high_leverage(), config())
        segment = select_bullpen_segment(
            forecast, inning=8, margin=1, bridge_needed=False, verified_opener_bulk=False
        )
        self.assertEqual(segment.role, BullpenRole.MIDDLE)

    def test_limited_arm_is_use_and_performance_risk_not_automatic_absence(self):
        forecast = forecast_bullpen(bullpen_with_limited_setup(), config())
        setup = forecast.by_role[BullpenRole.HIGH_LEVERAGE]
        self.assertTrue(setup.available)
        self.assertGreater(setup.log_standard_error, setup.base_log_standard_error)
```

- [ ] **Step 2: Run bullpen tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_components.BullpenForecastTests -v`

Expected: FAIL because `analysis.forecasting.bullpen` does not exist.

- [ ] **Step 3: Implement segment multipliers and role separation**

Compute each segment multiplier as `projected_ra9 / 4.50`, clipped to `[0.60, 1.90]`. Keep bridge, middle, high leverage, and closer records separate. Compute uncertainty as:

```python
segment_sigma = math.sqrt(
    segment.log_standard_error.value ** 2
    + (
        config.limited_reliever_log_se_penalty ** 2
        if segment.limited_performance_or_use_risk.value
        else 0
    )
)
```

Limited performance/use risk adds variance and a note but leaves `available=True`; explicitly unavailable segments are skipped. Determine these states only from the normalized, provenance-bearing input and never recalculate recent usage inside this module.

- [ ] **Step 4: Implement deterministic role selection**

Use this initial chain:

- Before inning six, a verified opener/bulk/PRIM remains part of the initial pitcher chain. Any other unplanned early exit uses a synthetic `league_unscheduled_f5_relief` prior with run-prevention multiplier `1.0` and log standard error `config.unscheduled_f5_relief_log_standard_error`; team bridge/middle quality and availability are forbidden from this segment.
- Innings six and seven: middle unless bridge innings are still required.
- Innings seven and eight with margin `<=3`: available high leverage.
- Inning nine or later with margin `<=3`: closer, then high leverage, then middle.
- Non-close late innings: middle or bridge, never force rested closer usage.

From inning six onward, return a role fallback or fail validation when no usable team segment exists; do not synthesize a league-average late arm. The one explicit exception is the versioned neutral unscheduled-F5 prior above, which exists solely to keep full-game bullpen quality out of F5 O/U.

- [ ] **Step 5: Run bullpen tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_components.BullpenForecastTests -v`

Expected: all bullpen tests PASS.

- [ ] **Step 6: Commit bullpen forecasts**

```bash
git add analysis/forecasting/bullpen.py tests/test_forecasting_components.py
git commit -m "feat: model bullpen role chains"
```

### Task 5: One-Generator Game Simulation

**Files:**
- Create: `analysis/forecasting/simulation.py`
- Create: `tests/test_forecasting_simulation.py`

**Interfaces:**
- `GameSimulationInput`
- `SimulatedGame`
- `DistributionSummary(p05, p10, p20, median, p80, p90, p95)`
- `ComponentDistribution(component_id, values, summary)`
- `SimulationResult`
- `simulate_game_distribution(input, forecast_run_id, config)`

- [ ] **Step 1: Write failing deterministic and baseball-state tests**

```python
class GameSimulationTests(unittest.TestCase):
    def test_same_run_id_reproduces_exact_scores(self):
        first = simulate_game_distribution(game_sim_input(), "a" * 64, fast_config())
        second = simulate_game_distribution(game_sim_input(), "a" * 64, fast_config())
        self.assertEqual(first.games, second.games)

    def test_different_run_id_changes_seeded_paths(self):
        first = simulate_game_distribution(game_sim_input(), "a" * 64, fast_config())
        second = simulate_game_distribution(game_sim_input(), "b" * 64, fast_config())
        self.assertNotEqual(first.games, second.games)

    def test_home_lead_skips_bottom_ninth(self):
        game = simulate_scripted_game(home_leads_after_top_ninth=True)
        self.assertFalse(game.bottom_ninth_played)

    def test_walkoff_is_capped_at_runs_needed_to_win(self):
        game = simulate_scripted_game(bottom_tenth_draw=5, home_deficit=1)
        self.assertEqual(game.home_full_runs - game.away_full_runs, 1)

    def test_starter_and_bullpen_are_not_added_as_separate_run_totals(self):
        source = inspect.getsource(mean_runs_for_half_inning)
        self.assertNotIn("projected_runs", source)
        self.assertIn("run_prevention_multiplier", source)

    def test_simulation_emits_reviewable_component_distributions(self):
        result = simulate_game_distribution(game_sim_input(), "a" * 64, fast_config())
        required = {
            "away_f5_runs", "home_f5_runs", "away_full_runs", "home_full_runs",
            "away_full_traffic", "home_full_traffic",
            "away_full_extra_base_hits", "home_full_extra_base_hits",
            "away_pitcher_chain_runs_allowed", "home_pitcher_chain_runs_allowed",
            "away_bullpen_runs_allowed", "home_bullpen_runs_allowed",
            "away_pitcher_chain_workload_outs", "home_pitcher_chain_workload_outs",
            "away_protected_bullpen_close_entry", "home_protected_bullpen_close_entry",
        }
        self.assertTrue(required.issubset(result.component_distributions))
        self.assertIsNotNone(
            result.component_distributions["away_pitcher_chain_workload_outs"].summary.p80
        )
```

- [ ] **Step 2: Run simulation tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_simulation -v`

Expected: FAIL because the simulator does not exist.

- [ ] **Step 3: Implement seeded latent component draws**

Seed `random.Random` with `int(forecast_run_id[:16], 16)`. For each simulated game:

- Draw each team's base offense multiplier once.
- Draw each explicit pitcher and bullpen-segment run-prevention multiplier once.
- Draw each applicable matchup multiplier once.
- Draw exact starter/opener/bulk outs before simulating innings.
- Reuse those latent draws across innings to preserve within-game correlation.

- [ ] **Step 4: Implement the sole run-intensity equation**

For a full inning against one segment:

```python
mean = (
    config.league_runs_per_team_game / 9
    * offense_multiplier
    * matchup_multiplier
    * run_prevention_multiplier
)
```

When a pitcher exits partway through an inning, use out fractions inside the same equation rather than adding component totals:

```python
combined_pitching_effect = (
    starter_fraction * starter_matchup * starter_run_prevention
    + bullpen_fraction * bullpen_matchup * bullpen_run_prevention
)
mean = league_inning_rate * offense_multiplier * combined_pitching_effect
```

Then make exactly one Gamma-Poisson run draw for that batting half-inning. In a mixed exit inning, allocate each realized run to the pitcher-chain or bullpen diagnostic with a Bernoulli probability equal to that segment's share of the combined run intensity; this attribution does not create another run.

For innings one through five, the replacement term may use only a verified opener/bulk/PRIM pitcher from the initial chain. An unscheduled exit uses `league_unscheduled_f5_relief`; never read team bridge, middle, setup, or closer multipliers until inning six. This keeps the same simulated F5 score paths valid for both side and total evaluation without leaking full-game bullpen quality into F5 O/U.

Using the same matchup draw and one mean-preserving diagnostic uncertainty draw centered at `1.0` with the lineup log standard error, generate review-only traffic and extra-base-hit counts:

```python
traffic_mean = offense.projected_traffic_per_9 / 9 * matchup_multiplier * diagnostic_uncertainty
extra_base_hit_mean = offense.projected_extra_base_hits_per_9 / 9 * matchup_multiplier * diagnostic_uncertainty
```

Draw both counts with `draw_gamma_poisson(rng, mean, config.gamma_poisson_shape)` once per batting half-inning. Draw traffic first, draw raw extra-base hits second, and store `extra_base_hits = min(traffic, raw_extra_base_hits)` so the diagnostic state remains coherent. These diagnostic draws never feed back into runs in this version, preventing conversion double counting.

- [ ] **Step 5: Implement F5, regulation, and extra innings**

- Complete both halves through inning five and snapshot F5 scores.
- Skip bottom nine when home already leads.
- Cap a bottom-nine-or-later walk-off at the exact runs required.
- Apply `extra_inning_multiplier=1.80` from inning ten onward.
- Continue until untied; at inning 30, use a seeded Bernoulli weighted by current extra-inning scoring intensities and award exactly one run.
- Retain all extra-inning runs in full-game totals.

Each `SimulatedGame` stores F5/full runs, F5/full traffic and extra-base hits by batting side, runs allowed while the initial pitcher chain is active, bullpen runs allowed, sampled combined chain workload outs, the set of bullpen segment IDs actually exposed, and whether a protected high-leverage/closer segment entered with a margin of three runs or fewer. Component IDs always use `away` or `home`, never a team abbreviation. Aggregate them into `ComponentDistribution` values and p05/p10/p20/median/p80/p90/p95 summaries. Emit binary `away_protected_bullpen_close_entry`/`home_protected_bullpen_close_entry` distributions and one binary `{side}_bullpen_segment_{segment_id}_exposed` distribution for every available segment. These outputs are the sole quantitative source for postgame path review; Plan 3 may ingest official component values but must not reconstruct forecast intervals from boxscores.

The fixed base ID set for each side is `{side}_f5_runs`, `{side}_full_runs`, `{side}_f5_traffic`, `{side}_full_traffic`, `{side}_f5_extra_base_hits`, `{side}_full_extra_base_hits`, `{side}_pitcher_chain_runs_allowed`, `{side}_bullpen_runs_allowed`, `{side}_pitcher_chain_workload_outs`, and `{side}_protected_bullpen_close_entry`. Here batting statistics use the batting side, while runs-allowed/workload/exposure statistics use the pitching side.

- [ ] **Step 6: Run simulation tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_simulation -v`

Expected: all simulation tests PASS with a reduced `simulations=500` test configuration.

- [ ] **Step 7: Commit the simulator**

```bash
git add analysis/forecasting/simulation.py tests/test_forecasting_simulation.py
git commit -m "feat: simulate deterministic game scores"
```

### Task 6: Market Settlement And Pure Forecast Engine

**Files:**
- Create: `analysis/forecasting/market.py`
- Create: `analysis/forecasting/engine.py`
- Modify: `analysis/forecasting/__init__.py`
- Create: `tests/test_forecasting_market.py`
- Create: `tests/test_forecasting_engine.py`

**Interfaces:**
- `OutcomeProbabilities`
- `MarketEvaluation`
- `evaluate_market(simulation, market_input)`
- `ForecastOutput`
- `run_forecast(normalized_game, config) -> ForecastOutput`

- [ ] **Step 1: Write failing settlement and value tests**

Use fixed score paths rather than Monte Carlo for exact expected results:

```python
class MarketSettlementTests(unittest.TestCase):
    def test_f5_tie_is_push_for_two_way_market(self):
        result = evaluate_market(scores([(2, 2, 4, 3)]), f5_side(line=0, price=1.91))
        self.assertEqual(result.probabilities.push, 1.0)
        self.assertIsNone(result.model_conditional_win_probability)
        self.assertIsNone(result.fair_decimal_price)
        self.assertIsNone(result.edge)
        self.assertEqual(result.expected_value, 0.0)

    def test_integer_total_can_push(self):
        result = evaluate_market(scores([(1, 1, 4, 4)]), full_total(line=8.0, side="over"))
        self.assertEqual(result.probabilities.push, 1.0)

    def test_full_game_ml_has_no_push(self):
        result = evaluate_market(scores([(2, 2, 5, 4)]), ml(side="home"))
        self.assertEqual(result.probabilities.push, 0.0)

    def test_push_aware_fair_price_and_ev(self):
        probabilities = OutcomeProbabilities(win=0.50, push=0.10, loss=0.40)
        result = price_market(probabilities, selected_price=2.00, opposing_price=1.90)
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

    def test_missing_entry_price_keeps_probability_and_flow_review(self):
        result = price_market(
            OutcomeProbabilities(win=0.55, push=0.0, loss=0.45),
            selected_price=None,
            opposing_price=None,
        )
        self.assertAlmostEqual(result.fair_decimal_price, 1 + 0.45 / 0.55)
        self.assertIsNone(result.expected_value)
        self.assertIsNone(result.market_no_vig_probability)
```

- [ ] **Step 2: Run market tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_market -v`

Expected: FAIL because market evaluation does not exist.

- [ ] **Step 3: Implement settlement and price formulas**

Evaluate selected side from every simulated score path. Fix formulas as:

```python
fair_decimal_price = 1 + p_loss / p_win
model_conditional = p_win / (1 - p_push)
selected_implied = 1 / selected_price
opposing_implied = 1 / opposing_price
market_no_vig = selected_implied / (selected_implied + opposing_implied)
edge = model_conditional - market_no_vig
```

Compute `expected_value = p_win * (selected_price - 1) - p_loss` only when selected price exists; otherwise return `expected_value=None` while preserving raw probability, push probability, fair price, and postgame review eligibility.

Guard degenerate cases before division:

- If `P(push) == 1`, return `expected_value=0.0` when a selected price exists (otherwise `None`) and `model_conditional_win_probability=None`, `fair_decimal_price=None`, and `edge=None`.
- If `P(win) == 0` with non-push outcomes, fair price is `None` and EV remains `-P(loss)`.
- No-vig market probability may still be computed from two prices, but edge remains `None` when model conditional probability is undefined.

Return no-vig/edge as `None` when opposing price is absent. Record unsupported three-way F5 markets but return `evaluated=False` until separately implemented.

- [ ] **Step 4: Write failing end-to-end engine tests**

```python
class ForecastEngineTests(unittest.TestCase):
    def test_engine_returns_every_verified_market_from_one_simulation(self):
        output = run_forecast(complete_game_input(), fast_config())
        self.assertEqual(
            set(output.markets),
            {"sea_f5", "f5_under", "sea_ml", "sea_plus_1_5", "full_under"},
        )
        self.assertEqual(output.simulation_count, fast_config().simulations)
        self.assertIn("away_full_traffic", output.component_distributions)

    def test_engine_is_reproducible_for_same_semantic_input(self):
        first = run_forecast(complete_game_input(captured_at=dt(1)), fast_config())
        second = run_forecast(complete_game_input(captured_at=dt(2)), fast_config())
        self.assertEqual(first.forecast_run_id, second.forecast_run_id)
        self.assertEqual(first, second)

    def test_engine_does_not_import_legacy_recommendation_functions(self):
        source = inspect.getsource(forecast_engine_module)
        self.assertNotIn("recommend_full_game", source)
        self.assertNotIn("evaluate_priority_admission", source)
```

- [ ] **Step 5: Implement pure orchestration**

`run_forecast` must:

1. Validate the normalized pregame input.
2. Build its deterministic forecast-run ID from semantic input plus full simulation configuration.
3. Forecast both pitcher chains, both offenses, and both bullpens once.
4. Simulate one shared set of game paths.
5. Evaluate every supplied verified market from those same paths.
6. Copy the simulator's fixed-ID `component_distributions` and their intervals into `ForecastOutput` without summarizing them away.
7. Return component diagnostics, intervals, simulation metadata, and market evaluations without writing ledger records or producing legacy `play/lean/pass` decisions.

- [ ] **Step 6: Run market and engine tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_market tests.test_forecasting_engine -v`

Expected: all market and engine tests PASS.

- [ ] **Step 7: Commit market evaluation and engine orchestration**

```bash
git add analysis/forecasting/market.py analysis/forecasting/engine.py analysis/forecasting/__init__.py tests/test_forecasting_market.py tests/test_forecasting_engine.py
git commit -m "feat: evaluate simulated betting markets"
```

### Task 7: Probability Engine Verification

**Files:**
- Verify all files created or modified in this plan.

- [ ] **Step 1: Run all forecasting tests**

Run: `python3 -m unittest tests.test_forecasting_inputs tests.test_ledger tests.test_forecasting_components tests.test_forecasting_simulation tests.test_forecasting_market tests.test_forecasting_engine -v`

Expected: all forecasting and ledger tests PASS.

- [ ] **Step 2: Run the full legacy regression suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all legacy and new tests PASS.

- [ ] **Step 3: Verify determinism outside the test runner**

Run: `python3 -m unittest tests.test_forecasting_engine.ForecastEngineTests.test_engine_is_reproducible_for_same_semantic_input tests.test_forecasting_simulation.GameSimulationTests.test_same_run_id_reproduces_exact_scores -v`

Expected: both determinism tests PASS with exact object equality.

- [ ] **Step 4: Check formatting and focused diff**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git diff -- analysis/forecasting tests/test_forecasting_components.py tests/test_forecasting_simulation.py tests/test_forecasting_market.py tests/test_forecasting_engine.py`

Expected: no edits to `analysis/model.py` and no ledger writes inside the pure engine.

## Completion Gate

- One deterministic score distribution feeds F5, ML, `+1.5`, F5 O/U, and full-game O/U.
- Starter, offense, and bullpen components never add overlapping run totals.
- All nine hitters are evaluated against the relevant pitch mix and velocity response by expected batter side.
- Missing samples widen uncertainty but do not lower the point estimate as weakness.
- Re-running the same semantic state returns byte-equivalent probabilities.
- Existing fixed-model decisions remain untouched.
