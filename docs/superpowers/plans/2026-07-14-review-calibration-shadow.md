# Review Calibration And Shadow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify frozen pregame forecasts against official outcomes, report calibration and betting performance without duplicate samples, and record new probabilities beside the unchanged legacy model in mandatory shadow mode.

**Architecture:** Add a review package that links one eligible capture/forecast pair to the latest official result and applies deterministic A/B/C/D flow rules. Calibration consumes deduplicated review rows, while a one-way shadow adapter copies already-computed legacy decisions into forecast records but has no API capable of mutating or replacing them.

**Tech Stack:** Python 3 frozen dataclasses, standard-library math/CSV/JSON, existing ledger and forecast engine, `unittest`, JSON configuration, Markdown documentation

---

## Dependencies And Constraints

- Complete `2026-07-14-forecasting-ledger-foundation.md` and `2026-07-14-probability-forecasting-engine.md` first.
- Use only official MLB outcomes and source revisions for postgame records.
- Never reconstruct a pregame feature from postgame data.
- Keep the top-level fixed-model version and all legacy recommendation behavior unchanged.
- The new engine remains `shadow`; production precedence requires a separate approved design.
- One official game can produce multiple market reviews, but unchanged repeated captures cannot increase sample size.
- Missing entry odds or closing lines must yield missing ROI/CLV fields, not invented values.

## File Map

- Create `analysis/review/__init__.py`: public review and calibration exports.
- Create `analysis/review/classification.py`: predictive intervals, component observations, deterministic path contradictions, and A/B/C/D classification.
- Create `analysis/review/calibration.py`: primary-row selection, Brier/log loss/ROI/CLV, grouping, and reports.
- Create `analysis/review/service.py`: official-result settlement, forecast-component binding, classification orchestration, and idempotent review persistence.
- Create `analysis/shadow.py`: one-way legacy snapshot adapter and idempotent capture/forecast/link recording.
- Create `tests/test_review_classification.py`: A/B/C/D and path-contradiction fixtures.
- Create `tests/test_calibration.py`: deduplication, push, probability-score, ROI, and CLV fixtures.
- Create `tests/test_review_service.py`: link/forecast/result joins, market settlement, component mapping, and review persistence.
- Create `tests/test_shadow.py`: immutable legacy precedence and ledger integration tests.
- Modify `analysis/pipeline.py`: expose shadow metadata without changing required markets or legacy authority.
- Modify `config/model.json`: add separate forecasting version/mode/configuration; do not change legacy thresholds.
- Modify `tests/test_pipeline.py`: assert shadow mode and unchanged legacy manifest behavior.
- Modify `README.md`: document shadow workflow and local ledger.
- Modify `docs/model.md`: document the non-authoritative probability layer and validation thresholds.
- Modify `docs/data-sources.md`: document observation timestamps and official-result revisions.
- Modify `docs/analysis-output-template.md`: distinguish legacy rank score from model probability/fair price.

### Task 1: Deterministic Postgame Flow Classification

**Files:**
- Create: `analysis/review/__init__.py`
- Create: `analysis/review/classification.py`
- Create: `tests/test_review_classification.py`

**Interfaces:**
- `PredictiveInterval`
- `ComponentForecast`
- `ComponentObservation`
- `PathReview`
- `ReviewCandidate`
- `ReviewClassification`
- `classify_review(candidate) -> ReviewClassification`

- [ ] **Step 1: Write failing A/B/C/D precedence tests**

```python
class ReviewClassificationTests(unittest.TestCase):
    def test_a_requires_win_and_at_least_seventy_five_percent_inside_80(self):
        result = classify_review(candidate(
            market_result="win",
            components=(inside80(), inside80(), inside80(), outside80_inside90()),
        ))
        self.assertEqual(result.classification, "A")

    def test_b_includes_losing_ticket_with_broadly_correct_flow(self):
        result = classify_review(candidate(
            market_result="loss",
            components=(inside80(), inside80(), outside80_inside90()),
        ))
        self.assertEqual(result.classification, "B")

    def test_winning_ticket_with_wrong_flow_is_c(self):
        result = classify_review(candidate(
            market_result="win",
            components=(inside80(), outside90()),
        ))
        self.assertEqual(result.classification, "C")

    def test_documented_injury_outside_90_takes_d_precedence(self):
        result = classify_review(candidate(
            market_result="loss",
            components=(outside90(component_id="starter_workload"),),
            event=exogenous_event("injury", affected="starter_workload"),
        ))
        self.assertEqual(result.classification, "D")

    def test_no_recommended_market_has_component_review_without_ab_class(self):
        result = classify_review(candidate(recommended=False, components=(inside80(),)))
        self.assertIsNone(result.classification)
        self.assertEqual(result.inside_80_ratio, 1.0)
```

- [ ] **Step 2: Run classification tests and verify RED**

Run: `python3 -m unittest tests.test_review_classification -v`

Expected: FAIL because `analysis.review` does not exist.

- [ ] **Step 3: Implement immutable interval and observation types**

```python
@dataclass(frozen=True)
class PredictiveInterval:
    p05: float
    p10: float
    p20: float
    median: float
    p80: float
    p90: float
    p95: float


@dataclass(frozen=True)
class ComponentForecast:
    component_id: str
    evidence_path: PriorityEvidencePath
    interval: PredictiveInterval
    direction: str  # high_is_supportive or low_is_supportive


@dataclass(frozen=True)
class ComponentObservation:
    component_id: str
    value: float


@dataclass(frozen=True)
class ExogenousEvent:
    event_type: str
    affected_component_id: str
    documented_source_observation_id: str
    unearned_runs: int = 0


class ContradictionRule(str, Enum):
    STARTER_SUPPRESSION = "starter_suppression"
    STARTER_COLLAPSE = "starter_collapse"
    TOP_ORDER_OR_RECENT_CONVERSION = "top_order_or_recent_conversion"
    LATE_BULLPEN_EXPOSURE = "late_bullpen_exposure"
    BULLPEN_PROTECTION = "bullpen_protection"


@dataclass(frozen=True)
class PrincipalPathBinding:
    evidence_path: PriorityEvidencePath
    component_ids: tuple[str, ...]
    contradiction_rule: "ContradictionRule"


@dataclass(frozen=True)
class PathReview:
    evidence_path: PriorityEvidencePath
    component_ids: tuple[str, ...]
    contradicted: bool
    inside_80_count: int
    component_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewCandidate:
    capture_forecast_link_record_id: str
    capture_id: str
    forecast_run_id: str
    official_result_record_id: str
    market_key: str
    recommended: bool
    market_result: str | None
    component_forecasts: tuple[ComponentForecast, ...]
    component_observations: tuple[ComponentObservation, ...]
    principal_paths: tuple[PrincipalPathBinding, ...]
    exogenous_events: tuple[ExogenousEvent, ...] = ()


@dataclass(frozen=True)
class ReviewClassification:
    classification: str | None
    inside_80_ratio: float
    contradicted_paths: tuple[PriorityEvidencePath, ...]
    path_reviews: tuple[PathReview, ...]
    notes: tuple[str, ...] = ()
```

Validate interval ordering; unique component IDs; complete observation coverage; and two to four unique principal canonical `PriorityEvidencePath` bindings per recommended candidate. Every bound component ID must exist in both forecast and observation tuples. Accept one or more components per path while counting each component once globally even if two paths reference it.

- [ ] **Step 4: Implement path contradictions**

Encode the approved rules directly:

- Starter suppression: chain runs allowed above `p90`.
- Starter collapse: chain runs allowed below `p10` while workload reaches forecast median or longer.
- Top-order/recent conversion: team runs, traffic, and XBH all below their `p10` values.
- Late-bullpen exposure: forecast segment not exposed while starter workload reaches `p80`; include `p80` in the component's metadata rather than estimating it after the game.
- Bullpen protection: bullpen runs above `p90` when protected innings begin within three runs.

Validate each contradiction rule against fixed binding shapes before classification:

- `STARTER_SUPPRESSION`: exactly `{pitching_side}_pitcher_chain_runs_allowed`.
- `STARTER_COLLAPSE`: chain runs allowed plus `{pitching_side}_pitcher_chain_workload_outs`.
- `TOP_ORDER_OR_RECENT_CONVERSION`: one horizon-matched runs component plus traffic and extra-base-hit components for the same batting side (`f5` for F5 markets, `full` otherwise).
- `LATE_BULLPEN_EXPOSURE`: `{pitching_side}_bullpen_segment_{segment_id}_exposed` plus pitcher-chain workload outs.
- `BULLPEN_PROTECTION`: `{pitching_side}_bullpen_runs_allowed` plus `{pitching_side}_protected_bullpen_close_entry`.

Reject a binding with missing, extra, cross-team, or cross-horizon components rather than interpreting it dynamically.

Do not classify normal sequencing variance as contradiction unless a predeclared component leaves its central 90% interval.

- [ ] **Step 5: Implement exhaustive class precedence**

Apply exactly:

```python
if qualifying_exogenous_event_outside_90:
    classification = "D"
elif any_path_contradicted or inside_80_ratio < 0.60:
    classification = "C"
elif market_result == "win" and inside_80_ratio >= 0.75:
    classification = "A"
else:
    classification = "B"
```

A defensive-error event qualifies for `D` only at three or more unearned runs. An event inside its 90% interval remains normal variance. If no market was recommended, return component metrics and `classification=None`.

- [ ] **Step 6: Run classification tests and verify GREEN**

Run: `python3 -m unittest tests.test_review_classification -v`

Expected: all classification tests PASS.

- [ ] **Step 7: Commit deterministic review classification**

```bash
git add analysis/review/__init__.py analysis/review/classification.py tests/test_review_classification.py
git commit -m "feat: classify forecast game flow"
```

### Task 2: Calibration, ROI, And Closing-Line Metrics

**Files:**
- Create: `analysis/review/calibration.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- `CaptureCutoffPolicy`
- `CalibrationRow`
- `CalibrationGroup`
- `select_primary_rows(store, cutoff_policy)`
- `build_calibration_report(rows, group_by)`

Use these concrete report inputs:

```python
@dataclass(frozen=True)
class CaptureCutoffPolicy:
    name: str
    minimum_seconds_before_first_pitch: int = 0
    required_lineup_state: str | None = None
    include_later_superseded_captures: bool = False


LATEST_PREGAME = CaptureCutoffPolicy(name="latest_pregame")

PROBABILITY_BUCKETS = (
    (0.00, 0.45), (0.45, 0.50), (0.50, 0.55),
    (0.55, 0.60), (0.60, 1.0000001),
)
PRIORITY_SCORE_BUCKETS = ((0, 69), (70, 74), (75, 79), (80, 84), (85, 100))


@dataclass(frozen=True)
class CalibrationRow:
    game_pk: int
    capture_id: str
    forecast_run_id: str
    market_key: str
    model_version: str
    market_type: str
    lineup_state: str
    starter_structure: str
    legacy_priority_score: int | None
    p_win: float
    p_push: float
    p_loss: float
    market_result: str
    selected_decimal_price: float | None
    closing_selected_decimal_price: float | None
    entry_no_vig_probability: float | None
    closing_no_vig_probability: float | None


@dataclass(frozen=True)
class CalibrationMetrics:
    sample_size: int
    wins: int
    losses: int
    pushes: int
    hit_rate: float | None
    binary_brier: float | None
    multiclass_brier: float | None
    log_loss: float | None
    profit_units: float | None
    roi: float | None
    mean_probability_clv: float | None
    mean_price_clv: float | None


@dataclass(frozen=True)
class CalibrationGroup:
    dimensions: tuple[tuple[str, str], ...]
    metrics: CalibrationMetrics
```

- [ ] **Step 1: Write failing deduplication and metric tests**

```python
class CalibrationTests(unittest.TestCase):
    def test_unchanged_repeated_capture_counts_once(self):
        rows = select_primary_rows(ledger_fixture(two_captures_same_forecast=True))
        self.assertEqual(len(rows), 1)

    def test_latest_changed_pregame_capture_is_primary(self):
        rows = select_primary_rows(ledger_fixture(lineup_update=True))
        self.assertEqual(rows[0].lineup_state, "confirmed")
        self.assertEqual(rows[0].forecast_run_id, "confirmed-run")

    def test_named_time_slice_can_evaluate_capture_later_superseded(self):
        policy = CaptureCutoffPolicy(
            name="projected_lineup_6h_before",
            minimum_seconds_before_first_pitch=21_600,
            required_lineup_state="projected",
            include_later_superseded_captures=True,
        )
        rows = select_primary_rows(ledger_fixture(lineup_update=True), policy)
        self.assertEqual(rows[0].lineup_state, "projected")

    def test_probability_and_priority_bucket_boundaries_are_fixed(self):
        self.assertEqual(probability_bucket(0.45), "0.45-0.50")
        self.assertEqual(priority_score_bucket(80), "80-84")

    def test_push_is_excluded_from_binary_brier_and_zero_for_roi(self):
        report = build_calibration_report((push_row(),), group_by=("market_type",))
        self.assertEqual(report.groups[0].metrics.pushes, 1)
        self.assertIsNone(report.groups[0].metrics.binary_brier)
        self.assertEqual(report.groups[0].metrics.profit_units, 0.0)

    def test_binary_brier_and_log_loss_use_conditional_probability(self):
        report = build_calibration_report((win_row(p_win=.45, p_push=.10),), group_by=())
        self.assertAlmostEqual(report.overall.binary_brier, (0.50 - 1.0) ** 2)
        self.assertAlmostEqual(report.overall.log_loss, -math.log(0.50))

    def test_probability_clv_uses_selected_direction(self):
        row = win_row(entry_no_vig=.52, closing_no_vig=.56)
        report = build_calibration_report((row,), group_by=())
        self.assertAlmostEqual(report.overall.mean_probability_clv, .04)

    def test_price_clv_is_positive_when_entry_price_beats_close(self):
        row = win_row(entry_price=2.00, closing_price=1.80)
        report = build_calibration_report((row,), group_by=())
        self.assertAlmostEqual(report.overall.mean_price_clv, (2.00 / 1.80) - 1)

    def test_missing_price_or_close_omits_roi_or_clv(self):
        report = build_calibration_report((row_without_prices(),), group_by=())
        self.assertIsNone(report.overall.roi)
        self.assertIsNone(report.overall.mean_probability_clv)
```

- [ ] **Step 2: Run calibration tests and verify RED**

Run: `python3 -m unittest tests.test_calibration -v`

Expected: FAIL because calibration reporting does not exist.

- [ ] **Step 3: Implement primary row selection**

Use ledger relationships, never list position. `analysis.review.calibration` is the sole owner of primary capture selection; the ledger exposes only relationship and supersession queries:

1. Reject captures at or after first pitch and captures with `out_of_sample_eligible=False`.
2. A capture satisfies a policy when `captured_at <= first_pitch - minimum_seconds_before_first_pitch` and its lineup state matches the optional requirement.
3. For `include_later_superseded_captures=False`, resolve supersession links and select the greatest `captured_at` among eligible non-superseded captures per game. For a named time-slice policy with the flag true, allow a capture later superseded after that slice and select the greatest eligible timestamp at or before the slice boundary. Resolve exact timestamp ties by lexicographically smallest `capture_id` for determinism.
4. Join its capture-forecast link and latest non-superseded official result revision.
5. Deduplicate by `(forecast_run_id, market_key)` after primary capture selection.
6. Permit earlier captures only under a separately named policy such as `CaptureCutoffPolicy("projected_lineup_6h_before", 21_600, "projected", True)`; never mix its rows into `LATEST_PREGAME`.

- [ ] **Step 4: Implement metric formulas**

For non-push binary outcomes:

```python
conditional_p = p_win / (1 - p_push)
brier = (conditional_p - outcome) ** 2
log_loss = -(outcome * log(p) + (1 - outcome) * log(1 - p))
```

Clip log-loss probability to `[1e-15, 1 - 1e-15]`. Multiclass Brier is the sum of squared error across win/push/loss. Profit for one unit is `price - 1` on win, `-1` on loss, and `0` on push. ROI is total profit divided by priced stakes only.

Probability CLV is `closing_no_vig_probability - entry_no_vig_probability` in the selected direction. Omit CLV when a matching close observation is unavailable.

Price CLV is `entry_selected_decimal_price / closing_selected_decimal_price - 1`; positive means the captured entry price beat the close. Retain both raw prices in each calibration row for audit.

- [ ] **Step 5: Add fixed grouping dimensions**

Support model version, market type, lineup state, starter structure, model probability bucket, and legacy priority-score bucket. Use conditional non-push model probability for `PROBABILITY_BUCKETS`. Buckets are lower-bound inclusive and upper-bound exclusive, except the final bucket includes `1.0`; priority buckets include both integer endpoints. Every report returns sample size, wins, losses, pushes, hit rate, binary Brier, multiclass Brier, log loss, profit, ROI, probability CLV, and price CLV with explicit `None` for unavailable metrics.

- [ ] **Step 6: Run calibration tests and verify GREEN**

Run: `python3 -m unittest tests.test_calibration -v`

Expected: all calibration tests PASS.

- [ ] **Step 7: Commit calibration reporting**

```bash
git add analysis/review/calibration.py tests/test_calibration.py
git commit -m "feat: report forecast calibration"
```

### Task 3: One-Way Shadow Adapter And Ledger Recording

**Files:**
- Create: `analysis/shadow.py`
- Create: `tests/test_shadow.py`

**Interfaces:**
- `LegacyDecisionSnapshot`
- `ShadowComparison`
- `compare_shadow_forecast(legacy, forecast)`
- `record_shadow_run(store, game_input, legacy, forecast, idempotency_keys)`

- [ ] **Step 1: Write failing immutability and idempotency tests**

```python
class ShadowAdapterTests(unittest.TestCase):
    def test_shadow_adapter_cannot_replace_legacy_candidates(self):
        legacy = legacy_snapshot(score=78, rating="play")
        forecast = forecast_output(model_probability=.41)
        comparison = compare_shadow_forecast(legacy, forecast)
        self.assertEqual(comparison.legacy.candidates, legacy.candidates)
        self.assertEqual(comparison.authority, "legacy")
        self.assertFalse(hasattr(comparison, "replacement_candidates"))

    def test_forecast_probability_does_not_change_legacy_priority(self):
        candidate = PriorityMarketCandidate(
            market="ml",
            score=78,
            decision_rating="play",
            evidence_paths=frozenset({PriorityEvidencePath.STARTER_SUPPRESSION}),
        )
        comparison = compare_shadow_forecast(
            legacy_snapshot(candidates=(candidate,)),
            forecast_output(model_probability=.32),
        )
        self.assertIs(comparison.legacy.candidates[0], candidate)

    def test_recording_same_shadow_run_is_idempotent(self):
        first = record_shadow_run(self.store, game(), legacy(), forecast(), keys())
        second = record_shadow_run(self.store, game(), legacy(), forecast(), keys())
        self.assertEqual(first.capture.capture_id, second.capture.capture_id)
        self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)

    def test_second_unchanged_capture_links_to_existing_forecast(self):
        first = record_shadow_run(self.store, game(captured_at=dt(1)), legacy(), forecast(), keys("a"))
        second = record_shadow_run(self.store, game(captured_at=dt(2)), legacy(), forecast(), keys("b"))
        self.assertNotEqual(first.capture.capture_id, second.capture.capture_id)
        self.assertEqual(first.forecast.forecast_run_id, second.forecast.forecast_run_id)

    def test_missing_binding_is_explicitly_review_ineligible(self):
        comparison = compare_shadow_forecast(
            legacy_snapshot(review_bindings=()), forecast_output()
        )
        market = comparison.market_comparisons[0]
        self.assertFalse(market.review_eligible)
        self.assertEqual(market.review_ineligibility_flags, ("missing_pregame_binding",))
```

- [ ] **Step 2: Run shadow tests and verify RED**

Run: `python3 -m unittest tests.test_shadow -v`

Expected: FAIL because `analysis.shadow` does not exist.

- [ ] **Step 3: Implement a one-way legacy snapshot**

```python
@dataclass(frozen=True)
class CandidateReviewBinding:
    market_key: str
    legacy_candidate_index: int
    legacy_market: str
    principal_paths: tuple[PrincipalPathBinding, ...]


@dataclass(frozen=True)
class LegacyDecisionSnapshot:
    model_version: str
    candidates: tuple[PriorityMarketCandidate, ...]
    final_priority: tuple[str, ...]
    review_bindings: tuple[CandidateReviewBinding, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowMarketComparison:
    market_key: str
    legacy_candidate_index: int
    legacy_market: str
    legacy_score: int
    legacy_rating: str
    model_win_probability: float
    fair_decimal_price: float | None
    market_no_vig_probability: float | None
    edge: float | None
    principal_paths: tuple[PrincipalPathBinding, ...]
    review_eligible: bool
    review_ineligibility_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowComparison:
    authority: str
    legacy: LegacyDecisionSnapshot
    forecast_run_id: str
    market_comparisons: tuple[ShadowMarketComparison, ...]
```

Map each binding by `legacy_candidate_index`, then verify `legacy_market == legacy.candidates[index].market` and that `market_key` exists in forecast market output. Require exactly one binding for every legacy candidate intended for postgame classification, with two to four canonical principal paths and fixed component IDs present in `ForecastOutput.component_distributions`.

Always emit one `ShadowMarketComparison` per legacy candidate. A missing/invalid binding stores `principal_paths=()`, `review_eligible=False`, and one or more fixed flags from `missing_pregame_binding`, `candidate_market_mismatch`, `unknown_market_key`, `invalid_path_count`, or `missing_component_distribution`. A valid binding stores `review_eligible=True` and no flags. The adapter may not invent or repair a path after capture.

The adapter accepts decisions already produced by legacy functions. It must not import `FullGameContext`, build market contexts, call `recommend_full_game`, call `evaluate_priority_admission`, or expose a method that returns replacement candidates.

- [ ] **Step 4: Implement ordered idempotent recording**

`record_shadow_run` performs:

1. Append/replay `CaptureRecord` from the full capture payload.
2. Append/replay the capture-independent `ForecastRecord` using `forecast:{run_id}`.
3. Append/replay `CaptureForecastLink` using both IDs.
4. Store the copied legacy snapshot inside forecast comparison output, never inside the semantic input hash.
5. Return all three immutable records.

On failure, leave already appended immutable records intact; a retry completes missing links idempotently.

- [ ] **Step 5: Run shadow tests and verify GREEN**

Run: `python3 -m unittest tests.test_shadow -v`

Expected: all shadow tests PASS.

- [ ] **Step 6: Commit shadow integration**

```bash
git add analysis/shadow.py tests/test_shadow.py
git commit -m "feat: record shadow forecast comparisons"
```

### Task 4: End-To-End Postgame Review Service

**Files:**
- Create: `analysis/review/service.py`
- Modify: `analysis/review/__init__.py`
- Create: `tests/test_review_service.py`

**Interfaces:**
- `build_review_candidate(store, link_record_id, result_record_id, market_key)`
- `build_and_record_review(store, link_record_id, result_record_id, market_key)`

- [ ] **Step 1: Write failing join, settlement, and persistence tests**

```python
class ReviewServiceTests(unittest.TestCase):
    def test_service_joins_exact_link_forecast_result_and_market(self):
        fixture = recorded_shadow_fixture()
        result = build_and_record_review(
            fixture.store,
            fixture.link.envelope.record_id,
            fixture.official_result.envelope.record_id,
            "sea_ml",
        )
        self.assertEqual(result.record.capture_id, fixture.capture.capture_id)
        self.assertEqual(result.record.forecast_run_id, fixture.forecast.forecast_run_id)
        self.assertEqual(result.record.official_result_record_id, fixture.official_result.envelope.record_id)

    def test_service_settles_market_from_captured_semantics_and_official_score(self):
        result = build_and_record_review_from_fixture(
            market=plus_handicap("SEA", 1.5),
            official_score=(3, 4),
        )
        self.assertEqual(result.record.market_result, "win")

    def test_service_uses_frozen_component_intervals(self):
        fixture = recorded_shadow_fixture()
        candidate = build_review_candidate(
            fixture.store, fixture.link_id, fixture.result_id, "sea_ml"
        )
        expected = fixture.forecast.outputs["component_distributions"]["away_full_runs"]
        self.assertEqual(candidate.component_forecasts[0].interval.p90, expected["summary"]["p90"])

    def test_same_review_retry_returns_one_record(self):
        first = build_and_record_review_from_fixture()
        second = build_and_record_review_from_fixture()
        self.assertEqual(first.record.review_id, second.record.review_id)
        self.assertEqual(len(tuple(first.store.iter_records("review"))), 1)

    def test_missing_pregame_path_binding_is_not_reconstructed(self):
        fixture = recorded_shadow_fixture(review_bindings=())
        with self.assertRaisesRegex(ValueError, "pregame principal-path binding"):
            build_review_candidate(fixture.store, fixture.link_id, fixture.result_id, "sea_ml")
```

- [ ] **Step 2: Run review-service tests and verify RED**

Run: `python3 -m unittest tests.test_review_service -v`

Expected: FAIL because `analysis.review.service` does not exist.

- [ ] **Step 3: Implement observed market settlement**

Add a pure `settle_observed_market(market_input, official_result)` helper in `analysis/forecasting/market.py` using the same F5/full-game/tie/push semantics as simulated settlement. The review service must use the semantic market stored in the forecast record and official score fields; it cannot use a narrative result label.

- [ ] **Step 4: Build a review candidate only from linked frozen records**

The service must:

1. Resolve `CaptureForecastLink`, `CaptureRecord`, `ForecastRecord`, and latest selected `OfficialResultRecord` by ID and reject mismatched game IDs.
2. Resolve the selected team to `away` or `home` from captured game identity. All component bindings and official component values use that side namespace; reject team-abbreviation component IDs.
3. Locate the exact `ShadowMarketComparison` by market key. If `review_eligible=False`, stop with its stored ineligibility flags; otherwise use its frozen principal paths. Never search for or synthesize a replacement binding.
4. Convert each referenced `DistributionSummary` directly to `PredictiveInterval`; never recompute forecast intervals from official outcomes.
5. Read matching official values from `OfficialResultRecord.component_values` and fail on a missing required component.
6. Convert only documented result events with source observation IDs into `ExogenousEvent` records.
7. Settle the market, call `classify_review`, and produce one complete `ReviewRecord` command.

- [ ] **Step 5: Append the review idempotently**

Use `review:{capture_id}:{forecast_run_id}:{official_result_record_id}:{market_key}`. An official result correction produces a new review linked to the new result revision; the prior review remains immutable and is excluded when its result is superseded.

- [ ] **Step 6: Run service, classification, and ledger tests and verify GREEN**

Run: `python3 -m unittest tests.test_review_service tests.test_review_classification tests.test_ledger -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit postgame review orchestration**

```bash
git add analysis/review/service.py analysis/review/__init__.py analysis/forecasting/market.py tests/test_review_service.py tests/test_forecasting_market.py
git commit -m "feat: persist objective postgame reviews"
```

### Task 5: Configuration And Manifest Safety Boundary

**Files:**
- Modify: `config/model.json`
- Modify: `analysis/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Existing top-level `version` remains the legacy fixed-model version.
- New `forecasting` block exposes shadow configuration separately.

- [ ] **Step 1: Write failing manifest and configuration tests**

```python
def test_manifest_declares_legacy_as_decision_authority(self):
    manifest = build_analysis_manifest(lineup_confirmed=True, starter_confirmed=True)
    self.assertEqual(manifest["decision_authority"], "legacy_fixed_model")
    self.assertEqual(manifest["forecasting"]["mode"], "shadow")
    self.assertFalse(manifest["forecasting"]["may_change_recommendations"])

def test_forecasting_has_separate_semantic_version(self):
    config = load_model_config()
    self.assertEqual(config["version"], "2026-07-13")
    self.assertEqual(config["forecasting"]["model_version"], "shadow-forecast-2026-07-14.1")

def test_shadow_metadata_does_not_change_required_outputs(self):
    manifest = build_analysis_manifest(lineup_confirmed=True, starter_confirmed=True)
    self.assertEqual(manifest["required_outputs"], ["F5", "F5 O/U", "ML", "+1.5", "O/U", "review"])
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `python3 -m unittest tests.test_pipeline -v`

Expected: FAIL because shadow metadata is absent.

- [ ] **Step 3: Add a separate immutable forecasting configuration block**

Add without changing existing thresholds:

```json
"forecasting": {
  "enabled": true,
  "mode": "shadow",
  "model_version": "shadow-forecast-2026-07-14.1",
  "may_change_recommendations": false,
  "simulations": 20000,
  "league_runs_per_team_game": 4.5,
  "league_woba": 0.320,
  "gamma_poisson_shape": 1.6,
  "extra_inning_multiplier": 1.8,
  "max_innings": 30,
  "offense_bounds": [0.65, 1.45],
  "matchup_bounds": [0.80, 1.20],
  "starter_ra9_bounds": [0.55, 1.75],
  "bullpen_ra9_bounds": [0.60, 1.90],
  "pitch_limit_high_bucket_transfer": 1.0,
  "pitch_limit_transfer_to_under_five": 0.4,
  "sample_shortage_log_se_penalty": 0.12,
  "unknown_bullpen_matchup_log_se_penalty": 0.08,
  "handoff_uncertainty_log_se_penalty": 0.10,
  "injury_return_log_se_penalty": 0.12,
  "recent_short_exit_log_se_penalty": 0.10,
  "limited_reliever_log_se_penalty": 0.10,
  "unscheduled_f5_relief_log_standard_error": 0.15,
  "pa_weights": [1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87],
  "confirmed_lineup_validation_games": 100,
  "market_family_tuning_decisions": 300
}
```

Expose the block in the manifest and add `decision_authority="legacy_fixed_model"`. Do not add the shadow probability to legacy `strength_adjustments` or required picks.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `python3 -m unittest tests.test_pipeline -v`

Expected: all pipeline tests PASS.

- [ ] **Step 5: Commit the safety boundary**

```bash
git add config/model.json analysis/pipeline.py tests/test_pipeline.py
git commit -m "feat: expose shadow forecasting mode"
```

### Task 6: Operator Documentation And Output Semantics

**Files:**
- Modify: `README.md`
- Modify: `docs/model.md`
- Modify: `docs/data-sources.md`
- Modify: `docs/analysis-output-template.md`

- [ ] **Step 1: Document the three-stage daily workflow**

Document:

1. Projected-lineup legacy analysis plus shadow capture.
2. Confirmed-lineup refresh creating a new capture/run only when semantic state changes.
3. Official-result ingestion, objective flow review, and calibration report.

State that source collection remains manual/external in this delivery and every normalized value needs provenance.

- [ ] **Step 2: Document output distinctions**

Add a written-review block, not a replacement pick column:

```markdown
Shadow probability: 54.2%
Fair decimal price: 1.84
Market no-vig probability: 51.0%
Shadow edge: +3.2 percentage points
Legacy authority: play/lean/pass and final priority remain unchanged
```

Explicitly state that the existing 0-100 value is a legacy rank score, not win probability.

- [ ] **Step 3: Document source and leakage requirements**

For every source observation document stable record/URL, `observed_at`, `available_at`, `fetched_at`, effective date, and content hash. Mark observations without defensible availability as retrospective-only.

- [ ] **Step 4: Document migration gates**

No production influence before at least 100 confirmed-lineup games. Do not tune a market family before 300 decisions when feasible. Any production precedence, rule deletion, or reweighting requires a new approved model version and design.

- [ ] **Step 5: Check documentation formatting and commit**

Run: `git diff --check`

Expected: no whitespace errors.

```bash
git add README.md docs/model.md docs/data-sources.md docs/analysis-output-template.md
git commit -m "docs: explain shadow forecast workflow"
```

### Task 7: Full System Verification

**Files:**
- Verify all implementation and documentation from all three plans.

- [ ] **Step 1: Run all new focused suites**

Run:

```bash
python3 -m unittest \
  tests.test_forecasting_inputs \
  tests.test_ledger \
  tests.test_forecasting_components \
  tests.test_forecasting_simulation \
  tests.test_forecasting_market \
  tests.test_forecasting_engine \
  tests.test_review_classification \
  tests.test_review_service \
  tests.test_calibration \
  tests.test_shadow \
  -v
```

Expected: all new tests PASS.

- [ ] **Step 2: Run the complete regression suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all existing 141 tests and all new tests PASS.

- [ ] **Step 3: Inspect the fixed and shadow manifest**

Run: `python3 -m analysis.pipeline --lineup confirmed`

Expected:

- `model_version` remains `2026-07-13`.
- `decision_authority` is `legacy_fixed_model`.
- `forecasting.mode` is `shadow`.
- Weather remains excluded.
- Required outputs remain F5, F5 O/U, ML, `+1.5`, O/U, and review.

- [ ] **Step 4: Verify deterministic replay and sample deduplication**

Run: `python3 -m unittest tests.test_shadow.ShadowAdapterTests.test_second_unchanged_capture_links_to_existing_forecast tests.test_calibration.CalibrationTests.test_unchanged_repeated_capture_counts_once tests.test_review_service.ReviewServiceTests.test_same_review_retry_returns_one_record -v`

Expected: all three tests PASS, proving two captures can share one forecast, one review replay stays one row, and the headline calibration sample remains one.

- [ ] **Step 5: Verify formatting and scope**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git diff --stat`

Expected: new work is isolated to forecasting/ledger/review/shadow modules, their tests, configuration metadata, and approved documentation. Legacy scoring changes are absent.

## Completion Gate

- Every reviewed market traces to an immutable pregame capture and deterministic forecast record.
- A/B/C/D classification is exhaustive, numeric, and independent of narrative hindsight.
- Repeated captures and retries cannot inflate hit rate, ROI, Brier score, or CLV samples.
- The shadow probability cannot mutate, replace, upgrade, downgrade, or reorder legacy picks.
- Calibration can separate correct game-flow reads from correct ticket outcomes.
- Production migration remains impossible without a separately approved change.
