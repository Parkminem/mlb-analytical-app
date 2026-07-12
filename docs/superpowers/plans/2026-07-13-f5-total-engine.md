# F5 Total Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently evaluated first-five-inning totals to the fixed MLB model, default table, and final priority competition.

**Architecture:** Add a dedicated `F5TotalContext` and `evaluate_f5_total` function that cannot receive full-game bullpen inputs. Add a separate total-market selection helper that suppresses correlated F5 and full-game totals unless independent paths are explicitly verified. Extend the manifest, configuration, and documentation without changing existing full-game O/U behavior.

**Tech Stack:** Python 3 dataclasses, `unittest`, JSON configuration, Markdown documentation

## Global Constraints

- Score each baseball input once and reuse it with F5-total-specific weights; do not double count evidence.
- Do not use full-game bullpen availability or quality in F5 O/U unless a verified opener/bulk pitcher is scheduled inside the first five innings.
- Do not infer an F5 line from the full-game total.
- Use decimal odds and pass when the current F5 total or decimal price is unavailable.
- Keep weather excluded unless the user provides it or explicitly requests it.
- Keep existing full-game O/U behavior unchanged.

---

### Task 1: Independent F5 Total Evaluator

**Files:**
- Modify: `analysis/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `F5TotalContext` containing market data, first-five starter stability/risk, early lineup paths, conversion, and opener/bulk verification.
- Produces: `evaluate_f5_total(context: F5TotalContext) -> TotalDecision` with `rating` equal to `over`, `under`, or `pass`.

- [ ] **Step 1: Write failing evaluator tests**

Add tests that import `F5TotalContext` and `evaluate_f5_total` and verify:

```python
def test_f5_total_passes_without_verified_market_line_or_price(self):
    result = evaluate_f5_total(F5TotalContext(
        market_total=None,
        decimal_odds=None,
        projected_total=4.8,
        starter_stability_a=45,
        starter_stability_b=45,
    ))
    self.assertEqual(result.rating, "pass")
    self.assertIn("F5 market total and decimal price are required", result.notes)

def test_bullpen_only_volatility_cannot_create_f5_over(self):
    result = evaluate_f5_total(F5TotalContext(
        market_total=4.5,
        decimal_odds=1.91,
        projected_total=4.6,
        starter_stability_a=72,
        starter_stability_b=74,
        both_starters_five_inning_floor=True,
    ))
    self.assertEqual(result.rating, "pass")

def test_stable_starters_and_muted_lineups_support_f5_under(self):
    result = evaluate_f5_total(F5TotalContext(
        market_total=4.5,
        decimal_odds=1.91,
        projected_total=3.9,
        starter_stability_a=78,
        starter_stability_b=76,
        lineup_muted_a=True,
        lineup_muted_b=True,
        both_starters_five_inning_floor=True,
    ))
    self.assertEqual(result.rating, "under")

def test_starter_risk_and_converted_early_path_support_f5_over(self):
    result = evaluate_f5_total(F5TotalContext(
        market_total=4.5,
        decimal_odds=1.91,
        projected_total=5.0,
        starter_stability_a=42,
        starter_stability_b=62,
        starter_command_or_hr_risk_a=True,
        lineup_early_scoring_path_b=True,
        recent_power_or_walk_conversion_b=True,
    ))
    self.assertEqual(result.rating, "over")

def test_unverified_opener_bulk_chain_blocks_f5_total(self):
    result = evaluate_f5_total(F5TotalContext(
        market_total=4.5,
        decimal_odds=1.91,
        projected_total=5.1,
        starter_stability_a=40,
        starter_stability_b=55,
        opener_bulk_structure=True,
        first_five_pitching_chain_verified=False,
        lineup_early_scoring_path_a=True,
        recent_power_or_walk_conversion_a=True,
    ))
    self.assertEqual(result.rating, "pass")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_model.F5TotalTests -v`

Expected: FAIL because `F5TotalContext` and `evaluate_f5_total` do not exist.

- [ ] **Step 3: Add the minimal F5 context and evaluator**

Add an immutable dataclass with no bullpen fields:

```python
@dataclass(frozen=True)
class F5TotalContext:
    market_total: float | None
    decimal_odds: float | None
    projected_total: float
    starter_stability_a: int
    starter_stability_b: int
    lineup_muted_a: bool = False
    lineup_muted_b: bool = False
    lineup_early_scoring_path_a: bool = False
    lineup_early_scoring_path_b: bool = False
    recent_power_or_walk_conversion_a: bool = False
    recent_power_or_walk_conversion_b: bool = False
    top_five_pitch_velocity_plus_a: int = 0
    top_five_pitch_velocity_plus_b: int = 0
    starter_command_or_hr_risk_a: bool = False
    starter_command_or_hr_risk_b: bool = False
    starter_first_inning_collapse_risk_a: bool = False
    starter_first_inning_collapse_risk_b: bool = False
    starter_injury_or_short_exit_risk_a: bool = False
    starter_injury_or_short_exit_risk_b: bool = False
    both_starters_five_inning_floor: bool = False
    opener_bulk_structure: bool = False
    first_five_pitching_chain_verified: bool = True
    pitch_velocity_unknown_rate_a: float = 0.0
    pitch_velocity_unknown_rate_b: float = 0.0
```

Implement `evaluate_f5_total` so it:

- Passes if market total or decimal price is missing.
- Passes an unverified opener/bulk first-five chain.
- Returns under only when the projected edge is at least `0.3`, both starters score at least `65`, both lineups are muted, risk flags are clear, and unknown pitch-velocity rate stays below `0.45`.
- Returns over only when projected total is at least `market_total - 0.1` and an early scoring path is paired with recent conversion or opposing starter risk.
- Requires two early scoring paths for market totals of `4.0` or lower.
- Passes a thin over when both starters have a stable five-inning floor and no starter risk exists.

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run: `python3 -m unittest tests.test_model.F5TotalTests -v`

Expected: all F5 total evaluator tests PASS.

### Task 2: Correlated Total Priority Selection

**Files:**
- Modify: `analysis/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `TotalMarketPriorityContext(f5_score, full_game_score, independent_paths)`.
- Produces: `select_total_priority_markets(...) -> TotalMarketPrioritySelection` with booleans for F5 and full-game inclusion.

- [ ] **Step 1: Write failing correlation tests**

```python
def test_correlated_totals_keep_only_higher_score(self):
    result = select_total_priority_markets(TotalMarketPriorityContext(
        f5_score=78,
        full_game_score=81,
        independent_paths=False,
    ))
    self.assertFalse(result.include_f5)
    self.assertTrue(result.include_full_game)

def test_independent_early_and_late_paths_allow_both_totals(self):
    result = select_total_priority_markets(TotalMarketPriorityContext(
        f5_score=76,
        full_game_score=79,
        independent_paths=True,
    ))
    self.assertTrue(result.include_f5)
    self.assertTrue(result.include_full_game)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_model.TotalMarketPriorityTests -v`

Expected: FAIL because the priority-selection interfaces do not exist.

- [ ] **Step 3: Implement minimal correlation selection**

Add immutable context and result dataclasses. When both scores exist and paths are not independent, include only the higher score; resolve equal scores in favor of F5 because it contains fewer game-state dependencies. When paths are independent, preserve both. Preserve a single available market unchanged.

- [ ] **Step 4: Run correlation tests and verify GREEN**

Run: `python3 -m unittest tests.test_model.TotalMarketPriorityTests -v`

Expected: all total-market priority tests PASS.

### Task 3: Manifest, Configuration, And Documentation

**Files:**
- Modify: `analysis/pipeline.py`
- Modify: `config/model.json`
- Modify: `tests/test_pipeline.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/model.md`
- Modify: `docs/analysis-output-template.md`
- Modify: `docs/new-chat-prompt.md`

**Interfaces:**
- Consumes: fixed model configuration and manifest builder.
- Produces: default required output list and table containing `F5 O/U`.

- [ ] **Step 1: Write failing manifest/config tests**

Add assertions that:

```python
self.assertIn("F5 O/U", manifest["required_outputs"])
self.assertTrue(config["f5_total_priority"]["independent_engine"])
self.assertFalse(config["f5_total_priority"]["use_full_game_bullpen_inputs"])
self.assertTrue(config["f5_total_priority"]["require_verified_market_line_and_price"])
self.assertTrue(config["f5_total_priority"]["deduplicate_correlated_full_game_total"])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_pipeline -v`

Expected: FAIL because the manifest and configuration do not contain F5 O/U.

- [ ] **Step 3: Update manifest, configuration, and docs**

Add `F5 O/U` to `required_outputs`. Add a `f5_total_priority` configuration section with the four tested booleans. Update the authoritative model and output template to use:

```markdown
| Game | F5 | F5 O/U | ML | +1.5 | O/U | Final |
```

Document that F5 totals exclude full-game bullpen effects, require verified market data, and compete in the same priority stack with correlated-total deduplication.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `python3 -m unittest tests.test_pipeline -v`

Expected: all pipeline tests PASS.

### Task 4: Full Verification

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: completed F5 total engine, priority selector, manifest, and documentation.
- Produces: a regression-safe fixed model update.

- [ ] **Step 1: Run the complete test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Inspect the fixed manifest**

Run: `python3 -m analysis.pipeline --lineup confirmed`

Expected: `required_outputs` includes `F5 O/U`, `include_weather` remains `false`, and no confidence flags appear for a confirmed lineup.

- [ ] **Step 3: Check formatting and unintended changes**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Review the focused diff**

Run: `git diff -- analysis/model.py analysis/pipeline.py config/model.json tests/test_model.py tests/test_pipeline.py AGENTS.md README.md docs/model.md docs/analysis-output-template.md docs/new-chat-prompt.md`

Expected: changes are limited to the approved F5 O/U feature and preserve all pre-existing user changes.
