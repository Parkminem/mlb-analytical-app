# Multi-Market Same-Game Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain one independently validated full-game side and one directionally compatible full-game total from the same game when each scores at least 72.

**Architecture:** Extend the existing legacy `PriorityMarketCandidate` with explicit full-game pair-coherence metadata, then change `assemble_same_game_priority` to filter invalid candidates, choose one winner per full-game market family, and run the existing canonical-evidence correlation checks. Keep F5 selection and Shadow authority unchanged, and expose the policy through fixed configuration, the pipeline manifest, and authoritative output documentation.

**Tech Stack:** Python 3.11+, frozen dataclasses, `unittest`, JSON configuration, Markdown model documentation.

## Global Constraints

- A market-engine `pass` never enters the priority stack.
- The playable multi-market threshold is exactly 72.
- At most one of `ml`, `plus_1_5`, or `minus_1_5` may remain from the side family.
- At most one `full_total` direction may remain from the total family.
- A side and a full-game total may coexist only when both carry explicit directional-coherence validation and have distinct non-empty canonical evidence paths.
- F5 side and F5 O/U keep their current independent evaluation and correlation rules.
- Shadow probability cannot select, remove, reorder, or break a tie among legacy candidates.
- Baseball inputs, market-engine scores, priority caps, and model version remain unchanged.

---

### Task 1: Implement family-aware same-game assembly

**Files:**
- Modify: `tests/test_model.py:90-240`
- Modify: `analysis/model.py:369-397`
- Modify: `analysis/model.py:555-610`

**Interfaces:**
- Consumes: existing `PriorityEvidencePath`, legacy candidate scores, and market-engine `decision_rating` values.
- Produces: `PriorityMarketCandidate(..., full_game_pair_compatible: bool = False)` and `assemble_same_game_priority(candidates, playable_min_score=72) -> SameGamePrioritySelection`.

- [ ] **Step 1: Write failing tests for compatible side and total retention**

Add these tests to `SameGamePriorityTests`:

```python
def test_compatible_atlanta_ml_and_over_both_enter_priority(self):
    result = model.assemble_same_game_priority(
        [
            model.PriorityMarketCandidate(
                "ml",
                77,
                "play",
                frozenset(
                    {
                        model.PriorityEvidencePath.TOP_ORDER_MATCHUP,
                        model.PriorityEvidencePath.BULLPEN_PROTECTION,
                    }
                ),
                full_game_pair_compatible=True,
            ),
            model.PriorityMarketCandidate(
                "full_total",
                83,
                "over",
                frozenset(
                    {
                        model.PriorityEvidencePath.STARTER_COLLAPSE,
                        model.PriorityEvidencePath.RECENT_RUN_CONVERSION,
                    }
                ),
                full_game_pair_compatible=True,
            ),
        ]
    )

    self.assertEqual(
        [(candidate.market, candidate.score) for candidate in result.included],
        [("full_total", 83), ("ml", 77)],
    )

def test_incompatible_side_and_total_keep_only_higher_score(self):
    result = model.assemble_same_game_priority(
        [
            model.PriorityMarketCandidate(
                "ml",
                77,
                "play",
                frozenset({model.PriorityEvidencePath.BULLPEN_PROTECTION}),
            ),
            model.PriorityMarketCandidate(
                "full_total",
                83,
                "over",
                frozenset({model.PriorityEvidencePath.STARTER_COLLAPSE}),
            ),
        ]
    )

    self.assertEqual([candidate.market for candidate in result.included], ["full_total"])
    self.assertEqual([candidate.market for candidate in result.excluded], ["ml"])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest \
  tests.test_model.SameGamePriorityTests.test_compatible_atlanta_ml_and_over_both_enter_priority \
  tests.test_model.SameGamePriorityTests.test_incompatible_side_and_total_keep_only_higher_score -v
```

Expected: FAIL because `PriorityMarketCandidate` does not accept `full_game_pair_compatible` and the assembler does not implement the new family policy.

- [ ] **Step 3: Add failing tests for family deduplication and admission threshold**

Add:

```python
def test_ml_and_handicap_keep_only_higher_scoring_side_family_candidate(self):
    result = model.assemble_same_game_priority(
        [
            model.PriorityMarketCandidate(
                "ml",
                77,
                "play",
                frozenset({model.PriorityEvidencePath.BULLPEN_PROTECTION}),
            ),
            model.PriorityMarketCandidate(
                "plus_1_5",
                80,
                "play",
                frozenset({model.PriorityEvidencePath.UNDERDOG_RUN_CREATION}),
            ),
        ]
    )

    self.assertEqual([candidate.market for candidate in result.included], ["plus_1_5"])
    self.assertEqual([candidate.market for candidate in result.excluded], ["ml"])

def test_over_and_under_keep_only_higher_scoring_total_family_candidate(self):
    result = model.assemble_same_game_priority(
        [
            model.PriorityMarketCandidate(
                "full_total",
                80,
                "under",
                frozenset({model.PriorityEvidencePath.STARTER_SUPPRESSION}),
            ),
            model.PriorityMarketCandidate(
                "full_total",
                83,
                "over",
                frozenset({model.PriorityEvidencePath.STARTER_COLLAPSE}),
            ),
        ]
    )

    self.assertEqual(
        [(candidate.decision_rating, candidate.score) for candidate in result.included],
        [("over", 83)],
    )

def test_below_72_and_missing_evidence_are_not_priority_eligible(self):
    result = model.assemble_same_game_priority(
        [
            model.PriorityMarketCandidate(
                "ml",
                71,
                "lean",
                frozenset({model.PriorityEvidencePath.BULLPEN_PROTECTION}),
            ),
            model.PriorityMarketCandidate("full_total", 80, "over"),
        ]
    )

    self.assertEqual(result.included, ())
    self.assertCountEqual(
        [candidate.market for candidate in result.excluded],
        ["ml", "full_total"],
    )
```

- [ ] **Step 4: Run the new family tests and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest tests.test_model.SameGamePriorityTests -v
```

Expected: FAIL on family deduplication, the 72-point threshold, and empty-evidence exclusion while existing pass and F5 tests continue to describe the old behavior.

- [ ] **Step 5: Implement the minimal family-aware assembler**

In `analysis/model.py`, extend the candidate without changing existing positional arguments:

```python
@dataclass(frozen=True)
class PriorityMarketCandidate:
    market: str
    score: int
    decision_rating: str
    evidence_paths: frozenset[PriorityEvidencePath] = field(default_factory=frozenset)
    full_game_pair_compatible: bool = False
```

Add fixed family constants and helpers near the assembler:

```python
PLAYABLE_PRIORITY_MIN_SCORE = 72
FULL_GAME_SIDE_MARKETS = frozenset({"ml", "plus_1_5", "minus_1_5"})
FULL_GAME_TOTAL_MARKETS = frozenset({"full_total"})


def _full_game_family(candidate: PriorityMarketCandidate) -> str | None:
    if candidate.market in FULL_GAME_SIDE_MARKETS:
        return "side"
    if candidate.market in FULL_GAME_TOTAL_MARKETS:
        return "total"
    return None
```

Update `assemble_same_game_priority` to:

```python
def assemble_same_game_priority(
    candidates: list[PriorityMarketCandidate],
    playable_min_score: int = PLAYABLE_PRIORITY_MIN_SCORE,
) -> SameGamePrioritySelection:
    # Validate canonical paths first.
    # Exclude pass, score < playable_min_score, and empty evidence.
    # Sort by score and the existing deterministic tie order. Shadow is not read.
    # Keep only the first eligible full-game side and first eligible full-game total.
    # Apply the existing disjoint-evidence check to every retained pair.
    # For a side-total pair, additionally require full_game_pair_compatible=True
    # on both candidates; otherwise retain the higher-scoring candidate only.
```

Keep the existing F5 tie order and canonical-path correlation behavior. Add `minus_1_5` immediately after `plus_1_5` in the deterministic tie order.

- [ ] **Step 6: Run model tests and verify GREEN**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest tests.test_model -v
```

Expected: all model tests PASS. Update the former `test_missing_evidence_cannot_retain_multiple_markets` assertion so the empty-evidence candidate is excluded even when it is the highest score, matching the approved specification.

- [ ] **Step 7: Commit the assembler behavior**

```bash
git add analysis/model.py tests/test_model.py
git commit -m "feat: allow compatible side and total priorities"
```

---

### Task 2: Expose and document the fixed priority policy

**Files:**
- Modify: `tests/test_pipeline.py:185-210`
- Modify: `config/model.json`
- Modify: `analysis/pipeline.py:105-125`
- Modify: `docs/model.md:492-512`
- Modify: `docs/analysis-output-template.md:5-30`
- Modify: `docs/superpowers/specs/2026-08-05-multi-market-priority-design.md:39-47`

**Interfaces:**
- Consumes: `priority_admission` configuration loaded by `load_model_config`.
- Produces: `build_analysis_manifest(...)["same_game_priority_policy"]` and authoritative user-facing rules consistent with Task 1.

- [ ] **Step 1: Write failing configuration and manifest assertions**

Add to `ModelConfigTests.test_default_config_preserves_fixed_source_and_weather_rules`:

```python
self.assertEqual(config["priority_admission"]["multi_market_min_score"], 72)
self.assertTrue(config["priority_admission"]["one_candidate_per_full_game_family"])
self.assertTrue(config["priority_admission"]["allow_compatible_side_total_pair"])
self.assertTrue(config["priority_admission"]["require_directional_coherence"])
self.assertTrue(config["priority_admission"]["f5_selection_rules_unchanged"])
```

Add to `AnalysisManifestTests`:

```python
def test_manifest_exposes_multi_market_same_game_policy(self):
    manifest = build_analysis_manifest(
        lineup_confirmed=True,
        starter_confirmed=True,
        include_weather=False,
    )

    policy = manifest["same_game_priority_policy"]
    self.assertEqual(policy["multi_market_min_score"], 72)
    self.assertTrue(policy["one_candidate_per_full_game_family"])
    self.assertTrue(policy["allow_compatible_side_total_pair"])
    self.assertTrue(policy["require_directional_coherence"])
    self.assertTrue(policy["f5_selection_rules_unchanged"])
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest tests.test_pipeline -v
```

Expected: FAIL with missing configuration keys and missing `same_game_priority_policy`.

- [ ] **Step 3: Add the approved configuration and manifest output**

Add these exact values to `config/model.json` under `priority_admission`:

```json
"multi_market_min_score": 72,
"one_candidate_per_full_game_family": true,
"allow_compatible_side_total_pair": true,
"require_directional_coherence": true,
"f5_selection_rules_unchanged": true
```

In `build_analysis_manifest`, expose a copy without creating a second source of truth:

```python
"same_game_priority_policy": deepcopy(model_config["priority_admission"]),
```

- [ ] **Step 4: Update authoritative documentation and output examples**

In `docs/model.md` and `docs/analysis-output-template.md`, replace the ambiguous one-choice language with these explicit rules:

```text
Select at most one full-game side-family market and at most one full-game total.
Retain both as separate priority entries when each is non-pass, scores at least
72, has non-empty canonical evidence, and the side-total game scripts are
explicitly validated as compatible. Show both in Final, for example ATL ML + O9.
```

State that global ranking is score-sorted, ML and a handicap never both enter by default, over and under never both enter, and F5 rules are unchanged.

Correct the approved design document's tie language to say that an exact legacy-score tie uses the deterministic market order. Shadow probability must not break the tie because Shadow cannot change or reorder legacy decisions.

- [ ] **Step 5: Run focused tests and documentation checks**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest \
  tests.test_model tests.test_pipeline -v
python3 -m analysis.pipeline --lineup confirmed
git diff --check
```

Expected: tests PASS, the manifest prints `same_game_priority_policy` with the five approved values, and `git diff --check` prints nothing.

- [ ] **Step 6: Run the full regression suite**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests PASS with no failures or errors.

- [ ] **Step 7: Commit policy exposure and documentation**

```bash
git add config/model.json analysis/pipeline.py tests/test_pipeline.py \
  docs/model.md docs/analysis-output-template.md \
  docs/superpowers/specs/2026-08-05-multi-market-priority-design.md
git commit -m "docs: expose multi-market priority policy"
```
