# Review Evidence Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep incomplete playable ML and O9.0+ theses out of priority and reject fresh captures that omit the corresponding frozen review evidence.

**Architecture:** Add one deterministic evidence-completeness evaluator that derives starter and per-team scoring coverage from canonical principal-path bindings. Reuse it in priority admission and fresh-capture validation, advance only the legacy model version, and preserve exact historical replay before all new-version checks.

**Tech Stack:** Python 3.13, frozen dataclasses, Pydantic v2, pytest/unittest, append-only JSONL ledger contracts.

---

## File Map

- Modify `analysis/model.py`: cap incomplete candidates without adding points.
- Modify `analysis/review/classification.py`: expose canonical scoring-side derivation for validated bindings.
- Create `analysis/review/evidence_contract.py`: derive one immutable evidence
  assessment from a candidate, matching binding, and verified market input.
- Modify `analysis/shadow.py`: enforce fresh recommended ML and O9.0+ binding completeness.
- Modify `analysis/pipeline.py` and `config/model.json`: advance and expose the fixed legacy contract.
- Modify `docs/model.md`: document the new admission and review rules.
- Modify `tests/test_model.py`: TDD coverage for score caps.
- Modify `tests/test_review_classification.py`: TDD coverage for scoring-side derivation.
- Create `tests/test_review_evidence_contract.py`: focused contract and cap-input coverage.
- Modify `tests/test_shadow.py`: unit coverage for fresh snapshot validation.
- Modify `tests/test_operator_pregame.py`: TDD coverage for fresh rejection and exact historical replay.
- Modify `tests/test_pipeline.py`: version and manifest/config contract coverage.

### Task 1: Canonical Evidence Completeness

**Files:**
- Modify: `analysis/review/classification.py`
- Test: `tests/test_review_classification.py`

- [ ] **Step 1: Write failing tests for scoring-side derivation**

Add tests proving that conversion components identify their batting side,
while starter-collapse and late-bullpen-exposure components identify the
opposite scoring side. Reject suppression/protection paths as over routes.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_review_classification.py -k scoring_side -q`

Expected: failures because the public canonical helper does not exist.

- [ ] **Step 3: Implement the minimal helper**

Add a typed helper that accepts one already validated
`PrincipalPathBinding` and returns `"away"`, `"home"`, or `None` for an over
scoring direction. Reuse the existing exact binding shapes; do not parse
free-form notes.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest tests/test_review_classification.py -k scoring_side -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/review/classification.py tests/test_review_classification.py
git commit -m "feat: derive canonical scoring-side evidence"
```

### Task 2: Priority Admission Caps

**Files:**
- Create: `analysis/review/evidence_contract.py`
- Modify: `analysis/model.py`
- Create: `tests/test_review_evidence_contract.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing cap tests**

Add cases proving:

- playable ML without `starter_suppression` or `starter_collapse` is capped at
  `71`;
- `starter_suppression` must bind the selected ML team's pitching side, while
  `starter_collapse` must bind the opposing pitching side;
- complete ML keeps its score;
- playable O9.0+ without opposite-side scoring coverage is capped at `69`;
- complete O9.0+ keeps its score;
- under, totals below 9.0, missing/unverified total lines, F5 totals even at
  9.0, and non-playable candidates are unchanged by the new contracts.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_model.py -k evidence_contract -q`

Expected: failures because evidence-aware admission is not implemented.

- [ ] **Step 3: Implement minimal evidence-aware admission**

Add `assess_priority_evidence_contract(candidate, binding, market_input)` in
`analysis/review/evidence_contract.py`. It validates the candidate/binding
identity, uses `MarketInput.market_type`, `selected_side`, `line`, and the
canonical component-side helper, and returns an immutable assessment with the
derived cap and reason codes. Add evidence-completeness fields to
`PriorityAdmissionContext` as one optional immutable assessment input defaulted
to `None`; existing callers with no assessment preserve their exact behavior.
Only the bound-candidate admission path supplies the assessment.
`evaluate_priority_admission` applies `min(score, 71)` for incomplete ML and
`min(score, 69)` only when the assessment identifies a
`MarketType.FULL_TOTAL` over with verified line `>= 9.0`. F5 totals cannot
activate this cap. Fresh-capture validation independently recomputes the
assessment, so caller-provided admission values cannot bypass the contract.

- [ ] **Step 4: Run focused and existing admission tests**

Run: `python -m pytest tests/test_model.py -k 'evidence_contract or PriorityAdmission' -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/model.py tests/test_model.py
git commit -m "feat: cap incomplete priority evidence"
```

### Task 3: Fresh Capture Contract

**Files:**
- Modify: `analysis/shadow.py`
- Test: `tests/test_shadow.py`
- Test: `tests/test_operator_pregame.py`

- [ ] **Step 1: Write failing fresh-capture tests**

Add unit tests that a new-version playable ML without a directionally valid
starter binding and an O9.0+ without both scoring directions are rejected.
Cover duplicate canonical paths, two routes covering the same scoring side,
wrong F5 component horizons, valid conversion routes for opposite teams, and
valid mixes using `starter_collapse` or `late_bullpen_exposure` for the
opponent's scoring direction. Add operator tests proving rejection occurs
before forecast or ledger creation.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_shadow.py tests/test_operator_pregame.py -k 'starter_evidence or high_total_two_sided' -q`

Expected: incomplete fresh captures are incorrectly accepted before the fix.

- [ ] **Step 3: Implement strict validation**

In `validate_recommended_review_contract`, call the shared assessment for each
candidate, matching binding, and verified market input. For the new model
version, apply this contract only when `score >= 72` and `decision_rating` is
`play` or `lean`, then reject a candidate whose submitted score exceeds its
derived evidence cap. `pass` and below-playable candidates remain outside this
contract. Report deterministic error codes and never synthesize missing paths.
Preserve the existing Shadow comparison as non-authoritative.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_shadow.py tests/test_operator_pregame.py -k 'starter_evidence or high_total_two_sided or exact_replay or stale_fresh' -q`

Expected: all selected tests pass, exact old-version replay is resolved before
the new checks, a non-exact old-version submission is rejected, and all
rejection paths leave zero forecast/ledger artifacts.

- [ ] **Step 5: Commit**

```bash
git add analysis/shadow.py tests/test_shadow.py tests/test_operator_pregame.py
git commit -m "feat: enforce frozen review evidence contracts"
```

### Task 4: Version, Configuration, And Documentation

**Files:**
- Modify: `analysis/pipeline.py`
- Modify: `config/model.json`
- Modify: `docs/model.md`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing version and manifest tests**

Expect `2026-08-05` and the two new caps/contracts in the manifest. Update the
legacy configuration checksum only after independently verifying the intended
diff.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: version/config assertions fail.

- [ ] **Step 3: Apply minimal version/config/doc changes**

Advance the fixed legacy version, expose `ml_starter_evidence_cap=71` and
`high_total_two_sided_over_cap=69`, and document that the rules are caps and
fresh-capture contracts only. Keep Shadow and forecasting configuration bytes
unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/pipeline.py config/model.json docs/model.md tests/test_pipeline.py
git commit -m "docs: publish review evidence model contract"
```

### Task 5: Regression And Full Verification

**Files:**
- Verify only unless a regression requires a focused fix.

- [ ] **Step 1: Run the directly affected suites**

Run: `python -m pytest tests/test_model.py tests/test_review_classification.py tests/test_review_evidence_contract.py tests/test_shadow.py tests/test_operator_pregame.py tests/test_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 3: Inspect scope and whitespace**

Run: `git diff --check && git status --short && git diff main...HEAD --stat`

Expected: only the approved `analysis/model.py`,
`analysis/review/classification.py`, new
`analysis/review/evidence_contract.py`, `analysis/shadow.py`, pipeline/config,
`docs/model.md`, spec/plan, and corresponding test changes appear.

- [ ] **Step 4: Run the fixed-model manifest**

Run: `python -m analysis.pipeline --lineup confirmed --starter confirmed`

Expected: model version `2026-08-05`, unchanged Shadow authority, and the new
priority-admission contract are present.
