# ML Bullpen Review Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require every fresh recommended full-game ML capture to freeze an auditable bullpen principal path without changing prediction outputs or historical replay.

**Architecture:** Add the rule to the core recommended-review contract and add a fresh-capture model-version guard in the Operator service after exact-replay resolution. Bump the fixed model version and document the behavior while preserving old immutable records.

**Tech Stack:** Python 3.14, dataclasses, pytest, JSON configuration, append-only JSONL ledger.

## Global Constraints

- Do not change legacy scores, market admission, Shadow probabilities, or final priorities.
- Do not mutate historical ledger records.
- Preserve exact replay before applying the new fresh-capture gate.
- Use the existing canonical `bullpen_protection` and `late_bullpen_exposure` paths only.

---

### Task 1: Contract Regression Tests

**Files:**
- Modify: `tests/test_shadow.py`
- Modify: `tests/test_operator_pregame.py`

**Interfaces:**
- Consumes: `validate_recommended_review_contract`, `run_projected`.
- Produces: regression coverage for ML bullpen binding and historical replay.

- [ ] **Step 1: Write failing core-contract tests**

Add tests proving that a current-version recommended ML without a bullpen path,
or with the path omitted from its binding, raises a deterministic validation
error; add passing cases for both canonical bullpen paths.

- [ ] **Step 2: Verify the core tests fail**

Run: `.venv/bin/pytest tests/test_shadow.py -k 'ml and bullpen and contract' -q`

Expected: the missing-path cases fail because the current validator accepts
them.

- [ ] **Step 3: Write failing Operator tests**

Add a fresh stale-version rejection test that asserts no ledger or bundle
artifact is written, plus an exact historical replay test that remains valid.

- [ ] **Step 4: Verify the Operator tests fail**

Run: `.venv/bin/pytest tests/test_operator_pregame.py -k 'legacy_model_version or ml_bullpen' -q`

Expected: the stale fresh bundle is accepted before implementation.

### Task 2: Minimal Contract And Version Implementation

**Files:**
- Modify: `analysis/shadow.py`
- Modify: `analysis/operator/service.py`
- Modify: `analysis/pipeline.py`
- Modify: `config/model.json`

**Interfaces:**
- Consumes: `LegacyDecisionSnapshot`, `PriorityEvidencePath`, Operator exact-replay flow.
- Produces: current-version ML bullpen validation and fresh-capture version enforcement.

- [ ] **Step 1: Implement ML bullpen-path validation**

For current-version `play`/`lean` ML candidates, require at least one bullpen
path in the candidate evidence set and the matching principal-path binding.

- [ ] **Step 2: Implement fresh-capture version enforcement**

After the exact-replay fast path, require
`bundle.payload.legacy_decision.model_version == LEGACY_MODEL_VERSION` and raise
`OperatorPreconditionError` before simulation or ledger append on mismatch.

- [ ] **Step 3: Advance the fixed model version**

Set `LEGACY_MODEL_VERSION` and `config/model.json` to `2026-08-02`; add
`priority_admission.require_ml_bullpen_review_path: true`.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest tests/test_shadow.py tests/test_operator_pregame.py tests/test_pipeline.py -q`

Expected: all focused tests pass.

### Task 3: Documentation And Full Verification

**Files:**
- Modify: `docs/model.md`
- Modify: `docs/analysis-output-template.md`
- Modify: `docs/shadow-operator.md`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: the implemented contract and model-version metadata.
- Produces: authoritative operating guidance and configuration regression coverage.

- [ ] **Step 1: Document the new rule**

State that every fresh recommended ML must freeze `bullpen_protection` or
`late_bullpen_exposure`, and that stale versions are allowed only through exact
historical replay.

- [ ] **Step 2: Update configuration expectations**

Update the expected fixed version and legacy configuration hash in the pipeline
tests.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 4: Inspect final scope**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only the intended model, validation, tests, and
documentation files are tracked changes. Existing untracked runtime data stays
unstaged.
