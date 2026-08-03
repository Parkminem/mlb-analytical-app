# Repeated Review Model Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the fixed model against the five repeated pregame reasoning failures and make under review bindings two-sided and auditable.

**Architecture:** Add negative-only gates to the existing F5-total and priority-admission contexts, then add one review-only canonical lineup-suppression path and a current-version under-binding contract. Version-gate every new contract so historical captures remain immutable and replayable.

**Tech Stack:** Python 3.14, frozen dataclasses, pytest, JSON configuration, append-only shadow ledger.

## Global Constraints

- Score each baseball input once and reuse it across markets.
- Add no positive scoring and no automatic market redirection.
- Preserve exact historical replay and the shadow forecast configuration.
- Apply the new under-binding contract only to fixed model version `2026-08-03`.

---

### Task 1: F5 And Side Admission Gates

**Files:**
- Modify: `tests/test_model.py`
- Modify: `analysis/model.py`

**Interfaces:**
- Consumes: `F5TotalContext`, `PriorityAdmissionContext`.
- Produces: deterministic negative-only F5-under and side-priority caps.

- [ ] **Step 1: Write failing F5 tests**

Add tests proving that missing two-sided verification and unresolved
long-horizon baseline risk each block an otherwise valid F5 under.

- [ ] **Step 2: Run the F5 tests and verify RED**

Run: `python -m pytest tests/test_model.py -k 'f5_total and (two_sided or long_horizon)' -q`

Expected: failure because the context fields and gates do not exist.

- [ ] **Step 3: Implement the minimal F5 gates**

Add required `two_sided_run_suppression_verified` plus per-starter baseline-risk
flags, and return `pass` before the existing under admission when either gate
fails.

- [ ] **Step 4: Write and run failing side-admission tests**

Test matchup-only traffic without conversion despite bullpen support, ML
without a full-game margin path, and availability-only or performance-weak
bullpen protection.

- [ ] **Step 5: Implement the minimal side caps**

Reuse the existing score caps: `74` for matchup-only conversion failure and
`71` for missing full-game margin or verified bullpen protection.

- [ ] **Step 6: Run focused model tests**

Run: `python -m pytest tests/test_model.py -q`

Expected: all model tests pass.

### Task 2: Two-Sided Under Review Contract

**Files:**
- Modify: `tests/test_review_classification.py`
- Modify: `tests/test_shadow.py`
- Modify: `analysis/model.py`
- Modify: `analysis/review/classification.py`
- Modify: `analysis/shadow.py`

**Interfaces:**
- Consumes: `PriorityEvidencePath`, `PrincipalPathBinding`, `MarketInput.selected_side`.
- Produces: canonical `lineup_suppression` review path and strict current-version under binding validation.

- [ ] **Step 1: Write failing path-shape and contradiction tests**

Require horizon-correct runs, traffic, and extra-base-hit components from one
side and classify an upper-tail lineup result as contradictory.

- [ ] **Step 2: Verify path tests fail**

Run: `python -m pytest tests/test_review_classification.py -k lineup_suppression -q`

Expected: failure because the path and rule do not exist.

- [ ] **Step 3: Implement the review-only path**

Add the canonical enum values, shape validator, and upper-tail contradiction
rule without changing forecast probabilities.

- [ ] **Step 4: Write failing recommended-under contract tests**

For version `2026-08-03`, reject a recommended under missing either canonical
path or using opposite namespaces; accept matching starter and lineup
namespaces. Keep old-version behavior unchanged.

- [ ] **Step 5: Implement the current-version under contract**

Read direction from the bound `MarketInput`, validate both paths and namespace
coverage, and return deterministic contract errors before capture.

- [ ] **Step 6: Run focused review and shadow tests**

Run: `python -m pytest tests/test_review_classification.py tests/test_shadow.py -q`

Expected: all focused tests pass.

### Task 3: Version, Configuration, And Documentation

**Files:**
- Modify: `analysis/pipeline.py`
- Modify: `config/model.json`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_operator_pregame.py`
- Modify: `docs/model.md`
- Modify: `docs/analysis-output-template.md`
- Modify: `docs/shadow-operator.md`

**Interfaces:**
- Consumes: fixed model manifest and fresh-capture version guard.
- Produces: fixed model version `2026-08-03` and authoritative operating guidance.

- [ ] **Step 1: Update failing version expectations first**

Change tests to require `2026-08-03` for fresh captures while retaining an
explicit exact-replay test for older records.

- [ ] **Step 2: Advance the implementation version and config flags**

Update `LEGACY_MODEL_VERSION`, `config/model.json`, and negative-only gate
metadata.

- [ ] **Step 3: Document each gate once**

Update the authoritative model, output template, and operator guide without
duplicating existing bullpen-role, opener/bulk, or market-pass rules.

- [ ] **Step 4: Run the complete suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Verify scope**

Run: `git diff --check`, `git status --short`, and inspect the final diff.

Expected: only the approved model, validation, tests, version, and docs change.
