# MLB Analysis Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable scoring core for F5/full-game MLB analysis with bullpen availability and market-aware totals.

**Architecture:** Create pure Python data classes and scoring functions under `analysis/`. Keep data fetching out of scope for this first pass so the model can be tested with deterministic inputs.

**Tech Stack:** Python standard library, `unittest`.

---

### Task 1: Bullpen Availability

**Files:**
- Create: `analysis/model.py`
- Test: `tests/test_model.py`

- [ ] Write failing tests for high-leverage reliever fatigue.
- [ ] Implement `RelieverAppearance` and `score_bullpen_availability`.
- [ ] Verify tests pass.

### Task 2: Starter Recent Form

**Files:**
- Modify: `analysis/model.py`
- Test: `tests/test_model.py`

- [ ] Write failing tests for starter command risk.
- [ ] Implement `StarterRecentForm` and `score_starter_recent_form`.
- [ ] Verify tests pass.

### Task 3: Recommendation Layer

**Files:**
- Modify: `analysis/model.py`
- Test: `tests/test_model.py`

- [ ] Write failing tests for F5 vs full-game separation.
- [ ] Implement `FullGameContext` and `recommend_full_game`.
- [ ] Verify tests pass.

### Task 4: Documentation

**Files:**
- Modify: `docs/model.md`
- Modify: `README.md`

- [ ] Document how to use the model.
- [ ] Verify examples match tests.
