# Total And Priority Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the three approved calibration findings without adding positive scoring or changing unrelated market engines.

**Architecture:** Keep baseball scoring unchanged. Add negative-only gates to the full-game under engine and frozen evidence contract, cap unsupported F5 overs at the recording boundary, and raise only the final priority floor from 72 to 75. Publish the behavior as legacy model version `2026-09-11` while retaining historical review compatibility.

**Tech Stack:** Python dataclasses, JSON configuration, pytest/unittest.

---

### Task 1: Lock The New Behavior With Failing Tests

**Files:**
- Modify: `tests/test_model.py`
- Modify: `tests/test_review_evidence_contract.py`
- Modify: `tests/test_pipeline.py`

- [x] Add tests proving scores 72-74 remain leans but are excluded from final priority.
- [x] Add tests proving an F5 over without frozen recent run-conversion evidence is capped at 74.
- [x] Add tests proving a full-game under requires two-sided recent and projected scoring suppression.
- [x] Add tests proving command, home-run, injury-return, or short-exit starter volatility blocks a full-game under at every total.
- [x] Run the targeted tests and verify they fail for the intended missing behavior.

### Task 2: Implement The Negative-Only Gates

**Files:**
- Modify: `analysis/model.py`
- Modify: `analysis/review/evidence_contract.py`
- Modify: `analysis/shadow.py`
- Modify: `analysis/pipeline.py`
- Modify: `config/model.json`

- [x] Add explicit full-game under suppression and starter-volatility context fields.
- [x] Apply the under gates before every full-game under return path.
- [x] Derive the same facts from the frozen normalized game input at recording time.
- [x] Cap unsupported F5 overs at 74 and unsupported full-game unders at 69.
- [x] Raise the playable/final-priority threshold to 75 without changing 65-74 lean display behavior.
- [x] Bump the current legacy model version while preserving historical contract versions.
- [x] Run targeted tests until green.

### Task 3: Publish And Verify The Contract

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/model.md`
- Modify: `docs/analysis-output-template.md`
- Modify: `docs/shadow-operator.md`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_shadow.py`
- Modify: `tests/test_operator_pregame.py`

- [x] Document the 75-point final-priority floor and both new total gates once in the authoritative model section.
- [x] Update current-version references while retaining historical version lists.
- [x] Run the full test suite and `git diff --check`.
- [x] Review the final diff for accidental rescoring or unrelated behavior changes.
