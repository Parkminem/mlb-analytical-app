# Priority Admission Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent market-engine pass decisions and duplicated evidence from entering the final priority stack without adding new baseball scores or caps.

**Architecture:** Add `assemble_same_game_priority`, which filters `pass` results and deduplicates candidates by overlap among canonical evidence-path enum values. Keep `evaluate_priority_admission` and all market-engine ownership unchanged, adapt the existing total-only selector to the assembler, and expose the assembler in the pipeline manifest.

**Tech Stack:** Python dataclasses, `unittest`, JSON configuration, Markdown model documentation.

---

### Task 1: Add failing candidate-selection tests

**Files:**
- Modify: `tests/test_model.py`

- [ ] Add a test proving a `pass` handicap or total is excluded regardless of score.
- [ ] Add tests for correlated and independent F5 side-total selection.
- [ ] Add three-candidate tests for full correlation, partial independence, and exact ties.
- [ ] Add a test proving missing evidence cannot retain multiple markets.
- [ ] Preserve existing F5-total versus full-game-total selection behavior.
- [ ] Run the focused tests and confirm they fail for missing behavior.

### Task 2: Implement generic same-game selection

**Files:**
- Modify: `analysis/model.py`

- [ ] Add canonical evidence-path enum values and a candidate dataclass containing market, score, decision, and evidence paths.
- [ ] Filter `pass` decisions before ranking.
- [ ] Select by score, deterministic tie order, and evidence overlap.
- [ ] Treat missing evidence as unverified independence.
- [ ] Delegate the existing total-only selector to the generic selector.
- [ ] Run focused tests and confirm they pass.

### Task 3: Synchronize the fixed specification

**Files:**
- Modify: `config/model.json`
- Modify: `docs/model.md`
- Modify: `docs/analysis-output-template.md`
- Modify: `analysis/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] Expose pass exclusion and unified same-game deduplication in configuration.
- [ ] Document that market-engine validations remain the sole owners of pass decisions.
- [ ] Expose `assemble_same_game_priority` as the final candidate assembler in the analysis manifest.
- [ ] Add an end-to-end test that feeds real handicap and total engine decisions into the assembler and verifies a high-scoring `pass` is absent from final output.
- [ ] Add configuration regression assertions.

### Task 4: Verify the repository

**Files:**
- Verify only

- [ ] Run `python3 -m unittest discover -s tests -v`.
- [ ] Run `python3 -m analysis.pipeline --lineup confirmed`.
- [ ] Inspect `git diff --check` and the final diff for duplication or unrelated changes.
