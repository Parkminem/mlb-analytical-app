# MLB Model Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Make the fixed MLB analysis model reusable without this chat and publish it as a private GitHub repository.

**Architecture:** Keep `docs/model.md` and `analysis/model.py` authoritative, add project-level agent instructions, and add a small standard-library pipeline that validates machine-readable configuration and produces a consistent checklist. Documentation templates preserve the daily analysis and review workflows without storing private data.

**Tech Stack:** Python standard library, JSON, Markdown, unittest, Git, GitHub CLI.

---

### Task 1: Add Reproducibility Configuration

**Files:**
- Create: `tests/test_pipeline.py`
- Create: `config/model.json`
- Create: `analysis/pipeline.py`

- [ ] Write failing tests for configuration loading, layer order, pitch-velocity grading, and checklist creation.
- [ ] Run `python3 -m unittest tests/test_pipeline.py` and confirm the missing module failure.
- [ ] Implement the minimal pipeline helpers and configuration.
- [ ] Run `python3 -m unittest tests/test_pipeline.py` and confirm the tests pass.

### Task 2: Add Future-Session Instructions and Templates

**Files:**
- Create: `AGENTS.md`
- Create: `docs/analysis-output-template.md`
- Create: `docs/new-chat-prompt.md`
- Create: `docs/data-sources.md`
- Create: `docs/reviews/README.md`
- Modify: `README.md`

- [ ] Document the fixed-model rule and required source hierarchy.
- [ ] Document the default analysis output and postgame review format.
- [ ] Replace the outdated README inventory with setup and usage instructions.

### Task 3: Verify and Publish

**Files:**
- Verify: all tracked files

- [ ] Run `python3 -m unittest discover -s tests -v`.
- [ ] Inspect `git status` and ensure caches, temporary HTML, and private reports are ignored.
- [ ] Commit the repository.
- [ ] Create private repository `Parkminem/mlb-analytical-app` and push `main`.
- [ ] Verify the GitHub remote and repository visibility.

