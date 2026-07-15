# Shadow Operator CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, idempotent three-stage CLI that records projected and confirmed shadow forecasts, ingests official results, records reviews, and writes deterministic calibration reports.

**Architecture:** Add a focused `analysis.operator` package over the existing forecasting, shadow, ledger, review, and calibration APIs. Versioned Pydantic JSON bundles are converted into existing frozen domain dataclasses, while orchestration enforces capture supersession, exact confirmed-forecast binding, result corrections, and leakage-safe closing prices. Runtime state remains append-only JSONL plus immutable bundle copies and deterministic JSON/Markdown reports.

**Tech Stack:** Python 3.14, frozen dataclasses, Pydantic 2 `TypeAdapter`, Typer, Rich, unittest/pytest, append-only JSONL ledger.

---

## File Structure

- Create `analysis/operator/__init__.py`: public operator result and service exports.
- Create `analysis/operator/errors.py`: stable operator exception categories and exit codes.
- Create `analysis/operator/schemas.py`: strict versioned projected, confirmed, and review bundle schemas.
- Create `analysis/operator/config.py`: immutable forecast-config registry loader.
- Create `analysis/operator/bundles.py`: canonical loading, hashing, capture keys, and immutable managed copies.
- Create `analysis/operator/service.py`: pregame and review orchestration.
- Create `analysis/operator/reporting.py`: deterministic JSON and Korean Markdown reports.
- Create `analysis/operator/cli.py`: Typer commands and exception-to-exit mapping.
- Create `analysis/operator/__main__.py`: `python -m analysis.operator` entry point.
- Create `config/forecasting/shadow-forecast-2026-07-14.1.json`: current immutable `SimulationConfig` registry entry.
- Modify `analysis/forecasting/validation.py`: expose typed relationship and timing/provenance validation failures.
- Modify `analysis/shadow.py`: expose capture-command construction and accept optional supersession.
- Modify `.gitignore`: ignore `data/shadow/`.
- Modify `README.md`: document the three operator commands and bundle workflow.
- Create `tests/test_operator_schemas.py`: schema and config validation.
- Create `tests/test_operator_pregame.py`: projected/confirmed idempotency and supersession.
- Create `tests/test_operator_review.py`: result, closing market, review, and report flow.
- Create `tests/test_operator_reporting.py`: deterministic report and migration-gate counts.
- Create `tests/test_operator_cli.py`: CLI output, defaults, path overrides, and exit codes.

### Task 1: Immutable Config Registry and Strict Bundle Schemas

**Files:**
- Create: `config/forecasting/shadow-forecast-2026-07-14.1.json`
- Create: `analysis/operator/errors.py`
- Create: `analysis/operator/config.py`
- Create: `analysis/operator/schemas.py`
- Modify: `analysis/forecasting/validation.py`
- Test: `tests/test_operator_schemas.py`
- Test: `tests/test_forecasting_inputs.py`

- [x] **Step 1: Write failing config-registry and schema tests**

Cover:

```python
def test_loads_current_registry_config():
    config = load_forecast_config("shadow-forecast-2026-07-14.1")
    assert config == SimulationConfig()

def test_unknown_registry_version_is_rejected():
    with pytest.raises(BundleValidationError, match="unknown forecast config"):
        load_forecast_config("missing")

def test_projected_bundle_round_trips_domain_values():
    bundle = parse_projected_bundle(projected_bundle_mapping())
    assert isinstance(bundle.payload.game_input, NormalizedGameInput)
    assert isinstance(bundle.payload.legacy_decision, LegacyDecisionSnapshot)

def test_bundle_rejects_unknown_fields_and_naive_timestamps():
    ...

def test_bundle_rejects_coercible_scalar_types_in_nested_domain_values():
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"]["identity"]["game_pk"] = "900001"
    with pytest.raises(BundleValidationError, match="integer"):
        parse_projected_bundle_json(json.dumps(mapping))

def test_review_bundle_requires_exact_frozen_ids_and_close_terms():
    ...

def test_pregame_bundle_runs_cross_field_domain_validation():
    mapping = projected_bundle_mapping()
    mapping["payload"]["game_input"]["away_lineup"]["team"] = "wrong"
    with pytest.raises(BundleValidationError, match="away lineup"):
        parse_projected_bundle_json(json.dumps(mapping))

def test_pregame_validation_uses_typed_failure_categories():
    with pytest.raises(PregameRelationshipError):
        validate_pregame_input(game_with_team_mismatch())
    with pytest.raises(PregameTimingError):
        validate_pregame_input(game_captured_at_first_pitch())
```

- [x] **Step 2: Run tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_operator_schemas.py -q`

Expected: collection failure because `analysis.operator` does not exist.

- [x] **Step 3: Add the immutable current config file**

Serialize every `SimulationConfig.to_mapping()` field. The loader compares the
file keys with `{field.name for field in dataclasses.fields(SimulationConfig)}`
and rejects both missing and unknown fields before calling `from_mapping`.
Keep registry files additive, reject filename/content version mismatches, and
test a second historical-version fixture with different complete values to
prove that loading does not fall back to current defaults.

- [x] **Step 4: Implement strict Pydantic schemas**

Use `ConfigDict(extra="forbid", frozen=True)` and direct domain types:

```python
class PregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    game_input: NormalizedGameInput
    legacy_decision: LegacyDecisionSnapshot
    forecast_config_version: str

class ProjectedBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    stage: Literal["projected"]
    game_pk: int
    created_at: datetime
    payload: PregamePayload
```

Add equivalent `ConfirmedBundle`, `ReviewPayload`, and `ReviewBundle`. Validate
aware timestamps, envelope/payload game identity, `close` observations, prices
above `1.0`, and exact frozen IDs. Map Pydantic failures to
`BundleValidationError` (exit 3). Load raw UTF-8 JSON with a duplicate-key and
non-finite-number precheck, then call
`TypeAdapter(BundleType).validate_json(raw_bytes, strict=True)`. Never use
coercive `model_validate` / `validate_python` on decoded mappings. Add nested
wrong-type tests for integers, floats, booleans, enums, and timestamps.
Add `PregameRelationshipError(ValueError)` and
`PregameTimingError(ValueError)` to `analysis.forecasting.validation`.
Relationship failures cover team, market, response, and observation-reference
integrity. Timing failures cover `captured_at >= first_pitch`, feature
availability/lookback after capture, and source observation availability after
capture. Existing callers that catch `ValueError` remain compatible.

After strict construction, service preflight calls `validate_pregame_input`.
Translate `PregameRelationshipError` to `BundleValidationError` (3) and
`PregameTimingError` to `OperatorPreconditionError` (4). Stage-specific lineup
state checks also remain preconditions (4). Do not classify failures by
matching error-message text.

- [x] **Step 5: Run schema tests and verify green**

Run: `.venv/bin/python -m pytest tests/test_operator_schemas.py -q`

- [x] **Step 6: Commit**

```bash
git add config/forecasting analysis/forecasting/validation.py analysis/operator/errors.py analysis/operator/config.py analysis/operator/schemas.py tests/test_forecasting_inputs.py tests/test_operator_schemas.py
git commit -m "feat: add versioned shadow operator bundles"
```

### Task 2: Shadow Capture Construction and Bundle Persistence

**Files:**
- Modify: `analysis/shadow.py`
- Create: `analysis/operator/bundles.py`
- Modify: `.gitignore`
- Test: `tests/test_shadow.py`
- Test: `tests/test_operator_pregame.py`

- [x] **Step 1: Write failing tests for capture construction and persistence**

Cover:

```python
def test_record_shadow_run_records_optional_supersession():
    second = record_shadow_run(..., supersedes_capture_id=first.capture.capture_id)
    assert second.capture.supersedes_capture_id == first.capture.capture_id

def test_capture_key_covers_schema_stage_and_intrinsic_command():
    assert intrinsic_capture_key(1, "projected", command_a) != intrinsic_capture_key(1, "confirmed", command_a)
    assert intrinsic_capture_key(1, "projected", command_a) == intrinsic_capture_key(
        1, "projected", replace(command_a, supersedes_capture_id="other")
    )

def test_managed_bundle_copy_is_immutable_and_replayable(tmp_path):
    ...
```

- [x] **Step 2: Run targeted tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_shadow.py tests/test_operator_pregame.py -q`

- [x] **Step 3: Refactor capture construction into a public helper**

Add a pure `build_shadow_capture_command(game_input, supersedes_capture_id=None)`
that performs the existing validation, eligibility, input hash, lineup state,
and payload construction. Make `record_shadow_run` use it and add a backward-
compatible keyword-only supersession argument.

- [x] **Step 4: Implement bundle canonicalization and persistence**

`bundles.py` must:

- Load UTF-8 JSON and reject duplicate keys/non-finite values.
- Canonicalize with the repository's `canonical_json`.
- Compute SHA-256 bundle identity.
- Compute capture keys from `(schema_version, stage,
  canonical_command_without_supersession)`. Tests must prove each schema/stage
  dimension changes the key and only `supersedes_capture_id` is excluded.
- Write `<stage>-v1-<hash16>.json` atomically.
- Reuse identical existing bytes and reject conflicting bytes.
- Map I/O failures to exit 6.

- [x] **Step 5: Add `data/shadow/` to `.gitignore`**

- [x] **Step 6: Run targeted tests and verify green**

Run: `.venv/bin/python -m pytest tests/test_shadow.py tests/test_operator_pregame.py -q`

- [x] **Step 7: Commit**

```bash
git add analysis/shadow.py analysis/operator/bundles.py .gitignore tests/test_shadow.py tests/test_operator_pregame.py
git commit -m "feat: add replay-safe shadow capture persistence"
```

### Task 3: Projected and Confirmed Pregame Service

**Files:**
- Create: `analysis/operator/service.py`
- Create: `analysis/operator/__init__.py`
- Test: `tests/test_operator_pregame.py`

- [x] **Step 1: Write failing projected/confirmed orchestration tests**

Required cases:

- Projected command creates capture, forecast, link, and managed bundle.
- Exact projected retry returns the same records.
- Same forecast semantics with later `captured_at` creates a distinct capture
  and reuses the deterministic forecast.
- Changed audit provenance creates a distinct capture.
- Updated projected state supersedes the latest active projected capture.
- Confirmed state supersedes the latest active pregame capture.
- Exact confirmed retry preserves the original supersession target.
- New projected state after active confirmation is rejected, while exact
  historical projected replay succeeds.
- Post-first-pitch captures and stage/lineup mismatches fail with exit 4.

- [x] **Step 2: Run pregame tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_operator_pregame.py -q`

- [x] **Step 3: Implement active capture and keyed replay helpers**

Read captures through `LedgerStore.iter_records("capture")`. Match an existing
record by `(record_kind, envelope.idempotency_key)` before choosing a target.
For new records, define active captures as those whose `capture_id` is not a
supersession target and sort by:

```python
(capture.captured_at, capture.envelope.appended_at, capture.envelope.record_id)
```

- [x] **Step 4: Implement `run_projected` and `run_confirmed`**

Both functions:

1. Validate stage and pre-first-pitch time.
2. Load the immutable config registry entry.
3. Build the intrinsic capture command and key.
4. Preflight replay or resolve the fixed supersession target.
5. Run `run_forecast` and `record_shadow_run`.
6. Persist the canonical managed bundle only after ledger operations succeed.
7. Return a frozen result with IDs, lineup state, replay flags, and bundle path.

- [x] **Step 5: Run targeted tests and verify green**

Run: `.venv/bin/python -m pytest tests/test_operator_pregame.py -q`

- [x] **Step 6: Commit**

```bash
git add analysis/operator/__init__.py analysis/operator/service.py tests/test_operator_pregame.py
git commit -m "feat: orchestrate projected and confirmed shadow runs"
```

### Task 4: Exact Confirmed Review and Official Result Ingestion

**Files:**
- Modify: `analysis/operator/service.py`
- Test: `tests/test_operator_review.py`

- [x] **Step 1: Write failing review-stage tests**

Cover:

- Exact capture/forecast/link validation.
- Refusal of projected, superseded, post-first-pitch, mismatched, or missing
  frozen records.
- Optional closing observations with complete semantic terms and timestamps.
- Rejection when a closing observation has null `available_at` or either
  `observed_at` / `available_at` falls outside
  `capture.captured_at <= timestamp <= capture.first_pitch`.
- Closing observation correction with explicit supersession.
- Initial official result and explicit correction lineage.
- Rejection unless `source_name == "MLB"` and `finalized_at > first_pitch`
  before any result append.
- Exact retry after close append, result append, or partial review completion.
- Rejection of inferred/self supersession and result score conservation errors.
- One recorded review for every exact review-eligible comparison, including
  structurally eligible legacy passes.

- [x] **Step 2: Run review tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_operator_review.py -q`

- [x] **Step 3: Implement frozen-record validation**

Resolve the bundle's three IDs directly from ledger records. Require:

```text
capture.game_pk == bundle.game_pk
capture.lineup_state == confirmed
capture.captured_at < capture.first_pitch
link.capture_id == capture.capture_id
link.forecast_run_id == forecast.forecast_run_id
capture is active
```

- [x] **Step 4: Implement closing and result preflight**

Before checking latest-target constraints, search each exact idempotency key.
When a key exists, recover its original supersession target for preflight but
still call the relevant `LedgerStore.append_*` method with the complete command;
the store's payload hash must confirm exact equality or raise an exit-5
idempotency conflict. For a new record, enforce explicit nullable targets and
require corrections to reference the current latest record. Validate the MLB
source and strict post-first-pitch result timing before either new append or
replay. For every closing observation, require non-null `available_at` and
validate both observation timestamps against the exact frozen capture interval
before append or replay.

- [x] **Step 5: Implement exact review creation**

Deserialize `shadow_comparison` from the frozen forecast. Select sorted unique
market keys where `review_eligible` is true and call
`build_and_record_review(store, link_record_id, result_record_id, market_key)`.
Return review IDs and classification counts without changing legacy priority.

- [x] **Step 6: Run review tests and verify green**

Run: `.venv/bin/python -m pytest tests/test_operator_review.py -q`

- [x] **Step 7: Commit**

```bash
git add analysis/operator/service.py tests/test_operator_review.py
git commit -m "feat: record exact confirmed shadow reviews"
```

### Task 5: Deterministic Calibration Reporting

**Files:**
- Create: `analysis/operator/reporting.py`
- Modify: `analysis/operator/service.py`
- Test: `tests/test_operator_reporting.py`

- [x] **Step 1: Write failing report tests**

Cover:

- `select_primary_rows` and `build_calibration_report` integration.
- Byte-identical JSON and Markdown for unchanged ledger state.
- `ledger_as_of` equals maximum ledger append timestamp, not wall clock.
- JSON contains the resolved ledger path as well as `ledger_as_of`.
- Confirmed game count uses unique reviewed confirmed game IDs.
- Per-family decision count uses unique confirmed `(forecast_run_id, market_key)`
  rows and includes eligible passes but excludes ineligible markets.
- Progress values for 100 games and 300 decisions.
- Missing price metrics render as unavailable, not zero.
- Atomic replacement and I/O failure mapping.

- [x] **Step 2: Run report tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_operator_reporting.py -q`

- [x] **Step 3: Implement report payload and atomic writers**

Build a versioned mapping with the resolved ledger path, `ledger_as_of`,
overall/group metrics, migration counts, thresholds, and progress. Serialize
JSON through `canonical_json` plus a trailing newline. Render a Korean Markdown
summary from the same mapping.
Write temporary sibling files, fsync, and `os.replace`.

- [x] **Step 4: Integrate reporting into `run_review`**

After idempotent reviews complete, rebuild rows from the full ledger, write
both reports, and return paths plus sample counts.

- [x] **Step 5: Run report tests and verify green**

Run: `.venv/bin/python -m pytest tests/test_operator_reporting.py -q`

- [x] **Step 6: Commit**

```bash
git add analysis/operator/reporting.py analysis/operator/service.py tests/test_operator_reporting.py
git commit -m "feat: write deterministic shadow calibration reports"
```

### Task 6: Typer CLI and Stable Exit Codes

**Files:**
- Create: `analysis/operator/cli.py`
- Create: `analysis/operator/__main__.py`
- Test: `tests/test_operator_cli.py`

- [x] **Step 1: Write failing CLI tests**

Use `typer.testing.CliRunner` to cover:

- All three commands and default paths.
- `--ledger`, `--bundle-root`, and `--report-dir` overrides.
- Success/replay summaries containing stable record IDs.
- Exit 2 for usage, 3 for schema, 4 for preconditions, 5 for ledger errors,
  and 6 for I/O.
- An injected unexpected exception exits 1.
- No traceback for expected operator errors.
- Projected/confirmed summaries expose lineup and replay state. Review summaries
  expose classification counts, calibration sample size, and both report paths.

- [x] **Step 2: Run CLI tests and verify red**

Run: `.venv/bin/python -m pytest tests/test_operator_cli.py -q`

- [x] **Step 3: Implement the Typer app**

Each command loads one bundle, creates `LedgerStore`, calls the matching service
function, and prints a concise Rich summary. Add boundary-specific exception
translation rather than a broad `ValueError` catch:

- JSON, Pydantic, and domain-construction failures inside `schemas.py` become
  `BundleValidationError` (3). Typed `PregameRelationshipError` from service
  preflight also becomes 3.
- Explicit service state/timestamp/source checks raise
  `OperatorPreconditionError` (4), as does typed `PregameTimingError`.
- `IdempotencyConflict`, `LedgerReferenceError`, and `LedgerCorruptionError`
  become `OperatorLedgerError` (5).
- Bundle/report `OSError` becomes `OperatorIOError` (6).
- Typer usage remains 2 and any otherwise unexpected exception is caught only
  at the CLI boundary and returns 1.

- [x] **Step 4: Add `python -m analysis.operator` entry point**

- [x] **Step 5: Run CLI tests and smoke help**

Run: `.venv/bin/python -m pytest tests/test_operator_cli.py -q`

Run: `.venv/bin/python -m analysis.operator --help`

- [x] **Step 6: Commit**

```bash
git add analysis/operator/cli.py analysis/operator/__main__.py tests/test_operator_cli.py
git commit -m "feat: add three-stage shadow operator cli"
```

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/model.md`
- Create: `docs/shadow-operator.md`
- Modify: `docs/superpowers/plans/2026-07-15-shadow-operator-cli.md`

- [x] **Step 1: Document exact daily usage and bundle locations**

Include natural-language-to-CLI mapping, defaults, frozen ID handoff, result
corrections, replay behavior, report interpretation, and the fact that no live
sources or automatic tuning are included.

- [x] **Step 2: Run focused tests**

```bash
.venv/bin/python -m pytest tests/test_operator_schemas.py tests/test_operator_pregame.py tests/test_operator_review.py tests/test_operator_reporting.py tests/test_operator_cli.py -q
```

- [x] **Step 3: Run the complete suite**

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q analysis tests
.venv/bin/python -m analysis.pipeline
git diff --check origin/main...HEAD
```

Expected: all tests pass; manifest keeps `legacy_fixed_model` authority,
forecasting mode `shadow`, and `may_change_recommendations=false`.

- [x] **Step 4: Run a synthetic three-stage CLI smoke test in a temporary directory**

Verify projected IDs, confirmed supersession/frozen IDs, result/review records,
and deterministic report replay.

- [x] **Step 5: Request independent code review and fix every P1/P2 finding**

- [x] **Step 6: Commit final documentation and verification updates**

```bash
git add README.md docs/model.md docs/shadow-operator.md docs/superpowers/plans/2026-07-15-shadow-operator-cli.md
git commit -m "docs: document shadow operator workflow"
```
