# Pregame Bundle Authority V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make schema-v2 pregame bundles the only source of fresh projected or confirmed execution while preserving every schema-v1 pregame artifact as authenticated exact replay only.

**Architecture:** Add an explicit `schema_version: 2` and `authority: "current"` envelope to current projected and confirmed bundles. Parse schema-v1 pregame artifacts into distinct relaxed-v1 or strict-v1 replay-only runtime types, authenticate them only from managed paths, and route them exclusively to exact replay services. Review bundles remain schema version 1.

**Tech Stack:** Python 3.14, Pydantic 2.13, frozen dataclasses, descriptor-relative filesystem I/O, append-only JSONL ledger, pytest/unittest.

## Global Constraints

- The fixed MLB model version remains exactly `2026-08-03`.
- New projected and confirmed bundles require the JSON integer `schema_version: 2` and the explicit literal `authority: "current"`.
- Review bundles remain schema version 1 and retain their current behavior.
- Every schema-v1 projected or confirmed artifact is replay-only, regardless of whether candidate compatibility fields are absent or present.
- A v1 replay requires an authenticated managed path and an exact existing capture, forecast, and link.
- Never rewrite historical managed bundle bytes or ledger records.
- Rejections must create no ledger record, forecast, link, supersession, report, or managed artifact.
- Preserve strict JSON duplicate-key, scalar, datetime, tuple, boolean, extra-field, and sanitized-error behavior.
- Preserve baseball scoring, market engines, F5 engines, same-game priority assembly, and Shadow decision authority.
- Use `apply_patch` for manual edits, TDD for every production change, and one focused commit per task.

---

### Task 1: Define Current V2 And Historical V1 Schema Types

**Files:**
- Modify: `analysis/operator/schemas.py:25-225`
- Modify: `analysis/operator/schemas.py:248-695`
- Modify: `tests/test_operator_schemas.py:43-110`
- Modify: `tests/test_operator_schemas.py:330-710`
- Modify: `tests/test_operator_pregame.py:90-330`

**Interfaces:**
- Consumes: `PriorityMarketCandidate`, `LegacyDecisionSnapshot`, `PregamePayload`, `_parse_bundle_json(...)`, and existing strict audit-presence helpers.
- Produces: `CURRENT_PREGAME_SCHEMA_VERSION = 2`, `CURRENT_PREGAME_AUTHORITY = "current"`, exact current `ProjectedBundle` and `ConfirmedBundle`, replay-only `HistoricalRelaxedProjectedBundleV1`, `HistoricalStrictProjectedBundleV1`, `HistoricalRelaxedConfirmedBundleV1`, and `HistoricalStrictConfirmedBundleV1`.
- Produces: `parse_historical_projected_bundle_json(...)` and `parse_historical_confirmed_bundle_json(...)` returning the appropriate v1 runtime union without changing original candidate-field presence.

- [ ] **Step 1: Add failing v2 envelope and laundering tests**

Add stage-parameterized tests that start with `projected_bundle_mapping()` and assert the current contract:

```python
@pytest.mark.parametrize(
    ("parser", "mapping_factory"),
    (
        (parse_projected_bundle_json, projected_bundle_mapping),
        (parse_confirmed_bundle_json, confirmed_bundle_mapping),
    ),
)
def test_current_pregame_requires_v2_authority(parser, mapping_factory):
    mapping = mapping_factory()
    mapping["schema_version"] = 2
    mapping["authority"] = "current"

    parsed = parser(json.dumps(mapping))

    assert parsed.schema_version == 2
    assert parsed.authority == "current"
```

Add negative cases for missing authority, `authority: "historical"`, schema
version 1, string schema version `"2"`, extra authority values, and missing
`same_game_pair_compatible` on any one of two candidates. Add current empty
candidate coverage proving that v2 authority, not candidate count, admits the
bundle.

Add projected and confirmed laundering cases:

```python
for validate in (
    lambda: current_type.model_validate(historical, from_attributes=True),
    lambda: TypeAdapter(current_type).validate_python(
        historical,
        from_attributes=True,
    ),
    lambda: current_type.model_validate_json(historical.model_dump_json()),
):
    with pytest.raises(ValidationError):
        validate()
```

Cover relaxed v1, strict v1, and zero-candidate v1 inputs. The fresh control
must contain both `schema_version: 2`, `authority: "current"`, and an explicit
boolean on every candidate.

- [ ] **Step 2: Run the schema tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_operator_schemas.py \
  -k 'v2 or authority or laundering or historical_shape'
```

Expected: failures showing current bundles still accept schema version 1 or
lack the explicit authority field, and v1 empty/from-attributes inputs can be
validated as current types.

- [ ] **Step 3: Implement version-specific envelopes and payloads**

Refactor the envelope so subclasses narrow the version. Remove
`_require_pair_compatibility_core_field(...)` and every direct mutation of
Pydantic's generated CoreSchema layout. Use the public `ValidateAs` API to
validate current candidate mappings with a required boolean and convert them
back to the existing frozen domain dataclasses:

```python
from typing import Annotated

from pydantic import ValidateAs

CURRENT_PREGAME_SCHEMA_VERSION = 2
CURRENT_PREGAME_AUTHORITY = "current"


class _CurrentPriorityCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    market: str
    score: int
    decision_rating: str
    evidence_paths: frozenset[PriorityEvidencePath]
    same_game_pair_compatible: bool


def _current_candidate(value: _CurrentPriorityCandidateInput):
    return PriorityMarketCandidate(
        market=value.market,
        score=value.score,
        decision_rating=value.decision_rating,
        evidence_paths=value.evidence_paths,
        same_game_pair_compatible=value.same_game_pair_compatible,
    )


_CurrentPriorityCandidate = Annotated[
    PriorityMarketCandidate,
    ValidateAs(_CurrentPriorityCandidateInput, _current_candidate),
]


class _CurrentLegacyDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: str
    candidates: tuple[_CurrentPriorityCandidate, ...]
    final_priority: tuple[str, ...]
    review_bindings: tuple[CandidateReviewBinding, ...]
    notes: tuple[str, ...] = ()


def _current_legacy(value: _CurrentLegacyDecisionInput):
    return LegacyDecisionSnapshot(
        model_version=value.model_version,
        candidates=value.candidates,
        final_priority=value.final_priority,
        review_bindings=value.review_bindings,
        notes=value.notes,
    )


_CurrentLegacyDecision = Annotated[
    LegacyDecisionSnapshot,
    ValidateAs(_CurrentLegacyDecisionInput, _current_legacy),
]


class _CurrentPregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    game_input: NormalizedGameInput
    legacy_decision: _CurrentLegacyDecision
    forecast_config_version: str


class _BundleEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    game_pk: int
    created_at: datetime


class _CurrentPregameBundle(_PregameBundle):
    schema_version: Literal[2]
    authority: Literal["current"]
    payload: _CurrentPregamePayload


class ReviewBundle(_BundleEnvelope):
    schema_version: Literal[1]
    stage: Literal["review"]
    payload: ReviewPayload
```

Keep the JSON-integer validator on `_BundleEnvelope`. The required input-model
boolean must make direct BaseModel and TypeAdapter JSON/Python paths reject
omission before constructing `PriorityMarketCandidate`. Preserve the existing
domain dataclass values after validation so forecasting and Shadow code require
no type changes. Remove bundle-level provenance logic that infers authority from
candidate count and remove the private CoreSchema layout dependency introduced
by commit `9cdab25`.

Define separate v1 payload/runtime shapes:

```python
class _HistoricalRelaxedPregamePayload(PregamePayload):
    @model_serializer(mode="wrap")
    def _preserve_omitted_compatibility(self, handler):
        payload = handler(self)
        for candidate in _legacy_candidate_mappings(payload):
            candidate.pop("same_game_pair_compatible", None)
        return payload


class _HistoricalStrictPregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    game_input: NormalizedGameInput
    legacy_decision: _CurrentLegacyDecision
    forecast_config_version: str


class _HistoricalPregameBundleV1(_PregameBundle):
    schema_version: Literal[1]
```

Create projected and confirmed relaxed/strict subclasses with exact stage and
payload annotations. The relaxed payload uses a relaxed candidate input model
whose fields exclude `same_game_pair_compatible` and whose converter constructs
the historical domain candidate with `False`. None of the historical types has
an `authority` field.

- [ ] **Step 4: Classify relaxed and strict v1 raw inputs deterministically**

Inspect raw decoded candidate mappings before Pydantic defaults are applied:

```python
def _historical_candidate_contract(value: object) -> Literal["relaxed", "strict"]:
    candidates = _raw_candidate_mappings(value)
    presence = tuple(
        "same_game_pair_compatible" in candidate for candidate in candidates
    )
    if not presence:
        return "relaxed"
    if all(presence):
        return "strict"
    if not any(presence):
        return "relaxed"
    raise BundleValidationError(
        "invalid historical bundle: mixed compatibility contract"
    )
```

Select the matching adapter and audit-presence rule. Relaxed audit fields omit
the compatibility field; strict audit fields include it. Reject mixed candidate
shapes. Preserve strict booleans and relaxed omission through `model_dump()`,
`model_dump_json()`, and nested payload dumps.

- [ ] **Step 5: Run schema and pregame construction tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_schemas.py \
  tests/test_operator_pregame.py \
  -k 'schema or bundle or authority or historical or laundering'
```

Expected: all selected tests pass. Confirm direct current v2 JSON parsing still
accepts strict datetime strings and rejects coercible schema/authority values.

- [ ] **Step 6: Commit the schema boundary**

```bash
git add analysis/operator/schemas.py \
  tests/test_operator_schemas.py tests/test_operator_pregame.py
git diff --cached --check
git commit -m "feat: introduce pregame bundle authority v2"
```

---

### Task 2: Authenticate Both V1 Shapes And Replay Them Without Writes

**Files:**
- Modify: `analysis/operator/bundles.py:29-215`
- Modify: `analysis/operator/service.py:89-109`
- Modify: `analysis/operator/service.py:923-1023`
- Modify: `tests/test_operator_artifact_fdio.py:60-190`
- Modify: `tests/test_operator_pregame.py:1362-1845`

**Interfaces:**
- Consumes: the four historical v1 runtime types and parsers from Task 1, `load_historical_managed_bundle(...)`, `_existing_exact_pregame_replay(...)`, and `ManagedBundleCopy`.
- Produces: `ReplayOnlyPregameBundle.bundle` as the union of all four replay-only types.
- Produces: `replay_historical_projected(...)` and `replay_historical_confirmed(...)` accepting authenticated relaxed-v1 or strict-v1 managed artifacts only.

- [ ] **Step 1: Add failing relaxed-v1 and strict-v1 exact-replay tests**

Extend the existing projected/confirmed dedicated replay test with a
`candidate_contract` parameter:

```python
@pytest.mark.parametrize("candidate_contract", ("relaxed", "strict"))
def test_v1_managed_bundle_is_exact_replay_only(
    candidate_contract,
    stage,
    runner,
    historical_replay,
):
    mapping = json.loads(bundle_canonical_bytes(current_v2_bundle))
    mapping.pop("authority")
    mapping["schema_version"] = 1
    if candidate_contract == "relaxed":
        for candidate in mapping["payload"]["legacy_decision"]["candidates"]:
            candidate.pop("same_game_pair_compatible")
    source = _write_hash_named_managed_bundle(mapping, bundles)
    historical_bundle = historical_parser(source.read_bytes())
    config = load_forecast_config(
        historical_bundle.payload.forecast_config_version
    )
    forecast = run_forecast(historical_bundle.payload.game_input, config)
    comparison = compare_shadow_forecast(
        historical_bundle.payload.legacy_decision,
        forecast,
    )
    capture_command = build_shadow_capture_command(
        historical_bundle.payload.game_input
    )
    records = record_shadow_run(
        LedgerStore(ledger),
        historical_bundle.payload.game_input,
        historical_bundle.payload.legacy_decision,
        forecast,
        ShadowRunIdempotencyKeys(
            capture=intrinsic_capture_key(
                1,
                stage,
                capture_command,
            )
        ),
        comparison=comparison,
    )

    replay = historical_replay(
        source,
        ledger_path=ledger,
        bundle_root=bundles,
    )

    assert replay.capture == records.capture
    assert replay.forecast == records.forecast
    assert replay.link == records.link
    assert replay.managed_bundle.path == source.resolve()
```

Use the existing ledger-rewrite helper where a relaxed historical forecast
must omit compatibility. For strict v1, retain the field in both bundle and
frozen legacy comparison.

For both shapes, call the replay service against an empty ledger and patch
`run_forecast`, `record_shadow_run`, and `persist_managed_bundle` to raise.
Assert the precondition error, no ledger, and unchanged managed file snapshot.

- [ ] **Step 2: Run focused replay tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_operator_pregame.py \
  -k 'v1 and (exact_replay or dedicated_replay or replay_only)'
```

Expected: strict-v1 cases fail because the existing historical loader either
parses them as current or accepts only the relaxed historical runtime type.

- [ ] **Step 3: Update authenticated loader classification**

Change imports and the replay wrapper union:

```python
HistoricalPregameBundleV1 = (
    HistoricalRelaxedProjectedBundleV1
    | HistoricalStrictProjectedBundleV1
    | HistoricalRelaxedConfirmedBundleV1
    | HistoricalStrictConfirmedBundleV1
)


@dataclass(frozen=True)
class ReplayOnlyPregameBundle:
    bundle: HistoricalPregameBundleV1
    managed_bundle: ManagedBundleCopy
```

After descriptor-relative authentication and hashing, apply these rules:

```python
if stage in ("projected", "confirmed") and filename_version == 1:
    bundle = _HISTORICAL_BUNDLE_PARSERS[stage](raw)
    replay_only = True
else:
    bundle = _BUNDLE_PARSERS[stage](raw)
    replay_only = False
```

Do not attempt the v2 current parser first for a v1 pregame artifact. Keep
review-v1 on `parse_review_bundle_json`. Reject v2 through the historical replay
API and retain the filename version, stage, hash, game directory, and content
identity checks.

- [ ] **Step 4: Update replay service type guards**

Replace the single expected type with an exact tuple:

```python
def _replay_historical_pregame(
    path: str | Path,
    *,
    expected_stage: str,
    expected_types: tuple[type[BaseModel], ...],
    ledger_path: str | Path,
    bundle_root: str | Path,
    registry_root: str | Path | None,
) -> PregameRunResult:
    artifact = load_historical_managed_bundle(path, root=bundle_root)
    if not isinstance(artifact, ReplayOnlyPregameBundle):
        raise OperatorPreconditionError(
            "historical replay requires a schema-v1 managed bundle"
        )
    bundle = artifact.bundle
    if type(bundle) not in expected_types or bundle.stage != expected_stage:
        raise OperatorPreconditionError("historical replay stage/type mismatch")
```

Pass the projected relaxed/strict tuple from
`replay_historical_projected(...)` and the confirmed tuple from
`replay_historical_confirmed(...)`. Keep exact replay lookup before every
append-capable operation.

- [ ] **Step 5: Extend filesystem and provenance regression tests**

Run every existing symlink, nonregular-file, validation/read swap, and
root/game/file substitution test once with relaxed v1 and once with strict v1.
Add assertions that:

```python
assert source.read_bytes() == original_bytes
assert artifact.managed_bundle.bundle_hash == hashlib.sha256(
    original_bytes
).hexdigest()
assert artifact.managed_bundle.path == source.resolve()
```

Confirm public serialization preserves strict field presence and relaxed field
omission for one, many, and zero candidates.

- [ ] **Step 6: Run loader and replay tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_artifact_fdio.py \
  tests/test_operator_pregame.py \
  -k 'historical or v1 or provenance or normal_runner'
```

Expected: all selected tests pass for projected/confirmed and relaxed/strict
v1 shapes with no forbidden side effects.

- [ ] **Step 7: Commit authenticated v1 replay support**

```bash
git add analysis/operator/bundles.py analysis/operator/service.py \
  tests/test_operator_artifact_fdio.py tests/test_operator_pregame.py
git diff --cached --check
git commit -m "fix: make schema v1 pregame replay only"
```

---

### Task 3: Route CLI Fresh Execution To V2 And Managed V1 To Replay

**Files:**
- Modify: `analysis/operator/cli.py:45-215`
- Modify: `analysis/operator/cli.py:320-395`
- Modify: `tests/test_operator_cli.py:110-520`
- Modify: `tests/test_shadow_performance_contract.py:90-135`

**Interfaces:**
- Consumes: current v2 `ProjectedBundle`/`ConfirmedBundle`, `ReplayOnlyPregameBundle`, `load_bundle(...)`, `load_historical_managed_bundle(...)`, and the two historical replay services.
- Produces: `_load_pregame_bundle(...) -> _PregameBundleLoad` with exactly one of `bundle` or `historical_path` populated.
- Produces: projected/confirmed commands that call normal runners only for exact current v2 types and historical replay services only for authenticated v1 managed paths.

- [ ] **Step 1: Add failing CLI dispatch tests for all authority states**

Parameterize projected and confirmed tests over:

```python
cases = (
    ("v2-current", "normal"),
    ("v1-relaxed-managed", "replay"),
    ("v1-strict-managed", "replay"),
    ("v1-relaxed-outside", "reject"),
    ("v1-strict-outside", "reject"),
    ("v2-missing-authority", "reject"),
)
```

Patch the opposite service to raise for each accepted case. Assert v1 outside
the managed root never calls `load_historical_managed_bundle`, and rejected
inputs print only sanitized validation/precondition text.

Add an empty-candidate v1 managed case and a direct
`from_attributes=True` conversion probe. Neither may reach `run_projected` or
`run_confirmed`.

- [ ] **Step 2: Run CLI tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_operator_cli.py \
  -k 'v2 or v1 or historical or dispatch or authority'
```

Expected: strict-v1 managed inputs or missing-authority v2 inputs follow the
old branch or fail with an incorrect type classification.

- [ ] **Step 3: Implement explicit CLI authority routing**

Keep strict loading first for normal inputs. When strict parsing fails, classify
only a path structurally inside the configured managed root:

```python
try:
    bundle = _load_stage_bundle(
        path,
        expected_type=expected_type,
        expected_stage=expected_stage,
        bundle_root=bundle_root,
    )
except BundleValidationError:
    if resolved_path.parent.parent != resolved_root:
        raise
    historical = load_historical_managed_bundle(path, root=bundle_root)
    if (
        not isinstance(historical, ReplayOnlyPregameBundle)
        or historical.bundle.schema_version != 1
        or historical.bundle.stage != expected_stage
    ):
        raise OperatorPreconditionError(
            "schema-v1 pregame replay stage/type mismatch"
        )
    return _PregameBundleLoad(bundle=None, historical_path=resolved_path)
```

The projected and confirmed commands retain the exclusive branch:

```python
if loaded.historical_path is not None:
    result = replay_historical_projected(
        loaded.historical_path,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )
else:
    assert type(loaded.bundle) is ProjectedBundle
    result = run_projected(
        loaded.bundle,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )
```

Do not pass parsed historical objects or caller-supplied provenance to the
replay service.

- [ ] **Step 4: Update all current bundle fixtures to v2**

Change shared current mapping/build helpers to emit:

```python
mapping["schema_version"] = 2
mapping["authority"] = "current"
```

Historical helpers must explicitly derive v1 bytes by setting version 1,
removing authority, and retaining or removing compatibility according to the
named strict/relaxed fixture. Do not let a generic current helper silently
stand in for historical input.

- [ ] **Step 5: Run CLI and performance-contract tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_cli.py \
  tests/test_shadow_performance_contract.py \
  tests/test_operator_schemas.py
```

Expected: all tests pass. The performance contract must continue to measure the
same confirmed analysis work; the extra v2 envelope validation is constant-time
and does not add a source fetch, simulation, or ledger scan.

- [ ] **Step 6: Commit CLI authority dispatch**

```bash
git add analysis/operator/cli.py tests/test_operator_cli.py \
  tests/test_shadow_performance_contract.py tests/test_operator_schemas.py
git diff --cached --check
git commit -m "feat: route pregame execution by bundle authority"
```

---

### Task 4: Document V2 Operation And Prove Full Compatibility

**Files:**
- Modify: `docs/model.md:42-60`
- Modify: `docs/shadow-operator.md:1-130`
- Modify: `docs/analysis-output-template.md:35-50`
- Modify: `README.md`
- Modify: `.superpowers/sdd/2026-08-05-historical-replay-authority/progress.md` (ignored execution ledger only)

**Interfaces:**
- Consumes: all v2 current and v1 replay behavior from Tasks 1-3.
- Produces: operator documentation that tells bundle producers to emit v2 for fresh pregame analysis and treats every v1 pregame path as exact replay only.

- [ ] **Step 1: Update user-facing bundle examples and workflow**

Replace projected/confirmed examples with:

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "confirmed"
}
```

State explicitly:

- fresh projected/confirmed commands require v2;
- review remains v1;
- v1 pregame artifacts work only from the managed root with an exact ledger
  replay;
- pending unexecuted v1 files must be regenerated;
- the model version remains `2026-08-03` and is independent from bundle schema
  version.

- [ ] **Step 2: Run documentation and manifest checks**

Run:

```bash
rg -n 'schema_version|authority|schema-v1|schema-v2' \
  README.md docs/model.md docs/shadow-operator.md \
  docs/analysis-output-template.md
.venv/bin/python -m analysis.pipeline --lineup confirmed
```

Expected: docs consistently describe v2 fresh execution and v1 replay. Pipeline
reports model `2026-08-03`, `fixed_model: true`, legacy decision authority,
Shadow mode, `may_change_recommendations: false`, the same-game assembler, and
`f5_market_engine_unchanged: true`.

- [ ] **Step 3: Run focused authority and assembler verification**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_schemas.py \
  tests/test_operator_artifact_fdio.py \
  tests/test_operator_pregame.py \
  tests/test_operator_cli.py \
  tests/test_model.py
```

Expected: all focused tests pass, including current empty/one/multiple candidate
v2 bundles, relaxed/strict v1 exact replay, laundering rejection, descriptor
authentication, F5 side plus F5 total coexistence, full-game side plus total
coexistence, and fixed family-winner behavior.

- [ ] **Step 4: Run the complete regression suite**

Run:

```bash
.venv/bin/python -m pytest -q
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 5: Run explicit authority probes**

For projected and confirmed stages, verify:

```python
assert current_type.model_validate(v2_mapping).authority == "current"
for historical in (
    relaxed_v1,
    strict_v1,
    relaxed_v1_empty,
    strict_v1_empty,
):
    with pytest.raises(ValidationError):
        current_type.model_validate(historical, from_attributes=True)
    with pytest.raises(ValidationError):
        current_type.model_validate_json(historical.model_dump_json())
```

Replay each v1 shape with an exact ledger and reject it with an empty ledger.
Snapshot ledger and managed bytes before every rejection and assert equality
afterward.

- [ ] **Step 6: Record final evidence and commit documentation**

Append exact RED/GREEN commands, test counts, commit hashes, independent review
status, and any adjudicated findings to the plan-owned ignored progress ledger.
Then commit tracked documentation only:

```bash
git add README.md docs/model.md docs/shadow-operator.md \
  docs/analysis-output-template.md
git diff --cached --check
git commit -m "docs: document pregame bundle authority v2"
```

- [ ] **Step 7: Request whole-branch review**

Generate a review package from commit `00c36cb` through the final implementation
HEAD. The reviewer must verify every acceptance criterion in
`docs/superpowers/specs/2026-08-06-pregame-bundle-authority-v2-design.md`, inspect
all authority entry points, and report Critical/Important/Minor findings before
the branch is offered for integration.
