# Historical Pregame Replay Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate historical schema-v1 replay from normal pregame execution so omitted compatibility data can resolve only an exact existing ledger replay and can never create fresh records or strict-looking artifacts.

**Architecture:** Normal projected and confirmed runners accept only strict current bundles. New stage-specific replay services accept a managed artifact path, authenticate and parse it inside the service boundary, look up the exact frozen capture/forecast/link, and either return that replay with the original managed provenance or reject without forecasting, appending, or persisting.

**Tech Stack:** Python 3.14, frozen dataclasses, Pydantic bundle schemas, append-only JSONL ledger, pytest/unittest.

## Global Constraints

- Keep model version `2026-08-03` and schema version 1 unchanged.
- Do not change baseball scoring, score caps, F5/F5 O/U engines, market engines, source hierarchy, or Shadow authority.
- Fresh projected/confirmed bundles must contain a real boolean `same_game_pair_compatible`.
- Historical defaulting is permitted only while resolving an exact existing replay.
- Historical replay must not call forecasting, append ledger records, or persist a new managed artifact.
- Apply manual edits with `apply_patch` and preserve unrelated worktree changes.

---

### Task 1: Isolate Historical Replay In The Service Layer

**Files:**
- Modify: `analysis/operator/bundles.py`
- Modify: `analysis/operator/service.py`
- Modify: `analysis/operator/__init__.py`
- Test: `tests/test_operator_pregame.py`

**Interfaces:**
- Consumes: `load_historical_managed_bundle(path, root=...)`, `_existing_exact_pregame_replay(...)`, and the existing `PregameRunResult`.
- Produces: `replay_historical_projected(path, *, ledger_path, bundle_root, registry_root=None) -> PregameRunResult` and `replay_historical_confirmed(...) -> PregameRunResult`.
- Produces: strict runtime contracts for `run_projected(ProjectedBundle, ...)` and `run_confirmed(ConfirmedBundle, ...)` that reject historical compatibility objects.

- [ ] **Step 1: Write failing normal-runner boundary tests**

Add a projected/confirmed parameterized test that constructs or loads a historical compatibility artifact and passes it directly to the normal runner:

```python
with pytest.raises(
    TypeError,
    match="normal pregame runner requires a strict current bundle",
):
    runner(
        historical_artifact,
        ledger_path=empty_ledger,
        bundle_root=bundles,
    )
assert not empty_ledger.exists()
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_pregame.py::test_normal_runner_rejects_historical_artifact -q
```

Expected: FAIL because the current normal runner accepts the compatibility wrapper.

- [ ] **Step 3: Write failing dedicated replay tests**

Add projected/confirmed tests proving the new service contract:

```python
replay = historical_replay(
    source_path,
    ledger_path=ledger,
    bundle_root=bundles,
)
assert replay.capture == original.capture
assert replay.forecast == original.forecast
assert replay.link == original.link
assert replay.managed_bundle.path == source_path.resolve()
assert replay.managed_bundle.replayed is True
```

Add the empty-ledger rejection and side-effect assertions:

```python
with pytest.raises(
    OperatorPreconditionError,
    match="historical replay requires an exact existing replay",
):
    historical_replay(
        source_path,
        ledger_path=empty_ledger,
        bundle_root=bundles,
    )
assert not empty_ledger.exists()
assert managed_files_after == managed_files_before
```

Patch `run_forecast`, `record_shadow_run`, and `persist_managed_bundle` to raise if invoked by the historical replay service so the test proves the forbidden paths are unreachable.

- [ ] **Step 4: Run the replay tests and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_pregame.py -k 'historical and (dedicated or normal_runner)' -q
```

Expected: FAIL because the dedicated replay functions do not exist and the normal service still owns the compatibility path.

- [ ] **Step 5: Implement the strict normal-runner boundary**

Keep `run_projected` and `run_confirmed` typed to strict bundles and add an exact runtime type guard before `_run_pregame`:

```python
def run_projected(bundle: ProjectedBundle, *, ...) -> PregameRunResult:
    if type(bundle) is not ProjectedBundle:
        raise TypeError(
            "normal pregame runner requires a strict current ProjectedBundle"
        )
    return _run_pregame(bundle, expected_stage="projected", ...)
```

Apply the corresponding `ConfirmedBundle` guard. Remove replay-wrapper handling and replay-only branching from `_run_pregame`; it must again operate only on a strict `_Bundle`.

- [ ] **Step 6: Implement path-authenticated historical replay services**

Create one private shared service and two public stage wrappers:

```python
def _replay_historical_pregame(
    path: str | Path,
    *,
    expected_stage: str,
    expected_type: type[ProjectedBundle] | type[ConfirmedBundle],
    ledger_path: str | Path,
    bundle_root: str | Path,
    registry_root: str | Path | None,
) -> PregameRunResult:
    artifact = load_historical_managed_bundle(path, root=bundle_root)
    if not isinstance(artifact, ReplayOnlyPregameBundle):
        raise OperatorPreconditionError(
            "historical replay requires a pre-change managed bundle"
        )
    bundle = artifact.bundle
    if type(bundle) is not expected_type or bundle.stage != expected_stage:
        raise OperatorPreconditionError("historical replay stage/type mismatch")
    # Derive immutable identities from the authenticated bundle and perform
    # only _existing_exact_pregame_replay.
```

If exact replay is absent, raise before opening an append-capable store. If it exists, construct `PregameRunResult` directly from the exact records and `artifact.managed_bundle`; do not call `persist_managed_bundle`.

- [ ] **Step 7: Export the dedicated service functions**

Expose only the two path-based replay functions from `analysis/operator/__init__.py`. Do not export a general runner that accepts caller-provided provenance metadata.

- [ ] **Step 8: Run focused service tests and verify GREEN**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_pregame.py -k 'historical or normal_runner' -q
```

Expected: all selected tests PASS for projected and confirmed stages.

- [ ] **Step 9: Commit Task 1**

```bash
git add analysis/operator/bundles.py analysis/operator/service.py \
  analysis/operator/__init__.py tests/test_operator_pregame.py
git commit -m "fix: isolate historical pregame replay"
```

### Task 2: Route CLI Historical Artifacts Only To Replay

**Files:**
- Modify: `analysis/operator/cli.py`
- Test: `tests/test_operator_cli.py`

**Interfaces:**
- Consumes: the strict `run_projected`/`run_confirmed` functions and the new `replay_historical_projected`/`replay_historical_confirmed` path-based services from Task 1.
- Produces: a CLI stage-load result that distinguishes strict current input from a historical managed path without passing compatibility metadata to a normal runner.

- [ ] **Step 1: Write failing CLI dispatch tests**

Add projected/confirmed tests with patched service calls:

```python
def fake_historical(path, *, ledger_path, bundle_root, registry_root=None):
    calls.append((path, ledger_path, bundle_root))
    return pregame_result

def forbidden_normal(*args, **kwargs):
    raise AssertionError("historical input must not reach normal runner")
```

Assert a managed pre-change path invokes only the dedicated historical service. Assert a strict current path invokes only the normal runner.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_cli.py -k 'historical and dispatch' -q
```

Expected: FAIL because current CLI passes the compatibility object to the normal runner.

- [ ] **Step 3: Implement explicit CLI load and dispatch states**

Introduce a private immutable result that carries either a strict bundle or the original historical path, never caller-supplied provenance metadata:

```python
@dataclass(frozen=True)
class _PregameBundleLoad:
    bundle: ProjectedBundle | ConfirmedBundle | None
    historical_path: Path | None
```

The loader first attempts strict parsing. For a managed pre-change artifact it uses the historical loader only to classify and validate the path, then returns `historical_path=resolved_path`. Projected and confirmed command handlers call the dedicated replay service for that state; otherwise they call the normal strict runner. Review loading remains on the existing strict path.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_cli.py -q
```

Expected: all CLI tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add analysis/operator/cli.py tests/test_operator_cli.py
git commit -m "fix: dispatch historical bundles to replay only"
```

### Task 3: Integration And Authority Regression

**Files:**
- Modify: `tests/test_operator_pregame.py`
- Modify: `tests/test_operator_cli.py`
- Modify: `.superpowers/sdd/2026-08-05-historical-replay-authority/progress.md`

**Interfaces:**
- Consumes: all strict and historical replay interfaces from Tasks 1 and 2.
- Produces: end-to-end proof that the authority boundary cannot be bypassed by path, hash, metadata, stage, or runner substitution.

- [ ] **Step 1: Add the end-to-end substitution regression**

For both stages, start from a genuine two-market historical exact replay. Then verify:

```python
returned = historical_replay(source, ledger_path=ledger, bundle_root=bundles)
assert returned.managed_bundle.path == source.resolve()

with pytest.raises(TypeError):
    normal_runner(
        caller_constructed_historical_artifact,
        ledger_path=empty_ledger,
        bundle_root=bundles,
    )

with pytest.raises(OperatorPreconditionError):
    historical_replay(
        strict_managed_path,
        ledger_path=empty_ledger,
        bundle_root=bundles,
    )
```

Assert the empty ledger and managed file set remain unchanged. Verify the historical service recomputes the full SHA-256 from source bytes rather than returning any caller value.

- [ ] **Step 2: Run focused authority tests**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest \
  tests/test_operator_pregame.py tests/test_operator_cli.py \
  -k 'historical or provenance or normal_runner' -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run the complete regression suite**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Verify the fixed manifest and worktree**

Run:

```bash
/Users/minhyeok/workspaces/mlb-analytical-app/.venv/bin/python \
  -m analysis.pipeline --lineup confirmed
git diff --check
git status --short
```

Expected: pipeline exits 0 with model version `2026-08-03`, fixed legacy authority, unchanged F5 policy, and Shadow non-authority; diff check emits no output.

- [ ] **Step 5: Update the SDD progress ledger and commit**

Record RED/GREEN evidence, exact test counts, review status, and commit SHAs in `.superpowers/sdd/2026-08-05-historical-replay-authority/progress.md` using `apply_patch`, then commit only tracked product/test changes if the SDD file is ignored:

```bash
git add tests/test_operator_pregame.py tests/test_operator_cli.py
git commit -m "test: lock historical replay authority boundary"
```

- [ ] **Step 6: Request final scoped code review**

Generate a review package from the design-plan base to HEAD and require an independent reviewer to verify the dedicated API cannot forecast, append, persist, or trust caller provenance. Resolve every Important or Critical finding before integration.
