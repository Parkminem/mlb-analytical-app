# Pregame Batch Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fast, bounded-parallel projected and confirmed batch commands that emit an auditable `N/N` receipt and fail closed when any capture/forecast/link triplet is missing.

**Architecture:** A new `analysis.operator.batch` module strictly loads a manifest and canonical current bundles, dispatches the unchanged single-game Operator service in worker processes, audits returned identities once against a fresh ledger checkpoint, and atomically publishes a derived receipt. Existing projected, confirmed, review, forecasting, and append-only ledger behavior remains authoritative and unchanged.

**Tech Stack:** Python 3.14, Pydantic 2, `concurrent.futures.ProcessPoolExecutor`, Typer, append-only JSONL, pytest.

## Global Constraints

- Keep legacy MLB model version `2026-08-05` unchanged.
- Keep `SimulationConfig.simulations == 20_000` and preserve deterministic seeds, probabilities, intervals, review classifications, and calibration output.
- Accept only fresh schema-v2/current projected or confirmed bundles in batch mode.
- Keep historical schema-v1 replay on the existing single-game commands.
- Parse every manifest and bundle before starting a worker or writing the ledger.
- Default to three workers and reject values outside `1..4`.
- Keep the ledger authoritative and the receipt rebuildable.
- Do not roll back successful immutable records when another batch entry fails.
- Use exact replay for safe retry and never infer missing IDs.

---

### Task 1: Strict Batch Manifest and Receipt Contracts

**Files:**
- Create: `analysis/operator/batch.py`
- Create: `tests/test_operator_batch.py`

**Interfaces:**
- Produces: `PregameBatchManifest`, `PregameBatchEntry`, `PreparedBatchEntry`, `PregameBatchSuccess`, `PregameBatchFailure`, `PregameBatchResult`.
- Produces: `load_pregame_batch_manifest(path: str | Path, *, expected_stage: str) -> PregameBatchManifest`.
- Produces: `prepare_pregame_batch(manifest: PregameBatchManifest, *, manifest_path: str | Path) -> tuple[PreparedBatchEntry, ...]`.
- Consumes: public schema-v2/current parsers from `analysis.operator.schemas` and canonical JSON from `analysis.forecasting.identity`.

- [ ] **Step 1: Write failing strict-manifest tests**

Add tests that write canonical JSON manifests and assert exact stage, positive
unique `game_pk`, non-empty unique bundle paths, no unknown fields, and relative
path resolution against the manifest directory. A representative test is:

```python
def test_prepare_batch_reads_and_canonicalizes_every_bundle_before_work(tmp_path):
    first = write_current_bundle(tmp_path, game_pk=900001, stage="confirmed")
    second = write_current_bundle(tmp_path, game_pk=900002, stage="confirmed")
    manifest_path = write_manifest(
        tmp_path,
        stage="confirmed",
        entries=((900001, first.name), (900002, second.name)),
    )

    manifest = load_pregame_batch_manifest(
        manifest_path,
        expected_stage="confirmed",
    )
    prepared = prepare_pregame_batch(manifest, manifest_path=manifest_path)

    assert tuple(item.game_pk for item in prepared) == (900001, 900002)
    assert all(item.bundle_path.is_absolute() for item in prepared)
    assert all(item.canonical_bundle.endswith(b"\n") is False for item in prepared)
```

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_batch.py -k "manifest or prepare" -q
```

Expected: collection fails because `analysis.operator.batch` does not exist.

- [ ] **Step 3: Implement strict manifest parsing and bundle preparation**

Use frozen Pydantic models with `strict=True`, `extra="forbid"`, and exact
`schema_version: Literal[1]`. Resolve paths without following or mutating input
files, read each file once, parse with `parse_projected_bundle_json` or
`parse_confirmed_bundle_json`, require `schema_version == 2`,
`authority == "current"`, exact stage and `game_pk`, then store canonical bytes
in `PreparedBatchEntry`.

- [ ] **Step 4: Add failing receipt invariants tests**

Test that `PregameBatchResult` rejects duplicate/missing result game IDs,
incorrect counts, `complete=True` with failures, and result order different
from manifest order. Also test canonical mapping output contains every success
triplet and every failure category.

- [ ] **Step 5: Implement immutable result and receipt mappings**

Implement result dataclasses whose `__post_init__` checks exact target coverage,
unique games, count conservation, and completion consistency. Provide
`to_mapping()` methods that preserve manifest order and JSON scalar types.

- [ ] **Step 6: Run Task 1 tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_batch.py -k "manifest or prepare or receipt" -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add analysis/operator/batch.py tests/test_operator_batch.py
git commit -m "feat: define strict pregame batch contracts"
```

---

### Task 2: Bounded Parallel Execution, Coverage Audit, and Atomic Receipt

**Files:**
- Modify: `analysis/operator/batch.py`
- Modify: `tests/test_operator_batch.py`

**Interfaces:**
- Produces: `run_pregame_batch(manifest_path: str | Path, *, expected_stage: str, ledger_path: str | Path, bundle_root: str | Path, receipt_path: str | Path, max_workers: int = 3, registry_root: str | Path | None = None) -> PregameBatchResult`.
- Produces: `audit_pregame_batch(result: PregameBatchResult, *, ledger_path: str | Path) -> PregameBatchResult`.
- Consumes: `run_projected`, `run_confirmed`, `LedgerStore.pregame_checkpoint()`, and the immutable prepared bundle bytes from Task 1.

- [ ] **Step 1: Write a failing two-game execution test**

Build two distinct current confirmed bundles with a test forecast registry using
12 simulations. Run with two workers against one temporary ledger and assert:

```python
result = run_pregame_batch(
    manifest_path,
    expected_stage="confirmed",
    ledger_path=ledger,
    bundle_root=bundle_root,
    receipt_path=receipt,
    registry_root=registry,
    max_workers=2,
)

assert result.complete is True
assert result.success_count == 2
assert result.failure_count == 0
assert receipt_mapping(receipt) == result.to_mapping()
assert {item.game_pk for item in result.successes} == {900001, 900002}
```

- [ ] **Step 2: Run the execution test and verify red**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_batch.py -k "two_game_execution" -q
```

Expected: FAIL because `run_pregame_batch` is not implemented.

- [ ] **Step 3: Implement the process worker and bounded scheduler**

Add one module-level picklable worker that receives stage, canonical bytes, and
resolved runtime paths, reparses the correct current bundle type, calls the
existing single-game service, and returns only serializable success or failure
data. Use `ProcessPoolExecutor(max_workers=max_workers)` and collect all futures
without cancelling successful immutable writes after another item fails.
Validate `max_workers` before preparing any work.

- [ ] **Step 4: Write failing exact-replay and partial-failure tests**

Run the same complete manifest twice and assert every second-run ID matches and
all replay flags are true. Add a manifest where one current bundle passes
static parsing but fails a service precondition; assert the other game remains
successful, `complete` is false, counts are `1/1`, the receipt exists, and an
exact rerun can reuse the successful triplet.

- [ ] **Step 5: Implement one-pass ledger coverage audit**

Load one fresh `LedgerCheckpoint` after workers finish. For each success,
require:

```python
capture = checkpoint.record("capture", success.capture_id)
forecast = checkpoint.record("forecast", success.forecast_run_id)
link = checkpoint.record(
    "capture_forecast_link",
    success.capture_forecast_link_record_id,
)
```

Require capture `game_pk`, link `capture_id`, and link `forecast_run_id` to
match the success row. Convert an invalid success to a stable
`coverage_audit` failure. Run the audit once per batch, never once per game.

- [ ] **Step 6: Write failing atomic-publication and validation tests**

Assert the target receipt is absent when manifest preparation fails, contains
canonical JSON plus one trailing newline after a partial or complete execution,
and is never observed with partial bytes when `os.replace` succeeds. Assert
worker counts `0` and `5` fail before execution.

- [ ] **Step 7: Implement atomic receipt publication**

Create the receipt directory, write canonical bytes to a same-directory
exclusive temporary file, flush and `fsync`, replace the destination atomically,
and clean up the temporary path on failure. Raise `OperatorIOError` if the
receipt cannot be published without changing successful ledger records.

- [ ] **Step 8: Run Task 2 and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_batch.py tests/test_operator_pregame.py tests/test_shadow_performance_contract.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add analysis/operator/batch.py tests/test_operator_batch.py
git commit -m "feat: run complete pregame batches"
```

---

### Task 3: CLI Receipt Contract and Operations Documentation

**Files:**
- Modify: `analysis/operator/cli.py`
- Modify: `tests/test_operator_cli.py`
- Modify: `README.md`
- Modify: `docs/model.md`
- Modify: `docs/shadow-operator.md`

**Interfaces:**
- Produces CLI commands: `batch-projected` and `batch-confirmed`.
- Consumes: `run_pregame_batch` from Task 2.

- [ ] **Step 1: Write failing CLI tests**

Monkeypatch `run_pregame_batch` to return complete and partial results. Assert
the complete command exits `0` and prints `complete: true`, `success_count`,
and per-game IDs. Assert a partial result prints `complete: false`, failure
details, and exits with code `7`. Assert invalid worker counts are rejected by
Typer before service invocation.

- [ ] **Step 2: Run CLI tests and verify red**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_cli.py -k "batch_projected or batch_confirmed" -q
```

Expected: FAIL because the batch commands do not exist.

- [ ] **Step 3: Add both Typer commands and concise output**

Each command accepts `--manifest`, `--ledger`, `--bundle-root`, `--receipt`, and
`--max-workers`. Invoke the service through the existing concise error boundary.
Print the batch summary and ordered success/failure rows. Raise a stable
`typer.Exit(code=7)` after printing a partial receipt; do not mislabel it as an
internal error.

- [ ] **Step 4: Update operational documentation**

Document manifest shape, bounded process behavior, `N/N` completion rule,
receipt fields, nonzero partial exit, and idempotent resume. State explicitly
that analysis responses may say `동결 완료` only for a complete receipt and
that the new commands do not alter model `2026-08-05` or the three-stage source
hierarchy.

- [ ] **Step 5: Run focused and full verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_batch.py tests/test_operator_cli.py tests/test_operator_pregame.py tests/test_shadow_performance_contract.py -q
.venv/bin/python -m pytest -q
python3 -m analysis.pipeline
```

Expected: all tests pass and the pipeline reports the unchanged fixed model.

- [ ] **Step 6: Commit Task 3**

```bash
git add analysis/operator/cli.py tests/test_operator_cli.py README.md docs/model.md docs/shadow-operator.md
git commit -m "feat: expose complete pregame batch commands"
```
