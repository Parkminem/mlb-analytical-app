# Pregame Batch Completeness Design

**Date:** 2026-08-06
**Status:** Approved

## Goal

Record every requested projected- or confirmed-lineup game through the existing
Operator without allowing a narrative analysis to be described as completely
frozen when one or more capture, forecast, or link records are missing. Reduce
wall-clock latency with bounded process parallelism while preserving the fixed
MLB model, 20,000 simulations, source hierarchy, deterministic forecast output,
and append-only ledger semantics.

## Scope

This change adds a batch execution and receipt layer around already prepared
schema-v2/current pregame bundles. It does not collect live sources, generate
lineups, change analysis weights, change simulation counts, or infer missing
bundle content. Live MLB, RotoWire, market, and bullpen acquisition remains the
responsibility of the existing external preparation workflow.

## Interface

The Operator exposes two commands:

```text
python -m analysis.operator batch-projected --manifest <path>
python -m analysis.operator batch-confirmed --manifest <path>
```

Both commands accept the existing `--ledger` and `--bundle-root` overrides,
plus `--receipt` and `--max-workers`. The default worker count is `3`; valid
values are `1` through `4`.

The strict manifest contract is:

```json
{
  "schema_version": 1,
  "stage": "confirmed",
  "entries": [
    {
      "game_pk": 824806,
      "bundle": "bundles/824806-confirmed.json"
    }
  ]
}
```

Manifest entries must be non-empty and have unique positive `game_pk` values
and unique bundle paths. Relative bundle paths resolve against the manifest
directory. Every bundle is read once by the parent, strictly parsed as
schema-v2/current, matched to the manifest stage and `game_pk`, and converted
to canonical bytes before any worker starts. Historical schema-v1 artifacts
remain available only through the existing single-game replay commands.

## Execution Model

The parent performs complete static manifest and bundle validation before any
ledger write. It then dispatches canonical bundle bytes to at most three worker
processes by default. Each worker reparses those immutable bytes through the
public current-bundle parser and invokes the existing `run_projected` or
`run_confirmed` service. The worker does not contain a second forecast or
ledger implementation.

The existing single-game service continues to own deterministic forecast ID
preflight, exact replay, the unchanged 20,000-game simulation, review-eligibility
validation, active-head CAS, atomic capture/forecast/link append, and immutable
managed-bundle persistence. Process workers may forecast concurrently; ledger
locks serialize the short authoritative append sections. Per-game execution
is idempotent, so a partially successful batch can be rerun safely.

## Completeness and Receipt

Every target has exactly one result row. A success row contains:

- `game_pk`
- canonical input bundle path
- `capture_id`
- `forecast_run_id`
- `capture_forecast_link_record_id`
- managed bundle path and hash
- capture, forecast, and link replay flags

A failure row contains the stable Operator error category, concise message,
exit code, and any authoritative partial-progress IDs. Unexpected worker
failures are recorded as internal failures without exposing tracebacks in the
normal CLI output.

After all workers return, the parent loads one fresh ledger checkpoint and
verifies that every reported capture, forecast, and link identity exists, that
the capture belongs to the expected game, and that the link references the
reported capture and forecast. A failed audit converts that row to failure.

The receipt contains the stage, target count, success count, failure count,
completion flag, worker count, and ordered per-game rows. It is written as
canonical JSON through an atomic same-directory replacement. The append-only
ledger remains authoritative; the receipt is a rebuildable operational
artifact.

The command exits `0` only when all target games validate successfully. Any
failure exits nonzero after writing the partial receipt. Successful rows are
not rolled back because ledger records are immutable and append-only. An exact
rerun returns the original records and can complete the receipt.

## Output Contract

CLI output begins with a batch summary including `complete`, `target_count`,
`success_count`, `failure_count`, and `receipt_path`, followed by one compact
row per game. Downstream responses may say `동결 완료` only when the receipt
has `complete: true` and `success_count == target_count`. Otherwise they must
say `부분 동결` and include each failed game and reason.

## Performance

The batch layer does not rerun the baseball analysis and does not run a second
Shadow forecast. It uses the same canonical bundle as the single source for
the Operator execution. Exact replays continue to skip simulation. Changed
games retain all 20,000 simulations and run with bounded process parallelism
because the simulation is CPU-bound Python code.

The maximum worker count of four limits memory and I/O amplification. The
default of three targets approximately three simulation waves for a typical
seven- to nine-game request instead of seven to nine sequential simulations.
Only one checkpoint coverage audit runs after the batch. No per-game full
ledger reconstruction is introduced.

## Failure and Recovery

- Invalid manifest or bundle input fails before workers start and writes no
  ledger records or receipt.
- Per-game Operator failures produce a partial receipt and a nonzero exit.
- Process crashes produce a stable internal failure row.
- Receipt-write failure does not invalidate successful ledger records and is
  reported as an I/O failure with rerun-safe guidance.
- The next exact batch rerun reuses complete records and retries only states
  that were not completed.
- Concurrent updates remain governed by the existing ledger lock and
  active-head compare-and-swap behavior.

## Testing

Tests must prove:

- strict manifest validation and path resolution;
- stage, `game_pk`, schema-v2, and current-authority checks occur before work;
- duplicate games and bundle paths are rejected;
- two changed games can execute with bounded process workers against one
  ledger and produce complete audited receipts;
- exact batch replay preserves all IDs and reports replay flags;
- one failed game produces a partial receipt and nonzero CLI exit while
  preserving successful records;
- coverage audit rejects missing or mismatched triplet identities;
- atomic receipt publication leaves no partial target file;
- `--max-workers` accepts only `1` through `4`;
- existing single-game projected, confirmed, review, simulation, and ledger
  tests remain unchanged and pass.

## Non-Goals

- No live data collector or RotoWire browser automation.
- No automatic table or prose generation from the receipt.
- No reduction in simulation count or forecast fidelity.
- No model-weight, threshold, priority, evidence-path, or calibration change.
- No transactional rollback across games.
