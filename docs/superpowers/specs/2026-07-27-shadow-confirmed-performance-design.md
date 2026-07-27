# Shadow Confirmed Performance Design

**Date:** 2026-07-27
**Status:** Approved direction, implementation pending

## Goal

Reduce confirmed-lineup freeze latency without changing any baseball input,
model weight, threshold, market decision, simulation count, random seed,
forecast probability, review classification, or calibration output.

The optimized path must preserve the existing append-only ledger as the source
of truth and must continue reading all historical records written by the
current format.

## Measured Baseline

The production-size ledger contains 458 records and is approximately 154 MB.
One strict `LedgerStore.iter_records()` pass takes 42.96 seconds and peaks at
approximately 1.50 GB RSS. A confirmed run invokes `_read_records` five times,
while one 20,000-game forecast takes 6.39 seconds and approximately 61 MB RSS.

Forecast records account for approximately 146.5 MB of the ledger. One
forecast stores 560,000 scalar component-distribution values and averages
1.24 MB. Removing only those raw arrays would reduce a representative forecast
record by 96.3%, but review requires the exact arrays, so they must be retained
losslessly rather than discarded.

## Invariants

- Keep `SimulationConfig.simulations == 20_000`.
- Keep deterministic forecast IDs, seeds, market probabilities, intervals,
  evidence paths, and legacy comparisons unchanged.
- Keep the current model and source hierarchy authoritative.
- Keep all historical JSONL records readable without migration or rewriting.
- Keep the JSONL ledger append-only and authoritative.
- Keep exact projected/confirmed capture lineage and idempotent replay.
- Keep crash recovery conservative. A stale or missing acceleration artifact
  must cause a safe rebuild, never an inferred success.
- Keep Savant historical data on the existing daily cache policy.
- Keep MLB schedule/probables, RotoWire lineups, market prices, and
  boxscore/bullpen usage live at each analysis.
- Do not introduce automatic model tuning, online learning, or result-aware
  pregame behavior.

## Scope

This delivery covers the confirmed/projected shadow persistence path:

1. A rebuildable, non-authoritative ledger checkpoint index.
2. A single pregame ledger session for lookup and append.
3. Exact replay detection before the expensive simulation.
4. Lossless compact encoding for new component distributions.
5. Performance and equivalence tests against the existing path.

CPU-parallel full-slate forecasts and a first-class live collector are deferred
until this storage foundation is verified. They are independent optimizations
and must not be mixed into the same correctness change.

## Architecture

### Rebuildable Ledger Checkpoint

Add `analysis/ledger/checkpoint.py`. A `LedgerCheckpoint` is a compact,
versioned JSON artifact stored adjacent to the ledger. It records:

- Ledger device, inode, byte length, nanosecond mtime, and SHA-256.
- Record count and latest append timestamp.
- Every record envelope's kind, record ID, idempotency key, payload hash, byte
  offset, and byte length.
- Capture metadata needed for active-head and supersession selection.
- Forecast run IDs and normalized input hashes.
- Capture-to-forecast link identities.

The checkpoint is derived state, never authority. On first use, after deletion,
or whenever ledger identity, size, mtime, or SHA-256 differs, the implementation
strictly reads and validates the complete ledger using the existing
`_read_records` path, then rebuilds the checkpoint from the validated records
and raw byte offsets.

On a matching checkpoint, pregame lookup loads only the compact index and
deserializes exact referenced records by their verified byte offsets. The
ledger SHA-256 is recalculated before trusting a checkpoint. This preserves
tamper/corruption detection while avoiding reconstruction of every historical
component array.

Checkpoint writes use the repository's descriptor-relative, no-follow,
atomic-replace patterns. A missing, malformed, symlinked, stale, or
contradictory checkpoint is ignored and rebuilt from the authoritative ledger.

### Pregame Ledger Session

Add a focused `PregameLedgerSession` in `analysis/ledger/pregame.py`.

The session:

- Acquires the existing ledger lock.
- Loads or rebuilds the checkpoint once.
- Resolves the exact capture key, current active capture, forecast ID, and
  exact link from indexed metadata.
- Reads exact replay records by offset when needed.
- Appends capture, forecast, and link under one exclusive session rather than
  invoking three independent full-ledger transactions.
- Fsyncs the authoritative ledger before atomically publishing the updated
  checkpoint.

If the process stops after ledger fsync but before checkpoint publication, the
next command observes a checkpoint/ledger fingerprint mismatch and performs a
strict rebuild. Existing partial-progress recovery remains authoritative.

The public `LedgerStore` methods and generic append behavior remain available
for review and compatibility. `record_shadow_run` uses the optimized atomic
pregame interface when the store provides it and falls back to the existing
three-method interface for test doubles and third-party callers.

### Replay Before Simulation

The operator can derive the deterministic forecast run ID from normalized game
input and immutable config without running the simulation.

Before `run_forecast`, it opens one pregame session and checks for an exact
capture, matching forecast, and matching link. Only a complete exact replay may
skip simulation. A new capture, changed market semantics, changed provenance,
changed lineup, changed legacy comparison, or incomplete relationship still
runs the complete 20,000-game forecast.

This restriction avoids reusing a forecast record whose stored
`shadow_comparison` belongs to different legacy input.

### Lossless Distribution Encoding

New forecast records compact each component distribution's `values` list using
deterministic `zlib` compression over canonical UTF-8 JSON and base64 encoding.
Each compact distribution contains:

```json
{
  "component_id": "away_total_runs",
  "summary": {},
  "values_encoding": "zlib-base64-canonical-json-v1",
  "values_count": 20000,
  "values_data": "..."
}
```

The canonical in-memory `ForecastOutput` is unchanged. Compression occurs only
when constructing the ledger payload. Review deserialization accepts both the
historical `values` list and the new compact representation, decodes the exact
list, validates its count, scalar domains, and summary, and constructs the
same immutable `ComponentDistribution`.

No quantization, rounding, sampling, histogram replacement, or value deletion
is allowed. Tests compare decoded values, summaries, reviews, and report bytes
with the uncompressed representation.

## Data Flow

```text
confirmed bundle
  -> strict bundle and pregame validation
  -> deterministic forecast_run_id preflight
  -> PregameLedgerSession
       -> validate/rebuild checkpoint
       -> exact replay?
            yes -> load exact capture/forecast/link, skip simulation
            no  -> run unchanged 20,000-game forecast
                 -> losslessly compact ledger-only distributions
                 -> append capture + forecast + link in one session
                 -> fsync ledger
                 -> atomically publish checkpoint
  -> persist immutable managed bundle
  -> return the same public PregameRunResult
```

## Error Handling

- Checkpoint failure never makes the ledger unavailable if the ledger is
  valid; rebuild and continue.
- Ledger corruption continues to raise `LedgerCorruptionError`.
- Blob/compact decoding errors are ledger corruption, not missing-data flags.
- A checkpoint whose offsets do not resolve to the indexed record is stale or
  corrupt and triggers a strict rebuild.
- Concurrent writers continue using the current lock and active-head CAS
  semantics.
- A changed ledger observed after preflight retries through the existing
  bounded head-change loop.

## Testing

Required tests:

- A production-like forecast round-trips through compact encoding with exact
  type and value equality.
- Historical uncompressed forecast records remain readable.
- Compact and uncompressed contexts produce byte-identical review and
  calibration mappings.
- Exact confirmed replay does not call `run_forecast`.
- A changed lineup or market still calls the full forecast.
- One optimized confirmed write performs one checkpoint validation/session and
  appends the same capture, forecast, and link semantics as the old path.
- Deleting or corrupting the checkpoint rebuilds it from the ledger.
- Modifying the ledger invalidates the checkpoint.
- Interrupted checkpoint publication leaves the ledger recoverable.
- Symlink and path-replacement attacks remain rejected.
- Concurrency and active-head tests continue to pass.
- A benchmark fixture demonstrates that a warm confirmed replay does not call
  full `_read_records` and that new compact forecast lines are materially
  smaller than historical lines.

## Acceptance Criteria

- All existing tests pass without changed model expectations.
- New equivalence, replay, checkpoint, corruption, and concurrency tests pass.
- The existing 458-record ledger can build a checkpoint without rewriting any
  ledger bytes.
- After the one-time checkpoint build, a confirmed run does not reconstruct
  all historical forecast distributions.
- Exact replay skips the 20,000-game simulation.
- New forecast records retain all 20,000 values for every component and decode
  exactly.
- A warm confirmed run targets less than 10 seconds of local compute for one
  changed game, excluding live-source network latency and response writing.

