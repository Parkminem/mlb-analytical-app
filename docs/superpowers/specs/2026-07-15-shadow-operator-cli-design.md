# Shadow Operator CLI Design

**Date:** 2026-07-15
**Status:** Approved direction, implementation pending

## Goal

Provide one safe, repeatable operator entry point for the daily three-stage
shadow workflow:

1. Record a projected-lineup pregame capture and forecast.
2. Record and freeze a confirmed-lineup pregame capture and forecast.
3. Ingest the official result, record objective reviews, and refresh the
   calibration report.

The operator consumes versioned per-game JSON bundles prepared by the normal
analysis process. It does not scrape MLB, RotoWire, Baseball Savant, or market
sites. Existing fixed-model decisions remain the only production authority.

## Non-Goals

- No live source collection or browser automation.
- No automatic model tuning, online learning, or production promotion.
- No change to legacy `play` / `lean` / `pass`, priority scores, or final
  rankings.
- No mutable database or rewrite of existing ledger records.
- No interactive entry of hitter, pitch, bullpen, or market fields.

## User Workflow

The user continues to issue natural-language requests. Codex prepares the
versioned JSON bundle and invokes the matching CLI command.

```text
python3 -m analysis.operator projected --bundle <path>
python3 -m analysis.operator confirmed --bundle <path>
python3 -m analysis.operator review --bundle <path>
```

Default runtime paths:

```text
data/ledger/shadow.jsonl
data/shadow/bundles/<game_pk>/
reports/private/shadow/calibration-latest.json
reports/private/shadow/calibration-latest.md
```

All runtime paths are already covered by Git ignore rules. The ledger remains
the authoritative append-only state. Bundle copies and reports are derived
operator artifacts.

## Architecture

Create a small `analysis.operator` package with explicit boundaries:

- `schemas.py`: versioned external JSON schemas and strict decoding into the
  existing forecasting, legacy-decision, result, and market-observation value
  objects.
- `bundles.py`: canonical JSON loading, content hashing, immutable managed
  bundle copies, and schema-version checks.
- `service.py`: stage orchestration over `run_forecast`, `record_shadow_run`,
  `LedgerStore`, `build_and_record_review`, `select_primary_rows`, and
  `build_calibration_report`.
- `reporting.py`: deterministic JSON and Korean Markdown calibration output.
- `cli.py` and `__main__.py`: Typer commands, path options, concise terminal
  summaries, and stable exit codes.

The operator layer must reuse existing domain validation. It may add external
schema validation and orchestration invariants, but it must not duplicate or
reweight baseball evidence.

## Versioned Bundle Contracts

Every bundle has these envelope fields:

```json
{
  "schema_version": 1,
  "stage": "projected | confirmed | review",
  "game_pk": 123456,
  "created_at": "2026-07-15T01:00:00Z",
  "payload": {}
}
```

Unknown envelope or payload fields are rejected. Timestamps require explicit
offsets and are normalized to UTC for canonical hashing.

### Projected and Confirmed Payload

The pregame payload contains:

- `game_input`: the complete `NormalizedGameInput` tree, including provenance,
  pitcher chain, all lineup slots, pitch/velocity responses, bullpen segments,
  and markets.
- `legacy_decision`: the complete frozen `LegacyDecisionSnapshot`, including
  exact candidates, decisions, evidence paths, final priority, and review
  bindings.

The CLI runs `run_forecast(game_input)` itself. A caller cannot provide a
precomputed shadow forecast.

Projected-stage requirements:

- At least one lineup is projected.
- `captured_at` is strictly before first pitch.
- The bundle stage and lineup state agree.

Confirmed-stage requirements:

- Both lineups are confirmed.
- `captured_at` is strictly before first pitch.
- The operator supersedes the latest active projected or older confirmed
  capture for the same game when the semantic state changed.
- The confirmed capture and deterministic forecast are the frozen primary
  pregame state used by the review command.

### Review Payload

The review payload contains:

- Official MLB result source identity and revision.
- F5 and full-game scores.
- Available component observations and documented external events.
- Optional closing market observations with decimal prices and observation
  timestamps.

The result contract reuses `OfficialResultCommand`, including score and
pitching-run conservation validation. Corrections append a new immutable
revision that supersedes the previous official result.

## Stage Semantics

### `projected`

1. Load and strictly validate the bundle.
2. Require projected lineup state and a pre-first-pitch capture time.
3. Run the deterministic shadow forecast.
4. Append or replay the capture, forecast, and link.
5. Store an immutable canonical bundle copy under the game directory.
6. Print record IDs, forecast ID, lineup state, and replay status.

### `confirmed`

1. Load and strictly validate the bundle.
2. Require both lineups to be confirmed and captured before first pitch.
3. Locate the latest active pregame capture for the game.
4. Append the confirmed capture with an explicit supersession relationship
   when it replaces an earlier semantic state.
5. Run, append or replay, and link the deterministic forecast.
6. Store the immutable canonical bundle copy and print frozen IDs.

The existing `record_shadow_run` API will receive a backward-compatible
optional `supersedes_capture_id` argument so the operator does not duplicate
private forecast serialization logic.

### `review`

1. Load and strictly validate the official-result bundle.
2. Require a latest active confirmed capture linked to a forecast and captured
   before first pitch. Refuse to silently use a projected capture.
3. Append or replay optional closing market observations.
4. Append or replay the official result revision.
5. Build and append one review for every exact review-eligible market in the
   frozen confirmed forecast. Legacy passes may retain component-flow review
   data but cannot enter production priority.
6. Select leakage-safe primary calibration rows from the whole ledger.
7. Build the calibration report and atomically replace the latest JSON and
   Markdown report files.
8. Print result ID, review counts, classification counts, report sample size,
   and report paths.

## Idempotency and Recovery

- Managed bundle identity is the SHA-256 hash of canonical bundle JSON.
- Capture idempotency keys derive from schema version, game ID, stage, and the
  canonical normalized-input hash.
- Forecast identity remains the existing deterministic forecast run ID.
- Result idempotency derives from game ID, official source identity, and source
  revision.
- Review idempotency remains the existing capture/forecast/result/market key.
- Report writes use a temporary sibling file followed by atomic replacement.

Repeating the same command must return existing records and produce the same
report. If execution stops after any append, rerunning the command completes
the remaining idempotent operations without editing prior records.

## Leakage and Safety Rules

- Pregame commands reject `captured_at >= first_pitch`.
- `review` never runs a new pregame forecast.
- `review` binds only to the frozen active confirmed capture selected before
  official result ingestion.
- Bundle `created_at` is metadata and never substitutes for source observation
  timestamps.
- Closing observations cannot be used as entry prices.
- Reports preserve projected and confirmed lineup state as separate grouping
  dimensions; formal migration counts use confirmed primary captures only.
- No report can alter model configuration or production recommendation state.

## Reporting

The JSON report is the deterministic machine-readable artifact. It includes:

- Generation timestamp and ledger path.
- Overall calibration metrics.
- Fixed grouping dimensions and group metrics.
- Confirmed-lineup game and decision counts.
- Migration-gate progress for 100 confirmed games and 300 decisions per market
  family.

The Korean Markdown report presents the same values without adding new model
interpretation. It keeps sample size, hit/push rates, Brier score, log loss,
ROI, and CLV separate and labels unavailable price-dependent metrics as
unavailable rather than zero.

## Error Handling

The CLI uses stable nonzero exits for:

- Invalid JSON or unsupported schema version.
- Stage and lineup-state mismatch.
- Pregame timestamp at or after first pitch.
- Missing or conflicting source provenance.
- Missing confirmed frozen forecast during review.
- Idempotency conflict or ledger corruption.
- Invalid official score/component conservation.
- Report write failure.

Errors identify the failing field or record relationship and never invent a
fallback value. Partial append-only progress is reported so the operator can
rerun safely.

## Test Strategy

Use test-driven development. Required coverage:

1. Strict projected, confirmed, and review bundle decoding.
2. Unknown fields, bad versions, naive timestamps, and stage mismatches.
3. Projected capture/forecast/link happy path.
4. Confirmed supersession and frozen forecast selection.
5. Semantic forecast reuse and exact-command idempotency.
6. Rejection of post-first-pitch captures and result leakage.
7. Review refusal when no confirmed forecast exists.
8. Official result ingestion, correction lineage, and closing observations.
9. Review creation for all exact eligible markets, including legacy pass flow
   handling.
10. Calibration row selection and deterministic JSON/Markdown reports.
11. Interrupted-stage retry behavior.
12. Typer CLI success output, stable failures, default paths, and path
    overrides.
13. Full existing regression suite and fixed-model manifest verification.

## Acceptance Criteria

- The three commands complete an end-to-end synthetic game in tests.
- The same commands can be rerun without duplicate semantic records.
- A confirmed forecast cannot be replaced by result-time or live information.
- The latest calibration report is reproducible from the ledger alone.
- Existing fixed-model recommendations and shadow non-authority guarantees are
  unchanged.
- `python3 -m unittest discover -s tests`, `pytest`, compile checks, and
  `python3 -m analysis.pipeline` all pass.
