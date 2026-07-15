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

The ledger remains the authoritative append-only state. Bundle copies and
reports are derived operator artifacts. Implementation must add `data/shadow/`
to `.gitignore`; `data/ledger/` and `reports/private/` are already ignored.

## Architecture

Create a small `analysis.operator` package with explicit boundaries:

- `schemas.py`: versioned external JSON schemas and strict decoding into the
  existing forecasting, legacy-decision, result, and market-observation value
  objects.
- `config.py`: immutable forecast-config registry loading. Each semantic shadow
  version has one complete JSON config file under `config/forecasting/`; files
  are additive and must never be edited after captures reference them.
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
- `forecast_config_version`: the exact immutable config-registry version used
  to construct `SimulationConfig`.

The CLI resolves the complete config from
`config/forecasting/<forecast_config_version>.json` and runs
`run_forecast(game_input, simulation_config)` itself. A caller cannot provide a
precomputed shadow forecast or config overrides. The registry file content is
included in the forecast identity, so retries remain deterministic after newer
config versions are added.

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

- The exact frozen `capture_id`, `forecast_run_id`, and
  `capture_forecast_link_record_id` returned by the confirmed command.
- Official MLB result source identity and revision.
- F5 and full-game scores.
- Available component observations and documented external events.
- Optional closing market observations with decimal prices and observation
  timestamps.

An initial official result sets `supersedes_result_record_id` to null. A
correction must provide the exact current latest result record ID; the operator
never infers a correction target. Each closing observation contains
`source_name`, `source_record`, `source_revision`, `market_key`, `market_type`,
`settlement_format`, `selected_side`, `line`, both decimal prices,
`observed_at`, `available_at`, and an explicit nullable
`supersedes_record_id`. `observation_kind` is fixed to `close`. A corrected
quote must explicitly reference the prior quote record.

The result contract reuses `OfficialResultCommand`, including score and
pitching-run conservation validation. Corrections append a new immutable
revision that supersedes the previous official result.

## Stage Semantics

### `projected`

1. Load and strictly validate the bundle.
2. Require projected lineup state and a pre-first-pitch capture time.
3. Apply the same keyed-capture preflight and active-capture ordering defined
   for `confirmed`. A changed projected state supersedes the latest active
   projected pregame capture for the game; an exact retry reuses its original
   target. If no keyed projected capture already exists and the game has an
   active confirmed capture, reject the new projected command. A projected
   command can never supersede or deactivate a confirmed capture.
4. Run the deterministic shadow forecast and append or replay the capture,
   forecast, and link.
5. Store an immutable canonical bundle copy under the game directory.
6. Print record IDs, forecast ID, lineup state, and replay status.

### `confirmed`

1. Load and strictly validate the bundle.
2. Require both lineups to be confirmed and captured before first pitch.
3. Locate the latest active pregame capture for the game.
4. Compute the capture idempotency key and first search the ledger for an
   existing capture with that key. On retry, reuse that record's exact
   `supersedes_capture_id`. Only when no keyed capture exists may the operator
   select the latest active capture as the supersession target. Active means a
   capture not referenced by another capture's `supersedes_capture_id`.
   Candidate ordering is `(captured_at, envelope.appended_at,
   envelope.record_id)` and the greatest tuple wins. Append the confirmed
   capture with that fixed target when it replaces an earlier semantic state.
5. Run, append or replay, and link the deterministic forecast.
6. Store the immutable canonical bundle copy and print frozen IDs.

The existing `record_shadow_run` API will receive a backward-compatible
optional `supersedes_capture_id` argument so the operator does not duplicate
private forecast serialization logic.

### `review`

1. Load and strictly validate the official-result bundle.
2. Resolve the exact capture, forecast, and link IDs supplied by the bundle.
   Require one matching relationship, confirmed lineup state, matching game
   ID, pre-first-pitch capture time, and an active capture. Refuse to select a
   different, newer, or projected capture. Multiple links are harmless because
   the bundle names one exact link and its forecast; mismatched IDs fail.
3. Append or replay optional closing market observations.
   Require both `observed_at` and `available_at` to be between the frozen
   capture time and first pitch, inclusive, and both decimal prices to exceed
   `1.0`. Quote correction targets are explicit and stable across retries.
4. Append or replay the official result revision using the bundle's explicit
   nullable `supersedes_result_record_id`. If a prior result exists, a null
   target is rejected; if a correction target is supplied, it must equal the
   current latest result unless the same result idempotency key already exists.
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
- Managed filenames are
  `<stage>-v<schema_version>-<bundle_hash_prefix_16>.json` inside the game
  directory. An existing path must contain identical canonical bytes.
- Capture idempotency keys derive from schema version, stage, and a SHA-256 hash
  of the exact intrinsic `CaptureCommand` content: game ID, `captured_at`, first
  pitch, lineup state, eligibility, normalized-input hash, and canonical
  capture payload. `supersedes_capture_id` is excluded because it is resolved
  after key lookup. This preserves distinct source-state/audit captures when
  forecast semantics are unchanged while making an exact command replayable.
- Forecast identity remains the existing deterministic forecast run ID.
- Result idempotency derives from game ID, official source identity, and source
  revision. The operator searches for that exact key before validating whether
  the bundle's explicit correction target is still latest, so a partially
  completed correction replays instead of attempting to supersede itself.
- Before deriving any supersession target, the operator checks for an existing
  record with the stage's idempotency key. A found record locks the original
  target and turns all later work into replay/completion behavior.
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

- `ledger_as_of`, defined as the maximum `envelope.appended_at` in the ledger,
  or null for an empty ledger, plus the ledger path. No wall-clock generation
  timestamp is serialized, so an unchanged ledger produces byte-identical
  reports.
- Overall calibration metrics.
- Fixed grouping dimensions and group metrics.
- Confirmed-lineup game and decision counts.
- Migration-gate progress for 100 confirmed games and 300 decisions per market
  family.

Migration denominators are fixed as follows:

- Confirmed games are unique `game_pk` values among leakage-safe primary
  calibration rows with `lineup_state == "confirmed"` and at least one exact
  recorded review.
- A market-family decision is one unique `(forecast_run_id, market_key)` primary
  confirmed row, grouped by canonical `market_type`.
- Exact review-eligible legacy passes are included because they are still
  frozen decisions with probabilities and settlements. Review-ineligible
  markets are excluded.

The Korean Markdown report presents the same values without adding new model
interpretation. It keeps sample size, hit/push rates, Brier score, log loss,
ROI, and CLV separate and labels unavailable price-dependent metrics as
unavailable rather than zero.

## Error Handling

The CLI uses stable exits:

- `0`: success or exact replay.
- `1`: unexpected internal failure.
- `2`: Typer usage or option error.
- `3`: invalid JSON, unsupported schema, or field validation failure.
- `4`: stage, timestamp, source, or frozen-record precondition failure.
- `5`: ledger reference, idempotency, or corruption failure.
- `6`: managed-bundle or report I/O failure.

The nonzero categories cover:

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
6. Distinct capture records for unchanged forecast semantics with different
   capture times or audit provenance.
7. Rejection of a new projected capture after confirmation while allowing an
   exact historical projected replay.
8. Rejection of post-first-pitch captures and result leakage.
9. Review refusal when no confirmed forecast exists.
10. Official result ingestion, correction lineage, and closing observations.
11. Review creation for all exact eligible markets, including legacy pass flow
   handling.
12. Calibration row selection and deterministic JSON/Markdown reports.
13. Interrupted-stage retry behavior.
14. Typer CLI success output, stable failures, default paths, and path
    overrides.
15. Config-registry version resolution and historical-version replay.
16. Byte-identical reports for an unchanged ledger and fixed migration-gate
    denominators.
17. Full existing regression suite and fixed-model manifest verification.

## Acceptance Criteria

- The three commands complete an end-to-end synthetic game in tests.
- The same commands can be rerun without duplicate semantic records.
- A confirmed forecast cannot be replaced by result-time or live information.
- The latest calibration report is reproducible from the ledger alone.
- Existing fixed-model recommendations and shadow non-authority guarantees are
  unchanged.
- `python3 -m unittest discover -s tests`, `pytest`, compile checks, and
  `python3 -m analysis.pipeline` all pass.
