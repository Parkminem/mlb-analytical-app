# Review Watchlist Design

## Goal

Track repeated postgame-review concerns automatically without changing the
fixed MLB model, legacy admission, priority ordering, Shadow probabilities, or
pregame analysis latency.

The first watchlist contains three approved targets:

1. F5 over early-run conversion.
2. Bullpen-protection path contradictions.
3. Lineup-suppression path contradictions on unders.

## Decision

Add a configuration-driven watchlist aggregation to calibration reporting.
The aggregator reads only the latest eligible review records already selected
for calibration. It does not load or decode 20,000-draw forecast component
arrays and it does not append new ledger records.

The watchlist configuration uses schema version 1. Calibration report output
advances from schema version 1 to schema version 2, where the top-level
`watchlist` array is required. These versions are independent of the bundle
schemas and fixed MLB model version.

This is an observation feature, not a model-version change. The fixed legacy
model remains `2026-08-05`.

## Watchlist Configuration

Store versioned definitions in `config/review-watchlist.json`. Each target has
a stable ID, display label, selector, minimum sample, and investigation rule.

### F5 Over Early Conversion

- Target ID: `f5_over_early_conversion`
- Select reviews whose market key is `f5_total:over`.
- Count wins, losses, pushes, A/B/C/D classes, and B-class losses.
- Stay `관찰` below 20 eligible decisions.
- At 20 or more decisions, become `정밀 검토` when either:
  - non-push hit rate is below 45%; or
  - at least five B-class losses exist and B-class losses are at least 25% of
    eligible decisions.
- Otherwise report `정상 범위`.

The B-loss branch distinguishes repeated timing or conversion misses from a
principal-path contradiction. It does not reclassify any historical review.

### Bullpen Protection Contradictions

- Target ID: `bullpen_protection_contradiction`
- Select recommended reviews containing a frozen `bullpen_protection`
  principal path.
- Count eligible path exposures and reviews where that exact path appears in
  `contradicted_paths`.
- Stay `관찰` below 20 eligible decisions.
- At 20 or more decisions, become `정밀 검토` when at least three exact-path
  contradictions exist and their rate is at least 15%.
- Otherwise report `정상 범위`.

### Lineup Suppression Contradictions

- Target ID: `lineup_suppression_contradiction`
- Select F5 or full-game under reviews containing a frozen
  `lineup_suppression` principal path.
- Count eligible path exposures and reviews where that exact path appears in
  `contradicted_paths`.
- Use the same 20-decision, three-contradiction, and 15% investigation rule as
  bullpen protection.

## Status And Model-Change Boundary

Watchlist statuses are informational:

- `관찰`: the target has not reached its minimum sample.
- `정상 범위`: the minimum sample is met and no investigation rule fires.
- `정밀 검토`: the minimum sample is met and an investigation rule fires.

No status changes a score, market direction, `play`/`lean`/`pass` decision,
same-game assembly, final priority, or Shadow output. A `정밀 검토` status is
not approval to modify the model, model version, or Shadow authority. All
initial statuses use confirmed-lineup review rows only.

The existing migration boundary remains authoritative: avoid tuning a market
family before 300 reviewed decisions when feasible. Any eventual model change
still requires a separate proposal, explicit user approval, a semantic model
version, tests, and historical replay preservation.

## Data Flow

1. Review uses the frozen confirmed capture, forecast, and link.
2. Calibration selects latest non-superseded eligible review records.
3. The watchlist aggregator receives those already-selected review rows.
4. It applies configuration selectors and computes deterministic counters.
5. Calibration JSON stores a top-level `watchlist` section.
6. Calibration Markdown renders a `반복 검증 Watchlist` table.

The aggregator never reads raw boxscores, pregame bundles, forecast draws,
component distributions, or narrative review text. It does not decode
compressed forecast values or rerun simulation. Its output is therefore
reproducible from the same calibration review set and configuration.

All initial watchlist targets require `lineup_state == "confirmed"`. Projected
lineup reviews remain available to general calibration but cannot affect these
repeated-verification counters.

## Output Contract

New review publication uses calibration report schema version 2. Schema v2
preserves the schema-v1 calibration fields and adds the required top-level
`watchlist` array. Historical schema-v1 reports remain readable without a
watchlist.

Each JSON watchlist row contains:

- `target_id`
- `label`
- `status`
- `eligible_count`
- `win_count`, `loss_count`, and `push_count` when applicable
- `classification_counts`
- `b_loss_count` when applicable
- `contradiction_count` and `contradiction_rate` when applicable
- `non_push_hit_rate` when applicable
- `minimum_sample`
- `trigger_reasons`

The Markdown table shows the target, status, sample, primary metric, trigger,
and the explicit note that watchlist output has no decision authority.

Coverage audit accepts the historical schema-v1 shape and the schema-v2 shape.
For v2 it strictly validates the watchlist array and row contract, while
leaving official calibration metrics unchanged; watchlist diagnostics do not
become official or audit-only performance metrics.

## Error Handling

- Reject an unreadable or invalid watchlist configuration before invoking the
  report publisher or publishing a new generation.
- Configuration is strict UTF-8 JSON with exact root and per-kind target keys.
  Duplicate JSON keys or target IDs, unsupported schema versions or target
  kinds, unknown or inapplicable keys, invalid rates, and non-positive sample
  thresholds are errors.
- A review row that lacks data required by a selector is excluded from that
  target and counted in an `excluded_count`; it is not guessed or repaired.
- Report publication remains atomic. A watchlist failure must not publish a
  partial calibration generation or move the `current` pointer.

## Publication Compatibility

- The current operator builds new reports only as schema v2. The publisher and
  renderer still understand schema v1 so an existing v1 generation can be
  verified and upgraded without rewriting it.
- Publishing schema v2 for the same ledger revision as current schema v1
  creates a distinct immutable v2 generation and atomically swaps `current`
  once. The v1 generation and bytes remain unchanged.
- After schema v2 is current, publication cannot downgrade to schema v1.
- Stale ledger revisions, equal record counts with different ledger state
  hashes, or unequal bytes for the same ledger revision and report schema fail
  closed and preserve the existing publication.

## Performance

The watchlist scans compact review rows already loaded by calibration. It does
not decode compressed forecast values or rerun simulation. Pregame projected
and confirmed commands do not invoke watchlist aggregation, so expected-lineup
and confirmed-lineup response time is unchanged. Review report generation adds
only a linear pass over the calibration review sample.

## Tests

- F5-over aggregation counts W/L/P, A/B/C/D, and B losses correctly.
- Every target remains `관찰` at 19 eligible decisions.
- F5 over enters `정밀 검토` at 20 decisions for each approved trigger and
  remains `정상 범위` when neither trigger fires.
- Each path target requires both three contradictions and a 15% rate after the
  20-decision minimum.
- Selectors ignore unrelated markets and unrelated contradicted paths.
- Superseded or calibration-ineligible reviews never enter watchlist counts.
- JSON and Markdown output are deterministic.
- Invalid configuration prevents partial report publication.
- Existing model manifest, legacy decisions, Shadow forecasts, and full test
  suite remain unchanged.

## Non-Goals

- No automatic model learning or threshold change.
- No new baseball scoring layer.
- No retrospective rewrite of ledger reviews.
- No use of projected-lineup-only reviews in confirmed calibration counts.
- No closing-line-value or bankroll policy change.
