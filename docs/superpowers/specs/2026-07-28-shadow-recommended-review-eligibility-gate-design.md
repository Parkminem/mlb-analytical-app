# Shadow Recommended Review Eligibility Gate Design

## Goal

Prevent a projected or confirmed pregame record from being newly frozen when
any legacy `play` or `lean` candidate cannot be reviewed from its immutable
pregame contract.

This change protects calibration coverage only. It must not change baseball
inputs, model weights, simulation output, market probability, legacy admission,
legacy score, or final-priority ordering.

## Background

The July 28, 2026 review exposed a coverage gap. Several displayed
recommendations had review bindings, but their component shapes violated the
fixed contradiction-rule contract. Shadow correctly stored
`review_eligible: false` with `invalid_path_count`, while the legacy fixed model
correctly remained the decision authority. The result was nevertheless
undesirable: displayed losing recommendations were absent from calibration,
making the aggregate report optimistic.

The invalid examples came from a date-specific bundle helper that treated
`starter_suppression` and `starter_collapse` as interchangeable and built
two-sided total bindings:

- `starter_suppression` requires one pitching side's
  `pitcher_chain_runs_allowed`.
- `starter_collapse` requires one pitching side's
  `pitcher_chain_runs_allowed` and `pitcher_chain_workload_outs`.
- `late_bullpen_exposure` requires one side's named bullpen-segment exposure
  component and the same side's pitcher-chain workload.

## Scope

### Included

- Apply the gate to both `projected` and `confirmed` new pregame records.
- Require every legacy candidate rated `play` or `lean` to have one valid,
  exact review binding.
- Require every final-priority item to resolve to a `play` or `lean` candidate.
- Keep `pass` candidates outside the hard gate. They may remain unbound or
  review-ineligible.
- Reject invalid recommended contracts before any ledger append or managed
  bundle persistence.
- Preserve exact replay of historical immutable pregame commands, including
  historical records that predate this gate.
- Produce a separate July 28 coverage audit without altering calibration.

### Excluded

- Retroactive edits to captures, forecasts, links, reviews, or calibration
  generations.
- Hindsight reconstruction of principal paths after an official result.
- Changes to fixed-model decisions, forecast probabilities, thresholds, or
  report formulas.
- Requiring review bindings for legacy `pass` candidates.

## Validation Architecture

### Phase 1: Static preflight

For a new projected or confirmed command, run a lightweight validation after
the exact-replay lookup and before the 20,000-draw forecast.

For each `play` or `lean` candidate:

1. Exactly one binding must target its candidate index.
2. `legacy_market` must match the candidate market.
3. The binding market key must exist in the normalized game market inputs.
4. The semantic market type must agree with the legacy market family.
5. The binding must contain two to four unique canonical evidence paths.
6. Every path must satisfy its contradiction-rule component shape and
   market-horizon compatibility.
7. Every component ID must use the captured `away_` or `home_` namespace.

Final priority must contain only markets represented by valid recommended
candidates. A duplicate or unresolved final-priority market is rejected.

This phase is linear in the small number of candidates and bound component IDs.
It performs no simulation, network request, or full-ledger scan.

### Phase 2: Dynamic forecast gate

After the forecast is built, compare it with the legacy decision using the
existing deterministic comparison code. Before `record_shadow_run` can append:

1. Every `play` or `lean` comparison must be `review_eligible`.
2. No required comparison may contain an ineligibility flag.
3. Every bound component ID must exist in the frozen component distributions.
4. The frozen market evaluation must match its semantic market fingerprint.

Keep the existing ineligible-comparison representation for `pass` candidates
and for reading historical records. The gate changes admission of new
recommended records, not deserialization of history.

### Exact replay

Exact replay lookup remains before the new-command gate. If the intrinsic
command already exists, the operator validates and returns the original
immutable capture, forecast, and latest exact link under the existing replay
rules. It does not create a new state and is not rejected merely because an old
recommended comparison was historically ineligible.

A behavior-changing bundle, new source state, or corrected legacy contract is a
new command and must pass both new gates.

## Failure Behavior

Reject a new pregame command with an operator precondition error that includes:

- candidate index and legacy market,
- decision rating,
- market key when available,
- fixed review-ineligibility flags or the exact binding-shape reason.

The failure must occur before capture, forecast, link, managed-bundle, or report
artifacts are written. Partial progress must not claim a durable capture.

The CLI should give an actionable error such as:

```text
recommended review contract is invalid:
candidate[4] f5_total (lean): invalid binding shape for starter_suppression
```

## July 28 Coverage Audit

Create immutable JSON and Markdown audit artifacts under:

```text
reports/private/shadow/audits/2026-07-28-review-coverage/
```

The audit records:

- exact frozen capture, forecast, and link IDs,
- excluded market keys and stored ineligibility flags,
- official settlement where the game was final,
- the five resolved omitted losses,
- the postponed Cleveland decision as unresolved,
- current official calibration metrics,
- audit-only metrics that include the settled omitted decisions.

The audit must prominently state that adjusted metrics are diagnostic and are
not calibration rows. It must not append ledger records or replace the current
calibration generation.

## Performance

Normal successful runs add only small in-memory validation over at most a few
candidates and paths. They add no forecast draw, source request, ledger scan, or
report generation.

Expected impact is negligible relative to simulation and ledger access. Invalid
new bundles become faster overall because static preflight stops them before
the forecast simulation.

Exact replay retains its current fast path and does not rerun the 20,000-draw
simulation.

## Testing

Add deterministic tests covering:

- projected new state rejects a `lean` candidate with no binding;
- confirmed new state rejects a `play` candidate with an invalid path shape;
- all valid `play` and `lean` bindings pass;
- a missing or invalid `pass` binding remains allowed;
- unresolved, duplicate, or pass-rated final priority is rejected;
- missing frozen component distribution fails before ledger append;
- a rejected command leaves ledger and managed-bundle state unchanged;
- historical exact replay remains available;
- projected and confirmed successful outputs are otherwise byte-identical to
  the pre-gate behavior;
- the July 28 audit totals and excluded-market inventory are deterministic.

Run the full operator, shadow, review, calibration, and reporting test suites
before completion.

## Success Criteria

- No new projected or confirmed capture can contain a review-ineligible
  `play` or `lean` candidate.
- `pass` behavior and legacy authority remain unchanged.
- Exact replay compatibility remains intact.
- The July 28 omission is visible without contaminating official calibration.
- Successful confirmed-analysis latency does not materially regress.
