# Priority Admission Enforcement Design

## Goal

Prevent existing fixed-model rules from being skipped between baseball scoring and
final priority assembly. This change adds no positive score and no new analysis
layer.

## Single enforcement point

Extend the existing `analysis.review.evidence_contract` pre-recording contract.
The contract already caps unsupported ML and high-total over candidates. It will
also enforce the fixed rules that were previously documented or applied in the
market engines but were not guaranteed at the recording boundary:

- cap a matchup-only side without a separate recent run-conversion path;
- remove a low-total over from priority unless it has two scoring sides and a
  verified late-variance path;
- cap a side against a neutral-heavy opposing lineup;
- cap a side whose selected-team matchup pitcher is returning from injury or has
  a recent short exit.

The caps reuse existing fixed-model levels: `74` for traffic/matchup uncertainty,
`71` for starter-risk sides, and `69` for unsupported overs.

## Input derivation

- Canonical candidate paths and review bindings determine matchup-only,
  two-sided scoring, and late-bullpen evidence.
- The frozen normalized game input determines the selected team, opposing lineup,
  and matchup pitcher.
- Opposing hitter grades reuse the existing pitch-type/velocity offense forecast
  and fixed `.360`/`.290` slot thresholds. A slot with at least 65% unknown
  exposure remains sample shortage rather than being counted as neutral.

## Integration

`validate_recommended_review_contract` receives the frozen normalized game input
from the operator and applies the evidence contract before any ledger write.
Existing callers that only validate candidate/binding structure remain supported,
but current operator recordings always use the full input-aware contract.

## Verification

Regression tests cover each new enforcement case, supported candidates that must
remain unchanged, and the operator call path. Existing model, shadow, schema, and
operator tests must continue to pass.
