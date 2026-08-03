# Repeated Review Model Hardening Design

## Goal

Apply only the model changes repeatedly supported by postgame reviews while
preserving the fixed layer order, single-scoring rule, historical replay, and
existing market direction logic.

## Decision

The change strengthens existing admission gates. It does not add a new
baseball scoring layer and does not award positive points.

### F5 Under

- Require an explicit two-sided run-suppression verification before an F5
  under can be admitted.
- Treat unresolved long-horizon starter baseline risk as a hard F5-under
  blocker even when the latest two or three starts improved.
- Keep the existing command, HR, first-inning, injury, workload, lineup, and
  pitch-velocity gates unchanged.

### Side Priority

- Cap a side whose edge is matchup or traffic only when recent run conversion
  does not support it. A bullpen edge cannot repair the missing offensive
  conversion path.
- Cap full-game ML when no independent full-game margin path exists beyond the
  first-five edge.
- Keep bullpen availability and bullpen performance support separate. A
  full-game side is capped when its protection case is availability-only or
  the available late-inning group lacks current performance reliability.
- Reuse the existing F5, ML, and `+1.5` engines; do not redirect automatically
  to another market.

### Review Binding Completeness

- Add the review-only canonical `lineup_suppression` evidence path. It binds
  one lineup's horizon-specific runs, traffic, and extra-base-hit components;
  an upper-tail result contradicts the path.
- A newly frozen recommended F5 or full-game under must bind both
  `starter_suppression` and `lineup_suppression`.
- The pitcher-chain and lineup components must use the same away/home
  namespace. This covers opposite scoring directions: that team's pitching
  chain suppresses the opponent while that team's lineup remains muted.
- The validator rejects incomplete bindings. It never invents a missing path.

## Version And Compatibility

- Advance the fixed legacy model version to `2026-08-03`.
- Apply the new under-binding contract only to `2026-08-03` fresh captures.
- Preserve the `2026-08-02` ML bullpen-binding contract and all older exact
  historical replay.
- Leave the shadow forecasting model and calibration thresholds unchanged.

## Non-Goals

- No new positive score or score bonus.
- No weather change.
- No change to pitch/velocity bands or hitter slot labels.
- No rewrite of historical ledger records.
- No automatic conversion of a capped ML into F5 or `+1.5`.

## Tests

- F5 under is blocked by missing two-sided verification or long-horizon
  starter risk.
- Matchup-only sides, ML without a full-game margin path, and availability-only
  bullpen protection receive only their intended caps.
- A complete independent candidate preserves its original score.
- New-version under bindings reject one-sided or directionally duplicated
  evidence and accept complete two-sided evidence.
- Existing ML bullpen contracts and historical replay remain valid.
- The full test suite remains green.
