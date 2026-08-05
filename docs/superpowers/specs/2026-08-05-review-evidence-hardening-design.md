# Review Evidence Hardening Design

## Goal

Prevent a high-priority full-game ML or over from being admitted and frozen
without the principal evidence needed to audit the exact pregame thesis. Keep
the fixed baseball layers, single-scoring rule, market thresholds, Shadow
authority boundary, and historical replay unchanged.

## Decision

Strengthen the final admission and fresh-capture review-binding contracts. The
change only caps incomplete candidates; it never awards points or creates a
new baseball score.

### Starter-Dependent Full-Game ML

- The admission input carries an explicit boolean
  `starter_dependent_full_game_ml`. Set it only when the pregame ML thesis uses
  starter stability, survival, or suppression as a principal reason. This
  flag, rather than prose inspection, is the deterministic trigger.
- A full-game ML candidate with that flag must set the matching immutable
  candidate metadata and declare `starter_suppression` in both its evidence
  set and principal-path binding.
- `starter_suppression` binds exactly one component:
  `{away|home}_pitcher_chain_runs_allowed`. It does not bind workload.
  `starter_collapse` remains the separate runs-plus-workload shape.
- A missing starter path caps the ML priority score at `71`, which keeps the
  incomplete candidate out of the final priority stack.
- An ML whose full-game case is genuinely independent of starter stability is
  not forced to invent a starter path. Existing full-game margin and bullpen
  requirements remain in force.

### Two-Sided Full-Game Over At 9.0 Or Higher

- The admission input carries an explicit boolean `two_sided_over_thesis`,
  populated from the existing two-sided volatility/scoring analysis. It is
  active only for a full-game over whose verified market total is `9.0` or
  higher. This flag, market, direction, and line form the deterministic
  trigger.
- A triggered candidate must set matching immutable candidate metadata and
  declare two distinct canonical scoring paths, one for each team's scoring
  direction, in both its evidence set and principal-path binding.
- A batting-team conversion route uses one of `top_order_matchup`,
  `recent_run_conversion`, or `underdog_run_creation` and binds exactly
  `{side}_full_runs`, `{side}_full_traffic`, and
  `{side}_full_extra_base_hits` for that batting side.
- An opponent-pitching failure route uses `starter_collapse` and binds exactly
  the opposing pitching side's `pitcher_chain_runs_allowed` and
  `pitcher_chain_workload_outs`, or uses `late_bullpen_exposure` and its
  existing same-pitching-side workload-plus-segment-exposure shape. Such a
  route counts toward the other team's scoring direction.
- The two routes must cover opposite scoring directions and use distinct
  `PriorityEvidencePath` values, preserving the existing unique-principal-path
  contract. `starter_suppression`, `lineup_suppression`, and
  `bullpen_protection` do not satisfy an over scoring route.
- If either side lacks an independent canonical route, cap the over at `69`.
  One strong lineup plus an unbound vague late-risk claim is not a priority
  over.

### Monitoring Only

- Record HR-only, low-traffic, high-strikeout underdog run creation as a review
  diagnostic. Do not add a new `+1.5` cap from one game.
- Record material Legacy-versus-Shadow disagreement as a re-review diagnostic.
  Shadow remains non-authoritative and cannot alter admission, score, or rank.

## Data Flow

1. Existing baseball layers produce their scores and canonical evidence once.
2. The market engine produces the unchanged preliminary market decision.
3. Final admission receives the explicit thesis flag, checks whether the
   candidate's stated thesis requires the new evidence contract, and applies
   only the relevant cap when incomplete.
4. Fresh-capture validation verifies that every required evidence path has a
   matching principal-path binding.
5. Postgame review consumes the frozen bindings without reconstructing or
   inventing omitted evidence.

## Version And Compatibility

- Advance the fixed legacy model version to `2026-08-05` for fresh captures.
- Apply the new metadata and binding contracts only to fresh
  `2026-08-05` captures.
- Preserve exact replay of all older captures and keep historical ledger
  records immutable. Exact replay is resolved before the fresh-version and new
  binding checks; a non-exact old-version submission remains rejected.
- Do not change forecast probabilities, simulation configuration, calibration
  thresholds, or Shadow comparison authority.

## Failure Handling

- Incomplete fresh bundles fail validation before a forecast or ledger record
  is written.
- A complete bundle whose admission cap lowers a candidate below the priority
  threshold remains a valid capture; it is simply omitted from priority.
- Diagnostics never repair missing evidence and never promote a market.

## Non-Goals

- No new positive score or score bonus.
- No change to pitch/velocity bands, lineup grading, weather, bullpen scoring,
  prices, or market direction.
- No immediate `+1.5` rule based on the PIT result.
- No Shadow veto or rank override.
- No rewrite of historical captures, forecasts, reviews, or calibration rows.
- No automatic operator-performance optimization in this change.

## Tests

- Starter-dependent ML without a candidate starter path is capped at `71`.
- Starter-dependent ML with complete canonical starter evidence keeps its
  original score.
- A fresh recommended starter-dependent ML without a matching review binding
  is rejected.
- A two-sided over at `9.0` or higher with only one auditable scoring direction
  is capped at `69`.
- A qualifying high-total over with independent evidence and bindings for both
  directions keeps its original score.
- Two routes using the same canonical path value, two routes covering the same
  scoring side, or a route with the wrong component horizon are rejected.
- One-sided totals and lower totals are not accidentally forced into the new
  two-sided contract when their existing engine does not claim that thesis.
- Historical replay, Shadow authority, same-game assembly, and the full test
  suite remain unchanged and green.
