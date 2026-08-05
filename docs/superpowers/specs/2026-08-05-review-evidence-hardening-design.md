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

- A full-game ML candidate that explicitly relies on starter stability,
  survival, or suppression must declare a canonical starter path in both the
  candidate evidence and its principal-path binding.
- The allowed canonical path is `starter_suppression`, bound to the relevant
  pitcher-chain runs allowed and workload components.
- A missing starter path caps the ML priority score at `71`, which keeps the
  incomplete candidate out of the final priority stack.
- An ML whose full-game case is genuinely independent of starter stability is
  not forced to invent a starter path. Existing full-game margin and bullpen
  requirements remain in force.

### Two-Sided Full-Game Over At 9.0 Or Higher

- A full-game over at a market total of `9.0` or higher that relies on
  two-sided scoring must declare an auditable scoring path for each team.
- Each side's path must be represented by canonical evidence already produced
  by the fixed layers, such as `top_order_matchup`, `recent_run_conversion`,
  `starter_collapse`, or `late_bullpen_exposure`.
- The principal-path binding must include the corresponding horizon-specific
  team or pitcher-chain components for both scoring directions.
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
3. Final admission checks whether the candidate's stated thesis requires the
   new evidence contract and applies only the relevant cap when incomplete.
4. Fresh-capture validation verifies that every required evidence path has a
   matching principal-path binding.
5. Postgame review consumes the frozen bindings without reconstructing or
   inventing omitted evidence.

## Version And Compatibility

- Advance the fixed legacy model version for fresh captures.
- Apply the new contracts only to captures made with the new model version.
- Preserve exact replay of all older captures and keep historical ledger
  records immutable.
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
- One-sided totals and lower totals are not accidentally forced into the new
  two-sided contract when their existing engine does not claim that thesis.
- Historical replay, Shadow authority, same-game assembly, and the full test
  suite remain unchanged and green.
