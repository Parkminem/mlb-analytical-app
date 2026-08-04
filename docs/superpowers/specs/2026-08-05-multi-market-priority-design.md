# Multi-Market Same-Game Priority Design

## Goal

Allow one side-family recommendation and one total-family recommendation per
game horizon to appear as separate entries in the final priority stack when
both are independently playable and directionally coherent.

For example, an Atlanta ML candidate scored at 77 and an over candidate scored
at 83 must both remain in the priority stack when both market engines pass them
and their game scripts can coexist.

## Scope

- The F5 side family contains the selected F5 side market.
- The F5 total family contains the selected F5 `under` or `over` market.
- The full-game side family contains `ML` and `+1.5` or `-1.5` markets.
- The full-game total family contains `under` and `over` markets.
- At most one candidate from each horizon-specific family may remain for a game.
- A game may therefore contribute zero to four priority entries before
  cross-horizon correlation deduplication.
- F5 market-engine evaluation remains unchanged, but F5 side and F5 total use
  the same final family, threshold, evidence, and coherence policy.
- This change does not alter baseball input scoring, market-engine validation,
  score caps, source hierarchy, or Shadow authority.

## Considered Approaches

### Family-slot assembler (selected)

Select the best eligible candidate inside each family, then retain both family
winners when their game scripts are compatible. This preserves one clear side
opinion and one clear total opinion without suppressing a valid second edge.

### Fully permissive candidate list

Keep every eligible market. This could expose ML and handicap variants based on
the same side edge and would duplicate recommendations, so it is rejected.

### Combined same-game pick object

Represent side plus total as one parlay-like item. This would blur independent
market probabilities and prices, so it is rejected. Retained candidates remain
separate priority entries.

## Eligibility And Selection

The same-game assembler applies these steps in order:

1. Exclude every candidate whose market engine returned `pass`.
2. Exclude every candidate below the existing playable priority threshold of
   72.
3. Exclude candidates with empty or non-canonical evidence paths.
4. Select the highest-scoring candidate in each horizon-specific side family.
   An exact legacy-score tie uses the deterministic market order. Shadow
   probability must not break the tie because Shadow cannot change or reorder
   legacy decisions.
5. Select the highest-scoring candidate in each horizon-specific total family.
   An over and an under for the same horizon can never both remain.
6. Retain both family winners only when directional coherence has been
   explicitly validated and each retained market has distinct, non-empty
   canonical evidence paths. Otherwise retain the higher-scoring candidate; use
   the deterministic market order on an exact legacy-score tie; Shadow
   probability must not break the tie because Shadow cannot change or reorder
   legacy decisions.

The candidate interface will carry explicit directional-coherence metadata,
while the assembler derives the fixed horizon-specific family from the market
identifier rather than attempting to infer compatibility from free-form review
text. Existing canonical evidence paths remain required and are reused; no
baseball evidence is rescored.

## Directional Coherence

Valid combinations include:

- `ML + over` when the selected team can win while the opponent contributes to
  the total, or when verified late volatility can raise scoring without breaking
  the side edge.
- `ML + under` when the selected team's win path is compatible with run
  suppression.
- `+1.5/-1.5 + over` when the handicap path and two-sided scoring path can both
  occur.
- `+1.5/-1.5 + under` when a close or controlled low-scoring game supports both
  markets.

These rules apply to F5 side-total pairs and full-game side-total pairs. An
upstream analysis must mark the pair incompatible when one recommendation
depends on a game script that materially weakens the other. Missing coherence
validation does not count as compatibility. Every same-game multi-market pair,
whether same-horizon or cross-horizon, requires distinct non-empty canonical
evidence paths so the same edge is not repeated.

## Output Behavior

- The summary table continues to show only one active side-family market.
- The `Final` cell may show both family winners, for example `ATL ML + O9.0`.
- The global priority list contains each retained market as its own entry with
  its own score and decimal price.
- The priority list remains globally score-sorted after same-game assembly.
- Review text states why the two same-game recommendations can coexist.

## Configuration And Manifest

The fixed model configuration and pipeline manifest will expose:

- the playable multi-market threshold of 72;
- one-winner-per-family behavior;
- cross-family coexistence when coherence is validated;
- unchanged pass exclusion and distinct canonical-evidence requirements for
  every retained multi-market combination;
- the same family policy for F5 candidates while preserving the F5 market
  engine and verified-line requirements.

## Testing

Tests will prove that:

- `ATL ML 77` and `ATL over 83` are both retained when validated as compatible;
- a compatible F5 side and F5 total are both retained at 72 or higher;
- ML and `+1.5` from the same game do not both remain;
- over and under from the same horizon do not both remain;
- an incompatible side-total pair keeps only the higher score;
- a candidate below 72, a `pass`, or a candidate without canonical evidence is
  excluded;
- existing F5 market evaluation remains unchanged while final admission uses
  the new family policy;
- the model configuration, manifest, authoritative documentation, and output
  template expose the new policy consistently.

## Non-Goals

- Creating parlays or estimating joint same-game probabilities.
- Letting Shadow probabilities replace fixed-model decisions.
- Adding new baseball layers or changing existing market scores.
- Automatically deriving directional compatibility from final scores or prose.
