# Multi-Market Same-Game Priority Design

## Goal

Allow one full-game side-family recommendation and one full-game total-family
recommendation from the same game to appear as separate entries in the final
priority stack when both are independently playable and directionally coherent.

For example, an Atlanta ML candidate scored at 77 and an over candidate scored
at 83 must both remain in the priority stack when both market engines pass them
and their game scripts can coexist.

## Scope

- The side family contains `ML` and `+1.5` or `-1.5` markets.
- The full-game total family contains `under` and `over` markets.
- At most one candidate from each family may remain for a game.
- A game may therefore contribute zero, one, or two full-game priority entries.
- F5 side and F5 total selection retain their current independent evaluation and
  correlation-deduplication rules.
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
4. Select the highest-scoring candidate in the side family. On a tied score,
   prefer the candidate with the higher model probability when available, then
   use the existing deterministic market order.
5. Select the highest-scoring candidate in the full-game total family. An over
   and an under can never both remain.
6. Retain both family winners only when directional coherence has been
   explicitly validated. Otherwise retain the higher-scoring candidate; use the
   existing deterministic order on an exact tie.

The candidate interface will carry explicit family and directional-coherence
metadata rather than attempting to infer compatibility from free-form review
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

An upstream analysis must mark the pair incompatible when one recommendation
depends on a game script that materially weakens the other. Missing coherence
validation does not count as compatibility.

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
- unchanged pass exclusion and canonical-evidence requirements;
- unchanged F5 correlation handling.

## Testing

Tests will prove that:

- `ATL ML 77` and `ATL over 83` are both retained when validated as compatible;
- ML and `+1.5` from the same game do not both remain;
- over and under from the same game do not both remain;
- an incompatible side-total pair keeps only the higher score;
- a candidate below 72, a `pass`, or a candidate without canonical evidence is
  excluded;
- existing F5 correlation and pass-exclusion behavior remains unchanged;
- the model configuration, manifest, authoritative documentation, and output
  template expose the new policy consistently.

## Non-Goals

- Creating parlays or estimating joint same-game probabilities.
- Letting Shadow probabilities replace fixed-model decisions.
- Adding new baseball layers or changing existing market scores.
- Automatically deriving directional compatibility from final scores or prose.
