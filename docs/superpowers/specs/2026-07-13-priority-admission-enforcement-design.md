# Priority Admission Enforcement Design

## Goal

Apply the existing fixed-model rules consistently when final priority candidates are assembled after the July 13 review, without adding a new analysis layer, new numeric caps, or rescoring baseball evidence.

## Scope

The change strengthens only final candidate construction and same-game market selection:

- Treat every market-engine `pass` as ineligible for the final priority stack. This preserves the existing `+1.5` survival/scoring-path rules and total one-sided/cold/high-line rules without evaluating them twice.
- Select F5 side, F5 total, and full-game total in one same-game stage.
- Keep only the highest-scoring market when candidates share an evidence key. Preserve candidates whose evidence-key sets are disjoint.
- Preserve current F5-total versus full-game-total tie behavior. For an exact F5-side versus F5-total tie on the same evidence, keep the F5 side as the primary directional market.

## Architecture

`evaluate_priority_admission` remains unchanged as a caps-only function. The existing handicap and total engines continue to own survival, scoring-path, starter-floor, cold-conversion, and one-sided-path validation.

The final assembler `assemble_same_game_priority` receives already-rated candidates with their score and canonical evidence paths. Evidence paths use a fixed enum derived from existing model layers, not free-form labels. The assembler first removes `pass` candidates, then processes remaining candidates by descending score and deterministic tie preference. A candidate is excluded when its evidence paths overlap a higher-priority selected candidate. The existing total-only selector delegates to this assembler for backward compatibility.

## Selection Rules

- `pass` is excluded, not converted into a scored lean.
- Higher score wins when evidence overlaps.
- Exact tie order is F5 side, F5 total, full-game total, then other markets.
- Multiple markets are retained only when each has a distinct, non-empty evidence-path set. Missing evidence means independence is unverified, so only the highest-ranked candidate is retained.

## Compatibility

The existing total-only selector API remains valid. The analysis manifest identifies `assemble_same_game_priority` as the required final candidate assembler. Existing model thresholds, source hierarchy, pitch-velocity grading, market scoring, and priority caps are unchanged.

## Verification

Regression tests cover pass exclusion using real market-engine decisions, two- and three-candidate correlation, partial independence, missing evidence, deterministic ties, backward-compatible total selection, manifest exposure, and the full test suite.
