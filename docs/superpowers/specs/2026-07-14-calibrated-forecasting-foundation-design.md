# Calibrated Forecasting Foundation Design

## Goal

Turn the current MLB rule framework into a reproducible forecasting system that can distinguish a correct baseball read from a correct betting result. The first delivery creates immutable pregame records, explicit game-flow forecasts, market probabilities, postgame classifications, and calibration reports without removing the existing fixed-model safety rules.

## Decision

Use an incremental hybrid architecture.

- Keep the existing source hierarchy, all-lineup pitch-and-velocity analysis, opener/bulk handling, bullpen role segmentation, and market validation gates.
- Add a deterministic forecasting core beneath those gates.
- Treat the existing rule engine as a compatibility and safety layer while the new probability outputs accumulate enough out-of-sample evidence.
- Do not tune weights or delete existing rules from the same games used to design this foundation.

Two alternatives were rejected:

- Adding more one-off caps to the current rule engine would preserve the main failure: narrative inputs are not converted into calibrated run and win probabilities.
- Replacing the entire model and every live data collector at once would make failures difficult to isolate and would mix data-access risk with forecasting risk.

## Scope

The first delivery includes:

- A normalized pregame input schema.
- Immutable, versioned prediction snapshots.
- Starter workload and run-allowance distributions.
- Lineup run-conversion estimates that reuse pitch-and-velocity evidence.
- Bullpen chain and exposure estimates by role segment.
- Deterministic simulation of F5 and full-game scores.
- Probabilities for F5 side, F5 O/U, ML, `+1.5`, and full-game O/U, including push probabilities.
- No-vig market comparison when both sides of a market are available.
- Postgame outcome records and flow classifications.
- Calibration, Brier-score, log-loss, ROI, and closing-line-value reports.
- A compatibility adapter that feeds the new outputs into existing market gates and same-game evidence deduplication.

The first delivery does not include automated RotoWire browser control, authenticated odds collection, weather enabled by default, live betting, automatic bet placement, or automatic production-weight retraining.

## Architecture

### Normalized Pregame Input

`analysis.forecasting.inputs` owns typed, source-neutral records for:

- Game identity, Korean date, official MLB game date, and first pitch.
- Lineup status and all nine batting slots.
- Starter, opener, bulk, and `PRIM` structure.
- Pitch mix by opposing batter side and average velocity.
- Per-hitter pitch-and-velocity evidence with sample strength.
- Recent lineup form and run-creation components.
- Starter current skills, expected pitch limit, and workload history.
- Bullpen arms, roles, quality, and recent usage.
- Market, line, decimal prices, and capture timestamp.

Collectors remain outside the forecasting core. They must normalize data before invoking the model, which allows official MLB, RotoWire, Savant, and odds sources to change without changing forecast logic.

### Feature Ownership

Each baseball input has one owner and one normalized feature value. Market engines may apply different weights but may not reconstruct or rescore the raw evidence.

- Pitch-and-velocity fit becomes an exposure-weighted continuous lineup feature, not a raw count of positive slots.
- Batting-order position controls expected plate-appearance weight.
- Sample shortage widens uncertainty and shrinks the estimate toward a handedness and pitch-type prior.
- Seven-day form remains a small modifier inside a blended recent-form feature rather than an independent strong signal.
- Starter ERA and WHIP remain descriptive inputs; survival and run estimates prioritize K-BB shape, command, home-run contact, pitches per inning, recent pitch count, and workload limits.

Every derived feature records its canonical evidence path so the existing same-game priority assembler can still prevent duplicate markets.

### Starter Forecast

`analysis.forecasting.starter` produces a discrete workload distribution and a run-rate distribution for each expected pitcher chain.

Required outputs:

- Probability of fewer than five innings.
- Probability of completing five innings.
- Probability of six or more innings.
- Expected innings and expected earned runs.
- Command, home-run, injury-return, short-exit, and source-uncertainty components.

For opener/bulk structures, the output covers opener, bulk, and uncertainty about the handoff. Unknown bulk timing widens the distribution; it does not automatically create an over.

### Lineup Run Conversion

`analysis.forecasting.offense` converts matchup evidence into a team scoring distribution.

Inputs include exposure-weighted pitch-and-velocity response, expected plate appearances, handedness, walk rate, strikeout rate, isolated power, barrel or extra-base contact, recent blended form, and lower-order continuity. The output separates:

- Traffic expectation.
- Extra-base and home-run conversion.
- Strikeout interruption risk.
- Expected F5 runs and full-game runs before bullpen adjustment.
- Uncertainty interval.

Traffic without power or sequencing support may raise baserunner expectation but cannot by itself create a high-confidence over or ML edge.

### Bullpen Chain

`analysis.forecasting.bullpen` models bridge/long relief, middle relief, setup/high leverage, and closer segments separately.

The chain uses expected starter workload to estimate the probability that each segment appears. Rested high-leverage arms protect late innings only when the game is likely to reach them without exposing a weak bridge first. Lower-relief fatigue remains separate from high-leverage availability.

Outputs include expected bullpen innings, runs, segment exposure probabilities, and late protection quality.

### Score Simulation

`analysis.forecasting.simulation` combines starter, offense, and bullpen distributions using a seeded, deterministic simulation. The initial implementation uses only the Python standard library so every run is reproducible without a new numerical dependency.

The simulator returns:

- Team F5 and full-game run distributions.
- Expected F5 and full-game scores with central intervals.
- F5 lead, tie, and loss probabilities.
- Full-game win probability.
- `+1.5` cover probability.
- Over, under, and push probabilities for verified F5 and full-game lines.

The random seed is derived from the prediction snapshot identifier and model version. Re-running the same snapshot must produce identical results.

### Market Value and Recommendation

`analysis.forecasting.market` converts decimal odds into no-vig implied probabilities when both market sides are present. It reports model probability, fair decimal price, market probability, edge, expected value, and push handling.

The new probability is distinct from:

- Data completeness.
- Predictive uncertainty.
- Existing rule-engine `play`, `lean`, or `pass` ratings.

Existing market validation remains authoritative during migration. A market-engine `pass` stays ineligible even when the raw simulation edge is positive. The 0-100 priority value is retained only as a rank score and is not described as a win probability.

### Prediction Ledger

`analysis.ledger` stores append-only JSON Lines records under a configurable data directory that is ignored by Git by default.

Each prediction snapshot contains:

- Stable snapshot ID and model version.
- Capture timestamp and source timestamps.
- Lineup and starter confirmation state.
- Normalized input hash.
- Feature summary and uncertainty flags.
- Expected game-flow outputs.
- Every evaluated market, line, price, probability, engine rating, and admission result.
- Final priority order.

Snapshots cannot be overwritten. A later lineup or line update creates a new snapshot linked to the previous snapshot. Postgame results are separate records linked by snapshot ID.

### Postgame Review and Calibration

`analysis.review` compares the frozen pregame forecast with official MLB outcomes.

Flow classification is objective and component-based:

- `A`: market result and the principal starter, offense, and bullpen paths matched.
- `B`: principal flow mostly matched, but market, line, or conversion outcome lost.
- `C`: a principal starter, offense, or bullpen path was materially wrong, regardless of ticket result.
- `D`: injury, ejection, defensive-error cluster, or another documented unpredictable event dominated the result.

The review records component errors before assigning the final class. Traffic alone cannot mark a side or total forecast correct.

Calibration reports group predictions by model version, market, lineup status, starter structure, probability bucket, and priority-score bucket. Reports include sample size, hit rate, push rate, Brier score, log loss, ROI, and closing-line value. Missing odds prevent ROI and closing-line calculations but do not prevent flow review.

## Data and Leakage Controls

- Pregame snapshots may use only records available before their capture timestamp.
- A postgame record never modifies its linked prediction snapshot.
- Historical evaluation uses walk-forward dates.
- Rules inspired by a reviewed date cannot be credited as out-of-sample performance on that date.
- Model changes create a new semantic model version.
- Production weights remain frozen until an explicitly approved sample threshold is met.

The initial validation target is at least 100 confirmed-lineup games. Weight tuning should wait for at least 300 market decisions per market family when feasible.

## Compatibility and Migration

The existing `analysis.model` API and fixed rules remain available. New modules consume normalized inputs and expose an adapter that creates the current market contexts. Existing unit tests remain regression tests.

Migration order:

1. Add schemas, append-only ledger, and deterministic IDs.
2. Add starter, offense, and bullpen distributions.
3. Add score simulation and market probability calculations.
4. Add postgame result ingestion and calibration reports.
5. Add the compatibility adapter and use it for daily analysis.
6. Accumulate a frozen out-of-sample sample before deleting, merging, or reweighting legacy rules.

The current 83 priority-related configuration flags are not deleted in the first delivery. After calibration, duplicated flags can be consolidated into numeric features and a small set of hard safety gates through a separately approved migration.

## Error Handling

- Missing required identity, line, or pitcher-chain data fails validation with an explicit error.
- Missing optional matchup samples widen uncertainty and use documented priors.
- Conflicting starter sources remain visible in the snapshot.
- Invalid odds, impossible probability totals, negative innings, and mutable snapshot attempts are rejected.
- A missing opposing price prevents no-vig value calculation; the model still returns its raw probability and fair price.
- A missing F5 line prevents an F5 total recommendation rather than inferring it from the full-game total.

## Testing

Tests are organized by component and written before implementation.

- Schema validation and stable serialization.
- Append-only ledger behavior and snapshot linking.
- Exact deterministic simulation for a fixed seed.
- Starter workload probabilities sum to one.
- Offense uncertainty increases with pitch-and-velocity sample shortage.
- Weak bridge exposure increases when starter survival falls.
- Market probabilities and push handling are internally consistent.
- No-vig conversion and expected-value calculations.
- Postgame A/B/C/D classification fixtures.
- Leakage guards reject outcome timestamps preceding prediction timestamps.
- Existing fixed-model and manifest regression suites remain green.

## Success Criteria

- The same normalized input and model version reproduce the same snapshot and probabilities.
- Every final recommendation can be traced to one immutable pregame snapshot.
- Postgame review can separate matchup, starter, bullpen, conversion, and market errors without narrative reinterpretation.
- Priority scores are no longer presented as calibrated probabilities.
- Market performance can be evaluated by probability bucket, ROI, and closing-line value.
- No legacy rule is removed until out-of-sample evidence supports the change.
