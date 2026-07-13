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

`analysis.forecasting.starter` produces a discrete workload distribution and one run-prevention multiplier for each expected pitcher chain. It does not produce a second independent run total. This prevents starter runs from being added on top of offense runs in simulation.

Required outputs:

- Probability of fewer than five innings.
- Probability of completing five innings.
- Probability of six or more innings.
- Expected innings and expected total runs allowed per nine innings. ERA remains descriptive; market settlement and simulation use total runs.
- Command, home-run, injury-return, short-exit, and source-uncertainty components.

For opener/bulk structures, the output covers opener, bulk, and uncertainty about the handoff. Unknown bulk timing widens the distribution; it does not automatically create an over.

The initial workload probabilities come from the pitcher's eligible starts before capture time:

- `P(<5 IP)`, `P(5.0-5.2 IP)`, and `P(>=6 IP)` use empirical bucket frequencies with one league-prior pseudo-start in each bucket.
- After a bucket is selected, exact recorded outs are sampled from that pitcher's eligible starts inside the bucket, again mixed with the versioned league prior. The result is an integer from zero through 27 outs, so bullpen entry is never inferred from decimal-inning arithmetic.
- A documented pitch limit shifts probability mass from `>=6 IP` to `<5 IP` and `5.0-5.2 IP` according to the versioned simulation configuration.
- Explicit opener/bulk workloads use the same buckets over combined expected chain innings, while each pitcher retains a separate run-prevention multiplier.

The initial run-prevention multiplier is `projected_RA9 / league_RA9`, clipped to `[0.55, 1.75]`. `projected_RA9` is a normalized input with its component values and provenance recorded. The first delivery does not invent a hidden RA9 estimator; collectors or manual analysis must supply the documented projection until a separately specified estimator is approved.

### Lineup Run Conversion

`analysis.forecasting.offense` converts matchup evidence into two multipliers and supporting diagnostics. It does not independently sample runs.

Inputs include exposure-weighted pitch-and-velocity response, expected plate appearances, handedness, walk rate, strikeout rate, isolated power, barrel or extra-base contact, recent blended form, and lower-order continuity. The output separates:

- Traffic expectation.
- Extra-base and home-run conversion.
- Strikeout interruption risk.
- A base offense multiplier and a pitcher-specific matchup multiplier.
- Expected traffic and extra-base-hit rates for postgame flow review.
- Uncertainty interval.

Traffic without power or sequencing support may raise baserunner expectation but cannot by itself create a high-confidence over or ML edge.

The initial frozen formulas are deliberately small and auditable. Team offense projections are expressed as runs scored, not runs allowed:

- `base_offense_multiplier = clip(projected_team_runs_per_9 / league_runs_per_team_game, 0.65, 1.45)`.
- `matchup_multiplier = clip(exposure_weighted_adjusted_wOBA / league_wOBA, 0.80, 1.20)`.
- Expected plate-appearance weights are `[1.11, 1.08, 1.06, 1.04, 1.01, 0.98, 0.94, 0.91, 0.87]`, normalized to sum to nine.
- Pitch-and-velocity observations retain the existing sample shrinkage for compatibility. Sample shortage increases uncertainty but cannot increase the point estimate.
- Recent form is already contained in `projected_team_runs_per_9`; it must not be multiplied into the run rate a second time.

Every offense and run-prevention multiplier carries a point estimate and a log-scale standard error. Once per simulated game, each latent multiplier is drawn as `exp(log(point) - sigma^2 / 2 + sigma * Z)` and clipped to its documented bounds. The mean-preserving correction prevents wider uncertainty from increasing expected offense. The same latent draw applies across that team's innings against the applicable pitcher segment, preserving within-game correlation.

`projected_team_runs_per_9`, traffic rate, extra-base-hit rate, and multiplier standard errors are normalized inputs with documented components and provenance. Replacing these frozen formulas requires a new model version.

### Bullpen Chain

`analysis.forecasting.bullpen` models bridge/long relief, middle relief, setup/high leverage, and closer segments separately. Each segment supplies a run-prevention multiplier, not an independent run total.

The chain uses expected starter workload to estimate the probability that each segment appears. Rested high-leverage arms protect late innings only when the game is likely to reach them without exposing a weak bridge first. Lower-relief fatigue remains separate from high-leverage availability.

Outputs include expected bullpen innings, segment exposure probabilities, run-prevention multipliers, and late protection quality. Segment multipliers use `projected_RA9 / league_RA9`, clipped to `[0.60, 1.90]`.

### Score Simulation

`analysis.forecasting.simulation` combines starter, offense, and bullpen distributions using a seeded, deterministic simulation. The initial implementation uses only the Python standard library so every run is reproducible without a new numerical dependency.

The model has one run generator. For team `T` in inning `i` against pitcher segment `P`:

`mean_runs(T, i, P) = league_runs_per_team_game / 9 * base_offense_multiplier(T) * matchup_multiplier(T, P) * run_prevention_multiplier(P)`

No starter, offense, or bullpen run estimate is added outside this equation. The default versioned parameters are:

- `league_runs_per_team_game = 4.50`
- Gamma-Poisson inning dispersion shape `k = 1.60`
- `simulations = 20_000`
- F5 innings `1-5`; regulation innings `1-9`
- Extra-inning run-intensity multiplier `1.80` for the automatic-runner environment

For each half-inning, the simulator draws `lambda ~ Gamma(shape=k, scale=mean_runs/k)` with `random.gammavariate`, then draws the Poisson count with a documented standard-library implementation. Starter workload is drawn first. After starter exit, bridge and middle segments fill innings before setup/high leverage and closer segments. High-leverage segments can enter from the seventh inning when the simulated margin is three runs or fewer. The bottom of the ninth is skipped when the home team already leads, and a bottom-half walk-off ends the game immediately. Tied regulation games simulate extra innings until the tie breaks, with a safety cap of 30 innings. At the cap, a seeded Bernoulli draw weighted by each team's current extra-inning scoring intensity awards exactly one run to the selected team, preserving both a winner and a coherent total. All simulated extra-inning runs count toward full-game totals.

The simulator returns:

- Team F5 and full-game run distributions.
- Expected F5 and full-game scores with central intervals.
- F5 lead, tie, and loss probabilities.
- Full-game win probability.
- `+1.5` cover probability.
- Over, under, and push probabilities for verified F5 and full-game lines.

The random seed is derived from the deterministic forecast-run identifier. Re-running the same forecast run must produce identical results.

### Market Value and Recommendation

`analysis.forecasting.market` converts decimal odds into no-vig implied probabilities when both market sides are present. It reports model probability, fair decimal price, market probability, edge, expected value, and push handling.

Settlement conventions are fixed:

- Full-game ML continues through extra innings and has no push.
- F5 side settles after five complete innings; a tie is a push unless the captured market explicitly declares a three-way format.
- Totals and handicaps push when the simulated result lands exactly on an integer line.
- For one unit staked at decimal price `d`, `EV = P(win) * (d - 1) - P(loss)`; pushes return the stake.
- Push-aware fair decimal price is `1 + P(loss) / P(win)` when `P(win) > 0`.
- Two-way no-vig probabilities normalize `1 / decimal_price` across both captured sides and represent probabilities conditional on a non-push result.
- Model-to-market edge compares `P(win) / (1 - P(push))` with the no-vig market probability.

Brier score and log loss are computed on conditional win probabilities after excluding pushes. A separate multiclass Brier score covers win/push/loss when push probability is nonzero. ROI includes pushes as zero profit.

Closing lines are appended as `MarketObservation(kind="close")` records by a collector or manual import. CLV is omitted when no closing observation exists. Probability CLV is `closing_no_vig_probability - entry_no_vig_probability` in the selected direction; price CLV is also retained for audit.

The new probability is distinct from:

- Data completeness.
- Predictive uncertainty.
- Existing rule-engine `play`, `lean`, or `pass` ratings.

The new engine runs in mandatory `shadow` mode in the first delivery. It records probabilities, fair prices, and comparisons but cannot alter legacy F5, ML, `+1.5`, total, or priority decisions. Existing market validation remains authoritative. A separately approved migration must define any future production precedence. The 0-100 priority value is retained only as a legacy rank score and is not described as a win probability.

### Prediction Ledger

`analysis.ledger` stores append-only JSON Lines records under a configurable data directory that is ignored by Git by default.

The records divide capture facts from deterministic forecast outputs:

- `CaptureRecord` contains random immutable `capture_id`, caller-supplied `idempotency_key`, capture and source timestamps, lineup and starter confirmation state, canonical normalized inputs and their hash, source provenance, and optional `supersedes_capture_id`.
- `ForecastRecord` contains deterministic `forecast_run_id`, model and simulation versions, normalized input hash, semantic market state, feature summaries and uncertainty flags, expected game-flow outputs, every evaluated market probability and admission result, copied legacy engine ratings, and the shadow comparison order.
- `CaptureForecastLink` contains its own record ID and idempotency key plus exactly one `capture_id` and one `forecast_run_id`.

`capture_id` is a UUID generated when a source-state capture is first appended. `forecast_run_id` is the SHA-256 hash of canonical normalized inputs, semantic market state, model version, and simulation configuration. Semantic market state contains only market type, settlement format, line, and decimal prices; capture, observation, and fetch timestamps are excluded along with outputs. The first 64 bits of `forecast_run_id` seed simulation. Multiple captures may share a forecast-run ID, but a capture ID is unique and cannot be reused. A later lineup or line update creates a new capture linked by `supersedes_capture_id` and normally creates a new forecast-run ID.

`ForecastRecord` is stored exactly once per `forecast_run_id`; its canonical payload never contains a `capture_id`. `CaptureForecastLink` joins one capture to one forecast run. Its key is the pair `(capture_id, forecast_run_id)`, allowing unchanged repeated captures to reference the same forecast without changing or duplicating it.

Official postgame results are immutable game-level records keyed by official game and source revision, not duplicated per capture. A review record explicitly links the selected `CaptureForecastLink`, `ForecastRecord`, and official result. The same forecast-run ID must always reproduce the same output, while different capture events remain separately auditable.

All ledger append commands require an `idempotency_key` scoped by record kind. The ledger keeps a durable key-to-record index alongside the append-only log:

- Replaying the same record kind and idempotency key with the same canonical payload returns the existing record ID and does not append another line.
- Reusing the key with a different canonical payload is rejected as an idempotency conflict.
- A genuine source-state change uses a new key and, where applicable, links to the prior record with `supersedes_record_id` or `supersedes_capture_id`.
- Forecast output uses `forecast:{forecast_run_id}`. A capture-to-forecast link uses `capture-forecast:{capture_id}:{forecast_run_id}`. Recomputing the same run or relinking the same capture therefore cannot append a duplicate or create a payload conflict.
- Source-backed captures, closing observations, and official results derive keys from source name, stable source record ID, source revision or observed-at value, and record purpose. Manual imports must provide a stable external key.
- An official result correction is a new revision that supersedes the prior result; it never mutates the old row.
- The primary calibration dataset selects the latest eligible non-superseded pregame capture per official game under a versioned cutoff policy, then counts each `(forecast_run_id, market_key)` at most once with the latest non-superseded official result. Earlier captures may be evaluated only in separately labeled time-slice reports. Repeated unchanged captures, superseded rows, and idempotent replays never increase the headline sample size.

### Postgame Review and Calibration

`analysis.review` compares the frozen pregame forecast with official MLB outcomes.

Flow classification is objective and component-based. Each candidate records two to four principal paths from the existing canonical evidence-path enum before the game. For each path, the forecast stores its applicable central 80% and 90% predictive intervals.

- `A`: the market wins, no principal path is contradicted, and at least 75% of principal component observations fall inside their central 80% intervals.
- `B`: no principal path is contradicted and at least 60% of principal component observations fall inside their central 80% intervals, but the candidate does not satisfy `A`. This includes losing or pushed tickets whose projected flow was broadly correct and winning tickets whose flow match was only partial.
- `C`: any principal path is contradicted by an outcome outside its central 90% interval, or fewer than 60% of principal component observations fall inside their central 80% intervals, regardless of ticket result.
- `D`: a documented injury, ejection, defensive-error cluster producing at least three unearned runs, or another predeclared exogenous event occurs and the affected component lands outside its central 90% interval.

Classification precedence is `D`, then `C`, then `A` or `B`. A winning ticket with a materially wrong flow is therefore `C`, not `A`. If no market was recommended, the system records component matches but no A/B market class.

Path contradiction is deterministic:

- Starter suppression is contradicted when pitcher-chain runs allowed exceed the 90th percentile.
- Starter collapse is contradicted when pitcher-chain runs allowed fall below the 10th percentile and workload reaches the forecast median or longer.
- Top-order or recent-conversion paths are contradicted when team F5 or full-game runs, traffic, and extra-base hits all fall below their 10th percentiles.
- Late-bullpen exposure is contradicted when the forecasted segment is not exposed and starter workload reaches or exceeds its 80th percentile.
- Bullpen protection is contradicted when bullpen runs allowed exceed the 90th percentile in a game that enters the protected segment within three runs.

The review records component errors before assigning the final class. Traffic alone cannot mark a side or total forecast correct.

Calibration reports group predictions by model version, market, lineup status, starter structure, probability bucket, and priority-score bucket. Reports include sample size, hit rate, push rate, Brier score, log loss, ROI, and closing-line value. Missing odds prevent ROI and closing-line calculations but do not prevent flow review.

## Data and Leakage Controls

- Pregame snapshots may use only records available before their capture timestamp.
- Every source observation records `source_name`, stable source record or URL, `observed_at`, `available_at`, `fetched_at`, effective date, and content hash.
- Every normalized feature records the source-observation IDs and the latest `available_at` used in its calculation.
- Snapshot validation requires every feature's latest `available_at` to be at or before `captured_at`, and every lookback window to end before `captured_at`.
- Historical observations without a defensible `available_at` may be stored as retrospective data but are excluded from out-of-sample calibration.
- A postgame record never modifies its linked prediction snapshot.
- Historical evaluation uses walk-forward dates.
- Rules inspired by a reviewed date cannot be credited as out-of-sample performance on that date.
- Model changes create a new semantic model version.
- Production weights remain frozen until an explicitly approved sample threshold is met.

The initial validation target is at least 100 confirmed-lineup games. Weight tuning should wait for at least 300 market decisions per market family when feasible.

## Compatibility and Migration

The existing `analysis.model` API and fixed rules remain available and authoritative. New modules consume normalized inputs in shadow mode. The first-delivery adapter may copy legacy decisions and canonical evidence paths into the new snapshot for comparison, but it may not create or modify current market contexts. Existing unit tests remain regression tests.

Migration order:

1. Add schemas, append-only ledger, and deterministic IDs.
2. Add starter, offense, and bullpen distributions.
3. Add score simulation and market probability calculations.
4. Add postgame result ingestion and calibration reports.
5. Add the shadow adapter and record both legacy decisions and new probabilities during daily analysis.
6. Accumulate a frozen out-of-sample sample before deleting, merging, or reweighting legacy rules.

The current 83 priority-related configuration flags are not deleted in the first delivery. After calibration, duplicated flags can be consolidated into numeric features and a small set of hard safety gates through a separately approved migration.

## Error Handling

- Missing required identity, line, or pitcher-chain data fails validation with an explicit error.
- Missing optional matchup samples widen uncertainty and use documented priors.
- Conflicting starter sources remain visible in the snapshot.
- Invalid odds, impossible probability totals, negative innings, and mutable snapshot attempts are rejected.
- Unsupported three-way markets are recorded but not evaluated until explicit settlement semantics are configured.
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
- F5 tie, integer-line push, extra-inning ML, and full-game-total settlement.
- Postgame A/B/C/D classification fixtures.
- Leakage guards reject outcome timestamps preceding prediction timestamps.
- Leakage guards reject features whose latest source availability exceeds capture time.
- Existing fixed-model and manifest regression suites remain green.

## Success Criteria

- The same normalized input, market state, simulation configuration, and model version reproduce the same forecast-run ID and probabilities; each capture remains a distinct immutable event.
- Every final recommendation can be traced to one immutable pregame snapshot.
- Postgame review can separate matchup, starter, bullpen, conversion, and market errors without narrative reinterpretation.
- Priority scores are no longer presented as calibrated probabilities.
- Market performance can be evaluated by probability bucket, ROI, and closing-line value.
- No legacy rule is removed until out-of-sample evidence supports the change.
