# MLB Analytical Model

This project scores MLB games by separating game-performance inputs from betting-market decisions.

This model is fixed for day-to-day analysis unless the user explicitly asks to change it.

## Core Principles

- Score each baseball input once, then reuse that score with different weights for F5, F5 O/U, full-game ML, `+1.5`, and full-game totals.
- Use RotoWire confirmed lineups when available and RotoWire projected lineups when they are not.
- Treat official MLB probable pitchers, live feeds, and boxscores as the authoritative pitcher sources.
- Treat pitch-type plus velocity splits as matchup signals, not standalone recommendations.
- Treat pitch-type plus velocity sample shortage as uncertainty, not as hitter weakness or automatic under support.
- Downgrade F5 before full-game ML when starter WHIP, BB, HR, or early-collapse risk is high.
- Use full-game ML only when the bullpen layer supports the starter and lineup edge.
- Include totals in the default output. Weather is included only when the user provides it or explicitly requests it.
- Let F5, F5 O/U, ML, `+1.5`, and O/U compete in the same final priority stack. Do not keep totals secondary when the total is cleaner than the side.
- Allow more than one market from the same game in the final priority stack when the edges are independent and non-conflicting, such as ML plus O/U or `+1.5` plus O/U.

## Model Flow

1. Data integrity
2. Starter-lineup matchup
3. Lineup form
4. Starter stability
5. Bullpen
6. Supporting context
7. Betting decision engines
8. Market value and final output

The first six stages create baseball-performance scores. The betting decision engines do not rescore the same evidence; they apply different weights to the existing scores.

## Legacy Authority and Shadow Forecasting

The deterministic forecasting engine runs in mandatory `shadow` mode. It records probabilities, fair prices, predictive distributions, and market comparisons beside the fixed model, but the legacy fixed model remains the only decision authority. Shadow output must never mutate, replace, reorder, upgrade, or downgrade a legacy candidate, its `play`/`lean`/`pass` decision, or the final priority stack.

Keep these output concepts separate:

- **Legacy admission decision:** fixed validation and the same-game candidate assembler determine `play`, `lean`, or `pass` and whether a market may enter final priority. A market-engine `pass` stays excluded regardless of shadow probability.
- **Legacy `0-100` priority score:** an ordinal rank for admitted legacy recommendations. It is **not a win probability**, model probability, confidence percentage, or fair-price input.
- **Shadow model probability and fair decimal price:** deterministic forecast outputs for the selected market. They describe the modeled outcome and its corresponding fair price; they do not admit or rank a legacy recommendation.
- **Uncertainty and predictive intervals:** distributions for scores, traffic, extra-base hits, workload, and other forecast components. They describe outcome dispersion and support objective postgame review; they are not admission decisions or the legacy rank score.

### Three-Stage Daily Operation

1. Run the complete legacy analysis with the RotoWire projected lineup when a confirmed lineup is unavailable, prepare a schema-v2 projected bundle with explicit `authority: "current"`, and run `.venv/bin/python -m analysis.operator projected --bundle <path>` to record the normalized pregame capture, deterministic shadow forecast, and exact capture-to-forecast link.
2. Refresh with the RotoWire confirmed lineup and current starter and market state, prepare a schema-v2 confirmed bundle with explicit `authority: "current"`, and run `.venv/bin/python -m analysis.operator confirmed --bundle <path>`. A genuine source-state event receives a new capture; a behavior-changing normalized input or semantic market change receives a new forecast run. An unchanged semantic state may link a later capture to the same deterministic forecast.
3. Put the exact frozen `capture_id`, `forecast_run_id`, and `capture_forecast_link_record_id` from the confirmed command into the schema-v1 review bundle with the official MLB result, then run `.venv/bin/python -m analysis.operator review --bundle <path>`. Review binds only to those records, performs the objective flow review, and builds the calibration report without forecasting again.

Fresh projected and confirmed envelopes use this contract, with `stage` set to the matching command:

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "confirmed"
}
```

Every schema-v1 projected or confirmed artifact is historical and replay-only, whether its candidates use the relaxed or strict v1 shape. It must be loaded from the authenticated managed-bundle root and match an exact existing capture, forecast, and link in the ledger. A pending schema-v1 pregame file that has never executed cannot create fresh state and must be regenerated as schema v2. Rejection and failed replay leave ledger and managed-bundle bytes unchanged.

The fixed MLB model version is `2026-08-05`; model version and bundle schema version are independent. This bundle authority boundary does not change MLB scoring, F5 side or F5 total engines, full-game market engines, the same-game candidate assembler, legacy decision authority, or mandatory Shadow behavior.

The `2026-08-05` contract inherits the prior ML bullpen review contract and the prior two-sided under review contract. The new starter-evidence and high-total-over requirements are additive fresh-capture checks; they do not remove or weaken those inherited bindings.

The operator derives capture idempotency intrinsically from the bundle schema version, stage, and exact capture command content except the supersession target. It searches that key before resolving lineage, so an exact replay returns the original immutable capture, forecast, link, and supersession target. New state appends a new capture and explicit supersession relationship; no prior ledger record is edited. Lower-level `record_shadow_run` remains an append-only API whose direct callers supply the capture key.

- every new projected or confirmed `play`/`lean` candidate must be review-eligible;
- `pass` candidates are excluded from the hard gate;
- static validation runs before simulation and dynamic validation before append;
- authenticated exact replay of managed schema-v1 pregame artifacts remains available only when the ledger already contains their exact capture, forecast, and link;
- a rejection writes no pregame ledger or managed-bundle artifact.

Live source collection and normalization remain manual or external in this delivery. Every normalized value must carry provenance to its source observations; collection timestamps alone do not make an input eligible for out-of-sample evaluation.
See `docs/shadow-operator.md` for bundle roles, path overrides, frozen-ID handoff, replay recovery, exit codes, and report publication paths.

### Forecast Feature Ownership and Double-Count Protection

The shadow offense forecast keeps the matchup-neutral offense base separate from the pitcher-specific pitch/velocity matchup:

- `projected_runs_per_9` is the matchup-neutral offense base and already owns recent-form and run-production information. `forecast_offense` derives `base_offense_multiplier` by dividing it by `league_runs_per_team_game` and clipping to the configured offense bounds.
- Pitcher-specific pitch/velocity response is not pre-applied to that base. `forecast_offense` weights each lineup slot's selected response by opposing-pitcher usage and batting-order plate-appearance share, then stores the resulting clipped `matchup_multiplier` separately for that pitcher.
- Resolve a switch hitter's expected side independently for every pitcher profile in an opener/bulk/PRIM chain. A right-handed pitcher uses the hitter's left-side pitch mix and response, while a left-handed pitcher uses the hitter's right-side values; never reuse the primary matchup pitcher's side for the whole chain.
- Sample each offense uncertainty source once. The lineup base log standard error belongs only to the base-offense latent, pitch/velocity sample shortage belongs only to that pitcher's matchup latent, and an unprofiled bullpen matchup uses only the configured unknown-bullpen matchup penalty. Their combined run-path variance arises from multiplying those independent latents, not by embedding the same sigma in more than one draw.
- Simulation applies the base offense multiplier, the applicable pitcher/segment matchup multiplier, and the run-prevention multiplier once when forming the half-inning run mean. It does not add a second offense total.
- Do not preblend a pitcher matchup into `projected_runs_per_9`, and do not multiply `recent_form_index` or another raw recent-form/run-production field into the forecast later. The implementation has no second recent-form multiplier.

### Pregame Principal-Path Binding

Before first pitch and before recording the capture, freeze the review contract for every recommended market:

- Freeze its `market_key` and exact legacy candidate identity. The current contract uses `legacy_candidate_index` plus `legacy_market`, pointing into the immutable `LegacyDecisionSnapshot` that also preserves the candidate score, `play`/`lean`/`pass` rating, evidence paths, and final priority.
- Freeze two to four unique canonical principal evidence paths. For each path, freeze its exact component IDs and its applicable deterministic contradiction-rule binding.
- In legacy model versions `2026-08-02`, `2026-08-03`, and `2026-08-05`, every recommended full-game ML candidate must declare and bind at least one of `bullpen_protection` or `late_bullpen_exposure`. Use the former when available late-inning protection supports the side and the latter when expected bullpen exposure limits or opposes it. Do not auto-inject a path that was not part of the pregame ML reasoning.
- In legacy model versions `2026-08-03` and `2026-08-05`, every recommended F5 or full-game under must declare and bind both `starter_suppression` and `lineup_suppression`. Their components must use the same away/home namespace so the two paths cover both scoring directions. Do not admit a one-sided under binding or infer the missing path after the result.
- In fresh `2026-08-05` captures, every playable full-game ML candidate must additionally declare and bind directionally valid `starter_suppression` or `starter_collapse` evidence for the selected side. A missing or mismatched starter path makes the candidate ineligible for priority and fresh review.
- In fresh `2026-08-05` captures, every playable full-game over with a verified market total of `9.0` or higher must additionally declare and bind two distinct canonical scoring paths covering both away and home scoring directions. A missing, duplicate, same-side, or mismatched route makes the candidate ineligible for priority and fresh review.
- Freeze the `ComponentDistribution` for every referenced component ID, including its `p05`, `p10`, `p20`, `median`, `p80`, `p90`, and `p95` summary. These values fix the central 80% (`p10`-`p90`) and central 90% (`p05`-`p95`) predictive intervals. Every bound ID must exist in that forecast's component distributions.

Postgame review may consume only these frozen bindings and intervals; it cannot invent, replace, or revise a path, component ID, rule, or interval after the outcome is known. A missing binding, candidate/market mismatch, unknown `market_key`, invalid path count or shape, or missing component distribution makes that market review-ineligible. Record the ineligibility reason rather than inferring a replacement review path.

Fresh projected and confirmed captures must use the current legacy model version. The Operator resolves an exact immutable replay first, so an old capture can still be replayed byte-for-byte; a stale version cannot create a new capture, forecast, link, or managed bundle.

## 1. Data Integrity

This layer determines whether the analysis inputs are reliable. It changes confidence, not team strength.

Inputs:

- Korean date to US game-date conversion
- Official MLB schedule and game state
- RotoWire confirmed or projected lineup status
- Official probable pitcher and RotoWire pitcher agreement
- Opener, bulk, `PRIM`, and bullpen-game identification
- Pitch-type plus velocity sample size
- Source freshness and cache policy

Rules:

- RotoWire remains the lineup source: confirmed lineup when posted, projected lineup when not posted.
- Official MLB probable pitchers override projected pitcher sources.
- Use daily caching only for Baseball Savant / Statcast historical pitch and hitter-response data.
- MLB official schedule, probable pitchers, opener/bulk status, RotoWire lineups, market odds and totals, and boxscore/bullpen usage must be requested realtime for each analysis.
- When official MLB probable pitchers and RotoWire disagree and no clear opener/bulk explanation resolves the conflict, downgrade F5 and full-game ML confidence.
- When the official probable pitcher and RotoWire disagree but the bulk or `PRIM` pitcher is clearly identified and used for the matchup, do not keep treating the game as an unresolved starter-source conflict.
- Do not keep a strong F5 recommendation when the starter is not confirmed close to game time.
- A projected lineup does not make a team weaker. It only lowers recommendation confidence.
- A `표본 부족` pitch-velocity result does not make a hitter weaker. It only raises matchup uncertainty.

### Opener and Bulk Pitchers

If the official probable pitcher is an opener and a separate bulk pitcher is clearly identified:

- Use the opener only as a first-inning or first-time-through-the-order variable.
- Use the bulk pitcher for pitch mix, velocity, lineup fit, F5, ML, and total analysis.
- Do not treat opener versus bulk labeling as a starter-source conflict.
- Do not treat a `PRIM`, rookie, or small-sample starter as a fade only because public MLB sample is thin.

If MLB's official probable pitcher and RotoWire's listed pitcher disagree:

- First check whether the MLB official pitcher is an opener, reliever, or short-stint arm.
- If the MLB official pitcher is an opener and RotoWire lists a different pitcher as the expected starter, treat the RotoWire pitcher as the bulk or real matchup pitcher when that structure is plausible.
- In that opener-plus-bulk case, do not downgrade the game as an unresolved starter-source conflict.
- Use the official MLB opener for first-inning and first-time-through-the-order risk only.
- Use the RotoWire bulk pitcher for the main pitch mix, pitch velocity, lineup fit, F5, ML, plus handicap, and O/U analysis.
- Do not treat a `PRIM` label by itself as stable bulk confirmation. Stable bulk treatment requires either explicit bulk confirmation or a recent workload of roughly 50 or more pitches.
- If a `PRIM` or bulk candidate lacks workload evidence and explicit confirmation, the candidate can still be used as the likely matchup pitcher, but keep starter-source uncertainty active and cap F5, ML, plus handicap, and O/U confidence.
- If the opener is damaged early but the identified bulk or `PRIM` pitcher has a stable middle-inning path, do not turn opener damage alone into a full-game side or over signal.
- If an opener/bulk structure is plausible but the bulk pitcher's timing or workload is unclear, downgrade F5 and full-game side confidence even when one bulk candidate can be used for the matchup.
- If no clear opener or bulk explanation resolves the disagreement, preserve the starter-source conflict flag and cap F5 and ML confidence.

## 2. Starter-Lineup Matchup

This layer owns all pitch-type plus velocity analysis. The matchup is scored once and then reused by every decision engine.

### Starter Profile

For each starter:

- Split pitch mix by opposing batter side.
- Record usage and average velocity for primary pitches.
- Track fastball, sinker, slider, changeup, cutter, curveball, sweeper, splitter, and other meaningful shapes.
- Use the batter side the hitter is expected to use against that pitcher.

### Batter Response

For every hitter in the lineup:

- Match the starter's pitch type and average velocity band.
- Use the exact band of average velocity `±1.5 mph` first.
- Use the same pitch type within `±3 mph` only as a fallback.
- Separate responses by pitcher handedness and batter side.
- Grade results as `플러스`, `중립`, `마이너스`, or `표본 부족`.

### Lineup Matchup Aggregation

The raw hitter slots are converted into one lineup-matchup score by checking:

- Total `플러스`, `중립`, `마이너스`, and `표본 부족` volume
- `플러스` concentration in the top five lineup spots
- `플러스` concentration among clear power bats
- Whether the edge is mostly fastball-only
- Whether the opposing starter can avoid the fastball edge with reliable secondary command
- Whether the lower lineup can extend innings and convert traffic into runs
- Whether a same-handedness stack with high uncertainty creates power variance
- Whether a neutral-heavy lineup can create traffic even without many `플러스` slots
- Whether `마이너스`-graded power bats still create one-swing variance

Rules:

- Ace reputation does not override actual lineup fit.
- Do not treat a lineup with six or more `중립` slots and two or fewer `마이너스` slots as weak only because it lacks `플러스` volume. Treat it as traffic-capable and cap the opposing side's confidence.
- If the opponent's top five contain three or more clear `플러스` matchups, reduce the ace bonus.
- If the opponent's top five contain three or more clear `플러스` matchups and the model-side starter has HR risk or collapse risk, downgrade the model-side ML and consider O/U or pass before side. If that starter also carries command or HR risk, block full-game ML priority rather than leaving it as a normal lean.
- If the model side is a very short favorite built mostly on an elite or near-elite starter, and the opponent's top five still contain two or more credible extra-base routes, cap F5 and full-game ML at `lean`. Elite starter stability reduces risk; it does not erase concentrated one-swing paths.
- If a lineup has many `플러스` and many `마이너스` matchups, treat the result as mixed rather than clearly hitter-friendly.
- If six or more `플러스` slots are present but fewer than three sit in the top five, or fewer than two belong to clear power bats, downgrade the lineup edge.
- If the edge is concentrated mostly on fastballs and the starter has reliable secondary command, downgrade the lineup edge.
- If most `플러스` slots come from fastball bands but the starter's primary secondary pitch is high-usage and recent command has rebounded, cap side and over strength unless the lineup also shows support against that secondary shape.
- If most same-handed hitters are `표본 부족`, treat the matchup as power variance rather than weakness.
- If a clear power bat is graded `마이너스`, do not treat that hitter as harmless. Use it as a variance flag for ML and totals, not as positive side evidence.
- If an underdog lineup has seven or more `플러스` slots against a favorite starter, downgrade the favorite's full-game ML confidence.
- If the model side's core run creators contain three or more `마이너스` pitch-velocity matchups, cap F5 strength even when the opposing starter's season line looks attackable.
- If the confirmed model-side lineup has broadly weak pitch-velocity fit and the model side is a short favorite at decimal odds of `1.65` or lower, cap full-game ML priority. Starter or bullpen edge can preserve a lean, but the short-priced ML should not remain a top-priority recommendation.

## 3. Lineup Form

This layer measures whether the lineup is currently converting matchup opportunities into runs.

Inputs:

- Recent seven-day team wOBA
- Extra-base hits and home runs
- Walk pressure
- Strikeout rate
- Direction of recent form: rising, stable, or falling
- Run efficiency
- Lower-order continuity

Rules:

- Recent form modifies the starter-lineup matchup score; it is not scored again inside F5 or ML.
- Do not convert `플러스` pitch-velocity volume into a strong side or over when recent power and walk pressure are weak.
- If a lineup has six or more `플러스` slots but recent run efficiency is poor, downgrade the lineup edge.
- If a lineup has many `플러스` pitch-velocity slots but has recently failed to convert traffic into runs, move ML toward pass or O/U review instead of forcing the side.
- If a lineup has many `플러스` pitch-velocity slots but also has a high recent strikeout rate against an unfamiliar, small-sample, or unusual-release starter, cap F5 and ML strength by one level. This is a conversion-risk rule, not a reversal to the other side.
- If recent form is driven mostly by isolated power but the lineup lacks walk pressure or lower-order conversion, cap F5 and ML strength. Recent wOBA alone does not prove sustainable run creation.
- If the lineup's recent run creation is broadly cold, downgrade ML first and review O/U before forcing the side. A starter edge does not guarantee a favorite ML when the lineup is not converting traffic into runs.
- A contact-oriented lineup can reduce confidence in a side that relies mainly on starter recent dominance.

## 4. Starter Stability

This layer measures starter quality, survival, and collapse risk.

Inputs:

- Recent ERA and WHIP
- Walks per nine
- Home runs per nine
- Pitches per inning
- First-inning risk
- Recent two- or three-start run prevention
- Strikeout-to-walk shape
- Expected innings and probability of reaching the sixth inning
- MLB sample size

Outputs:

- Starter stability score
- Starter survival score
- Command-risk flag
- HR-risk flag
- Collapse-risk flag
- Expected workload

Rules:

- Starter collapse risk downgrades F5 first.
- Recent two- or three-start improvement cannot erase unresolved long-horizon WHIP, BB, HR, or collapse risk. Keep that baseline-risk flag separate from the recent-form score and block F5 under while it remains active.
- Small-sample starters receive a command-risk penalty when walk shape is unstable.
- A small sample alone is not a fade.
- Strong recent run prevention can temper a lineup with many positive pitch-velocity slots.
- A stable `PRIM` or bulk pitcher is treated as a real starter-quality obstacle only when workload or explicit role evidence supports that stability.
- If a starter has strong recent strikeout-to-walk shape, low walk pressure, and a realistic sixth-inning path, treat that starter as a run-conversion suppressor. This caps opponent side strength and full-game over strength even when pitch-velocity `플러스` slots are present.
- If a starter's recent starts show controlled walks, strong strikeout-to-walk shape, and a realistic sixth-inning path, treat that as a full-game over cap. This is especially important when the over case is driven mainly by pitch-velocity `플러스` slots or season-level volatility rather than current command failure.
- For opener, bulk, or `PRIM` structures, a recent bulk workload of roughly 50 or more pitches with controlled walks is enough to delay lower-bullpen exposure. Treat that as an over cap unless the lineup has a clear early scoring path or the bulk pitcher has current command or HR risk.

## 5. Bullpen

This layer combines availability and quality into one bullpen score. Availability answers who can pitch; quality answers how well the available arms are expected to perform.

### Availability

Relievers are grouped by role:

- `closer`
- `setup`
- `high`
- `middle`
- `low`

Availability rules:

- Classify bullpen workload by role before using raw pitch totals. A 30+ pitch outing by a closer, setup arm, or high-leverage matchup reliever is materially different from a 30+ pitch outing by a long reliever or low-leverage arm.
- Pitching yesterday alone is not a downgrade when the reliever threw 20 or fewer pitches and is not on consecutive-day usage.
- Throwing 21-29 pitches yesterday creates a slight downgrade.
- Throwing 30 or more pitches yesterday makes a high-leverage arm limited with performance and usage risk, not automatically unavailable.
- Pitching on consecutive days limits the arm.
- Pitching three straight days makes a high-leverage arm unavailable by default.
- Multiple limited high-leverage arms create an additional bullpen downgrade.
- Lower-leverage or long-relief workload is tracked separately from closer, setup, and high-leverage availability. It should not be described as high-leverage fatigue.

### Quality by Inning Segment

Bullpen quality is split into:

- Bridge and long relief
- Setup and high leverage
- Closer and ninth inning

Rules:

- High-leverage availability matters more than raw total workload.
- Rested late-inning arms can outweigh heavy long-relief usage.
- When high-leverage arms are rested but low, middle, or long relief absorbed the previous game, treat ML and `+1.5` protection as preserved unless the current starter or bulk path is likely to force early lower-bullpen exposure.
- If only low, middle, or long relief absorbed a one-sided game while closer/setup/high-leverage arms rested, do not downgrade ML or `+1.5` for bullpen fatigue. Treat high-leverage availability as preserved.
- A lower-bullpen blowup is only a conditional late-total or blowout-tail risk when starter length is weak or lower-bullpen exposure is likely. It is not automatic support for the opponent's handicap.
- A weak closer or setup segment can cap full-game confidence even when the average bullpen score looks acceptable.
- Availability alone is not bullpen protection. Before using a rested late-inning group to support ML or `+1.5`, verify that its current performance and role-specific quality also support the close-game path.
- Lower-bullpen exposure matters for full-game totals and games where the starter is unlikely to work deep.
- Do not count weak or tired lower relief as full-game over support until starter length, bridge quality, or opener/bulk timing makes that lower-bullpen segment likely to appear. If the starter or bulk pitcher can cover the middle innings, lower-bullpen risk is only a late-tail note.
- The bullpen layer affects full-game ML, `+1.5`, and totals much more than F5.

## 6. Supporting Context

These inputs are secondary modifiers. Unknown values generally cap confidence rather than change the model side.

Inputs:

- Umpire zone tendency
- Defense and baserunning
- Catcher and battery impact
- Bench and substitute quality
- Bullpen handedness matching
- Park and weather, only when provided or requested

### Umpire Rules

- Narrow zone or high-walk umpire: downgrade starters with command risk.
- Wide zone or low-walk umpire: upgrade command-stable starters.
- Low-zone friendly umpire: upgrade sinker/changeup profiles.
- Outer-zone friendly umpire: upgrade slider/cutter or sweeper/slider profiles.
- Missing umpire assignment: keep the side, but cap F5 confidence.

### Weather and Park

- Weather is excluded by default.
- When the user provides BallparkPal or similar data, use it as the preferred run-environment source.

## 7. Betting Decision Engines

The decision engines reuse the layer scores above. They do not independently rescore pitch matchups, recent form, starter form, or bullpen usage.

### F5

F5 emphasizes:

- Starter-lineup matchup
- Starter stability and collapse risk
- Top-five early scoring support
- Umpire fit

Rules:

- Starter collapse risk downgrades F5 before ML.
- If early scoring support in the top five is weak, cap F5 at `lean`.
- If the starter source is unresolved, cap F5 confidence.
- Bullpen quality should not materially affect F5.

### Full-Game ML

Full-game ML emphasizes:

- F5 structure
- Lineup depth and recent run conversion
- Bullpen availability
- Bullpen quality by inning segment

Rules:

- Full-game ML may survive moderate starter volatility when the bullpen is rested and materially better.
- Late-inning bullpen risk can move a full-game ML recommendation toward F5 or `pass`.
- If the model side is a favorite at decimal odds of `1.60` or lower and the starter has command or HR risk, cap the side at `lean`.
- If a confirmed lineup removes a key power bat or the largest positive pitch-velocity slot, downgrade side priority at least one level.
- If the model side's starter is returning from injury or had a recent short exit, downgrade full-game ML as well as F5.
- If the best total case comes from two-sided run volatility, separate the over from the side and downgrade ML.
- If starter HR/collapse risk is identified and the opponent's top order has three or more pitch-velocity `플러스` slots, do not leave the model side as a primary ML recommendation. Move first to O/U when the run environment supports it, otherwise `pass` or reduce the side to a weak `lean`. If command or HR risk is also present, block full-game ML priority.
- If a short favorite's side case depends mainly on elite-starter trust but the opponent top order has multiple extra-base paths, keep the side as a capped lean rather than a priority play.
- If the opponent lineup is neutral-heavy with limited `마이너스` volume, do not use the absence of `플러스` slots as strong ML support. Cap or downgrade the side because traffic can still build runs.
- If the opponent has a `마이너스`-graded clear power bat, do not upgrade the model-side ML solely from that grade. Mark one-swing variance and reduce side confidence when the side is already thin.
- If the clearest edge is starter-window only, keep the recommendation as F5. Do not duplicate the same game into full-game ML priority unless the bullpen score, segmented bullpen quality, and late path clearly support the same side.
- Full-game ML requires an independent full-game margin path beyond the F5 edge. Without lineup conversion, starter length, or verified late protection capable of preserving or extending the margin, cap ML at `71` rather than promoting the first-five read.
- If the expected path requires the starter to work into or through the sixth before exposing a limited or risky bullpen segment, downgrade full-game ML and review F5 or O/U first.
- If the model-side starter is likely to exit before completing five innings or force four or more bullpen innings, cap full-game ML even when the high-leverage bullpen is rested. Keep F5 separate when the starter-window edge still holds.
- If the model-side starter has clear walk or traffic risk, keep the side as F5-only unless the bullpen quality and late path are strong enough to absorb that traffic. A correct starter-lineup read should not automatically become ML.
- Do not use the opponent's lower-bullpen-only usage as support for a full-game fade when that usage preserved their closer, setup, and high-leverage arms.
- Market movement does not create the baseball edge; it is reviewed after the model score is complete.

### Plus Handicap

The `+1.5` model is not a weaker ML recommendation. It requires:

- Starter survival
- A lineup capable of keeping the game close
- Bullpen protection
- A playable decimal price

Rules:

- Poor starter survival blocks `+1.5`, even with an attractive lineup matchup.
- Starter survival has priority over lineup fit in the `+1.5` model. A strong pitch-velocity edge cannot rescue a cover path when the underdog starter is unlikely to survive.
- Small-sample starters require a clearer survival score.
- Poor bullpen protection blocks `+1.5`.
- Own late-inning bullpen protection risk blocks `+1.5`, even when the starter and lineup matchup are playable. Do not rely on an underdog cover if the available setup, high-leverage, or closer path cannot reasonably protect a close game.
- Opponent late-inning risk can upgrade a lean, but cannot override poor survival.
- Opponent lower-bullpen damage does not count as opponent late-inning risk when their high-leverage arms were rested.
- If the model-side starter has a small MLB sample, walks per nine of `4.0` or higher, and strikeout-to-walk ratio below `2.0`, cap `+1.5` at `lean` even when the lineup matchup is attractive.
- If the model-side starter or bulk pitcher has a small MLB sample and the opponent's top order also has major pitch-velocity sample shortage, cap `+1.5` at `lean`. This is uncertainty on the cover path, not weakness for the opponent.
- If the model-side starter has repeated recent HR damage, cap `+1.5` at `lean` unless the survival score and bullpen protection are clearly strong.
- If the underdog starter has general command or HR risk, cap `+1.5` at `lean` even when the lineup matchup is attractive.
- If the underdog starter has clear collapse risk, block `+1.5` priority even when the lineup matchup is attractive.
- If the opposing starter has ace-level or near-ace current stability, strong K/BB shape, and a realistic sixth-inning path, downgrade `+1.5` unless the underdog has both top-five `플러스` concentration and recent run conversion.
- If the opposing starter can miss bats and realistically work deep, cap the underdog `+1.5` unless the underdog has at least three top-five `플러스` slots and recent run conversion. A few lower-order or fastball-only positives are not enough.
- If the underdog has fewer than two credible top-five scoring routes and the opposing starter has recent strikeout length, keep `+1.5` out of the final priority stack. Starter survival and a low-scoring script can support a review lean, but they are not enough for a priority cover without a real run path.
- A muted favorite offense or low-scoring script does not create a priority `+1.5` by itself. The underdog still needs an independent scoring path through top-order matchup fit, recent run conversion, walks/contact pressure, or a clearly exposed opposing bullpen segment.
- If the underdog's own recent run creation is cold and it lacks run conversion, pass `+1.5` even when the low-scoring or close-game script is plausible. A muted opponent offense is not enough by itself.
- If the underdog handicap requires the favorite starter to underperform but the favorite starter can miss bats and limit traffic, keep the handicap as a `lean` or `pass` instead of a priority recommendation.
- If the underdog starter has early collapse risk and the opponent top order has a live power path, avoid `+1.5`; review over first when total validation supports it.
- If the case relies mostly on fastball edges and lower-order continuity is weak, cap the handicap at `lean`.
- If the fastball-only edge also lacks secondary-pitch support, pass.

### F5 Totals

F5 O/U is a separate decision engine that reuses the existing starter, lineup, pitch-velocity, recent-form, and supporting-context scores with first-five weights.

- Require a verified current F5 total and decimal price. Never infer an F5 line from the full-game total.
- Exclude full-game bullpen availability and quality. Include a reliever only when a verified opener/bulk structure places that pitcher inside the first five innings.
- F5 under requires explicitly verified two-sided run suppression, two stable first-five pitching paths, controlled command and HR risk, muted early lineup paths, and no converted top-five matchup concentration.
- Unresolved long-horizon starter baseline risk blocks F5 under even when the latest two or three starts improved.
- F5 over requires a credible early scoring path supported by recent power or walk conversion, or by opposing starter command, HR, first-inning, injury-return, or short-exit risk.
- At F5 totals of `4.0` or lower, over requires credible early scoring paths for both teams.
- If both starters have strong K/BB shape and realistic five-inning floors, pass a thin F5 over that lacks starter risk.
- If opener-to-bulk timing inside the first five is not verified, pass F5 O/U rather than guessing the pitching chain.
- Full-game bullpen volatility cannot create an F5 over or block an otherwise qualified F5 under.

### Full-Game Totals

Totals use both teams' starter, lineup, and bullpen layers.

Totals priority rules:

- O/U is evaluated against sides, not after sides.
- If the favorite's lineup is weak but the side case rests mostly on starter superiority, review under before ML.
- If side confidence is thin because both teams have volatility, prefer O/U over forcing ML or `+1.5`.
- If two-sided starter or bullpen volatility creates a clearer run path than either side, upgrade over in the final priority stack.
- A total can outrank a side only after passing the under or over validation checks below.
- A side or handicap and a total from the same game can both enter the priority stack when each passes its own validation and the reasoning is not the same evidence double counted.
- Do not pair markets from the same game when one depends on a game script that weakens the other, such as a favorite ML requiring run suppression while the total recommendation requires broad two-sided scoring.
- When pairing a side or handicap with an under, require a coherent low-scoring win or cover path. Do not pair them if the side depends on opponent starter collapse, early multi-run separation, or the model side's offense driving most of the edge.
- When pairing a side or handicap with an over, require a separate over path beyond the side edge. Both teams must have a credible scoring path through lineup fit, walk/contact pressure, starter volatility, late bullpen exposure, or run environment.
- Do not add an over to a correct side simply because the model-side lineup has many `플러스` slots. If the side edge depends on the opposing offense being muted, downgrade the over or leave it out.
- If either starter projects for six or more quality innings with strong K/BB shape and the opposing lineup lacks recent walk/power conversion, cap the full-game over even when pitch-velocity slots look positive.
- At market totals of `9.5` or higher, pitch-velocity `플러스` volume is not enough for over priority. Require recent power or walk conversion plus at least one additional run path: two-sided command/collapse risk, meaningful lower-bullpen exposure, or a user-provided favorable run environment.
- If both starters show K/BB stability and likely sixth-inning length, downgrade high-total over before side markets unless lower-bullpen exposure or weather/run-environment support is clear.
- If both starters have a realistic sixth-inning path and the over case lacks either walk/on-base pressure or late-bullpen exposure, downgrade the over even when both lineups show pitch-velocity `플러스` volume.
- At market totals of `9.0` or higher, if both starters have a realistic five-inning floor, do not make the over a priority from lineup fit alone. Require extra support from two-sided command risk, lower-bullpen exposure, post-sixth bullpen exposure, recent power/walk conversion, or a user-provided favorable run environment.
- Before a full-game over enters the priority stack, identify a plausible F5 scoring path. At least one side must have top-order pitch-velocity fit plus recent conversion, or a starter command/HR/short-start risk that can create early traffic. Without that F5 path, keep the over as review-only even if late bullpen volatility exists.
- At extreme market totals of `13.5` or higher, do not upgrade over from park-adjusted scoring expectations or lineup fit alone. Require a clear two-sided collapse path: two-sided volatility plus command-risk starters, meaningful lower-bullpen exposure, or post-sixth bullpen exposure, supported by recent power or walk conversion.
- If one starter has recent zero-walk or low-walk starts with strong K/BB and a realistic sixth-inning path, do not let old WHIP/HR volatility or pitch-velocity `플러스` slots carry a full-game over by themselves. Keep the over only when the other side also adds an independent scoring path through walks/contact, power conversion, lower-bullpen exposure, or a user-provided run environment.
- When an opener, bulk, or `PRIM` structure is present, evaluate the bulk pitcher's likely middle-inning length before upgrading a high-total over. A stable bulk path can delay lower-bullpen exposure and temper the over.
- If the bulk or `PRIM` pitcher recently handled 50 or more pitches with acceptable walk shape, do not assume early lower-bullpen exposure. Upgrade the over only when the offense can score before the bullpen window or when the bulk pitcher has current command or HR risk.
- If the over case is mostly fastball-band `플러스` slots and the opposing starter's secondary command has recently rebounded, cap the over unless there is a separate walk, bullpen, or two-sided volatility path.
- If the over case is mostly pitch-velocity `플러스` volume but one lineup is recently failing to convert traffic into runs, block over priority unless there is clear lower-bullpen exposure, post-sixth bullpen exposure, or two-sided starter command risk.
- At market totals of `9.5` or higher, keep over as a lean candidate when one starter has clear BB/HR/short-start collapse risk and the opponent's lower bullpen is likely to be exposed, as long as recent power or walk conversion supports at least one live lineup path.
- At market totals of `10.5` or higher, pass over when one lineup's scoring path is weak. A high total needs two-sided lineup support, not one strong lineup plus a vague late-risk assumption.
- At market totals of `10.5` or higher, do not use bullpen fatigue as over support if an available bridge or long-relief path can absorb the tired arms. The over needs actual lower-bullpen exposure, not just yesterday's pitch counts.
- At totals of `7.0` or lower, one live lineup is not enough for over priority. Require two-sided scoring paths plus late-inning variance, because one shutdown starter can carry the game under the number.

Under rules:

- Under requires both starters to be stable.
- Under requires both full bullpens, including lower depth, to be stable.
- Under requires both lineups to be muted against the expected matchup.
- Under requires no clear concentration of center-order power slots against either starter.
- Under requires walk risk to be controlled by both starters and the full bullpen path.
- At low totals, pitch-velocity `플러스` slots alone do not block an under when both starters have recent length, stable WHIP, controlled K/BB shape, and both bullpens are stable.
- One stable starter is not enough for a full-game under.
- `표본 부족` is not under support.
- `마이너스` grades on clear power bats are not under support by themselves; they create volatility and can block a thin full-game under.
- Missing star bats are not under support by themselves when the remaining lineup depth still has live pitch-velocity, on-base, or power paths.
- Small-sample starters with command volatility reduce under confidence.
- In games of `7.0` or lower, meaningful late variance risk blocks the full-game under.
- If the starter window points under but the bullpen is unstable, treat the idea as F5-under logic only.
- At market totals of `7.5` or lower, recent starter command or HR volatility blocks a thin full-game under even when season-level starter scores look stable.
- At market totals of `7.5` or lower, recent opponent power or repeated HR-contact variance blocks a thin full-game under even when the pitch-velocity table is mostly neutral or mixed.
- At market totals of `7.5` or lower, a neutral-heavy lineup with limited `마이너스` volume blocks a thin under because traffic can build without obvious pitch-velocity `플러스` slots.
- At market totals up to `8.5`, a thin full-game under is also blocked by neutral-heavy traffic risk unless both starters have strong length paths and both full bullpens, including lower depth, are stable. One muted lineup is not enough.
- A weak bulk pitcher, unstable `PRIM`, or likely lower-bullpen collapse path blocks a thin full-game under, even when the top of the lineup is missing power.

Over rules:

- Over requires starter collapse risk, live lineup fit, lower-bullpen exposure, or a favorable run environment.
- Over requires at least one live lineup path; when the edge is volatility-driven, both teams should have a credible scoring path.
- Strict over validation requires both teams to have an on-base path through walks or contact and enough lower-order continuity to extend innings. Pitch-velocity `플러스` slots alone cannot carry an over.
- If the over case is mainly traffic/contact volume rather than HR, extra-base power, starter collapse, or clear post-sixth bullpen exposure, require recent power or run-conversion support before putting the over in the priority stack. High baserunner expectation alone is a lean, not a priority over.
- A pitch-velocity-driven over needs a first-five scoring route, not just a full-game contact projection. If the early route is weak and the over depends on late relief exposure, verify that the starter or bulk path actually reaches that lower-bullpen segment.
- A full-game over should not be upgraded from one-sided scoring only. If one lineup has the clear path but the other lacks recent conversion, on-base pressure, starter-collapse help, or late-bullpen exposure, pass or lower the over.
- A high market total is not an automatic pass. If both starters carry command risk, both teams have credible on-base paths, and two-sided volatility is clearer than the side, keep over in the priority stack even at `9.5` or higher.
- For totals of `9.5` or higher, require recent power or walk conversion before upgrading over. Do not treat unused traffic, isolated pitch-velocity positives, or weak recent run conversion as enough.
- For totals of `10.5` or higher, require both lineups to show a credible scoring path. If one side lacks pitch-velocity fit, on-base pressure, or recent conversion, pass the over even if the other side can score.
- For totals of `10.5` or higher, if the over depends on bullpen fatigue, verify that the bridge or long-relief segment cannot cover the middle innings. If it can cover, pass or lower the over.
- If both starters can suppress early scoring but one side has clear post-sixth bullpen exposure, do not let starter length automatically block the over. Keep over live when projected total is near the market and at least one lineup has a credible scoring path.
- Do not upgrade an over from same-game side confidence alone. The total must independently show two-sided scoring, starter collapse, bullpen exposure, or a clearly favorable run environment.
- Do not upgrade an over from starter season volatility alone when the starter's latest form shows better command and length than the season line. Recent K/BB and six-inning path can override a thin over case built on pitch-velocity fit.
- When one starter dominates the matchup through strikeouts, low walk pressure, and expected length, downgrade the over unless the opposing bullpen path is meaningfully exposed.
- When both starters have a credible five-inning floor at a market total of `9.0` or higher, downgrade the over unless a separate late or command-risk path exists. This rule caps only over priority; it does not automatically create under support.
- Cold recent run conversion blocks a pitch-velocity-driven over priority when bullpen exposure and two-sided command risk are absent. This is a pass rule, not an under upgrade.
- If a game script supports the model-side ML mostly through run prevention, avoid pairing that ML with over unless the opponent has a separate scoring path or late bullpen exposure.
- Positive matchup slots alone are not enough when recent power and walk pressure are weak.
- When over is driven by both teams' starter or bullpen volatility, prefer the total and avoid forcing a side.
- Starter-source conflicts are not over signals.
- Short-starter or bullpen-game labels do not automatically support over.
- A short official starter does not automatically support over when the expected bulk or `PRIM` pitcher can cover the middle innings with stable command.
- If both starters are likely to reach the sixth inning and neither has clear collapse risk, do not upgrade over from lineup fit alone.

Low-total starter-duel rule:

- At market totals of `7.5` or lower, review the under before a heavy favorite ML when both starters are stable, both bullpens are usable, and both lineups are muted.
- At market totals of `7.5` or lower, one dominant K/BB starter with a realistic length path can move under ahead of side even if the other starter is only moderate, as long as both lineups are muted, both bullpens are usable, and late HR/variance flags are controlled.
- At market totals of `7.5` or lower, do not use under as a priority recommendation unless late variance, bullpen depth, and center-order power variance are all controlled.
- At market totals of `7.5` or lower, do not let a starter-duel label override recent power or HR variance. The under remains playable only when those variance paths are controlled.
- At market totals of `7.5` or lower, if both starters are recently stable and likely to carry the game into the sixth inning, keep under as a candidate even when the lineups have moderate pitch-velocity `플러스` slots. This applies only when late variance, bullpen depth, high sample uncertainty, and clear power-bat variance are controlled.
- At market totals of `7.0` or lower, do not make over a priority only because the favorite lineup has strong pitch-velocity slots. Require the underdog lineup to have a credible scoring path as well.

## 8. Market Value and Final Output

Market data is applied after the baseball-performance scores are complete.

Inputs:

- Decimal odds
- Implied probability
- Market line movement
- Market signals
- Shadow model probability, fair decimal price, no-vig market probability, and edge for comparison only

Rules:

- Preserve the legacy price and validation logic as the authority for `play`, `lean`, `pass`, and final priority.
- Compare shadow model probability with the no-vig market probability in the written review only. This comparison cannot change legacy admission or rank.
- Market signals are review inputs, not automatic recommendations.
- If signals agree with the legacy model, they can confirm the existing recommendation.
- If signals disagree with the legacy model, review starter, bullpen, and price before changing direction. Shadow comparisons are not market signals for this rule.
- Do not double count market information inside the baseball-performance score.

### Confidence

Confidence is reduced by:

- Projected rather than confirmed lineup
- Unresolved starter source
- High pitch-velocity sample uncertainty
- Unknown supporting context
- Conflicting model layers
- Poor price

Confidence does not change which team is stronger. It changes whether the recommendation is `play`, `lean`, or `pass`.

### Default Output

When the user asks for analysis, return:

| Game | F5 | F5 O/U | ML | +1.5 | O/U | Umpire | Final |
|---|---:|---:|---:|---:|---:|---:|---|

Use decimal odds when discussing price. Avoid switching direction inside the same game summary. If the final choice is a plus handicap, frame the side as capable of winning or keeping the game close rather than mixing it with the opponent's ML.

Final priority rules:

- A market engine's `pass` decision is ineligible for the final priority stack regardless of its preliminary score. Do not convert a failed handicap or total validation back into a scored lean at the output stage.
- Select at most one full-game side-family market and at most one full-game total. Retain both as separate priority entries when each is non-pass, scores at least 72, has non-empty canonical evidence, and the side-total game scripts are explicitly validated as compatible. Show both in Final, for example ATL ML + O9.
- The global priority list remains score-sorted after same-game assembly.
- ML and a handicap never both enter by default, and over and under for the same horizon never both enter.
- F5 uses the same final family policy without changing its market engine.
- If F5 O/U and full-game O/U rely on the same scoring edge, keep only the higher-scoring market in the final priority stack. Keep both only when independent early and late paths are explicitly verified.
- Assemble F5 side, F5 total, full-game side family, and full-game total in one same-game selection stage. Use canonical evidence paths already produced by the fixed layers: starter suppression, starter collapse, lineup suppression, top-order matchup, recent run conversion, underdog run creation, late-bullpen exposure, and bullpen protection. `lineup_suppression` is review-only and does not add a baseball score.
- When same-game candidates share an evidence path, keep only the higher-scoring candidate. On an exact correlated tie, use the deterministic order `f5_side`, `f5_total`, `full_total`, `ml`, `plus_1_5`, then `minus_1_5`.
- Missing evidence paths do not prove independence. Multiple same-game markets may coexist only when every retained market has a distinct, non-empty evidence path.
- Every entry in the final priority stack should include a `0-100` priority score. This score is an output-ranking expression of the existing model edge, market value, data confidence, and layer agreement; it does not create a new analysis layer or rescore baseball inputs.
- Score guide: `80+` primary candidate, `72-79` playable candidate, `65-71` lean or price-dependent candidate, below `65` review-only unless the user asks for all leans.
- Apply one final priority-admission gate after market value. This gate may only cap the existing priority score; it must not add points or score any baseball input again.
- Cap a side at `74` when its edge is mainly pitch/velocity matchup or projected traffic and recent run conversion does not support that traffic becoming runs. A bullpen edge can protect a lead but cannot repair missing offensive conversion.
- For opener, bulk, or bullpen-game sides, verify the opener-to-bulk-to-bridge-to-high-leverage chain. If that full chain is not verified, cap F5, ML, and `+1.5` at `74` rather than treating the named bulk matchup as the whole game.
- For ML and `+1.5`, both high-leverage availability and current role-specific performance quality must protect the edge. If either fails, cap the full-game side at `71`; rested arms are not automatically reliable arms.
- For ML, require an independent full-game margin path beyond the first-five edge. Without one, cap ML at `71`; do not automatically redirect the capped pick to another market.
- Treat omitted bullpen-performance or full-game-margin verification as unverified, not as implicit support.
- Every playable full-game ML must have directionally valid `starter_suppression` or `starter_collapse` evidence for its selected side, with the matching canonical path frozen in the review binding. If that starter evidence is incomplete, cap ML at `71` and keep it out of priority.
- Every playable full-game over at a verified `9.0` or higher must have two distinct canonical scoring paths covering both teams' scoring directions, with matching frozen review bindings. If either direction is incomplete, cap the over at `69` and keep it out of priority. This fresh-review requirement is derived from the market and line, not a caller-supplied thesis flag.
- These evidence rules are negative-only admission caps: they do not rescore baseball inputs, award positive points, or change Shadow authority. Exact historical replay remains unchanged. HR-only `+1.5` evidence and Legacy-versus-Shadow disagreement are diagnostics only; neither changes admission, priority, or thresholds.
- If an over relies on a lineup with at least five `마이너스` slots and zero `중립` slots, require an independent lower-order or walk path, or a verified bullpen-over path. Without one, cap the over at `69`.
- Acceptable same-game combinations include side plus O/U and `+1.5` plus O/U.
- In the summary table, do not show both ML and `+1.5` as active side-family recommendations for the same game. Apply each exact legacy candidate's admission decision, then pass the admitted `PriorityMarketCandidate` values through the legacy same-game assembler. Display the assembler-retained side-family candidate; if both survive as independent candidates but only one display slot is allowed, use the higher legacy `0-100` priority score. An exact legacy-score tie uses the assembler's deterministic market order: `f5_side`, `f5_total`, `full_total`, `ml`, `plus_1_5`, then `minus_1_5`, so ML precedes `+1.5`. Shadow probability, fair price, no-vig probability, or shadow edge must not break the tie because Shadow cannot change or reorder legacy decisions.
- Avoid listing both ML and `+1.5` for the same team as separate priority entries unless the user explicitly asks for market alternatives, because that duplicates the same side edge.
- Keep every same-game combination directionally coherent and state why both markets can coexist.
- Same-game side plus over requires explicit validation that the side can win or cover while the opponent still contributes enough scoring, or that late bullpen exposure can lift the total without breaking the side.

## Objective Postgame Review

Review the official result against the frozen pregame component forecasts and the two to four principal evidence paths recorded for each recommended market. Central 80% and 90% predictive intervals are fixed before the game. Apply classes in strict precedence order `D > C > A/B`:

An official-result correction is a new immutable result revision with a new `source_revision` and `supersedes_record_id` pointing to the latest result record. Never edit the earlier result. Review and calibration use the latest non-superseded official result; review rejects a superseded revision as its selected result.

- `A`: the ticket wins, no principal path is contradicted, and at least 75% of principal component observations are inside their central 80% intervals.
- `B`: no principal path is contradicted and at least 60% of principal component observations are inside their central 80% intervals, but the result does not qualify for `A`. A loss or push can therefore be `B`.
- `C`: any principal path is contradicted by the fixed rules or fewer than 60% of principal component observations are inside their central 80% intervals, regardless of ticket result. A winning ticket with materially wrong flow is `C`.
- `D`: a supported, documented external event affects a principal-bound component and that component falls outside its central 90% interval. The exact event contract is defined below.

### Deterministic C Contradiction Rules

A principal path can be contradicted only when at least one component bound to that path falls outside its central 90% interval (`< p05` or `> p95`). After that common gate is met, only the matching fixed binding shape and rule may contradict the path:

- **Starter suppression:** pitcher-chain runs allowed are above `p90`. Because this path binds only that component, the common gate also requires the observation to be above `p95`.
- **Starter collapse:** pitcher-chain runs allowed are below `p10` and pitcher-chain workload is at or above its forecast median.
- **Lineup suppression:** the applicable F5 or full-game team runs, traffic, or extra-base hits finish above `p95`. A lower-tail result in another component cannot activate an upper-tail contradiction.
- **Top-order or recent conversion:** the applicable F5 or full-game team runs, traffic, and extra-base hits are all below `p10`.
- **Late-bullpen exposure:** the bound bullpen segment was not exposed and pitcher-chain workload is at or above `p80`.
- **Bullpen protection:** bullpen runs allowed are above `p90` and the game entered the protected close-game segment.

Normal sequencing variance is not a contradiction. Neither an isolated central-90 miss nor a fixed-rule shape without the central-90 gate is sufficient; both the common gate and the exact path rule must qualify.

### D External-Event Contract

The first delivery supports exactly `injury`, `ejection`, `defensive_error`, and `defensive_error_cluster`. `injury` and `ejection` have no unearned-run threshold. `defensive_error` and `defensive_error_cluster` require at least three unearned runs. Every event must affect a component bound to a principal path, include a stable non-empty `documented_source_observation_id`, and have that affected component outside its central 90% interval. Any other predeclared exogenous event type requires a future separately approved supported-type extension and cannot currently be assigned `D`.

If the legacy decision was `pass`, record component matches without assigning an A/B/C/D market class. Traffic alone is not a correct flow: a side or total path also needs its frozen run-conversion, extra-base-hit, starter, or bullpen components to satisfy the applicable review rules.

Calibration reports may group the frozen decisions by model version, market family, lineup state, starter structure, model-probability bucket, and legacy priority-score bucket. Keep sample size, hit and push rates, Brier score, log loss, ROI, and closing-line value distinct; missing prices omit price-dependent metrics rather than blocking flow review.

## Migration Gates

- Shadow forecasts have no production influence until at least 100 confirmed-lineup games have been collected and evaluated.
- Avoid tuning an individual market family before 300 decisions for that family when feasible.
- Production precedence, deletion of a legacy rule, or reweighting requires a separately approved design and a new semantic model version. A postgame review or calibration report cannot make that change by itself.

## Live Betting

Live recommendations should ignore stale pregame confidence when the game state materially changes.

Key live triggers:

- Starter exits before the third inning
- Six or more bullpen innings are needed
- Long-relief quality is below average
- A plus handicap offers enough cushion at a playable decimal price

Early starter exits block live ML unless the price is exceptional and long relief is stable.
