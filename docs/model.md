# MLB Analytical Model

This project scores MLB games by separating game-performance inputs from betting-market decisions.

This model is fixed for day-to-day analysis unless the user explicitly asks to change it.

## Core Principles

- Score each baseball input once, then reuse that score with different weights for F5, full-game ML, `+1.5`, and totals.
- Use RotoWire confirmed lineups when available and RotoWire projected lineups when they are not.
- Treat official MLB probable pitchers, live feeds, and boxscores as the authoritative pitcher sources.
- Treat pitch-type plus velocity splits as matchup signals, not standalone picks.
- Treat pitch-type plus velocity sample shortage as uncertainty, not as hitter weakness or automatic under support.
- Downgrade F5 before full-game ML when starter WHIP, BB, HR, or early-collapse risk is high.
- Use full-game ML only when the bullpen layer supports the starter and lineup edge.
- Include totals in the default output. Weather is included only when the user provides it or explicitly requests it.
- Let F5, ML, `+1.5`, and O/U compete in the same final priority stack. Do not keep totals secondary when the total is cleaner than the side.
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

## 1. Data Integrity

This layer determines whether the analysis inputs are reliable. It changes confidence, not team strength.

Inputs:

- Korean date to US game-date conversion
- Official MLB schedule and game state
- RotoWire confirmed or projected lineup status
- Official probable pitcher and RotoWire pitcher agreement
- Opener, bulk, `PRIM`, and bullpen-game identification
- Pitch-type plus velocity sample size

Rules:

- RotoWire remains the lineup source: confirmed lineup when posted, projected lineup when not posted.
- Official MLB probable pitchers override projected pitcher sources.
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
- If the opponent's top five contain three or more clear `플러스` matchups and the model-side starter has HR risk or collapse risk, downgrade the model-side ML and consider O/U or pass before side.
- If a lineup has many `플러스` and many `마이너스` matchups, treat the result as mixed rather than clearly hitter-friendly.
- If six or more `플러스` slots are present but fewer than three sit in the top five, or fewer than two belong to clear power bats, downgrade the lineup edge.
- If the edge is concentrated mostly on fastballs and the starter has reliable secondary command, downgrade the lineup edge.
- If most same-handed hitters are `표본 부족`, treat the matchup as power variance rather than weakness.
- If a clear power bat is graded `마이너스`, do not treat that hitter as harmless. Use it as a variance flag for ML and totals, not as positive side evidence.
- If an underdog lineup has seven or more `플러스` slots against a favorite starter, downgrade the favorite's full-game ML confidence.
- If the model side's core run creators contain three or more `마이너스` pitch-velocity matchups, cap F5 strength even when the opposing starter's season line looks attackable.

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
- Small-sample starters receive a command-risk penalty when walk shape is unstable.
- A small sample alone is not a fade.
- Strong recent run prevention can temper a lineup with many positive pitch-velocity slots.
- A stable `PRIM` or bulk pitcher is treated as a real starter-quality obstacle.
- If a starter has strong recent strikeout-to-walk shape, low walk pressure, and a realistic sixth-inning path, treat that starter as a run-conversion suppressor. This caps opponent side strength and full-game over strength even when pitch-velocity `플러스` slots are present.

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

- Pitching yesterday alone is not a downgrade when the reliever threw 20 or fewer pitches and is not on consecutive-day usage.
- Throwing 21-29 pitches yesterday creates a slight downgrade.
- Throwing 30 or more pitches yesterday makes a high-leverage arm limited with performance and usage risk, not automatically unavailable.
- Pitching on consecutive days limits the arm.
- Pitching three straight days makes a high-leverage arm unavailable by default.
- Multiple limited high-leverage arms create an additional bullpen downgrade.

### Quality by Inning Segment

Bullpen quality is split into:

- Bridge and long relief
- Setup and high leverage
- Closer and ninth inning

Rules:

- High-leverage availability matters more than raw total workload.
- Rested late-inning arms can outweigh heavy long-relief usage.
- A weak closer or setup segment can cap full-game confidence even when the average bullpen score looks acceptable.
- Lower-bullpen exposure matters for full-game totals and games where the starter is unlikely to work deep.
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
- Late-inning bullpen risk can move a full-game ML pick toward F5 or pass.
- If the model side is a favorite at decimal odds of `1.60` or lower and the starter has command or HR risk, cap the side at `lean`.
- If a confirmed lineup removes a key power bat or the largest positive pitch-velocity slot, downgrade side priority at least one level.
- If the model side's starter is returning from injury or had a recent short exit, downgrade full-game ML as well as F5.
- If the best total case comes from two-sided run volatility, separate the over from the side and downgrade ML.
- If starter HR/collapse risk is identified and the opponent's top order has three or more pitch-velocity `플러스` slots, do not leave the model side as a primary ML pick. Move first to O/U when the run environment supports it, otherwise pass or reduce the side to a weak lean.
- If the opponent lineup is neutral-heavy with limited `마이너스` volume, do not use the absence of `플러스` slots as strong ML support. Cap or downgrade the side because traffic can still build runs.
- If the opponent has a `마이너스`-graded clear power bat, do not upgrade the model-side ML solely from that grade. Mark one-swing variance and reduce side confidence when the side is already thin.
- If the clearest edge is starter-window only, keep the recommendation as F5. Do not duplicate the same game into full-game ML priority unless the bullpen score, segmented bullpen quality, and late path clearly support the same side.
- If the expected path requires the starter to work into or through the sixth before exposing a limited or risky bullpen segment, downgrade full-game ML and review F5 or O/U first.
- Market movement does not create the baseball edge; it is reviewed after the model score is complete.

### Plus Handicap

The `+1.5` model is not a weaker ML pick. It requires:

- Starter survival
- A lineup capable of keeping the game close
- Bullpen protection
- A playable decimal price

Rules:

- Poor starter survival blocks `+1.5`, even with an attractive lineup matchup.
- Small-sample starters require a clearer survival score.
- Poor bullpen protection blocks `+1.5`.
- Opponent late-inning risk can upgrade a lean, but cannot override poor survival.
- If the model-side starter has a small MLB sample, walks per nine of `4.0` or higher, and strikeout-to-walk ratio below `2.0`, cap `+1.5` at `lean` even when the lineup matchup is attractive.
- If the model-side starter has repeated recent HR damage, cap `+1.5` at `lean` unless the survival score and bullpen protection are clearly strong.
- If the opposing starter has ace-level or near-ace current stability, strong K/BB shape, and a realistic sixth-inning path, downgrade `+1.5` unless the underdog has both top-five `플러스` concentration and recent run conversion.
- If the underdog handicap requires the favorite starter to underperform but the favorite starter can miss bats and limit traffic, keep the handicap as a lean or pass instead of a priority pick.
- If the underdog starter has early collapse risk and the opponent top order has a live power path, avoid `+1.5`; review over first when total validation supports it.
- If the case relies mostly on fastball edges and lower-order continuity is weak, cap the handicap at `lean`.
- If the fastball-only edge also lacks secondary-pitch support, pass.

### Totals

Totals use both teams' starter, lineup, and bullpen layers.

Totals priority rules:

- O/U is evaluated against sides, not after sides.
- If the favorite's lineup is weak but the side case rests mostly on starter superiority, review under before ML.
- If side confidence is thin because both teams have volatility, prefer O/U over forcing ML or `+1.5`.
- If two-sided starter or bullpen volatility creates a clearer run path than either side, upgrade over in the final priority stack.
- A total can outrank a side only after passing the under or over validation checks below.
- A side or handicap and a total from the same game can both enter the priority stack when each passes its own validation and the reasoning is not the same evidence double counted.
- Do not pair markets from the same game when one depends on a game script that weakens the other, such as a favorite ML requiring run suppression while the total recommendation requires broad two-sided scoring.
- When pairing a side or handicap with an over, require a separate over path beyond the side edge. Both teams must have a credible scoring path through lineup fit, walk/contact pressure, starter volatility, late bullpen exposure, or run environment.
- Do not add an over to a correct side simply because the model-side lineup has many `플러스` slots. If the side edge depends on the opposing offense being muted, downgrade the over or leave it out.
- If either starter projects for six or more quality innings with strong K/BB shape and the opposing lineup lacks recent walk/power conversion, cap the full-game over even when pitch-velocity slots look positive.
- At market totals of `9.5` or higher, pitch-velocity `플러스` volume is not enough for over priority. Require recent power or walk conversion plus at least one additional run path: two-sided command/collapse risk, meaningful lower-bullpen exposure, or a user-provided favorable run environment.
- If both starters show K/BB stability and likely sixth-inning length, downgrade high-total over before side markets unless lower-bullpen exposure or weather/run-environment support is clear.
- If both starters have a realistic sixth-inning path and the over case lacks either walk/on-base pressure or late-bullpen exposure, downgrade the over even when both lineups show pitch-velocity `플러스` volume.
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
- Small-sample starters with command volatility reduce under confidence.
- In games of `7.0` or lower, meaningful late variance risk blocks the full-game under.
- If the starter window points under but the bullpen is unstable, treat the idea as F5-under logic only.
- At market totals of `7.5` or lower, recent starter command or HR volatility blocks a thin full-game under even when season-level starter scores look stable.
- At market totals of `7.5` or lower, a neutral-heavy lineup with limited `마이너스` volume blocks a thin under because traffic can build without obvious pitch-velocity `플러스` slots.

Over rules:

- Over requires starter collapse risk, live lineup fit, lower-bullpen exposure, or a favorable run environment.
- Over requires at least one live lineup path; when the edge is volatility-driven, both teams should have a credible scoring path.
- Strict over validation requires both teams to have an on-base path through walks or contact and enough lower-order continuity to extend innings. Pitch-velocity `플러스` slots alone cannot carry an over.
- A high market total is not an automatic pass. If both starters carry command risk, both teams have credible on-base paths, and two-sided volatility is clearer than the side, keep over in the priority stack even at `9.5` or higher.
- For totals of `9.5` or higher, require recent power or walk conversion before upgrading over. Do not treat unused traffic, isolated pitch-velocity positives, or weak recent run conversion as enough.
- If both starters can suppress early scoring but one side has clear post-sixth bullpen exposure, do not let starter length automatically block the over. Keep over live when projected total is near the market and at least one lineup has a credible scoring path.
- Do not upgrade an over from same-game side confidence alone. The total must independently show two-sided scoring, starter collapse, bullpen exposure, or a clearly favorable run environment.
- When one starter dominates the matchup through strikeouts, low walk pressure, and expected length, downgrade the over unless the opposing bullpen path is meaningfully exposed.
- If a game script supports the model-side ML mostly through run prevention, avoid pairing that ML with over unless the opponent has a separate scoring path or late bullpen exposure.
- Positive matchup slots alone are not enough when recent power and walk pressure are weak.
- When over is driven by both teams' starter or bullpen volatility, prefer the total and avoid forcing a side.
- Starter-source conflicts are not over signals.
- Short-starter or bullpen-game labels do not automatically support over.
- If both starters are likely to reach the sixth inning and neither has clear collapse risk, do not upgrade over from lineup fit alone.

Low-total starter-duel rule:

- At market totals of `7.5` or lower, review the under before a heavy favorite ML when both starters are stable, both bullpens are usable, and both lineups are muted.
- At market totals of `7.5` or lower, do not use under as a priority pick unless late variance, bullpen depth, and center-order power variance are all controlled.
- At market totals of `7.5` or lower, if both starters are recently stable and likely to carry the game into the sixth inning, keep under as a candidate even when the lineups have moderate pitch-velocity `플러스` slots. This applies only when late variance, bullpen depth, high sample uncertainty, and clear power-bat variance are controlled.
- At market totals of `7.0` or lower, do not make over a priority only because the favorite lineup has strong pitch-velocity slots. Require the underdog lineup to have a credible scoring path as well.

## 8. Market Value and Final Output

Market data is applied after the baseball-performance scores are complete.

Inputs:

- Decimal odds
- Implied probability
- Model probability
- Market line movement
- Market signals

Rules:

- Compare model probability with implied probability to determine value.
- Market signals are review inputs, not automatic picks.
- If signals agree with the model, they can confirm the pick.
- If signals disagree, review starter, bullpen, and price before changing direction.
- Do not double count market information inside the baseball-performance score.

### Confidence

Confidence is reduced by:

- Projected rather than confirmed lineup
- Unresolved starter source
- High pitch-velocity sample uncertainty
- Unknown supporting context
- Conflicting model layers
- Poor price

Confidence does not change which team is stronger. It changes whether the recommendation is `pick`, `lean`, or `pass`.

### Default Output

When the user asks for analysis, return:

| Game | F5 | ML | +1.5 | O/U | Umpire | Final |
|---|---:|---:|---:|---:|---:|---|

Use decimal odds when discussing price. Avoid switching direction inside the same game summary. If the final choice is a plus handicap, frame the side as capable of winning or keeping the game close rather than mixing it with the opponent's ML.

Final priority rules:

- The final priority stack may include multiple entries from one game when they are different markets and independently validated.
- Every entry in the final priority stack should include a `0-100` priority score. This score is an output-ranking expression of the existing model edge, market value, data confidence, and layer agreement; it does not create a new analysis layer or rescore baseball inputs.
- Score guide: `80+` primary candidate, `72-79` playable candidate, `65-71` lean or price-dependent candidate, below `65` review-only unless the user asks for all leans.
- Acceptable same-game combinations include side plus O/U and `+1.5` plus O/U.
- Avoid listing both ML and `+1.5` for the same team as separate priority entries unless the user explicitly asks for market alternatives, because that duplicates the same side edge.
- Keep every same-game combination directionally coherent and state why both markets can coexist.
- Same-game side plus over requires explicit validation that the side can win or cover while the opponent still contributes enough scoring, or that late bullpen exposure can lift the total without breaking the side.

## Live Betting

Live recommendations should ignore stale pregame confidence when the game state materially changes.

Key live triggers:

- Starter exits before the third inning
- Six or more bullpen innings are needed
- Long-relief quality is below average
- A plus handicap offers enough cushion at a playable decimal price

Early starter exits block live ML unless the price is exceptional and long relief is stable.
