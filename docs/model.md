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
- Do not keep a strong F5 recommendation when the starter is not confirmed close to game time.
- A projected lineup does not make a team weaker. It only lowers recommendation confidence.
- A `표본 부족` pitch-velocity result does not make a hitter weaker. It only raises matchup uncertainty.

### Opener and Bulk Pitchers

If the official probable pitcher is an opener and a separate bulk pitcher is clearly identified:

- Use the opener only as a first-inning or first-time-through-the-order variable.
- Use the bulk pitcher for pitch mix, velocity, lineup fit, F5, ML, and total analysis.
- Do not treat opener versus bulk labeling as a starter-source conflict.
- Do not treat a `PRIM`, rookie, or small-sample starter as a fade only because public MLB sample is thin.

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

Rules:

- Ace reputation does not override actual lineup fit.
- If the opponent's top five contain three or more clear `플러스` matchups, reduce the ace bonus.
- If a lineup has many `플러스` and many `마이너스` matchups, treat the result as mixed rather than clearly hitter-friendly.
- If six or more `플러스` slots are present but fewer than three sit in the top five, or fewer than two belong to clear power bats, downgrade the lineup edge.
- If the edge is concentrated mostly on fastballs and the starter has reliable secondary command, downgrade the lineup edge.
- If most same-handed hitters are `표본 부족`, treat the matchup as power variance rather than weakness.
- If an underdog lineup has seven or more `플러스` slots against a favorite starter, downgrade the favorite's full-game ML confidence.

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
- Throwing 30 or more pitches yesterday makes a high-leverage arm unavailable by default.
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
- If the case relies mostly on fastball edges and lower-order continuity is weak, cap the handicap at `lean`.
- If the fastball-only edge also lacks secondary-pitch support, pass.

### Totals

Totals use both teams' starter, lineup, and bullpen layers.

Under rules:

- Under requires both starters to be stable.
- Under requires both full bullpens, including lower depth, to be stable.
- Under requires both lineups to be muted against the expected matchup.
- One stable starter is not enough for a full-game under.
- `표본 부족` is not under support.
- Small-sample starters with command volatility reduce under confidence.
- In games of `7.0` or lower, meaningful late variance risk blocks the full-game under.
- If the starter window points under but the bullpen is unstable, treat the idea as F5-under logic only.

Over rules:

- Over requires starter collapse risk, live lineup fit, lower-bullpen exposure, or a favorable run environment.
- Positive matchup slots alone are not enough when recent power and walk pressure are weak.
- Starter-source conflicts are not over signals.
- Short-starter or bullpen-game labels do not automatically support over.
- If both starters are likely to reach the sixth inning and neither has clear collapse risk, do not upgrade over from lineup fit alone.

Low-total starter-duel rule:

- At market totals of `7.5` or lower, review the under before a heavy favorite ML when both starters are stable, both bullpens are usable, and both lineups are muted.

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

## Live Betting

Live recommendations should ignore stale pregame confidence when the game state materially changes.

Key live triggers:

- Starter exits before the third inning
- Six or more bullpen innings are needed
- Long-relief quality is below average
- A plus handicap offers enough cushion at a playable decimal price

Early starter exits block live ML unless the price is exceptional and long relief is stable.
