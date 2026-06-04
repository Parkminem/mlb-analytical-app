# MLB Analysis Project Instructions

This repository contains a fixed MLB game-analysis model. Before analyzing any game, read `docs/model.md` and follow it as the authoritative specification.

## Model Stability

- Do not modify the model, thresholds, source hierarchy, or output direction unless the user explicitly asks for a model change.
- When the user asks for a postgame review, identify whether a model change is justified, but do not apply it without explicit approval.
- Score each baseball input once and reuse it with different weights for F5, ML, `+1.5`, and O/U. Do not double count evidence.

## Required Sources

- Use official MLB schedule, game state, probable pitchers, live feeds, and boxscores as the authoritative pitcher sources.
- Use RotoWire confirmed lineups when available. If a lineup is not confirmed, use the RotoWire projected lineup and lower confidence only.
- Use Baseball Savant or Statcast for pitch mix, average velocity, handedness splits, and hitter pitch-type plus velocity response.
- Use decimal odds.
- Include weather only when the user provides it or explicitly requests it.

## Required Analysis Behavior

- Identify opener, bulk, `PRIM`, and bullpen-game structures. Use the bulk pitcher for the main matchup when clearly identified.
- For every available lineup hitter, analyze the expected batter side against the opposing pitcher.
- Split each starter's pitch usage by opposing batter side, then match pitch type and average velocity band against every hitter.
- Use exact velocity band `±1.5 mph` first and same-pitch fallback `±3 mph` only when needed.
- Label hitter slots as `플러스`, `중립`, `마이너스`, or `표본 부족`. Never treat `표본 부족` as hitter weakness.
- Explain both teams' all-lineup pitch/velocity response, recent form, starter risk, bullpen availability and quality, and umpire impact when available.
- Avoid ambiguous claims based only on handedness. State whether a conclusion comes from actual matchup data or a limited-sample inference.
- If a confirmed lineup removes a key power bat or the strongest positive pitch-velocity slot, downgrade side priority.
- If a starter is returning from injury or had a recent short exit, downgrade full-game ML as well as F5.
- Treat a high-leverage reliever who threw 30+ pitches yesterday as limited with performance/use risk, not automatically unavailable. Keep three straight days as unavailable.
- If the over case is driven by volatility on both sides, prefer O/U and downgrade ML.

## Default Output

- Return F5, ML, `+1.5`, and O/U unless the user requests a narrower market set.
- Keep the final direction consistent within each game.
- Include the market total and decimal price when available.
- Put umpire discussion in the written review rather than forcing it into the summary table.
- If data is missing or conflicting, state the limitation and cap confidence instead of inventing precision.

Use `python3 -m analysis.pipeline` to inspect the fixed analysis manifest and `docs/analysis-output-template.md` for the preferred response structure.
