# MLB Analysis Project Instructions

This repository contains a fixed MLB game-analysis model. Before analyzing any game, read `docs/model.md` and follow it as the authoritative specification.

## Model Stability

- Do not modify the model, thresholds, source hierarchy, or output direction unless the user explicitly asks for a model change.
- When the user asks for a postgame review, identify whether a model change is justified, but do not apply it without explicit approval.
- Score each baseball input once and reuse it with different weights for F5, F5 O/U, ML, `+1.5`, and O/U. Do not double count evidence.
- Before final priority ranking, exclude every market-engine `pass` and use the unified same-game candidate assembler. Keep multiple markets only when their canonical evidence paths are distinct and non-empty.

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
- Evaluate F5 O/U independently from full-game O/U. Exclude full-game bullpen effects unless a verified opener/bulk arm is scheduled inside the first five innings.
- Require a verified F5 total and decimal price. Never infer the F5 line from the full-game total.

## Required Pregame Recording Behavior

- For a multi-game projected-lineup recording request, prepare every schema-v2/current bundle first and run `python -m analysis.operator batch-projected --manifest <path> --receipt <path>` instead of stopping after in-memory analysis or ad hoc Shadow output.
- For a multi-game confirmed-lineup analysis or freeze request, prepare every schema-v2/current bundle first and run `python -m analysis.operator batch-confirmed --manifest <path> --receipt <path>`.
- Do not say a slate is recorded, frozen, or complete unless the batch receipt has `complete: true` and `success_count == target_count`. Report `부분 기록` or `부분 동결` with the failed games and reasons otherwise.
- Include the receipt `N/N` status and each successful game's `capture_id`, `forecast_run_id`, and `capture_forecast_link_record_id` in the operational handoff. In-memory tables, reviews, and Shadow simulations without these audited IDs are not frozen records.
- Exact reruns are safe. If a batch is partial, rerun the same manifest and path overrides rather than fabricating or retroactively inserting missing pregame records.
- Admit F5 under only after explicitly verifying both scoring directions. Unresolved long-horizon starter WHIP/BB/HR/collapse risk blocks F5 under even when recent starts improved.
- Cap matchup-only or traffic-only side strength when recent run conversion is absent; bullpen quality cannot substitute for offensive conversion.
- For ML, require an independent full-game margin path beyond the F5 edge. For ML and `+1.5`, rested bullpen availability must also have current role-specific performance support.
- For fresh model-version `2026-08-03` F5 or full-game unders, bind same-namespace `starter_suppression` and `lineup_suppression` review paths before capture.

## Default Output

- Return F5, F5 O/U, ML, `+1.5`, and O/U unless the user requests a narrower market set.
- Keep the final direction consistent within each game.
- Include the market total and decimal price when available.
- Put umpire discussion in the written review rather than forcing it into the summary table.
- If data is missing or conflicting, state the limitation and cap confidence instead of inventing precision.

Use `python3 -m analysis.pipeline` to inspect the fixed analysis manifest and `docs/analysis-output-template.md` for the preferred response structure.
