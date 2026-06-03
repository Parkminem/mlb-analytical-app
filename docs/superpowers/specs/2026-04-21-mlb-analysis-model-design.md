# MLB Analysis Model Design

## Goal

Add a reusable analysis core that separates F5 and full-game recommendations while accounting for bullpen fatigue, market totals, starter command risk, park/weather, and umpire information.

## Approach

The first version is a small Python package with pure scoring functions. It does not fetch live data directly. Instead, callers pass normalized inputs gathered from MLB Stats API, Baseball Savant, RotoWire, Action Network, weather, and umpire sources.

This keeps the model testable and lets us improve data collectors separately.

## Components

- `analysis.model.RelieverAppearance`: one reliever's recent usage.
- `analysis.model.BullpenAvailability`: score, unavailable arms, limited arms, and notes.
- `analysis.model.score_bullpen_availability`: evaluates recent bullpen usage.
- `analysis.model.StarterRecentForm`: starter quality and command-risk inputs.
- `analysis.model.score_starter_recent_form`: evaluates starter F5 reliability.
- `analysis.model.FullGameContext`: combines F5 edge, bullpen score, market total, and environment.
- `analysis.model.recommend_full_game`: returns ML and total guidance with pass/lean/play levels.
- `analysis.model.evaluate_line_value`: compares model probability against market implied probability.
- `analysis.model.calculate_confidence_score`: grades analysis completeness and caps confidence when caution flags remain.

## Rules

- High-leverage relievers are more important than low-leverage relievers.
- A fatigued bullpen should downgrade full-game ML and full-game under, not F5 edges.
- Pitch-type plus velocity samples below 10 PA are treated as weak evidence.
- Missing umpire data is neutral and should not block recommendations.
- Unstable inputs such as umpire tendency, bench quality, catcher/battery impact, and bullpen handedness begin as caution flags.
- Any unresolved caution flag caps confidence below high.

## Testing

Unit tests cover:

- Consecutive-day high-leverage usage lowers bullpen availability.
- High bullpen fatigue turns full-game ML from play/lean into pass.
- High bullpen fatigue blocks a full-game under even when starter/F5 factors are favorable.
- Small pitch-plus-velocity samples are down-weighted.
