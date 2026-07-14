# F5 Total Engine Design

## Goal

Add first-five-inning totals as an independent market output without creating a second set of baseball evidence or leaking full-game bullpen effects into the early-inning projection.

## Scope

- Add `F5 O/U` to the default analysis output and final priority competition.
- Evaluate F5 totals only when a current market line is available.
- Reuse the fixed model's already-scored starter, lineup, pitch-velocity, recent-form, and supporting-context inputs with F5-total-specific weights.
- Keep full-game O/U behavior unchanged.
- Exclude weather unless the user provides it or requests it, consistent with the fixed model.

## F5 Total Inputs

The F5 total engine reuses these existing inputs:

- Official starter or verified opener/bulk structure
- Starter pitch mix and average velocity by expected batter side
- Every confirmed or projected lineup hitter's pitch-type plus velocity response
- Recent seven-day lineup contact, walks, strikeouts, power, and run conversion
- Starter recent form, command, HR risk, first-inning risk, and projected five-inning survival
- Umpire context when reliable tendency data is available
- Current F5 total and decimal price

Bullpen availability and quality do not support or oppose an F5 total unless an opener/bulk structure projects a relief pitcher to work inside the first five innings. In that case, only the verified first-five pitching chain is included.

## Decision Rules

### F5 Under

F5 under requires both first-five pitching paths to be stable, controlled walk and HR risk, and no strong early scoring concentration in either lineup. A full-game bullpen weakness cannot block an otherwise valid F5 under.

F5 under is downgraded or passed when:

- Either starter has recent command, HR, injury-return, short-exit, or first-inning collapse risk.
- Either lineup has a concentrated top-five pitch-velocity advantage with recent power or walk conversion.
- The opener-to-bulk timing inside the first five is uncertain.
- The market line is too low to absorb one early multi-run inning.

### F5 Over

F5 over requires a credible early scoring route. At least one lineup must combine top-order pitch-velocity fit or recent conversion with opposing starter command, HR, or collapse risk. For low F5 totals, both teams need a credible scoring path.

F5 over is downgraded or passed when:

- The case relies on late bullpen exposure.
- Pitch-velocity positives are not supported by recent traffic or run conversion.
- Both starters have strong recent K/BB shape and a realistic five-inning floor.
- The opener/bulk chain can cover the first five with stable command.

## Market And Priority Behavior

- Add an `F5 O/U` column to the summary table between `F5` and `ML`.
- Use decimal prices when available.
- If the F5 total or price cannot be verified, display `기준점 미확인` or `패스`; do not infer a line from the full-game total.
- Let F5 O/U compete with F5 side, ML, `+1.5`, and full-game O/U using the existing `0-100` priority scale.
- Do not add positive points at the final priority-admission step.
- If F5 O/U and full-game O/U use substantially the same scoring edge, include only the higher-scoring market in the final priority stack.
- Allow both only when their paths are independently explained, such as a stable F5 under combined with verified late-bullpen full-game over exposure.
- Same-game side and F5 O/U may both qualify when their game scripts are coherent and neither duplicates the other's evidence.

## Output

The default table becomes:

| Game | F5 | F5 O/U | ML | +1.5 | O/U | Final |
|---|---|---|---|---|---|---|

Written reviews explain why the F5 total differs from the full-game total, especially when bullpen exposure creates the difference.

## Testing

Tests will verify:

- Bullpen-only volatility cannot create an F5 over.
- Stable starters and muted lineups can produce an F5 under even when full-game bullpen risk blocks the full-game under.
- Starter command or HR risk plus an early lineup path can produce an F5 over.
- Missing F5 market data returns a pass rather than an inferred line.
- Correlated F5 and full-game totals do not both enter the priority stack without independent validation.
- The pipeline manifest and output template include `F5 O/U`.
