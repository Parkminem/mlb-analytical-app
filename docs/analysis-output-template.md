# Analysis Output Template

Use this structure for daily MLB analysis. Keep the summary concise, then support it with matchup evidence.

## Summary Table

| Game | F5 | ML | +1.5 | O/U | Final |
|---|---|---|---|---|---|
| Away vs Home | pick / lean / pass | pick / lean / pass | pick / lean / pass | line + direction / pass | one or more non-conflicting choices |

Use decimal odds when prices are available.

The final priority stack may include more than one market from the same game when the edges are independently validated, such as ML plus O/U or `+1.5` plus O/U. Do not duplicate the same side edge by listing both ML and `+1.5` unless the user asks for alternatives.

Add a `0-100` priority score to each final priority entry. The score expresses the existing model edge, price/value, data confidence, and agreement across layers. It is not a separate baseball input and must not double count evidence.

When a same-game side or handicap plus O/U combination is listed, explain the independent total path. Do not list the total only because the side was strong.

## Game Review

### Away vs Home

**Lineup status:** RotoWire confirmed lineup or RotoWire projected lineup  
**Starter structure:** confirmed starter, opener/bulk, `PRIM`, or bullpen game  
**Confidence flags:** projected lineup, starter conflict, high sample uncertainty, missing umpire, or none

When MLB and RotoWire list different pitchers, state whether this is an unresolved conflict or an opener/bulk structure. If MLB lists an opener and RotoWire identifies the bulk pitcher, use the bulk pitcher for the main matchup and mention the opener only as first-inning context.

**Starter vs opposing lineup**

- Describe the starter's primary pitches and average velocity by opposing batter side.
- Summarize every hitter's pitch-type plus velocity slots.
- State the totals for `플러스`, `중립`, `마이너스`, and `표본 부족`.
- Explain top-five concentration, power-bat concentration, neutral-heavy traffic risk, negative-graded power-bat variance, fastball-only edges, and secondary-pitch risk.
- If the model side is a short favorite, state whether core hitters have three or more `마이너스` pitch-velocity matchups.

**Reverse matchup**

- Apply the same all-lineup process to the other starter and lineup.

**Recent form and starter stability**

- Compare seven-day lineup form, power, walk pressure, strikeout rate, and run efficiency.
- State whether pitch-velocity `플러스` volume is being converted into runs or only creating unused traffic.
- If pitch-velocity `플러스` volume is high but the lineup's recent strikeout rate is high against an unfamiliar or unusual-profile starter, state whether that caps F5/ML strength.
- Explain starter WHIP, BB, HR, first-inning, workload, and collapse risk.
- State whether either starter's K/BB shape and likely sixth-inning path can suppress pitch-velocity `플러스` slots.
- For `+1.5`, state whether small-sample command risk or repeated recent HR damage caps the side.

**Bullpen**

- State which closer, setup, and high-leverage arms are normal, limited, or unavailable.
- Explain bridge, setup, and closer quality separately when it affects ML, `+1.5`, or O/U.
- If the starter is expected to reach the sixth, state whether post-sixth bullpen exposure still downgrades ML or keeps over live.

**Umpire and context**

- Include umpire fit when available.
- Include weather only when provided or requested.

**Conclusion**

- Explain why F5 and ML may differ.
- If the edge is F5-only, state whether bullpen and late path are strong enough to keep ML. If not, avoid duplicating ML into the final priority.
- Explain why `+1.5` is playable or blocked.
- Explain why O/U is a play, lean, or pass at the current line.
- For overs, state whether both teams have an on-base path and lower-order continuity, not just pitch-velocity plus slots.
- For overs, if both starters can realistically reach the sixth inning, state whether walk/on-base pressure or late-bullpen exposure is strong enough to keep the over live.
- For same-game side plus over, state why the opponent can still contribute enough scoring, or why late bullpen exposure can lift the total without breaking the side.
- For high totals, state whether recent power or walk conversion plus command/collapse risk, lower-bullpen exposure, or run environment keeps the over live despite the number.
- For low totals, state whether starter length, recent command/HR stability, neutral-heavy traffic, and bullpen stability temper or block pitch-velocity plus-slot risk.
- For `+1.5`, state whether the opposing starter's K/BB shape and sixth-inning path cap the handicap even when the underdog has positive matchup slots.
- End with one or more consistent final directions when separate markets from the same game both deserve priority.
