# Data Sources and Authority

## Source Hierarchy

| Purpose | Primary Source | Notes |
|---|---|---|
| Schedule, game state, probable pitcher, live feed, boxscore | MLB official | Authoritative pitcher source |
| Confirmed and projected lineup | RotoWire Daily Lineups | Projected lineup lowers confidence only |
| Pitch mix, velocity, handedness splits, hitter response | Baseball Savant / Statcast | Use pitch type plus velocity together |
| Market odds and total | Current sportsbook or market page | Record decimal odds and timestamp |
| Weather and park | User-provided BallparkPal or equivalent | Excluded by default |
| Umpire | Available assignment and tendency source | Missing assignment caps F5 confidence |

## Data Handling

- Do not commit login credentials, cookies, sportsbook account data, or private reports.
- Do not commit large Statcast downloads or temporary HTML snapshots.
- Store reproducible source notes and timestamps in analysis reports when needed.
- When sources conflict, preserve the conflict as a confidence flag unless opener/bulk structure explains it.

## Cache Policy

| Source | Cache Policy | Reason |
|---|---:|---|
| Baseball Savant / Statcast historical pitch and hitter response data | 24 hours | Historical pitch-level samples are expensive to fetch and do not materially change within the analysis day. |
| MLB official schedule, probable pitchers, openers, bulk pitchers, live feed | Realtime | Starter and opener/bulk state can change close to first pitch. |
| RotoWire confirmed and projected lineups | Realtime | Lineups can move from projected to confirmed and can change after late scratches. |
| Market odds and O/U totals | Realtime | Prices and totals move continuously. |
| Boxscore and bullpen recent usage | Realtime | Bullpen availability depends on the latest completed or live game state. |

Only Savant/Statcast historical data may use daily caching by default. All other required sources must be freshly requested for each analysis run unless the user explicitly approves a different cache policy.

## Reproducibility Boundary

This repository preserves the model and its workflow. It does not guarantee that third-party sites expose the same live data forever, so every analysis should record the lineup status, starter status, market line, and data timestamp used.
