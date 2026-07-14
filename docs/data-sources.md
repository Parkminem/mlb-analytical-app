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

## Collection Boundary

Live source collection and normalization remain manual or external in this delivery. The forecasting core and `analysis.pipeline` do not scrape or authenticate to these sources. Before a shadow run, the operator or external collector must supply the normalized values and provenance; no value may be inferred from a later result or an undocumented source.

RotoWire confirmed lineups remain preferred. When only a RotoWire projected lineup is available, use it and lower confidence only. Use decimal odds. Weather remains excluded unless the user provides it or explicitly requests it.

## Provenance and Leakage

Every source observation must contain:

| Field | Required meaning |
|---|---|
| `source_record` | A stable identifier or URL for the exact source record |
| `observed_at` | The source observation's timestamp |
| `available_at` | The earliest defensible time the observation was available for pregame use |
| `fetched_at` | When the operator or collector retrieved it |
| `effective_date` | The game, slate, or statistical date to which it applies |
| `content_hash` | A stable hash of the captured source content |

Every normalized value must reference its source-observation IDs and record the latest `available_at` used; lookback features must also record their lookback end. Pregame validation requires those times to be at or before `captured_at`. Do not substitute `fetched_at` for an unknown `available_at`. When availability cannot be defended, retain the observation only as retrospective data and exclude its capture from out-of-sample calibration.

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

This repository preserves the model and its workflow. It does not guarantee that third-party sites expose the same live data forever, so every analysis must retain the full source provenance above together with lineup status, starter structure, market line, decimal prices, and capture time.
