# Analysis Output Template

Use this structure for daily MLB analysis. Keep the summary concise, then support it with matchup evidence.

## Summary Table

| Game | F5 | F5 O/U | ML | +1.5 | O/U | Final |
|---|---|---|---|---|---|---|
| Away vs Home | play / lean / pass | line + direction + play/lean/pass | play / lean / pass | play / lean / pass | line + direction + play/lean/pass | one or more non-conflicting choices |

Use decimal odds when prices are available.

In the summary table, show only one side-family market per game. Apply legacy admission to the exact ML and `+1.5` candidates, then use the legacy same-game assembler. Display its retained side-family candidate; if both independent candidates survive but only one display slot is allowed, use the higher legacy `0-100` priority score. For an exact-score tie, the assembler's fixed order places `ml` before `plus_1_5`. Mark the other side-family column as `pass` or `review only` and explain it when useful. Shadow probability, fair price, no-vig probability, and shadow edge cannot choose or break a tie for the displayed side-family market.

Select at most one full-game side-family market and at most one full-game total. Retain both as separate priority entries when each is non-pass, scores at least 75, has non-empty canonical evidence, and the side-total game scripts are explicitly validated as compatible. Scores `65-74` remain visible leans but do not enter the final priority list. Show both in Final, for example ATL ML + O9.

The global priority list is score-sorted after same-game assembly. ML and a handicap never both enter by default, and over and under for the same horizon never both enter. F5 uses the same final family policy without changing its market engine.

F5 O/U must use a verified F5 line and decimal price. Do not infer it from the full-game total. If F5 O/U and full-game O/U use the same scoring edge, list only the higher-scoring one in the final priority stack unless independent early and late paths are explicitly verified.

Before building the final priority stack, remove every market whose own decision engine returned `pass`, regardless of its preliminary score. Then apply the same family policy to F5 side, F5 O/U, full-game side, and full-game O/U. If candidates share starter suppression, starter collapse, top-order matchup, recent conversion, or another canonical evidence path, retain only the higher-scoring candidate. Missing evidence does not establish independence; keep multiple markets only when their distinct non-empty paths are explicitly verified.

Add a `0-100` priority score to each final priority entry. The score expresses the existing model edge, price/value, data confidence, and agreement across layers. It is a legacy rank score, **not a probability**, and is not a separate baseball input. It must not double count evidence.

When a same-game side or handicap plus O/U combination is listed, explain the independent total path. Do not list the total only because the side was strong.

## Pregame Principal-Path Binding

Before first pitch and before recording the capture, freeze the review contract for every recommended market:

- Record the `market_key` and exact legacy candidate identity: the current contract uses `legacy_candidate_index` plus `legacy_market`, with the candidate's score and `play`/`lean`/`pass` rating and the final priority preserved in the snapshot.
- Record two to four unique canonical principal evidence paths.
- For every recommended full-game ML in model versions `2026-08-02`, `2026-08-03`, `2026-08-05`, and `2026-09-11`, declare and bind `bullpen_protection` or `late_bullpen_exposure` according to the actual pregame ML reasoning. Never add a bullpen path after the result or merely to satisfy the contract.
- For every recommended F5 or full-game under in model versions `2026-08-03`, `2026-08-05`, and `2026-09-11`, declare and bind `starter_suppression` plus `lineup_suppression` with the same away/home namespace. This is the required two-sided review coverage; never freeze a one-sided under path.
- For a current `2026-09-11` full-game ML `play`/`lean` candidate with score `75` or higher, declare and bind a directionally valid starter path. `starter_suppression` means the selected side's own starter prevents runs; `starter_collapse` means the opposing starter fails. Without either exact route, the ML is capped at `71`, and a fresh bundle submitted above that cap is rejected.
- For a current `2026-09-11` verified full-game over line of `9.0` or higher with a `play`/`lean` score of at least `75`, bind two distinct canonical paths covering both away and home scoring directions. A conversion path (`top_order_matchup`, `recent_run_conversion`, or `underdog_run_creation`) scores for its batting side, while `starter_collapse` or `late_bullpen_exposure` scores for the side opposing the bound pitching side. Suppression/protection paths do not count as over scoring routes. Without both directions, the over is capped at `69`, and a fresh bundle submitted above that cap is rejected.
- For a current F5 over at score `75` or higher, bind `recent_run_conversion`; starter collapse alone cannot replace current power, walk, or run conversion. Missing conversion caps the candidate at `74`.
- For a current full-game under at score `75` or higher, both lineups must have included recent form below index `1.0` and matchup-neutral projected runs per nine below the configured league baseline. Either matchup pitcher's command, HR, injury-return, or recent-short-exit risk blocks the under. Missing suppression or present volatility caps it at `69`.
- Bind every path to exact component IDs and its exact deterministic contradiction rule.
- Freeze each bound `ComponentDistribution`, including `p05`, `p10`, `p20`, `median`, `p80`, `p90`, and `p95`; these fix the central 80% and 90% predictive intervals.

Postgame review cannot invent or change these paths, IDs, rules, or intervals. A missing/invalid binding, candidate/market mismatch, unknown market, invalid path count or shape, or missing component distribution makes that market review-ineligible; state the reason and do not infer a substitute. See [Pregame Principal-Path Binding](model.md#pregame-principal-path-binding) for the authoritative contract.

Use legacy model version `2026-09-11` for every fresh projected or confirmed capture. A stale model version is accepted only when the Operator finds and returns its exact immutable historical replay before fresh-capture validation.

## Operator Bundle Handoff

Fresh projected and confirmed commands require schema v2 with explicit current authority. Use the matching stage; a confirmed envelope begins:

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "confirmed"
}
```

Review remains schema v1 and receives the exact frozen IDs from the successful confirmed command. Every pregame schema-v1 shape, relaxed or strict, is historical exact replay only: it must come from the authenticated managed root and match an existing ledger capture, forecast, and link. Regenerate any pending unexecuted pregame v1 file as schema v2/current; rejection must not change ledger or managed bytes.

Bundle schema and model version are independent. The v2 handoff uses model `2026-09-11`; its approved negative-only total gates and `75`-point priority floor do not change Shadow recommendation authority or historical replay.

## Game Review

### Away vs Home

**Lineup status:** RotoWire confirmed lineup or RotoWire projected lineup  
**Starter structure:** confirmed starter, opener/bulk, `PRIM`, or bullpen game  
**Confidence flags:** projected lineup, starter conflict, high sample uncertainty, missing umpire, or none

When a shadow forecast is available, add the following block to the written review for the evaluated market. Do not add it as a replacement recommendation column in the summary table.

```text
Shadow probability: 54.2%
Fair decimal price: 1.84
Market no-vig probability: 51.0%
Shadow edge: +3.2 percentage points
Legacy authority: play/lean/pass and final priority remain unchanged
```

Keep the meanings separate: shadow probability and fair decimal price are model outputs; market no-vig probability and shadow edge are price comparisons; predictive intervals describe uncertainty in forecast components; and the legacy admission decision determines `play`/`lean`/`pass`. The `0-100` legacy rank score is not any of these. For a push-capable market, the displayed model and no-vig probabilities are conditional on a non-push result, and shadow edge is their difference in percentage points. If an opposing price or another required value is missing, state that the corresponding comparison is unavailable instead of inventing it.

Mandatory shadow mode cannot mutate, replace, reorder, upgrade, or downgrade any legacy recommendation. A strong or weak shadow edge is written context only.

When MLB and RotoWire list different pitchers, state whether this is an unresolved conflict or an opener/bulk structure. If MLB lists an opener and RotoWire identifies the bulk pitcher, use the bulk pitcher for the main matchup and mention the opener only as first-inning context.
Do not treat a `PRIM` label alone as stable bulk confirmation. State whether explicit bulk confirmation or a recent workload of roughly 50 or more pitches supports stable bulk treatment; otherwise mark opener/bulk timing uncertainty.

**Starter vs opposing lineup**

- Describe the starter's primary pitches and average velocity by opposing batter side.
- Summarize every hitter's pitch-type plus velocity slots.
- State the totals for `플러스`, `중립`, `마이너스`, and `표본 부족`.
- Explain top-five concentration, power-bat concentration, neutral-heavy traffic risk, negative-graded power-bat variance, fastball-only edges, and secondary-pitch risk.
- If the model side is a short favorite, state whether core hitters have three or more `마이너스` pitch-velocity matchups.

**Reverse matchup**

- Apply the same all-lineup process to the other starter and lineup.

**Recent form and starter stability**

- For shadow forecasting, keep `projected_runs_per_9` matchup-neutral; it already owns recent-form and run-production information. Keep pitcher-specific pitch/velocity response in the separate matchup multiplier, and do not apply recent form again later.
- Compare seven-day lineup form, power, walk pressure, strikeout rate, and run efficiency.
- State whether pitch-velocity `플러스` volume is being converted into runs or only creating unused traffic.
- If pitch-velocity `플러스` volume is high but the lineup's recent strikeout rate is high against an unfamiliar or unusual-profile starter, state whether that caps F5/ML strength.
- Explain starter WHIP, BB, HR, first-inning, workload, and collapse risk.
- State whether either starter's K/BB shape and likely sixth-inning path can suppress pitch-velocity `플러스` slots.
- If the model-side starter projects for fewer than five completed innings or creates four or more bullpen innings, state that full-game ML is capped separately from F5.
- If the model-side starter has clear walk or traffic risk, state whether the side is F5-only or whether the bullpen is strong enough to keep ML.
- For `+1.5`, state whether starter survival, general command/HR risk, small-sample command risk, or repeated recent HR damage caps or blocks the side.

**Bullpen**

- State which closer, setup, and high-leverage arms are normal, limited, or unavailable.
- Explain bridge, setup, and closer quality separately when it affects ML, `+1.5`, or O/U.
- If the starter is expected to reach the sixth, state whether post-sixth bullpen exposure still downgrades ML or keeps over live.

**Umpire and context**

- Include umpire fit when available.
- Include weather only when provided or requested.

**Conclusion**

- Exclude every market-engine `pass` before ranking or assigning final priority placement.
- Before assigning the final `0-100` score, apply the priority-admission caps using the already calculated run-conversion, pitching-chain, and bullpen results. Do not rescore or positively boost any input at this step.
- Explain why F5 and ML may differ.
- Explain why F5 O/U and full-game O/U may differ. Full-game bullpen risk is not an F5-total input unless a verified opener/bulk arm works inside the first five innings.
- If the edge is F5-only, state whether bullpen and late path are strong enough to keep ML. If not, avoid duplicating ML into the final priority.
- Explain why `+1.5` is playable or blocked. When ML and `+1.5` both pass legacy admission, use only their exact legacy candidates, legacy `0-100` priority scores, and the legacy same-game assembler to choose the one displayed side-family market. Shadow probability cannot choose a displayed side-family market.
- Explain why O/U is a play, lean, or pass at the current line.
- For overs, state whether both teams have an on-base path and lower-order continuity, not just pitch-velocity plus slots.
- For overs, if both starters can realistically reach the sixth inning, state whether walk/on-base pressure or late-bullpen exposure is strong enough to keep the over live.
- For high-total overs of `10.5` or higher, state whether both lineups have independent scoring paths. If one lineup path is weak, mark the over as pass or review-only.
- For high-total overs of `10.5` or higher, if bullpen fatigue is part of the over case, state whether available bridge or long relief can absorb the tired arms.
- For same-game side plus over, state why the opponent can still contribute enough scoring, or why late bullpen exposure can lift the total without breaking the side.
- For high totals, state whether recent power or walk conversion plus command/collapse risk, lower-bullpen exposure, or run environment keeps the over live despite the number.
- For low totals, state whether starter length, recent command/HR stability, neutral-heavy traffic, and bullpen stability temper or block pitch-velocity plus-slot risk.
- For low totals, state whether recent power or repeated HR-contact variance blocks the under even if the pitch-velocity table is mostly neutral or mixed.
- For `+1.5`, state whether the opposing starter's K/BB shape and sixth-inning path cap the handicap even when the underdog has positive matchup slots.
- For `+1.5`, state whether the underdog starter's survival risk overrides an otherwise positive pitch-velocity matchup.
- For `+1.5`, state whether the underdog has its own run-creation path. Do not make the handicap a priority from opponent run suppression alone.
- End with one or more consistent final directions when separate markets from the same game both deserve priority.

## Postgame Flow Review

Use the official MLB result and the exact frozen pregame capture-to-forecast link. For every recommended market, compare the observed components with the predeclared principal paths and predictive intervals, then apply the fixed precedence `D > C > A/B`:

Use only the latest non-superseded official result. A correction appends a new immutable revision with a new `source_revision` and `supersedes_record_id`; never edit the old result. Review and calibration select the latest non-superseded revision.

- `A`: win, no contradicted principal path, and at least 75% of principal components inside the central 80% intervals.
- `B`: no contradicted principal path and at least 60% inside the central 80% intervals, but not `A`; losses and pushes can be `B`.
- `C`: any contradicted principal path or fewer than 60% inside the central 80% intervals, even if the ticket won.
- `D`: a supported, documented external event affects a principal-bound component outside its central 90% interval.

For a `C` path contradiction, first require at least one bound component outside the central 90% interval, then apply exactly one fixed rule: starter suppression requires pitcher-chain runs above `p90`; starter collapse requires runs below `p10` and workload at or above the median; lineup suppression requires its own runs, traffic, or extra-base-hit component above `p95`; top-order/recent conversion requires runs, traffic, and extra-base hits all below `p10`; late-bullpen exposure requires no bound-segment exposure and workload at or above `p80`; bullpen protection requires bullpen runs above `p90` and protected close-game entry. Normal sequencing variance or either condition by itself is not a contradiction. See [Deterministic C Contradiction Rules](model.md#deterministic-c-contradiction-rules) for the authoritative contract.

The first delivery supports exactly `injury`, `ejection`, `defensive_error`, and `defensive_error_cluster` for `D`. Injury and ejection do not require unearned runs; `defensive_error` and `defensive_error_cluster` require at least three. Every event must affect a principal-bound component, include a stable non-empty `documented_source_observation_id`, and have that component outside its central 90% interval. Other predeclared exogenous event types require a future approved supported-type extension and are not currently assignable as `D`.

If the legacy market was `pass`, report component matches but no A/B/C/D market class. Traffic alone is not a correct flow; it must be supported by the applicable frozen run, extra-base-hit, starter, and bullpen observations.

After the market flow review, reproduce the calibration report schema-v2 watchlist in this form:

```markdown
## 반복 검증 Watchlist

> 확정 라인업 복기 관찰 전용이며 모델 결정 권한 없음

| 대상 | 상태 | 표본 | 제외 | 핵심 지표 | 트리거 |
|---|---|---:|---:|---|---|
```

Show the exact configured target label, one of `관찰`, `정상 범위`, or `정밀 검토`, the eligible/minimum sample, excluded count, primary metric, and trigger reasons. Only confirmed-lineup review rows contribute. Do not convert a watchlist status into a score, model change, priority change, recommendation, or Shadow decision.
