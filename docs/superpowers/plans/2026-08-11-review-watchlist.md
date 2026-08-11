# Review Watchlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic repeated-review watchlist counters to calibration JSON and Markdown without changing the fixed MLB model or pregame latency.

**Architecture:** Preserve compact review metadata on each selected `CalibrationRow`, evaluate three versioned watchlist definitions in a focused `analysis.review.watchlist` module, and pass immutable results into the existing atomic report publisher. The evaluator reads only already-selected confirmed review rows and never loads forecast draws or writes ledger records.

**Tech Stack:** Python 3.14, frozen dataclasses, JSON configuration, pytest, append-only JSONL ledger, atomic calibration report generations.

## Global Constraints

- Keep the fixed legacy model version at `2026-08-05`.
- Do not change scores, market directions, admission, priority assembly, or Shadow probabilities.
- Count only `lineup_state == "confirmed"` rows for the initial watchlist.
- Do not decode compressed 20,000-draw forecast arrays during watchlist evaluation.
- Keep the 20-decision watch threshold separate from the existing 300-decision market-family tuning boundary.
- A `정밀 검토` status never changes the model automatically.
- Preserve latest non-superseded review selection and historical replay behavior.

---

### Task 1: Preserve Compact Review Metadata In Calibration Rows

**Files:**
- Modify: `analysis/review/calibration.py:328-415, 1452-1528`
- Modify: `tests/test_calibration.py:37-68`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `ReviewRecord.classification`, `ReviewRecord.component_reviews`, `ReviewRecord.contradicted_paths`, and the selected capture market.
- Produces: required `CalibrationRow.selected_side: str`, `classification: str | None`, `principal_paths: tuple[str, ...]`, and `contradicted_paths: tuple[str, ...]` fields.

- [ ] **Step 1: Extend the calibration-row fixture and add failing validation tests**

Update `calibration_row()` defaults:

```python
"selected_side": "away",
"classification": "A",
"principal_paths": ("starter_suppression", "top_order_matchup"),
"contradicted_paths": (),
```

Add tests proving that the row is frozen and rejects an empty selected side,
unknown classification, duplicate principal paths, duplicate contradictions,
and contradictions that are not present in principal paths:

```python
def test_calibration_row_validates_compact_review_metadata(self):
    row = calibration_row()
    self.assertEqual(row.selected_side, "away")
    self.assertEqual(row.classification, "A")
    with self.assertRaisesRegex(ValueError, "selected_side"):
        calibration_row(selected_side="")
    with self.assertRaisesRegex(ValueError, "classification"):
        calibration_row(classification="E")
    with self.assertRaisesRegex(ValueError, "principal_paths"):
        calibration_row(principal_paths=("starter_suppression",) * 2)
    with self.assertRaisesRegex(ValueError, "contradicted_paths"):
        calibration_row(
            principal_paths=("starter_suppression",),
            contradicted_paths=("bullpen_protection",),
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_calibration.py -k compact_review_metadata -q`

Expected: FAIL because `CalibrationRow` does not accept the new fields.

- [ ] **Step 3: Add and validate the immutable fields**

Add these fields to `CalibrationRow` before probability fields:

```python
selected_side: str
classification: str | None
principal_paths: tuple[str, ...]
contradicted_paths: tuple[str, ...]
```

Validate with the existing non-empty helper and exact classifications:

```python
_REVIEW_CLASSIFICATIONS = frozenset(("A", "B", "C", "D"))

_require_nonempty_string(self.selected_side, "selected_side")
if self.classification is not None and self.classification not in _REVIEW_CLASSIFICATIONS:
    raise ValueError("classification must be A, B, C, D, or None")
for field_name in ("principal_paths", "contradicted_paths"):
    values = getattr(self, field_name)
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{field_name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
if not set(self.contradicted_paths).issubset(self.principal_paths):
    raise ValueError("contradicted_paths must be present in principal_paths")
```

- [ ] **Step 4: Add a failing row-selection preservation test**

In the existing fixture that appends a review, use component reviews containing
`bullpen_protection`, set `classification="C"`, and set
`contradicted_paths=("bullpen_protection",)`. Assert:

```python
row = select_primary_rows(self.store)[0]
self.assertEqual(row.selected_side, "away")
self.assertEqual(row.classification, "C")
self.assertEqual(
    row.principal_paths,
    ("starter_suppression", "bullpen_protection"),
)
self.assertEqual(row.contradicted_paths, ("bullpen_protection",))
```

- [ ] **Step 5: Run the preservation test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_calibration.py -k preserves_review_metadata -q`

Expected: FAIL because `_row_for_review()` does not populate the fields.

- [ ] **Step 6: Populate metadata without reading forecast distributions**

In `_row_for_review()`, derive paths directly from the compact review record:

```python
principal_paths = tuple(
    component_review["evidence_path"]
    for component_review in review.component_reviews
)
return CalibrationRow(
    # existing identity and market fields
    selected_side=capture_market["selected_side"],
    classification=review.classification,
    principal_paths=principal_paths,
    contradicted_paths=review.contradicted_paths,
    # existing probabilities and prices
)
```

Require `capture_market["selected_side"]` to be a non-empty string before row
construction. Do not inspect `component_distributions`.

- [ ] **Step 7: Run calibration tests**

Run: `.venv/bin/python -m pytest tests/test_calibration.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the row contract**

```bash
git add analysis/review/calibration.py tests/test_calibration.py
git commit -m "feat: preserve review metadata in calibration rows"
```

### Task 2: Add Strict Watchlist Configuration And Evaluation

**Files:**
- Create: `config/review-watchlist.json`
- Create: `analysis/review/watchlist.py`
- Create: `tests/test_review_watchlist.py`

**Interfaces:**
- Consumes: `tuple[CalibrationRow, ...]` and `ReviewWatchlistConfig`.
- Produces: `load_review_watchlist_config(path=DEFAULT_WATCHLIST_CONFIG_PATH) -> ReviewWatchlistConfig` and `build_review_watchlist(rows, config) -> tuple[WatchlistResult, ...]`.

- [ ] **Step 1: Add failing strict-configuration tests**

Create tests for the approved configuration, duplicate target IDs, unknown
target kinds, unknown keys, zero minimum sample, and rates outside `[0, 1]`:

```python
def test_loads_approved_watchlist_configuration(tmp_path: Path) -> None:
    path = write_config(tmp_path, APPROVED_CONFIG)
    config = load_review_watchlist_config(path)
    assert config.schema_version == 1
    assert tuple(target.target_id for target in config.targets) == (
        "f5_over_early_conversion",
        "bullpen_protection_contradiction",
        "lineup_suppression_contradiction",
    )

@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value["targets"].append(value["targets"][0]), "duplicate"),
        (lambda value: value["targets"][0].update(kind="unknown"), "kind"),
        (lambda value: value.update(extra=True), "unknown"),
        (lambda value: value["targets"][0].update(minimum_sample=0), "minimum"),
        (lambda value: value["targets"][0].update(hit_rate_below=1.1), "rate"),
    ),
)
def test_rejects_invalid_configuration(tmp_path, mutator, message):
    value = deepcopy(APPROVED_CONFIG)
    mutator(value)
    with pytest.raises(ValueError, match=message):
        load_review_watchlist_config(write_config(tmp_path, value))
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_review_watchlist.py -k configuration -q`

Expected: FAIL because the module and loader do not exist.

- [ ] **Step 3: Implement frozen configuration types and strict loader**

Define:

```python
DEFAULT_WATCHLIST_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "review-watchlist.json"
)

@dataclass(frozen=True)
class WatchlistTargetConfig:
    target_id: str
    label: str
    kind: str
    minimum_sample: int
    hit_rate_below: float | None = None
    b_loss_count_min: int | None = None
    b_loss_rate_min: float | None = None
    evidence_path: str | None = None
    contradiction_count_min: int | None = None
    contradiction_rate_min: float | None = None

@dataclass(frozen=True)
class ReviewWatchlistConfig:
    schema_version: int
    targets: tuple[WatchlistTargetConfig, ...]
```

Support only `f5_over_conversion` and `path_contradiction`. Validate exact key
sets per kind so misspelled or inapplicable thresholds fail closed.

- [ ] **Step 4: Add the approved configuration file**

```json
{
  "schema_version": 1,
  "targets": [
    {
      "target_id": "f5_over_early_conversion",
      "label": "F5 오버 초기 득점 전환",
      "kind": "f5_over_conversion",
      "minimum_sample": 20,
      "hit_rate_below": 0.45,
      "b_loss_count_min": 5,
      "b_loss_rate_min": 0.25
    },
    {
      "target_id": "bullpen_protection_contradiction",
      "label": "불펜 보호 경로 모순",
      "kind": "path_contradiction",
      "minimum_sample": 20,
      "evidence_path": "bullpen_protection",
      "contradiction_count_min": 3,
      "contradiction_rate_min": 0.15
    },
    {
      "target_id": "lineup_suppression_contradiction",
      "label": "언더 라인업 억제 경로 모순",
      "kind": "path_contradiction",
      "minimum_sample": 20,
      "evidence_path": "lineup_suppression",
      "contradiction_count_min": 3,
      "contradiction_rate_min": 0.15
    }
  ]
}
```

- [ ] **Step 5: Add failing F5-over evaluator tests**

Build rows with `lineup_state="confirmed"`, `market_type="f5_total"`, and
`selected_side="over"`. Verify:

```python
def test_f5_over_stays_observe_below_twenty() -> None:
    result = evaluate(19_losses())[0]
    assert result.status == "관찰"
    assert result.eligible_count == 19

def test_f5_over_triggers_on_low_hit_rate_at_twenty() -> None:
    result = evaluate(rows(wins=8, losses=12))[0]
    assert result.status == "정밀 검토"
    assert "non_push_hit_rate_below" in result.trigger_reasons

def test_f5_over_triggers_on_repeated_b_losses() -> None:
    result = evaluate(rows(wins=15, b_losses=5))[0]
    assert result.status == "정밀 검토"
    assert result.b_loss_count == 5

def test_f5_over_normal_when_no_trigger_fires() -> None:
    result = evaluate(rows(wins=12, losses=8, b_losses=4))[0]
    assert result.status == "정상 범위"
```

Also assert projected rows, F5 unders, and unclassified rows do not enter the
eligible count; matching unclassified rows increment `excluded_count`.

- [ ] **Step 6: Add failing path-contradiction evaluator tests**

Verify that 19 rows stay `관찰`, 20 rows with three exact contradictions at
15% enter `정밀 검토`, and either only two contradictions or a sub-15% rate
stays `정상 범위`. Assert unrelated `C` paths do not count.

```python
result = evaluate_path(
    evidence_path="bullpen_protection",
    eligible=20,
    contradicted=3,
)
assert result.status == "정밀 검토"
assert result.contradiction_rate == 0.15
```

For `lineup_suppression`, include only `f5_total` or `full_total` rows whose
`selected_side == "under"`.

- [ ] **Step 7: Run evaluator tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_review_watchlist.py -k 'f5_over or contradiction' -q`

Expected: FAIL because the evaluator does not exist.

- [ ] **Step 8: Implement deterministic watchlist results and selectors**

Define a frozen result with a complete mapping contract:

```python
@dataclass(frozen=True)
class WatchlistResult:
    target_id: str
    label: str
    status: str
    eligible_count: int
    excluded_count: int
    win_count: int
    loss_count: int
    push_count: int
    classification_counts: Mapping[str, int]
    b_loss_count: int | None
    contradiction_count: int | None
    contradiction_rate: float | None
    non_push_hit_rate: float | None
    minimum_sample: int
    trigger_reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "label": self.label,
            "status": self.status,
            "eligible_count": self.eligible_count,
            "excluded_count": self.excluded_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "push_count": self.push_count,
            "classification_counts": dict(self.classification_counts),
            "b_loss_count": self.b_loss_count,
            "contradiction_count": self.contradiction_count,
            "contradiction_rate": self.contradiction_rate,
            "non_push_hit_rate": self.non_push_hit_rate,
            "minimum_sample": self.minimum_sample,
            "trigger_reasons": list(self.trigger_reasons),
        }
```

Sort results by configuration order and sort trigger reasons deterministically.
Use exact comparisons from the approved design. Do not inspect probability or
forecast distribution fields.

- [ ] **Step 9: Run watchlist tests**

Run: `.venv/bin/python -m pytest tests/test_review_watchlist.py -q`

Expected: PASS.

- [ ] **Step 10: Commit the evaluator**

```bash
git add config/review-watchlist.json analysis/review/watchlist.py tests/test_review_watchlist.py
git commit -m "feat: add repeated review watchlist"
```

### Task 3: Publish Watchlist In Calibration JSON And Markdown

**Files:**
- Modify: `analysis/operator/reporting.py:30-60, 380-520`
- Modify: `analysis/operator/service.py:1820-1875`
- Modify: `tests/test_operator_reporting.py`

**Interfaces:**
- Consumes: `tuple[WatchlistResult, ...]` built from the same selected rows as the calibration report.
- Produces: report schema version `2`, top-level JSON `watchlist`, and Markdown section `반복 검증 Watchlist`.

- [ ] **Step 1: Add failing report-mapping and Markdown tests**

Update the report helper to load the approved config and call
`build_review_watchlist(rows, config)`. Assert:

```python
assert mapping["schema_version"] == REPORT_SCHEMA_VERSION == 2
assert [row["target_id"] for row in mapping["watchlist"]] == [
    "f5_over_early_conversion",
    "bullpen_protection_contradiction",
    "lineup_suppression_contradiction",
]
markdown = render_report_markdown(mapping)
assert "## 반복 검증 Watchlist" in markdown
assert "모델 결정 권한 없음" in markdown
assert "F5 오버 초기 득점 전환" in markdown
```

Add a deterministic-byte test proving identical rows and config produce the
same JSON and Markdown generation.

- [ ] **Step 2: Run report tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_operator_reporting.py -k watchlist -q`

Expected: FAIL because report schema v1 has no watchlist.

- [ ] **Step 3: Add the required watchlist argument and schema v2 mapping**

Change the pure mapping signature:

```python
def build_report_mapping(
    *,
    ledger_path: str | os.PathLike[str],
    ledger_records: tuple[object, ...],
    rows: tuple[CalibrationRow, ...],
    report: CalibrationReport,
    watchlist: tuple[WatchlistResult, ...],
) -> dict[str, object]:
```

Validate the tuple and add:

```python
"watchlist": [result.to_mapping() for result in watchlist],
```

Set `REPORT_SCHEMA_VERSION = 2`. This is only a report schema change; do not
change `config/model.json` version or `analysis.pipeline` model version.

- [ ] **Step 4: Render the Korean watchlist section**

Before migration progress, render:

```markdown
## 반복 검증 Watchlist

> 관찰 전용이며 모델 결정 권한 없음

| 대상 | 상태 | 표본 | 제외 | 핵심 지표 | 트리거 |
```

For F5 over, show hit rate and B-loss count. For path targets, show exact-path
contradiction count and rate. Render absent metrics as `사용 불가` and trigger
reasons as `없음`.

- [ ] **Step 5: Add a failing service-integration test**

Monkeypatch `load_review_watchlist_config` and `build_review_watchlist` in
`analysis.operator.service`, call `_calibration_report_state()`, and assert the
same selected `rows` tuple reaches both calibration and watchlist builders.

Add a failure test where config loading raises `ValueError`; verify no report
publisher is called and the error is mapped before publication.

- [ ] **Step 6: Run the service test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_operator_reporting.py -k 'service and watchlist' -q`

Expected: FAIL because the service does not build or pass watchlist results.

- [ ] **Step 7: Integrate config loading before atomic publication**

In `_calibration_report_state()`:

```python
rows = select_primary_rows(_LedgerSnapshot(ledger_records))
report = build_calibration_report(rows)
watchlist_config = load_review_watchlist_config()
watchlist = build_review_watchlist(rows, watchlist_config)
mapping = build_report_mapping(
    ledger_path=ledger_path,
    ledger_records=ledger_records,
    rows=rows,
    report=report,
    watchlist=watchlist,
)
```

Because this runs before `write_calibration_reports()`, invalid configuration
cannot publish a partial generation. Keep the existing stale-publication retry
and filesystem atomicity unchanged.

- [ ] **Step 8: Run reporting and operator tests**

Run: `.venv/bin/python -m pytest tests/test_operator_reporting.py tests/test_operator_review.py -q`

Expected: PASS.

- [ ] **Step 9: Commit reporting integration**

```bash
git add analysis/operator/reporting.py analysis/operator/service.py tests/test_operator_reporting.py
git commit -m "feat: publish review watchlist reports"
```

### Task 4: Document, Regenerate, And Verify The Current Watchlist

**Files:**
- Modify: `docs/model.md:600-640`
- Modify: `docs/analysis-output-template.md:140-170`
- Modify: `docs/shadow-operator.md:100-160`
- Modify: `docs/superpowers/specs/2026-08-11-review-watchlist-design.md`
- Verify generated ignored artifacts: `reports/private/shadow/calibration-latest.json`, `reports/private/shadow/calibration-latest.md`

**Interfaces:**
- Consumes: the implemented schema-v2 calibration report.
- Produces: authoritative operating guidance and a regenerated local report showing all three approved targets.

- [ ] **Step 1: Document the non-authoritative watchlist once per operating concern**

Add to `docs/model.md` under objective review:

```markdown
Calibration also reports a confirmed-lineup repeated-verification watchlist.
`관찰`, `정상 범위`, and `정밀 검토` are diagnostics only. They cannot change
legacy admission, scores, priority, Shadow output, or the fixed model without a
separate approved semantic model revision.
```

Add the expected postgame display to the output template and describe report
schema v2 plus `config/review-watchlist.json` in the operator guide. State that
watchlist evaluation does not decode forecast draws.

- [ ] **Step 2: Run documentation and manifest checks**

Run: `.venv/bin/python -m analysis.pipeline`

Expected: model version remains `2026-08-05`, fixed model remains true, and the
legacy policy mapping is unchanged.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 3: Run the complete test suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS with zero failures.

- [ ] **Step 4: Replay one exact review to regenerate the current report**

Run:

```bash
.venv/bin/python -m analysis.operator review \
  --bundle data/shadow/bundles/825048/review-v1-668e68889be7bfca.json
```

Expected: exact result/review replay, calibration report publication succeeds,
and no new duplicate review identity is created.

- [ ] **Step 5: Verify the current baseline counters**

Read `reports/private/shadow/calibration-latest.json` and assert:

```python
expected = {
    "f5_over_early_conversion": (6, "관찰"),
    "bullpen_protection_contradiction": (7, "관찰"),
    "lineup_suppression_contradiction": (2, "관찰"),
}
```

Also verify F5 over contains one win, five losses, and five B losses; bullpen
protection contains one exact contradiction; lineup suppression contains one
exact contradiction. The values are observational and must not alter the
priority model.

- [ ] **Step 6: Inspect scope and commit documentation**

Run: `git status --short`

Expected: only approved code, config, tests, and documentation are tracked;
ledger and private generated reports remain ignored operational artifacts.

```bash
git add docs/model.md docs/analysis-output-template.md docs/shadow-operator.md docs/superpowers/specs/2026-08-11-review-watchlist-design.md
git commit -m "docs: document repeated review watchlist"
```

- [ ] **Step 7: Final verification**

Run: `.venv/bin/python -m pytest -q`

Run: `git diff --check`

Run: `git status --short --branch`

Expected: all tests pass, no diff errors remain, and the branch is clean apart
from intentional commits not yet pushed.
