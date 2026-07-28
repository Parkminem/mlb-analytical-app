# Shadow Recommended Review Eligibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject every new projected or confirmed pregame state whose `play` or `lean` candidate lacks a valid frozen review contract, while preserving historical exact replay and recording the July 28 omissions in a separate audit.

**Architecture:** Add pure static and dynamic eligibility gates in `analysis.shadow`, invoke the static gate after exact-replay lookup but before simulation, and invoke the dynamic gate before any shadow-run append. Keep ineligible `pass` comparisons and historical records readable. Build the July 28 audit from an explicit immutable manifest so its adjusted metrics remain diagnostic and never become calibration rows.

**Tech Stack:** Python 3.14, frozen dataclasses, pytest, append-only JSONL ledger, deterministic canonical JSON and Markdown reports.

## Global Constraints

- Do not change baseball inputs, model weights, simulation output, market probability, legacy admission, legacy score, or final-priority ordering.
- Apply the new-record gate to both `projected` and `confirmed`.
- Require every `play` or `lean` candidate to be review-eligible; exclude `pass` candidates from the hard gate.
- Preserve exact replay of historical immutable commands.
- Perform static validation before the 20,000-draw simulation and dynamic validation before ledger append.
- Do not retroactively alter a capture, forecast, link, review, calibration row, or report generation.
- The July 28 adjusted metrics are audit-only and must not replace official calibration.

---

### Task 1: Pure Recommended Review Contract Gates

**Files:**
- Modify: `analysis/shadow.py`
- Modify: `tests/test_shadow.py`

**Interfaces:**
- Produces: `validate_recommended_review_contract(legacy: LegacyDecisionSnapshot, markets: tuple[MarketInput, ...]) -> None`
- Produces: `require_recommended_review_eligibility(comparison: ShadowComparison) -> None`
- Consumes: existing `validate_principal_path_bindings`, `MarketType`, `CandidateReviewBinding`, and `ShadowComparison`

- [ ] **Step 1: Add failing static-gate tests**

Add focused tests to `tests/test_shadow.py` that construct the existing
`LegacyDecisionSnapshot` fixtures and assert:

```python
def test_recommended_contract_rejects_missing_lean_binding() -> None:
    legacy = legacy_snapshot(
        candidates=(candidate(decision_rating="lean"),),
        review_bindings=(),
    )

    with pytest.raises(
        ValueError,
        match=r"candidate\\[0\\] ml \\(lean\\): missing_pregame_binding",
    ):
        validate_recommended_review_contract(
            legacy,
            complete_forecast_game().markets,
        )


def test_recommended_contract_allows_unbound_pass() -> None:
    legacy = legacy_snapshot(
        candidates=(candidate(decision_rating="pass"),),
        final_priority=(),
        review_bindings=(),
    )

    validate_recommended_review_contract(
        legacy,
        complete_forecast_game().markets,
    )
```

Add separate failures for:

- `starter_collapse` with only `*_pitcher_chain_runs_allowed`;
- `starter_suppression` with both away and home run components;
- a binding market key absent from `markets`;
- a binding whose `legacy_market` differs from its candidate;
- final priority referencing no candidate;
- final priority referencing a `pass` candidate.

- [ ] **Step 2: Run the static-gate tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_shadow.py -k "recommended_contract"
```

Expected: collection or assertion failures because
`validate_recommended_review_contract` does not exist.

- [ ] **Step 3: Implement the static gate**

In `analysis/shadow.py`, import `MarketInput` and add:

```python
_RECOMMENDED_DECISION_RATINGS = frozenset(("play", "lean"))


def validate_recommended_review_contract(
    legacy: LegacyDecisionSnapshot,
    markets: tuple[MarketInput, ...],
) -> None:
    if not isinstance(legacy, LegacyDecisionSnapshot):
        raise TypeError("legacy must be LegacyDecisionSnapshot")
    if not isinstance(markets, tuple) or any(
        not isinstance(market, MarketInput) for market in markets
    ):
        raise TypeError("markets must be a tuple of MarketInput values")

    markets_by_key = {market.market_key: market for market in markets}
    bindings_by_index = {
        binding.legacy_candidate_index: binding
        for binding in legacy.review_bindings
    }
    errors: list[str] = []
    recommended_markets: dict[str, int] = {}

    for index, candidate in enumerate(legacy.candidates):
        if candidate.decision_rating not in _RECOMMENDED_DECISION_RATINGS:
            continue
        if candidate.market in recommended_markets:
            errors.append(
                f"candidate[{index}] {candidate.market} "
                f"({candidate.decision_rating}): duplicate_recommended_market"
            )
            continue
        recommended_markets[candidate.market] = index
        binding = bindings_by_index.get(index)
        if binding is None:
            errors.append(
                f"candidate[{index}] {candidate.market} "
                f"({candidate.decision_rating}): missing_pregame_binding"
            )
            continue
        prefix = (
            f"candidate[{index}] {candidate.market} "
            f"({candidate.decision_rating})"
        )
        if binding.legacy_market != candidate.market:
            errors.append(f"{prefix}: candidate_market_mismatch")
            continue
        market = markets_by_key.get(binding.market_key)
        if market is None:
            errors.append(f"{prefix}: unknown_market_key {binding.market_key}")
            continue
        if market.market_type.value != candidate.market:
            errors.append(
                f"{prefix}: market_type_mismatch {market.market_type.value}"
            )
            continue
        try:
            validate_principal_path_bindings(
                market.market_type,
                binding.principal_paths,
            )
        except ValueError as error:
            errors.append(f"{prefix}: {error}")

    for market in legacy.final_priority:
        index = recommended_markets.get(market)
        if index is None:
            errors.append(
                f"final_priority {market}: requires one play or lean candidate"
            )

    if errors:
        raise ValueError(
            "recommended review contract is invalid: " + "; ".join(errors)
        )
```

Keep deterministic candidate order in the error message. Do not validate
unbound `pass` candidates.

- [ ] **Step 4: Add failing dynamic-gate tests**

Add tests that call `compare_shadow_forecast` with:

- one valid `play` comparison and one ineligible `pass`;
- one `lean` comparison carrying `invalid_path_count`;
- one `play` comparison carrying `missing_component_distribution`.

The valid case must return normally. The recommended ineligible cases must
raise a deterministic message containing candidate index, market, rating,
market key, and stored flags.

- [ ] **Step 5: Run the dynamic-gate tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_shadow.py -k "recommended_review_eligibility"
```

Expected: failure because `require_recommended_review_eligibility` does not
exist.

- [ ] **Step 6: Implement the dynamic gate and protect lower-level writes**

Add:

```python
def require_recommended_review_eligibility(
    comparison: ShadowComparison,
) -> None:
    if not isinstance(comparison, ShadowComparison):
        raise TypeError("comparison must be ShadowComparison")
    errors = []
    for candidate, market in zip(
        comparison.legacy.candidates,
        comparison.market_comparisons,
        strict=True,
    ):
        if candidate.decision_rating not in _RECOMMENDED_DECISION_RATINGS:
            continue
        if market.review_eligible:
            continue
        flags = ",".join(market.review_ineligibility_flags)
        errors.append(
            f"candidate[{market.legacy_candidate_index}] "
            f"{candidate.market} ({candidate.decision_rating}) "
            f"{market.market_key}: {flags}"
        )
    if errors:
        raise ValueError(
            "recommended review comparison is ineligible: "
            + "; ".join(errors)
        )
```

Call it in `record_shadow_run` immediately after
`comparison = compare_shadow_forecast(legacy, forecast)` and before any
capture or forecast command is appended. Export both new functions from
`analysis.shadow.__all__`.

- [ ] **Step 7: Run shadow tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_shadow.py
```

Expected: all tests pass.

Commit:

```bash
git add analysis/shadow.py tests/test_shadow.py
git commit -m "feat: gate recommended shadow review contracts"
```

---

### Task 2: Operator New-State Integration and Replay Compatibility

**Files:**
- Modify: `analysis/operator/service.py`
- Modify: `tests/test_operator_pregame.py`
- Modify: `docs/model.md`
- Modify: `docs/shadow-operator.md`

**Interfaces:**
- Consumes: `validate_recommended_review_contract` and `require_recommended_review_eligibility` from Task 1
- Produces: projected and confirmed operator behavior that rejects only new invalid recommended states

- [ ] **Step 1: Add a failing pre-simulation rejection test**

In `tests/test_operator_pregame.py`, create a projected bundle whose `lean`
candidate has a `starter_collapse` binding with only
`home_pitcher_chain_runs_allowed`. Monkeypatch
`analysis.operator.service.run_forecast` to raise `AssertionError` if called.

Assert:

```python
with pytest.raises(
    OperatorPreconditionError,
    match="recommended review contract is invalid",
):
    run_projected(bundle, ledger_path=ledger, bundle_root=bundles)

assert not ledger.exists()
assert not bundles.exists()
```

The expected failure must be the contract error, proving forecast simulation
was not entered.

- [ ] **Step 2: Run the new-state test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_pregame.py \
  -k "invalid_recommended_contract_before_forecast"
```

Expected: the forecast assertion fires or the invalid state is recorded.

- [ ] **Step 3: Integrate static and dynamic checks after replay lookup**

Import the Task 1 functions in `analysis/operator/service.py`.

In `_run_pregame`, preserve the current exact-replay branch first. Immediately
after `if exact_replay is not None: ... return`, validate the new command:

```python
try:
    validate_recommended_review_contract(
        bundle.payload.legacy_decision,
        bundle.payload.game_input.markets,
    )
except ValueError as error:
    raise OperatorPreconditionError(str(error)) from error
```

After `forecast = run_forecast(...)` and before the head-CAS append loop, build
the deterministic comparison and enforce the dynamic gate:

```python
comparison = compare_shadow_forecast(
    bundle.payload.legacy_decision,
    forecast,
)
try:
    require_recommended_review_eligibility(comparison)
except ValueError as error:
    raise OperatorPreconditionError(str(error)) from error
```

`record_shadow_run` repeats the dynamic gate as a lower-level safety boundary;
the operator precheck exists to return the expected precondition error before
any append.

- [ ] **Step 4: Add replay, pass, and dynamic-failure tests**

Add tests proving:

1. A valid projected and valid confirmed command still records and replays.
2. An unbound `pass` candidate remains accepted.
3. A dynamic missing-component failure leaves ledger and bundle state
   unchanged.
4. A historical exact replay remains available.

For the historical replay fixture, record the invalid command with both new
gate functions monkeypatched to no-op, restore the real functions, then replay
the exact same bundle. Assert the original capture, forecast, and link IDs are
returned with all replay flags true and no new ledger record is added.

- [ ] **Step 5: Run operator RED/GREEN verification**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_operator_pregame.py \
  tests/test_operator_cli.py
```

Expected: all tests pass.

- [ ] **Step 6: Document the operational gate**

Update `docs/model.md` and `docs/shadow-operator.md` with these exact rules:

- every new projected or confirmed `play`/`lean` candidate must be
  review-eligible;
- `pass` candidates are excluded from the hard gate;
- static validation runs before simulation and dynamic validation before
  append;
- exact replay of historical immutable commands remains available;
- a rejection writes no pregame ledger or managed-bundle artifact.

- [ ] **Step 7: Run focused suites and commit**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_shadow.py \
  tests/test_operator_pregame.py \
  tests/test_operator_cli.py \
  tests/test_operator_review.py \
  tests/test_review_service.py
```

Expected: all tests pass.

Commit:

```bash
git add \
  analysis/operator/service.py \
  tests/test_operator_pregame.py \
  docs/model.md \
  docs/shadow-operator.md
git commit -m "feat: enforce review eligibility before pregame freeze"
```

---

### Task 3: Deterministic July 28 Coverage Audit

**Files:**
- Create: `analysis/review/coverage_audit.py`
- Modify: `analysis/review/__init__.py`
- Create: `tests/test_review_coverage_audit.py`
- Create: `data/audits/2026-07-28-review-coverage-manifest.json`
- Create at runtime: `reports/private/shadow/audits/2026-07-28-review-coverage/audit.json`
- Create at runtime: `reports/private/shadow/audits/2026-07-28-review-coverage/audit.md`

**Interfaces:**
- Produces: `build_coverage_audit(ledger_path: Path, calibration_path: Path, manifest_path: Path) -> Mapping[str, object]`
- Produces: `render_coverage_audit_markdown(audit: Mapping[str, object]) -> str`
- Produces: module CLI `python -m analysis.review.coverage_audit`

- [ ] **Step 1: Add failing audit tests**

Create a small temporary ledger using existing ledger test fixtures with:

- one eligible reviewed win;
- one ineligible recommended loss;
- one ineligible postponed decision with no final result.

Write a manifest that names exact capture, forecast, link, and market keys.
Assert the audit:

```python
assert audit["official_calibration_unchanged"] is True
assert audit["resolved_omission_count"] == 1
assert audit["unresolved_omission_count"] == 1
assert audit["audit_only_metrics"]["losses"] == (
    audit["official_metrics"]["losses"] + 1
)
assert audit["entries"][0]["ineligibility_flags"] == [
    "invalid_path_count"
]
```

Add a deterministic rendering test that calls the builder twice and compares
canonical JSON and Markdown bytes.

- [ ] **Step 2: Run audit tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_review_coverage_audit.py
```

Expected: import failure because the audit module does not exist.

- [ ] **Step 3: Implement strict audit input and ledger resolution**

Implement strict JSON loading with exact fields:

```json
{
  "schema_version": 1,
  "audit_id": "2026-07-28-review-coverage",
  "entries": [
    {
      "game_pk": 822705,
      "capture_id": "37542c98c77540b7a5e197e23f47dc1c",
      "forecast_run_id": "517cb5c1a300b14900afd5d48dcc018ed22ab51850fb23e5733ae542fd208d56",
      "capture_forecast_link_record_id": "a11b949640c643fea5dbee4346c561a7",
      "market_key": "game:822705:full_total:over"
    }
  ]
}
```

For each entry:

1. Resolve the exact capture, forecast, and link.
2. Deserialize the link-bound shadow comparison.
3. Require the named market comparison to be recommended and ineligible.
4. Copy its stored flags without repairing its paths.
5. Resolve the latest official result when present.
6. Settle the frozen semantic market with existing settlement code.
7. Mark no-result entries unresolved.

Read official metrics from the supplied calibration JSON. Add settled omitted
outcomes only to the audit-only count, profit, and ROI fields. Do not write to
the ledger or calibration directory.

- [ ] **Step 4: Implement deterministic CLI and Markdown**

Support:

```bash
.venv/bin/python -m analysis.review.coverage_audit \
  --ledger data/ledger/shadow.jsonl \
  --calibration reports/private/shadow/current/calibration-latest.json \
  --manifest data/audits/2026-07-28-review-coverage-manifest.json \
  --output-dir reports/private/shadow/audits/2026-07-28-review-coverage
```

Write canonical JSON plus a Korean Markdown report that prominently includes:

```text
이 조정치는 감사용이며 공식 calibration row가 아닙니다.
```

Use atomic temporary-file replacement inside the requested output directory.

- [ ] **Step 5: Add the exact July 28 manifest and generate the audit**

The manifest must include:

- TOR F5 side, full total, and F5 total;
- ARI F5 total;
- PHI F5 total;
- postponed CLE F5 total.

Run the CLI and verify:

- five resolved omitted losses;
- one unresolved postponed decision;
- official sample remains `59`;
- audit-only sample is `64`;
- official record count and state hash are copied unchanged;
- official calibration files have unchanged hashes before and after.

- [ ] **Step 6: Run audit and regression tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_review_coverage_audit.py \
  tests/test_calibration.py \
  tests/test_operator_reporting.py
```

Expected: all tests pass.

Commit:

```bash
git add \
  analysis/review/coverage_audit.py \
  analysis/review/__init__.py \
  tests/test_review_coverage_audit.py \
  data/audits/2026-07-28-review-coverage-manifest.json
git commit -m "feat: add deterministic review coverage audit"
```

---

### Task 4: Full Verification and Performance Guard

**Files:**
- Modify if required by measured regression: `tests/test_operator_pregame.py`

**Interfaces:**
- Consumes: all prior task outputs
- Produces: verified branch with no model or calibration mutation

- [ ] **Step 1: Run full relevant test suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_shadow.py \
  tests/test_operator_schemas.py \
  tests/test_operator_pregame.py \
  tests/test_operator_cli.py \
  tests/test_operator_review.py \
  tests/test_review_service.py \
  tests/test_review_classification.py \
  tests/test_calibration.py \
  tests/test_operator_reporting.py
```

Expected: all tests and subtests pass.

- [ ] **Step 2: Measure valid preflight overhead**

Use a valid in-memory projected bundle and run the static gate 10,000 times
with `time.perf_counter`. Record median per-call time. Require the median to be
below `1 ms` on the current workspace machine.

Also assert with a monkeypatched counter that one successful new pregame
command still calls `run_forecast` exactly once and one exact replay calls it
zero times.

- [ ] **Step 3: Verify immutable operational state**

Record hashes before and after the audit generation:

```bash
shasum -a 256 \
  data/ledger/shadow.jsonl \
  reports/private/shadow/current/calibration-latest.json \
  reports/private/shadow/current/calibration-latest.md
```

Expected: all three hashes remain unchanged by audit generation.

- [ ] **Step 4: Inspect diff and commit any test-only performance guard**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -6
```

If the measured guard required a test change, commit only that test:

```bash
git add tests/test_operator_pregame.py
git commit -m "test: guard review eligibility preflight performance"
```

Otherwise do not create an empty commit.
