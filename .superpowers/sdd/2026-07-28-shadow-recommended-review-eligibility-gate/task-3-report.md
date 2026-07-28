# Task 3 Report: Deterministic July 28 Coverage Audit

## Status

Completed.

Committed in `be6e1cf4b23cf9bea985e0c47d80bba2a0d20919`
(`feat: add deterministic review coverage audit`).

## Implementation

- Added strict manifest and calibration loading with exact schema fields.
- Resolves exact capture, forecast, link, stored shadow comparison, market, and
  latest official-result relationships.
- Requires each manifest market to be stored as recommended and
  review-ineligible, and copies its frozen ineligibility flags unchanged.
- Settles frozen semantic markets through the existing settlement API.
- Produces canonical JSON and deterministic Korean Markdown through atomic
  output-file replacement.
- Preserves official non-count metrics, including Brier fields, in the
  audit-only view.
- Exposes the audit API lazily so module CLI execution is warning-free.

## TDD

The inherited focused suite initially passed:

```text
3 passed in 0.07s
```

Additional contract tests produced the expected RED failures for unauthorized
hit-rate adjustment, unknown calibration fields, and module CLI warnings:

```text
3 failed, 3 passed in 0.22s
```

The repaired focused suite passes:

```text
9 passed in 0.19s
```

## Verification

```text
.venv/bin/python -m pytest -q tests/test_review_coverage_audit.py tests/test_calibration.py tests/test_operator_reporting.py
116 passed, 86 subtests passed in 1.75s
```

The real CLI used:

```text
.venv/bin/python -m analysis.review.coverage_audit \
  --ledger data/ledger/shadow.jsonl \
  --calibration reports/private/shadow/generations/generation-v1-78635a8bfd3ccbce-00000000000000000539-ac553f3696d70b4b9a75300b4adb4aa2a2120577b5d2d2d9725db09f0853689d/calibration-latest.json \
  --manifest data/audits/2026-07-28-review-coverage-manifest.json \
  --output-dir reports/private/shadow/audits/2026-07-28-review-coverage
```

## Metrics

- Official: sample `59`, record `32-21-6`, priced `57`, profit
  `+1.993085u`, ROI `0.03496640350877193`.
- Audit-only: sample `64`, record `32-26-6`, priced `62`, profit
  `-3.0069150000000002u`, ROI `-0.048498629032258066`.
- Omissions: five resolved losses and one unresolved postponed decision.
- Official ledger revision: record count `539`, state hash
  `ac553f3696d70b4b9a75300b4adb4aa2a2120577b5d2d2d9725db09f0853689d`.
- Binary and multiclass Brier values are copied unchanged.

## Hash Invariance

Before and after the real CLI:

```text
12d8d045ffbacd4b726318854ada9c660aab206b68113758fa265e39239e8692  data/ledger/shadow.jsonl
926c8afc4f8d7a7395060afd9a14e9dcffa4ab64eae44d47d8695bdecb418a12  reports/private/shadow/generations/generation-v1-78635a8bfd3ccbce-00000000000000000539-ac553f3696d70b4b9a75300b4adb4aa2a2120577b5d2d2d9725db09f0853689d/calibration-latest.json
```

Generated runtime report hashes:

```text
35f4af7eaebdecbf54628b75ed7b8ef15b8cf1e19c8fba54fd4b3f88ee2a31b2  audit.json
e7a48b2c2cab5b4aa593872bd84b87824685df34875002cbcdcd7699307f7a90  audit.md
```

## Scope

The commit contains exactly:

- `analysis/review/coverage_audit.py`
- `analysis/review/__init__.py`
- `tests/test_review_coverage_audit.py`
- `data/audits/2026-07-28-review-coverage-manifest.json`

Runtime audit reports, official ledger/calibration data, and unrelated
untracked data files were not committed.

## Concerns

None.

## Review Follow-up

All five review findings were fixed in a follow-up commit.

- Ledger resolution is a single direct read-only stream. It retains only exact
  manifest capture, forecast, and link records plus the latest required
  official results; it does not construct a whole-ledger tuple or indexes for
  unrelated records.
- Manifest and calibration JSON reject duplicate keys and `NaN`/`Infinity`.
  Calibration loading validates the complete serialized schema, migration
  progress mappings, group shapes, and authoritative metric/group invariants.
- `audit_only_metrics` contains only `sample_size`, `wins`, `losses`,
  `pushes`, `priced_stake_count`, `profit_units`, and `roi`. Official metrics
  remain unchanged and complete.
- Missing or corrupt ledgers are opened with no-create read-only access, so
  they produce no ledger, parent, lock, checkpoint, or audit output artifacts.
- Expected module CLI validation/corruption failures print one deterministic
  `coverage-audit: ...` stderr line, create no output, and return exit code
  `2` without a traceback.

### Verification

```text
.venv/bin/python -m pytest -q tests/test_review_coverage_audit.py
23 passed in 0.40s

.venv/bin/python -m pytest -q tests/test_calibration.py tests/test_operator_reporting.py
107 passed, 86 subtests passed in 1.58s
```

The required real audit command completed successfully using the immutable
generation named above:

```text
/usr/bin/time -l ...
2.43 real, 2.39 user, 0.02 sys
62,357,504 maximum resident set size
```

Metrics remained exact:

- Official: sample `59`, record `32-21-6`, priced `57`, profit `+1.993085u`,
  ROI `0.03496640350877193`.
- Audit-only: sample `64`, record `32-26-6`, priced `62`, profit
  `-3.0069150000000002u`, ROI `-0.048498629032258066`.
- Omissions: five settled losses and one unresolved postponed decision.

Input hashes before and after the successful audit were unchanged:

```text
12d8d045ffbacd4b726318854ada9c660aab206b68113758fa265e39239e8692  data/ledger/shadow.jsonl
926c8afc4f8d7a7395060afd9a14e9dcffa4ab64eae44d47d8695bdecb418a12  reports/private/shadow/generations/generation-v1-78635a8bfd3ccbce-00000000000000000539-ac553f3696d70b4b9a75300b4adb4aa2a2120577b5d2d2d9725db09f0853689d/calibration-latest.json
```

Generated audit hashes:

```text
f38f659e88e85505fa0deb87e1859975d5d4f96b04d144c4c70bb54358814be4  audit.json
e7a48b2c2cab5b4aa593872bd84b87824685df34875002cbcdcd7699307f7a90  audit.md
```
