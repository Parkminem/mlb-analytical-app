# ML Bullpen Review Binding Design

## Goal

Prevent a fresh full-game ML recommendation from being frozen without an
auditable bullpen principal path when the fixed model uses the bullpen layer to
support that market. Preserve every historical capture and exact replay.

## Decision

Use a strict fresh-capture validation gate. A `play` or `lean` ML candidate in
the new legacy model version must declare and bind at least one of these
canonical evidence paths:

- `bullpen_protection`
- `late_bullpen_exposure`

The bound path continues to use the existing canonical component shapes and
contradiction rules. The validator never invents or injects a path.

## Version And Compatibility

- Advance the fixed legacy model version from `2026-07-13` to `2026-08-02`.
- A fresh projected or confirmed capture must use the current legacy model
  version.
- Exact historical replay remains the first operation. A byte-equivalent old
  capture can therefore replay without being rejected by the new contract.
- Old captures, forecasts, links, reviews, and calibration rows remain
  immutable and valid.

## Validation Flow

1. Resolve an exact historical replay before applying new validation.
2. For a fresh capture, reject a stale legacy model version.
3. Validate every recommended candidate and its review binding.
4. For ML only, require a bullpen path in both the candidate evidence set and
   its principal-path binding.
5. Continue the existing forecast, dynamic eligibility, and append-only ledger
   flow.

`pass` ML candidates and non-ML markets are unchanged.

## Non-Goals

- Do not alter ML scores, thresholds, prices, probabilities, or priority order.
- Do not add bullpen paths after first pitch or during review.
- Do not rewrite historical ledger data.
- Do not change F5, `+1.5`, or total-market binding requirements.

## Tests

- A new-version recommended ML without a bullpen path is rejected.
- Declaring the path only on the candidate but omitting it from the binding is
  rejected.
- `bullpen_protection` and `late_bullpen_exposure` each satisfy the contract
  with their existing canonical shapes.
- A fresh stale-version bundle is rejected before simulation and leaves no
  ledger artifact.
- An exact historical replay using the old version remains replayable.
- Existing shadow, operator, pipeline, review, and ledger suites remain green.
