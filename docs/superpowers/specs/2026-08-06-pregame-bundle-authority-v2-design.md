# Pregame Bundle Authority V2 Design

## Goal

Make current projected and confirmed pregame execution categorically distinct
from every schema-v1 historical artifact. A parsed, copied, serialized, empty,
or attribute-backed v1 object must never become a fresh current run. Existing v1
artifacts must remain available for exact immutable replay without rewriting
their bytes or ledger records.

This is an operator bundle-boundary change only. The fixed MLB model version,
baseball inputs, scoring, market engines, F5 engines, same-game priority policy,
Shadow forecast authority, and analysis output remain unchanged.

## Selected Architecture

### Current pregame bundles

New projected and confirmed bundles use:

```json
{
  "schema_version": 2,
  "authority": "current",
  "stage": "projected",
  "game_pk": 900001,
  "created_at": "2026-08-06T00:00:00Z",
  "payload": {}
}
```

The same envelope applies to `confirmed` with its matching stage.

- `schema_version` must be the JSON integer `2`.
- `authority` must be the exact string `current` and must be explicitly present.
- Every legacy decision candidate must explicitly contain a real boolean
  `same_game_pair_compatible`.
- Empty candidate tuples remain valid current decisions because authority is
  established at the bundle envelope rather than inferred from candidate
  contents.
- Direct BaseModel validation, TypeAdapter validation, JSON parsing, Python
  mapping parsing, and `from_attributes=True` all enforce the same v2 envelope.
- Normal projected and confirmed runners accept only the exact v2 current
  runtime types.

### Historical pregame bundles

Every projected or confirmed schema-v1 bundle is historical for execution
purposes, whether `same_game_pair_compatible` is absent or present.

- A v1 bundle can enter only through an authenticated managed artifact path.
- The dedicated historical replay service must resolve an exact existing
  capture, forecast, and link before returning anything.
- A v1 artifact against an empty or nonmatching ledger is rejected before
  simulation, append, supersession, or managed-bundle publication.
- A v1 bundle outside the configured managed root is rejected.
- No v1 bytes are upgraded, rewritten, or persisted as v2.
- Both historical shapes are preserved: relaxed v1 candidates that omit the
  compatibility field and strict v1 candidates that contain it. The loader
  retains which shape was authenticated so public serialization and provenance
  cannot silently change the original contract.

Separate runtime types represent relaxed-v1 and strict-v1 projected/confirmed
artifacts. They share replay behavior but cannot satisfy the v2 current runner
type guards.

### Review bundles

Review bundles remain schema version 1. They bind explicit frozen IDs and do not
create a new pregame capture or forecast, so they do not share the pregame
authority ambiguity. Their parsing, validation, and result-correction behavior
remain unchanged.

## Data Flow

### Fresh projected or confirmed execution

1. Decode JSON with duplicate-key and invalid-number protection.
2. Require the v2 pregame envelope and explicit `authority: current`.
3. Require all existing strict audit fields, including a real compatibility
   boolean on every candidate.
4. Construct the exact current projected or confirmed runtime type.
5. Run the existing pregame validation, Shadow forecast, ledger append, and
   immutable v2 managed-bundle persistence.

### Exact v1 replay

1. Authenticate the supplied managed path descriptor-relatively with no-follow
   checks and content hashing.
2. Classify the v1 artifact as relaxed or strict without converting it to v2.
3. Parse it into the matching replay-only runtime type.
4. Resolve the exact immutable capture, forecast, and link from the existing
   ledger.
5. Return the frozen records and original managed provenance, or reject without
   side effects.

### Rejected conversion attempts

The following all fail before reaching a normal runner:

- v1 `model_dump()` or `model_dump_json()` validated as a v2 current bundle;
- v1 nested payload dumps inserted into a candidate v2 envelope without the
  complete v2 envelope and explicit candidate booleans;
- v1 runtime objects passed through `from_attributes=True`;
- v1 empty-candidate bundles;
- caller-constructed replay-only runtime objects;
- v1 files outside the authenticated managed root.

## Compatibility And Rollout

- Existing managed v1 artifacts remain byte-for-byte unchanged.
- Existing v1 strict artifacts are reclassified as replay-only; they do not
  disappear and do not need migration.
- Any v1 bundle prepared but not yet executed must be regenerated as v2.
- Bundle producers, fixtures, CLI documentation, and output templates move to
  v2 atomically with the parser and runner changes.
- Managed filenames naturally use `v2` for new projected/confirmed artifacts.
- The MLB model version remains `2026-08-03`; bundle schema version and model
  version continue to be independent concepts.

## Error Handling

- A v1 input sent to a fresh projected/confirmed command reports that schema-v1
  pregame bundles are replay-only.
- Missing or invalid v2 authority reports a sanitized bundle-validation error.
- A managed v1 path without an exact ledger replay reports the existing
  historical exact-replay precondition failure.
- Every rejection leaves ledger bytes and the managed bundle tree unchanged.
- Error messages do not expose arbitrary dynamic input values or external path
  contents.

## Testing

### Current v2 contract

- Projected and confirmed v2 mappings parse through BaseModel, TypeAdapter, and
  operator parser paths.
- Empty, one-candidate, and multiple-candidate decisions are covered.
- Every candidate requires an explicit boolean.
- Missing, defaulted, wrong, or coercible authority/schema values are rejected.
- Normal runners reject every non-exact current runtime type.

### Historical v1 replay

- Relaxed and strict v1 projected/confirmed managed artifacts replay their exact
  existing capture, forecast, and link.
- The same artifacts reject against empty and nonmatching ledgers without side
  effects.
- Their original bytes, hashes, field-presence shape, and managed paths remain
  unchanged.
- Direct dumps, JSON dumps, nested payload dumps, empty-candidate dumps, copies,
  and `from_attributes=True` cannot construct a v2 current bundle.

### Regression

- Strict JSON datetime, tuple, boolean, duplicate-key, and extra-field behavior
  remains covered.
- Review schema-v1 behavior remains covered.
- CLI dispatch selects only v2 normal execution or authenticated v1 replay.
- Full tests verify unchanged forecasting, ledger, reporting, calibration,
  same-game multi-market selection, F5 coexistence, and model manifest output.

## Non-Goals

- Rewriting historical JSONL ledger records or managed bundle files.
- Changing MLB analysis weights, thresholds, confidence caps, or market scores.
- Allowing Shadow forecasts to change fixed-model recommendations.
- Migrating review bundles to schema version 2.
- Supporting fresh execution from any schema-v1 pregame bundle.

## Acceptance Criteria

1. Only an explicit schema-v2 current pregame envelope can reach normal
   projected or confirmed execution.
2. Every schema-v1 pregame artifact is replay-only and requires an authenticated
   managed path plus an exact existing ledger replay.
3. Relaxed and strict v1 artifacts retain exact immutable replay.
4. Empty-candidate and `from_attributes=True` laundering are impossible.
5. Rejections create no ledger or managed-bundle side effects.
6. Review bundles and all MLB analysis behavior remain unchanged.
7. The complete test suite, operator pipeline, and independent code review pass
   before integration.
