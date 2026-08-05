# Historical Pregame Replay Authority Design

## Goal

Preserve exact replay of pre-change schema-v1 projected and confirmed bundles
that omit `same_game_pair_compatible` without allowing those historical bytes,
or caller-constructed provenance metadata, to create a fresh capture, forecast,
link, or strict-looking managed bundle.

This is an authority-boundary correction for the approved multi-market priority
feature. It does not change baseball scoring, the score floor, market engines,
F5 evaluation, model version, schema version, or Shadow authority.

## Considered Approaches

### Dedicated historical replay API (selected)

Keep normal pregame execution strict and introduce stage-specific historical
replay entry points. The replay service accepts a managed artifact path, loads
and authenticates the artifact itself, and returns only an exact existing ledger
replay. It cannot run forecasting or append records.

This makes the authority explicit and prevents a caller from manufacturing a
trusted wrapper or substituting a different managed path, hash, or replay flag.

### Re-authenticate a public replay wrapper in the normal runner

The service could reload every wrapper's embedded path before use. This is a
smaller change, but it retains a mixed API where one runner both creates fresh
forecasts and accepts historical compatibility objects. Repeated review found
that this boundary is easy to lose during unwrapping or persistence, so this
approach is rejected.

### Introduce schema v2 and stop schema-v1 replay

A new schema could make the field mandatory everywhere, but it would discard
the required exact replay guarantee for existing captures and is unnecessarily
broad for this correction. It is rejected.

## API Boundaries

- `run_projected` accepts only a strict `ProjectedBundle`.
- `run_confirmed` accepts only a strict `ConfirmedBundle`.
- Both normal runners reject any historical compatibility artifact at runtime,
  even if a caller constructs one directly.
- A projected historical replay entry point accepts the artifact path, ledger
  path, managed root, and optional registry root.
- A confirmed historical replay entry point has the same contract.
- The historical entry points load the artifact from the supplied managed root
  inside the service boundary. They do not trust caller-provided parsed bundles,
  hashes, replay flags, or managed-copy objects.

The historical loader continues to verify the resolved non-symlink path,
managed-root location, game directory, stage, schema version, filename hash,
full content hash, and bundle identity. The service additionally requires the
loaded artifact to be the pre-change relaxed form; a strict current bundle is
not accepted by the historical replay API.

## Data Flow

### Strict current bundle

1. The CLI strictly parses the projected or confirmed bundle.
2. The CLI calls the corresponding normal runner.
3. The normal runner may replay an exact current record or create a fresh
   capture, forecast, and link under the existing rules.
4. Managed-bundle persistence remains unchanged.

### Pre-change managed artifact

1. Strict parsing fails only because the pre-change candidate field is absent.
2. The CLI dispatches the original managed path to the stage-specific
   historical replay API instead of passing a parsed compatibility object to a
   normal runner.
3. The replay service reloads and authenticates the path under the configured
   managed root, then parses the isolated historical form.
4. The service derives the capture key and forecast run ID and searches for the
   exact existing capture, forecast, and link.
5. If all frozen identities and outputs match, the service returns those records
   and the original authenticated managed path and hash.
6. If no exact replay exists, the service rejects before forecast execution,
   ledger creation, append, or managed-bundle persistence.

No historically defaulted bundle is serialized into a new strict artifact.

## CLI Behavior

The projected and confirmed commands retain their existing user-facing shape.
Their loader result distinguishes a strict bundle from an authenticated
historical replay request. The strict branch calls the normal runner; the
historical branch calls the dedicated replay service with the original path.

Review-stage loading remains strict because this migration applies only to
pregame legacy candidates.

## Error Handling

- A missing-field artifact outside the managed root is rejected by strict
  parsing.
- A hash-named missing-field artifact under an empty or non-matching ledger is
  rejected as `historical replay requires an exact existing replay`.
- A strict current artifact sent to the historical replay API is rejected.
- A caller-constructed historical wrapper sent to a normal runner is rejected.
- A path, hash, stage, schema, game identity, or symlink mismatch is rejected
  before ledger lookup.
- Historical replay never falls through to current contract validation,
  forecasting, append, or persistence.

## Testing

Tests will prove all of the following for projected and confirmed stages:

- a strict current bundle can still create a normal fresh run;
- a strict current exact replay remains unchanged;
- a hash-named pre-change artifact under an empty managed root cannot create a
  ledger or managed sibling;
- a genuine pre-change two-market artifact can replay its exact existing
  capture, forecast, and link;
- the replay result references the original authenticated path and full hash;
- replay creates no new strict-looking artifact;
- reusing the returned historical path against an empty ledger is rejected;
- direct submission of a caller-constructed compatibility wrapper to a normal
  runner is rejected;
- substituting a strict path, forged hash, or forged replay flag cannot affect
  the returned provenance because the service derives it from disk;
- frozen historical Shadow payloads remain readable;
- fresh schema-v1 input still requires a real boolean field;
- the full regression suite, pipeline manifest, and diff checks remain green.

## Non-Goals

- Migrating all bundles to a new schema version.
- Creating a general managed-artifact registry.
- Changing same-game candidate assembly, scores, market directions, or output
  ranking.
- Allowing historical bundles to produce corrected or superseding forecasts.
