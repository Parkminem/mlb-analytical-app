# Forecasting Ledger Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build typed, leakage-safe pregame inputs and an append-only idempotent ledger that can reproduce one deterministic forecast from any eligible source-state capture.

**Architecture:** Add new `analysis.forecasting` and `analysis.ledger` packages beside the authoritative `analysis.model` module. Capture records retain timing and provenance, forecast records are keyed only by semantic input state, and immutable link records join repeated captures to one forecast without duplicating calibration samples.

**Tech Stack:** Python 3 frozen dataclasses and enums, standard-library JSON/SHA-256/UUID/file locking, `unittest`, JSON Lines

---

## Dependencies And Constraints

- Implement this plan before the probability-engine and review/calibration plans.
- Do not modify scoring behavior in `analysis/model.py`.
- Use timezone-aware UTC timestamps internally; Korean date remains an explicit game-identity field.
- Treat unknown `available_at` as retrospective-only, never as eligible out-of-sample data.
- Keep local ledger files outside Git under `data/ledger/`.
- Canonical forecast identity excludes capture/fetch timestamps and generated IDs but includes semantic source content hashes, line, prices, model version, and simulation configuration.

## File Map

- Create `analysis/forecasting/__init__.py`: public forecasting-schema exports only.
- Create `analysis/forecasting/inputs.py`: source observations, provenance, lineup, pitcher-chain, bullpen, market, and normalized game-input dataclasses.
- Create `analysis/forecasting/identity.py`: canonical serialization, semantic payload construction, input hashes, and deterministic forecast-run IDs.
- Create `analysis/forecasting/validation.py`: timestamp, lineup, relationship, and leakage validation.
- Create `analysis/ledger/__init__.py`: public ledger exports.
- Create `analysis/ledger/records.py`: immutable capture, forecast, link, market observation, result, and review record envelopes.
- Create `analysis/ledger/store.py`: append-only JSONL storage, file locking, replay detection, conflict rejection, and record iteration.
- Create `tests/test_forecasting_inputs.py`: schema, canonicalization, forecast identity, and leakage tests.
- Create `tests/test_ledger.py`: append-only, replay, conflict, linking, correction, and calibration-deduplication fixtures.
- Modify `.gitignore`: ignore `data/ledger/`.

### Task 1: Typed Pregame Inputs And Provenance

**Files:**
- Create: `analysis/forecasting/__init__.py`
- Create: `analysis/forecasting/inputs.py`
- Test: `tests/test_forecasting_inputs.py`

**Interfaces:**
- `SourceObservation` records source identity and all timing fields.
- `FeatureProvenance` records observation IDs, latest availability, and lookback end.
- `NormalizedGameInput` contains both full lineups, pitcher chains, bullpens, markets, and source observations without importing legacy decision contexts.

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_forecasting_inputs.py` with focused builders and these first assertions:

```python
from datetime import date, datetime, timezone
import unittest

from analysis.forecasting.inputs import (
    BooleanFeature,
    FeatureProvenance,
    GameIdentity,
    LineupInput,
    LineupSlot,
    LineupState,
    NumericFeature,
    SourceObservation,
)


UTC = timezone.utc


class ForecastInputSchemaTests(unittest.TestCase):
    def test_source_timestamp_must_be_timezone_aware(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            SourceObservation(
                observation_id="mlb:1",
                source_name="MLB",
                source_record="game/1",
                observed_at=datetime(2026, 7, 14, 1, 0),
                available_at=datetime(2026, 7, 14, 1, 0, tzinfo=UTC),
                fetched_at=datetime(2026, 7, 14, 1, 1, tzinfo=UTC),
                effective_date=date(2026, 7, 14),
                content_hash="a" * 64,
            )

    def test_lineup_requires_exactly_nine_unique_slots(self):
        slots = tuple(
            LineupSlot(
                order=order,
                batter_id=f"b{order}",
                batter_name=f"Batter {order}",
                bats="R",
                expected_side="R",
                responses=(),
            )
            for order in range(1, 9)
        )
        with self.assertRaisesRegex(ValueError, "nine"):
            LineupInput(
                team="SEA",
                state=LineupState.CONFIRMED,
                state_provenance=provenance(),
                slots=slots,
                projected_runs_per_9=NumericFeature(4.5, provenance()),
                projected_traffic_per_9=NumericFeature(11.5, provenance()),
                projected_extra_base_hits_per_9=NumericFeature(2.4, provenance()),
                walk_rate=NumericFeature(0.08, provenance()),
                strikeout_rate=NumericFeature(0.22, provenance()),
                isolated_power=NumericFeature(0.16, provenance()),
                barrel_rate=NumericFeature(0.08, provenance()),
                recent_form_index=NumericFeature(1.0, provenance()),
                lower_order_continuity=NumericFeature(1.0, provenance()),
                log_standard_error=NumericFeature(0.08, provenance()),
                recent_form_included=BooleanFeature(True, provenance()),
            )

    def test_game_identity_keeps_korean_and_official_dates(self):
        identity = GameIdentity(
            game_pk=900001,
            korean_date=date(2026, 7, 15),
            official_game_date=date(2026, 7, 14),
            first_pitch=datetime(2026, 7, 14, 23, 10, tzinfo=UTC),
            away_team="SEA",
            home_team="MIN",
        )
        self.assertEqual(identity.korean_date.isoformat(), "2026-07-15")

    def test_missing_market_price_remains_valid_for_flow_review(self):
        market = market_input(selected_decimal_price=None, opposing_decimal_price=None)
        self.assertIsNone(market.selected_decimal_price)
```

Add a local `provenance()` fixture returning an eligible `FeatureProvenance`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_inputs.ForecastInputSchemaTests -v`

Expected: FAIL because `analysis.forecasting.inputs` does not exist.

- [ ] **Step 3: Add the minimal typed schema**

Implement frozen dataclasses and string enums in `analysis/forecasting/inputs.py`. Keep validation in `__post_init__` limited to structural invariants:

```python
class LineupState(str, Enum):
    PROJECTED = "projected"
    CONFIRMED = "confirmed"


class PitcherRole(str, Enum):
    STARTER = "starter"
    OPENER = "opener"
    BULK = "bulk"
    PRIM = "prim"


class BullpenRole(str, Enum):
    BRIDGE = "bridge"
    MIDDLE = "middle"
    HIGH_LEVERAGE = "high_leverage"
    CLOSER = "closer"


class MarketType(str, Enum):
    F5_SIDE = "f5_side"
    F5_TOTAL = "f5_total"
    ML = "ml"
    PLUS_1_5 = "plus_1_5"
    FULL_TOTAL = "full_total"


class SettlementFormat(str, Enum):
    TWO_WAY = "two_way"
    THREE_WAY = "three_way"


@dataclass(frozen=True)
class SourceObservation:
    observation_id: str
    source_name: str
    source_record: str
    observed_at: datetime
    available_at: datetime | None
    fetched_at: datetime
    effective_date: date
    content_hash: str


@dataclass(frozen=True)
class FeatureProvenance:
    observation_ids: tuple[str, ...]
    latest_available_at: datetime | None
    lookback_end: datetime | None


@dataclass(frozen=True)
class NumericFeature:
    value: float
    provenance: FeatureProvenance


@dataclass(frozen=True)
class IntegerFeature:
    value: int
    provenance: FeatureProvenance


@dataclass(frozen=True)
class BooleanFeature:
    value: bool
    provenance: FeatureProvenance


@dataclass(frozen=True)
class PitchUsage:
    pitcher_id: str
    opposing_side: str
    pitch_type: str
    usage: float
    average_velocity_mph: float
    provenance: FeatureProvenance


@dataclass(frozen=True)
class PitchVelocityResponse:
    batter_id: str
    pitcher_id: str
    expected_side: str
    pitch_type: str
    velocity_center_mph: float
    band_width_mph: float
    plate_appearances: int
    adjusted_woba: float
    sample_strength: str
    slot_label: str
    provenance: FeatureProvenance


@dataclass(frozen=True)
class LineupSlot:
    order: int
    batter_id: str
    batter_name: str
    bats: str
    expected_side: str
    responses: tuple[PitchVelocityResponse, ...]


@dataclass(frozen=True)
class LineupInput:
    team: str
    state: LineupState
    state_provenance: FeatureProvenance
    slots: tuple[LineupSlot, ...]
    projected_runs_per_9: NumericFeature
    projected_traffic_per_9: NumericFeature
    projected_extra_base_hits_per_9: NumericFeature
    walk_rate: NumericFeature
    strikeout_rate: NumericFeature
    isolated_power: NumericFeature
    barrel_rate: NumericFeature
    recent_form_index: NumericFeature
    lower_order_continuity: NumericFeature
    log_standard_error: NumericFeature
    recent_form_included: BooleanFeature


@dataclass(frozen=True)
class WorkloadHistory:
    recorded_outs: tuple[int, ...]
    provenance: FeatureProvenance


@dataclass(frozen=True)
class StarterProjectionComponents:
    strikeout_minus_walk_rate: NumericFeature
    walk_rate: NumericFeature
    home_runs_per_contact: NumericFeature
    pitches_per_inning: NumericFeature
    recent_pitch_count: NumericFeature
    recent_workload_outs: NumericFeature


@dataclass(frozen=True)
class PitcherInput:
    pitcher_id: str
    pitcher_name: str
    role: PitcherRole
    throws: str
    projected_ra9: NumericFeature
    projection_components: StarterProjectionComponents
    log_standard_error: NumericFeature
    eligible_workload: WorkloadHistory
    league_prior_workload: WorkloadHistory
    pitch_limit: IntegerFeature | None
    pitch_mix: tuple[PitchUsage, ...]
    command_risk: BooleanFeature
    home_run_risk: BooleanFeature
    injury_return: BooleanFeature
    recent_short_exit: BooleanFeature


@dataclass(frozen=True)
class PitcherChainInput:
    pitchers: tuple[PitcherInput, ...]
    official_pitcher_id: str
    rotowire_pitcher_id: str | None
    matchup_pitcher_id: str
    starter_confirmed: BooleanFeature
    starter_source_conflict: BooleanFeature
    handoff_verified: BooleanFeature
    timing_uncertain: BooleanFeature
    eligible_combined_workload: WorkloadHistory | None
    league_prior_combined_workload: WorkloadHistory | None
    combined_pitch_limit: IntegerFeature | None


@dataclass(frozen=True)
class RelieverAppearanceInput:
    pitcher_id: str
    days_ago: int
    pitches: int
    provenance: FeatureProvenance


@dataclass(frozen=True)
class BullpenSegmentInput:
    segment_id: str
    role: BullpenRole
    projected_ra9: NumericFeature
    log_standard_error: NumericFeature
    appearances: tuple[RelieverAppearanceInput, ...]
    available: BooleanFeature
    limited_performance_or_use_risk: BooleanFeature


@dataclass(frozen=True)
class BullpenInput:
    team: str
    segments: tuple[BullpenSegmentInput, ...]


@dataclass(frozen=True)
class MarketInput:
    market_key: str
    market_type: MarketType
    settlement_format: SettlementFormat
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None
    provenance: FeatureProvenance


@dataclass(frozen=True)
class GameIdentity:
    game_pk: int
    korean_date: date
    official_game_date: date
    first_pitch: datetime
    away_team: str
    home_team: str


@dataclass(frozen=True)
class NormalizedGameInput:
    identity: GameIdentity
    captured_at: datetime
    away_lineup: LineupInput
    home_lineup: LineupInput
    away_pitching: PitcherChainInput
    home_pitching: PitcherChainInput
    away_bullpen: BullpenInput
    home_bullpen: BullpenInput
    markets: tuple[MarketInput, ...]
    observations: tuple[SourceObservation, ...]
```

Validate timezone awareness, positive odds when present, usage in `[0, 1]`, legal batter sides, valid velocity bands (`1.5` or `3.0`), workload outs in `[0, 27]`, nonnegative reliever pitches, and exactly nine unique batting-order slots. Verify official, RotoWire, matchup, opener, bulk, and PRIM pitcher IDs reference members of the chain. Require combined workload and league-prior histories for any multi-pitcher opener/bulk/PRIM chain and forbid them for a single conventional starter. Do not assign strength penalties here.

- [ ] **Step 4: Export the public schema and verify GREEN**

Re-export only stable public types from `analysis/forecasting/__init__.py` and run:

Run: `python3 -m unittest tests.test_forecasting_inputs.ForecastInputSchemaTests -v`

Expected: all schema tests PASS.

- [ ] **Step 5: Commit the schema**

```bash
git add analysis/forecasting/__init__.py analysis/forecasting/inputs.py tests/test_forecasting_inputs.py
git commit -m "feat: add normalized forecasting inputs"
```

### Task 2: Canonical Identity And Leakage Validation

**Files:**
- Create: `analysis/forecasting/identity.py`
- Create: `analysis/forecasting/validation.py`
- Modify: `tests/test_forecasting_inputs.py`

**Interfaces:**
- `canonical_json(value) -> str`
- `normalized_input_hash(game_input) -> str`
- `build_forecast_run_id(game_input, model_version, simulation_config) -> str`
- `validate_pregame_input(game_input) -> ValidationResult`

- [ ] **Step 1: Write failing identity and leakage tests**

Add tests proving semantic equality and leakage enforcement:

```python
def test_forecast_id_ignores_capture_and_fetch_timestamps(self):
    first = complete_game_input(captured_at=dt(1), fetched_at=dt(0))
    second = complete_game_input(captured_at=dt(2), fetched_at=dt(1))
    self.assertEqual(
        build_forecast_run_id(first, "shadow-1", {"simulations": 20_000}),
        build_forecast_run_id(second, "shadow-1", {"simulations": 20_000}),
    )

def test_forecast_id_changes_when_line_or_lineup_changes(self):
    baseline = complete_game_input()
    changed_line = replace_market_line(baseline, 8.5)
    changed_lineup = replace_batter(baseline, order=4, batter_id="replacement")
    baseline_id = build_forecast_run_id(baseline, "shadow-1", config())
    self.assertNotEqual(baseline_id, build_forecast_run_id(changed_line, "shadow-1", config()))
    self.assertNotEqual(baseline_id, build_forecast_run_id(changed_lineup, "shadow-1", config()))

def test_future_available_feature_is_rejected(self):
    game = complete_game_input(
        captured_at=dt(1),
        feature_available_at=dt(2),
    )
    with self.assertRaisesRegex(ValueError, "available after capture"):
        validate_pregame_input(game)

def test_unknown_available_at_is_retrospective_only(self):
    result = validate_pregame_input(
        complete_game_input(feature_available_at=None)
    )
    self.assertFalse(result.out_of_sample_eligible)
    self.assertIn("unknown available_at", result.flags)
```

- [ ] **Step 2: Run identity tests and verify RED**

Run: `python3 -m unittest tests.test_forecasting_inputs.ForecastIdentityTests -v`

Expected: FAIL because identity and validation modules do not exist.

- [ ] **Step 3: Implement canonical serialization and semantic payloads**

In `identity.py`, recursively serialize frozen dataclasses, enums, dates, datetimes, tuples, lists, and mappings. Normalize datetimes to UTC with a `Z` suffix and sort mapping keys. Reject unsupported objects instead of using `repr`.

Build two payloads:

```python
def capture_payload(game: NormalizedGameInput) -> dict[str, object]:
    """Full timing, source, and normalized-input payload for audit."""


def semantic_forecast_payload(game: NormalizedGameInput) -> dict[str, object]:
    """Forecast inputs without captured_at/observed_at/fetched_at/generated IDs."""
```

The semantic payload must retain source name, stable source record, effective date, content hash, all normalized feature values, lineup status, pitcher structure, and market type/format/line/prices. Compute SHA-256 over UTF-8 canonical JSON:

```python
def build_forecast_run_id(game, model_version, simulation_config):
    payload = {
        "input": semantic_forecast_payload(game),
        "model_version": model_version,
        "simulation_config": simulation_config,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Implement explicit provenance traversal and validation**

Add `iter_feature_provenance(game)` that explicitly yields provenance from lineup status, every run-creation feature and uncertainty input, every pitch usage, every hitter response, pitcher RA9 plus every projection component and uncertainty input, individual and combined workload histories, individual and combined pitch limits, starter confirmation/conflict/risk flags, handoff state, every reliever appearance, bullpen availability/risk, bullpen-segment RA9 and uncertainty, and every market input. Do not rely on field-name reflection for leakage checks.

`validate_pregame_input` must:

- Reject `captured_at >= first_pitch` for primary pregame records.
- Reject a feature whose `latest_available_at > captured_at`.
- Reject a lookback ending after capture.
- Reject an observation whose `available_at > captured_at` when used by a feature.
- Reject missing referenced observation IDs.
- Return `out_of_sample_eligible=False` rather than rejecting when `available_at` is unknown.
- Confirm market keys are unique and both team/lineup/bullpen relationships match game identity.

- [ ] **Step 5: Run identity and schema tests and verify GREEN**

Run: `python3 -m unittest tests.test_forecasting_inputs -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit identity and leakage controls**

```bash
git add analysis/forecasting/identity.py analysis/forecasting/validation.py tests/test_forecasting_inputs.py
git commit -m "feat: add deterministic forecast identity"
```

### Task 3: Append-Only Idempotent Ledger

**Files:**
- Create: `analysis/ledger/__init__.py`
- Create: `analysis/ledger/records.py`
- Create: `analysis/ledger/store.py`
- Create: `tests/test_ledger.py`

**Interfaces:**
- `CaptureCommand`, `ForecastCommand`, `MarketObservationCommand`, `OfficialResultCommand`, `ReviewCommand`
- `LedgerStore.append_capture(...) -> CaptureRecord`
- `LedgerStore.append_forecast(...) -> ForecastRecord`
- `LedgerStore.link_capture_forecast(...) -> CaptureForecastLink`
- `LedgerStore.append_market_observation(...)`
- `LedgerStore.append_official_result(...)`
- `LedgerStore.iter_records(kind=None)`

- [ ] **Step 1: Write failing replay, conflict, and relationship tests**

Use `tempfile.TemporaryDirectory` and verify:

```python
def test_same_capture_retry_returns_existing_record(self):
    first = self.store.append_capture("capture:mlb:1:lineups-v1", capture_command())
    second = self.store.append_capture("capture:mlb:1:lineups-v1", capture_command())
    self.assertEqual(first.capture_id, second.capture_id)
    self.assertEqual(len(tuple(self.store.iter_records("capture"))), 1)

def test_same_key_with_different_payload_is_rejected(self):
    self.store.append_capture("capture:key", capture_command())
    with self.assertRaises(IdempotencyConflict):
        self.store.append_capture("capture:key", changed_capture_command())

def test_two_captures_can_link_to_one_forecast_without_duplicate_forecast(self):
    capture_a = self.store.append_capture("capture:a", capture_command())
    capture_b = self.store.append_capture("capture:b", capture_command(captured_at=dt(2)))
    forecast = self.store.append_forecast(forecast_command(run_id="f" * 64))
    self.store.link_capture_forecast(capture_a.capture_id, forecast.forecast_run_id)
    self.store.link_capture_forecast(capture_b.capture_id, forecast.forecast_run_id)
    self.assertEqual(len(tuple(self.store.iter_records("forecast"))), 1)
    self.assertEqual(len(tuple(self.store.iter_records("capture_forecast_link"))), 2)

def test_forecast_payload_cannot_contain_capture_id(self):
    with self.assertRaisesRegex(ValueError, "capture_id"):
        self.store.append_forecast(forecast_command(outputs={"capture_id": "x"}))

def test_corrected_result_supersedes_without_mutating_original(self):
    first = self.store.append_official_result(result_command(revision="1", away_runs=4))
    second = self.store.append_official_result(
        result_command(revision="2", away_runs=5, supersedes=first.record_id)
    )
    self.assertEqual(first.away_full_runs, 4)
    self.assertEqual(second.supersedes_record_id, first.record_id)

def test_close_with_different_market_semantics_is_not_matchable(self):
    entry = self.store.append_market_observation(entry_market(line=8.0))
    close = self.store.append_market_observation(close_market(line=8.5))
    self.assertFalse(self.store.market_observations_match(entry, close))

def test_review_requires_existing_link_forecast_result_and_market(self):
    with self.assertRaisesRegex(ValueError, "capture-forecast link"):
        self.store.append_review(review_command(link_record_id="missing"))
```

Also test reopening the store reconstructs the idempotency index from JSONL and a replay does not append a second line.

- [ ] **Step 2: Run ledger tests and verify RED**

Run: `python3 -m unittest tests.test_ledger -v`

Expected: FAIL because `analysis.ledger` does not exist.

- [ ] **Step 3: Define immutable record envelopes**

In `records.py`, define frozen records with a common envelope:

```python
@dataclass(frozen=True)
class RecordEnvelope:
    record_id: str
    record_kind: str
    idempotency_key: str
    payload_hash: str
    appended_at: datetime


@dataclass(frozen=True)
class CaptureRecord:
    envelope: RecordEnvelope
    capture_id: str
    game_pk: int
    captured_at: datetime
    first_pitch: datetime
    lineup_state: str
    out_of_sample_eligible: bool
    normalized_input_hash: str
    payload: Mapping[str, object]
    supersedes_capture_id: str | None = None


@dataclass(frozen=True)
class ForecastRecord:
    envelope: RecordEnvelope
    forecast_run_id: str
    model_version: str
    simulation_config_hash: str
    normalized_input_hash: str
    outputs: Mapping[str, object]


@dataclass(frozen=True)
class CaptureForecastLink:
    envelope: RecordEnvelope
    capture_id: str
    forecast_run_id: str


@dataclass(frozen=True)
class MarketObservationRecord:
    envelope: RecordEnvelope
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    observation_kind: str  # entry or close
    market_key: str
    market_type: str
    settlement_format: str
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None
    observed_at: datetime
    available_at: datetime | None
    supersedes_record_id: str | None = None


@dataclass(frozen=True)
class OfficialResultRecord:
    envelope: RecordEnvelope
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    finalized_at: datetime
    away_f5_runs: int
    home_f5_runs: int
    away_full_runs: int
    home_full_runs: int
    component_values: Mapping[str, float]
    documented_events: tuple[Mapping[str, object], ...]
    supersedes_record_id: str | None = None


@dataclass(frozen=True)
class ReviewRecord:
    envelope: RecordEnvelope
    review_id: str
    capture_forecast_link_record_id: str
    capture_id: str
    forecast_run_id: str
    official_result_record_id: str
    market_key: str
    market_result: str
    classification: str | None
    inside_80_ratio: float
    contradicted_paths: tuple[str, ...]
    component_reviews: tuple[Mapping[str, object], ...]
```

`component_values` uses the fixed `away`/`home` component IDs produced by the probability engine, including F5/full runs, traffic, extra-base hits, pitcher-chain runs, bullpen runs, starter workload outs, bullpen-segment exposure, and protected close-game bullpen entry. Reject team-abbreviation component namespaces. `ForecastRecord.outputs` and all mapping fields must contain canonical JSON data. Recursively reject `capture_id` inside forecast output.

Define frozen command dataclasses with the same domain fields but without generated envelope/record IDs. `CaptureCommand` includes game/capture timing, lineup state, eligibility, normalized input hash/payload, and optional superseded capture. `ForecastCommand` includes run/model/config/input hashes and outputs. `MarketObservationCommand`, `OfficialResultCommand`, and `ReviewCommand` contain exactly the corresponding record fields after `envelope`, with optional caller idempotency key kept as an append-method argument rather than payload. This one-to-one rule prevents a second undocumented persistence schema.

- [ ] **Step 4: Implement safe append and replay semantics**

In `store.py`:

- Open a dedicated `.lock` file and use `fcntl.flock(..., LOCK_EX)` around index rebuild plus append.
- Rebuild the `(record_kind, idempotency_key)` index from the JSONL file while holding the lock.
- Compare the logical payload hash before generating a UUID-backed capture ID.
- Return the existing deserialized record on an exact replay.
- Raise `IdempotencyConflict` when a key is reused with another logical payload.
- Append one canonical JSON object followed by `\n`, then `flush()` and `os.fsync()` before releasing the lock.
- Treat any optional `.index.json` as a disposable cache; the JSONL file is authoritative.
- Use `forecast:{forecast_run_id}` for forecasts and `capture-forecast:{capture_id}:{forecast_run_id}` for links.
- Verify referenced capture and forecast records exist before appending a link.
- Expose concrete append signatures:

```python
append_market_observation(command, idempotency_key=None) -> MarketObservationRecord
append_official_result(command, idempotency_key=None) -> OfficialResultRecord
append_review(command, idempotency_key=None) -> ReviewRecord
```

- Derive default keys as `market:{source_name}:{source_record}:{source_revision}:{observation_kind}:{market_key}`, `result:{source_name}:{game_pk}:{source_revision}`, and `review:{capture_id}:{forecast_run_id}:{official_result_record_id}:{market_key}`.
- Require a close observation to match entry `game_pk`, market type, settlement format, selected direction, and line before CLV use. A differing line remains auditable but is not a matching close for price CLV.
- Require result revisions after the first to reference the prior latest result with `supersedes_record_id`. Require review records to reference an existing link, forecast, result, and market key.

- [ ] **Step 5: Implement non-superseded ledger query primitives**

Add query helpers that return the immutable relationship graph without applying a reporting cutoff:

- `records_superseded_by(kind) -> Mapping[record_id, superseding_record_id]`
- `latest_official_result(game_pk) -> OfficialResultRecord | None`
- `links_for_capture(capture_id) -> tuple[CaptureForecastLink, ...]`
- `forecast_for_link(link_record_id) -> ForecastRecord`

Do not choose a primary capture or deduplicate calibration rows in the ledger package. The review/calibration plan owns one explicit `CaptureCutoffPolicy` and all headline sample selection.

- [ ] **Step 6: Run ledger tests and verify GREEN**

Run: `python3 -m unittest tests.test_ledger -v`

Expected: all ledger tests PASS, including close/reopen replay.

- [ ] **Step 7: Commit the ledger**

```bash
git add analysis/ledger/__init__.py analysis/ledger/records.py analysis/ledger/store.py tests/test_ledger.py
git commit -m "feat: add append-only forecast ledger"
```

### Task 4: Local Data Boundary And Foundation Verification

**Files:**
- Modify: `.gitignore`
- Verify: all files created in this plan

- [ ] **Step 1: Add the ledger data directory to Git ignore rules**

Add:

```gitignore
data/ledger/
```

- [ ] **Step 2: Verify the complete foundation test set**

Run: `python3 -m unittest tests.test_forecasting_inputs tests.test_ledger -v`

Expected: all new foundation tests PASS.

- [ ] **Step 3: Verify legacy regressions**

Run: `python3 -m unittest discover -s tests -v`

Expected: all existing 141 tests plus the new foundation tests PASS.

- [ ] **Step 4: Inspect manifest and formatting**

Run: `python3 -m analysis.pipeline --lineup confirmed`

Expected: the legacy fixed model remains authoritative and required outputs are unchanged.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Commit the local data boundary**

```bash
git add .gitignore
git commit -m "chore: ignore local forecast ledger"
```

## Completion Gate

- A capture retry cannot add a row or change a generated capture ID.
- An unchanged second capture can link to the same forecast without duplicating forecast or calibration rows.
- A changed lineup or market state produces a different forecast-run ID.
- Every eligible feature is traceable to observations available before capture.
- The legacy model and all existing tests remain unchanged.
