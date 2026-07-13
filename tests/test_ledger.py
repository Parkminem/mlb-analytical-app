from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from analysis.ledger import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    IdempotencyConflict,
    MarketObservationCommand,
    MarketObservationRecord,
    OfficialResultCommand,
    OfficialResultRecord,
    ReviewCommand,
    ReviewRecord,
    LedgerCorruptionError,
    LedgerReferenceError,
    LedgerStore,
    RecordEnvelope,
)
from analysis.forecasting.identity import canonical_json


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 14, 1, 30, tzinfo=UTC)
FIRST_PITCH = datetime(2026, 7, 14, 23, 10, tzinfo=UTC)
OBSERVED_AT = datetime(2026, 7, 14, 1, 20, tzinfo=UTC)
AVAILABLE_AT = datetime(2026, 7, 14, 1, 21, tzinfo=UTC)
FINALIZED_AT = datetime(2026, 7, 15, 3, 45, tzinfo=UTC)


def capture_command(**overrides: object) -> CaptureCommand:
    values: dict[str, object] = {
        "game_pk": 900001,
        "captured_at": CAPTURED_AT,
        "first_pitch": FIRST_PITCH,
        "lineup_state": "confirmed",
        "out_of_sample_eligible": True,
        "normalized_input_hash": "input-hash-1",
        "payload": {"away_team": "SEA", "home_team": "MIN"},
    }
    values.update(overrides)
    return CaptureCommand(**values)


def forecast_command(**overrides: object) -> ForecastCommand:
    values: dict[str, object] = {
        "forecast_run_id": "forecast-run-1",
        "model_version": "model-v1",
        "simulation_config_hash": "simulation-hash-1",
        "normalized_input_hash": "input-hash-1",
        "outputs": {
            "markets": [
                {
                    "market_key": "game:900001:ml:SEA",
                    "selected_probability": 0.55,
                }
            ]
        },
    }
    values.update(overrides)
    return ForecastCommand(**values)


def market_command(**overrides: object) -> MarketObservationCommand:
    values: dict[str, object] = {
        "game_pk": 900001,
        "source_name": "sportsbook",
        "source_record": "event-900001-market-1",
        "source_revision": "1",
        "observation_kind": "entry",
        "market_key": "game:900001:ml:SEA",
        "market_type": "ml",
        "settlement_format": "two_way",
        "selected_side": "SEA",
        "line": None,
        "selected_decimal_price": 1.91,
        "opposing_decimal_price": 1.95,
        "observed_at": OBSERVED_AT,
        "available_at": AVAILABLE_AT,
        "supersedes_record_id": None,
    }
    values.update(overrides)
    return MarketObservationCommand(**values)


def result_command(**overrides: object) -> OfficialResultCommand:
    values: dict[str, object] = {
        "game_pk": 900001,
        "source_name": "MLB",
        "source_record": "game/900001/boxscore",
        "source_revision": "1",
        "finalized_at": FINALIZED_AT,
        "away_f5_runs": 2,
        "home_f5_runs": 1,
        "away_full_runs": 4,
        "home_full_runs": 3,
        "component_values": {
            "away_f5_runs": 2.0,
            "home_f5_runs": 1.0,
            "away_bullpen_segment_high_leverage_exposed": 1.0,
            "home_protected_bullpen_close_entry": 0.0,
        },
        "documented_events": [
            {"event": "home_run", "inning": 3, "team_id": "away"}
        ],
        "supersedes_record_id": None,
    }
    values.update(overrides)
    return OfficialResultCommand(**values)


def review_command(**overrides: object) -> ReviewCommand:
    values: dict[str, object] = {
        "capture_forecast_link_record_id": "link-record-id",
        "capture_id": "capture-id",
        "forecast_run_id": "forecast-run-1",
        "official_result_record_id": "result-record-id",
        "market_key": "game:900001:ml:SEA",
        "market_result": "win",
        "classification": "inside_80",
        "inside_80_ratio": 0.75,
        "contradicted_paths": ["away_lineup.power"],
        "component_reviews": [
            {
                "evidence_path": "away_lineup.power",
                "classification": "supported",
            }
        ],
    }
    values.update(overrides)
    return ReviewCommand(**values)


def append_review_context(store: LedgerStore):
    capture = store.append_capture("capture:review", capture_command())
    forecast = store.append_forecast(forecast_command())
    link = store.link_capture_forecast(
        capture.capture_id,
        forecast.forecast_run_id,
    )
    market = store.append_market_observation(market_command())
    result = store.append_official_result(result_command())
    command = review_command(
        capture_forecast_link_record_id=link.envelope.record_id,
        capture_id=capture.capture_id,
        forecast_run_id=forecast.forecast_run_id,
        official_result_record_id=result.envelope.record_id,
        market_key=market.market_key,
    )
    return capture, forecast, link, market, result, command


def append_same_capture_in_process(path: str) -> None:
    LedgerStore(path).append_capture(
        "capture:concurrent",
        capture_command(payload={"source": "concurrent"}),
    )


class LedgerStoreTests(unittest.TestCase):
    def test_same_capture_retry_returns_same_capture_and_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)

            first = store.append_capture("capture:900001:pregame", capture_command())
            second = store.append_capture("capture:900001:pregame", capture_command())

            self.assertEqual(first.capture_id, second.capture_id)
            self.assertEqual(first, second)
            self.assertEqual(len(tuple(store.iter_records("capture"))), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_same_capture_key_with_different_payload_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            store.append_capture("capture:900001:pregame", capture_command())

            with self.assertRaises(IdempotencyConflict):
                store.append_capture(
                    "capture:900001:pregame",
                    capture_command(payload={"away_team": "SEA", "revision": 2}),
                )

    def test_reopen_rebuilds_capture_idempotency_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = LedgerStore(path).append_capture(
                "capture:900001:pregame",
                capture_command(),
            )
            line_count = len(path.read_text(encoding="utf-8").splitlines())

            replayed = LedgerStore(path).append_capture(
                "capture:900001:pregame",
                capture_command(),
            )

            self.assertEqual(replayed.capture_id, first.capture_id)
            self.assertEqual(
                len(path.read_text(encoding="utf-8").splitlines()),
                line_count,
            )

    def test_two_captures_link_to_one_forecast_without_duplicate_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            first_capture = store.append_capture("capture:1", capture_command())
            second_capture = store.append_capture(
                "capture:2",
                capture_command(
                    captured_at=datetime(2026, 7, 14, 1, 45, tzinfo=UTC),
                    payload={"lineup_revision": 2},
                ),
            )
            forecast = store.append_forecast(forecast_command())

            first_link = store.link_capture_forecast(
                first_capture.capture_id,
                forecast.forecast_run_id,
            )
            second_link = store.link_capture_forecast(
                second_capture.capture_id,
                forecast.forecast_run_id,
            )

            self.assertNotEqual(
                first_link.envelope.record_id,
                second_link.envelope.record_id,
            )
            self.assertEqual(len(tuple(store.iter_records("forecast"))), 1)
            self.assertEqual(
                len(tuple(store.iter_records("capture_forecast_link"))),
                2,
            )
            self.assertEqual(store.links_for_capture(first_capture.capture_id), (first_link,))
            self.assertEqual(
                store.forecast_for_link(second_link.envelope.record_id),
                forecast,
            )

    def test_forecast_replay_is_idempotent_and_changed_outputs_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            first = store.append_forecast(forecast_command())

            replayed = store.append_forecast(forecast_command())

            self.assertEqual(replayed, first)
            with self.assertRaises(IdempotencyConflict):
                store.append_forecast(
                    forecast_command(outputs={"selected_probability": 0.60})
                )
            self.assertEqual(len(tuple(store.iter_records("forecast"))), 1)

    def test_forecast_outputs_reject_nested_capture_identity(self) -> None:
        invalid_outputs = (
            {"capture_id": "capture-1"},
            {"nested": {"capture_id": "capture-1"}},
            {"nested": [{"deeper": {"capture_id": "capture-1"}}]},
        )

        for outputs in invalid_outputs:
            with self.subTest(outputs=outputs):
                with self.assertRaises(ValueError):
                    forecast_command(outputs=outputs)

    def test_caller_mutation_cannot_change_capture_or_forecast_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            payload = {"lineup": [{"batter": "A"}]}
            outputs = {"markets": [{"probability": 0.55}]}
            capture = store.append_capture(
                "capture:immutable",
                capture_command(payload=payload),
            )
            forecast = store.append_forecast(forecast_command(outputs=outputs))

            payload["lineup"][0]["batter"] = "MUTATED"
            outputs["markets"][0]["probability"] = 0.99

            self.assertEqual(capture.payload["lineup"][0]["batter"], "A")
            self.assertEqual(forecast.outputs["markets"][0]["probability"], 0.55)
            reopened = tuple(LedgerStore(path).iter_records())
            self.assertEqual(reopened[0], capture)
            self.assertEqual(reopened[1], forecast)

    def test_link_rejects_missing_capture_and_forecast_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(forecast_command())

            with self.assertRaises(LedgerReferenceError):
                store.link_capture_forecast("missing-capture", forecast.forecast_run_id)
            with self.assertRaises(LedgerReferenceError):
                store.link_capture_forecast(capture.capture_id, "missing-forecast")

    def test_capture_supersession_requires_existing_same_game_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_capture("capture:1", capture_command())
            replacement = store.append_capture(
                "capture:2",
                capture_command(
                    payload={"revision": 2},
                    supersedes_capture_id=original.capture_id,
                ),
            )

            self.assertEqual(replacement.supersedes_capture_id, original.capture_id)
            self.assertIsNone(original.supersedes_capture_id)

            with self.assertRaises(LedgerReferenceError):
                store.append_capture(
                    "capture:missing",
                    capture_command(supersedes_capture_id="missing-capture"),
                )

            other_game = store.append_capture(
                "capture:other-game",
                capture_command(game_pk=900002, payload={"game": 900002}),
            )
            with self.assertRaises(LedgerReferenceError):
                store.append_capture(
                    "capture:cross-game",
                    capture_command(
                        payload={"revision": 3},
                        supersedes_capture_id=other_game.capture_id,
                    ),
                )

    def test_commands_records_and_nested_snapshots_are_immutable(self) -> None:
        command = forecast_command(outputs={"nested": [{"value": 1.0}]})
        self.assertTrue(is_dataclass(command))
        self.assertTrue(command.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            command.model_version = "mutated"
        with self.assertRaises(TypeError):
            command.outputs["new"] = "value"
        with self.assertRaises(TypeError):
            command.outputs["nested"][0]["value"] = 2.0

        required_types = (
            RecordEnvelope,
            CaptureCommand,
            CaptureRecord,
            ForecastCommand,
            ForecastRecord,
            CaptureForecastLink,
        )
        for dataclass_type in required_types:
            with self.subTest(dataclass_type=dataclass_type.__name__):
                self.assertTrue(is_dataclass(dataclass_type))
                self.assertTrue(dataclass_type.__dataclass_params__.frozen)
                self.assertGreater(len(fields(dataclass_type)), 0)

    def test_forecast_outputs_reject_noncanonical_and_nonfinite_values(self) -> None:
        invalid_outputs = (
            {"bad": float("nan")},
            {"bad": float("inf")},
            {"bad": {1: "non-string key"}},
            {"bad": {"not", "json"}},
            {"bad": object()},
        )

        for outputs in invalid_outputs:
            with self.subTest(outputs=repr(outputs)):
                with self.assertRaises((TypeError, ValueError)):
                    forecast_command(outputs=outputs)

    def test_market_default_keys_replay_and_explicit_keys_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            optional_prices = market_command(
                selected_decimal_price=None,
                opposing_decimal_price=None,
            )

            first = store.append_market_observation(optional_prices)
            replayed = store.append_market_observation(optional_prices)

            self.assertEqual(replayed, first)
            self.assertIsNone(first.selected_decimal_price)
            self.assertIsNone(first.opposing_decimal_price)

            explicit = store.append_market_observation(
                market_command(
                    source_revision="2",
                    selected_decimal_price=1.88,
                ),
                idempotency_key="market:manual-key",
            )
            self.assertEqual(explicit.envelope.idempotency_key, "market:manual-key")
            with self.assertRaises(IdempotencyConflict):
                store.append_market_observation(
                    market_command(
                        source_revision="2",
                        selected_decimal_price=1.87,
                    ),
                    idempotency_key="market:manual-key",
                )
            self.assertEqual(len(tuple(store.iter_records("market_observation"))), 2)

    def test_market_match_requires_entry_close_and_matching_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            entry = store.append_market_observation(
                market_command(line=1.5, market_type="plus_1_5")
            )
            close = store.append_market_observation(
                market_command(
                    source_revision="2",
                    observation_kind="close",
                    line=1.5,
                    market_type="plus_1_5",
                    observed_at=datetime(2026, 7, 14, 22, 30, tzinfo=UTC),
                    available_at=datetime(2026, 7, 14, 22, 31, tzinfo=UTC),
                )
            )

            self.assertTrue(store.market_observations_match(entry, close))
            self.assertFalse(store.market_observations_match(close, entry))

            mismatches = (
                {"line": 2.5},
                {"selected_side": "MIN"},
                {"market_type": "ml"},
                {"settlement_format": "three_way"},
                {"game_pk": 900002},
            )
            for index, overrides in enumerate(mismatches, start=3):
                with self.subTest(overrides=overrides):
                    candidate_values = {
                        "source_revision": str(index),
                        "observation_kind": "close",
                        "line": 1.5,
                        "market_type": "plus_1_5",
                    }
                    candidate_values.update(overrides)
                    candidate = store.append_market_observation(
                        market_command(**candidate_values)
                    )
                    self.assertFalse(
                        store.market_observations_match(entry, candidate)
                    )

    def test_result_correction_supersedes_latest_without_mutating_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_official_result(result_command())
            correction = store.append_official_result(
                result_command(
                    source_revision="2",
                    away_full_runs=5,
                    component_values={
                        "away_full_runs": 5.0,
                        "home_full_runs": 3.0,
                    },
                    documented_events=[{"event": "scoring_correction"}],
                    supersedes_record_id=original.envelope.record_id,
                )
            )

            self.assertEqual(original.away_full_runs, 4)
            self.assertIsNone(original.supersedes_record_id)
            self.assertEqual(correction.away_full_runs, 5)
            self.assertEqual(
                correction.supersedes_record_id,
                original.envelope.record_id,
            )
            self.assertEqual(store.latest_official_result(900001), correction)
            superseded = store.records_superseded_by("official_result")
            self.assertEqual(
                superseded[original.envelope.record_id],
                correction.envelope.record_id,
            )
            with self.assertRaises(TypeError):
                superseded[original.envelope.record_id] = "mutated"

    def test_result_correction_requires_exact_latest_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            with self.assertRaises(LedgerReferenceError):
                store.append_official_result(
                    result_command(supersedes_record_id="missing-result")
                )

            original = store.append_official_result(result_command())
            with self.assertRaises(LedgerReferenceError):
                store.append_official_result(
                    result_command(source_revision="2", away_full_runs=5)
                )
            correction = store.append_official_result(
                result_command(
                    source_revision="2",
                    away_full_runs=5,
                    supersedes_record_id=original.envelope.record_id,
                )
            )

            with self.assertRaises(LedgerReferenceError):
                store.append_official_result(
                    result_command(
                        source_revision="3",
                        away_full_runs=6,
                        supersedes_record_id=original.envelope.record_id,
                    )
                )
            with self.assertRaises(LedgerReferenceError):
                store.append_official_result(
                    result_command(
                        source_revision="3",
                        away_full_runs=6,
                        supersedes_record_id="missing-result",
                    )
                )
            self.assertEqual(store.latest_official_result(900001), correction)
            self.assertIsNone(store.latest_official_result(999999))

    def test_result_component_keys_accept_fixed_and_dynamic_team_ids(self) -> None:
        fixed_keys = (
            "away_f5_runs",
            "home_f5_runs",
            "away_full_runs",
            "home_full_runs",
            "away_f5_traffic",
            "home_f5_traffic",
            "away_full_traffic",
            "home_full_traffic",
            "away_f5_extra_base_hits",
            "home_f5_extra_base_hits",
            "away_full_extra_base_hits",
            "home_full_extra_base_hits",
            "away_pitcher_chain_runs_allowed",
            "home_pitcher_chain_runs_allowed",
            "away_bullpen_runs_allowed",
            "home_bullpen_runs_allowed",
            "away_pitcher_chain_workload_outs",
            "home_pitcher_chain_workload_outs",
            "away_protected_bullpen_close_entry",
            "home_protected_bullpen_close_entry",
        )
        values = {key: float(index) for index, key in enumerate(fixed_keys)}
        values["away_bullpen_segment_bridge_exposed"] = 1.0
        values["home_bullpen_segment_high-leverage_exposed"] = 0.0

        command = result_command(component_values=values)

        self.assertEqual(set(command.component_values), set(values))
        invalid_keys = (
            "SEA_full_runs",
            "MIN_bullpen_runs_allowed",
            "runs",
            "away_arbitrary_metric",
            "away_bullpen_segment__exposed",
            "bullpen_segment_bridge_exposed",
        )
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    result_command(component_values={key: 1.0})

    def test_market_and_result_mapping_content_is_defensively_frozen(self) -> None:
        events = [{"event": "home_run", "details": ["third inning"]}]
        components = {"away_full_runs": 4.0}
        command = result_command(
            component_values=components,
            documented_events=events,
        )

        components["away_full_runs"] = 99.0
        events[0]["details"][0] = "mutated"

        self.assertEqual(command.component_values["away_full_runs"], 4.0)
        self.assertEqual(
            command.documented_events[0]["details"][0],
            "third inning",
        )
        with self.assertRaises(TypeError):
            command.component_values["away_full_runs"] = 5.0
        with self.assertRaises(TypeError):
            command.documented_events[0]["event"] = "mutated"

    def test_reopen_deserializes_all_current_record_kinds_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(forecast_command())
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            market = store.append_market_observation(market_command())
            result = store.append_official_result(result_command())

            reopened = LedgerStore(path)
            records = tuple(reopened.iter_records())

            self.assertEqual(
                tuple(type(record) for record in records),
                (
                    CaptureRecord,
                    ForecastRecord,
                    CaptureForecastLink,
                    MarketObservationRecord,
                    OfficialResultRecord,
                ),
            )
            self.assertEqual(reopened.forecast_for_link(link.envelope.record_id), forecast)
            self.assertEqual(reopened.latest_official_result(900001), result)
            self.assertEqual(tuple(reopened.iter_records("market_observation")), (market,))

    def test_lock_is_separate_and_rows_are_canonical_newline_delimited_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "ledger.jsonl"
            store = LedgerStore(path)
            self.assertTrue(path.with_name("ledger.jsonl.lock").is_file())
            self.assertFalse(path.exists())

            store.append_capture("capture:1", capture_command())
            store.append_forecast(forecast_command())

            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(raw.count(b"\n"), 2)
            lines = raw.decode("utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                self.assertEqual(line, canonical_json(json.loads(line)))

    def test_review_replays_with_generated_identity_and_matching_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            _, _, link, _, _, command = append_review_context(store)

            first = store.append_review(command)
            replayed = LedgerStore(path).append_review(command)

            self.assertEqual(replayed, first)
            self.assertTrue(first.review_id)
            self.assertEqual(
                first.capture_forecast_link_record_id,
                link.envelope.record_id,
            )
            self.assertEqual(len(tuple(store.iter_records("review"))), 1)

    def test_review_requires_existing_matching_link_result_and_market(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture, forecast, link, market, result, command = append_review_context(store)
            invalid_commands = (
                review_command(
                    capture_forecast_link_record_id="missing-link",
                    capture_id=capture.capture_id,
                    forecast_run_id=forecast.forecast_run_id,
                    official_result_record_id=result.envelope.record_id,
                    market_key=market.market_key,
                ),
                review_command(
                    capture_forecast_link_record_id=link.envelope.record_id,
                    capture_id="wrong-capture",
                    forecast_run_id=forecast.forecast_run_id,
                    official_result_record_id=result.envelope.record_id,
                    market_key=market.market_key,
                ),
                review_command(
                    capture_forecast_link_record_id=link.envelope.record_id,
                    capture_id=capture.capture_id,
                    forecast_run_id="wrong-forecast",
                    official_result_record_id=result.envelope.record_id,
                    market_key=market.market_key,
                ),
                review_command(
                    capture_forecast_link_record_id=link.envelope.record_id,
                    capture_id=capture.capture_id,
                    forecast_run_id=forecast.forecast_run_id,
                    official_result_record_id="missing-result",
                    market_key=market.market_key,
                ),
                review_command(
                    capture_forecast_link_record_id=link.envelope.record_id,
                    capture_id=capture.capture_id,
                    forecast_run_id=forecast.forecast_run_id,
                    official_result_record_id=result.envelope.record_id,
                    market_key="missing-market",
                ),
            )

            for index, invalid in enumerate(invalid_commands, start=1):
                with self.subTest(index=index):
                    with self.assertRaises(LedgerReferenceError):
                        store.append_review(
                            invalid,
                            idempotency_key=f"review:invalid:{index}",
                        )

            valid = store.append_review(command)
            self.assertEqual(valid.capture_id, capture.capture_id)

    def test_review_rejects_cross_game_result_and_market_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:review", capture_command())
            forecast = store.append_forecast(forecast_command())
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            other_market = store.append_market_observation(
                market_command(game_pk=900002, market_key="game:900002:ml:SEA")
            )
            other_result = store.append_official_result(
                result_command(game_pk=900002, source_record="game/900002/boxscore")
            )

            with self.assertRaises(LedgerReferenceError):
                store.append_review(
                    review_command(
                        capture_forecast_link_record_id=link.envelope.record_id,
                        capture_id=capture.capture_id,
                        forecast_run_id=forecast.forecast_run_id,
                        official_result_record_id=other_result.envelope.record_id,
                        market_key=other_market.market_key,
                    )
                )

            same_game_result = store.append_official_result(result_command())
            with self.assertRaises(LedgerReferenceError):
                store.append_review(
                    review_command(
                        capture_forecast_link_record_id=link.envelope.record_id,
                        capture_id=capture.capture_id,
                        forecast_run_id=forecast.forecast_run_id,
                        official_result_record_id=same_game_result.envelope.record_id,
                        market_key=other_market.market_key,
                    ),
                    idempotency_key="review:wrong-market-game",
                )

    def test_review_explicit_key_conflicts_and_nested_fields_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            _, _, _, _, _, command = append_review_context(store)
            review = store.append_review(command, idempotency_key="review:manual")

            with self.assertRaises(IdempotencyConflict):
                store.append_review(
                    review_command(
                        capture_forecast_link_record_id=(
                            command.capture_forecast_link_record_id
                        ),
                        capture_id=command.capture_id,
                        forecast_run_id=command.forecast_run_id,
                        official_result_record_id=command.official_result_record_id,
                        market_key=command.market_key,
                        classification="outside_80",
                    ),
                    idempotency_key="review:manual",
                )
            with self.assertRaises(TypeError):
                review.component_reviews[0]["new"] = {}
            with self.assertRaises(TypeError):
                review.component_reviews[0]["classification"] = "mutated"

    def test_result_default_replay_and_explicit_key_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            first = store.append_official_result(result_command())
            self.assertEqual(store.append_official_result(result_command()), first)

        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            store.append_official_result(
                result_command(),
                idempotency_key="result:manual",
            )
            with self.assertRaises(IdempotencyConflict):
                store.append_official_result(
                    result_command(away_full_runs=5),
                    idempotency_key="result:manual",
                )

    def test_explicit_idempotency_keys_must_not_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            with self.assertRaises(ValueError):
                store.append_capture("", capture_command())
            with self.assertRaises(ValueError):
                store.append_market_observation(market_command(), idempotency_key="")
            with self.assertRaises(ValueError):
                store.append_official_result(result_command(), idempotency_key="")

    def test_validation_rejects_bad_datetimes_numbers_ids_and_json_content(self) -> None:
        naive = datetime(2026, 7, 14, 1, 30)
        invalid_factories = (
            lambda: capture_command(captured_at=naive),
            lambda: capture_command(normalized_input_hash=""),
            lambda: market_command(observation_kind="snapshot"),
            lambda: market_command(line=float("nan")),
            lambda: market_command(selected_decimal_price=0.0),
            lambda: market_command(available_at=naive),
            lambda: result_command(away_full_runs=-1),
            lambda: result_command(
                component_values={"away_full_runs": float("inf")}
            ),
            lambda: result_command(documented_events=[object()]),
            lambda: result_command(documented_events=["not-a-mapping"]),
            lambda: review_command(inside_80_ratio=-0.1),
            lambda: review_command(inside_80_ratio=1.1),
            lambda: review_command(inside_80_ratio=float("nan")),
            lambda: review_command(contradicted_paths=[1]),
            lambda: review_command(component_reviews=[object()]),
        )

        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises((TypeError, ValueError)):
                    factory()

    def test_all_public_commands_and_records_are_frozen_dataclasses(self) -> None:
        public_types = (
            RecordEnvelope,
            CaptureCommand,
            CaptureRecord,
            ForecastCommand,
            ForecastRecord,
            CaptureForecastLink,
            MarketObservationCommand,
            MarketObservationRecord,
            OfficialResultCommand,
            OfficialResultRecord,
            ReviewCommand,
            ReviewRecord,
        )

        for dataclass_type in public_types:
            with self.subTest(dataclass_type=dataclass_type.__name__):
                self.assertTrue(is_dataclass(dataclass_type))
                self.assertTrue(dataclass_type.__dataclass_params__.frozen)

    def test_supersession_helpers_survive_reopen_for_capture_and_market(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            capture = store.append_capture("capture:1", capture_command())
            replacement = store.append_capture(
                "capture:2",
                capture_command(
                    payload={"revision": 2},
                    supersedes_capture_id=capture.capture_id,
                ),
            )
            entry = store.append_market_observation(market_command())
            corrected_entry = store.append_market_observation(
                market_command(
                    source_revision="2",
                    selected_decimal_price=1.89,
                    supersedes_record_id=entry.envelope.record_id,
                )
            )

            reopened = LedgerStore(path)
            self.assertEqual(
                reopened.records_superseded_by("capture")[
                    capture.envelope.record_id
                ],
                replacement.envelope.record_id,
            )
            self.assertEqual(
                reopened.records_superseded_by("market_observation")[
                    entry.envelope.record_id
                ],
                corrected_entry.envelope.record_id,
            )

    def test_malformed_unknown_and_tampered_rows_raise_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed_path = Path(directory) / "malformed.jsonl"
            malformed_path.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaises(LedgerCorruptionError):
                tuple(LedgerStore(malformed_path).iter_records())

        with tempfile.TemporaryDirectory() as directory:
            unknown_path = Path(directory) / "unknown.jsonl"
            LedgerStore(unknown_path).append_capture("capture:1", capture_command())
            unknown_row = json.loads(unknown_path.read_text(encoding="utf-8"))
            unknown_row["envelope"]["record_kind"] = "unknown_kind"
            unknown_path.write_text(
                f"{canonical_json(unknown_row)}\n",
                encoding="utf-8",
            )
            with self.assertRaises(LedgerCorruptionError):
                tuple(LedgerStore(unknown_path).iter_records())

        with tempfile.TemporaryDirectory() as directory:
            tampered_path = Path(directory) / "tampered.jsonl"
            LedgerStore(tampered_path).append_capture("capture:1", capture_command())
            tampered_row = json.loads(tampered_path.read_text(encoding="utf-8"))
            tampered_row["envelope"]["payload_hash"] = "0" * 64
            tampered_path.write_text(
                f"{canonical_json(tampered_row)}\n",
                encoding="utf-8",
            )
            with self.assertRaises(LedgerCorruptionError):
                tuple(LedgerStore(tampered_path).iter_records())

    def test_optional_availability_and_classification_remain_canonical(self) -> None:
        market = market_command(available_at=None)
        review = review_command(classification=None)

        self.assertIsNone(market.available_at)
        self.assertIsNone(review.classification)
        self.assertIsInstance(review.component_reviews, tuple)
        self.assertEqual(
            review.component_reviews[0]["evidence_path"],
            "away_lineup.power",
        )

    def test_concurrent_same_key_retries_append_exactly_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(
                    target=append_same_capture_in_process,
                    args=(str(path),),
                )
                for _ in range(8)
            ]

            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)

            self.assertTrue(all(process.exitcode == 0 for process in processes))
            records = tuple(LedgerStore(path).iter_records("capture"))
            self.assertEqual(len(records), 1)
            self.assertEqual(path.read_bytes().count(b"\n"), 1)

    def test_invalid_utf8_and_duplicate_json_keys_raise_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-utf8.jsonl"
            path.write_bytes(b"\xff\n")
            with self.assertRaises(LedgerCorruptionError):
                tuple(LedgerStore(path).iter_records())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-key.jsonl"
            LedgerStore(path).append_capture("capture:1", capture_command())
            valid_row = path.read_text(encoding="utf-8")
            duplicate_row = valid_row.replace(
                '"game_pk":900001',
                '"game_pk":900001,"game_pk":900001',
            )
            self.assertNotEqual(duplicate_row, valid_row)
            path.write_text(duplicate_row, encoding="utf-8")
            with self.assertRaises(LedgerCorruptionError):
                tuple(LedgerStore(path).iter_records())

    def test_cyclic_json_content_is_rejected_cleanly(self) -> None:
        cyclic_mapping: dict[str, object] = {}
        cyclic_mapping["self"] = cyclic_mapping
        cyclic_sequence: list[object] = []
        cyclic_sequence.append(cyclic_sequence)

        with self.assertRaisesRegex(ValueError, "cyclic"):
            capture_command(payload=cyclic_mapping)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            forecast_command(outputs={"cycle": cyclic_sequence})


if __name__ == "__main__":
    unittest.main()
