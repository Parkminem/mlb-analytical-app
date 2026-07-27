from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid

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
    LedgerActiveCaptureHeadChangedError,
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerMarketHeadChangedError,
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
        "payload": {
            "away_team": "SEA",
            "home_team": "MIN",
            "markets": [
                {
                    "market_key": "game:900001:ml:SEA",
                    "market_type": "ml",
                    "settlement_format": "two_way",
                    "selected_side": "away",
                    "line": None,
                    "selected_decimal_price": 1.91,
                    "opposing_decimal_price": 1.95,
                }
            ],
        },
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
            "market_inputs": {
                "game:900001:ml:SEA": {
                    "market_key": "game:900001:ml:SEA",
                    "market_type": "ml",
                    "settlement_format": "two_way",
                    "selected_side": "away",
                    "line": None,
                    "selected_decimal_price": 1.91,
                    "opposing_decimal_price": 1.95,
                }
            },
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
        "classification": "A",
        "inside_80_ratio": 0.75,
        "contradicted_paths": [],
        "component_reviews": [
            {
                "evidence_path": "starter_suppression",
                "component_ids": ["home_pitcher_chain_runs_allowed"],
                "contradicted": False,
                "inside_80_count": 1,
                "component_count": 1,
                "notes": [],
            },
            {
                "evidence_path": "top_order_matchup",
                "component_ids": [
                    "away_full_runs",
                    "away_full_traffic",
                    "away_full_extra_base_hits",
                ],
                "contradicted": False,
                "inside_80_count": 2,
                "component_count": 3,
                "notes": [],
            },
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


def _logical_payload_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def append_forged_record(
    path: Path,
    record_kind: str,
    idempotency_key: str,
    logical_payload: object,
    **record_fields: object,
) -> None:
    row = json.loads(canonical_json(logical_payload))
    if not isinstance(row, dict):
        raise AssertionError("forged logical payload must serialize as an object")
    row.update(record_fields)
    row["envelope"] = {
        "record_id": uuid.uuid4().hex,
        "record_kind": record_kind,
        "idempotency_key": idempotency_key,
        "payload_hash": _logical_payload_hash(logical_payload),
        "appended_at": CAPTURED_AT.isoformat(),
    }
    with path.open("a", encoding="utf-8", newline="") as ledger:
        ledger.write(f"{canonical_json(row)}\n")


def write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as ledger:
        for row in rows:
            ledger.write(f"{canonical_json(row)}\n")


def append_same_capture_in_process(path: str) -> None:
    LedgerStore(path).append_capture(
        "capture:concurrent",
        capture_command(payload={"source": "concurrent"}),
    )


def append_same_capture_and_report_in_process(path: str, result_queue) -> None:
    record = LedgerStore(path).append_capture(
        "capture:concurrent-alias",
        capture_command(payload={"source": "concurrent-alias"}),
    )
    result_queue.put(record.capture_id)


class LedgerStoreTests(unittest.TestCase):
    def _assert_unicode_separator_reopens_and_replays(
        self,
        separator: str,
        idempotency_key: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            command = capture_command(
                payload={"note": f"before{separator}after"}
            )
            first = LedgerStore(path).append_capture(idempotency_key, command)
            raw_after_first = path.read_bytes()

            reopened = LedgerStore(path)
            records = tuple(reopened.iter_records("capture"))
            replayed = reopened.append_capture(idempotency_key, command)

            self.assertEqual(raw_after_first.count(b"\n"), 1)
            self.assertIn(separator.encode("utf-8"), raw_after_first)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].payload["note"], f"before{separator}after")
            self.assertEqual(replayed.capture_id, first.capture_id)
            self.assertEqual(path.read_bytes(), raw_after_first)

    def test_u2028_payload_reopens_and_replays_as_one_jsonl_row(self) -> None:
        self._assert_unicode_separator_reopens_and_replays(
            "\u2028",
            "capture:unicode:u2028",
        )

    def test_u2029_payload_reopens_and_replays_as_one_jsonl_row(self) -> None:
        self._assert_unicode_separator_reopens_and_replays(
            "\u2029",
            "capture:unicode:u2029",
        )

    def test_jsonl_framing_rejects_missing_final_lf_and_empty_lf_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            LedgerStore(path).append_capture("capture:1", capture_command())
            valid_row = path.read_bytes()

            path.write_bytes(valid_row[:-1])
            with self.assertRaisesRegex(LedgerCorruptionError, "end with a newline"):
                tuple(LedgerStore(path).iter_records())

            path.write_bytes(valid_row + b"\n")
            with self.assertRaisesRegex(LedgerCorruptionError, "empty ledger row"):
                tuple(LedgerStore(path).iter_records())

    def test_jsonl_framing_rejects_raw_cr_terminator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            LedgerStore(path).append_capture("capture:cr", capture_command())
            valid_row = path.read_bytes()

            path.write_bytes(valid_row[:-1] + b"\r")

            with self.assertRaisesRegex(LedgerCorruptionError, "ASCII LF"):
                tuple(LedgerStore(path).iter_records())

    def test_jsonl_framing_rejects_crlf_terminator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            LedgerStore(path).append_capture("capture:crlf", capture_command())
            valid_row = path.read_bytes()

            path.write_bytes(valid_row[:-1] + b"\r\n")

            with self.assertRaisesRegex(LedgerCorruptionError, "ASCII LF"):
                tuple(LedgerStore(path).iter_records())

    def test_escaped_carriage_return_payload_reopens_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            command = capture_command(payload={"note": "before\rafter"})
            first = LedgerStore(path).append_capture("capture:escaped-cr", command)
            raw_after_first = path.read_bytes()

            reopened = LedgerStore(path)
            records = tuple(reopened.iter_records("capture"))
            replayed = reopened.append_capture("capture:escaped-cr", command)

            self.assertIn(b"before\\rafter", raw_after_first)
            self.assertNotIn(b"\r", raw_after_first)
            self.assertEqual(records[0].payload["note"], "before\rafter")
            self.assertEqual(replayed.capture_id, first.capture_id)
            self.assertEqual(path.read_bytes(), raw_after_first)

    def test_append_orders_lock_rebuild_write_flush_fsync_and_unlock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            events: list[str] = []
            real_flock = fcntl.flock
            real_fsync = os.fsync
            real_read_records = LedgerStore._read_records
            real_fdopen = os.fdopen

            def flock_spy(file_descriptor: int, operation: int) -> None:
                real_flock(file_descriptor, operation)
                if operation == fcntl.LOCK_EX:
                    events.append("lock_ex")
                elif operation == fcntl.LOCK_UN:
                    events.append("lock_un")

            def rebuild_spy(
                instance: LedgerStore,
                descriptor: int | None = None,
            ):
                records = real_read_records(instance, descriptor)
                events.append("rebuild")
                return records

            def fsync_spy(file_descriptor: int) -> None:
                real_fsync(file_descriptor)
                events.append("fsync")

            def fdopen_spy(file_descriptor: int, *args: object, **kwargs: object):
                handle = real_fdopen(file_descriptor, *args, **kwargs)
                mode = args[0] if args else kwargs.get("mode", "r")
                if mode != "a":
                    return handle

                wrapped = mock.MagicMock(wraps=handle)

                def enter_spy():
                    handle.__enter__()
                    return wrapped

                def write_spy(value: str) -> int:
                    written = handle.write(value)
                    events.append("write")
                    return written

                def flush_spy() -> None:
                    handle.flush()
                    events.append("flush")

                wrapped.__enter__.side_effect = enter_spy
                wrapped.__exit__.side_effect = handle.__exit__
                wrapped.write.side_effect = write_spy
                wrapped.flush.side_effect = flush_spy
                wrapped.fileno.side_effect = handle.fileno
                return wrapped

            with (
                mock.patch(
                    "analysis.ledger.store.fcntl.flock",
                    side_effect=flock_spy,
                ),
                mock.patch.object(
                    LedgerStore,
                    "_read_records",
                    autospec=True,
                    side_effect=rebuild_spy,
                ),
                mock.patch(
                    "analysis.ledger.store.os.fsync",
                    side_effect=fsync_spy,
                ),
                mock.patch(
                    "analysis.ledger.store.os.fdopen",
                    side_effect=fdopen_spy,
                ),
            ):
                store.append_capture("capture:ordered", capture_command())

            self.assertNotEqual(store.lock_path, store.path)
            self.assertTrue(store.lock_path.is_file())
            self.assertEqual(
                events,
                [
                    "lock_ex",
                    "lock_ex",
                    "rebuild",
                    "write",
                    "flush",
                    "fsync",
                    "lock_un",
                    "lock_un",
                ],
            )

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

    def test_capture_append_without_expected_head_keeps_legacy_behavior(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")

            first = store.append_capture("capture:legacy:1", capture_command())
            second = store.append_capture(
                "capture:legacy:2",
                capture_command(
                    captured_at=CAPTURED_AT + timedelta(minutes=1),
                    payload={"revision": 2},
                ),
            )

            self.assertIsNone(first.supersedes_capture_id)
            self.assertIsNone(second.supersedes_capture_id)
            self.assertEqual(len(tuple(store.iter_records("capture"))), 2)

    def test_capture_append_expected_head_is_checked_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            first = store.append_capture("capture:head:1", capture_command())

            with self.assertRaisesRegex(
                LedgerHeadChangedError,
                first.capture_id,
            ):
                store.append_capture(
                    "capture:head:2",
                    capture_command(
                        captured_at=CAPTURED_AT + timedelta(minutes=1),
                        payload={"revision": 2},
                    ),
                    expected_active_capture_id=None,
                )

            self.assertEqual(len(tuple(store.iter_records("capture"))), 1)

    def test_exact_capture_replay_returns_before_expected_head_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            command = capture_command()
            original = store.append_capture("capture:replay", command)
            store.append_capture(
                "capture:new-head",
                capture_command(
                    captured_at=CAPTURED_AT + timedelta(minutes=1),
                    payload={"revision": 2},
                ),
            )

            replay = store.append_capture(
                "capture:replay",
                command,
                expected_active_capture_id=None,
            )

            self.assertEqual(replay, original)
            self.assertEqual(len(tuple(store.iter_records("capture"))), 2)

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

    def test_forecast_append_changed_shadow_comparison_still_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            first = store.append_forecast(
                forecast_command(
                    outputs={
                        "selected_probability": 0.55,
                        "shadow_comparison": {"note": "original"},
                    }
                )
            )

            with self.assertRaises(IdempotencyConflict):
                store.append_forecast(
                    forecast_command(
                        outputs={
                            "selected_probability": 0.55,
                            "shadow_comparison": {"note": "corrected"},
                        }
                    )
                )

            self.assertEqual(
                tuple(store.iter_records("forecast")),
                (first,),
            )

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

    def test_link_rejects_mismatched_normalized_input_hash_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(
                forecast_command(normalized_input_hash="different-input-hash")
            )

            with self.assertRaisesRegex(
                LedgerReferenceError,
                "normalized_input_hash",
            ):
                store.link_capture_forecast(
                    capture.capture_id,
                    forecast.forecast_run_id,
                )

            self.assertEqual(
                tuple(store.iter_records("capture_forecast_link")),
                (),
            )

    def test_mismatched_link_is_corruption_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(
                forecast_command(normalized_input_hash="different-input-hash")
            )
            logical_payload = {
                "capture_id": capture.capture_id,
                "forecast_run_id": forecast.forecast_run_id,
            }
            append_forged_record(
                path,
                "capture_forecast_link",
                f"capture-forecast:{capture.capture_id}:{forecast.forecast_run_id}",
                logical_payload,
            )

            with self.assertRaisesRegex(
                LedgerCorruptionError,
                "normalized_input_hash",
            ):
                tuple(LedgerStore(path).iter_records())

    def test_multiple_active_link_roots_are_corruption_on_strict_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(forecast_command())
            store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            logical_payload = {
                "capture_id": capture.capture_id,
                "forecast_run_id": forecast.forecast_run_id,
            }
            append_forged_record(
                path,
                "capture_forecast_link",
                "capture-forecast:legacy-second-root",
                logical_payload,
            )

            with self.assertRaisesRegex(
                LedgerCorruptionError,
                "one active|multiple active",
            ):
                tuple(LedgerStore(path).iter_records())

    def test_link_correction_chain_remains_valid_on_strict_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            capture = store.append_capture("capture:1", capture_command())
            forecast = store.append_forecast(forecast_command())
            original = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            logical_payload = {
                "capture_id": capture.capture_id,
                "forecast_run_id": forecast.forecast_run_id,
                "shadow_comparison": {"notes": ["corrected"]},
                "supersedes_link_record_id": original.envelope.record_id,
            }
            append_forged_record(
                path,
                "capture_forecast_link",
                "capture-forecast:legacy-correction",
                logical_payload,
                shadow_comparison={"notes": ["corrected"]},
                supersedes_link_record_id=original.envelope.record_id,
            )

            links = tuple(
                LedgerStore(path).iter_records("capture_forecast_link")
            )

            self.assertEqual(len(links), 2)
            self.assertEqual(
                links[-1].supersedes_link_record_id,
                original.envelope.record_id,
            )

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

    def test_capture_supersession_rejects_direct_branch_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_capture("capture:1", capture_command())
            store.append_capture(
                "capture:2",
                capture_command(
                    payload={"revision": 2},
                    supersedes_capture_id=original.capture_id,
                ),
            )

            with self.assertRaisesRegex(LedgerReferenceError, "successor"):
                store.append_capture(
                    "capture:branch",
                    capture_command(
                        payload={"revision": "branch"},
                        supersedes_capture_id=original.capture_id,
                    ),
                )

    def test_capture_supersession_branch_is_corruption_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            original = store.append_capture("capture:1", capture_command())
            store.append_capture(
                "capture:2",
                capture_command(
                    payload={"revision": 2},
                    supersedes_capture_id=original.capture_id,
                ),
            )
            branch = capture_command(
                payload={"revision": "forged-branch"},
                supersedes_capture_id=original.capture_id,
            )
            append_forged_record(
                path,
                "capture",
                "capture:forged-branch",
                branch,
                capture_id=uuid.uuid4().hex,
            )

            with self.assertRaisesRegex(LedgerCorruptionError, "successor"):
                tuple(LedgerStore(path).iter_records())

    def test_capture_supersession_allows_linear_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            original = store.append_capture("capture:1", capture_command())
            second = store.append_capture(
                "capture:2",
                capture_command(
                    payload={"revision": 2},
                    supersedes_capture_id=original.capture_id,
                ),
            )
            third = store.append_capture(
                "capture:3",
                capture_command(
                    payload={"revision": 3},
                    supersedes_capture_id=second.capture_id,
                ),
            )

            relationships = LedgerStore(path).records_superseded_by("capture")
            self.assertEqual(
                relationships[original.envelope.record_id],
                second.envelope.record_id,
            )
            self.assertEqual(
                relationships[second.envelope.record_id],
                third.envelope.record_id,
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

    def test_frozen_mapping_equality_preserves_nested_json_scalar_types(self) -> None:
        boolean_mapping = capture_command(
            payload={"nested": {"values": [True, {"value": False}]}}
        ).payload
        numeric_mapping = capture_command(
            payload={"nested": {"values": [1, {"value": 0}]}}
        ).payload
        same_boolean_mapping = capture_command(
            payload={"nested": {"values": [True, {"value": False}]}}
        ).payload

        self.assertNotEqual(boolean_mapping, numeric_mapping)
        self.assertNotEqual(
            boolean_mapping["nested"],
            numeric_mapping["nested"],
        )
        self.assertNotEqual(
            boolean_mapping,
            {"nested": {"values": [1, {"value": 0}]}},
        )
        self.assertEqual(boolean_mapping, same_boolean_mapping)

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

    def test_market_supersession_rejects_direct_branch_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_market_observation(market_command())
            store.append_market_observation(
                market_command(
                    source_revision="2",
                    selected_decimal_price=1.90,
                    supersedes_record_id=original.envelope.record_id,
                )
            )

            with self.assertRaisesRegex(LedgerReferenceError, "successor"):
                store.append_market_observation(
                    market_command(
                        source_revision="branch",
                        selected_decimal_price=1.89,
                        supersedes_record_id=original.envelope.record_id,
                    )
                )

    def test_market_supersession_branch_is_corruption_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            original = store.append_market_observation(market_command())
            store.append_market_observation(
                market_command(
                    source_revision="2",
                    selected_decimal_price=1.90,
                    supersedes_record_id=original.envelope.record_id,
                )
            )
            branch = market_command(
                source_revision="forged-branch",
                selected_decimal_price=1.89,
                supersedes_record_id=original.envelope.record_id,
            )
            append_forged_record(
                path,
                "market_observation",
                "market:forged-branch",
                branch,
            )

            with self.assertRaisesRegex(LedgerCorruptionError, "successor"):
                tuple(LedgerStore(path).iter_records())

    def test_market_supersession_allows_linear_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            original = store.append_market_observation(market_command())
            second = store.append_market_observation(
                market_command(
                    source_revision="2",
                    selected_decimal_price=1.90,
                    supersedes_record_id=original.envelope.record_id,
                )
            )
            third = store.append_market_observation(
                market_command(
                    source_revision="3",
                    selected_decimal_price=1.89,
                    supersedes_record_id=second.envelope.record_id,
                )
            )

            relationships = LedgerStore(path).records_superseded_by(
                "market_observation"
            )
            self.assertEqual(
                relationships[original.envelope.record_id],
                second.envelope.record_id,
            )
            self.assertEqual(
                relationships[second.envelope.record_id],
                third.envelope.record_id,
            )

    def test_market_expected_latest_cas_detects_changed_stream_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_market_observation(market_command())
            correction_command = market_command(
                source_revision="2",
                selected_decimal_price=1.90,
                supersedes_record_id=original.envelope.record_id,
            )

            with self.assertRaises(LedgerMarketHeadChangedError):
                store.append_market_observation(
                    market_command(source_revision="unexpected-root"),
                    expected_latest_record_id=None,
                )

            correction = store.append_market_observation(
                correction_command,
                expected_latest_record_id=original.envelope.record_id,
            )
            replayed = store.append_market_observation(
                correction_command,
                expected_latest_record_id=None,
            )

            self.assertEqual(replayed, correction)
            self.assertEqual(
                tuple(store.iter_records("market_observation")),
                (original, correction),
            )

    def test_market_expected_latest_is_optional_for_existing_callers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            original = store.append_market_observation(market_command())

            parallel = store.append_market_observation(
                market_command(
                    source_revision="parallel-root",
                )
            )

            self.assertIsNone(parallel.supersedes_record_id)
            self.assertNotEqual(
                parallel.envelope.record_id,
                original.envelope.record_id,
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

    def test_official_result_rejects_tied_full_game_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "winner"):
            result_command(away_full_runs=4, home_full_runs=4)

    def test_result_source_revision_is_unique_across_custom_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            store.append_official_result(
                result_command(),
                idempotency_key="result:manual:first",
            )

            with self.assertRaisesRegex(
                LedgerReferenceError,
                "source revision",
            ):
                store.append_official_result(
                    result_command(),
                    idempotency_key="result:manual:duplicate",
                )

            self.assertEqual(len(tuple(store.iter_records("official_result"))), 1)

    def test_result_active_capture_is_checked_before_exact_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:active:result", capture_command())
            command = result_command()
            original = store.append_official_result(
                command,
                required_active_capture_id=capture.capture_id,
            )
            replacement = store.append_capture(
                "capture:active:result:replacement",
                capture_command(
                    captured_at=CAPTURED_AT + timedelta(minutes=1),
                    payload={"revision": 2},
                    supersedes_capture_id=capture.capture_id,
                ),
                expected_active_capture_id=capture.capture_id,
            )

            with self.assertRaisesRegex(
                LedgerActiveCaptureHeadChangedError,
                replacement.capture_id,
            ):
                store.append_official_result(
                    command,
                    required_active_capture_id=capture.capture_id,
                )

            self.assertEqual(
                tuple(store.iter_records("official_result")),
                (original,),
            )

    def test_duplicate_result_source_revision_is_corruption_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            original = store.append_official_result(result_command())
            duplicate = result_command(
                supersedes_record_id=original.envelope.record_id,
            )
            append_forged_record(
                path,
                "official_result",
                "result:forged-duplicate-revision",
                duplicate,
            )

            with self.assertRaisesRegex(
                LedgerCorruptionError,
                "source revision",
            ):
                tuple(LedgerStore(path).iter_records())

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
        values.update(
            away_f5_runs=2.0,
            home_f5_runs=1.0,
            away_full_runs=4.0,
            home_full_runs=3.0,
            away_pitcher_chain_runs_allowed=2.0,
            away_bullpen_runs_allowed=1.0,
            home_pitcher_chain_runs_allowed=3.0,
            home_bullpen_runs_allowed=1.0,
        )
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

    def test_official_score_components_match_explicit_scores_when_present(
        self,
    ) -> None:
        consistent = {
            "away_f5_runs": 2.0,
            "home_f5_runs": 1.0,
            "away_full_runs": 4.0,
            "home_full_runs": 3.0,
        }

        command = result_command(component_values=consistent)
        missing_optional = result_command(component_values={})

        self.assertEqual(dict(command.component_values), consistent)
        self.assertEqual(dict(missing_optional.component_values), {})
        for component_id in consistent:
            tampered = dict(consistent)
            tampered[component_id] = 20.0
            with self.subTest(component_id=component_id):
                with self.assertRaisesRegex(ValueError, component_id):
                    result_command(component_values=tampered)

        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            record = store.append_official_result(command)
            with self.assertRaisesRegex(ValueError, "away_full_runs"):
                replace(
                    record,
                    component_values={"away_full_runs": 3.0},
                )

    def test_pitching_run_components_must_conserve_opponent_full_score(
        self,
    ) -> None:
        consistent = {
            "away_pitcher_chain_runs_allowed": 2.0,
            "away_bullpen_runs_allowed": 1.0,
            "home_pitcher_chain_runs_allowed": 3.0,
            "home_bullpen_runs_allowed": 1.0,
        }

        command = result_command(component_values=consistent)
        self.assertEqual(dict(command.component_values), consistent)

        for component_id in (
            "away_pitcher_chain_runs_allowed",
            "home_bullpen_runs_allowed",
        ):
            tampered = dict(consistent)
            tampered[component_id] += 1.0
            with self.subTest(component_id=component_id):
                with self.assertRaisesRegex(
                    ValueError,
                    "pitcher-chain and bullpen runs allowed",
                ):
                    result_command(component_values=tampered)

        negative = dict(consistent)
        negative["away_pitcher_chain_runs_allowed"] = -1.0
        negative["away_bullpen_runs_allowed"] = 4.0
        with self.assertRaisesRegex(ValueError, "nonnegative whole-run counts"):
            result_command(component_values=negative)

        fractional = dict(consistent)
        fractional["away_pitcher_chain_runs_allowed"] = 1.5
        fractional["away_bullpen_runs_allowed"] = 1.5
        with self.assertRaisesRegex(ValueError, "nonnegative whole-run counts"):
            result_command(component_values=fractional)

    def test_single_pitching_run_component_cannot_exceed_opponent_score(
        self,
    ) -> None:
        impossible_components = {
            "away_pitcher_chain_runs_allowed": 4.0,
            "away_bullpen_runs_allowed": 4.0,
            "home_pitcher_chain_runs_allowed": 5.0,
            "home_bullpen_runs_allowed": 5.0,
        }

        for component_id, value in impossible_components.items():
            with self.subTest(component_id=component_id):
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot exceed the opponent full-game score",
                ):
                    result_command(component_values={component_id: value})

    def test_official_full_score_cannot_be_lower_than_f5_score(self) -> None:
        for side in ("away", "home"):
            with self.subTest(side=side):
                with self.assertRaisesRegex(ValueError, f"{side}_full_runs"):
                    result_command(
                        **{
                            f"{side}_f5_runs": 5,
                            f"{side}_full_runs": 4,
                        }
                    )

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

    def test_review_uses_frozen_forecast_market_without_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:review", capture_command())
            forecast = store.append_forecast(forecast_command())
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            result = store.append_official_result(result_command())
            command = review_command(
                capture_forecast_link_record_id=link.envelope.record_id,
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                official_result_record_id=result.envelope.record_id,
            )

            review = store.append_review(command)

            self.assertEqual(review.market_key, command.market_key)

    def test_review_market_horizon_comes_from_frozen_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:review", capture_command())
            forecast = store.append_forecast(forecast_command())
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            market = store.append_market_observation(
                market_command(market_type="f5_side")
            )
            result = store.append_official_result(result_command())
            component_reviews = list(review_command().component_reviews)
            component_reviews[1] = {
                **dict(component_reviews[1]),
                "component_ids": (
                    "away_f5_runs",
                    "away_f5_traffic",
                    "away_f5_extra_base_hits",
                ),
            }
            command = review_command(
                capture_forecast_link_record_id=link.envelope.record_id,
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                official_result_record_id=result.envelope.record_id,
                market_key=market.market_key,
                component_reviews=component_reviews,
            )

            with self.assertRaisesRegex(LedgerReferenceError, "market horizon"):
                store.append_review(command)

    def test_review_rejects_forecast_market_from_another_capture_game(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture = store.append_capture("capture:review", capture_command())
            foreign_key = "game:900002:ml:SEA"
            outputs = json.loads(canonical_json(forecast_command().outputs))
            market_input = outputs["market_inputs"].pop(
                "game:900001:ml:SEA"
            )
            market_input["market_key"] = foreign_key
            outputs["market_inputs"][foreign_key] = market_input
            outputs["markets"][0]["market_key"] = foreign_key
            forecast = store.append_forecast(
                forecast_command(outputs=outputs)
            )
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            result = store.append_official_result(result_command())
            command = review_command(
                capture_forecast_link_record_id=link.envelope.record_id,
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                official_result_record_id=result.envelope.record_id,
                market_key=foreign_key,
            )

            with self.assertRaisesRegex(LedgerReferenceError, "capture markets"):
                store.append_review(command)

    def test_review_rejects_forecast_type_that_disagrees_with_capture_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            f5_key = "game:900001:f5_side:SEA"
            payload = json.loads(canonical_json(capture_command().payload))
            payload["markets"][0]["market_key"] = f5_key
            payload["markets"][0]["market_type"] = "f5_side"
            capture = store.append_capture(
                "capture:review",
                capture_command(payload=payload),
            )
            outputs = json.loads(canonical_json(forecast_command().outputs))
            market_input = outputs["market_inputs"].pop(
                "game:900001:ml:SEA"
            )
            market_input["market_key"] = f5_key
            outputs["market_inputs"][f5_key] = market_input
            outputs["markets"][0]["market_key"] = f5_key
            forecast = store.append_forecast(
                forecast_command(outputs=outputs)
            )
            link = store.link_capture_forecast(
                capture.capture_id,
                forecast.forecast_run_id,
            )
            result = store.append_official_result(result_command())
            command = review_command(
                capture_forecast_link_record_id=link.envelope.record_id,
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                official_result_record_id=result.envelope.record_id,
                market_key=f5_key,
            )

            with self.assertRaisesRegex(LedgerReferenceError, "capture markets"):
                store.append_review(command)

    def test_stale_result_review_replays_but_cannot_append_under_a_new_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            _, _, _, _, result, command = append_review_context(store)
            original = store.append_review(command)
            store.append_official_result(
                result_command(
                    source_revision="2",
                    finalized_at=FINALIZED_AT + timedelta(minutes=5),
                    away_full_runs=5,
                    supersedes_record_id=result.envelope.record_id,
                )
            )

            replayed = store.append_review(command)

            self.assertEqual(replayed, original)
            with self.assertRaisesRegex(LedgerReferenceError, "latest"):
                store.append_review(
                    command,
                    idempotency_key="review:stale-result:new-key",
                )
            self.assertEqual(len(tuple(store.iter_records("review"))), 1)

    def test_review_active_capture_is_checked_before_exact_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture, _, _, _, _, command = append_review_context(store)
            original = store.append_review(
                command,
                required_active_capture_id=capture.capture_id,
            )
            replacement = store.append_capture(
                "capture:active:review:replacement",
                capture_command(
                    captured_at=CAPTURED_AT + timedelta(minutes=1),
                    payload={"revision": 2},
                    supersedes_capture_id=capture.capture_id,
                ),
                expected_active_capture_id=capture.capture_id,
            )

            with self.assertRaisesRegex(
                LedgerActiveCaptureHeadChangedError,
                replacement.capture_id,
            ):
                store.append_review(
                    command,
                    required_active_capture_id=capture.capture_id,
                )

            self.assertEqual(tuple(store.iter_records("review")), (original,))

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

    def test_review_rejects_conversion_components_from_wrong_market_horizon(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            _, _, _, _, _, command = append_review_context(store)
            component_reviews = list(command.component_reviews)
            component_reviews[1] = {
                **dict(component_reviews[1]),
                "component_ids": (
                    "away_f5_runs",
                    "away_f5_traffic",
                    "away_f5_extra_base_hits",
                ),
            }
            incompatible = replace(
                command,
                component_reviews=component_reviews,
            )

            with self.assertRaisesRegex(LedgerReferenceError, "market horizon"):
                store.append_review(incompatible)

    def test_review_rejects_same_side_suppression_and_collapse_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            _, _, _, _, _, command = append_review_context(store)
            incompatible = replace(
                command,
                inside_80_ratio=1.0,
                component_reviews=(
                    {
                        "evidence_path": "starter_suppression",
                        "component_ids": (
                            "home_pitcher_chain_runs_allowed",
                        ),
                        "contradicted": False,
                        "inside_80_count": 1,
                        "component_count": 1,
                        "notes": (),
                    },
                    {
                        "evidence_path": "starter_collapse",
                        "component_ids": (
                            "home_pitcher_chain_runs_allowed",
                            "home_pitcher_chain_workload_outs",
                        ),
                        "contradicted": False,
                        "inside_80_count": 2,
                        "component_count": 2,
                        "notes": (),
                    },
                ),
            )

            with self.assertRaisesRegex(LedgerReferenceError, "same pitching side"):
                store.append_review(incompatible)

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

    def test_review_rejects_result_finalized_before_capture_or_first_pitch(
        self,
    ) -> None:
        cases = (
            (CAPTURED_AT, "captured_at"),
            (FIRST_PITCH - timedelta(seconds=1), "first_pitch"),
        )

        for finalized_at, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    store = LedgerStore(Path(directory) / "ledger.jsonl")
                    capture = store.append_capture(
                        "capture:review-time",
                        capture_command(),
                    )
                    forecast = store.append_forecast(forecast_command())
                    link = store.link_capture_forecast(
                        capture.capture_id,
                        forecast.forecast_run_id,
                    )
                    market = store.append_market_observation(market_command())
                    result = store.append_official_result(
                        result_command(finalized_at=finalized_at)
                    )
                    command = review_command(
                        capture_forecast_link_record_id=(
                            link.envelope.record_id
                        ),
                        capture_id=capture.capture_id,
                        forecast_run_id=forecast.forecast_run_id,
                        official_result_record_id=result.envelope.record_id,
                        market_key=market.market_key,
                    )

                    with self.assertRaisesRegex(
                        LedgerReferenceError,
                        expected,
                    ):
                        store.append_review(command)

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
                        market_result="loss",
                        classification="B",
                    ),
                    idempotency_key="review:manual",
                )
            with self.assertRaises(TypeError):
                review.component_reviews[0]["new"] = {}
            with self.assertRaises(TypeError):
                review.component_reviews[0]["contradicted"] = True

    def test_review_command_rejects_invalid_results_and_classifications(self) -> None:
        invalid_overrides = (
            {"market_result": "bogus"},
            {"classification": "Z"},
            {
                "classification": "A",
                "market_result": "loss",
            },
            {
                "classification": "A",
                "inside_80_ratio": 0.74,
            },
            {
                "classification": "B",
                "inside_80_ratio": 0.59,
            },
            {
                "classification": "C",
                "inside_80_ratio": 0.60,
                "contradicted_paths": (),
            },
        )

        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    review_command(**overrides)

    def test_review_component_payload_must_match_contradicted_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "contradicted_paths"):
            review_command(
                classification="C",
                contradicted_paths=("top_order_matchup",),
            )

        with self.assertRaisesRegex(ValueError, "component_reviews"):
            review_command(
                component_reviews=(
                    {
                        "evidence_path": "starter_suppression",
                        "component_ids": ("home_pitcher_chain_runs_allowed",),
                        "contradicted": False,
                        "inside_80_count": 2,
                        "component_count": 1,
                        "notes": (),
                    },
                    {
                        "evidence_path": "top_order_matchup",
                        "component_ids": (
                            "away_full_runs",
                            "away_full_traffic",
                            "away_full_extra_base_hits",
                        ),
                        "contradicted": False,
                        "inside_80_count": 2,
                        "component_count": 3,
                        "notes": (),
                    },
                ),
            )

    def test_review_requires_canonical_two_to_four_principal_paths(self) -> None:
        one_path = review_command().component_reviews[:1]
        with self.assertRaisesRegex(ValueError, "two to four"):
            review_command(component_reviews=one_path)

        noncanonical = list(review_command().component_reviews)
        noncanonical[0] = {
            **dict(noncanonical[0]),
            "evidence_path": "away_lineup.power",
        }
        with self.assertRaisesRegex(ValueError, "canonical evidence path"):
            review_command(component_reviews=noncanonical)

    def test_review_inside_ratio_must_match_unique_component_statuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside_80_ratio"):
            review_command(inside_80_ratio=1.0)

    def test_review_component_ids_must_be_official_components(self) -> None:
        component_reviews = list(review_command().component_reviews)
        component_reviews[1] = {
            **dict(component_reviews[1]),
            "component_ids": ("invented_component",),
            "inside_80_count": 1,
            "component_count": 1,
        }

        with self.assertRaisesRegex(ValueError, "component_ids"):
            review_command(component_reviews=component_reviews)

    def test_review_component_shapes_must_match_canonical_paths(self) -> None:
        component_reviews = list(review_command().component_reviews)
        component_reviews[1] = {
            **dict(component_reviews[1]),
            "component_ids": ("away_full_runs",),
            "inside_80_count": 1,
            "component_count": 1,
        }

        with self.assertRaisesRegex(ValueError, "component shape"):
            review_command(component_reviews=component_reviews)

    def test_review_source_ids_are_canonical_and_classification_bound(self) -> None:
        review_fields = {field.name for field in fields(ReviewCommand)}
        self.assertIn(
            "qualifying_exogenous_source_observation_ids",
            review_fields,
            "ReviewCommand qualifying source IDs are missing",
        )

        source_ids = ["official:event:b", "official:event:a"]
        command = review_command(
            classification="D",
            qualifying_exogenous_source_observation_ids=source_ids,
        )
        source_ids.append("official:event:mutated")

        self.assertEqual(
            command.qualifying_exogenous_source_observation_ids,
            ("official:event:a", "official:event:b"),
        )
        with self.assertRaisesRegex(ValueError, "D requires"):
            review_command(
                classification="D",
                qualifying_exogenous_source_observation_ids=(),
            )
        with self.assertRaisesRegex(ValueError, "non-D"):
            review_command(
                classification="A",
                qualifying_exogenous_source_observation_ids=(
                    "official:event:a",
                ),
            )

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

    def test_generated_ledger_ids_are_lowercase_32_hex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LedgerStore(Path(directory) / "ledger.jsonl")
            capture, forecast, _, _, _, command = append_review_context(store)
            review = store.append_review(command)
            pattern = r"\A[0-9a-f]{32}\Z"

            for record in store.iter_records():
                with self.subTest(record_kind=record.envelope.record_kind):
                    self.assertRegex(record.envelope.record_id, pattern)
            self.assertRegex(capture.capture_id, pattern)
            self.assertRegex(review.review_id, pattern)
            self.assertEqual(forecast.forecast_run_id, "forecast-run-1")

    def test_tampered_generated_ids_are_corruption_on_reopen(self) -> None:
        tampered_values = {
            "record_id": "a" * 31,
            "capture_id": "g" * 32,
        }
        for field_name, tampered_value in tampered_values.items():
            with self.subTest(field_name=field_name):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "ledger.jsonl"
                    LedgerStore(path).append_capture(
                        "capture:tampered-id",
                        capture_command(),
                    )
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if field_name == "record_id":
                        row["envelope"]["record_id"] = tampered_value
                    else:
                        row["capture_id"] = tampered_value
                    write_jsonl_rows(path, [row])

                    with self.assertRaisesRegex(
                        LedgerCorruptionError,
                        field_name,
                    ):
                        tuple(LedgerStore(path).iter_records())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            *_, command = append_review_context(store)
            store.append_review(command)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            review_row = next(
                row
                for row in rows
                if row["envelope"]["record_kind"] == "review"
            )
            review_row["review_id"] = "A" * 32
            write_jsonl_rows(path, rows)

            with self.assertRaisesRegex(LedgerCorruptionError, "review_id"):
                tuple(LedgerStore(path).iter_records())

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

    def test_deep_malformed_json_raises_line_scoped_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-malformed.jsonl"
            LedgerStore(path).append_capture("capture:deep", capture_command())
            valid_row = path.read_text(encoding="utf-8")
            depth = sys.getrecursionlimit() * 2
            original_payload = (
                f'"payload":{canonical_json(capture_command().payload)}'
            )
            deep_payload = (
                '"payload":{"nested":'
                f"{'[' * depth}0{']' * depth}"
                "}"
            )
            malformed_row = valid_row.replace(original_payload, deep_payload)
            self.assertNotEqual(malformed_row, valid_row)
            path.write_text(malformed_row, encoding="utf-8")

            with self.assertRaisesRegex(LedgerCorruptionError, "line 1"):
                tuple(LedgerStore(path).iter_records())

    def test_optional_availability_and_classification_remain_canonical(self) -> None:
        market = market_command(available_at=None)
        review = review_command(classification=None)

        self.assertIsNone(market.available_at)
        self.assertIsNone(review.classification)
        self.assertIsInstance(review.component_reviews, tuple)
        self.assertEqual(
            review.component_reviews[0]["evidence_path"],
            "starter_suppression",
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

    def test_path_aliases_share_one_lock_and_concurrent_capture_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            symlink_path = Path(directory) / "ledger-alias.jsonl"
            symlink_path.symlink_to(path)
            relative_path = os.path.relpath(path, start=Path.cwd())

            real_store = LedgerStore(path)
            symlink_store = LedgerStore(symlink_path)
            relative_store = LedgerStore(relative_path)

            self.assertEqual(real_store.path, symlink_store.path)
            self.assertEqual(real_store.path, relative_store.path)
            self.assertEqual(real_store.lock_path, symlink_store.lock_path)
            self.assertEqual(real_store.lock_path, relative_store.lock_path)

            context = multiprocessing.get_context("fork")
            result_queue = context.Queue()
            aliases = [str(path), str(symlink_path)] * 4
            processes = [
                context.Process(
                    target=append_same_capture_and_report_in_process,
                    args=(alias, result_queue),
                )
                for alias in aliases
            ]

            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)

            self.assertTrue(all(process.exitcode == 0 for process in processes))
            capture_ids = [result_queue.get(timeout=10) for _ in processes]
            result_queue.close()
            result_queue.join_thread()
            records = tuple(real_store.iter_records("capture"))
            self.assertEqual(len(records), 1)
            self.assertEqual(set(capture_ids), {records[0].capture_id})
            self.assertEqual(path.read_bytes().count(b"\n"), 1)

    def test_hard_linked_ledger_is_rejected_on_construction_and_operation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            hard_link_path = Path(directory) / "ledger-hard-link.jsonl"
            store = LedgerStore(path)
            store.append_capture("capture:hard-link", capture_command())
            os.link(path, hard_link_path)

            with self.assertRaisesRegex(LedgerCorruptionError, "hard link"):
                LedgerStore(hard_link_path)
            with self.assertRaisesRegex(LedgerCorruptionError, "hard link"):
                tuple(store.iter_records())

    def test_ledger_path_symlink_swap_cannot_modify_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            victim = Path(directory) / "victim.txt"
            store = LedgerStore(path)
            victim.write_text("untouched\n", encoding="utf-8")
            path.symlink_to(victim)

            with self.assertRaisesRegex(LedgerCorruptionError, "symlink"):
                store.append_capture("capture:symlink-swap", capture_command())

            self.assertEqual(victim.read_text(encoding="utf-8"), "untouched\n")

    def test_regular_file_swap_between_read_and_write_cannot_receive_append(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_path = root / "ledger-displaced.jsonl"
            replacement_path = root / "ledger-replacement.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:trusted",
                capture_command(payload={"source": "trusted"}),
            )
            LedgerStore(replacement_path).append_capture(
                "capture:replacement",
                capture_command(payload={"source": "replacement"}),
            )
            trusted_content = path.read_bytes()
            replacement_content = replacement_path.read_bytes()
            real_read_records = store._read_records

            def swap_after_read(*args: object):
                records = real_read_records(*args)
                path.rename(displaced_path)
                replacement_path.rename(path)
                return records

            with mock.patch.object(
                store,
                "_read_records",
                side_effect=swap_after_read,
            ):
                with self.assertRaisesRegex(
                    LedgerCorruptionError,
                    "changed",
                ):
                    store.append_capture(
                        "capture:must-not-redirect",
                        capture_command(payload={"source": "must-not-redirect"}),
                    )

            self.assertEqual(displaced_path.read_bytes(), trusted_content)
            self.assertEqual(path.read_bytes(), replacement_content)

    def test_regular_file_replacement_after_construction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_path = root / "ledger-displaced.jsonl"
            replacement_path = root / "ledger-replacement.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:trusted",
                capture_command(payload={"source": "trusted"}),
            )
            LedgerStore(replacement_path).append_capture(
                "capture:replacement",
                capture_command(payload={"source": "replacement"}),
            )
            trusted_content = path.read_bytes()
            replacement_content = replacement_path.read_bytes()
            path.rename(displaced_path)
            replacement_path.rename(path)

            with self.assertRaisesRegex(LedgerCorruptionError, "identity"):
                store.append_capture(
                    "capture:must-not-replace",
                    capture_command(payload={"source": "must-not-replace"}),
                )

            self.assertEqual(displaced_path.read_bytes(), trusted_content)
            self.assertEqual(path.read_bytes(), replacement_content)

    def test_lock_swap_during_append_cannot_split_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_lock_path = root / "ledger-displaced.lock"
            replacement_lock_path = root / "ledger-replacement.lock"
            first_store = LedgerStore(path)
            first_read = threading.Event()
            release_first = threading.Event()
            second_read = threading.Event()
            results: list[CaptureRecord] = []
            errors: list[BaseException] = []
            real_first_read = first_store._read_records

            def pause_first_after_read(*args: object):
                records = real_first_read(*args)
                first_read.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError("first append was not released")
                return records

            def append_capture(store: LedgerStore) -> None:
                try:
                    results.append(
                        store.append_capture(
                            "capture:shared-lock-swap",
                            capture_command(payload={"source": "shared"}),
                        )
                    )
                except BaseException as error:
                    errors.append(error)

            first_thread = threading.Thread(
                target=append_capture,
                args=(first_store,),
            )
            with mock.patch.object(
                first_store,
                "_read_records",
                side_effect=pause_first_after_read,
            ):
                first_thread.start()
                self.assertTrue(first_read.wait(timeout=5))
                first_store.lock_path.rename(displaced_lock_path)
                replacement_lock_path.write_bytes(b"")
                replacement_lock_path.rename(first_store.lock_path)
                second_store = LedgerStore(path)
                real_second_read = second_store._read_records

                def signal_second_read(*args: object):
                    second_read.set()
                    return real_second_read(*args)

                second_thread = threading.Thread(
                    target=append_capture,
                    args=(second_store,),
                )
                try:
                    with mock.patch.object(
                        second_store,
                        "_read_records",
                        side_effect=signal_second_read,
                    ):
                        second_thread.start()
                        second_bypassed_first = second_read.wait(timeout=0.5)
                finally:
                    release_first.set()
                    first_thread.join(timeout=5)
                    second_thread.join(timeout=5)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertFalse(second_bypassed_first)
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(len({record.capture_id for record in results}), 1)
            self.assertEqual(path.read_bytes().count(b"\n"), 1)

    def test_swap_after_final_validation_cannot_return_append_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_path = root / "ledger-displaced.jsonl"
            replacement_path = root / "ledger-replacement.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:trusted-late-swap",
                capture_command(payload={"source": "trusted"}),
            )
            LedgerStore(replacement_path).append_capture(
                "capture:replacement-late-swap",
                capture_command(payload={"source": "replacement"}),
            )
            replacement_content = replacement_path.read_bytes()
            real_validate = store._validate_open_descriptor
            ledger_validations = 0

            def swap_after_final_validation(
                descriptor: int,
                file_name: str,
                path_label: str,
            ) -> None:
                nonlocal ledger_validations
                real_validate(descriptor, file_name, path_label)
                if file_name != store._ledger_name:
                    return
                ledger_validations += 1
                if ledger_validations == 4:
                    path.rename(displaced_path)
                    replacement_path.rename(path)

            with mock.patch.object(
                store,
                "_validate_open_descriptor",
                side_effect=swap_after_final_validation,
            ):
                with self.assertRaisesRegex(LedgerCorruptionError, "changed"):
                    store.append_capture(
                        "capture:must-not-report-success",
                        capture_command(payload={"source": "late-swap"}),
                    )

            self.assertEqual(path.read_bytes(), replacement_content)

    def test_deleted_bound_ledger_is_corruption_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:deleted-ledger",
                capture_command(payload={"source": "trusted"}),
            )
            path.unlink()

            with self.assertRaisesRegex(LedgerCorruptionError, "identity"):
                tuple(store.iter_records())

    def test_read_revalidates_ledger_identity_after_acquiring_inode_lock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_path = root / "ledger-displaced.jsonl"
            replacement_path = root / "ledger-replacement.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:trusted-read",
                capture_command(payload={"source": "trusted"}),
            )
            LedgerStore(replacement_path).append_capture(
                "capture:replacement-read",
                capture_command(payload={"source": "replacement"}),
            )
            replacement_content = replacement_path.read_bytes()
            real_validate = store._validate_open_descriptor
            swapped = False

            def swap_after_reader_open(
                descriptor: int,
                file_name: str,
                path_label: str,
            ) -> None:
                nonlocal swapped
                real_validate(descriptor, file_name, path_label)
                if file_name == store._ledger_name and not swapped:
                    path.rename(displaced_path)
                    replacement_path.rename(path)
                    swapped = True

            with mock.patch.object(
                store,
                "_validate_open_descriptor",
                side_effect=swap_after_reader_open,
            ):
                with self.assertRaisesRegex(LedgerCorruptionError, "changed"):
                    tuple(store.iter_records())

            self.assertEqual(path.read_bytes(), replacement_content)

    def test_exact_replay_revalidates_canonical_ledger_before_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.jsonl"
            displaced_path = root / "ledger-displaced.jsonl"
            replacement_path = root / "ledger-replacement.jsonl"
            store = LedgerStore(path)
            command = capture_command(payload={"source": "trusted"})
            store.append_capture("capture:trusted-replay", command)
            LedgerStore(replacement_path).append_capture(
                "capture:replacement-replay",
                capture_command(payload={"source": "replacement"}),
            )
            replacement_content = replacement_path.read_bytes()
            real_read_records = store._read_records

            def swap_after_read(*args: object):
                records = real_read_records(*args)
                path.rename(displaced_path)
                replacement_path.rename(path)
                return records

            with mock.patch.object(
                store,
                "_read_records",
                side_effect=swap_after_read,
            ):
                with self.assertRaisesRegex(LedgerCorruptionError, "changed"):
                    store.append_capture("capture:trusted-replay", command)

            self.assertEqual(path.read_bytes(), replacement_content)

    def test_shared_lock_failure_closes_opened_ledger_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            store = LedgerStore(path)
            store.append_capture(
                "capture:lock-failure",
                capture_command(payload={"source": "trusted"}),
            )
            opened_ledger_descriptors: list[int] = []
            real_open = store._open_no_follow
            real_flock = fcntl.flock

            def capture_ledger_descriptor(
                file_name: str,
                flags: int,
                path_label: str,
            ) -> int:
                descriptor = real_open(file_name, flags, path_label)
                if file_name == store._ledger_name:
                    opened_ledger_descriptors.append(descriptor)
                return descriptor

            def fail_shared_ledger_lock(
                descriptor: int,
                operation: int,
            ) -> None:
                descriptor_stat = os.fstat(descriptor)
                identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
                if operation == fcntl.LOCK_SH and identity == store._ledger_identity:
                    raise OSError("injected shared ledger lock failure")
                real_flock(descriptor, operation)

            with (
                mock.patch.object(
                    store,
                    "_open_no_follow",
                    side_effect=capture_ledger_descriptor,
                ),
                mock.patch(
                    "analysis.ledger.store.fcntl.flock",
                    side_effect=fail_shared_ledger_lock,
                ),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    tuple(store.iter_records())

            self.assertEqual(len(opened_ledger_descriptors), 1)
            with self.assertRaises(OSError):
                os.fstat(opened_ledger_descriptors[0])

    def test_parent_symlink_swap_cannot_redirect_append_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trusted_parent = root / "trusted"
            displaced_parent = root / "trusted-displaced"
            attacker_parent = root / "attacker"
            trusted_parent.mkdir()
            attacker_parent.mkdir()

            store = LedgerStore(trusted_parent / "ledger.jsonl")
            victim_path = attacker_parent / "ledger.jsonl"
            LedgerStore(victim_path).append_capture(
                "capture:victim",
                capture_command(payload={"source": "victim"}),
            )
            victim_content = victim_path.read_bytes()

            trusted_parent.rename(displaced_parent)
            trusted_parent.symlink_to(attacker_parent, target_is_directory=True)

            errors: dict[str, LedgerCorruptionError] = {}
            operations = {
                "append": lambda: store.append_capture(
                    "capture:redirected",
                    capture_command(payload={"source": "redirected"}),
                ),
                "read": lambda: tuple(store.iter_records()),
            }
            for operation_name, operation in operations.items():
                try:
                    operation()
                except LedgerCorruptionError as error:
                    errors[operation_name] = error

            self.assertEqual(victim_path.read_bytes(), victim_content)
            self.assertEqual(set(errors), {"append", "read"})
            self.assertTrue(
                all("parent" in str(error) for error in errors.values())
            )

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
