"""Checkpoint-backed lookup and single-session shadow pregame persistence."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from types import TracebackType
from typing import TYPE_CHECKING
import uuid

from analysis.forecasting.identity import canonical_json

from .checkpoint import (
    CheckpointError,
    IndexedRecord,
    LedgerCheckpoint,
    LedgerFingerprint,
    _indexed_record,
    _publish_checkpoint,
    build_checkpoint_from_descriptor,
    ledger_state,
)
from .records import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    RecordEnvelope,
)
from .store import (
    IdempotencyConflict,
    LedgerCorruptionError,
    LedgerHeadChangedError,
    LedgerReferenceError,
    _deserialize_record,
    _forecast_model_replay_is_semantically_equal,
    _logical_payload,
    _payload_hash,
    _reject_duplicate_keys,
    _reject_json_constant,
)


if TYPE_CHECKING:
    from analysis.shadow import ShadowRunRecords

    from .store import LedgerStore


@dataclass(frozen=True)
class PregameLookup:
    """Exact relationship records plus the current active game capture."""

    capture: CaptureRecord | None
    active_capture: CaptureRecord | None
    active_confirmed_capture: CaptureRecord | None
    forecast: ForecastRecord | None
    link: CaptureForecastLink | None


class PregameLedgerSession:
    """One lock-scoped checkpoint session over the authoritative ledger."""

    def __init__(self, store: "LedgerStore", *, exclusive: bool) -> None:
        self._store = store
        self._exclusive = exclusive
        self._lock_file = None
        self._ledger_descriptor = -1
        self._ledger_locked = False
        self._checkpoint: LedgerCheckpoint | None = None
        self._loaded_records: dict[str, object] = {}

    def __enter__(self) -> "PregameLedgerSession":
        lock_descriptor = self._store._open_no_follow(
            self._store._lock_name,
            os.O_RDWR,
            "ledger lock",
        )
        self._lock_file = os.fdopen(lock_descriptor, "r+b")
        lock_mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self._lock_file.fileno(), lock_mode)
            self._store._validate_open_descriptor(
                self._lock_file.fileno(),
                self._store._lock_name,
                "ledger lock",
            )
            ledger_flags = (
                os.O_RDWR | (os.O_APPEND if self._exclusive else 0)
                if self._exclusive
                else os.O_RDONLY
            )
            if self._store._create and self._store._ledger_identity is None:
                ledger_flags |= os.O_CREAT
            self._ledger_descriptor = self._store._open_no_follow(
                self._store._ledger_name,
                ledger_flags,
                "ledger",
            )
            fcntl.flock(self._ledger_descriptor, lock_mode)
            self._ledger_locked = True
            self._store._validate_open_descriptor(
                self._ledger_descriptor,
                self._store._ledger_name,
                "ledger",
            )
            self._checkpoint = self._load_or_rebuild_checkpoint()
            self._store._pregame_checkpoint_cache = self._checkpoint
            return self
        except Exception:
            self._release()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._release()

    def _release(self) -> None:
        if self._ledger_descriptor >= 0:
            try:
                if self._ledger_locked:
                    fcntl.flock(self._ledger_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._ledger_descriptor)
                self._ledger_descriptor = -1
                self._ledger_locked = False
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_file.close()
                self._lock_file = None

    def _load_or_rebuild_checkpoint(self) -> LedgerCheckpoint:
        try:
            ledger_stat, ledger_sha256 = ledger_state(self._ledger_descriptor)
        except CheckpointError as error:
            raise LedgerCorruptionError(str(error)) from error
        expected_fingerprint = LedgerFingerprint.from_stat(
            ledger_stat,
            ledger_sha256,
        )
        cached = self._store._pregame_checkpoint_cache
        if cached is not None and cached.fingerprint == expected_fingerprint:
            return cached
        try:
            return LedgerCheckpoint.load(
                self._store.checkpoint_path,
                ledger_stat,
                ledger_sha256,
                ledger_descriptor=self._ledger_descriptor,
            )
        except CheckpointError:
            records = self._store._read_records(self._ledger_descriptor)
            try:
                checkpoint = build_checkpoint_from_descriptor(
                    self._store.path,
                    self._ledger_descriptor,
                    records,
                    publish=self._store._create,
                )
                final_stat, final_sha256 = ledger_state(
                    self._ledger_descriptor
                )
            except CheckpointError as error:
                raise LedgerCorruptionError(str(error)) from error
            if checkpoint.fingerprint != LedgerFingerprint.from_stat(
                final_stat,
                final_sha256,
            ):
                raise LedgerCorruptionError(
                    "ledger changed during checkpoint rebuild"
                )
            return checkpoint
        except OSError as error:
            raise LedgerCorruptionError(
                "authoritative ledger could not be read during checkpoint "
                "validation"
            ) from error

    @property
    def checkpoint(self) -> LedgerCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("pregame ledger session is not open")
        return self._checkpoint

    def _load_indexed_record(self, indexed: IndexedRecord):
        cached = self._loaded_records.get(indexed.record_id)
        if cached is not None:
            return cached
        raw_line = os.pread(
            self._ledger_descriptor,
            indexed.byte_length,
            indexed.byte_offset,
        )
        if (
            len(raw_line) != indexed.byte_length
            or not raw_line.endswith(b"\n")
            or b"\n" in raw_line[:-1]
        ):
            raise LedgerCorruptionError(
                "checkpoint record does not resolve to one newline-terminated "
                "ledger row"
            )
        try:
            value = json.loads(
                raw_line[:-1].decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            record = _deserialize_record(value)
        except (
            KeyError,
            RecursionError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise LedgerCorruptionError(
                "checkpoint record cannot be deserialized from the ledger"
            ) from error
        if raw_line != canonical_json(record).encode("utf-8") + b"\n":
            raise LedgerCorruptionError(
                "checkpoint record is not a canonical ledger row"
            )
        envelope = record.envelope
        if (
            envelope.record_kind != indexed.record_kind
            or envelope.record_id != indexed.record_id
            or envelope.idempotency_key != indexed.idempotency_key
            or envelope.payload_hash != indexed.payload_hash
            or envelope.appended_at != indexed.appended_at
            or _payload_hash(_logical_payload(record)) != envelope.payload_hash
        ):
            raise LedgerCorruptionError(
                "checkpoint envelope metadata differs from the ledger record"
            )
        self._require_index_metadata(indexed, record)
        self._loaded_records[indexed.record_id] = record
        return record

    @staticmethod
    def _require_index_metadata(indexed: IndexedRecord, record: object) -> None:
        if isinstance(record, CaptureRecord):
            actual = (
                record.capture_id,
                record.game_pk,
                record.captured_at,
                record.first_pitch,
                record.normalized_input_hash,
                record.supersedes_capture_id,
            )
            expected = (
                indexed.capture_id,
                indexed.game_pk,
                indexed.captured_at,
                indexed.first_pitch,
                indexed.normalized_input_hash,
                indexed.supersedes_capture_id,
            )
        elif isinstance(record, ForecastRecord):
            actual = (
                record.forecast_run_id,
                record.normalized_input_hash,
            )
            expected = (
                indexed.forecast_run_id,
                indexed.normalized_input_hash,
            )
        elif isinstance(record, CaptureForecastLink):
            actual = (record.capture_id, record.forecast_run_id)
            expected = (indexed.capture_id, indexed.forecast_run_id)
        else:
            raise LedgerCorruptionError(
                "pregame checkpoint resolved an unsupported ledger record"
            )
        if actual != expected:
            raise LedgerCorruptionError(
                "checkpoint relationship metadata differs from the ledger record"
            )

    def _indexed_capture_for_key(
        self,
        capture_key: str,
    ) -> IndexedRecord | None:
        return next(
            (
                record
                for record in self.checkpoint.records
                if record.record_kind == "capture"
                and record.idempotency_key == capture_key
            ),
            None,
        )

    def _indexed_forecast(
        self,
        forecast_run_id: str,
    ) -> IndexedRecord | None:
        return next(
            (
                record
                for record in self.checkpoint.records
                if record.record_kind == "forecast"
                and record.forecast_run_id == forecast_run_id
            ),
            None,
        )

    def _indexed_links(
        self,
        capture_id: str,
        forecast_run_id: str,
    ) -> tuple[IndexedRecord, ...]:
        return tuple(
            record
            for record in self.checkpoint.records
            if record.record_kind == "capture_forecast_link"
            and record.capture_id == capture_id
            and record.forecast_run_id == forecast_run_id
        )

    def _active_link(
        self,
        capture_id: str,
        forecast_run_id: str,
    ) -> CaptureForecastLink | None:
        indexed_links = self._indexed_links(capture_id, forecast_run_id)
        if not indexed_links:
            return None
        links = tuple(
            self._load_indexed_record(indexed)
            for indexed in indexed_links
        )
        if any(not isinstance(link, CaptureForecastLink) for link in links):
            raise LedgerCorruptionError(
                "link checkpoint resolved wrong kind"
            )
        links_by_id = {
            link.envelope.record_id: link
            for link in links
            if isinstance(link, CaptureForecastLink)
        }
        superseded_ids: set[str] = set()
        for link in links_by_id.values():
            superseded_id = link.supersedes_link_record_id
            if superseded_id is None:
                continue
            superseded = links_by_id.get(superseded_id)
            if superseded is None:
                raise LedgerCorruptionError(
                    "capture-forecast link correction supersedes a missing "
                    "relationship"
                )
            if superseded_id in superseded_ids:
                raise LedgerCorruptionError(
                    "capture-forecast link correction lineage branches"
                )
            superseded_ids.add(superseded_id)
        active = tuple(
            link
            for record_id, link in links_by_id.items()
            if record_id not in superseded_ids
        )
        if len(active) != 1:
            raise LedgerCorruptionError(
                "capture-forecast relationship must have exactly one active link"
            )
        return active[0]

    def _active_captures(
        self,
        game_pk: int,
    ) -> tuple[IndexedRecord, ...]:
        captures = tuple(
            record
            for record in self.checkpoint.records
            if record.record_kind == "capture" and record.game_pk == game_pk
        )
        superseded_ids = frozenset(
            record.supersedes_capture_id
            for record in captures
            if record.supersedes_capture_id is not None
        )
        active = tuple(
            record
            for record in captures
            if record.captured_at is not None
            and record.first_pitch is not None
            and record.captured_at < record.first_pitch
            and record.capture_id not in superseded_ids
        )
        return active

    def _latest_active_capture(
        self,
        game_pk: int,
    ) -> IndexedRecord | None:
        active = self._active_captures(game_pk)
        if not active:
            return None
        return max(
            active,
            key=lambda record: (
                record.captured_at,
                record.appended_at,
                record.record_id,
            ),
        )

    def _active_confirmed_capture(
        self,
        game_pk: int,
    ) -> CaptureRecord | None:
        active = sorted(
            self._active_captures(game_pk),
            key=lambda record: (
                record.captured_at,
                record.appended_at,
                record.record_id,
            ),
            reverse=True,
        )
        for indexed in active:
            capture = self._load_indexed_record(indexed)
            if not isinstance(capture, CaptureRecord):
                raise LedgerCorruptionError(
                    "active capture checkpoint resolved wrong kind"
                )
            if capture.lineup_state == "confirmed":
                return capture
        return None

    def _validate_lookup_snapshot(self) -> None:
        try:
            ledger_stat, ledger_sha256 = ledger_state(
                self._ledger_descriptor
            )
        except CheckpointError as error:
            raise LedgerCorruptionError(str(error)) from error
        if LedgerFingerprint.from_stat(
            ledger_stat,
            ledger_sha256,
        ) != self.checkpoint.fingerprint:
            raise LedgerCorruptionError(
                "ledger changed during pregame lookup"
            )
        self._store._validate_open_descriptor(
            self._ledger_descriptor,
            self._store._ledger_name,
            "ledger",
        )
        return_stat = os.fstat(self._ledger_descriptor)
        if (
            return_stat.st_dev,
            return_stat.st_ino,
            return_stat.st_size,
            return_stat.st_mtime_ns,
        ) != (
            self.checkpoint.fingerprint.device,
            self.checkpoint.fingerprint.inode,
            self.checkpoint.fingerprint.size,
            self.checkpoint.fingerprint.mtime_ns,
        ):
            raise LedgerCorruptionError(
                "ledger changed during pregame lookup"
            )

    def lookup(
        self,
        capture_key: str,
        forecast_run_id: str,
        game_pk: int,
    ) -> PregameLookup:
        capture_index = self._indexed_capture_for_key(capture_key)
        active_index = self._latest_active_capture(game_pk)
        active_confirmed_capture = self._active_confirmed_capture(game_pk)
        forecast_index = self._indexed_forecast(forecast_run_id)
        link = (
            None
            if capture_index is None or forecast_index is None
            else self._active_link(
                capture_index.capture_id or "",
                forecast_run_id,
            )
        )
        capture = (
            None
            if capture_index is None
            else self._load_indexed_record(capture_index)
        )
        active_capture = (
            None
            if active_index is None
            else self._load_indexed_record(active_index)
        )
        forecast = (
            None
            if forecast_index is None
            else self._load_indexed_record(forecast_index)
        )
        if capture is not None and not isinstance(capture, CaptureRecord):
            raise LedgerCorruptionError("capture checkpoint resolved wrong kind")
        if active_capture is not None and not isinstance(
            active_capture,
            CaptureRecord,
        ):
            raise LedgerCorruptionError(
                "active capture checkpoint resolved wrong kind"
            )
        if forecast is not None and not isinstance(forecast, ForecastRecord):
            raise LedgerCorruptionError("forecast checkpoint resolved wrong kind")
        if link is not None and not isinstance(link, CaptureForecastLink):
            raise LedgerCorruptionError("link checkpoint resolved wrong kind")
        self._validate_lookup_snapshot()
        return PregameLookup(
            capture=capture,
            active_capture=active_capture,
            active_confirmed_capture=active_confirmed_capture,
            forecast=forecast,
            link=link,
        )

    def append_shadow_run(
        self,
        capture_key: str,
        capture_command: CaptureCommand,
        forecast_command: ForecastCommand,
        expected_active_capture_id: str | None,
    ) -> "ShadowRunRecords":
        capture, capture_is_new = self._resolve_capture(
            capture_key,
            capture_command,
            expected_active_capture_id,
        )
        forecast, forecast_is_new = self._resolve_forecast(forecast_command)
        if capture.normalized_input_hash != forecast.normalized_input_hash:
            raise LedgerReferenceError(
                "capture and forecast normalized_input_hash must match"
            )
        link, link_is_new = self._resolve_link(
            capture,
            forecast,
            forecast_command,
        )
        new_records = tuple(
            record
            for record, is_new in (
                (capture, capture_is_new),
                (forecast, forecast_is_new),
                (link, link_is_new),
            )
            if is_new
        )
        if new_records:
            self._append_records(new_records)

        from analysis.shadow import ShadowRunRecords

        return ShadowRunRecords(
            capture=capture,
            forecast=forecast,
            link=link,
            capture_replayed=not capture_is_new,
            forecast_replayed=not forecast_is_new,
            link_replayed=not link_is_new,
        )

    def _resolve_capture(
        self,
        capture_key: str,
        command: CaptureCommand,
        expected_active_capture_id: str | None,
    ) -> tuple[CaptureRecord, bool]:
        logical_hash = _payload_hash(command)
        existing_index = self._indexed_capture_for_key(capture_key)
        if existing_index is not None:
            existing = self._load_indexed_record(existing_index)
            if not isinstance(existing, CaptureRecord):
                raise LedgerCorruptionError(
                    "capture idempotency resolved to wrong kind"
                )
            if existing.envelope.payload_hash != logical_hash:
                raise IdempotencyConflict(
                    "idempotency key already used for another logical capture "
                    f"payload: {capture_key}"
                )
            return existing, False

        if (
            command.lineup_state == "projected"
            and self._active_confirmed_capture(command.game_pk) is not None
        ):
            raise LedgerHeadChangedError(
                "active confirmed pregame capture blocks a new projected capture"
            )
        active = self._latest_active_capture(command.game_pk)
        actual_active_capture_id = (
            None if active is None else active.capture_id
        )
        if actual_active_capture_id != expected_active_capture_id:
            raise LedgerHeadChangedError(
                "active pregame capture head changed: expected "
                f"{expected_active_capture_id!r}, found "
                f"{actual_active_capture_id!r}"
            )
        captures_by_id = {
            record.capture_id: record
            for record in self.checkpoint.records
            if record.record_kind == "capture"
        }
        supersession_targets = {
            record.supersedes_capture_id
            for record in captures_by_id.values()
            if record.supersedes_capture_id is not None
        }
        if command.supersedes_capture_id is not None:
            superseded = captures_by_id.get(command.supersedes_capture_id)
            if superseded is None:
                raise LedgerReferenceError(
                    "supersedes_capture_id must reference an existing capture"
                )
            if superseded.game_pk != command.game_pk:
                raise LedgerReferenceError(
                    "superseded capture must concern the same game"
                )
            if command.supersedes_capture_id in supersession_targets:
                raise LedgerReferenceError(
                    "capture supersession target already has a successor"
                )
        envelope = self._new_envelope("capture", capture_key, logical_hash)
        capture_id = uuid.uuid4().hex
        while capture_id == command.supersedes_capture_id:
            capture_id = uuid.uuid4().hex
        return (
            CaptureRecord(
                envelope=envelope,
                capture_id=capture_id,
                game_pk=command.game_pk,
                captured_at=command.captured_at,
                first_pitch=command.first_pitch,
                lineup_state=command.lineup_state,
                out_of_sample_eligible=command.out_of_sample_eligible,
                normalized_input_hash=command.normalized_input_hash,
                payload=command.payload,
                supersedes_capture_id=command.supersedes_capture_id,
            ),
            True,
        )

    def _resolve_forecast(
        self,
        command: ForecastCommand,
    ) -> tuple[ForecastRecord, bool]:
        logical_hash = _payload_hash(command)
        existing_index = self._indexed_forecast(command.forecast_run_id)
        if existing_index is not None:
            existing = self._load_indexed_record(existing_index)
            if not isinstance(existing, ForecastRecord):
                raise LedgerCorruptionError(
                    "forecast idempotency resolved to wrong kind"
                )
            if (
                existing.envelope.payload_hash == logical_hash
                or _forecast_model_replay_is_semantically_equal(
                    existing,
                    command,
                )
            ):
                return existing, False
            raise IdempotencyConflict(
                "idempotency key already used for another logical forecast "
                f"payload: forecast:{command.forecast_run_id}"
            )
        envelope = self._new_envelope(
            "forecast",
            f"forecast:{command.forecast_run_id}",
            logical_hash,
        )
        return (
            ForecastRecord(
                envelope=envelope,
                forecast_run_id=command.forecast_run_id,
                model_version=command.model_version,
                simulation_config_hash=command.simulation_config_hash,
                normalized_input_hash=command.normalized_input_hash,
                outputs=command.outputs,
            ),
            True,
        )

    def _resolve_link(
        self,
        capture: CaptureRecord,
        forecast: ForecastRecord,
        forecast_command: ForecastCommand,
    ) -> tuple[CaptureForecastLink, bool]:
        incoming_comparison = forecast_command.outputs.get(
            "shadow_comparison"
        )
        if incoming_comparison is not None and not isinstance(
            incoming_comparison,
            Mapping,
        ):
            raise LedgerReferenceError(
                "forecast command shadow_comparison must be a mapping"
            )
        existing = self._active_link(
            capture.capture_id,
            forecast.forecast_run_id,
        )
        if incoming_comparison is None and existing is not None:
            return existing, False
        stored_comparison = (
            forecast.outputs.get("shadow_comparison")
            if existing is None or existing.shadow_comparison is None
            else existing.shadow_comparison
        )
        if existing is not None and canonical_json(
            stored_comparison
        ) == canonical_json(incoming_comparison):
            return existing, False

        comparison_override = (
            None
            if existing is None
            and (
                incoming_comparison is None
                or canonical_json(stored_comparison)
                == canonical_json(incoming_comparison)
            )
            else incoming_comparison
        )
        supersedes_link_record_id = (
            None if existing is None else existing.envelope.record_id
        )
        if comparison_override is None:
            idempotency_key = (
                f"capture-forecast:{capture.capture_id}:"
                f"{forecast.forecast_run_id}"
            )
        else:
            comparison_hash = hashlib.sha256(
                canonical_json(comparison_override).encode("utf-8")
            ).hexdigest()
            predecessor = (
                "root"
                if supersedes_link_record_id is None
                else supersedes_link_record_id
            )
            idempotency_key = (
                f"capture-forecast:{capture.capture_id}:"
                f"{forecast.forecast_run_id}:shadow:"
                f"{predecessor}:{comparison_hash}"
            )
        logical_payload: dict[str, object] = {
            "capture_id": capture.capture_id,
            "forecast_run_id": forecast.forecast_run_id,
        }
        if comparison_override is not None:
            logical_payload["shadow_comparison"] = comparison_override
        if supersedes_link_record_id is not None:
            logical_payload["supersedes_link_record_id"] = (
                supersedes_link_record_id
            )
        logical_hash = _payload_hash(logical_payload)
        return (
            CaptureForecastLink(
                envelope=self._new_envelope(
                    "capture_forecast_link",
                    idempotency_key,
                    logical_hash,
                ),
                capture_id=capture.capture_id,
                forecast_run_id=forecast.forecast_run_id,
                shadow_comparison=comparison_override,
                supersedes_link_record_id=supersedes_link_record_id,
            ),
            True,
        )

    @staticmethod
    def _new_envelope(
        record_kind: str,
        idempotency_key: str,
        payload_hash: str,
    ) -> RecordEnvelope:
        return RecordEnvelope(
            record_id=uuid.uuid4().hex,
            record_kind=record_kind,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            appended_at=datetime.now(timezone.utc),
        )

    def _append_records(self, records: tuple[object, ...]) -> None:
        self._store._validate_open_descriptor(
            self._ledger_descriptor,
            self._store._ledger_name,
            "ledger",
        )
        byte_offset = self.checkpoint.fingerprint.size
        indexed_records: list[IndexedRecord] = []
        for record in records:
            raw_line = canonical_json(record).encode("utf-8") + b"\n"
            self._write_all(raw_line)
            indexed_records.append(
                _indexed_record(
                    record,
                    byte_offset=byte_offset,
                    byte_length=len(raw_line),
                )
            )
            byte_offset += len(raw_line)
        os.fsync(self._ledger_descriptor)
        self._store._validate_open_descriptor(
            self._ledger_descriptor,
            self._store._ledger_name,
            "ledger",
        )
        try:
            ledger_stat, ledger_sha256 = ledger_state(
                self._ledger_descriptor
            )
        except CheckpointError as error:
            raise LedgerCorruptionError(str(error)) from error
        all_indexed = self.checkpoint.records + tuple(indexed_records)
        checkpoint = LedgerCheckpoint(
            path=self._store.checkpoint_path,
            fingerprint=LedgerFingerprint.from_stat(
                ledger_stat,
                ledger_sha256,
            ),
            record_count=len(all_indexed),
            latest_appended_at=max(
                record.appended_at for record in all_indexed
            ),
            records=all_indexed,
        )
        self._checkpoint = checkpoint
        self._store._pregame_checkpoint_cache = checkpoint
        try:
            _publish_checkpoint(checkpoint)
        except (CheckpointError, OSError):
            pass

    def _write_all(self, content: bytes) -> None:
        written = 0
        while written < len(content):
            count = os.write(self._ledger_descriptor, content[written:])
            if count <= 0:
                raise OSError("ledger append made no progress")
            written += count


__all__ = ("PregameLedgerSession", "PregameLookup")
