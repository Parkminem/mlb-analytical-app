"""Disposable, strictly validated acceleration state for the JSONL ledger."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat

from analysis.forecasting.identity import canonical_json

from .records import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    MarketObservationRecord,
    OfficialResultRecord,
    ReviewRecord,
)


_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_SUFFIX = ".checkpoint-v1.json"
_KNOWN_RECORD_KINDS = frozenset(
    (
        "capture",
        "forecast",
        "capture_forecast_link",
        "market_observation",
        "official_result",
        "review",
    )
)
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_CAPTURE_METADATA_KEYS = frozenset(
    (
        "capture_id",
        "game_pk",
        "captured_at",
        "first_pitch",
        "normalized_input_hash",
        "supersedes_capture_id",
    )
)
_FORECAST_METADATA_KEYS = frozenset(
    ("forecast_run_id", "normalized_input_hash")
)
_LINK_METADATA_KEYS = frozenset(("capture_id", "forecast_run_id"))
_EMPTY_METADATA_KEYS = frozenset()
_ROW_INDEX_FIELDS = frozenset(
    (
        "capture_id",
        "captured_at",
        "envelope",
        "first_pitch",
        "forecast_run_id",
        "game_pk",
        "normalized_input_hash",
        "supersedes_capture_id",
    )
)

LedgerRecord = (
    CaptureRecord
    | ForecastRecord
    | CaptureForecastLink
    | MarketObservationRecord
    | OfficialResultRecord
    | ReviewRecord
)


class CheckpointError(ValueError):
    """Raised when disposable checkpoint state cannot be trusted."""


def _require_integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointError(f"{field_name} must be a non-empty string")
    return value


def _require_generated_id(value: object, field_name: str) -> str:
    result = _require_nonempty_string(value, field_name)
    if (
        len(result) != 32
        or any(character not in _LOWERCASE_HEX for character in result)
    ):
        raise CheckpointError(
            f"{field_name} must be exactly 32 lowercase hexadecimal characters"
        )
    return result


def _require_sha256(value: object, field_name: str) -> str:
    result = _require_nonempty_string(value, field_name)
    if (
        len(result) != 64
        or any(character not in _LOWERCASE_HEX for character in result)
    ):
        raise CheckpointError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters"
        )
    return result


def _parse_datetime(value: object, field_name: str) -> datetime:
    text = _require_nonempty_string(value, field_name)
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise CheckpointError(f"{field_name} must be an ISO datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CheckpointError(f"{field_name} must be timezone-aware")
    return parsed


def _parse_optional_datetime(
    value: object,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value, field_name)


def _require_exact_mapping(
    value: object,
    expected_keys: frozenset[str],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{field_name} must be a JSON object")
    if frozenset(value) != expected_keys:
        raise CheckpointError(
            f"{field_name} must contain exactly {sorted(expected_keys)!r}"
        )
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointError(f"duplicate checkpoint JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise CheckpointError(f"invalid checkpoint JSON number: {value}")


@dataclass(frozen=True)
class LedgerFingerprint:
    """Complete identity and content fingerprint for the authoritative ledger."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        _require_integer(self.device, "fingerprint.device")
        _require_integer(self.inode, "fingerprint.inode")
        _require_integer(self.size, "fingerprint.size")
        _require_integer(self.mtime_ns, "fingerprint.mtime_ns")
        _require_sha256(self.sha256, "fingerprint.sha256")

    @classmethod
    def from_stat(
        cls,
        ledger_stat: os.stat_result,
        ledger_sha256: str,
    ) -> "LedgerFingerprint":
        return cls(
            device=ledger_stat.st_dev,
            inode=ledger_stat.st_ino,
            size=ledger_stat.st_size,
            mtime_ns=ledger_stat.st_mtime_ns,
            sha256=ledger_sha256,
        )


@dataclass(frozen=True)
class IndexedRecord:
    """Compact envelope, offset, and pregame relationship metadata."""

    record_kind: str
    record_id: str
    idempotency_key: str
    payload_hash: str
    appended_at: datetime
    byte_offset: int
    byte_length: int
    identifier: str
    game_pk: int | None = None
    captured_at: datetime | None = None
    first_pitch: datetime | None = None
    normalized_input_hash: str | None = None
    supersedes_capture_id: str | None = None
    capture_id: str | None = None
    forecast_run_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_kind, str)
            or self.record_kind not in _KNOWN_RECORD_KINDS
        ):
            raise CheckpointError("indexed record has an unknown record_kind")
        _require_generated_id(self.record_id, "indexed record.record_id")
        _require_nonempty_string(
            self.idempotency_key,
            "indexed record.idempotency_key",
        )
        _require_sha256(self.payload_hash, "indexed record.payload_hash")
        if (
            not isinstance(self.appended_at, datetime)
            or self.appended_at.tzinfo is None
            or self.appended_at.utcoffset() is None
        ):
            raise CheckpointError(
                "indexed record.appended_at must be timezone-aware"
            )
        _require_integer(self.byte_offset, "indexed record.byte_offset")
        _require_integer(
            self.byte_length,
            "indexed record.byte_length",
            minimum=1,
        )
        _require_nonempty_string(self.identifier, "indexed record.identifier")
        self._validate_kind_metadata()

    def _validate_kind_metadata(self) -> None:
        optional_values = (
            self.game_pk,
            self.captured_at,
            self.first_pitch,
            self.normalized_input_hash,
            self.supersedes_capture_id,
            self.capture_id,
            self.forecast_run_id,
        )
        if self.record_kind == "capture":
            if self.game_pk is None:
                raise CheckpointError("capture checkpoint metadata requires game_pk")
            _require_integer(
                self.game_pk,
                "capture checkpoint metadata.game_pk",
                minimum=1,
            )
            for field_name, value in (
                ("captured_at", self.captured_at),
                ("first_pitch", self.first_pitch),
            ):
                if (
                    not isinstance(value, datetime)
                    or value.tzinfo is None
                    or value.utcoffset() is None
                ):
                    raise CheckpointError(
                        f"capture checkpoint metadata.{field_name} must be "
                        "timezone-aware"
                    )
            _require_nonempty_string(
                self.normalized_input_hash,
                "capture checkpoint metadata.normalized_input_hash",
            )
            capture_id = _require_generated_id(
                self.capture_id,
                "capture checkpoint metadata.capture_id",
            )
            if self.supersedes_capture_id is not None:
                _require_generated_id(
                    self.supersedes_capture_id,
                    "capture checkpoint metadata.supersedes_capture_id",
                )
            if self.forecast_run_id is not None or self.identifier != capture_id:
                raise CheckpointError("capture checkpoint metadata is contradictory")
            return
        if self.record_kind == "forecast":
            forecast_run_id = _require_nonempty_string(
                self.forecast_run_id,
                "forecast checkpoint metadata.forecast_run_id",
            )
            _require_nonempty_string(
                self.normalized_input_hash,
                "forecast checkpoint metadata.normalized_input_hash",
            )
            forbidden = (
                self.game_pk,
                self.captured_at,
                self.first_pitch,
                self.supersedes_capture_id,
                self.capture_id,
            )
            if any(value is not None for value in forbidden):
                raise CheckpointError("forecast checkpoint metadata is contradictory")
            if self.identifier != forecast_run_id:
                raise CheckpointError("forecast checkpoint identifier is contradictory")
            return
        if self.record_kind == "capture_forecast_link":
            _require_generated_id(
                self.capture_id,
                "link checkpoint metadata.capture_id",
            )
            _require_nonempty_string(
                self.forecast_run_id,
                "link checkpoint metadata.forecast_run_id",
            )
            forbidden = (
                self.game_pk,
                self.captured_at,
                self.first_pitch,
                self.normalized_input_hash,
                self.supersedes_capture_id,
            )
            if any(value is not None for value in forbidden):
                raise CheckpointError("link checkpoint metadata is contradictory")
            if self.identifier != self.record_id:
                raise CheckpointError("link checkpoint identifier is contradictory")
            return
        if any(value is not None for value in optional_values):
            raise CheckpointError(
                f"{self.record_kind} checkpoint metadata must be empty"
            )
        if self.identifier != self.record_id:
            raise CheckpointError(
                f"{self.record_kind} checkpoint identifier is contradictory"
            )


@dataclass(frozen=True)
class LedgerCheckpoint:
    """Versioned checkpoint whose validity is bound to a complete fingerprint."""

    path: Path
    fingerprint: LedgerFingerprint
    record_count: int
    latest_appended_at: datetime | None
    records: tuple[IndexedRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise CheckpointError("checkpoint path must be a Path")
        if not isinstance(self.fingerprint, LedgerFingerprint):
            raise CheckpointError(
                "checkpoint fingerprint must be a LedgerFingerprint"
            )
        _require_integer(self.record_count, "checkpoint.record_count")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, IndexedRecord) for record in self.records
        ):
            raise CheckpointError(
                "checkpoint records must be a tuple of IndexedRecord values"
            )
        if self.record_count != len(self.records):
            raise CheckpointError("checkpoint record_count does not match records")
        self._validate_offsets_and_relationships()

    def _validate_offsets_and_relationships(self) -> None:
        expected_offset = 0
        record_ids: set[str] = set()
        idempotency_keys: set[tuple[str, str]] = set()
        captures: dict[str, IndexedRecord] = {}
        forecasts: dict[str, IndexedRecord] = {}
        capture_supersession_targets: set[str] = set()
        for record in self.records:
            if record.byte_offset != expected_offset:
                raise CheckpointError(
                    "checkpoint record offsets must be contiguous and non-overlapping"
                )
            expected_offset += record.byte_length
            if expected_offset > self.fingerprint.size:
                raise CheckpointError("checkpoint record offset exceeds ledger size")
            if record.record_id in record_ids:
                raise CheckpointError("checkpoint contains duplicate record_id")
            record_ids.add(record.record_id)
            idempotency_key = (record.record_kind, record.idempotency_key)
            if idempotency_key in idempotency_keys:
                raise CheckpointError(
                    "checkpoint contains duplicate record idempotency key"
                )
            idempotency_keys.add(idempotency_key)

            if record.record_kind == "capture":
                if record.capture_id is None:
                    raise CheckpointError(
                        "capture checkpoint metadata requires capture_id"
                    )
                if record.capture_id in captures:
                    raise CheckpointError("checkpoint contains duplicate capture_id")
                if record.supersedes_capture_id is not None:
                    superseded = captures.get(record.supersedes_capture_id)
                    if superseded is None:
                        raise CheckpointError(
                            "checkpoint capture supersedes a missing capture"
                        )
                    if superseded.game_pk != record.game_pk:
                        raise CheckpointError(
                            "checkpoint capture supersedes a different game"
                        )
                    if (
                        record.supersedes_capture_id
                        in capture_supersession_targets
                    ):
                        raise CheckpointError(
                            "checkpoint capture supersession branches"
                        )
                    capture_supersession_targets.add(
                        record.supersedes_capture_id
                    )
                captures[record.capture_id] = record
            elif record.record_kind == "forecast":
                if record.forecast_run_id is None:
                    raise CheckpointError(
                        "forecast checkpoint metadata requires forecast_run_id"
                    )
                if record.forecast_run_id in forecasts:
                    raise CheckpointError(
                        "checkpoint contains duplicate forecast_run_id"
                    )
                forecasts[record.forecast_run_id] = record
            elif record.record_kind == "capture_forecast_link":
                capture = captures.get(record.capture_id or "")
                forecast = forecasts.get(record.forecast_run_id or "")
                if capture is None or forecast is None:
                    raise CheckpointError(
                        "checkpoint link references missing relationship metadata"
                    )
                if (
                    capture.normalized_input_hash
                    != forecast.normalized_input_hash
                ):
                    raise CheckpointError(
                        "checkpoint link normalized_input_hash is contradictory"
                    )

        if expected_offset != self.fingerprint.size:
            raise CheckpointError(
                "checkpoint offsets do not cover the complete ledger"
            )
        expected_latest = (
            None
            if not self.records
            else max(record.appended_at for record in self.records)
        )
        if self.latest_appended_at != expected_latest:
            raise CheckpointError(
                "checkpoint latest_appended_at does not match records"
            )

    def record(self, kind: str, identifier: str) -> IndexedRecord | None:
        """Return one logical record identity, or its envelope identity."""

        for indexed in self.records:
            if indexed.record_kind == kind and indexed.identifier == identifier:
                return indexed
        for indexed in self.records:
            if indexed.record_kind == kind and indexed.record_id == identifier:
                return indexed
        return None

    @classmethod
    def load(
        cls,
        path: Path,
        ledger_stat: os.stat_result,
        ledger_sha256: str,
        *,
        ledger_descriptor: int | None = None,
    ) -> "LedgerCheckpoint":
        """Load a checkpoint only when every ledger fingerprint field matches."""

        checkpoint_path = _absolute_path(path)
        raw_content = _read_checkpoint_file(checkpoint_path)
        try:
            value = json.loads(
                raw_content.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise CheckpointError("checkpoint is not valid UTF-8 JSON") from error
        try:
            checkpoint = _checkpoint_from_json(checkpoint_path, value)
        except RecursionError as error:
            raise CheckpointError(
                "checkpoint JSON nesting is too deep"
            ) from error
        expected_fingerprint = LedgerFingerprint.from_stat(
            ledger_stat,
            ledger_sha256,
        )
        if checkpoint.fingerprint != expected_fingerprint:
            raise CheckpointError(
                "checkpoint fingerprint differs from the complete ledger "
                "fingerprint"
            )
        if ledger_descriptor is not None:
            _validate_checkpoint_rows(checkpoint, ledger_descriptor)
        else:
            _validate_checkpoint_rows_from_path(
                checkpoint,
                ledger_stat,
                ledger_sha256,
            )
        return checkpoint


def checkpoint_path_for(ledger_path: Path) -> Path:
    path = _absolute_path(ledger_path)
    return path.with_name(f"{path.name}{_CHECKPOINT_SUFFIX}")


def ledger_state(
    descriptor: int,
) -> tuple[os.stat_result, str]:
    """Return stable descriptor metadata and SHA-256 without changing its offset."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise CheckpointError("ledger descriptor must reference a regular file")
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    stable_fields_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_fields_before != stable_fields_after or offset != after.st_size:
        raise CheckpointError("ledger changed while computing its fingerprint")
    return after, digest.hexdigest()


def build_checkpoint(
    ledger_path: Path,
    records: Sequence[LedgerRecord],
) -> LedgerCheckpoint:
    """Build and atomically publish a checkpoint from validated typed records."""

    from .store import LedgerStore

    path = _absolute_path(ledger_path)
    store = LedgerStore(path, _resolve_path_aliases=False)
    supplied_records = tuple(records)
    lock_descriptor = store._open_no_follow(
        store._lock_name,
        os.O_RDWR,
        "ledger lock",
    )
    with os.fdopen(lock_descriptor, "r+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            store._validate_open_descriptor(
                lock_file.fileno(),
                store._lock_name,
                "ledger lock",
            )
            ledger_descriptor = store._open_no_follow(
                store._ledger_name,
                os.O_RDONLY,
                "ledger",
            )
            ledger_locked = False
            try:
                fcntl.flock(ledger_descriptor, fcntl.LOCK_SH)
                ledger_locked = True
                authoritative_records = store._read_records(ledger_descriptor)
                if supplied_records != authoritative_records:
                    raise CheckpointError(
                        "supplied validated records differ from the "
                        "authoritative ledger"
                    )
                return build_checkpoint_from_descriptor(
                    path,
                    ledger_descriptor,
                    authoritative_records,
                )
            finally:
                if ledger_locked:
                    fcntl.flock(ledger_descriptor, fcntl.LOCK_UN)
                os.close(ledger_descriptor)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_checkpoint_from_descriptor(
    ledger_path: Path,
    ledger_descriptor: int,
    records: Sequence[LedgerRecord],
    *,
    publish: bool = True,
) -> LedgerCheckpoint:
    """Build from a caller-locked authoritative ledger descriptor."""

    path = _absolute_path(ledger_path)
    ledger_stat, ledger_sha256 = ledger_state(ledger_descriptor)
    raw_content = _read_descriptor(ledger_descriptor, ledger_stat.st_size)
    after_scan = os.fstat(ledger_descriptor)
    if (
        LedgerFingerprint.from_stat(after_scan, hashlib.sha256(raw_content).hexdigest())
        != LedgerFingerprint.from_stat(ledger_stat, ledger_sha256)
    ):
        raise CheckpointError("ledger changed while scanning raw offsets")
    indexed_records = _index_records(raw_content, records)
    checkpoint = LedgerCheckpoint(
        path=checkpoint_path_for(path),
        fingerprint=LedgerFingerprint.from_stat(
            ledger_stat,
            ledger_sha256,
        ),
        record_count=len(indexed_records),
        latest_appended_at=(
            None
            if not indexed_records
            else max(record.appended_at for record in indexed_records)
        ),
        records=indexed_records,
    )
    if publish:
        try:
            _publish_checkpoint(checkpoint)
        except (CheckpointError, OSError):
            # Publication is acceleration only; the validated in-memory index is usable.
            pass
    return checkpoint


def _read_descriptor(descriptor: int, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < expected_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, expected_size - offset),
            offset,
        )
        if not chunk:
            raise CheckpointError("ledger changed while scanning raw offsets")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, expected_size):
        raise CheckpointError("ledger changed while scanning raw offsets")
    return b"".join(chunks)


def _validate_checkpoint_rows_from_path(
    checkpoint: LedgerCheckpoint,
    ledger_stat: os.stat_result,
    ledger_sha256: str,
) -> None:
    checkpoint_name = checkpoint.path.name
    if not checkpoint_name.endswith(_CHECKPOINT_SUFFIX):
        raise CheckpointError("checkpoint path does not identify its ledger")
    ledger_name = checkpoint_name[: -len(_CHECKPOINT_SUFFIX)]
    if not ledger_name:
        raise CheckpointError("checkpoint path does not identify its ledger")
    parent_descriptor = _open_parent_directory(checkpoint.path.parent)
    try:
        ledger_descriptor = _open_regular_at(
            parent_descriptor,
            ledger_name,
            "ledger",
        )
        ledger_locked = False
        try:
            fcntl.flock(ledger_descriptor, fcntl.LOCK_SH)
            ledger_locked = True
            actual_stat, actual_sha256 = ledger_state(ledger_descriptor)
            if LedgerFingerprint.from_stat(
                actual_stat,
                actual_sha256,
            ) != LedgerFingerprint.from_stat(ledger_stat, ledger_sha256):
                raise CheckpointError(
                    "ledger changed while validating checkpoint rows"
                )
            _validate_checkpoint_rows(checkpoint, ledger_descriptor)
        finally:
            if ledger_locked:
                fcntl.flock(ledger_descriptor, fcntl.LOCK_UN)
            os.close(ledger_descriptor)
    finally:
        os.close(parent_descriptor)


def _validate_checkpoint_rows(
    checkpoint: LedgerCheckpoint,
    ledger_descriptor: int,
) -> None:
    for line_number, indexed in enumerate(checkpoint.records, start=1):
        raw_line = os.pread(
            ledger_descriptor,
            indexed.byte_length,
            indexed.byte_offset,
        )
        if (
            len(raw_line) != indexed.byte_length
            or not raw_line.endswith(b"\n")
        ):
            raise CheckpointError(
                f"checkpoint row {line_number} does not resolve to one ledger row"
            )
        fields = _extract_index_fields(raw_line[:-1], line_number)
        _validate_indexed_row(indexed, fields, line_number)


def _extract_index_fields(
    raw_row: bytes,
    line_number: int,
) -> dict[str, object]:
    try:
        value = json.loads(
            raw_row.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(value, dict):
            raise CheckpointError(
                f"checkpoint row {line_number} must be a JSON object"
            )
        return {
            key: item
            for key, item in value.items()
            if key in _ROW_INDEX_FIELDS
        }
    except (
        CheckpointError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ) as error:
        raise CheckpointError(
            f"checkpoint row {line_number} cannot be indexed safely"
        ) from error


def _validate_indexed_row(
    indexed: IndexedRecord,
    fields: dict[str, object],
    line_number: int,
) -> None:
    envelope = _require_exact_mapping(
        fields.get("envelope"),
        frozenset(
            (
                "record_id",
                "record_kind",
                "idempotency_key",
                "payload_hash",
                "appended_at",
            )
        ),
        f"checkpoint row {line_number}.envelope",
    )
    actual_envelope = (
        envelope["record_kind"],
        envelope["record_id"],
        envelope["idempotency_key"],
        envelope["payload_hash"],
        _parse_datetime(
            envelope["appended_at"],
            f"checkpoint row {line_number}.envelope.appended_at",
        ),
    )
    expected_envelope = (
        indexed.record_kind,
        indexed.record_id,
        indexed.idempotency_key,
        indexed.payload_hash,
        indexed.appended_at,
    )
    if actual_envelope != expected_envelope:
        raise CheckpointError(
            f"checkpoint row {line_number} envelope metadata differs from ledger"
        )

    if indexed.record_kind == "capture":
        actual_metadata = (
            fields.get("capture_id"),
            fields.get("game_pk"),
            _parse_datetime(
                fields.get("captured_at"),
                f"checkpoint row {line_number}.captured_at",
            ),
            _parse_datetime(
                fields.get("first_pitch"),
                f"checkpoint row {line_number}.first_pitch",
            ),
            fields.get("normalized_input_hash"),
            fields.get("supersedes_capture_id"),
        )
        expected_metadata = (
            indexed.capture_id,
            indexed.game_pk,
            indexed.captured_at,
            indexed.first_pitch,
            indexed.normalized_input_hash,
            indexed.supersedes_capture_id,
        )
    elif indexed.record_kind == "forecast":
        actual_metadata = (
            fields.get("forecast_run_id"),
            fields.get("normalized_input_hash"),
        )
        expected_metadata = (
            indexed.forecast_run_id,
            indexed.normalized_input_hash,
        )
    elif indexed.record_kind == "capture_forecast_link":
        actual_metadata = (
            fields.get("capture_id"),
            fields.get("forecast_run_id"),
        )
        expected_metadata = (
            indexed.capture_id,
            indexed.forecast_run_id,
        )
    else:
        return
    if actual_metadata != expected_metadata:
        raise CheckpointError(
            f"checkpoint row {line_number} relationship metadata differs from ledger"
        )


def _index_records(
    raw_content: bytes,
    records: Sequence[LedgerRecord],
) -> tuple[IndexedRecord, ...]:
    if raw_content and not raw_content.endswith(b"\n"):
        raise CheckpointError("ledger must end with an ASCII LF")
    raw_lines = (
        []
        if not raw_content
        else [
            line + b"\n"
            for line in raw_content[:-1].split(b"\n")
        ]
    )
    if len(raw_lines) != len(records):
        raise CheckpointError(
            "raw ledger row count differs from strictly validated records"
        )
    indexed: list[IndexedRecord] = []
    byte_offset = 0
    for line_number, (raw_line, record) in enumerate(
        zip(raw_lines, records),
        start=1,
    ):
        if not raw_line.endswith(b"\n"):
            raise CheckpointError(
                f"raw ledger row {line_number} is not newline terminated"
            )
        try:
            row = json.loads(
                raw_line[:-1].decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(row, dict):
                raise CheckpointError(
                    f"raw ledger row {line_number} must be a JSON object"
                )
            envelope = _require_exact_mapping(
                row.get("envelope"),
                frozenset(
                    (
                        "record_id",
                        "record_kind",
                        "idempotency_key",
                        "payload_hash",
                        "appended_at",
                    )
                ),
                f"raw ledger row {line_number}.envelope",
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointError(
                f"raw ledger row {line_number} is not valid JSON"
            ) from error
        expected_envelope = record.envelope
        actual_identity = (
            envelope["record_id"],
            envelope["record_kind"],
            envelope["idempotency_key"],
            envelope["payload_hash"],
            _parse_datetime(
                envelope["appended_at"],
                f"raw ledger row {line_number}.envelope.appended_at",
            ),
        )
        expected_identity = (
            expected_envelope.record_id,
            expected_envelope.record_kind,
            expected_envelope.idempotency_key,
            expected_envelope.payload_hash,
            expected_envelope.appended_at,
        )
        if actual_identity != expected_identity:
            raise CheckpointError(
                f"raw ledger row {line_number} disagrees with validated record"
            )
        indexed.append(
            _indexed_record(
                record,
                byte_offset=byte_offset,
                byte_length=len(raw_line),
            )
        )
        byte_offset += len(raw_line)
    return tuple(indexed)


def _indexed_record(
    record: LedgerRecord,
    *,
    byte_offset: int,
    byte_length: int,
) -> IndexedRecord:
    envelope = record.envelope
    values: dict[str, object] = {
        "record_kind": envelope.record_kind,
        "record_id": envelope.record_id,
        "idempotency_key": envelope.idempotency_key,
        "payload_hash": envelope.payload_hash,
        "appended_at": envelope.appended_at,
        "byte_offset": byte_offset,
        "byte_length": byte_length,
        "identifier": envelope.record_id,
    }
    if isinstance(record, CaptureRecord):
        values.update(
            identifier=record.capture_id,
            game_pk=record.game_pk,
            captured_at=record.captured_at,
            first_pitch=record.first_pitch,
            normalized_input_hash=record.normalized_input_hash,
            supersedes_capture_id=record.supersedes_capture_id,
            capture_id=record.capture_id,
        )
    elif isinstance(record, ForecastRecord):
        values.update(
            identifier=record.forecast_run_id,
            normalized_input_hash=record.normalized_input_hash,
            forecast_run_id=record.forecast_run_id,
        )
    elif isinstance(record, CaptureForecastLink):
        values.update(
            capture_id=record.capture_id,
            forecast_run_id=record.forecast_run_id,
        )
    elif not isinstance(
        record,
        (MarketObservationRecord, OfficialResultRecord, ReviewRecord),
    ):
        raise CheckpointError("validated records contain an unknown record type")
    return IndexedRecord(**values)


def _checkpoint_from_json(path: Path, value: object) -> LedgerCheckpoint:
    root = _require_exact_mapping(
        value,
        frozenset(
            (
                "schema_version",
                "fingerprint",
                "record_count",
                "latest_appended_at",
                "records",
            )
        ),
        "checkpoint",
    )
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != _CHECKPOINT_SCHEMA_VERSION
    ):
        raise CheckpointError("checkpoint schema_version is unsupported")
    fingerprint_value = _require_exact_mapping(
        root["fingerprint"],
        frozenset(("device", "inode", "size", "mtime_ns", "sha256")),
        "checkpoint.fingerprint",
    )
    fingerprint = LedgerFingerprint(
        device=fingerprint_value["device"],
        inode=fingerprint_value["inode"],
        size=fingerprint_value["size"],
        mtime_ns=fingerprint_value["mtime_ns"],
        sha256=fingerprint_value["sha256"],
    )
    records_value = root["records"]
    if not isinstance(records_value, list):
        raise CheckpointError("checkpoint.records must be a JSON array")
    records = tuple(
        _indexed_record_from_json(item, index)
        for index, item in enumerate(records_value)
    )
    return LedgerCheckpoint(
        path=path,
        fingerprint=fingerprint,
        record_count=_require_integer(
            root["record_count"],
            "checkpoint.record_count",
        ),
        latest_appended_at=_parse_optional_datetime(
            root["latest_appended_at"],
            "checkpoint.latest_appended_at",
        ),
        records=records,
    )


def _indexed_record_from_json(value: object, index: int) -> IndexedRecord:
    field_name = f"checkpoint.records[{index}]"
    item = _require_exact_mapping(
        value,
        frozenset(
            (
                "record_kind",
                "record_id",
                "idempotency_key",
                "payload_hash",
                "appended_at",
                "byte_offset",
                "byte_length",
                "identifier",
                "metadata",
            )
        ),
        field_name,
    )
    record_kind = item["record_kind"]
    if (
        not isinstance(record_kind, str)
        or record_kind not in _KNOWN_RECORD_KINDS
    ):
        raise CheckpointError(f"{field_name}.record_kind is unknown")
    metadata_keys = {
        "capture": _CAPTURE_METADATA_KEYS,
        "forecast": _FORECAST_METADATA_KEYS,
        "capture_forecast_link": _LINK_METADATA_KEYS,
        "market_observation": _EMPTY_METADATA_KEYS,
        "official_result": _EMPTY_METADATA_KEYS,
        "review": _EMPTY_METADATA_KEYS,
    }[record_kind]
    metadata = _require_exact_mapping(
        item["metadata"],
        metadata_keys,
        f"{field_name}.metadata",
    )
    values: dict[str, object] = {
        "record_kind": record_kind,
        "record_id": item["record_id"],
        "idempotency_key": item["idempotency_key"],
        "payload_hash": item["payload_hash"],
        "appended_at": _parse_datetime(
            item["appended_at"],
            f"{field_name}.appended_at",
        ),
        "byte_offset": item["byte_offset"],
        "byte_length": item["byte_length"],
        "identifier": item["identifier"],
    }
    if record_kind == "capture":
        values.update(
            capture_id=metadata["capture_id"],
            game_pk=metadata["game_pk"],
            captured_at=_parse_datetime(
                metadata["captured_at"],
                f"{field_name}.metadata.captured_at",
            ),
            first_pitch=_parse_datetime(
                metadata["first_pitch"],
                f"{field_name}.metadata.first_pitch",
            ),
            normalized_input_hash=metadata["normalized_input_hash"],
            supersedes_capture_id=metadata["supersedes_capture_id"],
        )
    elif record_kind == "forecast":
        values.update(
            forecast_run_id=metadata["forecast_run_id"],
            normalized_input_hash=metadata["normalized_input_hash"],
        )
    elif record_kind == "capture_forecast_link":
        values.update(
            capture_id=metadata["capture_id"],
            forecast_run_id=metadata["forecast_run_id"],
        )
    return IndexedRecord(**values)


def _checkpoint_json(checkpoint: LedgerCheckpoint) -> dict[str, object]:
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": {
            "device": checkpoint.fingerprint.device,
            "inode": checkpoint.fingerprint.inode,
            "size": checkpoint.fingerprint.size,
            "mtime_ns": checkpoint.fingerprint.mtime_ns,
            "sha256": checkpoint.fingerprint.sha256,
        },
        "record_count": checkpoint.record_count,
        "latest_appended_at": (
            None
            if checkpoint.latest_appended_at is None
            else checkpoint.latest_appended_at.isoformat()
        ),
        "records": [
            _indexed_record_json(record) for record in checkpoint.records
        ],
    }


def _indexed_record_json(record: IndexedRecord) -> dict[str, object]:
    metadata: dict[str, object]
    if record.record_kind == "capture":
        metadata = {
            "capture_id": record.capture_id,
            "game_pk": record.game_pk,
            "captured_at": (
                None
                if record.captured_at is None
                else record.captured_at.isoformat()
            ),
            "first_pitch": (
                None
                if record.first_pitch is None
                else record.first_pitch.isoformat()
            ),
            "normalized_input_hash": record.normalized_input_hash,
            "supersedes_capture_id": record.supersedes_capture_id,
        }
    elif record.record_kind == "forecast":
        metadata = {
            "forecast_run_id": record.forecast_run_id,
            "normalized_input_hash": record.normalized_input_hash,
        }
    elif record.record_kind == "capture_forecast_link":
        metadata = {
            "capture_id": record.capture_id,
            "forecast_run_id": record.forecast_run_id,
        }
    else:
        metadata = {}
    return {
        "record_kind": record.record_kind,
        "record_id": record.record_id,
        "idempotency_key": record.idempotency_key,
        "payload_hash": record.payload_hash,
        "appended_at": record.appended_at.isoformat(),
        "byte_offset": record.byte_offset,
        "byte_length": record.byte_length,
        "identifier": record.identifier,
        "metadata": metadata,
    }


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_parent_directory(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointError("checkpoint parent cannot be opened safely") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise CheckpointError(
                "checkpoint parent changed while being opened"
            )
    except OSError as error:
        os.close(descriptor)
        raise CheckpointError(
            "checkpoint parent changed while being opened"
        ) from error
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _validate_parent_directory(path: Path, descriptor: int) -> None:
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
    except OSError as error:
        raise CheckpointError("checkpoint parent changed") from error
    if (
        not stat.S_ISDIR(descriptor_stat.st_mode)
        or not stat.S_ISDIR(path_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise CheckpointError("checkpoint parent changed")


def _open_regular_at(
    parent_descriptor: int,
    name: str,
    label: str,
) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise CheckpointError(f"{label} must not be a symlink") from error
        raise CheckpointError(f"{label} cannot be opened") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(path_stat.st_mode):
            raise CheckpointError(f"{label} must not be a symlink")
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise CheckpointError(f"{label} must be one regular file")
    except OSError as error:
        os.close(descriptor)
        raise CheckpointError(f"{label} changed while being opened") from error
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _read_checkpoint_file(path: Path) -> bytes:
    parent_descriptor = _open_parent_directory(path.parent)
    try:
        descriptor = _open_regular_at(
            parent_descriptor,
            path.name,
            "checkpoint",
        )
        try:
            try:
                before = os.fstat(descriptor)
                with os.fdopen(descriptor, "rb") as checkpoint_file:
                    descriptor = -1
                    content = checkpoint_file.read()
                    after = os.fstat(checkpoint_file.fileno())
                path_stat = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                    )
                    != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    )
                    or (after.st_dev, after.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)
                ):
                    raise CheckpointError("checkpoint changed while being read")
                _validate_parent_directory(path.parent, parent_descriptor)
                return content
            except OSError as error:
                raise CheckpointError(
                    "checkpoint changed while being read"
                ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _publish_checkpoint(checkpoint: LedgerCheckpoint) -> None:
    path = checkpoint.path
    parent_descriptor = _open_parent_directory(path.parent)
    temp_name = (
        f".{path.name}.{secrets.token_hex(12)}.tmp"
    )
    temp_descriptor = -1
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        temp_descriptor = os.open(
            temp_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        content = (
            canonical_json(_checkpoint_json(checkpoint)).encode("utf-8")
            + b"\n"
        )
        with os.fdopen(temp_descriptor, "wb") as checkpoint_file:
            temp_descriptor = -1
            checkpoint_file.write(content)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        _validate_parent_directory(path.parent, parent_descriptor)
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        published = True
        os.fsync(parent_descriptor)
    finally:
        if temp_descriptor >= 0:
            os.close(temp_descriptor)
        if not published:
            try:
                os.unlink(temp_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


__all__ = (
    "CheckpointError",
    "IndexedRecord",
    "LedgerCheckpoint",
    "LedgerFingerprint",
    "build_checkpoint",
    "checkpoint_path_for",
)
