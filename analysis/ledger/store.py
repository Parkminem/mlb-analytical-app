"""Process-safe append-only JSONL persistence for ledger records."""

from collections.abc import Mapping
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable, TypeVar
import uuid
import weakref

from analysis.forecasting.identity import canonical_json

from .records import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
    MarketObservationCommand,
    MarketObservationRecord,
    OfficialResultCommand,
    OfficialResultRecord,
    RecordEnvelope,
    ReviewCommand,
    ReviewRecord,
    freeze_mapping,
)


_RecordT = TypeVar(
    "_RecordT",
    CaptureRecord,
    ForecastRecord,
    CaptureForecastLink,
    MarketObservationRecord,
    OfficialResultRecord,
    ReviewRecord,
)


class IdempotencyConflict(ValueError):
    """Raised when an idempotency key is reused for a different payload."""


class LedgerCorruptionError(ValueError):
    """Raised when authoritative JSONL cannot be safely reconstructed."""


class LedgerReferenceError(ValueError):
    """Raised when an append references a missing or mismatched record."""


def _payload_hash(command: object) -> str:
    encoded = canonical_json(command).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO datetime string")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _parse_optional_datetime(
    value: object,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value, field_name)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number: {value}")


def _command_for_capture(record: CaptureRecord) -> CaptureCommand:
    return CaptureCommand(
        game_pk=record.game_pk,
        captured_at=record.captured_at,
        first_pitch=record.first_pitch,
        lineup_state=record.lineup_state,
        out_of_sample_eligible=record.out_of_sample_eligible,
        normalized_input_hash=record.normalized_input_hash,
        payload=record.payload,
        supersedes_capture_id=record.supersedes_capture_id,
    )


def _command_for_forecast(record: ForecastRecord) -> ForecastCommand:
    return ForecastCommand(
        forecast_run_id=record.forecast_run_id,
        model_version=record.model_version,
        simulation_config_hash=record.simulation_config_hash,
        normalized_input_hash=record.normalized_input_hash,
        outputs=record.outputs,
    )


def _command_for_market(
    record: MarketObservationRecord,
) -> MarketObservationCommand:
    return MarketObservationCommand(
        game_pk=record.game_pk,
        source_name=record.source_name,
        source_record=record.source_record,
        source_revision=record.source_revision,
        observation_kind=record.observation_kind,
        market_key=record.market_key,
        market_type=record.market_type,
        settlement_format=record.settlement_format,
        selected_side=record.selected_side,
        line=record.line,
        selected_decimal_price=record.selected_decimal_price,
        opposing_decimal_price=record.opposing_decimal_price,
        observed_at=record.observed_at,
        available_at=record.available_at,
        supersedes_record_id=record.supersedes_record_id,
    )


def _command_for_result(record: OfficialResultRecord) -> OfficialResultCommand:
    return OfficialResultCommand(
        game_pk=record.game_pk,
        source_name=record.source_name,
        source_record=record.source_record,
        source_revision=record.source_revision,
        finalized_at=record.finalized_at,
        away_f5_runs=record.away_f5_runs,
        home_f5_runs=record.home_f5_runs,
        away_full_runs=record.away_full_runs,
        home_full_runs=record.home_full_runs,
        component_values=record.component_values,
        documented_events=record.documented_events,
        supersedes_record_id=record.supersedes_record_id,
    )


def _command_for_review(record: ReviewRecord) -> ReviewCommand:
    return ReviewCommand(
        capture_forecast_link_record_id=(
            record.capture_forecast_link_record_id
        ),
        capture_id=record.capture_id,
        forecast_run_id=record.forecast_run_id,
        official_result_record_id=record.official_result_record_id,
        market_key=record.market_key,
        market_result=record.market_result,
        classification=record.classification,
        inside_80_ratio=record.inside_80_ratio,
        contradicted_paths=record.contradicted_paths,
        component_reviews=record.component_reviews,
        qualifying_exogenous_source_observation_ids=(
            record.qualifying_exogenous_source_observation_ids
        ),
    )


def _logical_payload(record: _RecordT) -> object:
    if isinstance(record, CaptureRecord):
        return _command_for_capture(record)
    if isinstance(record, ForecastRecord):
        return _command_for_forecast(record)
    if isinstance(record, CaptureForecastLink):
        return {
            "capture_id": record.capture_id,
            "forecast_run_id": record.forecast_run_id,
        }
    if isinstance(record, MarketObservationRecord):
        return _command_for_market(record)
    if isinstance(record, OfficialResultRecord):
        return _command_for_result(record)
    if isinstance(record, ReviewRecord):
        return _command_for_review(record)
    raise TypeError(f"unsupported ledger record: {type(record).__name__}")


def _deserialize_record(value: object) -> _RecordT:
    if not isinstance(value, dict):
        raise ValueError("ledger row must be a JSON object")
    envelope_value = value.get("envelope")
    if not isinstance(envelope_value, dict):
        raise ValueError("ledger row envelope must be a JSON object")
    envelope_data = dict(envelope_value)
    envelope_data["appended_at"] = _parse_datetime(
        envelope_data.get("appended_at"),
        "appended_at",
    )
    envelope = RecordEnvelope(**envelope_data)
    record_data = dict(value)
    record_data["envelope"] = envelope
    if envelope.record_kind == "capture":
        record_data["captured_at"] = _parse_datetime(
            record_data.get("captured_at"),
            "captured_at",
        )
        record_data["first_pitch"] = _parse_datetime(
            record_data.get("first_pitch"),
            "first_pitch",
        )
        return CaptureRecord(**record_data)
    if envelope.record_kind == "forecast":
        return ForecastRecord(**record_data)
    if envelope.record_kind == "capture_forecast_link":
        return CaptureForecastLink(**record_data)
    if envelope.record_kind == "market_observation":
        record_data["observed_at"] = _parse_datetime(
            record_data.get("observed_at"),
            "observed_at",
        )
        record_data["available_at"] = _parse_optional_datetime(
            record_data.get("available_at"),
            "available_at",
        )
        return MarketObservationRecord(**record_data)
    if envelope.record_kind == "official_result":
        record_data["finalized_at"] = _parse_datetime(
            record_data.get("finalized_at"),
            "finalized_at",
        )
        return OfficialResultRecord(**record_data)
    if envelope.record_kind == "review":
        return ReviewRecord(**record_data)
    raise ValueError(f"unknown record kind: {envelope.record_kind}")


def _review_relationship_error(
    command: ReviewCommand,
    records: tuple[_RecordT, ...] | list[_RecordT],
) -> str | None:
    link = next(
        (
            record
            for record in records
            if isinstance(record, CaptureForecastLink)
            and record.envelope.record_id
            == command.capture_forecast_link_record_id
        ),
        None,
    )
    if link is None:
        return "capture_forecast_link_record_id must reference an existing link"
    if link.capture_id != command.capture_id:
        return "review capture_id must match the referenced link"
    if link.forecast_run_id != command.forecast_run_id:
        return "review forecast_run_id must match the referenced link"

    capture = next(
        (
            record
            for record in records
            if isinstance(record, CaptureRecord)
            and record.capture_id == command.capture_id
        ),
        None,
    )
    if capture is None:
        return "review capture_id must reference an existing capture"
    forecast = next(
        (
            record
            for record in records
            if isinstance(record, ForecastRecord)
            and record.forecast_run_id == command.forecast_run_id
        ),
        None,
    )
    if forecast is None:
        return "review forecast_run_id must reference an existing forecast"
    result = next(
        (
            record
            for record in records
            if isinstance(record, OfficialResultRecord)
            and record.envelope.record_id
            == command.official_result_record_id
        ),
        None,
    )
    if result is None:
        return "official_result_record_id must reference an existing result"
    if result.game_pk != capture.game_pk:
        return "review capture and official result must concern the same game"
    if result.finalized_at <= capture.captured_at:
        return (
            "official result finalized_at must be strictly after captured_at"
        )
    if result.finalized_at <= capture.first_pitch:
        return (
            "official result finalized_at must be strictly after first_pitch"
        )
    game_results = tuple(
        record
        for record in records
        if isinstance(record, OfficialResultRecord)
        and record.game_pk == result.game_pk
    )
    superseded_result_ids = frozenset(
        record.supersedes_record_id
        for record in game_results
        if record.supersedes_record_id is not None
    )
    latest_results = tuple(
        record
        for record in game_results
        if record.envelope.record_id not in superseded_result_ids
    )
    if (
        len(latest_results) != 1
        or latest_results[0].envelope.record_id
        != command.official_result_record_id
    ):
        return (
            "official_result_record_id must reference the current latest "
            "official result"
        )

    matching_markets = [
        record
        for record in records
        if isinstance(record, MarketObservationRecord)
        and record.market_key == command.market_key
    ]
    if not matching_markets:
        return "review market_key must reference an existing market observation"
    if not any(record.game_pk == capture.game_pk for record in matching_markets):
        return "review market observation must concern the capture game"
    return None


class LedgerStore:
    """A JSONL ledger whose file is the sole authoritative state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._parent_path = self.path.parent
        self._ledger_name = self.path.name
        self._lock_name = self.lock_path.name
        self._parent_descriptor = self._open_parent_directory()
        parent_stat = os.fstat(self._parent_descriptor)
        self._parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        self._parent_descriptor_finalizer = weakref.finalize(
            self,
            os.close,
            self._parent_descriptor,
        )
        try:
            self._reject_existing_hard_links()
            descriptor = self._open_no_follow(
                self._lock_name,
                os.O_CREAT | os.O_RDWR,
                "ledger lock",
            )
            os.close(descriptor)
        except Exception:
            self._parent_descriptor_finalizer()
            raise

    def _open_parent_directory(self) -> int:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._parent_path, flags)
        except OSError as error:
            raise LedgerCorruptionError(
                "ledger parent directory cannot be opened safely"
            ) from error
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = self._parent_path.lstat()
            if (
                not stat.S_ISDIR(descriptor_stat.st_mode)
                or not stat.S_ISDIR(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise LedgerCorruptionError(
                    "ledger parent directory changed while being opened"
                )
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _validate_parent_identity(self) -> None:
        try:
            descriptor_stat = os.fstat(self._parent_descriptor)
            path_stat = self._parent_path.lstat()
        except OSError as error:
            raise LedgerCorruptionError(
                "ledger parent directory changed after store construction"
            ) from error
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != self._parent_identity
            or (path_stat.st_dev, path_stat.st_ino) != self._parent_identity
        ):
            raise LedgerCorruptionError(
                "ledger parent directory changed after store construction"
            )

    @staticmethod
    def _reject_hard_links(link_count: int) -> None:
        if link_count > 1:
            raise LedgerCorruptionError(
                "ledger file has multiple hard links; path aliases are unsafe"
            )

    def _reject_existing_hard_links(self) -> None:
        try:
            descriptor = self._open_no_follow(
                self._ledger_name,
                os.O_RDONLY,
                "ledger",
            )
        except FileNotFoundError:
            return
        os.close(descriptor)

    def _open_no_follow(
        self,
        file_name: str,
        flags: int,
        path_label: str,
    ) -> int:
        self._validate_parent_identity()
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                file_name,
                flags | no_follow,
                0o600,
                dir_fd=self._parent_descriptor,
            )
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.EMLINK):
                raise LedgerCorruptionError(
                    f"{path_label} path must not be a symlink"
                ) from error
            raise
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(
                file_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(path_stat.st_mode):
                raise LedgerCorruptionError(
                    f"{path_label} path must not be a symlink"
                )
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise LedgerCorruptionError(
                    f"{path_label} path must reference a regular file"
                )
            if (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ) != (path_stat.st_dev, path_stat.st_ino):
                raise LedgerCorruptionError(
                    f"{path_label} path changed while being opened"
                )
            self._reject_hard_links(descriptor_stat.st_nlink)
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _read_records(self) -> tuple[_RecordT, ...]:
        try:
            descriptor = self._open_no_follow(
                self._ledger_name,
                os.O_RDONLY,
                "ledger",
            )
            with os.fdopen(descriptor, "rb") as ledger:
                raw_content = ledger.read()
        except FileNotFoundError:
            return ()
        try:
            content = raw_content.decode("utf-8")
        except UnicodeError as error:
            raise LedgerCorruptionError(
                f"ledger is not valid UTF-8: {error}"
            ) from error
        if raw_content and not raw_content.endswith(b"\n"):
            raise LedgerCorruptionError(
                "ledger must end with a newline byte (ASCII LF)"
            )

        records: list[_RecordT] = []
        record_ids: set[str] = set()
        capture_ids: set[str] = set()
        forecast_run_ids: set[str] = set()
        review_ids: set[str] = set()
        idempotency_keys: set[tuple[str, str]] = set()
        capture_supersession_targets: set[str] = set()
        market_supersession_targets: set[str] = set()
        captures_by_id: dict[str, CaptureRecord] = {}
        forecasts_by_id: dict[str, ForecastRecord] = {}
        markets_by_record_id: dict[str, MarketObservationRecord] = {}
        results_by_game: dict[int, list[OfficialResultRecord]] = {}
        result_revision_keys: set[tuple[int, str, str]] = set()
        for line_number, line in enumerate(content.split("\n")[:-1], start=1):
            if not line:
                raise LedgerCorruptionError(f"empty ledger row at line {line_number}")
            if line.endswith("\r"):
                raise LedgerCorruptionError(
                    f"ledger row {line_number} uses CRLF framing; ASCII LF is required"
                )
            try:
                record = _deserialize_record(
                    json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_json_constant,
                    )
                )
                if record.envelope.record_id in record_ids:
                    raise ValueError("duplicate record_id")
                index_key = (
                    record.envelope.record_kind,
                    record.envelope.idempotency_key,
                )
                if index_key in idempotency_keys:
                    raise ValueError("duplicate record idempotency key")
                if _payload_hash(_logical_payload(record)) != record.envelope.payload_hash:
                    raise ValueError("record payload hash mismatch")
                if isinstance(record, CaptureRecord):
                    if record.capture_id in capture_ids:
                        raise ValueError("duplicate capture_id")
                    if record.supersedes_capture_id is not None:
                        superseded = captures_by_id.get(record.supersedes_capture_id)
                        if superseded is None:
                            raise ValueError("capture supersedes missing capture")
                        if superseded.game_pk != record.game_pk:
                            raise ValueError("capture supersedes a different game")
                        if superseded.capture_id == record.capture_id:
                            raise ValueError("capture cannot supersede itself")
                        if (
                            record.supersedes_capture_id
                            in capture_supersession_targets
                        ):
                            raise ValueError(
                                "capture supersession target already has a successor"
                            )
                elif isinstance(record, ForecastRecord):
                    if record.forecast_run_id in forecast_run_ids:
                        raise ValueError("duplicate forecast_run_id")
                elif isinstance(record, CaptureForecastLink):
                    capture = captures_by_id.get(record.capture_id)
                    if capture is None:
                        raise ValueError("link references missing capture")
                    forecast = forecasts_by_id.get(record.forecast_run_id)
                    if forecast is None:
                        raise ValueError("link references missing forecast")
                    if (
                        capture.normalized_input_hash
                        != forecast.normalized_input_hash
                    ):
                        raise ValueError(
                            "link capture and forecast normalized_input_hash must match"
                        )
                elif isinstance(record, MarketObservationRecord):
                    if record.supersedes_record_id is not None:
                        superseded = markets_by_record_id.get(
                            record.supersedes_record_id
                        )
                        if superseded is None:
                            raise ValueError("market supersedes missing observation")
                        if (
                            superseded.game_pk != record.game_pk
                            or superseded.source_name != record.source_name
                            or superseded.source_record != record.source_record
                            or superseded.observation_kind != record.observation_kind
                            or superseded.market_key != record.market_key
                        ):
                            raise ValueError(
                                "market supersession must stay in one source market"
                            )
                        if (
                            record.supersedes_record_id
                            in market_supersession_targets
                        ):
                            raise ValueError(
                                "market supersession target already has a successor"
                            )
                elif isinstance(record, OfficialResultRecord):
                    revision_key = (
                        record.game_pk,
                        record.source_name,
                        record.source_revision,
                    )
                    if revision_key in result_revision_keys:
                        raise ValueError("duplicate official result source revision")
                    prior_results = results_by_game.get(record.game_pk, [])
                    if not prior_results:
                        if record.supersedes_record_id is not None:
                            raise ValueError(
                                "first official result cannot supersede a record"
                            )
                    elif (
                        record.supersedes_record_id
                        != prior_results[-1].envelope.record_id
                    ):
                        raise ValueError(
                            "official result must supersede the latest result"
                        )
                elif isinstance(record, ReviewRecord):
                    if record.review_id in review_ids:
                        raise ValueError("duplicate review_id")
                    relationship_error = _review_relationship_error(
                        _command_for_review(record),
                        records,
                    )
                    if relationship_error is not None:
                        raise ValueError(relationship_error)
            except (
                KeyError,
                RecursionError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise LedgerCorruptionError(
                    f"invalid ledger row at line {line_number}: {error}"
                ) from error
            records.append(record)
            record_ids.add(record.envelope.record_id)
            idempotency_keys.add(index_key)
            if isinstance(record, CaptureRecord):
                capture_ids.add(record.capture_id)
                captures_by_id[record.capture_id] = record
                if record.supersedes_capture_id is not None:
                    capture_supersession_targets.add(
                        record.supersedes_capture_id
                    )
            elif isinstance(record, ForecastRecord):
                forecast_run_ids.add(record.forecast_run_id)
                forecasts_by_id[record.forecast_run_id] = record
            elif isinstance(record, MarketObservationRecord):
                markets_by_record_id[record.envelope.record_id] = record
                if record.supersedes_record_id is not None:
                    market_supersession_targets.add(record.supersedes_record_id)
            elif isinstance(record, OfficialResultRecord):
                results_by_game.setdefault(record.game_pk, []).append(record)
                result_revision_keys.add(
                    (
                        record.game_pk,
                        record.source_name,
                        record.source_revision,
                    )
                )
            elif isinstance(record, ReviewRecord):
                review_ids.add(record.review_id)
        return tuple(records)

    def _append(
        self,
        record_kind: str,
        idempotency_key: str,
        logical_payload: object,
        validate: Callable[[tuple[_RecordT, ...]], None],
        build: Callable[[RecordEnvelope], _RecordT],
    ) -> _RecordT:
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        lock_descriptor = self._open_no_follow(
            self._lock_name,
            os.O_RDWR,
            "ledger lock",
        )
        with os.fdopen(lock_descriptor, "r+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_records()
                logical_hash = _payload_hash(logical_payload)
                for existing in records:
                    if (
                        existing.envelope.record_kind != record_kind
                        or existing.envelope.idempotency_key != idempotency_key
                    ):
                        continue
                    if existing.envelope.payload_hash == logical_hash:
                        return existing
                    raise IdempotencyConflict(
                        "idempotency key already used for another logical "
                        f"{record_kind} payload: {idempotency_key}"
                    )

                validate(records)
                envelope = RecordEnvelope(
                    record_id=uuid.uuid4().hex,
                    record_kind=record_kind,
                    idempotency_key=idempotency_key,
                    payload_hash=logical_hash,
                    appended_at=datetime.now(timezone.utc),
                )
                record = build(envelope)
                descriptor = self._open_no_follow(
                    self._ledger_name,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    "ledger",
                )
                with os.fdopen(
                    descriptor,
                    "a",
                    encoding="utf-8",
                    newline="",
                ) as ledger:
                    ledger.write(f"{canonical_json(record)}\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                return record
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def append_capture(
        self,
        idempotency_key: str,
        command: CaptureCommand,
    ) -> CaptureRecord:
        if not isinstance(command, CaptureCommand):
            raise TypeError("command must be CaptureCommand")

        def validate(records: tuple[_RecordT, ...]) -> None:
            captures_by_id = {
                record.capture_id: record
                for record in records
                if isinstance(record, CaptureRecord)
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

        def build(envelope: RecordEnvelope) -> CaptureRecord:
            capture_id = uuid.uuid4().hex
            while capture_id == command.supersedes_capture_id:
                capture_id = uuid.uuid4().hex
            return CaptureRecord(
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
            )

        record = self._append(
            "capture",
            idempotency_key,
            command,
            validate,
            build,
        )
        if not isinstance(record, CaptureRecord):
            raise LedgerCorruptionError("capture idempotency resolved to wrong kind")
        return record

    def append_forecast(self, command: ForecastCommand) -> ForecastRecord:
        if not isinstance(command, ForecastCommand):
            raise TypeError("command must be ForecastCommand")

        def build(envelope: RecordEnvelope) -> ForecastRecord:
            return ForecastRecord(
                envelope=envelope,
                forecast_run_id=command.forecast_run_id,
                model_version=command.model_version,
                simulation_config_hash=command.simulation_config_hash,
                normalized_input_hash=command.normalized_input_hash,
                outputs=command.outputs,
            )

        record = self._append(
            "forecast",
            f"forecast:{command.forecast_run_id}",
            command,
            lambda records: None,
            build,
        )
        if not isinstance(record, ForecastRecord):
            raise LedgerCorruptionError("forecast idempotency resolved to wrong kind")
        return record

    def link_capture_forecast(
        self,
        capture_id: str,
        forecast_run_id: str,
    ) -> CaptureForecastLink:
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise ValueError("capture_id must be a non-empty string")
        if not isinstance(forecast_run_id, str) or not forecast_run_id.strip():
            raise ValueError("forecast_run_id must be a non-empty string")
        logical_payload = {
            "capture_id": capture_id,
            "forecast_run_id": forecast_run_id,
        }

        def validate(records: tuple[_RecordT, ...]) -> None:
            captures_by_id = {
                record.capture_id: record
                for record in records
                if isinstance(record, CaptureRecord)
            }
            forecasts_by_id = {
                record.forecast_run_id: record
                for record in records
                if isinstance(record, ForecastRecord)
            }
            capture = captures_by_id.get(capture_id)
            if capture is None:
                raise LedgerReferenceError("capture_id must reference an existing capture")
            forecast = forecasts_by_id.get(forecast_run_id)
            if forecast is None:
                raise LedgerReferenceError(
                    "forecast_run_id must reference an existing forecast"
                )
            if capture.normalized_input_hash != forecast.normalized_input_hash:
                raise LedgerReferenceError(
                    "capture and forecast normalized_input_hash must match"
                )

        def build(envelope: RecordEnvelope) -> CaptureForecastLink:
            return CaptureForecastLink(
                envelope=envelope,
                capture_id=capture_id,
                forecast_run_id=forecast_run_id,
            )

        record = self._append(
            "capture_forecast_link",
            f"capture-forecast:{capture_id}:{forecast_run_id}",
            logical_payload,
            validate,
            build,
        )
        if not isinstance(record, CaptureForecastLink):
            raise LedgerCorruptionError("link idempotency resolved to wrong kind")
        return record

    def append_market_observation(
        self,
        command: MarketObservationCommand,
        idempotency_key: str | None = None,
    ) -> MarketObservationRecord:
        if not isinstance(command, MarketObservationCommand):
            raise TypeError("command must be MarketObservationCommand")
        resolved_key = (
            f"market:{command.source_name}:{command.source_record}:"
            f"{command.source_revision}:{command.observation_kind}:"
            f"{command.market_key}"
            if idempotency_key is None
            else idempotency_key
        )

        def validate(records: tuple[_RecordT, ...]) -> None:
            if command.supersedes_record_id is None:
                return
            superseded = next(
                (
                    record
                    for record in records
                    if isinstance(record, MarketObservationRecord)
                    and record.envelope.record_id == command.supersedes_record_id
                ),
                None,
            )
            if superseded is None:
                raise LedgerReferenceError(
                    "supersedes_record_id must reference a market observation"
                )
            if (
                superseded.game_pk != command.game_pk
                or superseded.source_name != command.source_name
                or superseded.source_record != command.source_record
                or superseded.observation_kind != command.observation_kind
                or superseded.market_key != command.market_key
            ):
                raise LedgerReferenceError(
                    "market supersession must stay in one source market"
                )
            if any(
                isinstance(record, MarketObservationRecord)
                and record.supersedes_record_id == command.supersedes_record_id
                for record in records
            ):
                raise LedgerReferenceError(
                    "market supersession target already has a successor"
                )

        def build(envelope: RecordEnvelope) -> MarketObservationRecord:
            return MarketObservationRecord(
                envelope=envelope,
                game_pk=command.game_pk,
                source_name=command.source_name,
                source_record=command.source_record,
                source_revision=command.source_revision,
                observation_kind=command.observation_kind,
                market_key=command.market_key,
                market_type=command.market_type,
                settlement_format=command.settlement_format,
                selected_side=command.selected_side,
                line=command.line,
                selected_decimal_price=command.selected_decimal_price,
                opposing_decimal_price=command.opposing_decimal_price,
                observed_at=command.observed_at,
                available_at=command.available_at,
                supersedes_record_id=command.supersedes_record_id,
            )

        record = self._append(
            "market_observation",
            resolved_key,
            command,
            validate,
            build,
        )
        if not isinstance(record, MarketObservationRecord):
            raise LedgerCorruptionError(
                "market idempotency resolved to wrong record kind"
            )
        return record

    def append_official_result(
        self,
        command: OfficialResultCommand,
        idempotency_key: str | None = None,
    ) -> OfficialResultRecord:
        if not isinstance(command, OfficialResultCommand):
            raise TypeError("command must be OfficialResultCommand")
        resolved_key = (
            f"result:{command.source_name}:{command.game_pk}:"
            f"{command.source_revision}"
            if idempotency_key is None
            else idempotency_key
        )

        def validate(records: tuple[_RecordT, ...]) -> None:
            if any(
                isinstance(record, OfficialResultRecord)
                and record.game_pk == command.game_pk
                and record.source_name == command.source_name
                and record.source_revision == command.source_revision
                for record in records
            ):
                raise LedgerReferenceError(
                    "official result source revision already exists"
                )
            prior_results = [
                record
                for record in records
                if isinstance(record, OfficialResultRecord)
                and record.game_pk == command.game_pk
            ]
            if not prior_results:
                if command.supersedes_record_id is not None:
                    raise LedgerReferenceError(
                        "first official result for a game cannot supersede"
                    )
                return
            latest = prior_results[-1]
            if command.supersedes_record_id != latest.envelope.record_id:
                raise LedgerReferenceError(
                    "official result correction must supersede the latest result"
                )

        def build(envelope: RecordEnvelope) -> OfficialResultRecord:
            return OfficialResultRecord(
                envelope=envelope,
                game_pk=command.game_pk,
                source_name=command.source_name,
                source_record=command.source_record,
                source_revision=command.source_revision,
                finalized_at=command.finalized_at,
                away_f5_runs=command.away_f5_runs,
                home_f5_runs=command.home_f5_runs,
                away_full_runs=command.away_full_runs,
                home_full_runs=command.home_full_runs,
                component_values=command.component_values,
                documented_events=command.documented_events,
                supersedes_record_id=command.supersedes_record_id,
            )

        record = self._append(
            "official_result",
            resolved_key,
            command,
            validate,
            build,
        )
        if not isinstance(record, OfficialResultRecord):
            raise LedgerCorruptionError(
                "result idempotency resolved to wrong record kind"
            )
        return record

    def append_review(
        self,
        command: ReviewCommand,
        idempotency_key: str | None = None,
    ) -> ReviewRecord:
        if not isinstance(command, ReviewCommand):
            raise TypeError("command must be ReviewCommand")
        resolved_key = (
            f"review:{command.capture_id}:{command.forecast_run_id}:"
            f"{command.official_result_record_id}:{command.market_key}"
            if idempotency_key is None
            else idempotency_key
        )

        def validate(records: tuple[_RecordT, ...]) -> None:
            relationship_error = _review_relationship_error(command, records)
            if relationship_error is not None:
                raise LedgerReferenceError(relationship_error)

        def build(envelope: RecordEnvelope) -> ReviewRecord:
            return ReviewRecord(
                envelope=envelope,
                review_id=uuid.uuid4().hex,
                capture_forecast_link_record_id=(
                    command.capture_forecast_link_record_id
                ),
                capture_id=command.capture_id,
                forecast_run_id=command.forecast_run_id,
                official_result_record_id=command.official_result_record_id,
                market_key=command.market_key,
                market_result=command.market_result,
                classification=command.classification,
                inside_80_ratio=command.inside_80_ratio,
                contradicted_paths=command.contradicted_paths,
                component_reviews=command.component_reviews,
                qualifying_exogenous_source_observation_ids=(
                    command.qualifying_exogenous_source_observation_ids
                ),
            )

        record = self._append(
            "review",
            resolved_key,
            command,
            validate,
            build,
        )
        if not isinstance(record, ReviewRecord):
            raise LedgerCorruptionError(
                "review idempotency resolved to wrong record kind"
            )
        return record

    def market_observations_match(
        self,
        entry: MarketObservationRecord,
        close: MarketObservationRecord,
    ) -> bool:
        if not isinstance(entry, MarketObservationRecord):
            raise TypeError("entry must be MarketObservationRecord")
        if not isinstance(close, MarketObservationRecord):
            raise TypeError("close must be MarketObservationRecord")
        return (
            entry.observation_kind == "entry"
            and close.observation_kind == "close"
            and entry.game_pk == close.game_pk
            and entry.market_type == close.market_type
            and entry.settlement_format == close.settlement_format
            and entry.selected_side == close.selected_side
            and entry.line == close.line
        )

    def records_superseded_by(self, kind: str) -> Mapping[str, object]:
        valid_kinds = {
            "capture",
            "forecast",
            "capture_forecast_link",
            "market_observation",
            "official_result",
            "review",
        }
        if kind not in valid_kinds:
            raise ValueError(f"unknown record kind: {kind}")
        records = tuple(self.iter_records())
        captures_by_id = {
            record.capture_id: record
            for record in records
            if isinstance(record, CaptureRecord)
        }
        relationships: dict[str, object] = {}
        for record in records:
            if record.envelope.record_kind != kind:
                continue
            if isinstance(record, CaptureRecord):
                if record.supersedes_capture_id is None:
                    continue
                old_record = captures_by_id[record.supersedes_capture_id]
                relationships[old_record.envelope.record_id] = (
                    record.envelope.record_id
                )
            elif isinstance(record, (MarketObservationRecord, OfficialResultRecord)):
                if record.supersedes_record_id is not None:
                    relationships[record.supersedes_record_id] = (
                        record.envelope.record_id
                    )
        return freeze_mapping(relationships, "supersession relationships")

    def latest_official_result(
        self,
        game_pk: int,
    ) -> OfficialResultRecord | None:
        results = [
            record
            for record in self.iter_records("official_result")
            if isinstance(record, OfficialResultRecord) and record.game_pk == game_pk
        ]
        if not results:
            return None
        superseded_ids = {
            record.supersedes_record_id
            for record in results
            if record.supersedes_record_id is not None
        }
        latest = [
            record
            for record in results
            if record.envelope.record_id not in superseded_ids
        ]
        if len(latest) != 1:
            raise LedgerCorruptionError(
                "official result corrections do not have one latest record"
            )
        return latest[0]

    def links_for_capture(self, capture_id: str) -> tuple[CaptureForecastLink, ...]:
        return tuple(
            record
            for record in self.iter_records("capture_forecast_link")
            if isinstance(record, CaptureForecastLink)
            and record.capture_id == capture_id
        )

    def forecast_for_link(self, link_record_id: str) -> ForecastRecord:
        records = tuple(self.iter_records())
        link = next(
            (
                record
                for record in records
                if isinstance(record, CaptureForecastLink)
                and record.envelope.record_id == link_record_id
            ),
            None,
        )
        if link is None:
            raise LedgerReferenceError(
                "link_record_id must reference an existing capture-forecast link"
            )
        forecast = next(
            (
                record
                for record in records
                if isinstance(record, ForecastRecord)
                and record.forecast_run_id == link.forecast_run_id
            ),
            None,
        )
        if forecast is None:
            raise LedgerCorruptionError("link references a missing forecast")
        return forecast

    def iter_records(self, kind: str | None = None):
        lock_descriptor = self._open_no_follow(
            self._lock_name,
            os.O_CREAT | os.O_RDWR,
            "ledger lock",
        )
        with os.fdopen(lock_descriptor, "r+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                records = self._read_records()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        if kind is None:
            return iter(records)
        return (record for record in records if record.envelope.record_kind == kind)


__all__ = (
    "IdempotencyConflict",
    "LedgerCorruptionError",
    "LedgerReferenceError",
    "LedgerStore",
)
