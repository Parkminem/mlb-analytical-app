"""Deterministic, read-only coverage audit for excluded shadow recommendations."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Final

from analysis.forecasting.engine import SemanticMarketInput
from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType, SettlementFormat
from analysis.forecasting.market import settle_observed_market
from analysis.ledger import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    OfficialResultRecord,
    RecordEnvelope,
    linked_shadow_comparison,
)
from analysis.ledger.store import (
    LedgerCorruptionError,
    _deserialize_record,
    _forecast_markets_match_capture,
    _logical_payload,
    _parse_datetime,
    _payload_hash,
    _reject_duplicate_keys,
    _reject_json_constant,
)
from analysis.review.calibration import (
    CalibrationGroup,
    CalibrationMetrics,
    CalibrationReport,
)
from analysis.shadow import deserialize_shadow_comparison


_LEDGER_RECORD_KINDS: Final = frozenset(
    (
        "capture",
        "forecast",
        "capture_forecast_link",
        "market_observation",
        "official_result",
        "review",
    )
)
_LEDGER_ENVELOPE_KEYS: Final = frozenset(
    (
        "record_id",
        "record_kind",
        "idempotency_key",
        "payload_hash",
        "appended_at",
    )
)
_FORECAST_RECORD_KEYS: Final = frozenset(
    (
        "envelope",
        "forecast_run_id",
        "model_version",
        "simulation_config_hash",
        "normalized_input_hash",
        "outputs",
    )
)
_MANIFEST_KEYS: Final = frozenset(("schema_version", "audit_id", "entries"))
_ENTRY_KEYS: Final = frozenset(
    (
        "game_pk",
        "capture_id",
        "forecast_run_id",
        "capture_forecast_link_record_id",
        "market_key",
    )
)
_CALIBRATION_KEYS: Final = frozenset(
    (
        "schema_version",
        "ledger_as_of",
        "ledger_path",
        "ledger_revision",
        "migration",
        "overall",
        "group_by",
        "groups",
    )
)
_LEDGER_REVISION_KEYS: Final = frozenset(("record_count", "state_hash"))
_GROUP_KEYS: Final = frozenset(("dimensions", "metrics"))
_MIGRATION_KEYS: Final = frozenset(("confirmed_games", "market_types"))
_PROGRESS_KEYS: Final = frozenset(
    (
        "count",
        "threshold",
        "completed",
        "remaining",
        "progress",
        "threshold_met",
    )
)
_OVERALL_METRIC_KEYS: Final = frozenset(
    (
        "sample_size",
        "wins",
        "losses",
        "pushes",
        "binary_sample_size",
        "multiclass_sample_size",
        "priced_stake_count",
        "probability_clv_sample_size",
        "price_clv_sample_size",
        "hit_rate",
        "push_rate",
        "binary_brier",
        "multiclass_brier",
        "log_loss",
        "profit_units",
        "roi",
        "mean_probability_clv",
        "mean_price_clv",
    )
)
_SEMANTIC_MARKET_KEYS: Final = frozenset(
    (
        "market_key",
        "market_type",
        "settlement_format",
        "selected_side",
        "line",
        "selected_decimal_price",
        "opposing_decimal_price",
    )
)
_RECOMMENDED_RATINGS: Final = frozenset(("play", "lean"))
_INPUT_ERROR_EXIT_CODE: Final = 2
_CANONICAL_JSON_ENCODER: Final = json.JSONEncoder(
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    keys = frozenset(value)
    if keys != expected:
        missing = ", ".join(sorted(expected - keys))
        unknown = ", ".join(sorted(keys - expected))
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise ValueError(f"{name} has invalid fields: {'; '.join(parts)}")


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist") from error
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    return _require_mapping(value, name)


def _load_manifest(path: Path) -> tuple[str, tuple[Mapping[str, object], ...]]:
    manifest = _load_json(path, "manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if (
        _require_positive_integer(
            manifest["schema_version"],
            "manifest schema_version",
        )
        != 1
    ):
        raise ValueError("manifest schema_version must be 1")
    audit_id = _require_nonempty_string(manifest["audit_id"], "audit_id")
    entries_value = manifest["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise ValueError("manifest entries must be a nonempty array")
    entries: list[Mapping[str, object]] = []
    seen: set[tuple[int, str, str, str, str]] = set()
    for index, entry_value in enumerate(entries_value):
        entry = _require_mapping(entry_value, f"manifest entries[{index}]")
        _require_exact_keys(entry, _ENTRY_KEYS, f"manifest entries[{index}]")
        game_pk = _require_positive_integer(entry["game_pk"], "game_pk")
        key = (
            game_pk,
            _require_nonempty_string(entry["capture_id"], "capture_id"),
            _require_nonempty_string(entry["forecast_run_id"], "forecast_run_id"),
            _require_nonempty_string(
                entry["capture_forecast_link_record_id"],
                "capture_forecast_link_record_id",
            ),
            _require_nonempty_string(entry["market_key"], "market_key"),
        )
        if key in seen:
            raise ValueError("manifest entries must be unique")
        seen.add(key)
        entries.append(entry)
    return audit_id, tuple(entries)


def _calibration_metrics(value: object, name: str) -> CalibrationMetrics:
    metrics = _require_mapping(value, name)
    _require_exact_keys(metrics, _OVERALL_METRIC_KEYS, name)
    try:
        return CalibrationMetrics(**dict(metrics))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid: {error}") from error


def _calibration_groups(
    value: object,
    group_by: tuple[str, ...],
) -> tuple[CalibrationGroup, ...]:
    if not isinstance(value, list):
        raise ValueError("groups must be a JSON array")
    groups: list[CalibrationGroup] = []
    for index, group_value in enumerate(value):
        group = _require_mapping(group_value, f"groups[{index}]")
        _require_exact_keys(group, _GROUP_KEYS, f"groups[{index}]")
        dimensions = _require_mapping(
            group["dimensions"],
            f"groups[{index}].dimensions",
        )
        _require_exact_keys(
            dimensions,
            frozenset(group_by),
            f"groups[{index}].dimensions",
        )
        try:
            groups.append(
                CalibrationGroup(
                    # JSON object ordering is not semantic; group_by is authoritative.
                    dimensions=tuple((name, dimensions[name]) for name in group_by),
                    metrics=_calibration_metrics(
                        group["metrics"],
                        f"groups[{index}].metrics",
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"groups[{index}] is invalid: {error}") from error
    return tuple(groups)


def _validate_progress(value: object, name: str) -> None:
    progress = _require_mapping(value, name)
    _require_exact_keys(progress, _PROGRESS_KEYS, name)
    count = _require_nonnegative_integer(progress["count"], f"{name}.count")
    threshold = _require_positive_integer(
        progress["threshold"],
        f"{name}.threshold",
    )
    completed = _require_nonnegative_integer(
        progress["completed"],
        f"{name}.completed",
    )
    remaining = _require_nonnegative_integer(
        progress["remaining"],
        f"{name}.remaining",
    )
    if completed != min(count, threshold):
        raise ValueError(f"{name}.completed must equal min(count, threshold)")
    if remaining != max(threshold - count, 0):
        raise ValueError(f"{name}.remaining must equal max(threshold - count, 0)")
    if progress["threshold_met"] is not (count >= threshold):
        raise ValueError(f"{name}.threshold_met must match count and threshold")
    expected_progress = completed / threshold
    if not math.isclose(
        _require_finite_number(progress["progress"], f"{name}.progress"),
        expected_progress,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{name}.progress must equal completed divided by threshold"
        )


def _validate_migration(value: object) -> None:
    migration = _require_mapping(value, "migration")
    _require_exact_keys(migration, _MIGRATION_KEYS, "migration")
    _validate_progress(migration["confirmed_games"], "migration.confirmed_games")
    market_types = _require_mapping(
        migration["market_types"],
        "migration.market_types",
    )
    expected_market_types = frozenset(
        market_type.value for market_type in MarketType
    )
    _require_exact_keys(
        market_types,
        expected_market_types,
        "migration.market_types",
    )
    for market_type in sorted(expected_market_types):
        _validate_progress(
            market_types[market_type],
            f"migration.market_types.{market_type}",
        )


def _load_calibration(path: Path) -> Mapping[str, object]:
    calibration = _load_json(path, "calibration")
    _require_exact_keys(calibration, _CALIBRATION_KEYS, "calibration")
    if _require_positive_integer(
        calibration["schema_version"],
        "calibration schema_version",
    ) != 1:
        raise ValueError("calibration schema_version must be 1")
    ledger_as_of = calibration["ledger_as_of"]
    if ledger_as_of is not None:
        _parse_datetime(ledger_as_of, "ledger_as_of")
    _require_nonempty_string(calibration["ledger_path"], "ledger_path")
    revision = _require_mapping(calibration["ledger_revision"], "ledger_revision")
    _require_exact_keys(revision, _LEDGER_REVISION_KEYS, "ledger_revision")
    _require_nonnegative_integer(revision["record_count"], "record_count")
    _require_nonempty_string(revision["state_hash"], "state_hash")
    overall = _calibration_metrics(calibration["overall"], "overall metrics")
    group_by_value = calibration["group_by"]
    if not isinstance(group_by_value, list):
        raise ValueError("group_by must be a JSON array")
    if any(not isinstance(dimension, str) for dimension in group_by_value):
        raise ValueError("group_by must contain strings")
    group_by = tuple(group_by_value)
    groups = _calibration_groups(calibration["groups"], group_by)
    try:
        CalibrationReport(
            overall=overall,
            groups=groups,
            group_by=group_by,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"calibration grouping is invalid: {error}") from error
    _validate_migration(calibration["migration"])
    return calibration


def _ledger_envelope(
    value: object,
    line_number: int,
) -> tuple[Mapping[str, object], RecordEnvelope]:
    row = _require_mapping(value, f"ledger row {line_number}")
    envelope_value = _require_mapping(
        row.get("envelope"),
        f"ledger row {line_number} envelope",
    )
    _require_exact_keys(
        envelope_value,
        _LEDGER_ENVELOPE_KEYS,
        f"ledger row {line_number} envelope",
    )
    envelope_data = dict(envelope_value)
    envelope_data["appended_at"] = _parse_datetime(
        envelope_data["appended_at"],
        "appended_at",
    )
    envelope = RecordEnvelope(**envelope_data)
    if envelope.record_kind not in _LEDGER_RECORD_KINDS:
        raise ValueError(f"unknown record kind: {envelope.record_kind}")
    return row, envelope


def _validated_record(
    value: object,
    line_number: int,
) -> object:
    try:
        record = _deserialize_record(value)
        if _payload_hash(_logical_payload(record)) != record.envelope.payload_hash:
            raise ValueError("record payload hash mismatch")
        return record
    except (KeyError, RecursionError, TypeError, ValueError) as error:
        raise LedgerCorruptionError(
            f"invalid ledger row at line {line_number}: {error}"
        ) from error


def _canonical_json_chunks(value: object):
    """Stream canonical JSON for JSON-decoded values without copying large rows."""

    def validate(item: object) -> None:
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("canonical floats must be finite")
            if item.is_integer():
                raise ValueError("canonical JSON must not contain integral floats")
        elif isinstance(item, Mapping):
            for key, nested_item in item.items():
                if not isinstance(key, str):
                    raise TypeError("canonical mapping keys must be strings")
                validate(nested_item)
        elif isinstance(item, list):
            for nested_item in item:
                validate(nested_item)

    validate(value)
    yield from _CANONICAL_JSON_ENCODER.iterencode(value)


def _has_canonical_json_framing(raw_line: bytes, value: object) -> bool:
    offset = 0
    for chunk in _canonical_json_chunks(value):
        encoded = chunk.encode("utf-8")
        if raw_line[offset : offset + len(encoded)] != encoded:
            return False
        offset += len(encoded)
    return raw_line[offset:] == b"\n"


def _payload_hash_from_canonical_chunks(value: object) -> str:
    digest = hashlib.sha256()
    for chunk in _canonical_json_chunks(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _validate_unselected_forecast(
    row: Mapping[str, object],
    envelope: RecordEnvelope,
    line_number: int,
) -> None:
    _require_exact_keys(
        row,
        _FORECAST_RECORD_KEYS,
        f"ledger row {line_number} forecast",
    )
    for field_name in (
        "forecast_run_id",
        "model_version",
        "simulation_config_hash",
        "normalized_input_hash",
    ):
        _require_nonempty_string(
            row[field_name],
            f"ledger row {line_number} forecast.{field_name}",
        )
    _require_mapping(
        row["outputs"],
        f"ledger row {line_number} forecast.outputs",
    )
    logical_payload = {
        field_name: row[field_name]
        for field_name in _FORECAST_RECORD_KEYS - {"envelope"}
    }
    if _payload_hash_from_canonical_chunks(logical_payload) != envelope.payload_hash:
        raise ValueError("record payload hash mismatch")


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ledger appended_at must be timezone-aware")
    normalized = value.astimezone(timezone.utc).isoformat()
    return f"{normalized[:-6]}Z"


def _streamed_ledger_revision() -> tuple[
    Callable[[RecordEnvelope], None],
    Callable[[], tuple[Mapping[str, object], str | None]],
]:
    """Accumulate reporting's canonical ledger revision without retaining rows."""

    state_hash = hashlib.sha256()
    state_hash.update(b"[")
    record_count = 0
    latest_appended_at: datetime | None = None

    def append(envelope: RecordEnvelope) -> None:
        nonlocal record_count, latest_appended_at
        if record_count:
            state_hash.update(b",")
        state_hash.update(
            canonical_json(
                {
                    "record_kind": envelope.record_kind,
                    "record_id": envelope.record_id,
                    "idempotency_key": envelope.idempotency_key,
                    "payload_hash": envelope.payload_hash,
                    "appended_at": _utc_iso(envelope.appended_at),
                }
            ).encode("utf-8")
        )
        record_count += 1
        if latest_appended_at is None or envelope.appended_at > latest_appended_at:
            latest_appended_at = envelope.appended_at

    def finish() -> tuple[Mapping[str, object], str | None]:
        state_hash.update(b"]")
        return (
            {
                "record_count": record_count,
                "state_hash": state_hash.hexdigest(),
            },
            None if latest_appended_at is None else _utc_iso(latest_appended_at),
        )

    return append, finish


def _slim_forecast(
    forecast: ForecastRecord,
    market_keys: frozenset[str],
) -> ForecastRecord:
    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    market_inputs = _require_mapping(
        outputs.get("market_inputs"),
        "forecast market_inputs",
    )
    slim_outputs: dict[str, object] = {
        "market_inputs": {
            market_key: market_inputs[market_key]
            for market_key in market_keys
            if market_key in market_inputs
        }
    }
    if "shadow_comparison" in outputs:
        slim_outputs["shadow_comparison"] = outputs["shadow_comparison"]
    return ForecastRecord(
        envelope=forecast.envelope,
        forecast_run_id=forecast.forecast_run_id,
        model_version=forecast.model_version,
        simulation_config_hash=forecast.simulation_config_hash,
        normalized_input_hash=forecast.normalized_input_hash,
        outputs=slim_outputs,
    )


def _resolve_ledger_records(
    ledger_path: Path,
    manifest_entries: Sequence[Mapping[str, object]],
) -> tuple[
    Mapping[str, CaptureRecord],
    Mapping[str, ForecastRecord],
    Mapping[str, CaptureForecastLink],
    Mapping[int, OfficialResultRecord],
    Mapping[str, object],
    str | None,
]:
    required_capture_ids = frozenset(
        _require_nonempty_string(entry["capture_id"], "capture_id")
        for entry in manifest_entries
    )
    required_forecast_ids = frozenset(
        _require_nonempty_string(entry["forecast_run_id"], "forecast_run_id")
        for entry in manifest_entries
    )
    required_link_ids = frozenset(
        _require_nonempty_string(
            entry["capture_forecast_link_record_id"],
            "capture_forecast_link_record_id",
        )
        for entry in manifest_entries
    )
    required_game_pks = frozenset(
        _require_positive_integer(entry["game_pk"], "game_pk")
        for entry in manifest_entries
    )
    market_keys_by_forecast: dict[str, set[str]] = {}
    capture_ids_by_forecast: dict[str, set[str]] = {}
    forecast_ids_by_capture: dict[str, set[str]] = {}
    for entry in manifest_entries:
        capture_id = _require_nonempty_string(entry["capture_id"], "capture_id")
        forecast_run_id = _require_nonempty_string(
            entry["forecast_run_id"],
            "forecast_run_id",
        )
        market_keys_by_forecast.setdefault(forecast_run_id, set()).add(
            _require_nonempty_string(entry["market_key"], "market_key")
        )
        capture_ids_by_forecast.setdefault(forecast_run_id, set()).add(capture_id)
        forecast_ids_by_capture.setdefault(capture_id, set()).add(forecast_run_id)

    captures: dict[str, CaptureRecord] = {}
    forecasts: dict[str, ForecastRecord] = {}
    pending_forecasts: dict[str, ForecastRecord] = {}
    links: dict[str, CaptureForecastLink] = {}
    latest_results: dict[int, OfficialResultRecord] = {}
    append_revision, finish_revision = _streamed_ledger_revision()

    def retain_bound_forecast(forecast_run_id: str) -> None:
        forecast = pending_forecasts.get(forecast_run_id)
        if forecast is None:
            return
        capture_ids = capture_ids_by_forecast[forecast_run_id]
        if not all(capture_id in captures for capture_id in capture_ids):
            return
        for capture_id in capture_ids:
            capture = captures[capture_id]
            if capture.normalized_input_hash != forecast.normalized_input_hash:
                raise LedgerCorruptionError(
                    "capture and forecast normalized_input_hash must match"
                )
            if not _forecast_markets_match_capture(capture, forecast):
                raise LedgerCorruptionError(
                    "forecast market_inputs must exactly match canonical "
                    "capture markets"
                )
        forecasts[forecast_run_id] = _slim_forecast(
            forecast,
            frozenset(market_keys_by_forecast[forecast_run_id]),
        )
        del pending_forecasts[forecast_run_id]
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(ledger_path, flags)
    except FileNotFoundError as error:
        raise ValueError(f"ledger does not exist: {ledger_path}") from error
    except OSError as error:
        raise LedgerCorruptionError(
            f"ledger cannot be opened read-only: {ledger_path}"
        ) from error

    with os.fdopen(descriptor, "rb") as ledger:
        for line_number, raw_line in enumerate(ledger, start=1):
            if not raw_line.endswith(b"\n"):
                raise LedgerCorruptionError(
                    "ledger must end with a newline byte (ASCII LF)"
                )
            if raw_line.endswith(b"\r\n"):
                raise LedgerCorruptionError(
                    f"ledger row {line_number} uses CRLF framing; ASCII LF is required"
                )
            try:
                line = raw_line[:-1].decode("utf-8")
                if not line:
                    raise ValueError("empty ledger row")
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
                row, envelope = _ledger_envelope(value, line_number)
                if not _has_canonical_json_framing(raw_line, value):
                    raise ValueError("ledger row must use canonical JSON")
                if (
                    envelope.record_kind == "forecast"
                    and row.get("forecast_run_id") not in required_forecast_ids
                ):
                    _validate_unselected_forecast(row, envelope, line_number)
                    append_revision(envelope)
                    continue
                record = _validated_record(value, line_number)
                append_revision(envelope)
            except LedgerCorruptionError:
                raise
            except (
                KeyError,
                RecursionError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise LedgerCorruptionError(
                    f"invalid ledger row at line {line_number}: {error}"
                ) from error

            if (
                isinstance(record, CaptureRecord)
                and record.capture_id in required_capture_ids
            ):
                if record.capture_id in captures:
                    raise LedgerCorruptionError("duplicate required capture_id")
                captures[record.capture_id] = record
                for forecast_run_id in forecast_ids_by_capture[record.capture_id]:
                    retain_bound_forecast(forecast_run_id)
            elif (
                isinstance(record, ForecastRecord)
                and record.forecast_run_id in required_forecast_ids
            ):
                if (
                    record.forecast_run_id in forecasts
                    or record.forecast_run_id in pending_forecasts
                ):
                    raise LedgerCorruptionError("duplicate required forecast_run_id")
                pending_forecasts[record.forecast_run_id] = record
                retain_bound_forecast(record.forecast_run_id)
            elif (
                isinstance(record, CaptureForecastLink)
                and record.envelope.record_id in required_link_ids
            ):
                if record.envelope.record_id in links:
                    raise LedgerCorruptionError("duplicate required link record_id")
                links[record.envelope.record_id] = record
            elif (
                isinstance(record, OfficialResultRecord)
                and record.game_pk in required_game_pks
            ):
                previous = latest_results.get(record.game_pk)
                if previous is None:
                    if record.supersedes_record_id is not None:
                        raise LedgerCorruptionError(
                            "first official result cannot supersede a record"
                        )
                elif (
                    record.supersedes_record_id
                    != previous.envelope.record_id
                ):
                    raise LedgerCorruptionError(
                        "official result must supersede the latest result"
                    )
                latest_results[record.game_pk] = record
    for forecast_run_id in tuple(pending_forecasts):
        retain_bound_forecast(forecast_run_id)
    revision, ledger_as_of = finish_revision()
    return captures, forecasts, links, latest_results, revision, ledger_as_of


def _validate_calibration_ledger_revision(
    calibration: Mapping[str, object],
    ledger_path: Path,
    ledger_revision: Mapping[str, object],
    ledger_as_of: str | None,
) -> None:
    if calibration["ledger_path"] != str(ledger_path.expanduser().resolve(strict=False)):
        raise ValueError("calibration ledger_path does not match ledger")
    calibration_revision = _require_mapping(
        calibration["ledger_revision"],
        "ledger_revision",
    )
    for field_name in ("record_count", "state_hash"):
        if calibration_revision[field_name] != ledger_revision[field_name]:
            raise ValueError(f"calibration {field_name} does not match ledger")
    if calibration["ledger_as_of"] != ledger_as_of:
        raise ValueError("calibration ledger_as_of does not match ledger")


def _semantic_market(forecast: ForecastRecord, market_key: str) -> SemanticMarketInput:
    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    inputs = _require_mapping(outputs.get("market_inputs"), "forecast market_inputs")
    payload = _require_mapping(inputs.get(market_key), f"market_inputs[{market_key}]")
    _require_exact_keys(payload, _SEMANTIC_MARKET_KEYS, f"market_inputs[{market_key}]")
    if payload["market_key"] != market_key:
        raise ValueError("market input key must match market_key")
    return SemanticMarketInput(
        market_key=market_key,
        market_type=MarketType(_require_nonempty_string(payload["market_type"], "market_type")),
        settlement_format=SettlementFormat(
            _require_nonempty_string(payload["settlement_format"], "settlement_format")
        ),
        selected_side=_require_nonempty_string(payload["selected_side"], "selected_side"),
        line=payload["line"],
        selected_decimal_price=payload["selected_decimal_price"],
        opposing_decimal_price=payload["opposing_decimal_price"],
    )


def _audit_entry(
    entry: Mapping[str, object],
    captures: Mapping[str, CaptureRecord],
    forecasts: Mapping[str, ForecastRecord],
    links: Mapping[str, CaptureForecastLink],
    latest_results: Mapping[int, OfficialResultRecord],
) -> Mapping[str, object]:
    game_pk = _require_positive_integer(entry["game_pk"], "game_pk")
    capture_id = _require_nonempty_string(entry["capture_id"], "capture_id")
    forecast_run_id = _require_nonempty_string(entry["forecast_run_id"], "forecast_run_id")
    link_record_id = _require_nonempty_string(
        entry["capture_forecast_link_record_id"],
        "capture_forecast_link_record_id",
    )
    market_key = _require_nonempty_string(entry["market_key"], "market_key")
    capture = captures.get(capture_id)
    forecast = forecasts.get(forecast_run_id)
    link = links.get(link_record_id)
    if capture is None or capture.game_pk != game_pk:
        raise ValueError(f"manifest capture does not resolve for {market_key}")
    if forecast is None:
        raise ValueError(f"manifest forecast does not resolve for {market_key}")
    if link is None or (
        link.capture_id != capture_id or link.forecast_run_id != forecast_run_id
    ):
        raise ValueError(f"manifest link relationship does not resolve for {market_key}")
    comparison = deserialize_shadow_comparison(linked_shadow_comparison(link, forecast))
    if comparison.forecast_run_id != forecast_run_id:
        raise ValueError(f"linked comparison forecast does not match for {market_key}")
    markets = [
        market for market in comparison.market_comparisons if market.market_key == market_key
    ]
    if len(markets) != 1:
        raise ValueError(f"named market comparison does not resolve for {market_key}")
    market_comparison = markets[0]
    if market_comparison.legacy_rating not in _RECOMMENDED_RATINGS:
        raise ValueError(f"named market is not recommended: {market_key}")
    if market_comparison.review_eligible:
        raise ValueError(f"named market is not review-ineligible: {market_key}")
    market = _semantic_market(forecast, market_key)
    result = latest_results.get(game_pk)
    payload: dict[str, object] = {
        "game_pk": game_pk,
        "capture_id": capture_id,
        "forecast_run_id": forecast_run_id,
        "capture_forecast_link_record_id": link_record_id,
        "market_key": market_key,
        "legacy_rating": market_comparison.legacy_rating,
        "ineligibility_flags": list(market_comparison.review_ineligibility_flags),
        "market_type": market.market_type.value,
        "settlement_format": market.settlement_format.value,
        "selected_side": market.selected_side,
        "line": market.line,
        "selected_decimal_price": market.selected_decimal_price,
    }
    if result is None:
        payload.update(
            {
                "status": "unresolved",
                "official_result_record_id": None,
                "market_result": None,
                "profit_units": None,
            }
        )
        return payload
    market_result = settle_observed_market(market, result)
    price = market.selected_decimal_price
    profit = (
        None
        if price is None
        else {"win": price - 1.0, "loss": -1.0, "push": 0.0}[market_result]
    )
    payload.update(
        {
            "status": "settled",
            "official_result_record_id": result.envelope.record_id,
            "market_result": market_result,
            "profit_units": profit,
        }
    )
    return payload


def _audit_only_metrics(
    official_metrics: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    wins = _require_nonnegative_integer(official_metrics["wins"], "official wins")
    losses = _require_nonnegative_integer(official_metrics["losses"], "official losses")
    pushes = _require_nonnegative_integer(official_metrics["pushes"], "official pushes")
    sample_size = _require_nonnegative_integer(
        official_metrics["sample_size"],
        "official sample_size",
    )
    priced = _require_nonnegative_integer(
        official_metrics["priced_stake_count"],
        "official priced_stake_count",
    )
    profit = official_metrics.get("profit_units")
    if profit is not None:
        profit = _require_finite_number(profit, "official profit_units")
    for entry in entries:
        if entry["status"] != "settled":
            continue
        sample_size += 1
        result = entry["market_result"]
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "push":
            pushes += 1
        else:
            raise ValueError("settled audit entry has an invalid market_result")
        entry_profit = entry["profit_units"]
        if entry_profit is not None:
            priced += 1
            profit = (0.0 if profit is None else profit) + _require_finite_number(
                entry_profit,
                "entry profit_units",
            )
    return {
        "sample_size": sample_size,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "priced_stake_count": priced,
        "profit_units": profit,
        "roi": None if profit is None or priced == 0 else profit / priced,
    }


def build_coverage_audit(
    ledger_path: Path,
    calibration_path: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    """Build a deterministic audit without modifying ledger or calibration inputs."""

    if not all(isinstance(path, Path) for path in (ledger_path, calibration_path, manifest_path)):
        raise TypeError("ledger_path, calibration_path, and manifest_path must be Paths")
    audit_id, manifest_entries = _load_manifest(manifest_path)
    calibration = _load_calibration(calibration_path)
    (
        captures,
        forecasts,
        links,
        latest_results,
        ledger_revision,
        ledger_as_of,
    ) = _resolve_ledger_records(
        ledger_path,
        manifest_entries,
    )
    _validate_calibration_ledger_revision(
        calibration,
        ledger_path,
        ledger_revision,
        ledger_as_of,
    )
    entries = tuple(
        _audit_entry(entry, captures, forecasts, links, latest_results)
        for entry in manifest_entries
    )
    official_metrics = dict(_require_mapping(calibration["overall"], "overall metrics"))
    return {
        "schema_version": 1,
        "audit_id": audit_id,
        "official_calibration_unchanged": True,
        "official_ledger_revision": dict(
            _require_mapping(calibration["ledger_revision"], "ledger_revision")
        ),
        "official_metrics": official_metrics,
        "audit_only_metrics": _audit_only_metrics(official_metrics, entries),
        "resolved_omission_count": sum(entry["status"] == "settled" for entry in entries),
        "unresolved_omission_count": sum(entry["status"] == "unresolved" for entry in entries),
        "entries": list(entries),
    }


def render_coverage_audit_markdown(audit: Mapping[str, object]) -> str:
    """Render the stable Korean operator summary for an audit payload."""

    audit_id = _require_nonempty_string(audit.get("audit_id"), "audit_id")
    official = _require_mapping(audit.get("official_metrics"), "official_metrics")
    adjusted = _require_mapping(audit.get("audit_only_metrics"), "audit_only_metrics")
    entries = audit.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    lines = [
        f"# Review Coverage Audit: {audit_id}",
        "",
        "> 이 조정치는 감사용이며 공식 calibration row가 아닙니다.",
        "",
        "## Official Calibration",
        f"- Sample: {official['sample_size']}",
        f"- Record: {official['wins']}W-{official['losses']}L-{official['pushes']}P",
        f"- Priced stake: {official['priced_stake_count']}",
        f"- Profit: {official['profit_units']}u",
        f"- ROI: {official['roi']}",
        "",
        "## Audit-only Coverage",
        f"- Resolved omissions: {audit['resolved_omission_count']}",
        f"- Unresolved omissions: {audit['unresolved_omission_count']}",
        f"- Sample: {adjusted['sample_size']}",
        f"- Record: {adjusted['wins']}W-{adjusted['losses']}L-{adjusted['pushes']}P",
        f"- Priced stake: {adjusted['priced_stake_count']}",
        f"- Profit: {adjusted['profit_units']}u",
        f"- ROI: {adjusted['roi']}",
        "",
        "## Omitted Markets",
    ]
    for entry in entries:
        item = _require_mapping(entry, "audit entry")
        flags = ", ".join(item["ineligibility_flags"])
        lines.append(
            f"- `{item['market_key']}`: {item['status']}; flags={flags}; "
            f"result={item['market_result']}"
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit; expected input errors return exit code 2."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("data/ledger/shadow.jsonl"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("reports/private/shadow/current/calibration-latest.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/audits/2026-07-28-review-coverage-manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/private/shadow/audits/2026-07-28-review-coverage"),
    )
    args = parser.parse_args(argv)
    try:
        audit = build_coverage_audit(args.ledger, args.calibration, args.manifest)
        _atomic_write(args.output_dir / "audit.json", canonical_json(audit) + "\n")
        _atomic_write(args.output_dir / "audit.md", render_coverage_audit_markdown(audit))
    except (LedgerCorruptionError, ValueError) as error:
        print(f"coverage-audit: {error}", file=sys.stderr)
        return _INPUT_ERROR_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
