"""Deterministic machine-readable and Korean calibration reports."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType
from analysis.ledger import ForecastRecord
from analysis.review.calibration import (
    CalibrationMetrics,
    CalibrationReport,
    CalibrationRow,
    GROUPING_DIMENSIONS,
)
from analysis.shadow import deserialize_shadow_comparison

from .errors import OperatorIOError, OperatorLedgerError
from .fdio import (
    create_temporary_directory,
    create_temporary_symlink,
    open_directory,
    read_regular_file,
    stat_at,
    verify_child_directory,
    verify_public_directory,
)


REPORT_SCHEMA_VERSION = 1
CONFIRMED_GAME_THRESHOLD = 100
DECISIONS_PER_MARKET_TYPE_THRESHOLD = 300
CALIBRATION_JSON_FILENAME = "calibration-latest.json"
CALIBRATION_MARKDOWN_FILENAME = "calibration-latest.md"
GENERATIONS_DIRNAME = "generations"
CURRENT_GENERATION_NAME = "current"
COMPATIBILITY_MIGRATION_MARKER_FILENAME = ".compatibility-migration-v1"
DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parents[2] / "reports" / "private" / "shadow"
)

_METRIC_LABELS = {
    "sample_size": "표본 수",
    "wins": "승",
    "losses": "패",
    "pushes": "푸시",
    "binary_sample_size": "이진 표본 수",
    "multiclass_sample_size": "다중분류 표본 수",
    "priced_stake_count": "가격 표본 수",
    "probability_clv_sample_size": "확률 CLV 표본 수",
    "price_clv_sample_size": "가격 CLV 표본 수",
    "hit_rate": "적중률",
    "push_rate": "푸시율",
    "binary_brier": "이진 Brier 점수",
    "multiclass_brier": "다중분류 Brier 점수",
    "log_loss": "로그 손실",
    "profit_units": "손익 단위",
    "roi": "ROI",
    "mean_probability_clv": "평균 확률 CLV",
    "mean_price_clv": "평균 가격 CLV",
}
_COMPATIBILITY_MIGRATION_MARKER_BYTES = (
    b"calibration-compatibility-migration-v1\n"
)


@dataclass(frozen=True)
class CalibrationReportArtifacts:
    """Paths written for one deterministic report mapping."""

    json_path: Path
    markdown_path: Path
    generation_dir: Path
    current_generation_path: Path
    replayed: bool

    @property
    def generation_json_path(self) -> Path:
        return self.generation_dir / CALIBRATION_JSON_FILENAME

    @property
    def generation_markdown_path(self) -> Path:
        return self.generation_dir / CALIBRATION_MARKDOWN_FILENAME


class _StaleCalibrationPublicationError(Exception):
    """A newer ledger revision was published before this writer acquired lock."""


@dataclass(frozen=True)
class _CurrentGeneration:
    generation_name: str
    mapping: Mapping[str, object] | None

    @property
    def is_compatibility_migration(self) -> bool:
        return self.mapping is None


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ledger appended_at must be timezone-aware")
    normalized = value.astimezone(timezone.utc).isoformat()
    return f"{normalized[:-6]}Z"


def _metrics_mapping(metrics: CalibrationMetrics) -> dict[str, object]:
    return {
        field.name: getattr(metrics, field.name)
        for field in fields(CalibrationMetrics)
    }


def _progress_mapping(count: int, threshold: int) -> dict[str, object]:
    completed = min(count, threshold)
    return {
        "count": count,
        "threshold": threshold,
        "completed": completed,
        "remaining": max(threshold - count, 0),
        "progress": completed / threshold,
        "threshold_met": count >= threshold,
    }


def _ledger_revision_mapping(
    ledger_records: tuple[object, ...],
) -> tuple[dict[str, object], list[datetime]]:
    identities: list[dict[str, str]] = []
    appended_at_values: list[datetime] = []
    for record in ledger_records:
        try:
            envelope = record.envelope
            record_kind = envelope.record_kind
            record_id = envelope.record_id
            idempotency_key = envelope.idempotency_key
            payload_hash = envelope.payload_hash
            appended_at = envelope.appended_at
        except AttributeError as error:
            raise ValueError(
                "ledger records must expose immutable envelope identity"
            ) from error
        identity_values = {
            "record_kind": record_kind,
            "record_id": record_id,
            "idempotency_key": idempotency_key,
            "payload_hash": payload_hash,
        }
        if any(
            not isinstance(value, str) or not value
            for value in identity_values.values()
        ):
            raise ValueError(
                "ledger envelope identity values must be non-empty strings"
            )
        if not isinstance(appended_at, datetime):
            raise ValueError("ledger appended_at must be a datetime")
        appended_at_text = _utc_iso(appended_at)
        identities.append(
            {
                **identity_values,
                "appended_at": appended_at_text,
            }
        )
        appended_at_values.append(appended_at)
    state_hash = hashlib.sha256(
        canonical_json(identities).encode("utf-8")
    ).hexdigest()
    return (
        {
            "record_count": len(identities),
            "state_hash": state_hash,
        },
        appended_at_values,
    )


def _migration_eligible_rows(
    rows: tuple[CalibrationRow, ...],
    ledger_records: tuple[object, ...],
) -> tuple[CalibrationRow, ...]:
    confirmed_rows = tuple(row for row in rows if row.lineup_state == "confirmed")
    required_forecast_ids = {
        row.forecast_run_id for row in confirmed_rows
    }
    forecasts_by_run_id: dict[str, ForecastRecord] = {}
    for record in ledger_records:
        if (
            not isinstance(record, ForecastRecord)
            or record.forecast_run_id not in required_forecast_ids
        ):
            continue
        if record.forecast_run_id in forecasts_by_run_id:
            raise ValueError(
                "duplicate frozen forecast required by migration row: "
                f"{record.forecast_run_id}"
            )
        forecasts_by_run_id[record.forecast_run_id] = record

    missing_forecasts = required_forecast_ids - forecasts_by_run_id.keys()
    if missing_forecasts:
        missing = ", ".join(sorted(missing_forecasts))
        raise ValueError(
            f"migration rows reference missing frozen forecasts: {missing}"
        )

    eligible_pairs: set[tuple[str, str]] = set()
    comparison_pairs: set[tuple[str, str]] = set()
    for forecast_run_id in sorted(required_forecast_ids):
        forecast = forecasts_by_run_id[forecast_run_id]
        try:
            comparison = deserialize_shadow_comparison(
                forecast.outputs.get("shadow_comparison")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "invalid frozen shadow comparison required by migration row "
                f"for forecast {forecast_run_id}: {error}"
            ) from error
        if comparison.forecast_run_id != forecast_run_id:
            raise ValueError(
                "frozen shadow comparison forecast_run_id does not match "
                f"migration forecast {forecast_run_id}"
            )
        for market in comparison.market_comparisons:
            pair = (forecast_run_id, market.market_key)
            if pair in comparison_pairs:
                raise ValueError(
                    "duplicate exact market in frozen shadow comparison: "
                    f"{forecast_run_id}, {market.market_key}"
                )
            comparison_pairs.add(pair)
            if market.review_eligible:
                eligible_pairs.add(pair)

    required_pairs = {
        (row.forecast_run_id, row.market_key) for row in confirmed_rows
    }
    missing_pairs = required_pairs - comparison_pairs
    if missing_pairs:
        missing = ", ".join(
            f"({forecast_run_id}, {market_key})"
            for forecast_run_id, market_key in sorted(missing_pairs)
        )
        raise ValueError(
            "migration rows reference missing exact frozen market comparisons: "
            f"{missing}"
        )
    return tuple(
        row
        for row in confirmed_rows
        if (row.forecast_run_id, row.market_key) in eligible_pairs
    )


def _migration_mapping(
    confirmed_rows: tuple[CalibrationRow, ...],
) -> dict[str, object]:
    confirmed_games = len({row.game_pk for row in confirmed_rows})
    market_types: dict[str, object] = {}
    for market_type in MarketType:
        decisions = {
            (row.forecast_run_id, row.market_key)
            for row in confirmed_rows
            if row.market_type == market_type.value
        }
        market_types[market_type.value] = _progress_mapping(
            len(decisions),
            DECISIONS_PER_MARKET_TYPE_THRESHOLD,
        )
    return {
        "confirmed_games": _progress_mapping(
            confirmed_games,
            CONFIRMED_GAME_THRESHOLD,
        ),
        "market_types": market_types,
    }


def build_report_mapping(
    *,
    ledger_path: str | os.PathLike[str],
    ledger_records: tuple[object, ...],
    rows: tuple[CalibrationRow, ...],
    report: CalibrationReport,
) -> dict[str, object]:
    """Build the complete deterministic report mapping from ledger state."""

    if not isinstance(ledger_records, tuple):
        raise TypeError("ledger_records must be a tuple")
    if not isinstance(rows, tuple) or any(
        not isinstance(row, CalibrationRow) for row in rows
    ):
        raise TypeError("rows must be a tuple of CalibrationRow values")
    if not isinstance(report, CalibrationReport):
        raise TypeError("report must be a CalibrationReport")
    if report.group_by != GROUPING_DIMENSIONS:
        raise ValueError("report must use the fixed calibration group dimensions")

    ledger_revision, appended_at_values = _ledger_revision_mapping(
        ledger_records
    )
    ledger_as_of = (
        None
        if not appended_at_values
        else _utc_iso(max(appended_at_values))
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ledger_path": str(Path(ledger_path).expanduser().resolve(strict=False)),
        "ledger_as_of": ledger_as_of,
        "ledger_revision": ledger_revision,
        "group_by": list(report.group_by),
        "overall": _metrics_mapping(report.overall),
        "groups": [
            {
                "dimensions": dict(group.dimensions),
                "metrics": _metrics_mapping(group.metrics),
            }
            for group in report.groups
        ],
        "migration": _migration_mapping(
            _migration_eligible_rows(rows, ledger_records)
        ),
    }


def _markdown_value(value: object) -> str:
    if value is None:
        return "사용 불가 (unavailable)"
    if isinstance(value, bool):
        return "충족 (true)" if value else "미충족 (false)"
    return canonical_json(value) if not isinstance(value, str) else value


def _append_metrics(
    lines: list[str],
    metrics: Mapping[str, object],
) -> None:
    lines.extend(("| 지표 | 값 |", "|---|---:|"))
    for field in fields(CalibrationMetrics):
        label = _METRIC_LABELS[field.name]
        lines.append(f"| {label} | {_markdown_value(metrics[field.name])} |")


def render_report_markdown(mapping: Mapping[str, object]) -> str:
    """Render Korean Markdown using only the machine-readable mapping."""

    overall = mapping["overall"]
    groups = mapping["groups"]
    migration = mapping["migration"]
    ledger_revision = mapping["ledger_revision"]
    if not isinstance(overall, Mapping):
        raise TypeError("overall must be a mapping")
    if not isinstance(groups, list):
        raise TypeError("groups must be a list")
    if not isinstance(migration, Mapping):
        raise TypeError("migration must be a mapping")
    if not isinstance(ledger_revision, Mapping):
        raise TypeError("ledger_revision must be a mapping")

    ledger_as_of = mapping["ledger_as_of"]
    lines = [
        "# 섀도 캘리브레이션 보고서",
        "",
        f"- 스키마 버전: {_markdown_value(mapping['schema_version'])}",
        f"- 원장 경로: `{mapping['ledger_path']}`",
        f"- 원장 기준 시각: {_markdown_value(ledger_as_of)}",
        "- 원장 레코드 수: "
        f"{_markdown_value(ledger_revision['record_count'])}",
        "- 원장 상태 해시: "
        f"`{_markdown_value(ledger_revision['state_hash'])}`",
        "- 고정 그룹 차원: "
        + ", ".join(str(value) for value in mapping["group_by"]),
        "",
        "## 전체 지표",
        "",
    ]
    _append_metrics(lines, overall)

    lines.extend(("", "## 그룹 지표"))
    if not groups:
        lines.extend(("", "그룹 표본 없음"))
    for group in groups:
        if not isinstance(group, Mapping):
            raise TypeError("group entries must be mappings")
        dimensions = group["dimensions"]
        metrics = group["metrics"]
        if not isinstance(dimensions, Mapping) or not isinstance(metrics, Mapping):
            raise TypeError("group dimensions and metrics must be mappings")
        dimension_text = ", ".join(
            f"{name}={value}" for name, value in dimensions.items()
        )
        lines.extend(("", f"### {dimension_text}", ""))
        _append_metrics(lines, metrics)

    confirmed_games = migration["confirmed_games"]
    market_types = migration["market_types"]
    if not isinstance(confirmed_games, Mapping) or not isinstance(
        market_types, Mapping
    ):
        raise TypeError("migration entries must be mappings")
    lines.extend(
        (
            "",
            "## 마이그레이션 진행",
            "",
            "| 구분 | 현재 수 | 기준 | 인정 수 | 남은 수 | 진행률 | 기준 충족 |",
            "|---|---:|---:|---:|---:|---:|---|",
        )
    )

    def append_progress(label: str, progress: Mapping[str, object]) -> None:
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _markdown_value(progress["count"]),
                    _markdown_value(progress["threshold"]),
                    _markdown_value(progress["completed"]),
                    _markdown_value(progress["remaining"]),
                    _markdown_value(progress["progress"]),
                    _markdown_value(progress["threshold_met"]),
                )
            )
            + " |"
        )

    append_progress("확정 라인업 경기", confirmed_games)
    for market_type, progress in market_types.items():
        if not isinstance(progress, Mapping):
            raise TypeError("market migration progress must be a mapping")
        append_progress(f"{market_type} 결정", progress)
    return "\n".join(lines)


def _open_directory_without_following_links(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
) -> int:
    return open_directory(path, dir_fd=dir_fd)


def _report_revision(
    mapping: Mapping[str, object],
    *,
    source: str,
) -> tuple[int, str]:
    revision = mapping.get("ledger_revision")
    if not isinstance(revision, Mapping):
        raise OperatorLedgerError(
            f"{source} ledger_revision must be a mapping"
        )
    record_count = revision.get("record_count")
    state_hash = revision.get("state_hash")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
    ):
        raise OperatorLedgerError(
            f"{source} ledger_revision record_count must be non-negative"
        )
    if (
        not isinstance(state_hash, str)
        or len(state_hash) != 64
        or any(character not in "0123456789abcdef" for character in state_hash)
    ):
        raise OperatorLedgerError(
            f"{source} ledger_revision state_hash must be lowercase SHA-256"
        )
    return record_count, state_hash


def _report_ledger_path(
    mapping: Mapping[str, object],
    *,
    source: str,
) -> str:
    ledger_path = mapping.get("ledger_path")
    if not isinstance(ledger_path, str) or not ledger_path:
        raise OperatorLedgerError(
            f"{source} ledger_path must be a non-empty string"
        )
    return ledger_path


def _open_generations_directory(report_descriptor: int) -> int:
    metadata = stat_at(report_descriptor, GENERATIONS_DIRNAME)
    if metadata is None:
        os.mkdir(GENERATIONS_DIRNAME, mode=0o700, dir_fd=report_descriptor)
        os.fsync(report_descriptor)
    elif stat.S_ISLNK(metadata.st_mode):
        raise OperatorIOError(
            "calibration generations directory cannot be a symbolic link"
        )
    elif not stat.S_ISDIR(metadata.st_mode):
        raise OperatorIOError(
            "calibration generations path must be a directory"
        )
    return _open_directory_without_following_links(
        GENERATIONS_DIRNAME,
        dir_fd=report_descriptor,
    )


def _open_generation_directory(
    generations_descriptor: int,
    generation_name: str,
) -> int:
    metadata = stat_at(generations_descriptor, generation_name)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise OperatorIOError(
            "calibration generation directory is missing or unsafe: "
            f"{generation_name}"
        )
    return _open_directory_without_following_links(
        generation_name,
        dir_fd=generations_descriptor,
    )


def _read_generation_bytes(
    generations_descriptor: int,
    generation_name: str,
) -> tuple[bytes, bytes]:
    generation_descriptor = _open_generation_directory(
        generations_descriptor,
        generation_name,
    )
    try:
        return (
            read_regular_file(
                generation_descriptor,
                CALIBRATION_JSON_FILENAME,
                label="calibration generation JSON",
            ),
            read_regular_file(
                generation_descriptor,
                CALIBRATION_MARKDOWN_FILENAME,
                label="calibration generation Markdown",
            ),
        )
    finally:
        os.close(generation_descriptor)


def _read_generation_mapping(raw_json: bytes) -> Mapping[str, object]:
    try:
        mapping = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorLedgerError(
            f"published calibration report JSON is invalid: {error}"
        ) from error
    if not isinstance(mapping, Mapping):
        raise OperatorLedgerError(
            "published calibration report JSON must contain a mapping"
        )
    return mapping


def _compatibility_generation_name(
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> str:
    digest = hashlib.sha256(
        b"json\0" + json_bytes + b"\0markdown\0" + markdown_bytes
    ).hexdigest()
    return f"compatibility-v1-{digest}"


def _current_generation_name(report_descriptor: int) -> str | None:
    current_metadata = stat_at(report_descriptor, CURRENT_GENERATION_NAME)
    if current_metadata is None:
        return None
    if not stat.S_ISLNK(current_metadata.st_mode):
        raise OperatorIOError(
            "calibration current generation must be a symbolic link"
        )
    target = Path(
        os.readlink(CURRENT_GENERATION_NAME, dir_fd=report_descriptor)
    )
    if (
        target.is_absolute()
        or len(target.parts) != 2
        or target.parts[0] != GENERATIONS_DIRNAME
        or target.parts[1] in ("", ".", "..")
    ):
        raise OperatorIOError(
            f"calibration current generation target is unsafe: {target}"
        )
    return target.parts[1]


def _current_references_generation(
    report_descriptor: int,
    generation_name: str,
) -> bool:
    return _current_generation_name(report_descriptor) == generation_name


def _current_generation(
    report_descriptor: int,
    generations_descriptor: int,
) -> _CurrentGeneration | None:
    generation_name = _current_generation_name(report_descriptor)
    if generation_name is None:
        return None
    raw_json, raw_markdown = _read_generation_bytes(
        generations_descriptor,
        generation_name,
    )
    generation_descriptor = _open_generation_directory(
        generations_descriptor,
        generation_name,
    )
    try:
        marker_metadata = stat_at(
            generation_descriptor,
            COMPATIBILITY_MIGRATION_MARKER_FILENAME,
        )
        if marker_metadata is None:
            return _CurrentGeneration(
                generation_name,
                _read_generation_mapping(raw_json),
            )
        if (
            stat.S_ISLNK(marker_metadata.st_mode)
            or not stat.S_ISREG(marker_metadata.st_mode)
        ):
            raise OperatorIOError(
                "calibration compatibility migration marker is unsafe"
            )
        if (
            read_regular_file(
                generation_descriptor,
                COMPATIBILITY_MIGRATION_MARKER_FILENAME,
                label="calibration compatibility migration marker",
            )
            != _COMPATIBILITY_MIGRATION_MARKER_BYTES
            or generation_name
            != _compatibility_generation_name(raw_json, raw_markdown)
        ):
            raise OperatorLedgerError(
                "calibration compatibility migration generation is corrupt"
            )
        return _CurrentGeneration(generation_name, None)
    finally:
        os.close(generation_descriptor)


def _compatibility_target(filename: str) -> str:
    return f"{CURRENT_GENERATION_NAME}/{filename}"


def _compatibility_path_status(
    report_descriptor: int,
    filename: str,
    target: str,
) -> str:
    metadata = stat_at(report_descriptor, filename)
    if metadata is None:
        return "missing"
    if stat.S_ISLNK(metadata.st_mode):
        if os.readlink(filename, dir_fd=report_descriptor) != target:
            raise OperatorIOError(
                "calibration report path has an unsafe symbolic link: "
                f"{filename}"
            )
        return "link"
    if not stat.S_ISREG(metadata.st_mode):
        raise OperatorIOError(
            f"calibration report path must be a regular file or managed link: "
            f"{filename}"
        )
    return "regular"


def _replace_with_symlink(
    report_descriptor: int,
    destination: str,
    target: str,
) -> None:
    staged_name = create_temporary_symlink(
        report_descriptor,
        target,
        prefix=f".{destination}.",
        suffix=".tmp",
        target_is_directory=False,
    )
    try:
        os.replace(
            staged_name,
            destination,
            src_dir_fd=report_descriptor,
            dst_dir_fd=report_descriptor,
        )
    finally:
        try:
            os.unlink(staged_name, dir_fd=report_descriptor)
        except FileNotFoundError:
            pass


def _compatibility_entries(
    report_descriptor: int,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            filename,
            _compatibility_target(filename),
        )
        for filename in (
            CALIBRATION_JSON_FILENAME,
            CALIBRATION_MARKDOWN_FILENAME,
        )
    )


def _prepare_compatibility_links(
    report_path: Path,
    report_descriptor: int,
    generations_descriptor: int,
    current: _CurrentGeneration | None,
) -> _CurrentGeneration | None:
    entries = _compatibility_entries(report_descriptor)
    statuses = {
        filename: _compatibility_path_status(
            report_descriptor,
            filename,
            target,
        )
        for filename, target in entries
    }
    regular_entries = tuple(
        entry for entry in entries if statuses[entry[0]] == "regular"
    )

    if not regular_entries:
        changed = False
        for filename, target in entries:
            if statuses[filename] != "missing":
                continue
            _replace_with_symlink(
                report_descriptor,
                filename,
                target,
            )
            changed = True
        if changed:
            os.fsync(report_descriptor)
        return current

    if any(status == "missing" for status in statuses.values()):
        raise OperatorIOError(
            "cannot atomically migrate an incomplete legacy calibration pair"
        )

    if current is None:
        if len(regular_entries) != len(entries):
            raise OperatorIOError(
                "cannot resume a partial legacy calibration conversion "
                "without its migration generation"
            )
        legacy_json = read_regular_file(
            report_descriptor,
            CALIBRATION_JSON_FILENAME,
            label="legacy calibration JSON",
        )
        legacy_markdown = read_regular_file(
            report_descriptor,
            CALIBRATION_MARKDOWN_FILENAME,
            label="legacy calibration Markdown",
        )
        # Convert each path through a current generation with identical bytes.
        migration_name, migration_created = (
            _materialize_compatibility_generation(
                report_descriptor,
                generations_descriptor,
                legacy_json,
                legacy_markdown,
            )
        )
        migration_pointer_replaced = False
        try:
            _publish_current_pointer(
                report_path,
                report_descriptor,
                generations_descriptor,
                migration_name,
                None,
            )
            migration_pointer_replaced = True
        finally:
            if migration_created and not migration_pointer_replaced:
                _remove_generation_if_unreferenced(
                    report_descriptor,
                    generations_descriptor,
                    migration_name,
                )
        current = _CurrentGeneration(migration_name, None)
    elif not current.is_compatibility_migration:
        raise OperatorIOError(
            "cannot atomically convert legacy calibration files while a "
            "published current generation exists"
        )

    for filename, _ in regular_entries:
        expected = _read_generation_member(
            generations_descriptor,
            current.generation_name,
            filename,
        )
        if read_regular_file(
            report_descriptor,
            filename,
            label=f"legacy calibration path {filename}",
        ) != expected:
            raise OperatorLedgerError(
                "legacy calibration compatibility content changed during "
                "migration"
            )

    for filename, target in regular_entries:
        _replace_with_symlink(
            report_descriptor,
            filename,
            target,
        )
    if regular_entries:
        os.fsync(report_descriptor)
    return current


def _require_equal_revision_content(
    generations_descriptor: int,
    generation_name: str,
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> None:
    published_json, published_markdown = _read_generation_bytes(
        generations_descriptor,
        generation_name,
    )
    mismatches: list[str] = []
    if published_json != json_bytes:
        mismatches.append("JSON")
    if published_markdown != markdown_bytes:
        mismatches.append("Markdown")
    if mismatches:
        raise OperatorLedgerError(
            "calibration report replay content conflict: "
            + " and ".join(mismatches)
            + " bytes differ from the published generation"
        )


def _verify_report_directories(
    report_path: Path,
    report_descriptor: int,
    generations_descriptor: int,
) -> None:
    verify_public_directory(
        report_path,
        report_descriptor,
        label="calibration report directory",
    )
    verify_child_directory(
        report_descriptor,
        GENERATIONS_DIRNAME,
        generations_descriptor,
        label="calibration generations directory",
    )


def _read_generation_member(
    generations_descriptor: int,
    generation_name: str,
    filename: str,
) -> bytes:
    generation_descriptor = _open_generation_directory(
        generations_descriptor,
        generation_name,
    )
    try:
        return read_regular_file(
            generation_descriptor,
            filename,
            label=f"calibration generation member {filename}",
        )
    finally:
        os.close(generation_descriptor)


def _write_generation_file(
    directory_descriptor: int,
    filename: str,
    content: bytes,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        filename,
        flags,
        0o600,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as report_file:
        report_file.write(content)
        report_file.flush()
        os.fsync(report_file.fileno())


def _remove_generation_directory(
    generations_descriptor: int,
    generation_name: str,
) -> None:
    metadata = stat_at(generations_descriptor, generation_name)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        return
    try:
        generation_descriptor = _open_generation_directory(
            generations_descriptor,
            generation_name,
        )
    except OSError:
        return
    try:
        for filename in (
            CALIBRATION_JSON_FILENAME,
            CALIBRATION_MARKDOWN_FILENAME,
            COMPATIBILITY_MIGRATION_MARKER_FILENAME,
        ):
            try:
                os.unlink(filename, dir_fd=generation_descriptor)
            except OSError:
                pass
        try:
            os.fsync(generation_descriptor)
        except OSError:
            pass
    finally:
        os.close(generation_descriptor)
    try:
        os.rmdir(generation_name, dir_fd=generations_descriptor)
    except OSError:
        return
    try:
        os.fsync(generations_descriptor)
    except OSError:
        pass


def _remove_generation_if_unreferenced(
    report_descriptor: int,
    generations_descriptor: int,
    generation_name: str,
) -> None:
    try:
        if _current_references_generation(
            report_descriptor,
            generation_name,
        ):
            return
    except (OSError, OperatorIOError):
        # Cleanup is conservative when current cannot be inspected safely.
        return
    _remove_generation_directory(
        generations_descriptor,
        generation_name,
    )


def _materialize_immutable_generation(
    report_descriptor: int,
    generations_descriptor: int,
    generation_name: str,
    members: tuple[tuple[str, bytes], ...],
    *,
    conflict_message: str,
) -> tuple[str, bool]:
    metadata = stat_at(generations_descriptor, generation_name)
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise OperatorIOError(
                "calibration generation path is unsafe: "
                f"{generation_name}"
            )
        generation_descriptor = _open_generation_directory(
            generations_descriptor,
            generation_name,
        )
        try:
            if any(
                read_regular_file(
                    generation_descriptor,
                    filename,
                    label=f"calibration generation member {filename}",
                )
                != content
                for filename, content in members
            ):
                raise OperatorLedgerError(conflict_message)
        finally:
            os.close(generation_descriptor)
        return generation_name, False

    temporary_descriptor, temporary_name = create_temporary_directory(
        generations_descriptor,
        prefix=f".{generation_name}.",
        suffix=".tmp",
    )
    try:
        for filename, content in members:
            _write_generation_file(
                temporary_descriptor,
                filename,
                content,
            )
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.rename(
            temporary_name,
            generation_name,
            src_dir_fd=generations_descriptor,
            dst_dir_fd=generations_descriptor,
        )
        temporary_name = ""
        try:
            os.fsync(generations_descriptor)
        except OSError:
            _remove_generation_if_unreferenced(
                report_descriptor,
                generations_descriptor,
                generation_name,
            )
            raise
        return generation_name, True
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name:
            _remove_generation_if_unreferenced(
                report_descriptor,
                generations_descriptor,
                temporary_name,
            )


def _materialize_compatibility_generation(
    report_descriptor: int,
    generations_descriptor: int,
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> tuple[str, bool]:
    generation_name = _compatibility_generation_name(
        json_bytes,
        markdown_bytes,
    )
    return _materialize_immutable_generation(
        report_descriptor,
        generations_descriptor,
        generation_name,
        (
            (CALIBRATION_JSON_FILENAME, json_bytes),
            (CALIBRATION_MARKDOWN_FILENAME, markdown_bytes),
            (
                COMPATIBILITY_MIGRATION_MARKER_FILENAME,
                _COMPATIBILITY_MIGRATION_MARKER_BYTES,
            ),
        ),
        conflict_message=(
            "legacy calibration compatibility generation conflicts with "
            "its content hash"
        ),
    )


def _materialize_generation(
    report_descriptor: int,
    generations_descriptor: int,
    generation_name: str,
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> tuple[str, bool]:
    return _materialize_immutable_generation(
        report_descriptor,
        generations_descriptor,
        generation_name,
        (
            (CALIBRATION_JSON_FILENAME, json_bytes),
            (CALIBRATION_MARKDOWN_FILENAME, markdown_bytes),
        ),
        conflict_message=(
            "calibration generation content conflicts with its ledger "
            "revision"
        ),
    )


def _swap_current_pointer(
    report_descriptor: int,
    generation_name: str,
) -> None:
    target = str(Path(GENERATIONS_DIRNAME) / generation_name)
    staged_name = create_temporary_symlink(
        report_descriptor,
        target,
        prefix=".current.",
        suffix=".tmp",
        target_is_directory=True,
    )
    try:
        os.replace(
            staged_name,
            CURRENT_GENERATION_NAME,
            src_dir_fd=report_descriptor,
            dst_dir_fd=report_descriptor,
        )
    finally:
        try:
            os.unlink(staged_name, dir_fd=report_descriptor)
        except FileNotFoundError:
            pass


def _restore_current_pointer(
    report_descriptor: int,
    generation_name: str | None,
) -> None:
    if generation_name is None:
        try:
            os.unlink(
                CURRENT_GENERATION_NAME,
                dir_fd=report_descriptor,
            )
        except FileNotFoundError:
            pass
    else:
        _swap_current_pointer(report_descriptor, generation_name)
    os.fsync(report_descriptor)


def _publish_current_pointer(
    report_path: Path,
    report_descriptor: int,
    generations_descriptor: int,
    generation_name: str,
    previous_generation_name: str | None,
) -> None:
    pointer_replaced = False
    try:
        _verify_report_directories(
            report_path,
            report_descriptor,
            generations_descriptor,
        )
        _swap_current_pointer(report_descriptor, generation_name)
        pointer_replaced = True
        os.fsync(report_descriptor)
        _verify_report_directories(
            report_path,
            report_descriptor,
            generations_descriptor,
        )
    except BaseException as error:
        if pointer_replaced and isinstance(error, OperatorIOError):
            try:
                if _current_references_generation(
                    report_descriptor,
                    generation_name,
                ):
                    _restore_current_pointer(
                        report_descriptor,
                        previous_generation_name,
                    )
            except (OSError, OperatorIOError) as rollback_error:
                raise OperatorIOError(
                    "unable to restore calibration current generation after "
                    f"publication failure: {rollback_error}"
                ) from rollback_error
        raise


def _artifacts(
    report_dir: Path,
    generation_name: str,
    *,
    replayed: bool,
) -> CalibrationReportArtifacts:
    generation_dir = report_dir / GENERATIONS_DIRNAME / generation_name
    return CalibrationReportArtifacts(
        json_path=report_dir / CALIBRATION_JSON_FILENAME,
        markdown_path=report_dir / CALIBRATION_MARKDOWN_FILENAME,
        generation_dir=generation_dir,
        current_generation_path=report_dir / CURRENT_GENERATION_NAME,
        replayed=replayed,
    )


def write_calibration_reports(
    mapping: Mapping[str, object],
    *,
    report_dir: str | os.PathLike[str] = DEFAULT_REPORT_DIR,
) -> CalibrationReportArtifacts:
    """Publish one immutable report generation through an atomic pointer."""

    json_bytes = (canonical_json(mapping) + "\n").encode("utf-8")
    markdown_bytes = (render_report_markdown(mapping) + "\n").encode("utf-8")
    incoming_count, incoming_hash = _report_revision(
        mapping,
        source="incoming calibration report",
    )
    incoming_ledger_path = _report_ledger_path(
        mapping,
        source="incoming calibration report",
    )
    directory_descriptor = -1
    generations_descriptor = -1
    locked = False
    generation_name: str | None = None
    generation_created = False
    pointer_replaced = False
    try:
        supplied_dir = Path(report_dir).expanduser()
        if supplied_dir.is_symlink():
            raise OperatorIOError(
                f"calibration report directory cannot be a symbolic link: "
                f"{supplied_dir}"
            )
        resolved_dir = supplied_dir.resolve(strict=False)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        directory_descriptor = _open_directory_without_following_links(
            resolved_dir
        )
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        locked = True
        generations_descriptor = _open_generations_directory(
            directory_descriptor
        )
        _verify_report_directories(
            resolved_dir,
            directory_descriptor,
            generations_descriptor,
        )
        os.fsync(directory_descriptor)

        current = _current_generation(
            directory_descriptor,
            generations_descriptor,
        )
        current = _prepare_compatibility_links(
            resolved_dir,
            directory_descriptor,
            generations_descriptor,
            current,
        )
        _verify_report_directories(
            resolved_dir,
            directory_descriptor,
            generations_descriptor,
        )
        if current is not None and current.mapping is not None:
            current_mapping = current.mapping
            current_count, current_hash = _report_revision(
                current_mapping,
                source="published calibration report",
            )
            current_ledger_path = _report_ledger_path(
                current_mapping,
                source="published calibration report",
            )
            if incoming_ledger_path == current_ledger_path:
                if incoming_count < current_count:
                    raise _StaleCalibrationPublicationError(
                        "incoming calibration ledger revision is stale: "
                        f"{incoming_count} < {current_count} records"
                    )
                if incoming_count == current_count:
                    if incoming_hash != current_hash:
                        raise OperatorLedgerError(
                            "calibration ledger revision conflict: equal record "
                            "counts have different state hashes"
                        )
                    _require_equal_revision_content(
                        generations_descriptor,
                        current.generation_name,
                        json_bytes,
                        markdown_bytes,
                    )
                    _verify_report_directories(
                        resolved_dir,
                        directory_descriptor,
                        generations_descriptor,
                    )
                    artifacts = _artifacts(
                        resolved_dir,
                        current.generation_name,
                        replayed=True,
                    )
                    return artifacts

        ledger_path_hash = hashlib.sha256(
            incoming_ledger_path.encode("utf-8")
        ).hexdigest()[:16]
        generation_name = (
            f"generation-v{REPORT_SCHEMA_VERSION}-"
            f"{ledger_path_hash}-{incoming_count:020d}-{incoming_hash}"
        )
        _, generation_created = _materialize_generation(
            directory_descriptor,
            generations_descriptor,
            generation_name,
            json_bytes,
            markdown_bytes,
        )
        _publish_current_pointer(
            resolved_dir,
            directory_descriptor,
            generations_descriptor,
            generation_name,
            None if current is None else current.generation_name,
        )
        pointer_replaced = True
        artifacts = _artifacts(
            resolved_dir,
            generation_name,
            replayed=False,
        )
        return artifacts
    except (
        _StaleCalibrationPublicationError,
        OperatorIOError,
        OperatorLedgerError,
    ):
        raise
    except OSError as error:
        raise OperatorIOError(
            f"unable to write calibration reports: {error}"
        ) from error
    finally:
        if (
            generation_created
            and not pointer_replaced
            and generation_name is not None
            and generations_descriptor >= 0
        ):
            _remove_generation_if_unreferenced(
                directory_descriptor,
                generations_descriptor,
                generation_name,
            )
        if generations_descriptor >= 0:
            try:
                os.close(generations_descriptor)
            except OSError:
                pass
        if directory_descriptor >= 0:
            try:
                if locked:
                    try:
                        fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                try:
                    os.close(directory_descriptor)
                except OSError:
                    pass


__all__ = (
    "CALIBRATION_JSON_FILENAME",
    "CALIBRATION_MARKDOWN_FILENAME",
    "CONFIRMED_GAME_THRESHOLD",
    "CalibrationReportArtifacts",
    "DECISIONS_PER_MARKET_TYPE_THRESHOLD",
    "DEFAULT_REPORT_DIR",
    "REPORT_SCHEMA_VERSION",
    "build_report_mapping",
    "render_report_markdown",
    "write_calibration_reports",
)
