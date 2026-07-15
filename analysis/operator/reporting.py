"""Deterministic machine-readable and Korean calibration reports."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import tempfile

from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType
from analysis.review.calibration import (
    CalibrationMetrics,
    CalibrationReport,
    CalibrationRow,
    GROUPING_DIMENSIONS,
)

from .errors import OperatorIOError


REPORT_SCHEMA_VERSION = 1
CONFIRMED_GAME_THRESHOLD = 100
DECISIONS_PER_MARKET_TYPE_THRESHOLD = 300
CALIBRATION_JSON_FILENAME = "calibration-latest.json"
CALIBRATION_MARKDOWN_FILENAME = "calibration-latest.md"
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


@dataclass(frozen=True)
class CalibrationReportArtifacts:
    """Paths written for one deterministic report mapping."""

    json_path: Path
    markdown_path: Path


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


def _migration_mapping(rows: tuple[CalibrationRow, ...]) -> dict[str, object]:
    confirmed_rows = tuple(row for row in rows if row.lineup_state == "confirmed")
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

    appended_at_values: list[datetime] = []
    for record in ledger_records:
        try:
            appended_at = record.envelope.appended_at
        except AttributeError as error:
            raise ValueError(
                "ledger records must expose envelope.appended_at"
            ) from error
        if not isinstance(appended_at, datetime):
            raise ValueError("ledger appended_at must be a datetime")
        if appended_at.tzinfo is None or appended_at.utcoffset() is None:
            raise ValueError("ledger appended_at must be timezone-aware")
        appended_at_values.append(appended_at)
    ledger_as_of = (
        None
        if not appended_at_values
        else _utc_iso(max(appended_at_values))
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ledger_path": str(Path(ledger_path).expanduser().resolve(strict=False)),
        "ledger_as_of": ledger_as_of,
        "group_by": list(report.group_by),
        "overall": _metrics_mapping(report.overall),
        "groups": [
            {
                "dimensions": dict(group.dimensions),
                "metrics": _metrics_mapping(group.metrics),
            }
            for group in report.groups
        ],
        "migration": _migration_mapping(rows),
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
    if not isinstance(overall, Mapping):
        raise TypeError("overall must be a mapping")
    if not isinstance(groups, list):
        raise TypeError("groups must be a list")
    if not isinstance(migration, Mapping):
        raise TypeError("migration must be a mapping")

    ledger_as_of = mapping["ledger_as_of"]
    lines = [
        "# 섀도 캘리브레이션 보고서",
        "",
        f"- 스키마 버전: {_markdown_value(mapping['schema_version'])}",
        f"- 원장 경로: `{mapping['ledger_path']}`",
        f"- 원장 기준 시각: {_markdown_value(ledger_as_of)}",
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


def _open_directory_without_following_links(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _require_safe_destination(path: Path) -> None:
    if path.is_symlink():
        raise OperatorIOError(
            f"calibration report path cannot be a symbolic link: {path}"
        )
    if path.exists() and not path.is_file():
        raise OperatorIOError(
            f"calibration report path must be a regular file: {path}"
        )


def _atomic_replace(
    destination: Path,
    content: bytes,
    directory_descriptor: int,
) -> None:
    temp_path: Path | None = None
    try:
        _require_safe_destination(destination)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        _require_safe_destination(destination)
        os.replace(temp_path, destination)
        temp_path = None
        os.fsync(directory_descriptor)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def write_calibration_reports(
    mapping: Mapping[str, object],
    *,
    report_dir: str | os.PathLike[str] = DEFAULT_REPORT_DIR,
) -> CalibrationReportArtifacts:
    """Atomically replace deterministic JSON and Markdown report files."""

    json_bytes = (canonical_json(mapping) + "\n").encode("utf-8")
    markdown_bytes = (render_report_markdown(mapping) + "\n").encode("utf-8")
    directory_descriptor = -1
    locked = False
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
        json_path = resolved_dir / CALIBRATION_JSON_FILENAME
        markdown_path = resolved_dir / CALIBRATION_MARKDOWN_FILENAME
        _atomic_replace(json_path, json_bytes, directory_descriptor)
        _atomic_replace(markdown_path, markdown_bytes, directory_descriptor)
        artifacts = CalibrationReportArtifacts(
            json_path=json_path,
            markdown_path=markdown_path,
        )
        fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
        locked = False
        os.close(directory_descriptor)
        directory_descriptor = -1
        return artifacts
    except OperatorIOError:
        raise
    except OSError as error:
        raise OperatorIOError(
            f"unable to write calibration reports: {error}"
        ) from error
    finally:
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
