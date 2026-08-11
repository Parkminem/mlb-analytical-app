"""Deterministic repeated-review watchlist configuration and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType

from analysis.review.calibration import CalibrationRow


DEFAULT_WATCHLIST_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "review-watchlist.json"
)

_SCHEMA_VERSION = 1
_COMMON_TARGET_KEYS = frozenset(
    ("target_id", "label", "kind", "minimum_sample")
)
_F5_OVER_TARGET_KEYS = _COMMON_TARGET_KEYS | frozenset(
    ("hit_rate_below", "b_loss_count_min", "b_loss_rate_min")
)
_PATH_TARGET_KEYS = _COMMON_TARGET_KEYS | frozenset(
    ("evidence_path", "contradiction_count_min", "contradiction_rate_min")
)
_CLASSIFICATIONS = ("A", "B", "C", "D")
_STATUSES = frozenset(("관찰", "정상 범위", "정밀 검토"))


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a rate within [0, 1]")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be a rate within [0, 1]")
    return rate


@dataclass(frozen=True)
class WatchlistTargetConfig:
    target_id: str
    label: str
    kind: str
    minimum_sample: int
    hit_rate_below: float | None = None
    b_loss_count_min: int | None = None
    b_loss_rate_min: float | None = None
    evidence_path: str | None = None
    contradiction_count_min: int | None = None
    contradiction_rate_min: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.target_id, "target_id")
        _require_nonempty_string(self.label, "label")
        _require_positive_integer(self.minimum_sample, "minimum_sample")

        if self.kind == "f5_over_conversion":
            object.__setattr__(
                self,
                "hit_rate_below",
                _require_rate(self.hit_rate_below, "hit_rate_below"),
            )
            _require_positive_integer(self.b_loss_count_min, "b_loss_count_min")
            object.__setattr__(
                self,
                "b_loss_rate_min",
                _require_rate(self.b_loss_rate_min, "b_loss_rate_min"),
            )
            if any(
                value is not None
                for value in (
                    self.evidence_path,
                    self.contradiction_count_min,
                    self.contradiction_rate_min,
                )
            ):
                raise ValueError("f5_over_conversion has inapplicable keys")
            return

        if self.kind == "path_contradiction":
            _require_nonempty_string(self.evidence_path, "evidence_path")
            _require_positive_integer(
                self.contradiction_count_min,
                "contradiction_count_min",
            )
            object.__setattr__(
                self,
                "contradiction_rate_min",
                _require_rate(
                    self.contradiction_rate_min,
                    "contradiction_rate_min",
                ),
            )
            if any(
                value is not None
                for value in (
                    self.hit_rate_below,
                    self.b_loss_count_min,
                    self.b_loss_rate_min,
                )
            ):
                raise ValueError("path_contradiction has inapplicable keys")
            return

        raise ValueError(f"unsupported watchlist target kind: {self.kind!r}")


@dataclass(frozen=True)
class ReviewWatchlistConfig:
    schema_version: int
    targets: tuple[WatchlistTargetConfig, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be the supported version {_SCHEMA_VERSION}"
            )
        if (
            not isinstance(self.targets, tuple)
            or not self.targets
            or any(
                not isinstance(target, WatchlistTargetConfig)
                for target in self.targets
            )
        ):
            raise ValueError("targets must be a nonempty tuple of target configs")
        target_ids = tuple(target.target_id for target in self.targets)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("duplicate watchlist target_id")


@dataclass(frozen=True)
class WatchlistResult:
    target_id: str
    label: str
    status: str
    eligible_count: int
    excluded_count: int
    win_count: int
    loss_count: int
    push_count: int
    classification_counts: Mapping[str, int]
    b_loss_count: int | None
    contradiction_count: int | None
    contradiction_rate: float | None
    non_push_hit_rate: float | None
    minimum_sample: int
    trigger_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.target_id, "target_id")
        _require_nonempty_string(self.label, "label")
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported watchlist status: {self.status!r}")
        for field_name in (
            "eligible_count",
            "excluded_count",
            "win_count",
            "loss_count",
            "push_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.eligible_count != self.win_count + self.loss_count + self.push_count:
            raise ValueError("eligible_count must equal outcome counts")

        if not isinstance(self.classification_counts, Mapping) or set(
            self.classification_counts
        ) != set(_CLASSIFICATIONS):
            raise ValueError("classification_counts must contain A, B, C, and D")
        frozen_counts: dict[str, int] = {}
        for classification in _CLASSIFICATIONS:
            count = self.classification_counts[classification]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "classification_counts must contain nonnegative integers"
                )
            frozen_counts[classification] = count
        if sum(frozen_counts.values()) > self.eligible_count:
            raise ValueError("classification_counts exceed eligible_count")
        object.__setattr__(
            self,
            "classification_counts",
            MappingProxyType(frozen_counts),
        )

        for field_name in ("b_loss_count", "contradiction_count"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > self.eligible_count
            ):
                raise ValueError(
                    f"{field_name} must be a valid nonnegative count or None"
                )
        for field_name in ("contradiction_rate", "non_push_hit_rate"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _require_rate(value, field_name))
        _require_positive_integer(self.minimum_sample, "minimum_sample")
        if (
            not isinstance(self.trigger_reasons, tuple)
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in self.trigger_reasons
            )
            or tuple(sorted(set(self.trigger_reasons))) != self.trigger_reasons
        ):
            raise ValueError(
                "trigger_reasons must be a sorted tuple of unique nonempty strings"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "label": self.label,
            "status": self.status,
            "eligible_count": self.eligible_count,
            "excluded_count": self.excluded_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "push_count": self.push_count,
            "classification_counts": dict(self.classification_counts),
            "b_loss_count": self.b_loss_count,
            "contradiction_count": self.contradiction_count,
            "contradiction_rate": self.contradiction_rate,
            "non_push_hit_rate": self.non_push_hit_rate,
            "minimum_sample": self.minimum_sample,
            "trigger_reasons": list(self.trigger_reasons),
        }


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


def _require_exact_keys(
    value: dict[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = expected - actual
    if missing:
        raise ValueError(f"{name} missing keys: {', '.join(sorted(missing))}")
    unknown = actual - expected
    if unknown:
        raise ValueError(
            f"{name} has unknown or inapplicable keys: "
            + ", ".join(sorted(unknown))
        )


def load_review_watchlist_config(
    path: str | Path = DEFAULT_WATCHLIST_CONFIG_PATH,
) -> ReviewWatchlistConfig:
    """Load one strict, immutable watchlist configuration."""

    config_path = Path(path)
    try:
        decoded = config_path.read_text(encoding="utf-8")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("watchlist configuration is not valid JSON") from error
    except (OSError, UnicodeError) as error:
        raise ValueError("watchlist configuration is unreadable") from error

    if not isinstance(raw, dict):
        raise ValueError("watchlist configuration root must be a JSON object")
    _require_exact_keys(raw, frozenset(("schema_version", "targets")), "root")

    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise ValueError(
            f"schema_version must be the supported version {_SCHEMA_VERSION}"
        )

    raw_targets = raw["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("targets must be a nonempty JSON array")

    targets: list[WatchlistTargetConfig] = []
    for index, raw_target in enumerate(raw_targets):
        name = f"target[{index}]"
        if not isinstance(raw_target, dict):
            raise ValueError(f"{name} must be a JSON object")
        kind = raw_target.get("kind")
        if kind == "f5_over_conversion":
            expected_keys = _F5_OVER_TARGET_KEYS
        elif kind == "path_contradiction":
            expected_keys = _PATH_TARGET_KEYS
        else:
            raise ValueError(f"unsupported watchlist target kind: {kind!r}")
        _require_exact_keys(raw_target, expected_keys, name)
        targets.append(WatchlistTargetConfig(**raw_target))

    return ReviewWatchlistConfig(
        schema_version=schema_version,
        targets=tuple(targets),
    )


def _classification_counts(rows: tuple[CalibrationRow, ...]) -> dict[str, int]:
    counts = {classification: 0 for classification in _CLASSIFICATIONS}
    for row in rows:
        if row.classification in counts:
            counts[row.classification] += 1
    return counts


def _outcome_counts(rows: tuple[CalibrationRow, ...]) -> tuple[int, int, int]:
    return (
        sum(row.market_result == "win" for row in rows),
        sum(row.market_result == "loss" for row in rows),
        sum(row.market_result == "push" for row in rows),
    )


def _status(
    eligible_count: int,
    minimum_sample: int,
    trigger_reasons: tuple[str, ...],
) -> str:
    if eligible_count < minimum_sample:
        return "관찰"
    if trigger_reasons:
        return "정밀 검토"
    return "정상 범위"


def _f5_over_result(
    rows: tuple[CalibrationRow, ...],
    target: WatchlistTargetConfig,
) -> WatchlistResult:
    eligible: list[CalibrationRow] = []
    excluded_count = 0
    for row in rows:
        if (
            row.lineup_state != "confirmed"
            or row.market_type != "f5_total"
            or row.selected_side != "over"
        ):
            continue
        if row.classification is None:
            excluded_count += 1
            continue
        eligible.append(row)

    eligible_rows = tuple(eligible)
    win_count, loss_count, push_count = _outcome_counts(eligible_rows)
    non_push_count = win_count + loss_count
    non_push_hit_rate = (
        win_count / non_push_count if non_push_count else None
    )
    b_loss_count = sum(
        row.classification == "B" and row.market_result == "loss"
        for row in eligible_rows
    )

    trigger_reasons: list[str] = []
    if len(eligible_rows) >= target.minimum_sample:
        if (
            non_push_hit_rate is not None
            and non_push_hit_rate < target.hit_rate_below
        ):
            trigger_reasons.append("non_push_hit_rate_below")
        if (
            b_loss_count >= target.b_loss_count_min
            and b_loss_count / len(eligible_rows) >= target.b_loss_rate_min
        ):
            trigger_reasons.append("b_loss_count_and_rate")
    reasons = tuple(sorted(trigger_reasons))

    return WatchlistResult(
        target_id=target.target_id,
        label=target.label,
        status=_status(len(eligible_rows), target.minimum_sample, reasons),
        eligible_count=len(eligible_rows),
        excluded_count=excluded_count,
        win_count=win_count,
        loss_count=loss_count,
        push_count=push_count,
        classification_counts=_classification_counts(eligible_rows),
        b_loss_count=b_loss_count,
        contradiction_count=None,
        contradiction_rate=None,
        non_push_hit_rate=non_push_hit_rate,
        minimum_sample=target.minimum_sample,
        trigger_reasons=reasons,
    )


def _path_contradiction_result(
    rows: tuple[CalibrationRow, ...],
    target: WatchlistTargetConfig,
) -> WatchlistResult:
    eligible: list[CalibrationRow] = []
    for row in rows:
        if row.lineup_state != "confirmed":
            continue
        if target.evidence_path == "lineup_suppression" and (
            row.market_type not in ("f5_total", "full_total")
            or row.selected_side != "under"
        ):
            continue
        if target.evidence_path not in row.principal_paths:
            continue
        eligible.append(row)

    eligible_rows = tuple(eligible)
    win_count, loss_count, push_count = _outcome_counts(eligible_rows)
    contradiction_count = sum(
        target.evidence_path in row.contradicted_paths
        for row in eligible_rows
    )
    contradiction_rate = (
        contradiction_count / len(eligible_rows) if eligible_rows else None
    )

    trigger_reasons: list[str] = []
    if (
        len(eligible_rows) >= target.minimum_sample
        and contradiction_count >= target.contradiction_count_min
        and contradiction_rate >= target.contradiction_rate_min
    ):
        trigger_reasons.append("contradiction_count_and_rate")
    reasons = tuple(sorted(trigger_reasons))

    return WatchlistResult(
        target_id=target.target_id,
        label=target.label,
        status=_status(len(eligible_rows), target.minimum_sample, reasons),
        eligible_count=len(eligible_rows),
        excluded_count=0,
        win_count=win_count,
        loss_count=loss_count,
        push_count=push_count,
        classification_counts=_classification_counts(eligible_rows),
        b_loss_count=None,
        contradiction_count=contradiction_count,
        contradiction_rate=contradiction_rate,
        non_push_hit_rate=None,
        minimum_sample=target.minimum_sample,
        trigger_reasons=reasons,
    )


def build_review_watchlist(
    rows: tuple[CalibrationRow, ...],
    config: ReviewWatchlistConfig,
) -> tuple[WatchlistResult, ...]:
    """Evaluate configured targets from compact, already-selected review rows."""

    if not isinstance(rows, tuple):
        raise ValueError("rows must be a tuple")
    if not isinstance(config, ReviewWatchlistConfig):
        raise ValueError("config must be a ReviewWatchlistConfig")

    results: list[WatchlistResult] = []
    for target in config.targets:
        if target.kind == "f5_over_conversion":
            results.append(_f5_over_result(rows, target))
        elif target.kind == "path_contradiction":
            results.append(_path_contradiction_result(rows, target))
        else:
            raise ValueError(f"unsupported watchlist target kind: {target.kind!r}")
    return tuple(results)
