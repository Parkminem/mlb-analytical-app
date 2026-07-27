"""Deterministic forecast calibration and betting-performance reports."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

from analysis.forecasting.identity import canonical_json
from analysis.forecasting.inputs import MarketType, SettlementFormat
from analysis.forecasting.market import market_is_push_capable
from analysis.ledger.records import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    MarketObservationRecord,
    OfficialResultRecord,
    ReviewRecord,
    linked_shadow_comparison,
)
from analysis.ledger.store import LedgerCorruptionError


PROBABILITY_BUCKETS = (
    (0.00, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 1.0000001),
)
GROUPING_DIMENSIONS = (
    "model_version",
    "market_type",
    "lineup_state",
    "starter_structure",
    "probability_bucket",
    "priority_score_bucket",
)
_MARKET_RESULTS = frozenset(("win", "loss", "push"))
_MARKET_TYPES = frozenset(
    ("f5_side", "f5_total", "ml", "plus_1_5", "full_total")
)
_SETTLEMENT_FORMATS = frozenset(("two_way", "three_way"))
_TOTAL_MARKET_TYPES = frozenset(("f5_total", "full_total"))
_PROBABILITY_TOLERANCE = 1e-12
_AGGREGATION_TOLERANCE = 1e-12
_UNAVAILABLE = "unavailable"
PRIORITY_SCORE_BUCKETS = (
    (0, 69),
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 100),
)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_probability(value: object, field_name: str) -> float:
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability within [0, 1]")
    return float(value)


def _require_optional_probability(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_probability(value, field_name)


def _require_optional_decimal_price(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if not _is_finite_number(value) or value <= 1.0:
        raise ValueError(
            f"{field_name} must be a finite decimal price above one when present"
        )
    return float(value)


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _validated_market_line(market_type: str, line: object) -> float | None:
    if market_type == "ml":
        if line is not None:
            raise ValueError("ML line must be absent")
        return None
    if not _is_finite_number(line):
        raise ValueError(f"{market_type} line must be present and finite")
    value = float(line)
    if market_type in _TOTAL_MARKET_TYPES and value < 0.0:
        raise ValueError("total line must be nonnegative")
    if market_type == "plus_1_5" and value != 1.5:
        raise ValueError("plus_1_5 line must be exactly 1.5")
    return value


@dataclass(frozen=True)
class CaptureCutoffPolicy:
    name: str
    minimum_seconds_before_first_pitch: int = 0
    required_lineup_state: str | None = None
    include_later_superseded_captures: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "name")
        minimum_seconds = self.minimum_seconds_before_first_pitch
        if (
            isinstance(minimum_seconds, bool)
            or not isinstance(minimum_seconds, int)
            or minimum_seconds < 0
        ):
            raise ValueError(
                "minimum_seconds_before_first_pitch must be a nonnegative integer"
            )
        if self.required_lineup_state is not None:
            _require_nonempty_string(
                self.required_lineup_state,
                "required_lineup_state",
            )
        if not isinstance(self.include_later_superseded_captures, bool):
            raise ValueError(
                "include_later_superseded_captures must be a boolean"
            )
        if self.name == "latest_pregame" and (
            self.minimum_seconds_before_first_pitch != 0
            or self.required_lineup_state is not None
            or self.include_later_superseded_captures
        ):
            raise ValueError(
                "latest_pregame is reserved for the canonical headline policy"
            )


LATEST_PREGAME = CaptureCutoffPolicy(name="latest_pregame")


def _market_terms(
    game_pk: int,
    market_type: str,
    settlement_format: str,
    selected_side: str,
    line: float | None,
) -> tuple[int, str, str, str, float | None]:
    return (game_pk, market_type, settlement_format, selected_side, line)


@dataclass(frozen=True)
class _CalibrationLedgerIndex:
    captures_by_game: Mapping[int, tuple[CaptureRecord, ...]]
    capture_relationships: Mapping[str, str]
    links_by_capture: Mapping[str, tuple[CaptureForecastLink, ...]]
    forecasts_by_run_id: Mapping[str, ForecastRecord]
    latest_results_by_game: Mapping[int, OfficialResultRecord]
    reviews_by_link_result: Mapping[
        tuple[str, str],
        Mapping[str, tuple[ReviewRecord, ...]],
    ]
    close_observations_by_terms: Mapping[
        tuple[int, str, str, str, float | None],
        tuple[MarketObservationRecord, ...],
    ]
    market_observations_by_record_id: Mapping[str, MarketObservationRecord]
    market_relationships: Mapping[str, str]


def _add_relationship(
    relationships: dict[str, str],
    previous_record_id: str,
    successor_record_id: str,
    relationship_name: str,
) -> None:
    existing = relationships.setdefault(previous_record_id, successor_record_id)
    if existing != successor_record_id:
        raise LedgerCorruptionError(
            f"{relationship_name} supersession target has multiple successors"
        )


def _build_ledger_index(records: tuple[object, ...]) -> _CalibrationLedgerIndex:
    captures_by_game: dict[int, list[CaptureRecord]] = {}
    captures_by_capture_id: dict[str, CaptureRecord] = {}
    captures_by_record_id: dict[str, CaptureRecord] = {}
    links_by_capture: dict[str, list[CaptureForecastLink]] = {}
    forecasts_by_run_id: dict[str, ForecastRecord] = {}
    results_by_game: dict[int, list[OfficialResultRecord]] = {}
    reviews_by_link_result: dict[
        tuple[str, str],
        dict[str, list[ReviewRecord]],
    ] = {}
    close_observations_by_terms: dict[
        tuple[int, str, str, str, float | None],
        list[MarketObservationRecord],
    ] = {}
    market_observations_by_record_id: dict[str, MarketObservationRecord] = {}
    market_relationships: dict[str, str] = {}

    for record in records:
        if isinstance(record, CaptureRecord):
            if record.capture_id in captures_by_capture_id:
                raise LedgerCorruptionError("duplicate capture_id in ledger")
            captures_by_capture_id[record.capture_id] = record
            captures_by_record_id[record.envelope.record_id] = record
            captures_by_game.setdefault(record.game_pk, []).append(record)
        elif isinstance(record, CaptureForecastLink):
            links_by_capture.setdefault(record.capture_id, []).append(record)
        elif isinstance(record, ForecastRecord):
            existing = forecasts_by_run_id.setdefault(
                record.forecast_run_id,
                record,
            )
            if existing != record:
                raise LedgerCorruptionError("duplicate forecast_run_id in ledger")
        elif isinstance(record, OfficialResultRecord):
            results_by_game.setdefault(record.game_pk, []).append(record)
        elif isinstance(record, ReviewRecord):
            by_market = reviews_by_link_result.setdefault(
                (
                    record.capture_forecast_link_record_id,
                    record.official_result_record_id,
                ),
                {},
            )
            by_market.setdefault(record.market_key, []).append(record)
        elif isinstance(record, MarketObservationRecord):
            market_observations_by_record_id[record.envelope.record_id] = record
            if record.supersedes_record_id is not None:
                _add_relationship(
                    market_relationships,
                    record.supersedes_record_id,
                    record.envelope.record_id,
                    "market observation",
                )
            if record.observation_kind == "close":
                key = _market_terms(
                    record.game_pk,
                    record.market_type,
                    record.settlement_format,
                    record.selected_side,
                    record.line,
                )
                close_observations_by_terms.setdefault(key, []).append(record)

    capture_relationships: dict[str, str] = {}
    for capture in captures_by_record_id.values():
        if capture.supersedes_capture_id is None:
            continue
        previous = captures_by_capture_id.get(capture.supersedes_capture_id)
        if previous is None:
            raise LedgerCorruptionError(
                "capture supersession references a missing capture_id"
            )
        _add_relationship(
            capture_relationships,
            previous.envelope.record_id,
            capture.envelope.record_id,
            "capture",
        )

    latest_results_by_game: dict[int, OfficialResultRecord] = {}
    for game_pk, results in results_by_game.items():
        superseded_record_ids = {
            result.supersedes_record_id
            for result in results
            if result.supersedes_record_id is not None
        }
        latest_results = tuple(
            result
            for result in results
            if result.envelope.record_id not in superseded_record_ids
        )
        if len(latest_results) != 1:
            raise LedgerCorruptionError(
                "official result corrections do not have one latest record"
            )
        latest_results_by_game[game_pk] = latest_results[0]

    return _CalibrationLedgerIndex(
        captures_by_game={
            game_pk: tuple(captures)
            for game_pk, captures in captures_by_game.items()
        },
        capture_relationships=capture_relationships,
        links_by_capture={
            capture_id: tuple(
                sorted(
                    links,
                    key=lambda link: (
                        link.forecast_run_id,
                        link.envelope.record_id,
                    ),
                )
            )
            for capture_id, links in links_by_capture.items()
        },
        forecasts_by_run_id=forecasts_by_run_id,
        latest_results_by_game=latest_results_by_game,
        reviews_by_link_result={
            link_result: {
                market_key: tuple(reviews)
                for market_key, reviews in reviews_by_market.items()
            }
            for link_result, reviews_by_market in reviews_by_link_result.items()
        },
        close_observations_by_terms={
            terms: tuple(observations)
            for terms, observations in close_observations_by_terms.items()
        },
        market_observations_by_record_id=market_observations_by_record_id,
        market_relationships=market_relationships,
    )


@dataclass(frozen=True)
class CalibrationRow:
    game_pk: int
    capture_id: str
    forecast_run_id: str
    market_key: str
    model_version: str
    market_type: str
    settlement_format: str
    line: float | None
    lineup_state: str
    starter_structure: str
    legacy_priority_score: int | None
    p_win: float
    p_push: float
    p_loss: float
    market_result: str
    selected_decimal_price: float | None
    closing_selected_decimal_price: float | None
    entry_no_vig_probability: float | None
    closing_no_vig_probability: float | None
    push_capable: bool = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.game_pk, bool) or not isinstance(self.game_pk, int):
            raise ValueError("game_pk must be a positive integer")
        if self.game_pk <= 0:
            raise ValueError("game_pk must be a positive integer")
        for field_name in (
            "capture_id",
            "forecast_run_id",
            "market_key",
            "model_version",
            "market_type",
            "settlement_format",
            "lineup_state",
            "starter_structure",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)

        if self.market_type not in _MARKET_TYPES:
            raise ValueError(f"unsupported market_type: {self.market_type!r}")
        if self.settlement_format not in _SETTLEMENT_FORMATS:
            raise ValueError(
                f"unsupported settlement_format: {self.settlement_format!r}"
            )
        if self.market_type != "f5_side" and self.settlement_format != "two_way":
            raise ValueError(
                "settlement_format must be two_way for this market_type"
            )
        line = _validated_market_line(self.market_type, self.line)
        push_capable = market_is_push_capable(
            MarketType(self.market_type),
            SettlementFormat(self.settlement_format),
            line,
        )

        score = self.legacy_priority_score
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 100
        ):
            raise ValueError(
                "legacy_priority_score must be an integer within [0, 100] or None"
            )

        probabilities = {
            field_name: _require_probability(getattr(self, field_name), field_name)
            for field_name in ("p_win", "p_push", "p_loss")
        }
        for field_name, probability in probabilities.items():
            if probability == 1.0 and any(
                other_probability != 0.0
                for other_name, other_probability in probabilities.items()
                if other_name != field_name
            ):
                raise ValueError(
                    f"{field_name} exact-one probability requires zero residual mass"
                )
        if not math.isclose(
            math.fsum(probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("p_win, p_push, and p_loss must sum to one")
        if (
            not isinstance(self.market_result, str)
            or self.market_result not in _MARKET_RESULTS
        ):
            raise ValueError("market_result must be win, loss, or push")
        if self.market_result != "push" and probabilities["p_push"] == 1.0:
            raise ValueError("a non-push result requires non-push model probability")
        if not push_capable and (
            probabilities["p_push"] > 0.0 or self.market_result == "push"
        ):
            raise ValueError(
                "non-push-capable rows cannot contain push probability or outcomes"
            )

        object.__setattr__(self, "line", line)
        object.__setattr__(self, "push_capable", push_capable)
        for field_name, value in probabilities.items():
            object.__setattr__(self, field_name, value)
        for field_name in (
            "selected_decimal_price",
            "closing_selected_decimal_price",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_optional_decimal_price(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        for field_name in (
            "entry_no_vig_probability",
            "closing_no_vig_probability",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_optional_probability(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class CalibrationMetrics:
    sample_size: int
    wins: int
    losses: int
    pushes: int
    binary_sample_size: int
    multiclass_sample_size: int
    priced_stake_count: int
    probability_clv_sample_size: int
    price_clv_sample_size: int
    hit_rate: float | None
    push_rate: float | None
    binary_brier: float | None
    multiclass_brier: float | None
    log_loss: float | None
    profit_units: float | None
    roi: float | None
    mean_probability_clv: float | None
    mean_price_clv: float | None

    def __post_init__(self) -> None:
        count_fields = (
            "sample_size",
            "wins",
            "losses",
            "pushes",
            "binary_sample_size",
            "multiclass_sample_size",
            "priced_stake_count",
            "probability_clv_sample_size",
            "price_clv_sample_size",
        )
        for field_name in count_fields:
            _require_nonnegative_integer(getattr(self, field_name), field_name)
        if self.wins + self.losses + self.pushes != self.sample_size:
            raise ValueError("outcome counts must sum to sample_size")
        if self.binary_sample_size != self.wins + self.losses:
            raise ValueError(
                "binary_sample_size must equal the non-push outcome count"
            )
        for field_name in (
            "multiclass_sample_size",
            "priced_stake_count",
            "probability_clv_sample_size",
            "price_clv_sample_size",
        ):
            if getattr(self, field_name) > self.sample_size:
                raise ValueError(f"{field_name} cannot exceed sample_size")
        if self.multiclass_sample_size < self.pushes:
            raise ValueError(
                "multiclass_sample_size must include every push outcome"
            )
        if self.price_clv_sample_size > self.priced_stake_count:
            raise ValueError(
                "price_clv_sample_size cannot exceed priced_stake_count"
            )
        if self.sample_size == 0:
            if self.push_rate is not None:
                raise ValueError("push_rate must be None for an empty sample")
        else:
            expected_push_rate = self.pushes / self.sample_size
            if (
                not _is_finite_number(self.push_rate)
                or self.push_rate != expected_push_rate
            ):
                raise ValueError("push_rate must match outcome counts")

        if self.binary_sample_size == 0:
            if self.hit_rate is not None:
                raise ValueError("hit_rate must be None without non-push outcomes")
        else:
            expected_hit_rate = self.wins / self.binary_sample_size
            if (
                not _is_finite_number(self.hit_rate)
                or self.hit_rate != expected_hit_rate
            ):
                raise ValueError("hit_rate must match non-push outcome counts")

        bounded_fields = {
            "binary_brier": (0.0, 1.0),
            "multiclass_brier": (0.0, 2.0),
        }
        for field_name, (minimum, maximum) in bounded_fields.items():
            value = getattr(self, field_name)
            if value is not None and (
                not _is_finite_number(value) or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{field_name} must be finite within [{minimum}, {maximum}]"
                )
        if self.log_loss is not None and (
            not _is_finite_number(self.log_loss) or self.log_loss < 0.0
        ):
            raise ValueError("log_loss must be a finite nonnegative number")
        if self.binary_sample_size == 0:
            if self.binary_brier is not None:
                raise ValueError(
                    "binary_brier must be None without non-push outcomes"
                )
            if self.log_loss is not None:
                raise ValueError("log_loss must be None without non-push outcomes")
        else:
            if self.binary_brier is None:
                raise ValueError(
                    "binary_brier is required with non-push outcomes"
                )
            if self.log_loss is None:
                raise ValueError("log_loss is required with non-push outcomes")
        if (self.multiclass_brier is None) != (
            self.multiclass_sample_size == 0
        ):
            requirement = "None" if self.multiclass_sample_size == 0 else "required"
            raise ValueError(
                "multiclass_brier must be "
                f"{requirement} for multiclass_sample_size"
            )
        for field_name in (
            "profit_units",
            "roi",
            "mean_probability_clv",
            "mean_price_clv",
        ):
            value = getattr(self, field_name)
            if value is not None and not _is_finite_number(value):
                raise ValueError(f"{field_name} must be finite when present")
        if (self.profit_units is None) != (self.roi is None):
            raise ValueError("profit_units and roi must be available together")
        if (self.profit_units is None) != (self.priced_stake_count == 0):
            requirement = "None" if self.priced_stake_count == 0 else "available"
            raise ValueError(
                f"profit_units and roi must be {requirement} for priced_stake_count"
            )
        if self.roi is not None and not math.isclose(
            self.roi,
            self.profit_units / self.priced_stake_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("roi must equal profit_units divided by priced_stake_count")
        availability_fields = (
            (
                "mean_probability_clv",
                self.probability_clv_sample_size,
            ),
            ("mean_price_clv", self.price_clv_sample_size),
        )
        for field_name, denominator in availability_fields:
            if (getattr(self, field_name) is None) != (denominator == 0):
                requirement = "None" if denominator == 0 else "required"
                raise ValueError(
                    f"{field_name} must be {requirement} when its sample size is "
                    f"{denominator}"
                )
        if self.sample_size == 0:
            unavailable_fields = (
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
            if any(
                getattr(self, field_name) is not None
                for field_name in unavailable_fields
            ):
                raise ValueError("empty metrics cannot contain calculated values")


@dataclass(frozen=True)
class CalibrationGroup:
    dimensions: tuple[tuple[str, str], ...]
    metrics: CalibrationMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, tuple):
            raise ValueError("dimensions must be a tuple")
        names: list[str] = []
        for dimension in self.dimensions:
            if not isinstance(dimension, tuple) or len(dimension) != 2:
                raise ValueError("dimensions must contain name-value pairs")
            name, value = dimension
            _require_nonempty_string(name, "dimension name")
            _require_nonempty_string(value, "dimension value")
            if name not in GROUPING_DIMENSIONS:
                raise ValueError(f"unknown grouping dimension: {name!r}")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("dimensions must have unique names")
        if not isinstance(self.metrics, CalibrationMetrics):
            raise ValueError("metrics must be CalibrationMetrics")


@dataclass(frozen=True)
class CalibrationReport:
    overall: CalibrationMetrics
    groups: tuple[CalibrationGroup, ...]
    group_by: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.overall, CalibrationMetrics):
            raise ValueError("overall must be CalibrationMetrics")
        _validate_group_by(self.group_by)
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, CalibrationGroup) for group in self.groups
        ):
            raise ValueError("groups must be a tuple of CalibrationGroup values")
        if not self.group_by and self.groups:
            raise ValueError("groups must be empty when group_by is empty")
        if self.group_by and self.overall.sample_size > 0 and not self.groups:
            raise ValueError("groups must partition a nonempty overall sample")
        expected_names = self.group_by
        if any(
            tuple(name for name, _ in group.dimensions) != expected_names
            for group in self.groups
        ):
            raise ValueError("group dimensions must match group_by")
        dimension_values = tuple(
            tuple(value for _, value in group.dimensions) for group in self.groups
        )
        if len(set(dimension_values)) != len(dimension_values):
            raise ValueError("groups must have unique dimension values")
        if dimension_values != tuple(sorted(dimension_values)):
            raise ValueError("groups must be deterministically sorted")
        if not self.groups:
            return
        partition_fields = (
            "sample_size",
            "wins",
            "losses",
            "pushes",
            "binary_sample_size",
            "multiclass_sample_size",
            "priced_stake_count",
            "probability_clv_sample_size",
            "price_clv_sample_size",
        )
        for field_name in partition_fields:
            grouped_count = sum(
                getattr(group.metrics, field_name) for group in self.groups
            )
            if grouped_count != getattr(self.overall, field_name):
                raise ValueError(
                    f"group {field_name} values must sum to overall {field_name}"
                )

        if self.overall.priced_stake_count > 0:
            grouped_profit = math.fsum(
                group.metrics.profit_units
                for group in self.groups
                if group.metrics.priced_stake_count > 0
            )
            if not math.isclose(
                grouped_profit,
                self.overall.profit_units,
                rel_tol=_AGGREGATION_TOLERANCE,
                abs_tol=_AGGREGATION_TOLERANCE,
            ):
                raise ValueError(
                    "group profit_units values must sum to overall profit_units"
                )

        weighted_metrics = (
            ("hit_rate", "binary_sample_size"),
            ("push_rate", "sample_size"),
            ("binary_brier", "binary_sample_size"),
            ("multiclass_brier", "multiclass_sample_size"),
            ("log_loss", "binary_sample_size"),
            ("roi", "priced_stake_count"),
            ("mean_probability_clv", "probability_clv_sample_size"),
            ("mean_price_clv", "price_clv_sample_size"),
        )
        for metric_name, denominator_name in weighted_metrics:
            denominator = getattr(self.overall, denominator_name)
            if denominator == 0:
                continue
            weighted_total = math.fsum(
                getattr(group.metrics, metric_name)
                * getattr(group.metrics, denominator_name)
                for group in self.groups
                if getattr(group.metrics, denominator_name) > 0
            )
            reconstructed = weighted_total / denominator
            if not math.isclose(
                reconstructed,
                getattr(self.overall, metric_name),
                rel_tol=_AGGREGATION_TOLERANCE,
                abs_tol=_AGGREGATION_TOLERANCE,
            ):
                raise ValueError(
                    f"group-weighted {metric_name} must reconstruct overall "
                    f"{metric_name}"
                )


def probability_bucket(probability: float) -> str:
    """Return the fixed lower-inclusive model-probability bucket."""

    if not _is_finite_number(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite number within [0, 1]")
    labels = (
        "0.00-0.45",
        "0.45-0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-1.00",
    )
    for (lower, upper), label in zip(PROBABILITY_BUCKETS, labels):
        if lower <= probability < upper:
            return label
    raise ValueError("probability does not fall in a configured bucket")


def priority_score_bucket(score: int) -> str:
    """Return the fixed inclusive legacy-priority bucket."""

    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("priority score must be an integer within [0, 100]")
    for lower, upper in PRIORITY_SCORE_BUCKETS:
        if lower <= score <= upper:
            return f"{lower}-{upper}"
    raise ValueError("priority score does not fall in a configured bucket")


def _validate_group_by(group_by: object) -> tuple[str, ...]:
    if not isinstance(group_by, tuple):
        raise ValueError("group_by must be a tuple")
    for dimension in group_by:
        if not isinstance(dimension, str) or dimension not in GROUPING_DIMENSIONS:
            raise ValueError(f"unknown grouping dimension: {dimension!r}")
    if len(set(group_by)) != len(group_by):
        raise ValueError("group_by dimensions must be unique")
    return group_by


def _conditional_probability(row: CalibrationRow) -> float | None:
    non_push_probability = math.fsum((row.p_win, row.p_loss))
    if non_push_probability == 0.0:
        return None
    if row.p_win == 0.0:
        return 0.0
    if row.p_loss == 0.0:
        return 1.0
    return row.p_win / non_push_probability


def _dimension_value(row: CalibrationRow, dimension: str) -> str:
    if dimension == "probability_bucket":
        probability = _conditional_probability(row)
        return _UNAVAILABLE if probability is None else probability_bucket(probability)
    if dimension == "priority_score_bucket":
        score = row.legacy_priority_score
        return _UNAVAILABLE if score is None else priority_score_bucket(score)
    value = getattr(row, dimension)
    if not isinstance(value, str):
        raise ValueError(f"{dimension} must resolve to a string")
    return value


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.fsum(values) / len(values)


def _metrics_for_rows(rows: tuple[CalibrationRow, ...]) -> CalibrationMetrics:
    wins = sum(row.market_result == "win" for row in rows)
    losses = sum(row.market_result == "loss" for row in rows)
    pushes = sum(row.market_result == "push" for row in rows)
    non_push_count = wins + losses

    binary_brier_values: list[float] = []
    log_loss_values: list[float] = []
    multiclass_brier_values: list[float] = []
    priced_profits: list[float] = []
    probability_clv_values: list[float] = []
    price_clv_values: list[float] = []

    for row in rows:
        if row.push_capable:
            outcome = {
                "win": (1.0, 0.0, 0.0),
                "push": (0.0, 1.0, 0.0),
                "loss": (0.0, 0.0, 1.0),
            }[row.market_result]
            multiclass_brier_values.append(
                math.fsum(
                    (
                        (row.p_win - outcome[0]) ** 2,
                        (row.p_push - outcome[1]) ** 2,
                        (row.p_loss - outcome[2]) ** 2,
                    )
                )
            )

        if row.market_result != "push":
            conditional_probability = _conditional_probability(row)
            if conditional_probability is None:
                raise ValueError(
                    "non-push outcomes require conditional model probability"
                )
            binary_outcome = 1.0 if row.market_result == "win" else 0.0
            binary_brier_values.append(
                (conditional_probability - binary_outcome) ** 2
            )
            clipped = min(max(conditional_probability, 1e-15), 1.0 - 1e-15)
            log_loss_values.append(
                -(
                    binary_outcome * math.log(clipped)
                    + (1.0 - binary_outcome) * math.log(1.0 - clipped)
                )
            )

        if row.selected_decimal_price is not None:
            profit = {
                "win": row.selected_decimal_price - 1.0,
                "loss": -1.0,
                "push": 0.0,
            }[row.market_result]
            priced_profits.append(profit)

        if (
            row.entry_no_vig_probability is not None
            and row.closing_no_vig_probability is not None
        ):
            probability_clv_values.append(
                row.closing_no_vig_probability
                - row.entry_no_vig_probability
            )
        if (
            row.selected_decimal_price is not None
            and row.closing_selected_decimal_price is not None
        ):
            price_clv_values.append(
                row.selected_decimal_price
                / row.closing_selected_decimal_price
                - 1.0
            )

    profit_units = None if not priced_profits else math.fsum(priced_profits)
    roi = None if profit_units is None else profit_units / len(priced_profits)
    return CalibrationMetrics(
        sample_size=len(rows),
        wins=wins,
        losses=losses,
        pushes=pushes,
        binary_sample_size=len(binary_brier_values),
        multiclass_sample_size=len(multiclass_brier_values),
        priced_stake_count=len(priced_profits),
        probability_clv_sample_size=len(probability_clv_values),
        price_clv_sample_size=len(price_clv_values),
        hit_rate=None if non_push_count == 0 else wins / non_push_count,
        push_rate=None if not rows else pushes / len(rows),
        binary_brier=_mean(binary_brier_values),
        multiclass_brier=_mean(multiclass_brier_values),
        log_loss=_mean(log_loss_values),
        profit_units=profit_units,
        roi=roi,
        mean_probability_clv=_mean(probability_clv_values),
        mean_price_clv=_mean(price_clv_values),
    )


def build_calibration_report(
    rows: tuple[CalibrationRow, ...],
    group_by: tuple[str, ...] = GROUPING_DIMENSIONS,
) -> CalibrationReport:
    """Aggregate validated calibration rows into deterministic fixed groups."""

    if not isinstance(rows, tuple) or any(
        not isinstance(row, CalibrationRow) for row in rows
    ):
        raise ValueError("rows must be a tuple of CalibrationRow values")
    dimensions = _validate_group_by(group_by)
    grouped_rows: dict[tuple[str, ...], list[CalibrationRow]] = {}
    if dimensions:
        for row in rows:
            key = tuple(_dimension_value(row, dimension) for dimension in dimensions)
            grouped_rows.setdefault(key, []).append(row)
    groups = tuple(
        CalibrationGroup(
            dimensions=tuple(zip(dimensions, key)),
            metrics=_metrics_for_rows(tuple(grouped_rows[key])),
        )
        for key in sorted(grouped_rows)
    )
    return CalibrationReport(
        overall=_metrics_for_rows(rows),
        groups=groups,
        group_by=dimensions,
    )


def _capture_is_eligible(
    capture: CaptureRecord,
    policy: CaptureCutoffPolicy,
) -> bool:
    if not capture.out_of_sample_eligible or capture.captured_at >= capture.first_pitch:
        return False
    cutoff = capture.first_pitch - timedelta(
        seconds=policy.minimum_seconds_before_first_pitch
    )
    if capture.captured_at > cutoff:
        return False
    return (
        policy.required_lineup_state is None
        or capture.lineup_state == policy.required_lineup_state
    )


def _has_eligible_successor(
    record_id: str,
    relationships: Mapping[str, object],
    eligible_record_ids: frozenset[str],
    cache: dict[str, bool] | None = None,
) -> bool:
    resolved = {} if cache is None else cache
    path: list[str] = []
    visited: set[str] = set()
    current = record_id
    while current not in resolved:
        if current in visited:
            raise ValueError("capture supersession relationships cannot contain cycles")
        visited.add(current)
        path.append(current)
        successor = relationships.get(current)
        if successor is None:
            resolved[current] = False
            break
        if not isinstance(successor, str) or not successor:
            raise ValueError(
                "capture supersession relationships must contain record IDs"
            )
        current = successor
    for current in reversed(path):
        if current in resolved:
            continue
        successor = relationships[current]
        resolved[current] = (
            successor in eligible_record_ids or resolved[successor]
        )
    return resolved[record_id]


def _has_successor_at_or_before_boundary(
    record_id: str,
    relationships: Mapping[str, object],
    captures_by_record_id: Mapping[str, CaptureRecord],
    boundary: datetime,
    cache: dict[str, bool] | None = None,
) -> bool:
    resolved = {} if cache is None else cache
    path: list[str] = []
    visited: set[str] = set()
    current = record_id
    while current not in resolved:
        if current in visited:
            raise ValueError("capture supersession relationships cannot contain cycles")
        visited.add(current)
        path.append(current)
        successor_id = relationships.get(current)
        if successor_id is None:
            resolved[current] = False
            break
        if not isinstance(successor_id, str) or not successor_id:
            raise ValueError(
                "capture supersession relationships must contain record IDs"
            )
        successor = captures_by_record_id.get(successor_id)
        if successor is None:
            raise ValueError(
                "capture supersession relationship references missing record"
            )
        current = successor_id
    for current in reversed(path):
        if current in resolved:
            continue
        successor_id = relationships[current]
        successor = captures_by_record_id[successor_id]
        resolved[current] = (
            successor.captured_at <= boundary or resolved[successor_id]
        )
    return resolved[record_id]


def _select_primary_captures(
    captures_by_game: Mapping[int, tuple[CaptureRecord, ...]],
    relationships: Mapping[str, object],
    policy: CaptureCutoffPolicy,
) -> tuple[CaptureRecord, ...]:
    eligible = tuple(
        capture
        for captures in captures_by_game.values()
        for capture in captures
        if _capture_is_eligible(capture, policy)
    )
    eligible_record_ids = frozenset(
        capture.envelope.record_id for capture in eligible
    )
    if policy.include_later_superseded_captures:
        captures_by_record_id = {
            capture.envelope.record_id: capture
            for captures in captures_by_game.values()
            for capture in captures
        }
        caches_by_boundary: dict[datetime, dict[str, bool]] = {}
        eligible = tuple(
            capture
            for capture in eligible
            if not _has_successor_at_or_before_boundary(
                capture.envelope.record_id,
                relationships,
                captures_by_record_id,
                capture.first_pitch
                - timedelta(seconds=policy.minimum_seconds_before_first_pitch),
                caches_by_boundary.setdefault(
                    capture.first_pitch
                    - timedelta(seconds=policy.minimum_seconds_before_first_pitch),
                    {},
                ),
            )
        )
    else:
        successor_cache: dict[str, bool] = {}
        eligible = tuple(
            capture
            for capture in eligible
            if not _has_eligible_successor(
                capture.envelope.record_id,
                relationships,
                eligible_record_ids,
                successor_cache,
            )
        )

    by_game: dict[int, list[CaptureRecord]] = {}
    for capture in eligible:
        by_game.setdefault(capture.game_pk, []).append(capture)

    selected: list[CaptureRecord] = []
    for game_pk in sorted(by_game):
        game_captures = by_game[game_pk]
        latest_timestamp = max(capture.captured_at for capture in game_captures)
        selected.append(
            min(
                (
                    capture
                    for capture in game_captures
                    if capture.captured_at == latest_timestamp
                ),
                key=lambda capture: capture.capture_id,
            )
        )
    return tuple(selected)


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _market_mappings(value: object, field_name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        markets: dict[str, object] = {}
        for item in value:
            market = _require_mapping(item, field_name)
            key = market.get("market_key")
            _require_nonempty_string(key, "market_key")
            if key in markets:
                raise ValueError(f"{field_name} contains duplicate market_key")
            markets[key] = market
        return markets
    raise ValueError(f"{field_name} must be a mapping or sequence of markets")


def _forecast_market(
    outputs: Mapping[str, object],
    market_key: str,
) -> Mapping[str, object]:
    roots: list[Mapping[str, object]] = [outputs]
    for field_name in ("forecast_output", "forecast"):
        nested = outputs.get(field_name)
        if nested is not None:
            roots.append(_require_mapping(nested, field_name))
    matching_markets: list[Mapping[str, object]] = []
    for root in roots:
        if "markets" not in root:
            continue
        markets = _market_mappings(root["markets"], "forecast markets")
        if market_key not in markets:
            continue
        matching_markets.append(
            _require_mapping(
                markets[market_key],
                f"forecast market {market_key}",
            )
        )
    if not matching_markets:
        raise ValueError(f"forecast output is missing market_key {market_key}")
    canonical_markets = {
        canonical_json(market) for market in matching_markets
    }
    if len(canonical_markets) != 1:
        raise ValueError(
            f"conflicting forecast market mappings for market_key {market_key}"
        )
    return matching_markets[0]


def _review_semantic_json(review: ReviewRecord) -> str:
    return canonical_json(
        {
            "capture_forecast_link_record_id": (
                review.capture_forecast_link_record_id
            ),
            "capture_id": review.capture_id,
            "forecast_run_id": review.forecast_run_id,
            "official_result_record_id": review.official_result_record_id,
            "market_key": review.market_key,
            "market_result": review.market_result,
            "classification": review.classification,
            "inside_80_ratio": review.inside_80_ratio,
            "contradicted_paths": review.contradicted_paths,
            "component_reviews": review.component_reviews,
        }
    )


def _capture_market(
    payload: Mapping[str, object],
    market_key: str,
) -> Mapping[str, object]:
    markets_value = payload.get("markets")
    markets = _market_mappings(markets_value, "capture markets")
    market = markets.get(market_key)
    if market is None:
        raise ValueError(f"capture payload is missing market_key {market_key}")
    return _require_mapping(market, f"capture market {market_key}")


def _market_probabilities(
    market: Mapping[str, object],
) -> tuple[object, object, object]:
    probabilities = market.get("probabilities")
    if probabilities is not None:
        values = _require_mapping(probabilities, "market probabilities")
        return values.get("win"), values.get("push"), values.get("loss")
    return market.get("p_win"), market.get("p_push"), market.get("p_loss")


def _legacy_priority_score(
    outputs: Mapping[str, object],
    market: Mapping[str, object],
    market_key: str,
    linked_comparison: object,
    *,
    corrected_link: bool,
) -> object:
    def score_from_roots(
        roots: tuple[Mapping[str, object], ...],
    ) -> tuple[bool, object]:
        for root in roots:
            comparisons = root.get("market_comparisons")
            if comparisons is None:
                continue
            if isinstance(comparisons, Mapping):
                comparison = comparisons.get(market_key)
                candidates = () if comparison is None else (comparison,)
            elif isinstance(comparisons, Sequence) and not isinstance(
                comparisons, (str, bytes)
            ):
                candidates = comparisons
            else:
                raise ValueError(
                    "market_comparisons must be a mapping or sequence"
                )
            for candidate in candidates:
                comparison = _require_mapping(
                    candidate,
                    "market comparison",
                )
                if comparison.get("market_key") != market_key and not (
                    isinstance(comparisons, Mapping)
                    and len(candidates) == 1
                ):
                    continue
                if "legacy_score" in comparison:
                    return True, comparison["legacy_score"]
                if "legacy_priority_score" in comparison:
                    return True, comparison["legacy_priority_score"]
        return False, None

    if corrected_link and linked_comparison is not None:
        found, score = score_from_roots(
            (
                _require_mapping(
                    linked_comparison,
                    "linked shadow comparison",
                ),
            )
        )
        if found:
            return score
    if "legacy_priority_score" in market:
        return market["legacy_priority_score"]
    comparison_roots: list[Mapping[str, object]] = []
    for field_name in ("shadow_comparison", "comparison", "shadow"):
        value = outputs.get(field_name)
        if value is not None:
            comparison_roots.append(_require_mapping(value, field_name))
    if "market_comparisons" in outputs:
        comparison_roots.append(outputs)
    _, score = score_from_roots(tuple(comparison_roots))
    return score


def _pitching_structure(
    payload: Mapping[str, object],
    side: str,
) -> str | None:
    pitching = payload.get(f"{side}_pitching")
    if not isinstance(pitching, Mapping):
        return None
    pitchers = pitching.get("pitchers")
    if not isinstance(pitchers, Sequence) or isinstance(pitchers, (str, bytes)):
        return None
    roles: list[str] = []
    for pitcher_value in pitchers:
        if not isinstance(pitcher_value, Mapping):
            return None
        role = pitcher_value.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        roles.append(role)
    return None if not roles else "_".join(roles)


def _starter_structure(payload: Mapping[str, object]) -> str:
    explicit = payload.get("starter_structure")
    if explicit is not None:
        _require_nonempty_string(explicit, "starter_structure")
        return explicit
    away = _pitching_structure(payload, "away")
    home = _pitching_structure(payload, "home")
    if away is None or home is None:
        return _UNAVAILABLE
    return f"{away}_vs_{home}"


def _matching_close(
    capture: CaptureRecord,
    market: Mapping[str, object],
    observations_by_terms: Mapping[
        tuple[int, str, str, str, float | None],
        tuple[MarketObservationRecord, ...],
    ],
    observations_by_record_id: Mapping[str, MarketObservationRecord],
    relationships: Mapping[str, object],
) -> MarketObservationRecord | None:
    market_type = market.get("market_type")
    settlement_format = market.get("settlement_format")
    selected_side = market.get("selected_side")
    line = market.get("line")
    for value, field_name in (
        (market_type, "market_type"),
        (settlement_format, "settlement_format"),
        (selected_side, "selected_side"),
    ):
        _require_nonempty_string(value, field_name)
    observations = observations_by_terms.get(
        _market_terms(
            capture.game_pk,
            market_type,
            settlement_format,
            selected_side,
            line,
        ),
        (),
    )
    successor_cache: dict[str, bool] = {}

    def has_eligible_successor(record_id: str) -> bool:
        path: list[str] = []
        visited: set[str] = set()
        current = record_id
        while current not in successor_cache:
            if current in visited:
                raise ValueError(
                    "market supersession relationships cannot contain cycles"
                )
            visited.add(current)
            path.append(current)
            successor_id = relationships.get(current)
            if successor_id is None:
                successor_cache[current] = False
                break
            if not isinstance(successor_id, str) or not successor_id:
                raise ValueError(
                    "market supersession relationships must contain record IDs"
                )
            successor = observations_by_record_id.get(successor_id)
            if successor is None:
                raise ValueError(
                    "market supersession relationship references missing record"
                )
            current = successor_id
        for current in reversed(path):
            if current in successor_cache:
                continue
            successor_id = relationships[current]
            successor = observations_by_record_id[successor_id]
            successor_is_eligible = (
                capture.captured_at <= successor.observed_at
                <= capture.first_pitch
                and successor.available_at is not None
                and capture.captured_at
                <= successor.available_at
                <= capture.first_pitch
            )
            successor_cache[current] = (
                successor_is_eligible or successor_cache[successor_id]
            )
        return successor_cache[record_id]

    matches = [
        observation
        for observation in observations
        if capture.captured_at
        <= observation.observed_at
        <= capture.first_pitch
        and observation.available_at is not None
        and capture.captured_at <= observation.available_at <= capture.first_pitch
        and not has_eligible_successor(observation.envelope.record_id)
    ]
    if not matches:
        return None
    latest_timestamp = max(observation.observed_at for observation in matches)
    latest_matches = tuple(
        observation
        for observation in matches
        if observation.observed_at == latest_timestamp
    )
    quote_values = {
        (
            observation.selected_decimal_price,
            observation.opposing_decimal_price,
        )
        for observation in latest_matches
    }
    if len(quote_values) != 1:
        raise ValueError(
            "conflicting closing quotes exist at the latest eligible timestamp"
        )
    return min(
        latest_matches,
        key=lambda observation: (
            observation.source_name,
            observation.source_record,
            observation.source_revision,
            observation.market_key,
        ),
    )


def _no_vig_probability(
    selected_price: object,
    opposing_price: object,
    field_prefix: str,
) -> float | None:
    selected = _require_optional_decimal_price(
        selected_price,
        f"{field_prefix}_selected_decimal_price",
    )
    opposing = _require_optional_decimal_price(
        opposing_price,
        f"{field_prefix}_opposing_decimal_price",
    )
    if selected is None or opposing is None:
        return None
    selected_implied = 1.0 / selected
    opposing_implied = 1.0 / opposing
    return selected_implied / (selected_implied + opposing_implied)


def _row_for_review(
    capture: CaptureRecord,
    forecast,
    link: CaptureForecastLink,
    review: ReviewRecord,
    ledger_index: _CalibrationLedgerIndex,
) -> CalibrationRow:
    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    forecast_market = _forecast_market(outputs, review.market_key)
    capture_payload = _require_mapping(capture.payload, "capture payload")
    capture_market = _capture_market(capture_payload, review.market_key)
    p_win, p_push, p_loss = _market_probabilities(forecast_market)

    selected_price = forecast_market.get(
        "selected_decimal_price",
        capture_market.get("selected_decimal_price"),
    )
    opposing_price = forecast_market.get(
        "opposing_decimal_price",
        capture_market.get("opposing_decimal_price"),
    )
    if "market_no_vig_probability" in forecast_market:
        entry_no_vig = forecast_market["market_no_vig_probability"]
    else:
        entry_no_vig = _no_vig_probability(
            selected_price,
            opposing_price,
            "entry",
        )

    close = _matching_close(
        capture,
        capture_market,
        ledger_index.close_observations_by_terms,
        ledger_index.market_observations_by_record_id,
        ledger_index.market_relationships,
    )
    closing_price = None if close is None else close.selected_decimal_price
    closing_no_vig = (
        None
        if close is None
        else _no_vig_probability(
            close.selected_decimal_price,
            close.opposing_decimal_price,
            "closing",
        )
    )
    market_type = capture_market.get("market_type")
    _require_nonempty_string(market_type, "market_type")
    settlement_format = capture_market.get("settlement_format")
    _require_nonempty_string(settlement_format, "settlement_format")
    return CalibrationRow(
        game_pk=capture.game_pk,
        capture_id=capture.capture_id,
        forecast_run_id=forecast.forecast_run_id,
        market_key=review.market_key,
        model_version=forecast.model_version,
        market_type=market_type,
        settlement_format=settlement_format,
        line=capture_market.get("line"),
        lineup_state=capture.lineup_state,
        starter_structure=_starter_structure(capture_payload),
        legacy_priority_score=_legacy_priority_score(
            outputs,
            forecast_market,
            review.market_key,
            linked_shadow_comparison(link, forecast),
            corrected_link=link.supersedes_link_record_id is not None,
        ),
        p_win=p_win,
        p_push=p_push,
        p_loss=p_loss,
        market_result=review.market_result,
        selected_decimal_price=selected_price,
        closing_selected_decimal_price=closing_price,
        entry_no_vig_probability=entry_no_vig,
        closing_no_vig_probability=closing_no_vig,
    )


def select_primary_rows(
    store,
    cutoff_policy: CaptureCutoffPolicy = LATEST_PREGAME,
) -> tuple[CalibrationRow, ...]:
    """Build one deterministic, leakage-safe calibration row per forecast market."""

    if not isinstance(cutoff_policy, CaptureCutoffPolicy):
        raise ValueError("cutoff_policy must be CaptureCutoffPolicy")
    ledger_index = _build_ledger_index(tuple(store.iter_records()))
    primary_captures = _select_primary_captures(
        ledger_index.captures_by_game,
        ledger_index.capture_relationships,
        cutoff_policy,
    )

    candidate_rows: list[CalibrationRow] = []
    for capture in primary_captures:
        latest_result = ledger_index.latest_results_by_game.get(capture.game_pk)
        if latest_result is None:
            continue
        capture_links = ledger_index.links_by_capture.get(
            capture.capture_id,
            (),
        )
        superseded_link_record_ids = frozenset(
            link.supersedes_link_record_id
            for link in capture_links
            if link.supersedes_link_record_id is not None
        )
        links = tuple(
            link
            for link in capture_links
            if link.envelope.record_id not in superseded_link_record_ids
        )
        for link in links:
            forecast = ledger_index.forecasts_by_run_id.get(link.forecast_run_id)
            if forecast is None:
                raise LedgerCorruptionError("link references a missing forecast")
            indexed_reviews = ledger_index.reviews_by_link_result.get(
                (
                    link.envelope.record_id,
                    latest_result.envelope.record_id,
                ),
                {},
            )
            exact_reviews: list[ReviewRecord] = []
            for market_key in sorted(indexed_reviews):
                matching_reviews = tuple(
                    review
                    for review in indexed_reviews[market_key]
                    if review.capture_id == capture.capture_id
                    and review.forecast_run_id == forecast.forecast_run_id
                )
                if not matching_reviews:
                    continue
                canonical_reviews: dict[str, ReviewRecord] = {}
                for review in matching_reviews:
                    canonical_reviews.setdefault(
                        _review_semantic_json(review),
                        review,
                    )
                if len(canonical_reviews) != 1:
                    raise ValueError(
                        "conflicting review records for one link, result, "
                        f"and market: {market_key}"
                    )
                exact_reviews.append(next(iter(canonical_reviews.values())))
            candidate_rows.extend(
                _row_for_review(
                    capture,
                    forecast,
                    link,
                    review,
                    ledger_index,
                )
                for review in exact_reviews
            )

    deduplicated: dict[tuple[str, str], CalibrationRow] = {}
    for row in candidate_rows:
        key = (row.forecast_run_id, row.market_key)
        existing = deduplicated.setdefault(key, row)
        if existing != row:
            raise ValueError(
                "conflicting calibration rows share forecast_run_id and market_key"
            )
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda row: (
                row.game_pk,
                row.forecast_run_id,
                row.market_key,
                row.capture_id,
            ),
        )
    )


__all__ = (
    "CalibrationGroup",
    "CalibrationMetrics",
    "CalibrationReport",
    "CalibrationRow",
    "CaptureCutoffPolicy",
    "GROUPING_DIMENSIONS",
    "LATEST_PREGAME",
    "PRIORITY_SCORE_BUCKETS",
    "PROBABILITY_BUCKETS",
    "build_calibration_report",
    "priority_score_bucket",
    "probability_bucket",
    "select_primary_rows",
)
