"""Immutable commands and records for the append-only ledger."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
import math

from analysis.model import PriorityEvidencePath


_LOWERCASE_HEX = frozenset("0123456789abcdef")
_MARKET_RESULTS = frozenset(("win", "loss", "push"))
_REVIEW_CLASSIFICATIONS = frozenset(("A", "B", "C", "D"))
_COMPONENT_REVIEW_KEYS = frozenset(
    (
        "evidence_path",
        "component_ids",
        "contradicted",
        "inside_80_count",
        "component_count",
        "notes",
    )
)
_PRIORITY_EVIDENCE_PATHS = frozenset(
    evidence_path.value for evidence_path in PriorityEvidencePath
)
_HORIZON_SCORING_EVIDENCE_PATHS = frozenset(
    (
        PriorityEvidencePath.LINEUP_SUPPRESSION.value,
        PriorityEvidencePath.TOP_ORDER_MATCHUP.value,
        PriorityEvidencePath.RECENT_RUN_CONVERSION.value,
        PriorityEvidencePath.UNDERDOG_RUN_CREATION.value,
    )
)


def _require_nonempty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_generated_id(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(
            f"{field_name} must be exactly 32 lowercase hexadecimal characters"
        )


def _require_aware_datetime(value: object, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


def _require_game_pk(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("game_pk must be a positive integer")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def _json_values_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if len(left) != len(right):
            return False
        return all(
            key in right and _json_values_equal(value, right[key])
            for key, value in left.items()
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right,
            (list, tuple),
        ):
            return False
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if _is_finite_number(left) or _is_finite_number(right):
        return _is_finite_number(left) and _is_finite_number(right) and left == right
    return False


class FrozenMapping(Mapping[str, object]):
    """Small immutable mapping backed only by recursively frozen tuples."""

    __slots__ = ("_items",)

    def __init__(self, items: tuple[tuple[str, object], ...]) -> None:
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("FrozenMapping does not support mutation")

    def __delattr__(self, name: str) -> None:
        raise TypeError("FrozenMapping does not support mutation")

    def __getitem__(self, key: str) -> object:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and _json_values_equal(self, other)

    def __repr__(self) -> str:
        content = ", ".join(f"{key!r}: {value!r}" for key, value in self._items)
        return f"FrozenMapping({{{content}}})"


def freeze_json(value: object, field_name: str) -> object:
    """Validate and recursively snapshot JSON-compatible data."""

    return _freeze_json(value, field_name, set())


def _freeze_json(
    value: object,
    field_name: str,
    active_containers: set[int],
) -> object:
    """Freeze one value while rejecting cycles in the current traversal path."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not _is_finite_number(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{field_name} cannot contain cyclic JSON content")
        active_containers.add(container_id)
        try:
            items: list[tuple[str, object]] = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{field_name} mapping keys must be strings")
                items.append(
                    (
                        key,
                        _freeze_json(item, field_name, active_containers),
                    )
                )
            return FrozenMapping(tuple(items))
        finally:
            active_containers.remove(container_id)
    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{field_name} cannot contain cyclic JSON content")
        active_containers.add(container_id)
        try:
            return tuple(
                _freeze_json(item, field_name, active_containers) for item in value
            )
        finally:
            active_containers.remove(container_id)
    raise TypeError(
        f"{field_name} must contain only canonical JSON-compatible values"
    )


def freeze_mapping(value: object, field_name: str) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = freeze_json(value, field_name)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError(f"{field_name} must be a mapping")
    return frozen


def freeze_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence")
    frozen = freeze_json(value, field_name)
    if not isinstance(frozen, tuple):
        raise TypeError(f"{field_name} must be a sequence")
    return frozen


def freeze_mapping_sequence(
    value: object,
    field_name: str,
) -> tuple[FrozenMapping, ...]:
    frozen = freeze_sequence(value, field_name)
    if not all(isinstance(item, FrozenMapping) for item in frozen):
        raise TypeError(f"{field_name} must contain only mappings")
    return frozen


def freeze_string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    frozen = freeze_sequence(value, field_name)
    for item in frozen:
        _require_nonempty(item, field_name)
    return frozen


def _require_optional_nonempty(value: object, field_name: str) -> None:
    if value is not None:
        _require_nonempty(value, field_name)


def _require_optional_aware_datetime(value: object, field_name: str) -> None:
    if value is not None:
        _require_aware_datetime(value, field_name)


def _require_optional_finite_number(value: object, field_name: str) -> None:
    if value is not None and not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite number when provided")


def _require_optional_positive_number(value: object, field_name: str) -> None:
    _require_optional_finite_number(value, field_name)
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")


def _require_nonnegative_run(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _validate_component_reviews(
    component_reviews: tuple[FrozenMapping, ...],
    inside_80_ratio: float,
) -> tuple[str, ...]:
    if len(component_reviews) > 4:
        raise ValueError("component_reviews cannot exceed four principal paths")
    contradicted_paths: list[str] = []
    evidence_paths: set[str] = set()
    for index, review in enumerate(component_reviews):
        if frozenset(review) != _COMPONENT_REVIEW_KEYS:
            raise ValueError(
                "component_reviews entries must contain exactly the canonical "
                f"keys at index {index}"
            )
        evidence_path = review["evidence_path"]
        _require_nonempty(
            evidence_path,
            f"component_reviews[{index}].evidence_path",
        )
        if evidence_path not in _PRIORITY_EVIDENCE_PATHS:
            raise ValueError(
                f"component_reviews[{index}] requires a canonical evidence path"
            )
        if evidence_path in evidence_paths:
            raise ValueError("component_reviews evidence paths must be unique")
        evidence_paths.add(evidence_path)

        component_ids = review["component_ids"]
        if not isinstance(component_ids, tuple) or not component_ids:
            raise ValueError(
                f"component_reviews[{index}].component_ids must be non-empty"
            )
        for component_id in component_ids:
            _require_nonempty(
                component_id,
                f"component_reviews[{index}].component_ids",
            )
            if not _valid_component_id(component_id):
                raise ValueError(
                    f"component_reviews[{index}].component_ids contains an "
                    "unsupported official component"
                )
        if len(set(component_ids)) != len(component_ids):
            raise ValueError(
                f"component_reviews[{index}].component_ids must be unique"
            )

        contradicted = review["contradicted"]
        if not isinstance(contradicted, bool):
            raise ValueError(
                f"component_reviews[{index}].contradicted must be boolean"
            )
        inside_80_count = review["inside_80_count"]
        component_count = review["component_count"]
        for field_name, value in (
            ("inside_80_count", inside_80_count),
            ("component_count", component_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"component_reviews[{index}].{field_name} must be an integer"
                )
        if component_count <= 0 or component_count != len(component_ids):
            raise ValueError(
                f"component_reviews[{index}].component_count must match component_ids"
            )
        if not _valid_component_review_shape(
            evidence_path,
            frozenset(component_ids),
        ):
            raise ValueError(
                f"component_reviews[{index}] has an invalid component shape"
            )
        if not 0 <= inside_80_count <= component_count:
            raise ValueError(
                f"component_reviews[{index}].inside_80_count is out of range"
            )

        notes = review["notes"]
        if not isinstance(notes, tuple):
            raise ValueError(
                f"component_reviews[{index}].notes must be a sequence"
            )
        for note in notes:
            _require_nonempty(note, f"component_reviews[{index}].notes")
        if contradicted:
            contradicted_paths.append(evidence_path)
    if component_reviews and not _inside_ratio_is_representable(
        component_reviews,
        inside_80_ratio,
    ):
        raise ValueError(
            "inside_80_ratio is inconsistent with unique component statuses"
        )
    return tuple(contradicted_paths)


_OFFICIAL_COMPONENT_IDS = frozenset(
    {
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
    }
)


def _valid_component_id(component_id: str) -> bool:
    if component_id in _OFFICIAL_COMPONENT_IDS:
        return True
    suffix = "_exposed"
    for team in ("away", "home"):
        prefix = f"{team}_bullpen_segment_"
        if component_id.startswith(prefix) and component_id.endswith(suffix):
            segment_id = component_id[len(prefix) : -len(suffix)]
            return bool(segment_id.strip())
    return False


def _valid_component_review_shape(
    evidence_path: str,
    component_ids: frozenset[str],
) -> bool:
    if evidence_path == PriorityEvidencePath.STARTER_SUPPRESSION.value:
        return any(
            component_ids == {f"{side}_pitcher_chain_runs_allowed"}
            for side in ("away", "home")
        )
    if evidence_path == PriorityEvidencePath.STARTER_COLLAPSE.value:
        return any(
            component_ids
            == {
                f"{side}_pitcher_chain_runs_allowed",
                f"{side}_pitcher_chain_workload_outs",
            }
            for side in ("away", "home")
        )
    if evidence_path in _HORIZON_SCORING_EVIDENCE_PATHS:
        return any(
            component_ids
            == {
                f"{side}_{horizon}_runs",
                f"{side}_{horizon}_traffic",
                f"{side}_{horizon}_extra_base_hits",
            }
            for side in ("away", "home")
            for horizon in ("f5", "full")
        )
    if evidence_path == PriorityEvidencePath.LATE_BULLPEN_EXPOSURE.value:
        if len(component_ids) != 2:
            return False
        for side in ("away", "home"):
            workload_id = f"{side}_pitcher_chain_workload_outs"
            if workload_id not in component_ids:
                continue
            exposure_id = next(iter(component_ids.difference((workload_id,))))
            prefix = f"{side}_bullpen_segment_"
            suffix = "_exposed"
            if exposure_id.startswith(prefix) and exposure_id.endswith(suffix):
                return bool(exposure_id[len(prefix) : -len(suffix)].strip())
        return False
    if evidence_path == PriorityEvidencePath.BULLPEN_PROTECTION.value:
        return any(
            component_ids
            == {
                f"{side}_bullpen_runs_allowed",
                f"{side}_protected_bullpen_close_entry",
            }
            for side in ("away", "home")
        )
    return False


def _inside_ratio_is_representable(
    component_reviews: tuple[FrozenMapping, ...],
    inside_80_ratio: float,
) -> bool:
    component_ids = tuple(
        sorted(
            {
                component_id
                for review in component_reviews
                for component_id in review["component_ids"]
            }
        )
    )
    target_float = inside_80_ratio * len(component_ids)
    target_count = round(target_float)
    if not math.isclose(
        target_float,
        target_count,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    component_indexes = {
        component_id: index for index, component_id in enumerate(component_ids)
    }
    constraints = tuple(
        (
            tuple(
                component_indexes[component_id]
                for component_id in review["component_ids"]
            ),
            review["inside_80_count"],
        )
        for review in component_reviews
    )
    for mask in range(1 << len(component_ids)):
        if mask.bit_count() != target_count:
            continue
        if all(
            sum((mask >> index) & 1 for index in indexes) == inside_count
            for indexes, inside_count in constraints
        ):
            return True
    return False


def _contains_mapping_key(value: object, prohibited_key: str) -> bool:
    if isinstance(value, Mapping):
        return prohibited_key in value or any(
            _contains_mapping_key(item, prohibited_key) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_mapping_key(item, prohibited_key) for item in value)
    return False


@dataclass(frozen=True)
class RecordEnvelope:
    record_id: str
    record_kind: str
    idempotency_key: str
    payload_hash: str
    appended_at: datetime

    def __post_init__(self) -> None:
        _require_generated_id(self.record_id, "record_id")
        _require_nonempty(self.record_kind, "record_kind")
        _require_nonempty(self.idempotency_key, "idempotency_key")
        _require_nonempty(self.payload_hash, "payload_hash")
        _require_aware_datetime(self.appended_at, "appended_at")


@dataclass(frozen=True)
class CaptureCommand:
    game_pk: int
    captured_at: datetime
    first_pitch: datetime
    lineup_state: str
    out_of_sample_eligible: bool
    normalized_input_hash: str
    payload: Mapping[str, object]
    supersedes_capture_id: str | None = None

    def __post_init__(self) -> None:
        _require_game_pk(self.game_pk)
        _require_aware_datetime(self.captured_at, "captured_at")
        _require_aware_datetime(self.first_pitch, "first_pitch")
        _require_nonempty(self.lineup_state, "lineup_state")
        if not isinstance(self.out_of_sample_eligible, bool):
            raise ValueError("out_of_sample_eligible must be a boolean")
        _require_nonempty(self.normalized_input_hash, "normalized_input_hash")
        object.__setattr__(self, "payload", freeze_mapping(self.payload, "payload"))
        if self.supersedes_capture_id is not None:
            _require_nonempty(self.supersedes_capture_id, "supersedes_capture_id")


@dataclass(frozen=True)
class CaptureRecord:
    envelope: RecordEnvelope
    capture_id: str
    game_pk: int
    captured_at: datetime
    first_pitch: datetime
    lineup_state: str
    out_of_sample_eligible: bool
    normalized_input_hash: str
    payload: Mapping[str, object]
    supersedes_capture_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "capture":
            raise ValueError("capture envelope record_kind must be capture")
        command = CaptureCommand(
            game_pk=self.game_pk,
            captured_at=self.captured_at,
            first_pitch=self.first_pitch,
            lineup_state=self.lineup_state,
            out_of_sample_eligible=self.out_of_sample_eligible,
            normalized_input_hash=self.normalized_input_hash,
            payload=self.payload,
            supersedes_capture_id=self.supersedes_capture_id,
        )
        _require_generated_id(self.capture_id, "capture_id")
        object.__setattr__(self, "payload", command.payload)


@dataclass(frozen=True)
class ForecastCommand:
    forecast_run_id: str
    model_version: str
    simulation_config_hash: str
    normalized_input_hash: str
    outputs: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_nonempty(self.forecast_run_id, "forecast_run_id")
        _require_nonempty(self.model_version, "model_version")
        _require_nonempty(self.simulation_config_hash, "simulation_config_hash")
        _require_nonempty(self.normalized_input_hash, "normalized_input_hash")
        outputs = freeze_mapping(self.outputs, "outputs")
        if _contains_mapping_key(outputs, "capture_id"):
            raise ValueError("forecast outputs cannot contain capture_id")
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True)
class ForecastRecord:
    envelope: RecordEnvelope
    forecast_run_id: str
    model_version: str
    simulation_config_hash: str
    normalized_input_hash: str
    outputs: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "forecast":
            raise ValueError("forecast envelope record_kind must be forecast")
        command = ForecastCommand(
            forecast_run_id=self.forecast_run_id,
            model_version=self.model_version,
            simulation_config_hash=self.simulation_config_hash,
            normalized_input_hash=self.normalized_input_hash,
            outputs=self.outputs,
        )
        object.__setattr__(self, "outputs", command.outputs)


@dataclass(frozen=True)
class CaptureForecastLink:
    envelope: RecordEnvelope
    capture_id: str
    forecast_run_id: str
    shadow_comparison: Mapping[str, object] | None = None
    supersedes_link_record_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "capture_forecast_link":
            raise ValueError(
                "capture-forecast link envelope record_kind must be "
                "capture_forecast_link"
            )
        _require_nonempty(self.capture_id, "capture_id")
        _require_nonempty(self.forecast_run_id, "forecast_run_id")
        if self.shadow_comparison is not None:
            object.__setattr__(
                self,
                "shadow_comparison",
                freeze_mapping(
                    self.shadow_comparison,
                    "shadow_comparison",
                ),
            )
        if self.supersedes_link_record_id is not None:
            _require_generated_id(
                self.supersedes_link_record_id,
                "supersedes_link_record_id",
            )
            if self.shadow_comparison is None:
                raise ValueError(
                    "a superseding link requires shadow_comparison"
                )
            if self.supersedes_link_record_id == self.envelope.record_id:
                raise ValueError("capture-forecast link cannot supersede itself")


def linked_shadow_comparison(
    link: CaptureForecastLink,
    forecast: ForecastRecord,
) -> object:
    """Resolve the immutable comparison bound by one exact relationship."""

    if not isinstance(link, CaptureForecastLink):
        raise TypeError("link must be CaptureForecastLink")
    if not isinstance(forecast, ForecastRecord):
        raise TypeError("forecast must be ForecastRecord")
    if link.forecast_run_id != forecast.forecast_run_id:
        raise ValueError("link and forecast_run_id must match")
    if link.shadow_comparison is not None:
        return link.shadow_comparison
    return forecast.outputs.get("shadow_comparison")


@dataclass(frozen=True)
class MarketObservationCommand:
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    observation_kind: str
    market_key: str
    market_type: str
    settlement_format: str
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None
    observed_at: datetime
    available_at: datetime | None
    supersedes_record_id: str | None = None

    def __post_init__(self) -> None:
        _require_game_pk(self.game_pk)
        _require_nonempty(self.source_name, "source_name")
        _require_nonempty(self.source_record, "source_record")
        _require_nonempty(self.source_revision, "source_revision")
        if self.observation_kind not in {"entry", "close"}:
            raise ValueError("observation_kind must be entry or close")
        _require_nonempty(self.market_key, "market_key")
        _require_nonempty(self.market_type, "market_type")
        _require_nonempty(self.settlement_format, "settlement_format")
        _require_nonempty(self.selected_side, "selected_side")
        _require_optional_finite_number(self.line, "line")
        _require_optional_positive_number(
            self.selected_decimal_price,
            "selected_decimal_price",
        )
        _require_optional_positive_number(
            self.opposing_decimal_price,
            "opposing_decimal_price",
        )
        _require_aware_datetime(self.observed_at, "observed_at")
        _require_optional_aware_datetime(self.available_at, "available_at")
        _require_optional_nonempty(
            self.supersedes_record_id,
            "supersedes_record_id",
        )


@dataclass(frozen=True)
class MarketObservationRecord:
    envelope: RecordEnvelope
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    observation_kind: str
    market_key: str
    market_type: str
    settlement_format: str
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None
    observed_at: datetime
    available_at: datetime | None
    supersedes_record_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "market_observation":
            raise ValueError(
                "market observation envelope record_kind must be market_observation"
            )
        MarketObservationCommand(
            game_pk=self.game_pk,
            source_name=self.source_name,
            source_record=self.source_record,
            source_revision=self.source_revision,
            observation_kind=self.observation_kind,
            market_key=self.market_key,
            market_type=self.market_type,
            settlement_format=self.settlement_format,
            selected_side=self.selected_side,
            line=self.line,
            selected_decimal_price=self.selected_decimal_price,
            opposing_decimal_price=self.opposing_decimal_price,
            observed_at=self.observed_at,
            available_at=self.available_at,
            supersedes_record_id=self.supersedes_record_id,
        )


@dataclass(frozen=True)
class OfficialResultCommand:
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    finalized_at: datetime
    away_f5_runs: int
    home_f5_runs: int
    away_full_runs: int
    home_full_runs: int
    component_values: Mapping[str, float]
    documented_events: tuple[Mapping[str, object], ...]
    supersedes_record_id: str | None = None

    def __post_init__(self) -> None:
        _require_game_pk(self.game_pk)
        _require_nonempty(self.source_name, "source_name")
        _require_nonempty(self.source_record, "source_record")
        _require_nonempty(self.source_revision, "source_revision")
        _require_aware_datetime(self.finalized_at, "finalized_at")
        _require_nonnegative_run(self.away_f5_runs, "away_f5_runs")
        _require_nonnegative_run(self.home_f5_runs, "home_f5_runs")
        _require_nonnegative_run(self.away_full_runs, "away_full_runs")
        _require_nonnegative_run(self.home_full_runs, "home_full_runs")
        for side in ("away", "home"):
            if getattr(self, f"{side}_full_runs") < getattr(
                self,
                f"{side}_f5_runs",
            ):
                raise ValueError(
                    f"{side}_full_runs cannot be lower than {side}_f5_runs"
                )
        if self.away_full_runs == self.home_full_runs:
            raise ValueError("official full-game result must identify a winner")
        component_values = freeze_mapping(
            self.component_values,
            "component_values",
        )
        for component_id, component_value in component_values.items():
            if not _valid_component_id(component_id):
                raise ValueError(f"unsupported component_values key: {component_id}")
            if not _is_finite_number(component_value):
                raise ValueError(
                    f"component_values[{component_id}] must be a finite number"
                )
        for score_field in (
            "away_f5_runs",
            "home_f5_runs",
            "away_full_runs",
            "home_full_runs",
        ):
            if (
                score_field in component_values
                and component_values[score_field] != getattr(self, score_field)
            ):
                raise ValueError(
                    f"component_values[{score_field}] must match {score_field}"
                )
        for pitching_side, opponent_side in (("away", "home"), ("home", "away")):
            chain_key = f"{pitching_side}_pitcher_chain_runs_allowed"
            bullpen_key = f"{pitching_side}_bullpen_runs_allowed"
            present_keys = tuple(
                key for key in (chain_key, bullpen_key) if key in component_values
            )
            opponent_runs = float(getattr(self, f"{opponent_side}_full_runs"))
            for key in present_keys:
                value = float(component_values[key])
                if value < 0.0 or not value.is_integer():
                    raise ValueError(
                        "pitcher-chain and bullpen runs allowed must be "
                        "nonnegative whole-run counts"
                    )
                if value > opponent_runs:
                    raise ValueError(
                        "a pitcher-chain or bullpen runs allowed component "
                        "cannot exceed the opponent full-game score"
                    )
            if len(present_keys) == 2:
                attributed_runs = math.fsum(
                    float(component_values[key])
                    for key in (chain_key, bullpen_key)
                )
                if attributed_runs != opponent_runs:
                    raise ValueError(
                        "pitcher-chain and bullpen runs allowed must sum to the "
                        "opponent full-game score"
                    )
        object.__setattr__(self, "component_values", component_values)
        object.__setattr__(
            self,
            "documented_events",
            freeze_mapping_sequence(self.documented_events, "documented_events"),
        )
        _require_optional_nonempty(
            self.supersedes_record_id,
            "supersedes_record_id",
        )


@dataclass(frozen=True)
class OfficialResultRecord:
    envelope: RecordEnvelope
    game_pk: int
    source_name: str
    source_record: str
    source_revision: str
    finalized_at: datetime
    away_f5_runs: int
    home_f5_runs: int
    away_full_runs: int
    home_full_runs: int
    component_values: Mapping[str, float]
    documented_events: tuple[Mapping[str, object], ...]
    supersedes_record_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "official_result":
            raise ValueError(
                "official result envelope record_kind must be official_result"
            )
        command = OfficialResultCommand(
            game_pk=self.game_pk,
            source_name=self.source_name,
            source_record=self.source_record,
            source_revision=self.source_revision,
            finalized_at=self.finalized_at,
            away_f5_runs=self.away_f5_runs,
            home_f5_runs=self.home_f5_runs,
            away_full_runs=self.away_full_runs,
            home_full_runs=self.home_full_runs,
            component_values=self.component_values,
            documented_events=self.documented_events,
            supersedes_record_id=self.supersedes_record_id,
        )
        object.__setattr__(self, "component_values", command.component_values)
        object.__setattr__(self, "documented_events", command.documented_events)


@dataclass(frozen=True)
class ReviewCommand:
    capture_forecast_link_record_id: str
    capture_id: str
    forecast_run_id: str
    official_result_record_id: str
    market_key: str
    market_result: str
    classification: str | None
    inside_80_ratio: float
    contradicted_paths: tuple[str, ...]
    component_reviews: tuple[Mapping[str, object], ...]
    qualifying_exogenous_source_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(
            self.capture_forecast_link_record_id,
            "capture_forecast_link_record_id",
        )
        _require_nonempty(self.capture_id, "capture_id")
        _require_nonempty(self.forecast_run_id, "forecast_run_id")
        _require_nonempty(
            self.official_result_record_id,
            "official_result_record_id",
        )
        _require_nonempty(self.market_key, "market_key")
        if self.market_result not in _MARKET_RESULTS:
            raise ValueError("market_result must be win, loss, or push")
        if self.classification is not None and (
            not isinstance(self.classification, str)
            or self.classification not in _REVIEW_CLASSIFICATIONS
        ):
            raise ValueError("classification must be A, B, C, D, or None")
        if not _is_finite_number(self.inside_80_ratio):
            raise ValueError("inside_80_ratio must be a finite number")
        if not 0 <= self.inside_80_ratio <= 1:
            raise ValueError("inside_80_ratio must be within [0, 1]")
        contradicted_paths = freeze_string_sequence(
            self.contradicted_paths,
            "contradicted_paths",
        )
        if len(set(contradicted_paths)) != len(contradicted_paths):
            raise ValueError("contradicted_paths must be unique")
        component_reviews = freeze_mapping_sequence(
            self.component_reviews,
            "component_reviews",
        )
        if self.classification is not None and not component_reviews:
            raise ValueError(
                "classified reviews require non-empty component_reviews"
            )
        if self.classification is not None and not 2 <= len(component_reviews) <= 4:
            raise ValueError(
                "classified reviews require two to four principal paths"
            )
        component_contradictions = _validate_component_reviews(
            component_reviews,
            self.inside_80_ratio,
        )
        if contradicted_paths != component_contradictions:
            raise ValueError(
                "contradicted_paths must match contradicted component_reviews"
            )
        object.__setattr__(self, "contradicted_paths", contradicted_paths)
        object.__setattr__(self, "component_reviews", component_reviews)
        source_ids = freeze_string_sequence(
            self.qualifying_exogenous_source_observation_ids,
            "qualifying_exogenous_source_observation_ids",
        )
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(
                "qualifying exogenous source IDs must be unique"
            )
        source_ids = tuple(sorted(source_ids))
        if self.classification == "D" and not source_ids:
            raise ValueError(
                "D requires qualifying exogenous source observation IDs"
            )
        if self.classification != "D" and source_ids:
            raise ValueError(
                "non-D classifications require empty qualifying source IDs"
            )
        has_contradictions = bool(contradicted_paths)
        if self.classification == "A" and (
            self.market_result != "win"
            or self.inside_80_ratio < 0.75
            or has_contradictions
        ):
            raise ValueError(
                "A requires a win, inside_80_ratio >= 0.75, and no contradictions"
            )
        if self.classification == "B" and (
            self.inside_80_ratio < 0.60
            or has_contradictions
            or (
                self.market_result == "win"
                and self.inside_80_ratio >= 0.75
            )
        ):
            raise ValueError(
                "B requires no contradictions and must not qualify for A or C"
            )
        if (
            self.classification == "C"
            and self.inside_80_ratio >= 0.60
            and not has_contradictions
        ):
            raise ValueError(
                "C requires a contradiction or inside_80_ratio < 0.60"
            )
        object.__setattr__(
            self,
            "qualifying_exogenous_source_observation_ids",
            source_ids,
        )


@dataclass(frozen=True)
class ReviewRecord:
    envelope: RecordEnvelope
    review_id: str
    capture_forecast_link_record_id: str
    capture_id: str
    forecast_run_id: str
    official_result_record_id: str
    market_key: str
    market_result: str
    classification: str | None
    inside_80_ratio: float
    contradicted_paths: tuple[str, ...]
    component_reviews: tuple[Mapping[str, object], ...]
    qualifying_exogenous_source_observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, RecordEnvelope):
            raise ValueError("envelope must be RecordEnvelope")
        if self.envelope.record_kind != "review":
            raise ValueError("review envelope record_kind must be review")
        _require_generated_id(self.review_id, "review_id")
        command = ReviewCommand(
            capture_forecast_link_record_id=(
                self.capture_forecast_link_record_id
            ),
            capture_id=self.capture_id,
            forecast_run_id=self.forecast_run_id,
            official_result_record_id=self.official_result_record_id,
            market_key=self.market_key,
            market_result=self.market_result,
            classification=self.classification,
            inside_80_ratio=self.inside_80_ratio,
            contradicted_paths=self.contradicted_paths,
            component_reviews=self.component_reviews,
            qualifying_exogenous_source_observation_ids=(
                self.qualifying_exogenous_source_observation_ids
            ),
        )
        object.__setattr__(self, "contradicted_paths", command.contradicted_paths)
        object.__setattr__(self, "component_reviews", command.component_reviews)
        object.__setattr__(
            self,
            "qualifying_exogenous_source_observation_ids",
            command.qualifying_exogenous_source_observation_ids,
        )


__all__ = (
    "CaptureCommand",
    "CaptureForecastLink",
    "CaptureRecord",
    "ForecastCommand",
    "ForecastRecord",
    "MarketObservationCommand",
    "MarketObservationRecord",
    "OfficialResultCommand",
    "OfficialResultRecord",
    "RecordEnvelope",
    "ReviewCommand",
    "ReviewRecord",
    "linked_shadow_comparison",
)
