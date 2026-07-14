"""All-lineup pitch and velocity matchup forecasting."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
import weakref

from .config import SimulationConfig
from .inputs import LineupInput, LineupSlot, PitcherInput, PitchUsage


_SLOT_LABELS = frozenset({"플러스", "중립", "마이너스", "표본 부족"})
_EXPECTED_SIDES = frozenset({"R", "L"})
_FLOAT_TOLERANCE = 1e-12


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_finite_positive(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")


def _require_finite_nonnegative(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite nonnegative number")


def _require_unit_interval(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")


def _freeze_mapping(value: object, field_name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


def _require_mapping_keys(
    value: Mapping,
    expected_keys: tuple[str, ...],
    field_name: str,
) -> None:
    if set(value) != set(expected_keys) or len(value) != len(expected_keys):
        raise ValueError(
            f"{field_name} keys must exactly match evaluated_pitch_types"
        )


def _normalize_to_total(
    values: tuple[float, ...],
    target: float,
    field_name: str,
) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in values:
        _require_finite_positive(value, field_name)
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{field_name} total must be finite and positive")

    normalized = [target * value / total for value in values]
    largest_index = max(range(len(normalized)), key=normalized.__getitem__)
    other_total = math.fsum(
        value
        for index, value in enumerate(normalized)
        if index != largest_index
    )
    correction_value = target - other_total
    normalized[largest_index] = correction_value

    residual = target - math.fsum(normalized)
    if residual != 0.0:
        direction = math.inf if residual > 0.0 else -math.inf
        normalized[largest_index] = math.nextafter(correction_value, direction)
        if math.fsum(normalized) != target:
            candidates = tuple(
                candidate
                for candidate in (
                    math.nextafter(correction_value, -math.inf),
                    math.nextafter(correction_value, math.inf),
                )
                if 0.0 < candidate <= target
            )

            def candidate_residual(candidate: float) -> tuple[float, float]:
                candidate_values = normalized.copy()
                candidate_values[largest_index] = candidate
                return abs(target - math.fsum(candidate_values)), candidate

            if candidates:
                normalized[largest_index] = min(
                    candidates,
                    key=candidate_residual,
                )

    if any(
        not _is_finite_number(value) or value <= 0
        for value in normalized
    ):
        raise ValueError(f"normalized {field_name} must be finite and positive")
    if math.fsum(normalized) != target:
        raise ValueError(f"normalized {field_name} must sum exactly to {target}")
    return tuple(normalized)


@dataclass(frozen=True)
class _OffenseInvariantContext:
    league_woba: float
    normalized_pa_weights: tuple[float, ...]
    matchup_bounds: tuple[float, float]
    lineup_log_standard_error: float
    sample_shortage_log_se_penalty: float

    def __post_init__(self) -> None:
        _require_finite_positive(self.league_woba, "league_woba")
        if (
            not isinstance(self.normalized_pa_weights, tuple)
            or len(self.normalized_pa_weights) != 9
        ):
            raise ValueError("normalized_pa_weights must contain nine values")
        for weight in self.normalized_pa_weights:
            _require_finite_positive(weight, "normalized_pa_weights")
        if math.fsum(self.normalized_pa_weights) != 9.0:
            raise ValueError("normalized_pa_weights must sum exactly to nine")
        if (
            not isinstance(self.matchup_bounds, tuple)
            or len(self.matchup_bounds) != 2
        ):
            raise ValueError("matchup_bounds must contain lower and upper bounds")
        lower, upper = self.matchup_bounds
        _require_finite_positive(lower, "matchup_bounds")
        _require_finite_positive(upper, "matchup_bounds")
        if lower > upper:
            raise ValueError("matchup_bounds lower bound cannot exceed upper bound")
        _require_finite_nonnegative(
            self.lineup_log_standard_error,
            "lineup_log_standard_error",
        )
        _require_finite_nonnegative(
            self.sample_shortage_log_se_penalty,
            "sample_shortage_log_se_penalty",
        )


@dataclass(frozen=True)
class SlotMatchupForecast:
    order: int
    batter_id: str
    batter_name: str
    expected_side: str
    pitcher_id: str
    evaluated_pitch_types: tuple[str, ...]
    normalized_pitch_usages: Mapping[str, float]
    average_velocities_mph: Mapping[str, float]
    selected_band_widths: Mapping[str, float | None]
    velocity_deltas_mph: Mapping[str, float | None]
    plate_appearances: Mapping[str, int]
    adjusted_wobas: Mapping[str, float]
    labels: tuple[str, ...]
    sample_strengths: Mapping[str, str | None]
    exposure_weighted_adjusted_woba: float
    unknown_exposure: float
    _context: _OffenseInvariantContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not _is_factory_context(self._context):
            raise ValueError(
                "SlotMatchupForecast must be constructed by forecast_offense"
            )
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise ValueError("order must be an integer")
        if not 1 <= self.order <= 9:
            raise ValueError("order must be within 1 through 9")
        _require_nonempty_string(self.batter_id, "batter_id")
        _require_nonempty_string(self.batter_name, "batter_name")
        if self.expected_side not in _EXPECTED_SIDES:
            raise ValueError("expected_side must be R or L")
        _require_nonempty_string(self.pitcher_id, "pitcher_id")

        if not isinstance(self.evaluated_pitch_types, tuple):
            raise ValueError("evaluated_pitch_types must be a tuple")
        for pitch_type in self.evaluated_pitch_types:
            _require_nonempty_string(pitch_type, "evaluated_pitch_types")
        if len(set(self.evaluated_pitch_types)) != len(
            self.evaluated_pitch_types
        ):
            raise ValueError("evaluated_pitch_types must be unique")

        mapping_fields = (
            "normalized_pitch_usages",
            "average_velocities_mph",
            "selected_band_widths",
            "velocity_deltas_mph",
            "plate_appearances",
            "adjusted_wobas",
            "sample_strengths",
        )
        frozen_mappings: dict[str, Mapping] = {}
        for field_name in mapping_fields:
            frozen = _freeze_mapping(getattr(self, field_name), field_name)
            _require_mapping_keys(
                frozen,
                self.evaluated_pitch_types,
                field_name,
            )
            frozen_mappings[field_name] = frozen

        if not isinstance(self.labels, tuple):
            raise ValueError("labels must be a tuple")
        if self.evaluated_pitch_types:
            if len(self.labels) != len(self.evaluated_pitch_types):
                raise ValueError(
                    "labels must align one-for-one with evaluated_pitch_types"
                )
        elif self.labels != ("표본 부족",):
            raise ValueError(
                "an unevaluated slot must have only the 표본 부족 label"
            )
        for label in self.labels:
            if label not in _SLOT_LABELS:
                raise ValueError("labels must use the fixed pitch-velocity labels")

        usages = frozen_mappings["normalized_pitch_usages"]
        velocities = frozen_mappings["average_velocities_mph"]
        bands = frozen_mappings["selected_band_widths"]
        deltas = frozen_mappings["velocity_deltas_mph"]
        appearances = frozen_mappings["plate_appearances"]
        adjusted_wobas = frozen_mappings["adjusted_wobas"]
        strengths = frozen_mappings["sample_strengths"]

        for pitch_type in self.evaluated_pitch_types:
            usage = usages[pitch_type]
            if not _is_finite_number(usage) or not 0.0 < usage <= 1.0:
                raise ValueError(
                    "normalized_pitch_usages must contain values in (0, 1]"
                )
            _require_finite_positive(
                velocities[pitch_type],
                "average_velocities_mph",
            )
            band_width = bands[pitch_type]
            velocity_delta = deltas[pitch_type]
            if band_width is not None and band_width not in (1.5, 3.0):
                raise ValueError(
                    "selected_band_widths must contain only 1.5, 3.0, or None"
                )
            if velocity_delta is not None:
                _require_finite_nonnegative(
                    velocity_delta,
                    "velocity_deltas_mph",
                )
            if (band_width is None) != (velocity_delta is None):
                raise ValueError(
                    "selected band and velocity delta must both be present or missing"
                )
            if (
                band_width is not None
                and velocity_delta is not None
                and velocity_delta > band_width
            ):
                raise ValueError("velocity delta cannot exceed the selected band")

            plate_appearances = appearances[pitch_type]
            if (
                isinstance(plate_appearances, bool)
                or not isinstance(plate_appearances, int)
                or plate_appearances < 0
            ):
                raise ValueError(
                    "plate_appearances must contain nonnegative integers"
                )
            _require_finite_nonnegative(
                adjusted_wobas[pitch_type],
                "adjusted_wobas",
            )
            sample_strength = strengths[pitch_type]
            label = self.labels[
                self.evaluated_pitch_types.index(pitch_type)
            ]
            if band_width is None:
                if plate_appearances != 0 or sample_strength is not None:
                    raise ValueError(
                        "missing responses require zero PA and no sample strength"
                    )
                if label != "표본 부족":
                    raise ValueError(
                        "missing responses must use the 표본 부족 label"
                    )
                if not math.isclose(
                    adjusted_wobas[pitch_type],
                    self._context.league_woba,
                    rel_tol=0.0,
                    abs_tol=_FLOAT_TOLERANCE,
                ):
                    raise ValueError(
                        "missing response adjusted_woba must equal construction "
                        "league_woba"
                    )
            else:
                _require_nonempty_string(sample_strength, "sample_strengths")

        if self.evaluated_pitch_types and math.fsum(usages.values()) != 1.0:
            raise ValueError("normalized_pitch_usages must sum exactly to one")

        _require_finite_nonnegative(
            self.exposure_weighted_adjusted_woba,
            "exposure_weighted_adjusted_woba",
        )
        _require_unit_interval(self.unknown_exposure, "unknown_exposure")
        if self.evaluated_pitch_types:
            expected_woba = math.fsum(
                usages[pitch_type] * adjusted_wobas[pitch_type]
                for pitch_type in self.evaluated_pitch_types
            )
            if not math.isclose(
                self.exposure_weighted_adjusted_woba,
                expected_woba,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    "exposure_weighted_adjusted_woba is inconsistent with pitch values"
                )
            expected_unknown = math.fsum(
                usages[pitch_type]
                for pitch_type in self.evaluated_pitch_types
                if bands[pitch_type] is None
            )
            if not math.isclose(
                self.unknown_exposure,
                expected_unknown,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    "unknown_exposure is inconsistent with missing pitch usage"
                )
        else:
            if self.unknown_exposure != 1.0:
                raise ValueError(
                    "an unevaluated slot must have unknown_exposure 1.0"
                )
            if not math.isclose(
                self.exposure_weighted_adjusted_woba,
                self._context.league_woba,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    "an unevaluated slot point value must equal construction "
                    "league_woba"
                )

        for field_name, frozen in frozen_mappings.items():
            object.__setattr__(self, field_name, frozen)


def _aggregate_matchup_log_standard_error(
    matchup_sigmas: Mapping[str, float],
) -> float:
    return max(matchup_sigmas.values())


@dataclass(frozen=True)
class OffenseForecast:
    base_offense_multiplier: float
    matchup_multipliers: Mapping[str, float]
    projected_traffic_per_9: float
    projected_extra_base_hits_per_9: float
    log_standard_error: float
    slot_forecasts: tuple[SlotMatchupForecast, ...]
    matchup_log_standard_errors: Mapping[str, float]
    matchup_unknown_exposures: Mapping[str, float]
    unknown_bullpen_matchup_log_se_penalty: float
    _context: _OffenseInvariantContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not _is_factory_context(self._context):
            raise ValueError("OffenseForecast must be constructed by forecast_offense")
        _require_finite_positive(
            self.base_offense_multiplier,
            "base_offense_multiplier",
        )
        _require_finite_nonnegative(
            self.projected_traffic_per_9,
            "projected_traffic_per_9",
        )
        _require_finite_nonnegative(
            self.projected_extra_base_hits_per_9,
            "projected_extra_base_hits_per_9",
        )
        _require_finite_nonnegative(
            self.log_standard_error,
            "log_standard_error",
        )
        _require_finite_nonnegative(
            self.unknown_bullpen_matchup_log_se_penalty,
            "unknown_bullpen_matchup_log_se_penalty",
        )

        multipliers = _freeze_mapping(
            self.matchup_multipliers,
            "matchup_multipliers",
        )
        sigmas = _freeze_mapping(
            self.matchup_log_standard_errors,
            "matchup_log_standard_errors",
        )
        unknowns = _freeze_mapping(
            self.matchup_unknown_exposures,
            "matchup_unknown_exposures",
        )
        pitcher_ids = tuple(multipliers)
        if not pitcher_ids:
            raise ValueError("matchup_multipliers must be nonempty")
        for pitcher_id in pitcher_ids:
            _require_nonempty_string(pitcher_id, "matchup_multipliers keys")
            _require_finite_positive(
                multipliers[pitcher_id],
                "matchup_multipliers",
            )
        if set(sigmas) != set(pitcher_ids) or len(sigmas) != len(pitcher_ids):
            raise ValueError(
                "matchup_log_standard_errors keys must match matchup_multipliers"
            )
        if set(unknowns) != set(pitcher_ids) or len(unknowns) != len(pitcher_ids):
            raise ValueError(
                "matchup_unknown_exposures keys must match matchup_multipliers"
            )
        for pitcher_id in pitcher_ids:
            _require_finite_nonnegative(
                sigmas[pitcher_id],
                "matchup_log_standard_errors",
            )
            _require_unit_interval(
                unknowns[pitcher_id],
                "matchup_unknown_exposures",
            )
        expected_log_standard_error = _aggregate_matchup_log_standard_error(
            sigmas
        )
        if not math.isclose(
            self.log_standard_error,
            expected_log_standard_error,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "log_standard_error must equal the maximum matchup sigma"
            )

        if not isinstance(self.slot_forecasts, tuple):
            raise ValueError("slot_forecasts must be a tuple")
        for slot_forecast in self.slot_forecasts:
            if not isinstance(slot_forecast, SlotMatchupForecast):
                raise ValueError(
                    "slot_forecasts must contain SlotMatchupForecast values"
                )
            if slot_forecast._context != self._context:
                raise ValueError(
                    "slot_forecasts must share the offense invariant context"
                )
        slot_pairs = tuple(
            (slot.order, slot.pitcher_id) for slot in self.slot_forecasts
        )
        if len(set(slot_pairs)) != len(slot_pairs):
            raise ValueError("slot_forecasts must have unique order/pitcher pairs")
        expected_pairs = tuple(
            (order, pitcher_id)
            for order in range(1, 10)
            for pitcher_id in pitcher_ids
        )
        if slot_pairs != expected_pairs:
            raise ValueError(
                "slot_forecasts must contain nine ordered slots per pitcher"
            )

        signatures_by_order: dict[int, tuple[str, str, str]] = {}
        for pitcher_id in pitcher_ids:
            pitcher_slots = tuple(
                slot
                for slot in self.slot_forecasts
                if slot.pitcher_id == pitcher_id
            )
            batter_ids = tuple(slot.batter_id for slot in pitcher_slots)
            if len(set(batter_ids)) != len(batter_ids):
                raise ValueError(
                    "slot_forecasts must have unique batter IDs for each pitcher"
                )
            for slot in pitcher_slots:
                signature = (
                    slot.batter_id,
                    slot.batter_name,
                    slot.expected_side,
                )
                previous = signatures_by_order.setdefault(slot.order, signature)
                if previous != signature:
                    raise ValueError(
                        "lineup slot identity must agree across pitcher profiles"
                    )

        for pitcher_id in pitcher_ids:
            expected_multiplier, expected_sigma, expected_unknown = (
                _summarize_pitcher_slots(
                    self.slot_forecasts,
                    pitcher_id,
                    self._context,
                )
            )
            expected_values = (
                (
                    "matchup_multipliers",
                    multipliers[pitcher_id],
                    expected_multiplier,
                ),
                (
                    "matchup_unknown_exposures",
                    unknowns[pitcher_id],
                    expected_unknown,
                ),
                (
                    "matchup_log_standard_errors",
                    sigmas[pitcher_id],
                    expected_sigma,
                ),
            )
            for field_name, actual, expected in expected_values:
                if not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=_FLOAT_TOLERANCE,
                ):
                    raise ValueError(
                        f"{field_name} is inconsistent with slot diagnostics"
                    )

        object.__setattr__(self, "matchup_multipliers", multipliers)
        object.__setattr__(self, "matchup_log_standard_errors", sigmas)
        object.__setattr__(self, "matchup_unknown_exposures", unknowns)

    def multiplier_for(self, pitcher_or_segment_id: str) -> float:
        return self.matchup_multipliers.get(pitcher_or_segment_id, 1.0)

    def log_standard_error_for(self, pitcher_or_segment_id: str) -> float:
        explicit = self.matchup_log_standard_errors.get(pitcher_or_segment_id)
        if explicit is not None:
            return explicit
        return math.hypot(
            self.log_standard_error,
            self.unknown_bullpen_matchup_log_se_penalty,
        )

    def slot_forecasts_for(
        self,
        pitcher_id: str,
    ) -> tuple[SlotMatchupForecast, ...]:
        return tuple(
            slot for slot in self.slot_forecasts if slot.pitcher_id == pitcher_id
        )


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def _candidate_sort_key(response: object, average_velocity_mph: float) -> tuple:
    adjusted_woba = response.adjusted_woba
    adjusted_key = (
        0,
        adjusted_woba,
    ) if _is_finite_number(adjusted_woba) else (1, repr(adjusted_woba))
    return (
        abs(response.velocity_center_mph - average_velocity_mph),
        -response.plate_appearances,
        response.velocity_center_mph,
        adjusted_key,
        response.sample_strength,
        response.slot_label,
    )


def _select_response(
    slot: LineupSlot,
    pitcher: PitcherInput,
    pitch_usage: PitchUsage,
):
    matching = tuple(
        response
        for response in slot.responses
        if response.batter_id == slot.batter_id
        and response.pitcher_id == pitcher.pitcher_id
        and response.expected_side == slot.expected_side
        and response.pitch_type == pitch_usage.pitch_type
    )
    exact = tuple(
        response
        for response in matching
        if response.band_width_mph == 1.5
        and abs(
            response.velocity_center_mph
            - pitch_usage.average_velocity_mph
        )
        <= 1.5
    )
    candidates = exact
    if not candidates:
        candidates = tuple(
            response
            for response in matching
            if response.band_width_mph == 3.0
            and abs(
                response.velocity_center_mph
                - pitch_usage.average_velocity_mph
            )
            <= 3.0
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda response: _candidate_sort_key(
            response,
            pitch_usage.average_velocity_mph,
        ),
    )


def _forecast_slot(
    slot: LineupSlot,
    pitcher: PitcherInput,
    config: SimulationConfig,
    context: _OffenseInvariantContext,
) -> SlotMatchupForecast:
    matching_positive_mix = tuple(
        pitch_usage
        for pitch_usage in pitcher.pitch_mix
        if pitch_usage.opposing_side == slot.expected_side
        and pitch_usage.usage > 0.0
    )
    if not matching_positive_mix:
        return SlotMatchupForecast(
            order=slot.order,
            batter_id=slot.batter_id,
            batter_name=slot.batter_name,
            expected_side=slot.expected_side,
            pitcher_id=pitcher.pitcher_id,
            evaluated_pitch_types=(),
            normalized_pitch_usages={},
            average_velocities_mph={},
            selected_band_widths={},
            velocity_deltas_mph={},
            plate_appearances={},
            adjusted_wobas={},
            labels=("표본 부족",),
            sample_strengths={},
            exposure_weighted_adjusted_woba=config.league_woba,
            unknown_exposure=1.0,
            _context=context,
        )

    normalized_values = _normalize_to_total(
        tuple(pitch_usage.usage for pitch_usage in matching_positive_mix),
        1.0,
        "matching pitch usage",
    )
    pitch_types = tuple(
        pitch_usage.pitch_type for pitch_usage in matching_positive_mix
    )
    normalized_usages = dict(zip(pitch_types, normalized_values))
    velocities: dict[str, float] = {}
    bands: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    appearances: dict[str, int] = {}
    adjusted_wobas: dict[str, float] = {}
    labels: list[str] = []
    strengths: dict[str, str | None] = {}

    for pitch_usage in matching_positive_mix:
        pitch_type = pitch_usage.pitch_type
        velocities[pitch_type] = pitch_usage.average_velocity_mph
        selected = _select_response(slot, pitcher, pitch_usage)
        if selected is None:
            bands[pitch_type] = None
            deltas[pitch_type] = None
            appearances[pitch_type] = 0
            adjusted_wobas[pitch_type] = config.league_woba
            labels.append("표본 부족")
            strengths[pitch_type] = None
            continue

        _require_finite_nonnegative(
            selected.adjusted_woba,
            "selected response adjusted_woba",
        )
        velocity_delta = abs(
            selected.velocity_center_mph
            - pitch_usage.average_velocity_mph
        )
        bands[pitch_type] = selected.band_width_mph
        deltas[pitch_type] = velocity_delta
        appearances[pitch_type] = selected.plate_appearances
        adjusted_wobas[pitch_type] = selected.adjusted_woba
        labels.append(selected.slot_label)
        strengths[pitch_type] = selected.sample_strength

    exposure_woba = math.fsum(
        normalized_usages[pitch_type] * adjusted_wobas[pitch_type]
        for pitch_type in pitch_types
    )
    unknown_exposure = math.fsum(
        normalized_usages[pitch_type]
        for pitch_type in pitch_types
        if bands[pitch_type] is None
    )
    if math.isclose(unknown_exposure, 0.0, rel_tol=0.0, abs_tol=1e-15):
        unknown_exposure = 0.0
    elif math.isclose(unknown_exposure, 1.0, rel_tol=0.0, abs_tol=1e-15):
        unknown_exposure = 1.0

    return SlotMatchupForecast(
        order=slot.order,
        batter_id=slot.batter_id,
        batter_name=slot.batter_name,
        expected_side=slot.expected_side,
        pitcher_id=pitcher.pitcher_id,
        evaluated_pitch_types=pitch_types,
        normalized_pitch_usages=normalized_usages,
        average_velocities_mph=velocities,
        selected_band_widths=bands,
        velocity_deltas_mph=deltas,
        plate_appearances=appearances,
        adjusted_wobas=adjusted_wobas,
        labels=tuple(labels),
        sample_strengths=strengths,
        exposure_weighted_adjusted_woba=exposure_woba,
        unknown_exposure=unknown_exposure,
        _context=context,
    )


def _validate_profile(profile: PitcherInput) -> None:
    if not isinstance(profile.pitch_mix, tuple):
        raise ValueError("pitch_mix must be a tuple")
    seen_usage_keys: set[tuple[str, str]] = set()
    for pitch_usage in profile.pitch_mix:
        if not isinstance(pitch_usage, PitchUsage):
            raise ValueError("pitch_mix must contain PitchUsage values")
        usage_key = (pitch_usage.opposing_side, pitch_usage.pitch_type)
        if usage_key in seen_usage_keys:
            raise ValueError(
                "duplicate pitch usage for opposing_side and pitch_type"
            )
        seen_usage_keys.add(usage_key)
        if not _is_finite_number(pitch_usage.usage) or not (
            0.0 <= pitch_usage.usage <= 1.0
        ):
            raise ValueError("pitch usage must be finite and within [0, 1]")
        _require_finite_positive(
            pitch_usage.average_velocity_mph,
            "pitch average_velocity_mph",
        )


def _weighted_average(
    values_by_order: Mapping[int, float],
    normalized_pa_weights: tuple[float, ...],
) -> float:
    ordered_values = tuple(values_by_order[order] for order in range(1, 10))
    if all(value == ordered_values[0] for value in ordered_values[1:]):
        return ordered_values[0]
    return math.fsum(
        normalized_pa_weights[order - 1] * values_by_order[order]
        for order in range(1, 10)
    ) / 9.0


def _canonical_unknown_exposure(value: float) -> float:
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return 0.0
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return 1.0
    return value


def _summarize_pitcher_slots(
    slot_forecasts: tuple[SlotMatchupForecast, ...],
    pitcher_id: str,
    context: _OffenseInvariantContext,
) -> tuple[float, float, float]:
    slots_by_order = {
        slot.order: slot
        for slot in slot_forecasts
        if slot.pitcher_id == pitcher_id
    }
    if set(slots_by_order) != set(range(1, 10)):
        raise ValueError("slot diagnostics must contain batting orders 1 through 9")
    exposure_woba = _weighted_average(
        {
            order: slots_by_order[order].exposure_weighted_adjusted_woba
            for order in range(1, 10)
        },
        context.normalized_pa_weights,
    )
    unknown_exposure = _canonical_unknown_exposure(
        _weighted_average(
            {
                order: slots_by_order[order].unknown_exposure
                for order in range(1, 10)
            },
            context.normalized_pa_weights,
        )
    )
    matchup_multiplier = _clip(
        exposure_woba / context.league_woba,
        context.matchup_bounds,
    )
    matchup_sigma = math.hypot(
        context.lineup_log_standard_error,
        unknown_exposure * context.sample_shortage_log_se_penalty,
    )
    return matchup_multiplier, matchup_sigma, unknown_exposure


def _forecast_offense_impl(
    lineup: LineupInput,
    opposing_profiles: tuple[PitcherInput, ...],
    config: SimulationConfig,
    context_issuer: Callable[..., _OffenseInvariantContext],
) -> OffenseForecast:
    if not isinstance(lineup, LineupInput):
        raise TypeError("lineup must be LineupInput")
    if not isinstance(opposing_profiles, tuple):
        raise TypeError("opposing_profiles must be a tuple")
    if not opposing_profiles:
        raise ValueError("opposing_profiles must be a nonempty tuple")
    for profile in opposing_profiles:
        if not isinstance(profile, PitcherInput):
            raise TypeError("opposing_profiles must contain PitcherInput values")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")

    pitcher_ids = tuple(profile.pitcher_id for profile in opposing_profiles)
    if len(set(pitcher_ids)) != len(pitcher_ids):
        raise ValueError("opposing_profiles must have unique pitcher_id values")
    for profile in opposing_profiles:
        _validate_profile(profile)

    projected_runs = lineup.projected_runs_per_9.value
    if not _is_finite_number(projected_runs) or projected_runs <= 0:
        raise ValueError(
            "lineup.projected_runs_per_9.value must be finite and positive"
        )
    projected_traffic = lineup.projected_traffic_per_9.value
    _require_finite_nonnegative(
        projected_traffic,
        "lineup.projected_traffic_per_9.value",
    )
    projected_extra_base_hits = lineup.projected_extra_base_hits_per_9.value
    _require_finite_nonnegative(
        projected_extra_base_hits,
        "lineup.projected_extra_base_hits_per_9.value",
    )
    lineup_log_standard_error = lineup.log_standard_error.value
    _require_finite_nonnegative(
        lineup_log_standard_error,
        "lineup.log_standard_error.value",
    )

    normalized_pa_weights = _normalize_to_total(
        tuple(config.pa_weights),
        9.0,
        "pa_weights",
    )
    context = context_issuer(
        league_woba=config.league_woba,
        normalized_pa_weights=normalized_pa_weights,
        matchup_bounds=config.matchup_bounds,
        lineup_log_standard_error=lineup_log_standard_error,
        sample_shortage_log_se_penalty=(
            config.sample_shortage_log_se_penalty
        ),
    )
    ordered_slots = tuple(sorted(lineup.slots, key=lambda slot: slot.order))
    slot_forecasts = tuple(
        _forecast_slot(slot, profile, config, context)
        for slot in ordered_slots
        for profile in opposing_profiles
    )

    matchup_multipliers: dict[str, float] = {}
    matchup_sigmas: dict[str, float] = {}
    matchup_unknowns: dict[str, float] = {}
    for profile in opposing_profiles:
        pitcher_id = profile.pitcher_id
        multiplier, sigma, unknown_exposure = _summarize_pitcher_slots(
            slot_forecasts,
            pitcher_id,
            context,
        )
        matchup_unknowns[pitcher_id] = unknown_exposure
        matchup_multipliers[pitcher_id] = multiplier
        matchup_sigmas[pitcher_id] = sigma

    base_offense_multiplier = _clip(
        projected_runs / config.league_runs_per_team_game,
        config.offense_bounds,
    )
    return OffenseForecast(
        base_offense_multiplier=base_offense_multiplier,
        matchup_multipliers=matchup_multipliers,
        projected_traffic_per_9=projected_traffic,
        projected_extra_base_hits_per_9=projected_extra_base_hits,
        log_standard_error=_aggregate_matchup_log_standard_error(
            matchup_sigmas
        ),
        slot_forecasts=slot_forecasts,
        matchup_log_standard_errors=matchup_sigmas,
        matchup_unknown_exposures=matchup_unknowns,
        unknown_bullpen_matchup_log_se_penalty=(
            config.unknown_bullpen_matchup_log_se_penalty
        ),
        _context=context,
    )


def _build_forecast_offense():
    issued_contexts = weakref.WeakValueDictionary()

    def issue_context(
        *,
        league_woba: float,
        normalized_pa_weights: tuple[float, ...],
        matchup_bounds: tuple[float, float],
        lineup_log_standard_error: float,
        sample_shortage_log_se_penalty: float,
    ) -> _OffenseInvariantContext:
        context = _OffenseInvariantContext(
            league_woba=league_woba,
            normalized_pa_weights=normalized_pa_weights,
            matchup_bounds=matchup_bounds,
            lineup_log_standard_error=lineup_log_standard_error,
            sample_shortage_log_se_penalty=sample_shortage_log_se_penalty,
        )
        issued_contexts[id(context)] = context
        return context

    def is_factory_context(value: object) -> bool:
        return (
            isinstance(value, _OffenseInvariantContext)
            and issued_contexts.get(id(value)) is value
        )

    def forecast_offense(
        lineup: LineupInput,
        opposing_profiles: tuple[PitcherInput, ...],
        config: SimulationConfig,
    ) -> OffenseForecast:
        return _forecast_offense_impl(
            lineup,
            opposing_profiles,
            config,
            issue_context,
        )

    return forecast_offense, is_factory_context


# Keep context issuance inaccessible from module consumers.
forecast_offense, _is_factory_context = _build_forecast_offense()
del _build_forecast_offense
