"""Versioned behavior configuration for forecast simulations."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math


_BOUNDS_FIELDS = (
    "offense_bounds",
    "matchup_bounds",
    "starter_ra9_bounds",
    "bullpen_ra9_bounds",
)
_TRANSFER_FIELDS = (
    "pitch_limit_high_bucket_transfer",
    "pitch_limit_transfer_to_under_five",
)
_LOG_SE_PENALTY_FIELDS = (
    "sample_shortage_log_se_penalty",
    "unknown_bullpen_matchup_log_se_penalty",
    "handoff_uncertainty_log_se_penalty",
    "injury_return_log_se_penalty",
    "recent_short_exit_log_se_penalty",
    "limited_reliever_log_se_penalty",
    "unscheduled_f5_relief_log_standard_error",
)
_TUPLE_FIELDS = (*_BOUNDS_FIELDS, "pa_weights")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _require_positive_number(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")


def _require_bounds(value: object, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must be a tuple of exactly two numbers")
    lower, upper = value
    _require_positive_number(lower, field_name)
    _require_positive_number(upper, field_name)
    if not lower <= 1.0 <= upper:
        raise ValueError(f"{field_name} must satisfy lower <= 1.0 <= upper")


def _require_unit_interval(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")


def _require_nonnegative_number(value: object, field_name: str) -> None:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite nonnegative number")


@dataclass(frozen=True)
class SimulationConfig:
    version: str = "shadow-forecast-2026-07-14.1"
    league_runs_per_team_game: float = 4.50
    league_woba: float = 0.320
    gamma_poisson_shape: float = 1.60
    simulations: int = 20_000
    extra_inning_multiplier: float = 1.80
    max_innings: int = 30
    offense_bounds: tuple[float, float] = (0.65, 1.45)
    matchup_bounds: tuple[float, float] = (0.80, 1.20)
    starter_ra9_bounds: tuple[float, float] = (0.55, 1.75)
    bullpen_ra9_bounds: tuple[float, float] = (0.60, 1.90)
    pa_weights: tuple[float, ...] = (
        1.11,
        1.08,
        1.06,
        1.04,
        1.01,
        0.98,
        0.94,
        0.91,
        0.87,
    )
    pitch_limit_high_bucket_transfer: float = 1.0
    pitch_limit_transfer_to_under_five: float = 0.40
    sample_shortage_log_se_penalty: float = 0.12
    unknown_bullpen_matchup_log_se_penalty: float = 0.08
    handoff_uncertainty_log_se_penalty: float = 0.10
    injury_return_log_se_penalty: float = 0.12
    recent_short_exit_log_se_penalty: float = 0.10
    limited_reliever_log_se_penalty: float = 0.10
    unscheduled_f5_relief_log_standard_error: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a nonempty string")

        for field_name in (
            "league_runs_per_team_game",
            "league_woba",
            "gamma_poisson_shape",
            "extra_inning_multiplier",
        ):
            _require_positive_number(getattr(self, field_name), field_name)

        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or self.simulations <= 0
        ):
            raise ValueError("simulations must be a positive integer")
        if (
            isinstance(self.max_innings, bool)
            or not isinstance(self.max_innings, int)
            or self.max_innings < 10
        ):
            raise ValueError("max_innings must be an integer of at least 10")

        for field_name in _BOUNDS_FIELDS:
            _require_bounds(getattr(self, field_name), field_name)

        if not isinstance(self.pa_weights, tuple) or len(self.pa_weights) != 9:
            raise ValueError("pa_weights must be a tuple of exactly nine values")
        for weight in self.pa_weights:
            _require_positive_number(weight, "pa_weights")

        for field_name in _TRANSFER_FIELDS:
            _require_unit_interval(getattr(self, field_name), field_name)
        for field_name in _LOG_SE_PENALTY_FIELDS:
            _require_nonnegative_number(getattr(self, field_name), field_name)

    def to_mapping(self) -> dict[str, object]:
        """Return every behavior field in stable dataclass order.

        Tuple fields are copied to JSON-compatible lists. The config itself is
        never mutated.
        """

        mapping: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            mapping[field.name] = list(value) if isinstance(value, tuple) else value
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SimulationConfig":
        """Build from complete data or known-key partial operator overrides.

        Omitted fields retain dataclass defaults. Unknown keys are always
        rejected, and JSON list representations of tuple fields are normalized
        at this serialization boundary.
        """

        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must implement collections.abc.Mapping")

        known_keys = {field.name for field in fields(cls)}
        unknown_keys = set(mapping).difference(known_keys)
        if unknown_keys:
            rendered = ", ".join(sorted(repr(key) for key in unknown_keys))
            raise ValueError(f"unknown config key(s): {rendered}")

        values = dict(mapping)
        for field_name in _TUPLE_FIELDS:
            if field_name in values and isinstance(values[field_name], list):
                values[field_name] = tuple(values[field_name])
        return cls(**values)
