"""Bullpen role forecasts and deterministic inning-segment selection."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from .config import SimulationConfig
from .inputs import BullpenInput, BullpenRole, BullpenSegmentInput


_BULLPEN_ROLE_ORDER = (
    BullpenRole.BRIDGE,
    BullpenRole.MIDDLE,
    BullpenRole.HIGH_LEVERAGE,
    BullpenRole.CLOSER,
)
_UNSCHEDULED_F5_SEGMENT_ID = "league_unscheduled_f5_relief"
_LIMITED_NOTES = ("limited_performance_or_use_risk",)
_UNSCHEDULED_F5_NOTES = ("unscheduled_f5_neutral_prior",)
_FLOAT_TOLERANCE = 1e-12

__all__ = (
    "BullpenSegmentForecast",
    "BullpenForecast",
    "forecast_bullpen",
    "select_bullpen_segment",
)


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


def _require_boolean(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_notes(value: object) -> None:
    if not isinstance(value, tuple):
        raise ValueError("notes must be a tuple")
    for note in value:
        _require_nonempty_string(note, "notes")


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


@dataclass(frozen=True)
class _BullpenInvariantContext:
    league_ra9: float
    bullpen_ra9_bounds: tuple[float, float]
    limited_reliever_log_se_penalty: float
    unscheduled_f5_relief_log_standard_error: float

    def __post_init__(self) -> None:
        _require_finite_positive(self.league_ra9, "league_ra9")
        if (
            not isinstance(self.bullpen_ra9_bounds, tuple)
            or len(self.bullpen_ra9_bounds) != 2
        ):
            raise ValueError("bullpen_ra9_bounds must contain two values")
        lower, upper = self.bullpen_ra9_bounds
        _require_finite_positive(lower, "bullpen_ra9_bounds")
        _require_finite_positive(upper, "bullpen_ra9_bounds")
        if lower > upper or not lower <= 1.0 <= upper:
            raise ValueError("bullpen_ra9_bounds must contain 1.0 in order")
        _require_finite_nonnegative(
            self.limited_reliever_log_se_penalty,
            "limited_reliever_log_se_penalty",
        )
        _require_finite_nonnegative(
            self.unscheduled_f5_relief_log_standard_error,
            "unscheduled_f5_relief_log_standard_error",
        )


@dataclass(frozen=True)
class BullpenSegmentForecast:
    segment_id: str
    role: BullpenRole
    projected_ra9: float
    run_prevention_multiplier: float
    base_log_standard_error: float
    log_standard_error: float
    available: bool
    limited_performance_or_use_risk: bool
    synthetic: bool
    notes: tuple[str, ...] = ()
    _context: _BullpenInvariantContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        context = self._context
        if not isinstance(context, _BullpenInvariantContext):
            raise ValueError(
                "BullpenSegmentForecast must be constructed by forecast_bullpen"
            )

        _require_nonempty_string(self.segment_id, "segment_id")
        if not isinstance(self.role, BullpenRole):
            raise ValueError("role must be BullpenRole")
        _require_finite_positive(self.projected_ra9, "projected_ra9")
        _require_finite_positive(
            self.run_prevention_multiplier,
            "run_prevention_multiplier",
        )
        _require_finite_nonnegative(
            self.base_log_standard_error,
            "base_log_standard_error",
        )
        _require_finite_nonnegative(
            self.log_standard_error,
            "log_standard_error",
        )
        _require_boolean(self.available, "available")
        _require_boolean(
            self.limited_performance_or_use_risk,
            "limited_performance_or_use_risk",
        )
        _require_boolean(self.synthetic, "synthetic")
        _require_notes(self.notes)

        if self.limited_performance_or_use_risk and not self.available:
            raise ValueError(
                "limited_performance_or_use_risk requires available=True"
            )

        expected_multiplier = _clip(
            self.projected_ra9 / context.league_ra9,
            context.bullpen_ra9_bounds,
        )
        if not math.isclose(
            self.run_prevention_multiplier,
            expected_multiplier,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "run_prevention_multiplier is inconsistent with projected_ra9"
            )

        expected_sigma = math.sqrt(
            self.base_log_standard_error ** 2
            + (
                context.limited_reliever_log_se_penalty ** 2
                if self.limited_performance_or_use_risk
                else 0
            )
        )
        if not math.isclose(
            self.log_standard_error,
            expected_sigma,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "log_standard_error is inconsistent with base uncertainty and risk"
            )

        expected_notes = (
            _UNSCHEDULED_F5_NOTES
            if self.synthetic
            else _LIMITED_NOTES
            if self.limited_performance_or_use_risk
            else ()
        )
        if self.notes != expected_notes:
            raise ValueError("notes are inconsistent with segment state")

        if self.synthetic:
            if self.segment_id != _UNSCHEDULED_F5_SEGMENT_ID:
                raise ValueError(
                    "synthetic segment_id must be league_unscheduled_f5_relief"
                )
            if self.role is not BullpenRole.BRIDGE:
                raise ValueError("synthetic F5 relief must use the bridge role marker")
            if not math.isclose(
                self.projected_ra9,
                context.league_ra9,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise ValueError("synthetic F5 projected_ra9 must equal league RA9")
            if not math.isclose(
                self.base_log_standard_error,
                context.unscheduled_f5_relief_log_standard_error,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    "synthetic F5 uncertainty must equal the configured prior"
                )
            if not self.available or self.limited_performance_or_use_risk:
                raise ValueError("synthetic F5 relief must be available and not limited")
        elif self.segment_id == _UNSCHEDULED_F5_SEGMENT_ID:
            raise ValueError(
                "team segment_id cannot use league_unscheduled_f5_relief"
            )

    @property
    def multiplier(self) -> float:
        return self.run_prevention_multiplier


@dataclass(frozen=True)
class BullpenForecast:
    team: str
    segments: tuple[BullpenSegmentForecast, ...]
    by_role: Mapping[BullpenRole, BullpenSegmentForecast]
    high_leverage_available: bool
    lower_relief_risk: bool
    unscheduled_f5_relief: BullpenSegmentForecast
    _context: _BullpenInvariantContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        context = self._context
        if not isinstance(context, _BullpenInvariantContext):
            raise ValueError("BullpenForecast must be constructed by forecast_bullpen")
        _require_nonempty_string(self.team, "team")

        if not isinstance(self.segments, tuple):
            raise ValueError("segments must be a tuple")
        if len(self.segments) != len(_BULLPEN_ROLE_ORDER):
            raise ValueError("segments must contain all four bullpen roles")
        for segment in self.segments:
            if not isinstance(segment, BullpenSegmentForecast):
                raise ValueError(
                    "segments must contain BullpenSegmentForecast values"
                )
            if segment.synthetic:
                raise ValueError("team segments cannot be synthetic")
            if segment._context is not context:
                raise ValueError("segments must share the bullpen invariant context")

        roles = tuple(segment.role for segment in self.segments)
        if roles != _BULLPEN_ROLE_ORDER:
            raise ValueError("segments roles must be complete, unique, and ordered")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segments must have unique segment_id values")

        if not isinstance(self.by_role, Mapping):
            raise ValueError("by_role must be a mapping")
        copied_by_role = dict(self.by_role)
        if (
            len(copied_by_role) != len(_BULLPEN_ROLE_ORDER)
            or set(copied_by_role) != set(_BULLPEN_ROLE_ORDER)
            or any(not isinstance(role, BullpenRole) for role in copied_by_role)
        ):
            raise ValueError("by_role keys must exactly match all bullpen roles")
        for role, segment in zip(_BULLPEN_ROLE_ORDER, self.segments):
            if copied_by_role[role] is not segment:
                raise ValueError("by_role values must reference matching segments")

        _require_boolean(
            self.high_leverage_available,
            "high_leverage_available",
        )
        _require_boolean(self.lower_relief_risk, "lower_relief_risk")
        expected_high_leverage_available = copied_by_role[
            BullpenRole.HIGH_LEVERAGE
        ].available
        if self.high_leverage_available != expected_high_leverage_available:
            raise ValueError(
                "high_leverage_available is inconsistent with high leverage state"
            )
        expected_lower_relief_risk = _lower_relief_risk(copied_by_role)
        if self.lower_relief_risk != expected_lower_relief_risk:
            raise ValueError(
                "lower_relief_risk is inconsistent with bridge/middle states"
            )

        if not isinstance(
            self.unscheduled_f5_relief,
            BullpenSegmentForecast,
        ):
            raise ValueError(
                "unscheduled_f5_relief must be BullpenSegmentForecast"
            )
        if (
            not self.unscheduled_f5_relief.synthetic
            or self.unscheduled_f5_relief._context is not context
        ):
            raise ValueError(
                "unscheduled_f5_relief must be the shared synthetic F5 prior"
            )

        ordered_mapping = {
            role: copied_by_role[role] for role in _BULLPEN_ROLE_ORDER
        }
        object.__setattr__(self, "by_role", MappingProxyType(ordered_mapping))

    @property
    def available_segments(self) -> tuple[BullpenSegmentForecast, ...]:
        return tuple(segment for segment in self.segments if segment.available)


def _lower_relief_risk(
    by_role: Mapping[BullpenRole, BullpenSegmentForecast],
) -> bool:
    return any(
        not by_role[role].available
        or by_role[role].limited_performance_or_use_risk
        for role in (BullpenRole.BRIDGE, BullpenRole.MIDDLE)
    )


def _forecast_team_segment(
    segment: BullpenSegmentInput,
    context: _BullpenInvariantContext,
) -> BullpenSegmentForecast:
    projected_ra9 = segment.projected_ra9.value
    _require_finite_positive(
        projected_ra9,
        "segment.projected_ra9.value",
    )
    base_sigma = segment.log_standard_error.value
    _require_finite_nonnegative(
        base_sigma,
        "segment.log_standard_error.value",
    )
    available = segment.available.value
    limited = segment.limited_performance_or_use_risk.value
    _require_boolean(available, "segment.available.value")
    _require_boolean(
        limited,
        "segment.limited_performance_or_use_risk.value",
    )
    if limited and not available:
        raise ValueError(
            "limited_performance_or_use_risk requires available=True"
        )

    multiplier = _clip(
        projected_ra9 / context.league_ra9,
        context.bullpen_ra9_bounds,
    )
    sigma = math.sqrt(
        base_sigma ** 2
        + (
            context.limited_reliever_log_se_penalty ** 2
            if limited
            else 0
        )
    )
    return BullpenSegmentForecast(
        segment_id=segment.segment_id,
        role=segment.role,
        projected_ra9=projected_ra9,
        run_prevention_multiplier=multiplier,
        base_log_standard_error=base_sigma,
        log_standard_error=sigma,
        available=available,
        limited_performance_or_use_risk=limited,
        synthetic=False,
        notes=_LIMITED_NOTES if limited else (),
        _context=context,
    )


def _unscheduled_f5_prior(
    context: _BullpenInvariantContext,
) -> BullpenSegmentForecast:
    sigma = context.unscheduled_f5_relief_log_standard_error
    return BullpenSegmentForecast(
        segment_id=_UNSCHEDULED_F5_SEGMENT_ID,
        role=BullpenRole.BRIDGE,
        projected_ra9=context.league_ra9,
        run_prevention_multiplier=1.0,
        base_log_standard_error=sigma,
        log_standard_error=sigma,
        available=True,
        limited_performance_or_use_risk=False,
        synthetic=True,
        notes=_UNSCHEDULED_F5_NOTES,
        _context=context,
    )


def forecast_bullpen(
    bullpen: BullpenInput,
    config: SimulationConfig,
) -> BullpenForecast:
    """Forecast each normalized bullpen role without recomputing usage flags."""

    if not isinstance(bullpen, BullpenInput):
        raise TypeError("bullpen must be BullpenInput")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")

    for segment in bullpen.segments:
        if not isinstance(segment, BullpenSegmentInput):
            raise ValueError(
                "bullpen.segments must contain BullpenSegmentInput values"
            )
        if not isinstance(segment.role, BullpenRole):
            raise ValueError("bullpen segment role must be BullpenRole")

    segment_ids = tuple(segment.segment_id for segment in bullpen.segments)
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("duplicate bullpen segment_id")
    roles = tuple(segment.role for segment in bullpen.segments)
    if len(set(roles)) != len(roles):
        raise ValueError("duplicate bullpen role")
    if len(roles) != len(_BULLPEN_ROLE_ORDER) or set(roles) != set(
        _BULLPEN_ROLE_ORDER
    ):
        raise ValueError(
            "bullpen roles must contain bridge, middle, high_leverage, and closer"
        )

    context = _BullpenInvariantContext(
        league_ra9=config.league_runs_per_team_game,
        bullpen_ra9_bounds=config.bullpen_ra9_bounds,
        limited_reliever_log_se_penalty=(
            config.limited_reliever_log_se_penalty
        ),
        unscheduled_f5_relief_log_standard_error=(
            config.unscheduled_f5_relief_log_standard_error
        ),
    )
    inputs_by_role = {segment.role: segment for segment in bullpen.segments}
    segments = tuple(
        _forecast_team_segment(inputs_by_role[role], context)
        for role in _BULLPEN_ROLE_ORDER
    )
    by_role = {segment.role: segment for segment in segments}
    return BullpenForecast(
        team=bullpen.team,
        segments=segments,
        by_role=by_role,
        high_leverage_available=by_role[BullpenRole.HIGH_LEVERAGE].available,
        lower_relief_risk=_lower_relief_risk(by_role),
        unscheduled_f5_relief=_unscheduled_f5_prior(context),
        _context=context,
    )


def _first_available(
    forecast: BullpenForecast,
    roles: tuple[BullpenRole, ...],
) -> BullpenSegmentForecast:
    for role in roles:
        segment = forecast.by_role[role]
        if segment.available:
            return segment
    role_names = ", ".join(role.value for role in roles)
    raise ValueError(
        f"no available team bullpen segment for role chain: {role_names}"
    )


def select_bullpen_segment(
    forecast: BullpenForecast,
    inning: int,
    margin: int,
    bridge_needed: bool,
    verified_opener_bulk: bool,
) -> BullpenSegmentForecast:
    """Select one role deterministically without mutating the bullpen forecast."""

    if not isinstance(forecast, BullpenForecast):
        raise TypeError("forecast must be BullpenForecast")
    if isinstance(inning, bool) or not isinstance(inning, int):
        raise TypeError("inning must be an integer")
    if inning < 1:
        raise ValueError("inning must be at least one")
    if isinstance(margin, bool) or not isinstance(margin, int):
        raise TypeError("margin must be an integer")
    if not isinstance(bridge_needed, bool):
        raise TypeError("bridge_needed must be a boolean")
    if not isinstance(verified_opener_bulk, bool):
        raise TypeError("verified_opener_bulk must be a boolean")

    if inning < 6:
        if verified_opener_bulk:
            raise ValueError(
                "verified opener/bulk/PRIM remains in the initial pitcher chain"
            )
        return forecast.unscheduled_f5_relief

    close_game = abs(margin) <= 3
    if close_game and inning in (7, 8):
        roles = (
            BullpenRole.HIGH_LEVERAGE,
            BullpenRole.MIDDLE,
            BullpenRole.BRIDGE,
        )
    elif bridge_needed and inning in (6, 7):
        roles = (BullpenRole.BRIDGE, BullpenRole.MIDDLE)
    elif close_game and inning >= 9:
        roles = (
            BullpenRole.CLOSER,
            BullpenRole.HIGH_LEVERAGE,
            BullpenRole.MIDDLE,
        )
    elif bridge_needed:
        roles = (BullpenRole.BRIDGE, BullpenRole.MIDDLE)
    else:
        roles = (BullpenRole.MIDDLE, BullpenRole.BRIDGE)
    return _first_available(forecast, roles)
