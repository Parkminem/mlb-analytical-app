"""Immutable normalized inputs for pregame forecasting."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math
from typing import TypeVar


class LineupState(str, Enum):
    PROJECTED = "projected"
    CONFIRMED = "confirmed"


class PitcherRole(str, Enum):
    STARTER = "starter"
    OPENER = "opener"
    BULK = "bulk"
    PRIM = "prim"


class BullpenRole(str, Enum):
    BRIDGE = "bridge"
    MIDDLE = "middle"
    HIGH_LEVERAGE = "high_leverage"
    CLOSER = "closer"


class MarketType(str, Enum):
    F5_SIDE = "f5_side"
    F5_TOTAL = "f5_total"
    ML = "ml"
    PLUS_1_5 = "plus_1_5"
    FULL_TOTAL = "full_total"


class SettlementFormat(str, Enum):
    TWO_WAY = "two_way"
    THREE_WAY = "three_way"


_T = TypeVar("_T")
_EXPECTED_SIDES = frozenset({"R", "L"})
_BATTING_SIDES = frozenset({"R", "L", "S"})
_SLOT_LABELS = frozenset({"플러스", "중립", "마이너스", "표본 부족"})


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_aware_datetime(
    value: object,
    field_name: str,
    *,
    optional: bool = False,
) -> None:
    if value is None and optional:
        return
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


def _require_date(value: object, field_name: str) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a date")


def _require_finite_number(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field_name} must be a finite number")


def _require_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _require_boolean(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_side(
    value: object,
    allowed: frozenset[str],
    field_name: str,
) -> None:
    if not isinstance(value, str) or value not in allowed:
        choices = "/".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of {choices}")


def _require_instance(value: object, expected_type: type[_T], field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")


def _require_optional_instance(
    value: object,
    expected_type: type[_T],
    field_name: str,
) -> None:
    if value is not None:
        _require_instance(value, expected_type, field_name)


def _require_tuple_of(
    values: object,
    expected_type: type[_T],
    field_name: str,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    for value in values:
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{field_name} must contain only {expected_type.__name__} values"
            )


def _require_enum(value: object, enum_type: type[_T], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be {enum_type.__name__}")


@dataclass(frozen=True)
class SourceObservation:
    observation_id: str
    source_name: str
    source_record: str
    observed_at: datetime
    available_at: datetime | None
    fetched_at: datetime
    effective_date: date
    content_hash: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.observation_id, "observation_id")
        _require_nonempty_string(self.source_name, "source_name")
        _require_nonempty_string(self.source_record, "source_record")
        _require_aware_datetime(self.observed_at, "observed_at")
        _require_aware_datetime(self.available_at, "available_at", optional=True)
        _require_aware_datetime(self.fetched_at, "fetched_at")
        _require_date(self.effective_date, "effective_date")
        _require_nonempty_string(self.content_hash, "content_hash")


@dataclass(frozen=True)
class FeatureProvenance:
    observation_ids: tuple[str, ...]
    latest_available_at: datetime | None
    lookback_end: datetime | None

    def __post_init__(self) -> None:
        _require_tuple_of(self.observation_ids, str, "observation_ids")
        if not self.observation_ids:
            raise ValueError("observation_ids must contain at least one identifier")
        for observation_id in self.observation_ids:
            _require_nonempty_string(observation_id, "observation_ids")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("observation_ids must be unique")
        _require_aware_datetime(
            self.latest_available_at,
            "latest_available_at",
            optional=True,
        )
        _require_aware_datetime(self.lookback_end, "lookback_end", optional=True)


@dataclass(frozen=True)
class NumericFeature:
    value: float
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_finite_number(self.value, "value")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class IntegerFeature:
    value: int
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_integer(self.value, "value")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class BooleanFeature:
    value: bool
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_boolean(self.value, "value")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class PitchUsage:
    pitcher_id: str
    opposing_side: str
    pitch_type: str
    usage: float
    average_velocity_mph: float
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        _require_side(self.opposing_side, _EXPECTED_SIDES, "opposing_side")
        _require_nonempty_string(self.pitch_type, "pitch_type")
        _require_finite_number(self.usage, "usage")
        if not 0.0 <= self.usage <= 1.0:
            raise ValueError("usage must be within [0, 1]")
        _require_finite_number(self.average_velocity_mph, "average_velocity_mph")
        if self.average_velocity_mph <= 0.0:
            raise ValueError("average_velocity_mph must be positive")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class PitchVelocityResponse:
    batter_id: str
    pitcher_id: str
    expected_side: str
    pitch_type: str
    velocity_center_mph: float
    band_width_mph: float
    plate_appearances: int
    adjusted_woba: float
    sample_strength: str
    slot_label: str
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_nonempty_string(self.batter_id, "batter_id")
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        _require_side(self.expected_side, _EXPECTED_SIDES, "expected_side")
        _require_nonempty_string(self.pitch_type, "pitch_type")
        _require_finite_number(self.velocity_center_mph, "velocity_center_mph")
        if self.velocity_center_mph <= 0.0:
            raise ValueError("velocity_center_mph must be positive")
        _require_finite_number(self.band_width_mph, "band_width_mph")
        if self.band_width_mph not in (1.5, 3.0):
            raise ValueError("band_width_mph must be exactly 1.5 or 3.0")
        _require_integer(self.plate_appearances, "plate_appearances")
        if self.plate_appearances < 0:
            raise ValueError("plate_appearances must be nonnegative")
        _require_finite_number(self.adjusted_woba, "adjusted_woba")
        _require_nonempty_string(self.sample_strength, "sample_strength")
        if not isinstance(self.slot_label, str) or self.slot_label not in _SLOT_LABELS:
            raise ValueError(
                "slot_label must be one of the fixed pitch-velocity labels"
            )
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class LineupSlot:
    order: int
    batter_id: str
    batter_name: str
    bats: str
    expected_side: str
    responses: tuple[PitchVelocityResponse, ...]

    def __post_init__(self) -> None:
        _require_integer(self.order, "order")
        if not 1 <= self.order <= 9:
            raise ValueError("order must be within 1 through 9")
        _require_nonempty_string(self.batter_id, "batter_id")
        _require_nonempty_string(self.batter_name, "batter_name")
        _require_side(self.bats, _BATTING_SIDES, "bats")
        _require_side(self.expected_side, _EXPECTED_SIDES, "expected_side")
        _require_tuple_of(self.responses, PitchVelocityResponse, "responses")
        for response in self.responses:
            if response.batter_id != self.batter_id:
                raise ValueError("responses must reference the lineup slot batter_id")


@dataclass(frozen=True)
class LineupInput:
    team: str
    state: LineupState
    state_provenance: FeatureProvenance
    slots: tuple[LineupSlot, ...]
    projected_runs_per_9: NumericFeature
    projected_traffic_per_9: NumericFeature
    projected_extra_base_hits_per_9: NumericFeature
    walk_rate: NumericFeature
    strikeout_rate: NumericFeature
    isolated_power: NumericFeature
    barrel_rate: NumericFeature
    recent_form_index: NumericFeature
    lower_order_continuity: NumericFeature
    log_standard_error: NumericFeature
    recent_form_included: BooleanFeature

    def __post_init__(self) -> None:
        _require_nonempty_string(self.team, "team")
        _require_enum(self.state, LineupState, "state")
        _require_instance(
            self.state_provenance,
            FeatureProvenance,
            "state_provenance",
        )
        _require_tuple_of(self.slots, LineupSlot, "slots")
        if len(self.slots) != 9:
            raise ValueError("lineup must contain exactly nine slots")
        if {slot.order for slot in self.slots} != set(range(1, 10)):
            raise ValueError("lineup order values must be unique and cover 1 through 9")
        numeric_fields = (
            "projected_runs_per_9",
            "projected_traffic_per_9",
            "projected_extra_base_hits_per_9",
            "walk_rate",
            "strikeout_rate",
            "isolated_power",
            "barrel_rate",
            "recent_form_index",
            "lower_order_continuity",
            "log_standard_error",
        )
        for field_name in numeric_fields:
            _require_instance(getattr(self, field_name), NumericFeature, field_name)
        _require_instance(
            self.recent_form_included,
            BooleanFeature,
            "recent_form_included",
        )


@dataclass(frozen=True)
class WorkloadHistory:
    recorded_outs: tuple[int, ...]
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.recorded_outs, tuple):
            raise ValueError("recorded_outs must be a tuple")
        for recorded_out in self.recorded_outs:
            _require_integer(recorded_out, "recorded_outs")
            if not 0 <= recorded_out <= 27:
                raise ValueError("recorded_outs must contain values within [0, 27]")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class StarterProjectionComponents:
    strikeout_minus_walk_rate: NumericFeature
    walk_rate: NumericFeature
    home_runs_per_contact: NumericFeature
    pitches_per_inning: NumericFeature
    recent_pitch_count: NumericFeature
    recent_workload_outs: NumericFeature

    def __post_init__(self) -> None:
        for field_name in (
            "strikeout_minus_walk_rate",
            "walk_rate",
            "home_runs_per_contact",
            "pitches_per_inning",
            "recent_pitch_count",
            "recent_workload_outs",
        ):
            _require_instance(getattr(self, field_name), NumericFeature, field_name)


@dataclass(frozen=True)
class PitcherInput:
    pitcher_id: str
    pitcher_name: str
    role: PitcherRole
    throws: str
    projected_ra9: NumericFeature
    projection_components: StarterProjectionComponents
    log_standard_error: NumericFeature
    eligible_workload: WorkloadHistory
    league_prior_workload: WorkloadHistory
    pitch_limit: IntegerFeature | None
    pitch_mix: tuple[PitchUsage, ...]
    command_risk: BooleanFeature
    home_run_risk: BooleanFeature
    injury_return: BooleanFeature
    recent_short_exit: BooleanFeature

    def __post_init__(self) -> None:
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        _require_nonempty_string(self.pitcher_name, "pitcher_name")
        _require_enum(self.role, PitcherRole, "role")
        _require_side(self.throws, _EXPECTED_SIDES, "throws")
        _require_instance(self.projected_ra9, NumericFeature, "projected_ra9")
        _require_instance(
            self.projection_components,
            StarterProjectionComponents,
            "projection_components",
        )
        _require_instance(self.log_standard_error, NumericFeature, "log_standard_error")
        _require_instance(self.eligible_workload, WorkloadHistory, "eligible_workload")
        _require_instance(
            self.league_prior_workload,
            WorkloadHistory,
            "league_prior_workload",
        )
        _require_optional_instance(self.pitch_limit, IntegerFeature, "pitch_limit")
        if self.pitch_limit is not None and self.pitch_limit.value < 0:
            raise ValueError("pitch_limit must be nonnegative")
        _require_tuple_of(self.pitch_mix, PitchUsage, "pitch_mix")
        for pitch_usage in self.pitch_mix:
            if pitch_usage.pitcher_id != self.pitcher_id:
                raise ValueError("pitch_mix must reference the PitcherInput pitcher_id")
        for field_name in (
            "command_risk",
            "home_run_risk",
            "injury_return",
            "recent_short_exit",
        ):
            _require_instance(getattr(self, field_name), BooleanFeature, field_name)


@dataclass(frozen=True)
class PitcherChainInput:
    pitchers: tuple[PitcherInput, ...]
    official_pitcher_id: str
    rotowire_pitcher_id: str | None
    matchup_pitcher_id: str
    starter_confirmed: BooleanFeature
    starter_source_conflict: BooleanFeature
    handoff_verified: BooleanFeature
    timing_uncertain: BooleanFeature
    eligible_combined_workload: WorkloadHistory | None
    league_prior_combined_workload: WorkloadHistory | None
    combined_pitch_limit: IntegerFeature | None

    def __post_init__(self) -> None:
        _require_tuple_of(self.pitchers, PitcherInput, "pitchers")
        if not self.pitchers:
            raise ValueError("pitchers must contain at least one PitcherInput")
        pitcher_ids = tuple(pitcher.pitcher_id for pitcher in self.pitchers)
        if len(set(pitcher_ids)) != len(pitcher_ids):
            raise ValueError("pitchers must have a unique pitcher_id")
        chain_ids = set(pitcher_ids)
        self._validate_pitcher_reference(
            self.official_pitcher_id,
            "official_pitcher_id",
            chain_ids,
        )
        self._validate_pitcher_reference(
            self.rotowire_pitcher_id,
            "rotowire_pitcher_id",
            chain_ids,
            optional=True,
        )
        self._validate_pitcher_reference(
            self.matchup_pitcher_id,
            "matchup_pitcher_id",
            chain_ids,
        )
        for field_name in (
            "starter_confirmed",
            "starter_source_conflict",
            "handoff_verified",
            "timing_uncertain",
        ):
            _require_instance(getattr(self, field_name), BooleanFeature, field_name)
        _require_optional_instance(
            self.eligible_combined_workload,
            WorkloadHistory,
            "eligible_combined_workload",
        )
        _require_optional_instance(
            self.league_prior_combined_workload,
            WorkloadHistory,
            "league_prior_combined_workload",
        )
        _require_optional_instance(
            self.combined_pitch_limit,
            IntegerFeature,
            "combined_pitch_limit",
        )
        if self.combined_pitch_limit is not None and self.combined_pitch_limit.value < 0:
            raise ValueError("combined_pitch_limit must be nonnegative")

        if len(self.pitchers) > 1:
            roles = tuple(pitcher.role for pitcher in self.pitchers)
            if roles[0] is not PitcherRole.OPENER:
                raise ValueError("first pitcher role must be opener")
            later_roles = roles[1:]
            if any(
                role not in {PitcherRole.BULK, PitcherRole.PRIM}
                for role in later_roles
            ):
                raise ValueError("later pitcher roles must be bulk or prim")
            if len(set(later_roles)) != len(later_roles):
                raise ValueError("later pitcher roles must be distinct")
            if self.eligible_combined_workload is None:
                raise ValueError(
                    "eligible_combined_workload is required for a multi-pitcher "
                    "opener/bulk/prim chain"
                )
            if self.league_prior_combined_workload is None:
                raise ValueError(
                    "league_prior_combined_workload is required for a multi-pitcher "
                    "opener/bulk/prim chain"
                )

        is_single_conventional_starter = (
            len(self.pitchers) == 1
            and self.pitchers[0].role is PitcherRole.STARTER
        )
        if is_single_conventional_starter and any(
            value is not None
            for value in (
                self.eligible_combined_workload,
                self.league_prior_combined_workload,
                self.combined_pitch_limit,
            )
        ):
            raise ValueError(
                "a single conventional starter cannot have combined workload fields"
            )

    @staticmethod
    def _validate_pitcher_reference(
        pitcher_id: object,
        field_name: str,
        chain_ids: set[str],
        *,
        optional: bool = False,
    ) -> None:
        if pitcher_id is None and optional:
            return
        _require_nonempty_string(pitcher_id, field_name)
        if pitcher_id not in chain_ids:
            raise ValueError(f"{field_name} must reference a pitcher in the chain")


@dataclass(frozen=True)
class RelieverAppearanceInput:
    pitcher_id: str
    days_ago: int
    pitches: int
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_nonempty_string(self.pitcher_id, "pitcher_id")
        _require_integer(self.days_ago, "days_ago")
        if self.days_ago < 0:
            raise ValueError("days_ago must be nonnegative")
        _require_integer(self.pitches, "pitches")
        if self.pitches < 0:
            raise ValueError("pitches must be nonnegative")
        _require_instance(self.provenance, FeatureProvenance, "provenance")


@dataclass(frozen=True)
class BullpenSegmentInput:
    segment_id: str
    role: BullpenRole
    projected_ra9: NumericFeature
    log_standard_error: NumericFeature
    appearances: tuple[RelieverAppearanceInput, ...]
    available: BooleanFeature
    limited_performance_or_use_risk: BooleanFeature

    def __post_init__(self) -> None:
        _require_nonempty_string(self.segment_id, "segment_id")
        _require_enum(self.role, BullpenRole, "role")
        _require_instance(self.projected_ra9, NumericFeature, "projected_ra9")
        _require_instance(self.log_standard_error, NumericFeature, "log_standard_error")
        _require_tuple_of(
            self.appearances,
            RelieverAppearanceInput,
            "appearances",
        )
        _require_instance(self.available, BooleanFeature, "available")
        _require_instance(
            self.limited_performance_or_use_risk,
            BooleanFeature,
            "limited_performance_or_use_risk",
        )


@dataclass(frozen=True)
class BullpenInput:
    team: str
    segments: tuple[BullpenSegmentInput, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.team, "team")
        _require_tuple_of(self.segments, BullpenSegmentInput, "segments")


@dataclass(frozen=True)
class MarketInput:
    market_key: str
    market_type: MarketType
    settlement_format: SettlementFormat
    selected_side: str
    line: float | None
    selected_decimal_price: float | None
    opposing_decimal_price: float | None
    provenance: FeatureProvenance

    def __post_init__(self) -> None:
        _require_nonempty_string(self.market_key, "market_key")
        _require_enum(self.market_type, MarketType, "market_type")
        _require_enum(
            self.settlement_format,
            SettlementFormat,
            "settlement_format",
        )
        _require_nonempty_string(self.selected_side, "selected_side")
        if self.line is not None:
            _require_finite_number(self.line, "line")
        self._validate_price(self.selected_decimal_price, "selected_decimal_price")
        self._validate_price(self.opposing_decimal_price, "opposing_decimal_price")
        _require_instance(self.provenance, FeatureProvenance, "provenance")

    @staticmethod
    def _validate_price(price: object, field_name: str) -> None:
        if price is None:
            return
        _require_finite_number(price, field_name)
        if price <= 0.0:
            raise ValueError(f"{field_name} must be positive when present")


@dataclass(frozen=True)
class GameIdentity:
    game_pk: int
    korean_date: date
    official_game_date: date
    first_pitch: datetime
    away_team: str
    home_team: str

    def __post_init__(self) -> None:
        _require_integer(self.game_pk, "game_pk")
        if self.game_pk <= 0:
            raise ValueError("game_pk must be positive")
        _require_date(self.korean_date, "korean_date")
        _require_date(self.official_game_date, "official_game_date")
        _require_aware_datetime(self.first_pitch, "first_pitch")
        _require_nonempty_string(self.away_team, "away_team")
        _require_nonempty_string(self.home_team, "home_team")


@dataclass(frozen=True)
class NormalizedGameInput:
    identity: GameIdentity
    captured_at: datetime
    away_lineup: LineupInput
    home_lineup: LineupInput
    away_pitching: PitcherChainInput
    home_pitching: PitcherChainInput
    away_bullpen: BullpenInput
    home_bullpen: BullpenInput
    markets: tuple[MarketInput, ...]
    observations: tuple[SourceObservation, ...]

    def __post_init__(self) -> None:
        _require_instance(self.identity, GameIdentity, "identity")
        _require_aware_datetime(self.captured_at, "captured_at")
        _require_instance(self.away_lineup, LineupInput, "away_lineup")
        _require_instance(self.home_lineup, LineupInput, "home_lineup")
        _require_instance(self.away_pitching, PitcherChainInput, "away_pitching")
        _require_instance(self.home_pitching, PitcherChainInput, "home_pitching")
        _require_instance(self.away_bullpen, BullpenInput, "away_bullpen")
        _require_instance(self.home_bullpen, BullpenInput, "home_bullpen")
        _require_tuple_of(self.markets, MarketInput, "markets")
        _require_tuple_of(self.observations, SourceObservation, "observations")
