from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from analysis.review.evidence_contract import PriorityEvidenceAssessment


HIGH_LEVERAGE_ROLES = {"closer", "setup", "high"}


@dataclass(frozen=True)
class RelieverAppearance:
    name: str
    role: str
    days_ago: int
    pitches: int


@dataclass(frozen=True)
class BullpenAvailability:
    score: int
    limited_arms: list[str] = field(default_factory=list)
    unavailable_arms: list[str] = field(default_factory=list)
    high_leverage_limited_arms: list[str] = field(default_factory=list)
    lower_leverage_limited_arms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StarterRecentForm:
    era: float
    whip: float
    walks: int
    innings: float
    home_runs: int
    pitches_per_inning: float


@dataclass(frozen=True)
class StarterScore:
    score: int
    command_risk: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StarterVolatilityContext:
    recent_whip: float
    recent_walks_per_nine: float
    recent_home_runs_per_nine: float
    first_inning_era: float
    recent_pitches_per_inning: float
    sample_innings: float


@dataclass(frozen=True)
class StarterVolatility:
    score: int
    collapse_risk: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BullpenSegment:
    name: str
    score: int
    recent_pitches: int
    is_high_leverage: bool


@dataclass(frozen=True)
class BullpenSegmentScore:
    full_game_score: int
    late_inning_risk: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LineupStatus:
    confirmed: bool
    source: str = "RotoWire"


@dataclass(frozen=True)
class LineupSource:
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StarterStructureContext:
    official_pitcher: str
    rotowire_pitcher: str | None = None
    official_pitcher_is_opener: bool = False
    rotowire_pitcher_is_bulk_candidate: bool = False
    rotowire_pitcher_bulk_confirmed: bool = False
    bulk_recent_workload_pitches: int | None = None


@dataclass(frozen=True)
class StarterStructure:
    matchup_pitcher: str
    opener_pitcher: str | None = None
    bulk_pitcher: str | None = None
    starter_source_conflict: bool = False
    confirmed_bulk_pitcher_used_for_matchup: bool = False
    bulk_timing_uncertain: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UmpireContext:
    zone_size: str
    walk_tendency: str
    favors_low_zone: bool
    favors_outer_zone: bool


@dataclass(frozen=True)
class UmpireAdjustment:
    adjusted_f5_edge: int
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PitchVelocityEvidence:
    strength: str
    adjusted_woba: float


@dataclass(frozen=True)
class MarketSignalContext:
    signal_side: str
    model_side: str
    signal_count: int
    line_moved_toward_signal: bool
    bullpen_supports_signal: bool


@dataclass(frozen=True)
class MarketSignal:
    action: str
    flip_pick: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiveGameState:
    inning: int
    run_differential: int
    starter_exited: bool
    bullpen_innings_needed: float
    long_relief_score: int
    decimal_odds: float
    handicap: float = 0.0


@dataclass(frozen=True)
class LiveEntry:
    recommendation: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HandicapContext:
    lineup_edge: int
    starter_stability: int
    bullpen_score: int
    opponent_late_risk: bool
    decimal_odds: float
    starter_survival_score: int | None = None
    starter_sample_risk: bool = False
    opponent_top_order_sample_shortage: bool = False
    lower_order_run_continuity: bool = True
    pitch_velocity_edge_fastball_only: bool = False
    secondary_pitch_support: bool = True
    starter_walks_per_nine: float | None = None
    starter_strikeout_walk_ratio: float | None = None
    starter_recent_home_run_repetition: bool = False
    starter_command_or_home_run_risk: bool = False
    underdog_starter_collapse_risk: bool = False
    opposing_starter_can_miss_bats_and_work_deep: bool = False
    top_five_pitch_velocity_plus_slots: int = 0
    recent_run_conversion: bool = True
    cold_underdog_run_creation: bool = False
    late_bullpen_protection_risk: bool = False


@dataclass(frozen=True)
class HandicapDecision:
    rating: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class F5TotalContext:
    market_total: float | None
    decimal_odds: float | None
    projected_total: float
    starter_stability_a: int
    starter_stability_b: int
    two_sided_run_suppression_verified: bool = False
    lineup_muted_a: bool = False
    lineup_muted_b: bool = False
    lineup_early_scoring_path_a: bool = False
    lineup_early_scoring_path_b: bool = False
    recent_power_or_walk_conversion_a: bool = False
    recent_power_or_walk_conversion_b: bool = False
    top_five_pitch_velocity_plus_a: int = 0
    top_five_pitch_velocity_plus_b: int = 0
    starter_command_or_hr_risk_a: bool = False
    starter_command_or_hr_risk_b: bool = False
    starter_first_inning_collapse_risk_a: bool = False
    starter_first_inning_collapse_risk_b: bool = False
    starter_injury_or_short_exit_risk_a: bool = False
    starter_injury_or_short_exit_risk_b: bool = False
    starter_long_horizon_baseline_risk_a: bool = False
    starter_long_horizon_baseline_risk_b: bool = False
    both_starters_five_inning_floor: bool = False
    opener_bulk_structure: bool = False
    first_five_pitching_chain_verified: bool = True
    pitch_velocity_unknown_rate_a: float = 0.0
    pitch_velocity_unknown_rate_b: float = 0.0


@dataclass(frozen=True)
class TotalContext:
    market_total: float
    projected_total: float
    starter_stability_a: int
    starter_stability_b: int
    lineup_contact_a: int
    lineup_contact_b: int
    bullpen_score_a: int
    bullpen_score_b: int
    lower_bullpen_risk: bool
    pitch_velocity_unknown_rate_a: float = 0.0
    pitch_velocity_unknown_rate_b: float = 0.0
    starter_source_conflict: bool = False
    short_starter_midgame_viable: bool = False
    both_starters_recently_stable: bool = False
    low_total_starter_duel: bool = False
    both_starters_can_work_deep: bool = False
    both_starters_likely_to_reach_sixth: bool = False
    low_total_late_variance_risk: bool = False
    starter_under_but_full_game_pen_risk: bool = False
    starter_length_tempers_plus_slots: bool = False
    negative_power_bat_variance: bool = False
    two_sided_over_volatility: bool = False
    weak_favorite_lineup_with_starter_edge: bool = False
    side_uncertain_total_preferred: bool = False
    under_validation_strict: bool = False
    over_validation_strict: bool = False
    recent_starter_command_or_hr_volatility: bool = False
    neutral_heavy_traffic_risk: bool = False
    low_total_recent_power_hr_variance: bool = False
    both_starters_command_risk: bool = False
    post_sixth_bullpen_exposure_risk: bool = False
    lineup_on_base_path_a: bool = True
    lineup_on_base_path_b: bool = True
    lower_order_run_continuity_a: bool = True
    lower_order_run_continuity_b: bool = True
    high_total_recent_power_walk_conversion: bool = True
    starter_kbb_length_suppresses_over: bool = False
    over_lacks_walk_or_late_bullpen_path: bool = False
    prim_bulk_length_stability: bool = False
    key_bats_missing_but_depth_still_live: bool = False
    weak_bulk_or_lower_bullpen_collapse_risk: bool = False
    one_sided_over_path: bool = False
    high_total_one_lineup_scoring_path_weak: bool = False
    bridge_relief_can_absorb_bullpen_fatigue: bool = False
    fastball_only_over_path_with_secondary_rebound: bool = False
    high_total_single_starter_collapse_with_late_path: bool = False
    cold_lineup_run_conversion_blocks_over: bool = False
    both_starters_have_five_inning_floor: bool = False
    one_starter_kbb_length_dominant: bool = False
    traffic_over_lacks_power_or_run_conversion: bool = False


@dataclass(frozen=True)
class TotalDecision:
    rating: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LineValueContext:
    model_probability: float
    implied_probability: float


@dataclass(frozen=True)
class LineValue:
    rating: str
    edge: float


@dataclass(frozen=True)
class AnalysisChecklist:
    confirmed_lineup: bool = False
    starter_pitch_velocity: bool = False
    batter_pitch_velocity: bool = False
    handedness_splits: bool = False
    recent_form: bool = False
    starter_recent_form: bool = False
    starter_command_risk: bool = False
    bullpen_availability: bool = False
    bullpen_quality: bool = False
    bullpen_handedness: bool = False
    park_weather: bool = False
    defense_baserunning: bool = False
    catcher_battery: bool = False
    run_creation_style: bool = False
    bench_quality: bool = False
    umpire: bool = False
    market_line: bool = False
    line_value: bool = False
    f5_full_game_split: bool = False
    review_tags: bool = False
    caution_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceScore:
    score: int
    grade: str
    caution_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PriorityAdmissionContext:
    market: str
    proposed_score: int
    traffic_only_edge: bool = False
    matchup_only_edge: bool = False
    recent_run_conversion_support: bool = True
    opener_bulk_structure: bool = False
    full_pitching_chain_verified: bool = True
    bullpen_availability_ok: bool = True
    bullpen_late_quality_ok: bool = True
    bullpen_performance_support_ok: bool = False
    full_game_margin_path: bool = False
    lineup_minus_slots: int = 0
    lineup_neutral_slots: int = 0
    lower_order_or_walk_path: bool = True
    bullpen_over_path: bool = False
    evidence_assessment: PriorityEvidenceAssessment | None = None


@dataclass(frozen=True)
class PriorityAdmission:
    score: int
    tier: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TotalMarketPriorityContext:
    f5_score: int | None
    full_game_score: int | None
    independent_paths: bool = False


@dataclass(frozen=True)
class TotalMarketPrioritySelection:
    include_f5: bool
    include_full_game: bool
    notes: list[str] = field(default_factory=list)


class PriorityEvidencePath(str, Enum):
    STARTER_SUPPRESSION = "starter_suppression"
    STARTER_COLLAPSE = "starter_collapse"
    LINEUP_SUPPRESSION = "lineup_suppression"
    TOP_ORDER_MATCHUP = "top_order_matchup"
    RECENT_RUN_CONVERSION = "recent_run_conversion"
    UNDERDOG_RUN_CREATION = "underdog_run_creation"
    LATE_BULLPEN_EXPOSURE = "late_bullpen_exposure"
    BULLPEN_PROTECTION = "bullpen_protection"


@dataclass(frozen=True)
class PriorityMarketCandidate:
    market: str
    score: int
    decision_rating: str
    evidence_paths: frozenset[PriorityEvidencePath] = field(default_factory=frozenset)
    same_game_pair_compatible: bool = False


@dataclass(frozen=True)
class SameGamePrioritySelection:
    included: tuple[PriorityMarketCandidate, ...]
    excluded: tuple[PriorityMarketCandidate, ...]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FullGameContext:
    f5_edge: int
    lineup_edge: int
    bullpen_score: int
    bullpen_quality: int
    market_total: float
    projected_total: float
    run_environment: int = 0
    starter_collapse_risk: bool = False
    late_bullpen_risk: bool = False
    model_side_is_favorite: bool = True
    opponent_pitch_velocity_plus_slots: int = 0
    opponent_prim_bulk_stable: bool = False
    lineup_pitch_velocity_plus_slots: int = 0
    opposing_starter_recent_dominance: bool = False
    opponent_contact_pressure: bool = False
    pitch_velocity_unknown_rate_against_model_starter: float = 0.0
    same_handedness_stack_against_model_starter: bool = False
    starter_source_conflict: bool = False
    top_order_pitch_velocity_plus_slots: int | None = None
    power_pitch_velocity_plus_slots: int | None = None
    low_recent_run_efficiency: bool = False
    model_side_decimal_odds: float | None = None
    starter_command_or_hr_risk: bool = False
    fastball_only_pitch_velocity_plus_slots: int = 0
    opposing_starter_has_two_pitch_command: bool = False
    opposing_prim_bulk_with_weak_lineup_fit: bool = False
    weak_early_scoring_support: bool = False
    key_power_bat_missing: bool = False
    injury_return_or_recent_short_exit_risk: bool = False
    two_sided_run_volatility: bool = False
    opponent_top_order_pitch_velocity_plus_slots: int = 0
    model_starter_hr_or_collapse_risk: bool = False
    confirmed_bulk_pitcher_used_for_matchup: bool = False
    opponent_neutral_pitch_velocity_slots: int = 0
    opponent_pitch_velocity_minus_slots: int = 0
    opponent_negative_power_bat_slots: int = 0
    f5_only_preferred: bool = False
    model_lineup_core_minus_slots: int = 0
    plus_slots_low_run_efficiency_prefers_total: bool = False
    post_sixth_bullpen_exposure_risk: bool = False
    short_starter_requires_extended_bullpen: bool = False
    high_recent_strikeout_rate: bool = False
    unfamiliar_opposing_starter: bool = False
    recent_form_without_on_base_or_lower_order_support: bool = False
    high_leverage_rested_after_lower_bullpen_game: bool = False
    cold_lineup_run_creation_risk: bool = False
    model_lineup_weak_pitch_velocity_fit: bool = False
    opposing_starter_secondary_command_rebound: bool = False
    opener_bulk_timing_uncertain: bool = False
    model_starter_walk_risk_requires_f5_only: bool = False
    model_starter_elite_or_near_elite: bool = False
    opponent_top_order_extra_base_threat_slots: int = 0


@dataclass(frozen=True)
class Recommendation:
    f5_ml: str
    full_game_ml: str
    full_game_total: str
    notes: list[str] = field(default_factory=list)


def evaluate_priority_admission(context: PriorityAdmissionContext) -> PriorityAdmission:
    """Apply final priority caps without rescoring any baseball input."""
    valid_markets = {"f5", "ml", "plus_1_5", "over", "under"}
    if context.market not in valid_markets:
        raise ValueError(f"unsupported priority market: {context.market}")

    score = max(0, min(100, context.proposed_score))
    notes: list[str] = []
    side_market = context.market in {"f5", "ml", "plus_1_5"}

    if (
        side_market
        and (context.traffic_only_edge or context.matchup_only_edge)
        and not context.recent_run_conversion_support
    ):
        score = min(score, 74)
        notes.append("matchup-only side lacks recent run-conversion support")

    if (
        side_market
        and context.opener_bulk_structure
        and not context.full_pitching_chain_verified
    ):
        score = min(score, 74)
        notes.append("opener-bulk-bridge chain is not fully verified")

    if context.market in {"ml", "plus_1_5"} and (
        not context.bullpen_availability_ok or not context.bullpen_late_quality_ok
    ):
        score = min(score, 71)
        notes.append("full-game side lacks late-bullpen quality protection")

    if (
        context.market in {"ml", "plus_1_5"}
        and not context.bullpen_performance_support_ok
    ):
        score = min(score, 71)
        notes.append("bullpen availability lacks current performance support")

    if context.market == "ml" and not context.full_game_margin_path:
        score = min(score, 71)
        notes.append("ML lacks an independent full-game margin path")

    discontinuous_over_path = (
        context.market == "over"
        and context.lineup_minus_slots >= 5
        and context.lineup_neutral_slots == 0
        and not context.lower_order_or_walk_path
        and not context.bullpen_over_path
    )
    if discontinuous_over_path:
        score = min(score, 69)
        notes.append("discontinuous lineup lacks an independent over path")

    if context.evidence_assessment is not None:
        from analysis.review.evidence_contract import PriorityEvidenceAssessment

        if not isinstance(context.evidence_assessment, PriorityEvidenceAssessment):
            raise TypeError(
                "evidence_assessment must be PriorityEvidenceAssessment or None"
            )
        if context.evidence_assessment.score_cap is not None:
            score = min(score, context.evidence_assessment.score_cap)
            notes.extend(context.evidence_assessment.reason_codes)

    if score >= 80:
        tier = "primary"
    elif score >= 72:
        tier = "playable"
    elif score >= 65:
        tier = "lean"
    else:
        tier = "review-only"

    return PriorityAdmission(score=score, tier=tier, notes=notes)


def select_total_priority_markets(
    context: TotalMarketPriorityContext,
) -> TotalMarketPrioritySelection:
    candidates: list[PriorityMarketCandidate] = []
    if context.f5_score is not None:
        candidates.append(
            PriorityMarketCandidate(
                market="f5_total",
                score=context.f5_score,
                decision_rating="play",
                evidence_paths=frozenset(
                    {PriorityEvidencePath.STARTER_COLLAPSE}
                ),
            )
        )
    if context.full_game_score is not None:
        full_game_path = (
            PriorityEvidencePath.LATE_BULLPEN_EXPOSURE
            if context.independent_paths
            else PriorityEvidencePath.STARTER_COLLAPSE
        )
        candidates.append(
            PriorityMarketCandidate(
                market="full_total",
                score=context.full_game_score,
                decision_rating="play",
                evidence_paths=frozenset({full_game_path}),
            )
        )

    selection = assemble_same_game_priority(candidates)
    included_markets = {candidate.market for candidate in selection.included}
    notes = list(selection.notes)
    if not context.independent_paths and len(candidates) == 2:
        if "f5_total" in included_markets:
            notes = ["correlated total markets keep the higher-scoring F5 total"]
        else:
            notes = ["correlated total markets keep the higher-scoring full-game total"]

    return TotalMarketPrioritySelection(
        include_f5="f5_total" in included_markets,
        include_full_game="full_total" in included_markets,
        notes=notes,
    )


PLAYABLE_PRIORITY_MIN_SCORE = 72
PRIORITY_MARKET_FAMILIES = {
    "f5_side": "f5_side",
    "f5_total": "f5_total",
    "ml": "full_side",
    "plus_1_5": "full_side",
    "minus_1_5": "full_side",
    "full_total": "full_total",
}


def _priority_market_family(candidate: PriorityMarketCandidate) -> str | None:
    return PRIORITY_MARKET_FAMILIES.get(candidate.market)


def assemble_same_game_priority(
    candidates: list[PriorityMarketCandidate],
    playable_min_score: int = PLAYABLE_PRIORITY_MIN_SCORE,
) -> SameGamePrioritySelection:
    """Assemble eligible markets by family, pair compatibility, and evidence paths."""
    tie_order = {
        "f5_side": 0,
        "f5_total": 1,
        "full_total": 2,
        "ml": 3,
        "plus_1_5": 4,
        "minus_1_5": 5,
    }
    included: list[PriorityMarketCandidate] = []
    excluded: list[PriorityMarketCandidate] = []
    notes: list[str] = []

    eligible: list[PriorityMarketCandidate] = []
    for candidate in candidates:
        if any(
            not isinstance(path, PriorityEvidencePath)
            for path in candidate.evidence_paths
        ):
            raise ValueError("evidence paths must use canonical PriorityEvidencePath values")
        if candidate.decision_rating == "pass":
            excluded.append(candidate)
            notes.append(f"{candidate.market} pass excluded from priority")
        elif candidate.score < playable_min_score:
            excluded.append(candidate)
            notes.append(f"{candidate.market} below playable priority score")
        elif not candidate.evidence_paths:
            excluded.append(candidate)
            notes.append(f"{candidate.market} excluded because evidence is unverified")
        else:
            eligible.append(candidate)

    eligible.sort(
        key=lambda candidate: (
            -candidate.score,
            tie_order.get(candidate.market, len(tie_order)),
            candidate.market,
        )
    )

    family_winners: set[str] = set()
    fixed_winners: list[PriorityMarketCandidate] = []
    for candidate in eligible:
        candidate_family = _priority_market_family(candidate)
        if candidate_family is not None and candidate_family in family_winners:
            excluded.append(candidate)
            notes.append(
                f"{candidate.market} excluded because its market family is already represented"
            )
            continue
        if candidate_family is not None:
            family_winners.add(candidate_family)
        fixed_winners.append(candidate)

    for candidate in fixed_winners:
        if not included:
            included.append(candidate)
            continue

        candidate_family = _priority_market_family(candidate)
        if candidate_family is not None and any(
            _priority_market_family(selected) == candidate_family
            for selected in included
        ):
            excluded.append(candidate)
            notes.append(f"{candidate.market} excluded because its market family is already represented")
            continue

        incompatible_side_total = any(
            (
                (
                    candidate_family in {"f5_side", "full_side"}
                    and _priority_market_family(selected)
                    in {"f5_total", "full_total"}
                    or candidate_family in {"f5_total", "full_total"}
                    and _priority_market_family(selected)
                    in {"f5_side", "full_side"}
                )
                and not (
                    candidate.same_game_pair_compatible
                    and selected.same_game_pair_compatible
                )
            )
            for selected in included
        )
        if incompatible_side_total:
            excluded.append(candidate)
            notes.append(
                f"{candidate.market} excluded because side-total compatibility is unverified"
            )
            continue

        independence_verified = bool(candidate.evidence_paths) and all(
            bool(selected.evidence_paths)
            and candidate.evidence_paths.isdisjoint(selected.evidence_paths)
            for selected in included
        )
        if independence_verified:
            included.append(candidate)
            continue

        excluded.append(candidate)
        notes.append(
            f"{candidate.market} excluded because same-game evidence is correlated or unverified"
        )

    return SameGamePrioritySelection(
        included=tuple(included),
        excluded=tuple(excluded),
        notes=notes,
    )


def score_bullpen_availability(appearances: list[RelieverAppearance]) -> BullpenAvailability:
    score = 100
    notes: list[str] = []
    limited: set[str] = set()
    unavailable: set[str] = set()
    by_arm: dict[str, list[RelieverAppearance]] = {}

    for appearance in appearances:
        by_arm.setdefault(appearance.name, []).append(appearance)

    for name, arm_usage in by_arm.items():
        role = arm_usage[0].role
        high_leverage = role in HIGH_LEVERAGE_ROLES
        recent_days = {usage.days_ago for usage in arm_usage}
        pitches_yesterday = sum(usage.pitches for usage in arm_usage if usage.days_ago == 1)
        pitches_two_days = sum(usage.pitches for usage in arm_usage if usage.days_ago in {1, 2})
        last_three_appearances = sum(1 for usage in arm_usage if usage.days_ago in {1, 2, 3})

        if {1, 2}.issubset(recent_days):
            score -= 16 if high_leverage else 4
            limited.add(name)
            notes.append(f"{name} used on consecutive days")

        if pitches_yesterday >= 30:
            score -= 18 if high_leverage else 6
            if high_leverage:
                limited.add(name)
            if high_leverage:
                notes.append(f"{name} threw 30+ pitches yesterday; performance/use risk")
            else:
                notes.append(f"{name} lower-leverage workload")
        elif pitches_yesterday >= 21:
            score -= 8 if high_leverage else 3
            if high_leverage:
                limited.add(name)
                notes.append(f"{name} threw 21-29 pitches yesterday")
            else:
                notes.append(f"{name} lower-leverage workload")

        if pitches_two_days >= 35:
            score -= 14 if high_leverage else 5
            limited.add(name)
            if high_leverage:
                notes.append(f"{name} threw 35+ pitches over two days")
            else:
                notes.append(f"{name} lower-leverage workload")

        if last_three_appearances >= 2:
            score -= 8 if high_leverage else 3

        if high_leverage and last_three_appearances >= 3:
            score -= 18
            unavailable.add(name)
            notes.append(f"{name} is likely unavailable")
        elif high_leverage and pitches_two_days >= 45:
            unavailable.add(name)
            notes.append(f"{name} is likely unavailable")

    high_leverage_limited = [name for name in limited if by_arm[name][0].role in HIGH_LEVERAGE_ROLES]
    lower_leverage_limited = [
        name for name in limited if by_arm[name][0].role not in HIGH_LEVERAGE_ROLES
    ]
    if len(high_leverage_limited) >= 2:
        score -= 10
        notes.append("multiple high-leverage arms limited")

    return BullpenAvailability(
        score=max(0, min(100, score)),
        limited_arms=sorted(limited),
        unavailable_arms=sorted(unavailable),
        high_leverage_limited_arms=sorted(high_leverage_limited),
        lower_leverage_limited_arms=sorted(lower_leverage_limited),
        notes=notes,
    )


def score_starter_recent_form(form: StarterRecentForm) -> StarterScore:
    score = 75
    notes: list[str] = []

    walks_per_nine = (form.walks / form.innings) * 9 if form.innings else 0
    home_runs_per_nine = (form.home_runs / form.innings) * 9 if form.innings else 0

    if form.whip >= 1.45:
        score -= 12
        notes.append("high WHIP")

    if walks_per_nine >= 4.0:
        score -= 16
        notes.append("walk risk")

    if form.pitches_per_inning >= 18:
        score -= 8
        notes.append("high pitches per inning")

    if form.era >= 5:
        score -= 8
        notes.append("run prevention risk")

    if home_runs_per_nine >= 1.4:
        score -= 8
        notes.append("home run risk")

    command_risk = walks_per_nine >= 4.0 or form.whip >= 1.45 or form.pitches_per_inning >= 18
    return StarterScore(score=max(0, min(100, score)), command_risk=command_risk, notes=notes)


def score_starter_volatility(context: StarterVolatilityContext) -> StarterVolatility:
    score = 75
    notes: list[str] = []

    if context.sample_innings < 15:
        score -= 8
        notes.append("small starter sample")

    if context.recent_whip >= 1.45:
        score -= 14
        notes.append("traffic risk")

    if context.recent_walks_per_nine >= 4.0:
        score -= 18
        notes.append("walk risk")

    if context.recent_home_runs_per_nine >= 1.4:
        score -= 10
        notes.append("home run risk")

    if context.first_inning_era >= 6.0:
        score -= 12
        notes.append("first-inning risk")

    if context.recent_pitches_per_inning >= 18:
        score -= 8
        notes.append("short-start risk")

    score = max(0, min(100, score))
    collapse_risk = score < 55 or (
        context.recent_walks_per_nine >= 4.5
        and (context.recent_whip >= 1.45 or context.first_inning_era >= 6.0)
    )

    return StarterVolatility(score=score, collapse_risk=collapse_risk, notes=notes)


def score_bullpen_segments(segments: list[BullpenSegment]) -> BullpenSegmentScore:
    if not segments:
        return BullpenSegmentScore(full_game_score=50, late_inning_risk=True, notes=["bullpen segments unknown"])

    notes: list[str] = []
    weighted_total = 0.0
    weight_sum = 0.0
    late_inning_risk = False

    for segment in segments:
        weight = 1.5 if segment.is_high_leverage else 1.0
        adjusted_score = segment.score

        if segment.recent_pitches >= 25:
            adjusted_score -= 10 if segment.is_high_leverage else 5
            notes.append(f"{segment.name} recent workload")

        if segment.name.lower() == "closer" and adjusted_score < 55:
            late_inning_risk = True
            notes.append("closer instability")

        if segment.is_high_leverage and adjusted_score < 50:
            late_inning_risk = True
            notes.append(f"{segment.name} high-leverage risk")

        weighted_total += max(0, adjusted_score) * weight
        weight_sum += weight

    full_game_score = round(weighted_total / weight_sum)
    if late_inning_risk:
        full_game_score = min(full_game_score, 64)

    return BullpenSegmentScore(
        full_game_score=max(0, min(100, full_game_score)),
        late_inning_risk=late_inning_risk,
        notes=notes,
    )


def resolve_lineup_source(status: LineupStatus) -> LineupSource:
    if status.confirmed:
        return LineupSource(source=f"{status.source} confirmed lineup")
    return LineupSource(
        source=f"{status.source} projected lineup",
        notes=["lineup not confirmed"],
    )


def resolve_starter_structure(context: StarterStructureContext) -> StarterStructure:
    notes: list[str] = []
    official_pitcher = context.official_pitcher.strip()
    rotowire_pitcher = (context.rotowire_pitcher or "").strip()
    sources_match = official_pitcher.casefold() == rotowire_pitcher.casefold()

    if not rotowire_pitcher or sources_match:
        return StarterStructure(
            matchup_pitcher=official_pitcher,
            notes=["official starter used for matchup"],
        )

    opener_with_bulk_candidate = (
        context.official_pitcher_is_opener
        and context.rotowire_pitcher_is_bulk_candidate
    )
    if opener_with_bulk_candidate:
        notes.append("official opener and RotoWire bulk candidate identified")
        stable_bulk_evidence = (
            context.rotowire_pitcher_bulk_confirmed
            or (
                context.bulk_recent_workload_pitches is not None
                and context.bulk_recent_workload_pitches >= 50
            )
        )
        if not stable_bulk_evidence:
            notes.append("PRIM/bulk role lacks workload or explicit confirmation")
            return StarterStructure(
                matchup_pitcher=rotowire_pitcher,
                opener_pitcher=official_pitcher,
                bulk_pitcher=rotowire_pitcher,
                starter_source_conflict=True,
                confirmed_bulk_pitcher_used_for_matchup=False,
                bulk_timing_uncertain=True,
                notes=notes,
            )

        notes.append("bulk pitcher is used for pitch-velocity matchup")
        return StarterStructure(
            matchup_pitcher=rotowire_pitcher,
            opener_pitcher=official_pitcher,
            bulk_pitcher=rotowire_pitcher,
            starter_source_conflict=False,
            confirmed_bulk_pitcher_used_for_matchup=True,
            notes=notes,
        )

    notes.append("official and RotoWire starters disagree without confirmed opener/bulk structure")
    return StarterStructure(
        matchup_pitcher=official_pitcher,
        starter_source_conflict=True,
        notes=notes,
    )


def apply_umpire_adjustment(
    base_f5_edge: int,
    umpire: UmpireContext,
    starter_command_risk: bool,
    pitch_profile: str = "",
) -> UmpireAdjustment:
    adjusted = base_f5_edge
    notes: list[str] = []

    if umpire.zone_size == "narrow" and (starter_command_risk or umpire.walk_tendency == "high"):
        adjusted -= 8
        notes.append("narrow zone amplifies command risk")

    if umpire.zone_size == "wide" and umpire.walk_tendency == "low" and not starter_command_risk:
        adjusted += 4
        notes.append("wide zone supports starter command")

    if umpire.favors_low_zone and pitch_profile == "sinker_changeup":
        adjusted += 4
        notes.append("low-zone friendly umpire supports sinker/changeup profile")

    if umpire.favors_outer_zone and pitch_profile in {"slider_cutter", "sweeper_slider"}:
        adjusted += 4
        notes.append("outer-zone friendly umpire supports slider/cutter profile")

    return UmpireAdjustment(adjusted_f5_edge=max(0, min(100, adjusted)), notes=notes)


def evaluate_pitch_velocity_sample(pa: int, woba: float) -> PitchVelocityEvidence:
    if pa < 10:
        return PitchVelocityEvidence(strength="weak", adjusted_woba=(woba * 0.35) + (0.320 * 0.65))
    if pa < 25:
        return PitchVelocityEvidence(strength="moderate", adjusted_woba=(woba * 0.65) + (0.320 * 0.35))
    return PitchVelocityEvidence(strength="strong", adjusted_woba=woba)


def evaluate_market_signal(context: MarketSignalContext) -> MarketSignal:
    notes: list[str] = []
    signal_agrees = context.signal_side == context.model_side
    strong_signal = context.signal_count >= 3 and context.line_moved_toward_signal

    if not strong_signal:
        return MarketSignal(action="ignore", flip_pick=False, notes=["weak market signal"])

    if signal_agrees:
        return MarketSignal(action="confirm", flip_pick=False, notes=["market confirms model side"])

    notes.append("market signal conflicts with model side")
    if not context.bullpen_supports_signal:
        notes.append("signal conflicts with bullpen model")
        return MarketSignal(action="review", flip_pick=False, notes=notes)

    return MarketSignal(action="upgrade", flip_pick=True, notes=notes)


def evaluate_plus_handicap(context: HandicapContext) -> HandicapDecision:
    notes: list[str] = []
    survival_score = context.starter_survival_score
    if survival_score is None:
        survival_score = context.starter_stability

    if survival_score < 60:
        notes.append("starter survival missing")
        return HandicapDecision(rating="pass", notes=notes)

    if context.starter_sample_risk and survival_score < 68:
        notes.append("small-sample starter survival risk")
        return HandicapDecision(rating="pass", notes=notes)

    cap_plus_handicap = False
    small_sample_command_risk = (
        context.starter_sample_risk
        and context.starter_walks_per_nine is not None
        and context.starter_walks_per_nine >= 4.0
        and context.starter_strikeout_walk_ratio is not None
        and context.starter_strikeout_walk_ratio < 2.0
    )
    if small_sample_command_risk:
        cap_plus_handicap = True
        notes.append("small-sample command risk caps plus handicap")

    if context.starter_sample_risk and context.opponent_top_order_sample_shortage:
        cap_plus_handicap = True
        notes.append("opponent top-order sample shortage with small-sample starter caps plus handicap")

    if context.starter_recent_home_run_repetition:
        cap_plus_handicap = True
        notes.append("recent starter HR repetition caps plus handicap")

    if context.underdog_starter_collapse_risk:
        notes.append("underdog starter collapse risk blocks plus handicap")
        return HandicapDecision(rating="pass", notes=notes)

    if context.starter_command_or_home_run_risk:
        cap_plus_handicap = True
        notes.append("starter command/HR risk caps plus handicap")

    if context.cold_underdog_run_creation and not context.recent_run_conversion:
        notes.append("cold underdog run creation blocks plus handicap")
        return HandicapDecision(rating="pass", notes=notes)

    if context.cold_underdog_run_creation:
        cap_plus_handicap = True
        notes.append("cold underdog run creation caps plus handicap")

    if context.opposing_starter_can_miss_bats_and_work_deep:
        has_cover_support = (
            context.top_five_pitch_velocity_plus_slots >= 3
            and context.recent_run_conversion
        )
        if not has_cover_support:
            cap_plus_handicap = True
            notes.append("opposing starter stability caps plus handicap")

    if context.bullpen_score < 55:
        notes.append("bullpen cannot protect handicap")
        return HandicapDecision(rating="pass", notes=notes)

    if context.late_bullpen_protection_risk:
        notes.append("own late-bullpen protection risk blocks plus handicap")
        return HandicapDecision(rating="pass", notes=notes)

    capped_by_run_continuity = (
        context.pitch_velocity_edge_fastball_only
        and not context.lower_order_run_continuity
    )
    if capped_by_run_continuity and not context.secondary_pitch_support:
        notes.append("fastball edge lacks secondary-pitch support")
        return HandicapDecision(rating="pass", notes=notes)

    if context.lineup_edge >= 62 and context.decimal_odds >= 1.60:
        if capped_by_run_continuity:
            notes.append("fastball-only edge with weak lower-order continuity caps plus handicap")
            return HandicapDecision(rating="lean", notes=notes)
        if cap_plus_handicap:
            return HandicapDecision(rating="lean", notes=notes)
        if context.opponent_late_risk:
            notes.append("opponent late risk supports plus handicap")
        return HandicapDecision(rating="play", notes=notes)

    if context.lineup_edge >= 58 and context.decimal_odds >= 1.70 and context.opponent_late_risk:
        return HandicapDecision(rating="lean", notes=["price and opponent late risk support lean"])

    return HandicapDecision(rating="pass", notes=notes)


def evaluate_f5_total(context: F5TotalContext) -> TotalDecision:
    notes: list[str] = []

    if context.market_total is None or context.decimal_odds is None:
        notes.append("F5 market total and decimal price are required")
        return TotalDecision(rating="pass", notes=notes)

    if context.opener_bulk_structure and not context.first_five_pitching_chain_verified:
        notes.append("first-five pitching chain is not verified")
        return TotalDecision(rating="pass", notes=notes)

    risk_a = (
        context.starter_command_or_hr_risk_a
        or context.starter_first_inning_collapse_risk_a
        or context.starter_injury_or_short_exit_risk_a
    )
    risk_b = (
        context.starter_command_or_hr_risk_b
        or context.starter_first_inning_collapse_risk_b
        or context.starter_injury_or_short_exit_risk_b
    )
    high_unknown_pitch_velocity = (
        context.pitch_velocity_unknown_rate_a >= 0.45
        or context.pitch_velocity_unknown_rate_b >= 0.45
    )
    concentrated_path_a = (
        context.top_five_pitch_velocity_plus_a >= 3
        and context.recent_power_or_walk_conversion_a
    )
    concentrated_path_b = (
        context.top_five_pitch_velocity_plus_b >= 3
        and context.recent_power_or_walk_conversion_b
    )
    early_path_a = context.lineup_early_scoring_path_a or concentrated_path_a
    early_path_b = context.lineup_early_scoring_path_b or concentrated_path_b
    supported_path_a = early_path_a and (
        context.recent_power_or_walk_conversion_a or risk_b
    )
    supported_path_b = early_path_b and (
        context.recent_power_or_walk_conversion_b or risk_a
    )

    under_edge = context.market_total - context.projected_total
    if under_edge >= 0.3:
        if not context.two_sided_run_suppression_verified:
            notes.append("F5 under requires verified two-sided run suppression")
            return TotalDecision(rating="pass", notes=notes)
        if (
            context.starter_long_horizon_baseline_risk_a
            or context.starter_long_horizon_baseline_risk_b
        ):
            notes.append("long-horizon starter baseline risk blocks F5 under")
            return TotalDecision(rating="pass", notes=notes)
        if risk_a or risk_b:
            notes.append("starter command, HR, or early-exit risk blocks F5 under")
            return TotalDecision(rating="pass", notes=notes)
        if context.starter_stability_a < 65 or context.starter_stability_b < 65:
            notes.append("F5 under requires both first-five pitching paths stable")
            return TotalDecision(rating="pass", notes=notes)
        if not context.lineup_muted_a or not context.lineup_muted_b:
            notes.append("F5 under requires both lineups muted early")
            return TotalDecision(rating="pass", notes=notes)
        if concentrated_path_a or concentrated_path_b:
            notes.append("converted top-five matchup concentration blocks F5 under")
            return TotalDecision(rating="pass", notes=notes)
        if high_unknown_pitch_velocity:
            notes.append("pitch-velocity sample shortage is not F5 under support")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("stable first-five pitching paths and muted lineups support under")
        return TotalDecision(rating="under", notes=notes)

    over_candidate = context.projected_total >= context.market_total - 0.1
    if over_candidate:
        if context.market_total <= 4.0 and not (supported_path_a and supported_path_b):
            notes.append("low F5 total over requires two early scoring paths")
            return TotalDecision(rating="pass", notes=notes)
        if not (supported_path_a or supported_path_b):
            notes.append("F5 over requires a converted early scoring path")
            return TotalDecision(rating="pass", notes=notes)
        if (
            context.both_starters_five_inning_floor
            and not risk_a
            and not risk_b
            and context.projected_total < context.market_total + 0.5
        ):
            notes.append("stable five-inning starter floors block thin F5 over")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("starter risk and converted early scoring path support over")
        return TotalDecision(rating="over", notes=notes)

    return TotalDecision(rating="pass", notes=notes)


def evaluate_total(context: TotalContext) -> TotalDecision:
    notes: list[str] = []
    total_edge = context.market_total - context.projected_total
    both_starters_stable = context.starter_stability_a >= 65 and context.starter_stability_b >= 65
    both_bullpens_stable = context.bullpen_score_a >= 65 and context.bullpen_score_b >= 65
    both_lineups_muted = context.lineup_contact_a <= 55 and context.lineup_contact_b <= 55
    high_unknown_pitch_velocity = (
        context.pitch_velocity_unknown_rate_a >= 0.45
        or context.pitch_velocity_unknown_rate_b >= 0.45
    )
    two_sided_on_base_and_continuity = (
        context.lineup_on_base_path_a
        and context.lineup_on_base_path_b
        and context.lower_order_run_continuity_a
        and context.lower_order_run_continuity_b
    )

    if context.starter_source_conflict:
        notes.append("starter source conflict lowers confidence but is not over support")

    if context.weak_favorite_lineup_with_starter_edge:
        notes.append("weak favorite lineup makes total review higher priority than ML")

    if context.side_uncertain_total_preferred:
        notes.append("side uncertainty moves O/U into the main priority stack")

    if context.starter_under_but_full_game_pen_risk:
        notes.append("starter under only; bullpen risk blocks full-game under")
        return TotalDecision(rating="pass", notes=notes)

    if context.low_total_late_variance_risk and context.market_total <= 7.0:
        notes.append("low total leaves no margin for late variance")
        return TotalDecision(rating="pass", notes=notes)

    low_total_under_candidate = (
        context.market_total <= 7.5
        and context.projected_total <= context.market_total + 0.2
        and (total_edge >= 0.3 or context.low_total_starter_duel)
    )
    if low_total_under_candidate and context.recent_starter_command_or_hr_volatility:
        notes.append("recent starter command/HR volatility blocks low-total under")
        return TotalDecision(rating="pass", notes=notes)

    if low_total_under_candidate and context.low_total_recent_power_hr_variance:
        notes.append("recent power and HR variance block low-total under")
        return TotalDecision(rating="pass", notes=notes)

    if low_total_under_candidate and context.neutral_heavy_traffic_risk:
        notes.append("neutral-heavy traffic risk blocks low-total under")
        return TotalDecision(rating="pass", notes=notes)

    thin_full_game_under_candidate = (
        context.market_total <= 8.5
        and context.projected_total <= context.market_total
        and total_edge >= 0.3
    )
    if thin_full_game_under_candidate and context.neutral_heavy_traffic_risk:
        notes.append("neutral-heavy traffic risk blocks thin full-game under")
        return TotalDecision(rating="pass", notes=notes)

    under_candidate = context.projected_total <= context.market_total and total_edge >= 0.3
    if under_candidate and context.key_bats_missing_but_depth_still_live:
        notes.append("key-bat absences are not under support when lineup depth remains live")
        return TotalDecision(rating="pass", notes=notes)

    if under_candidate and context.weak_bulk_or_lower_bullpen_collapse_risk:
        notes.append("weak bulk or lower-bullpen collapse risk blocks under")
        return TotalDecision(rating="pass", notes=notes)

    if total_edge >= 0.7:
        if context.negative_power_bat_variance:
            notes.append("negative pitch-velocity power bats add run variance, not under support")
            return TotalDecision(rating="pass", notes=notes)
        if high_unknown_pitch_velocity:
            notes.append("pitch-velocity sample shortage is uncertainty, not under support")
            return TotalDecision(rating="pass", notes=notes)
        if not both_starters_stable:
            notes.append("under requires both starters stable")
            return TotalDecision(rating="pass", notes=notes)
        if not both_bullpens_stable or context.lower_bullpen_risk:
            notes.append("under requires full bullpen stability")
            return TotalDecision(rating="pass", notes=notes)
        if not both_lineups_muted:
            notes.append("under requires muted lineup contact")
            return TotalDecision(rating="pass", notes=notes)
        if context.under_validation_strict:
            notes.append("strict under validation passed")
        notes.append("both starters and bullpens support under")
        return TotalDecision(rating="under", notes=notes)

    if (
        total_edge >= 0.3
        and context.both_starters_recently_stable
        and both_bullpens_stable
        and both_lineups_muted
        and not high_unknown_pitch_velocity
    ):
        if context.negative_power_bat_variance:
            notes.append("negative pitch-velocity power bats add run variance, not under support")
            return TotalDecision(rating="pass", notes=notes)
        if context.under_validation_strict:
            notes.append("strict under validation passed")
        notes.append("both starters' recent stability supports under")
        return TotalDecision(rating="under", notes=notes)

    if (
        context.low_total_starter_duel
        and context.market_total <= 7.5
        and context.projected_total <= context.market_total + 0.1
        and both_starters_stable
        and both_bullpens_stable
        and both_lineups_muted
        and not high_unknown_pitch_velocity
        and not context.lower_bullpen_risk
        and not context.low_total_recent_power_hr_variance
    ):
        if context.negative_power_bat_variance:
            notes.append("negative pitch-velocity power bats add run variance, not under support")
            return TotalDecision(rating="pass", notes=notes)
        if context.under_validation_strict:
            notes.append("strict under validation passed")
        notes.append("low-total starter duel prioritizes under over side")
        return TotalDecision(rating="under", notes=notes)

    if (
        context.one_starter_kbb_length_dominant
        and context.market_total <= 7.5
        and context.projected_total <= context.market_total + 0.2
        and (context.starter_stability_a >= 78 or context.starter_stability_b >= 78)
        and both_bullpens_stable
        and both_lineups_muted
        and not high_unknown_pitch_velocity
        and not context.lower_bullpen_risk
        and not context.low_total_late_variance_risk
        and not context.negative_power_bat_variance
        and not context.low_total_recent_power_hr_variance
    ):
        if context.under_validation_strict:
            notes.append("strict under validation passed")
        notes.append("dominant K/BB length starter prioritizes low-total under over side")
        return TotalDecision(rating="under", notes=notes)

    if (
        context.low_total_starter_duel
        and context.market_total <= 7.5
        and context.projected_total <= context.market_total + 0.2
        and both_starters_stable
        and both_bullpens_stable
        and not context.lower_bullpen_risk
        and not context.low_total_late_variance_risk
        and not context.negative_power_bat_variance
        and not context.low_total_recent_power_hr_variance
        and not high_unknown_pitch_velocity
        and context.both_starters_recently_stable
        and (context.both_starters_likely_to_reach_sixth or context.both_starters_can_work_deep)
        and context.starter_length_tempers_plus_slots
        and context.lineup_contact_a < 70
        and context.lineup_contact_b < 70
    ):
        if context.under_validation_strict:
            notes.append("strict under validation passed")
        notes.append("starter length tempers pitch-velocity plus slots at low total")
        return TotalDecision(rating="under", notes=notes)

    if (
        context.starter_kbb_length_suppresses_over
        and context.market_total >= 8.5
        and not context.lower_bullpen_risk
        and not context.post_sixth_bullpen_exposure_risk
    ):
        notes.append("starter K/BB length suppresses pitch-velocity over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.over_lacks_walk_or_late_bullpen_path
        and context.both_starters_likely_to_reach_sixth
        and both_starters_stable
        and not context.lower_bullpen_risk
        and not context.post_sixth_bullpen_exposure_risk
    ):
        notes.append("sixth-inning starter path requires walk or late-bullpen support for over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.prim_bulk_length_stability
        and context.market_total >= 9.5
        and context.projected_total <= context.market_total + 0.3
    ):
        notes.append("stable PRIM/bulk length tempers high-total over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.fastball_only_over_path_with_secondary_rebound
        and not context.lower_bullpen_risk
        and not context.post_sixth_bullpen_exposure_risk
        and not context.two_sided_over_volatility
    ):
        notes.append("fastball-only over path capped by secondary-command rebound")
        return TotalDecision(rating="pass", notes=notes)

    starter_collapse = context.starter_stability_a < 45 or context.starter_stability_b < 45
    lineup_fit = context.lineup_contact_a >= 70 or context.lineup_contact_b >= 70
    over_candidate = (
        context.projected_total >= context.market_total - 0.2
        and (
            starter_collapse
            or context.lower_bullpen_risk
            or context.post_sixth_bullpen_exposure_risk
            or context.two_sided_over_volatility
            or lineup_fit
        )
    )
    if (
        context.over_validation_strict
        and context.traffic_over_lacks_power_or_run_conversion
        and over_candidate
        and not context.post_sixth_bullpen_exposure_risk
        and not context.both_starters_command_risk
        and not (starter_collapse and context.high_total_recent_power_walk_conversion)
    ):
        notes.append("traffic over needs power or run-conversion support")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.over_validation_strict
        and context.cold_lineup_run_conversion_blocks_over
        and over_candidate
        and not context.lower_bullpen_risk
        and not context.post_sixth_bullpen_exposure_risk
        and not context.both_starters_command_risk
    ):
        notes.append("cold lineup run conversion blocks pitch-velocity over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.both_starters_have_five_inning_floor
        and context.market_total >= 9.0
        and over_candidate
        and not context.lower_bullpen_risk
        and not context.post_sixth_bullpen_exposure_risk
        and not context.both_starters_command_risk
        and not context.high_total_single_starter_collapse_with_late_path
    ):
        notes.append("five-inning starter floor requires extra over support")
        return TotalDecision(rating="pass", notes=notes)

    if context.one_sided_over_path and over_candidate:
        notes.append("one-sided scoring path blocks full-game over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.market_total >= 10.5
        and context.projected_total >= context.market_total - 0.5
        and context.high_total_one_lineup_scoring_path_weak
    ):
        notes.append("high-total over requires two-sided lineup scoring path")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.market_total >= 10.5
        and context.projected_total >= context.market_total - 0.5
        and context.bridge_relief_can_absorb_bullpen_fatigue
        and (context.lower_bullpen_risk or context.post_sixth_bullpen_exposure_risk)
        and not context.two_sided_over_volatility
    ):
        notes.append("available bridge relief can absorb bullpen-fatigue over path")
        return TotalDecision(rating="pass", notes=notes)

    extreme_total_without_clear_collapse = (
        context.market_total >= 13.5
        and context.projected_total >= context.market_total - 0.5
        and not (
            context.two_sided_over_volatility
            and context.high_total_recent_power_walk_conversion
            and (
                context.both_starters_command_risk
                or context.lower_bullpen_risk
                or context.post_sixth_bullpen_exposure_risk
            )
        )
    )
    if extreme_total_without_clear_collapse:
        notes.append("extreme total requires clear two-sided collapse path")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.high_total_single_starter_collapse_with_late_path
        and context.market_total >= 9.5
        and context.projected_total >= context.market_total - 0.5
        and (context.lower_bullpen_risk or context.post_sixth_bullpen_exposure_risk)
        and (context.lineup_contact_a >= 62 or context.lineup_contact_b >= 62)
    ):
        if context.over_validation_strict and not context.high_total_recent_power_walk_conversion:
            notes.append("high-total over requires recent power or walk conversion")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("single starter collapse plus late path keeps high-total over live")
        return TotalDecision(rating="over", notes=notes)

    if (
        context.post_sixth_bullpen_exposure_risk
        and context.projected_total >= context.market_total - 0.2
        and (context.lineup_contact_a >= 58 or context.lineup_contact_b >= 58)
    ):
        if context.over_validation_strict and not two_sided_on_base_and_continuity:
            notes.append("strict over validation requires two-sided on-base and lower-order continuity")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("post-sixth bullpen exposure keeps over live")
        return TotalDecision(rating="over", notes=notes)

    if (
        context.both_starters_can_work_deep
        and both_starters_stable
        and not context.lower_bullpen_risk
        and (context.lineup_contact_a >= 70 or context.lineup_contact_b >= 62)
    ):
        notes.append("starter workload can suppress full-game over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.both_starters_likely_to_reach_sixth
        and both_starters_stable
        and not context.lower_bullpen_risk
        and (context.lineup_contact_a >= 70 or context.lineup_contact_b >= 62)
    ):
        notes.append("starter length can mute lineup-fit over")
        return TotalDecision(rating="pass", notes=notes)

    if (
        context.market_total >= 9.5
        and context.both_starters_command_risk
        and context.two_sided_over_volatility
        and context.projected_total >= context.market_total - 0.3
        and context.lineup_contact_a >= 58
        and context.lineup_contact_b >= 58
        and (context.lineup_contact_a >= 62 or context.lineup_contact_b >= 62)
    ):
        if context.over_validation_strict and not context.high_total_recent_power_walk_conversion:
            notes.append("high-total over requires recent power or walk conversion")
            return TotalDecision(rating="pass", notes=notes)
        if context.over_validation_strict and not two_sided_on_base_and_continuity:
            notes.append("strict over validation requires two-sided on-base and lower-order continuity")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("two command-risk starters keep high-total over live")
        notes.append("two-sided volatility makes over preferable to side")
        return TotalDecision(rating="over", notes=notes)

    if starter_collapse and lineup_fit:
        if context.over_validation_strict and not two_sided_on_base_and_continuity:
            notes.append("strict over validation requires two-sided on-base and lower-order continuity")
            return TotalDecision(rating="pass", notes=notes)
        if context.over_validation_strict and not (
            context.lower_bullpen_risk
            or (
                context.lineup_contact_a >= 62
                and context.lineup_contact_b >= 62
            )
        ):
            notes.append("strict over validation requires two-sided scoring or bullpen exposure")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("starter collapse plus lineup fit supports over")
        if context.two_sided_over_volatility:
            notes.append("two-sided volatility makes over preferable to side")
        return TotalDecision(rating="over", notes=notes)

    if context.lower_bullpen_risk and (context.lineup_contact_a >= 62 or context.lineup_contact_b >= 62):
        if context.short_starter_midgame_viable and not starter_collapse:
            notes.append("short starter can reach middle innings; do not auto-upgrade over")
            return TotalDecision(rating="pass", notes=notes)
        if context.over_validation_strict and not two_sided_on_base_and_continuity:
            notes.append("strict over validation requires two-sided on-base and lower-order continuity")
            return TotalDecision(rating="pass", notes=notes)
        if context.over_validation_strict and context.lineup_contact_a < 58 and context.lineup_contact_b < 58:
            notes.append("strict over validation requires a live lineup path")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("lower bullpen risk supports over")
        if context.two_sided_over_volatility:
            notes.append("two-sided volatility makes over preferable to side")
        return TotalDecision(rating="over", notes=notes)

    if (
        context.two_sided_over_volatility
        and context.projected_total >= context.market_total - 0.2
        and (context.lineup_contact_a >= 62 or context.lineup_contact_b >= 62)
        and (starter_collapse or context.lower_bullpen_risk)
    ):
        if context.over_validation_strict and not two_sided_on_base_and_continuity:
            notes.append("strict over validation requires two-sided on-base and lower-order continuity")
            return TotalDecision(rating="pass", notes=notes)
        if context.over_validation_strict and not (context.lineup_contact_a >= 58 and context.lineup_contact_b >= 58):
            notes.append("strict over validation requires both teams to have a scoring path")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("two-sided volatility makes over preferable to side")
        return TotalDecision(rating="over", notes=notes)

    return TotalDecision(rating="pass", notes=notes)


def evaluate_live_entry(state: LiveGameState) -> LiveEntry:
    notes: list[str] = []

    if state.starter_exited and state.inning <= 3:
        notes.append("early starter exit")

    if state.handicap >= abs(min(0, state.run_differential)) and state.handicap >= 2.5:
        notes.append("handicap cushion")

    if state.starter_exited and state.bullpen_innings_needed >= 6.0 and state.long_relief_score < 60:
        return LiveEntry(recommendation="pass", notes=notes + ["long relief risk"])

    if "handicap cushion" in notes and state.decimal_odds >= 1.65 and state.long_relief_score >= 65:
        return LiveEntry(recommendation="lean", notes=notes)

    if state.decimal_odds >= 2.8 and state.long_relief_score >= 70 and state.run_differential >= -2:
        return LiveEntry(recommendation="lean", notes=notes + ["live price value"])

    return LiveEntry(recommendation="pass", notes=notes)


def evaluate_line_value(context: LineValueContext) -> LineValue:
    edge = context.model_probability - context.implied_probability
    if edge >= 0.04:
        rating = "value"
    elif edge <= -0.02:
        rating = "bad-price"
    else:
        rating = "fair"
    return LineValue(rating=rating, edge=edge)


def calculate_confidence_score(checklist: AnalysisChecklist) -> ConfidenceScore:
    weights = {
        "confirmed_lineup": 10,
        "starter_pitch_velocity": 8,
        "batter_pitch_velocity": 8,
        "handedness_splits": 6,
        "recent_form": 5,
        "starter_recent_form": 7,
        "starter_command_risk": 5,
        "bullpen_availability": 10,
        "bullpen_quality": 6,
        "bullpen_handedness": 4,
        "park_weather": 5,
        "defense_baserunning": 4,
        "catcher_battery": 3,
        "run_creation_style": 3,
        "bench_quality": 3,
        "umpire": 3,
        "market_line": 8,
        "line_value": 6,
        "f5_full_game_split": 6,
        "review_tags": 2,
    }
    score = sum(weight for field_name, weight in weights.items() if getattr(checklist, field_name))
    score -= min(12, len(checklist.caution_flags) * 3)
    score = max(0, min(100, score))

    if checklist.caution_flags and score > 79:
        score = 79

    if score >= 80:
        grade = "high"
    elif score >= 55:
        grade = "medium"
    else:
        grade = "low"

    return ConfidenceScore(score=score, grade=grade, caution_flags=list(checklist.caution_flags))


def recommend_full_game(context: FullGameContext) -> Recommendation:
    notes: list[str] = []
    adjusted_f5_edge = context.f5_edge
    if context.starter_collapse_risk:
        adjusted_f5_edge -= 10
        notes.append("starter collapse risk downgrades F5 first")

    if context.opponent_prim_bulk_stable and context.model_side_is_favorite:
        adjusted_f5_edge -= 8
        notes.append("stable PRIM/bulk pitcher lowers favorite side confidence")

    if context.opposing_starter_recent_dominance and context.lineup_pitch_velocity_plus_slots >= 7:
        adjusted_f5_edge -= 8
        notes.append("dominant opposing starter tempers pitch-velocity plus slots")

    if context.opponent_contact_pressure:
        adjusted_f5_edge -= 8
        notes.append("contact-oriented opponent tempers starter-form edge")

    if context.recent_form_without_on_base_or_lower_order_support:
        adjusted_f5_edge -= 8
        notes.append("recent power form lacks walk pressure or lower-order conversion")

    if context.cold_lineup_run_creation_risk:
        adjusted_f5_edge -= 8
        notes.append("cold lineup run creation risk moves side toward total review")

    if (
        context.pitch_velocity_unknown_rate_against_model_starter >= 0.65
        and context.same_handedness_stack_against_model_starter
    ):
        adjusted_f5_edge -= 8
        notes.append("same-handed lineup with pitch-velocity uncertainty adds power variance")

    if context.starter_source_conflict and not context.confirmed_bulk_pitcher_used_for_matchup:
        adjusted_f5_edge -= 10
        notes.append("starter source conflict blocks strong F5 confidence")
    elif context.starter_source_conflict and context.confirmed_bulk_pitcher_used_for_matchup:
        notes.append("confirmed bulk pitcher resolves starter-source conflict")

    if context.opposing_prim_bulk_with_weak_lineup_fit:
        adjusted_f5_edge -= 8
        notes.append("stable PRIM/bulk starter with weak lineup fit is not a fade")

    if context.opener_bulk_timing_uncertain:
        adjusted_f5_edge -= 8
        notes.append("uncertain opener/bulk timing caps F5 confidence")

    if context.weak_early_scoring_support:
        adjusted_f5_edge -= 8
        notes.append("weak early scoring support caps F5 confidence")

    if context.key_power_bat_missing:
        adjusted_f5_edge -= 10
        notes.append("key power bat missing lowers side priority")

    core_minus_concentration = context.model_lineup_core_minus_slots >= 3
    if core_minus_concentration:
        adjusted_f5_edge -= 10
        notes.append("core lineup pitch-velocity minus concentration caps F5")

    if context.injury_return_or_recent_short_exit_risk:
        adjusted_f5_edge -= 6
        notes.append("injury return or recent short exit caps full-game ML")

    opponent_top_order_attacks_starter_risk = (
        context.opponent_top_order_pitch_velocity_plus_slots >= 3
        and context.model_starter_hr_or_collapse_risk
    )
    if opponent_top_order_attacks_starter_risk:
        adjusted_f5_edge -= 6
        notes.append("opponent top-order pitch-velocity edge attacks starter HR/collapse risk")

    short_favorite_elite_starter_overtrust = (
        context.model_side_is_favorite
        and context.model_side_decimal_odds is not None
        and context.model_side_decimal_odds <= 1.55
        and context.model_starter_elite_or_near_elite
        and context.opponent_top_order_extra_base_threat_slots >= 2
    )
    if short_favorite_elite_starter_overtrust:
        adjusted_f5_edge -= 10
        notes.append(
            "short favorite elite-starter side capped by opponent top-order extra-base route"
        )

    opponent_neutral_depth = (
        context.opponent_neutral_pitch_velocity_slots >= 6
        and context.opponent_pitch_velocity_minus_slots <= 2
    )
    if opponent_neutral_depth:
        adjusted_f5_edge -= 4
        notes.append("neutral-heavy opponent lineup can create traffic")

    opponent_power_variance = context.opponent_negative_power_bat_slots >= 1
    if opponent_power_variance:
        adjusted_f5_edge -= 4
        notes.append("negative pitch-velocity power bats still carry one-swing variance")

    diffuse_plus_slots = (
        context.lineup_pitch_velocity_plus_slots >= 6
        and context.top_order_pitch_velocity_plus_slots is not None
        and context.power_pitch_velocity_plus_slots is not None
        and (
            context.top_order_pitch_velocity_plus_slots < 3
            or context.power_pitch_velocity_plus_slots < 2
        )
    )
    if diffuse_plus_slots:
        adjusted_f5_edge -= 6
        notes.append("plus slots lack top-order or power concentration")

    if context.low_recent_run_efficiency and context.lineup_pitch_velocity_plus_slots >= 6:
        adjusted_f5_edge -= 8
        notes.append("low recent run efficiency tempers pitch-velocity edge")

    strikeout_unfamiliar_starter_cap = (
        context.lineup_pitch_velocity_plus_slots >= 6
        and context.high_recent_strikeout_rate
        and context.unfamiliar_opposing_starter
    )
    if strikeout_unfamiliar_starter_cap:
        adjusted_f5_edge -= 10
        notes.append("high recent strikeout rate plus unfamiliar starter caps pitch-velocity side")

    short_favorite_risk = (
        context.model_side_is_favorite
        and context.model_side_decimal_odds is not None
        and context.model_side_decimal_odds <= 1.60
        and context.starter_command_or_hr_risk
    )
    if short_favorite_risk:
        adjusted_f5_edge -= 8
        notes.append("short favorite with starter command/HR risk capped")

    fastball_only_cap = (
        context.lineup_pitch_velocity_plus_slots >= 6
        and context.fastball_only_pitch_velocity_plus_slots >= 5
        and (
            context.opposing_starter_has_two_pitch_command
            or context.opposing_starter_secondary_command_rebound
        )
    )
    if fastball_only_cap:
        adjusted_f5_edge -= 8
        notes.append("fastball-only plus slots are capped by starter's secondary command")

    f5_ml = _edge_label(max(0, round(adjusted_f5_edge)))

    full_game_score = (
        context.f5_edge * 0.30
        + context.lineup_edge * 0.25
        + context.bullpen_score * 0.25
        + context.bullpen_quality * 0.20
    )

    if (
        context.high_leverage_rested_after_lower_bullpen_game
        and not context.late_bullpen_risk
    ):
        full_game_score += 4
        notes.append("lower-bullpen usage preserved high-leverage availability")

    underdog_pitch_fit_warning = (
        context.model_side_is_favorite
        and context.opponent_pitch_velocity_plus_slots >= 7
    )
    if underdog_pitch_fit_warning:
        full_game_score -= 10
        notes.append("underdog lineup has 7+ pitch-velocity plus slots; favorite ML downgraded")

    if context.opponent_prim_bulk_stable and context.model_side_is_favorite:
        full_game_score -= 8

    if context.opposing_starter_recent_dominance and context.lineup_pitch_velocity_plus_slots >= 7:
        full_game_score -= 8

    if context.opponent_contact_pressure:
        full_game_score -= 6

    if context.recent_form_without_on_base_or_lower_order_support:
        full_game_score -= 6

    if context.cold_lineup_run_creation_risk:
        full_game_score -= 10

    short_priced_weak_lineup_fit = (
        context.model_lineup_weak_pitch_velocity_fit
        and context.model_side_is_favorite
        and context.model_side_decimal_odds is not None
        and context.model_side_decimal_odds <= 1.65
    )
    if short_priced_weak_lineup_fit:
        full_game_score -= 12
        notes.append("weak confirmed lineup fit blocks short-priced ML priority")

    if (
        context.pitch_velocity_unknown_rate_against_model_starter >= 0.65
        and context.same_handedness_stack_against_model_starter
    ):
        full_game_score -= 6

    if context.starter_source_conflict and not context.confirmed_bulk_pitcher_used_for_matchup:
        full_game_score -= 8

    if context.opposing_prim_bulk_with_weak_lineup_fit:
        full_game_score -= 6

    if context.opener_bulk_timing_uncertain:
        full_game_score -= 10

    if diffuse_plus_slots:
        full_game_score -= 8

    if context.low_recent_run_efficiency and context.lineup_pitch_velocity_plus_slots >= 6:
        full_game_score -= 8

    if strikeout_unfamiliar_starter_cap:
        full_game_score -= 12

    if core_minus_concentration:
        full_game_score -= 10

    poor_conversion_with_plus_slots = (
        context.plus_slots_low_run_efficiency_prefers_total
        and context.low_recent_run_efficiency
        and context.lineup_pitch_velocity_plus_slots >= 6
    )
    if poor_conversion_with_plus_slots:
        full_game_score -= 12
        notes.append("plus slots with poor run conversion move ML to pass or total review")

    if short_favorite_risk:
        full_game_score -= 10

    if short_favorite_elite_starter_overtrust:
        full_game_score -= 12

    if fastball_only_cap:
        full_game_score -= 8

    if context.key_power_bat_missing:
        full_game_score -= 12

    if context.injury_return_or_recent_short_exit_risk:
        full_game_score -= 12

    if context.two_sided_run_volatility:
        full_game_score -= 12
        notes.append("two-sided run volatility separates over from side")

    if opponent_top_order_attacks_starter_risk:
        full_game_score -= 12

    if opponent_neutral_depth:
        full_game_score -= 6

    if opponent_power_variance:
        full_game_score -= 6

    if context.late_bullpen_risk:
        full_game_ml = "pass"
        notes.append("late bullpen risk downgrades full-game ML")
    elif context.bullpen_score < 55:
        full_game_ml = "pass"
        notes.append("bullpen fatigue downgrades full-game ML")
    else:
        full_game_ml = _edge_label(round(full_game_score))

    if poor_conversion_with_plus_slots:
        full_game_ml = "pass"

    if opponent_top_order_attacks_starter_risk and context.starter_command_or_hr_risk:
        notes.append("opponent top-order plus slots against HR-risk starter block ML priority")
        full_game_ml = "pass"

    if context.post_sixth_bullpen_exposure_risk:
        notes.append("post-sixth bullpen exposure downgrades full-game ML")
        full_game_ml = "pass"

    if context.short_starter_requires_extended_bullpen:
        notes.append("short starter path requires extended bullpen exposure")
        full_game_ml = "pass"

    if context.model_starter_walk_risk_requires_f5_only:
        notes.append("model starter walk risk keeps side F5-only")
        late_path_supports_ml = (
            context.bullpen_score >= 76
            and context.bullpen_quality >= 76
            and not context.late_bullpen_risk
        )
        if not late_path_supports_ml:
            full_game_ml = "pass"
        elif full_game_ml == "play":
            full_game_ml = "lean"

    if context.f5_only_preferred:
        notes.append("F5-only edge blocks duplicate full-game ML priority")
        late_path_supports_ml = (
            context.bullpen_score >= 72
            and context.bullpen_quality >= 72
            and not context.late_bullpen_risk
        )
        if not late_path_supports_ml:
            full_game_ml = "pass"
        elif full_game_ml == "play":
            full_game_ml = "lean"

    total_edge = context.market_total - context.projected_total
    if total_edge >= 1.0 and context.run_environment <= 0:
        full_game_total = "under"
    elif total_edge <= -1.0 or context.run_environment >= 2:
        full_game_total = "over"
    else:
        full_game_total = "pass"

    if full_game_total == "under" and context.late_bullpen_risk:
        full_game_total = "pass"
        notes.append("late bullpen risk blocks full-game under")
    elif full_game_total == "under" and context.bullpen_score < 60:
        full_game_total = "pass"
        notes.append("bullpen fatigue blocks full-game under")

    return Recommendation(f5_ml=f5_ml, full_game_ml=full_game_ml, full_game_total=full_game_total, notes=notes)


def _edge_label(score: int) -> str:
    if score >= 78:
        return "play"
    if score >= 62:
        return "lean"
    return "pass"
