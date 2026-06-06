from __future__ import annotations

from dataclasses import dataclass, field


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
    lower_order_run_continuity: bool = True
    pitch_velocity_edge_fastball_only: bool = False
    secondary_pitch_support: bool = True


@dataclass(frozen=True)
class HandicapDecision:
    rating: str
    notes: list[str] = field(default_factory=list)


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


@dataclass(frozen=True)
class Recommendation:
    f5_ml: str
    full_game_ml: str
    full_game_total: str
    notes: list[str] = field(default_factory=list)


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
            notes.append(f"{name} threw 30+ pitches yesterday; performance/use risk")
        elif pitches_yesterday >= 21:
            score -= 8 if high_leverage else 3
            if high_leverage:
                limited.add(name)
            notes.append(f"{name} threw 21-29 pitches yesterday")

        if pitches_two_days >= 35:
            score -= 14 if high_leverage else 5
            limited.add(name)
            notes.append(f"{name} threw 35+ pitches over two days")

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
    if len(high_leverage_limited) >= 2:
        score -= 10
        notes.append("multiple high-leverage arms limited")

    return BullpenAvailability(
        score=max(0, min(100, score)),
        limited_arms=sorted(limited),
        unavailable_arms=sorted(unavailable),
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

    if context.bullpen_score < 55:
        notes.append("bullpen cannot protect handicap")
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
        if context.opponent_late_risk:
            notes.append("opponent late risk supports plus handicap")
        return HandicapDecision(rating="play", notes=notes)

    if context.lineup_edge >= 58 and context.decimal_odds >= 1.70 and context.opponent_late_risk:
        return HandicapDecision(rating="lean", notes=["price and opponent late risk support lean"])

    return HandicapDecision(rating="pass", notes=notes)


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

    if context.starter_source_conflict:
        notes.append("starter source conflict lowers confidence but is not over support")

    if context.starter_under_but_full_game_pen_risk:
        notes.append("starter under only; bullpen risk blocks full-game under")
        return TotalDecision(rating="pass", notes=notes)

    if context.low_total_late_variance_risk and context.market_total <= 7.0:
        notes.append("low total leaves no margin for late variance")
        return TotalDecision(rating="pass", notes=notes)

    if total_edge >= 0.7:
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
        return TotalDecision(rating="under", notes=["both starters and bullpens support under"])

    if (
        total_edge >= 0.3
        and context.both_starters_recently_stable
        and both_bullpens_stable
        and both_lineups_muted
        and not high_unknown_pitch_velocity
    ):
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
    ):
        notes.append("low-total starter duel prioritizes under over side")
        return TotalDecision(rating="under", notes=notes)

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

    starter_collapse = context.starter_stability_a < 45 or context.starter_stability_b < 45
    lineup_fit = context.lineup_contact_a >= 70 or context.lineup_contact_b >= 70
    if starter_collapse and lineup_fit:
        notes.append("starter collapse plus lineup fit supports over")
        return TotalDecision(rating="over", notes=notes)

    if context.lower_bullpen_risk and (context.lineup_contact_a >= 62 or context.lineup_contact_b >= 62):
        if context.short_starter_midgame_viable and not starter_collapse:
            notes.append("short starter can reach middle innings; do not auto-upgrade over")
            return TotalDecision(rating="pass", notes=notes)
        notes.append("lower bullpen risk supports over")
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

    if context.weak_early_scoring_support:
        adjusted_f5_edge -= 8
        notes.append("weak early scoring support caps F5 confidence")

    if context.key_power_bat_missing:
        adjusted_f5_edge -= 10
        notes.append("key power bat missing lowers side priority")

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
        and context.opposing_starter_has_two_pitch_command
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

    if (
        context.pitch_velocity_unknown_rate_against_model_starter >= 0.65
        and context.same_handedness_stack_against_model_starter
    ):
        full_game_score -= 6

    if context.starter_source_conflict and not context.confirmed_bulk_pitcher_used_for_matchup:
        full_game_score -= 8

    if context.opposing_prim_bulk_with_weak_lineup_fit:
        full_game_score -= 6

    if diffuse_plus_slots:
        full_game_score -= 8

    if context.low_recent_run_efficiency and context.lineup_pitch_velocity_plus_slots >= 6:
        full_game_score -= 8

    if short_favorite_risk:
        full_game_score -= 10

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

    if context.late_bullpen_risk:
        full_game_ml = "pass"
        notes.append("late bullpen risk downgrades full-game ML")
    elif context.bullpen_score < 55:
        full_game_ml = "pass"
        notes.append("bullpen fatigue downgrades full-game ML")
    else:
        full_game_ml = _edge_label(round(full_game_score))

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
