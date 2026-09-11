"""Negative-only priority caps derived from frozen canonical evidence."""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.inputs import (
    MarketInput,
    MarketType,
    NormalizedGameInput,
)
from analysis.forecasting.offense import forecast_offense
from analysis.model import (
    PriorityAdmission,
    PriorityAdmissionContext,
    PriorityEvidencePath,
    PriorityMarketCandidate,
)

from .classification import (
    PrincipalPathBinding,
    derive_canonical_scoring_side,
    validate_principal_path_bindings,
)


if TYPE_CHECKING:
    from analysis.shadow import CandidateReviewBinding


_PLAYABLE_PRIORITY_MIN_SCORE = 75
_ML_STARTER_EVIDENCE_CAP = 71
_HIGH_TOTAL_TWO_SIDED_OVER_CAP = 69
_MATCHUP_ONLY_SIDE_CAP = 74
_NEUTRAL_HEAVY_SIDE_CAP = 74
_STARTER_RISK_SIDE_CAP = 71
_F5_OVER_CONVERSION_CAP = 74
_FULL_UNDER_SUPPORT_CAP = 69
_LOW_TOTAL_OVER_CAP = 69
_LOW_TOTAL_OVER_MAX = 7.0
_NEUTRAL_HEAVY_MIN_SLOTS = 6
_NEUTRAL_HEAVY_MAX_MINUS_SLOTS = 2
_PLUS_ADJUSTED_WOBA = 0.360
_MINUS_ADJUSTED_WOBA = 0.290
_SAMPLE_SHORTAGE_UNKNOWN_EXPOSURE = 0.65
_SCORING_SIDES = frozenset(("away", "home"))
_DECISION_RATINGS = frozenset(("play", "lean", "pass"))
_SIDE_MARKETS = frozenset((MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5))
_DIRECT_SCORING_PATHS = frozenset(
    (
        PriorityEvidencePath.STARTER_COLLAPSE,
        PriorityEvidencePath.TOP_ORDER_MATCHUP,
        PriorityEvidencePath.RECENT_RUN_CONVERSION,
        PriorityEvidencePath.UNDERDOG_RUN_CREATION,
    )
)
_MATCHUP_ONLY_PATHS = frozenset(
    (
        PriorityEvidencePath.TOP_ORDER_MATCHUP,
        PriorityEvidencePath.BULLPEN_PROTECTION,
        PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
    )
)


@dataclass(frozen=True)
class _SelectedSideGameFacts:
    matchup_pitcher_injury_or_short_exit: bool
    opponent_neutral_slots: int
    opponent_minus_slots: int


@dataclass(frozen=True)
class _FullUnderGameFacts:
    two_sided_recent_run_suppression: bool
    two_sided_projected_scoring_suppression: bool
    starter_volatility: bool


@dataclass(frozen=True)
class PriorityEvidenceAssessment:
    """Immutable, negative-only admission result from frozen evidence."""

    market_key: str
    market_type: MarketType
    selected_side: str
    priority_market: str
    score_cap: int | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.market_key, str) or not self.market_key:
            raise ValueError("market_key must be a nonempty string")
        if not isinstance(self.market_type, MarketType):
            raise ValueError("market_type must be MarketType")
        if not isinstance(self.selected_side, str) or not self.selected_side:
            raise ValueError("selected_side must be a nonempty string")
        if self.priority_market != _priority_market(
            self.market_type,
            self.selected_side,
        ):
            raise ValueError("priority_market must match market identity")
        if self.score_cap is not None and (
            isinstance(self.score_cap, bool)
            or not isinstance(self.score_cap, int)
            or not 0 <= self.score_cap <= 100
        ):
            raise ValueError("score_cap must be an integer in [0, 100] or None")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(code, str) or not code for code in self.reason_codes
        ):
            raise ValueError("reason_codes must be a tuple of nonempty strings")
        if self.score_cap is None and self.reason_codes:
            raise ValueError("reason_codes require a score_cap")


def _validate_identity(
    candidate: object,
    binding: object,
    market_input: object,
) -> tuple[PriorityMarketCandidate, tuple[PrincipalPathBinding, ...], MarketInput]:
    # Import locally because shadow eventually consumes this helper.
    from analysis.shadow import CandidateReviewBinding

    if not isinstance(candidate, PriorityMarketCandidate):
        raise TypeError("candidate must be PriorityMarketCandidate")
    if not isinstance(binding, CandidateReviewBinding):
        raise TypeError("binding must be CandidateReviewBinding")
    if not isinstance(market_input, MarketInput):
        raise TypeError("market_input must be MarketInput")

    legacy_market = binding.legacy_market
    market_key = binding.market_key
    principal_paths = binding.principal_paths
    if not isinstance(principal_paths, tuple) or any(
        not isinstance(path, PrincipalPathBinding) for path in principal_paths
    ):
        raise TypeError("binding principal_paths must be PrincipalPathBinding values")
    if candidate.market != legacy_market:
        raise ValueError("candidate_market_mismatch")
    if binding.market_key != market_input.market_key:
        raise ValueError("market_key_mismatch")
    if candidate.market != market_input.market_type.value:
        raise ValueError("market_type_mismatch")
    if (
        market_input.market_type
        in {MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5}
        and market_input.selected_side not in _SCORING_SIDES
    ):
        raise ValueError("side_market_selected_side_must_be_away_or_home")
    if (
        market_input.market_type in {MarketType.F5_TOTAL, MarketType.FULL_TOTAL}
        and market_input.selected_side not in {"over", "under"}
    ):
        raise ValueError("full_total_selected_side_must_be_over_or_under")
    if not isinstance(candidate.evidence_paths, frozenset) or any(
        not isinstance(path, PriorityEvidencePath) for path in candidate.evidence_paths
    ):
        raise ValueError("candidate_evidence_paths_must_be_canonical")
    if isinstance(candidate.score, bool) or not isinstance(candidate.score, int):
        raise ValueError("candidate_score_must_be_an_integer")
    if not isinstance(candidate.decision_rating, str):
        raise ValueError("candidate_decision_rating_must_be_a_string")
    return candidate, principal_paths, market_input


def _priority_market(market_type: MarketType, selected_side: str) -> str:
    if market_type is MarketType.F5_SIDE:
        return "f5"
    if market_type is MarketType.ML:
        return "ml"
    if market_type is MarketType.PLUS_1_5:
        return "plus_1_5"
    if market_type in {MarketType.F5_TOTAL, MarketType.FULL_TOTAL}:
        if selected_side not in {"over", "under"}:
            raise ValueError("full_total_selected_side_must_be_over_or_under")
        return selected_side
    raise ValueError("unsupported market_type")


def _assessment(
    market_input: MarketInput,
    *,
    score_cap: int | None = None,
    reason_codes: tuple[str, ...] = (),
) -> PriorityEvidenceAssessment:
    return PriorityEvidenceAssessment(
        market_key=market_input.market_key,
        market_type=market_input.market_type,
        selected_side=market_input.selected_side,
        priority_market=_priority_market(
            market_input.market_type,
            market_input.selected_side,
        ),
        score_cap=score_cap,
        reason_codes=reason_codes,
    )


def _is_playable_non_pass(
    candidate: PriorityMarketCandidate,
    playable_min_score: int,
) -> bool:
    return (
        candidate.decision_rating in {"play", "lean"}
        and candidate.score >= playable_min_score
    )


def _is_valid_path_for_market(
    binding: PrincipalPathBinding,
    market_type: MarketType,
) -> bool:
    try:
        validate_principal_path_bindings(market_type, (binding,))
    except ValueError:
        return False
    return True


def _ml_has_starter_path(
    candidate: PriorityMarketCandidate,
    principal_paths: tuple[PrincipalPathBinding, ...],
    selected_side: str,
) -> bool:
    if selected_side not in _SCORING_SIDES:
        return False
    opposing_side = "home" if selected_side == "away" else "away"
    for path in principal_paths:
        if path.evidence_path not in candidate.evidence_paths or not _is_valid_path_for_market(
            path, MarketType.ML
        ):
            continue
        component_ids = set(path.component_ids)
        if (
            path.evidence_path is PriorityEvidencePath.STARTER_SUPPRESSION
            and component_ids == {f"{selected_side}_pitcher_chain_runs_allowed"}
        ):
            return True
        if (
            path.evidence_path is PriorityEvidencePath.STARTER_COLLAPSE
            and component_ids
            == {
                f"{opposing_side}_pitcher_chain_runs_allowed",
                f"{opposing_side}_pitcher_chain_workload_outs",
            }
        ):
            return True
    return False


def _full_total_scoring_sides(
    candidate: PriorityMarketCandidate,
    principal_paths: tuple[PrincipalPathBinding, ...],
) -> set[str]:
    sides_by_path: dict[PriorityEvidencePath, set[str]] = {}
    for path in principal_paths:
        if path.evidence_path not in candidate.evidence_paths or not _is_valid_path_for_market(
            path, MarketType.FULL_TOTAL
        ):
            continue
        scoring_side = derive_canonical_scoring_side(path)
        if scoring_side is None:
            continue
        sides_by_path.setdefault(path.evidence_path, set()).add(scoring_side)

    # A repeated canonical path cannot stand in for two independent routes.
    return {
        next(iter(sides))
        for sides in sides_by_path.values()
        if len(sides) == 1
    }


def _direct_full_total_scoring_sides(
    candidate: PriorityMarketCandidate,
    principal_paths: tuple[PrincipalPathBinding, ...],
) -> set[str]:
    sides_by_path: dict[PriorityEvidencePath, set[str]] = {}
    for path in principal_paths:
        if (
            path.evidence_path not in candidate.evidence_paths
            or path.evidence_path not in _DIRECT_SCORING_PATHS
            or not _is_valid_path_for_market(path, MarketType.FULL_TOTAL)
        ):
            continue
        scoring_side = derive_canonical_scoring_side(path)
        if scoring_side is not None:
            sides_by_path.setdefault(path.evidence_path, set()).add(scoring_side)
    return {
        next(iter(sides))
        for sides in sides_by_path.values()
        if len(sides) == 1
    }


def _has_late_variance_path(
    candidate: PriorityMarketCandidate,
    principal_paths: tuple[PrincipalPathBinding, ...],
) -> bool:
    return any(
        path.evidence_path is PriorityEvidencePath.LATE_BULLPEN_EXPOSURE
        and path.evidence_path in candidate.evidence_paths
        and _is_valid_path_for_market(path, MarketType.FULL_TOTAL)
        for path in principal_paths
    )


def _slot_grade(
    labels: tuple[str, ...],
    exposure_woba: float,
    unknown_exposure: float,
) -> str:
    if unknown_exposure >= _SAMPLE_SHORTAGE_UNKNOWN_EXPOSURE:
        return "표본 부족"
    known_labels = set(labels).difference(("표본 부족",))
    if len(known_labels) == 1 and unknown_exposure == 0.0:
        return next(iter(known_labels))
    if exposure_woba >= _PLUS_ADJUSTED_WOBA:
        return "플러스"
    if exposure_woba < _MINUS_ADJUSTED_WOBA:
        return "마이너스"
    return "중립"


@lru_cache(maxsize=32)
def _derive_game_facts(
    game_input: NormalizedGameInput,
    simulation_config: SimulationConfig,
) -> tuple[_SelectedSideGameFacts, _SelectedSideGameFacts]:
    if not isinstance(game_input, NormalizedGameInput):
        raise TypeError("game_input must be NormalizedGameInput")
    if not isinstance(simulation_config, SimulationConfig):
        raise TypeError("simulation_config must be SimulationConfig")

    away_offense = forecast_offense(
        game_input.away_lineup,
        game_input.home_pitching.pitchers,
        simulation_config,
    )
    home_offense = forecast_offense(
        game_input.home_lineup,
        game_input.away_pitching.pitchers,
        simulation_config,
    )

    def side_facts(selected_side: str) -> _SelectedSideGameFacts:
        if selected_side == "away":
            selected_pitching = game_input.away_pitching
            opponent_offense = home_offense
        else:
            selected_pitching = game_input.home_pitching
            opponent_offense = away_offense

        matchup_pitcher = next(
            pitcher
            for pitcher in selected_pitching.pitchers
            if pitcher.pitcher_id == selected_pitching.matchup_pitcher_id
        )
        opponent_slots = opponent_offense.slot_forecasts_for(
            selected_pitching.matchup_pitcher_id
        )
        grades = tuple(
            _slot_grade(
                slot.labels,
                slot.exposure_weighted_adjusted_woba,
                slot.unknown_exposure,
            )
            for slot in opponent_slots
        )
        return _SelectedSideGameFacts(
            matchup_pitcher_injury_or_short_exit=(
                matchup_pitcher.injury_return.value
                or matchup_pitcher.recent_short_exit.value
            ),
            opponent_neutral_slots=grades.count("중립"),
            opponent_minus_slots=grades.count("마이너스"),
        )

    return side_facts("away"), side_facts("home")


def _game_facts_for_selected_side(
    game_input: NormalizedGameInput | None,
    simulation_config: SimulationConfig | None,
    selected_side: str,
) -> _SelectedSideGameFacts | None:
    if game_input is None:
        if simulation_config is not None:
            raise ValueError("simulation_config requires game_input")
        return None
    config = simulation_config or SimulationConfig()
    away_facts, home_facts = _derive_game_facts(game_input, config)
    return away_facts if selected_side == "away" else home_facts


@lru_cache(maxsize=32)
def _derive_full_under_game_facts(
    game_input: NormalizedGameInput,
    simulation_config: SimulationConfig,
) -> _FullUnderGameFacts:
    lineups = (game_input.away_lineup, game_input.home_lineup)
    matchup_pitchers = tuple(
        next(
            pitcher
            for pitcher in chain.pitchers
            if pitcher.pitcher_id == chain.matchup_pitcher_id
        )
        for chain in (game_input.away_pitching, game_input.home_pitching)
    )
    return _FullUnderGameFacts(
        two_sided_recent_run_suppression=all(
            lineup.recent_form_included.value
            and lineup.recent_form_index.value < 1.0
            for lineup in lineups
        ),
        two_sided_projected_scoring_suppression=all(
            lineup.projected_runs_per_9.value
            < simulation_config.league_runs_per_team_game
            for lineup in lineups
        ),
        starter_volatility=any(
            pitcher.command_risk.value
            or pitcher.home_run_risk.value
            or pitcher.injury_return.value
            or pitcher.recent_short_exit.value
            for pitcher in matchup_pitchers
        ),
    )


def assess_priority_evidence_contract(
    candidate: PriorityMarketCandidate,
    binding: "CandidateReviewBinding",
    market_input: MarketInput,
    *,
    game_input: NormalizedGameInput | None = None,
    simulation_config: SimulationConfig | None = None,
    playable_min_score: int = _PLAYABLE_PRIORITY_MIN_SCORE,
    enforce_current_total_hardening: bool = True,
) -> PriorityEvidenceAssessment:
    """Derive priority caps only from the candidate, binding, and market input."""

    candidate, principal_paths, market_input = _validate_identity(
        candidate,
        binding,
        market_input,
    )
    validate_principal_path_bindings(market_input.market_type, principal_paths)
    if (
        isinstance(playable_min_score, bool)
        or not isinstance(playable_min_score, int)
        or playable_min_score < 0
    ):
        raise ValueError("playable_min_score must be a nonnegative integer")
    if not isinstance(enforce_current_total_hardening, bool):
        raise TypeError("enforce_current_total_hardening must be a boolean")
    if not _is_playable_non_pass(candidate, playable_min_score):
        return _assessment(market_input)

    caps: list[tuple[int, str]] = []

    def add_cap(score_cap: int, reason_code: str) -> None:
        caps.append((score_cap, reason_code))

    if market_input.market_type is MarketType.ML and not _ml_has_starter_path(
        candidate,
        principal_paths,
        market_input.selected_side,
    ):
        add_cap(_ML_STARTER_EVIDENCE_CAP, "missing_ml_starter_evidence")

    matchup_only_side = (
        market_input.market_type in _SIDE_MARKETS
        and PriorityEvidencePath.TOP_ORDER_MATCHUP in candidate.evidence_paths
        and candidate.evidence_paths.issubset(_MATCHUP_ONLY_PATHS)
        and PriorityEvidencePath.RECENT_RUN_CONVERSION
        not in candidate.evidence_paths
    )
    if matchup_only_side:
        add_cap(
            _MATCHUP_ONLY_SIDE_CAP,
            "matchup_only_side_lacks_recent_run_conversion",
        )

    if (
        enforce_current_total_hardening
        and market_input.market_type is MarketType.F5_TOTAL
        and market_input.selected_side == "over"
        and PriorityEvidencePath.RECENT_RUN_CONVERSION
        not in candidate.evidence_paths
    ):
        add_cap(
            _F5_OVER_CONVERSION_CAP,
            "f5_over_lacks_recent_run_conversion",
        )

    if (
        enforce_current_total_hardening
        and market_input.market_type is MarketType.FULL_TOTAL
        and market_input.selected_side == "under"
        and game_input is not None
        and {
            PriorityEvidencePath.STARTER_SUPPRESSION,
            PriorityEvidencePath.LINEUP_SUPPRESSION,
        }.issubset(candidate.evidence_paths)
        and {
            PriorityEvidencePath.STARTER_SUPPRESSION,
            PriorityEvidencePath.LINEUP_SUPPRESSION,
        }.issubset(path.evidence_path for path in principal_paths)
    ):
        under_facts = _derive_full_under_game_facts(
            game_input,
            simulation_config or SimulationConfig(),
        )
        if not under_facts.two_sided_recent_run_suppression:
            add_cap(
                _FULL_UNDER_SUPPORT_CAP,
                "missing_two_sided_recent_run_suppression",
            )
        if not under_facts.two_sided_projected_scoring_suppression:
            add_cap(
                _FULL_UNDER_SUPPORT_CAP,
                "missing_two_sided_projected_scoring_suppression",
            )
        if under_facts.starter_volatility:
            add_cap(
                _FULL_UNDER_SUPPORT_CAP,
                "starter_volatility_blocks_full_game_under",
            )

    if (
        market_input.market_type is MarketType.FULL_TOTAL
        and market_input.selected_side == "over"
        and market_input.line is not None
        and market_input.line >= 9.0
        and _full_total_scoring_sides(candidate, principal_paths) != _SCORING_SIDES
    ):
        add_cap(
            _HIGH_TOTAL_TWO_SIDED_OVER_CAP,
            "missing_high_total_two_sided_over_evidence",
        )

    if (
        market_input.market_type is MarketType.FULL_TOTAL
        and market_input.selected_side == "over"
        and market_input.line is not None
        and market_input.line <= _LOW_TOTAL_OVER_MAX
        and (
            _direct_full_total_scoring_sides(candidate, principal_paths)
            != _SCORING_SIDES
            or not _has_late_variance_path(candidate, principal_paths)
        )
    ):
        add_cap(
            _LOW_TOTAL_OVER_CAP,
            "missing_low_total_two_sided_scoring_or_late_variance",
        )

    if market_input.market_type in _SIDE_MARKETS:
        game_facts = _game_facts_for_selected_side(
            game_input,
            simulation_config,
            market_input.selected_side,
        )
        if game_facts is not None:
            if game_facts.matchup_pitcher_injury_or_short_exit:
                add_cap(
                    _STARTER_RISK_SIDE_CAP,
                    "selected_side_matchup_pitcher_injury_or_short_exit",
                )
            if (
                game_facts.opponent_neutral_slots >= _NEUTRAL_HEAVY_MIN_SLOTS
                and game_facts.opponent_minus_slots
                <= _NEUTRAL_HEAVY_MAX_MINUS_SLOTS
            ):
                add_cap(
                    _NEUTRAL_HEAVY_SIDE_CAP,
                    "neutral_heavy_opponent_lineup",
                )

    if not caps:
        return _assessment(market_input)
    score_cap = min(cap for cap, _ in caps)
    return _assessment(
        market_input,
        score_cap=score_cap,
        reason_codes=tuple(reason for _, reason in caps),
    )


def evaluate_bound_priority_admission(
    context: PriorityAdmissionContext,
    *,
    candidate: PriorityMarketCandidate,
    candidate_index: int,
    binding: "CandidateReviewBinding",
    market_input: MarketInput,
) -> PriorityAdmission:
    """Apply evidence caps derived from one exact candidate-review binding."""

    if not isinstance(context, PriorityAdmissionContext):
        raise TypeError("context must be PriorityAdmissionContext")
    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        raise ValueError("candidate_index_must_be_a_nonnegative_integer")
    candidate, _, market_input = _validate_identity(
        candidate,
        binding,
        market_input,
    )
    if candidate.score != context.proposed_score:
        raise ValueError("candidate_score_mismatch")
    if candidate.decision_rating not in _DECISION_RATINGS:
        raise ValueError("candidate_decision_rating_is_unsupported")
    if binding.legacy_candidate_index != candidate_index:
        raise ValueError("binding_candidate_index_mismatch")

    assessment = assess_priority_evidence_contract(
        candidate,
        binding,
        market_input,
    )
    if context.market != assessment.priority_market:
        raise ValueError("priority_market_mapping_mismatch")

    from analysis.model import _evaluate_priority_admission

    return _evaluate_priority_admission(context, assessment)
