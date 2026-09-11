from dataclasses import fields, replace
from datetime import datetime, timezone

import pytest

from analysis.forecasting.inputs import (
    FeatureProvenance,
    MarketInput,
    MarketType,
    SettlementFormat,
)
from analysis.forecasting.config import SimulationConfig
from analysis.model import (
    PriorityAdmissionContext,
    PriorityEvidencePath,
    PriorityMarketCandidate,
)
from analysis.review.classification import (
    ContradictionRule,
    PrincipalPathBinding,
)
import analysis.review.evidence_contract as evidence_contract_module
from analysis.review.evidence_contract import assess_priority_evidence_contract
from analysis.pipeline import load_model_config
from analysis.shadow import CandidateReviewBinding
from tests.test_forecasting_inputs import complete_forecast_game


UTC = timezone.utc


def _provenance() -> FeatureProvenance:
    timestamp = datetime(2026, 8, 5, tzinfo=UTC)
    return FeatureProvenance(
        observation_ids=("market:1",),
        latest_available_at=timestamp,
        lookback_end=timestamp,
    )


def _market(
    market_type: MarketType,
    selected_side: str,
    *,
    line: float | None = None,
    market_key: str = "market:1",
) -> MarketInput:
    return MarketInput(
        market_key=market_key,
        market_type=market_type,
        settlement_format=SettlementFormat.TWO_WAY,
        selected_side=selected_side,
        line=line,
        selected_decimal_price=1.91,
        opposing_decimal_price=1.91,
        provenance=_provenance(),
    )


def _candidate(
    market: str,
    *,
    score: int = 80,
    rating: str = "play",
    paths: frozenset[PriorityEvidencePath] = frozenset(),
) -> PriorityMarketCandidate:
    return PriorityMarketCandidate(
        market=market,
        score=score,
        decision_rating=rating,
        evidence_paths=paths,
    )


def _binding(
    market: str,
    *paths: PrincipalPathBinding,
    market_key: str = "market:1",
) -> CandidateReviewBinding:
    return CandidateReviewBinding(
        market_key=market_key,
        legacy_candidate_index=0,
        legacy_market=market,
        principal_paths=paths,
    )


def _starter_suppression(side: str) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_SUPPRESSION,
        component_ids=(f"{side}_pitcher_chain_runs_allowed",),
        contradiction_rule=ContradictionRule.STARTER_SUPPRESSION,
    )


def _starter_collapse(side: str) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.STARTER_COLLAPSE,
        component_ids=(
            f"{side}_pitcher_chain_runs_allowed",
            f"{side}_pitcher_chain_workload_outs",
        ),
        contradiction_rule=ContradictionRule.STARTER_COLLAPSE,
    )


def _conversion(
    path: PriorityEvidencePath,
    side: str,
    horizon: str = "full",
) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=path,
        component_ids=(
            f"{side}_{horizon}_runs",
            f"{side}_{horizon}_traffic",
            f"{side}_{horizon}_extra_base_hits",
        ),
        contradiction_rule=ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION,
    )


def _late_exposure(side: str) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.LATE_BULLPEN_EXPOSURE,
        component_ids=(
            f"{side}_pitcher_chain_workload_outs",
            f"{side}_bullpen_segment_lower_exposed",
        ),
        contradiction_rule=ContradictionRule.LATE_BULLPEN_EXPOSURE,
    )


def _lineup_suppression(
    side: str,
    horizon: str = "full",
) -> PrincipalPathBinding:
    return PrincipalPathBinding(
        evidence_path=PriorityEvidencePath.LINEUP_SUPPRESSION,
        component_ids=(
            f"{side}_{horizon}_runs",
            f"{side}_{horizon}_traffic",
            f"{side}_{horizon}_extra_base_hits",
        ),
        contradiction_rule=ContradictionRule.LINEUP_SUPPRESSION,
    )


def _admission_context(market: str, score: int = 80) -> PriorityAdmissionContext:
    return PriorityAdmissionContext(
        market=market,
        proposed_score=score,
        bullpen_performance_support_ok=True,
        full_game_margin_path=True,
    )


def _bound_admission_score(
    candidate: PriorityMarketCandidate,
    binding: CandidateReviewBinding,
    market: MarketInput,
    *,
    score: int = 80,
    candidate_index: int = 0,
) -> int:
    return evidence_contract_module.evaluate_bound_priority_admission(
        _admission_context(
            "over" if market.selected_side == "over" else candidate.market,
            score,
        ),
        candidate=candidate,
        candidate_index=candidate_index,
        binding=binding,
        market_input=market,
    ).score


def test_complete_ml_starter_suppression_keeps_priority_score() -> None:
    candidate = _candidate(
        "ml",
        paths=frozenset({PriorityEvidencePath.STARTER_SUPPRESSION}),
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", _starter_suppression("away")),
        _market(MarketType.ML, "away"),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()
    assert _bound_admission_score(
        candidate,
        _binding("ml", _starter_suppression("away")),
        _market(MarketType.ML, "away"),
    ) == 80


def test_complete_ml_opposing_starter_collapse_keeps_priority_score() -> None:
    candidate = _candidate(
        "ml",
        paths=frozenset({PriorityEvidencePath.STARTER_COLLAPSE}),
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", _starter_collapse("home")),
        _market(MarketType.ML, "away"),
    )

    assert assessment.score_cap is None


def test_playable_ml_without_a_bound_starter_path_is_capped() -> None:
    candidate = _candidate("ml")
    binding = _binding(
        "ml",
        _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
    )
    market = _market(MarketType.ML, "away")
    assessment = assess_priority_evidence_contract(
        candidate,
        binding,
        market,
    )

    assert assessment.score_cap == 71
    assert assessment.reason_codes == ("missing_ml_starter_evidence",)
    assert _bound_admission_score(candidate, binding, market) == 71


def test_code_derived_caps_match_approved_model_config() -> None:
    priority_config = load_model_config()["priority_admission"]

    ml_assessment = assess_priority_evidence_contract(
        _candidate("ml"),
        _binding("ml"),
        _market(MarketType.ML, "away"),
    )
    total_assessment = assess_priority_evidence_contract(
        _candidate("full_total"),
        _binding("full_total"),
        _market(
            MarketType.FULL_TOTAL,
            "over",
            line=priority_config["high_total_two_sided_over_threshold"],
        ),
    )

    assert ml_assessment.score_cap == priority_config["ml_starter_evidence_cap"]
    assert (
        total_assessment.score_cap
        == priority_config["high_total_two_sided_over_cap"]
    )


def test_public_admission_context_does_not_accept_a_detached_assessment() -> None:
    assert "evidence_assessment" not in {
        field.name for field in fields(PriorityAdmissionContext)
    }
    assert "evidence_market_type" not in {
        field.name for field in fields(PriorityAdmissionContext)
    }


def test_bound_admission_rejects_candidate_score_mismatch() -> None:
    candidate = _candidate("ml", score=72)

    with pytest.raises(ValueError, match="candidate_score_mismatch"):
        evidence_contract_module.evaluate_bound_priority_admission(
            _admission_context("ml", 80),
            candidate=candidate,
            candidate_index=0,
            binding=_binding("ml"),
            market_input=_market(MarketType.ML, "away"),
        )


def test_bound_admission_rejects_binding_candidate_index_mismatch() -> None:
    candidate = _candidate("ml")

    with pytest.raises(ValueError, match="binding_candidate_index_mismatch"):
        evidence_contract_module.evaluate_bound_priority_admission(
            _admission_context("ml"),
            candidate=candidate,
            candidate_index=1,
            binding=_binding("ml"),
            market_input=_market(MarketType.ML, "away"),
        )


def test_bound_admission_rejects_unsupported_candidate_rating() -> None:
    candidate = _candidate("ml", rating="review-only")

    with pytest.raises(ValueError, match="candidate_decision_rating"):
        evidence_contract_module.evaluate_bound_priority_admission(
            _admission_context("ml"),
            candidate=candidate,
            candidate_index=0,
            binding=_binding("ml"),
            market_input=_market(MarketType.ML, "away"),
        )


def test_bound_admission_rejects_noncanonical_side_market_selection() -> None:
    candidate = _candidate("ml")

    with pytest.raises(ValueError, match="side_market_selected_side"):
        evidence_contract_module.evaluate_bound_priority_admission(
            _admission_context("ml"),
            candidate=candidate,
            candidate_index=0,
            binding=_binding("ml"),
            market_input=_market(MarketType.ML, "SEA"),
        )


def test_bound_admission_rejects_priority_market_mapping_mismatch() -> None:
    candidate = _candidate("full_total")

    with pytest.raises(ValueError, match="priority_market_mapping_mismatch"):
        evidence_contract_module.evaluate_bound_priority_admission(
            _admission_context("under"),
            candidate=candidate,
            candidate_index=0,
            binding=_binding("full_total"),
            market_input=_market(MarketType.FULL_TOTAL, "over", line=9.0),
        )


@pytest.mark.parametrize(
    ("path", "selected_side"),
    (
        (_starter_suppression("home"), "away"),
        (_starter_collapse("away"), "away"),
    ),
)
def test_ml_directionally_wrong_starter_path_is_capped(
    path: PrincipalPathBinding,
    selected_side: str,
) -> None:
    candidate = _candidate("ml", paths=frozenset({path.evidence_path}))
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", path),
        _market(MarketType.ML, selected_side),
    )

    assert assessment.score_cap == 71


def test_complete_high_total_over_with_two_scoring_sides_keeps_priority_score() -> None:
    away_path = _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away")
    home_path = _conversion(PriorityEvidencePath.RECENT_RUN_CONVERSION, "home")
    candidate = _candidate(
        "full_total",
        paths=frozenset({away_path.evidence_path, home_path.evidence_path}),
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", away_path, home_path),
        _market(MarketType.FULL_TOTAL, "over", line=9.0),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_high_total_over_with_one_scoring_side_is_capped() -> None:
    away_conversion = _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away")
    away_from_home_collapse = _starter_collapse("home")
    candidate = _candidate(
        "full_total",
        paths=frozenset(
            {away_conversion.evidence_path, away_from_home_collapse.evidence_path}
        ),
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", away_conversion, away_from_home_collapse),
        _market(MarketType.FULL_TOTAL, "over", line=9.0),
    )

    assert assessment.score_cap == 69
    assert assessment.reason_codes == (
        "missing_high_total_two_sided_over_evidence",
    )


def test_matchup_only_f5_side_without_recent_conversion_is_capped() -> None:
    matchup = _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away", "f5")
    candidate = _candidate(
        "f5_side",
        paths=frozenset({PriorityEvidencePath.TOP_ORDER_MATCHUP}),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("f5_side", matchup),
        _market(MarketType.F5_SIDE, "away"),
    )

    assert assessment.score_cap == 74
    assert assessment.reason_codes == (
        "matchup_only_side_lacks_recent_run_conversion",
    )


def test_matchup_side_with_recent_conversion_keeps_priority_score() -> None:
    matchup = _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away", "f5")
    conversion = _conversion(
        PriorityEvidencePath.RECENT_RUN_CONVERSION,
        "away",
        "f5",
    )
    candidate = _candidate(
        "f5_side",
        paths=frozenset(
            {
                PriorityEvidencePath.TOP_ORDER_MATCHUP,
                PriorityEvidencePath.RECENT_RUN_CONVERSION,
            }
        ),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("f5_side", matchup, conversion),
        _market(MarketType.F5_SIDE, "away"),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


@pytest.mark.parametrize(
    "paths",
    (
        (
            _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
            _conversion(PriorityEvidencePath.RECENT_RUN_CONVERSION, "home"),
        ),
        (
            _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
            _late_exposure("home"),
        ),
        (
            _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
            _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "home"),
            _late_exposure("home"),
        ),
    ),
)
def test_low_total_over_requires_two_scoring_sides_and_late_variance(
    paths: tuple[PrincipalPathBinding, ...],
) -> None:
    candidate = _candidate(
        "full_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", *paths),
        _market(MarketType.FULL_TOTAL, "over", line=7.0),
    )

    assert assessment.score_cap == 69
    assert assessment.reason_codes == (
        "missing_low_total_two_sided_scoring_or_late_variance",
    )


def test_complete_low_total_over_keeps_priority_score() -> None:
    paths = (
        _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away"),
        _conversion(PriorityEvidencePath.RECENT_RUN_CONVERSION, "home"),
        _late_exposure("home"),
    )
    candidate = _candidate(
        "full_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", *paths),
        _market(MarketType.FULL_TOTAL, "over", line=7.0),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_f5_over_without_recent_conversion_path_is_capped() -> None:
    paths = (
        _starter_collapse("home"),
        _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away", "f5"),
    )
    candidate = _candidate(
        "f5_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("f5_total", *paths),
        _market(MarketType.F5_TOTAL, "over", line=4.5),
    )

    assert assessment.score_cap == 74
    assert assessment.reason_codes == (
        "f5_over_lacks_recent_run_conversion",
    )


def test_f5_over_with_recent_conversion_path_keeps_priority_score() -> None:
    paths = (
        _starter_collapse("home"),
        _conversion(PriorityEvidencePath.RECENT_RUN_CONVERSION, "away", "f5"),
    )
    candidate = _candidate(
        "f5_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("f5_total", *paths),
        _market(MarketType.F5_TOTAL, "over", line=4.5),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_full_under_requires_recent_and_projected_scoring_suppression() -> None:
    game = complete_forecast_game()
    paths = (
        _starter_suppression("home"),
        _lineup_suppression("home"),
    )
    candidate = _candidate(
        "full_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", *paths),
        _market(MarketType.FULL_TOTAL, "under", line=8.5),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap == 69
    assert assessment.reason_codes == (
        "missing_two_sided_recent_run_suppression",
        "missing_two_sided_projected_scoring_suppression",
    )


def test_full_under_with_two_sided_suppression_keeps_priority_score() -> None:
    game = complete_forecast_game()
    suppressed_lineup = lambda lineup: replace(
        lineup,
        projected_runs_per_9=replace(lineup.projected_runs_per_9, value=4.2),
        recent_form_index=replace(lineup.recent_form_index, value=0.90),
    )
    game = replace(
        game,
        away_lineup=suppressed_lineup(game.away_lineup),
        home_lineup=suppressed_lineup(game.home_lineup),
    )
    paths = (
        _starter_suppression("home"),
        _lineup_suppression("home"),
    )
    candidate = _candidate(
        "full_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", *paths),
        _market(MarketType.FULL_TOTAL, "under", line=8.5),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


@pytest.mark.parametrize(
    "risk_field",
    ("command_risk", "home_run_risk", "injury_return", "recent_short_exit"),
)
def test_full_under_is_capped_by_matchup_pitcher_volatility(
    risk_field: str,
) -> None:
    game = complete_forecast_game()
    suppressed_lineup = lambda lineup: replace(
        lineup,
        projected_runs_per_9=replace(lineup.projected_runs_per_9, value=4.2),
        recent_form_index=replace(lineup.recent_form_index, value=0.90),
    )
    home_matchup_id = game.home_pitching.matchup_pitcher_id
    home_pitchers = tuple(
        replace(
            pitcher,
            **{
                risk_field: replace(getattr(pitcher, risk_field), value=True),
            },
        )
        if pitcher.pitcher_id == home_matchup_id
        else pitcher
        for pitcher in game.home_pitching.pitchers
    )
    game = replace(
        game,
        away_lineup=suppressed_lineup(game.away_lineup),
        home_lineup=suppressed_lineup(game.home_lineup),
        home_pitching=replace(game.home_pitching, pitchers=home_pitchers),
    )
    paths = (
        _starter_suppression("home"),
        _lineup_suppression("home"),
    )
    candidate = _candidate(
        "full_total",
        paths=frozenset(path.evidence_path for path in paths),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total", *paths),
        _market(MarketType.FULL_TOTAL, "under", line=8.5),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap == 69
    assert assessment.reason_codes == (
        "starter_volatility_blocks_full_game_under",
    )


@pytest.mark.parametrize("risk_field", ("injury_return", "recent_short_exit"))
def test_selected_side_matchup_pitcher_risk_is_capped(risk_field: str) -> None:
    game = complete_forecast_game()
    opponent_slots = tuple(
        replace(
            slot,
            responses=tuple(
                replace(response, adjusted_woba=0.380, slot_label="플러스")
                for response in slot.responses
            ),
        )
        for slot in game.home_lineup.slots
    )
    game = replace(
        game,
        home_lineup=replace(game.home_lineup, slots=opponent_slots),
    )
    matchup_id = game.away_pitching.matchup_pitcher_id
    pitchers = tuple(
        replace(
            pitcher,
            **{
                risk_field: replace(
                    getattr(pitcher, risk_field),
                    value=True,
                )
            },
        )
        if pitcher.pitcher_id == matchup_id
        else pitcher
        for pitcher in game.away_pitching.pitchers
    )
    game = replace(
        game,
        away_pitching=replace(game.away_pitching, pitchers=pitchers),
    )
    starter_path = _starter_suppression("away")
    candidate = _candidate(
        "ml",
        paths=frozenset({PriorityEvidencePath.STARTER_SUPPRESSION}),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", starter_path),
        _market(MarketType.ML, "away"),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap == 71
    assert assessment.reason_codes == (
        "selected_side_matchup_pitcher_injury_or_short_exit",
    )


def test_neutral_heavy_opponent_lineup_caps_side_priority() -> None:
    game = complete_forecast_game()
    slots = tuple(
        replace(
            slot,
            responses=tuple(
                replace(response, adjusted_woba=0.320, slot_label="중립")
                for response in slot.responses
            ),
        )
        if slot.order <= 7
        else slot
        for slot in game.home_lineup.slots
    )
    game = replace(game, home_lineup=replace(game.home_lineup, slots=slots))
    starter_path = _starter_suppression("away")
    candidate = _candidate(
        "ml",
        paths=frozenset({PriorityEvidencePath.STARTER_SUPPRESSION}),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", starter_path),
        _market(MarketType.ML, "away"),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap == 74
    assert assessment.reason_codes == ("neutral_heavy_opponent_lineup",)


def test_sample_shortage_opponent_slots_are_not_counted_as_neutral() -> None:
    game = complete_forecast_game()
    slots = tuple(
        replace(slot, responses=()) for slot in game.home_lineup.slots
    )
    game = replace(game, home_lineup=replace(game.home_lineup, slots=slots))
    starter_path = _starter_suppression("away")
    candidate = _candidate(
        "ml",
        paths=frozenset({PriorityEvidencePath.STARTER_SUPPRESSION}),
    )

    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", starter_path),
        _market(MarketType.ML, "away"),
        game_input=game,
        simulation_config=SimulationConfig(),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_full_total_requires_over_or_under_selected_side() -> None:
    with pytest.raises(ValueError, match="full_total_selected_side"):
        assess_priority_evidence_contract(
            _candidate("full_total"),
            _binding("full_total"),
            _market(MarketType.FULL_TOTAL, "away", line=9.0),
        )


def test_f5_total_requires_over_or_under_selected_side() -> None:
    with pytest.raises(ValueError, match="full_total_selected_side"):
        assess_priority_evidence_contract(
            _candidate("f5_total"),
            _binding("f5_total"),
            _market(MarketType.F5_TOTAL, "away", line=9.0),
        )


def test_contract_validates_cross_path_compatibility_before_assessment() -> None:
    suppression = _starter_suppression("away")
    collapse = _starter_collapse("away")
    candidate = _candidate(
        "ml",
        paths=frozenset({suppression.evidence_path, collapse.evidence_path}),
    )

    with pytest.raises(ValueError, match="cannot target the same pitching side"):
        assess_priority_evidence_contract(
            candidate,
            _binding("ml", suppression, collapse),
            _market(MarketType.ML, "away"),
        )


def test_contract_rejects_duck_typed_review_binding() -> None:
    class ImpostorBinding:
        legacy_market = "ml"
        market_key = "market:1"
        principal_paths: tuple[PrincipalPathBinding, ...] = ()

    with pytest.raises(TypeError, match="CandidateReviewBinding"):
        assess_priority_evidence_contract(
            _candidate("ml"),
            ImpostorBinding(),
            _market(MarketType.ML, "away"),
        )


@pytest.mark.parametrize(
    ("market_type", "selected_side", "line", "score", "rating"),
    (
        (MarketType.FULL_TOTAL, "under", 9.0, 80, "play"),
        (MarketType.FULL_TOTAL, "over", 8.5, 80, "play"),
        (MarketType.FULL_TOTAL, "over", None, 80, "play"),
        (MarketType.FULL_TOTAL, "over", 9.0, 71, "play"),
        (MarketType.FULL_TOTAL, "over", 9.0, 80, "pass"),
    ),
)
def test_non_qualifying_total_does_not_activate_high_total_contract(
    market_type: MarketType,
    selected_side: str,
    line: float | None,
    score: int,
    rating: str,
) -> None:
    candidate = _candidate(
        market_type.value,
        score=score,
        rating=rating,
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding(market_type.value),
        _market(market_type, selected_side, line=line),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_review_only_rating_does_not_activate_evidence_contract() -> None:
    candidate = _candidate(
        "full_total",
        score=80,
        rating="review-only",
    )
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("full_total"),
        _market(MarketType.FULL_TOTAL, "over", line=9.0),
    )

    assert assessment.score_cap is None
    assert assessment.reason_codes == ()


def test_candidate_binding_and_market_identity_must_match() -> None:
    candidate = _candidate("ml")

    with pytest.raises(ValueError, match="candidate_market_mismatch"):
        assess_priority_evidence_contract(
            candidate,
            _binding("full_total"),
            _market(MarketType.ML, "away"),
        )
