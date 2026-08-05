from datetime import datetime, timezone

import pytest

from analysis.forecasting.inputs import (
    FeatureProvenance,
    MarketInput,
    MarketType,
    SettlementFormat,
)
from analysis.model import (
    PriorityAdmissionContext,
    PriorityEvidencePath,
    PriorityMarketCandidate,
    evaluate_priority_admission,
)
from analysis.review.classification import (
    ContradictionRule,
    PrincipalPathBinding,
)
from analysis.review.evidence_contract import assess_priority_evidence_contract
from analysis.shadow import CandidateReviewBinding


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


def _admission_score(
    assessment: object,
    score: int = 80,
    evidence_market_type: MarketType = MarketType.ML,
) -> int:
    return evaluate_priority_admission(
        PriorityAdmissionContext(
            market="ml",
            proposed_score=score,
            bullpen_performance_support_ok=True,
            full_game_margin_path=True,
            evidence_assessment=assessment,
            evidence_market_type=evidence_market_type,
        )
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
    assert _admission_score(assessment) == 80


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
    assessment = assess_priority_evidence_contract(
        candidate,
        _binding("ml", _conversion(PriorityEvidencePath.TOP_ORDER_MATCHUP, "away")),
        _market(MarketType.ML, "away"),
    )

    assert assessment.score_cap == 71
    assert assessment.reason_codes == ("missing_ml_starter_evidence",)
    assert _admission_score(assessment) == 71


def test_assessment_cannot_cap_a_different_priority_market() -> None:
    assessment = assess_priority_evidence_contract(
        _candidate("ml"),
        _binding("ml"),
        _market(MarketType.ML, "away"),
    )

    with pytest.raises(ValueError, match="evidence_assessment_market_mismatch"):
        evaluate_priority_admission(
            PriorityAdmissionContext(
                market="over",
                proposed_score=80,
                evidence_assessment=assessment,
                evidence_market_type=MarketType.ML,
            )
        )


@pytest.mark.parametrize(
    ("assessment_market_type", "context_market_type"),
    (
        (MarketType.F5_TOTAL, MarketType.FULL_TOTAL),
        (MarketType.FULL_TOTAL, MarketType.F5_TOTAL),
    ),
)
def test_total_assessment_requires_the_exact_context_market_type(
    assessment_market_type: MarketType,
    context_market_type: MarketType,
) -> None:
    assessment = assess_priority_evidence_contract(
        _candidate(assessment_market_type.value),
        _binding(assessment_market_type.value),
        _market(assessment_market_type, "over", line=9.0),
    )

    with pytest.raises(
        ValueError,
        match="evidence_assessment_market_type_mismatch",
    ):
        evaluate_priority_admission(
            PriorityAdmissionContext(
                market="over",
                proposed_score=80,
                evidence_assessment=assessment,
                evidence_market_type=context_market_type,
            )
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
        (MarketType.F5_TOTAL, "over", 9.0, 80, "play"),
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
