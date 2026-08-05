"""Negative-only priority caps derived from frozen canonical evidence."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from analysis.forecasting.inputs import MarketInput, MarketType
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate

from .classification import (
    PrincipalPathBinding,
    derive_canonical_scoring_side,
    validate_principal_path_bindings,
)


if TYPE_CHECKING:
    from analysis.shadow import CandidateReviewBinding


_PLAYABLE_PRIORITY_MIN_SCORE = 72
_ML_STARTER_EVIDENCE_CAP = 71
_HIGH_TOTAL_TWO_SIDED_OVER_CAP = 69
_SCORING_SIDES = frozenset(("away", "home"))


@dataclass(frozen=True)
class PriorityEvidenceAssessment:
    """Immutable, negative-only admission result from frozen evidence."""

    score_cap: int | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
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
    if not isinstance(candidate, PriorityMarketCandidate):
        raise TypeError("candidate must be PriorityMarketCandidate")
    if not isinstance(market_input, MarketInput):
        raise TypeError("market_input must be MarketInput")

    legacy_market = getattr(binding, "legacy_market", None)
    market_key = getattr(binding, "market_key", None)
    principal_paths = getattr(binding, "principal_paths", None)
    if not isinstance(legacy_market, str) or not isinstance(market_key, str):
        raise TypeError("binding must provide canonical market identity")
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
    if not isinstance(candidate.evidence_paths, frozenset) or any(
        not isinstance(path, PriorityEvidencePath) for path in candidate.evidence_paths
    ):
        raise ValueError("candidate_evidence_paths_must_be_canonical")
    if isinstance(candidate.score, bool) or not isinstance(candidate.score, int):
        raise ValueError("candidate_score_must_be_an_integer")
    if not isinstance(candidate.decision_rating, str):
        raise ValueError("candidate_decision_rating_must_be_a_string")
    return candidate, principal_paths, market_input


def _is_playable_non_pass(candidate: PriorityMarketCandidate) -> bool:
    return (
        candidate.decision_rating in {"play", "lean"}
        and candidate.score >= _PLAYABLE_PRIORITY_MIN_SCORE
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


def assess_priority_evidence_contract(
    candidate: PriorityMarketCandidate,
    binding: "CandidateReviewBinding",
    market_input: MarketInput,
) -> PriorityEvidenceAssessment:
    """Derive priority caps only from the candidate, binding, and market input."""

    candidate, principal_paths, market_input = _validate_identity(
        candidate,
        binding,
        market_input,
    )
    if not _is_playable_non_pass(candidate):
        return PriorityEvidenceAssessment()

    if market_input.market_type is MarketType.ML and not _ml_has_starter_path(
        candidate,
        principal_paths,
        market_input.selected_side,
    ):
        return PriorityEvidenceAssessment(
            score_cap=_ML_STARTER_EVIDENCE_CAP,
            reason_codes=("missing_ml_starter_evidence",),
        )

    if (
        market_input.market_type is MarketType.FULL_TOTAL
        and market_input.selected_side == "over"
        and market_input.line is not None
        and market_input.line >= 9.0
        and _full_total_scoring_sides(candidate, principal_paths) != _SCORING_SIDES
    ):
        return PriorityEvidenceAssessment(
            score_cap=_HIGH_TOTAL_TWO_SIDED_OVER_CAP,
            reason_codes=("missing_high_total_two_sided_over_evidence",),
        )

    return PriorityEvidenceAssessment()
