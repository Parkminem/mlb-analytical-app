"""One-way comparison of legacy decisions with shadow forecast outputs."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

from analysis.forecasting.engine import (
    ForecastOutput,
    SemanticMarketInput,
    build_semantic_market_inputs,
    resolve_forecast_input,
)
from analysis.forecasting.identity import (
    build_forecast_run_id,
    canonical_json,
    capture_payload,
    normalized_input_hash,
)
from analysis.forecasting.inputs import NormalizedGameInput
from analysis.forecasting.validation import validate_pregame_input
from analysis.ledger.records import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review.classification import (
    PrincipalPathBinding,
    validate_principal_path_bindings,
)


_FORECAST_RUN_ID = re.compile(r"[0-9a-f]{64}\Z")
_INELIGIBILITY_FLAG_ORDER = (
    "missing_pregame_binding",
    "candidate_market_mismatch",
    "unknown_market_key",
    "invalid_path_count",
    "missing_component_distribution",
)
_INELIGIBILITY_FLAGS = frozenset(_INELIGIBILITY_FLAG_ORDER)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_nonnegative_index(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    for item in value:
        _require_nonempty_string(item, field_name)


def _require_principal_path_tuple(value: object) -> None:
    if not isinstance(value, tuple):
        raise ValueError("principal_paths must be a tuple")
    if any(not isinstance(path, PrincipalPathBinding) for path in value):
        raise ValueError(
            "principal_paths must contain PrincipalPathBinding values"
        )


def _require_optional_probability(value: object, field_name: str) -> None:
    if value is None:
        return
    if not _is_finite_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability in [0, 1]")


def _require_optional_finite(value: object, field_name: str) -> None:
    if value is not None and not _is_finite_number(value):
        raise ValueError(f"{field_name} must be finite when present")


def _paths_have_valid_count_and_identity(
    principal_paths: tuple[PrincipalPathBinding, ...],
) -> bool:
    if not 2 <= len(principal_paths) <= 4:
        return False
    evidence_paths = tuple(path.evidence_path for path in principal_paths)
    return (
        all(isinstance(path, PriorityEvidencePath) for path in evidence_paths)
        and len(set(evidence_paths)) == len(evidence_paths)
    )


@dataclass(frozen=True)
class CandidateReviewBinding:
    market_key: str
    legacy_candidate_index: int
    legacy_market: str
    principal_paths: tuple[PrincipalPathBinding, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.market_key, "market_key")
        _require_nonnegative_index(
            self.legacy_candidate_index,
            "legacy_candidate_index",
        )
        _require_nonempty_string(self.legacy_market, "legacy_market")
        _require_principal_path_tuple(self.principal_paths)


@dataclass(frozen=True)
class LegacyDecisionSnapshot:
    model_version: str
    candidates: tuple[PriorityMarketCandidate, ...]
    final_priority: tuple[str, ...]
    review_bindings: tuple[CandidateReviewBinding, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.model_version, "model_version")
        if not isinstance(self.candidates, tuple):
            raise ValueError("candidates must be a tuple")
        if any(
            not isinstance(candidate, PriorityMarketCandidate)
            for candidate in self.candidates
        ):
            raise ValueError(
                "candidates must contain PriorityMarketCandidate values"
            )
        for candidate in self.candidates:
            _require_nonempty_string(candidate.market, "candidate market")
            if isinstance(candidate.score, bool) or not isinstance(
                candidate.score,
                int,
            ):
                raise ValueError("candidate score must be an integer")
            _require_nonempty_string(
                candidate.decision_rating,
                "candidate decision_rating",
            )
            if not isinstance(candidate.evidence_paths, frozenset):
                raise ValueError("candidate evidence_paths must be a frozenset")
            if any(
                not isinstance(path, PriorityEvidencePath)
                for path in candidate.evidence_paths
            ):
                raise ValueError(
                    "candidate evidence_paths must use canonical "
                    "PriorityEvidencePath values"
                )

        _require_string_tuple(self.final_priority, "final_priority")
        if not isinstance(self.review_bindings, tuple):
            raise ValueError("review_bindings must be a tuple")
        if any(
            not isinstance(binding, CandidateReviewBinding)
            for binding in self.review_bindings
        ):
            raise ValueError(
                "review_bindings must contain CandidateReviewBinding values"
            )
        indices = tuple(
            binding.legacy_candidate_index for binding in self.review_bindings
        )
        if len(set(indices)) != len(indices):
            raise ValueError(
                "review_bindings must contain a unique candidate index"
            )
        if any(index >= len(self.candidates) for index in indices):
            raise ValueError("review binding candidate index must be in range")
        _require_string_tuple(self.notes, "notes")


@dataclass(frozen=True)
class ShadowMarketComparison:
    market_key: str
    legacy_candidate_index: int
    legacy_market: str
    legacy_score: int
    legacy_rating: str
    model_win_probability: float | None
    fair_decimal_price: float | None
    market_no_vig_probability: float | None
    edge: float | None
    principal_paths: tuple[PrincipalPathBinding, ...]
    review_eligible: bool
    review_ineligibility_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.market_key, "market_key")
        _require_nonnegative_index(
            self.legacy_candidate_index,
            "legacy_candidate_index",
        )
        _require_nonempty_string(self.legacy_market, "legacy_market")
        if isinstance(self.legacy_score, bool) or not isinstance(
            self.legacy_score,
            int,
        ):
            raise ValueError("legacy_score must be an integer")
        _require_nonempty_string(self.legacy_rating, "legacy_rating")
        _require_optional_probability(
            self.model_win_probability,
            "model_win_probability",
        )
        _require_optional_finite(self.fair_decimal_price, "fair_decimal_price")
        if (
            self.fair_decimal_price is not None
            and self.fair_decimal_price < 1.0
        ):
            raise ValueError("fair_decimal_price must be at least one")
        _require_optional_probability(
            self.market_no_vig_probability,
            "market_no_vig_probability",
        )
        _require_optional_finite(self.edge, "edge")
        if self.edge is not None and not -1.0 <= self.edge <= 1.0:
            raise ValueError("edge must be within [-1, 1]")
        _require_principal_path_tuple(self.principal_paths)
        if not isinstance(self.review_eligible, bool):
            raise ValueError("review_eligible must be a boolean")
        _require_string_tuple(
            self.review_ineligibility_flags,
            "review_ineligibility_flags",
        )

        flags = self.review_ineligibility_flags
        if self.review_eligible and flags:
            raise ValueError(
                "eligible comparisons cannot contain ineligibility flags"
            )
        if not self.review_eligible and not flags:
            raise ValueError("ineligible comparisons require at least one flag")
        if any(flag not in _INELIGIBILITY_FLAGS for flag in flags):
            raise ValueError(
                "review_ineligibility_flags must contain only fixed flags"
            )
        canonical_flags = tuple(
            flag for flag in _INELIGIBILITY_FLAG_ORDER if flag in flags
        )
        if flags != canonical_flags:
            raise ValueError(
                "review_ineligibility_flags must use fixed flag order without "
                "duplicates"
            )
        if self.review_eligible:
            if not _paths_have_valid_count_and_identity(self.principal_paths):
                raise ValueError(
                    "eligible comparisons require two to four unique canonical "
                    "principal paths"
                )
            if self.model_win_probability is None:
                raise ValueError(
                    "eligible comparisons require model_win_probability"
                )
        elif self.principal_paths:
            raise ValueError(
                "ineligible comparisons cannot retain principal paths"
            )


@dataclass(frozen=True)
class ShadowComparison:
    authority: str
    legacy: LegacyDecisionSnapshot
    forecast_run_id: str
    market_comparisons: tuple[ShadowMarketComparison, ...]

    def __post_init__(self) -> None:
        if self.authority != "legacy":
            raise ValueError("authority must always be legacy")
        if not isinstance(self.legacy, LegacyDecisionSnapshot):
            raise ValueError("legacy must be LegacyDecisionSnapshot")
        if (
            not isinstance(self.forecast_run_id, str)
            or _FORECAST_RUN_ID.fullmatch(self.forecast_run_id) is None
        ):
            raise ValueError(
                "forecast_run_id must be exactly 64 lowercase hex characters"
            )
        if not isinstance(self.market_comparisons, tuple) or any(
            not isinstance(comparison, ShadowMarketComparison)
            for comparison in self.market_comparisons
        ):
            raise ValueError(
                "market_comparisons must be a tuple of ShadowMarketComparison "
                "values"
            )
        if len(self.market_comparisons) != len(self.legacy.candidates):
            raise ValueError(
                "market_comparisons must contain one entry per legacy candidate"
            )
        for index, (candidate, comparison) in enumerate(
            zip(self.legacy.candidates, self.market_comparisons)
        ):
            if comparison.legacy_candidate_index != index:
                raise ValueError(
                    "market_comparisons must preserve legacy candidate order"
                )
            if (
                comparison.legacy_market != candidate.market
                or comparison.legacy_score != candidate.score
                or comparison.legacy_rating != candidate.decision_rating
            ):
                raise ValueError(
                    "market comparison must copy its legacy candidate fields"
                )


@dataclass(frozen=True)
class ShadowRunIdempotencyKeys:
    capture: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.capture, "capture idempotency key")


@dataclass(frozen=True)
class ShadowRunRecords:
    capture: CaptureRecord
    forecast: ForecastRecord
    link: CaptureForecastLink

    def __post_init__(self) -> None:
        if not isinstance(self.capture, CaptureRecord):
            raise ValueError("capture must be CaptureRecord")
        if not isinstance(self.forecast, ForecastRecord):
            raise ValueError("forecast must be ForecastRecord")
        if not isinstance(self.link, CaptureForecastLink):
            raise ValueError("link must be CaptureForecastLink")
        if self.link.capture_id != self.capture.capture_id:
            raise ValueError("link capture_id must match capture")
        if self.link.forecast_run_id != self.forecast.forecast_run_id:
            raise ValueError("link forecast_run_id must match forecast")


def _comparison_for_candidate(
    index: int,
    candidate: PriorityMarketCandidate,
    binding: CandidateReviewBinding | None,
    forecast: ForecastOutput,
    market_inputs: Mapping[str, SemanticMarketInput],
) -> ShadowMarketComparison:
    flags: set[str] = set()
    if binding is None:
        market_key = candidate.market
        principal_paths: tuple[PrincipalPathBinding, ...] = ()
        market_evaluation = None
        flags.add("missing_pregame_binding")
    else:
        market_key = binding.market_key
        principal_paths = binding.principal_paths
        if binding.legacy_market != candidate.market:
            flags.add("candidate_market_mismatch")
        market_evaluation = forecast.markets.get(market_key)
        market_input = market_inputs.get(market_key)
        if market_evaluation is None or market_input is None:
            flags.add("unknown_market_key")
        if not _paths_have_valid_count_and_identity(principal_paths):
            flags.add("invalid_path_count")
        if market_input is not None:
            try:
                validate_principal_path_bindings(
                    market_input.market_type,
                    principal_paths,
                )
            except ValueError:
                flags.add("invalid_path_count")
        if any(
            component_id not in forecast.component_distributions
            for path in principal_paths
            for component_id in path.component_ids
        ):
            flags.add("missing_component_distribution")

    review_eligible = not flags
    if review_eligible and (
        market_evaluation is None
        or market_evaluation.model_conditional_win_probability is None
    ):
        raise ValueError(
            f"bound forecast market {market_key!r} has no model win probability"
        )
    ordered_flags = tuple(
        flag for flag in _INELIGIBILITY_FLAG_ORDER if flag in flags
    )
    return ShadowMarketComparison(
        market_key=market_key,
        legacy_candidate_index=index,
        legacy_market=candidate.market,
        legacy_score=candidate.score,
        legacy_rating=candidate.decision_rating,
        model_win_probability=(
            None
            if market_evaluation is None
            else market_evaluation.model_conditional_win_probability
        ),
        fair_decimal_price=(
            None
            if market_evaluation is None
            else market_evaluation.fair_decimal_price
        ),
        market_no_vig_probability=(
            None
            if market_evaluation is None
            else market_evaluation.market_no_vig_probability
        ),
        edge=None if market_evaluation is None else market_evaluation.edge,
        principal_paths=principal_paths if review_eligible else (),
        review_eligible=review_eligible,
        review_ineligibility_flags=ordered_flags,
    )


def compare_shadow_forecast(
    legacy: LegacyDecisionSnapshot,
    forecast: ForecastOutput,
) -> ShadowComparison:
    """Copy forecast metrics beside legacy decisions without changing authority."""

    if not isinstance(legacy, LegacyDecisionSnapshot):
        raise TypeError("legacy must be LegacyDecisionSnapshot")
    if not isinstance(forecast, ForecastOutput):
        raise TypeError("forecast must be ForecastOutput")
    bindings_by_index = {
        binding.legacy_candidate_index: binding
        for binding in legacy.review_bindings
    }
    market_comparisons = tuple(
        _comparison_for_candidate(
            index,
            candidate,
            bindings_by_index.get(index),
            forecast,
            forecast.market_inputs,
        )
        for index, candidate in enumerate(legacy.candidates)
    )
    return ShadowComparison(
        authority="legacy",
        legacy=legacy,
        forecast_run_id=forecast.forecast_run_id,
        market_comparisons=market_comparisons,
    )


def _principal_path_payload(path: PrincipalPathBinding) -> dict[str, object]:
    return {
        "evidence_path": path.evidence_path.value,
        "component_ids": list(path.component_ids),
        "contradiction_rule": path.contradiction_rule.value,
    }


def _binding_payload(binding: CandidateReviewBinding) -> dict[str, object]:
    return {
        "market_key": binding.market_key,
        "legacy_candidate_index": binding.legacy_candidate_index,
        "legacy_market": binding.legacy_market,
        "principal_paths": [
            _principal_path_payload(path) for path in binding.principal_paths
        ],
    }


def _legacy_payload(legacy: LegacyDecisionSnapshot) -> dict[str, object]:
    return {
        "model_version": legacy.model_version,
        "candidates": [
            {
                "market": candidate.market,
                "score": candidate.score,
                "decision_rating": candidate.decision_rating,
                "evidence_paths": [
                    evidence_path.value
                    for evidence_path in PriorityEvidencePath
                    if evidence_path in candidate.evidence_paths
                ],
            }
            for candidate in legacy.candidates
        ],
        "final_priority": list(legacy.final_priority),
        "review_bindings": [
            _binding_payload(binding) for binding in legacy.review_bindings
        ],
        "notes": list(legacy.notes),
    }


def _market_comparison_payload(
    comparison: ShadowMarketComparison,
) -> dict[str, object]:
    return {
        "market_key": comparison.market_key,
        "legacy_candidate_index": comparison.legacy_candidate_index,
        "legacy_market": comparison.legacy_market,
        "legacy_score": comparison.legacy_score,
        "legacy_rating": comparison.legacy_rating,
        "model_win_probability": comparison.model_win_probability,
        "fair_decimal_price": comparison.fair_decimal_price,
        "market_no_vig_probability": comparison.market_no_vig_probability,
        "edge": comparison.edge,
        "principal_paths": [
            _principal_path_payload(path)
            for path in comparison.principal_paths
        ],
        "review_eligible": comparison.review_eligible,
        "review_ineligibility_flags": list(
            comparison.review_ineligibility_flags
        ),
    }


def _shadow_comparison_payload(
    comparison: ShadowComparison,
) -> dict[str, object]:
    return {
        "authority": comparison.authority,
        "legacy": _legacy_payload(comparison.legacy),
        "forecast_run_id": comparison.forecast_run_id,
        "market_comparisons": [
            _market_comparison_payload(market)
            for market in comparison.market_comparisons
        ],
    }


def _forecast_outputs(
    forecast: ForecastOutput,
    comparison: ShadowComparison,
) -> dict[str, object]:
    outputs = json.loads(canonical_json(forecast))
    if not isinstance(outputs, dict):
        raise TypeError("forecast output must serialize to a mapping")
    outputs["shadow_comparison"] = _shadow_comparison_payload(comparison)
    return outputs


def _simulation_config_hash(forecast: ForecastOutput) -> str:
    encoded = canonical_json(forecast.simulation_config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_lineup_state(game_input: NormalizedGameInput) -> str:
    lineup_states = (
        game_input.away_lineup.state.value,
        game_input.home_lineup.state.value,
    )
    if lineup_states == ("confirmed", "confirmed"):
        return "confirmed"
    return "projected"


def _require_store_interface(store: object) -> None:
    for method_name in (
        "append_capture",
        "append_forecast",
        "link_capture_forecast",
    ):
        if not callable(getattr(store, method_name, None)):
            raise TypeError(
                "store must provide append_capture, append_forecast, and "
                "link_capture_forecast"
            )


def record_shadow_run(
    store: object,
    game_input: NormalizedGameInput,
    legacy: LegacyDecisionSnapshot,
    forecast: ForecastOutput,
    idempotency_keys: ShadowRunIdempotencyKeys,
) -> ShadowRunRecords:
    """Append or replay capture, capture-independent forecast, and link."""

    _require_store_interface(store)
    if not isinstance(game_input, NormalizedGameInput):
        raise TypeError("game_input must be NormalizedGameInput")
    if not isinstance(idempotency_keys, ShadowRunIdempotencyKeys):
        raise TypeError(
            "idempotency_keys must be ShadowRunIdempotencyKeys"
        )
    validation = validate_pregame_input(game_input)
    resolved_game = resolve_forecast_input(game_input)
    expected_forecast_run_id = build_forecast_run_id(
        resolved_game,
        forecast.model_version,
        forecast.simulation_config,
    )
    if forecast.forecast_run_id != expected_forecast_run_id:
        raise ValueError(
            "forecast_run_id does not match canonical game_input, "
            "model_version, and simulation_config"
        )
    expected_market_inputs = build_semantic_market_inputs(
        resolved_game.markets
    )
    if forecast.market_inputs != expected_market_inputs:
        raise ValueError(
            "forecast market_inputs do not match canonical game_input markets"
        )
    comparison = compare_shadow_forecast(legacy, forecast)
    input_hash = normalized_input_hash(resolved_game)
    capture_command = CaptureCommand(
        game_pk=game_input.identity.game_pk,
        captured_at=game_input.captured_at,
        first_pitch=game_input.identity.first_pitch,
        lineup_state=_capture_lineup_state(game_input),
        out_of_sample_eligible=validation.out_of_sample_eligible,
        normalized_input_hash=input_hash,
        payload=capture_payload(game_input),
    )
    forecast_command = ForecastCommand(
        forecast_run_id=forecast.forecast_run_id,
        model_version=forecast.model_version,
        simulation_config_hash=_simulation_config_hash(forecast),
        normalized_input_hash=input_hash,
        outputs=_forecast_outputs(forecast, comparison),
    )

    capture = store.append_capture(idempotency_keys.capture, capture_command)
    forecast_record = store.append_forecast(forecast_command)
    link = store.link_capture_forecast(
        capture.capture_id,
        forecast_record.forecast_run_id,
    )
    return ShadowRunRecords(
        capture=capture,
        forecast=forecast_record,
        link=link,
    )


__all__ = (
    "CandidateReviewBinding",
    "LegacyDecisionSnapshot",
    "ShadowComparison",
    "ShadowMarketComparison",
    "ShadowRunIdempotencyKeys",
    "ShadowRunRecords",
    "compare_shadow_forecast",
    "record_shadow_run",
)
