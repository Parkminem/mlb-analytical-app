"""One-way comparison of legacy decisions with shadow forecast outputs."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
import re
from types import MappingProxyType

from analysis.forecasting.engine import (
    ForecastOutput,
    SemanticMarketInput,
    build_semantic_market_inputs,
    resolve_forecast_input,
)
from analysis.forecasting.config import SimulationConfig
from analysis.forecasting.bullpen import (
    BullpenForecast,
    deserialize_bullpen_forecast,
)
from analysis.forecasting.identity import (
    build_forecast_run_id,
    canonical_json,
    capture_payload,
    normalized_input_hash,
)
from analysis.forecasting.offense import (
    OffenseForecast,
    deserialize_offense_forecast,
)
from analysis.forecasting.starter import (
    PitcherChainForecast,
    deserialize_pitcher_chain_forecast,
)
from analysis.forecasting.inputs import (
    MarketType,
    NormalizedGameInput,
    SettlementFormat,
)
from analysis.forecasting.market import (
    MarketEvaluation,
    OutcomeProbabilities,
    evaluate_market_from_score_distributions,
    market_is_push_capable,
)
from analysis.forecasting.simulation import (
    ComponentDistribution,
    DistributionSummary,
    validate_component_distribution_domain,
)
from analysis.forecasting.validation import validate_pregame_input
from analysis.ledger.forecast_codec import (
    compact_forecast_outputs,
    expand_component_distributions,
)
from analysis.ledger.records import (
    CaptureCommand,
    CaptureForecastLink,
    CaptureRecord,
    ForecastCommand,
    ForecastRecord,
)
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review.classification import (
    ContradictionRule,
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
_EVIDENCE_PATH_ORDER = {
    path: index for index, path in enumerate(PriorityEvidencePath)
}
_SHADOW_COMPARISON_FIELDS = frozenset(
    ("authority", "legacy", "forecast_run_id", "market_comparisons")
)
_LEGACY_FIELDS = frozenset(
    (
        "model_version",
        "candidates",
        "final_priority",
        "review_bindings",
        "notes",
    )
)
_CANDIDATE_FIELDS = frozenset(
    ("market", "score", "decision_rating", "evidence_paths")
)
_REVIEW_BINDING_FIELDS = frozenset(
    (
        "market_key",
        "legacy_candidate_index",
        "legacy_market",
        "principal_paths",
    )
)
_PRINCIPAL_PATH_FIELDS = frozenset(
    ("evidence_path", "component_ids", "contradiction_rule")
)
_MARKET_COMPARISON_FIELDS = frozenset(
    (
        "market_key",
        "legacy_candidate_index",
        "legacy_market",
        "legacy_score",
        "legacy_rating",
        "model_win_probability",
        "fair_decimal_price",
        "market_no_vig_probability",
        "edge",
        "principal_paths",
        "review_eligible",
        "review_ineligibility_flags",
    )
)
_SEMANTIC_MARKET_INPUT_FIELDS = frozenset(
    (
        "market_key",
        "market_type",
        "settlement_format",
        "selected_side",
        "line",
        "selected_decimal_price",
        "opposing_decimal_price",
    )
)
_COMPARISON_EVALUATION_FIELDS = (
    ("model_win_probability", "model_conditional_win_probability"),
    ("fair_decimal_price", "fair_decimal_price"),
    ("market_no_vig_probability", "market_no_vig_probability"),
    ("edge", "edge"),
)
_OUTCOME_PROBABILITY_FIELDS = frozenset(("win", "push", "loss"))
_MARKET_EVALUATION_FIELDS = frozenset(
    (
        "probabilities",
        "model_conditional_win_probability",
        "fair_decimal_price",
        "market_no_vig_probability",
        "edge",
        "expected_value",
        "selected_decimal_price",
        "opposing_decimal_price",
        "evaluated",
        "reason",
        "semantic_fingerprint",
    )
)
_FORECAST_OUTPUT_FIELDS = frozenset(
    (
        "forecast_run_id",
        "config_version",
        "simulation_config",
        "simulation_count",
        "component_diagnostics",
        "component_distributions",
        "intervals",
        "market_inputs",
        "markets",
        "shadow_comparison",
    )
)
_DISTRIBUTION_FIELDS = frozenset(("component_id", "values", "summary"))
_DISTRIBUTION_SUMMARY_FIELDS = frozenset(
    ("p05", "p10", "p20", "median", "p80", "p90", "p95")
)
_DIAGNOSTIC_FIELDS = frozenset(
    (
        "away_pitcher_chain",
        "home_pitcher_chain",
        "away_offense",
        "home_offense",
        "away_bullpen",
        "home_bullpen",
    )
)
_BASE_COMPONENT_SUFFIXES = (
    "f5_runs",
    "full_runs",
    "f5_traffic",
    "full_traffic",
    "f5_extra_base_hits",
    "full_extra_base_hits",
    "pitcher_chain_runs_allowed",
    "bullpen_runs_allowed",
    "pitcher_chain_workload_outs",
    "protected_bullpen_close_entry",
)


class _ExpectedActiveCaptureIdUnset:
    __slots__ = ()


_EXPECTED_ACTIVE_CAPTURE_ID_UNSET = _ExpectedActiveCaptureIdUnset()


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
        for binding in self.review_bindings:
            candidate = self.candidates[binding.legacy_candidate_index]
            binding_evidence_paths = frozenset(
                path.evidence_path for path in binding.principal_paths
            )
            if not binding_evidence_paths.issubset(
                candidate.evidence_paths
            ):
                raise ValueError(
                    "review binding paths must be declared in candidate "
                    "evidence_paths"
                )
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
        bindings_by_index = {
            binding.legacy_candidate_index: binding
            for binding in self.legacy.review_bindings
        }
        for comparison in self.market_comparisons:
            binding = bindings_by_index.get(
                comparison.legacy_candidate_index
            )
            flags = comparison.review_ineligibility_flags
            if binding is None:
                if "candidate_market_mismatch" in flags:
                    raise ValueError(
                        "candidate_market_mismatch requires a review binding"
                    )
                if comparison.review_eligible:
                    raise ValueError(
                        "eligible market comparison requires an exact review binding"
                    )
                if "missing_pregame_binding" not in flags:
                    raise ValueError(
                        "comparison without a review binding requires "
                        "missing_pregame_binding"
                    )
                continue
            if "missing_pregame_binding" in flags:
                raise ValueError(
                    "comparison with a review binding cannot claim "
                    "missing_pregame_binding"
                )
            if comparison.market_key != binding.market_key:
                raise ValueError(
                    "market comparison must match its exact review binding"
                )
            expected_candidate_mismatch = (
                comparison.legacy_market != binding.legacy_market
            )
            if (
                "candidate_market_mismatch" in flags
            ) != expected_candidate_mismatch:
                raise ValueError(
                    "candidate_market_mismatch must match review binding "
                    "legacy_market identity"
                )
            if (
                comparison.review_eligible
                and comparison.principal_paths != binding.principal_paths
            ):
                raise ValueError(
                    "eligible market comparison must match its exact review binding"
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
    capture_replayed: bool | None = field(default=None, compare=False)
    forecast_replayed: bool | None = field(default=None, compare=False)
    link_replayed: bool | None = field(default=None, compare=False)

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
        for field_name, value in (
            ("capture_replayed", self.capture_replayed),
            ("forecast_replayed", self.forecast_replayed),
            ("link_replayed", self.link_replayed),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be bool or None")


@dataclass(frozen=True)
class FrozenForecastContext:
    forecast_run_id: str
    config_version: str
    simulation_config: SimulationConfig
    simulation_count: int
    market_inputs: Mapping[str, SemanticMarketInput]
    markets: Mapping[str, MarketEvaluation]
    component_diagnostics: Mapping[str, object]
    component_distributions: Mapping[str, ComponentDistribution]
    intervals: Mapping[str, DistributionSummary]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.forecast_run_id, str)
            or _FORECAST_RUN_ID.fullmatch(self.forecast_run_id) is None
        ):
            raise ValueError(
                "forecast_run_id must be exactly 64 lowercase hex characters"
            )
        _require_nonempty_string(self.config_version, "config_version")
        if not isinstance(self.simulation_config, SimulationConfig):
            raise TypeError("simulation_config must be SimulationConfig")
        if self.config_version != self.simulation_config.version:
            raise ValueError("config_version must match simulation_config")
        if (
            isinstance(self.simulation_count, bool)
            or not isinstance(self.simulation_count, int)
            or self.simulation_count != self.simulation_config.simulations
        ):
            raise ValueError(
                "simulation_count must match simulation_config simulations"
            )


def _payload_mapping(
    value: object,
    field_name: str,
    expected_fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} schema must be a mapping")
    actual_fields = frozenset(value)
    if actual_fields != expected_fields:
        missing = ", ".join(sorted(expected_fields - actual_fields)) or "none"
        additional = ", ".join(sorted(actual_fields - expected_fields)) or "none"
        raise ValueError(
            f"{field_name} schema mismatch; missing={missing}; "
            f"additional={additional}"
        )
    return value


def _payload_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} schema must be a sequence")
    return tuple(value)


def _payload_string_sequence(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    values = _payload_sequence(value, field_name)
    for item in values:
        _require_nonempty_string(item, field_name)
    return values


def _payload_enum(enum_type: type, value: object, field_name: str):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field_name} is unsupported: {value}") from error


def _principal_path_from_payload(value: object) -> PrincipalPathBinding:
    payload = _payload_mapping(
        value,
        "principal path",
        _PRINCIPAL_PATH_FIELDS,
    )
    component_ids = _payload_string_sequence(
        payload["component_ids"],
        "principal path component_ids",
    )
    path = PrincipalPathBinding(
        evidence_path=_payload_enum(
            PriorityEvidencePath,
            payload["evidence_path"],
            "principal path evidence_path",
        ),
        component_ids=component_ids,
        contradiction_rule=_payload_enum(
            ContradictionRule,
            payload["contradiction_rule"],
            "principal path contradiction_rule",
        ),
    )
    if component_ids != path.component_ids:
        raise ValueError(
            "principal path component_ids must use canonical order"
        )
    return path


def _principal_paths_from_payload(
    value: object,
) -> tuple[PrincipalPathBinding, ...]:
    return tuple(
        _principal_path_from_payload(path)
        for path in _payload_sequence(value, "principal_paths")
    )


def _candidate_from_payload(value: object) -> PriorityMarketCandidate:
    payload = _payload_mapping(value, "legacy candidate", _CANDIDATE_FIELDS)
    evidence_values = _payload_string_sequence(
        payload["evidence_paths"],
        "legacy candidate evidence_paths",
    )
    evidence_paths = tuple(
        _payload_enum(
            PriorityEvidencePath,
            evidence_path,
            "legacy candidate evidence_path",
        )
        for evidence_path in evidence_values
    )
    canonical_evidence_values = tuple(
        evidence_path.value
        for evidence_path in PriorityEvidencePath
        if evidence_path in evidence_paths
    )
    if evidence_values != canonical_evidence_values:
        raise ValueError(
            "legacy candidate evidence_paths must use canonical unique order"
        )
    return PriorityMarketCandidate(
        market=payload["market"],
        score=payload["score"],
        decision_rating=payload["decision_rating"],
        evidence_paths=frozenset(evidence_paths),
    )


def _binding_from_payload(value: object) -> CandidateReviewBinding:
    payload = _payload_mapping(
        value,
        "legacy review binding",
        _REVIEW_BINDING_FIELDS,
    )
    return CandidateReviewBinding(
        market_key=payload["market_key"],
        legacy_candidate_index=payload["legacy_candidate_index"],
        legacy_market=payload["legacy_market"],
        principal_paths=_principal_paths_from_payload(
            payload["principal_paths"]
        ),
    )


def _legacy_from_payload(value: object) -> LegacyDecisionSnapshot:
    payload = _payload_mapping(value, "shadow legacy", _LEGACY_FIELDS)
    return LegacyDecisionSnapshot(
        model_version=payload["model_version"],
        candidates=tuple(
            _candidate_from_payload(candidate)
            for candidate in _payload_sequence(
                payload["candidates"],
                "shadow legacy candidates",
            )
        ),
        final_priority=_payload_string_sequence(
            payload["final_priority"],
            "shadow legacy final_priority",
        ),
        review_bindings=tuple(
            _binding_from_payload(binding)
            for binding in _payload_sequence(
                payload["review_bindings"],
                "shadow legacy review_bindings",
            )
        ),
        notes=_payload_string_sequence(
            payload["notes"],
            "shadow legacy notes",
        ),
    )


def _market_comparison_from_payload(value: object) -> ShadowMarketComparison:
    payload = _payload_mapping(
        value,
        "shadow market comparison",
        _MARKET_COMPARISON_FIELDS,
    )
    return ShadowMarketComparison(
        market_key=payload["market_key"],
        legacy_candidate_index=payload["legacy_candidate_index"],
        legacy_market=payload["legacy_market"],
        legacy_score=payload["legacy_score"],
        legacy_rating=payload["legacy_rating"],
        model_win_probability=payload["model_win_probability"],
        fair_decimal_price=payload["fair_decimal_price"],
        market_no_vig_probability=payload["market_no_vig_probability"],
        edge=payload["edge"],
        principal_paths=_principal_paths_from_payload(
            payload["principal_paths"]
        ),
        review_eligible=payload["review_eligible"],
        review_ineligibility_flags=_payload_string_sequence(
            payload["review_ineligibility_flags"],
            "review_ineligibility_flags",
        ),
    )


def deserialize_shadow_comparison(value: object) -> ShadowComparison:
    """Reconstruct and validate one complete frozen shadow payload."""

    payload = _payload_mapping(
        value,
        "shadow_comparison",
        _SHADOW_COMPARISON_FIELDS,
    )
    return ShadowComparison(
        authority=payload["authority"],
        legacy=_legacy_from_payload(payload["legacy"]),
        forecast_run_id=payload["forecast_run_id"],
        market_comparisons=tuple(
            _market_comparison_from_payload(comparison)
            for comparison in _payload_sequence(
                payload["market_comparisons"],
                "shadow market_comparisons",
            )
        ),
    )


def _semantic_market_from_context(
    market_key: str,
    value: object,
) -> SemanticMarketInput:
    if isinstance(value, SemanticMarketInput):
        if value.market_key != market_key:
            raise ValueError(
                "market_inputs key must match semantic market_key"
            )
        return value
    payload = _payload_mapping(
        value,
        f"market_inputs[{market_key}]",
        _SEMANTIC_MARKET_INPUT_FIELDS,
    )
    if payload["market_key"] != market_key:
        raise ValueError(
            "market_inputs key must match semantic market_key"
        )
    return SemanticMarketInput(
        market_key=market_key,
        market_type=_payload_enum(
            MarketType,
            payload["market_type"],
            "market_type",
        ),
        settlement_format=_payload_enum(
            SettlementFormat,
            payload["settlement_format"],
            "settlement_format",
        ),
        selected_side=payload["selected_side"],
        line=payload["line"],
        selected_decimal_price=payload["selected_decimal_price"],
        opposing_decimal_price=payload["opposing_decimal_price"],
    )


def _require_semantic_market_contract(
    market: SemanticMarketInput,
    captured_team_aliases: frozenset[str],
) -> None:
    side_markets = frozenset(
        (MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5)
    )
    total_markets = frozenset((MarketType.F5_TOTAL, MarketType.FULL_TOTAL))
    if market.market_type in side_markets:
        if market.selected_side not in {"away", "home"} | captured_team_aliases:
            raise ValueError(
                "side market selected_side must be away/home or a captured "
                "team alias"
            )
    elif market.market_type in total_markets:
        if market.selected_side not in {"over", "under"}:
            raise ValueError("total market selected_side must be over or under")
    else:
        raise ValueError("market_type is unsupported")
    if market.market_type is MarketType.ML:
        if market.line is not None:
            raise ValueError("ML line must be absent")
    elif not _is_finite_number(market.line):
        raise ValueError(f"{market.market_type.value} line must be finite")
    elif market.market_type in total_markets and market.line < 0:
        raise ValueError("total market line must be nonnegative")
    elif market.market_type is MarketType.PLUS_1_5 and market.line != 1.5:
        raise ValueError("plus_1_5 line must be exactly 1.5")
    if (
        market.market_type is not MarketType.F5_SIDE
        and market.settlement_format is not SettlementFormat.TWO_WAY
    ):
        raise ValueError("settlement_format must be two_way for this market")
    market.semantic_fingerprint


def _market_evaluation_from_context(
    market_key: str,
    value: object,
) -> MarketEvaluation:
    if isinstance(value, MarketEvaluation):
        return value
    payload = _payload_mapping(
        value,
        f"markets[{market_key}]",
        _MARKET_EVALUATION_FIELDS,
    )
    probability_value = payload["probabilities"]
    probabilities = None
    if probability_value is not None:
        probability_payload = _payload_mapping(
            probability_value,
            f"markets[{market_key}] probabilities",
            _OUTCOME_PROBABILITY_FIELDS,
        )
        probabilities = OutcomeProbabilities(
            win=probability_payload["win"],
            push=probability_payload["push"],
            loss=probability_payload["loss"],
        )
    return MarketEvaluation(
        probabilities=probabilities,
        model_conditional_win_probability=(
            payload["model_conditional_win_probability"]
        ),
        fair_decimal_price=payload["fair_decimal_price"],
        market_no_vig_probability=payload["market_no_vig_probability"],
        edge=payload["edge"],
        expected_value=payload["expected_value"],
        selected_decimal_price=payload["selected_decimal_price"],
        opposing_decimal_price=payload["opposing_decimal_price"],
        evaluated=payload["evaluated"],
        reason=payload["reason"],
        semantic_fingerprint=payload["semantic_fingerprint"],
    )


def _validated_market_context(
    market_inputs: Mapping[str, object],
    markets: Mapping[str, object],
    *,
    away_team: str | None = None,
    home_team: str | None = None,
) -> tuple[
    Mapping[str, SemanticMarketInput],
    Mapping[str, MarketEvaluation],
]:
    if set(market_inputs) != set(markets):
        raise ValueError("market_inputs and markets keys must exactly match")
    if (away_team is None) != (home_team is None):
        raise ValueError("away_team and home_team must be provided together")
    captured_team_aliases: frozenset[str] = frozenset()
    if away_team is not None and home_team is not None:
        _require_nonempty_string(away_team, "away_team")
        _require_nonempty_string(home_team, "home_team")
        if away_team == home_team:
            raise ValueError("away_team and home_team must differ")
        captured_team_aliases = frozenset((away_team, home_team))
    parsed_inputs = {}
    parsed_evaluations = {}
    for market_key in sorted(market_inputs):
        _require_nonempty_string(market_key, "market key")
        semantic = _semantic_market_from_context(
            market_key,
            market_inputs[market_key],
        )
        _require_semantic_market_contract(semantic, captured_team_aliases)
        evaluation = _market_evaluation_from_context(
            market_key,
            markets[market_key],
        )
        if evaluation.semantic_fingerprint != semantic.semantic_fingerprint:
            raise ValueError(
                f"markets[{market_key}] semantic_fingerprint must match "
                "frozen market semantics"
            )
        if (
            evaluation.evaluated
            and evaluation.probabilities is not None
            and not market_is_push_capable(
                semantic.market_type,
                semantic.settlement_format,
                semantic.line,
            )
            and evaluation.probabilities.push != 0.0
        ):
            raise ValueError(
                f"markets[{market_key}] non-push-capable semantics cannot "
                "contain push probability"
            )
        unsupported_three_way_f5 = (
            semantic.market_type is MarketType.F5_SIDE
            and semantic.settlement_format is SettlementFormat.THREE_WAY
        )
        if unsupported_three_way_f5:
            if evaluation.evaluated or evaluation.reason != (
                "three-way F5 settlement is unsupported"
            ):
                raise ValueError(
                    "three-way F5 market must use the fixed unevaluated state"
                )
        elif not evaluation.evaluated:
            raise ValueError(
                f"ordinary market {market_key} cannot be unevaluated"
            )
        if evaluation.evaluated:
            for field_name in (
                "selected_decimal_price",
                "opposing_decimal_price",
            ):
                if getattr(evaluation, field_name) != getattr(
                    semantic,
                    field_name,
                ):
                    raise ValueError(
                        f"markets[{market_key}] {field_name} must match "
                        "market_inputs"
                    )
        parsed_inputs[market_key] = semantic
        parsed_evaluations[market_key] = evaluation
    return (
        MappingProxyType(parsed_inputs),
        MappingProxyType(parsed_evaluations),
    )


def _simulation_config_from_payload(value: object) -> SimulationConfig:
    expected_fields = frozenset(field.name for field in fields(SimulationConfig))
    payload = _payload_mapping(
        value,
        "forecast outputs simulation_config",
        expected_fields,
    )
    try:
        config = SimulationConfig.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"simulation_config is invalid: {error}") from error
    if canonical_json(payload) != canonical_json(config.to_mapping()):
        raise ValueError("simulation_config must be canonical")
    return config


def _component_diagnostics_from_payload(
    value: object,
    *,
    away_team: str | None,
    home_team: str | None,
) -> tuple[Mapping[str, object], frozenset[str]]:
    def decode(decoder, diagnostic: object, field_name: str) -> object:
        try:
            return decoder(diagnostic, field_name=field_name)
        except OverflowError as error:
            raise ValueError(
                f"{field_name} contains an overflowing numeric value"
            ) from error

    payload = _payload_mapping(
        value,
        "forecast outputs component_diagnostics",
        _DIAGNOSTIC_FIELDS,
    )
    diagnostics: dict[str, object] = {}
    for side in ("away", "home"):
        diagnostics[f"{side}_pitcher_chain"] = decode(
            deserialize_pitcher_chain_forecast,
            payload[f"{side}_pitcher_chain"],
            f"{side} pitcher-chain diagnostic",
        )
        diagnostics[f"{side}_offense"] = decode(
            deserialize_offense_forecast,
            payload[f"{side}_offense"],
            f"{side} offense diagnostic",
        )
        bullpen = decode(
            deserialize_bullpen_forecast,
            payload[f"{side}_bullpen"],
            f"{side} bullpen diagnostic",
        )
        expected_team = away_team if side == "away" else home_team
        if expected_team is not None and bullpen.team != expected_team:
            raise ValueError(
                f"{side} bullpen diagnostic team must match game identity"
            )
        diagnostics[f"{side}_bullpen"] = bullpen

    away_offense = diagnostics["away_offense"]
    home_offense = diagnostics["home_offense"]
    away_chain = diagnostics["away_pitcher_chain"]
    home_chain = diagnostics["home_pitcher_chain"]
    if (
        not isinstance(away_offense, OffenseForecast)
        or not isinstance(home_offense, OffenseForecast)
        or not isinstance(away_chain, PitcherChainForecast)
        or not isinstance(home_chain, PitcherChainForecast)
    ):
        raise TypeError("decoded component diagnostics have invalid types")
    expected_away_matchups = tuple(
        pitcher.pitcher_id for pitcher in home_chain.pitchers
    )
    expected_home_matchups = tuple(
        pitcher.pitcher_id for pitcher in away_chain.pitchers
    )
    if tuple(away_offense.matchup_multipliers) != expected_away_matchups:
        raise ValueError(
            "away offense diagnostic must match the home pitcher chain"
        )
    if tuple(home_offense.matchup_multipliers) != expected_home_matchups:
        raise ValueError(
            "home offense diagnostic must match the away pitcher chain"
        )

    component_ids = {
        f"{side}_{suffix}"
        for side in ("away", "home")
        for suffix in _BASE_COMPONENT_SUFFIXES
    }
    for side in ("away", "home"):
        bullpen = diagnostics[f"{side}_bullpen"]
        if not isinstance(bullpen, BullpenForecast):
            raise TypeError(f"{side} bullpen diagnostic has invalid type")
        component_ids.update(
            f"{side}_bullpen_segment_{segment.segment_id}_exposed"
            for segment in bullpen.available_segments
        )
    return (
        MappingProxyType(
            {key: diagnostics[key] for key in sorted(diagnostics)}
        ),
        frozenset(component_ids),
    )


def _distribution_summary_from_payload(
    value: object,
    field_name: str,
) -> DistributionSummary:
    payload = _payload_mapping(
        value,
        field_name,
        _DISTRIBUTION_SUMMARY_FIELDS,
    )
    return DistributionSummary(
        **{
            summary_field: payload[summary_field]
            for summary_field in _DISTRIBUTION_SUMMARY_FIELDS
        }
    )


def _component_distributions_from_payload(
    value: object,
    simulation_count: int,
    expected_component_ids: frozenset[str],
) -> Mapping[str, ComponentDistribution]:
    if not isinstance(value, Mapping):
        raise ValueError("forecast component_distributions must be a mapping")
    actual_component_ids = set(value)
    missing_component_ids = expected_component_ids - actual_component_ids
    if missing_component_ids:
        raise ValueError(
            "component distributions are missing diagnostic component IDs "
            "(missing_component_distribution): "
            + ", ".join(sorted(missing_component_ids))
        )
    unexpected_component_ids = actual_component_ids - expected_component_ids
    if unexpected_component_ids:
        raise ValueError(
            "component distributions contain unexpected diagnostic component "
            "IDs: "
            + ", ".join(sorted(unexpected_component_ids))
        )
    distributions = {}
    for component_id in sorted(value):
        _require_nonempty_string(component_id, "component distribution key")
        payload = _payload_mapping(
            value[component_id],
            f"component distribution {component_id}",
            _DISTRIBUTION_FIELDS,
        )
        if payload["component_id"] != component_id:
            raise ValueError(
                f"component distribution key must match component_id {component_id}"
            )
        values = _payload_sequence(
            payload["values"],
            f"component distribution {component_id} values",
        )
        if len(values) != simulation_count:
            raise ValueError(
                f"component distribution {component_id} values must match "
                "simulation_count"
            )
        distribution = ComponentDistribution(
            component_id=component_id,
            values=values,
            summary=_distribution_summary_from_payload(
                payload["summary"],
                f"component distribution {component_id} summary",
            ),
        )
        validate_component_distribution_domain(distribution)
        distributions[component_id] = distribution
    return MappingProxyType(distributions)


def _intervals_from_payload(
    value: object,
    distributions: Mapping[str, ComponentDistribution],
) -> Mapping[str, DistributionSummary]:
    if not isinstance(value, Mapping):
        raise ValueError("forecast intervals must be a mapping")
    if set(value) != set(distributions):
        raise ValueError(
            "interval keys must exactly match component distributions"
        )
    intervals = {}
    for component_id in sorted(value):
        interval = _distribution_summary_from_payload(
            value[component_id],
            f"interval {component_id}",
        )
        if interval != distributions[component_id].summary:
            raise ValueError(
                f"interval {component_id} must match distribution summary"
            )
        intervals[component_id] = interval
    return MappingProxyType(intervals)


def deserialize_forecast_context(
    value: object,
    *,
    away_team: str | None = None,
    home_team: str | None = None,
) -> FrozenForecastContext:
    """Reconstruct the complete immutable review-facing forecast context."""

    payload = _payload_mapping(
        value,
        "forecast outputs",
        _FORECAST_OUTPUT_FIELDS,
    )
    config = _simulation_config_from_payload(payload["simulation_config"])
    simulation_count = payload["simulation_count"]
    if (
        isinstance(simulation_count, bool)
        or not isinstance(simulation_count, int)
        or simulation_count <= 0
    ):
        raise ValueError("simulation_count must be a positive integer")
    market_inputs_value = payload["market_inputs"]
    markets_value = payload["markets"]
    if not isinstance(market_inputs_value, Mapping):
        raise ValueError("forecast market_inputs must be a mapping")
    if not isinstance(markets_value, Mapping):
        raise ValueError("forecast markets must be a mapping")
    market_inputs, markets = _validated_market_context(
        market_inputs_value,
        markets_value,
        away_team=away_team,
        home_team=home_team,
    )
    component_diagnostics, component_ids = (
        _component_diagnostics_from_payload(
            payload["component_diagnostics"],
            away_team=away_team,
            home_team=home_team,
        )
    )
    distributions = _component_distributions_from_payload(
        expand_component_distributions(payload["component_distributions"]),
        simulation_count,
        component_ids,
    )
    intervals = _intervals_from_payload(
        payload["intervals"],
        distributions,
    )
    return FrozenForecastContext(
        forecast_run_id=payload["forecast_run_id"],
        config_version=payload["config_version"],
        simulation_config=config,
        simulation_count=simulation_count,
        market_inputs=market_inputs,
        markets=markets,
        component_diagnostics=component_diagnostics,
        component_distributions=distributions,
        intervals=intervals,
    )


def validate_frozen_market_output_binding(
    context: FrozenForecastContext,
    *,
    away_team: str | None = None,
    home_team: str | None = None,
) -> None:
    """Bind stored market outputs to the exact frozen score distributions."""

    if not isinstance(context, FrozenForecastContext):
        raise TypeError("context must be FrozenForecastContext")
    if (away_team is None) != (home_team is None):
        raise ValueError("away_team and home_team must be provided together")
    score_component_ids = (
        "away_f5_runs",
        "home_f5_runs",
        "away_full_runs",
        "home_full_runs",
    )
    try:
        score_distributions = {
            component_id: context.component_distributions[
                component_id
            ].values
            for component_id in score_component_ids
        }
    except KeyError as error:
        raise ValueError(
            "frozen score distributions are incomplete"
        ) from error
    reconstructed_markets = {}
    for market_key, market_input in context.market_inputs.items():
        resolved_market_input = market_input
        if (
            market_input.market_type
            in (MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5)
            and market_input.selected_side not in {"away", "home"}
        ):
            if away_team is None or home_team is None:
                raise ValueError(
                    "captured team aliases require away_team and home_team"
                )
            selected_side = {
                away_team: "away",
                home_team: "home",
            }.get(market_input.selected_side)
            if selected_side is None:
                raise ValueError(
                    "side market selected_side is not a captured team alias"
                )
            resolved_market_input = replace(
                market_input,
                selected_side=selected_side,
            )
        reconstructed = evaluate_market_from_score_distributions(
            resolved_market_input,
            **score_distributions,
        )
        reconstructed_markets[market_key] = replace(
            reconstructed,
            semantic_fingerprint=(
                context.markets[market_key].semantic_fingerprint
            ),
        )
    if canonical_json(reconstructed_markets) != canonical_json(
        context.markets
    ):
        raise ValueError(
            "frozen market outputs must match score distributions"
        )


def _evaluation_fingerprint(market_key: str, value: object) -> str:
    fingerprint = (
        value.get("semantic_fingerprint")
        if isinstance(value, Mapping)
        else getattr(value, "semantic_fingerprint", None)
    )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            f"markets[{market_key}] semantic_fingerprint is missing"
        )
    return fingerprint


def _evaluation_field(
    market_key: str,
    evaluation: object,
    field_name: str,
) -> object:
    if isinstance(evaluation, Mapping):
        if field_name not in evaluation:
            raise ValueError(
                f"markets[{market_key}] is missing {field_name}"
            )
        return evaluation[field_name]
    if not hasattr(evaluation, field_name):
        raise ValueError(
            f"markets[{market_key}] is missing {field_name}"
        )
    return getattr(evaluation, field_name)


def _require_comparison_evaluation_match(
    comparison: ShadowMarketComparison,
    evaluation: object | None,
) -> None:
    for comparison_field, evaluation_field in _COMPARISON_EVALUATION_FIELDS:
        expected = (
            None
            if evaluation is None
            else _evaluation_field(
                comparison.market_key,
                evaluation,
                evaluation_field,
            )
        )
        if getattr(comparison, comparison_field) != expected:
            raise ValueError(
                f"shadow market evaluation {comparison_field} must match "
                f"markets[{comparison.market_key}]"
            )


def _binding_context_flags(
    binding: CandidateReviewBinding,
    market_inputs: Mapping[str, object],
    markets: Mapping[str, object],
    component_distributions: Mapping[str, object],
    parsed_market_inputs: dict[str, SemanticMarketInput],
) -> tuple[str, ...]:
    flags: set[str] = set()
    market_key = binding.market_key
    for path in binding.principal_paths:
        for component_id in path.component_ids:
            if not component_id.startswith(("away_", "home_")):
                raise ValueError(
                    "component IDs must use the captured away/home namespace; "
                    "team-abbreviation component IDs are unsupported"
                )
    semantic_market = None
    if market_key in market_inputs:
        semantic_market = _semantic_market_from_context(
            market_key,
            market_inputs[market_key],
        )
        parsed_market_inputs[market_key] = semantic_market
    if semantic_market is None or market_key not in markets:
        flags.add("unknown_market_key")
    elif (
        _evaluation_fingerprint(market_key, markets[market_key])
        != semantic_market.semantic_fingerprint
    ):
        raise ValueError(
            f"markets[{market_key}] semantic_fingerprint must match "
            "frozen market semantics"
        )

    invalid_paths = not _paths_have_valid_count_and_identity(
        binding.principal_paths
    )
    if semantic_market is not None:
        try:
            validate_principal_path_bindings(
                semantic_market.market_type,
                binding.principal_paths,
            )
        except ValueError:
            invalid_paths = True
    if invalid_paths:
        flags.add("invalid_path_count")
    if any(
        component_id not in component_distributions
        for path in binding.principal_paths
        for component_id in path.component_ids
    ):
        flags.add("missing_component_distribution")
    return tuple(
        flag for flag in _INELIGIBILITY_FLAG_ORDER if flag in flags
    )


def validate_shadow_forecast_context(
    comparison: ShadowComparison,
    market_inputs: object,
    markets: object,
    component_distributions: object,
    *,
    away_team: str | None = None,
    home_team: str | None = None,
) -> Mapping[str, SemanticMarketInput]:
    """Validate every shadow comparison against one frozen forecast output."""

    if not isinstance(comparison, ShadowComparison):
        raise TypeError("comparison must be ShadowComparison")
    if not isinstance(market_inputs, Mapping):
        raise ValueError("forecast market_inputs context must be a mapping")
    if not isinstance(markets, Mapping):
        raise ValueError("forecast markets context must be a mapping")
    if not isinstance(component_distributions, Mapping):
        raise ValueError(
            "forecast component_distributions context must be a mapping"
        )

    bindings_by_index = {
        binding.legacy_candidate_index: binding
        for binding in comparison.legacy.review_bindings
    }
    parsed_inputs, parsed_evaluations = _validated_market_context(
        market_inputs,
        markets,
        away_team=away_team,
        home_team=home_team,
    )
    parsed_market_inputs = dict(parsed_inputs)
    for market_comparison in comparison.market_comparisons:
        binding = bindings_by_index.get(
            market_comparison.legacy_candidate_index
        )
        if binding is None:
            expected_flags = ("missing_pregame_binding",)
        else:
            context_flags = set(
                _binding_context_flags(
                    binding,
                    parsed_inputs,
                    parsed_evaluations,
                    component_distributions,
                    parsed_market_inputs,
                )
            )
            if binding.legacy_market != market_comparison.legacy_market:
                context_flags.add("candidate_market_mismatch")
            expected_flags = tuple(
                flag
                for flag in _INELIGIBILITY_FLAG_ORDER
                if flag in context_flags
            )
        if market_comparison.review_ineligibility_flags != expected_flags:
            expected = ", ".join(expected_flags) or "none"
            stored = (
                ", ".join(
                    market_comparison.review_ineligibility_flags
                )
                or "none"
            )
            raise ValueError(
                "inconsistent shadow comparison flags for "
                f"{market_comparison.market_key}: expected={expected}; "
                f"stored={stored}"
            )
        evaluation = (
            None
            if binding is None
            else parsed_evaluations.get(binding.market_key)
        )
        _require_comparison_evaluation_match(
            market_comparison,
            evaluation,
        )
    return MappingProxyType(
        {
            key: parsed_market_inputs[key]
            for key in sorted(parsed_market_inputs)
        }
    )


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


def _principal_path_sort_key(
    path: PrincipalPathBinding,
) -> tuple[int, str, tuple[str, ...]]:
    return (
        _EVIDENCE_PATH_ORDER[path.evidence_path],
        path.contradiction_rule.value,
        path.component_ids,
    )


def _principal_paths_payload(
    paths: tuple[PrincipalPathBinding, ...],
) -> list[dict[str, object]]:
    return [
        _principal_path_payload(path)
        for path in sorted(paths, key=_principal_path_sort_key)
    ]


def _binding_payload(binding: CandidateReviewBinding) -> dict[str, object]:
    return {
        "market_key": binding.market_key,
        "legacy_candidate_index": binding.legacy_candidate_index,
        "legacy_market": binding.legacy_market,
        "principal_paths": _principal_paths_payload(binding.principal_paths),
    }


def canonical_legacy_decision_payload(
    legacy: LegacyDecisionSnapshot,
) -> dict[str, object]:
    """Serialize legacy content in the exact canonical persistence order."""

    if not isinstance(legacy, LegacyDecisionSnapshot):
        raise TypeError("legacy must be LegacyDecisionSnapshot")
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
            _binding_payload(binding)
            for binding in sorted(
                legacy.review_bindings,
                key=lambda binding: binding.legacy_candidate_index,
            )
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
        "principal_paths": _principal_paths_payload(
            comparison.principal_paths
        ),
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
        "legacy": canonical_legacy_decision_payload(comparison.legacy),
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
    outputs = compact_forecast_outputs(outputs)
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


def build_shadow_capture_command(
    game_input: NormalizedGameInput,
    supersedes_capture_id: str | None = None,
) -> CaptureCommand:
    """Build the validated, audit-complete command for a shadow capture."""

    if not isinstance(game_input, NormalizedGameInput):
        raise TypeError("game_input must be NormalizedGameInput")
    validation = validate_pregame_input(game_input)
    resolved_game = resolve_forecast_input(game_input)
    return CaptureCommand(
        game_pk=game_input.identity.game_pk,
        captured_at=game_input.captured_at,
        first_pitch=game_input.identity.first_pitch,
        lineup_state=_capture_lineup_state(game_input),
        out_of_sample_eligible=validation.out_of_sample_eligible,
        normalized_input_hash=normalized_input_hash(resolved_game),
        payload=capture_payload(game_input),
        supersedes_capture_id=supersedes_capture_id,
    )


def _require_legacy_store_interface(store: object) -> None:
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
    *,
    supersedes_capture_id: str | None = None,
    expected_active_capture_id: (
        str | None | _ExpectedActiveCaptureIdUnset
    ) = _EXPECTED_ACTIVE_CAPTURE_ID_UNSET,
) -> ShadowRunRecords:
    """Append or replay capture, capture-independent forecast, and link."""

    atomic_append = getattr(store, "append_shadow_run_atomic", None)
    if not callable(atomic_append):
        _require_legacy_store_interface(store)
    if not isinstance(game_input, NormalizedGameInput):
        raise TypeError("game_input must be NormalizedGameInput")
    if not isinstance(forecast, ForecastOutput):
        raise TypeError("forecast must be ForecastOutput")
    if not isinstance(idempotency_keys, ShadowRunIdempotencyKeys):
        raise TypeError(
            "idempotency_keys must be ShadowRunIdempotencyKeys"
        )
    capture_command = build_shadow_capture_command(
        game_input,
        supersedes_capture_id=supersedes_capture_id,
    )
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
    input_hash = capture_command.normalized_input_hash
    forecast_command = ForecastCommand(
        forecast_run_id=forecast.forecast_run_id,
        model_version=forecast.model_version,
        simulation_config_hash=_simulation_config_hash(forecast),
        normalized_input_hash=input_hash,
        outputs=_forecast_outputs(forecast, comparison),
    )

    if (
        expected_active_capture_id is not _EXPECTED_ACTIVE_CAPTURE_ID_UNSET
        and callable(atomic_append)
    ):
        return atomic_append(
            idempotency_keys.capture,
            capture_command,
            forecast_command,
            expected_active_capture_id,
        )

    _require_legacy_store_interface(store)
    if expected_active_capture_id is _EXPECTED_ACTIVE_CAPTURE_ID_UNSET:
        capture = store.append_capture(idempotency_keys.capture, capture_command)
    else:
        capture = store.append_capture(
            idempotency_keys.capture,
            capture_command,
            expected_active_capture_id=expected_active_capture_id,
        )
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
    "FrozenForecastContext",
    "LegacyDecisionSnapshot",
    "ShadowComparison",
    "ShadowMarketComparison",
    "ShadowRunIdempotencyKeys",
    "ShadowRunRecords",
    "build_shadow_capture_command",
    "canonical_legacy_decision_payload",
    "compare_shadow_forecast",
    "deserialize_forecast_context",
    "deserialize_shadow_comparison",
    "record_shadow_run",
    "validate_frozen_market_output_binding",
    "validate_shadow_forecast_context",
)
