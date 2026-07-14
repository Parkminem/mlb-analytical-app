"""Deterministic postgame game-flow classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from analysis.forecasting.inputs import MarketType
from analysis.model import PriorityEvidencePath


_DIRECTIONS = frozenset(("high_is_supportive", "low_is_supportive"))
_MARKET_RESULTS = frozenset(("win", "loss", "push"))
_CLASSIFICATIONS = frozenset(("A", "B", "C", "D"))
_EXOGENOUS_EVENT_TYPES = frozenset(
    (
        "injury",
        "ejection",
        "defensive_error",
        "defensive_error_cluster",
        "predeclared",
    )
)
_DEFENSIVE_ERROR_EVENT_TYPES = frozenset(
    ("defensive_error", "defensive_error_cluster")
)
_SIDES = ("away", "home")
_COUNT_COMPONENT_IDS = frozenset(
    f"{side}_{suffix}"
    for side in _SIDES
    for suffix in (
        "pitcher_chain_runs_allowed",
        "pitcher_chain_workload_outs",
        "f5_runs",
        "f5_traffic",
        "f5_extra_base_hits",
        "full_runs",
        "full_traffic",
        "full_extra_base_hits",
        "bullpen_runs_allowed",
    )
)
_MARKET_HORIZONS = {
    MarketType.F5_SIDE: "f5",
    MarketType.F5_TOTAL: "f5",
    MarketType.ML: "full",
    MarketType.PLUS_1_5: "full",
    MarketType.FULL_TOTAL: "full",
}
_EVIDENCE_PATH_ORDER = {
    evidence_path: index
    for index, evidence_path in enumerate(PriorityEvidencePath)
}


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    for item in value:
        _require_nonempty_string(item, field_name)
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must contain unique values")
    return value


def _require_notes(value: object) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(note, str) or not note.strip() for note in value
    ):
        raise ValueError("notes must be a tuple of nonempty strings")


def _require_canonical_path(value: object) -> None:
    if not isinstance(value, PriorityEvidencePath):
        raise ValueError("evidence_path must be a canonical PriorityEvidencePath")


def _is_binary_component_id(component_id: str) -> bool:
    return component_id.endswith("_exposed") or component_id.endswith(
        "_protected_bullpen_close_entry"
    )


@dataclass(frozen=True)
class PredictiveInterval:
    p05: float
    p10: float
    p20: float
    median: float
    p80: float
    p90: float
    p95: float

    def __post_init__(self) -> None:
        values = (
            self.p05,
            self.p10,
            self.p20,
            self.median,
            self.p80,
            self.p90,
            self.p95,
        )
        if any(not _is_finite_number(value) for value in values):
            raise ValueError("predictive interval values must be finite numbers")
        if any(first > second for first, second in zip(values, values[1:])):
            raise ValueError("predictive interval values must be nondecreasing")


def _interval_values(interval: PredictiveInterval) -> tuple[float, ...]:
    return (
        interval.p05,
        interval.p10,
        interval.p20,
        interval.median,
        interval.p80,
        interval.p90,
        interval.p95,
    )


@dataclass(frozen=True)
class ComponentForecast:
    component_id: str
    evidence_path: PriorityEvidencePath
    interval: PredictiveInterval
    direction: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.component_id, "component_id")
        _require_canonical_path(self.evidence_path)
        if not isinstance(self.interval, PredictiveInterval):
            raise ValueError("interval must be PredictiveInterval")
        interval_values = _interval_values(self.interval)
        if self.component_id in _COUNT_COMPONENT_IDS and any(
            value < 0.0 for value in interval_values
        ):
            raise ValueError("count forecast interval must be nonnegative")
        if _is_binary_component_id(self.component_id) and any(
            not 0.0 <= value <= 1.0 for value in interval_values
        ):
            raise ValueError("binary forecast interval must stay in [0, 1]")
        if not isinstance(self.direction, str) or self.direction not in _DIRECTIONS:
            raise ValueError(
                "direction must be high_is_supportive or low_is_supportive"
            )


@dataclass(frozen=True)
class ComponentObservation:
    component_id: str
    value: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.component_id, "component_id")
        if not _is_finite_number(self.value):
            raise ValueError("component observation value must be finite")
        if self.component_id in _COUNT_COMPONENT_IDS and (
            self.value < 0.0 or not float(self.value).is_integer()
        ):
            raise ValueError(
                "count observation must be a nonnegative integer"
            )
        if (
            _is_binary_component_id(self.component_id)
            and self.value not in (0.0, 1.0)
        ):
            raise ValueError("binary observation must be 0 or 1")


@dataclass(frozen=True)
class ExogenousEvent:
    event_type: str
    affected_component_id: str
    documented_source_observation_id: str
    unearned_runs: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_type, str)
            or self.event_type not in _EXOGENOUS_EVENT_TYPES
        ):
            raise ValueError("event_type must be a canonical supported string")
        _require_nonempty_string(
            self.affected_component_id,
            "affected_component_id",
        )
        _require_nonempty_string(
            self.documented_source_observation_id,
            "documented_source_observation_id",
        )
        if (
            isinstance(self.unearned_runs, bool)
            or not isinstance(self.unearned_runs, int)
            or self.unearned_runs < 0
        ):
            raise ValueError("unearned_runs must be a nonnegative integer")


class ContradictionRule(str, Enum):
    STARTER_SUPPRESSION = "starter_suppression"
    STARTER_COLLAPSE = "starter_collapse"
    TOP_ORDER_OR_RECENT_CONVERSION = "top_order_or_recent_conversion"
    LATE_BULLPEN_EXPOSURE = "late_bullpen_exposure"
    BULLPEN_PROTECTION = "bullpen_protection"


_CONTRADICTION_RULE_BY_EVIDENCE_PATH = {
    PriorityEvidencePath.STARTER_SUPPRESSION: (
        ContradictionRule.STARTER_SUPPRESSION
    ),
    PriorityEvidencePath.STARTER_COLLAPSE: ContradictionRule.STARTER_COLLAPSE,
    PriorityEvidencePath.TOP_ORDER_MATCHUP: (
        ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
    ),
    PriorityEvidencePath.RECENT_RUN_CONVERSION: (
        ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
    ),
    PriorityEvidencePath.UNDERDOG_RUN_CREATION: (
        ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION
    ),
    PriorityEvidencePath.LATE_BULLPEN_EXPOSURE: (
        ContradictionRule.LATE_BULLPEN_EXPOSURE
    ),
    PriorityEvidencePath.BULLPEN_PROTECTION: (
        ContradictionRule.BULLPEN_PROTECTION
    ),
}


@dataclass(frozen=True)
class PrincipalPathBinding:
    evidence_path: PriorityEvidencePath
    component_ids: tuple[str, ...]
    contradiction_rule: ContradictionRule

    def __post_init__(self) -> None:
        _require_canonical_path(self.evidence_path)
        component_ids = _require_string_tuple(
            self.component_ids,
            "component_ids",
        )
        if not component_ids:
            raise ValueError("component_ids must not be empty")
        if not isinstance(self.contradiction_rule, ContradictionRule):
            raise ValueError("contradiction_rule must be ContradictionRule")
        if (
            self.contradiction_rule
            is not _CONTRADICTION_RULE_BY_EVIDENCE_PATH[self.evidence_path]
        ):
            raise ValueError("contradiction_rule must match evidence_path")


@dataclass(frozen=True)
class PathReview:
    evidence_path: PriorityEvidencePath
    component_ids: tuple[str, ...]
    contradicted: bool
    inside_80_count: int
    component_count: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_canonical_path(self.evidence_path)
        component_ids = _require_string_tuple(
            self.component_ids,
            "component_ids",
        )
        if not component_ids:
            raise ValueError("component_ids must not be empty")
        if not isinstance(self.contradicted, bool):
            raise ValueError("contradicted must be a boolean")
        for value in (self.inside_80_count, self.component_count):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("component counts must be nonnegative integers")
        if (
            self.component_count != len(component_ids)
            or self.inside_80_count > self.component_count
        ):
            raise ValueError("component counts must match the bound components")
        _require_notes(self.notes)


def _shape_matches_starter_suppression(component_ids: set[str]) -> bool:
    return any(
        component_ids == {f"{side}_pitcher_chain_runs_allowed"}
        for side in _SIDES
    )


def _shape_matches_starter_collapse(component_ids: set[str]) -> bool:
    return any(
        component_ids
        == {
            f"{side}_pitcher_chain_runs_allowed",
            f"{side}_pitcher_chain_workload_outs",
        }
        for side in _SIDES
    )


def _shape_matches_conversion(
    component_ids: set[str],
    market_type: MarketType,
) -> bool:
    horizon = _MARKET_HORIZONS[market_type]
    return any(
        component_ids
        == {
            f"{side}_{horizon}_runs",
            f"{side}_{horizon}_traffic",
            f"{side}_{horizon}_extra_base_hits",
        }
        for side in _SIDES
    )


def _shape_matches_late_exposure(component_ids: set[str]) -> bool:
    if len(component_ids) != 2:
        return False
    for side in _SIDES:
        workload_id = f"{side}_pitcher_chain_workload_outs"
        if workload_id not in component_ids:
            continue
        exposure_ids = component_ids.difference((workload_id,))
        if len(exposure_ids) != 1:
            continue
        exposure_id = next(iter(exposure_ids))
        prefix = f"{side}_bullpen_segment_"
        suffix = "_exposed"
        if exposure_id.startswith(prefix) and exposure_id.endswith(suffix):
            segment_id = exposure_id[len(prefix) : -len(suffix)]
            if segment_id.strip():
                return True
    return False


def _shape_matches_bullpen_protection(component_ids: set[str]) -> bool:
    return any(
        component_ids
        == {
            f"{side}_bullpen_runs_allowed",
            f"{side}_protected_bullpen_close_entry",
        }
        for side in _SIDES
    )


def _validate_binding_shape(
    binding: PrincipalPathBinding,
    market_type: MarketType,
) -> None:
    component_ids = set(binding.component_ids)
    rule = binding.contradiction_rule
    matches = {
        ContradictionRule.STARTER_SUPPRESSION: (
            _shape_matches_starter_suppression(component_ids)
        ),
        ContradictionRule.STARTER_COLLAPSE: (
            _shape_matches_starter_collapse(component_ids)
        ),
        ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION: (
            _shape_matches_conversion(component_ids, market_type)
        ),
        ContradictionRule.LATE_BULLPEN_EXPOSURE: (
            _shape_matches_late_exposure(component_ids)
        ),
        ContradictionRule.BULLPEN_PROTECTION: (
            _shape_matches_bullpen_protection(component_ids)
        ),
    }
    if not matches[rule]:
        raise ValueError(f"invalid binding shape for {rule.value}")


@dataclass(frozen=True)
class ReviewCandidate:
    capture_forecast_link_record_id: str
    capture_id: str
    forecast_run_id: str
    official_result_record_id: str
    market_key: str
    market_type: MarketType
    recommended: bool
    market_result: str | None
    component_forecasts: tuple[ComponentForecast, ...]
    component_observations: tuple[ComponentObservation, ...]
    principal_paths: tuple[PrincipalPathBinding, ...]
    exogenous_events: tuple[ExogenousEvent, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "capture_forecast_link_record_id",
            "capture_id",
            "forecast_run_id",
            "official_result_record_id",
            "market_key",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if not isinstance(self.market_type, MarketType):
            raise ValueError("market_type must be MarketType")
        if not isinstance(self.recommended, bool):
            raise ValueError("recommended must be a boolean")
        if self.market_result is not None and (
            not isinstance(self.market_result, str)
            or self.market_result not in _MARKET_RESULTS
        ):
            raise ValueError("market_result must be win, loss, push, or None")
        if self.recommended and self.market_result is None:
            raise ValueError("recommended candidate requires market_result")

        if (
            not isinstance(self.component_forecasts, tuple)
            or not self.component_forecasts
        ):
            raise ValueError("component_forecasts must be a nonempty tuple")
        if any(
            not isinstance(forecast, ComponentForecast)
            for forecast in self.component_forecasts
        ):
            raise ValueError(
                "component_forecasts must contain ComponentForecast values"
            )
        forecast_ids = tuple(
            forecast.component_id for forecast in self.component_forecasts
        )
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("duplicate component forecast ID")
        forecasts_by_id = {
            forecast.component_id: forecast
            for forecast in self.component_forecasts
        }

        if not isinstance(self.component_observations, tuple):
            raise ValueError("component_observations must be a tuple")
        if any(
            not isinstance(observation, ComponentObservation)
            for observation in self.component_observations
        ):
            raise ValueError(
                "component_observations must contain ComponentObservation values"
            )
        observation_ids = tuple(
            observation.component_id for observation in self.component_observations
        )
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("duplicate component observation ID")
        if set(observation_ids) != set(forecast_ids):
            raise ValueError(
                "component observation coverage must exactly match forecasts"
            )

        if not isinstance(self.principal_paths, tuple):
            raise ValueError("principal_paths must be a tuple")
        if any(
            not isinstance(path, PrincipalPathBinding)
            for path in self.principal_paths
        ):
            raise ValueError(
                "principal_paths must contain PrincipalPathBinding values"
            )
        if self.recommended and not 2 <= len(self.principal_paths) <= 4:
            raise ValueError(
                "recommended candidates require two to four principal paths"
            )
        evidence_paths = tuple(
            path.evidence_path for path in self.principal_paths
        )
        if len(set(evidence_paths)) != len(evidence_paths):
            raise ValueError("candidate requires unique principal evidence paths")

        available_ids = set(forecast_ids).intersection(observation_ids)
        bound_paths_by_component: dict[str, set[PriorityEvidencePath]] = {}
        for path in self.principal_paths:
            _validate_binding_shape(path, self.market_type)
            missing = set(path.component_ids).difference(available_ids)
            if missing:
                raise ValueError(
                    "every bound component must exist in forecasts and observations"
                )
            for component_id in path.component_ids:
                bound_paths_by_component.setdefault(component_id, set()).add(
                    path.evidence_path
                )

        if any(
            forecasts_by_id[component_id].evidence_path not in evidence_paths
            for component_id, evidence_paths in bound_paths_by_component.items()
        ):
            raise ValueError(
                "bound component forecast must match binding evidence_path"
            )

        if not isinstance(self.exogenous_events, tuple):
            raise ValueError("exogenous_events must be a tuple")
        if any(
            not isinstance(event, ExogenousEvent)
            for event in self.exogenous_events
        ):
            raise ValueError("exogenous_events must contain ExogenousEvent values")
        for event in self.exogenous_events:
            if event.affected_component_id not in bound_paths_by_component:
                raise ValueError(
                    "exogenous event affected component must be principal-bound"
                )


@dataclass(frozen=True)
class ReviewClassification:
    classification: str | None
    inside_80_ratio: float
    contradicted_paths: tuple[PriorityEvidencePath, ...]
    path_reviews: tuple[PathReview, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.classification is not None and (
            not isinstance(self.classification, str)
            or self.classification not in _CLASSIFICATIONS
        ):
            raise ValueError("classification must be A, B, C, D, or None")
        if (
            not _is_finite_number(self.inside_80_ratio)
            or not 0.0 <= self.inside_80_ratio <= 1.0
        ):
            raise ValueError("inside_80_ratio must be a finite number in [0, 1]")
        if not isinstance(self.contradicted_paths, tuple) or any(
            not isinstance(path, PriorityEvidencePath)
            for path in self.contradicted_paths
        ):
            raise ValueError(
                "contradicted_paths must contain canonical PriorityEvidencePath values"
            )
        if len(set(self.contradicted_paths)) != len(self.contradicted_paths):
            raise ValueError("contradicted_paths must be unique")
        if not isinstance(self.path_reviews, tuple) or any(
            not isinstance(review, PathReview) for review in self.path_reviews
        ):
            raise ValueError("path_reviews must contain PathReview values")
        expected_paths = tuple(
            review.evidence_path
            for review in self.path_reviews
            if review.contradicted
        )
        if self.contradicted_paths != expected_paths:
            raise ValueError(
                "contradicted_paths must match contradicted path reviews"
            )
        has_contradictions = bool(self.contradicted_paths)
        if self.classification == "A" and (
            self.inside_80_ratio < 0.75 or has_contradictions
        ):
            raise ValueError(
                "A requires inside_80_ratio >= 0.75 and no contradictions"
            )
        if self.classification == "B" and (
            self.inside_80_ratio < 0.60 or has_contradictions
        ):
            raise ValueError(
                "B requires inside_80_ratio >= 0.60 and no contradictions"
            )
        if (
            self.classification == "C"
            and self.inside_80_ratio >= 0.60
            and not has_contradictions
        ):
            raise ValueError(
                "C requires a contradiction or inside_80_ratio < 0.60"
            )
        _require_notes(self.notes)


def _inside_80(
    forecast: ComponentForecast,
    observation: ComponentObservation,
) -> bool:
    return forecast.interval.p10 <= observation.value <= forecast.interval.p90


def _outside_90(
    forecast: ComponentForecast,
    observation: ComponentObservation,
) -> bool:
    return observation.value < forecast.interval.p05 or (
        observation.value > forecast.interval.p95
    )


def _components_for_path(
    binding: PrincipalPathBinding,
    forecasts: dict[str, ComponentForecast],
    observations: dict[str, ComponentObservation],
) -> tuple[tuple[ComponentForecast, ComponentObservation], ...]:
    return tuple(
        (forecasts[component_id], observations[component_id])
        for component_id in binding.component_ids
    )


def _contradicts_path(
    binding: PrincipalPathBinding,
    components: tuple[tuple[ComponentForecast, ComponentObservation], ...],
) -> bool:
    by_id = {
        forecast.component_id: (forecast, observation)
        for forecast, observation in components
    }
    leaves_central_90 = any(
        _outside_90(forecast, observation)
        for forecast, observation in components
    )
    if not leaves_central_90:
        return False

    rule = binding.contradiction_rule
    if rule is ContradictionRule.STARTER_SUPPRESSION:
        forecast, observation = components[0]
        return observation.value > forecast.interval.p90

    if rule is ContradictionRule.STARTER_COLLAPSE:
        runs_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_pitcher_chain_runs_allowed")
        )
        workload_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_pitcher_chain_workload_outs")
        )
        runs_forecast, runs_observation = by_id[runs_id]
        workload_forecast, workload_observation = by_id[workload_id]
        return (
            runs_observation.value < runs_forecast.interval.p10
            and workload_observation.value >= workload_forecast.interval.median
        )

    if rule is ContradictionRule.TOP_ORDER_OR_RECENT_CONVERSION:
        return all(
            observation.value < forecast.interval.p10
            for forecast, observation in components
        )

    if rule is ContradictionRule.LATE_BULLPEN_EXPOSURE:
        exposure_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_exposed")
        )
        workload_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_pitcher_chain_workload_outs")
        )
        _, exposure_observation = by_id[exposure_id]
        workload_forecast, workload_observation = by_id[workload_id]
        return (
            exposure_observation.value == 0
            and workload_observation.value >= workload_forecast.interval.p80
        )

    if rule is ContradictionRule.BULLPEN_PROTECTION:
        runs_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_bullpen_runs_allowed")
        )
        close_entry_id = next(
            component_id
            for component_id in by_id
            if component_id.endswith("_protected_bullpen_close_entry")
        )
        runs_forecast, runs_observation = by_id[runs_id]
        _, close_entry_observation = by_id[close_entry_id]
        return (
            runs_observation.value > runs_forecast.interval.p90
            and close_entry_observation.value == 1
        )

    raise AssertionError(f"unhandled contradiction rule: {rule}")


def _review_path(
    binding: PrincipalPathBinding,
    forecasts: dict[str, ComponentForecast],
    observations: dict[str, ComponentObservation],
) -> PathReview:
    components = _components_for_path(binding, forecasts, observations)
    return PathReview(
        evidence_path=binding.evidence_path,
        component_ids=binding.component_ids,
        contradicted=_contradicts_path(binding, components),
        inside_80_count=sum(
            _inside_80(forecast, observation)
            for forecast, observation in components
        ),
        component_count=len(components),
    )


def _has_qualifying_exogenous_event(
    candidate: ReviewCandidate,
    forecasts: dict[str, ComponentForecast],
    observations: dict[str, ComponentObservation],
) -> bool:
    for event in candidate.exogenous_events:
        if (
            event.event_type in _DEFENSIVE_ERROR_EVENT_TYPES
            and event.unearned_runs < 3
        ):
            continue
        component_id = event.affected_component_id
        if _outside_90(forecasts[component_id], observations[component_id]):
            return True
    return False


def classify_review(candidate: ReviewCandidate) -> ReviewClassification:
    """Classify a frozen forecast/outcome pair with fixed D/C/A/B precedence."""

    if not isinstance(candidate, ReviewCandidate):
        raise ValueError("candidate must be ReviewCandidate")

    forecasts = {
        forecast.component_id: forecast
        for forecast in candidate.component_forecasts
    }
    observations = {
        observation.component_id: observation
        for observation in candidate.component_observations
    }
    ordered_paths = sorted(
        candidate.principal_paths,
        key=lambda path: _EVIDENCE_PATH_ORDER[path.evidence_path],
    )
    path_reviews = tuple(
        _review_path(path, forecasts, observations)
        for path in ordered_paths
    )
    contradicted_paths = tuple(
        review.evidence_path for review in path_reviews if review.contradicted
    )
    principal_component_ids = frozenset(
        component_id
        for path in candidate.principal_paths
        for component_id in path.component_ids
    )
    inside_80_count = sum(
        _inside_80(forecasts[component_id], observations[component_id])
        for component_id in principal_component_ids
    )
    inside_80_ratio = (
        inside_80_count / len(principal_component_ids)
        if principal_component_ids
        else 1.0
    )

    if not candidate.recommended:
        classification = None
    elif _has_qualifying_exogenous_event(candidate, forecasts, observations):
        classification = "D"
    elif contradicted_paths or inside_80_ratio < 0.60:
        classification = "C"
    elif candidate.market_result == "win" and inside_80_ratio >= 0.75:
        classification = "A"
    else:
        classification = "B"

    return ReviewClassification(
        classification=classification,
        inside_80_ratio=inside_80_ratio,
        contradicted_paths=contradicted_paths,
        path_reviews=path_reviews,
    )


__all__ = (
    "ComponentForecast",
    "ComponentObservation",
    "ContradictionRule",
    "ExogenousEvent",
    "PathReview",
    "PredictiveInterval",
    "PrincipalPathBinding",
    "ReviewCandidate",
    "ReviewClassification",
    "classify_review",
)
