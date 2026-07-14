"""Postgame review orchestration over immutable forecast ledger records."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib

from analysis.forecasting.engine import SemanticMarketInput
from analysis.forecasting.identity import (
    build_forecast_run_id_from_capture_payload,
    canonical_json,
    normalized_input_hash_from_capture_payload,
)
from analysis.forecasting.inputs import MarketType
from analysis.forecasting.market import settle_observed_market
from analysis.forecasting.simulation import (
    ComponentDistribution,
    DistributionSummary,
)
from analysis.ledger.records import (
    CaptureForecastLink,
    CaptureRecord,
    ForecastRecord,
    OfficialResultRecord,
    ReviewCommand,
    ReviewRecord,
)
from analysis.model import PriorityEvidencePath

from .classification import (
    ComponentForecast,
    ComponentObservation,
    ExogenousEvent,
    PredictiveInterval,
    PrincipalPathBinding,
    ReviewCandidate,
    ReviewClassification,
    classify_review,
    validate_principal_path_bindings,
)


_SIDE_MARKETS = frozenset(
    (MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5)
)
_SUPPORTED_EVENT_TYPES = frozenset(
    ("injury", "ejection", "defensive_error", "defensive_error_cluster")
)
_SUMMARY_FIELDS = ("p05", "p10", "p20", "median", "p80", "p90", "p95")
_EVIDENCE_PATH_ORDER = {
    evidence_path: index
    for index, evidence_path in enumerate(PriorityEvidencePath)
}


@dataclass(frozen=True)
class RecordedReview:
    candidate: ReviewCandidate
    classification: ReviewClassification
    command: ReviewCommand
    record: ReviewRecord

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ReviewCandidate):
            raise TypeError("candidate must be ReviewCandidate")
        if not isinstance(self.classification, ReviewClassification):
            raise TypeError("classification must be ReviewClassification")
        if not isinstance(self.command, ReviewCommand):
            raise TypeError("command must be ReviewCommand")
        if not isinstance(self.record, ReviewRecord):
            raise TypeError("record must be ReviewRecord")
        if self.record.capture_id != self.candidate.capture_id:
            raise ValueError("record capture_id must match candidate")
        if self.record.forecast_run_id != self.candidate.forecast_run_id:
            raise ValueError("record forecast_run_id must match candidate")
        if self.record.official_result_record_id != (
            self.candidate.official_result_record_id
        ):
            raise ValueError(
                "record official_result_record_id must match candidate"
            )
        if self.record.market_key != self.candidate.market_key:
            raise ValueError("record market_key must match candidate")


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_store(store: object, *, append: bool = False) -> None:
    if not callable(getattr(store, "iter_records", None)):
        raise TypeError("store must provide iter_records")
    if append and not callable(getattr(store, "append_review", None)):
        raise TypeError("store must provide append_review")


def _exact_record_by_id(
    records: tuple[object, ...],
    record_type: type,
    record_id: str,
    field_name: str,
) -> object:
    matches = tuple(
        record
        for record in records
        if isinstance(record, record_type)
        and record.envelope.record_id == record_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"{field_name} must reference exactly one {record_type.__name__}"
        )
    return matches[0]


def _capture_for_link(
    records: tuple[object, ...],
    link: CaptureForecastLink,
) -> CaptureRecord:
    matches = tuple(
        record
        for record in records
        if isinstance(record, CaptureRecord)
        and record.capture_id == link.capture_id
    )
    if len(matches) != 1:
        raise ValueError("link must resolve exactly one CaptureRecord")
    return matches[0]


def _forecast_for_link(
    records: tuple[object, ...],
    link: CaptureForecastLink,
) -> ForecastRecord:
    matches = tuple(
        record
        for record in records
        if isinstance(record, ForecastRecord)
        and record.forecast_run_id == link.forecast_run_id
    )
    if len(matches) != 1:
        raise ValueError("link must resolve exactly one ForecastRecord")
    return matches[0]


def _require_latest_result(
    records: tuple[object, ...],
    result: OfficialResultRecord,
) -> None:
    game_results = tuple(
        record
        for record in records
        if isinstance(record, OfficialResultRecord)
        and record.game_pk == result.game_pk
    )
    superseded_ids = frozenset(
        record.supersedes_record_id
        for record in game_results
        if record.supersedes_record_id is not None
    )
    latest = tuple(
        record
        for record in game_results
        if record.envelope.record_id not in superseded_ids
    )
    if len(latest) != 1:
        raise ValueError(
            "official result revisions must resolve one latest valid record"
        )
    if latest[0].envelope.record_id != result.envelope.record_id:
        raise ValueError("result_record_id references a superseded official result")


def _capture_identity(capture: CaptureRecord) -> tuple[str, str]:
    payload = _require_mapping(capture.payload, "capture payload")
    identity = _require_mapping(payload.get("identity"), "capture payload identity")
    game_pk = identity.get("game_pk")
    if (
        isinstance(game_pk, bool)
        or not isinstance(game_pk, int)
        or game_pk != capture.game_pk
    ):
        raise ValueError("capture payload identity game_pk must match capture game_pk")
    away_team = _require_nonempty_string(
        identity.get("away_team"),
        "capture payload identity away_team",
    )
    home_team = _require_nonempty_string(
        identity.get("home_team"),
        "capture payload identity home_team",
    )
    if away_team == home_team:
        raise ValueError("captured away_team and home_team must differ")
    return away_team, home_team


def _require_deterministic_forecast_identity(
    capture: CaptureRecord,
    forecast: ForecastRecord,
    simulation_config: Mapping[str, object],
) -> None:
    expected_input_hash = normalized_input_hash_from_capture_payload(
        capture.payload
    )
    if capture.normalized_input_hash != expected_input_hash:
        raise ValueError(
            "CaptureRecord normalized_input_hash must match capture semantics"
        )
    if forecast.normalized_input_hash != expected_input_hash:
        raise ValueError(
            "ForecastRecord normalized_input_hash must match capture semantics"
        )
    expected_config_hash = hashlib.sha256(
        canonical_json(simulation_config).encode("utf-8")
    ).hexdigest()
    if forecast.simulation_config_hash != expected_config_hash:
        raise ValueError(
            "ForecastRecord simulation_config_hash must match frozen outputs"
        )
    expected_forecast_run_id = build_forecast_run_id_from_capture_payload(
        capture.payload,
        forecast.model_version,
        simulation_config,
    )
    if forecast.forecast_run_id != expected_forecast_run_id:
        raise ValueError(
            "ForecastRecord forecast_run_id must match canonical capture semantics, "
            "model_version, and simulation_config"
        )


def _resolve_selected_side(
    market: SemanticMarketInput,
    away_team: str,
    home_team: str,
) -> SemanticMarketInput:
    if market.market_type not in _SIDE_MARKETS:
        return market
    if market.selected_side in {"away", "home"}:
        return market
    side_by_team = {away_team: "away", home_team: "home"}
    selected_side = side_by_team.get(market.selected_side)
    if selected_side is None:
        raise ValueError(
            "side market selected_side must use away/home or a captured team alias"
        )
    return replace(market, selected_side=selected_side)


def _exact_shadow_comparison(
    outputs: Mapping[str, object],
    forecast_context: object,
    forecast_run_id: str,
    market_key: str,
    away_team: str,
    home_team: str,
) -> tuple[object, Mapping[str, SemanticMarketInput]]:
    from analysis.shadow import (
        FrozenForecastContext,
        deserialize_shadow_comparison,
        validate_shadow_forecast_context,
    )

    if not isinstance(forecast_context, FrozenForecastContext):
        raise TypeError("forecast_context must be FrozenForecastContext")

    shadow = deserialize_shadow_comparison(
        outputs.get("shadow_comparison")
    )
    if shadow.forecast_run_id != forecast_run_id:
        raise ValueError(
            "shadow_comparison forecast_run_id must match ForecastRecord"
        )
    semantic_market_inputs = validate_shadow_forecast_context(
        shadow,
        forecast_context.market_inputs,
        forecast_context.markets,
        forecast_context.component_distributions,
        away_team=away_team,
        home_team=home_team,
    )
    matches = tuple(
        comparison
        for comparison in shadow.market_comparisons
        if comparison.market_key == market_key
    )
    if len(matches) != 1:
        raise ValueError(
            f"exact pregame principal-path binding is missing for {market_key}"
        )
    comparison = matches[0]
    flags = comparison.review_ineligibility_flags
    if not comparison.review_eligible:
        if not flags:
            raise ValueError(
                "review-ineligible comparison must contain stored flags"
            )
        raise ValueError(
            "review-ineligible shadow comparison: " + ", ".join(flags)
        )
    if flags:
        raise ValueError("review-eligible comparison cannot contain flags")
    return comparison, semantic_market_inputs


def _require_component_namespace(component_id: str) -> None:
    if not component_id.startswith(("away_", "home_")):
        raise ValueError(
            "component IDs must use the captured away/home namespace; "
            "team-abbreviation component IDs are unsupported"
        )


def _principal_paths(
    comparison: object,
    market_type: MarketType,
) -> tuple[PrincipalPathBinding, ...]:
    from analysis.shadow import ShadowMarketComparison

    if not isinstance(comparison, ShadowMarketComparison):
        raise TypeError("comparison must be ShadowMarketComparison")
    paths = comparison.principal_paths
    if not 2 <= len(paths) <= 4:
        raise ValueError(
            "pregame principal-path binding requires two to four paths"
        )
    for path in paths:
        for component_id in path.component_ids:
            _require_component_namespace(component_id)
    if len({path.evidence_path for path in paths}) != len(paths):
        raise ValueError("pregame principal paths must be unique")
    validate_principal_path_bindings(market_type, paths)
    return paths


def _component_direction(
    evidence_path: PriorityEvidencePath,
    component_id: str,
) -> str:
    if evidence_path is PriorityEvidencePath.STARTER_SUPPRESSION:
        return "low_is_supportive"
    if evidence_path is PriorityEvidencePath.STARTER_COLLAPSE:
        return (
            "low_is_supportive"
            if component_id.endswith("_workload_outs")
            else "high_is_supportive"
        )
    if evidence_path is PriorityEvidencePath.LATE_BULLPEN_EXPOSURE:
        return (
            "high_is_supportive"
            if component_id.endswith("_exposed")
            else "low_is_supportive"
        )
    if evidence_path is PriorityEvidencePath.BULLPEN_PROTECTION:
        return (
            "low_is_supportive"
            if component_id.endswith("_runs_allowed")
            else "high_is_supportive"
        )
    return "high_is_supportive"


def _predictive_interval(summary: DistributionSummary) -> PredictiveInterval:
    return PredictiveInterval(
        **{field_name: getattr(summary, field_name) for field_name in _SUMMARY_FIELDS}
    )


def _component_pairs(
    distributions: Mapping[str, ComponentDistribution],
    result: OfficialResultRecord,
    paths: tuple[PrincipalPathBinding, ...],
) -> tuple[tuple[ComponentForecast, ...], tuple[ComponentObservation, ...]]:
    ordered_paths = tuple(
        sorted(paths, key=lambda path: _EVIDENCE_PATH_ORDER[path.evidence_path])
    )
    path_by_component = {}
    for path in ordered_paths:
        for component_id in path.component_ids:
            path_by_component.setdefault(component_id, path.evidence_path)

    forecasts = []
    observations = []
    for component_id in sorted(path_by_component):
        if component_id not in distributions:
            raise ValueError(
                f"missing frozen component distribution for {component_id}"
            )
        distribution = distributions[component_id]
        if component_id not in result.component_values:
            raise ValueError(
                "official result component_values is missing required "
                f"component {component_id}"
            )
        evidence_path = path_by_component[component_id]
        forecasts.append(
            ComponentForecast(
                component_id=component_id,
                evidence_path=evidence_path,
                interval=_predictive_interval(distribution.summary),
                direction=_component_direction(evidence_path, component_id),
            )
        )
        observations.append(
            ComponentObservation(
                component_id=component_id,
                value=float(result.component_values[component_id]),
            )
        )
    return tuple(forecasts), tuple(observations)


def _exogenous_events(
    result: OfficialResultRecord,
    required_component_ids: frozenset[str],
) -> tuple[ExogenousEvent, ...]:
    events = set()
    for event_value in result.documented_events:
        event = _require_mapping(event_value, "documented event")
        event_type = event.get("event_type")
        if (
            not isinstance(event_type, str)
            or event_type not in _SUPPORTED_EVENT_TYPES
        ):
            continue
        source_id = event.get("documented_source_observation_id")
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        affected_component_id = _require_nonempty_string(
            event.get("affected_component_id"),
            "documented event affected_component_id",
        )
        _require_component_namespace(affected_component_id)
        if affected_component_id not in required_component_ids:
            continue
        unearned_runs = event.get("unearned_runs", 0)
        converted = ExogenousEvent(
            event_type=event_type,
            affected_component_id=affected_component_id,
            documented_source_observation_id=source_id,
            unearned_runs=unearned_runs,
        )
        events.add(
            (
                converted.event_type,
                converted.affected_component_id,
                converted.documented_source_observation_id,
                converted.unearned_runs,
            )
        )
    return tuple(
        ExogenousEvent(
            event_type=event_type,
            affected_component_id=component_id,
            documented_source_observation_id=source_id,
            unearned_runs=unearned_runs,
        )
        for event_type, component_id, source_id, unearned_runs in sorted(events)
    )


def build_review_candidate(
    store: object,
    link_record_id: str,
    result_record_id: str,
    market_key: str,
) -> ReviewCandidate:
    """Build one objective candidate from exact linked frozen records."""

    _require_store(store)
    _require_nonempty_string(link_record_id, "link_record_id")
    _require_nonempty_string(result_record_id, "result_record_id")
    _require_nonempty_string(market_key, "market_key")
    records = tuple(store.iter_records())
    link = _exact_record_by_id(
        records,
        CaptureForecastLink,
        link_record_id,
        "link_record_id",
    )
    result = _exact_record_by_id(
        records,
        OfficialResultRecord,
        result_record_id,
        "result_record_id",
    )
    capture = _capture_for_link(records, link)
    forecast = _forecast_for_link(records, link)
    if capture.normalized_input_hash != forecast.normalized_input_hash:
        raise ValueError(
            "linked CaptureRecord and ForecastRecord normalized_input_hash mismatch"
        )
    if result.game_pk != capture.game_pk:
        raise ValueError("capture and official result game_pk mismatch")
    if result.finalized_at <= capture.captured_at:
        raise ValueError(
            "official result finalized_at must be strictly after captured_at"
        )
    if result.finalized_at <= capture.first_pitch:
        raise ValueError(
            "official result finalized_at must be strictly after first_pitch"
        )
    _require_latest_result(records, result)
    away_team, home_team = _capture_identity(capture)

    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    from analysis.shadow import deserialize_forecast_context

    forecast_context = deserialize_forecast_context(
        outputs,
        away_team=away_team,
        home_team=home_team,
    )
    _require_deterministic_forecast_identity(
        capture,
        forecast,
        forecast_context.simulation_config.to_mapping(),
    )
    if forecast_context.forecast_run_id != forecast.forecast_run_id:
        raise ValueError(
            "forecast outputs forecast_run_id must match ForecastRecord"
        )
    comparison, semantic_market_inputs = _exact_shadow_comparison(
        outputs,
        forecast_context,
        forecast.forecast_run_id,
        market_key,
        away_team,
        home_team,
    )
    if market_key not in semantic_market_inputs:
        raise ValueError(
            f"forecast outputs market_inputs is missing market_key {market_key}"
        )
    market = semantic_market_inputs[market_key]
    resolved_market = _resolve_selected_side(market, away_team, home_team)
    paths = _principal_paths(comparison, market.market_type)
    component_forecasts, component_observations = _component_pairs(
        forecast_context.component_distributions,
        result,
        paths,
    )
    required_component_ids = frozenset(
        forecast.component_id for forecast in component_forecasts
    )
    legacy_rating = _require_nonempty_string(
        comparison.legacy_rating,
        "shadow comparison legacy_rating",
    )
    market_result = settle_observed_market(resolved_market, result)
    return ReviewCandidate(
        capture_forecast_link_record_id=link.envelope.record_id,
        capture_id=capture.capture_id,
        forecast_run_id=forecast.forecast_run_id,
        official_result_record_id=result.envelope.record_id,
        market_key=market_key,
        market_type=market.market_type,
        recommended=legacy_rating != "pass",
        market_result=market_result,
        component_forecasts=component_forecasts,
        component_observations=component_observations,
        principal_paths=paths,
        exogenous_events=_exogenous_events(result, required_component_ids),
    )


def _component_review_payloads(
    classification: ReviewClassification,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "evidence_path": review.evidence_path.value,
            "component_ids": review.component_ids,
            "contradicted": review.contradicted,
            "inside_80_count": review.inside_80_count,
            "component_count": review.component_count,
            "notes": review.notes,
        }
        for review in classification.path_reviews
    )


def _review_command(
    candidate: ReviewCandidate,
    classification: ReviewClassification,
) -> ReviewCommand:
    if candidate.market_result is None:
        raise ValueError("settled review candidate requires market_result")
    return ReviewCommand(
        capture_forecast_link_record_id=(
            candidate.capture_forecast_link_record_id
        ),
        capture_id=candidate.capture_id,
        forecast_run_id=candidate.forecast_run_id,
        official_result_record_id=candidate.official_result_record_id,
        market_key=candidate.market_key,
        market_result=candidate.market_result,
        classification=classification.classification,
        inside_80_ratio=classification.inside_80_ratio,
        contradicted_paths=tuple(
            path.value for path in classification.contradicted_paths
        ),
        component_reviews=_component_review_payloads(classification),
        qualifying_exogenous_source_observation_ids=(
            classification.qualifying_exogenous_source_observation_ids
        ),
    )


def build_and_record_review(
    store: object,
    link_record_id: str,
    result_record_id: str,
    market_key: str,
) -> RecordedReview:
    """Classify and idempotently append one review after a pure build."""

    _require_store(store, append=True)
    candidate = build_review_candidate(
        store,
        link_record_id,
        result_record_id,
        market_key,
    )
    classification = classify_review(candidate)
    command = _review_command(candidate, classification)
    idempotency_key = (
        f"review:{candidate.capture_id}:{candidate.forecast_run_id}:"
        f"{candidate.official_result_record_id}:{candidate.market_key}"
    )
    record = store.append_review(command, idempotency_key=idempotency_key)
    if not isinstance(record, ReviewRecord):
        raise TypeError("store.append_review must return ReviewRecord")
    return RecordedReview(
        candidate=candidate,
        classification=classification,
        command=command,
        record=record,
    )


__all__ = (
    "RecordedReview",
    "build_and_record_review",
    "build_review_candidate",
)
