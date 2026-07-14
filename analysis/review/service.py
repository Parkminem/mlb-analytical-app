"""Postgame review orchestration over immutable forecast ledger records."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from analysis.forecasting.engine import SemanticMarketInput
from analysis.forecasting.inputs import MarketType, SettlementFormat
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
    ContradictionRule,
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


def _require_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence")
    return tuple(value)


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


def _parse_enum(enum_type: type, value: object, field_name: str):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field_name} is unsupported: {value}") from error


def _semantic_market_input(
    outputs: Mapping[str, object],
    market_key: str,
) -> SemanticMarketInput:
    market_inputs = _require_mapping(
        outputs.get("market_inputs"),
        "forecast outputs market_inputs",
    )
    if market_key not in market_inputs:
        raise ValueError(
            f"forecast outputs market_inputs is missing market_key {market_key}"
        )
    value = _require_mapping(
        market_inputs[market_key],
        f"market_inputs[{market_key}]",
    )
    expected_fields = {
        "market_key",
        "market_type",
        "settlement_format",
        "selected_side",
        "line",
        "selected_decimal_price",
        "opposing_decimal_price",
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"market_inputs[{market_key}] must match SemanticMarketInput schema"
        )
    if value["market_key"] != market_key:
        raise ValueError("market_inputs key and market_key must match")
    return SemanticMarketInput(
        market_key=market_key,
        market_type=_parse_enum(
            MarketType,
            value["market_type"],
            "market_type",
        ),
        settlement_format=_parse_enum(
            SettlementFormat,
            value["settlement_format"],
            "settlement_format",
        ),
        selected_side=_require_nonempty_string(
            value["selected_side"],
            "selected_side",
        ),
        line=value["line"],
        selected_decimal_price=value["selected_decimal_price"],
        opposing_decimal_price=value["opposing_decimal_price"],
    )


def _require_semantic_fingerprint(
    outputs: Mapping[str, object],
    market: SemanticMarketInput,
) -> None:
    markets = _require_mapping(outputs.get("markets"), "forecast outputs markets")
    if market.market_key not in markets:
        raise ValueError(
            f"forecast outputs markets is missing market_key {market.market_key}"
        )
    evaluation = _require_mapping(
        markets[market.market_key],
        f"markets[{market.market_key}]",
    )
    fingerprint = evaluation.get("semantic_fingerprint")
    if fingerprint != market.semantic_fingerprint:
        raise ValueError(
            "market evaluation semantic_fingerprint must match frozen semantics"
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
    forecast_run_id: str,
    market_key: str,
) -> Mapping[str, object]:
    shadow = _require_mapping(
        outputs.get("shadow_comparison"),
        "forecast outputs shadow_comparison",
    )
    if shadow.get("authority") != "legacy":
        raise ValueError("shadow_comparison authority must be legacy")
    if shadow.get("forecast_run_id") != forecast_run_id:
        raise ValueError(
            "shadow_comparison forecast_run_id must match ForecastRecord"
        )
    comparisons = _require_sequence(
        shadow.get("market_comparisons"),
        "shadow_comparison market_comparisons",
    )
    parsed_comparisons = tuple(
        _require_mapping(comparison, "shadow market comparison")
        for comparison in comparisons
    )
    matches = tuple(
        comparison
        for comparison in parsed_comparisons
        if comparison.get("market_key") == market_key
    )
    if len(matches) != 1:
        raise ValueError(
            f"exact pregame principal-path binding is missing for {market_key}"
        )
    comparison = matches[0]
    review_eligible = comparison.get("review_eligible")
    if not isinstance(review_eligible, bool):
        raise ValueError("review_eligible must be a boolean")
    flags = _require_sequence(
        comparison.get("review_ineligibility_flags"),
        "review_ineligibility_flags",
    )
    for flag in flags:
        _require_nonempty_string(flag, "review_ineligibility_flags")
    if not review_eligible:
        if not flags:
            raise ValueError(
                "review-ineligible comparison must contain stored flags"
            )
        raise ValueError(
            "review-ineligible shadow comparison: " + ", ".join(flags)
        )
    if flags:
        raise ValueError("review-eligible comparison cannot contain flags")
    return comparison


def _require_component_namespace(component_id: str) -> None:
    if not component_id.startswith(("away_", "home_")):
        raise ValueError(
            "component IDs must use the captured away/home namespace; "
            "team-abbreviation component IDs are unsupported"
        )


def _principal_paths(
    comparison: Mapping[str, object],
    market_type: MarketType,
) -> tuple[PrincipalPathBinding, ...]:
    path_values = _require_sequence(
        comparison.get("principal_paths"),
        "pregame principal-path binding",
    )
    if not 2 <= len(path_values) <= 4:
        raise ValueError(
            "pregame principal-path binding requires two to four paths"
        )
    paths = []
    for index, value in enumerate(path_values):
        path = _require_mapping(value, f"principal_paths[{index}]")
        component_values = _require_sequence(
            path.get("component_ids"),
            f"principal_paths[{index}].component_ids",
        )
        component_ids = tuple(
            _require_nonempty_string(
                component_id,
                f"principal_paths[{index}].component_ids",
            )
            for component_id in component_values
        )
        for component_id in component_ids:
            _require_component_namespace(component_id)
        paths.append(
            PrincipalPathBinding(
                evidence_path=_parse_enum(
                    PriorityEvidencePath,
                    path.get("evidence_path"),
                    "evidence_path",
                ),
                component_ids=component_ids,
                contradiction_rule=_parse_enum(
                    ContradictionRule,
                    path.get("contradiction_rule"),
                    "contradiction_rule",
                ),
            )
        )
    resolved = tuple(paths)
    if len({path.evidence_path for path in resolved}) != len(resolved):
        raise ValueError("pregame principal paths must be unique")
    validate_principal_path_bindings(market_type, resolved)
    return resolved


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


def _distribution_summary(
    component_id: str,
    value: Mapping[str, object],
) -> DistributionSummary:
    summary_value = _require_mapping(
        value.get("summary"),
        f"component distribution {component_id} summary",
    )
    if set(summary_value) != set(_SUMMARY_FIELDS):
        raise ValueError(
            f"component distribution {component_id} summary is malformed"
        )
    return DistributionSummary(
        **{field_name: summary_value[field_name] for field_name in _SUMMARY_FIELDS}
    )


def _component_distribution(
    distributions: Mapping[str, object],
    component_id: str,
    simulation_count: int,
) -> ComponentDistribution:
    if component_id not in distributions:
        raise ValueError(
            f"missing frozen component distribution for {component_id}"
        )
    value = _require_mapping(
        distributions[component_id],
        f"component distribution {component_id}",
    )
    if value.get("component_id") != component_id:
        raise ValueError(
            f"component distribution key must match component_id {component_id}"
        )
    values = _require_sequence(
        value.get("values"),
        f"component distribution {component_id} values",
    )
    if len(values) != simulation_count:
        raise ValueError(
            f"component distribution {component_id} values must match "
            "simulation_count"
        )
    return ComponentDistribution(
        component_id=component_id,
        values=values,
        summary=_distribution_summary(component_id, value),
    )


def _predictive_interval(summary: DistributionSummary) -> PredictiveInterval:
    return PredictiveInterval(
        **{field_name: getattr(summary, field_name) for field_name in _SUMMARY_FIELDS}
    )


def _component_pairs(
    outputs: Mapping[str, object],
    result: OfficialResultRecord,
    paths: tuple[PrincipalPathBinding, ...],
) -> tuple[tuple[ComponentForecast, ...], tuple[ComponentObservation, ...]]:
    simulation_count = outputs.get("simulation_count")
    if (
        isinstance(simulation_count, bool)
        or not isinstance(simulation_count, int)
        or simulation_count <= 0
    ):
        raise ValueError("forecast outputs simulation_count must be positive")
    distributions = _require_mapping(
        outputs.get("component_distributions"),
        "forecast outputs component_distributions",
    )
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
        distribution = _component_distribution(
            distributions,
            component_id,
            simulation_count,
        )
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
    _require_latest_result(records, result)
    away_team, home_team = _capture_identity(capture)

    outputs = _require_mapping(forecast.outputs, "forecast outputs")
    comparison = _exact_shadow_comparison(
        outputs,
        forecast.forecast_run_id,
        market_key,
    )
    market = _semantic_market_input(outputs, market_key)
    _require_semantic_fingerprint(outputs, market)
    resolved_market = _resolve_selected_side(market, away_team, home_team)
    paths = _principal_paths(comparison, market.market_type)
    component_forecasts, component_observations = _component_pairs(
        outputs,
        result,
        paths,
    )
    required_component_ids = frozenset(
        forecast.component_id for forecast in component_forecasts
    )
    legacy_rating = _require_nonempty_string(
        comparison.get("legacy_rating"),
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
