"""Strict JSON bundle schemas for shadow operator stages."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
import hashlib
import hmac
import json
from threading import RLock
from typing import Annotated, Literal, TypeVar
from weakref import ReferenceType, ref

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    StrictBool,
    TypeAdapter,
    ValidateAs,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from analysis.forecasting import NormalizedGameInput
from analysis.forecasting.identity import canonical_json, capture_payload
from analysis.ledger import MarketObservationCommand, OfficialResultCommand
from analysis.model import PriorityEvidencePath, PriorityMarketCandidate
from analysis.review.classification import PrincipalPathBinding
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
    canonical_legacy_decision_payload,
)

from .errors import BundleValidationError


CURRENT_PREGAME_SCHEMA_VERSION = 2
CURRENT_PREGAME_AUTHORITY = "current"


@dataclass(frozen=True)
class _IdentityAuthority:
    value_ref: ReferenceType[object]
    issued_type: type[object]
    graph: tuple[object, ...]
    digest: bytes


@dataclass(frozen=True)
class _ValidationEvent:
    value: object


_AuthorityRegistry = dict[int, _IdentityAuthority]
_DECISION_AUTHORITIES: _AuthorityRegistry = {}
_PAYLOAD_AUTHORITIES: _AuthorityRegistry = {}
_BUNDLE_AUTHORITIES: _AuthorityRegistry = {}
_VALIDATION_EVENTS: dict[int, ReferenceType[_ValidationEvent]] = {}
_AUTHORITY_LOCK = RLock()


def _issue_validation_event(value: object) -> _ValidationEvent:
    event = _ValidationEvent(value)
    identity = id(event)

    def remove(dead_ref: ReferenceType[_ValidationEvent]) -> None:
        with _AUTHORITY_LOCK:
            current = _VALIDATION_EVENTS.get(identity)
            if current is dead_ref:
                _VALIDATION_EVENTS.pop(identity, None)

    event_ref = ref(event, remove)
    with _AUTHORITY_LOCK:
        _VALIDATION_EVENTS[identity] = event_ref
    return event


def _consume_validation_event(event: object) -> bool:
    if type(event) is not _ValidationEvent:
        return False
    with _AUTHORITY_LOCK:
        event_ref = _VALIDATION_EVENTS.get(id(event))
        if event_ref is None or event_ref() is not event:
            return False
        _VALIDATION_EVENTS.pop(id(event), None)
        return True


def _identity_graph(root: object) -> tuple[object, ...]:
    """Capture the exact reachable public graph without retaining its root."""

    graph: list[object] = []
    seen = {id(root)}

    def visit(value: object) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        graph.append(value)

        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                visit(getattr(value, name))
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    if isinstance(root, BaseModel):
        for name in type(root).model_fields:
            visit(getattr(root, name))
    elif is_dataclass(root) and not isinstance(root, type):
        for field in fields(root):
            visit(getattr(root, field.name))
    return tuple(graph)


def _register_authority(
    registry: _AuthorityRegistry,
    value: object,
    digest: bytes,
) -> None:
    identity = id(value)
    graph = _identity_graph(value)

    def remove(dead_ref: ReferenceType[object]) -> None:
        with _AUTHORITY_LOCK:
            current = registry.get(identity)
            if current is not None and current.value_ref is dead_ref:
                registry.pop(identity, None)

    value_ref = ref(value, remove)
    authority = _IdentityAuthority(
        value_ref=value_ref,
        issued_type=type(value),
        graph=graph,
        digest=digest,
    )
    with _AUTHORITY_LOCK:
        registry[identity] = authority


def _has_authority(
    registry: _AuthorityRegistry,
    value: object,
    digest: bytes,
) -> bool:
    try:
        graph = _identity_graph(value)
    except (AttributeError, TypeError, ValueError):
        return False
    with _AUTHORITY_LOCK:
        authority = registry.get(id(value))
        if (
            authority is None
            or authority.value_ref() is not value
            or authority.issued_type is not type(value)
        ):
            return False
        return (
            len(authority.graph) == len(graph)
            and all(
                registered is current
                for registered, current in zip(authority.graph, graph)
            )
            and hmac.compare_digest(authority.digest, digest)
        )


class _CurrentLegacyDecisionSnapshot(LegacyDecisionSnapshot):
    """Strict current decision domain value issued by explicit validation."""


def _authority_digest(value: object) -> bytes:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).digest()


def _decision_authority_digest(value: object) -> bytes:
    return _authority_digest(
        {
            "authority": CURRENT_PREGAME_AUTHORITY,
            "schema_version": CURRENT_PREGAME_SCHEMA_VERSION,
            "legacy_decision": canonical_legacy_decision_payload(value),
        }
    )


def _payload_authority_digest(
    game_input: object,
    legacy_decision: object,
    forecast_config_version: object,
) -> bytes:
    return _authority_digest(
        {
            "game_input": capture_payload(game_input),
            "legacy_decision": canonical_legacy_decision_payload(
                legacy_decision
            ),
            "forecast_config_version": forecast_config_version,
        }
    )


def _bundle_authority_digest(
    *,
    schema_version: object,
    authority: object,
    stage: object,
    game_pk: object,
    created_at: object,
    payload: object,
) -> bytes:
    return _authority_digest(
        {
            "schema_version": schema_version,
            "authority": authority,
            "stage": stage,
            "game_pk": game_pk,
            "created_at": created_at,
            "payload_digest": _payload_authority_digest(
                payload.game_input,
                payload.legacy_decision,
                payload.forecast_config_version,
            ).hex(),
        }
    )


def _has_current_decision_authority(value: object) -> bool:
    if type(value) is not _CurrentLegacyDecisionSnapshot:
        return False
    try:
        if (
            type(value.model_version) is not str
            or not value.model_version.strip()
            or type(value.candidates) is not tuple
            or type(value.final_priority) is not tuple
            or type(value.review_bindings) is not tuple
            or type(value.notes) is not tuple
            or any(
                type(candidate) is not PriorityMarketCandidate
                for candidate in value.candidates
            )
            or any(
                type(priority) is not str
                for priority in value.final_priority
            )
            or any(
                type(binding) is not CandidateReviewBinding
                for binding in value.review_bindings
            )
            or any(type(note) is not str for note in value.notes)
        ):
            return False
        digest = _decision_authority_digest(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return _has_authority(_DECISION_AUTHORITIES, value, digest)


class PregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    game_input: NormalizedGameInput
    legacy_decision: LegacyDecisionSnapshot
    forecast_config_version: str

    @field_validator("forecast_config_version")
    @classmethod
    def _require_config_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("forecast_config_version must be nonempty")
        return value


class _BundleEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    game_pk: int
    created_at: datetime

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_json_integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be a JSON integer")
        return value

    @field_validator("created_at")
    @classmethod
    def _require_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class _PregameBundle(_BundleEnvelope):
    payload: PregamePayload

    @model_validator(mode="after")
    def _require_matching_game_pk(self) -> "_PregameBundle":
        if self.game_pk != self.payload.game_input.identity.game_pk:
            raise ValueError("game_pk must match payload game_input identity")
        return self


class _CurrentPriorityCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market: str
    score: int
    decision_rating: str
    evidence_paths: frozenset[PriorityEvidencePath]
    same_game_pair_compatible: StrictBool


def _current_candidate(
    value: _CurrentPriorityCandidateInput,
) -> PriorityMarketCandidate:
    return PriorityMarketCandidate(
        market=value.market,
        score=value.score,
        decision_rating=value.decision_rating,
        evidence_paths=value.evidence_paths,
        same_game_pair_compatible=value.same_game_pair_compatible,
    )


_CurrentPriorityCandidate = Annotated[
    PriorityMarketCandidate,
    ValidateAs(_CurrentPriorityCandidateInput, _current_candidate),
]


class _CurrentLegacyDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    candidates: tuple[_CurrentPriorityCandidate, ...]
    final_priority: tuple[str, ...]
    review_bindings: tuple[CandidateReviewBinding, ...]
    notes: tuple[str, ...] = ()

    @field_validator("candidates", mode="before")
    @classmethod
    def _require_explicit_candidate_mappings(
        cls,
        value: object,
    ) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        if any(not isinstance(candidate, Mapping) for candidate in value):
            raise ValueError(
                "current candidates must be supplied as explicit mappings"
            )
        normalized = []
        for candidate in value:
            candidate_mapping = dict(candidate)
            evidence_paths = candidate_mapping.get("evidence_paths")
            if isinstance(evidence_paths, (list, tuple, set, frozenset)):
                candidate_mapping["evidence_paths"] = frozenset(evidence_paths)
            normalized.append(candidate_mapping)
        return tuple(normalized)


def _current_legacy(
    value: _CurrentLegacyDecisionInput,
) -> LegacyDecisionSnapshot:
    decision = _CurrentLegacyDecisionSnapshot(
        model_version=value.model_version,
        candidates=value.candidates,
        final_priority=value.final_priority,
        review_bindings=value.review_bindings,
        notes=value.notes,
    )
    _register_authority(
        _DECISION_AUTHORITIES,
        decision,
        _decision_authority_digest(decision),
    )
    return decision


_CurrentLegacyDecision = Annotated[
    LegacyDecisionSnapshot,
    ValidateAs(_CurrentLegacyDecisionInput, _current_legacy),
]


def _historical_strict_legacy(
    value: _CurrentLegacyDecisionInput,
) -> LegacyDecisionSnapshot:
    return LegacyDecisionSnapshot(
        model_version=value.model_version,
        candidates=value.candidates,
        final_priority=value.final_priority,
        review_bindings=value.review_bindings,
        notes=value.notes,
    )


_HistoricalStrictLegacyDecision = Annotated[
    LegacyDecisionSnapshot,
    ValidateAs(_CurrentLegacyDecisionInput, _historical_strict_legacy),
]


class _HistoricalRelaxedPriorityCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    market: str
    score: int
    decision_rating: str
    evidence_paths: frozenset[PriorityEvidencePath]


def _historical_relaxed_candidate(
    value: _HistoricalRelaxedPriorityCandidateInput,
) -> PriorityMarketCandidate:
    return PriorityMarketCandidate(
        market=value.market,
        score=value.score,
        decision_rating=value.decision_rating,
        evidence_paths=value.evidence_paths,
        same_game_pair_compatible=False,
    )


_HistoricalRelaxedPriorityCandidate = Annotated[
    PriorityMarketCandidate,
    ValidateAs(
        _HistoricalRelaxedPriorityCandidateInput,
        _historical_relaxed_candidate,
    ),
]


class _HistoricalRelaxedLegacyDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    candidates: tuple[_HistoricalRelaxedPriorityCandidate, ...]
    final_priority: tuple[str, ...]
    review_bindings: tuple[CandidateReviewBinding, ...]
    notes: tuple[str, ...] = ()


def _historical_relaxed_legacy(
    value: _HistoricalRelaxedLegacyDecisionInput,
) -> LegacyDecisionSnapshot:
    return LegacyDecisionSnapshot(
        model_version=value.model_version,
        candidates=value.candidates,
        final_priority=value.final_priority,
        review_bindings=value.review_bindings,
        notes=value.notes,
    )


_HistoricalRelaxedLegacyDecision = Annotated[
    LegacyDecisionSnapshot,
    ValidateAs(
        _HistoricalRelaxedLegacyDecisionInput,
        _historical_relaxed_legacy,
    ),
]


class _CurrentPregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    game_input: NormalizedGameInput
    legacy_decision: _CurrentLegacyDecision
    forecast_config_version: str

    @field_validator("legacy_decision", mode="before")
    @classmethod
    def _require_raw_current_decision_mapping(cls, value: object) -> object:
        if isinstance(value, _CurrentLegacyDecisionInput) or not isinstance(
            value,
            Mapping,
        ):
            raise ValueError(
                "current legacy_decision must be supplied as a raw mapping"
            )
        normalized = dict(value)
        for field_name in ("final_priority", "notes"):
            field_value = normalized.get(field_name)
            if isinstance(field_value, list):
                normalized[field_name] = tuple(field_value)

        review_bindings = normalized.get("review_bindings")
        if isinstance(review_bindings, (list, tuple)):
            normalized_bindings = []
            for binding in review_bindings:
                if not isinstance(binding, Mapping):
                    normalized_bindings.append(binding)
                    continue
                binding_mapping = dict(binding)
                principal_paths = binding_mapping.get("principal_paths")
                if isinstance(principal_paths, (list, tuple)):
                    normalized_paths = []
                    for path in principal_paths:
                        if not isinstance(path, Mapping):
                            normalized_paths.append(path)
                            continue
                        path_mapping = dict(path)
                        component_ids = path_mapping.get("component_ids")
                        if isinstance(component_ids, list):
                            path_mapping["component_ids"] = tuple(component_ids)
                        normalized_paths.append(path_mapping)
                    binding_mapping["principal_paths"] = tuple(
                        normalized_paths
                    )
                normalized_bindings.append(binding_mapping)
            normalized["review_bindings"] = tuple(normalized_bindings)
        return normalized

    @field_validator("forecast_config_version")
    @classmethod
    def _require_config_version(cls, value: str) -> object:
        if not value.strip():
            raise ValueError("forecast_config_version must be nonempty")
        return _issue_validation_event(value)

    @model_validator(mode="after")
    def _issue_validation_authority(self) -> "_CurrentPregamePayload":
        event = self.forecast_config_version
        if type(event) is not _ValidationEvent:
            return self
        if not _consume_validation_event(event):
            raise ValueError("current payload validation event is invalid")
        object.__setattr__(self, "forecast_config_version", event.value)
        if not _has_current_decision_authority(self.legacy_decision):
            raise ValueError("current decision authority is invalid")
        _register_authority(
            _PAYLOAD_AUTHORITIES,
            self,
            _payload_authority_digest(
                self.game_input,
                self.legacy_decision,
                self.forecast_config_version,
            ),
        )
        return self


def is_current_pregame_payload(value: object) -> bool:
    """Return whether value retains its validated current-runtime authority."""

    if type(value) is not _CurrentPregamePayload:
        return False
    try:
        if (
            type(value.game_input) is not NormalizedGameInput
            or not _has_current_decision_authority(value.legacy_decision)
            or type(value.forecast_config_version) is not str
            or not value.forecast_config_version.strip()
        ):
            return False
        digest = _payload_authority_digest(
            value.game_input,
            value.legacy_decision,
            value.forecast_config_version,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return _has_authority(_PAYLOAD_AUTHORITIES, value, digest)


_CurrentPregamePayloadSource = (
    InstanceOf[_CurrentPregamePayload]
    | SkipJsonSchema[Mapping[str, object]]
)


def _legacy_candidate_mappings(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    legacy_decision = payload.get("legacy_decision")
    if not isinstance(legacy_decision, dict):
        return []
    candidates = legacy_decision.get("candidates")
    if not isinstance(candidates, (list, tuple)):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


class _HistoricalRelaxedPregamePayload(PregamePayload):
    legacy_decision: _HistoricalRelaxedLegacyDecision

    @model_serializer(mode="wrap")
    def _preserve_omitted_compatibility(self, handler):
        payload = handler(self)
        for candidate in _legacy_candidate_mappings(payload):
            candidate.pop("same_game_pair_compatible", None)
        return payload


class _HistoricalStrictPregamePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    game_input: NormalizedGameInput
    legacy_decision: _HistoricalStrictLegacyDecision
    forecast_config_version: str

    @field_validator("forecast_config_version")
    @classmethod
    def _require_config_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("forecast_config_version must be nonempty")
        return value


class _CurrentPregameBundle(_PregameBundle):
    schema_version: Literal[2]
    authority: Literal["current"]
    payload: _CurrentPregamePayloadSource = Field(union_mode="left_to_right")

    @field_validator("authority", mode="after")
    @classmethod
    def _mark_current_validation(cls, value: str) -> object:
        return _issue_validation_event(value)

    @field_validator("payload", mode="before")
    @classmethod
    def _require_raw_current_payload_mapping(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise ValueError(
                "current payload must be supplied as a raw mapping"
            )
        return value

    @field_validator("payload", mode="after")
    @classmethod
    def _materialize_current_payload(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> _CurrentPregamePayload:
        if type(value) is _CurrentPregamePayload:
            if not is_current_pregame_payload(value):
                raise ValueError("current payload authority is invalid")
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                "current payload must be supplied as a raw mapping"
            )
        if info.mode == "json":
            return _CurrentPregamePayload.model_validate_json(
                json.dumps(value),
                strict=True,
            )
        return _CurrentPregamePayload.model_validate(value)

    @model_validator(mode="after")
    def _issue_validation_authority(self) -> "_CurrentPregameBundle":
        event = self.authority
        if type(event) is not _ValidationEvent:
            return self
        if not _consume_validation_event(event):
            raise ValueError("current bundle validation event is invalid")
        object.__setattr__(self, "authority", event.value)
        if (
            not is_current_pregame_payload(self.payload)
            or self.game_pk != self.payload.game_input.identity.game_pk
        ):
            raise ValueError("current bundle authority is invalid")
        _register_authority(
            _BUNDLE_AUTHORITIES,
            self,
            _bundle_authority_digest(
                schema_version=self.schema_version,
                authority=self.authority,
                stage=self.stage,
                game_pk=self.game_pk,
                created_at=self.created_at,
                payload=self.payload,
            ),
        )
        return self

    @field_serializer("payload", return_type=_CurrentPregamePayload)
    def _serialize_current_payload(
        self,
        value: _CurrentPregamePayload,
    ) -> _CurrentPregamePayload:
        return value


class _HistoricalPregameBundleV1(_PregameBundle):
    schema_version: Literal[1]


class ProjectedBundle(_CurrentPregameBundle):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["projected"]


class ConfirmedBundle(_CurrentPregameBundle):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["confirmed"]


def is_current_pregame_bundle(
    value: object,
    *,
    expected_type: type[ProjectedBundle] | type[ConfirmedBundle],
    expected_stage: Literal["projected", "confirmed"],
) -> bool:
    """Verify raw-validation authority for the exact current object graph."""

    return (
        _current_pregame_bundle_authority_fingerprint(
            value,
            expected_type=expected_type,
            expected_stage=expected_stage,
        )
        is not None
    )


def _current_pregame_bundle_authority_fingerprint(
    value: object,
    *,
    expected_type: type[ProjectedBundle] | type[ConfirmedBundle],
    expected_stage: Literal["projected", "confirmed"],
) -> bytes | None:
    """Return the authenticated value fingerprint for an exact runtime graph."""

    if not _has_current_bundle_invariants(
        value,
        expected_type=expected_type,
        expected_stage=expected_stage,
    ):
        return None
    try:
        digest = _bundle_authority_digest(
            schema_version=value.schema_version,
            authority=value.authority,
            stage=value.stage,
            game_pk=value.game_pk,
            created_at=value.created_at,
            payload=value.payload,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if not _has_authority(_BUNDLE_AUTHORITIES, value, digest):
        return None
    return digest


def _has_current_bundle_invariants(
    value: object,
    *,
    expected_type: type[ProjectedBundle] | type[ConfirmedBundle],
    expected_stage: Literal["projected", "confirmed"],
) -> bool:
    if type(value) is not expected_type:
        return False
    try:
        return not (
            type(value.schema_version) is not int
            or value.schema_version != CURRENT_PREGAME_SCHEMA_VERSION
            or type(value.authority) is not str
            or value.authority != CURRENT_PREGAME_AUTHORITY
            or type(value.stage) is not str
            or value.stage != expected_stage
            or type(value.game_pk) is not int
            or type(value.created_at) is not datetime
            or value.created_at.tzinfo is None
            or value.created_at.utcoffset() is None
            or not is_current_pregame_payload(value.payload)
            or type(value.payload.game_input.identity.game_pk) is not int
            or value.game_pk != value.payload.game_input.identity.game_pk
        )
    except (AttributeError, TypeError, ValueError):
        return False


class HistoricalRelaxedProjectedBundleV1(_HistoricalPregameBundleV1):
    """Replay-only projected v1 bundle without compatibility fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["projected"]
    payload: _HistoricalRelaxedPregamePayload


class HistoricalStrictProjectedBundleV1(_HistoricalPregameBundleV1):
    """Replay-only projected v1 bundle with compatibility fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["projected"]
    payload: _HistoricalStrictPregamePayload


class HistoricalRelaxedConfirmedBundleV1(_HistoricalPregameBundleV1):
    """Replay-only confirmed v1 bundle without compatibility fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["confirmed"]
    payload: _HistoricalRelaxedPregamePayload


class HistoricalStrictConfirmedBundleV1(_HistoricalPregameBundleV1):
    """Replay-only confirmed v1 bundle with compatibility fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["confirmed"]
    payload: _HistoricalStrictPregamePayload


# Task 2 replaces these compatibility aliases in bundle/service consumers.
HistoricalProjectedBundle = HistoricalRelaxedProjectedBundleV1
HistoricalConfirmedBundle = HistoricalRelaxedConfirmedBundleV1


class ReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capture_id: str
    forecast_run_id: str
    capture_forecast_link_record_id: str
    official_result: OfficialResultCommand
    closing_observations: tuple[MarketObservationCommand, ...] = ()

    @field_validator(
        "capture_id",
        "forecast_run_id",
        "capture_forecast_link_record_id",
    )
    @classmethod
    def _require_exact_nonempty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("frozen record identifier must be nonempty")
        return value

    @model_validator(mode="after")
    def _require_closing_terms(self) -> "ReviewPayload":
        for observation in self.closing_observations:
            if observation.observation_kind != "close":
                raise ValueError(
                    "closing_observations observation_kind must be 'close'"
                )
            for field_name in (
                "selected_decimal_price",
                "opposing_decimal_price",
            ):
                price = getattr(observation, field_name)
                if price is None or price <= 1.0:
                    raise ValueError(
                        f"closing_observations {field_name} must be above 1"
                    )
            available_at = observation.available_at
            if (
                available_at is None
                or available_at.tzinfo is None
                or available_at.utcoffset() is None
            ):
                raise ValueError(
                    "closing_observations available_at must be timezone-aware"
                )
        return self


class ReviewBundle(_BundleEnvelope):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    stage: Literal["review"]
    payload: ReviewPayload

    @model_validator(mode="after")
    def _require_matching_game_pk(self) -> "ReviewBundle":
        if self.game_pk != self.payload.official_result.game_pk:
            raise ValueError("game_pk must match payload official_result")
        if any(
            observation.game_pk != self.game_pk
            for observation in self.payload.closing_observations
        ):
            raise ValueError("closing observation game_pk must match envelope")
        return self


_PROJECTED_ADAPTER = TypeAdapter(ProjectedBundle)
_CONFIRMED_ADAPTER = TypeAdapter(ConfirmedBundle)
_HISTORICAL_RELAXED_PROJECTED_ADAPTER = TypeAdapter(
    HistoricalRelaxedProjectedBundleV1
)
_HISTORICAL_STRICT_PROJECTED_ADAPTER = TypeAdapter(
    HistoricalStrictProjectedBundleV1
)
_HISTORICAL_RELAXED_CONFIRMED_ADAPTER = TypeAdapter(
    HistoricalRelaxedConfirmedBundleV1
)
_HISTORICAL_STRICT_CONFIRMED_ADAPTER = TypeAdapter(
    HistoricalStrictConfirmedBundleV1
)
_REVIEW_ADAPTER = TypeAdapter(ReviewBundle)
_BundleType = TypeVar("_BundleType", bound=BaseModel)
_AuditPresenceCheck = Callable[[object], None]
_KNOWN_BUNDLE_STAGES = frozenset(("projected", "confirmed", "review"))
_LEGACY_SNAPSHOT_FIELDS = tuple(
    field.name for field in fields(LegacyDecisionSnapshot)
)
_LEGACY_CANDIDATE_FIELDS = tuple(
    field.name for field in fields(PriorityMarketCandidate)
)
_HISTORICAL_LEGACY_CANDIDATE_FIELDS = tuple(
    field_name
    for field_name in _LEGACY_CANDIDATE_FIELDS
    if field_name != "same_game_pair_compatible"
)
_LEGACY_BINDING_FIELDS = tuple(
    field.name for field in fields(CandidateReviewBinding)
)
_PRINCIPAL_PATH_FIELDS = tuple(
    field.name for field in fields(PrincipalPathBinding)
)
_SAFE_VALIDATION_CATEGORIES = {
    "assertion_error": "invalid_value",
    "bool_parsing": "invalid_boolean",
    "bool_type": "invalid_boolean",
    "bytes_type": "invalid_bytes",
    "date_from_datetime_inexact": "invalid_date",
    "date_from_datetime_parsing": "invalid_date",
    "date_parsing": "invalid_date",
    "date_type": "invalid_date",
    "datetime_from_date_parsing": "invalid_datetime",
    "datetime_parsing": "invalid_datetime",
    "datetime_type": "invalid_datetime",
    "decimal_parsing": "invalid_number",
    "decimal_type": "invalid_number",
    "dict_type": "invalid_object",
    "enum": "invalid_choice",
    "extra_forbidden": "extra_forbidden",
    "finite_number": "invalid_number",
    "float_parsing": "invalid_number",
    "float_type": "invalid_number",
    "frozenset_type": "invalid_list",
    "greater_than": "invalid_value",
    "greater_than_equal": "invalid_value",
    "int_from_float": "invalid_integer",
    "int_parsing": "invalid_integer",
    "int_parsing_size": "invalid_integer",
    "int_type": "invalid_integer",
    "is_instance_of": "invalid_type",
    "less_than": "invalid_value",
    "less_than_equal": "invalid_value",
    "list_type": "invalid_list",
    "literal_error": "invalid_choice",
    "mapping_type": "invalid_object",
    "missing": "missing",
    "model_attributes_type": "invalid_object",
    "set_type": "invalid_list",
    "string_type": "invalid_string",
    "time_parsing": "invalid_time",
    "time_type": "invalid_time",
    "timezone_aware": "invalid_timezone",
    "timezone_naive": "invalid_timezone",
    "tuple_type": "invalid_list",
    "union_tag_invalid": "invalid_choice",
    "union_tag_not_found": "invalid_choice",
    "url_parsing": "invalid_url",
    "url_scheme": "invalid_url",
    "url_type": "invalid_url",
    "value_error": "invalid_value",
}


def _contains_dynamic_mapping_value(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    if "additionalProperties" in schema:
        return True
    items = schema.get("items")
    if isinstance(items, dict) and "additionalProperties" in items:
        return True
    for keyword in ("allOf", "anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list) and any(
            _contains_dynamic_mapping_value(option) for option in options
        ):
            return True
    return False


def _schema_location_policy(
    adapter: TypeAdapter[object],
) -> tuple[frozenset[str], frozenset[str]]:
    trusted_fields: set[str] = set()
    dynamic_mapping_fields: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                for field_name, field_schema in properties.items():
                    trusted_fields.add(field_name)
                    if _contains_dynamic_mapping_value(field_schema):
                        dynamic_mapping_fields.add(field_name)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(adapter.json_schema())
    return frozenset(trusted_fields), frozenset(dynamic_mapping_fields)


(
    _PROJECTED_TRUSTED_FIELDS,
    _PROJECTED_DYNAMIC_MAPPING_FIELDS,
) = _schema_location_policy(_PROJECTED_ADAPTER)
(
    _CONFIRMED_TRUSTED_FIELDS,
    _CONFIRMED_DYNAMIC_MAPPING_FIELDS,
) = _schema_location_policy(_CONFIRMED_ADAPTER)
(
    _REVIEW_TRUSTED_FIELDS,
    _REVIEW_DYNAMIC_MAPPING_FIELDS,
) = _schema_location_policy(_REVIEW_ADAPTER)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    del value
    raise ValueError("invalid JSON number")


def _require_explicit_fields(
    value: object,
    field_names: tuple[str, ...],
    path: str,
) -> None:
    if not isinstance(value, dict):
        return
    for field_name in field_names:
        if field_name not in value:
            raise BundleValidationError(
                f"missing explicit bundle field: {path}.{field_name}"
            )


def _bundle_payload(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("payload")
    return payload if isinstance(payload, dict) else None


def _raw_candidate_mappings(value: object) -> list[dict[str, object]]:
    payload = _bundle_payload(value)
    return _legacy_candidate_mappings(payload)


def _historical_candidate_contract(
    value: object,
) -> Literal["relaxed", "strict"]:
    candidates = _raw_candidate_mappings(value)
    presence = tuple(
        "same_game_pair_compatible" in candidate for candidate in candidates
    )
    if not presence:
        return "relaxed"
    if all(presence):
        return "strict"
    if not any(presence):
        return "relaxed"
    raise BundleValidationError(
        "invalid historical bundle: mixed compatibility contract"
    )


def _require_pregame_audit_fields(value: object) -> None:
    _require_pregame_audit_fields_with_candidates(
        value,
        _LEGACY_CANDIDATE_FIELDS,
    )


def _require_historical_relaxed_pregame_audit_fields(value: object) -> None:
    _require_pregame_audit_fields_with_candidates(
        value,
        _HISTORICAL_LEGACY_CANDIDATE_FIELDS,
    )


def _require_historical_strict_pregame_audit_fields(value: object) -> None:
    _require_pregame_audit_fields_with_candidates(
        value,
        _LEGACY_CANDIDATE_FIELDS,
    )


def _require_pregame_audit_fields_with_candidates(
    value: object,
    candidate_fields: tuple[str, ...],
) -> None:
    payload = _bundle_payload(value)
    if payload is None:
        return
    legacy_decision = payload.get("legacy_decision")
    if not isinstance(legacy_decision, dict):
        return
    legacy_path = "payload.legacy_decision"
    _require_explicit_fields(
        legacy_decision,
        _LEGACY_SNAPSHOT_FIELDS,
        legacy_path,
    )

    candidates = legacy_decision.get("candidates")
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates):
            _require_explicit_fields(
                candidate,
                candidate_fields,
                f"{legacy_path}.candidates[{index}]",
            )

    bindings = legacy_decision.get("review_bindings")
    if not isinstance(bindings, list):
        return
    for binding_index, binding in enumerate(bindings):
        binding_path = f"{legacy_path}.review_bindings[{binding_index}]"
        _require_explicit_fields(
            binding,
            _LEGACY_BINDING_FIELDS,
            binding_path,
        )
        if not isinstance(binding, dict):
            continue
        principal_paths = binding.get("principal_paths")
        if not isinstance(principal_paths, list):
            continue
        for path_index, principal_path in enumerate(principal_paths):
            _require_explicit_fields(
                principal_path,
                _PRINCIPAL_PATH_FIELDS,
                f"{binding_path}.principal_paths[{path_index}]",
            )


def _require_review_audit_fields(value: object) -> None:
    payload = _bundle_payload(value)
    if payload is None:
        return
    _require_explicit_fields(
        payload.get("official_result"),
        ("supersedes_record_id",),
        "payload.official_result",
    )
    closing_observations = payload.get("closing_observations")
    if not isinstance(closing_observations, list):
        return
    for index, observation in enumerate(closing_observations):
        _require_explicit_fields(
            observation,
            ("available_at", "supersedes_record_id"),
            f"payload.closing_observations[{index}]",
        )


def _raw_json_bytes(raw: str | bytes | bytearray) -> tuple[bytes, str]:
    if isinstance(raw, str):
        try:
            return raw.encode("utf-8"), raw
        except UnicodeEncodeError as error:
            raise BundleValidationError(
                "bundle JSON must be valid UTF-8"
            ) from error
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)
        try:
            return raw_bytes, raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BundleValidationError(
                "bundle JSON must be valid UTF-8"
            ) from error
    raise BundleValidationError("bundle JSON must be raw UTF-8 text or bytes")


def _decode_bundle_json(
    raw: str | bytes | bytearray,
) -> tuple[bytes, object]:
    raw_bytes, decoded = _raw_json_bytes(raw)
    try:
        decoded_value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise BundleValidationError(
            "invalid bundle JSON: malformed JSON"
        ) from error
    except ValueError as error:
        raise BundleValidationError(f"invalid bundle JSON: {error}") from error
    return raw_bytes, decoded_value


def bundle_stage_from_json(raw: str | bytes | bytearray) -> str:
    """Return a recognized envelope stage without validating its payload."""

    _, decoded_value = _decode_bundle_json(raw)
    if not isinstance(decoded_value, dict):
        raise BundleValidationError("invalid bundle: stage envelope")
    stage = decoded_value.get("stage")
    if not isinstance(stage, str) or stage not in _KNOWN_BUNDLE_STAGES:
        raise BundleValidationError("invalid bundle: stage envelope")
    return stage


def _validation_location(
    location: tuple[object, ...],
    *,
    trusted_fields: frozenset[str],
    dynamic_mapping_fields: frozenset[str],
    force_final_placeholder: bool,
) -> str:
    rendered = ""
    dynamic_segment_follows = False
    for index, part in enumerate(location):
        if type(part) is int:
            rendered += f"[{part}]"
            continue
        is_forced_placeholder = (
            force_final_placeholder and index == len(location) - 1
        )
        is_trusted = (
            isinstance(part, str)
            and part in trusted_fields
            and not dynamic_segment_follows
            and not is_forced_placeholder
        )
        safe_part = part if is_trusted else "<field>"
        rendered += ("." if rendered else "") + safe_part
        dynamic_segment_follows = (
            is_trusted and part in dynamic_mapping_fields
        )
    return rendered or "bundle"


def _safe_validation_diagnostics(
    error: ValidationError,
    *,
    trusted_fields: frozenset[str],
    dynamic_mapping_fields: frozenset[str],
) -> str:
    diagnostics: list[str] = []
    for detail in error.errors(include_input=False, include_url=False):
        raw_location = detail.get("loc", ())
        raw_type = detail.get("type")
        location = _validation_location(
            tuple(raw_location) if isinstance(raw_location, tuple) else (),
            trusted_fields=trusted_fields,
            dynamic_mapping_fields=dynamic_mapping_fields,
            force_final_placeholder=raw_type == "extra_forbidden",
        )
        category = _SAFE_VALIDATION_CATEGORIES.get(
            raw_type,
            "validation_error",
        )
        if raw_type == "value_error" and location.endswith("schema_version"):
            category = "invalid_integer"
        elif raw_type == "value_error" and location.endswith("created_at"):
            category = "invalid_timezone"
        diagnostics.append(f"{location}: {category}")
    return "; ".join(diagnostics) or "bundle validation failed"


def _parse_bundle_json(
    raw: str | bytes | bytearray,
    adapter: TypeAdapter[_BundleType],
    audit_presence_check: _AuditPresenceCheck,
    trusted_fields: frozenset[str],
    dynamic_mapping_fields: frozenset[str],
) -> _BundleType:
    raw_bytes, decoded_value = _decode_bundle_json(raw)
    audit_presence_check(decoded_value)

    try:
        return adapter.validate_json(raw_bytes, strict=True)
    except ValidationError as error:
        raise BundleValidationError(
            "invalid bundle: "
            + _safe_validation_diagnostics(
                error,
                trusted_fields=trusted_fields,
                dynamic_mapping_fields=dynamic_mapping_fields,
            )
        ) from error
    except (TypeError, ValueError) as error:
        raise BundleValidationError(
            f"invalid bundle: {error.__class__.__name__}"
        ) from error


def parse_projected_bundle_json(
    raw: str | bytes | bytearray,
) -> ProjectedBundle:
    return _parse_bundle_json(
        raw,
        _PROJECTED_ADAPTER,
        _require_pregame_audit_fields,
        _PROJECTED_TRUSTED_FIELDS,
        _PROJECTED_DYNAMIC_MAPPING_FIELDS,
    )


def parse_confirmed_bundle_json(
    raw: str | bytes | bytearray,
) -> ConfirmedBundle:
    return _parse_bundle_json(
        raw,
        _CONFIRMED_ADAPTER,
        _require_pregame_audit_fields,
        _CONFIRMED_TRUSTED_FIELDS,
        _CONFIRMED_DYNAMIC_MAPPING_FIELDS,
    )


def parse_review_bundle_json(
    raw: str | bytes | bytearray,
) -> ReviewBundle:
    return _parse_bundle_json(
        raw,
        _REVIEW_ADAPTER,
        _require_review_audit_fields,
        _REVIEW_TRUSTED_FIELDS,
        _REVIEW_DYNAMIC_MAPPING_FIELDS,
    )


def parse_historical_projected_bundle_json(
    raw: str | bytes | bytearray,
) -> HistoricalRelaxedProjectedBundleV1 | HistoricalStrictProjectedBundleV1:
    """Parse a projected schema-v1 bundle without changing its shape."""

    _, decoded_value = _decode_bundle_json(raw)
    if _historical_candidate_contract(decoded_value) == "strict":
        return _parse_bundle_json(
            raw,
            _HISTORICAL_STRICT_PROJECTED_ADAPTER,
            _require_historical_strict_pregame_audit_fields,
            _PROJECTED_TRUSTED_FIELDS,
            _PROJECTED_DYNAMIC_MAPPING_FIELDS,
        )
    return _parse_bundle_json(
        raw,
        _HISTORICAL_RELAXED_PROJECTED_ADAPTER,
        _require_historical_relaxed_pregame_audit_fields,
        _PROJECTED_TRUSTED_FIELDS,
        _PROJECTED_DYNAMIC_MAPPING_FIELDS,
    )


def parse_historical_confirmed_bundle_json(
    raw: str | bytes | bytearray,
) -> HistoricalRelaxedConfirmedBundleV1 | HistoricalStrictConfirmedBundleV1:
    """Parse a confirmed schema-v1 bundle without changing its shape."""

    _, decoded_value = _decode_bundle_json(raw)
    if _historical_candidate_contract(decoded_value) == "strict":
        return _parse_bundle_json(
            raw,
            _HISTORICAL_STRICT_CONFIRMED_ADAPTER,
            _require_historical_strict_pregame_audit_fields,
            _CONFIRMED_TRUSTED_FIELDS,
            _CONFIRMED_DYNAMIC_MAPPING_FIELDS,
        )
    return _parse_bundle_json(
        raw,
        _HISTORICAL_RELAXED_CONFIRMED_ADAPTER,
        _require_historical_relaxed_pregame_audit_fields,
        _CONFIRMED_TRUSTED_FIELDS,
        _CONFIRMED_DYNAMIC_MAPPING_FIELDS,
    )
