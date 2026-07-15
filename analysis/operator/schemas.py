"""Strict version-1 JSON bundle schemas for shadow operator stages."""

from collections.abc import Callable
from dataclasses import fields
from datetime import datetime
import json
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from analysis.forecasting import NormalizedGameInput
from analysis.ledger import MarketObservationCommand, OfficialResultCommand
from analysis.model import PriorityMarketCandidate
from analysis.review.classification import PrincipalPathBinding
from analysis.shadow import (
    CandidateReviewBinding,
    LegacyDecisionSnapshot,
)

from .errors import BundleValidationError


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

    schema_version: Literal[1]
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


class ProjectedBundle(_PregameBundle):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["projected"]


class ConfirmedBundle(_PregameBundle):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["confirmed"]


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
_LEGACY_BINDING_FIELDS = tuple(
    field.name for field in fields(CandidateReviewBinding)
)
_PRINCIPAL_PATH_FIELDS = tuple(
    field.name for field in fields(PrincipalPathBinding)
)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number: {value}")


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


def _require_pregame_audit_fields(value: object) -> None:
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
                _LEGACY_CANDIDATE_FIELDS,
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
                f"bundle JSON must be valid UTF-8: {error}"
            ) from error
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)
        try:
            return raw_bytes, raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BundleValidationError(
                f"bundle JSON must be valid UTF-8: {error}"
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
    except (json.JSONDecodeError, ValueError) as error:
        raise BundleValidationError(f"invalid bundle JSON: {error}") from error
    return raw_bytes, decoded_value


def bundle_stage_from_json(raw: str | bytes | bytearray) -> str | None:
    """Return a recognized envelope stage without validating its payload."""

    _, decoded_value = _decode_bundle_json(raw)
    if not isinstance(decoded_value, dict):
        return None
    stage = decoded_value.get("stage")
    return stage if stage in _KNOWN_BUNDLE_STAGES else None


def _validation_location(location: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in location) or "bundle"


def _safe_validation_diagnostics(error: ValidationError) -> str:
    diagnostics: list[str] = []
    for detail in error.errors(include_input=False, include_url=False):
        location = _validation_location(tuple(detail.get("loc", ())))
        message = str(detail.get("msg", "validation failed"))
        error_type = str(detail.get("type", "validation_error"))
        diagnostics.append(f"{location}: {message} [type={error_type}]")
    return "; ".join(diagnostics) or "bundle validation failed"


def _parse_bundle_json(
    raw: str | bytes | bytearray,
    adapter: TypeAdapter[_BundleType],
    audit_presence_check: _AuditPresenceCheck,
) -> _BundleType:
    raw_bytes, decoded_value = _decode_bundle_json(raw)
    audit_presence_check(decoded_value)

    try:
        return adapter.validate_json(raw_bytes, strict=True)
    except ValidationError as error:
        raise BundleValidationError(
            f"invalid bundle: {_safe_validation_diagnostics(error)}"
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
    )


def parse_confirmed_bundle_json(
    raw: str | bytes | bytearray,
) -> ConfirmedBundle:
    return _parse_bundle_json(
        raw,
        _CONFIRMED_ADAPTER,
        _require_pregame_audit_fields,
    )


def parse_review_bundle_json(
    raw: str | bytes | bytearray,
) -> ReviewBundle:
    return _parse_bundle_json(
        raw,
        _REVIEW_ADAPTER,
        _require_review_audit_fields,
    )
