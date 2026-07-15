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
