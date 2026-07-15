"""Strict version-1 JSON bundle schemas for shadow operator stages."""

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
from analysis.shadow import LegacyDecisionSnapshot

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


def _raw_json_bytes(raw: str | bytes | bytearray) -> tuple[bytes, str]:
    if isinstance(raw, str):
        return raw.encode("utf-8"), raw
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)
        try:
            return raw_bytes, raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BundleValidationError(
                f"bundle JSON must be valid UTF-8: {error}"
            ) from error
    raise BundleValidationError("bundle JSON must be raw UTF-8 text or bytes")


def _parse_bundle_json(
    raw: str | bytes | bytearray,
    adapter: TypeAdapter[_BundleType],
) -> _BundleType:
    raw_bytes, decoded = _raw_json_bytes(raw)
    try:
        json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise BundleValidationError(f"invalid bundle JSON: {error}") from error

    try:
        return adapter.validate_json(raw_bytes, strict=True)
    except (ValidationError, TypeError, ValueError) as error:
        raise BundleValidationError(f"invalid bundle: {error}") from error


def parse_projected_bundle_json(
    raw: str | bytes | bytearray,
) -> ProjectedBundle:
    return _parse_bundle_json(raw, _PROJECTED_ADAPTER)


def parse_confirmed_bundle_json(
    raw: str | bytes | bytearray,
) -> ConfirmedBundle:
    return _parse_bundle_json(raw, _CONFIRMED_ADAPTER)


def parse_review_bundle_json(
    raw: str | bytes | bytearray,
) -> ReviewBundle:
    return _parse_bundle_json(raw, _REVIEW_ADAPTER)
