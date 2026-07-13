"""Canonical payloads and deterministic identities for normalized forecasts."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math

from .inputs import FeatureProvenance, NormalizedGameInput, SourceObservation


def _mapping_key(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise TypeError("canonical mapping keys must be strings")
    return value


def _frozen_dataclass_fields(value: object) -> tuple[object, ...]:
    if isinstance(value, type) or not is_dataclass(value):
        raise TypeError(f"unsupported canonical type: {type(value).__name__}")
    parameters = getattr(type(value), "__dataclass_params__", None)
    if parameters is None or not parameters.frozen:
        raise TypeError("canonical dataclasses must be frozen")
    return fields(value)


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetime must be timezone-aware")
        utc_value = value.astimezone(timezone.utc).isoformat()
        return f"{utc_value[:-6]}Z"
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical floats must be finite")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in _frozen_dataclass_fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            canonical_key = _mapping_key(key)
            if canonical_key in converted:
                raise TypeError("canonical mapping keys must be unique strings")
            converted[canonical_key] = _canonical_value(item)
        return converted
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize supported values as deterministic compact JSON."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def capture_payload(game: NormalizedGameInput) -> dict[str, object]:
    """Return the complete normalized input, including all audit metadata."""

    if not isinstance(game, NormalizedGameInput):
        raise TypeError("game must be NormalizedGameInput")
    payload = _canonical_value(game)
    if not isinstance(payload, dict):
        raise TypeError("NormalizedGameInput must serialize to a mapping")
    return payload


_OMIT = object()


def _semantic_observation(observation: SourceObservation) -> dict[str, object]:
    return {
        "source_name": observation.source_name,
        "source_record": observation.source_record,
        "effective_date": _canonical_value(observation.effective_date),
        "content_hash": observation.content_hash,
    }


def _semantic_value(value: object) -> object:
    if isinstance(value, FeatureProvenance):
        return _OMIT
    if isinstance(value, SourceObservation):
        return _semantic_observation(value)
    if isinstance(value, Enum):
        return _semantic_value(value.value)
    if isinstance(value, datetime):
        return _canonical_value(value)
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _canonical_value(value)
    if is_dataclass(value) and not isinstance(value, type):
        converted: dict[str, object] = {}
        for field in _frozen_dataclass_fields(value):
            if isinstance(value, NormalizedGameInput) and field.name == "captured_at":
                continue
            field_value = getattr(value, field.name)
            if isinstance(value, NormalizedGameInput) and field.name == "observations":
                observations = [
                    _semantic_observation(observation)
                    for observation in field_value
                ]
                observations.sort(key=canonical_json)
                converted[field.name] = observations
                continue
            semantic_value = _semantic_value(field_value)
            if semantic_value is not _OMIT:
                converted[field.name] = semantic_value
        return converted
    if isinstance(value, (tuple, list)):
        converted_items = []
        for item in value:
            semantic_item = _semantic_value(item)
            if semantic_item is not _OMIT:
                converted_items.append(semantic_item)
        return converted_items
    if isinstance(value, Mapping):
        converted_mapping: dict[str, object] = {}
        for key, item in value.items():
            canonical_key = _mapping_key(key)
            if canonical_key in converted_mapping:
                raise TypeError("canonical mapping keys must be unique strings")
            semantic_item = _semantic_value(item)
            if semantic_item is not _OMIT:
                converted_mapping[canonical_key] = semantic_item
        return converted_mapping
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def semantic_forecast_payload(game: NormalizedGameInput) -> dict[str, object]:
    """Return only deterministic, behavior-changing forecast semantics."""

    if not isinstance(game, NormalizedGameInput):
        raise TypeError("game must be NormalizedGameInput")
    payload = _semantic_value(game)
    if not isinstance(payload, dict):
        raise TypeError("NormalizedGameInput must serialize to a mapping")
    return payload


def normalized_input_hash(game: NormalizedGameInput) -> str:
    """Hash the canonical semantic normalized input."""

    encoded = canonical_json(semantic_forecast_payload(game)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_forecast_run_id(
    game: NormalizedGameInput,
    model_version: str,
    simulation_config: Mapping[str, object],
) -> str:
    """Build a deterministic identity for one semantic forecast run."""

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    if not isinstance(simulation_config, Mapping):
        raise TypeError("simulation_config must be a mapping")
    payload = {
        "input": semantic_forecast_payload(game),
        "model_version": model_version,
        "simulation_config": simulation_config,
    }
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
