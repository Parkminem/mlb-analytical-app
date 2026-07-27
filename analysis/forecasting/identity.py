"""Canonical payloads and deterministic identities for normalized forecasts."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math
from types import UnionType
from typing import Union, get_args, get_origin

from .config import SimulationConfig
from .inputs import (
    FeatureProvenance,
    MarketType,
    NormalizedGameInput,
    SourceObservation,
)


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
        if value.is_integer():
            return int(value)
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


def _referenced_semantic_observations(
    game: NormalizedGameInput,
) -> list[dict[str, object]]:
    from .validation import iter_feature_provenance

    observations_by_id: dict[str, SourceObservation] = {}
    for observation in game.observations:
        if observation.observation_id in observations_by_id:
            raise ValueError(
                f"duplicate observation_id: {observation.observation_id}"
            )
        observations_by_id[observation.observation_id] = observation

    referenced_ids = {
        observation_id
        for provenance in iter_feature_provenance(game)
        for observation_id in provenance.observation_ids
    }
    missing_ids = referenced_ids - observations_by_id.keys()
    if missing_ids:
        raise ValueError(
            "missing observation_id referenced by feature: "
            f"{min(missing_ids)}"
        )

    observations_by_payload = {}
    for observation_id in referenced_ids:
        semantic_observation = _semantic_observation(
            observations_by_id[observation_id]
        )
        observations_by_payload[
            canonical_json(semantic_observation)
        ] = semantic_observation
    return [
        observations_by_payload[payload]
        for payload in sorted(observations_by_payload)
    ]


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
                converted[field.name] = _referenced_semantic_observations(value)
                continue
            if isinstance(value, NormalizedGameInput) and field.name == "markets":
                converted[field.name] = [
                    _semantic_value(market)
                    for market in sorted(
                        field_value,
                        key=lambda market: market.market_key,
                    )
                ]
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


def _capture_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be a canonical UTC datetime")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a canonical UTC datetime"
        ) from error
    if _canonical_value(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UTC datetime")
    return parsed


def _capture_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a canonical date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a canonical date")
    return parsed


def _capture_typed_value(
    value: object,
    expected_type: object,
    field_name: str,
) -> object:
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if origin in (Union, UnionType):
        if value is None and type(None) in arguments:
            return None
        errors = []
        for argument in arguments:
            if argument is type(None):
                continue
            try:
                return _capture_typed_value(value, argument, field_name)
            except (TypeError, ValueError) as error:
                errors.append(error)
        raise ValueError(f"{field_name} has an invalid value") from (
            errors[-1] if errors else None
        )
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a sequence")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TypeError(f"unsupported capture tuple type for {field_name}")
        return tuple(
            _capture_typed_value(item, arguments[0], f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    if expected_type is datetime:
        return _capture_datetime(value, field_name)
    if expected_type is date:
        return _capture_date(value, field_name)
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a canonical enum string")
        try:
            return expected_type(value)
        except ValueError as error:
            raise ValueError(
                f"{field_name} contains an unsupported enum value"
            ) from error
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        expected_fields = {field.name for field in fields(expected_type)}
        if set(value) != expected_fields:
            missing = ", ".join(sorted(expected_fields - set(value))) or "none"
            additional = ", ".join(sorted(set(value) - expected_fields)) or "none"
            raise ValueError(
                f"{field_name} schema mismatch; missing={missing}; "
                f"additional={additional}"
            )
        converted = {
            field.name: _capture_typed_value(
                value[field.name],
                field.type,
                f"{field_name}.{field.name}",
            )
            for field in fields(expected_type)
        }
        try:
            return expected_type(**converted)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} is invalid: {error}") from error
    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        return value
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} must be a boolean")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        return value
    if expected_type is float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{field_name} must be a finite number")
        return float(value)
    raise TypeError(f"unsupported capture field type for {field_name}")


def _resolved_capture_game(game: NormalizedGameInput) -> NormalizedGameInput:
    if game.identity.away_team == game.identity.home_team:
        raise ValueError("capture identity away_team and home_team must differ")
    side_by_team = {
        game.identity.away_team: "away",
        game.identity.home_team: "home",
    }
    side_market_types = frozenset(
        (MarketType.F5_SIDE, MarketType.ML, MarketType.PLUS_1_5)
    )
    resolved_markets = []
    market_keys = set()
    for market in game.markets:
        if market.market_key in market_keys:
            raise ValueError("capture markets must have unique market_key values")
        market_keys.add(market.market_key)
        if market.market_type in side_market_types:
            selected_side = market.selected_side
            if selected_side not in {"away", "home"}:
                selected_side = side_by_team.get(selected_side)
            if selected_side is None:
                raise ValueError(
                    "capture side market must use away/home or a captured team alias"
                )
            market = replace(market, selected_side=selected_side)
        resolved_markets.append(market)
    return replace(
        game,
        markets=tuple(
            sorted(resolved_markets, key=lambda market: market.market_key)
        ),
    )


def normalized_game_input_from_capture_payload(
    payload: Mapping[str, object],
) -> NormalizedGameInput:
    """Reconstruct one complete normalized game from its canonical capture."""

    if not isinstance(payload, Mapping):
        raise ValueError("capture payload must be a mapping")
    game = _capture_typed_value(
        payload,
        NormalizedGameInput,
        "capture payload",
    )
    if not isinstance(game, NormalizedGameInput):
        raise TypeError("capture payload must reconstruct NormalizedGameInput")
    return game


def semantic_forecast_payload_from_capture_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a complete capture payload and recover forecast semantics."""

    game = normalized_game_input_from_capture_payload(payload)
    return semantic_forecast_payload(_resolved_capture_game(game))


def normalized_input_hash_from_capture_payload(
    payload: Mapping[str, object],
) -> str:
    """Hash canonical forecast semantics recovered from a capture payload."""

    encoded = canonical_json(
        semantic_forecast_payload_from_capture_payload(payload)
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_input_hash(game: NormalizedGameInput) -> str:
    """Hash the canonical semantic normalized input."""

    encoded = canonical_json(semantic_forecast_payload(game)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forecast_run_id_from_semantic_payload(
    semantic_payload: Mapping[str, object],
    model_version: str,
    simulation_config: Mapping[str, object],
) -> str:
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    if not isinstance(simulation_config, Mapping):
        raise TypeError("simulation_config must be a mapping")
    payload = {
        "input": semantic_payload,
        "model_version": model_version,
        "simulation_config": simulation_config,
    }
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_forecast_run_id(
    game: NormalizedGameInput,
    model_version: str,
    simulation_config: Mapping[str, object],
) -> str:
    """Build a deterministic identity for one semantic forecast run."""

    return _forecast_run_id_from_semantic_payload(
        semantic_forecast_payload(game),
        model_version,
        simulation_config,
    )


def build_forecast_run_id_for_input(
    game_input: NormalizedGameInput,
    config: SimulationConfig,
) -> str:
    """Build the run ID through the same validated input path as forecasting."""

    from .engine import resolve_forecast_input
    from .validation import validate_pregame_input

    validate_pregame_input(game_input)
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")
    resolved_game = resolve_forecast_input(game_input)
    return build_forecast_run_id(
        resolved_game,
        config.version,
        config.to_mapping(),
    )


def build_forecast_run_id_from_capture_payload(
    payload: Mapping[str, object],
    model_version: str,
    simulation_config: Mapping[str, object],
) -> str:
    """Build a deterministic run ID from one frozen capture payload."""

    return _forecast_run_id_from_semantic_payload(
        semantic_forecast_payload_from_capture_payload(payload),
        model_version,
        simulation_config,
    )
