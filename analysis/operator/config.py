"""Immutable versioned forecast configuration registry."""

from dataclasses import fields
import json
from pathlib import Path

from analysis.forecasting.config import SimulationConfig

from .errors import BundleValidationError


DEFAULT_REGISTRY_ROOT = (
    Path(__file__).resolve().parents[2] / "config" / "forecasting"
)


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


def load_forecast_config(
    version: str,
    *,
    registry_root: str | Path | None = None,
) -> SimulationConfig:
    """Load one complete immutable config by its exact semantic version."""

    if (
        not isinstance(version, str)
        or not version.strip()
        or Path(version).name != version
    ):
        raise BundleValidationError(
            "forecast config version must be a nonempty filename-safe string"
        )

    root = DEFAULT_REGISTRY_ROOT if registry_root is None else Path(registry_root)
    path = root / f"{version}.json"
    if not path.is_file():
        raise BundleValidationError("unknown forecast config version")

    try:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        mapping = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise BundleValidationError(
            "invalid forecast config: unreadable or malformed JSON"
        ) from error

    if not isinstance(mapping, dict):
        raise BundleValidationError(
            "invalid forecast config: root must be a JSON object"
        )

    expected_fields = {field.name for field in fields(SimulationConfig)}
    actual_fields = set(mapping)
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        raise BundleValidationError("missing config fields")
    unknown_fields = actual_fields - expected_fields
    if unknown_fields:
        raise BundleValidationError("unknown config fields")
    if mapping["version"] != version:
        raise BundleValidationError(
            "forecast config filename/content version mismatch"
        )

    try:
        return SimulationConfig.from_mapping(mapping)
    except (TypeError, ValueError) as error:
        raise BundleValidationError(
            "invalid forecast config: simulation config validation"
        ) from error
