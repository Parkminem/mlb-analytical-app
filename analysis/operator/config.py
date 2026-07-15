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
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number: {value}")


def _render_field_error(label: str, field_names: set[str]) -> str:
    rendered = ", ".join(sorted(field_names))
    suffix = "s" if len(field_names) != 1 else ""
    return f"{label} config field{suffix}: {rendered}"


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
        raise BundleValidationError(f"unknown forecast config: {version}")

    try:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        mapping = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BundleValidationError(
            f"invalid forecast config {version}: {error}"
        ) from error

    if not isinstance(mapping, dict):
        raise BundleValidationError(
            f"invalid forecast config {version}: root must be a JSON object"
        )

    expected_fields = {field.name for field in fields(SimulationConfig)}
    actual_fields = set(mapping)
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        raise BundleValidationError(
            _render_field_error("missing", missing_fields)
        )
    unknown_fields = actual_fields - expected_fields
    if unknown_fields:
        raise BundleValidationError(
            _render_field_error("unknown", unknown_fields)
        )
    if mapping["version"] != version:
        raise BundleValidationError(
            "forecast config filename/content version mismatch: "
            f"{version!r} != {mapping['version']!r}"
        )

    try:
        return SimulationConfig.from_mapping(mapping)
    except (TypeError, ValueError) as error:
        raise BundleValidationError(
            f"invalid forecast config {version}: {error}"
        ) from error
