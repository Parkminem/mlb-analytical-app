"""Compact, lossless encoding for ledger-stored forecast distributions."""

from collections.abc import Mapping
import base64
import binascii
import json
import zlib

from analysis.forecasting.identity import canonical_json


_ENCODING = "zlib-base64-canonical-json-v1"
_HISTORICAL_DISTRIBUTION_FIELDS = frozenset(
    ("component_id", "values", "summary")
)
_COMPACT_DISTRIBUTION_FIELDS = frozenset(
    (
        "component_id",
        "summary",
        "values_encoding",
        "values_count",
        "values_data",
    )
)


def _require_exact_fields(
    value: object,
    expected_fields: frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    actual_fields = frozenset(value)
    unknown_fields = actual_fields - expected_fields
    if unknown_fields:
        raise ValueError(
            f"{field_name} contains unknown fields: "
            + ", ".join(sorted(unknown_fields))
        )
    missing_fields = expected_fields - actual_fields
    if missing_fields:
        raise ValueError(
            f"{field_name} is missing fields: "
            + ", ".join(sorted(missing_fields))
        )
    return value


def compact_forecast_outputs(outputs: Mapping[str, object]) -> dict[str, object]:
    """Replace ledger distribution values with a deterministic compact form."""

    if not isinstance(outputs, Mapping):
        raise TypeError("forecast outputs must be a mapping")
    distributions = outputs.get("component_distributions")
    if not isinstance(distributions, Mapping):
        raise ValueError("forecast component_distributions must be a mapping")

    compact_distributions: dict[str, object] = {}
    for component_id, value in distributions.items():
        payload = _require_exact_fields(
            value,
            _HISTORICAL_DISTRIBUTION_FIELDS,
            f"component distribution {component_id}",
        )
        values = payload["values"]
        if not isinstance(values, list):
            raise ValueError(
                f"component distribution {component_id} values must be a list"
            )
        encoded = base64.b64encode(
            zlib.compress(canonical_json(values).encode("utf-8"), level=6)
        ).decode("ascii")
        compact_distributions[component_id] = {
            "component_id": payload["component_id"],
            "summary": payload["summary"],
            "values_encoding": _ENCODING,
            "values_count": len(values),
            "values_data": encoded,
        }

    compact_outputs = dict(outputs)
    compact_outputs["component_distributions"] = compact_distributions
    return compact_outputs


def _expand_values(payload: Mapping[str, object], field_name: str) -> list[object]:
    if payload["values_encoding"] != _ENCODING:
        raise ValueError(f"{field_name} has unsupported values_encoding")
    values_data = payload["values_data"]
    if not isinstance(values_data, str):
        raise ValueError(f"{field_name} values_data must be a string")
    try:
        compressed = base64.b64decode(values_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{field_name} has invalid base64 values_data") from error
    try:
        encoded_values = zlib.decompress(compressed)
    except zlib.error as error:
        raise ValueError(f"{field_name} has invalid zlib values_data") from error
    try:
        values = json.loads(encoded_values.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} has invalid JSON values_data") from error
    if not isinstance(values, list):
        raise ValueError(f"{field_name} decoded values must be a list")
    values_count = payload["values_count"]
    if (
        isinstance(values_count, bool)
        or not isinstance(values_count, int)
        or values_count < 0
    ):
        raise ValueError(f"{field_name} values_count must be a nonnegative integer")
    if len(values) != values_count:
        raise ValueError(f"{field_name} values_count does not match decoded values")
    return values


def expand_component_distributions(value: object) -> object:
    """Restore compact distributions while accepting historical values payloads."""

    if not isinstance(value, Mapping):
        raise ValueError("forecast component_distributions must be a mapping")
    if all(
        isinstance(payload, Mapping) and "values" in payload
        for payload in value.values()
    ):
        for component_id, payload in value.items():
            _require_exact_fields(
                payload,
                _HISTORICAL_DISTRIBUTION_FIELDS,
                f"component distribution {component_id}",
            )
        return value

    expanded: dict[str, object] = {}
    for component_id, value_payload in value.items():
        field_name = f"component distribution {component_id}"
        payload = _require_exact_fields(
            value_payload,
            _COMPACT_DISTRIBUTION_FIELDS,
            field_name,
        )
        expanded[component_id] = {
            "component_id": payload["component_id"],
            "values": _expand_values(payload, field_name),
            "summary": payload["summary"],
        }
    return expanded


__all__ = (
    "compact_forecast_outputs",
    "expand_component_distributions",
)
