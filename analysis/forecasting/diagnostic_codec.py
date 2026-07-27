"""Strict conversion helpers for persisted forecast diagnostics."""

from collections.abc import Mapping, Sequence


def exact_mapping(
    value: object,
    expected_fields: frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    actual_fields = frozenset(value)
    if actual_fields != expected_fields:
        missing = ", ".join(sorted(expected_fields - actual_fields)) or "none"
        unknown = ", ".join(sorted(actual_fields - expected_fields)) or "none"
        raise ValueError(
            f"{field_name} schema mismatch; missing={missing}; unknown={unknown}"
        )
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def sequence(value: object, field_name: str) -> Sequence[object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ValueError(f"{field_name} must be a sequence")
    return value


def string_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def enum_value(enum_type: type, value: object, field_name: str):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical enum string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field_name} has an unsupported enum value") from error
