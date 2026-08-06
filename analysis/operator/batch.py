"""Bounded pregame batch contracts and execution support."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .bundles import bundle_canonical_bytes
from .errors import BundleValidationError
from .schemas import (
    ConfirmedBundle,
    ProjectedBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
)


_LOWERCASE_HEX = frozenset("0123456789abcdef")
_STAGES = frozenset(("projected", "confirmed"))


def _require_stage(value: object, field_name: str = "stage") -> str:
    if not isinstance(value, str) or value not in _STAGES:
        raise ValueError(f"{field_name} must be projected or confirmed")
    return value


def _require_positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_generated_id(
    value: object,
    field_name: str,
    *,
    length: int,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(
            f"{field_name} must be {length} lowercase hexadecimal characters"
        )
    return value


def _require_path(value: object, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a Path")
    return value


class PregameBatchEntry(BaseModel):
    """One exact expected game and its externally prepared bundle path."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    game_pk: int = Field(gt=0)
    bundle: str = Field(min_length=1)


class PregameBatchManifest(BaseModel):
    """Strict versioned target set for one pregame stage."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    stage: Literal["projected", "confirmed"]
    entries: tuple[PregameBatchEntry, ...] = Field(min_length=1)

    @field_validator("entries", mode="before")
    @classmethod
    def _materialize_json_entries(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("entries must be a JSON array")
        return tuple(value)

    @model_validator(mode="after")
    def _require_unambiguous_targets(self) -> "PregameBatchManifest":
        game_pks = tuple(entry.game_pk for entry in self.entries)
        if len(game_pks) != len(set(game_pks)):
            raise ValueError("entries must contain unique game_pk values")
        bundle_paths = tuple(entry.bundle for entry in self.entries)
        if len(bundle_paths) != len(set(bundle_paths)):
            raise ValueError("entries must contain unique bundle paths")
        return self


@dataclass(frozen=True)
class PreparedBatchEntry:
    """Canonical immutable bundle bytes ready for worker dispatch."""

    game_pk: int
    bundle_path: Path
    canonical_bundle: bytes

    def __post_init__(self) -> None:
        _require_positive_integer(self.game_pk, "game_pk")
        _require_path(self.bundle_path, "bundle_path")
        if not self.bundle_path.is_absolute():
            raise ValueError("bundle_path must be absolute")
        if not isinstance(self.canonical_bundle, bytes) or not self.canonical_bundle:
            raise ValueError("canonical_bundle must be nonempty bytes")


@dataclass(frozen=True)
class PregameBatchSuccess:
    """One complete capture, forecast, link, and managed-bundle receipt row."""

    game_pk: int
    bundle_path: Path
    capture_id: str
    forecast_run_id: str
    capture_forecast_link_record_id: str
    managed_bundle_path: Path
    managed_bundle_hash: str
    managed_bundle_replayed: bool
    capture_replayed: bool
    forecast_replayed: bool
    link_replayed: bool

    def __post_init__(self) -> None:
        _require_positive_integer(self.game_pk, "game_pk")
        _require_path(self.bundle_path, "bundle_path")
        _require_path(self.managed_bundle_path, "managed_bundle_path")
        _require_generated_id(self.capture_id, "capture_id", length=32)
        _require_generated_id(
            self.forecast_run_id,
            "forecast_run_id",
            length=64,
        )
        _require_generated_id(
            self.capture_forecast_link_record_id,
            "capture_forecast_link_record_id",
            length=32,
        )
        _require_generated_id(
            self.managed_bundle_hash,
            "managed_bundle_hash",
            length=64,
        )
        for field_name in (
            "managed_bundle_replayed",
            "capture_replayed",
            "forecast_replayed",
            "link_replayed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")

    def to_mapping(self) -> dict[str, object]:
        return {
            "game_pk": self.game_pk,
            "bundle_path": str(self.bundle_path),
            "status": "success",
            "capture_id": self.capture_id,
            "forecast_run_id": self.forecast_run_id,
            "capture_forecast_link_record_id": (
                self.capture_forecast_link_record_id
            ),
            "managed_bundle_path": str(self.managed_bundle_path),
            "managed_bundle_hash": self.managed_bundle_hash,
            "managed_bundle_replayed": self.managed_bundle_replayed,
            "capture_replayed": self.capture_replayed,
            "forecast_replayed": self.forecast_replayed,
            "link_replayed": self.link_replayed,
        }


@dataclass(frozen=True)
class PregameBatchFailure:
    """One stable per-game failure and optional authoritative partial progress."""

    game_pk: int
    bundle_path: Path
    error_type: str
    message: str
    exit_code: int
    capture_id: str | None = None
    forecast_run_id: str | None = None
    capture_forecast_link_record_id: str | None = None
    rerun_safe: bool = False

    def __post_init__(self) -> None:
        _require_positive_integer(self.game_pk, "game_pk")
        _require_path(self.bundle_path, "bundle_path")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("error_type must be a nonempty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a nonempty string")
        _require_positive_integer(self.exit_code, "exit_code")
        for field_name, length in (
            ("capture_id", 32),
            ("forecast_run_id", 64),
            ("capture_forecast_link_record_id", 32),
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_generated_id(value, field_name, length=length)
        if not isinstance(self.rerun_safe, bool):
            raise TypeError("rerun_safe must be a bool")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "game_pk": self.game_pk,
            "bundle_path": str(self.bundle_path),
            "status": "failure",
            "error_type": self.error_type,
            "message": self.message,
            "exit_code": self.exit_code,
            "rerun_safe": self.rerun_safe,
        }
        for field_name in (
            "capture_id",
            "forecast_run_id",
            "capture_forecast_link_record_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                mapping[field_name] = value
        return mapping


BatchResultEntry = PregameBatchSuccess | PregameBatchFailure


@dataclass(frozen=True)
class PregameBatchResult:
    """Complete ordered accounting for one requested pregame batch."""

    stage: str
    max_workers: int
    target_game_pks: tuple[int, ...]
    entries: tuple[BatchResultEntry, ...]

    def __post_init__(self) -> None:
        _require_stage(self.stage)
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= 4
        ):
            raise ValueError("max_workers must be an integer from 1 through 4")
        if not isinstance(self.target_game_pks, tuple) or not self.target_game_pks:
            raise ValueError("target_game_pks must be a nonempty tuple")
        for game_pk in self.target_game_pks:
            _require_positive_integer(game_pk, "target_game_pks")
        if len(self.target_game_pks) != len(set(self.target_game_pks)):
            raise ValueError("target_game_pks must be unique")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, (PregameBatchSuccess, PregameBatchFailure))
            for entry in self.entries
        ):
            raise TypeError("entries must be a tuple of batch result entries")
        actual = tuple(entry.game_pk for entry in self.entries)
        if set(actual) != set(self.target_game_pks) or len(actual) != len(
            self.target_game_pks
        ):
            raise ValueError("entries must provide exact target coverage")
        if actual != self.target_game_pks:
            raise ValueError("entries must preserve target order")

    @property
    def target_count(self) -> int:
        return len(self.target_game_pks)

    @property
    def successes(self) -> tuple[PregameBatchSuccess, ...]:
        return tuple(
            entry
            for entry in self.entries
            if isinstance(entry, PregameBatchSuccess)
        )

    @property
    def failures(self) -> tuple[PregameBatchFailure, ...]:
        return tuple(
            entry
            for entry in self.entries
            if isinstance(entry, PregameBatchFailure)
        )

    @property
    def success_count(self) -> int:
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def complete(self) -> bool:
        return self.success_count == self.target_count

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "complete": self.complete,
            "target_count": self.target_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "max_workers": self.max_workers,
            "entries": [entry.to_mapping() for entry in self.entries],
        }


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate manifest JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid manifest JSON number: {value}")


def load_pregame_batch_manifest(
    path: str | Path,
    *,
    expected_stage: str,
) -> PregameBatchManifest:
    """Strictly load one versioned projected or confirmed target manifest."""

    _require_stage(expected_stage, "expected_stage")
    manifest_path = Path(path).expanduser()
    try:
        raw = manifest_path.read_bytes()
        mapping = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        manifest = PregameBatchManifest.model_validate(mapping, strict=True)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        ValidationError,
    ) as error:
        raise BundleValidationError(
            f"invalid pregame batch manifest: {error}"
        ) from error
    if manifest.stage != expected_stage:
        raise BundleValidationError(
            "pregame batch manifest stage does not match the command stage"
        )
    return manifest


def prepare_pregame_batch(
    manifest: PregameBatchManifest,
    *,
    manifest_path: str | Path,
) -> tuple[PreparedBatchEntry, ...]:
    """Read and validate every current bundle before returning any work."""

    if not isinstance(manifest, PregameBatchManifest):
        raise TypeError("manifest must be a PregameBatchManifest")
    base = Path(manifest_path).expanduser().resolve(strict=False).parent
    parser = (
        parse_projected_bundle_json
        if manifest.stage == "projected"
        else parse_confirmed_bundle_json
    )
    expected_type = (
        ProjectedBundle if manifest.stage == "projected" else ConfirmedBundle
    )
    prepared: list[PreparedBatchEntry] = []
    resolved_paths: set[Path] = set()
    for entry in manifest.entries:
        supplied = Path(os.path.expanduser(entry.bundle))
        path = supplied if supplied.is_absolute() else base / supplied
        resolved = path.resolve(strict=False)
        if resolved in resolved_paths:
            raise BundleValidationError(
                "pregame batch entries must resolve to unique bundle paths"
            )
        resolved_paths.add(resolved)
        try:
            raw = resolved.read_bytes()
            bundle = parser(raw)
        except OSError as error:
            raise BundleValidationError(
                f"unable to read pregame batch bundle {resolved}: {error}"
            ) from error
        except (ValueError, ValidationError) as error:
            raise BundleValidationError(
                f"invalid pregame batch bundle {resolved}: {error}"
            ) from error
        if type(bundle) is not expected_type or bundle.stage != manifest.stage:
            raise BundleValidationError(
                f"pregame batch bundle stage mismatch: {resolved}"
            )
        if bundle.game_pk != entry.game_pk:
            raise BundleValidationError(
                f"pregame batch bundle game_pk mismatch: {resolved}"
            )
        prepared.append(
            PreparedBatchEntry(
                game_pk=entry.game_pk,
                bundle_path=resolved,
                canonical_bundle=bundle_canonical_bytes(bundle),
            )
        )
    return tuple(prepared)


__all__ = (
    "BatchResultEntry",
    "PregameBatchEntry",
    "PregameBatchFailure",
    "PregameBatchManifest",
    "PregameBatchResult",
    "PregameBatchSuccess",
    "PreparedBatchEntry",
    "load_pregame_batch_manifest",
    "prepare_pregame_batch",
)
