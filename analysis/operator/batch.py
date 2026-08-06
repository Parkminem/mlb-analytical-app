"""Bounded pregame batch contracts and execution support."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
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
from analysis.forecasting.identity import canonical_json
from analysis.ledger import LedgerStore

from .config import DEFAULT_REGISTRY_ROOT
from .errors import BundleValidationError, OperatorError, OperatorIOError
from .fdio import (
    create_exclusive_file,
    open_directory,
    stat_at,
    verify_public_directory,
)
from .schemas import (
    ConfirmedBundle,
    ProjectedBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
)
from .service import run_confirmed, run_projected


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


class PregameBatchReceiptError(OperatorIOError):
    """Receipt publication failed after audited ledger work completed."""

    def __init__(self, message: str, *, result: PregameBatchResult) -> None:
        if not isinstance(result, PregameBatchResult):
            raise TypeError("result must be a PregameBatchResult")
        super().__init__(message)
        self.result = result


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


def _worker_failure(
    entry: PreparedBatchEntry,
    error: OperatorError,
) -> PregameBatchFailure:
    progress = error.partial_progress
    return PregameBatchFailure(
        game_pk=entry.game_pk,
        bundle_path=entry.bundle_path,
        error_type=error.__class__.__name__,
        message=" ".join(str(error).split()) or error.__class__.__name__,
        exit_code=error.exit_code,
        capture_id=None if progress is None else progress.capture_id,
        forecast_run_id=(
            None if progress is None else progress.forecast_run_id
        ),
        capture_forecast_link_record_id=(
            None
            if progress is None
            else progress.capture_forecast_link_record_id
        ),
        rerun_safe=False if progress is None else progress.rerun_safe,
    )


def _execute_batch_entry(
    stage: str,
    entry: PreparedBatchEntry,
    ledger_path: str,
    bundle_root: str,
    registry_root: str | None,
) -> BatchResultEntry:
    """Run one immutable current bundle inside a process worker."""

    try:
        if stage == "projected":
            bundle = parse_projected_bundle_json(entry.canonical_bundle)
            result = run_projected(
                bundle,
                ledger_path=ledger_path,
                bundle_root=bundle_root,
                registry_root=registry_root,
            )
        elif stage == "confirmed":
            bundle = parse_confirmed_bundle_json(entry.canonical_bundle)
            result = run_confirmed(
                bundle,
                ledger_path=ledger_path,
                bundle_root=bundle_root,
                registry_root=registry_root,
            )
        else:
            raise ValueError("batch worker stage must be projected or confirmed")
        return PregameBatchSuccess(
            game_pk=result.game_pk,
            bundle_path=entry.bundle_path,
            capture_id=result.capture.capture_id,
            forecast_run_id=result.forecast.forecast_run_id,
            capture_forecast_link_record_id=result.link.envelope.record_id,
            managed_bundle_path=result.managed_bundle.path,
            managed_bundle_hash=result.managed_bundle.bundle_hash,
            managed_bundle_replayed=result.managed_bundle.replayed,
            capture_replayed=result.capture_replayed,
            forecast_replayed=result.forecast_replayed,
            link_replayed=result.link_replayed,
        )
    except OperatorError as error:
        return _worker_failure(entry, error)
    except Exception as error:
        return PregameBatchFailure(
            game_pk=entry.game_pk,
            bundle_path=entry.bundle_path,
            error_type="internal_worker_error",
            message=f"{error.__class__.__name__}: {' '.join(str(error).split())}",
            exit_code=1,
        )


def _crashed_worker_failure(
    entry: PreparedBatchEntry,
    error: BaseException,
) -> PregameBatchFailure:
    return PregameBatchFailure(
        game_pk=entry.game_pk,
        bundle_path=entry.bundle_path,
        error_type="worker_process_error",
        message=f"{error.__class__.__name__}: {' '.join(str(error).split())}",
        exit_code=1,
    )


def _execute_prepared_batch(
    stage: str,
    entries: tuple[PreparedBatchEntry, ...],
    *,
    ledger_path: Path,
    bundle_root: Path,
    registry_root: Path | None,
    max_workers: int,
) -> tuple[BatchResultEntry, ...]:
    runtime = (
        str(ledger_path),
        str(bundle_root),
        None if registry_root is None else str(registry_root),
    )
    if max_workers == 1:
        return tuple(
            _execute_batch_entry(stage, entry, *runtime) for entry in entries
        )

    by_game: dict[int, BatchResultEntry] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_batch_entry,
                stage,
                entry,
                *runtime,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                by_game[entry.game_pk] = future.result()
            except BaseException as error:
                by_game[entry.game_pk] = _crashed_worker_failure(entry, error)
    return tuple(by_game[entry.game_pk] for entry in entries)


def _coverage_failure(
    success: PregameBatchSuccess,
    message: str,
) -> PregameBatchFailure:
    return PregameBatchFailure(
        game_pk=success.game_pk,
        bundle_path=success.bundle_path,
        error_type="coverage_audit",
        message=message,
        exit_code=5,
        capture_id=success.capture_id,
        forecast_run_id=success.forecast_run_id,
        capture_forecast_link_record_id=(
            success.capture_forecast_link_record_id
        ),
        rerun_safe=True,
    )


def audit_pregame_batch(
    result: PregameBatchResult,
    *,
    ledger_path: str | Path,
) -> PregameBatchResult:
    """Verify every reported success triplet in one fresh ledger checkpoint."""

    if not isinstance(result, PregameBatchResult):
        raise TypeError("result must be a PregameBatchResult")
    if not result.successes:
        return result
    try:
        checkpoint = LedgerStore(Path(ledger_path)).pregame_checkpoint()
    except Exception as error:
        message = (
            "ledger checkpoint unavailable during coverage audit: "
            f"{error.__class__.__name__}"
        )
        return PregameBatchResult(
            stage=result.stage,
            max_workers=result.max_workers,
            target_game_pks=result.target_game_pks,
            entries=tuple(
                _coverage_failure(entry, message)
                if isinstance(entry, PregameBatchSuccess)
                else entry
                for entry in result.entries
            ),
        )

    audited: list[BatchResultEntry] = []
    for entry in result.entries:
        if isinstance(entry, PregameBatchFailure):
            audited.append(entry)
            continue
        capture = checkpoint.record("capture", entry.capture_id)
        forecast = checkpoint.record("forecast", entry.forecast_run_id)
        link = checkpoint.record(
            "capture_forecast_link",
            entry.capture_forecast_link_record_id,
        )
        mismatch: str | None = None
        if capture is None:
            mismatch = "reported capture is missing from the ledger"
        elif capture.game_pk != entry.game_pk:
            mismatch = "reported capture belongs to a different game"
        elif forecast is None:
            mismatch = "reported forecast is missing from the ledger"
        elif link is None:
            mismatch = "reported capture-forecast link is missing from the ledger"
        elif link.capture_id != entry.capture_id:
            mismatch = "reported link references a different capture"
        elif link.forecast_run_id != entry.forecast_run_id:
            mismatch = "reported link references a different forecast"
        audited.append(
            entry if mismatch is None else _coverage_failure(entry, mismatch)
        )
    return PregameBatchResult(
        stage=result.stage,
        max_workers=result.max_workers,
        target_game_pks=result.target_game_pks,
        entries=tuple(audited),
    )


def _publish_batch_receipt(
    path: Path,
    result: PregameBatchResult,
    *,
    directory_descriptor: int | None = None,
) -> None:
    content = (canonical_json(result.to_mapping()) + "\n").encode("utf-8")
    resolved = path.expanduser().resolve(strict=False)
    owned_directory_descriptor = -1
    temporary_name: str | None = None
    file_descriptor = -1
    try:
        if directory_descriptor is None:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            owned_directory_descriptor = open_directory(resolved.parent)
            directory_descriptor = owned_directory_descriptor
        verify_public_directory(
            resolved.parent,
            directory_descriptor,
            label="batch receipt directory",
        )
        existing = stat_at(directory_descriptor, resolved.name)
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise OperatorIOError("batch receipt path cannot be symbolic")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise OperatorIOError("batch receipt path must be a regular file")
        file_descriptor, temporary_name = create_exclusive_file(
            directory_descriptor,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
        )
        with os.fdopen(file_descriptor, "wb") as output:
            file_descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        verify_public_directory(
            resolved.parent,
            directory_descriptor,
            label="batch receipt directory",
        )
        os.replace(
            temporary_name,
            resolved.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = None
        os.fsync(directory_descriptor)
        verify_public_directory(
            resolved.parent,
            directory_descriptor,
            label="batch receipt directory",
        )
    except OperatorIOError:
        raise
    except OSError as error:
        raise OperatorIOError(
            f"unable to publish pregame batch receipt {path}: {error}"
        ) from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_name is not None and directory_descriptor is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass
        if owned_directory_descriptor >= 0:
            os.close(owned_directory_descriptor)


def _receipt_protected_paths(
    *,
    manifest_path: str | Path,
    entries: tuple[PreparedBatchEntry, ...],
    stage: str,
    ledger_path: Path,
    bundle_root: Path,
    registry_root: Path,
) -> set[Path]:
    protected = {
        Path(manifest_path).expanduser().resolve(strict=False),
        ledger_path,
        Path(f"{ledger_path}.checkpoint-v1.json"),
        Path(f"{ledger_path}.lock"),
        *(entry.bundle_path for entry in entries),
    }
    for entry in entries:
        digest = hashlib.sha256(entry.canonical_bundle).hexdigest()
        mapping = json.loads(entry.canonical_bundle)
        schema_version = mapping["schema_version"]
        config_version = mapping["payload"]["forecast_config_version"]
        protected.add(
            bundle_root
            / str(entry.game_pk)
            / f"{stage}-v{schema_version}-{digest[:16]}.json"
        )
        protected.add(registry_root / f"{config_version}.json")
    return protected


def _resolve_safe_receipt_path(
    receipt_path: str | Path,
    *,
    manifest_path: str | Path,
    entries: tuple[PreparedBatchEntry, ...],
    stage: str,
    ledger_path: Path,
    bundle_root: Path,
    registry_root: Path,
) -> Path:
    supplied = Path(receipt_path).expanduser()
    if supplied.is_symlink():
        raise OperatorIOError("batch receipt path cannot be symbolic")
    resolved = supplied.resolve(strict=False)
    protected = _receipt_protected_paths(
        manifest_path=manifest_path,
        entries=entries,
        stage=stage,
        ledger_path=ledger_path,
        bundle_root=bundle_root,
        registry_root=registry_root,
    )
    immutable_roots = (bundle_root, registry_root)
    inside_immutable_store = any(
        resolved == root or root in resolved.parents
        for root in immutable_roots
    )
    if resolved in protected or inside_immutable_store:
        raise OperatorIOError(
            "batch receipt path collides with an authoritative or input file"
        )
    if resolved.exists() and not resolved.is_file():
        raise OperatorIOError("batch receipt path must be a regular file")
    return resolved


def _open_receipt_directory(path: Path) -> int:
    descriptor = -1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = open_directory(path.parent)
        verify_public_directory(
            path.parent,
            descriptor,
            label="batch receipt directory",
        )
        existing = stat_at(descriptor, path.name)
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise OperatorIOError("batch receipt path cannot be symbolic")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise OperatorIOError("batch receipt path must be a regular file")
        return descriptor
    except OperatorIOError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise OperatorIOError(
            f"unable to authenticate batch receipt directory: {error}"
        ) from error


def run_pregame_batch(
    manifest_path: str | Path,
    *,
    expected_stage: str,
    ledger_path: str | Path,
    bundle_root: str | Path,
    receipt_path: str | Path,
    max_workers: int = 3,
    registry_root: str | Path | None = None,
) -> PregameBatchResult:
    """Execute, audit, and publish one projected or confirmed target batch."""

    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 4
    ):
        raise ValueError("max_workers must be an integer from 1 through 4")
    manifest = load_pregame_batch_manifest(
        manifest_path,
        expected_stage=expected_stage,
    )
    prepared = prepare_pregame_batch(
        manifest,
        manifest_path=manifest_path,
    )
    resolved_ledger = Path(ledger_path).expanduser().resolve(strict=False)
    resolved_bundle_root = Path(bundle_root).expanduser().resolve(strict=False)
    resolved_registry = (
        DEFAULT_REGISTRY_ROOT
        if registry_root is None
        else Path(registry_root).expanduser()
    )
    resolved_registry = resolved_registry.resolve(strict=False)
    resolved_receipt = _resolve_safe_receipt_path(
        receipt_path,
        manifest_path=manifest_path,
        entries=prepared,
        stage=manifest.stage,
        ledger_path=resolved_ledger,
        bundle_root=resolved_bundle_root,
        registry_root=resolved_registry,
    )
    receipt_directory_descriptor = _open_receipt_directory(resolved_receipt)
    try:
        # Initialize the shared lock before workers race to open the ledger.
        LedgerStore(resolved_ledger).pregame_checkpoint()
        entries = _execute_prepared_batch(
            manifest.stage,
            prepared,
            ledger_path=resolved_ledger,
            bundle_root=resolved_bundle_root,
            registry_root=resolved_registry,
            max_workers=max_workers,
        )
        result = PregameBatchResult(
            stage=manifest.stage,
            max_workers=max_workers,
            target_game_pks=tuple(entry.game_pk for entry in prepared),
            entries=entries,
        )
        audited = audit_pregame_batch(result, ledger_path=resolved_ledger)
        try:
            _publish_batch_receipt(
                resolved_receipt,
                audited,
                directory_descriptor=receipt_directory_descriptor,
            )
        except OperatorIOError as error:
            raise PregameBatchReceiptError(
                "batch ledger records are durable but receipt publication "
                "failed; rerun the exact manifest to rebuild the receipt "
                f"safely: {error}",
                result=audited,
            ) from error
        return audited
    finally:
        os.close(receipt_directory_descriptor)


__all__ = (
    "BatchResultEntry",
    "PregameBatchEntry",
    "PregameBatchFailure",
    "PregameBatchManifest",
    "PregameBatchResult",
    "PregameBatchReceiptError",
    "PregameBatchSuccess",
    "PreparedBatchEntry",
    "audit_pregame_batch",
    "load_pregame_batch_manifest",
    "prepare_pregame_batch",
    "run_pregame_batch",
)
