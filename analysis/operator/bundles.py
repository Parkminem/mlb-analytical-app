"""Canonical loading and immutable persistence for operator bundles."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

from pydantic import TypeAdapter

from analysis.forecasting.identity import canonical_json
from analysis.ledger import CaptureCommand

from .errors import BundleValidationError, OperatorIOError
from .schemas import (
    ConfirmedBundle,
    ProjectedBundle,
    ReviewBundle,
    parse_confirmed_bundle_json,
    parse_projected_bundle_json,
    parse_review_bundle_json,
)


_Bundle = ProjectedBundle | ConfirmedBundle | ReviewBundle
_BUNDLE_ADAPTERS = {
    ProjectedBundle: TypeAdapter(ProjectedBundle),
    ConfirmedBundle: TypeAdapter(ConfirmedBundle),
    ReviewBundle: TypeAdapter(ReviewBundle),
}
_BUNDLE_PARSERS = (
    parse_projected_bundle_json,
    parse_confirmed_bundle_json,
    parse_review_bundle_json,
)
DEFAULT_BUNDLE_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "shadow" / "bundles"
)


@dataclass(frozen=True)
class ManagedBundleCopy:
    path: Path
    bundle_hash: str
    replayed: bool

    @property
    def hash(self) -> str:
        return self.bundle_hash


def load_bundle(path: str | os.PathLike[str]) -> _Bundle:
    """Read raw bundle bytes and delegate strict domain parsing."""

    bundle_path = Path(path)
    try:
        raw = bundle_path.read_bytes()
    except OSError as error:
        raise OperatorIOError(
            f"unable to read bundle {bundle_path}: {error}"
        ) from error
    parse_errors: list[BundleValidationError] = []
    for candidate in _BUNDLE_PARSERS:
        try:
            return candidate(raw)
        except BundleValidationError as error:
            parse_errors.append(error)
    raise parse_errors[0]


def _serialized_bundle(bundle: _Bundle) -> dict[str, object]:
    adapter = _BUNDLE_ADAPTERS.get(type(bundle))
    if adapter is None:
        raise TypeError("bundle must be a validated operator bundle")
    payload = adapter.dump_python(
        bundle,
        mode="python",
        round_trip=True,
        warnings=False,
    )
    if not isinstance(payload, dict):
        raise TypeError("validated bundle must serialize to a mapping")
    normalized = _normalize_bundle_collections(payload)
    if not isinstance(normalized, dict):
        raise TypeError("normalized bundle must serialize to a mapping")
    return normalized


def _normalize_bundle_collections(value: object) -> object:
    """Convert unordered domain collections into deterministic JSON arrays."""

    if isinstance(value, (set, frozenset)):
        normalized = tuple(
            _normalize_bundle_collections(item) for item in value
        )
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Mapping):
        return {
            key: _normalize_bundle_collections(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_bundle_collections(item) for item in value]
    return value


def bundle_canonical_bytes(bundle: _Bundle) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a validated bundle."""

    return canonical_json(_serialized_bundle(bundle)).encode("utf-8")


def bundle_sha256(bundle: _Bundle) -> str:
    """Return the SHA-256 identity of canonical bundle bytes."""

    return hashlib.sha256(bundle_canonical_bytes(bundle)).hexdigest()


def intrinsic_capture_key(
    schema_version: int,
    stage: str,
    command: CaptureCommand,
) -> str:
    """Hash intrinsic capture content, excluding only its lineage target."""

    if type(schema_version) is not int or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a nonempty string")
    if not isinstance(command, CaptureCommand):
        raise TypeError("command must be CaptureCommand")
    command_payload = {
        field.name: getattr(command, field.name)
        for field in fields(CaptureCommand)
        if field.name != "supersedes_capture_id"
    }
    encoded = canonical_json(
        {
            "schema_version": schema_version,
            "stage": stage,
            "capture_command": command_payload,
        }
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_bundle_replays(path: Path, expected: bytes) -> bool:
    if path.is_symlink():
        raise OperatorIOError(
            f"managed bundle path cannot be a symbolic link: {path}"
        )
    if not path.exists():
        return False
    if not path.is_file():
        raise OperatorIOError(
            f"managed bundle path must be a regular file: {path}"
        )
    existing = path.read_bytes()
    if existing != expected:
        raise OperatorIOError(
            f"conflicting managed bundle bytes at existing path: {path}"
        )
    return True


def _open_directory_without_following_links(path: Path) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def persist_managed_bundle(
    bundle: _Bundle,
    *,
    root: str | os.PathLike[str] = DEFAULT_BUNDLE_ROOT,
) -> ManagedBundleCopy:
    """Create or replay one immutable canonical managed bundle copy."""

    canonical_bytes = bundle_canonical_bytes(bundle)
    bundle_hash = hashlib.sha256(canonical_bytes).hexdigest()
    temp_path: Path | None = None
    directory_descriptor = -1
    try:
        supplied_root = Path(root).expanduser()
        if supplied_root.is_symlink():
            raise OperatorIOError(
                f"managed bundle root cannot be a symbolic link: {supplied_root}"
            )
        resolved_root = supplied_root.resolve(strict=False)
        resolved_root.mkdir(parents=True, exist_ok=True)
        game_dir = resolved_root / str(bundle.game_pk)
        if game_dir.is_symlink():
            raise OperatorIOError(
                "managed bundle game directory cannot be a symbolic link: "
                f"{game_dir}"
            )
        game_dir.mkdir(exist_ok=True)
        resolved_game_dir = game_dir.resolve(strict=True)
        if (
            resolved_game_dir.parent != resolved_root
            or resolved_game_dir.name != str(bundle.game_pk)
        ):
            raise OperatorIOError(
                f"managed bundle game directory escaped root: {game_dir}"
            )

        filename = (
            f"{bundle.stage}-v{bundle.schema_version}-{bundle_hash[:16]}.json"
        )
        destination = resolved_game_dir / filename
        directory_descriptor = _open_directory_without_following_links(
            resolved_game_dir
        )
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        if _existing_bundle_replays(destination, canonical_bytes):
            return ManagedBundleCopy(
                path=destination,
                bundle_hash=bundle_hash,
                replayed=True,
            )

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=resolved_game_dir,
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(canonical_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())

        if _existing_bundle_replays(destination, canonical_bytes):
            temp_path.unlink()
            temp_path = None
            return ManagedBundleCopy(
                path=destination,
                bundle_hash=bundle_hash,
                replayed=True,
            )
        os.replace(temp_path, destination)
        temp_path = None
        os.fsync(directory_descriptor)
        return ManagedBundleCopy(
            path=destination,
            bundle_hash=bundle_hash,
            replayed=False,
        )
    except OperatorIOError:
        raise
    except OSError as error:
        raise OperatorIOError(
            f"unable to persist managed bundle: {error}"
        ) from error
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if directory_descriptor >= 0:
            try:
                fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(directory_descriptor)


__all__ = (
    "DEFAULT_BUNDLE_ROOT",
    "ManagedBundleCopy",
    "bundle_canonical_bytes",
    "bundle_sha256",
    "intrinsic_capture_key",
    "load_bundle",
    "persist_managed_bundle",
)
