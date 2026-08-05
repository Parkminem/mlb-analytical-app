"""Canonical loading and immutable persistence for operator bundles."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat

from pydantic import TypeAdapter

from analysis.forecasting.identity import canonical_json
from analysis.ledger import CaptureCommand

from .errors import BundleValidationError, OperatorIOError
from .fdio import (
    create_exclusive_file,
    open_directory,
    read_regular_file,
    stat_at,
    verify_child_directory,
    verify_public_directory,
)
from .schemas import (
    ConfirmedBundle,
    ProjectedBundle,
    ReviewBundle,
    bundle_stage_from_json,
    parse_historical_confirmed_bundle_json,
    parse_historical_projected_bundle_json,
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
_BUNDLE_PARSERS = {
    "projected": parse_projected_bundle_json,
    "confirmed": parse_confirmed_bundle_json,
    "review": parse_review_bundle_json,
}
_HISTORICAL_BUNDLE_PARSERS = {
    "projected": parse_historical_projected_bundle_json,
    "confirmed": parse_historical_confirmed_bundle_json,
    "review": parse_review_bundle_json,
}
_MANAGED_BUNDLE_FILENAME = re.compile(
    r"(?P<stage>projected|confirmed|review)-v(?P<schema_version>[0-9]+)-"
    r"(?P<hash_prefix>[0-9a-f]{16})\.json\Z"
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


@dataclass(frozen=True)
class ReplayOnlyPregameBundle:
    """A pre-change managed bundle that may only resolve an exact replay."""

    bundle: ProjectedBundle | ConfirmedBundle
    managed_bundle: ManagedBundleCopy


def load_bundle(path: str | os.PathLike[str]) -> _Bundle:
    """Read raw bundle bytes and delegate strict domain parsing."""

    bundle_path = Path(path)
    try:
        raw = bundle_path.read_bytes()
    except OSError as error:
        raise OperatorIOError(
            f"unable to read bundle {bundle_path}: {error}"
        ) from error
    stage = bundle_stage_from_json(raw)
    return _BUNDLE_PARSERS[stage](raw)


def load_historical_managed_bundle(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
) -> _Bundle | ReplayOnlyPregameBundle:
    """Read an authenticated managed artifact through historical schemas."""

    bundle_path = Path(path).expanduser()
    root_path = Path(root).expanduser()
    try:
        if bundle_path.is_symlink() or root_path.is_symlink():
            raise OperatorIOError("historical managed bundle path is symbolic")
        resolved_path = bundle_path.resolve(strict=True)
        resolved_root = root_path.resolve(strict=True)
        if resolved_path.parent.parent != resolved_root:
            raise OperatorIOError(
                "historical managed bundle must be inside its managed root"
            )
        filename_match = _MANAGED_BUNDLE_FILENAME.fullmatch(
            resolved_path.name
        )
        if filename_match is None:
            raise OperatorIOError(
                "historical managed bundle filename is invalid"
            )
        raw = resolved_path.read_bytes()
    except OperatorIOError:
        raise
    except OSError as error:
        raise OperatorIOError(
            f"unable to read historical managed bundle {bundle_path}: {error}"
        ) from error

    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash[:16] != filename_match.group("hash_prefix"):
        raise OperatorIOError("historical managed bundle hash is invalid")
    stage = bundle_stage_from_json(raw)
    replay_only = False
    try:
        bundle = _BUNDLE_PARSERS[stage](raw)
    except BundleValidationError:
        bundle = _HISTORICAL_BUNDLE_PARSERS[stage](raw)
        replay_only = stage in ("projected", "confirmed")
    if (
        stage != filename_match.group("stage")
        or bundle.schema_version
        != int(filename_match.group("schema_version"))
        or resolved_path.parent.name != str(bundle.game_pk)
    ):
        raise OperatorIOError(
            "historical managed bundle identity is inconsistent"
        )
    if replay_only:
        return ReplayOnlyPregameBundle(
            bundle=bundle,
            managed_bundle=ManagedBundleCopy(
                path=resolved_path,
                bundle_hash=actual_hash,
                replayed=True,
            ),
        )
    return bundle


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


def _existing_bundle_replays(
    directory_descriptor: int,
    filename: str,
    expected: bytes,
    display_path: Path,
) -> bool:
    metadata = stat_at(directory_descriptor, filename)
    if metadata is None:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise OperatorIOError(
            f"managed bundle path cannot be a symbolic link: {display_path}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise OperatorIOError(
            f"managed bundle path must be a regular file: {display_path}"
        )
    existing = read_regular_file(
        directory_descriptor,
        filename,
        label=f"managed bundle path {display_path}",
    )
    if existing != expected:
        raise OperatorIOError(
            "conflicting managed bundle bytes at existing path: "
            f"{display_path}"
        )
    return True


def _open_directory_without_following_links(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
) -> int:
    return open_directory(path, dir_fd=dir_fd)


def _open_game_directory(
    root_descriptor: int,
    game_name: str,
    display_path: Path,
) -> int:
    metadata = stat_at(root_descriptor, game_name)
    if metadata is None:
        os.mkdir(game_name, mode=0o700, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
    elif stat.S_ISLNK(metadata.st_mode):
        raise OperatorIOError(
            "managed bundle game directory cannot be a symbolic link: "
            f"{display_path}"
        )
    elif not stat.S_ISDIR(metadata.st_mode):
        raise OperatorIOError(
            f"managed bundle game path must be a directory: {display_path}"
        )
    return _open_directory_without_following_links(
        game_name,
        dir_fd=root_descriptor,
    )


def _verify_bundle_directories(
    root_path: Path,
    root_descriptor: int,
    game_name: str,
    game_descriptor: int,
) -> None:
    verify_public_directory(
        root_path,
        root_descriptor,
        label="managed bundle root",
    )
    verify_child_directory(
        root_descriptor,
        game_name,
        game_descriptor,
        label="managed bundle game directory",
    )


def persist_managed_bundle(
    bundle: _Bundle,
    *,
    root: str | os.PathLike[str] = DEFAULT_BUNDLE_ROOT,
) -> ManagedBundleCopy:
    """Create or replay one immutable canonical managed bundle copy."""

    canonical_bytes = bundle_canonical_bytes(bundle)
    bundle_hash = hashlib.sha256(canonical_bytes).hexdigest()
    temp_name: str | None = None
    root_descriptor = -1
    game_descriptor = -1
    locked = False
    try:
        supplied_root = Path(root).expanduser()
        if supplied_root.is_symlink():
            raise OperatorIOError(
                f"managed bundle root cannot be a symbolic link: {supplied_root}"
            )
        resolved_root = supplied_root.resolve(strict=False)
        resolved_root.mkdir(parents=True, exist_ok=True)
        root_descriptor = _open_directory_without_following_links(resolved_root)
        game_name = str(bundle.game_pk)
        game_dir = resolved_root / game_name
        game_descriptor = _open_game_directory(
            root_descriptor,
            game_name,
            game_dir,
        )

        filename = (
            f"{bundle.stage}-v{bundle.schema_version}-{bundle_hash[:16]}.json"
        )
        destination = game_dir / filename
        fcntl.flock(game_descriptor, fcntl.LOCK_EX)
        locked = True
        if _existing_bundle_replays(
            game_descriptor,
            filename,
            canonical_bytes,
            destination,
        ):
            _verify_bundle_directories(
                resolved_root,
                root_descriptor,
                game_name,
                game_descriptor,
            )
            return ManagedBundleCopy(
                path=destination,
                bundle_hash=bundle_hash,
                replayed=True,
            )

        descriptor, temp_name = create_exclusive_file(
            game_descriptor,
            prefix=f".{filename}.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(canonical_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())

        if _existing_bundle_replays(
            game_descriptor,
            filename,
            canonical_bytes,
            destination,
        ):
            os.unlink(temp_name, dir_fd=game_descriptor)
            temp_name = None
            os.fsync(game_descriptor)
            _verify_bundle_directories(
                resolved_root,
                root_descriptor,
                game_name,
                game_descriptor,
            )
            return ManagedBundleCopy(
                path=destination,
                bundle_hash=bundle_hash,
                replayed=True,
            )
        os.replace(
            temp_name,
            filename,
            src_dir_fd=game_descriptor,
            dst_dir_fd=game_descriptor,
        )
        temp_name = None
        os.fsync(game_descriptor)
        _verify_bundle_directories(
            resolved_root,
            root_descriptor,
            game_name,
            game_descriptor,
        )
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
        if temp_name is not None and game_descriptor >= 0:
            try:
                os.unlink(temp_name, dir_fd=game_descriptor)
                os.fsync(game_descriptor)
            except OSError:
                pass
        if game_descriptor >= 0:
            try:
                if locked:
                    fcntl.flock(game_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(game_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


__all__ = (
    "DEFAULT_BUNDLE_ROOT",
    "ManagedBundleCopy",
    "ReplayOnlyPregameBundle",
    "bundle_canonical_bytes",
    "bundle_sha256",
    "intrinsic_capture_key",
    "load_bundle",
    "load_historical_managed_bundle",
    "persist_managed_bundle",
)
