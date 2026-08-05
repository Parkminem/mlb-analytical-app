"""Descriptor-relative filesystem primitives for operator artifacts."""

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import secrets
import stat

from .errors import OperatorIOError


@dataclass(frozen=True)
class DirectoryIdentity:
    device: int
    inode: int


def directory_identity(descriptor: int) -> DirectoryIdentity:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OperatorIOError("opened artifact descriptor is not a directory")
    return DirectoryIdentity(metadata.st_dev, metadata.st_ino)


def open_directory(
    path: str | os.PathLike[str],
    *,
    dir_fd: int | None = None,
) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, dir_fd=dir_fd)


def open_regular_file(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
) -> int:
    metadata = stat_at(directory_descriptor, name)
    if metadata is None:
        raise OperatorIOError(f"{label} does not exist")
    if stat.S_ISLNK(metadata.st_mode):
        raise OperatorIOError(f"{label} cannot be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise OperatorIOError(f"{label} must be a regular file")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        linked = stat_at(directory_descriptor, name)
        expected_identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or linked is None
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
            or (linked.st_dev, linked.st_ino) != expected_identity
        ):
            raise OperatorIOError(f"{label} changed during authentication")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def stat_at(directory_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None


def verify_public_directory(
    path: Path,
    descriptor: int,
    *,
    label: str,
) -> None:
    expected = directory_identity(descriptor)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise OperatorIOError(f"{label} changed during publication") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != expected.device
        or metadata.st_ino != expected.inode
    ):
        raise OperatorIOError(f"{label} changed during publication")


def verify_child_directory(
    parent_descriptor: int,
    name: str,
    child_descriptor: int,
    *,
    label: str,
) -> None:
    expected = directory_identity(child_descriptor)
    metadata = stat_at(parent_descriptor, name)
    if (
        metadata is None
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != expected.device
        or metadata.st_ino != expected.inode
    ):
        raise OperatorIOError(f"{label} changed during publication")


def verify_child_regular_file(
    parent_descriptor: int,
    name: str,
    file_descriptor: int,
    *,
    label: str,
) -> None:
    expected = os.fstat(file_descriptor)
    metadata = stat_at(parent_descriptor, name)
    if (
        not stat.S_ISREG(expected.st_mode)
        or metadata is None
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_dev != expected.st_dev
        or metadata.st_ino != expected.st_ino
    ):
        raise OperatorIOError(f"{label} changed during authentication")


def read_open_regular_file(descriptor: int, *, label: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise OperatorIOError(f"{label} must be a regular file")
    with os.fdopen(os.dup(descriptor), "rb") as artifact_file:
        raw = artifact_file.read()
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise OperatorIOError(f"{label} changed while being read")
    return raw


def read_regular_file(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OperatorIOError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb") as artifact_file:
            descriptor = -1
            return artifact_file.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _temporary_name(prefix: str, suffix: str) -> str:
    return f"{prefix}{secrets.token_hex(12)}{suffix}"


def create_exclusive_file(
    directory_descriptor: int,
    *,
    prefix: str,
    suffix: str,
    mode: int = 0o600,
) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(128):
        name = _temporary_name(prefix, suffix)
        try:
            descriptor = os.open(
                name,
                flags,
                mode,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        return descriptor, name
    raise OSError(errno.EEXIST, "unable to allocate exclusive temp file")


def create_temporary_directory(
    directory_descriptor: int,
    *,
    prefix: str,
    suffix: str,
    mode: int = 0o700,
) -> tuple[int, str]:
    for _ in range(128):
        name = _temporary_name(prefix, suffix)
        try:
            os.mkdir(name, mode=mode, dir_fd=directory_descriptor)
        except FileExistsError:
            continue
        try:
            return open_directory(name, dir_fd=directory_descriptor), name
        except BaseException:
            try:
                os.rmdir(name, dir_fd=directory_descriptor)
            except OSError:
                pass
            raise
    raise OSError(errno.EEXIST, "unable to allocate exclusive temp directory")


def create_temporary_symlink(
    directory_descriptor: int,
    target: str,
    *,
    prefix: str,
    suffix: str,
    target_is_directory: bool,
) -> str:
    for _ in range(128):
        name = _temporary_name(prefix, suffix)
        try:
            os.symlink(
                target,
                name,
                target_is_directory=target_is_directory,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        return name
    raise OSError(errno.EEXIST, "unable to allocate exclusive temp symlink")


__all__ = (
    "DirectoryIdentity",
    "create_exclusive_file",
    "create_temporary_directory",
    "create_temporary_symlink",
    "directory_identity",
    "open_directory",
    "open_regular_file",
    "read_open_regular_file",
    "read_regular_file",
    "stat_at",
    "verify_child_directory",
    "verify_child_regular_file",
    "verify_public_directory",
)
