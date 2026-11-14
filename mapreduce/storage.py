"""Paths inside shared storage.

Users always name data with paths *relative to the storage root* (``/data`` inside the
containers), e.g. ``inputs/books``. This module turns those into real paths and makes sure
nobody can escape the root with ``..``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class StorageError(ValueError):
    """A storage path is invalid or does not exist."""


def normalize(rel: str) -> str:
    """Canonical form of a storage path: posix separators, no leading slash, no ``.``.

    A leading ``/`` is accepted and means "relative to the storage root".
    """
    parts = [p for p in PurePosixPath(rel.strip().replace("\\", "/")).parts if p not in ("/", ".")]
    if any(p == ".." for p in parts):
        raise StorageError(f"'..' is not allowed in storage paths: {rel!r}")
    return "/".join(parts)


def resolve(root: Path, rel: str) -> Path:
    """Absolute filesystem path for storage path ``rel`` under ``root``."""
    norm = normalize(rel)
    return root.joinpath(*norm.split("/")) if norm else root


def to_storage_path(root: Path, path: Path) -> str:
    """Inverse of :func:`resolve`: the storage-relative posix path of ``path``."""
    return path.relative_to(root).as_posix()


def is_hidden(name: str) -> bool:
    """Files and dirs starting with ``_`` or ``.`` are metadata, not input (e.g. ``_SUCCESS``)."""
    return name.startswith(("_", "."))


def list_input_files(root: Path, rel: str) -> list[tuple[str, int]]:
    """All input files under storage path ``rel`` as sorted ``(storage_path, size)`` pairs.

    ``rel`` may be a file or a directory (walked recursively). Hidden files/dirs are skipped.
    """
    base = resolve(root, rel)
    if base.is_file():
        return [(to_storage_path(root, base), base.stat().st_size)]
    if not base.is_dir():
        raise StorageError(f"input path does not exist: {normalize(rel) or '/'}")

    files: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not is_hidden(d))
        for name in sorted(filenames):
            if not is_hidden(name):
                full = Path(dirpath, name)
                files.append((to_storage_path(root, full), full.stat().st_size))
    files.sort()
    return files
