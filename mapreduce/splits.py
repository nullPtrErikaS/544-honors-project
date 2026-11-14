"""Input splits: cut the input into byte ranges and read whole lines from them.

The boss treats all input files as one long byte stream and cuts it into ``M`` nearly equal
ranges, so the number of map tasks is independent of the number of files. A split may span
several small files (one :class:`Segment` per file) and a large file may be spread across
several splits.

Record boundaries follow Hadoop's ``TextInputFormat`` rule: **a segment owns every line
that starts inside its byte range** ``[start, end)``. That guarantees each line is read
exactly once no matter where the cuts fall.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from mapreduce import storage


@dataclass(frozen=True)
class Segment:
    """Byte range ``[start, end)`` of one input file (``path`` is a storage path)."""

    path: str
    start: int
    end: int


def compute_splits(files: list[tuple[str, int]], num_splits: int) -> list[list[Segment]]:
    """Cut ``(path, size)`` files into ``num_splits`` splits of (almost) equal byte size.

    Split ``i`` covers global bytes ``[i*T//n, (i+1)*T//n)`` of the concatenated input,
    where ``T`` is the total size. Splits may be empty when the input is tiny.
    """
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    total = sum(size for _, size in files)
    bounds = [i * total // num_splits for i in range(num_splits + 1)]
    splits: list[list[Segment]] = [[] for _ in range(num_splits)]

    offset = 0
    for path, size in files:
        file_start, file_end = offset, offset + size
        for i in range(num_splits):
            lo, hi = max(bounds[i], file_start), min(bounds[i + 1], file_end)
            if lo < hi:
                splits[i].append(Segment(path, lo - file_start, hi - file_start))
        offset = file_end
    return splits


def read_segment(root: Path, seg: Segment) -> Iterator[str]:
    """Yield the lines (without line terminator) owned by ``seg``."""
    with open(storage.resolve(root, seg.path), "rb") as f:
        if seg.start > 0:
            # The line containing byte start-1 began before our range, so it belongs to the
            # previous segment. If byte start-1 is a newline this consumes just that byte.
            f.seek(seg.start - 1)
            f.readline()
        pos = f.tell()
        while pos < seg.end:
            line = f.readline()
            if not line:
                break
            pos += len(line)
            if line.endswith(b"\n"):
                line = line[:-1]
            if line.endswith(b"\r"):  # also accept Windows line endings
                line = line[:-1]
            yield line.decode("utf-8", errors="replace")
