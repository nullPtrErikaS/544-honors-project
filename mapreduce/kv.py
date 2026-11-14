"""Intermediate key/value format, partitioning, sorted runs, and the final output format.

Intermediate files are *sorted runs*: one ``json(key) TAB json(value)`` record per line,
sorted by the JSON-encoded key string. Working on encoded keys means

* the sort order is total even when keys have mixed types,
* reducers can compare and group keys without decoding them, and
* partitioning can hash a stable byte string.

``json.dumps`` escapes tabs and newlines, so TAB and LF are safe separators.
"""

from __future__ import annotations

import heapq
import json
import zlib
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing
from itertools import groupby
from operator import itemgetter
from pathlib import Path

Record = tuple[str, str]  # (encoded key, encoded value)

# sort_keys makes dict keys encode identically regardless of insertion order, so equal keys
# always land in the same partition and group.
encode_key = json.JSONEncoder(separators=(",", ":"), sort_keys=True).encode
encode_value = json.JSONEncoder(separators=(",", ":")).encode
decode = json.loads

_by_key = itemgetter(0)


def partition(encoded_key: str, num_partitions: int) -> int:
    """Reduce partition for a key.

    Uses CRC32 rather than ``hash()``: Python randomizes ``hash(str)`` per process
    (PYTHONHASHSEED), so different workers would disagree about where a key goes.
    """
    return zlib.crc32(encoded_key.encode()) % num_partitions


def partition_filename(p: int) -> str:
    return f"part-{p:05d}.kv"


def output_filename(p: int) -> str:
    return f"part-r-{p:05d}"


def write_run(path: Path, records: Iterable[Record]) -> int:
    """Write already-sorted records to ``path``; return bytes written."""
    nbytes = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in records:
            line = f"{k}\t{v}\n"
            f.write(line)
            nbytes += len(line)  # encoded JSON is ASCII, so chars == bytes
    return nbytes


def read_run(path: Path) -> Iterator[Record]:
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            yield k, v


def sort_records(records: list[Record]) -> list[Record]:
    records.sort(key=_by_key)
    return records


def merge_runs(paths: list[Path], stack: ExitStack) -> Iterator[Record]:
    """K-way merge of sorted run files (streaming; files are closed by ``stack``)."""
    readers = [stack.enter_context(closing(read_run(p))) for p in paths]
    return heapq.merge(*readers, key=_by_key)


def group_by_key(records: Iterable[Record]) -> Iterator[tuple[str, Iterator[Record]]]:
    """Group adjacent records with equal encoded keys (input must be sorted)."""
    return groupby(records, key=_by_key)


def _text(x: object) -> str:
    if isinstance(x, str) and not any(c in x for c in "\t\n\r"):
        return x
    return json.dumps(x)


def format_output(key: object, value: object) -> str:
    """One line of final job output: ``key TAB value``.

    Strings are written as-is (unless they contain a tab/newline); everything else as JSON.
    """
    return f"{_text(key)}\t{_text(value)}\n"
