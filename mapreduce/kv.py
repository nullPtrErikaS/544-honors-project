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
import itertools
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

MERGE_FACTOR = 100  # most run files one merge reads at once (like Hadoop's io.sort.factor)
_merge_ids = itertools.count()


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


def write_run(path: Path, records: Iterable[Record]) -> tuple[int, int]:
    """Write already-sorted records to ``path``; return ``(records, bytes)`` written."""
    nrecords = nbytes = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in records:
            line = f"{k}\t{v}\n"
            f.write(line)
            nrecords += 1
            nbytes += len(line)  # encoded JSON is ASCII, so chars == bytes
    return nrecords, nbytes


def read_run(path: Path) -> Iterator[Record]:
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            yield k, v


def sort_records(records: list[Record]) -> list[Record]:
    records.sort(key=_by_key)
    return records


def merge_runs(
    paths: list[Path],
    stack: ExitStack,
    tmp_dir: Path | None = None,
    factor: int = MERGE_FACTOR,
) -> Iterator[Record]:
    """K-way merge of sorted run files (streaming; files are closed by ``stack``).

    At most ``factor`` files are open at once, so a reduce task with thousands of map inputs
    doesn't run out of file descriptors. With more inputs, groups of ``factor`` runs are
    first merged into temporary runs in ``tmp_dir``, in as many passes as needed. The
    temporary runs are deleted when ``stack`` closes. Equal keys keep the order of ``paths``.
    """
    if len(paths) > factor:
        if tmp_dir is None:
            raise ValueError(f"merging more than {factor} runs needs a tmp_dir")
        paths = _merge_passes(paths, tmp_dir, factor, stack)
    readers = [stack.enter_context(closing(read_run(p))) for p in paths]
    return heapq.merge(*readers, key=_by_key)


def _merge_passes(paths: list[Path], tmp_dir: Path, factor: int, stack: ExitStack) -> list[Path]:
    temps: list[Path] = []  # temporary runs that still exist
    stack.callback(lambda: [p.unlink(missing_ok=True) for p in temps])
    while len(paths) > factor:
        merged = []
        for i in range(0, len(paths), factor):
            group = paths[i : i + factor]
            out = tmp_dir / f"_merge-{next(_merge_ids):06d}.kv"  # hidden: never job input
            with ExitStack() as inner:
                write_run(out, merge_runs(group, inner))
            for p in group:  # inputs from an earlier pass are no longer needed
                if p in temps:
                    p.unlink()
                    temps.remove(p)
            temps.append(out)
            merged.append(out)
        paths = merged
    return paths


def group_by_key(records: Iterable[Record]) -> Iterator[tuple[str, Iterator[Record]]]:
    """Group adjacent records with equal encoded keys (input must be sorted)."""
    return groupby(records, key=_by_key)


# TAB plus every character ``str.splitlines()`` treats as a line break. An output string
# containing one is JSON-encoded, so each record stays one line for any line reader.
_SEPARATORS = "\t\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
# json.dumps escapes control characters below U+0020, but with ensure_ascii=False it leaves
# these three raw.
_EXTRA_ESCAPES = str.maketrans({"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"})


def _text(x: object) -> str:
    if isinstance(x, str) and not any(c in x for c in _SEPARATORS):
        return x
    # ensure_ascii=False: the output is UTF-8, so "café" inside a list stays readable.
    return json.dumps(x, ensure_ascii=False).translate(_EXTRA_ESCAPES)


def format_output(key: object, value: object) -> str:
    """One line of final job output: ``key TAB value``.

    Strings are written as-is unless they contain a tab or a line break; everything else
    (and those strings) as JSON.
    """
    return f"{_text(key)}\t{_text(value)}\n"
