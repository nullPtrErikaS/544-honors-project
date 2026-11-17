"""Sequential reference implementation of a MapReduce job.

``mr local`` uses it so users can debug a job file on their own machine before submitting,
and the tests use it as the oracle that distributed runs must match. It is deliberately
written independently of :mod:`mapreduce.splits` and :mod:`mapreduce.tasks` (whole files,
in-memory grouping, no partitions) so that it can catch bugs in them.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from mapreduce import kv, storage
from mapreduce.jobcode import JobFunctions, pairs


def read_lines(data: bytes) -> list[str]:
    """Split file contents into lines with the same rules as the distributed reader."""
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()  # a trailing newline ends the last line; it doesn't start a new one
    return [
        (ln[:-1] if ln.endswith(b"\r") else ln).decode("utf-8", errors="replace") for ln in lines
    ]


def run_local(job: JobFunctions, root: Path, input_path: str) -> list[str]:
    """Run ``job`` over ``input_path`` (relative to ``root``); return output lines by key."""
    groups: dict[str, list[str]] = defaultdict(list)
    for path, _ in storage.list_input_files(root, input_path):
        for line in read_lines(storage.resolve(root, path).read_bytes()):
            for k, v in pairs(job.mapper(path, line), "mapper"):
                # Round-trip values through JSON so types match the distributed path.
                groups[kv.encode_key(k)].append(kv.encode_value(v))

    out = []
    for encoded_key in sorted(groups):
        values = (kv.decode(v) for v in groups[encoded_key])
        for k, v in pairs(job.reducer(kv.decode(encoded_key), values), "reducer"):
            out.append(kv.format_output(k, v))
    return out
