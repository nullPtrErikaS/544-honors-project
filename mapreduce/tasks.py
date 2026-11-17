"""Execute one map task or one reduce task.

These functions only touch the filesystem; they know nothing about gRPC or the boss, so
they can be unit-tested directly and reused by the worker.

Map task:    input segments --mapper--> buffer --sort/spill--> R sorted partition files
Reduce task: partition r of every map's output --k-way merge--> reducer --> one output file
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from mapreduce import kv
from mapreduce.jobcode import JobFunctions, pairs
from mapreduce.splits import Segment, read_segment

DEFAULT_SPILL_THRESHOLD = 500_000  # buffered map output records before sorting to disk
DEFAULT_SPILL_BYTES = 64 << 20  # ...or this much encoded data, whichever comes first
CANCEL_CHECK_EVERY = 1_000  # reduce input values between checks of the cancel flag

Counters = dict[str, int]


class TaskCancelled(BaseException):
    """The boss told us to stop (e.g. another attempt of this task already finished).

    A ``BaseException``, like ``GeneratorExit``: it is raised inside user code (the reducer's
    ``values`` iterator), and a reducer's ``except Exception`` must not swallow it.
    """


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()


class MapOutputBuffer:
    """Collects map output, partitions it, and writes sorted partition files.

    Records are buffered per partition. When ``spill_threshold`` records or ``spill_bytes``
    of encoded data are buffered, each partition is sorted and written as a *spill* run. At
    the end, spills are merged into one sorted file per partition, so memory use is bounded
    by the thresholds.
    """

    def __init__(
        self,
        num_partitions: int,
        out_dir: Path,
        spill_threshold: int,
        counters: Counters,
        spill_bytes: int = DEFAULT_SPILL_BYTES,
    ):
        self.num_partitions = num_partitions
        self.out_dir = out_dir
        self.spill_threshold = spill_threshold
        self.spill_bytes = spill_bytes
        self.counters = counters
        self.buffers: list[list[kv.Record]] = [[] for _ in range(num_partitions)]
        self.buffered = 0
        self.buffered_bytes = 0
        self.spill_files: list[list[Path]] = [[] for _ in range(num_partitions)]
        self.num_spills = 0

    def add(self, key: Any, value: Any) -> None:
        k, v = kv.encode_key(key), kv.encode_value(value)
        self.buffers[kv.partition(k, self.num_partitions)].append((k, v))
        self.buffered += 1
        self.buffered_bytes += len(k) + len(v)
        if self.buffered >= self.spill_threshold or self.buffered_bytes >= self.spill_bytes:
            self._spill()

    def _spill(self) -> None:
        for p, records in enumerate(self.buffers):
            if records:
                path = self.out_dir / f"spill-{self.num_spills:04d}-{p:05d}.kv"
                kv.write_run(path, kv.sort_records(records))
                self.spill_files[p].append(path)
        self.num_spills += 1
        self.buffers = [[] for _ in range(self.num_partitions)]
        self.buffered = self.buffered_bytes = 0

    def finish(self) -> None:
        """Write the final ``part-NNNNN.kv`` file for every partition (empty ones too)."""
        if self.num_spills and self.buffered:
            self._spill()
        nbytes = 0
        for p in range(self.num_partitions):
            final = self.out_dir / kv.partition_filename(p)
            if not self.num_spills:
                nbytes += kv.write_run(final, kv.sort_records(self.buffers[p]))
                continue
            with ExitStack() as stack:
                merged = kv.merge_runs(self.spill_files[p], stack, tmp_dir=self.out_dir)
                nbytes += kv.write_run(final, merged)
            for path in self.spill_files[p]:
                path.unlink()
        self.counters["spills"] = self.num_spills
        self.counters["intermediate_bytes"] = nbytes


def run_map_task(
    job: JobFunctions,
    root: Path,
    segments: list[Segment],
    num_reduce: int,
    out_dir: Path,
    *,
    spill_threshold: int = DEFAULT_SPILL_THRESHOLD,
    spill_bytes: int = DEFAULT_SPILL_BYTES,
    cancel: threading.Event | None = None,
) -> Counters:
    """Run ``job.mapper`` over every line of ``segments``; write partitions to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counters: Counters = {"map_input_records": 0, "map_output_records": 0}
    buffer = MapOutputBuffer(num_reduce, out_dir, spill_threshold, counters, spill_bytes)
    n_in = n_out = 0
    for seg in segments:
        for line in read_segment(root, seg):
            n_in += 1
            # Every line: the attempt being cancelled is usually the slow one, and is_set()
            # costs far less than running the mapper.
            _check_cancel(cancel)
            for k, v in pairs(job.mapper(seg.path, line), "mapper"):
                buffer.add(k, v)
                n_out += 1
    counters["map_input_records"] = n_in
    counters["map_output_records"] = n_out
    buffer.finish()
    return counters


def run_reduce_task(
    job: JobFunctions,
    map_output_dirs: list[Path],
    partition: int,
    out_file: Path,
    *,
    cancel: threading.Event | None = None,
) -> Counters:
    """Merge partition ``partition`` from every map output dir, reduce, write ``out_file``."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    counters: Counters = {
        "reduce_input_records": 0,
        "reduce_input_groups": 0,
        "reduce_output_records": 0,
    }
    inputs = [d / kv.partition_filename(partition) for d in map_output_dirs]

    def values(group: Iterator[kv.Record]) -> Iterator[Any]:
        for _, v in group:
            counters["reduce_input_records"] += 1
            if counters["reduce_input_records"] % CANCEL_CHECK_EVERY == 0:
                _check_cancel(cancel)
            yield kv.decode(v)

    with ExitStack() as stack:
        out = stack.enter_context(open(out_file, "w", encoding="utf-8", newline="\n"))
        merged = kv.merge_runs(inputs, stack, tmp_dir=out_file.parent)
        for encoded_key, group in kv.group_by_key(merged):
            _check_cancel(cancel)  # every key, and every CANCEL_CHECK_EVERY values within one
            counters["reduce_input_groups"] += 1
            for k, v in pairs(job.reducer(kv.decode(encoded_key), values(group)), "reducer"):
                out.write(kv.format_output(k, v))
                counters["reduce_output_records"] += 1
    counters["output_bytes"] = out_file.stat().st_size
    return counters
