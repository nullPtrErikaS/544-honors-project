"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mapreduce import jobcode, storage
from mapreduce.splits import compute_splits
from mapreduce.tasks import run_map_task, run_reduce_task

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "examples" / "jobs"


def load_example(name: str) -> jobcode.JobFunctions:
    path = JOBS / name
    return jobcode.load_job(path.read_text(), str(path))


def write_corpus(root: Path, rel_dir: str, files: dict[str, str | bytes]) -> None:
    base = storage.resolve(root, rel_dir)
    base.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (base / name).write_bytes(data.encode() if isinstance(data, str) else data)


SAMPLE = {
    "a.txt": "the quick brown fox\njumps over the lazy dog\n",
    "b.txt": "The dog barks.\nThe fox runs!\n\nno trailing newline here",
    "c.txt": "fox fox fox\r\nwindows line endings\r\n",
}


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    write_corpus(tmp_path, "inputs/sample", SAMPLE)
    return tmp_path


def run_in_process(
    job: jobcode.JobFunctions,
    root: Path,
    input_rel: str,
    num_maps: int,
    num_reduces: int,
    spill_threshold: int = 500_000,
) -> list[str]:
    """Run a job with the real split/map/reduce code but no boss or workers."""
    splits = compute_splits(storage.list_input_files(root, input_rel), num_maps)
    map_dirs = []
    for m, segments in enumerate(splits):
        out = root / "_jobs" / "test" / f"map-{m}"
        run_map_task(job, root, segments, num_reduces, out, spill_threshold=spill_threshold)
        map_dirs.append(out)
    lines: list[str] = []
    for r in range(num_reduces):
        out_file = root / "out" / f"part-r-{r:05d}"
        run_reduce_task(job, map_dirs, r, out_file)
        lines += out_file.read_text(encoding="utf-8").splitlines(keepends=True)
    return lines
