"""Unit tests for job directories on shared storage: setup, commit, and cleanup."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mapreduce import jobfiles
from mapreduce.jobcode import JobFunctions
from mapreduce.jobfiles import FilesystemCommitter
from mapreduce.tasks import run_map_task, run_reduce_task

JOB = "job-0001"


def wait_gone(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return not path.exists()


def write_reduce_attempt(root: Path, attempt_id: str, text: str) -> None:
    out_file = root / jobfiles.reduce_attempt_file("out", attempt_id, 0)
    out_file.parent.mkdir()  # like a worker: only its own attempt directory
    out_file.write_text(text)


@pytest.fixture
def committer(tmp_path) -> FilesystemCommitter:
    (tmp_path / "out").mkdir()  # the boss creates the output directory at submit time
    c = FilesystemCommitter(tmp_path)
    c.start_job(JOB, "out")
    return c


def test_start_job_creates_the_parents_attempts_write_into(tmp_path, committer):
    assert (tmp_path / "_jobs" / JOB / "map").is_dir()
    assert (tmp_path / "out" / "_temporary").is_dir()


def test_success_keeps_committed_output_and_removes_the_rest(tmp_path, committer):
    (tmp_path / jobfiles.map_attempt_dir(JOB, "m0")).mkdir()
    write_reduce_attempt(tmp_path, "r0", "x\t1\n")
    committer.commit_reduce("out", "r0", 0)
    write_reduce_attempt(tmp_path, "r0-backup", "the losing copy")
    committer.finish_job(JOB, "out", succeeded=True)

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["_SUCCESS", "part-r-00000"]
    assert (tmp_path / "out" / "part-r-00000").read_text() == "x\t1\n"
    assert not (tmp_path / "_jobs" / JOB).exists()  # gone right away, not eventually
    assert wait_gone(tmp_path / "_jobs" / "_trash" / JOB)


def test_failure_removes_partial_output(tmp_path, committer):
    write_reduce_attempt(tmp_path, "r0", "x\t1\n")
    committer.commit_reduce("out", "r0", 0)  # one partition made it before the job failed
    committer.finish_job(JOB, "out", succeeded=False)

    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "_jobs" / JOB).exists()
    assert wait_gone(tmp_path / "_jobs" / "_trash" / JOB)


def test_killed_attempt_cannot_recreate_a_finished_jobs_directories(tmp_path, committer):
    committer.finish_job(JOB, "out", succeeded=True)
    job = JobFunctions(mapper=lambda k, v: [(v, 1)], reducer=lambda k, vs: [(k, sum(vs))])
    late_reduce = tmp_path / jobfiles.reduce_attempt_file("out", "late-r", 0)
    late_map = tmp_path / jobfiles.map_attempt_dir(JOB, "late-m")

    with pytest.raises(FileNotFoundError):
        run_reduce_task(job, [], 0, late_reduce)
    with pytest.raises(FileNotFoundError):
        run_map_task(job, tmp_path, [], 1, late_map)
    assert not (tmp_path / "out" / "_temporary").exists()
    assert not (tmp_path / "_jobs" / JOB).exists()
