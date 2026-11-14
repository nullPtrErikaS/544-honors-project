"""Unit tests for the scheduling state machine (no threads, no network, fake clock)."""

from __future__ import annotations

import pytest

from mapreduce.scheduler import (
    AttemptState,
    JobSpec,
    JobState,
    Scheduler,
    TaskKind,
)
from mapreduce.splits import Segment


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeCommitter:
    def __init__(self) -> None:
        self.committed: list[tuple[str, int]] = []
        self.finished: list[tuple[str, bool]] = []
        self.fail_commits = False

    def commit_reduce(self, output: str, attempt_id: str, partition: int) -> None:
        if self.fail_commits:
            raise FileNotFoundError("temp file missing")
        self.committed.append((attempt_id, partition))

    def finish_job(self, job_id: str, output: str, succeeded: bool) -> None:
        self.finished.append((job_id, succeeded))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def committer() -> FakeCommitter:
    return FakeCommitter()


@pytest.fixture
def sched(clock, committer) -> Scheduler:
    return Scheduler(committer, clock=clock, wall_clock=lambda: 0.0)


def submit(sched: Scheduler, maps: int = 2, reduces: int = 2, **spec_kw):
    spec = JobSpec("in", "out", "src", "job.py", maps, reduces, **spec_kw)
    splits = [[Segment(f"in/f{m}", 0, 10)] for m in range(maps)]
    return sched.submit(spec, splits)


def run_to_completion(sched: Scheduler, workers=("w1",)) -> None:
    for _ in range(1000):
        progressed = False
        for w in workers:
            a = sched.try_assign(w)
            if a:
                sched.report(w, a.attempt_id, True, counters={"n": 1})
                progressed = True
        if not progressed:
            return
    raise AssertionError("did not finish")


def test_maps_then_reduces(sched, committer):
    job = submit(sched, maps=2, reduces=1)
    a1, a2 = sched.try_assign("w1"), sched.try_assign("w2")
    assert (a1.kind, a1.task_index, a2.task_index) == (TaskKind.MAP, 0, 1)
    assert a1.segments == [Segment("in/f0", 0, 10)]
    assert a1.output_dir == f"_jobs/{job.id}/map/{a1.attempt_id}"
    # No reduce can start until every map is committed.
    assert sched.try_assign("w3") is None
    assert job.state is JobState.MAPPING

    sched.report("w1", a1.attempt_id, True)
    assert sched.try_assign("w1") is None
    sched.report("w2", a2.attempt_id, True)
    assert job.state is JobState.REDUCING

    r = sched.try_assign("w1")
    assert r.kind is TaskKind.REDUCE
    assert r.map_output_dirs == [a1.output_dir, a2.output_dir]
    assert r.output_file == f"out/_temporary/{r.attempt_id}/part-r-00000"
    sched.report("w1", r.attempt_id, True)
    assert job.state is JobState.SUCCEEDED
    assert committer.committed == [(r.attempt_id, 0)]
    assert committer.finished == [(job.id, True)]


def test_jobs_run_in_fifo_order(sched):
    first, second = submit(sched, maps=1, reduces=1), submit(sched, maps=1, reduces=1)
    a = sched.try_assign("w1")
    assert a.job_id == first.id
    # While the first job's only map runs, an idle worker may start the second job.
    assert sched.try_assign("w2").job_id == second.id


def test_counters_sum_committed_attempts(sched):
    job = submit(sched, maps=3, reduces=2)
    run_to_completion(sched, workers=("w1", "w2"))
    assert job.state is JobState.SUCCEEDED
    assert job.counters["n"] == 5


def test_failed_attempt_is_retried_then_job_fails(sched, committer):
    job = submit(sched, maps=1, reduces=1)
    for i in range(sched.max_task_failures):
        a = sched.try_assign(f"w{i}")
        assert a is not None and a.task_index == 0
        sched.report(f"w{i}", a.attempt_id, False, "Traceback...\nKeyError: 'x'")
    assert job.state is JobState.FAILED
    assert "map-0 failed 4 times" in job.error and "KeyError" in job.error
    assert committer.finished == [(job.id, False)]
    assert sched.try_assign("w9") is None


def test_one_failure_then_success(sched):
    job = submit(sched, maps=1, reduces=1)
    a = sched.try_assign("w1")
    sched.report("w1", a.attempt_id, False, "boom")
    retry = sched.try_assign("w2")
    assert retry.task_index == 0 and retry.attempt_id.endswith("-a1")
    sched.report("w2", retry.attempt_id, True)
    run_to_completion(sched)
    assert job.state is JobState.SUCCEEDED


def test_failed_commit_counts_as_failure(sched, committer):
    job = submit(sched, maps=1, reduces=1)
    m = sched.try_assign("w1")
    sched.report("w1", m.attempt_id, True)
    committer.fail_commits = True
    r = sched.try_assign("w1")
    assert sched.report("w1", r.attempt_id, True) is False
    assert job.reduces[0].attempts[0].state is AttemptState.FAILED
    assert job.reduces[0].pending


def test_duplicate_or_unknown_reports_are_rejected(sched):
    submit(sched, maps=1, reduces=1)
    a = sched.try_assign("w1")
    assert sched.report("w1", a.attempt_id, True) is True
    assert sched.report("w1", a.attempt_id, True) is False
    assert sched.report("w1", "job-9999-m0000-a0", True) is False


def test_attempt_timing_uses_clock(sched, clock):
    job = submit(sched, maps=1, reduces=1)
    a = sched.try_assign("w1")
    clock.advance(2.5)
    sched.report("w1", a.attempt_id, True)
    attempt = job.maps[0].attempts[0]
    assert attempt.end - attempt.start == 2.5
