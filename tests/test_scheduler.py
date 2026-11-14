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


# ------------------------------------------------------------------ worker failures


def test_dead_worker_attempt_is_rescheduled(sched, clock):
    job = submit(sched, maps=2, reduces=1)
    a0 = sched.try_assign("w1")
    sched.try_assign("w2")
    sched.report("w1", a0.attempt_id, True)

    clock.advance(sched.worker_timeout - 1)
    sched.heartbeat("w1", [])
    assert sched.check_workers() == []
    clock.advance(2)  # w2 now silent for longer than the timeout; w1 is not
    assert sched.check_workers() == ["w2"]

    lost = job.maps[1].attempts[0]
    assert lost.state is AttemptState.LOST and "w2 died" in lost.error
    retry = sched.try_assign("w1")
    assert (retry.task_index, retry.attempt_id) == (1, f"{job.id}-m0001-a1")
    # Map 0 was committed by w1 and is not redone.
    assert len(job.maps[0].attempts) == 1


def test_committed_map_output_of_dead_worker_is_kept(sched, clock):
    """Map output lives on shared storage, so it survives the worker (unlike the paper)."""
    job = submit(sched, maps=2, reduces=1)
    for w in ("w1", "w2"):
        a = sched.try_assign(w)
        sched.report(w, a.attempt_id, True)
    assert job.state is JobState.REDUCING
    r = sched.try_assign("w2")
    clock.advance(sched.worker_timeout + 1)
    sched.heartbeat("w1", [])
    sched.check_workers()

    assert job.state is JobState.REDUCING
    assert all(len(t.attempts) == 1 and t.done for t in job.maps)
    retry = sched.try_assign("w1")
    assert retry.kind is TaskKind.REDUCE and retry.attempt_id != r.attempt_id
    sched.report("w1", retry.attempt_id, True)
    assert job.state is JobState.SUCCEEDED


def test_lost_attempts_do_not_count_as_failures(sched, clock):
    job = submit(sched, maps=1, reduces=1)
    for i in range(sched.max_task_failures + 2):
        sched.try_assign(f"w{i}")
        clock.advance(sched.worker_timeout + 1)
        sched.check_workers()
    assert job.state is JobState.MAPPING
    assert job.maps[0].failures == 0


def test_returning_worker_is_told_to_cancel_and_its_report_is_ignored(sched, clock):
    job = submit(sched, maps=1, reduces=1)
    a = sched.try_assign("w1")
    clock.advance(sched.worker_timeout + 1)
    sched.check_workers()
    assert not sched.workers["w1"].alive
    retry = sched.try_assign("w2")

    # w1 was only partitioned away. It comes back, still working on the old attempt.
    assert sched.heartbeat("w1", [a.attempt_id]) == [a.attempt_id]
    assert sched.workers["w1"].alive
    assert sched.report("w1", a.attempt_id, True) is False
    assert sched.report("w2", retry.attempt_id, True) is True
    assert job.maps[0].committed.worker_id == "w2"


def test_worker_asking_for_work_abandons_its_previous_attempt(sched):
    """A restarted container reuses its hostname; don't wait for the timeout."""
    job = submit(sched, maps=2, reduces=1)
    old = sched.try_assign("w1")
    new = sched.try_assign("w1")
    assert job.maps[0].attempts[0].state is AttemptState.LOST
    assert new.task_index == 0 and new.attempt_id != old.attempt_id


def test_heartbeat_cancels_attempts_of_failed_job(sched):
    job = submit(sched, maps=2, reduces=1)
    sched.max_task_failures = 1
    a0, a1 = sched.try_assign("w1"), sched.try_assign("w2")
    sched.report("w1", a0.attempt_id, False, "boom")
    assert job.state is JobState.FAILED
    assert sched.heartbeat("w2", [a1.attempt_id]) == [a1.attempt_id]
    assert sched.heartbeat("w2", []) == []


# ------------------------------------------------------------------ speculative execution


def start_wave(sched, clock, job, workers, finish, duration=2.0):
    """Assign one task to each worker, then let the workers in ``finish`` complete."""
    attempts = {w: sched.try_assign(w) for w in workers}
    clock.advance(duration)
    for w in finish:
        sched.report(w, attempts[w].attempt_id, True)
    return attempts


def test_no_backup_until_a_task_has_finished(sched, clock):
    submit(sched, maps=2, reduces=1)
    sched.try_assign("w1")
    sched.try_assign("w2")
    clock.advance(100)
    assert sched.try_assign("w3") is None


def test_backup_launched_for_straggler(sched, clock):
    job = submit(sched, maps=3, reduces=1)
    start_wave(sched, clock, job, ["w1", "w2", "w3"], finish=["w1", "w2"], duration=2.0)
    # Median finished duration is 2 s, so the threshold is 3 s; w3 has run 2 s.
    assert sched.try_assign("w1") is None
    clock.advance(1.5)
    backup = sched.try_assign("w1")
    assert backup.task_index == 2 and backup.attempt_id.endswith("-a1")
    assert job.maps[2].attempts[1].speculative
    # Only one backup per task.
    assert sched.try_assign("w2") is None


def test_straggler_never_backs_itself_up(sched, clock):
    job = submit(sched, maps=2, reduces=1)
    start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"])
    clock.advance(10)
    # w2 is busy with the straggler, but even if it asked, it must not get its own task.
    assert sched._straggler(job, "w2") is None
    assert sched._straggler(job, "w1") is job.maps[1]


def test_backup_wins_and_original_is_cancelled(sched, clock):
    job = submit(sched, maps=2, reduces=1)
    first = start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"])
    clock.advance(5)
    backup = sched.try_assign("w1")
    clock.advance(2)
    assert sched.report("w1", backup.attempt_id, True) is True

    original = job.maps[1].attempts[0]
    assert original.state is AttemptState.KILLED
    assert job.maps[1].committed.id == backup.attempt_id
    assert sched.heartbeat("w2", [first["w2"].attempt_id]) == [first["w2"].attempt_id]
    assert sched.report("w2", first["w2"].attempt_id, True) is False
    assert job.state is JobState.REDUCING


def test_original_wins_and_backup_is_cancelled(sched, clock):
    job = submit(sched, maps=2, reduces=1)
    first = start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"])
    clock.advance(5)
    backup = sched.try_assign("w1")
    sched.report("w2", first["w2"].attempt_id, True)
    assert job.maps[1].attempts[1].state is AttemptState.KILLED
    assert sched.heartbeat("w1", [backup.attempt_id]) == [backup.attempt_id]


def test_speculation_applies_to_reduce_phase(sched, clock):
    job = submit(sched, maps=1, reduces=2)
    run = sched.try_assign("w1")
    sched.report("w1", run.attempt_id, True)
    start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"])
    clock.advance(5)
    backup = sched.try_assign("w1")
    assert backup.kind is TaskKind.REDUCE and backup.task_index == 1


def test_short_tasks_are_not_speculated(sched, clock):
    job = submit(sched, maps=2, reduces=1)
    start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"], duration=0.1)
    clock.advance(0.5)  # 6x the median, but below the 1 s minimum
    assert sched.try_assign("w1") is None


def test_speculation_can_be_disabled_per_job(sched, clock):
    job = submit(sched, maps=2, reduces=1, speculative=False)
    start_wave(sched, clock, job, ["w1", "w2"], finish=["w1"])
    clock.advance(100)
    assert sched.try_assign("w1") is None
