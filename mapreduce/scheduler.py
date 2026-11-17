"""The boss's brain: job, task, and attempt state, plus the scheduling policy.

This module is pure logic. It does no networking, and the only filesystem work it does goes
through an injected committer. It also takes an injectable clock, so the unit tests can
drive every scenario deterministically.

Vocabulary:
    job      -- one submitted MapReduce program: M map tasks, then R reduce tasks
    task     -- one unit of work (map split m, or reduce partition r)
    attempt  -- one execution of a task on one worker; a task may have several
"""

from __future__ import annotations

import enum
import logging
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from mapreduce import jobfiles
from mapreduce.splits import Segment

log = logging.getLogger(__name__)

DEFAULT_MAX_TASK_FAILURES = 4
DEFAULT_MAX_TASK_LOSSES = 3  # a task that keeps losing its worker is probably killing it
DEFAULT_WORKER_TIMEOUT = 5.0  # seconds of silence before a worker is declared dead
DEFAULT_SPECULATION_FACTOR = 1.5  # straggler = running longer than 1.5x the median task
DEFAULT_SPECULATION_MIN_RUNTIME = 1.0  # ...and longer than this many seconds
RECHECK_INTERVAL = 0.25  # how often a waiting RequestTask re-evaluates time-based rules


class JobState(enum.Enum):
    QUEUED = "QUEUED"
    MAPPING = "MAPPING"
    REDUCING = "REDUCING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @property
    def finished(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED)


class TaskKind(enum.Enum):
    MAP = "MAP"
    REDUCE = "REDUCE"


class AttemptState(enum.Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    LOST = "LOST"


class Committer(Protocol):
    def start_job(self, job_id: str, output: str) -> None: ...
    def commit_reduce(self, output: str, attempt_id: str, partition: int) -> None: ...
    def finish_job(self, job_id: str, output: str, succeeded: bool) -> None: ...


@dataclass(frozen=True)
class JobSpec:
    input_path: str
    output_path: str
    source: str
    filename: str
    num_maps: int
    num_reduces: int
    speculative: bool = True  # launch backup attempts for stragglers
    use_combiner: bool = False  # run the job's combiner on map output


@dataclass(eq=False)
class Attempt:
    id: str
    task: Task
    worker_id: str
    start: float
    speculative: bool = False
    state: AttemptState = AttemptState.RUNNING
    end: float | None = None
    error: str = ""


@dataclass(eq=False)
class Task:
    job: Job
    kind: TaskKind
    index: int
    segments: list[Segment] = field(default_factory=list)  # map tasks only
    attempts: list[Attempt] = field(default_factory=list)
    committed: Attempt | None = None
    failures: int = 0
    losses: int = 0

    @property
    def name(self) -> str:
        return f"{self.job.id}/{self.kind.value.lower()}-{self.index}"

    @property
    def done(self) -> bool:
        return self.committed is not None

    @property
    def running(self) -> list[Attempt]:
        return [a for a in self.attempts if a.state is AttemptState.RUNNING]

    @property
    def pending(self) -> bool:
        return not self.done and not self.running


@dataclass(eq=False)
class Job:
    id: str
    spec: JobSpec
    submitted: float  # scheduler clock
    submitted_wall: float  # unix time, for display
    state: JobState = JobState.QUEUED
    maps: list[Task] = field(default_factory=list)
    reduces: list[Task] = field(default_factory=list)
    finished: float | None = None
    error: str = ""
    counters: Counter[str] = field(default_factory=Counter)

    def phase_tasks(self) -> list[Task]:
        if self.state in (JobState.QUEUED, JobState.MAPPING):
            return self.maps
        if self.state is JobState.REDUCING:
            return self.reduces
        return []

    def all_attempts(self) -> list[Attempt]:
        return [a for t in self.maps + self.reduces for a in t.attempts]


@dataclass(eq=False)
class Worker:
    id: str
    last_seen: float
    alive: bool = True
    current: Attempt | None = None
    succeeded: int = 0


@dataclass(frozen=True)
class Assignment:
    """Everything a worker needs to run one attempt (converted to a protobuf by the boss)."""

    job_id: str
    attempt_id: str
    kind: TaskKind
    task_index: int
    source: str
    filename: str
    num_reduces: int
    segments: list[Segment]
    map_output_dirs: list[str]
    output_dir: str
    output_file: str
    use_combiner: bool


class Scheduler:
    """Thread-safe job/task bookkeeping and scheduling policy.

    All public methods take ``self.lock``. Callers that read job state directly (to build
    status replies) must hold it too.
    """

    def __init__(
        self,
        committer: Committer,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        max_task_failures: int = DEFAULT_MAX_TASK_FAILURES,
        max_task_losses: int = DEFAULT_MAX_TASK_LOSSES,
        worker_timeout: float = DEFAULT_WORKER_TIMEOUT,
        speculation_factor: float = DEFAULT_SPECULATION_FACTOR,
        speculation_min_runtime: float = DEFAULT_SPECULATION_MIN_RUNTIME,
    ):
        self.committer = committer
        self.clock = clock
        self.wall_clock = wall_clock
        self.max_task_failures = max_task_failures
        self.max_task_losses = max_task_losses
        self.worker_timeout = worker_timeout
        self.speculation_factor = speculation_factor
        self.speculation_min_runtime = speculation_min_runtime
        self.lock = threading.Condition()
        self.jobs: dict[str, Job] = {}
        self.workers: dict[str, Worker] = {}
        self.attempts: dict[str, Attempt] = {}
        self._job_seq = 0

    # ------------------------------------------------------------------ jobs

    def submit(self, spec: JobSpec, splits: list[list[Segment]]) -> Job:
        with self.lock:
            self._job_seq += 1
            job = Job(f"job-{self._job_seq:04d}", spec, self.clock(), self.wall_clock())
            job.maps = [Task(job, TaskKind.MAP, i, segments=s) for i, s in enumerate(splits)]
            job.reduces = [Task(job, TaskKind.REDUCE, r) for r in range(spec.num_reduces)]
            # Before any task is handed out: attempts can't create these parents themselves.
            self.committer.start_job(job.id, spec.output_path)
            self.jobs[job.id] = job
            log.info(
                "%s submitted: %s, %d maps, %d reduces",
                job.id,
                spec.filename,
                len(job.maps),
                len(job.reduces),
            )
            self.lock.notify_all()
            return job

    def active_jobs(self) -> list[Job]:
        return [j for j in self.jobs.values() if not j.state.finished]

    # ------------------------------------------------------------------ workers

    def request_task(self, worker_id: str, timeout: float) -> Assignment | None:
        """Long poll: wait up to ``timeout`` seconds for a task for ``worker_id``."""
        deadline = time.monotonic() + timeout
        with self.lock:
            while True:
                assignment = self.try_assign(worker_id)
                remaining = deadline - time.monotonic()
                if assignment is not None or remaining <= 0:
                    return assignment
                self.lock.wait(min(remaining, RECHECK_INTERVAL))

    def try_assign(self, worker_id: str) -> Assignment | None:
        """Hand ``worker_id`` its next attempt, or return None if there is nothing to do."""
        with self.lock:
            worker = self._touch(worker_id)
            if worker.current is not None:
                # A worker only asks for work when idle, so whatever we thought it was running
                # is gone (e.g. the container restarted under the same name). Don't wait for
                # the timeout to reschedule it.
                self._lose(worker.current, "worker asked for new work")
            attempt = self._choose(worker_id)
            return self._assignment(attempt) if attempt else None

    def heartbeat(self, worker_id: str, running: list[str]) -> list[str]:
        """Record that a worker is alive; return which of its attempts it should cancel.

        An attempt is cancelled if the boss no longer considers it running on that worker:
        another attempt of the task won, the job failed, or the worker had been declared
        dead and its work was already handed to someone else.
        """
        with self.lock:
            self._touch(worker_id)
            cancel = []
            for attempt_id in running:
                attempt = self.attempts.get(attempt_id)
                if (
                    attempt is None
                    or attempt.state is not AttemptState.RUNNING
                    or attempt.worker_id != worker_id
                ):
                    cancel.append(attempt_id)
            return cancel

    def check_workers(self) -> list[str]:
        """Declare workers that have been silent too long dead; reschedule their attempts.

        Called periodically by the boss. Returns the ids of newly dead workers. Map output
        that dead workers already committed is *not* redone: it lives on the shared volume,
        not on the worker, so it is still readable.
        """
        with self.lock:
            now = self.clock()
            dead = []
            for worker in self.workers.values():
                if worker.alive and now - worker.last_seen > self.worker_timeout:
                    worker.alive = False
                    dead.append(worker.id)
                    log.warning(
                        "worker %s is dead (silent for %.1fs)", worker.id, now - worker.last_seen
                    )
                    for attempt in list(self.attempts.values()):
                        if attempt.worker_id == worker.id and attempt.state is AttemptState.RUNNING:
                            self._lose(attempt, f"worker {worker.id} died")
            if dead:
                self.lock.notify_all()
            return dead

    def report(
        self,
        worker_id: str,
        attempt_id: str,
        success: bool,
        error: str = "",
        counters: dict[str, int] | None = None,
    ) -> bool:
        """Record an attempt's outcome. Returns whether the result was accepted."""
        with self.lock:
            worker = self._touch(worker_id)
            attempt = self.attempts.get(attempt_id)
            if attempt is None or attempt.state is not AttemptState.RUNNING:
                log.info("ignoring report for %s (no longer running)", attempt_id)
                return False
            if worker.current is attempt:
                worker.current = None
            attempt.end = self.clock()
            if success:
                accepted = self._succeed(attempt, counters or {})
                worker.succeeded += accepted
            else:
                self._fail(attempt, error)
                accepted = True
            self.lock.notify_all()
            return accepted

    # ------------------------------------------------------------------ policy

    def _choose(self, worker_id: str) -> Attempt | None:
        """Pick work for an idle worker from the oldest job that has any.

        Within a job, pending tasks come first, then a backup of a straggler. Only then do
        we look at later jobs, so a queued job can't starve an older job's stragglers.
        """
        for job in self.active_jobs():
            for task in job.phase_tasks():
                if task.pending:
                    return self._launch(task, worker_id)
            if job.spec.speculative and (straggler := self._straggler(job, worker_id)):
                return self._launch(straggler, worker_id, speculative=True)
        return None

    def _straggler(self, job: Job, worker_id: str) -> Task | None:
        """The task in the job's current phase that most deserves a backup attempt, if any.

        A task qualifies if it has exactly one running attempt, on another worker, and that
        attempt has run for longer than ``speculation_factor`` x the median duration of the
        phase's finished tasks (and at least ``speculation_min_runtime``). We need at least
        one finished task to know what "normal" looks like.
        """
        tasks = job.phase_tasks()
        durations = [t.committed.end - t.committed.start for t in tasks if t.committed]
        if not durations:
            return None
        threshold = max(
            self.speculation_min_runtime, self.speculation_factor * statistics.median(durations)
        )
        now = self.clock()
        best, best_runtime = None, threshold
        for task in tasks:
            running = task.running
            if task.done or len(running) != 1 or running[0].worker_id == worker_id:
                continue
            runtime = now - running[0].start
            if runtime > best_runtime:
                best, best_runtime = task, runtime
        return best

    def _launch(self, task: Task, worker_id: str, speculative: bool = False) -> Attempt:
        job = task.job
        if job.state is JobState.QUEUED:
            job.state = JobState.MAPPING
        kind = "m" if task.kind is TaskKind.MAP else "r"
        attempt = Attempt(
            id=f"{job.id}-{kind}{task.index:04d}-a{len(task.attempts)}",
            task=task,
            worker_id=worker_id,
            start=self.clock(),
            speculative=speculative,
        )
        task.attempts.append(attempt)
        self.attempts[attempt.id] = attempt
        self.workers[worker_id].current = attempt
        log.info("%s -> %s%s", attempt.id, worker_id, " (backup)" if speculative else "")
        return attempt

    def _succeed(self, attempt: Attempt, counters: dict[str, int]) -> bool:
        task, job = attempt.task, attempt.task.job
        if task.kind is TaskKind.REDUCE:
            try:
                self.committer.commit_reduce(job.spec.output_path, attempt.id, task.index)
            except OSError as e:
                self._fail(attempt, f"commit failed: {e}")
                return False
        attempt.state = AttemptState.SUCCEEDED
        task.committed = attempt
        job.counters.update(counters)
        for other in task.running:  # the losing copy of a speculated task
            log.info("%s won; killing %s", attempt.id, other.id)
            self._kill(other)
        self._advance(job)
        return True

    def _fail(self, attempt: Attempt, error: str) -> None:
        task = attempt.task
        attempt.state = AttemptState.FAILED
        attempt.error = error
        task.failures += 1
        last_line = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        log.warning(
            "%s failed (%d/%d): %s", attempt.id, task.failures, self.max_task_failures, last_line
        )
        if task.failures >= self.max_task_failures:
            self._finish(
                task.job,
                JobState.FAILED,
                f"{task.name} failed {task.failures} times; last error:\n{error}",
            )

    def _advance(self, job: Job) -> None:
        """Move a job to its next phase when the current phase is complete."""
        if job.state is JobState.MAPPING and all(t.done for t in job.maps):
            job.state = JobState.REDUCING
            log.info("%s: map phase done, starting reduce phase", job.id)
        if job.state is JobState.REDUCING and all(t.done for t in job.reduces):
            self._finish(job, JobState.SUCCEEDED)

    def _finish(self, job: Job, state: JobState, error: str = "") -> None:
        job.state, job.error, job.finished = state, error, self.clock()
        for attempt in job.all_attempts():
            if attempt.state is AttemptState.RUNNING:
                self._kill(attempt)
        self.committer.finish_job(job.id, job.spec.output_path, state is JobState.SUCCEEDED)
        log.info("%s %s in %.2fs", job.id, state.value, job.finished - job.submitted)

    def _kill(self, attempt: Attempt) -> None:
        """Stop an attempt we no longer need. The worker learns via its next heartbeat."""
        self._stop(attempt, AttemptState.KILLED)

    def _lose(self, attempt: Attempt, reason: str) -> None:
        """An attempt's worker died. The task becomes pending again unless another attempt
        is still running.

        A lost attempt is usually the worker's fault, not the task's, so it doesn't count
        toward the failure limit. But a task whose code kills its worker (``os._exit``, a
        segfault, an out-of-memory kill) would be retried until no workers were left, so
        a task that loses its worker ``max_task_losses`` times fails the job.
        """
        log.warning("%s lost: %s", attempt.id, reason)
        self._stop(attempt, AttemptState.LOST)
        attempt.error = reason
        task = attempt.task
        task.losses += 1
        if task.losses >= self.max_task_losses and not task.job.state.finished:
            self._finish(
                task.job,
                JobState.FAILED,
                f"{task.name} lost its worker {task.losses} times; its code may be crashing "
                f"the worker process (last: {reason})",
            )

    def _stop(self, attempt: Attempt, state: AttemptState) -> None:
        attempt.state, attempt.end = state, self.clock()
        worker = self.workers.get(attempt.worker_id)
        if worker and worker.current is attempt:
            worker.current = None

    # ------------------------------------------------------------------ helpers

    def _touch(self, worker_id: str) -> Worker:
        worker = self.workers.get(worker_id)
        if worker is None:
            worker = self.workers[worker_id] = Worker(worker_id, self.clock())
            log.info("worker %s joined", worker_id)
        elif not worker.alive:
            worker.alive = True
            log.info("worker %s is back", worker_id)
        worker.last_seen = self.clock()
        return worker

    def _assignment(self, attempt: Attempt) -> Assignment:
        task, job = attempt.task, attempt.task.job
        if task.kind is TaskKind.MAP:
            inputs: list[str] = []
            out_dir, out_file = jobfiles.map_attempt_dir(job.id, attempt.id), ""
        else:
            # Reducers read only the *committed* attempt of every map task.
            inputs = [
                jobfiles.map_attempt_dir(job.id, t.committed.id) for t in job.maps if t.committed
            ]
            out_dir = ""
            out_file = jobfiles.reduce_attempt_file(job.spec.output_path, attempt.id, task.index)
        return Assignment(
            job_id=job.id,
            attempt_id=attempt.id,
            kind=task.kind,
            task_index=task.index,
            source=job.spec.source,
            filename=job.spec.filename,
            num_reduces=job.spec.num_reduces,
            segments=task.segments,
            map_output_dirs=inputs,
            output_dir=out_dir,
            output_file=out_file,
            use_combiner=job.spec.use_combiner,
        )
