"""A worker process: repeatedly ask the boss for a task, run it, report the result.

Each worker runs one task at a time. In Docker Compose every worker container is limited
to 1 CPU, so to use more CPUs you add workers, not threads.

Usage: ``python -m mapreduce.worker [--boss boss:5440] [--data /data] [--id worker1]``
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
import traceback
from pathlib import Path

import grpc

from mapreduce import jobcode, mr_pb2, mr_pb2_grpc, storage
from mapreduce.splits import Segment
from mapreduce.tasks import Counters, TaskCancelled, run_map_task, run_reduce_task

log = logging.getLogger("worker")

RPC_TIMEOUT = 30.0
MAX_ERROR_CHARS = 8_000
DEFAULT_HEARTBEAT_INTERVAL = 1.0


class Worker:
    def __init__(
        self,
        worker_id: str,
        boss: str,
        root: Path,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        self.id = worker_id
        self.root = root
        self.heartbeat_interval = heartbeat_interval
        self.channel = grpc.insecure_channel(boss)
        self.stub = mr_pb2_grpc.CoordinatorStub(self.channel)
        self._jobs: dict[str, jobcode.JobFunctions] = {}  # job_id -> loaded job file
        self._running: dict[str, threading.Event] = {}  # attempt_id -> its cancel flag
        self._stop = threading.Event()

    def run(self) -> None:
        """Main loop. Returns only after :meth:`stop`."""
        log.info("worker %s started", self.id)
        heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True, name="heartbeat")
        boss_down_logged = False
        while not self._stop.is_set():
            try:
                assignment = self.stub.RequestTask(
                    mr_pb2.TaskRequest(worker_id=self.id), timeout=RPC_TIMEOUT
                )
            except grpc.RpcError as e:
                if not boss_down_logged:
                    log.warning("cannot reach boss (%s); retrying", e.code())
                    boss_down_logged = True
                self._stop.wait(1.0)
                continue
            boss_down_logged = False
            if heartbeat.ident is None:
                # Heartbeat only once the boss has seen us ask for work. Otherwise a worker
                # still in its startup retry would already be listed as alive but idle.
                heartbeat.start()
            if assignment.has_task:
                self.execute(assignment)

    def stop(self) -> None:
        self._stop.set()

    def _heartbeat_loop(self) -> None:
        """Tell the boss we're alive; stop attempts the boss no longer wants."""
        while not self._stop.wait(self.heartbeat_interval):
            request = mr_pb2.HeartbeatRequest(
                worker_id=self.id, running_attempts=list(self._running)
            )
            try:
                reply = self.stub.Heartbeat(request, timeout=RPC_TIMEOUT)
            except grpc.RpcError:
                continue  # the main loop logs connectivity problems
            for attempt_id in reply.cancel_attempts:
                if (cancel := self._running.get(attempt_id)) is not None:
                    log.info("boss cancelled %s", attempt_id)
                    cancel.set()

    def execute(self, a: mr_pb2.TaskAssignment) -> None:
        log.info("running %s", a.attempt_id)
        started = time.monotonic()
        cancel = self._running[a.attempt_id] = threading.Event()
        counters: Counters = {}
        try:
            counters = self._run(a, cancel)
            success, error = True, ""
        except TaskCancelled:
            success, error = False, "cancelled by the boss"
        except Exception:
            success, error = False, traceback.format_exc()[-MAX_ERROR_CHARS:]
            log.warning("%s failed:\n%s", a.attempt_id, error)
        finally:
            del self._running[a.attempt_id]
        log.info(
            "%s %s in %.2fs",
            a.attempt_id,
            "done" if success else "failed",
            time.monotonic() - started,
        )
        self._report(a.attempt_id, success, error, counters)

    def _run(self, a: mr_pb2.TaskAssignment, cancel: threading.Event) -> Counters:
        job = self._load_job(a)
        if a.kind == mr_pb2.MAP:
            segments = [Segment(s.path, s.start, s.end) for s in a.segments]
            out_dir = storage.resolve(self.root, a.output_dir)
            return run_map_task(
                job,
                self.root,
                segments,
                a.num_reduce_tasks,
                out_dir,
                use_combiner=a.use_combiner,
                cancel=cancel,
            )
        inputs = [storage.resolve(self.root, d) for d in a.map_output_dirs]
        out_file = storage.resolve(self.root, a.output_file)
        return run_reduce_task(job, inputs, a.task_index, out_file, cancel=cancel)

    def _load_job(self, a: mr_pb2.TaskAssignment) -> jobcode.JobFunctions:
        if a.job_id not in self._jobs:
            if len(self._jobs) > 16:
                self._jobs.clear()
            self._jobs[a.job_id] = jobcode.load_job(a.job_source, a.job_filename)
        return self._jobs[a.job_id]

    def _report(self, attempt_id: str, success: bool, error: str, counters: Counters) -> None:
        report = mr_pb2.TaskReport(
            worker_id=self.id,
            attempt_id=attempt_id,
            success=success,
            error=error,
            counters=counters,
        )
        for delay in (0.5, 1, 2, 4, 8):
            try:
                self.stub.ReportTask(report, timeout=RPC_TIMEOUT)
                return
            except grpc.RpcError as e:
                log.warning("report for %s failed (%s); retrying", attempt_id, e.code())
                time.sleep(delay)
        log.error("giving up on reporting %s; the boss will reschedule it", attempt_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MapReduce worker")
    parser.add_argument("--boss", default=os.environ.get("MR_BOSS", "localhost:5440"))
    parser.add_argument("--data", default=os.environ.get("MR_DATA", "/data"), help="storage root")
    parser.add_argument("--id", default=os.environ.get("MR_WORKER_ID") or socket.gethostname())
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=float(os.environ.get("MR_HEARTBEAT_INTERVAL", DEFAULT_HEARTBEAT_INTERVAL)),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    Worker(args.id, args.boss, Path(args.data), args.heartbeat_interval).run()


if __name__ == "__main__":
    main()
