"""The boss process: gRPC front end for the scheduler and for shared storage.

It serves two gRPC services on one port:

* ``Boss``        -- for the client: submit jobs, check status, upload/download files
* ``Coordinator`` -- for workers: request tasks, report results

Usage: ``python -m mapreduce.boss [--port 5440] [--data /data]``
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import threading
from collections.abc import Iterator
from concurrent import futures
from pathlib import Path

import grpc

from mapreduce import jobcode, jobfiles, mr_pb2, mr_pb2_grpc, storage
from mapreduce.scheduler import Assignment, Attempt, Job, JobSpec, Scheduler
from mapreduce.splits import compute_splits

log = logging.getLogger("boss")

DEFAULT_PORT = 5440
CHUNK_SIZE = 1 << 20  # 1 MiB per streamed file chunk
LONG_POLL_SECONDS = 1.0


# ---------------------------------------------------------------------- conversions


def attempt_to_proto(attempt: Attempt) -> mr_pb2.AttemptInfo:
    t0 = attempt.task.job.submitted
    return mr_pb2.AttemptInfo(
        attempt_id=attempt.id,
        kind=mr_pb2.TaskKind.Value(attempt.task.kind.value),
        task_index=attempt.task.index,
        worker_id=attempt.worker_id,
        state=mr_pb2.AttemptState.Value("ATTEMPT_" + attempt.state.value),
        start=attempt.start - t0,
        end=attempt.end - t0 if attempt.end is not None else 0.0,
        speculative=attempt.speculative,
        error=attempt.error,
    )


def job_to_proto(job: Job, now: float, with_attempts: bool) -> mr_pb2.JobStatus:
    end = job.finished if job.finished is not None else now
    return mr_pb2.JobStatus(
        job_id=job.id,
        job_name=job.spec.filename,
        state=mr_pb2.JobState.Value("JOB_" + job.state.value),
        error=job.error,
        input_path=job.spec.input_path,
        output_path=job.spec.output_path,
        num_map_tasks=len(job.maps),
        num_reduce_tasks=len(job.reduces),
        maps_done=sum(t.done for t in job.maps),
        reduces_done=sum(t.done for t in job.reduces),
        submitted_at=job.submitted_wall,
        elapsed=end - job.submitted,
        counters=dict(job.counters),
        attempts=[attempt_to_proto(a) for a in job.all_attempts()] if with_attempts else [],
    )


def assignment_to_proto(a: Assignment) -> mr_pb2.TaskAssignment:
    return mr_pb2.TaskAssignment(
        has_task=True,
        job_id=a.job_id,
        attempt_id=a.attempt_id,
        kind=mr_pb2.TaskKind.Value(a.kind.value),
        task_index=a.task_index,
        job_source=a.source,
        job_filename=a.filename,
        num_reduce_tasks=a.num_reduces,
        segments=[mr_pb2.InputSegment(path=s.path, start=s.start, end=s.end) for s in a.segments],
        map_output_dirs=a.map_output_dirs,
        output_dir=a.output_dir,
        output_file=a.output_file,
    )


def overlaps(a: str, b: str) -> bool:
    """Whether storage paths a and b are the same or one contains the other."""
    return not a or not b or a == b or a.startswith(b + "/") or b.startswith(a + "/")


# ---------------------------------------------------------------------- client API


class BossService(mr_pb2_grpc.BossServicer):
    def __init__(self, scheduler: Scheduler, root: Path):
        self.scheduler = scheduler
        self.root = root
        self._submit_lock = threading.Lock()

    # --- jobs

    def SubmitJob(self, request, context):
        with self._submit_lock:  # validation + submission must be atomic (output dir checks)
            spec, splits = self._validate(request, context)
            job = self.scheduler.submit(spec, splits)
        return mr_pb2.SubmitJobReply(job_id=job.id)

    def _validate(self, req, context):
        filename = req.job_filename or "job.py"
        try:
            jobcode.top_level_names(req.job_source, filename)
            input_path = storage.normalize(req.input_path)
            output_path = storage.normalize(req.output_path)
        except (jobcode.JobCodeError, storage.StorageError) as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        if req.num_map_tasks < 1 or req.num_reduce_tasks < 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "need at least 1 map and 1 reduce task")
        if not output_path or storage.is_hidden(output_path.split("/")[0]):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid output path {output_path!r}")
        if overlaps(input_path, output_path):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "input and output paths overlap")
        with self.scheduler.lock:
            busy = [
                j.id
                for j in self.scheduler.active_jobs()
                if overlaps(j.spec.output_path, output_path)
            ]
        if busy:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"{busy[0]} is writing to {output_path}"
            )

        try:
            files = storage.list_input_files(self.root, input_path)
        except storage.StorageError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        if not files:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"no input files under {input_path}")

        out = storage.resolve(self.root, output_path)
        if out.exists():
            if not req.overwrite_output and (out.is_file() or any(out.iterdir())):
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"output path {output_path} already exists (use --overwrite to replace it)",
                )
            _remove(out)
        out.mkdir(parents=True)

        spec = JobSpec(
            input_path=input_path,
            output_path=output_path,
            source=req.job_source,
            filename=filename,
            num_maps=req.num_map_tasks,
            num_reduces=req.num_reduce_tasks,
        )
        return spec, compute_splits(files, req.num_map_tasks)

    def GetJob(self, request, context):
        with self.scheduler.lock:
            job = self.scheduler.jobs.get(request.job_id)
            if job is None:
                context.abort(grpc.StatusCode.NOT_FOUND, f"no such job: {request.job_id}")
            return job_to_proto(job, self.scheduler.clock(), with_attempts=True)

    def ListJobs(self, request, context):
        with self.scheduler.lock:
            now = self.scheduler.clock()
            jobs = [job_to_proto(j, now, with_attempts=False) for j in self.scheduler.jobs.values()]
        return mr_pb2.ListJobsReply(jobs=jobs)

    def ListWorkers(self, request, context):
        with self.scheduler.lock:
            now = self.scheduler.clock()
            workers = [
                mr_pb2.WorkerInfo(
                    worker_id=w.id,
                    alive=True,
                    last_seen_ago=now - w.last_seen,
                    current_attempt=w.current.id if w.current else "",
                    attempts_succeeded=w.succeeded,
                )
                for w in sorted(self.scheduler.workers.values(), key=lambda w: w.id)
            ]
        return mr_pb2.ListWorkersReply(workers=workers)

    # --- storage

    def _path(self, rel: str, context, *, writable: bool = False) -> Path:
        try:
            norm = storage.normalize(rel)
        except storage.StorageError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        if writable and (not norm or norm.split("/")[0] == jobfiles.JOBS_DIR):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"cannot write to {norm or '/'!r}")
        return storage.resolve(self.root, norm)

    def Upload(self, request_iterator, context):
        files = nbytes = 0
        current, f = None, None
        try:
            for chunk in request_iterator:
                if f is None or chunk.path != current:
                    if f is not None:
                        f.close()
                    current = chunk.path
                    path = self._path(current, context, writable=True)
                    if path.is_dir():
                        context.abort(grpc.StatusCode.ALREADY_EXISTS, f"{current} is a directory")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    f = open(path, "wb")  # noqa: SIM115 -- closed in finally / on next file
                    files += 1
                f.write(chunk.data)
                nbytes += len(chunk.data)
        finally:
            if f is not None:
                f.close()
        log.info("uploaded %d files (%d bytes)", files, nbytes)
        return mr_pb2.UploadReply(files=files, bytes=nbytes)

    def Download(self, request, context) -> Iterator[mr_pb2.FileChunk]:
        base = self._path(request.path, context)
        if base.is_file():
            yield from _stream_file(base, "")
            return
        if not base.is_dir():
            context.abort(grpc.StatusCode.NOT_FOUND, f"no such file or directory: {request.path}")
        for dirpath, dirnames, filenames in os.walk(base):
            hidden = (lambda n: False) if request.include_hidden else storage.is_hidden
            dirnames[:] = sorted(d for d in dirnames if not hidden(d))
            for name in sorted(n for n in filenames if not hidden(n)):
                path = Path(dirpath, name)
                yield from _stream_file(path, path.relative_to(base).as_posix())

    def ListFiles(self, request, context):
        base = self._path(request.path, context)
        if base.is_file():
            entries = [base]
        elif base.is_dir():
            entries = sorted(base.iterdir())
        else:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no such file or directory: {request.path}")
        infos = [
            mr_pb2.FileInfo(
                path=storage.to_storage_path(self.root, p),
                size=p.stat().st_size if p.is_file() else 0,
                is_dir=p.is_dir(),
            )
            for p in entries
        ]
        return mr_pb2.ListFilesReply(files=infos)

    def DeletePath(self, request, context):
        path = self._path(request.path, context, writable=True)
        if not path.exists():
            context.abort(grpc.StatusCode.NOT_FOUND, f"no such file or directory: {request.path}")
        _remove(path)
        return mr_pb2.DeletePathReply()


def _stream_file(path: Path, rel: str) -> Iterator[mr_pb2.FileChunk]:
    with open(path, "rb") as f:
        sent = False
        while data := f.read(CHUNK_SIZE):
            yield mr_pb2.FileChunk(path=rel, data=data)
            sent = True
        if not sent:
            yield mr_pb2.FileChunk(path=rel, data=b"")


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# ---------------------------------------------------------------------- worker API


class CoordinatorService(mr_pb2_grpc.CoordinatorServicer):
    def __init__(self, scheduler: Scheduler, long_poll: float = LONG_POLL_SECONDS):
        self.scheduler = scheduler
        self.long_poll = long_poll

    def RequestTask(self, request, context):
        assignment = self.scheduler.request_task(request.worker_id, self.long_poll)
        if assignment is None:
            return mr_pb2.TaskAssignment(has_task=False)
        return assignment_to_proto(assignment)

    def ReportTask(self, request, context):
        accepted = self.scheduler.report(
            request.worker_id,
            request.attempt_id,
            request.success,
            request.error,
            dict(request.counters),
        )
        return mr_pb2.TaskReportReply(accepted=accepted)


# ---------------------------------------------------------------------- main


def build_server(scheduler: Scheduler, root: Path, address: str) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    mr_pb2_grpc.add_BossServicer_to_server(BossService(scheduler, root), server)
    mr_pb2_grpc.add_CoordinatorServicer_to_server(CoordinatorService(scheduler), server)
    server.add_insecure_port(address)
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MapReduce boss")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MR_PORT", DEFAULT_PORT)))
    parser.add_argument("--data", default=os.environ.get("MR_DATA", "/data"), help="storage root")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    committer = jobfiles.FilesystemCommitter(root)
    committer.reset()
    scheduler = Scheduler(committer)
    server = build_server(scheduler, root, f"[::]:{args.port}")
    server.start()
    log.info("listening on port %d, storage root %s", args.port, root)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
