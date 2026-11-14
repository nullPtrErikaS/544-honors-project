"""``mr``: command-line client for the MapReduce cluster.

Runs on the host, outside the containers, and talks to the boss over gRPC
(``MR_BOSS``, default ``localhost:5440``). Run ``./mr --help`` for the commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

import grpc
from google.protobuf import json_format

from mapreduce import jobcode, mr_pb2, mr_pb2_grpc
from mapreduce.local import run_local

CHUNK_SIZE = 1 << 20
DEFAULT_BOSS = "localhost:5440"


class Client:
    """Programmatic client (used by the CLI, the tests, and the benchmarks)."""

    def __init__(self, address: str | None = None):
        self.address = address or os.environ.get("MR_BOSS", DEFAULT_BOSS)
        self.channel = grpc.insecure_channel(self.address)
        self.stub = mr_pb2_grpc.BossStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    # ------------------------------------------------------------------ storage

    def upload(self, local: Path, remote: str) -> mr_pb2.UploadReply:
        """Upload a local file or directory (recursively) to storage path ``remote``."""
        if local.is_file():
            dest = remote + local.name if remote.endswith("/") else remote
            targets = [(local, dest)]
        elif local.is_dir():
            files = sorted(p for p in local.rglob("*") if p.is_file())
            targets = [
                (p, f"{remote.rstrip('/')}/{p.relative_to(local).as_posix()}") for p in files
            ]
        else:
            raise FileNotFoundError(f"no such file or directory: {local}")

        def chunks() -> Iterator[mr_pb2.FileChunk]:
            for path, dest in targets:
                with open(path, "rb") as f:
                    yield mr_pb2.FileChunk(path=dest, data=f.read(CHUNK_SIZE))
                    while data := f.read(CHUNK_SIZE):
                        yield mr_pb2.FileChunk(path=dest, data=data)

        return self.stub.Upload(chunks())

    def download(self, remote: str, include_hidden: bool = False) -> Iterator[mr_pb2.FileChunk]:
        return self.stub.Download(
            mr_pb2.DownloadRequest(path=remote, include_hidden=include_hidden)
        )

    def download_to(self, remote: str, local: Path) -> int:
        """Download a file or directory; returns the number of files written."""
        files, current, f = 0, None, None
        try:
            for chunk in self.download(remote):
                if f is None or chunk.path != current:
                    if f is not None:
                        f.close()
                    current = chunk.path
                    dest = local / chunk.path if chunk.path else local
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    f = open(dest, "wb")  # noqa: SIM115 -- closed in finally / on next file
                    files += 1
                f.write(chunk.data)
        finally:
            if f is not None:
                f.close()
        return files

    def cat(self, remote: str, out: BinaryIO) -> None:
        for chunk in self.download(remote):
            out.write(chunk.data)

    def read_text(self, remote: str) -> str:
        return b"".join(c.data for c in self.download(remote)).decode("utf-8")

    def ls(self, remote: str) -> list[mr_pb2.FileInfo]:
        return list(self.stub.ListFiles(mr_pb2.ListFilesRequest(path=remote)).files)

    def rm(self, remote: str) -> None:
        self.stub.DeletePath(mr_pb2.DeletePathRequest(path=remote))

    # ------------------------------------------------------------------ jobs

    def submit(
        self,
        input_path: str,
        output_path: str,
        job_file: Path,
        maps: int,
        reduces: int,
        *,
        overwrite: bool = False,
        speculation: bool = True,
    ) -> str:
        source = job_file.read_text(encoding="utf-8")
        jobcode.top_level_names(source, job_file.name)  # fail fast on obvious mistakes
        request = mr_pb2.SubmitJobRequest(
            input_path=input_path,
            output_path=output_path,
            job_source=source,
            job_filename=job_file.name,
            num_map_tasks=maps,
            num_reduce_tasks=reduces,
            overwrite_output=overwrite,
            speculative_execution=speculation,
        )
        return self.stub.SubmitJob(request).job_id

    def status(self, job_id: str) -> mr_pb2.JobStatus:
        return self.stub.GetJob(mr_pb2.GetJobRequest(job_id=job_id))

    def wait(
        self,
        job_id: str,
        poll: float = 0.25,
        on_update: Callable[[mr_pb2.JobStatus], None] | None = None,
        timeout: float | None = None,
    ) -> mr_pb2.JobStatus:
        """Poll until the job finishes; return its final status."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.status(job_id)
            if on_update:
                on_update(status)
            if status.state in (mr_pb2.JOB_SUCCEEDED, mr_pb2.JOB_FAILED):
                return status
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"{job_id} still {state_name(status)} after {timeout}s")
            time.sleep(poll)

    def jobs(self) -> list[mr_pb2.JobStatus]:
        return list(self.stub.ListJobs(mr_pb2.ListJobsRequest()).jobs)

    def workers(self) -> list[mr_pb2.WorkerInfo]:
        return list(self.stub.ListWorkers(mr_pb2.ListWorkersRequest()).workers)


# ---------------------------------------------------------------------- formatting


def state_name(status: mr_pb2.JobStatus) -> str:
    return mr_pb2.JobState.Name(status.state).removeprefix("JOB_")


def status_to_dict(status: mr_pb2.JobStatus) -> dict:
    d = json_format.MessageToDict(
        status, preserving_proto_field_name=True, always_print_fields_with_no_presence=True
    )
    d["counters"] = {k: int(v) for k, v in sorted(d.get("counters", {}).items())}
    return d


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError


def progress_line(s: mr_pb2.JobStatus) -> str:
    return (
        f"{s.job_id}  {state_name(s):<9}  maps {s.maps_done}/{s.num_map_tasks}"
        f"  reduces {s.reduces_done}/{s.num_reduce_tasks}  {s.elapsed:6.1f}s"
    )


def print_status(s: mr_pb2.JobStatus, attempts: bool = True) -> None:
    print(f"{s.job_id}  {s.job_name}  {state_name(s)}  ({s.elapsed:.2f}s)")
    print(f"  input:    {s.input_path}")
    print(f"  output:   {s.output_path}")
    print(f"  maps:     {s.maps_done}/{s.num_map_tasks}")
    print(f"  reduces:  {s.reduces_done}/{s.num_reduce_tasks}")
    print(f"  options:  speculation {'on' if s.speculative_execution else 'off'}")
    backups = [a for a in s.attempts if a.speculative]
    if backups:
        won = sum(a.state == mr_pb2.ATTEMPT_SUCCEEDED for a in backups)
        print(f"  backups:  {len(backups)} launched, {won} finished first")
    lost = sum(a.state == mr_pb2.ATTEMPT_LOST for a in s.attempts)
    if lost:
        print(f"  lost:     {lost} attempt(s) on dead workers were re-executed")
    if s.error:
        print("  error:\n    " + s.error.replace("\n", "\n    "))
    if s.counters:
        print("  counters:")
        for name, value in sorted(s.counters.items()):
            print(f"    {name:<26}{value:>14,}")
    if attempts and s.attempts:
        print("  attempts:")
        print(f"    {'ATTEMPT':<24}{'WORKER':<12}{'STATE':<11}{'START':>8}{'END':>8}  NOTE")
        for a in s.attempts:
            state = mr_pb2.AttemptState.Name(a.state).removeprefix("ATTEMPT_")
            end = f"{a.end:8.2f}" if a.end else f"{'-':>8}"
            note = "backup" if a.speculative else ""
            if a.error:
                note += (" " if note else "") + a.error.strip().splitlines()[-1][:60]
            print(f"    {a.attempt_id:<24}{a.worker_id:<12}{state:<11}{a.start:8.2f}{end}  {note}")


# ---------------------------------------------------------------------- commands


def cmd_upload(client: Client, args) -> None:
    reply = client.upload(Path(args.local), args.remote)
    print(f"uploaded {reply.files} file(s), {human_bytes(reply.bytes)} -> {args.remote}")


def cmd_download(client: Client, args) -> None:
    n = client.download_to(args.remote, Path(args.local))
    print(f"downloaded {n} file(s) -> {args.local}")


def cmd_ls(client: Client, args) -> None:
    for info in client.ls(args.path):
        name = info.path + ("/" if info.is_dir else "")
        size = "-" if info.is_dir else human_bytes(info.size)
        print(f"{size:>10}  {name}")


def cmd_cat(client: Client, args) -> None:
    client.cat(args.path, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def cmd_rm(client: Client, args) -> None:
    client.rm(args.path)
    print(f"removed {args.path}")


def cmd_submit(client: Client, args) -> int:
    job_id = client.submit(
        args.input,
        args.output,
        Path(args.job),
        args.maps,
        args.reduces,
        overwrite=args.overwrite,
        speculation=not args.no_speculation,
    )
    if args.detach:
        print(job_id)
        return 0
    print(f"submitted {job_id}")
    tty = sys.stdout.isatty()
    last = ""

    def show(status: mr_pb2.JobStatus) -> None:
        nonlocal last
        line = progress_line(status)
        if tty:
            print("\r" + line, end="", flush=True)
        elif line.split()[:5] != last.split()[:5]:  # print on progress, not every tick
            print(line, flush=True)
        last = line

    status = client.wait(job_id, on_update=show)
    if tty:
        print()
    print_status(status, attempts=False)
    return 0 if status.state == mr_pb2.JOB_SUCCEEDED else 1


def cmd_status(client: Client, args) -> None:
    status = client.status(args.job_id)
    if args.json:
        print(json.dumps(status_to_dict(status), indent=2))
    else:
        print_status(status)


def cmd_jobs(client: Client, args) -> None:
    jobs = client.jobs()
    if not jobs:
        print("no jobs yet")
    for s in jobs:
        print(f"{progress_line(s)}  {s.job_name} -> {s.output_path}")


def cmd_workers(client: Client, args) -> None:
    workers = client.workers()
    if not workers:
        print("no workers have contacted the boss yet")
    for w in workers:
        state = "alive" if w.alive else "DEAD"
        running = w.current_attempt or "idle"
        print(
            f"{w.worker_id:<14}{state:<7}seen {w.last_seen_ago:5.1f}s ago  "
            f"{w.attempts_succeeded:4d} tasks done  {running}"
        )


def cmd_wait_ready(client: Client, args) -> int:
    """Block until the boss is up and enough workers have checked in (for scripts and CI)."""
    deadline = time.monotonic() + args.timeout
    alive = 0
    while time.monotonic() < deadline:
        try:
            alive = sum(w.alive for w in client.workers())
            if alive >= args.workers:
                print(f"cluster ready: {alive} worker(s)")
                return 0
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.UNAVAILABLE:
                raise
        time.sleep(0.5)
    print(f"error: only {alive} of {args.workers} workers after {args.timeout}s", file=sys.stderr)
    return 1


def cmd_local(args) -> None:
    source = Path(args.job).read_text(encoding="utf-8")
    job = jobcode.load_job(source, args.job)
    path = Path(args.input).resolve()
    cwd = Path.cwd().resolve()
    root = cwd if path.is_relative_to(cwd) else path.parent
    lines = run_local(job, root, path.relative_to(root).as_posix())
    sys.stdout.buffer.write("".join(lines).encode("utf-8"))  # same bytes as `mr cat`
    sys.stdout.buffer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mr",
        description="Client for the MapReduce cluster. "
        f"Connects to $MR_BOSS (default {DEFAULT_BOSS}).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("upload", help="copy a local file or directory into shared storage")
    p.add_argument("local")
    p.add_argument("remote")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("download", help="copy a file or directory out of shared storage")
    p.add_argument("remote")
    p.add_argument("local")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("ls", help="list a directory in shared storage")
    p.add_argument("path", nargs="?", default="")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("cat", help="print a file, or all part files of a job output directory")
    p.add_argument("path")
    p.set_defaults(func=cmd_cat)

    p = sub.add_parser("rm", help="delete a file or directory from shared storage")
    p.add_argument("path")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("submit", help="run a MapReduce job")
    p.add_argument("--input", required=True, help="input file or directory in shared storage")
    p.add_argument("--output", required=True, help="output directory in shared storage")
    p.add_argument("--job", required=True, help="local Python file defining mapper/reducer")
    p.add_argument("--maps", type=int, required=True, help="number of map tasks")
    p.add_argument("--reduces", type=int, required=True, help="number of reduce tasks")
    p.add_argument("--overwrite", action="store_true", help="replace the output if it exists")
    p.add_argument(
        "--no-speculation", action="store_true", help="never launch backup tasks for stragglers"
    )
    p.add_argument("--detach", action="store_true", help="print the job id and exit")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="show a job's progress, counters, and task attempts")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("jobs", help="list all jobs")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("workers", help="list workers and what they are running")
    p.set_defaults(func=cmd_workers)

    p = sub.add_parser("wait-ready", help="wait until the boss and N workers are up")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60)
    p.set_defaults(func=cmd_wait_ready)

    p = sub.add_parser("local", help="run a job sequentially on local files (for testing)")
    p.add_argument("--job", required=True)
    p.add_argument("--input", required=True, help="local file or directory")
    p.set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "local":
        cmd_local(args)
        return 0
    client = Client()
    try:
        return args.func(client, args) or 0
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            print(
                f"error: cannot reach the boss at {client.address}. "
                "Is the cluster running? (docker compose up -d)",
                file=sys.stderr,
            )
        else:
            print(f"error: {e.details()}", file=sys.stderr)
        return 1
    except (jobcode.JobCodeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
