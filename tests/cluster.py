"""Run a real boss and real workers as local processes, for integration tests.

Same code as the Docker deployment, but with a temp directory as shared storage and
localhost networking, so the tests run anywhere (including CI) in a few seconds.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import grpc

from mapreduce import mr_pb2
from mapreduce.client import Client

REPO = Path(__file__).resolve().parent.parent

# Fast failure detection so tests don't wait long (the defaults are 1 s / 5 s).
FAST_TIMEOUTS = {"MR_HEARTBEAT_INTERVAL": "0.2", "MR_WORKER_TIMEOUT": "1.5"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalCluster:
    def __init__(self, tmp_path: Path, num_workers: int = 3, boss_env: dict | None = None):
        self.root = tmp_path / "data"
        self.root.mkdir()
        self.log_dir = tmp_path / "logs"
        self.log_dir.mkdir()
        self.port = free_port()
        self.address = f"localhost:{self.port}"
        self.procs: dict[str, subprocess.Popen] = {}
        self._logs: list = []

        self._spawn(
            "boss",
            ["mapreduce.boss", "--port", str(self.port), "--data", str(self.root)],
            boss_env,
        )
        for i in range(num_workers):
            self.start_worker(f"w{i + 1}")
        self.client = Client(self.address)
        grpc.channel_ready_future(self.client.channel).result(timeout=20)
        self.wait_for_workers(num_workers)

    def wait_for_workers(self, n: int, timeout: float = 20) -> None:
        deadline = time.monotonic() + timeout
        while len([w for w in self.client.workers() if w.alive]) < n:
            if time.monotonic() > deadline:
                raise TimeoutError(f"only {len(self.client.workers())} of {n} workers joined")
            time.sleep(0.05)

    def start_worker(self, worker_id: str, env: dict | None = None) -> None:
        args = ["mapreduce.worker", "--boss", self.address, "--data", str(self.root)]
        self._spawn(worker_id, [*args, "--id", worker_id], env)

    def kill(self, name: str) -> None:
        """Hard-kill a process (SIGKILL on Linux), like a crashed container."""
        proc = self.procs.pop(name)
        proc.kill()
        proc.wait()

    def _spawn(self, name: str, args: list[str], env: dict | None) -> None:
        log = open(self.log_dir / f"{name}.log", "w")  # noqa: SIM115 -- closed in close()
        self._logs.append(log)
        self.procs[name] = subprocess.Popen(
            [sys.executable, "-m", *args],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **FAST_TIMEOUTS, **(env or {})},
        )

    def run_job(
        self,
        job: str | Path,
        input_path: str,
        output_path: str,
        maps: int,
        reduces: int,
        timeout: float = 60,
        **kwargs,
    ) -> mr_pb2.JobStatus:
        job_file = Path(job) if Path(job).is_absolute() else REPO / "examples" / "jobs" / job
        job_id = self.client.submit(input_path, output_path, job_file, maps, reduces, **kwargs)
        return self.client.wait(job_id, poll=0.1, timeout=timeout)

    def output_lines(self, output_path: str) -> list[str]:
        return self.client.read_text(output_path).splitlines(keepends=True)

    def dump_logs(self) -> str:
        for log in self._logs:
            if not log.closed:
                log.flush()
        return "\n".join(
            f"===== {p.name} =====\n{p.read_text(errors='replace')}"
            for p in sorted(self.log_dir.iterdir())
        )

    def close(self) -> None:
        self.client.close()
        for proc in self.procs.values():
            proc.kill()
        for proc in self.procs.values():
            proc.wait()
        for log in self._logs:
            log.close()
