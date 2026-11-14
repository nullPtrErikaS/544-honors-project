"""Performance experiments against the real Docker Compose deployment.

Prerequisites: the cluster is running (``docker compose up -d``) and the client
dependencies are installed. Each experiment appends one JSON line per run to
``bench/results/<experiment>.jsonl``; ``bench/plot.py`` turns those into plots.

    python bench/run.py all              # everything (about 40 minutes)
    python bench/run.py scaling --trials 3
    python bench/run.py combiner | speculation | failure

Every experiment runs word count over a synthetic 128 MiB Zipf corpus (generated and
uploaded on first use) and records the boss's own measurements: job time, counters, and
the start/end/worker/outcome of every task attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mapreduce import mr_pb2  # noqa: E402
from mapreduce.client import Client, status_to_dict  # noqa: E402

RESULTS = REPO / "bench" / "results"
WORKERS = ["worker1", "worker2", "worker3", "worker4"]
JOB = REPO / "examples" / "jobs" / "wordcount.py"
CORPUS_MB = 128
INPUT = f"inputs/bench{CORPUS_MB}"
OUTPUT = "outputs/bench"


# ---------------------------------------------------------------------- cluster control


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, check=True, capture_output=True, text=True).stdout


def compose(*args: str) -> str:
    return sh("docker", "compose", *args)


def container(service: str) -> str:
    return compose("ps", "-q", service).strip()


def set_cpus(service: str, cpus: float) -> None:
    """Throttle one worker to make it a straggler (like a noisy neighbor on its machine)."""
    sh("docker", "update", "--cpus", str(cpus), container(service))


def use_workers(client: Client, n: int) -> None:
    """Run exactly the first ``n`` workers, at full speed."""
    compose("up", "-d", "boss", *WORKERS[:n])
    if n < len(WORKERS):
        compose("stop", *WORKERS[n:])
    for w in WORKERS[:n]:
        set_cpus(w, 1.0)
    deadline = time.monotonic() + 60
    while True:  # wait until every wanted worker has checked in recently
        fresh = {w.worker_id for w in client.workers() if w.alive and w.last_seen_ago < 1.5}
        if set(WORKERS[:n]) <= fresh:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"workers not ready: {fresh}")
        time.sleep(0.25)


def ensure_input(client: Client) -> None:
    try:
        if client.ls(INPUT):
            return
    except Exception:  # noqa: BLE001 -- NOT_FOUND: fall through and create it
        pass
    local = REPO / "data" / f"bench{CORPUS_MB}"
    if not local.exists():
        print(f"generating {CORPUS_MB} MiB corpus in {local} ...", flush=True)
        sh(
            sys.executable,
            "scripts/gen_corpus.py",
            "--size-mb",
            str(CORPUS_MB),
            "--files",
            "16",
            str(local),
        )
    print(f"uploading {local} -> {INPUT} ...", flush=True)
    client.upload(local, INPUT)


# ---------------------------------------------------------------------- running jobs


def run_job(client: Client, maps: int, reduces: int, **options) -> dict:
    job_id = client.submit(INPUT, OUTPUT, JOB, maps, reduces, overwrite=True, **options)
    status = client.wait(job_id, poll=0.2, timeout=1800)
    if status.state != mr_pb2.JOB_SUCCEEDED:
        raise RuntimeError(f"{job_id} failed: {status.error}")
    result = status_to_dict(status)
    result["output_sha256"] = hashlib.sha256(
        "".join(sorted(client.read_text(OUTPUT).splitlines(keepends=True))).encode()
    ).hexdigest()
    return result


def record(experiment: str, config: dict, trial: int, result: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = {"experiment": experiment, "config": config, "trial": trial, **result}
    with open(RESULTS / f"{experiment}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    extra = {k: v for k, v in config.items()}
    print(f"  {experiment} {extra} trial {trial}: {result['elapsed']:.2f}s", flush=True)


def fresh_results(experiment: str) -> None:
    (RESULTS / f"{experiment}.jsonl").unlink(missing_ok=True)


# ---------------------------------------------------------------------- experiments


def scaling(client: Client, trials: int) -> None:
    """Job time vs number of workers (each worker = 1 CPU)."""
    fresh_results("scaling")
    for n in (1, 2, 3, 4):
        use_workers(client, n)
        run_job(client, 16, 4)  # warm-up (page cache, imports), not recorded
        for t in range(trials):
            record("scaling", {"workers": n}, t, run_job(client, 16, 4))
    use_workers(client, 4)


def combiner(client: Client, trials: int) -> None:
    """Word count with and without the map-side combiner."""
    fresh_results("combiner")
    use_workers(client, 4)
    run_job(client, 16, 4)
    for t in range(trials):
        for enabled in (True, False):
            record("combiner", {"combiner": enabled}, t, run_job(client, 16, 4, combiner=enabled))


def speculation(client: Client, trials: int) -> None:
    """One straggler worker (0.25 CPU) with and without backup tasks."""
    fresh_results("speculation")
    use_workers(client, 4)
    run_job(client, 8, 4)
    configs = [
        ("no straggler", 1.0, True),
        ("straggler, no speculation", 0.25, False),
        ("straggler + speculation", 0.25, True),
    ]
    for t in range(trials):
        for name, cpus, spec in configs:
            set_cpus("worker4", cpus)
            cfg = {"setup": name, "worker4_cpus": cpus, "speculation": spec}
            record("speculation", cfg, t, run_job(client, 8, 4, speculation=spec))
    set_cpus("worker4", 1.0)


def failure(client: Client, trials: int, kill_after: float = 4.0) -> None:
    """Kill a worker container partway through the map phase."""
    fresh_results("failure")
    use_workers(client, 4)
    run_job(client, 16, 4)
    for t in range(trials):
        use_workers(client, 4)  # bring back the worker killed in the previous trial
        record("failure", {"setup": "no failure"}, t, run_job(client, 16, 4))

        use_workers(client, 4)
        timer = threading.Timer(kill_after, compose, args=("kill", "worker2"))
        timer.start()
        result = run_job(client, 16, 4)
        timer.join()
        result["killed_at"] = kill_after
        record("failure", {"setup": "worker2 killed"}, t, result)
    use_workers(client, 4)


EXPERIMENTS = {
    "scaling": scaling,
    "combiner": combiner,
    "speculation": speculation,
    "failure": failure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    client = Client()
    ensure_input(client)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"== {name}", flush=True)
        EXPERIMENTS[name](client, args.trials)
    client.close()


if __name__ == "__main__":
    main()
