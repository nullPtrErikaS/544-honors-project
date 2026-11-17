"""End-to-end tests: real boss and worker processes talking gRPC over localhost."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

import grpc
import pytest

from mapreduce import jobcode, mr_pb2
from mapreduce.local import run_local

from .cluster import REPO, LocalCluster
from .conftest import JOBS, write_corpus


def make_corpus(path, num_files: int = 4, lines: int = 400, seed: int = 544) -> None:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(300)] + ["the", "fox", "dog", "MapReduce"]
    write_corpus(
        path,
        "",
        {
            f"part{i}.txt": "\n".join(
                " ".join(rng.choices(vocab, k=rng.randint(0, 12))) for _ in range(lines)
            )
            + "\n"
            for i in range(num_files)
        },
    )


@pytest.fixture
def cluster(tmp_path):
    c = LocalCluster(tmp_path, num_workers=3)
    make_corpus(tmp_path / "src")
    c.client.upload(tmp_path / "src", "inputs/corpus")
    yield c
    c.close()
    print(c.dump_logs())  # shown by pytest only if the test failed


def expected_output(cluster: LocalCluster, job_file: str, input_path: str) -> list[str]:
    job = jobcode.load_job((JOBS / job_file).read_text(), job_file)
    return run_local(job, cluster.root, input_path)


@pytest.mark.parametrize("job_file", ["wordcount.py", "inverted_index.py"])
def test_job_matches_sequential_reference(cluster, job_file):
    status = cluster.run_job(job_file, "inputs/corpus", "outputs/result", maps=7, reduces=3)
    assert status.state == mr_pb2.JOB_SUCCEEDED, status.error
    assert sorted(cluster.output_lines("outputs/result")) == expected_output(
        cluster, job_file, "inputs/corpus"
    )

    names = [f.path.rsplit("/", 1)[-1] for f in cluster.client.ls("outputs/result")]
    assert names == ["_SUCCESS"] + [f"part-r-{r:05d}" for r in range(3)]
    assert status.maps_done == 7 and status.reduces_done == 3
    assert status.counters["map_input_records"] == 1600
    # All three workers should have done some work.
    assert len({a.worker_id for a in status.attempts}) == 3


def test_intermediate_data_is_cleaned_up(cluster):
    status = cluster.run_job("wordcount.py", "inputs/corpus", "outputs/wc", maps=4, reduces=2)
    assert status.state == mr_pb2.JOB_SUCCEEDED
    jobs_dir = cluster.root / "_jobs" / status.job_id
    deadline = time.monotonic() + 5
    while jobs_dir.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not jobs_dir.exists()
    assert not (cluster.root / "outputs" / "wc" / "_temporary").exists()


def test_more_maps_than_lines(cluster, tmp_path):
    write_corpus(tmp_path / "tiny", "", {"t.txt": "a b\nb\n"})
    cluster.client.upload(tmp_path / "tiny", "inputs/tiny")
    status = cluster.run_job("wordcount.py", "inputs/tiny", "outputs/tiny", maps=10, reduces=4)
    assert status.state == mr_pb2.JOB_SUCCEEDED
    assert sorted(cluster.output_lines("outputs/tiny")) == ["a\t1\n", "b\t2\n"]


def test_user_exception_fails_job_with_traceback(cluster, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def mapper(k, v):\n    return {}['missing']\n\ndef reducer(k, vs):\n    yield k, 1\n"
    )
    status = cluster.run_job(bad, "inputs/corpus", "outputs/bad", maps=2, reduces=1)
    assert status.state == mr_pb2.JOB_FAILED
    assert "KeyError" in status.error and "bad.py" in status.error
    failed = [a for a in status.attempts if a.state == mr_pb2.ATTEMPT_FAILED]
    assert len(failed) >= 4  # retried before giving up


def test_invalid_jobs_are_rejected_at_submit(cluster, tmp_path):
    no_reducer = tmp_path / "no_reducer.py"
    no_reducer.write_text("def mapper(k, v):\n    yield v, 1\n")
    with pytest.raises(jobcode.JobCodeError, match="reducer"):
        cluster.run_job(no_reducer, "inputs/corpus", "outputs/x", maps=1, reduces=1)

    with pytest.raises(grpc.RpcError) as err:
        cluster.run_job("wordcount.py", "inputs/nope", "outputs/x", maps=1, reduces=1)
    assert err.value.code() == grpc.StatusCode.NOT_FOUND

    with pytest.raises(grpc.RpcError) as err:
        cluster.run_job("wordcount.py", "inputs/corpus", "inputs/corpus/out", maps=1, reduces=1)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_existing_output_requires_overwrite(cluster):
    assert cluster.run_job("wordcount.py", "inputs/corpus", "outputs/o", 2, 1).state == (
        mr_pb2.JOB_SUCCEEDED
    )
    with pytest.raises(grpc.RpcError) as err:
        cluster.run_job("wordcount.py", "inputs/corpus", "outputs/o", 2, 1)
    assert err.value.code() == grpc.StatusCode.ALREADY_EXISTS
    status = cluster.run_job("wordcount.py", "inputs/corpus", "outputs/o", 2, 2, overwrite=True)
    assert status.state == mr_pb2.JOB_SUCCEEDED
    # The boss deletes _temporary in the background after the job ends; don't race it.
    temp = cluster.root / "outputs" / "o" / "_temporary"
    deadline = time.monotonic() + 5
    while temp.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    names = [f.path.rsplit("/", 1)[-1] for f in cluster.client.ls("outputs/o")]
    assert names == ["_SUCCESS", "part-r-00000", "part-r-00001"]


def test_storage_round_trip(cluster, tmp_path):
    files = {f.path: f.size for f in cluster.client.ls("inputs/corpus")}
    assert sorted(files) == [f"inputs/corpus/part{i}.txt" for i in range(4)]
    assert cluster.client.download_to("inputs/corpus", tmp_path / "dl") == 4
    assert (tmp_path / "dl" / "part0.txt").read_bytes() == (tmp_path / "src/part0.txt").read_bytes()

    cluster.client.rm("inputs/corpus/part0.txt")
    assert len(cluster.client.ls("inputs/corpus")) == 3
    with pytest.raises(grpc.RpcError) as err:
        cluster.client.ls("inputs/../../etc")
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_cli_end_to_end(cluster, tmp_path):
    """Drive the real `mr` command line, as a user would."""

    def mr(*args: str) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "mapreduce.client", *args],
            cwd=REPO,
            env={**os.environ, "MR_BOSS": cluster.address},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert "uploaded 1 file(s)" in mr("upload", str(tmp_path / "src" / "part1.txt"), "cli/in.txt")
    out = mr(
        "submit",
        "--input=cli/in.txt",
        "--output=cli/out",
        "--job=examples/jobs/wordcount.py",
        "--maps=3",
        "--reduces=2",
    )
    assert "SUCCEEDED" in out
    job_id = out.split()[1]

    cat = mr("cat", "cli/out")
    local = mr("local", "--job=examples/jobs/wordcount.py", f"--input={tmp_path / 'src/part1.txt'}")
    assert sorted(cat.splitlines()) == local.splitlines()

    status = json.loads(mr("status", job_id, "--json"))
    assert status["state"] == "JOB_SUCCEEDED"
    assert len(status["attempts"]) == 5
    assert job_id in mr("jobs")
    assert "w1" in mr("workers")
    assert "part-r-00001" in mr("ls", "cli/out")
