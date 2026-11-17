import threading
from pathlib import Path

import pytest

from mapreduce import kv, storage
from mapreduce.jobcode import JobFunctions, load_job
from mapreduce.local import run_local
from mapreduce.splits import Segment, compute_splits
from mapreduce.tasks import TaskCancelled, run_map_task, run_reduce_task

from .conftest import load_example, run_in_process, write_corpus


@pytest.mark.parametrize("job_file", ["wordcount.py", "inverted_index.py"])
@pytest.mark.parametrize(("maps", "reduces", "spill"), [(1, 1, 500_000), (3, 2, 4), (8, 5, 1)])
def test_distributed_matches_local(sample_root, job_file, maps, reduces, spill):
    job = load_example(job_file)
    expected = run_local(job, sample_root, "inputs/sample")
    got = run_in_process(job, sample_root, "inputs/sample", maps, reduces, spill)
    assert sorted(got) == expected
    assert len(expected) > 10


def test_wordcount_result(sample_root):
    out = dict(
        line.rstrip("\n").split("\t")
        for line in run_local(load_example("wordcount.py"), sample_root, "inputs/sample")
    )
    assert out["fox"] == "5" and out["the"] == "4" and out["windows"] == "1"


def test_map_output_is_partitioned_and_sorted(sample_root: Path):
    job = load_example("wordcount.py")
    segments = [
        s
        for split in compute_splits(storage.list_input_files(sample_root, "inputs"), 1)
        for s in split
    ]
    out = sample_root / "m"
    counters = run_map_task(job, sample_root, segments, 3, out, spill_threshold=5)

    assert counters["spills"] > 1
    assert sorted(p.name for p in out.iterdir()) == [kv.partition_filename(p) for p in range(3)]
    total = 0
    for p in range(3):
        keys = [k for k, _ in kv.read_run(out / kv.partition_filename(p))]
        assert keys == sorted(keys)
        assert all(kv.partition(k, 3) == p for k in keys)
        total += len(keys)
    assert total == counters["map_output_records"]
    assert counters["intermediate_bytes"] == sum(f.stat().st_size for f in out.iterdir())


def test_empty_split_writes_empty_partitions(tmp_path):
    job = load_example("wordcount.py")
    counters = run_map_task(job, tmp_path, [], 2, tmp_path / "m")
    assert counters["map_input_records"] == 0
    assert all((tmp_path / "m" / kv.partition_filename(p)).stat().st_size == 0 for p in (0, 1))
    counters = run_reduce_task(job, [tmp_path / "m"], 1, tmp_path / "out")
    assert counters["reduce_output_records"] == 0


def test_map_task_can_be_cancelled(tmp_path):
    write_corpus(tmp_path, "in", {"big.txt": "word\n" * 5000})
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        run_map_task(
            load_example("wordcount.py"),
            tmp_path,
            [Segment("in/big.txt", 0, 25000)],
            1,
            tmp_path / "m",
            cancel=cancel,
        )


def test_user_exception_propagates(sample_root):
    job = load_job(
        "def mapper(k, v):\n    raise ValueError('bad record')\ndef reducer(k, vs): pass\n"
    )
    with pytest.raises(ValueError, match="bad record"):
        run_map_task(
            job, sample_root, [Segment("inputs/sample/a.txt", 0, 10)], 1, sample_root / "m"
        )


def test_reducer_cannot_swallow_cancellation(tmp_path):
    job = load_job(
        "def mapper(k, v):\n    yield v, 1\n"
        "def reducer(k, vs):\n"
        "    try:\n        total = sum(vs)\n    except Exception:\n        total = -1\n"
        "    yield k, total\n"
    )
    write_corpus(tmp_path, "in", {"f.txt": "w\n" * 3000})
    run_map_task(job, tmp_path, [Segment("in/f.txt", 0, 6000)], 1, tmp_path / "m")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        run_reduce_task(job, [tmp_path / "m"], 0, tmp_path / "out", cancel=cancel)


def test_map_checks_cancel_before_every_line(tmp_path):
    calls = []
    cancel = threading.Event()

    def mapper(k, v):
        calls.append(v)
        cancel.set()  # cancelled while handling the very first line
        yield v, 1

    job = JobFunctions(mapper=mapper, reducer=lambda k, vs: [(k, sum(vs))])
    write_corpus(tmp_path, "in", {"f.txt": "a\nb\nc\n"})
    with pytest.raises(TaskCancelled):
        run_map_task(job, tmp_path, [Segment("in/f.txt", 0, 6)], 1, tmp_path / "m", cancel=cancel)
    assert calls == ["a"]


def test_spills_on_buffered_bytes(sample_root):
    job = load_example("wordcount.py")
    [segments] = compute_splits(storage.list_input_files(sample_root, "inputs"), 1)
    by_bytes = run_map_task(job, sample_root, segments, 2, sample_root / "b", spill_bytes=40)
    by_records = run_map_task(job, sample_root, segments, 2, sample_root / "r")
    assert by_bytes["spills"] > 1 and by_records["spills"] == 0
    for p in range(2):
        name = kv.partition_filename(p)
        spilled, unspilled = sample_root / "b" / name, sample_root / "r" / name
        assert list(kv.read_run(spilled)) == list(kv.read_run(unspilled))
