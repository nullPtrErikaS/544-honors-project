import json
import os
import subprocess
import sys
from contextlib import ExitStack

import pytest

from mapreduce import kv

from .conftest import REPO

KEYS = ["the", "fox", "", "日本", 42, 3.5, None, ["a", 1], {"b": 1, "a": 2}]


def test_partition_is_stable_across_processes():
    """Python's hash() is salted per process; our partitioner must not be."""
    script = (
        "import json, sys; from mapreduce import kv; "
        "print(json.dumps([kv.partition(kv.encode_key(k), 7) for k in json.loads(sys.argv[1])]))"
    )
    results = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script, json.dumps(KEYS)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        results.add(out.strip())
    assert len(results) == 1
    assert json.loads(results.pop()) == [kv.partition(kv.encode_key(k), 7) for k in KEYS]


def test_partition_in_range():
    for k in KEYS:
        assert 0 <= kv.partition(kv.encode_key(k), 3) < 3


def test_dict_keys_encode_independent_of_order():
    assert kv.encode_key({"b": 1, "a": 2}) == kv.encode_key({"a": 2, "b": 1})


def test_encoded_records_never_contain_separators():
    for k in ["tab\there", "new\nline", "cr\rx"]:
        enc = kv.encode_key(k)
        assert "\t" not in enc and "\n" not in enc
        assert kv.decode(enc) == k


def test_run_round_trip_and_merge(tmp_path):
    run1 = kv.sort_records([(kv.encode_key(k), kv.encode_value(i)) for i, k in enumerate("dbca")])
    run2 = kv.sort_records([(kv.encode_key(k), kv.encode_value(i)) for i, k in enumerate("bze")])
    nbytes = kv.write_run(tmp_path / "1", run1)
    kv.write_run(tmp_path / "2", run2)
    assert nbytes == (tmp_path / "1").stat().st_size
    assert list(kv.read_run(tmp_path / "1")) == run1

    with ExitStack() as stack:
        merged = list(kv.merge_runs([tmp_path / "1", tmp_path / "2"], stack))
    assert [kv.decode(k) for k, _ in merged] == list("abbcdez")

    groups = {kv.decode(k): [kv.decode(v) for _, v in g] for k, g in kv.group_by_key(merged)}
    assert groups["b"] == [1, 0]


def test_format_output():
    assert kv.format_output("the", 3) == "the\t3\n"
    assert kv.format_output(["a", 1], {"x": [1]}) == '["a", 1]\t{"x": [1]}\n'
    assert kv.format_output("a\tb", "ok") == '"a\\tb"\tok\n'


def test_format_output_escapes_every_line_break():
    for sep in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029":
        line = kv.format_output(f"a{sep}b", [f"c{sep}d"])
        assert line.splitlines() == [line.rstrip("\n")], repr(sep)
        k, v = line.rstrip("\n").split("\t")
        assert json.loads(k) == f"a{sep}b" and json.loads(v) == [f"c{sep}d"]


def test_format_output_keeps_non_ascii_readable():
    assert kv.format_output("café", ["café.txt", "日本"]) == 'café\t["café.txt", "日本"]\n'


def test_merge_with_bounded_fan_in(tmp_path):
    """Many runs merge in passes, never more than `factor` files at once, same result."""
    runs = []
    for i in range(23):
        path = tmp_path / f"run{i}.kv"
        recs = sorted((kv.encode_key(f"k{j % 7}"), kv.encode_value([i, j])) for j in range(5))
        kv.write_run(path, recs)
        runs.append(path)
    with ExitStack() as stack:
        expected = list(kv.merge_runs(runs, stack))
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    with ExitStack() as stack:
        got = list(kv.merge_runs(runs, stack, tmp_dir=tmp, factor=3))
        assert all(p.name.startswith("_merge-") for p in tmp.iterdir())
    assert got == expected  # including the order of equal keys
    assert list(tmp.iterdir()) == []  # temporary runs are removed
    assert all(p.exists() for p in runs)  # inputs are left alone
    with pytest.raises(ValueError, match="needs a tmp_dir"), ExitStack() as stack:
        kv.merge_runs(runs, stack, factor=3)
