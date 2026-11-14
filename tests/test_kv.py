import json
import os
import subprocess
import sys
from contextlib import ExitStack

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
