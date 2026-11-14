import traceback

import pytest

from mapreduce.jobcode import JobCodeError, load_job, pairs, top_level_names

GOOD = """
import re as regex
from os import path

def mapper(key, value):
    yield value, 1

def reducer(key, values):
    yield key, sum(values)

combiner = reducer
"""


def test_top_level_names():
    names = top_level_names(GOOD)
    assert {"mapper", "reducer", "combiner", "regex", "path"} <= names


def test_missing_reducer_is_rejected():
    with pytest.raises(JobCodeError, match="reducer"):
        top_level_names("def mapper(k, v):\n    yield k, v\n")


def test_syntax_error_reports_line():
    with pytest.raises(JobCodeError, match="line 2"):
        top_level_names("def mapper(k, v):\n  yield k v\n", "bad.py")


def test_load_job():
    job = load_job(GOOD)
    assert list(job.mapper("f", "x")) == [("x", 1)]
    assert job.combiner is job.reducer
    assert load_job("def mapper(k, v): pass\ndef reducer(k, v): pass\n").combiner is None


def test_load_rejects_non_callable():
    with pytest.raises(JobCodeError, match="not callable"):
        load_job("mapper = 3\ndef reducer(k, v): pass\n")


def test_user_tracebacks_show_source_lines():
    job = load_job(
        "def mapper(k, v):\n    return 1 / 0  # boom\ndef reducer(k, v): pass\n", "myjob.py"
    )
    with pytest.raises(ZeroDivisionError) as info:
        job.mapper("k", "v")
    text = "".join(traceback.format_exception(info.value))
    assert "myjob.py" in text and "1 / 0  # boom" in text


def test_pairs():
    assert list(pairs(None, "mapper")) == []
    assert list(pairs([("a", 1)], "mapper")) == [("a", 1)]
    with pytest.raises(TypeError, match="mapper must yield"):
        list(pairs(["abc"], "mapper"))
