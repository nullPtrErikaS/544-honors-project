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


@pytest.mark.parametrize("item", ["of", {"a": 1, "b": 2}, ("a", 1, 2), 5, None])
def test_pairs_rejects_anything_but_two_item_tuples_or_lists(item):
    with pytest.raises(TypeError, match="mapper must yield"):
        list(pairs([item], "mapper"))
    assert list(pairs([["a", 1]], "mapper")) == [("a", 1)]


def test_definitions_inside_top_level_blocks_are_found():
    source = (
        "try:\n    import ujson as json\nexcept ImportError:\n    import json\n"
        "FAST = True\n"
        "if FAST:\n    def mapper(k, v):\n        yield v, 1\n"
        "else:\n    def mapper(k, v):\n        yield v, 2\n"
        "with open(__file__) if False else __import__('contextlib').nullcontext():\n"
        "    def reducer(k, vs):\n        yield k, sum(vs)\n"
    )
    assert {"mapper", "reducer", "json"} <= top_level_names(source)
    assert "inner" not in top_level_names(
        "def mapper(k, v):\n    def inner(): pass\n    yield v, 1\ndef reducer(k, vs): pass\n"
    )


def test_same_filename_jobs_keep_their_own_tracebacks():
    old = load_job("def mapper(k, v):\n    return 1 / 0  # old\ndef reducer(k, v): pass\n", "wc.py")
    load_job("def mapper(k, v):\n    return {}['x']  # new\ndef reducer(k, v): pass\n", "wc.py")
    with pytest.raises(ZeroDivisionError) as info:
        old.mapper("k", "v")
    text = "".join(traceback.format_exception(info.value))
    assert "wc.py" in text and "1 / 0  # old" in text and "# new" not in text
