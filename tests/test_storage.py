"""Unit tests for storage paths: normalization, resolution, and input listing."""

from __future__ import annotations

import pytest

from mapreduce import storage
from mapreduce.storage import StorageError


@pytest.mark.parametrize(
    "path, expected",
    [
        ("inputs/x", "inputs/x"),
        ("/inputs/x", "inputs/x"),
        ("//inputs/x", "inputs/x"),  # PurePosixPath would keep "//" as a separate root
        ("///inputs//x/", "inputs/x"),
        ("\\\\_jobs\\x", "_jobs/x"),
        ("./inputs/./x", "inputs/x"),
        ("  inputs/x  ", "inputs/x"),
        ("", ""),
        ("/", ""),
        (".", ""),
    ],
)
def test_normalize_is_canonical(path, expected):
    assert storage.normalize(path) == expected
    assert storage.normalize(expected) == expected


@pytest.mark.parametrize("path", ["..", "../etc", "inputs/../../etc", "//..", "a\\..\\b"])
def test_parent_references_are_rejected(path):
    with pytest.raises(StorageError, match=r"'\.\.'"):
        storage.normalize(path)


def test_resolve_stays_under_root(tmp_path):
    assert storage.resolve(tmp_path, "") == tmp_path
    assert storage.resolve(tmp_path, "//inputs/x") == tmp_path / "inputs" / "x"
    p = tmp_path / "inputs" / "x"
    assert storage.to_storage_path(tmp_path, p) == "inputs/x"


def test_list_input_files_skips_hidden_and_sorts(tmp_path):
    for rel, data in {
        "in/b.txt": "bb",
        "in/a.txt": "a",
        "in/sub/c.txt": "ccc",
        "in/_SUCCESS": "",
        "in/.hidden": "x",
        "in/_temporary/attempt/part": "x",
        "in/.git/config": "x",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    assert storage.list_input_files(tmp_path, "in") == [
        ("in/a.txt", 1),
        ("in/b.txt", 2),
        ("in/sub/c.txt", 3),
    ]
    assert storage.list_input_files(tmp_path, "/in/a.txt") == [("in/a.txt", 1)]


def test_missing_input_is_an_error(tmp_path):
    with pytest.raises(StorageError, match="does not exist: nope"):
        storage.list_input_files(tmp_path, "nope")
