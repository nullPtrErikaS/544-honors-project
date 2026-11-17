import random

import pytest

from mapreduce.local import read_lines
from mapreduce.splits import Segment, compute_splits, read_segment

WORDS = ["a", "bb", "ccc", "", "é", "日本語", "tab\there", "x" * 40]


def random_file(rng: random.Random) -> bytes:
    lines = [" ".join(rng.choices(WORDS, k=rng.randint(0, 6))) for _ in range(rng.randint(0, 30))]
    eol = rng.choice([b"\n", b"\r\n"])
    data = eol.join(line.encode() for line in lines)
    if lines and rng.random() < 0.7:
        data += eol  # sometimes no trailing newline
    return data


def test_every_line_is_read_exactly_once(tmp_path):
    """Property test: wherever the cuts fall, splits partition the lines exactly."""
    rng = random.Random(544)
    for trial in range(300):
        root = tmp_path / str(trial)
        root.mkdir()
        listing, expected = [], []
        for i in range(rng.randint(1, 4)):
            data = random_file(rng)
            (root / f"f{i}.txt").write_bytes(data)
            listing.append((f"f{i}.txt", len(data)))
            expected += read_lines(data)

        splits = compute_splits(listing, rng.randint(1, 15))
        got = [line for split in splits for seg in split for line in read_segment(root, seg)]
        assert got == expected, f"trial {trial}"


def test_splits_cover_all_bytes_evenly():
    files = [("a", 10), ("b", 0), ("c", 95), ("d", 3)]
    splits = compute_splits(files, 7)
    sizes = [sum(s.end - s.start for s in split) for split in splits]
    assert sum(sizes) == 108
    assert max(sizes) - min(sizes) <= 1
    # Segments of one file are contiguous and in order.
    c_segments = [s for split in splits for s in split if s.path == "c"]
    assert c_segments[0].start == 0 and c_segments[-1].end == 95
    assert all(x.end == y.start for x, y in zip(c_segments, c_segments[1:], strict=False))


def test_more_splits_than_bytes_gives_empty_splits():
    splits = compute_splits([("a", 3)], 8)
    assert len(splits) == 8
    assert sum(len(s) for s in splits) == 3


def test_split_count_must_be_positive():
    with pytest.raises(ValueError):
        compute_splits([("a", 3)], 0)


def test_segment_starting_on_line_boundary(tmp_path):
    (tmp_path / "f").write_bytes(b"ab\ncd\nef\n")
    # Byte 3 is the start of "cd": the segment starting there owns it.
    assert list(read_segment(tmp_path, Segment("f", 3, 4))) == ["cd"]
    assert list(read_segment(tmp_path, Segment("f", 0, 3))) == ["ab"]
    # A segment starting mid-line skips the partial line.
    assert list(read_segment(tmp_path, Segment("f", 4, 9))) == ["ef"]


def naive_splits(files, n):
    """The obvious O(files x splits) version, as a reference."""
    total = sum(size for _, size in files)
    bounds = [i * total // n for i in range(n + 1)]
    out, offset = [[] for _ in range(n)], 0
    for path, size in files:
        for i in range(n):
            lo, hi = max(bounds[i], offset), min(bounds[i + 1], offset + size)
            if lo < hi:
                out[i].append(Segment(path, lo - offset, hi - offset))
        offset += size
    return out


def test_compute_splits_matches_reference():
    rng = random.Random(7)
    for _ in range(500):
        files = [(f"f{i}", rng.choice([0, 0, 1, 3, 50, 400])) for i in range(rng.randint(0, 12))]
        n = rng.randint(1, 20)
        assert compute_splits(files, n) == naive_splits(files, n), (files, n)


def test_long_partial_line_is_skipped(tmp_path):
    long_line = b"x" * (3 * (1 << 20) + 17)
    (tmp_path / "f").write_bytes(long_line + b"\nnext\n")
    size = len(long_line) + 6
    assert list(read_segment(tmp_path, Segment("f", 1, size))) == ["next"]
    assert list(read_segment(tmp_path, Segment("f", 5, len(long_line)))) == []
