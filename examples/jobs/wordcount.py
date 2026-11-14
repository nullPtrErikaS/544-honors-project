"""Word count: how many times does each word appear in the input?

Run it with:
    ./mr submit --input inputs/sample --output outputs/wordcount \
                --job examples/jobs/wordcount.py --maps 8 --reduces 4
"""

import re

WORD = re.compile(r"[a-z0-9']+")


def mapper(key, value):
    """key: input file path, value: one line of text."""
    for word in WORD.findall(value.lower()):
        yield word, 1


def reducer(key, values):
    """values: every count emitted for this word."""
    yield key, sum(values)
