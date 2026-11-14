"""Inverted index: for every word, which input files contain it?

The mapper uses its *key* (the input file path), which word count ignores.
"""

import re

WORD = re.compile(r"[a-z0-9']+")


def mapper(key, value):
    for word in set(WORD.findall(value.lower())):
        yield word, key


def reducer(key, values):
    yield key, sorted(set(values))
