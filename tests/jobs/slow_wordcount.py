"""Word count that sleeps on every line, so tasks last long enough to crash or slow down.

The SLOW_FACTOR environment variable *of the worker process* multiplies the delay, which
lets a test make one worker a straggler without any test hooks in the system itself.
"""

import os
import time

DELAY = 0.003 * float(os.environ.get("SLOW_FACTOR", "1"))


def mapper(key, value):
    time.sleep(DELAY)
    for word in value.split():
        yield word, 1


def reducer(key, values):
    time.sleep(DELAY)
    yield key, sum(values)
