"""Where a job's files live in shared storage, and how the boss commits them.

Every task *attempt* writes to its own private location, so duplicate attempts (retries,
speculative backups, attempts on workers wrongly presumed dead) never interfere. The boss
is the only component that turns an attempt's output into a task's output:

* map:    ``_jobs/<job>/map/<attempt>/part-NNNNN.kv``. Committing just means remembering
          which attempt directory the reducers should read.
* reduce: ``<output>/_temporary/<attempt>/part-r-NNNNN``, atomically renamed to
          ``<output>/part-r-NNNNN`` on commit.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

from mapreduce import kv, storage

log = logging.getLogger(__name__)

JOBS_DIR = "_jobs"
TEMP_DIR = "_temporary"
SUCCESS_MARKER = "_SUCCESS"


def map_attempt_dir(job_id: str, attempt_id: str) -> str:
    return f"{JOBS_DIR}/{job_id}/map/{attempt_id}"


def reduce_attempt_file(output: str, attempt_id: str, partition: int) -> str:
    return f"{output}/{TEMP_DIR}/{attempt_id}/{kv.output_filename(partition)}"


def reduce_final_file(output: str, partition: int) -> str:
    return f"{output}/{kv.output_filename(partition)}"


class FilesystemCommitter:
    """Commits task output and cleans up after jobs, on the shared volume."""

    def __init__(self, root: Path):
        self.root = root

    def commit_reduce(self, output: str, attempt_id: str, partition: int) -> None:
        src = storage.resolve(self.root, reduce_attempt_file(output, attempt_id, partition))
        dst = storage.resolve(self.root, reduce_final_file(output, partition))
        os.replace(src, dst)  # atomic: readers see the old file or the new one, never half

    def finish_job(self, job_id: str, output: str, succeeded: bool) -> None:
        out = storage.resolve(self.root, output)
        if succeeded:
            out.mkdir(parents=True, exist_ok=True)
            (out / SUCCESS_MARKER).touch()
        # Deleting intermediate data can take a while; don't block the scheduler.
        doomed = [storage.resolve(self.root, f"{JOBS_DIR}/{job_id}"), out / TEMP_DIR]
        threading.Thread(target=_remove_all, args=(doomed,), daemon=True).start()

    def reset(self) -> None:
        """Remove intermediate data left behind by a previous boss process."""
        _remove_all([storage.resolve(self.root, JOBS_DIR)])


def _remove_all(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        log.debug("removed %s", path)
