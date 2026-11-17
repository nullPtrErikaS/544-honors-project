"""Where a job's files live in shared storage, and how the boss commits them.

Every task *attempt* writes to its own private location, so duplicate attempts (retries,
speculative backups, attempts on workers wrongly presumed dead) never interfere. The boss
is the only component that turns an attempt's output into a task's output:

* map:    ``_jobs/<job>/map/<attempt>/part-NNNNN.kv``. Committing just means remembering
          which attempt directory the reducers should read.
* reduce: ``<output>/_temporary/<attempt>/part-r-NNNNN``, atomically renamed to
          ``<output>/part-r-NNNNN`` on commit.

The boss creates the parent directories when a job starts and moves them away when it
ends. Workers only ever create their own attempt directory inside them, so an attempt
that was killed but is still running can't bring a finished job's directories back.
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
TRASH_DIR = f"{JOBS_DIR}/_trash"
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

    def start_job(self, job_id: str, output: str) -> None:
        """Create the directories that the job's attempts write into."""
        storage.resolve(self.root, f"{JOBS_DIR}/{job_id}/map").mkdir(parents=True, exist_ok=True)
        (storage.resolve(self.root, output) / TEMP_DIR).mkdir(parents=True, exist_ok=True)

    def finish_job(self, job_id: str, output: str, succeeded: bool) -> None:
        """Clean up after a job, and mark its output complete if it succeeded.

        A failed job's output directory is removed too, so its partial ``part-r-*`` files
        can't be mistaken for results. Everything goes into ``_jobs/_trash/<job>`` with a
        rename first, which is quick and atomic, so the paths are gone before this returns.
        The slow recursive delete runs in the background.
        """
        out = storage.resolve(self.root, output)
        trash = storage.resolve(self.root, f"{TRASH_DIR}/{job_id}")
        trash.mkdir(parents=True, exist_ok=True)
        doomed = [storage.resolve(self.root, f"{JOBS_DIR}/{job_id}"), out / TEMP_DIR]
        if not succeeded:
            doomed.append(out)
        leftovers = [trash]
        for i, path in enumerate(doomed):
            try:
                path.rename(trash / str(i))
            except FileNotFoundError:
                pass
            except OSError:
                # Windows can't rename a directory while a file in it is open (a killed
                # attempt still writing). Fall back to deleting it where it is.
                leftovers.append(path)
        if succeeded:
            out.mkdir(parents=True, exist_ok=True)
            (out / SUCCESS_MARKER).touch()
        threading.Thread(target=_remove_all, args=(leftovers,), daemon=True).start()

    def reset(self) -> None:
        """Remove intermediate data and trash left behind by a previous boss process."""
        _remove_all([storage.resolve(self.root, JOBS_DIR)])


def _remove_all(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        log.debug("removed %s", path)
