"""User job files: validation (boss side) and loading (worker side).

A job file is plain Python that defines ``mapper(key, value)`` and ``reducer(key, values)``,
and optionally ``combiner(key, values)`` with the reducer's signature.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
import types
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

REQUIRED = ("mapper", "reducer")

Fn = Callable[..., Iterable[Any] | None]


class JobCodeError(ValueError):
    """The job file is malformed (syntax error, missing functions, ...)."""


@dataclass(frozen=True)
class JobFunctions:
    mapper: Fn
    reducer: Fn
    combiner: Fn | None = None


def top_level_names(source: str, filename: str = "job.py") -> set[str]:
    """Names a job file defines at top level, found *without executing it*.

    The boss uses this to reject broken jobs at submit time without running user code.
    Raises :class:`JobCodeError` on syntax errors or if a required function is missing.
    """
    try:
        tree = ast.parse(source, filename)
    except SyntaxError as e:
        raise JobCodeError(f"{filename}, line {e.lineno}: {e.msg}") from None

    names: set[str] = set()
    for node in _import_time_statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)

    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise JobCodeError(f"{filename} must define {' and '.join(missing)}")
    return names


def _import_time_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run when the module is loaded, including inside top-level
    ``if``/``try``/``with`` blocks (but not inside functions or classes)."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            blocks = [node.body, node.orelse]
        elif isinstance(node, (ast.Try, ast.TryStar)):
            blocks = [node.body, node.orelse, node.finalbody, *(h.body for h in node.handlers)]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            blocks = [node.body]
        else:
            continue
        for block in blocks:
            yield from _import_time_statements(block)


def load_job(source: str, filename: str = "job.py") -> JobFunctions:
    """Execute a job file in a fresh module and return its functions."""
    module = types.ModuleType(filename.rsplit("/", 1)[-1].removesuffix(".py") or "job")
    module.__file__ = filename
    # Register the source so tracebacks from user code show the offending lines. The key
    # includes a hash of the source: a resubmitted, edited file with the same name must not
    # replace the source of an older job that is still loaded.
    key = f"{filename} [{hashlib.sha256(source.encode()).hexdigest()[:8]}]"
    linecache.cache[key] = (len(source), None, source.splitlines(True), key)
    try:
        code = compile(source, key, "exec")
    except SyntaxError as e:
        raise JobCodeError(f"{filename}, line {e.lineno}: {e.msg}") from None
    exec(code, module.__dict__)

    fns = {}
    for name in (*REQUIRED, "combiner"):
        fn = getattr(module, name, None)
        if fn is None and name in REQUIRED:
            raise JobCodeError(f"{filename} must define {name}")
        if fn is not None and not callable(fn):
            raise JobCodeError(f"{name} in {filename} is not callable")
        fns[name] = fn
    return JobFunctions(**fns)


def pairs(result: Iterable[Any] | None, fn_name: str) -> Iterable[tuple[Any, Any]]:
    """Iterate over the ``(key, value)`` pairs a user function yielded or returned."""
    if result is None:
        return
    for item in result:
        # Only tuples and lists: `k, v = "of"` would silently unpack a bare 2-letter string.
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{fn_name} must yield (key, value) pairs, got {item!r}")
        k, v = item
        yield k, v
