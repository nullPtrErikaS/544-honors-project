"""Generate the protobuf/gRPC Python modules from ``mapreduce/mr.proto``.

Run ``python -m mapreduce.protogen`` once after cloning (the Dockerfile does the same).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROTO = REPO / "mapreduce" / "mr.proto"
GENERATED = [REPO / "mapreduce" / name for name in ("mr_pb2.py", "mr_pb2_grpc.py", "mr_pb2.pyi")]


def is_stale() -> bool:
    mtime = PROTO.stat().st_mtime
    return any(not p.exists() or p.stat().st_mtime < mtime for p in GENERATED)


def build() -> None:
    from grpc_tools import protoc

    args = [
        "grpc_tools.protoc",
        f"-I{REPO}",
        f"--python_out={REPO}",
        f"--pyi_out={REPO}",
        f"--grpc_python_out={REPO}",
        str(PROTO),
    ]
    if protoc.main(args) != 0:
        raise RuntimeError("protoc failed")


def main() -> None:
    build()
    print("generated", ", ".join(str(p.relative_to(REPO)) for p in GENERATED), file=sys.stderr)


if __name__ == "__main__":
    main()
