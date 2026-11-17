"""Generate a deterministic synthetic text corpus for benchmarks.

Words follow a Zipf distribution (like natural language: a few words are very common, most
are rare), drawn from a made-up vocabulary, so the corpus is reproducible from a seed and
needs no download.

    python scripts/gen_corpus.py --size-mb 100 --files 8 data/corpus
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

SYLLABLES = ["ka", "lo", "mi", "ra", "ten", "su", "vo", "pel", "dra", "qui", "ne", "zor", "ba"]


def vocabulary(size: int, rng: random.Random) -> list[str]:
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choices(SYLLABLES, k=rng.randint(1, 4))))
    return sorted(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--size-mb", type=float, default=100, help="total size (MiB)")
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=20_000, help="distinct words")
    parser.add_argument("--zipf", type=float, default=1.1, help="Zipf exponent")
    parser.add_argument("--seed", type=int, default=544)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vocab = vocabulary(args.vocab, rng)
    rng.shuffle(vocab)
    cum_weights = list(
        itertools.accumulate(1 / (rank**args.zipf) for rank in range(1, len(vocab) + 1))
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file = int(args.size_mb * 2**20 / args.files)
    for i in range(args.files):
        path = args.out_dir / f"part-{i:03d}.txt"
        written = 0
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            while written < per_file:
                words = rng.choices(vocab, cum_weights=cum_weights, k=2000)
                lines = [" ".join(words[j : j + 10]) for j in range(0, len(words), 10)]
                chunk = "\n".join(lines) + "\n"
                f.write(chunk)
                written += len(chunk)
        print(f"wrote {path} ({written / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
