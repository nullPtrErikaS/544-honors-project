#!/usr/bin/env bash
# End-to-end check against a running cluster (docker compose up -d):
# upload the sample data, run word count, and compare the result with `mr local`.
set -euo pipefail
cd "$(dirname "$0")/.."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

./mr wait-ready --workers 4 --timeout 120
./mr upload examples/data/sample inputs/sample
./mr submit --input inputs/sample --output outputs/smoke-test \
            --job examples/jobs/wordcount.py --maps 8 --reduces 4 --overwrite
./mr cat outputs/smoke-test | LC_ALL=C sort > "$tmp/actual.tsv"
./mr local --job examples/jobs/wordcount.py --input examples/data/sample | LC_ALL=C sort > "$tmp/expected.tsv"

diff "$tmp/expected.tsv" "$tmp/actual.tsv"
echo "smoke test passed: $(wc -l < "$tmp/actual.tsv") words match the sequential reference"
