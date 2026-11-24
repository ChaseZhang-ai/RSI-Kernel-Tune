#!/usr/bin/env bash
# Everything that can be validated without a GPU.  Exits non-zero on the first failure.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

echo "== 1/5 unit tests =="
python3 -m pytest -q tests

echo
echo "== 2/5 selftest =="
python3 -m argus.cli selftest

echo
echo "== 3/5 reward-hack detection (CudaForge gate vs ARGUS) =="
python3 -m argus.cli hacklab --seed 1

echo
echo "== 4/5 attribution + pre-screen consistency =="
python3 -m argus.cli attrib

echo
echo "== 5/5 verification + search ablation (mock landscape) =="
python3 -m argus.cli ab --seeds 6 --evals 40

echo
echo "all local validations passed (no GPU used)."
