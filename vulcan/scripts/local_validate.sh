#!/usr/bin/env bash
# Local validation: everything that can be checked WITHOUT a GPU.
#
#   vulcan/scripts/local_validate.sh
#
# Runs the unit suite, the internal consistency self-test, the score-aware planner,
# and the mock A/B ablation.  All of it is CPU-only, standard-library-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

echo "== python =="; "$PY" --version
echo; echo "== unit tests (74 cases) =="
"$PY" -m unittest discover -s "$ROOT/tests" -p 'test_*.py'

echo; echo "== selftest =="
"$PY" -m vulcan selftest

echo; echo "== dataset / headroom =="
"$PY" -m vulcan stats

echo; echo "== score-aware plan (200 GPU-hours) =="
"$PY" -m vulcan plan --hours 200 | head -12

echo; echo "== mock A/B ablation =="
"$PY" -m vulcan demo --hours 200 --seed 7 | tail -14

echo; echo "ALL LOCAL CHECKS PASSED"
