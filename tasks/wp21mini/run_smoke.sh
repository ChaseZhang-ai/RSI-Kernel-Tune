#!/usr/bin/env bash
# The WP21 mini smoke runner: a real run on this machine's 3060 (stub-agent stand-in), producing chain and dataflow evidence.
#
# Path: the full peregrine run wrap (judge -> card -> seal -> calibrate -> NCU probe -> loop.toml ->
# workspace -> baseline -> loop), with the agent binary swapped for the stub via PK_CLAUDE_BIN
# (the scripts/agent-claude.sh contract).
# Prerequisites: no local.env needed (PK_* env vars take precedence); ~/venvs/ml (CUDA torch)
# serves as the job python.
# GPU exclusivity: real runs must invoke this script wrapped in ~/tools/bin/with-lock gpu 1800 --
# (waiting for the lock is the correct behavior):
#   ~/tools/bin/with-lock gpu 1800 -- bash tasks/wp21mini/run_smoke.sh [<run_dir>]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$HOME/venvs/forgekernel/bin/python"
RUN_DIR="${1:-}"

cd "$REPO"

# editable-install hijack (~/venvs/forgekernel points at /home/zcz/forgeKernel/ForgeKernel):
# PYTHONPATH pushes this clone's src to the front, so repo_root/hw_kb/.artifacts all land
# inside the clone.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# job-side python3 = the CUDA torch venv (jobs inherit the driver process env); the driver itself runs on the peregrine_kernel venv.
export PATH="$HOME/venvs/ml/bin:$PATH"
export CUTLASS_DIR="$HOME/tools/cutlass"   # the header library the job-env precheck wants (~/tools, header-only)
# WSL driver library: torch cpp_extension's -lcuda needs a libcuda.so dev symlink;
# WSL only ships /usr/lib/wsl/lib/libcuda.so.1, so LIBRARY_PATH gets a symlink dir.
export LIBRARY_PATH="$HOME/tools/wsl-cuda-lib:${LIBRARY_PATH:-}"
export PK_DELIVERY=local
export PK_GPU=0
export PK_AGENT_PY="$HOME/venvs/ml/bin/python3"
# every agent turn (teacher/student/judge) goes through the stub (agent-claude.sh's PK_CLAUDE_BIN escape hatch)
export PK_CLAUDE_BIN="$REPO/tasks/wp21mini/agent_stub.sh"

if [ -z "$RUN_DIR" ]; then
    NEW_OUT="$("$PY" -m peregrine_kernel new "$REPO/tasks/wp21mini")"
    RUN_DIR="$(printf '%s\n' "$NEW_OUT" \
        | grep -oP '(?<=^\[peregrine\] run directory )[^:]+' | head -1)"
    if [ -z "$RUN_DIR" ]; then
        echo "[wp21-smoke] could not parse run dir from: $NEW_OUT" >&2
        exit 1
    fi
fi

echo "[wp21-smoke] run dir: $RUN_DIR"
exec "$PY" -m peregrine_kernel run "$RUN_DIR"
