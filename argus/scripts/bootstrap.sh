#!/usr/bin/env bash
# Vendor the upstreams ARGUS reproduces / targets, at fixed commits.  Nothing here is
# committed: vendored trees are gitignored.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$HERE/vendor"
mkdir -p "$VENDOR"

CUDAFORGE_PIN="eaba693fa01bdafc9848334cada2d789e240ce3f"
SOLBENCH_PIN="a9fa0804c793d438e70850c33fe34426e66d53dd"

clone_pin() {  # url pin dest
  local url="$1" pin="$2" dest="$3"
  if [ -d "$dest/.git" ]; then
    echo "[bootstrap] $dest exists; fetching"
    git -C "$dest" fetch --all --quiet
  else
    echo "[bootstrap] cloning $url"
    git clone --quiet "$url" "$dest"
  fi
  git -C "$dest" checkout --quiet "$pin"
  echo "[bootstrap] $dest @ $(git -C "$dest" rev-parse HEAD)"
}

clone_pin "https://github.com/OptimAI-Lab/CudaForge.git" "$CUDAFORGE_PIN" "$VENDOR/CudaForge"
clone_pin "https://github.com/NVIDIA/SOL-ExecBench.git" "$SOLBENCH_PIN" "$VENDOR/SOL-ExecBench"

echo
echo "[bootstrap] done."
echo "  reproduction target (CudaForge) is at   $VENDOR/CudaForge"
echo "  leaderboard harness (SOL-ExecBench) is  $VENDOR/SOL-ExecBench"
echo "  GPU-only steps (env, NCU permissions, one task run) are in docs/B300_RUNBOOK.md"
