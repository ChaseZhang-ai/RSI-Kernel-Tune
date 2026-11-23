#!/usr/bin/env bash
# Bootstrap the reproduction: clone the pinned upstream ForgeKernel and install it.
#
#   vulcan/scripts/bootstrap.sh              # clone + checkout pin + editable install
#   vulcan/scripts/bootstrap.sh --no-install # clone + checkout only
#
# Requires git and Python >= 3.11 for the upstream (VULCAN itself runs on 3.9+).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/vendor/UPSTREAM.lock"
DEST="$ROOT/vendor/ForgeKernel"

REPO="$(grep -E '^repo=' "$LOCK" | cut -d= -f2-)"
COMMIT="$(grep -E '^commit=' "$LOCK" | cut -d= -f2-)"

echo "upstream: $REPO @ $COMMIT"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --depth 1 origin "$COMMIT" || true
else
  mkdir -p "$ROOT/vendor"
  git clone "$REPO" "$DEST"
fi
git -C "$DEST" checkout "$COMMIT"

if [ "${1:-}" != "--no-install" ]; then
  PY="${PYTHON:-python3.11}"
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "warning: $PY not found; upstream requires Python >= 3.11." >&2
    echo "         set PYTHON=... or install python3.11, then:" >&2
    echo "         $PY -m pip install -e $DEST" >&2
    exit 0
  fi
  "$PY" -m pip install -e "$DEST"
  echo "installed upstream into $PY"
fi

echo "done. next: scripts/local_validate.sh, then read docs/B300_RUNBOOK.md"
