#!/usr/bin/env bash
# RKT_CLAUDE_BIN needs a single executable path (the judge channel execs the
# list-argv directly, with no shell word-splitting), so this wrapper hands the call
# over to the stub in this directory (python + script as two argv words).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${RKT_AGENT_PY:-$HOME/venvs/ml/bin/python3}" "$HERE/agent_stub.py" "$@"
