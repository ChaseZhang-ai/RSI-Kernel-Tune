#!/usr/bin/env bash
# PK_CLAUDE_BIN 需要单一可执行路径（judge 通道按 list-argv 直接 exec，不做 shell 分词），
# 本 wrapper 把调用转给同目录的 stub（python + 脚本两个词在壳里拆开）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PK_AGENT_PY:-$HOME/venvs/ml/bin/python3}" "$HERE/agent_stub.py" "$@"
