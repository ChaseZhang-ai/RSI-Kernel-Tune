#!/usr/bin/env bash
# WP21 mini 冒烟 runner：本机 3060 上的真 run（stub agent 替身），产链路与数据流证据。
#
# 通路：forge run 全包装（judge→card→seal→calibrate→NCU probe→loop.toml→workspace→
# baseline→loop），agent 二进制经 FK_CLAUDE_BIN 换成 stub（scripts/agent-claude.sh 契约）。
# 前置：无需 local.env（FK_* 环境变量优先）；~/venvs/ml（CUDA torch）作 job python。
# GPU 独占：真跑必须以 ~/tools/bin/with-lock gpu 1800 -- 包裹调用本脚本（等待锁是正确行为）：
#   ~/tools/bin/with-lock gpu 1800 -- bash tasks/wp21mini/run_smoke.sh [<run_dir>]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$HOME/venvs/forgekernel/bin/python"
RUN_DIR="${1:-}"

cd "$REPO"

# editable 安装劫持（~/venvs/forgekernel 指向 /home/zcz/forgeKernel/ForgeKernel）：
# PYTHONPATH 把本 clone 的 src 压到最前，repo_root/hw_kb/.artifacts 全部落在 clone 内。
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# job 侧 python3 = CUDA torch venv（job 继承驱动进程环境）；驱动本体用 forgekernel venv。
export PATH="$HOME/venvs/ml/bin:$PATH"
export CUTLASS_DIR="$HOME/tools/cutlass"   # job-env 预检的头文件库（~/tools，header-only）
export FK_DELIVERY=local
export FK_GPU=0
export FK_AGENT_PY="$HOME/venvs/ml/bin/python3"
# 全部 agent 回合（teacher/student/judge）走 stub（agent-claude.sh 的 FK_CLAUDE_BIN 逃逸口）
export FK_CLAUDE_BIN="$REPO/tasks/wp21mini/agent_stub.sh"

if [ -z "$RUN_DIR" ]; then
    NEW_OUT="$("$PY" -m forgekernel new "$REPO/tasks/wp21mini")"
    RUN_DIR="$(printf '%s\n' "$NEW_OUT" \
        | grep -oP '(?<=^\[forge\] run directory )[^:]+' | head -1)"
    if [ -z "$RUN_DIR" ]; then
        echo "[wp21-smoke] could not parse run dir from: $NEW_OUT" >&2
        exit 1
    fi
fi

echo "[wp21-smoke] run dir: $RUN_DIR"
exec "$PY" -m forgekernel run "$RUN_DIR"
