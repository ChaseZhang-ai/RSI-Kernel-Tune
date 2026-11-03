#!/usr/bin/env bash
# smoke_wp19.sh — WP19 acceptance §3.6: the real-GPU smoke on the local 3060.
#
# Builds a minimal card tree from the RENDERED gpu_entry template (cardkit,
# with the WP19 cache/precheck blocks), then runs REAL jobs on GPU 0:
#   job A  the probe batch (the manifest's composed command): per-script
#          markers, verdicts, clock resets between scripts, DNF isolation;
#   job B  the same content under a different tree sha — the build-cache key
#          line must echo the SAME narrowed key (the cross-tree reuse the
#          narrowing exists for);
#   job C  narrow OFF — the whole-tree key (different per commit) — the
#          baseline miss-by-construction the T5 evidence needs.
# The log lines this script greps are pasted into CHANGES-WP19.md as the
# real-run evidence. Run under with-lock gpu:
#   ~/tools/bin/with-lock gpu 1800 -- bash scripts/smoke_wp19.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$(mktemp -d /tmp/wp19smoke.XXXXXX)"
OP="$BASE/operators/smoke"
mkdir -p "$OP"
echo "[smoke] tree: $OP"

PYTHON="${FK_SMOKE_PY:-$HOME/venvs/forgekernel/bin/python}"
PYTHONPATH="$REPO/src" "$PYTHON" - "$OP" <<'PYEOF'
import json, sys
from pathlib import Path
from forgekernel.cardkit import (GPU_ENTRY_TEMPLATE, _WP11_CACHE_BLOCK,
                                 _WP11_ALLNCU_CASE, render_gpu_entry)
op = Path(sys.argv[1])
(op / "eval" / "problem").mkdir(parents=True)
(op / "notes").mkdir()
# the rendered job entry: the WP19 key enumerator + cache + preflight ride it
text = render_gpu_entry(name="smoke", suite="forgekernel", arch="sm_86",
                        clock="", mem_clock="", require_lock="0", clock_note="")
(op / "gpu_entry.sh").write_text(text, encoding="utf-8")
(op / "eval" / "problem" / "problem.json").write_text(
    json.dumps({"kernel_file": "kernel.cu", "entry": "run"}), encoding="utf-8")
(op / "kernel.cu").write_text(
    "// fk-cuda-cflags: -O3\n#include \"smoke.cuh\"\n"
    "__global__ void smoke_kernel(float* x, int n) { int i = threadIdx.x; "
    "if (i < n) x[i] = x[i] * 2.0f; }\n", encoding="utf-8")
(op / "smoke.cuh").write_text("#pragma once\n#define SMOKE_TILE 128\n", encoding="utf-8")
# two self-contained probe scripts: the first PASSES, the second DNFs —
# per-script verdicts must not condemn each other
(op / "notes" / "probe_a.py").write_text(
    "print('PROBE PASS max_diff=0.001')\n", encoding="utf-8")
(op / "notes" / "probe_b.py").write_text(
    "import sys\nprint('about to die', flush=True)\nsys.exit(7)\n", encoding="utf-8")
PYEOF

cat > "$OP/submit.toml" <<EOF
[job]
timeout = 120
[repo]
delivery = "local"
branch = "main"
[exec]
gpu = "0"
EOF

git -C "$OP" init -q -b main
git -C "$OP" add -A
git -C "$OP" -c user.email=s@t -c user.name=s commit -qm "smoke tree"
cd "$OP"

RUN="$PYTHON -m forgekernel.submit smoke --base-dir $BASE"
FKENV="FK_BUILD_CACHE=1 FK_BUILD_CACHE_KEY_NARROW=1 FK_PROBE_BATCH_SETTLE_S=1"

echo "[smoke] job A: the probe batch (manifest-composed command)"
eval "$RUN submit --mode 'cmd:echo \"=== fk-probe-batch begin: notes/probe_a.py ===\" ; python3 notes/probe_a.py ; echo \"=== fk-probe-batch end: notes/probe_a.py rc=\$? ===\" ; sleep 1 ; echo \"=== fk-probe-batch begin: notes/probe_b.py ===\" ; python3 notes/probe_b.py ; echo \"=== fk-probe-batch end: notes/probe_b.py rc=\$? ===\"'"
sleep 8
LOGA=$(ls -t .fk-jobs/*.log 2>/dev/null | head -1)
[ -z "$LOGA" ] && LOGA=$(find . -name "*.log" -newer submit.toml 2>/dev/null | head -1)
echo "--- job A log (batch region) ---"
grep -E "fk-probe-batch|PROBE PASS|build cache:|task=cmd" "${LOGA:-/dev/null}" | head -12

echo "[smoke] job B: same kernel content, different tree sha (narrow ON)"
git -C "$OP" -c user.email=s@t -c user.name=s commit -q --allow-empty -m "empty move"
eval "env $FKENV $RUN submit --mode 'cmd:echo key-check'"
sleep 8
LOGB=$(ls -t .fk-jobs/*.log 2>/dev/null | head -1)
echo "--- job B build-cache line ---"
grep -E "build cache:" "${LOGB:-/dev/null}" | head -2

echo "[smoke] job C: narrow OFF (the baseline whole-tree key)"
eval "env FK_BUILD_CACHE=1 $RUN submit --mode 'cmd:echo key-check-off'"
sleep 8
LOGC=$(ls -t .fk-jobs/*.log 2>/dev/null | head -1)
echo "--- job C build-cache line ---"
grep -E "build cache:" "${LOGC:-/dev/null}" | head -2

echo "[smoke] done; tree kept at $OP"
