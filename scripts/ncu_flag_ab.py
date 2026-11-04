#!/usr/bin/env python
"""A/B the NCU collection flags on one operator card (WP9 task 1's verification).

The harness profiles with ``--cache-control none --clock-control none`` (WP9): the
profile then reads the SAME regime the timing windows run in — warm L2 and the
board's clock state — instead of NCU's default (flush every cache before every
pass, lock to base clocks). This script measures both regimes back to back on one
card's ``eval/ncu_run.py`` and reports the direction the WP9 rationale predicts:

  * ``lts__t_sector_hit_rate.pct`` (l2_hit_rate_pct) — warm >= flush, strictly
    higher for kernels whose operands actually sit in L2 across calls (attention
    K/V, repeated tiles). A warm reading BELOW the flush one contradicts the
    rationale: that is the ``--cache-control none`` revert signal.
  * ``compute_sol_pct`` / ``gpu__time_active.sum`` — base-clock vs the board's
    clock state moves the SOL denominator and the time; no direction is asserted
    here (a board-locked card and a free-running card land differently), the
    numbers are reported so the regime shift is visible.

Variance guard (the WP9 fallback rule): each regime is sampled ``--repeats``
times; if the warm reading's spread exceeds 2x the flush spread, the verdict
says to KEEP ``--clock-control none`` but REVERT ``--cache-control none`` (a
warmer cache made the reading noisier than the regime alignment is worth) and
record the data in CHANGES.

Needs a GPU machine with ``ncu`` on PATH and an operator directory whose card
carries ``eval/ncu_run.py`` and a ``gpu_entry.sh`` naming NCU_METRICS. Without
``--run`` it prints what it would execute and exits 0 (so CI / a CPU checkout
can smoke the argument handling). The machine this package was written on is an
RTX 3060 6GB — pass ``--launch-count 4..8`` there to keep the job short; the
directional conclusion stands, the absolute numbers do not travel to the board.

Usage:
    python scripts/ncu_flag_ab.py operators/<op> --run --launch-count 6
    python scripts/ncu_flag_ab.py operators/<op>            # dry run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The metrics line the card's gpu_entry.sh exports (kept in sync with
# inner/ncu.py KEY_METRICS — see the sync test in tests/test_ncu.py).
_METRICS_RE = re.compile(r"^NCU_METRICS=\"(.+)\"$", re.M)
_WARM_FLAGS = ["--cache-control", "none", "--clock-control", "none"]
_FLUSH_FLAGS = ["--cache-control", "all", "--clock-control", "base"]
# The comparisons the verdict reads. l2_hit_rate_pct is the directional one.
_L2 = "lts__t_sector_hit_rate.pct"


def card_metrics(op_dir: Path) -> str:
    """The card's NCU_METRICS string, read off its gpu_entry.sh."""
    entry = op_dir / "gpu_entry.sh"
    if not entry.is_file():
        raise SystemExit(f"{entry}: no gpu_entry.sh — pass an operator card directory")
    m = _METRICS_RE.search(entry.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"{entry}: no NCU_METRICS= line — this card's ncu task "
                         f"does not carry the harness metric set")
    return m.group(1)


def profile(op_dir: Path, flags: list[str], launch_count: int, metrics: str) -> str:
    """One ncu collection over the card's own ncu_run.py, CSV on stdout."""
    cmd = ["ncu", "--csv", "--target-processes", "all", "--profile-from-start", "off",
           *flags, "--launch-count", str(launch_count), "--kill", "yes",
           "--metrics", metrics, "python3", "eval/ncu_run.py"]
    print(f"[ncu_flag_ab] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=op_dir, capture_output=True, text=True)
    if proc.returncode != 0 or "==ERROR==" in proc.stderr:
        raise SystemExit(f"ncu failed rc={proc.returncode}: {proc.stderr[:2000]}")
    return proc.stdout


def _read(out: str, key: str):
    """One key-metric value off the profiled dominant kernel (None = not reported)."""
    from peregrine_kernel.inner.ncu import profiler_from_ncu
    prof = profiler_from_ncu(out)
    return prof.get(key) if prof else None


def _spread(values: list) -> float:
    """Relative spread of a regime's repeated readings (0 when it never read)."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    return (hi - lo) / lo if lo > 0 else float("inf")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("op_dir", type=Path, help="operator card directory (gpu_entry.sh + eval/ncu_run.py)")
    p.add_argument("--launch-count", type=int, default=6,
                   help="launches per profile (small on a 6GB consumer card)")
    p.add_argument("--repeats", type=int, default=2,
                   help="profiles per regime (the variance guard needs >= 2)")
    p.add_argument("--metrics", default=None, help="override the card's NCU_METRICS")
    p.add_argument("--run", action="store_true", help="actually run ncu (default: dry run)")
    args = p.parse_args()
    op_dir = args.op_dir.resolve()
    metrics = args.metrics or card_metrics(op_dir)

    warm_cmd = ("ncu --csv --target-processes all --profile-from-start off "
                + " ".join(_WARM_FLAGS) + f" --launch-count {args.launch_count} "
                "--kill yes --metrics \"$NCU_METRICS\" python3 eval/ncu_run.py")
    print(f"[ncu_flag_ab] op={op_dir.name} metrics={len(metrics.split(','))} names\n"
          f"[ncu_flag_ab] warm : {warm_cmd}\n"
          "[ncu_flag_ab] flush: same with " + " ".join(_FLUSH_FLAGS))
    if not args.run:
        print("[ncu_flag_ab] dry run — pass --run on a GPU machine to measure")
        return 0

    readings = {"warm": [], "flush": []}
    for regime, flags in (("warm", _WARM_FLAGS), ("flush", _FLUSH_FLAGS)):
        for _ in range(max(1, args.repeats)):
            out = profile(op_dir, flags, args.launch_count, metrics)
            readings[regime].append(_read(out, _L2))
        print(f"[ncu_flag_ab] {regime} {_L2}: {readings[regime]}")

    warm, flush = readings["warm"], readings["flush"]
    w, f = ([v for v in side if v is not None] for side in (warm, flush))
    if not w or not f:
        print("[ncu_flag_ab] INCONCLUSIVE: a regime produced no l2_hit_rate reading "
              "(metric dropped by the arch filter?) — record as no-data in CHANGES")
        return 1
    warm_mean = sum(w) / len(w)
    flush_mean = sum(f) / len(f)
    spread_w, spread_f = _spread(w), _spread(f)
    print(f"[ncu_flag_ab] warm mean {warm_mean:.3f} (spread {spread_w:.4f})  "
          f"flush mean {flush_mean:.3f} (spread {spread_f:.4f})")
    if warm_mean < flush_mean:
        print(f"[ncu_flag_ab] CONTRADICTS the rationale: warm L2 hit rate read BELOW "
              f"the flush one ({warm_mean:.3f} < {flush_mean:.3f}) — revert "
              f"--cache-control none for this card and record the data in CHANGES")
        return 2
    if spread_w > 2 * max(spread_f, 1e-12):
        print(f"[ncu_flag_ab] VARIANCE GUARD: warm spread {spread_w:.4f} > 2x the flush "
              f"spread {spread_f:.4f} — keep --clock-control none, revert "
              f"--cache-control none, record the data in CHANGES")
        return 3
    print(f"[ncu_flag_ab] direction holds: warm >= flush "
          f"(+{warm_mean - flush_mean:.3f} pct) within the variance guard — "
          f"the warm-regime reading stands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
