#!/usr/bin/env python
"""Backtest the analytic floor against the published kernels (WP9 task 2's gate).

For every kernel in ``scripts/floor_backtest.json`` (the 24 kernels of
``kernels/README.md`` plus the post-paper DSV4.1 kernel) this computes the
analytic floor of the kernel's frozen campaign problem and prints the ratio

    ratio = measured_best_ms / floor_ms

A real kernel must sit AT or ABOVE its floor: ratio < 1.0 means the floor is
above something that was actually measured, i.e. a formula bug — the script
exits non-zero on any such row (the acceptance gate). The README's healthy band
is [1.0, ~2.5]: a good kernel lands within ~2.5x of the problem floor.

DATA HONESTY: the repo records the published kernels' results as RELATIVE
speedups only — no absolute milliseconds exist anywhere in this tree. Entries
therefore ship ``best_ms: null`` and their ratio prints as no-data; the gate
covers every entry that CARRIES a measured best (fill ``best_ms`` from the run
archive / board — paper Figure 3 numbers, device median; the H100 rows ran at
SM clocks locked to 1590 MHz, which only widens the healthy band). Peak
parameters are matched to the results table's hardware (H100 rows price at the
sm90 peaks, B200 rows at the sm100 peaks); this box's RTX 3060 has no
comparable results and never participates.

``--selftest`` exercises the gate itself with synthetic problems whose measured
bests sit at known multiples of their floors (one row deliberately below 1.0,
run in a subprocess and expected to fail there).

Usage:
    python scripts/backtest_floors.py                 # the manifest table
    python scripts/backtest_floors.py --selftest      # the gate's own arithmetic
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgekernel.loop.analytic_floor import floor_for   # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "floor_backtest.json"
HEALTHY_BAND = (1.0, 2.5)


def _floor_problem(entry: dict) -> dict | None:
    """The floor problem of one manifest entry (None when the contract on file
    does not pin the campaign axes)."""
    if not entry.get("axes"):
        return None
    return {"arch": entry.get("arch", ""),
            "axes": entry["axes"],
            "inputs": entry.get("inputs") or [],
            "outputs": entry.get("outputs") or [],
            "family": entry.get("family"),
            "name": entry.get("name", "")}


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def run_manifest(manifest_path: Path = MANIFEST) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, failures = [], []
    print(f"{'kernel':<52} {'arch':<6} {'floor_ms':>10} {'best_ms':>9} {'ratio':>7}  band")
    print("-" * 100)
    for entry in data["kernels"]:
        problem = _floor_problem(entry)
        if problem is None:
            print(f"{entry['kernel']:<52} {entry.get('arch', ''):<6} "
                  f"{'—':>10} {'—':>9} {'no-data':>7}  contract on file does not pin the axes")
            continue
        floor = floor_for(problem)
        ms = floor.get("floor_ms")
        best = entry.get("best_ms")
        if ms is None:
            why = "; ".join(floor.get("assumptions") or [])[:60]
            print(f"{entry['kernel']:<52} {entry.get('arch', ''):<6} "
                  f"{'n/a':>10} {_fmt(best) if best else '—':>9} {'no-data':>7}  {why}")
            continue
        if not best:
            print(f"{entry['kernel']:<52} {entry.get('arch', ''):<6} "
                  f"{ms:>10.4f} {'—':>9} {'no-data':>7}  fill best_ms from the run archive")
            continue
        ratio = best / ms
        mark = "" if ratio >= 1.0 else "  <-- FLOOR ABOVE MEASURED: formula bug"
        if ratio < 1.0:
            failures.append(entry["kernel"])
        print(f"{entry['kernel']:<52} {entry.get('arch', ''):<6} "
              f"{ms:>10.4f} {best:>9.4f} {ratio:>7.3f}  "
              f"[{HEALTHY_BAND[0]:.1f}, {HEALTHY_BAND[1]:.1f}]{mark}")
        rows.append(ratio)
    print("-" * 100)
    print(f"{len(rows)} ratio(s) computed, {len(failures)} below 1.0; "
          f"the rest of the table has no measured best on file (repo records "
          f"relative speedups only) — fill best_ms in {manifest_path.name} to widen the gate")
    if failures:
        print(f"GATE FAILED: floor above a measured best for {', '.join(failures)}")
        return 1
    return 0


# --- the gate's own arithmetic (--selftest) -----------------------------------------

def _selftest_cases() -> list[dict]:
    """Synthetic problems with best_ms at known multiples of their floors, plus
    one deliberately broken row (floor above measured) the gate must catch."""
    gemm = {"kernel": "selftest/gemm_at_1.5x_floor", "arch": "sm100", "family": "gemm",
            "axes": {"m": 4096, "n": 4096, "k": 4096},
            "inputs": [{"name": "A", "shape": ["m", "k"], "dtype": "bfloat16"},
                       {"name": "B", "shape": ["k", "n"], "dtype": "bfloat16"}],
            "outputs": [{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}]}
    norm = {"kernel": "selftest/norm_at_1.1x_floor", "arch": "sm90", "family": "norm",
            "axes": {"m": 8192, "n": 4096},
            "inputs": [{"name": "x", "shape": ["m", "n"], "dtype": "bfloat16"},
                       {"name": "w", "shape": ["n"], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": ["m", "n"], "dtype": "bfloat16"}]}
    broken = {"kernel": "selftest/broken_floor_above_measured", "arch": "sm100",
              "family": "gemm",
              "axes": {"m": 128, "n": 128, "k": 128},
              "inputs": [{"name": "A", "shape": ["m", "k"], "dtype": "bfloat16"},
                         {"name": "B", "shape": ["k", "n"], "dtype": "bfloat16"}],
              "outputs": [{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}]}
    cases = []
    for entry, factor in ((gemm, 1.5), (norm, 1.1)):
        f = floor_for({k: entry[k] for k in ("arch", "axes", "inputs", "outputs", "family")})
        entry = dict(entry, best_ms=f["floor_ms"] * factor)
        cases.append(entry)
    # the broken row: "measured best" BELOW the floor — the formula would be wrong
    f = floor_for({k: broken[k] for k in ("arch", "axes", "inputs", "outputs", "family")})
    cases.append(dict(broken, best_ms=f["floor_ms"] * 0.9, _expect_fail=True))
    return cases


def run_selftest() -> int:
    """Runs the manifest gate over the synthetic cases in a subprocess and checks
    BOTH verdicts: exit 0 with only healthy rows, exit 1 when the broken row is
    present (the broken row rides in the same table, so the subprocess must fail)."""
    tmp = REPO_ROOT / "scripts" / ".floor_backtest_selftest.json"
    cases = _selftest_cases()
    broken_in_table = any(c.get("_expect_fail") for c in cases)
    try:
        tmp.write_text(json.dumps({"kernels": [{k: v for k, v in c.items()
                                                if not k.startswith("_")} for c in cases]},
                                   indent=1), encoding="utf-8")
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--manifest", str(tmp)],
                           capture_output=True, text=True)
        print(r.stdout, end="")
        if broken_in_table:
            ok = r.returncode == 1 and "GATE FAILED" in r.stdout \
                and "broken_floor_above_measured" in r.stdout
            print(f"[selftest] {'PASS' if ok else 'FAIL'}: the below-floor row was "
                  f"{'caught (exit 1)' if ok else 'NOT caught'}")
            return 0 if ok else 1
        ok = r.returncode == 0
        print(f"[selftest] {'PASS' if ok else 'FAIL'}: healthy rows exit 0")
        return 0 if ok else 1
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--selftest", action="store_true",
                   help="verify the gate's own arithmetic on synthetic cases")
    args = p.parse_args()
    if args.selftest:
        return run_selftest()
    if not args.manifest.is_file():
        raise SystemExit(f"{args.manifest}: no manifest")
    return run_manifest(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
