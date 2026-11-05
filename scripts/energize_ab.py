#!/usr/bin/env python3
"""energize_ab.py -- the parameterized skeleton of the energize acceptance A/B matrix
(WP21 / line-A G25; execution belongs to WP23).

This script only does three things: assemble commands per the matrix items of
docs/energize-protocol.md section 1, collect evidence, and print the comparison
report. **It never flips a default** — flipping a default is the WP23 integration
step's manual action, one default at a time; this script exists so that every A/B
round's command and evidence format stay uniform and re-checkable.

Usage (wrap every real run in the GPU lock):
    ~/tools/bin/with-lock gpu 1800 -- \\
        python scripts/energize_ab.py --item A4 --runs-dir .artifacts \\
        --task tasks/wp21mini --mode task-set

    python scripts/energize_ab.py --item A4 --runs-dir .artifacts --report-only

Skeleton notes: the batch section of `--mode task-set` is a commented placeholder
(WP23 connects it to whatever runner is current — usually two `rsi-kernel-tune run`
rounds, one off and one on); `--report-only` extracts the wall-clock composition
tables and the verdict comparison from completed run archives into a markdown
report. Numbers produced on the 3060 are chain evidence only; writing performance
conclusions from them is forbidden (red line 11).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The matrix items of docs/energize-protocol.md section 1:
# id -> (title, default switch, rollback switch, dependencies)
ITEMS: dict[str, dict] = {
    "A1": {"title": "ordinary task-set execution (control baseline)", "switch": None, "rollback": None,
            "depends": []},
    "A2": {"title": "NCU flags A/B (WP9 ncu_flag_ab direction confirmation)",
            "switch": "ncu regime flags (generation-point switches, WP9-native)", "rollback": "flags off returns to the old regime",
            "depends": ["scripts/ncu_flag_ab.py", "CUDA torch venv"]},
    "A3": {"title": "SPRT early-stop default", "switch": "[loop] sprt_early_stop",
            "rollback": "--sprt-early-stop=false (default false is the rollback)", "depends": ["WP15/WP19 merged"]},
    "A4": {"title": "build cache default", "switch": "RKT_BUILD_CACHE",
            "rollback": "RKT_BUILD_CACHE=0 (one line)",
            "depends": ["build log of the same kernel rebuilt across jobs",
                        "key-coverage cases (header change must miss / comment change must hit)"]},
    "A5": {"title": "floor_stop / floor_margin", "switch": "[target] floor_margin / [loop] floor_stop",
            "rollback": "keys default off is the rollback",
            "depends": ["floor_backtest.json backfill (--from-run automation missing — WP21 smoke defect D-9: "
                        "backtest_floors.py has no such flag; best_ms must be filled by hand from the run archive "
                        "per the script's own instructions, or the integration step lands the automation first)",
                        "WP15 resolvable v2"]},
    "A6": {"title": "survey channel (teacher_prior_survey)", "switch": "[loop] teacher_prior_survey",
            "rollback": "key default false is the rollback", "depends": ["mechanical presence check (F18)"]},
    "A7": {"title": "MC-DELIVERY activation", "switch": "the teacher milestone's multi_candidate block",
            "rollback": "the teacher not writing the block is the off; [loop] multi_candidate_max forwarding (dcfdcb5)",
            "depends": ["WP16 pool merged"]},
    "A8": {"title": "stagnation cli forwarding recheck (no flip)", "switch": None,
            "rollback": "the cli defaults apply when no args are passed", "depends": ["the dcfdcb5 seam"]},
}

# The default-flip order (protocol section 1 summary line); this skeleton only
# prints the hint, it does not execute it.
FLIP_ORDER = ["A8", "A2", "A4", "A3", "A5", "A6", "A7"]

# Wall-clock composition table lines (the driver's [wp11] output)
_WALLCLOCK_RE = re.compile(r"^\[wp11\]\s+(.+)$", re.M)
# verdict extraction (the verdict field of status.jsonl)
_STOP_RE = re.compile(r'"stop"\s*:\s*"([^"]+)"')


def run_item(args: argparse.Namespace) -> int:
    """Assemble (and with --execute run) the A/B two-round commands for a matrix
    item. The skeleton defaults to dry."""
    item = ITEMS[args.item]
    print(f"[energize-ab] item {args.item}: {item['title']}")
    print(f"[energize-ab] switch: {item['switch'] or '(baseline state, no switch)'}")
    print(f"[energize-ab] rollback: {item['rollback'] or '-'}")
    if item["depends"]:
        print("[energize-ab] prerequisites (incomplete = must not run):")
        for d in item["depends"]:
            print(f"  - {d}")
    order = FLIP_ORDER.index(args.item) if args.item in FLIP_ORDER else "-"
    print(f"[energize-ab] suggested flip-order position: {order} (protocol section 1: one default at a time)")

    if args.item in ("A8", "A2"):
        print("[energize-ab] this item flips no default: pure recheck / evidence; the batch section is the protocol section 1 row.")
        return 0

    if not args.execute:
        print("[energize-ab] dry skeleton (--execute assembles the real run; real runs must be wrapped in with-lock gpu):")
    for side, flag in (("A (off / baseline)", None), ("B (on / flipped)", item["switch"])):
        cmd = (f"rsi-kernel-tune run <runs-dir>/<{'off' if side.startswith('A') else 'on'}>-run  "
               f"# {side}"
               + (f"  env/key: {flag}" if flag else ""))
        print(f"  {cmd}")
    print("[energize-ab] after both rounds: --report-only summarizes the wall-clock composition and the verdict comparison.")
    print("[energize-ab] red line 11: 3060 numbers are chain evidence only; no performance conclusions.")
    return 0


def _wallclock_of(run_dir: Path) -> list[str]:
    """The [wp11] wall-clock composition lines in run.log."""
    log = run_dir / "run.log"
    if not log.is_file():
        return []
    return _WALLCLOCK_RE.findall(log.read_text(errors="replace"))


def _verdicts_of(run_dir: Path) -> list[str]:
    """The per-iteration verdict sequence of student/status.jsonl."""
    sj = run_dir / "ws" / "student" / "status.jsonl"
    if not sj.is_file():
        return []
    out = []
    for line in sj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line).get("verdict", ""))
        except ValueError:
            continue
    return out


def report(args: argparse.Namespace) -> int:
    """--report-only: extract evidence from completed run archives into a markdown
    comparison report."""
    runs = sorted(Path(args.runs_dir).glob("*")) if Path(args.runs_dir).is_dir() else []
    lines = ["# energize-ab report (chain evidence; 3060 numbers are not performance conclusions)", ""]
    for run in runs:
        if not run.is_dir():
            continue
        wc = _wallclock_of(run)
        vd = _verdicts_of(run)
        lines.append(f"## {run.name}")
        lines.append(f"- verdicts: {vd or '(no status.jsonl)'}")
        if wc:
            lines.append("- wall-clock composition:")
            lines.extend(f"  - {w}" for w in wc[-3:])
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[energize-ab] report -> {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="energize_ab.py",
        description="WP21 energize acceptance A/B matrix skeleton: assemble commands, collect evidence, report; never flips a default.")
    p.add_argument("--item", choices=sorted(ITEMS), default="A1",
                   help="the protocol section 1 matrix item (A1..A8)")
    p.add_argument("--mode", choices=("task-set", "same-job", "ncu-flag-ab"),
                   default="task-set",
                   help="the A/B shape: task-set = two full run rounds; same-job = the JustGoal same-job comparison; "
                        "ncu-flag-ab = the same kernel sampled once with old and new regime flags")
    p.add_argument("--task", default="tasks/wp21mini", help="the task card directory for the A/B")
    p.add_argument("--runs-dir", default=".artifacts", help="the run archive root")
    p.add_argument("--execute", action="store_true",
                   help="real run (the caller must still wrap it in with-lock gpu; the skeleton defaults to dry, printing the plan only)")
    p.add_argument("--report-only", action="store_true",
                   help="extract the wall-clock composition and verdict comparison from existing --runs-dir archives into a report")
    p.add_argument("--out", default=None, help="write the report to a file (default stdout)")
    args = p.parse_args(argv)

    if args.report_only:
        return report(args)
    return run_item(args)


if __name__ == "__main__":
    sys.exit(main())
