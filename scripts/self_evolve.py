#!/usr/bin/env python3
"""Self-evolution: each generation's best kernel becomes the next generation's reference.

    python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]

A generation is an ordinary run (`forge new`, then `forge run`). When it ends with a confirmed
best kernel faster than its reference by more than the noise floor, the next generation is
written as a new task, ``tasks/<op>-g<k>/``: the same task.toml, description and shapes, the
best kernel copied into its reference/ (so the new task carries everything it needs and can be
committed and rerun elsewhere), and a target ``--step`` times the new reference's speed.

The chain stops only when the hardware gives no more: at a generation whose gain is within the
noise floor, or one that ended without a confirmed best. The noise floor is the spread of the
generation's own same-job confirmation ratios, never below NOISE_FLOOR_MIN (the loop's plateau
margin). ``--max-gen`` caps the number of generations; there is no cap by default.

The reference has source, so every generation after the first runs in source mode: the
teacher reads the previous best, the student never does (the run seals it).

Only the public commands are used; nothing here is special to the harness.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forgekernel import launch  # noqa: E402
from forgekernel.runs import list_runs  # noqa: E402
from forgekernel.task import (TaskError, dump_task_toml, load_task,  # noqa: E402
                              read_task_toml)

REF_DIR = "reference"
# == forgekernel.loop.parse's noise constants (WP1): below the CAP, two readings of one
# kernel are not told apart, however tight a handful of confirmation jobs happened to
# be. Single source of truth: import the cap (and floor) from parse; this standalone
# script keeps local copies only for the case the import cannot resolve.
try:
    from forgekernel.loop.parse import RESOLVABLE_DIFF_CAP as _RESOLVABLE_CAP  # noqa: E402
except ImportError:  # pragma: no cover - forgekernel not importable (script run standalone)
    _RESOLVABLE_CAP = 0.015   # == forgekernel.loop.parse.RESOLVABLE_DIFF_CAP
NOISE_FLOOR_MIN = _RESOLVABLE_CAP


def latest_run(task_name: str):
    runs = [r for r in list_runs() if r.task_name == task_name]
    return max(runs, key=lambda r: r.record().get("created", "")) if runs else None


def noise_floor(best: dict) -> float:
    """The relative gain a generation must clear: the spread of its same-job ratios
    ((max - min) / min), never below NOISE_FLOOR_MIN."""
    ratios = best.get("ratios") or []
    spread = (max(ratios) - min(ratios)) / min(ratios) if len(ratios) > 1 else 0.0
    return max(NOISE_FLOOR_MIN, spread)


def next_generation(task, best: dict, dest: Path, step: float, run=None) -> Path:
    """tasks/<op>-g<k>/: the task with ``best`` as its reference and target ``step``x."""
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it or start the chain elsewhere")
    if task.language != "cutedsl":
        raise ValueError("self-evolution needs a Python kernel to seal as the next reference "
                         f"(this task's language is {task.language!r})")
    dest.mkdir(parents=True)
    for name in ("description.md", "shapes.json"):
        shutil.copy2(task.dir / name, dest / name)
    if (task.dir / "inputs").is_dir():
        shutil.copytree(task.dir / "inputs", dest / "inputs")
    (dest / REF_DIR).mkdir()
    (dest / REF_DIR / "kernel.py").write_text(best["kernel"], encoding="utf-8")
    cfg = read_task_toml(task.dir)
    cfg["reference"] = {"run": f"{REF_DIR}/kernel.py:run", "source": REF_DIR}
    cfg["target"] = {**cfg.get("target", {}), "ratio": float(step)}
    cfg["target"].pop("ms", None)
    (dest / "task.toml").write_text(dump_task_toml(cfg), encoding="utf-8")
    if run is not None:
        _write_carry(task, run, dest / "carry")
    load_task(dest)                       # the new generation must be a valid task
    return dest


# --- WP8: cross-generation carry ------------------------------------------------
# The parent run's closing record used to die with the run: the measured lever
# prices, the ARCH_PROBE readings, the dead-lever verdicts and the final
# milestone's decision graph were all re-bought from zero by the next
# generation on what is (for the reference tree) the same machine. carry/ hands
# them to the next generation's teacher through the distill/recon briefs'
# PRIOR-INPUT section — hypothesis-reuse material, re-verified line by line,
# never a gate anchor, never a price source (the driver-side injection carries
# that rule verbatim into the prompt).

_CARRY_MILESTONE = "milestone.md"
_CARRY_LEVERS = "levers.json"
_CARRY_ARCH_PROBE = "arch_probe.md"
_CARRY_FALSIFIED = "falsified.md"
_ARCH_PROBE_STAMP_RE = re.compile(r"<!-- ARCH_PROBE .*?-->", re.S)


def _write_carry(task, run, dest: Path) -> None:
    """Four artifacts from the parent run into ``tasks/<op>-g<k>/carry/``:
    the final milestone (decisions + contracts + rationale, its head note
    carrying the ARCH_PROBE stamp), the lever price table, the ARCH_PROBE
    readings, and this operator's falsified-lever rows. Best-effort by design:
    an artifact the run never produced (no plateau pricing, no probes) is
    recorded as absent, never guessed."""
    dest.mkdir(parents=True, exist_ok=True)
    notes = []

    def _put(name: str, text: str, what: str) -> None:
        if text.strip():
            (dest / name).write_text(text, encoding="utf-8")
            notes.append(f"- `{name}`: {what}")
        else:
            notes.append(f"- `{name}`: absent (the parent run produced none)")

    ms_path = run.operators_dir / task.name / "milestone.md"
    ms_text = ""
    if ms_path.is_file():
        ms_text = ms_path.read_text(encoding="utf-8", errors="replace")
    _put(_CARRY_MILESTONE, ms_text,
         "the parent's final milestone: decisions, contracts, rationale notes "
         "(its head note carries the parent's ARCH_PROBE stamp)")
    levers = ""
    levers_path = run.ws / "notes" / "levers.json"
    if levers_path.is_file():
        levers = levers_path.read_text(encoding="utf-8", errors="replace")
    _put(_CARRY_LEVERS, levers, "the parent's last lever price table")
    stamp = ""
    if ms_text:
        stamp = "\n\n".join(m.group(0) for m in
                            _ARCH_PROBE_STAMP_RE.finditer(ms_text))
    _put(_CARRY_ARCH_PROBE, stamp,
         "the parent's ARCH_PROBE readings (the stamp block(s) of the milestone head)")
    ledger = REPO_ROOT / "hw_kb" / "falsified_ledger.jsonl"
    rows = []
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("op_family") == task.name:
                rows.append(line)
    _put(_CARRY_FALSIFIED, "\n".join(rows) + ("\n" if rows else ""),
         "the parent's dead-lever ledger rows (class, price, why)")
    (dest / "README.md").write_text(
        "# carry: the parent generation's closing record\n\n"
        "Consumed by the next generation's teacher through the distill/recon briefs' "
        "PRIOR-INPUT section: for hypothesis reuse and reconciliation only — re-verify "
        "every line against the new run's own measurements; never a gate anchor, never "
        "a price source.\n\n" + "\n".join(notes) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="self_evolve.py", description=__doc__.split("\n\n")[0])
    p.add_argument("task", help="the first generation's task directory")
    p.add_argument("--step", type=float, default=1.05,
                   help="each new generation's target, x its reference's speed")
    p.add_argument("--max-gen", type=int, default=None,
                   help="stop after this many generations (default: no cap)")
    args = p.parse_args(argv)
    task_dir = Path(args.task).resolve()
    base = re.sub(r"-g\d+$", "", task_dir.name)
    for gen in itertools.count(1):
        try:
            task = load_task(task_dir)
        except TaskError as e:
            sys.stderr.write(f"[self-evolve] {e}\n")
            return 2
        print(f"[self-evolve] generation {gen}: {task_dir}", flush=True)
        rc = launch.cmd_new([str(task_dir)])
        run = latest_run(task.name) if rc == 0 else None
        if run is not None:
            rc = launch.cmd_run([str(run.dir)])
        best = launch.best_of(run, task) if run is not None else None
        if best is None or not best.get("kernel"):
            print(f"[self-evolve] generation {gen} ended without a confirmed best kernel "
                  f"(loop exit {rc}): the chain stops here", flush=True)
            return 0 if gen > 1 else 1
        gain = best.get("speedup")
        floor = noise_floor(best)
        print(f"[self-evolve] generation {gen}: best {best['sha'][:12]}, "
              + (f"{gain:.4f}x its reference" if gain else "speedup unknown")
              + f" (noise floor {floor:.1%})", flush=True)
        if gain is None or gain <= 1.0 + floor:
            print(f"[self-evolve] gain within the {floor:.1%} noise floor: the chain stops here",
                  flush=True)
            return 0
        if args.max_gen is not None and gen >= args.max_gen:
            print(f"[self-evolve] reached --max-gen {args.max_gen}", flush=True)
            return 0
        task_dir = next_generation(task, best, task_dir.parent / f"{base}-g{gen + 1}",
                                   args.step, run=run)
        print(f"[self-evolve] next generation: {task_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
