#!/usr/bin/env python3
"""Self-evolution: each generation's best kernel becomes the next generation's reference.

    python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]

A generation is an ordinary run (`peregrine new`, then `peregrine run`). When it ends with a confirmed
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

from peregrine_kernel import launch  # noqa: E402
from peregrine_kernel.runs import list_runs  # noqa: E402
from peregrine_kernel.task import (TaskError, dump_task_toml, load_task,  # noqa: E402
                              read_task_toml)

REF_DIR = "reference"
# == peregrine_kernel.loop.parse's noise constants (WP1): below the CAP, two readings of one
# kernel are not told apart, however tight a handful of confirmation jobs happened to
# be. Single source of truth: import the cap (and floor) from parse; this standalone
# script keeps local copies only for the case the import cannot resolve.
try:
    from peregrine_kernel.loop.parse import RESOLVABLE_DIFF_CAP as _RESOLVABLE_CAP  # noqa: E402
except ImportError:  # pragma: no cover - peregrine_kernel not importable (script run standalone)
    _RESOLVABLE_CAP = 0.015   # == peregrine_kernel.loop.parse.RESOLVABLE_DIFF_CAP
NOISE_FLOOR_MIN = _RESOLVABLE_CAP


def latest_run(task_name: str):
    runs = [r for r in list_runs() if r.task_name == task_name]
    return max(runs, key=lambda r: r.record().get("created", "")) if runs else None


def noise_floor(best: dict) -> float:
    """The relative gain a generation must clear: the spread of its same-job ratios
    ((max - min) / min), never below NOISE_FLOOR_MIN.

    Raises ValueError when a ratio is not a positive number: a zero (or negative)
    reading makes the division meaningless, and the chain must stop with one
    readable line instead of a ZeroDivisionError after a whole generation ran.
    """
    ratios = best.get("ratios") or []
    if len(ratios) <= 1:
        return NOISE_FLOOR_MIN
    bad = [r for r in ratios
           if not isinstance(r, (int, float)) or isinstance(r, bool) or r <= 0]
    if bad:
        raise ValueError(
            f"confirmation ratios {ratios!r} contain non-positive reading(s) {bad!r} "
            f"— the noise-floor spread (max-min)/min needs positive ratios; the "
            f"generation's confirming eval JSON is broken")
    return max(NOISE_FLOOR_MIN, (max(ratios) - min(ratios)) / min(ratios))


def _floor_progress_text(task, best: dict) -> str | None:
    """WP16 (C): the ``floor_progress = floor_ms / best_ms`` advisory line for
    one generation — the card's parsed analytic floor (WP9's floor_for, the
    single source; a card without a problem.json yields None and no line is
    printed). Report-only: nothing downstream reads it."""
    try:
        from peregrine_kernel.loop.analytic_floor import floor_from_op_dir
        info = floor_from_op_dir(Path(task.dir), arch="")
    except Exception:  # noqa: BLE001 — an advisory must never break the chain
        return None
    floor_ms = (info or {}).get("floor_ms")
    best_ms = best.get("median_ms")
    if not floor_ms or not best_ms or best_ms <= 0:
        return None
    return (f"floor_progress={floor_ms / best_ms:.3f} (analytic floor {floor_ms:.6g} ms / "
            f"best {best_ms:.6g} ms) — report-only; the generation gate is unchanged")


def next_generation(task, best: dict, dest: Path, step: float, run=None, best_data=None) -> Path:
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
        _write_carry(task, run, dest / "carry", best=best_data)
    child = load_task(dest)              # the new generation must be a valid task
    # M3 (docs/industrial-lane.md §6.7): the lane inherits with the task.toml
    # (read_task_toml -> dump_task_toml round-trips [protocol]); the explicit
    # guard is the regression tripwire — one generation chain is one lane end to
    # end, and a mismatch would be a wiring bug, not a lane change. Fail loudly.
    if child.protocol.memo_lane != task.protocol.memo_lane:
        raise ValueError(
            f"the generation chain changed lanes ({task.dir.name} "
            f"{'memo' if task.protocol.memo_lane else 'strict'} -> {dest.name} "
            f"{'memo' if child.protocol.memo_lane else 'strict'}): a child inherits "
            f"its parent's lane — a mismatch is a bug (docs/industrial-lane.md §6.7)")
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
# WP19 (task 7, B-B8): the carry grows from four artifacts to seven — probes/,
# ref_census.md (the parent best tree's SASS/NCU census, sha-stamped) and
# tools.md. The census is what the next generation's reference-pricing
# reconciliation consumes: the child's reference IS the parent's best kernel
# verbatim, so a census whose header sha matches the child's reference kernel
# byte-for-byte is a valid prior for the child's ref-side readings — adopted
# only after one confirmation job re-measures the ref within the band, never
# on the carry alone (reading reuse is acceleration, not measurement-skipping).
_CARRY_REF_CENSUS = "ref_census.md"
_CARRY_PROBES = "probes"
_CARRY_TOOLS = "tools.md"
# --- WP16: the arch graveyard rides the carry too — the parent's ARCH_PROBE
# losers (candidate, readings, dead reason, tunables). A schema of its own,
# disjoint from any other carry file (the WP19 carry extension adds its own
# artifacts; neither overwrites the other's).
_CARRY_GRAVEYARD = "arch_graveyard.jsonl"
_GRAVEYARD_SRC = "notes/arch_graveyard.jsonl"
_ARCH_PROBE_STAMP_RE = re.compile(r"<!-- ARCH_PROBE .*?-->", re.S)


def _reference_kernel_sha(kernel_src: str) -> str:
    """The census stamp: sha256 of the best kernel's exact source bytes. The
    child recomputes it over its own reference kernel — a match is the
    byte-identity the citation requires."""
    import hashlib
    return hashlib.sha256(kernel_src.encode("utf-8")).hexdigest()


def _best_tree_census_sections(run, best: dict) -> dict:
    """The parent best tree's census sections, from the run's evidence dir and
    its stall-price cache (the best tree's whole pricing log). Best-effort:
    a section the run never produced is absent, never guessed."""
    sections: dict[str, str] = {}
    ev = run.ws / "evidence"
    pairs = (("dossier", "student_orch_dossier_best.txt"),
             ("role", "student_role_exposure.txt"))
    for key, name in pairs:
        p = ev / name
        if p.is_file():
            sections[key] = p.read_text(encoding="utf-8", errors="replace")
    # the best tree's whole pricing log: the cached log whose dossier section
    # is the banked best dossier (same text, one collection)
    dossier = sections.get("dossier", "")
    cache = run.ws / "notes" / "wp11-stall-price"
    if dossier and cache.is_dir():
        import hashlib as _h
        want = _h.sha256(dossier.encode("utf-8")).hexdigest()
        for log_file in sorted(cache.glob("*.log")):
            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if dossier.strip() and dossier.strip() in text:
                sections["pricing_log"] = text
                sections["pricing_log_sha256"] = _h.sha256(
                    text.encode("utf-8")).hexdigest()
                break
    return sections


def _write_carry(task, run, dest: Path, best: dict | None = None) -> None:
    """The parent run's closing record into ``tasks/<op>-g<k>/carry/``:
    the four WP8 artifacts (milestone, lever table, ARCH_PROBE readings,
    falsified rows), the WP19 three (probes/, ref_census.md, tools.md), and
    (WP16) the ARCH_PROBE graveyard — the parent's losing skeletons with
    their readings and dead reasons, so the next generation's teacher never
    re-buys a dead skeleton. Each extension's files are a schema of its own;
    neither overwrites the other's.
    Best-effort by design: an artifact the run never produced (no plateau
    pricing, no probes) is recorded as absent, never guessed."""
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
        # M3 (docs/industrial-lane.md §6.7): the carry's machine-readable lane
        # stamp — the parent's lane as the top-level ``memo_lane`` key. Lever
        # prices are priors over a measurement stream, and across lanes the
        # stream differs; the stamp is what makes the carry auditable (the
        # child's PRIOR-INPUT injection drops a mismatching carry). Unparseable
        # lever text is left as it is: it was never a price table anyway.
        try:
            _levers_obj = json.loads(levers)
        except ValueError:
            _levers_obj = None
        if isinstance(_levers_obj, dict):
            _levers_obj["memo_lane"] = bool(task.protocol.memo_lane)
            levers = json.dumps(_levers_obj, indent=1) + "\n"
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
    # --- WP19 (task 7): probes/ + ref_census.md + tools.md -----------------
    probes_dest = dest / _CARRY_PROBES
    copied = 0
    notes_dir = run.ws / "notes"
    if notes_dir.is_dir():
        probes_dest.mkdir(parents=True, exist_ok=True)
        for probe in sorted(notes_dir.glob("*.py")):
            try:
                shutil.copy2(probe, probes_dest / probe.name)
                copied += 1
            except OSError:
                continue
    notes.append(f"- `{_CARRY_PROBES}/`: the parent's probe scripts "
                 f"({copied} file(s)) — hypothesis-reuse material for the child's "
                 "recon, re-verified line by line")
    census_sections = _best_tree_census_sections(run, best) if best else {}
    census = ""
    if best and best.get("kernel"):
        header = (f"<!-- pk-ref-census\n"
                  f"reference_kernel_sha256: {_reference_kernel_sha(best['kernel'])}\n"
                  f"parent_best_sha: {best.get('sha', '')}\n"
                  f"window: {best.get('window') or ''}\n"
                  f"median_ms: {best.get('median_ms')}\n"
                  f"ratios: {json.dumps(best.get('ratios') or [])}\n"
                  f"pricing_log_sha256: {census_sections.get('pricing_log_sha256', '')}\n"
                  f"-->\n")
        body = []
        for key in ("stall_ledger", "segment_ledger", "dossier", "role", "pm_timeline"):
            if key == "dossier":
                text = census_sections.get("dossier", "")
            elif key == "role":
                text = census_sections.get("role", "")
            elif key == "pm_timeline":
                text = _extract_section(census_sections.get("pricing_log", ""),
                                        r"(?ms)^=== PM-PHASE TIMELINE ===.*?(?=^=== |\Z)")
            else:
                text = _extract_section(census_sections.get("pricing_log", ""),
                                        _CENSUS_SECTION_RES[key])
            body.append(f"## {key}\n\n{text.strip() or '(absent)'}\n")
        census = header + "\n".join(body)
    _put(_CARRY_REF_CENSUS, census,
         "the parent best tree's SASS/NCU census, sha-stamped (the child's "
         "reference is this kernel verbatim): adopted for the child's ref-side "
         "readings only after one confirmation job reconciles within the band")
    (dest / _CARRY_TOOLS).write_text(
        "# carry tools: the harness measurement tools the parent ran\n\n"
        "- stall/segment/dossier/role census: `skills/stall_price_run.py --pm` "
        "(the frozen recipe; the run's `--pm-interval` is the loop's)\n"
        "- phase timeline: the same census collection (`=== PM-PHASE TIMELINE ===`)\n"
        "- timeline probe: `notes/timeline_probe.py` (the harness overlays its own "
        "copy on every probed ref)\n\n"
        "The child's harness re-seeds and overlays its own tool copies; these are "
        "seeds for reading the parent's numbers, never a reason to skip a measurement "
        "the child's own contract asks for.\n", encoding="utf-8")
    notes.append(f"- `{_CARRY_TOOLS}`: the measurement-tool seeds")

    graveyard = ""
    gy_path = run.ws / _GRAVEYARD_SRC
    if gy_path.is_file():
        graveyard = gy_path.read_text(encoding="utf-8", errors="replace")
    _put(_CARRY_GRAVEYARD, graveyard,
         "the parent's ARCH_PROBE losers (candidate, readings, dead reason, "
         "tunables) — hypothesis-reuse material like every carry line")
    (dest / "README.md").write_text(
        "# carry: the parent generation's closing record\n\n"
        "Consumed by the next generation's teacher through the distill/recon briefs' "
        "PRIOR-INPUT section: for hypothesis reuse and reconciliation only — re-verify "
        "every line against the new run's own measurements; never a gate anchor, never "
        "a price source.\n\n" + "\n".join(notes) + "\n", encoding="utf-8")


_CENSUS_SECTION_RES = {
    "stall_ledger": r"(?ms)^=== STALL LEDGER.*?(?=^=== |\Z)",
    "segment_ledger": r"(?ms)^=== SEGMENT.*?(?=^=== |\Z)",
}


def _extract_section(text: str, pattern: str) -> str:
    if not text:
        return ""
    m = re.search(pattern, text)
    return m.group(0) if m else ""


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
    # The chain continues the start task's own numbering: from tasks/rmsnorm-g3 the
    # first jump lands on rmsnorm-g4, not on the ancestor rmsnorm-g2 that the bare
    # generation counter produced (a backwards destination that either already
    # exists or collides with the start task itself).
    start = re.search(r"-g(\d+)$", task_dir.name)
    start_gen = int(start.group(1)) if start else 1
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
        if rc != 0:
            # `peregrine new` / `peregrine run` itself failed — an infrastructure
            # failure (bad machine config, a dead backend, an env problem), not the
            # loop's verdict. It must not read as the natural chain break below (which
            # is a RESULT and exits 0): say so, and hand the operator a distinct code.
            print(f"[self-evolve] generation {gen}: the harness failed "
                  f"(peregrine new/run exit {rc}) — this is not a chain stop; fix the "
                  f"environment and rerun", flush=True)
            return 3
        best = launch.best_of(run, task) if run is not None else None
        if best is None or not best.get("kernel"):
            print(f"[self-evolve] generation {gen} ended without a confirmed best kernel "
                  f"(loop exit {rc}): the chain stops here", flush=True)
            return 0 if gen > 1 else 1
        ratios = best.get("ratios") or []
        if any(not isinstance(r, (int, float)) or r <= 0 for r in ratios):
            print(f"[self-evolve] generation {gen}: best {best['sha'][:12]} but its "
                  f"confirmation ratios hold a non-positive reading ({ratios!r}); "
                  f"the noise floor is unmeasurable, so this generation counts as "
                  f"nothing to optimize: the chain stops here", flush=True)
            return 0
        gain = best.get("speedup")
        try:
            floor = noise_floor(best)
        except ValueError as e:
            sys.stderr.write(f"[self-evolve] generation {gen}: {e}\n")
            return 2
        print(f"[self-evolve] generation {gen}: best {best['sha'][:12]}, "
              + (f"{gain:.4f}x its reference" if gain else "speedup unknown")
              + f" (noise floor {floor:.1%})", flush=True)
        # --- WP16: floor_progress beside the generation gate — the operator
        # card's parsed analytic floor over this generation's own best.
        # Report-only advisory: the chain's stop rule (gain vs the noise floor)
        # and the --step target are untouched, and no external-benchmark
        # scoring concept is involved (both numbers are this run's own).
        if (fp := _floor_progress_text(task, best)) is not None:
            print(f"[self-evolve] generation {gen}: {fp}", flush=True)
        if gain is None or gain <= 1.0 + floor:
            print(f"[self-evolve] gain within the {floor:.1%} noise floor: the chain stops here",
                  flush=True)
            return 0
        if args.max_gen is not None and gen >= args.max_gen:
            print(f"[self-evolve] reached --max-gen {args.max_gen}", flush=True)
            return 0
        try:
            task_dir = next_generation(task, best,
                                       task_dir.parent / f"{base}-g{start_gen + gen}",
                                       args.step, run=run, best_data=best)
        except FileExistsError as e:
            sys.stderr.write(f"[self-evolve] {e}\n")
            return 2
        print(f"[self-evolve] next generation: {task_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
