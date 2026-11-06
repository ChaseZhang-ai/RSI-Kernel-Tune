#!/usr/bin/env python3
# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Arm A's corpus re-timer (docs/paper/experiment-matrix.md §3): plan the re-timing of the
released kernels in ``kernels/`` and collect the finished re-timing outputs into rows that
match the shared audit-bundle schema (``rsi-kernel-tune audit`` writes a finished run's report
at ``<run>/operators/<task>/student/audit.json``, ``schema_version = 1``;
``scripts/audit_rollup.py`` merges the per-run reports per (task x profile x seed)).

Two modes:

* ``--plan`` (offline, no GPU): read ``kernels/README.md``'s provenance tables and one kernel
  dir each, and emit one plan entry per B200-compatible kernel — its claimed speedup band,
  the reference implementation it was claimed against, the environment the claim was published
  under, and the job plan (scaffold → run → install released bytes → floor-calibration repeats
  → timing jobs per profile × seed) with the fields a later comparison needs.
* ``--collect`` (offline, no GPU): parse the finished re-timing outputs (the per-job JSON files
  the audit lane writes into the re-time root, format below), compute each row's measured noise
  floor with the framework's ``resolvable_diff`` arithmetic, and classify the verdict into the
  shared ``audit_class`` taxonomy with the pre-registered decision rules of
  ``docs/paper/experiment-matrix.md`` §4.

Re-time root format (what ``--collect`` reads). Every ``*.json`` under ``--retime-root`` whose
body carries ``task`` / ``seed`` / ``phase``:

* a floor-calibration file — ``"phase": "floor"`` —
  ``{"task", "seed", "phase": "floor", "ratios": [<ref/kernel per repeat job>]}``
  (R repeats of the unchanged tree; the ratios are paired same-job ref/kernel, so node speed
  cancels);
* one timing file per (profile, seed) — ``"phase": "time"`` —
  ``{"task", "seed", "phase": "time", "profile": "full" | "upstream_style", "ok": true,
    "paired_ratio": <ref/kernel>, "sentinel": null, "clock_drift": false,
    "fresh_recheck": "pass", "gpu_state": {...}}``
  or ``{"ok": false, "why": "build_failed|missing_dep|arch_mismatch|harness_refused", ...}``.

Design rules:

* **Offline and stdlib-only.** Human-run, no network I/O, and it imports nothing from
  ``src/rsi_kernel_tune`` (the same isolation discipline as ``scripts/eval-export/pack.py``):
  the plan is produced and the outputs are parsed long after the GPU jobs are done, and the
  unit tests run with synthetic fixtures only.
* **The claimed numbers are upstream's**, quoted verbatim from ``kernels/README.md``; they
  enter the rows as the claim being audited, never as this framework's own measurements
  (the anti-fabrication preamble of ``docs/paper/README.md``).
* **Classification never condemns on weak evidence**: the only condemning class requires a
  resolvable sign flip under a clean stack (see ``classify`` below).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = 1
PRODUCER = "scripts/retime_corpus.py"

# The resolvable_diff family's shared constants (src/rsi_kernel_tune/loop/parse.py):
# clamp(3 sigma, floor, cap), sigma = 1.4826 x MAD, minimum clean ratio samples below which
# the estimate returns the cap (the data-poor fallback, never a separate mode).
RESOLVABLE_FLOOR = 0.005
RESOLVABLE_CAP = 0.015
RESOLVABLE_MIN_SAMPLES = 4
MAD_TO_SIGMA = 1.4826

# seed set and floor-repeat count the plan fixes (pre-declared; experiment-matrix.md §3/§6)
SEEDS = (0, 1, 2)
FLOOR_REPEATS = 6

ARCHES = {"h100": "sm_90a", "b200": "sm_100a"}
# the honest stack (profile "full") and the released-corpus control condition
PROFILES = ("full", "upstream_style")

# what the claim was measured against, per kernels/README.md "Environment"
UPSTREAM_METRIC = (
    "device time, median over pipelined back-to-back launches "
    "(upstream harness; kernels/README.md Environment) — a cited upstream result"
)

_AUDIT_LANE = (
    "rsi-kernel-tune audit <RUN>   # the audit lane (schema_version 1 rows; lands with the "
    "audit bundle)"
)


# --- claim parsing (kernels/README.md provenance tables) -------------------------------

_RATIO_RANGE_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*[–\-—]\s*(?P<hi>\d+(?:\.\d+)?)\s*[×x]")
_RATIO_RE = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*[×x]")
_PCT_RE = re.compile(r"(?P<sign>[+−–—-]?)(?P<v>\d+(?:\.\d+)?)\s*%")
_PAREN_RE = re.compile(r"\((?P<note>[^()]*)\)")


def parse_claim(text: str) -> dict:
    """One ``Speedup`` cell → ``{"text", "pct_low", "pct_high", "note"}`` (percent form;
    positive = the released kernel faster). Handles the forms the tables actually use:
    ``+2.5%``, ``−0.2%`` (U+2212 and ASCII minus), ``1.104–1.105×`` (a ratio range, en dash),
    ``+9.9% (+10% vs FlashAttention-4)`` (the parenthetical is the cumulative-vs-expert note,
    kept verbatim — it is not the audited margin)."""
    claim: dict = {"text": text, "pct_low": None, "pct_high": None, "note": None}
    if m := _PAREN_RE.search(text):
        claim["note"] = m.group("note").strip()

    if m := _RATIO_RANGE_RE.search(text):
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        claim["pct_low"] = (lo - 1.0) * 100.0
        claim["pct_high"] = (hi - 1.0) * 100.0
        return claim
    if m := _RATIO_RE.search(text):
        v = (float(m.group("v")) - 1.0) * 100.0
        claim["pct_low"] = claim["pct_high"] = v
        return claim
    if m := _PCT_RE.search(text):
        v = float(m.group("v"))
        if m.group("sign") in ("−", "-", "–", "—"):
            v = -v
        claim["pct_low"] = claim["pct_high"] = v
        return claim
    raise ValueError(f"unparseable speedup text: {text!r}")


_TABLE_ROW_RE = re.compile(r"^\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*$")
_DIR_LINK_RE = re.compile(r"\[`(?P<text>[^`]+)`\]\((?P<target>[^)]+)\)")


def _clean_directory_cell(cell: str) -> str:
    """The Directory column holds either a plain backticked path or a markdown link
    (``[`path`](path/)``); the link target (or the backticked text) is the path."""
    m = _DIR_LINK_RE.search(cell.strip())
    raw = (m.group("target") if m else cell).strip().strip("`")
    return raw.rstrip("/")


def parse_readme_provenance(readme_path: Path) -> list[dict]:
    """The provenance rows of kernels/README.md: every table row with the six columns
    (Operator | GPU | Compared against | Speedup | Source | Directory). Heading/separator
    rows and the non-provenance tables (Generations, Environment) cannot match the six-cell
    shape with a parsable ``Speedup`` cell and are skipped."""
    rows: list[dict] = []
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if m is None:
            continue
        op, gpu, baseline, speedup, source, directory = (c.strip() for c in m.groups())
        directory = _clean_directory_cell(directory)
        if not directory or directory.startswith("-"):
            continue  # the header separator row
        try:
            claim = parse_claim(speedup)
        except ValueError:
            continue  # not a provenance row (a speedup cell is what defines one)
        rows.append({
            "operator": op,
            "gpu": gpu,
            "baseline": baseline,
            "source": source,
            "directory": directory,
            "claim": claim,
        })
    return rows


def _split_kernel_dir(directory: str) -> tuple[str, str, str, str] | None:
    """`<family>/<op>/<arch>[/<gen>]` → (family, op, arch, tail) or None when the path does
    not name a known arch segment; the generation leaf (gen0/gen1/gen2/from_agent) is the
    tail and becomes part of the task name, so two generations of one operator never
    collide into one task."""
    parts = [p for p in directory.split("/") if p]
    for i, p in enumerate(parts):
        if p in ARCHES:
            return parts[0], "/".join(parts[1:i]), p, "_".join(parts[i + 1:])
    return None


def _sanitize_op(name: str) -> str:
    """An operator/task name the task loader will accept: letters, digits, underscore."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return cleaned or "op"


def _detect_language(kernel_dir: Path) -> tuple[str, list[str]]:
    """`cuda-cpp` when CUDA sources sit beside kernel.py (the DSA post-paper kernel), else
    `cutedsl` (the paper kernels' CuTe DSL). Returns (language, extra_source_names)."""
    extra = sorted(p.name for p in kernel_dir.glob("*")
                   if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".h", ".cuh.hpp"))
    return ("cuda-cpp" if extra else "cutedsl"), extra


# --- plan mode -------------------------------------------------------------------------

def build_plan(kernels_dir: Path, *, arch: str, seeds: list[int], floor_repeats: int) -> dict:
    readme = kernels_dir / "README.md"
    if not readme.is_file():
        raise SystemExit(f"retime-corpus: no provenance README at {readme}")
    entries: list[dict] = []
    for row in parse_readme_provenance(readme):
        split = _split_kernel_dir(row["directory"])
        if split is None:
            continue
        family, op, row_arch, tail = split
        if arch != "all" and row_arch != arch:
            continue
        # the README's Directory column is relative to kernels/ itself
        kernel_dir = kernels_dir / row["directory"]
        kernel_py = kernel_dir / "kernel.py"
        if not kernel_py.is_file():
            print(f"retime-corpus: WARNING {row['directory']}: no kernel.py, skipping",
                  file=sys.stderr)
            continue
        language, extra_sources = _detect_language(kernel_dir)
        task = _sanitize_op("_".join(x for x in (op, tail, row_arch) if x))
        entry = {
            "task": task,
            "operator": op,
            "family": family,
            "arch": row_arch,
            "generation": tail or None,
            "sm": ARCHES[row_arch],
            "kernel_dir": f"kernels/{row['directory']}",
            "language": language,
            "sources": ["kernel.py", *extra_sources],
            "entry_point": "kernel.py::run",
            "claim": {
                "baseline": row["baseline"],
                "source": row["source"],
                "metric": UPSTREAM_METRIC,
                "pct_low": row["claim"]["pct_low"],
                "pct_high": row["claim"]["pct_high"],
                "text": row["claim"]["text"],
                "note": row["claim"]["note"],
            },
            "commands": {
                "scaffold": (
                    f"rsi-kernel-tune init {op} --from-kernels --arch {row_arch} "
                    "--axes '<json object from kernel.py's frozen contract>'"
                    if language == "cutedsl" else
                    "UNSUPPORTED-YET by `init --from-kernels` (launch.py refuses a CUDA C++ "
                    "published kernel as a sealed source reference; the binref path is a P2 "
                    f"backlog item) — the audit lane re-times kernels/{row['directory']} "
                    "directly (the .cu JIT-compiles under TORCH_CUDA_ARCH_LIST=10.0a)"),
                "new": f"rsi-kernel-tune new tasks/{task}",
                "install": (f"cp {row['directory']}/kernel.py <RUN>/ws/kernel.py"
                            "  # the card's student file (eval/problem/problem.json "
                            "kernel_file, default kernel.py)"
                            + ("  # + the CUDA sources beside it (cuda-cpp kernel)"
                               if language == "cuda-cpp" else "")),
                "floor": (f"R = {floor_repeats} unchanged-tree repeat bench jobs of <RUN>, "
                          f"profile=full — the audit lane ({_AUDIT_LANE}) executes them; "
                          "each writes one floor JSON (phase=floor) with its paired ratio"),
                "time": ("one timing job per (profile in {full, upstream_style} × seed in "
                         f"{seeds}) of <RUN>, profile=<p>, seed=<s> — the audit lane executes "
                         "them; each writes one timing JSON (phase=time)"),
                "audit": _AUDIT_LANE,
                "collect": ("python3 scripts/retime_corpus.py --collect "
                            "--retime-root <RETIME_ROOT> --plan-file <THIS_PLAN>"),
            },
        }
        entries.append(entry)
        entries.sort(key=lambda e: e["kernel_dir"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "retime-plan",
        "producer": PRODUCER,
        "kernels_dir": str(kernels_dir),
        "arch": arch,
        "seeds": seeds,
        "floor_repeats": floor_repeats,
        "floor": {"min": RESOLVABLE_FLOOR, "cap": RESOLVABLE_CAP},
        "upstream_env_citation": (
            "kernels/README.md Environment: B200 (sm_100a) — CUDA 13.0, PyTorch 2.13, "
            "nvidia-cutlass-dsl 4.6.0 or 4.6.2; reproduce before any Arm A job "
            "(docs/paper/repro.md §1)"),
        "kernels": entries,
    }


# --- the floor arithmetic (resolvable_diff's between-job pool, outside the loop) --------

def _clean_ratio(v) -> float | None:
    """A usable paired ratio is a finite float > 0 (the estimator's own entry hygiene: a
    NaN/Infinity JSON literal is a missing reading, never a sample)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v > 0.0 else None


def resolvable_floor(ratios: list, *, floor: float = RESOLVABLE_FLOOR,
                     cap: float = RESOLVABLE_CAP) -> tuple[float, int]:
    """The relative difference these re-timing measurements can resolve: the between-job
    pool of ``loop/parse.py resolvable_diff`` computed outside the loop — leave-one-out
    residuals of the job-level paired ratios (no datum pins its own baseline), sigma =
    MAD x 1.4826, ``clamp(3 sigma, floor, cap)``, the cap when fewer than
    RESOLVABLE_MIN_SAMPLES clean ratios exist. Returns (floor_fraction, n_clean)."""
    clean = [v for v in (_clean_ratio(r) for r in ratios) if v is not None]
    if len(clean) < RESOLVABLE_MIN_SAMPLES:
        return cap, len(clean)
    residuals = [statistics.median(clean[:i] + clean[i + 1:]) - v
                 for i, v in enumerate(clean) if len(clean) > 1]
    mad = statistics.median(abs(v - statistics.median(residuals)) for v in residuals)
    sigma = MAD_TO_SIGMA * mad
    return min(cap, max(floor, 3.0 * sigma)), len(clean)


# --- collect mode: the classification (experiment-matrix.md §4, pre-registered) ---------

def classify(claim_pct_low: float, claim_pct_high: float, measured_ratio: float | None,
             floor_frac: float, *, sentinel: str | None, clock_drift: bool,
             fresh_recheck: str) -> tuple[str, str]:
    """The shared decision rules, in precedence order (first match wins). Speedups are
    handled in percent here (the rows' currency); ``floor_frac`` is a fraction."""
    f_pct = floor_frac * 100.0
    c_lo, c_hi = claim_pct_low, claim_pct_high
    m = None if measured_ratio is None else (measured_ratio - 1.0) * 100.0

    if sentinel:
        return "sentinel_flagged", f"measurement-side sentinel tripped: {sentinel}"
    if fresh_recheck == "fail":
        return "sentinel_flagged", "the fresh re-check alarmed (judge); the reading is void"
    if measured_ratio is None:
        return "unverifiable", "no clean paired ratio (no usable measurement)"
    if clock_drift:
        return "unverifiable", ("clock-drift guard flagged the reading (not one measurement "
                                "condition); a flagged reading produces no verdict")
    if m + f_pct < 0.0:
        return "mismatch_suspected", (
            f"measured {m:+.2f}% resolves the opposite sign beyond the floor "
            f"(f = {f_pct:.2f}%) on a clean stack")
    if c_hi < f_pct or abs(m) < f_pct:
        return "noise_level", (
            f"the claim (max {c_hi:+.2f}%) sits under the floor (f = {f_pct:.2f}%) or the "
            f"reading straddles parity (m = {m:+.2f}%): indistinguishable from noise")
    interval = (m - f_pct, m + f_pct)
    if interval[1] < c_lo or interval[0] > c_hi:
        return "unconfirmed_edge", (
            f"sign replicates (m = {m:+.2f}% ≥ f = {f_pct:.2f}%) but the resolution "
            f"interval [{interval[0]:+.2f}%, {interval[1]:+.2f}%] misses the claimed band "
            f"[{c_lo:+.2f}%, {c_hi:+.2f}%]")
    return "survives", (
        f"measured {m:+.2f}% with floor {f_pct:.2f}% intersects the claimed band "
        f"[{c_lo:+.2f}%, {c_hi:+.2f}%]")


def collect(plan: dict, retime_root: Path) -> list[dict]:
    """Walk the re-time root, pair every timing file with its task's floor ratios, and emit
    shared-schema audit rows (one per task x profile x seed). Multiple timing files for one
    (task, profile, seed) are refused: ambiguous evidence is a caller bug, not a row."""
    claims = {e["task"]: e for e in plan["kernels"]}
    floors: dict[tuple, list] = {}
    times: dict[tuple, dict] = {}
    for path in sorted(retime_root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SystemExit(f"retime-corpus: {path} is not valid JSON: {exc}") from exc
        if not all(k in doc for k in ("task", "seed", "phase")):
            continue
        key = (doc["task"], int(doc["seed"]))
        if doc["phase"] == "floor":
            floors.setdefault(key, []).extend(doc.get("ratios") or [])
        elif doc["phase"] == "time":
            tkey = key + (str(doc.get("profile")),)
            if tkey in times:
                raise SystemExit(
                    f"retime-corpus: two timing files for task={tkey[0]} "
                    f"seed={tkey[1]} profile={tkey[2]} ({path} and {times[tkey]['_path']})")
            doc["_path"] = str(path)
            times[tkey] = doc
        else:
            raise SystemExit(f"retime-corpus: {path} has unknown phase {doc['phase']!r}")

    rows: list[dict] = []
    for (task, seed, profile), doc in sorted(times.items()):
        entry = claims.get(task)
        if entry is None:
            raise SystemExit(
                f"retime-corpus: timing file for unknown task {task!r} ({doc['_path']}); "
                "collect against the plan that produced the jobs")
        floor_frac, n_floor = resolvable_floor(floors.get((task, seed), []),
                                               floor=plan["floor"]["min"],
                                               cap=plan["floor"]["cap"])
        measured_ratio = _clean_ratio(doc.get("paired_ratio"))
        sentinel = doc.get("sentinel")
        sentinel = str(sentinel) if sentinel else None
        clock_drift = bool(doc.get("clock_drift", False))
        fresh = str(doc.get("fresh_recheck", "absent"))
        if not doc.get("ok", False):
            audit_class, reason = "unverifiable", f"the timing job failed: {doc.get('why', '?')}"
        else:
            audit_class, reason = classify(
                entry["claim"]["pct_low"], entry["claim"]["pct_high"], measured_ratio,
                floor_frac, sentinel=sentinel, clock_drift=clock_drift, fresh_recheck=fresh)
        m_pct = None if measured_ratio is None else (measured_ratio - 1.0) * 100.0
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "arm": "A",
            "task": task,
            "profile": profile,
            "seed": seed,
            "kernel_dir": entry["kernel_dir"],
            "language": entry["language"],
            "sm": entry["sm"],
            "claim": {
                "baseline": entry["claim"]["baseline"],
                "source": entry["claim"]["source"],
                "metric": entry["claim"]["metric"],
                "pct_low": entry["claim"]["pct_low"],
                "pct_high": entry["claim"]["pct_high"],
                "text": entry["claim"]["text"],
                "note": entry["claim"]["note"],
                "provenance": "kernels/README.md (a cited upstream result)",
            },
            "measured": {
                "paired_ratio": measured_ratio,
                "speedup_pct": m_pct,
                "floor_pct": floor_frac * 100.0,
                "interval_pct": None if m_pct is None else
                [m_pct - floor_frac * 100.0, m_pct + floor_frac * 100.0],
                "n_floor_ratios": n_floor,
                "sentinel": sentinel,
                "clock_drift": clock_drift,
                "fresh_recheck": fresh,
                "gpu_state": doc.get("gpu_state"),
                "timing_file": doc["_path"],
            },
            "audit_class": audit_class,
            "reason": reason,
        })
    return rows


# --- CLI -------------------------------------------------------------------------------

def _seeds_arg(text: str) -> list[int]:
    try:
        seeds = [int(s) for s in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--seeds: {exc}") from exc
    if not seeds or any(s < 0 for s in seeds):
        raise argparse.ArgumentTypeError("--seeds: non-negative integers, comma-separated")
    return seeds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plan the released-corpus re-timing (Arm A) and collect it into "
                    "audit-bundle rows (offline; no GPU, no network).")
    p.add_argument("--kernels-dir", default="kernels", help="the released kernels/ directory")
    p.add_argument("--arch", default="b200", choices=["b200", "h100", "all"],
                   help="plan only kernels targeted at this arch (default b200)")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS), type=_seeds_arg,
                   help="comma-separated seed set the plan declares (default 0,1,2)")
    p.add_argument("--floor-repeats", default=FLOOR_REPEATS, type=int,
                   help="floor-calibration repeat jobs per (kernel, seed) "
                        "(default %d)" % FLOOR_REPEATS)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true",
                      help="print the re-timing job plan (offline-testable)")
    mode.add_argument("--collect", action="store_true",
                      help="parse finished re-timing outputs into audit rows")
    p.add_argument("--plan-file", default=None,
                   help="--collect: the plan JSON produced by --plan "
                        "(default: re-derive it from --kernels-dir)")
    p.add_argument("--retime-root", default=None,
                   help="--collect: the directory of finished re-timing outputs")
    p.add_argument("--out", default="-", help="write the JSON here ('-' = stdout)")
    args = p.parse_args(argv)

    if args.plan:
        plan = build_plan(Path(args.kernels_dir), arch=args.arch, seeds=args.seeds,
                          floor_repeats=args.floor_repeats)
        payload = plan
    else:
        if not args.retime_root:
            p.error("--collect requires --retime-root")
        if args.plan_file:
            plan = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
        else:
            plan = build_plan(Path(args.kernels_dir), arch="all", seeds=args.seeds,
                              floor_repeats=args.floor_repeats)
        rows = collect(plan, Path(args.retime_root))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["audit_class"]] = counts.get(row["audit_class"], 0) + 1
        payload = {"schema_version": SCHEMA_VERSION, "kind": "retime-rows",
                   "producer": PRODUCER, "retime_root": args.retime_root,
                   "counts": counts, "rows": rows}

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"retime-corpus: wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
