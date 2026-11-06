#!/usr/bin/env python3
# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0
"""audit_rollup.py -- cross-run aggregation of per-run audit reports into ONE versioned bundle.

Every finished run can carry ``<run_dir>/ws/student/audit.json`` (schema_version 1,
written by ``rsi-kernel-tune audit <run>``): one row per audited verdict, each with
an ``audit_class`` in {survives, noise_level, unconfirmed_edge, mismatch_suspected,
sentinel_flagged, unverifiable}, plus the run's tallies and false-condemnation
flag. The audit paper needs those per-run files aggregated ACROSS runs: this
script walks the given run directories (or a runs-archive root such as
``.artifacts``, whose entries are run directories) and merges every audit.json it
finds into one bundle:

    audit-bundle.json = {
      "schema_version": 1,
      "generated":      <ISO-8601 UTC instant>,
      "runs":           [...one summary per collected run...],
      "groups":         [...(task x profile x seed) rows AND per-profile totals...],
      "totals":         {...the all-runs row, same shape as a group row...},
      "warnings":       [...skipped / rejected / filtered inputs...],
    }

Group rows carry the paper's survival classification: counts AND rates of
keeps_surviving / keeps_noise_level / keeps_unconfirmed / mismatch_suspected /
sentinel_flagged / unverifiable, the false_condemnations total (surfaced
prominently on stdout and in ``totals`` whenever it is nonzero — the audit
claiming a keep was condemned without cause is the worst failure mode the
toolchain has), and gate_degraded_share (the verdict-weighted mean of the
group's per-run shares; null when no run in the group reports one).

Tolerance: a missing or unparseable audit.json skips that run with a warning (a
partial run must not sink the whole rollup), and schema_version != 1 is rejected
with a warning — this tool speaks exactly version 1. But an UNKNOWN audit_class
is a hard error naming the offending file: silent schema drift would poison the
paper's numbers, so the bundle is not written at all in that case.

Usage:
    python scripts/audit_rollup.py .artifacts/run-a .artifacts/run-b --out audit-bundle.json
    python scripts/audit_rollup.py .artifacts --out audit-bundle.json --csv audit-verdicts.csv
    python scripts/audit_rollup.py .artifacts --out bundle.json --filter profile=full,task=<task>

Exit codes: 0 = bundle written; 1 = no usable audit.json collected; 2 = schema
error (unknown audit_class / malformed verdict rows) or CLI misuse.

No GPU, no network: standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
AUDIT_REL = ("ws", "student", "audit.json")
AUDIT_REL_STR = "/".join(AUDIT_REL)
AUDIT_CLASSES = ("survives", "noise_level", "unconfirmed_edge",
                 "mismatch_suspected", "sentinel_flagged", "unverifiable")
KEEP_CLASSES = ("survives", "noise_level", "unconfirmed_edge")
FILTER_KEYS = ("task", "profile", "seed")

# audit_class -> the group/tally count key it feeds (the paper's vocabulary)
_CLASS_COUNT_KEY = {
    "survives": "keeps_surviving",
    "noise_level": "keeps_noise_level",
    "unconfirmed_edge": "keeps_unconfirmed",
    "mismatch_suspected": "mismatch_suspected",
    "sentinel_flagged": "sentinel_flagged",
    "unverifiable": "unverifiable",
}
COUNT_KEYS = ("keeps", "keeps_surviving", "keeps_noise_level", "keeps_unconfirmed",
              "mismatch_suspected", "sentinel_flagged", "unverifiable")
CSV_CORE = ["run_id", "task", "profile", "seed", "verdict_index",
            "audit_class", "false_condemnation"]


class AuditSchemaError(Exception):
    """Unknown audit_class or malformed verdict rows — a hard, named error."""


@dataclass
class RunAudit:
    """One collected run's audit: its identity, its verdict rows, its rollup numbers."""

    run_id: str
    run_dir: str
    audit_file: str
    task: str
    profile: str
    seed: object
    verdicts: list
    false_condemnations: int
    gate_degraded_share: float | None
    tallies: dict

    @property
    def counts(self) -> dict:
        c = dict.fromkeys(COUNT_KEYS, 0)
        for v in self.verdicts:
            cls = v["audit_class"]
            c[_CLASS_COUNT_KEY[cls]] += 1
            if cls in KEEP_CLASSES:
                c["keeps"] += 1
        return c

    def summary(self) -> dict:
        return {"run_id": self.run_id,
                "run_dir": self.run_dir,
                "audit_file": self.audit_file,
                "task": self.task,
                "profile": self.profile,
                "seed": self.seed,
                "total_verdicts": len(self.verdicts),
                "counts": self.counts,
                "false_condemnations": self.false_condemnations,
                "gate_degraded_share": self.gate_degraded_share,
                "tallies": self.tallies}


# --- input discovery -----------------------------------------------------------------

def discover_audit_paths(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """Resolve the positional arguments to concrete audit.json paths.

    An argument that itself has ``ws/student/audit.json`` is one run; any other
    directory is treated as a runs-archive root whose directory entries are run
    dirs (the ``.artifacts`` shape). Missing audit files are skipped with a
    warning; duplicates (an archive root plus one of its own entries) are
    collapsed so no run is counted twice.
    """
    audits: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        key = path.resolve()
        if key in seen:
            warnings.append(f"skipped duplicate input: {path}")
            return
        seen.add(key)
        audits.append(path)

    for raw in inputs:
        base = Path(raw)
        direct = base.joinpath(*AUDIT_REL)
        if direct.is_file():
            add(direct)
            continue
        if not base.is_dir():
            warnings.append(f"skipped: {raw} is not a directory (neither a run dir nor a runs root)")
            continue
        candidates: list[Path] = []
        if (base / "ws").is_dir():
            warnings.append(f"skipped: {base} has ws/ but no {AUDIT_REL_STR} (partial run)")
        else:
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                candidate = child.joinpath(*AUDIT_REL)
                if candidate.is_file():
                    candidates.append(candidate)
                elif (child / "ws").is_dir():
                    warnings.append(f"skipped: {child} has ws/ but no {AUDIT_REL_STR} (partial run)")
            if not candidates:
                warnings.append(f"skipped: no {AUDIT_REL_STR} under {base} or its entries")
        for path in candidates:
            add(path)
    return audits, warnings


# --- loading and validation -----------------------------------------------------------

def load_run(audit_path: Path) -> tuple[RunAudit | None, str | None]:
    """Parse one audit.json -> (run, warning). ``warning`` is None on success;
    a corrupt or wrong-schema file yields (None, reason) and the rollup goes on
    without that run. An unknown audit_class (or malformed verdict rows) raises
    AuditSchemaError — the caller must fail loudly, not swallow it."""
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return None, f"skipped: {audit_path} is not valid JSON ({exc})"
    if not isinstance(data, dict):
        return None, f"skipped: {audit_path} is not a JSON object"
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        return None, (f"rejected: {audit_path} declares schema_version={version!r}; "
                      f"this rollup speaks exactly version {SCHEMA_VERSION}")
    verdicts = data.get("verdicts", [])
    if not isinstance(verdicts, list):
        raise AuditSchemaError(f"{audit_path}: 'verdicts' must be a list, "
                               f"got {type(verdicts).__name__}")
    for i, verdict in enumerate(verdicts):
        if not isinstance(verdict, dict):
            raise AuditSchemaError(f"{audit_path}: verdict #{i} is not an object")
        cls = verdict.get("audit_class")
        if cls not in AUDIT_CLASSES:
            raise AuditSchemaError(
                f"{audit_path}: verdict #{i} has unknown audit_class={cls!r} "
                f"(expected one of {', '.join(AUDIT_CLASSES)})")

    # False condemnations: per-verdict flags preferred; a run-level boolean is
    # the fallback for audits that only carry the run-wide flag.
    if any("false_condemnation" in v for v in verdicts):
        false_condemnations = sum(1 for v in verdicts if v.get("false_condemnation") is True)
    elif data.get("false_condemnation") is True:
        false_condemnations = 1
    else:
        false_condemnations = 0

    tallies = data.get("tallies")
    if not isinstance(tallies, dict):
        tallies = {}
    # gate_degraded_share may sit at the top level or inside tallies (both have
    # been written); a non-numeric value is simply "not reported".
    share = data.get("gate_degraded_share")
    if isinstance(share, bool) or not isinstance(share, (int, float)):
        share = tallies.get("gate_degraded_share")
    if isinstance(share, bool) or not isinstance(share, (int, float)):
        share = None
    run_dir = audit_path.parents[2]
    return RunAudit(run_id=str(data.get("run_id") or run_dir.name),
                    run_dir=str(run_dir),
                    audit_file=str(audit_path),
                    task=str(data.get("task") or "unknown"),
                    profile=str(data.get("profile") or "unknown"),
                    seed=data.get("seed"),
                    verdicts=verdicts,
                    false_condemnations=false_condemnations,
                    gate_degraded_share=None if share is None else round(float(share), 6),
                    tallies=tallies), None


# --- aggregation ----------------------------------------------------------------------

def _aggregate(group_by: str, label: dict, runs: list[RunAudit]) -> dict:
    """One group row: summed counts, rates over the group's verdicts, the
    false-condemnation total, and the verdict-weighted gate_degraded_share."""
    total = sum(len(r.verdicts) for r in runs)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for r in runs:
        for key, n in r.counts.items():
            counts[key] += n
    rates = None if total == 0 else {k: round(counts[k] / total, 6) for k in COUNT_KEYS}
    weighted = weight = 0.0
    for r in runs:
        if r.gate_degraded_share is not None:
            weighted += r.gate_degraded_share * len(r.verdicts)
            weight += len(r.verdicts)
    replay = 0.0
    for r in runs:
        v = r.tallies.get("replay_probe_suspected")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            replay += v
    if float(replay).is_integer():
        replay = int(replay)
    return {"group_by": group_by,
            **label,
            "runs": len(runs),
            "run_ids": sorted(r.run_id for r in runs),
            "total_verdicts": total,
            "counts": counts,
            "rates": rates,
            "false_condemnations": sum(r.false_condemnations for r in runs),
            "gate_degraded_share": None if weight == 0 else round(weighted / weight, 6),
            "replay_probe_suspected": replay}


def build_groups(runs: list[RunAudit]) -> list[dict]:
    """The paper's two group-bys: (task x profile x seed) rows, then per-profile totals."""
    by_seed: dict[tuple, list[RunAudit]] = {}
    by_profile: dict[str, list[RunAudit]] = {}
    for r in runs:
        by_seed.setdefault((r.task, r.profile, str(r.seed)), []).append(r)
        by_profile.setdefault(r.profile, []).append(r)
    groups = [_aggregate("task_x_profile_x_seed",
                         {"task": rs[0].task, "profile": rs[0].profile, "seed": rs[0].seed}, rs)
              for _, rs in sorted(by_seed.items())]
    groups += [_aggregate("profile", {"task": None, "profile": profile, "seed": None}, rs)
               for profile, rs in sorted(by_profile.items())]
    return groups


def build_bundle(runs: list[RunAudit], warnings: list[str]) -> dict:
    totals = _aggregate("all", {"task": None, "profile": None, "seed": None}, runs)
    return {"schema_version": SCHEMA_VERSION,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": [r.summary() for r in sorted(runs, key=lambda r: r.run_id)],
            "groups": build_groups(runs),
            "totals": totals,
            "warnings": list(warnings)}


# --- CSV ------------------------------------------------------------------------------

def write_csv(runs: list[RunAudit], csv_path: Path) -> int:
    """One row per verdict; core columns first, then every extra verdict field
    any run carries (blank where absent). Returns the data-row count."""
    extra: set[str] = set()
    for r in runs:
        for v in r.verdicts:
            extra.update(k for k in v if k not in CSV_CORE)
    fieldnames = CSV_CORE + sorted(extra)
    ordered = sorted(runs, key=lambda r: r.run_id)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in ordered:
            for i, v in enumerate(r.verdicts):
                row = {k: v.get(k, "") for k in fieldnames}
                row.update(run_id=r.run_id, task=r.task, profile=r.profile,
                           seed="" if r.seed is None else r.seed,
                           verdict_index=i,
                           audit_class=v.get("audit_class"),
                           false_condemnation=v.get("false_condemnation", ""))
                writer.writerow(row)
    return sum(len(r.verdicts) for r in ordered)


# --- report ---------------------------------------------------------------------------

_HDR = (f"{'task':<14} {'profile':<10} {'seed':<8} {'runs':>4} {'verdicts':>8} "
        f"{'keeps':>6} {'surv':>5} {'noise':>6} {'unconf':>6} {'mism':>5} "
        f"{'sent':>5} {'unver':>5} {'fcond':>5} {'gate_deg':>8}")


def _cell(x) -> str:
    return "-" if x is None else str(x)


def _print_table(rows: list[dict]) -> None:
    print(_HDR)
    print("-" * len(_HDR))
    for g in rows:
        c = g["counts"]
        gate = "-" if g["gate_degraded_share"] is None else f"{g['gate_degraded_share']:.3f}"
        print(f"{_cell(g['task']):<14} {_cell(g['profile']):<10} {_cell(g['seed']):<8} "
              f"{g['runs']:>4} {g['total_verdicts']:>8} {c['keeps']:>6} "
              f"{c['keeps_surviving']:>5} {c['keeps_noise_level']:>6} {c['keeps_unconfirmed']:>6} "
              f"{c['mismatch_suspected']:>5} {c['sentinel_flagged']:>5} {c['unverifiable']:>5} "
              f"{g['false_condemnations']:>5} {gate:>8}")


def print_report(bundle: dict, skipped: int, csv_path: Path | None) -> None:
    print(f"[audit-rollup] bundle schema_version={bundle['schema_version']}, "
          f"generated {bundle['generated']}")
    print(f"[audit-rollup] runs: {len(bundle['runs'])} collected, {skipped} "
          f"skipped/rejected/filtered; {bundle['totals']['total_verdicts']} verdicts")
    for label, selector in (("task x profile x seed", "task_x_profile_x_seed"), ("profile", "profile")):
        rows = [g for g in bundle["groups"] if g["group_by"] == selector]
        print(f"[audit-rollup] groups ({label}):")
        _print_table(rows)
    total_fc = bundle["totals"]["false_condemnations"]
    if total_fc:
        print(f"[audit-rollup] FALSE CONDEMNATIONS: {total_fc} -- the audit claims keeps were "
              f"condemned without cause; see groups[].false_condemnations")
    if bundle["warnings"]:
        print(f"[audit-rollup] warnings ({len(bundle['warnings'])}):")
        for w in bundle["warnings"]:
            print(f"  - {w}")
    if csv_path is not None:
        print(f"[audit-rollup] csv -> {csv_path}")


# --- CLI ------------------------------------------------------------------------------

def parse_filter(spec: str | None) -> dict[str, str]:
    """``task=...,profile=...`` -> dict; unknown keys are a CLI error."""
    if not spec:
        return {}
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or not key or key not in FILTER_KEYS:
            raise ValueError(f"filter entries must be k=v with k in "
                             f"{', '.join(FILTER_KEYS)}, got {part!r}")
        out[key] = value.strip()
    return out


def run_matches(run: RunAudit, filt: dict[str, str]) -> bool:
    for key, want in filt.items():
        have = getattr(run, key)
        if have is None or str(have) != want:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="audit_rollup.py",
        description="Merge every run's ws/student/audit.json into one versioned "
                    "audit-bundle.json with (task x profile x seed) and per-profile "
                    "survival-classification groups, plus an optional per-verdict CSV.")
    p.add_argument("run_dirs", nargs="+", metavar="RUN_DIR",
                   help="run directories, or a runs-archive root whose entries are "
                        "run dirs (e.g. .artifacts)")
    p.add_argument("--out", type=Path, required=True, help="bundle path (audit-bundle.json)")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional per-verdict CSV path (one row per verdict)")
    p.add_argument("--filter", default=None, metavar="K=V,...",
                   help=f"only roll up runs matching all of: {', '.join(FILTER_KEYS)}")
    args = p.parse_args(argv)

    try:
        filt = parse_filter(args.filter)
    except ValueError as exc:
        p.error(str(exc))

    warnings: list[str] = []
    audit_paths, discovery_warnings = discover_audit_paths(args.run_dirs)
    warnings.extend(discovery_warnings)

    runs: list[RunAudit] = []
    try:
        for path in audit_paths:
            run, warning = load_run(path)
            if warning:
                warnings.append(warning)
            if run is not None:
                runs.append(run)
    except AuditSchemaError as exc:
        print(f"[audit-rollup] SCHEMA ERROR: {exc}", file=sys.stderr)
        return 2

    if filt:
        kept = [r for r in runs if run_matches(r, filt)]
        why = ", ".join(f"{k}={v}" for k, v in sorted(filt.items()))
        for r in runs:
            if not run_matches(r, filt):
                warnings.append(f"filtered out: run {r.run_id} does not match --filter ({why})")
        runs = kept

    if not runs:
        print("[audit-rollup] no usable audit.json collected -- nothing to roll up",
              file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        return 1

    bundle = build_bundle(runs, warnings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    csv_path: Path | None = None
    if args.csv is not None:
        csv_path = args.csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(runs, csv_path)

    print_report(bundle, skipped=len(audit_paths) - len(bundle["runs"]), csv_path=csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
