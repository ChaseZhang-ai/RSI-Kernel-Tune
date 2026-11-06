# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for ``scripts/audit_rollup.py`` — the audit paper's cross-run rollup.

The rollup merges every run's ``ws/student/audit.json`` (schema_version 1, one
``audit_class`` per verdict) into one versioned bundle with (task x profile x
seed) and per-profile survival-classification groups. This suite pins, on
synthetic run trees only (no GPU, no real runs):

1. **Bundle structure and group arithmetic** — the six audit classes land in the
   right counts and rates for the (task x profile x seed) groups, the per-profile
   totals, and the all-runs totals; false condemnations surface (here one
   deliberate false condemnation in the noise-heavy baseline run).
2. **Tolerance** — a corrupt audit.json and a run whose audit.json is missing
   are skipped with named warnings while the rest of the rollup proceeds; a
   schema_version != 1 file is rejected with a warning.
3. **The hard error** — an unknown audit_class fails the whole rollup with exit
   2 and names the offending file, writing no bundle.
4. **CSV** — exactly one data row per collected verdict, filterable.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_rollup.py"


def _load():
    spec = importlib.util.spec_from_file_location("rkt_audit_rollup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # py3.14 dataclass annotations resolve via sys.modules
    spec.loader.exec_module(mod)
    return mod


# --- synthetic audit fixtures ---------------------------------------------------------

def _v(cls: str, fc: bool = False) -> dict:
    return {"verdict": f"keep-lever", "audit_class": cls,
            "false_condemnation": fc, "gate": "pass"}


_KEEP_CLASSES = {"survives", "noise_level", "unconfirmed_edge"}


_KEEP_CLASSES = ("survives", "noise_level", "unconfirmed_edge")


def _audit(run_id: str, task: str, profile: str, seed, verdicts: list[dict],
           tally_over: dict | None = None, **over) -> dict:
    """A schema_version-1 audit.json exactly as ``rsi-kernel-tune audit`` writes it,
    tallies in the schema's vocabulary (keeps_*, sentinel_trips, ...)."""
    counts = Counter(v["audit_class"] for v in verdicts)
    fc = sum(1 for v in verdicts if v.get("false_condemnation") is True)
    tallies = {"total_verdicts": len(verdicts),
               "keeps": sum(counts.get(c, 0) for c in _KEEP_CLASSES),
               "keeps_surviving": counts.get("survives", 0),
               "keeps_noise_level": counts.get("noise_level", 0),
               "keeps_unconfirmed": counts.get("unconfirmed_edge", 0),
               "sentinel_trips": counts.get("sentinel_flagged", 0),
               "replay_probe_suspected": 0,
               "false_condemnations": fc}
    tallies.update(tally_over or {})
    data = {"schema_version": 1, "run_id": run_id, "task": task, "profile": profile,
            "seed": seed, "verdicts": verdicts, "tallies": tallies,
            "false_condemnation": fc > 0}
    data.update(over)
    return data


def _write_run(root: Path, name: str, audit: dict | None = None,
               raw: str | None = None, partial: bool = False) -> Path:
    student = root / name / "ws" / "student"
    student.mkdir(parents=True)
    if audit is not None:
        (student / "audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    elif raw is not None:
        (student / "audit.json").write_text(raw, encoding="utf-8")
    elif not partial:
        student.rmdir()  # no ws/ at all: the dir is not recognizably a run
    return root / name


def _noise_heavy(run_id: str) -> dict:
    """Baseline profile: 6 noise_level / 2 survives / 1 unconfirmed_edge /
    1 mismatch_suspected; one noise verdict carries a FALSE condemnation."""
    verdicts = ([_v("noise_level")] + [_v("noise_level", fc=True)]
                + [_v("noise_level")] * 4 + [_v("survives")] * 2
                + [_v("unconfirmed_edge"), _v("mismatch_suspected")])
    return _audit(run_id, "task-t1", "baseline", 42, verdicts,
                  gate_degraded_share=0.2)


def _survives_heavy(run_id: str) -> dict:
    """Full profile: 7 survives / 1 noise_level / 1 sentinel_flagged / 1 unverifiable."""
    verdicts = ([_v("survives")] * 7 + [_v("noise_level"),
                _v("sentinel_flagged"), _v("unverifiable")])
    return _audit(run_id, "task-t1", "full", 42, verdicts,
                  tally_over={"replay_probe_suspected": 1},
                  gate_degraded_share=0.05)


def _small_full(run_id: str) -> dict:
    """Full profile, different task/seed: 2 survives / 1 unconfirmed_edge; no
    tallies, no gate share, verdicts without per-verdict flags (the run-level
    false_condemnation fallback path)."""
    return _audit("run-c", "task-t2", "full", 7,
                  [_v("survives")] * 2 + [_v("unconfirmed_edge")],
                  false_condemnation=False)


def _three_runs(root: Path) -> list[Path]:
    return [_write_run(root, "run-a", audit=_noise_heavy("run-a")),
            _write_run(root, "run-b", audit=_survives_heavy("run-b")),
            _write_run(root, "run-c", audit=_small_full("run-c"))]


# --- 1: bundle structure and group arithmetic -----------------------------------------

def test_bundle_structure_and_group_rates(tmp_path) -> None:
    mod = _load()
    out = tmp_path / "audit-bundle.json"
    runs = _three_runs(tmp_path)
    rc = mod.main([str(p) for p in runs] + ["--out", str(out)])
    assert rc == 0
    bundle = json.loads(out.read_text(encoding="utf-8"))

    assert bundle["schema_version"] == 1
    datetime.fromisoformat(bundle["generated"])  # an ISO-8601 instant
    assert bundle["warnings"] == []

    # per-run summaries
    assert {r["run_id"] for r in bundle["runs"]} == {"run-a", "run-b", "run-c"}
    run_a = next(r for r in bundle["runs"] if r["run_id"] == "run-a")
    assert run_a["profile"] == "baseline" and run_a["seed"] == 42
    assert run_a["total_verdicts"] == 10
    assert run_a["counts"]["keeps"] == 9
    assert run_a["counts"]["keeps_noise_level"] == 6
    assert run_a["false_condemnations"] == 1
    assert run_a["gate_degraded_share"] == pytest.approx(0.2)
    assert run_a["tallies"]["keeps_surviving"] == 2  # file tallies pass through

    # (task x profile x seed) groups: exactly the three fixtures
    seed_groups = {("task-t1", "baseline", 42): "run-a",
                   ("task-t1", "full", 42): "run-b",
                   ("task-t2", "full", 7): "run-c"}
    rows = {(g["task"], g["profile"], g["seed"]): g
            for g in bundle["groups"] if g["group_by"] == "task_x_profile_x_seed"}
    assert set(rows) == set(seed_groups)

    baseline = rows[("task-t1", "baseline", 42)]
    assert baseline["runs"] == 1 and baseline["total_verdicts"] == 10
    assert baseline["counts"] == {
        "keeps": 9, "keeps_surviving": 2, "keeps_noise_level": 6,
        "keeps_unconfirmed": 1, "mismatch_suspected": 1,
        "sentinel_flagged": 0, "unverifiable": 0}
    assert baseline["rates"] == pytest.approx({
        "keeps": 0.9, "keeps_surviving": 0.2, "keeps_noise_level": 0.6,
        "keeps_unconfirmed": 0.1, "mismatch_suspected": 0.1,
        "sentinel_flagged": 0.0, "unverifiable": 0.0})
    assert baseline["false_condemnations"] == 1
    assert baseline["gate_degraded_share"] == pytest.approx(0.2)

    # per-profile totals: full = run-b + run-c (verdict-weighted gate share)
    full = next(g for g in bundle["groups"] if g["group_by"] == "profile"
                and g["profile"] == "full")
    assert full["runs"] == 2 and full["total_verdicts"] == 13
    assert full["counts"]["keeps_surviving"] == 9
    assert full["counts"]["keeps_unconfirmed"] == 1
    assert full["counts"]["keeps_noise_level"] == 1
    assert full["counts"]["sentinel_flagged"] == 1
    assert full["counts"]["unverifiable"] == 1
    assert full["rates"]["keeps"] == pytest.approx(11 / 13)
    assert full["rates"]["keeps_surviving"] == pytest.approx(9 / 13)
    assert full["gate_degraded_share"] == pytest.approx(0.05)  # run-c reports none
    assert full["replay_probe_suspected"] == 1

    # all-runs totals: 23 verdicts, the one false condemnation surfaces here
    totals = bundle["totals"]
    assert totals["group_by"] == "all"
    assert totals["runs"] == 3 and totals["total_verdicts"] == 23
    assert totals["false_condemnations"] == 1
    # the six audit classes partition all verdicts; "keeps" is their keep subset
    assert sum(n for k, n in totals["counts"].items() if k != "keeps") == 23
    assert totals["counts"]["keeps"] == 20


def test_false_condemnations_surfaced_prominently(tmp_path, capsys) -> None:
    mod = _load()
    out = tmp_path / "audit-bundle.json"
    runs = _three_runs(tmp_path)
    assert mod.main([str(p) for p in runs] + ["--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "FALSE CONDEMNATIONS: 1" in stdout


# --- 2: tolerance (corrupt, missing, wrong schema) ------------------------------------

def test_corrupt_missing_and_wrong_schema_are_skipped_with_warnings(tmp_path) -> None:
    mod = _load()
    out = tmp_path / "audit-bundle.json"
    runs = _three_runs(tmp_path)
    corrupt = _write_run(tmp_path, "run-corrupt", raw="{not json at all")
    missing = _write_run(tmp_path, "run-missing", partial=True)  # ws/ but no audit.json
    stale = _write_run(tmp_path, "run-stale",
                       audit=_audit("run-stale", "task-t1", "baseline", 42,
                                    [_v("survives")], schema_version=2))
    argv = [str(p) for p in runs + [corrupt, missing, stale]] + ["--out", str(out)]
    assert mod.main(argv) == 0
    bundle = json.loads(out.read_text(encoding="utf-8"))

    # only the three good runs made it in
    assert {r["run_id"] for r in bundle["runs"]} == {"run-a", "run-b", "run-c"}
    warnings = "\n".join(bundle["warnings"])
    assert "run-corrupt" in warnings and "not valid JSON" in warnings
    assert "run-missing" in warnings  # partial run (has ws/, no audit.json)
    assert "run-stale" in warnings and "schema_version=2" in warnings

    # the totals still cover exactly the good runs
    assert bundle["totals"]["total_verdicts"] == 23
    baseline = next(g for g in bundle["groups"] if g["profile"] == "baseline"
                    and g["group_by"] == "task_x_profile_x_seed")
    assert baseline["total_verdicts"] == 10  # the stale-schema run did not leak in


def test_no_usable_runs_exits_1(tmp_path, capsys) -> None:
    mod = _load()
    corrupt = _write_run(tmp_path, "run-corrupt", raw="not json")
    missing = _write_run(tmp_path, "run-missing", partial=True)
    rc = mod.main([str(corrupt), str(missing),
                   "--out", str(tmp_path / "audit-bundle.json")])
    assert rc == 1
    assert "no usable audit.json" in capsys.readouterr().err


def test_runs_archive_root_is_walked(tmp_path) -> None:
    mod = _load()
    archive = tmp_path / ".artifacts"
    archive.mkdir()
    root_runs = _three_runs(archive)
    out = tmp_path / "audit-bundle.json"
    assert mod.main([str(archive), "--out", str(out)]) == 0
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert {r["run_id"] for r in bundle["runs"]} == {"run-a", "run-b", "run-c"}
    assert bundle["totals"]["runs"] == 3
    assert root_runs[0].is_dir()  # sanity: the fixture tree is where we think it is


# --- 3: the hard error ----------------------------------------------------------------

def test_unknown_audit_class_is_a_hard_error(tmp_path, capsys) -> None:
    mod = _load()
    good = _write_run(tmp_path, "run-a", audit=_noise_heavy("run-a"))
    bad = _write_run(tmp_path, "run-bad",
                     audit=_audit("run-bad", "task-t1", "full", 42, [_v("survives")]))
    bad_audit = bad / "ws" / "student" / "audit.json"
    data = json.loads(bad_audit.read_text(encoding="utf-8"))
    data["verdicts"].append({"verdict": "keep-lever", "audit_class": "mystery_class"})
    bad_audit.write_text(json.dumps(data), encoding="utf-8")

    out = tmp_path / "audit-bundle.json"
    rc = mod.main([str(good), str(bad), "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "SCHEMA ERROR" in err
    assert str(bad_audit) in err          # names the offending file
    assert "mystery_class" in err         # names the offending class
    assert not out.exists()               # no bundle is written on a schema error


# --- 4: CSV and --filter --------------------------------------------------------------

def _expected_class_counter() -> Counter:
    return (Counter({"noise_level": 6, "survives": 2, "unconfirmed_edge": 1,
                     "mismatch_suspected": 1})
            + Counter({"survives": 7, "noise_level": 1, "sentinel_flagged": 1,
                       "unverifiable": 1})
            + Counter({"survives": 2, "unconfirmed_edge": 1}))


def test_csv_has_one_row_per_verdict(tmp_path) -> None:
    mod = _load()
    out = tmp_path / "audit-bundle.json"
    csv_path = tmp_path / "audit-verdicts.csv"
    runs = _three_runs(tmp_path)
    assert mod.main([str(p) for p in runs] + ["--out", str(out), "--csv", str(csv_path)]) == 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 23  # == totals.total_verdicts, one row per verdict
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == bundle["totals"]["total_verdicts"]
    assert Counter(r["audit_class"] for r in rows) == _expected_class_counter()
    header = set(rows[0])
    assert {"run_id", "task", "profile", "seed", "verdict_index",
            "audit_class", "false_condemnation"} <= header
    # per-verdict flags survive the round trip; the extra "gate" field is carried
    assert sum(r["false_condemnation"] == "True" for r in rows) == 1
    assert all(r["gate"] == "pass" for r in rows)
    assert sorted({int(r["verdict_index"]) for r in rows if r["run_id"] == "run-a"}) \
        == list(range(10))


def test_filter_limits_runs_and_groups(tmp_path) -> None:
    mod = _load()
    out = tmp_path / "audit-bundle.json"
    runs = _three_runs(tmp_path)
    assert mod.main([str(p) for p in runs] + ["--out", str(out),
                                              "--filter", "profile=full"]) == 0
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert {r["run_id"] for r in bundle["runs"]} == {"run-b", "run-c"}
    seed_groups = [g for g in bundle["groups"] if g["group_by"] == "task_x_profile_x_seed"]
    assert {(g["task"], g["seed"]) for g in seed_groups} \
        == {("task-t1", 42), ("task-t2", 7)}
    profiles = {g["profile"] for g in bundle["groups"] if g["group_by"] == "profile"}
    assert profiles == {"full"}
    assert any("filtered out: run run-a" in w for w in bundle["warnings"])

    bad = tmp_path / "audit-bundle.json"
    with pytest.raises(SystemExit):
        mod.main([str(runs[0]), "--out", str(bad), "--filter", "kernel=foo"])


# --- CLI wiring (subprocess, like the paper pipeline will invoke it) -------------------

def test_cli_end_to_end(tmp_path) -> None:
    out = tmp_path / "audit-bundle.json"
    csv_path = tmp_path / "audit-verdicts.csv"
    runs = _three_runs(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *[str(p) for p in runs],
         "--out", str(out), "--csv", str(csv_path)],
        capture_output=True, text=True, cwd=str(REPO), check=False)
    assert proc.returncode == 0, proc.stderr
    assert "[audit-rollup]" in proc.stdout
    assert "FALSE CONDEMNATIONS: 1" in proc.stdout
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == 1
    assert bundle["totals"]["total_verdicts"] == 23
    with csv_path.open(newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 23
