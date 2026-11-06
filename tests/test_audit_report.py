# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""rsi-kernel-tune audit: the read-only provenance report of a finished run.

Fixtures are synthetic run directories laid out exactly as a run writes them
(runs.py / loop/cli.py): run.json, operators/<op>/{loop.toml,student/status.jsonl,
best_confirmation.json,correctness_monitor.jsonl}, check/groups.json, run.log.
Trajectory rows are written with the production writer (inner.trajectory) so the
audit reads what a real run persists, not a private shape.

The tests cover: KEEP-survives, KEEP-noise-level, KEEP-unconfirmed-edge,
replay-probe-suspected, a stream-sentinel violation riding the fresh-recheck
fail path, the all-mechanisms-off baseline profile, false-condemnation detection
(one positive fixture), and malformed / absent blocks (the audit degrades, never
crashes). A final smoke test parses a real archived run under .artifacts/ read-
only when one exists.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsi_kernel_tune.audit import AuditError, build_audit, cmd_audit, render_text
from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record
from rsi_kernel_tune.launch import COMMANDS

VERDICT_KEYS = {"iter", "verdict", "margin_pct", "resolvable_diff_pct",
                "within_noise_floor", "confirm_jobs", "confirm_readings",
                "best_standard_error_pct", "fresh_recheck", "replay_probe",
                "stream_sentinel", "timer_identity", "proc_census",
                "gate_degraded", "corr_margin_ratio", "judge_edge_recheck",
                "clock_drift", "compute_dtype", "audit_class",
                "false_condemnation"}
TALLY_KEYS = {"total_verdicts", "keeps", "keeps_surviving", "keeps_noise_level",
              "keeps_unconfirmed", "sentinel_trips", "replay_probe_suspected",
              "false_condemnations", "gate_degraded_share"}

STRICT = {"contract": "strict", "status": "pass", "truth": "float64"}


def make_run(tmp_path: Path, *, name: str = "demo") -> Path:
    """A run directory skeleton as runs.py lays it out (no trajectory yet)."""
    run = tmp_path / f"{name}_261101-000000-abcd"
    (run / "operators" / name / "student").mkdir(parents=True)
    (run / "check").mkdir()
    (run / "run.json").write_text(json.dumps({
        "task": name, "id": "261101-000000-abcd", "created": "x", "updated": "x",
        "status": "stopped", "stop": "meta_budget",
    }), encoding="utf-8")
    (run / "operators" / name / "loop.toml").write_text(
        '[loop]\ngate_window = "device"\n', encoding="utf-8")
    (run / "check" / "groups.json").write_text(json.dumps(
        {"tiny": [{"name": "g0", "seed": 4242}]}), encoding="utf-8")
    return run


def write_rows(run: Path, recs: list[IterationRecord], *, name: str = "demo") -> None:
    status = run / "operators" / name / "student" / "status.jsonl"
    for r in recs:
        append_record(status, r)


def row(**kw) -> IterationRecord:
    """One trajectory row with the schema-required keys defaulted."""
    base = dict(iter=0, decision_id="d0", change="technique", correctness=None,
                verdict="NEUTRAL")
    base.update(kw)
    return IterationRecord(**base)


def tree_digest(root: Path) -> dict[str, str]:
    """Relative path -> sha256 of every file under root (the read-only check)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------- fixture rows --


def full_rows() -> list[IterationRecord]:
    """The full-profile fixture: scaffold NEUTRAL, three KEEPs (survives /
    noise-level / unconfirmed-edge), a replay-probe conviction, a stream-sentinel
    conviction. precision blocks ride every evaluated row (the contract was on)."""
    return [
        row(iter=0, decision_id="d0", change="scaffold only"),
        row(iter=1, decision_id="d1", change="vectorized loads", correctness=True,
            verdict="KEEP", metric_before=2.0, device_median_ms=1.90,
            vs_ref_device=1.05, sha="aa1111", confirm_jobs=2,
            confirm_readings=[1.9, 1.88, 1.92], best_standard_error=0.01,
            precision=STRICT, gate_degraded=False, clock_drift=False,
            compute_dtype="fp32"),
        row(iter=2, decision_id="d2", change="smaller tile", correctness=True,
            verdict="KEEP", metric_before=1.90, device_median_ms=1.895,
            vs_ref_device=1.049, sha="aa2222", precision=STRICT,
            gate_degraded=False, corr_margin_ratio=0.5, compute_dtype="fp32"),
        row(iter=3, decision_id="d3", change="wider tile", correctness=True,
            verdict="KEEP", metric_before=1.895, device_median_ms=1.88,
            vs_ref_device=1.048, sha="aa3333", precision=STRICT, gate_degraded=True,
            corr_margin_ratio=0.95, confirm_jobs=0, compute_dtype="fp32"),
        row(iter=4, decision_id="d4", change="cache the block", correctness=False,
            verdict="REVERT", metric_before=1.88,
            error="replay_probe SUSPECTED (threshold x1.5): the timed windows did "
                  "not recompute their inputs",
            suspicious=True, replay_probe={"status": "suspected", "ratio_threshold": 1.5,
                                           "shapes": [{"name": "g0", "status": "suspected"}]},
            precision=STRICT, corr_margin_ratio=0.2),
        row(iter=5, decision_id="d5", change="side-stream prefetch", correctness=False,
            verdict="REVERT", metric_before=1.88,
            error="fresh_recheck FAIL (mode=stream): the correctness verdict did not "
                  "reproduce on fresh inputs after timing — a timed call drained far "
                  "more than its event window measured",
            suspicious=True, precision=STRICT),
    ]


def baseline_rows() -> list[IterationRecord]:
    """All mechanisms off: no precision block (the contract off), no WP9/WP20/WP29
    reading anywhere. Two KEEPs with plain timing rows."""
    return [
        row(iter=0, decision_id="b0", change="first cut", correctness=True,
            verdict="KEEP", metric_before=2.0, sha="bb0000"),
        row(iter=1, decision_id="b1", change="unroll", correctness=True,
            verdict="KEEP", metric_before=2.0, device_median_ms=1.90,
            vs_ref_device=1.05, sha="bb0001"),
    ]


# --------------------------------------------------------------------- helpers --


def write_confirm(run: Path, obj: dict | None, *, name: str = "demo") -> None:
    p = run / "operators" / name / "best_confirmation.json"
    if obj is not None:
        p.write_text(json.dumps(obj), encoding="utf-8")


def write_monitor(run: Path, status: str = "ok", *, name: str = "demo") -> None:
    p = run / "operators" / name / "correctness_monitor.jsonl"
    p.write_text(json.dumps({"time": "t", "sha": "aa1111", "job": "j",
                             "trigger": "finish", "status": status,
                             "findings": [], "ran": [], "not_applicable": [],
                             "result": {}}) + "\n", encoding="utf-8")


def write_runlog(run: Path) -> None:
    (run / "run.log").write_text(
        "[loop] something earlier\n"
        "[loop] [wp11] wall-clock composition of the loop (measurement only — no "
        "verdict reads this; shares over the 38.2s of measured buckets):\n"
        "  agent_turn_s     30.1s  78.8%\n"
        "  eval_job_s        8.1s  21.2%\n"
        "[loop] later line\n", encoding="utf-8")


# ----------------------------------------------------------------------- tests --


def test_command_registered():
    """The launch registry dispatches audit like the other run commands."""
    assert "audit" in COMMANDS


def test_full_profile_provenance_table(tmp_path):
    """The full-profile fixture: per-verdict classes, probe columns, margins."""
    run = make_run(tmp_path)
    write_rows(run, full_rows())
    write_confirm(run, {"sha": "aa1111", "window": "device", "median_ms": 1.9,
                        "jobs": [{"job": "j1", "ms": 1.9, "ratio": 1.05}]})
    write_monitor(run, "ok")
    write_runlog(run)

    report = build_audit(run)
    assert report["schema_version"] == 1
    assert report["run_id"] == "261101-000000-abcd"
    assert report["task"] == "demo"
    assert report["profile"] == "full"
    assert report["seed"] == "4242"
    assert report["gate_window"] == "device"
    assert report["correctness_monitor"] == "ok"
    assert report["wall_clock_wp11"]["buckets"]["eval_job_s"] == 8.1

    assert all(set(v) == VERDICT_KEYS for v in report["verdicts"])
    assert set(report["tallies"]) == TALLY_KEYS

    iters = {v["iter"]: v for v in report["verdicts"]}
    assert len(report["verdicts"]) == 6
    # scaffold NEUTRAL: no margin data, no defense reading, class unverifiable
    assert iters[0]["verdict"] == "NEUTRAL"
    assert iters[0]["margin_pct"] is None and iters[0]["within_noise_floor"] is None
    assert iters[0]["audit_class"] == "unverifiable"
    # KEEP-survives: margin (2.0-1.9)/2.0 = 5% over the estimator's 1.5% fallback,
    # confirmations on the row, no probe failures anywhere on it
    assert iters[1]["verdict"] == "KEEP"
    assert iters[1]["margin_pct"] == pytest.approx(5.0, abs=1e-3)
    assert iters[1]["within_noise_floor"] is False
    assert iters[1]["audit_class"] == "survives"
    assert iters[1]["confirm_jobs"] == 2 and iters[1]["confirm_readings"] == 3
    assert iters[1]["best_standard_error_pct"] == pytest.approx(1.0, abs=1e-6)
    assert iters[1]["false_condemnation"] is False
    # KEEP-noise-level: the 0.26% claimed gain sits under the 1.5% resolution
    assert iters[2]["margin_pct"] == pytest.approx(0.2631, abs=1e-3)
    assert iters[2]["within_noise_floor"] is True
    assert iters[2]["audit_class"] == "noise_level"
    # KEEP-unconfirmed-edge: the verdict sat on the judge's edge band (0.95) with
    # zero confirmation jobs; the class rule puts this ahead of the noise test
    assert iters[3]["audit_class"] == "unconfirmed_edge"
    assert iters[3]["gate_degraded"] is True
    # the replay-probe conviction: the probe column says suspected, the recheck
    # column stays null (the probe family owns the row, not the recheck family)
    assert iters[4]["verdict"] == "REVERT"
    assert iters[4]["replay_probe"] == "suspected"
    assert iters[4]["fresh_recheck"] is None
    assert iters[4]["audit_class"] == "mismatch_suspected"
    # the stream-sentinel conviction: fail_stream on the recheck column, the
    # violation on the sentinel column, the class is sentinel_flagged (a stream
    # lie is not a cache-replay mismatch)
    assert iters[5]["fresh_recheck"] == "fail_stream"
    assert iters[5]["stream_sentinel"] == "violation"
    assert iters[5]["timer_identity"] is None
    assert iters[5]["audit_class"] == "sentinel_flagged"

    t = report["tallies"]
    assert t == {"total_verdicts": 6, "keeps": 3, "keeps_surviving": 1,
                 "keeps_noise_level": 1, "keeps_unconfirmed": 1,
                 "sentinel_trips": 1, "replay_probe_suspected": 1,
                 "false_condemnations": 0, "gate_degraded_share": 0.3333}
    text = render_text(report)
    assert "false condemnations: 0" in text
    assert "survives" in text and "sentinel_flagged" in text


def test_baseline_profile_mechanisms_off(tmp_path):
    """All mechanisms off: the profile reads baseline and every probe column
    reads off; the KEEPs are unverifiable (nothing verified them)."""
    run = make_run(tmp_path)
    write_rows(run, baseline_rows())
    report = build_audit(run)
    assert report["profile"] == "baseline"
    assert report["verdicts"][0]["fresh_recheck"] == "off"
    assert report["verdicts"][0]["replay_probe"] == "off"
    assert report["verdicts"][0]["stream_sentinel"] == "off"
    assert report["verdicts"][0]["timer_identity"] == "off"
    assert report["verdicts"][0]["proc_census"] == "off"
    # the second KEEP has a real margin but nothing verified it: unverifiable,
    # not survives — the honest provenance for an unverified claim
    assert report["verdicts"][1]["margin_pct"] == pytest.approx(5.0, abs=1e-3)
    assert report["verdicts"][1]["within_noise_floor"] is False
    for v in report["verdicts"]:
        assert v["audit_class"] == "unverifiable"
    t = report["tallies"]
    assert t["keeps"] == 2 and t["keeps_surviving"] == 0
    assert t["sentinel_trips"] == 0 and t["replay_probe_suspected"] == 0
    assert t["false_condemnations"] == 0
    assert t["gate_degraded_share"] is None


def test_nondeterministic_form_maps_separately(tmp_path):
    """A fresh-recheck FAIL whose form is nondeterministic (the same-seed double
    run disagreeing with itself) reads fail_nondeterministic, classifies
    sentinel-free (it is neither a cache-replay mismatch nor a sentinel trip)."""
    run = make_run(tmp_path)
    write_rows(run, [
        row(iter=0, decision_id="n0", change="atomics reduction", correctness=False,
            verdict="REVERT", metric_before=2.0,
            error="fresh_recheck FAIL (mode=regen, form=nondeterministic): the "
                  "correctness verdict did not reproduce on fresh inputs",
            suspicious=True, fresh_recheck_form="nondeterministic"),
    ])
    report = build_audit(run)
    v = report["verdicts"][0]
    assert v["fresh_recheck"] == "fail_nondeterministic"
    assert v["stream_sentinel"] is None and v["timer_identity"] is None
    assert v["audit_class"] == "unverifiable"   # a REVERT with no sentinel reading
    assert report["tallies"]["sentinel_trips"] == 0


def test_false_condemnation_positive(tmp_path):
    """The ledger: a defense reading fired on the row whose sha is the final
    confirmed best — one false condemnation, reported as the red finding."""
    run = make_run(tmp_path)
    write_rows(run, [
        row(iter=0, decision_id="c0", change="plain keep", correctness=True,
            verdict="KEEP", metric_before=2.0, device_median_ms=1.5, sha="cafe0000"),
        row(iter=1, decision_id="c1", change="cache replay try", correctness=False,
            verdict="REVERT", metric_before=1.5,
            error="fresh_recheck FAIL (mode=process): the correctness verdict did "
                  "not reproduce on fresh inputs after timing",
            suspicious=True, sha="cafe0001"),
    ])
    write_confirm(run, {"sha": "cafe0001", "window": "device", "median_ms": 1.5,
                        "jobs": []})
    report = build_audit(run)
    assert report["verdicts"][1]["false_condemnation"] is True
    assert report["verdicts"][0]["false_condemnation"] is False
    assert report["tallies"]["false_condemnations"] == 1
    assert "RED FINDING" in render_text(report)
    # and the honest zero: the same run confirming the CLEAN keep instead
    write_confirm(run, {"sha": "cafe0000", "window": "device", "median_ms": 1.5,
                        "jobs": []})
    report = build_audit(run)
    assert all(v["false_condemnation"] is False for v in report["verdicts"])
    assert report["tallies"]["false_condemnations"] == 0
    assert "false condemnations: 0" in render_text(report)


def test_malformed_and_absent_blocks_degrade(tmp_path, capsys):
    """Malformed blocks, junk-typed fields, a torn final line and every optional
    file absent: the audit degrades to off / null and never crashes."""
    run = make_run(tmp_path)
    (run / "check" / "groups.json").unlink()   # no seed ledger on disk
    status = run / "operators" / "demo" / "student" / "status.jsonl"
    rows = [
        # a pre-WP20 row: only the earliest fields exist
        {"iter": 0, "decision_id": "m0", "change": "old row", "correctness": True,
         "verdict": "KEEP", "metric_before": 2.0, "device_median_ms": 1.5},
        # a row with junk-typed mechanism fields (a mangled trajectory)
        {"iter": 1, "decision_id": "m1", "change": "junk blocks", "correctness": True,
         "verdict": "KEEP", "metric_before": 1.5, "device_median_ms": 1.4,
         "replay_probe": "junk", "confirm_readings": "nope", "gate_degraded": "yes",
         "judge_edge_recheck": 42, "corr_margin_ratio": "oops"},
    ]
    status.write_text("\n".join(json.dumps(r) for r in rows)
                      + '\n{"iter": 2, "verdict": "KEEP", "correctness": tr',
                      encoding="utf-8")
    report = build_audit(run)   # must not raise
    assert report["schema_version"] == 1
    assert report["profile"] == "baseline"   # no mechanism evidence anywhere
    assert report["seed"] is None            # groups.json deleted above
    assert report["gate_window"] == "device"
    assert report["correctness_monitor"] is None
    assert report["wall_clock_wp11"] is None
    assert len(report["verdicts"]) == 2      # the torn final line is dropped
    v = report["verdicts"][1]
    assert v["replay_probe"] is None         # junk block -> missing reading
    assert v["confirm_readings"] is None
    assert v["gate_degraded"] is None        # junk-typed flag -> unreported
    assert v["judge_edge_recheck"] is None
    assert v["corr_margin_ratio"] is None
    assert v["fresh_recheck"] == "off"       # profile says the machinery was off
    assert all(vv["false_condemnation"] is False for vv in report["verdicts"])
    assert report["tallies"]["false_condemnations"] == 0


def test_malformed_confirmation_and_helper_files(tmp_path):
    """A corrupt best_confirmation.json disables the cross-reference (no crash, no
    false flags); absent optional files read null; groups.json seeds the seed."""
    run = make_run(tmp_path)
    write_rows(run, [
        row(iter=0, decision_id="k0", change="keep", correctness=True, verdict="KEEP",
            metric_before=2.0, device_median_ms=1.5, sha="aa9999",
            suspicious=True,  # fired evidence, but nothing to confirm against
            error="fresh_recheck FAIL (mode=process): x"),
    ])
    (run / "operators" / "demo" / "best_confirmation.json").write_text(
        "{not json", encoding="utf-8")
    (run / "run.log").write_text("no composition block here\n", encoding="utf-8")
    report = build_audit(run)
    assert report["verdicts"][0]["false_condemnation"] is False
    assert report["tallies"]["false_condemnations"] == 0
    assert report["correctness_monitor"] is None
    assert report["wall_clock_wp11"] is None
    assert report["seed"] == "4242"
    assert report["profile"] == "full"   # the fired reading is mechanism evidence


def test_command_writes_report_and_only_that(tmp_path, capsys):
    """cmd_audit writes audit.json at the contracted ws/student/ path under the
    run directory and NOTHING else in the run changes; --json-only prints the
    document itself."""
    run = make_run(tmp_path)
    write_rows(run, full_rows())
    write_confirm(run, {"sha": "aa1111", "window": "device", "median_ms": 1.9,
                        "jobs": []})
    write_monitor(run, "warn")
    before = tree_digest(run)

    rc = cmd_audit([str(run)])
    assert rc == 0
    report_path = run / "ws" / "student" / "audit.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1 and report["task"] == "demo"
    after = tree_digest(run)
    assert set(after) - set(before) == {str(report_path.relative_to(run))}
    for p, h in before.items():
        assert after[p] == h   # no existing byte touched
    out = capsys.readouterr().out
    assert "tallies:" in out and "audit report:" in out

    rc = cmd_audit([str(run), "--json-only"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["run_id"] == report["run_id"]
    assert "correctness monitor (finish): warn" not in capsys.readouterr().out


def test_unauditable_directories_exit_2(tmp_path, capsys):
    """Not a run / a run without a trajectory: a readable refusal, exit 2."""
    assert cmd_audit([str(tmp_path / "missing")]) == 2
    assert "not a directory" in capsys.readouterr().err
    empty = tmp_path / "empty_run"
    empty.mkdir()
    (empty / "run.json").write_text(json.dumps({"task": "x", "id": "i"}), encoding="utf-8")
    assert cmd_audit([str(empty)]) == 2
    assert "no trajectory found" in capsys.readouterr().err
    with pytest.raises(AuditError):
        build_audit(empty)


def test_smoke_real_archived_run_read_only():
    """One real archived run under .artifacts/, parsed read-only (build_audit
    writes nothing, byte for byte). Skipped where no such run exists."""
    from rsi_kernel_tune.audit import SCHEMA_VERSION

    src = Path(__file__).resolve().parent.parent / "src"
    artifacts = src.parent / ".artifacts"
    report, target = None, None
    if artifacts.is_dir():
        for d in sorted(artifacts.iterdir()):
            # an archived run whose trajectory holds at least one row: an empty
            # status file audits fine (zero verdicts) but proves nothing here
            if not (d / "run.json").is_file():
                continue
            for status in (d / "operators").glob("*/student/status.jsonl"):
                if not status.is_file() or not status.read_text(encoding="utf-8").strip():
                    continue
                try:
                    report = build_audit(d)
                except AuditError:
                    continue
                if report["verdicts"]:
                    target = d
                    break
                report = None
            if report is not None:
                break
    if report is None:
        pytest.skip("no archived run with a parsed trajectory under .artifacts/")
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["verdicts"]
    assert all(set(v) == VERDICT_KEYS for v in report["verdicts"])
    assert set(report["tallies"]) == TALLY_KEYS
    assert report["profile"] in ("baseline", "full", "unknown")
    before = tree_digest(target)
    again = build_audit(target)
    assert again == report                     # pure: same input, same report
    assert tree_digest(target) == before       # read-only, byte for byte
    # the dispatch face lists the command (the registry entry is wired in __main__)
    probe = subprocess.run(
        [sys.executable, "-m", "rsi_kernel_tune", "audit", "--help"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(src)})
    assert probe.returncode == 0 and "audit" in (probe.stdout + probe.stderr)
