"""Regression tests for the W1 cli/launch fix line (dedup rebuild).

Only the W1-unique faces live here; the corps' own tests on this branch cover
the shared ones (SEED-07 / SEED-10 / SEED-13 in their lifecycle/guards files,
SEED-19 / SEED-21 / SEED-36 / SEED-39 / SEED-48 in their cli/lifecycle files):

* a config-phase SystemExit still lands FAILED in the run record (W1-LIF-02);
* the clock-lock refusal budget counts extra submissions, like the lost-runner
  budget does (SEED-18 -- W1 re-lands the counter unification on top of the
  corps' retry-budget rework);
* `rsi-kernel-tune status` survives a torn status.jsonl line (P3-14);
* ``hours`` is documented as the loop's whole wall clock, the same way twice
  (W1-KNB-04).
"""

from __future__ import annotations

import pytest

from rsi_kernel_tune import launch

from test_launch import (  # tests/ is on sys.path under pytest
    _start,
    _task,
    repo,  # noqa: F401 -- the shared fixture
)


# -- W1-LIF-02: a config-phase SystemExit marks the run failed -------------------------------


def test_a_config_systemexit_marks_the_run_failed(repo, monkeypatch):
    """cli.main refuses bad [loop] config at launch with `raise SystemExit(...)` (six
    sites: reuse_unchanged_reading, gate_window, oracle_mode x3, kb_arch). Those flew
    past drive()'s `except Exception`, so the run stayed "running" forever -- no
    error, nothing to resume from."""
    from rsi_kernel_tune.loop import cli as loop_cli
    from rsi_kernel_tune.runs import list_runs
    from test_launch import _Teacher

    def refuse_config(argv):
        raise SystemExit('[loop] reuse_unchanged_reading must be "all", "gate" or '
                         '"off", got "every"')

    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", refuse_config)
    with pytest.raises(SystemExit):
        _start(_task(repo))
    rec = list_runs()[0].record()
    assert rec["status"] == "failed"
    assert "SystemExit" in (rec.get("error") or "") and "reuse_unchanged_reading" in rec["error"]


# -- P3-14: `rsi-kernel-tune status` survives a torn status.jsonl line ---------------------


def test_status_skips_a_half_written_line(tmp_path):
    # P3-14: a torn status.jsonl line must not crash `rsi-kernel-tune status`
    op_dir = tmp_path / "ops" / "op"
    (op_dir / "student").mkdir(parents=True)
    (op_dir / "student" / "status.jsonl").write_text(
        '{"verdict": "KEEP"}\n{"verdict": "DROP"\nnot json at all\n{"verdict": "KEEP"}\n')
    from types import SimpleNamespace
    run = SimpleNamespace(name="op", operators_dir=tmp_path / "ops", task_name="op",
                          record=lambda: {}, log=tmp_path / "no.log")
    text = launch.status_text(run)
    assert "iterations: 2 (2 kept)" in text
