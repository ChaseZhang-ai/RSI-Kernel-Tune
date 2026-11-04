"""Regression tests for the audit-corps R2 launch/lifecycle fix batch.

Ported from the R2 audit's untracked test archive
(audit-r2-2026-11-05/forgekernel-untracked-tests/test_lifecycle_fixes.py and
test_monitor_resume_fixes.py), with every assertion re-checked against this
repository's semantics and adapted where the two differ. Covered here:

* a run completes only on evidence: a loop-reported stop=target no longer
  completes a run whose confirmed best is vetoed by the correctness monitor
  (BUG-SEED-13);
* a one-sided (or chained) \u2297 companion declaration widens the in-place reopen
  closure in both directions, and a one-sided declaration is warned about
  (BUG-SEED-16);
* milestone section presence/order is matched at line start (BUG-SEED-26);
* task names are validated at the creation entry points, not just at load_task
  (BUG-SEED-36);
* git failures around committing/reporting are loud, not silent skips
  (BUG-SEED-38);
* the run-log tee's file handles are closed, and `peregrine status` survives a
  torn status line (BUG-SEED-39, ported hardening);
* the baseline CLI degrades narrowly and says so (BUG-SEED-40).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from peregrine_kernel import launch, taskcard
from peregrine_kernel.resume import Decision, ReopenPlan, compute_reopen
from peregrine_kernel.runs import list_runs

from test_launch import (  # noqa: F401 -- the repo fixture and the end-to-end stand-ins
    _confirmed, _loop_ends, _start, _task, _Teacher, repo,
)


def _confirmed_run(repo, monkeypatch, stop, confirm):
    """A full `peregrine new` + `peregrine run` whose loop ends with ``stop`` in
    its summary and ``confirm`` as its final confirmation (None: none written)."""
    from peregrine_kernel.loop import cli as loop_cli
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop=stop, confirm=confirm))
    rc = _start(_task(repo))
    rec = list_runs()[0].record()
    return rc, rec


# -- BUG-SEED-13: the run ends DONE only on a confirmed-at-target best kernel ----------------


def test_stop_at_target_with_an_alarmed_best_is_not_done(repo, monkeypatch):
    # the loop stopped at its target and its best kernel is fast, but the correctness
    # monitor alarmed on it: the run must stay stopped, exit 2
    rc, rec = _confirmed_run(repo, monkeypatch, "target",
                             _confirmed(8.0, monitor="alarm"))
    assert rc == 2
    assert rec["status"] == "stopped" and rec["stop"] == "target"
    assert rec["target_met"] is False


def test_stop_at_target_without_a_confirmation_records_no_target_met(repo, monkeypatch):
    # Adapted on port. The archive's version demands rc == 2 for a stop=target with
    # no confirmation on file, but tests/test_launch.py (frozen, read-only) pins the
    # historical contract in five places: the loop's own gate still completes such a
    # run (the end-to-end stand-ins never write a confirmation). What the fix must
    # hold here is honesty: an unconfirmed run never claims a met target.
    rc, rec = _confirmed_run(repo, monkeypatch, "target", None)
    assert rc == 0 and rec["status"] == "done" and rec["stop"] == "target"
    assert rec["target_met"] is False
    report = (list_runs()[0].dir / "report.md").read_text()
    assert "meets the target" not in report


def test_stop_at_target_with_a_confirmed_best_still_completes(repo, monkeypatch):
    # the positive control: stop=target AND a clean confirmation at the target is done
    rc, rec = _confirmed_run(repo, monkeypatch, "target", _confirmed(1.3))
    assert rc == 0
    assert rec["status"] == "done" and rec["target_met"] is True


# -- BUG-SEED-16: a one-sided ⊗ companion declaration keeps the pair together ----------------


def test_reopen_pulls_a_one_sided_companion_both_ways():
    traj = [Decision("D1"), Decision("D2", depends_on={"D1"}, companions={"D1"})]
    assert compute_reopen(traj, {"D2"}) == ReopenPlan(reopen=["D1", "D2"], kept=[])
    assert compute_reopen(traj, {"D1"}) == ReopenPlan(reopen=["D1", "D2"], kept=[])


def test_reopen_follows_a_companion_chain_to_the_fixpoint():
    traj = [Decision("A", companions={"B"}), Decision("B", companions={"C"}),
            Decision("C")]
    for changed in ("A", "B", "C"):
        assert compute_reopen(traj, {changed}).reopen == ["A", "B", "C"]
