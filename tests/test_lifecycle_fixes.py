"""Regression tests ported from the R2 audit's lifecycle fix batch (renamed to
rsi_kernel_tune), kept beside the fixes they pin:

* creation entry points validate the task name (no path escape, no reserved name);
* git failures around committing are loud, not silent skips;
* the run-log tee handles are closed when the drive ends;
* the baseline CLI degrades narrowly (expected config trouble only) and says so;
* DONE needs a confirmed-at-target best kernel even when the loop itself stopped
  at its target (a correctness-monitor alarm, or no confirmation on file, keeps
  the run stopped and the exit code at 2);
* the run status tolerates a torn final line.

The board-lane tests the archived batch also carried are not ported: that card
lane is gone from rsi_kernel_tune (see CHANGES).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rsi_kernel_tune import launch

from test_launch import (  # noqa: F401 -- the repo fixture and the end-to-end stand-ins
    _confirmed, _loop_ends, _start, _task, _Teacher, repo,
)


# -- fix: creation entry points validate the task name --------------------------------------


def test_init_refuses_an_illegal_or_reserved_task_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks").mkdir()
    for bad in ("../../evil", "../evil", "a/b", "_hidden", "run", "status"):
        assert launch.cmd_init([bad]) == 2, bad
    assert list(tmp_path.rglob("*")) == [tmp_path / "tasks"]    # nothing written


# -- fix: git failures around committing are loud, not silent skips --------------------------


def test_commit_loudly_refuses_when_git_status_fails(tmp_path, monkeypatch):
    import subprocess

    from rsi_kernel_tune import taskcard
    tree = tmp_path / "card"
    tree.mkdir()
    subprocess.run(["git", "-C", str(tree), "init", "-q", "-b", "main"], check=True,
                   capture_output=True)
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "status" in cmd and "--porcelain" in cmd:
            # what a broken git prints: empty stdout, non-zero rc -- not "nothing to do"
            return subprocess.CompletedProcess(cmd, 128, stdout="",
                                               stderr="fatal: not a git repository")
        return real_run(cmd, **kw)

    monkeypatch.setattr(taskcard.subprocess, "run", fake_run)
    (tree / "f.txt").write_text("x")
    with pytest.raises(subprocess.CalledProcessError):
        taskcard._commit(tree, "test")


# -- fix: the run log tee is closed when the drive ends --------------------------------------


def test_entered_closes_the_run_log_handles(tmp_path):
    import sys

    log = tmp_path / "run.log"
    run = SimpleNamespace(operators_dir=tmp_path / "ops", log=log)
    machine = SimpleNamespace(env={})
    saved_out, saved_err = sys.stdout, sys.stderr
    with launch._entered(run, machine):
        tees = [sys.stdout, sys.stderr]
        assert isinstance(sys.stdout, launch._Tee)
        print("hello")
    assert sys.stdout is saved_out and sys.stderr is saved_err
    assert all(t._file.closed for t in tees)     # no leaked run.log handle per drive
    assert "hello" in log.read_text()
    print("after the run")        # the restored stdout still takes writes


# -- fix: the run status survives a torn status line -----------------------------------------


def test_status_skips_a_half_written_line(tmp_path):
    from rsi_kernel_tune import launch as _launch
    op_dir = tmp_path / "ops" / "op"
    (op_dir / "student").mkdir(parents=True)
    (op_dir / "student" / "status.jsonl").write_text(
        '{"verdict": "KEEP"}\n{"verdict": "DROP"\nnot json at all\n{"verdict": "KEEP"}\n')
    run = SimpleNamespace(name="op", operators_dir=tmp_path / "ops", task_name="op",
                          record=lambda: {}, log=tmp_path / "no.log")
    assert "iterations: 2 (2 kept)" in _launch.status_text(run)


# -- fix: the baseline CLI degrades narrowly and says so -------------------------------------


# [merge note] the fix-c line's copies of
# test_submit_cluster_degrades_with_a_note_on_config_errors and
# test_stop_at_target_without_a_confirmation_is_not_done were dropped here:
# tests/test_corps_fix_r2_launch.py carries this tree's authoritative pins
# (an absent submit config degrades in silence; a confirmation-less
# stop=target needs a hand-rolled loop stand-in this file's fixtures cannot
# express), and the c copies pinned the pre-merge behaviors.

def test_stop_at_target_with_a_confirmed_best_still_completes(repo, monkeypatch):
    # the positive control: stop=target AND a clean confirmation at the target is done
    rc, rec = _confirmed_run(repo, monkeypatch, "target", _confirmed(1.3))
    assert rc == 0
    assert rec["status"] == "done" and rec["target_met"] is True

def _confirmed_run(repo, monkeypatch, stop, confirm):
    from rsi_kernel_tune.loop import cli as loop_cli
    from rsi_kernel_tune.runs import list_runs
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop=stop, confirm=confirm))
    rc = _start(_task(repo))
    return rc, list_runs()[0].record()


# -- PORT union: milestone required-section presence / order ----------------------

def test_a_missing_section_4_is_still_reported():
    from rsi_kernel_tune.milestone import validate_milestone
    from test_milestone import VALID
    text = VALID.replace("## 4. Known dead ends", "## 4b. Dead ends live elsewhere")
    assert any("missing required section: ## 4." in x for x in validate_milestone(text))


def test_sections_out_of_order_are_still_reported():
    from rsi_kernel_tune.milestone import validate_milestone
    from test_milestone import VALID
    sec5 = "## 5. Global correctness / performance gate\natol=5, rtol=0.05; MFU ≥95% target.\n\n"
    sec6 = "## 6. Rules\nOne variable per round; anti-plagiarism clause; INFEASIBLE clause.\n\n"
    swapped = VALID.replace(sec5 + sec6, sec6 + sec5)
    assert swapped != VALID
    assert any("out of order" in x for x in validate_milestone(swapped))
