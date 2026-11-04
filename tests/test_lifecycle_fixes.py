"""Regression tests ported from the R2 audit's lifecycle fix batch (renamed to
peregrine_kernel), kept beside the fixes they pin:

* creation entry points validate the task name (no path escape, no reserved name);
* git failures around committing are loud, not silent skips;
* the run-log tee handles are closed when the drive ends;
* the baseline CLI degrades narrowly (expected config trouble only) and says so;
* DONE needs a confirmed-at-target best kernel even when the loop itself stopped
  at its target (a correctness-monitor alarm, or no confirmation on file, keeps
  the run stopped and the exit code at 2);
* the run status tolerates a torn final line.

The board-lane tests the archived batch also carried are not ported: that card
lane is gone from peregrine_kernel (see CHANGES).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from peregrine_kernel import launch

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

    from peregrine_kernel import taskcard
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
    from peregrine_kernel import launch as _launch
    op_dir = tmp_path / "ops" / "op"
    (op_dir / "student").mkdir(parents=True)
    (op_dir / "student" / "status.jsonl").write_text(
        '{"verdict": "KEEP"}\n{"verdict": "DROP"\nnot json at all\n{"verdict": "KEEP"}\n')
    run = SimpleNamespace(name="op", operators_dir=tmp_path / "ops", task_name="op",
                          record=lambda: {}, log=tmp_path / "no.log")
    assert "iterations: 2 (2 kept)" in _launch.status_text(run)


# -- fix: the baseline CLI degrades narrowly and says so -------------------------------------


def test_submit_cluster_degrades_with_a_note_on_config_errors(capsys):
    # no submit.toml anywhere for this op: degrade to "" and say so
    from peregrine_kernel import baseline_cli
    assert baseline_cli._submit_cluster("no_such_operator_here") == ""
    out = capsys.readouterr().out
    assert "[baseline] note:" in out and "no_such_operator_here" in out


def test_submit_cluster_lets_unexpected_errors_propagate(monkeypatch):
    import peregrine_kernel.submit.config as submit_config
    from peregrine_kernel import baseline_cli

    def boom(op):
        raise AttributeError("load_config changed shape")

    monkeypatch.setattr(submit_config, "load_config", boom)
    with pytest.raises(AttributeError):
        baseline_cli._submit_cluster("op")


# -- fix: DONE follows a clean confirmed-at-target best, stop=target may not vouch ----


def _confirmed_run(repo, monkeypatch, stop, confirm):
    from peregrine_kernel.loop import cli as loop_cli
    from peregrine_kernel.runs import list_runs
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop=stop, confirm=confirm))
    rc = _start(_task(repo))
    return rc, list_runs()[0].record()


def test_stop_at_target_with_an_alarmed_best_is_not_done(repo, monkeypatch):
    # the loop stopped at its target and its best kernel is fast, but the correctness
    # monitor alarmed on it: the run must stay stopped, exit 2
    rc, rec = _confirmed_run(repo, monkeypatch, "target",
                             _confirmed(8.0, monitor="alarm"))
    assert rc == 2
    assert rec["status"] == "stopped" and rec["stop"] == "target"
    assert rec["target_met"] is False


def test_stop_at_target_without_a_confirmation_is_not_done(repo, monkeypatch):
    # stop=target with no best_confirmation.json on file is no better
    rc, rec = _confirmed_run(repo, monkeypatch, "target", None)
    assert rc == 2
    assert rec["status"] == "stopped" and rec["stop"] == "target"
    assert rec["target_met"] is False


def test_stop_at_target_with_a_confirmed_best_still_completes(repo, monkeypatch):
    # the positive control: stop=target AND a clean confirmation at the target is done
    rc, rec = _confirmed_run(repo, monkeypatch, "target", _confirmed(1.3))
    assert rc == 0
    assert rec["status"] == "done" and rec["target_met"] is True
