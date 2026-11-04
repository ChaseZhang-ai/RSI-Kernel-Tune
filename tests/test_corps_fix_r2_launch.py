"""Regression tests for the audit-corps R2 launch/lifecycle fix batch.

Ported from the R2 audit's untracked test archive (audit-r2-2026-11-05:
test_lifecycle_fixes.py, test_monitor_resume_fixes.py), with every assertion
re-checked against this repository's semantics and adapted where the two
differ. Covered here:

* a run completes only on evidence: a loop-reported stop=target no longer
  completes a run whose confirmed best is vetoed by the correctness monitor
  (BUG-SEED-13);
* a one-sided (or chained) ⊗ companion declaration widens the in-place reopen
  closure in both directions (BUG-SEED-16);
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

import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from peregrine_kernel import baseline_cli, launch, taskcard
from peregrine_kernel.milestone import validate_milestone
from peregrine_kernel.resume import Decision, ReopenPlan, compute_reopen
from peregrine_kernel.runs import list_runs

from test_milestone import VALID

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


# -- BUG-SEED-26: section presence/order is matched at line start ----------------------------


def test_a_numbered_subheading_does_not_stand_in_for_section_4():
    # "### 4. M2 pipeline" contains "## 4." as a substring; a real distillation
    # artifact wrote such sub-headings, and a substring find let the real section
    # go missing (BUG-SEED-26)
    text = VALID.replace("## 4. Known dead ends", "### 4. M2 pipeline")
    v = validate_milestone(text)
    assert any("missing required section: ## 4." in x for x in v)


def test_a_missing_section_4_is_still_reported():
    text = VALID.replace("## 4. Known dead ends", "## 4b. Dead ends live elsewhere")
    assert any("missing required section: ## 4." in x for x in validate_milestone(text))


def test_a_real_section_4_alongside_a_numbered_subheading_passes():
    text = VALID.replace("## 4. Known dead ends",
                         "## 4. Known dead ends\n\n### 4. M2 pipeline\n(sub plan)\n")
    assert not any("## 4." in x for x in validate_milestone(text))


def test_sections_out_of_order_are_still_reported():
    sec5 = "## 5. Global correctness / performance gate\natol=5, rtol=0.05; MFU ≥95% target.\n\n"
    sec6 = "## 6. Rules\nOne variable per round; anti-plagiarism clause; INFEASIBLE clause.\n\n"
    swapped = VALID.replace(sec5 + sec6, sec6 + sec5)
    assert swapped != VALID
    assert any("out of order" in x for x in validate_milestone(swapped))


# -- BUG-SEED-36: the creation entry points validate the task name ---------------------------


def test_init_refuses_an_illegal_or_reserved_task_name(tmp_path, monkeypatch):
    real = launch.repo_root()
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    shutil.copytree(real / "tasks" / "_template", tmp_path / "tasks" / "_template")
    (tmp_path / "tasks").mkdir(exist_ok=True)
    for bad in ("../../evil", "../evil", "a/b", "_hidden", "run", "status"):
        assert launch.cmd_init([bad]) == 2, bad
    # nothing written: tasks/ holds only the template the test itself copied
    assert sorted(p.name for p in (tmp_path / "tasks").iterdir()) == ["_template"]
    assert launch.cmd_init(["gemm"]) == 0                       # a good name still works


def test_init_refuses_an_illegal_name_before_the_scaffold_runs(tmp_path, monkeypatch):
    # the dash-arg route dispatches to the scaffolder; the name is checked first
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks").mkdir()
    from peregrine_kernel import scaffold
    called = []
    monkeypatch.setattr(scaffold, "main", lambda argv: called.append(argv) or 0)
    for bad in ("../../evil", "run"):
        assert launch.cmd_init([bad, "--kernel-file", "kernel.py"]) == 2, bad
    assert called == []          # refused before the scaffold ever ran


def test_scaffold_refuses_an_illegal_or_reserved_operator_name(tmp_path):
    from peregrine_kernel import scaffold
    for bad in ("../../evil", "../evil", "a/b", "_hidden", "run"):
        assert scaffold.main([bad, "--base-dir", str(tmp_path), "--no-git"]) == 2, bad
    assert list(tmp_path.rglob("*")) == []     # nothing created


def test_new_kernels_task_refuses_an_illegal_or_reserved_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks").mkdir()
    for bad in ("../evil", "run", "_hidden"):
        assert launch.new_kernels_task(
            [bad, "--from-kernels", "--axes", '{"n": 8}']) == 2, bad
    assert list(tmp_path.rglob("*")) == [tmp_path / "tasks"]


# -- BUG-SEED-38: git failures around committing and reporting are loud ----------------------


def test_commit_loudly_refuses_when_git_status_fails(tmp_path, monkeypatch):
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


def test_kb_changes_warn_instead_of_staying_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 128, stdout="",
                                           stderr="fatal: not a git repository")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    assert launch.kb_changes() == []
    out = capsys.readouterr().out
    assert "[peregrine]" in out and "knowledge-base" in out


# -- BUG-SEED-40: the baseline CLI degrades narrowly and says so -----------------------------


def test_submit_cluster_degrades_with_a_note_on_config_errors(capsys):
    # no submit.toml anywhere for this op: degrade to "" and say so
    assert baseline_cli._submit_cluster("no_such_operator_here") == ""
    out = capsys.readouterr().out
    assert "[baseline] note:" in out and "no_such_operator_here" in out


def test_submit_cluster_lets_unexpected_errors_propagate(monkeypatch):
    from peregrine_kernel.submit import config as submit_config

    def boom(op):
        raise AttributeError("load_config changed shape")

    monkeypatch.setattr(submit_config, "load_config", boom)
    with pytest.raises(AttributeError):
        baseline_cli._submit_cluster("op")


# -- BUG-SEED-39: the run-log tee is closed, `peregrine status` survives a torn line ----------


def test_entered_closes_the_run_log_handles(tmp_path):
    log = tmp_path / "run.log"
    run = SimpleNamespace(operators_dir=tmp_path / "ops", log=log)
    machine = SimpleNamespace(env={})
    saved_out, saved_err = sys.stdout, sys.stderr
    with launch._entered(run, machine):
        tees = [sys.stdout, sys.stderr]
        assert isinstance(sys.stdout, launch._Tee)
        print("hello")
    assert sys.stdout is saved_out and sys.stderr is saved_err
    assert all(t._file.closed for t in tees)
    assert "hello" in log.read_text()
    print("after the run")        # the restored stdout still takes writes


def test_status_skips_a_half_written_line(tmp_path):
    op_dir = tmp_path / "ops" / "op"
    (op_dir / "student").mkdir(parents=True)
    (op_dir / "student" / "status.jsonl").write_text(
        '{"verdict": "KEEP"}\n{"verdict": "DROP"\nnot json at all\n{"verdict": "KEEP"}\n')
    run = SimpleNamespace(name="op", operators_dir=tmp_path / "ops", task_name="op",
                          record=lambda: {}, log=tmp_path / "no.log")
    assert "iterations: 2 (2 kept)" in launch.status_text(run)
