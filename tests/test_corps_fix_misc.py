"""Regression tests for the audit-corps miscellaneous fix wave (BUG-SEED-23/26/27/
36/38/39/40/44 and the assertion upgrades BUG-SEED-46 / BUG-C11-01).

Each test names the bug it pins so the ledger stays traceable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.baseline import BASIS_CUPTI  # noqa: E402
from peregrine_kernel.timing_audit import host_path_note  # noqa: E402

from test_milestone import VALID as MILESTONE_VALID  # noqa: E402
from peregrine_kernel.milestone import validate_milestone  # noqa: E402


# --- BUG-SEED-23: the host-path note's thinner/thicker wording follows the direction --

def test_seed23_host_path_note_keeps_thinner_when_the_reference_is_faster():
    """ratio > 1: the student's wrapper is thicker, so the reference reads as
    thinner — the wording the pre-existing tests already pin."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.04 = 2.0
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.04)
    assert "2.0x thinner" in note
    assert "thicker" not in note


def test_seed23_host_path_note_says_thicker_when_the_reference_is_slower():
    """ratio < 1: the student's wrapper is the lighter one, so a fixed 'thinner'
    read backwards — the reference is thicker than the student's path."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.4 = 0.2
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.4)
    assert "5.0x thicker" in note
    assert "thinner" not in note
    assert "0.2x thinner" not in note


def test_seed23_host_path_note_handles_equal_enqueue_costs():
    """ratio == 1: neither direction, so neither thinner nor thicker may fire."""
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.08)
    assert "thinner" not in note and "thicker" not in note
    assert "the same thickness" in note


# --- BUG-SEED-26: validate_milestone's section presence must be heading-anchored ------

def test_seed26_a_low_level_heading_cannot_masquerade_as_a_required_section():
    """'### 4. M2 pipeline' contains '## 4.' as a substring; a plain text.find
    accepts it in place of the real '## 4.' section heading, while _section_body
    (which validates §3's extent) already matches at line start only."""
    masquerade = MILESTONE_VALID.replace("## 4. Known dead ends",
                                         "### 4. M2 pipeline notes")
    v = validate_milestone(masquerade)
    assert any("missing required section: ## 4." in x for x in v)


def test_seed26_a_real_section_heading_still_passes():
    """The anchored check must not reject the legitimate heading shapes."""
    assert not any("missing required section" in x
                   for x in validate_milestone(MILESTONE_VALID))


def test_seed26_the_order_check_uses_the_anchored_offsets_too():
    """Swapping two real headings is still caught as out of order (the order
    comparison shares the presence offsets)."""
    swapped = MILESTONE_VALID.replace(
        "## 1. Problem contract (Math Spec)", "@@ONE@@").replace(
        "## 2. Decision DAG + phase order", "## 1. Problem contract (Math Spec)"
    ).replace("@@ONE@@", "## 2. Decision DAG + phase order")
    v = validate_milestone(swapped)
    assert any("out of order" in x for x in v)


# --- BUG-SEED-27: self_evolve's ratio=0 and chain-destination errors stay readable ----

from test_self_evolve import KERNEL, _task, chain, se  # noqa: E402, F401


def test_seed27_noise_floor_zero_ratio_raises_a_readable_error():
    """A zero confirmation ratio used to raise ZeroDivisionError after the whole
    generation had run; it must be a named, readable error instead."""
    with pytest.raises(ValueError, match="non-positive"):
        se.noise_floor({"ratios": [0.0, 1.02]})
    with pytest.raises(ValueError, match="non-positive"):
        se.noise_floor({"ratios": [1.02, -0.5]})


def test_seed27_main_reports_ratio_zero_as_a_readable_error(chain, capsys):
    root, ran, gains, _ = chain
    gains += [(1.3, [0.0, 1.02])]
    assert se.main([str(_task(root))]) == 2
    err = capsys.readouterr().err
    assert "non-positive" in err and "generation 1" in err


def test_seed27_chain_from_a_suffixed_task_continues_forward_numbering(chain):
    """Starting from rmsnorm-g3 the first jump must land on rmsnorm-g4, not on the
    ancestor rmsnorm-g2 that the bare generation counter produced."""
    root, ran, gains, kb = chain
    gains += [1.3, 1.01]
    first = _task(root, name="rmsnorm-g3")
    assert se.main([str(first), "--max-gen", "4"]) == 0
    assert (root / "tasks" / "rmsnorm-g4" / "reference" / "kernel.py").read_text() == KERNEL
    assert ran == [first.resolve(), root / "tasks" / "rmsnorm-g4"]
    assert not (root / "tasks" / "rmsnorm-g2").exists()


# --- BUG-SEED-36: creation sites validate the task/operator name ----------------------

from peregrine_kernel import launch  # noqa: E402
from peregrine_kernel import scaffold as scaffold_mod  # noqa: E402
from peregrine_kernel.task import check_new_task_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bad", ["../../deep-escape", "../escape", "a/b", "..", ".",
                                 "", "_private", "run", "init", "status"])
def test_seed36_check_new_task_name_rejects_traversal_reserved_and_invalid_names(bad):
    with pytest.raises(ValueError):
        check_new_task_name(bad)


@pytest.mark.parametrize("good", ["rmsnorm", "demo-op", "op_2"])
def test_seed36_check_new_task_name_accepts_normal_names(good):
    check_new_task_name(good)  # must not raise


def test_seed36_new_operator_rejects_a_traversal_name(tmp_path):
    before = set(tmp_path.iterdir()) if tmp_path.exists() else set()
    with pytest.raises(ValueError, match="not a task name"):
        scaffold_mod.new_operator("../../deep-escape", base_dir=tmp_path,
                                  templates_root=REPO_ROOT)
    assert set(tmp_path.iterdir()) == before          # nothing created anywhere


def test_seed36_new_operator_accepts_a_normal_name(tmp_path):
    created = scaffold_mod.new_operator("demo-op", base_dir=tmp_path,
                                        templates_root=REPO_ROOT)
    assert created and (tmp_path / "operators" / "demo-op" / "README.md").is_file()


def test_seed36_cmd_init_rejects_a_traversal_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    rc = launch.cmd_init(["../../deep-escape"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a task name" in err
    assert not (tmp_path.parent / "deep-escape").exists()
    assert not (tmp_path / "tasks" / ".." / "deep-escape").exists()


def test_seed36_cmd_init_accepts_a_normal_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks" / "_template").mkdir(parents=True)
    (tmp_path / "tasks" / "_template" / "task.toml").write_text("[task]\n")
    rc = launch.cmd_init(["demo-op"])
    assert rc == 0
    assert (tmp_path / "tasks" / "demo-op" / "task.toml").is_file()


def test_seed36_scaffold_cli_reports_a_bad_name_readably(capsys):
    rc = scaffold_mod.main(["../escape"])
    assert rc == 2
    assert "not a task name" in capsys.readouterr().out


# --- BUG-SEED-38: a failed git status no longer masquerades as "no changes" ------------

import subprocess  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402

from peregrine_kernel import taskcard  # noqa: E402


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True,
                   capture_output=True)


def test_seed38_commit_raises_when_git_status_fails(tmp_path, monkeypatch):
    """A failed `git status --porcelain` (empty stdout) used to read as 'nothing to
    commit' and silently skip the commit; it must be a loud failure instead."""
    _git_repo(tmp_path)
    real_run = subprocess.run

    def fake_run(argv, *a, **kw):
        if "status" in argv:
            return _NS(returncode=128, stdout="", stderr="fatal: bad object HEAD")
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(taskcard.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="git status failed"):
        taskcard._commit(tmp_path, "should never land")


def test_seed38_commit_still_commits_changes_and_skips_a_clean_tree(tmp_path):
    _git_repo(tmp_path)
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    taskcard._commit(tmp_path, "add f")

    def n_commits():
        log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"],
                             capture_output=True, text=True, check=True)
        return log.stdout.splitlines()

    assert len(n_commits()) == 1 and "add f" in n_commits()[0]
    taskcard._commit(tmp_path, "nothing to commit")     # clean tree: no-op, no error
    assert len(n_commits()) == 1


def test_seed38_kb_changes_reports_a_failed_git_status(tmp_path, monkeypatch, capsys):
    """rc != 0 used to be read as an empty change set; now it warns on stderr."""
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)   # not a git repo
    assert launch.kb_changes() == []
    err = capsys.readouterr().err
    assert "unknown, not clean" in err and "rc=" in err


def test_seed38_kb_changes_still_lists_real_changes(tmp_path, monkeypatch):
    _git_repo(tmp_path)
    (tmp_path / "hw_kb").mkdir()
    (tmp_path / "hw_kb" / "leak.md").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    # porcelain lists an untracked directory as the directory itself
    assert launch.kb_changes() == ["hw_kb/"]


# --- BUG-SEED-39: the _Tee run.log handle is closed when the drive ends ---------------

import gc as _gc  # noqa: E402
import warnings as _warnings  # noqa: E402


def test_seed39_entered_closes_the_run_log_handle_and_stays_reentrant(tmp_path):
    """Two drives in one process: each drive's tee file is closed on exit (no
    ResourceWarning from the finalizer) and the second drive reopens run.log."""
    run = _NS(operators_dir=tmp_path / "ops", log=tmp_path / "run.log")
    machine = _NS(env={})
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        for i in range(2):
            with launch._entered(run, machine):
                tee = sys.stdout
                assert isinstance(tee, launch._Tee)
                print(f"drive {i} output")
            assert tee._file.closed          # explicitly closed, not finalizer-dependent
            del tee
        _gc.collect()
    leaked = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaked, leaked
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "drive 0 output" in log and "drive 1 output" in log


def test_seed39_tee_close_is_idempotent(tmp_path):
    t = launch._Tee(sys.stdout, tmp_path / "log")
    t.close()
    t.close()                                # a repeated close must not raise
    assert t._file.closed


def test_seed39_nested_entered_does_not_close_the_outer_tee(tmp_path):
    """A nested _entered sees stdout already teed and must leave the outer tee (and
    its handle) working."""
    outer_log, inner_log = tmp_path / "outer.log", tmp_path / "inner.log"
    outer_run = _NS(operators_dir=tmp_path / "o", log=outer_log)
    inner_run = _NS(operators_dir=tmp_path / "i", log=inner_log)
    machine = _NS(env={})
    with launch._entered(outer_run, machine):
        with launch._entered(inner_run, machine):
            pass
        assert isinstance(sys.stdout, launch._Tee) and not sys.stdout._file.closed
    assert sys.stdout is sys.__stdout__ or not isinstance(sys.stdout, launch._Tee)


# --- BUG-SEED-40: a broken submit config is reported, not silently unstamped ----------

from peregrine_kernel import baseline_cli  # noqa: E402


def _patch_load_config(monkeypatch, effect):
    import peregrine_kernel.submit.config as submit_config
    monkeypatch.setattr(submit_config, "load_config", effect)


def test_seed40_a_broken_submit_config_warns_on_stderr(tmp_path, monkeypatch, capsys):
    def broken(op):
        raise ValueError("invalid [repo] delivery 'carrier-pigeon'")

    _patch_load_config(monkeypatch, broken)
    assert baseline_cli._submit_cluster("demo-op") == ""
    err = capsys.readouterr().err
    assert "submit config for demo-op is broken" in err and "ValueError" in err


def test_seed40_a_missing_submit_config_stays_silent(tmp_path, monkeypatch, capsys):
    def absent(op):
        raise FileNotFoundError("missing submit config")

    _patch_load_config(monkeypatch, absent)
    assert baseline_cli._submit_cluster("demo-op") == ""
    assert capsys.readouterr().err == ""


def test_seed40_a_code_bug_is_no_longer_swallowed(tmp_path, monkeypatch):
    def renamed(op):
        raise AttributeError("module has no attribute 'load_confg'")

    _patch_load_config(monkeypatch, renamed)
    with pytest.raises(AttributeError):
        baseline_cli._submit_cluster("demo-op")


def test_seed40_a_working_config_still_stamps_the_cluster(tmp_path, monkeypatch):
    _patch_load_config(monkeypatch, lambda op: _NS(cluster="b200-1"))
    assert baseline_cli._submit_cluster("demo-op") == "b200-1"


# --- BUG-SEED-44: a hung deliverable check keeps the session resumable ----------------

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402


def _goal_leg_module():
    spec = importlib.util.spec_from_file_location(
        "claude_goal_misc",
        Path(__file__).resolve().parents[1] / "scripts" / "claude-goal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed44_a_hung_deliverable_check_keeps_the_session_resumable(
        tmp_path, monkeypatch, capsys):
    """The deliverable check's TimeoutExpired used to escape run(): the finally killed
    the tmux session first and save_session never ran, freezing session.json at
    launch's resumable:false. Now the leg ends rc=124 with a readable error and the
    session marker stays resumable."""
    mod = _goal_leg_module()
    td = tmp_path / "turn"
    td.mkdir()
    (td / "spec.json").write_text(json.dumps({
        "kind": "caller", "role": "student", "reply": str(td / "reply.md"),
        "check": "true", "condition": "PEREGRINE TURN. unit fixture."}), encoding="utf-8")
    leg = mod.GoalLeg("claude-test-1", "test-model", [], "do the thing", env={
        "PK_GOAL_TURN_DIR": str(td),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "cfg"),
        "PK_CLAUDE_BIN": "/usr/bin/true",
        "PK_GOAL_TMUX_SOCKET": f"fkgoal-misc-{os.getpid()}",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    })

    def fake_launch():
        leg.sid = "unit-sid-1234"

    def hung(tr):
        raise subprocess.TimeoutExpired(cmd=leg.spec["check"], timeout=300)

    monkeypatch.setattr(leg, "launch", fake_launch)
    monkeypatch.setattr(leg, "watch", hung)
    rc = leg.run()
    assert rc == 124
    st = json.loads((td / "session.json").read_text(encoding="utf-8"))
    assert st["resumable"] is True and st["sid"] == "unit-sid-1234"
    assert st["why"] == "check_timeout"
    err = capsys.readouterr().err
    assert "deliverable check hung" in err
    res = json.loads((td / "result.json").read_text(encoding="utf-8").splitlines()[-1])
    assert res["outcome"] == "check_timeout" and res["rc"] == 124
