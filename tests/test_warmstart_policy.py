"""The warmstart side of the agent policy (loop/agent_policy.py): what a run's agents may
read and where the student's network comes from.

Three claims are pinned here. (1) The existing implementations live in the card's ``prior/``,
inside the student's workspace, so no rule denies them to either role -- the policy names
them to the agents instead. (2) The student's network is the task's choice: online, the
network group of shell denials is lifted while the GPU-machine CLIs stay denied; offline,
today's behaviour is unchanged. (3) The local knowledge trees are readable and a deny entry
that would shadow one is reported rather than silently enforced."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from rsi_kernel_tune import launch, taskcard
from rsi_kernel_tune.loop import agent_policy as ap
from rsi_kernel_tune.task import TaskError, load_task


def _cfg(tmp_path: Path, **loop) -> dict:
    lc = {"ref_source_paths": [], "ref_source_urls": []}
    lc.update(loop)
    return {"loop": lc}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """A student workspace with a prior/ directory (as the card's initial commit has)."""
    workspace = tmp_path / "run" / "operators" / "op-ws"
    (workspace / "prior").mkdir(parents=True)
    (workspace / "prior" / "kernel.py").write_text("import torch\n\ndef run(a):\n    return a\n")
    (workspace / "prior" / "SOURCES.json").write_text('{"sources": []}')
    (workspace / "kernel.py").write_text("import torch\n")
    return workspace


def _policy(tmp_path: Path, ws: Path, **loop) -> ap.AgentPolicy:
    return ap.from_config(_cfg(tmp_path, prior_dir=str(ws / "prior"), **loop),
                          oracle_mode="source", repo_root=tmp_path / "run",
                          teacher_trees=[])


# -- prior/ is readable, and said so --------------------------------------------------

@pytest.mark.parametrize("role", ["student", "teacher"])
def test_prior_is_not_on_any_deny_list(tmp_path, ws, role):
    pol = _policy(tmp_path, ws)
    prior = (ws / "prior").resolve()
    assert all(prior != p and p not in prior.parents for p in pol.deny_read(role))
    s = pol.settings(role, ws)
    deny = s["permissions"]["deny"]
    assert not any(str(prior) in rule for rule in deny)
    assert all(str(prior) not in p for p in s["sandbox"]["filesystem"]["denyRead"])


@pytest.mark.parametrize("role", ["student", "teacher"])
def test_the_notice_points_at_the_prior_directory(tmp_path, ws, role):
    note = _policy(tmp_path, ws).notice(role)
    assert "Existing implementations" in note and str(ws / "prior") in note
    assert "Read, reuse and improve them" in note


def test_a_deny_entry_shadowing_a_knowledge_tree_is_reported(tmp_path, ws, capsys):
    trees = taskcard.local_knowledge_dirs()
    if not trees:
        pytest.skip("no _reference/ trees on this machine")
    pol = ap.from_config(_cfg(tmp_path, ref_source_paths=[str(trees[0].parent)],
                              local_knowledge_dirs=[str(t) for t in trees]),
                         oracle_mode="source", repo_root=tmp_path / "run", teacher_trees=[])
    denied = pol.settings("student", ws)["sandbox"]["filesystem"]["denyRead"]
    assert str(trees[0].parent) in denied          # the declaration stands
    assert "is under the denied path" in capsys.readouterr().err
    assert str(trees[0]) in pol.notice("student")  # and the contradiction is visible


# -- the student's network ------------------------------------------------------------

def test_an_online_student_loses_the_network_denials_but_not_the_gpu_machine(tmp_path, ws):
    pol = _policy(tmp_path, ws, student_online=True)
    assert pol.offline("student") is False
    s = pol.settings("student", ws)
    deny = s["permissions"]["deny"]
    assert "WebFetch" not in deny and "WebSearch" not in deny
    assert "Bash(curl:*)" not in deny and "Bash(pip install:*)" not in deny
    assert "network" not in s["sandbox"]
    assert "Bash(ssh:*)" in deny and "Bash(scp:*)" in deny
    assert "Bash(python -m rsi_kernel_tune.submit:*)" in deny
    assert pol.policy_document("student")["no_cluster"] is True
    note = pol.notice("student")
    assert "You may go online" in note and "No network" not in note
    assert "licence that permits it" in note


def test_an_offline_student_is_exactly_the_pre_warmstart_policy(tmp_path, ws):
    pol = _policy(tmp_path, ws, student_online=False)
    assert pol.offline("student") is True
    s = pol.settings("student", ws)
    deny = s["permissions"]["deny"]
    assert {"WebFetch", "WebSearch", "WebFetch(domain:*)"} <= set(deny)
    assert "Bash(curl:*)" in deny and "Bash(ssh:*)" in deny
    assert s["sandbox"]["network"]["deniedDomains"] == ["*"]
    assert "No network this turn" in pol.notice("student")


def test_the_teacher_is_online_either_way(tmp_path, ws):
    for value in (True, False):
        pol = _policy(tmp_path, ws, student_online=value)
        assert pol.offline("teacher") is False
        assert "WebFetch" not in pol.settings("teacher", ws)["permissions"]["deny"]


# -- the declaration ----------------------------------------------------------------

def test_declaration_problems_catch_bad_warmstart_keys(tmp_path):
    good = _cfg(tmp_path, student_online=True, prior_dir="/tmp/prior",
                prior_manifest="/tmp/prior.md", local_knowledge_dirs=["/tmp/kb"])
    assert ap.declaration_problems(good, ap.ORACLE_BLACKBOX) == [] or \
        all("ref_source" not in p for p in ap.declaration_problems(good, ap.ORACLE_BLACKBOX))
    bad = _cfg(tmp_path, student_online="yes", prior_dir=7, local_knowledge_dirs="kb")
    problems = ap.declaration_problems(bad, "source")
    assert any("student_online must be true or false" in p for p in problems)
    assert any("prior_dir must be a path" in p for p in problems)
    assert any("local_knowledge_dirs must be a list" in p for p in problems)


# -- task.toml [agent] online -------------------------------------------------------

def _kb(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "hw_kb" / "cutedsl" / "sm100").mkdir(parents=True, exist_ok=True)
    return root


def _task(tmp_path: Path, toml: str) -> Path:
    d = tmp_path / "tasks" / "op"
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n' + toml)
    (d / "description.md").write_text("y = x\n")
    (d / "shapes.json").write_text(json.dumps([{"name": "s", "axes": {"m": 4}}]))
    return d


def test_online_defaults_to_true_and_can_be_turned_off(tmp_path):
    t = load_task(_task(tmp_path, ""), kb_root=_kb(tmp_path))
    assert t.student.online is True and t.teacher.online is True
    t = load_task(_task(tmp_path, '[agent]\nonline = false\n'), kb_root=_kb(tmp_path))
    assert t.student.online is False
    t = load_task(_task(tmp_path, '[agent.student]\nonline = false\n'), kb_root=_kb(tmp_path))
    assert t.student.online is False and t.teacher.online is True


def test_an_invalid_online_value_is_refused(tmp_path):
    with pytest.raises(TaskError, match="online must be true or false"):
        load_task(_task(tmp_path, '[agent]\nonline = "yes"\n'), kb_root=_kb(tmp_path))


def test_describe_shows_whether_the_student_is_online(tmp_path):
    class Machine:
        def describe(self) -> str:
            return "a fake GPU"

    online = load_task(_task(tmp_path, ""), kb_root=_kb(tmp_path))
    assert "student claude-opus-5-5 (high) [online]" in launch.describe(online, Machine())
    offline = load_task(_task(tmp_path, '[agent]\nonline = false\n'), kb_root=_kb(tmp_path))
    assert "student claude-opus-5-5 (high) [offline]" in launch.describe(offline, Machine())
    # the teacher default is Codex-first; the student keeps Claude for offline/hook policy
    assert online.teacher.backend == "codex" and online.teacher.model == "sol6.2"
    assert online.student.backend == "claude" and online.student.model == "claude-opus-5-5"
    assert "teacher sol6.2 (high)" in launch.describe(online, Machine())


def test_the_loop_toml_of_a_plain_task_carries_student_online(tmp_path):
    t = load_task(_task(tmp_path, '[agent]\nonline = false\n'), kb_root=_kb(tmp_path))
    lines = taskcard.context_lines(tmp_path / "card", t)
    assert lines[0] == "student_online = false"
    # no card -> no prior dir, no manifest, and the machine's knowledge trees if any
    assert not any(line.startswith(("prior_dir", "prior_manifest")) for line in lines)
    assert tomllib.loads("[loop]\n" + "\n".join(lines))["loop"]["student_online"] is False
