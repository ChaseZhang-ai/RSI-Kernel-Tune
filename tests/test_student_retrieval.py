# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The retrieval permission (rsi-run.toml ``[student] retrieval``).

"off" (the default) is the L0 contract -- optimize from scratch -- byte for
byte: no flag, no policy change, no prompt text. "on" lets the student survey
public sources first (the existing online-student channel: WebFetch / WebSearch
present, the sandbox open, the notice switched), while every deny list stays
exactly as it was -- the reference implementation itself, and any optimized
implementation of it, remain unreadable. Retrieval is not the answer.
Runs on CPU.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_kernel_tune.loop import agent_policy as agent_policy_mod
from rsi_kernel_tune.loop import driver as driver_mod

from test_task import _kb, _task  # noqa: E402  (fixtures)


# -- the policy: on opens the network channel, nothing else ---------------------------


def _policy(**kw) -> agent_policy_mod.AgentPolicy:
    base = dict(oracle_mode="blackbox", repo_root=Path("/repo"))
    base.update(kw)
    return agent_policy_mod.AgentPolicy(**base)


def test_default_student_is_offline():
    p = _policy()
    assert p.offline("student") is True
    assert p.offline("teacher") is False


def test_retrieval_on_switches_the_student_online():
    p = _policy(student_online=True)
    assert p.offline("student") is False
    assert "WebFetch" not in p.settings("student", Path("/ws"))["permissions"]["deny"] or \
        "WebFetch" not in p.settings("student", Path("/ws"))["permissions"]["deny"][:1]


def test_retrieval_on_widens_no_deny_list(tmp_path):
    """The whole point: the reference's paths and repositories stay denied to an
    online student (retrieval != the answer)."""
    ref = tmp_path / "ref" / "src"
    ref.mkdir(parents=True)
    offline = _policy(ref_source_paths=(ref,), ref_source_urls=("github.com/org/RefLib",))
    online = _policy(ref_source_paths=(ref,), ref_source_urls=("github.com/org/RefLib",),
                     student_online=True)
    assert offline.deny_read("student") == online.deny_read("student")
    assert offline.deny_urls("student") == online.deny_urls("student") \
        == ["github.com/org/RefLib"]
    # and the online notice says code online must not be copied
    notice = online.notice("student")
    assert "must not be copied" in notice


# -- the driver: the injection ---------------------------------------------------------


def _ctx(**kw) -> SimpleNamespace:
    base = dict(compute_dtype="fp32", retrieval=False, memo_lane=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_retrieval_note_absent_by_default():
    assert driver_mod._retrieval_note(_ctx()) == ""
    assert driver_mod._retrieval_note(_ctx(retrieval=False)) == ""


def test_retrieval_note_is_small_and_names_the_limits():
    note = driver_mod._retrieval_note(_ctx(retrieval=True))
    assert "Public-source retrieval is allowed" in note
    # retrieval != the answer: the limits ride the injection
    assert "reference implementation" in note
    assert note.count("\n") <= 4          # one header + two lines


# -- the launcher: the flag rides only when permitted ---------------------------------


def test_run_loop_adds_the_flag_only_when_env_permits(tmp_path, monkeypatch):
    """RKT_STUDENT_RETRIEVAL=1 -> --student-retrieval; unset (the default) adds
    no flag at all (L0 byte-identical)."""
    from rsi_kernel_tune import launch
    from rsi_kernel_tune.loop import cli as loop_cli
    from rsi_kernel_tune.task import load_task
    calls = []
    monkeypatch.setattr(launch, "agent_command", lambda spec, role: f"agent-{role}")
    monkeypatch.setattr(loop_cli, "main", lambda argv: calls.append(argv) or 0)
    t = load_task(_task(tmp_path, '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                   '[reference]\nrun = "reference/run.py:run"\n'
                                   '[target]\nratio = 1.1\n',
                        files={"reference/run.py": "def run(x, w):\n    return x\n"}),
                  kb_root=_kb(tmp_path))
    run = SimpleNamespace(ws=tmp_path / "ws")
    monkeypatch.setenv("RKT_STUDENT_RETRIEVAL", "1")
    launch.run_loop(run, t, resuming=False, harvest_only=False)
    assert "--student-retrieval" in calls[-1]
    monkeypatch.delenv("RKT_STUDENT_RETRIEVAL")
    launch.run_loop(run, t, resuming=False, harvest_only=False)
    assert "--student-retrieval" not in calls[-1]
