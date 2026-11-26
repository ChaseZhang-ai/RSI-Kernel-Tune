"""Wiring of the throughput governor: loop/cli.py wraps both backends, and
launch.run_loop hands the [budget] policy to the loop as FK_BUDGET_POLICY."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_kernel_tune import launch
from rsi_kernel_tune.loop import cli as cli_mod
from rsi_kernel_tune.loop import governor as gov
from rsi_kernel_tune.loop.driver import AGENT_ROLE_TEACHER
from rsi_kernel_tune.task import Budget


class Inner:
    def __init__(self):
        self.prompts: list[str] = []

    def run_agent(self, prompt: str, *, role: str | None = None) -> str:
        self.prompts.append(prompt)
        return "ok"


def test_the_cli_helper_wraps_a_loop_backend(tmp_path, monkeypatch):
    monkeypatch.delenv(gov.DISABLE_ENV, raising=False)
    status = tmp_path / "student" / "status.jsonl"
    status.parent.mkdir(parents=True)
    status.write_text("", encoding="utf-8")
    policy = gov.policy_from_env(hours=2.0, status_path=status, environ={})
    inner = Inner()
    be = cli_mod._govern_backend(inner, policy=policy, status_path=status)
    assert isinstance(be, gov.GovernedBackend)
    # ... and the ledger is the one next to status.jsonl
    be.run_agent("# Meta-Agent — Diagnosis step (operator-agnostic prompt)\n",
                 role=AGENT_ROLE_TEACHER)
    assert (status.parent / gov.LEDGER_NAME).is_file()


def test_the_cli_helper_leaves_the_backend_alone_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(gov.DISABLE_ENV, "off")
    status = tmp_path / "student" / "status.jsonl"
    policy = gov.policy_from_env(hours=2.0, status_path=status, environ=os.environ)
    assert policy is None
    inner = Inner()
    assert cli_mod._govern_backend(inner, policy=policy, status_path=status) is inner


def test_both_real_backends_in_loop_cli_go_through_the_governor():
    """The two construction sites the task names (the loop backend and the reconnaissance
    backend) both route the new object through _govern_backend."""
    src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    assert src.count("= _govern_backend(") == 2
    assert "backend = _govern_backend(backend" in src
    assert "recon_backend = _govern_backend(recon_backend" in src


def _task(**budget) -> SimpleNamespace:
    values = dict(hours=2.0, max_meta=1, reference_hours=1.0)
    values.update(budget)
    spec = SimpleNamespace(backend="claude", provider="native", model="m",
                           effort=None, timeout_s=1)
    return SimpleNamespace(name="demo", budget=Budget(**values),
                           target=SimpleNamespace(ms=1.5, ratio=None),
                           student=spec, teacher=spec)


def test_run_loop_hands_the_policy_to_the_loop_and_restores_the_environment(
        tmp_path, monkeypatch):
    from rsi_kernel_tune.loop import cli as loop_cli
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        seen["env"] = os.environ.get(gov.POLICY_ENV)
        return 0

    monkeypatch.setattr(loop_cli, "main", fake_main)
    monkeypatch.delenv(gov.POLICY_ENV, raising=False)
    run = SimpleNamespace(ws=tmp_path / "ws")
    task = _task(teacher_share=0.25, min_student_iters=3,
                 max_teacher_turns_per_phase=4)
    assert launch.run_loop(run, task, resuming=False, harvest_only=False) == 0
    payload = json.loads(seen["env"])
    assert payload == {"hours": 2.0, "teacher_share": 0.25, "min_student_iters": 3,
                       "max_teacher_turns_per_phase": 4}
    assert gov.POLICY_ENV not in os.environ       # restored, not leaked

    # an existing value is put back, not dropped
    monkeypatch.setenv(gov.POLICY_ENV, "previous")
    assert launch.run_loop(run, task, resuming=False, harvest_only=False) == 0
    assert os.environ[gov.POLICY_ENV] == "previous"


def test_run_loop_restores_the_environment_even_when_the_loop_raises(
        tmp_path, monkeypatch):
    from rsi_kernel_tune.loop import cli as loop_cli

    def boom(argv):
        raise RuntimeError("loop blew up")

    monkeypatch.setattr(loop_cli, "main", boom)
    monkeypatch.delenv(gov.POLICY_ENV, raising=False)
    run = SimpleNamespace(ws=tmp_path / "ws")
    with pytest.raises(RuntimeError, match="loop blew up"):
        launch.run_loop(run, _task(), resuming=False, harvest_only=False)
    assert gov.POLICY_ENV not in os.environ


def test_policy_from_env_matches_what_the_launch_writes(tmp_path, monkeypatch):
    task = _task(hours=3.0, teacher_share=0.4, min_student_iters=2,
                 max_teacher_turns_per_phase=6)
    status = tmp_path / "status.jsonl"
    monkeypatch.setattr(cli_mod.governor_mod, "policy_from_env", gov.policy_from_env)
    payload = json.dumps({"hours": task.budget.hours,
                          "teacher_share": task.budget.teacher_share,
                          "min_student_iters": task.budget.min_student_iters,
                          "max_teacher_turns_per_phase":
                              task.budget.max_teacher_turns_per_phase})
    policy = gov.policy_from_env(hours=99.0, status_path=status,
                                 environ={gov.POLICY_ENV: payload})
    assert policy.hours == 3.0 and policy.teacher_share == 0.4
    assert policy.min_student_iters == 2 and policy.max_teacher_turns_per_phase == 6
