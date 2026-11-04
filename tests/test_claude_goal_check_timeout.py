"""BUG-SEED-44 — a hung deliverable check used to blow the goal leg up.

deliverable() ran the caller's check with timeout=300 and no handler: a check
that outlived its budget raised subprocess.TimeoutExpired straight through
watch() — run() only catches Interrupted — so the leg died in a traceback before
save_session and session.json kept its launch-time state ("running",
resumable=False). The agent session was fine; the next attempt of the turn
started over from zero instead of resuming. The check timing out is now just a
failed check: deliverable() returns (False, why) and the watch loop's existing
false-met / deadline handling finishes the leg, writing a fresh session.json.

No tmux needed here: the leg's tmux face is stubbed at its single subprocess
entry (Tmux.run), which is exactly the layer the fix sits above.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("claude_goal_timeout", _ROOT / "scripts" / "claude-goal.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _turn_dir(tmp_path: Path) -> Path:
    td = tmp_path / "turn"
    td.mkdir()
    (td / "spec.json").write_text(json.dumps(
        {"kind": "caller", "role": "student", "reply": str(td / "reply.md"),
         "check": "true", "condition": "PEREGRINE TURN. Deliver the reply."}))
    return td


def _leg(tmp_path: Path, td: Path, **env) -> "m.GoalLeg":
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    return m.GoalLeg("claude-test-1", "test-model", [], "prompt", env={
        "PK_GOAL_TURN_DIR": str(td), "CLAUDE_CONFIG_DIR": str(cfg), **env})


def _boom(*_a, **_k) -> None:
    raise subprocess.TimeoutExpired(cmd="check", timeout=300)


def test_a_hung_check_is_a_failed_check_not_a_crash(tmp_path, monkeypatch):
    leg = _leg(tmp_path, _turn_dir(tmp_path))
    leg.setup_turn()
    monkeypatch.setattr(m.subprocess, "run", _boom)
    assert leg.deliverable() == (False, "the deliverable check timed out after 300s")


def test_a_hung_check_at_the_deadline_still_finishes_the_leg_and_writes_the_session(
        tmp_path, monkeypatch):
    # the deadline is already due, so the first watch() iteration runs the check:
    # it must come back as a failed check and the leg must land in its normal
    # shutdown path — session.json freshly written by save_session, result.json
    # with the outcome — instead of dying before either.
    td = _turn_dir(tmp_path)
    leg = _leg(tmp_path, td, PK_GOAL_DEADLINE=str(time.time() - 100))
    monkeypatch.setattr(m.subprocess, "run", _boom)
    monkeypatch.setattr(m.time, "sleep", lambda _s: None)   # launch's pidfile wait
    monkeypatch.setattr(leg.tmux, "run",
                        lambda *a, **k: subprocess.CompletedProcess((), 0))
    monkeypatch.setattr(leg.tmux, "start", lambda *a, **k: None)

    rc = leg.run()                                   # raised TimeoutExpired before the fix
    assert rc == 124
    st = json.loads((td / "session.json").read_text())
    assert st["why"] == "timeout" and st["resumable"] is False
    res = json.loads((td / "result.json").read_text().splitlines()[-1])
    assert res["outcome"] == "timeout"
    assert not leg.rundir.exists()                   # the normal cleanup ran
