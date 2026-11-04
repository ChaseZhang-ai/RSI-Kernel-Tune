"""Regression tests for the audit-corps fix wave 4: the process-life-cycle, ssh job
channel, and cli-miscellaneous surface.

Each test names the BUG it pins: SEED-10 (an interrupted agent turn must take its whole
process group down), SEED-18 (the clock-lock retry budget's docs match its attempts
count), SEED-48 (a failed retry budget does not sleep after its last attempt, and a
call that ran to its own timeout is not retried), SEED-21 (a poll that never reaches a
terminal state counts toward the in-flight give-up), SEED-19 (a harness overlay build
failure refuses the reading instead of silently re-judging with the lineage's own
copies), C3-02 (reference label keys do not read as a declared reference), SEED-20 (the
remote job tree carries the op tree's identity, not just its basename), SEED-11 (the
ssh job deadline starts when the job begins, not while it queues for the host's GPU
lock), SEED-12 (an ssh drop after the job began is Lost and ends the remote session),
and T2C-02 (the ssh extract step is bounded and rides the transport channel).
"""
import hashlib
import json as _json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.loop import cli  # noqa: E402
from peregrine_kernel.loop.driver import BackendError  # noqa: E402
from peregrine_kernel.submit import direct, jobs  # noqa: E402


# --- BUG-SEED-10: an interrupted agent turn ends its whole process group ------------


def test_an_interrupted_agent_turn_ends_its_whole_process_group(tmp_path, monkeypatch):
    """The turn is a shell=True Popen; proc.kill() killed only the shell, and the agent
    (the leg script's grandchild) survived as an orphan spending API usage. The turn
    must run in a session of its own and its interrupt path must end that GROUP."""
    real_popen = subprocess.Popen
    holder = {}

    def popen(cmd, **kw):
        proc = real_popen(cmd, **kw)
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(cli.subprocess, "Popen", popen)

    class _NeverPump:
        """Stands in for the pump thread: the turn stays in its heartbeat loop until
        the interrupt lands."""

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt      # the interrupt lands on the caller's thread
            return None

    monkeypatch.setattr(cli.threading, "Thread", lambda *a, **k: _NeverPump())
    be = cli.RealBackend("op", tmp_path, None, "t", "t@t")
    be.agent_heartbeat_s = 0        # join(timeout=None): the interrupt lands on beat 1
    # The command's leg: shell -> bash -> sleep. The sleep is the "agent" grandchild
    # a bare proc.kill() leaves behind.
    with pytest.raises(KeyboardInterrupt):
        be._agent_turn("bash -c 'sleep 47.3 & wait'", "prompt", "student")
    assert holder["proc"].poll() is not None
    left = subprocess.run(["pgrep", "-f", r"sleep 47\.3"], capture_output=True,
                          text=True).stdout
    assert left.strip() == "", "the agent grandchild outlived the interrupted turn"


def test_agent_turn_start_new_session_is_passed(tmp_path, monkeypatch):
    """The group terminator assumes Popen(start_new_session=True) built the group: the
    turn's Popen must actually pass it."""
    seen = {}
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        seen.update(kw)
        return real_popen("true", **kw)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.threading, "Thread", lambda *a, **k:
                        type("_P", (), {"start": lambda self: None,
                                        "is_alive": lambda self: False,
                                        "join": lambda self, timeout=None: None})())
    be = cli.RealBackend("op", tmp_path, None, "t", "t@t")
    be._agent_turn("true", "prompt", "student")
    assert seen.get("start_new_session") is True
