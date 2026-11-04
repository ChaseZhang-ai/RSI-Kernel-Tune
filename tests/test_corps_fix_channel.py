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
from peregrine_kernel.loop.cli import RunnerLostError  # noqa: E402
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


# --- BUG-SEED-18: the retry budgets say what they spend ------------------------------

def _lost_backend(monkeypatch, tmp_path, *, lost_retries):
    """A backend whose every job comes back Lost (no runner, no verdict)."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    def fake_forge(op, verb, *a, **k):
        calls.append(verb)
        if verb == "submit":
            return type("P", (), {"returncode": 0,
                                  "stdout": f"[submit] job={len(calls)}\n"})()
        if verb == "status":
            return type("P", (), {"returncode": 0,
                                  "stdout": '{"id": 1, "status": "Lost"}\n'})()
        return type("P", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(cli, "_forge", fake_forge)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0,
                         lost_retries=lost_retries)
    return be, calls


def test_the_lost_budget_spends_its_extra_submits_then_stops(monkeypatch, tmp_path, capsys):
    """lost_retries reads "extra submits": the same value as clock_lock_retries buys
    one MORE attempt on this path (lost_retries + 1 total), and the run log never
    prints a count past the budget (the old "(4/3)" overflow)."""
    be, calls = _lost_backend(monkeypatch, tmp_path, lost_retries=1)
    with pytest.raises(RunnerLostError):
        be.submit_gpu("br", "all")
    assert calls.count("submit") == 2          # initial + 1 extra submit
    out = capsys.readouterr().out
    assert "extra submit 1/1" in out
    assert "/1)" not in out.replace("extra submit 1/1", ""), \
        "a lost count past the budget was printed"


def test_the_clock_lock_budget_prints_its_attempt_count(monkeypatch, tmp_path, capsys):
    """The resubmit line names the ATTEMPT about to be spent (an initial submit plus
    clock_lock_retries - 1 resubmits), not a resubmit number that over-reads it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    logs = iter([jobs.CLOCK_LOCK_REFUSED_MARKER + " lock not held\n"] * 5)
    seen = []

    def once(branch, mode):
        seen.append(mode)
        return next(logs)

    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    be = cli.RealBackend("op", ws, None, "t", "t@t")
    be.clock_lock_retry_sleep_s = 0
    monkeypatch.setattr(be, "_submit_gpu_once", once)
    with pytest.raises(cli.ClockLockRefusedError):
        be.submit_gpu("b", "all")
    assert len(seen) == be.clock_lock_retries  # attempts semantics, as ever
    out = capsys.readouterr().out
    assert "attempt 2 of 3" in out and "attempt 3 of 3" in out
    assert "attempt 4" not in out


# --- BUG-SEED-48: the retry budget does not sleep past its last attempt --------------

def _sleeps_recorder(monkeypatch):
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    return sleeps


def test_a_failing_retry_budget_never_sleeps_after_its_last_attempt(monkeypatch):
    """Three failing attempts with a 10 s base sleep cost two backoffs, not three: the
    old unconditional tail sleep added the last multiplier after the final failure."""
    sleeps = _sleeps_recorder(monkeypatch)
    be = cli.RealBackend("op", Path("/ws"), None, "n", "e", git_retries=3,
                         git_retry_sleep=10.0)
    proc = type("P", (), {"returncode": 1, "stderr": "boom"})()
    with pytest.raises(BackendError):
        be._retry_call(lambda: proc, what="submit all")
    assert sleeps == [10.0, 20.0]


def test_a_call_that_ran_to_its_own_timeout_is_not_retried(monkeypatch):
    """A hung submit comes back as one 600 s timeout, not three of them plus backoff
    (~31 minutes before this could say anything)."""
    import subprocess as sp

    sleeps = _sleeps_recorder(monkeypatch)
    be = cli.RealBackend("op", Path("/ws"), None, "n", "e", git_retries=3,
                         git_retry_sleep=10.0)
    calls = []

    def fn():
        calls.append(1)
        raise sp.TimeoutExpired("cmd", 600)

    with pytest.raises(BackendError, match="timed out"):
        be._retry_call(fn, what="submit all")
    assert calls == [1] and sleeps == []
