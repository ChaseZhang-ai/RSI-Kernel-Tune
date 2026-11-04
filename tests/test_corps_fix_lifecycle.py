"""Regression tests for the audit-corps fix wave 5 (the launch/monitor/resume/agent-policy
lifecycle surface): the run verdict never reads DONE out of the stop word, the
correctness monitor reports a hook that never injected and recovers a cut-short log,
the reopen plan keeps companion pairs together, the agent policy sees every agent
command, the monitor hook leaves a trace when it cannot write, and the policy hook
fails closed on an unreadable event."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_launch import _confirmed, _loop_ends, _start, _task, _Teacher  # noqa: E402

from peregrine_kernel import launch, monitor  # noqa: E402
from peregrine_kernel.runs import list_runs  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch place for runs and locks; the harness itself stays the real repository."""
    monkeypatch.setenv("PEREGRINE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PEREGRINE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("PEREGRINE_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("PK_DELIVERY", "local")
    monkeypatch.setenv("PK_GPU", "0")
    monkeypatch.setattr(launch, "probe_ncu", lambda task, op_dir: True)
    return tmp_path


# --- BUG-SEED-13: a correctness alarm (or a missing confirmation) on a target stop ----
# must end the run stopped, not done ---------------------------------------------------


def _drive_to_the_end(repo, monkeypatch, *, stop, confirm):
    """`peregrine new` + `peregrine run` with the loop stood in: it ends with ``stop``
    and ``confirm`` as the final confirmation exactly as given (no stand-in defaults)."""
    from peregrine_kernel.loop import cli as loop_cli
    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(stop=stop, confirm=confirm))
    return _start(task_dir), list_runs()[0]


def test_a_target_stop_with_a_correctness_alarm_is_not_done(repo, monkeypatch):
    """The loop stopped at the target, but the confirmed kernel's correctness monitor
    raised an alarm: `peregrine run` exits 2 and the run is stopped — DONE requires a
    confirmation with no alarm, whatever the stop word says."""
    rc, run = _drive_to_the_end(repo, monkeypatch, stop="target",
                                confirm=_confirmed(8.0, monitor="alarm"))
    assert rc == 2
    rec = run.record()
    assert rec["status"] == "stopped" and rec["stop"] == "target"
    assert rec["target_met"] is False


def test_a_target_stop_on_a_confirmed_best_is_done(repo, monkeypatch):
    """The mirror side of the contract: a target stop whose final confirmation meets
    the target with no alarm still ends done (the stop word alone decides nothing,
    either way)."""
    rc, run = _drive_to_the_end(repo, monkeypatch, stop="target", confirm=_confirmed(8.0))
    assert rc == 0
    rec = run.record()
    assert rec["status"] == "done" and rec["stop"] == "target" and rec["target_met"] is True


def test_a_target_stop_without_any_confirmation_is_not_done(repo, monkeypatch):
    """A target stop with no best_confirmation.json at all: `met` is false, so the run
    ends stopped instead of reading DONE out of the stop word."""
    from peregrine_kernel.loop import cli as loop_cli

    def main(argv):
        from peregrine_kernel.loop.cli import SUMMARY_FILE
        d = Path(os.environ["PEREGRINE_OPERATORS_DIR"]) / argv[0]
        (d / SUMMARY_FILE).write_text(json.dumps({"stop": "target"}))
        return 0

    task_dir = _task(repo)
    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", main)
    assert _start(task_dir) == 2
    rec = list_runs()[0].record()
    assert rec["status"] == "stopped" and rec["stop"] == "target" and rec["target_met"] is False


# --- BUG-SEED-14: a hook that never injected is reported, not misattributed -----------


def test_classify_reports_a_hook_that_never_injected():
    """Zero hook-<pid>.json files means sitecustomize never ran: the input reshaping and
    the output checks did not execute, so the verdict is a warn naming the injection —
    not an ok that explains the silence as a bench that never imports the kernel."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True, "hook": {"processes": 0, "calls": []}},
        {"profile": "seed1", "card_pass": True, "hook": {"processes": 0, "calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "warn"
    assert v["ran"] == [] and v["not_applicable"] == []
    whats = [f["what"] for f in v["findings"]]
    assert sum("did not inject" in w for w in whats) == 2
    assert not any("does not import the kernel" in w for w in whats)
    assert not any("did not change" in w for w in whats)


def test_classify_reports_one_profile_the_hook_never_injected():
    """A profile whose bench ran but whose hook did not inject is dropped from the
    counted profiles and named, while the injected base stays counted."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True,
         "hook": {"processes": 1, "calls": [{"fn": "kernel", "i": 0, "outputs": []}]}},
        {"profile": "seed1", "card_pass": True, "hook": {"processes": 0, "calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "warn" and v["ran"] == ["base"]
    assert any(f["profile"] == "seed1" and f["severity"] == "warn"
               and "did not inject" in f["what"] for f in v["findings"])


def test_classify_keeps_the_old_reading_for_records_without_a_process_count():
    """Records from before the runner counted its hook files carry no ``processes``:
    a call-less base stays the ordinary info finding, not an injection failure."""
    result = {"complete": True, "profiles": [
        {"profile": "base", "card_pass": True, "hook": {"calls": []}},
    ]}
    v = monitor.classify(result)
    assert v["status"] == "ok" and v["ran"] == ["base"]
    assert any("no kernel call was observed" in f["what"] for f in v["findings"])
