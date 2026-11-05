"""Regression tests for the audit wave FIX-R3E (launch): a harvest must never submit a
GPU job (BUG-CORPS-C-01), and a failure while finalizing a run must still land a
terminal status (BUG-CORPS-C-02)."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from rsi_kernel_tune import judge as judge_mod
from rsi_kernel_tune import launch
from rsi_kernel_tune.localenv import load_machine
from rsi_kernel_tune.runs import (STATUS_DONE, STATUS_FAILED, STATUS_STOPPED,
                                   new_run)
from rsi_kernel_tune.task import load_task


DESCRIPTION = "y = x / sqrt(mean(x^2) + 1e-6) * w\n"
SHAPES = [{"name": "small", "axes": {"m": 4, "n": 16}}]
REF_TOML = ('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
            '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n')
REF_PY = "import torch\n\ndef run(x, w):\n    return x * w\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI-KERNEL-TUNE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RSI-KERNEL-TUNE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("RSI-KERNEL-TUNE_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("RKT_DELIVERY", "local")
    monkeypatch.setenv("RKT_GPU", "0")
    return tmp_path


def _run(repo):
    """A created run of a small reference task, with the launch preamble and the loop
    stood in (no teacher, no GPU): ``calls`` records everything the preamble did."""
    d = repo / "tasks" / "rmsnorm"
    (d / "reference").mkdir(parents=True)
    (d / "repo" / "hw_kb" / "cutedsl" / "sm100").mkdir(parents=True)
    (d / "task.toml").write_text(REF_TOML)
    (d / "description.md").write_text(DESCRIPTION)
    (d / "shapes.json").write_text(json.dumps(SHAPES))
    (d / "reference" / "run.py").write_text(REF_PY)
    src = load_task(d, kb_root=d / "repo")
    return new_run(src, machine=load_machine(repo, environ={}))


GPU_JOB_STEPS = ("write_judge", "calibrate", "probe_ncu")


def _stub_preamble_and_loop(monkeypatch, calls, *, loop_rc=0, stop="target"):
    """Stand-ins for everything _drive does before the loop (judge writing, the trees,
    calibration, the NCU probe -- the GPU-job payers) and for the loop itself."""
    from rsi_kernel_tune.loop.cli import SUMMARY_FILE

    monkeypatch.setattr(judge_mod, "load_check", lambda d: calls.append("load_check"))

    def judge(*a, **k):
        calls.append("write_judge")

    def calibrate(task, op_dir):
        calls.append("calibrate")
        return {"match": True, "tolerance": {}}

    def probe(task, op_dir):
        calls.append("probe_ncu")
        return True

    monkeypatch.setattr(launch, "write_judge", judge)
    monkeypatch.setattr(launch, "calibrate", calibrate)
    monkeypatch.setattr(launch, "probe_ncu", probe)
    monkeypatch.setattr(launch.taskcard, "write_card",
                        lambda *a, **k: calls.append("write_card")
                        or Path(os.environ["RSI-KERNEL-TUNE_OPERATORS_DIR"]))
    monkeypatch.setattr(launch.taskcard, "seal_reference", lambda *a, **k: None)
    monkeypatch.setattr(launch.taskcard, "set_tolerance", lambda *a, **k: None)
    monkeypatch.setattr(launch.taskcard, "write_loop_toml", lambda *a, **k: None)
    monkeypatch.setattr(launch.taskcard, "clone_workspace", lambda *a, **k: None)

    def main(argv):
        op = Path(os.environ["RSI-KERNEL-TUNE_OPERATORS_DIR"]) / argv[0]
        op.mkdir(parents=True, exist_ok=True)
        (op / SUMMARY_FILE).write_text(json.dumps({"stop": stop}))
        if stop == "target":
            (op / "best_confirmation.json").write_text(json.dumps(
                {"sha": "ab" * 20, "window": "device", "median_ms": 0.02,
                 "median_ratio": 8.0}))
        return loop_rc

    fake = types.ModuleType("rsi_kernel_tune.loop.cli")
    fake.SUMMARY_FILE = SUMMARY_FILE
    fake.main = main
    import rsi_kernel_tune.loop as loop_pkg
    monkeypatch.setattr(loop_pkg, "cli", fake, raising=False)
    monkeypatch.setitem(sys.modules, "rsi_kernel_tune.loop.cli", fake)


# -- BUG-CORPS-C-01: `rsi-kernel-tune harvest` promises one teacher turn and zero GPU jobs ----


def test_harvest_of_a_started_run_submits_no_gpu_job(repo, monkeypatch):
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls)
    assert launch.drive(run, load_machine(repo, environ={}), resuming=True,
                        harvest_only=True) == 0
    assert [c for c in calls if c in GPU_JOB_STEPS] == []


def test_harvest_of_a_never_started_run_submits_no_gpu_job(repo, monkeypatch):
    """A run that never started has nothing to harvest: the loop refuses (rc 2) --
    without the preamble burning judge turns and a calibrate + NCU probe first."""
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls, loop_rc=2, stop=None)
    assert launch.drive(run, load_machine(repo, environ={}), resuming=True,
                        harvest_only=True) == 2
    assert [c for c in calls if c in GPU_JOB_STEPS] == []
    assert run.record()["status"] == STATUS_STOPPED


def test_a_plain_run_still_runs_the_preamble(repo, monkeypatch):
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls)
    assert launch.drive(run, load_machine(repo, environ={}), resuming=True) == 0
    assert [c for c in calls if c in GPU_JOB_STEPS] == list(GPU_JOB_STEPS)


# -- BUG-CORPS-C-02: the finalization must never leave the record stuck at running -----


def test_a_finalization_failure_lands_a_terminal_status(repo, monkeypatch):
    """`confirmed_at_target` blowing up used to escape drive() with the record still at
    "running": an operator's restart loop would rerun the run forever."""
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls)

    def boom(run, task):
        raise ValueError("junk median_ms")

    monkeypatch.setattr(launch, "confirmed_at_target", boom)
    assert launch.drive(run, load_machine(repo, environ={}), resuming=True) == 3
    rec = run.record()
    assert rec["status"] == STATUS_FAILED
    assert "ValueError" in rec.get("error", "")


def test_a_report_failure_keeps_the_run_terminal(repo, monkeypatch):
    """A report is worth having, not worth un-finishing a completed run over: the
    failure is reported and the terminal status and exit code stand."""
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls)

    def boom(run, task, rc):
        raise OSError("disk full")

    monkeypatch.setattr(launch, "write_report", boom)
    assert launch.drive(run, load_machine(repo, environ={}), resuming=True) == 0
    assert run.record()["status"] == STATUS_DONE


def test_a_loop_failure_still_lands_failed(repo, monkeypatch):
    run = _run(repo)
    calls: list = []
    _stub_preamble_and_loop(monkeypatch, calls)

    def boom(*a, **k):
        raise RuntimeError("the judge did not agree")

    monkeypatch.setattr(launch, "write_judge", boom)
    with pytest.raises(RuntimeError):
        launch.drive(run, load_machine(repo, environ={}), resuming=True)
    rec = run.record()
    assert rec["status"] == STATUS_FAILED
    assert "RuntimeError" in rec.get("error", "")
