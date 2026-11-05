"""Regression tests for the audit-corps R3D wave on the job lifecycle.

* BUG-CORPS-C-16: every runner crash used to be recorded Failed (an iteration
  charged to the kernel), infrastructure faults included. A fault of the
  machine or its tooling (a vanished file, a full disk, git refusing the
  archive, a spawn failure) is now Lost -- the loop resubmits and the kernel
  is not charged -- while a usage error (a malformed mode string) stays
  Failed and is reported as such, explicitly.
* BUG-CORPS-C-19: a runner pid could be reused by an unrelated process, making
  job_status report a phantom Running and stop_job signal a stranger. The pid
  file now records a /proc identity (the process's starttime) and the
  consumers verify it before believing or signalling the pid.

Test style follows tests/test_direct_delivery.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from peregrine_kernel.submit import direct, jobs as ds
from peregrine_kernel.submit.config import SubmitConfig, load_config

ENTRY_OK = """#!/bin/bash
echo "entry task=$1"
echo "task=$1 finished"
"""


def _op(base: Path, name: str = "op") -> tuple[Path, SubmitConfig]:
    d = base / "operators" / name
    d.mkdir(parents=True)
    (d / "gpu_entry.sh").write_text(ENTRY_OK)
    (d / "kernel.py").write_text("x = 1\n")
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "local"\nbranch = "main"\n'
        '[exec]\ngpu = "0"\n', encoding="utf-8")
    return d, SubmitConfig(branch="main", op_dir=str(d), base_dir=str(base),
                           timeout=60, delivery="local", exec_gpu="0")


@pytest.fixture(autouse=True)
def _locks(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))


class _Boom:
    """A DirectRunner stand-in whose run_mode raises a scripted exception."""

    def __init__(self, cfg, *a, **k):
        pass

    def run_mode(self, branch, mode, log_file):
        raise self.exception


# -- BUG-CORPS-C-16: infra faults are Lost, usage errors are explicit Failed ----


def test_an_infra_fault_is_lost_not_failed(tmp_path, monkeypatch):
    d, cfg = _op(tmp_path)
    ds.jobs_dir(d).joinpath("1").mkdir(parents=True)

    def boom(self, *a, **k):
        raise FileNotFoundError(2, "No such file or directory (a stat raced a deletion)")

    monkeypatch.setattr(_Boom, "run_mode", boom)
    monkeypatch.setattr(direct, "DirectRunner", _Boom)
    rc = ds.runner_main(cfg, "1", "perf", cfg.branch)
    log = ds.log_path(d, "1").read_text()
    assert ds.job_status(d, "1")["status"] == ds.STATUS_LOST
    assert "infrastructure fault" in log
    assert rc != 0


def test_a_usage_error_is_failed_and_named_as_such(tmp_path, monkeypatch):
    d, cfg = _op(tmp_path)
    ds.jobs_dir(d).joinpath("1").mkdir(parents=True)

    def boom(self, *a, **k):
        raise ValueError("No escaped character")   # shlex.split on a bad mode string

    monkeypatch.setattr(_Boom, "run_mode", boom)
    monkeypatch.setattr(direct, "DirectRunner", _Boom)
    rc = ds.runner_main(cfg, "1", "perf", cfg.branch)
    log = ds.log_path(d, "1").read_text()
    assert ds.job_status(d, "1")["status"] == ds.STATUS_FAILED
    assert "usage error" in log and "NOT a kernel failure" in log
    assert rc != 0


def test_a_git_archive_failure_is_lost(tmp_path, monkeypatch):
    """git refusing the tree (an old git without the exclude pathspec) is the
    tooling, not the kernel: the verdict is Lost."""
    d, cfg = _op(tmp_path)
    ds.jobs_dir(d).joinpath("1").mkdir(parents=True)

    def boom(self, *a, **k):
        raise RuntimeError("git archive main failed: pathspec not supported")

    monkeypatch.setattr(_Boom, "run_mode", boom)
    monkeypatch.setattr(direct, "DirectRunner", _Boom)
    ds.runner_main(cfg, "1", "perf", cfg.branch)
    assert ds.job_status(d, "1")["status"] == ds.STATUS_LOST


# -- BUG-CORPS-C-19: a pid file carries a /proc identity, verified before use ---


def _live_sleeper() -> subprocess.Popen:
    """A stand-in runner: alive, and in a session of its own (killpg-able)."""
    return subprocess.Popen(["sleep", "30"], start_new_session=True)


def test_proc_start_ticks_reads_a_live_process_only():
    sleeper = _live_sleeper()
    try:
        ident = ds.proc_start_ticks(sleeper.pid)
        assert ident and ident.isdigit()
        assert ds.proc_start_ticks(sleeper.pid) == ident     # stable while alive
    finally:
        sleeper.terminate()
        sleeper.wait()
    assert ds.proc_start_ticks(sleeper.pid) is None          # reaped: no identity


def test_job_status_rejects_a_pid_whose_identity_drifted(tmp_path):
    """A reused pid (alive, but not the process the pid file was written for)
    is not a running job: no phantom Running, and stop does not signal it."""
    d, _ = _op(tmp_path)
    ds.job_dir(d, "7").mkdir(parents=True)
    sleeper = _live_sleeper()
    try:
        ident = ds.proc_start_ticks(sleeper.pid)
        ds.pid_path(d, "7").write_text(f"{sleeper.pid} {ident}")
        assert ds.job_status(d, "7")["status"] == ds.STATUS_RUNNING
        ds.pid_path(d, "7").write_text(f"{sleeper.pid} {int(ident) + 7}")
        assert ds.job_status(d, "7")["status"] == ds.STATUS_LOST
        assert ds.stop_job(d, "7") is False
        assert sleeper.poll() is None                        # the stranger was untouched
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_a_legacy_pid_only_file_keeps_the_old_semantics(tmp_path):
    d, _ = _op(tmp_path)
    ds.job_dir(d, "7").mkdir(parents=True)
    sleeper = _live_sleeper()
    try:
        ds.pid_path(d, "7").write_text(str(sleeper.pid))
        assert ds.job_status(d, "7")["status"] == ds.STATUS_RUNNING
        assert ds.stop_job(d, "7") is True                   # signalled as before
        sleeper.wait(timeout=10)
        assert sleeper.returncode != 0
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait()


def test_spawn_runner_records_a_proc_identity(tmp_path, monkeypatch):
    d, cfg = _op(tmp_path)
    holder = _live_sleeper()     # stands in for the process spawn_runner detaches

    class _FakeProc:
        pid = holder.pid

    monkeypatch.setattr(ds.subprocess, "Popen", lambda *a, **k: _FakeProc())
    jid = ds.spawn_runner(cfg, "perf", "main")
    parts = ds.pid_path(d, jid).read_text().split()
    assert parts == [str(holder.pid), ds.proc_start_ticks(holder.pid)]
    holder.terminate()
    holder.wait()


def test_kill_session_skips_pids_reused_since_the_tree_was_read(monkeypatch):
    sent: list = []
    groups: list = []
    monkeypatch.setattr(direct.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(direct.os, "killpg", lambda pgid, sig: groups.append(pgid))
    monkeypatch.setattr(direct, "_descendants", lambda root: [101])

    class _Proc:
        pid = 100

        def wait(self, timeout=None):
            return 0

    calls = {"n": 0}
    sampled = {100: "7", 101: "8"}

    def ticks(pid):
        calls["n"] += 1
        # the first two reads sample the tree; afterwards every pid shows a
        # different occupant (exited and reused before the signal landed)
        return sampled[pid] if calls["n"] <= 2 else "999"

    monkeypatch.setattr(direct, "proc_start_ticks", ticks)
    direct._kill_session(_Proc(), grace_s=0)
    assert sent == [] and groups == []   # nothing ours, nothing of a stranger's


def test_kill_session_signals_pids_that_are_still_ours(monkeypatch):
    sent: list = []
    groups: list = []
    monkeypatch.setattr(direct.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(direct.os, "killpg", lambda pgid, sig: groups.append(pgid))
    monkeypatch.setattr(direct, "_descendants", lambda root: [101])
    monkeypatch.setattr(direct, "proc_start_ticks",
                        lambda pid: {100: "7", 101: "8"}[pid])

    class _Proc:
        pid = 100

        def wait(self, timeout=None):
            return 0

    direct._kill_session(_Proc(), grace_s=0)
    assert {100, 101} <= {pid for pid, _ in sent}
    assert 100 in groups


def test_kill_group_refuses_a_pid_that_left_our_identity(monkeypatch):
    groups: list = []
    monkeypatch.setattr(direct.os, "killpg", lambda pgid, sig: groups.append(pgid))
    monkeypatch.setattr(direct, "proc_start_ticks", lambda pid: "999")

    class _Proc:
        pid = 100

        def kill(self):
            groups.append("proc-kill")

        def wait(self, timeout=None):
            return 0

    direct._kill_group(_Proc(), identity="7")      # the pid is somebody else's now
    assert groups == []
    direct._kill_group(_Proc(), identity=None)     # no identity: the old behavior
    assert groups == [100]
