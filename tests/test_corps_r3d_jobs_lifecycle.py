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
