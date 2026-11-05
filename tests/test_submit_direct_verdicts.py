# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""ssh job verdicts and process hygiene in peregrine_kernel.submit.direct.

Ported from the upstream fix-wave tests (audit-r2-2026-11-05), adapted to this
tree's DirectRunner. One test per fix:

* the remote begin marker prints only once the host GPU lock is HELD (it is
  chained on flock's success inside the session script), so a deadline spent
  QUEUING on that lock is a Lost (resubmitted, never charged), while a deadline
  spent running is a Failed (charged) — BUG-SEED-11;
* an ssh client that dies (rc 255) with the job in flight leaves the remote
  session's outcome unknowable: Lost, and the orphaned session is cleaned up —
  BUG-SEED-12;
* two same-named clones on one host get different remote trees (the clone's
  absolute path is hashed into the tree's name) — BUG-SEED-20.

The scripted ssh stands in for subprocess.run / Popen in the spirit of
test_direct_delivery._FakeSsh, with a ``wait`` that can time out.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from peregrine_kernel.submit import direct
from peregrine_kernel.submit import jobs as ds
from peregrine_kernel.submit.config import SubmitConfig


class _SshFake:
    """Scripted ssh for DirectRunner.

    ``run`` serves the tree sync (rc 0) and records every stop command (any
    argv ending in a "pkill ..." line). ``popen`` serves each exec with an
    ``(rc, began)`` outcome: ``rc`` may be ``"timeout"`` (wait raises
    TimeoutExpired every time, like a process that never settles); ``began``
    writes the begin marker into the job log first."""

    PID = 4194304   # past the default pid_max: no real group is ever signalled

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.execs: list = []
        self.stops: list = []

    def run(self, argv, input=None, capture_output=False, timeout=None):
        if "pkill" in argv[-1]:
            self.stops.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, stdout=None, stderr=None, text=None, stdin=None, **kw):
        rc, began = self.outcomes.pop(0)
        if began:
            stdout.write(f"{ds.BEGIN_MARKER}\n")
        stdout.flush()
        self.execs.append(argv)
        fake = self

        class _P:
            pid = fake.PID

            stdin = None

            @staticmethod
            def wait(timeout=None):
                if rc == "timeout":
                    raise subprocess.TimeoutExpired(argv, timeout)
                return rc

            @staticmethod
            def kill():
                pass

        return _P()


def _git(d: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin:/usr/local/bin"})


def _ssh_op(tmp_path: Path, clone: str = "clone") -> SubmitConfig:
    """An op delivered over ssh, under a clone directory named ``clone``: the op
    tree sits at ``<tmp>/<clone>/checkout/operators/op`` (same-named clones
    must not collide on the host — BUG-SEED-20)."""
    d = tmp_path / clone / "checkout" / "operators" / "op"
    d.mkdir(parents=True)
    (d / "gpu_entry.sh").write_text("#!/bin/bash\n")
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "ssh"\nbranch = "main"\n'
        '[exec]\nhost = "box"\nworkdir = "/data/fk"\ngpu = "1"\n')
    _git(d, "init", "-q", "-b", "main")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    return SubmitConfig(branch="main", delivery="ssh", op_dir=str(d), base_dir=str(tmp_path),
                        exec_host="box", exec_workdir="/data/fk", exec_gpu="1")


def _ssh_run(cfg: SubmitConfig, fake: _SshFake) -> tuple[str, str]:
    log = Path(cfg.op_dir) / "log.txt"
    with open(log, "a+", encoding="utf-8") as lf:
        status = direct.DirectRunner(cfg, run=fake.run, popen=fake.popen,
                                     sleep=lambda s: None).run_mode("main", "perf", lf)
    return status, log.read_text()


# -- BUG-SEED-11: a deadline spent queuing on the host GPU lock is not the job's --

def test_a_deadline_spent_queuing_on_the_gpu_lock_is_lost_not_failed(tmp_path):
    """The begin marker prints only once flock holds the lock (it moved into the
    session script), so a wait that expires without it means the job queued for its
    whole window and NOTHING ran: Lost (resubmitted), never a Failed verdict that
    would cost the kernel an iteration."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([("timeout", False)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_LOST
    assert "queued on the GPU lock" in log and ds.TRANSPORT_ERROR_MARKER not in log
    assert fake.stops    # the queued session was cleaned up best-effort


def test_a_remote_queue_budget_expiry_is_lost_without_a_retry(tmp_path):
    """The session's flock -w expires on the host itself (rc = flock's -E code,
    no marker): the job queued out without beginning, so Lost -- and there is
    nothing to retry (the session is gone), unlike a pre-start transport blip."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(direct.QUEUE_TIMEOUT_RC, False)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_LOST
    assert "queued on the GPU lock" in log and ds.TRANSPORT_ERROR_MARKER not in log
    assert len(fake.execs) == 1 and not fake.stops


def test_a_real_run_timeout_is_still_failed(tmp_path):
    """A job that BEGAN running and then outlived its window is a charged Failed,
    exactly as before — the Lost verdict is for jobs that never started."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([("timeout", True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED
    assert "exceeded" in log


def test_the_remote_begin_marker_is_printed_by_the_lock_holder(tmp_path):
    """The exec command carries the marker inside the flock'd session script, not
    before setsid: a setsid/flock startup failure can then never read as "ran and
    failed", and time spent waiting for the lock reads as not-started."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(0, True)])
    _ssh_run(cfg, fake)
    remote = fake.execs[0][-1]
    marker_pos = remote.find(ds.BEGIN_MARKER)
    flock_pos = remote.find("flock")
    setsid_pos = remote.find("setsid")
    assert setsid_pos < flock_pos < marker_pos       # marker is printed inside flock's command
    assert remote.count(ds.BEGIN_MARKER) == 1        # the old bare pre-session echo is gone


# -- BUG-SEED-12: a dropped ssh client is the machine's fault, not the kernel's --

def test_a_dropped_ssh_connection_is_lost_when_the_job_was_in_flight(tmp_path):
    """ssh exits 255 when the connection dies (a ~3 min network blip kills the
    client long before the remote session): with the begin marker in the log the
    job WAS running and its outcome is unknowable, so Lost -- the kernel is not
    charged for a job that may well have succeeded -- and the orphaned remote
    session is cleaned up best-effort.

    (The upstream test also ran the began=False case and expected Lost there
    too; that half is NOT ported: with the marker printing only once the job
    actually began, a 255 with no marker means nothing of the job ran, which
    stays on the transport-retry channel the frozen test_direct_delivery pins
    for a never-started delivery.)"""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(255, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_LOST
    assert "outcome on the host is unknown" in log
    assert fake.stops and len(fake.execs) == 1       # orphan stopped, no pointless retry


def test_a_relayed_remote_exit_code_is_still_a_charged_failure(tmp_path):
    """rc 1-254 is the remote command's own exit code: trustworthy. With the marker
    (the job ran) it stays Failed -- a broken kernel must still cost an iteration."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(3, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and "exited rc=3" in log
    assert not fake.stops


def test_the_runner_records_a_lost_verdict_that_nothing_charges(tmp_path, monkeypatch):
    """jobs.runner_main writes run_mode's Lost verdict verbatim; the loop then
    resubmits it (loop-side: the runner-lost branch of _submit_gpu_once)."""
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text('[repo]\ndelivery = "local"\nbranch = "main"\n')
    monkeypatch.setattr(direct.DirectRunner, "run_mode",
                        lambda self, branch, mode, lf: ds.STATUS_LOST)
    ds.jobs_dir(d).joinpath("9").mkdir(parents=True)
    ds.runner_main(SubmitConfig(branch="main", op_dir=str(d)), "9", "perf", "main")
    assert ds.job_status(d, "9")["status"] == ds.STATUS_LOST


# -- BUG-SEED-20: two same-named clones on one host get different remote trees --

def test_two_same_named_clones_get_different_remote_trees(tmp_path):
    """Two clones (same middle segment) on one host must not resolve to one
    remote directory (the old basename-only run id made one run's rm -rf reach
    the other's tree). The clone's absolute path is hashed into the tree's
    name; it stays stable for the same clone.

    (Adapted from the upstream test, which hashed the run segment; here the
    hash rides the tree's final segment instead, so the frozen
    test_direct_delivery transport test's exact remote tree keeps matching.)"""
    cfg_a, cfg_b = _ssh_op(tmp_path, "a"), _ssh_op(tmp_path, "b")
    assert cfg_a.op_dir != cfg_b.op_dir
    assert Path(cfg_a.op_dir).name == Path(cfg_b.op_dir).name          # same op name
    assert Path(cfg_a.op_dir).parents[1].name == Path(cfg_b.op_dir).parents[1].name
    tree_a, tree_b = direct.remote_tree(cfg_a), direct.remote_tree(cfg_b)
    assert tree_a != tree_b
    assert tree_a.startswith("/data/fk/checkout/")        # readable prefix kept
    assert re.search(r"/op-[0-9a-f]{8}$", tree_a)        # the clone-path hash
    assert direct.remote_tree(cfg_a) == tree_a            # stable across calls
