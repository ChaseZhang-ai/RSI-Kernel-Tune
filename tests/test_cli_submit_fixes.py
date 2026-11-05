"""Loop CLI + submit fixes: honest ssh job verdicts.

Ported from the archived fix-wave's test_cli_submit_fixes.py (the sections that
pin the ssh delivery's verdict semantics), renamed and rebranded: peregrine_kernel
imports, PK_JOB_BEGIN, pk paths. One fix per section:

* the ssh job's begin marker prints only once the host GPU lock is held, so a
  deadline spent QUEUING on that lock is a Lost (resubmitted, not charged), and a
  deadline spent running is a Failed (charged) -- submit.direct._run_ssh;
* an ssh client that dies (rc 255) leaves the remote session's outcome unknowable:
  Lost, whatever the log shows -- the machine's fault, not the kernel's; the
  session id lives outside the synced tree, so the orphan stays stoppable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from peregrine_kernel.loop import cli
from peregrine_kernel.submit import direct
from peregrine_kernel.submit import jobs as ds
from peregrine_kernel.submit.config import SubmitConfig

# -- ssh verdict classification ---------------------------------------------------------------


class _SshFake:
    """Scripted ssh for DirectRunner, in the spirit of test_direct_delivery._FakeSsh
    but with a ``wait`` that can time out.

    ``run`` serves the tree sync (rc 0). ``popen`` serves each exec with an
    ``(rc, began)`` outcome: ``rc`` may be ``"timeout"`` (wait raises
    TimeoutExpired the first time, then behaves like a killed process); ``began``
    writes the begin marker into the job log first."""

    PID = 4194304          # past the default pid_max: no real group is ever signalled

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
    d = tmp_path / clone / "PeregrineKernel" / "operators" / "op"
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


def test_a_real_run_timeout_is_still_failed(tmp_path):
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([("timeout", True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED
    assert "exceeded" in log


def test_the_remote_begin_marker_is_printed_by_the_lock_holder(tmp_path):
    """The exec command carries the marker inside the flock'd session script, not
    before setsid: a setsid/flock startup failure can then never read as "ran and
    failed"."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(0, True)])
    _ssh_run(cfg, fake)
    remote = fake.execs[0][-1]
    marker_pos = remote.find(ds.BEGIN_MARKER)
    flock_pos = remote.find("flock")
    setsid_pos = remote.find("setsid")
    assert setsid_pos < flock_pos < marker_pos       # marker is printed inside flock's command
    assert remote.count(ds.BEGIN_MARKER) == 1        # the old bare pre-session echo is gone


def test_a_dropped_ssh_connection_is_lost_when_the_job_was_in_flight(tmp_path):
    """Adapted at the merge to this tree's verdict split (BUG-SEED-12): ssh rc 255
    with the begin marker on the log means the job RAN and its outcome is now
    unknowable -- Lost, the orphaned session ended best-effort. A rc 255 with NO
    marker never began: it is a transport failure that retries into
    TRANSPORT_ERROR, not a Lost."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(255, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_LOST
    assert "outcome on the host is unknown" in log
    assert fake.stops

    fake = _SshFake([(255, False)] * direct.TRANSPORT_ATTEMPTS)
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and ds.TRANSPORT_ERROR_MARKER in log
    assert not fake.stops


def test_a_relayed_remote_exit_code_is_still_a_charged_failure(tmp_path):
    """rc 1-254 is the remote command's own exit code: trustworthy. With the marker
    (the job ran) it stays Failed -- a broken kernel must still cost an iteration.
    And the session needs no cleanup: setsid --wait relayed it, so the chain (and
    the session with it) is already gone."""
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(3, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and "exited rc=3" in log
    assert not fake.stops


def test_the_runner_records_a_lost_verdict_that_nothing_charges(tmp_path, monkeypatch):
    """jobs.runner_main writes run_mode's Lost verdict verbatim; the loop then
    resubmits it (loop-side: _submit_gpu_once counts it against lost_retries and
    never charges the kernel)."""
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text('[repo]\ndelivery = "local"\nbranch = "main"\n')
    monkeypatch.setattr(direct.DirectRunner, "run_mode",
                        lambda self, branch, mode, lf: ds.STATUS_LOST)
    ds.jobs_dir(d).joinpath("9").mkdir(parents=True)
    ds.runner_main(SubmitConfig(branch="main", op_dir=str(d)), "9", "perf", "main")
    assert ds.job_status(d, "9")["status"] == ds.STATUS_LOST


# -- clock-lock retry budget -------------------------------------------------------------------


def test_clock_lock_retry_counts_total_submits_and_says_so(monkeypatch, capsys):
    """The budget is a TOTAL of ``clock_lock_retries`` submits (1 + 2 resubmits), the
    same counting the lost-runner budget uses, and the progress line names the
    current attempt out of that total (rewritten from the archived wording, which
    named a different progress line)."""
    refused = f"{ds.CLOCK_LOCK_REFUSED_MARKER} lock not held\n"
    be = cli.RealBackend("op", None, None, "n", "e")
    be.clock_lock_retry_sleep_s = 0
    seen: list = []

    def once(branch, mode):
        seen.append(mode)
        be.last_job_id = str(len(seen))
        return refused

    monkeypatch.setattr(be, "_submit_gpu_once", once)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    with pytest.raises(cli.ClockLockRefusedError) as why:
        be.submit_gpu("b", "all")
    assert len(seen) == be.clock_lock_retries == 3
    out = capsys.readouterr().out
    assert "resubmitting (attempt 2 of 3)" in out and "resubmitting (attempt 3 of 3)" in out
    assert "3 refusal(s) in a row" in str(why.value)
