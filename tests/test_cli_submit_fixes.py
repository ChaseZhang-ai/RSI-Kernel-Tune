"""Regression tests ported from the upstream fix wave (loop CLI + submit batch).

Only the mechanisms that exist in this lineage are pinned; each names the fix it
carries:

* jobs.runner_main writes run_mode's Lost verdict verbatim, so the loop (which
  resubmits a lost job) sees exactly what the runner recorded;
* a real run timeout -- the job began (the begin marker is in its log) and the
  deadline expired -- is still a charged Failed;
* a relayed remote exit code (rc 1-254 after the begin marker) is the remote
  command's own verdict: charged Failed, no cleanup stop;
* the clock-lock refusal budget counts TOTAL submits (1 + resubmits, the same
  counting the lost-runner budget uses) and the progress line names it.

Dropped from the source file, with reasons (upstream fix absent here -- see the
BUG-PORT ledger):

* the begin marker printed inside the flock'd session script, and the Lost
  verdicts for a deadline spent queuing on the GPU lock and for a dropped ssh
  connection (rc 255): the marker is echoed BEFORE setsid here, and rc 255 is a
  retried transport failure that ends in a charged Failed;
* two same-named clones getting different remote trees (clone-path hash):
  remote_tree here is ``<workdir>/<clone-repo-name>/<op>`` with no hash -- two
  same-named clones on one host share (and rm -rf) one tree;
* a poll timeout counting toward giving an in-flight job up: the poll-timeout
  BackendError escapes _submit_gpu_once without touching the inflight record;
* an overlay build failure degrading with a structured trace: the fail-open
  exists but no fallback record is written, and the announce then prints the
  misleading "the tree's copies match the card" line;
* _retry_call announcing retries and not sleeping after the last failure:
  retries are silent here and the final sleep still runs;
* an interrupted agent turn ending the whole process group (start_new_session):
  _agent_turn's Popen runs without a new session and its interrupt path kills
  only the shell wrapper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from peregrine_kernel.submit import direct
from peregrine_kernel.submit import jobs as ds
from peregrine_kernel.submit.config import SubmitConfig
from peregrine_kernel.loop import cli


# -- ssh verdict classification ------------------------------------------------


class _SshFake:
    """Scripted ssh for DirectRunner, in the spirit of test_direct_delivery's fake
    but with a ``wait`` that can time out.

    ``run`` serves the tree sync (rc 0). ``popen`` serves each exec with an
    ``(rc, began)`` outcome: ``rc`` may be ``"timeout"`` (wait raises
    TimeoutExpired); ``began`` writes the begin marker into the job log first."""

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


def test_a_real_run_timeout_is_still_failed(tmp_path):
    # the job began (its marker is in the log) and the runner's deadline expired:
    # a charged Failed -- the kernel spent its window and produced nothing
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([("timeout", True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED
    assert "exceeded" in log


def test_a_relayed_remote_exit_code_is_still_a_charged_failure(tmp_path):
    # rc 1-254 is the remote command's own exit code: trustworthy. With the marker
    # (the job ran) it stays Failed -- a broken kernel must still cost an iteration.
    cfg = _ssh_op(tmp_path)
    fake = _SshFake([(3, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and "exited rc=3" in log
    assert not fake.stops


def test_the_runner_records_a_lost_verdict_that_nothing_charges(tmp_path, monkeypatch):
    """jobs.runner_main writes run_mode's Lost verdict verbatim; the loop then
    resubmits it (the loop side of this contract)."""
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text('[repo]\ndelivery = "local"\nbranch = "main"\n')
    monkeypatch.setattr(direct.DirectRunner, "run_mode",
                        lambda self, branch, mode, lf: ds.STATUS_LOST)
    ds.jobs_dir(d).joinpath("9").mkdir(parents=True)
    ds.runner_main(SubmitConfig(branch="main", op_dir=str(d)), "9", "perf", "main")
    assert ds.job_status(d, "9")["status"] == ds.STATUS_LOST


# -- clock-lock retry counting ---------------------------------------------------


def test_clock_lock_retry_counts_total_submits_and_says_so(monkeypatch, capsys):
    """The budget is a TOTAL of ``clock_lock_retries`` submits (1 + resubmits), the
    same counting the lost-runner budget uses, and the progress line names it."""
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
    with pytest.raises(cli.ClockLockRefusedError):
        be.submit_gpu("b", "all")
    assert len(seen) == be.clock_lock_retries == 3
    out = capsys.readouterr().out
    assert "resubmitting (1/3)" in out and "resubmitting (2/3)" in out
