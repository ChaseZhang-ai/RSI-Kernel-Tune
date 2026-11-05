"""Ports of the audit-r2 archive's submit-channel regression tests.

Source: the audit-r2-2026-11-05 archive's untracked test directory (legacy-snapshot
naming), file test_cli_submit_fixes.py (written against the archived legacy
snapshot). Ported per the correction order of 2026-11-05: every assertion was
re-checked against the latest semantics, mechanical renames applied (legacy
package name -> ``peregrine_kernel``, ``FK_*`` -> ``PK_*``), and coordinates
re-anchored.

Subset rationale (full inventory in the porting report of CHANGES-corps-fix.md):
only the tests that ADD coverage over the in-session test_corps_fix_* wave are
ported. The archive's other ten tests were dropped: their premises were either
replaced by later mechanisms (SEED-11 removed the queue deadline, SEED-19 made
overlay failures fail closed, _retry_call no longer announces) or they are
already covered by test_corps_fix_channel.py / test_direct_delivery.py.

Each surviving test keeps its original intent; the docstring says where it
came from and what it pins on today's surface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from peregrine_kernel.submit import direct
from peregrine_kernel.submit import jobs as ds
from peregrine_kernel.submit.config import SubmitConfig

from test_direct_delivery import _FakeSsh, _ssh_cfg, _ssh_run


# -- a real run timeout is still a charged failure (archive: test_a_real_run_
#    timeout_is_still_failed) --------------------------------------------------


class _TimedSsh:
    """An ssh exec whose remote session prints the begin marker and then never
    finishes: every ``wait`` raises TimeoutExpired. The pid is past the default
    pid_max so no real group is ever signalled by the cleanup."""

    PID = 4194304

    def __init__(self):
        self.stops: list = []
        self.execs: list = []

    def run(self, argv, input=None, capture_output=False, timeout=None):
        if "pkill" in argv[-1]:
            self.stops.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, stdout=None, stderr=None, text=None, stdin=None, **kw):
        stdout.write(f"{ds.BEGIN_MARKER}\n")
        stdout.flush()
        self.execs.append(argv)
        fake = self

        class _P:
            pid = fake.PID
            stdin = None

            @staticmethod
            def wait(timeout=None):
                raise subprocess.TimeoutExpired(argv, timeout)

            @staticmethod
            def kill():
                pass

        return _P()


def test_a_real_run_timeout_on_ssh_is_still_a_charged_failure(tmp_path):
    """The job began (the marker printed under the host lock), then ran past its
    deadline: the session is stopped best-effort and the verdict is Failed -- a
    charged failure, not a transport retry and not a Lost. This pins the ssh
    branch; the local branch has its own deadline test in test_direct_delivery."""
    cfg = _ssh_cfg(tmp_path)
    fake = _TimedSsh()
    log = Path(cfg.op_dir) / "log.txt"
    with open(log, "a+", encoding="utf-8") as lf:
        status = direct.DirectRunner(cfg, run=fake.run, popen=fake.popen,
                                     sleep=lambda s: None).run_mode("main", "perf", lf)
    text = log.read_text()
    assert status == ds.STATUS_FAILED
    assert "exceeded" in text
    assert fake.stops, "the run-away session was not stopped on the host"
    assert ds.TRANSPORT_ERROR_MARKER not in text


# -- the begin marker lives inside the flocked session script (archive:
#    test_the_remote_begin_marker_is_printed_by_the_lock_holder) ---------------


def test_the_exec_command_prints_the_begin_marker_once_inside_the_flocked_session(
        tmp_path):
    """The exec command must carry the begin marker only inside the flock'd
    session script, never as a bare pre-session echo: a setsid/flock startup
    failure can then never read as "ran and failed", and a queued job cannot
    print a marker it has not earned. test_corps_fix_channel pins the behavior
    (a real session script under a held lock); this pins the rendered command's
    structure."""
    cfg = _ssh_cfg(tmp_path)
    fake = _FakeSsh([(0, True)])
    _ssh_run(cfg, fake)
    remote = fake.execs[0][-1]
    marker_pos = remote.find(ds.BEGIN_MARKER)
    flock_pos = remote.find("flock")
    setsid_pos = remote.find("setsid")
    assert setsid_pos < flock_pos < marker_pos   # marker is printed inside flock's command
    assert remote.count(ds.BEGIN_MARKER) == 1    # no bare pre-session echo anywhere


# -- the runner records a Lost verdict verbatim (archive:
#    test_the_runner_records_a_lost_verdict_that_nothing_charges) --------------


def test_the_runner_records_a_lost_verdict_that_nothing_charges(tmp_path, monkeypatch):
    """jobs.runner_main writes run_mode's verdict verbatim, whatever it is -- a
    Lost verdict reaches result.json as Lost, so the loop can resubmit it
    instead of charging the kernel (the loop side of that handoff is covered by
    test_corps_fix_channel's lost-budget tests)."""
    d = tmp_path / "operators" / "op"
    d.mkdir(parents=True)
    (d / "submit.toml").write_text('[repo]\ndelivery = "local"\nbranch = "main"\n')
    monkeypatch.setattr(direct.DirectRunner, "run_mode",
                        lambda self, branch, mode, lf: ds.STATUS_LOST)
    ds.jobs_dir(d).joinpath("9").mkdir(parents=True)
    ds.runner_main(SubmitConfig(branch="main", op_dir=str(d)), "9", "perf", "main")
    assert ds.job_status(d, "9")["status"] == ds.STATUS_LOST
