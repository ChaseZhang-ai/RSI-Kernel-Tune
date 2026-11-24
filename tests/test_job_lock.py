"""forgekernel.submit.direct: an external GPU lock in front of a local job (FK_JOB_LOCK_CMD).

A lock command shared with other work on the machine is put in front of the entry. What is
pinned here: the entry is started from inside the lock; a job that never got the lock is
"not delivered" (TRANSPORT_ERROR_MARKER, never charged to the kernel) while an entry that
failed after it began is the failure it always was; the wait for the lock is not taken out of
the entry's time; and ssh delivery, and a run that sets no lock, are as they were.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgekernel.submit import direct
from forgekernel.submit import jobs as ds
from forgekernel.submit.config import load_config
from test_direct_delivery import _FakeSsh, _locks, _op, _ssh_cfg, _ssh_run  # noqa: F401  (fixtures)

LOCK = "/bin/with-lock gpu 1800 --"
LOCK_ARGV = ["/bin/with-lock", "gpu", "1800", "--"]
REFUSED = "[gpu_entry.sh] REFUSED: this card times under the board's clock lock"

# Stands in for `with-lock`: skips the lock's own arguments up to `--`, says what it does
# around the command it wraps, gives up with with-lock's exit status when told to, and passes
# the command's own status through.
FAKELOCK = """#!/bin/bash
while [ "$1" != "--" ]; do shift; done
shift
echo "[fakelock] waiting"
if [ "${FAKE_LOCK_GIVE_UP:-0}" = 1 ]; then
  echo "[fakelock] busy, giving up"
  exit 3
fi
[ -z "${FAKE_LOCK_WAIT:-}" ] || sleep "$FAKE_LOCK_WAIT"
echo "[fakelock] acquired"
"$@"
rc=$?
echo "[fakelock] released rc=$rc"
exit $rc
"""


@pytest.fixture
def fakelock(tmp_path):
    p = tmp_path / "bin" / "fakelock"
    p.parent.mkdir(exist_ok=True)
    p.write_text(FAKELOCK, encoding="utf-8")
    p.chmod(0o755)
    return f"{p} gpu 1800 --"


@pytest.fixture(autouse=True)
def _no_lock_unless_a_test_sets_one(monkeypatch):
    monkeypatch.delenv(direct.JOB_LOCK_ENV, raising=False)


# --- the command -------------------------------------------------------------------------


def test_no_lock_command_is_no_prefix(monkeypatch):
    assert direct.JOB_LOCK_ENV == "FK_JOB_LOCK_CMD"
    assert direct.job_lock_prefix({}) == []
    assert direct.job_lock_prefix({"FK_JOB_LOCK_CMD": ""}) == []
    assert direct.job_lock_prefix({"FK_JOB_LOCK_CMD": "  \t "}) == []
    assert direct.job_lock_prefix({"FK_JOB_LOCK_CMD": LOCK}) == LOCK_ARGV
    assert direct.job_lock_prefix({"FK_JOB_LOCK_CMD": '"/a b/with-lock"  gpu\t1800 --'}) == \
        ["/a b/with-lock", "gpu", "1800", "--"]
    assert direct.job_lock_prefix({"FK_JOB_LOCK_CMD": "  " + LOCK + "  "}) == LOCK_ARGV
    with pytest.raises(ValueError):
        direct.job_lock_prefix({"FK_JOB_LOCK_CMD": '"/a b/with-lock gpu 1800 --'})
    monkeypatch.setenv("FK_JOB_LOCK_CMD", LOCK)
    assert direct.job_lock_prefix() == LOCK_ARGV            # the process environment by default


def test_the_entry_under_a_lock_is_started_by_a_shell_that_says_it_began():
    assert direct._BEGIN_SCRIPT == f'echo {ds.BEGIN_MARKER} && exec "$@"'
    entry = ["bash", "gpu_entry.sh", "cmd", "x y", "z"]
    assert direct._locked_entry_argv(entry) == ["bash", "-c", direct._BEGIN_SCRIPT, "bash", *entry]
    assert [*LOCK_ARGV, *direct._locked_entry_argv(entry)] == \
        LOCK_ARGV + ["bash", "-c", direct._BEGIN_SCRIPT, "bash", "bash", "gpu_entry.sh", "cmd",
                     "x y", "z"]


# --- what the runner makes of how the lock command ended (no process is started) ------------


def _fake(tmp_path, monkeypatch, *, out: str, rc: int = 0, error: Exception | None = None,
          lock: str | None = LOCK, entry: str = "#!/bin/bash\necho entry\n"):
    """Run one local job whose process is faked: it writes ``out`` to the job's log and answers
    ``rc`` (or raises ``error`` when started). Returns the status, the log and the argv."""
    _op(tmp_path, entry=entry)
    if lock is not None:
        monkeypatch.setenv(direct.JOB_LOCK_ENV, lock)
    seen: list = []

    def popen(argv, **kw):
        seen.append(argv)
        if error is not None:
            raise error
        kw["stdout"].write(out)
        kw["stdout"].flush()
        return SimpleNamespace(pid=0, wait=lambda timeout=None: rc)

    cfg = load_config("op", base_dir=tmp_path)
    log = Path(cfg.op_dir) / "log.txt"
    with open(log, "a+", encoding="utf-8") as lf:
        status = direct.DirectRunner(cfg, popen=popen).run_mode("main", "perf", lf)
    return status, log.read_text(encoding="utf-8"), (seen[0] if seen else None)


def test_a_lock_that_gave_up_waiting_is_not_delivered(tmp_path, monkeypatch):
    status, log, argv = _fake(tmp_path, monkeypatch, out="[with-lock] 锁 gpu 仍被占用，等待 1800s 后放弃\n", rc=3)
    assert status == ds.STATUS_FAILED
    assert log.count(ds.TRANSPORT_ERROR_MARKER) == 1
    marker = next(ln for ln in log.splitlines() if ds.TRANSPORT_ERROR_MARKER in ln)
    assert marker == (f"{ds.TRANSPORT_ERROR_MARKER} the GPU lock command {LOCK} ended with status 3 "
                      f"before the entry began (it gave up waiting for the lock); the entry never ran")
    assert argv == LOCK_ARGV + ["bash", "-c", direct._BEGIN_SCRIPT, "bash", "bash", "gpu_entry.sh", "perf"]
    # nothing says the entry ran, and nothing charges it
    assert ds.BEGIN_MARKER not in log and "exited rc=" not in log


def test_an_entry_that_exits_3_after_it_began_is_its_own_failure(tmp_path, monkeypatch):
    """A card's gpu_entry.sh exits 3 when the board's clock lock does not hold. through
    with-lock, which passes a command's status on, it must not read as a lock that gave up."""
    status, log, _ = _fake(tmp_path, monkeypatch, out=f"[with-lock] 已获得锁 gpu\n{ds.BEGIN_MARKER}\n{REFUSED}\n", rc=3)
    assert status == ds.STATUS_FAILED
    assert ds.TRANSPORT_ERROR_MARKER not in log
    assert f"{ds.BEGIN_MARKER}\n{REFUSED}" in log and f"exited rc=3" in log
    assert ds.clock_lock_refusal(log) == REFUSED          # what the loop resubmits on is still seen


@pytest.mark.parametrize("rc", [1, 2, 64, 125, 126, 127, 255])
def test_a_lock_command_that_ends_any_other_way_without_starting_the_entry_is_not_delivered(
        tmp_path, monkeypatch, rc):
    status, log, _ = _fake(tmp_path, monkeypatch, out=f"/bin/sh: with-lock: command not found\n", rc=rc)
    assert status == ds.STATUS_FAILED
    (line,) = [ln for ln in log.splitlines() if ds.TRANSPORT_ERROR_MARKER in ln]
    assert line == (f"{ds.TRANSPORT_ERROR_MARKER} the GPU lock command {LOCK} ended with status {rc} "
                    f"before the entry began; the entry never ran")
    assert "gave up" not in log and ds.BEGIN_MARKER not in log


def test_a_lock_command_that_answers_success_without_starting_the_entry_is_not_believed(
        tmp_path, monkeypatch):
    status, log, _ = _fake(tmp_path, monkeypatch, out="ok then\n", rc=0)
    assert status == ds.STATUS_FAILED
    assert f"{ds.TRANSPORT_ERROR_MARKER} the GPU lock command {LOCK} ended with status 0 before the entry began; " \
           f"the entry never ran" in log


def test_a_lock_command_that_cannot_be_started_is_not_delivered(tmp_path, monkeypatch):
    status, log, argv = _fake(tmp_path, monkeypatch, out="", error=FileNotFoundError(2, "No such file or directory"))
    assert status == ds.STATUS_FAILED and argv is not None
    assert f"{ds.TRANSPORT_ERROR_MARKER} the GPU lock command could not be started " \
           f"(FileNotFoundError(2, 'No such file or directory')); the entry never ran" in log


def test_a_lock_command_that_does_not_split_is_not_delivered_and_nothing_is_started(tmp_path, monkeypatch):
    status, log, argv = _fake(tmp_path, monkeypatch, out="", lock='"/a b/with-lock gpu --')
    assert status == ds.STATUS_FAILED and argv is None
    assert f"{ds.TRANSPORT_ERROR_MARKER} FK_JOB_LOCK_CMD does not split into a command" in log
    assert "No closing quotation" in log


def test_an_entry_that_ran_under_the_lock_and_succeeded(tmp_path, monkeypatch):
    status, log, _ = _fake(tmp_path, monkeypatch, out=f"{ds.BEGIN_MARKER}\ntask=perf finished\n")
    assert status == ds.STATUS_SUCCEEDED and ds.TRANSPORT_ERROR_MARKER not in log
    assert log.index("[direct] local GPU") < log.index(ds.BEGIN_MARKER) < log.index("task=perf finished")


def test_without_a_lock_command_nothing_changes(tmp_path, monkeypatch):
    status, log, argv = _fake(tmp_path, monkeypatch, out="task=perf finished\n", lock=None)
    assert status == ds.STATUS_SUCCEEDED and argv == ["bash", "gpu_entry.sh", "perf"]
    assert log == f"[direct] local GPU 2 branch=main mode=perf\n{ds.BEGIN_MARKER}\ntask=perf finished\n"
    status, log, argv = _fake(tmp_path / "b", monkeypatch, out="boom\n", rc=3, lock=None)
    assert status == ds.STATUS_FAILED and argv == ["bash", "gpu_entry.sh", "perf"]
    assert "exited rc=3" in log and ds.TRANSPORT_ERROR_MARKER not in log
    with pytest.raises(FileNotFoundError):                 # a machine fault is the runner's to report
        _fake(tmp_path / "c", monkeypatch, out="", error=FileNotFoundError(2, "No such file or directory"), lock=None)


# --- with a process: a lock command that behaves like with-lock ----------------------------


def _entry_wait_then_finish(wait: str) -> str:
    return f'#!/bin/bash\necho "entry task=$1 gpu=$CUDA_VISIBLE_DEVICES timeout=$TASK_TIMEOUT"\nls\nsleep {wait}\necho "task=$1 finished"\n'


def test_the_entry_runs_inside_the_lock_and_says_it_began_there(tmp_path, monkeypatch, fakelock):
    _op(tmp_path, entry=_entry_wait_then_finish("0"))
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 0
    log = ds.job_log(Path(cfg.op_dir), "1")
    lines = log.splitlines()
    order = [next(i for i, ln in enumerate(lines) if text in ln) for text in
             ("[fakelock] waiting", "[fakelock] acquired", ds.BEGIN_MARKER,
              "entry task=perf gpu=2 timeout=60", "[fakelock] released rc=0")]
    assert order == sorted(order)
    assert "kernel.py" in lines[order[2]: order[-1]]         # the entry's own output, inside
    assert ds.job_status(Path(cfg.op_dir), "1")["status"] == ds.STATUS_SUCCEEDED
    assert ds.TRANSPORT_ERROR_MARKER not in log and log.rstrip().endswith("[fakelock] released rc=0")


def test_a_lock_that_never_came_is_not_delivered_and_the_job_stops_there(tmp_path, monkeypatch, fakelock):
    monkeypatch.setenv("FAKE_LOCK_GIVE_UP", "1")
    _op(tmp_path, entry=_entry_wait_then_finish("0"))
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 1
    log = ds.job_log(Path(cfg.op_dir), "1")
    assert ds.job_status(Path(cfg.op_dir), "1")["status"] == ds.STATUS_FAILED
    assert ds.BEGIN_MARKER not in log and "entry task" not in log and "exited rc=" not in log
    (line,) = [ln for ln in log.splitlines() if ds.TRANSPORT_ERROR_MARKER in ln]
    assert line.endswith("ended with status 3 before the entry began (it gave up waiting for the lock); "
                         "the entry never ran")
    assert log.index("[fakelock] busy, giving up") < log.index(ds.TRANSPORT_ERROR_MARKER)


def test_the_cards_own_clock_lock_refusal_is_seen_as_what_it_is(tmp_path, monkeypatch, fakelock):
    _op(tmp_path, entry=f'#!/bin/bash\necho "{REFUSED}"\nexit 3\n')
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 1
    log = ds.job_log(Path(cfg.op_dir), "1")
    assert ds.clock_lock_refusal(log) == REFUSED
    assert ds.TRANSPORT_ERROR_MARKER not in log and "exited rc=3" in log


def test_waiting_for_the_lock_is_not_taken_out_of_the_entrys_time(tmp_path, monkeypatch, fakelock):
    """The lock came 1.2 s in and the entry then took 1.2 s of a 2 s limit: 2.4 s in all. Timed
    from the start of the wait the job would be killed and charged; it must succeed."""
    monkeypatch.setenv("FAKE_LOCK_WAIT", "1.2")
    monkeypatch.setattr(direct, "RUN_HEADROOM_S", 0)
    _op(tmp_path, entry=_entry_wait_then_finish("1.2"), timeout=2)
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    t0 = time.monotonic()
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 0, ds.job_log(Path(cfg.op_dir), "1")
    assert time.monotonic() - t0 > 2.2                     # it really did outlast the limit
    assert "task=perf finished" in ds.job_log(Path(cfg.op_dir), "1")


def test_an_entry_that_overruns_its_time_under_a_lock_is_still_killed(tmp_path, monkeypatch, fakelock):
    monkeypatch.setenv("FAKE_LOCK_WAIT", "0.4")
    monkeypatch.setattr(direct, "RUN_HEADROOM_S", 0)
    _op(tmp_path, entry=_entry_wait_then_finish("30"), timeout=1)
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    t0 = time.monotonic()
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 1
    assert time.monotonic() - t0 < 10
    log = ds.job_log(Path(cfg.op_dir), "1")
    assert "exceeded 1s" in log and ds.TRANSPORT_ERROR_MARKER not in log
    assert "task=perf finished" not in log
    assert subprocess.run(["pgrep", "-f", r"sleep 30"], capture_output=True, text=True).stdout.strip() == ""


def test_a_lock_that_never_starts_the_entry_is_given_up_on_within_the_jobs_time(
        tmp_path, monkeypatch, fakelock):
    monkeypatch.setenv("FAKE_LOCK_WAIT", "30")
    monkeypatch.setattr(direct, "RUN_HEADROOM_S", 0)
    monkeypatch.setattr(direct, "LOCK_POLL_S", 0.05)
    _op(tmp_path, entry=_entry_wait_then_finish("0"), timeout=1)
    monkeypatch.setenv(direct.JOB_LOCK_ENV, fakelock)
    cfg = load_config("op", base_dir=tmp_path)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True)
    t0 = time.monotonic()
    assert ds.runner_main(cfg, "1", "perf", cfg.branch) == 1
    assert time.monotonic() - t0 < 10
    log = ds.job_log(Path(cfg.op_dir), "1")
    (line,) = [ln for ln in log.splitlines() if ds.TRANSPORT_ERROR_MARKER in ln]
    assert line.endswith(f"did not start the entry within 1s -- killed; the entry never ran")
    assert "the GPU lock command " + fakelock + " did not" in line
    assert ds.BEGIN_MARKER not in log and "exceeded" not in log
    assert subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True, text=True).stdout.strip() == ""


# -- ssh ------------------------------------------------------------------------------------


def test_the_lock_command_is_for_local_delivery_only(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.JOB_LOCK_ENV, LOCK)
    cfg = _ssh_cfg(tmp_path)
    fake = _FakeSsh([(0, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_SUCCEEDED
    remote = fake.execs[0][-1]
    assert "with-lock" not in remote and 'echo FK_JOB_BEGIN' in remote
    assert remote.endswith("/data/fk/.gpu1.lock bash gpu_entry.sh perf")
