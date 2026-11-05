"""GPU jobs: one detached runner per job, its state in plain files.

``submit`` allocates a job id, detaches a runner process and prints ``job=<id>``; the
runner runs one gpu_entry task (peregrine_kernel.submit.direct: this machine, or one ssh host)
and writes the job's log and, on every exit path, its verdict. ``status`` / ``logs`` /
``stop`` read and act on those files, so the loop can poll a job without a server, and a
job outlives the process that submitted it (a restarted loop adopts it).

State lives in ``operators/<op>/.jobs/<id>/``:

* ``log.txt``     -- what gpu_entry printed (closed before the verdict is written, so
  the log of a finished job is whole);
* ``result.json`` -- ``{"id": <id>, "status": "Succeeded" | "Failed"}``;
* ``runner.pid``  -- the runner's pid (a session leader: stopping it stops its group);
* ``runner.log``  -- the runner process's own output (a crash before the log opens).

A job id is the submit time in epoch seconds, bumped past collisions.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from peregrine_kernel.submit.config import SubmitConfig

JOBS_DIRNAME = ".jobs"
LOCK_NAME = "lock"

STATUS_RUNNING = "Running"
STATUS_SUCCEEDED = "Succeeded"
STATUS_FAILED = "Failed"
# The runner is gone and left no verdict (a reboot, an OOM kill): nothing says the job
# ran, so the loop resubmits it rather than charging the kernel.
STATUS_LOST = "Lost"

# The first line a job's command prints on the GPU machine: everything before it is
# transport (the copy, the ssh session), everything after it is the job.
BEGIN_MARKER = "PK_JOB_BEGIN"
# The job never reached the GPU machine: the copy or the ssh session failed every
# attempt. Not the kernel's failure -- the loop reports it as a transport problem.
TRANSPORT_ERROR_MARKER = "[jobs] TRANSPORT ERROR (the job could not be delivered):"
# A card's gpu_entry.sh refuses to time when its declared measurement clock lock does
# not hold on the GPU. Matched at the START of a line only: a probe that runs a nested
# gpu_entry prints it indented. Keep the wording in sync with cardkit.py's template.
CLOCK_LOCK_REFUSED_MARKER = "[gpu_entry.sh] REFUSED: this card times under a configured clock lock"

# How long past the task's own timeout the runner waits before killing it.
RUN_HEADROOM_S = 180

# Exception shapes that mean the machine or its tooling failed UNDER the runner
# (a file racing a deletion, a full disk, mkdir refused, git refusing the
# archive, a spawn failure): no kernel code was judged, so the verdict is
# Lost -- the loop resubmits instead of charging the iteration. Anything else
# (a mode string shlex cannot split, ...) is how the caller asked, not how the
# machine answered: Failed, and named as a usage error in the log (a Lost
# would only resubmit a deterministic mistake forever).
INFRA_FAULTS = (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError)


def clock_lock_refusal(log: str | None) -> str | None:
    """The outer gpu_entry's clock-lock refusal line in ``log`` (stripped), else None."""
    for line in (log or "").splitlines():
        if line.startswith(CLOCK_LOCK_REFUSED_MARKER):
            return line.strip()
    return None


def jobs_dir(op_dir: Path) -> Path:
    return Path(op_dir) / JOBS_DIRNAME


def job_dir(op_dir: Path, job_id: str) -> Path:
    return jobs_dir(op_dir) / str(job_id)


def result_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "result.json"


def log_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "log.txt"


def pid_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "runner.pid"


def is_job(op_dir: Path, job_id: str) -> bool:
    return str(job_id).isdigit() and job_dir(op_dir, job_id).is_dir()


def alloc_job_id(op_dir: Path, now: float | None = None) -> str:
    """Epoch-seconds job id, bumped past collisions."""
    jid = int(now if now is not None else time.time())
    base = jobs_dir(op_dir)
    base.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            (base / str(jid)).mkdir()
            return str(jid)
        except FileExistsError:
            jid += 1


@contextlib.contextmanager
def locked(op_dir: Path):
    """One exclusive lock per operator tree: its jobs run one at a time (they share the
    tree's GPU, and the extracted tree)."""
    d = jobs_dir(op_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LOCK_NAME, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def proc_start_ticks(pid: int) -> str | None:
    """The process's /proc starttime (``stat`` field 22, clock ticks since boot):
    the identity a pid file carries so a pid reused by an unrelated process is
    not mistaken for the job's runner. None when it cannot be read (the process
    is gone, or this machine has no /proc)."""
    try:
        data = (Path("/proc") / str(pid) / "stat").read_bytes().decode("ascii",
                                                                       "replace")
    except (OSError, ValueError):
        return None
    tail = data.rpartition(")")[2].split()   # past comm, which may hold spaces
    return tail[19] if len(tail) > 19 else None   # field 22, seen from field 3


def _runner_record(op_dir: Path, job_id: str) -> tuple[int, str | None]:
    """The pid file's ``(pid, identity)``; identity is None for an older, pid-only
    file. A missing or unreadable file reads ``(0, None)``."""
    try:
        parts = pid_path(op_dir, job_id).read_text(encoding="utf-8").split()
        pid = int(parts[0])
    except (OSError, ValueError):
        return 0, None
    return pid, (parts[1] if len(parts) > 1 else None)


def _pid_is_job(pid: int, identity: str | None) -> bool:
    """True when ``pid`` is (still) the process the job's pid file names. With no
    identity on record (an older job, or a machine without /proc) the old
    signal-0 semantics stand; with one, the /proc starttime must match -- a pid
    whose occupant changed is somebody else's process, not this job's runner."""
    if not _pid_alive(pid):
        return False
    if identity is None or not Path("/proc").is_dir():
        return True
    return proc_start_ticks(pid) == identity


def job_status(op_dir: Path, job_id: str) -> dict:
    """``{"id", "status"}`` for a job: its verdict once written, Running while its runner
    lives, Lost when the runner is gone without one."""
    res = result_path(op_dir, job_id)
    if res.is_file():
        try:
            return json.loads(res.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"id": int(job_id), "status": STATUS_FAILED}
    pid, identity = _runner_record(op_dir, job_id)
    if pid and _pid_is_job(pid, identity):
        return {"id": int(job_id), "status": STATUS_RUNNING}
    return {"id": int(job_id), "status": STATUS_LOST}


def job_log(op_dir: Path, job_id: str) -> str:
    try:
        return log_path(op_dir, job_id).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def stop_job(op_dir: Path, job_id: str) -> bool:
    """SIGTERM a running job's runner group (the runner records it as Failed). True when a
    signal was sent."""
    if job_status(op_dir, job_id).get("status") != STATUS_RUNNING:
        return False
    pid, identity = _runner_record(op_dir, job_id)
    if not pid or not _pid_is_job(pid, identity):
        return False    # the pid is gone, or belongs to somebody else now
    try:
        # the runner is a session leader, so its pid is the group: the ssh session and
        # gpu_entry go with it
        os.killpg(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True


def spawn_runner(cfg: SubmitConfig, mode: str, branch: str) -> str:
    """Detach a runner for one mode of ``branch``; returns the new job id.

    The runner re-execs ``peregrine_kernel.submit <op> _run`` in its own session so it
    survives the submitting process (the loop submits and then polls)."""
    op_dir = Path(cfg.op_dir)
    jid = alloc_job_id(op_dir)
    with open(job_dir(op_dir, jid) / "runner.log", "w") as runner_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "peregrine_kernel.submit", op_dir.name, "_run", jid,
             "--mode", mode, "--branch", branch, "--base-dir", str(op_dir.parents[1])],
            stdout=runner_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    start = proc_start_ticks(proc.pid)
    pid_path(op_dir, jid).write_text(
        f"{proc.pid} {start}" if start else str(proc.pid), encoding="utf-8")
    return jid


def runner_main(cfg: SubmitConfig, job_id: str, mode: str, branch: str) -> int:
    """The detached runner: one mode, its verdict to result.json on every exit path."""
    from peregrine_kernel.submit.direct import DirectRunner

    op_dir = Path(cfg.op_dir)
    status = STATUS_FAILED
    log_path(op_dir, job_id).parent.mkdir(parents=True, exist_ok=True)
    # SIGTERM (`submit stop`) must become SystemExit so the finally below still writes a
    # verdict: a stopped job with none would read back as Lost and be resubmitted.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with open(log_path(op_dir, job_id), "a", encoding="utf-8") as lf:
            try:
                status = DirectRunner(cfg).run_mode(branch, mode, lf)
            except SystemExit:
                lf.write("[jobs] runner stopped (SIGTERM) -- recorded as Failed so nothing "
                         "resubmits it\n")
                raise
            except Exception as e:  # noqa: BLE001 -- the verdict must still be written
                if isinstance(e, INFRA_FAULTS):
                    status = STATUS_LOST
                    lf.write(f"[jobs] runner infrastructure fault "
                             f"({type(e).__name__}: {e}) -- recorded {STATUS_LOST}: "
                             f"the loop resubmits, the kernel is not charged\n")
                else:
                    status = STATUS_FAILED
                    lf.write(f"[jobs] usage error in the runner request "
                             f"({type(e).__name__}: {e}) -- recorded {STATUS_FAILED} "
                             f"(NOT a kernel failure; fix the caller)\n")
    finally:
        res = result_path(op_dir, job_id)
        tmp = res.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": int(job_id), "status": status}), encoding="utf-8")
        tmp.replace(res)
    return 0 if status == STATUS_SUCCEEDED else 1
