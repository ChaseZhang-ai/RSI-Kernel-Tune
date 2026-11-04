"""Run one gpu_entry task on this machine (``local``) or on one ssh host (``ssh``).

A job's bookkeeping (its id, log, verdict, the detached runner) is peregrine_kernel.submit.jobs;
this module is what the runner does.

One run:

1. ``git archive`` the requested branch of the op's tree and extract it fresh -- locally
   under ``<op>/.exec/tree``, or on the host
   under ``<workdir>/<run>/<op>`` -- so a run never inherits the last run's build products.
   (WP11: on the local delivery the tree's large committed input tensors may travel
   through a content-addressed store instead of every archive -- see INPUTS_ below;
   the extracted tree is byte-identical either way, and any inconsistency falls back
   to the full archive.)
2. Take the GPU lock, then ``bash gpu_entry.sh <mode>`` with the tree's ``[env]``,
   ``CUDA_VISIBLE_DEVICES`` and ``TASK_TIMEOUT``. The lock is per GPU and per machine
   (two runs on one GPU would time each other), shared by every op and every run.
3. Map the exit: rc 0 -> Succeeded; the task ran and failed -> Failed; ssh could not
   deliver it (no begin marker) -> retried, then Failed with TRANSPORT_ERROR_MARKER, which
   the loop reports as a transport problem rather than charging the kernel. A job whose
   window expired while it was still waiting on the host GPU lock never began (the begin
   marker prints only once the lock is held): it is Lost, which the loop resubmits. So is
   a job whose ssh client died (rc 255) with the job in flight: the remote session's
   outcome is unknowable, and the orphaned session is stopped before the verdict.

Nothing here kills processes it did not start: on a user's machine a GPU held by someone
else is theirs. The lock is what keeps peregrine_kernel's own runs apart.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import lzma
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

from peregrine_kernel.submit.config import SubmitConfig, is_secret_env
from peregrine_kernel.submit.jobs import (BEGIN_MARKER, RUN_HEADROOM_S, STATUS_FAILED,
                                     STATUS_LOST, STATUS_SUCCEEDED,
                                     TRANSPORT_ERROR_MARKER, locked)

LOCAL_TREE = ".exec/tree"
CMD_PREFIX = "cmd:"
# On the ssh host a job runs in a session of its own, and its session id is written here
# (in the job's tree): stopping the job ends its whole session. A process group is not
# enough -- gpu_entry's `timeout` moves itself and the task into a group of their own.
SESSION_FILE = ".pk_job.sid"
# Written in the session the job runs in (before it waits on the lock, so a merely
# queued session is still stoppable). _session_script builds it: the session id,
# then gpu_entry under the GPU lock ("$0" is the lock file, "$@" the entry command).
# The begin marker prints only once flock HOLDS the lock -- everything before it
# (ssh setup, the whole wait on a busy host GPU lock) is transport, never the job,
# and never counted against the task's deadline. The queue wait is bounded by
# flock -w (its own budget, independent of the run deadline): on expiry the session
# exits QUEUE_TIMEOUT_RC before any marker, and the runner reads that as a
# queued-out job (Lost), not a failed one. The lock travels across the final exec
# on fd 9 (inherited; no close-on-exec), so gpu_entry holds it for its whole run.
STOP_GRACE_S = 5
# flock -E's exit code when the -w wait for the host GPU lock expires (util-linux
# >= 2.24; an older host exits 1 instead and degrades to the retry path). Only
# ever interpreted when no begin marker was printed: the queue path is strictly
# pre-marker, so it cannot be a gpu_entry exit code.
QUEUE_TIMEOUT_RC = 190


def _session_script(queue_budget_s: float) -> str:
    return ('echo $$ > ' + SESSION_FILE
            + ' && exec 9>"$0"'
            + f' && flock -w {max(1, int(queue_budget_s))} -E {QUEUE_TIMEOUT_RC} 9'
            + ' && echo ' + BEGIN_MARKER
            + ' && exec "$@"')


LOCK_DIR_ENV = "PEREGRINE_LOCK_DIR"
TRANSPORT_ATTEMPTS = 3
TRANSPORT_RETRY_S = 10.0
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6")

# --- WP11 (task 1): the merged eval+NCU job -----------------------------------
#
# gpu_entry.sh's `allncu` task runs the timing protocol (exactly the `all` task)
# and then the NCU segment in its own process; `allncu:<command>` names the NCU
# segment's command (the vendored orchestration census runner, mirroring cmd:).
# The job's own budget is the SUM of the two segments: the eval task keeps its
# full TASK_TIMEOUT, the NCU segment keeps its own NCU_TIMEOUT (gpu_entry.sh's
# default below) -- the runner's deadline is sized to the same sum, so the
# single-segment timeout is never stretched over two segments.
MERGED_EVAL_NCU_PREFIX = "allncu"
MERGED_NCU_TIMEOUT_DEFAULT = 600.0   # == gpu_entry.sh's NCU_TIMEOUT default


def _merged_mode(mode: str) -> bool:
    return mode.split(":", 1)[0] == MERGED_EVAL_NCU_PREFIX


# --- WP11 (task 3): content-addressed input-tensor transport (local delivery) --
#
# Every job used to archive the op tree's committed input tensors (safetensors,
# GB scale on attention-class ops) and extract them again, per job. On the local
# delivery the runner instead syncs them once into a content-addressed store
# (~/.cache/peregrine_kernel-inputs/<sha256>, overridable PK_INPUTS_STORE), archives
# the tree with the inputs excluded, and materializes them back into the fresh
# tree after extraction -- each file verified against its sha256, any gap or
# mismatch falling back to the full archive. The tree a job sees is
# byte-identical either way; the store is a cache, never a correctness single
# point. PK_INPUTS_SYNC=0 (job env or the runner's environment) disables it; the
# ssh delivery keeps the full archive (cross-machine sync is not this package).
INPUTS_SYNC_MIN_BYTES = 1 << 20   # files at/above this ride the store (the tax is per byte)
INPUTS_DIR_SEGMENT = "inputs"     # a path component: <...>/inputs/<...> is input-tensor territory
INPUTS_STORE_ENV = "PK_INPUTS_STORE"
INPUTS_SYNC_ENV = "PK_INPUTS_SYNC"
# --- WP19 (task 1 addendum, A-G24/B-B7): inputs manifest cost cut ------------------------
# Every job used to read every input file twice in full (the source-side sha,
# then the store-side recheck) — GB-scale tensors, per job. The source-side
# hash is now memoized in a manifest keyed by (path, size, mtime_ns): an
# unchanged file is not re-read. The store-side recheck becomes a SAMPLED full
# hash (restore still hash-verifies every byte it copies, and the job falls
# back to the full archive on any mismatch, so a corrupted store entry can
# never reach a tree — the sampling only decides whether to re-copy). Any
# manifest trouble (missing, corrupt, unwritable) falls back to the full hash:
# the memo is a cache, never a correctness single point.
INPUTS_MANIFEST_NAME = "source-manifest.json"
INPUTS_VERIFY_SAMPLE_STRIDE = 4   # full-hash this store entry in 1 of every N jobs


def inputs_store_dir() -> Path:
    env = os.environ.get(INPUTS_STORE_ENV, "").strip()
    return Path(env) if env else Path.home() / ".cache" / "peregrine_kernel-inputs"


def _inputs_manifest_path(store: Path) -> Path:
    return store / INPUTS_MANIFEST_NAME


def _inputs_load_manifest(store: Path) -> dict:
    try:
        with open(_inputs_manifest_path(store), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _inputs_save_manifest(store: Path, data: dict) -> None:
    try:
        tmp = _inputs_manifest_path(store).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, _inputs_manifest_path(store))
    except OSError:
        pass


def _inputs_sync_enabled(cfg: SubmitConfig) -> bool:
    for source in (os.environ, cfg.env):
        if str(source.get(INPUTS_SYNC_ENV, "")).strip() == "0":
            return False
    return True


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inputs_manifest_entry(op_dir: Path, path: Path, memo: dict) -> dict | None:
    """One sync candidate: ``{path (in-tree), sha256, size}``, or None.

    WP19 (task 1 addendum): the source-side hash is memoized by (size, mtime_ns) —
    an unchanged file is not re-read. A stat mismatch, a missing memo or ANY
    trouble falls back to the full hash (and refreshes the memo)."""
    try:
        rel = path.relative_to(op_dir).as_posix()
        st = path.stat()
        size, mtime = st.st_size, st.st_mtime_ns
        cached = memo.get(rel)
        if (isinstance(cached, dict) and cached.get("size") == size
                and cached.get("mtime_ns") == mtime
                and isinstance(cached.get("sha256"), str)):
            h = cached["sha256"]
        else:
            h = _file_sha256(path)
            memo[rel] = {"size": size, "mtime_ns": mtime, "sha256": h}
    except (OSError, ValueError):
        return None
    return {"path": rel, "sha256": h, "size": size}


def sync_inputs_to_store(cfg: SubmitConfig, store: Path | None = None) -> list | None:
    """The op tree's large input tensors, content-addressed into the store.

    Returns the manifest ([{path, sha256}, ...]) when there is something to sync
    (the caller then archives with the inputs excluded), [] when the tree has no
    such files (archive everything, as always), None when sync is disabled. An
    existing store entry with the matching hash is left alone (WP19: the
    recheck is sampled — every INPUTS_VERIFY_SAMPLE_STRIDE-th candidate gets a
    full hash, the rest a size check; restore's own hash-verified copy is the
    safety net); anything that cannot be synced is left OUT of the manifest, so
    the archive keeps that file (partial sync degrades file-by-file, never to a
    broken tree)."""
    if not _inputs_sync_enabled(cfg):
        return None
    op_dir = Path(cfg.op_dir)
    if not op_dir.is_dir():
        return None
    store = store if store is not None else inputs_store_dir()
    try:
        store.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    memo = _inputs_load_manifest(store)
    manifest: list[dict] = []
    for idx, path in enumerate(sorted(op_dir.rglob("*"))):
        if not path.is_file() or path.stat().st_size < INPUTS_SYNC_MIN_BYTES:
            continue
        if INPUTS_DIR_SEGMENT not in path.relative_to(op_dir).parts:
            continue
        entry = _inputs_manifest_entry(op_dir, path, memo)
        if entry is None:
            continue
        dest = store / entry["sha256"]
        ok = dest.is_file()
        if ok and idx % INPUTS_VERIFY_SAMPLE_STRIDE == 0:
            ok = _file_sha256(dest) == entry["sha256"]   # sampled full recheck
        elif ok:
            try:
                ok = dest.stat().st_size == entry["size"]
            except OSError:
                ok = False
        if not ok:
            tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
            try:
                shutil.copyfile(path, tmp)
                os.replace(tmp, dest)
            except OSError:
                continue   # this file rides the archive instead
        manifest.append(entry)
    _inputs_save_manifest(store, memo)
    return manifest


def _git_archive(cfg: SubmitConfig, branch: str, *, exclude_inputs: bool = False) -> bytes:
    # The tree is this run's own: trusted even when its files belong to another uid (a
    # container's root reading a checkout it did not create), which git would refuse.
    argv = ["git", "-c", f"safe.directory={cfg.op_dir}", "-C", cfg.op_dir,
            "archive", "--format=tar", branch]
    if exclude_inputs:
        # WP11 (task 3): the inputs ride the store, not the archive. Excluded at
        # the git level, so the payload never carries them (a post-hoc strip would
        # still have paid the read); any pathspec trouble falls back to the full
        # archive at the caller.
        argv += ["--", ":", ":(glob,exclude)**/inputs/**"]
    proc = subprocess.run(argv, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {branch} failed in {cfg.op_dir}: "
                           f"{proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _payload(cfg: SubmitConfig, branch: str, manifest: list | None = None) -> bytes:
    return _git_archive(cfg, branch, exclude_inputs=bool(manifest))


def restore_inputs_from_store(tree: Path, manifest: list,
                              store: Path | None = None) -> None:
    """Materialize the synced inputs back into a freshly extracted tree.

    Every file is verified against its manifest hash before it lands (the copy
    is byte-identical or the call raises); a missing or mismatched store entry
    raises, and the caller falls back to the full archive."""
    store = store if store is not None else inputs_store_dir()
    for entry in manifest:
        src = store / entry["sha256"]
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"input store hash mismatch: {entry['path']}")
        dest = tree / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        try:
            os.chmod(dest, 0o644)
        except OSError:
            pass


def lock_dir() -> Path:
    """Where this machine's GPU locks live (one directory for every peregrine_kernel run)."""
    env = os.environ.get(LOCK_DIR_ENV, "").strip()
    return Path(env) if env else Path.home() / ".cache" / "peregrine_kernel" / "locks"


@contextmanager
def gpu_lock(gpu: str, log_file=None, poll_s: float = 30.0):
    """Hold this machine's lock on GPU ``gpu`` for the duration; say so while waiting.

    Yields the seconds spent waiting for the lock (0.0 when it was free) — WP11
    task 4's queue-wait telemetry reads it from the job log."""
    d = lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"gpu{gpu}.lock", "w") as fh:
        waited = 0.0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if log_file is not None and waited == 0.0:
                    log_file.write(f"[direct] GPU {gpu} is in use by another peregrine_kernel run "
                                   f"-- waiting\n")
                    log_file.flush()
                time.sleep(poll_s)
                waited += poll_s
        try:
            yield waited
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def remote_tree(cfg: SubmitConfig) -> str:
    """``<workdir>/<run>/<op>-<hash>``: one tree per op per clone, so concurrent
    runs on one host never share a directory. The run segment is only a basename,
    so two clones of the same name would still collide; the clone's absolute
    path is hashed into the tree's name to keep them apart (stable for the same
    clone, so re-delivery refreshes its own tree and _stop_cmd finds the same
    one)."""
    op_dir = Path(cfg.op_dir)
    if len(op_dir.parents) > 1:
        run = op_dir.parents[1].name
        clone = str(op_dir.parents[1])
    else:
        run = "repo"
        clone = str(op_dir)
    token = hashlib.sha1(clone.encode()).hexdigest()[:8]
    return f"{cfg.exec_workdir.rstrip('/')}/{run}/{op_dir.name}-{token}"


def _entry_argv(cfg: SubmitConfig, mode: str) -> list[str]:
    if mode.startswith(CMD_PREFIX):
        return ["bash", cfg.entry_script, "cmd", *shlex.split(mode[len(CMD_PREFIX):])]
    if mode.startswith(f"{MERGED_EVAL_NCU_PREFIX}:"):
        # allncu:<ncu command> — the merged task's NCU segment is the named
        # command (e.g. the vendored orchestration census runner), passed as
        # arguments exactly like the cmd task's (WP11 task 1).
        return ["bash", cfg.entry_script, MERGED_EVAL_NCU_PREFIX,
                *shlex.split(mode[len(MERGED_EVAL_NCU_PREFIX) + 1:])]
    return ["bash", cfg.entry_script, mode]


def _job_env(cfg: SubmitConfig, tree_sha: str | None = None) -> dict:
    env = {**cfg.env, "TASK_TIMEOUT": str(cfg.timeout), "CUDA_VISIBLE_DEVICES": cfg.exec_gpu}
    if tree_sha:
        # WP11 (task 2): the content key of the machine-level build cache (the
        # job tree's payload hash). The card's gpu_entry.sh combines it with the
        # job's own toolchain versions (nvcc, torch) when PK_BUILD_CACHE=1; a job
        # env that does not know the key simply ignores it.
        env["PK_TREE_SHA256"] = tree_sha
    return env


class DirectRunner:
    """Runs one mode of ``cfg``'s tree; ``run_mode`` returns the job's status."""

    def __init__(self, cfg: SubmitConfig, run=subprocess.run, popen=subprocess.Popen,
                 sleep=time.sleep):
        self.cfg = cfg
        self._run = run
        self._popen = popen
        self._sleep = sleep

    @property
    def deadline_s(self) -> float:
        return self._deadline_for("")

    def _deadline_for(self, mode: str) -> float:
        """The runner's kill deadline for one job. A merged eval+NCU job's budget
        is the SUM of its two segments (WP11 task 1), each segment keeping the
        budget its two-job path had: the eval task's full timeout (the job's
        TASK_TIMEOUT) plus — for a plain merged job — the NCU segment's own
        NCU_TIMEOUT (gpu_entry.sh's default here), or — for the census variant
        (``allncu:<cmd>``) — the census command's own TASK_TIMEOUT, the old cmd
        task's budget. The single-segment timeout is never stretched over two
        segments, and the runner never kills a merged job mid-profile for time
        it legitimately asked for."""
        base = float(self.cfg.timeout) + RUN_HEADROOM_S
        if not _merged_mode(mode):
            return base
        if mode.startswith(f"{MERGED_EVAL_NCU_PREFIX}:"):
            return base + float(self.cfg.timeout)   # census segment: TASK_TIMEOUT
        raw = str(_job_env(self.cfg).get("NCU_TIMEOUT", "") or MERGED_NCU_TIMEOUT_DEFAULT)
        try:
            ncu = float(raw)
        except ValueError:
            ncu = MERGED_NCU_TIMEOUT_DEFAULT
        return base + max(0.0, ncu)                 # NCU segment: NCU_TIMEOUT

    def _queue_budget_s(self, mode: str) -> float:
        """The remote session's own budget for WAITING on the host GPU lock
        (``flock -w``), independent of the task's run deadline: time spent queued
        is not time the task ran (the local path starts its deadline only after
        its lock is held). Sized to the run deadline -- a queued job gets as long
        to reach the lock as it would then get to run; the runner's total window
        for the ssh job is queue + run."""
        return self._deadline_for(mode)

    def run_mode(self, branch: str, mode: str, log_file) -> str:
        with locked(Path(self.cfg.op_dir)):
            manifest = None
            if self.cfg.delivery == "local" and _inputs_sync_enabled(self.cfg):
                # WP11 (task 3): sync the tree's large inputs into the
                # content-addressed store first; an empty manifest (no inputs)
                # keeps the plain archive, and a None (disabled) never strips.
                manifest = sync_inputs_to_store(self.cfg) or None
            payload = _payload(self.cfg, branch, manifest)
            tree_sha = hashlib.sha256(payload).hexdigest()
            if self.cfg.delivery == "local":
                return self._run_local(payload, branch, mode, log_file, manifest)
            return self._run_ssh(payload, branch, mode, log_file, tree_sha)

    # -- local ----------------------------------------------------------------------------

    def _run_local(self, payload: bytes, branch: str, mode: str, log_file,
                   manifest: list | None = None) -> str:
        tree = Path(self.cfg.op_dir) / LOCAL_TREE
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir(parents=True)
        _extract(payload, tree)
        if manifest:
            # WP11 (task 3): put the synced inputs back, hash-verified. ANY gap —
            # a store entry missing, a hash mismatch, an unwritable path — falls
            # back to the full archive: the cache never becomes a correctness
            # single point, and the tree the job sees is byte-identical either way.
            try:
                restore_inputs_from_store(tree, manifest)
            except (OSError, RuntimeError) as e:
                log_file.write(f"[direct] inputs sync failed ({e}) — re-delivering "
                               f"the full archive\n")
                log_file.flush()
                shutil.rmtree(tree, ignore_errors=True)
                tree.mkdir(parents=True)
                _extract(_git_archive(self.cfg, branch), tree)
        env = {**os.environ, **_job_env(self.cfg, hashlib.sha256(payload).hexdigest())}
        log_file.write(f"[direct] local GPU {self.cfg.exec_gpu} branch={branch} mode={mode}\n")
        log_file.flush()
        with gpu_lock(self.cfg.exec_gpu, log_file) as lock_wait:
            # WP11 (task 4): how long this job sat behind another run's GPU lock
            # (poll granularity 30 s) — the queue wait the driver's telemetry reads.
            if lock_wait:
                log_file.write(f"[direct] GPU {self.cfg.exec_gpu} acquired after "
                               f"{lock_wait:.0f}s wait\n")
                log_file.flush()
            log_file.write(f"{BEGIN_MARKER}\n")
            log_file.flush()
            proc = self._popen(_entry_argv(self.cfg, mode), cwd=tree, env=env,
                               stdout=log_file, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = self._deadline_for(mode)
            try:
                rc = proc.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                _kill_session(proc)
                log_file.write(f"[direct] gpu_entry {mode} exceeded {deadline:.0f}s "
                               f"-- killed\n")
                log_file.flush()
                return STATUS_FAILED
            except BaseException:   # stopped (SIGTERM -> SystemExit) or interrupted
                _kill_session(proc)
                raise
        if rc == 0:
            return STATUS_SUCCEEDED
        log_file.write(f"[direct] gpu_entry {mode} exited rc={rc}\n")
        log_file.flush()
        return STATUS_FAILED

    # -- ssh ------------------------------------------------------------------------------

    def _ssh(self, remote_cmd: str) -> list[str]:
        return ["ssh", *SSH_OPTS, self.cfg.exec_host, remote_cmd]

    def _extract_cmd(self) -> str:
        d = shlex.quote(remote_tree(self.cfg))
        return f"rm -rf {d} && mkdir -p {d} && tar xJ -C {d}"

    def _exec_cmd(self, mode: str) -> str:
        env = _job_env(self.cfg)
        secrets = sorted(k for k in env if is_secret_env(k))
        parts = [f"cd {shlex.quote(remote_tree(self.cfg))}"]
        if self.cfg.exec_env_file:
            parts.append(f". {shlex.quote(self.cfg.exec_env_file)}")
        parts += [f"export {k}={shlex.quote(str(v))}" for k, v in sorted(env.items())
                  if k not in secrets]
        # a secret is read off stdin, never spelled on a command line `ps` shows
        parts += [f"{{ IFS= read -r {k} || true; }} && export {k}" for k in secrets]
        lock = shlex.quote(f"{self.cfg.exec_workdir.rstrip('/')}/.gpu{self.cfg.exec_gpu}.lock")
        entry = " ".join(shlex.quote(a) for a in _entry_argv(self.cfg, mode))
        script = shlex.quote(_session_script(self._queue_budget_s(mode)))
        parts.append(f"setsid --wait sh -c {script} {lock} {entry}")
        return " && ".join(parts)

    def _stop_cmd(self) -> str:
        f = shlex.quote(f"{remote_tree(self.cfg)}/{SESSION_FILE}")
        return (f"[ -f {f} ] || exit 0; s=$(cat {f}); pkill -TERM -s \"$s\"; "
                f"sleep {STOP_GRACE_S}; pkill -KILL -s \"$s\"; rm -f {f}; exit 0")

    def _stop_remote(self, log_file) -> None:
        """End the job's session on the host (the ssh client going away does not)."""
        try:
            self._run(self._ssh(self._stop_cmd()), capture_output=True, timeout=120)
            log_file.write(f"[direct] stopped the job's session on {self.cfg.exec_host}\n")
        except Exception as e:  # noqa: BLE001 -- best effort; the verdict is written either way
            log_file.write(f"[direct] could not stop the job on {self.cfg.exec_host}: {e!r}\n")
        log_file.flush()

    def _run_ssh(self, payload: bytes, branch: str, mode: str, log_file,
                 tree_sha: str | None = None) -> str:
        # WP11 (task 3): the ssh delivery keeps the FULL archive — the input
        # store is a same-machine cache; cross-machine sync is not this package.
        # WP19 (task 8, B-B10): preset 6 -> 3 — the slowest lzma tier bought
        # nothing on tar archives (already-compressed tensors dominate), and the
        # compression CPU was per-job minutes on GB-scale inputs. Transparent
        # to the host: it still extracts `tar xJ`.
        packed = lzma.compress(payload, preset=3)
        env = _job_env(self.cfg, tree_sha)
        secrets = "".join(f"{env[k]}\n" for k in sorted(env) if is_secret_env(k)) or None
        failures: list[str] = []
        while True:
            sync = self._run(self._ssh(self._extract_cmd()), input=packed, capture_output=True)
            if sync.returncode != 0:
                err = sync.stderr.decode(errors="replace") if isinstance(sync.stderr, bytes) \
                    else str(sync.stderr or "")
                failures.append(f"sync to {self.cfg.exec_host} (rc={sync.returncode}): "
                                f"{err.strip()[-200:]}")
            else:
                log_file.write(f"[direct] ssh {self.cfg.exec_host} GPU {self.cfg.exec_gpu} "
                               f"branch={branch} mode={mode}\n")
                log_file.flush()
                start = log_file.tell() if hasattr(log_file, "tell") else None
                proc = self._popen(self._ssh(self._exec_cmd(mode)), stdout=log_file,
                                   stderr=subprocess.STDOUT, text=True,
                                   stdin=subprocess.PIPE if secrets else subprocess.DEVNULL,
                                   start_new_session=True)
                if secrets:
                    try:
                        proc.stdin.write(secrets)
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    deadline = self._deadline_for(mode)
                    queue_budget = self._queue_budget_s(mode)
                    # the queue wait has its own budget (the session's flock -w),
                    # so the runner's window is queue + run, never queue instead
                    # of run
                    rc = proc.wait(timeout=deadline + queue_budget)
                except subprocess.TimeoutExpired:
                    self._stop_remote(log_file)
                    _kill_group(proc)
                    if _began(log_file, start, BEGIN_MARKER):
                        log_file.write(f"[direct] gpu_entry {mode} exceeded "
                                       f"{deadline:.0f}s -- killed\n")
                        log_file.flush()
                        return STATUS_FAILED
                    # no begin marker: the job never began -- it queued on the
                    # host GPU lock (or the ssh session stalled before it) for
                    # its whole window. Nothing ran, so this is not the kernel's
                    # failure: Lost, which the loop resubmits.
                    log_file.write(f"[direct] gpu_entry {mode} queued on the GPU lock for "
                                   f"its whole {deadline + queue_budget:.0f}s window -- "
                                   f"nothing ran\n")
                    log_file.flush()
                    return STATUS_LOST
                except BaseException:   # stopped (SIGTERM -> SystemExit) or interrupted
                    self._stop_remote(log_file)
                    _kill_group(proc)
                    raise
                if rc == 0:
                    return STATUS_SUCCEEDED
                began = _began(log_file, start, BEGIN_MARKER)
                if rc == QUEUE_TIMEOUT_RC and not began:
                    # the session's flock -w expired: the job queued out without
                    # ever beginning (the marker prints only once the lock is
                    # held). Not the kernel's failure: Lost, resubmitted.
                    log_file.write(f"[direct] gpu_entry {mode} queued on the GPU lock past "
                                   f"its {self._queue_budget_s(mode):.0f}s queue budget -- "
                                   f"nothing ran\n")
                    log_file.flush()
                    return STATUS_LOST
                if began:
                    if rc == 255:
                        # ssh's own exit code: the client died (the network blip
                        # outlived ServerAlive's patience) while the job was in
                        # flight on the host -- the session kept running, and its
                        # outcome is unknowable. The machine's fault, not the
                        # kernel's: end the orphaned session (it still holds the
                        # GPU lock) and record Lost, which the loop resubmits.
                        self._stop_remote(log_file)
                        _kill_group(proc)
                        log_file.write(f"[direct] ssh to {self.cfg.exec_host} died (rc=255) "
                                       f"with the job in flight -- the outcome on the host "
                                       f"is unknown, recorded {STATUS_LOST}\n")
                        log_file.flush()
                        return STATUS_LOST
                    log_file.write(f"[direct] gpu_entry {mode} exited rc={rc}\n")
                    log_file.flush()
                    return STATUS_FAILED
                failures.append(f"gpu_entry {mode} on {self.cfg.exec_host} (rc={rc}, never "
                                f"started)")
            if len(failures) >= TRANSPORT_ATTEMPTS:
                log_file.write(f"{TRANSPORT_ERROR_MARKER} {len(failures)} transport failure(s) "
                               f"on {self.cfg.exec_host}: {' | '.join(failures)}\n")
                log_file.flush()
                return STATUS_FAILED
            log_file.write(f"[direct] {failures[-1]} -- retrying in {TRANSPORT_RETRY_S:.0f}s\n")
            log_file.flush()
            self._sleep(TRANSPORT_RETRY_S)


def _extract(payload: bytes, dest: Path) -> None:
    """Extract a ``git archive`` tar into ``dest``, refusing members that leave it."""
    with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:   # a Python without extraction filters
            root = dest.resolve()
            for m in tf.getmembers():
                target = (dest / m.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"payload member {m.name!r} leaves the tree")
                if m.issym() or m.islnk():
                    link = (target.parent / m.linkname).resolve()
                    if link != root and root not in link.parents:
                        raise RuntimeError(f"payload link {m.name!r} leaves the tree")
            tf.extractall(dest)


def _began(log_file, start, marker: str) -> bool:
    """Did the remote command print the begin marker after ``start`` in ``log_file``?"""
    try:
        log_file.flush()
        with open(log_file.name, encoding="utf-8", errors="replace") as fh:
            if start:
                fh.seek(start)
            return any(line.strip() == marker for line in fh)
    except (OSError, AttributeError):
        return False


def _descendants(root: int) -> list[int]:
    """Every process below ``root`` right now (ps works the same on Linux and macOS)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        try:
            pid, ppid = (int(x) for x in line.split())
        except ValueError:
            continue
        kids.setdefault(ppid, []).append(pid)
    found, stack = [], [root]
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in found:
                found.append(child)
                stack.append(child)
    return found


def _kill_session(proc, grace_s: float = STOP_GRACE_S) -> None:
    """End ``proc`` and everything below it. Its process group is not enough: gpu_entry's
    `timeout` moves the task into a group of its own. The tree is read before the first
    signal, while every member still hangs below ``proc``."""
    tree = [proc.pid, *_descendants(proc.pid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in tree:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            time.sleep(min(grace_s, 1.0))   # the rest of the tree, a moment to go too
    _kill_group(proc)


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001 -- best effort; the verdict is written either way
        pass
