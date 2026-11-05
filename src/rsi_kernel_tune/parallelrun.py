"""``rsi-kernel-tune run --config rsi-run.toml``: every kernel task of the run
configuration, in parallel.

The unified configuration (rsi_kernel_tune.runconfig) lists WHAT to optimize;
this module is the concurrency around it. One kernel task = one run: the parent
snapshots the task with the configuration's [api] provider/models and
[precision] declaration merged into the snapshot's task.toml (so the child run
solves exactly the declared problem), creates its own run directory under
.artifacts/ (operators/ and ws/ are per-run -- runs never write into each
other), and drives it in a child process (``python -m rsi_kernel_tune run
<dir>``) whose environment merges local.env with the key file the [api] section
names -- the key file's values win over local.env's, and the process environment
wins over both, the localenv rule.

Concurrency: ``[run] parallel = "auto"`` gives every kernel task its own slot --
add a kernel task, add a slot. An explicit value caps the slots. The children's
GPU jobs serialize on the existing per-GPU lock either way; with several GPUs
the slots are assigned round-robin over the GPUs the machine reports
(RKT_GPU per slot), so each slot's jobs land on their own device where there
is one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rsi_kernel_tune import task as task_mod
from rsi_kernel_tune.localenv import load_machine
from rsi_kernel_tune.paths import repo_root
from rsi_kernel_tune.runconfig import RunConfig, RunConfigError
from rsi_kernel_tune.runs import Run, _SNAPSHOT_IGNORE, new_run, open_run

RETRIEVAL_ENV = "RKT_STUDENT_RETRIEVAL"


def _say(msg: str) -> None:
    print(f"[rsi-kernel-tune] {msg}", flush=True)


def _gpu_count(environ=None) -> int:
    """How many GPUs the machine reports (nvidia-smi; 1 when it cannot say --
    every slot then shares the declared GPU and the per-GPU lock serializes
    their jobs, as today)."""
    import shutil as _shutil

    smi = _shutil.which("nvidia-smi", path=os.environ.get("PATH") if environ is None
                        else environ.get("PATH"))
    if smi is None:
        return 1
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15)
        rows = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")]
        return len(rows) or 1
    except (OSError, subprocess.SubprocessError):
        return 1


@dataclass(frozen=True)
class KernelPlan:
    """One kernel task's slice of the run configuration: its (validated) task
    dir, the slot it runs in and the GPU that slot maps to."""

    task_dir: Path
    slot: int
    gpu: str


def plan(cfg: RunConfig, *, environ=None) -> list[KernelPlan]:
    """Every kernel of ``cfg`` as a runnable slice: the task dir must be a real
    task; a ``from_kernels`` entry needs its task scaffolded first (the kernel's
    frozen contract is the user's to transcribe -- the tool does not guess it),
    and the error names the exact command. Slot = index; GPU = slot round-robin
    over the machine's GPUs."""
    from rsi_kernel_tune.runconfig import _kernel_tree

    n_gpus = _gpu_count(environ)
    out: list[KernelPlan] = []
    for i, entry in enumerate(cfg.kernels):
        if entry in cfg.kernel.tasks:
            base = Path(entry).expanduser()
            d = base if base.is_absolute() else (cfg.path.parent / base)
            if not d.is_dir():
                raise RunConfigError(cfg.path, [f"[kernel] task {entry!r}: {d} is not a "
                                                f"directory (it validated when the config "
                                                f"was written; was it moved?)"])
        else:
            # a published kernel: the task must already be scaffolded from it
            op = Path(entry).name
            d = (repo_root() / "tasks" / op)
            if not (d / "task.toml").is_file():
                raise RunConfigError(cfg.path, [
                    f"[kernel] from_kernels {entry!r}: tasks/{op} is not scaffolded yet -- "
                    f"run `rsi-kernel-tune init {op} --from-kernels --arch <arch> --axes "
                    f'<{{...}}>\' first (the axes come from the kernel docstring\'s frozen '
                    f"contract; the run config points at it, it does not guess it)"])
        out.append(KernelPlan(task_dir=d.resolve(), slot=i, gpu=str(i % n_gpus)))
    return out


def _snapshot_task(task_dir: Path, cfg: RunConfig, tmp_root: Path) -> task_mod.Task:
    """The task as the configuration declares it: [api] provider / per-role models
    and [precision] compute_dtype merged into a copy's task.toml (the child run
    snapshots and re-reads that copy, so it solves exactly the declared problem).
    A task that declares its own precision may not be silently re-declared: a
    conflict between the two declarations is refused."""
    task = task_mod.load_task(task_dir)
    own = task_mod.read_task_toml(task_dir)
    top: dict = {}
    if cfg.api.provider != task.teacher.provider or cfg.api.provider != task.student.provider:
        top.setdefault("agent", {})["provider"] = cfg.api.provider
    for role in ("teacher", "student"):
        model = cfg.api.models.get(role)
        if model and model != getattr(task, role).model:
            top.setdefault("agent", {})[role] = {"model": model}
    if cfg.precision.compute_dtype != task.precision.compute_dtype:
        if task.precision.declared:
            raise RunConfigError(cfg.path, [
                f"[precision] {task.name} declares compute_dtype="
                f"{task.precision.compute_dtype} but the run configuration declares "
                f"{cfg.precision.compute_dtype} -- one precision per kernel task: change "
                f"the task's [precision], or the run configuration's"])
        top["precision"] = {"compute_dtype": cfg.precision.compute_dtype}
    if not top:
        return task
    merged = task_mod.merge_config(own, top)
    snap = tmp_root / task.name
    import shutil

    shutil.copytree(task_dir, snap, ignore=_SNAPSHOT_IGNORE)
    (snap / task_mod.TASK_FILE).write_text(task_mod.dump_task_toml(merged), encoding="utf-8")
    return task_mod.load_task(snap)


def _child_env(cfg: RunConfig, *, gpu: str, environ=None) -> dict:
    """The child's environment: the process environment over the key file over
    local.env (the key file's values win over local.env's -- the run
    configuration is the more explicit statement; the process wins over both,
    the localenv rule), RKT_GPU for this slot, and the retrieval permission."""
    environ = os.environ if environ is None else environ
    from rsi_kernel_tune.localenv import LOCAL_ENV, read_env_file

    merged: dict = {}
    for source in (read_env_file(Path(repo_root()) / LOCAL_ENV),
                   cfg.api.key_env(cfg.path.parent)):
        for k, v in source.items():
            if v:
                merged[k] = v
    for k, v in environ.items():
        merged[k] = v
    merged["RKT_GPU"] = gpu
    if cfg.student.may_retrieve:
        merged[RETRIEVAL_ENV] = "1"
    if cfg.language.syntax_docs:
        # the [language] syntax reference, injected into every teacher and student
        # brief by the loop's agent policy ("read it first")
        merged["RKT_RUN_SYNTAX_DOCS"] = cfg.language.syntax_docs
    if cfg.language.toolchain_config:
        # the [language] toolchain config, passed through to the eval templates'
        # build face (taskcard puts it in the card's submit.toml [env])
        merged["RKT_RUN_TOOLCHAIN_CONFIG"] = cfg.language.toolchain_config
    return merged


def _spawn_default(cmd: list, env: dict) -> int:
    p = subprocess.run(cmd, env=env)
    return p.returncode


def drive(cfg: RunConfig, *, spawn=None, environ=None, new_run_fn=None) -> int:
    """Every kernel task of ``cfg``, in parallel (one slot per kernel task, or
    ``[run] parallel`` slots). Each child is a normal ``rsi-kernel-tune run`` of
    its own run directory; the parent waits, then reports every kernel's final
    status and exits 0 only when every kernel's confirmation met its target."""
    spawn = spawn or _spawn_default
    plans = plan(cfg, environ=environ)
    slots = cfg.run.slots(len(plans))
    _say(f"config run: {len(plans)} kernel task(s), {slots} concurrent slot(s) "
         f"({', '.join(f'{p.task_dir.name} on GPU {p.gpu}' for p in plans)})")
    machine = load_machine(environ=environ)
    import shutil
    import tempfile

    runs: list[tuple[KernelPlan, Run]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="rsi-run-snapshots-"))
    try:
        for p in plans:
            task = _snapshot_task(p.task_dir, cfg, tmp_root)
            runs.append((p, new_run(task, machine=machine) if new_run_fn is None
                         else new_run_fn(task, machine=machine)))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    def go(item: tuple[KernelPlan, Run]) -> int:
        p, run = item
        env = _child_env(cfg, gpu=p.gpu, environ=environ)
        cmd = [sys.executable, "-m", "rsi_kernel_tune", "run", str(run.dir)]
        return spawn(cmd, env)

    with ThreadPoolExecutor(max_workers=slots) as pool:
        codes = list(pool.map(go, runs))

    failed = []
    for (p, run), rc in zip(runs, codes):
        try:
            rec = open_run(run.dir).record()
        except Exception:  # noqa: BLE001 -- the report degrades, the codes decide
            rec = {}
        status = rec.get("status", "unknown")
        line = f"{run.name}: {status} (exit {rc})"
        if rec.get("target_met"):
            line += " -- the final confirmation meets the target"
        _say(line)
        if rc != 0 or status != "done":
            failed.append(run.name)
    if failed:
        _say(f"{len(failed)} kernel run(s) did not finish at their target: "
             + ", ".join(failed))
        return 2
    return 0
