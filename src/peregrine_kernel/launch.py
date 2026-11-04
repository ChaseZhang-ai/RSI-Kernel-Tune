"""``peregrine init | new | run | status | harvest``: the launch flow (docs/design/v1-launch.md).

    peregrine init <op>                    tasks/<op>/ from tasks/_template/
    peregrine new tasks/<op> [--dry-run]   a new run of the task: .artifacts/<op>_<id>/
    peregrine run .artifacts/<op>_<id>     run it; on a run that has started, continue it
    peregrine status [.artifacts/<op>_<id>] where a run is (every run when omitted)
    peregrine harvest .artifacts/<op>_<id> the end-of-run knowledge-base write-up of a stopped run

`peregrine run` exits 0 when the run ends at its target, 2 on any other stop (the loop's stop
reason is in run.json and report.md).

A run, in order (each step is skipped on resume once it is done):

  1. the judge   a teacher turn writes check/{definition.json, judge.py, groups.json} from
                 the task (peregrine_kernel.judge); checked statically, retried with the findings
  2. the trees   operators/<op>/ (the card) and, with a reference, operators/_<op>ref/,
                 the reference sealed into both (peregrine_kernel.taskcard)
  3. calibration with a reference: the reference against the judge on the GPU, on every
                 input group of every shape; a disagreement goes back to step 1 with the
                 numbers, an agreement freezes the tolerance. Without one the tolerance is
                 the dtype floor, and the /goal session that writes the reference is held
                 to it.
  4. the loop    loop.toml from the task, the student's workspace, the agent commands --
                 then the two-layer loop (peregrine_kernel.loop.cli), which resumes itself from
                 its own records
  5. the report  report.md: how the run ended, the best kernel, new knowledge-base entries
                 waiting for review

Nothing on the command line changes the task.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from peregrine_kernel import judge as judge_mod
from peregrine_kernel import runs as runs_mod
from peregrine_kernel import taskcard
from peregrine_kernel.agents import agent_command
from peregrine_kernel.localenv import MachineError, load_machine
from peregrine_kernel.paths import OPERATORS_ENV, repo_root
from peregrine_kernel.runs import RunError, new_run, open_run
from peregrine_kernel.task import (MODE_BLACKBOX, MODE_GENERATE, MODE_SOURCE,
                              TaskError, load_task)

TASKS_DIR = "tasks"
TEMPLATE_TASK = "_template"
CALIBRATE_MODE = "calibrate"
KB_DIRS = ("hw_kb", "docs/kb-provenance")
BEST_DIR = "best"


def _say(msg: str) -> None:
    print(f"[peregrine] {msg}", flush=True)


# -- init -----------------------------------------------------------------------------------

FROM_KERNELS_TASK_TOML = """# {op} -- a task frozen from the published kernel kernels/{kernel_dir} (`peregrine init --from-kernels`).
# The published kernel is the reference (sealed into the card's jobs by `peregrine run`, its
# source withheld); the judge is written from description.md, then calibrated against the
# reference on the GPU.

[task]
language = "{language}"
arch     = "{arch}"

[reference]
run      = "reference/run.py:run"
blackbox = true

# Everything else takes its value from global_config/; write only what changes.
# [protocol]
# sm_clock_mhz = 1500                 # optional: pin + verify the measurement clock
# [target]
# ratio  = 1.05                       # x the published kernel's speed (true overtaking)
# [budget]
# hours    = 4
# max_meta = 3
"""


def cmd_init(argv) -> int:
    """tasks/<op>/: a copy of tasks/_template/ to fill in; with --from-kernels, a task
    frozen from a published kernel (kernels/); with any other option, an operator tree
    made directly (peregrine_kernel.scaffold)."""
    if any(a.startswith("--from-kernels") for a in argv):
        return new_kernels_task(argv)
    if any(a.startswith("-") and a not in ("-h", "--help") for a in argv):
        from peregrine_kernel.scaffold import main as scaffold_main
        return scaffold_main(argv)
    p = argparse.ArgumentParser(prog="peregrine init",
                                description="Create tasks/<op>/ from tasks/_template/.")
    p.add_argument("op")
    args = p.parse_args(argv)
    from peregrine_kernel.scaffold import task_name_problems
    if problems := task_name_problems(args.op):
        for problem in problems:
            sys.stderr.write(f"[peregrine] {problem}\n")
        return 2
    d = repo_root() / TASKS_DIR / args.op
    if d.exists():
        sys.stderr.write(f"[peregrine] {d} already exists\n")
        return 2
    shutil.copytree(repo_root() / TASKS_DIR / TEMPLATE_TASK, d)
    _say(f"wrote {d}/: fill in description.md and shapes.json, adjust task.toml, add "
         f"reference/ if you have one, then `peregrine new {TASKS_DIR}/{args.op}`")
    return 0


def _find_kernel_tree(op: str, arch: str | None) -> Path:
    """kernels/<family>/<op>/<arch>/{kernel.py,kernel.cu} for the op (one family)."""
    root = repo_root() / "kernels"
    trees = sorted(p for pat in (f"*/{op}/*/kernel.py", f"*/{op}/*/kernel.cu")
                   for p in root.glob(pat) if p.is_file())
    if arch:
        trees = [t for t in trees if t.parent.name == arch]
    if len(trees) != 1:
        have = ", ".join(f"{t.parent.parent.name}/{t.parent.name}" for t in trees) or "none"
        hint = ("; pass --arch to disambiguate" if trees else
                "; found no tree at the supported layout kernels/<family>/<op>/<arch>/ "
                "(generational layouts <op>/<arch>/<gen>/ are not scaffoldable yet)")
        raise SystemExit(f"[peregrine] kernels/: {op}: {len(trees)} kernel tree(s) ({have})"
                         f"{hint}")
    return trees[0]


def new_kernels_task(argv) -> int:
    """tasks/<op>/ from a published kernel (``peregrine init <op> --from-kernels``): the
    kernel's own file is the reference (sealed black-box by `peregrine run`), its docstring
    is the frozen contract seeding description.md, and shapes.json starts with one
    campaign shape to fill from that contract. The framework's own task supply: the
    self-evolution has a problem to work on, and beating the published kernel here is
    a true overtaking."""
    import ast
    p = argparse.ArgumentParser(prog="peregrine init --from-kernels",
                                description="Create tasks/<op>/ from a published kernel.")
    p.add_argument("op")
    p.add_argument("--from-kernels", action="store_true",
                   help=argparse.SUPPRESS)      # the dispatch key; accepted and ignored here
    p.add_argument("--arch", default=None, help="the kernel tree's GPU dir (h100 / b200); "
                                                "needed when the op ships several")
    p.add_argument("--shape-name", default="campaign",
                   help="the single shape shapes.json starts with")
    p.add_argument("--axes", required=True,
                   help="JSON object: the campaign point's axes, from the kernel docstring's "
                        "frozen contract (the tool scaffolds; it does not guess the contract)")
    args = p.parse_args(argv)
    from peregrine_kernel.scaffold import task_name_problems
    if problems := task_name_problems(args.op):
        for problem in problems:
            sys.stderr.write(f"[peregrine] {problem}\n")
        return 2
    d = repo_root() / TASKS_DIR / args.op
    if d.exists():
        sys.stderr.write(f"[peregrine] {d} already exists\n")
        return 2
    try:
        axes = json.loads(args.axes)
    except ValueError as e:
        sys.stderr.write(f"[peregrine] --axes does not parse: {e}\n")
        return 2
    if not isinstance(axes, dict) or not axes or \
            not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v > 0
                    for k, v in axes.items()):
        sys.stderr.write("[peregrine] --axes must be a JSON object of axis name -> positive integer\n")
        return 2
    kfile = _find_kernel_tree(args.op, args.arch)
    if kfile.suffix == ".cu":
        sys.stderr.write(f"[peregrine] {kfile}: a CUDA C++ published kernel is not sealed as a "
                         f"source reference yet (binref path is a P2 backlog item)\n")
        return 2
    src = kfile.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    entry = None
    for node in reversed(tree.body):        # the published kernel's entry: its last top def
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            entry = node.name
            break
    if entry is None:
        sys.stderr.write(f"[peregrine] {kfile}: no top-level entry function found\n")
        return 2

    d.mkdir(parents=True)
    # the kernel's true kernels/-relative directory (family segment included):
    # the one provenance path both generated surfaces cite.
    kernel_dir = kfile.parent.relative_to(repo_root() / "kernels").as_posix()
    task_toml = FROM_KERNELS_TASK_TOML.format(
        op=args.op,
        kernel_dir=kernel_dir,
        language="cutedsl",
        arch={"h100": "sm90", "b200": "sm100"}.get(kfile.parent.name, kfile.parent.name))
    (d / "task.toml").write_text(task_toml, encoding="utf-8")
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text(
        src + ("" if src.endswith("\n") else "\n")
        + f"\n\n# `peregrine init --from-kernels`: the published kernel, exposed under the\n"
          f"# task reference's entry name. Sealed into the card's jobs by `peregrine run`\n"
          f"# (peregrine_kernel.sealref) -- the agents run, time and profile it, never read it.\n"
          f"run = {entry}\n",
        encoding="utf-8")
    (d / "description.md").write_text(
        f"# {args.op}\n\n"
        f"Published kernel `kernels/{kernel_dir}/kernel.py` "
        f"(its docstring below is the frozen contract: axes, dtypes, input semantics). The "
        f"reference is that kernel, sealed.\n\n"
        f"## Semantics (from the published kernel)\n\n{doc}\n\n"
        f"## Signature\n\n"
        f"- Fill every input/output: name, shape over the axes in shapes.json, dtype -- from "
        f"the frozen contract above.\n"
        f"- The judge is written from this file alone; calibrate it against the sealed "
        f"reference on the GPU before the loop.\n",
        encoding="utf-8")
    (d / "shapes.json").write_text(
        json.dumps([{ "name": args.shape_name, "axes": axes}], indent=2) + "\n",
        encoding="utf-8")
    _say(f"wrote {d}/ from {kfile.relative_to(repo_root())}: fill shapes.json's axes and "
         f"description.md's signature from the frozen contract, then "
         f"`peregrine new {TASKS_DIR}/{args.op}`")
    return 0


# -- new / run -----------------------------------------------------------------------------


def describe(task, machine) -> str:
    shapes = "; ".join(f"{s.name} {json.dumps(s.axes)} ({len(s.inputs)} input group(s) given"
                       + (f", {s.groups_to_add} to add" if s.groups_to_add else "") + ")"
                       for s in task.shapes)
    lines = [f"task {task.name}: {task.language} on {task.arch}, mode {task.mode}",
             "  shapes: " + shapes]
    if task.reference is not None:
        lines.append(f"  reference: {task.reference.run}"
                     + (f" (source {task.reference.source})" if task.mode == MODE_SOURCE
                        else " (black box)"))
    target = (f"{task.target.ms} ms" if task.target.ms is not None
              else f"{task.target.ratio}x the reference")
    lines += [f"  target: {target}, {task.target.timing} timing",
              f"  agents: teacher {_agent_line(task.teacher)}, student "
              f"{_agent_line(task.student)}",
              f"  budget: {task.budget.hours:g} h, {task.budget.max_meta} teacher round(s)"
              + (f", {task.budget.reference_hours:g} h to write the reference"
                 if task.mode == MODE_GENERATE else ""),
              f"  GPU: {machine.describe()}"]
    lines += [f"  note: {n}" for n in task.notes]
    return "\n".join(lines)


def _agent_line(spec) -> str:
    extra = [x for x in (spec.effort, None if spec.provider == "native" else spec.provider) if x]
    return spec.model + (f" ({', '.join(extra)})" if extra else "")


def _check_providers(task, machine) -> None:
    """A provider's key must be there before a run is created, not hours into it."""
    from peregrine_kernel import provider as provider_mod
    env = {**os.environ, **{k: v for k, v in machine.env.items() if v}}
    for spec in (task.teacher, task.student):
        provider_mod.grant(spec.backend, spec.provider, env)


def cmd_new(argv) -> int:
    """A new run of a task: the task is checked, the plan printed, and the run directory
    created under .artifacts/ (`peregrine run` runs it)."""
    from peregrine_kernel.provider import ProviderError
    p = argparse.ArgumentParser(prog="peregrine new", description="Create a new run of a task.")
    p.add_argument("task", help="the task directory (tasks/<op>)")
    p.add_argument("--dry-run", action="store_true", help="check and print the plan only")
    args = p.parse_args(argv)
    try:
        task = load_task(args.task)
        machine = load_machine()
        _check_providers(task, machine)
    except (TaskError, MachineError, ProviderError) as e:
        sys.stderr.write(f"[peregrine] {e}\n")
        return 2
    print(describe(task, machine), flush=True)
    if args.dry_run:
        return 0
    run = new_run(task, machine=machine)
    _say(f"run directory {run.dir}: `peregrine run {_shown(run.dir)}` runs it")
    return 0


def _shown(path: Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def cmd_run(argv) -> int:
    """Run a run: from the start when it is new, else from where it stopped (every step
    that is done is skipped)."""
    p = argparse.ArgumentParser(prog="peregrine run",
                                description="Run a run, or continue one that has started.")
    p.add_argument("run", help="the run directory (.artifacts/<op>_<id>)")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        machine = load_machine()
    except (RunError, MachineError) as e:
        sys.stderr.write(f"[peregrine] {e}\n")
        return 2
    started = run.record().get("status") != runs_mod.STATUS_CREATED
    return drive(run, machine, resuming=started)


def cmd_status(argv) -> int:
    p = argparse.ArgumentParser(prog="peregrine status")
    p.add_argument("run", nargs="?", help="a run directory; omitted: every run")
    args = p.parse_args(argv)
    if args.run is None:
        for r in runs_mod.list_runs():
            rec = r.record()
            print(f"{r.name}  {rec.get('status')}  {rec.get('updated')}")
        return 0
    try:
        run = open_run(args.run)
    except RunError as e:
        sys.stderr.write(f"[peregrine] {e}\n")
        return 2
    print(status_text(run))
    return 0


def status_text(run) -> str:
    rec = run.record()
    lines = [f"{run.name}: {rec.get('status')} (updated {rec.get('updated')})"]
    for key in ("stage", "loop_rc", "best", "error"):
        if rec.get(key) is not None:
            lines.append(f"  {key}: {rec[key]}")
    status = run.operators_dir / run.task_name / "student" / "status.jsonl"
    if status.is_file():
        rows = [json.loads(x) for x in status.read_text().splitlines() if x.strip()]
        keeps = [r for r in rows if r.get("verdict") == "KEEP"]
        lines.append(f"  iterations: {len(rows)} ({len(keeps)} kept)")
    if run.log.is_file():
        lines.append(f"  log: {run.log}")
    return "\n".join(lines)


def cmd_harvest(argv) -> int:
    p = argparse.ArgumentParser(prog="peregrine harvest")
    p.add_argument("run")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        machine = load_machine()
    except (RunError, MachineError) as e:
        sys.stderr.write(f"[peregrine] {e}\n")
        return 2
    return drive(run, machine, resuming=True, harvest_only=True)


# -- the run itself -------------------------------------------------------------------------


class _Tee:
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8")

    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _entered(run, machine):
    """While the run is driven, this process and everything it starts work on the run's
    trees with local.env's credentials, and its output goes to run.log as well; both are
    put back afterwards."""
    saved_env = {k: os.environ.get(k) for k in [OPERATORS_ENV, *machine.env]}
    saved_out, saved_err = sys.stdout, sys.stderr
    os.environ[OPERATORS_ENV] = str(run.operators_dir)
    for k, v in machine.env.items():
        if v:       # an empty KEY= line (local.env.example) must not blank a login
            os.environ.setdefault(k, v)
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, run.log)
        sys.stderr = _Tee(sys.stderr, run.log)
    try:
        yield
    finally:
        if sys.stdout is not saved_out:
            sys.stdout.flush()
            sys.stdout = saved_out
        if sys.stderr is not saved_err:
            sys.stderr.flush()
            sys.stderr = saved_err
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def deny_paths(run, task) -> list[str]:
    """What no student (and, in blackbox mode, no teacher) may read: the reference wherever
    a copy of it lives, every other run, the task's own extras."""
    out = []
    if task.reference is not None:
        out.append(str(task.reference.run_file.parent))
        if task.reference.source is not None:
            out.append(str(task.reference.source))
        src = Path(run.record().get("task_source", ""))
        if src.is_dir() and (src / "reference").exists():
            out.append(str(src / "reference"))
        out += [str(p) for p in task.reference.deny]
    for other in runs_mod.list_runs(run.dir.parent):
        if other.dir != run.dir:
            out.append(str(other.dir))
    return list(dict.fromkeys(out))


def drive(run, machine, *, resuming: bool = False, harvest_only: bool = False) -> int:
    with _entered(run, machine):
        task = load_task(run.task_dir, name=run.task_name)
        run.update(status=runs_mod.STATUS_RUNNING)
        try:
            rc = _drive(run, task, machine, resuming=resuming, harvest_only=harvest_only)
        except Exception as e:  # noqa: BLE001 -- the record must say why the run stopped
            run.update(status=runs_mod.STATUS_FAILED, error=f"{type(e).__name__}: {e}")
            raise
        stop = loop_stop(run, task) if rc == 0 else None
        met = rc == 0 and confirmed_at_target(run, task)
        # The exit-code contract (__main__.py): DONE only when the FINAL CONFIRMATION
        # meets the target with no correctness alarm on that kernel. The loop's own
        # stop reason ("target") may not vouch for it -- a finish-monitor alarm or a
        # missing best_confirmation.json must keep the run stopped (exit 2), the same
        # way a budget that ran out after the target was met still completes (met
        # holds there too).
        final = (runs_mod.STATUS_DONE if met else runs_mod.STATUS_STOPPED)
        run.update(status=final, loop_rc=rc, stop=stop, target_met=met)
        write_report(run, task, rc)
        return 0 if final == runs_mod.STATUS_DONE else 2


STOP_TARGET = "target"


def loop_stop(run, task) -> str | None:
    """Why the loop ended (its summary's ``stop``: target, meta_budget, infeasible, ...)."""
    from peregrine_kernel.loop.cli import SUMMARY_FILE
    try:
        return json.loads((run.operators_dir / task.name / SUMMARY_FILE).read_text()).get("stop")
    except (OSError, ValueError):
        return None


def confirmed_at_target(run, task) -> bool:
    """Whether the final confirmation of the best kernel meets the task's target, whatever
    stopped the loop (a budget can run out after the target was reached on noisy readings).
    A correctness-monitor alarm on that kernel rules it out."""
    c = confirmation(run, task)
    if c is None or (c.get("correctness_monitor") or {}).get("status") == "alarm":
        return False
    if task.target.ms is not None:
        return float(c["median_ms"]) <= task.target.ms
    return (task.target.ratio is not None and c["speedup"] is not None
            and c["speedup"] >= task.target.ratio)


def _drive(run, task, machine, *, resuming: bool, harvest_only: bool) -> int:
    check_dir = run.dir / judge_mod.CHECK_DIR
    op_dir = run.operators_dir / task.name
    deny = deny_paths(run, task)
    rec = run.record()
    feedback = ""
    for attempt in range(1, judge_mod.ATTEMPTS + 1):
        if not rec.get("judge_written"):
            write_judge(run, task, check_dir, deny, feedback)
            run.update(judge_written=True, stage="judge written")
            rec = run.record()
        check = judge_mod.load_check(check_dir)
        if not op_dir.exists():
            run.update(stage="writing the trees")
            taskcard.write_card(run, task, check, machine)
            if task.reference is not None:
                taskcard.seal_reference(run, task, op_dir)
        else:
            _refresh_machine(run, task, machine)
        if rec.get("tolerance_frozen"):
            break
        if task.reference is None:
            taskcard.set_tolerance(op_dir, {}, "the dtype floor: no reference to calibrate "
                                               "against (a generated one must pass the judge)")
            run.update(tolerance_frozen=True, stage="tolerance frozen")
            rec = run.record()
            break
        run.update(stage=f"calibrating the judge (attempt {attempt})")
        cal = calibrate(task, op_dir)
        if cal.get("match"):
            taskcard.set_tolerance(op_dir, cal["tolerance"],
                                   f"from the reference on every input group (attempt {attempt})")
            run.update(tolerance_frozen=True, stage="tolerance frozen",
                       tolerance=cal["tolerance"])
            rec = run.record()
            break
        feedback = ("The judge and the reference disagree:\n"
                    + "\n".join(f"- {m}" for m in cal.get("problems", [])[:20])
                    + "\nEither the judge computes something else than the reference, or an "
                      "input group is not legal for it. Fix the files.")
        _say(f"calibration: no match (attempt {attempt}/{judge_mod.ATTEMPTS})")
        _discard_trees(run, task)
        run.update(judge_written=False)
        rec = run.record()
    else:
        raise RuntimeError(f"the judge did not agree with the reference after "
                           f"{judge_mod.ATTEMPTS} attempts: {feedback}")
    run.update(stage="probing NCU")
    ncu = probe_ncu(task, op_dir)
    run.update(ncu_available=ncu, stage="the loop")
    if not ncu:
        _say("NCU cannot profile on this machine: the run degrades to timing only (no "
             "per-kernel signatures, no timeline / stall pricing)")
    taskcard.write_loop_toml(run, task, op_dir, deny=deny, ncu=ncu)
    taskcard.clone_workspace(op_dir, run.ws)
    return run_loop(run, task, resuming=resuming, harvest_only=harvest_only)


def write_judge(run, task, check_dir: Path, deny: list[str], feedback: str = "") -> None:
    """Teacher turns until the judge's files pass the static check."""
    from peregrine_kernel.loop.agent_policy import AgentPolicy
    from peregrine_kernel.loop.cli import RealBackend
    check_dir.mkdir(parents=True, exist_ok=True)
    blackbox = task.mode == MODE_BLACKBOX
    policy = AgentPolicy(oracle_mode="blackbox" if blackbox else "source",
                         repo_root=repo_root(), ref_source_paths=tuple(Path(p) for p in deny),
                         ref_source_urls=(), teacher_trees=(), dsl_source_dirs=(),
                         secret_files=(), language=task.loop_language)
    be = RealBackend(task.name, check_dir, None, "peregrine-judge", "peregrine-judge@localhost",
                     teacher_agent_cmd=agent_command(task.teacher, "teacher"),
                     agent_policy=policy)
    be.agent_cmd = be.teacher_agent_cmd
    source = task.reference.source if task.mode == MODE_SOURCE else None
    for attempt in range(1, judge_mod.ATTEMPTS + 1):
        _say(f"the judge: teacher turn {attempt}/{judge_mod.ATTEMPTS}")
        be.run_agent(judge_mod.judge_prompt(task, check_dir, reference_source=source,
                                            feedback=feedback), role="teacher")
        problems = judge_mod.check_problems(check_dir, task)
        if not problems:
            _say("the judge: files pass the static check")
            return
        feedback = "\n".join(f"- {p}" for p in problems)
        _say(f"the judge: {len(problems)} problem(s): {problems[:3]}")
    raise RuntimeError(f"the teacher's judge files still fail the check: {feedback}")


def calibrate(task, op_dir: Path) -> dict:
    """The reference against the judge on the GPU (the card's ``calibrate`` task)."""
    from peregrine_kernel.loop.cli import RealBackend
    be = RealBackend(task.name, op_dir, None, "peregrine-judge", "peregrine-judge@localhost")
    log = be.submit_gpu(taskcard.BRANCH, CALIBRATE_MODE)
    return parse_calibration(log)


def probe_ncu(task, op_dir: Path) -> bool:
    """Can NCU read the GPU's counters on this machine? One short job, every launch (the
    machine local.env names may have changed since the last one)."""
    from peregrine_kernel.loop.cli import BackendError, RealBackend
    be = RealBackend(task.name, op_dir, None, "peregrine-judge", "peregrine-judge@localhost")
    try:
        log = be.submit_gpu(taskcard.BRANCH, taskcard.NCU_PROBE_MODE)
    except BackendError as e:
        _say(f"the NCU probe job failed ({e}); assuming NCU is unavailable")
        return False
    return ncu_probe_ok(log)


def ncu_probe_ok(log: str) -> bool:
    """A metric row for the probe's kernel, and no counter-permission refusal."""
    if "ERR_NVGPUCTRPERM" in log:
        return False
    return any('"gpu__time_active.sum"' in line and line.rstrip().endswith('"')
               and any(ch.isdigit() for ch in line.rsplit('","', 1)[-1])
               for line in log.splitlines())


def parse_calibration(log: str) -> dict:
    tail = log.rsplit("=== JSON ===", 1)
    if len(tail) != 2:
        return {"match": False, "problems": ["the calibrate job printed no result:\n"
                                              + log[-2000:]]}
    try:
        obj, _ = json.JSONDecoder().raw_decode(tail[1].strip())
    except json.JSONDecodeError as e:
        return {"match": False, "problems": [f"the calibrate job's result does not parse: {e}"]}
    return obj.get("calibration") or {"match": False, "problems": ["no calibration block"]}


def _discard_trees(run, task) -> None:
    for name in (task.name, taskcard.ref_tree(task.name)):
        shutil.rmtree(run.operators_dir / name, ignore_errors=True)


def _refresh_machine(run, task, machine) -> None:
    """A resumed run follows local.env as it is now."""
    for name in (task.name, taskcard.ref_tree(task.name)):
        tree = run.operators_dir / name
        if (tree / "submit.toml").exists():
            timeout = taskcard.JOB_TIMEOUT_S if name == task.name else 2700
            taskcard.write_submit(tree, task, machine, timeout=timeout)
            taskcard._commit(tree, "Chore: follow local.env")


def run_loop(run, task, *, resuming: bool, harvest_only: bool) -> int:
    from peregrine_kernel.loop import cli as loop_cli
    argv = [task.name, "--workspace", str(run.ws),
            "--hours", f"{task.budget.hours:g}", "--max-meta", str(task.budget.max_meta),
            "--agent-cmd", agent_command(task.student, "student"),
            "--teacher-agent-cmd", agent_command(task.teacher, "teacher")]
    if task.target.ms is not None:
        argv += ["--target-ms", repr(task.target.ms)]
    else:
        argv += ["--equivalence-ratio", repr(task.target.ratio)]
    if task.protocol.memo_lane:
        # M3 (docs/industrial-lane.md §6.5): the task snapshot pins the lane; the
        # launcher re-read it (load_task above) and passes it to the loop like
        # every other task key. A run never changes lanes.
        argv.append("--memo-lane")
    if resuming:
        argv += ["--allow-preexisting-artifacts", "--allow-dirty-kb"]
    if harvest_only:
        argv.append("--harvest-only")
    _say("the loop: " + " ".join(argv[:5]) + " ...")
    return loop_cli.main(argv)


# -- the report -----------------------------------------------------------------------------


def kb_changes() -> list[str]:
    """Knowledge-base entries this machine's runs certified and nobody has committed yet."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "status", "--porcelain", "--",
                              *KB_DIRS], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


def confirmation(run, task) -> dict | None:
    """best_confirmation.json with "speedup" (x the reference, or None) filled in; None
    before a confirmation exists."""
    op_dir = run.operators_dir / task.name
    try:
        c = json.loads((op_dir / "best_confirmation.json").read_text())
    except (OSError, ValueError):
        return None
    if not c.get("sha") or not c.get("median_ms"):
        return None
    speedup = c.get("median_ratio")
    if speedup is None:
        # no reference rode the jobs (a generated reference): against its archived reading
        try:
            arch = json.loads((op_dir / "arch_baseline.json").read_text())
            key = "ms_device" if c.get("window") == "device" else "ms"
            if arch.get(key):
                speedup = float(arch[key]) / float(c["median_ms"])
        except (OSError, ValueError, TypeError):
            pass
    return {**c, "speedup": speedup}


def best_of(run, task) -> dict | None:
    """The run's confirmed best kernel: {"sha", "window", "median_ms", "speedup" (x the
    reference, or None), "ratios" (each confirmation job's same-job ratio; empty when the
    reference did not ride the jobs), "kernel" (its source)}; None before a confirmation
    exists."""
    c = confirmation(run, task)
    if c is None:
        return None
    kf = taskcard.kernel_file(task)
    src = subprocess.run(["git", "-C", str(run.ws), "show", f"{c['sha']}:{kf}"],
                         capture_output=True, text=True)
    return {"sha": c["sha"], "window": c.get("window"), "median_ms": c["median_ms"],
            "speedup": c["speedup"],
            "ratios": [j["ratio"] for j in c.get("jobs", []) if j.get("ratio") is not None],
            "kernel": src.stdout if src.returncode == 0 else None}


def write_report(run, task, rc: int) -> Path:
    rec = run.record()
    # M3 (docs/industrial-lane.md §6.6): every report carries its lane. The lane
    # also bounds what the speedup line may be read as — a MEMO number is service
    # latency over the task's repeating input stream, never a compute-speedup
    # claim, and STRICT remains the only source of paper capability numbers.
    lane = "memo" if task.protocol.memo_lane else "strict"
    lines = [f"# {run.name}", "", f"- task: {task.name} ({task.mode} mode)",
             f"- lane: {lane}"
             + (" — service latency over the task's repeating input stream; the "
                "speedup below is not a compute-speedup claim and is never comparable "
                "with STRICT numbers" if lane == "memo" else ""),
             f"- status: {rec.get('status')} (stop: {rec.get('stop') or 'unknown'}, "
             f"loop exit {rc})"]
    if rec.get("target_met"):
        # target_met is set only from confirmed_at_target(): a clean final
        # confirmation at the target, whatever stopped the loop.
        lines.append("- the final confirmation meets the target")
    best = best_of(run, task)
    if best is not None:
        lines.append(f"- best: {best['sha'][:12]} -- {best['window']} median {best['median_ms']} ms"
                     + (f", {best['speedup']:.4f}x the reference" if best.get("speedup") else ""))
        if best.get("kernel"):
            kf = run.dir / BEST_DIR / Path(taskcard.kernel_file(task)).name
            kf.parent.mkdir(exist_ok=True)
            kf.write_text(best["kernel"], encoding="utf-8")
            lines.append(f"- kernel: {kf.relative_to(run.dir)}")
    lines.append(f"- workspace: {run.ws}")
    kb = kb_changes()
    if kb:
        lines += ["", "## New knowledge-base entries (review, then commit or discard)", ""]
        lines += [f"- {p}" for p in kb]
        _say(f"{len(kb)} new knowledge-base file(s) wait for review: "
             + ", ".join(kb[:5]) + (" ..." if len(kb) > 5 else ""))
    p = run.dir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _say(f"report: {p}")
    return p


def cmd_export(argv) -> int:
    """The run's best kernel: its source."""
    p = argparse.ArgumentParser(prog="peregrine export")
    p.add_argument("run")
    p.add_argument("--out", default=None, help="write here (default: stdout)")
    args = p.parse_args(argv)
    try:
        run = open_run(args.run)
        task = load_task(run.task_dir, name=run.task_name)
    except (RunError, TaskError) as e:
        sys.stderr.write(f"[peregrine] {e}\n")
        return 2
    best = best_of(run, task)
    if best is None or not best.get("kernel"):
        sys.stderr.write("[peregrine] the run has no confirmed best kernel to export yet\n")
        return 2
    # M3 (docs/industrial-lane.md §6.6): the export's provenance note carries the
    # lane. stdout stays the kernel source, byte-clean — the note rides stderr
    # (and the --out confirmation line). A MEMO export is service latency over
    # its task's repeating input stream: never quote it as a compute speedup.
    lane = "memo" if task.protocol.memo_lane else "strict"
    note = (f"[peregrine] export lane: {lane}"
            + (" — service latency over the task's repeating input stream; not a "
               "compute-speedup claim, never comparable with STRICT numbers"
               if lane == "memo" else "") + "\n")
    sys.stderr.write(note)
    if args.out:
        Path(args.out).write_text(best["kernel"], encoding="utf-8")
        _say(f"wrote {args.out} (the best kernel, {best['sha'][:12]}, lane {lane})")
    else:
        sys.stdout.write(best["kernel"])
    return 0


COMMANDS = {"init": cmd_init, "new": cmd_new, "run": cmd_run, "status": cmd_status,
            "harvest": cmd_harvest, "export": cmd_export}
