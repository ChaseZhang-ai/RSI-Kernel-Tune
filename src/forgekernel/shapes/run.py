"""`forge shapes run`: one ordinary task per cluster, then each one's export.

Per cluster, in order, each step a child process of this interpreter (``forge``'s own
commands, nothing reimplemented)::

    forge init <cluster> <the plan's argv>       tasks/<cluster>/
    (write [target] ms and [budget] hours)       the target the plan derived
    forge new tasks/<cluster>                    .artifacts/<cluster>_<id>/
    forge run .artifacts/<cluster>_<id>          the loop (hours long, GPU-bound)
    forge export <run dir> --out .../solution.json

Nothing here uploads, starts an agent, or touches the loop's internals; ``forge run``
gets exactly the argv ``forge`` expects. **GPU exclusion is ForgeKernel's own** (its
per-GPU lock and ``FK_JOB_LOCK_CMD``, read from ``local.env`` by the job path): wrapping
`forge run` in an outer ``with-lock`` would take one lock for hours and hold it through
the whole run's own acquisitions, so it is deliberately not done here.

``runs.json`` is the resumable state (task §5.2): ``planned -> running -> solved |
failed``. A second `run` skips the clusters that are solved, resumes one that was
interrupted from the run directory it already created (``forge run`` continues a run where
it stopped), and retries a failed one from its failed step.

``--max-hours`` caps this *invocation*'s wall clock: when it runs out, the clusters not
started stay ``planned`` and the command exits 3 -- run it again to continue. The per-task
budget is the loop's own (``[budget] hours``), which ``--cluster-hours`` sets when given.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.shapes import SCHEMA, ShapesError
from forgekernel.shapes import cluster as cluster_mod
from forgekernel.shapes import plan as plan_mod

STATUS_PLANNED = "planned"
STATUS_RUNNING = "running"
STATUS_SOLVED = "solved"
STATUS_FAILED = "failed"
STATUSES = (STATUS_PLANNED, STATUS_RUNNING, STATUS_SOLVED, STATUS_FAILED)

RUNS_FILE = "runs.json"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNFINISHED = 3          # a cluster failed, or the budget ran out: resumable


@dataclass
class Proc:
    """What one child command answered."""
    rc: int
    output: str = ""


@dataclass
class ClusterState:
    name: str
    status: str = STATUS_PLANNED
    task: str = ""
    run: str = ""
    solution: str = ""
    target_ms: float | None = None
    hours: float | None = None
    steps: dict = field(default_factory=dict)
    started: str = ""
    finished: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"status": self.status, "task": self.task, "run": self.run,
                "solution": self.solution, "target_ms": self.target_ms,
                "hours": self.hours, "steps": dict(self.steps),
                "started": self.started, "finished": self.finished,
                "error": self.error, "notes": list(self.notes)}

    @classmethod
    def from_json(cls, name: str, data: dict) -> "ClusterState":
        state = cls(name=str(name))
        for key in ("status", "task", "run", "solution", "started", "finished", "error"):
            if data.get(key) is not None:
                setattr(state, key, str(data[key]))
        state.target_ms = data.get("target_ms")
        state.hours = data.get("hours")
        state.steps = dict(data.get("steps") or {})
        state.notes = list(data.get("notes") or ())
        return state

    @property
    def solved(self) -> bool:
        return self.status == STATUS_SOLVED and bool(self.solution) \
            and Path(self.solution).is_file()


@dataclass
class Runs:
    plan_path: str = ""
    sol_goal: float = plan_mod.DEFAULT_SOL_GOAL
    clusters: dict[str, ClusterState] = field(default_factory=dict)
    updated: str = ""

    def state(self, name: str) -> ClusterState:
        if name not in self.clusters:
            self.clusters[name] = ClusterState(name=name)
        return self.clusters[name]

    def as_json(self) -> dict:
        return {"schema": SCHEMA, "update": self.updated or _now(), "plan": self.plan_path,
                "sol_goal": self.sol_goal,
                "clusters": {k: v.as_json() for k, v in self.clusters.items()}}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated = _now()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.as_json(), indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Runs":
        p = Path(path)
        runs = cls()
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return runs
        runs.plan_path = str(doc.get("plan") or "")
        runs.sol_goal = float(doc.get("sol_goal") or plan_mod.DEFAULT_SOL_GOAL)
        runs.updated = str(doc.get("update") or "")
        for name, data in (doc.get("clusters") or {}).items():
            if isinstance(data, dict):
                runs.clusters[str(name)] = ClusterState.from_json(name, data)
        return runs


# --- the child commands ---------------------------------------------------------------------

def forge_argv(python: str, *args: str) -> list[str]:
    return [python, "-m", "forgekernel", *args]


def default_spawn(argv: list[str], *, cwd: Path, env: dict | None = None) -> Proc:
    """Run one child command, capturing its output (never a shell string)."""
    try:
        proc = subprocess.run(argv, cwd=str(cwd), env={**os.environ, **(env or {})},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=None)
    except OSError as e:
        return Proc(rc=127, output=f"could not start {argv[0]}: {e}")
    return Proc(rc=int(proc.returncode), output=proc.stdout or "")


_RUN_DIR_RE = re.compile(r"run directory (\S+)")


def run_dir_from(output: str, *, root: Path, task_name: str) -> Path | None:
    """The run directory `forge new` announced (else the newest matching one)."""
    hits = _RUN_DIR_RE.findall(output or "")
    for hit in reversed(hits):
        path = Path(hit)
        if not path.is_absolute():
            path = Path(root) / path
        if (path / "run.json").is_file():
            return path
    base = Path(root) / ".artifacts"
    if base.is_dir():
        cands = sorted((p for p in base.glob(f"{task_name}_*") if (p / "run.json").is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    return None


# --- task.toml: the target and the budget -----------------------------------------------------

def set_toml_keys(text: str, table: str, keys: dict,
                  *, drop: tuple[str, ...] = ()) -> str:
    """Set ``keys`` inside ``[table]`` of a task.toml's text, keeping the rest of it.

    The table's other keys and its comment lines survive; a key that is not there yet is
    appended to the table, and a table that is not there is appended to the file.
    ``drop`` removes keys that cannot coexist with the new ones: ``[target] ms`` and
    ``[target] ratio`` are alternatives, and a task.toml that named both would not load.
    """
    wanted = {k: v for k, v in keys.items() if v is not None}
    if not wanted:
        return text
    lines = text.splitlines()
    header = f"[{table}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        body = "\n".join(lines).rstrip()
        block = [header] + [f"{k} = {_toml_value(v)}" for k, v in wanted.items()]
        return (body + "\n\n" if body else "") + "\n".join(block) + "\n"
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("["):
        end += 1
    body_lines = lines[start + 1:end]
    seen: set[str] = set()
    out_body: list[str] = []
    for line in body_lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in wanted:
            out_body.append(f"{key} = {_toml_value(wanted[key])}")
            seen.add(key)
        elif key and key in drop:
            continue                      # an alternative key: the new one wins
        else:
            out_body.append(line)
    out_body += [f"{k} = {_toml_value(v)}" for k, v in wanted.items() if k not in seen]
    return "\n".join(lines[:start + 1] + out_body + lines[end:]) + "\n"


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    return json.dumps(str(value))


def apply_target(task_dir: Path, *, target_ms: float | None,
                 hours: float | None) -> list[str]:
    """Write the cluster's ``[target] ms`` and ``[budget] hours`` into its task.toml."""
    path = Path(task_dir) / "task.toml"
    notes: list[str] = []
    if not path.is_file():
        return [f"{path} is missing: the cluster keeps its configured target"]
    text = path.read_text(encoding="utf-8")
    keys: dict = {}
    if target_ms:
        keys["ms"] = round(float(target_ms), 9)
    if hours:
        keys["hours"] = round(float(hours), 6)
    if not keys:
        return notes
    new = text
    if "ms" in keys:
        new = set_toml_keys(new, "target", {"ms": keys["ms"]}, drop=("ratio",))
        notes.append(f"target: {keys['ms']:.6g} ms (the cluster's SOL goal)")
    if "hours" in keys:
        new = set_toml_keys(new, "budget", {"hours": keys["hours"]})
        notes.append(f"budget: {keys['hours']:g} h")
    if new != text:
        path.write_text(new, encoding="utf-8")
    return notes


# --- the run ------------------------------------------------------------------------------

def cluster_targets(plan: dict, sol_goal: float) -> dict[str, float]:
    """Each cluster's ``[target] ms`` for ``sol_goal`` (see cluster.target_ms).

    Keyed by the plan's cluster names (``<op>-c0``), which is how the ledger looks it up.
    """
    workloads = [cluster_mod.Workload.from_json(w) for w in plan.get("workloads") or []]
    clusters = [cluster_mod.Cluster(name=str(c["name"]),
                                    workloads=list(c.get("workloads") or ()),
                                    metric=str(c.get("metric") or cluster_mod.METRIC_T_SOL))
                for c in plan.get("clusters") or []]
    return cluster_mod.target_ms(clusters, workloads, sol_goal=sol_goal)


def run_plan(plan: dict, *, root: str | Path, runs_path: str | Path | None = None,
             python: str | None = None, spawn=default_spawn, max_hours: float | None = None,
             cluster_hours: float | None = None, sol_goal: float | None = None,
             dry_run: bool = False, clock=time.monotonic, log=None,
             env: dict | None = None) -> tuple[int, Runs]:
    """Drive every cluster the plan names; returns ``(exit code, the ledger)``."""
    root = Path(root).expanduser()
    python = python or sys.executable
    goal = float(plan.get("sol_goal") if sol_goal is None else sol_goal)
    product = Path(plan.get("product_dir") or plan_mod.product_dir(Path(plan["task"])))
    path = Path(runs_path) if runs_path else product / RUNS_FILE
    runs = Runs.load(path) if (not dry_run and path.is_file()) else Runs()
    runs.plan_path = str(path.parent / "plan.json")
    runs.sol_goal = goal
    targets = cluster_targets(plan, goal)
    started = clock()
    unfinished = False
    say = log or (lambda message: None)

    def exhausted() -> bool:
        return max_hours is not None and (clock() - started) > float(max_hours)

    for doc in plan.get("clusters") or []:
        name = str(doc["name"])
        state = runs.state(name)
        task_dir = root / "tasks" / name
        state.task = str(task_dir)
        state.target_ms = doc.get("target_ms") or targets.get(name)
        hours = float(cluster_hours) if cluster_hours else None
        state.hours = hours
        argv_init = forge_argv(python, "init", name, *doc.get("forge_init_argv") or [])
        argv_new = forge_argv(python, "new", f"tasks/{name}")
        steps: list[tuple[str, list[str]]] = []
        if not (task_dir / "task.toml").is_file():
            steps.append(("init", argv_init))
        steps.append(("new", argv_new))
        steps.append(("run", None))
        if dry_run:
            say(f"# {name}: {state.status}"
                + (f" (run {state.run})" if state.run else ""))
            for step, argv in steps:
                if step == "run":
                    argv = forge_argv(python, "run", state.run or f"<the dir `forge new` "
                                                                 f"announces>")
                say("  " + " ".join(argv))
            if state.target_ms:
                say(f"  # [target] ms = {state.target_ms:.9g} in tasks/{name}/task.toml")
            if hours:
                say(f"  # [budget] hours = {hours:g} in tasks/{name}/task.toml")
            say(f"  " + " ".join(forge_argv(
                python, "export", state.run or "<run dir>", "--out",
                str(product / "clusters" / name / "solution.json"))))
            continue
        if state.solved:
            say(f"{name}: solved ({state.solution})")
            continue
        if exhausted():
            state.notes.append("not started: --max-hours was exhausted (run again to "
                               "continue)")
            unfinished = True
            continue
        state.status = STATUS_RUNNING
        state.started = state.started or _now()
        state.error = ""
        runs.save(path)
        resumed = bool(state.run) and (Path(state.run) / "run.json").is_file()
        say(f"{name}: {'resuming' if resumed else 'starting'}"
            + (f" from {state.run}" if resumed else ""))
        ok = True
        for step, argv in steps:
            if step == "run":
                if not state.run:
                    ok = False
                    state.error = "no run directory: `forge new` did not announce one"
                    break
                argv = forge_argv(python, "run", state.run)
            if step == "new" and resumed:
                continue
            rc, output = _step(spawn, argv, root=root, env=env)
            state.steps[step] = rc
            if rc != 0:
                ok = False
                state.error = f"{step} exited {rc}: {_tail(output)}"
                break
            if step == "init":
                for note in apply_target(task_dir, target_ms=state.target_ms, hours=hours):
                    state.notes.append(note)
            if step == "new":
                found = run_dir_from(output, root=root, task_name=name)
                if found is None:
                    ok = False
                    state.error = f"new printed no run directory: {_tail(output)}"
                    break
                state.run = str(found)
            runs.save(path)
        if ok:
            out = product / "clusters" / name / "solution.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            rc, output = _step(spawn, forge_argv(python, "export", state.run, "--out",
                                                 str(out)), root=root, env=env)
            state.steps["export"] = rc
            if rc == 0 and out.is_file():
                state.solution = str(out)
                state.status = STATUS_SOLVED
                state.finished = _now()
                say(f"{name}: solved -- {out}")
            else:
                ok = False
                state.error = f"export exited {rc}: {_tail(output)}"
        if not ok:
            state.status = STATUS_FAILED
            state.finished = _now()
            unfinished = True
            say(f"{name}: failed -- {state.error}")
        runs.save(path)
    if not dry_run:
        runs.save(path)
    return (EXIT_UNFINISHED if unfinished else EXIT_OK), runs


def _step(spawn, argv: list[str], *, root: Path, env: dict | None) -> tuple[int, str]:
    try:
        proc = spawn(argv, cwd=root, env=env)
    except TypeError:                                # a fake that takes only the argv
        proc = spawn(argv)
    rc = getattr(proc, "rc", None)
    if rc is None:
        rc, output = proc if isinstance(proc, tuple) else (int(proc), "")
    else:
        output = getattr(proc, "output", "")
    return int(rc), str(output or "")


def _tail(output: str, limit: int = 400) -> str:
    text = (output or "").strip()
    return text[-limit:] if len(text) > limit else text


def _plan_hours(plan: dict, root: Path, name: str) -> float:
    """The cluster task's configured loop hours (its task.toml, else the global default)."""
    import tomllib

    for path in (root / "tasks" / name / "task.toml", root / "global_config" / "loop.toml"):
        try:
            cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hours = (cfg.get("budget") or {}).get("hours")
        if isinstance(hours, (int, float)) and hours > 0:
            return float(hours)
    return 4.0


def format_runs(runs: Runs, plan: dict | None = None) -> str:
    lines = [f"{'cluster':<24}{'status':<10}{'target ms':>12}{'run':<46}solution"]
    for name, state in runs.clusters.items():
        target = "-" if state.target_ms is None else f"{state.target_ms:.6g}"
        lines.append(f"{name:<24}{state.status:<10}{target:>12}  "
                     f"{(state.run or '-'):<44}{state.solution or '-'}")
        for note in state.notes:
            lines.append(f"    note: {note}")
        if state.error:
            lines.append(f"    error: {state.error}")
    if plan:
        done = sum(1 for s in runs.clusters.values() if s.solved)
        lines.append(f"\n{done}/{len(plan.get('clusters') or [])} cluster(s) solved")
    return "\n".join(lines)


def states_of(runs: Runs) -> dict[str, str]:
    return {name: state.status for name, state in runs.clusters.items()}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


__all__ = ["ClusterState", "Proc", "Runs", "apply_target", "cluster_targets",
           "default_spawn", "forge_argv", "format_runs", "run_dir_from", "run_plan",
           "set_toml_keys", "states_of"]
