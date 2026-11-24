"""The population: N lineages on one problem, generation by generation.

One generation is one ordinary ``forge run`` per active lineage (each its own
task directory and run directory), then this module's bookkeeping:

* every run's confirmed best is archived (``archive.py``) with every
  correctness-passing version;
* the generation's readings become :class:`~forgekernel.evolve.migration.View`
  rows and ``migration.plan`` decides what the next generation starts from;
* the plan, the archive and the ledger are written to disk before the next
  generation begins, so killing the process and running the same command again
  continues where it stopped -- a finished generation is not run again, and an
  interrupted lineage's run is continued by ``forge run``'s own resume.

Directory (``.artifacts/evolve/<problem>-pop_<id>/``, gitignored)::

    population.json   config + per-generation state + stop reason (rewritten atomically)
    ledger.jsonl      append-only event log
    variants/…        archive.py's kernels, parents, results, per-version history
    solutions/…       exported solution JSONs / rebuilt board records for re-inits
    archive.md        the human-readable table

Nothing here starts an agent itself: a lineage is a ``forge run`` subprocess, at
most ``--parallel P`` of them at a time.  The GPU work inside each run is
serialised by ForgeKernel's own per-GPU lock, so parallel lineages mean parallel
agent thinking, not concurrent benchmarking.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel import launch
from forgekernel.evolve import archive as archive_mod
from forgekernel.evolve import lineage as lineage_mod
from forgekernel.evolve import metric as metric_mod
from forgekernel.evolve import migration
from forgekernel.evolve.metric import Metric
from forgekernel.runs import list_runs, open_run, runs_root
from forgekernel.task import Task, load_task

POP_DIR = ".artifacts/evolve"
SCHEMA = 1
STATE_FILE = "population.json"
LEDGER_FILE = "ledger.jsonl"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# -- configuration -----------------------------------------------------------------------


@dataclass
class Config:
    problem: str
    base_task: str
    lineages: int = 2
    generations: int | None = None
    step: float = 1.05
    parallel: int = 2
    langs: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    lessons: str | None = None
    elite: int | None = None
    refuge: int = 1
    migration_from: int = 2
    s_floor: float = metric_mod.SOL_FLOOR_DEFAULT
    max_hours: float | None = None
    max_fails: int = 2
    from_lib: bool = False
    blackbox: bool = False
    efforts: list[str] = field(default_factory=list)
    target_profile: str | None = None
    population: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "problem", "base_task", "lineages", "generations", "step", "parallel",
            "langs", "hypotheses", "lessons", "elite", "refuge", "migration_from",
            "s_floor", "max_hours", "max_fails", "from_lib", "blackbox", "efforts",
            "target_profile", "population")}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# -- environment handed to each `forge run` ---------------------------------------------


def context_env(mapping: dict) -> dict[str, str]:
    """``runcontext.env_for`` for the names run context defines, else the documented
    env-var names.  ``{TARGET_PROFILE_PATH}`` etc. are what the prompts read; a name run
    context does not define (``hypotheses``) keeps its documented ``FK_<NAME>_PATH``."""
    clean = {str(k): str(v) for k, v in mapping.items() if v}
    if not clean:
        return {}
    known: set[str] = set()
    try:
        from forgekernel.runcontext import CONTEXT_FILES
        known = set(CONTEXT_FILES)
    except Exception:  # noqa: BLE001 — absent/partial package = the documented names
        known = set()
    env: dict[str, str] = {}
    if known:
        from forgekernel.runcontext import env_for
        env.update({str(k): str(v) for k, v in env_for(
            {k: v for k, v in clean.items() if k in known}).items()})
    rest = {k: v for k, v in clean.items() if k not in known}
    env.update({f"FK_{name.upper()}_PATH": path for name, path in rest.items()})
    return env


def default_target_profile(repo: Path) -> Path | None:
    """The target profile a lineage's runs carry: the rendered profile when task targethw's
    renderer is installed (the file the prompts describe), else ``target/B200.json``."""
    env = os.environ.get("FK_TARGET_PROFILE_PATH", "").strip()
    if env and Path(env).is_file():
        return Path(env).expanduser().resolve()
    p = Path(repo) / "target" / "B200.json"
    if not p.is_file():
        return None
    try:
        from forgekernel import target_profile as tp  # type: ignore
    except ImportError:
        return p.resolve()
    materialize = getattr(tp, "materialize", None)
    if callable(materialize):
        try:
            return Path(materialize(repo)).resolve()
        except Exception:          # noqa: BLE001 -- the JSON itself is still a profile
            pass
    return p.resolve()


# -- default process plumbing ------------------------------------------------------------


def default_new(task_dir: Path) -> Path | None:
    """``forge new <task_dir>`` in this process; the run directory it created."""
    rc = launch.cmd_new([str(task_dir)])
    if rc != 0:
        return None
    runs = [r for r in list_runs() if r.task_name == Path(task_dir).name]
    if not runs:
        return None
    return max(runs, key=lambda r: r.record().get("created", "")).dir


def default_run(run_dir: Path, env: dict) -> int:
    """``python3 -m forgekernel run <run_dir>`` in a child process (it resumes itself).

    The run directory is checked against the runs root first, the argument list is
    passed with ``shell=False``, and the child imports the harness from the
    repository through PYTHONPATH.
    """
    run_path = Path(run_dir).expanduser().resolve()
    try:
        run_path.relative_to(runs_root().resolve())
    except (ValueError, OSError):
        return 2
    if not run_path.is_dir():
        return 2
    repo = Path(launch.repo_root())
    full = {**os.environ, **env}
    src = str(repo / "src")
    full["PYTHONPATH"] = src + (os.pathsep + full["PYTHONPATH"]
                                if full.get("PYTHONPATH") else "")
    return subprocess.run(["python3", "-m", "forgekernel", "run", str(run_path)],
                          cwd=str(repo), env=full, shell=False).returncode


def initial_state(root: Path, config: Config, base_task: Task,
                  seeds: list[dict] | None, *, log=print) -> dict:
    """The population's starting state (``create`` persists it; ``--dry-run`` does not)."""
    langs = list(config.langs) or [base_task.language]
    lineages = []
    for i in range(max(1, int(config.lineages))):
        seed = seeds[i] if seeds and i < len(seeds) else None
        lineages.append({
            "i": i,
            "language": langs[i % len(langs)],
            "hypothesis": _hypothesis_for(config.hypotheses, i, log),
            "effort": (config.efforts[i % len(config.efforts)]
                       if config.efforts else None),
            "seed": seed,
            "retired": False, "retire_reason": "",
            "fails": 0, "last_variant": "", "last_metric": None,
        })
    return {
        "schema": SCHEMA,
        "id": Path(root).name,
        "problem": config.problem,
        "base_task": str(Path(config.base_task).resolve()),
        "created": now_iso(),
        "updated": now_iso(),
        "started_at": time.time(),
        "config": config.as_dict(),
        "lineages": lineages,
        "generations": [],
        "next_generation": 1,
        "stopped": None,
    }


# -- the population ----------------------------------------------------------------------


class Population:
    def __init__(self, root: Path, state: dict, config: Config, *,
                 base_task: Task, repo: Path | None = None,
                 new_fn=None, run_fn=None, metric_fn=None, best_fn=None,
                 clock=time.time, log=print):
        self.root = Path(root)
        self.state = state
        self.config = config
        self.base_task = base_task
        self.repo = Path(repo) if repo is not None else launch.repo_root()
        self.new_fn = new_fn or default_new
        self.run_fn = run_fn or default_run
        self.metric_fn = metric_fn or metric_mod.metric_of
        self.best_fn = best_fn or launch.best_of
        self.clock = clock
        self.log = log
        self.init_kernel_supported = lineage_mod.warmstart_available()

    # -- persistence ---------------------------------------------------------------------

    def save(self) -> None:
        self.state["updated"] = now_iso()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.root / STATE_FILE)

    def ledger(self, event: str, **payload) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {"ts": now_iso(), "event": event, **payload}
        with open(self.root / LEDGER_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # -- construction --------------------------------------------------------------------

    @classmethod
    def create(cls, root: Path, config: Config, *, base_task: Task, seeds: list[dict] | None,
               repo: Path | None = None, **kw) -> "Population":
        state = initial_state(root, config, base_task, seeds,
                              log=kw.get("log", print))
        pop = cls(Path(root), state, config, base_task=base_task, repo=repo, **kw)
        pop.root.mkdir(parents=True, exist_ok=True)
        pop.save()
        pop.ledger("population_started", problem=config.problem,
                   lineages=state["lineages"], config=config.as_dict())
        return pop

    @classmethod
    def load(cls, root: Path, *, base_task: Task, repo: Path | None = None, **kw
             ) -> "Population":
        state = json.loads((Path(root) / STATE_FILE).read_text(encoding="utf-8"))
        config = Config.from_dict(state.get("config") or {})
        return cls(Path(root), state, config, base_task=base_task, repo=repo, **kw)

    # -- state helpers -------------------------------------------------------------------

    @property
    def lineages(self) -> list[dict]:
        return self.state["lineages"]

    def _generation(self, gen: int) -> dict:
        for g in self.state["generations"]:
            if g["gen"] == gen:
                return g
        g = {"gen": gen, "state": "in_progress", "started": now_iso(), "finished": None,
             "lineages": [{"i": lin["i"], "state": "retired" if lin["retired"] else "pending",
                           "task": None, "run": None, "reference": None,
                           "metric": None, "best": None, "variant": None,
                           "rc": None, "failure": "", "versions": 0,
                           "started": None, "finished": None}
                          for lin in self.lineages],
             "decisions": {}}
        self.state["generations"].append(g)
        self.state["generations"].sort(key=lambda x: x["gen"])
        self.save()
        self.ledger("generation_started", generation=gen)
        return g

    def _previous(self, gen: int) -> dict | None:
        for g in self.state["generations"]:
            if g["gen"] == gen - 1:
                return g
        return None

    def _rec(self, g: dict, i: int) -> dict:
        for rec in g["lineages"]:
            if rec["i"] == i:
                return rec
        raise KeyError(i)

    def _active_lineages(self) -> list[dict]:
        return [lin for lin in self.lineages if not lin["retired"]]

    def _bump_fail(self, i: int) -> None:
        lin = self.lineages[i]
        lin["fails"] = int(lin.get("fails") or 0) + 1

    # -- the run -------------------------------------------------------------------------

    def run(self) -> int:
        stop = self.state.get("stopped")
        if stop and stop.get("reason") != "interrupted":
            self.log(f"[evolve] {self.root.name} already stopped: {stop.get('reason')}")
            return self._exit_rc()
        if stop:
            # an interrupted run is resumable: clear the marker and pick up at the
            # generation that was in flight (its lineages' recorded runs resume).
            self.state["stopped"] = None
            self.save()
            self.ledger("population_resumed",
                        generation=self.state.get("next_generation"))
        try:
            while True:
                if self._time_exceeded():
                    self._stop("max_hours", int(self.state["next_generation"]) - 1)
                    break
                gen = int(self.state["next_generation"])
                if self.config.generations and gen > int(self.config.generations):
                    self._stop("generations", gen - 1)
                    break
                g = self._generation(gen)
                self._ensure_tasks(g, gen)
                self._ensure_runs(g)
                self._execute(g, gen)
                plan = self._decide(g, gen)
                g["state"] = "done"
                g["finished"] = now_iso()
                self.state["next_generation"] = gen + 1
                self.save()
                self.ledger("generation_done", generation=gen,
                            next=self.state["next_generation"])
                self.write_archive()
                views = self._views(g, gen)
                if plan.all_retired(views) or not self._active_lineages():
                    self._stop("all_retired", gen)
                    break
                if plan.no_progress():
                    self._stop("stalled", gen)
                    break
        except KeyboardInterrupt:
            self.log("[evolve] interrupted; state is on disk — rerun the same command "
                     "to continue")
            self._stop("interrupted", int(self.state["next_generation"]), interrupted=True)
            return 130
        return self._exit_rc()

    def _time_exceeded(self) -> bool:
        if not self.config.max_hours:
            return False
        started = float(self.state.get("started_at") or time.time())
        return (self.clock() - started) > float(self.config.max_hours) * 3600.0

    def _stop(self, reason: str, generation: int, *, interrupted: bool = False) -> None:
        self.state["stopped"] = {"reason": reason, "generation": generation,
                                "at": now_iso()}
        self.save()
        self.ledger("stopped", reason=reason, generation=generation,
                    interrupted=interrupted)
        self.write_archive()
        self.log(f"[evolve] stopped: {reason} (after generation {generation})")

    def _exit_rc(self) -> int:
        confirmed = any(rec.get("best") for g in self.state["generations"]
                        for rec in g["lineages"])
        gens = len(self.state["generations"])
        # scripts/self_evolve.py: a first generation without a confirmed best exits 1,
        # later stops exit 0.
        return 1 if (not confirmed and gens <= 1) else 0

    # -- per-generation steps ------------------------------------------------------------

    def _ensure_tasks(self, g: dict, gen: int) -> None:
        for lin in self.lineages:
            if lin["retired"]:
                continue
            rec = self._rec(g, lin["i"])
            if rec["state"] in ("done", "failed", "retired"):
                continue
            if rec.get("task") and Path(rec["task"]).exists():
                if not rec.get("run"):
                    rec["state"] = "task"
                self.save()
                continue
            dest = self.repo / "tasks" / lineage_mod.task_name(
                self.base_task.name, lin["i"], gen, self.config.lineages)
            if dest.exists() and Path(dest).resolve() == Path(self.base_task.dir).resolve():
                # One lineage's first generation IS the base task (the old chain's
                # naming); it is run, not rewritten.
                rec["task"] = str(dest)
                rec["reference"] = {"kind": "base", "variant": "", "seed": "",
                                    "language": self.base_task.language,
                                    "kernel": None, "run": None}
                rec["state"] = "task"
                self.save()
                self.ledger("lineage_task", generation=gen, lineage=lin["i"],
                            task=str(dest), reference=rec["reference"])
                continue
            start = self._start_for(g, lin, gen)
            req = lineage_mod.Request(
                base=self.base_task, dest=dest, language=lin["language"],
                generation=gen, lineages=max(1, int(self.config.lineages)),
                step=float(self.config.step),
                reference=start if isinstance(start, lineage_mod.Reference) else None,
                blackbox=bool(self.config.blackbox),
                init_kernel_supported=self.init_kernel_supported,
                effort=lin.get("effort"), workdir=self.root / "solutions",
                add_exclude=True)
            try:
                if start == "retry":
                    prev = self._previous(gen)
                    prev_task = self._rec(prev, lin["i"]).get("task") if prev else None
                    if not prev_task or not Path(prev_task).exists():
                        raise lineage_mod.LineageError(
                            "no previous task directory to retry from")
                    record = lineage_mod.copy_task(req, Path(prev_task))
                else:
                    record = lineage_mod.make_task(req)
            except Exception as e:  # noqa: BLE001 — one lineage never kills the population
                rec["state"] = "failed"
                rec["failure"] = f"{e.__class__.__name__}: {e}"
                self._bump_fail(lin["i"])
                self.save()
                self.ledger("lineage_failed", generation=gen, lineage=lin["i"],
                            stage="task", error=rec["failure"])
                self.log(f"[evolve] l{lin['i']} g{gen}: task not written: {e}")
                continue
            rec["task"] = str(dest)
            rec["reference"] = record.get("reference")
            rec["state"] = "task"
            self.save()
            self.ledger("lineage_task", generation=gen, lineage=lin["i"],
                        task=str(dest), reference=rec["reference"],
                        handoff=record.get("handoff", ""))
            if record.get("handoff"):
                self.log(f"[evolve] l{lin['i']} g{gen}: handoff {record['handoff']}")

    def _start_for(self, g: dict, lin: dict, gen: int):
        """The reference descriptor for this lineage's generation, ``"retry"``, or None
        (copy the base task)."""
        if gen == 1:
            seed = lin.get("seed")
            if seed and Path(str(seed.get("path") or "")).is_file():
                return lineage_mod.Reference(
                    kind="seed", kernel=Path(str(seed["path"])),
                    language=str(seed.get("language") or lin["language"]),
                    seed_id=str(seed.get("id") or ""))
            return None
        prev = self._previous(gen)
        if prev is None:
            return None
        decision = (prev.get("decisions") or {}).get(str(lin["i"]))
        if not isinstance(decision, dict):
            return None
        action = decision.get("action")
        if action == "migrate":
            return self._variant_reference(prev, int(decision.get("donor")), "foreign",
                                           lin)
        if action == "retry":
            return "retry"
        return self._variant_reference(prev, lin["i"], "self", lin)

    def _variant_reference(self, g: dict, i: int, kind: str, lin: dict):
        rec = self._rec(g, i)
        best = rec.get("best") or {}
        kernel = best.get("kernel_path")
        if not kernel or not Path(kernel).is_file():
            raise lineage_mod.LineageError(
                f"lineage {i} of generation {g['gen']} has no archived kernel to "
                f"hand over")
        if self.base_task.leaderboard is None and Path(kernel).suffix != ".py":
            # Documented limitation (task evolve §5.3): a .cu donor cannot become an
            # ordinary task's source reference.  Retry with the same reference; the
            # lineage keeps evolving inside the next run (and the archive keeps the
            # .cu version for campaign).
            self.log(f"[evolve] l{lin['i']}: its best is a {Path(kernel).suffix} kernel; "
                     f"an ordinary task cannot take it as a source reference — the next "
                     f"generation retries from the same reference")
            return "retry"
        return lineage_mod.Reference(kind=kind, kernel=Path(kernel),
                                     run=Path(rec["run"]) if rec.get("run") else None,
                                     language=str(rec.get("language")
                                                  or self.lineages[i]["language"]),
                                     variant=str(rec.get("variant") or ""))

    def _ensure_runs(self, g: dict) -> None:
        for rec in g["lineages"]:
            if rec["state"] != "task":
                continue
            try:
                run_dir = self.new_fn(Path(rec["task"]))
            except Exception as e:  # noqa: BLE001
                run_dir = None
                rec["failure"] = f"{e.__class__.__name__}: {e}"
            if not run_dir:
                rec["state"] = "failed"
                rec["failure"] = rec.get("failure") or "forge new produced no run"
                self._bump_fail(rec["i"])
                self.save()
                self.ledger("lineage_failed", generation=g["gen"], lineage=rec["i"],
                            stage="new", error=rec["failure"])
                continue
            rec["run"] = str(run_dir)
            rec["state"] = "new"
            self.save()
            self.ledger("lineage_run", generation=g["gen"], lineage=rec["i"],
                        run=rec["run"])

    def _execute(self, g: dict, gen: int) -> None:
        pending = [rec for rec in g["lineages"]
                   if rec["state"] in ("new", "run", "running")]
        if not pending:
            return
        for rec in pending:
            rec["state"] = "running"
            rec["started"] = rec.get("started") or now_iso()
        self.save()
        envs = {rec["i"]: self._run_env(rec) for rec in pending}
        parallel = max(1, int(self.config.parallel))
        errors: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(self._call_run, Path(rec["run"]), envs[rec["i"]]): rec
                    for rec in pending}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    rec["rc"] = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec["rc"] = None
                    errors[rec["i"]] = f"{e.__class__.__name__}: {e}"
        self.save()
        for rec in pending:
            if rec["i"] in errors:
                rec["failure"] = errors[rec["i"]]
                self.ledger("lineage_failed", generation=gen, lineage=rec["i"],
                            stage="run", error=rec["failure"])
            self._evaluate(g, rec, gen)

    def _call_run(self, run_dir: Path, env: dict) -> int:
        return int(self.run_fn(run_dir, env))

    def _run_env(self, rec: dict) -> dict:
        mapping = {}
        if self.config.target_profile:
            mapping["target"] = self.config.target_profile
        lin = self.lineages[rec["i"]]
        if lin.get("hypothesis"):
            mapping["hypotheses"] = lin["hypothesis"]
        if self.config.lessons and Path(self.config.lessons).is_file():
            mapping["lessons"] = self.config.lessons
        return context_env(mapping)

    def _evaluate(self, g: dict, rec: dict, gen: int) -> None:
        lin = self.lineages[rec["i"]]
        try:
            task = load_task(rec["task"])
            run = open_run(rec["run"])
            best = self.best_fn(run, task)
            metric = (self.metric_fn(run, task, best=best, s_floor=self.config.s_floor)
                      if best is not None else None)
        except Exception as e:  # noqa: BLE001
            rec["state"] = "done"
            rec["failure"] = f"{e.__class__.__name__}: {e}"
            rec["metric"] = None
            rec["best"] = None
            lin["fails"] = int(lin.get("fails") or 0) + 1
            self.save()
            self.ledger("lineage_failed", generation=gen, lineage=rec["i"],
                        stage="measure", error=rec["failure"])
            self.log(f"[evolve] l{rec['i']} g{gen}: no reading ({e})")
            return
        rec["metric"] = metric.as_dict() if metric is not None else None
        if best is None or not best.get("kernel"):
            rec["state"] = "done"
            rec["best"] = None
            lin["fails"] = int(lin.get("fails") or 0) + 1
            lin["last_metric"] = None
            self.save()
            self.ledger("lineage_result", generation=gen, lineage=rec["i"],
                        metric=None, rc=rec.get("rc"), best=None)
            self.log(f"[evolve] l{rec['i']} g{gen}: no confirmed best "
                     f"(loop exit {rec.get('rc')})")
            return
        variant = lineage_mod.variant_name(rec["i"], gen)
        try:
            versions = archive_mod.collect_versions(run, task)
        except Exception as e:  # noqa: BLE001
            versions = []
            self.log(f"[evolve] l{rec['i']} g{gen}: version history unavailable ({e})")
        suffix = Path(metric_mod.board_kernel_file(task)).suffix or ".py"
        result = {
            "lineage": rec["i"], "generation": gen,
            "language": lin["language"],
            "reference": (rec.get("reference") or {}).get("kind", "base"),
            "reference_variant": (rec.get("reference") or {}).get("variant", ""),
            "metric": metric.as_dict() if metric is not None else None,
            "floor": metric.floor if metric is not None else None,
            "run": rec["run"], "sha": best.get("sha"),
            "task": rec["task"], "rc": rec.get("rc"),
            "confidence": list(metric.confidence) if metric else [],
            "board_pending": metric.board_pending if metric else False,
            "kernel_file": metric_mod.board_kernel_file(task),
        }
        try:
            d = archive_mod.write_variant(
                self.root, variant=variant, kernel_text=best["kernel"],
                kernel_suffix=suffix, parent=self._parent_for(rec, gen),
                result=result, versions=versions)
        except Exception as e:  # noqa: BLE001
            rec["state"] = "done"
            rec["failure"] = f"archive: {e.__class__.__name__}: {e}"
            rec["best"] = {"sha": best.get("sha"), "kernel_path": None, "variant": variant}
            self._bump_fail(rec["i"])
            self.save()
            self.ledger("lineage_failed", generation=gen, lineage=rec["i"],
                        stage="archive", error=rec["failure"])
            return
        rec["state"] = "done"
        rec["variant"] = variant
        rec["versions"] = len(versions)
        rec["best"] = {"sha": best.get("sha"),
                       "ms": best.get("median_ms"),
                       "kernel_path": str(d / f"kernel{suffix}"),
                       "variant": variant,
                       "history": str(d / "versions.json")}
        rec["finished"] = now_iso()
        lin["fails"] = 0
        lin["last_variant"] = variant
        lin["last_metric"] = rec["metric"]
        self.save()
        self.ledger("lineage_result", generation=gen, lineage=rec["i"],
                    metric=rec["metric"], rc=rec.get("rc"), variant=variant,
                    sha=best.get("sha"), versions=len(versions))
        self.log(f"[evolve] l{rec['i']} g{gen}: {variant} {best.get('sha', '')[:12]} "
                 f"{_metric_text(rec['metric'])} ({len(versions)} passing version(s))")

    def _parent_for(self, rec: dict, gen: int) -> str:
        if gen == 1:
            return "init"
        ref = rec.get("reference") or {}
        if ref.get("kind") == "foreign" and ref.get("variant"):
            return str(ref["variant"])
        prev = self._previous(gen)
        if prev is not None:
            prev_rec = self._rec(prev, rec["i"])
            if prev_rec.get("variant"):
                return str(prev_rec["variant"])
        return "init"

    # -- decisions -----------------------------------------------------------------------

    def _views(self, g: dict, gen: int) -> list[migration.View]:
        prev = self._previous(gen)
        out = []
        for lin in self.lineages:
            rec = self._rec(g, lin["i"])
            metric = Metric.from_dict(rec.get("metric"))
            prev_metric = None
            if prev is not None:
                prev_metric = Metric.from_dict(self._rec(prev, lin["i"]).get("metric"))
            ref = rec.get("reference") or {}
            kind = str(ref.get("kind") or "base")
            out.append(migration.View(
                index=lin["i"], language=lin["language"], metric=metric,
                prev_metric=prev_metric,
                reference_kind="self" if kind in ("self", "retry") else kind,
                adopted_global=(kind == "foreign"),
                fails=int(lin.get("fails") or 0),
                retired=bool(lin["retired"]), variant=str(rec.get("variant") or "")))
        return out

    def _decide(self, g: dict, gen: int) -> migration.Plan:
        views = self._views(g, gen)
        legacy = int(self.config.lineages) == 1
        plan = migration.plan(
            views, generation=gen, elite_k=self.config.elite,
            refuge=int(self.config.refuge),
            migration_from=int(self.config.migration_from),
            max_fails=int(self.config.max_fails), legacy=legacy)
        g["decisions"] = {str(d.index): d.as_dict() for d in plan.decisions}
        for d in plan.decisions:
            if d.action == migration.RETIRE:
                lin = self.lineages[d.index]
                lin["retired"] = True
                lin["retire_reason"] = d.reason
                self.ledger("retirement", generation=gen, lineage=d.index,
                            reason=d.reason)
            elif d.action == migration.MIGRATE:
                self.ledger("migration", generation=gen, lineage=d.index,
                            donor=d.donor, reason=d.reason)
        self.save()
        return plan

    # -- archive -------------------------------------------------------------------------

    def write_archive(self) -> Path:
        """Rebuild ``archive.md`` from the state on disk (mechanical; no prose)."""
        rows = []
        for g in self.state.get("generations") or []:
            for rec in g["lineages"]:
                if not rec.get("variant"):
                    continue
                vdir = archive_mod.variant_dir(self.root, rec["variant"])
                kernels = sorted(vdir.glob("kernel.*"))
                kernel_text = kernels[0].read_text(encoding="utf-8", errors="replace") \
                    if kernels else ""
                versions = []
                try:
                    versions = json.loads((vdir / "versions.json").read_text())
                except (OSError, ValueError):
                    pass
                metric = rec.get("metric") or {}
                rows.append({
                    "lineage": rec["i"], "generation": g["gen"],
                    "language": str(metric.get("language") or self.lineages[rec["i"]]["language"]),
                    "reference": (rec.get("reference") or {}).get("kind", "base"),
                    "metric": metric, "floor": metric.get("floor"),
                    "sha": (rec.get("best") or {}).get("sha"),
                    "run": rec.get("run"), "variant": rec["variant"],
                    "board_pending": bool(metric.get("confidence")
                                          and "dev-only" in metric["confidence"]),
                    "kernel_text": kernel_text, "versions": versions})
        rows.sort(key=lambda r: (r["generation"], r["lineage"]))
        text = archive_mod.archive_markdown(problem=self.config.problem,
                                           population_id=str(self.state.get("id")),
                                           rows=rows,
                                           stopped=self.state.get("stopped"))
        return archive_mod.write_archive(self.root, text)

    # -- read-only views for status ------------------------------------------------------

    def summary(self) -> str:
        cfg = self.config
        lines = [f"population {self.state.get('id')}: {cfg.problem} — "
                 f"{'stopped' if self.state.get('stopped') else 'active'}",
                 f"  base task: {self.state.get('base_task')}",
                 f"  lineages: {cfg.lineages} (languages {cfg.langs or [self.base_task.language]}), "
                 f"parallel {cfg.parallel}",
                 f"  generations: {len(self.state.get('generations') or [])} done, "
                 f"next {self.state.get('next_generation')}"]
        if self.state.get("stopped"):
            s = self.state["stopped"]
            lines.append(f"  stopped: {s.get('reason')} after generation {s.get('generation')}")
        for g in self.state.get("generations") or []:
            lines.append(f"  generation {g['gen']} ({g['state']}):")
            for rec in g["lineages"]:
                m = _metric_text(rec.get("metric"))
                sha = ((rec.get("best") or {}).get("sha") or "-")[:12]
                ref = (rec.get("reference") or {}).get("kind", "-")
                state = rec["state"]
                extra = f" {m} {sha}" if rec.get("metric") else ""
                fail = f" — {rec['failure']}" if rec.get("failure") else ""
                lines.append(f"    l{rec['i']}: {state} ({ref}){extra}{fail}")
            for i, d in sorted((g.get("decisions") or {}).items()):
                if d.get("action") in (migration.MIGRATE,):
                    lines.append(f"    decision l{i}: migrate from l{d.get('donor')}")
        for lin in self.lineages:
            if lin["retired"]:
                lines.append(f"  l{lin['i']} retired: {lin['retire_reason']}")
        return "\n".join(lines)


def _metric_text(metric: dict | None) -> str:
    if not metric:
        return "-"
    kind, value = metric.get("kind"), metric.get("value")
    if value is None:
        return "-"
    if kind == "sol":
        return f"sol {float(value):.4f}"
    if kind == "speedup":
        return f"{float(value):.4f}x"
    return f"{float(value):.6g} ms"


def _hypothesis_for(sources: list[str], i: int, log=print) -> str | None:
    """``--hypotheses``: a directory's ``lineage-<i>.md`` or a file shared by all."""
    for src in sources or []:
        p = Path(src).expanduser()
        if p.is_dir():
            cand = p / f"lineage-{i}.md"
            if cand.is_file():
                return str(cand.resolve())
            continue
        if p.is_file():
            return str(p.resolve())
        log(f"[evolve] --hypotheses {p}: not a file or directory; ignored")
    return None


# -- roots -------------------------------------------------------------------------------


def populations(repo: Path, problem: str) -> list[Path]:
    base = Path(repo) / POP_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob(f"{problem}-pop_*") if (p / STATE_FILE).is_file())


def resolve_root(repo: Path, problem: str, *, explicit: str | None = None,
                 new: bool = False) -> tuple[Path, bool]:
    """(population root, resuming?).  ``explicit`` --population wins; else the latest
    population of this problem is resumed unless ``new``."""
    if explicit:
        p = Path(explicit).expanduser()
        return p, (p / STATE_FILE).is_file()
    found = populations(repo, problem)
    if found and not new:
        return found[-1], True
    stamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
    return Path(repo) / POP_DIR / f"{problem}-pop_{stamp}-{secrets.token_hex(2)}", False
