"""One problem's pipeline: starting points -> task -> target -> optimization -> versions.

`campaign run` executes the plan one problem at a time (``--parallel`` problems in
flight). Each problem is a fixed sequence of stages, each recorded in
``state/<op>.json`` so a second invocation resumes instead of repeating:

1. **board record** -- the board's ``kernels/<id>`` payload (T_b / T_SOL per workload)
   from ``board_records/`` when it was cached earlier, else fetched once from the board's
   public API and cached. `forge init --sol-board <file>` then freezes it into the card,
   which is what makes the card's eval JSON carry a SOL score and the loop work offline
   afterwards. A record that does not line up (wrong workload count, no ``sol_ms``) is
   *not* used: the card is made with ``--no-sol-board`` and the readings then carry no S.
2. **task** -- ``forge init <op> --leaderboard sol-execbench --dataset … --definition
   <subset>/<problem> --workload <all>`` (``--init-kernel`` when a start is already known),
   then the task is validated with `forgekernel.task.load_task`.
3. **starts** -- every starting point the installed tools offer: the reference library's
   exact hits and same-family neighbours (a neighbour goes through ``forge lib adapt``
   first -- supplement 1), the torch baselines, and the definition's own reference as the
   always-present floor. Each is scored with `benchx` (``--repeat 2``) and sorted by mean
   SOL score.
4. **target** -- ``[target] ms`` = the best scored start's geomean reading / ``--step``
   (default 1.10), written into ``task.toml``; with no scored start the harness default
   stands and a note says so.
5. **optimize** -- ``forge evolve <task> --lineages 3 --generations 1`` when the evolve
   task is installed (one lineage per available language route), else ``forge new`` +
   ``forge run``. The target profile context file rides the child's environment whenever
   it exists (background §4); roofline / lessons join when the plan names them.
6. **versions** -- every KEEP iteration of every run that passed correctness, *plus* the
   starts: the kernel source from the workspace's git (`git show <sha>:<kernel>`) and its
   per-workload readings from ``student/status.jsonl``, scored against the card's own
   T_b / T_SOL. The run's ``best_confirmation.json`` is added too (it may name a sha the
   trajectory only measured through a ``reading_of`` row).
7. **export** -- `campaign.compose` picks and writes ``exports/<op>/submission.py``.
8. **submit** -- only with ``--submit`` (default off): the board task takes it from here.

A run that stops early is not a failure here: 6/7 still use whatever versions exist. A
*stage* that fails records its error and the problem stops at that stage; the next
invocation retries it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.campaign import compose, plan as plan_mod, solscore, sync
from forgekernel.campaign import targets as targets_mod
from forgekernel.campaign import tools as tools_mod

STATE_DIR = "state"
EXPORTS_DIR = "exports"
RECORDS_DIR = "board_records"
STARTS_DIR = "starts"
KEEP_VERDICTS = ("KEEP",)
STAGES = ("board_record", "task", "starts", "target", "optimize", "versions", "export",
          "submit")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[campaign] {msg}", flush=True)


# --- state ----------------------------------------------------------------------------------

@dataclass
class ProblemState:
    op: str
    name: str
    track: str
    kernel_id: int
    task_dir: str = ""
    status: str = "planned"             # planned | running | done | failed
    stages: dict = field(default_factory=dict)
    runs: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    export: dict = field(default_factory=dict)
    error: str = ""
    updated: str = ""

    def as_json(self) -> dict:
        return {"op": self.op, "name": self.name, "track": self.track,
                "kernel_id": self.kernel_id, "task_dir": self.task_dir,
                "status": self.status, "stages": self.stages, "runs": self.runs,
                "candidates": self.candidates, "export": self.export,
                "error": self.error, "updated": self.updated or _now()}


class StateStore:
    def __init__(self, work_dir: str | Path):
        self.dir = Path(work_dir) / STATE_DIR

    def path(self, op: str) -> Path:
        return self.dir / f"{op}.json"

    def load(self, op: str) -> ProblemState | None:
        p = self.path(op)
        if not p.is_file():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return ProblemState(**{k: v for k, v in obj.items() if k in ProblemState.__annotations__})

    def save(self, st: ProblemState) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        st.updated = _now()
        p = self.path(st.op)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.as_json(), indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(p)


# --- one problem ------------------------------------------------------------------------------

class ProblemRunner:
    """All the stages of one problem. Every external call goes through ``tools``."""

    def __init__(self, tools: tools_mod.Tools, entry: dict, *, work_dir: str | Path,
                 state: StateStore, ledger: sync.Ledger, dataset: str,
                 step: float = solscore.DEFAULT_STEP, hours: float | None = None,
                 parallel: int = 1, route: str = "auto", submit: bool = False,
                 offline: bool = False, allow_slow_cold_start: bool = False,
                 force: tuple[str, ...] = (), quiet: bool = False):
        self.tools = tools
        self.entry = entry
        self.work = Path(work_dir)
        self.state = state
        self.ledger = ledger
        self.dataset = dataset or (entry.get("dataset") or "")
        self.step = float(step)
        self.hours = float(hours if hours is not None else entry.get("hours") or 6.0)
        self.parallel = max(1, int(parallel))
        self.route = route
        self.submit = submit
        self.offline = offline
        self.allow_slow = allow_slow_cold_start
        self.force = set(force)
        self.quiet = quiet

    # -- plumbing -----------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if not self.quiet:
            log(f"{self.entry['name']}: {msg}")

    def _record_error(self, st: ProblemState, stage: str, msg: str) -> None:
        st.stages[stage] = "failed"
        st.error = f"{stage}: {msg}"
        st.status = "failed"
        self.state.save(st)
        self.ledger.append({"event": "error", "problem": st.name, "track": st.track,
                            "kernel_id": st.kernel_id, "source": sync.DEV, "note": st.error})
        self._say(f"{stage} failed: {msg}")

    def run(self) -> ProblemState:
        st = self.state.load(self.entry["op"]) or ProblemState(
            op=self.entry["op"], name=self.entry["name"], track=self.entry["track"],
            kernel_id=int(self.entry["kernel_id"]))
        if st.status == "done" and "export" not in self.force:
            self._say("already done (state says so; --force export re-exports)")
            return st
        st.status = "running"
        st.error = ""
        self.state.save(st)
        stages = [("board_record", self.stage_board_record),
                  ("task", self.stage_task),
                  ("starts", self.stage_starts),
                  ("target", self.stage_target),
                  ("optimize", self.stage_optimize),
                  ("versions", self.stage_versions),
                  ("export", self.stage_export),
                  ("submit", self.stage_submit)]
        for name, fn in stages:
            if st.stages.get(name) == "done" and name not in self.force:
                continue
            try:
                fn(st)
            except Exception as e:                              # noqa: BLE001
                self._record_error(st, name, f"{type(e).__name__}: {e}")
                return st
            st.stages.setdefault(name, "done")
            if st.stages[name] != "skipped":
                st.stages[name] = "done"
            self.state.save(st)
        st.status = "done"
        self.state.save(st)
        self._say(f"done: {st.export.get('path') or 'no export'}")
        return st

    # -- 1. the board's T_b / T_SOL ----------------------------------------------------------
    def record_path(self) -> Path:
        return self.work / RECORDS_DIR / f"{int(self.entry['kernel_id'])}.json"

    def stage_board_record(self, st: ProblemState) -> None:
        if self.tools.bench is None and self.offline:
            self._say("offline and no bench: no SOL score will be available")
        rec = self.record_path()
        if rec.is_file():
            self._say(f"board record cached at {rec}")
            return
        if self.offline:
            self._say("offline: the card gets --no-sol-board (readings carry no S)")
            return
        try:
            from forgekernel.leaderboard import fetch_sol_board
            payload = fetch_sol_board(st.name, solscore.TRACK_SUBSET[st.track])
        except Exception as e:                                  # noqa: BLE001
            self._say(f"the board's kernels/{st.kernel_id} was not read ({e}); the card "
                      f"gets --no-sol-board")
            return
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        self._say(f"board record saved to {rec}")

    def usable_record(self) -> Path | None:
        rec = self.record_path()
        if not rec.is_file():
            return None
        try:
            obj = json.loads(rec.read_text(encoding="utf-8"))
        except ValueError:
            return None
        obj = obj.get("data", obj)
        wls = obj.get("workloads")
        if not isinstance(wls, list) or not wls:
            return None
        if not all(isinstance(w, dict) and w.get("baseline_latency_ms") and w.get("sol_ms")
                   for w in wls):
            return None
        return rec

    # -- 2. the task ---------------------------------------------------------------------------
    def task_dir(self) -> Path:
        return Path(self.tools.root) / "tasks" / self.entry["op"]

    def workload_selector(self) -> str:
        sel = self.entry.get("workloads") or "all"
        if sel != "all":
            return str(sel)
        path = (Path(self.dataset) / self.entry["subset"] / self.entry["name"]
                / "workload.jsonl")
        if not path.is_file():
            raise FileNotFoundError(f"{path}: no workload list (is --dataset the tree "
                                    f"download_solexecbench.py wrote?)")
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if n <= 0:
            raise ValueError(f"{path}: no workloads")
        return "0" if n == 1 else f"0-{n - 1}"

    def stage_task(self, st: ProblemState) -> None:
        task_dir = self.task_dir()
        st.task_dir = str(task_dir)
        if task_dir.is_dir():
            self._say(f"task {task_dir} exists; not re-initialised")
            return
        rec = self.usable_record()
        argv = list(self.entry["init_argv"])
        selector = self.workload_selector()
        argv = _set_board_flag(_replace_flag(argv, "--workload", selector), rec)
        if self.route == "cuda":
            # the cuda mode's product is a single .cu; its export needs cuda_to_inline
            argv = _replace_flag(argv, "--kernel-file", "kernel.cu")
            self._say("route: cuda (the .cu export needs cuda_to_inline / seedlib)")
        rc = self.tools.call_init(argv)
        if rc != 0 or not task_dir.is_dir():
            raise RuntimeError(f"forge init exited {rc}; no task at {task_dir}")
        self._validate_task(task_dir)
        self._say(f"task created: {task_dir} (workloads {selector})")

    def _validate_task(self, task_dir: Path) -> None:
        try:
            from forgekernel.task import load_task
        except ImportError:                                    # pragma: no cover
            return
        task = load_task(task_dir)
        self._say(f"task validates: {targets_mod.family_of(task.name)}, language "
                  f"{task.language}")

    # -- 3. starting points --------------------------------------------------------------------
    def stage_starts(self, st: ProblemState) -> None:
        prob = compose.load_problem(self.task_dir())
        cands: list[compose.Candidate] = []
        notes: list[str] = []
        if self.tools.lib_find is None:
            notes.append("no reference library (seedlib not installed): the torch baselines "
                         "and the definition's reference are the starts")

        # the reference library: exact hits, then same-family neighbours through adapt
        if self.tools.lib_find is not None:
            try:
                found = self.tools.lib_find(st.name, 5)
            except Exception as e:                              # noqa: BLE001
                found = []
                notes.append(f"lib find failed ({type(e).__name__}: {e})")
            exact = [e for e in found if st.name in (e.get("problem_keys") or [])]
            family = [e for e in found if e not in exact]
            for e in exact[:3]:
                src = _read_kernel_entry(e)
                if src:
                    cands.append(compose.Candidate(
                        name=f"lib:{e.get('id')}", source=src, language=_lang(e),
                        meta={"source": e.get("source") or e.get("path") or "",
                              "license": e.get("license") or "unknown"},
                        origin="starter",
                        notes=[f"reference library entry {e.get('id')}"]))
            if not exact and family and self.tools.lib_adapt is not None:
                for e in family[:2]:
                    out_dir = self.work / STARTS_DIR / st.op / f"adapt-{e.get('id')}"
                    try:
                        made = self.tools.lib_adapt(st.name, str(e.get("id")), out_dir,
                                                    dataset=self.dataset)
                    except Exception as ex:                     # noqa: BLE001
                        notes.append(f"adapt {e.get('id')} failed ({type(ex).__name__})")
                        continue
                    if made is not None:
                        cands.append(compose.Candidate(
                            name=f"adapt:{e.get('id')}", source=Path(made).read_text(encoding="utf-8"),
                            meta={"derived_from": f"lib:{e.get('id')}",
                                  "license": e.get("license") or "unknown"},
                            origin="adapt",
                            notes=[f"adapted from reference library entry {e.get('id')}"]))

        # torch baselines
        if self.tools.lib_baselines is not None:
            out = self.work / STARTS_DIR / st.op / "baselines"
            try:
                made = self.tools.lib_baselines(st.name, out, dataset=self.dataset)
            except Exception as e:                              # noqa: BLE001
                made = None
                notes.append(f"lib baselines failed ({type(e).__name__}: {e})")
            if made is not None:
                for f in sorted(Path(made).glob("*.py")):
                    cands.append(compose.Candidate(
                        name=f"baseline:{f.stem}", source=f.read_text(encoding="utf-8"),
                        meta={"source": f"forge lib baselines ({f.name})",
                              "license": "Apache-2.0"},
                        origin="baseline"))

        # the definition's own reference: always exportable, the floor
        from_ref = compose.reference_only_submission(prob)
        cands.append(compose.Candidate(
            name="reference", source=from_ref, language="python",
            meta={"source": "the board definition's reference", "license": "n/a (board "
                  "definition)"}, origin="starter",
            notes=["the eager reference wrapped as run(): a correct floor, never fast"]))

        # score them
        scored = self._bench_all(st, cands)
        st.candidates = [_cand_json(c) for c in scored]
        self.state.save(st)
        for n in notes:
            self._say(n)

    def _bench_all(self, st: ProblemState, cands: list[compose.Candidate],
                   repeat: int = 2) -> list[compose.Candidate]:
        if self.tools.bench is None:
            for c in cands:
                c.notes.append("not scored: benchx is not installed")
            return cands
        task_dir = self.task_dir()
        for c in cands:
            path = self.work / STARTS_DIR / st.op / f"{_slug(c.name)}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(c.source, encoding="utf-8")
            try:
                reading = self.tools.bench(task_dir, path, repeat=repeat, mode="dev")
            except Exception as e:                              # noqa: BLE001
                c.correct = False
                c.notes.append(f"bench failed: {type(e).__name__}: {e}")
                continue
            c.mean_s = reading.mean_s
            c.geomean_ms = reading.geomean_ms
            c.cold_start_s = reading.cold_start_s
            c.correct = bool(reading.correct)
            c.kernel_file = str(path)
            if not reading.correct and reading.error:
                c.notes.append(reading.error[:400])
            sync.record_reading(self.ledger, problem=st.name, track=st.track,
                                kernel_id=st.kernel_id, reading=reading,
                                candidate=c.name, event="starter",
                                note=f"origin={c.origin}")
        return cands

    # -- 4. the target -------------------------------------------------------------------------
    def stage_target(self, st: ProblemState) -> None:
        prob = compose.load_problem(self.task_dir())
        best = None
        for c in (_cand_from_json(d) for d in st.candidates):
            if c.correct and c.geomean_ms:
                if best is None or c.geomean_ms < best.geomean_ms:
                    best = c
        if best is None:
            self._say("no scored starting point: the task keeps its default target")
            return
        ms = best.geomean_ms / self.step
        _set_target_ms(self.task_dir() / "task.toml", ms)
        self._validate_task(self.task_dir())
        self._say(f"target: {ms:.6g} ms ({best.name} {best.geomean_ms:.6g} ms / "
                  f"{self.step:g})")
        self._seed_kernel(st, best, prob)

    def _seed_kernel(self, st: ProblemState, cand: compose.Candidate, prob: dict) -> None:
        """Make the best start the student's first kernel (warmstart's `--init-kernel`
        validation when it is installed, a plain write otherwise)."""
        task_dir = self.task_dir()
        card = task_dir / "leaderboard" / "card"
        kf = prob.get("kernel_file", "kernel.py")
        if str(kf).endswith(".cu"):
            self._say("the card's kernel file is CUDA C++; seeding it from a Python start "
                      "is not supported -- left for the student")
            return
        target = card / Path(kf).name
        if not target.is_file():
            target = card / kf
        target.parent.mkdir(parents=True, exist_ok=True)
        source = cand.source or _read(cand.kernel_file)
        if not source:
            return
        try:
            from forgekernel.warmstart.validate import validate_source
        except ImportError:
            validate_source = None
        if validate_source is not None:
            try:
                problems = validate_source(source, prob.get("definition") or {})
            except TypeError:
                problems = validate_source(source)
            if problems:
                self._say(f"the start {cand.name} failed warmstart validation "
                          f"({problems[:2]}); seeding anyway is refused")
                return
        target.write_text(source, encoding="utf-8")
        self._say(f"seeded {target.relative_to(task_dir)} from {cand.name}")

    # -- 5. optimization -----------------------------------------------------------------------
    def stage_optimize(self, st: ProblemState) -> None:
        task_dir = self.task_dir()
        before = set(self._run_dirs(st.op))
        env = self._context_env()
        if env:
            self._say("context: " + ", ".join(f"{k}={Path(v).name}" for k, v in env.items()))
        if self.tools.evolve is not None:
            argv = ["evolve", str(task_dir), "--lineages", "3", "--generations", "1",
                    "--parallel", str(self.parallel), "--max-hours", f"{self.hours:g}"]
            langs = [e["route"] for e in plan_mod.languages_now()
                     if e["available"] and e["route"] in ("triton", "cutedsl", "cuda")]
            if langs:
                argv += ["--langs", ",".join(langs)]
            self._say("optimizing with forge evolve (3 lineages)")
            rc = self.tools.call_evolve(argv, env)
        else:
            rc = self.tools.call_new(["new", str(task_dir)], env)
            if rc != 0:
                raise RuntimeError(f"forge new exited {rc}")
            runs = self._run_dirs(st.op)
            new = [r for r in runs if r not in before]
            if not new:
                raise RuntimeError("forge new created no run directory")
            run_dir = new[-1]
            self._say(f"optimizing with forge run ({Path(run_dir).name}, "
                      f"budget {self.hours:g} h)")
            rc = self.tools.call_run(["run", run_dir], env)
        st.runs = sorted(set(self._run_dirs(st.op)))
        self.state.save(st)
        if rc != 0:
            # a run that stops short still leaves versions behind: collect them
            self._say(f"the loop exited {rc}; collecting whatever versions exist")

    def _context_env(self) -> dict:
        """The run context files this problem carries, as environment variables.

        The target picture is mandatory when it exists (background §4); roofline,
        lessons and a prior list join when the plan names files that exist."""
        files = {}
        target = self.tools.target_profile()
        if target is not None:
            files["target"] = target
        for key, env_name in (("roofline", "FK_ROOFLINE_PATH"),
                              ("lessons", "FK_LESSONS_PATH"), ("prior", "FK_PRIOR_PATH")):
            val = (self.entry.get(key) or "").strip() or os.environ.get(env_name, "").strip()
            if val and Path(val).is_file():
                files[key] = val
        return self.tools.context_env(files)

    def _run_dirs(self, op: str) -> list[str]:
        """Run directories of this op (and evolve populations), newest last."""
        from forgekernel.runs import list_runs
        out = []
        for r in list_runs(Path(self.tools.root) / ".artifacts"):
            if r.task_name == op or r.dir.name.startswith(f"{op}_"):
                out.append(str(r.dir))
        for pop in sorted((Path(self.tools.root) / ".artifacts" / "evolve").glob(f"{op}*")):
            if pop.is_dir():
                out.append(str(pop))
        return sorted(set(out))

    # -- 6. versions ---------------------------------------------------------------------------
    def stage_versions(self, st: ProblemState) -> None:
        prob = compose.load_problem(self.task_dir())
        board = compose.card_board(prob)
        kf = prob.get("kernel_file", "kernel.py")
        seen = {_cand_json(c).get("sha") for c in (_cand_from_json(d) for d in st.candidates)}
        cands = [_cand_from_json(d) for d in st.candidates]
        found = 0
        for run_dir in self._run_dirs(st.op):
            for c in _versions_of_run(Path(run_dir), st, kf, board):
                if c.sha and c.sha in seen:
                    continue
                seen.add(c.sha)
                cands.append(c)
                found += 1
        st.candidates = [_cand_json(c) for c in cands]
        self.state.save(st)
        self._say(f"{found} version(s) collected from the runs ({len(cands)} candidates total)")

    # -- 7. export -----------------------------------------------------------------------------
    def stage_export(self, st: ProblemState) -> None:
        task_dir = self.task_dir()
        cands = [_cand_from_json(d) for d in st.candidates]
        out = self.work / EXPORTS_DIR / st.op / "submission.py"
        res = compose.compose(task_dir, cands, out=out, tools=self.tools,
                              allow_slow_cold_start=self.allow_slow)
        st.export = {"path": str(res.path) if res.path else None, "kind": res.kind,
                     "candidate": res.candidate, "mean_s": res.mean_s,
                     "cold_start": res.cold_start, "error": res.error,
                     "rejected": res.rejected, "risk": res.risk, "notes": res.notes,
                     "licence": res.licence}
        self.state.save(st)
        if not res.ok:
            raise RuntimeError(res.error or "compose produced no submission.py")
        self.ledger.append({"event": "compose", "problem": st.name, "track": st.track,
                            "kernel_id": st.kernel_id, "source": sync.DEV,
                            "candidate": res.candidate, "mean_s": res.mean_s,
                            "cold_start_s": (res.cold_start or {}).get("s"),
                            "note": f"kind={res.kind} path={res.path}"})
        for r in res.risk:
            self.ledger.append({"event": "note", "problem": st.name, "track": st.track,
                                "kernel_id": st.kernel_id, "source": sync.DEV, "note": r})

    # -- 8. submit -----------------------------------------------------------------------------
    def stage_submit(self, st: ProblemState) -> None:
        if not self.submit:
            st.stages["submit"] = "skipped"
            self._say("not submitting (default; --submit turns it on)")
            return
        if self.tools.board_submit is None:
            raise RuntimeError("--submit was given but the board task is not installed")
        path = (st.export or {}).get("path")
        if not path or not Path(path).is_file():
            raise RuntimeError("nothing exported to submit")
        res = self.tools.board_submit(st.name, Path(path))
        self.ledger.append({"event": "submit", "problem": st.name, "track": st.track,
                            "kernel_id": st.kernel_id, "source": sync.BOARD,
                            "note": json.dumps(res)[:1500]})
        self._say(f"submitted: {res.get('rc')}")


# --- helpers -----------------------------------------------------------------------------------

def _set_board_flag(argv: list[str], record: Path | None) -> list[str]:
    """The card's T_b / T_SOL flag, exactly one of ``--sol-board <record>`` /
    ``--no-sol-board``: a cached record is passed as a file (the card freezes it and the
    run needs no network), otherwise the card is made without one."""
    out = _drop_flag(argv, "--sol-board")
    out = _drop_flag(out, "--no-sol-board")
    return [*out, "--sol-board", str(record)] if record is not None else [*out, "--no-sol-board"]


def _drop_flag(argv: list[str], flag: str) -> list[str]:
    """Remove a valueless flag (``--no-sol-board``) wherever it is."""
    return [a for a in argv if a != flag]


def _replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a ``--flag value`` pair (or ``--flag=value``) with a new pair, in place."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            i += 2 if i + 1 < len(argv) else 1
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return [*out, flag, str(value)]


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80]


def _read(path: str | Path) -> str:
    p = Path(path) if path else None
    return p.read_text(encoding="utf-8") if p and p.is_file() else ""


def _lang(entry: dict) -> str:
    lang = str(entry.get("language") or "python")
    return "cuda-cpp" if lang in ("cuda", "cuda_cpp") else "python"


def _read_kernel_entry(entry: dict) -> str:
    path = entry.get("path") or (str(Path(entry.get("dir", "")) / "kernel.py")
                                 if entry.get("dir") else "")
    if not path:
        return ""
    p = Path(path)
    if p.is_dir():
        p = p / "kernel.py"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _cand_json(c: compose.Candidate) -> dict:
    return {"name": c.name, "language": c.language, "mean_s": c.mean_s,
            "geomean_ms": c.geomean_ms, "cold_start_s": c.cold_start_s,
            "correct": c.correct, "sha": c.sha, "run": c.run,
            "kernel_file": c.kernel_file, "meta": c.meta, "origin": c.origin,
            "notes": c.notes, "source": c.source}


def _cand_from_json(d: dict) -> compose.Candidate:
    return compose.Candidate(**{k: v for k, v in d.items()
                                if k in compose.Candidate.__dataclass_fields__})


def _versions_of_run(run_dir: Path, st: ProblemState, kernel_file: str, board) -> list[compose.Candidate]:
    """Every KEEP version of one run (a run directory or an evolve population).

    ``kernel_file`` is the card's: a ``.cu`` means the run is a CUDA-mode one, and each
    version then carries ``language="cuda-cpp"`` so the export knows it needs
    `cuda_to_inline` (supplement 3) instead of trying to parse CUDA as Python."""
    language = "cuda-cpp" if str(kernel_file).endswith(".cu") else "python"
    if (run_dir / "population.json").is_file() or (run_dir / "variants").is_dir():
        return _versions_of_population(run_dir, st, kernel_file, board)
    op_dir = run_dir / "operators" / _task_name_of(run_dir)
    if not op_dir.is_dir():
        cands = sorted(p for p in (run_dir / "operators").glob("*") if p.is_dir())
        op_dir = cands[0] if cands else op_dir
    status = op_dir / "student" / "status.jsonl"
    if not status.is_file():
        return []
    rows = _status_rows(status)
    ws = run_dir / "ws"
    out: list[compose.Candidate] = []
    seen_sha: set[str] = set()
    by_iter = {r.get("iter"): r for r in rows}
    for r in rows:
        if r.get("verdict") not in KEEP_VERDICTS or r.get("correctness") is False:
            continue
        src_iter = r
        if r.get("reading_of") is not None and r["reading_of"] in by_iter:
            # a row whose tree is an earlier row's tree: its version IS that row's
            src_iter = by_iter[r["reading_of"]]
        sha = src_iter.get("sha") or r.get("sha")
        if sha and sha in seen_sha:
            continue
        source = _git_show(ws, sha, kernel_file) if sha else ""
        if not source:
            continue
        seen_sha.add(sha)
        lat = _latencies(src_iter)
        block = solscore.reading_s(lat, board) if lat else {}
        out.append(compose.Candidate(
            name=f"{Path(run_dir).name}:it{r.get('iter')}", source=source,
            language=language,
            mean_s=block.get("s_mean"), geomean_ms=solscore.geomean(lat.values()) if lat else None,
            correct=True, sha=sha or "", run=str(run_dir), kernel_file=kernel_file,
            meta={"source": f"forge run {Path(run_dir).name}", "license": "campaign"},
            origin="run",
            notes=[f"iteration {r.get('iter')} KEEP"
                   + (f", reading of it{src_iter.get('iter')}"
                      if src_iter is not r else "")]))
    best = _best_confirmation(run_dir, op_dir)
    if best and best.get("sha") and best.get("kernel"):
        out.append(compose.Candidate(
            name=f"{Path(run_dir).name}:confirmed", source=best["kernel"],
            language=language,
            mean_s=None, geomean_ms=best.get("median_ms"), cold_start_s=None,
            correct=True, sha=best["sha"], run=str(run_dir), kernel_file=kernel_file,
            meta={"source": f"forge run {Path(run_dir).name} (best_confirmation.json)",
                  "license": "campaign"}, origin="run",
            notes=["the run's final confirmation of its best kernel"]))
    return out


def _versions_of_population(pop: Path, st: ProblemState, kernel_file: str,
                            board) -> list[compose.Candidate]:
    """An evolve population's versions.

    Evolve's own archive is the stable shape (its task file pins it): every kept variant
    is ``variants/<lineage>-g<gen>/kernel.<ext>`` with a ``result.json`` beside it. The
    alternate route -- a ``result.json`` that names the run directory it came from --
    is followed too, with the key names it plausibly uses, and everything dedupes by
    source so a population that offers both does not double-count."""
    out, seen = [], set()
    extension = Path(kernel_file).suffix or ".py"
    for variant in sorted(pop.glob("variants/*")):
        kernel = variant / f"kernel{extension}"
        if not kernel.is_file():
            continue
        src = kernel.read_text(encoding="utf-8")
        if src in seen:
            continue
        seen.add(src)
        meas = _result_json(variant / "result.json")
        out.append(compose.Candidate(
            name=f"{pop.name}:{variant.name}", source=src,
            language="cuda-cpp" if extension == ".cu" else "python",
            mean_s=_pick(meas, "mean_s", "sol_score", "score", "metric"),
            geomean_ms=_pick(meas, "geomean_ms", "median_ms"),
            correct=True, sha=str(_pick(meas, "sha") or ""), run=str(variant),
            kernel_file=kernel_file,
            meta={"source": f"forage evolve {pop.name}/{variant.name}",
                  "license": "campaign"}, origin="run",
            notes=[f"evolve variant {variant.name}"]))
    for res in sorted(pop.rglob("result.json")):
        obj = _result_json(res)
        run = _first_dir(obj, "run", "run_dir", "run_path")
        if run is None:
            continue
        for c in _versions_of_run(run, st, kernel_file, board):
            if c.source in seen:
                continue
            seen.add(c.source)
            c.name = f"{pop.name}:{res.parent.name}:{c.name}"
            out.append(c)
    return out


def _result_json(path: Path) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _pick(obj: dict, *keys):
    for k in keys:
        if obj.get(k) is not None:
            return obj[k]
    return None


def _first_dir(obj: dict, *keys) -> Path | None:
    for k in keys:
        val = obj.get(k)
        if val and Path(val).is_dir():
            return Path(val)
    return None


def _task_name_of(run_dir: Path) -> str:
    """The task name a run directory belongs to (its run.json's ``task``)."""
    p = run_dir / "run.json"
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("task") or "")
    except (OSError, ValueError):
        return ""


def _status_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latencies(row: dict) -> dict:
    out = {}
    for w in row.get("workloads") or []:
        if not isinstance(w, dict):
            continue
        t = w.get("device_median_ms", w.get("bwd_median_ms"))
        if t is not None:
            out[str(w.get("name"))] = float(t)
    if not out:
        t = row.get("device_median_ms", row.get("bwd_median_ms"))
        if t is not None:
            out["wl0"] = float(t)
    return out


def _best_confirmation(run_dir: Path, op_dir: Path) -> dict | None:
    p = op_dir / "best_confirmation.json"
    if not p.is_file():
        return None
    try:
        c = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not c.get("sha"):
        return None
    sha = c["sha"]
    src = _git_show(run_dir / "ws", sha, _kernel_name(op_dir))
    if src:
        c["kernel"] = src
    return c


def _kernel_name(op_dir: Path) -> str:
    for name in ("kernel.py", "kernel.cu"):
        if (op_dir / name).is_file():
            return name
    return "kernel.py"


def _git_show(ws: Path, sha: str, kernel_file: str) -> str:
    if not sha or not ws.is_dir():
        return ""
    try:
        p = subprocess.run(["git", "-C", str(ws), "show",
                            f"{sha}:{Path(kernel_file).name}"],
                           capture_output=True, text=True, timeout=60, shell=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout if p.returncode == 0 else ""


def _set_target_ms(task_toml: Path, ms: float) -> None:
    """Write ``[target] ms`` into a task.toml, leaving every other line alone."""
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    out, in_target, inserted = [], False, False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_target and not inserted:
                out.append(f"ms     = {ms:.6g}")
                inserted = True
            in_target = stripped == "[target]"
            out.append(line)
            continue
        if in_target:
            if stripped.startswith("ms") and "=" in stripped:
                if not inserted:
                    out.append(f"ms     = {ms:.6g}")
                    inserted = True
                continue
            if stripped.startswith("ratio") and "=" in stripped and not stripped.startswith("#"):
                out.append(f"# {stripped}   # campaign: an explicit ms target wins")
                continue
        out.append(line)
    if in_target and not inserted:
        out.append(f"ms     = {ms:.6g}")
        inserted = True
    if not inserted:
        out += ["", "[target]", f"ms     = {ms:.6g}"]
    task_toml.write_text("\n".join(out) + "\n", encoding="utf-8")


# --- the driver --------------------------------------------------------------------------------

def run_plan(plan: dict, *, tools: tools_mod.Tools, work_dir: str | Path,
             problem_names: list[str] | None = None, parallel: int | None = None,
             hours: float | None = None, step: float | None = None, route: str = "auto",
             submit: bool = False, offline: bool = False,
             allow_slow_cold_start: bool = False, force: tuple[str, ...] = (),
             quiet: bool = False) -> list[ProblemState]:
    """Run every selected problem of the plan; ``parallel`` bounds the problems in flight."""
    work = Path(work_dir)
    state = StateStore(work)
    ledger = sync.Ledger(work / sync.LEDGER_FILE)
    entries = list(plan.get("problems") or [])
    if problem_names:
        wanted = set(problem_names)
        entries = [e for e in entries if e["name"] in wanted or e["op"] in wanted]
        missing = wanted - {e["name"] for e in entries} - {e["op"] for e in entries}
        if missing:
            raise ValueError(f"not in the plan: {', '.join(sorted(missing))}")
    par = max(1, int(parallel if parallel is not None else plan.get("parallel") or 1))
    runners = [ProblemRunner(tools, e, work_dir=work, state=state, ledger=ledger,
                             dataset=plan.get("dataset") or "",
                             step=step or plan.get("step") or solscore.DEFAULT_STEP,
                             hours=hours, parallel=par, route=route, submit=submit,
                             offline=offline,
                             allow_slow_cold_start=allow_slow_cold_start, force=force,
                             quiet=quiet) for e in entries]
    if par == 1 or len(runners) <= 1:
        return [r.run() for r in runners]
    with ThreadPoolExecutor(max_workers=par) as pool:
        futures = [pool.submit(r.run) for r in runners]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:                              # noqa: BLE001
                log(f"a problem runner raised: {type(e).__name__}: {e}")
    return out


def status(work_dir: str | Path, plan: dict) -> str:
    """One line per problem of the plan: its state and export."""
    state = StateStore(work_dir)
    lines = []
    for e in plan.get("problems") or []:
        st = state.load(e["op"])
        if st is None:
            lines.append(f"{e['name']:<52} {e['track']:<16} planned")
            continue
        exp = st.export or {}
        stages = ",".join(f"{k}:{v}" for k, v in st.stages.items())
        lines.append(f"{st.name:<52} {st.track:<16} {st.status:<8} "
                     f"{Path(exp.get('path')).name if exp.get('path') else '-':<14} "
                     f"{stages}")
    return "\n".join(lines)


def write_run_report(work_dir: str | Path, plan: dict) -> Path:
    """`run.md`: what each problem got, in the order it happened."""
    work = Path(work_dir)
    state = StateStore(work)
    lines = ["# campaign run", ""]
    for e in plan.get("problems") or []:
        st = state.load(e["op"])
        lines += [f"## {e['name']} ({e['track']})", ""]
        if st is None:
            lines += ["- not started", ""]
            continue
        lines += [f"- status: {st.status}", f"- task: {st.task_dir or '-'}"]
        if st.runs:
            lines.append(f"- runs: {', '.join(Path(r).name for r in st.runs)}")
        if st.candidates:
            best = sorted((c for c in st.candidates if c.get("mean_s") is not None),
                          key=lambda c: -c["mean_s"])[:3]
            for c in best:
                lines.append(f"- candidate {c['name']}: mean S {c['mean_s']:.4f}, "
                             f"{c.get('geomean_ms')} ms, origin {c.get('origin')}")
        exp = st.export or {}
        if exp:
            lines.append(f"- export: {exp.get('path')} ({exp.get('kind')}, "
                         f"{exp.get('candidate')})")
            if exp.get("rejected"):
                lines.append(f"- rejected: {len(exp['rejected'])} version(s)")
            for r in exp.get("risk") or []:
                lines.append(f"- risk: {r}")
        if st.error:
            lines.append(f"- error: {st.error}")
        lines.append("")
    p = work / "run.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def copy_export(work_dir: str | Path, op: str, dest: Path) -> Path:
    src = Path(work_dir) / EXPORTS_DIR / op / "submission.py"
    if not src.is_file():
        raise FileNotFoundError(f"{src}: no export for {op} (run the problem first)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest
