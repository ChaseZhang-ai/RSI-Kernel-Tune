"""The battle plan: which problems, from which starting points, in which languages.

`campaign plan` turns the gap list into a `plan.json` a later `campaign run` executes
without asking anything again. It records, per problem, the exact `forge init` arguments
(the dataset subset, the definition, the workload selector), the run's budget, the
target derivation (the fastest starting point's reading / ``--step``), and the
implementation routes to try.

Two deliberate choices, both from the task file:

* **the pilot defaults to the three named problems** (`069_rms_norm` (L1),
  `001_fp8_attention_output_projection` (Quant) and
  `030_attention_output_projection_with_residual` (L1)): gap-list leaders, simple
  shapes, three different tracks -- one pilot crossing two tracks proves the pipeline
  against different subsets. `--pilot auto` instead takes the top of the
  gap x feasibility ordering and says why each was picked.
* **the language mix runs in parallel, not as a vote** (supplement 2): the plan lists
  every route -- Triton, CuTe DSL, CUDA C++ (RSI-Kernel-Tune's `cuda` mode), and the
  torch-operator / inline-CUDA baselines -- with whether the current
  `rsi_kernel_tune.task.LANGUAGES` accepts it and whether its product can even enter a
  collection submission. A route that cannot be exported is kept in the plan with the
  reason (`cuda` -> "needs klib"), because hiding it would hide the fact that the
  language with the best public record is the one the export path cannot take today.

The plan never talks to a GPU and never runs a command; `--dry-run` prints the argv
each problem would execute, and that is also what `run` reads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rsi_kernel_tune.campaign import solscore, targets

# The documented pilot (task §6): gap-list leaders, simple shapes, three tracks.
DEFAULT_PILOT = ("069_rms_norm", "001_fp8_attention_output_projection",
                 "030_attention_output_projection_with_residual")
DEFAULT_HOURS = 6.0
DEFAULT_PARALLEL = 2
DEFAULT_PILOT_COUNT = 3
# The implementation routes, in the order the plan prefers them. `torch` is the
# reference-baseline route (seedlib's `forge lib baselines`), not a board language.
ROUTES = ("triton", "cutedsl", "cuda", "torch")
ROUTE_NOTE = {
    "triton": "Triton -- compiled per device at runtime; no architecture pinning",
    "cutedsl": "CuTe DSL -- the base repo's default board language",
    "cuda": "CUDA C++ (RSI-Kernel-Tune's cuda mode) -- its .cu must go through "
            "cuda_to_inline (seedlib) before a collection submission can take it",
    "torch": "torch operators / baseline generator -- the reference wrapped as run()",
}
PLAN_SCHEMA = 1


@dataclass
class PlanProblem:
    kernel_id: int
    name: str
    track: str
    op: str
    subset: str
    definition: str                 # "<subset>/<name>", the --definition argument
    leader: float
    best_known: float | None
    gap: float
    starter: str
    starter_detail: str
    workloads: str = "all"          # --workload selector, "all" resolved at run time
    hours: float = DEFAULT_HOURS
    init_argv: list[str] = field(default_factory=list)
    status: str = "planned"
    notes: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"kernel_id": self.kernel_id, "name": self.name, "track": self.track,
                "op": self.op, "subset": self.subset, "definition": self.definition,
                "leader": self.leader, "best_known": self.best_known, "gap": self.gap,
                "starter": self.starter, "starter_detail": self.starter_detail,
                "workloads": self.workloads, "hours": self.hours,
                "init_argv": list(self.init_argv), "status": self.status,
                "notes": list(self.notes)}


@dataclass
class Plan:
    snapshot: str
    dataset: str
    incumbent: str
    problems: list[PlanProblem]
    work_dir: str
    step: float = solscore.DEFAULT_STEP
    parallel: int = DEFAULT_PARALLEL
    hours_per_problem: float = DEFAULT_HOURS
    languages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created: str = ""

    def as_json(self) -> dict:
        return {"schema": PLAN_SCHEMA, "created": self.created or _now(),
                "snapshot": self.snapshot, "dataset": self.dataset,
                "incumbent": self.incumbent, "work_dir": self.work_dir,
                "step": self.step, "parallel": self.parallel,
                "hours_per_problem": self.hours_per_problem,
                "languages": self.languages, "warnings": self.warnings,
                "problems": [p.as_json() for p in self.problems]}

    def problem(self, name_or_op: str) -> PlanProblem | None:
        for p in self.problems:
            if name_or_op in (p.name, p.op):
                return p
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def languages_now() -> list[dict]:
    """Which implementation routes the current harness can actually take, and why not.

    `rsi_kernel_tune.task.LANGUAGES` is read at call time (the triton task widens it), so a
    plan made today says what today's checkout supports."""
    try:
        from rsi_kernel_tune.task import LANGUAGES
    except ImportError:                                         # pragma: no cover
        LANGUAGES = ("cutedsl", "cuda")
    out = []
    for route in ROUTES:
        entry = {"route": route, "note": ROUTE_NOTE[route]}
        if route == "torch":
            entry.update(available=True,
                         export="python (the reference, wrapped, behind try/except)")
        elif route in LANGUAGES:
            entry.update(available=True, export=("python (single file)" if route != "cuda"
                                                 else "needs klib: cuda_to_inline"))
        else:
            entry.update(available=False,
                         export=f"not in rsi_kernel_tune.task.LANGUAGES {tuple(LANGUAGES)}")
        out.append(entry)
    return out


def op_name(snap: targets.Snapshot, row: targets.TargetRow) -> str:
    """The task directory name: the problem name when it is unique on the board, else
    ``<name>__<track>`` (names repeat across tracks -- `001_...` exists in four)."""
    same = [k for k in snap.kernel_ids if snap.name_of(k) == row.name]
    return row.name if len(same) == 1 else f"{row.name}__{row.track}"


def parse_selection(spec: str | None, snap: targets.Snapshot,
                    rows: list[targets.TargetRow], *, count: int = DEFAULT_PILOT_COUNT
                    ) -> tuple[list[targets.TargetRow], str]:
    """``(rows, why)`` for ``--pilot`` / ``--problems``.

    ``None`` or ``"default"`` -> the documented pilot; ``"auto"`` -> the top ``count`` of
    the gap x feasibility ordering; a bare integer -> the top N by gap; anything else is
    a comma-separated list of ``name`` / ``track:name`` items."""
    by_name: dict[str, list[targets.TargetRow]] = {}
    for r in rows:
        by_name.setdefault(r.name, []).append(r)
    spec_s = (spec or "default").strip()
    if spec_s in ("", "default"):
        picked, missing = [], []
        for name in DEFAULT_PILOT:
            hit = by_name.get(name)
            if hit:
                picked.append(hit[0])
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"the default pilot problems are not on this snapshot: "
                             f"{', '.join(missing)} (pass --problems to choose)")
        return picked, "the documented pilot (gap-list leaders, simple shapes, 3 tracks)"
    if spec_s == "auto":
        top = targets.ranked(rows, by="priority")[:max(1, count)]
        return top, f"top {len(top)} by gap x feasibility (--pilot auto)"
    if spec_s.isdigit():
        n = max(1, int(spec_s))
        return targets.ranked(rows)[:n], f"top {n} by gap"
    picked = []
    for item in [x.strip() for x in spec_s.replace(" ", ",").split(",") if x.strip()]:
        track, _, name = item.rpartition(":")
        hit = by_name.get(name if track else item) or []
        if track:
            hit = [r for r in hit if r.track == track]
        if not hit:
            near = [r for r in rows if (name if track else item) in r.name][:3]
            raise ValueError(f"{item!r} is not a problem of this snapshot"
                             + (f" (near: {', '.join(r.name for r in near)})" if near else ""))
        if len(hit) > 1:
            raise ValueError(f"{item!r} is ambiguous across tracks "
                             f"({', '.join(r.track for r in hit)}); write track:name")
        picked.append(hit[0])
    return picked, "explicit --problems"


def build_plan(snap: targets.Snapshot, *, selection: str | None, dataset: str,
               work_dir: str | Path, step: float = solscore.DEFAULT_STEP,
               hours: float = DEFAULT_HOURS, parallel: int = DEFAULT_PARALLEL,
               count: int = DEFAULT_PILOT_COUNT, workloads: str = "all",
               incumbent: str | None = None, ours: dict[str, float] | None = None,
               starters: dict[str, tuple[str, str]] | None = None,
               with_prior: bool = True, root: str | Path | None = None) -> Plan:
    """Assemble the plan. No I/O besides reading `kernel_library/index.json` (for the
    starter column) and the dataset directory's existence (a warning only)."""
    starters = starters if starters is not None else targets.starter_situation(
        snap, root or Path.cwd())
    rows, who, why = targets.rows(snap, incumbent=incumbent, ours=ours,
                                  starters=starters, with_prior=with_prior)
    picked, how = parse_selection(selection, snap, rows, count=count)
    langs = languages_now()
    ds = Path(dataset).expanduser()
    warnings = list(snap.warnings)
    if dataset and not ds.is_dir():
        warnings.append(f"dataset {ds} does not exist yet: `forge init` will fail until "
                        f"download_solexecbench.py has written data/benchmark/<subset>/")
    problems = []
    auto = selection == "auto"
    for r in picked:
        op = op_name(snap, r)
        subset = solscore.TRACK_SUBSET[r.track]
        p = PlanProblem(
            kernel_id=r.kernel_id, name=r.name, track=r.track, op=op, subset=subset,
            definition=f"{subset}/{r.name}", leader=r.leader, best_known=r.best_known,
            gap=r.gap, starter=r.starter, starter_detail=r.starter_detail,
            workloads=workloads, hours=hours)
        p.init_argv = init_argv(p, dataset=str(ds))
        if auto:
            # why this one: the gap, and the prior's factor when it moved the pick
            p.notes.append(f"picked for gap {r.gap:+.4f} ({r.priority_why}; priority "
                           f"{r.priority or 0:.4f})")
        if r.starter == targets.STARTER_UNKNOWN:
            p.notes.append("no reference library: the torch baselines are the first starts")
        if r.track == "quant":
            p.notes.append("quant track: a truncated-precision candidate must pass the "
                           "card's tolerance on every workload before it is exported")
        if any(not e["available"] for e in langs if e["route"] != "torch"):
            p.notes.append("some routes are not accepted by this checkout's LANGUAGES; "
                           "the plan lists them anyway (supplement 2)")
        problems.append(p)
    plan = Plan(snapshot=str(snap.dir), dataset=str(ds), incumbent=who, problems=problems,
                work_dir=str(work_dir), step=step, parallel=parallel,
                hours_per_problem=hours, languages=langs, warnings=warnings,
                created=_now())
    plan.warnings.append(f"pilot selection: {how}")
    plan.warnings.append(f"incumbent: {who} ({why.get('by')}, overall "
                         f"{why.get('overall', 0):.6f})")
    return plan


def init_argv(problem: PlanProblem, *, dataset: str, sol_board: str | None = None,
              init_kernel: str | None = None, kernel_file: str | None = None) -> list[str]:
    """The arguments of this problem's `forge init` (everything after the subcommand)."""
    argv = [problem.op, "--leaderboard", "sol-execbench",
            "--dataset", dataset, "--definition", problem.definition]
    if problem.workloads and problem.workloads != "all":
        argv += ["--workload", problem.workloads]
    if sol_board:
        argv += ["--sol-board", str(sol_board)]
    else:
        argv += ["--no-sol-board"]
    if kernel_file:
        argv += ["--kernel-file", str(kernel_file)]
    if init_kernel:
        argv += ["--init-kernel", str(init_kernel)]
    return argv


def write_plan(plan: Plan, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan.as_json(), indent=1, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def load_plan(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "plan.json"
    if not p.is_file():
        raise FileNotFoundError(f"no plan at {p} (`forge campaign plan` writes one)")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{p}: schema {obj.get('schema')!r}, expected {PLAN_SCHEMA}")
    return obj


def format_plan(plan: Plan) -> str:
    lines = [f"campaign plan -> {plan.work_dir}", f"  snapshot: {plan.snapshot}",
             f"  dataset:  {plan.dataset}", f"  incumbent: {plan.incumbent}",
             f"  step {plan.step:g}, {plan.hours_per_problem:g} h/problem, "
             f"parallel {plan.parallel}"]
    for p in plan.problems:
        lines.append(f"  [{p.track}] {p.name} (gap {p.gap:+.4f}, start {p.starter}) "
                     f"-> tasks/{p.op}/")
        if p.notes:
            lines += [f"      note: {n}" for n in p.notes]
    for e in plan.languages:
        flag = "ok " if e["available"] else "no "
        lines.append(f"  route {flag}{e['route']:<8} export: {e['export']}")
    for w in plan.warnings:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


def dry_run_commands(plan: Plan, *, evolve_available: bool = False) -> list[str]:
    """The exact argv `run` would execute for each problem, for `--dry-run`."""
    out = []
    for p in plan.problems:
        out.append("forge init " + " ".join(p.init_argv))
        out.append(f"forge bench tasks/{p.op} --candidate <each start> --repeat 2   # when "
                   f"benchx is installed; otherwise the start is taken unscored")
        out.append(f"# write [target] ms = best start's geomean / {plan.step:g} into "
                   f"tasks/{p.op}/task.toml")
        if evolve_available:
            out.append(f"forge evolve tasks/{p.op} --lineages 3 --parallel "
                       f"{max(1, plan.parallel)} --generations 1 --max-hours {p.hours:g}")
        else:
            out.append(f"forge new tasks/{p.op} && forge run .artifacts/{p.op}_<id>   "
                       f"# evolve not installed; budget {p.hours:g} h")
        out.append(f"forge campaign export {p.name}   # compose every correct version into "
                   f"submission.py")
    return out
