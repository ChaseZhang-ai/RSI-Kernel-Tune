"""`forge shapes plan`: the card's workloads -> clusters -> one task per cluster.

The plan is the whole contract between the three commands: ``run`` executes the
``forge init`` argv it records, and ``compose`` scores candidates against the card it
names and the workload list it carries -- so neither has to look at the card again to
know what the clusters were.

Reading the card:

* the task directory may be a real task (``tasks/<op>``: the card is ``[leaderboard]
  card`` from its ``task.toml``, conventionally ``leaderboard/card``) or the card
  directory itself (``.../eval/leaderboard/problem.json`` inside it);
* a card with one workload is not a clustering problem: `plan` says so and stops
  (exit 0). A directory that is not a board card at all is a usage error (exit 2).

The `forge init` arguments of a cluster are **the original problem's arguments** (what
``--init-args`` gives, else what the card's provenance lets us rebuild) with the workload
selector replaced by that cluster's indexes. ``--init-args`` is the authority: it is the
only way to carry things the card does not retain (a saved ``--sol-board`` record, a
``--init-kernel``, a ``--kernel-file``). Without it the derivation keeps the suite, the
dataset and the definition and passes ``--no-sol-board``, because the board's T_b / T_SOL
block frozen in the card is not a board *record* that ``forge init`` can re-validate --
and a note says so. The cluster's own target is written from the original card
regardless (``target_ms``), and scoring happens in `compose` on the original card.

Task directories are working files: they are listed in ``.git/info/exclude`` (found with
``git rev-parse --git-path``, so a worktree writes the right file), never in
``.gitignore``, and never committed.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rsi_kernel_tune.paths import repo_root
from rsi_kernel_tune.shapes import PRODUCT_DIR, SCHEMA, ShapesError
from rsi_kernel_tune.shapes import cluster as cluster_mod

PROBLEM_FILE = "eval/leaderboard/problem.json"
TASK_FILE = "task.toml"
TASKS_DIR = "tasks"
DEFAULT_SOL_GOAL = 0.90

# Flags that select the workloads a task is built for: the cluster supplies its own.
_WORKLOAD_FLAGS = ("--workload", "--check-workload")


class NothingToCluster(ShapesError):
    """The card is a fine board problem with one workload: there is nothing to split, and
    that is not a failure -- `plan` says so and exits 0 (task §5.1)."""


# --- the card ---------------------------------------------------------------------------

def find_card(task_dir: str | Path) -> Path:
    """The card directory (one holding ``eval/leaderboard/problem.json``).

    Raises :class:`ShapesError` naming what was looked for and where, so a mistyped path
    reads as a usage error instead of as "this problem has no workloads".
    """
    d = Path(task_dir).expanduser()
    if not d.is_dir():
        raise ShapesError(f"{d}: no such task or card directory")
    if (d / PROBLEM_FILE).is_file():
        return d
    toml = d / TASK_FILE
    if toml.is_file():
        card = _card_from_toml(toml)
        if card is not None and (d / card / PROBLEM_FILE).is_file():
            return d / card
    for rel in ("leaderboard/card", "card", "."):
        if (d / rel / PROBLEM_FILE).is_file():
            return (d / rel).resolve()
    hits = sorted(p for p in d.glob(f"*/{PROBLEM_FILE}") if p.is_file())
    if len(hits) == 1:
        return hits[0].parent
    raise ShapesError(
        f"{d}: not a leaderboard task (no {PROBLEM_FILE}; a task's card is named by "
        f"[leaderboard] card in its {TASK_FILE}). `forge shapes` works on board problems "
        f"only: their card carries the board's judge, inputs and clock")


def _card_from_toml(toml: Path) -> str | None:
    import tomllib

    try:
        cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    card = (cfg.get("leaderboard") or {}).get("card")
    return str(card) if card else None


def load_problem(task_dir: str | Path) -> tuple[dict, Path]:
    """``(the card's problem.json, the card directory)``."""
    card = find_card(task_dir)
    try:
        problem = json.loads((card / PROBLEM_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ShapesError(f"{card / PROBLEM_FILE} does not read as a card: {e}") from None
    if not isinstance(problem, dict) or not (problem.get("definition") or {}).get("name"):
        raise ShapesError(f"{card / PROBLEM_FILE}: not a board problem (no definition)")
    return problem, card


def op_of(task_dir: str | Path, problem: dict) -> str:
    """The task's name: the directory's, else the definition's."""
    d = Path(task_dir).expanduser()
    if d.name and d.name not in (".", "..", ""):
        return d.name
    return str(problem["definition"]["name"])


# --- the products -------------------------------------------------------------------------

def product_dir(task_dir: str | Path) -> Path:
    """``tasks/<op>/shapes/``: plan/run/compose's working directory."""
    return Path(task_dir).expanduser() / PRODUCT_DIR


def plan_path(task_dir: str | Path) -> Path:
    return product_dir(task_dir) / "plan.json"


# --- forge init arguments -------------------------------------------------------------------

def split_init_args(text: str | None) -> list[str]:
    """``--init-args``'s string -> an argv: shell quoting honoured, whitespace split."""
    if not text:
        return []
    return shlex.split(str(text), comments=False, posix=True)


def strip_workload_flags(argv: list[str]) -> list[str]:
    """``argv`` without its workload selectors (the cluster supplies its own)."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in _WORKLOAD_FLAGS:
            skip = True
            continue
        if any(token.startswith(f + "=") for f in _WORKLOAD_FLAGS):
            continue
        out.append(token)
    return out


def _drop_leading_op(argv: list[str], op: str) -> list[str]:
    """Tolerate ``--init-args "tasks/<op> --leaderboard ..."``: the op is not an argument."""
    if argv and not argv[0].startswith("-") and (
            argv[0] == op or argv[0].endswith(f"/{op}")):
        return argv[1:]
    return argv


def derive_init_args(problem: dict, card: Path) -> tuple[list[str], list[str]]:
    """The `forge init` arguments a card's provenance supports, and what it cannot.

    Uses only what the card records: ``suite``, ``source.dataset_root`` and
    ``source.problem`` / the definition's name. The board's T_b / T_SOL block in the card
    is *frozen readings*, not the ``kernels/<id>`` record ``forge init --sol-board``
    validates, so the derivation does not pretend to have it.
    """
    notes: list[str] = []
    suite = str(problem.get("suite") or "")
    if not suite:
        raise ShapesError("the card names no suite: pass --init-args with the original "
                          "`forge init` arguments")
    source = problem.get("source") or {}
    dataset = str(source.get("dataset_root") or "")
    definition = problem.get("definition") or {}
    name = str(definition.get("name") or "")
    argv = ["--leaderboard", suite]
    if not dataset:
        raise ShapesError(
            f"the card's source records no dataset_root: pass --init-args with the "
            f"original `forge init` arguments (`forge init <op> --leaderboard {suite} "
            f"--dataset <dir> --definition <name> ...`)")
    argv += ["--dataset", dataset]
    if suite == "sol-execbench":
        subset = str(source.get("problem") or "")
        if not subset:
            raise ShapesError("the card's source records no <subset>/<problem>: pass "
                              "--init-args with the original `forge init` arguments")
        argv += ["--definition", subset]
        argv.append("--no-sol-board")
        notes.append(
            "derived init arguments carry --no-sol-board: the board's T_b / T_SOL frozen "
            "in the card is not a saved kernels/<id> record that `forge init --sol-board` "
            "re-validates. The cluster's target is still written from the card's table, "
            "and `shapes compose` scores on the original card; pass --init-args with "
            "--sol-board <file> to keep SOL scoring inside the cluster runs")
    else:
        if not name:
            raise ShapesError("the card's definition has no name: pass --init-args")
        argv += ["--definition", name]
    return argv, notes


def init_argv_for(base: list[str], workloads: list[int]) -> list[str]:
    """The base arguments with the cluster's ``--workload`` selector appended."""
    if not workloads:
        raise ShapesError("a cluster with no workload cannot be run")
    return strip_workload_flags(list(base)) + ["--workload", ",".join(str(i) for i in workloads)]


# --- the plan ---------------------------------------------------------------------------

@dataclass
class Plan:
    op: str
    task_dir: Path
    card: Path
    suite: str
    definition: str
    workload_list: list[cluster_mod.Workload]
    clusters: list[cluster_mod.Cluster]
    init_base: list[str]
    targets: dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    sol_goal: float = DEFAULT_SOL_GOAL
    warnings: list[str] = field(default_factory=list)
    created: str = ""

    def name_of(self, c: cluster_mod.Cluster) -> str:
        return f"{self.op}-{c.name}"

    def cluster_json(self) -> list[dict]:
        return [{"name": self.name_of(c), "workloads": list(c.workloads),
                 "metric": c.metric, "t_sol_range": c.t_sol_range,
                 "elements_range": c.elements_range, "size": len(c.workloads),
                 "target_ms": self.targets.get(c.name),
                 "forge_init_argv": init_argv_for(self.init_base, c.workloads),
                 "reason": c.reason}
                for c in self.clusters]

    def as_json(self) -> dict:
        return {"schema": SCHEMA, "created": self.created or _now(),
                "op": self.op, "task": str(self.task_dir), "card": str(self.card),
                "product_dir": str(product_dir(self.task_dir)),
                "suite": self.suite, "definition": self.definition,
                "sol_goal": self.sol_goal, "params": dict(self.params),
                "init_argv_base": list(self.init_base),
                "workloads": [w.as_json() for w in self.workload_list],
                "clusters": self.cluster_json(),
                "warnings": list(self.warnings)}

    def cluster_of(self) -> dict[int, str]:
        return {i: self.name_of(c) for c in self.clusters for i in c.workloads}

    def workload(self, index: int) -> cluster_mod.Workload | None:
        return next((w for w in self.workload_list if w.index == index), None)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def build_plan(task_dir: str | Path, *, init_args: str | None = None,
               max_clusters: int = cluster_mod.DEFAULT_MAX_CLUSTERS,
               split_ratio: float = cluster_mod.DEFAULT_SPLIT_RATIO,
               min_size: int = cluster_mod.DEFAULT_MIN_SIZE,
               sol_goal: float = DEFAULT_SOL_GOAL,
               now: str | None = None) -> Plan:
    """Assemble a plan from the card. No I/O beyond reading it (and no GPU)."""
    problem, card = load_problem(task_dir)
    d = Path(task_dir).expanduser().resolve()
    op = op_of(d, problem)
    workloads = cluster_mod.card_workloads(problem)
    if len(workloads) < 2:
        raise NothingToCluster(
            f"{op}: the card times {len(workloads)} workload -- nothing to cluster. One "
            f"workload is one shape: a single specialised kernel is already the optimal "
            f"decomposition (use `forge bench` / `forge run` on the task itself)")
    notes: list[str] = []
    try:
        clusters = cluster_mod.cluster(workloads, max_clusters=max_clusters,
                                       split_ratio=split_ratio, min_size=min_size,
                                       notes=notes)
    except ValueError as e:
        raise ShapesError(f"{op}: {e}") from None
    problems = cluster_mod.check_partition(clusters, workloads)
    if problems:
        raise ShapesError("the clustering does not partition the card: "
                          + "; ".join(problems))
    base = split_init_args(init_args)
    base = _drop_leading_op(base, op)
    had_selector = any(t == "--workload" or t.startswith("--workload=") for t in base) \
        or any(t == "--check-workload" or t.startswith("--check-workload=") for t in base)
    base = strip_workload_flags(base)
    warnings: list[str] = []
    if base:
        warnings.append("init arguments: --init-args (used verbatim, workload selector "
                        "replaced per cluster)")
    else:
        base, derived = derive_init_args(problem, card)
        warnings.append("init arguments: derived from the card's provenance (pass "
                        "--init-args to override)")
        warnings += derived
    if base and not had_selector and init_args:
        warnings.append("the base arguments carry no --workload: the original task timed "
                        "the dataset's default selection")
    plan = Plan(op=op, task_dir=d, card=card.resolve(),
                suite=str(problem.get("suite") or ""),
                definition=str((problem.get("definition") or {}).get("name") or ""),
                workload_list=workloads, clusters=clusters, init_base=base,
                params={"max_clusters": int(max_clusters),
                        "split_ratio": float(split_ratio), "min_size": int(min_size)},
                sol_goal=float(sol_goal), warnings=warnings,
                created=now or _now())
    plan.targets = cluster_mod.target_ms(clusters, workloads, sol_goal=sol_goal)
    missing = [plan.name_of(c) for c in clusters if c.name not in plan.targets]
    if missing:
        warnings.append("no [target] ms for " + ", ".join(missing)
                        + ": the card carries no usable T_b / T_SOL (the cluster task keeps "
                          "whatever target its own configuration gives)")
    wl_counts = ", ".join(f"{plan.name_of(c)}={len(c.workloads)}" for c in clusters)
    warnings.append(f"{len(clusters)} cluster(s) over {len(workloads)} workloads "
                    f"({wl_counts}); metric {clusters[0].metric}")
    warnings += notes
    return plan


def write_plan(plan: Plan, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan.as_json(), indent=1, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def write_plan_doc(doc: dict, product: str | Path) -> Path:
    """Write an (already rendered) plan document into the product directory."""
    p = Path(product) / "plan.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load_plan(path: str | Path) -> dict:
    """A plan document (``<dir>`` or ``<dir>/plan.json``)."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / PRODUCT_DIR / "plan.json" if (p / PRODUCT_DIR / "plan.json").is_file() \
            else p / "plan.json"
    if not p.is_file():
        raise FileNotFoundError(f"no plan at {p} (`forge shapes plan` writes one)")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {obj.get('schema')!r}, expected {SCHEMA}")
    if not obj.get("clusters"):
        raise ValueError(f"{p}: no clusters")
    return obj


def plan_for(task_dir: str | Path, **kw) -> tuple[dict, Path | None]:
    """The plan of a task: the saved one, else a freshly built one (not written).

    Used by `compose`, so ``forge shapes compose tasks/<op>`` works on a task that was
    never planned (campaign calls it that way) -- the defaults are the documented ones.
    """
    d = Path(task_dir).expanduser()
    for p in (plan_path(d), d / "plan.json"):
        if p.is_file():
            return load_plan(p), p
    plan = build_plan(d, **{k: v for k, v in kw.items()
                            if k in ("init_args", "max_clusters", "split_ratio",
                                     "min_size", "sol_goal")})
    return plan.as_json(), None


def plan_format(plan: dict) -> str:
    """A human summary of a plan document."""
    lines = [f"shapes plan: {plan.get('op')} ({plan.get('definition')}, "
             f"{plan.get('suite')})",
             f"  card: {plan.get('card')}",
             f"  sol goal: {plan.get('sol_goal')}"]
    for c in plan.get("clusters") or []:
        target = c.get("target_ms")
        lines.append(f"  {c['name']}: workloads {c['workloads']} "
                     f"(metric {c.get('metric')}, size {c.get('size')})"
                     + (f", target {target:.6g} ms" if target else ""))
        lines.append(f"      forge init {c['name']} " + " ".join(c["forge_init_argv"]))
        if c.get("reason"):
            lines.append(f"      {c['reason']}")
    for w in plan.get("warnings") or []:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


# --- the exclude list -----------------------------------------------------------------------

def ensure_excluded(root: str | Path, rels: list[str]) -> dict:
    """Add ``rels`` to the repository's ``.git/info/exclude``; never ``.gitignore``.

    ``git rev-parse --git-path info/exclude`` resolves the file even when ``root`` is a
    worktree (where ``.git`` is a file). Returns ``{path, added, error}``; a repository
    that cannot be asked is a warning, not a failure (the task directories are never
    committed by `run` anyway -- it only runs `forge` commands).
    """
    out = {"path": "", "added": [], "error": ""}
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-path",
                               "info/exclude"], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if proc.returncode != 0:
        out["error"] = (proc.stderr or proc.stdout).strip()[:300] or "git rev-parse failed"
        return out
    path = Path(proc.stdout.strip())
    if not path.is_absolute():
        path = Path(root) / path
    out["path"] = str(path)
    try:
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    have = {line.strip() for line in existing.splitlines()}
    wanted = [f"/{r.strip('/')}/" for r in rels if r and f"/{r.strip('/')}/" not in have]
    if not wanted:
        return out
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write("# forge shapes: per-cluster task directories (never committed)\n")
            fh.write("\n".join(wanted) + "\n")
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["added"] = wanted
    return out


def excluded_rels(plan: dict) -> list[str]:
    """What a plan's directories are, relative to the repository root.

    The cluster tasks are always ``tasks/<cluster>``: `forge init` writes under the
    checkout's ``tasks/`` whatever directory the plan was made from. The product directory is
    added only when it lives under ``tasks/`` too (a fixture or a scratch tree outside the
    task tree has no business in the exclude list), and an entry outside the repository has
    no relative form at all.
    """
    root = repo_root()
    rels = [f"{TASKS_DIR}/{c.get('name')}" for c in plan.get("clusters") or []
            if c.get("name")]
    product = plan.get("product_dir")
    if product:
        try:
            text = str(Path(product).relative_to(root))
        except ValueError:
            text = ""
        if text.startswith(f"{TASKS_DIR}/"):
            rels.append(text)
    return rels
