"""Lineage task directories: one directory per (lineage, generation), the thing a
generation's ``forge run`` actually runs.

Two kinds of base task, two ways to hand a start to the next generation (task
evolve §5.3, revised by its 2nd-version addendum):

ordinary task (``description.md`` + ``shapes.json`` + a source reference)
    the donor's kernel is copied into ``reference/kernel.py`` and
    ``[reference] run/source`` point at it -- the same source-mode handoff
    ``scripts/self_evolve.py`` makes.  Only Python kernels can be handed over
    this way; a ``.cu`` donor is refused with that explanation.

    ``--init-kernel`` (task warmstart) is not what an ordinary task's student
    reads: the student starts from the reference the loop seals.  When the
    handoff is readable this module therefore prefers warmstart's ``--init-kernel``
    for board tasks and, for ordinary tasks, copies the donor into the task's
    ``prior/`` directory (readable, and the source reference is readable too).

board task (``leaderboard/card``; ``[reference]`` is refused by ``load_task``)
    the next generation is a fresh ``forge init --leaderboard …`` with either
      * ``--init-kernel <donor>/kernel.py`` -- the readable default, needs task
        warmstart to accept the flag; or
      * ``--sealed-ref <solution.json>`` -- ``launch.cmd_export`` of the donor's
        confirmed best, the black-box path (``--sealed-ref`` / ``--blackbox``),
        or the fallback when warmstart is not installed.
    The board record (T_b / T_SOL) is carried over when the card covers every
    workload of its problem, so the descendant card still scores SOL; otherwise
    the re-init is told ``--no-sol-board`` and the lineage's metric falls back to
    the in-job reference ratio.

Every directory this module writes is a working directory: it is appended to the
repository's local ``.git/info/exclude`` and must not be committed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel import launch, task as task_mod
from forgekernel.task import Task, dump_task_toml, load_task, read_task_toml

DONOR_KERNEL = "kernel.py"
REF_DIR = "reference"
PRIOR_DIR = "prior"


class LineageError(ValueError):
    """The next generation's task could not be written; the message says why."""


@dataclass(frozen=True)
class Reference:
    """What the new task starts from.  ``kind`` is what the archive records:
    ``self`` (this lineage's own best), ``foreign`` (another lineage's best),
    ``seed`` (a reference-library candidate)."""

    kind: str
    kernel: Path | None = None      # a Python kernel file (the readable handoff)
    run: Path | None = None         # the donor run, for cmd_export
    language: str = "cutedsl"
    variant: str = ""               # l<i>-g<k> of the donor
    seed_id: str = ""

    def describe(self) -> dict:
        return {"kind": self.kind, "variant": self.variant, "seed": self.seed_id,
                "language": self.language,
                "kernel": str(self.kernel) if self.kernel else None,
                "run": str(self.run) if self.run else None}


@dataclass
class Request:
    base: Task
    dest: Path
    language: str
    generation: int
    lineages: int
    step: float
    reference: Reference | None = None
    blackbox: bool = False          # board handoff: sealed, never readable
    init_kernel_supported: bool = True   # task warmstart installed (--init-kernel)
    effort: str | None = None
    prior: tuple[Path, ...] = ()
    workdir: Path = Path(".")
    add_exclude: bool = True

    @property
    def verifies_target(self) -> bool:
        """Generation 1 of ``--lineages 1`` keeps the base task's target -- the
        regression baseline of scripts/self_evolve.py; everything else targets
        ``--step`` times its reference."""
        return not (self.lineages == 1 and self.generation == 1)


# -- names -------------------------------------------------------------------------------


def base_name(name: str) -> str:
    """``scripts/self_evolve.py``'s ``re.sub(r"-g\\d+$", "", name)``."""
    return re.sub(r"-g\d+$", "", name)


def task_name(name: str, lineage: int, generation: int, lineages: int) -> str:
    """``<op>-g<k>`` for one lineage (the old script's naming); ``<op>.e<i>.g<k>`` else."""
    base = base_name(name)
    if lineages == 1:
        return name if generation == 1 else f"{base}-g{generation}"
    return f"{base}.e{lineage}.g{generation}"


def variant_name(lineage: int, generation: int) -> str:
    return f"l{lineage}-g{generation}"


def check_language(language: str) -> None:
    accepted = tuple(task_mod.LANGUAGES)
    if language not in accepted:
        raise LineageError(f"language {language!r} is not accepted by this build: "
                           f"task.LANGUAGES = {list(accepted)} (a language lands here "
                           f"when its task does)")


# -- reference-library seeds (task seedlib, soft dependency) ------------------------------

ALLOWED_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause",
                              "bsd-2", "bsd-3", "bsd"})


def _license_allows(value: object) -> bool:
    return str(value or "").strip().lower() in ALLOWED_LICENSES


def find_seeds(problem: str, k: int, *, log=print) -> list[dict]:
    """The first ``k`` usable candidates of ``forge lib find <problem>``.

    Soft dependency on task seedlib (``forgekernel.klib``): the index is consulted
    through a callable ``find`` / ``find_kernels`` / ``search`` with the problem
    name and the count; entries are dicts with ``id``, ``path`` (a single-file
    kernel.py), ``language`` and ``license``.  Anything unreadable, unlicensed or
    without a source file is skipped with a log line -- never a hard failure.
    """
    try:
        import forgekernel.klib as klib
    except Exception:  # noqa: BLE001 — absent or half-installed = unavailable
        log("[evolve] --from-lib: forgekernel.klib is not installed; starting every "
            "lineage from the task's own reference instead")
        return []
    fn = None
    for name in ("find", "find_kernels", "search"):
        fn = getattr(klib, name, None)
        if callable(fn):
            break
    if fn is None:
        log("[evolve] --from-lib: forgekernel.klib has no find()/search(); falling back "
            "to the task's own reference")
        return []
    result = None
    for args, kwargs in (((problem, k), {}), ((problem,), {"k": k}),
                         ((problem,), {"limit": k}), ((problem,), {})):
        try:
            result = fn(*args, **kwargs)
            break
        except TypeError:
            continue
        except Exception as e:  # noqa: BLE001
            log(f"[evolve] --from-lib: klib lookup failed ({e}); falling back")
            return []
    if isinstance(result, dict):
        for key in ("items", "results", "kernels", "entries"):
            if isinstance(result.get(key), list):
                result = result[key]
                break
    if not isinstance(result, list):
        log("[evolve] --from-lib: klib returned no list; falling back")
        return []
    seeds: list[dict] = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not path:
            continue
        p = Path(str(path)).expanduser()
        if not p.is_file():
            log(f"[evolve] --from-lib: skipping {entry.get('id')}: {p} is not a file")
            continue
        if not _license_allows(entry.get("license")):
            log(f"[evolve] --from-lib: skipping {entry.get('id')}: license "
                f"{entry.get('license')!r} is not in the merge-allowed set")
            continue
        seeds.append({"id": str(entry.get("id") or p.parent.name),
                      "path": str(p), "language": str(entry.get("language") or ""),
                      "license": str(entry.get("license") or ""),
                      "reported": entry.get("reported") or {}})
        if len(seeds) >= k:
            break
    if not seeds:
        log(f"[evolve] --from-lib: no usable candidate for {problem!r}; falling back "
            f"to the task's own reference")
    return seeds


# -- generation --------------------------------------------------------------------------


def make_task(req: Request) -> dict:
    """Write ``req.dest`` and return the record the population archives."""
    check_language(req.language)
    if req.dest.exists():
        raise FileExistsError(f"{req.dest} exists; remove it or start the population "
                              f"elsewhere")
    if req.base.leaderboard is not None:
        record = _board_task(req)
    else:
        record = _ordinary_task(req)
    if req.add_exclude:
        add_exclude_pattern(req.base.dir.parent, req.dest)
    record.update({"task": str(req.dest), "name": req.dest.name,
                   "language": req.language, "generation": req.generation,
                   "lineage": None})
    return record


def copy_task(req: Request, src: Path) -> dict:
    """A generation with no new start (a retry after a generation without a confirmed
    best): the previous generation's directory, copied and re-patched."""
    check_language(req.language)
    if req.dest.exists():
        raise FileExistsError(f"{req.dest} exists")
    shutil.copytree(src, req.dest)
    cfg = read_task_toml(req.dest)
    if req.base.leaderboard is not None:
        _patch_board_toml(cfg, req)
    else:
        task_tbl = dict(cfg.get("task") or {})
        task_tbl["language"] = req.language
        cfg["task"] = task_tbl
        _apply_target(cfg, req.step if req.verifies_target else None)
        _apply_effort(cfg, req.effort)
    _write_toml(req.dest, cfg)
    _validate(req.dest)
    if req.add_exclude:
        add_exclude_pattern(req.base.dir.parent, req.dest)
    return {"reference": {"kind": "retry", "variant": str(src.name), "seed": "",
                          "language": req.language, "kernel": None, "run": None},
            "handoff": "retry (same reference)",
            "task": str(req.dest), "name": req.dest.name, "language": req.language,
            "generation": req.generation, "lineage": None}


def _ordinary_task(req: Request) -> dict:
    base, dest = req.base, req.dest
    ref = req.reference
    if ref is not None and not _is_python(ref):
        raise LineageError(
            f"the donor ({ref.variant or ref.seed_id or 'a reference'}) is a "
            f"{ref.language!r} kernel; a .cu donor cannot become an ordinary task's "
            f"source reference (only Python kernels can, task evolve §5.3)")
    dest.mkdir(parents=True)
    for name in ("description.md", "shapes.json"):
        if (base.dir / name).is_file():
            shutil.copy2(base.dir / name, dest / name)
    if (base.dir / "inputs").is_dir():
        shutil.copytree(base.dir / "inputs", dest / "inputs")
    cfg = read_task_toml(base.dir)
    task_tbl = dict(cfg.get("task") or {})
    task_tbl["language"] = req.language
    cfg["task"] = task_tbl
    if ref is None:
        if (base.dir / REF_DIR).is_dir():
            shutil.copytree(base.dir / REF_DIR, dest / REF_DIR)
    else:
        _write_reference(dest, ref)
        cfg["reference"] = {"run": f"{REF_DIR}/{DONOR_KERNEL}:run",
                            "source": REF_DIR}
    _apply_target(cfg, req.step if req.verifies_target else None)
    _apply_effort(cfg, req.effort)
    _write_toml(dest, cfg)
    _validate(dest)
    _write_prior(dest, req, ref)
    return {"reference": (ref.describe() if ref is not None
                          else {"kind": "base", "variant": "", "seed": "",
                                "language": base.language, "kernel": None, "run": None}),
            "handoff": ("reference/kernel.py" if ref is not None and ref.kernel
                        else "base reference (copied)")}


def _write_reference(dest: Path, ref: Reference) -> None:
    if ref.kernel is None or not Path(ref.kernel).is_file():
        raise LineageError(f"the donor kernel for {ref.variant or ref.seed_id} is gone "
                           f"({ref.kernel}); cannot build the next generation's reference")
    dest.joinpath(REF_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(ref.kernel), dest / REF_DIR / DONOR_KERNEL)


def _apply_target(cfg: dict, step: float | None) -> None:
    target = dict(cfg.get("target") or {})
    if step is None:
        return                      # generation 1, one lineage: the base task's own target
    target.pop("ms", None)
    target["ratio"] = float(step)
    cfg["target"] = target


def _apply_effort(cfg: dict, effort: str | None) -> None:
    """``--efforts``: the lineage's student run one setting, so the pilot can A/B them."""
    if not effort:
        return
    agent = dict(cfg.get("agent") or {})
    student = dict(agent.get("student") or {})
    student["effort"] = effort
    agent["student"] = student
    cfg["agent"] = agent


def _write_prior(dest: Path, req: Request, ref: Reference | None) -> None:
    files: list[Path] = list(req.prior)
    if ref is not None and ref.kernel is not None and Path(ref.kernel).is_file():
        files.append(Path(ref.kernel))
    if not files:
        return
    dest.joinpath(PRIOR_DIR).mkdir(exist_ok=True)
    names = []
    for p in files:
        target = dest / PRIOR_DIR / p.name
        if target.exists():
            tag = hashlib.sha256(str(p).encode()).hexdigest()[:8]
            target = dest / PRIOR_DIR / f"{p.stem}-{tag}{p.suffix}"
        shutil.copy2(p, target)
        names.append(target.name)
    (dest / PRIOR_DIR / "README.md").write_text(
        "# Prior implementations\n\nRead-only starting points handed to this lineage "
        "(task warmstart seeds them for the student; the loop's student may read "
        "anything under the workspace).\n\n"
        + "\n".join(f"- `{n}`" for n in names) + "\n", encoding="utf-8")


# -- board tasks -------------------------------------------------------------------------


def _problem(task: Task) -> dict:
    lb = task.leaderboard
    assert lb is not None
    path = lb.card / "eval/leaderboard/problem.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LineageError(f"cannot read the board card {path}: {e}") from None


def _board_selectors(problem: dict) -> tuple[str, str]:
    """(``--workload`` selector, ``--check-workload`` selector or "").

    Indices when the card records them, uuids otherwise."""
    group = problem.get("group") or []
    if group:
        timed = ",".join(str(g["index"]) for g in group
                         if isinstance(g.get("index"), int))
    else:
        timed = _uuid_or_index(problem.get("workload") or {})
    checks = ",".join(str(g["index"]) for g in problem.get("check_only") or []
                      if isinstance(g.get("index"), int)) if problem.get("check_only") else ""
    if not timed:
        raise LineageError("the card records no workload selector (problem.json has "
                           "neither group indices nor a workload uuid)")
    return timed, checks


def _uuid_or_index(workload: dict) -> str:
    if workload.get("uuid"):
        return str(workload["uuid"])
    if isinstance(workload.get("index"), int):
        return str(workload["index"])
    return ""


def board_record_from_card(problem: dict) -> dict | None:
    """The board's ``kernels/{id}`` record rebuilt from the card, or None.

    ``sol_problem`` accepts a saved record only when it lists the problem's every
    workload in dataset order, so this returns a record only when the card's
    timed + check-only entries cover indices ``0..n-1`` exactly.  That is the
    common shape (a card built from the whole problem); a card over a sub-range
    gets ``--no-sol-board`` and keeps its SOL block out of the new card."""
    entries: dict[int, dict] = {}
    rows: list[dict] = []
    if problem.get("group"):
        rows = list(problem["group"])
    else:
        rows = [{"index": problem.get("workload_index"),
                 "workload": problem.get("workload"),
                 "sol_board": problem.get("sol_board")}]
    rows += list(problem.get("check_only") or [])
    for row in rows:
        index = row.get("index")
        board = row.get("sol_board")
        wl = row.get("workload") or {}
        if not isinstance(index, int) or not isinstance(board, dict):
            return None
        try:
            t_b = float(board["t_b_ms"])
            t_sol = float(board["t_sol_ms"])
        except (KeyError, TypeError, ValueError):
            return None
        entries[index] = {"axes": dict(wl.get("axes") or {}),
                          "baseline_latency_ms": t_b, "sol_ms": t_sol}
    if sorted(entries) != list(range(len(entries))):
        return None
    board0 = next((r.get("sol_board") for r in rows if isinstance(r, dict)
                   and r.get("sol_board")), None)
    out = {"id": (board0 or {}).get("kernel_id"),
           "name": (problem.get("definition") or {}).get("name"),
           "workloads": [entries[i] for i in sorted(entries)]}
    if out["name"] is None:
        return None
    return out


def _kernel_name_for(language: str) -> str:
    return "kernel.cu" if language == "cuda" else "kernel.py"


def _board_language_mismatch(req: Request, problem: dict) -> bool:
    """The frozen card's kernel file must match the lineage's language: the card's
    eval reads ``problem.json``'s ``kernel_file``, so the student must edit exactly
    that file."""
    return _board_language_mismatch_type(req.language, problem)


def _assert_board_language(dest: Path, language: str) -> None:
    """The card's kernel_file must match the lineage's language (the eval reads it)."""
    problem = _problem(load_task(dest))
    if _board_language_mismatch_type(language, problem):
        raise LineageError(
            f"{dest.name}: the card's kernel_file is "
            f"{(problem.get('kernel_file') or 'kernel.py')!r} but the lineage's "
            f"language is {language!r}: the student would edit a file the card's "
            f"eval never reads")


def _board_language_mismatch_type(language: str, problem: dict) -> bool:
    return Path(str(problem.get("kernel_file") or "kernel.py")).suffix \
        != Path(_kernel_name_for(language)).suffix


def _board_task(req: Request) -> dict:
    problem = _problem(req.base)
    ref = req.reference
    need_init = (ref is not None or req.blackbox
                 or _board_language_mismatch(req, problem))
    dest = req.dest
    if not need_init:
        # Generation start from the same board problem: copy the frozen task and
        # patch it.  No dataset, no network, no re-init.
        shutil.copytree(req.base.dir, dest)
        cfg = read_task_toml(dest)
        _patch_board_toml(cfg, req)
        _write_toml(dest, cfg)
        _validate(dest)
        _assert_board_language(dest, req.language)
        _write_prior(dest, req, ref)
        return {"reference": (ref.describe() if ref is not None
                              else {"kind": "base", "variant": "", "seed": "",
                                    "language": req.base.language, "kernel": None,
                                    "run": None}),
                "handoff": "copy (same board problem)"}
    argv, prepared = _board_init_argv(req, problem, ref)
    rc, out = _run_init(argv)
    if rc != 0:
        raise LineageError(f"`forge init` for {dest.name} failed (rc {rc}): "
                           f"{out.strip()[-800:] or 'no output'}")
    cfg = read_task_toml(dest)
    _patch_board_toml(cfg, req)
    _write_toml(dest, cfg)
    _validate(dest)
    _assert_board_language(dest, req.language)
    _write_prior(dest, req, ref)
    if "reference" not in prepared:
        prepared["reference"] = {
            "kind": ref.kind if ref is not None else "base",
            "variant": ref.variant if ref is not None else "",
            "seed": ref.seed_id if ref is not None else "",
            "language": (ref.language if ref is not None else req.base.language),
            "kernel": str(ref.kernel) if ref is not None and ref.kernel else None,
            "run": str(ref.run) if ref is not None and ref.run else None}
    return prepared


def _board_init_argv(req: Request, problem: dict, ref: Reference | None
                     ) -> tuple[list[str], dict]:
    suite = str(problem.get("suite") or "")
    source = problem.get("source") or {}
    dataset = source.get("dataset_root")
    definition = source.get("problem") or (problem.get("definition") or {}).get("name")
    if not suite or not dataset or not definition:
        raise LineageError("the card lacks the provenance needed to re-init the board "
                           "problem (source.dataset_root / source.problem)")
    if not Path(str(dataset)).is_dir():
        raise LineageError(f"the dataset tree {dataset} is not on this machine; a board "
                           f"task cannot be re-initialised without it")
    timed, checks = _board_selectors(problem)
    argv = [req.dest.name, "--leaderboard", suite, "--dataset", str(dataset),
            "--definition", str(definition), "--workload", timed]
    if checks:
        argv += ["--check-workload", checks]
    record = board_record_from_card(problem)
    prepared: dict = {"board_record": None, "handoff": ""}
    if record is not None:
        req.workdir.mkdir(parents=True, exist_ok=True)
        rec_path = req.workdir / "board_record.json"
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        argv += ["--sol-board", str(rec_path)]
        prepared["board_record"] = str(rec_path)
    else:
        argv += ["--no-sol-board"]
    if req.language == "cuda":
        argv += ["--kernel-file", "kernel.cu"]
    if ref is None:
        return argv, prepared
    # A readable start means the student's first kernel IS the donor's.  A .cu
    # donor has no Python ``run`` for warmstart to validate and a cuda lineage
    # edits kernel.cu, so both go through the sealed reference instead.
    readable = (req.init_kernel_supported and not req.blackbox and _is_python(ref)
                and req.language != "cuda")
    if readable:
        argv += ["--init-kernel", str(Path(ref.kernel))]
        prepared["handoff"] = "init-kernel"
        return argv, prepared
    if ref.run is None:
        if ref.kind == "seed":
            # --from-lib with no way to hand a seed over (warmstart absent, --blackbox,
            # or a cuda lineage): the 2nd-version fallback -- start from the task's own
            # reference instead, and say so.
            prepared["handoff"] = ("seed dropped: no readable handoff available "
                                   "(warmstart absent / --sealed-ref / cuda lineage)")
            prepared["reference"] = {"kind": "base", "variant": "", "seed": ref.seed_id,
                                     "language": req.base.language, "kernel": None,
                                     "run": None}
            return argv, prepared
        raise LineageError(f"cannot hand {ref.variant or ref.seed_id} to {req.dest.name}: "
                           f"it has no run to export a sealed reference from")
    sol = _export_solution(req, ref)
    argv += ["--sealed-ref", str(sol)]
    prepared["handoff"] = "sealed-ref" if req.blackbox else "sealed-ref (warmstart absent)"
    prepared["solution"] = str(sol)
    return argv, prepared


# The population sets this from lineage.warmstart_available() (still overridable
# for tests).  Kept as a Request-independent module flag because --init-kernel is a
# property of the installed build, not of one request.
def warmstart_available() -> bool:
    """Task warmstart installs ``forgekernel.warmstart`` and its ``--init-kernel``."""
    try:
        import forgekernel.warmstart  # noqa: F401
    except Exception:  # noqa: BLE001 — absent or half-installed = unavailable
        return False
    return True


def _is_python(ref: Reference) -> bool:
    return bool(ref.kernel) and Path(str(ref.kernel)).suffix == ".py"


def _export_solution(req: Request, ref: Reference) -> Path:
    if ref.run is None:
        raise LineageError(f"cannot export a sealed reference for {ref.variant or ref.seed_id}: "
                           f"the donor has no run directory")
    req.workdir.mkdir(parents=True, exist_ok=True)
    sol = req.workdir / f"{ref.variant or ref.seed_id or 'donor'}.solution.json"
    argv = [str(ref.run), "--out", str(sol), "--name",
            f"{req.dest.name}_{ref.variant or ref.seed_id or 'donor'}"]
    rc = _call(launch.cmd_export, argv)
    if rc != 0 or not sol.is_file():
        raise LineageError(f"`forge export` of the donor run {ref.run} failed (rc {rc}): "
                           f"no solution JSON at {sol}")
    return sol


def _patch_board_toml(cfg: dict, req: Request) -> None:
    task_tbl = dict(cfg.get("task") or {})
    task_tbl["language"] = req.language
    cfg["task"] = task_tbl
    _apply_target(cfg, req.step if req.verifies_target else None)
    _apply_effort(cfg, req.effort)


def _write_toml(dest: Path, cfg: dict) -> None:
    (dest / "task.toml").write_text(dump_task_toml(cfg), encoding="utf-8")


def _validate(dest: Path) -> None:
    try:
        load_task(dest)
    except TaskErrorLike as e:  # pragma: no cover - exercised via the message
        raise LineageError(f"the generated task {dest} does not validate: {e}") from None


TaskErrorLike = task_mod.TaskError


def _call(fn, argv: list[str]) -> int:
    """Call a launch/init entry point in-process, turning argparse's SystemExit and
    any OSError into an rc + message (the caller reports both)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return int(fn(argv))
    except SystemExit as e:
        sys.stderr.write(buf.getvalue())
        return int(e.code or 0)
    except OSError as e:
        buf.write(f"{e.__class__.__name__}: {e}")
        return 2


def _run_init(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = int(launch.cmd_init(argv))
    except SystemExit as e:
        rc = int(e.code or 0)
    except OSError as e:
        buf.write(f"{e.__class__.__name__}: {e}")
        rc = 2
    return rc, buf.getvalue()


# -- local git exclude -------------------------------------------------------------------


def add_exclude_pattern(task_root: Path, dest: Path) -> None:
    """Append ``dest``'s relative path to the repository's local ``.git/info/exclude``
    (never ``.gitignore``: these directories are work products, not repository content)."""
    repo = launch.repo_root()
    try:
        rel = dest.resolve().relative_to(repo.resolve())
    except ValueError:
        try:
            rel = dest.resolve().relative_to(Path(task_root).resolve().parent)
        except ValueError:
            return
    pattern = f"/{rel.as_posix().rstrip('/')}/"
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-path",
                              "info/exclude"], capture_output=True, text=True, timeout=30)
        exclude = Path(out.stdout.strip())
        if not exclude.is_absolute():
            exclude = repo / exclude
    except (OSError, subprocess.SubprocessError):
        return
    try:
        text = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        if pattern in text.splitlines():
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with open(exclude, "a", encoding="utf-8") as fh:
            if text and not text.endswith("\n"):
                fh.write("\n")
            fh.write(f"# forge evolve lineage tasks (work products)\n{pattern}\n")
    except OSError:
        return
