"""The bench runner: one candidate, one card, one temporary run, the loop's own job path.

The point of reusing the loop's machinery (``taskcard`` for the trees, ``RealBackend``
for the GPU job) instead of shelling out to the card by hand is that the reading is the
*same reading*: same tree, same ``gpu_entry.sh``, same ``eval/bench.py``, same clocks,
same eval JSON -- so a number from ``forge bench`` and a number from an iteration of a
real run are comparable.

What a bench does, in order:

1. a temporary run (``.artifacts/benchx_<id>/``, the ``runs`` layout -- deleted when the
   bench ends unless ``--keep``);
2. the card installed into it (``taskcard.install_board_trees``) and a workspace cloned
   from it (``taskcard.clone_workspace``);
3. the candidate written into the workspace's kernel file, committed and pushed to the
   card's branch (the job payload is a ``git archive`` of that branch);
4. a cold-start probe job (``cmd:python3 benchx_coldstart.py``), then the eval job(s)
   (``RealBackend.submit_gpu(branch, "all")``, ``repeat`` of them);
5. the logs read by :mod:`forgekernel.benchx.reader`, the soft target check attached.

Nothing here uploads anything, reads the board, or starts an agent.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from forgekernel import runs as runs_mod
from forgekernel import taskcard
from forgekernel.benchx import reader
from forgekernel.benchx.api import BenchError, BenchResult
from forgekernel.paths import OPERATORS_ENV, repo_root
from forgekernel.task import Leaderboard, TaskError, load_task

__all__ = ["BENCH_RUN_PREFIX", "COLD_PROBE_FILE", "COLD_PROBE_MODE", "BareCardTask",
           "BenchError", "Candidate", "CardBench", "commit_candidate", "confidence_hints",
           "kernel_file_of", "load_board", "new_bench_run", "read_problem",
           "select_workloads", "target_compile_check", "trim_card_problem",
           "write_candidate"]

BENCH_RUN_PREFIX = "benchx_"
COLD_PROBE_FILE = "benchx_coldstart.py"
COLD_PROBE_MODE = f"cmd:python3 {COLD_PROBE_FILE}"
"""One job's mode string; ``gpu_entry.sh cmd`` runs it in the job tree (same env, same
GPU, same clocks as the eval job -- only the process is new)."""

COLD_PROBE_SOURCE = '''"""benchx's cold-start probe (written by `forge bench`; the temp run removes it).

"How long from this process starting to the card's first successful run() return" --
every compile included: an inline-CUDA load_inline, a Triton / CuTe DSL JIT, a
torch.compile. The board re-compiles on every evaluation, so this is the number that
decides whether the candidate finishes the board's job at all.

It imports the implementation exactly the way eval/bench.py does (`leaderboard._impl`),
draws the card's first input set and makes one call; nothing is judged or timed.
"""
import time

T0 = time.perf_counter()          # before torch, before the candidate: import included

import torch  # noqa: E402


def main() -> int:
    import leaderboard as lb

    problem = lb.load_problem()
    impl, dps = lb._impl(problem, "kernel")
    inp = lb.first_inputs(problem, "cuda")
    lb.call_impl(problem, impl, inp, dps, "cuda", no_grad=True)
    cold = time.perf_counter() - T0
    print(f"[benchx] COLD_START_S {cold:.3f} "
          f"(card {problem.get('definition', {}).get('name')}, "
          f"kernel {problem.get('kernel_file')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# -- the task behind a card directory ---------------------------------------------------------


@dataclass(frozen=True)
class BareCardTask:
    """A card directory with no ``task.toml``: the fields ``taskcard`` and the submit
    path actually read. ``forge bench`` accepts ``tasks/<op>`` (a real task) as well."""

    name: str
    dir: Path
    language: str
    arch: str
    leaderboard: Leaderboard


def _safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "")).strip("-.")
    return name or "card"


def read_problem(card: Path) -> dict:
    p = Path(card) / taskcard.PROBLEM
    if not p.is_file():
        raise BenchError(f"{p} is missing: {card} is not a frozen board card")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BenchError(f"{p} does not parse: {e}") from None


def load_board(card_dir: "str | Path"):
    """(the task (or :class:`BareCardTask`), its problem); BenchError otherwise."""
    d = Path(card_dir).expanduser()
    if not d.is_dir():
        raise BenchError(f"{d} is not a directory")
    if (d / "task.toml").is_file():
        try:
            task = load_task(d)
        except TaskError as e:
            raise BenchError(f"{d} does not load as a task: {e}") from None
        if task.leaderboard is None:
            raise BenchError(
                f"{d} is not a leaderboard task (no [leaderboard] table): `forge bench` "
                f"benches board problems only -- their card carries the board's own "
                f"judge, inputs and clock")
        return task, read_problem(task.leaderboard.card)
    problem = read_problem(d)
    name = _safe_name(problem.get("definition", {}).get("name") or d.name)
    language = "cuda" if str(problem.get("kernel_file", "")).endswith(".cu") else "cutedsl"
    return (BareCardTask(name=name, dir=d, language=language, arch="sm100",
                         leaderboard=Leaderboard(suite=str(problem.get("suite")),
                                                 card=d, ref=None)),
            problem)


def kernel_file_of(task, problem: dict) -> str:
    return str(problem.get("kernel_file") or taskcard.kernel_file(task))


# -- the temporary run ------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _head_commit() -> str | None:
    p = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() or None if p.returncode == 0 else None


def new_bench_run(task, *, root: Path | None = None, machine=None, now=time.time):
    """``<runs root>/benchx_<stamp>-<hex>/`` with the task snapshotted, in the ``runs``
    layout (``run.json`` + ``task/`` + ``operators/``), so ``forge status`` sees it.

    ``runs.new_run`` names its directory ``<task>_<id>``; bench runs are named
    ``benchx_*`` so a reader can tell a bench from a loop run at a glance, which is why
    this writes ``new_run``'s record itself instead of calling it.
    """
    root = Path(root) if root is not None else runs_mod.runs_root()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%y%m%d-%H%M%S", time.localtime(now()))
    for _ in range(8):
        rid = f"{stamp}-{secrets.token_hex(2)}"
        d = root / f"{BENCH_RUN_PREFIX}{rid}"
        try:
            d.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise BenchError(f"could not allocate a bench run directory under {root}")
    shutil.copytree(task.dir, d / runs_mod.SNAPSHOT_DIR,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc",
                                                  ".DS_Store"))
    rec = {
        "task": task.name,
        "task_source": str(task.dir),
        "id": rid,
        "created": _now(),
        "updated": _now(),
        "snapshot_sha256": runs_mod.snapshot_digest(d / runs_mod.SNAPSHOT_DIR),
        "forgekernel_commit": _head_commit(),
        "machine": ({"delivery": machine.delivery, "gpu": machine.gpu,
                     "host": machine.remote_host or None} if machine is not None else None),
        "status": runs_mod.STATUS_CREATED,
        "kind": "benchx",
    }
    _write_json(d / runs_mod.RUN_FILE, rec)
    return runs_mod.Run(dir=d, task_name=task.name, id=rid)


class _OpEnv:
    """``FORGEKERNEL_OPERATORS_DIR`` for the submit path (``RealBackend``'s ``_forge``
    and the detached runner both inherit the process environment)."""

    def __init__(self, operators: Path):
        self.operators = Path(operators)
        self._old: str | None = None

    def __enter__(self):
        import os

        self._old = os.environ.get(OPERATORS_ENV)
        os.environ[OPERATORS_ENV] = str(self.operators)
        return self

    def __exit__(self, *exc):
        import os

        if self._old is None:
            os.environ.pop(OPERATORS_ENV, None)
        else:
            os.environ[OPERATORS_ENV] = self._old
        return False


# -- the candidate ----------------------------------------------------------------------------


@dataclass
class Candidate:
    """What was written into the workspace, for the record."""

    entry: str
    files: list[str]
    dps_adapter: str = ""          # "" | "value-returning-to-dps" | "dps-to-value-returning"
    source: str = ""               # the entry file's text (for the target check)

    def to_dict(self) -> dict:
        return {"entry": self.entry, "files": list(self.files),
                "dps_adapter": self.dps_adapter}


def candidate_sources(candidate: "Path | dict") -> tuple[str, dict[str, str], bool | None]:
    """(the entry file's relative path, {relative path: text}, the solution's dps or None).

    A ``.py`` file (a path, or a string path): one file. An official solution JSON (a
    dict, or a path to one): ``spec.entry_point``'s file plus every file in ``sources``.
    """
    if isinstance(candidate, dict):
        entry = str((candidate.get("spec") or {}).get("entry_point") or "kernel.py")
        entry = entry.split("::", 1)[0]
        files: dict[str, str] = {}
        for src in candidate.get("sources") or ():
            if isinstance(src, dict) and src.get("path") is not None:
                files[str(src["path"])] = str(src.get("content") or "")
        if entry not in files:
            text = candidate.get("source") or candidate.get("code")
            if text is None:
                raise BenchError(f"the solution JSON declares entry_point {entry!r} but "
                                 f"carries no sources for it")
            files[entry] = str(text)
        dps = (candidate.get("spec") or {}).get("destination_passing_style")
        return entry, files, (bool(dps) if dps is not None else None)
    p = Path(candidate).expanduser()
    if not p.is_file():
        raise BenchError(f"the candidate {p} is not a file")
    if p.suffix == ".json":
        try:
            return candidate_sources(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            raise BenchError(f"the candidate {p} does not parse as JSON: {e}") from None
    if p.suffix != ".py":
        raise BenchError(f"the candidate {p} is neither a .py file nor a solution JSON")
    return p.name, {p.name: p.read_text(encoding="utf-8")}, None


_VALUE_TO_DPS = '''"""benchx's calling-convention adapter (written by `forge bench`).

The solution returns its outputs; this card's protocol writes them into preallocated
outputs. The adapter copies one into the other, in the definition's output order --
the computation and its precision are the solution's, unchanged.
"""
import torch

_OUTPUTS = {outputs!r}
_ENTRY = {entry!r}

from {module} import run as _run  # noqa: E402


def _values(result):
    if isinstance(result, dict):
        return [result[name] for name in _OUTPUTS]
    if isinstance(result, (tuple, list)):
        return list(result)
    return [result]


def run(*args):
    ins, outs = list(args[:-len(_OUTPUTS)]), list(args[-len(_OUTPUTS):])
    for dst, value in zip(outs, _values(_run(*ins))):
        if isinstance(value, torch.Tensor):
            dst.copy_(value.reshape(dst.shape))
        else:
            dst.copy_(torch.as_tensor(value, dtype=dst.dtype, device=dst.device))
    return outs
'''

_DPS_TO_VALUE = '''"""benchx's calling-convention adapter (written by `forge bench`).

The solution writes into preallocated outputs; this card's protocol wants the outputs
returned. The adapter allocates them the card's own way (``leaderboard.allocate_outputs``)
and hands them to the solution -- the computation and its precision are the solution's.
"""
_ENTRY = {entry!r}

import leaderboard as _lb  # noqa: E402

from {module} import run as _run  # noqa: E402

_PROBLEM = _lb.load_problem()
_OUTS = None


def run(*args):
    global _OUTS
    if _OUTS is None:
        _OUTS = _lb.allocate_outputs(_PROBLEM, "cuda")
    _run(*args, *_OUTS)
    return _OUTS
'''


def write_candidate(ws: Path, candidate: "Path | dict", *, kernel_file: str,
                    problem: dict) -> Candidate:
    """Write the candidate into the workspace at ``kernel_file`` (plus its other files).

    An official solution JSON declares its own calling convention
    (``spec.destination_passing_style``); the card's is ``problem["dps"]``. When they
    differ, a thin adapter is written at ``kernel_file`` and the solution's entry file
    is kept beside it -- the protocol is the card's, the computation stays the
    solution's (recorded in ``Candidate.dps_adapter``).
    """
    entry, files, dps = candidate_sources(candidate)
    card_dps = bool(problem.get("dps", True))
    out_files: list[str] = []
    adapter = ""
    entry_text = files.get(entry, "")
    if dps is None or dps == card_dps:
        (ws / kernel_file).write_text(entry_text, encoding="utf-8")
        out_files.append(kernel_file)
    else:
        module = "_benchx_entry"
        (ws / f"{module}.py").write_text(entry_text, encoding="utf-8")
        out_files.append(f"{module}.py")
        if dps:
            adapter = "dps-to-value-returning"
            text = _DPS_TO_VALUE.format(entry=entry, module=module)
        else:
            adapter = "value-returning-to-dps"
            outputs = [str(n) for n in (problem.get("definition", {}).get("outputs") or {})]
            text = _VALUE_TO_DPS.format(outputs=outputs, entry=entry, module=module)
        (ws / kernel_file).write_text(text, encoding="utf-8")
        out_files.append(kernel_file)
    for rel, text in files.items():
        if rel == entry:
            continue
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        out_files.append(rel)
    return Candidate(entry=kernel_file, files=sorted(set(out_files)),
                     dps_adapter=adapter, source=entry_text)


def select_workloads(spec: "str | None", names: list[str]) -> list[str] | None:
    """``None`` for "all"; else the names an ``0,3,5-9`` / name list selects."""
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return None
    wanted: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise BenchError(f"--workloads {spec!r}: {part} is an empty range")
            for i in range(lo, hi + 1):
                if not 0 <= i < len(names):
                    raise BenchError(f"--workloads {spec!r}: no workload {i} "
                                     f"(the card has {len(names)})")
                wanted.append(names[i])
            continue
        if part.isdigit():
            i = int(part)
            if not 0 <= i < len(names):
                raise BenchError(f"--workloads {spec!r}: no workload {i} "
                                 f"(the card has {len(names)})")
            wanted.append(names[i])
            continue
        if part not in names:
            raise BenchError(f"--workloads {spec!r}: {part!r} names no workload of this "
                             f"card ({', '.join(names)})")
        wanted.append(part)
    if not wanted:
        raise BenchError(f"--workloads {spec!r} selected nothing")
    return list(dict.fromkeys(wanted))


def trim_card_problem(tree: Path, problem: dict, wanted: list[str]) -> dict:
    """Restrict ``tree``'s ``eval/leaderboard/problem.json`` to ``wanted`` and return the
    new problem (``tree`` is the workspace, so the trim rides the pushed commit).

    The card's own ``apply_workload_scope`` does the same transformation for its meta
    tools; doing it statically here (no ``FK_LB_WORKLOAD_SCOPE``) keeps the eval JSON a
    real verdict over the kept workloads -- a scoped *job* would make ``parse_eval``
    refuse the gate. The subset is recorded in ``integrity["workload_subset"]``.
    """
    if not wanted:
        return problem
    group = problem.get("group") or []
    if len(group) <= 1:
        names = {str((group[0] if group else {}).get("name") or ""),
                 str((problem.get("workload") or {}).get("uuid") or "")}
        if not set(wanted) & names:
            raise BenchError("this card has one workload; only --workloads all is valid")
        return problem
    kept = [g for g in group if str(g.get("name")) in wanted]
    if len(kept) == len(group):
        return problem
    out = dict(problem)
    out["group"] = kept
    out["workload"] = kept[0]["workload"]
    out.pop("sol_board", None)
    if len(kept) == 1 and kept[0].get("sol_board"):
        out["sol_board"] = kept[0]["sol_board"]
    out["meta_workloads"] = [str(g.get("name")) for g in kept]
    (Path(tree) / taskcard.PROBLEM).write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(tree), *args], check=True,
                          capture_output=True, text=True, env=taskcard._git_env())


def commit_candidate(ws: Path, message: str, branch: str = taskcard.BRANCH) -> str:
    """Commit the workspace's changes and push them into the card's branch (the job
    payload is the card's ``git archive`` of that branch); returns the new sha."""
    taskcard._commit(ws, message)
    _git(ws, "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}")
    return _git(ws, "rev-parse", "HEAD").stdout.strip()


# -- the target check (a soft dependency on the targethw task) ---------------------------------

_FEATURES = (
    ("inline_cuda", re.compile(r"cpp_extension|load_inline|CUDAExtension|nvcc")),
    ("triton", re.compile(r"^\s*(import|from)\s+triton|@triton\.jit", re.M)),
    ("cutedsl", re.compile(r"\bcutlass\.cute\b|\bimport\s+cutlass\b|cutlass\.dsl")),
    ("torch_compile", re.compile(r"torch\.compile\(")),
)


def _optional_rules():
    """The target-profile module, when the targthw task delivered one, and why not."""
    import importlib

    for name in ("forgekernel.targethw.rules", "forgekernel.targethw"):
        try:
            return importlib.import_module(name), ""
        except ModuleNotFoundError as e:
            missing = str(getattr(e, "name", "") or "")
            if missing == name or name.startswith(missing + "."):
                continue          # the module itself (or a parent) is absent
            return None, f"{name} is present but does not import: {e}"
        except Exception as e:  # noqa: BLE001 -- a broken optional module is a note
            return None, f"{name} is present but does not import: {e!r}"
    return None, ""


def target_compile_check(source: str, *, kernel_file: str, language: str, arch: str,
                         problem: dict) -> dict:
    """The export-time target check for one candidate.

    Builtin (always): a syntax compile and the feature flags a target check must cover
    (inline CUDA needs the board's ``-gencode arch=compute_100a,code=sm_100a``; Triton /
    CuTe DSL compile at run time). Optional: when the targthw task's module is
    importable, its ``target_compile(source, kernel_file=, language=, arch=)`` result is
    attached under ``external`` -- benchx never depends on it.
    """
    out: dict = {"kernel_file": kernel_file, "language": language, "arch": arch,
                 "features": [name for name, pat in _FEATURES if pat.search(source or "")],
                 "engine": "builtin-syntax"}
    try:
        compile(source or "", kernel_file, "exec")
        out["status"] = "ok"
    except SyntaxError as e:
        out["status"] = "failed"
        out["error"] = f"{kernel_file}:{e.lineno}: {e.msg}"
    mod, why = _optional_rules()
    fn = getattr(mod, "target_compile", None) if mod is not None else None
    if fn is None:
        out["external"] = "unavailable"
        if why:
            out["external_note"] = why
    else:
        try:
            out["external"] = fn(source or "", kernel_file=kernel_file,
                                 language=language, arch=arch)
        except Exception as e:  # noqa: BLE001 -- the optional check never fails a bench
            out["external"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    return out


def confidence_hints(source: str, *, kernel_file: str, language: str, arch: str,
                     integrity: dict) -> list[str]:
    """``["dev-only"]`` plus whatever the target profile module says about the source."""
    hints = ["dev-only"]
    mod, why = _optional_rules()
    fn = getattr(mod, "confidence_hints", None) if mod is not None else None
    if fn is None:
        if why:
            integrity["confidence_hints_note"] = why
        return hints
    try:
        extra = fn(source or "", kernel_file=kernel_file, language=language, arch=arch)
        for hint in extra or ():
            text = str(hint)
            if text and text not in hints:
                hints.append(text)
    except Exception as e:  # noqa: BLE001
        integrity["confidence_hints_note"] = f"{type(e).__name__}: {e}"
    return hints


# -- the bench --------------------------------------------------------------------------------


class CardBench:
    """The real :class:`forgekernel.benchx.api.Bench`: a card, a candidate, GPU job(s)."""

    def __init__(self, *, repeat: int = 1, keep: bool = False, machine=None,
                 poll_s: float = 3.0, log=None):
        self.repeat = max(1, int(repeat))
        self.keep = keep
        self._machine = machine
        self.poll_s = poll_s
        self._log = log or (lambda msg: None)

    # -- public ------------------------------------------------------------------------

    def bench(self, card_dir: "Path", candidate: "Path | dict", *, workloads=None,
              mode: str = "dev") -> BenchResult:
        if mode not in ("dev", "target"):
            raise BenchError(f"mode must be 'dev' or 'target', got {mode!r}")
        task, problem = load_board(card_dir)
        return self.bench_task(task, problem, candidate, workloads=workloads, mode=mode)

    def bench_task(self, task, problem: dict, candidate: "Path | dict", *,
                   workloads=None, mode: str = "dev") -> BenchResult:
        """The same bench, with the task and its problem already loaded (``forge bench``
        loads them once for its own validation and messages)."""
        if mode not in ("dev", "target"):
            raise BenchError(f"mode must be 'dev' or 'target', got {mode!r}")
        names = reader.card_workload_names(problem)
        wanted = self._selection(workloads, names)
        if wanted is not None and set(wanted) == set(names):
            wanted = None                    # the whole card: nothing to trim or record
        kernel_file = kernel_file_of(task, problem)
        if mode == "target":
            return self._check_target(task, problem, candidate, kernel_file)
        return self._bench_dev(task, problem, candidate, wanted=wanted,
                               kernel_file=kernel_file)

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _selection(workloads, names: list[str]) -> list[str] | None:
        if workloads is None:
            return None
        if isinstance(workloads, str):
            return select_workloads(workloads, names)
        wanted = [str(w) for w in workloads]
        unknown = [w for w in wanted if w not in names]
        if unknown:
            raise BenchError(f"unknown workload(s) {unknown}; the card has {names}")
        return wanted or None

    def _check_target(self, task, problem: dict, candidate, kernel_file: str) -> BenchResult:
        entry, files, _ = candidate_sources(candidate)
        source = files.get(entry, "")
        integrity: dict = {"mode": "target",
                           "verdict": "not run (target mode: only the compile check)"}
        check = target_compile_check(source, kernel_file=kernel_file,
                                     language=task.language, arch=task.arch,
                                     problem=problem)
        integrity["target_compile"] = check
        return BenchResult(correct=False, workloads=[], mean_s=None, fast1=0,
                           integrity=integrity, job="", elapsed_s=0.0, error="",
                           cold_start_s=None,
                           confidence=confidence_hints(source, kernel_file=kernel_file,
                                                       language=task.language,
                                                       arch=task.arch,
                                                       integrity=integrity))

    def _machine_or_load(self):
        if self._machine is not None:
            return self._machine
        from forgekernel.localenv import load_machine

        return load_machine()

    def _bench_dev(self, task, problem: dict, candidate, *, wanted, kernel_file: str):
        from forgekernel.loop import cli as loop_cli

        started = time.monotonic()
        machine = self._machine_or_load()
        run = new_bench_run(task, machine=machine)
        logs: list[str] = []
        jobs: list[str] = []
        cold_log: str | None = None
        cold_error = ""
        integrity: dict = {}
        written: Candidate | None = None
        try:
            run.update(status=runs_mod.STATUS_RUNNING, stage="installing the card")
            taskcard.install_board_trees(run, task, machine)
            op_dir = run.operators_dir / task.name
            self._log(f"bench run {run.dir}")
            run.update(stage="cloning the workspace")
            ws = taskcard.clone_workspace(op_dir, run.ws)
            run.update(stage="writing the candidate")
            written = write_candidate(ws, candidate, kernel_file=kernel_file,
                                      problem=problem)
            integrity["candidate"] = written.to_dict()
            # The trim goes into the *workspace* (``ws``), not the card: it must ride the
            # pushed commit, or the push would meet a dirty card working tree
            # (receive.denyCurrentBranch=updateInstead refuses those).
            eff_problem = (trim_card_problem(ws, problem, wanted)
                           if wanted is not None else problem)
            (ws / COLD_PROBE_FILE).write_text(COLD_PROBE_SOURCE, encoding="utf-8")
            sha = commit_candidate(ws, f"benchx: {kernel_file} "
                                       f"({written.dps_adapter or 'as declared'})")
            integrity["commit"] = sha
            run.update(stage="the eval jobs", commit=sha)
            be = loop_cli.RealBackend(task.name, op_dir, None, "benchx",
                                      "benchx@localhost", poll_s=self.poll_s)
            with _OpEnv(run.operators_dir):
                cold_log, cold_error = self._cold_start(be)
                for i in range(self.repeat):
                    self._log(f"eval job {i + 1}/{self.repeat}")
                    logs.append(be.submit_gpu(taskcard.BRANCH, "all"))
                    jobs.append(str(getattr(be, "last_job_id", "") or ""))
            run.update(stage="reading the logs", jobs=jobs)
        except BenchError:
            self._finish(run, None)
            raise
        except Exception as e:  # noqa: BLE001 -- the job path failed; say why, in full
            integrity["failure_kind"] = type(e).__name__
            cold = reader.parse_cold_start(cold_log)
            integrity.update(reader.cold_start_state(cold, probe_ran=cold_log is not None,
                                                     error=cold_error))
            result = BenchResult(
                correct=False, workloads=[], mean_s=None, fast1=0,
                integrity=integrity, job=",".join(j for j in jobs if j),
                elapsed_s=round(time.monotonic() - started, 3),
                error=f"{type(e).__name__}: {e}",
                cold_start_s=cold, confidence=["dev-only"])
            self._finish(run, result)
            return result
        result = reader.build_result(
            logs, problem=eff_problem, jobs=jobs, cold_log=cold_log,
            cold_error=cold_error, elapsed_s=round(time.monotonic() - started, 3),
            extra_integrity=integrity, workload_subset=wanted,
            card_workloads=len(reader.card_workload_names(problem)))
        entry_source = written.source if written is not None else ""
        result.integrity["target_compile"] = target_compile_check(
            entry_source, kernel_file=kernel_file, language=task.language,
            arch=task.arch, problem=problem)
        result.confidence = confidence_hints(entry_source, kernel_file=kernel_file,
                                             language=task.language, arch=task.arch,
                                             integrity=result.integrity)
        self._finish(run, result)
        return result

    def _cold_start(self, be) -> tuple[str | None, str]:
        """The cold-start probe job, before the eval jobs: it compiles the candidate, so
        running it first is what makes its number the board's number (a fresh process
        with cold caches) and leaves the eval job's caches warm."""
        self._log(f"cold-start probe job ({COLD_PROBE_MODE})")
        try:
            return be.submit_gpu(taskcard.BRANCH, COLD_PROBE_MODE), ""
        except Exception as e:  # noqa: BLE001 -- the probe never fails the bench
            return None, f"{type(e).__name__}: {e}"[:600]

    def _finish(self, run, result: BenchResult | None) -> None:
        try:
            if result is not None:
                run.update(status=(runs_mod.STATUS_DONE if result.correct
                                   else runs_mod.STATUS_FAILED),
                           error=(result.error or None))
            else:
                run.update(status=runs_mod.STATUS_FAILED)
        except Exception:  # noqa: BLE001 -- bookkeeping must not mask the bench
            pass
        if self.keep:
            self._log(f"--keep: the bench run stays at {run.dir}")
        else:
            shutil.rmtree(run.dir, ignore_errors=True)
