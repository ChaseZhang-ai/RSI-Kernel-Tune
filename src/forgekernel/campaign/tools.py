"""The outside world a campaign touches, behind one injectable registry.

Every other task's deliverable is reached through this module and nowhere else:
`benchx` scores a candidate, `seedlib` finds / adapts / covers starting points,
`shapes` composes per-workload winners, `guard` bans licences and prints the banner,
`evolve` runs the multi-lineage loop, `board` submits and reads results, and
`targethw`'s target profile enters the loop through a context file. None of them is
required: `resolve()` probes each one once, records what it found (or why not) in
`notes`, and leaves the slot `None`. A caller that finds a slot None records the
consequence in its own output ("not exported: needs klib") instead of inventing a
substitute -- a missing tool narrows what is *exported*, never what is *claimed*.

The probe for each slot is deliberately shallow: an import of the module the other
task's file-ownership section names, and a call of the entry point that task's own
documentation names (a `main(argv)` for its `forge <cmd>`, or a documented function).
When one slips, the note says which name was looked for, so the fix is one edit here
rather than a hunt through the caller.

`BenchReading` is this module's own mirror of the styb in `forgekernel.benchx.api`
(`BenchResult`): a benchx result is converted field by field, and a fake in a test
builds one directly. Everything downstream reads this type only, so benchx can change
its internals without touching campaign.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from forgekernel.paths import repo_root

# The four context files the run context (`forgekernel/runcontext.py`, step 0) exports
# into the loop's prompts. Used as the fallback path below when that module is absent.
CONTEXT_ENV = {"target": "FK_TARGET_PROFILE_PATH", "roofline": "FK_ROOFLINE_PATH",
               "lessons": "FK_LESSONS_PATH", "prior": "FK_PRIOR_PATH"}
TARGET_PROFILE_REL = Path("target") / "B200.json"
try:                                       # the cold-start budget has one home (task benchx)
    from forgekernel.benchx.api import COLD_START_BUDGET_S as COLD_START_LIMIT_S
except ImportError:                        # benchx not installed: the documented default
    COLD_START_LIMIT_S = 100.0


@dataclass
class WorkloadReading:
    """One workload of a bench reading (benchx's `WorkloadReading`, mirrored)."""
    name: str
    axes: dict = field(default_factory=dict)
    t_ms: float | None = None
    t_b: float | None = None
    t_sol: float | None = None
    s: float | None = None
    correct: bool | None = None
    error: str = ""


@dataclass
class BenchReading:
    """The result of scoring one candidate on one card (benchx's `BenchResult`, mirrored).

    ``mean_s`` is the board's mean-over-workloads SOL score when the card carried the
    board's T_b / T_SOL; None when it carried none (the reading is then a latency
    reading, and `s` per workload is None too -- see campaign.solscore)."""
    correct: bool
    workloads: list[WorkloadReading] = field(default_factory=list)
    mean_s: float | None = None
    fast1: int = 0
    integrity: dict = field(default_factory=dict)
    job: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    cold_start_s: float | None = None
    confidence: list[str] = field(default_factory=list)
    candidate: str = ""

    @property
    def latencies(self) -> dict:
        return {w.name: float(w.t_ms) for w in self.workloads if w.t_ms is not None}

    @property
    def geomean_ms(self) -> float | None:
        from forgekernel.campaign.solscore import geomean
        return geomean(self.latencies.values())

    def cold_start_verdict(self, limit: float = COLD_START_LIMIT_S) -> tuple[str, str]:
        """``("ok"|"too-slow"|"unmeasured", why)`` for the export gate."""
        if self.cold_start_s is None:
            return "unmeasured", "the bench did not time a cold start"
        if float(self.cold_start_s) > float(limit):
            return "too-slow", (f"cold start {float(self.cold_start_s):.1f}s > {float(limit):.0f}s "
                                f"-- a board job that compiles this long is timed out at 0")
        return "ok", f"cold start {float(self.cold_start_s):.1f}s"


def bench_reading_from(obj: Any, candidate: str = "") -> BenchReading:
    """A benchx `BenchResult` (or a decoded `forge bench --json` file) as a BenchReading.

    Read field by field (dict or attribute) so a partial result or a newer benchx does
    not break the conversion; anything missing keeps its default."""
    def get(key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    wls = []
    for w in (get("workloads") or []):
        def wget(k, d=None, w=w):
            return w.get(k, d) if isinstance(w, dict) else getattr(w, k, d)
        wls.append(WorkloadReading(
            name=str(wget("name", "")), axes=dict(wget("axes") or {}),
            t_ms=_f(wget("t_ms") if wget("t_ms") is not None else wget("device_median_ms")),
            t_b=_f(wget("t_b")), t_sol=_f(wget("t_sol")), s=_f(wget("s")),
            correct=wget("correct"), error=str(wget("error") or "")))
    correct = get("correct", get("correctness", False))
    return BenchReading(
        correct=bool(correct), workloads=wls, mean_s=_f(get("mean_s")),
        fast1=int(get("fast1") or 0), integrity=dict(get("integrity") or {}),
        job=str(get("job") or ""), elapsed_s=float(get("elapsed_s") or 0.0),
        error=str(get("error") or ""), cold_start_s=_f(get("cold_start_s")),
        confidence=list(get("confidence") or []), candidate=candidate)


def _f(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Tools:
    """The injectable outside world. A None slot is a tool that is not installed (or was
    not passed in a test); `notes` says what was looked for.

    Every slot is a plain callable, so a test passes a fake and asserts on the arguments
    it received -- that is the whole seam between campaign and the rest of the harness.
    """
    root: Path = field(default_factory=repo_root)
    python: str = field(default_factory=lambda: sys.executable)
    run_log: Path | None = None
    bench: Callable[..., BenchReading] | None = None
    init_task: Callable[[list[str]], int] | None = None
    new_run: Callable[[list[str]], int] | None = None
    run_loop: Callable[[list[str]], int] | None = None
    evolve: Callable[[list[str]], int] | None = None
    lib_find: Callable[..., list[dict]] | None = None
    lib_baselines: Callable[..., Path | None] | None = None
    lib_adapt: Callable[..., Path | None] | None = None
    lib_cover: Callable[..., Path | None] | None = None
    shapes_compose: Callable[..., Path | None] | None = None
    cuda_to_inline: Callable[..., str] | None = None
    license_check: Callable[..., dict] | None = None
    notices_banner: Callable[..., str] | None = None
    export_check: Callable[..., dict] | None = None
    integrity_multiseed: Callable[..., dict] | None = None
    board_submit: Callable[..., dict] | None = None
    collection_validate: Callable[..., dict] | None = None
    notes: list[str] = field(default_factory=list)

    # -- the loop, through the repository's own commands ---------------------------------
    def _cli(self, subcommand: str, rest: list[str], env_extra: dict | None = None) -> int:
        """`forge <subcommand> <rest>` as a child process of this interpreter."""
        return self._spawn([self.python, "-m", "forgekernel", subcommand, *rest],
                           env_extra=env_extra)

    def _spawn(self, command: list[str], env_extra: dict | None = None,
               cwd: Path | None = None) -> int:
        """Run one child process: an argument list, never a shell string.

        ``executable=`` runs ``command[0]`` (this interpreter, by absolute path) while
        the argument list keeps a plain name in slot 0 -- no PATH lookup of the
        interpreter, no shell, and nothing in the argument list is ever re-parsed."""
        if not isinstance(command, (list, tuple)) or not all(isinstance(a, str) for a in command):
            raise ToolError("a child command must be a list of strings")
        env = {**os.environ, **(env_extra or {})}
        log = None
        if self.run_log is not None:
            Path(self.run_log).parent.mkdir(parents=True, exist_ok=True)
            log = open(self.run_log, "a", encoding="utf-8")
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
        try:
            proc = subprocess.run(["python", *command[1:]], executable=command[0],
                                  shell=False, cwd=str(cwd or self.root), env=env,
                                  stdout=log, stderr=subprocess.STDOUT, text=True)
        finally:
            if log is not None:
                log.close()
        return int(proc.returncode)

    def call_init(self, argv: list[str], env: dict | None = None) -> int:
        if self.init_task is None:
            raise ToolError("forge init is unavailable")
        return int(_call(self.init_task, argv, env))

    def call_new(self, argv: list[str], env: dict | None = None) -> int:
        if self.new_run is None:
            raise ToolError("forge new is unavailable")
        return int(_call(self.new_run, argv, env))

    def call_run(self, argv: list[str], env: dict | None = None) -> int:
        """Run the loop for a run directory; the context env rides the child process."""
        if self.run_loop is None:
            raise ToolError("forge run is unavailable")
        return int(_call(self.run_loop, argv, env))

    def call_evolve(self, argv: list[str], env: dict | None = None) -> int:
        if self.evolve is None:
            raise ToolError("forge evolve is unavailable (the evolve task has not landed)")
        return int(_call(self.evolve, argv, env))

    def main_of(self, module: str, argv: list[str]) -> int:
        """`main(argv)` of a `forgekernel.ext.<x>` module, in this process (their CLI is
        the documented interface; the ext discovery hook is not needed to call it)."""
        mod = importlib.import_module(module)
        try:
            return int(mod.main(argv))
        except SystemExit as e:                       # argparse's own exit
            return int(e.code or 0)

    # -- context files -------------------------------------------------------------------
    def context_env(self, files: dict[str, str | Path]) -> dict:
        """``{FK_*: path}`` for the named context files. Uses `forgekernel.runcontext`'s
        own mapping when that module exists (step 0), else the same four names here."""
        items = {k: str(v) for k, v in files.items() if v}
        local = {CONTEXT_ENV[k]: v for k, v in items.items() if k in CONTEXT_ENV}
        try:
            from forgekernel import runcontext  # type: ignore
        except ImportError:
            return local
        env_for = getattr(runcontext, "env_for", None)
        return dict(env_for(items)) if env_for is not None else local

    def target_profile(self) -> Path | None:
        """The target picture every run must carry (background §4): the rendered
        ``target/B200.rendered.md`` when task targethw's renderer is installed (the file
        the prompts describe), else ``target/B200.json`` under the repo, or
        ``FK_TARGET_PROFILE_PATH`` when the environment names one."""
        env = os.environ.get("FK_TARGET_PROFILE_PATH", "").strip()
        if env and Path(env).is_file():
            return Path(env)
        p = Path(self.root) / TARGET_PROFILE_REL
        if not p.is_file():
            return None
        try:
            from forgekernel import target_profile as tp  # type: ignore
        except ImportError:
            return p
        materialize = getattr(tp, "materialize", None)
        if callable(materialize):
            try:
                return Path(materialize(self.root))
            except Exception:      # noqa: BLE001 -- the JSON itself is still a profile
                pass
        return p

    def available(self) -> dict[str, bool]:
        return {k: getattr(self, k) is not None for k in
                ("bench", "init_task", "new_run", "run_loop", "evolve", "lib_find",
                 "lib_baselines", "lib_adapt", "lib_cover", "shapes_compose",
                 "cuda_to_inline", "license_check", "notices_banner", "export_check",
                 "integrity_multiseed", "board_submit", "collection_validate")}


class ToolError(RuntimeError):
    pass


def _call(slot: Callable, argv: list[str], env: dict | None) -> int:
    """Call an injected slot with the environment when it takes one (fakes may not)."""
    if env is None:
        return int(slot(argv))
    try:
        return int(slot(argv, env))
    except TypeError:
        return int(slot(argv))


# --- resolution ---------------------------------------------------------------------------

def resolve(root: str | Path | None = None, *, run_log: Path | None = None) -> Tools:
    """Probe every optional task's module and wire what is there.

    Network access is never needed here: the board's public T_b / T_SOL fetch happens in
    `forge init` (which accepts `--sol-board <file>` for offline use), and campaign never
    calls the board's API itself."""
    t = Tools(root=Path(root) if root else repo_root(), run_log=run_log)

    # forge init / new / run: the base repository's own commands, always present.
    t.init_task = lambda argv, env=None: t._cli("init", list(argv), env)
    t.new_run = lambda argv, env=None: t._cli("new", list(argv), env)
    t.run_loop = lambda argv, env=None: t._cli("run", list(argv), env)

    t.bench = _resolve_bench(t)
    t.evolve = _resolve_main(t, "evolve", ("forgekernel.ext.evolve", "forgekernel.evolve"))
    t.lib_find = _resolve_lib_find(t)
    t.lib_baselines = _resolve_lib_paths(t, "baselines")
    t.lib_adapt = _resolve_lib_adapt(t)
    t.lib_cover = _resolve_lib_cover(t)
    t.shapes_compose = _resolve_shapes(t)
    t.cuda_to_inline = _resolve_cuda_to_inline(t)
    t.license_check, t.notices_banner = _resolve_licence(t)
    t.export_check = _resolve_export_check(t)
    t.integrity_multiseed = _resolve_multiseed(t)
    t.board_submit = _resolve_board(t)
    t.collection_validate = _resolve_collection_validate(t)
    return t


def _resolve_main(t: Tools, name: str, modules: tuple[str, ...]) -> Callable | None:
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
        t.notes.append(f"{name}: via {mod}.main")
        return lambda argv, env=None, m=mod: t.main_of(m, argv)
    t.notes.append(f"{name}: unavailable (no {', '.join(modules)})")
    return None


def _resolve_bench(t: Tools) -> Callable | None:
    """benchx: the class in `forgekernel.benchx.api`, else `forge bench --json`."""
    try:
        api = importlib.import_module("forgekernel.benchx.api")
    except ImportError:
        api = None
    if api is not None:
        cls = getattr(api, "CardBench", None) or getattr(api, "RealBench", None)
        if cls is not None:
            try:
                impl = cls()
            except Exception as e:                              # noqa: BLE001
                t.notes.append(f"bench: {cls.__name__}() refused: {type(e).__name__}: {e}")
            else:
                t.notes.append(f"bench: via forgekernel.benchx.api.{cls.__name__}")
                return lambda task_dir, candidate, **kw: bench_reading_from(
                    impl.bench(Path(task_dir), candidate, **kw), str(candidate))
        else:
            t.notes.append("bench: forgekernel.benchx.api has no CardBench/RealBench")
    try:
        importlib.import_module("forgekernel.ext.bench")
    except ImportError:
        if api is None:
            t.notes.append("bench: unavailable (no forgekernel.benchx.api, no "
                           "forgekernel.ext.bench)")
        return None
    t.notes.append("bench: via forgekernel.ext.bench --json")
    return lambda task_dir, candidate, **kw: _bench_via_cli(t, task_dir, candidate, **kw)


def _bench_via_cli(t: Tools, task_dir, candidate, *, workloads=None, mode="dev",
                   repeat=None, **_) -> BenchReading:
    argv = [str(task_dir), "--candidate", str(candidate), "--mode", str(mode)]
    if workloads:
        argv += ["--workloads", workloads if isinstance(workloads, str)
                 else ",".join(str(w) for w in workloads)]
    if repeat:
        argv += ["--repeat", str(repeat)]
    with tempfile.TemporaryDirectory(prefix="campaign-bench-") as tmp:
        out = Path(tmp) / "bench.json"
        argv += ["--json", str(out)]
        rc = t.main_of("forgekernel.ext.bench", ["bench", *argv])
        if rc != 0 or not out.is_file():
            return BenchReading(correct=False, candidate=str(candidate),
                                error=f"forge bench exited {rc} with no JSON")
        return bench_reading_from(json.loads(out.read_text(encoding="utf-8")),
                                  str(candidate))


def _lib_available(t: Tools, what: str) -> bool:
    try:
        importlib.import_module("forgekernel.ext.lib")
    except ImportError:
        t.notes.append(f"{what}: unavailable (no forgekernel.ext.lib; the seedlib task "
                       f"has not landed)")
        return False
    return True


def _lib_json(t: Tools, argv: list[str]) -> Any:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = t.main_of("forgekernel.ext.lib", argv)
    text = buf.getvalue().strip()
    if rc != 0:
        raise ToolError(f"forge lib {' '.join(argv)} exited {rc}")
    start = text.find("[") if "[" in text else text.find("{")
    if start < 0:
        return []
    return json.loads(text[start:])


def _resolve_lib_find(t: Tools) -> Callable | None:
    if not _lib_available(t, "lib_find"):
        return None
    t.notes.append("lib_find: via forge lib find --json")
    return lambda problem, k=5: list(_lib_json(t, ["find", str(problem), "--k", str(k),
                                                   "--json"]))


def _resolve_lib_paths(t: Tools, sub: str) -> Callable | None:
    """`forge lib baselines` / `cover` write a directory of candidates; return its path."""
    if not _lib_available(t, f"lib_{sub}"):
        return None
    t.notes.append(f"lib_{sub}: via forge lib {sub}")

    def call(problem_or_dataset, out: Path, **kw) -> Path | None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        argv = [sub, str(problem_or_dataset), "--out", str(out)]
        for k, v in kw.items():
            if v is not None:
                argv += [f"--{k.replace('_', '-')}", str(v)]
        rc = t.main_of("forgekernel.ext.lib", argv)
        return out if rc == 0 else None
    return call


def _resolve_lib_adapt(t: Tools) -> Callable | None:
    if not _lib_available(t, "lib_adapt"):
        return None
    t.notes.append("lib_adapt: via forge lib adapt")

    def call(problem: str, seed: str, out_dir: Path, dataset: str | None = None,
             rounds: int = 3, **_) -> Path | None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = ["adapt", str(problem), "--seed", str(seed), "--out", str(out_dir),
                "--rounds", str(rounds)]
        if dataset:
            argv += ["--dataset", str(dataset)]
        rc = t.main_of("forgekernel.ext.lib", argv)
        if rc != 0:
            return None
        hits = sorted(out_dir.rglob("submission.py")) or sorted(out_dir.glob("*.py"))
        return hits[0] if hits else None
    return call


def _resolve_lib_cover(t: Tools) -> Callable | None:
    """`forge lib cover --dataset … --out …`: the per-problem reference fallback."""
    if not _lib_available(t, "lib_cover"):
        return None
    t.notes.append("lib_cover: via forge lib cover")

    def call(dataset: str, out_dir: Path, **_) -> Path | None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = t.main_of("forgekernel.ext.lib", ["cover", "--dataset", str(dataset),
                                               "--out", str(out_dir)])
        return out_dir if rc == 0 else None
    return call


def _resolve_shapes(t: Tools) -> Callable | None:
    """`forge shapes compose` (its documented CLI). Returns the submission.py it wrote."""
    try:
        importlib.import_module("forgekernel.ext.shapes")
    except ImportError:
        t.notes.append("shapes_compose: unavailable (no forgekernel.ext.shapes)")
        return None
    t.notes.append("shapes_compose: via forge shapes compose")

    def call(task_dir: Path, also: list[Path], out: Path, extra: list[str] | None = None,
             **_) -> Path | None:
        out = Path(out)
        argv = ["compose", str(task_dir), "--out", str(out)]
        for a in also:
            argv += ["--also", str(a)]
        argv += list(extra or [])
        t.main_of("forgekernel.ext.shapes", argv)
        for cand in (out, Path(task_dir) / "shapes" / "submission.py",
                     Path(task_dir) / "shapes" / "compose" / "submission.py"):
            if cand.is_file() and cand.suffix == ".py":
                return cand
        if out.is_dir():
            hits = sorted(out.rglob("submission.py"))
            if hits:
                return hits[0]
        return None
    return call


def _resolve_cuda_to_inline(t: Tools) -> Callable | None:
    """`forgekernel.klib.normalize.cuda_to_inline` (seedlib): a .cu (+ headers) -> the
    source of a single-file submission.py."""
    try:
        mod = importlib.import_module("forgekernel.klib.normalize")
    except ImportError:
        t.notes.append("cuda_to_inline: unavailable (no forgekernel.klib.normalize; the "
                       "seedlib task has not landed)")
        return None
    fn = getattr(mod, "cuda_to_inline", None)
    if fn is None:
        t.notes.append("cuda_to_inline: forgekernel.klib.normalize has no cuda_to_inline")
        return None
    t.notes.append("cuda_to_inline: via forgekernel.klib.normalize.cuda_to_inline")

    def call(cu: Path, *, headers: list[Path] | None = None, entry: str | None = None,
             flags: list[str] | None = None, **_) -> str:
        kw: dict = {}
        if headers:
            kw["headers"] = [str(h) for h in headers]
        if entry:
            kw["entry"] = entry
        if flags:
            kw["flags"] = list(flags)
        for attempt in (kw, {k: v for k, v in kw.items() if k in ("entry",)}):
            try:
                text = fn(str(cu), **attempt)
            except TypeError:
                continue
            if isinstance(text, str) and text.strip():
                return text
            raise ToolError("cuda_to_inline returned no source")
        return str(fn(str(cu)))                    # last resort: the plain signature
    return call


def _resolve_licence(t: Tools) -> tuple[Callable | None, Callable | None]:
    try:
        mod = importlib.import_module("forgekernel.integrity")
    except ImportError:
        t.notes.append("license_check: unavailable (no forgekernel.integrity; the guard "
                       "task has not landed)")
        return None, None
    check = getattr(mod, "license_check", None)
    notices = getattr(mod, "notices", None)
    banner = getattr(notices, "banner", None) if notices is not None else None
    if check is None:
        t.notes.append("license_check: forgekernel.integrity has no license_check")
    else:
        t.notes.append("license_check: via forgekernel.integrity.license_check")
    if banner is None:
        t.notes.append("notices_banner: unavailable (no forgekernel.integrity.notices.banner)")
    else:
        t.notes.append("notices_banner: via forgekernel.integrity.notices.banner")
    return check, banner


def _resolve_export_check(t: Tools) -> Callable | None:
    try:
        mod = importlib.import_module("forgekernel.integrity")
    except ImportError:
        return None
    for name in ("export_check", "submission_check", "check_submission"):
        fn = getattr(mod, name, None)
        if fn is not None:
            t.notes.append(f"export_check: via forgekernel.integrity.{name}")
            return fn
    t.notes.append("export_check: forgekernel.integrity has no export_check")
    return None


def _resolve_multiseed(t: Tools) -> Callable | None:
    """`forgekernel.integrity.multiseed.check` (guard): N fresh-input draws before export.

    ``check(card_dir, candidate)`` runs the card once per seed and answers
    ``{"status": "pass" | "fail" | "unavailable", ...}``; without a GPU it is
    ``unavailable`` and the export is not blocked."""
    try:
        mod = importlib.import_module("forgekernel.integrity.multiseed")
    except ImportError:
        t.notes.append("integrity_multiseed: unavailable (no forgekernel.integrity.multiseed; "
                       "the guard task has not landed)")
        return None
    fn = getattr(mod, "check", None)
    if fn is None:
        t.notes.append("integrity_multiseed: forgekernel.integrity.multiseed has no check")
        return None
    t.notes.append("integrity_multiseed: via forgekernel.integrity.multiseed.check")
    return fn


def _resolve_board(t: Tools) -> Callable | None:
    try:
        importlib.import_module("forgekernel.ext.board")
    except ImportError:
        t.notes.append("board_submit: unavailable (no forgekernel.ext.board; the board "
                       "task has not landed)")
        return None
    t.notes.append("board_submit: via forge board submit")

    def call(problem: str, submission: Path, wait: bool = True, **_) -> dict:
        buf = io.StringIO()
        argv = ["submit", str(problem), str(submission)] + (["--wait"] if wait else [])
        with contextlib.redirect_stdout(buf):
            rc = t.main_of("forgekernel.ext.board", argv)
        return {"rc": rc, "stdout": buf.getvalue()[-4000:]}
    return call


def _resolve_collection_validate(t: Tools) -> Callable | None:
    """The board task's collection self-check (`forgekernel.board.validate`, pure functions)."""
    try:
        mod = importlib.import_module("forgekernel.board.validate")
    except ImportError:
        t.notes.append("collection_validate: unavailable (no forgekernel.board.validate)")
        return None
    for name in ("validate_collection", "check_collection", "validate_archive",
                 "collection_problems"):
        fn = getattr(mod, name, None)
        if fn is not None:
            t.notes.append(f"collection_validate: via forgekernel.board.validate.{name}")
            return fn
    t.notes.append("collection_validate: forgekernel.board.validate has no entry point")
    return None
