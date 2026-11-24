"""From a run's versions to one `submission.py` a collection submission can take.

A track submission is a zip of ``<kernel_name>/submission.py`` files, each a **Python**
source with a top-level ``run`` (background §3.1) -- so a finished kernel has to come out
of the harness as one file that takes the board's positional arguments and writes the
pre-allocated outputs. This module does that, and the last two things the task file
demands of it:

* **the outer try / except** (task §5.6): the chosen version is called first; any
  exception falls back to the definition's own eager PyTorch reference, which is pasted
  into the file as ordinary code (locals isolated by renaming, never `eval`/`exec`), so an
  edge shape a specialised kernel mishandles costs that workload's score instead of the
  whole problem's -- a reference always passes correctness. The fallback is the
  *definition's* reference, not a re-implementation.
* **the cold-start gate** (supplement 4): a candidate whose measured cold start exceeds
  100 s is not exported unless `--allow-slow-cold-start`; a candidate whose cold start was
  never measured is exported with ``cold_start: "unmeasured"`` in the manifest. On the
  board every evaluation recompiles, so a solution that compiles too long scores 0.

Three kinds of candidate are accepted, and each records what it is:

* ``python`` -- a single-file kernel (``kernel.py``) from a `forge run`; inlined by
  renaming its top-level ``run`` to ``_fk_candidate_run`` (an AST edit, so decorators keep
  seeing their source -- Triton and CuTe DSL require that);
* ``shapes`` -- a dispatcher `forge shapes compose` wrote; same inlining;
* ``cuda-inline`` -- a ``.cu`` converted by `forgekernel.klib.normalize.cuda_to_inline`
  (seedlib). Without that tool the candidate is reported ``not exported: needs klib`` and
  never silently dropped.

Licences: a candidate's ``meta`` (``license`` / ``source``) goes to `guard`'s
``license_check`` before anything is written; a problem it finds blocks the export, and
`guard`'s ``notices.banner`` becomes the file's header. With guard absent the header
carries the same provenance lines, marked ``license check: unavailable``.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.campaign import solscore
from forgekernel.campaign import tools as tools_mod

PROBLEM_FILE = "eval/leaderboard/problem.json"
CANDIDATE_RUN = "_fk_candidate_run"
REFERENCE_RUN = "_fk_reference_run"
REF_PREFIX = "_fk_ref_"
FAST_ENOUGH = tools_mod.COLD_START_LIMIT_S


@dataclass
class Candidate:
    """One version of a problem's kernel, ready (or not) to be exported."""
    name: str
    source: str = ""
    language: str = "python"          # python | shapes | cuda-inline | cuda-cpp
    mean_s: float | None = None
    geomean_ms: float | None = None
    cold_start_s: float | None = None
    correct: bool = True
    sha: str = ""
    run: str = ""
    kernel_file: str = "kernel.py"
    meta: dict = field(default_factory=dict)     # license / source / derived_from / ...
    origin: str = ""                             # starter | run | adapt | baseline | shapes
    notes: list[str] = field(default_factory=list)

    @property
    def sort_key(self):
        """Better first: mean S descending when scored, else geomean latency ascending."""
        if self.mean_s is not None:
            return (0, -float(self.mean_s), self.geomean_ms or 1e9)
        return (1, 0.0, self.geomean_ms if self.geomean_ms is not None else 1e9)

    def cold_state(self, limit: float = FAST_ENOUGH) -> tuple[str, str]:
        return tools_mod.BenchReading(correct=self.correct,
                                      cold_start_s=self.cold_start_s,
                                      ).cold_start_verdict(limit)


@dataclass
class ComposeResult:
    path: Path | None
    kind: str                       # single | shapes | cuda-inline | reference | none
    candidate: str = ""
    mean_s: float | None = None
    cold_start: dict = field(default_factory=dict)
    licence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.error


# --- the card ------------------------------------------------------------------------------

def card_dir(task_dir: str | Path) -> Path:
    """Where a task's frozen card lives: ``<task>/eval/leaderboard`` when the directory
    handed in *is* a card, else ``[leaderboard] card`` from its task.toml (``forge init``
    writes ``leaderboard/card``)."""
    d = Path(task_dir)
    if (d / PROBLEM_FILE).is_file():
        return d / PROBLEM_FILE
    toml = d / "task.toml"
    if toml.is_file():
        import tomllib
        try:
            card = (tomllib.loads(toml.read_text(encoding="utf-8")).get("leaderboard")
                    or {}).get("card")
        except (OSError, ValueError):
            card = None
        if card and (d / card / PROBLEM_FILE).is_file():
            return d / card / PROBLEM_FILE
    hits = sorted(d.glob(f"*/{PROBLEM_FILE}")) + sorted(d.glob(f"*/*/{PROBLEM_FILE}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{d}: no frozen board card ({PROBLEM_FILE} under a "
                            f"[leaderboard] card directory)")


def load_problem(task_dir: str | Path) -> dict:
    return json.loads(card_dir(task_dir).read_text(encoding="utf-8"))


def output_names(problem: dict) -> list[str]:
    return list((problem.get("definition") or {}).get("outputs") or {})


def dps(problem: dict) -> bool:
    return bool(problem.get("dps", True))


def card_board(problem: dict):
    """The card's T_b / T_SOL mapping (flat, or per group entry name) or None."""
    if problem.get("group"):
        return {g["name"]: g["sol_board"] for g in problem["group"] if g.get("sol_board")} or None
    return problem.get("sol_board")


# --- AST surgery: rename module-level bindings ---------------------------------------------

class _RenameModuleBindings(ast.NodeTransformer):
    """Rename the module-level names ``mapping`` says, everywhere they are referenced.

    Only ``Name`` nodes change (attribute names, keyword names and string contents do
    not), and a plain dotted ``import a.b`` binding is left alone (renaming it would
    change what the name points at). Used twice: a candidate's ``run`` -> the exported
    entry, and every top-level binding of the reference -> a ``_fk_ref_`` prefix, so the
    reference's helpers can never shadow -- or be shadowed by -- the candidate's."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)
        self.renamed: set[str] = set()

    def visit_Name(self, node):                            # noqa: N802
        if node.id in self.mapping:
            self.renamed.add(node.id)
            node.id = self.mapping[node.id]
        return node

    def visit_FunctionDef(self, node):                     # noqa: N802
        node.name = self._bound(node.name)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):                        # noqa: N802
        node.name = self._bound(node.name)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):                          # noqa: N802
        for alias in node.names:
            if alias.asname is None:
                if "." in alias.name:                      # binds the root package: leave it
                    continue
                new = self._bound(alias.name)
                if new != alias.name:
                    alias.asname = new
            else:
                new = self._bound(alias.asname)
                if new != alias.asname:
                    alias.asname = new
        return node

    def visit_ImportFrom(self, node):                      # noqa: N802
        for alias in node.names:
            if alias.name == "*":
                continue
            old = alias.asname or alias.name
            new = self._bound(old)
            if new != old:
                alias.asname = new
        return node

    def _bound(self, name: str) -> str:
        if name in self.mapping:
            self.renamed.add(name)
            return self.mapping[name]
        return name


def _module_bindings(tree: ast.Module) -> list[str]:
    """Every name bound at module level (functions, classes, assignments, imports)."""
    out: list[str] = []

    def add(name: str | None):
        if name and name not in out:
            out.append(name)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add(t.id if isinstance(t, ast.Name) else None)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    add(alias.asname)
                elif "." not in alias.name:
                    add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    add(alias.asname or alias.name)
    return out


def rename_module_bindings(source: str, mapping: dict[str, str]) -> tuple[str, set[str]]:
    tree = ast.parse(source)
    ren = _RenameModuleBindings(mapping)
    tree = ren.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), ren.renamed


def inline_source(source: str) -> tuple[str, bool]:
    """``(source with its top-level run renamed to _fk_candidate_run, whether one was found)``.

    Raises SyntaxError (the caller reports it as a rejected candidate: a version that does
    not parse cannot be exported)."""
    tree = ast.parse(source)
    found = "run" in _module_bindings(tree)
    if not found:
        return source, False
    return rename_module_bindings(source, {"run": CANDIDATE_RUN})


def isolate_reference(source: str) -> str:
    """The reference's source with every top-level binding prefixed, its ``run`` included.

    The prefix is what makes pasting it beside a candidate safe: the reference's helpers
    cannot shadow the candidate's names, and the candidate's cannot shadow the
    reference's, so the fallback keeps behaving exactly like the definition's reference.
    The entry keeps a fixed name (``_fk_reference_run``) whatever the prefix does."""
    tree = ast.parse(source)
    binds = _module_bindings(tree)
    if "run" not in binds:
        raise ValueError("the definition's reference defines no top-level run()")
    mapping = {n: f"{REF_PREFIX}{n}" for n in binds}
    mapping["run"] = REFERENCE_RUN
    text, _ = rename_module_bindings(source, mapping)
    return text


# --- the reference fallback -----------------------------------------------------------------

_REF_HEADER = '''
# --- fallback: the definition's own reference ---------------------------------------------
# Pasted as ordinary code, its top-level names prefixed so it shares nothing with the
# candidate above. Called only when the candidate raises: a shape the specialised kernel
# mishandles then costs that workload's score, never correctness for the whole problem.
{reference}

_FK_OUTPUTS = {out_names!r}
_FK_DPS = {dps!r}


def _fk_fill(outs, vals):
    import torch
    if isinstance(vals, dict):
        vals = [vals[n] for n in _FK_OUTPUTS]
    elif not isinstance(vals, (tuple, list)):
        vals = [vals]
    if len(vals) != len(outs):
        raise RuntimeError("the reference returned %d value(s) for %d output(s)"
                           % (len(vals), len(outs)))
    for dst, v in zip(outs, vals):
        if hasattr(dst, "copy_") and hasattr(dst, "dtype"):
            dst.copy_(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
        else:
            raise RuntimeError("a pre-allocated output is not a tensor; cannot write it")
'''

_TAIL_DPS = '''
def run(*args):
    try:
        return {cand}(*args)
    except Exception:
        n = len(_FK_OUTPUTS)
        outs = list(args[-n:]) if n else []
        ins = list(args[:len(args) - n]) if n else list(args)
        _fk_fill(outs, {ref}(*ins))
        return None
'''

_TAIL_VALUE = '''
def run(*args):
    try:
        return {cand}(*args)
    except Exception:
        return {ref}(*args)
'''

_TAIL_REF_ONLY_DPS = '''
def run(*args):
    n = len(_FK_OUTPUTS)
    outs = list(args[-n:]) if n else []
    ins = list(args[:len(args) - n]) if n else list(args)
    _fk_fill(outs, {ref}(*ins))
    return None
'''

_TAIL_REF_ONLY_VALUE = '''
def run(*args):
    return {ref}(*args)
'''


def reference_block(problem: dict) -> str:
    """The fallback statements: the reference (isolated), its DPS writer, the constants."""
    defn = problem.get("definition") or {}
    src = defn.get("reference")
    if not src:
        raise ValueError(f"{defn.get('name', '?')}: the definition carries no reference")
    return _REF_HEADER.format(reference=isolate_reference(str(src)),
                              out_names=output_names(problem), dps=dps(problem))


def reference_only_submission(problem: dict, *, header: str = "") -> str:
    """A complete `submission.py` that is nothing but the reference (pack's coverage
    fallback; compose falls back to it when every candidate was rejected)."""
    tail = _TAIL_REF_ONLY_DPS if dps(problem) else _TAIL_REF_ONLY_VALUE
    return header + reference_block(problem) + tail.format(ref=REFERENCE_RUN)


def compose_source(problem: dict, candidate_src: str, *, header: str = "") -> str:
    """The exported file: banner, the candidate inlined, the reference, the wrapper."""
    tail = _TAIL_DPS if dps(problem) else _TAIL_VALUE
    return (header + candidate_src.rstrip() + "\n\n"
            + reference_block(problem)
            + tail.format(cand=CANDIDATE_RUN, ref=REFERENCE_RUN))


# --- the licence header ----------------------------------------------------------------------

def provenance_header(problem: dict, candidate: Candidate | None, *,
                      license_report: dict | None, banner: str | None,
                      tool_notes: list[str] | None = None) -> str:
    """The file's opening comment: provenance, licence, and who vouched for it."""
    name = (problem.get("definition") or {}).get("name", "?")
    lines = [f'"""{name} -- campaign export (ForgeKernel).',
             "",
             "One file, top-level run() with the board's positional contract (outputs",
             "pre-allocated, last). The kernel below is this campaign's best correct",
             "version; the fallback at the end is the definition's own reference."]
    if candidate is not None:
        meta = candidate.meta or {}
        src = meta.get("source") or meta.get("derived_from") or "this campaign's runs"
        lic = meta.get("license") or (license_report or {}).get("license") or "unspecified"
        lines += ["", f"kernel: {candidate.name}",
                  f"origin: {candidate.origin or 'run'}",
                  f"source: {src}",
                  f"license: {lic}"]
        if candidate.sha:
            lines.append(f"sha: {candidate.sha[:12]}")
        for n in candidate.notes:
            lines.append(f"note: {n}")
    if license_report and license_report.get("problems"):
        lines += ["", "license check: " + "; ".join(str(p) for p in license_report["problems"])]
    elif license_report is None:
        lines += ["", "license check: unavailable (guard not installed)"]
    for n in dict.fromkeys(tool_notes or []):
        if n in " ".join(lines):
            continue                       # already said above (the licence line)
        lines.append(f"tool: {n}")
    lines.append('"""')
    text = "\n".join(lines) + "\n"
    return (banner.rstrip() + "\n\n" + text) if banner else text


# --- choosing and writing --------------------------------------------------------------------

def choose(candidates: list[Candidate], *, allow_slow_cold_start: bool = False,
           limit: float = FAST_ENOUGH) -> tuple[list[Candidate], list[dict], list[str]]:
    """``(exportable candidates best first, rejected with reasons, notes)``.

    Order: correct ones only, best score first; a candidate whose cold start exceeds
    ``limit`` is rejected unless allowed; ``unmeasured`` is kept with a note. Every
    exportable candidate is returned -- `compose` tries them in this order, because a
    candidate can still fail at export time (a ``.cu`` with no ``cuda_to_inline``)."""
    notes: list[str] = []
    usable, rejected = [], []
    for c in sorted((c for c in candidates if c.correct), key=lambda x: x.sort_key):
        state, why = c.cold_state(limit)
        if state == "too-slow" and not allow_slow_cold_start:
            rejected.append({"candidate": c.name, "reason": f"cold start: {why}"})
            continue
        if state == "too-slow":
            notes.append(f"{c.name}: {why} (allowed by --allow-slow-cold-start)")
        elif state == "unmeasured":
            notes.append(f"{c.name}: cold start unmeasured")
        usable.append(c)
    for c in candidates:
        if not c.correct:
            rejected.append({"candidate": c.name, "reason": "did not pass correctness"
                             + (f": {c.notes[0]}" if c.notes else "")})
    if not usable:
        notes.append("no exportable candidate: every version was incorrect or too slow")
    return usable, rejected, notes


def export(problem: dict, candidate: Candidate, *, out: Path, tools: tools_mod.Tools,
           allow_slow_cold_start: bool = False) -> ComposeResult:
    """Write one candidate as a `submission.py`; every gate's verdict lands in the result."""
    notes: list[str] = []
    risk: list[str] = []
    licence: dict = {}
    banner = None
    if tools.license_check is not None:
        try:
            licence = dict(tools.license_check(dict(candidate.meta or {})) or {})
        except Exception as e:                                  # noqa: BLE001
            licence = {"problems": [f"license_check raised {type(e).__name__}: {e}"]}
        if licence.get("problems"):
            return ComposeResult(path=None, kind="none", candidate=candidate.name,
                                 licence=licence,
                                 error="licence check refused the candidate: "
                                       + "; ".join(str(p) for p in licence["problems"]))
    else:
        notes.append("license check: unavailable (guard not installed)")
    if tools.notices_banner is not None:
        try:
            banner = tools.notices_banner([dict(candidate.meta or {})])
        except Exception as e:                                  # noqa: BLE001
            notes.append(f"notices.banner raised {type(e).__name__}: {e}")
            banner = None

    source = candidate.source
    kind = candidate.language
    if candidate.language in ("cuda-cpp", "cuda"):
        if tools.cuda_to_inline is None:
            return ComposeResult(path=None, kind="none", candidate=candidate.name,
                                 licence=licence,
                                 error="not exported: needs klib (a cuda-mode .cu enters a "
                                       "collection submission only through "
                                       "klib.normalize.cuda_to_inline)")
        flags = candidate.meta.get("cudaflags") or []
        source = tools.cuda_to_inline(Path(candidate.kernel_file),
                                      entry=candidate.meta.get("entry"), flags=flags)
        kind = "cuda-inline"
        notes.append("converted from CUDA C++ by klib.normalize.cuda_to_inline")
        risk.append("inline-cuda: the pinned evaluator blocks torch.utils.cpp_extension "
                    "load/load_inline on the GPU server (eval_driver.py); verify on the "
                    "live v1.1 stack before trusting this path")
    elif candidate.language not in ("python", "shapes", "cuda-inline"):
        return ComposeResult(path=None, kind="none", candidate=candidate.name,
                             error=f"unknown candidate language {candidate.language!r}")

    try:
        inlined, found = inline_source(source)
    except SyntaxError as e:
        return ComposeResult(path=None, kind="none", candidate=candidate.name,
                             error=f"the candidate does not parse: {e}")
    if not found:
        return ComposeResult(path=None, kind="none", candidate=candidate.name,
                             error="the candidate has no top-level run() to export")
    header = provenance_header(problem, candidate, license_report=licence or None,
                               banner=banner, tool_notes=notes)
    text = compose_source(problem, inlined, header=header)
    ast.parse(text)                       # the file must parse before it is written
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    state, why = candidate.cold_state()
    if tools.export_check is not None:
        try:
            verdict = dict(tools.export_check(out) or {})
        except Exception as e:                                  # noqa: BLE001
            verdict = {"error": f"{type(e).__name__}: {e}"}
    else:
        verdict = {"unavailable": True}
    notes.append(f"export check: {verdict}")
    return ComposeResult(path=out, kind=kind, candidate=candidate.name,
                         mean_s=candidate.mean_s,
                         cold_start={"state": state, "why": why,
                                     "s": candidate.cold_start_s},
                         licence=licence, notes=notes, risk=risk)


def compose(task_dir: str | Path, candidates: list[Candidate], *, out: Path,
            tools: tools_mod.Tools, allow_slow_cold_start: bool = False,
            use_shapes: bool = True) -> ComposeResult:
    """The export of one problem: per-workload winners when `shapes` is installed, else
    the single best-scoring version.

    `shapes compose` gets the task directory; when it returns nothing the
    single-candidate path runs and a note says so."""
    problem = load_problem(task_dir)
    notes: list[str] = []
    if use_shapes and tools.shapes_compose is not None and len(candidates) > 1:
        out_dir = Path(out).parent / "shapes"
        try:
            made = tools.shapes_compose(Path(task_dir), also=[], out=out_dir)
            if made is not None and Path(made).is_file():
                cand = Candidate(name=f"shapes:{Path(made).parent.name}",
                                 source=Path(made).read_text(encoding="utf-8"),
                                 language="shapes",
                                 meta={"source": "forge shapes compose"},
                                 origin="shapes")
                res = export(problem, cand, out=out, tools=tools,
                             allow_slow_cold_start=allow_slow_cold_start)
                res.notes.insert(0, "composed by forge shapes (per-workload winners)")
                return res
            notes.append("shapes compose produced no submission.py; fell back to the "
                         "single best version")
        except Exception as e:                                  # noqa: BLE001
            notes.append(f"shapes compose failed ({type(e).__name__}: {e}); fell back to "
                         f"the single best version")
    best_list, rejected, why = choose(candidates,
                                      allow_slow_cold_start=allow_slow_cold_start)
    if not best_list:
        return ComposeResult(path=None, kind="none", rejected=rejected,
                             notes=notes + why, error="no exportable candidate")
    # try them in score order: a candidate may still fail at export (a .cu with no
    # klib, a source that does not parse), and the next best is then the answer
    failures = []
    for cand in best_list:
        res = export(problem, cand, out=out, tools=tools,
                     allow_slow_cold_start=allow_slow_cold_start)
        res.rejected = rejected + failures
        res.notes = notes + why + res.notes
        if res.ok:
            return res
        failures.append({"candidate": cand.name, "reason": res.error or "export failed"})
    return ComposeResult(path=None, kind="none", candidate=best_list[0].name,
                         rejected=rejected + failures, notes=notes + why,
                         error="every exportable candidate failed to export: "
                               + "; ".join(f"{f['candidate']}: {f['reason']}"
                                           for f in failures[:4]))


def now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
