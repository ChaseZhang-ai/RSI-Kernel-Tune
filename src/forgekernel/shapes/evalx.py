"""Candidates, and the one way to score them: benchx on the problem's own card.

A `compose` needs a matrix of readings: rows are implementations, columns are the card's
workloads, cells are the card's own latency reading. Only benchx may produce those cells
(task §5.3 / supplement 4): this module writes each candidate out as a ``.py`` file (what
``Bench.bench`` takes), calls it once per candidate for **every** workload of the card,
and never times or judges anything itself. When benchx is not importable, `compose` says
"needs benchx" and stops -- a second evaluator would produce numbers that are not the
card's, which is worse than no matrix.

Where candidates come from (supplement 3)::

    cluster   tasks/<op>/shapes/clusters/<name>/solution.json   each cluster's export
    overall   an export of one run over the whole card (``--also``) / any solution
    extra     ``--also`` further: another agent's solution, a hand-written variant
    version   every correctness-passing iteration of the cluster runs (status.jsonl)
    baseline  `forge lib baselines`' mechanical torch candidates (soft dependency)
    eager     the definition's own reference, wrapped as a legal DPS solution

A candidate that does not parse, or has no top-level ``run``, is dropped with a note:
it could not enter the submission either.

Readings are cached by the candidate's source hash (and the card's), so re-running
`compose` after adding one candidate only benches that candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.paths import repo_root
from forgekernel.shapes import ShapesError

KIND_CLUSTER = "cluster"
KIND_OVERALL = "overall"
KIND_EXTRA = "extra"
KIND_VERSION = "version"
KIND_LIBRARY = "library"
KIND_BASELINE = "baseline"
KIND_EAGER = "eager"

# The order the routing prefers on a tie (task §5.3: specialists first, the eager
# reference last). `version` sits with the cluster it came from, `library` (a kernel the
# reference library already holds for this problem) with the extras it competes with.
KIND_ORDER = {KIND_CLUSTER: 0, KIND_VERSION: 1, KIND_OVERALL: 2, KIND_EXTRA: 3,
              KIND_LIBRARY: 4, KIND_BASELINE: 5, KIND_EAGER: 6}


@dataclass
class Candidate:
    """One implementation that could be routed to a workload.

    ``source`` is the file's text (the submission inlines it); ``path`` is where it lives
    on disk (``None`` for a candidate built in memory -- `compose` writes it before it
    benches). ``kind`` is where it came from, which is what the tie-break orders on.
    """
    label: str
    kind: str
    source: str = ""
    path: Path | None = None
    language: str = "python"
    sha: str = ""
    origin: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sha and self.source:
            self.sha = sha256_text(self.source)

    @property
    def slug(self) -> str:
        return slug(self.label)

    def to_json(self) -> dict:
        return {"label": self.label, "kind": self.kind, "sha": self.sha,
                "language": self.language, "origin": self.origin,
                "path": str(self.path) if self.path else "", "notes": list(self.notes)}

    @classmethod
    def from_json(cls, data: dict) -> "Candidate":
        return cls(label=str(data["label"]), kind=str(data.get("kind") or KIND_EXTRA),
                   path=Path(data["path"]) if data.get("path") else None,
                   language=str(data.get("language") or "python"),
                   sha=str(data.get("sha") or ""), origin=str(data.get("origin") or ""),
                   notes=list(data.get("notes") or ()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-.") or "candidate"


# --- is this a kernel? ----------------------------------------------------------------

def kernel_findings(source: str, *, entry: str = "run") -> list[str]:
    """Why ``source`` cannot be a candidate at all (empty: it can)."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"does not parse: {e.msg} (line {e.lineno})"]
    names = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if entry not in names:
        return [f"defines no top-level {entry}()"]
    return []


def candidate_from_source(source: str, label: str, kind: str, *, out_dir: Path | None,
                          language: str = "python", origin: str = "",
                          notes: list[str] | None = None) -> Candidate | None:
    """A candidate for ``source``, or None (with the reason) when it is not a kernel."""
    findings = kernel_findings(source)
    if findings:
        return None
    path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{slug(label)}.py"
        path.write_text(source, encoding="utf-8")
    return Candidate(label=label, kind=kind, source=source, path=path,
                     language=language, origin=origin, notes=list(notes or ()))


# --- where the candidates come from -----------------------------------------------------

def solution_source(doc: dict) -> tuple[str, str]:
    """``(the entry file's source, its path)`` of an official solution JSON."""
    spec = doc.get("spec") or {}
    entry = str(spec.get("entry_point") or "kernel.py").split("::", 1)[0]
    sources = {str(s.get("path")): str(s.get("content") or "")
               for s in doc.get("sources") or () if isinstance(s, dict)}
    if entry not in sources:
        raise ShapesError(f"the solution's entry_point names {entry!r}, which its sources "
                          f"do not carry ({', '.join(sorted(sources)) or 'none'})")
    langs = spec.get("languages") or spec.get("language") or []
    language = ",".join(str(x) for x in langs) if isinstance(langs, list) else str(langs)
    return sources[entry], language or "python"


def read_solution(path: str | Path) -> tuple[str, str]:
    p = Path(path).expanduser()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ShapesError(f"{p}: not a solution JSON ({e})") from None
    return solution_source(doc)


def load_candidate(path: str | Path, kind: str, *, label: str = "",
                   out_dir: Path | None = None) -> Candidate:
    """A candidate from a ``.py`` kernel or an official solution JSON on disk."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ShapesError(f"{p}: no such candidate file")
    if p.suffix == ".json":
        source, language = read_solution(p)
        name = p.parent.name if p.name in ("submission.json", "solution.json") else p.stem
    elif p.suffix in (".py", ".cu"):
        source = p.read_text(encoding="utf-8")
        language = "cuda-cpp" if p.suffix == ".cu" else "python"
        name = p.stem
    else:
        raise ShapesError(f"{p}: a candidate is a .py kernel or a solution .json")
    text = label or f"{kind}:{name}"
    if p.suffix == ".cu":
        # CUDA C++ has no single-file form yet (klib.normalize.cuda_to_inline is what
        # turns it into one); it is reported, never silently inlined as source.
        return Candidate(label=text, kind=kind, source=source, path=p,
                         language="cuda-cpp",
                         notes=["CUDA C++ (.cu): not composable yet -- it enters a "
                                "single-file submission only through "
                                "klib.normalize.cuda_to_inline (seedlib)"])
    cand = candidate_from_source(source, text, kind, out_dir=out_dir, language=language,
                                 origin=str(p))
    if cand is None:
        raise ShapesError(f"{p}: " + "; ".join(kernel_findings(source)))
    if cand.path is None:
        cand.path = p                  # bench it where it is; nothing is copied
    cand.sha = sha256_file(p)
    return cand


def cluster_candidates(plan: dict, *, out_dir: Path | None = None) -> list[Candidate]:
    """Each cluster's exported best kernel (``clusters/<name>/solution.json``)."""
    out: list[Candidate] = []
    base = Path(plan.get("product_dir") or "")
    for c in plan.get("clusters") or []:
        name = str(c["name"])
        path = base / "clusters" / name / "solution.json"
        if not path.is_file():
            out.append(Candidate(label=name, kind=KIND_CLUSTER,
                                 notes=[f"no export at {path} (`forge shapes run` writes it)"]))
            continue
        try:
            source, language = read_solution(path)
        except ShapesError as e:
            out.append(Candidate(label=name, kind=KIND_CLUSTER, notes=[str(e)]))
            continue
        cand = candidate_from_source(source, name, KIND_CLUSTER, out_dir=out_dir,
                                     language=language, origin=str(path))
        if cand is None:
            out.append(Candidate(label=name, kind=KIND_CLUSTER,
                                 notes=[f"{path}: " + "; ".join(kernel_findings(source))]))
            continue
        out.append(cand)
    return out


_STATUS = "status.jsonl"


def versions_of_run(run_dir: str | Path, kernel_file: str = "kernel.py",
                    *, limit: int = 12) -> tuple[list[Candidate], list[str]]:
    """Every correctness-passing version of one run (``status.jsonl`` + the ws git).

    A run's ``best_confirmation`` is one version; the loop keeps many other *correct*
    ones, and a version that lost the geomean gate may still be the fastest on one
    workload -- which is exactly what per-workload dispatch is for. Sources come from
    the run workspace's own git record (``git show <sha>:<kernel_file>``), so nothing is
    taken on trust from the log.
    """
    notes: list[str] = []
    run = Path(run_dir).expanduser()
    if not run.is_dir():
        return [], [f"{run}: no such run directory"]
    task_name = ""
    try:
        task_name = str(json.loads((run / "run.json").read_text(encoding="utf-8"))
                        .get("task") or "")
    except (OSError, json.JSONDecodeError):
        pass
    op_dir = run / "operators" / task_name if task_name else None
    if op_dir is None or not op_dir.is_dir():
        hits = sorted(p for p in (run / "operators").glob("*") if p.is_dir()) \
            if (run / "operators").is_dir() else []
        op_dir = hits[0] if hits else None
    status = (op_dir / "student" / _STATUS) if op_dir is not None else None
    if status is None or not status.is_file():
        return [], [f"{run}: no student/status.jsonl (nothing to read versions from)"]
    rows: list[dict] = []
    for line in status.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    ws = run / "ws"
    out: list[Candidate] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("verdict") != "KEEP" or row.get("correctness") is False:
            continue
        sha = str(row.get("sha") or "")
        if not sha or sha in seen or len(out) >= limit:
            continue
        source = _git_show(ws, sha, kernel_file)
        if not source:
            continue
        seen.add(sha)
        label = f"v:{run.name}:it{row.get('iter')}"
        cand = candidate_from_source(source, label, KIND_VERSION, out_dir=None,
                                     origin=f"git:{sha[:12]}:{kernel_file}",
                                     notes=[f"iteration {row.get('iter')} KEEP"])
        if cand is None:
            notes.append(f"{label}: does not parse as a kernel, skipped")
            continue
        cand.sha = sha
        out.append(cand)
    # An evolve population's variants are the same kind of candidate.
    for variant in sorted(run.glob("variants/*")):
        kernel = variant / Path(kernel_file).name
        if not kernel.is_file() or len(out) >= limit:
            continue
        source = kernel.read_text(encoding="utf-8")
        cand = candidate_from_source(source, f"v:{run.name}:{variant.name}", KIND_VERSION,
                                     out_dir=None, origin=str(kernel),
                                     notes=[f"evolve variant {variant.name}"])
        if cand is not None:
            out.append(cand)
    return out, notes


def _git_show(ws: Path, sha: str, rel: str) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(ws), "show", f"{sha}:{rel}"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def library_candidates(problem: dict, *, root: str | Path | None = None,
                       out_dir: Path | None = None) -> tuple[list[Candidate], list[str]]:
    """The reference library's kernels for **this** problem (soft: seedlib may be absent).

    A kernel whose ``problem_keys`` name this problem is already a candidate: it is one file
    with a positional ``run``, and its licence is on the entry. Only entries the library
    marks submittable are taken (Apache-2.0 / MIT / BSD and ``usable_in_submission``); a
    kernel that only matches a *family* keyword is not: turning a family reference into a
    kernel for this problem is `forge lib adapt`'s job, and its output reaches `compose` as
    an ``--also`` candidate.
    """
    notes: list[str] = []
    try:
        from forgekernel.klib import schema as klib
    except ImportError:
        return [], ["kernel library: unavailable (the seedlib task has not landed)"]
    base = Path(root) if root is not None else repo_root()
    index = base / "kernel_library" / "index.json"
    if not index.is_file():
        return [], [f"kernel library: no index at {index}"]
    try:
        entries = klib.load_index(index)
    except Exception as e:                                   # noqa: BLE001 -- soft
        return [], [f"kernel library: {index} does not read ({type(e).__name__}: {e})"]
    wanted = {str((problem.get("definition") or {}).get("name") or "").lower(),
              str((problem.get("source") or {}).get("problem") or "").rsplit("/", 1)[-1].lower()}
    wanted.discard("")
    out: list[Candidate] = []
    skipped: list[str] = []
    for entry in entries:
        keys = {str(k).lower() for k in (entry.problem_keys or ())}
        if not (keys & wanted):
            continue
        if not klib.submittable(entry):
            skipped.append(f"{entry.id} (licence {entry.license!r})")
            continue
        path = base / "kernel_library" / str(entry.path)
        if not path.is_file():
            skipped.append(f"{entry.id} ({path} is missing)")
            continue
        source = path.read_text(encoding="utf-8")
        cand = candidate_from_source(source, f"lib:{entry.id}", KIND_LIBRARY,
                                     out_dir=out_dir, language=str(entry.language or "python"),
                                     origin=str(path),
                                     notes=[f"kernel library entry {entry.id} "
                                            f"({entry.license})"])
        if cand is None:
            skipped.append(f"{entry.id} (not a kernel: "
                           f"{'; '.join(kernel_findings(source))})")
            continue
        cand.sha = sha256_text(source)
        out.append(cand)
    notes.append(f"kernel library: {len(out)} submittable entry(ies) name this problem"
                 + (f"; skipped {', '.join(skipped)}" if skipped else ""))
    return out, notes


def lib_baselines(problem: dict, *, out_dir: Path, log=None) -> tuple[list[Candidate], list[str]]:
    """`forge lib baselines`' mechanical candidates (soft: seedlib may not be installed)."""
    notes: list[str] = []
    try:
        import contextlib
        import io

        from forgekernel.ext import lib as lib_ext       # noqa: F401 -- probe only
    except ImportError:
        return [], ["forge lib baselines: unavailable (the seedlib task has not landed)"]
    source = problem.get("source") or {}
    dataset, dataset_root = str(source.get("dataset") or ""), str(source.get("dataset_root") or "")
    name = str(source.get("problem") or (problem.get("definition") or {}).get("name") or "")
    if not dataset_root or not name:
        return [], ["forge lib baselines: the card records no dataset root, skipped"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ["baselines", name, "--dataset", dataset_root, "--out", str(out_dir)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = lib_ext.main(argv)
    except SystemExit as e:                                  # argparse's own exit
        rc = int(e.code or 0)
    except Exception as e:                                   # noqa: BLE001 -- soft
        return [], [f"forge lib baselines raised {type(e).__name__}: {e}"]
    if rc != 0:
        return [], [f"forge lib baselines exited {rc}: {buf.getvalue().strip()[:300]}"]
    out: list[Candidate] = []
    for path in sorted(out_dir.glob("*.py")):
        try:
            cand = load_candidate(path, KIND_BASELINE, label=f"lib:{path.stem}")
        except ShapesError as e:
            notes.append(str(e))
            continue
        out.append(cand)
    notes.append(f"forge lib baselines wrote {len(out)} candidate(s) into {out_dir}"
                 + (f" (dataset {dataset or dataset_root})" if dataset or dataset_root else ""))
    return out, notes


# --- benchx ---------------------------------------------------------------------------

def benchx_module():
    """The benchx module, or ShapesError: there is no second evaluator (supplement 4)."""
    try:
        import forgekernel.benchx.api as api
    except ImportError as e:                                 # pragma: no cover
        raise ShapesError("`shapes compose` needs benchx (forgekernel.benchx); it is not "
                          f"importable ({e}). No matrix is built without it: a second "
                          "evaluator would not read the card's clock") from None
    return api


def make_bench(*, repeat: int = 1, keep: bool = False, log=None):
    """The real bench (`CardBench`), or ShapesError when this checkout has no runner."""
    benchx_module()
    try:
        from forgekernel.benchx.runner import CardBench
    except ImportError as e:                                 # pragma: no cover
        raise ShapesError(f"forgekernel.benchx.runner is not importable ({e}); "
                          f"`shapes compose` needs the real bench") from None
    try:
        return CardBench(repeat=repeat, keep=keep, log=log)
    except TypeError:                                        # a bench without the log hook
        return CardBench(repeat=repeat, keep=keep)


CACHE_FILE = "matrix.json"


class ReadingCache:
    """Bench readings by candidate hash, invalidated by the card's own hash."""

    def __init__(self, path: Path | None = None, *, card_sha: str = "",
                 entries: dict | None = None):
        self.path = Path(path) if path else None
        self.card_sha = card_sha
        self.entries: dict[str, dict] = dict(entries or {})

    @classmethod
    def load(cls, path: str | Path, *, card_sha: str = "") -> "ReadingCache":
        p = Path(path)
        cache = cls(p, card_sha=card_sha)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cache
        if doc.get("card_sha") and card_sha and doc["card_sha"] != card_sha:
            return cache                       # a different card: nothing is comparable
        for row in doc.get("candidates") or []:
            if isinstance(row, dict) and row.get("sha") and isinstance(row.get("result"), dict):
                cache.entries[str(row["sha"])] = row["result"]
        return cache

    def get(self, cand: Candidate) -> dict | None:
        return self.entries.get(cand.sha) if cand.sha else None

    def put(self, cand: Candidate, result) -> None:
        if cand.sha:
            self.entries[cand.sha] = _result_json(result)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"card_sha": self.card_sha, "updated": _now(),
               "candidates": [{"sha": k, "result": v} for k, v in self.entries.items()]}
        self.path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")


def _result_json(result) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return dict(result)


def card_sha(card: Path) -> str:
    p = Path(card) / "eval/leaderboard/problem.json"
    return sha256_file(p) if p.is_file() else ""


def bench_all(candidates: list[Candidate], card: Path, bench, *,
              cache: ReadingCache | None = None, reuse: bool = True,
              mode: str = "dev", log=None) -> list[tuple[Candidate, object]]:
    """One reading per candidate over **every** workload of the card.

    A bench that raises, or reports ``correct=False``, is recorded as it came -- a
    candidate that is only correct on its own cluster is the normal case here, and the
    card's verdict on each workload is what excludes it there. ``cache`` reuses a
    candidate's whole reading when its source hash matches (nothing is re-judged: the
    cached object *is* the card's earlier verdict).
    """
    out: list[tuple[Candidate, object]] = []
    for cand in candidates:
        if cand.path is None and cand.source:
            raise ShapesError(f"{cand.label}: a candidate is benched from a file; "
                              f"`compose` writes it first")
        cached = cache.get(cand) if (cache is not None and reuse) else None
        if cached is not None:
            if log:
                log(f"{cand.label}: reused the cached reading")
            out.append((cand, cached))
            continue
        if cand.path is None:
            out.append((cand, {"correct": False, "workloads": [],
                              "error": cand.notes[0] if cand.notes else "no source"}))
            continue
        if log:
            log(f"{cand.label}: {mode} bench on {Path(card).name} "
                f"({Path(cand.path).name})")
        try:
            result = bench.bench(Path(card), Path(cand.path), workloads=None, mode=mode)
        except Exception as e:                               # noqa: BLE001 -- one row
            result = {"correct": False, "workloads": [], "error":
                      f"the bench raised {type(e).__name__}: {e}"}
        out.append((cand, result))
        if cache is not None:
            cache.put(cand, result)
            cache.save()
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
