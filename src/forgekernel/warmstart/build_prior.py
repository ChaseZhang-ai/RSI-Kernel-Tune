"""``forge prior build``: the run-context manifest of what already exists for a problem.

Writes ``prior.md`` -- the file the loop hands the teacher and the student as
``{PRIOR_PATH}`` (00-background §7.5). Every line is mechanically derived from data that
already exists: the card's ``problem.json`` (definition, DPS, tolerances, T_b / T_SOL), the
card's ``prior/SOURCES.json`` + the files themselves (language, licence, hash), the
reference-library index when it exists, and the local knowledge trees when they are
present next to the base repository. No model writes prose here; rebuilding it after a
change produces exactly the text the change implies.

The manifest is deliberately blunt about licences: an implementation is listed as usable
only when its licence is Apache-2.0 / MIT / BSD-2-Clause / BSD-3-Clause (00-background §8),
and everything else is listed once, under "not reusable", so the student knows it is
there and that copying it into a submission is not allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from forgekernel.warmstart import seed, validate

MANIFEST_NAME = "prior.md"
# The reference library's index, relative to the repository (00-background §8).
KLIB_INDEX = Path("kernel_library") / "index.json"
# The local knowledge trees: `_reference/` sits beside the base repository.
KNOWLEDGE_TREES = ("KernelWiki", "ncu-report-skill")
DEFAULT_STEP = 1.2          # global_config/loop.toml's [target] ratio

PROHIBITED = (
    "monkey-patching a timing function",
    "starting a CUDA stream outside the measured path",
    "running background threads during evaluation",
    "returning a lazy proxy instead of the real tensors",
    "degenerate outputs (NaN / inf / all-zero) that disagree with the reference",
)


# -- locating the pieces ---------------------------------------------------------------


def _repo_root() -> Path:
    from forgekernel.paths import repo_root
    return repo_root()


def reference_root() -> Path:
    """``_reference/``: the sibling of the checkout running the loop (taskcard.reference_root
    walks up from a worktree too). Never an absolute literal."""
    try:
        from forgekernel.taskcard import reference_root as find
    except Exception:
        return _repo_root().parent / "_reference"
    return find()


def knowledge_trees() -> list[tuple[str, Path]]:
    root = reference_root()
    return [(name, root / name) for name in KNOWLEDGE_TREES if (root / name).is_dir()]


def card_dir(task) -> Path | None:
    """The frozen card the task names, or None for a task that gets one at run time."""
    lb = getattr(task, "leaderboard", None)
    return Path(lb.card) if lb is not None else None


def prior_dir(task, op_dir: Path | None = None) -> Path | None:
    """Where this task's existing implementations live: the card's ``prior/`` when there is
    a card, else ``<task>/prior/`` for a task generated at run time."""
    base = card_dir(task) if op_dir is None else op_dir
    if base is None:
        base = Path(task.dir)
    for candidate in (base / seed.PRIOR_DIRNAME, Path(task.dir) / seed.PRIOR_DIRNAME):
        if candidate.is_dir():
            return candidate
    return None


def problem_of(task, op_dir: Path | None = None) -> dict | None:
    """The card's ``problem.json`` (None when the task has no card yet)."""
    base = card_dir(task) if op_dir is None else op_dir
    if base is None:
        return None
    from forgekernel.leaderboard import PROBLEM_FILE
    try:
        data = json.loads((Path(base) / PROBLEM_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# -- the reference library -------------------------------------------------------------


def _as_mapping(entry) -> dict | None:
    """A klib entry as a dict: the seedlib schema's dataclass or a plain mapping."""
    if isinstance(entry, dict):
        return entry
    try:
        import dataclasses
        if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
            return dataclasses.asdict(entry)
    except Exception:
        pass
    if hasattr(entry, "__dict__"):
        return dict(vars(entry))
    return None


def klib_entries(index: Path | None = None) -> list[dict]:
    """The reference library's entries, or [] when the index (the seedlib task's
    deliverable) is not on this machine. Read-only and duck-typed: a dataclass or a dict."""
    path = Path(index) if index is not None else _repo_root() / KLIB_INDEX
    if not path.is_file():
        return []
    data: list = []
    try:
        from forgekernel.klib import load_index          # the seedlib task's loader
    except Exception:
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        data = blob.get("entries") if isinstance(blob, dict) else blob
    else:
        try:
            data = load_index(path)
        except Exception:
            return []
    out = []
    for entry in data or []:
        if (mapping := _as_mapping(entry)) is not None:
            out.append(mapping)
    return out


def klib_usable(entry: dict) -> bool:
    """The seedlib schema's own verdict when it is importable, else the licence rule."""
    try:
        from forgekernel.klib.schema import submittable
    except Exception:
        return seed.licenses_allow_reuse(entry.get("license"))
    try:
        return bool(submittable(entry))
    except Exception:
        return seed.licenses_allow_reuse(entry.get("license"))


def _names_for(task, problem: dict | None) -> set[str]:
    names = {str(task.name)}
    if problem:
        defn = problem.get("definition") or {}
        names.add(str(defn.get("name") or ""))
        source = problem.get("source") or {}
        if source.get("problem"):
            names.add(Path(str(source["problem"])).name)
    names.discard("")
    return names


def _family_of(name: str) -> str | None:
    try:
        from forgekernel.family import family_key
    except Exception:
        return None
    try:
        return family_key(name)
    except Exception:
        return None


def select_klib(entries: list[dict], task, problem: dict | None) -> list[dict]:
    """Entries whose ``problem_keys`` name this problem or its family, or whose ``family`` is
    its family (the seedlib schema: ``problem_keys`` holds board problem names and family keys)."""
    names = _names_for(task, problem)
    families = {_family_of(n) for n in names} - {None}
    out = []
    for e in entries:
        keys = {str(k) for k in (e.get("problem_keys") or [])}
        hit = bool(keys & names) or bool(keys & families)
        if not hit and families and e.get("family"):
            hit = _family_of(str(e["family"])) in families or e.get("family") in families
        if hit:
            out.append(e)
    return out


def _entry_index_path(entry: dict) -> Path | None:
    """The file a library entry names: ``path`` is relative to the library root
    (``kernel_library/kernels/<id>/kernel.py`` in the seedlib schema), so an absolute path,
    the repository-relative one and the library-relative one are all tried."""
    raw = entry.get("path")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    candidates = [p] if p.is_absolute() else [
        _repo_root() / p, _repo_root() / KLIB_INDEX.parent / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _source_of(entry: dict) -> str:
    """A library entry's provenance as one cell: ``repo:path@commit``, or the string as is."""
    source = entry.get("source")
    if isinstance(source, dict):
        parts = [str(source.get(k) or "") for k in ("repo", "path", "commit")]
        text = ":".join(parts[:2]) + (f"@{parts[2]}" if parts[2] else "")
        return text.strip(":@") or "—"
    return str(source or "—")


def _score_of(entry: dict) -> str:
    reported = entry.get("reported")
    verified = entry.get("verified_local")
    if verified is not None and not isinstance(verified, dict):
        verified = _as_mapping(verified)
    parts = []
    for label, value in (("board", reported), ("local", verified)):
        if value:
            if not isinstance(value, str):
                try:
                    value = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    value = str(value)
            parts.append(f"{label}: {value}")
    return "; ".join(parts) or "—"


def _local_reference_ms(entry: dict) -> float | None:
    """A measured reference time from the library's local verification, when it has one."""
    verified = entry.get("verified_local")
    if verified is not None and not isinstance(verified, dict):
        verified = _as_mapping(verified)
    if not isinstance(verified, dict):
        return None
    for key in ("t_ms", "median_ms", "ms", "device_median_ms"):
        v = verified.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


# -- the implementations on disk -------------------------------------------------------


def _language_of_files(files: list[Path]) -> str:
    """The language of an implementation: the first file that classifies, else unknown."""
    seen = []
    for f in files:
        if f.suffix not in (".py", ".cu", ".cuh"):
            continue
        try:
            language, method = validate.infer_language(f.read_text(encoding="utf-8"),
                                                       filename=f.name)
        except (validate.LanguageError, UnicodeDecodeError):
            seen.append("python (unclassified)")
            continue
        return language if language != "cutedsl" or method == "cutedsl" else "pytorch"
    return seen[0] if seen else "—"


def prior_rows(task, op_dir: Path | None = None) -> list[dict]:
    """One row per entry copied into ``prior/``: path, language, licence, score, origin."""
    base = prior_dir(task, op_dir)
    if base is None:
        return []
    root = base.parent
    rows = []
    for record in seed.read_sources(root):
        files = [base / f["path"] for f in record.get("files") or []]
        hashes = {f.get("sha256") for f in record.get("files") or [] if f.get("sha256")}
        rows.append({
            "path": f"{base.name}/{record.get('path')}",
            "role": record.get("role") or "",
            "language": _language_of_files([p for p in files if p.is_file()]),
            "license": record.get("license") or "unknown",
            "origin": record.get("origin") or "",
            "sha256": hashes,
            "score": "",
        })
    # a file an operator dropped in by hand (no SOURCES.json record)
    recorded = {r["path"].split("/", 1)[1] for r in rows}
    grouped: dict[str, list[Path]] = {}
    for f in seed.prior_files(root):
        rel = str(f.relative_to(base))
        top = rel.split("/", 1)[0]
        if top in recorded:
            continue
        grouped.setdefault(top, []).append(f)
    for top, files in grouped.items():
        rows.append({"path": f"{base.name}/{top}", "role": "",
                     "language": _language_of_files(files),
                     "license": seed.detect_license(files[0]), "origin": "",
                     "sha256": {seed.sha256(f) for f in files}, "score": ""})
    return rows


def _attach_scores(rows: list[dict], selected: list[dict]) -> None:
    """A row whose content is a library entry gets that entry's score (match by content)."""
    by_hash: dict[str, dict] = {}
    for entry in selected:
        path = _entry_index_path(entry)
        if path is not None:
            try:
                by_hash.setdefault(seed.sha256(path), entry)
            except OSError:
                continue
    for row in rows:
        for h in row["sha256"]:
            if h in by_hash:
                row["score"] = _score_of(by_hash[h])
                break


# -- rendering -------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
            for row in rows]
    return out


def _defn_rows(definition: dict, key: str) -> list[list[str]]:
    rows = []
    for name, spec in (definition.get(key) or {}).items():
        shape = spec.get("shape") if isinstance(spec, dict) else None
        rows.append([str(name), str(spec.get("dtype") if isinstance(spec, dict) else ""),
                     "scalar" if shape is None else "[" + ", ".join(map(str, shape)) + "]"])
    return rows


def _workload_rows(problem: dict) -> list[list[str]]:
    """One row per timed workload: axes, T_b, T_SOL, the tolerance the board applies."""
    if not problem:
        return []
    rows = []
    group = problem.get("group") or [{"name": problem["workload"].get("uuid", "workload"),
                                      "workload": problem["workload"]}]
    for g in group:
        wl = g.get("workload") or {}
        board = g.get("sol_board") or problem.get("sol_board") or {}
        tol = wl.get("tolerance") or {}
        tol_text = ", ".join(f"{k}={v}" for k, v in sorted(tol.items())) or "the board's default"
        rows.append([str(g.get("name")), json.dumps(wl.get("axes") or {}),
                     _ms(board.get("t_b_ms")), _ms(board.get("t_sol_ms")), tol_text])
    return rows


def _ms(v) -> str:
    return f"{float(v):.6g}" if isinstance(v, (int, float)) and v > 0 else "—"


def manifest_text(task, *, op_dir: Path | None = None, problem: dict | None = None,
                  entries: list[dict] | None = None, include_klib: bool = False) -> str:
    """The manifest. ``problem`` / ``entries`` are injected by tests; a normal build reads
    them from the card and the index."""
    if problem is None:
        problem = problem_of(task, op_dir)
    candidates = entries if entries is not None else klib_entries()
    selected = select_klib(candidates, task, problem) if include_klib else []
    rows = prior_rows(task, op_dir)
    if include_klib:
        _attach_scores(rows, selected)

    names = _names_for(task, problem)
    title = sorted(names)[0] if names else task.name
    lines = [f"# Existing implementations and rules: {title}", "",
             f"<!-- GENERATED by `forge prior build` from {Path(task.dir)}. "
             f"Rebuild it rather than editing it. -->", ""]

    # 1. the rules
    defaults = (problem or {}).get("dps", True)
    lines += ["## The rules this kernel must satisfy", "",
              "- **Entry**: one module-level `run`, called by POSITION. The contract is "
              "argument order and count; parameter names are not enforced.",
              ("- **Outputs**: destination-passing style — the outputs arrive pre-allocated "
               "after the inputs; write them in place and return nothing." if defaults else
               "- **Outputs**: value-returning — `run` returns the outputs (a tensor, or a "
               "tuple in the definition's order)."),
              "- **Submission form**: one `submission.py` per kernel, single file, top-level "
              "`run`; a collection submission accepts `.py` only.",
              "- **Allowed dependencies**: PyTorch, Triton, CuTe DSL / CUTLASS, cuBLAS, cuDNN. "
              "FlashInfer / FlashAttention / FlashMLA / DeepGEMM may only be vendored, never "
              "imported.", ]
    definition = (problem or {}).get("definition") or {}
    if definition:
        if definition.get("axes"):
            axes = ", ".join(f"{n}={a.get('value', a.get('expression', 'var'))}"
                             for n, a in definition["axes"].items())
            lines.append(f"- **Axes**: {axes}")
        for key, label in (("inputs", "Inputs"), ("outputs", "Outputs")):
            rows_in = _defn_rows(definition, key)
            if rows_in:
                lines += ["", f"{label}:", ""]
                lines += _table(["name", "dtype", "shape"], rows_in)
    lines += ["", "Prohibited (a workload that breaks any of these scores 0):", ""]
    lines += [f"- {p}" for p in PROHIBITED]

    # 2. the workloads the card times
    workload_rows = _workload_rows(problem)
    if workload_rows:
        lines += ["", "## What it is scored against", "",
                  "A workload's score is `S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))`: "
                  "`T_k = T_b` scores 0.5, `T_k = T_SOL` scores 1.0.", ""]
        lines += _table(["workload", "axes", "T_b (ms)", "T_SOL (ms)", "tolerance"],
                        workload_rows)

    # 3. the target the run starts with
    lines += ["", "## Target", ""]
    target = getattr(task, "target", None)
    ref_ms, ref_where = None, ""
    for entry in selected:
        ref_ms = _local_reference_ms(entry)
        if ref_ms is not None:
            ref_where = str(entry.get("id") or entry.get("path") or "the reference library")
            break
    if ref_ms is not None:
        step = target.ratio if target is not None and target.ratio else DEFAULT_STEP
        lines.append(f"- The reference library measures {ref_where} at {ref_ms:.6g} ms on "
                     f"this machine; `[target] ms = {ref_ms:.6g} / {step:g} = "
                     f"{ref_ms / step:.6g}` ms (the task's ratio, else {DEFAULT_STEP:g}).")
    else:
        lines.append("- No locally measured reference time: the card's own target applies "
                     "(the loop anchors it to the reference it measures).")
    if target is not None and target.ms is not None:
        lines.append(f"- The task sets `[target] ms = {target.ms:g}` explicitly.")

    # 4. the implementations that are already here
    lines += ["", "## Existing implementations on this machine", ""]
    reusable = [r for r in rows if seed.licenses_allow_reuse(r["license"])]
    others = [r for r in rows if not seed.licenses_allow_reuse(r["license"])]
    if reusable:
        def source_cell(r: dict) -> str:
            if r["role"] == "init-kernel":
                return f"the starting point ({r['origin']})" if r["origin"] else "the starting point"
            return r["origin"] or "—"

        lines += _table(["path", "language", "license", "score", "source"],
                        [[f"`{r['path']}`", r["language"], r["license"], r["score"] or "—",
                          source_cell(r)] for r in reusable])
        lines += ["", "Start from the highest-scoring entry whose licence permits reuse; "
                  "everything above may be read, reused and modified.", ""]
    else:
        lines += ["No implementation with a licence that permits incorporation is here yet.", ""]
    if others:
        lines += ["Present but NOT reusable (its licence does not permit incorporation into "
                  "a submission; read it for reference only, do not copy its code):", ""]
        lines += [f"- `{r['path']}` — license: {r['license']}" for r in others]
        lines.append("")

    # 5. the reference library
    if include_klib:
        lines += ["", "## Reference library (klib)", ""]
        if not candidates:
            lines.append("The reference library index is not on this machine "
                         "(no `kernel_library/index.json`): skipped.")
        elif not selected:
            lines.append("The reference library has no entry for this problem or its family.")
        else:
            lines += _table(["id", "family", "language", "license", "usable", "score",
                             "source", "path"],
                            [[str(e.get("id") or "—"), str(e.get("family") or "—"),
                              str(e.get("language") or "—"), str(e.get("license") or "unknown"),
                              "yes" if klib_usable(e) else "no",
                              _score_of(e), _source_of(e), str(e.get("path") or "—")]
                             for e in selected])
            lines += ["", "Only entries licensed Apache-2.0 / MIT / BSD may be incorporated."]

    # 6. local knowledge
    lines += ["", "## Local knowledge you may read", ""]
    trees = knowledge_trees()
    if trees:
        lines += ["Read-only, outside the workspace: nothing here is copied into a "
                  "submission, and reading it is allowed.", ""]
        for name, path in trees:
            if name == "KernelWiki":
                lines.append(f"- `{path}` — the Blackwell / Hopper knowledge base (MIT). "
                             f"Query: `python3 scripts/query.py --tag tcgen05 --compact`; "
                             f"read one page: `python3 scripts/get_page.py <id>`; its "
                             f"`artifacts/` holds complete upstream kernel files.")
            else:
                lines.append(f"- `{path}` — the NCU diagnosis manual (MIT).")
    else:
        lines.append("Not present on this machine (`_reference/` sits beside the base "
                     "repository); skip this section.")
    return "\n".join(lines).rstrip() + "\n"


def build_prior(task_dir: str | Path, *, klib: bool = False, out: Path | None = None,
                index: Path | None = None, kb_root: Path | None = None) -> Path:
    """Write the manifest for ``task_dir`` and return its path (default: the task's
    ``prior.md``)."""
    from forgekernel.task import load_task
    task = load_task(task_dir, kb_root=kb_root)
    kwargs = {"entries": klib_entries(index), "include_klib": klib}
    text = manifest_text(task, **kwargs)
    path = Path(out) if out is not None else Path(task.dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def env_for(path: str | Path) -> dict[str, str]:
    """The environment ``forge run`` (the loop) reads the manifest from: the run-context
    file ``prior`` -> ``FK_PRIOR_PATH`` -> ``{PRIOR_PATH}``. Uses the base's
    ``forgekernel.runcontext`` when that step landed, else the documented variable."""
    try:
        from forgekernel.runcontext import env_for as context_env_for
    except Exception:
        return {"FK_PRIOR_PATH": str(path)}
    try:
        return context_env_for({"prior": str(path)})
    except Exception:
        return {"FK_PRIOR_PATH": str(path)}


def env_line(path: str | Path) -> str:
    return " ".join(f"{k}={v}" for k, v in env_for(path).items())
