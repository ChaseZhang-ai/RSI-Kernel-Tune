"""Pure functions that turn a kernel's source into the library's one deliverable shape.

Every entry in ``kernel_library/kernels/<id>/kernel.py`` is a **single Python file with a
top-level ``run(*args)``**, positional arguments, pre-allocated outputs last (the submission
contract). Sources arrive in these shapes and each has a rule here:

1. **single-file Python** — keep the source verbatim; if its entry is not called ``run``,
   append ``run = <entry>``. The calculation is never touched.
2. **multi-file Python** — merge into one file, dependency order first. Each other file's
   top-level bindings are prefixed (``_klib_<stem>__<name>``) and every reference to them is
   rewritten, so side-by-side definitions cannot shadow each other; the entry file is
   appended last with its imports of the inlined modules rewritten to the prefixed names.
   The rewrite is textual (AST positions → byte splices): comments, license headers and
   formatting survive.
3. **CUDA C++** (``.cu``/``.cpp`` + headers, a ``PYBIND11_MODULE`` binding) — ``cuda_to_inline``
   embeds the sources in a generated ``submission.py`` that builds them once at import with
   ``torch.utils.cpp_extension.load_inline``, with the gencode derived from the running
   device (never a literal).

Nothing here runs a kernel or touches a GPU: it is text in, text out, and every result
carries a ``status`` of ``ok`` or ``unconvertible`` with the reason.
"""

from __future__ import annotations

import ast
import hashlib
import re
import symtable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

#: Suffixes treated as C++/CUDA source.
CUDA_SUFFIXES = (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".inl")

#: A complete submission must not pin a code-generation target: the same file is built on
#: the development machine (sm_103) and graded on the target (sm_100). These patterns find a
#: *literal* target in generated wrapper code (an embedded kernel's own text is not scanned).
ARCH_LITERAL = re.compile(r"\b(?:sm|compute)_(\d{2,3})([a-z]?)\b")
GENCODE_LITERAL = re.compile(r"-gencode[= ]arch=compute_\w+,code=(?:sm|compute)_\w+")
ARCH_LIST_LITERAL = re.compile(r"""TORCH_CUDA_ARCH_LIST["']?\s*[=:]\s*["'][^"']+["']""")
SM_COUNT_LITERAL = re.compile(r"\b(?:num_?sms|sm_?count|SM_COUNT|NUM_SMS)\s*=\s*\d+")

#: A generated wrapper must contain this to prove the arch is derived at runtime.
RUNTIME_ARCH_MARKER = "get_device_capability"

#: Marker words a variant's own header uses for a trap worth quoting in ``find`` output.
TRAP_WORDS = ("dead-end", "dead end", "measured dead", "do not", "do NOT", "must not",
              "gated off", "trap", "Trap", "!!!")


class NormalizeError(ValueError):
    """The source cannot be brought into the deliverable shape (say why)."""


@dataclass
class NormalizeResult:
    """What a normalization produced. ``status == "unconvertible"`` ⇒ ``source is None``."""

    status: str
    source: str | None
    language: str
    entry: str
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.source)

    def text(self) -> str:
        if self.source is None:
            raise NormalizeError(f"unconvertible: {self.reason}")
        return self.source


def unconvertible(reason: str, *, language: str = "", entry: str = "run",
                  notes: Iterable[str] = (), files: Iterable[str] = ()) -> NormalizeResult:
    return NormalizeResult(status="unconvertible", source=None, language=language,
                           entry=entry, reason=reason, notes=list(notes), files=list(files))


# --------------------------------------------------------------------------------------
# parsing / inspection
# --------------------------------------------------------------------------------------

def parse(source: str, filename: str = "<kernel>") -> ast.Module:
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise NormalizeError(f"{filename}: not valid Python: {exc}") from exc


def top_level_functions(source: str) -> list[str]:
    """Names of top-level ``def``/``class`` in order of definition."""
    tree = parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def ensure_entry(source: str, entry: str | None = None) -> tuple[str, str]:
    """``(source, "run")`` — the source with ``run`` defined.

    ``run`` already defined ⇒ unchanged. Otherwise ``entry`` (or, when not given, the last
    public top-level function) is aliased: ``run = <name>``. A source with no top-level
    function at all is refused.
    """
    names = top_level_functions(source)
    if "run" in names:
        return source, "run"
    if entry and entry not in names:
        public = [n for n in names if not n.startswith("_")]
        raise NormalizeError(
            f"entry {entry!r} is not defined; top-level definitions: {public or names or 'none'}")
    chosen = entry
    if not chosen:
        public = [n for n in names if not n.startswith("_")]
        chosen = public[-1] if public else (names[-1] if names else None)
    if not chosen:
        raise NormalizeError("no top-level function or class to expose as `run`")
    body = source if source.endswith("\n") else source + "\n"
    body += (f"\n\n# klib: the library's entry point is `run`; this file's entry is `{chosen}`.\n"
             f"run = {chosen}\n")
    parse(body)  # the alias must parse (a class gets a callable alias too)
    return body, "run"


def has_entry(source: str) -> bool:
    try:
        return "run" in top_level_functions(source)
    except NormalizeError:
        return False


def detect_language(source: str) -> str:
    """``triton`` / ``cutedsl`` / ``cuda-inline`` / ``pytorch`` from the source's imports."""
    text = source
    if re.search(r"\bimport\s+triton\b|\bfrom\s+triton\b|@triton\.jit", text):
        return "triton"
    if re.search(r"\bimport\s+cutlass\b|\bfrom\s+cutlass\b|\bcutlass\.cute\b|@cute\.(jit|kernel)",
                 text):
        return "cutedsl"
    if "load_inline" in text or "cpp_extension" in text:
        return "cuda-inline"
    return "pytorch"


def function_source(source: str, name: str) -> str | None:
    """The source text of one top-level function (used by tests and by ``adapt``)."""
    tree = parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    return None


def static_rules_check(source: str, *, require_arch_provenance: bool = True) -> list[str]:
    """Target-profile rules a *generated wrapper* must satisfy (the soft check).

    Returns the violations (empty = clean). Pass only wrapper text here, never an embedded
    kernel: a CUDA kernel's own text may legitimately mention an architecture.
    ``require_arch_provenance`` is off for a candidate that compiles nothing at runtime (a
    plain PyTorch or Triton kernel derives no architecture, so there is nothing to prove).
    """
    problems: list[str] = []
    if not source.strip():
        return problems
    if require_arch_provenance and RUNTIME_ARCH_MARKER not in source:
        problems.append(
            "the code-generation target is not derived from the running device "
            f"(no {RUNTIME_ARCH_MARKER})")
    if GENCODE_LITERAL.search(source) or ARCH_LIST_LITERAL.search(source):
        problems.append("a code-generation target is pinned as a literal")
    elif ARCH_LITERAL.search(source):
        problems.append(
            "an architecture appears as a literal (derive it from the device instead)")
    if SM_COUNT_LITERAL.search(source):
        problems.append("an SM count is hardcoded (query the device at run time)")
    return problems


def compiles_cuda(source: str) -> bool:
    """Whether a candidate builds CUDA/C++ at run time (then the arch must be derived)."""
    return ("load_inline" in source or "cpp_extension" in source
            or re.search(r"\bfrom\s+torch\.utils\.cpp_extension\b", source) is not None)


# --------------------------------------------------------------------------------------
# splices and literals
# --------------------------------------------------------------------------------------

def string_literal(text: str) -> str:
    """A Python literal holding ``text``, preferring readable triple quotes."""
    body = text if text.endswith("\n") else text + "\n"
    for quote in ('"""', "'''"):
        if quote not in body and not body.endswith("\\"):
            return f"r{quote}{body}{quote}"
    return repr(text)


Edit = tuple[int, int, int, int, str]  # (start_line, start_col, end_line, end_col, text)


def _offset(lines: list[str], lineno: int, col: int) -> int:
    return sum(len(line) for line in lines[:lineno - 1]) + col


def apply_edits(source: str, edits: Iterable[Edit]) -> str:
    """Apply non-overlapping AST-positioned replacements, bottom-up."""
    lines = source.splitlines(keepends=True)
    resolved: list[tuple[int, int, str]] = []
    for sl, sc, el, ec, text in edits:
        lo = _offset(lines, sl, sc)
        hi = _offset(lines, el, ec)
        if hi < lo:
            raise NormalizeError(f"bad edit span {sl}:{sc}..{el}:{ec}")
        resolved.append((lo, hi, text))
    resolved.sort(key=lambda item: (item[0], item[1]), reverse=True)
    last_lo = len(source) + 1
    out = source
    for lo, hi, text in resolved:
        if hi > last_lo:
            raise NormalizeError("overlapping edits")
        out = out[:lo] + text + out[hi:]
        last_lo = lo
    return out


def rename_symbol(source: str, old: str, new: str) -> str:
    """Rename a top-level symbol textually, keeping comments and formatting byte for byte."""
    tree = parse(source)
    edits: list[Edit] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == old:
            kw = len("class ") if isinstance(node, ast.ClassDef) else len("def ")
            edits.append((node.lineno, node.col_offset + kw,
                          node.lineno, node.col_offset + kw + len(old), new))
        elif isinstance(node, ast.Name) and node.id == old:
            edits.append((node.lineno, node.col_offset, node.end_lineno,
                          node.end_col_offset, new))
    if not edits:
        raise NormalizeError(f"symbol {old!r} not found")
    out = apply_edits(source, edits)
    parse(out)
    return out


# --------------------------------------------------------------------------------------
# multi-file merge: dependency order + private-name prefixes
# --------------------------------------------------------------------------------------

def module_stem(path: str) -> str:
    return Path(path).stem


def _module_bindings(source: str, filename: str = "<module>") -> list[str]:
    """Top-level bindings of a module (defs, classes, assignments, imports), in order."""
    try:
        table = symtable.symtable(source, filename, "exec")
    except SyntaxError as exc:  # already parsed by the caller; defensive
        raise NormalizeError(f"{filename}: not valid Python: {exc}") from exc
    names: list[str] = []
    for sym in table.get_symbols():
        name = sym.get_name()
        if name.startswith("__") and name.endswith("__"):
            continue
        if sym.is_assigned() or sym.is_imported():
            names.append(name)
    return names


def _prefix_for(stem: str) -> str:
    return "_klib_" + re.sub(r"[^0-9A-Za-z_]", "_", stem) + "__"


def _alias_text(original: str, bound: str) -> str:
    return f"{original} as {bound}"


def _prefix_module(source: str, prefix: str, path: str,
                   notes: list[str]) -> tuple[str, dict[str, str]]:
    """Prefix a module's top-level bindings and every reference to them."""
    tree = parse(source, path)
    bindings = _module_bindings(source, path)
    mapping = {name: prefix + name for name in bindings}
    edits: list[Edit] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name in mapping:
            kw = len("class ") if isinstance(node, ast.ClassDef) else len("def ")
            edits.append((node.lineno, node.col_offset + kw,
                          node.lineno, node.col_offset + kw + len(node.name),
                          mapping[node.name]))
        elif isinstance(node, ast.Name) and node.id in mapping:
            edits.append((node.lineno, node.col_offset, node.end_lineno,
                          node.end_col_offset, mapping[node.id]))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            renamed = [mapping.get(n, n) for n in node.names]
            if renamed != node.names:
                keyword = "global" if isinstance(node, ast.Global) else "nonlocal"
                edits.append((node.lineno, node.col_offset, node.end_lineno,
                              node.end_col_offset, f"{keyword} {', '.join(renamed)}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in mapping:
                    edits.append((alias.lineno, alias.col_offset, alias.end_lineno,
                                  alias.end_col_offset,
                                  _alias_text(alias.name, mapping[bound])))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    notes.append(f"{path}: `from {node.module} import *` cannot be prefixed")
                    continue
                bound = alias.asname or alias.name
                if bound in mapping:
                    edits.append((alias.lineno, alias.col_offset, alias.end_lineno,
                                  alias.end_col_offset,
                                  _alias_text(alias.name, mapping[bound])))
    out = apply_edits(source, edits)
    parse(out, path)
    return out, mapping


def _rewrite_entry_imports(source: str, modules: Mapping[str, dict[str, str]],
                            path: str, notes: list[str]) -> str:
    """Rewrite the entry file's references to inlined modules to their prefixed names."""
    tree = parse(source, path)
    edits: list[Edit] = []
    aliases: dict[str, str] = {}  # local name -> module stem
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            stems = [a.name.split(".")[0] for a in node.names]
            if len(node.names) == 1 and stems[0] in modules:
                alias = node.names[0].asname or stems[0]
                aliases[alias] = stems[0]
                edits.append((node.lineno, node.col_offset, node.end_lineno,
                              node.end_col_offset, f"# klib: `{alias}` is inlined below"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            stem = node.module.split(".")[0]
            if stem in modules and len(node.names) == 1:
                alias = node.names[0]
                if alias.name == "*":
                    notes.append(f"{path}: star import from inlined module {stem!r} kept as-is")
                    continue
                bound = alias.asname or alias.name
                mapped = modules[stem].get(alias.name)
                if mapped is None:
                    notes.append(f"{path}: imports {alias.name!r} from {stem!r} but the inlined "
                                 f"module does not define it")
                    continue
                aliases[bound] = stem
                edits.append((node.lineno, node.col_offset, node.end_lineno,
                              node.end_col_offset, f"{bound} = {mapped}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            stem = aliases.get(node.value.id)
            if stem is None:
                continue
            mapped = modules[stem].get(node.attr)
            if mapped is None:
                notes.append(f"{path}: attribute {node.value.id}.{node.attr} is not a top-level "
                             f"binding of the inlined module {stem!r}; left unchanged")
                continue
            edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, mapped))
    return apply_edits(source, edits) if edits else source


def _import_deps(source: str, known: Mapping[str, str]) -> set[str]:
    """Stems of ``known`` that ``source`` imports (or loads by file name)."""
    deps: set[str] = set()
    for stem in known:
        if re.search(rf"\b(?:import\s+{re.escape(stem)}\b|from\s+{re.escape(stem)}\s+import"
                     rf"|{re.escape(stem)}\.py)", source):
            deps.add(stem)
    return deps


def order_by_dependency(sources: Mapping[str, str], *, entry_path: str | None = None,
                        notes: list[str] | None = None) -> list[str]:
    """Paths of ``sources`` with dependencies first and the entry file last."""
    notes = notes if notes is not None else []
    paths = list(sources)
    stems = {module_stem(p): p for p in paths}
    entry = entry_path
    if entry is None or entry not in sources:
        entry = _pick_entry(sources)
    deps: dict[str, set[str]] = {}
    for path in paths:
        dep_stems = _import_deps(sources[path], {s: p for s, p in stems.items() if p != path})
        deps[path] = {stems[s] for s in dep_stems}
    ordered: list[str] = []
    placed: set[str] = set()
    visiting: set[str] = set()

    def visit(path: str) -> None:
        if path in placed:
            return
        if path in visiting:
            notes.append(f"circular import between merged files (kept in given order): {path}")
            return
        visiting.add(path)
        for dep in sorted(deps.get(path, ())):
            if dep != entry:
                visit(dep)
        visiting.discard(path)
        placed.add(path)
        ordered.append(path)

    for path in paths:
        if path != entry:
            visit(path)
    placed.add(entry)
    ordered.append(entry)
    return ordered


def _pick_entry(sources: Mapping[str, str]) -> str:
    """The file that carries the entry: a top-level ``run``, else the binding, else name."""
    for path, text in sources.items():
        if Path(path).suffix == ".py" and has_entry(text):
            return path
    for path, text in sources.items():
        if "PYBIND11_MODULE" in text:
            return path
    for candidate in ("kernel.py", "main.py"):
        for path in sources:
            if Path(path).name == candidate:
                return path
    return list(sources)[-1]


_MERGE_HEADER = '''"""Merged by rsi_kernel_tune.klib.normalize — do not edit; edit the original sources.

Inlined, in dependency order: {inlined}
Entry file (verbatim last): {entry}
Every inlined module's top-level bindings are prefixed `_klib_<module>__` and all references
to them are rewritten, so side-by-side definitions cannot shadow each other.
"""
'''


def merge_python_sources(sources: Mapping[str, str], *, entry_path: str | None = None,
                         entry: str | None = None, header: str = "") -> NormalizeResult:
    """Merge a multi-file Python solution into one file with a top-level ``run``."""
    if not sources:
        return unconvertible("no sources")
    notes: list[str] = []
    order = order_by_dependency(sources, entry_path=entry_path, notes=notes)
    entry_file = order[-1]
    inlined = order[:-1]
    modules: dict[str, dict[str, str]] = {}
    blocks: list[str] = []
    for path in inlined:
        stem = module_stem(path)
        prefix = _prefix_for(stem)
        try:
            prefixed, mapping = _prefix_module(sources[path], prefix, path, notes)
        except NormalizeError as exc:
            return unconvertible(f"merging {path} failed: {exc}", notes=notes, files=order)
        modules[stem] = mapping
        blocks.append(f"\n# ---------------------------------------------------------------- inlined: {path}\n"
                      f"{prefixed}")
        if "spec_from_file_location" in sources[path]:
            notes.append(f"{path}: loads a sibling file by path at run time; after the merge "
                         f"that call still reads from disk and the file may need adaptation")
    try:
        entry_text = _rewrite_entry_imports(sources[entry_file], modules, entry_file, notes)
    except NormalizeError as exc:
        return unconvertible(f"rewriting the entry file failed: {exc}", notes=notes, files=order)
    if "spec_from_file_location" in entry_text or re.search(r"open\([^)]*\.py", entry_text):
        notes.append(f"{entry_file}: loads a sibling file by path at run time; after the merge "
                     f"that call still reads from disk and the file may need adaptation")
    merged = (_MERGE_HEADER.format(inlined=", ".join(inlined) or "(none)", entry=entry_file)
              + "\n".join(blocks))
    merged += f"\n\n# ---------------------------------------------------------------- entry: {entry_file}\n"
    merged += entry_text
    if header:
        merged = (header if header.endswith("\n\n") else header + "\n\n") + merged
    try:
        merged, entry_name = ensure_entry(merged, entry)
    except NormalizeError as exc:
        return unconvertible(f"merge failed: {exc}", language=detect_language(entry_text),
                             notes=notes, files=order)
    return NormalizeResult(status="ok", source=merged, language=detect_language(merged),
                           entry=entry_name, notes=notes, files=order)


def single_python(source: str, *, entry: str | None = None, header: str = "") -> NormalizeResult:
    """A single-file Python kernel: keep it, make sure ``run`` is defined."""
    try:
        body, entry_name = ensure_entry(source, entry)
    except NormalizeError as exc:
        return unconvertible(str(exc), language=detect_language(source))
    if header:
        body = (header if header.endswith("\n\n") else header + "\n\n") + body
    notes: list[str] = []
    if "spec_from_file_location" in source or re.search(r"open\([^)]*\.py", source):
        notes.append("loads a sibling file at run time; the single file may need adaptation")
    return NormalizeResult(status="ok", source=body, language=detect_language(body),
                           entry=entry_name, notes=notes)


def bundle_files(sources: Mapping[str, str], *, language: str, reason: str,
                 header: str = "", entry: str = "run") -> NormalizeResult:
    """A reference bundle for sources that cannot become one runnable file mechanically.

    The file parses, carries every original byte in ``_SOURCES`` (so ``forge lib adapt`` and a
    human can read it), and defines ``run`` as a loud refusal. The caller records
    ``runnable: false`` in ``meta.json``; nothing presents it as a submission.
    """
    if not sources:
        return unconvertible("no sources")
    docstring = f'"""{header + chr(10) + chr(10) if header else ""}Reference bundle — {reason}"""\n'
    lines = [docstring,
             "_SOURCES = {\n"
             + "\n".join(f"    {path!r}: {string_literal(text)},"
                         for path, text in sorted(sources.items()))
             + "\n}\n",
             "\n\ndef run(*args):\n"
             '    """Not a submission: this artifact is reference material (see `_SOURCES`)."""\n'
             '    raise NotImplementedError(\n'
             f'        "reference bundle ({reason}): adapt it first (`forge lib adapt`)")\n']
    body = "\n".join(lines)
    try:
        parse(body)
    except NormalizeError as exc:
        return unconvertible(f"bundle did not parse: {exc}")
    if not has_entry(body):
        return unconvertible("bundle has no `run`")
    return NormalizeResult(status="ok", source=body, language=language, entry=entry,
                           notes=[f"bundled files: {', '.join(sorted(sources))}",
                                  f"not runnable: {reason}"],
                           files=sorted(sources))


def bundle_sources(entry_source: str, siblings: Mapping[str, str], *, entry: str = "run",
                   reason: str = "", header: str = "") -> NormalizeResult:
    """A reference bundle: the entry source and its sibling files, embedded as literals.

    For a source whose real build cannot be reduced to one file mechanically (a ``kernel.py``
    that JIT-builds sibling ``.cu``/``.cuh`` files from disk, say). The file parses, exposes
    ``run``, and carries every original byte for ``forge lib adapt`` to work from; it is
    marked not runnable by the caller's meta, so nothing presents it as a submission.
    """
    parts = [(header + "\n\n" if header else "") +
             f'"""Reference bundle — {reason or "not runnable as one file; use `forge lib adapt`"}"""\n']
    names = sorted(Path(p).name for p in siblings)
    parts.append(
        "_SIBLING_SOURCES = {\n"
        + "\n".join(f"    {Path(p).name!r}: {string_literal(text)},"
                    for p, text in sorted(siblings.items()))
        + "\n}\n")
    parts.append("# The original entry file follows, verbatim.\n")
    parts.append(entry_source)
    body = "\n".join(parts)
    try:
        body, entry_name = ensure_entry(body, entry)
        parse(body)
    except NormalizeError as exc:
        return unconvertible(f"bundle did not parse: {exc}")
    return NormalizeResult(status="ok", source=body, language=detect_language(entry_source),
                           entry=entry_name, notes=[f"bundled siblings: {', '.join(names)}"],
                           files=[*sorted(siblings), "kernel.py"])


# --------------------------------------------------------------------------------------
# CUDA C++ → one Python file that compiles it inline
# --------------------------------------------------------------------------------------

CUDA_FLAGS_LINE = re.compile(r"^[ \t]*//[ \t]*fk-cuda-?c?flags:(.*)$", re.M)
DEFAULT_CUDA_FLAGS = ("-O3", "--use_fast_math")
DEFAULT_LDFLAGS = ("-lcuda",)

#: The official evaluation image's CUTLASS header roots (background doc §2).
CUTLASS_INCLUDE_DIRS = ("/usr/local/cutlass/include",
                        "/usr/local/cutlass/tools/util/include")


def cuda_flags(source: str) -> list[str]:
    """nvcc flags a CUDA kernel file asks for: SOL's defaults plus its flag line."""
    extra = [f for line in CUDA_FLAGS_LINE.findall(source) for f in line.split()]
    flags = list(DEFAULT_CUDA_FLAGS)
    flags += [f for f in extra if f not in flags]
    return flags


def inline_quoted_headers(source: str, files: Mapping[str, str], *, seen: set[str] | None = None,
                          notes: list[str] | None = None) -> str:
    """Replace ``#include "x.cuh"`` with the file's text when ``x.cuh`` is among ``files``."""
    seen = seen if seen is not None else set()
    notes = notes if notes is not None else []
    by_name = {Path(p).name: p for p in files}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        path = by_name.get(name)
        if path is None or name in seen:
            return match.group(0)
        seen.add(name)
        notes.append("inlined header: " + name)
        body = inline_quoted_headers(files[path], files, seen=seen, notes=notes)
        return f"// ---- inlined from {name} ----\n{body}\n// ---- end {name} ----"

    return re.sub(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"[ \t]*$', repl, source, flags=re.M)


def cuda_to_inline(sources: Mapping[str, str], *, entry: str = "run", n_outputs: int = 0,
                   module_name: str | None = None, extra_cflags: Iterable[str] = (),
                   extra_ldflags: Iterable[str] = (), header: str = "",
                   cuda_flags_override: Iterable[str] | None = None) -> NormalizeResult:
    """A single-file ``submission.py`` from CUDA/C++ sources with a ``PYBIND11_MODULE``.

    The sources are embedded in one string, quoted ``#include``s among them are inlined, the
    ``PYBIND11_MODULE`` binding is used as-is (``load_inline(..., functions=None)``), and the
    gencode is built from ``torch.cuda.get_device_capability()`` at import. ``n_outputs`` is
    how many pre-allocated outputs arrive last (DPS): when the binding returns tensors they
    are copied into them.
    """
    if not sources:
        return unconvertible("no sources")
    notes: list[str] = []
    binding_owner = next((p for p, t in sources.items() if "PYBIND11_MODULE" in t), None)
    if binding_owner is None:
        return unconvertible(
            "no PYBIND11_MODULE binding in any source: the extension cannot be built "
            "without one (a cuda_cpp solution must carry its own torch binding)",
            files=list(sources))
    order = [p for p in sources if p != binding_owner] + [binding_owner]
    merged_parts: list[str] = []
    for path in order:
        text = inline_quoted_headers(sources[path], sources, notes=notes)
        merged_parts.append(f"// ================================ source: {path}\n{text}")
    merged = "\n\n".join(merged_parts)

    if cuda_flags_override is not None:
        flags = list(cuda_flags_override)
    else:
        flags = []
        for path in order:
            flags += [f for f in cuda_flags(sources[path]) if f not in flags]
    notes.append("nvcc flags: " + " ".join(flags))
    pinned = [f for f in flags if f.startswith(("-gencode", "-arch"))
              or f.startswith("--generate-code")]
    if pinned:
        flags = [f for f in flags if f not in pinned]
        notes.append("dropped pinned code-generation flags (" + " ".join(pinned) + "); the "
                     "generated wrapper derives the target from the running device")
    if len(order) > 1:
        notes.append(f"{len(order)} translation units merged; duplicate static symbols across "
                     f"them would collide")
    ldflags = list(extra_ldflags)
    if not any("cuda" in str(f) for f in ldflags):
        ldflags = [*DEFAULT_LDFLAGS, *ldflags]

    name = module_name or ("fk_seed_" + hashlib.sha256(merged.encode()).hexdigest()[:12])
    body = _INLINE_TEMPLATE.format(
        header=(header if header else ""),
        module_name=name,
        literal=string_literal(merged),
        cuda_flags=", ".join(repr(f) for f in flags),
        extra_cflags=", ".join(repr(f) for f in extra_cflags),
        extra_ldflags=", ".join(repr(f) for f in ldflags),
        include_dirs=", ".join(repr(p) for p in CUTLASS_INCLUDE_DIRS),
        n_outputs=int(n_outputs),
    )
    try:
        parse(body)
    except NormalizeError as exc:  # cannot happen by construction; fail loudly if it does
        return unconvertible(f"generated submission did not parse: {exc}", notes=notes,
                             files=list(sources))
    problems = static_rules_check(_wrapper_only(body))
    if problems:
        notes.append("static rules: " + "; ".join(problems))
    return NormalizeResult(status="ok", source=body, language="cuda-inline", entry=entry,
                           notes=notes, files=list(sources))


_INLINE_TEMPLATE = '''"""{header}
Generated by rsi_kernel_tune.klib.normalize.cuda_to_inline — a single-file submission that
compiles its embedded CUDA source once, at import, for the compute capability of the device
it runs on. Do not edit the embedded source by hand; edit the original and regenerate.
"""
import hashlib
import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = {literal}

_CUDA_FLAGS = [{cuda_flags}]
_EXTRA_CFLAGS = [{extra_cflags}]
_EXTRA_LDFLAGS = [{extra_ldflags}]
_INCLUDE_DIRS = [{include_dirs}]
_N_OUTPUTS = {n_outputs}


def _arch_flags():
    """The code-generation target of the device this runs on (never a literal)."""
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return ["-gencode=arch=compute_{{0}}{{1}}a,code=sm_{{0}}{{1}}a".format(major, minor)]


def _include_paths():
    """CUTLASS header roots of the official evaluation image that exist on this machine."""
    return [p for p in _INCLUDE_DIRS if os.path.isdir(p)]


def _build():
    digest = hashlib.sha256(_CUDA_SRC.encode()).hexdigest()[:12]
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    name = "{module_name}_" + digest + ("_" + "{{0}}{{1}}".format(*cc) if cc else "")
    flags = list(_CUDA_FLAGS) + _arch_flags()
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cflags=list(_EXTRA_CFLAGS),
        extra_cuda_cflags=flags,
        extra_ldflags=list(_EXTRA_LDFLAGS),
        extra_include_paths=_include_paths(),
        functions=None,
        with_cuda=True,
        verbose=bool(os.environ.get("FK_SEED_VERBOSE")),
    )


_MODULE = _build()


def run(*args):
    """Positional call into the extension; outputs last (DPS) are copied into if returned."""
    result = _MODULE.run(*args)
    if result is None or _N_OUTPUTS <= 0:
        return result
    outs = args[-_N_OUTPUTS:]
    if isinstance(result, (tuple, list)):
        for dst, src in zip(outs, result):
            dst.copy_(src)
    else:
        outs[0].copy_(result)
    return None
'''


def _wrapper_only(source: str) -> str:
    """The generated wrapper's own text: everything outside the embedded source literal."""
    parts = source.split("_CUDA_SRC = ", 1)
    if len(parts) != 2:
        return source
    head, rest = parts
    end = rest.find("\n\n_CUDA_FLAGS")
    return head + (rest[end:] if end >= 0 else rest)


# --------------------------------------------------------------------------------------
# helpers shared by the extractors
# --------------------------------------------------------------------------------------

def trap_line(text: str, *, max_lines: int = 160) -> str:
    """The first line of a variant's own header that records a trap or a dead end.

    Mechanically extracted (the first matching line, stripped of its comment marker) — the
    README/header's own words, not a summary.
    """
    for line in text.splitlines()[:max_lines]:
        stripped = line.strip().lstrip("#").lstrip("/").strip()
        if not stripped:
            continue
        if any(word in stripped for word in TRAP_WORDS):
            return stripped[:240]
    return ""


def license_of_text(text: str, *, max_lines: int = 60) -> str:
    """The license an SPDX tag or license header in ``text`` states, else ``""``.

    Reads only what the file says about itself: an explicit ``SPDX-License-Identifier``, an
    Apache 2.0 header, an MIT header or the BSD-3-Clause grant.
    """
    head = "\n".join(text.splitlines()[:max_lines])
    match = re.search(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)", head)
    if match:
        return match.group(1)
    if re.search(r"Licensed under the Apache License, Version 2\.0", head) \
            or re.search(r"Apache License\s*\n?\s*Version 2\.0", head):
        return "Apache-2.0"
    if re.search(r"Permission is hereby granted, free of charge", head):
        return "MIT"
    if re.search(r"Redistribution and use in source and binary forms", head):
        return "BSD-3-Clause"
    return ""
