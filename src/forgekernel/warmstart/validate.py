"""Is this file usable as the kernel a run starts from? (``--init-kernel``, pure functions)

A warmstart hands the student a ready-made implementation instead of an empty stub, so
the file must already be a legal kernel of the card it lands in. Nothing here runs the
file, imports it, or touches a GPU: the checks are static, so a bad choice fails at
``forge init`` time with every problem listed at once, not hours into a run.

Four checks, in the order they read:

1. **it parses** (a ``.py`` by ``ast``; a ``.cu`` by the binding / entry shape the card
   builds);
2. **it has a module-level ``run``** -- the card calls the entry by name, never by
   ``getattr`` on something a decorator returned;
3. **its signature survives the card's call**: the call passes the definition's inputs
   and (DPS) its pre-allocated outputs positionally, so ``run`` takes exactly that many
   positional parameters, or ``*args``; ``**kwargs`` is harmless, a required
   keyword-only parameter is not;
4. **it does not do what a submission must not** -- read a secret out of the
   environment, or write outside the workspace. This is the basic subset of the
   project's submission rule (00-background §9.7), not a security audit.

Language inference is here too: with no ``[task].language`` declared, the imports say
what the file is. A kernel that imports ``triton`` is a Triton kernel; one that imports
``cutlass`` is a CuTe DSL one; one that imports only the host stack (``torch``, the
stdlib, ``numpy``) is the Python host of the CuTe DSL mode -- which is also where an
inline-CUDA solution lives (``torch.utils.cpp_extension``). Anything else is refused
with the imports it saw, because guessing would put a kernel in a mode that cannot
build or export it.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# The task languages this build accepts (forgekernel.task.LANGUAGES at call time -- the
# triton task adds "triton"; until it merges, an inferred triton kernel is refused).
CUDA_SUFFIXES = (".cu", ".cuh")

# Imports that say nothing about the kernel's language: the host stack a Python kernel
# of either mode (and an inline-CUDA one) is allowed to use.
HOST_MODULES = frozenset({
    "abc", "argparse", "array", "builtins", "collections", "contextlib", "copy", "ctypes",
    "dataclasses", "enum", "functools", "glob", "hashlib", "importlib", "inspect",
    "itertools", "json", "logging", "math", "numbers", "operator", "os", "pathlib",
    "random", "re", "shutil", "statistics", "string", "struct", "subprocess", "sys",
    "tempfile", "textwrap", "threading", "time", "traceback", "types", "typing",
    "unittest", "uuid", "warnings", "weakref",
    # non-stdlib, still host-only: numeric plumbing and the tile/launcher helpers a
    # PyTorch kernel uses (an inline-CUDA solution imports these too)
    "numpy", "torch",
})

# A name that looks like a credential. Reading one is the basic forbidden thing a
# submission must not do (00-background §9.7 / §3.4); matching the name is enough here.
_SECRET_NAME = re.compile(
    r"(?i)(token|secret|passw|api[_-]?key|access[_-]?key|private[_-]?key|credential)")
_WRITE_MODES = frozenset("wax+")
_ENV_ATTRS = ("environ", "getenv")

# `torch::Tensor run(...)` / `void run(...)` in a .cu, and its parameter list.
_CPP_ENTRY = re.compile(r"(?:void|torch::Tensor|at::Tensor)\s+run\s*\(([^)]*)\)")
_CPP_BIND = re.compile(r"PYBIND11_MODULE\s*\(\s*TORCH_EXTENSION_NAME\s*,")


class LanguageError(ValueError):
    """The file's language cannot be told (or is not accepted by this build)."""


@dataclass(frozen=True)
class EntryCheck:
    """What the static check found. ``problems`` empty == the file may be the kernel."""

    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    language: str | None = None      # the [task] language it maps to
    method: str | None = None        # triton | cutedsl | pytorch | cuda
    signature: tuple[str, ...] = field(default=())   # run's positional parameters
    varargs: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems

    def report(self) -> str:
        """One line for the operator: what was accepted, and how it was read."""
        if self.language is None:
            return "; ".join(self.notes) or "no language inferred"
        how = "" if self.method in (None, self.language) else f" ({self.method} method)"
        sig = "run(*args)" if self.varargs else f"run({', '.join(self.signature)})"
        return f"language {self.language}{how}, entry {sig}"


# -- language -----------------------------------------------------------------------


def _imports(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level:
            mods.add("<relative>")
    return mods


def _language_of(mods: set[str], filename: str) -> tuple[str, str]:
    """(task language, method) for a parsed Python file, or LanguageError."""
    if "triton" in mods:
        return "triton", "triton"
    if "cutlass" in mods:
        return "cutedsl", "cutedsl"
    unknown = sorted(m for m in mods if m not in HOST_MODULES and not m.startswith("_"))
    if unknown:
        raise LanguageError(
            f"{Path(filename).name} imports {', '.join(unknown)} -- not a kernel language "
            f"this build accepts (recognised: triton, cutlass.cute, or a host-only "
            f"PyTorch kernel). A kernel that needs another package must be rewritten "
            f"against the allowed ones (PyTorch, Triton, CuTe DSL, CUTLASS) before it can "
            f"be a starting point")
    return "cutedsl", "pytorch"


def infer_language(source: str, *, filename: str = "kernel.py") -> tuple[str, str]:
    """(task language, method) from the source itself. ``LanguageError`` otherwise.

    ``filename``'s suffix decides the CUDA case: a ``.cu`` is built by the card as one
    translation unit with its own binding, so there is nothing to infer from imports."""
    if Path(filename).suffix in CUDA_SUFFIXES:
        return "cuda", "cuda"
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise LanguageError(f"{Path(filename).name} does not parse: {_syntax(e)}") from None
    return _language_of(_imports(tree), filename)


def check_language(language: str, *, languages: tuple[str, ...] | list[str]) -> list[str]:
    """Is this [task] language one the build accepts? (task.LANGUAGES at the call site)"""
    if language in languages:
        return []
    return [f"this build's [task] languages are {list(languages)}, so a {language} kernel "
            f"cannot be a task's kernel yet (pass --kernel-file to choose a file the "
            f"task can build, or wait for that language to merge)"]


# -- the entry ----------------------------------------------------------------------


def _syntax(e: SyntaxError) -> str:
    return f"syntax error at line {e.lineno}, column {e.offset}: {e.msg}"


def _top_level_run(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            return node
    return None


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[str], bool, bool]:
    """(positional-or-keyword names, has *args, has required keyword-only params)."""
    a = fn.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    # a keyword-only parameter without a default cannot be filled by a positional call
    required_kwonly = any(d is None for d in a.kw_defaults)
    return names, a.vararg is not None, required_kwonly


def _expected_arity(definition: dict | None, dps: bool) -> int | None:
    if not definition:
        return None
    try:
        n_in = len(definition["inputs"])
        n_out = len(definition["outputs"])
    except (KeyError, TypeError):
        return None
    return n_in + n_out if dps else n_in


def _arity_problem(names: list[str], expected: int, definition: dict | None,
                   dps: bool) -> list[str]:
    what = ("the card passes the definition's inputs, then its pre-allocated outputs "
            "(DPS), positionally" if dps else
            "the card passes the definition's inputs positionally")
    ins = list((definition or {}).get("inputs") or {})
    outs = list((definition or {}).get("outputs") or {})
    return [f"run takes {len(names)} positional parameter(s) but {what}: {expected} "
            f"({len(ins)} input(s){f' + {len(outs)} output(s)' if dps else ''}); "
            f"run({', '.join(names)}) is called as run({', '.join(ins + (outs if dps else []))})"]


def _entry_problems(source: str, *, filename: str, definition: dict | None,
                    dps: bool) -> tuple[list[str], list[str], tuple[str, ...], bool]:
    tree = ast.parse(source, filename=filename)     # caller parsed once already; cheap
    fn = _top_level_run(tree)
    if fn is None:
        nested = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "run"]
        where = (f" (line {nested[0].lineno}: defined, but not at module level)"
                 if nested else "")
        return ([f"no top-level def run(...){where}: the card calls the entry by name and "
                 f"position, so run must be a module-level function"], [], (), False)
    if isinstance(fn, ast.AsyncFunctionDef):
        return (["run is async: the card calls it synchronously and would never await it"],
                [], (), False)
    names, varargs, required_kwonly = _signature(fn)
    problems, notes = [], []
    if required_kwonly:
        problems.append("run has a required keyword-only parameter; the card fills "
                        "parameters by position only")
    expected = _expected_arity(definition, dps)
    if expected is None:
        notes.append("the card's problem.json gives no input/output counts: the arity was "
                     "not checked against the definition")
    elif not varargs and len(names) != expected:
        problems += _arity_problem(names, expected, definition, dps)
    if varargs:
        notes.append("run(*args): the definition's inputs and outputs arrive positionally, "
                     "unpacked from *args")
    return problems, notes, tuple(names), varargs


# -- what a submission must not do ---------------------------------------------------

def _literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr in _ENV_ATTRS
            and isinstance(node.value, ast.Name) and node.value.id == "os")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _mode_writes(explicit: str | None) -> bool:
    return explicit is not None and any(ch in explicit for ch in _WRITE_MODES)


def _outside_workspace(path: str) -> bool:
    """A literal path that leaves the workspace: absolute, or climbing with '..'."""
    return os.path.isabs(path) or ".." in Path(path).parts


def _env_get(call: ast.Call) -> bool:
    """Is this call os.getenv(...) or os.environ.get(...)?"""
    fn = call.func
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr == "getenv":
        return isinstance(fn.value, ast.Name) and fn.value.id == "os"
    if fn.attr == "get":
        return _is_environ(fn.value)
    return False


def _forbidden_python(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_environ(node.value):
            key = _literal(node.slice)
            if key and _SECRET_NAME.search(key):
                out.append(f"line {node.lineno}: reads the secret {key!r} from the "
                           f"environment; a submission may not read credentials")
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        args = list(node.args)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if _env_get(node):
            key = _literal(args[0] if args else kw.get("key"))
            if key and _SECRET_NAME.search(key):
                out.append(f"line {node.lineno}: reads the secret {key!r} from the "
                           f"environment; a submission may not read credentials")
        elif name == "open":
            path = _literal(args[0] if args else kw.get("file"))
            mode = _literal(args[1] if len(args) > 1 else kw.get("mode"))
            if path and _mode_writes(mode) and _outside_workspace(path):
                out.append(f"line {node.lineno}: writes {path!r}, outside the workspace; "
                           f"a submission writes only where it is run")
        elif name in ("write_text", "write_bytes", "write"):
            receiver = getattr(node.func, "value", None)
            if isinstance(receiver, ast.Call) and _call_name(receiver.func) in ("Path", "open"):
                literal = _literal(receiver.args[0] if receiver.args else None)
                if literal and _outside_workspace(literal):
                    out.append(f"line {node.lineno}: writes {literal!r}, outside the "
                               f"workspace; a submission writes only where it is run")
    return out


def _forbidden_cuda(source: str) -> list[str]:
    out: list[str] = []
    for i, line in enumerate(source.splitlines(), 1):
        text = line.split("//", 1)[0]
        m = re.search(r"getenv\s*\(\s*\"([^\"]+)\"", text)
        if m and _SECRET_NAME.search(m.group(1)):
            out.append(f"line {i}: reads the secret {m.group(1)!r} from the environment; "
                       f"a submission may not read credentials")
        m = re.search(r"fopen\s*\(\s*\"([^\"]+)\"", text)
        if m and _outside_workspace(m.group(1)):
            out.append(f"line {i}: writes {m.group(1)!r}, outside the workspace; a "
                       f"submission writes only where it is run")
    return out


def _cuda_entry(source: str, definition: dict | None, dps: bool
                ) -> tuple[list[str], list[str], tuple[str, ...]]:
    problems, notes = [], []
    if not _CPP_BIND.search(source):
        problems.append("no PYBIND11_MODULE(TORCH_EXTENSION_NAME, m): the card builds this "
                        "file as its own module and expects the binding")
    m = _CPP_ENTRY.search(source)
    if m is None:
        return problems + ["no `void run(...)` / `torch::Tensor run(...)` definition"], notes, ()
    names = tuple(re.split(r"[\s&*]+", p.strip())[-1]
                  for p in m.group(1).split(",") if p.strip())
    expected = _expected_arity(definition, dps)
    if expected is None:
        notes.append("the card's problem.json gives no input/output counts: the arity was "
                     "not checked against the definition")
    elif len(names) != expected:
        problems += _arity_problem(list(names), expected, definition, dps)
    return problems, notes, names


# -- the whole check -----------------------------------------------------------------


def check_source(source: str, *, filename: str = "kernel.py", definition: dict | None = None,
                 dps: bool = True, languages: tuple[str, ...] | list[str] | None = None
                 ) -> EntryCheck:
    """Everything above in one pass; ``languages`` defaults to this build's LANGUAGES."""
    notes: list[str] = []
    problems: list[str] = []
    signature: tuple[str, ...] = ()
    varargs = False
    try:
        language, method = infer_language(source, filename=filename)
    except LanguageError as e:
        return EntryCheck(problems=(str(e),), notes=())
    if languages is None:
        from forgekernel.task import LANGUAGES
        languages = LANGUAGES
    problems += check_language(language, languages=languages)
    if Path(filename).suffix in CUDA_SUFFIXES:
        p, n, signature = _cuda_entry(source, definition, dps)
        problems += p
        notes += n
        problems += _forbidden_cuda(source)
    else:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return EntryCheck(problems=(f"{Path(filename).name}: {_syntax(e)}",), notes=())
        p, n, signature, varargs = _entry_problems(source, filename=filename,
                                                   definition=definition, dps=dps)
        problems += p
        notes += n
        problems += _forbidden_python(tree)
    return EntryCheck(problems=tuple(problems), notes=tuple(notes), language=language,
                      method=method, signature=signature, varargs=varargs)


def check_file(path: str | Path, *, definition: dict | None = None, dps: bool = True,
               languages: tuple[str, ...] | list[str] | None = None) -> EntryCheck:
    """``check_source`` for a file on disk; a missing / undecodable file is a problem, not
    an exception (the caller lists it with the rest)."""
    p = Path(path).expanduser()
    try:
        raw = p.read_bytes()
    except OSError as e:
        return EntryCheck(problems=(f"{p}: cannot be read ({e.strerror or e})",))
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return EntryCheck(problems=(f"{p.name}: not UTF-8 text ({e.reason} at byte "
                                    f"{e.start})",))
    return check_source(source, filename=p.name, definition=definition, dps=dps,
                        languages=languages)
