"""Static review of a submission's source, before it is exported or trusted.

The loop is rewarded by a number, and a number can be gamed -- the board's own
Prohibited Behavior list is a catalogue of the ways (patching the timing function,
hiding work on a stream outside the measured window, starting threads, returning a lazy
proxy instead of a tensor, changing global flags so the reference slows down).  The
runtime checks in ``eval/leaderboard.py`` catch most of them by what they do; this
module catches the ones that leave a mark in the source, and catches them *before* a
file goes to the board.

Three severities, the words the export guard uses:

``block``  a submission carrying this must not be exported (``forge export`` refuses
           unless ``--allow-integrity-findings`` is given; campaign's compose refuses
           outright).  Each rule is written to have no legitimate reading in a kernel
           submission: an assignment to ``os.environ``, a patch on an imported module's
           attribute, a thread started from the kernel.
``warn``   legal, but it is the shape a hack takes (``torch.cuda.CUDAGraph``, a
           non-default stream, a wall clock read on the kernel's path, ``ctypes``).  A
           human should look; the fresh-input recheck in the card is what actually
           covers the graph case.
``info``   worth a glance, no verdict (an inlined CUDA build via
           ``torch.utils.cpp_extension``, a capability branch, ``torch.compile``).

False positives cost more than misses here: a rule that fires on ordinary kernels is a
rule the team learns to ignore, and ``--allow-integrity-findings`` becomes the default.
Every rule is written against a real positive *and* a real negative (the repo's own
``kernels/`` tree is the negative corpus -- see ``docs/integrity.md``), and the
heuristics (a cache keyed by ``data_ptr()``, a module-level call counter) are narrow by
construction.

This module must stay importable without torch: it reads source text, never executes it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

BLOCK = "block"
WARN = "warn"
INFO = "info"
SEVERITIES = (BLOCK, WARN, INFO)

# The board's upload limit for one file (a collection archive allows 10 MB per file).
MAX_SUBMISSION_BYTES = 10 * 1024 * 1024

# Rule ids: the name is the contract (the report, the docs and the tests cite them).
ENV_ASSIGN = "env-assign"
FLAG_ASSIGN = "torch-flag-assign"
FLAG_RESTORED = "torch-flag-restored"
MONKEYPATCH = "monkeypatch"
TIMING_PATCH = "timing-function-patch"
THREADS = "thread-or-process"
EVAL_PATHS = "eval-path-read"
POINTER_CACHE = "pointer-keyed-cache"
CALL_COUNT = "call-count-branch"
PROXY_RETURN = "proxy-return"
NO_RUN = "no-top-level-run"
SYNTAX_ERROR = "syntax-error"
TOO_LARGE = "too-large"
ABSOLUTE_PATH = "absolute-path"
PATH_TRAVERSAL = "path-traversal"
SECRETS = "credential-access"
CUDA_GRAPH = "cuda-graph"
WALL_CLOCK = "wall-clock"
EVENT_CONSTRUCT = "event-construct"
CTYPES = "ctypes-or-cupti"
NON_DEFAULT_STREAM = "non-default-stream"
INLINE_CUDA = "inline-cuda"
TORCH_COMPILE = "torch-compile"
ENV_READ = "env-read"

# ``torch.cuda.Event.elapsed_time`` and the clocks around it: patching any of these is the
# first entry on the board's prohibited list.
_TIMING_TARGETS = frozenset({
    "torch.cuda.Event.elapsed_time", "torch.cuda.Event.elapsed_time_ns",
    "torch.cuda.Event.record", "torch.cuda.Event.query", "torch.cuda.synchronize",
    "torch.cuda.current_stream", "torch.profiler.profile",
    "torch.autograd.profiler.profile", "time.perf_counter", "time.perf_counter_ns",
    "time.time", "time.time_ns", "time.monotonic", "time.monotonic_ns",
    "time.process_time", "time.process_time_ns",
})
_TIMING_ATTRS = frozenset({"elapsed_time", "elapsed_time_ns", "perf_counter",
                           "perf_counter_ns", "monotonic", "monotonic_ns", "synchronize"})

_THREAD_CALLS = frozenset({
    "threading.Thread", "threading.Timer", "_thread.start_new_thread",
    "thread.start_new_thread", "multiprocessing.Process", "multiprocessing.Pool",
    "multiprocessing.pool.Pool", "concurrent.futures.ThreadPoolExecutor",
    "concurrent.futures.ProcessPoolExecutor", "os.fork", "os.forkpty", "os.posix_spawn",
    "subprocess.Popen",
})

_STREAM_CALLS = frozenset({
    "torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.set_stream",
    "torch.cuda.ExternalStream",
})

_WALL_CLOCK_CALLS = frozenset({
    "time.perf_counter", "time.perf_counter_ns", "time.time", "time.time_ns",
    "time.monotonic", "time.monotonic_ns", "time.process_time", "time.process_time_ns",
})

_SECRET_ENV_CALLS = frozenset({"os.environ.get", "os.getenv", "os.environ.pop",
                               "os.environ.setdefault", "os.environ.__getitem__"})

# A module whose attribute a submission has no business rewriting is recognised from the
# file's own `import` statements (see `_imports`): a from-imported name is a local
# binding, and rebinding it patches nothing.

# Environment-variable names a submission has no business reading or writing: they carry
# the sealed reference's key, or they name the harness's own switches.
_SECRET_ENV_RE = re.compile(
    r"\bFK_REF_KEY\b|\bSOL_EXECBENCH_TOKEN\b|\bHF_TOKEN\b|\bAWS_SECRET\w*\b"
    r"|\bapi[_-]?key\b|\bpassword\b|\bsecret[_-]?key\b"
    r"|\.ssh[\\/]|\bid_rsa\b|\.netrc\b|\.aws[\\/]", re.I)

# Paths that belong to the evaluation harness, not to a kernel.
_EVAL_PATH_RE = re.compile(
    r"(?:^|[\\/])(?:eval|ref_bin)(?:[\\/]|$)|submit\.local\.toml|best_confirmation"
    r"|problem\.json|\.artifacts[\\/]", re.I)

# Absolute paths are blocked unless they name a toolchain directory: an inlined CUDA
# source may legitimately `#include "/usr/local/cutlass/include/..."` or pass `-I/opt/...`
# (the job image keeps CUTLASS at a system path).  A path into a home directory, a
# working tree or a temp dir has no legitimate reading in a submission.
_ABS_PATH_RE = re.compile(
    r"(?:^|[\"'])(?:/(?:home|root|tmp|mnt|media|srv|data|workspace|Users|var|proc|sys)/"
    r"|[A-Za-z]:[\\/])")
_TOOLCHAIN_PREFIXES = ("/usr/", "/opt/", "/usr/local/")

# A literal that looks like a path with a `..` segment (traversal out of the card).
_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")

_PREPROCESSOR_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|"
                              r"include|pragma|error|warning|line)\b")


@dataclass(frozen=True)
class Finding:
    """One rule hit: which rule, how bad, where, and the line it fired on."""
    rule: str
    severity: str
    file: str = ""
    line: int = 0
    snippet: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"rule": self.rule, "severity": self.severity, "file": self.file,
                "line": self.line, "snippet": self.snippet, "detail": self.detail}

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}" if self.file and self.line else \
            (self.file or (f"line {self.line}" if self.line else ""))
        head = f"[{self.severity}] {self.rule}"
        extra = f" ({self.detail})" if self.detail else ""
        body = self.snippet.strip()[:120]
        return f"{head}: {where}: {body}{extra}" if where else f"{head}: {body}{extra}"


def blocks(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == BLOCK]


def warn_or_worse(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity in (BLOCK, WARN)]


def summarize(findings: list[Finding]) -> str:
    counts = {BLOCK: 0, WARN: 0, INFO: 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return (f"{counts[BLOCK]} block, {counts[WARN]} warn, {counts[INFO]} info "
            f"({len(findings)} finding(s))")


# -- reading the source ----------------------------------------------------------------

def _attr_path(node: ast.AST | None) -> str | None:
    """``a.b.c`` for an attribute chain rooted at a name, else None."""
    if node is None:
        return None
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _imports(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """(module bindings, from-import bindings), as ``local name -> dotted path``.

    ``import torch`` / ``import torch.cuda as tc`` bind a *module*: assigning
    ``torch.cuda.Event.elapsed_time`` writes into that module.  ``from m import n`` binds
    a *name*: assigning ``n`` rebinds the local name and patches nothing, which is why the
    two are kept apart (``_T = ...`` after ``from cutlass._mlir.extras import types as _T``
    is ordinary code and must not read as a monkey patch).
    """
    modules: dict[str, str] = {}
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    modules[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, names


def _resolve(node: ast.AST | None, imports: dict[str, str]) -> str | None:
    """The dotted path an expression names, with the first component's import resolved."""
    path = _attr_path(node)
    if path is None:
        return None
    root, _, rest = path.partition(".")
    base = imports.get(root)
    if base is None:
        return path
    return f"{base}.{rest}" if rest else base


def _module_root(path: str | None, modules: dict[str, str]) -> str | None:
    """The module a dotted path starts in, when the root is a module *import*.

    A from-imported name is a local binding: ``_T = ...`` after
    ``from cutlass._mlir.extras import types as _T`` writes nothing outside this file.
    """
    if not path:
        return None
    parts = path.split(".")
    if len(parts) < 2:
        return None
    return modules.get(parts[0])


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers covered by a docstring (module / class / function / method).

    A docstring may name a path or an environment variable -- it documents the kernel.  A
    string literal anywhere else is data the program uses, and a rule about paths applies.
    """
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            start = first.value.lineno
            end = getattr(first.value, "end_lineno", start) or start
            covered.update(range(start, end + 1))
    return covered


def _looks_like_a_source_blob(text: str) -> bool:
    """An embedded CUDA / Triton source: multi-line, or carrying preprocessor lines."""
    return "\n" in text or bool(_PREPROCESSOR_RE.match(text))


# -- the linter ------------------------------------------------------------------------

class _Linter(ast.NodeVisitor):
    def __init__(self, source: str, filename: str, submission: bool = True):
        self.source = source
        self.filename = filename
        self.submission = submission
        self.lines = source.splitlines()
        self.findings: list[Finding] = []
        self.imports: dict[str, str] = {}
        self.modules: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.scope: list[str] = []                  # enclosing function names
        self._scope_nodes: list[ast.AST] = []        # the functions themselves
        self.flag_writes: dict[tuple[str, str], list[int]] = {}
        self.global_names: set[str] = set()
        self.classes: dict[str, ast.ClassDef] = {}
        self.run_node: ast.AST | None = None
        self.prose: set[int] = set()

    # -- recording --------------------------------------------------------------------
    def add(self, rule: str, severity: str, line: int, detail: str = "") -> None:
        snippet = self.lines[line - 1] if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(rule=rule, severity=severity, file=self.filename,
                                     line=line, snippet=snippet.strip()[:200], detail=detail))

    # -- module -----------------------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        self.modules, self.names = _imports(node)
        self.imports = {**self.names, **self.modules}
        self.prose = _docstring_lines(node)
        self.classes = {n.name: n for n in ast.walk(node) if isinstance(n, ast.ClassDef)}
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "run":
                self.run_node = stmt
        for stmt in node.body:
            self.visit(stmt)
        self._classify_flag_writes()
        self._literal_rules(node)
        self._graph_rule(node)
        self._return_rules()
        if self.submission and self.run_node is None:
            self.add(NO_RUN, BLOCK, 1,
                     "a collection submission needs a top-level run() (the board's entry)")

    # -- scopes -----------------------------------------------------------------------
    def _visit_function(self, node) -> None:
        self.scope.append(node.name)
        self._scope_nodes.append(node)
        locals_before = set(self.global_names)
        for stmt in node.body:
            self.visit(stmt)
        # decorators / defaults may hold calls (e.g. torch.compile)
        for expr in (*node.decorator_list, *node.args.defaults,
                     *[d for d in node.args.kw_defaults if d is not None]):
            self.visit(expr)
        self._call_count_rule(node, set(self.global_names) - locals_before)
        self._scope_nodes.pop()
        self.scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            self.visit(base)
        for stmt in node.body:
            self.visit(stmt)

    # -- statements -------------------------------------------------------------------
    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.startswith("torch.utils.cpp_extension"):
                self.add(INLINE_CUDA, INFO, node.lineno,
                         "inlined CUDA build: legal for a collection submission")
            elif alias.name.split(".")[0] in ("ctypes", "cffi", "_ctypes"):
                self.add(CTYPES, WARN, node.lineno,
                         f"imports {alias.name}: a native escape hatch the harness cannot "
                         f"see into")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod.startswith("torch.utils.cpp_extension"):
            self.add(INLINE_CUDA, INFO, node.lineno,
                     "inlined CUDA build: legal for a collection submission")
        elif mod.split(".")[0] in ("ctypes", "cffi", "_ctypes"):
            self.add(CTYPES, WARN, node.lineno, f"from {mod} import ...: a native escape hatch")

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._assignment_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assignment_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._assignment_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            path = _resolve(target.value, self.imports) if isinstance(target, ast.Subscript) else None
            if path == "os.environ":
                self.add(ENV_ASSIGN, BLOCK, node.lineno,
                         "removing an environment variable the harness set")
        self.generic_visit(node)

    def _assignment_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assignment_target(elt, lineno)
            return
        if isinstance(target, ast.Subscript):
            base = _resolve(target.value, self.imports)
            if base == "os.environ":
                self.add(ENV_ASSIGN, BLOCK, lineno,
                         "the evaluation owns its environment; a kernel may not set it")
            elif base == "sys.modules":
                self.add(MONKEYPATCH, BLOCK, lineno, "replacing an entry in sys.modules")
            return                              # visit_Subscript reads the key
        raw = _attr_path(target)
        path = _resolve(target, self.imports)
        if path is None:
            return
        if path.startswith("os.environ"):
            self.add(ENV_ASSIGN, BLOCK, lineno,
                     "the evaluation owns its environment; a kernel may not set it")
            return
        if path == "sys.modules":
            self.add(MONKEYPATCH, BLOCK, lineno, "replacing sys.modules")
            return
        module_root = _module_root(raw, self.modules)
        if (module_root and path in _TIMING_TARGETS) or raw in _TIMING_TARGETS:
            self.add(TIMING_PATCH, BLOCK, lineno,
                     "the board's prohibited list: patching a timing function")
            return
        if module_root and raw.split(".")[-1] in _TIMING_ATTRS:
            self.add(TIMING_PATCH, BLOCK, lineno,
                     "the board's prohibited list: patching a timing function")
            return
        if path.startswith("torch.backends.") or path == "torch.set_float32_matmul_precision":
            key = (self.scope[-1] if self.scope else "", path)
            self.flag_writes.setdefault(key, []).append(lineno)
            return
        # a patch on an imported module is `module.attr = ...` written on the module
        # object: the root must be a module *import*, and the path must have an attribute
        if module_root:
            self.add(MONKEYPATCH, BLOCK, lineno,
                     f"assigning {raw}: a patch on an imported module's attribute")

    def visit_Call(self, node: ast.Call) -> None:
        self._call_rules(node)
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------------------
    def _call_rules(self, node: ast.Call) -> None:
        path = _resolve(node.func, self.imports)
        name = _attr_path(node.func)
        line = node.lineno
        target = path or name
        if target is None:
            return
        if target in ("os.putenv", "os.unsetenv"):
            self.add(ENV_ASSIGN, BLOCK, line, "the evaluation owns its environment")
        elif target in ("os.environ.update", "os.environ.setdefault", "os.environ.pop",
                        "os.environ.clear"):
            self.add(ENV_ASSIGN, BLOCK, line, "the evaluation owns its environment")
        elif target == "torch.set_float32_matmul_precision":
            key = (self.scope[-1] if self.scope else "", target)
            self.flag_writes.setdefault(key, []).append(line)
        elif target in _THREAD_CALLS:
            self.add(THREADS, BLOCK, line,
                     f"{target}: the board's prohibited list (threads or processes during "
                     f"evaluation)")
        elif target in _WALL_CLOCK_CALLS:
            self.add(WALL_CLOCK, WARN, line, f"{target} on the kernel's path")
        elif target in _STREAM_CALLS:
            self.add(NON_DEFAULT_STREAM, WARN, line,
                     f"{target}: legal for overlap; a per-call span clock that ignored streams "
                     f"would miss the work (the card's is CUPTI, which does not)")
        elif target == "torch.cuda.Event":
            self.add(EVENT_CONSTRUCT, WARN, line, "the kernel builds its own CUDA event")
        elif target == "torch.utils.cpp_extension.load_inline":
            self.add(INLINE_CUDA, INFO, line, "inlined CUDA build: legal")
        elif target == "torch.utils.cpp_extension.load":
            self.add(INLINE_CUDA, INFO, line,
                     "an extension build: legal, keep it inside run()'s first call")
        elif target in ("torch.compile", "torch._dynamo.optimize"):
            self.add(TORCH_COMPILE, INFO, line, "torch.compile in the submission")
        elif target == "importlib.reload":
            self.add(MONKEYPATCH, BLOCK, line, "reloading a module during evaluation")
        elif target in _SECRET_ENV_CALLS:
            key = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) else None
            if key and _SECRET_ENV_RE.search(key):
                self.add(SECRETS, BLOCK, line, f"reads {key!r}")
            elif key:
                self.add(ENV_READ, INFO, line, f"reads {key!r} at run time")
        self._setattr_rule(node, target)

    def _setattr_rule(self, node: ast.Call, target: str | None) -> None:
        if target in ("setattr", "builtins.setattr") and len(node.args) >= 2:
            obj = _resolve(node.args[0], self.imports)
            attr = node.args[1].value if isinstance(node.args[1], ast.Constant) \
                and isinstance(node.args[1].value, str) else None
            if obj and attr:
                if attr in _TIMING_ATTRS or f"{obj}.{attr}" in _TIMING_TARGETS:
                    self.add(TIMING_PATCH, BLOCK, node.lineno,
                             f"setattr({obj}, {attr!r}): patching a timing function")
                elif _module_root(obj, self.modules):
                    self.add(MONKEYPATCH, BLOCK, node.lineno,
                             f"setattr({obj}, {attr!r}) on an imported module")

    def _call_count_rule(self, fn: ast.AST, names: set[str]) -> None:
        """``global N; N += 1; if N ...`` -- behaviour that depends on the call number.

        Narrow on purpose: the name must be declared ``global`` *in this function* and be
        both incremented and read in a branch.  A kernel that keeps a module-level table of
        compiled variants does none of that.
        """
        for name in sorted(names):
            incr = [n for n in ast.walk(fn)
                    if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                    and n.target.id == name]
            if not incr:
                continue
            branched = any(isinstance(n, (ast.Compare, ast.IfExp)) and
                           any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(n))
                           for n in ast.walk(fn))
            if branched:
                self.add(CALL_COUNT, BLOCK, incr[0].lineno,
                         f"global {name} counts calls and changes the result: the board "
                         f"calls run() many times and expects the same answer")

    # -- subscripts -------------------------------------------------------------------
    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._pointer_key(node, node.lineno)
        self.generic_visit(node)

    def _pointer_key(self, node: ast.Subscript, line: int) -> None:
        for part in ast.walk(node.slice):
            if isinstance(part, ast.Call):
                called = _attr_path(part.func) or ""
                if called.endswith("data_ptr"):
                    self.add(POINTER_CACHE, BLOCK, line,
                             "a container keyed by data_ptr(): a cache that answers by address")
                    return
                if called == "id":
                    self.add(POINTER_CACHE, BLOCK, line,
                             "a container keyed by id(): a cache that answers by identity")
                    return

    def visit_Return(self, node: ast.Return) -> None:
        self.generic_visit(node)

    # -- after the walk ---------------------------------------------------------------
    def _classify_flag_writes(self) -> None:
        """A torch.backends write is allowed only where it is restored.

        In a function a flag may be raised if the same flag is written again in the same
        function (the restore) -- a pair is what a well-behaved kernel looks like, and the
        report says so as ``info``.  At module level there is no ``run()`` return to
        restore before, so the flag is still set when the harness reads its reference.
        """
        for (scope, path), lines in sorted(self.flag_writes.items(),
                                           key=lambda kv: kv[1][0]):
            line = lines[0]
            if scope == "":
                self.add(FLAG_ASSIGN, BLOCK, line,
                         f"{path}: set at import time, so it is still set when the harness "
                         f"runs its own reference")
            elif len(lines) >= 2:
                self.add(FLAG_RESTORED, INFO, line,
                         f"{path}: written {len(lines)}x in {scope}() -- the second write is "
                         f"the restore; keep it in a finally")
            else:
                self.add(FLAG_ASSIGN, BLOCK, line,
                         f"{path} is written in {scope}() and never restored before the call "
                         f"returns")

    def _literal_rules(self, tree: ast.AST) -> None:
        seen: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            line = node.lineno
            if line in self.prose or line in seen:
                continue
            text = node.value
            if len(text) > 4096 or _looks_like_a_source_blob(text):
                # an embedded kernel source: its own comments and include paths are not the
                # submission's file access.  (targethw's rules read it for arch / SM issues.)
                continue
            seen.add(line)
            if _SECRET_ENV_RE.search(text):
                self.add(SECRETS, BLOCK, line, "names a credential-like variable or key")
            if not self.submission:
                continue
            if _EVAL_PATH_RE.search(text):
                self.add(EVAL_PATHS, BLOCK, line,
                         "names a path inside the evaluation harness (eval/, ref_bin/, "
                         "problem.json, submit.local.toml)")
            if _TRAVERSAL_RE.search(text) and ("/" in text or "\\" in text):
                self.add(PATH_TRAVERSAL, BLOCK, line, f"a `..` path segment: {text[:60]!r}")
            if _ABS_PATH_RE.search(text) and not text.startswith(_TOOLCHAIN_PREFIXES):
                self.add(ABSOLUTE_PATH, BLOCK, line,
                         f"an absolute path: {text[:60]!r} (a submission must not reach "
                         f"outside its own directory)")

    def _graph_rule(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            name = getattr(node, "id", None) or getattr(node, "attr", None)
            if name == "CUDAGraph":
                self.add(CUDA_GRAPH, WARN, node.lineno,
                         "a CUDA graph: legal, and the reason the card rechecks with fresh "
                         "inputs (a stale capture matches an old input)")
                return
            if name in ("capture_begin", "graph_pool_handle"):
                self.add(CUDA_GRAPH, WARN, node.lineno, "a CUDA graph capture")
                return

    def _return_rules(self) -> None:
        """A returned object that is not a tensor: the board rejects a lazy proxy."""
        run = self.run_node
        if run is None:
            return
        proxies = {name for name, cls in self.classes.items()
                   if not _subclasses_tensor(cls, self.imports)}
        for node in ast.walk(run):
            if not isinstance(node, ast.Return):
                continue
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                    and value.func.id in proxies:
                self.add(PROXY_RETURN, BLOCK, node.lineno,
                         f"run() returns an instance of {value.func.id}, not a torch.Tensor")
            elif isinstance(value, ast.Name) and value.id in proxies:
                self.add(PROXY_RETURN, BLOCK, node.lineno,
                         f"run() returns {value.id}, not a torch.Tensor")
        for name, cls in self.classes.items():
            hooks = {n.name for n in cls.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if hooks & {"__torch_function__", "__torch_dispatch__"} and \
                    _instantiated_in(run, name):
                self.add(PROXY_RETURN, BLOCK, cls.lineno,
                         f"{name} wraps torch calls (__torch_*__) and run() builds one: a "
                         f"lazy proxy, which the board rejects")


def _subclasses_tensor(cls: ast.ClassDef, imports: dict[str, str]) -> bool:
    for base in cls.bases:
        path = _resolve(base, imports) or _attr_path(base) or ""
        if path.split(".")[-1] in ("Tensor", "Parameter"):
            return True
    return False


def _instantiated_in(scope: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
               for n in ast.walk(scope))


# -- the entry points ------------------------------------------------------------------

def lint_source(source: str, filename: str = "<submission>", *,
                submission: bool = True) -> list[Finding]:
    """Every finding for one file's source, in line order.  Never raises on bad source.

    ``submission`` says whether the file is the file the board receives: only then are the
    shape rules applied (a top-level ``run``, the 10 MB limit) and only then do the path
    rules matter.  A RSI-Kernel-Tune ``kernel.py`` from the loop is linted with
    ``submission=False`` -- its entry point and its surrounding tree are the loop's.
    """
    findings: list[Finding] = []
    if submission:
        size = len(source.encode("utf-8"))
        if size > MAX_SUBMISSION_BYTES:
            findings.append(Finding(rule=TOO_LARGE, severity=BLOCK, file=filename, line=0,
                                    snippet=f"{size} bytes",
                                    detail=f"the board accepts {MAX_SUBMISSION_BYTES} bytes "
                                           f"per file"))
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        findings.append(Finding(rule=SYNTAX_ERROR, severity=BLOCK, file=filename,
                                line=int(e.lineno or 0), snippet=str(e.msg or ""),
                                detail="the file does not parse"))
        return findings
    linter = _Linter(source, filename, submission=submission)
    linter.visit(tree)
    seen: set[tuple[str, int]] = set()
    for f in linter.findings:
        key = (f.rule, f.line)
        if key in seen:
            continue
        seen.add(key)
        findings.append(f)
    return sorted(findings, key=lambda f: (f.line, f.rule))


def lint_texts(texts: dict[str, str], *, submission: bool = True) -> list[Finding]:
    """Every finding over a multi-file submission (path -> source)."""
    out: list[Finding] = []
    for name, text in texts.items():
        out += lint_source(text, filename=str(name), submission=submission)
    return out


def lint_paths(paths, *, submission: bool = False) -> list[Finding]:
    """``lint_texts`` over files (or directories) on disk.

    Defaults to ``submission=False``: the caller is reviewing a tree of kernels (the
    repo's ``kernels/``), not one file about to be uploaded.  A ``.cu`` / ``.h`` file is
    not Python and is skipped -- a collection submission is a single ``.py``.
    """
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*.py") if q.is_file())
        elif p.is_file():
            files.append(p)
    texts: dict[str, str] = {}
    for p in files:
        if p.suffix != ".py":
            continue
        try:
            texts[str(p)] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return lint_texts(texts, submission=submission)


def lint_file(path, *, submission: bool = True) -> list[Finding]:
    p = Path(path)
    try:
        return lint_source(p.read_text(encoding="utf-8", errors="replace"), str(p),
                           submission=submission)
    except OSError as e:
        return [Finding(rule="unreadable", severity=BLOCK, file=str(p), line=0,
                        snippet=type(e).__name__, detail=str(e)[:200])]
