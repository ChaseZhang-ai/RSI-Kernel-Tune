"""Does this submission compile for the target GPU? (asked without a target GPU.)

The board recompiles every submission on a B200; a submission that only builds for the
development architecture is a zero that costs a whole track submission to discover. This
module answers the question on the development machine:

``cuda-inline``  the CUDA C++ strings inside a ``submission.py`` are extracted and compiled
                 with the board's flags -- ``-gencode arch=compute_100a,code=sm_100a -O3
                 --use_fast_math`` plus CUTLASS / torch includes -- and **not run** (no B200
                 is needed to know whether it builds for sm_100a).
``triton``       each ``@triton.jit`` kernel is cross-compiled with ``triton.compile``
                 against an explicit ``GPUTarget("cuda", 100, 32)``: Triton's own code
                 generator and ptxas decide, so a construct the target cannot express fails
                 here rather than on the board.
``cutedsl``      a CuTe DSL kernel needs a device of the target architecture to compile, so
                 this path reports ``unknown`` with that reason when there is none, and
                 checks statically instead (see the report's limitations).
``static``       the rules of ``targethw.rules``: an architecture literal, an SM-count
                 literal, a gencode naming only the development architecture.

Every conclusion is ``ok`` / ``fail: <the compiler's own words>`` / ``unknown: <why>``. Only
``fail`` is a verdict about the submission; ``unknown`` is a verdict about this machine
(no nvcc, no Triton, no target device), and it always says which.

Compiling happens on a copy in a temporary directory: the submission is read, never imported
(importing it would run its module-level code) and never modified.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.targethw import (
    TARGET_ARCH,
    cc_of_arch,
    compute_arch,
    cuda_arch,
    find_cutlass,
    find_nvcc,
    include_paths,
    rules,
    torch_cuda_defines,
    torch_include_paths,
)

OK = "ok"
FAIL = "fail"
UNKNOWN = "unknown"

KERNEL_FILE_NAMES = ("submission.py", "kernel.py", "kernel.cu")
PROBLEM_FILES = ("eval/leaderboard/problem.json", "leaderboard/problem.json")


class CompileCheckError(ValueError):
    """The input is not something this check can read (a missing path, no kernel in it)."""


@dataclass(frozen=True)
class Conclusion:
    target: str            # cuda-inline | triton | cutedsl | static
    status: str            # ok | fail | unknown
    detail: str = ""
    line: int = 0

    def to_dict(self) -> dict:
        return {"target": self.target, "status": self.status, "detail": self.detail,
                "line": self.line}

    def __str__(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"{self.status:7s} {self.target}{where}: {self.detail}"


@dataclass
class Report:
    source: str
    arch: str
    conclusions: list[Conclusion] = field(default_factory=list)
    findings: list[rules.Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No ``fail``: unknowns are not failures (they are this machine's limits)."""
        return not any(c.status == FAIL for c in self.conclusions)

    @property
    def unknown(self) -> list[Conclusion]:
        return [c for c in self.conclusions if c.status == UNKNOWN]

    @property
    def failed(self) -> list[Conclusion]:
        return [c for c in self.conclusions if c.status == FAIL]

    def exit_code(self) -> int:
        return 0 if self.ok else 2

    def to_dict(self) -> dict:
        return {"source": self.source, "arch": self.arch, "ok": self.ok,
                "conclusions": [c.to_dict() for c in self.conclusions],
                "findings": [f.to_dict() for f in self.findings], "notes": self.notes,
                "targets": sorted({c.target for c in self.conclusions})}

    def lines(self) -> list[str]:
        out = [f"{self.source}: {rules.summarize(self.findings)}"]
        out += [str(c) for c in self.conclusions]
        out += [f"note: {n}" for n in self.notes]
        return out


# -- reading the submission -------------------------------------------------------------

def read_source(path: str | Path) -> tuple[str, str]:
    """``(label, source)`` for a submission file or a task directory."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = _find_in_dir(p)
    try:
        return str(p), p.read_text(encoding="utf-8")
    except OSError as e:
        raise CompileCheckError(f"{p}: {e.strerror or e}") from None


def _find_in_dir(d: Path) -> Path:
    for name in KERNEL_FILE_NAMES:
        if (d / name).is_file():
            return d / name
    for rel in PROBLEM_FILES:
        problem_file = d / rel
        if not problem_file.is_file():
            continue
        try:
            problem = json.loads(problem_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        kernel_file = problem.get("kernel_file")
        if kernel_file and (d / kernel_file).is_file():
            return d / kernel_file
    raise CompileCheckError(
        f"{d}: no submission to check (looked for {', '.join(KERNEL_FILE_NAMES)} and the "
        f"kernel_file of {PROBLEM_FILES[0]})")


def cuda_sources_in(source: str) -> list[tuple[int, str]]:
    """The CUDA C++ strings inside a submission, as ``(line, text)``.

    A string holding ``__global__`` is CUDA source: that is how a submission embeds it
    (``load_inline(cuda_sources=...)``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and "__global__" in node.value]


def triton_kernels_in(source: str) -> list[tuple[str, int]]:
    """``(name, line)`` of every ``@triton.jit`` function defined at any level."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [(node.name, node.lineno) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any((d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")) == "jit"
                    for d in node.decorator_list)]


# -- cuda-inline ------------------------------------------------------------------------

def _has_cutlass_include(text: str) -> bool:
    return "cutlass" in text or "cute/" in text


def check_cuda_source(text: str, *, arch: str = TARGET_ARCH, timeout: int = 900,
                      workdir: Path | None = None, line: int = 0) -> Conclusion:
    """Compile one CUDA source for ``arch`` with the board's flags; do not run it."""
    nvcc = find_nvcc()
    if nvcc is None:
        return Conclusion("cuda-inline", UNKNOWN,
                          "no nvcc on this machine: set NVCC or CUDA_HOME to compile CUDA "
                          "sources for the target", line)
    cutlass = find_cutlass()
    if _has_cutlass_include(text) and cutlass is None:
        return Conclusion("cuda-inline", UNKNOWN,
                          "the source includes CUTLASS / CuTe headers and no CUTLASS tree was "
                          "found (set CUTLASS_DIR): cannot compile it here", line)
    includes = include_paths(cutlass) + torch_include_paths()
    tmp = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="fk_cc_"))
    tmp.mkdir(parents=True, exist_ok=True)
    src_path = tmp / "submission_kernel.cu"
    obj_path = tmp / "submission_kernel.o"
    src_path.write_text(text, encoding="utf-8")
    gencode_flag = f"-gencode=arch={compute_arch(arch)},code={cuda_arch(arch)}"
    try:
        proc = subprocess.run(
            ["nvcc", "-c", str(src_path), "-o", str(obj_path), gencode_flag,
             "-O3", "--use_fast_math", "-std=c++20", *torch_cuda_defines(),
             *[f"-I{p}" for p in includes]],
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return Conclusion("cuda-inline", UNKNOWN,
                          "nvcc is not on PATH (set NVCC or CUDA_HOME)", line)
    except subprocess.TimeoutExpired:
        return Conclusion("cuda-inline", FAIL,
                          f"nvcc did not finish within {timeout}s: the board's compile "
                          f"timeout is 120s", line)
    except Exception as e:  # noqa: BLE001 -- reported, never raised at the caller
        return Conclusion("cuda-inline", UNKNOWN,
                          f"could not run nvcc: {type(e).__name__}: {e}", line)
    if proc.returncode == 0 and obj_path.is_file():
        return Conclusion("cuda-inline", OK,
                          f"compiled for {cuda_arch(arch)} ({gencode_flag}, -O3, "
                          f"--use_fast_math)", line)
    words = _first_error(proc.stderr or proc.stdout)
    if _looks_like_missing_environment(words):
        return Conclusion("cuda-inline", UNKNOWN,
                          f"cannot compile here (a missing include or toolkit file): {words}",
                          line)
    return Conclusion("cuda-inline", FAIL, f"nvcc for {cuda_arch(arch)}: {words}", line)


def _first_error(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    errors = [ln for ln in lines if "error" in ln.lower()]
    chosen = errors[0] if errors else (lines[-1] if lines else "no output")
    return chosen[:400]


def _looks_like_missing_environment(message: str) -> bool:
    lowered = message.lower()
    if "cannot open source file" in lowered or "no such file or directory" in lowered:
        return True
    return "fatal error: cuda" in lowered


# -- triton -----------------------------------------------------------------------------

_TRITON_PREAMBLE = "import triton\nimport triton.language as tl\n"


def _load_kernel(source: str, node, workdir: Path, name: str):
    """Import one extracted kernel (its temporary module holds that kernel and nothing else).

    The segment starts at the first decorator (``ast.get_source_segment`` would drop
    ``@triton.jit``) and is dedented, so a kernel nested in the submission still becomes a
    module-level function in the temporary module.
    """
    import importlib.util
    import textwrap

    lines = source.splitlines()
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end = node.end_lineno or node.lineno
    segment = textwrap.dedent("\n".join(lines[start - 1:end]))
    path = workdir / f"_kernel_{name}.py"
    path.write_text(_TRITON_PREAMBLE + "\n\n" + segment + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"fk_cc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, node.name)


def _signature_for(fn) -> dict | None:
    """A Triton signature for a kernel, or None when a parameter cannot be inferred.

    Pointers are guessed as ``*fp32`` (a name ending in ``_ptr``, or an annotation mentioning
    a tensor), integers as ``i32``, and a ``tl.constexpr`` parameter takes a placeholder
    value. The guess exists to make the *compiler* run: the question is whether the target
    architecture accepts the kernel, not whether this is the exact call the harness will make.

    The parameter list comes from the wrapped function (a ``JITFunction``'s own signature is
    ``(*args, **kwargs)``), and ``constexprs`` -- Triton's own record of which parameters are
    compile-time -- decides the constexpr slots.
    """
    import inspect

    import triton.language as tl

    inner = getattr(fn, "fn", fn)
    try:
        params = list(inspect.signature(inner).parameters.values())
    except (TypeError, ValueError):
        params = []
    names = list(getattr(fn, "arg_names", None) or [p.name for p in params])
    if not names:
        return None
    constexprs = getattr(fn, "constexprs", None) or []
    constexpr_names = {c for c in constexprs if isinstance(c, str)}
    constexpr_index = {c for c in constexprs if isinstance(c, int)}
    out: dict = {}
    for index, name in enumerate(names):
        param = params[index] if index < len(params) else None
        annotation = param.annotation if param is not None else inspect.Parameter.empty
        default = param.default if param is not None else inspect.Parameter.empty
        text = "" if annotation is inspect.Parameter.empty else str(annotation)
        if (index in constexpr_index or name in constexpr_names
                or "constexpr" in text or isinstance(default, tl.constexpr)):
            out[name] = "constexpr"
        elif name.endswith("_ptr") or "Tensor" in text or "tensor" in text:
            out[name] = "*fp32"
        elif "bool" in text:
            out[name] = "i1"
        elif "float" in text or isinstance(default, float):
            out[name] = "fp32"
        else:
            out[name] = "i32"          # a size, an index: the common case
    return out


def check_triton_kernel(fn, *, arch: str = TARGET_ARCH, line: int = 0) -> Conclusion:
    """Cross-compile one ``@triton.jit`` kernel for ``arch`` (no device involved)."""
    major, minor = cc_of_arch(arch)
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        from triton.compiler import compile as triton_compile
    except Exception as e:  # noqa: BLE001 -- no Triton here: not a verdict about the kernel
        return Conclusion("triton", UNKNOWN,
                          f"Triton is not importable here ({type(e).__name__}: {e}): the "
                          f"Triton path is checked statically only", line)
    signature = _signature_for(fn)
    if signature is None:
        return Conclusion("triton", UNKNOWN,
                          "a kernel parameter's type cannot be inferred (annotate it or name "
                          "pointers `*_ptr`): not compiled here", line)
    constexprs = {k: 64 for k, v in signature.items() if v == "constexpr"}
    try:
        src = ASTSource(fn, signature, constexprs)
        compiled = triton_compile(src, target=GPUTarget("cuda", int(f"{major}{minor}"), 32),
                                  options={"num_warps": 4})
        cubin = bytes(compiled.asm.get("cubin", b""))
    except Exception as e:  # noqa: BLE001 -- classified below
        message = _first_error("".join(traceback.format_exception_only(type(e), e)))
        if _looks_like_target_problem(message):
            return Conclusion("triton", FAIL,
                              f"Triton cannot build this kernel for CC {major}.{minor}: "
                              f"{message}", line)
        return Conclusion("triton", UNKNOWN,
                          f"Triton failed before code generation ({message}): this check is "
                          f"about the target, not about the call the harness will make", line)
    if not cubin:
        return Conclusion("triton", FAIL,
                          f"Triton produced no cubin for CC {major}.{minor}", line)
    return Conclusion("triton", OK,
                      f"Triton {getattr(triton, '__version__', '?')} compiled it to a "
                      f"{len(cubin)}-byte cubin for CC {major}.{minor}", line)


def _looks_like_target_problem(message: str) -> bool:
    """Does this error say the *target* is the problem (rather than our guessed signature)?

    Only the message is read, never a traceback: a traceback carries the file names of
    Triton's own modules, and matching on those would turn every failure into a target
    failure.
    """
    lowered = message.lower()
    return any(word in lowered for word in (
        "ptxas", "compute_100", "sm_100", "no such instruction", "unsupported",
        "is not supported", "not supported on", "requires sm_", "arch='sm"))


# -- cutedsl ----------------------------------------------------------------------------

def check_cutedsl(source: str, *, arch: str = TARGET_ARCH, line: int = 0) -> Conclusion:
    """What can be said about a CuTe DSL submission without a device of the target arch."""
    if "cutlass.cute" not in source and "cute.kernel" not in source:
        return Conclusion("cutedsl", OK, "no CuTe DSL kernel in this submission", line)
    try:
        import cutlass.cute  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return Conclusion("cutedsl", UNKNOWN,
                          f"nvidia-cutlass-dsl is not importable here "
                          f"({type(e).__name__}: {e}): CuTe DSL sources are checked statically "
                          f"only", line)
    major, minor = cc_of_arch(arch)
    return Conclusion("cutedsl", UNKNOWN,
                      f"a CuTe DSL kernel compiles by loading it on a device of the target "
                      f"architecture (CC {major}.{minor}) and this machine has none: it is "
                      f"compiled here on the development GPU and checked statically against "
                      f"{cuda_arch(arch)}", line)


# -- the whole check ---------------------------------------------------------------------

def check_source(source: str, *, label: str = "<source>", arch: str = TARGET_ARCH,
                 timeout: int = 900, workdir: str | Path | None = None,
                 run_rules: bool = True) -> Report:
    """Every conclusion for one submission's source (compiled on a copy, never imported)."""
    report = Report(source=label, arch=arch)
    if run_rules:
        report.findings = rules.check_submission(source, rules.language_of(source))
        blocks = rules.blocks(report.findings)
        if blocks:
            report.conclusions.append(Conclusion(
                "static", FAIL, "; ".join(str(f) for f in blocks[:4]), blocks[0].line))
        else:
            warns = rules.warn_or_worse(report.findings)
            report.conclusions.append(Conclusion(
                "static", OK, "no blocking rule fired"
                + (f" ({len(warns)} warning(s))" if warns else "")))
    tmp_root = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix="fk_cc_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    for line, text in cuda_sources_in(source):
        report.conclusions.append(check_cuda_source(text, arch=arch, timeout=timeout,
                                                    workdir=tmp_root, line=line))
    report.conclusions += _triton_conclusions(source, arch=arch, workdir=tmp_root)
    report.conclusions.append(check_cutedsl(source, arch=arch))
    report.notes += _limitations(source, arch)
    return report


def _triton_conclusions(source: str, *, arch: str, workdir: Path) -> list[Conclusion]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Conclusion("triton", FAIL, f"the source does not parse: {e}")]
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and any((d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")) == "jit"
                     for d in node.decorator_list)]
    if not nodes:
        detail = ("no @triton.jit kernel in this submission" if "triton" not in source
                  else "triton is imported but no @triton.jit kernel is defined")
        return [Conclusion("triton", OK, detail)]
    out: list[Conclusion] = []
    for node in nodes:
        try:
            fn = _load_kernel(source, node, workdir, node.name)
        except Exception as e:  # noqa: BLE001 -- the kernel needs its module scope
            out.append(Conclusion(
                "triton", UNKNOWN,
                f"{node.name} could not be loaded on its own ({type(e).__name__}: {e}): it "
                f"depends on module-level names, so it is checked by the card's own compile "
                f"instead", node.lineno))
            continue
        out.append(check_triton_kernel(fn, arch=arch, line=node.lineno))
    return out


def _limitations(source: str, arch: str) -> list[str]:
    notes = []
    if "cutlass" in source:
        notes.append("CuTe DSL is compiled on the development GPU, not for the target: the "
                     "target compile check for it is static only")
    notes.append(f"CUDA sources are compiled with the board's flags ({cuda_arch(arch)}, -O3, "
                 f"--use_fast_math) and not run; a submission's own extra flags are reflected "
                 f"through the static rules, not through this build")
    return notes


def check_file(path: str | Path, **kwargs) -> Report:
    """``check_source`` for a submission file or a task directory."""
    label, source = read_source(path)
    return check_source(source, label=label, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """``python -m forgekernel.targethw.compile_check <path>`` (the CLI is `forge target`)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.stderr.write("usage: compile_check <submission.py | task dir>\n")
        return 2
    try:
        report = check_file(argv[0])
    except CompileCheckError as e:
        sys.stderr.write(f"{e}\n")
        return 2
    for line in report.lines():
        sys.stdout.write(line + "\n")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
