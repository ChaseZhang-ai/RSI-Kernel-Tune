"""A minimal target-compile check for a Triton kernel file (provisional).

The board's readings are B200 (CC 10.0); the development GPU is a B300 (CC 10.3).
A Triton kernel that compiles here may still fail there -- and a kernel that only
compiles here *because* it leans on a 10.3-only lowering fails there for sure. This
module answers, before an export: **does this kernel file lower for the target?**

What it does

* **Static contract** (always): the file parses, defines a top-level ``run``, has at
  least one ``@triton.jit`` kernel, names no development-GPU architecture
  (``sm_103`` / ``compute_103`` / ``TRITON_OVERRIDE_ARCH``), and is told off (a
  warning, not a failure) for ``@triton.autotune``: the board's protocol allows ten
  warmup calls and flushes L2 before every timed call, so the config an autotuner
  picks from its own timings is picked on the wrong side of the cache flush. Ship
  the shape-keyed ``num_warps`` / ``num_stages`` / ``BLOCK`` table instead.
* **Cross-compile** (when Triton is importable): every ``@triton.jit`` kernel is
  handed to ``triton.compile`` with an explicit ``GPUTarget("cuda", <arch>, 32)``
  and never launched. Nothing of the kernel runs on a device, so this works on a
  machine with no GPU (it needs the target's ``ptxas``, which the Triton wheel
  carries).

What it cannot know

A kernel's *launch* signature (pointer dtypes, integer widths, the values of its
``tl.constexpr`` parameters) lives at the call site inside ``run``, not in the
kernel's definition. Without it the kernels are compiled with placeholder types
(``*fp32`` pointers, ``i32`` integers) and, when a ``constexpr`` value is unknown,
are reported as *not checked* rather than failed. Pass ``--signature`` /
``--constexprs`` (or the same dicts to :func:`check_file`) for a faithful reading of
one kernel; a file with several kernels can be checked kernel by kernel.

Provisional: task targethw's ``compile-check`` owns the general target-compilation
gate (CuTe DSL / CUDA included) and supersedes this module; until it lands, this is
what an export can call. It is deliberately import-free at module level (no torch,
no triton) so ``--help`` works on a machine that has neither.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import re
import sys
import tokenize
import uuid
from dataclasses import dataclass
from pathlib import Path

# The board's target: B200 = compute capability 10.0, compiled as sm_100a.
DEFAULT_TARGET = "10.0"
PROBE_WARP_SIZE = 32

# Names that pin a kernel to the development GPU instead of the target.
ARCH_PIN_RES = (
    (re.compile(r"\bsm_?10[0-9]a?\b"), "names a computed architecture (sm_10x)"),
    (re.compile(r"\bcompute_?10[0-9]\b"), "names a computed architecture (compute_10x)"),
    (re.compile(r"\bTRITON_OVERRIDE_ARCH\b"), "overrides the compile target at run time"),
)
AUTOTUNE = re.compile(r"@\s*triton\.autotune\s*\(")


def parse_target(target: str | int) -> tuple[int, str]:
    """``(arch_number, label)`` for ``"10.0"`` / ``"sm_100a"`` / ``100``.

    Triton's ``GPUTarget`` takes the capability as one integer (``100`` for 10.0);
    the label is what every message and verdict line prints."""
    raw = str(target).strip()
    m = re.fullmatch(r"(?:sm_|compute_|arch=)?(\d+)\.(\d+)a?", raw)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        return major * 10 + minor, f"{major}.{minor}"
    m = re.fullmatch(r"(?:sm_|compute_)?(\d{2,3})a?", raw)
    if m:
        arch = int(m.group(1))
        return arch, f"{arch // 10}.{arch % 10}"
    raise ValueError(f"target must name a compute capability (\"10.0\", \"sm_100a\", 100), "
                     f"got {target!r}")


@dataclass(frozen=True)
class KernelCheck:
    """One ``@triton.jit`` kernel's reading: compiled for the target, or why not."""

    name: str
    status: str            # "ok" | "skipped" | "failed"
    detail: str = ""
    faithful: bool = True   # False: placeholder types were used (no launch signature)


@dataclass(frozen=True)
class TargetCheck:
    """The whole file's reading.

    ``ok`` is a clean pass (the static contract holds and every kernel it compiled
    was compiled faithfully); ``passed`` also accepts the *partial* case an export
    must not be blocked by -- kernels the check could not read (no launch signature)
    are then named in ``warnings`` and counted as unverified. ``findings`` are static
    contract findings (fatal: no run / no kernel / an arch pin); ``warnings`` are
    advice and the unverified-kernel notes."""

    path: str
    target: str
    arch: int
    available: bool                  # Triton could be imported and used here
    kernels: tuple[KernelCheck, ...] = ()
    findings: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str = ""                  # why the check could not run at all

    @property
    def unverified(self) -> tuple[KernelCheck, ...]:
        return tuple(k for k in self.kernels if k.status != "ok")

    @property
    def compiled(self) -> tuple[KernelCheck, ...]:
        return tuple(k for k in self.kernels if k.status == "ok")

    @property
    def ok(self) -> bool:
        """A clean pass: nothing failed faithfully, nothing was left unverified."""
        return (self.available and not self.findings and not self.error
                and bool(self.kernels) and not self.unverified)

    @property
    def passed(self) -> bool:
        """Whether an export may proceed: the static contract holds and no kernel
        failed *faithfully* (a placeholder-typed failure is unverified, not failed)."""
        if not self.available or self.findings or self.error:
            return False
        return not any(k.status == "failed" and k.faithful for k in self.kernels)

    def line(self) -> str:
        """The one verdict line (the shape the harness parses)."""
        if not self.available:
            state = "UNAVAILABLE"
        elif not self.passed:
            state = "FAIL"
        elif self.ok:
            state = "OK"
        else:
            state = "PARTIAL"
        return (f"TRITON-COMPILE {state} target={self.target} kernels={len(self.kernels)} "
                f"compiled={len(self.compiled)} unverified={len(self.unverified)}")


# ---------------------------------------------------------------- static contract

def _code_without_comments(src: str) -> str:
    """``src`` with every ``#`` comment blanked (string literals kept: an environment
    variable's name lives in a string, a note in a comment does not)."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


def static_findings(src: str) -> tuple[list[str], list[str]]:
    """(fatal findings, warnings) read off the kernel file's source text."""
    findings: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"the kernel file does not parse: {e}"], []
    code = _code_without_comments(src)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        findings.append("no top-level run(): the card's entry is a function named run")
    if not re.search(r"@\s*triton\.jit\b", code):
        findings.append("no @triton.jit kernel: a triton solution's compute is @triton.jit")
    for rx, why in ARCH_PIN_RES:
        for m in rx.finditer(code):
            findings.append(f"the source {why} ({m.group(0)}) — a kernel compiled for the "
                            f"development GPU is not the target's kernel")
            break
    if AUTOTUNE.search(code):
        warnings.append(
            "@triton.autotune: the board warms up 10 calls and flushes L2 before every "
            "timed call, so a config tuned from in-run timings is tuned on the wrong side "
            "of the flush and re-tunes (compiles + benches) inside the job's clock; ship a "
            "shape-keyed num_warps / num_stages / BLOCK table instead")
    return findings, warnings


# ---------------------------------------------------------------- the kernels

def _jit_functions(module) -> list:
    """Every ``@triton.jit`` function the module defines, in source order.

    Handles a bare ``JITFunction`` and the ``Autotuner`` wrapper (its ``.fn`` is the
    jitted function), by the classes Triton itself exposes -- no name guessing."""
    from triton.runtime.jit import JITFunction
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:  # a build without the autotuner module
        Autotuner = None  # type: ignore[assignment]
    out = []
    for _name, value in vars(module).items():
        fn = value
        if Autotuner is not None and isinstance(value, Autotuner):
            fn = getattr(value, "fn", None)
        if isinstance(fn, JITFunction):
            out.append(fn)
    return out


def _params(jit_fn) -> list[tuple[str, object]]:
    """``(name, annotation)`` per kernel parameter, in order.

    Triton keeps its own ``KernelParam`` list (an object whose ``annotation`` is the
    source text of the declaration, e.g. ``"constexpr"``); ``arg_names`` is the
    fallback for a build that does not expose it."""
    out = []
    for p in list(getattr(jit_fn, "params", []) or []):
        out.append((str(getattr(p, "name", "")), getattr(p, "annotation", None)))
    if not out:
        out = [(str(n), None) for n in list(getattr(jit_fn, "arg_names", []) or [])]
    return out


def _is_constexpr_param(annotation) -> bool:
    """Whether a parameter's declaration says ``tl.constexpr`` (Triton stores the
    annotation's source text, or the class itself on a plain inspect signature)."""
    if annotation is None:
        return False
    text = str(annotation)
    return ("constexpr" in text) or text.endswith("core.constexpr")


def _signature_of(jit_fn, constexprs: dict) -> tuple[dict, dict, bool]:
    """``(signature, constants, faithful)`` for one kernel.

    A ``tl.constexpr`` parameter needs a value: with one supplied it is pinned,
    without one a ``KeyError`` names it and the caller reports the kernel unchecked.
    Everything else gets a placeholder type -- pointer-like names ``*fp32``, integer
    names ``i32``, the rest ``fp32`` -- which compiles but does not prove the dtype
    path; ``faithful`` is False in that case."""
    sig: dict = {}
    consts: dict = {}
    faithful = True
    for pname, ann in _params(jit_fn):
        if not pname or pname == "self":
            continue
        if pname in constexprs:
            sig[pname] = "constexpr"
            consts[pname] = constexprs[pname]
        elif _is_constexpr_param(ann):
            raise KeyError(pname)
        elif "ptr" in pname.lower():
            sig[pname] = "*fp32"
            faithful = False
        elif re.search(r"(^|_)(n|m|k|size|len|num|stride|offset|count)", pname, re.I):
            sig[pname] = "i32"
            faithful = False
        else:
            sig[pname] = "fp32"
            faithful = False
    return sig, consts, faithful


def _fn_name(jit_fn) -> str:
    """A JITFunction's own name (its wrapper delegates attribute reads to ``fn``)."""
    return (getattr(jit_fn, "__name__", None)
            or getattr(getattr(jit_fn, "fn", None), "__name__", None) or "?")


def _import_kernel(path: Path):
    """The kernel file as a module. Imported under a unique name so two checks in one
    process cannot collide in ``sys.modules``."""
    name = f"_fk_triton_check_{uuid.uuid4().hex[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"{path}: not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def check_file(path: str | Path, *, target: str | int = DEFAULT_TARGET,
               signature: dict | None = None, constexprs: dict | None = None,
               kernel: str | None = None, num_warps: int = 4,
               num_stages: int = 3) -> TargetCheck:
    """Check one Triton kernel file against ``target`` (see the module docstring).

    ``signature`` / ``constexprs``: the launch signature of ONE kernel, named by
    ``kernel`` when the file has several (``None``: placeholders, as described).
    ``check_file`` never launches anything and never writes a file."""
    p = Path(path)
    label_arch, label = parse_target(target)
    try:
        src = p.read_text(encoding="utf-8")
    except OSError as e:
        return TargetCheck(str(p), label, label_arch, available=False,
                           error=f"unreadable: {e}")
    findings, warnings = static_findings(src)
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource
    except Exception as e:  # noqa: BLE001 — a missing/broken Triton is a reading, not a crash
        return TargetCheck(str(p), label, label_arch, available=False, findings=tuple(findings),
                           warnings=tuple(warnings),
                           error=f"triton is not usable here ({type(e).__name__}: {e}); the "
                                 f"cross-compile needs the eval image's Triton")
    try:
        module = _import_kernel(p)
    except Exception as e:  # noqa: BLE001
        return TargetCheck(str(p), label, label_arch, available=True, findings=tuple(findings),
                           warnings=tuple(warnings),
                           error=f"the kernel file does not import here ({type(e).__name__}: "
                                 f"{str(e)[:400]})")
    jits = _jit_functions(module)
    kernels: list[KernelCheck] = []
    chosen = None
    if signature or constexprs:
        # A launch signature belongs to ONE kernel: the caller names it, or there is
        # only one to mean.
        chosen = kernel or (getattr(jits[0], "__name__", None) if len(jits) == 1 else None)
        if chosen is None:
            warnings.append("--signature was given without --kernel on a file with several "
                            "kernels: name the kernel the signature belongs to")
    for fn in ([] if (signature or constexprs) and chosen is None else jits):
        name = _fn_name(fn)
        try:
            if chosen is not None and name != chosen:
                kernels.append(KernelCheck(
                    name, "skipped",
                    f"no launch signature given (--kernel {chosen} is the one checked)", True))
                continue
            if chosen is not None:
                sig, consts, faithful = dict(signature or {}), dict(constexprs or {}), True
            else:
                sig, consts, faithful = _signature_of(fn, dict(constexprs or {}))
        except KeyError as missing:
            kernels.append(KernelCheck(
                name, "skipped", f"needs a value for constexpr {str(missing)} (--constexprs)",
                True))
            continue
        src_obj = ASTSource(fn=fn, signature=sig, constexprs=consts)
        try:
            compiled = triton.compile(src_obj, target=GPUTarget("cuda", label_arch, PROBE_WARP_SIZE),
                                      options={"num_warps": int(num_warps),
                                               "num_stages": int(num_stages)})
            asm = getattr(compiled, "asm", {}) or {}
            ptx = asm.get("ptx")
            kernels.append(KernelCheck(name, "ok",
                                       f"ptx {len(ptx)} bytes" if ptx else "compiled", faithful))
        except Exception as e:  # noqa: BLE001 — the compiler's own error is the reading
            kernels.append(KernelCheck(name, "failed",
                                       f"{type(e).__name__}: {str(e)[:400]}", faithful))
    if not kernels:
        warnings.append("the file defines no @triton.jit kernel this check could read")
    elif not any(k.status == "ok" for k in kernels):
        warnings.append("no kernel was compiled: pass --signature / --constexprs for the "
                        "launch signature the kernel runs with")
    for k in kernels:
        if k.status == "failed" and not k.faithful:
            warnings.append(f"kernel {k.name} failed with placeholder argument types — that is "
                            f"not a verdict on the kernel: pass --signature / --constexprs")
    return TargetCheck(str(p), label, label_arch, available=True, kernels=tuple(kernels),
                       findings=tuple(findings), warnings=tuple(warnings))


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forgekernel.triton_check",
        description="Cross-compile a Triton kernel file to an explicit target (no GPU, "
                    "no launch) and read its static contract.")
    ap.add_argument("path", help="the kernel file (one Python module with @triton.jit kernels)")
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help=f"compute capability (default {DEFAULT_TARGET} = B200; \"sm_86\" for "
                         f"the RTX 3060 smoke)")
    ap.add_argument("--signature", default=None,
                    help="JSON dict of the one kernel's argument types, e.g. "
                         "'{\"x_ptr\": \"*bf16\", \"w_ptr\": \"*bf16\", \"n\": \"i32\"}'")
    ap.add_argument("--constexprs", default=None,
                    help="JSON dict of that kernel's constexpr values, e.g. '{\"BLOCK\": 1024}'")
    ap.add_argument("--kernel", default=None,
                    help="which @triton.jit function --signature belongs to (needed when the "
                         "file has several)")
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print the whole reading as JSON")
    args = ap.parse_args(argv)
    try:
        signature = json.loads(args.signature) if args.signature else None
        constexprs = json.loads(args.constexprs) if args.constexprs else None
    except json.JSONDecodeError as e:
        ap.error(f"--signature / --constexprs must be JSON: {e}")
    if signature is not None and not isinstance(signature, dict):
        ap.error("--signature must be a JSON object")
    if constexprs is not None and not isinstance(constexprs, dict):
        ap.error("--constexprs must be a JSON object")
    res = check_file(args.path, target=args.target, signature=signature,
                     constexprs=constexprs, kernel=args.kernel,
                     num_warps=args.num_warps, num_stages=args.num_stages)
    if args.json:
        print(json.dumps({
            "ok": res.ok, "passed": res.passed, "path": res.path, "target": res.target,
            "arch": res.arch, "available": res.available, "error": res.error,
            "kernels": [k.__dict__ for k in res.kernels],
            "findings": list(res.findings), "warnings": list(res.warnings)}, indent=2))
    else:
        print(res.line(), flush=True)
        if res.error:
            print(f"  note: {res.error}", flush=True)
        for k in res.kernels:
            print(f"  kernel {k.name}: {k.status}"
                  + ("" if k.faithful else " (placeholder argument types)") + f" — {k.detail}",
                  flush=True)
        for f in res.findings:
            print(f"  finding: {f}", flush=True)
        for w in res.warnings:
            print(f"  warning: {w}", flush=True)
    if not res.available:
        return 2          # the check could not run here (no usable Triton)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
