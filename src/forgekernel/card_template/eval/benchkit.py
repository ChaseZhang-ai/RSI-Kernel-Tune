"""The card's bench engine: judge and time a forgekernel task's kernel.

Copy into <op>/eval/ (``forge run`` does), do not edit. The generated eval/bench.py
calls ``run_bench``; eval/ncu_run.py calls ``ncu_window``.

One card = the task's problem, frozen in ``eval/problem/problem.json``: the definition
(signature over axes; ``reference`` = the JUDGE, the PyTorch semantics the teacher wrote
from the task's description, with its ``make_inputs``), every shape's input groups
(the task's files, then the groups the judge's author added), the tolerance calibrated
from the reference, and -- when the task has one -- the reference itself, sealed
(``closed_ref.py``) so it rides the student's jobs without being readable.

Correctness: every input group of every shape, the judge on a copy of the inputs, then
the kernel (the ``task_correctness`` protocol below). Timing: per shape,
timing.measure_windows_workloads over its input groups (burn-in, then mirrored
reference / kernel / kernel / reference on each -- timing.measure_paired's rule), the
geomean over the groups; across shapes, the geomean over the shapes. The eval JSON maps
the readings into the keys the driver gates on:

    device_median_ms / ref_device_ms / vs_ref_device   the gate reading
    bwd_median_ms / ref_median_ms / vs_ref             timing.time_e2e, for the log
    host_enqueue_ms, device_window_basis               the precondition audit
    bench                                              timer, per-shape readings, clock

A multi-shape task (problem.json ``group``) is judged and timed shape by shape -- each
one's burn-in its own -- and passes only when every shape passes (the first failure ends
the run, named). The gate keys are the geometric mean over the shapes; one per-shape
block (with its ``name``) rides under ``performance.workloads``.

The loop's meta tools (lever pricing, stall / SASS pricing, the timeline probe) read a
multi-shape card one shape at a time: ``FK_LB_WORKLOAD_SCOPE`` = ``representative`` or a
group entry's name (``wl5``) narrows both the bench and ``ncu_window`` to that shape,
and the JSON then carries ``workload_scope`` (a meta reading, never a gate).

Entry points (the generated eval/bench.py and eval/ncu_run.py call these):

    run_bench()   --mode all|correctness|perf|ref|calibrate  [--impl kernel|ref]
    ncu_window()  the kernel's calls inside a profiler start/stop window

Provenance: the random-input draw and the input name heuristics are adapted from
SOL-ExecBench (core/bench/io.py, Apache-2.0); see NOTICE at the repository root.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import math
import operator
import os
import re
import shutil
import statistics
import subprocess
import sys
import tokenize
from pathlib import Path

import torch




# WP9 fresh-recheck: the child process re-running correctness under a layer-2
# seed draws its inputs at a DIFFERENT seed than the timed rounds did, so a
# content-keyed cache (the correct outputs stored against the input values)
# misses as well. The task suite's generators draw unseeded, so its layer 1
# already carries fresh content; the env pins the child's shift explicitly.
FRESH_SEED_ENV = "FK_FRESH_SEED"
FRESH_RECHECK_ENV = "FK_FRESH_RECHECK"


def _recheck_seed_shift() -> int:
    """The seed SHIFT a fresh-recheck child applies to the task's per-group
    generator seeds: in the child, drawn inputs must differ in content from the
    timed draw (the same shift cannot fire in the parent — the env is child-only,
    and the default shift is 0 outside a recheck)."""
    if not os.environ.get(FRESH_RECHECK_ENV):
        return 0
    try:
        return int(os.environ.get(FRESH_SEED_ENV, 7919))
    except ValueError:
        return 7919


# Packed fp4 (two E2M1 codes per byte); older torch builds lack it, and only fp4
# definitions need it.
_FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)
BURN_IN_S = 4.0
PROBLEM_FILE = Path(__file__).resolve().parent / "problem" / "problem.json"


# --- problem / definition ------------------------------------------------------

def load_problem(path=None) -> dict:
    """The card's frozen problem (``eval/problem/problem.json``)."""
    p = Path(path) if path else PROBLEM_FILE
    problem = json.loads(p.read_text(encoding="utf-8"))
    problem["_dir"] = str(p.parent)
    return problem


def workload_views(problem: dict) -> list[dict]:
    """The card's workloads, each as a problem dict judged and timed on its own: a group
    card's every ``group`` entry (its workload, ``_wl_name``), or the card itself."""
    group = problem.get("group") or []
    if len(group) <= 1:
        return [problem]
    out = []
    for g in group:
        v = dict(problem)
        v["workload"] = g["workload"]
        v["_wl_name"] = str(g.get("name") or f"wl{g.get('index')}")
        out.append(v)
    return out


def describe_card_workloads(problem: dict) -> str:
    views = workload_views(problem)
    if len(views) == 1:
        return f"workload={problem['workload'].get('uuid')}"
    return (f"workloads={','.join(v['_wl_name'] for v in views)} (group of {len(views)}; "
            f"geomean gate)")


_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def resolve_shape_expression(expr: str, env: dict) -> int:
    """An ``expr`` axis: arithmetic over axis names."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise NameError(f"Unknown variable: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    value = ev(ast.parse(expr, mode="eval").body)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"expression {expr!r} is not an integer: {value}")
        value = int(value)
    return value


def resolved_axes(definition: dict, workload_axes: dict) -> dict:
    """Const axes, the workload's var axes, then every expr axis."""
    axes = definition.get("axes") or {}
    out = {n: int(a["value"]) for n, a in axes.items() if a.get("type") == "const"}
    out.update({n: int(v) for n, v in (workload_axes or {}).items()})
    missing = [n for n, a in axes.items() if a.get("type") == "var" and n not in out]
    if missing:
        raise ValueError(f"workload gives no value for var axes {missing}")
    for n, a in axes.items():
        if a.get("type") == "expr":
            out[n] = resolve_shape_expression(a["expression"], out)
    return out


def tensor_shapes(specs: dict, axes: dict) -> dict:
    """name -> concrete shape tuple (None for a python scalar)."""
    out = {}
    for name, spec in specs.items():
        shape = spec.get("shape")
        if shape is None:
            out[name] = None
            continue
        out[name] = tuple(int(a) if str(a).isdigit() else int(axes[a]) for a in shape)
    return out


def torch_dtype(name: str) -> torch.dtype:
    table = {
        "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2, "int64": torch.int64, "int32": torch.int32,
        "int16": torch.int16, "int8": torch.int8, "bool": torch.bool,
    }
    if name in ("float4_e2m1", "float4_e2m1fn_x2"):
        if _FP4X2 is None:
            raise ValueError(f"dtype {name!r} needs a torch build with float4_e2m1fn_x2")
        return _FP4X2
    if name not in table:
        raise ValueError(f"Unsupported dtype '{name}'")
    return table[name]


def _exec_source(source: str, filename: str) -> dict:
    ns: dict = {"__name__": f"benchkit_{Path(filename).stem}", "__file__": filename}
    exec(compile(source, filename, "exec"), ns)  # noqa: S102 — the task's own judge/reference
    return ns


def reference_callables(problem: dict):
    """(run, custom_inputs_fn | None) from the definition's reference source."""
    defn = problem["definition"]
    ns = _exec_source(defn["reference"], f"<{defn['name']} reference>")
    custom = defn.get("custom_inputs_entrypoint")
    return ns["run"], (ns[custom] if custom else None)


def load_entry(path: Path, entry: str):
    """``entry`` from the python file at ``path``, imported as a module with its own
    directory on sys.path (a solution may import its sibling files, at call time too)."""
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        f"lb_{path.parent.name}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # under the correctness monitor, its hook watches the kernel's calls; a by-path
    # load never goes through the import hook it installs, so hand the module over
    hook = sys.modules.get("fk_monitor_hook")
    if hook is not None and hasattr(hook, "watch_loaded"):
        hook.watch_loaded(mod, str(path))
    return getattr(mod, entry)


def load_binary_entry(path: Path, module: str, entry: str):
    """``entry`` of a compiled extension — a sealed binary reference (problem.json
    reference_impl kind ``binary``, ``ref_bin/<module>.so``). Loaded under its build-time
    module name, which fixes its ``PyInit_`` symbol; torch is already imported, which the
    extension links against."""
    path = Path(path).resolve()
    loader = importlib.machinery.ExtensionFileLoader(module, str(path))
    spec = importlib.util.spec_from_loader(module, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return getattr(mod, entry)


def load_sealed_entry(path: Path, module: str, entry: str | None, key_env: str = "FK_REF_KEY"):
    """``entry`` of a sealed SOURCE reference (problem.json reference_impl kind
    ``sealed``, ``ref_bin/ref_payload.bin``): decoded in memory by the closed_ref.py next
    to this file, with the key from the job environment. The callable runs the
    reference inside its source window and withholds its exceptions' text."""
    cr_path = Path(__file__).resolve().with_name("closed_ref.py")
    spec = importlib.util.spec_from_file_location("_lb_closed_ref", cr_path)
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    fn = cr.entry(entry, str(Path(path).resolve()), name=module, key_env=key_env)
    # Decoded: the key has done its job, and the student's kernel is imported into this
    # same process next (the fence refuses a kernel that names the key or os.environ).
    os.environ.pop(key_env, None)
    return fn


_PRECISION: dict = {}


def _precision():
    """The card's eval/precision.py (the precision gate's measure), or None when the card
    predates it."""
    if "mod" not in _PRECISION:
        path = Path(__file__).resolve().with_name("precision.py")
        mod = None
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_lb_precision", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        _PRECISION["mod"] = mod
    return _PRECISION["mod"]


# A CUDA C++ kernel file (problem.json kernel_file ``*.cu``, loop language ``cuda-sol``) is
# compiled here: torch cpp_extension, the file's own PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
# binding, CUTLASS headers on the include path, -O3 --use_fast_math and -lcuda, sm_100a unless
# a flag names an arch, plus the flags of any ``// fk-cuda-cflags: ...`` line. The card's
# gpu_entry.sh NCU build and the loop's pricing tools read the same list, so the bytes judged
# and timed here are the bytes every other tool builds. The file is compiled alone (copied
# into its build directory): a solution is this one file.
CUDA_KERNEL_SUFFIXES = (".cu",)
CUDA_DEFAULT_CFLAGS = ["-O3", "--use_fast_math"]
CUDA_DEFAULT_LDFLAGS = ["-lcuda"]
CUDA_GENCODE = "-gencode=arch=compute_100a,code=sm_100a"
CUDA_FLAGS_LINE = re.compile(r"^[ \t]*//[ \t]*fk-cuda-cflags:(.*)$", re.M)
# "1" from the loop's stall pricing (recon/stall_price_run.py): the build adds -lineinfo.
CUDA_LINEINFO_ENV = "FK_CUDA_LINEINFO"
# What an fk-cuda-cflags line may hold: code generation and diagnostics, never a path, a
# library or a linker option (those could pull the sealed reference into the build).
CUDA_FLAG_OK = re.compile(
    r"-O[0-3]|--use_fast_math|-use_fast_math|--?ftz=(true|false)|--?prec-(div|sqrt)=(true|false)"
    r"|--?fmad=(true|false)|-lineinfo|--?extra-device-vectorization|--?maxrregcount=\d+"
    r"|-Xptxas=[-\w,=.]+|--ptxas-options=[-\w,=.]+|-D[A-Za-z_]\w*(=[-\w.]+)?|-U[A-Za-z_]\w*"
    r"|-std=c\+\+\d\d|--?expt-relaxed-constexpr|--?expt-extended-lambda|--?restrict"
    r"|-gencode=arch=compute_\w+,code=(sm|compute)_\w+|-arch=(sm|compute)_\w+"
    r"|--?default-stream=(legacy|null|per-thread)|-Xcompiler=-O[0-3]|-w|-Wno-[-\w]+")


def cuda_kernel_flags(src: str) -> list[str]:
    """nvcc flags of a CUDA kernel file: the defaults, the file's fk-cuda-cflags, the arch.
    Raises ValueError on a flag outside CUDA_FLAG_OK."""
    extra = [f for line in CUDA_FLAGS_LINE.findall(src) for f in line.split()]
    if bad := [f for f in extra if not CUDA_FLAG_OK.fullmatch(f)]:
        raise ValueError(f"fk-cuda-cflags: {bad} not allowed (code generation / diagnostics "
                         f"flags only: no paths, libraries or linker options)")
    flags = CUDA_DEFAULT_CFLAGS + [f for f in extra if f not in CUDA_DEFAULT_CFLAGS]
    if not any(f.startswith(("-gencode", "-arch")) for f in flags):
        flags.append(CUDA_GENCODE)
    return flags


def load_cuda_entry(path: Path, entry: str):
    """``entry`` of a CUDA C++ kernel file, built with torch cpp_extension
    (see CUDA_KERNEL_SUFFIXES). The build is keyed by the source and the flags, so an
    unchanged file loads the module already built in this job's build directory."""
    import hashlib
    import tempfile
    import torch.utils.cpp_extension as ext
    path = Path(path).resolve()
    src = path.read_text(encoding="utf-8")
    flags = cuda_kernel_flags(src)
    # A profiling run that maps SASS back to source lines (the stall ledger) builds with
    # -lineinfo: its own build under its own prefix, so the timed build and the census /
    # SASS-sha readers (orch_census_run.find_cuda_build: fk_kernel_*) never load it.
    lineinfo = os.environ.get(CUDA_LINEINFO_ENV) == "1" and "-lineinfo" not in flags
    if lineinfo:
        flags = flags + ["-lineinfo"]
    key = hashlib.sha256("\0".join([src, " ".join(flags), str(torch.__version__)]).encode()).hexdigest()
    module = f"{'fk_lineinfo_' if lineinfo else 'fk_kernel_'}{key[:12]}"
    root = os.environ.get("FK_KERNEL_BUILD_DIR") or os.path.join(tempfile.gettempdir(), "fk_kernel_build")
    build = Path(root) / module
    build.mkdir(parents=True, exist_ok=True)
    # Rewrite the build's source copy only when it differs: a fresh mtime marks the
    # object dirty for ninja, and every new process (each bench / pricing variant /
    # NCU target of one job) would recompile an unchanged kernel (~35 s with CUTLASS).
    staged = build / path.name
    try:
        same = staged.read_text(encoding="utf-8") == src
    except (OSError, UnicodeDecodeError):
        same = False
    if not same:
        staged.write_text(src, encoding="utf-8")
    cutlass = os.environ.get("CUTLASS_DIR", "/usr/local/cutlass")
    mod = ext.load(name=module, sources=[str(build / path.name)],
                   extra_cuda_cflags=list(flags), extra_ldflags=list(CUDA_DEFAULT_LDFLAGS),
                   extra_include_paths=[f"{cutlass}/include", f"{cutlass}/tools/util/include"],
                   build_directory=str(build), verbose=False, is_python_module=True)
    if not hasattr(mod, entry):
        raise AttributeError(f"{path.name}: the module defines no {entry!r} "
                             f"(PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def(\"{entry}\", ...); }})")
    global _RESUSAGE
    _RESUSAGE = _collect_resusage(build / f"{module}.so")
    return getattr(mod, entry)


# --- build-time resource telemetry (cuobjdump -res-usage) -----------------------------
#
# The binary was just built: regs / static smem / per-thread stack (a SPILL CLIFF)
# are readable NOW, at zero GPU cost, a whole iteration before NCU would confirm
# them. One RES-USAGE protocol line per kernel goes to stdout (the job log — the
# SASS-OPERANDS convention, greppable by hand), and the full list rides the eval
# JSON's top-level `resusage` key. smem is STATIC smem only: dynamic smem is a
# launch parameter and lives nowhere in the binary (NCU's smem_dynamic covers it).
# Best-effort on purpose: a missing cuobjdump or an unreadable dump prints nothing
# and embeds no key — it must never fail the bench it rides.
#
# parse_resusage_text duplicates forgekernel.inner.ncu.parse_resusage_text (the
# tested reference implementation) because the job image has no forgekernel;
# keep the two in sync.
_RESUSAGE_FUNCTION = re.compile(r"^\s*function\s+(\S+?):\s*$", re.I)
_RESUSAGE_TOKEN = re.compile(r"\b(REG|STACK|SHARED):(\d+)\b")
# dump token -> protocol field (STACK is the per-thread call stack frame: cstack)
_RESUSAGE_FIELD = {"reg": "regs", "stack": "cstack", "shared": "smem"}
_RESUSAGE_FIELD_ORDER = (("regs", "regs"), ("smem", "smem"), ("cstack", "cstack"))
_RESUSAGE: list = []


def parse_resusage_text(text: str) -> list:
    """`cuobjdump -res-usage` output -> ``[{kernel, regs?, smem?, cstack?}]``.

    One entry per kernel function, first-seen order; a kernel named again (a
    second arch in the same fatbin) is overwritten by its later reading (the
    highest arch ships). A function block with no readable numbers yields no
    entry: an unparseable dump degrades to [], never a raise."""
    out: dict = {}
    order: list = []
    current = None
    for line in text.splitlines():
        if (m := _RESUSAGE_FUNCTION.match(line)):
            current = m.group(1)
            if current not in out:
                order.append(current)
                out[current] = {"kernel": current}
            continue
        if current is None:
            continue
        for key, num in _RESUSAGE_TOKEN.findall(line):
            out[current][_RESUSAGE_FIELD[key.lower()]] = int(num)
    return [out[k] for k in order if len(out[k]) > 1]


def resusage_line(entry: dict) -> str:
    """One entry -> its protocol line: ``RES-USAGE kernel=<name> regs=<n> smem=<n> cstack=<n>``."""
    parts = [f"RES-USAGE kernel={entry['kernel']}"]
    parts += [f"{short}={entry[key]}" for key, short in _RESUSAGE_FIELD_ORDER
              if entry.get(key) is not None]
    return " ".join(parts)


def _cuobjdump_exe() -> str:
    """The cuobjdump of the toolkit that built the kernel (nvcc's dir first), else PATH."""
    exe = "cuobjdump.exe" if os.name == "nt" else "cuobjdump"
    if (found := shutil.which(exe)):
        return found
    import torch.utils.cpp_extension as ext
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                 getattr(ext, "CUDA_HOME", None), "/usr/local/cuda"):
        if base and os.path.isfile(candidate := os.path.join(base, "bin", exe)):
            return candidate
    return exe


def _collect_resusage(so_path: Path) -> list:
    """The built kernel binary -> resusage entries; prints one protocol line per
    kernel into the job log. [] (silently) when cuobjdump is absent or fails."""
    try:
        proc = subprocess.run([_cuobjdump_exe(), "-res-usage", str(so_path)],
                              capture_output=True, text=True, timeout=120)
        entries = parse_resusage_text(proc.stdout or "")
    except Exception:
        return []
    for e in entries:
        print(resusage_line(e), flush=True)
    return entries


# A binary reference runs in the student's process, so the kernel file may not reach for
# it, for the loader machinery, or for the compiled-code toolchain (comments ignored,
# strings not: a path in a string is a reach too). The reference's module name is added
# per card.
KERNEL_FENCE = (r"ref_bin", r"ExtensionFileLoader", r"extension_file", r"cpp_extension",
                r"load_inline", r"load_library", r"\bctypes\b", r"\bcffi\b", r"cuobjdump",
                r"nvdisasm", r"\bsubprocess\b", r"os\.system", r"\bimportlib\b", r"PyInit_",
                r"sys\.modules", r"__main__", r"__loader__", r"_getframe", r"\bgc\b",
                r"\binspect\b", r"\.so\b")
# A sealed SOURCE reference (closed_ref.py) runs in the student's process as a module
# whose source the loader holds in memory, whose JIT objects keep it (Triton's .src),
# and whose compiled IR lands in the JIT's on-disk cache: the kernel file may not reach
# for the loader, the payload, its key, the interpreter's reflection hooks (a tracer
# sees the reference's frames and globals) or the JIT caches and dumps.
SEALED_FENCE = (r"closed_ref", r"ref_payload", r"_REF_KEY", r"\blinecache\b", r"/nonexistent",
                r"unseal", r"keystream", r"\bzlib\b", r"b85decode", r"settrace", r"setprofile",
                r"\bmonitoring\b", r"f_globals", r"f_locals", r"f_back", r"tb_frame", r"__globals__",
                r"__builtins__", r"__import__", r"\bbuiltins\b", r"__closed_ref", r"meta_path",
                r"\.triton\b", r"triton/cache", r"TRITON_CACHE", r"TRITON_DUMP", r"TRITON_KERNEL_DUMP",
                r"MLIR_ENABLE_DUMP", r"CUTE_DSL_KEEP", r"CUTE_DSL_DUMP", r"\.ttg?ir\b", r"\.llir\b",
                r"\.ptx\b", r"\.cubin\b", r"\benviron\b", r"/proc/")


def _code_without_comments(src: str) -> str:
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


# A CUDA C++ kernel file runs in the reference's process too, and C++ reaches further than
# Python: the dynamic loader, the interpreter's C API and pybind11's, the CUDA module and
# library loaders (a loaded fatbin's kernels), the file system, the environment, other
# processes. Comments are ignored, strings are not.
CUDA_KERNEL_FENCE = (r"ref_bin", r"PyInit_", r"\.so\b", r"\bdl(open|mopen|sym|vsym|info|addr)\b",
                     r"dl_iterate_phdr", r"link_map", r"RTLD_", r"\bPy[A-Z]\w*", r"\b_Py\w*",
                     r"\bpy::", r"\bpybind11::", r"\bimport\s*\(", r"c10::Dispatcher", r"findSchema",
                     r"torch::jit", r"\bcu(Module|Library|Kernel)\w*", r"\bcuda(Library|Kernel)\w*",
                     r"cudaGetFuncBySymbol", r"Enumerate", r"\bgetenv\b", r"\benviron\b", r"/proc/",
                     r"\bsystem\s*\(", r"\bpopen\b", r"\bexec[lv]\w*\s*\(", r"\bfork\s*\(",
                     r"\bfopen\b", r"fstream", r"\bopen\s*\(", r"\bmmap\b", r"readlink", r"opendir",
                     r"cuobjdump", r"nvdisasm", r"#\s*include\s*\"[^\"]*\.(so|py)\"",
                     r"##", r"syscall")   # token pasting would spell any of the above


def _cpp_code_without_comments(src: str) -> str:
    """C/C++ source with // and /* */ comments blanked (string and char literals kept)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def kernel_fence(problem: dict) -> str | None:
    """Why the student's kernel file may not run next to this card's sealed (binary or
    source) reference, or None (also None for any other kind of reference)."""
    ref = problem.get("reference_impl") or {}
    kind = ref.get("kind")
    if kind not in ("binary", "sealed"):
        return None
    card = Path(problem["_dir"]).parent.parent
    kfile = problem.get("kernel_file", "kernel.py")
    src = (card / kfile).read_text()
    cuda = Path(kfile).suffix in CUDA_KERNEL_SUFFIXES
    if cuda:
        try:
            cuda_kernel_flags(src)
        except ValueError as e:
            return f"the kernel's build flags reach outside code generation ({e})"
    code = _cpp_code_without_comments(src) if cuda else _code_without_comments(src)
    pats = list(CUDA_KERNEL_FENCE if cuda else KERNEL_FENCE) \
        + (list(SEALED_FENCE) if kind == "sealed" else []) \
        + ([re.escape(ref["module"])] if ref.get("module") else []) \
        + ([re.escape(ref["key_env"])] if ref.get("key_env") else []) \
        + [r"(^|[^A-Za-z0-9_])" + re.escape(n) + r"([^A-Za-z0-9_]|$)"
           for n in ref.get("deny_modules") or []]   # the reference's installed package
    m = re.search("|".join(pats), code, re.M)
    what = "binary reference or compiled-code tooling" if kind == "binary" else \
        "reference, its loader, its key or the JIT's caches"
    return (f"the kernel reaches for the sealed {what} "
            f"({m.group(0)!r} matches the card's fence)") if m else None


# --- inputs ------------------------------------------------------------------------

def _cast_to_fp4x2(x):
    """Packed fp4 draw: E2M1 codes packed 2 per byte (io.py _cast_to_fp4x2; see NOTICE)."""
    r = torch.zeros_like(x, dtype=torch.uint8)
    r[(x >= 0.0) & (x <= 0.25)] = 0
    r[(x > 0.25) & (x < 0.75)] = 1
    r[(x >= 0.75) & (x <= 1.25)] = 2
    r[(x > 1.25) & (x < 1.75)] = 3
    r[(x >= 1.75) & (x <= 2.5)] = 4
    r[(x > 2.5) & (x < 3.5)] = 5
    r[(x >= 3.5) & (x <= 5.0)] = 6
    r[x > 5.0] = 7
    r[(x >= -0.25) & (x < 0.0)] = 8
    r[(x < -0.25) & (x > -0.75)] = 9
    r[(x <= -0.75) & (x >= -1.25)] = 10
    r[(x < -1.25) & (x > -1.75)] = 11
    r[(x <= -1.75) & (x >= -2.5)] = 12
    r[(x < -2.5) & (x > -3.5)] = 13
    r[(x <= -3.5) & (x >= -5.0)] = 14
    r[x < -5.0] = 15
    return (r[..., ::2] + r[..., 1::2] * 16).view(_FP4X2)


def _rand_tensor(shape, dtype, device):
    """The plain random draw (io.py _rand_tensor; see NOTICE)."""
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    if _FP4X2 is not None and dtype == _FP4X2:
        return _cast_to_fp4x2(torch.randn(shape, dtype=torch.float32, device=device))
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        low, high = {torch.int8: (-128, 128)}.get(dtype, (-1024, 1024))
        return torch.randint(low, high, shape, device=device, dtype=dtype)
    raise ValueError(f"Unsupported random dtype: {dtype}")


# The input name heuristics (io.py _generate_heuristic_tensor; same predicates, same
# order; see NOTICE).
_WEIGHT_SUFFIXES = ("_weight", "_weights", "_proj", "_projs", "_proj_weight",
                    "_proj_weights", "_weight_matrix")


def _is_weight_matrix(name, shape):
    if len(shape) < 2:
        return False
    if name.endswith(_WEIGHT_SUFFIXES) or name == "weight":
        return True
    stripped = name.rstrip("0123456789")
    return bool(stripped and stripped in ("weight",) or stripped.endswith(_WEIGHT_SUFFIXES))


def _is_norm_affine(name, whole, suffix):
    if name == whole:
        return True
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_causal_attention_mask(name, shape, description):
    if len(shape) < 2 or shape[-1] != shape[-2]:
        return False
    if name in ("attention_mask", "causal_mask"):
        return True
    d = (description or "").lower()
    return bool(description and "attention mask" in d and "causal" in d)


def _is_binary_mask(name, description):
    if name in ("x_mask", "text_mask", "aspect_ratio_mask", "drop_mask", "attention_mask"):
        return True
    if name.endswith("_mask") and description:
        d = description.lower()
        return any(k in d for k in ("binary", "{0, 1}", "0 or 1", "1.0 for valid",
                                    "0.0 for masked"))
    return False


def _is_positive_tensor(name):
    names = ("rstd", "std", "variance", "var")
    if name in names or name.endswith(("_rstd", "_std", "_variance", "_var")):
        return True
    base = name.rstrip("0123456789")
    return base in names or base.endswith(("_rstd", "_std", "_var"))


def _is_softmax_output(name, description):
    if name in ("attn_weights", "attention_weights", "routing_weights"):
        return True
    d = (description or "").lower()
    return bool(description and "softmax" in d and "output" in d)


def _heuristic_tensor(name, shape, dtype, device, description):
    """A heuristic draw for a known input name; None = fall back to _rand_tensor."""
    if not dtype.is_floating_point:
        return None
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, _FP4X2):
        return None
    if _is_norm_affine(name, "norm_weight", "_weight"):
        return torch.ones(shape, dtype=dtype, device=device)
    if _is_norm_affine(name, "norm_bias", "_bias"):
        return torch.zeros(shape, dtype=dtype, device=device)
    if _is_causal_attention_mask(name, shape, description):
        seq_len = shape[-1]
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask[..., causal] = torch.finfo(dtype).min
        return mask
    if _is_binary_mask(name, description):
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-math.pi, math.pi)
        if name in ("cos", "cos_cached", "rope_cos"):
            return torch.cos(t).to(dtype)
        return torch.sin(t).to(dtype)
    if _is_positive_tensor(name):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1
    if name in ("A", "A_log", "A_cumsum", "g"):
        if name == "A_cumsum":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-0.1, 0.0).cumsum(dim=-1)
        if name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        if name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)
    if _is_softmax_output(name, description):
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)
    if _is_weight_matrix(name, shape):
        return torch.randn(shape, dtype=dtype, device=device) / math.sqrt(shape[-1])
    return None


_SAFETENSORS_CACHE: dict = {}


def _safetensor(problem: dict, name: str, spec: dict, shape, dtype) -> torch.Tensor:
    """A workload's safetensors input, resolved against the card's problem directory
    (``forge run`` copies the files there)."""
    import safetensors.torch as st
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(problem["_dir"]) / path
    path = path.resolve()
    if path not in _SAFETENSORS_CACHE:
        _SAFETENSORS_CACHE[path] = st.load_file(str(path))
    tensors = _SAFETENSORS_CACHE[path]
    if spec["tensor_key"] not in tensors:
        raise ValueError(f"Missing key '{spec['tensor_key']}' in '{path}'")
    t = tensors[spec["tensor_key"]]
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"'{name}' expected {list(shape)}, got {list(t.shape)}")
    if t.dtype != dtype:
        raise ValueError(f"'{name}' expected {dtype}, got {t.dtype}")
    return t.contiguous()


def gen_inputs(problem: dict, device, custom_inputs_fn=None, stress: str | None = None,
               touched: list | None = None) -> list:
    """One input set in definition order, drawn as random tensors (the precision
    record's stress draws and the recon tools' probe inputs). A shape's real input
    groups -- the task's files and the judge's make_inputs seeds -- are drawn by
    ``task_inputs``, never here; ``custom_inputs_fn`` is accepted and ignored (the
    historical signature; the task's custom channel is make_inputs).

    ``stress`` (a precision.STRESS_PROFILES name): every PLAIN normal float draw (not a
    heuristic tensor -- norm weights, masks, rope tables, positive statistics -- not a
    workload / fp8 input) is passed through precision.stress_tensor; the names
    of the inputs it moved are appended to ``touched``. Only the precision record reads
    stressed inputs: correctness and timing never do."""
    defn, wl = problem["definition"], problem["workload"]
    axes = resolved_axes(defn, wl.get("axes") or {})
    shapes = tensor_shapes(defn["inputs"], axes)
    winputs = wl.get("inputs") or {}
    dev = torch.device(device)
    out = []
    for name, spec in defn["inputs"].items():
        dtype = torch_dtype(spec["dtype"])
        src = winputs.get(name) or {"type": "random"}
        kind = src.get("type", "random")
        if kind == "safetensors":
            out.append(_safetensor(problem, name, src, shapes[name], dtype).to(dev))
        elif kind == "scalar":
            out.append(src["value"])
        else:
            shape = shapes[name]
            if shape is None:
                out.append(_rand_tensor((), dtype, dev).item())
                continue
            value = _heuristic_tensor(name, tuple(shape), dtype, dev,
                                      spec.get("description"))
            plain = value is None
            if plain:
                value = _rand_tensor(shape, dtype, dev)
            if stress and plain and dtype in (torch.float32, torch.float16, torch.bfloat16):
                value = _precision().stress_tensor(value, stress)
                if touched is not None:
                    touched.append(name)
            out.append(value)
    return out


def allocate_outputs(problem: dict, device) -> list:
    defn = problem["definition"]
    axes = resolved_axes(defn, problem["workload"].get("axes") or {})
    shapes = tensor_shapes(defn["outputs"], axes)
    return [torch.empty(shapes[n], dtype=torch_dtype(s["dtype"]), device=device)
            for n, s in defn["outputs"].items()]


def normalize_outputs(problem: dict, result, device) -> list:
    """A value-returning call's result as the definition's output list
    (a tensor / scalar / tuple / list, in the definition's output order)."""
    defn = problem["definition"]
    names = list(defn["outputs"])
    dtypes = {n: torch_dtype(s["dtype"]) for n, s in defn["outputs"].items()}
    dev = torch.device(device)

    def as_tensor(name, v):
        if isinstance(v, torch.Tensor):
            return v.to(dev) if v.device != dev else v
        return torch.tensor(v, dtype=dtypes[name], device=dev)

    if isinstance(result, (tuple, list)):
        if len(result) != len(names):
            raise ValueError(f"Tuple/list has {len(result)} elements but {len(names)} "
                             f"outputs expected")
        return [as_tensor(n, v) for n, v in zip(names, result)]
    if len(names) != 1:
        raise ValueError(f"Single value returned but {len(names)} outputs expected")
    if not isinstance(result, (torch.Tensor, int, float, bool)):
        raise ValueError(f"Unexpected return type {type(result).__name__}")
    return [as_tensor(names[0], result)]


def call_impl(problem: dict, fn, inputs: list, dps: bool, device, *, no_grad: bool) -> list:
    """Run one implementation in its calling convention; its outputs, synchronized."""
    with (torch.no_grad() if no_grad else contextlib.nullcontext()):
        if dps:
            outs = allocate_outputs(problem, device)
            fn(*inputs, *outs)
        else:
            outs = normalize_outputs(problem, fn(*inputs), device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return outs


# --- correctness protocols -----------------------------------------------------------

def _impl(problem: dict, which: str):
    """(callable, dps) for ``kernel`` (the student's), ``ref`` (the in-job reference)
    or ``definition`` (the definition's own reference run)."""
    card = Path(problem["_dir"]).parent.parent
    if which == "kernel":
        kpath = card / problem.get("kernel_file", "kernel.py")
        load = load_cuda_entry if kpath.suffix in CUDA_KERNEL_SUFFIXES else load_entry
        return load(kpath, problem.get("entry", "run")), bool(problem.get("dps", True))
    ref = problem.get("reference_impl") or {"kind": "definition"}
    if which == "definition" or ref.get("kind") == "definition":
        return reference_callables(problem)[0], False
    if ref.get("kind") == "binary":
        return (load_binary_entry(Path(problem["_dir"]) / ref["path"], ref["module"],
                                  ref.get("entry", "run")),
                bool(ref.get("dps", True)))
    if ref.get("kind") == "sealed":
        return (load_sealed_entry(Path(problem["_dir"]) / ref["path"], ref["module"],
                                  ref.get("entry"), ref.get("key_env") or "FK_REF_KEY"),
                bool(ref.get("dps", True)))
    return (load_entry(Path(problem["_dir"]) / ref["path"], ref.get("entry", "run")),
            bool(ref.get("dps", False)))


def warm_reference(problem: dict, ref_fn, ref_dps: bool, device) -> None:
    """Call an in-job reference once before correctness. The bench snapshots the
    thread count after the kernel's first call and flags more threads after timing
    as injection; a reference whose first call starts threads (a compiled extension's
    pool, a JIT) would otherwise have them counted against the kernel when it is
    first timed."""
    inp = first_inputs(problem, device)
    call_impl(problem, ref_fn, inp, ref_dps, device, no_grad=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --- the eval JSON ---------------------------------------------------------------------

def _round(v, n=4):
    return None if v is None else round(float(v), n)


# What gpu_entry.sh exports about the clock lock; its first word is "locked" only when
# the lock it asked for was read back.
CLOCK_ENV = "LB_CLOCK_STATE"


def clock_state() -> str:
    return os.environ.get(CLOCK_ENV) or "unknown (not launched through gpu_entry.sh)"


def _side_windows(fn, dps: bool, problem: dict, inp: list, device) -> dict:
    """timing.py's e2e window and host-enqueue clock on one set (log / audit only)."""
    import timing
    outs = allocate_outputs(problem, device) if dps else []

    def call():
        fn(*inp, *outs)

    e2e = timing.time_e2e(call)
    return {"median_ms": statistics.median(e2e), "min_ms": min(e2e),
            "host_enqueue_ms": timing.time_host_enqueue(call)}


# ``workload_scope`` and the loop's gate parser refuses it: a meta reading, never a gate.
# ncu_window honours it too, so a tool's NCU reads the workload its bench timed.
WORKLOAD_SCOPE_ENV = "FK_LB_WORKLOAD_SCOPE"
_SCOPE: dict = {}
_WHOLE_SCOPES = ("", "all")


def apply_workload_scope(problem: dict, env=None) -> dict | None:
    """Narrow a group card to one workload when ``FK_LB_WORKLOAD_SCOPE`` names one --
    ``representative`` (the first ``group`` entry) or an entry's name; returns the scope
    record (None: the card is timed whole -- unset / ``all``, or a one-workload card). A name
    the group does not have raises: a typo must not time the whole group."""
    raw = ((os.environ if env is None else env).get(WORKLOAD_SCOPE_ENV) or "").strip()
    group = problem.get("group") or []
    if raw.lower() in _WHOLE_SCOPES or len(group) <= 1:
        return None
    every = group
    names = [str(g.get("name") or f"wl{g.get('index')}") for g in every]
    if raw.lower() == "representative":
        pick = 0
    elif raw in names:
        pick = names.index(raw)
    else:
        raise ValueError(f"{WORKLOAD_SCOPE_ENV}={raw!r} names no workload of this group card "
                         f"({', '.join(names)}; or 'representative')")
    one = every[pick]
    problem["group"] = [one]
    problem["workload"] = one["workload"]
    return {"workloads": "representative" if raw.lower() == "representative" else names[pick],
            "rep": names[pick], "of": len(group)}


def _emit(result: dict) -> None:
    if _SCOPE:
        result = {**result, "workload_scope": dict(_SCOPE)}
    if _RESUSAGE:
        # Build-time resource readings of the student kernel's build (cuobjdump
        # -res-usage, see load_cuda_entry): top-level key, per-kernel entries.
        # A correctness FAIL still carries it — the spill cliff is visible before
        # any timing would confirm it.
        result = {**result, "resusage": list(_RESUSAGE)}
    print("\n=== JSON ===")
    print(json.dumps(result, indent=2, allow_nan=True))


def precision_on(problem: dict) -> bool:
    """Whether this card measures the precision record: by default YES -- the strict
    contract is every task's default (WP24: the board-default-off semantic is gone
    with the board). problem.json ``precision_contract`` = "off" turns the measure
    off for the card. ``FK_PRECISION=0`` counts only as read at import
    (``PRECISION_ENV_OFF``): the kernel module, imported later, cannot switch it off."""
    return (str(problem.get("precision_contract") or "strict").strip().lower()
            in ("strict", "official")
            and not PRECISION_ENV_OFF)



# --- WP9: the timing-fresh-input recheck (the agent-side cheat insurance) ---------
#
# Correctness and timing share ONE workload instance (the ``sets`` the judge built
# are the sets the windows time), so a module-level stateful kernel — compute the
# verdict once on the judged tensors, then serve the timed calls from a side
# buffer keyed on data pointers / shapes / module state — passes correctness,
# times like a no-op, and is caught by nothing else (harness_guard scans files,
# prep snapshots catch mutation, host_bound catches host-only paths). After ALL
# timing windows close, the bench re-runs the correctness protocol in a
# FRESH PROCESS: fresh allocation, fresh module state — every pointer moved, so a
# pointer-keyed cache misses (layer 1). A card whose inputs are DRAWN (the judge's
# make_inputs) also re-draws at a different seed (FRESH_SEED_ENV; file-based
# inputs carry fresh content by re-reading alone), which catches content-keyed
# caches too (layer 2). Both layers use the card's
# own input code — no task invariant can be broken by construction. A fail is a
# REWARD-HACK reading: the driver records the iteration as REVERT and marks the
# row suspicious. The recheck NEVER runs inside the timing windows (contract C5)
# and does not touch measure_paired.
FRESH_RECHECK_TIMEOUT_S = 1800
_VERDICT_LINE = "FRESH-RECHECK-VERDICT "


def _fresh_recheck_mode(problem: dict) -> str:
    """Which layers this card's recheck runs: 'regen' when the draw's CONTENT
    differs from the timed draw (the judge's make_inputs draws per group),
    'process' when only the allocation is fresh (file-based inputs)."""
    return "regen" if _task_custom(problem)[1] is not None else "process"


def _fresh_child(problem: dict, *, seed: int | None) -> "dict | None":
    """One fresh-process correctness run over the SAME card: returns the parsed
    verdict, or None when the child could not produce one (env failure — a 'no
    reading', never counted as a fail)."""
    env = dict(os.environ)
    env[FRESH_RECHECK_ENV] = "1"
    if seed is not None:
        env[FRESH_SEED_ENV] = str(seed)
    bench = (Path(__file__).parent / "bench.py").resolve()
    root = (Path(__file__).parent.parent).resolve()   # the workspace (eval/ below it)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "eval")]
        + [p for p in [env.get("PYTHONPATH", "")] if p])
    try:
        proc = subprocess.run(
            [sys.executable, str(bench), "--mode", "correctness", "--fresh-recheck"],
            cwd=str(root), env=env, capture_output=True, text=True,
            timeout=int(os.environ.get("FK_FRESH_RECHECK_TIMEOUT_S",
                                       FRESH_RECHECK_TIMEOUT_S)))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"pass": None, "detail": f"recheck child did not run: {type(exc).__name__}: {exc}"}
    for line in proc.stdout.splitlines():
        if line.startswith(_VERDICT_LINE):
            try:
                return json.loads(line[len(_VERDICT_LINE):])
            except ValueError:
                break
    return {"pass": None,
            "detail": f"recheck child printed no verdict (rc={proc.returncode})"}


def _fresh_recheck_block(problem: dict) -> "dict | None":
    """The parent's recheck: layer 1 always (fresh process, fresh allocation);
    layer 2 (a different seed) whenever the card's inputs are drawn. The result
    rides the eval JSON's top-level ``fresh_recheck``; pass=None layers are
    reported, never scored."""
    if os.environ.get(FRESH_RECHECK_ENV):
        return None                      # a child never nests
    mode = _fresh_recheck_mode(problem)
    child = _fresh_child(problem, seed=None)
    return {"mode": mode, "layers": [child], "pass": child.get("pass")}


def _run_fresh_recheck_child(problem: dict, args) -> int:
    """The child side (--fresh-recheck): the correctness protocol over the
    timed workloads, one machine-readable verdict line, no nested recheck, no
    timing. Exit code stays 0: the parent reads the verdict, not the rc. The
    device is the job's GPU; FK_FRESH_RECHECK_DEVICE exists so the protocol can
    be rehearsed on a CPU box (tests) without touching the GPU path."""
    device = os.environ.get("FK_FRESH_RECHECK_DEVICE", "cuda")
    impl, dps = _impl(problem, "kernel")
    try:
        verdict, _ = _correctness(problem, impl, dps, device)
    except Exception as exc:  # noqa: BLE001 — the impl is foreign code
        print(_VERDICT_LINE + json.dumps({"pass": None,
                                          "detail": f"{type(exc).__name__}: {exc}"}))
        return 0
    print(_VERDICT_LINE + json.dumps({"pass": bool(verdict.get("pass")),
                                      "status": verdict.get("status"),
                                      "seed": verdict.get("seed")}))
    return 0


def _correctness(problem: dict, impl, dps: bool, device):
    """The correctness protocol (task_correctness: every input group of every shape).
    With the precision contract on (``precision_on`` -- the default), a timed workload
    also gets the precision record (eval/precision.py) on the same input sets plus the
    stress profiles -- attached to the verdict as ``precision``, never changing
    ``pass``. Not on the reference timed alone (``_no_precision``)."""
    if not precision_on(problem) or problem.get("_no_precision"):
        return task_correctness(problem, impl, dps, device)
    P = _precision()
    if P is None:
        out = task_correctness(
            problem, impl, dps, device)
        out[0]["precision"] = {"status": "unavailable",
                               "why": "eval/precision.py is missing (the card predates the "
                                      "precision gate: copy it from src/forgekernel/card_template/eval/)"}
        return out
    prec = P.PrecisionRecorder()
    verdict, sets = task_correctness(
        problem, impl, dps, device, prec=prec)
    if verdict.get("pass"):
        _precision_stress(problem, impl, dps, device, prec)
    verdict["precision"] = prec.block()
    return verdict, sets


# The precision record (eval/precision.py): the first PRECISION_SETS input sets the
# judge runs, then one input set per stress profile (seeded, the RNG state restored
# around it, so the protocol's own draws never move).
PRECISION_SETS = 3
PRECISION_STRESS_SEED = 7100
PRECISION_ENV = "FK_PRECISION"
# Read once, at import -- before any kernel module runs in this process.
PRECISION_ENV_OFF = os.environ.get(PRECISION_ENV, "").strip().lower() in ("0", "off")


def _precision_snapshot(prec, inp: list):
    """A host copy of ``inp`` taken before the reference or the kernel touches it (None
    when nothing is measured): the float64 truth runs on the inputs as drawn, not on what
    an in-place definition (a residual or cache written back) left in them."""
    return _precision().snapshot(inp) if prec is not None else None


def _precision_measure(problem: dict, prec, case: str, outs: list, refs: list, ref_run,
                       pristine, device, stress: str | None = None) -> None:
    P = _precision()
    names = list(problem["definition"]["outputs"])
    prec.measure(case, outs, refs,
                 lambda: P.truth_of(ref_run, P.restore(pristine),
                                    normalize=lambda r: normalize_outputs(problem, r, device)),
                 names=names, stress=stress)


def _precision_stress(problem: dict, impl, dps: bool, device, prec) -> None:
    P = _precision()
    ref_run, custom = reference_callables(problem)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    for j, profile in enumerate(P.STRESS_PROFILES):
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(PRECISION_STRESS_SEED + j)
            touched: list = []
            try:
                inp = gen_inputs(problem, device, custom_inputs_fn=custom, stress=profile,
                                 touched=touched)
            except Exception as e:  # noqa: BLE001
                prec.unavailable("stress", f"inputs: {type(e).__name__}: {e}", stress=profile)
                continue
            if not touched:
                prec.unavailable("stress", "no plain random float input to stress",
                                 stress=profile)
                continue
            pristine = _precision_snapshot(prec, inp)
            try:
                refs = call_impl(problem, ref_run, inp, False, device, no_grad=True)
            except Exception as e:  # noqa: BLE001
                prec.unavailable("stress", f"the reference raised: {type(e).__name__}: {e}",
                                 stress=profile)
                continue
            try:
                outs = call_impl(problem, impl, inp, dps, device, no_grad=True)
            except Exception as e:  # noqa: BLE001 — the solution is foreign code
                prec.unavailable("stress", f"the kernel raised on the stressed input: "
                                           f"{type(e).__name__}: {str(e)[:200]}", stress=profile)
                continue
            _precision_measure(problem, prec, "stress", outs, refs, ref_run, pristine, device,
                               stress=profile)


def _correctness_json(verdict: dict) -> dict:
    out = {k: v for k, v in verdict.items() if k not in ("checks", "precision")}
    out["max_diff"] = verdict.get("max_abs")
    out["checks"] = verdict.get("checks", [])[:40]
    if verdict.get("pass"):
        out.pop("error", None)
    return out


def judge_workloads(problem: dict, impl, dps: bool, device) -> tuple[dict, list]:
    """The correctness protocol on every shape of the card in turn, stopping at the
    first failure; (verdict, [(view, its timed sets), ...] -- the timed shapes only).
    A one-shape card: ``_correctness`` itself."""
    views = workload_views(problem)
    if len(views) == 1:
        verdict, sets = _correctness(problem, impl, dps, device)
        return verdict, [(problem, sets)]
    per, runs, first_bad = [], [], None
    for v in views:
        verdict, sets = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        runs.append((v, sets))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
    return group_verdict(per, len(views), first_bad), runs


def _view_name(problem: dict) -> str:
    """A one-workload card's own workload, named for the verdict rows."""
    g = (problem.get("group") or [None])[0]
    return str(g.get("name")) if g else str(problem["workload"].get("uuid"))


def group_verdict(per: list, n_views: int, first_bad=None) -> dict:
    """One verdict over a group's per-workload verdicts [(view, verdict), ...]: the first
    failure's status and error (prefixed with its workload), the worst error stats, and a
    compact per-workload list."""
    out = {"pass": first_bad is None and len(per) == n_views, "status": "PASSED",
           "max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0, "checks": [],
           "n_workloads": n_views, "workloads": []}
    for v, verdict in per:
        name = v["_wl_name"]
        out["workloads"].append({"name": name, "status": verdict.get("status"),
                                 "max_abs": verdict.get("max_abs"),
                                 "matched_ratio": verdict.get("matched_ratio")})
        for k in ("max_abs", "max_rel"):
            x = verdict.get(k)
            if isinstance(x, (int, float)) and not math.isnan(x):
                out[k] = max(out[k], float(x))
        if isinstance(verdict.get("matched_ratio"), (int, float)):
            out["matched_ratio"] = min(out["matched_ratio"], verdict["matched_ratio"])
        out["checks"] += [{"workload": name, **c} for c in verdict.get("checks", [])
                          if isinstance(c, dict)]
        for k in ("tolerance", "seed"):
            out.setdefault(k, verdict.get(k))
    if (prec := precision_result(per)) is not None:
        out["precision"] = prec
    if first_bad is not None:
        v, verdict = first_bad
        axes = json.dumps(v["workload"].get("axes"))
        out["status"] = verdict.get("status")
        out["error"] = f"workload {v['_wl_name']} {axes}: {verdict.get('error')}"
        if len(per) < n_views:
            out["error"] += f" ({n_views - len(per)} later workload(s) not run)"
    return out


def precision_result(per: list) -> dict | None:
    """The eval JSON's top-level ``precision`` block over [(view, verdict), ...]: the
    timed workloads' records (a group card's merged, rows named by workload)."""
    named = [(v.get("_wl_name") or _view_name(v), x.get("precision")) for v, x in per
             if isinstance(x.get("precision"), dict)]
    if not named:
        return None
    P = _precision()
    if len(named) == 1 or P is None or any(b.get("schema") is None for _, b in named):
        return named[0][1]
    return P.merge_blocks(named)


def _say_precision(block: dict | None) -> None:
    if not block:
        return
    w = block.get("worst") or {}
    worst = (f"; worst {w.get('case')}{' [' + w['stress'] + ']' if w.get('stress') else ''} "
             f"{w.get('output')}: e_cand {w.get('e_cand')} vs e_ref {w.get('e_ref')} "
             f"(limit {w.get('limit')})") if w else ""
    status = str(block.get("status", "?")).upper()
    print(f"[bench] precision: {status} ({block.get('n_checks', 0)} checks, truth "
          f"{block.get('truth')}){worst}"
          + (f" — {block.get('why')}" if block.get("why") and status != "PASS" else "")
          + (" (recorded only: [loop] precision_contract decides what it costs)"
             if status == "FAIL" else ""))


def group_performance_block(problem: dict, blocks: list) -> dict:
    """A multi-shape card's ``performance``: the per-shape blocks (each with its
    ``name``) under ``workloads``, and the gate keys at the top level as the geometric
    mean over them (timing.workloads_performance's contract). ``vs_ref_device`` =
    geomean reference / geomean kernel = the geomean of the per-shape ratios."""
    out: dict = {}
    for k in _GEO_KEYS:
        vals = [b.get(k) for b in blocks]
        out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                               for x in vals) else None
    bases = list(dict.fromkeys(b.get("device_window_basis") for b in blocks))
    out["device_window_basis"] = bases[0] if len(bases) == 1 \
        else "mixed: " + ", ".join(map(str, bases))
    hb = [b["host_bound_ratio"] for b in blocks if b.get("host_bound_ratio") is not None]
    out["host_bound_ratio"] = max(hb) if hb else None
    paired = [b for b in blocks if b.get("ref_device_ms")]
    if len(paired) == len(blocks):
        out["in_job_ref"] = "paired"
        for k in _REF_GEO_KEYS:
            vals = [b.get(k) for b in blocks]
            out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                                   for x in vals) else None
        # the geomean of the per-shape ratios (each from unrounded readings): dividing
        # the two rounded geomeans would move a us-scale ratio by tenths of a percent
        for k in ("vs_ref", "vs_ref_device"):
            vals = [b.get(k) for b in blocks]
            out[k] = _round(_geomean(vals)) if all(isinstance(x, (int, float)) and x > 0
                                                   for x in vals) else None
        rhb = [b["ref_host_bound_ratio"] for b in blocks
               if b.get("ref_host_bound_ratio") is not None]
        out["ref_host_bound_ratio"] = max(rhb) if rhb else None
    else:
        missing = [b["name"] for b in blocks if not b.get("ref_device_ms")]
        out["in_job_ref"] = (blocks[0].get("in_job_ref") if len(missing) == len(blocks)
                             else f"partial: no reference reading on {', '.join(missing)}")
    out["aggregate"] = "geomean"
    out["n_workloads"] = len(blocks)
    lbs = [b["bench"] for b in blocks]
    lb = {k: lbs[0][k] for k in ("timer", "protocol", "definition", "reference", "clock")
          if k in lbs[0]}
    lb["reading"] = f"geometric mean over {len(blocks)} shapes of: {lbs[0].get('reading')}"
    lb["workloads"] = [b["name"] for b in blocks]
    out["bench"] = lb
    out["workloads"] = blocks
    return out


def _geomean(values) -> float:
    vals = [float(x) for x in values]
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


# Gate keys a multi-shape card reports as the geomean over its shapes (host-bound
# ratios as the worst shape's, so the precondition audit is never softened by averaging).
_GEO_KEYS = ("bwd_median_ms", "min_ms", "device_median_ms", "host_enqueue_ms")
_REF_GEO_KEYS = ("ref_median_ms", "ref_device_ms", "ref_host_enqueue_ms")


# --- the forgekernel task suite ------------------------------------------------------
#
# The task's problem.json (written by `forge run`) carries:
#
#   definition   the signature (inputs / outputs: shape over the axes, dtype) and, as
#                ``reference``, the JUDGE: the PyTorch semantics the teacher wrote from the
#                task's description, with ``make_inputs`` (custom_inputs_entrypoint)
#   workload / group
#                the task's shapes (a group card when there are several); each workload's
#                ``input_groups``: {"name", "kind": "file", "path"} -- every input from a
#                safetensors file -- or {"name", "kind": "generated", "seed", "intent"} --
#                make_inputs(axes, group=<name>, seed=<seed>, device=<device>)
#   tolerance    per output: TASK_TOL_MARGIN x the largest normalized error the reference
#                showed against the judge on every input group (``--mode calibrate``),
#                floored per dtype
#   timing       "device" (timing.py's pipelined device window) or "e2e" (per call)
#   reference_impl
#                the task's reference (sealed: decoded in memory, timed beside the kernel)
#
# Correctness: every input group of every shape, the judge on a copy of the inputs, then the
# implementation; per output, shape and dtype must match, non-finite values must sit where
# the judge has the same ones, and the normalized error max|y - j| / max|j| (finite
# positions) must not exceed the output's tolerance. Integer and bool outputs match exactly.
# Timing: per shape, timing.measure_windows_workloads over its input groups (burn-in, then
# reference / kernel / kernel / reference on each), the geomean over the groups; across
# shapes, the geomean over the shapes. Every input group of every shape takes part.

def _lazy_output(outs: list):
    for t in outs:
        if type(t) is not torch.Tensor:
            return f"output is {type(t).__name__}, not torch.Tensor (lazy evaluation)"
    return None


# Two units of the output dtype's rounding, relative to the output's scale: the tolerance
# never falls below it, so a reference that matches the judge bit for bit does not demand
# the same of a kernel that only reorders a sum.
TASK_FLOOR = {"float64": 2 * 2.0 ** -53, "float32": 2 * 2.0 ** -24, "float16": 2 * 2.0 ** -11,
              "bfloat16": 2 * 2.0 ** -8, "float8_e4m3fn": 2 * 2.0 ** -4,
              "float8_e5m2": 2 * 2.0 ** -3}
# The tolerance is this many times the reference's largest error. The reference's error is
# one draw from its own summation order on the card's inputs: another correct kernel sums
# in another order and lands as often above that draw as below it, and on a fresh seed the
# reference itself can land above it. Twice the error absorbs that spread and stays far
# below what a lower-precision path costs.
TASK_TOL_MARGIN = 2.0
# A reference further than this from the judge on any output does not compute what the
# judge computes: calibration reports no match instead of turning the gap into a tolerance.
TASK_MATCH_CAP = 0.05
TASK_EXACT_DTYPES = ("int64", "int32", "int16", "int8", "bool")
BASIS_PLAIN = "plain_batch"


def task_groups(problem: dict) -> list[dict]:
    return list(problem["workload"].get("input_groups") or [])


def first_inputs(problem: dict, device) -> list:
    """One input set of the card's first shape: its first input group."""
    view = _task_shapes(problem)[0]
    return task_inputs(view, task_groups(view)[0], device)


def _task_custom(problem: dict):
    defn = problem["definition"]
    ns = _exec_source(defn["reference"], f"<{defn['name']} judge>")
    return ns["run"], ns[defn.get("custom_inputs_entrypoint") or "make_inputs"]


def task_inputs(problem: dict, group: dict, device, make_inputs=None) -> list:
    """One input group of the problem's workload, in definition order, checked against
    the signature (a wrong shape or dtype is the judge's or the file's fault, raised as
    ValueError -- never charged to the kernel)."""
    defn = problem["definition"]
    axes = resolved_axes(defn, problem["workload"].get("axes") or {})
    shapes = tensor_shapes(defn["inputs"], axes)
    dev = torch.device(device)
    if group.get("kind") == "file":
        path = Path(group["path"])
        if not path.is_absolute():
            path = Path(problem["_dir"]) / path
        got = read_safetensors(path.resolve())
    else:
        if make_inputs is None:
            make_inputs = _task_custom(problem)[1]
        got = make_inputs(dict(axes), group=group["name"],
                          seed=int(group.get("seed", 0)) + _recheck_seed_shift(),
                          device=dev)
        if not isinstance(got, dict):
            raise ValueError(f"make_inputs returned {type(got).__name__}, not a dict of inputs")
    out = []
    for name, spec in defn["inputs"].items():
        if name not in got:
            raise ValueError(f"input group {group['name']!r}: no input {name!r}")
        v = got[name]
        if shapes[name] is None:
            out.append(v.item() if isinstance(v, torch.Tensor) else v)
            continue
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"input group {group['name']!r}: {name} is "
                             f"{type(v).__name__}, not a tensor")
        want = torch_dtype(spec["dtype"])
        if tuple(v.shape) != tuple(shapes[name]) or v.dtype != want:
            raise ValueError(f"input group {group['name']!r}: {name} is {tuple(v.shape)} "
                             f"{v.dtype}, the signature says {tuple(shapes[name])} {want}")
        out.append(v.to(dev).contiguous())
    return out


_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
              "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
              "F8_E5M2": torch.float8_e5m2, "I64": torch.int64, "I32": torch.int32,
              "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def read_safetensors(path: Path) -> dict:
    """name -> tensor (CPU) of a safetensors file, read with torch alone: an image without
    the safetensors package still reads a task's input files."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    body = raw[8 + n:]
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPES.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: {name} has unsupported dtype {spec['dtype']}")
        start, end = spec["data_offsets"]
        flat = torch.frombuffer(bytearray(body[start:end]), dtype=dtype) if end > start \
            else torch.empty(0, dtype=dtype)
        out[name] = flat.reshape(spec["shape"])
    return out


def _clone(inp: list) -> list:
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inp]


def task_error(out, ref, dtype: str) -> tuple[float, str | None]:
    """(normalized error, why it fails regardless of tolerance or None)."""
    if tuple(out.shape) != tuple(ref.shape):
        return math.inf, f"shape {tuple(out.shape)} != judge {tuple(ref.shape)}"
    if out.dtype != ref.dtype:
        return math.inf, f"dtype {out.dtype} != judge {ref.dtype}"
    if dtype in TASK_EXACT_DTYPES:
        bad = int((out != ref).sum().item())
        return (0.0, None) if bad == 0 else (math.inf, f"{bad} element(s) differ (exact dtype)")
    x, y = out.to(torch.float64), ref.to(torch.float64)
    yf, xf = torch.isfinite(y), torch.isfinite(x)
    if bool((yf & ~xf).any()):
        return math.inf, f"{int((yf & ~xf).sum())} non-finite value(s) where the judge is finite"
    nonfin = ~yf
    if bool(nonfin.any()):
        same = (torch.isnan(x) & torch.isnan(y)) | (x == y)
        if not bool(same[nonfin].all()):
            return math.inf, "non-finite values differ from the judge's"
    if not bool(yf.any()):
        return 0.0, None
    scale = float(y[yf].abs().max().item())
    err = float((x[yf] - y[yf]).abs().max().item())
    if err == 0.0:
        return 0.0, None
    return (err / scale if scale > 0 else math.inf), None


def task_tolerance(problem: dict) -> dict:
    """Per output: the calibrated tolerance, never below the dtype floor."""
    cal = (problem.get("tolerance") or {}).get("outputs") or {}
    out = {}
    for name, spec in problem["definition"]["outputs"].items():
        floor = TASK_FLOOR.get(spec["dtype"], 0.0)
        out[name] = max(float(cal.get(name, 0.0)), floor)
    return out


def task_correctness(problem: dict, impl, dps: bool, device, prec=None) -> tuple[dict, list]:
    """Every input group of the problem's workload: the judge, then ``impl``. Returns
    (verdict, the input sets in group order -- every one is timed)."""
    judge, make_inputs = _task_custom(problem)
    tol = task_tolerance(problem)
    names = list(problem["definition"]["outputs"])
    dtypes = {n: s["dtype"] for n, s in problem["definition"]["outputs"].items()}
    verdict = {"pass": False, "status": "PASSED", "max_abs": 0.0, "max_rel": 0.0,
               "matched_ratio": 1.0, "checks": [],
               "tolerance": {"outputs": tol, "rule": "max|y - judge| / max|judge| <= tolerance "
                             "per output (finite positions); non-finite values where the judge "
                             "has them; integer / bool outputs exact"}}
    sets: list = []

    def fail(status, error):
        verdict.update(status=status, error=error)
        return verdict, sets

    for gi, group in enumerate(task_groups(problem)):
        try:
            inp = task_inputs(problem, group, device, make_inputs)
        except Exception as exc:  # noqa: BLE001 -- the judge's / the file's, not the kernel's
            return fail("INPUT_ERROR", f"input group {group['name']}: {type(exc).__name__}: {exc}")
        sets.append(inp)
        pristine = _precision_snapshot(prec, inp) if prec is not None and gi < PRECISION_SETS else None
        refs = call_impl(problem, judge, _clone(inp), False, device, no_grad=True)
        try:
            outs = call_impl(problem, impl, inp, dps, device, no_grad=True)
        except Exception as exc:  # noqa: BLE001 — the implementation is foreign code
            return fail("RUNTIME_ERROR", f"input group {group['name']}: {type(exc).__name__}: {exc}")
        if (why := _lazy_output(outs)):
            return fail("REWARD_HACK", why)
        if pristine is not None:
            _precision_measure(problem, prec, f"group_{group['name']}", outs, refs, judge,
                               pristine, device)
        for name, o, r in zip(names, outs, refs):
            e, why = task_error(o, r, dtypes[name])
            verdict["checks"].append({"group": group["name"], "output": name,
                                      "error": None if math.isinf(e) else e,
                                      "tolerance": tol[name], **({"why": why} if why else {})})
            if not math.isinf(e):
                verdict["max_rel"] = max(verdict["max_rel"], e)
            if why or e > tol[name]:
                return fail("INCORRECT_NUMERICAL",
                            f"input group {group['name']} output {name}: "
                            + (why or f"normalized error {e:.4g} > tolerance {tol[name]:.4g}"))
    verdict["pass"] = True
    return verdict, sets


def _task_side(problem: dict, sets: list, kernel, kernel_dps: bool, ref=None,
               ref_dps: bool = False, device="cuda") -> dict:
    """One shape's performance block: timing.py over its input groups, each paired with
    the reference when one is timed."""
    import timing

    def calls(fn, dps, inp):
        if fn is None:
            return None
        outs = allocate_outputs(problem, device) if dps else []
        return [lambda: fn(*inp, *outs)]

    groups = [g["name"] for g in task_groups(problem)]
    work = [(name, calls(kernel, kernel_dps, inp), calls(ref, ref_dps, inp))
            for name, inp in zip(groups, sets)]
    result = timing.measure_windows_workloads(work, burn_in_s=BURN_IN_S)
    block = timing.workloads_performance(result)
    block["device_window_basis"] = BASIS_PLAIN
    block["bench"] = {
        "timer": "timing.py",
        "protocol": {"aggregate": "geomean over input groups", "window": problem.get("timing")},
        "reading": "geometric mean over the shape's input groups of timing.py's windows",
        "kernel_per_set_ms": [_round(w.get("device_median_ms")) for w in block["workloads"]],
        "ref_per_set_ms": [_round(w.get("ref_device_ms")) for w in block["workloads"]
                           if w.get("ref_device_ms") is not None],
        "definition": problem["definition"]["name"],
        "workload": problem["workload"].get("uuid"),
        "reference": (problem.get("reference_impl") or {}).get("name", "none"),
        "clock": clock_state(),
    }
    return block


def _task_shapes(problem: dict) -> list[dict]:
    """Every shape of the task as a problem view (a one-shape task is its own view)."""
    views = workload_views(problem)
    if len(views) == 1:
        views = [dict(problem, _wl_name=str(problem["workload"].get("uuid")))]
    return views


def _task_aggregate(problem: dict, blocks: list) -> dict:
    """One shape: its own block (geomean over its input groups). Several: the geomean over
    the shapes, each shape's block under ``workloads``."""
    if len(blocks) == 1:
        return {k: v for k, v in blocks[0].items() if k not in ("name", "axes")}
    return group_performance_block(problem, blocks)


def _run_task(problem: dict, args, impl, dps: bool, ref_fn, ref_dps: bool, timed_ref,
              device) -> int:
    """A task's correctness + timing, shape by shape, stopping at the first shape that
    fails."""
    views = _task_shapes(problem)
    per, blocks, first_bad = [], [], None
    for v in views:
        verdict, sets = _correctness(v, impl, dps, device)
        per.append((v, verdict))
        print(f"[task] {v['_wl_name']} {json.dumps(v['workload'].get('axes'))}: correctness "
              f"{verdict['status']}" + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
        if not verdict["pass"]:
            first_bad = (v, verdict)
            break
        if args.mode == "correctness":
            continue
        block = _task_side(v, sets, impl, dps, timed_ref, ref_dps, device)
        blocks.append({"name": v["_wl_name"], "axes": v["workload"].get("axes"), **block})
        print(f"[task] {v['_wl_name']}: kernel device {block['device_median_ms']} ms, e2e "
              f"{block['bwd_median_ms']} ms"
              + (f"; reference device {block['ref_device_ms']} ms, x{block['vs_ref_device']}"
                 if block.get("ref_device_ms") else f" [reference: {block.get('in_job_ref')}]"))
    verdict = group_verdict(per, len(views), first_bad)
    result = {"correctness": _correctness_json(verdict), "performance": None}
    print(f"[task] correctness: {verdict['status']} ({len(per)}/{len(views)} shapes judged)"
          + (f" — {verdict.get('error')}" if not verdict["pass"] else ""))
    if not verdict["pass"] or args.mode == "correctness":
        _emit(result)
        return 0
    perf = _task_aggregate(problem, blocks)
    result["performance"] = perf
    print(f"[task] {len(blocks)} shape(s): kernel device {perf['device_median_ms']} ms, e2e "
          f"{perf['bwd_median_ms']} ms"
          + (f"; reference device {perf['ref_device_ms']} ms, x{perf['vs_ref_device']}"
             if perf.get("vs_ref_device") else f" [reference: {perf.get('in_job_ref')}]")
          + f"; clock {clock_state()}")
    _maybe_fresh_recheck(result, problem, args)   # --- WP9: after ALL windows
    _emit(result)
    return 0


def _task_ref_alone(problem: dict, ref_fn, ref_dps: bool, device) -> int:
    """--mode ref: the reference alone, every input group of every shape, in the same
    geomean structure the bench reports (so the archive and the gate name the same
    workloads)."""
    blocks = []
    for v in _task_shapes(problem):
        verdict, sets = _correctness(dict(v, _no_precision=True), ref_fn, ref_dps, device)
        if not verdict["pass"]:
            _emit({"error": f"the reference fails the judge on {v['_wl_name']}: "
                            f"{verdict.get('error')}"})
            return 1
        block = _task_side(v, sets, ref_fn, ref_dps, None, False, device)
        blocks.append({"name": v["_wl_name"], "axes": v["workload"].get("axes"), **block})
    perf = _task_aggregate(problem, blocks)
    out = {"median_ms": perf["bwd_median_ms"], "device_median_ms": perf["device_median_ms"],
           "device_window_basis": BASIS_PLAIN, "host_enqueue_ms": perf.get("host_enqueue_ms"),
           "bench_timer": "timing.py", "clock": clock_state(),
           "aggregate": "geomean", "n_workloads": perf.get("n_workloads", len(blocks)),
           "workloads": [{"name": w.get("name"), "median_ms": w.get("bwd_median_ms"),
                          "device_median_ms": w.get("device_median_ms")}
                         for w in perf.get("workloads", [])]}
    _emit(out)
    return 0


def _task_calibrate(problem: dict, ref_fn, ref_dps: bool, device) -> int:
    """--mode calibrate: the reference against the judge on every input group of every
    shape; per output the largest normalized error, whether it is close enough to call a
    match (TASK_MATCH_CAP), and the tolerance it implies (TASK_TOL_MARGIN x that error,
    never below the dtype floor)."""
    outputs = problem["definition"]["outputs"]
    worst = {n: 0.0 for n in outputs}
    rows, problems = [], []
    for v in _task_shapes(problem):
        judge, make_inputs = _task_custom(v)
        for group in task_groups(v):
            where = f"{v['_wl_name']}/{group['name']}"
            try:
                inp = task_inputs(v, group, device, make_inputs)
                refs = call_impl(v, judge, _clone(inp), False, device, no_grad=True)
                outs = call_impl(v, ref_fn, inp, ref_dps, device, no_grad=True)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{where}: {type(exc).__name__}: {exc}")
                continue
            for name, o, r in zip(outputs, outs, refs):
                e, why = task_error(o, r, outputs[name]["dtype"])
                rows.append({"shape": v["_wl_name"], "group": group["name"], "output": name,
                             "error": None if math.isinf(e) else e, **({"why": why} if why else {})})
                if why:
                    problems.append(f"{where} output {name}: {why}")
                elif e > TASK_MATCH_CAP:
                    problems.append(f"{where} output {name}: normalized error {e:.4g} > "
                                    f"{TASK_MATCH_CAP} -- the judge and the reference disagree")
                else:
                    worst[name] = max(worst[name], e)
    tol = {n: max(TASK_TOL_MARGIN * worst[n], TASK_FLOOR.get(outputs[n]["dtype"], 0.0))
           for n in outputs}
    _emit({"calibration": {"match": not problems, "problems": problems[:40],
                           "max_error": worst, "margin": TASK_TOL_MARGIN, "tolerance": tol,
                           "floor": {n: TASK_FLOOR.get(s["dtype"], 0.0) for n, s in outputs.items()},
                           "cap": TASK_MATCH_CAP, "rows": rows[:400]}})
    print(f"[task] calibrate: {'MATCH' if not problems else 'NO MATCH'}; tolerance {tol}")
    return 0


def run_bench(argv=None, problem_path=None) -> int:
    """The card's eval/bench.py: correctness then timing, the task's way."""
    p = argparse.ArgumentParser(prog="bench.py")
    p.add_argument("--mode", choices=("all", "correctness", "perf", "ref", "calibrate"),
                   default="all",
                   help="calibrate: the reference against the judge (tolerance freeze)")
    p.add_argument("--impl", choices=("kernel", "ref"), default="kernel",
                   help="ref: the in-job reference judged and timed AS the implementation "
                        "(card self-test; vs_ref_device must read ~1.0)")
    # WP9 fresh-recheck child mode: the parent's LAST step spawns a fresh process
    # in this mode after all timing windows closed; the child runs the correctness
    # protocol once and prints one FRESH-RECHECK-VERDICT line.
    p.add_argument("--fresh-recheck", action="store_true",
                   help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    problem = load_problem(problem_path)
    if args.fresh_recheck:
        return _run_fresh_recheck_child(problem, args)
    _SCOPE.clear()
    if scope := apply_workload_scope(problem):
        _SCOPE.update(scope)
        which = "representative workload" if scope["workloads"] == "representative" else "workload"
        print(f"[bench] WORKLOAD SCOPE: {which} {scope['rep']} only "
              f"(1 of {scope['of']}; {WORKLOAD_SCOPE_ENV}) — a meta reading, never a gate")
    device = "cuda"
    print(f"[bench] definition={problem['definition']['name']} "
          f"{describe_card_workloads(problem)}")

    ref_fn, ref_dps = _impl(problem, "ref")
    if args.mode == "calibrate":
        return _task_calibrate(problem, ref_fn, ref_dps, device)
    if args.mode == "ref":
        return _task_ref_alone(problem, ref_fn, ref_dps, device)
    if args.impl == "kernel" and (why := kernel_fence(problem)):
        print(f"[bench] correctness: FENCE — {why}")
        _emit({"correctness": {"pass": False, "status": "FENCE", "error": why},
               "performance": None})
        return 0
    timed_ref = ref_fn if problem.get("time_reference", True) else None
    if timed_ref is not None and args.mode != "correctness" and \
            (problem.get("reference_impl") or {}).get("kind") not in (None, "definition"):
        warm_reference(problem, ref_fn, ref_dps, device)
    try:
        impl, dps = (ref_fn, ref_dps) if args.impl == "ref" else _impl(problem, "kernel")
    except Exception as e:  # noqa: BLE001
        if Path(problem.get("kernel_file", "kernel.py")).suffix not in CUDA_KERNEL_SUFFIXES:
            raise
        why = f"{type(e).__name__}: {e}"
        print(f"[bench] correctness: COMPILE_ERROR — {why[-6000:]}")
        _emit({"correctness": {"pass": False, "status": "COMPILE_ERROR", "error": why[-6000:]},
               "performance": None})
        return 0
    return _run_task(problem, args, impl, dps, ref_fn, ref_dps, timed_ref, device)


def _maybe_fresh_recheck(result: dict, problem: dict, args) -> None:
    """The timing-fresh-input recheck (WP9), attached after ALL timing windows
    closed on a PASSING kernel the student wrote. A self-test (--impl ref) has
    nothing to recheck (the reference is the trusted side), and a correctness
    FAIL never reaches timing."""
    if args.impl != "kernel" or not (result.get("correctness") or {}).get("pass"):
        return
    if not (result.get("performance") or {}):
        return
    block = _fresh_recheck_block(problem)
    if block is not None:
        result["fresh_recheck"] = block
        print(f"[bench] fresh recheck ({block['mode']}): "
              + ("PASS" if block.get("pass") is True else
                 "FAIL" if block.get("pass") is False else "no reading"))



def ncu_window(problem_path=None, calls: int = 8, which: str = "kernel") -> None:
    """eval/ncu_run.py: the kernel's calls on one input set inside profiler start/stop
    (input construction and JIT warm-up stay outside the NCU window). ``which="ref"``
    profiles the in-job reference instead (a sealed reference's recon tree). A
    multi-shape card profiles its representative, or the shape ``FK_LB_WORKLOAD_SCOPE``
    names (a meta tool reading the card one shape at a time)."""
    problem = load_problem(problem_path)
    if scope := apply_workload_scope(problem):
        print(f"[bench] ncu: workload {scope['rep']} (1 of {scope['of']}; "
              f"{WORKLOAD_SCOPE_ENV})", flush=True)
    if which == "kernel" and (why := kernel_fence(problem)):
        # the bench's fence, on the profiler's path into the kernel file too
        print(f"[bench] ncu: FENCE — {why}")
        raise SystemExit(3)
    kernel, dps = _impl(problem, which)
    # every input group of every shape (the one shape a meta tool scoped to, if any),
    # the calls spread over them: a profile of one group is a profile tuned to it
    pairs = [(v, task_inputs(v, g, "cuda")) for v in _task_shapes(problem)
             for g in task_groups(v)]
    per = max(1, calls // len(pairs))
    outs = {id(v): allocate_outputs(v, "cuda") if dps else [] for v, _ in pairs}
    for v, inp in pairs:
        for _ in range(3):
            kernel(*inp, *outs[id(v)])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for v, inp in pairs:
        for _ in range(per):
            kernel(*inp, *outs[id(v)])
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    raise SystemExit(run_bench())
