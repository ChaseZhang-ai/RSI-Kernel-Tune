#!/usr/bin/env python3
# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""External evaluation export packager (WP27).

Packages a Peregrine-Kernel optimization product (the kernel that passed the run's
full gate) into the submission format of an external evaluation suite: the NVIDIA
SOL suite (free B200 evaluation) or the FlashInfer suite. The full protocol, the
suites' judgment criteria, and the full suite names live in
``docs/external-eval.md``; the code surface says only ``sol`` / ``fib`` (the WP24
zero-residue gate greps the code surface, and the suites' names stay out of it on
purpose -- the doc is the mapping).

Design rules (WP27 task pack, 2026-11-04):

* **Manual, opt-in, offline.** This tool is human-run, never inside any automatic
  loop, and performs zero network I/O: uploading the package is the user's manual
  step. It imports nothing from ``src/peregrine_kernel`` (isolation is pinned by
  tests) and uses the standard library only.
* **Honest export.** The packaged kernel is the gated product verbatim -- the bytes
  are copied without any post-processing, and ``artifact_sha256.json`` anchors them
  (kernel bytes, task-contract files, solution JSON) plus the provenance of where
  the kernel came from (run directory, best KEEP iteration, its sha and reading).
* **Format knowledge is mined, not invented.** The solution JSON layout, the spec
  fields, the language table and the nvcc flag derivation replicate the pre-WP24
  export path (git history, commit ``82fa28d^`` — the exporter the WP24 de-boarding
  removed; the board templates at ``65d735a^``), cross-checked against the current
  card builder (``src/peregrine_kernel/cardkit.py``) and the current eval template's
  CUDA build section (``card_template/eval/benchkit.py``): the flags the card builds
  with are the flags exported.

Usage (one entry point)::

    python3 scripts/eval-export/pack.py --target sol|fib|both OUT_DIR PRODUCT ...

    # product, exactly one of:
    --run-dir .artifacts/<task>_<id> --op <op>   # auto-locate the best KEEP kernel
    --tree operators/<op>                        # an operator tree as it stands
    --kernel kernel.py                           # an explicit kernel file

    # submission metadata:
    --definition <suite-dataset-definition-name>  # e.g. the suite's definition id
    --name <solution-name> --author <author> --hardware B200

    # contract fallbacks when the tree carries no contract file:
    --entry run --inputs a,b --outputs c

Preflight, three stages (the report rides the package as ``preflight.json``):

  (a) manifest/lint static checks -- always: contract parses, kernel signature
      matches the contract's inputs/outputs, language resolvable, CUDA flags within
      the card's allowlist, Python source compiles (syntax), solution JSON
      well-formed. A failure here refuses the export.
  (b) compile check -- only when this machine has the toolchain: ``.cu`` kernels
      are built with torch cpp_extension + nvcc when both are present; Python
      kernels are imported when their imports resolve. Toolchain missing -> SKIP
      with a warning (the target machine has it; this one may not).
  (c) correctness on the task contract's inputs -- only with torch + a CUDA device:
      deterministic inputs from the contract's declared shapes/dtypes, the
      contract's own reference against the kernel, the contract's tolerance rule.
      No GPU / no runnable reference -> SKIP with a warning. This stage is a
      convenience smoke re-check, never a substitute for the run's full gate.

Exit codes: 0 = package written; 2 = refused (no package left behind).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# The tool's own version stamp (bumped with format changes).
TOOL_VERSION = "1.0"

TARGET_FIB = "fib"
TARGET_SOL = "sol"
TARGETS = (TARGET_FIB, TARGET_SOL)

# The export language table, mined from the pre-WP24 export path (git history):
# key -> (suite name for FIB spec, languages list for SOL spec, SOL dependencies).
EXPORT_LANGS = {
    "cutedsl": ("python", ["cute_dsl"], ["cutlass"]),
    "triton": ("triton", ["triton"], ["triton >= 2.3"]),
    "pytorch": ("python", ["pytorch"], ["torch"]),
    "cuda-sol": ("cuda", ["cuda_cpp"], []),
}

# --- CUDA build constants: mirrors card_template/eval/benchkit.py (the flags the
# card builds with are the flags exported; kept in sync by the fidelity test). ---

CUDA_KERNEL_SUFFIXES = (".cu",)
CUDA_DEFAULT_CFLAGS = ["-O3", "--use_fast_math"]
CUDA_DEFAULT_LDFLAGS = ["-lcuda"]
CUDA_GENCODE = "-gencode=arch=compute_100a,code=sm_100a"
CUDA_FLAGS_LINE = re.compile(r"^[ \t]*//[ \t]*fk-cuda-cflags:(.*)$", re.M)
CUDA_FLAG_OK = re.compile(
    r"-O[0-3]|--use_fast_math|-use_fast_math|--?ftz=(true|false)|--?prec-(div|sqrt)=(true|false)"
    r"|--?fmad=(true|false)|-lineinfo|--?extra-device-vectorization|--?maxrregcount=\d+"
    r"|-Xptxas=[-\w,=.]+|--ptxas-options=[-\w,=.]+|-D[A-Za-z_]\w*(=[-\w.]+)?|-U[A-Za-z_]\w*"
    r"|-std=c\+\+\d\d|--?expt-relaxed-constexpr|--?expt-extended-lambda|--?restrict"
    r"|-gencode=arch=compute_\w+,code=(sm|compute)_\w+|-arch=(sm|compute)_\w+"
    r"|--?default-stream=(legacy|null|per-thread)|-Xcompiler=-O[0-3]|-w|-Wno-[-\w]+")

LANE_NOTE = ("manual evaluation export lane: opt-in, human-run, zero network I/O; "
             "not imported by src/ and not part of any automatic loop")

# Default dtype epsilons for the correctness tolerance floor ("never below two units
# of the output dtype's rounding"): 2 * eps per dtype; exact dtypes get 0.
DTYPE_EPS = {
    "float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07,
    "float16": 9.765625e-04, "bfloat16": 7.8125e-02,
}
INT_DTYPE_RANGES = {
    "int8": (-128, 127), "int16": (-32768, 32767), "int32": (-1024, 1024),
    "int64": (-1024, 1024), "uint8": (0, 255),
}


class Refused(ValueError):
    """The export is refused (stage (a) failure or bad input); nothing is written."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- product location -----------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def best_keep(op_dir: Path) -> dict:
    """The best green KEEP row of a run's operator trajectory.

    Selection: every row with verdict KEEP and correctness True that carries a
    reading; the reading is the gate window's device value when the row has one,
    else the per-call value; the minimum wins. The chosen window is recorded --
    this tool reads the trajectory the run wrote, it does not re-judge it.
    """
    status = op_dir / "student" / "status.jsonl"
    if not status.is_file():
        raise Refused(f"{op_dir}: no student/status.jsonl (is this a run's operator tree?)")
    rows = []
    for r in _read_jsonl(status):
        if r.get("verdict") != "KEEP" or r.get("correctness") is not True:
            continue
        for window in ("device_median_ms", "bwd_median_ms"):
            if r.get(window) is not None:
                rows.append((window, float(r[window]), r))
                break
    if not rows:
        raise Refused(f"{op_dir}: no evaluated KEEP row in student/status.jsonl "
                      f"-- nothing to export")
    window, reading, row = min(rows, key=lambda t: t[1])
    clusters = sorted({t[2].get("cluster") for t in rows} - {""})
    return {"window": window, "reading_ms": reading, "iter": row.get("iter"),
            "sha": row.get("sha", ""), "cluster": row.get("cluster", ""),
            "change": row.get("change", ""), "row": row, "clusters": clusters}


def kernel_from_run(run_dir: Path, op: str) -> tuple[bytes, str, dict]:
    """The best KEEP kernel of a run: trajectory selection, then the kernel file's
    content at that iteration's sha from the run's git repositories (the student
    workspace first, then the card). When no repository holds the sha, the working
    tree's file is exported with an explicit fallback warning -- the manifest
    records exactly which bytes left."""
    op_dir = Path(run_dir) / "operators" / op
    if not op_dir.is_dir():
        raise Refused(f"{run_dir}: no operators/{op}/ under this run directory")
    keep = best_keep(op_dir)
    problem = contract_problem_file(op_dir)
    kfile = (problem or {}).get("kernel_file") or _tree_kernel_file(op_dir) or "kernel.py"
    sha = keep.get("sha") or ""
    notes = []
    if len(keep.get("clusters") or []) > 1:
        notes.append("green KEEP rows span clusters "
                     + ", ".join(keep["clusters"]) + "; absolute readings are "
                     "same-machine comparable only -- the chosen row's cluster is "
                     "recorded here")
    provenance = {"kind": "run-best-keep", "run_dir": str(Path(run_dir).resolve()),
                  "op": op, "iteration": keep["iter"], "sha": sha,
                  "selection": f"best KEEP by {keep['window']} over green KEEP rows",
                  "reading": {keep["window"]: keep["reading_ms"]},
                  "cluster": keep["cluster"], "change": keep["change"],
                  "notes": notes}
    if sha:
        for repo in (Path(run_dir) / "ws", op_dir):
            if (repo / ".git").exists():
                r = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{kfile}"],
                                   capture_output=True)
                if r.returncode == 0:
                    provenance["bytes_source"] = f"git show {sha[:12]}:{kfile} in {repo.name}"
                    return r.stdout, kfile, provenance | {"sha_found": True}
        provenance["warning"] = (f"no repository under the run holds sha {sha[:12]}; "
                                 f"exported the working tree's {kfile} instead")
    else:
        provenance["warning"] = ("the best KEEP row carries no sha; exported the "
                                 f"working tree's {kfile}")
    path = op_dir / kfile
    if not path.is_file():
        raise Refused(f"{op_dir}: kernel file {kfile} missing")
    provenance["bytes_source"] = f"working tree {path}"
    provenance["sha_found"] = False
    return path.read_bytes(), kfile, provenance


def kernel_from_tree(tree: Path) -> tuple[bytes, str, dict]:
    """The kernel file of an operator tree as it stands."""
    tree = Path(tree)
    if not tree.is_dir():
        raise Refused(f"{tree}: not a directory")
    problem = contract_problem_file(tree)
    kfile = (problem or {}).get("kernel_file") or _tree_kernel_file(tree)
    if not kfile:
        raise Refused(f"{tree}: no kernel.py / kernel.cu in this tree; pass --kernel")
    path = tree / kfile
    if not path.is_file():
        raise Refused(f"{tree}: kernel file {kfile} missing")
    provenance = {"kind": "operator-tree", "tree": str(tree.resolve()),
                  "kernel_file": kfile,
                  "bytes_source": f"working tree {path}", "sha_found": None}
    git = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if git.returncode == 0:
        provenance["tree_git_head"] = git.stdout.strip()
    return path.read_bytes(), kfile, provenance


def _tree_kernel_file(tree: Path) -> str | None:
    for name in ("kernel.py", "kernel.cu"):
        if (tree / name).is_file():
            return name
    return None


def is_cuda_kernel_file(kfile: str) -> bool:
    return Path(kfile).suffix in CUDA_KERNEL_SUFFIXES


# --- contract -------------------------------------------------------------------


def contract_problem_file(tree: Path) -> dict | None:
    """The task contract of a current card: eval/problem/problem.json."""
    path = Path(tree) / "eval" / "problem" / "problem.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: unreadable task contract ({e})")


def contract_window_toml(tree: Path) -> dict | None:
    """The timed-window contract as a signature fallback: [student] inputs/outputs."""
    path = Path(tree) / "window.toml"
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: unreadable window contract ({e})")
    student = data.get("student") or {}
    ins, outs = list(student.get("inputs") or []), list(student.get("outputs") or [])
    if not ins:
        return None
    return {"entry": "run", "inputs": ins, "outputs": outs, "dps": None,
            "kernel_file": None, "definition": None}


class Contract:
    def __init__(self, definition, entry, inputs, outputs, dps, kernel_file,
                 reference_src, tolerance, contract_files):
        self.definition = definition
        self.entry = entry
        self.inputs = inputs
        self.outputs = outputs
        self.dps = dps                  # None = decide from the kernel signature
        self.kernel_file = kernel_file
        self.reference_src = reference_src
        self.tolerance = tolerance or {}
        self.contract_files = contract_files   # [(relpath, Path)] hashed into the manifest


def read_contract(tree: Path, args) -> Contract:
    """The task contract, from the most specific source available:

    1. the card's ``eval/problem/problem.json`` (current generated cards);
    2. the tree's ``window.toml`` ([student] inputs/outputs);
    3. the command line (--entry/--inputs/--outputs) -- the mode an exported
       released kernel (``kernels/``) uses.
    """
    files: list[tuple[str, Path]] = []
    problem = contract_problem_file(tree)
    if problem is not None:
        rel = "eval/problem/problem.json"
        files.append((rel, tree / rel))
        defn = problem.get("definition") or {}
        ins = list((defn.get("inputs") or {}).keys())
        outs = list((defn.get("outputs") or {}).keys())
        if not ins:
            raise Refused(f"{tree}: contract declares no inputs")
        reference = problem.get("reference_impl") or {}
        ref_src = defn.get("reference") if reference.get("kind", "definition") == "definition" \
            else None
        dps = problem.get("dps")
        if dps is None:
            dps = bool(problem.get("destination_passing_style", False))
        return Contract(defn.get("name"), problem.get("entry", "run"), ins, outs,
                        dps, problem.get("kernel_file"),
                        ref_src, problem.get("tolerance"), files)
    window = contract_window_toml(tree)
    if window is not None:
        files.append(("window.toml", tree / "window.toml"))
        task = _task_toml(tree)
        return Contract(window["definition"] or (task or {}).get("name"),
                        window["entry"], window["inputs"], window["outputs"],
                        window["dps"], None, None, {}, files)
    ins = [s.strip() for s in (args.inputs or "").split(",") if s.strip()]
    outs = [s.strip() for s in (args.outputs or "").split(",") if s.strip()]
    if not ins:
        raise Refused(f"{tree}: no contract found (no eval/problem/problem.json, no "
                      f"window.toml with [student] inputs) -- pass --inputs/--outputs")
    return Contract(args.definition, args.entry or "run", ins, outs, None,
                    None, None, {}, files)


def _task_toml(tree: Path) -> dict | None:
    path = Path(tree) / "task.toml"
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("task") or {}


# --- language -------------------------------------------------------------------


def infer_language(src: str, kfile: str, tree: Path | None, override: str | None) -> str:
    """cutedsl / triton / pytorch / cuda-sol: the file suffix, then the declared
    loop/task language, then the kernel's imports (the pre-WP24 export rule)."""
    if override:
        if override not in EXPORT_LANGS:
            raise Refused(f"--language must be one of {sorted(EXPORT_LANGS)}, got {override!r}")
        return override
    if is_cuda_kernel_file(kfile):
        return "cuda-sol"
    declared = None
    if tree is not None:
        task = _task_toml(tree)
        declared = (task or {}).get("language")
        if declared is None:
            declared = _loop_toml_language(tree)
    if declared in ("cutedsl", "triton"):
        return declared
    if declared == "cuda":
        return "cuda-sol"
    try:
        mods = {a.name.split(".")[0] for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.ImportFrom) and n.module}
    except SyntaxError as e:
        raise Refused(f"{kfile} does not parse: {e}")
    if "cutlass" in mods:
        return "cutedsl"
    if "triton" in mods:
        return "triton"
    return "pytorch"


def _loop_toml_language(tree: Path) -> str | None:
    path = Path(tree) / "loop.toml"
    if not path.is_file():
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("loop", {}).get("language")
    except (OSError, ValueError):
        return None


# --- signature ------------------------------------------------------------------


def python_signature(src: str, entry: str = "run") -> tuple[str, list[str]]:
    """(entry, parameters) of the kernel file's top-level entry function."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise Refused(f"kernel does not parse: {e}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            return entry, [a.arg for a in node.args.posonlyargs + node.args.args]
    raise Refused(f"the kernel defines no top-level {entry}()")


_CPP_FN = r"(?:void|torch::Tensor|at::Tensor)\s+{entry}\s*\(([^)]*)\)"


def cuda_signature(src: str, kfile: str, entry: str = "run") -> tuple[str, list[str]]:
    """(entry, parameters) of a CUDA C++ kernel file's bound entry."""
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", " ", src, flags=re.S)
    if not re.search(r"PYBIND11_MODULE\s*\(\s*TORCH_EXTENSION_NAME\s*,", stripped):
        raise Refused(f"{kfile} has no PYBIND11_MODULE(TORCH_EXTENSION_NAME, m): "
                      f"the build names the module")
    if not re.search(rf'\.def\s*\(\s*"{re.escape(entry)}"', stripped):
        raise Refused(f'{kfile} binds no "{entry}" (m.def("{entry}", &{entry}))')
    m = re.search(_CPP_FN.format(entry=re.escape(entry)), stripped)
    if m is None:
        raise Refused(f"{kfile} defines no `void {entry}(...)` "
                      f"(or `torch::Tensor {entry}(...)`)")
    params = [re.split(r"[\s&*]+", p.strip())[-1] for p in m.group(1).split(",") if p.strip()]
    return entry, params


def resolve_dps(params: list[str], inputs: list[str], outputs: list[str],
                declared: bool | None) -> tuple[bool, list[str]]:
    """The calling convention the signature actually implements, checked against the
    contract: inputs only = value-returning, inputs then outputs = DPS. A signature
    that matches neither form is refused (the pre-WP24 exporter refused a drifted
    signature; so does this one)."""
    if len(params) == len(inputs) and params == inputs:
        want = inputs
        dps = False
    elif len(params) == len(inputs) + len(outputs) \
            and params == inputs + outputs and outputs:
        want = inputs + outputs
        dps = True
    else:
        outs = (", ".join(outputs)) if outputs else "<output>"
        raise Refused(
            f"the kernel signature run({', '.join(params)}) does not match the contract: "
            f"the definition wants run({', '.join(inputs)}) (value-returning) or "
            f"run({', '.join(inputs + outputs)}) (destination-passing; outputs: {outs})")
    if declared is not None and bool(declared) != dps:
        raise Refused(f"the contract declares destination_passing_style={bool(declared)} "
                      f"but the signature implements {dps}")
    return dps, want


def cuda_kernel_flags(src: str) -> list[str]:
    """The nvcc flags the card builds this file with: the defaults, the file's
    fk-cuda-cflags line, the arch (mirrors card_template/eval/benchkit.py)."""
    extra = [f for line in CUDA_FLAGS_LINE.findall(src) for f in line.split()]
    if bad := [f for f in extra if not CUDA_FLAG_OK.fullmatch(f)]:
        raise Refused(f"fk-cuda-cflags: {bad} not allowed (code generation / diagnostics "
                      f"flags only: no paths, libraries or linker options)")
    flags = CUDA_DEFAULT_CFLAGS + [f for f in extra if f not in CUDA_DEFAULT_CFLAGS]
    if not any(f.startswith(("-gencode", "-arch")) for f in flags):
        flags.append(CUDA_GENCODE)
    return flags


# --- submission document (mined format) -----------------------------------------


def build_solution(target: str, *, name: str, definition: str, author: str,
                   description: str, language: str, entry: str, kfile: str,
                   src: str, dps: bool, hardware: str) -> dict:
    """The suite's solution JSON for this kernel (the pre-WP24 export format)."""
    if language not in EXPORT_LANGS:
        raise Refused(f"language must be one of {sorted(EXPORT_LANGS)}, got {language!r}")
    fib_lang, sol_langs, sol_deps = EXPORT_LANGS[language]
    entry_point = f"{Path(kfile).name}::{entry}"
    sources = [{"path": Path(kfile).name, "content": src}]
    if target == TARGET_FIB:
        spec = {"language": fib_lang, "target_hardware": [f"NVIDIA {hardware}"],
                "entry_point": entry_point, "dependencies": [],
                "destination_passing_style": dps}
    elif target == TARGET_SOL:
        spec = {"languages": sol_langs, "target_hardware": [hardware],
                "entry_point": entry_point, "destination_passing_style": dps,
                "dependencies": sol_deps}
        if language == "cuda-sol":  # the flags the card built it with
            spec["compile_options"] = {"cuda_cflags": cuda_kernel_flags(src)}
    else:
        raise Refused(f"--target must be one of {', '.join(TARGETS)} (or 'both')")
    return {"name": name, "definition": definition, "author": author,
            "description": description, "spec": spec, "sources": sources}


def solution_filename(targets: list[str]) -> dict[str, str]:
    """One ``solution.json`` per suite: plain when a single target is requested,
    suffixed when both are packaged together."""
    if len(targets) == 1:
        return {targets[0]: "solution.json"}
    return {t: f"solution-{t}.json" for t in targets}


# --- preflight ------------------------------------------------------------------


class Report:
    def __init__(self):
        self.stages: list[dict] = []

    def add(self, stage: str, status: str, detail: str) -> None:
        self.stages.append({"stage": stage, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(s["status"] == "FAIL" for s in self.stages)

    def dict(self) -> dict:
        return {"schema": 1, "lanes_note": LANE_NOTE, "stages": self.stages}


def preflight_lint(src: bytes, kfile: str, contract: Contract, language: str,
                   solutions: dict[str, dict]) -> tuple[bool, list[str], str]:
    """Stage (a): manifest/lint static checks. Any failure refuses the export.
    Returns (dps, the contract's expected parameters, a report detail line)."""
    text = src.decode("utf-8")  # a UnicodeDecodeError is a refusal
    if not text.strip():
        raise Refused(f"{kfile} is empty")
    entry = contract.entry
    inputs, outputs = contract.inputs, contract.outputs
    if is_cuda_kernel_file(kfile):
        got_entry, params = cuda_signature(text, kfile, entry)
        flags = cuda_kernel_flags(text)  # allowlist violations refuse here
    else:
        got_entry, params = python_signature(text, entry)
        flags = None
        compile(text, kfile, "exec")  # a SyntaxError refuses here
    dps, want = resolve_dps(params, inputs, outputs, contract.dps)
    if got_entry != entry:
        raise Refused(f"{kfile} defines {got_entry}() but the contract declares "
                      f"entry {entry!r}")
    for target, sol in solutions.items():
        for key in ("name", "definition", "author", "description", "spec", "sources"):
            if key not in sol:
                raise Refused(f"internal: solution for {target} lacks {key!r}")
        if not sol["name"] or os.sep in sol["name"] or "/" in sol["name"]:
            raise Refused(f"solution name {sol['name']!r} is not a usable name")
        if not sol["definition"]:
            raise Refused("the submission names no definition (pass --definition with "
                          "the suite dataset's definition name)")
        if not sol["sources"]:
            raise Refused("internal: empty sources")
    detail = (f"entry {entry}({', '.join(want)}) ok; destination_passing_style={dps}; "
              f"language={language}; contract files: "
              f"{', '.join(rel for rel, _ in contract.contract_files) or 'command line'}")
    if flags is not None:
        detail += f"; nvcc flags: {' '.join(flags)}"
    return dps, want, detail


def preflight_compile(src: bytes, kfile: str, work: Path, report: Report) -> None:
    """Stage (b): compile check, only when this machine has the toolchain. ``work``
    is a scratch directory (never the package)."""
    text = src.decode("utf-8")
    if is_cuda_kernel_file(kfile):
        if shutil.which("nvcc") is None:
            report.add("compile", "SKIP", "no nvcc on PATH (the target machine builds it)")
            return
        try:
            import torch
            import torch.utils.cpp_extension as ext
        except ImportError:
            report.add("compile", "SKIP", "nvcc present but torch is not importable "
                       "here (the card's build needs torch headers)")
            return
        if not getattr(torch, "version", None) or not torch.version.cuda:
            report.add("compile", "SKIP", "this torch is a CPU build; the card's "
                       "CUDA build happens on the target machine")
            return
        staged = work / kfile
        staged.write_text(text, encoding="utf-8")
        cutlass = os.environ.get("CUTLASS_DIR", "/usr/local/cutlass")
        try:
            ext.load(name=f"pk_eval_export_check_{sha256_bytes(src)[:12]}",
                     sources=[str(staged)], extra_cuda_cflags=cuda_kernel_flags(text),
                     extra_ldflags=list(CUDA_DEFAULT_LDFLAGS),
                     extra_include_paths=[f"{cutlass}/include",
                                          f"{cutlass}/tools/util/include"],
                     build_directory=str(work), verbose=False, is_python_module=True)
        except Exception as e:  # noqa: BLE001 — classified below
            msg = f"{type(e).__name__}: {e}"
            # a broken local toolchain is a SKIP, a compiler verdict is a FAIL
            env_gap = ("CUDA_HOME" in str(e) or "ninja" in str(e).lower()
                       or "not find" in str(e) or "No such file" in str(e))
            report.add("compile", "SKIP" if env_gap else "FAIL",
                       (f"the local toolchain is not usable here ({msg})" if env_gap
                        else f"the compiler rejected {kfile}: {msg}"))
            return
        report.add("compile", "PASS", f"nvcc + torch cpp_extension built {kfile} "
                   f"with the exported flags")
        return
    # Python languages: the syntax check already ran in stage (a); here the module
    # is imported in a bounded subprocess when its imports resolve on this machine.
    # Import behavior is environment-dependent (the real deps live on the target
    # machine, and a kernel may JIT-build at import), so this stage is advisory:
    # PASS / SKIP / WARN, never FAIL -- the refusing gate for Python is stage (a).
    probe = ("import importlib.util, json, sys, traceback\n"
             "try:\n"
             "    spec = importlib.util.spec_from_file_location('_export_probe', sys.argv[1])\n"
             "    mod = importlib.util.module_from_spec(spec)\n"
             "    spec.loader.exec_module(mod)\n"
             "except BaseException as e:\n"
             "    print(json.dumps({'ok': False, 'kind': type(e).__name__,\n"
             "                          'err': traceback.format_exc(limit=4)}))\n"
             "    sys.exit(0)\n"
             "print(json.dumps({'ok': True, 'kind': '', 'err': ''}))\n")
    try:
        r = subprocess.run([sys.executable, "-I", "-c", probe, str(work / kfile)],
                           capture_output=True, text=True,
                           timeout=int(os.environ.get("PK_EVAL_EXPORT_IMPORT_TIMEOUT",
                                                      "300")), check=False)
    except subprocess.TimeoutExpired:
        report.add("compile", "SKIP", f"importing {kfile} did not settle within the "
                   f"time budget (module-level JIT build?); the target machine decides")
        return
    if r.returncode != 0 or not r.stdout.strip():
        report.add("compile", "WARN", f"the import probe itself failed "
                   f"(rc={r.returncode}): {r.stderr.strip()[:200]}")
        return
    verdict = json.loads(r.stdout.strip().splitlines()[-1])
    if verdict["ok"]:
        report.add("compile", "PASS", f"{kfile} imports and defines run()")
    elif verdict["kind"] in ("ImportError", "ModuleNotFoundError"):
        report.add("compile", "SKIP", f"module imports resolve only on the target "
                   f"machine ({verdict['kind']}: {verdict['err'].strip().splitlines()[-1][:160] if verdict['err'] else ''})")
    else:
        report.add("compile", "WARN", f"the module raised {verdict['kind']} on import "
                   f"here (environment-dependent; the run's gate judged the real "
                   f"environment): {verdict['err'].strip().splitlines()[-1][:160] if verdict['err'] else ''}")


def resolve_shape_expression(expr, env: dict) -> int:
    """An ``expr`` shape term: an axis name, or arithmetic over axis names."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise Refused(f"unknown axis {node.id!r} in shape expression {expr!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub,
                                                               ast.Mult, ast.FloorDiv)):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left // right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise Refused(f"unsupported shape expression {expr!r}")
    value = ev(ast.parse(str(expr), mode="eval").body)
    if value <= 0:
        raise Refused(f"shape expression {expr!r} resolved to {value}")
    return value


def _smoke_tensor(spec: dict, shapes_env: dict, torch, gen) -> "torch.Tensor":
    """One contract input, deterministically drawn (CPU, seed fixed by the caller)."""
    shape = [resolve_shape_expression(e, shapes_env) for e in spec.get("shape", [])]
    dtype = str(spec.get("dtype", "float32"))
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise Refused(f"input dtype {dtype!r} not available in this torch")
    if dtype in DTYPE_EPS or dtype.startswith("float"):
        return torch.randn(*shape, dtype=torch_dtype, generator=gen)
    if dtype in INT_DTYPE_RANGES:
        low, high = INT_DTYPE_RANGES[dtype]
        return torch.randint(low, high, shape, dtype=torch_dtype, generator=gen)
    if dtype == "bool":
        return (torch.rand(shape, generator=gen) < 0.5).to(torch_dtype)
    raise Refused(f"input dtype {dtype!r} has no deterministic smoke generator")


def preflight_correctness(src: bytes, kfile: str, contract: Contract, dps: bool,
                          work: Path, report: Report) -> None:
    """Stage (c): correctness on the task contract's inputs. Needs torch + a CUDA
    device + a runnable reference in the contract; anything less is SKIP+WARN --
    this stage never replaces the run's own full gate."""
    try:
        import torch
    except ImportError:
        report.add("correctness", "SKIP", "torch is not importable here")
        return
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        report.add("correctness", "SKIP", "no CUDA device on this machine "
                   "(the run's own gate remains the correctness authority)")
        return
    reference_src = contract.reference_src
    if not reference_src:
        report.add("correctness", "SKIP", "the contract carries no runnable reference "
                   "(sealed / binary reference) -- the run's gate already judged it")
        return
    problem = None
    for rel, path in contract.contract_files:
        if rel.endswith("problem.json"):
            problem = json.loads(path.read_text(encoding="utf-8"))
    defn = (problem or {}).get("definition") or {}
    workloads = [(problem or {}).get("workload") or {}]
    workloads += [g.get("workload", {}) for g in (problem or {}).get("group", [])]
    import importlib.util
    try:
        ref_ns: dict = {}
        exec(compile(reference_src, "<contract-reference>", "exec"), ref_ns)  # noqa: S102
        ref_fn = ref_ns[contract.entry]
        modname = f"pk_eval_export_probe_{sha256_bytes(src)[:12]}"
        spec = importlib.util.spec_from_file_location(modname, work / kfile)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kern_fn = getattr(mod, contract.entry)
    except Exception as e:  # noqa: BLE001 — environment gap, not a verdict
        report.add("correctness", "WARN", f"could not load the kernel or the reference "
                   f"here ({type(e).__name__}: {e})")
        return
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)
    checked = 0
    try:
        for wl in workloads:
            axes_env = dict(wl.get("axes") or {})
            try:
                inputs = [_smoke_tensor(s, axes_env, torch, gen)
                          for s in (defn.get("inputs") or {}).values()]
            except (Refused, RuntimeError) as e:
                report.add("correctness", "SKIP", f"contract inputs not drawable here ({e})")
                return
            # from here through the comparison, a machine that cannot run the
            # reference or the kernel (fp8 reference off sm90, an OOM, a kernel
            # that will not launch here) is a SKIP-stage WARN like the load
            # guard above -- never a traceback out of the tool
            inputs = [t.to(device) for t in inputs]
            with torch.no_grad():
                refs = ref_fn(*[t.clone() for t in inputs])
            if not isinstance(refs, (tuple, list)):
                refs = (refs,)
            if dps:
                outs = [torch.empty_like(r) for r in refs]
                with torch.no_grad():
                    kern_fn(*inputs, *outs)
            else:
                with torch.no_grad():
                    outs = kern_fn(*inputs)
                if not isinstance(outs, (tuple, list)):
                    outs = (outs,)
            names = list((defn.get("outputs") or {}).keys()) \
                or [f"output_{i}" for i in range(len(outs))]
            tol_map = ((contract.tolerance or {}).get("outputs") or {})
            for name, got, want in zip(names, outs, refs):
                got_c, want_c = got.float(), want.float()
                denom = want_c.abs().max().clamp(min=1e-30)
                err = (got_c - want_c).abs().max().item() / denom.item()
                dtype_name = str(want.dtype).replace("torch.", "")
                tol = float(tol_map.get(name, 2 * DTYPE_EPS.get(dtype_name, 1e-2)))
                if got.shape != want.shape or err > tol:
                    report.add("correctness", "FAIL",
                               f"{name}: max rel err {err:.3e} > tol {tol:.3e} "
                               f"(shape got {tuple(got.shape)} want {tuple(want.shape)})")
                    return
            checked += 1
    except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 -- environment gap, not a verdict
        report.add("correctness", "WARN", f"could not run the reference or the kernel "
                   f"here ({type(e).__name__}: {e})")
        return
    report.add("correctness", "PASS",
               f"{checked} contract workload(s) pass the contract's tolerance on this GPU")


# --- package --------------------------------------------------------------------


def package_readme(targets: list[str], solutions: dict, hardware: str) -> str:
    # full suite names live in docs/external-eval.md (the code surface keeps them out)
    names = {"fib": "FlashInfer", "sol": "the NVIDIA SOL suite"}
    lines = [
        "# External evaluation submission package",
        "",
        "Generated by Peregrine-Kernel's `scripts/eval-export/pack.py`: the",
        "manual, opt-in evaluation export lane (zero network I/O; uploading",
        "is a human step).",
        "",
        f"Target suite(s): {', '.join(names[t] for t in targets)}. "
        f"Target hardware: {hardware}.",
        "",
        "Contents:",
        "",
    ]
    for target, fname in solution_filename(targets).items():
        lines.append(f"- `{fname}` — the {names[target]} submission document "
                     f"(definition `{solutions[target]['definition']}`, name "
                     f"`{solutions[target]['name']}`); submit it through that suite's "
                     f"own submission channel, by hand.")
    lines += [
        "- the kernel file — the exported product, byte-identical to the gated artifact;",
        "- `preflight.json` — the three-stage preflight report recorded at export time;",
        "- `artifact_sha256.json` — the hash manifest anchoring every file in this",
        "  package plus the kernel's provenance (which run, which iteration, which",
        "  reading). The exported bytes are the gated product verbatim: no",
        "  post-processing happened between the gate and this package.",
        "",
        "Protocol, judgment criteria and the honesty red lines:",
        "see `docs/external-eval.md` in the repository.",
        "",
        "<!-- SPDX-FileCopyrightText: Copyright 2026 The Peregrine-Kernel Authors -->",
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "",
    ]
    return "\n".join(lines)


def write_package(out_dir: Path, targets: list[str], solutions: dict[str, dict],
                  src: bytes, kfile: str, language: str, report: Report,
                  provenance: dict, contract: Contract, hardware: str) -> dict:
    """Write the package: solution JSON(s), the kernel verbatim, the preflight
    report, then the hash manifest (written last; excludes itself)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for target, fname in solution_filename(targets).items():
        (out_dir / fname).write_text(
            json.dumps(solutions[target], indent=2) + "\n", encoding="utf-8")
        files[fname] = sha256_file(out_dir / fname)
    (out_dir / kfile).write_bytes(src)
    files[kfile] = sha256_file(out_dir / kfile)
    (out_dir / "preflight.json").write_text(
        json.dumps(report.dict(), indent=2) + "\n", encoding="utf-8")
    files["preflight.json"] = sha256_file(out_dir / "preflight.json")
    (out_dir / "README.md").write_text(
        package_readme(targets, solutions, hardware), encoding="utf-8")
    files["README.md"] = sha256_file(out_dir / "README.md")
    manifest = {
        "schema": 1,
        "created_utc": utc_now(),
        "tool": {"name": "scripts/eval-export/pack.py", "version": TOOL_VERSION,
                 "lane": LANE_NOTE},
        "targets": targets,
        "product": provenance | {"kernel_file": kfile,
                                 "kernel_sha256": files[kfile],
                                 "kernel_bytes": len(src),
                                 "exported_verbatim": True},
        "contract": {
            "definition": solutions[targets[0]]["definition"],
            "entry": contract.entry,
            "inputs": contract.inputs,
            "outputs": contract.outputs,
            "destination_passing_style": solutions[targets[0]]["spec"][
                "destination_passing_style"],
            "language": language,
            "files": [{"path": rel, "sha256": sha256_file(path)}
                      for rel, path in contract.contract_files],
        },
        "files": files,
    }
    (out_dir / "artifact_sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --- main -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="eval-export",
        description="Package a gated kernel into an external evaluation suite's "
                    "submission format (manual, offline; protocol in "
                    "docs/external-eval.md)")
    p.add_argument("out", help="package output directory (must not exist)")
    p.add_argument("--target", default="both", choices=[*TARGETS, "both"],
                   help="which suite's format to emit (default: both)")
    product = p.add_mutually_exclusive_group(required=True)
    product.add_argument("--run-dir", help="a run directory; auto-locate its best KEEP")
    product.add_argument("--tree", help="an operator tree (kernel + contract files)")
    product.add_argument("--kernel", help="an explicit kernel file")
    p.add_argument("--op", help="operator name under --run-dir's operators/")
    p.add_argument("--definition", default=None,
                   help="the suite dataset's definition name (submission metadata)")
    p.add_argument("--name", default=None, help="solution name")
    p.add_argument("--author", default="peregrine-kernel", help="solution author")
    p.add_argument("--description", default=None, help="solution description")
    p.add_argument("--hardware", default="B200", help="target hardware (default B200)")
    p.add_argument("--language", default=None, choices=sorted(EXPORT_LANGS),
                   help="override the inferred kernel language")
    p.add_argument("--entry", default=None, help="entry name when the tree carries no "
                   "contract (default run)")
    p.add_argument("--inputs", default=None,
                   help="comma-separated input names when the tree carries no contract")
    p.add_argument("--outputs", default=None,
                   help="comma-separated output names when the tree carries no contract")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"[eval-export] refused: {out_dir} already exists", file=sys.stderr)
        return 2
    report = Report()
    try:
        # 1. locate the product
        if args.kernel:
            kpath = Path(args.kernel)
            if not kpath.is_file():
                raise Refused(f"{kpath}: not a file")
            src = kpath.read_bytes()
            kfile = kpath.name
            tree = kpath.parent
            provenance = {"kind": "explicit-kernel-file", "kernel": str(kpath.resolve()),
                          "bytes_source": str(kpath), "sha_found": None,
                          "warning": "explicit kernel export: no run provenance -- "
                                     "the hash anchors the bytes; the gate claim is "
                                     "the operator's"}
        elif args.tree:
            tree = Path(args.tree)
            src, kfile, provenance = kernel_from_tree(tree)
        else:
            if not (args.run_dir and args.op):
                p.error("--run-dir needs --op")
            tree = Path(args.run_dir) / "operators" / args.op
            src, kfile, provenance = kernel_from_run(Path(args.run_dir), args.op)
        report.add("locate", "PASS", json.dumps(provenance, default=str))

        # 2. contract, language, signature, submission documents
        contract = read_contract(tree, args)
        if args.definition:
            contract.definition = args.definition
        if args.entry:
            contract.entry = args.entry
        definition = contract.definition or provenance.get("op") or Path(kfile).stem
        text = src.decode("utf-8")
        language = infer_language(text, kfile, tree if tree.is_dir() else None,
                                  args.language)
        if is_cuda_kernel_file(kfile):
            _, params = cuda_signature(text, kfile, contract.entry)
        else:
            _, params = python_signature(text, contract.entry)
        dps, _ = resolve_dps(params, contract.inputs, contract.outputs, contract.dps)
        name = args.name or f"{definition.split('/')[-1]}_peregrine"
        description = args.description or (
            f"Peregrine-Kernel {language} kernel for {definition}")
        targets = list(TARGETS) if args.target == "both" else [args.target]
        solutions = {t: build_solution(
            t, name=name, definition=definition, author=args.author,
            description=description, language=language, entry=contract.entry,
            kfile=kfile, src=text, dps=dps, hardware=args.hardware) for t in targets}

        # 3. preflight (a): lint / manifest static checks -- a failure refuses
        _, _, lint_detail = preflight_lint(src, kfile, contract, language, solutions)
        report.add("lint", "PASS", lint_detail)
    except (Refused, SyntaxError, UnicodeDecodeError) as e:
        print(f"[eval-export] refused: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    # 4. preflight (b) + (c) in a scratch dir: advisory, FAIL refuses (no package),
    #    SKIP proceeds with a warning in the report
    import tempfile
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory(prefix="pk-eval-export-") as tmp:
            work = Path(tmp)
            (work / kfile).write_bytes(src)
            preflight_compile(src, kfile, work, report)
            preflight_correctness(src, kfile, contract, dps, work, report)
    finally:
        sys.dont_write_bytecode = bytecode
    if report.failed:
        for stage in report.stages:
            if stage["status"] == "FAIL":
                print(f"[eval-export] preflight {stage['stage']}: FAIL -- "
                      f"{stage['detail']}", file=sys.stderr)
        print("[eval-export] refused: the preflight failed; nothing was written",
              file=sys.stderr)
        return 2

    manifest = write_package(out_dir, targets, solutions, src, kfile, language,
                             report, provenance, contract, args.hardware)
    print(f"[eval-export] wrote {out_dir} "
          f"({len(targets)} suite format(s): {', '.join(targets)}; kernel {kfile} "
          f"{len(src)} bytes, sha256 {manifest['product']['kernel_sha256'][:12]})")
    for stage in report.stages:
        print(f"[eval-export]   {stage['stage']:12s} {stage['status']}")
    print("[eval-export] submit by hand; the protocol is docs/external-eval.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
