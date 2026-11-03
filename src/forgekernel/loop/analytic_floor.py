"""The analytic floor: what the problem definition alone says no kernel can beat.

``floor_opt`` used to rest entirely on the teacher's hand estimate — one hard
formula (``t_mma = FLOPs ÷ peak tensor throughput``) and judgement for the memory
side, so the INFEASIBLE gate (``floor_opt > target × 1.20``) and WP5's headroom
stop both stood on a number nobody had anchored. This module is the machine
denominator: bytes and FLOPs computed from the task's tensor shapes and dtypes
alone — quantities independent of any implementation — and a time floor from the
architecture's published peak rates.

Two guarantees the consumers rely on:

  **Provable, not clever.** Every formula is a LOWER bound on the work any
  correct implementation must do: each operand byte counted once (tile re-reads
  only add traffic), each scalar MAC counted once, conservative choices
  everywhere an operator allows more than one work shape (causal attention is
  charged the masked half even when the card does not say causal — a non-causal
  kernel then has its projections and softmax uncharged, a causal one matches).
  A family whose floor cannot be stated without guessing returns ``flops=None``
  and falls back to the pure bandwidth roofline — an absent floor is recoverable,
  a wrong one poisons every gate that cites it.

  **Advisory only.** Nothing here enters a machine gate: the INFEASIBLE rule,
  the precision gate and the timing protocol are untouched. The floor renders
  into the TEACHER's NCU ground-truth section (never the student digest), and
  the opt-in ``[target] floor_margin`` / ``[loop] floor_stop`` run beside the
  existing stop logic, never instead of it.

Peak rates are the vendors' dense, no-sparsity datasheet numbers (paper peaks):
a real kernel cannot exceed them, which is what makes ``floor_ms`` provable.
The band's upper end applies documented achievable fractions (STREAM-class
bandwidth, cuBLAS-class MFU) — inside the band's upper end is "at the floor",
not "impossible".
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

# --- peak rates -----------------------------------------------------------------
#
# Per-architecture published peaks: (bytes/s, FLOP/s per dtype class). Dense
# (no-sparsity) values, the paper-peak convention. Sources, per row:
#   sm90  H100 SXM — NVIDIA H100 datasheet: HBM3 3.35 TB/s; FP16/BF16 tensor
#         core 989.4 TFLOPS dense (1979 with sparsity); FP8 1978.9 dense.
#   sm100 B200 — NVIDIA Blackwell datasheet: HBM3e 8 TB/s; FP16/BF16 2250
#         TFLOPS dense (4500 sparse); FP8 4500 dense; FP4 9000 dense.
#   sm80  A100 SXM — NVIDIA A100 datasheet: HBM2e 2039 GB/s; FP16 312 TFLOPS
#         dense (624 sparse).
#   consumer RTX 3060 (GA106) — GeForce Ampere figures: GDDR6 360 GB/s; FP16
#         tensor 25.3 TFLOPS dense (FP16 accumulate). Direction-only: this is
#         the dev box, not a target; the published-kernel backtest never uses it.
_PEAKS: dict[str, dict[str, Any]] = {
    "sm80": {"bw": 2039e9,
             "flops": {"fp64": 9.7e12, "fp32": 19.5e12, "tf32": 156e12,
                       "fp16": 312e12, "fp8": 624e12}},
    "sm90": {"bw": 3350e9,
             "flops": {"fp64": 34e12, "fp32": 67e12, "tf32": 494.7e12,
                       "fp16": 989.4e12, "fp8": 1978.9e12}},
    "sm100": {"bw": 8000e9,
              "flops": {"fp64": 40e12, "fp32": 80e12, "tf32": 1120e12,
                        "fp16": 2250e12, "fp8": 4500e12, "fp4": 9000e12}},
    "consumer": {"bw": 360e9,
                 "flops": {"fp32": 12.7e12, "fp16": 25.3e12}},
}
# Achievable fractions the band's upper end uses: what STREAM / cuBLAS reach on
# the same silicon (measured-class efficiency, not a guarantee — a kernel at the
# upper end is at the floor, one below the lower end is impossible).
_ACHIEVABLE = {"bw": 0.80, "flops": 0.80}

# Bytes per element by dtype name (the dataset suites' spellings included).
_DTYPE_BYTES: dict[str, float] = {
    "float64": 8, "fp64": 8, "double": 8, "f64": 8,
    "float32": 4, "fp32": 4, "float": 4, "f32": 4,
    "tf32": 4,
    "float16": 2, "fp16": 2, "half": 2, "f16": 2,
    "bfloat16": 2, "bf16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "fp8e4m3": 1, "fp8e5m2": 1,
    "float8": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
    "int8": 1, "s8": 1, "u8": 1, "uint8": 1, "byte": 1, "bool": 1,
    "int16": 2, "s16": 2, "uint16": 2, "fp6": 0.75,
    "float4_e2m1fn_x2": 0.5, "fp4": 0.5, "int4": 0.5,
    "int32": 4, "s32": 4, "uint32": 4, "i32": 4,
    "int64": 8, "s64": 8, "uint64": 8, "i64": 8, "long": 8,
}

# The flops class a dtype's MACs price at (peak table keys).
_DTYPE_CLASS: dict[str, str] = {
    "float64": "fp64", "fp64": "fp64", "double": "fp64", "f64": "fp64",
    "float32": "fp32", "fp32": "fp32", "float": "fp32", "f32": "fp32",
    "tf32": "tf32",
    "float16": "fp16", "fp16": "fp16", "half": "fp16", "f16": "fp16",
    "bfloat16": "fp16", "bf16": "fp16",
    "float8_e4m3fn": "fp8", "float8_e5m2": "fp8", "fp8e4m3": "fp8",
    "fp8e5m2": "fp8", "float8": "fp8", "fp8": "fp8", "e4m3": "fp8", "e5m2": "fp8",
    "float4_e2m1fn_x2": "fp4", "fp4": "fp4",
}

_AXIS_ALIASES: dict[str, tuple[str, ...]] = {
    "batch": ("batch", "batch_size", "bz", "b", "n"),
    "heads": ("head", "heads", "num_heads", "head_count", "hq", "h"),
    "seq": ("seq", "seq_len", "seqlen", "sequence_length", "sl", "s", "ctx", "context"),
    "dim": ("dim", "head_dim", "headdim", "d", "dk", "hd", "d_model", "hidden"),
}

_SHAPE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def peak_table() -> dict[str, dict[str, Any]]:
    """The peak-rate table (a copy; tests and the backtest read it)."""
    return {arch: {**row, "flops": dict(row["flops"])} for arch, row in _PEAKS.items()}


# --- shape resolution ------------------------------------------------------------

class _AxesEnv(dict):
    """Name lookup that records what it could not resolve."""


class _ShapeResolver(ast.NodeVisitor):
    """Safe evaluation of a shape expression: integer literals, axis names and
    + - * / // only (the shapes-grammar subset; see NOTICE)."""

    def __init__(self, axes: dict):
        self.axes = axes
        self.unresolved = False

    def eval(self, expr: object) -> float | None:
        try:
            tree = ast.parse(str(expr), mode="eval")
        except SyntaxError:
            self.unresolved = True
            return None
        return self.visit(tree.body)

    def visit_Name(self, node) -> float | None:
        v = self.axes.get(node.id)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            self.unresolved = True
            return None
        return float(v)

    def visit_Constant(self, node) -> float | None:
        return node.value if isinstance(node.value, (int, float)) else None

    def visit_BinOp(self, node) -> float | None:
        l, r = self.visit(node.left), self.visit(node.right)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r if r else None
        if isinstance(node.op, ast.FloorDiv):
            return l // r if r else None
        self.unresolved = True
        return None

    def generic_visit(self, node) -> float | None:
        self.unresolved = True
        return None


def resolve_dim(spec: object, axes: dict, resolver: _ShapeResolver) -> int | None:
    """One shape entry: an int, an axis name, or an arithmetic expression over axes."""
    if isinstance(spec, bool) or not isinstance(spec, (int, float, str)):
        resolver.unresolved = True
        return None
    if isinstance(spec, (int, float)):
        return int(spec)
    v = resolver.eval(spec)
    return int(v) if v is not None and v == int(v) and v > 0 else None


# --- tensor bytes ----------------------------------------------------------------

def _tensor_bytes(t: dict, axes: dict, assumptions: list[str]) -> int:
    """One tensor's bytes; 0 with an assumption note when a dim/dtype is unreadable
    (skipped terms only shrink the sum, so the total stays a lower bound)."""
    shape = t.get("shape")
    if shape is None:
        assumptions.append(f"{t.get('name', '?')}: scalar (no shape) — 0 bytes charged")
        return 0
    resolver = _ShapeResolver(axes)
    dims = [resolve_dim(s, axes, resolver) for s in (shape if isinstance(shape, (list, tuple)) else [shape])]
    if resolver.unresolved or any(d is None or d <= 0 for d in dims):
        assumptions.append(f"{t.get('name', '?')}: shape {shape!r} not resolvable from "
                           f"axes — charged 0 bytes (variable/sparse shape)")
        return 0
    n = 1
    for d in dims:
        n *= d
    dtype = str(t.get("dtype", "")).strip().lower()
    size = _DTYPE_BYTES.get(dtype)
    if size is None:
        assumptions.append(f"{t.get('name', '?')}: dtype {dtype!r} unknown — charged 0 bytes")
        return 0
    return int(math.ceil(n * size))


def tensors_bytes(tensors: list[dict], axes: dict, assumptions: list[str]) -> int:
    """Σ(operand bytes): every input and output counted ONCE — the provable traffic
    lower bound (tile re-reads, staging and sparsity bookkeeping only add)."""
    return sum(_tensor_bytes(t, axes, assumptions) for t in tensors)


# --- family floors ---------------------------------------------------------------

def _axis(axes: dict, kind: str) -> int | None:
    for name in _AXIS_ALIASES[kind]:
        v = axes.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return int(v)
    return None


def family_of(problem: dict) -> str | None:
    """The problem's operator family: an explicit ``family`` field wins, else the
    definition's op_type / name. None = no provable flops formula (the bytes
    roofline still applies)."""
    fam = problem.get("family")
    if fam:
        return str(fam)
    text = " ".join(str(problem.get(k, "")) for k in ("op_type", "name")).lower()
    if "gemm" in text or "matmul" in text:
        return "gemm"
    if "attention" in text or "attn" in text:
        return "attention"
    if "conv" in text:
        return "conv"
    if "norm" in text:
        return "norm"
    if any(w in text for w in ("reduce", "sum", "cumsum", "scan", "prefix")):
        return "reduction"
    if any(w in text for w in ("elementwise", "softmax", "activation", "gelu", "silu", "rope")):
        return "elementwise"
    axes = problem.get("axes") or {}
    if {"m", "n", "k"} <= set(axes):
        return "gemm"
    return None


def _gemm_flops(axes: dict, assumptions: list[str]) -> int | None:
    m, n, k = (axes.get(a) for a in ("m", "n", "k"))
    if None in (m, n, k) or min(m, n, k) <= 0:
        assumptions.append("gemm: m/n/k axes not resolvable — flops not charged")
        return None
    return 2 * int(m) * int(n) * int(k)          # every MAC once


def _attention_flops(problem: dict, axes: dict, assumptions: list[str]) -> int | None:
    batch, heads, seq, dim = (_axis(axes, k) for k in ("batch", "heads", "seq", "dim"))
    if None in (batch, heads, seq, dim):
        assumptions.append("attention: batch/heads/seq/dim axes not resolvable — flops not charged")
        return None
    bh = batch * heads
    # Score + AV matmuls, 2 FLOP/MAC. Three states, every one provable:
    #   axes["causal"] explicitly False -> the full table, 4·BH·L²·d (a non-causal
    #     kernel must compute every score entry);
    #   causal (declared truthy, or the problem name says so) -> the masked half,
    #     2·BH·L²·d, exactly what a causal kernel must do;
    #   UNDECLARED -> the half as well: a correct kernel may be causal, so the
    #     full table could sit ABOVE its necessary work — an unprovable floor,
    #     and "a wrong one poisons every gate that cites it". A non-causal kernel
    #     then has softmax and its extra half uncharged: the floor under-counts,
    #     never over-counts. (Any value other than the boolean False — "false",
    #     0, "" — counts as undeclared; only a real False proves non-causality.)
    factor = 4 if axes.get("causal") is False else 2
    return factor * bh * seq * seq * dim


def _conv_flops(axes: dict, assumptions: list[str]) -> int | None:
    """2·N·H_out·W_out·C_out·(C_in·k_h·k_w) — the transposed-convolution MAC count
    (the im2col GEMM's flops); 3-D adds k_d. Any algorithm (direct, FFT, Winograd)
    can only do fewer MACs than the im2col formulation charges."""
    keys = (("n", ("n", "batch")), ("h_out", ("h_out", "out_h", "o_h")),
            ("w_out", ("w_out", "out_w", "o_w")), ("c_out", ("c_out", "out_c", "oc")),
            ("c_in", ("c_in", "in_c", "ic")), ("k_h", ("k_h", "kh", "kernel_h")),
            ("k_w", ("k_w", "kw", "kernel_w")))
    vals = {}
    for key, names in keys:
        for nm in names:
            v = axes.get(nm)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                vals[key] = int(v)
                break
    kd = axes.get("k_d") or axes.get("kd") or 1
    missing = [k for k, _ in keys if k not in vals]
    if missing:
        assumptions.append(f"conv: axes {missing} not resolvable — flops not charged")
        return None
    macs = vals["n"] * vals["h_out"] * vals["w_out"] * vals["c_out"] * vals["c_in"] \
        * vals["k_h"] * vals["k_w"] * int(kd)
    return 2 * macs


def _family_flops(family: str | None, problem: dict,
                  assumptions: list[str]) -> int | None:
    axes = problem.get("axes") or {}
    if family == "gemm":
        return _gemm_flops(axes, assumptions)
    if family == "attention":
        return _attention_flops(problem, axes, assumptions)
    if family == "conv":
        return _conv_flops(axes, assumptions)
    if family == "reduction":
        # Tree lower bound: one add per reduced element (n-1 ≈ n), on the largest
        # input tensor's element count. Only provable when a single input carries
        # the reduced axis; else no charge.
        ins = problem.get("inputs") or []
        if len(ins) == 1:
            resolver = _ShapeResolver(axes)
            dims = [resolve_dim(s, axes, resolver) for s in (ins[0].get("shape") or [])]
            if dims and not resolver.unresolved and all(d and d > 0 for d in dims):
                n = 1
                for d in dims:
                    n *= d
                return n
        assumptions.append("reduction: input element count not resolvable — flops not charged")
        return None
    # norm / elementwise: byte-stream dominated (a handful of FLOPs per element —
    # charging nothing keeps the floor provable).
    return None


def _dominant_class(problem: dict) -> str:
    """The flops class the problem prices at: the dtype class carrying the most
    operand bytes (the tensors the MACs actually read — an fp8 GEMM's fp32 scale
    tensors or an attention bwd's fp32 lse must not move the peak to fp32),
    defaulting to fp16 (the accumulator precision is the kernel's choice, not
    the problem's)."""
    per_class: dict[str, int] = {}
    for t in (problem.get("inputs") or []) + (problem.get("outputs") or []):
        dtype = str(t.get("dtype", "")).strip().lower()
        size = _DTYPE_BYTES.get(dtype)
        if size is None:
            continue
        shape = t.get("shape")
        if shape is None or not isinstance(shape, (list, tuple)):
            continue
        resolver = _ShapeResolver(problem.get("axes") or {})
        dims = [resolve_dim(s, problem.get("axes") or {}, resolver) for s in shape]
        if resolver.unresolved or any(d is None or d <= 0 for d in dims):
            continue
        n = 1
        for d in dims:
            n *= d
        cls = _DTYPE_CLASS.get(dtype, "fp16")
        per_class[cls] = per_class.get(cls, 0) + int(n * size)
    if not per_class:
        return "fp16"
    return max(per_class, key=per_class.get)


def floor_for(problem: dict) -> dict:
    """The analytic floor of one problem instance.

    ``problem``: {"arch": "sm100", "axes": {...},
                  "inputs": [{"name", "shape", "dtype"}...],
                  "outputs": [...], ("family" | "op_type" | "name" for the family)}

    Returns {"bytes_min", "flops", "floor_ms", "floor_ms_band", "family",
             "assumptions"} — ``floor_ms`` is the provable floor (band's lower
    end), ``floor_ms_band`` is (provable, achievable); both None when nothing
    could be charged (unresolvable shapes, unknown arch) or no peaks exist.
    """
    assumptions: list[str] = []
    family = family_of(problem)
    axes = problem.get("axes") or {}
    tensors = (problem.get("inputs") or []) + (problem.get("outputs") or [])
    bytes_min = tensors_bytes(tensors, axes, assumptions)
    flops = _family_flops(family, problem, assumptions)

    arch = str(problem.get("arch", ""))
    row = _PEAKS.get(arch)
    if row is None:
        assumptions.append(f"arch {arch!r}: no peak row — floor_ms not computed")
    cls = _dominant_class(problem)
    bw = float(row["bw"]) if row else 0.0
    peak = row["flops"].get(cls) if row else None
    t_bytes = bytes_min / bw if bytes_min > 0 and bw > 0 else None
    t_flops = flops / peak if flops and peak else None
    if t_bytes is None and t_flops is None:
        floor_ms = band = None
    else:
        # both sides are seconds; the contract unit is ms
        floor_ms = max(t for t in (t_bytes, t_flops) if t is not None) * 1e3
        hi_bw = t_bytes / _ACHIEVABLE["bw"] if t_bytes is not None else 0.0
        hi_fl = t_flops / _ACHIEVABLE["flops"] if t_flops is not None else 0.0
        band = (floor_ms, max(hi_bw, hi_fl) * 1e3)
    return {"bytes_min": bytes_min, "flops": flops, "floor_ms": floor_ms,
            "floor_ms_band": band, "family": family, "assumptions": assumptions}


# --- problem from a card ---------------------------------------------------------

def problem_from_card(problem_json: dict, workload: dict | None = None,
                      arch: str = "") -> dict | None:
    """The floor problem of one card workload: definition inputs/outputs resolved at
    the workload's axes. Returns None when the card carries no definition tensors
    (nothing is provable). The card's ``hw_arch`` (loop.toml kb_arch) supplies the
    arch when the caller has it."""
    definition = problem_json.get("definition") or {}
    wl = workload or problem_json.get("workload") or {}
    axes = wl.get("axes") or {}
    if not axes:
        return None

    def _tensors(spec: dict) -> list[dict]:
        out = []
        for name, t in (spec or {}).items():
            if isinstance(t, dict):
                out.append({"name": name, "shape": t.get("shape"), "dtype": t.get("dtype")})
        return out

    problem: dict = {"arch": arch or str(problem_json.get("arch") or ""),
                     "axes": dict(axes),
                     "inputs": _tensors(definition.get("inputs")),
                     "outputs": _tensors(definition.get("outputs")),
                     "op_type": definition.get("op_type", ""),
                     "name": definition.get("name", "")}
    # scalars ride the axes (a definition input marked scalar has no shape and
    # charges 0 — the axis value itself is what the formulas read)
    return problem


# --- the teacher-facing line -----------------------------------------------------

def floor_line(floor: dict | None) -> str:
    """The NCU ground-truth tail line: ``floor=<x.xx ms> (bytes=<n> flops=<n|n/a>)``.
    A None floor (no provable bytes, unknown arch) renders the reason instead of a
    number — never a fabricated anchor."""
    if not floor:
        return "floor=n/a (no problem definition on file)"
    ms = floor.get("floor_ms")
    if ms is None:
        why = "; ".join(floor.get("assumptions") or []) or "nothing provable"
        return f"floor=n/a ({why[:160]})"
    flops = floor.get("flops")
    return (f"floor={ms:.2e} ms (bytes={floor.get('bytes_min')} "
            f"flops={flops if flops is not None else 'n/a'}, "
            f"at-floor band ≤ {floor['floor_ms_band'][1]:.2e} ms, family={floor.get('family')})")


def mfu_headroom_line(mfu: float | None) -> str:
    """The MFU → headroom line: ``MFU=<x>% (headroom <y>%)`` — the speedup left if
    the best reached 100% MFU (1/mfu − 1), the quantity the teacher's floor_opt
    arithmetic needs. No reading → no line."""
    if not mfu or mfu <= 0 or mfu > 100:
        return ""
    headroom = (100.0 / mfu - 1.0) * 100.0
    return f"MFU={mfu:.1f}% (headroom {headroom:.1f}% — time if MFU reached 100%)"


def floor_from_op_dir(op_dir: Path, arch: str = "") -> dict | None:
    """The floor of an operator card's representative workload, read off
    ``eval/problem/problem.json``. None when the card has no problem file."""
    p = Path(op_dir) / "eval" / "problem" / "problem.json"
    if not p.is_file():
        return None
    import json
    try:
        card = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    problem = problem_from_card(card, arch=arch)
    if problem is None:
        return None
    out = floor_for(problem)
    out["assumptions"] = out.get("assumptions", [])
    return out
