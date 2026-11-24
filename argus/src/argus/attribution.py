"""Causal, counter-grounded performance attribution (ARGUS contribution C2a).

CudaForge hands the Judge a JSON dump of 24 raw counters and asks the LLM to pick one
optimisation.  The counter block carries no *causal* statement: it does not say which
resource moved, by how much, or whether the latency change is even explicable by the
counters.  ARGUS turns the counters into:

* a **stall profile** (where the warps actually wait),
* a **delta attribution** between consecutive candidates -- the resource change that
  accounts for the observed latency change -- with an explicit *consistency* check that
  flags an unexplained speedup (a cheap, first-class reward-hack signal), and
* an ordered **lever list** whose gains are bounded by the analytic model.

``build_judge_block`` renders all of that as a compact, causal prompt block, replacing
CudaForge's raw dump so the Judge reasons about *the next lever and its ceiling* rather
than about a wall of percentages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import roofline as rl

# Warp-stall counters that CudaForge already collects (run_ncu.METRICS).
STALL_KEYS: Dict[str, str] = {
    "long_scoreboard": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "short_scoreboard": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "memory_dependency": "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "barrier": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "branch_resolving": "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
}

#: Map the dominant stall to the mechanism most likely to remove it.
STALL_TO_MECHANISM: Dict[str, str] = {
    "long_scoreboard": "pipeline",
    "short_scoreboard": "pipeline",
    "memory_dependency": "l2_tiling",
    "barrier": "warp_specialize",
    "branch_resolving": "warp_specialize",
}


@dataclass
class StallProfile:
    values: Dict[str, float]
    dominant: Optional[str]
    total: float


def stall_profile(counters: dict) -> StallProfile:
    values = {}
    for name, key in STALL_KEYS.items():
        v = counters.get(key)
        if v is not None:
            values[name] = float(v)
    if not values:
        return StallProfile({}, None, 0.0)
    dominant = max(values, key=values.get)
    return StallProfile(values, dominant, sum(values.values()))


# --------------------------------------------------------------------------- #
# Delta attribution
# --------------------------------------------------------------------------- #

@dataclass
class AttributionReport:
    actual_speedup: float
    factor_speedups: Dict[str, float]      # per-resource best-case speedup
    dominant_factor: Optional[str]
    consistent: bool
    unexplained_speedup: bool
    flags: List[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["factor_speedups"] = {k: round(v, 3) for k, v in self.factor_speedups.items()}
        return d


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return 1.0
    return a / b


def attribute_delta(
    prev_counters: dict,
    new_counters: dict,
    peak: rl.GpuPeak,
    flops: float,
    *,
    compute_dtype: str = "bf16",
    tolerance: float = 0.20,
) -> AttributionReport:
    """Explain the latency change from ``prev`` to ``new`` using the counters.

    For every physical resource the model defines a *best-case* speedup that a change in
    that resource alone could produce.  If the observed speedup exceeds the largest
    resource speedup by more than ``tolerance`` and the counters are essentially
    unchanged, the speedup is **unexplained** -- the signature of a reward hack (skipped
    work, narrowed precision, memoisation) or of a measurement artefact.
    """
    pp = rl.counters_to_point(prev_counters, peak, flops, compute_dtype)
    np_ = rl.counters_to_point(new_counters, peak, flops, compute_dtype)
    actual = _safe_ratio(pp.time_s, np_.time_s)

    factors: Dict[str, float] = {}
    # Fewer DRAM bytes (traffic reduction via reuse/fusion).
    factors["dram_traffic"] = _safe_ratio(pp.dram_bytes, np_.dram_bytes)
    # Higher achieved bandwidth (better memory efficiency).
    factors["achieved_bw"] = _safe_ratio(np_.achieved_dram_bw, pp.achieved_dram_bw)
    # Fewer instructions (algorithmic change / work removal).
    pi = float(prev_counters.get("sm__inst_executed.sum", 0.0) or 0.0)
    ni = float(new_counters.get("sm__inst_executed.sum", 0.0) or 0.0)
    factors["instructions"] = _safe_ratio(pi, ni)
    # Lower occupancy pressure (more work per SM) via registers/smem.
    pr = float(prev_counters.get("launch__registers_per_thread", 0.0) or 0.0)
    nr = float(new_counters.get("launch__registers_per_thread", 0.0) or 0.0)
    factors["occupancy"] = _safe_ratio(nr, pr) if (pr and nr) else 1.0
    # Fewer stalls.
    ps, ns = stall_profile(prev_counters), stall_profile(new_counters)
    factors["stalls"] = _safe_ratio(ps.total, ns.total) if (ps.total and ns.total) else 1.0

    dominant = max(factors, key=factors.get) if factors else None
    best = max(factors.values()) if factors else 1.0
    unexplained = actual > (1.0 + tolerance) and best < 1.05

    flags: List[str] = []
    if unexplained:
        flags.append("unexplained_speedup")
    if actual > (1.0 + tolerance) and best > 1.0 and actual > best * (1.0 + tolerance):
        flags.append("speedup_exceeds_model_bound")
    if factors.get("instructions", 1.0) > 1.2 and factors.get("achieved_bw", 1.0) < 1.05 \
            and factors.get("dram_traffic", 1.0) < 1.05:
        flags.append("work_removed_not_efficiency")

    consistent = not unexplained
    summary = (
        f"actual {actual:.2f}x; largest resource factor {dominant}={best:.2f}x "
        f"({'consistent' if consistent else 'UNEXPLAINED'})"
    )
    return AttributionReport(actual, factors, dominant, consistent, unexplained, flags, summary)


# --------------------------------------------------------------------------- #
# Lever list
# --------------------------------------------------------------------------- #

@dataclass
class Lever:
    mechanism: str
    lo: float
    hi: float
    priority: int
    rationale: str
    hack_risk: bool = False


def lever_list(
    point: rl.RooflinePoint,
    peak: rl.GpuPeak,
    stalls: Optional[StallProfile] = None,
    *,
    top_k: int = 4,
) -> List[Lever]:
    """Order candidate mechanisms for the *current* operating point, model-bounded."""
    candidates = ["vectorize", "l2_tiling", "pipeline", "tensor_core", "epilogue_fuse",
                  "warp_specialize", "split_k"]
    kwargs: Dict[str, float] = {}
    if stalls and stalls.dominant:
        kwargs["stall_fraction"] = min(max(stalls.total / 100.0, 0.0), 0.8)
        # Bias toward the mechanism that attacks the dominant stall.
        preferred = STALL_TO_MECHANISM.get(stalls.dominant)
        if preferred and preferred not in candidates:
            candidates.append(preferred)

    bounds = rl.rank_mechanisms(point, peak, candidates, **kwargs)
    levers: List[Lever] = []
    for i, b in enumerate(bounds[:top_k]):
        rationale = b.reason
        if stalls and stalls.dominant and STALL_TO_MECHANISM.get(stalls.dominant) == b.name:
            rationale += f" (attacks dominant stall '{stalls.dominant}')"
        levers.append(Lever(b.name, b.lo, b.hi, i + 1, rationale, hack_risk=(b.name == "reduce_precision")))
    return levers


# --------------------------------------------------------------------------- #
# Prompt block
# --------------------------------------------------------------------------- #

def build_judge_block(
    counters: dict,
    peak: rl.GpuPeak,
    flops: float,
    *,
    prev_counters: Optional[dict] = None,
    compute_dtype: str = "bf16",
    kernel_name: str = "kernel",
) -> str:
    """Render the causal, model-grounded block the ARGUS Judge reads (replaces the dump)."""
    point = rl.counters_to_point(counters, peak, flops, compute_dtype)
    stalls = stall_profile(counters)
    lines = [
        f"## Operating point ({kernel_name})",
        f"- device time: {point.time_s*1e3:.4f} ms",
        f"- DRAM traffic: {point.dram_bytes/1e6:.1f} MB at {point.achieved_dram_bw/1e9:.0f} GB/s "
        f"({point.dram_pct*100:.0f}% of {peak.dram_bw/1e12:.1f} TB/s peak)",
        f"- compute: {point.achieved_flops/1e12:.1f} TFLOP/s ({compute_dtype}, "
        f"{point.compute_pct*100:.0f}% of peak; tensor pipe {point.tensor_pct*100:.0f}%)",
        f"- arithmetic intensity: {point.arithmetic_intensity:.2f} flop/byte",
        f"- bottleneck: **{point.bottleneck.upper()}**",
    ]
    if point.l2_bytes:
        lines.append(f"- L2 traffic: {point.l2_bytes/1e6:.1f} MB (reuse "
                     f"{point.dram_bytes/max(point.l2_bytes,1e-9):.2f} of DRAM)")
    if stalls.dominant:
        top = sorted(stalls.values.items(), key=lambda kv: kv[1], reverse=True)[:3]
        lines.append("- top stalls: " + ", ".join(f"{k}={v:.0f}%" for k, v in top))

    if prev_counters is not None:
        rep = attribute_delta(prev_counters, counters, peak, flops, compute_dtype=compute_dtype)
        lines.append("## What changed since the last candidate")
        lines.append(f"- {rep.summary}")
        lines.append("- factor speedups: " + ", ".join(
            f"{k}={v:.2f}x" for k, v in sorted(rep.factor_speedups.items(),
                                              key=lambda kv: kv[1], reverse=True)))
        lines.append("- flags: " + (", ".join(rep.flags) if rep.flags else "none"))

    lines.append("## Candidate levers (model-bounded)")
    for lv in lever_list(point, peak, stalls, top_k=4):
        tag = " [HACK RISK]" if lv.hack_risk else ""
        lines.append(f"{lv.priority}. `{lv.mechanism}` -> <= {lv.hi:.2f}x{tag}: {lv.rationale}")
    lines.append("Pick **one** lever that is both feasible and highest-ceiling; if every "
                 "lever is capped near 1.0x, say so and propose the structural change instead.")
    return "\n".join(lines)
