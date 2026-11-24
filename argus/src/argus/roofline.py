"""Analytic roofline model + mechanism feasibility bounds (ARGUS contribution C2b).

CudaForge feeds the raw Nsight-Compute counter block straight into the Judge prompt and
lets the LLM guess what to do next.  ARGUS instead keeps a *bounded analytic model* of the
machine so it can:

1. turn counters into an achieved-bandwidth / achieved-flops point and a bottleneck class
   (``counters_to_point`` / ``bottleneck_class``);
2. bound the multiplicative speedup that a proposed *mechanism* can possibly deliver from
   that point (`mechanism_bounds`); and
3. **pre-screen** a proposed mechanism before spending a GPU evaluation on it
   (`prescreen`) -- the scarce resource on the target hardware.

The model is deliberately simple and its numbers are priors: the peak constants are
approximate and should be replaced by measured peaks (``GpuPeak.from_file``).  What the
model buys is the *ranking* and the *ordering* of mechanisms, and a hard upper bound that
refuses physically impossible claims (a memory-bound kernel cannot be sped up by a tensor
core when the tensor pipe is already idle and DRAM is saturated).

All units are SI: seconds, bytes, flops.  Latency in ms where it enters the score.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GpuPeak:
    """Approximate peak rates for a target GPU (override with measured values)."""

    name: str
    dram_bw: float            # bytes / s, HBM
    l2_bw: float              # bytes / s, L2
    fp32_flops: float         # flops / s, FP32 pipe
    tensor_bf16: float        # flops / s, dense bf16/fp16 tensor core
    tensor_fp8: float         # flops / s, dense fp8 tensor core
    sm_clock: float           # Hz, steady-state SM clock
    sm_count: int
    note: str = ""

    def from_file(path) -> "GpuPeak":
        raw = json.loads(Path(path).read_text())
        return GpuPeak(**raw)


#: Priors.  Board measurement locks the B200 SM clock at 1.5 GHz; B300 is Blackwell-Ultra.
PEAKS: Dict[str, GpuPeak] = {
    "H100": GpuPeak(
        name="H100-SXM",
        dram_bw=3.35e12, l2_bw=12.0e12,
        fp32_flops=67e12, tensor_bf16=989e12, tensor_fp8=1979e12,
        sm_clock=1.755e9, sm_count=132,
        note="Hopper sm90; dense (non-sparse) tensor rates",
    ),
    "B200": GpuPeak(
        name="B200",
        dram_bw=8.0e12, l2_bw=20.0e12,
        fp32_flops=75e12, tensor_bf16=2250e12, tensor_fp8=4500e12,
        sm_clock=1.5e9, sm_count=148,
        note="Blackwell sm100; SM clock locked at 1500 MHz on the board",
    ),
    "B300": GpuPeak(
        name="B300",
        dram_bw=8.0e12, l2_bw=20.0e12,
        fp32_flops=80e12, tensor_bf16=2500e12, tensor_fp8=5000e12,
        sm_clock=1.5e9, sm_count=148,
        note="Blackwell Ultra sm100a; PRIOR only -- replace with measured peaks",
    ),
}

#: Bottleneck classes.
MEMORY = "memory"
COMPUTE = "compute"
LATENCY = "latency"
BALANCED = "balanced"


@dataclass
class RooflinePoint:
    """An achieved operating point derived from counters."""

    flops: float                 # useful flops in the measured region
    dram_bytes: float            # HBM traffic
    l2_bytes: float              # L2 traffic (0 if unknown)
    time_s: float                # measured device time
    achieved_dram_bw: float      # bytes / s
    achieved_flops: float        # flops / s
    dram_pct: float              # of peak
    compute_pct: float           # of the relevant compute peak
    tensor_pct: float            # of tensor peak (0 if unused)
    arithmetic_intensity: float  # flops / byte
    bottleneck: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def counters_to_point(
    counters: dict,
    peak: GpuPeak,
    flops: float,
    compute_dtype: str = "bf16",
) -> RooflinePoint:
    """Build a :class:`RooflinePoint` from an Nsight-Compute counter dict.

    Recognised keys (a subset of CudaForge's ``run_ncu.METRICS``):
      ``dram__bytes_read.sum``, ``dram__bytes_write.sum`` (bytes),
      ``sm__cycles_active.avg`` (cycles), ``sm__inst_executed_pipe_tensor...`` pct,
      and optionally ``lts__throughput...`` / ``lts__t_sector_*``.
    A missing time is estimated from cycles / clock.
    """
    read = float(counters.get("dram__bytes_read.sum", 0.0) or 0.0)
    write = float(counters.get("dram__bytes_write.sum", 0.0) or 0.0)
    dram_bytes = read + write

    cycles = float(counters.get("sm__cycles_active.avg", 0.0) or 0.0)
    time_s = float(counters.get("time_s", 0.0) or 0.0)
    if time_s <= 0.0 and cycles > 0.0:
        time_s = cycles / peak.sm_clock
    if time_s <= 0.0:
        raise ValueError("cannot determine time: need 'time_s' or 'sm__cycles_active.avg'")

    compute_peak = {
        "fp32": peak.fp32_flops,
        "bf16": peak.tensor_bf16,
        "fp16": peak.tensor_bf16,
        "fp8": peak.tensor_fp8,
    }.get(compute_dtype, peak.tensor_bf16)

    achieved_dram_bw = dram_bytes / time_s
    achieved_flops = flops / time_s if flops else 0.0
    tensor_pct_raw = counters.get("sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active")

    pt = RooflinePoint(
        flops=flops,
        dram_bytes=dram_bytes,
        l2_bytes=float(counters.get("l2_bytes", 0.0) or 0.0),
        time_s=time_s,
        achieved_dram_bw=achieved_dram_bw,
        achieved_flops=achieved_flops,
        dram_pct=achieved_dram_bw / peak.dram_bw,
        compute_pct=(achieved_flops / compute_peak) if flops else 0.0,
        tensor_pct=(float(tensor_pct_raw) / 100.0) if tensor_pct_raw else 0.0,
        arithmetic_intensity=(flops / dram_bytes) if dram_bytes else math.inf,
        bottleneck=BALANCED,
    )
    pt.bottleneck = bottleneck_class(pt, peak, compute_dtype)
    return pt


def bottleneck_class(point: RooflinePoint, peak: GpuPeak, compute_dtype: str = "bf16") -> str:
    """Classify a point by comparing the memory-time and compute-time lower bounds."""
    compute_peak = {
        "fp32": peak.fp32_flops,
        "bf16": peak.tensor_bf16,
        "fp16": peak.tensor_bf16,
        "fp8": peak.tensor_fp8,
    }.get(compute_dtype, peak.tensor_bf16)
    mem_time = point.dram_bytes / peak.dram_bw if point.dram_bytes else 0.0
    comp_time = (point.flops / compute_peak) if point.flops else 0.0
    roof = max(mem_time, comp_time)
    if roof <= 0.0 or point.time_s <= 0.0:
        return LATENCY
    util = roof / point.time_s                       # 1 => perfectly at the roof
    if mem_time >= 1.3 * comp_time:
        return MEMORY
    if comp_time >= 1.3 * mem_time:
        return COMPUTE
    return LATENCY if util < 0.5 else BALANCED


# --------------------------------------------------------------------------- #
# Mechanism bounds
# --------------------------------------------------------------------------- #

@dataclass
class MechanismBound:
    """A bounded, physically-motivated prediction for one mechanism."""

    name: str
    lo: float          # lower bound on achievable speedup (multiplicative, >= 1)
    hi: float          # upper bound
    feasible: bool     # hi >= FEASIBLE_MIN
    reason: str
    evidence: Dict[str, float] = field(default_factory=dict)


FEASIBLE_MIN = 1.05

#: Hardware mechanism prerequisites.  Applying a mechanism before its prerequisite is an
#: *unrecoverable executable-structure mistake* on the target -- tensor-core MMA needs the
#: operand layout tiled first, warp specialisation needs the software pipeline first.  This
#: is an architecture fact (the sort of thing a hardware knowledge base encodes), and it is
#: what lets ARGUS pre-screen a mechanism out of the UCB arms before a GPU eval is spent on
#: a candidate that would lock itself into a worse ceiling.
PREREQS: Dict[str, Tuple[str, int]] = {
    "l2_tiling": ("vectorize", 1),
    "tensor_core": ("l2_tiling", 1),
    "warp_specialize": ("pipeline", 1),
}


def prereq_satisfied(mechanism: str, levels: Dict[str, int]) -> bool:
    req = PREREQS.get(mechanism)
    if req is None:
        return True
    return levels.get(req[0], 0) >= req[1]


def _cap(value: float, lo: float = 1.0, hi: float = 1000.0) -> float:
    return max(lo, min(hi, value))


def mechanism_bounds(
    mechanism: str,
    point: RooflinePoint,
    peak: GpuPeak,
    *,
    reuse_factor: float = 2.0,
    stall_fraction: float = 0.0,
) -> MechanismBound:
    """Bound the multiplicative speedup a named mechanism can deliver from ``point``.

    Mechanisms (the same vocabulary the Judge is asked to use):

    ``vectorize``        widen loads/stores so achieved DRAM BW approaches peak.
    ``l2_tiling``        raise on-chip reuse (cut DRAM bytes by ``reuse_factor``).
    ``pipeline``         overlap memory and compute (removes ``stall_fraction`` of stalls).
    ``epilogue_fuse``    drop a second DRAM pass over the output (halves output traffic).
    ``tensor_core``      move the hot loop onto the tensor pipe.
    ``warp_specialize``  remove barrier stalls.
    ``split_k``          add parallelism to a latency-bound reduction.
    ``reduce_precision`` move to a narrower fp dtype -- *flagged* as a hack risk.
    """
    bw = point.achieved_dram_bw
    bw_headroom = peak.dram_bw / max(bw, 1e-9)
    compute_headroom = (peak.fp32_flops / point.achieved_flops) if point.achieved_flops else 1.0

    if mechanism == "vectorize":
        hi = _cap(bw_headroom, 1.0, 8.0)
        return MechanismBound(
            mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
            f"achieved DRAM BW is {point.dram_pct*100:.0f}% of peak; widening accesses can "
            f"recover at most {hi:.2f}x",
            {"dram_pct": point.dram_pct},
        )

    if mechanism == "l2_tiling":
        if point.bottleneck != MEMORY:
            return MechanismBound(mechanism, 1.0, 1.0, False,
                                  "not memory-bound; reuse does not help",
                                  {"bottleneck": point.bottleneck})
        hi = _cap(reuse_factor, 1.0, 20.0)
        # Cannot beat the compute/compute-or-DRAM floor.
        floor_factor = 1.0
        if point.compute_pct > 0:
            floor_factor = 1.0 / max(point.compute_pct, 1e-9)
        hi = min(hi, max(1.0, floor_factor))
        return MechanismBound(mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
                              f"DRAM traffic cut up to {reuse_factor:.1f}x, floored by the "
                              f"compute/DRAM lower bound",
                              {"reuse_factor": reuse_factor})

    if mechanism == "pipeline":
        sf = _cap(stall_fraction, 0.0, 0.95)
        hi = _cap(1.0 / max(1.0 - sf, 1e-3), 1.0, 10.0)
        return MechanismBound(mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
                              f"stall fraction {sf:.2f} => at most {hi:.2f}x",
                              {"stall_fraction": sf})

    if mechanism == "epilogue_fuse":
        if point.dram_bytes <= 0:
            return MechanismBound(mechanism, 1.0, 1.0, False, "no DRAM traffic", {})
        # Removing one full pass over a tensor of comparable size: bounded by total traffic.
        hi = _cap(2.0, 1.0, 4.0)
        return MechanismBound(mechanism, 1.0, hi, point.bottleneck == MEMORY,
                              "fusing the epilogue removes at most one tensor pass",
                              {})

    if mechanism == "tensor_core":
        if point.bottleneck == MEMORY:
            return MechanismBound(mechanism, 1.0, 1.0, False,
                                  "memory-bound; the tensor pipe cannot move DRAM bytes faster",
                                  {"bottleneck": point.bottleneck})
        hi = _cap(peak.tensor_bf16 / max(peak.fp32_flops, 1e-9), 1.0, 32.0)
        return MechanismBound(mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
                              f"tensor/fp32 peak ratio caps the win at {hi:.1f}x",
                              {"tensor_over_fp32": hi})

    if mechanism == "warp_specialize":
        sf = _cap(stall_fraction, 0.0, 0.95)
        hi = _cap(1.0 / max(1.0 - sf, 1e-3), 1.0, 6.0)
        return MechanismBound(mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
                              f"barrier stalls {sf:.2f}", {"stall_fraction": sf})

    if mechanism == "split_k":
        if point.bottleneck != LATENCY:
            return MechanismBound(mechanism, 1.0, 1.0, False,
                                  "already roof-limited; extra parallelism will not help", {})
        hi = _cap(compute_headroom, 1.0, 16.0)
        return MechanismBound(mechanism, 1.0, hi, hi >= FEASIBLE_MIN,
                              "latency-bound: exposure of parallelism is the lever", {})

    if mechanism == "reduce_precision":
        return MechanismBound(mechanism, 1.0, 4.0, True,
                              "flagged: narrower dtypes are the #1 SOL reward-hack class "
                              "(259/589 flagged submissions); the verifier must clear it",
                              {"hack_risk": 1.0})

    return MechanismBound(mechanism, 1.0, 1.0, False, "unknown mechanism", {})


@dataclass
class PrescreenResult:
    mechanism: str
    feasible: bool
    bound: MechanismBound
    would_spend_eval: bool
    reason: str


def prescreen(
    mechanism: str,
    point: RooflinePoint,
    peak: GpuPeak,
    *,
    min_hi: float = FEASIBLE_MIN,
    **kwargs,
) -> PrescreenResult:
    """Decide, *in model space*, whether to spend a GPU evaluation on ``mechanism``.

    A mechanism whose model bound cannot reach ``min_hi`` is skipped without touching the
    GPU.  This is the "pre-screen before you spend" saving that matters when evaluations
    are the scarce resource.
    """
    bound = mechanism_bounds(mechanism, point, peak, **kwargs)
    spend = bound.hi >= min_hi
    reason = (
        f"model bound hi={bound.hi:.2f}x -> spend an evaluation"
        if spend
        else f"model bound hi={bound.hi:.2f}x below {min_hi:.2f}x -> skip (save the GPU)"
    )
    return PrescreenResult(mechanism, bound.feasible, bound, spend, f"{bound.reason}; {reason}")


def rank_mechanisms(
    point: RooflinePoint,
    peak: GpuPeak,
    candidates: List[str],
    **kwargs,
) -> List[MechanismBound]:
    """Model-space ordering of mechanisms by upper bound (descending)."""
    bounds = [mechanism_bounds(m, point, peak, **kwargs) for m in candidates]
    return sorted(bounds, key=lambda b: b.hi, reverse=True)
