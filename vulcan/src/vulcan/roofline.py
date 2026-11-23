"""Roofline diagnosis: turn profile counters into a bottleneck class and levers.

The knowledge graph retrieves mechanisms by *measured* features (``dram_pct``,
``tensor_pct``, ``bytes_per_flop`` ...).  This module produces those features from raw
counters and classifies the kernel as memory-, compute- or latency-bound.  It is pure
arithmetic, so the mapping is testable without a GPU and identical on the B300.

Ridge point: ``AI* = peak_flops / peak_bandwidth``.  Below it the kernel is
bandwidth-bound; above it, compute-bound.  A kernel that is far below *both* ceilings is
latency/occupancy-bound (too little parallel work, launch gaps, or poor overlap), which
needs different levers than either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HardwarePeak:
    bandwidth_bytes_per_s: float
    flops_per_s: float
    name: str = ""

    def ridge_point(self) -> float:
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth must be > 0")
        return self.flops_per_s / self.bandwidth_bytes_per_s


@dataclass
class Counters:
    elapsed_s: float
    dram_bytes: float
    flops: float

    def __post_init__(self) -> None:
        if self.elapsed_s <= 0:
            raise ValueError("elapsed_s must be > 0")


@dataclass
class Diagnosis:
    bound: str                     # "memory" | "compute" | "latency"
    arithmetic_intensity: float
    achieved_bandwidth: float
    achieved_flops: float
    pct_of_bw_ceiling: float
    pct_of_flop_ceiling: float
    features: Dict[str, float]
    levers: List[str]

    def as_dict(self) -> dict:
        return {
            "bound": self.bound,
            "arithmetic_intensity": round(self.arithmetic_intensity, 3),
            "pct_of_bw_ceiling": round(self.pct_of_bw_ceiling, 2),
            "pct_of_flop_ceiling": round(self.pct_of_flop_ceiling, 2),
            "features": {k: round(v, 3) for k, v in self.features.items()},
            "levers": self.levers,
        }


_MEMORY_LEVERS = [
    "L2-aware CTA rasterization (re-read amplification)",
    "TMA / cp.async.bulk prefetch and deeper staging",
    "tile/register reuse to raise arithmetic intensity",
    "coalesced access atoms; vectorised loads; avoid bank conflicts",
    "fuse the epilogue to avoid a full DRAM round-trip of the output",
]
_COMPUTE_LEVERS = [
    "raise MMA/tensor-core utilisation (atom shape, warp roles)",
    "pipeline depth / overlapping MMA with elementwise",
    "remove instruction overhead in the inner loop (index math, predication)",
    "keep the accumulator in the legal precision to avoid fallback paths",
    "persistent CTAs to cut launch and tail effects",
]
_LATENCY_LEVERS = [
    "persistent CTAs / split work further to fill the SMs",
    "batch small independent work items into one launch",
    "reduce synchronisation and handshake depth",
    "prefetch so loads overlap the first MMA of a tile",
]


def diagnose(counters: Counters, peak: HardwarePeak,
             latency_frac: float = 0.30) -> Diagnosis:
    """Classify the bottleneck and propose levers.

    ``latency_frac``: if the kernel reaches less than this fraction of *both*
    ceilings, it is treated as latency-bound rather than memory/compute-bound.
    """
    ai = counters.flops / max(counters.dram_bytes, 1e-12)
    bw = counters.dram_bytes / counters.elapsed_s
    flops = counters.flops / counters.elapsed_s
    bw_ceiling = bw / peak.bandwidth_bytes_per_s if peak.bandwidth_bytes_per_s else 0.0
    fl_ceiling = flops / peak.flops_per_s if peak.flops_per_s else 0.0
    ridge = peak.ridge_point()

    if bw_ceiling < latency_frac and fl_ceiling < latency_frac:
        bound = "latency"
        levers = list(_LATENCY_LEVERS)
    elif ai < ridge:
        bound = "memory"
        levers = list(_MEMORY_LEVERS)
    else:
        bound = "compute"
        levers = list(_COMPUTE_LEVERS)

    features = {
        "arithmetic_intensity": ai,
        "bytes_per_flop": 1.0 / ai if ai > 0 else float("inf"),
        "dram_pct": 100.0 * bw_ceiling,
        "tensor_pct": 100.0 * fl_ceiling,
        "ridge_point": ridge,
    }
    return Diagnosis(
        bound=bound,
        arithmetic_intensity=ai,
        achieved_bandwidth=bw,
        achieved_flops=flops,
        pct_of_bw_ceiling=100.0 * bw_ceiling,
        pct_of_flop_ceiling=100.0 * fl_ceiling,
        features=features,
        levers=levers,
    )


# Reference peaks (dense, no sparsity).  B300/GB300 figures are approximate and must be
# replaced by the values the board's T_SOL implies; kept here for ratio sanity checks.
PEAKS: Dict[str, HardwarePeak] = {
    "H100_SXM_bf16": HardwarePeak(3.35e12, 989e12, "H100 SXM (bf16 dense)"),
    "B200_bf16": HardwarePeak(8.0e12, 2250e12, "B200 (bf16 dense)"),
}
