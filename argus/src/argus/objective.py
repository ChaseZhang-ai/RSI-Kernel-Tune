"""The exact SOL-ExecBench scoring algebra.

This is *shared substrate*, not an ARGUS contribution: the same formula is used by the
sibling track (``vulcan/``) and by every serious entrant, and it is recovered directly
from the public board.  ARGUS needs it so that (a) the search optimizes the metric the
board actually scores, and (b) the verifier can reproduce the board's per-element
tolerance rule.  ARGUS's own contributions live in ``verifier`` / ``attribution`` /
``search``.

Rules (SOL-ExecBench, evaluation stack v1.1, collection = one kernel family):

    per workload   S_w = 1 / (1 + (T_w - T_SOL,w) / (T_b,w - T_SOL,w))
    per kernel     S_k = geomean over the kernel's workloads of S_w
    collection     S   = unweighted arithmetic mean over the kernels of S_k

``S = 0.5`` when a workload exactly matches the Scoring Baseline ``T_b`` and ``S -> 1`` as
it approaches the Speed-of-Light bound ``T_SOL``.  ``Fast1`` counts workloads strictly
faster than ``T_b``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# Default correctness tolerance published by SOL-ExecBench (docs/definition.md).
SOL_ATOL = 1e-2
SOL_RTOL = 1e-2
SOL_MATCHED_RATIO = 0.99

# Reference anchors on every board.
SCORING_BASELINE_SCORE = 0.5
SOL_BOUND_SCORE = 1.0


class ObjectiveError(ValueError):
    """Raised for ill-formed targets (e.g. T_b == T_SOL with a non-trivial target)."""


def workload_score(t: float, t_sol: float, t_b: float) -> float:
    """SOL score of one workload from its latency ``t`` (ms) and the two anchors."""
    if t_sol < 0 or t_b < 0:
        raise ObjectiveError("latencies must be non-negative")
    gap = t_b - t_sol
    if abs(gap) <= 0.0:
        # Degenerate: the baseline already sits on the roof.  Monotone clamp.
        return SOL_BOUND_SCORE if t <= t_sol else SCORING_BASELINE_SCORE
    s = 1.0 / (1.0 + (t - t_sol) / gap)
    # The roof is unreachable by construction; a latency below T_SOL is a measurement
    # artefact, so the score is clamped to the bound.
    return min(s, SOL_BOUND_SCORE)


def latency_for_score(s: float, t_sol: float, t_b: float) -> float:
    """Invert :func:`workload_score`: the latency that would yield score ``s``.

    ``s`` is clamped to ``[0.5, 1.0)``; below 0.5 the required latency exceeds ``t_b`` and
    the score is almost flat, so targets there are pointless.
    """
    if not (0.0 < s < 1.0):
        if s >= 1.0:
            return t_sol
        raise ObjectiveError("score must be in (0, 1)")
    s = max(s, SCORING_BASELINE_SCORE)
    gap = t_b - t_sol
    return t_sol + (1.0 - s) / s * gap


def kernel_score(workload_scores: Sequence[float]) -> float:
    """Geometric mean of a kernel's per-workload scores (the board's rule)."""
    vals = [max(v, 1e-300) for v in workload_scores]
    if not vals:
        raise ObjectiveError("kernel has no workloads")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def collection_score(kernel_scores: Sequence[float]) -> float:
    """Unweighted arithmetic mean of per-kernel scores (the board's rule)."""
    if not kernel_scores:
        raise ObjectiveError("collection has no kernels")
    return sum(kernel_scores) / len(kernel_scores)


def speedup_vs_baseline(t: float, t_b: float) -> float:
    """Raw speedup over the Scoring Baseline (the board's ``Avg Speedup`` column)."""
    return t_b / max(t, 1e-300)


def is_fast1(t: float, t_b: float) -> bool:
    """A workload earns Fast1 credit when it is *strictly* faster than ``T_b``."""
    return t < t_b


@dataclass(frozen=True)
class WorkloadTarget:
    """One workload's anchors and, optionally, the latency needed to reach ``target_s``."""

    name: str
    t_sol: float
    t_b: float
    target_s: float = SOL_BOUND_SCORE

    @property
    def target_ms(self) -> float:
        return latency_for_score(self.target_s, self.t_sol, self.t_b)

    @property
    def headroom(self) -> float:
        """Score points available between the baseline and the roof (1 - 0.5 = 0.5 max)."""
        return SOL_BOUND_SCORE - SCORING_BASELINE_SCORE


@dataclass
class KernelRecord:
    """A kernel's per-workload anchors, i.e. one entry of a collection."""

    name: str
    workloads: List[WorkloadTarget]

    def best_public_score(self, published_scores: Optional[Iterable[float]] = None) -> float:
        if published_scores is None:
            return 0.0
        return max(published_scores)


def collection_from_latencies(
    kernel_latencies: dict,
    anchors: dict,
) -> float:
    """Convenience: build a collection score from ``{kernel: [t...]}`` and
    ``{kernel: [(t_sol, t_b)...]}``."""
    per_kernel: List[float] = []
    for kname, lats in kernel_latencies.items():
        pairs = anchors[kname]
        per_kernel.append(
            kernel_score([workload_score(t, ts, tb) for t, (ts, tb) in zip(lats, pairs)])
        )
    return collection_score(per_kernel)
