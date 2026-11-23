"""Objective algebra for the SOL-ExecBench / FlashInfer-Bench leaderboard.

The board scores a *collection* of kernels.  This module encodes the scoring rule
exactly, because optimizing anything else (raw speedup, arithmetic-mean latency,
percentage-of-baseline) lands in the wrong place.

Scoring rule (recovered from public leaderboard data and the upstream harness docs
``forgekernel`` ships in ``docs/leaderboards.md``)::

    per workload    S_w = 1 / (1 + (T_w - T_SOL,w) / (T_b,w - T_SOL,w))
    per kernel      S_k = geomean_w S_w                    # geometric mean over workloads
    collection      S   = mean_k S_k                       # UNWEIGHTED arithmetic mean

``T_b`` is the board's *Scoring Baseline* latency and ``T_SOL`` its *SOL Bound*
latency, both published per workload.  ``S = 0.5`` at ``T_b`` and ``S -> 1`` as the
kernel approaches the speed-of-light floor.

The ``collection_score`` / ``kernel_score`` helpers reproduce the published numbers
to six decimals (ForgeKernel 0.815204, Golden Bridge 0.815321) when fed the public
per-kernel scores; see ``data/`` and ``docs/LEADERBOARD_ANALYSIS.md``.

Consequences that drive the rest of VULCAN:

* Because the collection score is an *unweighted mean per kernel*, a kernel with
  ``S = 0.63`` has ten times the headroom of a kernel at ``S = 0.99`` yet counts the
  same.  Effort belongs in the mid/low kernels, not in polishing attention kernels
  that already sit near the floor.
* ``S`` is convex in latency, so the marginal score from a fixed *fractional* latency
  cut grows as the kernel leaves the floor; the allocator (``vulcan.allocator``)
  therefore works with a latency-improvement model, not a speedup model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class ScoreError(ValueError):
    """Raised when the scoring inputs are physically inconsistent."""


# --------------------------------------------------------------------------- #
# Core scalar algebra
# --------------------------------------------------------------------------- #

def validate_bounds(t_b_ms: float, t_sol_ms: float) -> None:
    """The board publishes a baseline above its speed-of-light floor."""
    if not (t_b_ms > 0.0):
        raise ScoreError(f"baseline latency must be > 0, got {t_b_ms!r}")
    if not (t_sol_ms >= 0.0):
        raise ScoreError(f"SOL latency must be >= 0, got {t_sol_ms!r}")
    if t_sol_ms >= t_b_ms:
        raise ScoreError(
            f"SOL floor {t_sol_ms!r} must be strictly below baseline {t_b_ms!r}"
        )


def sol_score(t_ms: float, t_b_ms: float, t_sol_ms: float) -> float:
    """One workload's SOL score ``S``.

    ``S = 1 / (1 + (T - T_SOL) / (T_b - T_SOL))``, clipped to ``(0, 1]``.

    ``T <= T_SOL`` yields exactly ``1.0``; a very slow kernel approaches ``0``.
    """
    validate_bounds(t_b_ms, t_sol_ms)
    if not (t_ms > 0.0) or not math.isfinite(t_ms):
        raise ScoreError(f"candidate latency must be finite and > 0, got {t_ms!r}")
    if t_ms <= t_sol_ms:
        return 1.0
    gap = t_b_ms - t_sol_ms
    return 1.0 / (1.0 + (t_ms - t_sol_ms) / gap)


def latency_for_score(s: float, t_b_ms: float, t_sol_ms: float) -> float:
    """Inverse of :func:`sol_score`: the latency needed to reach score ``s``.

    Useful to turn a score target into the ``[target] ms`` the upstream loop needs.
    """
    validate_bounds(t_b_ms, t_sol_ms)
    if not (0.0 < s <= 1.0):
        raise ScoreError(f"score must be in (0, 1], got {s!r}")
    if s >= 1.0:
        return t_sol_ms
    gap = t_b_ms - t_sol_ms
    return t_sol_ms + gap * (1.0 / s - 1.0)


def is_fast(t_ms: float, t_b_ms: float) -> bool:
    """``Fast_1`` counts a workload only when it is *strictly* faster than baseline."""
    return t_ms < t_b_ms


def score_sensitivity(t_ms: float, t_b_ms: float, t_sol_ms: float) -> float:
    """``|dS/dT|``: score gained per unit of latency removed (always positive)."""
    validate_bounds(t_b_ms, t_sol_ms)
    if t_ms <= t_sol_ms:
        return 0.0
    gap = t_b_ms - t_sol_ms
    return gap / (gap + (t_ms - t_sol_ms)) ** 2


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Workload:
    """One (definition, shape) pair the board times."""

    name: str
    t_b_ms: float
    t_sol_ms: float

    def __post_init__(self) -> None:
        validate_bounds(self.t_b_ms, self.t_sol_ms)

    @property
    def ratio_floor(self) -> float:
        """``T_SOL / T_b``: how much of the baseline latency is irreducible."""
        return self.t_sol_ms / self.t_b_ms


@dataclass
class KernelResult:
    """A candidate latency per workload for one kernel."""

    name: str
    latencies_ms: Sequence[float]
    workloads: Sequence[Workload]

    def __post_init__(self) -> None:
        if len(self.latencies_ms) != len(self.workloads):
            raise ScoreError(
                f"{self.name}: {len(self.latencies_ms)} latencies for "
                f"{len(self.workloads)} workloads"
            )


@dataclass
class CollectionResult:
    """A full collection: one latency per workload, for every kernel."""

    kernels: Dict[str, KernelResult] = field(default_factory=dict)

    def add(self, kernel: KernelResult) -> "CollectionResult":
        self.kernels[kernel.name] = kernel
        return self


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def kernel_score(result: KernelResult) -> float:
    """Geometric mean of the per-workload SOL scores.

    A workload that is incorrect is scored ``0`` and drags the geometric mean to
    ``0`` for that kernel -- which is why ``precision_guard`` treats correctness as
    a hard constraint, not a trade-off.
    """
    if not result.workloads:
        raise ScoreError(f"{result.name}: no workloads")
    logs: List[float] = []
    for lat, w in zip(result.latencies_ms, result.workloads):
        s = sol_score(lat, w.t_b_ms, w.t_sol_ms)
        if s <= 0.0:
            return 0.0
        logs.append(math.log(s))
    return math.exp(sum(logs) / len(logs))


def kernel_fast1(result: KernelResult) -> Tuple[int, int]:
    """``(fast, total)`` workloads strictly faster than the scoring baseline."""
    fast = sum(1 for lat, w in zip(result.latencies_ms, result.workloads)
               if is_fast(lat, w.t_b_ms))
    return fast, len(result.workloads)


def collection_score(kernel_scores: Iterable[float]) -> float:
    """The board's collection score: unweighted arithmetic mean over kernels."""
    values = list(kernel_scores)
    if not values:
        raise ScoreError("empty collection")
    return sum(values) / len(values)


def collection_score_from_results(collection: CollectionResult) -> float:
    return collection_score(kernel_score(k) for k in collection.kernels.values())


def collection_fast1(collection: CollectionResult) -> Tuple[int, int]:
    fast = total = 0
    for k in collection.kernels.values():
        f, t = kernel_fast1(k)
        fast += f
        total += t
    return fast, total


def collection_speedup(
    collection: CollectionResult,
    kernel_weighting: str = "geomean",
) -> float:
    """Average speedup over the scoring baseline (secondary metric, for reporting).

    The board reports a geometric mean of per-kernel geometric-mean speedups; this
    helper mirrors that while tolerating the same inputs as the score helpers.
    """
    per_kernel: List[float] = []
    for k in collection.kernels.values():
        logs = [math.log(w.t_b_ms / lat)
                for lat, w in zip(k.latencies_ms, k.workloads)]
        per_kernel.append(math.exp(sum(logs) / len(logs)))
    if kernel_weighting == "geomean":
        return math.exp(sum(math.log(s) for s in per_kernel) / len(per_kernel))
    if kernel_weighting == "mean":
        return sum(per_kernel) / len(per_kernel)
    raise ScoreError(f"unknown kernel_weighting {kernel_weighting!r}")


# --------------------------------------------------------------------------- #
# Target planning: "what latency do I need on kernel k to hit collection score X?"
# --------------------------------------------------------------------------- #

def required_kernel_score_for_collection(
    target_collection: float,
    other_kernel_scores: Sequence[float],
) -> float:
    """Per-kernel score ``s`` such that the collection mean equals ``target``.

    Given the (fixed) scores of all other kernels, the kernel in question must reach
    ``s = target * n - sum(others)`` where ``n = len(others) + 1``.
    """
    n = len(other_kernel_scores) + 1
    needed = target_collection * n - sum(other_kernel_scores)
    return needed


def kernel_latency_targets(
    workloads: Sequence[Workload],
    target_kernel_score: float,
) -> List[float]:
    """Per-workload latencies that make the kernel's geometric-mean score equal ``target``.

    The target is split uniformly in score space, i.e. every workload is driven to the
    same score ``target_kernel_score``; this is the minimum-variance allocation because
    the geometric mean is maximised, for a fixed product, at equal factors.
    """
    return [latency_for_score(target_kernel_score, w.t_b_ms, w.t_sol_ms)
            for w in workloads]


def headroom_ranking(collection: CollectionResult) -> List[Tuple[str, float, float]]:
    """``[(kernel, score, headroom=1-score), ...]`` sorted by largest headroom first.

    Equal kernel weight in the collection mean makes raw headroom the correct
    first-order priority signal.
    """
    rows = [(name, kernel_score(k), 1.0 - kernel_score(k))
            for name, k in collection.kernels.items()]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows
