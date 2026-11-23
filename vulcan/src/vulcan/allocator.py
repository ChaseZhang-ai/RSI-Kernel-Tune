"""Score-aware budget allocation.

Given a fixed number of GPU hours and a *latency-improvement model*, where should the
hours go so the **leaderboard's** collection score rises the most?

This is the part most kernel-agent systems get wrong.  They target "beat the
reference by X%" per kernel, which is not the board's objective.  Because the
collection score is an unweighted mean of per-kernel scores and ``S`` is a concave
function of *remaining gap to the SOL floor*, the marginal value of an hour differs
enormously between kernels.  ``allocate`` maximises the projected collection score
directly.

Improvement model (documented assumption, recalibrated from each campaign's measured
history): a kernel's distance to the SOL floor decays exponentially with effort, but
only down to a per-workload floor that one campaign cannot pass,

    t(h) = T_SOL + (t0 - T_SOL) * (floor_frac + (1 - floor_frac) * exp(-rate * h))

``rate`` is "fractional gap closed per GPU-hour" and ``floor_frac`` is the fraction of
the gap a single campaign is assumed unable to close (default ``DEFAULT_FLOOR_FRAC``).
Both are planning priors, replaced by measured values once a run has produced
latency-vs-hours observations.  The allocation *rule* is exact given the model; only
the model itself is an estimate, and the CLI can re-fit it from campaign logs.

The ``floor_frac`` term matters: without it the pure exponential drives every kernel
to the SOL floor and the projection becomes fantasy.  A bounded model keeps the plan
honest and still ranks kernels correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import objective

DEFAULT_RATE = 0.05            # 5% of the remaining gap to SOL closed per GPU-hour
DEFAULT_FLOOR_FRAC = 0.45      # fraction of the gap one campaign cannot close
DEFAULT_QUANTUM = 0.5          # hours per allocation slot


def predict_latency(
    t_ms: float,
    t_sol_ms: float,
    rate: float,
    hours: float,
    floor_frac: float = DEFAULT_FLOOR_FRAC,
) -> float:
    """Latency after ``hours`` at fractional-gap-close rate ``rate``, floored."""
    if hours < 0.0:
        raise ValueError("hours must be >= 0")
    if not (0.0 <= floor_frac <= 1.0):
        raise ValueError("floor_frac must be in [0, 1]")
    gap = max(0.0, t_ms - t_sol_ms)
    return t_sol_ms + gap * (floor_frac + (1.0 - floor_frac) * math.exp(-rate * hours))


@dataclass
class WorkloadState:
    """One workload's current latency and its improvement prior."""

    t_ms: float
    t_b_ms: float
    t_sol_ms: float
    rate: float = DEFAULT_RATE
    floor_frac: float = DEFAULT_FLOOR_FRAC
    name: str = ""

    def __post_init__(self) -> None:
        objective.validate_bounds(self.t_b_ms, self.t_sol_ms)
        if self.t_ms <= 0.0:
            raise ValueError("t_ms must be > 0")

    def latency_at(self, hours: float) -> float:
        return predict_latency(self.t_ms, self.t_sol_ms, self.rate, hours,
                               self.floor_frac)

    def score_at(self, hours: float) -> float:
        return objective.sol_score(self.latency_at(hours), self.t_b_ms, self.t_sol_ms)

    def is_fast_at(self, hours: float) -> bool:
        return objective.is_fast(self.latency_at(hours), self.t_b_ms)


@dataclass
class KernelState:
    """A kernel is a group of workloads scored by geometric mean."""

    name: str
    workloads: List[WorkloadState] = field(default_factory=list)

    def value_at(self, hours: float) -> float:
        if not self.workloads:
            raise ValueError(f"{self.name}: no workloads")
        logs = [math.log(max(self.workloads[i].score_at(hours), 1e-300))
                for i in range(len(self.workloads))]
        return math.exp(sum(logs) / len(logs))

    def fast_fraction_at(self, hours: float) -> float:
        fast = sum(1 for w in self.workloads if w.is_fast_at(hours))
        return fast / len(self.workloads)


@dataclass
class Allocation:
    """An hours assignment plus its projected effect."""

    hours: Dict[str, float]
    score_before: float
    score_after: float
    fast1_before: Tuple[int, int]
    fast1_after: Tuple[int, int]
    method: str
    total_hours: float

    @property
    def gain(self) -> float:
        return self.score_after - self.score_before

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "total_hours": self.total_hours,
            "score_before": round(self.score_before, 6),
            "score_after": round(self.score_after, 6),
            "score_gain": round(self.gain, 6),
            "fast1_before": list(self.fast1_before),
            "fast1_after": list(self.fast1_after),
            "hours": {k: round(v, 3) for k, v in sorted(self.hours.items())},
        }


# --------------------------------------------------------------------------- #
# Projection helpers
# --------------------------------------------------------------------------- #

def project_collection(
    kernels: Sequence[KernelState],
    hours: Mapping[str, float],
    fast1_weight: float = 0.0,
) -> Tuple[float, Tuple[int, int]]:
    """Projected (collection score, (fast, total)) under an hours assignment."""
    total_weighted = 0.0
    fast = 0
    total = 0
    for k in kernels:
        h = hours.get(k.name, 0.0)
        total_weighted += k.value_at(h) + fast1_weight * k.fast_fraction_at(h)
        f = sum(1 for w in k.workloads if w.is_fast_at(h))
        fast += f
        total += len(k.workloads)
    return total_weighted / len(kernels), (fast, total)


def current_collection(kernels: Sequence[KernelState]) -> Tuple[float, Tuple[int, int]]:
    return project_collection(kernels, {}, 0.0)


# --------------------------------------------------------------------------- #
# Allocation: exact DP and fast greedy
# --------------------------------------------------------------------------- #

def _value_tables(
    kernels: Sequence[KernelState],
    units: int,
    quantum: float,
    fast1_weight: float,
) -> List[List[float]]:
    tables: List[List[float]] = []
    for k in kernels:
        row = [k.value_at(u * quantum) + fast1_weight * k.fast_fraction_at(u * quantum)
               for u in range(units + 1)]
        tables.append(row)
    return tables


def _finalize(
    kernels: Sequence[KernelState],
    hours: Dict[str, float],
    method: str,
    total_hours: float,
    fast1_weight: float,
) -> Allocation:
    before_score, before_fast = current_collection(kernels)
    after_score, after_fast = project_collection(kernels, hours, fast1_weight)
    return Allocation(
        hours=hours,
        score_before=before_score,
        score_after=after_score,
        fast1_before=before_fast,
        fast1_after=after_fast,
        method=method,
        total_hours=total_hours,
    )


def allocate_dp(
    kernels: Sequence[KernelState],
    total_hours: float,
    quantum: float = DEFAULT_QUANTUM,
    fast1_weight: float = 0.0,
) -> Allocation:
    """Exact maximisation of the projected collection score over integer slots.

    Multiple-choice knapsack: each kernel contributes ``value_k[u]`` for ``u`` slots.
    ``O(N * U^2)``; ``U`` is small (a 500-hour budget at 0.5 h/slot is 1000 slots).
    """
    if not kernels:
        raise ValueError("no kernels")
    if quantum <= 0.0:
        raise ValueError("quantum must be > 0")
    units = int(math.floor(total_hours / quantum + 1e-9))
    if units <= 0:
        return _finalize(kernels, {}, "dp", total_hours, fast1_weight)
    tables = _value_tables(kernels, units, quantum, fast1_weight)

    neg = float("-inf")
    # dp[j] = best total value using exactly j slots
    dp = [neg] * (units + 1)
    dp[0] = 0.0
    choice: List[List[int]] = []  # per kernel, chosen slots for each j
    for k_idx in range(len(kernels)):
        new_dp = [neg] * (units + 1)
        pick = [-1] * (units + 1)
        row = tables[k_idx]
        for j in range(units + 1):
            best = neg
            best_u = -1
            # u slots given to this kernel; remaining j-u to previous kernels
            for u in range(j + 1):
                prev = dp[j - u]
                if prev == neg:
                    continue
                val = prev + row[u]
                if val > best:
                    best = val
                    best_u = u
            new_dp[j] = best
            pick[j] = best_u
        dp = new_dp
        choice.append(pick)

    # Backtrack from the best feasible total <= units.
    j = max(range(units + 1), key=lambda x: dp[x])
    hours: Dict[str, float] = {}
    remaining = j
    for k_idx in range(len(kernels) - 1, -1, -1):
        u = choice[k_idx][remaining]
        if u < 0:
            u = 0
        hours[kernels[k_idx].name] = u * quantum
        remaining -= u
    return _finalize(kernels, hours, "dp", total_hours, fast1_weight)


def allocate_greedy(
    kernels: Sequence[KernelState],
    total_hours: float,
    quantum: float = DEFAULT_QUANTUM,
    fast1_weight: float = 0.0,
) -> Allocation:
    """Marginal-greedy allocation: fast, and optimal for concave value tables.

    Repeatedly hands the next ``quantum`` to whichever kernel yields the largest
    immediate projected-score increase.  Use the DP when the value tables are not
    concave (it always wins or ties; see ``tests/test_allocator.py``).
    """
    if not kernels:
        raise ValueError("no kernels")
    units = int(math.floor(total_hours / quantum + 1e-9))
    hours = {k.name: 0.0 for k in kernels}
    if units <= 0:
        return _finalize(kernels, hours, "greedy", total_hours, fast1_weight)

    by_name = {k.name: k for k in kernels}
    # current value per kernel
    cur = [k.value_at(0.0) + fast1_weight * k.fast_fraction_at(0.0) for k in kernels]
    for _ in range(units):
        best_i = -1
        best_delta = float("-inf")
        for i, k in enumerate(kernels):
            h = hours[k.name] + quantum
            val = k.value_at(h) + fast1_weight * k.fast_fraction_at(h)
            delta = val - cur[i]
            if delta > best_delta:
                best_delta = delta
                best_i = i
        if best_i < 0 or best_delta <= 0.0:
            break
        hours[kernels[best_i].name] += quantum
        cur[best_i] += best_delta
    return _finalize(kernels, hours, "greedy", total_hours, fast1_weight)


def allocate(
    kernels: Sequence[KernelState],
    total_hours: float,
    quantum: float = DEFAULT_QUANTUM,
    fast1_weight: float = 0.0,
    method: str = "auto",
) -> Allocation:
    """Pick an allocation strategy.

    ``auto`` uses exact DP when the slot count is tractable, else greedy.
    """
    units = int(math.floor(total_hours / quantum + 1e-9))
    if method == "dp":
        return allocate_dp(kernels, total_hours, quantum, fast1_weight)
    if method == "greedy":
        return allocate_greedy(kernels, total_hours, quantum, fast1_weight)
    if method != "auto":
        raise ValueError(f"unknown method {method!r}")
    if units * units * max(1, len(kernels)) <= 50_000_000:
        return allocate_dp(kernels, total_hours, quantum, fast1_weight)
    return allocate_greedy(kernels, total_hours, quantum, fast1_weight)


# --------------------------------------------------------------------------- #
# Straightforward baselines to beat (used by the A/B in the mock campaign)
# --------------------------------------------------------------------------- #

def allocate_uniform(
    kernels: Sequence[KernelState],
    total_hours: float,
) -> Allocation:
    """Equal hours per kernel -- the naive strategy."""
    hours = {k.name: total_hours / len(kernels) for k in kernels}
    return _finalize(kernels, hours, "uniform", total_hours, 0.0)


def allocate_by_headroom(
    kernels: Sequence[KernelState],
    total_hours: float,
) -> Allocation:
    """Hours proportional to current headroom ``1 - S`` -- a smarter but still
    model-free heuristic, used as a stronger baseline in the A/B."""
    heads = {k.name: max(1e-6, 1.0 - k.value_at(0.0)) for k in kernels}
    total = sum(heads.values())
    hours = {name: total_hours * heads[name] / total for name in heads}
    return _finalize(kernels, hours, "headroom", total_hours, 0.0)
