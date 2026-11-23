"""Campaign supervisor: the closed loop that spends a budget and reports the score.

It composes the pieces:

1. ``allocator``    decides how many GPU-hours each kernel gets;
2. ``curriculum``   decides the order kernels are attempted in;
3. ``portfolio``    runs successive halving over each kernel's architecture hypotheses;
4. ``stats``        decides whether a gain is real (promotion / revert);
5. ``kb_graph``     (optional) records certified/refuted architecture outcomes.

The backend is injected, so the exact same loop runs against the synthetic GPU on a
laptop (``backends.mock``) and against ``forge`` on the B300 (``backends.forge``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import allocator, curriculum, objective, portfolio
from .allocator import KernelState, WorkloadState
from .backends.base import CampaignBackend, EvalResult
from .curriculum import TaskSpec


def reference_latency(state: KernelState) -> float:
    """Geometric mean of the kernel's current workload latencies (a scalar anchor)."""
    logs = [math.log(max(w.t_ms, 1e-12)) for w in state.workloads]
    return math.exp(sum(logs) / len(logs))


def scale_state(state: KernelState, ratio: float) -> KernelState:
    """A copy of ``state`` with every latency multiplied by ``ratio`` (floored at SOL)."""
    new_ws = []
    for w in state.workloads:
        t = max(w.t_sol_ms, w.t_ms * ratio)
        new_ws.append(WorkloadState(t_ms=t, t_b_ms=w.t_b_ms, t_sol_ms=w.t_sol_ms,
                                    rate=w.rate, floor_frac=w.floor_frac, name=w.name))
    return KernelState(name=state.name, workloads=new_ws)


def fast1_counts(state: KernelState) -> Tuple[int, int]:
    fast = sum(1 for w in state.workloads if w.is_fast_at(0.0))
    return fast, len(state.workloads)


@dataclass
class TaskOutcome:
    name: str
    architecture: str
    hours: float
    correct: bool
    score_before: float
    score_after: float
    latency_before_ms: float
    latency_after_ms: float
    fast1_before: Tuple[int, int]
    fast1_after: Tuple[int, int]
    promoted: bool
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "kernel": self.name,
            "architecture": self.architecture,
            "hours": round(self.hours, 3),
            "correct": self.correct,
            "score_before": round(self.score_before, 5),
            "score_after": round(self.score_after, 5),
            "score_delta": round(self.score_after - self.score_before, 5),
            "fast1_before": list(self.fast1_before),
            "fast1_after": list(self.fast1_after),
            "promoted": self.promoted,
            "notes": self.notes,
        }


@dataclass
class CampaignReport:
    strategy: str
    total_hours: float
    score_before: float
    score_after: float
    fast1_before: Tuple[int, int]
    fast1_after: Tuple[int, int]
    outcomes: List[TaskOutcome] = field(default_factory=list)
    allocation: Dict[str, float] = field(default_factory=dict)
    backend_calls: int = 0

    @property
    def gain(self) -> float:
        return self.score_after - self.score_before

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_hours": self.total_hours,
            "score_before": round(self.score_before, 6),
            "score_after": round(self.score_after, 6),
            "score_gain": round(self.gain, 6),
            "fast1_before": list(self.fast1_before),
            "fast1_after": list(self.fast1_after),
            "backend_calls": self.backend_calls,
            "allocation": {k: round(v, 3) for k, v in sorted(self.allocation.items())},
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def _aggregate(states: Sequence[KernelState]) -> Tuple[float, Tuple[int, int]]:
    if not states:
        return 0.0, (0, 0)
    scores = [s.value_at(0.0) for s in states]
    fast = sum(1 for s in states for w in s.workloads if w.is_fast_at(0.0))
    total = sum(len(s.workloads) for s in states)
    return sum(scores) / len(scores), (fast, total)


class Supervisor:
    """Runs one campaign over a set of tasks within a GPU-hour budget."""

    def __init__(
        self,
        backend: CampaignBackend,
        strategy: str = "vulcan",
        quantum: float = 0.5,
        portfolio_budgets: Sequence[float] = (1.0, 3.0, 7.0),
        eta: int = 2,
        fast1_weight: float = 0.0,
        order_strategy: str = "hybrid",
        min_margin: float = 0.0,
    ):
        self.backend = backend
        self.strategy = strategy
        self.quantum = quantum
        self.portfolio_budgets = list(portfolio_budgets)
        self.eta = eta
        self.fast1_weight = fast1_weight
        self.order_strategy = order_strategy
        self.min_margin = min_margin

    # -- helpers ----------------------------------------------------------- #
    def _eval_arch(self, spec: TaskSpec, arch: str, hours: float) -> Tuple[float, EvalResult]:
        ref = reference_latency(spec.state)
        res = self.backend.evaluate(spec.name, ref, arch, hours)
        if not res.correct or not math.isfinite(res.latency_ms):
            return 0.0, res
        ratio = res.latency_ms / ref if ref > 0 else 1.0
        scaled = scale_state(spec.state, ratio)
        return scaled.value_at(0.0), res

    # -- strategies -------------------------------------------------------- #
    def _hours_per_kernel(self, specs: Sequence[TaskSpec], total_hours: float) -> Dict[str, float]:
        states = [s.state for s in specs]
        if self.strategy in ("vulcan", "vulcan_alloc"):
            return allocator.allocate(states, total_hours, self.quantum,
                                      self.fast1_weight, method="auto").hours
        if self.strategy in ("headroom", "vulcan_portfolio"):
            if self.strategy == "vulcan_portfolio":
                return allocator.allocate_uniform(states, total_hours).hours
            return allocator.allocate_by_headroom(states, total_hours).hours
        if self.strategy == "uniform":
            return allocator.allocate_uniform(states, total_hours).hours
        raise ValueError(f"unknown strategy {self.strategy!r}")

    def _single_arch(self, spec: TaskSpec) -> str:
        return spec.architectures[0] if spec.architectures else f"{spec.name}/default"

    def run(self, specs: Sequence[TaskSpec], total_hours: float) -> CampaignReport:
        if not specs:
            raise ValueError("no specs")
        score_before, fast_before = _aggregate([s.state for s in specs])
        hours = self._hours_per_kernel(specs, total_hours)

        ordered = curriculum.order(specs, self.order_strategy, step_hours=self.quantum)
        outcomes: List[TaskOutcome] = []

        for spec in ordered:
            h = hours.get(spec.name, 0.0)
            before = spec.state.value_at(0.0)
            before_lat = reference_latency(spec.state)
            fb = fast1_counts(spec.state)
            if h <= 0.0:
                outcomes.append(TaskOutcome(spec.name, "", 0.0, True, before, before,
                                            before_lat, before_lat, fb, fb, False,
                                            "no budget"))
                continue

            if self.strategy in ("vulcan", "vulcan_portfolio") and len(spec.architectures) > 1:
                budgets = [min(b, h) for b in self.portfolio_budgets]
                if budgets[-1] < h:
                    budgets.append(h)
                budgets = sorted(set(round(b, 6) for b in budgets))

                def evaluate(arch: str, hrs: float, _spec=spec) -> float:
                    val, _ = self._eval_arch(_spec, arch, hrs)
                    return val

                halving = portfolio.successive_halving(spec.architectures, evaluate,
                                                       budgets, self.eta)
                arch = halving.best.name
            else:
                arch = self._single_arch(spec)
                budgets = [h]

            val, res = self._eval_arch(spec, arch, h)
            ref = reference_latency(spec.state)
            if res.correct and math.isfinite(res.latency_ms):
                ratio = res.latency_ms / ref
                after_state = scale_state(spec.state, ratio)
                after = after_state.value_at(0.0)
                after_lat = res.latency_ms
                fa = fast1_counts(after_state)
            else:
                after, after_lat, fa = before, before_lat, fb

            promoted = after > before + max(1e-12, self.min_margin * before)
            if not promoted:
                after, after_lat, fa = before, before_lat, fb
                notes = "reverted (no significant gain)" if res.correct else "invalid kernel"
            else:
                notes = "promoted"
            outcomes.append(TaskOutcome(
                spec.name, arch, h, res.correct, before, after,
                before_lat, after_lat, fb, fa, promoted, notes,
            ))
            # advance the in-memory reference so later kernels see the champion
            if promoted:
                spec.state = scale_state(spec.state, after_lat / before_lat)

        score_after, fast_after = _aggregate([s.state for s in specs])
        calls = getattr(self.backend, "calls", 0)
        return CampaignReport(
            strategy=self.strategy,
            total_hours=total_hours,
            score_before=score_before,
            score_after=score_after,
            fast1_before=fast_before,
            fast1_after=fast_after,
            outcomes=outcomes,
            allocation=hours,
            backend_calls=calls,
        )
