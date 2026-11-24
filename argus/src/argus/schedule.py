"""Lightweight evaluation-budget planner (ARGUS supporting component).

Not a contribution -- the sibling track owns the exact score-aware allocation.  ARGUS needs
only enough scheduling to spend a scarce evaluation budget sensibly: give more evaluations
to a task when (a) it has headroom on the board score and (b) the model says a mechanism is
still feasible there.  A marginal-greedy pass is plenty; the interesting ARGUS machinery is
*what a single evaluation is worth* (verification + attribution), not how to divide hours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TaskPriority:
    name: str
    score: float               # current board score of the task (0.5 baseline .. 1.0 roof)
    op_class: str = "generic"
    bottleneck: str = "balanced"
    max_feasible_gain: float = 1.0    # from roofline.prescreen, best mechanism bound


def priority_weight(t: TaskPriority) -> float:
    """Marginal value of one more evaluation on this task.

    ``1 - score`` is the remaining score headroom (equal weight per kernel on the board);
    ``max_feasible_gain - 1`` is how much the model thinks is left on the table.  Their
    product is a cheap, defensible proxy for the marginal eval value.
    """
    headroom = max(1.0 - t.score, 0.0)
    feasible = max(t.max_feasible_gain - 1.0, 0.0)
    return headroom * (1.0 + feasible)


def plan_eval_budget(
    tasks: List[TaskPriority],
    total_evals: int,
    *,
    min_evals: int = 2,
) -> Dict[str, int]:
    """Marginal-greedy division of ``total_evals`` across tasks by :func:`priority_weight`."""
    if not tasks:
        return {}
    alloc = {t.name: min_evals for t in tasks}
    remaining = total_evals - min_evals * len(tasks)
    if remaining <= 0:
        # trim proportionally to min_evals
        return _trim(alloc, total_evals)
    weights = [priority_weight(t) for t in tasks]
    wsum = sum(weights) or 1.0
    for t, w in zip(tasks, weights):
        alloc[t.name] += int(remaining * w / wsum)
    # hand out the rounding remainder to the highest-weight task
    used = sum(alloc.values())
    order = sorted(tasks, key=lambda t: priority_weight(t), reverse=True)
    i = 0
    while used < total_evals and order:
        alloc[order[i % len(order)].name] += 1
        used += 1
        i += 1
    return alloc


def _trim(alloc: Dict[str, int], total: int) -> Dict[str, int]:
    """Reduce a min-eval allocation to fit ``total`` (drop the smallest-value tasks)."""
    names = list(alloc)
    out = {n: 0 for n in names}
    for n in names[:total]:
        out[n] = 1
    return out
