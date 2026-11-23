"""Architecture portfolio search: spend the first hours on the *right* M0.

ForgeKernel's own stated limitation is that optimization can settle into a local
optimum because the architecture (work partition, dataflow, pipeline) is fixed early
and later rounds only tune inside it.  A single chain of self-evolution inherits that
risk.

VULCAN counters it with a *portfolio*: the architecture agent proposes ``K`` genuinely
different M0 architectures (not ``K`` seeds of one), each is scored cheaply, and the
budget concentrates on the survivors via successive halving.  This is the classic
best-arm-identification setting, and successive halving is near-optimal for it.

All functions are deterministic given ``seed`` so the choice is reproducible on the
GPU box.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

EvaluateFn = Callable[[str, float], float]   # (arm_name, cumulative_hours) -> score


@dataclass
class ArmRecord:
    name: str
    architecture: str = ""
    score: float = float("-inf")
    hours: float = 0.0
    notes: str = ""


@dataclass
class HalvingResult:
    best: ArmRecord
    survivors: List[str]
    history: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "best": self.best.name,
            "best_score": round(self.best.score, 6) if math.isfinite(self.best.score) else None,
            "survivors": self.survivors,
            "history": self.history,
        }


def successive_halving(
    arms: Sequence[str],
    evaluate: EvaluateFn,
    budgets: Sequence[float],
    eta: int = 3,
    maximize: bool = True,
) -> HalvingResult:
    """Successive halving over architecture arms.

    ``budgets[r]`` is the cumulative hours each surviving arm gets in round ``r``.
    After each round the worst ``1 - 1/eta`` fraction is dropped; the last survivor
    is returned.  ``evaluate(arm, hours)`` must be deterministic (or seeded
    internally) so re-scoring at a larger budget is fair.
    """
    if not arms:
        raise ValueError("no arms")
    if not budgets or any(b < 0 for b in budgets):
        raise ValueError("budgets must be a non-empty non-negative sequence")
    if eta < 2:
        raise ValueError("eta must be >= 2")

    survivors = list(arms)
    history: List[Dict[str, object]] = []
    records: Dict[str, ArmRecord] = {a: ArmRecord(name=a) for a in arms}
    sign = 1.0 if maximize else -1.0

    for r, budget in enumerate(budgets):
        if len(survivors) == 1 and r > 0:
            # still give the winner its full budget to record a real score
            pass
        scored: List[Tuple[float, str]] = []
        for a in survivors:
            s = evaluate(a, budget)
            records[a].score = s
            records[a].hours = budget
            scored.append((s, a))
        scored.sort(key=lambda t: sign * t[0], reverse=True)
        keep = max(1, math.ceil(len(survivors) / eta))
        survivors = [a for _, a in scored[:keep]]
        history.append({
            "round": r,
            "budget_hours": budget,
            "scored": {a: round(s, 6) for s, a in scored},
            "kept": survivors,
        })

        if len(survivors) == 1:
            break

    best_name = survivors[0]
    # ensure the winner is evaluated at the final budget if halving stopped early
    final_budget = budgets[-1]
    if records[best_name].hours < final_budget:
        records[best_name].score = evaluate(best_name, final_budget)
        records[best_name].hours = final_budget
    return HalvingResult(records[best_name], survivors, history)


class UCB1:
    """Upper-confidence-bound bandit over arms (reward in any bounded range)."""

    def __init__(self, arms: Sequence[str], c: float = 1.0):
        if not arms:
            raise ValueError("no arms")
        self.arms = list(arms)
        self.c = c
        self.pulls: Dict[str, int] = {a: 0 for a in arms}
        self.means: Dict[str, float] = {a: 0.0 for a in arms}
        self.reward_min = float("inf")
        self.reward_max = float("-inf")
        self.t = 0

    def _range(self) -> float:
        if not math.isfinite(self.reward_min) or self.reward_max <= self.reward_min:
            return 1.0
        return self.reward_max - self.reward_min

    def select(self) -> str:
        self.t += 1
        for a in self.arms:
            if self.pulls[a] == 0:
                return a
        rng = self._range()
        best_a, best_v = self.arms[0], float("-inf")
        for a in self.arms:
            mean = self.means[a]
            bonus = self.c * rng * math.sqrt(math.log(self.t) / self.pulls[a])
            v = mean + bonus
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.pulls:
            raise KeyError(arm)
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)
        n = self.pulls[arm] + 1
        self.means[arm] = self.means[arm] + (reward - self.means[arm]) / n
        self.pulls[arm] = n


def ucb_identify_best(
    arms: Sequence[str],
    evaluate: Callable[[str], float],
    total_pulls: int,
    c: float = 1.0,
) -> Tuple[str, Dict[str, int]]:
    """Use UCB1 to identify the best arm within ``total_pulls`` evaluations."""
    bandit = UCB1(arms, c=c)
    for _ in range(total_pulls):
        a = bandit.select()
        bandit.update(a, evaluate(a))
    best = max(arms, key=lambda a: (bandit.means[a], -bandit.pulls[a]))
    return best, dict(bandit.pulls)


def architecture_hypotheses(base_name: str) -> List[Tuple[str, str]]:
    """A starter set of *structurally different* M0 hypotheses for a kernel.

    These are prompts to the architecture agent, not fixed designs: the point is to
    force diversity in work partition / dataflow / pipeline rather than parameter
    jitter within one design.  Each is a name plus a one-line brief.
    """
    return [
        (f"{base_name}/cta-per-tile",
         "Classic: one work tile per CTA, no cross-item persistence."),
        (f"{base_name}/persistent-cta",
         "Persistent CTAs; each processes many work items; prefetch next item's inputs."),
        (f"{base_name}/warp-specialized",
         "Producer/consumer warp roles; async copy staged through smem; MMA warps never stall."),
        (f"{base_name}/split-k",
         "Split the reduction (K/seq) across CTAs with a deterministic combine pass."),
        (f"{base_name}/register-resident",
         "Keep the working set in registers/RS operands; avoid smem round-trips where legal."),
        (f"{base_name}/fused-epilogue",
         "Absorb the neighbouring elementwise/quantize/rope work into the main kernel."),
    ]
