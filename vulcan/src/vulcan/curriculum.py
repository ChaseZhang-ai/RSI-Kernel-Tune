"""Campaign ordering: which kernel to attack first, and in what company.

Two effects decide the order, and they pull in different directions:

* **Value** -- with an unweighted-per-kernel mean and concavity near the floor, the
  next hour is worth most on a specific kernel.  ``allocator`` already computes that
  marginal order; front-loading it is a "quick win" ordering.
* **Knowledge transfer** -- mechanisms certified on one kernel are reused by its
  family (rmsnorm -> rmsnorm, gemm -> gemm, gqa decode -> gqa prefill).  Solving a
  family's easy members first seeds the knowledge graph for its hard members.

``hybrid_order`` interleaves them: take the best value kernel, then the easiest
unseen family, and continue, so the campaign both scores early and keeps learning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import allocator
from .allocator import KernelState


_PREFIX_RE = re.compile(r"^\d+_")
# shape tokens are glued to their label: n128, k2048, h32, kv8, d128, ps1, e32, ng8, ...
_SHAPE_TOKEN_RE = re.compile(r"^(?:n|k|m|h|kv|d|ps|e|i|ng|kg|topk)\d+$")


def family_of(name: str) -> str:
    """Coarse operator family used for knowledge transfer.

    ``013_gqa_paged_decode_h32_kv8_d128_ps1`` -> ``gqa_paged_decode``;
    ``004_gemm_n128_k2048`` -> ``gemm``; ``001_fused_add_rmsnorm_h2048`` ->
    ``fused_add_rmsnorm``.
    """
    base = _PREFIX_RE.sub("", name)
    parts = base.split("_")
    while len(parts) > 1 and (_SHAPE_TOKEN_RE.match(parts[-1]) or parts[-1].isdigit()):
        parts.pop()
    return "_".join(parts) if parts else base


@dataclass
class TaskSpec:
    """A kernel to optimize, plus the architecture hypotheses to try."""

    name: str
    state: KernelState
    architectures: List[str] = field(default_factory=list)
    op_class: str = "attention"

    @property
    def family(self) -> str:
        return family_of(self.name)

    def current_score(self) -> float:
        return self.state.value_at(0.0)

    def fast1_gap(self) -> int:
        return sum(1 for w in self.state.workloads if not w.is_fast_at(0.0))


def difficulty(spec: TaskSpec) -> float:
    """Higher == harder/more expensive.  Headroom, shape count and stragglers."""
    headroom = 1.0 - spec.current_score()
    n = len(spec.state.workloads)
    return headroom * (1.0 + 0.05 * n) * (1.0 + 0.1 * spec.fast1_gap())


def value_order(specs: Sequence[TaskSpec], step_hours: float = 1.0) -> List[str]:
    """Order kernels by decreasing marginal score gain of one more ``step_hours``."""
    kernels = [s.state for s in specs]
    gains: List[Tuple[float, str]] = []
    for s in specs:
        now = s.state.value_at(0.0)
        later = s.state.value_at(step_hours)
        gains.append((later - now, s.name))
    gains.sort(reverse=True)
    return [name for _, name in gains]


def curriculum_order(specs: Sequence[TaskSpec]) -> List[str]:
    """Easiest first, but never start a family's hard member before an easy one."""
    by_family: Dict[str, List[TaskSpec]] = {}
    for s in specs:
        by_family.setdefault(s.family, []).append(s)
    out: List[str] = []
    # order families by their easiest member's difficulty
    fams = sorted(by_family.items(),
                  key=lambda kv: min(difficulty(s) for s in kv[1]))
    for _, members in fams:
        for s in sorted(members, key=difficulty):
            out.append(s.name)
    return out


def hybrid_order(specs: Sequence[TaskSpec], step_hours: float = 1.0) -> List[str]:
    """Interleave best-value and easiest-per-family picks."""
    remaining = {s.name: s for s in specs}
    val = value_order(specs, step_hours)
    cur = curriculum_order(specs)
    val_rank = {n: i for i, n in enumerate(val)}
    cur_rank = {n: i for i, n in enumerate(cur)}
    out: List[str] = []
    while remaining:
        # pick the candidate minimising the sum of ranks (both criteria matter)
        choice = min(remaining,
                     key=lambda n: val_rank[n] + cur_rank[n])
        out.append(choice)
        remaining.pop(choice)
    return out


def order(
    specs: Sequence[TaskSpec],
    strategy: str = "hybrid",
    step_hours: float = 1.0,
) -> List[TaskSpec]:
    if strategy == "value":
        names = value_order(specs, step_hours)
    elif strategy == "curriculum":
        names = curriculum_order(specs)
    elif strategy == "hybrid":
        names = hybrid_order(specs, step_hours)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    by_name = {s.name: s for s in specs}
    return [by_name[n] for n in names]


def family_seed_plan(specs: Sequence[TaskSpec]) -> List[Tuple[str, List[str]]]:
    """For each family, the order in which to attempt it (easiest first)."""
    by_family: Dict[str, List[TaskSpec]] = {}
    for s in specs:
        by_family.setdefault(s.family, []).append(s)
    plan = []
    for fam, members in sorted(by_family.items(),
                               key=lambda kv: min(difficulty(s) for s in kv[1])):
        plan.append((fam, [s.name for s in sorted(members, key=difficulty)]))
    return plan
