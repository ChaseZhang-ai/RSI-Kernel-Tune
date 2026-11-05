"""Checkpoint invalidation for prefix reuse (design doc §6.3).

The inner student loop runs an ordered trajectory of decisions, each producing a
checkpoint. When the outer meta loop patches a milestone (changing some set of
entries -- decision ids, params), only the decisions that *depend* on a
changed entry become stale. The earliest stale decision and everything after it
are invalidated; the inner loop resumes from the checkpoint just before it.

This makes the cost asymmetry in the design explicit:
  * additive patch (new entry nothing depends on yet) -> nothing invalidated;
  * foundation patch (entry the first decision depends on) -> full restart.
Foundation patches are cheap only while the trajectory is still short, which is
why the meta loop front-loads foundation work (design doc §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Decision:
    """One inner-loop decision.

    id:         the milestone entry this decision implements (e.g. "D2.3").
    depends_on: other milestone entries it relies on (e.g. {"D1.1", "D1.2"}).
    """

    id: str
    depends_on: frozenset[str] = field(default_factory=frozenset)
    # ⊗ companions: decisions that must land in the same commit. Not DAG edges —
    # kept beside the dependency set so a resume plan can name the group. The
    # reopen closure reads them symmetrically (a companion named by a reopened
    # decision joins even when it declared no edge back): a pair split across a
    # reopen would leave one side validating a tree the other side rewrites.
    companions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Accept any iterable for ergonomics; store as frozenset.
        if not isinstance(self.depends_on, frozenset):
            object.__setattr__(self, "depends_on", frozenset(self.depends_on))


@dataclass(frozen=True)
class ResumePlan:
    """Result of a patch.

    resume_after: id of the last decision to keep, or None to restart from
                  baseline (nothing reusable).
    invalidated:  ids of decisions that must be re-run, in trajectory order.
    """

    resume_after: str | None
    invalidated: list[str]


def compute_resume(
    trajectory: list[Decision], changed_entries: Iterable[str]
) -> ResumePlan:
    """Compute the resume plan for a milestone patch.

    A decision is affected when the patch touches its own id or any entry it
    depends on. The earliest affected decision determines the resume point.
    """
    changed = frozenset(changed_entries)
    for i, decision in enumerate(trajectory):
        if decision.id in changed or (decision.depends_on & changed):
            resume_after = trajectory[i - 1].id if i > 0 else None
            invalidated = [d.id for d in trajectory[i:]]
            return ResumePlan(resume_after=resume_after, invalidated=invalidated)

    tip = trajectory[-1].id if trajectory else None
    return ResumePlan(resume_after=tip, invalidated=[])


@dataclass(frozen=True)
class ReopenPlan:
    """Result of a patch under in-place reopening.

    reopen: decisions to redo ON THE CURRENT TREE, in trajectory order — the
            dependency closure of the changed entries (a changed decision, every
            decision that transitively depends on one, and every ⊗ companion of
            any of those — read in both declaration directions, since companions
            land in one commit whoever named the pair).
    kept:   decisions that stay as built (not in the closure), trajectory order.
    """

    reopen: list[str]
    kept: list[str]


def compute_reopen(
    trajectory: list[Decision], changed_entries: Iterable[str]
) -> ReopenPlan:
    """The dependency closure of a patch — what must be redone, nothing more.

    ``compute_resume`` invalidates the earliest affected decision AND EVERY
    LATER ONE, dependent or not, because prefix reuse can only cut the
    trajectory at one point. That turned a one-decision foundation patch into
    a whole-segment rebuild (a sparse-attention run spent six of ten meta steps
    rebuilding M0). Reopening in place needs the exact set instead: the tree
    stays, the closure is redone on it, the rest is kept.
    """
    changed = frozenset(changed_entries)
    reopen: set[str] = set()
    # Companion edges are symmetric (companions land in one commit), but the ⊗
    # declaration may be written on one side only: when a reopened decision names
    # a companion, that companion is pulled in even if it never named the edge
    # back — one side of the pair kept would validate a tree the other rewrites.
    companions: dict[str, set[str]] = {}
    for d in trajectory:
        for c in d.companions:
            companions.setdefault(d.id, set()).add(c)
            companions.setdefault(c, set()).add(d.id)
    # Fixpoint over the trajectory: dependencies are declared by id and may
    # point at decisions later in the list (companions especially), so one
    # ordered pass is not enough.
    while True:
        hot = changed | reopen
        grew = False
        for d in trajectory:
            if d.id in reopen:
                continue
            if (d.id in changed or (d.depends_on & changed)
                    or (d.depends_on & reopen) or (companions.get(d.id, set()) & reopen)):
                reopen.add(d.id)
                grew = True
        if not grew:
            break
    return ReopenPlan(reopen=[d.id for d in trajectory if d.id in reopen],
                      kept=[d.id for d in trajectory if d.id not in reopen])
