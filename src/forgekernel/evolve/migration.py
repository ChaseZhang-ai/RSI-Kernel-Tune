"""Selection and migration across a generation's lineages -- pure functions.

Task evolve §5.4, restated as the rules this module implements:

1. the global best is the active lineage with the highest metric (ties: the lower
   lineage index); the elites are the top ``--elite K`` (default
   ``max(1, ceil(N / 3))``);
2. a non-elite lineage that made no gain this generation (its own gain is at or
   below the noise floor) and whose metric is below the global best by more than
   the floor adopts the global best as the next generation's start (migration);
3. at least ``--refuge`` non-elite lineages never migrate -- the population always
   keeps an architecture that is not the global best.  Among the non-elite, a
   lineage with a *different language* from the global best is preferred; ties go
   to the better metric, then to the lower index.  A refuge that stalls retires;
4. a lineage retires when it stalls after having adopted the global best, or when
   it produced no confirmed best for ``--max-fails`` consecutive generations;
5. migration starts only after generation ``--migration-from G0`` (explore first).

Nothing here touches the filesystem: the population feeds it :class:`View` rows
and applies the returned :class:`Decision` rows.  That is what makes the rules
testable, and what lets the test reproduce "one chain stops at a local optimum,
three lineages walk past it" from scripted readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from forgekernel.evolve.metric import Metric, improvement, is_stalled, self_gain

RUN = "self"          # run again with this lineage's own best as the start
MIGRATE = "migrate"   # run with the global best as the start
RETRY = "retry"       # no confirmed best: run the same reference again
RETIRE = "retire"     # this lineage leaves the population
SKIP = "skip"         # already retired


@dataclass(frozen=True)
class View:
    """One active lineage at decision time."""

    index: int
    language: str
    metric: Metric | None = None
    prev_metric: Metric | None = None
    reference_kind: str = "self"   # "self" | "foreign" | "base" | "seed"
    adopted_global: bool = False   # its current reference IS a donor (migrated)
    fails: int = 0                 # consecutive generations without a best, THIS one included
    retired: bool = False
    variant: str = ""              # l<i>-g<gen>, for the record

    @property
    def active(self) -> bool:
        return not self.retired

    def gain(self, *, legacy: bool = False) -> float | None:
        return self_gain(self.metric, self.prev_metric,
                         reference_kind=self.reference_kind, legacy=legacy)

    def stalled(self, *, floor: float | None = None, legacy: bool = False) -> bool:
        if self.metric is None:
            return True
        return is_stalled(self.gain(legacy=legacy), self.metric, floor)


@dataclass(frozen=True)
class Decision:
    index: int
    action: str
    donor: int | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {"index": self.index, "action": self.action, "donor": self.donor,
                "reason": self.reason}


@dataclass
class Plan:
    decisions: list[Decision] = field(default_factory=list)
    elites: list[int] = field(default_factory=list)
    refuges: list[int] = field(default_factory=list)
    best: int | None = None
    stalled: list[int] = field(default_factory=list)

    def for_index(self, index: int) -> Decision:
        for d in self.decisions:
            if d.index == index:
                return d
        return Decision(index, SKIP, reason="not an active lineage")

    @property
    def migrations(self) -> list[Decision]:
        return [d for d in self.decisions if d.action == MIGRATE]

    @property
    def retirements(self) -> list[Decision]:
        return [d for d in self.decisions if d.action == RETIRE]

    def all_retired(self, views: list[View]) -> bool:
        return not any(v.active for v in views) or not any(
            d.action in (RUN, MIGRATE, RETRY) for d in self.decisions)

    def no_progress(self) -> bool:
        """Every active lineage stalled and nothing is scheduled to migrate: the
        next generation would repeat this one byte for byte."""
        if self.migrations:
            return False
        live = [d for d in self.decisions if d.action in (RUN, MIGRATE, RETRY)]
        return bool(live) and all(d.action == RUN for d in live) and bool(self.stalled) \
            and set(self.stalled) == {d.index for d in live}


def default_elites(n: int) -> int:
    return max(1, math.ceil(n / 3))


def ranked(views: list[View]) -> list[View]:
    """Active lineages best first.  The order is ``Metric.order``: an absolute
    quantity (SOL score, or the median latency behind a speedup) -- a speedup alone
    is against each lineage's own reference and not comparable.  A lineage without
    a reading sorts last; ties keep the lower lineage index."""
    active = [v for v in views if v.active]
    active.sort(key=lambda v: (v.metric is None,
                               -(v.metric.order if v.metric is not None else 0.0),
                               v.index))
    return active


def global_best(views: list[View]) -> View | None:
    order = ranked(views)
    return order[0] if order and order[0].metric is not None else None


def select_elites(views: list[View], k: int | None = None) -> list[int]:
    order = ranked(views)
    n = len([v for v in views if v.active])
    k = default_elites(n) if k is None else max(0, int(k))
    return [v.index for v in order[:k]]


def select_refuges(views: list[View], elites: list[int], count: int,
                   best: View | None) -> list[int]:
    """The non-elite lineages that must not migrate.

    Different language from the global best first, then the better metric, then the
    lower index; ``count`` of them, capped by the number of non-elite lineages.
    """
    non_elite = [v for v in ranked(views) if v.index not in set(elites)
                 and v.metric is not None]
    if not non_elite or count <= 0:
        return []

    def key(v: View):
        return (v.language == (best.language if best is not None else ""),
                -(v.metric.order if v.metric is not None else 0.0), v.index)

    return [v.index for v in sorted(non_elite, key=key)[:count]]


def plan(views: list[View], *, generation: int, elite_k: int | None = None,
         refuge: int = 1, migration_from: int = 2, floor: float | None = None,
         max_fails: int = 2, legacy: bool = False) -> Plan:
    """Decide, for every active lineage, what the next generation starts from.

    ``generation`` is the generation just finished: migration is allowed when
    ``generation >= migration_from`` (so ``--migration-from 2`` first migrates
    after generation 2, into generation 3).
    ``floor`` overrides every metric's own floor (the population leaves it None:
    ordinary tasks use the confirmation spread, board tasks ``--s-floor``).
    ``legacy`` is ``--lineages 1``: exactly ``scripts/self_evolve.py``'s rule
    (a gain that does not clear the floor ends the chain)."""
    active = [v for v in views if v.active]
    out = Plan(elites=select_elites(active, elite_k))
    best = global_best(active)
    out.best = best.index if best is not None else None
    out.refuges = select_refuges(active, out.elites, refuge, best) if not legacy else []
    allow_migration = (not legacy) and migration_from is not None \
        and generation >= migration_from

    for v in sorted(active, key=lambda x: x.index):
        if v.metric is None:
            fails = max(1, int(v.fails))
            if legacy or fails >= max(1, int(max_fails)):
                out.decisions.append(Decision(
                    v.index, RETIRE,
                    reason=f"no confirmed best for {fails} generation(s)"))
            else:
                out.decisions.append(Decision(
                    v.index, RETRY, reason="no confirmed best this generation"))
            continue
        gain = v.gain(legacy=legacy)
        stalled = is_stalled(gain, v.metric, floor)
        if stalled:
            out.stalled.append(v.index)
        if legacy:
            out.decisions.append(Decision(
                v.index, RUN if not stalled else RETIRE,
                reason=("gain clears the noise floor" if not stalled
                        else "gain within the noise floor: the chain stops")))
            continue
        if v.index in out.refuges:
            out.decisions.append(Decision(
                v.index, RETIRE if stalled else RUN,
                reason=("refuge lineage stalled: retired" if stalled
                        else "refuge lineage: keeps its own architecture")))
            continue
        if stalled and v.adopted_global:
            out.decisions.append(Decision(
                v.index, RETIRE,
                reason="stalled after adopting the global best"))
            continue
        if stalled and allow_migration and best is not None and best.index != v.index:
            gap = improvement(best.metric, v.metric)
            if gap > (v.metric.floor if floor is None else float(floor)):
                out.decisions.append(Decision(
                    v.index, MIGRATE, donor=best.index,
                    reason=f"stalled; global best is +{gap:.4g} ahead"))
                continue
        out.decisions.append(Decision(v.index, RUN, reason=(
            "improved" if not stalled else "stalled, but no donor is far enough ahead")))
    return out
