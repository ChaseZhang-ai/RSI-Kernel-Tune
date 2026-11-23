"""Selection, stall detection, migration, refuge and retirement -- the pure rules.

``scripts/self_evolve.py`` is the regression baseline for ``--lineages 1``; the
rest of the file exercises task evolve §5.4's rules on scripted readings, with no
filesystem and no GPU.  The last test is the one the task file asks for by name:
one chain stops at a local optimum while three lineages walk past it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from forgekernel.evolve import metric as metric_mod
from forgekernel.evolve import migration
from forgekernel.evolve.metric import Metric
from forgekernel.evolve.migration import View, plan

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("self_evolve", REPO / "scripts" / "self_evolve.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)


def m(value, kind="speedup", floor=0.015, ms=None):
    return Metric(kind=kind, value=value, floor=floor, ms=ms)


def view(i, value, *, lang="cutedsl", prev=None, kind="speedup", floor=0.015,
         ref="self", adopted=False, fails=0, ms=None, prev_ms=None):
    cur = m(value, kind, floor, ms)
    before = m(prev, kind, floor, prev_ms) if prev is not None else None
    return View(index=i, language=lang, metric=cur, prev_metric=before,
                reference_kind=ref, adopted_global=adopted, fails=fails)


# -- regression: the old chain's numbers -------------------------------------------------


def test_noise_floor_matches_self_evolve():
    for best in ({"ratios": []}, {"ratios": [1.10, 1.101]},
                 {"ratios": [1.0, 1.05, 1.02]}, {"ratios": [1.2]}):
        assert metric_mod.noise_floor(best) == se.noise_floor(best)
    assert metric_mod.NOISE_FLOOR_MIN == se.NOISE_FLOOR_MIN


def test_one_lineage_names_match_the_old_chain():
    # the old script: the input directory for generation 1, <base>-g<k> after it
    from forgekernel.evolve.lineage import task_name
    assert task_name("rmsnorm", 0, 1, 1) == "rmsnorm"
    assert [task_name("rmsnorm", 0, g, 1) for g in range(2, 5)] == \
        ["rmsnorm-g2", "rmsnorm-g3", "rmsnorm-g4"]
    # a chain started from rmsnorm-g3 strips the suffix, exactly like the old script
    assert task_name("rmsnorm-g3", 0, 2, 1) == "rmsnorm-g2"


def test_legacy_stall_rule_matches_the_old_one():
    # 1.03 clears the 1.5% minimum but the ratios spread 4%: the old chain stops
    cur = m(1.03, floor=se.noise_floor({"ratios": [1.01, 1.03, 1.0504]}))
    v = View(index=0, language="cutedsl", metric=cur)
    p = plan([v], generation=1, legacy=True)
    assert p.for_index(0).action == migration.RETIRE
    # 1.30 clears it: the chain continues
    v2 = View(index=0, language="cutedsl", metric=m(1.30))
    assert plan([v2], generation=1, legacy=True).for_index(0).action == migration.RUN


def test_one_lineage_no_best_retires_and_would_exit_1():
    v = View(index=0, language="cutedsl", metric=None)
    p = plan([v], generation=1, legacy=True)
    assert p.for_index(0).action == migration.RETIRE
    assert p.all_retired([v])


# -- elites / stall / migration ----------------------------------------------------------


def test_elites_are_the_top_k_with_a_stable_tie_break():
    views = [view(0, 1.1), view(1, 1.3), view(2, 1.2), view(3, 1.0)]
    assert migration.select_elites(views, 2) == [1, 2]
    tied = [view(0, 1.2), view(1, 1.2), view(2, 1.2)]
    assert migration.select_elites(tied, 2) == [0, 1]
    assert migration.default_elites(3) == 1
    assert migration.default_elites(4) == 2


def test_stall_is_the_floor_on_the_lineage_s_own_gain():
    # speedup with a self reference: gain = value - 1
    assert view(0, 1.01).stalled()
    assert not view(0, 1.02).stalled()
    # sol: gain is score points, so the floor is absolute
    assert View(index=0, language="c", metric=m(0.60, "sol", 0.005),
                prev_metric=m(0.598, "sol", 0.005)).stalled()
    assert not View(index=0, language="c", metric=m(0.62, "sol", 0.005),
                    prev_metric=m(0.60, "sol", 0.005)).stalled()
    # an adopted reference compares absolute ms instead of the two ratios
    v = View(index=0, language="c", metric=m(1.0, ms=2.0), prev_metric=m(2.0, ms=2.1),
             reference_kind="foreign")
    assert v.gain() == pytest.approx(2.1 / 2.0 - 1.0)


def test_migration_fires_only_when_the_gap_beats_the_floor():
    views = [view(0, 1.30, ms=1.00), view(1, 1.00, ms=1.30),
             view(2, 1.00, ms=1.28)]
    p = plan(views, generation=2, elite_k=1, refuge=0, migration_from=2)
    assert p.best == 0
    assert p.for_index(1).action == migration.MIGRATE
    assert p.for_index(1).donor == 0
    assert p.for_index(2).action == migration.MIGRATE
    # ... and not when the laggard is within the floor of the best
    views = [view(0, 1.20, ms=1.000), view(1, 1.00, ms=1.005)]
    p = plan(views, generation=2, elite_k=1, refuge=0, migration_from=2)
    assert p.for_index(1).action == migration.RUN


def test_no_migration_before_migration_from():
    views = [view(0, 1.30, ms=1.0), view(1, 1.0, ms=1.3)]
    p = plan(views, generation=1, elite_k=1, refuge=0, migration_from=2)
    assert p.for_index(1).action == migration.RUN
    assert not p.migrations


def test_migration_needs_a_real_improvement_not_just_a_stall():
    # the laggard stalls and the best is only +0.5%: below the 1.5% floor
    views = [view(0, 1.005, ms=1.000), view(1, 1.0, ms=1.005)]
    p = plan(views, generation=2, elite_k=1, refuge=0, migration_from=2)
    assert p.for_index(1).action == migration.RUN


# -- refuge ------------------------------------------------------------------------------


def test_refuge_always_exists_and_prefers_a_different_language():
    views = [view(0, 1.40, lang="cutedsl", ms=1.00),
             view(1, 1.00, lang="cutedsl", ms=1.30),      # stalled, same language
             view(2, 1.00, lang="cuda", ms=1.35)]         # stalled, different language
    p = plan(views, generation=2, elite_k=1, refuge=1, migration_from=2)
    assert p.refuges == [2]
    assert p.for_index(2).action == migration.RETIRE   # a stalled refuge retires
    assert p.for_index(1).action == migration.MIGRATE


def test_a_healthy_refuge_never_migrates():
    views = [view(0, 1.40, lang="cutedsl", ms=1.00),
             view(1, 1.00, lang="cutedsl", ms=1.30),
             view(2, 1.10, lang="cuda", ms=1.20)]
    p = plan(views, generation=2, elite_k=1, refuge=1, migration_from=2)
    assert p.refuges == [2]
    assert p.for_index(2).action == migration.RUN
    assert p.for_index(1).action == migration.MIGRATE


def test_refuge_prefers_the_better_metric_when_languages_tie():
    views = [view(0, 1.40, lang="cuda", ms=1.00),
             view(1, 1.00, lang="cuda", ms=1.30),
             view(2, 1.05, lang="cuda", ms=1.20),
             view(3, 0.99, lang="cuda", ms=1.35)]
    p = plan(views, generation=2, elite_k=1, refuge=1, migration_from=2)
    assert p.refuges == [2]     # no different language: best metric among non-elite


# -- retirement / stops ------------------------------------------------------------------


def test_stalling_after_adopting_the_global_best_retires():
    a = view(0, 1.50, ms=1.00)
    adopted = view(1, 1.05, ms=1.10, prev=1.05, prev_ms=1.10,
                   ref="foreign", adopted=True)
    p = plan([a, adopted], generation=3, elite_k=1, refuge=0, migration_from=2)
    d = p.for_index(1)
    assert d.action == migration.RETIRE
    assert "adopting" in d.reason


def test_consecutive_failures_retire_at_max_fails():
    v = View(index=0, language="cutedsl", metric=None, fails=1)
    first = plan([v], generation=1, max_fails=2)
    assert first.for_index(0).action == migration.RETRY
    v2 = View(index=0, language="cutedsl", metric=None, fails=2)
    second = plan([v2], generation=2, max_fails=2)
    assert second.for_index(0).action == migration.RETIRE


def test_all_retired_is_a_stop():
    views = [View(index=0, language="cutedsl", metric=m(1.05)),
             View(index=1, language="cutedsl", metric=m(1.0))]
    views = [dataclasses.replace(v, retired=True) for v in views]
    p = plan(views, generation=2)
    assert p.decisions == []
    assert p.all_retired(views)


# -- the task file's named scenario ------------------------------------------------------


def _simulate(script, *, lineages, languages, generations, migration_from=2,
              elite_k=1, refuge=1):
    """Run the decision loop on scripted readings.

    ``script[(lineage, generation, start)]`` is ``(speedup, ms)`` that lineage's run
    would produce when the generation started from ``start`` (``"self"`` or a donor
    index); ``script[(lineage, generation)]`` covers a self start.  Speedup is the
    gain against that generation's reference; ``ms`` is the absolute latency, the
    only quantity comparable across lineages.  A missing entry means the lineage
    produced no confirmed best.

    Returns the best absolute median (ms) the population ever reached.
    """
    start = {i: "self" for i in range(lineages)}
    prev = {i: None for i in range(lineages)}
    retired = {i: False for i in range(lineages)}
    best_ms = float("inf")
    for gen in range(1, generations + 1):
        vals = {}
        for i in range(lineages):
            if retired[i]:
                continue
            key = (i, gen, start[i])
            if key in script:
                vals[i] = script[key]
            elif start[i] == "self":
                vals[i] = script.get((i, gen))
            else:
                vals[i] = None
        views = [View(index=i, language=languages[i],
                      metric=(m(vals[i][0], ms=vals[i][1])
                              if vals.get(i) is not None else None),
                      prev_metric=prev[i],
                      reference_kind=("foreign" if start[i] != "self" else "self"),
                      adopted_global=(start[i] != "self"), retired=retired[i])
                 for i in range(lineages)]
        p = plan(views, generation=gen, elite_k=elite_k, refuge=refuge,
                 migration_from=migration_from)
        for value in vals.values():
            if value is not None:
                best_ms = min(best_ms, value[1])
        for i in range(lineages):
            if retired[i]:
                continue
            d = p.for_index(i)
            if d.action == migration.RETIRE:
                retired[i] = True
            elif d.action == migration.MIGRATE:
                start[i] = d.donor
            prev[i] = views[i].metric
        if p.all_retired(views) or p.no_progress():
            break
    return best_ms


def test_one_chain_stops_at_a_local_optimum_while_three_walk_past_it():
    # Single chain: reaches 1.05x and then plateaus at 1.00x (ms 1.0).  For N=3,
    # l1's architecture is fastest (ms 0.99) while l2 (a different language) is the
    # refuge; l0 plateaus at ms 1.30, adopts l1 after generation 2 and reaches ms
    # 0.85.  The single chain's ceiling WAS the local optimum; the population walks
    # past it.
    single = _simulate({(0, 1): (1.05, 1.00), (0, 2): (1.00, 1.00),
                        (0, 3): (1.00, 1.00)},
                       lineages=1, languages=["cutedsl"], generations=4, refuge=0)
    assert single == pytest.approx(1.00)
    pop = _simulate({
        (0, 1): (1.02, 1.30), (0, 2): (1.00, 1.30),
        (1, 1): (1.30, 1.00), (1, 2): (1.01, 0.99), (1, 3): (1.00, 0.98),
        (1, 4, 0): (1.10, 0.80),
        (2, 1): (1.20, 1.10), (2, 2): (1.02, 1.08), (2, 3): (1.00, 1.08),
        (0, 3, 1): (1.15, 0.85), (0, 4, 1): (1.00, 0.85),
    }, lineages=3, languages=["cutedsl", "cutedsl", "cuda"],
        generations=5, migration_from=2, elite_k=1, refuge=1)
    assert pop < single
    assert pop == pytest.approx(0.80)
