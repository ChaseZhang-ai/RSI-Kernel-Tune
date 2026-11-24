import pytest

from argus.backends.mock import MockBackend, MockConfig, parse_code
from argus.cudaforge import RunConfig, run_argus_loop, run_cudaforge_loop
from argus.search import MECHANISMS, PopulationSearch, SearchConfig, greedy_chain

FAST = MockConfig(shapes=((64, 64), (64, 16)), seed=3, temptation=False)


def _be(strict=True, temptation=False):
    return MockBackend("t", MockConfig(shapes=((64, 64), (64, 16)), seed=3,
                                       strict_verification=strict, temptation=temptation))


def test_population_beats_greedy_equal_budget():
    greedy = run_cudaforge_loop(_be(), RunConfig(rounds=30))
    argus = run_argus_loop(_be(), RunConfig(rounds=40),
                           SearchConfig(n_lineages=4, mechanisms=list(MECHANISMS),
                                        max_evals=40, seed=3))
    assert argus.best_score >= greedy.best_score + 0.05


def test_argus_reaches_the_floor():
    be = _be()
    argus = run_argus_loop(be, RunConfig(rounds=40),
                           SearchConfig(n_lineages=4, mechanisms=list(MECHANISMS),
                                        max_evals=40, seed=3))
    assert argus.best_score == pytest.approx(be.optimum_score(), abs=1e-6)


def test_heads_are_monotone_regression_proof():
    be = _be()
    sc = SearchConfig(n_lineages=3, mechanisms=list(MECHANISMS), max_evals=45, seed=1)
    s = PopulationSearch(sc, be.seed, lambda c, m, l: be.propose(c, m, l, ""),
                         lambda c, f: be.eval_kernel(c, f))
    prev = {l: float("-inf") for l in range(sc.n_lineages)}
    while True:
        s.step()
        for l, h in s.heads.items():
            if h is not None:
                assert h.score >= prev[l] - 1e-12     # a head never regresses
                prev[l] = h.score
        if s.n_evals >= sc.max_evals:
            break


def test_greedy_parent_is_the_latest_candidate():
    be = _be()
    sr = greedy_chain(be.seed, lambda c, m, l: be.propose(c, m, l, ""),
                      lambda c, f: be.eval_kernel(c, f), mechanisms=list(MECHANISMS),
                      max_evals=10)
    for i in range(1, len(sr.candidates)):
        assert sr.candidates[i].parent == sr.candidates[i - 1].cid


def test_arm_filter_prevents_locking():
    be = _be()
    sc = SearchConfig(n_lineages=3, mechanisms=list(MECHANISMS), max_evals=45, seed=1,
                      arm_filter=lambda m, code: _prereq_ok(m, code))
    s = PopulationSearch(sc, be.seed, lambda c, m, l: be.propose(c, m, l, ""),
                         lambda c, f: be.eval_kernel(c, f))
    s.run()
    # no candidate applied a mechanism before its prerequisite -> nothing got locked
    for cand in s.candidates.values():
        assert len(parse_code(cand.code)["locked"]) == 0
    assert s.best is not None and s.best.score == pytest.approx(be.optimum_score(), abs=1e-6)


def _prereq_ok(mech, code):
    from argus import roofline as rl
    return rl.prereq_satisfied(mech, parse_code(code)["mech"])
