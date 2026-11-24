import math
import numpy as np
import pytest

from argus import objective as ob


def test_workload_score_anchors():
    assert ob.workload_score(0.9, 0.1, 0.9) == pytest.approx(0.5, abs=1e-9)   # at the baseline T_b
    assert ob.workload_score(0.1, 0.1, 0.9) == pytest.approx(1.0, abs=1e-9)   # at the roof T_SOL
    # below the roof is clamped (a measurement artefact cannot beat the roof)
    assert ob.workload_score(0.05, 0.1, 0.9) == pytest.approx(1.0, abs=1e-9)


def test_latency_for_score_roundtrip():
    for s in (0.5, 0.6, 0.75, 0.9, 0.95):
        t = ob.latency_for_score(s, 0.02, 0.2)
        assert ob.workload_score(t, 0.02, 0.2) == pytest.approx(s, abs=1e-9)


def test_kernel_and_collection_aggregation():
    assert ob.kernel_score([0.9, 0.9]) == pytest.approx(0.9)
    assert ob.kernel_score([1.0, 0.25]) == pytest.approx(0.5)        # geometric mean
    assert ob.collection_score([0.5, 1.0]) == pytest.approx(0.75)    # arithmetic mean


def test_score_is_convex_and_monotone_in_latency():
    ts, tb = 0.05, 0.5
    lats = [0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
    scores = [ob.workload_score(t, ts, tb) for t in lats]
    assert all(a >= b for a, b in zip(scores, scores[1:]))            # monotone decreasing
    # S is convex in latency: the |dS/dT| slope is steeper near the roof than far from it
    slope_near = (ob.workload_score(0.05, ts, tb) - ob.workload_score(0.15, ts, tb)) / 0.1
    slope_far = (ob.workload_score(0.40, ts, tb) - ob.workload_score(0.50, ts, tb)) / 0.1
    assert slope_near > slope_far


def test_fast1_and_speedup():
    assert ob.is_fast1(0.4, 0.5) and not ob.is_fast1(0.5, 0.5)
    assert ob.speedup_vs_baseline(0.25, 0.5) == pytest.approx(2.0)


def test_degenerate_anchors():
    assert ob.workload_score(0.1, 0.1, 0.1) == ob.SOL_BOUND_SCORE
    assert ob.workload_score(0.2, 0.1, 0.1) == ob.SCORING_BASELINE_SCORE
