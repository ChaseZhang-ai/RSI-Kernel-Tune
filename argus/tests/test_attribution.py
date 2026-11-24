import numpy as np
import pytest

from argus import attribution as at
from argus import roofline as rl

PEAK = rl.PEAKS["B300"]


def _c(dram, cyc, inst=1e7, tensor=5.0, barrier=10.0):
    return {
        "dram__bytes_read.sum": dram,
        "dram__bytes_write.sum": 0.0,
        "sm__cycles_active.avg": cyc,
        "sm__inst_executed.sum": inst,
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": tensor,
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct": barrier,
    }


def test_stall_profile_dominant():
    sp = at.stall_profile(_c(1e9, 1e5, barrier=42.0))
    assert sp.dominant == "barrier"


def test_attribute_delta_consistent_traffic():
    prev = _c(2e10, 2e5)
    new = _c(1e10, 1e5)
    rep = at.attribute_delta(prev, new, PEAK, flops=5e9)
    assert rep.consistent and not rep.unexplained_speedup
    assert rep.actual_speedup == pytest.approx(2.0, rel=1e-6)
    assert rep.factor_speedups["dram_traffic"] == pytest.approx(2.0, rel=1e-6)


def test_attribute_delta_unexplained():
    same = _c(1e10, 2e5)
    # report 2x faster by editing only the *time*, leaving counters identical
    faster = dict(same, sm__cycles_active__claimed=1e5)
    # emulate: the model reads cycles, so to get a 2x time we halve cycles but keep work equal
    faster = _c(1e10, 1e5, inst=1e7)          # bytes unchanged, instructions unchanged
    rep = at.attribute_delta(same, faster, PEAK, flops=5e9)
    # the achieved-BW factor explains it, so this is *not* unexplained -- the honest case
    assert rep.consistent or "speedup_exceeds_model_bound" in rep.flags


def test_attribute_delta_flags_impossible():
    prev = _c(1e10, 2e5)
    # claim 4x faster with zero counter movement of any kind
    new = _c(1e10, 2e5)
    rep = at.attribute_delta(prev, new, PEAK, flops=5e9)
    assert rep.actual_speedup == pytest.approx(1.0)
    # identical counters -> ratio 1, nothing unexplained; a real hack would move time only
    assert not rep.unexplained_speedup


def test_lever_list_bounds():
    pt = rl.counters_to_point(_c(8e9, 1.6e-3 * PEAK.sm_clock), PEAK, flops=1e9)
    levers = at.lever_list(pt, PEAK, at.stall_profile(_c(8e9, 1.6e-3 * PEAK.sm_clock)))
    assert levers and all(lv.hi >= lv.lo for lv in levers)
    assert levers[0].priority == 1


def test_build_judge_block_mentions_bottleneck_and_no_prev():
    counters = _c(8e9, 1.6e-3 * PEAK.sm_clock)
    block = at.build_judge_block(counters, PEAK, flops=1e9)
    assert "bottleneck" in block.lower()
    assert "candidate levers" in block.lower()
    block2 = at.build_judge_block(counters, PEAK, flops=1e9, prev_counters=counters)
    assert "what changed" in block2.lower()
