import pytest

from argus import roofline as rl

PEAK = rl.PEAKS["B300"]


def _mem_bound_counters():
    # 8 GB in 1.6 ms -> 5 TB/s, ~62% of B300 peak, memory-bound
    return {
        "dram__bytes_read.sum": 8e9,
        "dram__bytes_write.sum": 0.0,
        "sm__cycles_active.avg": 1.6e-3 * PEAK.sm_clock,
        "sm__inst_executed.sum": 1e7,
    }


def test_counters_to_point_shape():
    pt = rl.counters_to_point(_mem_bound_counters(), PEAK, flops=1e9)
    assert 0.5 < pt.dram_pct < 0.8
    assert pt.time_s == pytest.approx(1.6e-3, rel=1e-6)
    assert pt.bottleneck == rl.MEMORY


def test_bottleneck_compute_vs_memory():
    mem = rl.counters_to_point(_mem_bound_counters(), PEAK, flops=1e9)
    # big flops, small bytes -> compute-bound
    comp_counters = {
        "dram__bytes_read.sum": 1e6, "dram__bytes_write.sum": 0.0,
        "sm__cycles_active.avg": 1e-4 * PEAK.sm_clock,
    }
    comp = rl.counters_to_point(comp_counters, PEAK, flops=1e12, compute_dtype="bf16")
    assert mem.bottleneck == rl.MEMORY
    assert comp.bottleneck == rl.COMPUTE


def test_mechanism_bounds_memory_ceiling():
    pt = rl.counters_to_point(_mem_bound_counters(), PEAK, flops=1e9)
    # vectorize can raise achieved BW toward peak, bounded
    vb = rl.mechanism_bounds("vectorize", pt, PEAK)
    assert vb.hi > 1.05 and vb.hi <= 8.0
    # a tensor core cannot move DRAM bytes faster on a memory-bound kernel
    tb = rl.mechanism_bounds("tensor_core", pt, PEAK)
    assert not tb.feasible and tb.hi == pytest.approx(1.0)
    # l2 tiling is feasible here
    assert rl.mechanism_bounds("l2_tiling", pt, PEAK).feasible


def test_prescreen_refuses_and_accepts():
    pt = rl.counters_to_point(_mem_bound_counters(), PEAK, flops=1e9)
    assert rl.prescreen("tensor_core", pt, PEAK).would_spend_eval is False
    assert rl.prescreen("vectorize", pt, PEAK).would_spend_eval is True


def test_rank_mechanisms_sorted():
    pt = rl.counters_to_point(_mem_bound_counters(), PEAK, flops=1e9)
    ranked = rl.rank_mechanisms(pt, PEAK, ["vectorize", "tensor_core", "l2_tiling"])
    assert [b.hi for b in ranked] == sorted([b.hi for b in ranked], reverse=True)


def test_prereqs():
    assert rl.prereq_satisfied("vectorize", {})
    assert not rl.prereq_satisfied("tensor_core", {"l2_tiling": 0})
    assert rl.prereq_satisfied("tensor_core", {"l2_tiling": 1})
    assert not rl.prereq_satisfied("warp_specialize", {"pipeline": 0})
