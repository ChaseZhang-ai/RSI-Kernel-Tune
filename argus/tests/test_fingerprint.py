import pytest

from argus import fingerprint as fp
from argus import roofline as rl

PEAK = rl.PEAKS["B300"]


def _rec(t_ms, dram=1e9, inst=1e7, tensor=5.0, write=1e9):
    return {
        "time_s": t_ms / 1e3,
        "dram__bytes_read.sum": dram,
        "dram__bytes_write.sum": write,
        "sm__inst_executed.sum": inst,
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": tensor,
    }


def test_no_anomaly_for_honest_improvement():
    # time down 20% *and* bytes down 20% -> explained
    recs = [_rec(1.0), _rec(0.8, dram=0.8e9, write=0.8e9)]
    outs = fp.fingerprint_audit(recs, PEAK, flops=1e9)
    assert all(o.passed for o in outs if o.severity == "hard")


def test_unexplained_time_flagged():
    # time down 30% but every counter identical -> hard flag
    recs = [_rec(1.0), _rec(0.7)]
    outs = fp.fingerprint_audit(recs, PEAK, flops=1e9)
    assert any("unexplained_time" in o.name and not o.passed for o in outs)


def test_store_skip_flagged():
    recs = [_rec(1.0), _rec(0.6, write=0.1e9)]
    outs = fp.fingerprint_audit(recs, PEAK, flops=1e9)
    assert any("store_skip" in o.name and not o.passed for o in outs)


def test_hack_risk_score_monotone():
    clean = fp.hack_risk_score([_rec(1.0), _rec(0.8, dram=0.8e9, write=0.8e9)], PEAK, 1e9)
    dirty = fp.hack_risk_score([_rec(1.0), _rec(0.5)], PEAK, 1e9)
    assert dirty >= clean
