"""Counter-fingerprint audit over a run trajectory (part of ARGUS contribution C1/C2).

The verifier catches hacks that change the *output*.  A second, independent line of
defence catches hacks that leave the output untouched but change *what the hardware did*:
a kernel that reports a faster time while the counters show the same bytes moved, the same
instructions retired and the same tensor-pipe utilisation did not get faster -- it got
mis-measured, or it skipped work that the timings never saw.  SOL's own screening catches
some of this; ARGUS catches it *during search*, before a wasted submission.

This module consumes the counter/latency trajectory the search already records and returns
explicit, evidence-backed flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import roofline as rl
from .verifier import HARD, SOFT, CheckOutcome


@dataclass
class FingerprintConfig:
    time_drop: float = 0.15          # relative latency drop that demands an explanation
    counter_move: float = 0.05       # relative counter move that counts as "changed"
    bw_over_peak: float = 1.05       # >1 means a physically impossible achieved BW
    store_drop: float = 0.5          # output-write fraction drop that indicates store skipping


def _rel(a: float, b: float) -> float:
    return (a - b) / b if b else 0.0


def fingerprint_audit(
    records: List[dict],
    peak: rl.GpuPeak,
    flops: float,
    *,
    config: Optional[FingerprintConfig] = None,
) -> List[CheckOutcome]:
    """Audit consecutive ``(counters, latency)`` records.

    Each record is a dict of counters plus ``time_s`` (or ``latency_ms``).  Returns a list
    of outcomes; any HARD failure should block promotion of the candidate it refers to.
    """
    cfg = config or FingerprintConfig()
    out: List[CheckOutcome] = []
    if len(records) < 2:
        return [CheckOutcome("fingerprint", True, SOFT, "not enough records to audit")]

    for i in range(1, len(records)):
        prev, new = records[i - 1], records[i]
        t_prev = _time(prev)
        t_new = _time(new)
        if t_prev <= 0 or t_new <= 0:
            continue
        speedup = t_prev / t_new

        if speedup <= 1.0 + cfg.time_drop:
            continue

        # 1) A real speedup must move at least one physical counter.
        moved = _max_counter_move(prev, new)
        if moved < cfg.counter_move:
            out.append(CheckOutcome(
                f"fingerprint:unexplained_time[{i}]", False, HARD,
                f"latency improved {speedup:.2f}x but no physical counter moved "
                f"(max change {moved*100:.1f}%)",
                {"speedup": speedup, "max_counter_move": moved}))

        # 2) Achieved BW above peak => measurement manipulation or L2-resident misread.
        try:
            pt = rl.counters_to_point(new, peak, flops)
            if pt.achieved_dram_bw > cfg.bw_over_peak * peak.dram_bw:
                out.append(CheckOutcome(
                    f"fingerprint:bw_over_peak[{i}]", False, HARD,
                    f"achieved DRAM BW {pt.achieved_dram_bw/1e12:.2f} TB/s exceeds the "
                    f"{peak.dram_bw/1e12:.2f} TB/s peak", {"achieved_bw": pt.achieved_dram_bw}))
        except ValueError:
            pass

        # 3) Output stores vanished => skipped stores (writes never happened).
        w_prev = float(prev.get("dram__bytes_write.sum", 0.0) or 0.0)
        w_new = float(new.get("dram__bytes_write.sum", 0.0) or 0.0)
        if w_prev > 0 and _rel(w_new, w_prev) < -cfg.store_drop and speedup > 1.0 + cfg.time_drop:
            out.append(CheckOutcome(
                f"fingerprint:store_skip[{i}]", False, HARD,
                f"write traffic fell {(_rel(w_new, w_prev))*100:.0f}% while time improved "
                f"{speedup:.2f}x -> output stores may be skipped",
                {"write_prev": w_prev, "write_new": w_new}))

        # 4) Tensor pipe jumped while accuracy margin shrank => narrowed precision.
        tp_prev = float(prev.get("sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", 0.0) or 0.0)
        tp_new = float(new.get("sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", 0.0) or 0.0)
        if tp_new > max(2 * tp_prev, tp_prev + 20.0) and speedup > 1.0 + cfg.time_drop:
            out.append(CheckOutcome(
                f"fingerprint:tensor_pipe_jump[{i}]", False, SOFT,
                f"tensor-pipe utilisation {tp_prev:.0f}% -> {tp_new:.0f}% while time improved "
                f"{speedup:.2f}x -> confirm the precision contract",
                {"tensor_prev": tp_prev, "tensor_new": tp_new}))

    if not out:
        out.append(CheckOutcome("fingerprint", True, SOFT, "no fingerprint anomalies"))
    return out


def _time(rec: dict) -> float:
    if "time_s" in rec and rec["time_s"]:
        return float(rec["time_s"])
    if "latency_ms" in rec and rec["latency_ms"]:
        return float(rec["latency_ms"]) / 1e3
    return 0.0


_COUNTER_KEYS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "lts__t_sector_hit_rate.pct",
]


def _max_counter_move(prev: dict, new: dict) -> float:
    moves = []
    for k in _COUNTER_KEYS:
        a = float(prev.get(k, 0.0) or 0.0)
        b = float(new.get(k, 0.0) or 0.0)
        if a != 0.0:
            moves.append(abs((b - a) / a))
    return max(moves) if moves else 0.0


def hack_risk_score(records: List[dict], peak: rl.GpuPeak, flops: float) -> float:
    """Aggregate hack risk in ``[0, 1]`` from the fingerprint outcomes (HARD=1, SOFT=0.3)."""
    outs = fingerprint_audit(records, peak, flops)
    score = 0.0
    for o in outs:
        if o.passed:
            continue
        score += 1.0 if o.severity == HARD else 0.3
    return min(score / 3.0, 1.0)
