"""Regression test ported from the audit snapshot's monitor/resume fix batch.

Only the assertion that holds in today's semantics is kept here; the rest of
the snapshot file waits for fixes owned by other lines:

* classify's zero-hook-process finding (BUG-SEED-14), parse_log's partial
  rebuild when the final JSON is killed mid-dump (BUG-SEED-15), the symmetric
  companion closure (BUG-SEED-16), the hook's own dump-failure record
  (BUG-SEED-41), the per-process call merge (BUG-SEED-42) and load_records'
  torn-final-line tolerance (BUG-SEED-43) are alive and owned by another fix
  line — their regression tests port when those fixes land;
* the max_diff signature's scientific-notation forms (BUG-SEED-25) and the
  inner results parser (BUG-SEED-28) are alive, no line assigned yet;
* the snapshot's digest assertions referenced per-curve ``clock_drift`` keys
  and an ``sm_clock_mhz`` inside ``achieved`` — shapes that no longer exist.
  BUG-SEED-30's fix took a different form (WP15): the digest carries a
  top-level ``clock`` header read from the newest measured row's gpu_state,
  and the curve stays a plain objective series — that contract is what this
  port pins.
"""

from __future__ import annotations

from rsi_kernel_tune.inner.trajectory import (
    IterationRecord,
    build_digest,
)


def _ir(i, *, verdict="KEEP", error="", **kw):
    kw.setdefault("bwd_median_ms", None if error else 2.0)
    return IterationRecord(iter=i, decision_id="D", change="c",
                           correctness=None if verdict == "NEUTRAL" else not error,
                           verdict=verdict, error=error, **kw)


def test_digest_carries_the_clock_telemetry_header():
    # WP15's answer to the write-only clock telemetry: the digest surfaces the
    # newest measured row's gpu_state as the top-level "clock" header (the
    # cross-job comparison's first reading), beside the objective curve.
    recs = [_ir(0, bwd_median_ms=3.0, device_median_ms=2.9),
            _ir(1, bwd_median_ms=2.5, device_median_ms=2.4, clock_drift=True,
                gpu_state={"sm_clock_mhz": 1590.0, "temperature_c": 61.0}),
            _ir(2, verdict="REVERT", error="dq max_diff=0.9")]
    d = build_digest(recs, {"value": 2.0})
    assert d["clock"] == {"iter": 1, "sm_clock_mhz": 1590.0, "temperature_c": 61.0}
    # the achieved block still names the gate-window best row
    assert d["achieved"]["iter"] == 1
    # a trajectory with no gpu_state telemetry carries an empty clock header
    d2 = build_digest([_ir(0, bwd_median_ms=3.0)], {"value": 2.0}, gate_window="e2e")
    assert d2["clock"] == {}

# -- PORT union: monitor/resume faces (parse_log, classify joins, results scan) --

import json

from rsi_kernel_tune import monitor
from rsi_kernel_tune.inner.results import parse_perf


def _p_rec(profile, card_pass=True, *, calls=()):
    """A per-profile record the way runtime/run.py writes it."""
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": [],
            "hook": {"calls": list(calls), "perturbed": 0, "reseeded": 0,
                     "errors": []}}


def _call(i=0, *, abs_sum=10.0):
    return {"fn": "call", "i": i, "inputs_nonfinite": 0, "in_sums": [1.0],
            "outputs": [{"nan": 0, "posinf": 0, "neginf": 0, "abs_sum": abs_sum,
                         "sq_sum": abs_sum / 2}]}


def test_parse_log_a_complete_final_json_still_wins():
    final = {"profiles": [_p_rec("base")], "skipped": [], "complete": True}
    text = (f"{monitor.PROFILE_MARKER}\n{json.dumps(_p_rec('base'))}\n"
            f"{monitor.MARKER}\n{json.dumps(final)}")
    res = monitor.parse_log(text)
    assert res == final and res["complete"] is True


def test_cross_run_comparison_joins_calls_ignoring_pids():
    def rec(profile, pid, abs_sum, card_pass=True):
        block = _p_rec(profile, card_pass, calls=[_call(abs_sum=abs_sum)])
        block["hook"]["calls"][0]["pid"] = pid
        return block

    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("repeat", 200, 10.1)]}
    v = monitor.classify(res)
    assert v["status"] == "warn" and "race" in v["findings"][0]["what"]
    # unchanged inputs across two different pids still read as "not applicable"
    res = {"complete": True,
           "profiles": [rec("base", 100, 10.0), rec("amp", 200, 10.0)]}
    v = monitor.classify(res)
    assert v["not_applicable"] == ["amp"]


def test_json_scan_survives_braces_inside_strings():
    text = 'noise {"performance": {"bwd_median_ms": 1.0, "label": "a}b{c"}} {"x": 1}'
    assert parse_perf(text)["bwd_median_ms"] == 1.0
