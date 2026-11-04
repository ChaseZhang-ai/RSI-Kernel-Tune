"""Regression tests for the audit-corps R2 parse.py fix cluster.

Four ledger bugs, one test section each:

* BUG-SEED-04 — ``resolvable_diff`` pooled reused-reading rows (``reading_of``
  set, the driver's copies of an earlier row's reading) into the ratio and
  window-pair pools; duplicated samples are not noise draws and flattened the
  between-job sigma onto the floor.
* BUG-SEED-22 — NaN / inf timing readings in the eval JSON reached the parsed
  ``EvalResult`` and locked every downstream min/comparison.
* BUG-SEED-47 — the per-tensor correctness failure detail was hardcoded to
  dq/dk/dv and degraded to "no failure detail" for any other tensor name.
* BUG-SEED-50 — PROBE_REQUEST / PROBE-MANIFEST / EXHAUSTED matched anywhere in
  the text, so a prose mention of a marker terminated a run or burned a job.
"""

from __future__ import annotations

from peregrine_kernel.inner.trajectory import GATE_WINDOW_DEVICE, IterationRecord, gate_ms
from peregrine_kernel.loop.parse import parse_eval, resolvable_diff


def _rec(i, ratio=None, *, reading_of=None, ms=2.0, dev=None, ref=None):
    """One evaluated KEEP row, mirroring test_wp15's construction."""
    kw = dict(device_median_ms=ms, correctness=True, verdict="KEEP",
              decision_id=f"D{i}", change="c")
    if ratio is not None:
        kw["vs_ref_device"] = ratio
    if dev and ref:
        kw["workloads"] = [{"name": f"w{i}", "device_readings_ms": list(dev),
                            "ref_device_readings_ms": list(ref)}]
    if reading_of is not None:
        kw["reading_of"] = reading_of
    return IterationRecord(iter=i, **kw)


# --- BUG-SEED-04: reused reading rows are not noise draws ----------------------

def test_resolvable_diff_ratio_pool_skips_reused_rows():
    """Triage experiment: three fresh ratios [1.000, 1.020, 1.040] plus three
    copies of the middle row (reading_of set). The copies duplicated 1.020 four
    times in the pool; the leave-one-out residuals then flattened to zero and
    the band clamped to the floor — three real samples were made to look like
    six agreeing ones. Reused rows are excluded (driver.py's _runner_up_pool,
    _climb_streak and the stagnation ledger already exclude them): with fewer
    than RESOLVABLE_MIN_SAMPLES real ratios left the estimate honestly returns
    the cap, the data-poor fallback."""
    fresh = [_rec(i, r) for i, r in enumerate((1.000, 1.020, 1.040), start=1)]
    reused = [_rec(4 + i, 1.020, reading_of=2) for i in range(3)]
    assert resolvable_diff(fresh + reused, GATE_WINDOW_DEVICE) == 0.015
    assert resolvable_diff(fresh, GATE_WINDOW_DEVICE) == 0.015


def test_resolvable_diff_reused_rows_change_nothing_they_touch():
    """The pool invariant the bug breaks, on a pairs-bearing history: appending
    reused copies of existing rows must not move the estimate — the copies ride
    the ratio pool AND the window-pair pool (identical raw arrays, identical
    per-record centers), and before the fix the duplicated pairs also crowded
    later records' real pairs out of the RESOLVABLE_MAX_WINDOW_PAIRS window."""
    dev = [2.40, 2.44, 2.42, 2.415]
    ref = [2.42, 2.46, 2.43, 2.435]
    fresh = [_rec(1, 1.01, dev=dev, ref=ref), _rec(2, 1.005, dev=dev, ref=ref),
             _rec(3, 1.012, dev=dev, ref=ref), _rec(4, 1.004, dev=dev, ref=ref),
             _rec(5, 1.008, dev=dev, ref=ref), _rec(6, 1.011, dev=dev, ref=ref)]
    copies = [_rec(7, 1.008, reading_of=2, dev=dev, ref=ref),
              _rec(8, 1.011, reading_of=3, dev=dev, ref=ref)]
    without = resolvable_diff(fresh, GATE_WINDOW_DEVICE)
    assert resolvable_diff(fresh + copies, GATE_WINDOW_DEVICE) == without


# --- BUG-SEED-22: NaN / inf timing readings never reach the parsed eval --------

def test_parse_eval_drops_non_finite_timing_readings():
    """The probe verdict path audits a reported diff for finiteness (the
    NaN/inf branch of _audit_pass); the timing readings took isinstance checks
    only, so a bench fault's NaN reached the parsed record and locked every
    downstream comparison (min(recs, key=gate_ms) kept returning the NaN row,
    plateau's recent_best >= prior_best * (1 - margin) went permanently False).
    A non-finite reading is a bench fault, not a fact: dropped at parse, the
    field stays None / the list entry absent, exactly like an unreported one."""
    log = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": {'
           '"bwd_median_ms": NaN, "device_median_ms": Infinity, '
           '"vs_ref_device": NaN, "host_enqueue_ms": Infinity, '
           '"device_readings_ms": [1.0, NaN, Infinity, 1.02], '
           '"ref_device_readings_ms": [Infinity, 2.0], '
           '"workloads": [{"name": "w0", "device_median_ms": NaN, '
           '"bwd_median_ms": 1.5, "host_enqueue_ms": NaN, '
           '"device_readings_ms": [NaN, 3.0, Infinity]}], "n_workloads": 1}}')
    ev = parse_eval(log)
    assert ev.correctness is True
    assert ev.bwd_median_ms is None
    assert ev.device_median_ms is None
    assert ev.vs_ref_device is None
    assert ev.host_enqueue_ms is None
    assert ev.device_readings_ms == [1.0, 1.02]
    assert ev.ref_device_readings_ms == [2.0]
    w = ev.workloads[0]
    assert "device_median_ms" not in w
    assert "host_enqueue_ms" not in w
    assert w["bwd_median_ms"] == 1.5
    assert w["device_readings_ms"] == [3.0]
    assert "device_spread_pct" not in w
    # the gate-consumption shape the bug locked: a NaN window reads as no
    # reading (None), never as a row min() can keep selecting forever
    assert gate_ms(ev, GATE_WINDOW_DEVICE) is None


def test_parse_eval_finite_readings_ride_unchanged():
    log = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": {'
           '"bwd_median_ms": 2.6, "device_median_ms": 0.25, '
           '"device_readings_ms": [0.25, 0.26], '
           '"workloads": [{"name": "w0", "device_median_ms": 0.24, '
           '"device_readings_ms": [0.23, 0.24, 0.25]}], "n_workloads": 1}}')
    ev = parse_eval(log)
    assert ev.bwd_median_ms == 2.6
    assert ev.device_median_ms == 0.25
    assert ev.device_readings_ms == [0.25, 0.26]
    assert ev.workloads[0]["device_median_ms"] == 0.24
    assert ev.workloads[0]["device_readings_ms"] == [0.23, 0.24, 0.25]
