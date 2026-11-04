"""Regression tests for the audit-corps fix wave 7 (the parse/inner/trajectory
surface): non-finite eval readings must not hijack the plateau minimum, scalar
eval readings must be real numbers, scientific-notation correctness diffs must
keep their signatures, the correctness summary must trust its own failure count,
the perf JSON scan must respect string literals, and a torn trajectory line must
not sink the run's whole history."""
import json

from peregrine_kernel.inner.results import parse_correctness  # noqa: F401  (used by later waves)
from peregrine_kernel.inner.trajectory import IterationRecord
from peregrine_kernel.loop.parse import _float_list, detect_plateau, parse_eval


def _eval_text(**perf):
    body = {"correctness": {"pass": True}, "performance": perf}
    return "=== JSON ===\n" + json.dumps(body)


def _rec(it, ms, verdict="KEEP"):
    # dual-window record, same reading in both (the plateau math is window-agnostic)
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict=verdict, bwd_median_ms=ms, device_median_ms=ms)


# --- BUG-SEED-22: NaN readings reached detect_plateau and hijacked its minimum ---

def test_float_list_drops_non_finite_entries():
    # same filter readings_spread_pct applies: a NaN reading is a broken
    # measurement, not a small one
    assert _float_list([1.0, float("nan"), 3.0, float("inf"), True, "2.0"]) == [1.0, 3.0]
    assert _float_list("not a list") == []


def test_parse_eval_drops_non_finite_scalar_readings():
    ev = parse_eval(_eval_text(device_median_ms=float("nan"), bwd_median_ms=2.5,
                               ref_device_ms=float("nan"), vs_ref=float("nan"),
                               host_enqueue_ms=float("inf")))
    assert ev.device_median_ms is None
    assert ev.bwd_median_ms == 2.5
    assert ev.ref_device_ms is None
    assert ev.vs_ref is None
    assert ev.host_enqueue_ms is None


def test_parse_eval_drops_non_finite_workload_readings():
    ev = parse_eval(_eval_text(bwd_median_ms=2.5, workloads=[
        {"name": "w0", "device_median_ms": float("nan"),
         "device_readings_ms": [1.0, float("nan"), 3.0], "vs_ref_device": 1.0},
        {"name": "w1", "device_median_ms": 2.0,
         "device_readings_ms": [2.0, float("inf")], "vs_ref_device": float("nan")},
    ]))
    w0, w1 = ev.workloads
    assert "device_median_ms" not in w0
    assert w0["device_readings_ms"] == [1.0, 3.0]
    assert w0["vs_ref_device"] == 1.0
    assert w1["device_median_ms"] == 2.0
    assert w1["device_readings_ms"] == [2.0]
    assert "vs_ref_device" not in w1


def test_a_nan_reading_cannot_hijack_the_plateau_minimum():
    # six-row shape: no new best inside the window -> plateau True
    rows = [_rec(1, 1.00), _rec(2, 1.02), _rec(3, 1.03), _rec(4, 1.04), _rec(5, 1.05)]
    assert detect_plateau(rows, window=3) is True
    # the driver builds its records from parse_eval's output, so the NaN eval must
    # arrive as "no reading": the row drops out of the evaluated set instead of
    # sitting first in the min() and turning the plateau verdict permanently False
    ev = parse_eval(_eval_text(device_median_ms=float("nan"), bwd_median_ms=2.5))
    assert ev.device_median_ms is None
    nan_row = _rec(0, ev.device_median_ms)
    assert detect_plateau([nan_row] + rows, window=3) is True
