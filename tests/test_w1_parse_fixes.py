"""Regression tests for the W1 parse/inner fix line (audit-corps FA5a, dedup rebuild).

Only the W1-unique faces live here; the corps' own tests on this branch cover
the shared ones (SEED-22 / SEED-25 / SEED-28 / SEED-43 / SEED-50). The ID in a
docstring names the finding; one test per audit item.
"""

import json

from peregrine_kernel.inner.trajectory import IterationRecord
from peregrine_kernel.loop.parse import (
    _gate_ratio,
    parse_eval,
    resolvable_diff,
)


# --- SEED-47 [P2]: the numeric FAIL's tensor names come from the bench JSON ------

def test_numeric_fail_detail_names_non_attention_tensors():
    """SEED-47: the per-tensor summary used a hardcoded ("dq", "dk", "dv") — the
    vocabulary of ONE operator. A numeric FAIL of any other operator degraded to
    "no failure detail in the eval JSON"; the names now come from the JSON."""
    log = ('=== JSON ===\n{"correctness": {"pass": false, '
           '"out_main": {"max_diff": 0.4, "tolerance": 0.01}, '
           '"w_grad": {"error": "shape mismatch: got (64, 32) want (32, 64)"}}, '
           '"performance": null}')
    ev = parse_eval(log)
    assert "out_main: max_diff=0.4 tol=0.01" in ev.error
    assert "w_grad: shape mismatch" in ev.error
    assert "no failure detail" not in ev.error


def test_numeric_fail_keeps_the_attention_tensor_names():
    """SEED-47, no regression: the attention triple still reads as before."""
    log = ('=== JSON ===\n{"correctness": {"pass": false, '
           '"dq": {"max_diff": 3.1, "tolerance": 0.9}, '
           '"dk": {"error": "bad dk: shape=() dtype=None"}}, '
           '"performance": null}')
    ev = parse_eval(log)
    assert "dq: max_diff=3.1" in ev.error
    assert "dk: bad dk" in ev.error


# --- P3-2 [P3]: _gate_ratio rejects non-finite ratios at the entry ---------------


def _rec(it, ms=2.0, vs_ref=None, vs_dev=None):
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
                           vs_ref=vs_ref, vs_ref_device=vs_dev)


def test_gate_ratio_rejects_non_finite_ratios():
    """P3-2: a NaN/inf ratio is not a sample — it would pin max(ratios)/the
    median and poison every residual of the resolution estimate."""
    assert _gate_ratio(_rec(0, vs_dev=float("nan")), "device") is None
    assert _gate_ratio(_rec(0, vs_dev=float("inf")), "device") is None
    assert _gate_ratio(_rec(0, vs_ref=float("nan")), "e2e") is None
    assert _gate_ratio(_rec(0, vs_dev=1.02), "device") == 1.02   # finite: untouched


def test_non_finite_ratio_does_not_widen_the_resolution_band():
    """P3-2, end to end: a poisoned ratio reads as absent, so the estimator sees
    only the finite history (here: too few finite samples -> the cap fallback,
    not a NaN-driven number)."""
    recs = [_rec(0, vs_dev=float("nan"))]
    assert resolvable_diff(recs, "device") == resolvable_diff([], "device")
