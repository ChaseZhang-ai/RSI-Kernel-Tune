"""Regression tests for the W1 parse/inner fix line (audit-corps FA5a, dedup rebuild).

Only the W1-unique faces live here; the corps' own tests on this branch cover
the shared ones (SEED-22 / SEED-25 / SEED-28 / SEED-43 / SEED-50). The ID in a
docstring names the finding; one test per audit item.
"""

import json

import pytest

from peregrine_kernel.inner.trajectory import IterationRecord
from peregrine_kernel.loop.parse import (
    _gate_ratio,
    parse_eval,
    parse_probe_verdict,
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


# --- P3-10 [P3]: "PROBE PASSED" keeps its max_diff capture -----------------------

def test_probe_passed_suffix_keeps_max_diff():
    """P3-10: the PASSED spelling let the trailing 'ED' eat the optional
    max_diff= group (after 'PASS' comes 'ED', not \s+max_diff=), so a real
    reading was reported unreported."""
    v = parse_probe_verdict("PROBE PASSED max_diff=0.5")
    assert v.passed is True and v.max_diff == 0.5
    c = v.contracts[0]
    assert c.verdict == "PASS" and c.raw == "0.5" and c.note == ""
    # FAILED still reads as FAIL, with its reading kept for the feedback
    v = parse_probe_verdict("PROBE FAILED max_diff=2.5")
    assert v.passed is False and v.max_diff == 2.5 and v.found is True
    # the bare spellings parse exactly as before
    assert parse_probe_verdict("PROBE PASS max_diff=0.0").passed is True
    assert parse_probe_verdict("PROBE FAIL max_diff=1.13").passed is False
# --- P3-10 [P3]: "PROBE PASSED" keeps its max_diff capture -----------------------

def test_probe_passed_suffix_keeps_max_diff():
    """P3-10: the PASSED spelling let the trailing 'ED' eat the optional
    max_diff= group (after 'PASS' comes 'ED', not \s+max_diff=), so a real
    reading was reported unreported."""
    v = parse_probe_verdict("PROBE PASSED max_diff=0.5")
    assert v.passed is True and v.max_diff == 0.5
    c = v.contracts[0]
    assert c.verdict == "PASS" and c.raw == "0.5" and c.note == ""
    # FAILED still reads as FAIL, with its reading kept for the feedback
    v = parse_probe_verdict("PROBE FAILED max_diff=2.5")
    assert v.passed is False and v.max_diff == 2.5 and v.found is True
    # the bare spellings parse exactly as before
    assert parse_probe_verdict("PROBE PASS max_diff=0.0").passed is True
    assert parse_probe_verdict("PROBE FAIL max_diff=1.13").passed is False
# --- P3-5 [P3]: update_last_outcome validates the outcome at the entry -----------

from peregrine_kernel.meta_trajectory import (MetaRecord, append_meta_record,
                                              load_meta_records, update_last_outcome)


def test_update_last_outcome_rejects_an_invalid_outcome(tmp_path):
    """P3-5: the attribute write bypassed __post_init__, so a bad outcome landed
    on disk and made the whole meta_status unreadable at the next load. The
    entry now validates before anything is written; the file is untouched."""
    p = tmp_path / "meta_status.jsonl"
    append_meta_record(p, MetaRecord(meta_step=0, cause_class="G1",
                                     deficient_locus="D0", patch_class="additive",
                                     changed_entries=[], patch_direction="x"))
    with pytest.raises(ValueError):
        update_last_outcome(p, outcome="BOGUS_OUTCOME")
    recs = load_meta_records(p)                    # the file still loads
    assert recs[0].outcome == "PENDING"            # ...and was not touched
    update_last_outcome(p, outcome="KEEP_PATCH", best_after=1.9)
    assert load_meta_records(p)[0].outcome == "KEEP_PATCH"
