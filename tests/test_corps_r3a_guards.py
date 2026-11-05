"""Regression tests for the R3A audit wave: guard anchoring in the inner loop.

BUG-CH-A-02 (P3): the background lane's temperature guard called float() on the
eval JSON's gpu_state.temperature_c with no guard (only load_records was
protected — "a guard must never break the window"), so one non-numeric value in
a persisted status row crashed every later open_window deterministically, at a
point the restart cannot get past.

BUG-CH-A-01 (P3): _GATE_ROW matched the substring "gate" anywhere in a
decision_id, so ordinary names like "D1.4-mitigate-spill" or "D2.1-investigate"
were treated as gate rows — losing the pre-green scaffold exemption, being
refused by the placeholder ledger, and gaining gate-only reuse privileges. The
anchored paradigm already in the file (_STAG_GATE_ROW / _GATE_ID: M<k>-gate)
is the correct shape.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import (EVAL, NCU, FakeBackend, _ctx,               \
                               run_inner_iteration)  # noqa: E402

from peregrine_kernel.loop import driver

_REC = lambda did, change, kf="flash_attn_dsl/flash_bwd.py": (  # noqa: E731
    f"decision_id: {did}\nchange: {change}\nfiles_touched: {kf}\n")


# --- BUG-CH-A-02 -----------------------------------------------------------


def _status_with_temperature(tmp_path, temp_value):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    row = {"iter": 0, "decision_id": "D1.1", "change": "x", "verdict": "KEEP",
           "correctness": True, "gpu_state": {"temperature_c": temp_value}}
    with open(ctx.status_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return ctx


def test_temperature_guard_reads_a_numeric_temperature(tmp_path):
    ctx = _status_with_temperature(tmp_path, 90.0)
    ok, why = driver._lane_temperature_check(ctx)
    assert ok is False and "90.0" in why   # the hot reading still fires the guard


def test_temperature_guard_tolerates_a_non_numeric_temperature(tmp_path):
    """A damaged/hand-edited temperature reads as "no usable temperature" — the
    guard returns the window open instead of raising out of open_window."""
    for junk in ({"c": 80}, ["80"], "hot", ""):
        ctx = _status_with_temperature(tmp_path, junk)
        ok, why = driver._lane_temperature_check(ctx)
        assert ok is True and why == "", junk


def test_temperature_guard_falls_through_to_the_last_usable_reading(tmp_path):
    """A junk newest row must not blind the guard: the scan continues to older
    rows, so a real over-limit reading still closes the window."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    rows = [{"iter": 0, "decision_id": "D1.1", "change": "x", "verdict": "KEEP",
             "correctness": True, "gpu_state": {"temperature_c": 95.0}},
            {"iter": 1, "decision_id": "D1.2", "change": "y", "verdict": "KEEP",
             "correctness": True, "gpu_state": {"temperature_c": {"c": 80}}}]
    with open(ctx.status_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    ok, why = driver._lane_temperature_check(ctx)
    assert ok is False and "95.0" in why


# --- BUG-CH-A-01 -----------------------------------------------------------


def test_gate_row_regex_is_anchored_like_the_stagnation_one():
    """The gate-row shape is M<k>-gate (see _STAG_GATE_ROW / _GATE_ID); words
    that merely contain "gate" are ordinary decisions."""
    for did in ("D1.4-mitigate-spill", "D2.1-investigate", "D3.2-aggregate",
                "D0.7-gatehouse", "M0-gate-2", "legate"):
        assert not driver._GATE_ROW.search(did), did
    for did in ("M0-gate", "M1-gate", "m2-gate", "M12-gate"):
        assert driver._GATE_ROW.search(did), did


def test_a_gate_substring_decision_keeps_scaffold_treatment(tmp_path):
    """Pre-green, an ordinary decision whose name contains "gate" inside another
    word must take the scaffold branch (no full eval bought), not the gate-row
    path — the old substring match sent it to the eval and its hard REVERT."""
    be = FakeBackend([_REC("D1.4-mitigate-spill", "reduce the spill walks")], [])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=None, last_good_ref="sha0",
                              green=False)
    assert rec.decision_id == "D1.4-mitigate-spill"
    assert rec.verdict == "NEUTRAL" and rec.correctness is None
    assert not [c for c in be.calls if c[0] == "gpu"]   # the eval stayed unbought


def test_a_real_gate_row_still_gets_the_full_eval(tmp_path):
    """The anchored regex keeps every legitimate gate-row behaviour: M1-gate in
    a pre-green segment is still evaluated (never scaffolded)."""
    logs = [EVAL(2.6), NCU]
    be = FakeBackend([_REC("M1-gate", "close the phase")], logs)
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=None, last_good_ref="sha0",
                              green=False)
    assert rec.decision_id == "M1-gate"
    assert not (rec.verdict == "NEUTRAL" and rec.correctness is None)
    assert [c for c in be.calls if c[0] == "gpu"]   # the gate row bought its eval
