"""The paired re-anchor's log must tolerate the None absolute gap.

A resumed run whose gate reading covers other workloads than its reference
archive has its absolute target suspended (_target_workloads_mismatch), so the
phantom-gap check hands the paired re-anchor ``gap=None`` (driver.py
``_paired_reanchor(ctx, best, None)``). When the same-card re-measure then
CONFIRMS a gap (median ratio below the equivalence ratio), the closing log line
used to format that None unconditionally — "(absolute account said
+{gap:.1%})" — and died with a TypeError AFTER the two confirm GPU jobs were
already bought, killing the process at a point a watchdog relaunch would crash
at again. These tests pin the degraded (not fatal) behavior.
"""

from __future__ import annotations

import json

import pytest

from test_loop_driver import _ConfirmBackend, _EVAL_CONFIRM, _ctx

from peregrine_kernel.inner.trajectory import IterationRecord, append_record
from peregrine_kernel.loop.driver import _phantom_gap_reanchor


def _mismatch_ctx(tmp_path, be, *, eq=None):
    """A ref_in_job run whose best KEEP reads over 4 workloads against a
    single-input reference archive: the absolute target is suspended, the
    phantom-gap check may only quote the same-card ratio (gap=None)."""
    ctx = _ctx(tmp_path, be, target_ms=2.0)
    ctx.gate_window = "device"
    ctx.reanchor_band = 0.06
    ctx.target_anchor_ms = 2.0
    ctx.target_anchor_cluster = ""
    def _no_ref_channel():
        raise AssertionError("fresh ref channel must not be consulted here")
    ctx.reanchor_ref = _no_ref_channel
    ctx.ref_in_job = True
    if eq is not None:
        ctx.equivalence_ratio = eq
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest", n_workloads=4))
    return ctx


def test_paired_reanchor_survives_a_confirmed_gap_under_a_none_absolute_gap(
        tmp_path, capsys):
    """resumed run + workloads mismatch + ref_in_job: the same-card re-measure
    comes back below the equivalence ratio, the diagnosis is handed the paired
    residual, and the log degrades to ratio-only wording instead of formatting
    the missing absolute residual."""
    ctx = _mismatch_ctx(tmp_path, _ConfirmBackend(
        [], [_EVAL_CONFIRM(2.07, 2.03), _EVAL_CONFIRM(2.07, 2.01)]))
    out = _phantom_gap_reanchor(ctx, 2.07)
    assert out is None                                    # confirmed gap: diagnose on, no stop
    assert ctx.reanchor_note and "Same-card paired re-measure" in ctx.reanchor_note
    assert "+2.5%" in ctx.reanchor_note                   # 1/0.9758 - 1
    assert ("the absolute target and the gate reading cover different workloads"
            in ctx.reanchor_note)
    # the account persists with an explicitly null absolute gap
    rec = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "confirmed_gap" and rec["estimator"] == "paired-in-job"
    assert rec["gap_vs_absolute_target"] is None
    med = (2.03 / 2.07 + 2.01 / 2.07) / 2          # the two draws' median ratio
    assert ctx.paired_residual_memo == (2.07, pytest.approx(1 / med - 1))
    # the log line that used to raise: ratio-only, no absolute-account clause
    line = next(l for l in capsys.readouterr().out.splitlines()
                if "paired residual" in l)
    assert "median ratio 0.9758 < 1.0" in line
    assert "+2.5%" in line and "absolute account said" not in line


def test_paired_reanchor_reuse_of_an_unconfirmed_measure_also_skips_the_absolute_account(
        tmp_path):
    """The same gap=None reaches the closing log through the memoised re-measure
    of the same sha (an earlier iter's unconfirmed parity reading): no new jobs
    are bought, the reuse stands, and the log still does not crash."""
    ctx = _mismatch_ctx(tmp_path, _ConfirmBackend([], []))
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    assert not any(c[0] == "gpu" for c in ctx.backend.calls)
    assert ctx.reanchor_note and "+1.5%" in ctx.reanchor_note   # 1/0.985 - 1
