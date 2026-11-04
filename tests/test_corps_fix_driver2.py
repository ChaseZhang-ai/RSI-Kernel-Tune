"""Regression tests for the audit-corps fix wave 2 (driver.py measurement face).

Covers: BUG-SEED-05 (patch pricing band across jobs), BUG-SEED-06 (watermark
confirmation must also beat the old best), BUG-SEED-31 (drift baseline reads
the same cluster's best), BUG-C1-02 (probe IO survives a bad byte) and
BUG-C1-03 (cluster anchors only consumed for the window they were read in).
"""

from __future__ import annotations

import json

import pytest

from peregrine_kernel.inner.trajectory import IterationRecord as IR, append_record
from peregrine_kernel.loop.driver import _judge_patch, _PendingPatch

from test_loop_driver import EVAL_DW, FakeBackend, _ctx


# ---------------------------------------------------------------- BUG-SEED-05

def test_patch_pricing_absolute_branch_requires_the_cross_job_band(tmp_path):
    """A -3.8% absolute-ms gain across jobs used to earn KEEP_PATCH off the
    run's same-job ratio band (rd, data-poor cap 1.5%); the cross-job band the
    codebase itself assigns to absolute ms comparisons (6%) must govern."""
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, IR(iter=0, decision_id="D", change="x",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.6, device_median_ms=2.6,
                                      sha="s0"))
    append_record(ctx.status_path, IR(iter=1, decision_id="D1", change="y",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.5, device_median_ms=2.5,
                                      sha="s1"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.6, milestone_before="m",
                                          good_ref="good", records_before=1))
    assert out == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_patch_pricing_absolute_branch_still_credits_a_real_gain(tmp_path):
    """A gain that clears the 6% cross-job band (-15.4%) is still credited."""
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, IR(iter=0, decision_id="D", change="x",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.6, device_median_ms=2.6,
                                      sha="s0"))
    append_record(ctx.status_path, IR(iter=1, decision_id="D1", change="y",
                                      correctness=True, verdict="KEEP",
                                      bwd_median_ms=2.2, device_median_ms=2.2,
                                      sha="s1"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.6, milestone_before="m",
                                          good_ref="good", records_before=1))
    assert out == "KEEP_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


# ---------------------------------------------------------------- BUG-SEED-06

def test_watermark_confirmation_past_the_old_best_does_not_confirm(tmp_path, capsys):
    """best 100 -> trigger draw 98.0 -> confirmation 100.9: the two draws agree
    inside the margin (2.9 <= 2.94), yet the confirmation draw is WORSE than the
    old best — the jump did not replicate. It used to be confirmed and the
    median (99.45) banked as a gain; now it settles unconfirmed and the row is
    marked low-confidence."""
    from peregrine_kernel.loop.driver import _confirm_suspicious_watermark
    from peregrine_kernel.loop.parse import parse_eval
    be = FakeBackend([], [EVAL_DW(105.0, 100.9)])   # the confirmation job's draw
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    ev = parse_eval(EVAL_DW(103.0, 98.0))           # the trigger draw
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 100.0, profiler={})
    assert info["confirm_jobs"] == 1                # band trigger: one budgeted job
    assert info["low_confidence"] is True           # draw spread 2.96% > rd band
    text = capsys.readouterr().out
    assert "did not replicate" in text              # the old-best dissent is logged
    assert "did not confirm" in text                # settles unconfirmed
    assert "confirmed — recording" not in text
    assert out.device_median_ms == pytest.approx((98.0 + 100.9) / 2)


def test_watermark_confirmation_below_the_old_best_still_confirms(tmp_path, capsys):
    """The dissent rule only bites past the old best: a confirmation draw that
    agrees inside the margin AND clears the old best confirms as before."""
    from peregrine_kernel.loop.driver import _confirm_suspicious_watermark
    from peregrine_kernel.loop.parse import parse_eval
    be = FakeBackend([], [EVAL_DW(105.0, 99.0)])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    ev = parse_eval(EVAL_DW(103.0, 98.0))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 100.0, profiler={})
    text = capsys.readouterr().out
    assert "confirmed — recording" in text          # median recorded as confirmed
    assert info["confirm_jobs"] == 1
    assert out.device_median_ms == pytest.approx((98.0 + 99.0) / 2)
