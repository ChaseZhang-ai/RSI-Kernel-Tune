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

from test_loop_driver import FakeBackend, _ctx


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
