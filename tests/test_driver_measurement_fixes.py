"""Regression tests for the measurement-fix batch on loop/driver.py.

Ported from the ForgeKernel fix wave (audit-r2-2026-11-05): each test names the
fix it pins (same numbering as the fix list). The fixtures reuse
test_loop_driver.py's FakeBackend/_ctx, so the whole file runs without a GPU.
"""

from __future__ import annotations

import pytest

import peregrine_kernel.loop.driver as drv
from peregrine_kernel.inner.trajectory import IterationRecord, append_record

from test_loop_driver import FakeBackend, _ctx


# --- fix 1: _paired_reanchor(gap=None) must not crash on the final log line ----

_CONFIRM = {"jobs": [{"ratio": 0.9}, {"ratio": 0.9}]}  # quiet re-measure, median 0.9


def _paired_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.equivalence_ratio = 1.0
    ctx.reanchor_memo = None
    ctx.paired_memo = {}
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.0, sha="shabest"))
    return ctx


def test_paired_reanchor_without_an_absolute_gap_does_not_crash(tmp_path, monkeypatch,
                                                                capsys):
    # The ref_in_job path is entered with gap=None when the absolute target
    # covers other workloads than the gate reading (resumed run + workload
    # mismatch). The closing log line used to interpolate gap unconditionally
    # and raised TypeError there — after the confirmed-gap account was already
    # computed and persisted, and after the two confirmation jobs were bought,
    # so a watchdog restart bought them again.
    be = FakeBackend([], [])
    ctx = _paired_ctx(tmp_path, be)
    monkeypatch.setattr(drv, "run_best_confirmation", lambda *a, **k: dict(_CONFIRM))
    assert drv._paired_reanchor(ctx, 1.0, None) is None
    out = capsys.readouterr().out
    assert "absolute account said" not in out
    assert "paired residual" in out          # the confirmed-gap account still logged


def test_paired_reanchor_still_quotes_the_absolute_gap_when_it_has_one(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    be = FakeBackend([], [])
    ctx = _paired_ctx(tmp_path, be)
    monkeypatch.setattr(drv, "run_best_confirmation", lambda *a, **k: dict(_CONFIRM))
    assert drv._paired_reanchor(ctx, 1.0, 0.066) is None
    assert "absolute account said +6.6%" in capsys.readouterr().out
