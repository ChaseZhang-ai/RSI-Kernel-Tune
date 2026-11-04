"""Regression tests for the measurement-fix batch on loop/driver.py.

Ported from the ForgeKernel fix wave (audit-r2-2026-11-05): each test names the
fix it pins (same numbering as the fix list). The fixtures reuse
test_loop_driver.py's FakeBackend/_ctx, so the whole file runs without a GPU.
"""

from __future__ import annotations

import pytest

import peregrine_kernel.loop.driver as drv
from peregrine_kernel.inner.trajectory import IterationRecord, append_record, load_records

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


# --- BUG-SEED-02: the unmeasured REVERT rows carry no verdict on green ---------

def test_unparseable_revert_row_is_neutral_on_green(tmp_path):
    # The record-is-None branches (budget exhausted, nothing evaluated: gate
    # refusal / probe undelivered / probe blocked / unparseable) used to file
    # correctness=False: run_inner_loop read that as "not green" and dropped the
    # lineage into scaffold mode, and _green_on_resume kept the False across
    # restarts. Nothing was measured, so the row must move nothing (the same
    # neutrality _harness_edit_revert already documents for its un-evaluated
    # REVERTs).
    be = FakeBackend(["garbage"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = drv.run_inner_iteration(ctx, 0, metric_before=2.6, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "unparseable"
    assert rec.correctness is None
    # a lineage whose only row is unmeasured does not resume as under construction
    assert drv._green_on_resume(load_records(ctx.status_path), "sha0") is None


def test_probe_blocked_revert_row_is_neutral_on_green(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n", encoding="utf-8")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = drv.run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "probe-blocked"
    assert rec.correctness is None


# --- BUG-C1-02: probe reads/writes survive non-UTF-8 bytes ----------------------

def test_bad_bytes_in_a_kept_probe_do_not_stop_the_run(tmp_path):
    # The kept-probe read sides used bare read_text(): one stray non-UTF-8 byte
    # in the probe file raised UnicodeDecodeError (not caught by the OSError
    # handlers) — the reset that ends the iteration never happened because the
    # keep blew up first, and the next iteration's re-materialisation blew up
    # the same way (it runs every iteration at run_inner_iteration's top).
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_bytes(b"print('rig \xff')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = drv.run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.decision_id == "probe-blocked"      # the iteration ended normally
    assert "notes/p.py" in ctx.kept_probes         # and the keep survived the reset
    drv._restore_kept_probes(ctx)                  # the next iteration's restore
    assert (tmp_path / "notes" / "p.py").read_bytes().startswith(b"print('rig ")


# --- BUG-SEED-29: the clean-stop safety net carries the traceback ---------------

def test_unexpected_error_log_carries_the_traceback(tmp_path, monkeypatch, capsys):
    # The unattended-run safety net logged {e!r} only: for a bug reached through
    # a few call frames (a None riding into a formatter) the repr is one line
    # with no location, so the 2am operator had nothing to debug from and the
    # lost run could not be triaged. The exception's traceback rides the log now.
    def _boom(ctx, iter_no, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(drv, "run_inner_iteration", _boom)
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    assert drv.run_inner_loop(ctx, start_iter=0, initial_green=True) == "error"
    out = capsys.readouterr().out
    assert "unexpected error, stopping cleanly: RuntimeError('boom')" in out
    assert "Traceback (most recent call last)" in out
    assert "_boom" in out   # the frames under the loop, not just the message
