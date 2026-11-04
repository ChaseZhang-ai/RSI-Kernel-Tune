"""Regression tests ported from the audit snapshot's measurement-fix batch.

Each test pins one audited defect in today's semantics (the bug ids below are
the ledger's): BUG-SEED-01 (the gap=None anchor-log crash), BUG-SEED-02
(never-evaluated REVERT rows claim no correctness verdict), BUG-SEED-05 (patch
pricing reads the cross-job band), BUG-SEED-06 (a confirmation draw past the
old best dissents), BUG-SEED-29 (the safety net logs the traceback), and
BUG-SEED-31 (the drift baseline reads the same cluster's best).

The fixtures reuse test_loop_driver.py's FakeBackend/_ctx and the dual-window
eval log, so the whole file runs without a GPU.
"""

from __future__ import annotations

import pytest

import peregrine_kernel.loop.driver as drv
from peregrine_kernel.inner.trajectory import IterationRecord, append_record, load_records
from peregrine_kernel.loop.driver import run_inner_iteration
from peregrine_kernel.loop.parse import parse_eval

from test_loop_driver import EVAL_DW, FakeBackend, _ctx


# --- BUG-SEED-01: _paired_reanchor(gap=None) must not crash on the final log line

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
    # covers other workloads than the gate reading. The closing log line used to
    # interpolate gap unconditionally and raised TypeError there — after the
    # confirmed-gap account was already computed and persisted.
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


# --- BUG-SEED-02: unmeasured REVERT rows carry no verdict on green

def test_unparseable_revert_row_is_neutral_on_green(tmp_path):
    # The record-is-None branches (budget exhausted, nothing evaluated) used to
    # file correctness=False: run_inner_loop read that as "not green" and dropped
    # the lineage into scaffold mode, and _green_on_resume kept the False across
    # restarts. Nothing was measured, so the row must move nothing.
    be = FakeBackend(["garbage"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = run_inner_iteration(ctx, 0, metric_before=2.6, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "unparseable"
    assert rec.correctness is None
    # a lineage whose only row is unmeasured does not resume as under construction
    assert drv._green_on_resume(load_records(ctx.status_path), "sha0") is None


def test_probe_blocked_revert_row_is_neutral_on_green(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev")
    assert rec.verdict == "REVERT" and rec.decision_id == "probe-blocked"
    assert rec.correctness is None


# --- BUG-SEED-05: _judge_patch's absolute branch reads the cross-job band

def _judge_ctx(tmp_path, be, best_after):
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D", change="x", correctness=True, verdict="KEEP",
        bwd_median_ms=best_after, device_median_ms=best_after, sha="s0"))
    return ctx


def _judge(ctx, best_before=2.66):
    from peregrine_kernel.loop.driver import _PendingPatch, _judge_patch
    return _judge_patch(ctx, _PendingPatch(best_before=best_before,
                                           milestone_before="snapshot",
                                           good_ref="good"))


def test_patch_absolute_gain_inside_the_cross_job_band_reverts(tmp_path):
    # 2.66 -> 2.60 is 2.26%: past the same-job ratio noise but inside the 6%
    # cross-job node-draw band (_GATE_REGRESSION_TOLERANCE_MS — one unchanged
    # commit moves ~6% between draws), so absolute ms cannot attribute it.
    be = FakeBackend([], [])
    ctx = _judge_ctx(tmp_path, be, best_after=2.60)
    assert _judge(ctx) == "REVERT_PATCH"
    assert ("reset", "good") in be.calls


def test_patch_absolute_gain_beyond_the_cross_job_band_is_kept(tmp_path):
    # 2.66 -> 2.35 is ~11.7%: a gain no node draw fakes — KEEP_PATCH, no rollback
    be = FakeBackend([], [])
    ctx = _judge_ctx(tmp_path, be, best_after=2.35)
    assert _judge(ctx) == "KEEP_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


# --- BUG-SEED-06: the confirmation must agree AND beat the old best ------------

class _GpuLogBackend:
    """Just enough backend for _confirm_suspicious_watermark: hands out eval logs."""

    def __init__(self, logs):
        self.logs = list(logs)
        self.submitted = []

    def submit_gpu(self, branch, mode):
        self.submitted.append(mode)
        return self.logs.pop(0)


def _wm_ctx(tmp_path, be, margin=0.03):
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = margin
    return ctx


def _ratio_row(it, ms, ratio):
    return IterationRecord(iter=it, decision_id=f"D{it}", change="c", correctness=True,
                           verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
                           vs_ref_device=ratio)


def _seed_own_readings(ctx, n, ms=2.0):
    for i in range(n):
        append_record(ctx.status_path, _ratio_row(i, ms, 1.20))


def test_confirmation_slower_than_the_old_best_is_not_a_confirmed_gain(tmp_path):
    # band trigger (2% gain over a quiet ratio history) with the full budget of
    # two jobs: the first confirmation reads 2.01 — SLOWER than the old best 2.0,
    # yet within the trigger-draw-anchored band. That draw is a dissent (the jump
    # did not replicate), the budget's second job is bought anyway, and the
    # median of all three draws settles the watermark unconfirmed.
    be = _GpuLogBackend([EVAL_DW(2.4, 2.01), EVAL_DW(2.4, 2.03)])
    ctx = _wm_ctx(tmp_path, be)
    ctx.confirm_budget_per_iter = 2
    _seed_own_readings(ctx, 6)
    ev = parse_eval(EVAL_DW(2.4, 1.96))
    out, info = drv._confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx(2.01)  # median of 1.96/2.01/2.03
    assert info["confirm_jobs"] == 2
    assert be.submitted == ["all", "all"]


def test_band_trigger_buys_one_budgeted_job_and_the_median_takes_the_watermark(tmp_path):
    # WP15 semantics (the snapshot's "own-sample support bar" is gone): a 2.5%
    # gain over a quiet ratio history sits inside the margin but past what the
    # run's ratios resolve — the WP1 band buys ONE budgeted confirmation job, and
    # once the confirmation agrees below the old best, the median of the two
    # draws takes the watermark. A data-poor run has no free pass here.
    be = _GpuLogBackend([EVAL_DW(2.4, 1.97)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_own_readings(ctx, 6)
    ev = parse_eval(EVAL_DW(2.4, 1.95))
    out, info = drv._confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.95 + 1.97) / 2)
    assert info["confirm_jobs"] == 1
    assert be.submitted == ["all"]


# --- BUG-SEED-29: the clean-stop safety net carries the traceback

def test_unexpected_error_log_carries_the_traceback(tmp_path, monkeypatch, capsys):
    from peregrine_kernel.loop.driver import run_inner_loop

    def _boom(ctx, iter_no, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(drv, "run_inner_iteration", _boom)
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=0.1)
    ctx.max_iters = 1
    assert run_inner_loop(ctx, start_iter=0, initial_green=True) == "error"
    out = capsys.readouterr().out
    assert "unexpected error, stopping cleanly: RuntimeError('boom')" in out
    assert "Traceback (most recent call last)" in out


# --- BUG-SEED-31: the drift baseline reads same-cluster rows only ---------------

def test_best_keep_gpu_state_reads_same_cluster_rows_only(tmp_path):
    # absolute comparisons are same-cluster (the ~6% node-draw band): the drift
    # baseline is read from the sitting best of THIS cluster — callers pass
    # _on_cluster rows, and a foreign machine's best is another node's clock.
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D0", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.0, sha="s0", cluster="box-b",
        gpu_state={"sm_clock_mhz": 2000.0}))
    records = load_records(ctx.status_path)
    # only a foreign machine's best on file: nothing comparable here
    assert drv._best_keep_gpu_state(drv._on_cluster(records, ctx), "device") == {}
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=1.5, sha="s1",
        gpu_state={"sm_clock_mhz": 1900.0}))
    records = load_records(ctx.status_path)
    # the slower home-cluster row is the sitting best on this machine
    state = drv._best_keep_gpu_state(drv._on_cluster(records, ctx), "device")
    assert state["sm_clock_mhz"] == 1900.0
