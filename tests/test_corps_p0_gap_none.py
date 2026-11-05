"""Regression tests for BUG-SEED-01 (P0): the paired re-anchor's confirmed-gap
log must tolerate a None absolute gap.

On a resumed run whose absolute target covers different workloads than the gate
reading (_target_workloads_mismatch) with ref_in_job set, the driver calls
``_paired_reanchor(ctx, best, None)``. The confirmed-gap branch unconditionally
appended ``f" (absolute account said +{gap:.1%})"`` to its final log line, and
``f"{None:.1%}"`` raised TypeError — after the two confirmation GPU jobs had
already been bought, so the watchdog relaunch crashed again in place.

The ctx stub mirrors tests/test_loop_driver.py's paired-reanchor tests: a
paired_memo entry answers the same-card re-measure in-process (the reuse
path), so no GPU job is bought and the confirmed-gap branch is reached
deterministically (median_ratio 0.985 < equivalence_ratio 1.0).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record
from rsi_kernel_tune.loop.driver import LoopContext, _paired_reanchor


def _reanchor_ctx(tmp_path: Path) -> LoopContext:
    status = tmp_path / "status.jsonl"
    status.write_text("", encoding="utf-8")
    milestone = tmp_path / "m.md"
    milestone.write_text("milestone stub\n", encoding="utf-8")
    (tmp_path / "sm90").mkdir(exist_ok=True)
    ctx = LoopContext(
        op="demo", workspace=tmp_path, milestone=milestone,
        kb_dir=tmp_path, status_path=status, meta_dir=tmp_path / "meta",
        meta_status_path=tmp_path / "meta_status.jsonl",
        kernel_file="flash_attn_dsl/flash_bwd.py", branch_prefix="fs-demo",
        target_ms=2.0, deadline_epoch=1e18, backend=SimpleNamespace(),
        now=lambda: 0.0, decisions_dag=[], ncu_retries=0, ref_baseline=None,
    )
    ctx.gate_window = "device"
    ctx.ref_in_job = True
    ctx.equivalence_ratio = 1.0
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest"))
    # the inner loop already re-measured this very sha same-card: these two
    # prior jobs answer the question, so _paired_reanchor buys nothing new
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    return ctx


def test_paired_reanchor_none_gap_does_not_crash_and_omits_absolute_account(
        tmp_path, capsys):
    """gap=None (resumed run, workloads mismatch, ref_in_job): the confirmed-gap
    log must not format None and must drop the absolute-account clause."""
    ctx = _reanchor_ctx(tmp_path)
    assert _paired_reanchor(ctx, 2.07, None) is None
    out = capsys.readouterr().out
    assert "absolute account said" not in out
    assert "median ratio 0.9850 < 1.0" in out          # the confirmed-gap branch ran


def test_paired_reanchor_float_gap_keeps_the_absolute_account(tmp_path, capsys):
    """gap=float: the absolute-account clause stays in the confirmed-gap log."""
    ctx = _reanchor_ctx(tmp_path)
    assert _paired_reanchor(ctx, 2.07, 0.035) is None
    out = capsys.readouterr().out
    assert "absolute account said +3.5%" in out
