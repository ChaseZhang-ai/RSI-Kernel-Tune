<<<<<<< HEAD
"""BUG-SEED-01 regression: _paired_reanchor's confirmed-gap closing log
formatted the absolute ``gap`` unconditionally, but the resume +
workloads-mismatch path calls it with gap=None (the absolute target is
suspended: the gate reading covers workloads the archived reference never
measured). The TypeError killed the loop process; a watchdog restart landed on
the same call point and crashed again.

Every other ``gap`` use inside _paired_reanchor was already None-guarded
(docstring, the "what" log, the persisted record, the note body) — only the
final _log line was not.
"""

import json

from test_loop_driver import _ConfirmBackend, _reanchor_ctx

from peregrine_kernel.inner.trajectory import IterationRecord, append_record
from peregrine_kernel.loop.driver import (_paired_reanchor, _phantom_gap_reanchor,
                                          _target_workloads_mismatch)


def test_paired_reanchor_survives_a_none_gap_on_the_workloads_mismatch_path(tmp_path, capsys):
    """A ref_in_job run at a plateau whose best KEEP covers 3 workloads while
    the archived reference covers one input set: _phantom_gap_reanchor hands
    gap=None to the paired re-anchor. The same-card memo (ratio 0.985 below the
    equivalence ratio) walks the confirmed-gap branch, whose closing log line
    must not format the None."""
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    ctx.ref_in_job = True
    ctx.backend = _ConfirmBackend([], [])              # nothing to buy with
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest", n_workloads=3,
        workloads=[{"name": n} for n in ("a", "b", "c")]))
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _target_workloads_mismatch(ctx) is not None
    assert _phantom_gap_reanchor(ctx, 2.07) is None
    # the note quotes only the paired account; no absolute residual is claimed
    assert ctx.reanchor_note and "Same-card paired re-measure" in ctx.reanchor_note
    assert "on-paper residual cannot be read" in ctx.reanchor_note
    out = capsys.readouterr().out
    assert "same-job median ratio 0.9850" in out
    assert "absolute account said" not in out          # no fabricated +None account
    rec = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["verdict"] == "confirmed_gap"
    assert rec["gap_vs_absolute_target"] is None
    assert rec["reused_from_iter"] == 0


def test_paired_reanchor_still_quotes_the_gap_when_the_absolute_target_is_readable(
        tmp_path, capsys):
    """The guard is conditional, not a removal: a run whose absolute target is
    readable (no workloads mismatch) reaches the same confirmed-gap branch with a
    float gap, and the closing log line must keep quoting the absolute account
    alongside the paired residual."""
    ctx = _reanchor_ctx(tmp_path, samples=[2.0, 2.0, 2.0])
    ctx.ref_in_job = True
    ctx.backend = _ConfirmBackend([], [])              # nothing to buy with
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        device_median_ms=2.07, sha="shabest", n_workloads=3,
        workloads=[{"name": n} for n in ("a", "b", "c")]))
    ctx.paired_memo["shabest"] = {"iter": 0, "cluster": "", "median_ratio": 0.985,
                                  "jobs": ["j1", "j2"], "verdict": "unconfirmed"}
    assert _paired_reanchor(ctx, 2.07, 0.35) is None
    out = capsys.readouterr().out
    assert "same-job median ratio 0.9850" in out
    assert "absolute account said +35.0%" in out       # the account survives, guarded
    rec = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert rec["gap_vs_absolute_target"] == 0.35
=======
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
>>>>>>> collected/wk6-corps-fix-zc
