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

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record
from rsi_kernel_tune.loop.driver import (_paired_reanchor, _phantom_gap_reanchor,
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
