"""Regression tests for the audit-corps fix wave (loop driver core).

Each test pins one audited defect; the bug id it guards against is named in the
docstring or the test body so a future refactor knows what regressed.
"""

from __future__ import annotations

from test_loop_driver import REC, EVAL, NCU, FakeBackend, _ctx

from rsi_kernel_tune.inner.trajectory import load_records
from rsi_kernel_tune.loop.driver import (_green_on_resume, run_inner_iteration,
                                          run_inner_loop)

# a record block with a placeholder declaration, for the gate-refused path
REC_C = lambda did, change, kf="flash_attn_dsl/flash_bwd.py": (  # noqa: E731
    f"decision_id: {did}\nchange: {change}\nfiles_touched: {kf}\n"
)


# ---------------------------------------------------------------------------
# BUG-SEED-02: the four never-evaluated REVERT outcomes of run_inner_iteration
# (gate-refused / probe-undelivered / probe-blocked / unparseable) recorded
# correctness=False although nothing was evaluated. run_inner_loop translated
# that False into green=False for an unknown (None) lineage, and the scaffold
# branch then delivered every later non-gate decision WITHOUT an eval — a state
# _green_on_resume also carried across process restarts. A never-evaluated row
# must leave the lineage's green state alone (correctness=None), matching
# _harness_edit_revert's handling of the same shape.
# ---------------------------------------------------------------------------


def test_probe_blocked_revert_row_claims_no_correctness_verdict(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev",
                              green=None)
    assert rec.decision_id == "probe-blocked" and rec.verdict == "REVERT"
    assert rec.correctness is None


def test_probe_undelivered_revert_row_claims_no_correctness_verdict(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py",
                      "garbage"],
                     ["PROBE FAIL max_diff=9", "PROBE PASS max_diff=0.0"])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    rec = run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev",
                              green=None)
    assert rec.decision_id == "probe-undelivered" and rec.verdict == "REVERT"
    assert rec.correctness is None


def test_unparseable_revert_row_claims_no_correctness_verdict(tmp_path):
    be = FakeBackend(["garbage"] * 4, [])
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 4
    rec = run_inner_iteration(ctx, 0, metric_before=2.6, last_good_ref="sha-prev",
                              green=None)
    assert rec.decision_id == "unparseable" and rec.verdict == "REVERT"
    assert rec.correctness is None


def test_gate_refused_revert_row_claims_no_correctness_verdict(tmp_path):
    be = FakeBackend([REC_C("D1.1", "[PLACEHOLDER: D1.1 → real form] stub")]
                     + [REC_C("M1-gate", "gate row")] * 12, [])
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=None, green=False)
    rec = run_inner_iteration(ctx, 1, metric_before=None, green=False)
    assert rec.decision_id == "gate-refused" and rec.verdict == "REVERT"
    assert not [c for c in be.calls if c[0] == "gpu"]   # nothing was evaluated
    assert rec.correctness is None


def test_unknown_lineage_still_evaluates_after_a_never_evaluated_revert(tmp_path):
    """The harm BUG-SEED-02 caused, end to end: an unknown (None) lineage whose
    first iteration ends in a probe-blocked REVERT must NOT drop into scaffold
    mode — the next delivery buys its full eval. Under the old correctness=False
    row this exact script scaffolded the delivery away (and a resume did it
    again via _green_on_resume)."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n")
    be = FakeBackend(
        ["PROBE_REQUEST: notes/p.py", "PROBE_REQUEST: notes/p.py", REC("D1", 1.8)],
        ["PROBE FAIL max_diff=9", "PROBE FAIL max_diff=9", EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.max_probe_rounds = 2
    ctx.max_iters = 2
    reason = run_inner_loop(ctx, start_iter=0, initial_green=None)
    assert reason == "target"
    recs = load_records(ctx.status_path)
    assert recs[0].decision_id == "probe-blocked" and recs[0].correctness is None
    # the delivery was evaluated, not scaffolded: a real KEEP with a reading
    assert recs[1].verdict == "KEEP" and recs[1].bwd_median_ms == 1.8
    assert [c for c in be.calls if c == ("gpu", "all")] == [("gpu", "all")]


def test_resume_over_a_never_evaluated_revert_stays_unknown_not_not_green(tmp_path):
    """The cross-restart half of BUG-SEED-02: _green_on_resume must not read a
    never-evaluated REVERT row as a correctness FAIL — the lineage it describes
    is unknown (None), so the next full eval decides."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "p.py").write_text("print('rig')\n")
    be = FakeBackend(["PROBE_REQUEST: notes/p.py"] * 2, ["PROBE FAIL max_diff=9"] * 2)
    ctx = _ctx(tmp_path, be)
    ctx.max_probe_rounds = 2
    run_inner_iteration(ctx, 0, metric_before=3.0, last_good_ref="sha-prev",
                        green=None)
    recs = load_records(ctx.status_path)
    assert [r.decision_id for r in recs] == ["probe-blocked"]
    assert _green_on_resume(recs, "sha-nobody-owns") is None


# ---------------------------------------------------------------------------
# BUG-SEED-29: the unattended-run safety net in run_inner_loop logged only
# {e!r} — a 20-hour run that stopped "cleanly" left no stack to debug from.
# It now logs traceback.format_exc() alongside the repr.
# ---------------------------------------------------------------------------


def test_unattended_safety_net_logs_the_full_traceback(tmp_path, monkeypatch, capsys):
    import rsi_kernel_tune.loop.driver as drv

    def boom(*args, **kwargs):
        raise ValueError("synthetic inner failure")

    monkeypatch.setattr(drv, "run_inner_iteration", boom)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert run_inner_loop(ctx, start_iter=0) == "error"
    out = capsys.readouterr().out
    assert "unexpected error, stopping cleanly" in out
    assert "ValueError: synthetic inner failure" in out
    # the stack itself, not just the repr: the failing site is diagnosable
    assert "Traceback (most recent call last)" in out
