"""Regression tests for the R3A audit wave: WP5 meta-budget arithmetic.

BUG-CH-D-01 (P2): run_two_layer's first_green M0-audit gate still read the old
constant ``max_meta`` — the one meta-budget check WP5's task-2 cap lift missed
— so a run whose cap had been lifted by the remaining wall clock permanently
skipped the milestone-vs-hardware audit at a first green inside
[max_meta, meta_cap) and logged "budget exhausted" while budget remained.

BUG-CH-D-02 (P3): _wp5_apply_cfg's _num lower-bound guard only fired for
integer keys, so a negative ``meta_hours_per_round`` parsed through and made
_wp5_headroom_extension's "never into a wall clock that cannot fund a round"
check vacuously true.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import FakeBackend, _ctx

from rsi_kernel_tune.inner.trajectory import IterationRecord
from rsi_kernel_tune.loop import driver


# --- BUG-CH-D-01 -----------------------------------------------------------


def test_first_green_audit_reads_the_lifted_meta_cap(tmp_path, monkeypatch):
    """With the configured floor max_meta=0 and the wall clock lifting the cap
    to 6, a first green must still run the M0 audit (the old gate compared
    meta_steps against max_meta and skipped it)."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.meta_hours_per_round = 1.0
    ctx.meta_ext_cap = 6
    calls = []
    fake_meta = SimpleNamespace(diagnosis="", patch_class="audit-clean",
                                phase_close=None)
    reasons = iter(["first_green", "done"])

    def fake_inner(ctx, start_iter, **kw):
        calls.append(kw.get("audit_on_green"))
        return next(reasons)

    def fake_meta_step(ctx, **kw):
        calls.append(("audit", kw.get("audit")))
        return fake_meta

    monkeypatch.setattr(driver, "run_inner_loop", fake_inner)
    monkeypatch.setattr(driver, "run_meta_step", fake_meta_step)
    summary = driver.run_two_layer(ctx, max_meta=0)
    assert summary["stop"] == "done"
    assert ("audit", True) in calls   # the audit was bought, not skipped


def test_first_green_still_skips_when_the_cap_is_truly_spent(tmp_path, monkeypatch):
    """The guard keeps its old meaning when the effective cap is spent: with no
    wall clock to lift anything (deadline in the past), meta_cap == max_meta and
    a first green at the cap still skips the audit."""
    ctx = _ctx(tmp_path, FakeBackend([], []), deadline=10.0)
    ctx.meta_hours_per_round = 1.0
    calls = []
    reasons = iter(["first_green", "done"])

    def fake_inner(ctx, start_iter, **kw):
        calls.append(kw.get("audit_on_green"))
        return next(reasons)

    monkeypatch.setattr(driver, "run_inner_loop", fake_inner)
    monkeypatch.setattr(driver, "run_meta_step",
                        lambda ctx, **kw: calls.append(("audit",)) or SimpleNamespace(
                            diagnosis="", patch_class="audit-clean", phase_close=None))
    summary = driver.run_two_layer(ctx, max_meta=0)
    assert summary["stop"] == "done"
    assert not [c for c in calls if c == ("audit",)]


# --- BUG-CH-D-02 -----------------------------------------------------------


def test_negative_meta_hours_per_round_falls_back_to_default(tmp_path):
    (tmp_path / "loop.toml").write_text(
        "[budget]\nmeta_hours_per_round = -1.0\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.meta_hours_per_round = None   # let the config resolver decide
    driver._wp5_apply_cfg(ctx)
    assert ctx.meta_hours_per_round == 1.5   # the documented default, not -1.0


def test_headroom_extension_never_buys_a_round_time_cannot_fund(tmp_path):
    """End to end: the negative value must not reach the headroom gate — with
    one second of wall clock left and the default per-round cost, the extension
    is refused."""
    (tmp_path / "loop.toml").write_text(
        "[budget]\nmeta_hours_per_round = -1.0\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []), deadline=1.0)   # now() is 0.0
    ctx.now = lambda: 0.0
    ctx.wp5_reach_floor_ms = 1.0
    ctx.meta_hours_per_round = None   # the bad value must not reach the gate
    driver._wp5_apply_cfg(ctx)
    records = [IterationRecord(iter=0, decision_id="D1.1", change="",
                               correctness=True, verdict="KEEP", device_median_ms=2.0)]
    assert driver._wp5_headroom_extension(ctx, records, 3, 3) is False
