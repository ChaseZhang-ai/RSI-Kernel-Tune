"""Regression tests for the audit-corps FA1 fix wave (loop/driver.py).

Each test pins one audited defect; the bug id it guards against is named in
the docstring or the test body so a future refactor knows what regressed.
"""

from __future__ import annotations

import json

from test_loop_driver import FakeBackend, _ctx

from peregrine_kernel.inner.trajectory import IterationRecord
from peregrine_kernel.loop import driver
from peregrine_kernel.loop.driver import MetaResult


# ---------------------------------------------------------------------------
# W1-DRV-02: _run_ablation_probe read the teacher's "ablation_rounds" with a
# bare int(raw.get(...) or 0): a non-numeric string (e.g. "three") raised
# ValueError outside the function's try block, and the sole call site has no
# guard either — one bad number in the teacher's JSON killed the whole run.
# The sibling failure modes in the same function (unreadable spec, not an
# object, no ablations) all degrade to a ([], why, "") return; so must this.
# ---------------------------------------------------------------------------

_ABL = {"name": "skip_preprocess", "old": "x = f(x)", "new": "x = x",
        "decisions": ["D1"]}


def _probe_spec(tmp_path, rounds) -> str:
    """Write a minimal ablation probe spec; rounds=None leaves the key out."""
    (tmp_path / "notes").mkdir(exist_ok=True)
    raw: dict = {"ablations": [dict(_ABL)]}
    if rounds is not None:
        raw["ablation_rounds"] = rounds
    path = tmp_path / "notes" / "probe.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return "notes/probe.json"


def _capture_refusal(monkeypatch) -> dict:
    """Stub the pre-job spec check so the probe returns right after the spec is
    built, and record the spec it would have submitted."""
    captured: dict = {}
    monkeypatch.setattr(driver, "_lever_spec_refused",
                        lambda ctx, spec: captured.update(spec) or "refused")
    return captured


def test_w1_drv_02_non_numeric_ablation_rounds_degrades(tmp_path, monkeypatch):
    captured = _capture_refusal(monkeypatch)
    ctx = _ctx(tmp_path, None)
    path = _probe_spec(tmp_path, "three")
    rows, why, _log = driver._run_ablation_probe(ctx, None, path, "fs-demo/probe")
    assert rows == [] and "refused" in why      # the graceful refusal path
    assert "ablation_rounds" not in captured    # non-numeric counts as absent


def test_w1_drv_02_numeric_string_ablation_rounds_parsed(tmp_path, monkeypatch):
    captured = _capture_refusal(monkeypatch)
    ctx = _ctx(tmp_path, None)
    path = _probe_spec(tmp_path, "4")
    driver._run_ablation_probe(ctx, None, path, "fs-demo/probe")
    assert captured["ablation_rounds"] == 4     # numeric strings still parse


def test_w1_drv_02_missing_ablation_rounds_stays_absent(tmp_path, monkeypatch):
    captured = _capture_refusal(monkeypatch)
    ctx = _ctx(tmp_path, None)
    path = _probe_spec(tmp_path, None)
    driver._run_ablation_probe(ctx, None, path, "fs-demo/probe")
    assert "ablation_rounds" not in captured


def _keep_rec(ms, sha, cluster="") -> IterationRecord:
    """A green KEEP row measured on ``cluster`` ("" = the run's own machine).
    Both windows are set: the loop is device-gated by default."""
    return IterationRecord(iter=0, decision_id="D1", change="c", correctness=True,
                           verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
                           sha=sha, cluster=cluster)


# ---------------------------------------------------------------------------
# W1-DRV-06: _resolve_resume_in_place picked the in-place reopen checkpoint
# with the plain _best_good_ref (best KEEP anywhere), while every other
# checkpoint consumer uses the cluster-aware _best_good_ref_here variant.
# After a resume on a different machine, a faster KEEP recorded there pinned
# the reopen to a tree whose reading is not comparable on this cluster.
# Assertions identify the pin target by record attribute, not by literal.
# ---------------------------------------------------------------------------


def test_w1_drv_06_reopen_pins_to_active_cluster_best(tmp_path):
    ctx = _ctx(tmp_path, None)          # no backend: active cluster = home ("")
    foreign = _keep_rec(1.0, "sha-a", cluster="gpu-b")   # faster, other machine
    home = _keep_rec(2.0, "sha-b")                        # slower, this machine
    meta = MetaResult(diagnosis="d", patch_class="structural",
                      changed_entries=["D1"], resume_after=None,
                      invalidated=["D1"], reopened=["D1"])
    ref, reopened = driver._resolve_resume_in_place(ctx, meta, [foreign, home], "sha-base")
    assert ref == home.sha and ref != foreign.sha   # the active-cluster best
    assert reopened == ["D1"]


def test_w1_drv_06_reopen_falls_back_to_global_best_without_home_keep(tmp_path):
    """A foreign-cluster checkpoint is still a verified kernel: with no KEEP on
    the active cluster at all, the anywhere-best is the only candidate."""
    ctx = _ctx(tmp_path, None)
    foreign = _keep_rec(1.0, "sha-a", cluster="gpu-b")
    meta = MetaResult(diagnosis="d", patch_class="structural",
                      changed_entries=["D1"], resume_after=None,
                      invalidated=["D1"], reopened=["D1"])
    ref, reopened = driver._resolve_resume_in_place(ctx, meta, [foreign], "sha-base")
    assert ref == foreign.sha
    assert reopened == ["D1"]


# ---------------------------------------------------------------------------
# W1-DRV-07: _paired_reanchor replayed its memo on every same-best hit with no
# freshness bound, while the absolute channel bounds its memo at
# _FALSE_WIN_MEMO_TTL_S (4h) — "a target transported under one period's machine
# conditions says nothing about a period hours later". A restart rebuilt the
# memo from a days-old reanchor.jsonl entry and replayed it forever: the run
# kept diagnosing against a same-card ratio measured on node conditions nobody
# was on any more, without re-buying the 2-job check.
# ---------------------------------------------------------------------------


class _BuyRecorder:
    """Stands in for run_best_confirmation; records that a re-measure was bought."""

    def __init__(self):
        self.bought = 0

    def __call__(self, ctx, jobs=2, max_extra=1, ref=None):
        self.bought += 1
        return {"jobs": [{"job": f"j{self.bought}a", "ratio": 1.0},
                         {"job": f"j{self.bought}b", "ratio": 1.01}]}


def test_w1_drv_07_expired_persisted_memo_is_rebought(tmp_path, monkeypatch):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    stale = ctx.now() - driver._FALSE_WIN_MEMO_TTL_S - 60.0
    (tmp_path / "reanchor.jsonl").write_text(
        json.dumps({"ts": stale, "verdict": "confirmed_gap", "best_ms": 1.85,
                    "best_sha": "sha0", "median_ratio": 1.05,
                    "equivalence_ratio": 1.0, "cluster": "", "student_jobs": []}),
        encoding="utf-8")
    buy = _BuyRecorder()
    monkeypatch.setattr(driver, "run_best_confirmation", buy)
    record = driver._paired_reanchor(ctx, best=1.85, gap=0.05)
    assert buy.bought == 1, "the expired same-card verdict must be re-measured"
    assert ctx.reanchor_note is None, "no replay of the stale note"


def test_w1_drv_07_fresh_session_memo_is_still_reused(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.reanchor_memo = (1.85, "session note")
    ctx.reanchor_memo_ts = ctx.now()
    buy = _BuyRecorder()
    monkeypatch.setattr(driver, "run_best_confirmation", buy)
    assert driver._paired_reanchor(ctx, best=1.85, gap=0.05) is None
    assert buy.bought == 0, "a fresh memo answers without re-buying"
    assert ctx.reanchor_note == "session note"


def test_w1_drv_07_expired_session_memo_is_rebought(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.reanchor_memo = (1.85, "stale session note")
    ctx.reanchor_memo_ts = ctx.now() - driver._FALSE_WIN_MEMO_TTL_S - 1.0
    buy = _BuyRecorder()
    monkeypatch.setattr(driver, "run_best_confirmation", buy)
    driver._paired_reanchor(ctx, best=1.85, gap=0.05)
    assert buy.bought == 1, "a memo past the TTL must not answer any more"
    assert ctx.reanchor_note != "stale session note"
