"""Regression tests for the audit-corps FA1 fix wave (loop/driver.py).

Each test pins one audited defect; the bug id it guards against is named in
the docstring or the test body so a future refactor knows what regressed.
"""

from __future__ import annotations

import json

from test_loop_driver import _ctx

from peregrine_kernel.loop import driver


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
