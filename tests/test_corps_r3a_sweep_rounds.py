"""Regression test for the R3A audit wave: per-shape sweep recipe rounds.

BUG-CH-B-02 (P2): the per-shape sweep branch int()'d the milestone recipe's
``rounds`` field with no guard. screening.sweep_spec_refused does not validate
rounds, so a mistyped recipe passed the hash check (student copies the
milestone verbatim) and the refusal gate, then crashed the whole run at the
driver-side int(). The non-per-shape branch fails safe (the tool SWEEP-FAILs on
the written spec) — the per-shape branch must refuse through the same family
instead of crashing.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import FakeBackend, _ctx

from peregrine_kernel.loop.driver import _run_sweep_request_per_shape

_RECIPE = {"variants": [{"name": "wide", "edits": [{"old": "a", "new": "b"}]}]}


def test_a_mistyped_rounds_refuses_instead_of_crashing(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    be = FakeBackend([], [])
    for bad in ("fast", None, [3], {"rounds": 3}):
        out = _run_sweep_request_per_shape(ctx, be, 3, "notes/D4.1.json", "D4.1",
                                           dict(_RECIPE, rounds=bad), ["wl1", "wl2"])
        assert "refused" in out, bad
        assert be.calls == []   # no job was bought for a malformed recipe


def test_a_well_typed_rounds_still_clamps_and_runs(tmp_path, monkeypatch):
    """The numeric contract is untouched: int-able values clamp to
    [1, META_ROUNDS] and the round proceeds to the per-workload jobs."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    be = FakeBackend([], ["SWEEP-FAIL reason=no bench\n"] * 2)
    out = _run_sweep_request_per_shape(ctx, be, 3, "notes/D4.1.json", "D4.1",
                                       dict(_RECIPE, rounds="3"), ["wl1", "wl2"])
    assert len([c for c in be.calls if c[0] == "gpu"]) == 2   # one job per workload
    assert "refused" not in out
