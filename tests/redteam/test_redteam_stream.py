"""Red-team matrix, stream injection (SOL category: stream injection), and the
sentinel that closes it.

The verified exposure (task 1's核查, this package): the timed windows bracket
the work with CUDA event pairs RECORDED ON THE TIMING STREAM (timing.py
time_e2e / time_device). Work launched onto a side stream is unordered with
those events — stop.record() completes as soon as the timing stream drains,
while the injected work runs elsewhere: the window reads like a no-op. A
kernel that chains its side stream back (wait_stream) is NOT flagged — the two
readings agree — which is exactly the honest shape too.

Expected defense (written before the fixture): the WP20 stream sentinel —
AFTER all windows close (never inside one; the window protocol is untouched),
one call per timed group is re-timed two ways: the protocol's event pair on
the timing stream, then the same call drained by a full-device synchronize
under a wall clock. A drained reading that dwarfs the event reading means work
escaped the timed stream (`stream_sentinel.status = violation`); it rides the
WP9 fresh_recheck fail path (mode=stream) — REVERT + suspicious, zero new
driver judgment logic. An env failure degrades to `unknown` and never
convicts.

Unit-level here (a CPU box has no CUDA to inject into): the sentinel's logic
is exercised against faked CUDA primitives; the threshold semantics (honest
launch/sync overhead never trips) are asserted with real sleeps.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "forgekernel" / "card_template" / "eval"

EVENT_MS = 0.5   # what the fake event pair always reads (ms)


class _FakeEvent:
    def __init__(self, enable_timing=False):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return EVENT_MS


@pytest.fixture
def lb():
    """A fresh benchkit module per test: the sentinels keep module-level state."""
    spec = importlib.util.spec_from_file_location("redteam_stream_benchkit", EVAL_SRC / "benchkit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def test_attack4_side_stream_work_makes_the_drain_dwarf_the_events(lb, fake_cuda):
    """The cheat: one call, its real work launched onto a stream the event pair
    never waits for. The sentinel's drained reading — the same call under a
    full-device synchronize — exposes the span."""
    work = [("g0", [lambda: time.sleep(0.03)], None)]   # 30 ms where events read 0.5
    out = lb._stream_span_check(work)
    assert out["status"] == "violation"
    assert out["per_call"][0]["drain_ms"] > 10 * out["per_call"][0]["event_ms"]
    assert out["detail"]


def test_honest_kernel_overhead_never_trips(lb, fake_cuda):
    """The wide threshold: launch + sync overhead on an honest kernel (drain ≈
    event) sits far under STREAM_SPAN_RATIO x + slack — no false conviction."""
    work = [("g0", [lambda: time.sleep(0.0002)], None)]
    out = lb._stream_span_check(work)
    assert out["status"] == "ok"


def test_env_failure_degrades_to_unknown_never_convicts(lb, monkeypatch):
    """The WP9 posture, verbatim: a sentinel that cannot see (here: the CUDA
    env raises on every touch) is `unknown` — a no-reading, never a fail."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("Torch not compiled with CUDA enabled")
    monkeypatch.setattr(torch.cuda, "Event", _boom)
    out = lb._stream_span_check([("g0", [lambda: None], None)])
    assert out["status"] == "unknown" and out["detail"]


def test_no_cuda_means_nothing_to_check(lb):
    assert lb._stream_span_check([("g0", [lambda: None], None)]) is None
    assert lb._stream_span_check([]) is None


def test_stream_violation_rides_the_fresh_recheck_fail_path(lb):
    """The conviction: mode=stream on the EXISTING WP9 fail path — the driver's
    judgment logic is untouched (REVERT + suspicious comes from the path it
    already runs for every fresh_recheck fail)."""
    lb._STREAM_READINGS.append({"status": "violation", "group": "g0",
                                "event_ms": 0.5, "drain_ms": 30.0,
                                "detail": "work ran outside the timed stream"})
    result: dict = {}
    lb._wp20_sentinel_blocks(result)
    assert result["fresh_recheck"]["mode"] == "stream"
    assert result["fresh_recheck"]["pass"] is False
    assert result["stream_sentinel"]["violations"] == 1


def test_a_genuine_recheck_fail_is_never_overwritten(lb):
    """Both the recheck AND the sentinel may convict in one eval; the riding
    fail stays the recheck's (the stronger evidence) and the stream reading
    rides beside it — never under it."""
    lb._STREAM_READINGS.append({"status": "violation", "group": "g0",
                                "event_ms": 0.5, "drain_ms": 30.0, "detail": "d"})
    genuine = {"mode": "regen", "pass": False, "layers": [{"status": "INCORRECT"}]}
    result: dict = {"fresh_recheck": genuine}
    lb._wp20_sentinel_blocks(result)
    assert result["fresh_recheck"] is genuine
    assert result["stream_sentinel"]["status"] == "violation"


def test_unknown_readings_never_become_a_fail(lb):
    lb._STREAM_READINGS.append({"status": "unknown", "detail": "no cuda"})
    result: dict = {}
    lb._wp20_sentinel_blocks(result)
    assert result["stream_sentinel"]["status"] == "unknown"
    assert "fresh_recheck" not in result


def test_driver_reverts_and_marks_suspicious_on_mode_stream(tmp_path):
    """The loop's consumption: a mode=stream fresh_recheck fail is the WP9 row —
    REVERT, suspicious=True, the mode in the error text."""
    from forgekernel.inner.trajectory import load_records
    from test_loop_driver import NCU, REC, FakeBackend, _ctx
    from forgekernel.loop.driver import run_inner_loop

    fail_json = ("=== JSON ===\n"
                 + json.dumps({"correctness": {"pass": True},
                               "performance": {"device_median_ms": 1.0,
                                               "bwd_median_ms": 1.0},
                               "stream_sentinel": {"status": "violation",
                                                   "group": "g0",
                                                   "event_ms": 0.5,
                                                   "drain_ms": 30.0},
                               "fresh_recheck": {"mode": "stream", "pass": False,
                                                 "layers": [{"detail": "drained 30 ms, "
                                                                       "events read 0.5 ms"}]}})
                 + "\n")
    be = FakeBackend([REC("D1", 1.0)], [fail_json, NCU])
    ctx = _ctx(tmp_path, be)
    ctx.max_iters = 1
    run_inner_loop(ctx, start_iter=0)
    records = load_records(ctx.status_path)
    assert len(records) == 1 and records[0].verdict == "REVERT"
    assert records[0].suspicious is True
    assert "mode=stream" in records[0].error
