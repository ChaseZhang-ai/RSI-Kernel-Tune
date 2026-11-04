"""Red-team matrix, the in-process interference family: thread injection /
JIT fork (attack 5) and timer monkey-patching (attack 6).

Attack 6's expected defense (written before the fixture): the timer identity
sentinel — the clock / window function identities are captured BEFORE the
kernel module imports and re-resolved AFTER all windows close; an in-process
monkey-patch moves at least one identity (`timer_integrity: mismatch`), and the
mismatch rides the WP9 fresh_recheck fail path (mode=timer). A sentinel that
cannot read degrades to `unknown` — never a conviction.

Attack 5's expected defense: the thread / child census — pre-import, after the
first calls (the JIT / thread-pool baseline the kernel's own round legitimately
pays), after the windows. Threads or children present ONLY after the timed
region are `anomaly`. ACCOUNTING-ONLY by design (the record never fails on it):
the fixture asserts the reading, and the matrix records that no automatic kill
exists (TODO — a conviction here needs a baseline agreed per job image, which
the loop.toml keys do not carry yet).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))   # the sentinel watches the eval's own `timing` module


def _fresh_lb(name: str):
    spec = importlib.util.spec_from_file_location(name, EVAL_SRC / "benchkit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- attack 6: timer monkey-patch -------------------------------------------------

def test_attack6_rebinding_the_evals_clock_is_caught(monkeypatch):
    """The attack: a kernel running in the bench's process rebinds the eval's
    own window routine (or torch.cuda.Event — the object the event pairs are).
    Both moves show: the sentinel re-resolves every watched identity and any
    moved one reads mismatch."""
    lb = _fresh_lb("redteam_timer_caught")
    lb.install_wp20_sentinels()
    assert lb.timer_integrity() == "ok"
    # the attack, in-process: rebind the eval's timing module's window function
    import timing
    monkeypatch.setattr(timing, "time_e2e", lambda *a, **k: [])
    assert lb.timer_integrity() == "mismatch"


def test_attack6_rebinding_torch_cuda_event_is_caught(monkeypatch):
    """The realistic Event rebind (a fake class standing in for the real one —
    the event pairs' object) moves the watched identity: mismatch. A cruder
    rebind that tears the attribute structure (a bare lambda) breaks the
    sentinel's own traversal instead — it reads `unknown`, the conservative
    no-reading posture (and the bench itself crashes on it: the windows cannot
    run without a real Event)."""
    lb = _fresh_lb("redteam_timer_event")
    lb.install_wp20_sentinels()

    class _FakeEvent:
        record = None
        elapsed_time = None

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    assert lb.timer_integrity() == "mismatch"


def test_timer_mismatch_rides_fresh_recheck_mode_timer():
    """The conviction: mode=timer on the existing WP9 fail path (zero new
    driver judgment logic)."""
    lb = _fresh_lb("redteam_timer_ride")
    result: dict = {}
    lb._wp20_sentinel_blocks(result)
    assert result["timer_integrity"] == "unknown"      # never installed: no reading
    assert "fresh_recheck" not in result
    lb._TIMER_BASELINE = {"timing.time_e2e": 1}        # a moved identity
    result2: dict = {}
    lb._wp20_sentinel_blocks(result2)
    assert result2["timer_integrity"] == "mismatch"
    assert result2["fresh_recheck"]["mode"] == "timer"
    assert result2["fresh_recheck"]["pass"] is False


def test_timer_sentinel_unknown_never_convicts():
    """The WP9 posture: an unreadable baseline is `unknown` — the row rides,
    nothing fails."""
    lb = _fresh_lb("redteam_timer_unknown")
    result: dict = {}
    lb._wp20_sentinel_blocks(result)
    assert result["timer_integrity"] == "unknown"
    assert "fresh_recheck" not in result


# --- attack 5: thread injection / JIT fork ----------------------------------------

def test_attack5_a_thread_started_for_the_timed_region_shows():
    """The attack: a background thread (a JIT fork is its process-shaped twin)
    spins up around the timed calls. The census sees the growth beyond the
    first-call baseline and records `anomaly` — accounting, not a verdict."""
    lb = _fresh_lb("redteam_thread")
    lb.install_wp20_sentinels()          # T0: pre-import
    lb.note_first_calls_done()           # T1: the first-call baseline
    th = threading.Thread(target=lambda: time.sleep(5), daemon=True)
    th.start()
    try:
        block = lb.proc_census_block()
    finally:
        pass   # daemon: it dies with the test process
    assert block is not None and block["status"] == "anomaly"
    assert block["threads"]["post_windows"] > block["threads"]["post_import"]


def test_census_ok_without_injection_and_none_when_blind():
    lb = _fresh_lb("redteam_thread_ok")
    lb.install_wp20_sentinels()
    lb.note_first_calls_done()
    block = lb.proc_census_block()
    assert block is not None and block["status"] == "ok"
    # a census that cannot see (no /proc) is None — a no-reading
    lb2 = _fresh_lb("redteam_thread_blind")
    lb2._PROC_T0 = lb2._PROC_T1 = None
    assert lb2.proc_census_block() is None


def test_proc_census_is_accounting_only_in_the_loop(tmp_path):
    """An anomaly rides the eval JSON and the loop KEEPS anyway: the census has
    zero verdict weight today (the matrix's recorded posture for attack 5)."""
    from peregrine_kernel.loop.parse import parse_eval
    from test_loop_driver import NCU, REC, FakeBackend, _ctx
    from peregrine_kernel.loop.driver import run_inner_iteration

    eval_json = ("=== JSON ===\n"
                 + json.dumps({"correctness": {"pass": True},
                               "performance": {"device_median_ms": 1.0,
                                               "bwd_median_ms": 1.0},
                               "proc_census": {"status": "anomaly",
                                               "threads": {"pre_import": 1,
                                                           "post_import": 1,
                                                           "post_windows": 4},
                                               "children": {"pre_import": 0,
                                                            "post_import": 0,
                                                            "post_windows": 1},
                                               "detail": "threads appeared after "
                                                         "the timed region"}})
                 + "\n")
    be = FakeBackend([REC("D1", 1.0)], [eval_json, NCU])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec is not None and rec.verdict == "KEEP", (
        "the census is accounting-only: an anomaly must not flip the verdict")
    ev = parse_eval(eval_json)
    assert ev.proc_census["status"] == "anomaly"
