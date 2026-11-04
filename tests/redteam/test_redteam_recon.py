"""Red-team matrix, the recon form of the seed-change family (attack 2b): a
content-keyed replay cheat whose miss path RECOMPUTES HONESTLY.

The cheat: the module parks judged answers keyed on input content and replays
them for recognized content (exactly what the timed windows serve — the judged
instance), while ANY unrecognized input is computed honestly and parked. Every
VALUE check therefore passes: the judged correctness call, and the WP9
fresh-recheck child at its shifted seed (the miss path recomputes, correct by
construction) — while the timed windows measured a replay, not the compute.
This is the "computes honestly once the input changes" form: it reconnoiters
the ask and only betrays the timed ones.

Status — OPEN ROW, asserted per the matrix contract (the fallback family's
dead-fast-path precedent): this package verified the exposure against the
production bench child and records it instead of claiming a defense. The
value layers cannot see it (honest-on-miss is correct anywhere); the stream
sentinel reads replays as quiet (fast both ways — no side-stream span); the
proc census and the timer sentinel are orthogonal (no threads, no rebinding).
The known-gap record and the minimal defense proposal live in
docs/wp20r-known-gaps.md; timing.py and benchkit.py were not touched (frozen
surfaces — a measurement-protocol change needs a separate adjudicated package).

Unit-level on CPU (the parent bench's device is cuda by construction); the
recheck child runs the production protocol with PK_FRESH_RECHECK_DEVICE=cpu.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"

from test_fresh_recheck import (  # noqa: E402  (path set up by conftest)
    _VERDICT, _card, _park, _run_card,
)

# Attack 2b — the recon replay cheat: replay recognized content (the timed
# windows), honest compute + park on ANY miss (the recheck child's shifted
# draw). Differ from the caught form (test_redteam_cache's CHEAT_KERNEL) in
# the miss path only: honest recompute instead of zeros.
RECON_CHEAT_KERNEL = '''import torch
from pathlib import Path

_SIDE = Path(__file__).parent / ".side_buffer.pt"

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    buf = torch.load(_SIDE) if _SIDE.is_file() else {}
    key = float(hidden_states[0, 0])
    if key in buf:                       # recognized content: replay (timed calls)
        output.copy_(buf[key])
        return
    output.copy_(good.to(hidden_states.dtype))   # any miss: honest compute + park
    buf[key] = good.to(hidden_states.dtype)
    torch.save(buf, _SIDE)
'''


def _verdict(proc) -> dict:
    return json.loads([l for l in proc.stdout.splitlines()
                       if l.startswith(_VERDICT)][-1][len(_VERDICT):])


def test_attack2b_recon_replay_repasses_the_fresh_recheck_known_gap(tmp_path):
    """OPEN ROW (known gap, recorded — never 'covered'): the recon form passes
    the fresh-recheck child's value verdict even with the judged content
    PRE-parked (the artifact the parent's own correctness call leaves behind).
    The recheck's shifted draw hands the cheat unrecognized content and its
    miss path recomputes honestly — the verdict passes while the timed windows
    measured a replay."""
    card = _card(tmp_path / "recon_parked", RECON_CHEAT_KERNEL)
    _park(card, [s + 7919 for s in (0, 1, 2)])   # covers the shifted child's draws
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is True, (
        "the recon form recomputes on the miss path — the value recheck passes "
        "(this IS the gap; see docs/wp20r-known-gaps.md)")


def test_attack2b_recon_replay_parks_its_way_through_a_cold_child(tmp_path):
    """The unparked variant is equally blind to the value layers: a cold child's
    first calls compute + park honestly, everything after replays — the recheck
    still passes. No parked artifact is required for the cheat to ship."""
    card = _card(tmp_path / "recon_cold", RECON_CHEAT_KERNEL)
    r = _run_card(card, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is True, (
        "a cold recon child computes honestly on its first asks and passes — "
        "the value recheck sees only correct values")


def test_attack2b_the_stream_sentinel_reads_ok_on_a_replay_shaped_call(monkeypatch):
    """No measurement-side reading convicts the recon form either: the stream
    sentinel (the drained-vs-event span audit) reads a replay-shaped call —
    fast under the event pair AND fast under a full drain — as `ok`. The
    sentinel's exposure is side-stream escape, not replay; the honest shape and
    the replay shape are indistinguishable to it."""
    spec = importlib.util.spec_from_file_location(
        "redteam_recon_benchkit", EVAL_SRC / "benchkit.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)

    class _FakeEvent:
        def __init__(self, enable_timing=False):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.5   # what the fake event pair always reads (ms)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    # one replay-shaped call per group: a parked-tensor copy, sub-millisecond
    parked = torch.zeros(5, 64)

    def _replay():
        parked.fill_(0.0)
        time.sleep(0.0002)

    work = [("g0", [_replay], None)]
    out = lb._stream_span_check(work)
    assert out["status"] == "ok", (
        "a replay reads quiet to the stream sentinel — no existing reading "
        "convicts the recon form")
