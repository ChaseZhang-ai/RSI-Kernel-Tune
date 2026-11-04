"""Red-team matrix, the cache-output-reuse family (SOL category: caching).

Expected defense, written before the attacks: the WP9 timing-fresh-input
recheck, which re-runs the correctness protocol AFTER all timing windows close
(never inside a window; measure_paired untouched):
  layer 1 — a FRESH PROCESS: every allocation moves, module state resets, so a
    pointer/state-keyed replay misses and the post-artifact path serves garbage;
  layer 2 — the child re-DRAWS the generated inputs at a shifted seed, so a
    content-keyed replay misses too.

The rehearsal runs the production bench child (eval/bench.py --mode correctness
--fresh-recheck) on CPU (PK_FRESH_RECHECK_DEVICE=cpu); torch required. The
driver-side consumption (REVERT + suspicious) is asserted in
test_redteam_accounting.py / test_fresh_recheck.py.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from test_fresh_recheck import (  # noqa: E402  (path set up by conftest)
    CHEAT_KERNEL, _VERDICT, _card, _park, _run_card,
)

# Attack 1 (pointer/state-keyed cache): the module parks the judged answers
# keyed on the input tensor's data_ptr — the timed calls replay by pointer, and
# any caller whose tensors it did not pre-compute gets zeros. Expected defense:
# layer 1 — the recheck child is a fresh process, the parked artifact survives
# on disk but its KEYS do not match any fresh allocation, so the artifact-miss
# path serves zeros and the verdict fails.
PTR_CHEAT_KERNEL = '''import torch
from pathlib import Path

_SIDE = Path(__file__).parent / ".ptr_buffer.pt"

def run(hidden_states, weight, output):
    x = hidden_states.float()
    good = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())
    buf = torch.load(_SIDE) if _SIDE.is_file() else {}
    key = int(hidden_states.data_ptr())
    if key in buf:                       # the timed calls replay by pointer
        output.copy_(buf[key])
        return
    if _SIDE.is_file():                  # a fresh allocation the buffer misses:
        output.zero_()                   # the cheater's bet — nobody ever asks
        return
    output.copy_(good.to(hidden_states.dtype))   # the correctness call is honest
    buf[key] = good.to(hidden_states.dtype)
    torch.save(buf, _SIDE)
'''


def _verdict(proc) -> dict:
    return json.loads([l for l in proc.stdout.splitlines()
                       if l.startswith(_VERDICT)][-1][len(_VERDICT):])


def test_attack1_pointer_keyed_cache_misses_in_a_fresh_process(tmp_path):
    """Defense: fresh_recheck layer 1. The parked artifact survives the process
    boundary; the fresh allocations' pointers do not match its keys — zeros come
    back and the recheck verdict fails. The honest kernel re-passes."""
    cheat = _card(tmp_path / "cheat", PTR_CHEAT_KERNEL)
    # the artifact, parked as the eval's own correctness call would leave it
    # (keyed on a pointer only the eval process ever saw)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 64, generator=g)
    good = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    import torch as _t
    _t.save({123456789: good}, cheat / ".ptr_buffer.pt")
    r = _run_card(cheat, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is False, (
        "the pointer-keyed artifact must miss every fresh allocation (layer 1)")


def test_attack2_content_keyed_cache_misses_the_shifted_draw(tmp_path):
    """Defense: fresh_recheck layer 2. A content-keyed cheat survives a re-judge
    of the SAME content and fails the SHIFTED draw — the seed shift is what
    moves the content under the cache. (The same-seed fingerprint the driver
    records on such a fail is exercised in test_redteam_accounting.)"""
    card_hit = _card(tmp_path / "hit", CHEAT_KERNEL)
    _park(card_hit, [s + 7919 for s in (0, 1, 2)])   # covers the shifted child's draws
    r = _run_card(card_hit, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is True, (
        "content the buffer covers re-passes — the cheat survives a same-content ask")

    card_miss = _card(tmp_path / "miss", CHEAT_KERNEL)
    _park(card_miss, [0, 1, 2])                      # covers only the JUDGED instance
    r = _run_card(card_miss, "--mode", "correctness", "--fresh-recheck")
    assert _verdict(r)["pass"] is False, (
        "the shifted draw must miss a judged-content cache (layer 2)")
