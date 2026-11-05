# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""FIX-R3G regression pins for BUG-CH-J-04 (``skills/timeline/timeline_probe.py``).

The probe re-runs with wait sites only when the mode=all table's perturbation
exceeds the 3% budget. On a tree whose markers carry no ``.begin``/``.end``
pair (a legal, segment-only marking style) that rerun's stamping raises
``ValueError("no wait site to keep in mode=waits")`` — and the loop's
``except ValueError`` treated it like a first-stamping failure: TIMELINE-FAIL,
return, and the already-computed mode=all table was thrown away. The pins:

* the rerun's ValueError falls through to the table already read (the
  SiteCapExceeded branch's same discipline): the probe delivers the mode=all
  table with a warning, and the driver reads it as usable;
* a direct ``--mode waits`` run on such a tree still fails — only the rerun
  degrades, the first stamping keeps its FAIL.
"""

from __future__ import annotations

import sys

import pytest

from rsi_kernel_tune.skills.timeline_probe import (
    NROW, NSITE, NSLOT, NWARP, parse_timeline,
)

TOY_NO_WAIT_PAIRS = '''"""toy kernel"""
import cutlass.cute as cute


def _seg(site, row=0):
    """timeline marker: no-op here"""
    return None


@cute.kernel
def kern(x):
    _seg("aux.entry")
    for ci in range(128):
        _seg("state.A.top", ci)
        _seg("state.C.newv.drain", ci + 0)
        _seg("state.E.end", ci)
'''

SITES = ["aux.entry", "state.A.top", "state.C.newv.drain", "state.E.end"]


class _FakeArray:
    def __init__(self, nested):
        self.nested = nested

    def any(self):
        return any(v for slot in self.nested for w in slot for row in w for v in row)

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.nested


class _FakeNumpy:
    def __init__(self):
        self.saved = {}

    def save(self, path, arr):
        self.saved[str(path)] = arr
        with open(path, "wb") as fh:
            fh.write(b"NUMPY")

    def load(self, path):
        if str(path) not in self.saved:
            raise ValueError("not a slab")
        return _FakeArray(self.saved[str(path)])


def _no_wait_slab(rows=10, warps=(4, 5)):
    """Stamps for the mode=all sites of TOY_NO_WAIT_PAIRS: one role walking
    top -> drain -> end per row."""
    arr = [[[[0] * NSITE for _ in range(NROW)] for _ in range(NWARP)]
           for _ in range(NSLOT)]
    idx = {n: i for i, n in enumerate(SITES)}
    for row in range(rows):
        t = 10_000 + row * 1000
        for w in warps:
            arr[0][w][row][idx["aux.entry"]] = t - 50
            arr[0][w][row][idx["state.A.top"]] = t
            arr[0][w][row][idx["state.C.newv.drain"]] = t + 300
            arr[0][w][row][idx["state.E.end"]] = t + 600
    return arr


def _mock_benches(tp, monkeypatch, *, base_ms, stamped_ms, slab):
    """Base bench passes; the stamped bench passes and leaves ``slab`` at
    TL_SLAB_OUT; the stamped variant's device time perturbs by
    stamped_ms / base_ms (> 1.03 here, so the waits rerun arms)."""
    fake = _FakeNumpy()
    monkeypatch.setitem(sys.modules, "numpy", fake)
    monkeypatch.setattr(tp, "run_bench", lambda label, env=None: {
        "correctness": {"pass": True, "max_diff": 0.0},
        "performance": {"device_median_ms": base_ms}})

    def stamped_bench(label, env=None, timeout=2400):
        if slab is not None:
            fake.save(env["TL_SLAB_OUT"], slab)
        return {"correctness": {"pass": True, "max_diff": 0.0},
                "performance": {"device_median_ms": stamped_ms}}, [], 0
    monkeypatch.setattr(tp, "run_bench_sampling_clock", stamped_bench)
    return fake


@pytest.fixture
def tp():
    from rsi_kernel_tune.skills import timeline_probe as tp
    return tp


def test_perturbed_all_mode_keeps_its_table_when_waits_rerun_cannot_stamp(
        tp, tmp_path, monkeypatch, capsys):
    """The waits rerun's ValueError must not discard the mode=all table: the
    probe warns and delivers what it already read (the driver sees a usable
    mode=all timeline), instead of failing the whole probe."""
    (tmp_path / "kernel.py").write_text(TOY_NO_WAIT_PAIRS, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_benches(tp, monkeypatch, base_ms=0.5, stamped_ms=0.6,
                  slab=_no_wait_slab())
    assert tp.main(["--mode", "all"]) == 0
    out = capsys.readouterr().out
    # the rerun's stamping failure is a warning naming the misses, not a FAIL
    assert "TIMELINE-FAIL" not in out
    assert "TIMELINE-WARN no wait site to keep in mode=waits" in out
    # the table already computed in mode=all is what the driver gets
    assert "TIMELINE-OK mode=all" in out
    tl = parse_timeline(out)
    assert tl["ok"] is True and tl["usable"] is True and tl["mode"] == "all"
    assert abs(tl["perturbation"] - 1.2) < 1e-9
    assert any(w.startswith("perturbation 1.2000 > 1.03 even in mode=all") for w in tl["warn"])
    # the kernel file is restored after the stamped runs
    assert (tmp_path / "kernel.py").read_text(encoding="utf-8") == TOY_NO_WAIT_PAIRS


def test_direct_waits_mode_on_a_pairless_tree_still_fails(
        tp, tmp_path, monkeypatch, capsys):
    """Only the rerun degrades. A direct --mode waits run on a tree with no
    .begin/.end pair has no table to fall back on and keeps its FAIL."""
    (tmp_path / "kernel.py").write_text(TOY_NO_WAIT_PAIRS, encoding="utf-8")
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    _mock_benches(tp, monkeypatch, base_ms=0.5, stamped_ms=0.5, slab=None)
    assert tp.main(["--mode", "waits"]) == 0
    out = capsys.readouterr().out
    assert "TIMELINE-FAIL reason=markers (no wait site to keep in mode=waits)" in out
    tl = parse_timeline(out)
    assert tl["ok"] is False and tl["usable"] is False
