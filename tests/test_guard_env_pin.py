# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0
"""The guard knobs are pinned at bench start (the guard-knobs block in
benchkit.py). A source-reference card runs the kernel module inside the bench
process with no kernel fence (kernel_fence guards only the sealed/binary
reference forms), so between bench start and the post-window reads the kernel
code could flip any RKT_* switch through os.environ: RKT_REPLAY_PROBE=0 stops
the WP29 probe from spawning (the G1 replay cheat ships clean), a set
RKT_FRESH_RECHECK silently skips the WP9 recheck, a mutated RKT_FRESH_SEED
points the recheck child's layer-2 draw back at the judged content, an infinite
RKT_REPLAY_PROBE_RATIO convicts nothing. run_bench pins the bench-start values
and every guard read consults the pin; these tests pin the contract."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL_SRC = (Path(__file__).resolve().parents[1] / "src" / "rsi_kernel_tune"
            / "card_template" / "eval")


@pytest.fixture()
def bk():
    torch = pytest.importorskip("torch")
    assert torch is not None  # the benchkit imports torch at module load
    spec = importlib.util.spec_from_file_location("guard_env_pin_benchkit",
                                                  EVAL_SRC / "benchkit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ARGS = SimpleNamespace(impl="kernel")
BLOCKS = [{"name": "w0", "device_median_ms": 0.005}]


def test_pinned_reads_ignore_post_pin_mutation(bk, monkeypatch):
    """The in-process kernel flips the switches after its import; the guards
    still read what the harness was configured with."""
    monkeypatch.setenv("RKT_REPLAY_PROBE", "1")
    monkeypatch.delenv("RKT_FRESH_RECHECK", raising=False)
    monkeypatch.delenv("RKT_MEMO_LANE", raising=False)
    bk._pin_guard_env()
    try:
        monkeypatch.setenv("RKT_REPLAY_PROBE", "0")        # the probe killer
        monkeypatch.setenv("RKT_REPLAY_PROBE_RATIO", "1e9")  # the threshold killer
        monkeypatch.setenv("RKT_FRESH_RECHECK", "1")       # the recheck skipper
        monkeypatch.setenv("RKT_FRESH_SEED", "0")          # the draw-shift killer
        monkeypatch.setenv("RKT_MEMO_LANE", "1")           # the lane re-router
        assert bk._replay_probe_enabled() is True
        assert bk._replay_probe_ratio() == 3.0
        assert bk._memo_lane() is False
        assert bk._guard_env_get("RKT_FRESH_RECHECK") is None
        assert bk._recheck_seed_shift() == 0   # the parent is not a recheck child
    finally:
        bk._unpin_guard_env()


def test_unpinned_reads_stay_live(bk, monkeypatch):
    """Without an active bench run the readers behave exactly as before (direct
    readers, tests, child processes)."""
    monkeypatch.setenv("RKT_REPLAY_PROBE", "0")
    assert bk._replay_probe_enabled() is False
    monkeypatch.setenv("RKT_REPLAY_PROBE_RATIO", "5.0")
    assert bk._replay_probe_ratio() == 5.0
    monkeypatch.setenv("RKT_MEMO_LANE", "1")
    assert bk._memo_lane() is True
    monkeypatch.setenv("RKT_FRESH_RECHECK", "1")
    monkeypatch.setenv("RKT_FRESH_SEED", "100")
    assert bk._recheck_seed_shift() == 100


def test_run_bench_scopes_the_pin_to_one_call(bk, monkeypatch):
    """The pin is taken at run_bench entry (before any kernel code can run) and
    released when the call returns."""
    monkeypatch.setenv("RKT_REPLAY_PROBE", "1")
    seen: dict = {}

    def fake_run_bench(args, problem_path):
        seen["during"] = bk._replay_probe_enabled()
        monkeypatch.setenv("RKT_REPLAY_PROBE", "0")   # a kernel import, mid-run
        seen["after_mutation"] = bk._replay_probe_enabled()
        return 0

    monkeypatch.setattr(bk, "_run_bench", fake_run_bench)
    assert bk.run_bench([]) == 0
    assert seen == {"during": True, "after_mutation": True}
    assert bk._GUARD_ENV is None
    assert bk._replay_probe_enabled() is False   # live reads again


def test_child_spawn_env_carries_the_pinned_knobs(bk, monkeypatch):
    """The recheck child's layer-2 draw shift rides the ambient RKT_FRESH_SEED
    when the parent pins no explicit seed — the spawn env must carry the pinned
    value, not a kernel-mutated one (shift 0 would redraw the judged content a
    content-keyed cache already answered)."""
    runs: list = []

    def _run(*argv, **kwargs):
        runs.append(kwargs)
        class _P:
            returncode = 0
            stdout = ""
        return _P()

    monkeypatch.setenv("RKT_FRESH_SEED", "7919")
    bk._pin_guard_env()
    try:
        monkeypatch.setenv("RKT_FRESH_SEED", "0")   # the kernel's mutation
        monkeypatch.setattr(bk, "subprocess", SimpleNamespace(run=_run))
        bk._fresh_child({}, seed=None)
        assert runs[0]["env"]["RKT_FRESH_SEED"] == "7919"
    finally:
        bk._unpin_guard_env()


def test_kernel_env_kill_cannot_disarm_the_probe_at_the_production_seam(bk, monkeypatch):
    """The WP29 recon cheat with os.environ['RKT_REPLAY_PROBE']='0' at module
    import (the exact shape the red-team fixture convict catches without the
    kill line): under the bench pin the probe still buys its child and the
    suspected verdict still lands in the eval JSON."""
    monkeypatch.setattr(bk, "_probe_child",
                        lambda problem: {"shapes": [{"name": "w0",
                                                     "probe_median_ms": 2.0}]})
    bk._pin_guard_env()
    try:
        ns: dict = {}
        exec("import os\nos.environ['RKT_REPLAY_PROBE'] = '0'\n", ns)  # the import
        assert bk._replay_probe_enabled() is True
        result = {"correctness": {"pass": True},
                  "performance": {"device_median_ms": 0.005}}
        bk._maybe_replay_probe(result, {}, ARGS, BLOCKS)
        assert result["replay_probe"]["status"] == "suspected"
    finally:
        bk._unpin_guard_env()


def test_kernel_env_kill_cannot_skip_the_recheck_gate(bk, monkeypatch):
    """A set RKT_FRESH_RECHECK in the poisoned environment would make the
    parent-side recheck read 'a child never nests' and silently skip; pinned,
    the gate still runs."""
    bk._pin_guard_env()
    try:
        ns: dict = {}
        exec("import os\nos.environ['RKT_FRESH_RECHECK'] = '1'\n", ns)
        mode = bk._fresh_recheck_mode(
            {"name": "rmsnorm_h64", "schema": 1,
             "definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm",
                            "inputs": {}, "outputs": {},
                            "reference": "def run(x):\n    return x\n\ndef make_inputs(axes_and_scalars, group=None, seed=0, device=\"cpu\"):\n    import torch\n    g = torch.Generator().manual_seed(seed)\n    return {\"x\": torch.randn(4, generator=g, device=device)}\n"},
             "workload": {"uuid": "w0", "axes": {},
                          "input_groups": [{"name": "g0", "kind": "drawn",
                                            "seed": 0}]}})
        # the gate is what guards the spawn; _fresh_recheck_mode returning at all
        # means the early-return did not fire (a live read would have returned
        # None out of _fresh_recheck_block before reaching here)
        assert mode in ("regen", "process")
    finally:
        bk._unpin_guard_env()
