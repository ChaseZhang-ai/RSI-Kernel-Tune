# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The defense gates read the environment as it stood BEFORE the kernel imported.

The exposure (hunt4, reproduced on CPU): every post-window defense gate — the
lane switch, the fresh-recheck's child-nests guard and layer-2 seed shift, the
replay probe's switch / threshold / timeout / child marker — read ``os.environ``
LIVE, after the kernel module had already run in the same process, and no fence
covers a kernel that WRITES the environment (the sealed-reference and CUDA
fences ban ``environ``; a definition-reference card runs unfenced and a
binary-reference card's KERNEL_FENCE does not name it). One import-time
``os.environ["RKT_FRESH_SEED"] = "0"`` in the kernel file neutralized the
recheck child's shifted draw, so a content-keyed cache that fails the recheck
today passes it with the line added (the control and the tampered variant are
both pinned below, at the child the production parent spawns). The fix is the
precision switch's house pattern ("read once, at import -- before any kernel
module runs"): the environment is captured at the kernel-import boundary and
the defense gates read the capture; reads before any kernel import (the tests'
in-process posture, no capture taken) fall through to the live environment, so
every operator/launcher behavior is unchanged.

Runs on CPU (torch required); the child-level test runs a real recheck child.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"


def _benchkit():
    """The template benchkit, loaded fresh (the card copies it verbatim; the
    tests drive the same code the production bench runs)."""
    spec = importlib.util.spec_from_file_location(
        "hunt4_defense_env_benchkit", EVAL_SRC / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    return bk


# -- the gates are frozen at the kernel-import boundary (in process) -----------------

def test_gate_reads_after_the_capture_ignore_later_env_flips(monkeypatch):
    """Simulate the production order: the kernel imports (the capture fires at
    the boundary), THEN the kernel's import code flips the environment — the
    gates must read the captured values, not the flipped ones."""
    bk = _benchkit()
    monkeypatch.delenv(bk.MEMO_LANE_ENV, raising=False)
    monkeypatch.setenv(bk.REPLAY_PROBE_ENV, "1")
    monkeypatch.setenv(bk.REPLAY_PROBE_RATIO_ENV, "3.0")
    monkeypatch.delenv(bk.FRESH_RECHECK_ENV, raising=False)
    bk._capture_kernel_import_env()          # --- the kernel-import boundary
    # the kernel's import code flips everything it can reach:
    monkeypatch.setenv(bk.MEMO_LANE_ENV, "1")
    monkeypatch.setenv(bk.REPLAY_PROBE_ENV, "0")
    monkeypatch.setenv(bk.REPLAY_PROBE_RATIO_ENV, "inf")
    monkeypatch.setenv(bk.REPLAY_PROBE_CHILD_ENV, "1")
    monkeypatch.setenv(bk.FRESH_RECHECK_ENV, "1")
    assert bk._memo_lane() is False          # the lane did not move
    assert bk._replay_probe_enabled() is True
    assert bk._replay_probe_ratio() == 3.0   # not inf: the probe was not muted
    assert bk._recheck_seed_shift() == 0     # the parent stays at shift 0
    # the child-nests guards read the capture: no marker at capture -> the parent
    # still buys its recheck / probe even though the kernel "became" a child
    bought: list = []

    def _fresh_canary(problem, *, seed):
        bought.append("rc")
        return {"pass": None}

    monkeypatch.setattr(bk, "_fresh_child", _fresh_canary)
    monkeypatch.setattr(bk, "_fresh_recheck_mode", lambda problem: "regen")
    monkeypatch.setattr(bk, "_nondeterminism_fingerprint", lambda *a, **k: {})
    block = bk._fresh_recheck_block({"definition": {"name": "x"}})
    assert block["pass"] is None and bought == ["rc"]    # the parent ran its recheck


def test_a_marker_present_at_the_capture_still_reads_as_a_child(monkeypatch):
    """The launcher's own markers (the child processes) must keep working: what
    the spawn env carried at import time is what the gates read."""
    bk = _benchkit()
    monkeypatch.setenv(bk.FRESH_RECHECK_ENV, "1")
    monkeypatch.setenv(bk.REPLAY_PROBE_CHILD_ENV, "1")
    bk._capture_kernel_import_env()
    assert bk._recheck_seed_shift() == 7919  # the child's default shift stands
    assert bk._fresh_recheck_block({}) is None   # a child never nests


def test_reads_before_any_kernel_import_stay_live(monkeypatch):
    """No kernel has imported in this process (the tests' posture, and every
    direct knob call): the gates read the live environment exactly as before —
    an operator override set at launch still applies."""
    bk = _benchkit()
    assert bk._ENV_AT_KERNEL_IMPORT is None
    monkeypatch.setenv(bk.MEMO_LANE_ENV, "1")
    assert bk._memo_lane() is True
    monkeypatch.setenv(bk.REPLAY_PROBE_RATIO_ENV, "5.0")
    assert bk._replay_probe_ratio() == 5.0
    monkeypatch.setenv(bk.REPLAY_PROBE_TIMEOUT_ENV, "45")
    assert bk._replay_probe_timeout_s() == 45


def test_a_kernel_env_write_never_reaches_the_spawned_children(monkeypatch):
    """The children spawn with the environment as it stood before the kernel's
    import: the kernel's own writes ride nowhere."""
    bk = _benchkit()
    runs: list = []

    def fake_run(*args, **kwargs):
        runs.append(kwargs)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    import subprocess
    stub = SimpleNamespace(run=fake_run)
    stub.TimeoutExpired = subprocess.TimeoutExpired
    monkeypatch.setattr(bk, "subprocess", stub)
    bk._capture_kernel_import_env()
    monkeypatch.setenv("RKT_FRESH_SEED", "0")            # the kernel's import write
    monkeypatch.setenv("RKT_MEMO_LANE", "1")
    problem = {"_dir": str(REPO)}                          # never loaded by the stub
    bk._fresh_child(problem, seed=None)
    bk._probe_child({})
    assert len(runs) == 2
    for call in runs:
        env = call["env"]
        assert env.get("RKT_FRESH_SEED") != "0"            # the write did not inherit
        assert env.get(bk.MEMO_LANE_ENV) != "1"
    assert runs[1]["timeout"] == bk.REPLAY_PROBE_TIMEOUT_S


# -- the child-level attack, at the spawn the production parent buys ------------------

TAMPER = '''import os
os.environ["RKT_FRESH_SEED"] = "0"   # the child's layer-2 shift, neutralized
'''


def _tampered_child_fails(tmp_path: Path) -> None:
    """The G1-shaped cache cheat (judged content parked, zeros on miss) plus the
    import-time env write: at the production child's spawn the shift must stay
    7919 — the cheat must fail the recheck exactly like the untampered control."""
    from test_fresh_recheck import CHEAT_KERNEL, _card, _park, _run_card
    tampered = _card(tmp_path / "tampered", TAMPER + CHEAT_KERNEL)
    _park(tampered, [0, 1, 2])   # the cache covers the JUDGED content only
    r = _run_card(tampered, "--mode", "correctness", "--fresh-recheck")
    verdict = json.loads([l for l in r.stdout.splitlines()
                          if l.startswith("FRESH-RECHECK-VERDICT ")][-1][22:])
    assert verdict["pass"] is False, (
        "an import-time env write must not un-shift the recheck child's draw")


def test_kernel_env_write_cannot_unshift_the_recheck_child(tmp_path):
    _tampered_child_fails(tmp_path)


def test_replay_probe_ratio_rejects_non_threshold_values(monkeypatch):
    """Zero, negative and inf/nan parse but are not thresholds: everything
    suspects (or nothing ever does) under them — an honest kernel would read
    suspected (a hard suspicious REVERT) or a cheat would sail through on
    status pass — so they keep the conservative default like junk does."""
    bk = _benchkit()
    payload = {"shapes": [{"name": "w0", "probe_median_ms": 0.6}]}
    bk._probe_child = lambda problem: payload
    blocks = [{"name": "w0", "device_median_ms": 0.5}]
    for bad in ("0", "-1", "inf", "-inf", "nan"):
        monkeypatch.setenv(bk.REPLAY_PROBE_RATIO_ENV, bad)
        assert bk._replay_probe_ratio() == bk.REPLAY_PROBE_RATIO_DEFAULT, bad
        block = bk._replay_probe_block({}, blocks)
        assert block["status"] == "pass", bad
    monkeypatch.setenv(bk.REPLAY_PROBE_RATIO_ENV, "5.0")
    assert bk._replay_probe_ratio() == 5.0    # a real override still applies


def test_probe_task_input_pins_reach_the_shift_reader(monkeypatch):
    """The probe child's per-call pin writes the live env AND the snapshot (the
    shift reads go through the snapshot), and restores both — a shift the spawn
    env carried comes back after the pin."""
    bk = _benchkit()
    monkeypatch.setenv(bk.FRESH_RECHECK_ENV, "1")   # the probe child's spawn state
    monkeypatch.setenv(bk.FRESH_SEED_ENV, "7919")   # the spawn's own shift value
    bk._capture_kernel_import_env()
    seen = []

    def fake_task_inputs(view, group, device):
        seen.append(bk._recheck_seed_shift())
        return []

    monkeypatch.setattr(bk, "task_inputs", fake_task_inputs)
    group = {"name": "g0", "kind": "drawn", "seed": 5}
    bk._probe_task_inputs({}, group, "cpu", 5 + 104729 + 1000)
    bk._probe_task_inputs({}, group, "cpu", 5 + 104729 + 2000)
    assert seen == [104729 + 1000, 104729 + 2000]   # per-call pins, not one stale shift
    assert bk._ENV_AT_KERNEL_IMPORT.get(bk.FRESH_SEED_ENV) == "7919"   # restored
