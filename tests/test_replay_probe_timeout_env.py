# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The probe child's timeout knob survives a junk env value (the WP29 fix).

The exposure: ``_probe_child`` read ``RKT_REPLAY_PROBE_TIMEOUT_S`` with a bare
``int()`` and no ``ValueError`` guard. A mistyped operator value ("30m",
"1,800", "600s" -- submit.toml [env] passes values through unscrubbed) raised
in ``subprocess.run``'s argument evaluation, which happens AFTER every timing
window has closed and BEFORE ``_emit``: the eval JSON is never printed and the
whole GPU eval is wasted. Every sibling numeric knob in this module
(``_replay_probe_ratio``, the gate-degraded ratio, ``_recheck_seed_shift``)
guards its parse and keeps the default on junk; the timeout knob now does the
same, and an operator override that parses still applies.

Runs on CPU; ``subprocess.run`` is stubbed so no real probe child spawns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"

torch = pytest.importorskip("torch")


def _benchkit():
    """The template benchkit, loaded fresh (the card copies it verbatim; the
    tests drive the same code the production bench child runs)."""
    spec = importlib.util.spec_from_file_location(
        "wp29_probe_timeout_benchkit", EVAL_SRC / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    return bk


def _stub_child_run(bk, runs):
    """Replace benchkit's ``subprocess`` with a stub that records the run
    kwargs and answers like a child that printed no verdict."""

    def fake_run(*args, **kwargs):
        runs.append(kwargs)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    import subprocess

    stub = SimpleNamespace(run=fake_run)
    stub.TimeoutExpired = subprocess.TimeoutExpired
    return stub


def test_junk_probe_timeout_keeps_the_default_and_the_bench_completes(monkeypatch):
    """The regression: a junk RKT_REPLAY_PROBE_TIMEOUT_S must not raise out of
    ``_probe_child`` (which would crash the bench after all windows closed).
    The child spawn must receive the conservative default."""
    bk = _benchkit()
    runs: list = []
    monkeypatch.setenv(bk.REPLAY_PROBE_TIMEOUT_ENV, "1800 seconds")
    monkeypatch.setattr(bk, "subprocess", _stub_child_run(bk, runs))

    verdict = bk._probe_child({})

    assert len(runs) == 1
    assert runs[0]["timeout"] == bk.REPLAY_PROBE_TIMEOUT_S
    assert verdict["status"] == "unavailable"


def test_parseable_probe_timeout_override_still_applies(monkeypatch):
    """The guard only absorbs junk: a real operator override must reach the
    child spawn unchanged."""
    bk = _benchkit()
    runs: list = []
    monkeypatch.setenv(bk.REPLAY_PROBE_TIMEOUT_ENV, "600")
    monkeypatch.setattr(bk, "subprocess", _stub_child_run(bk, runs))

    bk._probe_child({})

    assert len(runs) == 1
    assert runs[0]["timeout"] == 600


def test_probe_timeout_parser_matches_the_sibling_knob_contract(monkeypatch):
    """The parser's table: junk and empty fall back to the default, a valid
    value and the absent env pass through -- the same contract
    ``_replay_probe_ratio`` documents."""
    bk = _benchkit()
    env = bk.REPLAY_PROBE_TIMEOUT_ENV
    default = bk.REPLAY_PROBE_TIMEOUT_S

    monkeypatch.setenv(env, "junk")
    assert bk._replay_probe_timeout_s() == default
    monkeypatch.setenv(env, "")
    assert bk._replay_probe_timeout_s() == default
    monkeypatch.setenv(env, "45")
    assert bk._replay_probe_timeout_s() == 45
    monkeypatch.delenv(env, raising=False)
    assert bk._replay_probe_timeout_s() == default
