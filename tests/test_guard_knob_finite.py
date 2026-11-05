# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0
"""The guard knobs' junk-safe contract covers values that READ as numbers but
void the knob's meaning, not just values that fail float():

  RKT_REPLAY_PROBE_RATIO=nan|inf -> the probe compares False against every
    reading and silently fails OPEN (never convicts);
  RKT_REPLAY_PROBE_RATIO=-1|0   -> every honest kernel's re-time exceeds the
    threshold and the probe fails CLOSED on the whole fleet (universal REVERT).

Both spellings would ride the eval JSON's ``ratio_threshold`` looking like a
deliberate configuration, so they keep the conservative default instead — the
same contract the parser already documents for unreadable values. The gate's
degraded-share accounting knob gets the non-finite guard too (nan would claim
"not degraded" against every share)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL_SRC = (Path(__file__).resolve().parents[1] / "src" / "rsi_kernel_tune"
            / "card_template" / "eval")

RATIO_ENV = "RKT_REPLAY_PROBE_RATIO"


@pytest.fixture()
def bk():
    torch = pytest.importorskip("torch")
    assert torch is not None  # the benchkit imports torch at module load
    spec = importlib.util.spec_from_file_location("guard_knob_finite_benchkit",
                                                  EVAL_SRC / "benchkit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("junk", ["nan", "inf", "-inf", "-1", "0"])
def test_meaning_voiding_ratio_values_keep_the_default(bk, monkeypatch, junk):
    monkeypatch.setenv(RATIO_ENV, junk)
    assert bk._replay_probe_ratio() == bk.REPLAY_PROBE_RATIO_DEFAULT


def test_strict_but_finite_ratios_still_apply(bk, monkeypatch):
    """The guard absorbs only the meaning-voiding spellings: a deliberately
    strict threshold (stricter than the default) passes through."""
    monkeypatch.setenv(RATIO_ENV, "1.0")
    assert bk._replay_probe_ratio() == 1.0
    monkeypatch.setenv(RATIO_ENV, "5.0")
    assert bk._replay_probe_ratio() == 5.0
    monkeypatch.delenv(RATIO_ENV, raising=False)
    assert bk._replay_probe_ratio() == bk.REPLAY_PROBE_RATIO_DEFAULT


def test_nan_ratio_no_longer_fails_the_probe_open(bk, monkeypatch):
    """End to end at the assembly: a ratio the probe can never exceed must not
    read `pass` — the junk value keeps the default and the honest verdict
    (suspected) lands."""
    monkeypatch.setenv(RATIO_ENV, "nan")
    monkeypatch.setattr(bk, "_probe_child",
                        lambda problem: {"shapes": [{"name": "w0",
                                                     "probe_median_ms": 2.0}]})
    block = bk._replay_probe_block({}, [{"name": "w0", "device_median_ms": 0.005}])
    assert block["status"] == "suspected"
    assert block["ratio_threshold"] == bk.REPLAY_PROBE_RATIO_DEFAULT


def test_gate_degraded_nan_keeps_the_accounting_meaningful(bk, monkeypatch):
    """nan compares False against every share: pre-fix it claimed 'not
    degraded' on a block that IS degraded. The default (0.5) reads it right."""
    monkeypatch.setenv(bk.GATE_DEGRADED_RATIO_ENV, "nan")
    block = {"rows": [{"status": "unavailable"}, {"status": "unavailable"},
                      {"status": "ran"}]}
    assert bk.gate_degraded(block) is True   # 2 > 0.5*3 -- degraded
    monkeypatch.delenv(bk.GATE_DEGRADED_RATIO_ENV, raising=False)
    assert bk.gate_degraded(block) is True   # the default says the same
