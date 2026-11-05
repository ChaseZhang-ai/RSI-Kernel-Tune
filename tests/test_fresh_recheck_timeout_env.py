# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0
"""The fresh-recheck child's timeout knob (RKT_FRESH_RECHECK_TIMEOUT_S) must
honor the same junk-safe contract as its WP29 sibling (the replay probe's
RKT_REPLAY_PROBE_TIMEOUT_S, hardened in the T5C-03 fix): an unreadable value
falls back to the conservative default instead of raising out of ``_fresh_child``
— a read that fires AFTER every timing window closed, where an unguarded
ValueError would kill the bench before the eval JSON exists and waste the whole
GPU eval. The sibling kept the bare ``int()`` spelling the WP29 knob had been
copied from; these tests pin both halves of the contract."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL_SRC = (Path(__file__).resolve().parents[1] / "src" / "rsi_kernel_tune"
            / "card_template" / "eval")

_ENV = "RKT_FRESH_RECHECK_TIMEOUT_S"


@pytest.fixture()
def bk():
    torch = pytest.importorskip("torch")
    assert torch is not None  # the benchkit imports torch at module load
    spec = importlib.util.spec_from_file_location("fresh_recheck_timeout_benchkit",
                                                  EVAL_SRC / "benchkit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_child_run(bk, runs):
    """A subprocess.run stub: record the spawn kwargs, produce no verdict line
    (the child 'said nothing' -> pass=None, never a conviction)."""

    def _run(*argv, **kwargs):
        runs.append(kwargs)
        class _P:
            returncode = 0
            stdout = ""
        return _P()

    return _run


def test_junk_recheck_timeout_keeps_the_default(monkeypatch, bk):
    """Junk must not raise out of ``_fresh_child``; the spawn gets the default."""
    runs: list = []
    monkeypatch.setenv(_ENV, "1800 seconds")
    monkeypatch.setattr(bk, "subprocess", SimpleNamespace(run=_stub_child_run(bk, runs)))
    verdict = bk._fresh_child({}, seed=None)
    assert len(runs) == 1
    assert runs[0]["timeout"] == bk.FRESH_RECHECK_TIMEOUT_S
    assert verdict == {"pass": None,
                       "detail": "recheck child printed no verdict (rc=0)"}


def test_empty_recheck_timeout_keeps_the_default(monkeypatch, bk):
    runs: list = []
    monkeypatch.setenv(_ENV, "")
    monkeypatch.setattr(bk, "subprocess", SimpleNamespace(run=_stub_child_run(bk, runs)))
    bk._fresh_child({}, seed=None)
    assert runs[0]["timeout"] == bk.FRESH_RECHECK_TIMEOUT_S


def test_parseable_recheck_timeout_override_still_applies(monkeypatch, bk):
    """The guard only absorbs junk: a real operator override reaches the spawn."""
    runs: list = []
    monkeypatch.setenv(_ENV, "600")
    monkeypatch.setattr(bk, "subprocess", SimpleNamespace(run=_stub_child_run(bk, runs)))
    bk._fresh_child({}, seed=None)
    assert runs[0]["timeout"] == 600


def test_recheck_timeout_parser_matches_the_probe_sibling_contract(monkeypatch, bk):
    """Same table as ``_replay_probe_timeout_s``: junk and empty fall back to
    the default; a valid value and the absent env pass through."""
    default = bk.FRESH_RECHECK_TIMEOUT_S
    monkeypatch.setenv(_ENV, "junk")
    assert bk._fresh_recheck_timeout_s() == default
    monkeypatch.setenv(_ENV, "")
    assert bk._fresh_recheck_timeout_s() == default
    monkeypatch.setenv(_ENV, "45")
    assert bk._fresh_recheck_timeout_s() == 45
    monkeypatch.delenv(_ENV, raising=False)
    assert bk._fresh_recheck_timeout_s() == default
