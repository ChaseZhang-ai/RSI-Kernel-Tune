# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit-corps R3F fix wave over the monitor surface.

Covered here:
- BUG-CH-H-01: numeric env knobs are parsed with a readable fallback instead
  of a bare float()/int(): a typo'd TASK_TIMEOUT used to kill the monitor job
  before any output existed (GPU job burned, zero MARKER), and a typo'd
  RKT_MON_MAX_CALLS used to kill the hook install so the whole job watched
  nothing (the failure one stderr line nobody reads).
- BUG-CH-H-02: the seed profiles' change detection sees the install-time
  default-generator shift: a bench that never seeds still draws something new
  under seed1/seed2, but when the kernel writes every tensor parameter both
  runs' input signatures blank to the same all-None and the explicit-draw
  counters stay 0 -- the card's FAIL used to be dropped as "not applicable"
  with the attribution inverted. The hook now reports the install-time shift
  and classify counts it (only in the written-positions fallback, where the
  signature channel is actually blinded; a bench with no RNG draws keeps its
  not-applicable reading).

The classify tests are pure dicts; the hook/runner tests need torch (skipped
without it).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rsi_kernel_tune import monitor

REPO = Path(__file__).resolve().parents[1]
HOOK_DIR = monitor.RUNTIME_DIR / "hook"

needs_torch = pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode != 0,
    reason="torch not available")


def _python_on_hook(code: str, env_extra: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(HOOK_DIR), RKT_MON_OUT=str(tmp_path / "out"),
               **env_extra)
    env.pop("RKT_MON_PROFILE", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)


# --- BUG-CH-H-01: the runner's TASK_TIMEOUT -----------------------------------------

def _runtime_run(tmp_path: Path):
    """The runtime runner, importable from a card-shaped tree (.rkt_monitor/run.py)."""
    (tmp_path / "card" / ".rkt_monitor").mkdir(parents=True)
    src = (tmp_path / "card" / ".rkt_monitor" / "run.py")
    src.write_text(monitor.runtime_files()[f"{monitor.TREE_DIR}/run.py"], encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("corps_r3f_monitor_run", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_typoed_task_timeout_is_readable_not_a_bare_crash(tmp_path, monkeypatch, capsys):
    """TASK_TIMEOUT=30m used to raise ValueError from main() before any output:
    the job died with a traceback, zero MARKER, and the parse read "printed no
    result". It now names the variable and falls back to the default budget."""
    mod = _runtime_run(tmp_path)
    monkeypatch.setenv("TASK_TIMEOUT", "30m")
    rc = mod.main(["--profiles", "", "--no-cardref"])   # no profiles: env parse is all there is
    out = capsys.readouterr().out
    assert rc == 0
    assert "TASK_TIMEOUT" in out and "30m" in out
    assert "budget 3600" in out                        # the default budget, not a crash


def test_a_numeric_task_timeout_still_parses(tmp_path, monkeypatch, capsys):
    mod = _runtime_run(tmp_path)
    monkeypatch.setenv("TASK_TIMEOUT", "900")
    rc = mod.main(["--profiles", "", "--no-cardref"])
    out = capsys.readouterr().out
    assert rc == 0 and "budget 765" in out             # 0.85 * 900, unchanged


def test_an_unset_task_timeout_uses_the_default_budget(tmp_path, monkeypatch, capsys):
    mod = _runtime_run(tmp_path)
    monkeypatch.delenv("TASK_TIMEOUT", raising=False)
    rc = mod.main(["--profiles", "", "--no-cardref"])
    out = capsys.readouterr().out
    assert rc == 0 and "budget 3600" in out


# --- BUG-CH-H-01: the hook's RKT_MON_MAX_CALLS ---------------------------------------

def test_a_typoed_max_calls_keeps_the_hook_alive(tmp_path):
    """RKT_MON_MAX_CALLS=64 calls used to kill install() (one stderr line, the
    whole job watched nothing). It now keeps the default and records why."""
    p = _python_on_hook(
        "import rkt_monitor_hook as h\n"
        "print(h._state.get('installed'), h._state.get('max_calls'))\n"
        "print([e for e in h._state.get('errors') or [] if 'RKT_MON_MAX_CALLS' in e])\n",
        {"RKT_MON_PROFILE": "base", "RKT_MON_MAX_CALLS": "64 calls"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "hook not installed" not in p.stderr
    assert "True 64" in p.stdout                       # installed, default kept
    assert "RKT_MON_MAX_CALLS" in p.stdout              # the misconfiguration is recorded


def test_a_numeric_max_calls_still_parses(tmp_path):
    p = _python_on_hook(
        "import rkt_monitor_hook as h\nprint(h._state.get('installed'), h._state.get('max_calls'))\n",
        {"RKT_MON_PROFILE": "base", "RKT_MON_MAX_CALLS": "12"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "True 12" in p.stdout


def test_an_unset_max_calls_keeps_the_default(tmp_path):
    p = _python_on_hook(
        "import rkt_monitor_hook as h\nprint(h._state.get('installed'), h._state.get('max_calls'))\n",
        {"RKT_MON_PROFILE": "base"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "True 64" in p.stdout


# --- BUG-CH-H-02 --------------------------------------------------------------------

def _call_all_written(i=0):
    """One kernel call that writes every tensor parameter (a full in-place
    shape): the hook blanks every position of ``in_sums`` for written args, so
    both runs' signatures read as the same all-None list."""
    return {"fn": "call", "i": i, "n_inputs": 2, "in_sums": [1.0, 2.0],
            "in_nonfinite": [0, 0], "written": [0, 1], "inputs_nonfinite": 0,
            "outputs": [{"slot": "arg0", "nan": 0, "posinf": 0, "neginf": 0,
                         "abs_sum": 10.0, "sq_sum": 5.0}]}


def _rec(profile, card_pass=True, calls=(), **kw):
    hook = {"calls": list(calls), "perturbed": kw.pop("perturbed", 1),
            "reseeded": kw.pop("reseeded", 0),
            "default_reseeded": kw.pop("default_reseeded", None)}
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": kw.pop("cases", []), "hook": hook, **kw}


def test_a_seed_fail_survives_full_write_params_via_the_default_shift():
    """The blind geometry: signatures blanked by full-write params, counters 0
    (the bench never seeds), yet the install-time default-generator shift made
    seed1 draw different inputs -- the card FAIL is the monitor's core alarm
    and must not be dropped as not applicable with the reason inverted."""
    base = _rec("base", calls=[_call_all_written()], perturbed=0, reseeded=0,
                default_reseeded=False)
    seed1 = _rec("seed1", card_pass=False, calls=[_call_all_written()],
                 perturbed=0, reseeded=0, default_reseeded=True,
                 cases=[{"case": "mid", "pass": False}])
    v = monitor.classify({"profiles": [base, seed1]})
    assert "seed1" in v["ran"] and "seed1" not in v["not_applicable"]
    assert v["status"] == "alarm"
    assert "does not generalise" in v["findings"][0]["what"]


def test_old_records_without_the_flag_keep_their_reading():
    """Hook json from before the flag existed: the fallback reads as before."""
    base = _rec("base", calls=[_call_all_written()], perturbed=0, reseeded=0)
    seed1 = _rec("seed1", card_pass=True, calls=[_call_all_written()],
                 perturbed=0, reseeded=0)
    v = monitor.classify({"profiles": [base, seed1]})
    assert v["not_applicable"] == ["seed1"] and v["status"] == "ok"


def test_a_passing_seed_profile_still_counts_as_run():
    """A shifted seed that changed the inputs and the card still passes: the
    profile counts as ran, no finding."""
    base = _rec("base", calls=[_call_all_written()], perturbed=0, reseeded=0,
                default_reseeded=False)
    seed1 = _rec("seed1", card_pass=True, calls=[_call_all_written()],
                 perturbed=0, reseeded=0, default_reseeded=True)
    v = monitor.classify({"profiles": [base, seed1]})
    assert v["ran"] == ["base", "seed1"] and not v["not_applicable"]
    assert v["status"] == "ok"


def test_no_rng_bench_keeps_not_applicable():
    """A bench with no torch RNG draws: the signature channel works (nothing
    written) so the profile stays not applicable -- the fix must not widen."""
    base = _rec("base", calls=[{"fn": "call", "i": 0, "in_sums": [1.0],
                                "outputs": [{"nan": 0, "abs_sum": 10.0, "sq_sum": 5.0}]}],
                perturbed=0, reseeded=0, default_reseeded=True)
    seed1 = _rec("seed1", card_pass=True, calls=[{"fn": "call", "i": 0, "in_sums": [1.0],
                                                  "outputs": [{"nan": 0, "abs_sum": 10.0,
                                                               "sq_sum": 5.0}]}],
                 perturbed=0, reseeded=0, default_reseeded=True)
    v = monitor.classify({"profiles": [base, seed1]})
    assert v["not_applicable"] == ["seed1"]


@needs_torch
def test_hook_reports_the_install_time_default_seed_shift(tmp_path):
    """Under a seed profile the hook's install-time ``manual_seed(off)`` rides
    the hook json as ``default_reseeded`` -- the evidence a never-seeding bench
    still carries."""
    p = _python_on_hook(
        "import rkt_monitor_hook as h\nprint(h._state.get('default_reseeded'))\n",
        {"RKT_MON_PROFILE": "seed1"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "True"


@needs_torch
def test_non_seed_profiles_do_not_claim_the_default_shift(tmp_path):
    p = _python_on_hook(
        "import rkt_monitor_hook as h\nprint(h._state.get('default_reseeded'))\n",
        {"RKT_MON_PROFILE": "spikes"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False"


@needs_torch
def test_base_profile_does_not_claim_the_default_shift(tmp_path):
    p = _python_on_hook(
        "import rkt_monitor_hook as h\nprint(h._state.get('default_reseeded'))\n",
        {"RKT_MON_PROFILE": "base"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False"
