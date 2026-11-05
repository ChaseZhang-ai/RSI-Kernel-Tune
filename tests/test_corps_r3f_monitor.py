# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit-corps R3F fix wave over the monitor surface.

Covered here:
- BUG-CH-H-01: numeric env knobs are parsed with a readable fallback instead
  of a bare float()/int(): a typo'd TASK_TIMEOUT used to kill the monitor job
  before any output existed (GPU job burned, zero MARKER), and a typo'd
  PK_MON_MAX_CALLS used to kill the hook install so the whole job watched
  nothing (the failure one stderr line nobody reads).
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

from peregrine_kernel import monitor

REPO = Path(__file__).resolve().parents[1]
HOOK_DIR = monitor.RUNTIME_DIR / "hook"

needs_torch = pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode != 0,
    reason="torch not available")


def _python_on_hook(code: str, env_extra: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(HOOK_DIR), PK_MON_OUT=str(tmp_path / "out"),
               **env_extra)
    env.pop("PK_MON_PROFILE", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)


# --- BUG-CH-H-01: the runner's TASK_TIMEOUT -----------------------------------------

def _runtime_run(tmp_path: Path):
    """The runtime runner, importable from a card-shaped tree (.pk_monitor/run.py)."""
    (tmp_path / "card" / ".pk_monitor").mkdir(parents=True)
    src = (tmp_path / "card" / ".pk_monitor" / "run.py")
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


# --- BUG-CH-H-01: the hook's PK_MON_MAX_CALLS ---------------------------------------

def test_a_typoed_max_calls_keeps_the_hook_alive(tmp_path):
    """PK_MON_MAX_CALLS=64 calls used to kill install() (one stderr line, the
    whole job watched nothing). It now keeps the default and records why."""
    p = _python_on_hook(
        "import pk_monitor_hook as h\n"
        "print(h._state.get('installed'), h._state.get('max_calls'))\n"
        "print([e for e in h._state.get('errors') or [] if 'PK_MON_MAX_CALLS' in e])\n",
        {"PK_MON_PROFILE": "base", "PK_MON_MAX_CALLS": "64 calls"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "hook not installed" not in p.stderr
    assert "True 64" in p.stdout                       # installed, default kept
    assert "PK_MON_MAX_CALLS" in p.stdout              # the misconfiguration is recorded


def test_a_numeric_max_calls_still_parses(tmp_path):
    p = _python_on_hook(
        "import pk_monitor_hook as h\nprint(h._state.get('installed'), h._state.get('max_calls'))\n",
        {"PK_MON_PROFILE": "base", "PK_MON_MAX_CALLS": "12"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "True 12" in p.stdout


def test_an_unset_max_calls_keeps_the_default(tmp_path):
    p = _python_on_hook(
        "import pk_monitor_hook as h\nprint(h._state.get('installed'), h._state.get('max_calls'))\n",
        {"PK_MON_PROFILE": "base"}, tmp_path)
    assert p.returncode == 0, p.stderr
    assert "True 64" in p.stdout
