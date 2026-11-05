"""Regression tests for the W1 audit fix W1-LIF-01 (monitor CLI kernel-file default):

* the standalone ``rsi-kernel-tune <op> monitor`` CLI resolves --kernel-file from the
  card's own problem.json (cuda -> kernel.cu) instead of hardcoding kernel.py --
  monitor/cli.py.

(The wave's SEED-14 / SEED-15 / SEED-41 / SEED-42 faces are covered by the
 corps' own tests on this branch; only the W1-unique LIF-01 cases live here.
 The _chain hardening cases are appended by their own commit below.)

Ported from the R2 audit's untracked test batch, rewritten for the
rsi_kernel_tune tree. CPU only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rsi_kernel_tune import monitor


# --- W1-LIF-01: the CLI defaults to the card's own kernel file ---------------------
def test_monitor_cli_defaults_to_the_cards_kernel_file(tmp_path, capsys):
    from rsi_kernel_tune.monitor import cli

    op = "cudaop"
    op_dir = tmp_path / "operators" / op
    problem = op_dir / "eval" / "problem"
    problem.mkdir(parents=True)
    (problem / "problem.json").write_text('{"kernel_file": "kernel.cu"}', encoding="utf-8")
    ws = tmp_path / "ws"
    for args in (["init", "-q", str(ws)],
                 ["-C", str(ws), "config", "user.name", "t"],
                 ["-C", str(ws), "config", "user.email", "t@t"],
                 ["-C", str(ws), "commit", "--allow-empty", "-q", "-m", "seed"]):
        subprocess.run(["git", *args], check=True)
    rc = cli.main([op, "--base-dir", str(tmp_path), "--workspace", str(ws), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "--kernel kernel.cu" in out, out      # was hardcoded kernel.py before


def test_monitor_cli_an_explicit_kernel_file_still_wins(tmp_path, capsys):
    from rsi_kernel_tune.monitor import cli

    op = "cudaop"
    op_dir = tmp_path / "operators" / op
    problem = op_dir / "eval" / "problem"
    problem.mkdir(parents=True)
    (problem / "problem.json").write_text('{"kernel_file": "kernel.cu"}', encoding="utf-8")
    ws = tmp_path / "ws"
    for args in (["init", "-q", str(ws)],
                 ["-C", str(ws), "config", "user.name", "t"],
                 ["-C", str(ws), "config", "user.email", "t@t"],
                 ["-C", str(ws), "commit", "--allow-empty", "-q", "-m", "seed"]):
        subprocess.run(["git", *args], check=True)
    rc = cli.main([op, "--base-dir", str(tmp_path), "--workspace", str(ws),
                   "--kernel-file", "src/kernel.py", "--dry-run"])
    assert rc == 0
    assert "--kernel src/kernel.py" in capsys.readouterr().out


def test_monitor_cli_a_card_without_kernel_file_falls_back_to_kernel_py(tmp_path, capsys):
    from rsi_kernel_tune.monitor import cli

    op = "pyop"
    op_dir = tmp_path / "operators" / op
    op_dir.mkdir(parents=True)          # no eval/problem/problem.json at all
    ws = tmp_path / "ws"
    for args in (["init", "-q", str(ws)],
                 ["-C", str(ws), "config", "user.name", "t"],
                 ["-C", str(ws), "config", "user.email", "t@t"],
                 ["-C", str(ws), "commit", "--allow-empty", "-q", "-m", "seed"]):
        subprocess.run(["git", *args], check=True)
    rc = cli.main([op, "--base-dir", str(tmp_path), "--workspace", str(ws), "--dry-run"])
    assert rc == 0
    assert "--kernel kernel.py" in capsys.readouterr().out


# --- SEED-07: _chain never executes a planted workspace sitecustomize --------------
@pytest.fixture
def startup_env(monkeypatch):
    """A clean start-up: no monitor profile, a saved sys.path and sitecustomize
    slot — the module under test mutates all three while it runs."""
    monkeypatch.delenv("RKT_MON_PROFILE", raising=False)
    had = sys.modules.pop("sitecustomize", None)
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    cur = sys.modules.pop("sitecustomize", None)
    if had is not None:
        sys.modules["sitecustomize"] = had
    elif cur is not None and cur is not had:
        sys.modules["sitecustomize"] = cur  # someone restored a different module


def _load_runtime_sitecustomize():
    import importlib.util
    path = Path(monitor.RUNTIME_DIR) / "hook" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("sitecustomize", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sitecustomize"] = mod     # as interpreter start-up would have it
    spec.loader.exec_module(mod)
    return mod


def test_chain_skips_a_planted_workspace_sitecustomize(tmp_path, monkeypatch, startup_env):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sitecustomize.py").write_text(
        "import sys as _s\n_s.modules['__rkt_planted_ran__'] = True\n", encoding="utf-8")
    image = tmp_path / "site-packages"     # the harness's own copy, outside the tree
    image.mkdir()
    (image / "sitecustomize.py").write_text("CHAIN_MARKER = 'image'\n", encoding="utf-8")
    monkeypatch.setenv("RKT_MON_ROOT", str(ws))
    monkeypatch.chdir(ws)
    sys.path[:] = [str(ws), str(image)]    # planted first: the old _chain imported it
    _load_runtime_sitecustomize()
    chained = sys.modules["sitecustomize"]
    assert getattr(chained, "CHAIN_MARKER", None) == "image"
    assert "__rkt_planted_ran__" not in sys.modules
    assert sys.path == [str(ws), str(image)]   # the path is put back untouched


def test_chain_skips_a_planted_sitecustomize_by_cwd_alone(tmp_path, monkeypatch, startup_env):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "sitecustomize.py").write_text(
        "import sys as _s\n_s.modules['__rkt_planted_ran__'] = True\n", encoding="utf-8")
    image = tmp_path / "site-packages"
    image.mkdir()
    (image / "sitecustomize.py").write_text("CHAIN_MARKER = 'image'\n", encoding="utf-8")
    monkeypatch.delenv("RKT_MON_ROOT", raising=False)   # no root given: cwd is the guard
    monkeypatch.chdir(ws)
    sys.path[:] = [str(ws), str(image)]
    _load_runtime_sitecustomize()
    assert getattr(sys.modules["sitecustomize"], "CHAIN_MARKER", None) == "image"
    assert "__rkt_planted_ran__" not in sys.modules


def test_chain_still_reaches_the_harness_sitecustomize(tmp_path, monkeypatch, startup_env):
    image = tmp_path / "site-packages"
    image.mkdir()
    (image / "sitecustomize.py").write_text("CHAIN_MARKER = 'image'\n", encoding="utf-8")
    monkeypatch.setenv("RKT_MON_ROOT", str(tmp_path / "ws"))   # ws itself has none
    sys.path[:] = [str(image)]
    ours = _load_runtime_sitecustomize()
    assert getattr(sys.modules["sitecustomize"], "CHAIN_MARKER", None) == "image"
    assert sys.modules["sitecustomize"] is not ours


def test_chain_with_nothing_to_import_restores_ours(tmp_path, monkeypatch, startup_env):
    monkeypatch.delenv("RKT_MON_ROOT", raising=False)
    sys.path[:] = [str(tmp_path / "empty")]    # no sitecustomize anywhere on it
    ours = _load_runtime_sitecustomize()
    assert sys.modules["sitecustomize"] is ours
