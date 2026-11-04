"""Regression tests for the W1 audit fix W1-LIF-01 (monitor CLI kernel-file default):

* the standalone ``peregrine <op> monitor`` CLI resolves --kernel-file from the
  card's own problem.json (cuda -> kernel.cu) instead of hardcoding kernel.py --
  monitor/cli.py.

(The wave's SEED-14 / SEED-15 / SEED-41 / SEED-42 faces are covered by the
 corps' own tests on this branch; only the W1-unique LIF-01 cases live here.
 The _chain hardening cases are appended by their own commit below.)

Ported from the R2 audit's untracked test batch, rewritten for the
peregrine_kernel tree. CPU only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# --- W1-LIF-01: the CLI defaults to the card's own kernel file ---------------------
def test_monitor_cli_defaults_to_the_cards_kernel_file(tmp_path, capsys):
    from peregrine_kernel.monitor import cli

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
    from peregrine_kernel.monitor import cli

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
    from peregrine_kernel.monitor import cli

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
