"""Regression pins for the BUG-CORPS-C-05 fix (audit wave R3C).

A card that is not a git repository cannot feed the harness overlay/revert guards:
``card_harness_files`` returns None and both silently degrade to "the job runs the run
tree's own copies" — the exact self-judged eval a delivery that edits ``eval/`` wants.
``--no-git`` into the live repository is therefore refused unless the degradation is
named on purpose (``--allow-unguarded-harness``); an explicit ``--base-dir`` is a
staging/test target by contract and is warned, not refused.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rsi_kernel_tune import scaffold

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """An isolated main-repo root holding copies of the real templates."""
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "meta")
    shutil.copytree(REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template",
                    tmp_path / "src" / "rsi_kernel_tune" / "card_template")
    return tmp_path


def test_no_git_into_the_live_repo_is_refused_until_named(tmp_path, capsys, monkeypatch):
    """``--base-dir`` unset: the target is the live repository's operators/, where
    runs happen — refuse before creating anything."""
    monkeypatch.setattr(scaffold, "_repo_root", lambda: tmp_path)
    rc = scaffold.main(["demo_op", "--no-git"])
    assert rc == 2
    assert not (tmp_path / "operators" / "demo_op").exists()   # refused before creating
    err = capsys.readouterr().err
    assert "--allow-unguarded-harness" in err
    assert "harness overlay/revert guards" in err


def test_the_explicit_opt_in_creates_the_card(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(scaffold, "_repo_root", lambda: tmp_path)
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "meta")
    shutil.copytree(REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template",
                    tmp_path / "src" / "rsi_kernel_tune" / "card_template")
    rc = scaffold.main(["demo_op", "--no-git", "--allow-unguarded-harness"])
    assert rc == 0
    assert (tmp_path / "operators" / "demo_op" / "README.md").is_file()
    assert not (tmp_path / "operators" / "demo_op" / ".git").exists()
    assert "WARNING" not in capsys.readouterr().err


def test_a_staging_base_dir_is_warned_not_refused(tmp_path, capsys):
    base = tmp_path / "staging"
    shutil.copytree(REPO_ROOT / "meta", base / "meta")
    shutil.copytree(REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template",
                    base / "src" / "rsi_kernel_tune" / "card_template")
    rc = scaffold.main(["demo_op", "--base-dir", str(base), "--no-git"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "guards" in err
    assert (base / "operators" / "demo_op" / "README.md").is_file()


def test_the_default_still_inits_git(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "_repo_root", lambda: tmp_path)
    shutil.copytree(REPO_ROOT / "meta", tmp_path / "meta")
    shutil.copytree(REPO_ROOT / "src" / "rsi_kernel_tune" / "card_template",
                    tmp_path / "src" / "rsi_kernel_tune" / "card_template")
    rc = scaffold.main(["demo_op"])
    assert rc == 0
    assert (tmp_path / "operators" / "demo_op" / ".git").is_dir()
