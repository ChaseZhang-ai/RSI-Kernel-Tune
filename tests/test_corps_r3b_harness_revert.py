"""BUG-CORPS-C-11 regression: the driver's harness revert feeds git a ``-z`` diff.

``harness_changes``/``harness_drift`` used to run ``git diff --name-only`` without
``-z`` and split on newlines, so a non-ASCII harness path arrived in git's quoted
spelling (``"eval/b\\303\\251nch.py"``), matched no harness prefix, and a delivery
that edited the judge escaped the REVERT. The overlay side (``card_harness_files``,
``tree_harness_files``) has always read its listings with ``-z`` and was immune —
this pins the same caliber on the driver side, revert included.
"""

import subprocess
from pathlib import Path

from rsi_kernel_tune.loop import cli

ACCENT = "bénch.py"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _card_and_ws(tmp_path: Path):
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    (card / "kernel.py").write_text("def run(): pass\n", encoding="utf-8")
    (card / "gpu_entry.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (card / "eval" / "bench.py").write_text("print('judge')\n", encoding="utf-8")
    (card / "eval" / ACCENT).write_text("print('accent judge')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(card)], check=True)
    _commit(card, "card")
    ws = tmp_path / "ws"
    subprocess.run(["git", "clone", "-q", str(card), str(ws)], check=True)
    return card, ws


def _backend(ws: Path) -> cli.RealBackend:
    be = cli.RealBackend("demo", ws, None, "t", "t@t")
    be.kernel_file = "kernel.py"
    return be


def test_an_edited_escaped_harness_file_is_caught_and_reverted(tmp_path: Path):
    """A delivery that edits eval/bénch.py is reported by harness_changes and
    restored by restore_harness — it must not escape the REVERT."""
    card, ws = _card_and_ws(tmp_path)
    be = _backend(ws)
    anchor = be.head()
    (ws / "eval" / ACCENT).write_text("print('PASS always')\n", encoding="utf-8")
    head = _commit(ws, "tamper")
    assert be.harness_changes(anchor, head) == [f"eval/{ACCENT}"]
    be.restore_harness(anchor, [f"eval/{ACCENT}"], "restore")
    assert (ws / "eval" / ACCENT).read_text(encoding="utf-8") == "print('accent judge')\n"


def test_an_untracked_escaped_harness_file_is_drift(tmp_path: Path):
    """The launch drift preflight sees an untracked non-ASCII harness file too."""
    card, ws = _card_and_ws(tmp_path)
    be = _backend(ws)
    anchor = be.head()
    (ws / "eval" / "zénew.py").write_text("x\n", encoding="utf-8")
    drift = be.harness_drift(anchor)
    # the tracked bénch.py matches the anchor (clone is clean); the new untracked
    # non-ASCII file is what drift must surface — before -z it arrived quoted
    assert drift == ["eval/zénew.py"]


def test_untouched_harness_reads_empty_as_before(tmp_path: Path):
    """No non-ASCII names, no changes: the -z switch changes nothing."""
    card, ws = _card_and_ws(tmp_path)
    be = _backend(ws)
    anchor = be.head()
    assert be.harness_changes(anchor, anchor) == []
    assert be.harness_drift(anchor) == []
