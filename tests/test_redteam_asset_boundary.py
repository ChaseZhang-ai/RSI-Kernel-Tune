# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""WP20R task 3: the red-team asset boundary (contract C16), pinned as a
regression test.

C16: the cheat fixtures live only under ``tests/redteam/``; the technique
descriptions exist only in the fixtures' comments and the WP20R CHANGES
record; nothing leaks into student-visible paths (``src/`` including
``card_template/``, ``meta/``, ``skills/``, ``scripts/``, ``kernels/``,
``tasks/``, ``hw_kb/``, ``global_config/``) or the docs.

This test pins three facts:

1. the fixtures' technique markers (the attack-code identifiers and the parked
   artifact names) appear in NO tracked file outside ``tests/`` and the
   CHANGES records — a future edit that copies an attack shape into src/,
   a teacher prompt, or a student-readable template fails here;
2. every red-team fixture runs by default: no ``pytest.mark.skip`` /
   ``pytest.mark.xfail`` / inline ``pytest.skip`` in the matrix (the module
   level ``pytest.importorskip("torch")`` is an environment guard, not a case
   skip — the box this suite runs on has torch);
3. the matrix is collected by the default pytest run at its WP20R delivery
   size or larger — shrinking the matrix must be a reviewed decision, never a
   silent drop.

The WP9 defense-side scaffold (``tests/test_fresh_recheck.py``) predates the
red-team directory and hosts the recheck defense's own cheat fixture; it is
inside the sanctioned ``tests/`` exclusion, as the red-team conftest documents.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REDTEAM = REPO / "tests" / "redteam"

# Strings that identify the red-team attack shapes: the cheat kernel symbols
# and the side-channel artifact filenames the fixtures park their judged
# answers in. Defense-side vocabulary (fresh recheck, stream sentinel, ...) is
# deliberately NOT listed — the defenses themselves document what they defeat.
MARKERS = (
    "CHEAT_KERNEL",
    "FALLBACK_WRONG_FASTPATH",
    "FALLBACK_DEAD_FASTPATH",
    ".side_buffer.pt",
    ".ptr_buffer.pt",
    "cheater's bet",
)

# Sanctioned homes of the technique material: the red-team fixtures themselves,
# and the CHANGES records (contract C16's allowed surfaces).
SANCTIONED_PREFIXES = ("tests/",)
SANCTIONED_FILES = re.compile(r"^CHANGES-.*\.md$")


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                         capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def test_technique_markers_stay_inside_the_redteam_directory():
    """No tracked file outside tests/ and the CHANGES records carries a
    red-team technique marker — the boundary holds over the whole shipped tree
    (src/, meta/, skills/, scripts/, kernels/, tasks/, hw_kb/, global_config/,
    docs/ included)."""
    leaks = []
    for rel in _tracked_files():
        if any(rel.startswith(pre) for pre in SANCTIONED_PREFIXES):
            continue
        if SANCTIONED_FILES.match(rel):
            continue
        blob = (REPO / rel).read_bytes()
        for marker in MARKERS:
            if marker.encode("utf-8") in blob:
                leaks.append(f"{rel}: {marker!r}")
    assert not leaks, (
        "red-team technique markers leaked outside tests/redteam/ (contract "
        f"C16): {leaks}")


def test_the_fixtures_carry_no_skip_or_xfail():
    """Every red-team fixture runs in the default suite — skipping a cheat
    silently un-verifies the defense it exercises (WP20R task 3)."""
    offenders = []
    for f in sorted(REDTEAM.glob("test_*.py")):
        text = f.read_text(encoding="utf-8")
        for banned in ("pytest.mark.skip", "pytest.mark.xfail",
                       "pytest.skip(", "pytest.xfail("):
            if banned in text:
                offenders.append(f"{f.name}: {banned}")
    assert not offenders, f"the red-team matrix must not skip: {offenders}"


def test_the_matrix_is_collected_at_its_delivery_size():
    """The default pytest run collects the whole matrix (the WP20R delivery
    size is the floor): the cheat set cannot shrink without a review."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/redteam",
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True, check=True)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    assert m, f"collection produced no count: {out.stdout[-400:]}"
    collected = int(m.group(1))
    assert collected >= 40, (
        f"the red-team matrix shrank below its WP20R delivery size: "
        f"{collected} collected, floor 40")
