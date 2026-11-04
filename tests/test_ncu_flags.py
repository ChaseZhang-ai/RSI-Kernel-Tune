"""WP9 task 1: every NCU command generation point carries the regime-alignment flags.

The timing protocol is warm-L2 + the card's own clock state; NCU's defaults
(flush all caches every pass, lock to base clocks) describe a different machine.
Three generation points exist — the two sealed-reference recon trees and the
generated card's gpu_entry template — and all three must opt out of both
defaults, or a teacher reads l2_hit_rate / utilization pct from a regime the
gate numbers never ran in.
"""

from __future__ import annotations

import re
from pathlib import Path

from peregrine_kernel import binref, cardkit, sealref

REPO_ROOT = Path(__file__).resolve().parents[1]

# One ncu invocation of a generated script, in order: the flags WP9 adds sit
# between --profile-from-start and --launch-count on every generation point.
_CMD_RE = re.compile(
    r"ncu\b.*?--profile-from-start\s+off\s+\\\n?\s*--cache-control\s+none\s+"
    r"--clock-control\s+none\s+\\\n?\s*--launch-count\s+\d+", re.S)


def _ncu_commands(text: str) -> list[str]:
    """Every `ncu ...` invocation line block the generated script carries."""
    return [m.group(0) for m in re.finditer(r"ncu_bounded ncu [^;]*?eval/ncu_run\.py", text, re.S)]


# A minimal but well-formed card gpu_entry.sh: _split_dispatch keys on the
# `correctness` task line, guard_card's guard block is stripped again here.
_MIN_CARD = ('#!/bin/bash\n# card\nset -euo pipefail\n'
             'case "$TASK" in\n'
             '    correctness) run_with_timeout python3 eval/bench.py --mode correctness ;;\n'
             '    cmd)         run_with_timeout bash -c "$*" ;;\n'
             'esac\n')


def test_sealref_recon_tree_ncu_carries_the_flags() -> None:
    text = sealref.ref_tree_gpu_entry(_MIN_CARD, "mod", "KEY")
    cmds = _ncu_commands(text)
    assert cmds, "sealref recon tree lost its ncu task"
    assert all(_CMD_RE.search(c) for c in cmds)


def test_binref_recon_tree_ncu_carries_the_flags() -> None:
    text = binref.ref_tree_gpu_entry(_MIN_CARD, ["kernel.cu"])
    cmds = _ncu_commands(text)
    assert len(cmds) == 2, "binref recon tree's ncu + binref tasks"
    assert all(_CMD_RE.search(c) for c in cmds)


def test_gpu_entry_template_ncu_carries_the_flags() -> None:
    text = cardkit.render_gpu_entry(name="op", suite="peregrine_kernel", arch="sm_100a",
                                    clock="1500", mem_clock="3996", require_lock="1",
                                    clock_note="n")
    cmds = _ncu_commands(text)
    assert cmds
    assert all(_CMD_RE.search(c) for c in cmds)
    # The launch-count post-edit in taskcard.write_card keys on the literal
    # "--launch-count 24": keep it, or every card silently keeps 24.
    assert "--launch-count 24" in text


def test_generated_card_entry_carries_the_flags() -> None:
    """End to end through the same .replace() taskcard.write_card applies."""
    text = cardkit.render_gpu_entry(name="op", suite="peregrine_kernel", arch="sm_100a",
                                    clock="1500", mem_clock="3996", require_lock="1",
                                    clock_note="n")
    text = text.replace("--launch-count 24", "--launch-count 48")
    cmds = _ncu_commands(text)
    assert all(_CMD_RE.search(c) for c in cmds)


def test_repo_has_no_unflagged_ncu_generation_point_left() -> None:
    """Grep the source tree the way the WP9 brief did: any ncu invocation naming
    --profile-from-start without --cache-control none is a missed generation point."""
    missed = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"--profile-from-start", text):
            window = text[max(0, m.start() - 200):m.end() + 300]
            if "--cache-control" not in window:
                missed.append(str(path))
    assert missed == [], f"ncu generation points without the WP9 flags: {missed}"


def test_ncu_flag_ab_script_dry_run(tmp_path) -> None:
    """The A/B script's dry run: argument handling + the metrics it would use."""
    import subprocess
    import sys
    op = tmp_path / "op"
    (op / "eval").mkdir(parents=True)
    (op / "gpu_entry.sh").write_text(
        'NCU_METRICS="gpu__time_active.sum,lts__t_sector_hit_rate.pct"\n', encoding="utf-8")
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "ncu_flag_ab.py"), str(op)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--cache-control none" in r.stdout and "dry run" in r.stdout
