"""Regression test for the audit-corps fix wave: a quarantined KB entry keeps
its tree's name in the quarantine directory.

BUG-SEED-49: the quarantine hangs off the kb repo root, shared by every tree
(``sm90/``, ``sm100/``, ...), but the entry was banked under ``src.name`` and
POSIX rename silently overwrites. Quarantining the same-named entry from two
trees left ONE file behind — the other tree's evidence vanished without a
trace, from the exact mechanism whose job is to preserve it for review by
hand. The banked name now carries the finding's relative path
(``<arch>__<entry>.md``).
"""

from __future__ import annotations

from pathlib import Path

from peregrine_kernel.loop.driver import _quarantine_leaky_kb
from test_loop_driver import FakeBackend, _ctx


def _kb(tmp_path: Path, *arches: str) -> Path:
    kb = tmp_path / "hw_kb" / "cutedsl"
    for a in arches:
        (kb / a).mkdir(parents=True, exist_ok=True)
        (kb / a / "index.md").write_text("# index\n", encoding="utf-8")
    return kb


def test_same_named_entries_from_two_trees_both_survive_the_quarantine(tmp_path):
    """The BUG-SEED-49 incident shape: sm90/overview.md and sm100/overview.md
    both fail the certify lint. Both must sit in the quarantine afterwards,
    each still readable — not one file holding whichever tree moved last."""
    kb = _kb(tmp_path, "sm90", "sm100")
    (kb / "sm90" / "overview.md").write_text("sm90 recipe: extracted from the teacher oracle\n",
                                             encoding="utf-8")
    (kb / "sm100" / "overview.md").write_text("sm100 recipe: extracted from the teacher oracle\n",
                                              encoding="utf-8")
    qdir = tmp_path / "kb_quarantine"

    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm90"
    _quarantine_leaky_kb(ctx, ["sm90/overview.md:1: banned narrative"])
    ctx.kb_arch = "sm100"
    _quarantine_leaky_kb(ctx, ["sm100/overview.md:1: banned narrative"])

    sm90_q = qdir / "sm90__overview.md"
    sm100_q = qdir / "sm100__overview.md"
    assert sm90_q.is_file() and "sm90 recipe" in sm90_q.read_text(encoding="utf-8")
    assert sm100_q.is_file() and "sm100 recipe" in sm100_q.read_text(encoding="utf-8")
    # both trees are clean: nothing leaky is left student-readable
    assert not (kb / "sm90" / "overview.md").exists()
    assert not (kb / "sm100" / "overview.md").exists()


def test_a_deep_finding_keeps_its_whole_relative_path(tmp_path):
    rel = "sm90/sub/dir/entry.md"
    kb = _kb(tmp_path, "sm90")
    (kb / "sm90" / "sub" / "dir").mkdir(parents=True, exist_ok=True)
    (kb / "sm90" / "sub" / "dir" / "entry.md").write_text("deep leak\n", encoding="utf-8")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = kb
    ctx.kb_arch = "sm90"
    _quarantine_leaky_kb(ctx, [f"{rel}:3: banned narrative"])
    banked = tmp_path / "kb_quarantine" / "sm90__sub__dir__entry.md"
    assert banked.is_file() and banked.read_text(encoding="utf-8") == "deep leak\n"
