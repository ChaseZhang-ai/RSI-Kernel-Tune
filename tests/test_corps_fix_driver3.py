"""Corps-fix wave 8 regression tests (driver.py / parse.py misc batch).

Each test pins one audit finding from handoff/triage-wk6/triage-T5.md
(BUG-SEED-32/33/34/35/49, BUG-T5C-01) and triage-T1.md (BUG-T1C-01).
"""
import json
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

from peregrine_kernel.inner.trajectory import IterationRecord, append_record
from peregrine_kernel.loop.driver import _save_ref_struct_table
from peregrine_kernel.loop.parse import (
    RESOLVABLE_DIFF_CAP,
    RESOLVABLE_DIFF_FLOOR,
    resolvable_diff,
)

from test_loop_driver import FakeBackend, _ctx


# --- BUG-T5C-01: _save_ref_struct_table's OSError log lost the destination ---


def test_struct_table_write_failure_log_names_the_destination(tmp_path, capsys):
    """Observed: the OSError log read `could not persist ref_struct_table.txt to
    {e}` — the exception text sat in the destination slot and the path it tried
    to write was absent, so a write failure said what went wrong but never where.
    The sibling bank logs all read `could not persist <name> to <dst>: <e>`."""
    from peregrine_kernel.loop.driver import _REF_STRUCT_TABLE_NAME

    recon = tmp_path / "recon"
    recon.mkdir()
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = recon / "report.md"
    # the destination is a directory: write_text fails with an OSError whose
    # message does not contain the path — only the log can carry it
    (recon / _REF_STRUCT_TABLE_NAME).mkdir()
    log = ("=== STRUCT TABLE BEGIN ===\nphase table body\n"
           "=== STRUCT TABLE END ===\n")
    _save_ref_struct_table(ctx, log)
    out = capsys.readouterr().out
    dst = recon / _REF_STRUCT_TABLE_NAME
    assert f"could not persist {_REF_STRUCT_TABLE_NAME} to {dst}:" in out, out
    assert "Is a directory" in out, "the error text stays in its own slot"


def test_struct_table_bank_still_writes_on_a_normal_path(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.recon_path = tmp_path / "recon" / "report.md"
    ctx.recon_path.parent.mkdir()
    log = ("=== STRUCT TABLE BEGIN ===\nphase table body\n"
           "=== STRUCT TABLE END ===\n")
    _save_ref_struct_table(ctx, log)
    assert (tmp_path / "recon" / "ref_struct_table.txt").read_text(
        encoding="utf-8") == "phase table body"


# --- BUG-SEED-49: _quarantine_leaky_kb collapsed same-basename entries --------


def _kb_ctx(tmp_path):
    """A ctx whose kb repo root is tmp_path/repo (the quarantine's parent)."""
    from peregrine_kernel.loop import lang_mode

    ctx = _ctx(tmp_path, FakeBackend([], []))
    repo = tmp_path / "repo"
    kb = repo / "hw_kb" / "cutedsl"
    (kb / "sm100").mkdir(parents=True)
    (kb / "hopper").mkdir(parents=True)
    ctx.kb_dir = kb
    assert lang_mode.kb_repo_root(kb) == repo
    return ctx, kb


def test_quarantine_keeps_same_basename_entries_from_different_archs(tmp_path):
    """Observed: dst = qdir / src.name dropped the <arch>/ prefix, so two
    subtrees' same-named leaky entries renamed onto one file — rename(2)
    replaces silently, the earlier entry vanished from both its tree and the
    quarantine, and the reviewing human saw one leak, not two."""
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    (kb / "sm100" / "leak.md").write_text("sm100 leak body\n", encoding="utf-8")
    (kb / "hopper" / "leak.md").write_text("hopper leak body\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/leak.md:1: banned narrative",
                               "hopper/leak.md:1: banned narrative"])
    qdir = tmp_path / "repo" / "kb_quarantine"
    landed = sorted(p.name for p in qdir.iterdir())
    assert len(landed) == 2, landed
    assert not (kb / "sm100" / "leak.md").exists()
    assert not (kb / "hopper" / "leak.md").exists()
    bodies = {p.name: p.read_text(encoding="utf-8")
              for p in qdir.iterdir()}
    assert "sm100 leak body\n" in bodies.values()
    assert "hopper leak body\n" in bodies.values()


def test_quarantine_does_not_replace_a_previous_run_s_entry(tmp_path):
    """A second quarantine (a later run) with the same entry name must not
    replace the file the first run left for the human — it flattens its own
    relative path beside it."""
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    qdir = tmp_path / "repo" / "kb_quarantine"
    qdir.mkdir(parents=True)
    (qdir / "leak.md").write_text("earlier run's leak\n", encoding="utf-8")
    (kb / "sm100" / "leak.md").write_text("this run's leak\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/leak.md:1: banned narrative"])
    assert (qdir / "leak.md").read_text(encoding="utf-8") == "earlier run's leak\n"
    assert (qdir / "sm100__leak.md").read_text(encoding="utf-8") == "this run's leak\n"


def test_quarantine_unique_names_stay_flat(tmp_path):
    """The incident fix must not rename the common case: a unique basename with
    no collision keeps its plain quarantine name (the existing consumers and
    the human-facing layout are unchanged)."""
    from peregrine_kernel.loop.driver import _quarantine_leaky_kb

    ctx, kb = _kb_ctx(tmp_path)
    (kb / "sm100" / "oddity.md").write_text("body\n", encoding="utf-8")
    _quarantine_leaky_kb(ctx, ["sm100/oddity.md:1: banned narrative"])
    assert (tmp_path / "repo" / "kb_quarantine" / "oddity.md").is_file()
