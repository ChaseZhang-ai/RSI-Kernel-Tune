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
