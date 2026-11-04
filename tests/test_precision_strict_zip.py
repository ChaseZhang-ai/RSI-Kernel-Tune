# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""The precision measure's zip is strict (the unequal-length fix).

The exposure: ``PrecisionRecorder.measure`` zipped ``(outs, refs, truths)``
without ``strict=True``, so a hand-written bench handing in unequal-length
lists got a SILENT TRUNCATION -- the unpaired outputs produced no row at all,
and a gate that reads only ``status == "fail"`` rows saw a clean block. The
recorder's "never raises" contract then swallowed the truncation into a fake
pass. With ``strict=True`` the mismatch raises inside ``measure`` and lands in
the existing ``except``: an ``unavailable`` row for the case -- fail-closed,
no new code path, nothing silently dropped.

Runs on CPU (torch required).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_EVAL = REPO / "src" / "peregrine_kernel" / "card_template" / "eval"

torch = pytest.importorskip("torch")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("pk_precision_strict_zip", TEMPLATE_EVAL / "precision.py")


def _unavailable_why(rec, case):
    rows = [r for r in rec.rows if r.get("case") == case
            and r.get("status") == "unavailable"]
    return rows[0]["why"] if len(rows) == 1 else None


def test_unequal_output_lists_leave_an_unavailable_row_not_a_silent_pass():
    """Two kernel outputs against one reference output: the unmeasured second
    output used to vanish (one pass row, no unavailable row, nothing said).
    The mismatch must now fail the case closed."""
    rec = P.PrecisionRecorder()
    rec.measure("case0", [torch.zeros(2), torch.ones(3)], [torch.zeros(2)],
                [torch.zeros(2)], names=["out0", "out1"])
    why = _unavailable_why(rec, "case0")
    assert why is not None, "the length mismatch was truncated silently"
    assert why.startswith("ValueError:")
    assert "zip()" in why


def test_unequal_truth_lists_leave_an_unavailable_row_not_a_silent_pass():
    """The strictness covers every side of the zip: a truth list longer than
    the outputs must surface the same way, not truncate the extra truths."""
    rec = P.PrecisionRecorder()
    rec.measure("case1", [torch.zeros(2)], [torch.zeros(2)],
                [torch.zeros(2), torch.ones(3)])
    why = _unavailable_why(rec, "case1")
    assert why is not None, "the length mismatch was truncated silently"
    assert why.startswith("ValueError:")
    assert "zip()" in why


def test_equal_length_multi_output_still_measures_every_output():
    """The strict zip must not break the honest multi-output path: equal-length
    lists still yield one row per output and no unavailable row."""
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    rec = P.PrecisionRecorder()
    rec.measure("case2", [a.clone(), b.clone()], [a, b], [a.clone(), b.clone()],
                names=["out0", "out1"])
    assert [r["output"] for r in rec.rows] == ["out0", "out1"]
    assert all(r["status"] == "pass" for r in rec.rows)
