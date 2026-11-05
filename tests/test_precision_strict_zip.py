# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""BUG-SEED-24 — PrecisionRecorder.measure silently truncated mismatched inputs.

``zip(outputs, ref_outputs, truth)`` without ``strict=True`` stops at the shortest
list: one output against two reference/truth tensors produced a single judged row
and a precision block that read "pass", hiding the alignment break that makes any
row meaningless. The zip is strict now, so a length mismatch lands in measure()'s
existing handler as an explicit unavailable row — the judge never crashes on it.

The mismatch case is adapted from the audit archive's
test_guard_fixes.py::test_measure_with_mismatched_output_counts_is_unavailable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
_precision_spec = importlib.util.spec_from_file_location(
    "pk_template_precision", REPO / "src" / "peregrine_kernel" / "card_template" / "eval" / "precision.py")
P = importlib.util.module_from_spec(_precision_spec)
_precision_spec.loader.exec_module(P)


def test_a_length_mismatch_is_an_unavailable_row_never_a_false_pass():
    x = torch.randn(4)
    rec = P.PrecisionRecorder()
    rec.measure("short", [x, x], [x], [x, x])      # the reference side is shorter
    row = rec.rows[-1]
    assert row["status"] == "unavailable"
    assert "ValueError" in row["why"]              # the strict-zip mismatch, on the record


def test_an_exact_match_still_judges_normally():
    x = torch.randn(4)
    rec = P.PrecisionRecorder()
    rec.measure("exact", [x], [x], [x])
    assert rec.rows[-1]["status"] in ("pass", "fail")


def test_a_mismatch_is_on_the_record_not_a_judge_crash():
    # the failure stays inside measure(): rows carry it, block() renders it, and
    # nothing raises out of either
    x = torch.randn(4)
    rec = P.PrecisionRecorder()
    rec.measure("long", [x], [x, x], [x, x])       # the reference side is longer
    assert rec.rows[-1]["status"] == "unavailable"
    block = rec.block()
    assert block["rows"][-1]["status"] == "unavailable"
    assert any("zip()" in r.get("why", "") for r in block["rows"])


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
