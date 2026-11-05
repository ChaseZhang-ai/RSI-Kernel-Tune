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
