"""BUG-SEED-23 — host_path_note named the wrong side of the enqueue comparison.

Both audit lines fixed the direction word; the merge kept the r3 line's
implementation, whose parenthetical speaks about the REFERENCE ("the in-job
reference enqueues in N ms (...)" — so the word follows the reference relative
to the student's wrapper: a lighter reference reads "thinner", a heavier one
"thicker", an equal one "the same thickness"). The direction must always match
the actual ratio — that is the one fact the note exists to record. The
operator-facing counterpart host_path_feedback stays direction-neutral by
design ("yours is N x that") — that neutrality is pinned here too.

The two-direction case is adapted from the audit archive's
test_guard_fixes.py::test_host_path_note_names_both_directions, so the behavior
stays pinned after the archive was not ported wholesale.
"""

from __future__ import annotations

import pytest

from rsi_kernel_tune.baseline import BASIS_CUPTI
from rsi_kernel_tune.timing_audit import host_path_feedback, host_path_note

# A span-basis reading the note fires on (share of the window >= the 0.5 warn ratio).
DEVICE_MS = 0.01522


def test_a_wrapper_thicker_than_the_reference_marks_the_reference_thinner():
    # ratio = ours / ref = 3.0: the reference is the lighter side, and the
    # parenthetical (whose subject is the reference) says exactly that
    note = host_path_note(3.0, 1.0, basis=BASIS_CUPTI, ref_host_enqueue_ms=1.0)
    assert "3.0x thinner" in note
    assert "thicker" not in note


def test_a_wrapper_thinner_than_the_reference_marks_the_reference_thicker():
    # ratio = 0.5: the reference enqueues twice as much — it is the thicker side
    note = host_path_note(0.5, 1.0, basis=BASIS_CUPTI, ref_host_enqueue_ms=1.0)
    assert "2.0x thicker" in note
    assert "thinner" not in note


def test_the_note_names_both_directions_and_never_the_wrong_one():
    # the archive's live-reading case: a thin wrapper next to a fat reference,
    # and a fat wrapper next to a thin reference, on the same device reading
    thin = host_path_note(0.02, DEVICE_MS, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.06463)
    thick = host_path_note(0.08, DEVICE_MS, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.02)
    assert "3.2x thicker" in thin         # ref 0.06463 vs ours 0.02: the reference is bigger
    assert "4.0x thinner" in thick        # ref 0.02 vs ours 0.08: the reference is smaller
    assert "thinner" not in thin and "thicker" not in thick


def test_an_equal_wrapper_says_the_same_thickness():
    # the ratio == 1.0 boundary: neither direction word applies
    note = host_path_note(0.08, DEVICE_MS, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.08)
    assert "the same thickness" in note
    assert "thicker" not in note and "thinner" not in note


def test_the_student_feedback_stays_direction_neutral():
    # host_path_feedback never took a side ("yours is N x that"); it must not grow
    # the note's thicker/thinner wording (or lose the neutral ratio) either.
    fb = host_path_feedback(0.08, DEVICE_MS, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.02)
    assert "4.0x that" in fb
    assert "thicker" not in fb and "thinner" not in fb


@pytest.mark.parametrize("host,ref", [(0.0, 1.0), (1.0, 0.0)])
def test_a_zero_reference_or_wrapper_keeps_the_note_silent(host, ref):
    # a zero enqueue clock is not a direction; the ratio line simply does not render
    note = host_path_note(host, 1.0, basis=BASIS_CUPTI, ref_host_enqueue_ms=ref)
    assert "x thicker" not in note and "x thinner" not in note
