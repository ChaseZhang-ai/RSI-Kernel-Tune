"""Regression tests for the audit-corps miscellaneous fix wave (BUG-SEED-23/26/27/
36/38/39/40/44 and the assertion upgrades BUG-SEED-46 / BUG-C11-01).

Each test names the bug it pins so the ledger stays traceable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.baseline import BASIS_CUPTI  # noqa: E402
from peregrine_kernel.timing_audit import host_path_note  # noqa: E402


# --- BUG-SEED-23: the host-path note's thinner/thicker wording follows the direction --

def test_seed23_host_path_note_keeps_thinner_when_the_reference_is_faster():
    """ratio > 1: the student's wrapper is thicker, so the reference reads as
    thinner — the wording the pre-existing tests already pin."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.04 = 2.0
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.04)
    assert "2.0x thinner" in note
    assert "thicker" not in note


def test_seed23_host_path_note_says_thicker_when_the_reference_is_slower():
    """ratio < 1: the student's wrapper is the lighter one, so a fixed 'thinner'
    read backwards — the reference is thicker than the student's path."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.4 = 0.2
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.4)
    assert "5.0x thicker" in note
    assert "thinner" not in note
    assert "0.2x thinner" not in note


def test_seed23_host_path_note_handles_equal_enqueue_costs():
    """ratio == 1: neither direction, so neither thinner nor thicker may fire."""
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.08)
    assert "thinner" not in note and "thicker" not in note
    assert "the same thickness" in note
