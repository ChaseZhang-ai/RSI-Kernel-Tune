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

from test_milestone import VALID as MILESTONE_VALID  # noqa: E402
from peregrine_kernel.milestone import validate_milestone  # noqa: E402


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


# --- BUG-SEED-26: validate_milestone's section presence must be heading-anchored ------

def test_seed26_a_low_level_heading_cannot_masquerade_as_a_required_section():
    """'### 4. M2 pipeline' contains '## 4.' as a substring; a plain text.find
    accepts it in place of the real '## 4.' section heading, while _section_body
    (which validates §3's extent) already matches at line start only."""
    masquerade = MILESTONE_VALID.replace("## 4. Known dead ends",
                                         "### 4. M2 pipeline notes")
    v = validate_milestone(masquerade)
    assert any("missing required section: ## 4." in x for x in v)


def test_seed26_a_real_section_heading_still_passes():
    """The anchored check must not reject the legitimate heading shapes."""
    assert not any("missing required section" in x
                   for x in validate_milestone(MILESTONE_VALID))


def test_seed26_the_order_check_uses_the_anchored_offsets_too():
    """Swapping two real headings is still caught as out of order (the order
    comparison shares the presence offsets)."""
    swapped = MILESTONE_VALID.replace(
        "## 1. Problem contract (Math Spec)", "@@ONE@@").replace(
        "## 2. Decision DAG + phase order", "## 1. Problem contract (Math Spec)"
    ).replace("@@ONE@@", "## 2. Decision DAG + phase order")
    v = validate_milestone(swapped)
    assert any("out of order" in x for x in v)
