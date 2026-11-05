# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit-corps R3F fix wave over loop/kbmenu.py.

Covered here:
- BUG-CH-F-01: the rule table's ``_num(...) or <high sentinel>`` reads a
  measured 0.0 as a missing key: 0.0% L2 hit rate, 0.0% compute SOL and
  0.0 waves/SM are exactly the strongest form of the symptoms l2-low,
  prefetch-lead, latency-bound, unbound-attribution and waves-low exist to
  catch, yet every one of them stayed silent. A missing key must still read
  as "not measured" (the sentinel); a measured 0.0 must compare as itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsi_kernel_tune.loop import kbmenu


# measured 0.0 on every key the five rules consult (dram/tensor ride along so
# the memory-side thresholds have realistic company)
ZERO_READINGS = {"l2_hit_rate_pct": 0.0, "compute_sol_pct": 0.0,
                 "memory_sol_pct": 30.0, "waves_per_sm": 0.0, "tensor_pct": 0.0,
                 "dram_pct": 30.0}


def test_zero_readings_fire_the_rules_that_exist_for_them():
    """l2_hit_rate_pct=0.0 / compute_sol_pct=0.0 / waves_per_sm=0.0 are
    measurements, not gaps: l2-low, prefetch-lead, latency-bound,
    unbound-attribution and waves-low all fire on them."""
    fired = set(kbmenu.fired_rule_names(ZERO_READINGS, {}))
    assert {"l2-low", "prefetch-lead", "latency-bound",
            "unbound-attribution", "waves-low"} <= fired, fired


def test_zero_readings_render_into_the_menu():
    """The menu itself is not empty on an all-zero SOL profile (it used to
    render the empty string, so the teacher got no injection at all)."""
    menu = kbmenu.build_kb_menu(readings=ZERO_READINGS, resusage={})
    assert isinstance(menu, str) and menu.strip()
    assert "l2_hit_rate_pct=0" in menu


def test_missing_keys_stay_not_measured():
    """The sentinel semantics survive: a key that is absent (or non-numeric)
    still reads as not measured -- no rule fires on a gap."""
    fired = set(kbmenu.fired_rule_names({}, {}))
    assert not ({"l2-low", "prefetch-lead", "latency-bound",
                 "unbound-attribution", "waves-low"} & fired), fired


def test_small_but_nonzero_l2_hits_still_fire():
    """Parity with the pre-fix behavior on ordinary readings: 5% L2 hit rate
    fires l2-low and prefetch-lead as it always did."""
    fired = set(kbmenu.fired_rule_names({"l2_hit_rate_pct": 5.0,
                                         "memory_sol_pct": 80.0}, {}))
    assert {"l2-low", "prefetch-lead"} <= fired, fired
