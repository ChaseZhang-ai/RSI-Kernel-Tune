# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""FIX-R3G regression pins for BUG-CH-J-01 (``scripts/backtest_sh.py``).

The WP16 halving backtest's SH arm must actually run the successive-halving
rule. The pre-fix script handed the SH arm ``width * ROUNDS`` bench calls
while the arm walks the WIDE grid (``2 * width`` points): exactly one round
fits, while the elimination gate needs ``ROUNDS`` completed rounds with two
readings per survivor. The gate was therefore unreachable in every default
cell — the SH arm was process-identical to the plain wide sweep the "rule
cost" gate compared it against, and the gate measured sampling noise with
itself. The pins here:

* the per-cell budget funds the SH arm's whole walk (the wide-grid rounds
  plus the survivors' re-funded rounds);
* at that budget every simulated SH pass eliminates at least one variant —
  the rule runs and the two gated processes are distinct;
* the elimination criterion runs in the production ratio domain (the
  simulated surface's scores are positive, like the tool's speedup ratios);
* the pre-fix budget is detectably vacuous (zero eliminations), so the
  budget cannot silently regress to it;
* ``main`` fails a cell whose SH arm never eliminated (gate vacuous).
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "backtest_sh.py"


@pytest.fixture(scope="module")
def bt():
    spec = importlib.util.spec_from_file_location("rkt_corps_r3g_backtest_sh", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cell_budget_funds_the_sh_walk(bt):
    """The cell budget must fund ROUNDS rounds over the wide grid plus the
    survivors' re-funded ROUNDS rounds. The pre-fix ``width * ROUNDS`` bought
    the wide grid a single round and the elimination gate — which needs
    ROUNDS completed rounds — was unreachable."""
    for width in bt.GRIDS:
        assert bt.cell_budget(width) >= bt.ROUNDS * (2 * width) + bt.ROUNDS * width


def test_surface_scores_are_positive(bt):
    """The cut criterion is production's relative ratio (boundary/score - 1),
    defined on positive speedup scores; on a negative-valued surface the
    ratio inverts and every cut candidate "survives"."""
    for seed in range(20):
        truth, _best = bt.surface(16, random.Random(seed))
        assert min(truth) > 0.0


def test_sh_arm_eliminated_at_the_default_cells(bt):
    """At the script's per-cell budget the SH arm must eliminate at least one
    variant in every simulated pass — the elimination gate actually fires and
    SH is a distinct process from the plain wide sweep."""
    for width in bt.GRIDS:
        budget = bt.cell_budget(width)
        for sigma in bt.SIGMAS:
            for seed in range(12):
                truth_wide, _best = bt.surface(2 * width, random.Random(seed))
                _winner, cuts = bt.run_sh(truth_wide, bt.ROUNDS, budget, sigma,
                                          random.Random(seed))
                assert cuts >= 1, (width, sigma, seed)


def test_prefixed_budget_is_detectably_vacuous(bt):
    """The pre-fix budget (``width * ROUNDS``) buys the wide grid one round;
    the elimination gate needs ROUNDS completed rounds — nobody is ever
    eliminated, which is exactly the vacuity the fix removes."""
    for width in bt.GRIDS:
        budget = width * bt.ROUNDS
        for sigma in bt.SIGMAS:
            truth_wide, _best = bt.surface(2 * width, random.Random(0))
            _winner, cuts = bt.run_sh(truth_wide, bt.ROUNDS, budget, sigma,
                                      random.Random(0))
            assert cuts == 0, (width, sigma)


def test_main_fails_a_vacuous_cell(bt, monkeypatch):
    """A cell whose SH arm never eliminates must FAIL (gate vacuous) — not
    pass by comparing the plain wide sweep with itself."""

    def no_halving(truth, rounds, budget, sigma, rng):
        return max(range(len(truth)), key=lambda i: truth[i]), 0

    monkeypatch.setattr(bt, "run_sh", no_halving)
    assert bt.main(["40"]) == 1
