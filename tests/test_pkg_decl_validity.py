"""Regression tests for the audit-corps fix wave: a malformed ``part k of n``
declaration must read the same way at every site that judges it.

BUG-SEED-34: the scaffold exemption (``_declared_construction``) accepted any
opening ``part k of n`` with k < n — including ``part 0 of n`` and over-cap
``part 1 of 99`` — while the pricing side (``_open_package_tail``) rejected the
very same strings as malformed and priced the delivery normally (no HOLD, so a
gain-less delivery was rolled back at price). One string was "under
construction, buy no eval" in one place and "malformed, price as delivered" in
the other. The validity predicate (``1 <= k < n <= _PKG_MAX_STEPS``) now lives
in the shared parser ``_pkg_decl``, so a malformed declaration is no plea
anywhere: the delivery is evaluated and priced normally.
"""

from __future__ import annotations

from rsi_kernel_tune.inner.trajectory import IterationRecord as IR
from rsi_kernel_tune.loop.driver import (_PKG_MAX_STEPS, _declared_construction,
                                          _open_package_tail, _pkg_decl)


def _row(change: str, verdict: str = "KEEP") -> IR:
    return IR(iter=0, decision_id="D0.1", change=change,
              correctness=None if verdict == "NEUTRAL" else True,
              verdict=verdict, sha="s0" if verdict != "REVERT" else "")


def test_a_malformed_part_zero_declaration_is_no_package_anywhere():
    """The contradiction BUG-SEED-34 pinned, verbatim: ``part 0 of 3`` opened a
    free-eval scaffold row while the pricing side called it malformed and rolled
    the same delivery back at price. Both readers now reject it."""
    change = "part 0 of 3: wire the loads first"
    assert _pkg_decl(change) is None
    assert _declared_construction(change) is None        # no scaffold, a real eval
    assert _open_package_tail([_row(change)]) is None    # no HOLD, priced normally


def test_an_over_cap_declaration_no_longer_buys_the_scaffold_exemption():
    """The cap used to be a pricing-side-only rule: ``part 1 of 9`` scaffolded
    the row for free while every pricing path refused the plea. The shared
    predicate rejects it before either reader sees it."""
    change = f"grand rewrite, part 1 of {_PKG_MAX_STEPS + 5}"
    assert _pkg_decl(change) is None
    assert _declared_construction(change) is None
    assert _open_package_tail([_row(change)]) is None


def test_a_valid_declaration_still_reads_the_same_everywhere():
    change = "part 2 of 3 attempt"
    assert _pkg_decl(change) is not None
    assert _declared_construction(change) == "part 2 of 3"
    assert _open_package_tail([_row(change)]) == (2, 3)


def test_finished_and_inverted_declarations_stay_rejected_everywhere():
    for change in ("part 3 of 3 — done", "part 4 of 3 — inverted"):
        assert _pkg_decl(change) is None
        assert _declared_construction(change) is None
        assert _open_package_tail([_row(change)]) is None


def test_a_malformed_latest_keep_does_not_count_toward_a_later_package_budget():
    """_open_package_tail counts declared KEEPs as the plea's budget: rows that
    never opened a package (malformed) must not spend one either."""
    segment = [_row("part 0 of 3: wire the loads first"),
               _row("part 1 of 3: the atoms")]
    assert _open_package_tail(segment) == (1, 3)
