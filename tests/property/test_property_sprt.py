# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests (hypothesis) for the SPRT early-stop verdicts.

Hunt 2 (task 178): ``sprt_stop_verdict`` is the loss-only early stop (WP11/WP15,
contract C14) -- the rule whose selling point is that the false-win direction is
"arithmetically impossible, not merely rare". These properties hold it to that
claim over adversarial inputs:

* P7 the profit direction never stops: any candidate ratio >= 1.0 (any spread,
  any scale of paired readings) reads NOT a clear loss -- the stop can never
  manufacture a KEEP by stopping early;
* P8 the band boundary behaves per the documented semantics: parity never
  stops, a stop implies the ratio sits below parity minus the band floor,
  the verdict is monotone in the ratio, and with a zero floor the boundary
  sits exactly at parity (strict inequality, both ulp sides checked);
* P9 fewer than SPRT_MIN_PAIRED usable pairs never stops -- junk pairs
  (bools/None/NaN/Inf/negative/zero) are not evidence;
* P10 the win-stop verdict (C14's coupled direction) only fires past parity
  plus margin plus band, never on junk pairs, never with a non-positive
  margin.

max_examples <= 200, deadline=None.
"""

from __future__ import annotations

import math
import statistics

from hypothesis import given, settings
from hypothesis import strategies as st

from rsi_kernel_tune.card_template.eval.timing import (
    SPRT_BAND_CAP,
    SPRT_BAND_FLOOR,
    SPRT_MIN_PAIRED,
    sprt_band,
    sprt_stop_verdict,
    sprt_win_stop_verdict,
)

# positive paired ratios at any scale the harness could emit
POSITIVE_RATIO = st.floats(min_value=1e-9, max_value=1e9, allow_nan=False,
                           allow_infinity=False)
PAIR_LIST = st.lists(POSITIVE_RATIO, min_size=SPRT_MIN_PAIRED, max_size=24)

# junk a malformed reading stream can contribute; none of it is usable evidence
# under the verdict's own pair filter (isinstance real number, not bool, > 0).
# +/-inf is deliberately NOT here: the verdict's filter is `float(r) > 0.0`, so
# a +inf pair counts as a pair BY DESIGN (reported as an observation in the
# hunt report); NaN fails `> 0.0` and is excluded as evidence.
UNUSABLE_PAIR = st.one_of(
    st.none(),
    st.booleans(),
    st.just(float("nan")),
    st.integers(max_value=0),
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
)


@settings(deadline=None, max_examples=200)
@given(ratio=st.one_of(st.floats(min_value=1.0, max_value=1e9),
                       st.just(float("inf"))),
       ratios=PAIR_LIST,
       floor=st.floats(min_value=0.0, max_value=0.01),
       cap=st.floats(min_value=0.01, max_value=0.05))
def test_property_profit_direction_never_stops(ratio, ratios, floor, cap):
    """P7 (C14): the loss-only stop can never fire on the profit direction.
    Any candidate ratio >= 1.0 is not a clear loss for ANY positive paired
    history and ANY ordered band config -- the stop therefore cannot shorten a
    winning measurement, which is what makes the manufactured-KEEP direction
    structurally impossible."""
    assert sprt_stop_verdict(ratio, ratios, floor=floor, cap=cap) is False


@settings(deadline=None, max_examples=200)
@given(ratios=PAIR_LIST,
       probe=st.floats(min_value=0.9, max_value=1.1),
       floor=st.floats(min_value=0.0, max_value=0.01),
       cap=st.floats(min_value=0.01, max_value=0.05))
def test_property_loss_band_boundary_matches_the_documented_semantics(
        ratios, probe, floor, cap):
    """P8: the verdict is the documented band rule -- band = clamp(3 x 1.4826 x
    MAD of the pairs, floor, cap), True iff ratio + band < 1 (strict, so the
    boundary itself is NOT a loss) -- plus the derived invariants: parity never
    stops, a stop implies the ratio sits below parity minus the floor, and the
    verdict is monotone in the ratio."""
    usable = [float(r) for r in ratios
              if isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0.0]
    assert len(usable) >= SPRT_MIN_PAIRED
    mad = statistics.median(abs(r - statistics.median(usable)) for r in usable)
    band = min(max(3.0 * 1.4826 * mad, floor), cap)

    # strict inequality on the band boundary
    boundary = 1.0 - band
    assert sprt_stop_verdict(boundary, ratios, floor=floor, cap=cap) is False
    assert sprt_stop_verdict(probe, ratios, floor=floor, cap=cap) \
        == (probe + band < 1.0)
    # parity never reads a loss, at any spread or config
    assert sprt_stop_verdict(1.0, ratios, floor=floor, cap=cap) is False
    # a stop implies the candidate sits below parity minus the floor (the
    # band is floored, so no configuration widens the keepable range past it)
    if sprt_stop_verdict(probe, ratios, floor=floor, cap=cap):
        assert probe < 1.0 - floor
    # monotone: a smaller ratio never loses a stop the bigger one earned
    slightly_above = math.nextafter(probe, math.inf)
    if sprt_stop_verdict(slightly_above, ratios, floor=floor, cap=cap):
        assert sprt_stop_verdict(probe, ratios, floor=floor, cap=cap) is True


@settings(deadline=None, max_examples=200)
@given(const=st.floats(min_value=1e-9, max_value=1e9))
def test_property_zero_floor_boundary_is_exactly_at_parity_ulp_sharp(const):
    """P8b: with a zero floor and constant pairs (MAD 0 -> band 0) the stop
    boundary sits exactly at parity: ratio + 0 < 1 is exact float arithmetic,
    so parity itself must not stop and one ulp below parity must."""
    const_pairs = [const] * SPRT_MIN_PAIRED
    assert sprt_stop_verdict(1.0, const_pairs, floor=0.0) is False
    assert sprt_stop_verdict(math.nextafter(1.0, -math.inf), const_pairs,
                             floor=0.0) is True


@settings(deadline=None, max_examples=200)
@given(usables=st.lists(POSITIVE_RATIO, max_size=SPRT_MIN_PAIRED - 1),
       junk=st.lists(UNUSABLE_PAIR, max_size=4),
       ratio=st.sampled_from([0.0, -3.0, float("nan"), float("inf"),
                              0.999, 1.0, 2.5]))
def test_property_insufficient_pairs_never_stop(usables, junk, ratio):
    """P9: fewer than SPRT_MIN_PAIRED usable pairs cannot speak -- the verdict
    is False whatever the candidate ratio carries (including non-finite and
    negative junk, which must not crash the rule)."""
    ratios = usables + junk
    usable_n = sum(1 for r in ratios
                   if isinstance(r, (int, float)) and not isinstance(r, bool)
                   and r > 0.0 and math.isfinite(r))
    assert usable_n < SPRT_MIN_PAIRED
    assert sprt_stop_verdict(ratio, ratios) is False


@settings(deadline=None, max_examples=200)
@given(const=st.floats(min_value=1e-9, max_value=1e9),
       margin=st.sampled_from([0.0, 0.03, 0.0625, -0.01]))
def test_property_win_stop_boundary_is_parity_plus_margin_ulp_sharp(const, margin):
    """P10 (C14's coupled direction): the win stop fires only when even the
    interval's BOTTOM sits above parity by more than the margin. With constant
    pairs the band is the floor exactly, so the comparison `ratio - floor >
    1 + margin` is exact float arithmetic; the verdict is False at parity and
    far below it, True far above it, monotone in the ratio, never fired on
    insufficient pairs and never with a non-positive margin."""
    const_pairs = [const] * (SPRT_MIN_PAIRED + 2)
    band = sprt_band(const_pairs)
    # the honest (unclamped) band of a constant history is the floor exactly
    assert band == SPRT_BAND_FLOOR

    parity = 1.0 + margin
    # a ratio AT parity can never be a clear win (the bottom sits below parity:
    # subtracting a positive band from x lands strictly below x)
    assert sprt_win_stop_verdict(parity, const_pairs, margin) is False
    if margin <= 0:
        assert sprt_win_stop_verdict(1e9, const_pairs, margin) is False
        return
    # far above the boundary is a clear win; one band-width above parity is not
    assert sprt_win_stop_verdict(parity + 10.0 * band, const_pairs, margin) is True
    assert sprt_win_stop_verdict(parity + band, const_pairs, margin) is False
    # monotone: one ulp below a winning ratio cannot stop where it did not
    winner = parity + 10.0 * band
    assert sprt_win_stop_verdict(math.nextafter(winner, -math.inf),
                                 const_pairs, margin) is True
    # a win stop never fires on insufficient pairs
    assert sprt_win_stop_verdict(parity + 1e9, const_pairs[:SPRT_MIN_PAIRED - 1],
                                 margin) is False
