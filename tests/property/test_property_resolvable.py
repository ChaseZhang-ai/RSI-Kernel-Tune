# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests (hypothesis) for the statistical core of loop/parse.py.

Hunt 2 (task 178): the statistics layer's failure mode is "every input the unit
tests happened to try was well-formed". These properties generate adversarial
record distributions -- NaN/Inf readings, single-element histories, constant
histories, 1e-6..1e6 scale extremes, junk injected through the C8 ``raw`` seam --
and pin the invariants the WP15 contracts state:

* P1 ``resolvable_diff`` output is always clamped to [floor, cap] (the
  implementation's own constants, and caller-supplied ordered pairs);
* P2 adversarial readings never crash and never leave the clamp (graceful
  degradation, not an exception and not a NaN verdict);
* P3 fewer than RESOLVABLE_MIN_SAMPLES ratio samples returns exactly the cap --
  the historical fallback value, never a mode (C8);
* P4 constant readings converge the band to the floor -- a constant history
  carries no noise and the estimate must not diverge;
* P5 detect_plateau's margin IS the resolvable_diff band under the same
  floor/cap/raw forwarding (contract C8's G8.2 identity), and an empty
  confirmation memo is the no-injection state;
* P6 a recent window that improves by more than the noise ceiling is never a
  plateau.

Two record shapes exercise both admission paths of ``_window_pairs``: the
production ``IterationRecord`` (per-window readings ride the ``workloads``
blocks) and the duck-typed EvalResult-like row (top-level raw arrays, what a
parsed eval JSON carries). resolvable_diff reads both through getattr.

max_examples <= 200, deadline=None (the estimators allocate freely; wall-clock
budgets only cause flakes).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from rsi_kernel_tune.inner.trajectory import (
    GATE_WINDOW_DEVICE,
    GATE_WINDOW_E2E,
    IterationRecord,
    gate_ms,
)
from rsi_kernel_tune.loop.parse import (
    RESOLVABLE_DIFF_CAP,
    RESOLVABLE_DIFF_FLOOR,
    RESOLVABLE_MIN_SAMPLES,
    _drift_clean,
    detect_plateau,
    resolvable_diff,
)

GATE_WINDOWS = (GATE_WINDOW_DEVICE, GATE_WINDOW_E2E)

# --- adversarial value pools -------------------------------------------------
#
# Everything a malformed bench JSON can put in a numeric field: None, bools (a
# bool IS an int in Python -- the parsers must reject it), NaN/Inf literals
# (json.loads accepts them), zero/negative readings, and finite values across
# six orders of magnitude.
ANY_NUMBER = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=-1e-6, allow_nan=False, allow_infinity=False),
)

RAW_LISTS = st.one_of(st.none(), st.lists(ANY_NUMBER, max_size=8))

WORKLOAD_BLOCKS = st.lists(
    st.fixed_dictionaries(
        {
            "name": st.sampled_from(["w0", "w1", "w2"]),
            "device_readings_ms": RAW_LISTS,
            "ref_device_readings_ms": RAW_LISTS,
        }
    ),
    max_size=2,
)


@st.composite
def _iteration_row(draw):
    """A production IterationRecord with adversarial numeric fields; the
    per-window raw readings ride the workload blocks (the trajectory's shape)."""
    return IterationRecord(
        iter=draw(st.integers(min_value=0, max_value=100)),
        decision_id="D",
        change="c",
        correctness=True,
        verdict="KEEP",
        bwd_median_ms=draw(ANY_NUMBER),
        device_median_ms=draw(ANY_NUMBER),
        vs_ref=draw(ANY_NUMBER),
        vs_ref_device=draw(ANY_NUMBER),
        workloads=draw(WORKLOAD_BLOCKS),
        clock_drift=draw(st.booleans()),
        reading_of=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=40))),
    )


@st.composite
def _evalresult_row(draw):
    """An EvalResult-like row (SimpleNamespace, the estimator is duck-typed):
    the mirrored per-window arrays sit at the top level, as parse_eval emits
    them -- including non-list junk types a malformed block can produce."""
    return SimpleNamespace(
        device_median_ms=draw(ANY_NUMBER),
        bwd_median_ms=draw(ANY_NUMBER),
        vs_ref=draw(ANY_NUMBER),
        vs_ref_device=draw(ANY_NUMBER),
        device_readings_ms=draw(RAW_LISTS),
        ref_device_readings_ms=draw(RAW_LISTS),
        workloads=draw(WORKLOAD_BLOCKS),
        clock_drift=draw(st.booleans()),
        reading_of=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=40))),
    )


ROW = st.one_of(_iteration_row(), _evalresult_row())


@st.composite
def record_history(draw, max_size: int = 8):
    """A history of rows with adversarial numeric fields."""
    n = draw(st.integers(min_value=1, max_value=max_size))
    return [draw(ROW) for _ in range(n)]


RAW_INJECTION = st.one_of(
    st.none(),
    st.just({}),
    st.fixed_dictionaries({"ratios": st.lists(ANY_NUMBER, max_size=8)}),
    st.fixed_dictionaries({"pairs": st.lists(ANY_NUMBER, max_size=8)}),
    st.fixed_dictionaries(
        {"ratios": st.lists(ANY_NUMBER, max_size=8), "pairs": st.lists(ANY_NUMBER, max_size=8)}
    ),
)

# caller-supplied clamp bounds, kept ordered (floor <= cap -- the documented
# config contract for [loop] resolvable_floor / resolvable_cap)
ORDERED_BOUNDS = st.tuples(
    st.floats(min_value=0.0005, max_value=0.01),
    st.floats(min_value=0.01, max_value=0.05),
)


def _usable_ratio_count(records, gate_window) -> int:
    """How many rows would feed resolvable_diff's between-job pool: a row that
    gate_ms can read, that carries no reused reading, and whose gate-window ratio
    is a finite real number (the estimator's own admission rule)."""
    n = 0
    for r in records:
        if gate_ms(r, gate_window) is None or getattr(r, "reading_of", None) is not None:
            continue
        v = getattr(r, "vs_ref_device" if gate_window == GATE_WINDOW_DEVICE else "vs_ref", None)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        n += 1
    return n


@settings(deadline=None, max_examples=200)
@given(history=record_history(), gate_window=st.sampled_from(GATE_WINDOWS), raw=RAW_INJECTION,
       bounds=ORDERED_BOUNDS)
def test_property_resolvable_diff_output_stays_in_floor_cap(history, gate_window, raw, bounds):
    """P1: the estimate never leaves [floor, cap] -- the implementation's own
    constants by default, the caller's ordered pair when config is supplied."""
    rd = resolvable_diff(history, gate_window, raw=raw)
    assert isinstance(rd, float)
    assert math.isfinite(rd)
    assert RESOLVABLE_DIFF_FLOOR <= rd <= RESOLVABLE_DIFF_CAP

    floor, cap = bounds
    rd_cfg = resolvable_diff(history, gate_window, floor=floor, cap=cap, raw=raw)
    assert math.isfinite(rd_cfg)
    assert floor <= rd_cfg <= cap


@settings(deadline=None, max_examples=200)
@given(history=record_history(max_size=10), gate_window=st.sampled_from(GATE_WINDOWS),
       raw=RAW_INJECTION, window=st.integers(min_value=1, max_value=6))
def test_property_adversarial_readings_degrade_gracefully(history, gate_window, raw, window):
    """P2: NaN/Inf/single-element/all-constant/1e-6..1e6-scale histories never
    raise, never return a non-finite verdict, and never leave the clamp -- the
    data-poverty fallback (cap) is the graceful answer, not a crash."""
    rd = resolvable_diff(history, gate_window, raw=raw)
    assert isinstance(rd, float)
    assert math.isfinite(rd)
    assert RESOLVABLE_DIFF_FLOOR <= rd <= RESOLVABLE_DIFF_CAP

    plateau = detect_plateau(history, window=window, gate_window=gate_window, raw=raw)
    assert isinstance(plateau, bool)


UNUSABLE_RATIO = st.one_of(
    st.none(),
    st.booleans(),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.integers(max_value=0),
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
)


@settings(deadline=None, max_examples=200)
@given(ratio_rows=st.integers(min_value=0, max_value=RESOLVABLE_MIN_SAMPLES - 1),
       filler_rows=st.integers(min_value=0, max_value=5),
       ratio=st.floats(min_value=1e-6, max_value=1e6),
       gate_window=st.sampled_from(GATE_WINDOWS),
       junk=st.lists(UNUSABLE_RATIO, max_size=4))
def test_property_data_poor_history_returns_the_cap_exactly(
        ratio_rows, filler_rows, ratio, gate_window, junk):
    """P3: fewer than RESOLVABLE_MIN_SAMPLES usable ratio samples returns the
    cap, exactly -- the historical fallback value, never an estimated mode (C8),
    whatever the junk the rows otherwise carry. An empty ``raw`` injection is
    bit-identical to no injection (the memo-empty state)."""
    recs = [
        IterationRecord(iter=i, decision_id="D", change="c", correctness=True,
                        verdict="KEEP", device_median_ms=1.0, bwd_median_ms=1.0,
                        vs_ref=ratio, vs_ref_device=ratio)
        for i in range(ratio_rows)
    ]
    recs += [
        IterationRecord(iter=100 + i, decision_id="D", change="c", correctness=True,
                        verdict="KEEP", device_median_ms=1.0, bwd_median_ms=1.0,
                        vs_ref=None, vs_ref_device=None)
        for i in range(filler_rows)
    ]
    assert _usable_ratio_count(recs, gate_window) < RESOLVABLE_MIN_SAMPLES

    assert resolvable_diff(recs, gate_window) == RESOLVABLE_DIFF_CAP
    # unusable injection values (bools/None/NaN/negative/zero) add no samples
    assert resolvable_diff(recs, gate_window, raw={"ratios": junk}) == RESOLVABLE_DIFF_CAP
    # the memo-empty state is the no-injection state
    assert resolvable_diff(recs, gate_window, raw={}) == resolvable_diff(recs, gate_window)


@settings(deadline=None, max_examples=200)
@given(n=st.integers(min_value=RESOLVABLE_MIN_SAMPLES, max_value=8),
       ms0=st.floats(min_value=1e-6, max_value=1e6),
       ratio0=st.floats(min_value=1e-6, max_value=1e6),
       gate_window=st.sampled_from(GATE_WINDOWS))
def test_property_constant_readings_converge_to_the_floor(n, ms0, ratio0, gate_window):
    """P4: a constant history (one reading level, one ratio, constant paired
    window readings at any scale) carries zero noise -- the band must converge
    to the floor, never diverge."""
    recs = [
        IterationRecord(iter=i, decision_id="D", change="c", correctness=True,
                        verdict="KEEP", device_median_ms=ms0, bwd_median_ms=ms0,
                        vs_ref=ratio0, vs_ref_device=ratio0,
                        workloads=[{"name": "w0",
                                    "device_readings_ms": [ms0, ms0 * 2, ms0],
                                    "ref_device_readings_ms": [ms0 * ratio0, ms0 * ratio0 * 2,
                                                               ms0 * ratio0]}])
        for i in range(n)
    ]
    assert resolvable_diff(recs, gate_window) == RESOLVABLE_DIFF_FLOOR
    # and a constant history that sets no new best is a plateau under that band
    assert detect_plateau(recs, window=3) is True


def _manual_plateau(records, window, noise_floor, gate_window, floor, cap, raw):
    """detect_plateau's documented arithmetic, with the margin read from
    resolvable_diff directly (contract C8's G8.2: one estimator, both consumers)."""
    evaluated = [r for r in records if gate_ms(r, gate_window) is not None]
    if len(evaluated) <= window:
        return False
    margin = min(
        noise_floor,
        resolvable_diff(
            records,
            gate_window,
            floor=RESOLVABLE_DIFF_FLOOR if floor is None else floor,
            cap=noise_floor if cap is None else cap,
            raw=raw,
        ),
    )
    evaluated = _drift_clean(evaluated, window + 1)
    prior_best = min(gate_ms(r, gate_window) for r in evaluated[:-window])
    recent_best = min(gate_ms(r, gate_window) for r in evaluated[-window:])
    return recent_best >= prior_best * (1.0 - margin)


@settings(deadline=None, max_examples=200)
@given(history=record_history(max_size=9), gate_window=st.sampled_from(GATE_WINDOWS),
       window=st.integers(min_value=1, max_value=5),
       noise_floor=st.floats(min_value=0.001, max_value=0.02),
       floor=st.one_of(st.none(), st.floats(min_value=0.0005, max_value=0.01)),
       cap=st.one_of(st.none(), st.floats(min_value=0.005, max_value=0.05)),
       raw=RAW_INJECTION)
def test_property_detect_plateau_margin_is_the_resolvable_band(
        history, gate_window, window, noise_floor, floor, cap, raw):
    """P5 (G8.2 as a property): detect_plateau judges the window with EXACTLY the
    resolvable_diff band -- same estimator, same floor/cap defaults, same raw
    forwarding. If the plateau margin and the watermark confirmation band ever
    drift apart, this goes red. The empty-memo and no-memo states are the same
    band."""
    got = detect_plateau(history, window=window, noise_floor=noise_floor,
                         gate_window=gate_window, floor=floor, cap=cap, raw=raw)
    expected = _manual_plateau(history, window, noise_floor, gate_window, floor, cap, raw)
    assert got == expected

    # the memo-empty injection state is the no-injection state (G8.2's two
    # states must move together -- _confirm_raw_memo returns {} when empty)
    assert (resolvable_diff(history, gate_window, raw={})
            == resolvable_diff(history, gate_window, raw=None))


@settings(deadline=None, max_examples=200)
@given(base=st.floats(min_value=0.1, max_value=10.0),
       noise_floor=st.floats(min_value=0.002, max_value=0.015),
       window=st.integers(min_value=1, max_value=5),
       prior_spread=st.floats(min_value=0.0, max_value=0.5),
       ratio0=st.floats(min_value=0.5, max_value=1.5))
def test_property_a_supra_noise_improving_window_is_never_a_plateau(
        base, noise_floor, window, prior_spread, ratio0):
    """P6: when the recent window's best improves by more than the noise ceiling,
    the verdict is NOT a plateau -- no matter what the ratio history looks like.
    The margin is min(noise_floor, resolvable_diff) <= noise_floor, so an
    improvement factor safely below 1 - noise_floor cannot read as plateau."""
    # prior segment: w+1 evaluated rows at/above base (the running best >= base)
    prior = [base * (1.0 + prior_spread * (i / max(window, 1)))
             for i in range(window + 1)]
    # recent segment: every row below base * 0.98, so the recent best improves
    # by > 2% — beyond any noise_floor this strategy draws (<= 1.5%)
    recent = [base * 0.97 for _ in range(window)]
    ms = prior + recent
    recs = [
        IterationRecord(iter=i, decision_id="D", change="c", correctness=True,
                        verdict="KEEP", device_median_ms=m, bwd_median_ms=m,
                        vs_ref=ratio0, vs_ref_device=ratio0)
        for i, m in enumerate(ms)
    ]
    assert detect_plateau(recs, window=window, noise_floor=noise_floor) is False
