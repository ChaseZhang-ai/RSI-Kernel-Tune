# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests (hypothesis) for the WP29 replay-probe judgment.

Hunt 2 (task 178): the parent-side verdict over the probe child's honest-compute
medians (``benchkit._replay_probe_block``) is a pure ratio judgment once the
child payload is fixed -- these properties pin it over adversarial medians:

* P11 junk medians (zero / negative / NaN / Inf / bools / None-typed junk on
  either side) never crash the judgment and never convict without a genuine
  comparison: ``suspected`` requires a row where BOTH medians are positive
  numbers and the unrounded probe/judged ratio exceeds the threshold;
* P12 the threshold boundary is strict and ulp-exact: probe = threshold x judged
  reads pass (the design line is `probe_median > RKT_REPLAY_PROBE_RATIO x
  judged_median`), one ulp above reads suspected, one ulp below reads pass;
* P13 an honest probe (any +/-20% jitter over any shape combination at any
  base scale) never reads suspected under the default threshold;
* P14 a single timed shape matches the single probe shape positionally (names
  are allowed to disagree), and the row carries the rounded medians.

The benchkit module is loaded fresh from the card-template source (the same
code the production bench child runs); ``_probe_child`` is replaced per
example with the generated payload, and the threshold env var is pinned to
the default so the boundary properties are exact.

Runs on CPU (torch required by the benchkit import); no GPU is needed.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"

RATIO_ENV = "RKT_REPLAY_PROBE_RATIO"
DEFAULT_THRESHOLD = 3.0


def _benchkit():
    spec = importlib.util.spec_from_file_location(
        "hunt2_property_replay_probe_benchkit", EVAL_SRC / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    return bk


BK = _benchkit()
_ORIGINAL_PROBE_CHILD = BK._probe_child

# junk a malformed bench block can put in a median field. +inf and -inf are
# numbers under the implementation's guard (`isinstance(..., (int, float)) and
# > 0` admits +inf); zero / negative / NaN are filtered by that guard.
ANY_MEDIAN = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.booleans(),
)


def _run_block(payload, blocks):
    """``_replay_probe_block`` with the generated child payload and the
    threshold env pinned to the default (saved/restored around the call)."""
    saved_env = os.environ.get(RATIO_ENV)
    os.environ.pop(RATIO_ENV, None)
    BK._probe_child = lambda problem: payload
    try:
        return BK._replay_probe_block({}, blocks)
    finally:
        BK._probe_child = _ORIGINAL_PROBE_CHILD
        if saved_env is None:
            os.environ.pop(RATIO_ENV, None)
        else:
            os.environ[RATIO_ENV] = saved_env


@settings(deadline=None, max_examples=200)
@given(judged=ANY_MEDIAN, probe=ANY_MEDIAN,
       names=st.tuples(st.sampled_from(["w0", "w1", "other"]),
                       st.sampled_from(["w0", "w1", "other"])))
def test_property_junk_medians_never_crash_or_convict(judged, probe, names):
    """P11: junk on either side degrades the verdict -- it never raises, the
    status stays in the contract's vocabulary, and `suspected` is only ever
    produced from a genuine comparison (both medians positive numbers, the
    unrounded ratio above the threshold). A probe that cannot speak never
    convicts."""
    block_name, probe_name = names
    blocks = [{"name": block_name, "device_median_ms": judged}]
    payload = {"shapes": [{"name": probe_name, "probe_median_ms": probe}]}
    out = _run_block(payload, blocks)

    assert out["status"] in ("pass", "suspected", "unavailable")
    assert isinstance(out["ratio_threshold"], float)
    rows = out["shapes"]
    assert len(rows) == 1
    row = rows[0]
    if out["status"] == "suspected":
        assert row["status"] == "suspected"
        # a genuine comparison: both sides positive numbers (bools are ints in
        # Python -- the implementation's isinstance guard admits them as 1/0),
        # and the UNROUNDED ratio, as this test computes it from the same
        # floats, exceeds the threshold
        assert isinstance(judged, (int, float)) and judged > 0
        assert isinstance(probe, (int, float)) and probe > 0
        assert probe / judged > out["ratio_threshold"]
    else:
        assert row.get("status") != "suspected"


@settings(deadline=None, max_examples=200)
@given(judged=st.sampled_from([0.5, 1.0, 2.0, 4.0, 1024.0]),
       name=st.sampled_from(["w0", "shape-a", "x"]))
def test_property_threshold_boundary_is_strict_and_ulp_exact(judged, name):
    """P12: `probe_median > ratio x judged_median` is STRICT and the float
    boundary is exact for power-of-two judged medians (probe = 3 x judged
    divides back to exactly 3.0; one ulp of probe moves the ratio exactly one
    ulp of 3.0): at the threshold -> pass, one ulp above -> suspected, one ulp
    below -> pass."""
    at = judged * DEFAULT_THRESHOLD
    above = math.nextafter(at, math.inf)
    below = math.nextafter(at, -math.inf)
    blocks = [{"name": name, "device_median_ms": judged}]

    out_at = _run_block({"shapes": [{"name": name, "probe_median_ms": at}]}, blocks)
    assert out_at["status"] == "pass"
    assert out_at["shapes"][0]["ratio"] == DEFAULT_THRESHOLD

    out_above = _run_block({"shapes": [{"name": name, "probe_median_ms": above}]}, blocks)
    assert out_above["status"] == "suspected"
    assert out_above["shapes"][0]["status"] == "suspected"

    out_below = _run_block({"shapes": [{"name": name, "probe_median_ms": below}]}, blocks)
    assert out_below["status"] == "pass"
    assert out_below["shapes"][0].get("status") is None


@settings(deadline=None, max_examples=200)
@given(n_shapes=st.integers(min_value=1, max_value=5),
       base=st.floats(min_value=1e-3, max_value=1e3),
       jitters=st.lists(st.floats(min_value=-0.2, max_value=0.2), min_size=5, max_size=5),
       skew=st.floats(min_value=0.9, max_value=1.1))
def test_property_an_honest_probe_is_never_suspected(n_shapes, base, jitters, skew):
    """P13: an honest compute probe -- each shape's probe median within +/-20%
    of its judged median, over any shape count, any base scale and any jitter
    combination -- never reads suspected under the default threshold (ratios
    are bounded by 1.2 << 3.0). Every shape is compared, so the verdict is
    `pass`, not `unavailable`."""
    blocks, shapes = [], []
    for i in range(n_shapes):
        judged = base * skew * (1.0 + 0.05 * i)
        probe = judged * (1.0 + jitters[i])
        blocks.append({"name": f"w{i}", "device_median_ms": judged})
        shapes.append({"name": f"w{i}", "probe_median_ms": probe})
    out = _run_block({"shapes": shapes}, blocks)

    assert out["status"] == "pass"
    for i, row in enumerate(out["shapes"]):
        assert row.get("status") is None
        assert row["judged_median_ms"] == round(blocks[i]["device_median_ms"], 4)
        assert row["probe_median_ms"] == round(shapes[i]["probe_median_ms"], 4)
        assert row["ratio"] <= 1.2


@settings(deadline=None, max_examples=200)
@given(judged=st.floats(min_value=1e-3, max_value=1e3),
       probe=st.floats(min_value=1e-3, max_value=1e4),
       block_name=st.sampled_from(["only", "w0", "shape-x"]),
       probe_name=st.sampled_from(["totally-different", "w9", "shape-y"]))
def test_property_single_sample_matches_positionally(judged, probe, block_name, probe_name):
    """P14: one timed shape and one probe shape match positionally even when
    the names disagree (the parent's one-shape-on-both-sides rule), and the
    verdict follows the design line on the same floats the test computes."""
    blocks = [{"name": block_name, "device_median_ms": judged}]
    payload = {"shapes": [{"name": probe_name, "probe_median_ms": probe}]}
    out = _run_block(payload, blocks)

    assert out["status"] == ("suspected" if probe / judged > DEFAULT_THRESHOLD
                             else "pass")
    row = out["shapes"][0]
    assert row["judged_median_ms"] == round(judged, 4)
    assert row["probe_median_ms"] == round(probe, 4)
    assert row["ratio"] == round(probe / judged, 4)
