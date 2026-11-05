# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit-corps R3F fix wave over loop/analytic_floor.py.

Covered here:
- BUG-CH-F-02: JSON-legal but wrong-typed axes values crashed the floor
  formulas -- a string axis (``"axes": {"m": "4096"}``, an operator hand-write)
  raised TypeError out of ``_gemm_flops``, and the ``Infinity``/``NaN``
  literals ``json.loads`` accepts natively raised OverflowError (int(inf)) in
  ``_gemm_flops``, ``_axis``, ``_conv_flops`` and ``resolve_dim``. Both escape
  the driver's ``(OSError, ValueError)`` catch (the CORPS-C1 lesson: that
  guard is not enough), so one malformed axes entry used to end the whole
  run -- where the contract is a degradation to "no analytic floor" (an
  absent floor is recoverable, a crash poisons every gate that cites one).

The CORPS-C1 fix (a non-dict workload/axes degrades to None in
``problem_from_card``) does not cover these: here the axes IS a dict, the
individual VALUES are the wrong type, and ``floor_for`` is also called
directly with hand-built problems. Guards go at the formula sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from peregrine_kernel.loop import analytic_floor as af

_GEMM = {"arch": "sm80", "family": "gemm", "op_type": "gemm", "name": "test_gemm",
         "inputs": [{"name": "a", "shape": ["m", "k"], "dtype": "fp16"}],
         "outputs": [{"name": "c", "shape": ["m", "n"], "dtype": "fp16"}]}


def _floor(axes, **over):
    return af.floor_for({**_GEMM, "axes": axes, **over})


def test_string_axis_degrades_not_crashes():
    """The strongest form of the operator hand-write: every m/n/k axis a string.
    The gemm flops charge is dropped (bytes roofline survives), no raise."""
    floor = _floor({"m": "4096", "n": 8, "k": 8},
                   inputs=[{"name": "a", "shape": [64, 32], "dtype": "fp16"}],
                   outputs=[{"name": "c", "shape": [64, 8], "dtype": "fp16"}])
    assert floor["flops"] is None
    assert any("m/n/k" in a for a in floor["assumptions"])
    assert floor["floor_ms"] is not None          # the bytes side still prices


def test_one_string_among_numbers_degrades_not_crashes():
    """min("4096", 8, 8) used to TypeError even with two clean axes."""
    floor = _floor({"m": "4096", "n": 8, "k": 8})
    assert floor["flops"] is None


def test_infinity_axis_degrades_not_crashes():
    """json.loads parses Infinity natively; int(inf) used to OverflowError out
    of _gemm_flops."""
    floor = _floor({"m": float("inf"), "n": 8, "k": 8})
    assert floor["flops"] is None


def test_nan_axis_degrades_not_crashes():
    """int(nan) raised ValueError (caught downstream, but the formula itself
    must simply call the axis unresolvable)."""
    floor = _floor({"m": float("nan"), "n": 8, "k": 8})
    assert floor["flops"] is None


def test_numeric_inf_shape_entry_degrades_not_crashes():
    """A shape entry that json.loads turned into inf: resolve_dim used to
    OverflowError."""
    floor = _floor({"m": 8, "n": 8, "k": 8},
                   inputs=[{"name": "a", "shape": [float("inf"), 8], "dtype": "fp16"}],
                   outputs=[])
    assert floor["bytes_min"] == 0
    assert any("not resolvable" in a for a in floor["assumptions"])


def test_expression_over_an_inf_axis_degrades_not_crashes():
    """A shape expression over an inf axis value: v == int(v) used to
    OverflowError."""
    floor = _floor({"m": 8, "n": 8, "k": float("inf")},
                   inputs=[{"name": "a", "shape": ["m*k", 8], "dtype": "fp16"}],
                   outputs=[])
    assert floor["bytes_min"] == 0


def test_inf_axis_in_attention_and_conv_degrades_not_crashes():
    """_axis / _conv_flops share the int(v) conversion -- both must treat a
    non-finite axis as unresolvable."""
    att = af.floor_for({"arch": "sm80", "family": "attention",
                        "axes": {"batch": float("inf"), "heads": 2, "seq": 8, "dim": 8},
                        "inputs": [], "outputs": []})
    assert att["flops"] is None
    conv = af.floor_for({"arch": "sm80", "family": "conv",
                         "axes": {"n": 2, "h_out": 8, "w_out": 8, "c_out": 4, "c_in": 4,
                                  "k_h": 1, "k_w": 1, "k_d": float("inf")},
                         "inputs": [], "outputs": []})
    # an unreadable depth reads as absent (the default 1): the floor stays
    # provable and under-counts, and nothing raises
    assert conv["flops"] == 2 * 2 * 8 * 8 * 4 * 4 * 1


def test_clean_gemm_still_prices():
    """Parity: sane axes price the full 2mnk flops charge as before."""
    floor = _floor({"m": 64, "n": 64, "k": 32})
    assert floor["flops"] == 2 * 64 * 64 * 32
    assert floor["floor_ms"] is not None


def test_problem_from_card_junk_value_axes_still_degrades_via_floor_for():
    """A dict workload whose axes VALUES are junk reaches floor_for from
    problem_from_card too: the whole path stays on the None/assumption
    channel (CORPS-C1 covered only the non-dict shapes)."""
    card = {"arch": "sm80",
            "definition": {"name": "gemm_op", "op_type": "gemm",
                           "inputs": {"a": {"shape": [64, 32], "dtype": "fp16"}},
                           "outputs": {"c": {"shape": [64, 8], "dtype": "fp16"}}},
            "workload": {"uuid": "w0", "axes": {"m": "4096", "n": 8, "k": 8}}}
    problem = af.problem_from_card(card)
    assert problem is not None                     # a dict axes passes C1's guard
    floor = af.floor_for(problem)
    assert floor["flops"] is None and floor["floor_ms"] is not None
