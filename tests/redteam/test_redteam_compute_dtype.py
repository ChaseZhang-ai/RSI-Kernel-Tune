# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Red-team fixture (WP22): a kernel computing BELOW the declared precision is
detected and convicted.

The contract sentence promises that going below the declared compute_dtype is a
contract violation (REVERT). This pins the detectable scenario that promise
rests on, through the measure the bench actually runs (eval/precision.py's
PrecisionRecorder, formulas untouched): with the reference evaluated at the
declared dtype -- benchkit._refs_at_declared, the path _precision_measure takes
-- a candidate that silently computes one grade coarser than the declaration
lands past k*e_ref + floor (the declared output dtype's ulp) and the strict
gate turns it into a correctness failure naming the declaration. The same
candidate AT the declaration passes. Runs on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"

from rsi_kernel_tune.loop.precision_gate import describe, gate_failure  # noqa: E402

DECLARED = "fp16"
BELOW = "bf16"          # one grade coarser than the declaration: the violation


def _benchkit(env_dtype: str):
    old = os.environ.pop("RKT_COMPUTE_DTYPE", None)
    os.environ["RKT_COMPUTE_DTYPE"] = env_dtype
    try:
        spec = importlib.util.spec_from_file_location(
            f"redteam_wp22_{env_dtype}", EVAL_SRC / "benchkit.py")
        bk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bk)
        return bk
    finally:
        os.environ.pop("RKT_COMPUTE_DTYPE", None)
        if old is not None:
            os.environ["RKT_COMPUTE_DTYPE"] = old


def _precision_mod():
    spec = importlib.util.spec_from_file_location("redteam_wp22_precision",
                                                  EVAL_SRC / "precision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBLEM = {"definition": {"outputs": {"y": {"dtype": "float32"}}}, "dps": False}


def _ref(x, w):
    """The reference implementation: an honest fp32 GEMM."""
    return x.float() @ w.float()


def _kernel(dtype):
    """A candidate 'kernel' fixed to compute at ``dtype`` -- what a student that
    quantizes its operands/accumulation to ``dtype`` produces."""
    def k(x, w):
        return (x.to(dtype).matmul(w.to(dtype))).float()
    return k


@pytest.fixture
def scenario():
    torch.manual_seed(7)
    inp = [torch.randn(64, 64), torch.randn(64, 64)]
    truth = inp[0].double() @ inp[1].double()
    # the reference side as the bench computes it under the declaration (the
    # declaration-side evaluation _precision_measure performs):
    bk = _benchkit(DECLARED)
    P = _precision_mod()
    refs = bk._refs_at_declared(PROBLEM, _ref, inp, [], "cpu")
    return P, inp, truth, refs


def _measure(P, inp, truth, refs, kernel) -> dict:
    rec = P.PrecisionRecorder()
    rec.measure("case0", kernel(*inp), refs,
                lambda: ([truth], "float64", ""))
    return rec.block()


def test_going_below_the_declaration_is_detected_and_convicted(scenario):
    """The promised scenario, detectable: e_cand(below) > 3*e_ref(at declaration)
    + floor(fp16 ulp), and the gate's conviction names the declared contract."""
    P, inp, truth, refs = scenario
    e_ref = P.rel_l2(refs[0].double(), truth)
    below = _measure(P, inp, truth, refs, _kernel({"bf16": torch.bfloat16}[BELOW]))
    assert below["status"] == "fail", below["why"]
    w = below["worst"]
    assert w["e_cand"] > w["limit"]
    # e_ref here is the reference AT the declaration (not the native fp32 one):
    assert w["e_ref"] == pytest.approx(e_ref, rel=1e-9)
    summary = json.loads(json.dumps({
        "status": below["status"], "k": below["k"], "n_checks": below["n_checks"],
        "n_failed": below["n_failed"], "worst": below["worst"]}))
    why = gate_failure("strict", summary, compute_dtype=DECLARED)
    assert why is not None
    assert f"compute_dtype={DECLARED}" in why and "contract violation (REVERT)" in why


def test_working_at_the_declaration_passes(scenario):
    """The other half of the contract: a kernel that computes AT the declared
    dtype is not convicted (within the same k*e_ref + floor bound)."""
    P, inp, truth, refs = scenario
    at = _measure(P, inp, truth, refs, _kernel(torch.float16))
    assert at["status"] == "pass", at.get("why")


def test_the_floor_is_the_declared_output_dtype_ulp(scenario):
    """The bound the rows are judged against uses the DECLARED output dtype's
    unit roundoff (the row's dtype is the declaration after the reference-side
    evaluation), not fp32's."""
    P, inp, truth, refs = scenario
    block = _measure(P, inp, truth, refs, _kernel(torch.float16))
    w = block["worst"]
    assert w["dtype"] == "float16"
    # floor = 1 ulp of fp16 (2^-11 relative) -- visible in the limit arithmetic
    e_ref = w["e_ref"]
    expected = 3 * e_ref + max(P.FLOOR_ULPS * P.unit_roundoff(torch.float16), P.FLOOR_REL)
    assert w["limit"] == pytest.approx(expected, rel=1e-6)


def test_an_fp32_row_is_never_marked_with_the_declaration(scenario):
    """And the default stays the default: the same scenario run without a
    declaration reports fp32 rows and the historical gate text."""
    P, inp, truth, refs = scenario
    bk = _benchkit("fp32")
    native = bk.normalize_outputs(PROBLEM, _ref(*inp), "cpu")
    assert all(r.dtype == torch.float32 for r in native)
    block = _measure(P, inp, truth, native, _kernel(torch.float16))
    assert block["worst"]["dtype"] == "float32"
    assert "compute_dtype" not in describe({"contract": "strict", "status": "pass"})
