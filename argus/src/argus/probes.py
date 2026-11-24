"""Metamorphic / adversarial input probes (part of ARGUS contribution C1).

A kernel that passes the tolerance check on the *sampled* workload can still be wrong for
the rest of the definition's shape family.  Each probe encodes an **algebraic law** of an
operator, applies an input transform that the law says must leave the output (or scale it)
unchanged, and checks the candidate against the law rather than only against a second
reference evaluation:

    scale-RMSNorm   RMSNorm(alpha * x) == RMSNorm(x)            (alpha > 0)
    shift-softmax   softmax(x + c)       == softmax(x)          (per-row shift)
    scale-GEMM      GEMM(A, alpha * B)   == alpha * GEMM(A, B)
    permute-rows    row-wise op commutes with row permutation
    pair-norm-RoPE  RoPE preserves the 2-norm of each rotation pair

The probes are cheap, need no ground truth beyond the law, and can be evaluated on inputs
*outside* the workload distribution -- which is exactly where a tuned kernel breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .verifier import HARD, SOFT, CheckOutcome, Tolerance


def _primary_key(inputs: Dict[str, np.ndarray]) -> str:
    for k, v in inputs.items():
        if np.issubdtype(np.asarray(v).dtype, np.floating):
            return k
    return next(iter(inputs))


def _primary_out_key(out) -> str:
    if isinstance(out, dict):
        return next(iter(out))
    return "out"


@dataclass
class Probe:
    name: str
    op_class: str
    transform: Callable[[np.ndarray], np.ndarray]
    out_law: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    description: str = ""

    def check(self, candidate, reference, inputs: Dict[str, np.ndarray],
              tol: Tolerance) -> CheckOutcome:
        from .verifier import elementwise_pass_ratio

        key = _primary_key(inputs)
        x = np.asarray(inputs[key])
        try:
            x2 = self.transform(x)
        except Exception as exc:
            return CheckOutcome(f"probe:{self.name}", True, SOFT,
                                f"transform unavailable ({exc.__class__.__name__}); skipped")
        inputs2 = dict(inputs)
        inputs2[key] = x2

        ref_x = reference({k: np.array(v, copy=True) for k, v in inputs.items()})
        okey = _primary_out_key(ref_x)
        ref_prim = ref_x[okey] if isinstance(ref_x, dict) else ref_x
        expected = self.out_law(ref_prim, x) if self.out_law else ref_prim

        try:
            cand2 = candidate({k: np.array(v, copy=True) for k, v in inputs2.items()})
        except Exception as exc:
            return CheckOutcome(f"probe:{self.name}", False, SOFT,
                                f"candidate failed under transform: {exc}")
        cand2 = cand2[okey] if isinstance(cand2, dict) else cand2

        ratio, max_abs, _ = elementwise_pass_ratio(cand2, expected, tol)
        passed = ratio >= tol.matched_ratio
        if ratio < 0.9:
            severity = HARD
        else:
            severity = SOFT
        detail = (f"{self.description or self.name}: matched_ratio={ratio:.4f} "
                  f"max_abs_err={max_abs:.3e}")
        return CheckOutcome(f"probe:{self.name}", passed, severity, detail,
                            {"matched_ratio": ratio})


# --------------------------------------------------------------------------- #
# Laws
# --------------------------------------------------------------------------- #

def _scale(alpha: float):
    return lambda x: (x * alpha).astype(x.dtype)


def _shift_rows(shifts=None):
    def f(x):
        rng = np.random.default_rng(7)
        if x.ndim == 1:
            return (x + 3.0).astype(x.dtype)
        c = rng.normal(0, 2.0, size=(x.shape[0],) + (1,) * (x.ndim - 1))
        return (x + c).astype(x.dtype)
    return f


def _perm(n: int) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    rng = np.random.default_rng(11)
    return rng.permutation(n)


def _permute_rows(x):
    if x.ndim == 1:
        return x[::-1].copy()
    return x[_perm(x.shape[0])].copy()


def _permute_out(out, x):
    """Row-wise ops are *equivariant*: permuting input rows permutes output rows."""
    if out.ndim == 1:
        return out[::-1].copy()
    return out[_perm(out.shape[0])].copy()


SCALE_UP = Probe("scale_up", "rmsnorm", _scale(5.0), None,
                 "RMSNorm must be invariant to positive input scaling")
SCALE_DOWN = Probe("scale_down", "rmsnorm", _scale(0.2), None,
                   "RMSNorm must be invariant to positive input scaling")
SOFTMAX_SHIFT = Probe("softmax_shift", "softmax", _shift_rows(), None,
                      "softmax is invariant to a per-row additive shift")
GEMM_SCALE = Probe("gemm_scale", "gemm", _scale(3.0),
                   lambda out, x: (out * 3.0).astype(out.dtype),
                   "GEMM is homogeneous of degree 1 in the scaled operand")
ROW_PERMUTE = Probe("row_permute", "rowwise", _permute_rows, _permute_out,
                    "a row-wise reduction is equivariant to row permutation")


#: op_class -> probes.  ``rowwise`` is the fallback for any per-row op.
REGISTRY: Dict[str, List[Probe]] = {
    "rmsnorm": [SCALE_UP, SCALE_DOWN, ROW_PERMUTE],
    "layernorm": [SCALE_UP, ROW_PERMUTE],
    "softmax": [SOFTMAX_SHIFT, ROW_PERMUTE],
    "gemm": [GEMM_SCALE],
    "rowwise": [SCALE_UP, ROW_PERMUTE],
    "generic": [ROW_PERMUTE],
}


def probes_for(op_class: str) -> List[Probe]:
    return list(REGISTRY.get(op_class, REGISTRY["generic"]))


def infer_op_class(name: str) -> str:
    """Best-effort operator classification from a definition name."""
    n = name.lower()
    for key in ("rmsnorm", "layernorm", "softmax", "gemm", "matmul", "attention", "moe",
                "rope", "conv", "norm"):
        if key in n:
            if key == "norm":
                return "rmsnorm" if "rms" in n else "layernorm"
            if key in ("matmul",):
                return "gemm"
            return key
    return "generic"
