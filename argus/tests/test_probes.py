import numpy as np
import pytest

from argus import probes as pb
from argus import verifier as vf
from argus.verifier import Tolerance

TOL = Tolerance()


def _ref(inp):
    x = np.asarray(inp["x"], dtype=np.float64)
    return {"out": (x / np.sqrt((x ** 2).mean(-1, keepdims=True))).astype(np.float32)}


X = {"x": np.random.default_rng(2).normal(size=(6, 8)).astype(np.float32)}


def test_registry_by_op_class():
    assert any(p.name == "scale_up" for p in pb.probes_for("rmsnorm"))
    assert any(p.name == "softmax_shift" for p in pb.probes_for("softmax"))
    assert pb.probes_for("unknown")            # falls back to the generic set


def test_infer_op_class():
    assert pb.infer_op_class("001_fused_add_rmsnorm_h2048") == "rmsnorm"
    assert pb.infer_op_class("gemm_n4096_k4096") == "gemm"
    assert pb.infer_op_class("whatever") == "generic"


def test_scale_probe_passes_for_honest():
    honest = lambda i: _ref(i)                                  # noqa: E731
    oc = pb.SCALE_UP.check(honest, _ref, X, TOL)
    assert oc.passed


def test_scale_probe_flags_scaled_output():
    # a kernel that forgets to normalise: output scales with input -> breaks invariance
    broken = lambda i: {"out": np.asarray(i["x"], dtype=np.float32) * 2.0}   # noqa: E731
    oc = pb.SCALE_UP.check(broken, _ref, X, TOL)
    assert not oc.passed


def test_row_permute_equivariance():
    honest = lambda i: _ref(i)                                  # noqa: E731
    assert pb.ROW_PERMUTE.check(honest, _ref, X, TOL).passed


def test_softmax_shift_invariance():
    def sm(inp):
        x = np.asarray(inp["x"], dtype=np.float64)
        e = np.exp(x - x.max(-1, keepdims=True))
        return {"out": (e / e.sum(-1, keepdims=True)).astype(np.float32)}

    assert pb.SOFTMAX_SHIFT.check(sm, sm, X, TOL).passed
