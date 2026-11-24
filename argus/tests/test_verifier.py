import numpy as np
import pytest

from argus import probes as pb
from argus import verifier as vf
from argus.verifier import PrecisionContract, Tolerance, VerifyConfig


def _rmsnorm_ref(x):
    x64 = np.asarray(x, dtype=np.float64)
    return (x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-6)).astype(np.float32)


def _rmsnorm_ref64(x):
    x64 = np.asarray(x, dtype=np.float64)
    return x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-6)


def _tf32(x):
    a = np.asarray(x, dtype=np.float32)
    return (a.view(np.int32) & np.int32(~((1 << 13) - 1))).view(np.float32)


X = {"x": np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)}
Y = {"x": np.random.default_rng(1).normal(size=(8, 64)).astype(np.float32)}


def _ref(inp):
    return {"out": _rmsnorm_ref(inp["x"])}


def _ref64(inp):
    return {"out": _rmsnorm_ref64(inp["x"])}


def _cfg(**kw):
    return VerifyConfig(tolerance=Tolerance(), contract=PrecisionContract(), probes=[
        pb.SCALE_UP, pb.SCALE_DOWN, pb.ROW_PERMUTE], **kw)


def _verify(cand, workloads=(X,), source=""):
    return vf.verify_kernel(candidate=cand, reference=_ref, reference_fp64=_ref64,
                            workloads=list(workloads), config=_cfg(), source=source)


def test_honest_kernel_accepted():
    v = _verify(lambda i: {"out": _rmsnorm_ref(i["x"])})
    assert v.accepted, v.render()


def test_identity_rejected():
    v = _verify(lambda i: {"out": i["x"]})
    assert not v.accepted


def test_memoised_rejected():
    state = {}

    def memo(i):
        state.setdefault("out", _rmsnorm_ref(i["x"]))
        return {"out": state["out"]}

    v = _verify(memo)
    assert not v.accepted
    assert "memoized" in v.flags or any("memo" in o.name for o in v.hard_failures)


def test_shape_specialised_rejected_by_multi_workload():
    def tuned(i):
        if np.asarray(i["x"]).shape[1] != 16:
            return {"out": np.zeros_like(i["x"])}
        return {"out": _rmsnorm_ref(i["x"])}

    assert _verify(tuned, workloads=(X,)).accepted          # passes the single tuned shape
    assert not _verify(tuned, workloads=(X, Y)).accepted    # caught by multi-workload


def test_precision_downgrade_rejected():
    def tf32_kernel(i):
        return {"out": _rmsnorm_ref(_tf32(i["x"]))}

    v = _verify(tf32_kernel)
    assert not v.accepted
    assert "precision_downgrade" in v.flags


def test_board_tolerance_rule_vs_allclose():
    # an error just above a tight allclose(1e-3) but inside the board rule passes the board
    # element rule and fails only the loose 1e-3 rule -- the two rules differ by construction
    tol = Tolerance(atol=1e-2, rtol=1e-2, matched_ratio=0.99)
    ref = np.ones((4,), dtype=np.float32)
    out = ref + 5e-3
    ratio, mae, _ = vf.elementwise_pass_ratio(out, ref, tol)
    assert ratio == 1.0
    assert not np.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_static_scan_flags():
    outcomes = vf.scan_static("import ref\nout = x.half()\nlru_cache\n"
                              "cudaDeviceSetLimit()\nreturn x\n")
    names = {o.name for o in outcomes if not o.passed}
    assert any("delegate_reference" in n for n in names)
    assert any("clock_tamper" in n for n in names)
    assert any("memoize" in n for n in names)


def test_probe_scale_invariance_holds_for_honest():
    v = _verify(lambda i: {"out": _rmsnorm_ref(i["x"])})
    scale = [o for o in v.outcomes if o.name == "probe:scale_up"]
    assert scale and scale[0].passed
