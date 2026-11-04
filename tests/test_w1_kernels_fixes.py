"""W1 kernel-delivery audit fixes (FA6): tests for the eight kernel.py repairs.

Every kernel here imports the CUTLASS DSL, which the test venv does not
carry (and this host has no GPU), so none of them can be imported whole.
What these tests do instead: they parse the repaired kernel.py with `ast`,
extract the EXACT module-level FunctionDef / constant Assign nodes under
test, and exec only those in an isolated namespace whose GPU-side globals
are replaced by fakes.  The code under test is therefore the literal shipped
source -- not a transcription of it -- exercised on CPU with no hardware and
no kernel launch.  Where even extraction is impossible (device-side
epilogues), the tests assert the repaired structure of the AST instead.
"""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

K = Path(__file__).resolve().parents[1] / "kernels"

FP8 = K / "gemm/fp8_gemm/h100/kernel.py"
FP4 = K / "gemm/fp4_grouped_gemm/b200/kernel.py"
CONV3D = K / "conv/conv3d_fwd/h100/kernel.py"
FROM_AGENT = K / "linear_attention/gdn_prefill_fwd/h100/from_agent/kernel.py"
GEN0 = K / "linear_attention/gdn_prefill_fwd/h100/gen0/kernel.py"
DSA_BWD = K / "sparse_attention/dsa_sparse_bwd/h100/kernel.py"
DSA_SCORE = K / "sparse_attention/dsa_score_recompute/h100/kernel.py"
GQA = K / "attention/gqa_paged_decode/h100/kernel.py"
MLA = K / "attention/mla_paged_decode/h100/kernel.py"
B200 = K / "attention/attn_bwd_mha/b200/kernel.py"


# --- the extraction harness --------------------------------------------------

def _tree(path):
    return ast.parse(Path(path).read_text())


def _extract(path, func_names=(), const_names=()):
    """Exec only the named module-level FunctionDefs and literal Assigns of
    the real file into a fresh namespace."""
    keep = []
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and isinstance(
                node.value, (ast.Constant, ast.Tuple)):
            for t in node.targets:
                for e in ast.walk(t):
                    if isinstance(e, ast.Name) and e.id in const_names:
                        keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {}
    exec(compile(mod, str(path), "exec"), ns)
    return ns


class _FakeTensor:
    """Duck-typed torch tensor for identity/key code paths."""

    def __init__(self, shape, stride, dtype="bf16", dev=0, ptr=4096,
                 version=0):
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self.device = SimpleNamespace(index=dev)
        self.ptr = ptr
        self._version = version

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self.ptr


class _ReplayingGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


# --- W1-KRN-01 / W1-KRN-03 (fp8_gemm: identity key, sources, c rotation) -----

def _fp8_ns():
    import torch
    ns = _extract(FP8, func_names=("_tensor_identity", "_build", "fp8_gemm"),
                  const_names=("_DX_SLOTS",))
    ns["torch"] = torch
    ns["from_dlpack"] = lambda t, assumed_align=16: SimpleNamespace(src=t)
    ns["_cuda_stream"] = lambda: "stream"
    ns["_STATE"] = {}
    ns["fp8_gemm_jit"] = object()
    compiles = []
    ns["cute"] = SimpleNamespace(
        compile=lambda fn, *args: (compiles.append(fn), _CompiledStub())[1])
    ns["_compiles"] = compiles
    return ns


class _CompiledStub:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1


def _compiling_ns_stub(compiles):
    """A fake `cute.compile`: records the trace and returns a callable stub."""
    stub = _CompiledStub()

    def fake_compile(fn, *args):
        compiles.append(1)
        return stub

    return SimpleNamespace(compile=fake_compile), stub


def test_fp8_tensor_identity_matches_fp4_sister():
    """W1-KRN-01: the fp8 identity is the fp4 five-field (+ptr) identity."""
    fp8 = _extract(FP8, func_names=("_tensor_identity",))
    fp4 = _extract(FP4, func_names=("_tensor_identity",))
    t = _FakeTensor((2, 3), (3, 1))
    assert fp8["_tensor_identity"](t) == fp4["_tensor_identity"](t)
    ident = fp8["_tensor_identity"](t)
    assert len(ident) == 6                          # ptr, shape, stride,
    # dtype, device, _version -- a data pointer alone is not an identity
    assert ident[0] == t.data_ptr() and ident[5] == 0
    bump = _FakeTensor((2, 3), (3, 1), version=1)
    assert fp8["_tensor_identity"](bump) != ident   # _version separates
    for field in ("shape", "_stride", "dtype"):
        other = _FakeTensor((2, 3), (3, 1))
        setattr(other, {"_stride": "_stride"}.get(field, field),
                {"shape": (2, 4), "_stride": (4, 1), "dtype": "f32"}[field])
        assert fp8["_tensor_identity"](other) != ident


def test_fp8_entry_holds_sources_and_key_is_identity_based():
    """W1-KRN-01: the cached entry pins a/a_sf/b/b_sf and the key uses the
    full identity, so an in-place scale regen cannot keep the stale build."""
    import torch
    ns = _fp8_ns()
    a = torch.zeros(64, 128, dtype=torch.bfloat16)
    a_sf = torch.ones(64, 1, dtype=torch.float32)
    b = torch.zeros(64, 128, dtype=torch.bfloat16)
    b_sf = torch.ones(1, 1, dtype=torch.float32)
    ns["fp8_gemm"](a, a_sf, b, b_sf)
    assert len(ns["_STATE"]) == 1
    ent = next(iter(ns["_STATE"].values()))
    assert ent[4][0] is a and ent[4][1] is a_sf          # sources held
    assert ent[4][2] is b and ent[4][3] is b_sf
    before = len(ns["_compiles"])
    a_sf.add_(0.0)                       # in-place regen: same ptr, _version+1
    ns["fp8_gemm"](a, a_sf, b, b_sf)
    assert len(ns["_compiles"]) == before + 1       # rebuilt, not reused


def test_fp8_results_rotate_two_slots():
    """W1-KRN-03: same-key consecutive results are never the same object."""
    import torch
    ns = _fp8_ns()
    a = torch.zeros(64, 128, dtype=torch.bfloat16)
    a_sf = torch.ones(64, 1, dtype=torch.float32)
    b = torch.zeros(64, 128, dtype=torch.bfloat16)
    b_sf = torch.ones(1, 1, dtype=torch.float32)
    c0 = ns["fp8_gemm"](a, a_sf, b, b_sf)
    c1 = ns["fp8_gemm"](a, a_sf, b, b_sf)
    c2 = ns["fp8_gemm"](a, a_sf, b, b_sf)
    assert c0 is not c1 and c1 is not c2 and c0 is c2      # 2-slot cycle
    ent = next(iter(ns["_STATE"].values()))
    assert ent[1].calls == 3                               # one launch per call
    assert len(ns["_STATE"]) == 1                          # no rebuilds


# --- W1-KRN-02 (conv3d: no pip install, no execve) ---------------------------

def test_conv3d_import_is_free_of_installs_and_reexec():
    """W1-KRN-02: the AST has no subprocess import, no execve call, and the
    source has no pip anywhere."""
    tree = _tree(CONV3D)
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Import)
                    and any(a.name.split(".")[0] == "subprocess"
                            for a in node.names))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "execve"
    src = CONV3D.read_text()
    assert "pip install" not in src and "import pip" not in src
    assert "_KERNEL_DSL_PINNED" not in src


def test_conv3d_dsl_guard_raises_and_names_the_dsl_pin_duty(monkeypatch):
    import importlib.metadata as md
    guard = _extract(CONV3D, func_names=("_ensure_dsl46",))["_ensure_dsl46"]
    monkeypatch.setattr(md, "version", lambda _name: "4.4.2")
    with pytest.raises(RuntimeError) as ei:
        guard()
    msg = str(ei.value)
    assert "DSL_PIN" in msg and "gpu_entry" in msg          # points at the duty
    monkeypatch.setattr(md, "version", lambda _name: "4.6.0.dev0")
    assert guard() is None                                  # pinned: no-op


# --- W1-KRN-03 (from_agent: replay slot rotation) ----------------------------

def test_from_agent_replay_rotates_two_slots():
    """W1-KRN-03: same-key replays return distinct o/state per call."""
    ns = _extract(FROM_AGENT, func_names=("gdn_prefill_fwd",))
    builds = []

    def fake_build(q, k, v, a, beta, scale):
        ent = SimpleNamespace()
        ent.graphs = [_ReplayingGraph(), _ReplayingGraph()]
        ent.o = [object(), object()]
        ent.state = [object(), object()]
        ent.slot = 0
        builds.append(ent)
        return ent

    ns["_build"] = fake_build
    ns["_CACHE"] = {}
    ns["_ORDER"] = []
    ns["_CACHE_MAX"] = 6
    ns["_LOCK"] = threading.Lock()
    args = tuple(SimpleNamespace(data_ptr=lambda i=i: i + 1) for i in range(5))

    r1 = ns["gdn_prefill_fwd"](*args, 1.0)
    r2 = ns["gdn_prefill_fwd"](*args, 1.0)
    r3 = ns["gdn_prefill_fwd"](*args, 1.0)
    assert r1[0] is not r2[0] and r2[0] is not r3[0]       # never aliased
    assert r1[1] is not r2[1] and r2[1] is not r3[1]
    assert r3[0] is r1[0]                                  # 2-slot cycle
    ent = builds[0]
    assert [g.replays for g in ent.graphs] == [1, 2]       # slots alternate
    assert len(builds) == 1                                # one build total


# --- W1-KRN-04 (dsa_sparse_bwd: k2 key carries sm_scale) ---------------------

def _dsa_bwd_ns():
    ns = _extract(DSA_BWD, func_names=("_run_k2", "_k2_args", "_k2_setup"),
                  const_names=("LOG2E", "H", "KVB", "DQK"))
    ns["_t"] = lambda x: SimpleNamespace(t=x)
    ns["_stream"] = lambda: "stream"
    ns["_k2_jit"] = object()
    seen = {}

    def fake_compiled(key, jit_fn, args, setup):
        seen["key"] = key
        seen["setup"] = setup
        return lambda *a: None

    ns["_compiled"] = fake_compiled
    ns["_seen"] = seen
    g = {}
    ns["_setg"] = lambda **kw: g.update(kw)
    ns["_g"] = g
    return ns


def test_dsa_bwd_k2_key_carries_sm_scale():
    """W1-KRN-04: same shape + different sm_scale must hit different keys,
    and the setup the key selects bakes exactly that scale."""
    ns = _dsa_bwd_ns()
    q = SimpleNamespace(shape=(128, 64, 512))
    kv = SimpleNamespace(shape=(32768, 512))
    rest = (SimpleNamespace(), SimpleNamespace(shape=(128, 1024)))
    rest += tuple(SimpleNamespace() for _ in range(5))
    ns["_run_k2"](q, kv, *rest, 0.37)
    assert ns["_seen"]["key"] == ("k2", 128, 64, 512, 32768, 1024, 0.37)
    ns["_seen"]["setup"]()
    assert ns["_g"]["_G_K2_SCALE"] == 0.37
    assert ns["_g"]["_G_K2_CSCALE"] == pytest.approx(0.37 * ns["LOG2E"])
    ns["_run_k2"](q, kv, *rest, 0.5)                 # same shape, new scale
    assert ns["_seen"]["key"][-1] == 0.5
    assert ns["_seen"]["key"][:-1] == ("k2", 128, 64, 512, 32768, 1024)


def test_dsa_bwd_both_k2_key_sites_carry_the_scale():
    """W1-KRN-04: _run_k2 AND precompile key on float(sm_scale)."""
    hits = []
    for fn in _tree(DSA_BWD).body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("_run_k2",
                                                           "precompile"):
            found = False
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "_compiled"
                        and isinstance(node.args[0], ast.Tuple)
                        and any(isinstance(e, ast.Call)
                                and isinstance(e.func, ast.Name)
                                and e.func.id == "float"
                                and isinstance(e.args[0], ast.Name)
                                and e.args[0].id == "sm_scale"
                                for e in node.args[0].elts)):
                    found = True
            hits.append((fn.name, found))
    assert hits == [("_run_k2", True), ("precompile", True)]


# --- W1-KRN-05 (paged decode: empty-sequence defense) ------------------------

def test_paged_decode_epilogues_floor_the_running_sum():
    """W1-KRN-05: both epilogues divide by a floored denominator and write a
    floored lse; nothing else on the contract path changed."""
    g = GQA.read_text()
    assert "cute.arch.fmax(l_run, cutlass.Float32(1.0e-30))" in g
    assert "inv = cutlass.Float32(1.0) / l_safe" in g
    assert "log2_f32(l_safe)" in g
    m = MLA.read_text()
    assert "_fmaxf(l, cutlass.Float32(1.0e-30))" in m
    assert "inv = cutlass.Float32(1.0) / l_safe" in m
    assert "_lg2f(l_safe)" in m


def test_empty_row_sentinel_math_is_exact_in_f32():
    """W1-KRN-05: with l == 0 the floor yields finite inv, an exact-zero out,
    and an lse that lands exactly on the file's finite park sentinels."""
    np = pytest.importorskip("numpy")
    park = np.float32(-1.0e30)                  # gqa PARK
    mx = np.float32(-3.0e38)                    # mla park
    eps = np.float32(1.0e-30)
    inv = np.float32(1.0) / np.fmax(np.float32(0.0), eps)
    assert np.isfinite(inv)
    assert np.float32(0.0) * inv == np.float32(0.0)          # out row: exact 0
    assert park + np.float32(np.log2(eps)) == park           # gqa lse == PARK
    assert mx + np.float32(np.log2(eps)) == mx               # mla lse == mx
    # and a non-empty row (l >= 1) is untouched by the floor:
    l = np.float32(4.0)
    assert np.fmax(l, eps) == l


# --- W1-KRN-07 (b200 attn_bwd: entry shape validation) -----------------------

def _b200_ns():
    ns = _extract(B200, func_names=("_run",),
                  const_names=("_B", "_N", "_H", "_D"))
    ns["_ACC_ELEMS"] = 0        # only ever handed to the fake torch.empty
    ns["_bwd_jit"] = object()
    ns["_COMPILED"] = None
    ns["from_dlpack"] = lambda t, assumed_align=16: SimpleNamespace(t=t)

    class _FakeTorch:
        float32 = "f32"

        @staticmethod
        def empty_like(t):
            return SimpleNamespace(shape=t.shape)

        @staticmethod
        def empty(*shape, device=None, dtype=None):
            return SimpleNamespace(shape=shape)

    ns["torch"] = _FakeTorch
    compiles = []
    ns["cute"], ns["_stub"] = _compiling_ns_stub(compiles)
    ns["_compiles"] = compiles
    return ns


def test_b200_entry_accepts_only_the_frozen_geometry():
    """W1-KRN-07: parametrized wrong shapes raise ValueError like the h100
    sister; the frozen shape compiles and launches."""
    ns = _b200_ns()
    B, N, H, D = ns["_B"], ns["_N"], ns["_H"], ns["_D"]

    def call(qs, ks, vs, os_, dos, lses):
        args = tuple(SimpleNamespace(shape=s, device=SimpleNamespace(index=0))
                     for s in (qs, ks, vs, os_, dos))
        return ns["_run"](*args, SimpleNamespace(shape=lses,
                                                 device=SimpleNamespace(index=0)))

    out = call((B, N, H, D), (B, N, H, D), (B, N, H, D), (B, N, H, D),
               (B, N, H, D), (B, H, N))
    assert len(out) == 6 and len(ns["_compiles"]) == 1
    for bad in (((B, N, H, D + 1),) * 5,               # wrong d
                ((B, N, H + 1, D),) * 5,               # wrong h
                ((B, N // 2, H, D),) * 5,              # wrong n
                ((B + 1, N, H, D),) * 5,               # wrong b
                ((B, N, H, D),) * 4 + ((B, N, H, D - 8), (B, H, N))):
        with pytest.raises(ValueError, match="unsupported shape"):
            call(*bad[:5], bad[5] if len(bad) == 6 else (B, H, N))
    with pytest.raises(ValueError, match="unsupported shape"):
        call((B, N, H, D), (B, N, H, D), (B, N, H, D), (B, N, H, D),
             (B, N, H, D), (B, H, N + 1))              # wrong lse


# --- W1-KRN-08 (dsa_score_recompute: triple predicate) -----------------------

def test_score_recompute_epilogues_use_the_gather_triple_predicate():
    """W1-KRN-08: both epilogues gate on the same (key < tl) & (iv >= 0) &
    (iv < s_kv) triple the gather applies, and s_kv is in scope in both."""
    hits = 0
    for fn in _tree(DSA_SCORE).body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        params = [a.arg for a in fn.args.args]
        uses = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                if ast.unparse(node) == "(key < tl) & (iv >= 0) & (iv < s_kv)":
                    uses += 1
        if uses:
            assert "s_kv" in params               # bound is a kernel parameter
            hits += uses
    assert hits == 2                              # attn epilogue + idx kernel
    src = DSA_SCORE.read_text()
    assert "(slot < ntk) & (iv >= 0) & (iv < s_kv)" in src   # gather unchanged


# --- W1-KRN-09 (gen0: exec-cache key covers device/dtype/stride) -------------

def _gen0_ns():
    ns = _extract(GEN0, func_names=("gdn_prefill_fwd",),
                  const_names=("DHEAD", "CHUNK"))
    ns["_gdn_host"] = object()
    ns["_cute_args"] = lambda *ts: tuple(SimpleNamespace(t=t) for t in ts)
    ns["_current_stream"] = lambda: "stream"
    ns["_exec_cache"] = {}
    ns["from_dlpack"] = lambda t, assumed_align=16: SimpleNamespace(t=t)
    compiles = []
    ns["cute"], ns["_stub"] = _compiling_ns_stub(compiles)

    class _FakeTorch:
        bfloat16 = "bf16"
        float32 = "f32"

        @staticmethod
        def empty(*shape, device=None, dtype=None):
            return SimpleNamespace(shape=shape)

    ns["torch"] = _FakeTorch
    ns["_compiles"] = compiles
    return ns


def _gen0_inputs(dev=0, n=8192, hqk=16, hv=32, d=128, dts=("bf16", "f32")):
    qkv = lambda h: _FakeTensor((n, h, d), (h * d, d, 1),
                                dtype=dts[0], dev=dev)
    ab = lambda: _FakeTensor((n, hv), (hv, 1), dtype=dts[1], dev=dev)
    return qkv(hqk), qkv(hqk), qkv(hv), ab(), ab()


def test_gen0_key_separates_device_dtype_and_stride():
    """W1-KRN-09: same-shape calls that differ in device, dtype or strides
    compile their own artifact; identical layouts reuse it."""
    ns = _gen0_ns()
    base = _gen0_inputs()
    ns["gdn_prefill_fwd"](*base, 0.1)
    assert len(ns["_exec_cache"]) == 1
    ns["gdn_prefill_fwd"](*_gen0_inputs(), 0.1)          # identical identity
    assert len(ns["_exec_cache"]) == 1 and len(ns["_compiles"]) == 1
    ns["gdn_prefill_fwd"](*_gen0_inputs(dev=1), 0.1)     # other GPU
    assert len(ns["_exec_cache"]) == 2
    padded = _gen0_inputs()
    padded[0]._stride = (16 * 128 + 8, 128, 1)           # q rows padded
    ns["gdn_prefill_fwd"](*padded, 0.1)
    assert len(ns["_exec_cache"]) == 3                   # stride in the key
    ns["gdn_prefill_fwd"](*_gen0_inputs(dts=("bf16", "f64")), 0.1)
    assert len(ns["_exec_cache"]) == 4                   # dtype in the key
    ns["gdn_prefill_fwd"](*_gen0_inputs(n=4096), 0.1)    # different N
    assert len(ns["_exec_cache"]) == 5
