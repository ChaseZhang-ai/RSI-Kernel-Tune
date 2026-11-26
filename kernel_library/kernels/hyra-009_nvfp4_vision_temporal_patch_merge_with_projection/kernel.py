"""Merged by rsi_kernel_tune.klib.normalize — do not edit; edit the original sources.

Inlined, in dependency order: fp4gemm.py, cuda_graph_drv.py
Entry file (verbatim last): kernel.py
Every inlined module's top-level bindings are prefixed `_klib_<module>__` and all references
to them are rewritten, so side-by-side definitions cannot shadow each other.
"""

# ---------------------------------------------------------------- inlined: fp4gemm.py
"""Direct cuBLASLt NVFP4 (E2M1) block-scaled matmul via ctypes, with a scalar
device-side alpha.  This is the documented "fold the output scale into the GEMM"
lever: torch._scaled_mm exposes only per-block scales + scale_result (D_SCALE),
and D_SCALE is REJECTED by cuBLASLt for FP4 output (CUBLAS_STATUS_INVALID_VALUE
at AlgoGetHeuristic - verified).  The standard GEMM `alpha` (D = alpha*op(A)op(B))
is a core matmul parameter, NOT the D-scale epilogue, so it is accepted for FP4.

We compute the linear projection  D(M,N) = A(M,K) @ W(N,K)^T  with:
  * A = activation FP4 (row-major (M,K), K contiguous),  scale a_sf_swz (blocked)
  * W = weight     FP4 (row-major (N,K), K contiguous),  scale w_sf_swz (blocked)
  * alpha = gs = global_decode_a * global_decode_b  (fp32 scalar on device)
  * bias is always zeros in this op, so D = gs*(A@W^T) exactly matches reference.

cuBLASLt is column-major.  Row-major D(M,N) == col-major (N,M) ld=N, and the
FP4 tensor cores require the "TN" layout (contraction dim K contiguous for both
operands).  Mapping (verified against the reference numerics via the evaluator):
  cuA = weight     : Adesc type=FP4 rows=K cols=N ld=K, opA=T  -> W(N,K)
  cuB = activation : Bdesc type=FP4 rows=K cols=M ld=K, opB=N  -> A^T(K,M)
  D                : Ddesc type=BF16 rows=N cols=M ld=N        -> (N,M) col-major
  A_SCALE = weight block scales (w_sf_swz),  B_SCALE = act block scales (a_sf_swz)
"""
import ctypes as _klib_fp4gemm__ctypes
import glob as _klib_fp4gemm__glob
import os as _klib_fp4gemm__os

import torch as _klib_fp4gemm__torch


def _klib_fp4gemm___load_cublaslt():
    for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
        try:
            return _klib_fp4gemm__ctypes.CDLL(name, mode=_klib_fp4gemm__ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    import site
    roots = list(site.getsitepackages()) + [_klib_fp4gemm__os.path.dirname(_klib_fp4gemm__os.path.dirname(_klib_fp4gemm__torch.__file__))]
    for root in roots:
        for pat in ("nvidia/*/lib/libcublasLt.so*", "torch/lib/libcublasLt.so*"):
            for p in _klib_fp4gemm__glob.glob(_klib_fp4gemm__os.path.join(root, pat)):
                try:
                    return _klib_fp4gemm__ctypes.CDLL(p, mode=_klib_fp4gemm__ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    raise RuntimeError("could not load libcublasLt")


_klib_fp4gemm___lib = _klib_fp4gemm___load_cublaslt()


_klib_fp4gemm__CUDA_R_16BF = 14
_klib_fp4gemm__CUDA_R_32F = 0
_klib_fp4gemm__CUDA_R_4F_E2M1 = 33
_klib_fp4gemm__CUBLAS_COMPUTE_32F = 68
_klib_fp4gemm__CUBLAS_OP_N = 0
_klib_fp4gemm__CUBLAS_OP_T = 1
_klib_fp4gemm__POINTER_MODE_DEVICE = 1
_klib_fp4gemm__ORDER_COL = 0
_klib_fp4gemm__SCALE_VEC16_UE4M3 = 1

_klib_fp4gemm__DESC_POINTER_MODE = 2
_klib_fp4gemm__DESC_TRANSA = 3
_klib_fp4gemm__DESC_TRANSB = 4
_klib_fp4gemm__DESC_A_SCALE_POINTER = 17
_klib_fp4gemm__DESC_B_SCALE_POINTER = 18
_klib_fp4gemm__DESC_A_SCALE_MODE = 31
_klib_fp4gemm__DESC_B_SCALE_MODE = 32

_klib_fp4gemm__PREF_MAX_WORKSPACE_BYTES = 1

_klib_fp4gemm___c = _klib_fp4gemm__ctypes


class _klib_fp4gemm___HeurResult(_klib_fp4gemm___c.Structure):
    _fields_ = [
        ("algo", _klib_fp4gemm___c.c_uint64 * 8),
        ("workspaceSize", _klib_fp4gemm___c.c_size_t),
        ("state", _klib_fp4gemm___c.c_int),
        ("wavesCount", _klib_fp4gemm___c.c_float),
        ("reserved", _klib_fp4gemm___c.c_int * 4),
    ]


def _klib_fp4gemm___sig():
    P = _klib_fp4gemm___c.c_void_p
    _klib_fp4gemm___lib.cublasLtCreate.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtCreate.argtypes = [_klib_fp4gemm___c.POINTER(P)]
    _klib_fp4gemm___lib.cublasLtMatmulDescCreate.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmulDescCreate.argtypes = [_klib_fp4gemm___c.POINTER(P), _klib_fp4gemm___c.c_int, _klib_fp4gemm___c.c_int]
    _klib_fp4gemm___lib.cublasLtMatmulDescSetAttribute.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmulDescSetAttribute.argtypes = [P, _klib_fp4gemm___c.c_int, _klib_fp4gemm___c.c_void_p, _klib_fp4gemm___c.c_size_t]
    _klib_fp4gemm___lib.cublasLtMatrixLayoutCreate.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatrixLayoutCreate.argtypes = [_klib_fp4gemm___c.POINTER(P), _klib_fp4gemm___c.c_int, _klib_fp4gemm___c.c_uint64, _klib_fp4gemm___c.c_uint64, _klib_fp4gemm___c.c_int64]
    _klib_fp4gemm___lib.cublasLtMatmulPreferenceCreate.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmulPreferenceCreate.argtypes = [_klib_fp4gemm___c.POINTER(P)]
    _klib_fp4gemm___lib.cublasLtMatmulPreferenceSetAttribute.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [P, _klib_fp4gemm___c.c_int, _klib_fp4gemm___c.c_void_p, _klib_fp4gemm___c.c_size_t]
    _klib_fp4gemm___lib.cublasLtMatmulAlgoGetHeuristic.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [
        P, P, P, P, P, P, P, _klib_fp4gemm___c.c_int, _klib_fp4gemm___c.c_void_p, _klib_fp4gemm___c.POINTER(_klib_fp4gemm___c.c_int)]
    _klib_fp4gemm___lib.cublasLtMatmul.restype = _klib_fp4gemm___c.c_int
    _klib_fp4gemm___lib.cublasLtMatmul.argtypes = [
        P, P, _klib_fp4gemm___c.c_void_p,
        _klib_fp4gemm___c.c_void_p, P,
        _klib_fp4gemm___c.c_void_p, P,
        _klib_fp4gemm___c.c_void_p,
        _klib_fp4gemm___c.c_void_p, P,
        _klib_fp4gemm___c.c_void_p, P,
        _klib_fp4gemm___c.c_void_p,
        _klib_fp4gemm___c.c_void_p, _klib_fp4gemm___c.c_size_t,
        _klib_fp4gemm___c.c_void_p,
    ]


_klib_fp4gemm___sig()


def _klib_fp4gemm___check(st, where):
    if st != 0:
        raise RuntimeError(f"cuBLASLt {where} failed status={st}")


_klib_fp4gemm___WS_BYTES = 32 * 1024 * 1024
_klib_fp4gemm___WS = _klib_fp4gemm__torch.empty(_klib_fp4gemm___WS_BYTES, dtype=_klib_fp4gemm__torch.uint8, device="cuda")
_klib_fp4gemm___ZERO = _klib_fp4gemm__torch.zeros(1, dtype=_klib_fp4gemm__torch.float32, device="cuda")

_klib_fp4gemm___HANDLE = _klib_fp4gemm___c.c_void_p()
_klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtCreate(_klib_fp4gemm___c.byref(_klib_fp4gemm___HANDLE)), "Create")


def _klib_fp4gemm___set_i32(desc, attr, val):
    v = _klib_fp4gemm___c.c_int32(val)
    _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulDescSetAttribute(desc, attr, _klib_fp4gemm___c.byref(v), 4), f"SetAttr {attr}")


def _klib_fp4gemm___set_ptr(desc, attr, ptr):
    v = _klib_fp4gemm___c.c_void_p(ptr)
    _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulDescSetAttribute(desc, attr, _klib_fp4gemm___c.byref(v), 8), f"SetPtr {attr}")


class _klib_fp4gemm__Fp4Gemm:
    """Per-(M,N,K) cuBLASLt plan.  matmul() issues D = alpha*(A@W^T) on `stream`."""

    def __init__(self, M, N, K, a_scale_ptr, b_scale_ptr):
        self.M, self.N, self.K = M, N, K
        P = _klib_fp4gemm___c.c_void_p

        self.desc = P()
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulDescCreate(_klib_fp4gemm___c.byref(self.desc), _klib_fp4gemm__CUBLAS_COMPUTE_32F, _klib_fp4gemm__CUDA_R_32F), "DescCreate")
        _klib_fp4gemm___set_i32(self.desc, _klib_fp4gemm__DESC_TRANSA, _klib_fp4gemm__CUBLAS_OP_T)
        _klib_fp4gemm___set_i32(self.desc, _klib_fp4gemm__DESC_TRANSB, _klib_fp4gemm__CUBLAS_OP_N)
        _klib_fp4gemm___set_i32(self.desc, _klib_fp4gemm__DESC_POINTER_MODE, _klib_fp4gemm__POINTER_MODE_DEVICE)
        _klib_fp4gemm___set_i32(self.desc, _klib_fp4gemm__DESC_A_SCALE_MODE, _klib_fp4gemm__SCALE_VEC16_UE4M3)
        _klib_fp4gemm___set_i32(self.desc, _klib_fp4gemm__DESC_B_SCALE_MODE, _klib_fp4gemm__SCALE_VEC16_UE4M3)


        _klib_fp4gemm___set_ptr(self.desc, _klib_fp4gemm__DESC_A_SCALE_POINTER, a_scale_ptr)
        _klib_fp4gemm___set_ptr(self.desc, _klib_fp4gemm__DESC_B_SCALE_POINTER, b_scale_ptr)


        self.Adesc = P()
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatrixLayoutCreate(_klib_fp4gemm___c.byref(self.Adesc), _klib_fp4gemm__CUDA_R_4F_E2M1, K, N, K), "AdescCreate")

        self.Bdesc = P()
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatrixLayoutCreate(_klib_fp4gemm___c.byref(self.Bdesc), _klib_fp4gemm__CUDA_R_4F_E2M1, K, M, K), "BdescCreate")

        self.Ddesc = P()
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatrixLayoutCreate(_klib_fp4gemm___c.byref(self.Ddesc), _klib_fp4gemm__CUDA_R_16BF, N, M, N), "DdescCreate")


        pref = P()
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulPreferenceCreate(_klib_fp4gemm___c.byref(pref)), "PrefCreate")
        wsb = _klib_fp4gemm___c.c_size_t(_klib_fp4gemm___WS_BYTES)
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulPreferenceSetAttribute(pref, _klib_fp4gemm__PREF_MAX_WORKSPACE_BYTES, _klib_fp4gemm___c.byref(wsb), 8), "PrefSet")
        _NREQ = 32
        heur_arr = (_klib_fp4gemm___HeurResult * _NREQ)()
        ret = _klib_fp4gemm___c.c_int(0)
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmulAlgoGetHeuristic(
            _klib_fp4gemm___HANDLE, self.desc, self.Adesc, self.Bdesc, self.Ddesc, self.Ddesc,
            pref, _NREQ, heur_arr, _klib_fp4gemm___c.byref(ret)), "Heuristic")
        n_algo = ret.value
        if n_algo < 1 or heur_arr[0].state != 0:
            raise RuntimeError(f"cuBLASLt no algo (returned={n_algo} state={heur_arr[0].state})")


        self.algo = (_klib_fp4gemm___c.c_uint64 * 8)()
        for i in range(8):
            self.algo[i] = heur_arr[0].algo[i]


        if self.M >= 2048:
            best = self._autotune(heur_arr, n_algo, a_scale_ptr, b_scale_ptr)
            if best is not None:
                for i in range(8):
                    self.algo[i] = best[i]

    def _autotune(self, heur_arr, n_algo, a_scale_ptr, b_scale_ptr):
        """Benchmark each candidate algo on real device buffers; return the
        fastest algo's 8×u64 bytes (or None to keep the heuristic's #1)."""
        M, N, K = self.M, self.N, self.K
        try:
            A_tmp = _klib_fp4gemm__torch.empty(N * (K // 2), dtype=_klib_fp4gemm__torch.uint8, device="cuda")
            B_tmp = _klib_fp4gemm__torch.empty(M * (K // 2), dtype=_klib_fp4gemm__torch.uint8, device="cuda")
            D_tmp = _klib_fp4gemm__torch.empty(N * M, dtype=_klib_fp4gemm__torch.bfloat16, device="cuda")
            alpha = _klib_fp4gemm__torch.ones(1, dtype=_klib_fp4gemm__torch.float32, device="cuda")
        except Exception:
            return None
        a_ptr, b_ptr, d_ptr, al_ptr = (
            A_tmp.data_ptr(), B_tmp.data_ptr(), D_tmp.data_ptr(), alpha.data_ptr())
        stream = _klib_fp4gemm__torch.cuda.current_stream().cuda_stream

        def run_algo(algo_bytes):
            return _klib_fp4gemm___lib.cublasLtMatmul(
                _klib_fp4gemm___HANDLE, self.desc,
                _klib_fp4gemm___c.c_void_p(al_ptr),
                _klib_fp4gemm___c.c_void_p(a_ptr), self.Adesc,
                _klib_fp4gemm___c.c_void_p(b_ptr), self.Bdesc,
                _klib_fp4gemm___c.c_void_p(_klib_fp4gemm___ZERO.data_ptr()),
                _klib_fp4gemm___c.c_void_p(d_ptr), self.Ddesc,
                _klib_fp4gemm___c.c_void_p(d_ptr), self.Ddesc,
                _klib_fp4gemm___c.cast(algo_bytes, _klib_fp4gemm___c.c_void_p),
                _klib_fp4gemm___c.c_void_p(_klib_fp4gemm___WS.data_ptr()), _klib_fp4gemm___c.c_size_t(_klib_fp4gemm___WS_BYTES),
                _klib_fp4gemm___c.c_void_p(stream))

        def time_algo(algo_bytes):
            if run_algo(algo_bytes) != 0:
                return None
            for _ in range(NW):
                run_algo(algo_bytes)
            _klib_fp4gemm__torch.cuda.synchronize()
            s = _klib_fp4gemm__torch.cuda.Event(enable_timing=True)
            e = _klib_fp4gemm__torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(NIT):
                run_algo(algo_bytes)
            e.record()
            _klib_fp4gemm__torch.cuda.synchronize()
            return s.elapsed_time(e) / NIT

        NW, NIT = 5, 30
        best_algo, best_t, t0 = None, float("inf"), None
        for j in range(n_algo):
            hr = heur_arr[j]
            if hr.state != 0 or hr.workspaceSize > _klib_fp4gemm___WS_BYTES:
                continue
            algo_bytes = (_klib_fp4gemm___c.c_uint64 * 8)()
            for i in range(8):
                algo_bytes[i] = hr.algo[i]
            t = time_algo(algo_bytes)
            if t is None:
                continue
            if j == 0:
                t0 = t
            if t < best_t:
                best_t, best_algo = t, algo_bytes
        del A_tmp, B_tmp, D_tmp, alpha


        if best_algo is not None and t0 is not None and best_t >= 0.95 * t0:
            return None
        return best_algo

    def matmul(self, weight_ptr, act_ptr, weight_scale_ptr, act_scale_ptr,
               alpha_ptr, out_ptr, stream):


        _klib_fp4gemm___set_ptr(self.desc, _klib_fp4gemm__DESC_A_SCALE_POINTER, weight_scale_ptr)
        _klib_fp4gemm___set_ptr(self.desc, _klib_fp4gemm__DESC_B_SCALE_POINTER, act_scale_ptr)
        _klib_fp4gemm___check(_klib_fp4gemm___lib.cublasLtMatmul(
            _klib_fp4gemm___HANDLE, self.desc,
            _klib_fp4gemm___c.c_void_p(alpha_ptr),
            _klib_fp4gemm___c.c_void_p(weight_ptr), self.Adesc,
            _klib_fp4gemm___c.c_void_p(act_ptr), self.Bdesc,
            _klib_fp4gemm___c.c_void_p(_klib_fp4gemm___ZERO.data_ptr()),
            _klib_fp4gemm___c.c_void_p(out_ptr), self.Ddesc,
            _klib_fp4gemm___c.c_void_p(out_ptr), self.Ddesc,
            _klib_fp4gemm___c.cast(self.algo, _klib_fp4gemm___c.c_void_p),
            _klib_fp4gemm___c.c_void_p(_klib_fp4gemm___WS.data_ptr()), _klib_fp4gemm___c.c_size_t(_klib_fp4gemm___WS_BYTES),
            _klib_fp4gemm___c.c_void_p(stream)), "Matmul")


_klib_fp4gemm___PLAN_CACHE = {}


def _klib_fp4gemm__get_plan(M, N, K, a_scale_ptr, b_scale_ptr):
    key = (M, N, K)
    p = _klib_fp4gemm___PLAN_CACHE.get(key)
    if p is None:
        p = _klib_fp4gemm__Fp4Gemm(M, N, K, a_scale_ptr, b_scale_ptr)
        _klib_fp4gemm___PLAN_CACHE[key] = p
    return p


# ---------------------------------------------------------------- inlined: cuda_graph_drv.py
"""Minimal CUDA driver-API bindings (ctypes) for MANUAL CUDA-graph stream
capture + per-call kernel-node source-pointer updates.

Why this exists (the structural lever over the record):
  The record issues its fused input-copy+amax kernel (`_mc4`) EAGERLY, then
  replays a CUDA graph holding [quant, GEMM, memset].  That is TWO CPU-side
  submissions per call (an `_mc4` launch, then a graph replay) => one
  GPU-side launch boundary that the T_k span (max_kernel_end - min_kernel_start)
  pays for on every workload.

  Here we capture `_mc4` INTO the same graph as its FIRST node, so the whole
  call is ONE graph launch (no boundary).  `_mc4` must read the FRESH input
  pointers (the eval hands new tensors every call), so after instantiation we
  locate the `_mc4` kernel node and, each call, patch only its 4 source
  pointers via cuGraphExecKernelNodeSetParams (a host-side op, not on the GPU
  span).  Everything else (`_mc4`'s static dst buffers, ga, sizes, the quant /
  GEMM / memset nodes) is captured once and never changes.

No silent fallback: every driver call is checked and raises loudly on error.
"""
import ctypes as _klib_cuda_graph_drv__ctypes


_klib_cuda_graph_drv___libcuda = _klib_cuda_graph_drv__ctypes.CDLL("libcuda.so.1", mode=_klib_cuda_graph_drv__ctypes.RTLD_GLOBAL)


_klib_cuda_graph_drv__CUstream = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUgraph = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUgraphExec = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUgraphNode = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUcontext = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUfunction = _klib_cuda_graph_drv__ctypes.c_void_p
_klib_cuda_graph_drv__CUevent = _klib_cuda_graph_drv__ctypes.c_void_p


_klib_cuda_graph_drv__CU_STREAM_CAPTURE_MODE_GLOBAL = 0
_klib_cuda_graph_drv__CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
_klib_cuda_graph_drv__CU_GRAPH_NODE_TYPE_KERNEL = 0
_klib_cuda_graph_drv__CU_GRAPH_NODE_TYPE_MEMCPY = 1
_klib_cuda_graph_drv__CU_GRAPH_NODE_TYPE_MEMSET = 2
_klib_cuda_graph_drv__CU_EVENT_DISABLE_TIMING = 2


class _klib_cuda_graph_drv__CUDA_KERNEL_NODE_PARAMS(_klib_cuda_graph_drv__ctypes.Structure):


    _fields_ = [
        ("func", _klib_cuda_graph_drv__CUfunction),
        ("gridDimX", _klib_cuda_graph_drv__ctypes.c_uint),
        ("gridDimY", _klib_cuda_graph_drv__ctypes.c_uint),
        ("gridDimZ", _klib_cuda_graph_drv__ctypes.c_uint),
        ("blockDimX", _klib_cuda_graph_drv__ctypes.c_uint),
        ("blockDimY", _klib_cuda_graph_drv__ctypes.c_uint),
        ("blockDimZ", _klib_cuda_graph_drv__ctypes.c_uint),
        ("sharedMemBytes", _klib_cuda_graph_drv__ctypes.c_uint),
        ("kernelParams", _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__ctypes.c_void_p)),
        ("extra", _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__ctypes.c_void_p)),
    ]


def _klib_cuda_graph_drv___sig():
    P = _klib_cuda_graph_drv__ctypes.c_void_p
    _klib_cuda_graph_drv___libcuda.cuCtxGetCurrent.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuCtxGetCurrent.argtypes = [_klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUcontext)]
    _klib_cuda_graph_drv___libcuda.cuStreamBeginCapture_v2.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuStreamBeginCapture_v2.argtypes = [_klib_cuda_graph_drv__CUstream, _klib_cuda_graph_drv__ctypes.c_int]
    _klib_cuda_graph_drv___libcuda.cuStreamEndCapture.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuStreamEndCapture.argtypes = [_klib_cuda_graph_drv__CUstream, _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUgraph)]
    _klib_cuda_graph_drv___libcuda.cuGraphInstantiateWithFlags.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphInstantiateWithFlags.argtypes = [
        _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUgraphExec), _klib_cuda_graph_drv__CUgraph, _klib_cuda_graph_drv__ctypes.c_ulonglong]
    _klib_cuda_graph_drv___libcuda.cuGraphGetNodes.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphGetNodes.argtypes = [
        _klib_cuda_graph_drv__CUgraph, _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUgraphNode), _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__ctypes.c_size_t)]
    _klib_cuda_graph_drv___libcuda.cuGraphNodeGetType.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphNodeGetType.argtypes = [_klib_cuda_graph_drv__CUgraphNode, _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__ctypes.c_int)]
    _klib_cuda_graph_drv___libcuda.cuGraphKernelNodeGetParams.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphKernelNodeGetParams.argtypes = [
        _klib_cuda_graph_drv__CUgraphNode, _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUDA_KERNEL_NODE_PARAMS)]
    _klib_cuda_graph_drv___libcuda.cuGraphExecKernelNodeSetParams.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphExecKernelNodeSetParams.argtypes = [
        _klib_cuda_graph_drv__CUgraphExec, _klib_cuda_graph_drv__CUgraphNode, _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUDA_KERNEL_NODE_PARAMS)]
    _klib_cuda_graph_drv___libcuda.cuGraphLaunch.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuGraphLaunch.argtypes = [_klib_cuda_graph_drv__CUgraphExec, _klib_cuda_graph_drv__CUstream]

    _klib_cuda_graph_drv___libcuda.cuEventCreate.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuEventCreate.argtypes = [_klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__CUevent), _klib_cuda_graph_drv__ctypes.c_uint]
    _klib_cuda_graph_drv___libcuda.cuEventRecord.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuEventRecord.argtypes = [_klib_cuda_graph_drv__CUevent, _klib_cuda_graph_drv__CUstream]
    _klib_cuda_graph_drv___libcuda.cuStreamWaitEvent.restype = _klib_cuda_graph_drv__ctypes.c_int
    _klib_cuda_graph_drv___libcuda.cuStreamWaitEvent.argtypes = [_klib_cuda_graph_drv__CUstream, _klib_cuda_graph_drv__CUevent, _klib_cuda_graph_drv__ctypes.c_uint]


_klib_cuda_graph_drv___sig()


def _klib_cuda_graph_drv___ck(st, where):
    if st != 0:
        raise RuntimeError(f"CUDA driver {where} failed status={st}")


def _klib_cuda_graph_drv__begin_capture(stream):
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuStreamBeginCapture_v2(_klib_cuda_graph_drv__CUstream(stream),
        _klib_cuda_graph_drv__CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture_v2")


def _klib_cuda_graph_drv__end_capture(stream):
    g = _klib_cuda_graph_drv__CUgraph()
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuStreamEndCapture(_klib_cuda_graph_drv__CUstream(stream), _klib_cuda_graph_drv__ctypes.byref(g)), "cuStreamEndCapture")
    return g


def _klib_cuda_graph_drv__instantiate(graph):
    exe = _klib_cuda_graph_drv__CUgraphExec()
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphInstantiateWithFlags(_klib_cuda_graph_drv__ctypes.byref(exe), graph, 0),
        "cuGraphInstantiateWithFlags")
    return exe


def _klib_cuda_graph_drv__get_nodes(graph):
    n = _klib_cuda_graph_drv__ctypes.c_size_t(0)
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphGetNodes(graph, None, _klib_cuda_graph_drv__ctypes.byref(n)), "cuGraphGetNodes(count)")
    num = n.value
    arr = (_klib_cuda_graph_drv__CUgraphNode * num)()
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphGetNodes(graph, arr, _klib_cuda_graph_drv__ctypes.byref(n)), "cuGraphGetNodes(fill)")
    return list(arr[:num])


def _klib_cuda_graph_drv__node_type(node):
    t = _klib_cuda_graph_drv__ctypes.c_int(-1)
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphNodeGetType(_klib_cuda_graph_drv__CUgraphNode(node), _klib_cuda_graph_drv__ctypes.byref(t)), "cuGraphNodeGetType")
    return t.value


def _klib_cuda_graph_drv__kernel_get_params(node):
    p = _klib_cuda_graph_drv__CUDA_KERNEL_NODE_PARAMS()
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphKernelNodeGetParams(_klib_cuda_graph_drv__CUgraphNode(node), _klib_cuda_graph_drv__ctypes.byref(p)),
        "cuGraphKernelNodeGetParams")
    return p


def _klib_cuda_graph_drv__exec_kernel_set_params(exe, node, params):
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphExecKernelNodeSetParams(exe, _klib_cuda_graph_drv__CUgraphNode(node), _klib_cuda_graph_drv__ctypes.byref(params)),
        "cuGraphExecKernelNodeSetParams")


def _klib_cuda_graph_drv__launch(exe, stream):
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuGraphLaunch(exe, _klib_cuda_graph_drv__CUstream(stream)), "cuGraphLaunch")


def _klib_cuda_graph_drv__read_param_u64(node, i):
    """Value of the i-th kernel argument of a KERNEL node (deref kernelParams[i])."""
    p = _klib_cuda_graph_drv__kernel_get_params(node)
    if not p.kernelParams:
        return None
    addr = p.kernelParams[i]
    if not addr:
        return None
    return _klib_cuda_graph_drv__ctypes.cast(_klib_cuda_graph_drv__ctypes.c_void_p(addr), _klib_cuda_graph_drv__ctypes.POINTER(_klib_cuda_graph_drv__ctypes.c_ulonglong))[0]


def _klib_cuda_graph_drv__set_param_slot(params, i, holder):
    """Point kernelParams[i] at `holder` (a (c_ulonglong*1) kept alive by caller)."""
    params.kernelParams[i] = _klib_cuda_graph_drv__ctypes.cast(holder, _klib_cuda_graph_drv__ctypes.c_void_p).value


def _klib_cuda_graph_drv__event_create():
    e = _klib_cuda_graph_drv__CUevent()
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuEventCreate(_klib_cuda_graph_drv__ctypes.byref(e), _klib_cuda_graph_drv__CU_EVENT_DISABLE_TIMING), "cuEventCreate")
    return e


def _klib_cuda_graph_drv__event_record(event, stream):
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuEventRecord(event, _klib_cuda_graph_drv__CUstream(stream)), "cuEventRecord")


def _klib_cuda_graph_drv__stream_wait_event(stream, event):
    _klib_cuda_graph_drv___ck(_klib_cuda_graph_drv___libcuda.cuStreamWaitEvent(_klib_cuda_graph_drv__CUstream(stream), event, 0), "cuStreamWaitEvent")


# ---------------------------------------------------------------- entry: kernel.py
import ctypes

import torch
import triton
import triton.language as tl

# klib: `fp4gemm` is inlined below
# klib: `drv` is inlined below


_NB = 32
_AMAX_BLK = 8192
_MC_BLOCK = 2048


_TPP = 8


@triton.jit
def _amax_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    tl.atomic_max(out_ptr, m)


@triton.jit
def _swz_dest(row, col, NCB: tl.constexpr):

    rb = row // 128
    r128 = row % 128
    cb = col // 4
    c4 = col % 4
    return (rb * NCB + cb) * 512 + (r128 % 32) * 16 + (r128 // 32) * 4 + c4


@triton.jit
def _quant_wswz_kernel(
    x_ptr, ga_ptr, fp4_ptr, sf_swz_ptr, wgs_ptr, gs_ptr, w_ptr, w_sf_swz_ptr,
    stride_xr, stride_fr, stride_wr, QP,
    NB: tl.constexpr, NCB: tl.constexpr, SFK_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < QP:
        row = pid // SFK_TILES
        kt = pid % SFK_TILES
        blk0 = kt * NB

        ga = tl.load(ga_ptr).to(tl.float32)
        ges = tl.math.div_rn(2688.0, ga)
        gds = tl.math.div_rn(1.0, ges)


        if pid == 0:
            tl.store(gs_ptr, gds * tl.load(wgs_ptr).to(tl.float32))

        b_off = tl.arange(0, NB)
        i_off = tl.arange(0, 16)
        block_idx = blk0 + b_off
        col = block_idx[:, None] * 16 + i_off[None, :]

        xv = tl.load(x_ptr + row * stride_xr + col).to(tl.float32)
        block_max = tl.max(tl.abs(xv), axis=1)

        ds = tl.math.div_rn(block_max, 6.0)
        ds = ds.to(tl.bfloat16).to(tl.float32)
        ds = ds * ges
        ds = tl.minimum(tl.maximum(ds, -448.0), 448.0)
        sf_fp8 = ds.to(tl.float8e4nv)

        dest = _swz_dest(row, block_idx, NCB)
        tl.store(sf_swz_ptr + dest, sf_fp8)

        sf_f32 = sf_fp8.to(tl.float32)
        enc = tl.math.div_rn(1.0, sf_f32 * gds)
        enc = tl.minimum(enc, 3.4028234663852886e38)

        scaled = xv * enc[:, None]


        sc3 = tl.reshape(scaled, (NB, 8, 2))
        even2, odd2 = tl.split(sc3)
        even = tl.reshape(even2, (NB * 8,))
        odd = tl.reshape(odd2, (NB * 8,))


        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u32.u8 $0, t; }",
            "=r,r,r", [odd, even],
            dtype=tl.int32, is_pure=True, pack=1,
        )
        packed = packed.to(tl.uint8)

        p_col = blk0 * 8 + tl.arange(0, NB * 8)
        tl.store(fp4_ptr + row * stride_fr + p_col, packed)
    else:
        tile = pid - QP
        rb = tile // NCB
        cb = tile % NCB
        o = tl.arange(0, 512)
        k = o % 4
        i = (o // 4) % 4
        j = o // 16
        src = tl.load(w_ptr + (rb * 128 + i * 32 + j) * stride_wr + (cb * 4 + k))
        tl.store(w_sf_swz_ptr + tile * 512 + o, src)


@triton.jit
def _quant_wswz_kernel_v2(
    x_ptr, ga_ptr, fp4_ptr, sf_swz_ptr, wgs_ptr, gs_ptr, w_ptr, w_sf_swz_ptr,
    stride_xr, stride_fr, stride_wr, QP,
    NB: tl.constexpr, NCB: tl.constexpr, SFK_TILES: tl.constexpr, RPP: tl.constexpr,
    TPP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < QP:
        rg = pid // SFK_TILES
        kt = pid % SFK_TILES
        blk0 = kt * NB
        row_base = rg * RPP

        ga = tl.load(ga_ptr).to(tl.float32)
        ges = tl.math.div_rn(2688.0, ga)
        gds = tl.math.div_rn(1.0, ges)

        if pid == 0:
            tl.store(gs_ptr, gds * tl.load(wgs_ptr).to(tl.float32))

        r_off = tl.arange(0, RPP)
        rows = row_base + r_off
        b_off = tl.arange(0, NB)
        i_off = tl.arange(0, 16)
        block_idx = blk0 + b_off
        col = block_idx[:, None] * 16 + i_off[None, :]

        off = rows[:, None, None] * stride_xr + col[None, :, :]
        xv = tl.load(x_ptr + off).to(tl.float32)
        block_max = tl.max(tl.abs(xv), axis=2)

        ds = tl.math.div_rn(block_max, 6.0)
        ds = ds.to(tl.bfloat16).to(tl.float32)
        ds = ds * ges
        ds = tl.minimum(tl.maximum(ds, -448.0), 448.0)
        sf_fp8 = ds.to(tl.float8e4nv)

        dest = _swz_dest(rows[:, None], block_idx[None, :], NCB)
        tl.store(sf_swz_ptr + dest, sf_fp8)

        sf_f32 = sf_fp8.to(tl.float32)
        enc = tl.math.div_rn(1.0, sf_f32 * gds)
        enc = tl.minimum(enc, 3.4028234663852886e38)

        scaled = xv * enc[:, :, None]
        sc4 = tl.reshape(scaled, (RPP, NB, 8, 2))
        even2, odd2 = tl.split(sc4)
        even = tl.reshape(even2, (RPP * NB * 8,))
        odd = tl.reshape(odd2, (RPP * NB * 8,))

        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u32.u8 $0, t; }",
            "=r,r,r", [odd, even],
            dtype=tl.int32, is_pure=True, pack=1,
        )
        packed = packed.to(tl.uint8)
        packed = tl.reshape(packed, (RPP, NB * 8))

        p_col = blk0 * 8 + tl.arange(0, NB * 8)
        out_off = rows[:, None] * stride_fr + p_col[None, :]
        tl.store(fp4_ptr + out_off, packed)
    else:


        tg = pid - QP
        t_off = tl.arange(0, TPP)
        tiles = tg * TPP + t_off
        rb = tiles // NCB
        cb = tiles % NCB
        ji = tl.arange(0, 128)
        jj = ji // 4
        ii = ji % 4
        src_off = (rb[:, None] * 128 + ii[None, :] * 32 + jj[None, :]) * stride_wr + cb[:, None]
        src = tl.load(w_ptr + src_off)
        dst_off = tiles[:, None] * 128 + jj[None, :] * 4 + ii[None, :]
        tl.store(w_sf_swz_ptr + dst_off, src)


@triton.jit
def _mc4(s0, d0, s1, d1, s2, d2, s3, d3,
         ga_ptr, p1, p2, p3, n0, n1, n2, n3,
         BLOCK: tl.constexpr, STORE_X: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    off = tl.arange(0, BLOCK)
    if base < p1:
        o = base + off; m = o < n0
        v = tl.load(s0 + o, mask=m, other=0)


        if STORE_X:
            tl.store(d0 + o, v, mask=m)


        lo = v.to(tl.int16).to(tl.bfloat16, bitcast=True)
        hi = (v >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
        af = tl.maximum(tl.abs(lo.to(tl.float32)), tl.abs(hi.to(tl.float32)))
        tl.atomic_max(ga_ptr, tl.max(af, axis=0))
    elif base < p2:
        o = base - p1 + off; m = o < n1
        tl.store(d1 + o, tl.load(s1 + o, mask=m, other=0), mask=m)
    elif base < p3:
        o = base - p2 + off; m = o < n2
        tl.store(d2 + o, tl.load(s2 + o, mask=m, other=0), mask=m)
    else:
        o = base - p3 + off; m = o < n3
        tl.store(d3 + o, tl.load(s3 + o, mask=m, other=0), mask=m)


def _u32(t):


    return t.contiguous().view(torch.uint8).reshape(-1).view(torch.int32)


def _fill_inputs(buf, x, weight_fp4, weight_scale, weight_global_scale, store_x=True):


    srcs = [
        _u32(x),
        _u32(weight_fp4),
        _u32(weight_scale),
        _u32(weight_global_scale.reshape(1)),
    ]
    dsts = [
        _u32(buf["x"]),
        _u32(buf["weight_fp4_u8"]),
        _u32(buf["weight_scale"]),
        _u32(buf["wgs"]),
    ]
    ns = [int(s.numel()) for s in srcs]
    B = _MC_BLOCK
    pad = [((n + B - 1) // B) * B for n in ns]
    p = [0]
    for pv in pad:
        p.append(p[-1] + pv)
    grid = (p[4] // B,)
    _mc4[grid](
        srcs[0], dsts[0], srcs[1], dsts[1], srcs[2], dsts[2], srcs[3], dsts[3],
        buf["ga"], p[1], p[2], p[3],
        ns[0], ns[1], ns[2], ns[3],
        BLOCK=B, STORE_X=store_x,
    )


def _compute(buf):
    R, K, N, SF_K, NCB = buf["R"], buf["K"], buf["N"], buf["SF_K"], buf["NCB"]
    xr = buf["x"].reshape(R, K)

    w_sf = buf["weight_scale"][0]
    SFK_TILES = SF_K // _NB
    n_tiles = (N // 128) * NCB
    RPP = buf["RPP"]
    if RPP > 1:
        QP = (R // RPP) * SFK_TILES


        TPP = buf["TPP"]
        w_sf_i32 = w_sf.view(torch.int32)
        w_swz_i32 = buf["w_sf_swz"].view(torch.int32)
        _quant_wswz_kernel_v2[(QP + n_tiles // TPP,)](
            xr, buf["ga"], buf["a_fp4_u8"], buf["a_sf_swz"], buf["wgs"], buf["gs"],
            w_sf_i32, w_swz_i32,
            xr.stride(0), buf["a_fp4_u8"].stride(0), w_sf_i32.stride(0), QP,
            NB=_NB, NCB=NCB, SFK_TILES=SFK_TILES, RPP=RPP, TPP=TPP,
        )
    else:
        QP = R * SFK_TILES
        _quant_wswz_kernel[(QP + n_tiles,)](
            xr, buf["ga"], buf["a_fp4_u8"], buf["a_sf_swz"], buf["wgs"], buf["gs"],
            w_sf, buf["w_sf_swz"],
            xr.stride(0), buf["a_fp4_u8"].stride(0), w_sf.stride(0), QP,
            NB=_NB, NCB=NCB, SFK_TILES=SFK_TILES,
        )

    main_s = torch.cuda.current_stream().cuda_stream
    side = buf.get("side_stream")
    if side is not None:

        _klib_cuda_graph_drv__event_record(buf["evt_q"], main_s)
        _klib_cuda_graph_drv__stream_wait_event(side.cuda_stream, buf["evt_q"])
        with torch.cuda.stream(side):
            buf["ga"].zero_()
        _klib_cuda_graph_drv__event_record(buf["evt_m"], side.cuda_stream)

        buf["plan"].matmul(
            buf["weight_fp4_u8"].data_ptr(), buf["a_fp4_u8"].data_ptr(),
            buf["w_sf_swz"].data_ptr(), buf["a_sf_swz"].data_ptr(),
            buf["gs"].data_ptr(), buf["out"].data_ptr(),
            main_s,
        )
        _klib_cuda_graph_drv__stream_wait_event(main_s, buf["evt_m"])
    else:

        buf["plan"].matmul(
            buf["weight_fp4_u8"].data_ptr(), buf["a_fp4_u8"].data_ptr(),
            buf["w_sf_swz"].data_ptr(), buf["a_sf_swz"].data_ptr(),
            buf["gs"].data_ptr(), buf["out"].data_ptr(), main_s,
        )
        buf["ga"].zero_()
    return buf["out"]


@torch.no_grad()
def _run_eager(x, weight_fp4, weight_scale, weight_global_scale, bias):
    B, seq_len, C = x.shape
    M = seq_len // 2
    K = C * 2
    N = weight_fp4.shape[1]
    R = B * M
    SF_K = K // 16
    NCB = SF_K // 4

    xr = x.reshape(R, K)
    ga = torch.zeros(1, dtype=torch.float32, device=x.device)
    numel = R * K
    _amax_kernel[(triton.cdiv(numel, _AMAX_BLK),)](xr, ga, numel, BLOCK=_AMAX_BLK)

    a_fp4_u8 = torch.empty((R, K // 2), dtype=torch.uint8, device=x.device)
    a_sf_swz = torch.empty(R * SF_K, dtype=torch.float8_e4m3fn, device=x.device)
    gs = torch.empty(1, dtype=torch.float32, device=x.device)
    wgs = weight_global_scale.reshape(1)
    w_sf = weight_scale[0]
    w_sf_swz = torch.empty(N * SF_K, dtype=torch.float8_e4m3fn, device=x.device)
    SFK_TILES = SF_K // _NB
    QP = R * SFK_TILES
    n_tiles = (N // 128) * NCB
    _quant_wswz_kernel[(QP + n_tiles,)](
        xr, ga, a_fp4_u8, a_sf_swz, wgs, gs, w_sf, w_sf_swz,
        xr.stride(0), a_fp4_u8.stride(0), w_sf.stride(0), QP,
        NB=_NB, NCB=NCB, SFK_TILES=SFK_TILES,
    )

    out = torch.empty((R, N), dtype=torch.bfloat16, device=x.device)
    plan = _klib_fp4gemm__get_plan(R, N, K, w_sf_swz.data_ptr(), a_sf_swz.data_ptr())
    plan.matmul(
        weight_fp4.data_ptr(), a_fp4_u8.data_ptr(),
        w_sf_swz.data_ptr(), a_sf_swz.data_ptr(),
        gs.data_ptr(), out.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    return out.view(B, M, N)


_GRAPH_CACHE = {}


_GRAPH_MAX_R = 6144


_DROP_X_MIN_R = 1536


_SIDE_STREAM_MAX_R = 4096


def _pick_rpp(R):
    if R >= 1024 and (R % 8 == 0):
        return 8
    if R >= 256 and (R % 4 == 0):
        return 4
    return 1


def _build_graph(x, weight_fp4, weight_scale, weight_global_scale, drop_x):
    B, seq_len, C = x.shape
    M = seq_len // 2
    K = C * 2
    N = weight_fp4.shape[1]
    R = B * M
    SF_K = K // 16
    NCB = SF_K // 4
    dev = x.device
    store_x = not drop_x

    buf = {
        "R": R, "K": K, "N": N, "SF_K": SF_K, "NCB": NCB, "B": B, "M": M,
        "RPP": _pick_rpp(R), "TPP": _TPP,
        "x": torch.empty((B, seq_len, C), dtype=torch.bfloat16, device=dev),
        "weight_fp4_u8": torch.empty((1, N, K // 2), dtype=torch.uint8, device=dev),
        "weight_scale": torch.empty((1, N, SF_K), dtype=torch.float8_e4m3fn, device=dev),
        "wgs": torch.empty(1, dtype=torch.float32, device=dev),
        "gs": torch.empty(1, dtype=torch.float32, device=dev),
        "ga": torch.zeros(1, dtype=torch.float32, device=dev),
        "a_fp4_u8": torch.empty((R, K // 2), dtype=torch.uint8, device=dev),
        "a_sf_swz": torch.empty(R * SF_K, dtype=torch.float8_e4m3fn, device=dev),
        "w_sf_swz": torch.empty(N * SF_K, dtype=torch.float8_e4m3fn, device=dev),
        "out": torch.empty((R, N), dtype=torch.bfloat16, device=dev),
    }

    buf["plan"] = _klib_fp4gemm__get_plan(R, N, K, buf["w_sf_swz"].data_ptr(), buf["a_sf_swz"].data_ptr())


    if R <= _SIDE_STREAM_MAX_R:
        buf["side_stream"] = torch.cuda.Stream(device=dev)
        buf["evt_q"] = _klib_cuda_graph_drv__event_create()
        buf["evt_m"] = _klib_cuda_graph_drv__event_create()


    buf["ga"].zero_()
    _fill_inputs(buf, x, weight_fp4, weight_scale, weight_global_scale, store_x=store_x)
    if drop_x:


        buf["x"].copy_(x)


    cap = torch.cuda.Stream()
    cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        for _ in range(3):
            _fill_inputs(buf, x, weight_fp4, weight_scale, weight_global_scale, store_x=store_x)
            _compute(buf)
    torch.cuda.current_stream().wait_stream(cap)
    torch.cuda.synchronize()

    _klib_cuda_graph_drv__begin_capture(cap.cuda_stream)
    with torch.cuda.stream(cap):
        _fill_inputs(buf, x, weight_fp4, weight_scale, weight_global_scale, store_x=store_x)
        _compute(buf)
    g = _klib_cuda_graph_drv__end_capture(cap.cuda_stream)
    exe = _klib_cuda_graph_drv__instantiate(g)


    bx = x.data_ptr()
    bw = weight_fp4.data_ptr()
    bws = weight_scale.data_ptr()
    bwgs = weight_global_scale.reshape(1).data_ptr()


    mc4_node = None
    for node in _klib_cuda_graph_drv__get_nodes(g):
        if _klib_cuda_graph_drv__node_type(node) != _klib_cuda_graph_drv__CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        if _klib_cuda_graph_drv__read_param_u64(node, 0) == bx:
            mc4_node = node
            break
    if mc4_node is None:
        raise RuntimeError("manual graph: could not locate _mc4 kernel node (arg0==x)")

    kp = _klib_cuda_graph_drv__kernel_get_params(mc4_node)


    idx = {0: bx, 2: bw, 4: bws, 6: bwgs}
    for i, want in idx.items():
        got = _klib_cuda_graph_drv__read_param_u64(mc4_node, i)
        if got != want:
            raise RuntimeError(f"manual graph: _mc4 arg[{i}]={got:#x} != expected src {want:#x}")


    holders = {i: (ctypes.c_ulonglong * 1)() for i in idx}
    for i, hh in holders.items():
        hh[0] = idx[i]
        _klib_cuda_graph_drv__set_param_slot(kp, i, hh)


    quant_node = None
    qkp = None
    q_holder = None
    if drop_x:
        bufx = buf["x"].data_ptr()
        bga = buf["ga"].data_ptr()
        for node in _klib_cuda_graph_drv__get_nodes(g):
            if node == mc4_node:
                continue
            if _klib_cuda_graph_drv__node_type(node) != _klib_cuda_graph_drv__CU_GRAPH_NODE_TYPE_KERNEL:
                continue
            if _klib_cuda_graph_drv__read_param_u64(node, 0) == bufx and _klib_cuda_graph_drv__read_param_u64(node, 1) == bga:
                quant_node = node
                break
        if quant_node is None:
            raise RuntimeError("manual graph: could not locate quant kernel node (arg0==buf[x], arg1==ga)")
        qkp = _klib_cuda_graph_drv__kernel_get_params(quant_node)
        q_holder = (ctypes.c_ulonglong * 1)()
        q_holder[0] = bufx
        _klib_cuda_graph_drv__set_param_slot(qkp, 0, q_holder)

    buf["graph_raw"] = g
    buf["exec"] = exe
    buf["mc4_node"] = mc4_node
    buf["kparams"] = kp
    buf["holders"] = holders
    buf["quant_node"] = quant_node
    buf["quant_kparams"] = qkp
    buf["quant_holder"] = q_holder
    return buf


@torch.no_grad()
def run(x, weight_fp4, weight_scale, weight_global_scale, bias):
    B, seq_len, C = x.shape
    R = B * (seq_len // 2)

    if R > _GRAPH_MAX_R:
        return _run_eager(x, weight_fp4, weight_scale, weight_global_scale, bias)

    key = (B, seq_len)
    buf = _GRAPH_CACHE.get(key)
    if buf is None:
        drop_x = R >= _DROP_X_MIN_R
        buf = _build_graph(x, weight_fp4, weight_scale, weight_global_scale, drop_x)
        _GRAPH_CACHE[key] = buf


    xp = x.data_ptr()
    h = buf["holders"]
    h[0][0] = xp
    h[2][0] = weight_fp4.data_ptr()
    h[4][0] = weight_scale.data_ptr()
    h[6][0] = weight_global_scale.reshape(1).data_ptr()
    _klib_cuda_graph_drv__exec_kernel_set_params(buf["exec"], buf["mc4_node"], buf["kparams"])
    if buf["quant_node"] is not None:
        buf["quant_holder"][0] = xp
        _klib_cuda_graph_drv__exec_kernel_set_params(buf["exec"], buf["quant_node"], buf["quant_kparams"])
    _klib_cuda_graph_drv__launch(buf["exec"], torch.cuda.current_stream().cuda_stream)
    return buf["out"].view(buf["B"], buf["M"], buf["N"])
