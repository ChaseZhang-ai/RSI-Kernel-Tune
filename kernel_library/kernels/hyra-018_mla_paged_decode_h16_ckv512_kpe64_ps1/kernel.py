"""Merged by forgekernel.klib.normalize — do not edit; edit the original sources.

Inlined, in dependency order: cuda_mla.py, cuda_mma.py
Entry file (verbatim last): kernel.py
Every inlined module's top-level bindings are prefixed `_klib_<module>__` and all references
to them are rewritten, so side-by-side definitions cannot shadow each other.
"""

# ---------------------------------------------------------------- inlined: cuda_mla.py
"""Hand-written CUDA (NVRTC via cupy) flash-decode for the bs=1 MLA regime.

Why: the operator is deeply LATENCY/launch-bound (every workload sits below the
HBM roofline). bs=1 is the highest score-ROI regime (dS/dmicros ~ 0.015) and its
reduction is PER-HEAD-INDEPENDENT. The KV cache is huge (~1GB) and gathered at
RANDOM pages -> each token is an HBM-latency miss (not L2-resident), so the win is
having MANY CTAs in flight (hide latency across SMs), not fewer launches.

Two paths (both use the FIXED-SHIFT trick: sm_scale is constant, so
w_t = 2^(dot_t*scale2 - SHIFT); accumulate l+=w_t, acc+=w_t*Kc as a PLAIN SUM ->
tokens independent -> full ILP; combine is a plain sum. Softmax is shift-invariant
so this is mathematically exact. Validated safe for bs=1 by the champion's
`_mla_atomic_combine` (same MSHIFT=16)):

  * TINY nnz  -> single-launch `mla_bs1`: grid=(H,), one CTA/head, NWARP warps
    split the tokens intra-CTA, NO cross-CTA combine. Beats the champion's fused
    tiny path (nnz=8: 4.9us vs 7.3us).
  * LARGER nnz -> token-split: `mla_bs1_partial` grid=(H, NSPLIT) computes per-CTA
    partial (l, acc) over a token chunk into scratch; `mla_bs1_combine` grid=(H,)
    plain-sums the NSPLIT partials -> final out/lse. Many CTAs => hides the random
    gather latency the single-launch (16 CTAs) could not.

bf16<->f32 done manually (bit ops) -> no cuda_bf16.h dependency for NVRTC.
"""
import numpy as _klib_cuda_mla__np
import torch as _klib_cuda_mla__torch

_klib_cuda_mla__LOG2E = 1.4426950408889634


_klib_cuda_mla___NWARP_P = 4
_klib_cuda_mla___NSPLIT_MAX = 64
_klib_cuda_mla___CHUNK_TARGET = 32
_klib_cuda_mla___TINY_MAX = 64

_klib_cuda_mla___COMMON = r"""
#define DCKV 512
#define DPE  64
#define NEG_INF __uint_as_float(0xff800000u)

__device__ __forceinline__ float bf2f(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;              // round-to-nearest-even
    return (unsigned short)(u >> 16);
}
"""


_klib_cuda_mla___SRC_SINGLE = _klib_cuda_mla___COMMON + r"""
#define NWARP NW_PLACEHOLDER
extern "C" __global__ void mla_bs1(
    const unsigned short* __restrict__ q_nope,
    const unsigned short* __restrict__ q_pe,
    const unsigned short* __restrict__ ckv,
    const unsigned short* __restrict__ kpe,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    float scale2, float shift,
    int sq1, int sp1, int so1, int slse1, int ckv_s0, int kpe_s0)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float sm_l[NWARP];
    __shared__ float sm_acc[NWARP * DCKV];

    const int page_beg = kv_indptr[0];
    const int L = kv_indptr[1] - page_beg;

    float qn_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) qn_reg[i] = bf2f(q_nope[h * sq1 + lane + 32 * i]);
    float qp0 = bf2f(q_pe[h * sp1 + lane]);
    float qp1 = bf2f(q_pe[h * sp1 + lane + 32]);

    const int tpw = (L + NWARP - 1) / NWARP;
    int t0 = warp_id * tpw;
    int t1 = t0 + tpw; if (t1 > L) t1 = L;

    float l_w = 0.f;
    float acc_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc_reg[i] = 0.f;

    #pragma unroll 4
    for (int t = t0; t < t1; ++t) {
        int page = kv_indices[page_beg + t];
        const unsigned short* Kc = ckv + (long)page * ckv_s0;
        const unsigned short* Kp = kpe + (long)page * kpe_s0;
        float kc_reg[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) kc_reg[i] = bf2f(Kc[lane + 32 * i]);
        float partial = qp0 * bf2f(Kp[lane]) + qp1 * bf2f(Kp[lane + 32]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) partial += qn_reg[i] * kc_reg[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        float w = exp2f(partial * scale2 - shift);
        l_w += w;
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc_reg[i] += w * kc_reg[i];
    }

    if (lane == 0) sm_l[warp_id] = l_w;
    #pragma unroll
    for (int i = 0; i < 16; ++i) sm_acc[warp_id * DCKV + lane + 32 * i] = acc_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float ltot = 0.f;
        #pragma unroll
        for (int w = 0; w < NWARP; ++w) ltot += sm_l[w];
        float inv = (ltot > 0.f) ? (1.f / ltot) : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int e = lane + 32 * i;
            float o = 0.f;
            #pragma unroll
            for (int w = 0; w < NWARP; ++w) o += sm_acc[w * DCKV + e];
            out[h * so1 + e] = f2bf(o * inv);
        }
        if (lane == 0) lse[h * slse1] = (ltot > 0.f) ? (shift + log2f(ltot)) : NEG_INF;
    }
}
"""


_klib_cuda_mla___SRC_SPLIT = _klib_cuda_mla___COMMON + r"""
#define NWARP NWP_PLACEHOLDER
extern "C" __global__ void mla_bs1_partial(
    const unsigned short* __restrict__ q_nope,
    const unsigned short* __restrict__ q_pe,
    const unsigned short* __restrict__ ckv,
    const unsigned short* __restrict__ kpe,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float* __restrict__ pl,          // [H*NSPLIT]
    float* __restrict__ pa,          // [H*NSPLIT*DCKV]
    float scale2, float shift, int NSPLIT,
    int sq1, int sp1, int ckv_s0, int kpe_s0)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float sm_l[NWARP];
    __shared__ float sm_acc[NWARP * DCKV];

    const int page_beg = kv_indptr[0];
    const int L = kv_indptr[1] - page_beg;

    // this split's contiguous token chunk, then split across warps
    const int chunk = (L + NSPLIT - 1) / NSPLIT;
    int cs = s * chunk;
    int ce = cs + chunk; if (ce > L) ce = L;

    float qn_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) qn_reg[i] = bf2f(q_nope[h * sq1 + lane + 32 * i]);
    float qp0 = bf2f(q_pe[h * sp1 + lane]);
    float qp1 = bf2f(q_pe[h * sp1 + lane + 32]);

    const int span = ce - cs;
    const int tpw = (span + NWARP - 1) / NWARP;
    int t0 = cs + warp_id * tpw;
    int t1 = t0 + tpw; if (t1 > ce) t1 = ce;
    if (t0 > ce) t0 = ce;

    float l_w = 0.f;
    float acc_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc_reg[i] = 0.f;

    #pragma unroll 4
    for (int t = t0; t < t1; ++t) {
        int page = kv_indices[page_beg + t];
        const unsigned short* Kc = ckv + (long)page * ckv_s0;
        const unsigned short* Kp = kpe + (long)page * kpe_s0;
        float kc_reg[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) kc_reg[i] = bf2f(Kc[lane + 32 * i]);
        float partial = qp0 * bf2f(Kp[lane]) + qp1 * bf2f(Kp[lane + 32]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) partial += qn_reg[i] * kc_reg[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        float w = exp2f(partial * scale2 - shift);
        l_w += w;
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc_reg[i] += w * kc_reg[i];
    }

    if (lane == 0) sm_l[warp_id] = l_w;
    #pragma unroll
    for (int i = 0; i < 16; ++i) sm_acc[warp_id * DCKV + lane + 32 * i] = acc_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float ltot = 0.f;
        #pragma unroll
        for (int w = 0; w < NWARP; ++w) ltot += sm_l[w];
        long base = ((long)h * NSPLIT + s) * DCKV;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int e = lane + 32 * i;
            float o = 0.f;
            #pragma unroll
            for (int w = 0; w < NWARP; ++w) o += sm_acc[w * DCKV + e];
            pa[base + e] = o;
        }
        if (lane == 0) pl[(long)h * NSPLIT + s] = ltot;
    }
}

extern "C" __global__ void mla_bs1_combine(
    const float* __restrict__ pl,
    const float* __restrict__ pa,
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    float shift, int NSPLIT,
    int so1, int slse1)
{
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;   // block = 32

    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
    float ltot = 0.f;
    for (int s = 0; s < NSPLIT; ++s) {
        ltot += pl[(long)h * NSPLIT + s];
        long base = ((long)h * NSPLIT + s) * DCKV;
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += pa[base + lane + 32 * i];
    }
    float inv = (ltot > 0.f) ? (1.f / ltot) : 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[h * so1 + lane + 32 * i] = f2bf(acc[i] * inv);
    if (lane == 0) lse[h * slse1] = (ltot > 0.f) ? (shift + log2f(ltot)) : NEG_INF;
}
"""

_klib_cuda_mla___MODS_SINGLE = {}
_klib_cuda_mla___MOD_SPLIT = None
_klib_cuda_mla___SCRATCH = {}


def _klib_cuda_mla___get_single(nw):
    if nw not in _klib_cuda_mla___MODS_SINGLE:
        import cupy as cp
        src = _klib_cuda_mla___SRC_SINGLE.replace("NW_PLACEHOLDER", str(nw))
        mod = cp.RawModule(code=src, options=("--std=c++14",), backend="nvrtc")
        _klib_cuda_mla___MODS_SINGLE[nw] = mod.get_function("mla_bs1")
    return _klib_cuda_mla___MODS_SINGLE[nw]


def _klib_cuda_mla___get_split():
    global _klib_cuda_mla___MOD_SPLIT
    if _klib_cuda_mla___MOD_SPLIT is None:
        import cupy as cp
        src = _klib_cuda_mla___SRC_SPLIT.replace("NWP_PLACEHOLDER", str(_klib_cuda_mla___NWARP_P))
        mod = cp.RawModule(code=src, options=("--std=c++14",), backend="nvrtc")
        _klib_cuda_mla___MOD_SPLIT = (mod.get_function("mla_bs1_partial"),
                      mod.get_function("mla_bs1_combine"))
    return _klib_cuda_mla___MOD_SPLIT


def _klib_cuda_mla___get_scratch(H, device):
    key = (H, device)
    if key not in _klib_cuda_mla___SCRATCH:
        pl = _klib_cuda_mla__torch.empty((H * _klib_cuda_mla___NSPLIT_MAX,), dtype=_klib_cuda_mla__torch.float32, device=device)
        pa = _klib_cuda_mla__torch.empty((H * _klib_cuda_mla___NSPLIT_MAX * 512,), dtype=_klib_cuda_mla__torch.float32, device=device)
        _klib_cuda_mla___SCRATCH[key] = (pl, pa)
    return _klib_cuda_mla___SCRATCH[key]


def _klib_cuda_mla___wrap(t, dt):
    import cupy as cp
    nbytes = t.numel() * t.element_size()
    mem = cp.cuda.UnownedMemory(t.data_ptr(), nbytes, owner=t)
    return cp.ndarray(t.shape, dtype=dt, memptr=cp.cuda.MemoryPointer(mem, 0))


def _klib_cuda_mla___pick_nsplit(nnz):
    ns = (nnz + _klib_cuda_mla___CHUNK_TARGET - 1) // _klib_cuda_mla___CHUNK_TARGET
    if ns < 2:
        ns = 2
    if ns > _klib_cuda_mla___NSPLIT_MAX:
        ns = _klib_cuda_mla___NSPLIT_MAX
    return ns


def _klib_cuda_mla__mla_decode_bs1(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
                   out, lse, scale2, shift=16.0):
    """bs=1 only. ckv/kpe are the reshaped [-1, D] / [-1, DPE] views."""
    import cupy as cp
    H = q_nope.shape[1]
    nnz = kv_indices.shape[0]
    stream = cp.cuda.ExternalStream(_klib_cuda_mla__torch.cuda.current_stream().cuda_stream)

    if nnz <= _klib_cuda_mla___TINY_MAX:
        nw = 8
        ker = _klib_cuda_mla___get_single(nw)
        args = (
            _klib_cuda_mla___wrap(q_nope, cp.uint16), _klib_cuda_mla___wrap(q_pe, cp.uint16),
            _klib_cuda_mla___wrap(ckv, cp.uint16), _klib_cuda_mla___wrap(kpe, cp.uint16),
            _klib_cuda_mla___wrap(kv_indptr, cp.int32), _klib_cuda_mla___wrap(kv_indices, cp.int32),
            _klib_cuda_mla___wrap(out, cp.uint16), _klib_cuda_mla___wrap(lse, cp.float32),
            _klib_cuda_mla__np.float32(scale2), _klib_cuda_mla__np.float32(shift),
            _klib_cuda_mla__np.int32(q_nope.stride(1)), _klib_cuda_mla__np.int32(q_pe.stride(1)),
            _klib_cuda_mla__np.int32(out.stride(1)), _klib_cuda_mla__np.int32(lse.stride(1)),
            _klib_cuda_mla__np.int32(ckv.stride(0)), _klib_cuda_mla__np.int32(kpe.stride(0)),
        )
        with stream:
            ker((H,), (32 * nw,), args)
        return


    nsplit = _klib_cuda_mla___pick_nsplit(nnz)
    partial, combine = _klib_cuda_mla___get_split()
    pl, pa = _klib_cuda_mla___get_scratch(H, q_nope.device)
    p_args = (
        _klib_cuda_mla___wrap(q_nope, cp.uint16), _klib_cuda_mla___wrap(q_pe, cp.uint16),
        _klib_cuda_mla___wrap(ckv, cp.uint16), _klib_cuda_mla___wrap(kpe, cp.uint16),
        _klib_cuda_mla___wrap(kv_indptr, cp.int32), _klib_cuda_mla___wrap(kv_indices, cp.int32),
        _klib_cuda_mla___wrap(pl, cp.float32), _klib_cuda_mla___wrap(pa, cp.float32),
        _klib_cuda_mla__np.float32(scale2), _klib_cuda_mla__np.float32(shift), _klib_cuda_mla__np.int32(nsplit),
        _klib_cuda_mla__np.int32(q_nope.stride(1)), _klib_cuda_mla__np.int32(q_pe.stride(1)),
        _klib_cuda_mla__np.int32(ckv.stride(0)), _klib_cuda_mla__np.int32(kpe.stride(0)),
    )
    c_args = (
        _klib_cuda_mla___wrap(pl, cp.float32), _klib_cuda_mla___wrap(pa, cp.float32),
        _klib_cuda_mla___wrap(out, cp.uint16), _klib_cuda_mla___wrap(lse, cp.float32),
        _klib_cuda_mla__np.float32(shift), _klib_cuda_mla__np.int32(nsplit),
        _klib_cuda_mla__np.int32(out.stride(1)), _klib_cuda_mla__np.int32(lse.stride(1)),
    )
    with stream:
        partial((H, nsplit), (32 * _klib_cuda_mla___NWARP_P,), p_args)
        combine((H,), (32,), c_args)


_klib_cuda_mla___SRC_AH = _klib_cuda_mla___COMMON + r"""
extern "C" __global__ void mla_bs1_ah(
    const unsigned short* __restrict__ q_nope,
    const unsigned short* __restrict__ q_pe,
    const unsigned short* __restrict__ ckv,
    const unsigned short* __restrict__ kpe,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float* __restrict__ acc_g,   // [H*DCKV] persistent scratch
    float* __restrict__ l_g,     // [H]      persistent scratch
    int*   __restrict__ cnt,     // [1]      persistent counter (pre-zeroed, self-reset)
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    float scale2, float shift, int NSPLIT, int CHUNK,
    int sq1, int sp1, int so1, int slse1, int ckv_s0, int kpe_s0)
{
    extern __shared__ unsigned short s_mem[];        // s_kc[CHUNK*512] then s_kp[CHUNK*64]
    unsigned short* s_kc = s_mem;
    unsigned short* s_kp = s_mem + CHUNK * DCKV;

    const int tid = threadIdx.x;
    const int h = tid >> 5;          // one warp per head
    const int lane = tid & 31;
    const int s = blockIdx.x;

    const int page_beg = kv_indptr[0];
    const int L = kv_indptr[1] - page_beg;
    int cs = s * CHUNK;
    int ce = cs + CHUNK; if (ce > L) ce = L;
    const int nc = ce - cs;          // tokens this CTA handles (<= CHUNK)

    // ---- stage this chunk's Kc/Kp into shared memory: ONE coalesced HBM load
    // per token (thread tid loads element tid of each token's 512-vec) ----
    #pragma unroll 4
    for (int j = 0; j < nc; ++j) {
        int page = kv_indices[page_beg + cs + j];
        s_kc[j * DCKV + tid] = ckv[(long)page * ckv_s0 + tid];
        if (tid < DPE) s_kp[j * DPE + tid] = kpe[(long)page * kpe_s0 + tid];
    }
    __syncthreads();

    float qn_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) qn_reg[i] = bf2f(q_nope[h * sq1 + lane + 32 * i]);
    float qp0 = bf2f(q_pe[h * sp1 + lane]);
    float qp1 = bf2f(q_pe[h * sp1 + lane + 32]);

    float l_w = 0.f;
    float acc_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc_reg[i] = 0.f;

    for (int j = 0; j < nc; ++j) {
        float kc_reg[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) kc_reg[i] = bf2f(s_kc[j * DCKV + lane + 32 * i]);
        float partial = qp0 * bf2f(s_kp[j * DPE + lane]) + qp1 * bf2f(s_kp[j * DPE + lane + 32]);
        #pragma unroll
        for (int i = 0; i < 16; ++i) partial += qn_reg[i] * kc_reg[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        float w = exp2f(fminf(partial * scale2 - shift, 60.0f));
        l_w += w;
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc_reg[i] += w * kc_reg[i];
    }

    if (NSPLIT == 1) {
        float inv = (l_w > 0.f) ? (1.f / l_w) : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) out[h * so1 + lane + 32 * i] = f2bf(acc_reg[i] * inv);
        if (lane == 0) lse[h * slse1] = (l_w > 0.f) ? (shift + log2f(l_w)) : NEG_INF;
        return;
    }

    // ---- token-split: atomic-reduce this CTA's partial into persistent scratch ----
    if (lane == 0) atomicAdd(&l_g[h], l_w);
    #pragma unroll
    for (int i = 0; i < 16; ++i) atomicAdd(&acc_g[h * DCKV + lane + 32 * i], acc_reg[i]);

    __threadfence();
    __shared__ int s_last;
    if (tid == 0) {
        int old = atomicAdd(cnt, 1);
        s_last = (old == NSPLIT - 1) ? 1 : 0;
    }
    __syncthreads();
    if (s_last) {
        float ltot = l_g[h];
        float inv = (ltot > 0.f) ? (1.f / ltot) : 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int e = lane + 32 * i;
            out[h * so1 + e] = f2bf(acc_g[h * DCKV + e] * inv);
            acc_g[h * DCKV + e] = 0.f;      // reset scratch for next call
        }
        if (lane == 0) {
            lse[h * slse1] = (ltot > 0.f) ? (shift + log2f(ltot)) : NEG_INF;
            l_g[h] = 0.f;
        }
        if (tid == 0) *cnt = 0;
    }
}
"""

_klib_cuda_mla___MOD_AH = None
_klib_cuda_mla___AH_SCRATCH = {}


_klib_cuda_mla___AH_CHUNK_SMALL = 16
_klib_cuda_mla___AH_CHUNK_LARGE = 32
_klib_cuda_mla___AH_SMALL_NNZ = 1000


def _klib_cuda_mla___get_ah():
    global _klib_cuda_mla___MOD_AH
    if _klib_cuda_mla___MOD_AH is None:
        import cupy as cp
        mod = cp.RawModule(code=_klib_cuda_mla___SRC_AH, options=("--std=c++14",), backend="nvrtc")
        _klib_cuda_mla___MOD_AH = mod.get_function("mla_bs1_ah")
    return _klib_cuda_mla___MOD_AH


def _klib_cuda_mla___get_ah_scratch(H, D, device):
    key = (H, D, device)
    if key not in _klib_cuda_mla___AH_SCRATCH:
        acc_g = _klib_cuda_mla__torch.zeros(H * D, dtype=_klib_cuda_mla__torch.float32, device=device)
        l_g = _klib_cuda_mla__torch.zeros(H, dtype=_klib_cuda_mla__torch.float32, device=device)
        cnt = _klib_cuda_mla__torch.zeros(1, dtype=_klib_cuda_mla__torch.int32, device=device)
        _klib_cuda_mla___AH_SCRATCH[key] = (acc_g, l_g, cnt)
    return _klib_cuda_mla___AH_SCRATCH[key]


def _klib_cuda_mla__mla_decode_bs1_ah(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
                      out, lse, scale2, shift=16.0, chunk=None):
    """bs=1 all-heads token-split atomic, Kc/Kp staged in shared memory."""
    import cupy as cp
    H = q_nope.shape[1]
    D = ckv.shape[1]
    DPE = kpe.shape[1]
    nnz = kv_indices.shape[0]
    if chunk is None:
        chunk = _klib_cuda_mla___AH_CHUNK_SMALL if nnz < _klib_cuda_mla___AH_SMALL_NNZ else _klib_cuda_mla___AH_CHUNK_LARGE
    nsplit = (nnz + chunk - 1) // chunk
    if nsplit < 1:
        nsplit = 1
    ker = _klib_cuda_mla___get_ah()
    acc_g, l_g, cnt = _klib_cuda_mla___get_ah_scratch(H, D, q_nope.device)
    smem_bytes = chunk * (D + DPE) * 2
    stream = cp.cuda.ExternalStream(_klib_cuda_mla__torch.cuda.current_stream().cuda_stream)
    args = (
        _klib_cuda_mla___wrap(q_nope, cp.uint16), _klib_cuda_mla___wrap(q_pe, cp.uint16),
        _klib_cuda_mla___wrap(ckv, cp.uint16), _klib_cuda_mla___wrap(kpe, cp.uint16),
        _klib_cuda_mla___wrap(kv_indptr, cp.int32), _klib_cuda_mla___wrap(kv_indices, cp.int32),
        _klib_cuda_mla___wrap(acc_g, cp.float32), _klib_cuda_mla___wrap(l_g, cp.float32), _klib_cuda_mla___wrap(cnt, cp.int32),
        _klib_cuda_mla___wrap(out, cp.uint16), _klib_cuda_mla___wrap(lse, cp.float32),
        _klib_cuda_mla__np.float32(scale2), _klib_cuda_mla__np.float32(shift), _klib_cuda_mla__np.int32(nsplit), _klib_cuda_mla__np.int32(chunk),
        _klib_cuda_mla__np.int32(q_nope.stride(1)), _klib_cuda_mla__np.int32(q_pe.stride(1)),
        _klib_cuda_mla__np.int32(out.stride(1)), _klib_cuda_mla__np.int32(lse.stride(1)),
        _klib_cuda_mla__np.int32(ckv.stride(0)), _klib_cuda_mla__np.int32(kpe.stride(0)),
    )
    with stream:
        ker((nsplit,), (32 * H,), args, shared_mem=smem_bytes)


# ---------------------------------------------------------------- inlined: cuda_mma.py
"""Single-launch TENSOR-CORE (WMMA m16n16k16 bf16) flash-decode for the B=1 MLA regime.

===========================================================================================
!!! MEASURED DEAD-END (this node, validate 0418) - CORRECT but ~19x TOO SLOW. GATED OFF. !!!
This kernel COMPILES via nvrtc on sm_100 and is NUMERICALLY CORRECT (all 47 workloads pass,
B=1 max_abs_err 1.56e-2 == champion). But it runs ~181us for B=1 vs the champion Triton
atomic path's 9.3-11.2us. Per-kernel profile: `mla_bs1_wmma` med 181us, FLAT across nnz
(7 CTAs @ nnz=108 and 170 CTAs @ nnz=2708 both ~181us => per-CTA FIXED cost, not atomic
contention). The scalar `mla_bs1_ah` (same 8192 atomics/CTA + smem staging) was only 22us,
so the ~159us excess is WMMA-OP-BOUND: 68 mma_sync + 68 load_matrix_sync per CTA (36 QK
k-tiles + 32 AV d-tiles) at ~2.3us EACH. Native mma is ~16ns, so ~2.3us/op => legacy WMMA
m16n16k16 is emulated / not on the fast tensor-core path on Blackwell (which uses tcgen05),
AND block=32 (1 warp/CTA) hides zero latency on the DEPENDENT 36-mma QK accumulation chain.
NEXT AGENT: do NOT reuse the WMMA C++ API here. The tensor-core lever needs raw
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` inline PTX (control the fragment
layout directly), 4+ warps/CTA running INDEPENDENT token-tiles (overlap the mma latency),
and ONE coalesced [16,512] atomicAdd per CTA (float4 red.global.add), not 32 per-d-tile
store+atomic loops. Kept wired-but-gated (_USE_MMA_B1=False in kernel.py) as a correct,
validated reference for the plumbing (cp view/stream, fixed-shift, threadfence finalize).
===========================================================================================


THE identified lever (NOTES_NEXT_LEVER.md): B=1 is the highest score-ROI regime and is
pure overhead-bound (KV L2-resident, ns of real compute). The champion's Triton atomic
path sits at ~9.3-11.2us for nnz>=108 -- above CUDA's ~4.5us single-launch floor because
Triton's per-kernel floor is ~7us (measured: nnz=8 Triton fused 7.3us vs CUDA 4.5us).
FOUR independent nodes proved a SCALAR warp-shuffle dot over 16 heads cannot match one
tensor-core op that does all 16 heads at once. So this kernel does the 16-head QK and the
softmax*V with WMMA (m16n16k16, bf16 in / f32 acc) in a SINGLE launch:

  grid=(NSPLIT,), block=32 (1 warp). Each CTA owns a CHUNK=16-token tile (one WMMA N-tile):
   * stage the tile's gathered Kc[16,512]+Kp[16,64] into shared mem (Kcat[16,576]);
   * QK: C[16,16] = q_nope[16,512] @ Kc[16,512]^T + q_pe[16,64] @ Kp[16,64]^T
        (A-frags loaded straight from q_nope/q_pe global; B-frags col_major from smem Kcat);
   * FIXED-SHIFT (sm_scale const): w = exp2(C*scale2 - MSHIFT)  -> plain sum, atomic-able;
   * mask padding tokens (last CTA) to w=0; l[h] += row-sum(w);
   * AV: for each of 32 d-tiles, acc[16,16] = w[16,16] @ Kc[16, dt*16:+16]  (WMMA), atomicAdd
        into persistent fp32 scratch acc_g[16,512]; l atomicAdd into l_g[16];
   * threadfence + counter: the LAST CTA finalizes out=acc_g/l_g (bf16 RNE), lse=MSHIFT+log2(l_g),
        and RESETS the scratch (stream-ordered before the next call) -> truly ONE launch.

bf16<->f32 done via bit ops for the final store; WMMA handles the matmul conversions.
"""
import numpy as _klib_cuda_mma__np
import torch as _klib_cuda_mma__torch

_klib_cuda_mma__LOG2E = 1.4426950408889634


_klib_cuda_mma___MMA_SRC = r"""
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#define DCKV 512
#define DPE  64
#define DTOT 576
#define H    16
#define TILE 16
#define NDT  32          // 512 / 16 d-tiles for AV
#define NKT_C 32         // 512 / 16 k-tiles for QK (ckv)
#define NKT_P 4          // 64  / 16 k-tiles for QK (kpe)

__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (unsigned short)(u >> 16);
}

extern "C" __global__ void mla_bs1_wmma(
    const unsigned short* __restrict__ q_nope,   // [H, DCKV] bf16 bits
    const unsigned short* __restrict__ q_pe,     // [H, DPE]  bf16 bits
    const unsigned short* __restrict__ ckv,      // [num_pages, DCKV]
    const unsigned short* __restrict__ kpe,      // [num_pages, DPE]
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float* __restrict__ acc_g,   // [H*DCKV] persistent scratch
    float* __restrict__ l_g,     // [H]      persistent scratch
    int*   __restrict__ cnt,     // [1]      persistent counter (self-reset)
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    float scale2, float shift, int NSPLIT,
    int sq1, int sp1, int so1, int slse1, int ckv_s0, int kpe_s0)
{
    const int lane = threadIdx.x;            // 0..31 (single warp)
    const int s = blockIdx.x;

    __shared__ __align__(16) unsigned short s_k[TILE * DTOT];   // gathered Kcat[16,576]
    __shared__ __align__(16) float s_w[TILE * TILE];            // logits/weights [16,16] row-major
    __shared__ __align__(16) __nv_bfloat16 s_wb[TILE * TILE];   // bf16 weights for AV A-frag
    __shared__ __align__(16) float s_at[TILE * TILE];           // AV acc store buffer [16,16]

    const int page_beg = kv_indptr[0];
    const int L = kv_indptr[1] - page_beg;
    const int cs = s * TILE;
    int nc = L - cs; if (nc > TILE) nc = TILE; if (nc < 0) nc = 0;

    // ---- stage this tile's Kc/Kp into shared mem (Kcat[16,576]) ----
    for (int j = 0; j < TILE; ++j) {
        if (j < nc) {
            int page = kv_indices[page_beg + cs + j];
            const unsigned short* Kc = ckv + (long)page * ckv_s0;
            const unsigned short* Kp = kpe + (long)page * kpe_s0;
            for (int d = lane; d < DCKV; d += 32) s_k[j * DTOT + d] = Kc[d];
            for (int d = lane; d < DPE;  d += 32) s_k[j * DTOT + DCKV + d] = Kp[d];
        } else {
            for (int d = lane; d < DTOT; d += 32) s_k[j * DTOT + d] = 0;
        }
    }
    __syncwarp();

    // ---- QK: C[16,16] = q_nope @ Kc^T + q_pe @ Kp^T (WMMA) ----
    const __nv_bfloat16* qn = reinterpret_cast<const __nv_bfloat16*>(q_nope);
    const __nv_bfloat16* qp = reinterpret_cast<const __nv_bfloat16*>(q_pe);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(s_k);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int kt = 0; kt < NKT_C; ++kt) {
        wmma::load_matrix_sync(a_frag, qn + kt * 16, sq1);          // q_nope[16, kt*16], ldm=sq1
        wmma::load_matrix_sync(b_frag, kb + kt * 16, DTOT);         // Kc col_major, ldm=576
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    #pragma unroll
    for (int kt = 0; kt < NKT_P; ++kt) {
        wmma::load_matrix_sync(a_frag, qp + kt * 16, sp1);          // q_pe[16, kt*16], ldm=sp1
        wmma::load_matrix_sync(b_frag, kb + DCKV + kt * 16, DTOT);  // Kp col_major, ldm=576
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ---- fixed-shift weights (elementwise on the accumulator fragment) ----
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i)
        c_frag.x[i] = exp2f(fminf(c_frag.x[i] * scale2 - shift, 60.0f));
    wmma::store_matrix_sync(s_w, c_frag, TILE, wmma::mem_row_major);   // s_w[m*16+n] = w
    __syncwarp();

    // mask padding tokens (n >= nc) to 0, build bf16 copy for AV, accumulate l per head
    for (int idx = lane; idx < TILE * TILE; idx += 32) {
        int n = idx & 15;
        float wv = (n < nc) ? s_w[idx] : 0.0f;
        s_w[idx] = wv;
        s_wb[idx] = __float2bfloat16(wv);
    }
    __syncwarp();
    // l_local[h] = sum_n s_w[h,n]  (lane h<16 sums its row)
    if (lane < H) {
        float lsum = 0.0f;
        #pragma unroll
        for (int n = 0; n < TILE; ++n) lsum += s_w[lane * TILE + n];
        atomicAdd(&l_g[lane], lsum);
    }

    // ---- AV: acc[16,512] += w[16,16] @ Kc[16, :]  (WMMA over 32 d-tiles) ----
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> wa_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> kd_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> av_frag;
    wmma::load_matrix_sync(wa_frag, s_wb, TILE);   // w[16,16] row_major, ldm=16
    #pragma unroll 4
    for (int dt = 0; dt < NDT; ++dt) {
        wmma::fill_fragment(av_frag, 0.0f);
        wmma::load_matrix_sync(kd_frag, kb + dt * 16, DTOT);  // Kc[16 tok, dt*16:+16] row_major ldm=576
        wmma::mma_sync(av_frag, wa_frag, kd_frag, av_frag);
        wmma::store_matrix_sync(s_at, av_frag, TILE, wmma::mem_row_major);  // [16,16] -> [h, d_local]
        __syncwarp();
        for (int idx = lane; idx < TILE * TILE; idx += 32) {
            int h = idx >> 4;         // 0..15
            int dloc = idx & 15;      // 0..15
            atomicAdd(&acc_g[h * DCKV + dt * 16 + dloc], s_at[idx]);
        }
        __syncwarp();
    }

    // ---- single-launch finalize: last CTA writes out/lse and resets scratch ----
    __threadfence();
    __shared__ int s_last;
    if (lane == 0) {
        int old = atomicAdd(cnt, 1);
        s_last = (old == NSPLIT - 1) ? 1 : 0;
    }
    __syncwarp();
    if (s_last) {
        for (int e = lane; e < H * DCKV; e += 32) {
            int h = e / DCKV;
            float ltot = l_g[h];
            float inv = (ltot > 0.0f) ? (1.0f / ltot) : 0.0f;
            out[h * so1 + (e % DCKV)] = f2bf(acc_g[e] * inv);
            acc_g[e] = 0.0f;          // reset for next call
        }
        if (lane < H) {
            float ltot = l_g[lane];
            lse[lane * slse1] = (ltot > 0.0f) ? (shift + log2f(ltot)) : __uint_as_float(0xff800000u);
            l_g[lane] = 0.0f;
        }
        if (lane == 0) *cnt = 0;
    }
}
"""

_klib_cuda_mma___MOD = None
_klib_cuda_mma___SCRATCH = {}


def _klib_cuda_mma___get_kernel():
    global _klib_cuda_mma___MOD
    if _klib_cuda_mma___MOD is None:
        import cupy as cp
        mod = cp.RawModule(code=_klib_cuda_mma___MMA_SRC, backend="nvrtc",
                           options=("--std=c++17",))
        _klib_cuda_mma___MOD = mod.get_function("mla_bs1_wmma")
    return _klib_cuda_mma___MOD


def _klib_cuda_mma___get_scratch(H, D, device):
    key = (H, D, device)
    if key not in _klib_cuda_mma___SCRATCH:
        acc_g = _klib_cuda_mma__torch.zeros(H * D, dtype=_klib_cuda_mma__torch.float32, device=device)
        l_g = _klib_cuda_mma__torch.zeros(H, dtype=_klib_cuda_mma__torch.float32, device=device)
        cnt = _klib_cuda_mma__torch.zeros(1, dtype=_klib_cuda_mma__torch.int32, device=device)
        _klib_cuda_mma___SCRATCH[key] = (acc_g, l_g, cnt)
    return _klib_cuda_mma___SCRATCH[key]


def _klib_cuda_mma___wrap(t, dt):
    import cupy as cp
    nbytes = t.numel() * t.element_size()
    mem = cp.cuda.UnownedMemory(t.data_ptr(), nbytes, owner=t)
    return cp.ndarray(t.shape, dtype=dt, memptr=cp.cuda.MemoryPointer(mem, 0))


def _klib_cuda_mma__mla_decode_bs1_wmma(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
                        out, lse, scale2, shift=16.0):
    """B=1 WMMA token-split decode. ckv/kpe are reshaped [-1,D]/[-1,DPE] views."""
    import cupy as cp
    H = q_nope.shape[1]
    D = ckv.shape[1]
    nnz = kv_indices.shape[0]
    nsplit = (nnz + 15) // 16
    if nsplit < 1:
        nsplit = 1
    ker = _klib_cuda_mma___get_kernel()
    acc_g, l_g, cnt = _klib_cuda_mma___get_scratch(H, D, q_nope.device)
    stream = cp.cuda.ExternalStream(_klib_cuda_mma__torch.cuda.current_stream().cuda_stream)
    args = (
        _klib_cuda_mma___wrap(q_nope, cp.uint16), _klib_cuda_mma___wrap(q_pe, cp.uint16),
        _klib_cuda_mma___wrap(ckv, cp.uint16), _klib_cuda_mma___wrap(kpe, cp.uint16),
        _klib_cuda_mma___wrap(kv_indptr, cp.int32), _klib_cuda_mma___wrap(kv_indices, cp.int32),
        _klib_cuda_mma___wrap(acc_g, cp.float32), _klib_cuda_mma___wrap(l_g, cp.float32), _klib_cuda_mma___wrap(cnt, cp.int32),
        _klib_cuda_mma___wrap(out, cp.uint16), _klib_cuda_mma___wrap(lse, cp.float32),
        _klib_cuda_mma__np.float32(scale2), _klib_cuda_mma__np.float32(shift), _klib_cuda_mma__np.int32(nsplit),
        _klib_cuda_mma__np.int32(q_nope.stride(1)), _klib_cuda_mma__np.int32(q_pe.stride(1)),
        _klib_cuda_mma__np.int32(out.stride(1)), _klib_cuda_mma__np.int32(lse.stride(1)),
        _klib_cuda_mma__np.int32(ckv.stride(0)), _klib_cuda_mma__np.int32(kpe.stride(0)),
    )
    with stream:
        ker((nsplit,), (32,), args)


# ---------------------------------------------------------------- entry: kernel.py
import os
import importlib.util
import torch
import triton
import triton.language as tl
import numpy as _np
import cupy as _cp

LOG2E = 1.4426950408889634


def _load_cuda_mla():
    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = os.getcwd()
    _p = os.path.join(_dir, "cuda_mla.py")
    _spec = importlib.util.spec_from_file_location("cuda_mla", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


_cuda_mla = _load_cuda_mla()


def _load_cuda_mma():
    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = os.getcwd()
    _p = os.path.join(_dir, "cuda_mma.py")
    _spec = importlib.util.spec_from_file_location("cuda_mma", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


_cuda_mma = _load_cuda_mma()


_USE_MMA_B1 = False


_MLA_B1_SRC = r'''
extern "C" __global__ void mla_b1_atomic(
    const unsigned short* __restrict__ q_nope,
    const unsigned short* __restrict__ q_pe,
    const unsigned short* __restrict__ ckv,
    const unsigned short* __restrict__ kpe,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float* __restrict__ g_acc,
    float* __restrict__ g_l,
    int* __restrict__ g_cnt,
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    float scale2, float mshift,
    int CHUNK, int num_splits,
    int qn_sh, int qp_sh, int ckv_sp, int kpe_sp, int o_sh, int lse_sh)
{
    // ONE BLOCK per head: grid=(H,), block=512 (WPB=16 warps). The 16 warps split
    // the tokens (warp w streams tokens w, w+16, w+32, ...), each warp keeping a
    // register partial (l + acc[16 dims/lane], lane L owns dims {L,L+32,..L+480}).
    // Then ONE shared-memory reduction across the 16 warps -> final out/lse, written
    // directly. NO cross-block atomics / NO __threadfence / NO persistent scratch:
    // the expensive parts of a split-K reduction are gone; only 2 __syncthreads.
    const int D = 512, DPE = 64, WPB = 16, DPL = 16, DPEL = 2;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;   // 0..15
    int h = blockIdx.x;            // one block per head

    int indptr0 = kv_indptr[0];
    int L = kv_indptr[1] - indptr0;

    float qn[DPL], qp[DPEL], acc[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; i++) {
        unsigned short raw = q_nope[h * qn_sh + lane + 32 * i];
        qn[i] = __int_as_float(((unsigned int)raw) << 16);
        acc[i] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < DPEL; j++) {
        unsigned short raw = q_pe[h * qp_sh + lane + 32 * j];
        qp[j] = __int_as_float(((unsigned int)raw) << 16);
    }
    float l_acc = 0.0f;

    // SOFTWARE-PIPELINED token loop (this node's lever). The naive per-token loop
    // (load kc -> immediately dot) STALLS the warp ~1 L2 latency (~200cyc) per
    // token because the dot depends on the just-issued gather -> that stall is the
    // ~0.076us/token the prior Design C measured, NOT the shuffle chain. Here we
    // gather U tokens' Kc/Kp FIRST (all 18*U loads/warp are mutually independent,
    // so the memory system overlaps them -> deep MLP hides the latency), THEN do
    // the U dots. Warp w owns tokens {w, w+16, ...}; one unrolled step covers U
    // tokens spaced by WPB. Data is L2-resident during timing (nnz<=2708 => <=3MB
    // << 50MB L2), so this is latency-bound -- exactly what prefetch fixes.
    const int U = 4;
    int t = warp;
    for (; t + WPB * (U - 1) < L; t += WPB * U) {
        int pg[U];
        #pragma unroll
        for (int u = 0; u < U; u++) pg[u] = kv_indices[indptr0 + t + WPB * u];
        float kc[U][DPL];
        float kp[U][DPEL];
        #pragma unroll
        for (int u = 0; u < U; u++) {
            long cbase = (long)pg[u] * ckv_sp;
            #pragma unroll
            for (int i = 0; i < DPL; i++) {
                unsigned short raw = ckv[cbase + lane + 32 * i];
                kc[u][i] = __int_as_float(((unsigned int)raw) << 16);
            }
            long pbase = (long)pg[u] * kpe_sp;
            #pragma unroll
            for (int j = 0; j < DPEL; j++) {
                unsigned short raw = kpe[pbase + lane + 32 * j];
                kp[u][j] = __int_as_float(((unsigned int)raw) << 16);
            }
        }
        #pragma unroll
        for (int u = 0; u < U; u++) {
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPL; i++) partial += qn[i] * kc[u][i];
            #pragma unroll
            for (int j = 0; j < DPEL; j++) partial += qp[j] * kp[u][j];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            float w = exp2f(fminf(partial * scale2 - mshift, 60.0f));
            l_acc += w;
            #pragma unroll
            for (int i = 0; i < DPL; i++) acc[i] += w * kc[u][i];
        }
    }
    // remainder tokens (fewer than a full U-group left for this warp)
    for (; t < L; t += WPB) {
        int page = kv_indices[indptr0 + t];
        long cbase = (long)page * ckv_sp;
        long pbase = (long)page * kpe_sp;
        float kc[DPL];
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL; i++) {
            unsigned short raw = ckv[cbase + lane + 32 * i];
            float f = __int_as_float(((unsigned int)raw) << 16);
            kc[i] = f;
            partial += qn[i] * f;
        }
        #pragma unroll
        for (int j = 0; j < DPEL; j++) {
            unsigned short raw = kpe[pbase + lane + 32 * j];
            partial += qp[j] * __int_as_float(((unsigned int)raw) << 16);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        float w = exp2f(fminf(partial * scale2 - mshift, 60.0f));
        l_acc += w;
        #pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] += w * kc[i];
    }

    // cross-warp reduction in shared memory (2 syncs, not per-token)
    __shared__ float sh_acc[WPB * D];
    __shared__ float sh_l[WPB];
    #pragma unroll
    for (int i = 0; i < DPL; i++) sh_acc[warp * D + lane + 32 * i] = acc[i];
    if (lane == 0) sh_l[warp] = l_acc;
    __syncthreads();

    int g = threadIdx.x;   // 0..511 -> output dim g
    float accf = 0.0f;
    #pragma unroll
    for (int w = 0; w < WPB; w++) accf += sh_acc[w * D + g];
    float lv = 0.0f;
    #pragma unroll
    for (int w = 0; w < WPB; w++) lv += sh_l[w];
    float o = (lv > 0.0f) ? (accf / lv) : 0.0f;
    unsigned int xi = (unsigned int)__float_as_int(o);
    unsigned int rr = xi + 0x7fffu + ((xi >> 16) & 1u);
    out[h * o_sh + g] = (unsigned short)(rr >> 16);
    if (g == 0)
        lse[h * lse_sh] = (lv > 0.0f) ? (mshift + log2f(lv))
                                      : __int_as_float(0xff800000u);
}
'''

_MLA_B1_MSHIFT = 16.0
_MLA_B1_CHUNK = 32
_MLA_B1_WPB = 16


_B1_CUDA_MAX_NNZ = 64


_B1_SPLIT_MAX = 64
_mla_b1_kernel = None
_mla_b1_scratch = {}


def _get_mla_b1_kernel():
    global _mla_b1_kernel
    if _mla_b1_kernel is None:
        mod = _cp.RawModule(code=_MLA_B1_SRC, backend='nvrtc', options=('-std=c++17',))
        _mla_b1_kernel = mod.get_function('mla_b1_atomic')
    return _mla_b1_kernel


def _get_mla_b1_scratch(H, D, device):
    key = (H, D, device)
    bufs = _mla_b1_scratch.get(key)
    if bufs is None:
        acc = torch.zeros(H * D, dtype=torch.float32, device=device)
        l = torch.zeros(H, dtype=torch.float32, device=device)
        cnt = torch.zeros(H, dtype=torch.int32, device=device)
        bufs = (acc, l, cnt)
        _mla_b1_scratch[key] = bufs
    return bufs


def _cp_view(t, dtype):
    mem = _cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * t.element_size(), t)
    return _cp.ndarray((t.numel(),), dtype=dtype, memptr=_cp.cuda.MemoryPointer(mem, 0))


def _run_b1_cuda(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices, out, lse, scale2, H, D, DPE):
    device = q_nope.device
    L = kv_indices.shape[0]
    CHUNK = _MLA_B1_CHUNK
    num_splits = (L + CHUNK - 1) // CHUNK
    if num_splits < 1:
        num_splits = 1
    acc_g, l_g, cnt_g = _get_mla_b1_scratch(H, D, device)
    kern = _get_mla_b1_kernel()
    args = (
        _cp_view(q_nope, _cp.uint16), _cp_view(q_pe, _cp.uint16),
        _cp_view(ckv, _cp.uint16), _cp_view(kpe, _cp.uint16),
        _cp_view(kv_indptr, _cp.int32), _cp_view(kv_indices, _cp.int32),
        _cp_view(acc_g, _cp.float32), _cp_view(l_g, _cp.float32), _cp_view(cnt_g, _cp.int32),
        _cp_view(out, _cp.uint16), _cp_view(lse, _cp.float32),
        _np.float32(scale2), _np.float32(_MLA_B1_MSHIFT),
        _np.int32(CHUNK), _np.int32(num_splits),
        _np.int32(q_nope.stride(1)), _np.int32(q_pe.stride(1)),
        _np.int32(ckv.stride(0)), _np.int32(kpe.stride(0)),
        _np.int32(out.stride(1)), _np.int32(lse.stride(1)),
    )
    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    with _cp.cuda.Device(device.index), _cp.cuda.ExternalStream(stream_ptr):
        kern((H,), (_MLA_B1_WPB * 32,), args)


@triton.jit
def _mla_fused(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    out_ptr, lse_ptr, scale2,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    seq_start = tl.load(kv_indptr_ptr + b)
    seq_end = tl.load(kv_indptr_ptr + b + 1)
    L = seq_end - seq_start

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)

    q_nope = tl.load(q_nope_ptr + b * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    q_pe = tl.load(q_pe_ptr + b * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_dpe[None, :])

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)

    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L
        page = tl.load(kv_indices_ptr + seq_start + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q_nope, tl.trans(kc))
        qk += tl.dot(q_pe, tl.trans(kp))
        qk = qk * scale2
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kc.dtype), kc)
        m_i = m_new

    empty = l_i == 0.0
    inv = tl.where(empty, 0.0, 1.0 / l_i)
    out = acc * inv[:, None]
    tl.store(out_ptr + b * stride_o_b + offs_h[:, None] * stride_o_h + offs_d[None, :],
             out.to(tl.bfloat16))
    lse_v = tl.where(empty, -float("inf"), m_i + tl.log2(l_i))
    tl.store(lse_ptr + b * stride_lse_b + offs_h, lse_v)


@triton.jit
def _mla_decode_bh(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    out_ptr, lse_ptr, scale2,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_start = tl.load(kv_indptr_ptr + b)
    seq_end = tl.load(kv_indptr_ptr + b + 1)
    L = seq_end - seq_start

    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)

    q_nope = tl.load(q_nope_ptr + b * stride_qn_b + h * stride_qn_h + offs_d).to(tl.float32)
    q_pe = tl.load(q_pe_ptr + b * stride_qp_b + h * stride_qp_h + offs_dpe).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)

    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L
        page = tl.load(kv_indices_ptr + seq_start + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        qk = tl.sum(q_nope[None, :] * kc, axis=1) + tl.sum(q_pe[None, :] * kp, axis=1)
        qk = qk * scale2
        qk = tl.where(mask_n, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
        m_i = m_new

    empty = l_i == 0.0
    inv = tl.where(empty, 0.0, 1.0 / l_i)
    out = acc * inv
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + offs_d, out.to(tl.bfloat16))
    lse_v = tl.where(empty, -float("inf"), m_i + tl.log2(l_i))
    tl.store(lse_ptr + b * stride_lse_b + h, lse_v)


@triton.jit
def _mla_atomic_combine(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    acc_g_ptr, l_g_ptr, cnt_ptr,
    out_ptr, lse_ptr, scale2, MSHIFT, NUM_SPLITS, CHUNK,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_ag_h,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    s = tl.program_id(0)
    seq_start = tl.load(kv_indptr_ptr + 0)
    seq_end = tl.load(kv_indptr_ptr + 1)
    L = seq_end - seq_start
    start = s * CHUNK
    stop = tl.minimum(start + CHUNK, L)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)
    qn = tl.load(q_nope_ptr + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    qp = tl.load(q_pe_ptr + offs_h[:, None] * stride_qp_h + offs_dpe[None, :])

    l_acc = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)
    for tok in range(start, stop, BLOCK_N):
        offs_n = tok + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stop
        page = tl.load(kv_indices_ptr + seq_start + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(qn, tl.trans(kc)) + tl.dot(qp, tl.trans(kp))
        qk = qk * scale2 - MSHIFT
        qk = tl.minimum(qk, 60.0)
        w = tl.where(mask_n[None, :], tl.exp2(qk), 0.0)
        l_acc += tl.sum(w, axis=1)
        acc += tl.dot(w.to(kc.dtype), kc)


    tl.atomic_add(l_g_ptr + offs_h, l_acc, sem="relaxed")
    tl.atomic_add(acc_g_ptr + offs_h[:, None] * stride_ag_h + offs_d[None, :], acc, sem="relaxed")
    old = tl.atomic_add(cnt_ptr, 1.0, sem="acq_rel")
    if old == NUM_SPLITS - 1:
        l_g = tl.load(l_g_ptr + offs_h)
        a_g = tl.load(acc_g_ptr + offs_h[:, None] * stride_ag_h + offs_d[None, :])
        empty = l_g == 0.0
        inv = tl.where(empty, 0.0, 1.0 / l_g)
        out = a_g * inv[:, None]
        tl.store(out_ptr + offs_h[:, None] * stride_o_h + offs_d[None, :], out.to(tl.bfloat16))
        lse_v = tl.where(empty, -float("inf"), MSHIFT + tl.log2(l_g))
        tl.store(lse_ptr + offs_h, lse_v)


        tl.store(l_g_ptr + offs_h, tl.zeros([H], tl.float32))
        tl.store(acc_g_ptr + offs_h[:, None] * stride_ag_h + offs_d[None, :],
                 tl.zeros([H, D], tl.float32))
        tl.store(cnt_ptr, 0.0)


@triton.jit
def _mla_atomic_multi(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    acc_g_ptr, l_g_ptr, cnt_ptr,
    out_ptr, lse_ptr, scale2, MSHIFT, CHUNK,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_agb, stride_agh,
    stride_o_b, stride_o_h,
    stride_lse_b,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    p = tl.program_id(0)


    prev = 0
    b_idx = 0
    local = 0
    ncb_b = 1
    beg_b = 0
    L_b = 0
    found = p < 0
    for bb in range(0, B):
        beg = tl.load(kv_indptr_ptr + bb)
        end = tl.load(kv_indptr_ptr + bb + 1)
        Lbb = end - beg
        ncb = tl.maximum(tl.cdiv(Lbb, CHUNK), 1)
        take = (~found) & (p < prev + ncb)
        b_idx = tl.where(take, bb, b_idx)
        local = tl.where(take, p - prev, local)
        ncb_b = tl.where(take, ncb, ncb_b)
        beg_b = tl.where(take, beg, beg_b)
        L_b = tl.where(take, Lbb, L_b)
        found = found | take
        prev = prev + ncb

    start = local * CHUNK
    stop = tl.minimum(start + CHUNK, L_b)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)
    qn = tl.load(q_nope_ptr + b_idx * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    qp = tl.load(q_pe_ptr + b_idx * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_dpe[None, :])

    l_acc = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)
    for tok in range(start, stop, BLOCK_N):
        offs_n = tok + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stop
        page = tl.load(kv_indices_ptr + beg_b + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(qn, tl.trans(kc)) + tl.dot(qp, tl.trans(kp))
        qk = qk * scale2 - MSHIFT
        qk = tl.minimum(qk, 60.0)
        w = tl.where(mask_n[None, :], tl.exp2(qk), 0.0)
        l_acc += tl.sum(w, axis=1)
        acc += tl.dot(w.to(kc.dtype), kc)

    if found:
        ab = acc_g_ptr + b_idx * stride_agb
        lb = l_g_ptr + b_idx * H
        tl.atomic_add(lb + offs_h, l_acc, sem="relaxed")
        tl.atomic_add(ab + offs_h[:, None] * stride_agh + offs_d[None, :], acc, sem="relaxed")
        old = tl.atomic_add(cnt_ptr + b_idx, 1.0, sem="acq_rel")
        if old == ncb_b - 1:
            l_g = tl.load(lb + offs_h)
            a_g = tl.load(ab + offs_h[:, None] * stride_agh + offs_d[None, :])
            empty = l_g == 0.0
            inv = tl.where(empty, 0.0, 1.0 / l_g)
            out = a_g * inv[:, None]
            tl.store(out_ptr + b_idx * stride_o_b + offs_h[:, None] * stride_o_h + offs_d[None, :],
                     out.to(tl.bfloat16))
            lse_v = tl.where(empty, -float("inf"), MSHIFT + tl.log2(l_g))
            tl.store(lse_ptr + b_idx * stride_lse_b + offs_h, lse_v)

            tl.store(lb + offs_h, tl.zeros([H], tl.float32))
            tl.store(ab + offs_h[:, None] * stride_agh + offs_d[None, :], tl.zeros([H, D], tl.float32))
            tl.store(cnt_ptr + b_idx, 0.0)


@triton.jit
def _mla_partial(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    m_part_ptr, l_part_ptr, acc_part_ptr, batch_start_ptr,
    scale2,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_mp_h, stride_mp_p,
    stride_ap_h, stride_ap_p,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr,
    BLOCK_N: tl.constexpr, CHUNK: tl.constexpr,
):
    p = tl.program_id(0)


    is_writer = p == 0
    if is_writer:
        tl.store(batch_start_ptr + 0, p - p)
    prev = p - p
    b_idx = p - p
    local = p - p
    found = p < 0
    for bb in range(0, B):
        beg = tl.load(kv_indptr_ptr + bb)
        end = tl.load(kv_indptr_ptr + bb + 1)
        ncb = tl.cdiv(end - beg, CHUNK)
        take = (~found) & (p < prev + ncb)
        b_idx = tl.where(take, bb, b_idx)
        local = tl.where(take, p - prev, local)
        found = found | take
        prev = prev + ncb
        if is_writer:
            tl.store(batch_start_ptr + (bb + 1), prev)

    beg_b = tl.load(kv_indptr_ptr + b_idx)
    end_b = tl.load(kv_indptr_ptr + b_idx + 1)
    L = end_b - beg_b
    start = local * CHUNK
    stop = tl.minimum(start + CHUNK, L)
    start = tl.where(found, start, 0)
    stop = tl.where(found, stop, 0)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)

    qn = tl.load(q_nope_ptr + b_idx * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    qp = tl.load(q_pe_ptr + b_idx * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_dpe[None, :])

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)

    for tok in range(start, stop, BLOCK_N):
        offs_n = tok + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stop
        page = tl.load(kv_indices_ptr + beg_b + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(qn, tl.trans(kc))
        qk += tl.dot(qp, tl.trans(kp))
        qk = qk * scale2
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        pw = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pw, axis=1)
        acc = acc * alpha[:, None] + tl.dot(pw.to(kc.dtype), kc)
        m_i = m_new


    tl.store(m_part_ptr + offs_h * stride_mp_h + p * stride_mp_p, m_i)
    tl.store(l_part_ptr + offs_h * stride_mp_h + p * stride_mp_p, l_i)
    tl.store(acc_part_ptr + offs_h[:, None] * stride_ap_h + p * stride_ap_p + offs_d[None, :],
             acc.to(tl.float16))


@triton.jit
def _mla_partial_fs(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr,
    kv_indptr_ptr, kv_indices_ptr,
    l_part_ptr, acc_part_ptr, batch_start_ptr,
    scale2, MSHIFT,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_ckv_p, stride_kpe_p,
    stride_lp_h, stride_lp_p,
    stride_ap_h, stride_ap_p,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr,
    BLOCK_N: tl.constexpr, CHUNK: tl.constexpr,
):
    p = tl.program_id(0)

    is_writer = p == 0
    if is_writer:
        tl.store(batch_start_ptr + 0, p - p)
    prev = p - p
    b_idx = p - p
    local = p - p
    found = p < 0
    for bb in range(0, B):
        beg = tl.load(kv_indptr_ptr + bb)
        end = tl.load(kv_indptr_ptr + bb + 1)
        ncb = tl.cdiv(end - beg, CHUNK)
        take = (~found) & (p < prev + ncb)
        b_idx = tl.where(take, bb, b_idx)
        local = tl.where(take, p - prev, local)
        found = found | take
        prev = prev + ncb
        if is_writer:
            tl.store(batch_start_ptr + (bb + 1), prev)

    beg_b = tl.load(kv_indptr_ptr + b_idx)
    end_b = tl.load(kv_indptr_ptr + b_idx + 1)
    L = end_b - beg_b
    start = local * CHUNK
    stop = tl.minimum(start + CHUNK, L)
    start = tl.where(found, start, 0)
    stop = tl.where(found, stop, 0)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)

    qn = tl.load(q_nope_ptr + b_idx * stride_qn_b + offs_h[:, None] * stride_qn_h + offs_d[None, :])
    qp = tl.load(q_pe_ptr + b_idx * stride_qp_b + offs_h[:, None] * stride_qp_h + offs_dpe[None, :])

    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)
    for tok in range(start, stop, BLOCK_N):
        offs_n = tok + tl.arange(0, BLOCK_N)
        mask_n = offs_n < stop
        page = tl.load(kv_indices_ptr + beg_b + offs_n, mask=mask_n, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=mask_n[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=mask_n[:, None], other=0.0)
        qk = tl.dot(qn, tl.trans(kc)) + tl.dot(qp, tl.trans(kp))
        qk = qk * scale2 - MSHIFT
        qk = tl.minimum(qk, 60.0)
        w = tl.where(mask_n[None, :], tl.exp2(qk), 0.0)
        l_i += tl.sum(w, axis=1)
        acc += tl.dot(w.to(kc.dtype), kc)

    tl.store(l_part_ptr + offs_h * stride_lp_h + p * stride_lp_p, l_i)
    tl.store(acc_part_ptr + offs_h[:, None] * stride_ap_h + p * stride_ap_p + offs_d[None, :],
             acc.to(tl.float16))


@triton.jit
def _mla_combine_fs(
    l_part_ptr, acc_part_ptr, batch_start_ptr,
    out_ptr, lse_ptr, MSHIFT,
    stride_lp_h, stride_lp_p,
    stride_ap_h, stride_ap_p,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)

    p_start = tl.load(batch_start_ptr + b)
    p_end = tl.load(batch_start_ptr + b + 1)
    lp_base = h * stride_lp_h
    ap_base = h * stride_ap_h

    l_g = 0.0
    acc = tl.zeros([D], tl.float32)
    for pp in range(p_start, p_end):
        ls = tl.load(l_part_ptr + lp_base + pp * stride_lp_p)
        a = tl.load(acc_part_ptr + ap_base + pp * stride_ap_p + offs_d)
        l_g += ls
        acc += a.to(tl.float32)

    is_empty = l_g == 0.0
    inv_l = tl.where(is_empty, 0.0, 1.0 / l_g)
    out = acc * inv_l
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + offs_d, out.to(tl.bfloat16))
    lse_val = tl.where(is_empty, -float("inf"), MSHIFT + tl.log2(l_g))
    tl.store(lse_ptr + b * stride_lse_b + h, lse_val)


@triton.jit
def _mla_combine_serial(
    m_part_ptr, l_part_ptr, acc_part_ptr, batch_start_ptr,
    out_ptr, lse_ptr,
    stride_mp_h, stride_mp_p,
    stride_ap_h, stride_ap_p,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    neg_inf = -float("inf")

    p_start = tl.load(batch_start_ptr + b)
    p_end = tl.load(batch_start_ptr + b + 1)

    mp_base = h * stride_mp_h
    ap_base = h * stride_ap_h

    m_g = neg_inf
    for pp in range(p_start, p_end):
        ms = tl.load(m_part_ptr + mp_base + pp * stride_mp_p)
        m_g = tl.maximum(m_g, ms)

    l_g = 0.0
    acc = tl.zeros([D], tl.float32)
    for pp in range(p_start, p_end):
        ms = tl.load(m_part_ptr + mp_base + pp * stride_mp_p)
        ls = tl.load(l_part_ptr + mp_base + pp * stride_mp_p)
        scale = tl.where(ms == neg_inf, 0.0, tl.exp2(ms - m_g))
        a = tl.load(acc_part_ptr + ap_base + pp * stride_ap_p + offs_d)
        l_g += ls * scale
        acc += a.to(tl.float32) * scale

    is_empty = (m_g == neg_inf) | (l_g == 0.0)
    inv_l = tl.where(is_empty, 0.0, 1.0 / l_g)
    out = acc * inv_l
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + offs_d, out.to(tl.bfloat16))
    lse_val = tl.where(is_empty, neg_inf, m_g + tl.log2(l_g))
    tl.store(lse_ptr + b * stride_lse_b + h, lse_val)


@triton.jit
def _mla_combine(
    m_part_ptr, l_part_ptr, acc_part_ptr, batch_start_ptr,
    out_ptr, lse_ptr,
    stride_mp_h, stride_mp_p,
    stride_ap_h, stride_ap_p,
    stride_o_b, stride_o_h,
    stride_lse_b,
    H: tl.constexpr, D: tl.constexpr, BLOCK_P: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    neg_inf = -float("inf")

    p_start = tl.load(batch_start_ptr + b)
    p_end = tl.load(batch_start_ptr + b + 1)

    mp_base = h * stride_mp_h
    ap_base = h * stride_ap_h

    m_i = neg_inf
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)

    for blk in range(p_start, p_end, BLOCK_P):
        offs_p = blk + tl.arange(0, BLOCK_P)
        mask_p = offs_p < p_end

        m_s = tl.load(m_part_ptr + mp_base + offs_p * stride_mp_p, mask=mask_p, other=neg_inf)
        l_s = tl.load(l_part_ptr + mp_base + offs_p * stride_mp_p, mask=mask_p, other=0.0)
        acc_s = tl.load(
            acc_part_ptr + ap_base + offs_p[:, None] * stride_ap_p + offs_d[None, :],
            mask=mask_p[:, None], other=0.0,
        )

        m_blk = tl.max(m_s, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        scale_s = tl.where(m_s == neg_inf, 0.0, tl.exp2(m_s - m_new))
        l_blk = tl.sum(l_s * scale_s, axis=0)
        acc_blk = tl.sum(acc_s.to(tl.float32) * scale_s[:, None], axis=0)

        alpha_i = tl.where(m_i == neg_inf, 0.0, tl.exp2(m_i - m_new))
        l_i = l_i * alpha_i + l_blk
        acc = acc * alpha_i + acc_blk
        m_i = m_new

    is_empty = (m_i == neg_inf) | (l_i == 0.0)
    inv_l = tl.where(is_empty, 0.0, 1.0 / l_i)
    out = acc * inv_l
    tl.store(out_ptr + b * stride_o_b + h * stride_o_h + offs_d, out.to(tl.bfloat16))
    lse_val = tl.where(is_empty, neg_inf, m_i + tl.log2(l_i))
    tl.store(lse_ptr + b * stride_lse_b + h, lse_val)


_FUSED_MAX = 64


_FUSED_STAGES = 2

_BLOCK_N = 64
_PARTIAL_BLOCK_N = 64


_TARGET_PROGS = 256
_BLOCK_P = 16


_COMBINE_PAR_MAX_B = 16
_COMBINE_PAR_MIN_AVG_NCB = 3


_BH_MAX_B = 0
_BH_BLOCK_N = 64
_BH_WARPS = 4
_BH_STAGES = 3


_ATOMIC_MAX_B = 1
_ATOMIC_MSHIFT = 16.0
_ATOMIC_BLOCK_N = 32
_ATOMIC_CHUNK = 32


_ATOMIC_SMALL_NNZ = 1000
_ATOMIC_BLOCK_N_SMALL = 16
_ATOMIC_WARPS = 8


_ATOMIC_STAGES = 2


_ATOMIC_MULTI_MAX_B = 16


_ATOMIC_MULTI_MIN_NNZ = 900
_ATOMIC_MULTI_MSHIFT = 16.0
_ATOMIC_MULTI_BLOCK_N = 64


_ATOMIC_MULTI_WARPS = 8
_ATOMIC_MULTI_WARPS_LARGE = 4
_ATOMIC_MULTI_WARPS_NNZ = 9000
_ATOMIC_MULTI_STAGES = 2


_ATOMIC_SCRATCH = {}


def _get_atomic_scratch(H, D, device):
    key = (H, D, device)
    buf = _ATOMIC_SCRATCH.get(key)
    if buf is None:
        buf = torch.zeros(H * D + H + 1, dtype=torch.float32, device=device)
        _ATOMIC_SCRATCH[key] = buf
    return buf


_ATOMIC_MULTI_SCRATCH = {}


def _get_atomic_multi_scratch(B, H, D, device):
    key = (B, H, D, device)
    bufs = _ATOMIC_MULTI_SCRATCH.get(key)
    if bufs is None:
        acc_g = torch.zeros((B, H, D), dtype=torch.float32, device=device)
        l_g = torch.zeros((B, H), dtype=torch.float32, device=device)
        cnt = torch.zeros((B,), dtype=torch.float32, device=device)
        bufs = (acc_g, l_g, cnt)
        _ATOMIC_MULTI_SCRATCH[key] = bufs
    return bufs


def _choose_chunk(nnz):


    c = (nnz + _TARGET_PROGS // 2) // _TARGET_PROGS
    if c < _BLOCK_N:
        c = _BLOCK_N
    return int(c)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    B, H, D = q_nope.shape
    DPE = q_pe.shape[-1]
    device = q_nope.device

    ckv = ckv_cache.reshape(-1, D)
    kpe = kpe_cache.reshape(-1, DPE)
    nnz = kv_indices.shape[0]

    scale2 = float(sm_scale) * LOG2E

    out = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)


    if B == 1 and nnz <= _B1_CUDA_MAX_NNZ:
        _run_b1_cuda(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
                     out, lse, scale2, H, D, DPE)
        return {"output": out, "lse": lse}


    if B == 1 and _USE_MMA_B1:
        _cuda_mma.mla_decode_bs1_wmma(q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
                                      out, lse, scale2, _MLA_B1_MSHIFT)
        return {"output": out, "lse": lse}


    if nnz <= _FUSED_MAX:
        _mla_fused[(B,)](
            q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
            out, lse, scale2,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            out.stride(0), out.stride(1),
            lse.stride(0),
            H=H, D=D, DPE=DPE, BLOCK_N=_BLOCK_N,
            num_warps=4, num_stages=_FUSED_STAGES,
        )
        return {"output": out, "lse": lse}


    if B <= _ATOMIC_MAX_B:

        if nnz < _ATOMIC_SMALL_NNZ:
            BLK = _ATOMIC_BLOCK_N_SMALL
        else:
            BLK = _ATOMIC_BLOCK_N
        CHUNK = BLK
        num_splits = (nnz + CHUNK - 1) // CHUNK
        if num_splits < 1:
            num_splits = 1
        scratch = _get_atomic_scratch(H, D, device)
        acc_g = scratch[: H * D].view(H, D)
        l_g = scratch[H * D : H * D + H]
        cnt = scratch[H * D + H : H * D + H + 1]
        _mla_atomic_combine[(num_splits,)](
            q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
            acc_g, l_g, cnt,
            out, lse, scale2, _ATOMIC_MSHIFT, num_splits, CHUNK,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            acc_g.stride(0),
            out.stride(0), out.stride(1),
            lse.stride(0),
            H=H, D=D, DPE=DPE, BLOCK_N=BLK,
            num_warps=_ATOMIC_WARPS, num_stages=_ATOMIC_STAGES,
        )
        return {"output": out, "lse": lse}


    if B <= _ATOMIC_MULTI_MAX_B and nnz >= _ATOMIC_MULTI_MIN_NNZ:
        CHUNK = _choose_chunk(nnz)
        num_programs = nnz // CHUNK + B + 1
        am_warps = _ATOMIC_MULTI_WARPS if nnz < _ATOMIC_MULTI_WARPS_NNZ else _ATOMIC_MULTI_WARPS_LARGE
        acc_g, l_g, cnt = _get_atomic_multi_scratch(B, H, D, device)
        _mla_atomic_multi[(num_programs,)](
            q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
            acc_g, l_g, cnt,
            out, lse, scale2, _ATOMIC_MULTI_MSHIFT, CHUNK,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            acc_g.stride(0), acc_g.stride(1),
            out.stride(0), out.stride(1),
            lse.stride(0),
            B=B, H=H, D=D, DPE=DPE, BLOCK_N=_ATOMIC_MULTI_BLOCK_N,
            num_warps=am_warps, num_stages=_ATOMIC_MULTI_STAGES,
        )
        return {"output": out, "lse": lse}


    if B > _ATOMIC_MULTI_MAX_B:
        CHUNK = _choose_chunk(nnz)
        num_programs = nnz // CHUNK + B + 1
        l_part = torch.empty((H, num_programs), dtype=torch.float32, device=device)
        acc_part = torch.empty((H, num_programs, D), dtype=torch.float16, device=device)
        batch_start = torch.empty((B + 1,), dtype=torch.int32, device=device)
        _mla_partial_fs[(num_programs,)](
            q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
            l_part, acc_part, batch_start,
            scale2, _ATOMIC_MULTI_MSHIFT,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            ckv.stride(0), kpe.stride(0),
            l_part.stride(0), l_part.stride(1),
            acc_part.stride(0), acc_part.stride(1),
            B=B, H=H, D=D, DPE=DPE, BLOCK_N=_PARTIAL_BLOCK_N, CHUNK=CHUNK,
            num_warps=4, num_stages=2,
        )
        _mla_combine_fs[(B, H)](
            l_part, acc_part, batch_start,
            out, lse, _ATOMIC_MULTI_MSHIFT,
            l_part.stride(0), l_part.stride(1),
            acc_part.stride(0), acc_part.stride(1),
            out.stride(0), out.stride(1),
            lse.stride(0),
            H=H, D=D,
            num_warps=4, num_stages=1,
        )
        return {"output": out, "lse": lse}


    CHUNK = _choose_chunk(nnz)

    num_programs = nnz // CHUNK + B + 1


    m_part = torch.empty((H, num_programs), dtype=torch.float32, device=device)
    l_part = torch.empty((H, num_programs), dtype=torch.float32, device=device)
    acc_part = torch.empty((H, num_programs, D), dtype=torch.float16, device=device)
    batch_start = torch.empty((B + 1,), dtype=torch.int32, device=device)

    _mla_partial[(num_programs,)](
        q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
        m_part, l_part, acc_part, batch_start,
        scale2,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv.stride(0), kpe.stride(0),
        m_part.stride(0), m_part.stride(1),
        acc_part.stride(0), acc_part.stride(1),
        B=B, H=H, D=D, DPE=DPE, BLOCK_N=_PARTIAL_BLOCK_N, CHUNK=CHUNK,
        num_warps=4, num_stages=2,
    )

    use_parallel = (B <= _COMBINE_PAR_MAX_B) and (num_programs >= _COMBINE_PAR_MIN_AVG_NCB * B)
    if use_parallel:

        _mla_combine[(B, H)](
            m_part, l_part, acc_part, batch_start,
            out, lse,
            m_part.stride(0), m_part.stride(1),
            acc_part.stride(0), acc_part.stride(1),
            out.stride(0), out.stride(1),
            lse.stride(0),
            H=H, D=D, BLOCK_P=_BLOCK_P,
            num_warps=4, num_stages=2,
        )
    else:

        _mla_combine_serial[(B, H)](
            m_part, l_part, acc_part, batch_start,
            out, lse,
            m_part.stride(0), m_part.stride(1),
            acc_part.stride(0), acc_part.stride(1),
            out.stride(0), out.stride(1),
            lse.stride(0),
            H=H, D=D,
            num_warps=4, num_stages=1,
        )
    return {"output": out, "lse": lse}
