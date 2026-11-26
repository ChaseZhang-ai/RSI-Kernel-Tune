# SPDX-License-Identifier: Apache-2.0
# Vector memory operations and packed FP32 arithmetic adapted from the RSI
# FlashInfer 001 entry. Row splitting and grouped CTA ideas from CompoundingKernel
# Explorer (QianQu/FlashInfer/002_fused_add_rmsnorm_h4096).
from ctypes import c_void_p
from functools import lru_cache

import torch
import triton
import triton.language as tl
from cuda.bindings import driver, nvrtc

_CUDA = r"""
using U32 = unsigned int;
using U64 = unsigned long long;
struct Vec { U32 x[8]; };

__device__ __forceinline__ U64 pair(U32 lo, U32 hi) {
    U64 p;
    asm("mov.b64 %0, {%1,%2};" : "=l"(p) : "r"(lo), "r"(hi));
    return p;
}

__device__ __forceinline__ U64 widen(U32 v) {
    return pair(v << 16, v & 0xffff0000u);
}

__device__ __forceinline__ U64 add_pairs(U64 a, U64 b) {
    U64 p;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(p) : "l"(a), "l"(b));
    return p;
}

__device__ __forceinline__ U64 add_bf16_pair(U32 h, U32 r) {
    U64 result;
    U64 residual = widen(r);
    // Widen the BF16 operand exactly and add in FP32.
    asm("{.reg .b16 h0, h1; .reg .f32 r0, r1, lo, hi; "
        "mov.b32 {h0,h1}, %1; mov.b64 {r0,r1}, %2;"
        "add.rn.f32.bf16 lo, h0, r0; add.rn.f32.bf16 hi, h1, r1; "
        "mov.b64 %0, {lo,hi};}"
        : "=l"(result) : "r"(h), "l"(residual));
    return result;
}

__device__ __forceinline__ U64 mul(U64 a, U64 b) {
    U64 p;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(p) : "l"(a), "l"(b));
    return p;
}

__device__ __forceinline__ U64 fma_pair(U64 a, U64 b, U64 c) {
    U64 p;
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(p) : "l"(a), "l"(b), "l"(c));
    return p;
}

__device__ __forceinline__ U32 pack(U64 p) {
    float lo, hi;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(lo), "=f"(hi) : "l"(p));
    U32 v;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(v) : "f"(hi), "f"(lo));
    return v;
}

__device__ __forceinline__ float horizontal(U64 p) {
    float lo, hi;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(lo), "=f"(hi) : "l"(p));
    return lo + hi;
}

__device__ __forceinline__ U64 keep_policy() {
    U64 p;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ U64 stream_policy() {
    U64 p;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

template <bool IsWeight>
__device__ __forceinline__ Vec load_vec(const unsigned short* src, U64 policy) {
    Vec v;
    if (IsWeight) {
        asm volatile("ld.global.nc.L1::evict_last.L2::cache_hint.v8.u32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                     : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]),
                       "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7])
                     : "l"(src), "l"(policy));
    } else {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::128B.v8.u32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                     : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]),
                       "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7])
                     : "l"(src), "l"(policy));
    }
    return v;
}

__device__ __forceinline__ void store_vec(unsigned short* dst, Vec v, U64 policy) {
    asm volatile("st.global.L1::no_allocate.L2::cache_hint.v8.u32 "
                 "[%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(dst), "r"(v.x[0]), "r"(v.x[1]), "r"(v.x[2]), "r"(v.x[3]),
                    "r"(v.x[4]), "r"(v.x[5]), "r"(v.x[6]), "r"(v.x[7]),
                    "l"(policy) : "memory");
}

struct Vec4 { U32 x[4]; };

template <int V> struct Vector;
template <> struct Vector<8> { using Type = Vec4; };
template <> struct Vector<16> { using Type = Vec; };

template <int V, bool Weight>
__device__ __forceinline__ typename Vector<V>::Type load(const unsigned short* src, U64 policy) {
    if constexpr (V == 16) {
        return load_vec<Weight>(src, policy);
    } else {
        Vec4 v;
        asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3])
                     : "l"(src), "l"(policy));
        return v;
    }
}

template <int V>
__device__ __forceinline__ void store(unsigned short* dst, typename Vector<V>::Type v, U64 policy) {
    if constexpr (V == 16) {
        store_vec(dst, v, policy);
    } else {
        asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;"
                     :: "l"(dst), "r"(v.x[0]), "r"(v.x[1]), "r"(v.x[2]), "r"(v.x[3]),
                        "l"(policy) : "memory");
    }
}


template <int Threads, int Width, int Parts>
__device__ __forceinline__ void rmsnorm(
    const unsigned short* __restrict__ h, const unsigned short* __restrict__ residual,
    const unsigned short* __restrict__ w, unsigned short* __restrict__ out) {
    constexpr int chunks = 4096 / (Threads * Width);
    constexpr int pairs = Width / 2;
    constexpr int part_width = 4096 / Parts;
    __shared__ float partials[Threads / 32];

    const int row = blockIdx.x / Parts;
    const int part = blockIdx.x % Parts;
    const int lane = threadIdx.x;
    const U64 keep = keep_policy();
    const U64 input_policy = Parts == 1 ? stream_policy() : keep;
    U64 x[chunks][pairs];
    typename Vector<Width>::Type weights[chunks];

    // Rotate the input columns so that each CTA can write a different slice.
    // All CTAs compute the same full-row RMS from the two current inputs.
    #pragma unroll
    for (int c = 0; c < chunks; ++c) {
        const int local_col = c * Threads * Width + lane * Width;
        const int col = local_col ^ (part * part_width);
        if (local_col < part_width)
            weights[c] = load<Width, true>(w + col, keep);
    }

    U64 sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int c = 0; c < chunks; ++c) {
        const int col = (c * Threads * Width + lane * Width) ^ (part * part_width);
        const int offset = row * 4096 + col;
        const auto a = load<Width, false>(h + offset, input_policy);
        const auto b = load<Width, false>(residual + offset, input_policy);
        #pragma unroll
        for (int k = 0; k < pairs; k += 2) {
            x[c][k] = add_bf16_pair(a.x[k], b.x[k]);
            x[c][k + 1] = add_bf16_pair(a.x[k + 1], b.x[k + 1]);
            sum0 = fma_pair(x[c][k], x[c][k], sum0);
            sum1 = fma_pair(x[c][k + 1], x[c][k + 1], sum1);
        }
    }

    float stat = horizontal(add_pairs(sum0, sum1));
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        stat += __shfl_xor_sync(0xffffffff, stat, delta);
    if (lane % 32 == 0)
        partials[lane / 32] = stat;
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < Threads / 32; ++warp)
        sum += partials[warp];
    const U32 inv = __float_as_uint(rsqrtf(sum * (1.0f / 4096) + 1.0e-5f));
    const U64 scale = pair(inv, inv);
    #pragma unroll
    for (int c = 0; c < chunks; ++c) {
        const int local_col = c * Threads * Width + lane * Width;
        if (local_col < part_width) {
            typename Vector<Width>::Type y;
            #pragma unroll
            for (int k = 0; k < pairs; ++k)
                y.x[k] = pack(mul(mul(x[c][k], scale), widen(weights[c].x[k])));
            const int offset = row * 4096 + (local_col ^ (part * part_width));
            store<Width>(out + offset, y, keep);
        }
    }
}

#define INSTANCE(name, threads, width, parts)                                  \
extern "C" __global__ __launch_bounds__(threads) void name(                    \
    const unsigned short* h, const unsigned short* r, const unsigned short* w,  \
    unsigned short* o) {                                                       \
    rmsnorm<threads, width, parts>(h, r, w, o);                                  \
}

INSTANCE(quartile_rows, 256, 16, 4)
INSTANCE(half_rows, 256, 16, 2)
INSTANCE(streaming_rows, 128, 8, 1)
"""

def _checked(result):
    code, *values = result
    if int(code):
        raise RuntimeError(f"CUDA operation failed with {code}")
    return values[0] if len(values) == 1 else values


@lru_cache(maxsize=None)
def _compile(capability, context):
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"fused_rmsnorm.cu", 0, [], []))
    try:
        flags = [f"--gpu-architecture=sm_{capability[0]}{capability[1]}".encode(), b"--std=c++17"]
        status = nvrtc.nvrtcCompileProgram(program, len(flags), flags)[0]
        if int(status):
            log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace").strip("\x00"))
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))
    return _checked(driver.cuModuleLoadData(cubin))


@lru_cache(maxsize=None)
def _function(capability, context, name):
    module = _compile(capability, context)
    return _checked(driver.cuModuleGetFunction(module, name.encode()))


def _launch_native(h, r, w, o, name, threads, parts):
    context = _checked(driver.cuCtxGetCurrent())
    fn = _function(torch.cuda.get_device_capability(h.device), int(context), name)
    values = (h.data_ptr(), r.data_ptr(), w.data_ptr(), o.data_ptr())
    types = (c_void_p,) * 4
    blocks = h.shape[0] * parts
    stream = torch.cuda.current_stream(h.device).cuda_stream
    _checked(driver.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream,
                                   (values, types), 0))
    return fn


@triton.jit
def _strided_row(H, R, W, O, HS0: tl.constexpr, HS1: tl.constexpr,
                 RS0: tl.constexpr, RS1: tl.constexpr, WS: tl.constexpr,
                 OS0: tl.constexpr, OS1: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    h = tl.load(H + row * HS0 + col * HS1).to(tl.float32)
    r = tl.load(R + row * RS0 + col * RS1).to(tl.float32)
    w = tl.load(W + col * WS).to(tl.float32)
    x = h + r
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 4096) + 1.0e-5)
    tl.store(O + row * OS0 + col * OS1, ((x * scale) * w).to(tl.bfloat16))


def run(hidden_states, residual, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return

    tensors = (hidden_states, residual, weight, output)
    if any(not tensor.is_contiguous() for tensor in tensors) or any(tensor.data_ptr() % 32 for tensor in tensors):
        _strided_row[(batch,)](
            *tensors, *hidden_states.stride(), *residual.stride(), *weight.stride(),
            *output.stride(), num_warps=4,
        )
        return

    if batch <= 16:
        name, threads, parts = "quartile_rows", 256, 4
    elif batch <= 170:
        name, threads, parts = "half_rows", 256, 2
    else:
        name, threads, parts = "streaming_rows", 128, 1
    with torch.cuda.device(output.device):
        _launch_native(*tensors, name, threads, parts)
