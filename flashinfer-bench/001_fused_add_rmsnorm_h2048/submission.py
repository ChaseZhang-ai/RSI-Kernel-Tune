# SPDX-License-Identifier: Apache-2.0
# Wide memory operations, CTA replication, and cache policies are inspired by
# CompoundingKernel-Explorer, FlashInfer/001_fused_add_rmsnorm_h2048 (QianQu).
from ctypes import c_int, c_void_p
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

__device__ __forceinline__ float sum_squares(Vec h, Vec r, U64 (&x)[8]) {
    U64 evens = 0;
    U64 odds = 0;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        x[i] = add_bf16_pair(h.x[i], r.x[i]);
        x[i+1] = add_bf16_pair(h.x[i+1], r.x[i+1]);
        evens = fma_pair(x[i], x[i], evens);
        odds = fma_pair(x[i+1], x[i+1], odds);
    }
    return horizontal(add_pairs(evens, odds));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, delta);
    return v;
}

__device__ __forceinline__ Vec normalize(const U64 (&x)[8], const U64 (&w)[8], float inv) {
    Vec out;
    U32 bits = __float_as_uint(inv);
    U64 scale = pair(bits, bits);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        out.x[i] = pack(mul(mul(x[i], scale), w[i]));
    return out;
}

__device__ __forceinline__ void widen_weights(Vec v, U64 (&w)[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        w[i] = widen(v.x[i]);
}

extern "C" __global__ __launch_bounds__(128)
void small(const unsigned short* h, const unsigned short* r, const unsigned short* w,
           unsigned short* out, int log_parts) {
    const int row = blockIdx.x >> log_parts;
    const int part = blockIdx.x & ((1 << log_parts) - 1);
    const int lane = threadIdx.x;
    const int span = 128 >> log_parts;
    const int col = 16 * (lane ^ (part * span));
    const int offset = row * 2048 + col;
    const U64 keep = keep_policy();
    const U64 stream = stream_policy();

    U64 weights[8];
    if (lane < span)
        widen_weights(load_vec<true>(w + col, keep), weights);
    U64 input_policy = log_parts >= 3 ? keep : stream;
    Vec hv = load_vec<false>(h + offset, input_policy);
    Vec rv = load_vec<false>(r + offset, input_policy);

    U64 x[8];
    float stat = sum_squares(hv, rv, x);
    #pragma unroll
    for (int delta = 16; delta >= 4; delta >>= 1)
        stat += __shfl_xor_sync(0xffffffffu, stat, delta);
    __shared__ __align__(16) float partials[16];
    if ((lane & 31) < 4)
        partials[(lane >> 5) * 4 + (lane & 31)] = stat;
    __syncthreads();
    if (lane < span) {
        float sum[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            sum[i] = partials[i];
        #pragma unroll
        for (int stride = 8; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int i = 0; i < stride; ++i)
                sum[i] += sum[i + stride];
        }
        float total = sum[0];
        float inv = rsqrtf(total * (1.0f / 2048) + 1.0e-6f);
        store_vec(out + offset, normalize(x, weights, inv), keep);
    }
}

extern "C" __global__ __launch_bounds__(128, 6)
void row_pair(const unsigned short* h, const unsigned short* r, const unsigned short* w,
              unsigned short* out, int rows) {
    const int row = 2 * blockIdx.x;
    const int next = row + 1 < rows ? row + 1 : row;
    const int lane = threadIdx.x;
    const int col = 16 * lane;
    const U64 keep = keep_policy();
    const U64 stream = stream_policy();

    Vec h0 = load_vec<false>(h + row * 2048 + col, stream);
    Vec h1 = load_vec<false>(h + next * 2048 + col, stream);
    Vec r0 = load_vec<false>(r + row * 2048 + col, stream);
    Vec r1 = load_vec<false>(r + next * 2048 + col, stream);
    U64 weights[8];
    widen_weights(load_vec<true>(w + col, keep), weights);

    U64 x0[8];
    U64 x1[8];
    float stat0 = warp_sum(sum_squares(h0, r0, x0));
    float stat1 = warp_sum(sum_squares(h1, r1, x1));
    __shared__ float partials[2][4];
    if ((lane & 31) == 0) {
        partials[0][lane >> 5] = stat0;
        partials[1][lane >> 5] = stat1;
    }
    __syncthreads();
    float total0 = (partials[0][0] + partials[0][1]) + (partials[0][2] + partials[0][3]);
    float total1 = (partials[1][0] + partials[1][1]) + (partials[1][2] + partials[1][3]);
    float inv0 = rsqrtf(total0 * (1.0f / 2048) + 1.0e-6f);
    float inv1 = rsqrtf(total1 * (1.0f / 2048) + 1.0e-6f);
    if (next != row)
        store_vec(out + next * 2048 + col, normalize(x1, weights, inv1), keep);
    store_vec(out + row * 2048 + col, normalize(x0, weights, inv0), keep);
}
"""


def _checked(result):
    code, *values = result
    if int(code):
        raise RuntimeError(f"CUDA operation failed with {code}")
    return values[0] if len(values) == 1 else values


@lru_cache(maxsize=None)
def _compile(major, minor, context):
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"fused_rmsnorm.cu", 0, [], []))
    try:
        flags = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        result = nvrtc.nvrtcCompileProgram(program, len(flags), flags)[0]
        if int(result):
            log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace").strip("\x00"))
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))

    module = _checked(driver.cuModuleLoadData(cubin))
    small = _checked(driver.cuModuleGetFunction(module, b"small"))
    pair = _checked(driver.cuModuleGetFunction(module, b"row_pair"))
    return module, small, pair


def _cluster_attr(kind, width):
    attr = driver.CUlaunchAttribute()
    attr.id = kind
    dim = (attr.value.clusterDim if kind == driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
           else attr.value.preferredClusterDim)
    dim.x, dim.y, dim.z = width, 1, 1
    return attr


@triton.jit
def _unaligned_row(H, R, W, Y):
    col = tl.arange(0, 2048)
    offset = tl.program_id(0) * 2048 + col
    h = tl.load(H + offset).to(tl.float32)
    r = tl.load(R + offset).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    x = h + r
    inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048) + 1.0e-6)
    tl.store(Y + offset, ((x * inv) * w).to(tl.bfloat16))


@lru_cache(maxsize=None)
def _launch_attributes(cluster):
    ids = driver.CUlaunchAttributeID
    attrs = [_cluster_attr(ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, 1)]
    if cluster > 1:
        attrs.append(_cluster_attr(ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, cluster))
    return attrs


def run(hidden_states, residual, weight, output):
    batch = hidden_states.shape[0]
    pointers = (hidden_states.data_ptr(), residual.data_ptr(), weight.data_ptr(), output.data_ptr())
    if any(pointer % 32 for pointer in pointers):
        _unaligned_row[(batch,)](hidden_states, residual, weight, output, num_warps=4)
        return

    device = output.device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        context = _checked(driver.cuCtxGetCurrent())
        _, small, pair = _compile(*torch.cuda.get_device_capability(device), int(context))
        if batch == 1:
            kernel, blocks, metadata, cluster = small, 16, 4, 4
        elif batch <= 79:
            cluster = 4 if batch < 48 and batch % 2 == 0 else 1
            kernel, blocks, metadata = small, batch * 2, 1
        else:
            kernel, blocks, metadata, cluster = pair, (batch + 1) // 2, batch, 1

        attrs = _launch_attributes(cluster)
        config = driver.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = blocks, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 128, 1, 1
        config.hStream = stream
        config.attrs = attrs
        config.numAttrs = len(attrs)
        values = (*pointers, metadata)
        types = (c_void_p, c_void_p, c_void_p, c_void_p, c_int)
        _checked(driver.cuLaunchKernelEx(config, kernel, (values, types), 0))
