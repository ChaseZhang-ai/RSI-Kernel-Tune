# SPDX-License-Identifier: Apache-2.0
"""Destination-passing BF16 RMSNorm for rows of width 1536."""

# The padded fallback follows RSI-Kernel-Tune's Triton RMSNorm example.
# CTA replication, prefetch, and cluster scheduling were inspired by
# QianQu/CompoundingKernel-Explorer, FlashInfer/023_rmsnorm_h1536 (Claude).
# Packed FP32 PTX arithmetic follows RSI-Kernel-Tune's 001 RMSNorm.
from ctypes import c_int, c_void_p
from functools import lru_cache

import torch
import triton
import triton.language as tl
from cuda.bindings import driver, nvrtc


_CUDA = r"""
using U32 = unsigned int;
using U64 = unsigned long long;
struct Vec { U32 v[4]; };
constexpr int H = 1536;
constexpr int Width = 8;

__device__ __forceinline__ U64 pair(U32 lo, U32 hi) {
    U64 result;
    asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(lo), "r"(hi));
    return result;
}

__device__ __forceinline__ U64 widen(U32 x) {
    return pair(x << 16, x & 0xffff0000u);
}

__device__ __forceinline__ U64 square_add(U64 x, U64 acc) {
    U64 result;
    asm("fma.rn.f32x2 %0,%1,%1,%2;" : "=l"(result) : "l"(x), "l"(acc));
    return result;
}

__device__ __forceinline__ U64 mul(U64 a, U64 b) {
    U64 result;
    asm("mul.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ float horizontal(U64 x) {
    float lo, hi;
    asm("mov.b64 {%0,%1},%2;" : "=f"(lo), "=f"(hi) : "l"(x));
    return lo + hi;
}

__device__ __forceinline__ U32 round_bf16(U64 x) {
    float lo, hi;
    U32 bits;
    asm("mov.b64 {%0,%1},%2;" : "=f"(lo), "=f"(hi) : "l"(x));
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;" : "=r"(bits) : "f"(hi), "f"(lo));
    return bits;
}

__device__ __forceinline__ Vec read_vec(const unsigned short* p) {
    Vec x;
    asm("ld.global.v4.u32 {%0,%1,%2,%3},[%4];"
        : "=r"(x.v[0]), "=r"(x.v[1]), "=r"(x.v[2]), "=r"(x.v[3]) : "l"(p));
    return x;
}

__device__ __forceinline__ void write_vec(unsigned short* p, Vec y) {
    asm volatile("st.global.v4.u32 [%0],{%1,%2,%3,%4};"
        :: "l"(p), "r"(y.v[0]), "r"(y.v[1]), "r"(y.v[2]), "r"(y.v[3]) : "memory");
}

__device__ __forceinline__ void accumulate(Vec v, U64 (&acc)[4], U64 (&x)[4]) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        x[k] = widen(v.v[k]);
        acc[k] = square_add(x[k], acc[k]);
    }
}

__device__ __forceinline__ Vec normalize(const U64 (&x)[4], Vec w, U64 scale) {
    Vec y;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        y.v[k] = round_bf16(mul(mul(x[k], scale), widen(w.v[k])));
    return y;
}

__device__ __forceinline__ float sum_half_warp(float v) {
    float a = __shfl_xor_sync(0xffffffffu, v, 2);
    float b = __shfl_xor_sync(0xffffffffu, v, 4);
    float c = __shfl_xor_sync(0xffffffffu, v, 6);
    v = (v + a) + (b + c);
    a = __shfl_xor_sync(0xffffffffu, v, 8);
    b = __shfl_xor_sync(0xffffffffu, v, 16);
    c = __shfl_xor_sync(0xffffffffu, v, 24);
    return (v + a) + (b + c);
}

template <int Threads, int Parts, int RowsPerCTA, bool Pdl>
__device__ __forceinline__ void rmsnorm_row(const unsigned short* x, const unsigned short* w,
                                          unsigned short* y, int rows, float* partial) {
    constexpr int Warps = Threads / 32;
    int subgroup = threadIdx.x / Threads;
    int tid = threadIdx.x % Threads;
    int lane = tid % 32;
    int warp = tid / 32;
    int part = blockIdx.x >= rows ? 1 : 0;
    int row = Parts == 1 ? blockIdx.x * RowsPerCTA + subgroup : blockIdx.x - part * rows;
    const unsigned short* base = x + row * H;
    int col = (Parts == 1 ? tid : part * Threads + tid) * Width;

    // Prefetch the row without consuming its values before a PDL dependency.
    if (row < rows && tid < 24)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(base + tid * 64));
    if constexpr (Pdl)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (row >= rows) return;

    // Both inputs may have been written by a preceding kernel on this stream.
    U64 dependency = 0;
    if constexpr (Pdl)
        asm volatile("griddepcontrol.wait; mov.u64 %0,0;" : "=l"(dependency) :: "memory");
    base += dependency;
    Vec weight = read_vec(w + col + dependency);

    U64 values[4];
    U64 acc[4] = {};
    accumulate(read_vec(base + col), acc, values);
    if constexpr (Parts == 2) {
        // Both CTAs reduce the whole row. Each writes only its own half.
        U64 other[4];
        int other_col = ((1 - part) * Threads + tid) * Width;
        accumulate(read_vec(base + other_col), acc, other);
    }

    float sum = (horizontal(acc[0]) + horizontal(acc[1]))
              + (horizontal(acc[2]) + horizontal(acc[3]));
    sum = sum_half_warp(sum);
    // Publish both half-warp sums to a separate shared bank group per warp.
    float* shared = partial + 16 * Warps * subgroup;
    if (lane < 2 * Warps)
        shared[(lane / 2) * 16 + 2 * warp + lane % 2] = sum;
    if constexpr (RowsPerCTA == 2) {
        // An incomplete tail CTA may have only its first row active.
        asm volatile("bar.sync %0,%1;" :: "r"(1 + subgroup), "n"(Threads) : "memory");
    } else {
        __syncthreads();
    }
    U32 addr = static_cast<U32>(__cvta_generic_to_shared(shared + 16 * warp));
    float p0, p1, p2, p3, p4, p5;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3},[%4];"
                 : "=f"(p0), "=f"(p1), "=f"(p2), "=f"(p3) : "r"(addr) : "memory");
    asm volatile("ld.shared.v2.f32 {%0,%1},[%2+16];"
                 : "=f"(p4), "=f"(p5) : "r"(addr) : "memory");
    sum = ((p0 + p1) + (p2 + p3)) + (p4 + p5);
    if constexpr (Warps == 6) {
        float p6, p7, p8, p9, p10, p11;
        asm volatile("ld.shared.v2.f32 {%0,%1},[%2+24];"
                     : "=f"(p6), "=f"(p7) : "r"(addr) : "memory");
        asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3},[%4+32];"
                     : "=f"(p8), "=f"(p9), "=f"(p10), "=f"(p11) : "r"(addr) : "memory");
        sum += (p6 + p7) + ((p8 + p9) + (p10 + p11));
    }
    U32 inv = __float_as_uint(rsqrtf(__fmaf_rn(sum, 1.0f / H, 1.0e-6f)));
    write_vec(y + row * H + col, normalize(values, weight, pair(inv, inv)));
}

#define RMSNORM(NAME, THREADS, PARTS, ROWS, PDL) \
    extern "C" __global__ __launch_bounds__(THREADS * ROWS) \
    void NAME(const unsigned short* x, const unsigned short* w, unsigned short* y, int rows) { \
        __shared__ __align__(16) float partial[16 * (THREADS / 32) * ROWS]; \
        rmsnorm_row<THREADS, PARTS, ROWS, PDL>(x, w, y, rows, partial); \
    }

RMSNORM(rmsnorm_split, 96, 2, 1, false)
RMSNORM(rmsnorm_split_pdl, 96, 2, 1, true)
RMSNORM(rmsnorm_single_pdl, 192, 1, 1, true)
RMSNORM(rmsnorm_pair, 192, 1, 2, false)
"""


def _checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA operation failed with {status}")
    return values[0] if len(values) == 1 else values


@lru_cache(maxsize=None)
def _module(major, minor, context):
    # Cache each binary within the CUDA context it was loaded into.
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"rmsnorm_h1536.cu", 0, [], []))
    try:
        flags = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        status = nvrtc.nvrtcCompileProgram(program, len(flags), flags)[0]
        if int(status):
            log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace").rstrip("\x00"))
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))
    return _checked(driver.cuModuleLoadData(cubin))


@lru_cache(maxsize=None)
def _kernel(major, minor, context, name):
    module = _module(major, minor, context)
    kernel = _checked(driver.cuModuleGetFunction(module, name.encode()))
    attrs = driver.CUfunction_attribute
    if name == "rmsnorm_split":
        _checked(driver.cuFuncSetAttribute(kernel, attrs.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
    if name != "rmsnorm_split_pdl":
        limit = 100 * 1024 if name == "rmsnorm_split" else 48 * 1024
        _checked(driver.cuFuncSetAttribute(kernel, attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, limit))
        _checked(driver.cuFuncSetAttribute(kernel, attrs.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100))
    return kernel


@lru_cache(maxsize=None)
def _launch_attributes(cluster, pdl):
    attrs = []
    ids = driver.CUlaunchAttributeID
    if cluster > 1:
        attr = driver.CUlaunchAttribute()
        attr.id = ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = cluster, 1, 1
        attrs.append(attr)
    if pdl:
        attr = driver.CUlaunchAttribute()
        attr.id = ids.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        attrs.append(attr)
    return attrs


@triton.jit
def _strided_row(x, w, y, xs0, xs1, ws, ys0, ys1):
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    valid = col < 1536
    val = tl.load(x + row * xs0 + col * xs1, valid, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(val * val, 0) * (1.0 / 1536) + 1.0e-6)
    weight = tl.load(w + col * ws, valid, 0).to(tl.float32)
    tl.store(y + row * ys0 + col * ys1, (val * scale) * weight, valid)


@triton.jit
def _exact_rows(x, w, y, rows):
    r = tl.program_id(0) * 2 + tl.arange(0, 2)
    c = tl.arange(0, 512)
    addr = x + r[:, None] * 1536 + c[None, :]
    valid = r[:, None] < rows
    a = tl.load(addr, valid, 0).to(tl.float32)
    b = tl.load(addr + 512, valid, 0).to(tl.float32)
    d = tl.load(addr + 1024, valid, 0).to(tl.float32)
    weight0 = tl.load(w + c).to(tl.float32)
    weight1 = tl.load(w + c + 512).to(tl.float32)
    weight2 = tl.load(w + c + 1024).to(tl.float32)
    inv = tl.rsqrt(tl.sum(a * a + b * b + d * d, 1) * (1.0 / 1536) + 1.0e-6)
    oa = y + r[:, None] * 1536 + c[None, :]
    tl.store(oa, (a * inv[:, None]) * weight0[None, :], valid)
    tl.store(oa + 512, (b * inv[:, None]) * weight1[None, :], valid)
    tl.store(oa + 1024, (d * inv[:, None]) * weight2[None, :], valid)


# Cluster width and per-CTA dynamic shared memory control block placement.
_SPLIT_CLUSTERS = {7: (7, 48 * 1024), 18: (9, 48 * 1024), 64: (16, 100 * 1024)}


def _native_launch(x, w, y, rows):
    if rows == 1:
        name, threads, blocks, cluster, shared, pdl = "rmsnorm_split_pdl", 96, 2, 2, 0, True
    elif rows == 32:
        name, threads, blocks, cluster, shared, pdl = "rmsnorm_single_pdl", 192, rows, 8, 48 * 1024, True
    elif rows <= 128:
        cluster, shared = _SPLIT_CLUSTERS.get(rows, (1, 0))
        name, threads, blocks, pdl = "rmsnorm_split", 96, rows * 2, False
    else:
        name, threads, blocks = "rmsnorm_pair", 384, triton.cdiv(rows, 2)
        cluster, shared, pdl = 2, 48 * 1024, False

    device = y.device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        context = int(_checked(driver.cuCtxGetCurrent()))
        kernel = _kernel(*torch.cuda.get_device_capability(device), context, name)
        args = ((x.data_ptr(), w.data_ptr(), y.data_ptr(), rows), (c_void_p, c_void_p, c_void_p, c_int))
        attrs = _launch_attributes(cluster, pdl)
        if attrs:
            config = driver.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = triton.cdiv(blocks, cluster) * cluster, 1, 1
            config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
            config.sharedMemBytes, config.hStream = shared, stream
            config.attrs, config.numAttrs = attrs, len(attrs)
            _checked(driver.cuLaunchKernelEx(config, kernel, args, 0))
        else:
            _checked(driver.cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, stream, args, 0))


def run(hidden_states, weight, output):
    x, w, y = hidden_states, weight, output
    rows = x.shape[0]
    if not rows:
        return
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (x, w, y)):
        _strided_row[(rows,)](x, w, y, *x.stride(), w.stride(0), *y.stride(), num_warps=4)
    elif rows <= 1024:
        _native_launch(x, w, y, rows)
    else:
        _exact_rows[(triton.cdiv(rows, 2),)](x, w, y, rows, num_warps=2)
