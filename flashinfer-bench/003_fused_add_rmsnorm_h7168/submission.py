# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 add and RMSNorm at hidden size 7168.

The replicated small-batch reduction and wide memory operations were inspired by
QianQu's CompoundingKernel-Explorer, FlashInfer/003_fused_add_rmsnorm_h7168.
Load-balanced CTA scheduling follows the same published implementation.
The unaligned route adapts the Triton RMSNorm example in RSI-Kernel-Tune.
"""

from ctypes import c_void_p
from functools import lru_cache

import torch
import triton
import triton.language as tl
from cuda.bindings import driver, nvrtc

_CUDA = r"""
using U16 = unsigned short;
using U32 = unsigned int;
using U64 = unsigned long long;
constexpr int H = 7168;

struct Bf16Vector { U32 words[8]; };

__device__ __forceinline__ Bf16Vector load(const U16* p) {
    Bf16Vector v;
    asm("ld.global.L1::evict_last.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v.words[0]), "=r"(v.words[1]), "=r"(v.words[2]), "=r"(v.words[3]),
          "=r"(v.words[4]), "=r"(v.words[5]), "=r"(v.words[6]), "=r"(v.words[7]) : "l"(p));
    return v;
}

__device__ __forceinline__ void store(U16* p, const Bf16Vector& v) {
    asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "l"(p), "r"(v.words[0]), "r"(v.words[1]), "r"(v.words[2]), "r"(v.words[3]),
           "r"(v.words[4]), "r"(v.words[5]), "r"(v.words[6]), "r"(v.words[7]) : "memory");
}

__device__ __forceinline__ U64 pair(U32 l, U32 h) {
    U64 result;
    asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(l), "r"(h));
    return result;
}

__device__ __forceinline__ U64 widen(U32 v) {
    return pair(v << 16, v & 0xffff0000u);
}

__device__ __forceinline__ U64 add(U64 a, U64 b) {
    U64 result;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ U64 mul(U64 a, U64 b) {
    U64 result;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ U64 square_accumulate(U64 a, U64 c) {
    U64 result;
    asm("fma.rn.f32x2 %0, %1, %1, %2;" : "=l"(result) : "l"(a), "l"(c));
    return result;
}

__device__ __forceinline__ float horizontal(U64 v) {
    float a, b;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(a), "=f"(b) : "l"(v));
    return a + b;
}

__device__ __forceinline__ U32 narrow(U64 v) {
    float a, b;
    U32 result;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(a), "=f"(b) : "l"(v));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(b), "f"(a));
    return result;
}

template <int Warps>
__device__ __forceinline__ float reduce(float value, float* partials) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, delta);
    if (lane == 0)
        partials[warp] = value;
    __syncthreads();

    float sum[Warps];
    #pragma unroll
    for (int j = 0; j < Warps; ++j)
        sum[j] = partials[j];
    #pragma unroll
    for (int step = 1; step < Warps; step *= 2) {
        #pragma unroll
        for (int j = 0; j < Warps; j += step * 2)
            if (j + step < Warps)
                sum[j] += sum[j + step];
    }
    return sum[0];
}

template <int Threads, int Parts>
__device__ __forceinline__ void fused(const U16* h, const U16* r, const U16* w, U16* out) {
    constexpr int VectorSize = 16;
    constexpr int Tiles = H / (Threads * VectorSize);
    constexpr int Slice = H / Parts;
    static_assert(H % (Threads * VectorSize) == 0 && H % (Parts * VectorSize) == 0);

    const int base = blockIdx.x * H;
    const int start = blockIdx.y * Slice;
    int column[Tiles];
    bool owns_output[Tiles];
    Bf16Vector weights[Tiles];
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int local = (tile * Threads + threadIdx.x) * VectorSize;
        // Each replicated block reduces the whole row and writes one stripe.
        const int rotated = local + start;
        column[tile] = rotated < H ? rotated : rotated - H;
        owns_output[tile] = local < Slice;
        if constexpr (Parts > 1) {
            if (owns_output[tile])
                weights[tile] = load(w + column[tile]);
        }
    }

    U64 values[Tiles][8];
    U64 sums[2] = {0, 0};
    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int offset = base + column[tile];
        const auto hidden = load(h + offset);
        const auto residual = load(r + offset);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[tile][j] = add(widen(hidden.words[j]), widen(residual.words[j]));
            sums[j % 2] = square_accumulate(values[tile][j], sums[j % 2]);
        }
    }
    __shared__ float partials[Threads / 32];
    const float total = reduce<Threads / 32>(horizontal(add(sums[0], sums[1])), partials);
    const U32 inv = __float_as_uint(rsqrtf(total * (1.0f / H) + 1e-6f));
    const U64 scale = pair(inv, inv);

    #pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        if (owns_output[tile]) {
            if constexpr (Parts == 1)
                weights[tile] = load(w + column[tile]);
            Bf16Vector result;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                result.words[j] = narrow(mul(mul(values[tile][j], scale),
                                             widen(weights[tile].words[j])));
            }
            store(out + base + column[tile], result);
        }
    }
}

extern "C" __global__ __launch_bounds__(448, 2)
void split_four(const U16* h, const U16* r, const U16* w, U16* out) {
    fused<448, 4>(h, r, w, out);
}

extern "C" __global__ __launch_bounds__(448, 2)
void split_two(const U16* h, const U16* r, const U16* w, U16* out) {
    fused<448, 2>(h, r, w, out);
}

extern "C" __global__ __launch_bounds__(224, 4)
void whole_row(const U16* h, const U16* r, const U16* w, U16* out) {
    fused<224, 1>(h, r, w, out);
}
"""


def _checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA operation failed with {status}")
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
            raise RuntimeError(log.decode(errors="replace").rstrip("\x00"))
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))

    module = _checked(driver.cuModuleLoadData(cubin))
    functions = {
        name: _checked(driver.cuModuleGetFunction(module, name.encode()))
        for name in ("split_four", "split_two", "whole_row")
    }
    return module, functions


@triton.jit
def _unaligned_row(hidden, residual, weight, output):
    col = tl.arange(0, 8192)
    offset = tl.program_id(0) * 7168 + col
    x = tl.load(hidden + offset, col < 7168, 0).to(tl.float32)
    r = tl.load(residual + offset, col < 7168, 0).to(tl.float32)
    w = tl.load(weight + col, col < 7168, 0).to(tl.float32)
    v = x + r
    inv_rms = tl.rsqrt(tl.sum(v * v, 0) * (1.0 / 7168) + 1e-6)
    tl.store(output + offset, (v * inv_rms) * w, col < 7168)


@lru_cache(maxsize=None)
def _cluster_attrs(stripes):
    ids = driver.CUlaunchAttributeID
    cluster = driver.CUlaunchAttribute()
    cluster.id = ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    cluster.value.clusterDim.x = 1
    cluster.value.clusterDim.y = stripes
    cluster.value.clusterDim.z = 1

    policy = driver.CUlaunchAttribute()
    policy.id = ids.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
    policy.value.clusterSchedulingPolicyPreference = (
        driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING
    )
    return [cluster, policy]


def run(hidden_states, residual, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return
    pointers = (hidden_states.data_ptr(), residual.data_ptr(), weight.data_ptr(), output.data_ptr())
    if any(pointer % 32 for pointer in pointers):
        _unaligned_row[(batch,)](hidden_states, residual, weight, output, num_warps=8)
        return

    # Replicate reductions only while a row per CTA leaves the GPU underfilled.
    if batch <= 40:
        kernel, threads, parts = "split_four", 448, 4
    elif batch <= 64:
        kernel, threads, parts = "split_two", 448, 2
    else:
        kernel, threads, parts = "whole_row", 224, 1

    device = output.device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        context = _checked(driver.cuCtxGetCurrent())
        _, functions = _compile(*torch.cuda.get_device_capability(device), int(context))
        parameters = (pointers, (c_void_p,) * 4)
        if parts == 1:
            _checked(driver.cuLaunchKernel(
                functions[kernel], batch, 1, 1, threads, 1, 1, 0, stream, parameters, 0,
            ))
            return

        # Co-schedule the four stripes of a row at low batch sizes to share L2 reads.
        # At higher short-batch occupancy, let the scheduler balance CTAs individually.
        attrs = _cluster_attrs(parts if batch <= 24 else 1)
        config = driver.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = batch, parts, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
        config.hStream = stream
        config.attrs = attrs
        config.numAttrs = len(attrs)
        _checked(driver.cuLaunchKernelEx(config, functions[kernel], parameters, 0))
