"""Standalone BF16 RMSNorm with FP32 arithmetic and a DPS entry point."""

from ctypes import c_void_p
from functools import lru_cache

import torch
import triton
import triton.language as tl
from cuda.bindings import driver, nvrtc


# Packed FP32 operations and wide memory accesses build on RSI-Kernel-Tune's
# 001_fused_add_rmsnorm_h2048 candidate. CompoundingKernel-Explorer's FlashInfer
# 025 kernel inspired the replicated single-row reduction.
_CUDA = r"""
using U32 = unsigned int;
using U64 = unsigned long long;


struct Packed { U32 word[8]; };

template<bool WEIGHT>
__device__ __forceinline__ Packed load(const unsigned short* address) {
    Packed result;
    if constexpr (WEIGHT) {
        asm volatile("ld.global.nc.L1::evict_last.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(result.word[0]), "=r"(result.word[1]), "=r"(result.word[2]), "=r"(result.word[3]),
              "=r"(result.word[4]), "=r"(result.word[5]), "=r"(result.word[6]), "=r"(result.word[7])
            : "l"(address));
    } else {
        asm volatile("ld.global.nc.L1::no_allocate.L2::128B.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(result.word[0]), "=r"(result.word[1]), "=r"(result.word[2]), "=r"(result.word[3]),
              "=r"(result.word[4]), "=r"(result.word[5]), "=r"(result.word[6]), "=r"(result.word[7])
            : "l"(address));
    }
    return result;
}

__device__ __forceinline__ void store(unsigned short* address, const Packed& value) {
    asm volatile("st.global.L1::no_allocate.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "l"(address), "r"(value.word[0]), "r"(value.word[1]), "r"(value.word[2]), "r"(value.word[3]),
           "r"(value.word[4]), "r"(value.word[5]), "r"(value.word[6]), "r"(value.word[7]) : "memory");
}

__device__ __forceinline__ U64 join(U32 low, U32 high) {
    U64 result;
    asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(low), "r"(high));
    return result;
}

__device__ __forceinline__ U64 widen(U32 bf16) {
    return join(bf16 << 16, bf16 & 0xffff0000u);
}

__device__ __forceinline__ U64 multiply(U64 a, U64 b) {
    U64 result;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ U64 add(U64 a, U64 b) {
    U64 result;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ float horizontal_sum(U64 v) {
    float low, high;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(low), "=f"(high) : "l"(v));
    return low + high;
}

__device__ __forceinline__ U32 narrow(U64 v) {
    float low, high;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(low), "=f"(high) : "l"(v));
    U32 result;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(high), "f"(low));
    return result;
}


template<int PARTITIONS>
__device__ __forceinline__ void normalize(const unsigned short* x, const unsigned short* w,
                                           unsigned short* y) {
    const int column = threadIdx.x * 16;
    const int offset = blockIdx.x * 4096 + column;
    const bool owns_output = PARTITIONS == 1 || column / (4096 / PARTITIONS) == blockIdx.y;
    __shared__ float warp_sums[8];

    // Issue weight loads early and keep the packed activation across the barrier.
    Packed weight;
    if (owns_output) {
        weight = load<true>(w + column);
    }
    Packed values = load<false>(x + offset);
    U64 squares[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        U64 v = widen(values.word[i]);
        squares[i] = multiply(v, v);
    }
    #pragma unroll
    for (int stride = 4; stride > 0; stride /= 2) {
        #pragma unroll
        for (int i = 0; i < stride; ++i) {
            squares[i] = add(squares[i], squares[i + stride]);
        }
    }
    float sum = horizontal_sum(squares[0]);
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, stride);
    }
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    if (owns_output) {
        float left = (warp_sums[0] + warp_sums[1]) + (warp_sums[2] + warp_sums[3]);
        float right = (warp_sums[4] + warp_sums[5]) + (warp_sums[6] + warp_sums[7]);
        float inv = rsqrtf((left + right) * (1.0f / 4096.0f) + 1e-5f);
        U64 inv_pair = join(__float_as_uint(inv), __float_as_uint(inv));
        Packed out;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            U64 normalized = multiply(widen(values.word[i]), inv_pair);
            out.word[i] = narrow(multiply(normalized, widen(weight.word[i])));
        }
        store(y + offset, out);
    }
}

extern "C" __global__ __launch_bounds__(256) void one_row(
    const unsigned short* x, const unsigned short* w, unsigned short* y) {
    normalize<1>(x, w, y);
}

extern "C" __global__ __launch_bounds__(256) void split_row(
    const unsigned short* x, const unsigned short* w, unsigned short* y) {
    normalize<8>(x, w, y);
}
"""

def _checked(result):
    code, *values = result
    if int(code):
        raise RuntimeError(f"CUDA operation failed with {code}")
    return values[0] if len(values) == 1 else values


@lru_cache(maxsize=None)
def _compile(major, minor, context):
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"rmsnorm_4096.cu", 0, [], []))
    try:
        flags = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
        if int(nvrtc.nvrtcCompileProgram(program, len(flags), flags)[0]):
            log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors="replace").strip("\x00"))
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))
    module = _checked(driver.cuModuleLoadData(cubin))
    one_row = _checked(driver.cuModuleGetFunction(module, b"one_row"))
    split_row = _checked(driver.cuModuleGetFunction(module, b"split_row"))
    return module, one_row, split_row


# Adapted from kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py.
@triton.jit
def _rmsnorm(hidden_states, weight, output, X0: tl.constexpr, X1: tl.constexpr,
             W0: tl.constexpr, Y0: tl.constexpr, Y1: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    x = tl.load(hidden_states + row * X0 + col * X1).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 4096) + 1e-5)
    w = tl.load(weight + col * W0).to(tl.float32)
    tl.store(output + row * Y0 + col * Y1, ((x * inv_rms) * w).to(tl.bfloat16))


def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return

    device = output.device
    tensors = (hidden_states, weight, output)
    capability = torch.cuda.get_device_capability(device)
    pointers = tuple(t.data_ptr() for t in tensors)
    if (batch > 256 or capability[0] != 10 or any(p % 32 for p in pointers)
            or not all(t.is_contiguous() for t in tensors)):
        _rmsnorm[(batch,)](*tensors, *hidden_states.stride(), weight.stride(0),
                           *output.stride(), num_warps=4)
        return

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        context = _checked(driver.cuCtxGetCurrent())
        _, one_row, split_row = _compile(*capability, int(context))
        kernel, partitions = (split_row, 8) if batch == 1 else (one_row, 1)
        args = (pointers, (c_void_p, c_void_p, c_void_p))
        _checked(driver.cuLaunchKernel(kernel, batch, partitions, 1, 256, 1, 1, 0,
                                       stream, args, 0))
