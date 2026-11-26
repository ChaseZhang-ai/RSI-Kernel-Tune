"""Destination-passing BF16 RMSNorm for hidden size 2048.

Reduction and cache layout ideas from QianQu's CompoundingKernel Explorer,
FlashInfer/024_rmsnorm_h2048. The strided fallback follows the repository's
sol-example-triton-rmsnorm starting point.
"""
from ctypes import c_int, c_void_p
from functools import lru_cache

from cuda.bindings import driver, nvrtc
import torch
import triton
import triton.language as tl


_CUDA = r"""
using U32 = unsigned int;
using U64 = unsigned long long;

template <int Words>
struct Vec {
    U32 data[Words];
};

template <int Words, int Policy>
__device__ __forceinline__ Vec<Words> read_vec(const unsigned short* ptr) {
    Vec<Words> v;
    if constexpr (Words == 8) {
        if constexpr (Policy == 1) {
            asm volatile("ld.global.nc.L1::evict_last.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(v.data[0]), "=r"(v.data[1]), "=r"(v.data[2]), "=r"(v.data[3]),
                  "=r"(v.data[4]), "=r"(v.data[5]), "=r"(v.data[6]), "=r"(v.data[7])
                : "l"(ptr));
        } else if constexpr (Policy == 2) {
            asm volatile("ld.global.L1::no_allocate.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(v.data[0]), "=r"(v.data[1]), "=r"(v.data[2]), "=r"(v.data[3]),
                  "=r"(v.data[4]), "=r"(v.data[5]), "=r"(v.data[6]), "=r"(v.data[7])
                : "l"(ptr));
        } else {
            asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(v.data[0]), "=r"(v.data[1]), "=r"(v.data[2]), "=r"(v.data[3]),
                  "=r"(v.data[4]), "=r"(v.data[5]), "=r"(v.data[6]), "=r"(v.data[7])
                : "l"(ptr));
        }
    } else {
        if constexpr (Policy == 1) {
            asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v.data[0]), "=r"(v.data[1]), "=r"(v.data[2]), "=r"(v.data[3]) : "l"(ptr));
        } else {
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v.data[0]), "=r"(v.data[1]), "=r"(v.data[2]), "=r"(v.data[3]) : "l"(ptr));
        }
    }
    return v;
}

template <int Words, bool Stream>
__device__ __forceinline__ void write_vec(unsigned short* ptr, Vec<Words> v) {
    if constexpr (Words == 8) {
        if constexpr (Stream) {
            asm volatile("st.global.L1::no_allocate.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                :: "l"(ptr), "r"(v.data[0]), "r"(v.data[1]), "r"(v.data[2]), "r"(v.data[3]),
                   "r"(v.data[4]), "r"(v.data[5]), "r"(v.data[6]), "r"(v.data[7]) : "memory");
        } else {
            asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                :: "l"(ptr), "r"(v.data[0]), "r"(v.data[1]), "r"(v.data[2]), "r"(v.data[3]),
                   "r"(v.data[4]), "r"(v.data[5]), "r"(v.data[6]), "r"(v.data[7]) : "memory");
        }
    } else {
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
            :: "l"(ptr), "r"(v.data[0]), "r"(v.data[1]), "r"(v.data[2]), "r"(v.data[3]) : "memory");
    }
}

__device__ __forceinline__ U64 merge(U32 a, U32 b) {
    U64 p;
    asm("mov.b64 %0, {%1,%2};" : "=l"(p) : "r"(a), "r"(b));
    return p;
}

__device__ __forceinline__ U64 widen(U32 p) {
    return merge(p << 16, p & 0xffff0000u);
}

template <int Words>
__device__ __forceinline__ float sum_squares(Vec<Words> v) {
    U64 even = 0;
    U64 odd = 0;
    #pragma unroll
    for (int i = 0; i < Words; i += 2) {
        U64 a = widen(v.data[i]);
        U64 b = widen(v.data[i + 1]);
        asm("fma.rn.f32x2 %0, %1, %1, %0;" : "+l"(even) : "l"(a));
        asm("fma.rn.f32x2 %0, %1, %1, %0;" : "+l"(odd) : "l"(b));
    }
    U64 p;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(p) : "l"(even), "l"(odd));
    float a, b;
    asm("mov.b64 {%0,%1}, %2;" : "=f"(a), "=f"(b) : "l"(p));
    return a + b;
}

template <int Words>
__device__ __forceinline__ Vec<Words> normalize(Vec<Words> x, Vec<Words> w, float inv) {
    U64 scale = merge(__float_as_uint(inv), __float_as_uint(inv));
    Vec<Words> y;
    #pragma unroll
    for (int i = 0; i < Words; ++i) {
        U64 p, q;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(p) : "l"(widen(x.data[i])), "l"(scale));
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(q) : "l"(p), "l"(widen(w.data[i])));
        float a, b;
        asm("mov.b64 {%0,%1}, %2;" : "=f"(a), "=f"(b) : "l"(q));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(y.data[i]) : "f"(b), "f"(a));
    }
    return y;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int step = 16; step > 0; step /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, step);
    }
    return val;
}

template <int Warps>
__device__ __forceinline__ float total(const float* p) {
    float a = (p[0] + p[1]) + (p[2] + p[3]);
    if constexpr (Warps == 8) {
        a += (p[4] + p[5]) + (p[6] + p[7]);
    }
    return a;
}


// One prefetch CTA and two normalization CTAs per row. The two halves
// have disjoint stores, and each CTA computes the complete FP32 statistic.
extern "C" __global__ __launch_bounds__(256)
void small_rows(const unsigned short* x, const unsigned short* w,
                unsigned short* y, int rows) {
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    if (block < rows) {
        if (tid == 0) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 4096;"
                         :: "l"(x + block * 2048));
            if (block == 0) {
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 4096;" :: "l"(w));
            }
        }
        return;
    }

    const bool high_half = block >= 2 * rows;
    const int row = block - (high_half ? 2 * rows : rows);
    const int col = tid * 8;
    const bool write = (tid >= 128) == high_half;
    Vec<4> weights;
    if (write) {
        weights = read_vec<4, 1>(w + col);
    }
    const auto v = read_vec<4, 0>(x + row * 2048 + col);
    const float sum = warp_sum(sum_squares(v));
    __shared__ float partial[8];
    if (tid % 32 == 0) {
        partial[tid / 32] = sum;
    }
    __syncthreads();
    if (write) {
        const float mean = total<8>(partial) * (1.0f / 2048);
        const float inv = rsqrtf(mean + 1e-6f);
        write_vec<4, false>(y + row * 2048 + col, normalize(v, weights, inv));
    }
}

// The two rows share one weight vector per thread.
extern "C" __global__ __launch_bounds__(128) void pair_reuse(const unsigned short* x, const unsigned short* w,
                                           unsigned short* y, int rows) {
    constexpr int Rows = 2;
    const int row0 = blockIdx.x * Rows;
    const int tid = threadIdx.x;
    const int col = tid * 16;
    const auto weights = read_vec<8, 1>(w + col);
    Vec<8> activations[Rows];
    float sums[Rows];
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        int row = min(row0 + i, rows - 1);
        activations[i] = read_vec<8, 0>(x + row * 2048 + col);
    }
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        sums[i] = warp_sum(sum_squares(activations[i]));
    }
    __shared__ float partial[Rows][4];
    if (tid % 32 == 0) {
        #pragma unroll
        for (int i = 0; i < Rows; ++i)
            partial[i][tid / 32] = sums[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        if (row0 + i < rows) {
            float mean = total<4>(partial[i]) * (1.0f / 2048);
            float inv = rsqrtf(mean + 1e-6f);
            write_vec<8, false>(y + (row0 + i) * 2048 + col,
                                 normalize(activations[i], weights, inv));
        }
    }
}


// Two row groups reduce independently. Prefetch an upcoming 8 KiB tile
// into L2 and bypass L1 allocation for the activation and output streams.
extern "C" __global__ __launch_bounds__(256)
void pair_stream(const unsigned short* x, const unsigned short* w,
                 unsigned short* y, int rows) {
    constexpr int PrefetchTiles = 8 * 148;
    const int tile = gridDim.x - 1 - blockIdx.x;
    const int local_row = threadIdx.x / 128;
    const int row = tile * 2 + local_row;
    const int tid = threadIdx.x % 128;
    const int col = tid * 16;
    if (threadIdx.x == 0 && tile >= PrefetchTiles) {
        const int next_row = (tile - PrefetchTiles) * 2;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 8192;"
                     :: "l"(x + next_row * 2048));
    }
    const auto weights = read_vec<8, 1>(w + col);
    const auto v = read_vec<8, 2>(x + min(row, rows - 1) * 2048 + col);
    const float sum = warp_sum(sum_squares(v));
    __shared__ float partial[2][4];
    if (tid % 32 == 0) {
        partial[local_row][tid / 32] = sum;
    }
    __syncthreads();
    if (row < rows) {
        const float mean = total<4>(partial[local_row]) * (1.0f / 2048);
        const float inv = rsqrtf(mean + 1e-6f);
        write_vec<8, true>(y + row * 2048 + col, normalize(v, weights, inv));
    }
}
"""


def _checked(result):
    code, *values = result
    if int(code):
        raise RuntimeError(f"CUDA operation failed with {code}")
    return values[0] if len(values) == 1 else values


@lru_cache(maxsize=None)
def _compile(major, minor, context):
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"rmsnorm024.cu", 0, [], []))
    try:
        flags = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17",
                 b"--fmad=false"]
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
def _kernel(module, name):
    return _checked(driver.cuModuleGetFunction(module, name.encode()))

@triton.jit
def _strided_row(X, W, Y, X_ROW, X_COL: tl.constexpr, W_COL: tl.constexpr,
                 Y_ROW, Y_COL: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    x = tl.load(X + row * X_ROW + col * X_COL).to(tl.float32)
    w = tl.load(W + col * W_COL).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / 2048) + 1e-6)
    tl.store(Y + row * Y_ROW + col * Y_COL, ((x * inv) * w).to(tl.bfloat16))


def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return

    tensors = (hidden_states, weight, output)
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    device = hidden_states.device
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 10 or any(pointer % 32 for pointer in pointers) or not all(
            tensor.is_contiguous() for tensor in tensors
        ):
            _strided_row[(batch,)](
                *tensors, *hidden_states.stride(), weight.stride(0), *output.stride(),
                num_warps=4, enable_fp_fusion=False,
            )
            return

        if batch <= 79:
            name, blocks, threads = "small_rows", 3 * batch, 256
        elif batch < 16000:
            name, blocks, threads = "pair_reuse", (batch + 1) // 2, 128
        else:
            name, blocks, threads = "pair_stream", (batch + 1) // 2, 256

        stream = torch.cuda.current_stream(device).cuda_stream
        context = _checked(driver.cuCtxGetCurrent())
        module = _compile(*capability, int(context))
        kernel = _kernel(module, name)
        args = (*pointers, batch)
        types = (c_void_p, c_void_p, c_void_p, c_int)
        _checked(driver.cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0,
                                       stream, (args, types), 0))
