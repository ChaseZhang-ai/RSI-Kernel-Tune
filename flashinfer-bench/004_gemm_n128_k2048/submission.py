"""Destination-passing FP16 GEMM for N=128 and K=2048.

Register-fed MMA uses the shared K permutation described by CompoundingKernel
Explorer (FlashInfer 004). The large-tile staging and persistent scheduling
follow the repository's H100 GEMM and Triton's warp-specialization examples.
"""


import ctypes
import inspect
import shutil
import subprocess
import tempfile
import torch
import triton
from functools import lru_cache
from pathlib import Path
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as bw
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDescriptor


_CUDA_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint4 load128(const __half* ptr) {
    uint4 value;
    asm volatile("ld.global.nc.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr));
    return value;
}

__device__ __forceinline__ void mma(float4& out,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(out.x), "+f"(out.y), "+f"(out.z), "+f"(out.w)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 load_row(const __half* ptr, int row, int m, int k) {
    return row < m ? load128(ptr + row * 2048 + k) : uint4{};
}

__device__ __forceinline__ void add_to(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

template<int BM, int W, int BN, bool Parallel = false>
__global__ void register_mma(const __half* __restrict__ A, const __half* __restrict__ B,
                             __half* __restrict__ C, int m) {
    constexpr int CHUNKS = 64 / W;
    constexpr int GROUP = CHUNKS >= 8 ? 8 : CHUNKS;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int mr = blockIdx.y * BM;
    const int nc = blockIdx.x * BN;
    const int k = warp * CHUNKS * 32 + t * 8;
    const int row = mr + g;
    float4 acc{};
    for (int base = 0; base < CHUNKS; base += GROUP) {
        uint4 lhs0[GROUP], lhs1[GROUP], rhs[GROUP];
        // The same K permutation for both operands lets each lane load
        // eight consecutive FP16 values for two MMA instructions.
        #pragma unroll
        for (int i = 0; i < GROUP; ++i) {
            const int offset = k + (base + i) * 32;
            rhs[i] = load128(B + (nc + g % BN) * 2048 + offset);
            lhs0[i] = load_row(A, row, m, offset);
            if constexpr (BM > 8) lhs1[i] = load_row(A, row + 8, m, offset);
            else lhs1[i] = uint4{};
        }
        #pragma unroll
        for (int i = 0; i < GROUP; ++i) {
            const uint4 a = lhs0[i], b = lhs1[i], c = rhs[i];
            mma(acc, a.x, b.x, a.y, b.y, c.x, c.y);
            mma(acc, a.z, b.z, a.w, b.w, c.z, c.w);
        }
    }

    __shared__ float4 partial[W][32];
    partial[warp][lane] = acc;
    __syncthreads();
    if constexpr (Parallel) {
        // Four lanes share the warp-partial reduction of each output pair.
        const int pair = threadIdx.x / 4;
        const int part = threadIdx.x % 4;
        if (pair < BM * (BN / 2)) {
            const int r = pair / (BN / 2);
            const int n = pair % (BN / 2);
            const int owner = (r % 8) * 4 + n;
            float4 sum = partial[part][owner];
            #pragma unroll
            for (int w = 4; w < W; w += 4) add_to(sum, partial[part + w][owner]);
            float x = r < 8 ? sum.x : sum.z;
            float y = r < 8 ? sum.y : sum.w;
            #pragma unroll
            for (int delta = 2; delta > 0; delta /= 2) {
                x += __shfl_xor_sync(0xffffffffu, x, delta, 4);
                y += __shfl_xor_sync(0xffffffffu, y, delta, 4);
            }
            if (part == 0 && mr + r < m)
                *reinterpret_cast<__half2*>(C + (mr + r) * 128 + nc + 2 * n) = __floats2half2_rn(x, y);
        }
    } else {
        if (warp == 0) {
            #pragma unroll
            for (int i = 1; i < W; ++i) add_to(acc, partial[i][lane]);
        }

        if (warp == 0 && 2 * t < BN) {
            const int n = nc + 2 * t;
            if (row < m)
                *reinterpret_cast<__half2*>(C + row * 128 + n) = __floats2half2_rn(acc.x, acc.y);
            if constexpr (BM > 8) {
                if (row + 8 < m)
                    *reinterpret_cast<__half2*>(C + (row + 8) * 128 + n) = __floats2half2_rn(acc.z, acc.w);
            }
        }
    }
}

template<int BM, int W, int BN = 8, bool Parallel = false>
cudaError_t launch(const __half* a, const __half* b, __half* c, int m, cudaStream_t stream) {
    const dim3 grid(128 / BN, (m + BM - 1) / BM);
    register_mma<BM, W, BN, Parallel><<<grid, W * 32, 0, stream>>>(a, b, c, m);
    return cudaGetLastError();
}

__device__ __forceinline__ void multiply_pair(float& lo, float& hi, unsigned a, unsigned b) {
    asm("{ .reg .f16 al, ah, bl, bh; "
        "mov.b32 {al, ah}, %2; mov.b32 {bl, bh}, %3; "
        "fma.rn.f32.f16 %0, al, bl, %0; fma.rn.f32.f16 %1, ah, bh, %1; }"
        : "+f"(lo), "+f"(hi) : "r"(a), "r"(b));
}

template<int R>
__global__ void vector_gemv(const __half* __restrict__ A, const __half* __restrict__ B,
                            __half* __restrict__ C, int m) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int col = blockIdx.x;
    const int row = blockIdx.y * R;
    const int k = threadIdx.x * 8;
    const uint4 weights = load128(B + col * 2048 + k);
    __shared__ float partial[8][R];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        const uint4 values = load_row(A, row + i, m, k);
        float lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
        multiply_pair(lo0, hi0, values.x, weights.x);
        multiply_pair(lo1, hi1, values.y, weights.y);
        multiply_pair(lo0, hi0, values.z, weights.z);
        multiply_pair(lo1, hi1, values.w, weights.w);
        float value = (lo0 + lo1) + (hi0 + hi1);
        #pragma unroll
        for (int step = 16; step > 0; step /= 2)
            value += __shfl_xor_sync(0xffffffffu, value, step);
        if (lane == 0) partial[warp][i] = value;
    }
    __syncthreads();
    if (threadIdx.x < R && row + threadIdx.x < m) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) value += partial[i][threadIdx.x];
        C[(row + threadIdx.x) * 128 + col] = __float2half_rn(value);
    }
}

template<int R>
cudaError_t launch_vector(const __half* a, const __half* b, __half* c, int m, cudaStream_t stream) {
    vector_gemv<R><<<dim3(128, (m + R - 1) / R), 256, 0, stream>>>(a, b, c, m);
    return cudaGetLastError();
}

template<int BM, int BN, int W>
__global__ void wide_mma(const __half* __restrict__ A, const __half* __restrict__ B,
                         __half* __restrict__ C, int m) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int group = lane / 4;
    const int row = blockIdx.y * BM + group;
    const int col = blockIdx.x * BN;
    const int k = warp * (2048 / W) + (lane % 4) * 8;
    float4 acc[BM / 16][BN / 8]{};
    #pragma unroll 4
    for (int step = 0; step < 64 / W; ++step) {
        uint4 a[BM / 8], b[BN / 8];
        #pragma unroll
        for (int i = 0; i < BM / 8; ++i)
            a[i] = load_row(A, row + i * 8, m, k + step * 32);
        #pragma unroll
        for (int j = 0; j < BN / 8; ++j)
            b[j] = load128(B + (col + group + j * 8) * 2048 + k + step * 32);
        #pragma unroll
        for (int i = 0; i < BM / 16; ++i) {
            const uint4 low = a[2 * i], high = a[2 * i + 1];
            #pragma unroll
            for (int j = 0; j < BN / 8; ++j) {
                const uint4 rhs = b[j];
                mma(acc[i][j], low.x, high.x, low.y, high.y, rhs.x, rhs.y);
                mma(acc[i][j], low.z, high.z, low.w, high.w, rhs.z, rhs.w);
            }
        }
    }

    __shared__ float4 partial[BM / 16][BN / 8][W][32];
    #pragma unroll
    for (int i = 0; i < BM / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < BN / 8; ++j)
            partial[i][j][warp][lane] = acc[i][j];
    }
    __syncthreads();
    // Give each output fragment to a different warp for the epilogue.
    for (int fragment = warp; fragment < (BM / 16) * (BN / 8); fragment += W) {
        const int i = fragment / (BN / 8);
        const int j = fragment % (BN / 8);
        float4 value{};
        #pragma unroll
        for (int w = 0; w < W; ++w) add_to(value, partial[i][j][w][lane]);
        const int r = row + i * 16;
        const int n = col + j * 8 + 2 * (lane % 4);
        if (r < m)
            *reinterpret_cast<__half2*>(C + r * 128 + n) = __floats2half2_rn(value.x, value.y);
        if (r + 8 < m)
            *reinterpret_cast<__half2*>(C + (r + 8) * 128 + n) = __floats2half2_rn(value.z, value.w);
    }
}

template<int BM, int BN, int W>
cudaError_t launch_wide(const __half* a, const __half* b, __half* c, int m, cudaStream_t stream) {
    wide_mma<BM, BN, W><<<dim3(128 / BN, (m + BM - 1) / BM), W * 32, 0, stream>>>(a, b, c, m);
    return cudaGetLastError();
}


extern "C" int dispatch(const __half* a, const __half* b, __half* c, int m,
                        cudaStream_t stream, int variant) {
    switch (variant) {
        case 9: return launch<16, 16>(a, b, c, m, stream);
        case 15: return launch_vector<1>(a, b, c, m, stream);
        case 16: return launch_vector<2>(a, b, c, m, stream);
        case 25: return launch<8, 16, 8, true>(a, b, c, m, stream);
        case 27: return launch_wide<16, 16, 16>(a, b, c, m, stream);
        case 28: return launch_wide<16, 32, 8>(a, b, c, m, stream);
    }
    return cudaErrorInvalidValue;
}
"""


@lru_cache(maxsize=None)
def _compile(capability):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError("nvcc is required to build the small-M GEMM")
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc")
    major, minor = capability
    with tempfile.TemporaryDirectory(prefix="gemm_n128_") as work:
        source = Path(work) / "gemm.cu"
        binary = Path(work) / "gemm.so"
        source.write_text(_CUDA_SOURCE)
        result = subprocess.run(
            [nvcc, f"-arch=sm_{major}{minor}", "-O3", "-std=c++17", "--shared",
             "-Xcompiler=-fPIC", str(source), "-o", str(binary)],
            capture_output=True, text=True, timeout=90,
        )
        if result.returncode:
            raise RuntimeError(f"Native compilation failed: {result.stderr}")
        library = ctypes.CDLL(str(binary))
    dispatch = library.dispatch
    dispatch.argtypes = [ctypes.c_void_p] * 3 + [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ]
    dispatch.restype = ctypes.c_int
    return dispatch


def _native(A, B, C, variant):
    m = A.shape[0]
    with torch.cuda.device(A.device):
        dispatch = _compile(torch.cuda.get_device_capability(A.device))
        stream = torch.cuda.current_stream(A.device).cuda_stream
        status = dispatch(A.data_ptr(), B.data_ptr(), C.data_ptr(), m, stream, variant)
    if status:
        raise RuntimeError(f"Small-M GEMM launch failed with CUDA error {status}")


_LIST_WS = gl.constexpr("functions_and_args" in inspect.signature(gl.warp_specialize).parameters)
_STRIDED_TMEM = gl.constexpr("col_stride" in inspect.signature(bw.TensorMemoryLayout).parameters)


@gluon.jit
def _load(state):
    a, b, c, sa, sb, filled, freed, acc, done, released, M = state
    BM: gl.constexpr = a.block_type.shape[0]
    BK: gl.constexpr = a.block_type.shape[1]
    STAGES: gl.constexpr = sa.shape[0]
    sequence = 0
    for tile in range(gl.program_id(0), gl.cdiv(M, BM), gl.num_programs(0)):
        for step in range(2048 // BK):
            slot = sequence % STAGES
            phase = (sequence // STAGES) & 1
            mbarrier.wait(freed.index(slot), phase ^ 1)
            bar = filled.index(slot)
            mbarrier.expect(bar, (BM + 128) * BK * 2)
            # Spread simultaneous TMA requests across K sectors.
            k_step = (step + tile * 5) % (2048 // BK)
            tma.async_copy_global_to_shared(a, [tile * BM, k_step * BK], bar, sa.index(slot))
            tma.async_copy_global_to_shared(b, [0, k_step * BK], bar, sb.index(slot))
            sequence += 1


@gluon.jit
def _multiply(state):
    a, b, c, sa, sb, filled, freed, acc, done, released, M = state
    BM: gl.constexpr = a.block_type.shape[0]
    BK: gl.constexpr = a.block_type.shape[1]
    STAGES: gl.constexpr = sa.shape[0]
    sequence = 0
    ordinal = 0
    for tile in range(gl.program_id(0), gl.cdiv(M, BM), gl.num_programs(0)):
        output_slot = ordinal % 2
        output_phase = (ordinal // 2) & 1
        mbarrier.wait(released.index(output_slot), output_phase ^ 1)
        for step in range(2048 // BK):
            slot = sequence % STAGES
            phase = (sequence // STAGES) & 1
            mbarrier.wait(filled.index(slot), phase)
            bw.tcgen05_mma(sb.index(slot), sa.index(slot).permute((1, 0)),
                           acc.index(output_slot), use_acc=step != 0)
            bw.tcgen05_commit(freed.index(slot))
            sequence += 1
        bw.tcgen05_commit(done.index(output_slot))
        ordinal += 1


@gluon.jit
def _store(state):
    a, b, c, sa, sb, filled, freed, acc, done, released, M = state
    BM: gl.constexpr = a.block_type.shape[0]
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [4, 1], [0, 1])
    n = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    m = gl.arange(0, BM, layout=gl.SliceLayout(0, layout))
    ordinal = 0
    for tile in range(gl.program_id(0), gl.cdiv(M, BM), gl.num_programs(0)):
        slot = ordinal % 2
        phase = (ordinal // 2) & 1
        mbarrier.wait(done.index(slot), phase)
        result = acc.index(slot).load(layout).to(gl.float16)
        mbarrier.arrive(released.index(slot), count=1)
        rows = tile * BM + m
        gl.store(c + rows[None, :] * 128 + n[:, None], result, rows[None, :] < M)
        ordinal += 1


@gluon.jit
def _barriers(COUNT: gl.constexpr):
    bars = gl.allocate_shared_memory(gl.int64, [COUNT, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(COUNT):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def _staged(a, b, c, M, STAGES: gl.constexpr):
    BM: gl.constexpr = a.block_type.shape[0]
    BK: gl.constexpr = a.block_type.shape[1]
    sa = gl.allocate_shared_memory(gl.float16, [STAGES, BM, BK], a.layout)
    sb = gl.allocate_shared_memory(gl.float16, [STAGES, 128, BK], b.layout)
    if _STRIDED_TMEM:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BM], col_stride=1)
    else:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BM], unpacked=True)
    acc = bw.allocate_tensor_memory(gl.float32, [2, 128, BM], layout)
    args = ((a, b, c, sa, sb, _barriers(STAGES), _barriers(STAGES),
             acc, _barriers(2), _barriers(2), M),)
    if _LIST_WS:
        gl.warp_specialize([(_store, args), (_load, args), (_multiply, args)], [1, 1], [24, 24])
    else:
        gl.warp_specialize(default_args=args, default_partition=_store, worker_args=args,
                           worker_partitions=[_load, _multiply], worker_num_warps=[1, 1],
                           worker_num_regs=[24, 24])


def _staged_launch(A, B, C):
    def descriptor(tensor, shape):
        layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float16)
        return GluonDescriptor.from_tensor(tensor, shape, layout)

    m = A.shape[0]
    # Where 64-row blocks fit in one wave, use the smaller M tile for occupancy.
    bm, bk, stages = (64, 128, 4) if m <= 9216 else (128, 64, 6)
    tiles = triton.cdiv(m, bm)
    _staged[(min(148, tiles),)](
        descriptor(A, [bm, bk]), descriptor(B, [128, bk]), C, m, stages, num_warps=4)


def run(A, B, C):
    """Write the FP32-accumulated GEMM to the caller's FP16 destination."""
    m = A.shape[0]
    if m == 0:
        return
    if m <= 8:
        # Each block reuses one B column for one or two activation rows.
        _native(A, B, C, 15 if m == 2 else 16)
    elif m <= 64:
        _native(A, B, C, 25)
    elif m <= 128:
        _native(A, B, C, 9)
    elif m <= 512:
        _native(A, B, C, 27)
    elif m < 2048:
        _native(A, B, C, 28)
    else:
        _staged_launch(A, B, C)
