# SPDX-License-Identifier: Apache-2.0
"""FP16 destination-passing GEMM for FlashInfer-Bench operator 005.

Small-shape distributed FP32 reduction and swapped-operand MMA were inspired by
QianQu's CompoundingKernel Explorer, FlashInfer/005_gemm_n256_k7168.
https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/005_gemm_n256_k7168

Persistent scheduling and staged TMA draw on the repository H100 GEMM.
https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py
"""
from ctypes import c_int, c_void_p
from functools import lru_cache

import torch
import triton
import triton.language as tl
from cuda.bindings import driver, nvrtc
from triton.tools.tensor_descriptor import TensorDescriptor as TmaDescriptor


_CUDA = r"""
using U32 = unsigned int;
using U64 = unsigned long long;
using Half = unsigned short;
constexpr int K = 7168;
constexpr int N = 256;
constexpr U32 WARP = 0xffffffffu;

struct Vec {
    U32 x[8];
};

template <bool Weights>
__device__ __forceinline__ Vec load32(const Half* p) {
    Vec v;
    if constexpr (Weights) {
        asm("ld.global.nc.L1::no_allocate.L2::128B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]),
              "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7]) : "l"(p));
    } else {
        asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(v.x[0]), "=r"(v.x[1]), "=r"(v.x[2]), "=r"(v.x[3]),
              "=r"(v.x[4]), "=r"(v.x[5]), "=r"(v.x[6]), "=r"(v.x[7]) : "l"(p));
    }
    return v;
}

__device__ __forceinline__ float dot2(U32 a, U32 b) {
    float v;
    asm("{ .reg .b16 a0,a1,b0,b1; .reg .f32 p; "
        "mov.b32 {a0,a1}, %1; mov.b32 {b0,b1}, %2; "
        "fma.rn.f32.f16 p, a0, b0, 0f00000000; fma.rn.f32.f16 %0, a1, b1, p; }"
        : "=f"(v) : "r"(a), "r"(b));
    return v;
}

__device__ __forceinline__ float warp_dot(const Vec& a, const Vec& b) {
    float v = ((dot2(a.x[0], b.x[0]) + dot2(a.x[1], b.x[1])) +
              (dot2(a.x[2], b.x[2]) + dot2(a.x[3], b.x[3]))) +
              ((dot2(a.x[4], b.x[4]) + dot2(a.x[5], b.x[5])) +
              (dot2(a.x[6], b.x[6]) + dot2(a.x[7], b.x[7])));
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2)
        v += __shfl_down_sync(WARP, v, offset);
    return v;
}

__device__ __forceinline__ Half fp16(float v) {
    Half h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

__device__ __forceinline__ U32 shared_addr(const void* p) {
    return static_cast<U32>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ U32 remote_addr(const void* p, U32 rank) {
    U32 remote;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(shared_addr(p)), "r"(rank));
    return remote;
}

__device__ __forceinline__ void arm(U64* barrier, U32 bytes) {
    const U32 addr = shared_addr(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1; "
                 "fence.mbarrier_init.release.cluster; "
                 "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void receive(U64* barrier) {
    asm volatile("{ .reg .pred ready; AGAIN: "
                 "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 ready, [%0], 0; "
                 "@!ready bra AGAIN; }" :: "r"(shared_addr(barrier)) : "memory");
}

template <int NR>
__global__ __launch_bounds__(448, 2)
void gemv_direct(const Half* A, const Half* B, Half* C, int /* M */) {
    __shared__ float partial[NR][14];
    const int row = blockIdx.y;
    const int weight = blockIdx.x * NR;
    const int k = 16 * threadIdx.x;
    const Vec a = load32<false>(A + row * K + k);
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
        const Vec b = load32<true>(B + (weight + j) * K + k);
        const float sum = warp_dot(a, b);
        if (threadIdx.x % 32 == 0)
            partial[j][threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < 14; ++w)
            sum += partial[threadIdx.x][w];
        C[row * N + weight + threadIdx.x] = fp16(sum);
    }
}

__device__ __forceinline__ void mma(float (&c)[4], U32 a0, U32 a1, U32 a2, U32 a3, U32 b0, U32 b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct alignas(16) Float4 {
    float x, y, z, w;

    __device__ __forceinline__ void add(const Float4& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        w += rhs.w;
    }
};

__device__ __forceinline__ void send4(Float4* dst, const Float4& v, U64* barrier, U32 owner) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                 :: "r"(remote_addr(dst, owner)), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w),
                    "r"(remote_addr(barrier, owner)) : "memory");
}

template <int TM, int TN, int Cluster, int Warps, bool Swap = false, bool PeerReduce = false>
__global__ __cluster_dims__(Cluster, 1, 1) __launch_bounds__(32 * Warps, Warps > 8 ? 1 : 2)
void gemm_cluster(const Half* A, const Half* B, Half* C, int m) {
    // Pad each shared row to distribute the warp fragment stores across banks.
    constexpr int padded_n = TN + 8;
    constexpr int groups = TM * TN / 4;
    constexpr int per_owner = (groups + Cluster - 1) / Cluster;
    constexpr int k_per_rank = K / Cluster;
    constexpr int k_per_warp = k_per_rank / Warps;
    static_assert(K == k_per_warp * Warps * Cluster && k_per_warp % 64 == 0);
    constexpr int mma_m = Swap ? TN / 16 : TM / 16;
    constexpr int mma_n = Swap ? TM / 8 : TN / 8;

    const int group = blockIdx.x / Cluster;
    const int rank = blockIdx.x % Cluster;
    const int m0 = blockIdx.y * TM;
    const int n0 = group * TN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int g = tid % 32 / 4;
    const int t = tid % 4;
    const int k = ((rank + group) % Cluster) * k_per_rank + warp * k_per_warp + t * 16;
    const int owner_start = rank * per_owner;
    const int valid_groups = PeerReduce ? min(m - m0, TM) * (TN / 4) : groups;
    const int owner_end = min(owner_start + per_owner, valid_groups);
    const int owner_rotation = owner_start < valid_groups ? owner_start : 0;

    __shared__ U64 barrier;
    extern __shared__ __align__(16) unsigned char shared[];
    auto red = reinterpret_cast<float (*)[TM][padded_n]>(shared);
    __shared__ Float4 partial[Cluster][per_owner];
    float acc[mma_m][mma_n][4] = {};

    #pragma unroll
    for (int step = 0; step < k_per_warp; step += 64) {
        Vec a[TM / 8], b[TN / 8];
        #pragma unroll
        for (int j = 0; j < TN / 8; ++j)
            b[j] = load32<true>(B + (n0 + j * 8 + g) * K + k + step);
        #pragma unroll
        for (int i = 0; i < TM / 8; ++i) {
            const int row = m0 + i * 8 + g;
            a[i] = row < m ? load32<false>(A + row * K + k + step) : Vec{};
        }
        if (step == 0) {
            if (tid == 0 && owner_start < owner_end)
                arm(&barrier, (owner_end - owner_start) * (Cluster - int(PeerReduce)) * sizeof(Float4));
            cluster_arrive();
        }
        #pragma unroll
        for (int i = 0; i < mma_m; ++i) {
            #pragma unroll
            for (int j = 0; j < mma_n; ++j) {
                const Vec& lhs0 = Swap ? b[2 * i] : a[2 * i];
                const Vec& lhs1 = Swap ? b[2 * i + 1] : a[2 * i + 1];
                const Vec& rhs = Swap ? a[j] : b[j];
                #pragma unroll
                for (int word = 0; word < 8; word += 2)
                    mma(acc[i][j], lhs0.x[word], lhs1.x[word], lhs0.x[word + 1], lhs1.x[word + 1],
                        rhs.x[word], rhs.x[word + 1]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < mma_m; ++i) {
        #pragma unroll
        for (int j = 0; j < mma_n; ++j) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = i * 16 + g + 8 * (l / 2);
                const int col = j * 8 + 2 * t + l % 2;
                if constexpr (Swap)
                    red[warp][col][row] = acc[i][j][l];
                else
                    red[warp][row][col] = acc[i][j][l];
            }
        }
    }
    __syncthreads();
    cluster_wait();
    if constexpr (!PeerReduce) {
        // Each rank owns a disjoint output range and receives FP32 partials from all ranks.
        for (int i = tid; i < groups; i += 32 * Warps) {
            const int row = i / (TN / 4);
            const int col = i % (TN / 4) * 4;
            Float4 value{};
            #pragma unroll
            for (int w = 0; w < Warps; ++w)
                value.add(*reinterpret_cast<const Float4*>(&red[w][row][col]));
            const int owner = i / per_owner;
            send4(&partial[rank][i % per_owner], value, &barrier, owner);
        }
        for (int i = owner_start + tid; i < owner_end; i += 32 * Warps) {
            receive(&barrier);
            Float4 value{};
            #pragma unroll
            for (int peer = 0; peer < Cluster; ++peer)
                value.add(partial[peer][i - owner_start]);
            const int row = m0 + i / (TN / 4);
            const int col = n0 + i % (TN / 4) * 4;
            if (row < m) {
                const Half output[4] = {fp16(value.x), fp16(value.y), fp16(value.z), fp16(value.w)};
                asm volatile("st.global.v4.b16 [%0], {%1,%2,%3,%4};"
                             :: "l"(C + row * N + col), "h"(output[0]), "h"(output[1]),
                                "h"(output[2]), "h"(output[3]) : "memory");
            }
        }
    } else {
        // Rotate the FP32 output groups so the first lanes retain this
        // rank's own contribution. Only peers transmit to the owner.
        // Groups beyond the M tail neither send partials nor arm a receive.
        static_assert(groups <= 32 * Warps);
        Float4 own{};
        if (tid < valid_groups) {
            int i = tid + owner_rotation;
            if (i >= valid_groups)
                i -= valid_groups;
            const int row = i / (TN / 4);
            const int col = i % (TN / 4) * 4;
            Float4 sum{};
            #pragma unroll
            for (int w = 0; w < Warps; ++w)
                sum.add(*reinterpret_cast<const Float4*>(&red[w][row][col]));
            const int owner = i / per_owner;
            if (owner == rank)
                own = sum;
            else
                send4(&partial[rank][i % per_owner], sum, &barrier, owner);
        }
        if (tid < owner_end - owner_start) {
            receive(&barrier);
            #pragma unroll
            for (int peer = 0; peer < Cluster; ++peer) {
                if (peer != rank)
                    own.add(partial[peer][tid]);
            }
            const int i = owner_start + tid;
            const int row = m0 + i / (TN / 4);
            const int col = n0 + i % (TN / 4) * 4;
            U32 lo, hi;
            asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(lo) : "f"(own.x), "f"(own.y));
            asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(hi) : "f"(own.z), "f"(own.w));
            asm volatile("st.global.v2.b32 [%0], {%1,%2};"
                         :: "l"(C + row * N + col), "r"(lo), "r"(hi) : "memory");
        }
    }
}
"""


def _checked(result):
    code, *values = result
    if int(code):
        raise RuntimeError(f"CUDA operation failed with {code}")
    return values[0] if len(values) == 1 else values


_VARIANTS = {
    "gemv_direct_2": "&gemv_direct<2>",
    "mma_8_16_7_8_swap": "&gemm_cluster<8, 16, 7, 8, true>",
    "mma_16_8_7_8_normal": "&gemm_cluster<16, 8, 7, 8, false>",
    "mma_16_16_7_8_normal": "&gemm_cluster<16, 16, 7, 8, false>",
    "mma_32_16_7_8_normal": "&gemm_cluster<32, 16, 7, 8, false, true>",
    "mma_32_32_8_7_normal": "&gemm_cluster<32, 32, 8, 7, false>",
}

@lru_cache(maxsize=None)
def _module(major, minor, context):
    program = _checked(nvrtc.nvrtcCreateProgram(_CUDA.encode(), b"gemm_small.cu", 0, [], []))
    try:
        for expr in _VARIANTS.values():
            _checked(nvrtc.nvrtcAddNameExpression(program, expr.encode()))
        flags = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"--std=c++17"]
        result = nvrtc.nvrtcCompileProgram(program, len(flags), flags)[0]
        if int(result):
            log = bytearray(_checked(nvrtc.nvrtcGetProgramLogSize(program)))
            _checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError("NVRTC build failure: " + log.decode(errors="replace").strip("\x00"))
        lowered = {name: bytes(_checked(nvrtc.nvrtcGetLoweredName(program, expr.encode())))
                   for name, expr in _VARIANTS.items()}
        cubin = bytearray(_checked(nvrtc.nvrtcGetCUBINSize(program)))
        _checked(nvrtc.nvrtcGetCUBIN(program, cubin))
    finally:
        _checked(nvrtc.nvrtcDestroyProgram(program))
    module = _checked(driver.cuModuleLoadData(cubin))
    functions = {}
    for name, symbol in lowered.items():
        fn = _checked(driver.cuModuleGetFunction(module, symbol))
        smem = 0
        if name.startswith("mma_"):
            tm, tn, _, warps = map(int, name.split("_")[1:5])
            smem = warps * tm * (tn + 8) * 4
            _checked(driver.cuFuncSetAttribute(
                fn, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            ))
        functions[name] = (fn, smem)
    return functions


@lru_cache(maxsize=None)
def _balance():
    attr = driver.CUlaunchAttribute()
    attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
    attr.value.clusterSchedulingPolicyPreference = (
        driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING
    )
    return [attr]


def _native(A, B, C, name, x, y, threads, clustered):
    m = A.shape[0]
    device = A.device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        context = _checked(driver.cuCtxGetCurrent())
        module = _module(*torch.cuda.get_device_capability(device), int(context))
        fn, smem = module[name]
        params = ((A.data_ptr(), B.data_ptr(), C.data_ptr(), m),
                  (c_void_p, c_void_p, c_void_p, c_int))
        if clustered:
            attrs = _balance()
            config = driver.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = x, y, 1
            config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
            config.hStream = stream
            config.sharedMemBytes = smem
            config.attrs, config.numAttrs = attrs, len(attrs)
            _checked(driver.cuLaunchKernelEx(config, fn, params, 0))
        else:
            _checked(driver.cuLaunchKernel(fn, x, y, 1, threads, 1, 1, smem, stream, params, 0))


@triton.jit
def _strided_gemm(A, B, C, M, AM, AK, BM, BK, CM, CN):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 64)
    a = A + rows[:, None] * AM + k[None, :] * AK
    b = B + cols[:, None] * BM + k[None, :] * BK
    acc = tl.full((32, 32), 0, tl.float32)
    for ki in range(112):
        av = tl.load(a + ki * 64 * AK, rows[:, None] < M, 0)
        bv = tl.load(b + ki * 64 * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + rows[:, None] * CM + cols[None, :] * CN, acc, rows[:, None] < M)



@triton.jit
def _tma_gemm(a, b, c, C, M, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
              WORKERS: tl.constexpr, SWAP: tl.constexpr, TMA_STORE: tl.constexpr):
    tiles_n: tl.constexpr = 256 // TN
    tiles_m = tl.cdiv(M, TM)
    for tile in range(tl.program_id(0), tiles_m * tiles_n, WORKERS):
        row = tile // tiles_n * TM
        col = tile % tiles_n * TN
        if SWAP:
            acc = tl.full((TN, TM), 0, tl.float32)
        else:
            acc = tl.full((TM, TN), 0, tl.float32)
        for ki in range(7168 // TK):
            k = ki * TK
            av = a.load([row, k])
            bv = b.load([col, k])
            if SWAP:
                acc = tl.dot(bv, tl.trans(av), acc)
            else:
                acc = tl.dot(av, tl.trans(bv), acc)
        if SWAP:
            values = tl.trans(acc)
        else:
            values = acc
        if TMA_STORE:
            c.store([row, col], values.to(tl.float16))
        else:
            rows = row + tl.arange(0, TM)
            cols = col + tl.arange(0, TN)
            tl.store(C + rows[:, None] * 256 + cols[None, :], values, rows[:, None] < M)


def _tma(A, B, C, tm, tn, tk, stages, persistent=False, swap=False, tma_store=True):
    m = A.shape[0]
    tiles = triton.cdiv(m, tm) * (256 // tn)
    if persistent:
        sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
        workers = min(tiles, sm_count)
    else:
        workers = tiles
    _tma_gemm[(workers,)](
        TmaDescriptor.from_tensor(A, [tm, tk]), TmaDescriptor.from_tensor(B, [tn, tk]),
        TmaDescriptor.from_tensor(C, [tm, tn]) if tma_store else None,
        C, m, tm, tn, tk, workers, swap, tma_store, num_warps=4, num_stages=stages,
    )


def _compact(A, B, C, tm, tn, cluster, warps, swap=False):
    name = f'mma_{tm}_{tn}_{cluster}_{warps}_{"swap" if swap else "normal"}'
    _native(A, B, C, name, 256 // tn * cluster, triton.cdiv(A.shape[0], tm), 32 * warps, True)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 32 for tensor in (A, B, C)):
        _strided_gemm[(triton.cdiv(m, 32), 8)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(), num_warps=4, num_stages=3
        )
    elif m == 1:
        _native(A, B, C, "gemv_direct_2", 128, 1, 448, False)
    elif m <= 8:
        _compact(A, B, C, 8, 16, 7, 8, swap=True)
    elif m <= 16:
        _compact(A, B, C, 16, 8, 7, 8)
    elif m <= 32:
        _compact(A, B, C, 16, 16, 7, 8)
    elif m <= 64:
        # Two 32-row tiles give the dense 53–63 range more resident work.
        _compact(A, B, C, 32, 16, 7, 8)
    elif m <= 128:
        _compact(A, B, C, 32, 32, 8, 7)
    elif m <= 1024:
        _tma(A, B, C, 32, 64, 128, 4, swap=True)
    else:
        _tma(A, B, C, 128, 256, 64, 4, persistent=True, tma_store=False)
