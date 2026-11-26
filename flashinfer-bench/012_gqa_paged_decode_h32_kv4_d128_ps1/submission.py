"""Paged grouped-query decode for BF16 caches and FP32 base-2 LSE.

Tile reuse follows RSI-Kernel-Tune's H100 decode seed. The transposed
K @ Q.T and V.T @ P layout and split-KV scheduling were inspired by
QianQu/CompoundingKernel-Explorer, FlashInfer 012 (Claude via Amazon Bedrock).
"""

import ctypes
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language._core import builtin


@builtin
def _mma_v2(a, b, acc, _semantic=None):
    # This uses the same Triton dot lowering on versions before mma_v2
    # became a public Gluon function.
    dot = _semantic.dot(a, b, acc, input_precision=None,
                        max_num_imprecise_acc=None, out_dtype=acc.dtype)
    return gl.tensor(dot.handle, acc.type)


@triton.jit
def _short(Q, K, V, Indptr, Indices, O, L, scale,
           T, N: tl.constexpr):
    head = tl.program_id(0)
    begin = tl.load(Indptr)
    end = tl.load(Indptr + 1)
    d = tl.arange(0, 128)
    tokens = tl.arange(0, N)
    indices = tl.load(Indices + tokens, tokens < T, 0)
    q = tl.load(Q + head * 128 + d).to(tl.float32)
    page = indices
    live = tokens < T
    valid = (tokens >= begin) & (tokens < end)
    offset = page[:, None] * 512 + (head // 8) * 128 + d[None, :]
    k = tl.load(K + offset, live[:, None], 0).to(tl.float32)
    v = tl.load(V + offset, live[:, None], 0).to(tl.float32)
    k = tl.where(valid[:, None], k, 0)
    v = tl.where(valid[:, None], v, 0)
    score = tl.where(valid, tl.sum(k * q[None, :], 1) * scale, float("-inf"))
    peak = tl.max(score, 0)
    p = tl.exp2(score - tl.where(begin < end, peak, 0))
    den = tl.sum(p, 0)
    divisor = tl.where(den > 0, den, 1)
    tl.store(O + head * 128 + d, tl.sum(p[:, None] * v, 0) / divisor)
    tl.store(L + head, peak + tl.log2(divisor))


@gluon.jit
def _single_row(Q, K, V, Indptr, Indices, O, L, scale, T,
                N: gl.constexpr, D: gl.constexpr, NW: gl.constexpr):
    group = gl.program_id(1)
    dv_part = gl.program_id(2)
    start = gl.load(Indptr)
    end = gl.load(Indptr + 1)
    GL: gl.constexpr = BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    MM: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[NW, 1], instr_shape=[16, 8])
    A: gl.constexpr = DotOperandLayout(0, MM, 2)
    BB: gl.constexpr = DotOperandLayout(1, MM, 2)
    h = gl.arange(0, 8, layout=gl.constexpr(SliceLayout(1, GL)))
    d = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, GL)))
    dv = dv_part * D + gl.arange(0, D, layout=gl.constexpr(SliceLayout(0, GL)))
    token = gl.arange(0, N, layout=gl.constexpr(SliceLayout(1, GL)))
    head = group * 8 + h
    q = gl.load(Q + head[:, None] * 128 + d[None, :])
    page = gl.load(Indices + token, token < T, 0)
    live = (token >= start) & (token < end)
    kv = page[:, None] * 512 + group * 128
    k = gl.load(K + kv + d[None, :], (token < T)[:, None], 0)
    v = gl.load(V + kv + dv[None, :], (token < T)[:, None], 0)
    k = gl.where(live[:, None], k, 0)
    v = gl.where(live[:, None], v, 0)
    logit = _mma_v2(gl.convert_layout(k, A), gl.convert_layout(gl.permute(q, (1, 0)), BB),
                   gl.full((N, 8), 0, gl.float32, layout=MM)) * scale
    logit = gl.where(gl.convert_layout(live, gl.constexpr(SliceLayout(1, MM)))[:, None],
                     logit, float("-inf"))
    peak = gl.max(logit, 0)
    prob = gl.exp2(logit - gl.where(start < end, peak, 0)[None, :])
    den = gl.sum(prob, 0)
    acc = _mma_v2(gl.convert_layout(gl.permute(v, (1, 0)), A),
                 gl.convert_layout(prob.to(gl.bfloat16), BB),
                 gl.full((D, 8), 0, gl.float32, layout=MM))
    acc = gl.convert_layout(gl.permute(acc, (1, 0)), GL)
    peak = gl.convert_layout(peak, gl.constexpr(SliceLayout(1, GL)))
    den = gl.convert_layout(den, gl.constexpr(SliceLayout(1, GL)))
    safe = gl.where(den > 0, den, 1)
    gl.store(O + head[:, None] * 128 + dv[None, :], acc / safe[:, None])
    if dv_part == 0:
        gl.store(L + head, peak + gl.log2(safe))


@triton.jit
def _split(Q, K, V, Indptr, Indices, Scratch, scale,
           B: tl.constexpr, SEARCH: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
           VEC_SEARCH: tl.constexpr):
    group = tl.program_id(1)
    slot = tl.program_id(0)
    # A row owns slots floor(indptr[row] / N) + row onward.
    # At most one slot per row is padding, including for empty rows.
    if VEC_SEARCH:
        rows = tl.arange(0, triton.next_power_of_2(B))
        starts = tl.load(Indptr + rows, rows < B, 0)
        eligible = (rows < B) & (starts // N + rows <= slot)
        row = tl.max(tl.where(eligible, rows, 0), 0)
        first = tl.sum(tl.where(rows == row, starts, 0), 0)
    else:
        lo = 0
        hi = B
        for _ in tl.static_range(0, SEARCH):
            mid = (lo + hi) // 2
            base = tl.load(Indptr + mid) // N + mid
            lo, hi = tl.where(slot >= base, mid, lo), tl.where(slot >= base, hi, mid)
        row = lo
        first = tl.load(Indptr + row)
    start = first + (slot - first // N - row) * N
    end = tl.load(Indptr + row + 1)
    if start < end:
        h = tl.arange(0, 16)
        d = tl.arange(0, 128)
        t = start + tl.arange(0, N)
        head = row * 32 + group * 8 + h
        q = tl.load(Q + head[:, None] * 128 + d[None, :], h[:, None] < 8, 0)
        page = tl.load(Indices + t, t < end, 0)
        kv = page[:, None] * 512 + group * 128 + d[None, :]
        k = tl.load(K + kv, t[:, None] < end, 0)
        v = tl.load(V + kv, t[:, None] < end, 0)
        score = tl.dot(q, tl.trans(k)) * scale
        score = tl.where(t[None, :] < end, score, float("-inf"))
        peak = tl.max(score, 1)
        p = tl.exp2(score - peak[:, None])
        den = tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), v)
        record = (slot * 32 + group * 8 + h) * S
        tl.store(Scratch + record[:, None] + d[None, :], acc, h[:, None] < 8)
        tl.store(Scratch + record + 128, peak, h < 8)
        tl.store(Scratch + record + 129, den, h < 8)


@triton.jit
def _merge(Indptr, Scratch, O, L, N: tl.constexpr, M: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    first_slot = start // N + row
    count = tl.cdiv(end - start, N)
    i = tl.arange(0, M)
    d = tl.arange(0, 128)
    maximum = float("-inf")
    total = 0.0
    acc = tl.full((128,), 0, tl.float32)
    for begin in range(0, count, M):
        chunk = begin + i
        base = ((first_slot + chunk) * 32 + head) * S
        peak = tl.load(Scratch + base + 128, chunk < count, float("-inf"))
        den = tl.load(Scratch + base + 129, chunk < count, 0)
        next_maximum = tl.maximum(maximum, tl.max(peak, 0))
        weights = tl.exp2(peak - next_maximum)
        decay = tl.exp2(maximum - next_maximum)
        partial = tl.load(Scratch + base[:, None] + d[None, :], chunk[:, None] < count, 0)
        acc = acc * decay + tl.sum(partial * weights[:, None], 0)
        total = total * decay + tl.sum(den * weights, 0)
        maximum = next_maximum
    safe = tl.where(total > 0, total, 1)
    tl.store(O + (row * 32 + head) * 128 + d, acc / safe)
    tl.store(L + row * 32 + head, maximum + tl.log2(safe))



def _launch_split(q, k, v, indptr, indices, sm_scale, output, lse, *, tile,
                  warps, merge_tile=32, vector_search=True, stride=136):
    batch, tokens = q.shape[0], indices.numel()
    slots = tokens // tile + batch
    scratch = torch.empty((slots, 32, stride), dtype=torch.float32, device=q.device)
    _split[(slots, 4)](q, k, v, indptr, indices, scratch, sm_scale * 1.4426950408889634,
                       batch, (batch - 1).bit_length(), tile, stride, vector_search, num_warps=warps)
    merge_tile = min(merge_tile, triton.next_power_of_2(max(1, triton.cdiv(tokens, tile))))
    _merge[(batch, 32)](indptr, scratch, output, lse, tile, merge_tile, stride, num_warps=4)


def _run_triton(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch, tokens = q.shape[0], kv_indices.numel()
    if batch == 0:
        return
    scale = sm_scale * 1.4426950408889634
    if batch == 1 and tokens <= 4:
        _short[(32,)](q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
                      scale, tokens, triton.next_power_of_2(max(1, tokens)), num_warps=1)
    elif batch == 1 and tokens <= 256:
        _single_row[(1, 4, 2)](q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
                               scale, tokens, max(16, triton.next_power_of_2(tokens)), 64, 4, num_warps=4)
    else:
        if batch == 1:
            tile, warps = 32, 4
        elif batch <= 16 and tokens < 256 * batch:
            tile, warps = 32, 2
        elif batch <= 16:
            tile, warps = 128, 8
        else:
            tile, warps = 128, 4
        _launch_split(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
                      tile=tile, warps=warps)


_CUDA_SOURCE = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

struct Params {
    const bf16 *q, *k, *v;
    const int *indptr, *indices;
    bf16 *output;
    float *lse;
    float scale;
};

__device__ __forceinline__ unsigned smem_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void copy16(uint4* dst, const bf16* src, bool live) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_address(dst)), "l"(src), "r"(live ? 16 : 0));
}

template<bool Transpose>
__device__ __forceinline__ void load4(unsigned (&x)[4], uint4* ptr) {
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                     : "r"(smem_address(ptr)));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                     : "r"(smem_address(ptr)));
    }
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack_transpose(float a, float b) {
    unsigned in, out;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(in) : "f"(b), "f"(a));
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(out) : "r"(in));
    return out;
}

template<int WIDTH>
__device__ __forceinline__ void stage(uint4* buf, const Params& p, int group, int dim,
                                     int start, int end, int lane) {
    constexpr int KCOL = 16;
    constexpr int VCOL = WIDTH / 8;
    const int row_pair = lane / KCOL;
    const int col = lane % KCOL;
#pragma unroll
    for (int r = 0; r < 16; r += 2) {
        const int row = r + row_pair;
        const bool live = start + row < end;
        const int page = live ? p.indices[start + row] : 0;
        const int offset = page * 512 + group * 128;
        copy16(buf + row * KCOL + (col ^ (row % 8)), p.k + offset + col * 8, live);
        if (col < VCOL) {
            copy16(buf + 16 * KCOL + row * VCOL + (col ^ (row % VCOL)),
                   p.v + offset + dim + col * 8, live);
        }
    }
    asm volatile("cp.async.commit_group;");
}

template<int WIDTH, int WARPS, int CLUSTER, bool PIPE>
__global__ void decode(const Params p) {
    constexpr int VCOL = WIDTH / 8;
    constexpr int PAGE_BUF = 16 * (16 + VCOL);
    constexpr int STAGES = PIPE ? 2 : 1;
    constexpr int STAT = WIDTH * 8;
    constexpr int RECORD = STAT + 16;
    extern __shared__ __align__(16) unsigned char shared[];
    uint4* buf = reinterpret_cast<uint4*>(shared);
    int rank = blockIdx.x % CLUSTER;
    int group = blockIdx.x / CLUSTER % 4;
    int dim = blockIdx.x / (CLUSTER * 4) * WIDTH;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    int g = lane / 4;
    int t = lane % 4;
    int start = p.indptr[b];
    int end = p.indptr[b + 1];
    int tiles = (end - start + 15) / 16;
    int chunk = (tiles + CLUSTER - 1) / CLUSTER;
    int tile = rank * chunk + warp;
    int tile_end = min(tiles, (rank + 1) * chunk);
    uint4* warp_buf = buf + warp * PAGE_BUF * STAGES;

    if (tile < tile_end) stage<WIDTH>(warp_buf, p, group, dim, start + tile * 16, end, lane);
    unsigned q0[8], q1[8];
    const unsigned* q = reinterpret_cast<const unsigned*>(p.q + (b * 32 + group * 8 + g) * 128);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q0[i] = q[i * 8 + t];
        q1[i] = q[i * 8 + t + 4];
    }

    float peak[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float den[2] = {};
    float out[WIDTH / 16][4] = {};
    int stage_index = 0;
    for (; tile < tile_end; tile += WARPS) {
        uint4* current = warp_buf + stage_index * PAGE_BUF;
        if constexpr (PIPE) {
            stage_index ^= 1;
            if (tile + WARPS < tile_end) {
                stage<WIDTH>(warp_buf + stage_index * PAGE_BUF, p, group, dim,
                             start + (tile + WARPS) * 16, end, lane);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncwarp();
        float s[4] = {};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned k[4];
            int r = lane % 16;
            load4<false>(k, current + r * 16 + ((i * 2 + lane / 16) ^ (r % 8)));
            mma(s, k, q0[i], q1[i]);
        }
        int pos = start + tile * 16 + g;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float x = pos < end ? s[h] * p.scale : -CUDART_INF_F;
            float y = pos + 8 < end ? s[h + 2] * p.scale : -CUDART_INF_F;
            float m = fmaxf(x, y);
#pragma unroll
            for (int j = 4; j <= 16; j *= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, j));
            m = fmaxf(m, peak[h]);
            float alpha = exp2f(peak[h] - m);
            s[h] = exp2f(x - m);
            s[h + 2] = exp2f(y - m);
            float l = s[h] + s[h + 2];
#pragma unroll
            for (int j = 4; j <= 16; j *= 2) l += __shfl_xor_sync(0xffffffffu, l, j);
#pragma unroll
            for (int i = 0; i < WIDTH / 16; ++i) {
                out[i][h] *= alpha;
                out[i][h + 2] *= alpha;
            }
            den[h] = den[h] * alpha + l;
            peak[h] = m;
        }
        unsigned w0 = pack_transpose(s[0], s[1]);
        unsigned w1 = pack_transpose(s[2], s[3]);
#pragma unroll
        for (int i = 0; i < WIDTH / 16; ++i) {
            unsigned v[4];
            int row = lane % 8 + 8 * (lane / 16);
            int col = i * 2 + (lane / 8) % 2;
            load4<true>(v, current + 16 * 16 + row * VCOL + (col ^ (row % VCOL)));
            mma(out[i], v, w0, w1);
        }
        __syncwarp();
        if constexpr (!PIPE) {
            if (tile + WARPS < tile_end)
                stage<WIDTH>(warp_buf, p, group, dim, start + (tile + WARPS) * 16, end, lane);
        }
    }
    __syncthreads();

    float* partial = reinterpret_cast<float*>(shared);
    float* mine = partial + warp * RECORD;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int head = 2 * t + h;
        if (g == 0) {
            mine[STAT + head] = peak[h];
            mine[STAT + 8 + head] = den[h];
        }
#pragma unroll
        for (int i = 0; i < WIDTH / 16; ++i) {
            mine[head * WIDTH + i * 16 + g] = out[i][h];
            mine[head * WIDTH + i * 16 + g + 8] = out[i][h + 2];
        }
    }
    __syncthreads();

    float* local = partial + WARPS * RECORD;
    for (int i = tid; i < STAT; i += WARPS * 32) {
        int head = i / WIDTH;
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) m = fmaxf(m, partial[w * RECORD + STAT + head]);
        float safe_m = isfinite(m) ? m : 0.0f;
        float a = 0.0f, l = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* r = partial + w * RECORD;
            float weight = exp2f(r[STAT + head] - safe_m);
            a += weight * r[i];
            l += weight * r[STAT + 8 + head];
        }
        float normalized = l > 0 ? a / l : 0.0f;
        float lse = l > 0 ? m + log2f(l) : -CUDART_INF_F;
        if constexpr (CLUSTER == 1) {
            int h = b * 32 + group * 8 + head;
            p.output[h * 128 + dim + i % WIDTH] = __float2bfloat16(normalized);
            if (dim == 0 && i % WIDTH == 0) p.lse[h] = lse;
        } else {
            local[i] = normalized;
            if (i % WIDTH == 0) local[STAT + head] = lse;
        }
    }

    if constexpr (CLUSTER > 1) {
        auto cluster = cg::this_cluster();
        cluster.sync();
        const float* records[CLUSTER];
#pragma unroll
        for (int j = 0; j < CLUSTER; ++j) records[j] = cluster.map_shared_rank(local, j);
        for (int i = rank * (STAT / CLUSTER) + tid; i < (rank + 1) * (STAT / CLUSTER);
             i += WARPS * 32) {
            int head = i / WIDTH;
            float m = -CUDART_INF_F;
            float lse[CLUSTER];
#pragma unroll
            for (int j = 0; j < CLUSTER; ++j) {
                lse[j] = records[j][STAT + head];
                m = fmaxf(m, lse[j]);
            }
            float safe_m = isfinite(m) ? m : 0.0f;
            float a = 0.0f, l = 0.0f;
#pragma unroll
            for (int j = 0; j < CLUSTER; ++j) {
                float w = exp2f(lse[j] - safe_m);
                a += w * records[j][i];
                l += w;
            }
            int h = b * 32 + group * 8 + head;
            p.output[h * 128 + dim + i % WIDTH] = __float2bfloat16(l > 0 ? a / l : 0.0f);
            if (dim == 0 && i % WIDTH == 0) p.lse[h] = l > 0 ? m + log2f(l) : -CUDART_INF_F;
        }
        cluster.sync();
    }
}

template<int WIDTH, int WARPS, int CLUSTER, bool PIPE>
int launch(Params p, int rows, cudaStream_t stream) {
    constexpr int BYTES = 16 * (16 + WIDTH / 8) * 16 * WARPS * (PIPE ? 2 : 1);
    constexpr int REDUCE = (WARPS + 1) * (WIDTH * 8 + 16) * sizeof(float);
    constexpr int SMEM = BYTES > REDUCE ? BYTES : REDUCE;
    auto fn = decode<WIDTH, WARPS, CLUSTER, PIPE>;
    if constexpr (SMEM > 49152) {
        const auto status = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (status != cudaSuccess) return status;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(4 * (128 / WIDTH) * CLUSTER, rows);
    cfg.blockDim = dim3(32 * WARPS);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute attr{};
    if constexpr (CLUSTER > 1) {
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {CLUSTER, 1, 1};
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&cfg, fn, p);
}

extern "C" int run_native(const void* q, const void* k, const void* v, const void* indptr,
                          const void* indices, void* output, void* lse, float scale,
                          int rows, int variant, void* stream) {
    Params p{static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v),
             static_cast<const int*>(indptr), static_cast<const int*>(indices),
             static_cast<bf16*>(output), static_cast<float*>(lse), scale};
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (variant) {
        case 0: return launch<64, 4, 4, true>(p, rows, st);
        case 1: return launch<128, 4, 8, true>(p, rows, st);
        case 2: return launch<128, 4, 4, true>(p, rows, st);
    }
    return cudaErrorInvalidValue;
}
'''


@lru_cache(maxsize=None)
def _compile_native(capability):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError("CUDA compiler is required for clustered paged decode")
        nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc")
    major, minor = capability
    with tempfile.TemporaryDirectory(prefix="gqa_decode_build_") as work:
        source = Path(work) / "decode.cu"
        binary = Path(work) / "decode.so"
        source.write_text(_CUDA_SOURCE)
        subprocess.run(
            [nvcc, f"-arch=sm_{major}{minor}", "-O3", "--use_fast_math",
             "-std=c++17", "--shared", "-Xcompiler=-fPIC", str(source), "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=100,
        )
        library = ctypes.CDLL(str(binary))
    launch = library.run_native
    launch.argtypes = [ctypes.c_void_p] * 7 + [
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    return launch


def _run_cluster(q, k, v, indptr, indices, sm_scale, output, lse, variant):
    launch = _compile_native(torch.cuda.get_device_capability(q.device))
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device)
        status = launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), indptr.data_ptr(),
                        indices.data_ptr(), output.data_ptr(), lse.data_ptr(),
                        sm_scale * 1.4426950408889634, q.shape[0], variant, stream.cuda_stream)
    if status:
        raise RuntimeError(f"Clustered paged decode launch failed with CUDA error {status}")


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch, tokens = q.shape[0], kv_indices.numel()
    args = q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse
    if batch == 1 and tokens > 128:
        _run_cluster(*args, 0 if tokens <= 256 else 1)
    elif 1 < batch <= 16:
        if tokens < 256 * batch:
            variant = 0
        elif tokens < 768 * batch:
            variant = 2
        else:
            variant = 1
        _run_cluster(*args, variant)
    elif batch == 64:
        _run_cluster(*args, 2)
    else:
        _run_triton(*args)
