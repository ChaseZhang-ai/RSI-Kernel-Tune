"""KernelWiki artifact PR-1236 (Dao-AILab/flash-attention)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/hopper/benchmark_flash_attention_fp8.py': r'''# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func
from flash_attn_interface import flash_attn_func, _flash_attn_forward

try:
    from triton_fused_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    import cudnn
except ImportError:
    cudnn = None


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E5M2
    else:
        raise ValueError("Unsupported tensor data type.")

def cudnn_spda_setup(qkv, seqlen_q, seqlen_k, causal=False):
    b, _, _, nheads, headdim = qkv.shape
    assert cudnn is not None, 'CUDNN is not available'
    o_gpu = torch.zeros(b, seqlen_q, nheads, headdim, dtype=qkv.dtype, device=qkv.device)
    o_gpu_transposed = torch.as_strided(
        o_gpu,
        [b, nheads, seqlen_q, headdim],
        [nheads * seqlen_q * headdim, headdim, nheads * headdim, 1],
    )
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=qkv.device)
    amax_s_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    amax_o_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(qkv.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    new_q = torch.as_strided(
        qkv,
        [b, nheads, seqlen_q, headdim],
        [seqlen_q * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=0,
    )
    q = graph.tensor(
        name = "Q",
        dim = list(new_q.shape),
        stride = list(new_q.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_k = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim,
    )
    k = graph.tensor(
        name = "K",
        dim = list(new_k.shape),
        stride = list(new_k.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_v = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim * 2,
    )
    v = graph.tensor(
        name = "V",
        dim = list(new_v.shape),
        stride = list(new_v.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )

    def get_default_scale_tensor():
        return graph.tensor(
            dim = [1, 1, 1, 1],
            stride = [1, 1, 1, 1],
            data_type=cudnn.data_type.FLOAT
        )

    default_scale_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_q = get_default_scale_tensor()
    descale_k = get_default_scale_tensor()
    descale_v = get_default_scale_tensor()
    descale_s = get_default_scale_tensor()
    scale_s = get_default_scale_tensor()
    scale_o = get_default_scale_tensor()

    o, _, amax_s, amax_o = graph.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        is_inference=True,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        name="sdpa",
    )

    o.set_output(True).set_dim(o_gpu_transposed.shape).set_stride(o_gpu_transposed.stride())

    amax_s.set_output(False).set_dim(amax_s_gpu.shape).set_stride(amax_s_gpu.stride())
    amax_o.set_output(False).set_dim(amax_o_gpu.shape).set_stride(amax_o_gpu.stride())
    # stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: new_q,
        k: new_k,
        v: new_v,
        descale_q: default_scale_gpu,
        descale_k: default_scale_gpu,
        descale_v: default_scale_gpu,
        descale_s: default_scale_gpu,
        scale_s: default_scale_gpu,
        scale_o: default_scale_gpu,
        o: o_gpu_transposed,
        amax_s: amax_s_gpu,
        amax_o: amax_o_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu, amax_o_gpu

    return run


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


torch.manual_seed(0)

repeats = 30
device = 'cuda'
# dtype = torch.float16
dtype = torch.float8_e4m3fn

# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4224), (2, 8448), (1, 8448 * 2)]
bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)]
# bs_seqlen_vals = [(4, 4096), (2, 8192), (1, 8192 * 2)]
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048)]
causal_vals = [False, True]
headdim_vals = [64, 128, 256]
dim = 2048
# dim = 256
dropout_p = 0.0

methods = (["Pytorch", "Flash3"]
        + (["cuDNN"] if cudnn is not None else [])
        # + (["Triton"] if attention_triton is not None else [])
        #    + (["xformers.c"] if xops is not None else [])
        #    + (["xformers.f"] if xops is not None else [])
           )

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            torch.cuda.empty_cache()
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=False) for _ in range(3)]
            
            qkv = torch.stack([q, k, v], dim=2)
            qkv = qkv.to(torch.bfloat16)
            f = time_fwd(attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
            time_f[config, "Pytorch"] = f
            res_baseline = attention_pytorch(qkv, dropout_p, causal=causal)

            if attention_triton is not None:
                q_transposed = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                k_transposed = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                v_transposed = v.transpose(1, 2).contiguous().permute(0, 1, 3, 2).to(torch.float8_e4m3fn)
                scale = 1 / math.sqrt(headdim)
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=5, verbose=False, desc='Triton'
                )
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=repeats, verbose=False, desc='Triton'
                )
                time_f[config, "Triton"] = f
                res = attention_triton(
                    q_transposed, k_transposed, v_transposed.permute(0, 1, 3, 2),
                    causal, scale
                ).half().transpose(1, 2)
                torch.testing.assert_close(res, res_baseline, atol=0.5, rtol=0.5)

            # out = torch.empty_like(q)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            softmax_scale = q.shape[-1] ** (-0.5)
            descale_q = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            descale_k = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            descale_v = torch.tensor([1.0], dtype=torch.float32, device='cuda')

            # f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
            f = time_fwd(
                _flash_attn_forward,
                q, 
                k, 
                v, 
                softmax_scale, 
                causal=causal,
                window_size=(-1,-1),
                descale_q=descale_q, 
                descale_k=descale_k, 
                descale_v=descale_v, 
                repeats=repeats, 
                verbose=False
            )

            # res = flash_attn_func(q, k, v, causal=causal)
            # torch.testing.assert_close(res.half(), res_baseline, atol=0.05, rtol=0.05)

            time_f[config, "Flash3"] = f

            if cudnn is not None:
                qkv_fp8 = qkv.to(dtype)
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                f = time_fwd(
                    cudnn_spda_setup(
                        qkv_fp8, seqlen, seqlen,
                        causal=causal
                    ),
                    repeats=repeats, verbose=False
                )
                time_f[config, "cuDNN"] = f
                # res, amax_o = cudnn_spda_setup(
                #     qkv_fp8, seqlen, seqlen,
                #     causal=causal
                # )()
                # res = res.half()
                # TODO: CUDNN has numerics issues when
                # num_heads=16, dim=128, seq_len=1024, batch_size=2
                # or larger sizes.
                # res_cpu = res.cpu().reshape(-1)
                # res_baseline_cpu = res_baseline.cpu().reshape(-1)
                # print(amax_o)
                # print(res)
                # print(res_baseline)
                # for i in range(len(res_cpu)):
                #     item = res_cpu[i]
                #     item_baseline = res_baseline_cpu[i]
                #     if abs(item - item_baseline) > 0.5:
                #         print(i)
                #         print(item)
                #         print(item_baseline)
                # torch.testing.assert_close(res, res_baseline, atol=0.05, rtol=0.05)

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                #print (time_f[config,method])
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1e3} ms, "
                )


# with open('flash3_attn_time.plk', 'wb') as fp:
#     pickle.dump((time_f, time_b, time_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
''',
    'key-files/hopper/block_info.h': r"""/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
""",
    'key-files/hopper/combine.h': r"""
#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include "cutlass/layout/layout.h"
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class SmemShape, class SmemShapeMaxSplits>
struct SharedStorageLSE {
    cute::array_aligned<Element, cute::size_v<SmemShape>> smem_lse;
    cute::array_aligned<bool, cute::size_v<SmemShapeMaxSplits>> smem_valid_splits;
};

// DONT use Kernel_traits here to avoid redundant compilation.
// template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
template<typename Element, typename ElementAccum, int kHeadDim, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
__global__ void combine_attn_seqk_parallel(Params const params) {
    // using Element = typename Kernel_traits::OutputType;
    // using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = int64_t; // Kernel_traits::index_t
    constexpr int kMaxSplits = 1 << Log_max_splits;
    // constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = 128; //Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    //__shared__ __align__(16) ElementAccum sLSE[kMaxSplits][kBlockM+1];
    extern __shared__  char smem_[];
    using SharedStorage = SharedStorageLSE<ElementAccum, Shape<Int<kMaxSplits>, Int<kBlockM+1>>, Shape<Int<kMaxSplits>>>;
    SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(smem_);
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse.data()), Shape<Int<kMaxSplits>, Int<kBlockM+1>>{});
    Tensor sValidSplits = make_tensor(make_smem_ptr(shared_storage.smem_valid_splits.data()), Shape<Int<kMaxSplits>>{});

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;
    //if (cute::thread0()) print ("final %d %d %d %d\n",  params.b, params.h, params.seqlen_q, params.b * params.h * params.seqlen_q); 

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));

    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = params.seqlenq_ngroups_swapped ? make_stride(params.b, params.seqlen_q * params.b, 1) : make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE(row,col) = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    __syncthreads();

    // Reduce along the kBlockM dimension to determine valid splits (store in SMEM)
    // One thread per split. Know NumThreads = 128 >= NumMaxSplits
    if (tidx < kMaxSplits) {
        bool is_valid_split = false;
        #pragma unroll
        for (int col = 0; col < kBlockM; ++col) {
            if(sLSE(tidx,col) != -INFINITY) {
                is_valid_split = true;
            }
        }
        sValidSplits(tidx) = is_valid_split;
    }
    __syncthreads();
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        //if (bidx == 0 && tidx < 128) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE(row,col) : -INFINITY;

    }
    //return;

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
            gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
        }
    }
    //if (cute::thread0()) printf ("lse_logsum = %f\n", lse_logsum);

    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE(row,col) = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    //if (cute::thread0()) print_tensor (cOaccum);
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        // DONT copy in Oaccum if lse(split) = -inf for all kBlockM.
        if(sValidSplits(split)) {            
            flash::copy</*Is_even_MN=*/false, Is_even_K>(
                gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
            );
            #pragma unroll
            for (int m = 0; m < size<1>(tOrOaccum); ++m) {
                int row = get<0>(tOcOaccum(0, m, 0));
                ElementAccum lse_scale = sLSE(split,row);
                if (lse_scale != 0.f) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                        #pragma unroll
                        for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                            tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                            //tOrO(i, m, k) += tOrOaccum(i, m, k);
                        }
                    }
                }
            //if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE(split, 0), sLSE(split, 1)); print_tensor(tOrOaccum); }
            }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }
     //if (cute::thread0()) { print_tensor(tOrO); }

    Tensor rO = flash::convert_type<Element>(tOrO);
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        //if (cute::thread0()) print ("final %d %d %d %d %d\n", idx, params.b, params.h, params.seqlen_q, params.b * params.h * params.seqlen_q); 
        if (idx < params.b * params.h * params.seqlen_q) {
            //print ("final2\n"); 
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    // TODO: Should check if this is using vectorized store, but it seems pretty fast
                    copy(rO(_, m, k), gO);
                    //if (cute::thread0()) { print ("final\n"); print_tensor(gO); }
                    // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                    // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
                }
            }
        }
    }
}

}
""",
    'key-files/hopper/epilogue_fwd_sm90_tma.hpp': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

// template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename Element_>
template <typename Ktraits, typename Seqlen_traits>
struct CollectiveEpilogueFwd {

    using InputType = typename Ktraits::Element;
    using Element = typename Ktraits::OutputType;    
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int kBlockH = Ktraits::kBlockH;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;    

    static constexpr int kNWarps = Ktraits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr bool Is_WS = Ktraits::Is_WS;

    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumMmaThreads = kNThreads - NumCopyThreads;

    static constexpr bool Is_split = Ktraits::Is_split;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

#ifndef NO_FP8_COLUMN_PERMUTE
    static constexpr bool epi_column_permute = is_same_v<InputType, cutlass::float_e4m3_t>;
#else
    static constexpr bool epi_column_permute = false;
#endif

    using GmemShapeOT = std::conditional_t<
        Is_split,
        typename Seqlen_traits::ShapeOAccumT,
        typename Seqlen_traits::ShapeT
    >;
    using GmemStrideOT = std::conditional_t<
        Is_split,
        typename Seqlen_traits::StrideOAccumT,
        typename Seqlen_traits::StrideT
    >;
    using GmemLayoutOT = std::conditional_t<
        Is_split,
        typename Seqlen_traits::LayoutOAccumT,
        typename Seqlen_traits::LayoutT
    >;

    using GmemLayoutLseT = std::conditional_t<
        Is_split,
        typename Seqlen_traits::LayoutLseAccumT,
        typename Seqlen_traits::LayoutLseT
    >;

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutOCopy = typename Ktraits::SmemLayoutOCopy;
    using TileShapeOCopy = typename Ktraits::TileShapeOCopy;

    using SmemCopyAtomO = std::conditional_t<Is_split, 
        Copy_Atom<UniversalCopy<Element>, Element>, Copy_Atom<cute::SM90_U32x4_STSM_N, Element>>;
    using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>>;

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;
    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(
            make_gmem_ptr(static_cast<Element*>(nullptr)), 
            GmemShapeOT{}, 
            GmemStrideOT{}
        ),
        SmemLayoutOCopy{},
        TileShapeOCopy{},
        _1{}));  // no mcast for O

    // These are for storing the output tensor without TMA (e.g., for setting output to zero and var-seq-len)
    static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<Element>);
    static_assert(kHeadDim % kNumVecElem == 0);
    static constexpr int kNumThreadsPerRow = kHeadDim / kNumVecElem;
    static_assert(NumMmaThreads % kNumThreadsPerRow == 0);
    static constexpr int kNumRows = NumMmaThreads / kNumThreadsPerRow;
    using TiledCopyOAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, Element>;
    using TiledCopyOThrLayout = decltype(cute::make_layout(
        cute::make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}),
        LayoutRight{}));
    using TiledCopyOValLayout = decltype(cute::make_layout(
        cute::make_shape(_1{}, Int<kNumVecElem>{}),
        LayoutRight{}));
    using TiledCopyO = decltype(make_tiled_copy(
        TiledCopyOAtom{},
        TiledCopyOThrLayout{}, // Thr layout
        TiledCopyOValLayout{} // Val layout
    ));

    // used for rmem -> smem O copy in fp8 kernel to undo column permutation
    using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                 Stride<_4, _32, _1, _0>>;
    using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim/16>>,
                                Stride<_0, _2, Stride<_4, _1>, _8>>;
    using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                      ThreadLayoutrO{}, ValueLayoutrO{}));
    using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDim/16>>;
    using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_O;
        GmemLayoutOT const layout_O;
        float* ptr_LSE;
        GmemLayoutLseT const layout_LSE;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_O;
        GmemLayoutOT const layout_O;
        float* ptr_LSE;
        GmemLayoutLseT const layout_LSE;
        TMA_O tma_store_O;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.layout_O);
        TMA_O tma_store_O = make_tma_copy(
            GmemTiledCopyOTMA{},
            mO,
            SmemLayoutOCopy{},
            TileShapeOCopy{},
            _1{}); // no mcast for O
        return {args.ptr_O, args.layout_O, args.ptr_LSE, args.layout_LSE, tma_store_O};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params) {
        if constexpr (!Seqlen_traits::UseVarSeqLen && !No_smem_O) {
            cute::prefetch_tma_descriptor(epilogue_params.tma_store_O.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& epilogue_params,
          FrgTensorO const& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord,
          const Seqlen_traits& seqlen_traits_q,
          const cutlass::FastDivmod& qhead_per_khead_divmod
          ) {

        auto [m_block, n_split_idx, bidh, bidb] = block_coord;
        const int bidh_kv = qhead_per_khead_divmod.divide(bidh);
        const int h_block = bidh % int(qhead_per_khead_divmod);

        Tensor tOrO_out = flash::convert_type<Element>(tOrO);
        if constexpr(!No_smem_O) {
            if constexpr (!epi_column_permute) {
                Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
                auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
                auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

                Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);  // ((Atom,AtomNum), MMA_M, MMA_N)
                Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

                // Make sure all WGs have finished reading V
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty) /*id*/);
                cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
                cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            } else {
                TiledCopyrO rmem_tiled_copy_O;
                Tensor sOacc = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutrO{});
                auto rmem_thr_copy_O = rmem_tiled_copy_O.get_thread_slice(thread_idx);
                
                Tensor taccOsO = rmem_thr_copy_O.partition_D(sOacc);
                Tensor taccOrO = make_tensor(tOrO_out.data(), shape(taccOsO));

                // Make sure all WGs have finished reading V
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty) /*id*/);        
                cute::copy(rmem_tiled_copy_O, taccOrO, taccOsO);
                cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
        }

        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), epilogue_params.layout_LSE);
        Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        // taccOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
        Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // 2 * MMA_M        
        
        if constexpr(!Seqlen_traits::UseGQAPacking) {
            Tensor gLSE = seqlen_traits_q.get_lse_local_tile_tensor<Is_split>(
                mLSE, Shape<Int<kBlockM>>{}, bidh, bidb, n_split_idx)(_, m_block);
            if (get<1>(taccOcO_row(_0{})) == 0) {
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    const int row = get<0>(taccOcO_row(mi));                
                    if (row < seqlen_traits_q.actual_seq_len - m_block * kBlockM) {
                        gLSE(row) = lse(mi);
                    }
                }
            }
        } else {
            // shape<1>(epilogue_params.layout_O) == h/h_k
            // In common case where ceil_div(h/h_k, kBlockH) == 1,
            // int(qhead_per_khead_divmod) == 1, bidh_kv == bidh, h_block == 0
            const int h_offset = shape<1>(epilogue_params.layout_O) * bidh_kv +
                    h_block * kBlockH;
            const int m_bound = seqlen_traits_q.actual_seq_len - m_block * (kBlockM/kBlockH);
            const int h_bound = shape<1>(epilogue_params.layout_O) - h_block * kBlockH;
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));                
                const int h_local = row % kBlockH;
                const int m_local = row/kBlockH;             
                if(h_local < h_bound && m_local < m_bound) {
                    Tensor gLSE = seqlen_traits_q.get_lse_local_tile_tensor<Is_split>(mLSE,
                        Shape<Int<kBlockM/kBlockH>>{}, h_offset + h_local, bidb, n_split_idx)
                        (_, m_block);
                    gLSE(m_local) = lse(mi);
                }
            }
        }
       
        if constexpr (No_smem_O) { 
            flash::write_rmem_to_gmem<Seqlen_traits::UseGQAPacking, epi_column_permute>(
                tOrO_out, epilogue_params.ptr_O, epilogue_params.layout_O, TileShapeOCopy{}, 
                m_block, h_block, bidh, bidh_kv, bidb, n_split_idx,
                tiled_mma, seqlen_traits_q, thread_idx);
        } else {
            int write_warp_idx = kNWarps - 1;
            if (cutlass::canonical_warp_idx_sync() == write_warp_idx) {
                cutlass::arch::NamedBarrier::sync(
                    NumMmaThreads + cutlass::NumThreadsPerWarp, 
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
                );
            }
            TiledCopyO gmem_tiled_copy_O;
            Tensor sO_out = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutOCopy{});        
            if constexpr(!Seqlen_traits::UseGQAPacking) {
                flash::write_O<!Seqlen_traits::UseVarSeqLen, No_smem_O, Is_split, NumCopyThreads>(
                    epilogue_params.ptr_O, epilogue_params.tma_store_O, gmem_tiled_copy_O, 
                    epilogue_params.layout_O, TileShapeOCopy{}, sO_out, 
                    m_block, bidh, bidb, n_split_idx, seqlen_traits_q, write_warp_idx, tiled_mma, tOrO_out
                );
            } else {
                Tensor mO = epilogue_params.tma_store_O.get_tma_tensor(epilogue_params.layout_O.shape());
                Tensor gO = seqlen_traits_q.get_o_local_tile_tensor<Is_split>(
                    mO, TileShapeOCopy{}, bidh_kv, bidb, n_split_idx)
                    (_, _, _, m_block, h_block);  // (bM/bH, bH, K)
                auto block_tma_O = epilogue_params.tma_store_O.get_slice(_0{});
                Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
                Tensor tOsO = block_tma_O.partition_S(sO_out);  // (TMA, TMA_M, TMA_K)
                int const lane_predicate = cute::elect_one_sync();
                int const warp_idx = cutlass::canonical_warp_idx_sync();
                if (warp_idx == write_warp_idx && lane_predicate) {
                    cute::copy(epilogue_params.tma_store_O, tOsO, tOgO);
                    tma_store_arrive();
                }
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        if constexpr(!No_smem_O) { tma_store_wait<0>(); }
    }

    // Write 0 to output and -inf to LSE
    template<typename SharedStorage>
    CUTLASS_DEVICE void
    store_zero(
          Params const& epilogue_params,
          SharedStorage& shared_storage,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord,
          const Seqlen_traits& seqlen_traits_q
          ) {
        static_assert(!Seqlen_traits::UseGQAPacking, "Don't call store_zero for gqa packed layouts.");
        auto [m_block, n_split_idx, bidh, bidb] = block_coord;

        if constexpr(!Is_split) {
            Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.layout_O);
            Tensor gO = seqlen_traits_q.get_o_local_tile_tensor<Is_split>(
                mO, select<0, 2>(TileShape_MNK{}), bidh, bidb, n_split_idx
            )(_, _, m_block);  // (M, K)

            TiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
            Tensor tOrO = make_fragment_like(tOgO);
            clear(tOrO);
            // Construct identity layout for sO
            Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.layout_O.shape()); }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_traits_q.actual_seq_len - m_block * kBlockM
            );
        }
        
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), epilogue_params.layout_LSE);
        Tensor gLSE = seqlen_traits_q.get_lse_local_tile_tensor<Is_split>(
            mLSE, Shape<Int<kBlockM>>{}, bidh, bidb, n_split_idx)(_, m_block);
        static_assert(kBlockM <= NumMmaThreads);
        if (thread_idx < min(kBlockM, seqlen_traits_q.actual_seq_len - m_block * kBlockM)) {
            gLSE(thread_idx) = !Is_split ? INFINITY : -INFINITY;
        }
    }

    // Write 0 to output and -inf to LSE
    template<typename SharedStorage>
    CUTLASS_DEVICE void
    store_zero_gqa(
          Params const& epilogue_params,
          SharedStorage& shared_storage,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord,
          const Seqlen_traits& seqlen_traits_q,
          const cutlass::FastDivmod& qhead_per_khead_divmod
          ) {
        static_assert(Seqlen_traits::UseGQAPacking, "Special store_zero method for GQA packed layouts.");
        auto [m_block, n_split_idx, bidh, bidb] = block_coord;
        const int bidh_kv = qhead_per_khead_divmod.divide(bidh);
        const int h_block = bidh % int(qhead_per_khead_divmod);        
        const int h_bound = min(shape<1>(epilogue_params.layout_O) - h_block * kBlockH, kBlockH);
        const int m_bound = min(seqlen_traits_q.actual_seq_len - m_block * (kBlockM/kBlockH), kBlockM/kBlockH);
        
        if constexpr(!Is_split) {
            Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.layout_O);
            Tensor gO = seqlen_traits_q.get_o_local_tile_tensor<Is_split>(
                        mO, TileShapeOCopy{}, bidh_kv, bidb, n_split_idx)
                            (_, _, _, m_block, h_block); // (bM/bH, bH, K)
            TiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            if constexpr(kNumRows <= kBlockH) {
                // slice into bM/bH and write out zero tiles (bH, K)
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO(0,_,_));
                Tensor tOrO = make_fragment_like(tOgO);
                clear(tOrO);
                Tensor cO = cute::make_identity_tensor(select<1, 2>(TileShapeOCopy{}));
                Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
                // dummy predicate, unused since Is_even_K=true
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
                #pragma unroll
                for(int m = 0; m < m_bound; ++m) {                
                    tOgO = gmem_thr_copy_O.partition_D(gO(m,_,_));
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, h_bound
                    );
                }
            } else {
                // slice into bH and write out zero tiles (bM/bH, K)
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_,0,_));
                Tensor tOrO = make_fragment_like(tOgO);
                clear(tOrO);
                Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShapeOCopy{}));
                Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
                // dummy predicate, unused since Is_even_K=true
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
                #pragma unroll
                for(int h = 0; h < h_bound; ++h) {                
                    tOgO = gmem_thr_copy_O.partition_D(gO(_,h,_));
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true,
                                /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, m_bound
                    );
                }
            }
        }

        const int h_offset = shape<1>(epilogue_params.layout_O) * bidh_kv + h_block * kBlockH;
        const int thread_idx_h = thread_idx % kBlockH;
        const int thread_idx_m = thread_idx / kBlockH;
        
        Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), epilogue_params.layout_LSE);
        Tensor gLSE = seqlen_traits_q.get_lse_local_tile_tensor<Is_split>(
            mLSE, Shape<Int<kBlockM/kBlockH>>{}, h_offset + thread_idx_h, bidb, n_split_idx)(_, m_block);
        if(thread_idx_h < h_bound && thread_idx_m < m_bound) {
            gLSE(thread_idx_m) = !Is_split ? INFINITY : -INFINITY;
        }
    }

};

} // namespace flash
""",
    'key-files/hopper/flash.h': r"""/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The stride between rows of Oaccum.
    index_t oaccum_batch_stride;
    index_t oaccum_row_stride;
    index_t oaccum_head_stride;
    index_t oaccum_split_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q, total_k;
    int b_k;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
    uint32_t scale_softmax_log2_half2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    // If provided, the actual length of each q / o sequence.
    int * __restrict__ seqused_q;
    // If provided, the actual length of each k / v sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_e4m3;
    bool is_causal;
    bool is_local;
    bool is_kv_cache;
    bool use_gqa_packing;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse; // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    int * __restrict__ tile_count_semaphore;
    float * __restrict__ descale_q_ptr;
    float * __restrict__ descale_k_ptr;
    float * __restrict__ descale_v_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;

    int *__restrict__ dq_semaphore;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, int kBlockH> void run_mha_fwd_gqa_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
""",
    'key-files/hopper/flash_api.cpp': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t b_k,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped=false,
                      bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
    params.is_kv_cache = false;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    TORCH_CHECK(
        bool(params.cu_seqlens_q) == bool(params.cu_seqlens_k),
        "cu_seqlens_q and cu_seqlens_k must be both null or non-null"
    );

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.b_k = b_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.    
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    window_size_left = std::min(int(seqlen_k), window_size_left);
    window_size_right = std::min(int(seqlen_k), window_size_right);
    if (window_size_left < 0) { window_size_left = seqlen_k; }
    if (window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_causal = window_size_left == int(seqlen_k) && window_size_right == 0;
    if ((window_size_left < int(seqlen_k) || window_size_right < int(seqlen_k)) && !params.is_causal) {
        params.is_local = true;
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic) {

    set_params_fprop(params,
                     b, b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_q,
                     seqused_k,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 80%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int batch_nheads, int num_SMs, int num_n_blocks,
    int max_splits, int head_size, bool use_one_mma_wg) {
    // Goal of the starting threshold is to determine whether to split or not.
    // Empirically, the efficiency threshold can be much lower than 80% depending on num_n_blocks.
    int num_m_blocks = batch_nheads_mblocks/batch_nheads;
    float start_threshold;
    float num_n_blocksf = float(num_n_blocks);
    if (head_size == 128) {
        if (std::log2f(num_n_blocksf) <= 4) { // 2048 -- .25
            start_threshold = .20f + (std::log2f(num_n_blocksf) - 3) * .05f;
        } else if (std::log2f(num_n_blocksf) <= 5) { // 4096 -- .25
            start_threshold = .25f;
        } else if (std::log2f(num_n_blocksf) <= 6) { // 8192 -- .36
            start_threshold = .28f + (std::log2f(num_n_blocksf) - 5) * .08f;
        } else if (std::log2f(num_n_blocksf) <= 7) { // 16K -- .42
            start_threshold = .36f + (std::log2f(num_n_blocksf) - 6) * .06f;
        } else {
            // Just split freely
            start_threshold = .8f;
        }
        if (num_m_blocks > 1 && start_threshold < .5f)
            start_threshold += .05f * (std::log2f(num_n_blocksf) - 2);
    } else if (head_size == 256) {
        // TODO for hdim 256
        if (num_n_blocks <= 40) {
            start_threshold = .24f;
        } else if (std::log2f(num_n_blocksf) <= 8) {
            start_threshold = .33f + std::max(0.f, (std::log2f(num_n_blocksf) - std::log2f(50)) * 0.02971f);
        } else {
            // Just split freely
            start_threshold = .8f;
        }
    } else if (head_size == 64) {
        if (use_one_mma_wg) {
            if (std::log2f(num_n_blocksf) <= 4) { // 2K -- .33
                start_threshold = .33f;
            } else if (std::log2f(num_n_blocksf) <= 5) { // 4K -- .37
                start_threshold = .33f + (std::log2f(num_n_blocksf) - 4) * .04f;
            } else if (std::log2f(num_n_blocksf) <= 6) { // 8K -- .40
                start_threshold = .37f + (std::log2f(num_n_blocksf) - 5) * .03f;
            } else if (std::log2f(num_n_blocksf) <= 7) { // 16K -- .43
                start_threshold = .4f + (std::log2f(num_n_blocksf) - 6) * .03f;
            } else if (std::log2f(num_n_blocksf) <= 8) { // 32K -- .46
                start_threshold = .43f + (std::log2f(num_n_blocksf) - 7) * .03f;
            } else {
                start_threshold = .8f;
            }
        } else {
            if (std::log2f(num_n_blocksf) <= 6) { // 8K -- .5
                start_threshold = .5f;
            } else {
                start_threshold = .8f;
            }
        }
    } else {
        // placeholder for other hdims
        start_threshold = .8f;
    }

    float first_wave = float(batch_nheads_mblocks) / num_SMs;
    // printf("Start threshold and wave = %f, %f.\n", start_threshold, first_wave);
    // Only use start_threshold if initial work doesn't exceed one wave
    if ((first_wave/ceil(first_wave) > start_threshold && first_wave <= 1.f) ||
        (first_wave/ceil(first_wave) > .8f)) {
        return 1;
    }
    // if (first_wave_batch_nheads > start_threshold) { return 1; }
    // if (first_wave_batch_nheads > start_threshold || first_wave > .8f) { return 1; }
    // if (float(batch_nheads)/num_SMs > start_threshold) { return 1; }

    // If num_n_blocks is too small, use 1 split
    // For example, we never split for hdim = 128 and seqlen_k = 512,
    // or for hdim = 128, seqlen_k = 1024, and one MMA warpgroup.
    if (num_n_blocks < 8 || (use_one_mma_wg && num_n_blocks < 10)) { return 1; }

    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    
    // NOTE: disable split eligibility check for FA3 since we have dynamic tile scheduler
    // for exiting splits with no work early, and check leads to efficiency quantization issues.
    // Comment from FA2:
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    // auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    //     return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    // };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        // if (!is_split_eligible(num_splits)) {
        //     efficiency.push_back(0.f);
        // } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, n_waves = %f, ceil(n_waves) = %f,  eff = %f\n", num_splits, n_waves, ceil(n_waves), eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        // }
    }
    // Correct for excessive splitting with e.g. 1 bsz*nheads*mblocks
    // Empirically, efficiency threshold in these cases is about 40% for 64K seqlen_k
    float threshold = num_m_blocks == 1 ? std::min(0.3f + batch_nheads * 0.1f, 0.8f) : 0.8f;
    threshold = threshold * max_efficiency;
    // printf("Max efficiency = %f. Threshold = %f.\n", max_efficiency, threshold);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        // if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] > threshold) {
            // printf("num_splits chosen = %d, threshold = %f, efficiency = %f.\n", num_splits, threshold, efficiency[num_splits - 1]);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int num_heads_k, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, bool use_gqa_packing, bool is_causal, struct c10::TensorOptions opts) {
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            const int gqa_ratio = num_heads / num_heads_k;
            const int block_h = 1 << static_cast<int>(std::ceil(std::log2(std::clamp(gqa_ratio, 1, 32))));
            const int block_m = head_size == 64 ? 192 : 128;
            const bool use_one_mma_wg = max_seqlen_q <= 64/block_h;
            
            int block_n = 128;
            if (head_size == 128 && !is_causal) {
                block_n = 176;
            } else if (head_size == 256) {
                block_n = use_one_mma_wg ? 96 : 80;
            }
            const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
            const int batch_nheads = use_gqa_packing ? batch_size * num_heads_k : batch_size * num_heads;
            const int batch_nheads_mblocks = use_gqa_packing
                ? ceildiv(max_seqlen_q, block_m / block_h) * batch_nheads
                : ceildiv(max_seqlen_q, block_m) * batch_nheads;
            params.num_splits = num_splits_heuristic(batch_nheads_mblocks, batch_nheads,
                dprops->multiProcessorCount, num_n_blocks, 128, head_size, use_one_mma_wg);
            // printf("Num splits heuristic = %d.\n", params.num_splits);
	    }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
            params.oaccum_row_stride = out_accum.stride(-2);
            params.oaccum_head_stride = out_accum.stride(-3);
            params.oaccum_batch_stride = out_accum.stride(-4);
            params.oaccum_split_stride = out_accum.stride(0);
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}


void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) { 

    int dtype = 1;
    if (params.is_bf16) { dtype = 2; }
    else if (params.is_e4m3) { dtype = 3; }
    PREC_SWITCH(dtype, Element, [&] {
      HEADDIM_SWITCH(params.d, kHeadSize, [&] {
        if(!params.use_gqa_packing) {
          run_mha_fwd_<Element, kHeadSize>(params, stream);
        } else {
          QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
            run_mha_fwd_gqa_<Element, kHeadSize, kBlockH>(params, stream);
          });
        }
      });
    });

#if 0
    if (!params.is_e4m3) { 
        if (params.is_bf16) {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::bfloat16_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::bfloat16_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::bfloat16_t, 256>(params, stream);
            }
        } else {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::half_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::half_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::half_t, 256>(params, stream);
            }
        }
    } else {
        if (params.d == 64) {
            run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
        } else if (params.d == 128) {
            run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
        } else if (params.d == 256) {
            run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
        }
    }
#endif
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        c10::optional<at::Tensor> &descale_q_, // 1
        c10::optional<at::Tensor> &descale_k_, // 1
        c10::optional<at::Tensor> &descale_v_, // 1
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool use_gqa_packing = false
        ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention-3 only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
                "FlashAttention-3 only support fp16, bf16, or fp8 e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // Guard against mistaken setting of gqa flag
    if (num_heads == num_heads_k) { use_gqa_packing = false; }

    TORCH_CHECK(head_size_og == 64 || head_size_og == 128 || head_size_og == 256, "Only support head size 64, 128, and 256 for now");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn
                    ? (out.dtype() == at::kBFloat16)
                    : (out.dtype() == q_dtype),
                "Output must have the same dtype as input dtype if dtype is "
                "not fp8, or fp16 for fp8 input.");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (q_dtype == at::ScalarType::Float8_e4m3fn)
            out = torch::empty_like(q_padded, at::kBFloat16);
        else
            out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right);

    auto tile_count_semaphore = is_causal || params.is_local
        ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    at::Tensor descale_q, descale_k, descale_v;
    if(q_dtype == at::ScalarType::Float8_e4m3fn) {
        if (descale_q_.has_value()) {
            descale_q = descale_q_.value();
            CHECK_DEVICE(descale_q);
            CHECK_SHAPE(descale_q, 1);
        } else { descale_q = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_k_.has_value()) {
            descale_k = descale_k_.value();
            CHECK_DEVICE(descale_k);
            CHECK_SHAPE(descale_k, 1);
        } else { descale_k = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_v_.has_value()) {
            descale_v = descale_v_.value();
            CHECK_DEVICE(descale_v);
            CHECK_SHAPE(descale_v, 1);
        } else { descale_v = torch::ones({1}, opts.dtype(at::kFloat)); }
        params.descale_q_ptr = descale_q.data_ptr<float>();
        params.descale_k_ptr = descale_k.data_ptr<float>();
        params.descale_v_ptr = descale_v.data_ptr<float>();
    } else {
        params.descale_q_ptr = nullptr;
        params.descale_k_ptr = nullptr;
        params.descale_v_ptr = nullptr;
    }
    
    params.use_gqa_packing = use_gqa_packing;

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               int window_size_left,
               int window_size_right) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int num_heads_k = k.size(1);

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    if (seqused_q.has_value()){
        auto seqused_q_ = seqused_q.value();
        TORCH_CHECK(seqused_q_.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q_.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q_.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q_, batch_size);
    }

    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     /*p_d=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     /*seqlenq_ngroups_swapped=*/false,
                     /*unpadded_lse=*/true);
    params.total_q = total_q;
    params.total_k = total_k;

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  if (!params.is_bf16) {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::half_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::half_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::half_t, 128>(params, stream);
    }
  } else {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::bfloat16_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::bfloat16_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::bfloat16_t, 128>(params, stream);
    }
  }
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : round_multiple(head_size, 32);
    // This should match the kernel configs
    const int kBlockM = head_size <= 64 ? 128 : (head_size < 256 ? 64 : 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    // dk_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));
    // dv_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if (is_causal) { window_size_right = 0; }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     dq_accum.data_ptr(),
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right,
                     deterministic);
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    // printf("dq_semaphore: %p, [%d, %d, %d]\n", params.dq_semaphore, (seqlen_q + 64 - 1) / 64, batch_size, num_heads);

    if (seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum};
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
               const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x seqlen_q
               c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
               c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float softmax_scale,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : round_multiple(head_size, 32);
    // This should match the kernel configs
    const int kBlockM = head_size <= 64 ? 128 : (head_size < 256 ? 64 : 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    int const total_q_padded_rounded = round_multiple(total_q + batch_size * 128, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    if (seqused_q.has_value()){
        auto seqused_q_ = seqused_q.value();
        TORCH_CHECK(seqused_q_.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q_.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q_.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q_, batch_size);
    }

    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    dq_accum = torch::empty({num_heads, total_q_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    // dk_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));
    // dv_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     dq_accum.data_ptr(),
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right,
                     deterministic);
    params.total_q = total_q;
    params.total_k = total_k;
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();

    if (max_seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If max_seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum, softmax_lse_log2 };
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                c10::optional<at::Tensor> &descale_q_, // 1
                c10::optional<at::Tensor> &descale_k_, // 1
                c10::optional<at::Tensor> &descale_v_, // 1
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                int max_seqlen_k_hint,
                bool use_gqa_packing
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention-3 only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
                "FlashAttention-3 only support fp16, bf16, or fp8 e4m3 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // Guard against mistaken setting of gqa flag
    if (num_heads == num_heads_k) { use_gqa_packing = false; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped =
        seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
        window_size_right < 0 && head_size_og % 8 == 0 &&
        !alibi_slopes_.has_value() && !use_gqa_packing;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn
                    ? (out.dtype() == at::kBFloat16)
                    : (out.dtype() == q_dtype),
                "Output must have the same dtype as input dtype if dtype is "
                "not fp8, or fp16 for fp8 input.");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (q_dtype == at::ScalarType::Float8_e4m3fn) {
            out = torch::empty_like(q_padded, at::kBFloat16);
        }
        else
            out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size_c,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right
                     );

    at::Tensor descale_q, descale_k, descale_v;
    if(q_dtype == at::ScalarType::Float8_e4m3fn) {
        if (descale_q_.has_value()) {
            descale_q = descale_q_.value();
            CHECK_DEVICE(descale_q);
            CHECK_SHAPE(descale_q, 1);
        } else { descale_q = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_k_.has_value()) {
            descale_k = descale_k_.value();
            CHECK_DEVICE(descale_k);
            CHECK_SHAPE(descale_k, 1);
        } else { descale_k = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_v_.has_value()) {
            descale_v = descale_v_.value();
            CHECK_DEVICE(descale_v);
            CHECK_SHAPE(descale_v, 1);
        } else { descale_v = torch::ones({1}, opts.dtype(at::kFloat)); }
        params.descale_q_ptr = descale_q.data_ptr<float>();
        params.descale_k_ptr = descale_k.data_ptr<float>();
        params.descale_v_ptr = descale_v.data_ptr<float>();
    } else {
        params.descale_q_ptr = nullptr;
        params.descale_k_ptr = nullptr;
        params.descale_v_ptr = nullptr;
    }
    
    params.is_kv_cache = true;

    params.use_gqa_packing = use_gqa_packing;

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);
        params.seqused_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        TORCH_CHECK(false, "Left Padding K is not supported");
        //params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
       params, batch_size, num_heads, num_heads_k, head_size, max_seqlen_k_hint, seqlen_q,
       head_size_rounded, /*dropout*/ 0.f, num_splits, dprops, use_gqa_packing, is_causal, opts);
    
    auto tile_count_semaphore = is_causal || params.is_local || params.num_splits != 1
        ? torch::zeros({1}, opts.dtype(torch::kInt32))
        : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;

    TORCH_CHECK(!alibi_slopes_.has_value(), "Alibi Slopes are not supported yet");
    //set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    //run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
    run_mha_fwd(params, stream);

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }

    return {out, softmax_lse};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}
""",
    'key-files/hopper/flash_fwd_hdim128_bf16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 128, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::bfloat16_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_bf16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 128, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::bfloat16_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_bf16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 128, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::bfloat16_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_bf16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 128, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::bfloat16_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_bf16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 128, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::bfloat16_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_e4m3_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 128, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp8_gqa<cutlass::float_e4m3_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_e4m3_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 128, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp8_gqa<cutlass::float_e4m3_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_e4m3_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 128, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp8_gqa<cutlass::float_e4m3_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_e4m3_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 128, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp8_gqa<cutlass::float_e4m3_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_e4m3_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 128, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp8_gqa<cutlass::float_e4m3_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_fp16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 128, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::half_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_fp16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 128, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::half_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_fp16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 128, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::half_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_fp16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 128, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::half_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim128_fp16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 128, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_gqa<cutlass::half_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_bf16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 256, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::bfloat16_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_bf16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 256, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::bfloat16_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_bf16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 256, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::bfloat16_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_bf16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 256, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::bfloat16_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_bf16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 256, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::bfloat16_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_e4m3_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 256, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8_gqa<cutlass::float_e4m3_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_e4m3_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 256, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8_gqa<cutlass::float_e4m3_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_e4m3_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 256, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8_gqa<cutlass::float_e4m3_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_e4m3_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 256, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8_gqa<cutlass::float_e4m3_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_e4m3_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 256, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_fp8_gqa<cutlass::float_e4m3_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_fp16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 256, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::half_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_fp16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 256, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::half_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_fp16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 256, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::half_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_fp16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 256, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::half_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim256_fp16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 256, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256_gqa<cutlass::half_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_bf16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 64, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::bfloat16_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_bf16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 64, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::bfloat16_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_bf16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 64, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::bfloat16_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_bf16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 64, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::bfloat16_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_bf16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::bfloat16_t, 64, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::bfloat16_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_e4m3_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 64, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8_gqa<cutlass::float_e4m3_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_e4m3_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 64, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8_gqa<cutlass::float_e4m3_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_e4m3_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 64, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8_gqa<cutlass::float_e4m3_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_e4m3_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 64, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8_gqa<cutlass::float_e4m3_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_e4m3_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::float_e4m3_t, 64, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8_gqa<cutlass::float_e4m3_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_fp16_gqa16_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 64, 16>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::half_t, 16>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_fp16_gqa2_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 64, 2>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::half_t, 2>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_fp16_gqa32_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 64, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::half_t, 32>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_fp16_gqa4_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 64, 4>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::half_t, 4>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_hdim64_fp16_gqa8_sm90.cu': r"""// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_gqa_<cutlass::half_t, 64, 8>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_gqa<cutlass::half_t, 8>(params, stream);
}
""",
    'key-files/hopper/flash_fwd_kernel.h': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "flash.h"
#include "utils.h"
#include "softmax.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, bool Is_local, typename TileScheduler, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                    Seqlen_traits_Q seqlen_traits_q, Seqlen_traits seqlen_traits_k
                    ) {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockH = Ktraits::kBlockH;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kHeadDim = Ktraits::kHeadDim;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        if constexpr (!No_smem_O) { shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/); }
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    // static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    static_assert(Ktraits::kNWarps == 8 || Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 24 : 32>();

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            TileScheduler scheduler(&shared_storage.tile_count_semaphore);
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(scheduler_params);
                auto [m_block, n_split_idx, bidh, bidb] = block_coord;

                seqlen_traits_q.init(bidb);
                seqlen_traits_k.init(bidb);
                if constexpr(seqlen_traits_q.UseVarSeqLen) {
                    // NOTE: to support in future with gqa packed layouts, changed kBlockM to kBlockM/kBlockH
                    if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                        continue;
                    }
                }
                int n_block_min = 0, n_block_max;
                collective_mainloop.get_n_block_min_max(
                        mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                        n_block_min, n_block_max);
                if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                    if(n_block_max <= n_block_min) {
                        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                        scheduler.broadcast_next_work(work_tile_info);
                        continue;
                    }
                }
                collective_mainloop.load(
                    mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
                    shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
                    seqlen_traits_q, seqlen_traits_k, n_block_min, n_block_max);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
        }
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 16 ? 160 : Ktraits::kNWarps == 12 ? 240 : 256>();

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;

        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
        // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

        collective_mainloop.mma_init();
        scheduler.init_consumer();

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax(mainloop_params.softmax_scale_log2);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            seqlen_traits_q.init(bidb);
            seqlen_traits_k.init(bidb);
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layouts, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_max, n_block_min = 0;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                    if constexpr(!Seqlen_traits_Q::UseGQAPacking) {
                        collective_epilogue.store_zero(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q);
                    } else {
                        collective_epilogue.store_zero_gqa(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);
                    }
                    continue;
                }   
            }         

            collective_mainloop.mma(
                mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
                tOrO, softmax, n_block_min, n_block_max, threadIdx.x - NumCopyThreads, work_idx,
                m_block, shared_storage, seqlen_traits_q, seqlen_traits_k);
                // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx, shared_storage);
            collective_epilogue.store(
                epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);

            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

template <typename Ktraits, bool Is_causal, bool Is_local, typename TileScheduler, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws_fp8(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>::Params const mainloop_params,
                        CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>::Params const epilogue_params,
                        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                        Seqlen_traits_Q seqlen_traits_q, Seqlen_traits seqlen_traits_k
                        ) {

    using Element = typename Ktraits::Element;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockH = Ktraits::kBlockH;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr bool Delay_V_release = Is_causal && Ktraits::kHeadDim == 128 && Ktraits::kNWarps != 8;    
    static constexpr bool Use_max_offset = true;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // additional pipeline to synchronize out-of-place smem transpose of V
    PipelineParamsVt pipeline_params_vt;
    pipeline_params_vt.producer_arv_count = NumCopyThreads;
    pipeline_params_vt.consumer_arv_count = NumMmaThreads;
    MainloopPipelineVt pipeline_vt(shared_storage.pipeline_vt, pipeline_params_vt);
    
    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        if constexpr (!No_smem_O) { shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/); }
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    // pipeline_v has producer warpgroup for its consumer in fp8 kernel
    pipeline_params.num_consumers = NumCopyThreads;
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    float descale_q = *mainloop_params.descale_q_ptr;
    float descale_k = *mainloop_params.descale_k_ptr;
    float descale_v = *mainloop_params.descale_v_ptr;
    shared_storage.softmax_scale_qk_log2 = mainloop_params.softmax_scale_log2 * descale_q * descale_k;
    shared_storage.descale_v = descale_v;
    shared_storage.seqlen_init_k = seqlen_traits_k.UseVarSeqLen || bool(seqlen_traits_k.seq_used);

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    static_assert(Ktraits::kNWarps == 8 || Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 16 ? 32 : Ktraits::kNWarps == 12 ? 40 : 56>();
            
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>(); 
        PipelineState smem_pipe_read, smem_pipe_release;

        int work_idx = 0;

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        for (auto work_tile_info = scheduler.get_initial_work();
                work_tile_info.is_valid(scheduler_params);
                work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            if constexpr (seqlen_traits_q.UseVarSeqLen) { seqlen_traits_q.init(bidb); }
            if (shared_storage.seqlen_init_k) { seqlen_traits_k.init_no_guard(bidb); }
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layout, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_min = 0, n_block_max;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local ||seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {
                    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                    scheduler.broadcast_next_work(work_tile_info);
                    // need to sync producer warpgroup
                    cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
                    continue;
                }
            }
            collective_mainloop.load_fp8(
                mainloop_params, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, smem_pipe_read,
                shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
                seqlen_traits_q, seqlen_traits_k, n_block_min, n_block_max);
            ++work_idx;
            // don't need to sync producer warpgroup here
            // if constexpr (Is_causal) {
            //     cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/); }
        }
        collective_mainloop.load_tail_one_write(pipeline_k, pipeline_v, smem_pipe_write);
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 16 ? 160 : Ktraits::kNWarps == 12 ? 232 : 256>();        

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;
        PipelineState smem_pipe_read;
        PipelineState smem_pipe_release;

        collective_mainloop.mma_init();
        scheduler.init_consumer();

        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), Use_max_offset> softmax(shared_storage.softmax_scale_qk_log2);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            if constexpr (seqlen_traits_q.UseVarSeqLen) { seqlen_traits_q.init(bidb); }
            if (shared_storage.seqlen_init_k) { seqlen_traits_k.init_no_guard(bidb); }
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layout, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_max, n_block_min = 0;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                    if constexpr(!Seqlen_traits_Q::UseGQAPacking) {
                        collective_epilogue.store_zero(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q);
                    } else {
                        collective_epilogue.store_zero_gqa(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);
                    }
                    continue;
                }
            }
            
            collective_mainloop.mma_fp8<Delay_V_release>(
                mainloop_params, pipeline_k, pipeline_vt, smem_pipe_read, smem_pipe_release,
                tOrO, softmax, n_block_min, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block,
                shared_storage, seqlen_traits_q, seqlen_traits_k);

            collective_epilogue.store(
                epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);

            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

} // namespace flash
""",
    'key-files/hopper/flash_fwd_launch_template.h': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"
#include "utils.h"
#include "combine.h"

template<typename Kernel_traits, bool Is_causal, bool Is_local, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using OutputType = typename Kernel_traits::OutputType;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    constexpr static bool Is_split = Kernel_traits::Is_split;
    static_assert(Seqlen_traits_Q::UseGQAPacking == (Kernel_traits::kBlockH > 1), "If kBlockH > 1, use gqa packed layouts");
    static_assert(!(Is_split && Seqlen_traits::UseVarSeqLen), "Split KV not yet supported for variable seqlen.");

    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits_Q>;
    using Scheduler = std::conditional_t<
        Seqlen_traits::UseVarSeqLen, 
        flash::SingleTileScheduler,
        std::conditional_t<!Is_causal && !Is_local && !Is_split,
            flash::StaticPersistentTileScheduler<Is_split>,
            flash::DynamicPersistentTileScheduler<
                Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup,
                Kernel_traits::NumProducerThreads,
                Is_split
            >
    >>;
    // using Scheduler = flash::SingleTileScheduler;
    Seqlen_traits_Q seqlen_traits_q(
        params.total_q, params.seqlen_q, params.cu_seqlens_q, params.seqused_q);
    Seqlen_traits seqlen_traits_k(
        params.total_k, params.seqlen_k, params.cu_seqlens_k, params.seqused_k);

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),            
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, 
                params.q_row_stride, params.q_head_stride, params.q_batch_stride
            ),  // layout_Q
            static_cast<Element const*>(params.k_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b_k, 
                params.k_row_stride, params.k_head_stride, params.k_batch_stride
            ),  // layout_K
            static_cast<Element const*>(params.v_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b_k, 
                params.v_row_stride, params.v_head_stride, params.v_batch_stride
            ),  // layout_V
            params.scale_softmax_log2,
            params.descale_q_ptr,
            params.descale_k_ptr,
            params.descale_v_ptr,
            params.window_size_left,
            params.window_size_right,
            ceil_div(params.h_h_k_ratio, Kernel_traits::kBlockH),
            params.cache_batch_idx,
            Is_split ? params.num_splits : 1
        });
    typename CollectiveEpilogue::Params epilogue_params = [&] {
        if constexpr(!Is_split) {
            return CollectiveEpilogue::to_underlying_arguments({            
                static_cast<OutputType*>(params.o_ptr),
                seqlen_traits_q.get_gmem_layout(
                    params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, 
                    params.o_row_stride, params.o_head_stride, params.o_batch_stride
                ),  // layout_O
                static_cast<float*>(params.softmax_lse_ptr),            
                seqlen_traits_q.get_lse_gmem_layout(
                    params.seqlen_q, params.h, params.b
                )  // layout_LSE
            });
        } else {
            return CollectiveEpilogue::to_underlying_arguments({
                static_cast<OutputType*>(params.oaccum_ptr), 
                seqlen_traits_q.get_oaccum_gmem_layout(
                    params.seqlen_q, params.d, params.h_k, params.b, params.h_h_k_ratio, params.num_splits,
                    params.oaccum_row_stride, params.oaccum_head_stride, params.oaccum_batch_stride,  
                    params.oaccum_split_stride
                ), // layout_O
                static_cast<float*>(params.softmax_lseaccum_ptr),            
                seqlen_traits_q.get_lseaccum_gmem_layout(
                    params.seqlen_q, params.h, params.b, params.num_splits
                ), // layout_LSE
            });
        }
    }();

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM/Kernel_traits::kBlockH);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});    
    int num_blocks_h = params.h_k * ceil_div(params.h_h_k_ratio, Kernel_traits::kBlockH);
    typename Scheduler::Arguments scheduler_args =
        {num_blocks_m, Is_split ? params.num_splits : 1, num_blocks_h, params.b, params.tile_count_semaphore};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);    

    // Get the ptr to kernel function.
    void *kernel;
    if constexpr(cutlass::sizeof_bits_v<Element> == 8)
        kernel = (void *)flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Is_local, Scheduler, Seqlen_traits, Seqlen_traits_Q>;
    else
        kernel = (void *)flash::compute_attn_ws<Kernel_traits, Is_causal, Is_local, Scheduler, Seqlen_traits, Seqlen_traits_Q>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // int smem_size_o = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_o));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, o = %d.\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_o);
    if (smem_size >= 48 * 1024) {
       CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    if constexpr(size(ClusterShape{}) > 1) {
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(
            launch_params, kernel, mainloop_params, epilogue_params, 
            scheduler_params, seqlen_traits_q, seqlen_traits_k);
    } else {
        if constexpr(cutlass::sizeof_bits_v<Element> == 8) {
            flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Is_local, Scheduler, Seqlen_traits, Seqlen_traits_Q>
                <<<grid_dims, block_dims, smem_size, stream>>>
                (mainloop_params, epilogue_params, scheduler_params, seqlen_traits_q, seqlen_traits_k);
        } else {
            flash::compute_attn_ws<Kernel_traits, Is_causal, Is_local, Scheduler, Seqlen_traits, Seqlen_traits_Q>
                <<<grid_dims, block_dims, smem_size, stream>>>
                (mainloop_params, epilogue_params, scheduler_params, seqlen_traits_q, seqlen_traits_k);
        }

    }
    CHECK_CUDA_KERNEL_LAUNCH();

    if constexpr (Is_split) {
      using FinalOutputType = typename Kernel_traits::FinalOutputType;
      static_assert(is_same_v<OutputType, float>, "Assume OutputType of main kernel is float.");
      static_assert(is_same_v<ElementAccum, float>, "ElementAccum must be float.");
      // We want kBlockM to be as small as possible for more parallelism.
      // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
      // If headdim is divisible by 64, then we set kBlockM = 8, etc.
      constexpr static int kHeadDim = Kernel_traits::kHeadDim;
      constexpr static int kBlockM = kHeadDim % 128 == 0 ? 4 : (kHeadDim % 64 == 0 ? 8 : 16);
      constexpr static bool Is_even_K = true; // always true for our current setting
      void *kernel_combine;
      int smem_size_combine;
      NUM_SPLITS_SWITCH(params.num_splits, kLogMaxSplits, [&] {
        constexpr static int kMaxSplits = 1 << kLogMaxSplits;
        kernel_combine = (void *) flash::combine_attn_seqk_parallel<
          FinalOutputType, ElementAccum, kHeadDim, kBlockM, kLogMaxSplits, Is_even_K, Flash_fwd_params>;
        smem_size_combine = sizeof(
          flash::SharedStorageLSE<float, Shape<Int<kMaxSplits>, Int<kBlockM+1>>, Shape<Int<kMaxSplits>>>);
      });
      if (smem_size_combine >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel_combine, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_combine));
      }
      dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
      dim3 block_dims_combine(128);
      dim3 cluster_dims_combine(1, 1, 1);
      cutlass::ClusterLaunchParams launch_params_combine{
          grid_combine, block_dims_combine, cluster_dims_combine, smem_size_combine, stream};
      cutlass::launch_kernel_on_cluster(launch_params_combine, kernel_combine, params);
      CHECK_CUDA_KERNEL_LAUNCH();
    }
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static bool UseCluster = false;

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        MMA_3WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
          SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
              // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 192) % 2 == 0 && !Is_causal && !Is_local && !Is_split
              //             && kNumMmaWGs == 3 && !Seqlen_traits::UseVarSeqLen, UseCluster, [&] {
                run_flash_fwd<
                  Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, 128, 4 + kNumMmaWGs * 4,
                      2, false, UseCluster ? 2 : 1, T, !Seqlen_traits::UseVarSeqLen && Is_split>,
                  Is_causal,
                  Is_local && !Is_causal,
                  Seqlen_traits
                >(params, stream);
              // });
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    
    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.is_local, Is_local, [&] {
          SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
              // Only use Cluster if number of tiles along seqlen_q is even
              // and not Is_causal, Is_split, or varseqlen
              BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Is_local && !Is_split
                          && kNumMmaWGs == 2 && !Seqlen_traits::UseVarSeqLen, UseCluster, [&] {
                run_flash_fwd<
                  Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, (Is_causal || Is_local) ? 128 : 176,
                      4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1, 
                      T, !Seqlen_traits::UseVarSeqLen && Is_split>, 
                  Is_causal,
                  Is_local && !Is_causal,
                  Seqlen_traits
                >(params, stream);
              });
            });
          });
        });
      });
    });
}



template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;

    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.is_local, Is_local, [&] {
          SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
            BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
              // Only use Cluster if number of tiles along seqlen_q is even
              // and not Is_causal, Is_split, or varseqlen
              BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Is_local && !Is_split
                          && kNumMmaWGs == 2 && !Seqlen_traits::UseVarSeqLen, UseCluster, [&] {
                run_flash_fwd<
                  Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, kNumMmaWGs == 1 ? 96 : 80,
                      4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1,
                      T, !Seqlen_traits::UseVarSeqLen && Is_split>, 
                  Is_causal,
                  Is_local && !Is_causal,
                  Seqlen_traits
                >(params, stream);
              });
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim64_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockN = 128;
    constexpr static int kStages = 4;
    // constexpr static bool UseCluster = false;
    // constexpr static int kBlockM = 192;
    // constexpr static int kNWarps = 4 + kBlockM/16;
    using Seqlen_traits = flash::FixedSeqLenTraits;

    MMA_3WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.is_local, Is_local, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 192) % 2 == 0 && !Is_causal && !Is_local && !Is_split
                        && kNumMmaWGs == 3, UseCluster, [&] {
              run_flash_fwd<
                Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                  kStages, false, UseCluster ? 2 : 1, T, Is_split>,
                Is_causal,
                Is_local && !Is_causal,
                Seqlen_traits
              >(params, stream);
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockN = 256;
    constexpr static int kStages = 2;
    // constexpr static int kBlockM = 128;
    // constexpr static int kNWarps = 4 + kBlockM/16;
    using Seqlen_traits = flash::FixedSeqLenTraits;

    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.is_local, Is_local, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Is_local && !Is_split
                        && kNumMmaWGs == 2, UseCluster, [&] {
              run_flash_fwd<
                Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                  kStages, false, UseCluster ? 2 : 1, T, Is_split>,
                Is_causal,
                Is_local && !Is_causal,
                Seqlen_traits
              >(params, stream);
            });
          });
        });
      });
    });
}

template<typename T>
void run_mha_fwd_hdim256_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256; 
    constexpr static int kBlockN = 128;
    constexpr static int kStages = 2;
    // constexpr static int kBlockM = 128;
    // constexpr static int kNWarps = 4 + kBlockM/16;
    using Seqlen_traits = flash::FixedSeqLenTraits;

    MMA_2WG_SWITCH(params.seqlen_q, kNumMmaWGs, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.is_local, Is_local, [&] {
          BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
            BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128) % 2 == 0 && !Is_causal && !Is_local && !Is_split
                        && kNumMmaWGs == 2, UseCluster, [&] {
              run_flash_fwd<
                Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                  kStages, false, UseCluster ? 2 : 1, T, Is_split>,
                Is_causal,
                Is_local && !Is_causal,
                Seqlen_traits
              >(params, stream);
            });
          });
        });
      });
    });
}

/*
** GQA methods
*/

template<typename T, int kBlockH>
void run_mha_fwd_hdim64_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 64;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_3WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 192/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 3, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, 128, 4 + kNumMmaWGs * 4,
                  2, false, UseCluster ? 2 : 1, T, !Seqlen_traits::UseVarSeqLen && Is_split, kBlockH>,
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}

template<typename T, int kBlockH>
void run_mha_fwd_hdim128_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 128;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 2, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, (Is_causal || Is_local) ? 128 : 176,
                  4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1, T, Is_split, kBlockH>, 
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}

template<typename T, int kBlockH>
void run_mha_fwd_hdim256_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 256;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 2, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits<Headdim, kNumMmaWGs * 64, kNumMmaWGs == 1 ? 96 : 80,
                  4 + kNumMmaWGs * 4, 2, false, UseCluster ? 2 : 1, T, Is_split, kBlockH>, 
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}

template<typename T, int kBlockH>
void run_mha_fwd_hdim64_fp8_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 64;
  constexpr static int kBlockN = 128;
  constexpr static int kStages = 4;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_3WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 192/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 3, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                kStages, false, UseCluster ? 2 : 1, T, Is_split, kBlockH>,
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}

template<typename T, int kBlockH>
void run_mha_fwd_hdim128_fp8_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 128;
  constexpr static int kBlockN = 256;
  constexpr static int kStages = 2;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 2, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                kStages, false, UseCluster ? 2 : 1, T, Is_split, kBlockH>,
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}

template<typename T, int kBlockH>
void run_mha_fwd_hdim256_fp8_gqa(Flash_fwd_params &params, cudaStream_t stream) {
  constexpr static int Headdim = 256;
  constexpr static int kBlockN = 128;
  constexpr static int kStages = 2;
  constexpr static bool UseCluster = false;
  using Seqlen_traits = flash::FixedSeqLenTraits;
  using Seqlen_traits_Q = flash::FixedGQASeqLenTraits;

  MMA_2WG_SWITCH(kBlockH * params.seqlen_q, kNumMmaWGs, [&] {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      BOOL_SWITCH(params.is_local, Is_local, [&] {
        BOOL_SWITCH(params.num_splits > 1, Is_split, [&] {
          // BOOL_SWITCH(cutlass::ceil_div(params.seqlen_q, 128/kBlockH) % 2 == 0 && !Is_causal && !Is_local && !Is_split
          //             && kNumMmaWGs == 2, UseCluster, [&] {
            run_flash_fwd<
              Flash_fwd_kernel_traits_fp8<Headdim, kNumMmaWGs * 64, kBlockN, 4 + kNumMmaWGs * 4,
                kStages, false, UseCluster ? 2 : 1, T, Is_split, kBlockH>,
              Is_causal,
              Is_local && !Is_causal,
              Seqlen_traits,
              Seqlen_traits_Q
            >(params, stream);
          // });
        });
      });
    });
  });
}
""",
    'key-files/hopper/kernel_traits.h': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

// Use if Oaccum is too large for SharedStorageQKVO
template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOaccum {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;    
    union {    
        struct {    
            cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

// SharedStorage struct with no smem for O
template <int kStages, class Gemm1Type, class Gemm2Type, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV>
struct SharedStorageQKV {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

// Use if Oaccum is too large for SharedStorageQKVOVt
template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVtaccum {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        struct {
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV>
struct SharedStorageQKVVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::half_t, bool Is_split_=false, int kBlockH_ = 1>
struct Flash_fwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using FinalOutputType = elem_type;
    using OutputType = std::conditional_t<Is_split_, float, FinalOutputType>;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;
    static_assert(!(Is_WS && Is_Q_in_regs), "Warp-specialization does not support Q in registers");

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockH = kBlockH_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM % kBlockH == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;

    static constexpr bool Is_split = Is_split_;
    static constexpr bool No_smem_O = Is_split;

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Is_Q_in_regs,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{})),
                                   GMMA::Major::K, GMMA::Major::MN>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // for gmem -> smem Q copy 
    using FactoringLayoutQ = Layout<Shape<Int<kBlockM/kBlockH>, Int<kBlockH>, Int<kHeadDim>>,
        Stride<Int<kBlockH>, _1, Int<kBlockM>>>;
    using TileShapeQCopy = std::conditional_t<(kBlockH > 1),
        decltype(shape(FactoringLayoutQ{})), decltype(select<0, 2>(TileShape_MNK{}))>;
    using SmemLayoutQCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutQ{}, FactoringLayoutQ{})), SmemLayoutQ>;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), Int<kStages>{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutVt =
        decltype(composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                        Step<_2, _1, _3>{})));
    
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));
    // for smem -> gmem O copy
    using TileShapeOCopy = TileShapeQCopy;
    using SmemLayoutOCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutO{}, FactoringLayoutQ{})), SmemLayoutO>;

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = std::conditional_t<!No_smem_O,
        SharedStorageQKVO<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>,
        SharedStorageQKV<kStages, Element, Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;

};

// Traits struct for fp8 kernel with in-kernel transpose
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::float_e4m3_t, bool Is_split_ = false, int kBlockH_ = 1>
struct Flash_fwd_kernel_traits_fp8 {
    using Element = elem_type;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using ElementAccum = float;
    using FinalOutputType = cutlass::bfloat16_t;
    using OutputType = std::conditional_t<Is_split_, float, FinalOutputType>;
    using index_t = int64_t;

    static constexpr bool Is_split = Is_split_;
    static constexpr bool No_smem_O = false;
    // NOTE: not using smem for epilogue degrades perf substantially.
    // static constexpr bool No_smem_O = Is_split;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;    
    static_assert(!Is_Q_in_regs, "Warp-specialization does not support Q in registers");    

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockH = kBlockH_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM % kBlockH == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;
    static_assert(kStages > 1);

    // Use this to save enough smem when writing out in float precision.
    static constexpr bool VO_union_all = Is_split && (kBlockM != 64) && (kHeadDim == 256);

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;    
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{}))>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // for gmem -> smem Q copy
    using FactoringLayoutQ = Layout<Shape<Int<kBlockM/kBlockH>, Int<kBlockH>, Int<kHeadDim>>,
        Stride<Int<kBlockH>, _1, Int<kBlockM>>>;
    using TileShapeQCopy = std::conditional_t<(kBlockH > 1),
        decltype(shape(FactoringLayoutQ{})), decltype(select<0, 2>(TileShape_MNK{}))>;
    using SmemLayoutQCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutQ{}, FactoringLayoutQ{})), SmemLayoutQ>;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using TransposeShapeAtomV = Shape<_64, _64>;    
    using SmemLayoutAtomV = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    
    // for fp8 in-kernel transpose -- src layout
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
    using FactoringShapeV = decltype(make_shape(SmemShapeLDSM{},
        shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{}), shape<3>(SmemLayoutDivideV{})));
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    // for fp8 in-kernel transpose -- dst layout
    using SmemLayoutVtTrans =
        decltype(composition(SmemLayoutVt{},
                             make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1, _3>{})));
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
#ifndef NO_FP8_COLUMN_PERMUTE
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
    using FactoringShapeVt = decltype(make_shape(SmemShapeSTSM{},
        shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{}), shape<3>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));
    // for smem -> gmem O copy
    using TileShapeOCopy = TileShapeQCopy;
    using SmemLayoutOCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutO{}, FactoringLayoutQ{})), SmemLayoutO>;

    // used for rmem -> smem O copy in fp8 kernel to undo column permutation
    using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                 Stride<_4, _32, _1, _0>>;
    using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim/16>>,
                                Stride<_0, _2, Stride<_4, _1>, _8>>;
    using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, OutputType>{},
                      ThreadLayoutrO{}, ValueLayoutrO{}));

    using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDim/16>>;
    using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = std::conditional_t<!No_smem_O,
        std::conditional_t<!VO_union_all,
            SharedStorageQKVOVt<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>,
            SharedStorageQKVOVtaccum<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>>,
        SharedStorageQKVVt<kStages, Element, Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKV<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        cute::array_aligned<float, cute::cosize_v<SmemLayoutdQacc>> smem_dqacc;
        cute::array_aligned<float, 128> smem_lse;
        cute::array_aligned<float, 128> smem_dpsum;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS, class SmemLayoutdQacc,
          class SmemLayoutdK, class SmemLayoutdV>
struct SharedStorageQKVdOdKVWS<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
        cute::array_aligned<float, cute::cosize_v<SmemLayoutdQacc>> smem_dqacc;
        cute::array_aligned<float, 128> smem_lse;
        cute::array_aligned<float, 128> smem_dpsum;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_K;
        cutlass::arch::ClusterTransactionBarrier barrier_V;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_do;
    };
};

template <bool Has_P_smem, int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar;

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar<true, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQ> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
            };
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterTransactionBarrier barrier_dO;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    };
};

template <int kStages, class Element, class OutputType, class SmemLayoutQ, class SmemLayoutdO,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutP, class SmemLayoutdS,
          class SmemLayoutdQ>
struct SharedStorageQKVdOdKVSeqqPar<false, kStages, Element, OutputType, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQ> {
    struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
            };
        };
        union {  // Put smem_p in a union just so we can still refer to it in the struct, even if it's not used.
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdS>> smem_ds;
        };
    };
    struct {
        cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterTransactionBarrier barrier_dO;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
         int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         int kClusterN_ = 1, typename elem_type=cutlass::half_t>
struct Flash_bwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kNThreadsNonWS = 8 * cutlass::NumThreadsPerWarp;
    // static constexpr int kNThreadsdQ = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kNThreadsdQ = 2 * cutlass::NumThreadsPerWarpGroup;

    static_assert(kNWarps_ == 8 || kNWarps_ == 12);

    static constexpr bool Is_WS = kNWarps_ >= 12;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<_1, Int<kClusterN>, _1>;

    static constexpr int kStages = 2;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;
    static_assert(!(SdP_swapAB && dKV_swapAB));  // If SdP_swapAB, then we don't swap for dKV

    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == 2 && AtomLayoutMdQ == 2 && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kBlockM>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<2 / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockN>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<2 / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !SdP_swapAB,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::MN, GMMA::Major::MN>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockM>, Int<kBlockN>>
        // Shape<Int<kBlockM>, Int<kHeadDim >, Int<kBlockN>>,
        // Shape<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<2 / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
        // Layout<Shape<Int<1>, Int<1>, _1>>,
        // Layout<Shape<Int<1>, Int<1>, _1>>
    >;
    static constexpr GMMA::Major MmadQMajorA = !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
    static constexpr GMMA::Major MmadQMajorB = !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !dQ_swapAB,
            std::conditional_t<
                Mma_dQ_is_RS,
                decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
                decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>())
            >,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::MN, GMMA::Major::K>())
        >{},
        AtomLayoutdQ{}));

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydKV = cute::SM90_TMA_STORE;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
#else
    static constexpr bool Has_cp_async = false;
#endif
    // For the dot_do_o preprocessing kernel
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreadsNonWS % kGmemThreadsPerRow == 0, "kNThreadsNonWS must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreadsNonWS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreadsdQ / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreadsdQ / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreadsdQ / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQ{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO = SmemLayoutQ;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutAtomdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdS = decltype(tile_to_shape(SmemLayoutAtomdS{}, select<0, 1>(TileShape_MNK{})));

    // using SmemLayoutAtomdQacc = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementAccum,
    //     decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // using SmemLayoutdQacc = decltype(tile_to_shape(SmemLayoutAtomdQacc{}, select<0, 2>(TileShape_MNK{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPt =
        decltype(cute::composition(SmemLayoutP{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdSt =
        decltype(cute::composition(SmemLayoutdS{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));

    // using SmemLayoutdQacct =
    //     decltype(cute::composition(SmemLayoutdQacc{},
    //                                make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
    //                                            make_stride(Int<kBlockM>{}, _1{}))));

    using SmemLayoutdK = SmemLayoutK;
    using SmemLayoutdV = SmemLayoutV;
    using SmemLayoutdKt = SmemLayoutKt;
    using SmemLayoutdVt = SmemLayoutKt;

    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using SmemLayoutAtomdQ = decltype(
        // composition(Swizzle<kSwizzle, 3, 3>{},
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<kNThreadsdQ / 32>, Int<32>>,
                           Stride<Int<32>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    using SmemLayoutAtomdQaccTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementAccum,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdQaccTMA = decltype(tile_to_shape(SmemLayoutAtomdQaccTMA{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutdQacc = SmemLayoutdQ;
    using SmemLayoutdQacct = SmemLayoutdQt;
    using SmemLayoutdQacc2 = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _2{})));
    // using SmemLayoutdQacc = decltype(tile_to_shape(SmemLayoutAtomdQacc{}, select<0, 2>(TileShape_MNK{})));
    // using SmemLayoutdQacct =
    //     decltype(cute::composition(SmemLayoutdQacc{},
    //                                make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
    //                                            make_stride(Int<kBlockM>{}, _1{}))));
    using RmemTiledCopydQacc = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    // using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<!SdP_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    using SharedStorage = std::conditional_t<
        !Is_WS,
        SharedStorageQKVdOdKV<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
                              SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdK, SmemLayoutdV>,
        SharedStorageQKVdOdKVWS<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
                              SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQacc, SmemLayoutdK, SmemLayoutdV>
                              // SmemLayoutK, SmemLayoutV, SmemLayoutdS, SmemLayoutdQacc2, SmemLayoutdK, SmemLayoutdV>
    >;

    // using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages * 2>;
    // using PipelineState = typename cutlass::PipelineState<kStages * 2>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
         int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         int kClusterN_ = 1, typename elem_type=cutlass::half_t>
struct Flash_bwd_seqqpar_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

    static_assert(kNWarps_ == 8);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterN = kClusterN_;
    using ClusterShape_MNK = Shape<_1, Int<kClusterN>, _1>;

    static constexpr int kStages = 2;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;
    static_assert(!(SdP_swapAB && dKV_swapAB));  // If SdP_swapAB, then we don't swap for dKV

    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == 2 && AtomLayoutMdQ == 2 && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN / (2 / AtomLayoutMSdP)>, Int<kBlockM>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<2 / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutNdKV)>, Int<kBlockN>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<2 / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !SdP_swapAB,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::MN, GMMA::Major::MN>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim / (2 / AtomLayoutMdQ)>, Int<kBlockM>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<2 / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<2 / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    static constexpr GMMA::Major MmadQMajorA = !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
    static constexpr GMMA::Major MmadQMajorB = !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !dQ_swapAB,
            std::conditional_t<
                Mma_dQ_is_RS,
                decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
                decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>())
            >,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::MN, GMMA::Major::K>())
        >{},
        AtomLayoutdQ{}));

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydKV = cute::SM90_TMA_STORE;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
#else
    static constexpr bool Has_cp_async = false;
#endif
    // For the dot_do_o preprocessing kernel
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutdO = SmemLayoutQ;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutAtomdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutdS = decltype(tile_to_shape(SmemLayoutAtomdS{}, select<0, 1>(TileShape_MNK{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim>{}))));
    using SmemLayoutPt =
        decltype(cute::composition(SmemLayoutP{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    using SmemLayoutdSt =
        decltype(cute::composition(SmemLayoutdS{},
                                   make_layout(make_shape(get<1>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));

    using SmemLayoutdK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdV = SmemLayoutdK;
    using SmemLayoutdKt = SmemLayoutKt;
    using SmemLayoutdVt = SmemLayoutKt;
    using SmemLayoutdQTMA = decltype(tile_to_shape(SmemLayoutAtomK{}, select<0, 2>(TileShape_MNK{})));

    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutdQt =
        decltype(cute::composition(SmemLayoutdQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                               make_stride(Int<kBlockM>{}, _1{}))));
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);

    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    static constexpr int kSmemdKVSize = size(SmemLayoutdKV{}) * sizeof(Element) * 2;

    // using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<!SdP_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;
    using SmemCopyAtomdQ = Copy_Atom<
        std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
        Element>;

    using SharedStorage = SharedStorageQKVdOdKVSeqqPar<!SdP_swapAB, kStages, Element, Element, SmemLayoutQ, SmemLayoutdO,
        SmemLayoutK, SmemLayoutV, SmemLayoutP, SmemLayoutdS, SmemLayoutdQTMA>;

    // using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages * 2>;
    // using PipelineState = typename cutlass::PipelineState<kStages * 2>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

// 4 warps
struct SmemTransposeFp8_64x64 {

  using Element = cutlass::float_e4m3_t;
  
  using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
  using ldsm_value_shape = Shape<_2, _8, _2, _1>;  
  using ldsm_value_stride = Stride<_2, _4, _1, _0>;
  using TiledCopyLDSM = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
      Layout<ldsm_value_shape, ldsm_value_stride>{}));
  TiledCopyLDSM tiled_copy_ldsm;  

  using stsm_thread_shape = Shape<_4, _1, _8, _4>;
  // using stsm_thread_stride = Stride<_1, _0, _4, _32>;
#ifndef NO_FP8_COLUMN_PERMUTE
  using stsm_value_shape = Shape<_4, _4, _1, _2>;
  using stsm_value_stride = Stride<_1, _8, _0, _4>;
#else
  using stsm_value_shape = Shape<_4, _4, _2, _1>;
  using stsm_value_stride = Stride<_1, _8, _4, _0>;
#endif

  using TiledCopySTSM =
      decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                               Layout<stsm_thread_shape>{},
                               Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void operator()(SmemTensor &&s_in, SmemTensorOut &&s_out) {
    using namespace cute;

    auto tid = threadIdx.x;
    auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
    auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

    auto tXsX = thr_copy_ldsm.partition_S(s_in);
    auto tXrX = make_tensor<Element>(shape(tXsX));    
    auto tXsX_out = thr_copy_stsm.partition_D(s_out);

    cute::copy(tiled_copy_ldsm, tXsX, tXrX);

    auto data = tXrX.data();
    // size(tXrX) == 32
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size(tXrX); n += 8) {
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
      auto upper = data_32bit[0];
      auto lower = data_32bit[1];
      data_32bit[0] = __byte_perm(upper, lower, 0x6420);
      data_32bit[1] = __byte_perm(upper, lower, 0x7531);
    }

    cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
  }
};

template <typename Ktraits, bool Is_causal, bool Is_local, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;
    // static constexpr int kBlockM = Ktraits::kBlockM;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kBlockH = Ktraits::kBlockH;
    static constexpr bool Is_split = Ktraits::Is_split;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
    
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutQCopy = typename Ktraits::SmemLayoutQCopy;
    using TileShapeQCopy = typename Ktraits::TileShapeQCopy;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopyQ{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            repeat_like(typename Seqlen_traits_Q::StrideT{}, int32_t(0)), 
            typename Seqlen_traits_Q::StrideT{}
        ),
        SmemLayoutQCopy{},
        TileShapeQCopy{},
        _1{}));  // no mcast for Q

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)), 
            typename Seqlen_traits::StrideT{}
        ),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    // TMA_V may differ from TMA_K for fp8 kernel (e.g. swizzling mode)
    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)),
            repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
            typename Seqlen_traits::StrideT{}
        ),
        take<0, 2>(SmemLayoutV{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using MainloopPipelineNoTMA = typename Ktraits::MainloopPipelineNoTMA;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

    // static constexpr bool UseSchedulerBarrier = kHeadDim <= 128;
    static constexpr bool UseSchedulerBarrier = Ktraits::kNWarps >= 12 && 
        (cutlass::sizeof_bits_v<Element> == 8 ? kHeadDim >= 128 : kHeadDim <= 128);

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        typename Seqlen_traits_Q::LayoutT layout_Q;
        Element const* ptr_K;
        typename Seqlen_traits::LayoutT layout_K;
        Element const* ptr_V;
        typename Seqlen_traits::LayoutT layout_V;
        float const softmax_scale_log2;        
        float const* descale_q_ptr;
        float const* descale_k_ptr;
        float const* descale_v_ptr;
        int window_size_left;
        int window_size_right;
        int const qhead_per_khead;
        int const* cache_batch_idx;
        int const num_splits;
    };

    // Device side kernel params
    struct Params {
        typename Seqlen_traits_Q::LayoutT layout_Q;
        typename Seqlen_traits::LayoutT layout_K;
        typename Seqlen_traits::LayoutT layout_V;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;        
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        float const softmax_scale_log2;        
        float const* descale_q_ptr;
        float const* descale_k_ptr;
        float const* descale_v_ptr;
        int window_size_left;
        int window_size_right;
        int const* cache_batch_idx;
        cutlass::FastDivmod num_splits_divmod;
    };


    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQCopy{},
            TileShapeQCopy{},
            _1{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K);
        TMA_K tma_load_K = make_tma_copy(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        return {args.layout_Q, args.layout_K, args.layout_V,
                cutlass::FastDivmod(args.qhead_per_khead),
                tma_load_Q, tma_load_K, tma_load_V,
                args.softmax_scale_log2,
                args.descale_q_ptr, args.descale_k_ptr, args.descale_v_ptr,
                args.window_size_left, args.window_size_right,
                args.cache_batch_idx,
                cutlass::FastDivmod(args.num_splits)};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    void get_n_block_min_max(
          Params const& mainloop_params,
          int m_block, 
          int n_split_idx,
          const Seqlen_traits_Q& seqlen_traits_q,
          const Seqlen_traits& seqlen_traits_k,
          int& n_block_min,
          int& n_block_max
        ) {
        // static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockM_div_H = get<0>(TileShape_MNK{})/Ktraits::kBlockH;
        int const seqlen_q = seqlen_traits_q.actual_seq_len;
        int const seqlen_k = seqlen_traits_k.actual_seq_len;
        n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        
        if constexpr(Is_split) {
            int const n_blocks_per_split
                = mainloop_params.num_splits_divmod.divide(n_block_max + int(mainloop_params.num_splits_divmod) - 1);
            n_block_min = n_split_idx * n_blocks_per_split;
            n_block_max = std::min(n_block_max, (n_split_idx + 1) * n_blocks_per_split);
        }

        if constexpr (Is_causal) {
            n_block_max = std::min(
                n_block_max,
                cute::ceil_div((m_block + 1) * kBlockM_div_H + seqlen_k - seqlen_q, kBlockN));
        } else if constexpr (Is_local) {
            n_block_max = std::min(
                n_block_max,
                cute::ceil_div((m_block + 1) * kBlockM_div_H + seqlen_k - seqlen_q + mainloop_params.window_size_right, kBlockN));
            n_block_min = std::max(
                n_block_min,
                (m_block * kBlockM_div_H + seqlen_k - seqlen_q - mainloop_params.window_size_left) / kBlockN);
        }
    }

    CUTLASS_DEVICE
    void get_n_block_max(
          Params const& mainloop_params,
          int m_block, 
          const Seqlen_traits_Q& seqlen_traits_q,
          const Seqlen_traits& seqlen_traits_k,
          int& n_block_max
        ) {
        // static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockM_div_H = get<0>(TileShape_MNK{})/Ktraits::kBlockH;
        int const seqlen_q = seqlen_traits_q.actual_seq_len;
        int const seqlen_k = seqlen_traits_k.actual_seq_len;
        n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                cute::ceil_div((m_block + 1) * kBlockM_div_H + seqlen_k - seqlen_q, kBlockN));
        }
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int work_idx,
         const Seqlen_traits_Q& seqlen_traits_q,
         const Seqlen_traits& seqlen_traits_k,
         int n_block_min,
         int n_block_max
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQCopy{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
        Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

        auto [m_block, n_split_idx, bidh, bidb] = block_coord;
        const int bidb_cache = mainloop_params.cache_batch_idx == nullptr ? bidb : mainloop_params.cache_batch_idx[bidb];
        const int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = [&] {
            // Need this inside lambda to capture structured binding
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;
            if constexpr(Seqlen_traits_Q::UseGQAPacking) {
                return seqlen_traits_q.get_local_tile_tensor(
                    mQ, TileShapeQCopy{}, bidh_kv, bidb)
                        (_, _, _, m_block, bidh % int(mainloop_params.qhead_per_khead_divmod));  // (M/H, H, K)
            } else {
                return seqlen_traits_q.get_local_tile_tensor(
                    mQ, TileShapeQCopy{}, bidh, bidb)(_, _, m_block);  // (M, K)
            }
        }();
        Tensor gK = seqlen_traits_k.get_local_tile_tensor(
            mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb_cache);  // (N, K, _)
        Tensor gV = seqlen_traits_k.get_local_tile_tensor(
            mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb_cache);  // (N, K, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int n_block = n_block_max - 1;

        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
        }

        // Wait for the MMA warpgroups to say that smem_q is ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

        if (lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        }

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        if constexpr (!No_smem_O) { shared_storage.barrier_O.wait((work_idx + 1) % 2); }
        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block > n_block_min; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgK(_, n_block - 1), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }

        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        if (lane_predicate) {
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        }
        scheduler.broadcast_next_work(work_tile_info);
        
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load_fp8(Params const& mainloop_params,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         MainloopPipelineNoTMA pipeline_vt,         
         PipelineState& smem_pipe_write,
         PipelineState& smem_pipe_read,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int work_idx,
         const Seqlen_traits_Q& seqlen_traits_q,
         const Seqlen_traits& seqlen_traits_k,
         int n_block_min,
         int n_block_max
         ) {
        
        using SmemLayoutTransposeV = typename Ktraits::SmemLayoutTransposeV;
        using SmemLayoutTransposeVt = typename Ktraits::SmemLayoutTransposeVt;

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQCopy{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
        
        Tensor sV_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutTransposeV{}));
        Tensor sVt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v_out.data()), SmemLayoutTransposeVt{}));

        auto smem_transpose_V = SmemTransposeFp8_64x64();
        auto do_transpose_V = [&](int stage) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < shape<2>(SmemLayoutTransposeV{}); ++j) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < shape<1>(SmemLayoutTransposeV{}); ++i) {
                smem_transpose_V(flatten(sV_divide(_, i, j, stage)),
                                flatten(sVt_divide(_, i, j, stage)));
                }
            }
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
        };

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
        Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

        auto [m_block, split_idx, bidh, bidb] = block_coord;
        const int bidb_cache = mainloop_params.cache_batch_idx == nullptr ? bidb : mainloop_params.cache_batch_idx[bidb];
        const int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = [&] {
            // Need this inside lambda to capture structured binding
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;
            if constexpr(Seqlen_traits_Q::UseGQAPacking) {
                return seqlen_traits_q.get_local_tile_tensor(
                    mQ, TileShapeQCopy{}, bidh_kv, bidb)
                        (_, _, _, m_block, bidh % int(mainloop_params.qhead_per_khead_divmod));  // (M/H, H, K)
            } else {
                return seqlen_traits_q.get_local_tile_tensor(
                    mQ, TileShapeQCopy{}, bidh, bidb)(_, _, m_block);  // (M, K)
            }
        }();
        Tensor gK = seqlen_traits_k.get_local_tile_tensor(
            mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb_cache);  // (N, K, _)
        Tensor gV = seqlen_traits_k.get_local_tile_tensor(
            mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb_cache);  // (N, K, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int n_block = n_block_max - 1;

        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            pipeline_k.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
        }

        // Wait for the MMA warpgroups to say that smem_q is ready
        // for fp8, change from NumThreadsPerWarp to NumThreadsPerWarpGroup
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
            if constexpr(!Ktraits::VO_union_all) {
                pipeline_v.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
            }

        }
        // With fp8 kernel, smem_o is in union with smem_v_out,
        // except for split kernel + hdim 256,
        // so could use NamedBarrier instead of ClusterBarrier.
        // But, this doesn't appear to have any benefit.
        if constexpr (!No_smem_O) { shared_storage.barrier_O.wait((work_idx + 1) % 2); }

        if constexpr(Ktraits::VO_union_all) {
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_v.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
            }
        }
            
        #pragma unroll 2
        for (; n_block > n_block_min; --n_block) {
            pipeline_v.consumer_wait(smem_pipe_read);
            pipeline_vt.producer_acquire(smem_pipe_write);
            do_transpose_V(smem_pipe_read.index());
            pipeline_vt.producer_commit(smem_pipe_write);
            pipeline_v.consumer_release(smem_pipe_read);

            ++smem_pipe_write;
            ++smem_pipe_read;
            
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tKgK(_, n_block-1), tKsK(_, smem_pipe_write.index()));
                pipeline_v.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tVgV(_, n_block-1), tVsV(_, smem_pipe_write.index()));
            }                                                                
        }       

        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        scheduler.broadcast_next_work(work_tile_info);
        
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        do_transpose_V(smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);

        ++smem_pipe_write;
        ++smem_pipe_read; 
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
              PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail_one_write(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
              PipelineState& smem_pipe_write) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline_k.producer_tail(smem_pipe_write);
          pipeline_v.producer_tail(smem_pipe_write);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (!UseSchedulerBarrier) {
            return;
        } else {
            static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
            if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (3 - cutlass::canonical_warp_group_idx()) /*id*/);
            } else {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 2 ? cutlass::canonical_warp_group_idx() + 1 : cutlass::canonical_warp_group_idx() + 1 - 3)  /*id*/);
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 1 ? cutlass::canonical_warp_group_idx() + 2 : cutlass::canonical_warp_group_idx() + 2 - 3)  /*id*/);
            }
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producer (warp 0) that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + Ktraits::NumProducerThreads, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);                
        if constexpr (!UseSchedulerBarrier) {
            return;
        } else {
            static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
            if (cutlass::canonical_warp_group_idx() > 1) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 1 /*id*/);
            }
            if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
                if (cutlass::canonical_warp_group_idx() > 2) {
                    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 2 /*id*/);
                }
            }
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_min,
        int n_block_max,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage,
        const Seqlen_traits_Q& seqlen_traits_q,
        const Seqlen_traits& seqlen_traits_k
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockH = Ktraits::kBlockH;
        static constexpr int kBlockM_div_H = get<0>(TileShape_MNK{}) / kBlockH;

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

        typename Ktraits::TiledMma0 tiled_mma0;
        typename Ktraits::TiledMma1 tiled_mma1;
        auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors" for first matmul.
        Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
        Tensor tSrK = threadMma0.partition_fragment_B(sK);
        // Allocate "fragments/descriptors" for second matmul.
        // Note: S becomes P.
        Tensor tOrV = threadMma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
        int const seqlen_q = seqlen_traits_q.actual_seq_len;
        int const seqlen_k = seqlen_traits_k.actual_seq_len;
        int n_block = n_block_max - 1;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_Q.wait(work_idx % 2); }

        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        warp_scheduler_barrier_arrive();
        if constexpr (!No_smem_O) {
            if (work_idx != 0) {
                int lane_predicate = cute::elect_one_sync();
                if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
                    tma_store_wait<0>();
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.barrier_O.arrive(cta_id, lane_predicate);
                    }
                }
            }
        }
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        auto col_limit_right = [&](int row, int n_block) {
            int col_limit_base = row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM_div_H;
            if constexpr(Is_local)
                return col_limit_base + mainloop_params.window_size_right;
            else
                return col_limit_base;
        };
        auto col_limit_left = [&](int row, int n_block) {
            return std::max(
                0,
                row + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM_div_H - mainloop_params.window_size_left
            );
        };
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal && !Is_local) {  // Just masking based on col
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else {  // mask based on both row and col
                    // using std::min is faster than doing col >= limit0 or col >= limit1
                    // Need to cast get<1>(tScS(i)) to (signed) int since by default it's unsigned, and the
                    // right hand side can be negative and might be converted to a very large unsigned integer.
                    int row = int(get<0>(tScS(i))) / kBlockH;
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN, col_limit_right(row, n_block))) {
                        tSrS(i) = -INFINITY;
                    } else if constexpr(Is_local) {
                        if (int(get<1>(tScS(i))) < col_limit_left(row, n_block)) {
                            tSrS(i) = -INFINITY;
                        }
                    }
                } 
            }
        }

        softmax.template online_softmax</*Is_first=*/true>(tSrS);
 
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
        Tensor scores_scale = make_fragment_like(softmax.row_max);
        clear(scores_scale);

        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM_div_H, kBlockN) + 1;
        // Only go through these if Is_causal, since n_masking_steps = 1 when !Is_causal
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block > n_block_min; ++masking_step, --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            if (masking_step > 0) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                int row = int(get<0>(tScS(i))) / kBlockH;
                if (int(get<1>(tScS(i))) >= col_limit_right(row, n_block - 1)) {
                    tSrS(i) = -INFINITY;
                }
            }
            cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS), scores_scale);
            softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }

        #pragma unroll 1
        for (; n_block > n_block_min; --n_block) {
            Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            softmax.rescale_o(tOrO, scores_scale);
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K

            if constexpr(Is_local) {
                Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                Tensor tScS = threadMma0.partition_C(cS);
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    int row = int(get<0>(tScS(i))) / kBlockH;
                    if (
                        int(get<1>(tScS(i))) >= col_limit_right(row, n_block - 1) ||
                        int(get<1>(tScS(i))) < col_limit_left(row, n_block - 1)
                    ) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
            // auto scores_scale = softmax.template max</*Is_first=*/false>(tSrS);
            cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS), scores_scale);
            softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS);

            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            // softmax.rescale_o(tOrO, scores_scale);
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }
        // Tell warp 0 that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        softmax.rescale_o(tOrO, scores_scale);
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        cute::copy(softmax.template finalize</*Is_dropout=*/false, Is_split>(tSrS), scores_scale);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
        ++smem_pipe_read_v;
        softmax.rescale_o(tOrO, scores_scale);
        return;
    }

    template <bool Delay_V_release = false, typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE void
    mma_fp8(Params const& mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipelineNoTMA pipeline_vt,
        PipelineState& smem_pipe_read,
        PipelineState& smem_pipe_release,        
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_min,
        int n_block_max,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage,
        const Seqlen_traits_Q& seqlen_traits_q,
        const Seqlen_traits& seqlen_traits_k
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        // static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kBlockH = Ktraits::kBlockH;
        static constexpr int kBlockM_div_H = get<0>(TileShape_MNK{}) / kBlockH;

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v_out.data()), SmemLayoutVt{});

        typename Ktraits::TiledMma0 tiled_mma0;
        typename Ktraits::TiledMma1 tiled_mma1;
        auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors" for first matmul.
        Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
        Tensor tSrK = threadMma0.partition_fragment_B(sK);
        // Allocate "fragments/descriptors" for second matmul.
        Tensor tOrV = threadMma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
        int const seqlen_q = seqlen_traits_q.actual_seq_len;
        int const seqlen_k = seqlen_traits_k.actual_seq_len;
        int n_block = n_block_max - 1;
        
        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_Q.wait(work_idx % 2); }
        
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));        
        
        consumer_wait(pipeline_k, smem_pipe_read);                        
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        if constexpr (!No_smem_O) {
            if (work_idx != 0) {        
                int lane_predicate = cute::elect_one_sync();
                if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
                    tma_store_wait<0>();
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.barrier_O.arrive(cta_id, lane_predicate);
                    }
                }        
            }
        }
        warpgroup_wait<0>();
        warp_scheduler_barrier_arrive();
        pipeline_k.consumer_release(smem_pipe_read);

        auto col_limit_right = [&](int row, int n_block) {
            int col_limit_base = row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM_div_H;
            if constexpr(Is_local)
                return col_limit_base + mainloop_params.window_size_right;
            else
                return col_limit_base;
        };
        auto col_limit_left = [&](int row, int n_block) {
            return std::max(
                0,
                row + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM_div_H - mainloop_params.window_size_left
            );
        };       
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal && !Is_local) {  // Just masking based on col                
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else {  // mask based on both row and col
                    int row = int(get<0>(tScS(i))) / kBlockH;
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN, col_limit_right(row, n_block))) {
                        tSrS(i) = -INFINITY;
                    } else if constexpr(Is_local) {
                        if (int(get<1>(tScS(i))) < col_limit_left(row, n_block)) {
                            tSrS(i) = -INFINITY;
                        }
                    }
                }
            }
        }

        softmax.template online_softmax</*Is_first=*/true>(tSrS);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);
        
        Tensor scores_scale = make_fragment_like(softmax.row_max);
        clear(scores_scale);
        
        consumer_wait(pipeline_vt, smem_pipe_read);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);                
        if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }

        ++smem_pipe_read;
        --n_block;
        constexpr int extra_iterations = !Is_causal ? kStages - 1 : cute::ceil_div(kBlockM_div_H, kBlockN);        

        if constexpr(Is_causal) {
            CUTLASS_PRAGMA_UNROLL
            for (int iter = 0; iter < extra_iterations && n_block >= n_block_min; ++iter, --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

                Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                Tensor tScS = threadMma0.partition_C(cS);
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    int row = int(get<0>(tScS(i))) / kBlockH;
                    if (int(get<1>(tScS(i))) >= col_limit_right(row, n_block)) {
                        tSrS(i) = -INFINITY;
                    }
                }

                warp_scheduler_barrier_arrive();
                pipeline_k.consumer_release(smem_pipe_read);
                if constexpr(Delay_V_release) {
                    pipeline_vt.consumer_release(smem_pipe_release);
                    ++smem_pipe_release;
                }
                consumer_wait(pipeline_vt, smem_pipe_read);
                
                cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);
                
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);            
                if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }
                ++smem_pipe_read;
            }
        } else if constexpr(!Is_local) { 
            CUTLASS_PRAGMA_UNROLL      
            for (int iter = 0; iter < extra_iterations && n_block >= n_block_min; ++iter, --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                if constexpr(Delay_V_release) {
                    pipeline_vt.consumer_release(smem_pipe_release);
                    ++smem_pipe_release;
                }
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warp_scheduler_barrier_arrive();
                if constexpr(!Delay_V_release) { pipeline_k.consumer_release(smem_pipe_read); }
                else { consumer_wait(pipeline_vt, smem_pipe_read); }
                
                cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                if constexpr (Delay_V_release) { pipeline_k.consumer_release(smem_pipe_read); }
                else { consumer_wait(pipeline_vt, smem_pipe_read); }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }                
                ++smem_pipe_read;
            }
        }

        if constexpr(Delay_V_release) {
            warp_scheduler_barrier_sync();
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= n_block_min; --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);                
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

                if constexpr(Is_local) {
                    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                    Tensor tScS = threadMma0.partition_C(cS);
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        int row = int(get<0>(tScS(i))) / kBlockH;
                        if (
                            int(get<1>(tScS(i))) >= col_limit_right(row, n_block) ||
                            int(get<1>(tScS(i))) < col_limit_left(row, n_block)
                        ) {
                            tSrS(i) = -INFINITY;
                        }
                    }
                }

                warp_scheduler_barrier_arrive();                
                pipeline_k.consumer_release(smem_pipe_read);
                pipeline_vt.consumer_release(smem_pipe_release);

                cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);
                
                consumer_wait(pipeline_vt, smem_pipe_read);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                warp_scheduler_barrier_sync();
                ++smem_pipe_read;
                ++smem_pipe_release;
            }
            warp_scheduler_barrier_arrive();
            pipeline_vt.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
        } else {
            if constexpr (kHeadDim == 128) { warp_scheduler_barrier_sync(); }
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= n_block_min; --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                if constexpr (kHeadDim == 256) { warp_scheduler_barrier_sync(); }
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

                if constexpr(Is_local) {
                    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                    Tensor tScS = threadMma0.partition_C(cS);
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        int row = int(get<0>(tScS(i))) / kBlockH;
                        if (
                            int(get<1>(tScS(i))) >= col_limit_right(row, n_block) ||
                            int(get<1>(tScS(i))) < col_limit_left(row, n_block)
                        ) {
                            tSrS(i) = -INFINITY;
                        }
                    }
                }

                warp_scheduler_barrier_arrive();
                pipeline_k.consumer_release(smem_pipe_read);

                cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/Is_local>(tSrS);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                consumer_wait(pipeline_vt, smem_pipe_read);
                if constexpr (kHeadDim == 128) { warp_scheduler_barrier_sync(); }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                pipeline_vt.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }
            if constexpr (kHeadDim == 128) { warp_scheduler_barrier_arrive(); }
        }
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        cute::copy(softmax.template finalize</*Is_dropout=*/false, Is_split>(tSrS, shared_storage.descale_v), scores_scale);
        softmax.rescale_o(tOrO, scores_scale);
        return;
    }

};

} // namespace flash
""",
    'key-files/hopper/seq_len.h': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <array>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

namespace flash {

static constexpr int kMaxTileSize = 128;

template <bool UseVarSeqLen_, bool UseGQAPacking_> class SeqLenTraits {
public:
  static_assert(!(UseVarSeqLen_ && UseGQAPacking_),
    "Variable sequence length with GQA parallelization not implemented yet.");

  // Total number of queries / keys. Unpadded.
  int sum_s = 0;
  // seq len offsets.
  int *cu_seq_len = nullptr;
  // actual seq len array.
  int *seq_used = nullptr;
  // seq len of the current batch.
  int actual_seq_len = -1;

  // Whether this is for fixed-seq-len or var-seq-len.
  static constexpr bool UseVarSeqLen = UseVarSeqLen_;
  static constexpr bool UseGQAPacking = UseGQAPacking_;

  using ShapeT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t, int32_t>,
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>,
        cute::Shape<int32_t, int32_t, int32_t, int32_t>
      >
  >;
  using StrideT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1, int64_t>, 
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int64_t, int64_t, _1, int64_t, int64_t>,
        cute::Shape<int64_t, _1, int64_t, int64_t>
      >
  >;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t>, 
      cute::Shape<int32_t, int32_t, int32_t>
  >;
  using StrideLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1>, 
      cute::Shape<int64_t, int64_t, _1>
  >;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  // Not used for varseqlen
  using ShapeOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>
  >;
  using StrideOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int64_t, int64_t, _1, int64_t, int64_t, int64_t>,
    cute::Shape<int64_t, _1, int64_t, int64_t, int64_t>
  >;
  using LayoutOAccumT = cute::Layout<ShapeOAccumT, StrideOAccumT>;

  using ShapeLseAccumT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLseAccumT = cute::Shape<int64_t, int64_t, int64_t, _1>;
  using LayoutLseAccumT = cute::Layout<ShapeLseAccumT, StrideLseAccumT>;

  CUTLASS_HOST SeqLenTraits() {}

  CUTLASS_HOST SeqLenTraits(
      int sum_s, int max_seq_len, int *cu_seq_len = nullptr, int *seq_used = nullptr): 
      sum_s(sum_s), cu_seq_len(cu_seq_len), seq_used(seq_used), actual_seq_len(max_seq_len) {}

  CUTLASS_DEVICE void init(int bidb) {
    // TODO: add leftpad, seqlen_new for kv cache support
    if (seq_used) {
      actual_seq_len = seq_used[bidb];
    }
  }

  CUTLASS_DEVICE void init_no_guard(int bidb) {
    actual_seq_len = seq_used[bidb];
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h, int b, 
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    // static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio,
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));    
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h, int b, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of lse tensor in BHM format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(
      int m, int h, int b, bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    return make_layout(make_shape(b, h, m),
                       make_stride(int64_t(h * m), int64_t(m), cute::_1()));
  }

  // Returns the layout of lse tensor in TBHM format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_lseaccum_gmem_layout(
      int m, int h, int b, int num_splits, bool padded = false) const {
    return make_layout(make_shape(num_splits, b, h, m),
                       make_stride(int64_t(b * h * m), int64_t(h * m), int64_t(m), cute::_1()));
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, bool padded = false) const {
    auto g_tensor = local_tile(
      m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_lse_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, int n_split_idx, bool padded = false) const {
    // m_tensor has shape (B, H, M) or (splits, B, H, M)
    // Expect tile shape (bM)
    // Returns g_tensor of shape = (bM, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(m_tensor(bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(m_tensor(n_split_idx, bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    }
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_o_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape,
      int bidh, int bidb, int split_idx, bool padded = false) const {
    // static_assert(!UseVarSeqLen, "Don't use get_o_local_tile_tensor with VarSeqLen.");
    // m_tensor has shape (M, K, H, B) or (M, K, H, B, splits)
    // Expect tile shape (bM, K)
    // Returns g_tensor of shape = (bM, K, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb, split_idx), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    }
  }
  
};

using FixedSeqLenTraits = SeqLenTraits<false, false>;
using VarSeqLenTraits = SeqLenTraits<true, false>;
using FixedGQASeqLenTraits = SeqLenTraits<false, true>;

template <>
CUTLASS_DEVICE void VarSeqLenTraits::init(int bidb) {
  actual_seq_len = 
      seq_used ? seq_used[bidb] : (cu_seq_len[bidb + 1] - cu_seq_len[bidb]);
}

template <>
CUTLASS_DEVICE void FixedGQASeqLenTraits::init(int bidb) {
  // no op
}

// Returns the static layout of a var-seq-len tensor in global memory based on
// max_seq_len and max_batch_size.
// padded: only useful for var-seq-len for dq_accum and softmax_d.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_gmem_layout(
    int m, int k, int h, int b, 
    int64_t m_stride, int64_t h_stride, int64_t b_stride,
    bool padded) const {
  return make_layout(
    make_shape(sum_s + (padded ? kMaxTileSize * b : 0), k, h), 
    make_stride(m_stride, cute::_1{}, h_stride));
}

template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio,
    int64_t m_stride, int64_t h_stride, int64_t b_stride,
    bool padded) const {
  return make_layout(
    make_shape(sum_s + (padded ? kMaxTileSize * b : 0), k, h_k * h_h_k_ratio), 
    make_stride(m_stride, cute::_1{}, h_stride));
}

// padded: only useful for var-seq-len for dq_accum and softmax_d.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_lse_gmem_layout(
    int m, int h, int b, bool padded) const {
  return make_layout(
    make_shape(h, sum_s + (padded ? kMaxTileSize * b : 0)), 
    make_stride(int64_t(sum_s + (padded ? kMaxTileSize * b : 0)), cute::_1()));
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, bool padded) const {
  auto g_offset = local_tile(
      m_tensor(_, _, bidh), 
      cute::make_shape(1, get<1>(tile_shape)), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0), _0{}));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(
        cute::make_shape(actual_seq_len, get<1>(tile_shape)), 
        g_offset.stride()
      ));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

// TODO: restructure to not duplicate code
template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_o_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, int n_split_idx, bool padded) const {
  static_assert(!Is_split, "Don't currently support split kv kernel with VarSeqLenTraits");
  auto g_offset = local_tile(
      m_tensor(_, _, bidh), 
      cute::make_shape(1, get<1>(tile_shape)), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0), _0{}));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(
        cute::make_shape(actual_seq_len, get<1>(tile_shape)), 
        g_offset.stride()
      ));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_lse_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, int n_split_idx, bool padded) const {
  static_assert(!Is_split, "Don't currently support split kv kernel with VarSeqLenTraits");
  auto g_offset = local_tile(
      m_tensor(bidh, _), cute::make_shape(_1{}), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0)));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(cute::make_shape(actual_seq_len), cute::make_shape(_1{})));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

// Returns layout of QO tensor in (M,H/HK,K,HK,B) format in global memory.
template <>
CUTLASS_HOST_DEVICE auto FixedGQASeqLenTraits::get_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio,
    int64_t m_stride, int64_t h_stride, int64_t b_stride, bool padded) const {
  return make_layout(make_shape(m, h_h_k_ratio, k, h_k, b),
                     make_stride(m_stride, h_stride, cute::_1{},
                                 h_stride * h_h_k_ratio, b_stride));
}

// Returns layout of Oaccum tensor in (M,H/HK,K,HK,B,T) format in global memory.
template <>
CUTLASS_HOST_DEVICE auto FixedGQASeqLenTraits::get_oaccum_gmem_layout(
    int m, int k, int h_k, int b, int h_h_k_ratio, int num_splits,
    int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
    bool padded) const {
  return make_layout(make_shape(m, h_h_k_ratio, k, h_k, b, num_splits),
                     make_stride(m_stride, h_stride, cute::_1{},
                                 h_stride * h_h_k_ratio, b_stride,
                                 split_stride));
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto FixedGQASeqLenTraits::get_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape, 
    int bidh_kv, int bidb, bool padded) const {
  // m_tensor has shape (M, H/H_K, K, H_K, B)
  // Expect tile_shape (bM/bH, bH, K)
  // Returns g_tensor of shape (bM/bH, bH, K, ceil_div(M,bM/bH), ceil_div(H/H_K,bH))
  auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb), tile_shape, make_coord(_, _, _0{}));
  return g_tensor;
}

template <>
template <bool Is_split, typename MTensor, typename Shape>
CUTLASS_DEVICE auto FixedGQASeqLenTraits::get_o_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh_kv, int bidb, int split_idx, bool padded) const {
  // m_tensor has shape (M, H/H_K, K, H_K, B) or (M, H/H_K, K, H_K, B, splits)
  // Expect tile_shape (bM/bH, bH, K)
  // Returns g_tensor of shape (bM/bH, bH, K, ceil_div(M,bM/bH), ceil_div(H/H_K,bH))
  if constexpr(!Is_split) {
    auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb), tile_shape, make_coord(_, _, _0{}));
    return g_tensor;
  } else {
    auto g_tensor = local_tile(
      m_tensor(_, _, _, bidh_kv, bidb, split_idx), tile_shape, make_coord(_, _, _0{}));
    return g_tensor;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
""",
    'key-files/hopper/static_switch.h': r"""// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define PREC_SWITCH(PRECTYPE, NAME, ...)                                       \
  [&] {                                                                        \
    if (PRECTYPE == 3) {                                                       \
      using NAME = cutlass::float_e4m3_t;                                      \
      return __VA_ARGS__();                                                    \
    } else if (PRECTYPE == 2) {                                                \
      using NAME = cutlass::bfloat16_t;                                        \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME = cutlass::half_t;                                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...)                               \
  [&] {                                                                        \
    if (HEADDIM == 64) {                                                       \
      constexpr static int CONST_NAME = 64;                                    \
      return __VA_ARGS__();                                                    \
    } else if (HEADDIM == 128) {                                               \
      constexpr static int CONST_NAME = 128;                                   \
      return __VA_ARGS__();                                                    \
    } else  {                                                                  \
      constexpr static int CONST_NAME = 256;                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, NAME, ...)                              \
  [&] {                                                                        \
    bool useSeqLen = USE_VAR_SEQ_LEN;                                          \
    if (useSeqLen) {                                                           \
      using NAME = flash::VarSeqLenTraits;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME = flash::FixedSeqLenTraits;                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define SEQLEN_SWITCH_FWD(VAR_SEQ_LEN_Q, SEQ_USED_K, NAME_Q, NAME_K, ...)      \
  [&] {                                                                        \
    bool useVarSeqLenQ = VAR_SEQ_LEN_Q;                                        \
    bool useSeqUsedK = SEQ_USED_K;                                             \
    if (useVarSeqLenQ) {                                                       \
      using NAME_Q = flash::VarSeqLenTraits;                                   \
      using NAME_K = flash::VarSeqLenTraits;                                   \
      return __VA_ARGS__();                                                    \
    } else if (useSeqUsedK) {                                                  \
      using NAME_Q = flash::FixedSeqLenTraits;                                 \
      using NAME_K = flash::FixedSeqLenTraitsDynamic;                          \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      using NAME_Q = flash::FixedSeqLenTraits;                                 \
      using NAME_K = flash::FixedSeqLenTraits;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define QUERYHEAD_SWITCH(QUERYHEADS, CONST_NAME, ...)                          \
  [&] {                                                                        \
    if (QUERYHEADS <= 2) {                                                     \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 4) {                                              \
      constexpr static int CONST_NAME = 4;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 8) {                                              \
      constexpr static int CONST_NAME = 8;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QUERYHEADS <= 16) {                                             \
      constexpr static int CONST_NAME = 16;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 32;                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define MMA_3WG_SWITCH(QLEN, CONST_NAME, ...)                                  \
  [&] {                                                                        \
    if (QLEN <= 64) {                                                          \
      constexpr static int CONST_NAME = 1;                                     \
      return __VA_ARGS__();                                                    \
    } else if (QLEN <= 128) {                                                  \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 3;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define MMA_2WG_SWITCH(QLEN, CONST_NAME, ...)                                  \
  [&] {                                                                        \
    if (QLEN <= 64) {                                                          \
      constexpr static int CONST_NAME = 1;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int CONST_NAME = 2;                                     \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define NUM_SPLITS_SWITCH(NUM_SPLITS, LOG_MAX_SPLITS, ...)                     \
  [&] {                                                                        \
    if (NUM_SPLITS <= 2) {                                                     \
      constexpr static int LOG_MAX_SPLITS = 1;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 4) {                                              \
      constexpr static int LOG_MAX_SPLITS = 2;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 8) {                                              \
      constexpr static int LOG_MAX_SPLITS = 3;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 16) {                                             \
      constexpr static int LOG_MAX_SPLITS = 4;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 32) {                                             \
      constexpr static int LOG_MAX_SPLITS = 5;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 64) {                                             \
      constexpr static int LOG_MAX_SPLITS = 6;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static int LOG_MAX_SPLITS = 7;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
""",
    'key-files/hopper/tile_scheduler.hpp': r"""/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

struct SingleTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_splits, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
    };

    // Device side kernel params
    struct Params {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
    }

    struct WorkTileInfo {
        int M_idx = 0;
        int H_idx = 0;
        int B_idx = 0;
        bool is_valid_tile = false;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return is_valid_tile;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {M_idx, 1, H_idx, B_idx};
        }

    };

    CUTLASS_DEVICE
    SingleTileScheduler(int* tile_count_smem_) { }

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {-1, -1, -1, false};
    }

};

///////////////////////////////////////////////////////////////////////////////

template <bool Is_split = false>
class StaticPersistentTileScheduler {

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_splits, num_head, num_batch;
        int* const tile_count_semaphore = nullptr;
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, split_divmod, head_divmod;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        // return {args.num_blocks_m * args.num_head * args.num_batch,
        //         cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head)};
        return {args.num_blocks_m * args.num_splits * args.num_head * args.num_batch,                
                cutlass::FastDivmod(args.num_blocks_m),
                cutlass::FastDivmod(args.num_splits),
                cutlass::FastDivmod(args.num_head)};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, split_idx, bidh, bidb;
            if constexpr(!Is_split) {
                bidb = params.head_divmod.divmod(bidh,
                         params.m_block_divmod.divmod(m_block, tile_idx));
                return {m_block, 1, bidh, bidb};
            } else {
                bidb = params.head_divmod.divmod(bidh,
                         params.split_divmod.divmod(split_idx,
                           params.m_block_divmod.divmod(m_block, tile_idx)));
                return {m_block, split_idx, bidh, bidb};
            }
        }

    };

    CUTLASS_DEVICE
    StaticPersistentTileScheduler(int* tile_count_smem_) {};

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {}

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }

};

template<int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
    int NumProducerThreads = cutlass::NumThreadsPerWarp,
    bool Is_split = false>
class DynamicPersistentTileScheduler {

protected:
    int* const tile_count_smem;

public:

    // Host side kernel arguments
    struct Arguments {
        int const num_blocks_m, num_splits, num_head, num_batch;
        int* const tile_count_semaphore;
    };

    // Device side kernel params
    struct Params {
        int const total_blocks;        
        cutlass::FastDivmod const m_block_divmod, split_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        // return {args.num_blocks_m * args.num_head * args.num_batch,
        //         cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head),
        //         args.tile_count_semaphore};
        return {args.num_blocks_m * args.num_splits * args.num_head * args.num_batch,                
                cutlass::FastDivmod(args.num_blocks_m),
                cutlass::FastDivmod(args.num_splits),
                cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_dim(Arguments const& args, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int m_block, split_idx, bidh, bidb;
            if constexpr(!Is_split) {
                bidb = params.head_divmod.divmod(bidh,
                         params.m_block_divmod.divmod(m_block, tile_idx));
                return {m_block, 1, bidh, bidb};
            } else {
                bidb = params.head_divmod.divmod(bidh,
                         params.split_divmod.divmod(split_idx,
                           params.m_block_divmod.divmod(m_block, tile_idx)));
                return {m_block, split_idx, bidh, bidb};
            }
        }

    };

    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(int* tile_count_smem_) : tile_count_smem(tile_count_smem_) {};

    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work() const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    CUTLASS_DEVICE
    void
    broadcast_next_work(WorkTileInfo& current_work) const {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
        if (threadIdx.x % NumProducerThreads == 0) {
            *tile_count_smem = current_work.tile_idx;
        }
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
    }

    template<bool IsProducer=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducer && NumProducerThreads == cutlass::NumThreadsPerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of the producer threads (warp 0)
            return {__shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/)};
        } else if constexpr (IsProducer && NumProducerThreads == cutlass::NumThreadsPerWarpGroup) {
            // TODO: investigate optimal synchronize
            int tile_idx = *tile_count_smem;
            return {tile_idx};
        } else {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
            int tile_idx = *tile_count_smem;
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<int>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
            return {tile_idx};
        }
    }

};

} // flash
""",
    'key-files/hopper/utils.h': r"""/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM90, convert acc_layout from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_transposed_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
        auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
        return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout), make_layout(get<2, 1>(l), get<2>(acc_layout)));
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    }
};

// Convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs_fp8(Layout acc_layout) {
    using X = Underscore;    
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _4>{});  // (2, 2, (2, N / 32)))    
    return make_layout(make_layout(Shape<_4, _2, _2>{}),
                       get<1>(acc_layout),
                       make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Byte permute for fp8 kernel
template <typename Fragment>
CUTLASS_DEVICE void permute_regs_A_to_C(Fragment &accum) {  

  auto data = accum.data();    

  #pragma unroll  
  for (int n = 0; n < size(accum); n += 8) {
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
      auto upper = data_32bit[0];
      auto lower = data_32bit[1];
      data_32bit[0] = __byte_perm(upper, lower, 0x5410);
      data_32bit[1] = __byte_perm(upper, lower, 0x7632);        
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Tensor out = make_tensor_like<To_type>(tensor);
    // cute::copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
    // return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2,
          typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    } else {
        // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_split, int NumCopyThreads, typename ElemO, typename TMACopyO, typename LayoutO, 
          typename TileShapeO, typename SMemO, typename SeqLenTraits>
__forceinline__ __device__ void write_tma(
        ElemO* O, const TMACopyO& tma_store_O,
        const LayoutO& layout_O, const TileShapeO& tile_shape_O,
        const SMemO& sO, int m_block, int bidh, int bidb, int n_split_idx,
        const SeqLenTraits& seqlen_traits_o, int write_warp_idx) {
    Tensor mO = tma_store_O.get_tma_tensor(layout_O.shape());
    Tensor gO = seqlen_traits_o.get_o_local_tile_tensor<Is_split>(
        mO, tile_shape_O, bidh, bidb, n_split_idx
    )(_, _, m_block);  // (M, K)
    auto block_tma_O = tma_store_O.get_slice(_0{});
    Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
    Tensor tOsO = block_tma_O.partition_S(sO);  // (TMA, TMA_M, TMA_K)

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == write_warp_idx && lane_predicate) {
        cute::copy(tma_store_O, tOsO, tOgO);
        tma_store_arrive();
    }
    // Note: no wait here.
    // tma_store_wait<0>();
}

// Epilogue that copies RMEM -> GMEM directly for GQA enabled.
// Reports as uncoalesced stores by the profiler
template <bool Use_gqa_layout, bool Column_permute_fp8, bool Is_split = true, typename TensorO, typename OutputType,
          typename LayoutO, typename TileShapeO, typename TiledMma, typename SeqLenTraits>
__forceinline__ __device__ void write_rmem_to_gmem(
        TensorO &tOrO, OutputType *O, const LayoutO& layout_O, TileShapeO tile_shape_O,
        int m_block, int h_block, int bidh, int bidh_kv, int bidb, int n_split_idx,
        TiledMma& tiled_mma, const SeqLenTraits& seqlen_traits_o, int thread_idx) {
    static_assert(is_same_v<typename TensorO::value_type, float>, "rmem dtype must be float");
    Tensor mO = make_tensor(make_gmem_ptr(O), layout_O);
    Tensor gO = [&] {
        if constexpr(Use_gqa_layout) {
            return seqlen_traits_o.get_o_local_tile_tensor<Is_split>(
                mO, tile_shape_O, bidh_kv, bidb, n_split_idx
                )(_, _, _, m_block, h_block);  // (bM/bH, bH, K)
        } else {
            return seqlen_traits_o.get_o_local_tile_tensor<Is_split>(
                mO, tile_shape_O, bidh, bidb, n_split_idx
                )(_, _, m_block);  // (bM, bK)
        }
    }();
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto tile_shape_mnk = cute::tile_shape(tiled_mma);
    Tensor cO = cute::make_identity_tensor(select<0, 1>(tile_shape_mnk));
    Tensor tOcO = thread_mma.partition_C(cO);
    // tOcO has shape ((2, 2, V), MMA_M, MMA_N), we only take only the row indices.
    Tensor tOcO_row = tOcO(make_coord(_0{}, _, _0{}), _, _0{});
    // reshape from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
    const int m_bound = seqlen_traits_o.actual_seq_len - m_block * size<0>(gO);
    // hardcoded col_idx to circumvent reg spilling with counting tensor
    const int col_start_idx = !Column_permute_fp8 ? 2 * (thread_idx % 4) : 4 * (thread_idx % 4);

    if constexpr (Use_gqa_layout) {
        static constexpr int kBlockH = size<1>(gO);
        const int h_bound = shape<1>(layout_O) - h_block * kBlockH;
        #pragma unroll
        for(int nrow = 0; nrow < size<0>(tOrO_rowcol); ++nrow) {
            const int row = int(get<0>(tOcO_row(nrow)));
            const int h_local = row % kBlockH;
            const int m_local = row / kBlockH;
            if(h_local < h_bound && m_local < m_bound) {
                if constexpr(!Column_permute_fp8) {
                    Tensor tOrO_nrow_float2 = recast<float2>(tOrO_rowcol(nrow, _));
                    #pragma unroll
                    for (int ncol = 0; ncol < size<1>(tOrO_rowcol)/2; ++ncol) {
                        *reinterpret_cast<float2*>(&(gO(m_local, h_local, col_start_idx + 8 * ncol))) = 
                            tOrO_nrow_float2(ncol);
                    }
                } else {
                    Tensor tOrO_nrow = tOrO_rowcol(nrow, _);
                    #pragma unroll
                    for (int ncol = 0; ncol < size<1>(tOrO_rowcol); ncol += 4) {
                        gO(m_local, h_local, col_start_idx + 4 * ncol) = tOrO_nrow(ncol);
                        gO(m_local, h_local, col_start_idx + 4 * ncol + 2) = tOrO_nrow(ncol + 1);
                        gO(m_local, h_local, col_start_idx + 4 * ncol + 1) = tOrO_nrow(ncol + 2);
                        gO(m_local, h_local, col_start_idx + 4 * ncol + 3) = tOrO_nrow(ncol + 3);
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for(int nrow = 0; nrow < size<0>(tOrO_rowcol); ++nrow) {
            const int row = int(get<0>(tOcO_row(nrow)));
            if(row < m_bound) {
                if constexpr(!Column_permute_fp8) {
                    Tensor tOrO_nrow_float2 = recast<float2>(tOrO_rowcol(nrow, _));
                    #pragma unroll
                    for (int ncol = 0; ncol < size<1>(tOrO_rowcol)/2; ++ncol) {
                        *reinterpret_cast<float2*>(&(gO(row, col_start_idx + 8 * ncol))) = 
                            tOrO_nrow_float2(ncol);
                    }
                } else {
                    Tensor tOrO_nrow = tOrO_rowcol(nrow, _);
                    #pragma unroll
                    for (int ncol = 0; ncol < size<1>(tOrO_rowcol); ncol += 4) {
                        gO(row, col_start_idx + 4 * ncol) = tOrO_nrow(ncol);
                        gO(row, col_start_idx + 4 * ncol + 2) = tOrO_nrow(ncol + 1);
                        gO(row, col_start_idx + 4 * ncol + 1) = tOrO_nrow(ncol + 2);
                        gO(row, col_start_idx + 4 * ncol + 3) = tOrO_nrow(ncol + 3);
                    }
                }
            }
        }
    }
}

template <int NumCopyThreads, typename ElemO, typename TiledCopyO, typename LayoutO, 
          typename TileShapeO, typename SMemO, typename SeqLenTraits>
__forceinline__ __device__ void write_tiled(
        ElemO* O, const TiledCopyO& tiled_copy_O,
        const LayoutO& layout_O, const TileShapeO& tile_shape_O,
        const SMemO& sO, int m_block, int bidh, int bidb,
        const SeqLenTraits& seqlen_traits_o) {
    Tensor mO = make_tensor(make_gmem_ptr(O), layout_O);
    Tensor gO = seqlen_traits_o.get_local_tile_tensor(
        mO, tile_shape_O, bidh, bidb
    )(_, _, m_block);  // (M, K)

    ThrCopy thr_copy_O = tiled_copy_O.get_slice(threadIdx.x - NumCopyThreads);
    Tensor tOgO = thr_copy_O.partition_D(gO); // (CPY,CPY_M,CPY_K,k)
    Tensor tOsO = thr_copy_O.partition_S(sO); // (CPY,CPY_M,CPY_K)

    // Prepare for TiledCopy.
    // Grouping is needed because cute::copy_if() does group_modes<1, R> for src and dst.
    // After grouping, the first dim is number of elements to read together.
    Tensor tOsOFlatten = cute::flatten(tOsO);
    Tensor tOsOGroup = cute::group_modes<1, rank(tOsOFlatten)>(tOsOFlatten);
    Tensor tOgOFlatten = cute::flatten(tOgO);
    Tensor tOgOGroup = cute::group_modes<1, rank(tOgOFlatten)>(tOgOFlatten);

    // Get thread coords to global index mapping.
    Tensor gOCounting = cute::make_identity_tensor(gO.shape());
    Tensor tSgOCounting = thr_copy_O.partition_D(gOCounting);
    Tensor tSgOCountingFlatten = cute::flatten(tSgOCounting);
    Tensor tSgOCountingGrouped =
        cute::group_modes<1, rank(tSgOCountingFlatten)>(tSgOCountingFlatten);

    // Write out to GMEM.
    const int kNumMsPerTile = get<0>(tile_shape_O);
    int cta_m = std::min(
        seqlen_traits_o.actual_seq_len - m_block * kNumMsPerTile, kNumMsPerTile
    );
    if (cta_m == kNumMsPerTile) {
        copy(tiled_copy_O, tOsOGroup, tOgOGroup);
    } else {
        auto predicate_fn = [&](auto coords) {
            auto s_coords = tSgOCountingGrouped(_0{}, coords);
            return elem_less(get<0>(s_coords), cta_m);
        };
        copy_if(tiled_copy_O, predicate_fn, tOsOGroup, tOgOGroup);
    }
}

template <bool IsTMACopy, bool IsRegToGmem, bool Is_split, int NumCopyThreads, typename ElemO, 
          typename TMACopyO, typename TiledCopyO, typename LayoutO, 
          typename TileShapeO, typename SMemO, typename SeqLenTraits, class TensorO, typename TiledMma>
__forceinline__ __device__ void write_O(
        ElemO* O, const TMACopyO& tma_copy_O, const TiledCopyO& tiled_copy_O,
        const LayoutO& layout_O, const TileShapeO& tile_shape_O,
        const SMemO& sO, int m_block, int bidh, int bidb, int n_split_idx,
        const SeqLenTraits& seqlen_traits_o, int write_warp_idx, TiledMma & tiledMma1, TensorO & tOrO) {

    if constexpr (IsRegToGmem) {
        static_assert(Is_split, "use write_rmem_to_gmem with split kv kernel only");
	    write_rmem_to_gmem(tOrO, O, layout_O, tile_shape_O, m_block, bidh, bidb, n_split_idx,
		     tiledMma1, seqlen_traits_o, threadIdx.x - NumCopyThreads);
    } else if constexpr (IsTMACopy) {
        write_tma<Is_split, NumCopyThreads>(O, tma_copy_O, layout_O, tile_shape_O, sO, m_block, bidh, bidb,
            n_split_idx, seqlen_traits_o, write_warp_idx);
    } else {
        static_assert(!Is_split, "Don't use write_tiled with split kv kernel");
        write_tiled<NumCopyThreads>(O, tiled_copy_O, layout_O, tile_shape_O, sO, m_block, bidh, bidb, seqlen_traits_o);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
