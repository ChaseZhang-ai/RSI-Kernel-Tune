"""KernelWiki artifact PR-304 (deepseek-ai/DeepGEMM)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/csrc/apis/attention.hpp': r"""#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/smxx_fp8_fp4_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"
#endif

#include "layout.hpp"

namespace deep_gemm::attention {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_gemm_nt_skip_head_mid(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const std::tuple<int, int, int>& head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a.first);
    const auto [n , k_] = get_shape<2>(b.first);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check head splits and N
    const auto [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, disable_ue8m0_cast);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    const auto epilogue_type = fmt::format("epilogue::transform::EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat and std::get<1>(recipe.value()) != 1) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, major_sfb, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        // NOTES: Only granularity 128 and FP8 are exposed in the API
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static torch::Tensor fp8_fp4_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                        const std::tuple<torch::Tensor, torch::Tensor>& kv,
                                        const torch::Tensor& weights,
                                        const torch::Tensor& cu_seq_len_k_start,
                                        const torch::Tensor& cu_seq_len_k_end,
                                        const bool& clean_logits,
                                        const int& max_seqlen_k,
                                        const at::ScalarType& logits_dtype) {
    const auto [q_fp, q_sf] = q;
    const auto [kv_fp, kv_sf] = kv;
    const bool is_fp4 = q_sf.has_value();
    int seq_len, seq_len_kv, num_heads, head_dim;

    if (is_fp4) {
        // Check FP4 Q
        std::tie(seq_len, num_heads, head_dim) = get_shape<3>(q_fp);
        head_dim *= 2;
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == kPackedFP4);

        // Check SF Q
        auto [_seq_len, _num_heads] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);

        // Check FP4 KV
        int _head_dim;
        std::tie(seq_len_kv, _head_dim) = get_shape<2>(kv_fp);
        _head_dim *= 2;
        DG_HOST_ASSERT(head_dim == _head_dim);
        DG_HOST_ASSERT(kv_fp.is_contiguous());
        DG_HOST_ASSERT(kv_fp.scalar_type() == kPackedFP4);

        // Check SF KV
        auto [_seq_len_kv] = get_shape<1>(kv_sf);
        DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
        DG_HOST_ASSERT(kv_sf.is_contiguous());
        DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kInt32);
    } else {
        // Check FP8 Q
        std::tie(seq_len, num_heads, head_dim) = get_shape<3>(q_fp);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check FP4 KV
        int _head_dim;
        std::tie(seq_len_kv, _head_dim) = get_shape<2>(kv_fp);
        DG_HOST_ASSERT(head_dim == _head_dim);
        DG_HOST_ASSERT(kv_fp.is_contiguous());
        DG_HOST_ASSERT(kv_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check SF KV
        auto [_seq_len_kv] = get_shape<1>(kv_sf);
        DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
        DG_HOST_ASSERT(kv_sf.is_contiguous());
        DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kFloat);
    }
    
    // Check weights
    auto [_seq_len, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);

    // Check cu_seq_len_k_start
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);

    // Check cu_seq_len_k_end
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    // Allocate output
    constexpr int block_qh = 128;
    constexpr int block_kv = 256;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);

    torch::Tensor logits;
    int aligned_seq_len = align(seq_len, block_q), stride_logits;
    if (max_seqlen_k == 0) {
        // Logits stride must be 16-byte aligned
        stride_logits = align(seq_len_kv + block_kv, 8);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
    } else {
        stride_logits = align(max_seqlen_k, block_kv);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, max_seqlen_k)});
        DG_HOST_ASSERT(not clean_logits);
    }

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (is_fp4 and arch_major == 10) {
        sm100_fp4_mqa_logits(q_fp, q_sf.value(), kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                             seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv);
    } else if (not is_fp4 and (arch_major == 9 or arch_major == 10)) {
        smxx_fp8_mqa_logits(q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, stride_logits);
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const int batch_size = context_lens.size(0);
    const int next_n = context_lens.size(1);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    // Create metadata tensor
    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        DG_HOST_ASSERT(block_kv == 64 or (arch_major == 10 and block_kv == 32));
        smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, is_context_lens_2d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    return schedule_metadata;
}

static torch::Tensor fp8_fp4_paged_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                              const torch::Tensor& fused_kv_cache,
                                              const torch::Tensor& weights,
                                              const torch::Tensor& context_lens,
                                              const torch::Tensor& block_table,
                                              const torch::Tensor& schedule_meta,
                                              const int& max_context_len,
                                              const bool& clean_logits,
                                              const at::ScalarType& logits_dtype) {
    const auto [q_fp, q_sf] = q;
    const bool is_fp4 = q_sf.has_value();

    torch::Tensor kv_cache, kv_cache_sf;
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv;
    int kv_cache_stride_bytes;
    int block_table_stride = block_table.stride(0);
    int num_sms = device_runtime->get_num_sms();

    if (is_fp4) {
        // Check FP4 Q
        std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q_fp);
        head_dim *= 2;
        DG_HOST_ASSERT(next_n >= 1);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == kPackedFP4);

        // Check SF Q
        auto [_batch_size, _next_n, _num_heads] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _batch_size and next_n == _next_n and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);

        // Check fused KV cache
        int num_heads_kv, fp4_with_sf_bytes;
        std::tie(num_kv_blocks, block_kv, num_heads_kv, fp4_with_sf_bytes) = get_shape<4>(fused_kv_cache);
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        DG_HOST_ASSERT(num_heads_kv == 1 and fp4_with_sf_bytes == head_dim / 2 + static_cast<int>(sizeof(int)));
        DG_HOST_ASSERT(fused_kv_cache.stride(1) == fp4_with_sf_bytes and fused_kv_cache.stride(3) == 1);
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

        // Derive FP4 values and SF tensor
        kv_cache_stride_bytes = fused_kv_cache.stride(0);
        DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(int) == 0);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim / 2},
            {kv_cache_stride_bytes, head_dim / 2, 1},
            torch::TensorOptions().dtype(kPackedFP4)
        );
        kv_cache_sf = torch::from_blob(
            fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim / 2,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(int)), 1},
            torch::TensorOptions().dtype(torch::kInt32)
        );
    } else {
        // Check FP8 Q
        std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q_fp);
        DG_HOST_ASSERT(next_n >= 1);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check fused KV cache
        int num_heads_kv, head_dim_with_sf;
        std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        DG_HOST_ASSERT(num_heads_kv == 1 and head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
        DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf and fused_kv_cache.stride(3) == 1);
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

        // Derive FP8 values and SF tensor
        kv_cache_stride_bytes = fused_kv_cache.stride(0);
        DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
        );
        kv_cache_sf = torch::from_blob(
            fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
            torch::TensorOptions().dtype(torch::kFloat32)
        );

        // Weights must be contiguous for FP8
        DG_HOST_ASSERT(weights.is_contiguous());
    }

    // Check weights
    auto [_batch_size_next_n, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(_batch_size_next_n == batch_size * next_n and _num_heads == num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);

    // Check block table
    auto [_batch_size, _max_block_len] = get_shape<2>(block_table);
    DG_HOST_ASSERT(_batch_size == batch_size);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);

    // Check schedule metadata
    auto [_schedule_meta_size, _meta_info_size] = get_shape<2>(schedule_meta);
    DG_HOST_ASSERT(_schedule_meta_size == num_sms + 1 and _meta_info_size == 2);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    // Check context lengths
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const auto [__batch_size, _next_n] = get_shape<2>(context_lens);
    DG_HOST_ASSERT(batch_size == __batch_size and next_n == _next_n);
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);

    // Allocate output
    constexpr int split_kv = 256;
    const auto aligned_max_context_len = align(max_context_len, split_kv);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q_fp.options().dtype(logits_dtype));
    logits = logits.slice(-1, 0, max_context_len);
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (is_fp4 and arch_major == 10) {
        sm100_fp4_paged_mqa_logits(q_fp, q_sf.value(), kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, schedule_meta,
                                   logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                   aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else if (not is_fp4 and (arch_major == 9 or arch_major == 10)) {
        smxx_fp8_paged_mqa_logits(q_fp, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, schedule_meta,
                                  logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits) {
        DG_HOST_ASSERT(not is_context_lens_2d);
        smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    }
    return logits;
}


// Legacy API wrappers
static torch::Tensor fp8_mqa_logits(const torch::Tensor& q,
                                    const std::tuple<torch::Tensor, torch::Tensor>& kv,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seq_len_k_start,
                                    const torch::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits,
                                    const int& max_seqlen_k) {
    return fp8_fp4_mqa_logits(std::make_tuple(q, std::nullopt), kv, weights, 
                              cu_seq_len_k_start, cu_seq_len_k_end,
                              clean_logits, max_seqlen_k, torch::kFloat);
}

static torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                          const torch::Tensor& fused_kv_cache,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& context_lens,
                                          const torch::Tensor& block_table,
                                          const torch::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits) {
    return fp8_fp4_paged_mqa_logits(std::make_tuple(q, std::nullopt), fused_kv_cache, weights,
                                    context_lens, block_table, schedule_meta,
                                    max_context_len, clean_logits, torch::kFloat);
}
#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("fp8_gemm_nt_skip_head_mid", &fp8_gemm_nt_skip_head_mid,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("head_splits"),
          py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_mqa_logits", &fp8_fp4_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0,
          py::arg("logits_dtype") = torch::kFloat32);
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"));
    m.def("fp8_fp4_paged_mqa_logits", &fp8_fp4_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("clean_logits") = false,
          py::arg("logits_dtype") = torch::kFloat32);
    // Legacy API
    m.def("fp8_mqa_logits", &fp8_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0);
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false);
#endif
}

} // namespace deep_gemm::attention
""",
    'key-files/csrc/apis/einsum.hpp': r"""#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"
#endif

namespace deep_gemm::einsum {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void bmk_bnk_mn(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                       const std::optional<torch::Tensor>& c) {
    // Currently FP32 only support the accumulated expression
    if (d.scalar_type() == torch::kFloat) {
        DG_HOST_ASSERT(c->data_ptr() == d.data_ptr() and c->sizes() == d.sizes() and c->strides() == d.strides());
    } else {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(not c.has_value());

        const auto workspace = torch::empty_like(d, d.options().dtype(torch::kFloat32));
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace.data_ptr(), 0, workspace.nbytes(),
                              c10::cuda::getCurrentCUDAStream()));
        bmk_bnk_mn(a, b, workspace, workspace);

        // This line has an implicit FP32-to-BF16 casting
        d.copy_(workspace);
        return;
    }

    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    const auto [s , m, k ] = get_shape<3>(a);
    const auto [s_, n, k_] = get_shape<3>(b);
    DG_HOST_ASSERT(s == s_ and k == k_);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 10) {
        sm100_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D, const bool& use_cublaslt) {
    const auto [b , h  , r ] = get_shape<3>(A);
    const auto [h_, d  , r_] = get_shape<3>(B);
    const auto [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (use_cublaslt) {
        cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_hdr_bhr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D, const bool& use_cublaslt) {
    const auto [b , h  , d ] = get_shape<3>(A);
    const auto [h_, d_ , r ] = get_shape<3>(B);
    const auto [b_, h__, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (use_cublaslt) {
        cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void einsum(const std::string& expr,
                   const torch::Tensor& a,
                   const torch::Tensor& b,
                   const torch::Tensor& d,
                   const std::optional<torch::Tensor>& c,
                   const bool& use_cublaslt) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        DG_HOST_ASSERT(c->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    }

    // Some hardcoded Einstein sum kernels
    // TODO: support any expression
    // TODO: canonicalize expression
    if (expr == "bmk,bnk->mn") {
        DG_HOST_ASSERT(not use_cublaslt);
        bmk_bnk_mn(a, b, d, c);
    } else if (expr == "bhr,hdr->bhd") {
        DG_HOST_ASSERT(not c.has_value());
        bhr_hdr_bhd(a, b, d, use_cublaslt);
    } else if (expr == "bhd,hdr->bhr") {
        DG_HOST_ASSERT(not c.has_value());
        bhd_hdr_bhr(a, b, d, use_cublaslt);
    } else {
        DG_HOST_UNREACHABLE(fmt::format("Unsupported einsum expression: {}", expr));
    }
}

static void fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                    const torch::Tensor& b, const torch::Tensor& sfb,
                    const torch::Tensor& d,
                    const std::optional<torch::Tensor>& c,
                    std::optional<std::tuple<int, int, int>> recipe,
                    const std::string& compiled_dims) {
    // Shape must be `[B, M, K] @ [B, N, K].T`
    const auto major_a = a.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    const auto major_b = b.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d.stride(-1) == 1);

    // Type and shape checks
    const auto [batch_size  , m , k ] = get_shape<3>(a);
    const auto [batch_size_ , n , k_] = get_shape<3>(b);
    const auto [batch_size__, m_, n_] = get_shape<3>(d);
    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size_);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (batch_size == 0 or gemm::early_return(m, n, k, d, c))
        return;

    // Transform scaling factors
    const auto [transformed_sfa, transformed_sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d, batch_size, m, n, k, gran_k_a, gran_k_b, major_a, major_b, compiled_dims);
    } else {
        const auto major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);
        sm90_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d, batch_size, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    }
}

static void fp8_einsum(const std::string& expr,
                       const std::pair<torch::Tensor, torch::Tensor>& a,
                       const std::pair<torch::Tensor, torch::Tensor>& b,
                       const torch::Tensor& d,
                       const std::optional<torch::Tensor>& c,
                       const std::tuple<int, int, int>& recipe) {
    // Some hardcoded Einstein sum kernels
    const auto arch_major = device_runtime->get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        // Permute dims to satisfy the order of (batch_size, m, n, k)
        // (batch_size, m, n, k): (h, b, d, r)
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        const auto perm_d = d.permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, b.first, b.second, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,hdr->bhr" and arch_major == 10) {
        // (batch_size, m, n, k): (h, b, r, d)
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        const auto perm_b = b.first.permute({0, 2, 1});
        const auto perm_sfb = b.second.permute({0, 2, 1});
        const auto perm_d = d.permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,bhr->hdr" and arch_major == 10) {
        // (batch_size, m, n, k): (h, d, r, b)
        const auto perm_a = a.first.permute({1, 2, 0});
        const auto perm_sfa = a.second.permute({1, 2, 0});
        const auto perm_b = b.first.permute({1, 2, 0});
        const auto perm_sfb = b.second.permute({1, 2, 0});
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, d, c, recipe, "mn");
    } else {
        DG_HOST_UNREACHABLE(fmt::format("Unsupported einsum expression: {}", expr));
    }
}
#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("use_cublaslt") = false);
    m.def("fp8_einsum", &fp8_einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"),  py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 128, 128));
#endif
}

} // namespace deep_gemm::einsum
""",
    'key-files/csrc/apis/gemm.hpp': r"""#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#endif 

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            std::optional<std::tuple<int, int, int>> recipe,
                            std::optional<std::tuple<int, int>> recipe_a,
                            std::optional<std::tuple<int, int>> recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n , k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);

    // Dispatch into different implements
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
        if (gran_n == 1) {
            sm90_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
        } else {
            const auto major_sfb = get_major_type_ab(sfb);
            sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
        }
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void fp8_fp4_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                    {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                num_groups, m, n, k, major_a, major_b, major_sfb,
                                                compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, major_b,
                                                     compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const std::optional<std::tuple<int, int, int>>& recipe,
                                                 const std::optional<std::tuple<int, int>>& recipe_a,
                                                 const std::optional<std::tuple<int, int>>& recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout) {
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                         d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, std::nullopt);
}

static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups_ , n , k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, major_sfb, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::vector<int>& ks,
                                             const torch::Tensor& ks_tensor,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a.first);
    const auto [sum_k__, n_] = get_shape<2>(b.first);
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, std::accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks, ks_tensor, recipe);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks, ks_tensor, recipe);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks, ks_tensor, gran_k,
                                      cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::vector<int>& ks,
                                             const torch::Tensor& ks_tensor,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto sum_mk = a.first.numel();
    const auto sum_nk = b.first.numel();
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
    DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks, ks_tensor, recipe);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks, ks_tensor, recipe);

    // Allocate tensormap buffer
    // `4` means the double buffering for both A and B operands (2 * 2)
    const auto num_sms = device_runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
                                                a.first.options().dtype(torch::kByte));

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks, ks_tensor, tensor_map_buffer,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

#if DG_TENSORMAP_COMPATIBLE
static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims,
                                            use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_bf16_gemm_nn_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout) {
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2),
                                      d, grouped_layout, compiled_dims, use_psum_layout, std::nullopt);
}

static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                          const torch::Tensor& d, const torch::Tensor& masked_m,
                                          const int& expected_m, const std::string& compiled_dims) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto [num_groups, m, k] = get_shape<3>(a);
    const auto [num_groups_, n, k_] = get_shape<3>(b);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_contiguous(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const std::vector<int>& ks,
                                              const torch::Tensor& ks_tensor,
                                              const std::optional<torch::Tensor>& c,
                                              const std::string& compiled_dims) {
    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a);
    const auto [sum_k__, n_] = get_shape<2>(b);
    const int sum_k = std::accumulate(ks.begin(), ks.end(), 0);
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, std::accumulate(ks.begin(), ks.end(), 0), d, c))
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks, ks_tensor,
                                 cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_k_grouped_gemm(a, b, c, d, m, n, ks, ks_tensor,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

static void cublaslt_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

static void cublaslt_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}

static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // FP8 FP4 GEMMs
    m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_nn", &fp8_fp4_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_tn", &fp8_fp4_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_tt", &fp8_fp4_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous", &m_grouped_fp8_fp4_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false,
          py::arg("expected_m_for_psum_layout") = std::nullopt);
    m.def("m_grouped_fp8_fp4_gemm_nn_contiguous", &m_grouped_fp8_fp4_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false);
    m.def("m_grouped_fp8_fp4_gemm_nt_masked", &m_grouped_fp8_fp4_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false);
    m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
          py::arg("ks_tensor"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn");
    m.def("k_grouped_fp8_gemm_nt_contiguous", &k_grouped_fp8_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
          py::arg("ks_tensor"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn");

    // FP8 GEMM alias names
    m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
    m.attr("fp8_gemm_nn") = m.attr("fp8_fp4_gemm_nn");
    m.attr("fp8_gemm_tn") = m.attr("fp8_fp4_gemm_tn");
    m.attr("fp8_gemm_tt") = m.attr("fp8_fp4_gemm_tt");
    m.attr("m_grouped_fp8_gemm_nt_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nt_contiguous");
    m.attr("m_grouped_fp8_gemm_nn_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nn_contiguous");
    m.attr("m_grouped_fp8_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
#endif

#if DG_TENSORMAP_COMPATIBLE
    // BF16 GEMMs
    m.def("bf16_gemm_nt", &bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_nn", &bf16_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_tn", &bf16_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("bf16_gemm_tt", &bf16_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("expected_m_for_psum_layout") = std::nullopt);
    m.def("m_grouped_bf16_gemm_nn_contiguous", &m_grouped_bf16_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false);
    m.def("m_grouped_bf16_gemm_nt_masked", &m_grouped_bf16_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("compiled_dims") = "nk");
    m.def("k_grouped_bf16_gemm_tn_contiguous", &k_grouped_bf16_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
          py::arg("ks_tensor"), py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
#endif

    // cuBLASLt GEMMs
    m.def("cublaslt_gemm_nt", &cublaslt_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_nn", &cublaslt_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tn", &cublaslt_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tt", &cublaslt_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
}

} // namespace deep_gemm::gemm
""",
    'key-files/csrc/apis/hyperconnection.hpp': r"""#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp"
#endif

namespace deep_gemm::hyperconnection {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 const std::optional<int>& num_splits) {
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    check_major_type_cd(d);

    // S must be contiguous
    DG_HOST_ASSERT(sqr_sum.is_contiguous());

    // Type and shape checks
    const auto [m, k ] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    if (num_splits.has_value()) {
        const auto [num_splits_, m_, n_] = get_shape<3>(d);
        const auto [num_splits__, m__] = get_shape<2>(sqr_sum);
        DG_HOST_ASSERT(num_splits.value() == num_splits_ and num_splits.value() == num_splits__ and num_splits.value() >= 1);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    } else {
        const auto [m_, n_] = get_shape<2>(d);
        const auto [m__] = get_shape<1>(sqr_sum);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    }
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 10) {
        sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
#endif
}

} // namespace deep_gemm::hyperconnection
""",
    'key-files/csrc/apis/layout.hpp': r"""#pragma once

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_layout.hpp"
#endif

namespace deep_gemm::layout {

#if DG_TENSORMAP_COMPATIBLE
static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::variant<std::tuple<int, int, int>,
                                                                          std::tuple<int, int>>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const std::optional<bool>& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
    const auto arch_major = device_runtime->get_arch_major();

    // Get granularity MN/K from recipe
    int gran_mn, gran_k;
    if (auto p = std::get_if<std::tuple<int, int, int>>(&recipe)) {
        DG_HOST_ASSERT(is_sfa.has_value());
        gran_mn = is_sfa.value() ? std::get<0>(*p) : std::get<1>(*p);
        gran_k = std::get<2>(*p);
    } else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {
        DG_HOST_ASSERT(not is_sfa.has_value());
        std::tie(gran_mn, gran_k) = *p;
    } else {
        DG_HOST_UNREACHABLE("Invalid recipe");
    }

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, x, gran_k) on SM100: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto broadcasted = gran_mn == 1 ? sf :
                                 sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, gran_k) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(
        const torch::Tensor& sfa, const torch::Tensor& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false) {
    // Use default recipe, if none is specified
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());

    // Must be either 'recipe' or the 'recipe_a' + 'recipe_b' pair.
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());

    // Transform SFA and SFB layout
    const auto transformed_sfa = recipe.has_value() ? transform_sf_into_required_layout(sfa, m, k, recipe.value(), num_groups_a, true, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfa, m, k, recipe_a.value(), num_groups_a, std::nullopt, disable_ue8m0_cast);
    const auto transformed_sfb = recipe.has_value() ? transform_sf_into_required_layout(sfb, n, k, recipe.value(), num_groups_b, false, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfb, n, k, recipe_b.value(), num_groups_b, std::nullopt, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::vector<int>& ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);

    const auto arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k);

    // INT on SM100
    if (sf.scalar_type() == torch::kInt and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
      py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
      py::arg("num_groups") = std::nullopt,
      py::arg("is_sfa") = std::nullopt,
      py::arg("disable_ue8m0_cast") = false);

    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
#endif

    m.def("set_mk_alignment_for_contiguous_layout", [&](const int& new_value) {
        heuristics_runtime->set_mk_alignment_for_contiguous_layout(new_value);
    });
    m.def("get_mk_alignment_for_contiguous_layout", [&]() {
        return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    });
    m.def("get_theoretical_mk_alignment_for_contiguous_layout", [&](const std::optional<int>& expected_m) {
        return heuristics_runtime->get_theoretical_mk_alignment_for_contiguous_layout(expected_m);
    }, py::arg("expected_m") = std::nullopt);
}

} // namespace deep_gemm::layout
""",
    'key-files/csrc/apis/mega.hpp': r"""#pragma once

#include <functional>
#include <pybind11/functional.h>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"

namespace deep_gemm::mega {

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);

    // Workspace bytes
    const auto block_m = get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const auto workspace = layout::Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, block_m);

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 32);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Buffer configs
    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    const auto num_padded_sf_pool_tokens = layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m);

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    // Check SF buffer requirements
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(num_padded_sf_pool_tokens % 4 == 0);

    // Slice function: creates `(x, x_sf, topk_weights, topk_idx, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf)` tensor views from the raw buffer
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_padded_sf_pool_tokens, hidden / 128},
            {1, num_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_padded_sf_pool_tokens, intermediate_hidden / 128},
            {1, num_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_;
    const auto [l2_weights, l2_weights_sf] = l2_weights_;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

   // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, "swiglu");
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_block_m_for_mega_moe", &get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
#endif
}

} // namespace deep_gemm::mega
""",
    'key-files/csrc/apis/runtime.hpp': r"""#pragma once

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"

namespace deep_gemm::runtime {

static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("set_pdl", [&](const bool& new_enable_pdl) {
        device_runtime->set_pdl(new_enable_pdl);
    });
    m.def("get_pdl", [&]() {
        return device_runtime->get_pdl();
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
        IncludeParser::prepare_init(library_root_path);
#endif
    });
}

} // namespace deep_gemm::runtime
""",
    'key-files/csrc/indexing/main.cu': r"""// GEMM kernels
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

// Attention kernels
#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp4_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh>
#include <deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh>

// Einsum kernels
#include <deep_gemm/impls/sm90_bmk_bnk_mn.cuh>
#include <deep_gemm/impls/sm100_bmk_bnk_mn.cuh>

// Hyperconnection kernels
#include <deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh>
#include <deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh>

// Layout kernels
#include <deep_gemm/impls/smxx_layout.cuh>
#include <deep_gemm/impls/smxx_clean_logits.cuh>

// Mega kernels
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

int main() {
    return 0;
}
""",
    'key-files/csrc/jit/cache.hpp': r"""#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
""",
    'key-files/csrc/jit/compiler.hpp': r"""#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <regex>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path cuda_home;
    static std::filesystem::path cuobjdump_path;

    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not cuobjdump_path.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))
            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    static void fsync_path(const std::filesystem::path& path) {
        const auto fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }

    // Recursively fsync a directory: files and subdirectories first (bottom-up), then the directory itself
    // NOTES: ensures data and directory entries are visible on other nodes in distributed filesystems
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_directory())
                fsync_dir(entry.path());
            else if (entry.is_regular_file())
                fsync_path(entry.path());
        }
        fsync_path(dir_path);
    }

    static void put(const std::filesystem::path& path, const std::string& data) {
        std::ofstream out(path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        fsync_path(path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Compile into a temporary directory, then atomically rename the whole directory
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);

        // Compile into the temporary directory
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_PTX")) {
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
        } else {
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // Disassemble if needed
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_SASS")) {
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            disassemble(tmp_cubin_path, tmp_sass_path);
        }

        // Fsync before rename to ensure visibility on distributed filesystems
        fsync_dir(tmp_dir_path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(dir_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        if (error_code) {
            // Another rank beat us, then clean up our dir and use the existing one
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            safe_remove_all(tmp_dir_path);
        }

        // Put into the runtime cache
        const auto runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("cuobjdump failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto command = std::string(nvcc_path) + " --version";
        const auto [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        // Avoid cwd files shadowing C++ standard library headers
        const auto compile_dir = make_tmp_dir();
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            if (ptx_return_code != 0) {
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                DG_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // Check local memory usage
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class NVRTCCompiler final: public Compiler {
public:
    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories list
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        // Add PCH support for version 12.8 and above
        // NOTES: PCH is vital for compilation speed
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Override the compiler flags
        // Only NVRTC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {} --device-int128",
                            flags, include_dirs, arch, pch_flags);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Parse compilation options
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        // Convert to C-style string array for NVRTC
        std::vector<const char*> option_cstrs;
        for (const auto& opt: options)
            option_cstrs.push_back(opt.c_str());

        // Print compiler command if requested
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt: options)
                printf("%s ", opt.c_str());
            printf("\n");
        }

        // Create NVRTC program and compile
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto compile_result = nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        // Get and print compiler log
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        if (ptx_path.has_value()) {
            // Get PTX size and data if needed
            size_t ptx_size;
            DG_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
            std::string ptx_data(ptx_size, '\0');
            DG_NVRTC_CHECK(nvrtcGetPTX(program, ptx_data.data()));

            // Write into the file system
            put(ptx_path.value(), ptx_data);
        }

        // Get CUBIN size and data
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);

        // Cleanup
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace deep_gemm
""",
    'key-files/csrc/jit/device_runtime.hpp': r"""#pragma once

#include <cublasLt.h>
#include <torch/version.h>
#include <ATen/cuda/CUDAContext.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

#define PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3))

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0, tc_util = 0;
    bool enable_pdl = false;
    std::shared_ptr<cudaDeviceProp> cached_prop;

    // cuBLASLt utils
    static constexpr size_t kCublasLtWorkspaceSize = 32 * 1024 * 1024;

public:
    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    torch::Tensor cublaslt_workspace;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    explicit DeviceRuntime() {

        // Whether to use PyTorch cuBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_cublaslt_handle = get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;
#if not PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE
        DG_HOST_ASSERT(not use_pytorch_managed_cublaslt_handle and "PyTorch does not support to get cuBLASLt handle");
#endif

        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
        // when the workspace tensor is destructed after CUDA driver shutdown.
        use_temp_cublaslt_workspace = get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));

        if (not use_temp_cublaslt_workspace)
            cublaslt_workspace = torch::empty({kCublasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
    }

    ~DeviceRuntime() noexcept(false) {
        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
#if PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE
        if (use_pytorch_managed_cublaslt_handle)
            return at::cuda::getCurrentCUDABlasLtHandle();
#endif

        // Self-managed handle
        return cublaslt_handle;
    }

    torch::Tensor get_cublaslt_workspace() const {
        if (use_temp_cublaslt_workspace)
            return torch::empty({kCublasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
        return cublaslt_workspace;
    }

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr) {
            int device_idx;
            cudaDeviceProp prop;
            DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
            DG_CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_idx));
            cached_prop = std::make_shared<cudaDeviceProp>(prop);
        }
        return cached_prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= get_prop()->multiProcessorCount);
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = get_prop()->multiProcessorCount;
        return num_sms;
    }

    int get_l2_cache_size() {
        return get_prop()->l2CacheSize;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

    void set_pdl(const bool& new_enable_pdl) {
        enable_pdl = new_enable_pdl;
    }

    bool get_pdl() const {
        return enable_pdl;
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace deep_gemm
""",
    'key-files/csrc/jit/handle.hpp': r"""#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>

#include "../utils/exception.hpp"
#include "../utils/compatibility.hpp"

namespace deep_gemm {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        DG_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&(name)); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        DG_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryLoadFromFile);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuKernelGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

#if CUDART_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_RUNTIME_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel{};
    DG_CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    DG_CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto error = cudaLibraryUnload(library);
    DG_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim, const bool& enable_pdl) {
    if (smem_size > 0)
        DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    LaunchConfigHandle config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Create attributes
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attrs[2];
    config.numAttrs = 0;
    config.attrs = attrs;

    // Cluster size
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
    }

    // Dependent kernel launch
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
    }

    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use CUDA driver API
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

// `cuLibraryEnumerateKernels` is supported since CUDA Driver API 12.4
#if CUDA_VERSION >= 12040
    #define DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryGetKernelCount);
    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);
    using LibraryHandle = CUlibrary;
#else
    using LibraryHandle = CUmodule;
#endif

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_DRIVER_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel;

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    unsigned int num_kernels;
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryGetKernelCount(&num_kernels, library));
    if (num_kernels != 1) {
        const auto dir_path = cubin_path.parent_path();
        printf("Corrupted JIT cache directory (expected 1 kernel, found %u): %s, "
               "please run `rm -rf %s` and restart your task.\n",
               num_kernels, dir_path.c_str(), dir_path.c_str());
        DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
    }

    CUkernel cu_kernel;
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryEnumerateKernels(&cu_kernel, 1, library));
    DG_CUDA_DRIVER_CHECK(lazy_cuKernelGetFunction(&kernel, cu_kernel));
#else
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));
#endif

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    const auto error = lazy_cuLibraryUnload(library);
#else
    const auto error = lazy_cuModuleUnload(library);
#endif
    DG_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                 const cudaStream_t& stream, const int& smem_size,
                                                 const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim, const bool& enable_pdl) {
    if (smem_size > 0)
        DG_CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));

    LaunchConfigHandle config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_size;
    config.hStream = stream;
    
    // Create attributes
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attrs[2];
    config.numAttrs = 0;
    config.attrs = attrs;

    // Cluster size
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim.x = static_cast<unsigned>(cluster_dim);
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
    }

    // Dependent kernel launch
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
    }

    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}
#endif

} // namespace deep_gemm
""",
    'key-files/csrc/jit/include_parser.hpp': r"""#pragma once

#include <filesystem>
#include <regex>
#include <string>
#include <vector>

#include "../utils/format.hpp"
#include "../utils/system.hpp"

namespace deep_gemm {

class IncludeParser {
    std::unordered_map<std::string, std::optional<std::string>> cache;

    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {
        std::vector<std::string> includes;
        const std::regex pattern(R"(#\s*include\s*[<"][^>"]+[>"])");
        std::sregex_iterator iter(code.begin(), code.end(), pattern);
        const std::sregex_iterator end;

        // TODO: parse relative paths as well
        for (; iter != end; ++ iter) {
            const auto include_str = iter->str();
            const int len = include_str.length();
            if (include_str.substr(0, 10) == "#include <" and include_str[len - 1] == '>' and include_str[10] != ' ' and include_str[len - 2] != ' ') {
                std::string filename = include_str.substr(10, len - 11);
                if (filename.substr(0, 9) == "deep_gemm")  // We only parse `<deep_gemm/*>`
                    includes.push_back(filename);
            } else {
                std::string error_info = fmt::format("Non-standard include: {}", include_str);
                if (file_path != "")
                    error_info += fmt::format(" ({})", file_path.string());
                DG_HOST_UNREACHABLE(error_info);
            }
        }
        return includes;
    }

public:
    static std::filesystem::path library_include_path;

    static void prepare_init(const std::string& library_root_path) {
        library_include_path = std::filesystem::path(library_root_path) / "include";
    }

    std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {
        std::stringstream ss;
        for (const auto& i: get_includes(code))
            ss << get_hash_value_by_path(library_include_path / i) << "$";
        if (not exclude_code)
            ss << "#" << get_hex_digest(code);
        return get_hex_digest(ss.str());
    }

    std::string get_hash_value_by_path(const std::filesystem::path& path) {
        // Check whether hit in cache
        // ReSharper disable once CppUseAssociativeContains
        if (cache.count(path) > 0) {
            const auto opt = cache[path];
            if (not opt.has_value())
                DG_HOST_UNREACHABLE(fmt::format("Circular include may occur: {}", path.string()));
            return opt.value();
        }

        // Read file and calculate hash recursively
        std::ifstream in(path);
        if (not in.is_open())
            DG_HOST_UNREACHABLE(fmt::format("Failed to open: {}", path.string()));
        std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        cache[path] = std::nullopt;
        return (cache[path] = get_hash_value(code, false)).value();
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(IncludeParser, library_include_path);

static auto include_parser = std::make_shared<IncludeParser>();

}  // namespace deep_gemm
""",
    'key-files/csrc/jit/kernel_runtime.hpp': r"""#pragma once

#include <chrono>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"
#include "handle.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

struct LaunchArgs {
    std::pair<int, int> grid_dim;
    int num_threads;
    int smem_size;
    int cluster_dim;
    bool enable_pdl;

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& enable_pdl = true):
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), enable_pdl(enable_pdl) {}

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& enable_pdl = true):
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), enable_pdl(enable_pdl) {}
};

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;

    LibraryHandle library;
    KernelHandle kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // Check `prepare_init`
        DG_HOST_ASSERT(not cuda_home.empty());

        // NOLINT(*-pro-type-member-init)
        const auto cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        const auto cubin_path = dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // Record start time
        std::chrono::high_resolution_clock::time_point start_time;
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME"))
            start_time = std::chrono::high_resolution_clock::now();

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
        // Load from the library
        kernel = load_kernel(cubin_path, {}, &library);
#else
        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        const auto [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        DG_HOST_ASSERT(exit_code == 0);
        std::istringstream iss(symbols);
        std::vector<std::string> symbol_names;
        for (std::string line; std::getline(iss, line); ) {
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto name) { return line.find(name) != std::string::npos; })) {
                const auto last_space = line.rfind(' ');
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }

        // Print symbols
        if (symbol_names.size() != 1 or get_env<int>("DG_JIT_DEBUG")) {
            printf("Symbols: ");
            printf(" > CUBIN: %s\n", cubin_path.c_str());
            printf(" > Raw symbols: %s\n", symbols.c_str());
            printf(" > Parsed symbols:\n");
            for (const auto& symbol: symbol_names)
                printf("   > %s, ", symbol.c_str());
        }
        DG_HOST_ASSERT(symbol_names.size() == 1);

        // Load from the library
        kernel = load_kernel(cubin_path, symbol_names[0], &library);
#endif

        // Print load time
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME")) {
            std::chrono::duration<double, std::milli> load_time = std::chrono::high_resolution_clock::now() - start_time;
            printf("Load time (%s): %.2lf ms\n", dir_path.c_str(), load_time.count());
        }
    }

    static void prepare_init(const std::string& cuda_home_path_by_python) {
        cuda_home = cuda_home_path_by_python;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        if (not std::filesystem::exists(dir_path))
            return false;

        // NOTES: if the directory exists, `kernel.cu` and `kernel.cubin` must both exist,
        // because the directory is created atomically via rename
        if (not std::filesystem::exists(dir_path / "kernel.cu") or
            not std::filesystem::exists(dir_path / "kernel.cubin")) {
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.cubin): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());
            DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }
        return true;
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);

        // NOTES: we require that `generate_impl`'s includes never change
        static std::string include_hash;
        if (include_hash.empty())
            include_hash = include_parser->get_hash_value(code);

        // TODO: optimize string concat performance
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Generated kernel code:\n%s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        const auto kernel = kernel_runtime->kernel;
        const auto stream = at::cuda::getCurrentCUDAStream();
        LaunchArgs launch_args = args.launch_args;

        // Allow runtime override from Python.
        // NOTES: the default is enabled.
        launch_args.enable_pdl = device_runtime->get_pdl();

        const dim3 grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
                               static_cast<unsigned>(launch_args.grid_dim.second),
                               1};
        const dim3 block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        auto config = construct_launch_config(kernel, stream, launch_args.smem_size,
                                              grid_dim, block_dim, launch_args.cluster_dim, launch_args.enable_pdl);

        // Launch in the derived class
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d, shared memory: %d bytes, cluster: %d, pdl: %d, stream: %ld\n",
                   launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                   launch_args.smem_size, launch_args.cluster_dim, launch_args.enable_pdl, stream.id());
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/common.hpp': r"""#pragma once

#include <unordered_set>
#include <deep_gemm/common/types.cuh>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        std::stringstream ss;
        ss << desc;
        const auto key = ss.str();

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/config.hpp': r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <c10/core/ScalarType.h>
#include <deep_gemm/common/types.cuh>

#include "../../utils/math.hpp"

namespace deep_gemm {

/// GEMM descriptors
struct GemmDesc {
    GemmType gemm_type;
    KernelType kernel_type;
    int m, n, k, num_groups;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    cute::UMMA::Major major_a;
    cute::UMMA::Major major_b;
    bool with_accumulation;

    // Requirements from users
    int num_sms, tc_util;
    std::string compiled_dims;

    // Shape for heuristic generation
    int expected_m = 0, expected_n = 0, expected_k = 0, expected_num_groups = 0;
    int get_expected_m() const { return expected_m > 0 ? expected_m : m; }
    int get_expected_n() const { return expected_n > 0 ? expected_n : n; }
    int get_expected_k() const { return expected_k > 0 ? expected_k : k; }
    int get_expected_num_groups() const { return expected_num_groups > 0 ? expected_num_groups : num_groups; }

    MmaKind get_mma_kind() const {
        return a_dtype == torch::kBFloat16 ? MmaKind::BF16 : MmaKind::MXFP8FP4;
    }

    void check_validity() const {
        if (get_mma_kind() == MmaKind::BF16) {
            DG_HOST_ASSERT(a_dtype == torch::kBFloat16 and b_dtype == torch::kBFloat16);
        } else {
            DG_HOST_ASSERT(a_dtype == torch::kFloat8_e4m3fn or a_dtype == kPackedFP4);
            DG_HOST_ASSERT(b_dtype == torch::kFloat8_e4m3fn or b_dtype == kPackedFP4);
        }
        DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat);
        DG_HOST_ASSERT(num_sms % 2 == 0);
    }

    friend std::ostream& operator << (std::ostream& os, const GemmDesc& desc) {
        MmaKind mma_kind = desc.get_mma_kind();
        os << "GemmDesc(gemm_type=" << static_cast<int>(desc.gemm_type)
           << ", kernel_type=" << static_cast<int>(desc.kernel_type)
           << ", m=" << desc.m << ", n=" << desc.n << ", k=" << desc.k
           << ", num_groups=" << desc.num_groups
           << ", major_a=" << static_cast<int>(desc.major_a)
           << ", major_b=" << static_cast<int>(desc.major_b)
           << ", mma_kind=" << static_cast<int>(mma_kind)
           << ", a_dtype=" << c10::toString(desc.a_dtype)
           << ", b_dtype=" << c10::toString(desc.b_dtype)
           << ", cd_dtype=" << c10::toString(desc.cd_dtype)
           << ", with_accumulation=" << static_cast<int>(desc.with_accumulation)
           << ", num_sms=" << desc.num_sms
           << ", tc_util=" << desc.tc_util
           << ", compiled_dims=" << desc.compiled_dims
           << ", expected_m=" << desc.expected_m
           << ", expected_n=" << desc.expected_n
           << ", expected_k=" << desc.expected_k
           << ", expected_num_groups=" << desc.expected_num_groups << ")";
        return os;
    }
};

/// GEMM configs
struct Layout {
    int swap_ab;
    int block_m, block_n, block_k;
    int cluster_m, cluster_n;

    int get_cluster_size() const {
        return cluster_m * cluster_n;
    }

    friend std::ostream& operator << (std::ostream& os, const Layout& layout) {
        os << "Layout(swap_ab=" << layout.swap_ab
           << ", block_m=" << layout.block_m << ", block_n=" << layout.block_n << ", block_k=" << layout.block_k
           << ", cluster_m=" << layout.cluster_m << ", cluster_n=" << layout.cluster_n << ")";
        return os;
    }
};

struct StorageConfig {
    int load_block_m, load_block_n;
    int store_block_m, store_block_n;

    int swizzle_a_mode, swizzle_b_mode;
    int swizzle_cd_mode;

    friend std::ostream& operator << (std::ostream& os, const StorageConfig& config) {
        os << "StorageConfig("
           << "load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m << ", store_block_n=" << config.store_block_n
           << ", swizzle_a_mode=" << config.swizzle_a_mode << ", swizzle_b_mode=" << config.swizzle_b_mode
           << ", swizzle_cd_mode=" << config.swizzle_cd_mode << ")";
        return os;
    }
};

struct PipelineConfig {
    int smem_size;
    int num_stages;

    friend std::ostream& operator << (std::ostream& os, const PipelineConfig& config) {
        os << "PipelineConfig("
           << "smem_size=" << config.smem_size
           << ", num_stages=" << config.num_stages << ")";
        return os;
    }
};

struct LaunchConfig {
    int num_sms;
    int num_sms_per_cluster;
    int num_threads;

    int num_tma_threads;
    int num_math_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const LaunchConfig& config) {
        os << "LaunchConfig("
           << "num_sms=" << config.num_sms << ", num_sms_per_cluster=" << config.num_sms_per_cluster
           << ", num_threads=" << config.num_threads
           << ", num_tma_threads=" << config.num_tma_threads << ", num_math_threads=" << config.num_math_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

struct GemmConfig {
    Layout layout;
    StorageConfig storage_config;
    PipelineConfig pipeline_config;
    LaunchConfig launch_config;

    friend std::ostream& operator << (std::ostream& os, const GemmConfig& config) {
        os << "GemmConfig("
           << "layout=" << config.layout
           << ", storage_config=" << config.storage_config
           << ", pipeline_config=" << config.pipeline_config
           << ", launch_config=" << config.launch_config << ")";
        return os;
    }
};

/// Config comparators
struct LayoutInfo {
    int num_waves;
    int last_wave_util;
    int64_t num_cycles;
    Layout layout;

    friend std::ostream& operator << (std::ostream& os, const LayoutInfo& config) {
        os << "LayoutInfo("
           << "num_waves=" << config.num_waves
           << ", last_wave_util=" << config.last_wave_util 
           << ", num_cycles=" << config.num_cycles << ")";
        return os;
    }
};

}  // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/mega_moe.hpp': r"""#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static int get_block_m_for_mega_moe(const int& num_ranks, const int& num_experts,
                                    const int& num_max_tokens_per_rank, const int& num_topk) {
    // TODO: compute based on configs
    return 192;
}

static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    // Reduce per-expert block count by this factor since uneven routing leaves some experts with fewer tokens
    constexpr int kImbalanceFactor = 2;

    // TODO: support num_experts_per_rank > 32
    // Find the largest divisor of num_experts_per_rank that fits in 32 as the upper bound
    int max_num_experts_per_wave = std::min(32, num_experts_per_rank);
    while (max_num_experts_per_wave > 1 and num_experts_per_rank % max_num_experts_per_wave != 0)
        -- max_num_experts_per_wave;

    // Count L1 blocks per expert assuming tokens are evenly spread across experts
    const int expected_tokens_per_expert =
        num_tokens * num_topk / num_experts_per_rank + 1;
    const int num_m_blocks = ceil_div(expected_tokens_per_expert, block_m);
    const int num_n_blocks = intermediate_hidden / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

    // Pick the smallest value whose total blocks (after imbalance reduction) can keep all SMs busy
    int num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    num_experts_per_wave = std::min(num_experts_per_wave, max_num_experts_per_wave);

    // Round up to the nearest divisor of num_experts_per_rank so every wave processes the same count
    while (num_experts_per_wave < max_num_experts_per_wave and num_experts_per_rank % num_experts_per_wave != 0)
        ++ num_experts_per_wave;

    return num_experts_per_wave;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // Amax reduction
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = sf_block_m * 4;
    const int smem_sfb_per_stage = sf_block_n * 4;

    // Per-stage: A tile + B tile + SFA tile + SFB tile + full/empty barriers
    const int smem_per_stage = load_block_m * block_k + block_n * block_k + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // Select maximum num_stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden) {
    // Block tiling
    const int block_m = get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const int block_n = 128;
    const int block_k = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int store_block_m = 32;
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
    const int num_padded_sf_pool_tokens = layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // Waves
    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;
    const int num_epilogue_threads = 256;

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, store_block_m,
        sf_block_m, sf_block_n,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/runtime.hpp': r"""#pragma once

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/lazy_init.hpp"

namespace deep_gemm {

class HeuristicsRuntime {
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

public:
    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
        if (device_runtime->get_arch_major() != 10)
            return kLegacyMKAlignmentForContiguousLayout;

        int block_m = 240, mma_step = 16;
        if (expected_m.has_value()) {
            // Reduce `block_m` while ensuring it covers `m`
            for (; block_m > 32 and block_m - mma_step >= expected_m.value(); block_m -= mma_step);
        }
        return block_m;
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/sm100.hpp': r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "runtime.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16: return {0, 0};
            case MmaKind::MXFP8FP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block K is always in a fixed manner
        const int block_k = 128 / get_element_size(desc.get_mma_kind());

        // Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout or
            desc.gemm_type == GemmType::MGroupedMasked) {
            const bool swap_ab = true;
            const auto block_n = 128;
            const auto block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
            const auto cluster_m = 1;
            const auto cluster_n = ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 ? 2 : 1;
            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};
            std::vector<Layout> candidates = {layout};
            return candidates;
        }

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int swap_ab = 0; swap_ab < 2; ++ swap_ab) {
            // Block M/N candidates
            std::vector<int> block_m_candidates;
            std::vector<int> block_n_candidates;            
            if (swap_ab) {
                int step = std::lcm(16, heuristics_runtime->get_block_m_multiple_of());
                int end = 256;
                for (int i = step; i <= end; i += step)
                    block_m_candidates.push_back(i);

                // TODO: consider other block N
                block_n_candidates = {128};
            } else {
                // NOTES: smaller block M can avoid TMA L2 OOB bound
                // TODO: consider block M = 256
                if (desc.m <= 32) block_m_candidates = {32};
                else if (desc.m <= 64) block_m_candidates = {64};
                else block_m_candidates = {128};

                // Small block size for small shape
                if (16 % heuristics_runtime->get_block_n_multiple_of() == 0)
                    block_n_candidates.push_back(16);
                int step = std::lcm(32, heuristics_runtime->get_block_n_multiple_of());
                // For small K, fewer store blocks improve store/compute overlap and reduce epilogue bottleneck
                int end = desc.k <= 256 ? 128 : 256;
                for (int i = step; i <= end; i += step)
                    block_n_candidates.push_back(i);
            }

            for (int cluster_m = 1; cluster_m <= 2; ++ cluster_m) {
                // After swapping, layout A/D can only do on cluster N
                if (swap_ab == 1 and cluster_m > 1)
                    continue;

                for (int cluster_n = 1; cluster_n <= 2; ++ cluster_n) {
                    // We only support cluster 2
                    if (cluster_m * cluster_n > 2)
                        continue;

                    // Only support layout A/D
                    if (swap_ab == 0 and cluster_n > 1)
                        continue;

                    // SM count must be divisible
                    if (desc.num_sms % (cluster_m * cluster_n) != 0)
                        continue;

                    for (int block_m: block_m_candidates) {
                        // Ensure large swizzle sizes (32B swizzle yields poor performance)
                        const auto swizzle_a_requirement = desc.a_dtype == kPackedFP4 ? 128 : 64;
                        // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                        const auto load_block_m_requirement = desc.major_a == cute::UMMA::Major::MN ? swizzle_a_requirement : 8;
                        if ((block_m / cluster_n) % load_block_m_requirement != 0)
                            continue;

                        // Shape must be divisible for multicast
                        if (ceil_div(desc.m, block_m) % cluster_m != 0)
                            continue;

                        for (int block_n: block_n_candidates) {
                            // Ensure large swizzle sizes (32B swizzle yields poor performance)
                            const auto swizzle_b_requirement = desc.b_dtype == kPackedFP4 ? 128 : 64;
                            // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                            const auto load_block_n_requirement = desc.major_b == cute::UMMA::Major::MN ? swizzle_b_requirement : 8;
                            if ((block_n / cluster_m) % load_block_n_requirement != 0)
                                continue;

                            // Shape must be divisible for multicast
                            if (ceil_div(desc.n, block_n) % cluster_n != 0)
                                continue;

                            // SwapAB requires block N is layout A/D' UMMA M
                            constexpr int layout_ad_m = 128;
                            if (swap_ab and block_n != layout_ad_m)
                                continue;

                            // Check tensor memory capacity
                            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, desc.get_mma_kind());
                            const auto tmem_sf_cols = desc.get_mma_kind() == MmaKind::MXFP8FP4 ? sf_block_m / 32 + sf_block_n / 32 : 0;
                            const auto umma_n = swap_ab ? block_m : block_n;
                            if (2 * umma_n + tmem_sf_cols > 512)
                                continue;

                            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};

                            // When neither A nor B is MN major, 128B swizzle is always feasible
                            if (desc.major_a == cute::UMMA::Major::K or desc.major_b == cute::UMMA::Major::K) {
                                const auto storage_config = get_storage_config(desc, layout);
                                if (storage_config.swizzle_a_mode != 128 or storage_config.swizzle_b_mode != 128)
                                    continue;
                            }

                            candidates.push_back(layout);
                        }
                    }
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int layout_ad_m = 128;
        constexpr int umma_step_n = 16;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        const auto load_block_m = layout.block_m / layout.cluster_n;
        const auto load_block_n = layout.block_n / layout.cluster_m;
        const auto store_block_m = layout.swap_ab ? umma_step_n : std::min(layout_ad_m, layout.block_m);
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        // TODO: support FP4 sub-byte
        const auto swizzle_mode_a = get_swizzle_mode(
            desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(
            desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n, c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_cd = get_swizzle_mode(
            store_block_n, c10::elementSize(desc.cd_dtype));

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 32;

        // C/D for TMA stores
        const int smem_cd = layout.swap_ab ? storage_config.store_block_m * storage_config.store_block_n * c10::elementSize(desc.cd_dtype) * 2
                                           : storage_config.store_block_m * storage_config.swizzle_cd_mode * 2;

        // TODO: remove SF barriers for BF16 GEMMs
        // TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        // NOTES: the last barrier is for tensor core utilization control
        const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 2 + 8;

        // Tensor memory pointer
        const int smem_tmem_ptr = 4;

        // Calculate A/B per stages
        // TODO: consider FP4
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // Calculate SF A/B per stages
        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(
                layout.block_m, layout.block_n, desc.get_mma_kind());
            smem_sfa_per_stage = sf_block_m * 4;
            smem_sfb_per_stage = sf_block_n * 4;
        }

        // Calculate stages
        int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
        int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            256,
            32, 128, 128, 128
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;
        // TODO: calculate expected cycles
        return {num_waves, last_wave_util, 0, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Single wave is always better
        if ((a.num_waves == 1 or b.num_waves == 1) and a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Doing multicast is better
        if (a.layout.get_cluster_size() != b.layout.get_cluster_size())
            return a.layout.get_cluster_size() > b.layout.get_cluster_size();

        // Smaller number of waves is better
        if (a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Larger last wave utilization is better
        if (a.last_wave_util != b.last_wave_util)
            return a.last_wave_util > b.last_wave_util;

        // More stages is better
        // Same block M, smaller block N is better
        // Same block N, smaller block M is better
        if (a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n)
            return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;

        // Less shared memory C/D, more stages is better
        return a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n;
    }
};

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/sm90.hpp': r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM90ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block M candidates
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::Normal or
            desc.gemm_type == GemmType::Batched or
            desc.gemm_type == GemmType::KGroupedContiguous) {
            // TODO: check 256's performance
            block_m_candidates = {64, 128};
            // NOTES: smaller block M can avoid TMA L2 OOB bound
            if (desc.m <= 16) block_m_candidates.push_back(16);
            if (desc.m <= 32) block_m_candidates.push_back(32);

            // BF16 output GEMM supports 256
            if (desc.cd_dtype != torch::kFloat)
                block_m_candidates.push_back(256);
        } else if (desc.gemm_type == GemmType::MGroupedContiguous or
                   desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = std::vector{heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else if (desc.gemm_type == GemmType::MGroupedMasked) {
            block_m_candidates = {64, 128};
        }

        // Block N candidates
        std::vector<int> block_n_candidates;
        int step = std::lcm(16, heuristics_runtime->get_block_n_multiple_of());
        int start = step;
        // Avoid bank conflicts for 1D1D kernel FP32 output
        if (desc.kernel_type == KernelType::Kernel1D1D and desc.cd_dtype == torch::kFloat) {
            DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K);
            DG_HOST_ASSERT(desc.major_b == cute::UMMA::Major::K);
            start = 24;
            block_n_candidates.push_back(16);
        }
        // Register spills
        int end = 256;
        if (desc.kernel_type == KernelType::Kernel1D2D)
            end = 192;
        if (desc.kernel_type == KernelType::Kernel1D1D)
            end = 160;
        // Enumerate
        for (int i = start; i <= end; i += step)
            block_n_candidates.push_back(i);

        // Block K is always in a fixed manner
        const int block_k = 128 / get_element_size(desc.get_mma_kind());

        // Disable multicast for performance
        const bool disable_multicast =
            // The number of k-groups is large (a heuristic)
            (desc.gemm_type == GemmType::KGroupedContiguous and desc.num_groups > 4) or
            // Not supported
            (desc.gemm_type == GemmType::Batched);

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int cluster_m = 1; cluster_m <= (disable_multicast ? 1 : 2); ++ cluster_m) {
            for (int cluster_n = 1; cluster_n <= (disable_multicast ? 1 : 2); ++ cluster_n) {
                // We only support cluster 2
                if (cluster_m * cluster_n > 2)
                    continue;

                // SM count must be divisible
                if (desc.num_sms % (cluster_m * cluster_n) != 0)
                    continue;

                for (int block_m: block_m_candidates) {
                    for (int block_n: block_n_candidates) {
                        // 1D2D kernel unroll requirement
                        if (desc.kernel_type == KernelType::Kernel1D2D and block_n > block_k and (block_n % (block_n - block_k) != 0 and block_k % (block_n - block_k) != 0))
                            continue;

                        // Multicast legality for masked layout
                        // TODO: add some comments about it
                        if ((desc.gemm_type == GemmType::MGroupedMasked or desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) and
                            ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0)
                            continue;

                        // The block sizes cannot be too large (for enough registers), so at least one dim less than 128
                        if (block_m > 128 and block_n > 128)
                            continue;

                        // Calculate swizzling
                        const auto layout = Layout{0, block_m, block_n, block_k, cluster_m, cluster_n};
                        const auto storage_config = get_storage_config(desc, layout);

                        // Make sure swizzling is large enough (32B's performance is low)
                        if (storage_config.swizzle_a_mode % 64 != 0 or storage_config.swizzle_b_mode % 64 != 0)
                            continue;
                        
                        // To hide TMA latency, the stage count should be at least 3; for small matrices, at least 4
                        int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
                        if (num_stages < 3 or (block_m * block_n < 128 * 192 and num_stages < 4))
                            continue;

                        candidates.push_back(layout);
                    }
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int wgmma_m = 64;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        // TODO: support swap AB
        DG_HOST_ASSERT(layout.swap_ab == 0);
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;
        // 1D1D kernel will do single warp-group stores
        const auto store_block_m = desc.kernel_type == KernelType::Kernel1D1D ? wgmma_m : layout.block_m;
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        const auto swizzle_mode_a = get_swizzle_mode(
            desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(
            desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n, c10::elementSize(desc.b_dtype));
        // We only enable swizzling for non-FP32 outputs
        const auto swizzle_mode_cd = desc.cd_dtype != torch::kFloat ?
            get_swizzle_mode(store_block_n, c10::elementSize(desc.cd_dtype)) : 0;

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        // TODO: consider swap AB
        // C/D for TMA stores
        // NOTES: 1024 is for TMA swizzling alignment requirement
        const int smem_cd =
            align(layout.block_m * layout.block_n * static_cast<int>(c10::elementSize(desc.cd_dtype)), 1024);
        const int smem_barriers = kNumMaxStages * 8 * 2;

        // Calculate A/B per stages
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // Calculate SF A/B per stages
        const int smem_sfa_per_stage = desc.kernel_type == KernelType::KernelNoSF ?
            0 : align(layout.block_m * static_cast<int>(sizeof(float)), 128);
        const int smem_sfb_per_stage = desc.kernel_type != KernelType::Kernel1D1D ?
            0 : align(layout.block_n * static_cast<int>(sizeof(float)), 128);

        // Extra SFB sizes for 1D2D kernels
        const int use_uniform_sfb = layout.block_k % layout.block_n == 0 ? 1 : 2;
        const int smem_extra_sfb = desc.kernel_type != KernelType::Kernel1D2D ?
            0 : align<int>(ceil_div(desc.k, layout.block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);

        // Extra tensormap for 1D1D kernels
        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        // Calculate stages
        const int smem_extra = smem_cd + smem_barriers + smem_extra_sfb + smem_tensormap;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = 128;
        const int num_math_threads = layout.block_m <= 64 ? 128 : 256;
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0 // Meaningless for SM90
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // Utils
        const int l2_bandwidth_per_cycle = std::min(64. * desc.num_sms, 8e6 / (1.3e3)); // B/cycle
        const int l1_bandwidth_per_cycle = 128 * desc.num_sms; // B/cycle
        const int wgmma_m = 64;
        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int elem_size_cd = c10::elementSize(desc.cd_dtype);
        DG_HOST_ASSERT(desc.a_dtype == desc.b_dtype);

        // Data movement per block
        int64_t expected_k = desc.get_expected_k();
        int64_t num_bytes_l2_ab = expected_k * (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m) * elem_size_ab;
        int64_t num_bytes_l1_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        int64_t num_bytes_l1_tc = expected_k * (std::max(wgmma_m, layout.block_m) + layout.block_n) * elem_size_ab
                                  + layout.block_m * layout.block_n * elem_size_cd;
        int64_t num_bytes_l1_l2_cd = layout.block_m * layout.block_n * elem_size_cd * (desc.with_accumulation ? 2 : 1);

        // HBM bandwidth and total compute (Tensor/CUDA cores) are constant across configs
        // We only model L1/L2 cycles as they are the primary variables between configs
        int64_t num_l2_cycles = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bandwidth_per_cycle;
        int64_t num_l1_cycles = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bandwidth_per_cycle;
        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency;

        // Disable multicasting if only one wave exists
        if (layout.cluster_n * layout.cluster_m > 1 and num_waves <= 1)
            num_cycles = std::numeric_limits<int64_t>::max();

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/heuristics/utils.hpp': r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/epilogue.hpp': r"""#pragma once

#include <optional>
#include <string>

namespace deep_gemm {

static std::string get_default_epilogue_type(const std::optional<std::string>& epilogue_type) {
    return epilogue_type.value_or("epilogue::transform::EpilogueIdentity");
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/runtime_utils.hpp': r"""#pragma once

#include <cuda.h>
#include <torch/python.h>

#include "../heuristics/sm90.hpp"
#include "../../jit/handle.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

static std::pair<int, int> get_inner_outer_dims(const cute::UMMA::Major& major, const int& k, const int& mn) {
    return major == cute::UMMA::Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

static int get_non_contiguous_dim(const cute::UMMA::Major& major) {
    return major == cute::UMMA::Major::K ? -2 : -1;
}

static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    if (heuristics_runtime->get_ignore_compile_dims())
        return 0;

    for (const char& c: compiled_dims) {
        if (name == c)
            return dim;
    }
    return 0;
}

static std::string to_string(const cute::UMMA::Major& major) {
    switch (major) {
        case cute::UMMA::Major::K:  return "cute::UMMA::Major::K";
        case cute::UMMA::Major::MN: return "cute::UMMA::Major::MN";
    }
    DG_HOST_UNREACHABLE("Unknown major");
}

static std::string to_string(const GemmType& type) {
    switch (type) {
        case GemmType::Normal:                              return "GemmType::Normal";
        case GemmType::MGroupedContiguous:                  return "GemmType::MGroupedContiguous";
        case GemmType::MGroupedMasked:                      return "GemmType::MGroupedMasked";
        case GemmType::MGroupedContiguousWithPsumLayout:    return "GemmType::MGroupedContiguousWithPsumLayout";
        case GemmType::KGroupedContiguous:                  return "GemmType::KGroupedContiguous";
        case GemmType::Batched:                             return "GemmType::Batched";
    }
    DG_HOST_UNREACHABLE("Unknown GEMM type");
}

static std::string to_string(const at::ScalarType& dtype) {
    switch (dtype) {
        case torch::kInt:           return "int";
        case torch::kFloat:         return "float";
        case torch::kBFloat16:      return "cutlass::bfloat16_t";
        case torch::kFloat8_e4m3fn: return "cutlass::float_e4m3_t";
        case kPackedFP4:            return "cutlass::detail::float_e2m1_unpacksmem_t";
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static std::string to_string(const float& v) {
    if (std::isfinite(v)) {
        return fmt::format(R"({:a}f)", v);
    } else if (std::isinf(v)) {
        return v > 0 ? "cute::numeric_limits<float>::infinity()"
                     : "-cute::numeric_limits<float>::infinity()";
    }
    DG_HOST_UNREACHABLE("NaN input is not supported");
}

static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype,
                                                          const bool& allow_tf32,
                                                          const bool& fp4_unpacked_smem) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;

    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
#if CUDA_VERSION >= 12080
        case kPackedFP4:            return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                                             : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#endif
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode, const int& base) {
#if CUDA_VERSION >= 12080
    if (base != 0) {
        DG_HOST_ASSERT(base == 32 and mode == 128);
        return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }
#endif

    DG_HOST_ASSERT(base == 0);
    switch (mode) {
        case   0:
        case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: DG_HOST_UNREACHABLE("Unsupported swizzling mode");
    }
}

static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim,
                                    const int& gmem_outer_stride,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);

        // Fix FP4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_inner_dim = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    if (get_env<int>("DG_JIT_DEBUG")) {
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d (base: %d), elem size: %d, pointer: %llu\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride, swizzle_mode, swizzle_base, elem_size,
               reinterpret_cast<unsigned long long>(t.data_ptr()));
    }
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_tma_3d_desc(const torch::Tensor& t,
                                    int gmem_dim_0, int gmem_dim_1, int gmem_dim_2,
                                    int smem_dim_0, int smem_dim_1, int smem_dim_2,
                                    const int& gmem_stride_0, const int& gmem_stride_1,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false,
                                    const bool& fp4_unpacked_smem = true) {
    const auto elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0)
        smem_dim_0 = swizzle_mode / elem_size;

    if (t.scalar_type() == kPackedFP4) {
        // Inner dim must be a multiple of 64B for .b4x16_p64
        DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_dim_0 % 128 == 0);

        // Fix fp4 packed smem
        if (not fp4_unpacked_smem and swizzle_mode != 0)
            smem_dim_0 = swizzle_mode * 2;
    }

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1), static_cast<cuuint64_t>(gmem_dim_2),};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1), static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_0 * elem_size), static_cast<cuuint64_t>(gmem_stride_1 * elem_size)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    if (get_env<int>("DG_JIT_DEBUG")) {
        printf("Making 3D TMA desc: global memory: %d %d %d, shared memory: %d %d %d, outer stride: %d %d, swizzle: %d, elem size: %d\n",
               gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2,
               gmem_stride_0, gmem_stride_1, swizzle_mode, elem_size);
    }
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32, fp4_unpacked_smem),
        3, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_tma_a_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_b_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_n, const int& shape_k,
                                   const int& block_n, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false) {
    const auto [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);

    // `num_groups` is always applied into the outer dimensions
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim * num_groups,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
                                    const int& shape_m, const int& shape_n,
                                    const int& block_m, const int& block_n,
                                    const int& outer_stride,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    // Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    // bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, block_m,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_sf_desc(const cute::UMMA::Major& major,
                                    const torch::Tensor& t,
                                    int shape_mn, int shape_k,
                                    const int& block_mn, const int& gran_k,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, gran_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                            block_mn, 1,
                            shape_mn,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BF16GemmRuntime final: public LaunchRuntime<SM100BF16GemmRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation, to_string(args.gemm_desc.cd_dtype),
        args.gemm_desc.tc_util);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void sm100_bf16_gemm(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& c,
                            const torch::Tensor& d,
                            const int& m, const int& n, const int& k,
                            const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                            const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM100BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

static void sm100_m_grouped_bf16_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const bool& use_psum_layout,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    // NOTES: If actual M is dynamic, estimate config via `num_groups` and `expected_m`.
    //        Otherwise, treat the contiguous layout as a whole.
    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM100BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_m_grouped_gemm_contiguous", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

static void sm100_m_grouped_bf16_gemm_masked(const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& num_groups, const int& m, const int& n, const int& k,
                                             const int& expected_m,
                                             const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                             const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM100BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_m_grouped_gemm_masked", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

static void sm100_bf16_k_grouped_gemm(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const std::optional<torch::Tensor>& c,
                                      const torch::Tensor& d,
                                      const int& m, const int& n,
                                      const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                      const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                      const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    int sum_k = 0;
    for (const auto k: ks) {
        sum_k += k;
        DG_HOST_ASSERT(k % 128 == 0);
    }
    const auto num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto max_k = *std::max_element(ks.begin(), ks.end());
    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(1)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Launch kernel
    const SM100BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_k_grouped_gemm", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

static void sm100_bf16_bhr_hdr_bhd(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims = "nk") {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = d, .k = r, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(tensor_a, r, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.layout.block_k, config.storage_config.load_block_n, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, d, b, h,
                                                config.storage_config.store_block_n, config.storage_config.store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM100BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_bhr_hdr_bhd", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

static void sm100_bf16_bhd_hdr_bhr(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims = "nk") {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = r, .k = d, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::MN,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(tensor_a, d, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.storage_config.load_block_n, config.layout.block_k, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, r, b, h,
                                                config.storage_config.store_block_n, config.storage_config.store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM100BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_bhd_hdr_bhr", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BmkBnkMnRuntime final: public LaunchRuntime<SM100BmkBnkMnRuntime> {
public:
    struct Args {
        int s, m, n, k;
        int block_m, block_n, block_k;
        int split_factor;
        int swizzle_ab_mode, swizzle_cd_mode;
        int num_stages;
        int num_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bmk_bnk_mn.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bmn_bnk_mn_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {}
    >);
}};
)",
        args.m, args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.split_factor,
        args.swizzle_ab_mode, args.swizzle_cd_mode,
        args.num_stages, args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.s, args.tensor_map_a, args.tensor_map_b, args.tensor_map_d));
    }
};


static void sm100_bmn_bnk_mn_gemm(const torch::Tensor &a,
                                  const torch::Tensor &b,
                                  const torch::Tensor &d,
                                  const int &s, const int &m, const int &n, const int &k) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int num_threads = 128;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);
    DG_HOST_ASSERT(static_cast<int64_t>(s) * static_cast<int64_t>(std::max(m, n)) <= std::numeric_limits<int>::max());

    const int swizzle_ab_mode = get_swizzle_mode(block_k, static_cast<int>(a.element_size()));
    const int swizzle_cd_mode = get_swizzle_mode(block_n, static_cast<int>(d.element_size()));

    // Get best config
    const int num_sms = device_runtime->get_num_sms();
    const int num_mn_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
    const int num_sk_blocks = s * (k / block_k);
    const int split_factor = ceil_div(num_sk_blocks, std::max(num_sms / num_mn_blocks, 1));

    // Select best number of stages
    // NOTES: we select 4 as start, as it is tested to be faster than values > 4
    int num_stages = 4, smem_size = 0;
    while (true) {
        const int smem_cd = block_m * swizzle_cd_mode * 2;
        const int smem_a_per_stage = block_m * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_b_per_stage = block_n * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_barrier = num_stages * 8 * 3 + 2 * 8 * 2 + 8;
        const int smem_tmem_ptr = 4;

        smem_size = 0;
        smem_size += smem_cd;
        smem_size += (smem_a_per_stage + smem_b_per_stage) * num_stages;
        smem_size += smem_barrier;
        smem_size += smem_tmem_ptr;
        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;

        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("S: %d, M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split-K factor: %d"
               "stages: %d, shared memory: %d, swizzle AB: %d, swizzle CD: %d\n",
               s, m, n, k, block_m, block_n, block_k, split_factor,
               num_stages, smem_size, swizzle_ab_mode, swizzle_cd_mode);
    }

    const auto tensor_map_a = make_tma_2d_desc(a, k, s * m, block_k, block_m, k, swizzle_ab_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, s * n, block_k, block_n, k, swizzle_ab_mode);
    const auto tensor_map_d = make_tma_2d_desc(d, n, m, block_n, block_m, n, swizzle_cd_mode);

    const SM100BmkBnkMnRuntime::Args args = {
        .s = s, .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .split_factor = split_factor,
        .swizzle_ab_mode = swizzle_ab_mode,
        .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_threads = num_threads,
        .launch_args = LaunchArgs(num_mn_blocks * ceil_div(num_sk_blocks, split_factor), num_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100BmkBnkMnRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bmn_bnk_mn_gemm", code);
    SM100BmkBnkMnRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP8FP4Gemm1D1DRuntime final: public LaunchRuntime<SM100FP8FP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        // TODO: move into descriptor
        const std::optional<std::string> epilogue_type;

        // TODO: move into descriptor
        int gran_k_a, gran_k_b;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: rename files
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {}, {}, {},
        {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        args.gran_k_a, args.gran_k_b,
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.a_dtype), to_string(args.gemm_desc.b_dtype), to_string(args.gemm_desc.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims,
                                    const std::optional<std::string>& epilogue_type = std::nullopt) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto cd = c.value_or(d);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, 1, 0);

    // Launch
    const SM100FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = epilogue_type,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& grouped_layout,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const std::optional<int>& expected_m_for_psum_layout) {
    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    // NOTES: If actual M is dynamic, estimate config via `num_groups` and `expected_m`.
    //        Otherwise, treat the contiguous layout as a whole.
    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, num_groups, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& masked_m,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const int& expected_m,
                                                     const int& gran_k_a, const int& gran_k_b,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, num_groups, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, num_groups, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp8_fp4_gemm_masked_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_k_grouped_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                          const torch::Tensor& b, const torch::Tensor& sfb,
                                          const std::optional<torch::Tensor>& c,
                                          const torch::Tensor& d,
                                          const int& m, const int& n,
                                          const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                          const int& gran_k,
                                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                          const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    const int gran_k_a = gran_k;
    const int gran_k_b = gran_k;

    int sum_k = 0, sum_sf_k = 0;
    for (const auto k: ks) {
        sum_k += k, sum_sf_k += ceil_div(k, gran_k * 4);
        DG_HOST_ASSERT(k % gran_k == 0);
    }
    const auto num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto max_k = *std::max_element(ks.begin(), ks.end());
    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(1)), num_groups,
                                                config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * gran_k_a * 4,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * gran_k_b * 4,
                                                 config.layout.block_n, gran_k_b, 1, 0);

    // Launch kernel
    const SM100FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_k_grouped_fp8_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                          const torch::Tensor& b, const torch::Tensor& sfb,
                          const std::optional<torch::Tensor>& c,
                          const torch::Tensor& d,
                          const int& batch_size, const int& m, const int& n, const int& k,
                          const int& gran_k_a, const int& gran_k_b,
                          const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                          const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const int load_block_m = config.storage_config.load_block_m;
    const auto [inner_dim_a, outer_dim_a] = get_inner_outer_dims(major_a, k, m);
    const auto [inner_block_a, outer_block_a] = get_inner_outer_dims(major_a, config.layout.block_k, load_block_m);
    const auto tensor_map_a = make_tma_3d_desc(a, inner_dim_a, outer_dim_a, batch_size,
                                               inner_block_a, outer_block_a, 1,
                                               a.stride(major_a == cute::UMMA::Major::K ? 1 : 2),
                                               a.stride(0),
                                               config.storage_config.swizzle_a_mode);

    const int load_block_n = config.storage_config.load_block_n;
    const auto [inner_dim_b, outer_dim_b] = get_inner_outer_dims(major_b, k, n);
    const auto [inner_block_b, outer_block_b] = get_inner_outer_dims(major_b, config.layout.block_k, load_block_n);
    const auto tensor_map_b = make_tma_3d_desc(b, inner_dim_b, outer_dim_b, batch_size,
                                               inner_block_b, outer_block_b, 1,
                                               b.stride(major_b == cute::UMMA::Major::K ? 1 : 2),
                                               b.stride(0),
                                               config.storage_config.swizzle_b_mode);

    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, batch_size,
                                                store_block_n, store_block_m, 1,
                                                d.stride(1), d.stride(0),
                                                config.storage_config.swizzle_cd_mode);

    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, batch_size, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, batch_size, 0);

    // Launch
    const SM100FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_gemm_1d1d", code);
    SM100FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden);

    // Make tensormap
    constexpr int kGranK = 32;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100BF16HCPrenormGemmRuntime final: public LaunchRuntime<SM100BF16HCPrenormGemmRuntime> {
public:
    struct Args {
        int m, n, k;
        int block_m, block_n, block_k;
        int num_splits;
        int swizzle_cd_mode;
        int num_stages;
        int num_mma_threads, num_cast_and_reduce_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        float* sqr_sum;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}
    >);
}};
)",
        args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.num_splits,
        args.swizzle_cd_mode,
        args.num_stages,
        args.num_mma_threads, args.num_cast_and_reduce_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.m, args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.sqr_sum));
    }
};

static void sm100_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& sqr_sum,
                                       const int& m, const int& n, const int& k,
                                       const int& num_splits) {
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    constexpr int num_mma_threads = 128;
    constexpr int num_cast_and_reduce_threads = 128;

    const int block_n = align(n, 16);
    DG_HOST_ASSERT(n <= block_n);
    DG_HOST_ASSERT(n <= 128 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    const auto swizzle_cd_mode = get_swizzle_mode(block_n, sizeof(float));
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, a.element_size()), 0,
                                              true);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              block_n, block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, b.element_size()), 0,
                                              true);
    const auto tensor_map_d = num_splits == 1 ? make_tma_cd_desc(d, m, n,
                                                                 block_m, block_n,
                                                                 static_cast<int>(d.stride(-2)), 1,
                                                                 swizzle_cd_mode)
                                               : make_tma_3d_desc(d, n, m, num_splits,
                                                                  block_n, block_m, 1,
                                                                  static_cast<int>(d.stride(-2)),
                                                                  static_cast<int>(d.stride(-3)),
                                                                  swizzle_cd_mode);

    // Calculate stages
    int num_stages = 12, smem_size = 0;
    while (num_stages > 0) {
        const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_b_per_stage = block_n * block_k * static_cast<int>(sizeof(float));
        const int smem_cd = block_m * swizzle_cd_mode;
        const int smem_barriers = (num_stages * 4 + 1) * 8;
        const int smem_tmem_ptr = 4;
        smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages +
                    smem_cd + smem_barriers + smem_tmem_ptr;

        if (smem_size <= SM100ArchSpec::smem_capacity)
            break;
        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split K: %d"
               "stages: %d, shared memory: %d, swizzle CD: %d\n",
               m, n, k, block_m, block_n, block_k, num_splits,
               num_stages, smem_size, swizzle_cd_mode);
    }

    // Launch
    const SM100BF16HCPrenormGemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_splits = num_splits,
        .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_mma_threads = num_mma_threads,
        .num_cast_and_reduce_threads = num_cast_and_reduce_threads,
        .launch_args = LaunchArgs(num_splits * ceil_div(m, block_m), num_mma_threads + num_cast_and_reduce_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .sqr_sum = sqr_sum.data_ptr<float>()
    };
    const auto code = SM100BF16HCPrenormGemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_tf32_hc_prenorm_gemm", code);
    SM100BF16HCPrenormGemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90BF16GemmRuntime final: public LaunchRuntime<SM90BF16GemmRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
        // TODO: add CD dtype
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode,
        args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        // TODO: refactor with cluster M/N
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void sm90_bf16_gemm(const torch::Tensor& a,
                           const torch::Tensor& b,
                           const std::optional<torch::Tensor>& c,
                           const torch::Tensor& d,
                           const int& m, const int& n, const int& k,
                           const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                           const std::string& compiled_dims) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_gemm", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_m_grouped_bf16_gemm_contiguous(const torch::Tensor& a,
                                                const torch::Tensor& b,
                                                const torch::Tensor& d,
                                                const torch::Tensor& m_indices,
                                                const int& num_groups, const int& m, const int& n, const int& k,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                const std::string& compiled_dims,
                                                const bool& use_psum_layout,
                                                const std::optional<int>& expected_m_for_psum_layout) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(k % 64 == 0);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_bf16_gemm_contiguous", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_m_grouped_gemm_masked(const torch::Tensor& a,
                                            const torch::Tensor& b,
                                            const torch::Tensor& d,
                                            const torch::Tensor& masked_m,
                                            const int& num_groups, const int& m, const int& n, const int& k,
                                            const int& expected_m,
                                            const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                            const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(k % 64 == 0);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = 0, .expected_k = 0, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_m_grouped_gemm_masked", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_k_grouped_gemm(const torch::Tensor& a,
                                     const torch::Tensor& b,
                                     const std::optional<torch::Tensor>& c,
                                     const torch::Tensor& d,
                                     const int& m, const int& n,
                                     const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                     const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    int sum_k = 0;
    for (const auto k: ks) {
        sum_k += k;
        DG_HOST_ASSERT(k % 128 == 0);
    }
    const auto num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto max_k = *std::max_element(ks.begin(), ks.end());
    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Create tensor descriptors
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(1)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Launch kernel
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_k_grouped_gemm", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_bhr_hdr_bhd(const torch::Tensor& tensor_a,
                                  const torch::Tensor& tensor_b,
                                  const torch::Tensor& tensor_d,
                                  const int& b, const int& h, const int& r, const int& d,
                                  const std::string& compiled_dims = "nk") {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = d, .k = r, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    const int load_block_m = config.storage_config.load_block_m;
    const auto tensor_map_a = make_tma_3d_desc(tensor_a, r, b, h,
                                               config.layout.block_k, load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const int load_block_n = config.storage_config.load_block_n;
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.layout.block_k, load_block_n, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, d, b, h,
                                                store_block_n, store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_bhr_hdr_bhd", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

static void sm90_bf16_bhd_hdr_bhr(const torch::Tensor& tensor_a,
                                  const torch::Tensor& tensor_b,
                                  const torch::Tensor& tensor_d,
                                  const int& b, const int& h, const int& r, const int& d,
                                  const std::string& compiled_dims = "nk") {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = r, .k = d, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::MN,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    const int load_block_m = config.storage_config.load_block_m;
    const auto tensor_map_a = make_tma_3d_desc(tensor_a, d, b, h,
                                               config.layout.block_k, load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const int load_block_n = config.storage_config.load_block_n;
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               load_block_n, config.layout.block_k, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, r, b, h,
                                                store_block_n, store_block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);
    // Launch
    const SM90BF16GemmRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bf16_bhd_hdr_bhr", code);
    SM90BF16GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90BmkBnkMnRuntime final: public LaunchRuntime<SM90BmkBnkMnRuntime> {
public:
    struct Args {
        int s, m, n, k;
        int block_m, block_n, block_k;
        int split_factor;
        int num_stages;
        int num_tma_threads, num_math_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        float* d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bmk_bnk_mn.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bmn_bnk_mn_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {},
        {}, {}
    >);
}};
)",
        args.m, args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.split_factor,
        args.num_stages,
        args.num_tma_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.s, args.tensor_map_a, args.tensor_map_b, args.d));
    }
};


static void sm90_bmn_bnk_mn_gemm(const torch::Tensor &a,
                                 const torch::Tensor &b,
                                 const torch::Tensor &d,
                                 const int &s, const int &m, const int &n, const int &k) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);
    DG_HOST_ASSERT(static_cast<int64_t>(s) * static_cast<int64_t>(std::max(m, n)) <= std::numeric_limits<int>::max());

    const int swizzle_ab_mode = get_swizzle_mode(block_k, static_cast<int>(a.element_size()));
    DG_HOST_ASSERT(swizzle_ab_mode == 128);

    // Get best config
    const int num_sms = device_runtime->get_num_sms();
    const int num_mn_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
    const int num_sk_blocks = s * (k / block_k);
    const int split_factor = ceil_div(num_sk_blocks, std::max(num_sms / num_mn_blocks, 1));

    // Select best number of stages
    int num_stages = 4, smem_size = 0;
    while (true) {
        const int smem_a_per_stage = block_m * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_b_per_stage = block_n * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_barrier = num_stages * 8 * 2;

        smem_size = 0;
        smem_size += (smem_a_per_stage + smem_b_per_stage) * num_stages;
        smem_size += smem_barrier;

        if (smem_size <= SM90ArchSpec::smem_capacity)
            break;

        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("S: %d, M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split-K factor: %d"
               "stages: %d, shared memory: %d, swizzle AB: %d\n",
               s, m, n, k, block_m, block_n, block_k, split_factor,
               num_stages, smem_size, swizzle_ab_mode);
    }

    const auto tensor_map_a = make_tma_2d_desc(a, k, s * m, block_k, block_m, k, swizzle_ab_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, s * n, block_k, block_n, k, swizzle_ab_mode);

    const SM90BmkBnkMnRuntime::Args& args = {
        .s = s, .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .split_factor = split_factor,
        .num_stages = num_stages,
        .num_tma_threads = num_tma_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_mn_blocks * ceil_div(num_sk_blocks, split_factor), num_tma_threads + num_math_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .d = d.data_ptr<float>()
    };
    const auto code = SM90BmkBnkMnRuntime::generate(args);
    const auto runtime = compiler->build("sm90_bmn_bnk_mn_gemm", code);
    SM90BmkBnkMnRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D1DRuntime final: public LaunchRuntime<SM90FP8Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *gmem_a_ptr;
        void *gmem_b_ptr;
        void *grouped_layout;
        void *tensor_map_buffer;
        CUtensorMap tensor_map_a_base;
        CUtensorMap tensor_map_b_base;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d1d_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        to_string(args.gemm_desc.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_a_ptr, args.gmem_b_ptr,
            args.grouped_layout,
            args.tensor_map_buffer,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a_base, args.tensor_map_b_base,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm90_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const std::string& compiled_dims) {
    DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k, k, 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k, k, 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, config.layout.block_k, 1, 0);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                0);

    // Launch
    const SM90FP8Gemm1D1DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a_base = tensor_map_a,
        .tensor_map_b_base = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90FP8Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d1d", code);

    SM90FP8Gemm1D1DRuntime::launch(runtime, args);
}

static void sm90_k_grouped_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                         const std::optional<torch::Tensor>& c,
                                         const torch::Tensor& d,
                                         const int& m, const int& n,
                                         const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                         const torch::Tensor& tensor_map_buffer,
                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                         const std::string& compiled_dims) {
    DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // TODO: refactor with the mk alignment function
    const auto num_groups = static_cast<int>(ks.size());
    int first_k = 0, sum_k = 0, sum_sf_k = 0, max_k = 0;
    for (int i = 0; i < num_groups; ++ i) {
        if (first_k == 0 and ks[i] != 0)
            first_k = ks[i];
        sum_k += ks[i], sum_sf_k += ceil_div(ks[i], 128);
        max_k = std::max(max_k, ks[i]);
        DG_HOST_ASSERT(ks[i] % 128 == 0);
    }

    // Get config using max K for better performance
    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

    const auto tensor_map_a_base = make_tma_a_desc(major_a, a, m, first_k,
                                                   config.storage_config.load_block_m,
                                                   config.layout.block_k, first_k, 1,
                                                   config.storage_config.swizzle_a_mode);
    const auto tensor_map_b_base = make_tma_b_desc(major_b, b, n, first_k,
                                                   config.storage_config.load_block_n,
                                                   config.layout.block_k, first_k, 1,
                                                   config.storage_config.swizzle_b_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * 128,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * 128,
                                                 config.layout.block_n, config.layout.block_k, 1, 0);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    // Launch
    const SM90FP8Gemm1D1DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .gmem_a_ptr = a.data_ptr(),
        .gmem_b_ptr = b.data_ptr(),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_buffer = tensor_map_buffer.data_ptr(),
        .tensor_map_a_base = tensor_map_a_base,
        .tensor_map_b_base = tensor_map_b_base,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90FP8Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d1d", code);

    SM90FP8Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D2DRuntime final: public LaunchRuntime<SM90FP8Gemm1D2DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        // TODO: move this into `gemm_desc`
        const std::optional<std::string>& epilogue_type;

        cute::UMMA::Major major_sfb;
        void *sfb, *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
        // TODO: add CD dtype
        to_string(args.major_sfb),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sfb, args.grouped_layout,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_d, args.tensor_map_sfa));
    }
};

static void sm90_fp8_gemm_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                               const std::string& compiled_dims,
                               const std::optional<std::string>& epilogue_type = std::nullopt) {
    DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = epilogue_type,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_gemm_contiguous_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                                    const torch::Tensor& d,
                                                    const torch::Tensor& m_indices,
                                                    const int& num_groups, const int& m, const int& n, const int& k,
                                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                    const std::string& compiled_dims,
                                                    const bool& use_psum_layout,
                                                    const std::optional<int>& expected_m_for_psum_layout) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_fp8_gemm_contiguous_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_gemm_masked_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                const torch::Tensor& b, const torch::Tensor& sfb,
                                                const torch::Tensor& d,
                                                const torch::Tensor& masked_m,
                                                const int& num_groups, const int& m, const int& n, const int& k,
                                                const int& expected_m,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, num_groups, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_m_grouped_gemm_masked_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                         const torch::Tensor& b, const torch::Tensor& sfb,
                         const std::optional<torch::Tensor>& c,
                         const torch::Tensor& d,
                         const int& batch_size, const int& m, const int& n, const int& k,
                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                         const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const int load_block_m = config.storage_config.load_block_m;
    const auto tensor_map_a = make_tma_3d_desc(a, k, m, batch_size,
                                               config.layout.block_k, load_block_m, 1,
                                               a.stride(1),
                                               a.stride(0),
                                               config.storage_config.swizzle_a_mode);

    const int load_block_n = config.storage_config.load_block_n;
    const auto tensor_map_b = make_tma_3d_desc(b, k, n, batch_size,
                                               config.layout.block_k, load_block_n, 1,
                                               b.stride(1),
                                               b.stride(0),
                                               config.storage_config.swizzle_b_mode);

    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_d = make_tma_3d_desc(d, n, m, batch_size,
                                               store_block_n, store_block_m, 1,
                                               d.stride(1), d.stride(0),
                                               config.storage_config.swizzle_cd_mode);

    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, batch_size, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90BF16HCPrenormGemmRuntime final: public LaunchRuntime<SM90BF16HCPrenormGemmRuntime> {
public:
    struct Args {
        int m, n, k;
        int block_m, block_n, block_k;
        int num_splits;
        int swizzle_cd_mode;
        int num_stages;
        int num_math_threads, num_tma_threads;

        LaunchArgs launch_args;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        float* sqr_sum;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}
    >);
}};
)",
        args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.num_splits,
        args.swizzle_cd_mode,
        args.num_stages,
        args.num_math_threads, args.num_tma_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.m, args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.sqr_sum));
    }
};

static void sm90_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const torch::Tensor& d,
                                      const torch::Tensor& sqr_sum,
                                      const int& m, const int& n, const int& k,
                                      const int& num_splits) {
    constexpr int block_m = 64;
    constexpr int block_k = 64;
    constexpr int num_math_threads = 128;
    constexpr int num_tma_threads = 128;
    constexpr int num_threads = num_math_threads + num_tma_threads;

    const int block_n = align(n, 16);
    DG_HOST_ASSERT(n <= block_n);
    // Only support small N for now
    DG_HOST_ASSERT(n <= 32 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    const auto swizzle_cd_mode = get_swizzle_mode(block_n, sizeof(float));
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, a.element_size()), 0,
                                              true);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              block_n, block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(cute::UMMA::Major::K))), 1,
                                              get_swizzle_mode(block_k, b.element_size()), 0,
                                              true);
    const auto tensor_map_d = num_splits == 1 ? make_tma_cd_desc(d, m, n,
                                                                 block_m, block_n,
                                                                 static_cast<int>(d.stride(-2)), 1,
                                                                 swizzle_cd_mode)
                                               : make_tma_3d_desc(d, n, m, num_splits,
                                                                  block_n, block_m, 1,
                                                                  static_cast<int>(d.stride(-2)),
                                                                  static_cast<int>(d.stride(-3)),
                                                                  swizzle_cd_mode);

    // Calculate stages
    int num_stages = 12, smem_size = 0;
    while (num_stages > 0) {
        const int smem_a_per_stage = block_m * block_k * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_b_per_stage = block_n * block_k * static_cast<int>(sizeof(float));
        const int smem_cd = block_m * swizzle_cd_mode;
        const int smem_barriers = num_stages * 2 * 8;
        smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages +
                    smem_cd + smem_barriers;

        if (smem_size <= SM90ArchSpec::smem_capacity)
            break;
        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (get_env("DG_JIT_DEBUG", 0)) {
        printf("M: %d, N: %d, K: %d -> "
               "block M: %d, block N: %d, block K: %d, split K: %d"
               "stages: %d, shared memory: %d, swizzle CD: %d\n",
               m, n, k, block_m, block_n, block_k, num_splits,
               num_stages, smem_size, swizzle_cd_mode);
    }

    smem_size = SM90ArchSpec::smem_capacity;

    // Launch
    const SM90BF16HCPrenormGemmRuntime::Args& args = {
        .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_splits = num_splits,
        .swizzle_cd_mode = swizzle_cd_mode,
        .num_stages = num_stages,
        .num_math_threads = num_math_threads,
        .num_tma_threads = num_tma_threads,
        .launch_args = LaunchArgs(num_splits * ceil_div(m, block_m), num_threads, smem_size),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .sqr_sum = sqr_sum.data_ptr<float>()
    };
    const auto code = SM90BF16HCPrenormGemmRuntime::generate(args);
    const auto runtime = compiler->build("sm90_tf32_hc_prenorm_gemm", code);
    SM90BF16HCPrenormGemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/smxx_clean_logits.hpp': r"""#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

class SMXXCleanLogitsRuntime final: public LaunchRuntime<SMXXCleanLogitsRuntime> {
public:
    struct Args {
        int next_n;
        int seq_len;
        int seq_len_kv;
        uint64_t stride_logits;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;
        at::ScalarType logits_dtype;

        int block_kv;
        int num_warps;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_clean_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<
        {}, {}, {}, {}
    >);
}};
)", args.next_n, args.block_kv, args.num_warps, to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_logits),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end, args.logits
        ));
    }
};

static void smxx_clean_logits(const torch::Tensor& logits,
                              const std::optional<torch::Tensor>& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end,
                              const int& next_n,
                              const int& seq_len, const int& seq_len_kv,
                              const uint64_t &stride_logits) {
    const int block_kv = 8192;
    const int num_warps = 8;
    const int smem_size = block_kv * sizeof(float);

    // Launch
    const SMXXCleanLogitsRuntime::Args& args = {
        .next_n = next_n,
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_logits = stride_logits,
        .cu_seq_len_k_start = cu_seq_len_k_start.has_value() ? cu_seq_len_k_start.value().data_ptr<int>() : nullptr,
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .logits_dtype = logits.scalar_type(),
        .block_kv = block_kv,
        .num_warps = num_warps,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_warps * 32, smem_size)
    };
    const auto code = SMXXCleanLogitsRuntime::generate(args);
    const auto runtime = compiler->build("smxx_clean_logits", code);
    SMXXCleanLogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/smxx_cublaslt.hpp': r"""#pragma once

#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cute/arch/mma_sm100_umma.hpp>

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/compatibility.hpp"

namespace deep_gemm {

static auto get_cublaslt_layout(const cudaDataType& type, const int& rows, const int& cols, const int& ld,
                                const std::optional<int>& batch_count = std::nullopt,
                                const std::optional<int>& batch_offset = std::nullopt) {
    cublasLtMatrixLayout_t layout;
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    if (batch_count.has_value()) {
        DG_HOST_ASSERT(batch_offset.has_value());

        const int64_t batch_offset_int64 = batch_offset.value();
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count.value(), sizeof(batch_count.value())));
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_int64, sizeof(batch_offset_int64)));
    }
    return layout;
}

static void call_cublaslt_api(const cublasOperation_t& trans_a,
                              const cublasOperation_t& trans_b,
                              const cublasLtMatrixLayout_t& layout_a,
                              const cublasLtMatrixLayout_t& layout_b,
                              const cublasLtMatrixLayout_t& layout_d,
                              const torch::Tensor& a,
                              const torch::Tensor& b,
                              const torch::Tensor& d,
                              const bool& accumulate) {
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    cudaDataType_t scale_type = CUDA_R_32F;

    // Operation description
    cublasLtMatmulDesc_t desc;
    DG_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&desc, compute_type, scale_type));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

#if DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE
    const int math_sms = device_runtime->get_num_sms();
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sms, sizeof(math_sms)));
#endif

#if DG_FP8_COMPATIBLE and DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE
    bool fp8_fast_accumulate = false;
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fp8_fast_accumulate, sizeof(fp8_fast_accumulate)));
#endif

    // Get cuBLASLt handle, workspace, and stream
    const auto handle = device_runtime->get_cublaslt_handle();
    const auto workspace = device_runtime->get_cublaslt_workspace();
    const auto workspace_bytes = workspace.nbytes();
    const auto stream = at::cuda::getCurrentCUDAStream();

    // Algorithm selection
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_heuristic_results = 0;
    uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspace_bytes, sizeof(workspace_bytes)));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                           &reduction_scheme_mask, sizeof(reduction_scheme_mask)));
    DG_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_d, layout_d,
                                                     pref, 1, &heuristic, &num_heuristic_results));
    DG_HOST_ASSERT(num_heuristic_results == 1 and "Unable to find any algorithm for the GEMM");

    // Call: D = alpha * (A @ B) + beta * C
    const float alpha = 1.0, beta = accumulate ? 1.0 : 0.0;
    DG_CUBLASLT_CHECK(cublasLtMatmul(handle,                                // Light handle
                                     desc,                                  // Operation description
                                     &alpha,                                // Alpha
                                     b.data_ptr(), layout_a,                // A
                                     a.data_ptr(), layout_b,                // B
                                     &beta,                                 // Beta
                                     d.data_ptr(), layout_d,                // C
                                     d.data_ptr(), layout_d,                // D
                                     &heuristic.algo,                       // Algorithm
                                     workspace.data_ptr(), workspace_bytes, // Workspace
                                     stream));                              // Stream

    // Free memory
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_a));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_b));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_d));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(desc));
}

static void cublaslt_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs,
                          const torch::Tensor& out,
                          const int& m, const int& n, const int& k,
                          const cute::UMMA::Major& a_major, const cute::UMMA::Major& b_major,
                          const bool& accumulate) {
    const auto trans_a = b_major == cute::UMMA::Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto trans_b = a_major == cute::UMMA::Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Matrix layouts
    const auto cuda_type_a = at::cuda::ScalarTypeToCudaDataType(rhs.scalar_type());
    const auto cuda_type_b = at::cuda::ScalarTypeToCudaDataType(lhs.scalar_type());
    const auto cuda_type_d = at::cuda::ScalarTypeToCudaDataType(out.scalar_type());
    const auto layout_a = b_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_a, k, n, rhs.stride(0))
                                                          : get_cublaslt_layout(cuda_type_a, n, k, rhs.stride(1));
    const auto layout_b = a_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_b, k, m, lhs.stride(0))
                                                          : get_cublaslt_layout(cuda_type_b, m, k, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(cuda_type_d, n, m, out.stride(0));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate);
}

static void cublaslt_bhr_hdr_bhd(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = d, n = b, k = r;
    const auto trans_a = CUBLAS_OP_T;
    const auto trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, k, m, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}


static void cublaslt_bhd_hdr_bhr(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = r, n = b, k = d;
    const auto trans_a = CUBLAS_OP_N;
    const auto trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, m, k, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/smxx_fp8_fp4_mqa_logits.hpp': r"""#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXFP8MQALogitsRuntime final: public LaunchRuntime<SMXXFP8MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits<
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_mqa_logits(const torch::Tensor& q,
                                const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                const torch::Tensor& weights,
                                const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end,
                                const torch::Tensor& logits,
                                const at::ScalarType& logits_dtype,
                                const int& seq_len, const int& seq_len_kv,
                                const int& max_seqlen_k, const int& stride_logits,
                                const int& num_heads, const int& head_dim,
                                const int& block_q, const int& block_kv) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    const int num_math_threads = (device_runtime->get_arch_major() == 10 ? 256 : 512);

    // Use compressed logits format when max_seqlen_k is specified
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q, num_heads, 0);

    // Calculate shared memory size
    int smem_size = 0;
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXFP8MQALogitsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SMXXFP8MQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("smxx_fp8_mqa_logits", code);
    SMXXFP8MQALogitsRuntime::launch(runtime, args);
}

class SM100FP4MQALogitsRuntime final: public LaunchRuntime<SM100FP4MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_mqa_logits<
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_fp4_mqa_logits(const torch::Tensor& q, const torch::Tensor& sf_q,
                                 const torch::Tensor& kv, const torch::Tensor& sf_kv,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start,
                                 const torch::Tensor& cu_seq_len_k_end,
                                 const torch::Tensor& logits,
                                 const at::ScalarType& logits_dtype,
                                 const int& seq_len, const int& seq_len_kv,
                                 const int& max_seqlen_k, const int& stride_logits,
                                 const int& num_heads, const int& head_dim,
                                 const int& block_q, const int& block_kv) {
    constexpr int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    constexpr int num_q_stages = 3, num_kv_stages = 6, num_tmem_stages = 3;

    // Use compressed logits format when max_seqlen_k is specified
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Construct TMAs
    // `head_dim` must be 128 for 64B swizzling
    DG_HOST_ASSERT(head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads,
                                               static_cast<int>(q.stride(1)),
                                               head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, seq_len,
                                                  num_heads, block_q,
                                                  static_cast<int>(sf_q.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv,
                                                static_cast<int>(kv.stride(0)),
                                                head_dim / 2, 0, false, false);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                                   get_tma_aligned_size(seq_len_kv, static_cast<int>(sf_kv.element_size())), 1,
                                                   block_kv, 1, 0, 0);

    // Calculate shared memory size
    const int smem_q_size_per_stage = block_q * num_heads * head_dim / 2;
    const int smem_sf_q_size_per_stage = align(block_q * num_heads, 128) * sizeof(int);
    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int smem_sf_kv_size_per_stage = align(block_kv, 128) * sizeof(int);
    const int smem_weight_size_per_stage = block_q * num_heads * sizeof(float);

    const int smem_barriers = (num_q_stages + num_kv_stages + num_tmem_stages) * 2 * 8;
    const int smem_tmem_ptr = 4;
    const int smem_size = num_q_stages * (smem_q_size_per_stage + smem_sf_q_size_per_stage + smem_weight_size_per_stage) + 
                          num_kv_stages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) + 
                          smem_barriers + smem_tmem_ptr;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SM100FP4MQALogitsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100FP4MQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_mqa_logits", code);
    SM100FP4MQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp': r"""#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXPagedMQALogitsMetadataRuntime final: public LaunchRuntime<SMXXPagedMQALogitsMetadataRuntime> {
public:
    struct Args {
        int aligned_batch_size;
        int split_kv;
        int num_sms;
        
        int batch_size;
        int next_n;
        bool is_context_lens_2d;
        int* context_lens;
        int* schedule_metadata;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::smxx_paged_mqa_logits_metadata<
        {}, {}, {}
    >);
}};
)", args.aligned_batch_size, args.split_kv, args.num_sms);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.next_n,
            args.is_context_lens_2d,
            args.context_lens,
            args.schedule_metadata
        ));
    }
};

static void smxx_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                           const torch::Tensor& schedule_metadata,
                                           const int& batch_size, const int& next_n,
                                           const int& block_kv, const int& num_sms,
                                           const bool& is_context_lens_2d) {
    constexpr int split_kv = 256;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    DG_HOST_ASSERT(split_kv % block_kv == 0);

    // Calculate shared memory size
    const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXPagedMQALogitsMetadataRuntime::Args& args = {
        .aligned_batch_size = aligned_batch_size,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .batch_size = batch_size,
        .next_n = next_n,
        .is_context_lens_2d = is_context_lens_2d,
        .context_lens = context_lens.data_ptr<int>(),
        .schedule_metadata = schedule_metadata.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto code = SMXXPagedMQALogitsMetadataRuntime::generate(args);
    const auto runtime = compiler->build("smxx_paged_mqa_logits_metadata", code);
    SMXXPagedMQALogitsMetadataRuntime::launch(runtime, args);
}

class SMXXFP8PagedMQALogitsRuntime final: public LaunchRuntime<SMXXFP8PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

        int* context_lens;
        void* logits;
        int* block_table;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", arch, arch,
    args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d,
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_paged_mqa_logits(const torch::Tensor& q,
                                      const torch::Tensor& kv_cache,
                                      const torch::Tensor& kv_cache_scales,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& context_lens,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& schedule_meta,
                                      const at::ScalarType& logits_dtype,
                                      const int& batch_size, const int& next_n,
                                      const int& num_heads, const int& head_dim,
                                      const int& num_kv_blocks, const int& block_kv,
                                      const bool& is_context_lens_2d,
                                      const int& logits_stride,
                                      const int& block_table_stride,
                                      const int& num_sms,
                                      const int& split_kv) {
    const int num_specialized_threads = 128;
    const int mma_m = (device_runtime->get_arch_major() == 10 ? 128 : 64);
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    const int num_q_stages = 3, num_kv_stages = (device_runtime->get_arch_major() == 10 ? 4 : 3);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    // Construct TMAs
    const int next_n_atom = (next_n % 2 == 0) ? 2 : 1;
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim);

    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_cache_scales, block_kv, num_kv_blocks,
                                                       block_kv, 1,
                                                       static_cast<int>(kv_cache_scales.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    // Calculate shared memory size
    int smem_size = 0;
    if (device_runtime->get_arch_major() == 9) {
        const int swizzle_alignment = head_dim * 8;

        const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
        const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
        const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);

        const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
        const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) + align(num_kv_stages * 8 * 2, swizzle_alignment);

        // Allocate some shared memory for UMMA barriers and tensor memory pointer, although it is not used in SM90
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
        DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    } else {
        const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim * static_cast<int>(q.element_size());
        const int smem_kv_size_per_stage = split_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int smem_kv_scale_size_per_stage = split_kv * static_cast<int>(kv_cache_scales.element_size());
        const int smem_weight_size_per_stage = next_n_atom * num_heads * static_cast<int>(weights.element_size());

        const int smem_barriers = (num_q_stages + num_kv_stages) * 2 * 8;
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = num_q_stages * (smem_q_size_per_stage + smem_weight_size_per_stage) + 
                    num_kv_stages * (smem_kv_size_per_stage + smem_kv_scale_size_per_stage) + 
                    smem_barriers + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
    }

    // Launch
    const SMXXFP8PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SMXXFP8PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("smxx_fp8_paged_mqa_logits", code);
    SMXXFP8PagedMQALogitsRuntime::launch(runtime, args);
}

class SM100FP4PagedMQALogitsRuntime final: public LaunchRuntime<SM100FP4PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

        int* context_lens;
        void* logits;
        int* block_table;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_paged_mqa_logits<
        {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d,
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_fp4_paged_mqa_logits(const torch::Tensor& q,
                                       const torch::Tensor& sf_q,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_sf,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& context_lens,
                                       const torch::Tensor& logits,
                                       const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta,
                                       const at::ScalarType& logits_dtype,
                                       const int& batch_size, const int& next_n,
                                       const int& num_heads, const int& head_dim,
                                       const int& num_kv_blocks, const int& block_kv,
                                       const bool& is_context_lens_2d,
                                       const int& logits_stride,
                                       const int& block_table_stride,
                                       const int& num_sms,
                                       const int& split_kv) {
    const int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    DG_HOST_ASSERT(split_kv == 256 and logits_stride % split_kv == 0);

    // TODO: tuning num_stages
    const int num_q_stages = 3, num_kv_stages = 6, num_tmem_stages = 3;
    const int next_n_atom = (next_n % 2 == 0) ? 2 : 1;

    // `head_dim` must be 128 for 64B swizzling
    DG_HOST_ASSERT(head_dim == 128);

    // Using 2D TMA as tensor q is asserted contiguous
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim / 2, 0, false, false);
    // NOTES: `sf_q` is a 3D tensor, while `weights` is a 2D tensor
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, batch_size * next_n,
                                                  num_heads, next_n_atom,
                                                  static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, block_kv, num_kv_blocks,
                                                   block_kv, 1,
                                                   static_cast<int>(kv_cache_sf.stride(0)), 0);

    // Calculate shared memory size
    const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim / 2;
    const int smem_sf_q_size_per_stage = align(next_n_atom * num_heads, 128) * sizeof(int);
    const int smem_kv_size_per_stage = split_kv * head_dim / 2;
    const int smem_sf_kv_size_per_stage = align(split_kv, 128) * sizeof(int);
    const int smem_weight_size_per_stage = next_n_atom * num_heads * sizeof(float);

    const int smem_barriers = (num_q_stages + num_kv_stages + num_tmem_stages) * 2 * 8;
    const int smem_tmem_ptr = 4;
    const int smem_size = num_q_stages * (smem_q_size_per_stage + smem_sf_q_size_per_stage + smem_weight_size_per_stage) + 
                          num_kv_stages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) + 
                          smem_barriers + smem_tmem_ptr;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SM100FP4PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100FP4PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_paged_mqa_logits", code);
    SM100FP4PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
""",
    'key-files/csrc/jit_kernels/impls/smxx_layout.hpp': r"""#pragma once

#include <torch/python.h>

#include "../../jit/kernel_runtime.hpp"
#include "../../jit/compiler.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

class TransposeFP32Runtime final: public LaunchRuntime<TransposeFP32Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int block_mn;
        void *sf, *out;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));
    }
};

class TransposeAndPackFP32IntoUE8M0Runtime final: public LaunchRuntime<TransposeAndPackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int mn, sf_k;
        int block_mn;
        void *sf, *out;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        {}, {}, {}
    >);
}};
)", args.launch_args.num_threads, args.block_mn, args.sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.sf, args.out, static_cast<uint32_t>(args.mn)));
    }
};

class PackFP32IntoUE8M0Runtime final: public LaunchRuntime<PackFP32IntoUE8M0Runtime> {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k, gran_k;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *ks;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pack_fp32_into_ue8m0<
        {}, {}, {}, {}
    >);
}};
)", args.num_groups, args.launch_args.num_threads, args.block_mn, args.block_packed_sf_k);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sf, args.out, args.ks, args.mn, args.sf_k, args.packed_sf_k, args.gran_k));
    }
};

static std::tuple<int, int, int, int, int, torch::Tensor> preprocess_sf(const torch::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::kFloat);
    const auto batched_sf = dim == 2 ? sf.unsqueeze(0) : sf;

    const auto [num_groups, mn, sf_k] = get_shape<3>(batched_sf);
    const auto tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& sf) {
    const auto [dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

    const auto out = torch::empty_strided({num_groups, mn, sf_k},
                                          {tma_aligned_mn * sf_k, 1, tma_aligned_mn},
                                          batched_sf.options());

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        out.copy_(batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int num_threads = 512;
        const auto smem_size = block_mn * (sf_k + (1 - (sf_k % 2))) * static_cast<int>(sizeof(float));
        const TransposeFP32Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .block_mn = block_mn,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_groups}, num_threads, smem_size)
        };

        const auto code = TransposeFP32Runtime::generate(args);
        const auto runtime = compiler->build("transpose_fp32", code);
        TransposeFP32Runtime::launch(runtime, args);
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::Tensor& sf) {
    const auto sf_reshaped = (sf.dim() == 2) ? sf.unsqueeze(0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto ue8m0_tensor = sf_reshaped.view(torch::kInt32).bitwise_right_shift(23).to(torch::kUInt8);

    // Second, make padded packed tensors
    const auto [num_groups, mn, k] = get_shape<3>(sf_reshaped);
    const auto aligned_mn = get_tma_aligned_size(mn, 4);
    const auto aligned_k  = align(k, 4);

    const auto options = torch::TensorOptions().device(sf.device()).dtype(torch::kUInt8);
    auto padded = torch::zeros({num_groups, aligned_mn, aligned_k}, options);
    // ReSharper disable once CppExpressionWithoutSideEffects
    padded.slice(1, 0, mn).slice(2, 0, k).copy_(ue8m0_tensor);
    padded = padded.view(-1).view(torch::kInt32).view({num_groups, aligned_mn, aligned_k / 4});

    // Finally, transpose
    auto out = torch::empty_strided({num_groups, aligned_mn, aligned_k / 4},
                                    {aligned_mn * (aligned_k / 4), 1, aligned_mn},
                                    at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    out = out.copy_(padded).slice(1, 0, mn);
    return (sf.dim() == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf) {
    const auto [dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto packed_sf_k = ceil_div(sf_k, 4);
    const auto out = torch::empty_strided({num_groups, mn, packed_sf_k},
                                          {packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn},
                                          at::TensorOptions().device(batched_sf.device()).dtype(torch::kInt));
    // Launch the kernel
    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_groups > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        constexpr int block_mn = 48;
        constexpr int num_threads = 512;
        const TransposeAndPackFP32IntoUE8M0Runtime::Args& args = {
            .mn = mn,
            .sf_k = sf_k,
            .block_mn = block_mn,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), num_groups}, num_threads, block_mn * sf_k * 4)
        };

        const auto code = TransposeAndPackFP32IntoUE8M0Runtime::generate(args);
        const auto runtime = compiler->build("transpose_and_pack_fp32_into_ue8m0", code);
        TransposeAndPackFP32IntoUE8M0Runtime::launch(runtime, args);
    } else {
        if (mn % 4 != 0 or num_groups > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;
        const PackFP32IntoUE8M0Runtime::Args& args = {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .ks = nullptr,
            .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
        };

        const auto code = PackFP32IntoUE8M0Runtime::generate(args);
        const auto runtime = compiler->build("pack_fp32_into_ue8m0", code);
        PackFP32IntoUE8M0Runtime::launch(runtime, args);
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                                            const torch::Tensor& ks_tensor,
                                                                            const std::vector<int>& ks,
                                                                            const int gran_k) {
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    const auto [sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(ks.size());

    int ref_sf_k = 0, packed_sf_k = 0;
    for (const auto k: ks)
        ref_sf_k += ceil_div(k, gran_k), packed_sf_k += ceil_div(k, gran_k * 4);
    DG_HOST_ASSERT(sf.is_contiguous());
    DG_HOST_ASSERT(ref_sf_k == sf_k);
    DG_HOST_ASSERT(num_groups <= 128 and mn % 4 == 0);

    const auto out = torch::empty({packed_sf_k, mn}, at::TensorOptions().device(sf.device()).dtype(torch::kInt));

    constexpr int block_mn = 128;
    constexpr int block_packed_sf_k = 16;
    constexpr int num_threads = 512;
    const PackFP32IntoUE8M0Runtime::Args& args = {
        .num_groups = num_groups,
        .mn = mn,
        .sf_k = sf_k,
        .packed_sf_k = packed_sf_k,
        .gran_k = gran_k,
        .block_mn = block_mn,
        .block_packed_sf_k = block_packed_sf_k,
        .sf = sf.data_ptr(),
        .out = out.data_ptr(),
        .ks = ks_tensor.data_ptr(),
        .launch_args = LaunchArgs({ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k)}, num_threads)
    };

    const auto code = PackFP32IntoUE8M0Runtime::generate(args);
    const auto runtime = compiler->build("pack_fp32_into_ue8m0", code);
    PackFP32IntoUE8M0Runtime::launch(runtime, args);
    return out;
}

} // namespace deep_gemm
""",
    'key-files/csrc/python_api.cpp': r"""#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
""",
    'key-files/csrc/utils/exception.hpp': r"""#pragma once

#include <cublasLt.h>
#include <exception>
#include <string>
#include <sstream>

#include "compatibility.hpp"

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_NVRTC_CHECK
#define DG_NVRTC_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != NVRTC_SUCCESS) { \
        throw DGException("NVRTC", __FILE__, __LINE__, nvrtcGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef DG_CUDA_DRIVER_CHECK
#define DG_CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        std::stringstream ss; \
        const char *name, *info; \
        lazy_cuGetErrorName(e, &name), lazy_cuGetErrorString(e, &info); \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
        throw DGException("CUDA driver", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_CUDA_RUNTIME_CHECK
#define DG_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw DGException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_CUBLASLT_CHECK

#if !DG_CUBLAS_GET_ERROR_STRING_COMPATIBLE
inline const char* cublasGetStatusString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}
#endif

#define DG_CUBLASLT_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        std::ostringstream ss; \
        ss << static_cast<int>(e) << " (" << cublasGetStatusString(e) << ")"; \
        throw DGException("cuBLASLt", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

} // namespace deep_gemm
""",
    'key-files/csrc/utils/hash.hpp': r"""#pragma once

#include <string>

namespace deep_gemm {

static uint64_t fnv1a(const std::vector<char>& data, const uint64_t& seed) {
    uint64_t h = seed;
    const uint64_t prime = 0x100000001b3ull;
    for (const char& c: data) {
        h ^= static_cast<uint8_t>(c);
        h *= prime;
    }
    return h;
}

static std::string get_hex_digest(const std::vector<char>& data) {
    const auto state_0 = fnv1a(data, 0xc6a4a7935bd1e995ull);
    const auto state_1 = fnv1a(data, 0x9e3779b97f4a7c15ull);

    // Split-mix 64
    const auto split_mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << split_mix(state_0)
        << std::setw(16) << split_mix(state_1);
    return oss.str();
}

static std::string get_hex_digest(const std::string& data) {
    return get_hex_digest(std::vector<char>{data.begin(), data.end()});
}

} // namespace deep_gemm
""",
    'key-files/csrc/utils/layout.hpp': r"""#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

#include "math.hpp"
#include "exception.hpp"
#include "../jit/device_runtime.hpp"

namespace deep_gemm {

// Major-ness stuffs
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if (dim == 3)
        DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

static cute::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check(t);
    return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
}

static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_requires_k_major() {
    return device_runtime->get_arch_major() == 9;
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.dim() == N);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

static std::tuple<int, int> check_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [mn, k] = get_shape<2>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(mn, k);
}

static std::tuple<int, int, int> check_grouped_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [num_groups, mn, k] = get_shape<3>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(num_groups, mn, k);
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format
            std::make_tuple(1,   1, 128);   // 1D1D kernels
    }
    DG_HOST_UNREACHABLE("Unknown recipe");
}

// SF layouts
static torch::Tensor check_sf_layout(const torch::Tensor& sf,
                                     const int& mn, const int& k,
                                     const int& gran_mn, const int& gran_k,
                                     const std::optional<int>& num_groups,
                                     const bool& tma_stride_check = false,
                                     const bool& sm90_sfb_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto sf_dtype = sf.scalar_type();
    DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.size(-3) == num_groups.value());
    DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));

    // TMA stride checks: TMA aligned and MN-major
    if (tma_stride_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
        // Check contiguity in the MN direction
        DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);
        DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
    }

    // SM90 SFB must be contiguous, or contiguous after transposing the last two dimensions
    if (sm90_sfb_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.size(-2) * sf.size(-1));
        DG_HOST_ASSERT((sf.stride(-1) == 1 and sf.stride(-2) == sf.size(-1)) or
                       (sf.stride(-1) == sf.size(-2) and sf.stride(-2) == 1));
    }
    return sf;
}

} // namespace deep_gemm
""",
    'key-files/csrc/utils/math.hpp': r"""// TODO: merge this file with `math.cuh` (the device part)
#pragma once

#include <torch/python.h>

#include "exception.hpp"

namespace deep_gemm {

// TODO: use `torch::kFloat4_e2m1fn_x2`
constexpr auto kPackedFP4 = torch::kInt8;

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

} // namespace deep_gemm
""",
    'key-files/csrc/utils/system.hpp': r"""#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <memory>
#include <unistd.h>

#include "exception.hpp"
#include "format.hpp"

namespace deep_gemm {

// ReSharper disable once CppNotAllPathsReturnValue
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    const auto c_str = std::getenv(name.c_str());
    if (c_str == nullptr)
        return default_value;

    // Read the env and convert to the desired type
    if constexpr (std::is_same_v<dtype_t, std::string>) {
        return std::string(c_str);
    } else if constexpr (std::is_same_v<dtype_t, int>) {
        int value;
        std::sscanf(c_str, "%d", &value);
        return value;
    } else {
        DG_HOST_ASSERT(false and "Unexpected type");
    }
}

static std::tuple<int, std::string> call_external_command(std::string command) {
    command = command + " 2>&1";
    const auto deleter = [](FILE* f) { if (f) pclose(f); };
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(command.c_str(), "r"), deleter);
    DG_HOST_ASSERT(pipe != nullptr);

    std::array<char, 512> buffer;
    std::string output;
    while (fgets(buffer.data(), buffer.size(), pipe.get()))
        output += buffer.data();
    const auto status = pclose(pipe.release());
    // NOTES: if the child was killed by a signal (e.g., SIGINT from Ctrl+C),
    // WEXITSTATUS would incorrectly return 0. Treat signal death as failure.
    const auto exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    return {exit_code, output};
}

static std::vector<std::filesystem::path> collect_files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;
    std::function<void(const std::filesystem::path&)> impl;
    impl = [&](const std::filesystem::path& dir) {
        for (const auto& entry: std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
                impl(entry.path());
            } else if (entry.is_regular_file() and entry.path().extension() == ".cuh") {
                files.emplace_back(entry.path());
            }
        }
    };
    impl(root);

    // Be consistent
    std::sort(files.begin(), files.end());
    return files;
}

static std::filesystem::path make_dirs(const std::filesystem::path& path) {
    // OK if existed
    std::error_code capture;
    const bool created = std::filesystem::create_directories(path, capture);
    if (not (created or capture.value() == 0)) {
        DG_HOST_UNREACHABLE(fmt::format("Failed to make directory: {}, created: {}, value: {}",
                                        path.c_str(), created, capture.value()));
    }
    if (created and get_env<int>("DG_JIT_DEBUG"))
        printf("Create directory: %s\n", path.c_str());
    return path;
}

static std::string get_uuid() {
    static std::random_device rd;
    static std::mt19937 gen([]() {
        return rd() ^ std::chrono::steady_clock::now().time_since_epoch().count();
    }());
    static std::uniform_int_distribution<uint32_t> dist;

    std::stringstream ss;
    ss << getpid() << "-"
       << std::hex << std::setfill('0')
       << std::setw(8) << dist(gen) << "-"
       << std::setw(8) << dist(gen) << "-"
       << std::setw(8) << dist(gen);
    return ss.str();
}

static void safe_remove_all(const std::filesystem::path& path) {
    std::error_code ec;
    if (not std::filesystem::exists(path, ec) or ec)
        return;

    // A single file
    if (not std::filesystem::is_directory(path, ec) or ec) {
        std::filesystem::remove(path, ec);
        return;
    }

    // Remove directory
    auto it = std::filesystem::directory_iterator(path,
        std::filesystem::directory_options::skip_permission_denied, ec);
    for (auto end = std::filesystem::directory_iterator(); it != end and not ec;) {
        const auto entry_path = it->path();

        // Increase firstly to avoid failures
        it.increment(ec);
        if (ec)
            break;

        // Recursively clean
        safe_remove_all(entry_path);
    }
    std::filesystem::remove(path, ec);
}

} // deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/comm/barrier.cuh': r"""#pragma once

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        do {
            new_value = ptx::ld_acq(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival (with 30s timeout, at 2 GHz)
        constexpr int64_t kNumTimeoutCycles = 30ll * 2000000000ll;
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    printf("DeepGEMM NVLink barrier timeout (30s): rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
                }
            }
        }
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
""",
    'key-files/deep_gemm/include/deep_gemm/common/compile.cuh': r"""#pragma once

#include <cutlass/detail/helper_macros.hpp>

#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
#define DG_IN_CUDA_COMPILATION
#endif

#if defined(__NVCC__) || (defined(__clang__) and defined(__CUDA__))
#define CUTLASS_HOST_DEVICE_NOINLINE  __device__ __host__
#define CUTLASS_DEVICE_NOINLINE __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE_NOINLINE __device__
#define CUTLASS_DEVICE_NOINLINE __device__
#else
#define CUTLASS_HOST_DEVICE_NOINLINE
#define CUTLASS_DEVICE_NOINLINE
#endif
""",
    'key-files/deep_gemm/include/deep_gemm/common/cute_tie.cuh': r"""#pragma once

#include <cute/int_tuple.hpp>

namespace cute {

struct ignore_t {
    template <typename T>
    constexpr const ignore_t& operator=(T&&) const noexcept {
        return *this;
    }
};

inline constexpr ignore_t ignore{};

} // namespace cute

#define CUTE_TIE_CONCAT_IMPL(A, B) A##B
#define CUTE_TIE_CONCAT(A, B) CUTE_TIE_CONCAT_IMPL(A, B)

#define CUTE_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define CUTE_TIE_COUNT_ARGS(...) \
    CUTE_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define CUTE_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::cute::get<I>(TUPLE)
#define CUTE_TIE_OP_ASSIGN(I, TUPLE, VAR) VAR = ::cute::get<I>(TUPLE)

#define CUTE_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define CUTE_TIE_APPLY_OP_2(OP, T, V1, V2) OP(0, T, V1); OP(1, T, V2);
#define CUTE_TIE_APPLY_OP_3(OP, T, V1, V2, V3) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3);
#define CUTE_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4);
#define CUTE_TIE_APPLY_OP_5(OP, T, V1, V2, V3, V4, V5) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4); OP(4, T, V5);

#define CUTE_TIE_DECL(TUPLE_EXPR, ...) \
    auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
        CUTE_TIE_OP_DECL, \
        CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), \
        __VA_ARGS__ \
    )

#define CUTE_TIE(TUPLE_EXPR, ...) \
    do { \
        auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
        CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
            CUTE_TIE_OP_ASSIGN, \
            CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), \
            __VA_ARGS__ \
        ); \
    } while (0)
""",
    'key-files/deep_gemm/include/deep_gemm/common/exception.cuh': r"""#pragma once

#include <cuda/std/cstdint>
#include <deep_gemm/common/compile.cuh>

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif
""",
    'key-files/deep_gemm/include/deep_gemm/common/math.cuh': r"""#pragma once

#include <cuda/std/cstdint>
#include <deep_gemm/common/compile.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::math {

/// Pointer operations
template <typename dtype_t = void>
CUTLASS_HOST_DEVICE dtype_t* advance_ptr(void* ptr, const uint64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<uint8_t*>(ptr) + num_bytes);
}

/// Math functions
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
CUTLASS_DEVICE void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

#ifdef DG_IN_CUDA_COMPILATION
CUTLASS_DEVICE float2 fma2(const float2& a, const float2& b, const float2& c) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(
        __fmaf_rn(a.x, b.x, c.x),
        __fmaf_rn(a.y, b.y, c.y)
    );
#endif
}

CUTLASS_HOST_DEVICE float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

/// Casting
template <typename old_t>
CUTLASS_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

CUTLASS_DEVICE float fast_pow2(const int& x) {
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

CUTLASS_DEVICE int fast_log2_ceil(float x) {
    const auto bits = *reinterpret_cast<uint32_t*>(&x);
    const auto exp = bits >> 23;
    const auto man = bits & ((1 << 23) - 1);
    return exp - 127 + (man != 0);
}

template <bool kUseUE8M0 = true>
CUTLASS_DEVICE void get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {
    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");
    const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};
    const auto scaled = __fmul2_rn(amax, finfo_factor);
    const auto exp_x = fast_log2_ceil(scaled.x);
    const auto exp_y = fast_log2_ceil(scaled.y);
    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);
    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);
}

/// Reduction
CUTLASS_DEVICE uint32_t warp_inclusive_sum(uint32_t value, const uint32_t& lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

// Operation functors
template <typename T> struct ReduceSum { CUTLASS_DEVICE T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { CUTLASS_DEVICE T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct ReduceAnd { CUTLASS_DEVICE T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { CUTLASS_DEVICE T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
CUTLASS_DEVICE T warp_reduce(T value, Op op) {
    DG_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <uint32_t kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
CUTLASS_DEVICE T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}
#endif

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/common/tma_copy.cuh': r"""#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

    if constexpr (not kIs3DTMA) {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                             inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                    }
                }
            #endif
        }
    } else {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                            inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace deep_gemm::tma
""",
    'key-files/deep_gemm/include/deep_gemm/common/types.cuh': r"""#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
};

constexpr CUTLASS_HOST_DEVICE int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        default: return 0;
    }
}

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/common/utils.cuh': r"""#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

} // namespace deep_gemm::utils
""",
    'key-files/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh': r"""#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const cute::TmaDescriptor& tensor_map_cd) {
    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M waves
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        // Issue every swizzled atom and pipeline STSM and TMA store
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            // Wait shared memory to be released
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // The pipeline stage
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);

            // Store into shared memory
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                // Calculate the index of the bank group to be written in the atom
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                // Reshape the atom in another view and swizzle
                //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                // Source and destination memory address
                uint32_t tmem_addr = tmem_base_addr +                                       // Accumulator offset
                                     w * BLOCK_N +                                          // Wave offset
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                auto smem_ptr = smem_base_ptr +                                             // Base pointer
                                epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                // Load from tensor memory, store into shared memory
                uint32_t values[kNumElemsPerBankGroup];
                if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                    // For FP32 output, read and store
                    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                } else {
                    // For BF16 output, read, cast and store
                    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::st_shared(
                        smem_ptr,
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7])
                    );
                }
            }

            // Notify tensor memory empty (only at the leader CTA) arrival ASAP
            // NOTES: only the last stage needs to do this
            if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // Synchronize all threads and issue TMA
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

} // namespace deep_gemm::epilogue
""",
    'key-files/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh': r"""#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd_swap_ab(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                       const uint32_t& tmem_base_addr,
                       const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                       const uint32_t& effective_m,
                       const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                       const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                       const cute::TmaDescriptor& tensor_map_cd) {
    // NOTES: The epilogue requires a full warpgroup to read all 128 TMEM rows,
    //          implying STORE_BLOCK_N must be 128.
    DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");

    // TMA checks
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M blocks
    const auto num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s, advance_store_pipeline()) {
        // Wait shared memory to be released
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++ i) {
            uint32_t tmem_addr = tmem_base_addr +
                                 s * STORE_BLOCK_M +            // Store stage offset
                                 i * kNumSwizzleAtomRows;       // In-block offset
            uint32_t values[kNumSwizzleAtomRows];

            // Warps cooperatively write an atomic block to shared memory
            DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            uint32_t inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) + outer_atom_offset + inner_atom_offset;

            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                // NOTES: Swizzling is not required in this case, but used here for consistency with other cases
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                                                                 values[4], values[5], values[6], values[7]);
                uint32_t col = lane_idx / 4;

                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                    auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                                  + (col ^ row) * kNumBankGroupBytes
                                                  + (lane_idx % 4) * sizeof(float);
                    ptx::st_shared(reinterpret_cast<uint32_t*>(smem_ptr), values[row]);
                }
            } else {
                // Load from TMEM using `.16x256b` shape to satisfy STSM layout requirements
                // Start from lane index 0
                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                       values[0], values[1], values[2], values[3]);
                // Start from lane index 16
                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                       values[4], values[5], values[6], values[7]);
                cutlass::arch::fence_view_async_tmem_load();

                // Destination shared memory address
                uint32_t row = lane_idx % 8;
                uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                              + (col ^ row) * kNumBankGroupBytes;

                // Store matrix with transposition
                ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(values[0], values[1]),
                                                  math::cast_into_bf16_and_pack(values[2], values[3]),
                                                  math::cast_into_bf16_and_pack(values[4], values[5]),
                                                  math::cast_into_bf16_and_pack(values[6], values[7]),
                                                  smem_ptr);
            }
        }

        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
        // NOTES: only the last stage needs to do this
        if (s == num_stores - 1) {
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barrier->arrive(0u);
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                auto smem_ptr = smem_cd[tma_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                uint32_t m_idx = base_m_idx + s * STORE_BLOCK_M;
                uint32_t n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N_ATOM>(base_n_idx + i * STORE_BLOCK_N_ATOM);

                // Issue 2D or 3D TMA store
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                }
            }
            cute::tma_store_arrive();
        }
        __syncwarp();
    }
}

} // namespace deep_gemm::epilogue
""",
    'key-files/deep_gemm/include/deep_gemm/epilogue/transform.cuh': r"""#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::epilogue::transform {

struct EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }
};

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 and kMid % STORE_BLOCK_N == 0 and
                         kRight % STORE_BLOCK_N == 0, "Invalid head splits config");
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};

} // namespace deep_gemm::epilogue::transform
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh': r"""#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_impl(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory out of bound for UMMA");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers              = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, kNumMulticast, batch_idx);

                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // UMMA and empty barrier arrival alias
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        uint32_t atom_k_idx = k * UMMA_K / BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K, k * UMMA_K % BLOCK_ATOM_K);
                        if (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    __syncwarp();

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<
                (not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_bmk_bnk_mn.cuh': r"""#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/util/type_traits.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
sm100_bmn_bnk_mn_gemm_impl(uint32_t shape_s,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == LAYOUT_AD_M and BLOCK_N == 128 and BLOCK_K == 64, "Invalid block size");
    DG_STATIC_ASSERT(kSwizzleABMode == 128 and kSwizzleCDMode == 128, "Invalid swizzle mode");

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Real tensor memory size and offsets
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<BLOCK_N>();

    // Fill D/A/B
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (i * SMEM_CD_SIZE_PER_STAGE));
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE +
            kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barrier = barrier_start_ptr + (kNumStages * 2);

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
        }
        tmem_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Block indices
    const uint32_t num_n_blocks = math::ceil_div(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * math::ceil_div(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages = cute::min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx == 0) {
        // TMA load warp
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            const auto& stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

            uint32_t m_idx = BLOCK_M * m_block_idx;
            uint32_t n_idx = BLOCK_N * n_block_idx;
            uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
            uint32_t k_idx = sk_idx % SHAPE_K;
            uint32_t s_idx = sk_idx / SHAPE_K;

            // Issue TMAs
            if (cute::elect_one_sync()) {
                tma::copy<BLOCK_K, BLOCK_M, kSwizzleABMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx + s_idx * SHAPE_M);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleABMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx + s_idx * SHAPE_N);
            }

            // Arrive at full barriers
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            if (cute::elect_one_sync())
                full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }
    } else if (warp_idx == 1) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::bfloat16_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleABMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleABMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Wait tensor memory empty barrier arrival
        ptx::tcgen05_after_thread_sync();

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);
            ptx::tcgen05_after_thread_sync();

            // Issue UMMA in the leader CTA
            const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, stage_idx);
            const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, stage_idx);
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_M, kSwizzleABMode, cutlass::bfloat16_t>(
                        a_desc_base_lo, 0, k * UMMA_K);
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_N, kSwizzleABMode, cutlass::bfloat16_t>(
                        b_desc_base_lo, 0, k * UMMA_K);
                    ptx::SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, 0, s > 0 or k > 0, runtime_instr_desc);
                }
            }

            // Commit to the mbarrier object
            // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
    }

    // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
    // i.e., no need for `tmem_ptr |= (warp_idx * 32) << 16`.
    // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
    if (warp_idx == 2)
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(float);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

    // Wait UMMA arrival
    tmem_full_barrier->wait(0);
    ptx::tcgen05_after_thread_sync();

    // Load from tensor memory into registers, and write shared memory with STSM
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // Issue every swizzled atom and pipeline STSM and TMA store
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
    #pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++ s) {
        // Wait shared memory to be released
        if (s >= kNumTMAStoreStages) {
            if (warp_idx == 0 and cute::elect_one_sync())
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier(kNumThreads).sync();
        }

        // The pipeline stage
        const auto tma_stage_idx = s % kNumTMAStoreStages;
        const auto m_idx = m_block_idx * BLOCK_M;
        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
            // Calculate the index of the bank group to be written in the atom
            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

            // Reshape the atom in another view and swizzle
            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
            col ^= row % (kSwizzleCDMode / 16);

            // Source and destination memory address
            uint32_t tmem_addr = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                            warp_idx * 32 * kSwizzleCDMode +                            // Warp offset
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

            // Load from tensor memory, store into shared memory
            uint32_t values[kNumElemsPerBankGroup];
            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier(kNumThreads).sync();
        if (warp_idx == 0 and cute::elect_one_sync()) {
            cute::SM90_TMA_REDUCE_ADD_2D::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);
            cute::tma_store_arrive();
        }
    }

    // Deallocate tensor memory by warp 1
    // NOTES: warp 0 is doing TMA stores
    if (warp_idx == 1)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

}
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp4_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k,
                          const uint32_t logits_stride,
                          const uint32_t* cu_seq_len_k_start,
                          const uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // UMMA configs
    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = 64;
    static constexpr uint32_t kNumSFQ  = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarpGroups * UMMA_M and BLOCK_KV % kNumUTCCPAlignedElems == 0, "Invalid `BLOCK_KV`");

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = BLOCK_Q * kNumHeads * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQ * sizeof(int);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = BLOCK_KV * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
    });
    const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages
                                                    + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_tmem_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
    auto empty_tmem_barriers = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

    // Tensor memory configs
    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            full_tmem_barriers[i]->init(1);
            empty_tmem_barriers[i]->init(128);
        }
        cutlass::arch::fence_barrier_init();
    }

    // Allocate tensor memory
    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    __syncthreads();

    // Scheduler
    const uint32_t num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    auto load_schedule = [&](const uint32_t& q_idx) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto row_idx = cute::min(q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cute::min(cu_seq_len_k_start[row_idx], seq_len_kv);
            seq_k_end[i] = cute::min(cu_seq_len_k_end[row_idx], seq_len_kv);
            start = cute::min(start, seq_k_start[i]);
            end = cute::max(end, seq_k_end[i]);
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {start, math::ceil_div(end - start, BLOCK_KV)};
    };

    // Make Q, KV and TMEM pipeline
    auto make_pipeline = [](const uint32_t& num_stages) {
        // Return current stage and phase, and advance pipeline by steps
        return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable -> cute::tuple<uint32_t, uint32_t> {
            uint32_t current_idx = iter_idx;
            iter_idx += step;
            return {current_idx % num_stages, (current_idx / num_stages) & 1};
        };
    };
    auto advance_q_pipeline    = make_pipeline(kNumQStages);
    auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
    auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading Q
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Enumerate Q blocks
        if (cute::elect_one_sync()) {
            for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
                // Wait Q consumer release
                CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                cute::SM90_TMA_LOAD_2D::copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_q[q_stage_idx], 0, q_idx * BLOCK_Q * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, full_q_barriers[q_stage_idx], smem_sf_q[q_stage_idx], 0, q_idx * BLOCK_Q);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[q_stage_idx], smem_weights[q_stage_idx], 0, q_idx * BLOCK_Q);
                full_q_barriers[q_stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + kRealNumSFQ * sizeof(int) + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        // TMA warp for loading KV cache
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
            // Enumerate Q blocks
            for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
                // Load KV block ranges
                CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

                // Enumerate KV blocks
                for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                    // Wait KV consumer release
                    CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    cute::SM90_TMA_LOAD_2D::copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                                 static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                 smem_kv[kv_stage_idx], 0, kv_start + kv_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                                              smem_sf_kv[kv_stage_idx],
                                              kv_start + kv_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        // UMMA warp
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        // Enumerate Q blocks
        for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
            // Load KV block ranges
            CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Transpose and copy SF Q
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
                cutlass::arch::fence_view_async_shared();
                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                __syncwarp();
            }

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Wait TMA KV arrivals
                CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Transpose
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                    cutlass::arch::fence_view_async_shared();
                }

                // UMMA with SF
                if (cute::elect_one_sync()) {
                    // Copy SF KV
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        // Wait TMEM release
                        CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        // Issue UMMA with SF
                        #pragma unroll
                        for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                            auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * 2, k * 2);
                            // TODO: generalize umma desc
                            DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
                            auto a_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            auto b_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_q[q_stage_idx] + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            ptx::SM100_MMA_MXF4_SS::fma(
                                a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                        }
                        // TODO: move this into `deep_gemm/ptx/tcgen05.cuh`
                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
            }

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline. Without this, UMMA can consume
            // kNumQStages Q blocks before math warps release any, causing a
            // circular dependency: UMMA waits full_q -> TMA_Q waits empty_q
            // -> Math waits full_tmem -> UMMA (already moved on).
            empty_q_barriers[q_stage_idx]->arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = threadIdx.x;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr uint32_t N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Math warpgroups process TMEM stages alternately
        // Advance pipeline to align with the assigned stage
        advance_tmem_pipeline(math_warpgroup_idx);

        // Local register buffers
        float accum[kNumHeads];
        float weights[BLOCK_Q][kNumHeads];

        // Enumerate Q blocks
        for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
            // Load KV block ranges
            CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            // TODO: optimize bank conflicts
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Calculate KV offset in advance
                auto kv_offset = kv_start + kv_idx * BLOCK_KV + math_thread_idx;

                // Advance pipeline by `kNumMathWarpGroups` steps
                // Wait UMMA arrival
                CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
                ptx::tcgen05_after_thread_sync();

                // Reduce over the head dim and store
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load accumulator from TMEM
                    uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                    tmem_load(cute::Int<kNumHeads>{}, tmem_addr, accum);

                    // Release TMEM empty
                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_tmem_barriers[tmem_stage_idx]->arrive();
                    }

                    // Accumulate weighted ReLU in parallel
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    auto result = static_cast<logits_dtype_t>(sum.x + sum.y);

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    // TODO: optimize performance
                    const auto q_offset = (q_idx * BLOCK_Q + i) * static_cast<uint64_t>(logits_stride);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                            logits[q_offset + kv_offset - seq_k_start[i]] = result;
                    } else {
                        logits[q_offset + kv_offset] = result;
                    }
                    __syncwarp();
                }
            }

            // Release last Q empty
            empty_q_barriers[q_stage_idx]->arrive();
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp4_paged_mqa_logits(const uint32_t batch_size,
                                const uint32_t logits_stride, const uint32_t block_table_stride,
                                const uint32_t* context_lens, logits_dtype_t* logits,
                                const uint32_t* block_table, const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // Next-N atom configs
    static constexpr uint32_t kNextNAtom = (kNextN % 2 == 0) ? 2 : 1;
    static constexpr uint32_t kNumNextNAtoms = kNextN / kNextNAtom;
    static constexpr bool kSingleAtom = (kNumNextNAtoms == 1);

    // UMMA configs
    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = kNextNAtom * kNumHeads;
    static constexpr uint32_t UMMA_K = 64;
    static constexpr uint32_t kNumSFQAtom  = math::constexpr_align(kNextNAtom * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kRealNumSFQAtom = kNextNAtom * kNumHeads;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = kNextNAtom * kNumHeads * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQAtom * sizeof(int);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = SPLIT_KV * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
    });
    const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages
                                                    + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_tmem_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
    auto empty_tmem_barriers = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

    // Tensor memory configs
    constexpr uint32_t kNumAccumTmemCols = kNextNAtom * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQAtom / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQAtom / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumTmemStages; ++i) {
                full_tmem_barriers[i]->init(1);
                empty_tmem_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Scheduler
    constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
    using Scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, BLOCK_KV, kNumBlocksPerSplit, kNumNextNAtoms>;
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit, "Invalid `SPLIT_KV`");

    // Make Q, KV and TMEM pipeline
    auto make_pipeline = [](const uint32_t& num_stages) {
        // Return current stage and phase, and advance pipeline by steps
        return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable -> cute::tuple<uint32_t, uint32_t> {
            uint32_t current_idx = iter_idx;
            iter_idx += step;
            return {current_idx % num_stages, (current_idx / num_stages) & 1};
        };
    };
    auto advance_q_pipeline    = make_pipeline(kNumQStages);
    auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
    auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading Q
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
            auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);

            // Persistently schedule over blocks
            // Initialize outside valid range to indicate no previous task
            uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
            uint32_t q_atom_idx, _, __;
            while (scheduler.fetch_next_task(q_atom_idx, _, __)) {
                // Issue TMA Q when (q_idx, atom_idx) changes
                if (q_atom_idx != last_q_atom_idx) {
                    // Wait Q consumer release
                    CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
                    empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                    // Issue TMA Q
                    cute::SM90_TMA_LOAD_2D::copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
                                                 static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                 smem_q[q_stage_idx], 0, q_atom_idx * kNextNAtom * kNumHeads);
                    tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_sf_q, full_q_barriers[q_stage_idx], smem_sf_q[q_stage_idx], 0, q_atom_idx * kNextNAtom);
                    tma::copy<kNumHeads, kNextNAtom, 0>(&tensor_map_weights, full_q_barriers[q_stage_idx], smem_weights[q_stage_idx], 0, q_atom_idx * kNextNAtom);
                    full_q_barriers[q_stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + kRealNumSFQAtom * sizeof(int) + SMEM_WEIGHT_SIZE_PER_STAGE);
                }
                last_q_atom_idx = q_atom_idx;
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        // TMA warp for loading KV cache
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);

        // Persistently schedule over blocks
        uint32_t kv_block_idx_ptr = 32, kv_block_idx_storage;
        uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
        uint32_t q_atom_idx, kv_idx, num_kv;
        while (scheduler.fetch_next_task(q_atom_idx, kv_idx, num_kv)) {
            // Reset block table cache on kv restart
            if (q_atom_idx != last_q_atom_idx)
                kv_block_idx_ptr = 32;
            last_q_atom_idx = q_atom_idx;

            // Coalesced load of block table
            if (kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                const auto block_table_offset = (q_atom_idx / kNumNextNAtoms) * static_cast<uint64_t>(block_table_stride);
                kv_block_idx_storage = (kv_idx + lane_idx < num_kv)
                    ? block_table[block_table_offset + kv_idx + lane_idx] : 0;
            }

            // Broadcast KV block indices 
            int kv_block_idx[kNumBlocksPerSplit];
            #pragma unroll
            for (int i = 0; i < kNumBlocksPerSplit; ++ i)
                kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
            kv_block_idx_ptr += kNumBlocksPerSplit;
            DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `SPLIT_KV`");

            // Wait KV consumer release
            CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
            
            // Issue TMA KV
            if (cute::elect_one_sync()) {
                empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
                #pragma unroll
                for (int i = 0; i < kNumBlocksPerSplit; ++ i) {
                    cute::SM90_TMA_LOAD_3D::copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                                 static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                 smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim / 2) * i,
                                                 0, 0, kv_block_idx[i]);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                                              smem_sf_kv[kv_stage_idx] + BLOCK_KV * i,
                                              0, kv_block_idx[i]);
                }
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        // UMMA warp
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        // Persistently schedule over blocks
        uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
        uint32_t q_atom_idx, kv_idx, _;
        while (scheduler.fetch_next_task(q_atom_idx, kv_idx, _)) {
            // Wait TMA Q arrivals
            uint32_t q_stage_idx, q_phase;
            if (q_atom_idx != last_q_atom_idx) {
                CUTE_TIE(advance_q_pipeline(), q_stage_idx, q_phase);

                // Release previous Q empty (UMMA warp must participate to prevent
                // running ahead of math warps in the Q pipeline)
                if (last_q_atom_idx != batch_size * kNumNextNAtoms)
                    empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]->arrive();

                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Transpose and copy SF Q
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQAtom / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                    cutlass::arch::fence_view_async_shared();
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    if (cute::elect_one_sync())
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                    __syncwarp();
                }
            }
            last_q_atom_idx = q_atom_idx;

            // Wait TMA KV arrivals
            CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Transpose
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
                cutlass::arch::fence_view_async_shared();
            }

            // UMMA with SF
            if (cute::elect_one_sync()) {
                // Copy SF KV
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                }

                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    // Wait TMEM release
                    CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
                    uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                    empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
                    ptx::tcgen05_after_thread_sync();

                    // Issue UMMA with SF
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * 2, k * 2);
                        // TODO: generalize UMMA desc
                        DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
                        auto a_desc = mma::sm100::make_smem_desc(
                            cute::UMMA::LayoutType::SWIZZLE_64B,
                            smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                            8 * (kHeadDim / 2), 0);
                        auto b_desc = mma::sm100::make_smem_desc(
                            cute::UMMA::LayoutType::SWIZZLE_64B,
                            smem_q[q_stage_idx] + k * UMMA_K / 2,
                            8 * (kHeadDim / 2), 0);
                        ptx::SM100_MMA_MXF4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                                    kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                    }
                    // TODO: move this PTX into headers
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                 ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
                }
            }
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Math warpgroups process TMEM stages alternately
        // Advance pipeline to align with the assigned stage
        advance_tmem_pipeline(math_warpgroup_idx);

        // Local register buffers
        float accum[kNumHeads];
        float weights[kNextNAtom][kNumHeads];

        // Persistently schedule over blocks
        uint32_t last_q_atom_idx = batch_size * kNumNextNAtoms;
        uint32_t q_atom_idx, kv_idx, _;
        while (scheduler.fetch_next_task(q_atom_idx, kv_idx, _)) {
            if (q_atom_idx != last_q_atom_idx) {
                CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);

                // Release last Q empty
                if (last_q_atom_idx != batch_size * kNumNextNAtoms)
                    empty_q_barriers[(q_stage_idx + kNumQStages - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrivals
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        float4 raw = ptx::ld_shared((float4*)(smem_weights[q_stage_idx] + i * kNumHeads + j));
                        weights[i][j + 0] = raw.x;
                        weights[i][j + 1] = raw.y;
                        weights[i][j + 2] = raw.z;
                        weights[i][j + 3] = raw.w;
                    }
                }
            }
            last_q_atom_idx = q_atom_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_atom_idx * kNextNAtom * static_cast<uint64_t>(logits_stride) + kv_idx * BLOCK_KV + math_thread_idx;

            // Advance pipeline by `kNumMathWarpGroups` steps
            // Wait UMMA arrival
            CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
            full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();

            // Reduce over the head dim and store
            #pragma unroll
            for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                // Load accumulator from TMEM
                uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                tmem_load(cute::Int<kNumHeads>{}, tmem_addr, accum);

                // Release TMEM empty
                if (i == kNextNAtom - 1) {
                    ptx::tcgen05_before_thread_sync();
                    empty_tmem_barriers[tmem_stage_idx]->arrive();
                }

                // Accumulate weighted ReLU in parallel
                auto sum_0 = make_float2(0, 0);
                auto sum_1 = make_float2(0, 0);

                const auto transform = [&](const uint32_t& j, const float2& sum) {
                    auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                    auto b = make_float2(weights[i][j], weights[i][j + 1]);
                    return __ffma2_rn(a, b, sum);
                };

                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; j += 4) {
                    sum_0 = transform(j, sum_0);
                    sum_1 = transform(j + 2, sum_1);
                }

                auto sum = __fadd2_rn(sum_0, sum_1);
                auto result = static_cast<logits_dtype_t>(sum.x + sum.y);

                // Store into the global memory
                const auto dst_offset = kv_offset + i * static_cast<uint64_t>(logits_stride);
                if constexpr(sizeof(logits_dtype_t) == 2) {
                    // Pack two adjacent bf16 lanes into uint32 for wider store
                    uint16_t my_bits = *reinterpret_cast<const uint16_t*>(&result);
                    uint16_t neighbor_bits = __shfl_down_sync(0xffffffff, my_bits, 1);
                    uint32_t packed;
                    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(my_bits), "h"(neighbor_bits));
                    if (lane_idx % 2 == 0)
                        *reinterpret_cast<uint32_t*>(logits + dst_offset) = packed;
                } else {
                    logits[dst_offset] = result;
                }
                // this sync warp prevent the next load tmem from reordering
                // nvcc may reorder it to overlap with the current tmem load, lead to large register usage
                __syncwarp();
            }
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh': r"""#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_fp4_gemm_1d1d_impl(int* grouped_layout,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT((kGemmType != GemmType::KGroupedContiguous) or kGranKA == kGranKB, "K-grouped SF requires kGranKA == kGranKB");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);;
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kGranKA * 4>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
                                         SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

                // Issue SFA and SFB TMAs at certain stages
                // No swizzling, so one TMA for one SF is enough
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::SF_K>(
                        shape_sfa_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
                        shape_sfb_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx);
                    tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                // Arrive at full barriers
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                if (cute::elect_one_sync()) {
                    // Do SF copy at certain stages
                    // TODO: process shared memory descriptor by addition
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                    }

                    // Issue UMMA
                    using mma_t = cute::conditional_t<
                        kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const uint32_t sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = kSwapAB ?
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id):
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       kTmemStartColOfSFB, kTmemStartColOfSFA);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       kTmemStartColOfSFA, kTmemStartColOfSFB);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }

                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh': r"""#pragma once

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // Workspaces
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk, BLOCK_M);

    // Token and buffer layouts
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Registered inputs
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 32 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    // NOTES: GEMM shared memory starts after the dispatch region, aligned to 1024 bytes
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE
    );

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SF shared memory: SFA and SFB per pipeline stage
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Epilogue amax reduction shared memory
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i); });
    auto combine_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + i); });
    auto tmem_ptr_in_smem       = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumEpilogueWarps * 2);

    // A cluster sync is essential for 2CTA tensor memory allocation
    cute::cluster_sync();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at all CTAs
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                tmem_full_barriers[i]->init(1);
                // Arrive only at the leader CTA
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    cute::cluster_sync();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Adjust registers
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // Grid sync
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        // Cache expert token counts in registers (same pattern as scheduler)
        scheduler.fetch_expert_recv_count();

        // Per-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;

                // Update pool block offset for the new expert
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);

                // Move start and end to the next expert
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                // Compute active count and min across all ranks
                // NOTES: reduce within each lane first, then warp-reduce once
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                // Hit in the current round
                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                // Move into the next round
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // TMA load token from remote rank into shared memory
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            // Load and store SF (overlaps with TMA token load)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_expert);
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and token data
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                // Load weights
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                // Wait for TMA token load to complete
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                // Store token to local L1 buffer via TMA
                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                // Write source metadata for combine write-back
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Wait for token TMA store to complete
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // Clean workspace for the next usage
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count
                if (thread_idx == 0)
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;

                // Clean L1 and L2 arrival stuffs
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );
    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_sfa_k = math::ceil_div(shape_k, kGranK * 4u);

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait the entire token arrival for linear 1
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            }

            uint64_t cached_l2_arrival_mask = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait current K block arrival
                if (block_phase == sched::BlockPhase::Linear2) {
                    // The L1 output's block N is halved into `BLOCK_K / 2`, so we have to wait 2 L1 blocks' arrival
                    DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                    const uint64_t needed = 3ull << (k_block_idx * 2);
                    if ((cached_l2_arrival_mask & needed) != needed) {
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        do {
                            cached_l2_arrival_mask = ptx::ld_acq_gpu(ptr);
                        } while ((cached_l2_arrival_mask & needed) != needed);
                    }
                }

                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute token offset from pool block index
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx;

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * 2 + SF_BLOCK_M * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        // UTCCP copy SFA and SFB to TMEM
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }

                        // Issue UMMA
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        uint32_t current_iter_idx = 0;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (block_phase == sched::BlockPhase::Linear1) {
                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                //   (values[0], values[2]), (values[1], values[3]),
                //   (values[4], values[6]), (values[5], values[7])
                float stored_cached_weight = 0;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 swiglu_values[kNumAtomsPerStore * 2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        DG_STATIC_ASSERT(WG_BLOCK_M % 32 == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0) {
                            stored_cached_weight = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .get_base_ptr<float>();
                        }

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Apply SwiGLU: silu(gate) * up
                        // Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction
                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());
                        if (lane_idx < 4)
                            smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `smem_amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax
                        const float2 wp_amax =
                            smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        // Cast
                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                     + i * ATOM_M * L1_OUT_BLOCK_N
                                                                     + row * L1_OUT_BLOCK_N
                                                                     + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // Store SF to `l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            // TODO: I believe the expression can be optimized
                            const uint32_t token_idx_in_expert = m_block_idx * BLOCK_M
                                + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2;
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + transform_sf_token_idx(token_idx_in_expert);
                            sf_base_ptr[k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[k_uint_idx * mn_stride + (sf_pool_token_idx + 4) * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
            } else {
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: only use first 16 lanes for address
                        // NOTES: 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // One warp per row, now the layout is different from shared memory storing
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // Read from shared memory
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results and write back
        // NOTES: reuse shared memory from start up to the barriers
        // 1 token, 1 topk latency: ~3 us
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        // Iterate over all tokens
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            // Iterate all chunks
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        // Move
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // Load
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // Load the first selection
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // Prefetch next top-k into the buffer while current is being accumulated
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // Accumulate
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                // Cast
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // Wait share memory release and write
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA store the token chunk
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k, const uint32_t stride_logits,
                          uint32_t* cu_seq_len_k_start,
                          uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // Normally, `h (kNumHeads) == 32` and `d (kHeadDim) == 64`
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Types
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // TMA configs
    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    // Tensor memory allocation
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);

    // Initialize barriers
    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Block scheduler
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, math::ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Prefetch
        const auto issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                            smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Issue UMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto tmem_start = warpgroup_idx * UMMA_N;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Local register buffers
        float weights[BLOCK_Q][kNumHeads];

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

                // Wait UMMA arrival
                full_umma_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);
                ptx::tcgen05_after_thread_sync();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load accumulator from TMEM
                    float accum[kNumHeads];
                    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum);

                    // Release TMEM empty
                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_umma_barriers[warpgroup_idx]->arrive();
                    }

                    // Accumulate weighted ReLU in parallel
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    auto result = static_cast<logits_dtype_t>(scale_kv * (sum.x + sum.y));

                    // Store into the global memory
                    const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                            logits[q_offset + kv_offset - seq_k_start[i]] = result;
                    } else {
                        logits[q_offset + kv_offset] = result;
                    }
                    __syncwarp();
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_paged_mqa_logits(const uint32_t batch_size,
                                const uint32_t logits_stride, const uint32_t block_table_stride,
                                const uint32_t* context_lens, logits_dtype_t* logits,
                                const uint32_t* block_table, const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // Next-N atom configs
    static constexpr uint32_t kNextNAtom = (kNextN % 2 == 0) ? 2 : 1;
    static constexpr uint32_t kNumNextNAtoms = kNextN / kNextNAtom;
    static constexpr bool kSingleAtom = (kNumNextNAtoms == 1);

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextNAtom * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i);
    });
    constexpr auto smem_offset = SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages;
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto umma_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + i; });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + kNumMathWarpGroups + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(umma_barrier_ptr + kNumMathWarpGroups * 2);

    constexpr uint32_t kNumTmemCols = kNextNAtom * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Scheduler
    constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
    using Scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, BLOCK_KV, kNumBlocksPerSplit, kNumNextNAtoms>;
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit, "Invalid `SPLIT_KV`");

    // Q and KV pipeline
    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = kNextNAtom * kNumHeads;
    DG_STATIC_ASSERT(SPLIT_KV == UMMA_M * kNumMathWarpGroups, "Invalid `SPLIT_KV`");

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_atom_idx) {
            if (cute::elect_one_sync()) {
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_atom_idx * kNextNAtom * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_atom_idx * kNextNAtom);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize outside valid range to indicate no previous task
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx, num_kv;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_atom_idx), q_iter_idx = 1;

        uint32_t kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when (q, atom) changes
            bool prefetch_q = (q_atom_idx != next_q_atom_idx) and scheduler.exist_q_atom_idx(next_q_atom_idx + 1);

            if (q_atom_idx != next_q_atom_idx)
                kv_block_idx_ptr = 32;

            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Read KV block index
            // TODO(xuzhean): consider -1
            if (kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                const auto block_table_offset = (q_atom_idx / kNumNextNAtoms) * static_cast<uint64_t>(block_table_stride);
                kv_block_idx_storage = (kv_idx + lane_idx < num_kv)
                    ? block_table[block_table_offset + kv_idx + lane_idx] : 0;
            }
            DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `UMMA_M`");

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_atom_idx + 1);
            }

            uint32_t kv_block_idx[kNumBlocksPerSplit];
            #pragma unroll
            for (uint32_t i = 0; i < kNumBlocksPerSplit; ++ i)
                kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
            kv_block_idx_ptr += kNumBlocksPerSplit;

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumBlocksPerSplit; ++ i) {
                    tma::copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                          smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim) * i,
                                                                          0, 0, 1, kv_block_idx[i]);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx] + BLOCK_KV * i,
                                              0, kv_block_idx[i]);
                }
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv);
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 1;

        while (scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)) {
            if (q_atom_idx != next_q_atom_idx) {
                // Release previous Q empty (UMMA warp must participate to prevent
                // running ahead of math warps in the Q pipeline)
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);
            }

            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;

            // Wait KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                empty_umma_barriers[i]->wait(umma_phase);    
                ptx::tcgen05_after_thread_sync();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                    auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                    auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_q[q_stage_idx], 0, k * UMMA_K);
                    cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
            }
            umma_phase ^= 1;
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        auto scheduler = Scheduler(sm_idx, context_lens, schedule_meta);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        // Offsets
        const auto math_warpgroup_idx = warpgroup_idx;
        const auto tmem_start = math_warpgroup_idx * UMMA_N;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Local register buffers
        float weights[kNextNAtom][kNumHeads];

        // Initialize outside valid range to indicate no previous task
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 0;

        while (scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)) {
            // Q or atom changes
            if (q_atom_idx != next_q_atom_idx) {
                // Release last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; ++ j)
                        weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
                }
            }

            // Get current task indices
            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_atom_idx * kNextNAtom * static_cast<uint64_t>(logits_stride) + kv_idx * BLOCK_KV;

            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Read per-KV scales
            float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

            // Wait UMMA arrival
            full_umma_barriers[math_warpgroup_idx]->wait(umma_phase);
            ptx::tcgen05_after_thread_sync();
            umma_phase ^= 1;

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

            #pragma unroll
            for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                // Load accumulator from TMEM
                float accum[kNumHeads];
                tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum);

                // Release TMEM empty
                if (i == kNextNAtom - 1) {
                    ptx::tcgen05_before_thread_sync();
                    empty_umma_barriers[math_warpgroup_idx]->arrive();
                }

                // Accumulate weighted ReLU in parallel
                auto sum_0 = make_float2(0, 0);
                auto sum_1 = make_float2(0, 0);

                const auto transform = [&](const uint32_t& j, const float2& sum) {
                    auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                    auto b = make_float2(weights[i][j], weights[i][j + 1]);
                    return __ffma2_rn(a, b, sum);
                };

                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; j += 4) {
                    sum_0 = transform(j, sum_0);
                    sum_1 = transform(j + 2, sum_1);
                }

                auto sum = __fadd2_rn(sum_0, sum_1);
                auto result = static_cast<logits_dtype_t>(scale_kv * (sum.x + sum.y));

                // Store into the global memory
                logits[kv_offset + i * static_cast<uint64_t>(logits_stride) + math_thread_idx] = result;
                __syncwarp();
            }
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh': r"""#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE
uint32_t get_swizzled_smem_offset(const uint32_t& offset, const uint32_t& lane_idx) {
    // Calculate the index of the bank group to be written in the atom
    const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);

    // Reshape the atom in another view and swizzle
    //  - original: `(BLOCK_N, kSwizzleMode / kSwizzleBase)`
    //  - new: `(BLOCK_N * kSwizzleMode / kSwizzleBase / kNumBankGroups, kNumBankGroups)`
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);

    return row * 128 + col * kSwizzleBase;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMMAThreads, uint32_t kNumCastAndReduceThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumMMAThreads + kNumCastAndReduceThreads, 1)
sm100_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                                float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Configs
    constexpr uint32_t kNumCastStages = 2;
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    constexpr auto kMajorA = cute::UMMA::Major::K;
    constexpr auto kMajorB = cute::UMMA::Major::K;
    DG_STATIC_ASSERT(kNumCastStages <= kNumStages, "Invalid cast stages");
    DG_STATIC_ASSERT(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMMAThreads == 128, "Invalid MMA threads");

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Data on shared memory (layout as ordered below)
    // Fill D/A/B pointers
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers           = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_cast_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto empty_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto empty_cast_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_full_barrier       = barrier_start_ptr + kNumStages * 4;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 4 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            full_cast_barriers[i]->init(kNumCastAndReduceThreads);
            empty_barriers[i]->init(1);
            empty_cast_barriers[i]->init(1);
        }
        tmem_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Dispatch warps into different roles
    if (warp_idx < kNumMMAThreads / 32) {
        // TMA load warp
        if (warp_idx == 0 and cute::elect_one_sync()) {
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait consumer release
                const auto& stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

                // Compute offsets
                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t k_idx = k_offset + s * BLOCK_K;

                // Issue TMAs
                tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);

                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
            }
        }

        // MMA issue warp
        if (warp_idx == 1) {
            // Make instruction descriptor
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / sizeof(float);
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / sizeof(float);
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                   BLOCK_M, BLOCK_N, kMajorA, kMajorB>;
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                          UMMA_M, UMMA_N, kMajorA, kMajorB>();
            const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto b_desc = mma::sm100::make_umma_desc<kMajorB, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>(smem_b[0], 0, 0);
            const uint32_t& b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            // Checks for MMA instructions
            // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Launch MMAs
            // We can not unroll this part
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait TMA arrival
                const auto& stage_idx = s % kNumStages;
                const auto& cast_stage_idx = s % kNumCastStages;
                full_cast_barriers[cast_stage_idx]->wait((s / kNumCastStages) & 1);
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    const uint32_t& atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    const uint32_t& in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    const uint32_t& offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, float>(b_desc_base_lo, offset, in_atom_idx);
                    umma_t::fma(BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc, BLOCK_K * kNumCastStages, s > 0 or k > 0, runtime_instr_desc);
                }

                // Commit
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_cast_barriers[cast_stage_idx]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            }

            // Commit to epilogue threads
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
        }

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Only support layout F (M = 64) and D (M = 128)
        DG_STATIC_ASSERT(BLOCK_M == 64 or BLOCK_M == 128, "Invalid block M");

        // Wait UMMA arrival
        tmem_full_barrier->wait(0);
        ptx::tcgen05_after_thread_sync();

        // Load from tensor memory into registers, and write shared memory with STSM
        DG_STATIC_ASSERT(kNumMMAThreads == 128, "Epilogue threads not enough");

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerBankGroup; ++ i) {
            // Source and destination memory address
            uint32_t tmem_addr = BLOCK_K * kNumCastStages + i * kNumElemsPerBankGroup;
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) +                   // Base pointer
                            warp_idx * BLOCK_M / 4 * kSwizzleCDMode +               // Warp offset
                            get_swizzled_smem_offset<kSwizzleCDMode>(i, lane_idx);  // In-atom offset

            // Load from tensor memory, store into shared memory
            uint32_t values[kNumElemsPerBankGroup];
            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            if (BLOCK_M == 128 or (BLOCK_M == 64 and lane_idx < 16))
                ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
            if constexpr (BLOCK_M == 64)
                __syncwarp();
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);
        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            cute::tma_store_arrive();
        }

        // Deallocate tensor memory by warp 1
        // NOTES: warp 0 is waiting TMA store
        if (warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    } else {
        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(kNumCastAndReduceThreads == 128, "Invalid cast-and-reduce threads");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;

        // TODO: make even larger block K
        DG_STATIC_ASSERT(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "Invalid block K");

        // Launch reductions
        float2 sum[2] = {float2{0, 0}, float2{0, 0}};
        #pragma unroll kNumStages
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Load from shared memory into tensor memory using movement shape `.16x256b` (shared memory part is 128b)
            constexpr uint32_t kNumBankGroupBytes = 16;
            constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
            constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
            const auto& smem_base_ptr = reinterpret_cast<uint8_t*>(smem_a[stage_idx]) +    // Base pointer
                                        sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;   // Warp offset

            // 4 lanes shared a bank group
            uint32_t uint32_values[2][kNumLoads];
            DG_STATIC_ASSERT(kNumLoads % 2 == 0, "Invalid number of loads");
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; i += 2) {
                auto smem_ptr = smem_base_ptr + get_swizzled_smem_offset<kSwizzleAMode>(i + lane_idx / 16, lane_idx % 16);
                ptx::SM90_U32x4_LDSM_N::copy(uint32_values[0][i + 0], uint32_values[1][i + 0],
                                             uint32_values[0][i + 1], uint32_values[1][i + 1],
                                             smem_ptr);
            }

            // Wait tensor memory empty
            const auto& cast_stage_idx = s % kNumCastStages;
            empty_cast_barriers[cast_stage_idx]->wait(((s / kNumCastStages) & 1) ^ 1);

            // Cast, reduce and store into tensor memory
            float2 fp32x2_values[2][kNumLoads];
            const auto& upper_view = reinterpret_cast<uint32_t*>(&fp32x2_values[0]);
            const auto& lower_view = reinterpret_cast<uint32_t*>(&fp32x2_values[1]);
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                #pragma unroll
                for (uint32_t u = 0; u < 2; ++ u) {
                    fp32x2_values[u][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&uint32_values[u][i]));
                    sum[u] = __ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);
                }

                // Store upper and lower part at the same time
                const auto idx_0 = i * 2, idx_1 = i * 2 + 1;
                cute::SM100_TMEM_STORE_16dp256b1x::copy(
                    upper_view[idx_0], upper_view[idx_1],
                    lower_view[idx_0], lower_view[idx_1],
                    cast_stage_idx * BLOCK_K + i * 8);
            }
            cutlass::arch::fence_view_async_tmem_store();

            // Arrive for issuing MMAs
            ptx::tcgen05_before_thread_sync();
            full_cast_barriers[cast_stage_idx]->arrive();
        }

        // Intra-warp reduction and write back
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++ u) {
            const auto reduced_sum = math::warp_reduce_sum<4>(sum[u].x + sum[u].y);
            const auto m_idx = m_block_idx * BLOCK_M + sub_warp_idx * BLOCK_M_PER_WARP + lane_idx / 4 + u * 8;
            if (lane_idx % 4 == 0 and m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum;
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh': r"""#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_gemm_impl(int* grouped_layout,
                    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `warpgroup_wait<0>()` overhead
    constexpr uint32_t kDoMergeStages =
        kNumStages_ >= 10 and
        kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K and
        kNumMathThreads == 128;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 5;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    // Types
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorB>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B/D must be aligned to 1024 bytes");

    // D/A/B shared memory
    auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 48;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];

                    const auto m_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    const auto n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // Issue TMAs
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                    if constexpr (kMajorA == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorA == cute::UMMA::Major::MN)
                        tma::copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_a_idx, num_tma_multicast_a, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], k_b_idx, n_idx, num_tma_multicast_b, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::MN)
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], n_idx, k_b_idx, num_tma_multicast_b, batch_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // TODO: remove some useless computation for unaligned Ms
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                // Wait TMA arrivals
                full_barriers[stage_idx]->wait(phase);

                // Commit WGMMA instructions
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorB, BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode, nv_bfloat16>(
                            b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M); ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Notify barrier arrival
                empty_barrier_arrive(stage_idx);
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                // Write back to shared memory using STSM and issue TMA stores
                DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
                DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        // Swizzle or padding into the correct address
                        uint8_t* smem_ptr = nullptr;
                        if constexpr (kSwizzleDMode > 0) {
                            // Calculate the swizzling atom offset and in-atom offset
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                            //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);

                            // Add back into the base pointer
                            // NOTES: think twice before modifying this, as changes may affect the number of instructions
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                                m_offset * kSwizzleDMode +                                 // Wave offset
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                        } else {
                            // No swizzling
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }

                        // NOTES: only 16 lanes' addresses are used
                        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
            } else {
                // Use `st.shared` if STSM is not available
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    auto smem_d_0 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 0) * BLOCK_N + (lane_idx % 4) * 2);
                    auto smem_d_1 = reinterpret_cast<float2*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx / 4 + 8) * BLOCK_N + (lane_idx % 4) * 2);
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        ptx::st_shared(smem_d_0 + i * 4, make_float2(shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]));
                        ptx::st_shared(smem_d_1 + i * 4, make_float2(shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]));
                    }
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);

            // Use TMA store to write back to global memory
            const auto m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_cd, smem_ptr,
                                                  n_block_idx * BLOCK_N + in_block_n_offset,
                                                  m_idx, scheduler.current_group_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                        cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr,
                                     n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_bmk_bnk_mn.cuh': r"""#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bmn_bnk_mn_gemm_impl(const uint32_t shape_s,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                          float *d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Types
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Shared memory
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == 128, "Invalid block M");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid number of TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    // Fill shared memory pointers
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + (i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + (kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

   // Block indices
    const uint32_t num_n_blocks = math::ceil_div(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * math::ceil_div(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages = cute::min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            #pragma unroll
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait consumer release
                const auto stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait((s / kNumStages + 1) & 1);

                auto& full_barrier = *full_barriers[stage_idx];
                const uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
                const uint32_t k_idx = sk_idx % SHAPE_K;
                const uint32_t s_idx = sk_idx / SHAPE_K;

                constexpr uint32_t kSwizzle = BLOCK_K * sizeof(nv_bfloat16);
                tma::copy<BLOCK_K, BLOCK_M, kSwizzle>(
                    &tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_block_idx * BLOCK_M + s_idx * SHAPE_M, 1);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzle>(
                    &tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_block_idx * BLOCK_N + s_idx * SHAPE_N, 1);
                full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        float accum[WGMMA::kNumAccum] = {0};

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrivals
            const auto stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Commit WGMMA instructions
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + (math_wg_idx * WGMMA::M) * BLOCK_K + k * WGMMA::K, 1);
                auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                WGMMA::wgmma(desc_a, desc_b, accum, 1);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_wait<0>();

            // Notify barrier arrival at the last warpgroup wave
            empty_barriers[stage_idx]->arrive();
        }

        const auto row = m_block_idx * BLOCK_M + warp_idx * 16 + lane_idx / 4;
        const auto col = n_block_idx * BLOCK_N + (lane_idx % 4) * 2;
        #pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
            if (col + i * 8 >= SHAPE_N)
                break;
            if (row < SHAPE_M) {
                atomicAdd(reinterpret_cast<float2*>(d + (row + 0) * SHAPE_N + col + i * 8),
                          make_float2(accum[i * 4 + 0], accum[i * 4 + 1]));
            }
            if (row + 8 < SHAPE_M) {
                atomicAdd(reinterpret_cast<float2*>(d + (row + 8) * SHAPE_N + col + i * 8),
                          make_float2(accum[i * 4 + 2], accum[i * 4 + 3]));
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh': r"""#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(__nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                        int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Tensor maps on shared and global memory
    auto smem_tensor_map_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer);
    auto smem_tensor_map_b = smem_tensor_map_a + 1;
    auto gmem_tensor_map_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tensor_map_b = gmem_tensor_map_a + 1;

    // Data on shared memory
    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE)); 
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE));
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    });

    // Barriers on shared memory
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // Load tensormap A/B to shared memory
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a = tensor_map_a_base;
            *smem_tensor_map_b = tensor_map_b_base;
        }

        // Initialize barriers
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Pipeline unroll control
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // Register reconfigurations (more math registers are needed with unrolling)
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();
    
    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, 128u>(shape_m, shape_n, shape_k, grouped_layout);

    // TMA and MMA pipeline
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
    };
    uint32_t iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;

            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
                
                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                if (kGemmType == GemmType::KGroupedContiguous && last_group_idx != scheduler.current_group_idx) {  
                    last_group_idx = scheduler.current_group_idx;  

                    // Directly update current tensor map
                    const uint64_t current_k_offset = scheduler.current_k_cumsum;
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, scheduler.current_shape_k);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b, scheduler.current_shape_k, scheduler.current_shape_k);
                    *(gmem_tensor_map_a) = *(smem_tensor_map_a);  
                    *(gmem_tensor_map_b) = *(smem_tensor_map_b);  
                    ptx::tensor_map_release_gpu();

                    // Immediately acquire current tensor map
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a);
                    ptx::tensor_map_acquire_gpu(gmem_tensor_map_b);
                }

                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t sf_k_idx = scheduler.current_sf_k_cumsum + k_block_idx;
                    const auto tensor_map_a_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_a : &tensor_map_a_base);
                    const auto tensor_map_b_ptr = (kGemmType == GemmType::KGroupedContiguous ? gmem_tensor_map_b : &tensor_map_b_base);
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b_ptr, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Accumulation for WGMMA or CUDA promotion
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
            float2 scales_b[WGMMA::kNumAccum / 4];

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // Wait TMA arrivals
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // Read A scales
                // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);

                // Read B scales
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = ptx::ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));

                // Commit WGMMA instructions
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Notify barrier arrival
                empty_barrier_arrive(stage_idx);

                // Promote with scales
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float &scale_b_0 = scales_b[i].x;
                    const float &scale_b_1 = scales_b[i].y;
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                }
            }

            // Flush previous stores
            if (warp_idx % 4 == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // Store to D shared memory
            const auto smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
            const auto smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                ptx::st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                ptx::st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // Use TMA store to write back to global memory
            if (warp_idx % 4 == 0 and cute::elect_one_sync()) {
                cute::SM90_TMA_REDUCE_ADD_2D::copy(
                    &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                    current_group_idx * shape_m + m_block_idx * BLOCK_M + r_0);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh': r"""#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = math::align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                             smem_sfa[stage_idx], m_block_idx * BLOCK_M, scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales, 1, k_block_idx),
                             num_tma_multicast_a);

                    // Issue TMA B
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    ptx::st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb] : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
            
            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // Read B scales
                        float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx), scale_b_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);

                        // Wait TMA arrivals
                        full_barriers[stage_idx]->wait(phase);

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            auto scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            auto scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                         const uint32_t max_seqlen_k, const uint32_t stride_logits,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         logits_dtype_t* logits,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Initialize barriers
    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    const auto sm_idx = blockIdx.x;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, math::ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp remains
        if (not is_tma_load_warp)
            return;

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = ptx::get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // Issue WGMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[q_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Store into the global memory
                    const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset + v_0_offset and kv_offset + v_0_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_0_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_0);
                        if (seq_k_start[i] <= kv_offset + v_1_offset and kv_offset + v_1_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_1_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_1);
                    } else {
                        logits[q_offset + kv_offset + v_0_offset] = static_cast<logits_dtype_t>(v_0);
                        logits[q_offset + kv_offset + v_1_offset] = static_cast<logits_dtype_t>(v_1);
                    }
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_paged_mqa_logits(const uint32_t batch_size,
                               const uint32_t logits_stride, const uint32_t block_table_stride,
                               const uint32_t* context_lens, logits_dtype_t* logits,
                               const uint32_t* block_table, const uint32_t* schedule_meta,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<kNextN * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors
    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = math::constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 math::constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  math::constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q data and barriers on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    // Separate math warpgroups and tma load warps into KV groups
    // Each math warpgroup corresponds to a tma load warp
    const auto kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    // Per group KV data and barriers on shared memory
    const auto smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    // Initialize barriers
    if (warp_idx >= kNumMathThreads / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 64;
    constexpr uint32_t kNumMathRegisters = 104;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Scheduler
    auto scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, BLOCK_KV, kNumMathWarpGroups, 1>(
        blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    // Q and KV pipeline
    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma::copy<kHeadDim, kNextN * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma::copy<kNextN * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_idx * kNextN);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current Q changes
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_atom_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    block_table[q_idx * static_cast<uint64_t>(block_table_stride) + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)] : 0);
            }
            const auto kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            // Issue TMA KV
            if (cute::elect_one_sync()) {
                tma::copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                      smem_kv[kv_stage_idx], 0, 0, 1, kv_block_idx);
                tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                          smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        float accum[WGMMA::kNumAccum], weights[kNextN][kNumHeads / 4];
        const auto sub_warp_offset = (warp_idx % 4) * 16;
        const auto v_0_offset = lane_idx / 4 + 0;
        const auto v_1_offset = lane_idx / 4 + 8;

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current Q changes
            if (q_idx != next_q_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            // Get current Q and KV index
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_idx * kNextN * static_cast<uint64_t>(logits_stride) + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Compute `[kNextN * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [kNextN, BLOCK_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Issue WGMMA
            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                auto desc_a = mma::sm90::make_smem_desc(
                    smem_kv[kv_stage_idx] + k * WGMMA::K,
                    mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                auto desc_b = mma::sm90::make_smem_desc(
                    smem_q[q_stage_idx] + k * WGMMA::K,
                    mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);

            // Read per-KV scales
            float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            // Wait WGMMA
            ptx::warpgroup_wait<0>();

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
            DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == kNextN, "Invalid accumulation");
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                auto shifted_accum = accum + i * kNumAccumPerReduce;
                const auto transform = [&](const uint32_t& j) {
                    return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                };

                // Intra-thread reduction
                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; k ++)
                        sum[k] += transform(j * 4 + k);
                }
                float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                // Inter-thread reduction
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto offset = static_cast<int>(1u << j);
                    v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                    v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                }

                // Store into the global memory
                // NOTES: we have redundant writes here, consider more carefully
                logits[kv_offset + i * static_cast<uint64_t>(logits_stride) + v_0_offset] = static_cast<logits_dtype_t>(v_0);
                logits[kv_offset + i * static_cast<uint64_t>(logits_stride) + v_1_offset] = static_cast<logits_dtype_t>(v_1);
            }
        }
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh': r"""#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE
uint32_t get_swizzled_bank_group_idx(const uint32_t& offset, const uint32_t& lane_idx) {
    constexpr uint32_t kGroupsInSwizzleRange = kSwizzleMode / kSwizzleBase;

    const auto bank_group_idx = offset + lane_idx * kGroupsInSwizzleRange;

    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = kGroupsInSwizzleRange == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % kGroupsInSwizzleRange;

    return (row * kNumBankGroups + col) % kGroupsInSwizzleRange;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
sm90_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                               float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // kSwizzleAMode and kSwizzleBMode must be 128 for now
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");
    DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");
    DG_STATIC_ASSERT(kSwizzleBMode == 128, "Invalid swizzle B mode");

    DG_STATIC_ASSERT(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMathThreads == 128, "Invalid MMA threads");

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Data on shared memory (layout as ordered below)
    // Fill D/A/B pointers
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers           = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(128);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 256;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // TMA load warp
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait consumer release
            const auto stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

            // Compute offsets
            uint32_t m_idx = m_block_idx * BLOCK_M;
            uint32_t k_idx = k_offset + s * BLOCK_K;

            // Issue TMAs
            tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
            tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);

            // Arrive at full barriers
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }

        for (uint32_t s = num_total_stages; s < num_total_stages + kNumStages; ++ s) {
            const auto stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
        }
    } else if (warp_idx < kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "Invalid block K");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        constexpr uint32_t WGMMA_M = 64;
        constexpr uint32_t WGMMA_N = BLOCK_N;
        constexpr uint32_t WGMMA_K = 8;

        using WGMMA = typename mma::sm90::TF32MMASelector<WGMMA_N, true>::type;
        float accum[WGMMA::kNumAccum] = {0};

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
        float sqr_sum_acc_0 = 0;
        float sqr_sum_acc_1 = 0;

        #pragma unroll kNumStages < 8 ? kNumStages : kNumStages / 2
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            constexpr uint32_t kNumRegPerWgmma = WGMMA::M * WGMMA::K / 128;
            constexpr uint32_t kNumWgmmaPerBlockK = BLOCK_K / WGMMA::K;

            float a[kNumRegPerWgmma * kNumWgmmaPerBlockK];
            // Assume swizzle A mode is 128
            DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");

            // Load BF16 A fragment from shared memory into registers, and transpose to FP32
            uint32_t row = warp_idx * 16 + lane_idx / 4;
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                // Refer to the A layout in https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n8-a
                uint32_t bank_group_idx = (row ^ i) % 8;
                nv_bfloat16* a_bf16_smem_ptr_upper = smem_a[stage_idx] + row * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;
                nv_bfloat16* a_bf16_smem_ptr_lower = smem_a[stage_idx] + (row + 8) * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;

                uint32_t elem_offset = lane_idx % 4;
                nv_bfloat16 a_bf16[kNumRegPerWgmma];
                a_bf16[0] = a_bf16_smem_ptr_upper[elem_offset];
                a_bf16[2] = a_bf16_smem_ptr_upper[elem_offset + 4];
                a_bf16[1] = a_bf16_smem_ptr_lower[elem_offset];
                a_bf16[3] = a_bf16_smem_ptr_lower[elem_offset + 4];

                auto a_bf16x2_ptr = reinterpret_cast<nv_bfloat162*>(a_bf16);
                auto a_float2_ptr = reinterpret_cast<float2*>(a);
                float2 a_float2_0 = __bfloat1622float2(a_bf16x2_ptr[0]);
                float2 a_float2_1 = __bfloat1622float2(a_bf16x2_ptr[1]);
                a_float2_ptr[i * 2 + 0] = a_float2_0;
                a_float2_ptr[i * 2 + 1] = a_float2_1;
                sqr_sum_acc_0 += a_float2_0.x * a_float2_0.x + a_float2_1.x * a_float2_1.x;
                sqr_sum_acc_1 += a_float2_0.y * a_float2_0.y + a_float2_1.y * a_float2_1.y;
            }

            ptx::warpgroup_wait<0>();
            if (s > 0)
                empty_barriers[(s - 1) % kNumStages]->arrive();

            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();

            constexpr int kNumElemsInSwizzleRange = 128 / sizeof(float);
            constexpr uint32_t kNumWgmmaInSwizzleRange = kNumElemsInSwizzleRange / WGMMA::K;
            DG_STATIC_ASSERT(BLOCK_K % kNumElemsInSwizzleRange == 0, "Invalid block K");

            #pragma unroll
            for (int i = 0; i < BLOCK_K / kNumElemsInSwizzleRange; i++) {
                #pragma unroll
                for (int k = 0; k < kNumElemsInSwizzleRange / WGMMA::K; k++) {
                    auto b_desc = mma::sm90::make_smem_desc(
                        smem_b[stage_idx] + i * BLOCK_N * kNumElemsInSwizzleRange + k * WGMMA::K, 1);
                    WGMMA::wgmma(a + (i * kNumWgmmaInSwizzleRange + k) * kNumRegPerWgmma, b_desc, accum, 1);
                }
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                ptx::warpgroup_fence_operand(accum[i]);
        }

        const auto& reduced_sum_0 = math::warp_reduce_sum<4>(sqr_sum_acc_0);
        const auto& reduced_sum_1 = math::warp_reduce_sum<4>(sqr_sum_acc_1);

        const auto& m_idx = m_block_idx * BLOCK_M + (warp_idx * BLOCK_M_PER_WARP + lane_idx / 4);
        if (lane_idx % 4 == 0) {
            if (m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum_0;
            if (m_idx + 8 < shape_m)
                sqr_sum[m_offset + m_idx + 8] = reduced_sum_1;
        }
        ptx::warpgroup_wait<0>();
        empty_barriers[(num_total_stages-1) % kNumStages]->arrive();

        // Write accum to shared memory
        // Every 2 threads (one pair) will write to the same bank group (16 bytes).
        // Refer to the D layout in https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n8-d
        uint32_t is_odd_pair = lane_idx / 2 % 2;

        // Four threads per group; write the data to the same row.
        uint32_t row_idx = lane_idx / 4;

        // Even/odd index pairs write to the same column, we need to reorder idx:
        // group even pair indices consecutively, and likewise for odd ones.
        uint32_t reordered_pair_idx = is_odd_pair * 8 + row_idx;

        auto shifted_smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) +
                                (warp_idx * BLOCK_M_PER_WARP + row_idx) * kSwizzleCDMode +  // Row offset, each warp has 16 rows
                                lane_idx % 2 * 8;                                           // One thread of a pair writes 8 bytes

        #pragma unroll
        for (uint32_t i = 0; i < (kSwizzleCDMode / sizeof(float)) / 4; i += 2) {
            // Get the swizzled bank group index (16 bytes per group)
            uint32_t bank_group_idx = get_swizzled_bank_group_idx<kSwizzleCDMode>(i + is_odd_pair, reordered_pair_idx);
            auto smem_ptr = shifted_smem_ptr + bank_group_idx * kNumBankGroupBytes; // Col offset, 16 bytes per group

            // 0/1 write to the same row, 2/3 write to another row
            auto values = reinterpret_cast<uint32_t*>(accum + i * 2);
            ptx::st_shared(smem_ptr, values[0], values[1]);
            ptx::st_shared(smem_ptr + 8 * kSwizzleCDMode, values[2], values[3]);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, 1);

        // Issue TMA stores
        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            cute::tma_store_arrive();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
""",
    'key-files/deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t BLOCK_KV, uint32_t kNumWarps, typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumWarps * 32, 1)
void smxx_clean_logits(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_logits,
                       const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end, logits_dtype_t* logits) {
    const uint32_t num_sms = gridDim.x;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    constexpr uint32_t kAlignment = 16 / sizeof(logits_dtype_t);
    const logits_dtype_t neg_inf = -cute::numeric_limits<logits_dtype_t>::infinity();

    // Allocate filled `-inf` shared memory
    extern __shared__ __align__(1024) logits_dtype_t smem_buffer[];
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < BLOCK_KV; i += kNumWarps * 32)
        smem_buffer[i] = neg_inf;
    cute::tma_store_fence();
    __syncthreads();

    // Assign sequence to each warp
    const auto assign_task = [&](const uint32_t& num, const uint32_t& idx,
                                 const uint32_t& start, const uint32_t& total) -> cute::tuple<uint32_t, uint32_t> {
        const auto per = total / num, rem = total % num;
        return {start + idx * per + cute::min(idx, rem), per + (idx < rem)};
    };
    CUTE_TIE_DECL(assign_task(num_sms, sm_idx, 0, seq_len), sm_seq_start, sm_seq_len);
    CUTE_TIE_DECL(assign_task(kNumWarps, warp_idx, sm_seq_start, sm_seq_len), warp_seq_start, warp_seq_len);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (cute::elect_one_sync()) {
        for (uint32_t i = warp_seq_start; i < warp_seq_start + warp_seq_len; ++ i) {
            const auto ks = cu_seq_len_k_start == nullptr ? 0 : cu_seq_len_k_start[i / kNextN];
            const auto ke = cu_seq_len_k_end[i / kNextN] - kNextN + i % kNextN + 1;
            const auto aligned_ks = ks / kAlignment * kAlignment, aligned_ke = (ke + kAlignment - 1) / kAlignment * kAlignment;

            for (uint32_t left = 0; left < seq_len_kv; left += BLOCK_KV) {
                const auto right = cute::min(left + BLOCK_KV, static_cast<uint32_t>(stride_logits));
                if (right <= ks or ke <= left) {
                    cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + left, (right - left) * sizeof(logits_dtype_t));
                } else {
                    if (left < aligned_ks)
                        cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + left, (aligned_ks - left) * sizeof(logits_dtype_t));
                    if (aligned_ke < right)
                        cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + aligned_ke, (right - aligned_ke) * sizeof(logits_dtype_t));
                }
            }
        }
    }
    __syncwarp();

    for (uint32_t i = warp_seq_start; i < warp_seq_start + warp_seq_len; ++ i) {
        const auto ks = cu_seq_len_k_start == nullptr ? 0 : cu_seq_len_k_start[i / kNextN];
        const auto ke = cu_seq_len_k_end[i / kNextN] - kNextN + i % kNextN + 1;
        const auto aligned_ks = ks / kAlignment * kAlignment, aligned_ke = (ke + kAlignment - 1) / kAlignment * kAlignment;
        for (uint32_t j = aligned_ks; j < ks; ++ j)
            logits[i * stride_logits + j] = neg_inf;
        for (uint32_t j = ke; j < aligned_ke; ++ j)
            logits[i * stride_logits + j] = neg_inf;
    }
}

}
""",
    'key-files/deep_gemm/include/deep_gemm/impls/smxx_layout.cuh': r"""#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K,
          uint32_t PADDED_SF_K = SF_K + (1 - (SF_K % 2))>
CUTLASS_GLOBAL void transpose_fp32(const float* sf, float* out, const uint32_t mn) {
    typedef typename utils::Vectorized<sizeof(float) * SF_K>::vec_t in_vec_t;
    constexpr static uint32_t kNumElemsPerVec = sizeof(in_vec_t) / sizeof(float);
    constexpr static uint32_t SF_VEC_K = SF_K / kNumElemsPerVec;

    // Shapes and strides
    extern __shared__ float smem_buffer[];
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(float));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = math::align<uint32_t>(mn, kNumTMAAlignedElems);

    // Shift into the block
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * SF_K;
    const auto& local_sf = reinterpret_cast<const in_vec_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Load
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_VEC_K; i += kNumThreads) {
        auto in_vec = local_sf[i];
        const auto& in_values = reinterpret_cast<float*>(&in_vec);

        const auto& row = i / SF_VEC_K, col = (i % SF_VEC_K) * kNumElemsPerVec;
        #pragma unroll
        for (uint32_t j = 0; j < kNumElemsPerVec; ++ j)
            smem_buffer[row * PADDED_SF_K + col + j] = in_values[j];
    }
    __syncthreads();

    // Store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_K; i += kNumThreads) {
        const auto& sf_k_idx = i / in_block_mn, mn_idx = i % in_block_mn;
        const auto& global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx;
        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ptx::ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);
    }
}

// NOTES: the two kernels below always pack the K dimension

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K>
CUTLASS_GLOBAL void transpose_and_pack_fp32_into_ue8m0(float* sf, uint32_t* out, const uint32_t mn) {
    extern __shared__ uint32_t smem_buffer[];

    // Shapes and strides
    constexpr auto kNumPackedSFK = math::constexpr_ceil_div(SF_K, 4u);
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(int));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = math::align<uint64_t>(mn, kNumTMAAlignedElems);

    // Shift into the group
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * kNumPackedSFK;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Load FP32 SFs
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block size");
    const auto local_sf = reinterpret_cast<uint32_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));
    const auto num_values = in_block_mn * SF_K;
    const auto num_uint4 = num_values / 4;
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < num_uint4; i += kNumThreads) {
        const auto& [x, y, z, w] = reinterpret_cast<const uint4*>(local_sf)[i];
        ptx::st_shared(reinterpret_cast<uint4*>(smem_buffer) + i, x, y, z, w);
    }

    // Fill unaligned values as well
    if (const auto unaligned_idx = num_uint4 * 4 + threadIdx.x; unaligned_idx < num_values)
        ptx::st_shared(smem_buffer + unaligned_idx, local_sf[unaligned_idx]);
    __syncthreads();

    // Pack into UE8M0 and store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < (kNumPackedSFK * BLOCK_MN); i += kNumThreads) {
        const auto sf_k_pack_idx = i / BLOCK_MN, mn_idx = i % BLOCK_MN;

        // Load shared memory
        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const auto sf_k_idx = sf_k_pack_idx * 4 + j;
            values[j] = sf_k_idx < SF_K ? ptx::ld_shared(smem_buffer + mn_idx * SF_K + sf_k_idx) : 0;
        }

        // Pack and store
        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >>  7u);
        packed |= (values[3] <<  1u);
        if (const auto global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx; global_mn_idx < mn)
            out[sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}

template <uint32_t kNumGroups, uint32_t kNumThreads,
          uint32_t BLOCK_MN, uint32_t BLOCK_PACKED_SF_K, bool kTransposed = true>
CUTLASS_GLOBAL void pack_fp32_into_ue8m0(float* sf, uint32_t* out, uint32_t* ks,
                                         const uint32_t mn, uint32_t sf_k, const uint32_t packed_sf_k,
                                         const uint32_t gran_k) {
    // Always packing the K dimension
    // NOTES: should also assert `mn % 4 == 0` at launch
    DG_STATIC_ASSERT(kTransposed, "Currently only support transposed SFs (MN-major)");
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_PACKED_SF_K == kNumThreads / 32, "Invalid block sizes");

    // Shapes and strides
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto in_block_mn_uint4 = in_block_mn / 4;
    const auto in_block_packed_sf_k = min(BLOCK_PACKED_SF_K, packed_sf_k - blockIdx.y * BLOCK_PACKED_SF_K);

    // Shift into the right block along MN
    sf += blockIdx.x * BLOCK_MN;
    out += blockIdx.x * BLOCK_MN;

    // Each warp is responsible for a packed row
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    const auto packed_sf_k_idx = static_cast<uint64_t>(blockIdx.y) * BLOCK_PACKED_SF_K + warp_idx;
    if (warp_idx >= in_block_packed_sf_k)
        return;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Make an offset on the input
    uint32_t input_offset = 0;
    if constexpr (kNumGroups > 1) {
        // Load each group's size
        DG_STATIC_ASSERT(kNumGroups <= 128, "Too many groups");
        uint32_t group_ks[4];
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i) {
            const auto group_idx = lane_idx * 4 + i;
            group_ks[i] = group_idx < kNumGroups ? ks[group_idx] : 0;
        }
        __syncwarp();

        // Make the offset
        sf_k = 0;
        uint32_t sum_packed_sf_k = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumGroups; ++ i) {
            const auto sf_k_in_group = __shfl_sync(0xffffffff, group_ks[i % 4] / gran_k, i / 4);
            sf_k += sf_k_in_group;
            sum_packed_sf_k += math::ceil_div(sf_k_in_group, 4u);
            if (packed_sf_k_idx < sum_packed_sf_k)
                break;
            if (const auto remainder = sf_k_in_group % 4; remainder > 0)
                input_offset += 4 - remainder;
        }
    }

    for (uint32_t mn_idx = ptx::get_lane_idx(); mn_idx < in_block_mn_uint4; mn_idx += 32) {
        // Load
        uint4 values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            values[j] = make_uint4(0, 0, 0, 0);
            if (const auto sf_k_idx = packed_sf_k_idx * 4 + j - input_offset; sf_k_idx < sf_k)
                values[j] = reinterpret_cast<const uint4*>(sf + sf_k_idx * mn)[mn_idx];
        }

        // Pack and store
        uint4 packed;
        packed.x = (values[0].x >> 23u) | (values[1].x >> 15u) | (values[2].x >> 7u) | (values[3].x << 1u);
        packed.y = (values[0].y >> 23u) | (values[1].y >> 15u) | (values[2].y >> 7u) | (values[3].y << 1u);
        packed.z = (values[0].z >> 23u) | (values[1].z >> 15u) | (values[2].z >> 7u) | (values[3].z << 1u);
        packed.w = (values[0].w >> 23u) | (values[1].w >> 15u) | (values[2].w >> 7u) | (values[3].w << 1u);
        reinterpret_cast<uint4*>(out + packed_sf_k_idx * mn)[mn_idx] = packed;
    }
}

} // namespace deep_gemm
""",
    'key-files/deep_gemm/include/deep_gemm/layout/mega_moe.cuh': r"""#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank, T block_m) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (block_m - 1),
        block_m);
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& block_m):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank, block_m);
        num_max_pool_blocks = num_max_pool_tokens / block_m;
        DG_UNIFIED_ASSERT(num_max_tokens_per_rank % block_m == 0);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        // Pad L1 entry count to even so that the `l2_arrival_mask` is 8-byte aligned
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& max_num_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

} // namespace deep_gemm::layout
""",
    'key-files/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh': r"""#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout
""",
    'key-files/deep_gemm/include/deep_gemm/mma/sm100.cuh': r"""#pragma once

#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>

namespace deep_gemm::mma::sm100 {

/// Shared memory descriptor
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    cute::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // Start address
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // Base offset
    desc.base_offset_ = 0;

    // SBO and LBO
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // NOTES: the UTCCP layout is K-major by default
    // Atom size: 8 x 128 bits
    // {SBO, LBO} means the byte stride between atoms on {MN, K}
    // Since the UTCCP we used is 128b-wide (only 1 atom on K), so LBO can be zero
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

CUTLASS_DEVICE
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

CUTLASS_DEVICE
static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type) {
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

/// UMMA descriptors
// ReSharper disable once CppNotAllPathsReturnValue
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    // A special case
    if constexpr ((cute::is_same_v<dtype_t, float> and kMajorMode == cute::UMMA::Major::MN) or kUseBase32) {
        DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
        return cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
    }

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return cute::UMMA::LayoutType::SWIZZLE_128B;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_umma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K ? 1 : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {
    return base + (((offset + k_idx * get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, bool kUseBase32 = false, typename dtype_t>
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be the same as `BLOCK_K * sizeof(dtype_t)`
        // also, atom index must be 0, so that each block has exactly one swizzle atom on the K axis
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    } else {
        constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    }
}

CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptorBlockScaled& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptor& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

} // namespace deep_gemm::mma::sm100
""",
    'key-files/deep_gemm/include/deep_gemm/mma/sm90.cuh': r"""#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::mma::sm90 {

/// MMA
template <int N_, typename MMA>
struct FP8MMA {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 40) return MMA_64x40x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 48) return MMA_64x48x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 56) return MMA_64x56x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 72) return MMA_64x72x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 80) return MMA_64x80x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 88) return MMA_64x88x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 96) return MMA_64x96x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 104) return MMA_64x104x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 112) return MMA_64x112x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 120) return MMA_64x120x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 136) return MMA_64x136x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 144) return MMA_64x144x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 152) return MMA_64x152x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 160) return MMA_64x160x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 168) return MMA_64x168x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 176) return MMA_64x176x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 184) return MMA_64x184x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 192) return MMA_64x192x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 200) return MMA_64x200x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 208) return MMA_64x208x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 216) return MMA_64x216x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 224) return MMA_64x224x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 232) return MMA_64x232x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 240) return MMA_64x240x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 248) return MMA_64x248x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
    }

    static constexpr auto select_type() {
        return FP8MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct BF16MMA {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <cute::UMMA::Major kMajor>
constexpr cute::SM90::GMMA::Major to_sm90_major() {
    DG_STATIC_ASSERT(kMajor == cute::UMMA::Major::K or kMajor == cute::UMMA::Major::MN, "Invalid major-ness");
    return kMajor == cute::UMMA::Major::K ? cute::SM90::GMMA::Major::K : cute::SM90::GMMA::Major::MN;
}

template <int N,
          cute::UMMA::Major kMajorA = cute::UMMA::Major::K,
          cute::UMMA::Major kMajorB = cute::UMMA::Major::K>
struct BF16MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto kGMMAMajorA = to_sm90_major<kMajorA>();
        constexpr auto kGMMAMajorB = to_sm90_major<kMajorB>();
        if constexpr (N == 8) return MMA_64x8x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 16) return MMA_64x16x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 24) return MMA_64x24x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 32) return MMA_64x32x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 40) return MMA_64x40x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 48) return MMA_64x48x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 56) return MMA_64x56x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 64) return MMA_64x64x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 72) return MMA_64x72x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 80) return MMA_64x80x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 88) return MMA_64x88x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 96) return MMA_64x96x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 104) return MMA_64x104x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 112) return MMA_64x112x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 120) return MMA_64x120x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 128) return MMA_64x128x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 136) return MMA_64x136x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 144) return MMA_64x144x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 152) return MMA_64x152x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 160) return MMA_64x160x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 168) return MMA_64x168x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 176) return MMA_64x176x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 184) return MMA_64x184x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 192) return MMA_64x192x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 200) return MMA_64x200x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 208) return MMA_64x208x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 216) return MMA_64x216x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 224) return MMA_64x224x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 232) return MMA_64x232x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 240) return MMA_64x240x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 248) return MMA_64x248x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
        if constexpr (N == 256) return MMA_64x256x16_F32BF16BF16_SS<kGMMAMajorA, kGMMAMajorB>();
    }

    static constexpr auto select_type() {
        return BF16MMA<N, decltype(select_mma())>();
    }

    using type = decltype(select_type());
};

template <int N_, typename MMA>
struct TF32MMARS {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint32_t* a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(a[0], a[1], a[2], a[3], desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }

    CUTLASS_DEVICE static void wgmma(float* a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(reinterpret_cast<uint32_t*>(a), desc_b, d, scale_d, cute::make_index_sequence<N_/2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 8;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N, bool kUseRS = true>
struct TF32MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (kUseRS) {
            if constexpr (N == 8) return MMA_64x8x8_F32TF32TF32_RS_TN();
            if constexpr (N == 16) return MMA_64x16x8_F32TF32TF32_RS_TN();
            if constexpr (N == 32) return MMA_64x32x8_F32TF32TF32_RS_TN();
            if constexpr (N == 64) return MMA_64x64x8_F32TF32TF32_RS_TN();
            if constexpr (N == 128) return MMA_64x128x8_F32TF32TF32_RS_TN();
            if constexpr (N == 256) return MMA_64x256x8_F32TF32TF32_RS_TN();
            DG_STATIC_ASSERT(N == 8 or N == 16 or N == 32 or N == 64 or N == 128 or N == 256, "Invalid N");
        }
    }

    static constexpr auto select_type() {
        if constexpr (kUseRS) {
            return TF32MMARS<N, decltype(select_mma())>();
        } else {
            DG_STATIC_ASSERT(kUseRS, "SS mode is not supported for TF32MMASelector for now");
        }
    }

    using type = decltype(select_type());
};

/// Shared memory descriptor
template <class PointerType>
CUTLASS_DEVICE cute::GmmaDescriptor
make_smem_desc(PointerType smem_ptr, const int& layout_type,
               const uint32_t& leading_byte_offset = 0,
               const uint32_t& stride_byte_offset = 1024) {
    // NOTES: the default LBO and SBO are for K-major types
    cute::GmmaDescriptor desc;
    const auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_gmma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K ? 1 : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

// ReSharper disable once CppNotAllPathsReturnValue
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, typename dtype_t>
constexpr static cute::SM90::GMMA::LayoutType to_gmma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 16)  return cute::SM90::GMMA::LayoutType::INTERLEAVE;
    if constexpr (kSwizzleMode == 32)  return cute::SM90::GMMA::LayoutType::B32;
    if constexpr (kSwizzleMode == 64)  return cute::SM90::GMMA::LayoutType::B64;
    if constexpr (kSwizzleMode == 128) return cute::SM90::GMMA::LayoutType::B128;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
uint32_t advance_gmma_desc_lo(const uint32_t& base, const uint32_t& mn_idx, const uint32_t& k_idx, const uint32_t& offset = 0) {
    return base + (((offset + mn_idx * BLOCK_K + k_idx * get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>()) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
cute::GmmaDescriptor make_gmma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_gmma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_gmma_layout_type<kMajorMode, kSwizzleMode, dtype_t>();
    constexpr uint32_t num_non_contiguous = 128 / 16;
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be 128B (also, atom index must be 0), as `BLOCK_K` is always 128
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k, static_cast<uint32_t>(layout_type),
                              leading_byte_offset, stride_byte_offset);
    } else {
        constexpr uint32_t BLOCK_MN_ATOM = get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k, static_cast<uint32_t>(layout_type),
                              leading_byte_offset, stride_byte_offset);
    }
}

// ReSharper disable once CppNotAllPathsReturnValue
template <uint32_t kHeadDim>
static constexpr int to_swizzle_cute_type() {
    DG_STATIC_ASSERT(kHeadDim == 32 or kHeadDim == 64 or kHeadDim == 128, "Invalid swizzling");
    if constexpr (kHeadDim == 32)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B32);
    if constexpr (kHeadDim == 64)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B64);
    if constexpr (kHeadDim == 128)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B128);
}

} // namespace deep_gemm::mma::sm90
""",
    'key-files/deep_gemm/include/deep_gemm/ptx/ld_st.cuh': r"""#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

namespace deep_gemm::ptx {

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
CUTLASS_HOST_DEVICE longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

/// LD/ST matrix
// TODO: remove `struct`
struct SM90_U32x2_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(dst_0), "=r"(dst_1)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

struct SM90_U32x4_LDSM_N {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
                     : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_T {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)),
                        "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

template <typename dtype_t>
struct SM100_U8x4_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src = *reinterpret_cast<uint32_t*>(&src_0);
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src));
    }
};

template <typename dtype_t>
struct SM100_U8x8_STSM_T {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

/// Shared memory
CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

CUTLASS_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

CUTLASS_DEVICE void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

CUTLASS_DEVICE void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}

CUTLASS_DEVICE void st_shared_bulk(void* smem_ptr, const uint32_t& num_bytes) {
    // `size` must be 64-bit before PTX ISA 9.0
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;" ::
                 "l"(__cvta_generic_to_shared(smem_ptr)), "l"(static_cast<uint64_t>(num_bytes)));
}

/// Global memory
CUTLASS_DEVICE uint64_t ld_volatile(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_sys(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE void st_relaxed_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("st.L1::no_allocate.relaxed.sys.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

/// Atomics
CUTLASS_DEVICE uint64_t atomic_add(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint64_t atomic_add_sys(const uint64_t* ptr, const uint64_t& value) {
    uint64_t ret;
    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(ret) : "l"(ptr), "l"(value));
    return ret;
}

CUTLASS_DEVICE uint32_t atomic_add_rel(const uint32_t* ptr, const uint32_t& value) {
    uint32_t ret;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__forceinline__ __device__ void red_add(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_or_rel_sys(const uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.sys.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_or_rel_gpu(uint64_t* ptr, const uint64_t& value) {
    asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(value));
}

CUTLASS_DEVICE void red_add_rel(const uint32_t* ptr, const uint32_t& value) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

CUTLASS_DEVICE int ld_acq_sys(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint32_t ld_acq_sys(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint64_t ld_acq_gpu(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

/// Predicated loads
CUTLASS_DEVICE longlong4_t ld_gez_pred(const longlong4_t* ptr, const int& pred) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, 0;\n\t"
        "  @p ld.global.L2::256B.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(pred)
        : "memory");
    return ret;
}

/// Prefetch
CUTLASS_DEVICE void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

} // namespace deep_gemm::ptx
""",
    'key-files/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh': r"""#pragma once

namespace deep_gemm::ptx {

/// UMMA versions with relaxed assertions
struct SM100_MMA_F16BF16_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F16BF16_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF8F6F4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_F8F6F4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

struct SM100_MMA_MXF4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_F16BF16_WS_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p; \n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c));
    }
};

/// Tensor memory operations
CUTLASS_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTLASS_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace deep_gemm::ptx
""",
    'key-files/deep_gemm/include/deep_gemm/ptx/tma.cuh': r"""#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_desc.hpp>

namespace deep_gemm::ptx {

// Tensor-map instructions
CUTLASS_DEVICE void tensor_map_release_gpu() {
    asm volatile ("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}

CUTLASS_DEVICE void tensor_map_acquire_gpu(const cute::TmaDescriptor* gmem_desc_ptr) {
    auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
    asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(gmem_int_desc) : "memory");
}

CUTLASS_DEVICE void tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor* smem_desc, const void* new_addr) {
    auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
    const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile ("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_int_desc), "l"(new_int64_addr));
}

CUTLASS_DEVICE void tensor_map_replace_global_inner_dim_stride_in_smem(cute::TmaDescriptor* smem_desc, const uint32_t& new_dim, const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

/// TMA instructions
CUTLASS_DEVICE void mbarrier_arrive(
    cutlass::arch::ClusterTransactionBarrier* ptr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_arrive_and_set_tx(
    cutlass::arch::ClusterTransactionBarrier* ptr, const uint32_t& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_wait_and_flip_phase(
    cutlass::arch::ClusterTransactionBarrier* ptr, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

CUTLASS_DEVICE void tma_load_1d(
    const void* dst_ptr, const void* src_ptr,
    cutlass::arch::ClusterTransactionBarrier* mbarrier_ptr,
    const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_FIRST) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))),
        "l"(hint)
        : "memory");
}

CUTLASS_DEVICE void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    // NOTES: this function does not have `.read`
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

CUTLASS_DEVICE
void tma_gather4(const void* desc_ptr, cutlass::arch::ClusterTransactionBarrier& mbarrier,
                 void* smem_ptr, const uint32_t& col_idx, const int4& row_idxs, const uint64_t& cache_hint) {
    const auto smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    const auto mbarrier_addr = cute::cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbarrier_addr), "l"(cache_hint)
        : "memory"
    );
}

} // namespace deep_gemm::ptx
""",
    'key-files/deep_gemm/include/deep_gemm/ptx/utils.cuh': r"""#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

CUTLASS_DEVICE uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

CUTLASS_DEVICE void sync_aligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

CUTLASS_DEVICE void sync_unaligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

template <typename dtype_t>
CUTLASS_DEVICE dtype_t exchange(dtype_t ptr, const uint32_t& src_lane_idx) {
    DG_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&ptr);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

CUTLASS_DEVICE void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = __bfloat1622float2(b);
    a.x += x, a.y += y;
#endif
}

} // namespace deep_gemm::ptx
""",
    'key-files/deep_gemm/include/deep_gemm/ptx/wgmma.cuh': r"""#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

} // namespace deep_gemm::ptx
""",
    'key-files/deep_gemm/include/deep_gemm/scheduler/gemm.cuh': r"""#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::sched {

enum class IndexType {
    MN,
    K,
    SF_K,
};

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,  // for k-grouped GEMM only: 128 on SM90 (float SF), gran_k * 4 on SM100 (packed UE8M0 SF)
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    uint32_t next_group_idx, next_shape_k;

    // Only used for k-grouped gemm
    CUTLASS_DEVICE void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = grouped_layout[group_idx];
            if (shape_k > 0)
                break;
        }
    }

    // ReSharper disable once CppPossiblyUninitializedMember
    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout = nullptr) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            get_next_k_group(current_group_idx, current_shape_k);
            next_group_idx = current_group_idx + 1;
            get_next_k_group(next_group_idx, next_shape_k);
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN)
                    offset = current_group_idx * shape_dim;
                else if constexpr (kIndexType == IndexType::K)
                    offset = current_k_cumsum;
                else if constexpr (kIndexType == IndexType::SF_K)
                    offset = current_sf_k_cumsum;
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = math::align(current_psum_m, BLOCK_M);
                current_psum_m = grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                if (next_block_idx < (current_num_valid_groups + 1) * num_blocks)
                    break;

                // Move to check the next group
                current_k_cumsum += current_shape_k;
                current_sf_k_cumsum += math::ceil_div(current_shape_k, SF_K_ALIGNMENT);
                current_num_valid_groups ++;

                current_group_idx = next_group_idx ++;
                current_shape_k = next_shape_k;
                get_next_k_group(next_group_idx, next_shape_k);
            }

            get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            // For SM90 only
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // Always aligned on N (constant bypass)
                                num_m_blocks % kNumMulticast == 0 or                  // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;                    // Peer CTA in bound
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            // Unreachable 
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm::sched
""",
    'key-files/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh': r"""#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const layout::Workspace& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }

        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched
""",
    'key-files/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh': r"""#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n, const bool is_context_lens_2d,
                                    const uint32_t* context_lens, uint32_t* schedule_metadata) {
    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    uint32_t num_segs[kAlignedBatchSize / 32];
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        const uint32_t q_idx = k * 32 + lane_idx;
        const uint32_t lens_idx = (is_context_lens_2d ? q_idx * next_n + next_n - 1 : q_idx);
        const uint32_t context_len = (q_idx < batch_size ? context_lens[lens_idx] : 0);
        num_segs[k] = math::ceil_div(context_len, SPLIT_KV);
    }

    __shared__ uint32_t prefix_sum[kAlignedBatchSize];
    uint32_t sum = 0;
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        uint32_t x = num_segs[k];
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffff, x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 32 + lane_idx] = x;
        sum = __shfl_sync(0xffffffff, x, 31);
    }

    const uint32_t num_next_n_atoms = next_n / ((next_n % 2 == 0) ? 2 : 1);
    const uint32_t total = sum * num_next_n_atoms;
    const uint32_t q = total / kNumSMs, r = total % kNumSMs;
    for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
        uint32_t q_idx = 0;
        while (q_idx < batch_size and prefix_sum[q_idx] * num_next_n_atoms <= seg_starts)
            ++ q_idx;
        const uint32_t offset_in_q = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1] * num_next_n_atoms);
        const uint32_t num_segs_q = (q_idx == 0 ? prefix_sum[0] : prefix_sum[q_idx] - prefix_sum[q_idx - 1]);
        const uint32_t atom_idx = num_segs_q > 0 ? offset_in_q / num_segs_q : 0;
        const uint32_t kv_split_idx = num_segs_q > 0 ? offset_in_q % num_segs_q : 0;
        const uint32_t q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
        __syncwarp();

        schedule_metadata[sm_idx * 2] = q_atom_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

template <uint32_t kNextN, bool kIsContextLens2D,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit,
          uint32_t kNumNextNAtoms>
struct PagedMQALogitsScheduler {
    const uint32_t* context_lens;

    uint32_t current_q_atom_idx, current_kv_idx;
    uint32_t end_q_atom_idx, end_kv_idx;
    uint32_t current_num_kv;

    CUTLASS_DEVICE uint32_t get_num_kv(const uint32_t& q_atom_idx) const {
        const uint32_t q_idx = q_atom_idx / kNumNextNAtoms;
        const auto lens_idx = (kIsContextLens2D ? q_idx * kNextN + kNextN - 1 : q_idx);
        return math::ceil_div(context_lens[lens_idx], BLOCK_KV);
    }

    CUTLASS_DEVICE explicit PagedMQALogitsScheduler(const uint32_t& sm_idx, const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->context_lens = context_lens;

        const auto current_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        current_q_atom_idx = current_pack.x, current_kv_idx = current_pack.y * kNumBlocksPerSplit;
        end_q_atom_idx = end_pack.x, end_kv_idx = end_pack.y * kNumBlocksPerSplit;

        current_num_kv = get_num_kv(current_q_atom_idx);
    }

    CUTLASS_DEVICE bool fetch_next_task(uint32_t &q_atom_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_atom_idx = current_q_atom_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (current_q_atom_idx == end_q_atom_idx and current_kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumBlocksPerSplit;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_atom_idx;
            current_kv_idx = 0;
            if (current_q_atom_idx % kNumNextNAtoms == 0 and exist_q_atom_idx(current_q_atom_idx)) {
                current_num_kv = get_num_kv(current_q_atom_idx);
            }
        }
        return true;
    }

    CUTLASS_DEVICE bool exist_q_atom_idx(const uint32_t& q_atom_idx) const {
        return q_atom_idx < end_q_atom_idx or (q_atom_idx == end_q_atom_idx and 0 < end_kv_idx);
    }
};

} // namespace deep_gemm::sched
""",
    'key-files/third-party/tilelang_ops/__init__.py': r"""from .swiglu_apply_weight_to_fp8 import swiglu_apply_weight_to_fp8
""",
    'key-files/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py': r"""import deep_gemm
import tilelang
import torch
from math import gcd
from tilelang import language as T

from .utils import get_sf_and_inv, get_sf_shape


@tilelang.jit
def _swiglu_apply_weight_to_fp8_tl(
    half_hidden: int,
    num_per_channels: int,
    use_col_major_scales: bool,
    round_scale: bool,
    ue8m0_scale: bool,
    num_ctas: int,
    has_topk_weights: bool,
    has_avail_tokens: bool,
    has_clamp_value: bool,
    output_bf16: bool,
    fast_math: bool,
) -> None:
    in_dtype = T.bfloat16
    w_dtype = T.float32
    out_dtype = T.float8_e4m3fn
    out_sf_dtype = T.uint8 if ue8m0_scale else T.float32
    num_dtype = T.int32
    num_tokens = T.dynamic("num_tokens")
    assert half_hidden % 16 == 0
    if not output_bf16:
        assert half_hidden % num_per_channels == 0

    num_block_h = max(half_hidden // gcd(half_hidden, 16 * 1024), 1)
    assert num_block_h <= num_ctas, "not supported hidden size"
    blk_h = half_hidden // num_block_h

    layout_h = blk_h // 16

    blk_n = 1024 // layout_h

    def local_layout(i, j):  # noqa: ANN001
        thread_id = i * layout_h + j // 16
        local_id = j % 16
        return thread_id, local_id

    def local_layout_3d(i, j, k):  # noqa: ANN001
        return local_layout(i, j * num_per_channels + k)

    @T.macro
    def main(
        bi: int,
        bh: int,
        num_ctas: int,
        x: T.Tensor[(num_tokens, half_hidden * 2), in_dtype],  # type: ignore
        topk_weights: T.Tensor[num_tokens, w_dtype],  # type: ignore
        avail_tokens: T.Tensor[1, num_dtype],  # type: ignore
        out: T.Tensor[(num_tokens, half_hidden), out_dtype],  # type: ignore
        out_sf: T.Tensor[get_sf_shape(num_tokens, half_hidden, num_per_channels, ue8m0_scale, use_col_major_scales), out_sf_dtype],  # type: ignore
        out_bf16: T.Tensor[(num_tokens, half_hidden), T.bfloat16],  # type: ignore
        clamp_value: T.float32,
    ):
        gate_frag = T.alloc_fragment((blk_n, blk_h), T.float32)
        up_frag = T.alloc_fragment((blk_n, blk_h), T.float32)
        y_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels, num_per_channels), T.float32)
        y_f8_frag = T.alloc_fragment((blk_n, blk_h), out_dtype)
        T.annotate_layout(
            {
                gate_frag: T.Fragment(gate_frag.shape, forward_fn=local_layout),
                up_frag: T.Fragment(up_frag.shape, forward_fn=local_layout),
                y_frag: T.Fragment(y_frag.shape, forward_fn=local_layout_3d),
                y_f8_frag: T.Fragment(y_f8_frag.shape, forward_fn=local_layout),
            }
        )

        T.assume(0 <= bh * blk_h + blk_h <= half_hidden)

        for i, j in T.Parallel(blk_n, blk_h):
            gate_frag[i, j] = x[bi + i * num_ctas, bh * blk_h + j]
        for i, j in T.Parallel(blk_n, blk_h):
            up_frag[i, j] = x[bi + i * num_ctas, half_hidden + bh * blk_h + j]

        topk_weight = T.alloc_fragment((blk_n,), T.float32)
        for i in T.Parallel(blk_n):
            topk_weight[i] = topk_weights[bi + i * num_ctas] if has_topk_weights else 1.0

        zero = T.alloc_var(T.float32, 0.0)
        for i, j in T.Parallel(blk_n, blk_h):
            if has_clamp_value:
                up_frag[i, j] = T.min(clamp_value, T.max(-clamp_value, up_frag[i, j]))
                gate_frag[i, j] = T.min(clamp_value, gate_frag[i, j])
            y_frag[i, j // num_per_channels, j % num_per_channels] = (
                gate_frag[i, j] / (1 + T.exp(-gate_frag[i, j])) * up_frag[i, j] * topk_weight[i] + zero
            )  # HACK : + 0 for vectorize

        y_max_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels), T.float32)
        sf_inv_frag = T.alloc_fragment((blk_n, blk_h // num_per_channels), T.float32)
        T.reduce_absmax(T.reshape(y_frag, (blk_n, blk_h // num_per_channels, num_per_channels)), y_max_frag)
        for i, j in T.Parallel(blk_n, blk_h // num_per_channels):
            clamped_amax = T.max(y_max_frag[i, j], 1e-4)
            sf, sf_inv = get_sf_and_inv(clamped_amax, round_scale, ue8m0_scale)
            i_index = bi + i * num_ctas
            j_index = blk_h // num_per_channels * bh + j
            # Store SF
            if ue8m0_scale:
                out_sf[j_index // 4, i_index * 4 + j_index % 4] = sf
            elif use_col_major_scales:
                out_sf[j_index, i_index] = sf
            else:
                out_sf[i_index, j_index] = sf
            sf_inv_frag[i, j] = sf_inv

        for i, j in T.Parallel(blk_n, blk_h):
            y_f8_frag[i, j] = y_frag[i, j // num_per_channels, j % num_per_channels] * sf_inv_frag[i, j // num_per_channels]

        for i, j in T.Parallel(blk_n, blk_h):
            out[bi + i * num_ctas, blk_h * bh + j] = y_f8_frag[i, j]

        if output_bf16:
            for i, j in T.Parallel(blk_n, blk_h):
                out_bf16[bi + i * num_ctas, blk_h * bh + j] = y_frag[i, j // num_per_channels, j % num_per_channels]

    @T.prim_func
    def _swiglu_apply_weight_to_fp8(
        x: T.Tensor[(num_tokens, half_hidden * 2), in_dtype],  # type: ignore
        topk_weights: T.Tensor[num_tokens, w_dtype],  # type: ignore
        avail_tokens: T.Tensor[1, num_dtype],  # type: ignore
        out: T.Tensor[(num_tokens, half_hidden), out_dtype],  # type: ignore
        out_sf: T.Tensor[get_sf_shape(num_tokens, half_hidden, num_per_channels, ue8m0_scale, use_col_major_scales), out_sf_dtype],  # type: ignore
        out_bf16: T.Tensor[(num_tokens, half_hidden), T.bfloat16],  # type: ignore
        clamp_value: T.float32,
    ):
        # we actually don't use this
        _ = num_tokens
        # simplest schedule: one token one block, but pipelined as persistent
        with T.Kernel(num_ctas, threads=1024) as cta_id:
            avail_tokens_l = avail_tokens[0] if has_avail_tokens else num_tokens
            T.pdl_sync()  # avail_tokens must be const.
            T.assume(0 <= avail_tokens_l <= num_tokens)
            thread_idx = T.get_thread_binding()
            new_num_ctas = num_ctas // num_block_h
            if cta_id >= new_num_ctas * num_block_h:
                T.thread_return()
            for bi in T.serial(cta_id // num_block_h, avail_tokens_l - thread_idx // layout_h * new_num_ctas, new_num_ctas * blk_n):
                main(bi, cta_id % num_block_h, new_num_ctas, x, topk_weights, avail_tokens, out, out_sf, out_bf16, clamp_value)

    return _swiglu_apply_weight_to_fp8


def swiglu_apply_weight_to_fp8(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    avail_tokens: torch.Tensor | None,
    num_per_channels: int,
    use_col_major_scales: bool,
    round_scale: bool,
    ue8m0_scale: bool,
    clamp_value: float | None = None,
    fmt: str = "e4m3",
    num_sms: int | None = None,
    output_bf16: bool = False,
    fast_math: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert fmt == "e4m3"
    if num_sms is None:
        num_sms = deep_gemm.get_num_sms()

    num_tokens, hidden_size = x.shape
    assert hidden_size % (2 * num_per_channels) == 0

    y = torch.empty(
        (num_tokens, hidden_size // 2),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    y_sf = torch.empty(
        get_sf_shape(num_tokens, hidden_size // 2, num_per_channels, ue8m0_scale, use_col_major_scales),
        device=x.device,
        dtype=(torch.uint8 if ue8m0_scale else torch.float32),
    )

    y_bf16 = torch.empty((num_tokens, hidden_size // 2), device=x.device, dtype=torch.bfloat16) if output_bf16 else None

    if num_tokens > 0:
        _swiglu_apply_weight_to_fp8_tl.pass_configs = {
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: fast_math,
        }
        kernel = _swiglu_apply_weight_to_fp8_tl(
            hidden_size // 2,
            num_per_channels,
            use_col_major_scales,
            round_scale,
            ue8m0_scale,
            num_sms,
            topk_weights is not None,
            avail_tokens is not None,
            clamp_value is not None,
            output_bf16,
            fast_math
        )
        kernel(x, topk_weights, avail_tokens.view(1) if avail_tokens is not None else None, y, y_sf, y_bf16, clamp_value or 0.0)

    if ue8m0_scale:
        if num_tokens == 0:
            y_sf.as_strided_(y_sf.size(), (0, 1))
        y_sf = y_sf.view(dtype=torch.int32)
    if output_bf16:
        return y, y_sf.T[:num_tokens], y_bf16
    else:
        return y, y_sf.T[:num_tokens]
""",
    'key-files/third-party/tilelang_ops/utils.py': r"""from typing import Any
from tilelang import language as T


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_sf_shape(
    num_tokens: int,
    hidden: int,
    num_per_channels: int,
    use_ue8m0: bool,
    use_col_major_sf: bool,
) -> tuple[int, int]:
    num_scales = ceil_div(hidden, num_per_channels)
    num_scales = ceil_div(num_scales, 4) if use_ue8m0 else num_scales

    # For col-major SF, TMA must be aligned into 16 bytes
    # For UE8M0, we must use col-major SF, and 4 UE8M0 are expanded into the inner dim (token)
    num_sf_tokens = num_tokens
    if use_col_major_sf:
        num_sf_tokens = align(num_tokens, 4)
        num_sf_tokens = num_sf_tokens * 4 if use_ue8m0 else num_sf_tokens

    return (num_scales, num_sf_tokens) if use_col_major_sf else (num_sf_tokens, num_scales)


def get_sf_and_inv(amax: float, round_sf: bool, use_ue8m0: bool) -> tuple[Any, Any]:
    sf = amax / 448.0
    if not round_sf:
        return sf, 448.0 / amax

    # Round into 2's power
    bits = T.reinterpret("uint32", sf)
    exp = (bits >> 23) & 0xFF
    man_bits = bits & ((1 << 23) - 1)
    exp_scale = T.reinterpret("int32", exp - 127 + (man_bits != 0))
    if use_ue8m0:  # noqa: SIM108
        sf = T.Cast("uint8", exp_scale + 127)
    else:
        sf = T.reinterpret("float", (127 + exp_scale) << 23)
    return sf, T.reinterpret("float", (127 - exp_scale) << 23)
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
