"""KernelWiki artifact PR-16032 (vllm-project/vllm)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/csrc/attention/mla/cutlass_mla_entry.cu': r"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#if defined ENABLE_CUTLASS_MLA && ENABLE_CUTLASS_MLA
void cutlass_mla_decode_sm100a(torch::Tensor const& out,
                               torch::Tensor const& q_nope,
                               torch::Tensor const& q_pe,
                               torch::Tensor const& kv_c_and_k_pe_cache,
                               torch::Tensor const& seq_lens,
                               torch::Tensor const& page_table, double scale);
#endif

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale) {
#if defined ENABLE_CUTLASS_MLA && ENABLE_CUTLASS_MLA
  return cutlass_mla_decode_sm100a(out, q_nope, q_pe, kv_c_and_k_pe_cache,
                                   seq_lens, page_table, scale);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled cutlass MLA");
}
""",
    'key-files/csrc/attention/mla/cutlass_mla_kernels.cu': r"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass_extensions/common.hpp"

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

using namespace cute;
using namespace cutlass::fmha::kernel;

template <typename T, bool PersistenceOption = true>
struct MlaSm100 {
  using Element = T;
  using ElementAcc = float;
  using ElementOut = T;

  using TileShape = Shape<_128, _128, Shape<_512, _64>>;
  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;

  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler =
      std::conditional_t<PersistenceOption, Sm100MlaPersistentTileScheduler,
                         Sm100MlaIndividualTileScheduler>;

  using FmhaKernel =
      cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
          TileShape, Element, ElementAcc, ElementOut, ElementAcc, TileScheduler,
          /*kIsCpAsync=*/true>;
  using Fmha = cutlass::fmha::device::MLA<FmhaKernel>;
};

template <typename T>
typename T::Fmha::Arguments args_from_options(
    at::Tensor const& out, at::Tensor const& q_nope, at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache, at::Tensor const& seq_lens,
    at::Tensor const& page_table, double scale) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  int batches = q_nope.sizes()[0];
  int page_count_per_seq = page_table.sizes()[1];
  int page_count_total = kv_c_and_k_pe_cache.sizes()[0];
  int page_size = kv_c_and_k_pe_cache.sizes()[1];
  int max_seq_len = page_size * page_count_per_seq;
  using TileShapeH = typename T::TileShapeH;
  using TileShapeD = typename T::TileShapeD;
  auto problem_shape =
      cute::make_tuple(TileShapeH{}, max_seq_len, TileShapeD{}, batches);

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideLSE = typename T::StrideLSE;

  StrideQ stride_Q_latent = cute::make_tuple(
      static_cast<int64_t>(D_latent), _1{}, static_cast<int64_t>(H * D_latent));
  StrideQ stride_Q_rope = cute::make_tuple(static_cast<int64_t>(D_rope), _1{},
                                           static_cast<int64_t>(H * D_rope));
  StrideK stride_C =
      cute::make_tuple(static_cast<int64_t>(D_latent + D_rope), _1{},
                       static_cast<int64_t>(page_size * (D_latent + D_rope)));
  StrideLSE stride_PT = cute::make_stride(_1{}, page_count_per_seq);
  StrideLSE stride_LSE = cute::make_tuple(_1{}, static_cast<int>(H));
  StrideO stride_O = cute::make_tuple(static_cast<int64_t>(D_latent), _1{},
                                      static_cast<int64_t>(H * D_latent));

  using Element = typename T::Element;
  using ElementOut = typename T::ElementOut;
  using ElementAcc = typename T::ElementAcc;
  auto Q_latent_ptr = static_cast<Element*>(q_nope.data_ptr());
  auto Q_rope_ptr = static_cast<Element*>(q_pe.data_ptr());
  auto C_ptr = static_cast<Element*>(kv_c_and_k_pe_cache.data_ptr());
  auto scale_f = static_cast<float>(scale);
  typename T::Fmha::Arguments arguments{
      problem_shape,
      {scale_f, Q_latent_ptr, stride_Q_latent, Q_rope_ptr, stride_Q_rope, C_ptr,
       stride_C, C_ptr + D_latent, stride_C,
       static_cast<int*>(seq_lens.data_ptr()),
       static_cast<int*>(page_table.data_ptr()), stride_PT, page_count_total,
       page_size},
      {static_cast<ElementOut*>(out.data_ptr()), stride_O,
       static_cast<ElementAcc*>(nullptr), stride_LSE},
      hw_info,
      -1,       // split_kv
      nullptr,  // is_var_split_kv
  };
  // TODO(kaixih@nvidia): When split_kv=-1 and is_var_split_kv=false, we compute
  // split_kv automatically based on batch size and sequence length to balance
  // workload across available SMs. Consider using var_split_kv for manual
  // control if needed.
  T::Fmha::set_split_kv(arguments);
  return arguments;
}

template <typename Element>
void runMla(at::Tensor const& out, at::Tensor const& q_nope,
            at::Tensor const& q_pe, at::Tensor const& kv_c_and_k_pe_cache,
            at::Tensor const& seq_lens, at::Tensor const& page_table,
            float scale, cudaStream_t stream) {
  using MlaSm100Type = MlaSm100<Element>;
  typename MlaSm100Type::Fmha fmha;
  auto arguments = args_from_options<MlaSm100Type>(
      out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, scale);
  size_t workspace_size = MlaSm100Type::Fmha::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(q_nope.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(fmha.can_implement(arguments));

  CUTLASS_CHECK(fmha.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(fmha.run(arguments, workspace.data_ptr(), stream));
}

void cutlass_mla_decode_sm100a(torch::Tensor const& out,
                               torch::Tensor const& q_nope,
                               torch::Tensor const& q_pe,
                               torch::Tensor const& kv_c_and_k_pe_cache,
                               torch::Tensor const& seq_lens,
                               torch::Tensor const& page_table, double scale) {
  TORCH_CHECK(q_nope.device().is_cuda(), "q_nope must be on CUDA");
  TORCH_CHECK(q_nope.dim() == 3, "q_nope must be a 3D tensor");
  TORCH_CHECK(q_pe.dim() == 3, "q_pe must be a 3D tensor");
  TORCH_CHECK(kv_c_and_k_pe_cache.dim() == 3,
              "kv_c_and_k_pe_cache must be a 3D tensor");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be a 1D tensor");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be a 2D tensor");
  TORCH_CHECK(out.dim() == 3, "out must be a 3D tensor");

  auto B_q_nope = q_nope.size(0);
  auto H_q_nope = q_nope.size(1);
  auto D_q_nope = q_nope.size(2);
  auto B_q_pe = q_pe.size(0);
  auto H_q_pe = q_pe.size(1);
  auto D_q_pe = q_pe.size(2);
  auto B_pt = page_table.size(0);
  auto PAGE_NUM = page_table.size(1);
  auto PAGE_SIZE = kv_c_and_k_pe_cache.size(1);
  auto D_ckv = kv_c_and_k_pe_cache.size(2);
  auto B_o = out.size(0);
  auto H_o = out.size(1);
  auto D_o = out.size(2);

  TORCH_CHECK(D_q_nope == 512, "D_q_nope must be equal to 512");
  TORCH_CHECK(D_q_pe == 64, "D_q_pe must be equal to 64");
  TORCH_CHECK(D_ckv == 576, "D_ckv must be equal to 576");
  TORCH_CHECK(H_q_nope == H_q_pe && H_q_nope == H_o && H_o == 128,
              "H_q_nope, H_q_pe, and H_o must be equal to 128");
  TORCH_CHECK(PAGE_SIZE > 0 && (PAGE_SIZE & (PAGE_SIZE - 1)) == 0,
              "PAGE_SIZE must be a power of 2");
  TORCH_CHECK(
      B_q_nope == B_q_pe && B_q_nope == B_pt && B_q_nope == B_o,
      "Batch dims must be same for page_table, q_nope and q_pe, and out");
  TORCH_CHECK(PAGE_NUM % (128 / PAGE_SIZE) == 0,
              "PAGE_NUM must be divisible by 128 / PAGE_SIZE");
  TORCH_CHECK(D_o == 512, "D_o must be equal to 512");

  TORCH_CHECK(q_nope.dtype() == at::ScalarType::Half ||
                  q_nope.dtype() == at::ScalarType::BFloat16 ||
                  q_nope.dtype() == at::ScalarType::Float8_e4m3fn,
              "q_nope must be a half, bfloat16, or float8_e4m3fn tensor");
  TORCH_CHECK(kv_c_and_k_pe_cache.dtype() == q_nope.dtype() &&
                  q_nope.dtype() == q_pe.dtype(),
              "kv_c_and_k_pe_cache, q_nope, and q_pe must be the same type");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32,
              "seq_lens must be a 32-bit integer tensor");
  TORCH_CHECK(page_table.dtype() == torch::kInt32,
              "page_table must be a 32-bit integer tensor");

  auto in_dtype = q_nope.dtype();
  at::cuda::CUDAGuard device_guard{(char)q_nope.get_device()};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q_nope.get_device());
  if (in_dtype == at::ScalarType::Half) {
    runMla<cutlass::half_t>(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens,
                            page_table, scale, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    runMla<cutlass::bfloat16_t>(out, q_nope, q_pe, kv_c_and_k_pe_cache,
                                seq_lens, page_table, scale, stream);
  } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
    runMla<cutlass::float_e4m3_t>(out, q_nope, q_pe, kv_c_and_k_pe_cache,
                                  seq_lens, page_table, scale, stream);
  } else {
    TORCH_CHECK(false, "Unsupported input data type of MLA");
  }
}
""",
    'key-files/csrc/ops.h': r"""#pragma once

#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

#ifndef USE_ROCM
void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse);
#endif

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void rms_norm_static_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                               torch::Tensor& weight, torch::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::Tensor& out,
                                         torch::Tensor& input,
                                         torch::Tensor& residual,
                                         torch::Tensor& weight,
                                         torch::Tensor& scale, double epsilon);

void rms_norm_dynamic_per_token_quant(torch::Tensor& out,
                                      torch::Tensor const& input,
                                      torch::Tensor const& weight,
                                      torch::Tensor& scales,
                                      double const epsilon,
                                      std::optional<torch::Tensor> scale_ub,
                                      std::optional<torch::Tensor> residual);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              torch::Tensor& key, int64_t head_size,
                              torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void advance_step_flashattn(int64_t num_seqs, int64_t num_queries,
                            int64_t block_size, torch::Tensor& input_tokens,
                            torch::Tensor& sampled_token_ids,
                            torch::Tensor& input_positions,
                            torch::Tensor& seq_lens,
                            torch::Tensor& slot_mapping,
                            torch::Tensor& block_tables);

void advance_step_flashinfer(
    int64_t num_seqs, int64_t num_queries, int64_t block_size,
    torch::Tensor& input_tokens, torch::Tensor& sampled_token_ids,
    torch::Tensor& input_positions, torch::Tensor& seq_lens,
    torch::Tensor& slot_mapping, torch::Tensor& block_tables,
    torch::Tensor& paged_kv_indices, torch::Tensor& paged_kv_indptr,
    torch::Tensor& paged_kv_last_page_len, torch::Tensor& block_table_bounds);

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

#ifndef USE_ROCM
torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const std::vector<int64_t>& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias);

torch::Tensor aqlm_dequant(
    const torch::Tensor& codes, const torch::Tensor& codebooks,
    const std::vector<int64_t>& codebook_partition_sizes);

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm);
#endif

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

#ifndef USE_ROCM

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A,
                           torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf,
                           torch::Tensor const& alpha);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides);

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias);

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability);

void cutlass_scaled_sparse_mm(torch::Tensor& out, torch::Tensor const& a,
                              torch::Tensor const& b, torch::Tensor const& e,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              std::optional<torch::Tensor> const& bias);

std::vector<torch::Tensor> cutlass_sparse_compress(torch::Tensor const& a);

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_scale,
                      torch::Tensor const& input_scale);
#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    std::optional<torch::Tensor> const& scale_ub);

void selective_scan_fwd(const torch::Tensor& u, const torch::Tensor& delta,
                        const torch::Tensor& A, const torch::Tensor& B,
                        const torch::Tensor& C,
                        const std::optional<torch::Tensor>& D_,
                        const std::optional<torch::Tensor>& z_,
                        const std::optional<torch::Tensor>& delta_bias_,
                        bool delta_softplus,
                        const std::optional<torch::Tensor>& query_start_loc,
                        const std::optional<torch::Tensor>& cache_indices,
                        const std::optional<torch::Tensor>& has_initial_state,
                        const torch::Tensor& ssm_states, int64_t pad_slot_id);

void causal_conv1d_update(const at::Tensor& x, const at::Tensor& conv_state,
                          const at::Tensor& weight,
                          const std::optional<at::Tensor>& bias_,
                          bool silu_activation,
                          const std::optional<at::Tensor>& cache_seqlens_,
                          const std::optional<at::Tensor>& conv_state_indices_,
                          int64_t pad_slot_id);

void causal_conv1d_fwd(const at::Tensor& x, const at::Tensor& weight,
                       const std::optional<at::Tensor>& bias_,
                       const std::optional<at::Tensor>& conv_states,
                       const std::optional<at::Tensor>& query_start_loc,
                       const std::optional<at::Tensor>& cache_indices,
                       const std::optional<at::Tensor>& has_initial_state,
                       bool silu_activation, int64_t pad_slot_id);

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);
""",
    'key-files/csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu': r"""/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel Perf config
template <typename T>
struct KernelTraits;

template <>
struct KernelTraits<float> {
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

template <>
struct KernelTraits<cutlass::half_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

template <typename T>
struct Fp4GemmSm100 {
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = T;
  using ElementC = T;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits<T>::MmaTileShape;
  using ClusterShape = typename KernelTraits<T>::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits<T>::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename T>
typename T::Gemm::Arguments args_from_options(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf, at::Tensor const& alpha,
    int64_t M, int64_t N, int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm100BlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

template <typename T>
void runGemm(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
             at::Tensor const& A_sf, at::Tensor const& B_sf,
             at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
             cudaStream_t stream) {
  typename Fp4GemmSm100<T>::Gemm gemm;

  auto arguments =
      args_from_options<Fp4GemmSm100<T>>(D, A, B, A_sf, B_sf, alpha, m, n, k);

  size_t workspace_size = Fp4GemmSm100<T>::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
#else
template <typename T>
void runGemm(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
             at::Tensor const& A_sf, at::Tensor const& B_sf,
             at::Tensor const& alpha, int64_t m, int64_t n, int64_t k,
             cudaStream_t stream) {
  TORCH_CHECK(false,
              "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
              "a CUTLASS 3.8 source directory to enable support.");
}
#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A,
                                  torch::Tensor const& B,
                                  torch::Tensor const& A_sf,
                                  torch::Tensor const& B_sf,
                                  torch::Tensor const& alpha) {
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
              "a and b shapes cannot be multiplied (", A.sizes()[0], "x",
              A.sizes()[1], " and ", B.sizes()[0], "x", B.sizes()[1], ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got a shape: (", A.sizes()[0], "x", A.sizes()[1],
              "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(A_sf.sizes()[1] == B_sf.sizes()[1],
              "scale_a and scale_b shapes cannot be multiplied (",
              A_sf.sizes()[0], "x", A_sf.sizes()[1], " and ", B_sf.sizes()[0],
              "x", B_sf.sizes()[1], ")");
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to a shape (", rounded_m,
              "x", rounded_k, "), but got a shape (", A_sf.sizes()[0], "x",
              A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to a shape (", rounded_n,
              "x", rounded_k, "), but got a shape (", B_sf.sizes()[0], "x",
              B_sf.sizes()[1], ")");

  auto out_dtype = D.dtype();
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  if (out_dtype == at::ScalarType::Half) {
    runGemm<cutlass::half_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    runGemm<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::Float) {
    runGemm<float>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm");
  }
}
""",
    'key-files/csrc/torch_bindings.cpp': r"""#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //

  // The default behavior in PyTorch 2.6 is "requires_contiguous", so we need
  // to override this for many GEMMs with the following tag. Otherwise,
  // torch.compile will force all input tensors to be contiguous(), which
  // will break many custom ops that require column-major weight matrices.
  // TODO: remove this for PyTorch 2.8, when the default is planned to switch
  // to match exact eager-mode strides.
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

#ifndef USE_ROCM
  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
  ops.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse) -> ()");
  ops.impl("merge_attn_states", torch::kCUDA, &merge_attn_states);
#endif

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // prepare_inputs advance_step
  ops.def(
      "advance_step_flashattn(int num_seqs, int num_queries, int block_size, "
      "Tensor! input_tokens, Tensor sampled_token_ids, "
      "Tensor! input_positions, Tensor! seq_lens, Tensor! slot_mapping, "
      "Tensor block_tables) -> ()");
  ops.impl("advance_step_flashattn", torch::kCUDA, &advance_step_flashattn);

  ops.def(
      "advance_step_flashinfer("
      "    int num_seqs, int num_queries, int block_size,"
      "    Tensor! input_tokens, Tensor sampled_token_ids,"
      "    Tensor! input_positions, Tensor! seq_lens, Tensor! slot_mapping,"
      "    Tensor block_tables, Tensor! paged_kv_indices,"
      "    Tensor! paged_kv_indptr, Tensor! paged_kv_last_page_len,"
      "    Tensor! block_table_bounds"
      ") -> ()");
  ops.impl("advance_step_flashinfer", torch::kCUDA, &advance_step_flashinfer);

  // Compute MLA decode using cutlass.
  ops.def(
      "cutlass_mla_decode(Tensor! out, Tensor q_nope, Tensor q_pe,"
      "                   Tensor kv_c_and_k_pe_cache, Tensor seq_lens,"
      "                   Tensor page_table, float scale) -> ()");
  ops.impl("cutlass_mla_decode", torch::kCUDA, &cutlass_mla_decode);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");
  ops.impl("rms_norm_static_fp8_quant", torch::kCUDA,
           &rms_norm_static_fp8_quant);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kCUDA,
           &fused_add_rms_norm_static_fp8_quant);

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.impl("rms_norm_dynamic_per_token_quant", torch::kCUDA,
           &rms_norm_dynamic_per_token_quant);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def(
      "batched_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor! key, int head_size,"
      "                         Tensor cos_sin_cache, bool is_neox,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

  // Quantization ops
#ifndef USE_ROCM
  // Quantized GEMM for AQLM.
  ops.def(
      "aqlm_gemm(Tensor input, Tensor codes, Tensor codebooks, "
      "Tensor scales, int[] codebook_partition_sizes, Tensor? bias) "
      "-> Tensor",
      {stride_tag});
  ops.impl("aqlm_gemm", torch::kCUDA, &aqlm_gemm);

  // Decompression method for AQLM.
  ops.def(
      "aqlm_dequant(Tensor codes, Tensor codebooks, "
      "int[] codebook_partition_sizes) -> Tensor",
      {stride_tag});
  ops.impl("aqlm_dequant", torch::kCUDA, &aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters) -> Tensor",
      {stride_tag});
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor",
      {stride_tag});
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, SymInt size_k) -> "
      "Tensor",
      {stride_tag});
  // conditionally compiled so impl in source file

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k) -> Tensor",
      {stride_tag});
  //  conditionally compiled so impl in source file

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor",
      {stride_tag});
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // conditionally compiled so impl registration is in source file

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool has_zp, bool use_atomic_add, bool use_fp32_reduce, "
      "bool is_zp_float) -> Tensor",
      {stride_tag});
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file
#endif

  // Dequantization for GGML.
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  // mmq kernel for GGML.
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  // moe kernel for GGML.
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  ops.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);

#ifndef USE_ROCM
  // fp8_marlin Optimized Quantized GEMM for FP8 weight-only.
  ops.def(
      "fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int num_bits, SymInt size_m, SymInt size_n, "
      "SymInt size_k) -> Tensor",
      {stride_tag});
  // conditionally compiled so impl registration is in source file

  // marlin_qqq_gemm for QQQ.
  ops.def(
      "marlin_qqq_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor s_tok, Tensor s_ch, Tensor s_group, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, "
      "SymInt size_k) -> Tensor",
      {stride_tag});
  // conditionally compiled so impl registration is in source file

  // CUTLASS nvfp4 block scaled GEMM
  ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()",
      {stride_tag});
  ops.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()",
      {stride_tag});
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()",
      {stride_tag});
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);

  // Check if cutlass grouped gemm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_group_gemm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_group_gemm_supported", &cutlass_group_gemm_supported);

  // CUTLASS w8a8 grouped GEMM
  ops.def(
      "cutlass_moe_mm(Tensor! out_tensors, Tensor a_tensors, Tensor b_tensors, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor c_strides) -> ()",
      {stride_tag});
  ops.impl("cutlass_moe_mm", torch::kCUDA, &cutlass_moe_mm);

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM. It takes topk_ids as an input, and computes expert_offsets
  // (token start indices of each expert). In addition to this, it computes
  // problem sizes for each expert's multiplication used by the two mms called
  // from fused MoE operation, and arrays with permutations required to shuffle
  // and de-shuffle the input/output of the fused operation.
  ops.def(
      "get_cutlass_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k) -> ()",
      {stride_tag});
  ops.impl("get_cutlass_moe_mm_data", torch::kCUDA, &get_cutlass_moe_mm_data);

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");
  ops.impl("cutlass_scaled_mm_supports_block_fp8",
           &cutlass_scaled_mm_supports_block_fp8);

  // Check if cutlass sparse scaled_mm is supported for CUDA devices of the
  // given capability
  ops.def(
      "cutlass_sparse_scaled_mm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_sparse_scaled_mm_supported",
           &cutlass_sparse_scaled_mm_supported);

  // CUTLASS sparse GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_sparse_mm(Tensor! out, Tensor a,"
      "                         Tensor bt_nzs,"
      "                         Tensor bt_meta, Tensor a_scales,"
      "                         Tensor b_scales, Tensor? bias) -> ()",
      {stride_tag});
  ops.impl("cutlass_scaled_sparse_mm", torch::kCUDA, &cutlass_scaled_sparse_mm);

  // CUTLASS sparse matrix compressor
  ops.def("cutlass_sparse_compress(Tensor a) -> Tensor[]");
  ops.impl("cutlass_sparse_compress", &cutlass_sparse_compress);

  // Mamba selective scan kernel
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int pad_slot_id) -> ()");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  ops.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
  ops.impl("causal_conv1d_update", torch::kCUDA, &causal_conv1d_update);

  ops.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
  ops.impl("causal_conv1d_fwd", torch::kCUDA, &causal_conv1d_fwd);

  // Compute NVFP4 block quantized tensor.
  ops.def(
      "scaled_fp4_quant(Tensor! output, Tensor input,"
      "                 Tensor! output_scale, Tensor input_scale) -> ()");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  // Check if cutlass_scaled_mm_fp4 is supported for CUDA devices
  // of the given capability
  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp4", &cutlass_scaled_mm_supports_fp4);
#endif

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, int bit) "
      "-> Tensor",
      {stride_tag});
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);

#ifndef USE_ROCM
  // reorder weight for AllSpark Ampere W8A16 Fused Gemm kernel
  ops.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, "
      "bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, "
      "Tensor!? b_zeros_reorder, "
      "int K, int N, int N_32align) -> ()");
  //  conditionally compiled so impl in source file

  // AllSpark quantization ops
  ops.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, "
      "SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt "
      "CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");
  //  conditionally compiled so impl in source file
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  cache_ops.def(
      "copy_blocks_mla(Tensor(a!)[] kv_caches, Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks_mla", torch::kCUDA, &copy_blocks_mla);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kCUDA, &concat_and_cache_mla);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);

  // Gather cache blocks from src_cache to dst.
  cache_ops.def(
      "gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_cache", torch::kCUDA, &gather_cache);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);

  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);

  custom_ar.def("free_shared_buffer", &free_shared_buffer);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
""",
    'key-files/vllm/_custom_ops.py': r'''# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.library

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType

logger = init_logger(__name__)

if not current_platform.is_tpu() and not current_platform.is_hpu():
    try:
        import vllm._C
    except ImportError as e:
        logger.warning("Failed to import from vllm._C with %r", e)

supports_moe_ops = False
with contextlib.suppress(ImportError):
    import vllm._moe_C  # noqa: F401
    supports_moe_ops = True

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v2(
        out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step)


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._rocm_C.paged_attention(out, exp_sum, max_logits, tmp_out, query,
                                      key_cache, value_cache, num_kv_heads,
                                      scale, block_tables, seq_lens,
                                      query_start_loc, block_size, max_seq_len,
                                      alibi_slopes, kv_cache_dtype, k_scale,
                                      v_scale)


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cpu.mla_decode_kvcache(out, query, kv_cache, scale,
                                        block_tables, seq_lens)


# merge attn states ops
def merge_attn_states(output: torch.Tensor,
                      prefix_output: torch.Tensor,
                      prefix_lse: torch.Tensor,
                      suffix_output: torch.Tensor,
                      suffix_lse: torch.Tensor,
                      output_lse: Optional[torch.Tensor] = None) -> None:
    torch.ops._C.merge_attn_states(output, output_lse, prefix_output,
                                   prefix_lse, suffix_output, suffix_lse)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    torch.ops._C.batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    return torch.ops._C.advance_step_flashattn(num_seqs, num_queries,
                                               block_size, input_tokens,
                                               sampled_token_ids,
                                               input_positions, seq_lens,
                                               slot_mapping, block_tables)


def advance_step_flashinfer(num_seqs: int, num_queries: int, block_size: int,
                            input_tokens: torch.Tensor,
                            sampled_token_ids: torch.Tensor,
                            input_positions: torch.Tensor,
                            seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                            block_tables: torch.Tensor,
                            paged_kv_indices: torch.Tensor,
                            paged_kv_indptr: torch.Tensor,
                            paged_kv_last_page_len: torch.Tensor,
                            block_table_bound: torch.Tensor) -> None:

    return torch.ops._C.advance_step_flashinfer(
        num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
        input_positions, seq_lens, slot_mapping, block_tables,
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
        block_table_bound)


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)

    torch.ops._C.rms_norm_dynamic_per_token_quant(output, input, weight,
                                                  scales, epsilon, scale_ub,
                                                  residual)
    return output, scales


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton)
        return awq_dequantize_triton(qweight, scales, zeros)
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters,
                                       thx, thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_gemm_triton)
        return awq_gemm_triton(input, qweight, qzeros, scales, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit)


if hasattr(torch.ops._C, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_gptq_qzeros: torch.Tensor,
                        b_gptq_scales: torch.Tensor, b_g_idx: torch.Tensor,
                        use_exllama: bool, bit: int) -> torch.Tensor:
        return torch.empty((a.size(0), b_q_weight.size(1)),
                           dtype=a.dtype,
                           device=a.device)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                    size_n, size_k)


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, b_q_type: ScalarType,
                        size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_24_gemm(a, b_q_weight, b_meta, b_scales,
                                            workspace, b_q_type.id, size_m,
                                            size_n, size_k)


if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):

    @register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                  b_meta: torch.Tensor, b_scales: torch.Tensor,
                                  workspace: torch.Tensor,
                                  b_q_type: ScalarType, size_m: torch.SymInt,
                                  size_n: torch.SymInt,
                                  size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               b_zeros: torch.Tensor,
                               g_idx: torch.Tensor,
                               perm: torch.Tensor,
                               workspace: torch.Tensor,
                               b_q_type: ScalarType,
                               size_m: torch.SymInt,
                               size_n: torch.SymInt,
                               size_k: torch.SymInt,
                               is_k_full: bool,
                               has_zp: bool = False,
                               use_atomic_add: bool = False,
                               use_fp32_reduce: bool = False,
                               is_zp_float: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::marlin_qqq_gemm")
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              s_tok: torch.Tensor, s_ch: torch.Tensor,
                              s_group: torch.Tensor, workspace: torch.Tensor,
                              size_m: torch.SymInt, size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                          b_scales: torch.Tensor, workspace: torch.Tensor,
                          size_m: torch.SymInt, size_n: torch.SymInt,
                          size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight: torch.Tensor, scales: torch.Tensor,
                             zeros: torch.Tensor, split_k_iters: torch.SymInt,
                             thx: int, thy: int) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c),
                           dtype=scales.dtype,
                           device=scales.device)

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(input: torch.Tensor, qweight: torch.Tensor,
                       qzeros: torch.Tensor, scales: torch.Tensor,
                       split_k_iters: torch.SymInt) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty((split_k_iters, num_in_feats, qweight.size(1) * 8),
                           dtype=input.dtype,
                           device=input.device).sum(0)

    @register_fake("_C::aqlm_gemm")
    def _aqlm_gemm_fake(input: torch.Tensor, codes: torch.Tensor,
                        codebooks: torch.Tensor, scales: torch.Tensor,
                        codebook_partition_sizes: list[int],
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
        out_features = codes.size(0) * codebooks.size(2)
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)

        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @register_fake("_C::aqlm_dequant")
    def _aqlm_dequant_fake(
            codes: torch.Tensor, codebooks: torch.Tensor,
            codebook_partition_sizes: list[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @register_fake("_C::fp8_marlin_gemm")
    def _fp8_marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              b_scales: torch.Tensor, workspace: torch.Tensor,
                              num_bits: int, size_m: torch.SymInt,
                              size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)

    @register_fake("_C::machete_mm")
    def machete_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
            b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
            group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
        return torch.empty_like(b_q_weight,
                                memory_format=torch.contiguous_format)


if hasattr(torch.ops._C, "allspark_w8a16_gemm"):

    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(a: torch.Tensor, b_qweight: torch.Tensor,
                                  b_scales: torch.Tensor,
                                  b_qzeros: Optional[torch.Tensor],
                                  n: torch.SymInt, group_size: torch.SymInt,
                                  sm_count: torch.SymInt,
                                  sm_version: torch.SymInt,
                                  CUBLAS_M_THRESHOLD: torch.SymInt,
                                  has_zp: bool,
                                  n32k16_reorder: bool) -> torch.Tensor:
        m = a.size(0)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
            W: torch.Tensor,
            quant_type: int,
            m: torch.SymInt,
            n: torch.SymInt,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((1, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row),
                           dtype=torch.float16,
                           device=W.device)


# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_scaled_fp4_mm(a: torch.Tensor, b: torch.Tensor,
                          block_scale_a: torch.Tensor,
                          block_scale_b: torch.Tensor, alpha: torch.Tensor,
                          out_dtype: torch.dtype) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b,
                                       alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(
        cuda_device_capability)


def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    if current_platform.is_rocm():
        triton_scaled_mm_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm")
        triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype
    assert azp is None or azp.numel() == a.shape[0]

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj,
                                       azp, bias)
    return out


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(
        cuda_device_capability)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)

def cutlass_sparse_compress(a: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    assert (a.dtype in [
        torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16
    ])
    assert (a.is_contiguous())

    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    assert (a.shape[1] % (2 * elemsPerMetaElem) == 0)

    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
        a: torch.Tensor,
        bt_nzs: torch.Tensor,
        bt_meta: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    assert (bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] \
        and bias.dtype == out_dtype

    m = a.shape[0]
    n = bt_nzs.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_sparse_mm(out, a, bt_nzs, bt_meta, scale_a,
                                          scale_b, bias)

    return out


def get_cutlass_moe_mm_data(
        topk_ids: torch.Tensor, expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor, problem_sizes2: torch.Tensor,
        input_permutation: torch.Tensor, output_permutation: torch.Tensor,
        num_experts: int, n: int, k: int):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    """
    torch.ops._C.get_cutlass_moe_mm_data(topk_ids, expert_offsets,
                                         problem_sizes1, problem_sizes2,
                                         input_permutation, output_permutation,
                                         num_experts, n, k)


def cutlass_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor,
                   b_tensors: torch.Tensor, a_scales: torch.Tensor,
                   b_scales: torch.Tensor, expert_offsets: torch.Tensor,
                   problem_sizes: torch.Tensor, a_strides: torch.Tensor,
                   b_strides: torch.Tensor, c_strides: torch.Tensor):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    torch.ops._C.cutlass_moe_mm(out_tensors, a_tensors, b_tensors, a_scales,
                                b_scales, expert_offsets, problem_sizes,
                                a_strides, b_strides, c_strides)


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: list[int],
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.ops._C.aqlm_gemm(input, codes, codebooks, scales,
                                  codebook_partition_sizes, bias)


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: list[int]) -> torch.Tensor:
    return torch.ops._C.aqlm_dequant(codes, codebooks,
                                     codebook_partition_sizes)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                           num_bits)


# gptq_marlin
def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int,
                      num_bits: int) -> torch.Tensor:
    return torch.ops._C.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.gptq_marlin_repack(b_q_weight[e], perm[e],
                                                    size_k, size_n, num_bits)
    return output


def awq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                          size_k: int, size_n: int,
                          num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.awq_marlin_repack(b_q_weight[e], size_k,
                                                   size_n, num_bits)
    return output


def gptq_marlin_gemm(a: torch.Tensor,
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     b_zeros: torch.Tensor,
                     g_idx: torch.Tensor,
                     perm: torch.Tensor,
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool,
                     has_zp: bool = False,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_gemm(a, b_q_weight, b_scales, b_zeros,
                                         g_idx, perm, workspace, b_q_type.id,
                                         size_m, size_n, size_k, is_k_full,
                                         has_zp, use_atomic_add,
                                         use_fp32_reduce, is_zp_float)


# fp8 marlin
def fp8_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    b_scales: torch.Tensor, workspace: torch.Tensor,
                    num_bits: int, size_m: int, size_n: int,
                    size_k: int) -> torch.Tensor:
    return torch.ops._C.fp8_marlin_gemm(a, b_q_weight, b_scales, workspace,
                                        num_bits, size_m, size_n, size_k)


# machete
def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype],
        group_zeros_type: Optional[torch.dtype] = None,
        channel_scales_type: Optional[torch.dtype] = None,
        token_scales_type: Optional[torch.dtype] = None,
        out_type: Optional[torch.dtype] = None) -> list[str]:
    return torch.ops._C.machete_supported_schedules(
        a_type, b_type.id, group_scales_type, group_zeros_type,
        channel_scales_type, token_scales_type, out_type)


def machete_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None) -> torch.Tensor:
    return torch.ops._C.machete_mm(a, b_q, b_type.id, out_type, b_group_scales,
                                   b_group_zeros, b_group_size,
                                   b_channel_scales, a_token_scales, schedule)


def machete_prepack_B(
        b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(b_q_weight, a_type, b_type.id,
                                          group_scales_type)


if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor,
                           perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.permute_cols(a, perm)


# fp4
def scaled_fp4_quant(
        input: torch.Tensor,
        input_global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert not current_platform.is_rocm()
    assert input.ndim >= 1, (
        f'input.ndim needs to be >= 1, but got {input.ndim}.')
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, (
        f'last dim has to be multiple of 16, but got {n}.')
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f'input.dtype needs to be fp16 or bf16 but got {input.dtype}.')

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty((rounded_m, rounded_n // 4),
                               device=device,
                               dtype=torch.int32)

    torch.ops._C.scaled_fp4_quant(output, input, output_scale,
                                  input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    shape: Union[tuple[int, int], torch.Size] = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert (scale.numel() == 1 or num_token_padding is None)
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale


# gptq allspark
def allspark_repack_weight(
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        has_zp: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] :
            rearranged weight, scale, and optionally zero_point.
    """
    K = qweight.shape[0]
    N = qweight.shape[1]
    N_32align = (N + 32 - 1) // 32 * 32

    qweight_reorder = torch.empty((N_32align, K),
                                  device=qweight.device,
                                  dtype=qweight.dtype)
    scale_reorder = torch.empty((1, N_32align),
                                device=scale.device,
                                dtype=scale.dtype)
    zero_point_reorder = None
    if has_zp:
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization.")
        zero_point_reorder = torch.empty((1, N_32align),
                                         device=zero_point.device,
                                         dtype=zero_point.dtype)

    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight, scale, zero_point, has_zp, qweight_reorder, scale_reorder,
        zero_point_reorder, K, N, N_32align)

    return qweight_reorder, scale_reorder, zero_point_reorder


def allspark_w8a16_gemm(a: torch.Tensor, b_qweight: torch.Tensor,
                        b_scales: torch.Tensor,
                        b_qzeros: Optional[torch.Tensor], n: int,
                        group_size: int, sm_count: int, sm_version: int,
                        CUBLAS_M_THRESHOLD: int, has_zp: bool,
                        n32k16_reorder: bool) -> torch.Tensor:

    return torch.ops._C.allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros,
                                            n, group_size, sm_count,
                                            sm_version, CUBLAS_M_THRESHOLD,
                                            has_zp, n32k16_reorder)


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp


# qqq ops
def marlin_qqq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    s_tok: torch.Tensor, s_ch: torch.Tensor,
                    s_group: torch.Tensor, workspace: torch.Tensor,
                    size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_qqq_gemm(a, b_q_weight, s_tok, s_ch, s_group,
                                        workspace, size_m, size_n, size_k)


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int, n: int,
                    dtype: Optional[torch.dtype]) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8(X, W, sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, quant_type, row,
                                    top_k, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor,
                      bias_: Optional[torch.Tensor],
                      conv_states: Optional[torch.Tensor],
                      query_start_loc: Optional[torch.Tensor],
                      cache_indices: Optional[torch.Tensor],
                      has_initial_state: Optional[torch.Tensor],
                      silu_activation: bool, pad_slot_id: int):
    torch.ops._C.causal_conv1d_fwd(x, weight, bias_, conv_states,
                                   query_start_loc, cache_indices,
                                   has_initial_state, silu_activation,
                                   pad_slot_id)


def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor,
                         weight: torch.Tensor, bias_: Optional[torch.Tensor],
                         silu_activation: bool,
                         cache_seqlens: Optional[torch.Tensor],
                         conv_state_indices: Optional[torch.Tensor],
                         pad_slot_id: int):
    torch.ops._C.causal_conv1d_update(x, conv_state, weight, bias_,
                                      silu_activation, cache_seqlens,
                                      conv_state_indices, pad_slot_id)


def selective_scan_fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor,
                       D_: Optional[torch.Tensor], z_: Optional[torch.Tensor],
                       delta_bias_: Optional[torch.Tensor],
                       delta_softplus: bool,
                       query_start_loc: Optional[torch.Tensor],
                       cache_indices: Optional[torch.Tensor],
                       has_initial_state: Optional[torch.Tensor],
                       ssm_states: torch.Tensor, pad_slot_id: int):
    torch.ops._C.selective_scan_fwd(u, delta, A, B, C, D_, z_, delta_bias_,
                                    delta_softplus, query_start_loc,
                                    cache_indices, has_initial_state,
                                    ssm_states, pad_slot_id)


# ROCm skinny gemms
def LLMM1(a: torch.Tensor, b: torch.Tensor,
          rows_per_block: int) -> torch.Tensor:
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


def wvSplitK(a: torch.Tensor, b: torch.Tensor, cu_count: int) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(a, b, cu_count)


def wvSplitKQ(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype,
              scale_a: torch.Tensor, scale_b: torch.Tensor,
              cu_count: int) -> torch.Tensor:
    out = torch.empty((b.shape[0], a.shape[0]),
                      dtype=out_dtype,
                      device=b.device)
    torch.ops._rocm_C.wvSplitKQ(a, b, out, scale_a, scale_b, cu_count)
    return out


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    torch.ops._moe_C.moe_sum(input, output)


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.moe_align_block_size(topk_ids, num_experts, block_size,
                                          sorted_token_ids, experts_ids,
                                          num_tokens_post_pad)


def sgl_moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                             block_size: int, sorted_token_ids: torch.Tensor,
                             experts_ids: torch.Tensor,
                             num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.sgl_moe_align_block_size(topk_ids, num_experts,
                                              block_size, sorted_token_ids,
                                              experts_ids, num_tokens_post_pad)


def moe_wna16_gemm(input: torch.Tensor, output: torch.Tensor,
                   b_qweight: torch.Tensor, b_scales: torch.Tensor,
                   b_qzeros: Optional[torch.Tensor],
                   topk_weights: Optional[torch.Tensor],
                   sorted_token_ids: torch.Tensor, experts_ids: torch.Tensor,
                   num_tokens_post_pad: torch.Tensor, top_k: int,
                   BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
                   bit: int) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only "
            "available on CUDA platforms")
    torch.ops._moe_C.moe_wna16_gemm(input, output, b_qweight, b_scales,
                                    b_qzeros, topk_weights, sorted_token_ids,
                                    experts_ids, num_tokens_post_pad, top_k,
                                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                    bit)


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: torch.Tensor) -> None:
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids,
                                  token_expert_indicies, gating_output)


def moe_wna16_marlin_gemm(input: torch.Tensor, output: Optional[torch.Tensor],
                          b_qweight: torch.Tensor, b_scales: torch.Tensor,
                          b_qzeros: Optional[torch.Tensor],
                          g_idx: Optional[torch.Tensor],
                          perm: Optional[torch.Tensor],
                          workspace: torch.Tensor,
                          sorted_token_ids: torch.Tensor,
                          expert_ids: torch.Tensor,
                          num_tokens_past_padded: torch.Tensor,
                          topk_weights: torch.Tensor, moe_block_size: int,
                          top_k: int, mul_topk_weights: bool, is_ep: bool,
                          b_q_type: ScalarType, size_m: int, size_n: int,
                          size_k: int, is_k_full: bool, use_atomic_add: bool,
                          use_fp32_reduce: bool,
                          is_zp_float: bool) -> torch.Tensor:
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        input, output, b_qweight, b_scales, b_qzeros, g_idx, perm, workspace,
        sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
        moe_block_size, top_k, mul_topk_weights, is_ep, b_q_type.id, size_m,
        size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce,
        is_zp_float)


if supports_moe_ops and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(a: torch.Tensor, b_q_weights: torch.Tensor,
                             sorted_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             topk_ids: torch.Tensor, b_scales: torch.Tensor,
                             b_zero_points: torch.Tensor, g_idx: torch.Tensor,
                             perm: torch.Tensor, workspace: torch.Tensor,
                             b_q_type: ScalarType, size_m: torch.SymInt,
                             size_n: torch.SymInt, size_k: torch.SymInt,
                             is_k_full: bool, num_experts: int, topk: int,
                             moe_block_size: int, replicate_input: bool,
                             apply_weights: bool) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n),
                           dtype=a.dtype,
                           device=a.device)

    @register_fake("_moe_C::moe_wna16_marlin_gemm")
    def moe_wna16_marlin_gemm_fake(input: torch.Tensor,
                                   output: Optional[torch.Tensor],
                                   b_qweight: torch.Tensor,
                                   b_scales: torch.Tensor,
                                   b_qzeros: Optional[torch.Tensor],
                                   g_idx: Optional[torch.Tensor],
                                   perm: Optional[torch.Tensor],
                                   workspace: torch.Tensor,
                                   sorted_token_ids: torch.Tensor,
                                   expert_ids: torch.Tensor,
                                   num_tokens_past_padded: torch.Tensor,
                                   topk_weights: torch.Tensor,
                                   moe_block_size: int, top_k: int,
                                   mul_topk_weights: bool, is_ep: bool,
                                   b_q_type: ScalarType, size_m: int,
                                   size_n: int, size_k: int, is_k_full: bool,
                                   use_atomic_add: bool, use_fp32_reduce: bool,
                                   is_zp_float: bool) -> torch.Tensor:
        return torch.empty((size_m * top_k, size_n),
                           dtype=input.dtype,
                           device=input.device)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(key, value, key_cache,
                                             value_cache, slot_mapping,
                                             kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale)


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache,
                                                slot_mapping, kv_cache_dtype,
                                                scale)


def copy_blocks(key_caches: list[torch.Tensor],
                value_caches: list[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def copy_blocks_mla(kv_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def gather_cache(src_cache: torch.Tensor,
                 dst: torch.Tensor,
                 block_table: torch.Tensor,
                 cu_seq_lens: torch.Tensor,
                 batch_size: int,
                 seq_starts: Optional[torch.Tensor] = None) -> None:
    torch.ops._C_cache_ops.gather_cache(src_cache, dst, block_table,
                                        cu_seq_lens, batch_size, seq_starts)


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device)


# custom ar
def init_custom_ar(ipc_tensors: list[torch.Tensor], rank_data: torch.Tensor,
                   rank: int, fully_connected: bool) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(ipc_tensors, rank_data, rank,
                                                 fully_connected)


def all_reduce(fa: int, inp: torch.Tensor, out: torch.Tensor, reg_buffer: int,
               reg_buffer_sz_bytes: int) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer,
                                      reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa: int, handles: list[list[int]],
                           offsets: list[list[int]]) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._C.get_flash_mla_metadata(cache_seqlens,
                                               num_heads_per_head_k,
                                               num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def cutlass_mla_decode(out: torch.Tensor, q_nope: torch.Tensor,
                       q_pe: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor,
                       seq_lens: torch.Tensor, page_table: torch.Tensor,
                       scale: float) -> torch.Tensor:
    torch.ops._C.cutlass_mla_decode(out, q_nope, q_pe, kv_c_and_k_pe_cache,
                                    seq_lens, page_table, scale)
    return out
''',
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
