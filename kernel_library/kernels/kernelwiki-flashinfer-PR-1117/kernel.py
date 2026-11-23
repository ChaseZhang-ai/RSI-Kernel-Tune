"""KernelWiki artifact PR-1117 (flashinfer-ai/flashinfer)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/csrc/batch_decode.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <optional>

#include "batch_decode_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    at::Tensor empty_q_data, at::Tensor empty_kv_data) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;

  auto q_scalar_type = empty_q_data.scalar_type();
  auto kv_scalar_type = empty_kv_data.scalar_type();

  TORCH_CHECK(head_dim_qk == head_dim_vo,
              "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
              "cores template for different head dim");

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
          cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
              static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
              static_cast<void*>(int_workspace_buffer.data_ptr()),
              static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
              batch_size, num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/stream, work_estimation_func);

          TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                      cudaGetErrorString(status));
          return true;
        });
      });

  return vec_to_tensor(plan_info.ToVector());
}

void BatchDecodeWithPagedKVCacheRun(at::Tensor float_workspace_buffer,
                                    at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                    at::Tensor q, at::Tensor paged_k_cache,
                                    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
                                    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
                                    at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  auto device = q.device();
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;

  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }
  uint32_t head_dim_qk = q.size(2);
  uint32_t head_dim_vo = paged_v_cache.size(3);

  TORCH_CHECK(head_dim_qk == head_dim_vo,
              "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
              "cores template for different head dim");

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q.size(1));
  }

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  // get q_scalar_type and kv_scalar_type
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_QK, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));

        Params params;
        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.paged_kv = paged_kv;
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.padded_batch_size = 0;
        params.num_qo_heads = num_qo_heads;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;
        params.request_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;

        cudaError_t status =
            flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM_QK, POS_ENCODING_MODE,
                                                              AttentionVariant>(params, tmp_v,
                                                                                tmp_s, enable_pdl,
                                                                                /*stream=*/stream);
        TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}
""",
    'key-files/csrc/batch_decode_jit_pybind.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_decode_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    at::Tensor empty_q_data, at::Tensor empty_kv_data);

void BatchDecodeWithPagedKVCacheRun(at::Tensor float_workspace_buffer,
                                    at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                    at::Tensor q, at::Tensor paged_k_cache,
                                    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
                                    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
                                    at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batched decode with paged KV-Cache plan
  m.def("plan", BatchDecodeWithPagedKVCachePlan);
  // Batched decode with paged KV-Cache run
  m.def("run", BatchDecodeWithPagedKVCacheRun);
}
""",
    'key-files/csrc/batch_decode_mla_cute_sm80.cu': r"""
#include <flashinfer/attention/decode_mla_cute_sm80.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "mla_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

at::Tensor BatchDecodeWithPagedKVCachePlanMLA(at::Tensor float_workspace_buffer,
                                              at::Tensor int_workspace_buffer,
                                              at::Tensor page_locked_int_workspace_buffer,
                                              at::Tensor indptr, int64_t batch_size,
                                              int64_t num_qo_heads, int64_t page_size,
                                              bool enable_cuda_graph) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMlaCuteSM80<
      HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, AttentionVariant, Params>;
  cudaError_t status =
      DecodePlan<HEAD_DIM_CKV, flashinfer::PosEncodingMode::kNone, AttentionVariant, Params>(
          static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
          static_cast<void*>(int_workspace_buffer.data_ptr()),
          static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
          int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
          batch_size, num_qo_heads, page_size, enable_cuda_graph, /*stream=*/stream,
          work_estimation_func);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCachePlanMLA failed with error ",
              cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q_nope, at::Tensor q_pe, at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, double sm_scale, int64_t window_left, double logits_soft_cap, double rope_scale,
    double rope_theta, std::optional<at::Tensor> maybe_lse,
    bool enable_pdl  // fake placeholder, sm80 does not support pdl
) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  auto device = q_nope.device();
  int64_t batch_size = q_nope.size(0);
  int64_t num_qo_heads = q_nope.size(1);
  int64_t page_size = paged_ckv_cache.size(1);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q_nope.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q_nope.size(1));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_mla_t<DTypeKV, IdType> paged_kv(
      page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
      static_cast<DTypeKV*>(paged_ckv_cache.data_ptr()), paged_ckv_cache.strides().data(),
      static_cast<DTypeKV*>(paged_kpe_cache.data_ptr()), paged_kpe_cache.strides().data(),
      static_cast<IdType*>(paged_kv_indices.data_ptr()),
      static_cast<IdType*>(paged_kv_indptr.data_ptr()),
      static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
  Params params(static_cast<DTypeQ*>(q_nope.data_ptr()), static_cast<DTypeQ*>(q_pe.data_ptr()),
                /*q_offset=*/nullptr, paged_kv, static_cast<DTypeO*>(o.data_ptr()),
                /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
                num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  params.request_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;

  const c10::cuda::OptionalCUDAGuard device_guard(paged_ckv_cache.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t status = BatchDecodeWithPagedKVCacheDispatchedMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                                                        QO_TILE_LEN, Params>(
      params, tmp_v, tmp_s, /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
""",
    'key-files/csrc/batch_decode_mla_plan.cu': r"""#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "mla_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

at::Tensor BatchDecodeWithPagedKVCachePlanMLA(at::Tensor float_workspace_buffer,
                                              at::Tensor int_workspace_buffer,
                                              at::Tensor page_locked_int_workspace_buffer,
                                              at::Tensor indptr, int64_t batch_size,
                                              int64_t num_qo_heads, int64_t page_size,
                                              bool enable_cuda_graph) {
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  DecodePlanInfo plan_info;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  auto work_estimation_func =
      BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                                             AttentionVariant, Params>;
  cudaError_t status =
      DecodePlan<HEAD_DIM_CKV, flashinfer::PosEncodingMode::kRoPELlama, AttentionVariant, Params>(
          static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
          static_cast<void*>(int_workspace_buffer.data_ptr()),
          static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
          int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
          batch_size, num_qo_heads, page_size, enable_cuda_graph, /*stream=*/stream,
          work_estimation_func);

  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCachePlanMLA failed with error ",
              cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}
""",
    'key-files/csrc/batch_decode_mla_pybind.cu': r"""#include "mla_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchDecodeWithPagedKVCachePlanMLA(at::Tensor float_workspace_buffer,
                                              at::Tensor int_workspace_buffer,
                                              at::Tensor page_locked_int_workspace_buffer,
                                              at::Tensor indptr, int64_t batch_size,
                                              int64_t num_qo_heads, int64_t page_size,
                                              bool enable_cuda_graph);

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q_nope, at::Tensor q_pe, at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, double sm_scale, int64_t window_left, double logits_soft_cap, double rope_scale,
    double rope_theta, std::optional<at::Tensor> maybe_lse, bool enable_pdl);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("plan", BatchDecodeWithPagedKVCachePlanMLA);
  m.def("run", BatchDecodeWithPagedKVCacheRunMLA);
}
""",
    'key-files/csrc/batch_decode_mla_run.cu': r"""#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <optional>

#include "mla_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q_nope, at::Tensor q_pe, at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, double sm_scale, int64_t window_left, double logits_soft_cap, double rope_scale,
    double rope_theta, std::optional<at::Tensor> maybe_lse, bool enable_pdl, int64_t cuda_stream) {
  DecodePlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  auto device = q_nope.device();
  int64_t batch_size = q_nope.size(0);
  int64_t num_qo_heads = q_nope.size(1);
  int64_t page_size = paged_ckv_cache.size(1);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size, lse.size(0), q_nope.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q_nope.size(1));
  }

  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  paged_kv_mla_t<DTypeKV, IdType> paged_kv(
      page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
      static_cast<DTypeKV*>(paged_ckv_cache.data_ptr()), paged_ckv_cache.strides().data(),
      static_cast<DTypeKV*>(paged_kpe_cache.data_ptr()), paged_kpe_cache.strides().data(),
      static_cast<IdType*>(paged_kv_indices.data_ptr()),
      static_cast<IdType*>(paged_kv_indptr.data_ptr()),
      static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
  Params params(static_cast<DTypeQ*>(q_nope.data_ptr()), static_cast<DTypeQ*>(q_pe.data_ptr()),
                /*q_offset=*/nullptr, paged_kv, static_cast<DTypeO*>(o.data_ptr()),
                /*lse=*/(maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr),
                num_qo_heads, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);

  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  params.request_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask =
          GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionVariant,
                                               Params>(params, tmp_v, tmp_s, enable_pdl,
                                                       /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess, "BatchDecodeWithPagedKVCache failed with error ",
              cudaGetErrorString(status));
}
""",
    'key-files/csrc/batch_prefill.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "batch_prefill_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  PrefillPlanInfo plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
      kv_indptr.data_ptr<IdType>(), total_num_rows, batch_size, num_qo_heads, num_kv_heads,
      head_dim_qk, head_dim_vo, page_size, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "Failed to plan prefill with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheRun(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                      at::Tensor q, at::Tensor k, at::Tensor v,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o,
                                      std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q.size(1);
  int64_t head_dim_qk = q.size(2);
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k.size(1) : k.size(0);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), k_stride_n, k_stride_h, v_stride_n,
           v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k.stride(0);
    k_stride_h = k.stride(1);
    v_stride_n = v.stride(0);
    v_stride_h = v.stride(1);
  } else {
    k_stride_h = k.stride(0);
    k_stride_n = k.stride(1);
    v_stride_h = v.stride(0);
    v_stride_n = v.stride(1);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        params.k = static_cast<DTypeKV*>(k.data_ptr());
        params.v = static_cast<DTypeKV*>(v.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());
        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.kv_indptr = static_cast<IdType*>(kv_indptr.data_ptr());
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.k_stride_n = k_stride_n;
        params.k_stride_h = k_stride_h;
        params.v_stride_n = v_stride_n;
        params.v_stride_h = v_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithRaggedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  auto device = q.device();
  int64_t batch_size = paged_kv_indptr.size(0) - 1;
  int64_t num_qo_heads = q.size(1);
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q.size(2);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer.data_ptr());

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q.data_ptr());
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
            static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices.data_ptr()),
            static_cast<IdType*>(paged_kv_indptr.data_ptr()),
            static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));
        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
        params.o = static_cast<DTypeO*>(o.data_ptr());

        params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        TORCH_CHECK(status == cudaSuccess, "BatchPrefillWithPagedKVCache failed with error ",
                    cudaGetErrorString(status));
        return true;
      });
}
""",
    'key-files/csrc/batch_prefill_fp8_sm90.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "batch_prefill_sm90_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchFP8PrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                      cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchPrefillWithKVCacheSM90Plan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  flashinfer::PrefillPlanSM90Info plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cudaError_t status =
      PrefillSM90Plan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
                      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
                      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
                      kv_indptr.data_ptr<IdType>(), kv_len_arr.data_ptr<IdType>(), total_num_rows,
                      batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
                      causal, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "PrefillSM90Plan failed with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheSM90Run(at::Tensor float_workspace_buffer,
                                          at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                          at::Tensor q, at::Tensor k, at::Tensor v,
                                          at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o,
                                          std::optional<at::Tensor> maybe_lse,
                                          int64_t mask_mode_code, int64_t layout,
                                          int64_t window_left,
                                          bool enable_pdl  // placeholder
                                              ADDITIONAL_FUNC_PARAMS) {
  return;  // TODO: Implement this function
}

void BatchPrefillWithPagedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t num_kv_heads, page_size;
  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = paged_v_cache.size(3);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_swa = window_left != -1;

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
        params.k_ptr = static_cast<DTypeKV*>(paged_k_cache.data_ptr());
        params.v_ptr = static_cast<DTypeKV*>(paged_v_cache.data_ptr());
        params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
        params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.q_stride_n = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.o_stride_n = o.stride(0);
        params.o_stride_h = o.stride(1);
        if (kv_layout == QKVLayout::kNHD) {
          // (num_pages, page_size, num_heads, head_dim)
          params.k_stride_n = paged_k_cache.stride(1);
          params.k_stride_h = paged_k_cache.stride(2);
          params.v_stride_n = paged_v_cache.stride(1);
          params.v_stride_h = paged_v_cache.stride(2);
        } else {
          // (num_pages, num_heads, page_size, head_dim)
          params.k_stride_h = paged_k_cache.stride(1);
          params.k_stride_n = paged_k_cache.stride(2);
          params.v_stride_h = paged_v_cache.stride(1);
          params.v_stride_n = paged_v_cache.stride(2);
        }
        params.nnz_qo = q.size(0);
        params.num_qo_heads = q.size(1);
        params.num_kv_heads = num_kv_heads;
        params.group_size = params.num_qo_heads / num_kv_heads;
        params.page_size = page_size;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.batch_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.batch_indices_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.kv_indices = static_cast<IdType*>(paged_kv_indices.data_ptr());

        ADDITIONAL_PARAMS_SETTER

        // Not support various head_dim for now
        static_assert(HEAD_DIM_QK == HEAD_DIM_VO, "head_dim_qk and head_dim_vo should be the same");
        // Currently only support same quantization precision
        static_assert(std::is_same_v<DTypeQ, DTypeKV>);

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status =
              BatchFP8PrefillWithPagedKVCacheDispatched<HEAD_DIM_QK, MASK_MODE, USE_SLIDING_WINDOW,
                                                        SAME_SCHEDULER_FOR_ALL_HEADS,
                                                        AttentionVariant>(params, enable_pdl,
                                                                          stream);
          TORCH_CHECK(status == cudaSuccess,
                      "BatchPrefillWithPagedKVCacheSM90Run failed with error: ",
                      cudaGetErrorString(status));
          return true;
        });
      });
}
""",
    'key-files/csrc/batch_prefill_jit_pybind.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_prefill_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal);

void BatchPrefillWithRaggedKVCacheRun(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                      at::Tensor q, at::Tensor k, at::Tensor v,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o,
                                      std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout, int64_t window_left,
                                      bool enable_pdl ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batch-request prefill attention with KV-Cache plan
  m.def("plan", BatchPrefillWithKVCachePlan);
  // Batch-request prefill attention with KV-Cache operator
  m.def("ragged_run", BatchPrefillWithRaggedKVCacheRun);
  // Batch-request prefill attention with KV-Cache operator
  m.def("paged_run", BatchPrefillWithPagedKVCacheRun);
}
""",
    'key-files/csrc/batch_prefill_sm90.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "batch_prefill_sm90_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                    cudaStream_t stream);

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                   cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

at::Tensor BatchPrefillWithKVCacheSM90Plan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  flashinfer::PrefillPlanSM90Info plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cudaError_t status =
      PrefillSM90Plan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
                      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
                      int_workspace_size_in_bytes, plan_info, qo_indptr.data_ptr<IdType>(),
                      kv_indptr.data_ptr<IdType>(), kv_len_arr.data_ptr<IdType>(), total_num_rows,
                      batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
                      causal, enable_cuda_graph, /*sizeof_dtype_o=*/2, stream);

  TORCH_CHECK(status == cudaSuccess,
              "PrefillSM90Plan failed with error: ", cudaGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchPrefillWithRaggedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = v.size(2);

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_swa = window_left != -1;

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
        params.k_ptr = static_cast<DTypeKV*>(k.data_ptr());
        params.v_ptr = static_cast<DTypeKV*>(v.data_ptr());
        params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
        params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.q_stride_n = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.o_stride_n = o.stride(0);
        params.o_stride_h = o.stride(1);
        if (kv_layout == QKVLayout::kNHD) {
          params.k_stride_n = k.stride(0);
          params.k_stride_h = k.stride(1);
          params.v_stride_n = v.stride(0);
          params.v_stride_h = v.stride(1);
        } else {
          params.k_stride_h = k.stride(0);
          params.k_stride_n = k.stride(1);
          params.v_stride_h = v.stride(0);
          params.v_stride_n = v.stride(1);
        }
        params.nnz_qo = q.size(0);
        params.nnz_kv = k.size(0);
        params.num_qo_heads = q.size(1);
        params.num_kv_heads = k.size(1);
        params.group_size = params.num_qo_heads / params.num_kv_heads;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.batch_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.batch_indices_offset);

        ADDITIONAL_PARAMS_SETTER

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithRaggedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, enable_pdl, stream);
          TORCH_CHECK(status == cudaSuccess,
                      "BatchPrefillWithRaggedKVCacheSM90Run failed with error: ",
                      cudaGetErrorString(status));
          return true;
        });
      });
}

void BatchPrefillWithPagedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS) {
  PrefillPlanSM90Info plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == q.size(0), lse.size(0), q.size(0));
    TORCH_CHECK(lse.size(1) == q.size(1), lse.size(1), q.size(1));
  }
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t num_kv_heads, page_size;
  int64_t head_dim_qk = q.size(2);
  int64_t head_dim_vo = paged_v_cache.size(3);
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = paged_k_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  bool use_swa = window_left != -1;

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q.data_ptr());
        params.k_ptr = static_cast<DTypeKV*>(paged_k_cache.data_ptr());
        params.v_ptr = static_cast<DTypeKV*>(paged_v_cache.data_ptr());
        params.o_ptr = static_cast<DTypeO*>(o.data_ptr());
        params.lse_ptr = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
        params.q_stride_n = q.stride(0);
        params.q_stride_h = q.stride(1);
        params.o_stride_n = o.stride(0);
        params.o_stride_h = o.stride(1);
        if (kv_layout == QKVLayout::kNHD) {
          // (num_pages, page_size, num_heads, head_dim)
          params.k_stride_n = paged_k_cache.stride(1);
          params.k_stride_h = paged_k_cache.stride(2);
          params.v_stride_n = paged_v_cache.stride(1);
          params.v_stride_h = paged_v_cache.stride(2);
        } else {
          // (num_pages, num_heads, page_size, head_dim)
          params.k_stride_h = paged_k_cache.stride(1);
          params.k_stride_n = paged_k_cache.stride(2);
          params.v_stride_h = paged_v_cache.stride(1);
          params.v_stride_n = paged_v_cache.stride(2);
        }
        params.nnz_qo = q.size(0);
        params.num_qo_heads = q.size(1);
        params.num_kv_heads = num_kv_heads;
        params.group_size = params.num_qo_heads / num_kv_heads;
        params.page_size = page_size;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.batch_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.batch_indices_offset);
        params.kv_indices = static_cast<IdType*>(paged_kv_indices.data_ptr());

        ADDITIONAL_PARAMS_SETTER

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, enable_pdl, stream);
          TORCH_CHECK(status == cudaSuccess,
                      "BatchPrefillWithPagedKVCacheSM90Run failed with error: ",
                      cudaGetErrorString(status));
          return true;
        });
      });
}
""",
    'key-files/csrc/batch_prefill_sm90_jit_pybind.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_prefill_sm90_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchPrefillWithKVCacheSM90Plan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal);

void BatchPrefillWithRaggedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batch-request prefill attention with KV-Cache plan
  m.def("plan", BatchPrefillWithKVCacheSM90Plan);
  // Batch-request prefill attention with KV-Cache operator
  m.def("ragged_run", BatchPrefillWithRaggedKVCacheSM90Run);
  // Batch-request prefill attention with KV-Cache operator
  m.def("paged_run", BatchPrefillWithPagedKVCacheSM90Run);
}
""",
    'key-files/csrc/flashinfer_ops.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aot_default_additional_params.h"
#include "pytorch_extension_utils.h"

//========== activation ==========

void silu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl);
void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl);
void gelu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl);

//========== cascade ==========

void merge_state(at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_merged, at::Tensor s_merged);

void merge_state_in_place(at::Tensor v, at::Tensor s, at::Tensor v_other, at::Tensor s_other,
                          std::optional<at::Tensor> mask);

void merge_states(at::Tensor v, at::Tensor s, at::Tensor v_merged, at::Tensor s_merged);

//========== decode ==========

void single_decode_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                 at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t layout,
                                 int64_t window_left SINGLE_DECODE_ADDITIONAL_FUNC_PARAMS);

at::Tensor BatchDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    at::Tensor empty_q_data, at::Tensor empty_kv_data);

void BatchDecodeWithPagedKVCacheRun(at::Tensor float_workspace_buffer,
                                    at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                    at::Tensor q, at::Tensor paged_k_cache,
                                    at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
                                    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
                                    at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                    int64_t kv_layout_code, int64_t window_left,
                                    bool enable_pdl BATCH_DECODE_ADDITIONAL_FUNC_PARAMS);

//========== gemm ==========

void bmm_fp8(at::Tensor A, at::Tensor B, at::Tensor D, at::Tensor A_scale, at::Tensor B_scale,
             at::Tensor workspace_buffer, int64_t cublas_handle);

void CutlassSegmentGEMM(at::Tensor workspace_buffer, at::Tensor all_problems, at::Tensor x_ptr,
                        at::Tensor w_ptr, at::Tensor y_ptr, at::Tensor x_ld, at::Tensor w_ld,
                        at::Tensor y_ld, at::Tensor empty_x_data, bool weight_column_major);

//========== norm ==========

void rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps, bool enable_pdl);

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                       bool enable_pdl);

void gemma_rmsnorm(at::Tensor& out, at::Tensor& input, at::Tensor& weight, double eps,
                   bool enable_pdl);

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight,
                             double eps, bool enable_pdl);

//========== page ==========

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t layout);

void append_paged_mla_kv_cache(at::Tensor append_ckv, at::Tensor append_kpe,
                               at::Tensor batch_indices, at::Tensor positions, at::Tensor ckv_cache,
                               at::Tensor kpe_cache, at::Tensor kv_indices, at::Tensor kv_indptr,
                               at::Tensor kv_last_page_len);

void block_sparse_indices_to_vector_sparse_offsets(
    at::Tensor block_sparse_indices, at::Tensor block_sparse_indptr,
    at::Tensor vector_sparse_offsets, at::Tensor vector_sparse_indptr, at::Tensor kv_len_arr,
    int64_t stride_block, int64_t stride_n, int64_t batch_size, int64_t block_size);

//========== prefill ==========

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  int64_t mask_mode_code, int64_t layout,
                                  int64_t window_left SINGLE_PREFILL_ADDITIONAL_FUNC_PARAMS);

at::Tensor BatchPrefillWithKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal);

void BatchPrefillWithRaggedKVCacheRun(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
                                      at::Tensor q, at::Tensor k, at::Tensor v,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr, at::Tensor o,
                                      std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                                      int64_t layout,
                                      int64_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left BATCH_PREFILL_ADDITIONAL_FUNC_PARAMS);

//========== pod-attention =========
void pod_with_kv_cache_tensor(
    // Prefill params
    at::Tensor q_p, at::Tensor k_p, at::Tensor v_p, at::Tensor tmp_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_alibi_slopes_p, double logits_soft_cap_p, double sm_scale_p,
    double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl);
//========== quantization ==========

void packbits(at::Tensor x, const std::string& bitorder, at::Tensor y);

void segment_packbits(at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr,
                      const std::string& bitorder, at::Tensor y);

//========== rope ==========

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta);

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length);

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta);

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length);

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_sin_cache,
                                      at::Tensor pos_ids, bool interleave);

//========== sampling ==========

void sampling_from_probs(at::Tensor probs, at::Tensor output,
                         std::optional<at::Tensor> maybe_indices, bool deterministic,
                         std::optional<at::Generator> gen);

void sampling_from_logits(at::Tensor logits, at::Tensor output,
                          std::optional<at::Tensor> maybe_indices, bool deterministic,
                          std::optional<at::Generator> gen);

void top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, std::optional<at::Generator> gen);

void top_k_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, std::optional<at::Generator> gen);

void min_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                               std::optional<at::Tensor> maybe_indices,
                               std::optional<at::Tensor> maybe_min_p_arr, double min_p_val,
                               bool deterministic, std::optional<at::Generator> gen);

void top_k_top_p_sampling_from_probs(at::Tensor probs, at::Tensor output,
                                     std::optional<at::Tensor> maybe_indices,
                                     std::optional<at::Tensor> maybe_top_k_arr, double top_k_val,
                                     std::optional<at::Tensor> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, std::optional<at::Generator> gen);

void top_p_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_p_arr, double top_p_val);

void top_k_renorm_probs(at::Tensor probs, at::Tensor renorm_probs,
                        std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val);

void top_k_mask_logits(at::Tensor logits, at::Tensor mask_logits,
                       std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val);

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor target_probs, at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num, bool deterministic,
                                std::optional<at::Generator> gen);

//========== Torch Library ==========

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // activation
  // Fused SiLU and Mul
  m.def("silu_and_mul", silu_and_mul);
  // Fused GeLU Tanh and Mul
  m.def("gelu_tanh_and_mul", gelu_tanh_and_mul);
  // Fused GeLU and Mul
  m.def("gelu_and_mul", gelu_and_mul);

  // cascade
  // Merge two self-attention states
  m.def("merge_state", merge_state);
  // Merge another self-attention state in-place.
  m.def("merge_state_in_place", merge_state_in_place);
  // "Merge multiple self-attention states"
  m.def("merge_states", merge_states);

  // decode
  // "Single-request decode with KV-Cache operator"
  m.def("single_decode_with_kv_cache", single_decode_with_kv_cache);
  m.def("batch_decode_with_paged_kv_cache_plan", BatchDecodeWithPagedKVCachePlan);
  m.def("batch_decode_with_paged_kv_cache_run", BatchDecodeWithPagedKVCacheRun);

  // gemm
  // BMM FP8
  m.def("bmm_fp8", bmm_fp8);
  // Cutlass Segment GEMM operator
  m.def("cutlass_segment_gemm", CutlassSegmentGEMM);

  // norm
  // Root mean square normalization
  m.def("rmsnorm", rmsnorm);
  // Fused add root mean square normalization
  m.def("fused_add_rmsnorm", fused_add_rmsnorm);
  // Gemma Root mean square normalization
  m.def("gemma_rmsnorm", gemma_rmsnorm);
  // Gemma Fused add root mean square normalization
  m.def("gemma_fused_add_rmsnorm", gemma_fused_add_rmsnorm);

  // page
  // Append paged KV-Cache operator
  m.def("append_paged_kv_cache", append_paged_kv_cache);
  // Append paged MLA KV-Cache operator
  m.def("append_paged_mla_kv_cache", append_paged_mla_kv_cache);
  // Precompute block sparse offsets
  m.def("block_sparse_indices_to_vector_sparse_offsets",
        block_sparse_indices_to_vector_sparse_offsets);

  // prefill
  // Single-request prefill attention with KV-Cache operator
  m.def("single_prefill_with_kv_cache", single_prefill_with_kv_cache);
  m.def("batch_prefill_with_kv_cache_plan", BatchPrefillWithKVCachePlan);
  m.def("batch_prefill_with_ragged_kv_cache_run", BatchPrefillWithRaggedKVCacheRun);
  m.def("batch_prefill_with_paged_kv_cache_run", BatchPrefillWithPagedKVCacheRun);

  // pod-attention
  // Temporarily disabled because we don't generate the implementation yet.
  // m.def("pod_with_kv_cache_tensor", pod_with_kv_cache_tensor);

  // quantization
  // GPU packbits operator
  m.def("packbits", packbits);
  // GPU segment packbits operator
  m.def("segment_packbits", segment_packbits);

  // rope
  // "Apply RoPE"
  m.def("apply_rope", apply_rope);
  // "Apply Llama 3.1 style RoPE"
  m.def("apply_llama31_rope", apply_llama31_rope);
  // "Apply RoPE with positional ids"
  m.def("apply_rope_pos_ids", apply_rope_pos_ids);
  // "Apply Llama 3.1 style RoPE with positional ids"
  m.def("apply_llama31_rope_pos_ids", apply_llama31_rope_pos_ids);
  // "Apply RoPE with positional ids and cosine/sine cache"
  m.def("apply_rope_pos_ids_cos_sin_cache", apply_rope_pos_ids_cos_sin_cache);

  // sampling
  // Sample from probabilities
  m.def("sampling_from_probs", sampling_from_probs);
  // Sample from logits
  m.def("sampling_from_logits", sampling_from_logits);
  // Top-k sampling from probabilities
  m.def("top_k_sampling_from_probs", top_k_sampling_from_probs);
  // Min-p sampling from probabilities
  m.def("min_p_sampling_from_probs", min_p_sampling_from_probs);
  // Top-p sampling from probabilities
  m.def("top_p_sampling_from_probs", top_p_sampling_from_probs);
  // Top-k and top-p sampling from probabilities
  m.def("top_k_top_p_sampling_from_probs", top_k_top_p_sampling_from_probs);
  // Renormalize probabilities by top-k mask
  m.def("top_k_renorm_probs", top_k_renorm_probs);
  // Renormalize probabilities by top-p mask
  m.def("top_p_renorm_probs", top_p_renorm_probs);
  // Mask logits by top-k mask
  m.def("top_k_mask_logits", top_k_mask_logits);
  // Speculative sampling from sequence of probabilities
  m.def("chain_speculative_sampling", chain_speculative_sampling);
}
""",
    'key-files/csrc/flashinfer_ops_sm90.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aot_default_additional_params.h"
#include "pytorch_extension_utils.h"

void CutlassSegmentGEMMSM90(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                            at::Tensor all_problems, at::Tensor x_ptr, at::Tensor w_ptr,
                            at::Tensor y_ptr, at::Tensor x_stride, at::Tensor weight_stride,
                            at::Tensor y_stride, at::Tensor empty_x_data, at::Tensor empty_y_data,
                            bool weight_column_major);

void single_prefill_with_kv_cache_sm90(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp, at::Tensor o,
    std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left SINGLE_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS);

at::Tensor BatchPrefillWithKVCacheSM90Plan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal);

void BatchPrefillWithRaggedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_indptr, at::Tensor kv_indptr,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl BATCH_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS);

void BatchPrefillWithPagedKVCacheSM90Run(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor qo_indptr,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code, int64_t layout,
    int64_t window_left, bool enable_pdl BATCH_PREFILL_SM90_ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // "Cutlass Segment GEMM operator for SM90"
  m.def("cutlass_segment_gemm_sm90", CutlassSegmentGEMMSM90);
  m.def("single_prefill_with_kv_cache_sm90", single_prefill_with_kv_cache_sm90);
  m.def("batch_prefill_with_kv_cache_sm90_plan", BatchPrefillWithKVCacheSM90Plan);
  m.def("batch_prefill_with_ragged_kv_cache_sm90_run", BatchPrefillWithRaggedKVCacheSM90Run);
  m.def("batch_prefill_with_paged_kv_cache_sm90_run", BatchPrefillWithPagedKVCacheSM90Run);
}
""",
    'key-files/csrc/pod.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "pod_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q,
          MaskMode MASK_MODE_D, typename PrefillAttentionVariant, typename DecodeAttentionVariant,
          typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params,
                                           typename PrefillParams::DTypeO* tmp,
                                           DecodeParams decode_params,
                                           typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                           bool enable_pdl, cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

void pod_with_kv_cache_tensor(
    // Prefill params
    at::Tensor q_p, at::Tensor k_p, at::Tensor v_p, at::Tensor tmp_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_alibi_slopes_p, double logits_soft_cap_p, double sm_scale_p,
    double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl) {
  // Prefill setup
  unsigned int head_dim_qk = q_p.size(2);
  unsigned int kv_len_p, qo_len_p, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout_p = static_cast<QKVLayout>(layout_p);
  qo_len_p = q_p.size(0);
  num_qo_heads = q_p.size(1);
  uint32_t q_stride_n_p = q_p.stride(0), q_stride_h_p = q_p.stride(1), k_stride_n_p, k_stride_h_p,
           v_stride_n_p, v_stride_h_p;
  if (kv_layout_p == QKVLayout::kNHD) {
    kv_len_p = k_p.size(0);
    num_kv_heads = k_p.size(1);
    k_stride_n_p = k_p.stride(0);
    k_stride_h_p = k_p.stride(1);
    v_stride_n_p = v_p.stride(0);
    v_stride_h_p = v_p.stride(1);
  } else {
    kv_len_p = k_p.size(1);
    num_kv_heads = k_p.size(0);
    k_stride_h_p = k_p.stride(0);
    k_stride_n_p = k_p.stride(1);
    v_stride_h_p = v_p.stride(0);
    v_stride_n_p = v_p.stride(1);
  }
  if (maybe_lse_p) {
    const auto& lse = *maybe_lse_p;
    TORCH_CHECK(lse.size(0) == qo_len_p, lse.size(0), q_p.size(0));
    TORCH_CHECK(lse.size(1) == num_qo_heads, lse.size(1), q_p.size(1));
  }

  const MaskMode mask_mode_p = static_cast<MaskMode>(mask_mode_code_p);

  auto q_scalar_type = q_p.scalar_type();
  auto kv_scalar_type = k_p.scalar_type();

  // Decode setup (Tensor decode = batched prefill)
  PrefillPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));
  QKVLayout kv_layout_d = static_cast<QKVLayout>(layout_d);
  auto device = q_d.device();
  int64_t batch_size = paged_kv_indptr_d.size(0) - 1;
  int64_t num_qo_heads_d = q_d.size(1);

  TORCH_CHECK(num_qo_heads == num_qo_heads_d,
              "POD currently requires same # Query heads for prefill and decode");

  int64_t num_kv_heads_d, page_size_d;
  uint32_t head_dim_qk_d = q_d.size(2);
  if (kv_layout_d == QKVLayout::kHND) {
    num_kv_heads_d = paged_k_cache_d.size(1);
    page_size_d = paged_k_cache_d.size(2);
  } else {
    page_size_d = paged_k_cache_d.size(1);
    num_kv_heads_d = paged_k_cache_d.size(2);
  }
  TORCH_CHECK(num_kv_heads == num_kv_heads_d,
              "POD currently requires same # KV heads for prefill and decode; Prefill: ",
              num_kv_heads, ", Decode: ", num_kv_heads_d);

  if (maybe_lse_d) {
    const auto& lse = *maybe_lse_d;
    TORCH_CHECK(lse.size(0) == q_d.size(0), lse.size(0), q_d.size(0));
    TORCH_CHECK(lse.size(1) == q_d.size(1), lse.size(1), q_d.size(1));
  }

  void* float_buffer_ptr = static_cast<void*>(float_workspace_buffer_d.data_ptr());
  void* int_buffer_ptr = static_cast<void*>(int_workspace_buffer_d.data_ptr());

  const MaskMode mask_mode_d = static_cast<MaskMode>(mask_mode_code_d);
  auto q_scalar_type_d = q_d.scalar_type();
  auto kv_scalar_type_d = paged_k_cache_d.scalar_type();

  // get q_stride_n and q_stride_h
  const auto q_stride_n_d = q_d.stride(0);
  const auto q_stride_h_d = q_d.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides_d = nullptr;
  auto k_strides_d = paged_k_cache_d.strides();
  auto v_strides_d = paged_v_cache_d.strides();
  TORCH_CHECK(k_strides_d == v_strides_d, "k/v strides must be identical");
  kv_cache_strides_d = k_strides_d.data();

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer_d.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(
      MASK_MODE_P, MASK_MODE_D, DTypeQ, DTypeKV, HEAD_DIM_QK, USE_SLIDING_WINDOW_P,
      USE_SLIDING_WINDOW_D, USE_LOGITS_SOFT_CAP, [&] {
        PrefillParams prefill_params;
        {
          // Make params a reference to prefill_params to set values
          PrefillParams& params = prefill_params;
          params.q = static_cast<DTypeQ*>(q_p.data_ptr());
          params.k = static_cast<DTypeKV*>(k_p.data_ptr());
          params.v = static_cast<DTypeKV*>(v_p.data_ptr());
          params.o = static_cast<DTypeO*>(o_p.data_ptr());
          params.lse = maybe_lse_p ? static_cast<float*>(maybe_lse_p->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.num_kv_heads = num_kv_heads;
          params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
          params.qo_len = qo_len_p;
          params.kv_len = kv_len_p;
          params.q_stride_n = q_stride_n_p;
          params.q_stride_h = q_stride_h_p;
          params.k_stride_n = k_stride_n_p;
          params.k_stride_h = k_stride_h_p;
          params.v_stride_n = v_stride_n_p;
          params.v_stride_h = v_stride_h_p;

          params.window_left = window_left_p;
          params.partition_kv = false;

          params.maybe_custom_mask = maybe_custom_mask_p
                                         ? static_cast<uint8_t*>(maybe_custom_mask_p->data_ptr())
                                         : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_p
                                          ? static_cast<float*>(maybe_alibi_slopes_p->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = logits_soft_cap_p;
          params.sm_scale = sm_scale_p;
          params.rope_rcp_scale = rope_rcp_scale_p;
          params.rope_rcp_theta = rope_rcp_theta_p;
        }

        DecodeParams decode_params;
        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        {
          DecodeParams& params = decode_params;
          params.q = static_cast<DTypeQ*>(q_d.data_ptr());
          paged_kv_t<DTypeKV, IdType> paged_kv(
              num_kv_heads, page_size_d, HEAD_DIM_VO, batch_size, kv_layout_d,
              static_cast<DTypeKV*>(paged_k_cache_d.data_ptr()),
              static_cast<DTypeKV*>(paged_v_cache_d.data_ptr()), kv_cache_strides_d,
              static_cast<IdType*>(paged_kv_indices_d.data_ptr()),
              static_cast<IdType*>(paged_kv_indptr_d.data_ptr()),
              static_cast<IdType*>(paged_kv_last_page_len_d.data_ptr()));
          params.paged_kv = paged_kv;
          params.q_indptr = static_cast<IdType*>(qo_indptr_d.data_ptr());
          params.o = static_cast<DTypeO*>(o_d.data_ptr());

          params.lse = maybe_lse_d ? static_cast<float*>(maybe_lse_d->data_ptr()) : nullptr;
          params.num_qo_heads = num_qo_heads;
          params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
          params.q_stride_n = q_stride_n_d;
          params.q_stride_h = q_stride_h_d;
          params.window_left = window_left_d;

          params.request_indices = nullptr;
          params.qo_tile_indices = nullptr;
          params.kv_tile_indices = nullptr;
          params.merge_indptr = nullptr;
          params.o_indptr = nullptr;
          params.kv_chunk_size_ptr = nullptr;
          params.block_valid_mask = nullptr;
          params.total_num_rows = nullptr;
          params.max_total_num_rows = 0;
          params.padded_batch_size = 0;
          params.partition_kv = false;

          params.maybe_mask_indptr = maybe_mask_indptr_d
                                         ? static_cast<int32_t*>(maybe_mask_indptr_d->data_ptr())
                                         : nullptr;
          params.maybe_alibi_slopes = maybe_alibi_slopes_d
                                          ? static_cast<float*>(maybe_alibi_slopes_d->data_ptr())
                                          : nullptr;
          params.logits_soft_cap = logits_soft_cap_d;
          params.sm_scale = sm_scale_d;
          params.rope_rcp_scale = rope_rcp_scale_d;
          params.rope_rcp_theta = rope_rcp_theta_d;

          params.request_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
          params.qo_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
          params.kv_tile_indices =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
          params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
          params.kv_chunk_size_ptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
          if (plan_info.split_kv) {
            params.merge_indptr =
                GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
            tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
            if (plan_info.enable_cuda_graph) {
              params.block_valid_mask =
                  GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
            }
          }
          params.padded_batch_size = plan_info.padded_batch_size;
          params.max_total_num_rows = plan_info.total_num_rows;
          if (plan_info.enable_cuda_graph) {
            params.total_num_rows =
                GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
          }
        }

        constexpr bool use_custom_mask_p = MASK_MODE_P == MaskMode::kCustom;
        using PrefillAttentionVariant =
            DefaultAttention</*use_custom_mask=*/use_custom_mask_p, USE_SLIDING_WINDOW_P,
                             USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
        constexpr bool use_custom_mask_d = MASK_MODE_D == MaskMode::kCustom;
        using DecodeAttentionVariant =
            DefaultAttention</*use_custom_mask=*/use_custom_mask_d, USE_SLIDING_WINDOW_D,
                             USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
        // DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
        constexpr size_t CTA_TILE_Q = 16;
        cudaError_t status = flashinfer::PODWithKVCacheTensorDispatched<
            HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, MASK_MODE_P,
            CTA_TILE_Q, MASK_MODE_D, PrefillAttentionVariant, DecodeAttentionVariant>(
            prefill_params, static_cast<DTypeO*>(tmp_p.data_ptr()), decode_params, tmp_v, tmp_s,
            enable_pdl, stream);
        TORCH_CHECK(status == cudaSuccess, "PODWithKVCache kernel launch failed, error: " +
                                               std::string(cudaGetErrorString(status)));
        //});
      });
}
""",
    'key-files/csrc/pod_jit_pybind.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pod_config.inc"
#include "pytorch_extension_utils.h"

void pod_with_kv_cache_tensor(
    // Prefill params
    at::Tensor q_p, at::Tensor k_p, at::Tensor v_p, at::Tensor tmp_p, at::Tensor o_p,
    std::optional<at::Tensor> maybe_lse_p, int64_t mask_mode_code_p, int64_t layout_p,
    int64_t window_left_p, std::optional<at::Tensor> maybe_custom_mask_p,
    std::optional<at::Tensor> maybe_alibi_slopes_p, double logits_soft_cap_p, double sm_scale_p,
    double rope_rcp_scale_p, double rope_rcp_theta_p,
    // Decode params
    at::Tensor float_workspace_buffer_d, at::Tensor int_workspace_buffer_d,
    at::Tensor plan_info_vec, at::Tensor q_d, at::Tensor paged_k_cache_d,
    at::Tensor paged_v_cache_d, at::Tensor qo_indptr_d, at::Tensor paged_kv_indptr_d,
    at::Tensor paged_kv_indices_d, at::Tensor paged_kv_last_page_len_d, at::Tensor o_d,
    std::optional<at::Tensor> maybe_lse_d, int64_t mask_mode_code_d, int64_t layout_d,
    int64_t window_left_d, std::optional<at::Tensor> maybe_custom_mask_d,
    std::optional<at::Tensor> maybe_mask_indptr_d, std::optional<at::Tensor> maybe_alibi_slopes_d,
    double logits_soft_cap_d, double sm_scale_d, double rope_rcp_scale_d, double rope_rcp_theta_d,
    bool enable_pdl);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Batch-request prefill attention with KV-Cache operator
  m.def("pod_with_kv_cache_tensor", pod_with_kv_cache_tensor);
}
""",
    'key-files/include/flashinfer/attention/cascade.cuh': r"""/*!
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_CASCADE_CUH_
#define FLASHINFER_CASCADE_CUH_

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../utils.cuh"
#include "state.cuh"

namespace flashinfer {

using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

/*!
 * \brief The CUDA kernel that merges the self-attention state of two index sets A and B.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeO The data type of v_merged.
 * \param v_a The partial v of index set A. (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void MergeStateKernel(DTypeIn* __restrict__ v_a, float* __restrict__ s_a,
                                 DTypeIn* __restrict__ v_b, float* __restrict__ s_b,
                                 DTypeO* __restrict__ v_merged, float* __restrict__ s_merged,
                                 uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  float s_a_val = s_a[pos * num_heads + head_idx];
  float s_b_val = s_b[pos * num_heads + head_idx];
  float s_max = max(s_a_val, s_b_val);
  s_a_val = math::ptx_exp2(s_a_val - s_max);
  s_b_val = math::ptx_exp2(s_b_val - s_max);
  float a_scale = s_a_val / (s_a_val + s_b_val);
  float b_scale = s_b_val / (s_a_val + s_b_val);
  vec_t<float, vec_size> v_a_vec, v_b_vec, v_merged_vec;
  v_a_vec.cast_load(v_a + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_b_vec.cast_load(v_b + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_merged_vec[i] = a_scale * v_a_vec[i] + b_scale * v_b_vec[i];
  }
  v_merged_vec.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = math::ptx_log2(s_a_val + s_b_val) + s_max;
  }
}

/*!
 * \brief The CUDA kernel that merges the self-attention state with another state in-place.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DType>
__global__ void MergeStateInPlaceKernel(DType* __restrict__ v, float* __restrict__ s,
                                        DType* __restrict__ v_other, float* __restrict__ s_other,
                                        uint8_t* __restrict__ mask, uint32_t num_heads,
                                        uint32_t head_dim) {
  uint32_t pos = blockIdx.x;

  if (mask != nullptr && mask[pos] == 0) return;

  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t head_idx = ty;

  float s_val = s[pos * num_heads + head_idx];
  float s_other_val = s_other[pos * num_heads + head_idx];
  float s_max = max(s_val, s_other_val);
  s_val = math::ptx_exp2(s_val - s_max);
  s_other_val = math::ptx_exp2(s_other_val - s_max);
  float scale = s_val / (s_val + s_other_val);
  float other_scale = s_other_val / (s_val + s_other_val);
  vec_t<float, vec_size> v_vec, v_other_vec;
  v_vec.cast_load(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_other_vec.cast_load(v_other + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec[i] = scale * v_vec[i] + other_scale * v_other_vec[i];
  }
  v_vec.cast_store(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s != nullptr) {
    s[pos * num_heads + head_idx] = math::ptx_log2(s_val + s_other_val) + s_max;
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                       float* s_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sum(vec_t<float, vec_size>& v, DTypeIn* v_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  v.cast_store(v_smem + ty * head_dim + tx * vec_size);
  v.fill(DTypeIn(0.f));
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    vec_t<float, vec_size> v_iter;
    v_iter.cast_load(v_smem + iter * head_dim + tx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v[i] += v_iter[i];
    }
  }
}

template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void AttentionSumKernel(DTypeIn* __restrict__ V, DTypeO* __restrict__ v_sum,
                                   uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  if (num_index_sets == 0) {
    vec_t<DTypeO, vec_size> v;
    v.fill(DTypeO(0.f));
    v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    return;
  }

  if (num_index_sets == 1) {
    vec_t<DTypeO, vec_size> v;
    v.cast_load(V + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    return;
  }

  vec_t<float, vec_size> v_sum_vec;
  v_sum_vec.fill(0.f);
#pragma unroll 2
  for (uint32_t iter = 0; iter < num_index_sets; ++iter) {
    vec_t<float, vec_size> v;
    v.cast_load(V + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim +
                tx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v_sum_vec[i] += v[i];
    }
  }

  v_sum_vec.cast_store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
}

template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void MergeStatesKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                  DTypeO* __restrict__ v_merged, float* __restrict__ s_merged,
                                  uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  if (num_index_sets == 0) {
    vec_t<DTypeO, vec_size> v;
    v.fill(DTypeO(0.f));
    v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = -math::inf;
    }
    return;
  }

  if (num_index_sets == 1) {
    vec_t<DTypeO, vec_size> v;
    v.cast_load(V + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = S[pos * num_heads + head_idx];
    }
    return;
  }

  state_t<vec_size> st;
#pragma unroll 2
  for (uint32_t iter = 0; iter < num_index_sets; ++iter) {
    float s = S[(pos * num_index_sets + iter) * num_heads + head_idx];
    vec_t<float, vec_size> v;
    v.cast_load(V + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim +
                tx * vec_size);
    st.merge(v, s, 1);
  }

  st.normalize();
  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets,
 *   accelerated for larger number of index sets.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param V The partial v of index sets. (n, num_index_sets, h, d)
 * \param S The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO>
__global__ void MergeStatesLargeNumIndexSetsKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                                   DTypeO* __restrict__ v_merged,
                                                   float* __restrict__ s_merged,
                                                   uint32_t num_index_sets, uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = blockIdx.y;
  state_t<vec_size> st;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;

  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#pragma unroll
  for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
        V + ((pos * num_index_sets + (iter * bdy + ty)) * num_heads + head_idx) * head_dim +
            tx * vec_size,
        (iter * bdy + ty) < num_index_sets);
    cp_async::commit_group();
  }
#pragma unroll 4
  for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
    if (iter % bdx == 0) {
      s_smem[ty * bdx + tx] =
          iter * bdy + (ty * bdx + tx) < num_index_sets
              ? S[(pos * num_index_sets + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
              : 0.f;
      __syncthreads();
    }
    cp_async::wait_group<num_smem_stages - 1>();
    __syncthreads();
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
    if (iter * bdy + ty < num_index_sets) {
      float s = s_smem[(iter % bdx) * bdy + ty];
      st.merge(v, s, 1);
    }
    __syncthreads();
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
        V +
            ((pos * num_index_sets + ((iter + num_smem_stages) * bdy + ty)) * num_heads +
             head_idx) *
                head_dim +
            tx * vec_size,
        (iter + num_smem_stages) * bdy + ty < num_index_sets);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  __syncthreads();

  st.normalize();
  threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
  st.normalize();

  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel to merge self-attention states of multiple index sets, the number of
 * index sets at each position might vary.
 *
 * For CUDA graph support, the kernel can be built with a maximum sequence length and executed
 * using a truncated, dynamic sequence length passed through `seq_len_ptr`.
 *
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param V The partial v of index sets. (nnz, h, d)
 * \param S The logsumexp value of index sets. (nnz, h)
 * \param indptr The start offsets of each position in the variable length array.
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param max_seq_len The maximum sequence length supported by the kernel.
 * \param seq_len_ptr The current sequence length (number of positions populated in indptr).
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO, typename IdType>
__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  const uint32_t seq_len = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {
    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    state_t<vec_size> st;
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] = -math::inf;
      }
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] = S[indptr[pos] * num_heads + head_idx];
      }
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      if (iter % bdx == 0) {
        s_smem[ty * bdx + tx] =
            iter * bdy + (ty * bdx + tx) < num_index_sets
                ? S[(indptr[pos] + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
                : 0.f;
        __syncthreads();
      }
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();
      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
        float s = s_smem[(iter % bdx) * bdy + ty];
        st.merge(v, s, 1);
      }
      __syncthreads();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    st.normalize();
    threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
    st.normalize();

    st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO, typename IdType>
__global__ void PersistentVariableLengthAttentionSumKernel(DTypeIn* __restrict__ V, IdType* indptr,
                                                           DTypeO* __restrict__ v_sum,
                                                           uint32_t max_seq_len,
                                                           uint32_t* __restrict__ seq_len_ptr,
                                                           uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  const uint32_t seq_len = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;

  vec_t<float, vec_size> v_sum_vec;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {
    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();
      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          v_sum_vec[i] += v[i];
        }
      }
      __syncthreads();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    threadblock_sum<bdx, bdy, vec_size>(v_sum_vec, v_smem);

    v_sum_vec.cast_store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/*!
 * \brief Merge the self-attention state of two index sets A and B.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeO The data type of v_merged.
 * \param v_a The partial v of index set A (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeO>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeO* v_merged,
                       float* s_merged, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
                       cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateKernel<vec_size, DTypeIn, DTypeO>;
    void* args[] = {&v_a, &s_a, &v_b, &s_b, &v_merged, &s_merged, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge the self-attention state with another state in place.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <typename DType>
cudaError_t MergeStateInPlace(DType* v, float* s, DType* v_other, float* s_other, uint32_t seq_len,
                              uint32_t num_heads, uint32_t head_dim, uint8_t* mask = nullptr,
                              cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateInPlaceKernel<vec_size, DType>;
    void* args[] = {&v, &s, &v_other, &s_other, &mask, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge self-attention states of a list of index sets.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param v The partial v of index sets. (n, num_index_sets, h, d)
 * \param s The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_index_sets The number of index sets.
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note s are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeO>
cudaError_t MergeStates(DTypeIn* v, float* s, DTypeO* v_merged, float* s_merged,
                        uint32_t num_index_sets, uint32_t seq_len, uint32_t num_heads,
                        uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    if (num_index_sets >= seq_len) {
      constexpr uint32_t num_threads = 128;
      constexpr uint32_t bdy = num_threads / bdx;
      dim3 nblks(seq_len, num_heads);
      dim3 nthrs(bdx, bdy);
      constexpr uint32_t num_smem_stages = 4;
      auto kernel =
          MergeStatesLargeNumIndexSetsKernel<vec_size, bdx, bdy, num_smem_stages, DTypeIn, DTypeO>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads};
      uint32_t smem_size =
          num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      uint32_t bdy = num_heads;
      dim3 nblks(seq_len);
      dim3 nthrs(bdx, bdy);
      auto kernel = MergeStatesKernel<vec_size, DTypeIn, DTypeO>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads, &head_dim};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    }
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO>
cudaError_t AttentionSum(DTypeIn* v, DTypeO* v_sum, uint32_t num_index_sets, uint32_t seq_len,
                         uint32_t num_heads, uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = AttentionSumKernel<vec_size, DTypeIn, DTypeO>;
    void* args[] = {&v, &v_sum, &num_index_sets, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO, typename IdType>
cudaError_t VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeO* v_merged,
                                      float* s_merged, uint32_t max_seq_len, uint32_t* seq_len,
                                      uint32_t num_heads, uint32_t head_dim, bool enable_pdl,
                                      cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t num_smem_stages = 4;
    uint32_t smem_size =
        num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
    auto kernel = PersistentVariableLengthMergeStatesKernel<vec_size, bdx, bdy, num_smem_stages,
                                                            DTypeIn, DTypeO, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(max_seq_len * num_heads, num_sms));

    dim3 nblks(num_sms * num_blocks_per_sm);
    dim3 nthrs(bdx, bdy);
    void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &max_seq_len, &seq_len, &num_heads};
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // PDL launch
    if (enable_pdl) {
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v, s, indptr, v_merged, s_merged,
                                              max_seq_len, seq_len, num_heads));
    } else {
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO, typename IdType>
cudaError_t VariableLengthAttentionSum(DTypeIn* v, IdType* indptr, DTypeO* v_sum,
                                       uint32_t max_seq_len, uint32_t* seq_len, uint32_t num_heads,
                                       uint32_t head_dim, bool enable_pdl,
                                       cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t num_smem_stages = 4;
    uint32_t smem_size = num_smem_stages * bdy * head_dim * sizeof(DTypeIn);
    auto kernel = PersistentVariableLengthAttentionSumKernel<vec_size, bdx, bdy, num_smem_stages,
                                                             DTypeIn, DTypeO, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(max_seq_len * num_heads, num_sms));

    dim3 nblks(num_sms * num_blocks_per_sm);
    dim3 nthrs(bdx, bdy);
    void* args[] = {&v, &indptr, &v_sum, &max_seq_len, &seq_len, &num_heads};
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    if (enable_pdl) {
      // PDL launch
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
      FLASHINFER_CUDA_CALL(
          cudaLaunchKernelEx(&config, kernel, v, indptr, v_sum, max_seq_len, seq_len, num_heads));
    } else {
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_CASCADE_CUH_
""",
    'key-files/include/flashinfer/attention/decode.cuh': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state(AttentionVariant variant, state_t<vec_size>& st,
                                           float* smem, float* smem_md, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void SingleDecodeWithKVCacheKernel(const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_qo_heads = params.num_qo_heads;

  extern __shared__ uint8_t smem[];
  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }

    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h, freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, /*batch_idx=*/0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
        kv_head_idx, s, st_local, tx, ty, tz);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    block.sync();

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local, reinterpret_cast<float*>(smem), smem_md,
                                      tx, ty, tz);
  if constexpr (variant.use_softmax) {
    st_local.normalize();
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
  if constexpr (variant.use_softmax) {
    st.normalize();
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param pos_encoding_mode The positional encoding mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t seq_len = params.kv_len;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeKV)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DTypeKV) +
          2U * bdy * bdz * sizeof(float);
      auto kernel =
          SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                        vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      if (seq_len <= 256 || tmp == nullptr) {
        // no need to use partition-kv kernel
        dim3 nblks = dim3(1, num_kv_heads);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        params.kv_chunk_size = seq_len;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // use partition-kv kernel
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        int dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        dim3 nblks = dim3(num_chunks, num_kv_heads);
        if (nblks.x == 0 || nblks.y == 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        dim3 nthrs = dim3(bdx, bdy, bdz);
        float* tmp_lse = (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM);
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp;
        params.lse = tmp_lse;
        params.kv_chunk_size = kv_chunk_size;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(
              MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      auto kernel =
          BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                            vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 nblks(padded_batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);

      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk_and_update_local_stat_mla(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* ckv_smem,
    const vec_t<float, vec_size_ckv>& q_nope_vec, const T* kpe_smem,
    const vec_t<float, vec_size_kpe>& q_pe_vec, const vec_t<float, vec_size_kpe>& freq,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, state_t<vec_size_ckv>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  float s[tile_size];
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> ckv_vec;
    ckv_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);

    vec_t<float, vec_size_kpe> kpe_vec;
    kpe_vec.cast_load(kpe_smem + j * head_dim_kpe + tx * vec_size_kpe);

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      s[j] += q_nope_vec[i] * ckv_vec[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size_kpe; ++i) {
      s[j] += q_pe_vec[i] * kpe_vec[i];
    }
    s[j] *= params.sm_scale;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size_ckv; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> v_vec;
    v_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q_nope = params.q_nope;
  const DTypeQ* q_pe = params.q_pe;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const IdType* q_rope_offset = params.q_rope_offset;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const float rope_rcp_scale = params.rope_rcp_scale;
  const float rope_rcp_theta = params.rope_rcp_theta;
  const bool partition_kv = params.partition_kv;
  params.sm_scale *= math::log2e;

  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const uint32_t t_offset = dim3_offset(bdy, bdx, tz, ty, tx);

  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[batch_idx]) return;
  const uint32_t mapped_batch_idx = params.request_indices[batch_idx];

  const uint32_t orig_seq_len = paged_kv.get_length(mapped_batch_idx);
  int32_t q_rope_offset_val =
      q_rope_offset == nullptr ? (orig_seq_len - 1) : q_rope_offset[mapped_batch_idx];

  const uint32_t kv_chunk_idx_in_orig_mapped_batch = params.kv_tile_indices[batch_idx];
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t cur_chunk_start =
      partition_kv ? kv_chunk_idx_in_orig_mapped_batch * kv_chunk_size : 0;
  const uint32_t cur_chunk_end =
      partition_kv ? min((kv_chunk_idx_in_orig_mapped_batch + 1) * kv_chunk_size, orig_seq_len)
                   : orig_seq_len;
  const uint32_t cur_chunk_len = cur_chunk_end - cur_chunk_start;

  uint32_t packed_page_iter_base =
      paged_kv.indptr[mapped_batch_idx] * paged_kv.page_size + cur_chunk_start;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  constexpr uint32_t kv_iter_len = bdy * bdz;
  constexpr uint32_t compute_qk_tile = bdy;

  extern __attribute__((shared)) uint8_t smem[];
  DTypeKV* ckv_smem = (DTypeKV*)smem;
  DTypeKV* kpe_smem = (DTypeKV*)((uint8_t*)ckv_smem +
                                 num_stages_smem * kv_iter_len * head_dim_ckv * sizeof(DTypeKV));
  size_t* ckv_offset_smem = (size_t*)((uint8_t*)kpe_smem + num_stages_smem * kv_iter_len *
                                                               head_dim_kpe * sizeof(DTypeKV));
  size_t* kpe_offset_smem = (size_t*)((uint8_t*)ckv_offset_smem + bdx * bdy * bdz * sizeof(size_t));
  float* smem_md = (float*)ckv_offset_smem;

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size_ckv> q_nope_vec[tile_size_qo_heads];
  vec_t<float, vec_size_kpe> q_pe_vec[tile_size_qo_heads];
  state_t<vec_size_ckv> st[tile_size_qo_heads];
  uint32_t qo_head_idx[tile_size_qo_heads];

  vec_t<float, vec_size_kpe> freq;

#pragma unroll
  for (uint32_t i = 0; i < vec_size_kpe; ++i) {
    freq[i] = rope_rcp_scale * __powf(rope_rcp_theta, float(2 * ((tx * vec_size_kpe + i) / 2)) /
                                                          float(head_dim_kpe));
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // load q_nope and q_pe tile
#pragma unroll
  for (int i = 0; i < tile_size_qo_heads; ++i) {
    qo_head_idx[i] = dim3_offset(bdy, tile_size_qo_heads, blockIdx.y, threadIdx.y, i);
    if (qo_head_idx[i] < num_qo_heads) {
      q_nope_vec[i].cast_load(q_nope +
                              (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                              tx * vec_size_ckv);
      q_pe_vec[i].cast_load(q_pe +
                            (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_kpe +
                            tx * vec_size_kpe);
    }
  }

  // init paged-cache read offset to be used
  uint32_t q, r;
  paged_kv.page_size.divmod(packed_page_iter_base + t_offset, q, r);
  ckv_offset_smem[t_offset] = paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
  kpe_offset_smem[t_offset] = paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size_ckv * 8;
  constexpr uint32_t tx_fold = vec_size_ckv / vec_size_kpe;
  static_assert(num_stages_smem <= bdx);
  size_t offset_bytes;
  bool is_valid_range;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    is_valid_range = (iter * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;

    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes =
        kpe_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);

    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll
  for (uint32_t iter = 0; iter < ceil_div(cur_chunk_len, kv_iter_len); ++iter) {
    cp_async::wait_group<1 * num_stages_smem - 1>();
    block.sync();
    const int32_t kv_idx_base =
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
        cur_chunk_start + iter * kv_iter_len;
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      compute_qk_and_update_local_stat_mla<vec_size_ckv, vec_size_kpe, bdx, compute_qk_tile>(
          params, variant, mapped_batch_idx,
          ckv_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_ckv, q_nope_vec[i],
          kpe_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_kpe, q_pe_vec[i],
          freq, kv_idx_base,
          /*iter_base*/ iter * kv_iter_len, /*iter_bound*/ cur_chunk_len, st[i]);
    }

    if ((iter + num_stages_smem) % bdx == 0) {
      uint32_t q, r;
      paged_kv.page_size.divmod(
          packed_page_iter_base + (iter + num_stages_smem) * kv_iter_len + t_offset, q, r);
      ckv_offset_smem[t_offset] =
          paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
      kpe_offset_smem[t_offset] =
          paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
    }
    block.sync();

    is_valid_range =
        ((iter + num_stages_smem) * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;
    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes = kpe_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  if (bdz != 1) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads)
        sync_state<vec_size_ckv, bdx, bdy, bdz>(variant, st[i], (float*)smem, smem_md, tx, ty, tz);
    }
  }

  if (tz == 0) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads) {
        st[i].normalize();
        st[i].o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                           tx * vec_size_ckv);

        if (lse != nullptr) {
          lse[batch_idx * num_qo_heads + qo_head_idx[i]] = st[i].get_lse();
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA(Params params, typename Params::DTypeO* tmp_v,
                                                     float* tmp_s, bool enable_pdl,
                                                     cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
  constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
  constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

  constexpr uint32_t bdy = 8;
  constexpr uint32_t tile_size_qo_heads = 2;
  constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  const uint32_t gdy = ceil_div(num_qo_heads, qo_heads_per_block);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nblks(padded_batch_size, gdy);
    dim3 nthrs(bdx, bdy, bdz);

    // PDL launch config
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config;
    if (enable_pdl) {
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
    }

    if (tmp_v == nullptr) {
      // do not use partition-kv kernel
      params.partition_kv = false;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    } else {
      // use partition-kv kernel
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
          tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr, num_qo_heads,
          HEAD_DIM_CKV, enable_pdl, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
""",
    'key-files/include/flashinfer/attention/hopper/prefill_sm90.cuh': r"""/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
#define FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>

#include "../../cutlass_utils.cuh"
#include "../../exception.h"
#include "../mask.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "mainloop_mma.cuh"
#include "sparse_mainloop.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

DEFINE_HAS_MEMBER(maybe_prefix_len_ptr)
DEFINE_HAS_MEMBER(maybe_token_pos_in_items_ptr)
DEFINE_HAS_MEMBER(token_pos_in_items_len)
DEFINE_HAS_MEMBER(maybe_max_item_len_ptr)

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits,
          bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename TileScheduler,
          bool MULTIITEMSCORING = false>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS* cutlass::NumThreadsPerWarp, 1)
    PrefillWithKVCacheKernel(CUTE_GRID_CONSTANT
                             typename CollectiveMainloop::Params const mainloop_params,
                             CUTE_GRID_CONSTANT
                             typename CollectiveEpilogue::Params const epilogue_params,
                             CUTE_GRID_CONSTANT
                             typename TileScheduler::Params const scheduler_params) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

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
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(/*num_threads=*/1);
    shared_storage.barrier_O.init(/*num_threads=*/1);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  __syncthreads();

  uint32_t* maybe_prefix_len_ptr = nullptr;
  if constexpr (has_maybe_prefix_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_prefix_len_ptr = mainloop_params.additional_params.maybe_prefix_len_ptr;
  }
  uint16_t* maybe_token_pos_in_items_ptr = nullptr;
  if constexpr (has_maybe_token_pos_in_items_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_token_pos_in_items_ptr = mainloop_params.additional_params.maybe_token_pos_in_items_ptr;
  }
  uint32_t token_pos_in_items_len = 0;
  if constexpr (has_token_pos_in_items_len_v<decltype(mainloop_params.additional_params)>) {
    token_pos_in_items_len = mainloop_params.additional_params.token_pos_in_items_len;
  }
  uint16_t* maybe_max_item_len_ptr = nullptr;
  if constexpr (has_maybe_max_item_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_max_item_len_ptr = mainloop_params.additional_params.maybe_max_item_len_ptr;
  }

  if (warp_group_idx == 0) {  // Producer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24 : 32>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (!use_tma_load_kv || warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
           work_tile_info.is_valid(scheduler_params);
           work_tile_info = scheduler.template get_next_work</*is_producer=*/true>(
               scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len,
              batch_idx] = block_coord;

        if (q_tile_idx * CTA_Q >= qo_len) {
          continue;
        }
        int num_kv_tiles =
            collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
        if (num_kv_tiles <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        int num_kv_tiles_outside_items_window = 0;
        int num_kv_tiles_prefix = 0;
        if constexpr (MULTIITEMSCORING) {
          auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
          auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
          auto valid_items_window_len =
              std::max(0, (q_tile_idx + 1) * CTA_Q + kv_len - qo_len - max_item_len);
          num_kv_tiles_outside_items_window = cute::ceil_div(valid_items_window_len, CTA_KV);
          num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
        }
        if constexpr (MULTIITEMSCORING) {
          collective_mainloop.load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
              num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
        } else {
          collective_mainloop.load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx);
        }
        ++work_idx;
      }
      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    }
  } else {  // Consumer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240 : 160>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 216 : 144>();
    }

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
    // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

    CollectiveMainloop::WarpScheduler::mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/false>(scheduler_params,
                                                                                  work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      AttentionVariant variant(mainloop_params, block_coord);
      auto attention_updater =
          variant.template GetAttentionUpdater<2 * (2 * CTA_Q / NUM_MMA_THREADS)>();

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS, block_coord);
        continue;
      }

      int swa_begin_kv_tile_idx = 0;
      int swa_end_kv_tile_idx = -1;
      if constexpr (LEFT_SLIDING_WINDOW) {
        swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
        swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
      }

      uint32_t prefix_len = 0;
      uint16_t* token_pos_in_items = nullptr;
      if constexpr (MULTIITEMSCORING) {
        prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        token_pos_in_items = maybe_token_pos_in_items_ptr + batch_idx * token_pos_in_items_len;
      }
      int num_kv_tiles_outside_items_window = 0;
      int num_kv_tiles_prefix = 0;
      if constexpr (MULTIITEMSCORING) {
        auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
        auto valid_items_window_len =
            std::max(0, (q_tile_idx + 1) * CTA_Q + kv_len - qo_len - max_item_len);
        num_kv_tiles_outside_items_window = cute::ceil_div(valid_items_window_len, CTA_KV);
        num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
      }
      mma_f16<Ktraits, /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW, CAUSAL, MULTIITEMSCORING,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params, variant, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
          tOrO, attention_updater, num_kv_tiles, swa_begin_kv_tile_idx, swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS, work_idx, q_tile_idx, shared_storage, qo_len, kv_len,
          qo_head_idx, kv_head_idx, prefix_len, token_pos_in_items,
          num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
      collective_epilogue.store(epilogue_params, tOrO, attention_updater.get_lse(), shared_storage,
                                tiled_mma_pv, threadIdx.x - NUM_COPY_THREADS, block_coord);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename Params>
cudaError_t SinglePrefillWithKVCacheKernelTraitsDispatched(Params& params, cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;

  using CollectiveMainloop =
      CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler = SingleTileScheduler;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM_QK,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM_VO,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<DTypeO*>(params.o_ptr),
          get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM_VO,
                          params.o_stride_n,
                          params.o_stride_h),  // layout_O
          static_cast<float*>(params.lse_ptr),
          get_lse_gmem_layout(params.qo_len, params.num_qo_heads),  // layout_LSE
      });

  int num_tiles_q = cutlass::ceil_div(params.qo_len, KernelTraits::CTA_Q);
  // TODO(Zihao): also support kv-head major
  typename Scheduler::Arguments scheduler_args = {
      num_tiles_q, params.num_qo_heads, params.qo_len, params.kv_len,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads)};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int num_ctas = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(num_ctas);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params, bool MULTIITEMSCORING = false>
cudaError_t BatchPrefillWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                               cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop = SparseCollectiveMainloop<typename Params::AdditionalParams,
                                                      KernelTraits, CAUSAL, MULTIITEMSCORING>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM_QK, params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM_VO, params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.kv_indices, params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_VO,
                          params.o_stride_n,
                          params.o_stride_h),                                       // layout_O
          params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
      });

  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler, MULTIITEMSCORING>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheKernelTraitsDispatched(Params& params,
                                                                cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM_QK,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM_VO,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_VO,
                          params.o_stride_n,
                          params.o_stride_h),                                       // layout_O
          params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
      });

  // NOTE(Zihao): add support for kv head-major later
  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, bool CAUSAL>
constexpr auto getCTATileSize() {
  if constexpr (HEAD_DIM_QK == HEAD_DIM_VO) {
    if constexpr (HEAD_DIM_QK == 64) {
      return std::make_tuple(192, 128);
    } else if constexpr (HEAD_DIM_QK == 128) {
      if constexpr (CAUSAL) {
        return std::make_tuple(128, 128);
      } else {
        return std::make_tuple(128, 192);
      }
    } else {
      return std::make_tuple(128, 64);
    }
  } else {
    // NOTE(Zihao) hack for deepseek prefill
    static_assert(HEAD_DIM_QK == 192 && HEAD_DIM_VO == 128);
    return std::make_tuple(128, 128);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params& params, cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr auto CTA_TILE_SIZE = getCTATileSize<HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL>();
  SinglePrefillWithKVCacheKernelTraitsDispatched<
      AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM_QK, HEAD_DIM_VO,
                            /*CTA_Q_=*/get<0>(CTA_TILE_SIZE),
                            /*CTA_KV_=*/get<1>(CTA_TILE_SIZE),
                            /*NUM_STAGES_=*/2, typename Params::DTypeQ, typename Params::DTypeKV,
                            typename Params::DTypeO, typename Params::IdType, AttentionVariant>,
      LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                    cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr auto CTA_TILE_SIZE = getCTATileSize<HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL>();
  BatchPrefillWithRaggedKVCacheKernelTraitsDispatched<
      AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM_QK, HEAD_DIM_VO,
                            /*CTA_Q_=*/get<0>(CTA_TILE_SIZE),
                            /*CTA_KV_=*/get<1>(CTA_TILE_SIZE),
                            /*NUM_STAGES_=*/2, typename Params::DTypeQ, typename Params::DTypeKV,
                            typename Params::DTypeO, typename Params::IdType, AttentionVariant>,
      LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                   cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr bool MULTIITEMSCORING = MASK_MODE == MaskMode::kMultiItemScoring;
  if constexpr (HEAD_DIM_QK == HEAD_DIM_VO) {
    if constexpr (HEAD_DIM_VO == 64) {
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize later
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/192,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, Params, MULTIITEMSCORING>(
          params, stream);
    } else if constexpr (HEAD_DIM_VO == 128) {
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, Params, MULTIITEMSCORING>(
          params, stream);
    } else {
      // HEAD_DIM == 256;
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 256, need to optimize later
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/32,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, Params, MULTIITEMSCORING>(
          params, stream);
    }
  } else {
    return cudaErrorNotSupported;
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
""",
    'key-files/include/flashinfer/attention/hopper/quantization/prefill_sm90.cuh': r"""/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <driver_types.h>

#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <type_traits>
#include <vector>

#include "../../../cutlass_utils.cuh"
#include "../../../exception.h"
#include "../../mask.cuh"
#include "../sparse_mainloop.cuh"
#include "../tile_scheduler.cuh"
#include "../utils.cuh"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop_load.cuh"
#include "mainloop_mma.cuh"
#include "mainloop_sparse_load.cuh"

namespace flashinfer {

using namespace cute;

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits,
          bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS* cutlass::NumThreadsPerWarp, 1)
    FP8PrefillWithKVCacheKernel(CUTE_GRID_CONSTANT
                                typename CollectiveMainloop::Params const mainloop_params,
                                CUTE_GRID_CONSTANT
                                typename CollectiveEpilogue::Params const epilogue_params,
                                CUTE_GRID_CONSTANT
                                typename TileScheduler::Params const scheduler_params) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  // We always assign one WG as producer
  // For FP8 kernel, all 4 warps collectively process ldmatrix with ldmatrix
  static constexpr int NUM_COPY_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;
  // Pipeline for loading K/V
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Pipeline for transposing V
  using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

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
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(/*num_threads=*/1);
    shared_storage.barrier_O.init(/*num_threads=*/1);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k = [&] {
    if constexpr (use_tma_load_kv) {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    // specialized for shared memory of V transpose
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    if constexpr (use_tma_load_kv) {
      pipeline_params.num_consumers = NUM_COPY_THREADS;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      pipeline_params.consumer_arv_count = NUM_COPY_THREADS;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  // Init pipeline_vt for transpose and consumed by mma
  PipelineParamsVt pipeline_params_vt;
  pipeline_params_vt.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params_vt.consumer_arv_count = NUM_MMA_THREADS;
  MainloopPipelineVt pipeline_vt(shared_storage.pipeline_vt, pipeline_params_vt);

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  __syncthreads();

  if (warp_group_idx == 0) {  // Producer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24 : 32>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    // Here no condition as the entire warp group is used as producer
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read;

    int work_idx = 0;

    TileScheduler scheduler;
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/true>(scheduler_params,
                                                                                 work_tile_info)) {
      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <= 0) {
        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        scheduler.broadcast_next_work(work_tile_info);
        continue;
      }
      collective_mainloop.load<LEFT_SLIDING_WINDOW>(
          mainloop_params, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, smem_pipe_read,
          shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx);
      ++work_idx;
    }
    collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write);

  } else {  // Consumer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240 : 160>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 216 : 144>();
    }

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
    // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

    CollectiveMainloop::WarpScheduler::mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/false>(scheduler_params,
                                                                                  work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_QKD{}));
      clear(tOrO);

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      AttentionVariant variant(mainloop_params, block_coord);
      auto attention_updater =
          variant.template GetAttentionUpdater<2 * (2 * CTA_Q / NUM_MMA_THREADS)>();

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS, block_coord);
        continue;
      }

      int swa_begin_kv_tile_idx = 0;
      int swa_end_kv_tile_idx = -1;
      if constexpr (LEFT_SLIDING_WINDOW) {
        swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
        swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
      }

      mma_fp8<Ktraits, /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW, CAUSAL,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params, variant, pipeline_k, pipeline_vt, smem_pipe_read_k, smem_pipe_read_v,
          tOrO, attention_updater, num_kv_tiles, swa_begin_kv_tile_idx, swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS, work_idx, q_tile_idx, shared_storage, qo_len, kv_len,
          qo_head_idx, kv_head_idx, batch_idx);

      collective_epilogue.store(epilogue_params, tOrO, attention_updater.get_lse(), shared_storage,
                                tiled_mma_pv, threadIdx.x - NUM_COPY_THREADS, block_coord);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename Params>
cudaError_t SingleFP8PrefillWithKVCacheKernelTraitsDispatched(Params& params, cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using TileShape_QKD = typename KernelTraits::TileShape_QKD;

  using CollectiveMainloop =
      FP8CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = FP8CollectiveEpilogue<KernelTraits>;
  using Scheduler = SingleTileScheduler;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<DTypeO*>(params.o_ptr),
          get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM,
                          params.o_stride_n,
                          params.o_stride_h),  // layout_O
          static_cast<float*>(params.lse_ptr),
          get_lse_gmem_layout(params.qo_len, params.num_qo_heads),  // layout_LSE
      });

  int num_tiles_q = cutlass::ceil_div(params.qo_len, KernelTraits::CTA_Q);
  // TODO(Zihao): also support kv-head major
  typename Scheduler::Arguments scheduler_args = {
      num_tiles_q, params.num_qo_heads, params.qo_len, params.kv_len,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads)};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  auto kernel =
      (void*)FP8PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                         LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int num_ctas = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(num_ctas);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params>
cudaError_t BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                                  cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      FP8SparseCollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = FP8CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM, params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM, params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.kv_indices, params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                          params.o_stride_n,
                          params.o_stride_h),                                       // layout_O
          params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
      });

  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)FP8PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                         LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SingleFP8PrefillWithKVCacheDispatched(Params& params, cudaStream_t stream) {
  static_assert(cutlass::sizeof_bits_v<typename Params::DTypeQ> == 8);
  static_assert(cutlass::sizeof_bits_v<typename Params::DTypeKV> == 8);
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM == 64) {
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/192,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/4, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/192,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  } else {
    // HEAD_DIM == 256;
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  }
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchFP8PrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                      cudaStream_t stream) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM == 64) {
    // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize later
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/192,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else {
    // HEAD_DIM == 256;
    // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 256, need to optimize later
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_
""",
    'key-files/include/flashinfer/attention/pod.cuh': r"""#ifndef FLASHINFER_POD_CUH_
#define FLASHINFER_POD_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "decode.cuh"
#include "mask.cuh"
#include "prefill.cuh"
#include "variants.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

enum Operation {
  PREFILL = 0,
  DECODE = 1,
};

template <typename KTraits_P, typename KTraits_D, bool PartitionKV_P, typename PrefillParams,
          typename DecodeParams>
__global__ __launch_bounds__(std::max(
    KTraits_P::NUM_THREADS,
    KTraits_D::NUM_THREADS)) void PODWithKVCacheTensorKernel(const uint32_t xsize,
                                                             const __grid_constant__ PrefillParams
                                                                 prefill_params,
                                                             const __grid_constant__ DecodeParams
                                                                 decode_params,
                                                             int* tbAssign) {
  extern __shared__ uint8_t smem[];
  // PREFILL VARS
  const uint32_t num_kv_heads_p = prefill_params.num_kv_heads;
  const uint32_t num_chunks = prefill_params.partition_kv;
  const uint32_t qo_len = prefill_params.qo_len;

  // DECODE VARS
  const uint32_t padded_bsize = decode_params.padded_batch_size;
  const uint32_t num_kv_heads_d = decode_params.paged_kv.num_heads;

  // THREADBLOCKS
  const uint32_t prefill_blocks = num_kv_heads_p * xsize * (PartitionKV_P ? num_chunks : 1);
  const uint32_t decode_blocks = padded_bsize * num_kv_heads_d;

  int op;
  int linear_bid;
  // SM-aware CTA scheduler
  if (threadIdx.x == 0) {
    // TODO_AK: If num_threads dont match, use virtual sub-CTAs.
    // Requires changing block-level sync in main prefill/decode kernels.
    constexpr int blk_factor_p = 1;
    constexpr int blk_factor_d = 1;

    // SM-aware threadblock scheduler code
    // Find out which SM this threadblock is scheduled on
    int num_SMs;
    // WARNING: nsmid has only been tested on A100/H100, and matches SM count
    // No guarantee this will work on other GPUs
    asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
    asm volatile("mov.u32 %0, %smid;" : "=r"(linear_bid));
    const int prefill_slots = (prefill_blocks + blk_factor_p - 1) / blk_factor_p;
    const int decode_slots = (decode_blocks + blk_factor_d - 1) / blk_factor_d;

    if (prefill_slots <= decode_slots) {
      // Total tags = (decode + prefill) / min(decode, prefill)
      // = 1 + decode / prefill; when prefill < decode
      const int total_tags = decode_slots / prefill_slots + 1;
      // For this SM, what's the next operation we want to run?
      op = (atomicAdd(&tbAssign[linear_bid], 1) % total_tags);
      if (op > 0) {
        op = 1;
      }
    } else {
      // Total tags = (decode + prefill) / min(decode, prefill)
      // = 1 + prefill / decode; when decode < prefill
      const int pref_tags = prefill_slots / decode_slots;

      // For this SM, what's the next operation we want to run?
      op = (atomicAdd(&tbAssign[linear_bid], 1) % (pref_tags + 1));
      if (op < pref_tags) {
        op = 0;
      } else {
        op = 1;
      }
    }

    // Get the next blockId for that operation
    linear_bid = atomicAdd(&tbAssign[num_SMs + op], 1);
    // If the blockId obtained exceeds the max blockIds for that op, switch to the other op
    if (op == 0 && linear_bid >= prefill_slots) {
      linear_bid = atomicAdd(&tbAssign[num_SMs + 1], 1);
      op = !op;
    } else if (op == 1 && linear_bid >= decode_slots) {
      op = !op;
      linear_bid = atomicAdd(&tbAssign[num_SMs + 0], 1);
    }
    // Write the blockId and operation to shared memory
    ((int*)smem)[0] = linear_bid;
    ((int*)smem)[1] = op;
  }
  // Sync to wait for dynamic scheduler to finish
  __syncthreads();
  // Fetch from shared memory the assigned blockId and operation.
  linear_bid = ((int*)smem)[0];
  op = ((int*)smem)[1];
  // Sync to force all threads to wait
  __syncthreads();

  if (op == PREFILL) {
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if (linear_tid >= 32 * KTraits_P::NUM_WARPS_Q * KTraits_P::NUM_WARPS_KV) return;

    const dim3 tid = dim3(linear_tid % 32, (linear_tid / 32) % KTraits_P::NUM_WARPS_Q,
                          (linear_tid / 32) / KTraits_P::NUM_WARPS_Q);
    // dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1, num_kv_heads);
    // dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), num_chunks, num_kv_heads);
    //  BlockID exceeds limit
    if (linear_bid >= prefill_blocks) return;

    const uint32_t bx = linear_bid % xsize;
    auto& smem_storage = reinterpret_cast<typename KTraits_P::SharedStorage&>(smem);
    // Not partition_kv
    if constexpr (!PartitionKV_P) {
      const uint32_t chunk_idx = 0;
      const uint32_t kv_head_idx = linear_bid / xsize;
      SinglePrefillWithKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, bx, chunk_idx,
                                                kv_head_idx, 1, num_kv_heads_p);
    } else {
      const uint32_t chunk_idx = (linear_bid / xsize) % num_chunks;
      const uint32_t kv_head_idx = linear_bid / (xsize * num_chunks);
      SinglePrefillWithKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, bx, chunk_idx,
                                                kv_head_idx, num_chunks, num_kv_heads_p);
    }
  } else /* OP == DECODE */ {
    auto& smem_storage = reinterpret_cast<typename KTraits_D::SharedStorage&>(smem);
    // dim3 nblks_d(padded_batch_size_d, 1, num_kv_heads);
    if (linear_bid >= decode_blocks) return;

    const uint32_t bx = linear_bid % padded_bsize;
    const uint32_t kv_head_idx = linear_bid / padded_bsize;

    // dim3 nthrs_d(32, NUM_WARPS_Q_D, NUM_WARPS_KV_D);
    const uint32_t linear_tid = threadIdx.x;
    // Return if threadId exceeds number of threads for this op
    if (linear_tid >= 32 * KTraits_D::NUM_WARPS_Q * KTraits_D::NUM_WARPS_KV) return;

    const dim3 tid = dim3(linear_tid % 32, (linear_tid / 32) % KTraits_D::NUM_WARPS_Q,
                          (linear_tid / 32) / KTraits_D::NUM_WARPS_Q);

    BatchPrefillWithPagedKVCacheDevice<KTraits_D>(decode_params, smem_storage, tid, bx, kv_head_idx,
                                                  num_kv_heads_d);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE_P, uint32_t CTA_TILE_Q_D,
          MaskMode MASK_MODE_D, typename PrefillAttentionVariant, typename DecodeAttentionVariant,
          typename PrefillParams, typename DecodeParams>
cudaError_t PODWithKVCacheTensorDispatched(PrefillParams prefill_params,
                                           typename PrefillParams::DTypeO* tmp_p,
                                           DecodeParams decode_params,
                                           typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                           bool enable_pdl, cudaStream_t stream) {
  static_assert(std::is_same<typename PrefillParams::DTypeQ, typename DecodeParams::DTypeQ>::value);
  static_assert(
      std::is_same<typename PrefillParams::DTypeKV, typename DecodeParams::DTypeKV>::value);
  static_assert(std::is_same<typename PrefillParams::DTypeO, typename DecodeParams::DTypeO>::value);
  // Ensure heads match
  assert(prefill_params.num_kv_heads == decode_params.paged_kv.num_heads);
  assert(prefill_params.num_qo_heads == decode_params.num_qo_heads);
  // Prefill variable setup
  using DTypeQ_P = typename PrefillParams::DTypeQ;
  using DTypeKV_P = typename PrefillParams::DTypeKV;
  using DTypeO_P = typename PrefillParams::DTypeO;
  const uint32_t num_qo_heads = prefill_params.num_qo_heads;
  const uint32_t num_kv_heads = prefill_params.num_kv_heads;
  const uint32_t qo_len = prefill_params.qo_len;
  const uint32_t kv_len = prefill_params.kv_len;
  if (kv_len < qo_len && MASK_MODE_P == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;

  uint32_t cta_tile_q_p = 0;
  int64_t unpacked_qo_len = qo_len * group_size;
  if (unpacked_qo_len > 64 && HEAD_DIM_VO < 256) {
    cta_tile_q_p = 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (unpacked_qo_len > 16) {
        // avg_packed_qo_len <= 64
        cta_tile_q_p = 64;
      } else {
        // avg_packed_qo_len <= 16
        cta_tile_q_p = 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      cta_tile_q_p = 64;
    }
  }

  // Decode vars setup
  using DTypeQ_D = typename DecodeParams::DTypeQ;
  using DTypeKV_D = typename DecodeParams::DTypeKV;
  using DTypeO_D = typename DecodeParams::DTypeO;
  const uint32_t padded_batch_size_d = decode_params.padded_batch_size;
  constexpr uint32_t NUM_MMA_Q_D = get_num_mma_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_Q_D = get_num_warps_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_KV_D = get_num_warps_kv(CTA_TILE_Q_D);

  if (padded_batch_size_d == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  // constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  // constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum_D =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_D, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_D) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  constexpr uint32_t max_num_mma_kv_reg_d =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_D == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_D);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_mma_kv_smem_d =
      (max_smem_per_threadblock / (16 * HEAD_DIM_QK * sizeof(DTypeQ_D)) -
       NUM_MMA_Q_D * NUM_WARPS_Q_D) /
      (2 * NUM_WARPS_KV_D);

  DISPATCH_CTA_TILE_Q(cta_tile_q_p, CTA_TILE_Q_P, {
    constexpr uint32_t NUM_WARPS_Q_P = get_num_warps_q(CTA_TILE_Q_P);
    constexpr uint32_t NUM_WARPS_KV_P = get_num_warps_kv(CTA_TILE_Q_P);
    constexpr uint32_t NUM_MMA_Q_P = get_num_mma_q(CTA_TILE_Q_P);

    using DTypeQKAccum_P =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ_P, half>, half,
                                  float>::type;

    // we expect each sm execute two threadblocks
    // TODO(Zihao): fix the following computation
    const int num_ctas_per_sm_p =
        max_smem_per_sm > (16 * HEAD_DIM_QK * sizeof(DTypeQ_P) * 16) ? 2 : 1;
    const int max_smem_per_threadblock_p = max_smem_per_sm / num_ctas_per_sm_p;

    constexpr uint32_t max_num_mma_kv_reg_p =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q_P == 2 &&
         POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q_P);
    // TODO(Zihao): fix the following computation
    const uint32_t max_num_mma_kv_smem_p =
        (max_smem_per_threadblock_p / (16 * HEAD_DIM_QK * sizeof(DTypeQ_P)) -
         NUM_MMA_Q_P * NUM_WARPS_Q_P) /
        (2 * NUM_WARPS_KV_P);

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_p, max_num_mma_kv_reg_p), NUM_MMA_KV_P, {
      using KTraits_P =
          KernelTraits<MASK_MODE_P, CTA_TILE_Q_P, NUM_MMA_Q_P, NUM_MMA_KV_P, NUM_MMA_D_QK,
                       NUM_MMA_D_VO, NUM_WARPS_Q_P, NUM_WARPS_KV_P, POS_ENCODING_MODE, DTypeQ_P,
                       DTypeKV_P, DTypeO_P, DTypeQKAccum_P, typename PrefillParams::IdType,
                       PrefillAttentionVariant>;

      if constexpr (KTraits_P::IsInvalid()) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q_P
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV_P << " NUM_WARPS_Q=" << NUM_WARPS_Q_P
                << " NUM_WARPS_KV=" << NUM_WARPS_KV_P
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        // Decode stuff
        // TODO: Is there a way to avoid this nested dispatch?
        DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_d, max_num_mma_kv_reg_d), NUM_MMA_KV_D, {
          using KTraits_D =
              KernelTraits<MASK_MODE_D, CTA_TILE_Q_D, NUM_MMA_Q_D, NUM_MMA_KV_D, NUM_MMA_D_QK,
                           NUM_MMA_D_VO, NUM_WARPS_Q_D, NUM_WARPS_KV_D, POS_ENCODING_MODE, DTypeQ_D,
                           DTypeKV_D, DTypeO_D, DTypeQKAccum_D, typename DecodeParams::IdType,
                           DecodeAttentionVariant>;
          if constexpr (KTraits_D::IsInvalid()) {
            // Invalid configuration, skip
            std::ostringstream err_msg;
            err_msg
                << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q_D
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV_D << " NUM_WARPS_Q=" << NUM_WARPS_Q_D
                << " NUM_WARPS_KV=" << NUM_WARPS_KV_D
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
            FLASHINFER_ERROR(err_msg.str());
          } else {
            // End decode stuff
            constexpr uint32_t num_threads_p = (NUM_WARPS_Q_P * NUM_WARPS_KV_P) * WARP_SIZE;
            size_t smem_size_p = sizeof(typename KTraits_P::SharedStorage);
            size_t smem_size_d = sizeof(typename KTraits_D::SharedStorage);

            auto kernel =
                PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, true, PrefillParams, DecodeParams>;
            // Prefill: decide num_splits for split-kv
            int num_blocks_per_sm = 0;
            int num_sm = 0;
            FLASHINFER_CUDA_CALL(
                cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
            // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            //     &num_blocks_per_sm, kernel, num_threads_p, smem_size_p));
            //  Above function returns 0 for some reason, so we use a workaround
            num_blocks_per_sm = std::max(
                1, std::min((int)(max_smem_per_sm / smem_size_p), (int)(256 / num_threads_p)));
            uint32_t max_num_kv_chunks =
                (num_blocks_per_sm * num_sm) /
                (num_kv_heads * ceil_div(qo_len * group_size, KTraits_P::CTA_TILE_Q));
            uint32_t num_chunks;
            if (max_num_kv_chunks > 0) {
              uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
              num_chunks = ceil_div(kv_len, chunk_size);
            } else {
              num_chunks = 0;
            }

            // Setup new prefill params if (not) split
            auto o_p = prefill_params.o;
            auto lse_p = prefill_params.lse;
            float* tmp_lse = (float*)(tmp_p + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
            if (num_chunks <= 1 || tmp_p == nullptr) {
              // Enough parallelism, do not split-kv
              prefill_params.partition_kv = 0;
              kernel = PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, false, PrefillParams,
                                                  DecodeParams>;
            } else {
              // Use cooperative groups to increase occupancy
              prefill_params.partition_kv = num_chunks;
              prefill_params.o = tmp_p;
              prefill_params.lse = tmp_lse;
              kernel = PODWithKVCacheTensorKernel<KTraits_P, KTraits_D, true, PrefillParams,
                                                  DecodeParams>;
            }

            // Setup new decode params if (not) split
            auto o_d = decode_params.o;
            auto lse_d = decode_params.lse;
            if (tmp_v == nullptr) {
              // do not partition kv
              decode_params.partition_kv = false;
            } else {
              decode_params.partition_kv = true;
              decode_params.o = tmp_v;
              decode_params.lse = tmp_s;
            }
            uint32_t xsize = ceil_div(qo_len * group_size, KTraits_P::CTA_TILE_Q);
            int nblks_p(xsize * (prefill_params.partition_kv ? prefill_params.partition_kv : 1) *
                        num_kv_heads);
            int nthrs_p(32 * NUM_WARPS_Q_P * NUM_WARPS_KV_P);

            int nblks_d(padded_batch_size_d * 1 * num_kv_heads);
            int nthrs_d(32 * NUM_WARPS_Q_D * NUM_WARPS_KV_D);

            // ******* Select final combined sizes here ******* /
            size_t smem_size = max(smem_size_p, smem_size_d);
            int nblks = nblks_p + nblks_d;
            int nthrs = max(nthrs_p, nthrs_d);

            // printf("Smem: prefill %zu, decode %zu, total %zu\n", smem_size_p, smem_size_d,
            // smem_size); printf("Blocks: prefill %d, decode %d, total %d\n", nblks_p, nblks_d,
            // nblks); printf("Threads: prefill %d, decode %d, total %d\n", nthrs_p, nthrs_d,
            // nthrs);
            //  ************************************************ /

            static int* tbAssign = nullptr;
            if (tbAssign == nullptr) cudaMalloc(&tbAssign, sizeof(int) * (num_sm + 2));
            cudaMemset(tbAssign, 0, sizeof(int) * (num_sm + 2));

            // Setup kernel arguments
            void* args[] = {(void*)&xsize, (void*)&prefill_params, (void*)&decode_params,
                            (void*)&tbAssign};
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

            // Launch kernel
            if (enable_pdl) {
              cudaLaunchAttribute attribute[1];
              cudaLaunchConfig_t config;
              attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
              attribute[0].val.programmaticStreamSerializationAllowed = 1;
              config.attrs = attribute;
              config.numAttrs = 1;
              config.gridDim = nblks;
              config.blockDim = nthrs;
              config.dynamicSmemBytes = smem_size;
              config.stream = stream;
              FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, xsize, prefill_params,
                                                      decode_params, tbAssign));
            } else {
              FLASHINFER_CUDA_CALL(
                  cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
            }
            FLASHINFER_CUDA_CALL(
                cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

            // Post-kernel stuff for split-kv prefill
            if (!(num_chunks <= 1 || tmp_p == nullptr)) {
              if constexpr (PrefillAttentionVariant::use_softmax) {
                FLASHINFER_CUDA_CALL(MergeStates(tmp_p, tmp_lse, o_p, lse_p, num_chunks, qo_len,
                                                 num_qo_heads, HEAD_DIM_VO, stream));
              } else {
                FLASHINFER_CUDA_CALL(AttentionSum(tmp_p, o_p, num_chunks, qo_len, num_qo_heads,
                                                  HEAD_DIM_VO, stream));
              }
            }
            // Post-kernel stuff for split-kv decode
            if (tmp_v != nullptr) {
              if constexpr (DecodeAttentionVariant::use_softmax) {
                FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                    tmp_v, tmp_s, decode_params.merge_indptr, o_d, lse_d,
                    decode_params.max_total_num_rows, decode_params.total_num_rows, num_qo_heads,
                    HEAD_DIM_VO, stream));
              } else {
                FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
                    tmp_v, decode_params.merge_indptr, o_d, decode_params.max_total_num_rows,
                    decode_params.total_num_rows, num_qo_heads, HEAD_DIM_VO, stream));
              }
            }
          }
        });
      }
    });
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
""",
    'key-files/include/flashinfer/attention/prefill.cuh': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#ifdef FP16_QK_REDUCTION_SUPPORTED
#include "../fp16.h"
#endif
#include "../frag_layout_swizzle.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"
namespace flashinfer {

DEFINE_HAS_MEMBER(maybe_q_rope_offset)
DEFINE_HAS_MEMBER(maybe_k_rope_offset)
DEFINE_HAS_MEMBER(maybe_prefix_len_ptr)
DEFINE_HAS_MEMBER(maybe_token_pos_in_items_ptr)
DEFINE_HAS_MEMBER(token_pos_in_items_len)
DEFINE_HAS_MEMBER(maybe_max_item_len_ptr)

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t WARP_SIZE = 32;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}

template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];
      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
    };
    struct {  // NOTE(Zihao): synchronize attention states across warps
      alignas(
          16) std::conditional_t<NUM_WARPS_KV == 1, float[1],
                                 float[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO]> cta_sync_o_smem;
      alignas(16) std::conditional_t<NUM_WARPS_KV == 1, float2[1],
                                     float2[NUM_WARPS_KV * CTA_TILE_Q]> cta_sync_md_smem;
    };
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];
  };
};

template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_, uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_, uint32_t NUM_WARPS_Q_,
          uint32_t NUM_WARPS_KV_, PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_>
struct KernelTraits {
  static constexpr MaskMode MASK_MODE = MASK_MODE_;
  static constexpr uint32_t NUM_MMA_Q = NUM_MMA_Q_;
  static constexpr uint32_t NUM_MMA_KV = NUM_MMA_KV_;
  static constexpr uint32_t NUM_MMA_D_QK = NUM_MMA_D_QK_;
  static constexpr uint32_t NUM_MMA_D_VO = NUM_MMA_D_VO_;
  static constexpr uint32_t NUM_WARPS_Q = NUM_WARPS_Q_;
  static constexpr uint32_t NUM_WARPS_KV = NUM_WARPS_KV_;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  static constexpr uint32_t HEAD_DIM_QK = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO = NUM_MMA_D_VO * 16;
  static constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM_QK / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_K = HEAD_DIM_QK / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_V = HEAD_DIM_VO / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_O = HEAD_DIM_VO / upcast_size<DTypeO_>();
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16;

  static constexpr SwizzleMode SWIZZLE_MODE_Q = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KV =
      (sizeof(DTypeKV_) == 1 && HEAD_DIM_VO == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  static constexpr uint32_t KV_THR_LAYOUT_ROW = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 8;
  static constexpr uint32_t KV_THR_LAYOUT_COL = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 4;
  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using DTypeQKAccum = DTypeQKAccum_;
  using IdType = IdType_;
  using AttentionVariant = AttentionVariant_;

  static constexpr bool IsInvalid() {
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && NUM_MMA_D_VO > 4 &&
             NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
            (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
  }

  using SharedStorage = SharedStorageQKVO<NUM_WARPS_KV, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_QK,
                                          HEAD_DIM_VO, DTypeQ, DTypeKV, DTypeO>;
#ifdef FP16_QK_REDUCTION_SUPPORTED
  template <typename DT>
  static constexpr DT getNegInf() {
    if constexpr (std::is_same<DT, __half>::value) {
      return std::bit_cast<half>(fp16_ieee_from_fp32_value(-math::inf));
    } else {
      return static_cast<DTypeQKAccum>(-math::inf);
    }
  }

  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? getNegInf<DTypeQKAccum>() : DTypeQKAccum(0.f);
#else
  static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                "then recompile to support fp16 reduction");
  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? DTypeQKAccum(-math::inf) : DTypeQKAccum(0.f);
#endif
};

namespace {

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q(const uint32_t tid_y = threadIdx.y) {
  if constexpr (KTraits::NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return tid_y;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_kv(const uint32_t tid_z = threadIdx.z) {
  if constexpr (KTraits::NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return tid_z;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx(const uint32_t tid_y = threadIdx.y,
                                                 const uint32_t tid_z = threadIdx.z) {
  return get_warp_idx_kv<KTraits>(tid_z) * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid_y);
}

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_rope_offset) {
  float pos[2] = {static_cast<float>(q_rope_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_rope_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <bool produce_v, SharedMemFillMode fill_mode, typename KTraits>
__device__ __forceinline__ void produce_kv(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                           uint32_t* smem_offset, typename KTraits::DTypeKV** gptr,
                                           const uint32_t stride_n, const uint32_t kv_idx_base,
                                           const uint32_t kv_len, const dim3 tid = threadIdx) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), lane_idx = tid.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        *gptr += 8 * upcast_size<DTypeKV>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DTypeKV) * NUM_MMA_D;
      *gptr += NUM_WARPS * 4 * stride_n - sizeof(DTypeKV) * NUM_MMA_D * upcast_size<DTypeKV>();
    }
    *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset);
      kv_idx += NUM_WARPS * 8;
      *gptr += NUM_WARPS * 8 * stride_n;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  }
}

template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t kv_idx_base, const size_t* thr_local_kv_offset, const uint32_t kv_len,
    const dim3 tid = threadIdx) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), lane_idx = tid.x;
  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset);
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4], const float rope_rcp_scale,
                                               const float rope_rcp_theta,
                                               const uint32_t tid_x = threadIdx.x) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  const uint32_t lane_idx = tid_x;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[mma_d][j] =
          rope_rcp_scale *
          __powf(rope_rcp_theta,
                 float(2 * ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                            (HEAD_DIM / 2))) /
                     float(HEAD_DIM));
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,
                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2]) {
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m[mma_q][j] = typename KTraits::DTypeQKAccum(-math::inf);
        d[mma_q][j] = 1.f;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const dim3 tid = threadIdx) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  const uint32_t lane_idx = tid.x, warp_idx_x = get_warp_idx_q<KTraits>(tid.y);

  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    uint32_t q_smem_offset_w = q_smem->get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr =
            q_ptr_base + q * q_stride_n + r * q_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          // load q fragment from gmem to smem
          q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                              q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
          q_ptr += 8 * upcast_size<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, UPCAST_STRIDE_Q>(q_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_QK;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = tid.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<typename KTraits::DTypeQ>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di],
            q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4,
            group_size);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const typename KTraits::IdType* q_rope_offset,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const uint_fastdiv group_size,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = tid.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<typename KTraits::DTypeQ, typename KTraits::IdType>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di], q_packed_idx_base + mma_q * 16 + lane_idx / 4, group_size,
            q_rope_offset);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    float (*rope_freq)[4], const dim3 tid = threadIdx) {
  using DTypeKV = typename KTraits::DTypeKV;
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = tid.x;
  if constexpr (KTraits::NUM_MMA_D_QK == 4 && KTraits::NUM_WARPS_Q == 4) {
    static_assert(KTraits::NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<KTraits>(tid.y);
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(KTraits::NUM_MMA_KV % 2 == 0,
                  "when NUM_MMA_D_QK == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * UPCAST_STRIDE_K;
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV / 2; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[mma_di], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * UPCAST_STRIDE_K;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + KTraits::NUM_MMA_KV) * 16 * UPCAST_STRIDE_K;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y),
                   warp_idx_z = get_warp_idx_kv<KTraits>(tid.z);
    static_assert(KTraits::NUM_MMA_D_QK % (2 * KTraits::NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*NUM_MMA_KV                | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*NUM_MMA_KV-32*NUM_MMA_KV    | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * KTraits::NUM_MMA_KV * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_MMA_D_QK / (2 * KTraits::NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * KTraits::NUM_WARPS_Q;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[mma_di], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->template advance_offset_by_column<2 * KTraits::NUM_WARPS_Q>(
                k_smem_offset_r_first_half, mma_di);
      }
      *k_smem_offset_r += 16 * UPCAST_STRIDE_K;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8]) {
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, mma_d) -
                       KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
      } else {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(*k_smem_offset_r);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          if (mma_d == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          }
        } else if (std::is_same_v<typename KTraits::DTypeQKAccum, half>) {
          if (mma_d == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                (uint32_t*)s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[mma_q][mma_kv],
                                                      a_frag[mma_q], b_frag);
          }
        }
      }
    }
    if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
      if (mma_d % 2 == 1) {
        *k_smem_offset_r =
            k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d / 2);
      }
      *k_smem_offset_r -= KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d) -
                         KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    }
  }
  *q_smem_offset_r -= KTraits::NUM_MMA_D_QK * 2;
  *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * sizeof(typename KTraits::DTypeKV);
}

template <typename KTraits, typename Params, typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint_fastdiv group_size,
    DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8], const dim3 tid = threadIdx,
    const uint32_t kv_head_idx = blockIdx.z) {
  const uint32_t lane_idx = tid.x;
  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];
  float logits = 0., logitsTransformed = 0.;

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];

#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          logits = std::bit_cast<float>(fp16_ieee_to_fp32_value(s_frag[mma_q][mma_kv][reg_id]));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          logits = s_frag[mma_q][mma_kv][reg_id];
        }
#else
        static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                      "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                      "then recompile to support fp16 reduction");
        logits = s_frag[mma_q][mma_kv][reg_id];
#endif
        logitsTransformed = variant.LogitsTransform(params, logits, batch_idx, q_idx, kv_idx,
                                                    qo_head_idx, kv_head_idx);
#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] =
              std::bit_cast<half>(fp16_ieee_from_fp32_value(logitsTransformed));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
        }
#else
        s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
#endif
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t chunk_end, const uint_fastdiv group_size,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8], const dim3 tid = threadIdx,
    const uint32_t kv_head_idx = blockIdx.z) {
  const uint32_t lane_idx = tid.x;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  uint32_t q[NUM_MMA_Q][2], r[NUM_MMA_Q][2];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask_multi_item_scoring(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t window_left, const uint32_t chunk_end,
    const uint_fastdiv group_size, typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    // new arguments for compact description of mask
    const uint32_t prefix_len, uint16_t* token_pos_in_items, const uint32_t lane_idx = threadIdx.x,
    const uint32_t kv_head_idx = blockIdx.z) {
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  uint32_t q[NUM_MMA_Q][2], r[NUM_MMA_Q][2];

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }
  // prefetching global memory to registers
  uint16_t token_pos_in_items_regs[NUM_MMA_Q][(4 / 2)];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t eff_reg_id = 0; eff_reg_id < (4 / 2); ++eff_reg_id) {
      const uint32_t q_idx = q[mma_q][eff_reg_id];
      // use __ldca to hint compiler to cache in L1 for further reuse by other tiles
      const int idx_in_original_seq = q_idx + kv_len - qo_len;
      if (idx_in_original_seq >= prefix_len & idx_in_original_seq < kv_len) {
        token_pos_in_items_regs[mma_q][eff_reg_id] =
            __ldca(token_pos_in_items + idx_in_original_seq - prefix_len);
      }
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        const uint32_t idx_in_original_seq = q_idx + kv_len - qo_len;
        const bool out_of_boundary = kv_idx > idx_in_original_seq || (kv_idx >= chunk_end) ||
                                     kv_idx + window_left < idx_in_original_seq;
        const bool is_prefix = idx_in_original_seq < prefix_len;
        if (out_of_boundary || is_prefix) {
          s_frag[mma_q][mma_kv][reg_id] =
              out_of_boundary ? (KTraits::MaskFillValue) : s_frag[mma_q][mma_kv][reg_id];
        } else {
          s_frag[mma_q][mma_kv][reg_id] =
              (kv_idx < prefix_len |
               (idx_in_original_seq < kv_idx + token_pos_in_items_regs[mma_q][((reg_id % 4) / 2)]))
                  ? s_frag[mma_q][mma_kv][reg_id]
                  : (KTraits::MaskFillValue);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states(
    typename KTraits::AttentionVariant variant,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],
    float (*d)[2]) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr bool use_softmax = AttentionVariant::use_softmax;

  if constexpr (use_softmax) {
    const float sm_scale = variant.sm_scale_log2;
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_prev = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            float m_local =
                max(max(s_frag[mma_q][mma_kv][j * 2 + 0], s_frag[mma_q][mma_kv][j * 2 + 1]),
                    max(s_frag[mma_q][mma_kv][j * 2 + 4], s_frag[mma_q][mma_kv][j * 2 + 5]));
            m[mma_q][j] = max(m[mma_q][j], m_local);
          }
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x2));
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x1));

          float o_scale = math::ptx_exp2(m_prev * sm_scale - m[mma_q][j] * sm_scale);
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            s_frag[mma_q][mma_kv][j * 2 + 0] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 0] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 1] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 1] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 5] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 5] * sm_scale - m[mma_q][j] * sm_scale);
          }
        }
      }
    } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
      const half2 sm_scale = __float2half2_rn(variant.sm_scale_log2);
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        half m_prev[2];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          m_prev[j] = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            half2 m_local = __hmax2(*(half2*)&s_frag[mma_q][mma_kv][j * 2],
                                    *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]);
            m[mma_q][j] = __hmax(m[mma_q][j], __hmax(m_local.x, m_local.y));
          }
        }
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2));
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1));
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float o_scale = math::ptx_exp2(float(m_prev[j] * sm_scale.x - m[mma_q][j] * sm_scale.x));
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
          half2 m2 = make_half2(m[mma_q][j], m[mma_q][j]);
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            *(half2*)&s_frag[mma_q][mma_kv][j * 2] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2] * sm_scale - m2 * sm_scale);
            *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m2 * sm_scale);
          }
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], float (*d)[2]) {
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];
  if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],
                                                           s_frag[mma_q][mma_kv]);
      }
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);
        } else {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag[mma_q][mma_kv]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
      uint32_t b_frag[4];
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
        } else {
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
        swap(b_frag[1], b_frag[2]);
      } else {
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
      }
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
        }
      }
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
          *v_smem_offset_r =
              v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d / 2);
        }
      } else {
        *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d);
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V>(*v_smem_offset_r) -
        sizeof(typename KTraits::DTypeKV) * KTraits::NUM_MMA_D_VO;
  }
  *v_smem_offset_r -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V;
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d(float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2]) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  if constexpr (AttentionVariant::use_softmax) {
    float d_rcp[KTraits::NUM_MMA_Q][2];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        d_rcp[mma_q][j] = (m[mma_q][j] != typename KTraits::DTypeQKAccum(-math::inf))
                              ? math::ptx_rcp(d[mma_q][j])
                              : 0.f;
      }
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[mma_q][mma_d][reg_id] =
              o_frag[mma_q][mma_d][reg_id] * d_rcp[mma_q][(reg_id % 4) / 2];
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m(typename KTraits::AttentionVariant variant,
                                           typename KTraits::DTypeQKAccum (*m)[2]) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        if (m[mma_q][j] != typename KTraits::DTypeQKAccum(-math::inf)) {
          m[mma_q][j] *= variant.sm_scale_log2;
        }
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along threadIdx.z.
 */
template <typename KTraits>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::SharedStorage* smem_storage,
    typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2], const uint32_t warp_idx,
    const uint32_t lane_idx, const dim3 tid = threadIdx) {
  // only necessary when blockDim.z > 1
  if constexpr (KTraits::NUM_WARPS_KV > 1) {
    float* smem_o = smem_storage->cta_sync_o_smem;
    float2* smem_md = smem_storage->cta_sync_md_smem;
    // o: [num_warps, NUM_MMA_Q, NUM_MMA_D_VO, WARP_SIZE(32), 8]
    // md: [num_warps, NUM_MMA_Q, 16, 2 (m/d)]
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        vec_t<float, 8>::memcpy(
            smem_o + (((warp_idx * KTraits::NUM_MMA_Q + mma_q) * KTraits::NUM_MMA_D_VO + mma_d) *
                          WARP_SIZE +
                      lane_idx) *
                         8,
            o_frag[mma_q][mma_d]);
      }
    }

    if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          smem_md[((warp_idx * KTraits::NUM_MMA_Q + mma_q) * 2 + j) * 8 + lane_idx / 4] =
              make_float2(float(m[mma_q][j]), d[mma_q][j]);
        }
      }

      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        float o_scale[2][KTraits::NUM_WARPS_KV];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_new = -math::inf, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float mi = md.x;
            o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
          }
          m[mma_q][j] = typename KTraits::DTypeQKAccum(m_new);
          d[mma_q][j] = d_new;
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o + ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                    KTraits::NUM_MMA_Q +
                                mma_q) *
                                   KTraits::NUM_MMA_D_VO +
                               mma_d) *
                                  WARP_SIZE +
                              lane_idx) *
                                 8);

#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    } else {
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o + ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                    KTraits::NUM_MMA_Q +
                                mma_q) *
                                   KTraits::NUM_MMA_D_VO +
                               mma_d) *
                                  WARP_SIZE +
                              lane_idx) *
                                 8);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size, const dim3 tid = threadIdx) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y);
  const uint32_t lane_idx = tid.x;

  if constexpr (sizeof(DTypeO) == 4) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / 4 + mma_q * 16 + j * 8, q, r);
        const uint32_t o_idx = q;
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          if (o_idx < qo_upper_bound) {
            *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                       (lane_idx % 4) * 2) =
                *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][j * 2]);
            *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                       8 + (lane_idx % 4) * 2) =
                *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][4 + j * 2]);
          }
        }
      }
    }
  } else {
    if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          uint32_t o_frag_f16[8 / 2];
          vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);

#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
          uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
              (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx % 16,
              mma_d * 2 + lane_idx / 16);
          o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
          uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
              (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
          ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
          ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * UPCAST_STRIDE_O))[lane_idx % 4] =
              o_frag_f16[1];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                       8 * UPCAST_STRIDE_O))[lane_idx % 4] = o_frag_f16[3];
#endif
        }
      }

      uint32_t o_smem_offset_w = o_smem->get_permuted_offset<UPCAST_STRIDE_O>(
          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2 * 2; ++j) {
          uint32_t q, r;
          group_size.divmod(o_packed_idx_base + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
          const uint32_t o_idx = q;
          DTypeO* o_ptr =
              o_ptr_base + q * o_stride_n + r * o_stride_h + (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
          for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_VO / 4; ++mma_do) {
            if (o_idx < qo_upper_bound) {
              o_smem->store_128b(o_smem_offset_w, o_ptr);
            }
            o_ptr += 8 * upcast_size<DTypeO>();
            o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, mma_do);
          }
          o_smem_offset_w =
              o_smem->template advance_offset_by_row<4, UPCAST_STRIDE_O>(o_smem_offset_w) -
              2 * KTraits::NUM_MMA_D_VO;
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
 * \tparam POS_ENCODING_MODE The positional encoding mode.
 * \tparam NUM_MMA_Q The number of fragments in x dimension.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
 * \tparam DTypeO The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param rope_rcp_scale 1/(rope_scale), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param rope_rcp_theta 1/(rope_theta), where rope_theta is the theta
 *   used in RoPE.
 */
template <typename KTraits, typename Params>
__device__ __forceinline__ void SinglePrefillWithKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t chunk_idx = blockIdx.y,
    const uint32_t kv_head_idx = blockIdx.z, const uint32_t num_chunks = gridDim.y,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    DTypeQ* q = params.q;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    DTypeO* o = params.o;
    float* lse = params.lse;
    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;
    const bool partition_kv = params.partition_kv;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t v_stride_h = params.v_stride_h;
    const int32_t maybe_window_left = params.window_left;
    const uint_fastdiv& group_size = params.group_size;

    static_assert(sizeof(DTypeQ) == 2);
    const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
    const uint32_t num_qo_heads = num_kv_heads * group_size;

    const uint32_t max_chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
    const uint32_t chunk_start = partition_kv ? chunk_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((chunk_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    auto block = cg::this_thread_block();
    auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
    AttentionVariant variant(params, /*batch_idx=*/0, smem);
    const uint32_t window_left = variant.window_left;

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];
    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    // cooperative fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base =
        (bx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
    DTypeQ* q_ptr_base = q + (kv_head_idx * group_size) * q_stride_h;
    DTypeO* o_ptr_base = partition_kv
                             ? o + chunk_idx * o_stride_n + (kv_head_idx * group_size) * o_stride_h
                             : o + (kv_head_idx * group_size) * o_stride_h;

    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);
    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_ptr_base, q_stride_n, q_stride_h,
                                group_size, &qo_smem, tid);

    cp_async::commit_group();
    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq, tid);
      block.sync();
    }

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

    const uint32_t num_iterations =
        ceil_div(MASK_MODE == MaskMode::kCausal
                     ? min(chunk_size,
                           sub_if_greater_or_zero(
                               kv_len - qo_len + ((bx + 1) * CTA_TILE_Q) / group_size, chunk_start))
                     : chunk_size,
                 CTA_TILE_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * CTA_TILE_Q / group_size,
                                        qo_len + window_left + chunk_start),
                 CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len + (bx * CTA_TILE_Q) / group_size - qo_len, chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

    DTypeKV* k_ptr =
        k +
        (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * k_stride_n +
        kv_head_idx * k_stride_h + (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
    DTypeKV* v_ptr =
        v +
        (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * v_stride_n +
        kv_head_idx * v_stride_h + (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);
    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                           k_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                            v_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<KTraits>(chunk_start + iter * CTA_TILE_KV, &k_smem,
                                             &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/0, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/0, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
          k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
          v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    // normalize d
    normalize_d<KTraits>(o_frag, m, d);

    // write back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr || partition_kv) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_len) {
                if (partition_kv) {
                  lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[qo_idx * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void SinglePrefillWithKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  SinglePrefillWithKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t qo_len = params.qo_len;
  const uint32_t kv_len = params.kv_len;
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  int64_t packed_qo_len = qo_len * group_size;
  uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO);

  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                  float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    const int num_ctas_per_sm =
        max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                                (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
            ? 2
            : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
        ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      using KTraits =
          KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                       NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                       DTypeQKAccum, typename Params::IdType, AttentionVariant>;
      if constexpr (KTraits::IsInvalid()) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
                << " NUM_WARPS_KV=" << NUM_WARPS_KV
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
        auto kernel = SinglePrefillWithKVCacheKernel<KTraits, Params>;
        size_t smem_size = sizeof(typename KTraits::SharedStorage);
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_num_kv_chunks = (num_blocks_per_sm * num_sm) /
                                     (num_kv_heads * ceil_div(qo_len * group_size, CTA_TILE_Q));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          params.partition_kv = false;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          params.partition_kv = true;
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
          auto o = params.o;
          auto lse = params.lse;
          params.o = tmp;
          params.lse = tmp_lse;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), num_chunks, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                             HEAD_DIM_VO, stream));
          } else {
            FLASHINFER_CUDA_CALL(
                AttentionSum(tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
          }
        }
      }
    })
  });
  return cudaSuccess;
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    DTypeQ* q = params.q;
    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    IdType* q_indptr = params.q_indptr;
    IdType* kv_indptr = params.kv_indptr;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const bool partition_kv = params.partition_kv;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t v_stride_h = params.v_stride_h;
    const int32_t maybe_window_left = params.window_left;
    const uint_fastdiv& group_size = params.group_size;

    static_assert(sizeof(DTypeQ) == 2);
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
    const dim3& tid = threadIdx;

    auto block = cg::this_thread_block();
    const uint32_t bx = blockIdx.x, lane_idx = tid.x,
                   warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), kv_head_idx = blockIdx.z;
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = group_size * num_kv_heads;
    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    extern __shared__ uint8_t smem[];
    auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

    DTypeQ* q_ptr_base =
        q + q_indptr[request_idx] * q_stride_n + kv_head_idx * group_size * q_stride_h;

    DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h
                                      : o + o_indptr[request_idx] * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h;

    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                q_stride_h, group_size, &qo_smem, tid);

    cp_async::commit_group();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      IdType* q_rope_offset = nullptr;

      if constexpr (has_maybe_q_rope_offset_v<Params>) {
        q_rope_offset = params.maybe_q_rope_offset;
      }
      if (!q_rope_offset) {
        q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                             &qo_smem, &q_smem_offset_r, rope_freq, tid);
      } else {
        q_smem_inplace_apply_rotary_with_pos<KTraits>(
            qo_packed_idx_base, q_rope_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, tid);
      }
      block.sync();
    }

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size,
                                   chunk_start))
             : chunk_size),
        CTA_TILE_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                        qo_len + window_left + chunk_start),
                 CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                          chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);

    DTypeKV* k_ptr = k +
                     (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                      lane_idx / KV_THR_LAYOUT_COL) *
                         k_stride_n +
                     kv_head_idx * k_stride_h +
                     (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
    DTypeKV* v_ptr = v +
                     (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                      lane_idx / KV_THR_LAYOUT_COL) *
                         v_stride_n +
                     kv_head_idx * v_stride_h +
                     (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();

    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                           k_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                            v_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        IdType* k_rope_offset = nullptr;
        if constexpr (has_maybe_k_rope_offset_v<Params>) {
          k_rope_offset = params.maybe_k_rope_offset;
        }
        k_smem_inplace_apply_rotary<KTraits>(
            (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
                iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
          k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
          v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    // normalize d
    normalize_d<KTraits>(o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (AttentionVariant::use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_len) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t kv_head_idx = blockIdx.z,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    DTypeQ* q = params.q;
    IdType* q_indptr = params.q_indptr;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
    const bool partition_kv = params.partition_kv;
    const int32_t maybe_window_left = params.window_left;
    const uint_fastdiv& group_size = params.group_size;

    uint32_t* maybe_prefix_len_ptr = nullptr;
    if constexpr (has_maybe_prefix_len_ptr_v<Params>) {
      maybe_prefix_len_ptr = params.maybe_prefix_len_ptr;
    }
    uint16_t* maybe_token_pos_in_items_ptr = nullptr;
    if constexpr (has_maybe_token_pos_in_items_ptr_v<Params>) {
      maybe_token_pos_in_items_ptr = params.maybe_token_pos_in_items_ptr;
    }
    uint32_t token_pos_in_items_len = 0;
    if constexpr (has_token_pos_in_items_len_v<Params>) {
      token_pos_in_items_len = params.token_pos_in_items_len;
    }
    uint16_t* maybe_max_item_len_ptr = nullptr;
    if constexpr (has_maybe_max_item_len_ptr_v<Params>) {
      maybe_max_item_len_ptr = params.maybe_max_item_len_ptr;
    }

    static_assert(sizeof(DTypeQ) == 2);
    auto block = cg::this_thread_block();
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

    const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_qo_heads = num_kv_heads * group_size;

    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

    DTypeQ* q_ptr_base =
        q + q_indptr[request_idx] * q_stride_n + (kv_head_idx * group_size) * q_stride_h;
    DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h
                                      : o + o_indptr[request_idx] * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h;
    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                q_stride_h, group_size, &qo_smem, tid);

    cp_async::commit_group();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      IdType* q_rope_offset = nullptr;
      if constexpr (has_maybe_q_rope_offset_v<Params>) {
        q_rope_offset = params.maybe_q_rope_offset;
      }
      if (q_rope_offset == nullptr) {
        q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                             &qo_smem, &q_smem_offset_r, rope_freq, tid);
      } else {
        q_smem_inplace_apply_rotary_with_pos<KTraits>(
            qo_packed_idx_base, q_rope_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, tid);
      }
      block.sync();
    }

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
    size_t thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_COL / 2 / NUM_WARPS_Q];

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t packed_page_iter_base =
        paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
    for (uint32_t i = 0;
         i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                    lane_idx / KV_THR_LAYOUT_COL +
                                    KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
    }
    page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                    chunk_size, tid);
    cp_async::commit_group();
    page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, 0, thr_local_kv_offset,
                                   chunk_size, tid);
    cp_async::commit_group();

    uint32_t num_iterations_prefix;
    uint32_t num_iterations_mask;
    uint32_t num_iterations = 0;

    if constexpr (MASK_MODE != MaskMode::kMultiItemScoring) {
      num_iterations =
          ceil_div((MASK_MODE == MaskMode::kCausal
                        ? min(chunk_size,
                              sub_if_greater_or_zero(
                                  kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size,
                                  chunk_start))
                        : chunk_size),
                   CTA_TILE_KV);
    } else if constexpr (MASK_MODE == MaskMode::kMultiItemScoring) {
      num_iterations_prefix = ceil_div(
          min(min(chunk_size, sub_if_greater_or_zero(
                                  kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size,
                                  chunk_start)),
              sub_if_greater_or_zero(__ldg(maybe_prefix_len_ptr + request_idx), chunk_start)),
          CTA_TILE_KV);
      num_iterations_mask = max(
          min(chunk_size,
              sub_if_greater_or_zero(
                  sub_if_greater_or_zero(kv_len - qo_len + (qo_tile_idx * CTA_TILE_Q) / group_size,
                                         __ldg(maybe_max_item_len_ptr + request_idx)),
                  chunk_start)) /
              (CTA_TILE_KV),
          num_iterations_prefix);

      num_iterations = max(
          num_iterations_mask,
          ceil_div(
              min(chunk_size, sub_if_greater_or_zero(
                                  kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size,
                                  chunk_start)),
              CTA_TILE_KV));
    }

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                        qo_len + window_left + chunk_start),
                 CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                          chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations;
         iter = (MASK_MODE == MaskMode::kMultiItemScoring)
                    ? ((iter + 1 == num_iterations_prefix) ? num_iterations_mask : (iter + 1))
                    : (iter + 1)) {
      const uint32_t prefetch_skip_step =
          (MASK_MODE == MaskMode::kMultiItemScoring)
              ? ((iter + 1 == num_iterations_prefix) ? (num_iterations_mask - num_iterations_prefix)
                                                     : 0)
              : 0;
      packed_page_iter_base += (1 + prefetch_skip_step) * CTA_TILE_KV;
#pragma unroll
      for (uint32_t i = 0;
           i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                      lane_idx / KV_THR_LAYOUT_COL +
                                      KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                  page_iter, entry_idx);
        thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
            page_iter, kv_head_idx, entry_idx,
            (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
      }
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<KTraits>(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                chunk_start + iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom) {
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag);
      } else {
        if constexpr (MASK_MODE != MaskMode::kMultiItemScoring) {
          if (iter >= mask_iteration || iter < window_iteration) {
            logits_mask<KTraits>(
                params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                chunk_start +
                    (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
                qo_len, kv_len, chunk_end, group_size, s_frag);
          }
        } else if constexpr (MASK_MODE == MaskMode::kMultiItemScoring) {
          if (iter + 1 >= num_iterations_prefix) {
            logits_mask_multi_item_scoring<KTraits>(
                params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                chunk_start +
                    (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
                qo_len, kv_len, window_left, chunk_end, group_size, s_frag,
                __ldg(maybe_prefix_len_ptr + request_idx),
                maybe_token_pos_in_items_ptr + request_idx * token_pos_in_items_len, tid.x,
                kv_head_idx);
          } else {
            if (iter >= mask_iteration || iter < window_iteration) {
              logits_mask<KTraits>(
                  params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                  chunk_start +
                      (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16,
                  qo_len, kv_len, chunk_end, group_size, s_frag);
            }
          }
        }
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      page_produce_kv<false, KTraits>(k_smem, &k_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                      thr_local_kv_offset, chunk_size, tid);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, KTraits>(v_smem, &v_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                                     thr_local_kv_offset, chunk_size, tid);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    // normalize d
    normalize_d<KTraits>(o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write_back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_upper_bound) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithRaggedKVCacheKernel<KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }

      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // partition kv
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
              num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        }
      }
    }
  });
  return cudaSuccess;
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithPagedKVCacheKernel<KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }

      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
              num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        }
      }
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
""",
    'key-files/tvm_binding/batch_decode.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <optional>

#include "batch_decode_config.inc"
#include "tvm_binding_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

IntTuple BatchDecodeWithPagedKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  DecodePlanInfo plan_info;

  CHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto work_estimation_func = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
          cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
              static_cast<char*>(float_workspace_buffer->data) +
                  float_workspace_buffer->byte_offset,
              float_workspace_size_in_bytes,
              static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
              static_cast<char*>(page_locked_int_workspace_buffer->data) +
                  page_locked_int_workspace_buffer->byte_offset,
              int_workspace_size_in_bytes, plan_info,
              static_cast<IdType*>(indptr->data) + indptr->byte_offset / sizeof(IdType), batch_size,
              num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/stream, work_estimation_func);

          CHECK(status == cudaSuccess)
              << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
          return true;
        });
      });

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

void BatchDecodeWithPagedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
    DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset, DLTensor* paged_kv_rope_pos_offset,
    DLTensor* o, DLTensor* lse, int64_t pos_encoding_mode_code, int64_t kv_layout_code,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  DecodePlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  int64_t batch_size = q->shape[0];
  int64_t num_qo_heads = q->shape[1];
  int64_t num_kv_heads, page_size;

  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_kv_cache->shape[2];
    page_size = paged_kv_cache->shape[3];
  } else {
    page_size = paged_kv_cache->shape[2];
    num_kv_heads = paged_kv_cache->shape[3];
  }
  uint32_t head_dim_qk = q->shape[2];
  uint32_t head_dim_vo = paged_kv_cache->shape[4];

  CHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  // get q_scalar_type and kv_scalar_type
  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(paged_kv_cache->dtype);

  // get q_stride_n and q_stride_h
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  const auto q_stride_n = q_strides[0];
  const auto q_stride_h = q_strides[1];

  // get kv_cache_strides
  int64_t kv_cache_strides[4] = {
      paged_kv_cache->strides ? paged_kv_cache->strides[0]
                              : paged_kv_cache->shape[1] * paged_kv_cache->shape[2] *
                                    paged_kv_cache->shape[3] * paged_kv_cache->shape[4],
      paged_kv_cache->strides ? paged_kv_cache->strides[2]
                              : paged_kv_cache->shape[3] * paged_kv_cache->shape[4],    //
      paged_kv_cache->strides ? paged_kv_cache->strides[3] : paged_kv_cache->shape[4],  //
      paged_kv_cache->strides ? paged_kv_cache->strides[4] : 1};
  int64_t v_offset = paged_kv_cache->strides ? paged_kv_cache->strides[1]
                                             : paged_kv_cache->shape[2] * paged_kv_cache->shape[3] *
                                                   paged_kv_cache->shape[4];

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_QK, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV),
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV) + v_offset,
            kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices->data) +
                paged_kv_indices->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_indptr->data) +
                paged_kv_indptr->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_last_page_len->data) +
                paged_kv_last_page_len->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_rope_pos_offset->data) +
                paged_kv_rope_pos_offset->byte_offset / sizeof(IdType));

        Params params;
        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.paged_kv = paged_kv;
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.padded_batch_size = 0;
        params.num_qo_heads = num_qo_heads;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.decode_maybe_q_rope_offset =
            static_cast<IdType*>(q_rope_offset->data) + q_rope_offset->byte_offset / sizeof(IdType);
        params.window_left = window_left;
        params.request_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;
        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;

        cudaError_t status =
            flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM_QK, POS_ENCODING_MODE,
                                                              AttentionVariant>(
                params, tmp_v, tmp_s, enable_pdl, stream);
        CHECK(status == cudaSuccess)
            << "BatchDecodeWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
""",
    'key-files/tvm_binding/batch_decode_jit_tvm_binding.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_decode_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchDecodeWithPagedKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t pos_encoding_mode_code, int64_t window_left, int64_t head_dim_qk, int64_t head_dim_vo,
    DataType q_scalar_type, DataType kv_scalar_type, TVMStreamHandle cuda_stream);

void BatchDecodeWithPagedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
    DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset, DLTensor* paged_kv_rope_pos_offset,
    DLTensor* o, DLTensor* lse, int64_t pos_encoding_mode_code, int64_t kv_layout_code,
    int64_t window_left, bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_decode_with_paged_kv_cache_plan,
                              BatchDecodeWithPagedKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_decode_with_paged_kv_cache_run, BatchDecodeWithPagedKVCacheRun);
""",
    'key-files/tvm_binding/batch_prefill.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <optional>

#include "batch_prefill_config.inc"
#include "tvm_binding_utils.h"

namespace flashinfer {

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

IntTuple BatchPrefillWithKVCachePlan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  PrefillPlanInfo plan_info;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = PrefillPlan<IdType>(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      enable_cuda_graph,
      /*sizeof_dtype_o=*/2, stream);

  CHECK(status == cudaSuccess) << "Failed to plan prefill with error: "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

void BatchPrefillWithRaggedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr, DLTensor* kv_indptr,
    DLTensor* q_rope_offset, DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
    int64_t mask_mode_code, int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  PrefillPlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);

  int64_t num_qo_heads = q->shape[1];
  int64_t head_dim_qk = q->shape[2];
  int64_t num_kv_heads = (kv_layout == QKVLayout::kNHD) ? k->shape[1] : k->shape[0];
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  int64_t k_strides[3] = {k->strides ? k->strides[0] : k->shape[1] * k->shape[2],  //
                          k->strides ? k->strides[1] : k->shape[2],                //
                          k->strides ? k->strides[2] : 1};
  int64_t v_strides[3] = {v->strides ? v->strides[0] : v->shape[1] * v->shape[2],  //
                          v->strides ? v->strides[1] : v->shape[2],                //
                          v->strides ? v->strides[2] : 1};
  uint32_t q_stride_n = q_strides[0], q_stride_h = q_strides[1];
  uint32_t k_stride_n, k_stride_h, v_stride_n, v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k_strides[0];
    k_stride_h = k_strides[1];
    v_stride_n = v_strides[0];
    v_stride_h = v_strides[1];
  } else {
    k_stride_h = k_strides[0];
    k_stride_n = k_strides[1];
    v_stride_h = v_strides[0];
    v_stride_n = v_strides[1];
  }

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);

  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(k->dtype);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.k = static_cast<DTypeKV*>(k->data) + k->byte_offset / sizeof(DTypeKV);
        params.v = static_cast<DTypeKV*>(v->data) + v->byte_offset / sizeof(DTypeKV);
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.q_indptr =
            static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType);
        params.kv_indptr =
            static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType);
        params.num_qo_heads = num_qo_heads;
        params.num_kv_heads = num_kv_heads;
        params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.maybe_k_rope_offset = k_rope_offset != nullptr
                                         ? static_cast<IdType*>(k_rope_offset->data) +
                                               k_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.k_stride_n = k_stride_n;
        params.k_stride_h = k_stride_h;
        params.v_stride_n = v_stride_n;
        params.v_stride_h = v_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        CHECK(status == cudaSuccess)
            << "BatchPrefillWithRaggedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}

void BatchPrefillWithPagedKVCacheRun(DLTensor* float_workspace_buffer,
                                     DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
                                     DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr,
                                     DLTensor* paged_kv_indptr, DLTensor* paged_kv_indices,
                                     DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
                                     DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse,
                                     int64_t mask_mode_code, int64_t pos_encoding_mode_code,
                                     int64_t layout, int64_t window_left, bool enable_pdl,
                                     ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  PrefillPlanInfo plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t batch_size = paged_kv_indptr->shape[0] - 1;
  int64_t num_qo_heads = q->shape[1];
  int64_t num_kv_heads, page_size;
  uint32_t head_dim_qk = q->shape[2];
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_kv_cache->shape[2];
    page_size = paged_kv_cache->shape[3];
  } else {
    page_size = paged_kv_cache->shape[2];
    num_kv_heads = paged_kv_cache->shape[3];
  }

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);
  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(paged_kv_cache->dtype);

  // get q_stride_n and q_stride_h
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  const auto q_stride_n = q_strides[0];
  const auto q_stride_h = q_strides[1];

  // get kv_cache_strides
  int64_t kv_cache_strides[4] = {
      paged_kv_cache->strides ? paged_kv_cache->strides[0]
                              : paged_kv_cache->shape[1] * paged_kv_cache->shape[2] *
                                    paged_kv_cache->shape[3] * paged_kv_cache->shape[4],
      paged_kv_cache->strides ? paged_kv_cache->strides[2]
                              : paged_kv_cache->shape[3] * paged_kv_cache->shape[4],    //
      paged_kv_cache->strides ? paged_kv_cache->strides[3] : paged_kv_cache->shape[4],  //
      paged_kv_cache->strides ? paged_kv_cache->strides[4] : 1};
  int64_t v_offset = paged_kv_cache->strides ? paged_kv_cache->strides[1]
                                             : paged_kv_cache->shape[2] * paged_kv_cache->shape[3] *
                                                   paged_kv_cache->shape[4];

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant,
      RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        paged_kv_t<DTypeKV, IdType> paged_kv(
            num_kv_heads, page_size, HEAD_DIM_VO, batch_size, kv_layout,
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV),
            static_cast<DTypeKV*>(paged_kv_cache->data) +
                paged_kv_cache->byte_offset / sizeof(DTypeKV) + v_offset,
            kv_cache_strides,
            static_cast<IdType*>(paged_kv_indices->data) +
                paged_kv_indices->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_indptr->data) +
                paged_kv_indptr->byte_offset / sizeof(IdType),
            static_cast<IdType*>(paged_kv_last_page_len->data) +
                paged_kv_last_page_len->byte_offset / sizeof(IdType),
            paged_kv_rope_pos_offset != nullptr
                ? static_cast<IdType*>(paged_kv_rope_pos_offset->data) +
                      paged_kv_rope_pos_offset->byte_offset / sizeof(IdType)
                : nullptr);
        params.paged_kv = paged_kv;
        params.q_indptr =
            static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType);
        params.o = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);

        params.lse = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.num_qo_heads = num_qo_heads;
        params.group_size = uint_fastdiv(num_qo_heads / paged_kv.num_heads);
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.window_left = window_left;

        params.request_indices = nullptr;
        params.qo_tile_indices = nullptr;
        params.kv_tile_indices = nullptr;
        params.merge_indptr = nullptr;
        params.o_indptr = nullptr;
        params.kv_chunk_size_ptr = nullptr;
        params.block_valid_mask = nullptr;
        params.total_num_rows = nullptr;
        params.max_total_num_rows = 0;
        params.padded_batch_size = 0;
        params.partition_kv = false;

        ADDITIONAL_PARAMS_SETTER

        DTypeO* tmp_v = nullptr;
        float* tmp_s = nullptr;

        params.request_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.request_indices_offset);
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_chunk_size_ptr_offset);
        if (plan_info.split_kv) {
          params.merge_indptr =
              GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_indptr_offset);
          tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.v_offset);
          tmp_s = GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.s_offset);
          if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                GetPtrFromBaseOffset<bool>(int_buffer_ptr, plan_info.block_valid_mask_offset);
          }
        }
        params.padded_batch_size = plan_info.padded_batch_size;
        params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
          params.total_num_rows =
              GetPtrFromBaseOffset<uint32_t>(int_buffer_ptr, plan_info.total_num_rows_offset);
        }

        cudaError_t status = cudaSuccess;

        DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
          status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
              CTA_TILE_Q, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              /*use_fp16_qk_reduction=*/USE_FP16_QK_REDUCTION, MASK_MODE, AttentionVariant,
              PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);
        });

        CHECK(status == cudaSuccess)
            << "BatchPrefillWithPagedKVCache failed with error " << cudaGetErrorString(status);
        return true;
      });
}
""",
    'key-files/tvm_binding/batch_prefill_jit_tvm_binding.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_prefill_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchPrefillWithKVCachePlan(DLTensor* float_workspace_buffer,
                                     DLTensor* int_workspace_buffer,
                                     DLTensor* page_locked_int_workspace_buffer,
                                     DLTensor* qo_indptr, DLTensor* kv_indptr, IntTuple kv_len_arr,
                                     int64_t total_num_rows, int64_t batch_size,
                                     int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size,
                                     bool enable_cuda_graph, int64_t head_dim_qk,
                                     int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream);

void BatchPrefillWithRaggedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr, DLTensor* kv_indptr,
    DLTensor* q_rope_offset, DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
    int64_t mask_mode_code, int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

void BatchPrefillWithPagedKVCacheRun(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr, DLTensor* paged_kv_indptr,
    DLTensor* paged_kv_indices, DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
    DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse, int64_t mask_mode_code,
    int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_kv_cache_plan, BatchPrefillWithKVCachePlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_ragged_kv_cache_run,
                              BatchPrefillWithRaggedKVCacheRun);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_paged_kv_cache_run,
                              BatchPrefillWithPagedKVCacheRun);
""",
    'key-files/tvm_binding/batch_prefill_sm90.cu': r"""/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <flashinfer/attention/hopper/utils.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <optional>

#include "batch_prefill_sm90_config.inc"
#include "tvm_binding_utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                    cudaStream_t stream);

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                   cudaStream_t stream);

}  // namespace flashinfer

using namespace flashinfer;

IntTuple BatchPrefillWithKVCacheSM90Plan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();
  std::vector<IdType> kv_len_vec{kv_len_arr->data, kv_len_arr->data + kv_len_arr->size};

  flashinfer::PrefillPlanSM90Info plan_info;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

  cudaError_t status = PrefillSM90Plan(
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset,
      float_workspace_size_in_bytes,
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset,
      static_cast<char*>(page_locked_int_workspace_buffer->data) +
          page_locked_int_workspace_buffer->byte_offset,
      int_workspace_size_in_bytes, plan_info,
      static_cast<IdType*>(qo_indptr->data) + qo_indptr->byte_offset / sizeof(IdType),
      static_cast<IdType*>(kv_indptr->data) + kv_indptr->byte_offset / sizeof(IdType),
      kv_len_vec.data(), total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim_qk,
      head_dim_vo, page_size, causal, enable_cuda_graph,
      /*sizeof_dtype_o=*/2, stream);

  CHECK(status == cudaSuccess) << "PrefillSM90Plan failed with error: "
                               << cudaGetErrorString(status);

  std::vector<int64_t> plan_info_vec = plan_info.ToVector();
  return IntTuple{plan_info_vec.begin(), plan_info_vec.end()};
}

void BatchPrefillWithRaggedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr, DLTensor* kv_indptr,
    DLTensor* q_rope_offset, DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
    int64_t mask_mode_code, int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  PrefillPlanSM90Info plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  int64_t head_dim_qk = q->shape[2];
  int64_t head_dim_vo = v->shape[2];

  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(k->dtype);

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);
  bool use_swa = window_left != -1;

  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  int64_t k_strides[3] = {k->strides ? k->strides[0] : k->shape[1] * k->shape[2],  //
                          k->strides ? k->strides[1] : k->shape[2],                //
                          k->strides ? k->strides[2] : 1};
  int64_t v_strides[3] = {v->strides ? v->strides[0] : v->shape[1] * v->shape[2],  //
                          v->strides ? v->strides[1] : v->shape[2],                //
                          v->strides ? v->strides[2] : 1};
  int64_t o_strides[3] = {o->strides ? o->strides[0] : o->shape[1] * o->shape[2],  //
                          o->strides ? o->strides[1] : o->shape[2],                //
                          o->strides ? o->strides[2] : 1};
  uint32_t q_stride_n = q_strides[0], q_stride_h = q_strides[1];
  uint32_t o_stride_n = o_strides[0], o_stride_h = o_strides[1];
  uint32_t k_stride_n, k_stride_h, v_stride_n, v_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = k_strides[0];
    k_stride_h = k_strides[1];
    v_stride_n = v_strides[0];
    v_stride_h = v_strides[1];
  } else {
    k_stride_h = k_strides[0];
    k_stride_n = k_strides[1];
    v_stride_h = v_strides[0];
    v_stride_n = v_strides[1];
  }

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        RaggedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.k_ptr = static_cast<DTypeKV*>(k->data) + k->byte_offset / sizeof(DTypeKV);
        params.v_ptr = static_cast<DTypeKV*>(v->data) + v->byte_offset / sizeof(DTypeKV);
        params.o_ptr = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse_ptr = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;
        params.k_stride_n = k_stride_n;
        params.k_stride_h = k_stride_h;
        params.v_stride_n = v_stride_n;
        params.v_stride_h = v_stride_h;
        params.nnz_qo = q->shape[0];
        params.nnz_kv = k->shape[0];
        params.num_qo_heads = q->shape[1];
        params.num_kv_heads = k->shape[1];
        params.group_size = params.num_qo_heads / params.num_kv_heads;
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.maybe_k_rope_offset = k_rope_offset != nullptr
                                         ? static_cast<IdType*>(k_rope_offset->data) +
                                               k_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);

        ADDITIONAL_PARAMS_SETTER

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithRaggedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, enable_pdl, stream);
          CHECK(status == cudaSuccess) << "BatchPrefillWithRaggedKVCacheSM90Run failed with error: "
                                       << cudaGetErrorString(status);
          return true;
        });
      });
}

void BatchPrefillWithPagedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr, DLTensor* paged_kv_indptr,
    DLTensor* paged_kv_indices, DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
    DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse, int64_t mask_mode_code,
    int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream) {
  PrefillPlanSM90Info plan_info;
  std::vector<int64_t> plan_info_vec_(plan_info_vec->data,
                                      plan_info_vec->data + plan_info_vec->size);
  plan_info.FromVector(plan_info_vec_);

  CHECK(lse->shape[0] == q->shape[0]) << "LSE shape mismatch on dim 0";
  CHECK(lse->shape[1] == q->shape[1]) << "LSE shape mismatch on dim 1";

  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  int64_t num_kv_heads, page_size;
  int64_t head_dim_qk = q->shape[2];
  int64_t head_dim_vo = paged_kv_cache->shape[3];
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_kv_cache->shape[2];
    page_size = paged_kv_cache->shape[3];
  } else {
    page_size = paged_kv_cache->shape[2];
    num_kv_heads = paged_kv_cache->shape[3];
  }

  void* float_buffer_ptr =
      static_cast<char*>(float_workspace_buffer->data) + float_workspace_buffer->byte_offset;
  void* int_buffer_ptr =
      static_cast<char*>(int_workspace_buffer->data) + int_workspace_buffer->byte_offset;

  DataType q_scalar_type(q->dtype);
  DataType kv_scalar_type(paged_kv_cache->dtype);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  const PosEncodingMode pos_encoding_mode = static_cast<PosEncodingMode>(pos_encoding_mode_code);
  bool use_swa = window_left != -1;

  // get q_stride_n and q_stride_h
  int64_t q_strides[3] = {q->strides ? q->strides[0] : q->shape[1] * q->shape[2],  //
                          q->strides ? q->strides[1] : q->shape[2],                //
                          q->strides ? q->strides[2] : 1};
  int64_t o_strides[3] = {o->strides ? o->strides[0] : o->shape[1] * o->shape[2],  //
                          o->strides ? o->strides[1] : o->shape[2],                //
                          o->strides ? o->strides[2] : 1};
  const auto q_stride_n = q_strides[0];
  const auto q_stride_h = q_strides[1];
  const auto o_stride_n = o_strides[0];
  const auto o_stride_h = o_strides[1];

  // get kv_cache_strides
  int64_t kv_cache_strides[4] = {
      paged_kv_cache->strides ? paged_kv_cache->strides[0]
                              : paged_kv_cache->shape[1] * paged_kv_cache->shape[2] *
                                    paged_kv_cache->shape[3] * paged_kv_cache->shape[4],
      paged_kv_cache->strides ? paged_kv_cache->strides[2]
                              : paged_kv_cache->shape[3] * paged_kv_cache->shape[4],    //
      paged_kv_cache->strides ? paged_kv_cache->strides[3] : paged_kv_cache->shape[4],  //
      paged_kv_cache->strides ? paged_kv_cache->strides[4] : 1};
  int64_t v_offset = paged_kv_cache->strides ? paged_kv_cache->strides[1]
                                             : paged_kv_cache->shape[2] * paged_kv_cache->shape[3] *
                                                   paged_kv_cache->shape[4];

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, USE_SLIDING_WINDOW,
      USE_LOGITS_SOFT_CAP, AttentionVariant, RaggedParams, PagedParams, [&] {
        PagedParams params;

        params.q_ptr = static_cast<DTypeQ*>(q->data) + q->byte_offset / sizeof(DTypeQ);
        params.k_ptr = static_cast<DTypeKV*>(paged_kv_cache->data) +
                       paged_kv_cache->byte_offset / sizeof(DTypeKV);
        params.v_ptr = static_cast<DTypeKV*>(paged_kv_cache->data) +
                       paged_kv_cache->byte_offset / sizeof(DTypeKV) + v_offset;
        params.o_ptr = static_cast<DTypeO*>(o->data) + o->byte_offset / sizeof(DTypeO);
        params.lse_ptr = static_cast<float*>(lse->data) + lse->byte_offset / sizeof(float);
        params.q_stride_n = q_stride_n;
        params.q_stride_h = q_stride_h;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;
        if (kv_layout == QKVLayout::kNHD) {
          // (num_pages, page_size, num_heads, head_dim)
          params.k_stride_n = kv_cache_strides[1];
          params.k_stride_h = kv_cache_strides[2];
          params.v_stride_n = kv_cache_strides[1];
          params.v_stride_h = kv_cache_strides[2];
        } else {
          // (num_pages, num_heads, page_size, head_dim)
          params.k_stride_h = kv_cache_strides[1];
          params.k_stride_n = kv_cache_strides[2];
          params.v_stride_h = kv_cache_strides[1];
          params.v_stride_n = kv_cache_strides[2];
        }
        params.nnz_qo = q->shape[0];
        params.num_qo_heads = q->shape[1];
        params.num_kv_heads = num_kv_heads;
        params.group_size = params.num_qo_heads / num_kv_heads;
        params.maybe_q_rope_offset = q_rope_offset != nullptr
                                         ? static_cast<IdType*>(q_rope_offset->data) +
                                               q_rope_offset->byte_offset / sizeof(IdType)
                                         : nullptr;
        params.page_size = page_size;
        params.window_left = window_left;
        params.causal = mask_mode_code == 1;
        params.qo_tile_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_tile_indices_offset);
        params.qo_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.qo_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.qo_len_offset);
        params.kv_lens = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.head_indices =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.head_indices_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.kv_indices = static_cast<IdType*>(paged_kv_indices->data) +
                            paged_kv_indices->byte_offset / sizeof(IdType);

        ADDITIONAL_PARAMS_SETTER

        bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
        DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, [&] {
          cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<
              HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
              AttentionVariant>(params, stream);
          CHECK(status == cudaSuccess) << "BatchPrefillWithPagedKVCacheSM90Run failed with error: "
                                       << cudaGetErrorString(status);
          return true;
        });
      });
}
""",
    'key-files/tvm_binding/batch_prefill_sm90_jit_tvm_binding.cu': r"""/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "batch_prefill_sm90_config.inc"
#include "tvm_binding_utils.h"

IntTuple BatchPrefillWithKVCacheSM90Plan(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer,
    DLTensor* page_locked_int_workspace_buffer, DLTensor* qo_indptr, DLTensor* kv_indptr,
    IntTuple kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, TVMStreamHandle cuda_stream);

void BatchPrefillWithRaggedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* qo_indptr, DLTensor* kv_indptr,
    DLTensor* q_rope_offset, DLTensor* k_rope_offset, DLTensor* o, DLTensor* lse,
    int64_t mask_mode_code, int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

void BatchPrefillWithPagedKVCacheSM90Run(
    DLTensor* float_workspace_buffer, DLTensor* int_workspace_buffer, IntTuple plan_info_vec,
    DLTensor* q, DLTensor* paged_kv_cache, DLTensor* qo_indptr, DLTensor* paged_kv_indptr,
    DLTensor* paged_kv_indices, DLTensor* paged_kv_last_page_len, DLTensor* q_rope_offset,
    DLTensor* paged_kv_rope_pos_offset, DLTensor* o, DLTensor* lse, int64_t mask_mode_code,
    int64_t pos_encoding_mode_code, int64_t layout, int64_t window_left,
    bool enable_pdl ADDITIONAL_FUNC_PARAMS, TVMStreamHandle cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_kv_cache_plan, BatchPrefillWithKVCacheSM90Plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_ragged_kv_cache_run,
                              BatchPrefillWithRaggedKVCacheSM90Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_prefill_with_paged_kv_cache_run,
                              BatchPrefillWithPagedKVCacheSM90Run);
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
