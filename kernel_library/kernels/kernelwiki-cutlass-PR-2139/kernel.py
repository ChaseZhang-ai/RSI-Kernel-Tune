"""KernelWiki artifact PR-2139 (NVIDIA/cutlass)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Blocked scale Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrate a blocked scaled FP8 GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. This example shows all important fusions used by FP8 gemm kernels, i.e., blocked scale factor for
    A, B tensor, the abs_max value of D tensor.
    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 \
        --save_aux=false --save_amax=false \
        --device_scale=false --raster=h --swizzle=2
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

// Includes from examples directory
#include "helper.h"
#include "hopper_fp8_commandline.hpp"
#include "reference/host/gemm_with_blockwise_scaling.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Auxiliary matrix configuration and other fusion types
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;
using         ElementAmax  = float;
using         ElementBias  = float;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_128>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<>;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopWithBlockWiseScaling,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
using ElementAmax       = typename EpilogueOutputOp::ElementAmax;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using StrideAux = StrideD;

constexpr bool IsDFp8 =
    cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool IsAuxFp8 =
    cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideAux stride_aux;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
uint32_t mma_promotion_interval;
cutlass::HostTensor<ElementBlockScale, LayoutA> blockscale_tensor_A;
cutlass::HostTensor<ElementBlockScale, LayoutB> blockscale_tensor_B;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_aux;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_ref_aux;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_aux;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<Element>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

/// Helper to initialize a block of device data (scale_tensors)
  template <typename Element, typename Layout>
  bool initialize_scale_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;

      scope_min = -1;
      scope_max = 1;

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options<RasterOrderOptions> &options) {

  // Find Block Scaling tensor shapes based on problem shape and TileShape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
  auto blockscale_m = cute::get<0>(blockscale_shape);
  auto blockscale_n = cute::get<1>(blockscale_shape);
  auto blockscale_k = cute::get<2>(blockscale_shape);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));
  stride_aux = stride_D;



  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto blockscale_a_coord = cutlass::make_Coord(blockscale_m * options.l, blockscale_k);
  auto blockscale_b_coord = cutlass::make_Coord(blockscale_k, blockscale_n * options.l);

  tensor_A.resize(a_coord);
  blockscale_tensor_A.resize(blockscale_a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_B.resize(blockscale_b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  cutlass::Distribution::Kind dist_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_C = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleA = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleB = cutlass::Distribution::Uniform;

  initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022);
  initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023);
  initialize_tensor(tensor_C.host_view(), dist_C, seed + 2024);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026);

#if 0 // Dump blockscaled tensors
  std::cout << "blockscale_tensor_A: " << blockscale_a_coord << std::endl;
  std::cout << blockscale_tensor_A.host_view() << "\n";
  std::cout << "blockscale_tensor_B: " << blockscale_b_coord << std::endl;
  std::cout << blockscale_tensor_B.host_view() << "\n";
#endif

  // Print block scaling tensors on the host side.
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();

  // Note : This value has to match the KernelSchedule::ScalePromotionInterval
  // Else kernel will fail can_implement() check
  // Deprecation Notice : We plan to remove this params member in an upcoming release
  // Users can safely delete this line from their code, since the default is already 4
  mma_promotion_interval = 4;

  if (options.save_aux) {
    tensor_aux.resize(c_coord);
    tensor_aux.sync_device();
    tensor_ref_aux.resize(c_coord);
  }

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_C.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));
    scale_aux.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);
    cutlass::reference::host::TensorFill(scale_aux.host_view(), options.scale_aux);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_C.sync_device();
    scale_D.sync_device();
    scale_aux.sync_device();
  }

  if (IsDFp8 && options.save_amax) {
    abs_max_D.resize(cutlass::make_Coord(1));
    abs_max_D.sync_device();
    reference_abs_max_D.resize(cutlass::make_Coord(1));
  }

  if (IsAuxFp8 && options.save_aux && options.save_amax) {
    abs_max_aux.resize(cutlass::make_Coord(1));
    abs_max_aux.sync_device();
    reference_abs_max_aux.resize(cutlass::make_Coord(1));
  }
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options<RasterOrderOptions> &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(),
     stride_A,
     tensor_B.device_data(),
     stride_B,
     mma_promotion_interval,
     blockscale_tensor_A.device_data(),
     blockscale_tensor_B.device_data()
     },
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;
  fusion_args.alpha_ptr = scalar_alpha.device_data();
  fusion_args.beta_ptr = scalar_beta.device_data();
  fusion_args.scale_a = options.scale_a;
  fusion_args.scale_b = options.scale_b;
  fusion_args.scale_c = options.scale_c;
  fusion_args.scale_a_ptr = scale_A.device_data();
  fusion_args.scale_b_ptr = scale_B.device_data();
  fusion_args.scale_c_ptr = scale_C.device_data();

  // ignored if tensor types are not fp8
  fusion_args.scale_d = options.scale_d;
  fusion_args.scale_aux = options.scale_aux;
  fusion_args.scale_d_ptr = scale_D.device_data();
  fusion_args.scale_aux_ptr = scale_aux.device_data();

  // leaving/setting these as nullptr disables the fusion at runtime
  fusion_args.bias_ptr = nullptr;

  if (options.save_aux) {
    fusion_args.aux_ptr = tensor_aux.device_data();
    fusion_args.dAux = stride_aux;
    if (options.save_amax) {
      fusion_args.amax_aux_ptr = abs_max_aux.device_data();
    }
  }

  if (options.save_amax) {
    fusion_args.amax_D_ptr = abs_max_D.device_data();
  }

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options<RasterOrderOptions> &options) {
  //
  // Compute reference output
  //

  // Block scaling tensors shapes based CTA Block (TileShape) and GEMM Problem shape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
  auto blockscale_m = cute::get<0>(blockscale_shape);
  auto blockscale_n = cute::get<1>(blockscale_shape);
  auto blockscale_k = cute::get<2>(blockscale_shape);

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.k, options.l),
                                stride_A
                              )
                            );
  auto B = cute::make_tensor(tensor_B.host_data(),
                             cute::make_layout(
                               cute::make_shape(options.n, options.k, options.l),
                               stride_B
                              )
                            );
  auto C = cute::make_tensor(tensor_C.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_C
                              )
                            );
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_D
                              )
                            );
  auto Aux = cute::make_tensor(tensor_ref_aux.host_data(),
                               cute::make_layout(
                                  cute::make_shape(options.m, options.n, options.l),
                                  stride_aux
                                )
                              );

  auto blockscale_A = cute::make_tensor(blockscale_tensor_A.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_m, blockscale_k, options.l),
                                          cute::make_stride(blockscale_k, 1, blockscale_m * blockscale_k)
                                        )
                                      );
  auto blockscale_B = cute::make_tensor(blockscale_tensor_B.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_n, blockscale_k, options.l),
                                          cute::make_stride(blockscale_k, 1, blockscale_n * blockscale_k)
                                        )
                                      );

  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                               decltype(A), decltype(B),
                                               decltype(blockscale_A), decltype(blockscale_B),
                                               TileShape> mainloop_params{
                                               A, B,                         // Operand Tensors
                                               blockscale_A, blockscale_B    // Blockwise scaling Tensors
                                              };

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t, // bias
      decltype(Aux),
      unused_t, // valpha
      unused_t, // vbeta
      ActivationFunctor
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.Aux = Aux;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  epilogue_params.scale_c = options.scale_c;
  epilogue_params.scale_d = options.scale_d;
  epilogue_params.scale_aux = options.scale_aux;
  epilogue_params.abs_max_D = reference_abs_max_D.host_data();
  epilogue_params.abs_max_Aux = reference_abs_max_aux.host_data();

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

  if (false) {
    std::cout << "tensor_ref_D.host_view() {" << std::endl
              << tensor_ref_D.host_view() << std::endl
              << "}"  << std::endl;
    std::cout << "tensor_D.host_view() {" << std::endl
              << tensor_D.host_view() << std::endl
              << "}"  << std::endl;
  }

  if (IsDFp8 && options.save_amax) {
    abs_max_D.sync_host();
    passed &= abs_max_D.at(cutlass::make_Coord(0)) == reference_abs_max_D.at(cutlass::make_Coord(0));
  }

  if (options.save_aux) {
    tensor_aux.sync_host();
    passed &= cutlass::reference::host::TensorEquals(tensor_ref_aux.host_view(), tensor_aux.host_view());
    if (IsAuxFp8 && options.save_amax) {
      abs_max_aux.sync_host();
      passed &= abs_max_aux.at(cutlass::make_Coord(0)) == reference_abs_max_aux.at(cutlass::make_Coord(0));
    }
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options<RasterOrderOptions> &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verify) {
    result.passed = verify(options);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  }

  // if (!result.passed) {
  //  exit(-1);
  // }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    for (int iter = 0; iter < options.warmup + options.iterations; ++iter) {
      if (iter == options.warmup)
        timer.start();
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options<RasterOrderOptions> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Grouped scale Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrate a grouped scaled FP8 GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. This example shows all important fusions used by FP8 gemm kernels, i.e., grouped scale factor along M for
    A, blocked scale factor along K for A tensor, blocked scale factor for B tensor, the abs_max value of D tensor.
    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling  \
        --m=2816 --n=3072 --k=16384 \
        --save_aux=false --save_amax=false \
        --device_scale=false --raster=h --swizzle=2
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

// Includes from examples directory
#include "helper.h"
#include "hopper_fp8_commandline.hpp"
#include "reference/host/gemm_with_groupwise_scaling.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Auxiliary matrix configuration and other fusion types
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;
using         ElementAmax  = float;
using         ElementBias  = float;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using TileShape_  = Shape<_128,_128,_128>;  // This one is just to make the compiler happy with verify()...

// ScaleGranularity{M,N}: number of {rows in A}/{columns in B} that share the same scaling factor
// Given TileShape = Shape<_128,_128,_128>:
//   ScaleGranularityM == 128 and ScaleGranularityN == 128 --> 2Dx2D (the shape of the scaling factor)
//   ScaleGranularityM == 1   and ScaleGranularityN == 128 --> 1Dx2D scaling
//   ScaleGranularityM == 128 and ScaleGranularityN == 1   --> 2Dx1D scaling
//   ScaleGranularityM == 1   and ScaleGranularityN == 1   --> 1Dx1D scaling
template <int ScaleGranularityM_, int ScaleGranularityN_>
struct GroupScaleConfig {
  using ArchTag       = cutlass::arch::Sm90;                          // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;               // Operator class tag
  using TileShape     = Shape<_128,_128,_128>;                        // Threadblock-level tile size
  using ClusterShape  = Shape<_1,_2,_1>;                              // Shape of the threadblocks in a cluster

  static constexpr int ScaleGranularityM = ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_;
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static_assert(size<0>(TileShape{}) == ScaleGranularityM * ScaleMsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert(size<1>(TileShape{}) == ScaleGranularityN * ScaleNsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along N.");

  using KernelSchedule    = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM_, ScaleGranularityN_>;
  using EpilogueSchedule  = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType  = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation   = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;
};

using GroupScale1D1DConfig = GroupScaleConfig<                    1,                     1>;
using GroupScale1D2DConfig = GroupScaleConfig<                    1, size<1>(TileShape_{})>;
using GroupScale2D1DConfig = GroupScaleConfig<size<0>(TileShape_{}),                     1>;
using GroupScale2D2DConfig = GroupScaleConfig<size<0>(TileShape_{}), size<1>(TileShape_{})>;

template <typename ScheduleConfig>
struct GroupScaleGemm {
  using ArchTag           = typename ScheduleConfig::ArchTag;
  using OperatorClass     = typename ScheduleConfig::OperatorClass;
  using TileShape         = typename ScheduleConfig::TileShape;
  using ClusterShape      = typename ScheduleConfig::ClusterShape;
  using KernelSchedule    = typename ScheduleConfig::KernelSchedule;
  using EpilogueSchedule  = typename ScheduleConfig::EpilogueSchedule;
  using EpilogueTileType  = typename ScheduleConfig::EpilogueTileType;
  using FusionOperation   = typename ScheduleConfig::FusionOperation;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

  using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

  using GemmKernelDefault = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloopWithGroupWiseScaling,
      CollectiveEpilogue
  >;

  using GemmKernelStreamK = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloopWithGroupWiseScaling,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler
  >;

  using GemmDefault = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelDefault>;
  using GemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStreamK>;
};

using GroupScale1D1DGemm = GroupScaleGemm<GroupScale1D1DConfig>;
using GroupScale1D2DGemm = GroupScaleGemm<GroupScale1D2DConfig>;
using GroupScale2D1DGemm = GroupScaleGemm<GroupScale2D1DConfig>;
using GroupScale2D2DGemm = GroupScaleGemm<GroupScale2D2DConfig>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename GroupScale1D1DGemm::GemmDefault::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
using ElementAmax       = typename EpilogueOutputOp::ElementAmax;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename GroupScale1D1DGemm::GemmDefault::GemmKernel::StrideA;
using StrideB = typename GroupScale1D1DGemm::GemmDefault::GemmKernel::StrideB;
using StrideC = typename GroupScale1D1DGemm::GemmDefault::GemmKernel::StrideC;
using StrideD = typename GroupScale1D1DGemm::GemmDefault::GemmKernel::StrideD;
using StrideAux = StrideD;

constexpr bool IsDFp8 =
    cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool IsAuxFp8 =
    cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
    cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideAux stride_aux;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
uint32_t mma_promotion_interval;
cutlass::HostTensor<ElementBlockScale, LayoutA> blockscale_tensor_A;
cutlass::HostTensor<ElementBlockScale, LayoutB> blockscale_tensor_B;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_aux;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_ref_aux;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_D;
cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_aux;
cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_aux;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<Element>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

/// Helper to initialize a block of device data (scale_tensors)
  template <typename Element, typename Layout>
  bool initialize_scale_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;

      scope_min = -1;
      scope_max = 1;

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

/// Initialize operands to be used in the GEMM and reference GEMM
template <typename GroupScaleConfig>
void initialize(const Options<RasterOrderOptions> &options) {

  using TileShape = typename GroupScaleConfig::TileShape;
  const int ScaleMsPerTile = GroupScaleConfig::ScaleMsPerTile;
  const int ScaleNsPerTile = GroupScaleConfig::ScaleNsPerTile;

  // Find Group Scaling tensor shapes based on `ScaleGranularityM`, problem shape, and TileShape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
  auto groupscale_m = cute::get<0>(blockscale_shape) * ScaleMsPerTile; // We need to pad along M in scale tensor of A to prevent illegal memory access.
  auto groupscale_n = cute::get<1>(blockscale_shape) * ScaleNsPerTile; // We need to pad along N in scale tensor of A to prevent illegal memory access.
  auto blockscale_k = cute::get<2>(blockscale_shape);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));
  stride_aux = stride_D;

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto groupscale_a_coord = cutlass::make_Coord(groupscale_m * options.l, blockscale_k);
  auto groupscale_b_coord = cutlass::make_Coord(groupscale_n * options.l, blockscale_k);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_A.resize(groupscale_a_coord);
  blockscale_tensor_B.resize(groupscale_b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  cutlass::Distribution::Kind dist_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_C = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleA = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleB = cutlass::Distribution::Uniform;

  initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022);
  initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023);
  initialize_tensor(tensor_C.host_view(), dist_C, seed + 2024);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026);

#if 0 // Dump blockscaled tensors
  std::cout << "blockscale_tensor_A: " << groupscale_a_coord << std::endl;
  std::cout << blockscale_tensor_A.host_view() << "\n";
  std::cout << "blockscale_tensor_B: " << groupscale_b_coord << std::endl;
  std::cout << blockscale_tensor_B.host_view() << "\n";
#endif

  // Print group scaling tensors on the host side.
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();

  // Note : This value has to match the KernelSchedule::ScalePromotionInterval
  // Else kernel will fail can_implement() check
  // Deprecation Notice : We plan to remove this params member in an upcoming release
  // Users can safely delete this line from their code, since the default is already 4
  mma_promotion_interval = 4;

  if (options.save_aux) {
    tensor_aux.resize(c_coord);
    tensor_aux.sync_device();
    tensor_ref_aux.resize(c_coord);
  }

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_C.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));
    scale_aux.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);
    cutlass::reference::host::TensorFill(scale_aux.host_view(), options.scale_aux);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_C.sync_device();
    scale_D.sync_device();
    scale_aux.sync_device();
  }

  if (IsDFp8 && options.save_amax) {
    abs_max_D.resize(cutlass::make_Coord(1));
    initialize_tensor(abs_max_D.host_view(), cutlass::Distribution::AllZeros, 0);
    abs_max_D.sync_device();
    reference_abs_max_D.resize(cutlass::make_Coord(1));
    initialize_tensor(reference_abs_max_D.host_view(), cutlass::Distribution::AllZeros, 0);
  }

  if (IsAuxFp8 && options.save_aux && options.save_amax) {
    abs_max_aux.resize(cutlass::make_Coord(1));
    initialize_tensor(abs_max_aux.host_view(), cutlass::Distribution::AllZeros, 0);
    abs_max_aux.sync_device();
    reference_abs_max_aux.resize(cutlass::make_Coord(1));
    initialize_tensor(reference_abs_max_aux.host_view(), cutlass::Distribution::AllZeros, 0);
  }
}

/// Populates a Gemm::Arguments structure from the given commandline options
template<typename GemmArguments>
GemmArguments args_from_options(const Options<RasterOrderOptions> &options)
{
  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(),
     stride_A,
     tensor_B.device_data(),
     stride_B,
     mma_promotion_interval,
     blockscale_tensor_A.device_data(),
     blockscale_tensor_B.device_data()
     },
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;
  fusion_args.alpha_ptr = scalar_alpha.device_data();
  fusion_args.beta_ptr = scalar_beta.device_data();
  fusion_args.scale_a = options.scale_a;
  fusion_args.scale_b = options.scale_b;
  fusion_args.scale_c = options.scale_c;
  fusion_args.scale_a_ptr = scale_A.device_data();
  fusion_args.scale_b_ptr = scale_B.device_data();
  fusion_args.scale_c_ptr = scale_C.device_data();

  // ignored if tensor types are not fp8
  fusion_args.scale_d = options.scale_d;
  fusion_args.scale_aux = options.scale_aux;
  fusion_args.scale_d_ptr = scale_D.device_data();
  fusion_args.scale_aux_ptr = scale_aux.device_data();

  // leaving/setting these as nullptr disables the fusion at runtime
  fusion_args.bias_ptr = nullptr;

  if (options.save_aux) {
    fusion_args.aux_ptr = tensor_aux.device_data();
    fusion_args.dAux = stride_aux;
    if (options.save_amax) {
      fusion_args.amax_aux_ptr = abs_max_aux.device_data();
    }
  }

  if (options.save_amax) {
    fusion_args.amax_D_ptr = abs_max_D.device_data();
  }

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

/// Don't know why the compiler does not like verify() being templated...
bool verify(const Options<RasterOrderOptions> &options, const int ScaleMsPerTile, const int ScaleNsPerTile) {
  //
  // Compute reference output
  //

  // Group scaling tensors shapes based `ScaleGranularityM`, CTA Block (TileShape) and GEMM Problem shape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape_{})));
  auto blockscale_m = cute::get<0>(blockscale_shape);
  auto blockscale_n = cute::get<1>(blockscale_shape);
  auto blockscale_k = cute::get<2>(blockscale_shape);

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.k, options.l),
                                stride_A
                              )
                            );
  auto B = cute::make_tensor(tensor_B.host_data(),
                             cute::make_layout(
                               cute::make_shape(options.n, options.k, options.l),
                               stride_B
                              )
                            );
  auto C = cute::make_tensor(tensor_C.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_C
                              )
                            );
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_D
                              )
                            );
  auto Aux = cute::make_tensor(tensor_ref_aux.host_data(),
                               cute::make_layout(
                                  cute::make_shape(options.m, options.n, options.l),
                                  stride_aux
                                )
                              );

  auto blockscale_A = cute::make_tensor(blockscale_tensor_A.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_m, ScaleMsPerTile, blockscale_k, options.l),
                                          cute::make_stride(blockscale_k * ScaleMsPerTile, 1, ScaleMsPerTile, blockscale_m * blockscale_k * ScaleMsPerTile)
                                        )
                                      );
  auto blockscale_B = cute::make_tensor(blockscale_tensor_B.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_n, ScaleNsPerTile, blockscale_k, options.l),
                                          cute::make_stride(blockscale_k * ScaleNsPerTile, 1, ScaleNsPerTile, blockscale_n * blockscale_k * ScaleNsPerTile)
                                        )
                                      );

  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                               decltype(A), decltype(B),
                                               decltype(blockscale_A), decltype(blockscale_B),
                                               TileShape_> mainloop_params{
                                               A, B,                         // Operand Tensors
                                               blockscale_A, blockscale_B    // Groupwise scaling Tensors
                                              };

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t, // bias
      decltype(Aux),
      unused_t, // valpha
      unused_t, // vbeta
      ActivationFunctor
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.Aux = Aux;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  epilogue_params.scale_c = options.scale_c;
  epilogue_params.scale_d = options.scale_d;
  epilogue_params.scale_aux = options.scale_aux;
  epilogue_params.abs_max_D = reference_abs_max_D.host_data();
  epilogue_params.abs_max_Aux = reference_abs_max_aux.host_data();

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

#if 0
  std::cout << "tensor_ref_D.host_view() {" << std::endl
            << tensor_ref_D.host_view() << std::endl
            << "}"  << std::endl;
  std::cout << "tensor_D.host_view() {" << std::endl
            << tensor_D.host_view() << std::endl
            << "}"  << std::endl;
#endif

  if (IsDFp8 && options.save_amax) {
    abs_max_D.sync_host();
    passed &= abs_max_D.at(cutlass::make_Coord(0)) == reference_abs_max_D.at(cutlass::make_Coord(0));
  }

  if (options.save_aux) {
    tensor_aux.sync_host();
    passed &= cutlass::reference::host::TensorEquals(tensor_ref_aux.host_view(), tensor_aux.host_view());
    if (IsAuxFp8 && options.save_amax) {
      abs_max_aux.sync_host();
      passed &= abs_max_aux.at(cutlass::make_Coord(0)) == reference_abs_max_aux.at(cutlass::make_Coord(0));
    }
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename GroupScaleConfig, typename Gemm>
int run(Options<RasterOrderOptions> &options)
{
  using TileShape = typename GroupScaleConfig::TileShape;
  const int ScaleGranularityM = GroupScaleConfig::ScaleGranularityM;
  const int ScaleGranularityN = GroupScaleConfig::ScaleGranularityN;
  const int ScaleMsPerTile    = GroupScaleConfig::ScaleMsPerTile;
  const int ScaleNsPerTile    = GroupScaleConfig::ScaleNsPerTile;

  initialize<GroupScaleConfig>(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verify) {
    result.passed = verify(options, ScaleMsPerTile, ScaleNsPerTile);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  }

  if (!result.passed) {
   exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Tile shape (M, N, K): " << size<0>(TileShape{}) << ", " << size<1>(TileShape{}) << ", " << size<2>(TileShape{}) << std::endl;
    std::cout << "  ScaleGranularityM: " << ScaleGranularityM << " (ScaleMsPerTile: " << ScaleMsPerTile << ")" << std::endl;
    std::cout << "  ScaleGranularityN: " << ScaleGranularityN << " (ScaleNsPerTile: " << ScaleNsPerTile << ")" << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
    fflush(stdout);
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options<RasterOrderOptions> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  std::cout << "Basic split-K GEMM kernel" << std::endl;
  run<GroupScale1D1DConfig, GroupScale1D1DGemm::GemmDefault>(options);
  std::cout << std::endl;
  run<GroupScale1D2DConfig, GroupScale1D2DGemm::GemmDefault>(options);
  std::cout << std::endl;
  run<GroupScale2D1DConfig, GroupScale2D1DGemm::GemmDefault>(options);
  std::cout << std::endl;
  run<GroupScale2D2DConfig, GroupScale2D2DGemm::GemmDefault>(options);
  std::cout << std::endl;

  std::cout << std::endl;

  std::cout << "StreamK GEMM kernel" << std::endl;
  run<GroupScale1D1DConfig, GroupScale1D1DGemm::GemmStreamK>(options);
  std::cout << std::endl;
  run<GroupScale1D2DConfig, GroupScale1D2DGemm::GemmStreamK>(options);
  std::cout << std::endl;
  run<GroupScale2D1DConfig, GroupScale2D1DGemm::GemmStreamK>(options);
  std::cout << std::endl;
  run<GroupScale2D2DConfig, GroupScale2D2DGemm::GemmStreamK>(options);
  std::cout << std::endl;
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/hopper_fp8_commandline.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Command line options parsing
template<typename RasterOrderOptions>
struct Options {

  bool help = false;
  bool verify = true;

  float alpha = 1.f, beta = 0.f;
  float scale_a = 1.f, scale_b = 1.f, scale_c = 1.f, scale_d = 1.f, scale_aux = 1.f;
  bool device_scale = false;
  bool save_aux = true;
  bool save_amax = true;
  int iterations = 1000;
  int warmup = 1000;
  int m = 1024, n = 512, k = 1024, l = 1;
  RasterOrderOptions raster;
  int swizzle;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("scale_a", scale_a, 1.f);
    cmd.get_cmd_line_argument("scale_b", scale_b, 1.f);
    cmd.get_cmd_line_argument("scale_c", scale_c, 1.f);
    cmd.get_cmd_line_argument("scale_d", scale_d, 1.f);
    cmd.get_cmd_line_argument("scale_aux", scale_aux, 1.f);
    cmd.get_cmd_line_argument("device_scale", device_scale, false);
    cmd.get_cmd_line_argument("save_aux", save_aux, true);
    cmd.get_cmd_line_argument("save_amax", save_amax, true);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verify);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling\n\n"
      << "  Hopper FP8 GEMM using a Warp Specialized kernel with Blockwise Scaling.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the l extent (batch) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --scale_a=<f32>             Scaling factor for A\n"
      << "  --scale_b=<f32>             Scaling factor for B\n"
      << "  --scale_c=<f32>             Scaling factor for C\n"
      << "  --scale_d=<f32>             Scaling factor for D (ignored for non-fp8 D)\n"
      << "  --scale_aux=<f32>           Scaling factor for the auxiliary tensor (ignored for non-fp8 aux)\n"
      << "  --device_scale=<bool>       Copy scalars to device memory before kernel launch (default: false)\n"
      << "  --save_aux=<bool>           Save the pre-activation as an auxiliary tensor (default: true)\n"
      << "  --save_amax=<bool>          Save the pre-scaled max absolute value of any fp8 outputs (aux and/or D) (default: true)\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};
""",
    'key-files/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Grouped scale Hopper FP8 Grouped GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrates a grouped scaled FP8 Grouped GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. This example also showcases on-the-fly modification of TMA
    descriptors to move between groups/problem_count (represented by groups).
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 --save_aux=false --save_amax=false \
        --raster=h --swizzle=2 --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/tensor_fill.h"

// Includes from examples directory
#include "helper.h"
#include "hopper_fp8_commandline.hpp"
#include "reference/host/gemm_with_groupwise_scaling.h"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using TileShape_  = Shape<_128,_128,_128>;  // This one is just to make the compiler happy with verify()...

// ScaleGranularity{M,N}: number of {rows in A}/{columns in B} that share the same scaling factor
// Given TileShape = Shape<_128,_128,_128>:
//   ScaleGranularityM == 128 and ScaleGranularityN == 128 --> 2Dx2D (the shape of the scaling factor)
//   ScaleGranularityM == 1   and ScaleGranularityN == 128 --> 1Dx2D scaling
//   ScaleGranularityM == 128 and ScaleGranularityN == 1   --> 2Dx1D scaling
//   ScaleGranularityM == 1   and ScaleGranularityN == 1   --> 1Dx1D scaling
template <int ScaleGranularityM_, int ScaleGranularityN_>
struct GroupScaleConfig {
  using ArchTag       = cutlass::arch::Sm90;                          // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;               // Operator class tag
  using TileShape     = Shape<_128,_128,_128>;                        // Threadblock-level tile size
  using ClusterShape  = Shape<_1,_2,_1>;                              // Shape of the threadblocks in a cluster

  static constexpr int ScaleGranularityM = ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_;
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static_assert(size<0>(TileShape{}) == ScaleGranularityM * ScaleMsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert(size<1>(TileShape{}) == ScaleGranularityN * ScaleNsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along N.");

  using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM_, ScaleGranularityN_>;
  using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using EpilogueTileType  = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;
};

using GroupScale1D1DConfig = GroupScaleConfig<                    1,                     1>;
using GroupScale1D2DConfig = GroupScaleConfig<                    1, size<1>(TileShape_{})>;
using GroupScale2D1DConfig = GroupScaleConfig<size<0>(TileShape_{}),                     1>;
using GroupScale2D2DConfig = GroupScaleConfig<size<0>(TileShape_{}), size<1>(TileShape_{})>;

template <typename ScheduleConfig>
struct GroupScaleGemm {
  using ArchTag           = typename ScheduleConfig::ArchTag;
  using OperatorClass     = typename ScheduleConfig::OperatorClass;
  using TileShape         = typename ScheduleConfig::TileShape;
  using ClusterShape      = typename ScheduleConfig::ClusterShape;
  using KernelSchedule    = typename ScheduleConfig::KernelSchedule;
  using EpilogueSchedule  = typename ScheduleConfig::EpilogueSchedule;
  using EpilogueTileType  = typename ScheduleConfig::EpilogueTileType;
  using FusionOperation   = typename ScheduleConfig::FusionOperation;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

  using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloopWithGroupWiseScaling,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using GroupScale1D1DGemm = GroupScaleGemm<GroupScale1D1DConfig>;
using GroupScale1D2DGemm = GroupScaleGemm<GroupScale1D2DConfig>;
using GroupScale2D1DGemm = GroupScaleGemm<GroupScale2D1DConfig>;
using GroupScale2D2DGemm = GroupScaleGemm<GroupScale2D2DConfig>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename GroupScale1D1DGemm::Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename GroupScale1D1DGemm::Gemm::GemmKernel::InternalStrideA;
using StrideB = typename GroupScale1D1DGemm::Gemm::GemmKernel::InternalStrideB;
using StrideC = typename GroupScale1D1DGemm::Gemm::GemmKernel::InternalStrideC;
using StrideD = typename GroupScale1D1DGemm::Gemm::GemmKernel::InternalStrideD;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization

cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_blockscale_A;
std::vector<int64_t> offset_blockscale_B;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementD> block_D;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

cutlass::DeviceAllocation<const ElementA *> ptr_A;
cutlass::DeviceAllocation<const ElementB *> ptr_B;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<ElementD *> ptr_D;
cutlass::DeviceAllocation<ElementD *> ptr_ref_D;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) 

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<Shape<int,int,int>>::RasterOrderOptions;

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class ScopeMin = std::nullopt_t, class ScopeMax = std::nullopt_t>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023,
  ScopeMin scope_min = std::nullopt, ScopeMax scope_max = std::nullopt) {

  double _scope_max, _scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  if (bits_input == 1) {
    _scope_max = 2;
    _scope_min = 0;
  } else if (bits_input <= 8) {
    _scope_max = 2;
    _scope_min = -2;
  } else if (bits_input == 16) {
    _scope_max = 5;
    _scope_min = -5;
  } else {
    _scope_max = 8;
    _scope_min = -8;
  }
  if constexpr (!std::is_same_v<ScopeMax, std::nullopt_t>) {
    _scope_max = scope_max;
  }
  if constexpr (!std::is_same_v<ScopeMin, std::nullopt_t>) {
    _scope_min = scope_min;
  }
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, (Element) _scope_max, (Element) _scope_min, 0);

  return true;
}

/// Allocates device-side data
template <typename OptionType>
void allocate(const OptionType &options) {

  using TileShape = typename OptionType::GroupScaleConfig::TileShape;
  const int ScaleMsPerTile = OptionType::GroupScaleConfig::ScaleMsPerTile;
  const int ScaleNsPerTile = OptionType::GroupScaleConfig::ScaleNsPerTile;

  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;

  offset_A.clear();
  offset_B.clear();
  offset_C.clear();
  offset_D.clear();
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();
  stride_A_host.clear();
  stride_B_host.clear();
  stride_C_host.clear();
  stride_D_host.clear();
  
  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(problem), TileShape{})));
    auto groupscale_m = cute::get<0>(blockscale_shape) * ScaleMsPerTile; // We need to pad along M in scale tensor of A to prevent illegal memory access.
    auto groupscale_n = cute::get<1>(blockscale_shape) * ScaleNsPerTile; // We need to pad along N in scale tensor of A to prevent illegal memory access.
    auto blockscale_k = cute::get<2>(blockscale_shape);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_blockscale_A.push_back(total_elements_blockscale_A);
    offset_blockscale_B.push_back(total_elements_blockscale_B);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_blockscale_A = groupscale_m * blockscale_k;
    int64_t elements_blockscale_B = groupscale_n * blockscale_k;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_blockscale_A += elements_blockscale_A;
    total_elements_blockscale_B += elements_blockscale_B;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
  blockscale_block_A.reset(total_elements_blockscale_A);
  blockscale_block_B.reset(total_elements_blockscale_B);
}

/// Initialize operands to be used in the GEMM and reference GEMM
template <typename OptionType>
void initialize(const OptionType &options) {

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementD *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_A_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_B_host(options.groups);

  alpha_host.clear();
  beta_host.clear();

  for (int i = 0; i < options.groups; i++) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    ptr_blockscale_A_host.at(i) = blockscale_block_A.get() + offset_blockscale_A.at(i);
    ptr_blockscale_B_host.at(i) = blockscale_block_B.get() + offset_blockscale_B.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_blockscale_A.reset(options.groups);
  ptr_blockscale_A.copy_from_host(ptr_blockscale_A_host.data());

  ptr_blockscale_B.reset(options.groups);
  ptr_blockscale_B.copy_from_host(ptr_blockscale_B_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_block(block_A, seed + 2022);
  initialize_block(block_B, seed + 2023);
  initialize_block(block_C, seed + 2024);
  initialize_block(blockscale_block_A, seed + 2025, -1, 1);
  initialize_block(blockscale_block_B, seed + 2026, -1, 1);

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

}

/// Populates a Gemm::Arguments structure from the given commandline options
template<typename GemmArguments, typename OptionType>
GemmArguments args_from_options(const OptionType &options, bool host_problem_shapes_available = true)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GroupScale1D1DGemm::Gemm::GemmKernel>(device_id);

  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), host_problem_shapes_available ? options.problem_sizes_host.data() : (decltype(options.problem_sizes_host.data())) nullptr},
    {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
     ptr_blockscale_A.get(),
     ptr_blockscale_B.get()
    },
    {
      {}, // epilogue.thread
      ptr_C.get(), stride_C.get(),
      ptr_D.get(), stride_D.get()
    },
    kernel_hw_info
  };

  auto &fusion_args = arguments.epilogue.thread;
  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }
  else {
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.beta_ptr_array = beta_device.get();
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
  }

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

template <typename OptionType>
bool verify(const OptionType &options) {

  //
  // Compute reference output
  //

  std::vector<ElementA> block_A_host(block_A.size());
  std::vector<ElementB> block_B_host(block_B.size());
  std::vector<ElementC> block_C_host(block_C.size());
  std::vector<ElementD> block_D_host_kernel(block_D.size());
  std::vector<ElementD> block_D_host_ref(block_D.size());
  std::vector<ElementBlockScale> blockscale_block_A_host(blockscale_block_A.size());
  std::vector<ElementBlockScale> blockscale_block_B_host(blockscale_block_B.size());

  block_A.copy_to_host(block_A_host.data());
  block_B.copy_to_host(block_B_host.data());
  block_C.copy_to_host(block_C_host.data());
  block_D.copy_to_host(block_D_host_kernel.data());
  blockscale_block_A.copy_to_host(blockscale_block_A_host.data());
  blockscale_block_B.copy_to_host(blockscale_block_B_host.data());

  bool passed = true;
  for (int group_idx = 0; group_idx < options.groups; group_idx++) {
    // Group scaling tensors shapes based `ScaleGranularityM`, CTA Block (TileShape) and GEMM Problem shape
    auto [m, n, k] = options.problem_sizes_host.at(group_idx);
    auto gemm_problem_shape = cute::make_shape(m, n, k);
    auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape_{})));
    auto blockscale_m = cute::get<0>(blockscale_shape);
    auto blockscale_n = cute::get<1>(blockscale_shape);
    auto blockscale_k = cute::get<2>(blockscale_shape);
    auto groupscale_m = blockscale_m * OptionType::GroupScaleConfig::ScaleMsPerTile;
    auto groupscale_n = blockscale_n * OptionType::GroupScaleConfig::ScaleNsPerTile;

    // Create instantiation for device reference gemm kernel
    auto A = cute::make_tensor(block_A_host.data() + offset_A.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, k, 1),
                                  stride_A_host.at(group_idx)
                                )
                              );
    auto B = cute::make_tensor(block_B_host.data() + offset_B.at(group_idx),
                              cute::make_layout(
                                cute::make_shape(n, k, 1),
                                stride_B_host.at(group_idx)
                                )
                              );
    auto C = cute::make_tensor(block_C_host.data() + offset_C.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_C_host.at(group_idx)
                                )
                              );
    auto D = cute::make_tensor(block_D_host_ref.data() + offset_D.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_D_host.at(group_idx)
                                )
                              );

    auto blockscale_A = cute::make_tensor(blockscale_block_A_host.data() + offset_blockscale_A.at(group_idx),
                                          cute::make_layout(
                                            cute::make_shape(groupscale_m, blockscale_k, 1),
                                            cute::make_stride(1, groupscale_m, groupscale_m * blockscale_k)
                                          )
                                        );
    auto blockscale_B = cute::make_tensor(blockscale_block_B_host.data() + offset_blockscale_B.at(group_idx),
                                          cute::make_layout(
                                            cute::make_shape(groupscale_n, blockscale_k, 1),
                                            cute::make_stride(1, groupscale_n, groupscale_n * blockscale_k)
                                          )
                                        );

    using unused_t = decltype(D);

    cutlass::reference::host::GettMainloopParams<
      ElementAccumulator,
      decltype(A),
      decltype(B),
      decltype(blockscale_A),
      decltype(blockscale_B),
      TileShape_
    > mainloop_params{
        A, B,                         // Operand Tensors
        blockscale_A, blockscale_B    // Groupwise scaling Tensors
    };

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D),
        unused_t, // bias
        unused_t, // Aux
        unused_t, // valpha
        unused_t, // vbeta
        ActivationFunctor
    > epilogue_params;

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha_host.at(group_idx);
    epilogue_params.beta = beta_host.at(group_idx);

    // get reference result
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    auto this_group_passed = std::equal(
      // std::execution::par_unseq,
      block_D_host_ref.data() + offset_D.at(group_idx),
      block_D_host_ref.data() + offset_D.at(group_idx) + m * n,
      block_D_host_kernel.data() + offset_D.at(group_idx)
    );
    
    passed &= this_group_passed;

#if 0
    std::cout << "Group: " << group_idx << " M: " << m << " N: " << n << " K: " << k << " Status: " << this_group_passed << std::endl;
#endif

  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm, typename OptionType>
int run(OptionType &options, bool host_problem_shapes_available = true)
{
  using TileShape = typename OptionType::GroupScaleConfig::TileShape;
  const int ScaleGranularityM = OptionType::GroupScaleConfig::ScaleGranularityM;
  const int ScaleGranularityN = OptionType::GroupScaleConfig::ScaleGranularityN;
  const int ScaleMsPerTile    = OptionType::GroupScaleConfig::ScaleMsPerTile;
  const int ScaleNsPerTile    = OptionType::GroupScaleConfig::ScaleNsPerTile;

  allocate(options);
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options, host_problem_shapes_available);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
   exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
    for (int32_t i = 0; i < options.groups; ++i) {
      std::cout << "    " << options.problem_sizes_host.at(i);
      std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
    }
    std::cout << "  Groups      : " << options.groups  << std::endl;
    std::cout << "  Tile shape (M, N, K): " << size<0>(TileShape{}) << ", " << size<1>(TileShape{}) << ", " << size<2>(TileShape{}) << std::endl;
    std::cout << "  ScaleGranularityM: " << ScaleGranularityM << " (ScaleMsPerTile: " << ScaleMsPerTile << ")" << std::endl;
    std::cout << "  ScaleGranularityN: " << ScaleGranularityN << " (ScaleNsPerTile: " << ScaleNsPerTile << ")" << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
    fflush(stdout);
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

  //
  // Parse options
  //

  Options<RasterOrderOptions, ProblemShape, GroupScale1D1DConfig> options_1d1d;
  Options<RasterOrderOptions, ProblemShape, GroupScale1D2DConfig> options_1d2d;
  Options<RasterOrderOptions, ProblemShape, GroupScale2D1DConfig> options_2d1d;
  Options<RasterOrderOptions, ProblemShape, GroupScale2D2DConfig> options_2d2d;

  options_1d1d.parse(argc, args);
  options_1d2d.parse(argc, args);
  options_2d1d.parse(argc, args);
  options_2d2d.parse(argc, args);

  if (options_1d1d.help) {
    options_1d1d.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

  auto run_tests = [&] (bool host_problem_shapes_available = true) {
    std::cout << "Grouped GEMM kernel with 1D1D group scale" << std::endl;
    run<GroupScale1D1DGemm::Gemm>(options_1d1d, host_problem_shapes_available);
    std::cout << "Grouped GEMM kernel with 1D2D group scale" << std::endl;
    run<GroupScale1D2DGemm::Gemm>(options_1d2d, host_problem_shapes_available);
    std::cout << "Grouped GEMM kernel with 2D1D group scale" << std::endl;
    run<GroupScale2D1DGemm::Gemm>(options_2d1d, host_problem_shapes_available);
    std::cout << "Grouped GEMM kernel with 2D2D group scale" << std::endl;
    run<GroupScale2D2DGemm::Gemm>(options_2d2d, host_problem_shapes_available);
    std::cout << std::endl;
  };

  std::cout << "Running tests with host problem shapes:" << std::endl;
  run_tests(true);
  std::cout << "Running tests without host problem shapes:" << std::endl;
  run_tests(false);

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/hopper_fp8_commandline.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Command line options parsing
template<typename _RasterOrderOptions, typename _ProblemShape, typename _GroupScaleConfig>
struct Options {

  using RasterOrderOptions = _RasterOrderOptions;
  using ProblemShape = _ProblemShape;
  using GroupScaleConfig = _GroupScaleConfig;

  bool help = false;

  float alpha = 1.f, beta = 0.f;
  int iterations = 1000;
  int m = 1024, n = 512, k = 1024, groups = 10;
  std::string benchmark_path;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
  int const k_alignment = 128;
  int const m_alignment = 128;
  int const n_alignment = 128;

  RasterOrderOptions raster;
  int swizzle;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

  }

  void randomize_problems(cutlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_host.reserve(groups);

    for (int i = groups; i > 0; i--) {
      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;
      if (m < 1) {
        m = m_alignment * ((rand() % (64 * alignment / m_alignment)) + 1);
      }
      if (n < 1) {
        n = n_alignment * ((rand() % (64 * alignment / n_alignment)) + 1);
      }
      if (k < 1) {
        k = k_alignment * ((rand() % (32 * alignment / k_alignment)) + 1);
      }
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
      }
    }
    groups = static_cast<int>(problem_sizes_host.size());

    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling\n\n"
      << "  Hopper FP8 Grouped GEMM using a Warp Specialized kernel with Blockwise Scaling.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = 0ull;

    for (auto const [m, n, k] : problem_sizes_host) {
      fmas += static_cast<uint64_t>(m) *
              static_cast<uint64_t>(n) *
              static_cast<uint64_t>(k);
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};
""",
    'key-files/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/reference/host/gemm_with_groupwise_scaling.h': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Reference implementation for GETT in host-side code.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/gemm.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/relatively_equal.h"
#include <iostream>
#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

template<class T, class = void>
struct ElementTraits {
  using type = T;
};

template<class T>
struct ElementTraits<T, std::enable_if_t<!std::is_same_v<decltype(std::declval<T>().get()), void> > >  {
  using type = decltype(std::declval<T>().get());
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementAccumulator_,
  class TensorA_,                                                                                         // (M, K, L)
  class TensorB_,                                                                                         // (N, K, L)
  class TensorScaleA_,                                                                                    // (m, k, L)
  class TensorScaleB_,                                                                                    // (n, k, L)
  class TileShape_
>
struct GettMainloopParams {
  using ElementAccumulator = ElementAccumulator_;
  using TensorA = TensorA_;
  using TensorB = TensorB_;
  using EngineA = typename TensorA::engine_type;
  using LayoutA = typename TensorA::layout_type;
  using EngineB = typename TensorB::engine_type;
  using LayoutB = typename TensorB::layout_type;

  using TensorScaleA = TensorScaleA_;
  using TensorScaleB = TensorScaleB_;
  using TileShape = TileShape_;
  using EngineScaleA = typename TensorScaleA::engine_type;
  using EngineScaleB = typename TensorScaleB::engine_type;

  TensorA A{};
  TensorB B{};
  TensorScaleA ScaleA{};
  TensorScaleB ScaleB{};  
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  class ElementScalar_,
  class ElementScalingFactor_,
  class ElementAccumulator_,
  class ElementCompute_,
  class TensorC_,                                                                                          // (M, N, L)
  class TensorD_,                                                                                          // (M, N, L)
  class VectorBias_ = TensorD_,                                                                            //    (M, 1)
  class TensorAux_ = TensorD_,                                                                             // (M, N, L)
  class VectorAlpha_ = TensorD_,                                                                           //    (M, 1)
  class VectorBeta_ = VectorAlpha_,                                                                        //    (M, 1)
  class ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementCompute_>,
  class BiasBinaryOp_ = cutlass::plus<ElementCompute_>,
  bool PerColumnBias_ = false
>
struct GettEpilogueParams {
  using ElementScalar = ElementScalar_;
  using ElementScalingFactor = ElementScalingFactor_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using TensorC = TensorC_;
  using TensorD = TensorD_;
  using TensorAux = TensorAux_;
  using VectorBias = VectorBias_;
  using VectorAlpha = VectorAlpha_;
  using VectorBeta = VectorBeta_;
  using ActivationFunctor = ActivationFunctor_;
  using BiasBinaryOp = BiasBinaryOp_;

  using EngineC = typename TensorC::engine_type;
  using LayoutC = typename TensorC::layout_type;
  using EngineD =  typename TensorD::engine_type;
  using LayoutD = typename TensorD::layout_type;
  static constexpr bool PerColumnBias = PerColumnBias_;
  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorC C{};
  TensorD D{};
  VectorBias Bias{};
  TensorAux Aux{};
  VectorAlpha Valpha{};
  VectorBeta Vbeta{};
  ElementCompute st = ElementCompute(1);

  ElementAccumulator* abs_max_D = nullptr;
  ElementAccumulator* abs_max_Aux = nullptr;

  ElementScalingFactor scale_a = ElementScalingFactor(1);
  ElementScalingFactor scale_b = ElementScalingFactor(1);
  ElementScalingFactor scale_c = ElementScalingFactor(1);
  ElementScalingFactor scale_d = ElementScalingFactor(1);
  ElementScalingFactor scale_aux = ElementScalingFactor(1);

  bool beta_per_channel_scaling = false;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - General Tensor-Tensor contraction reference kernel with Groupwise scaling
template <
  class MainloopParams,
  class EpilogueParams
>
void Gett(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{

  static int constexpr kBlockM = cute::get<0>(typename MainloopParams::TileShape{});
  static int constexpr kBlockN = cute::get<1>(typename MainloopParams::TileShape{});
  // printf("mainloop_params.ScaleA.layout()"); cute::print(mainloop_params.ScaleA.layout()); printf("\n");
  // printf("mainloop_params.ScaleB.layout()"); cute::print(mainloop_params.ScaleB.layout()); printf("\n");

#if defined(_OPENMP)
  #pragma omp parallel for collapse(3)
#endif
  for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
    for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
      for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
        typename MainloopParams::ElementAccumulator acc[kBlockM][kBlockN];
        gett_mainloop(mainloop_params, m, n, l, acc);
        gett_epilogue(epilogue_params, m, n, l, acc);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Mainloop
template <class MainloopParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_mainloop(
    MainloopParams const& mainloop_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == 3, "M, K, B");
  static_assert(cute::rank(typename MainloopParams::LayoutB{}) == 3, "N, K, B");

  using cute::raw_pointer_cast;

  using ElementA = typename ElementTraits<typename MainloopParams::EngineA::value_type>::type;
  using ElementB = typename ElementTraits<typename MainloopParams::EngineB::value_type>::type;
  using ElementBlockScaleA = typename ElementTraits<typename MainloopParams::EngineScaleA::value_type>::type;
  using ElementBlockScaleB = typename ElementTraits<typename MainloopParams::EngineScaleB::value_type>::type;

  using RingOp = multiply_add<ElementAccumulator, ElementAccumulator, ElementAccumulator>;
  RingOp fma_op;

  multiplies<ElementAccumulator> scale_op;

  static int constexpr kBlockK = cute::get<2>(typename MainloopParams::TileShape{});;

  // Tempo accumulators to seperate blockwise accumulation
  typename MainloopParams::ElementAccumulator acc_temp[kBlockM][kBlockN];

  // Zero out accumulators
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      acc[m_b][n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      acc_temp[m_b][n_b] = ElementAccumulator(0);
    }
  }

  const int M = cute::size<0>(mainloop_params.A.layout());
  const int N = cute::size<0>(mainloop_params.B.layout());

  const int ScaleGranularityM = M / cute::size<0>(mainloop_params.ScaleA.layout());
  const int ScaleGranularityN = N / cute::size<0>(mainloop_params.ScaleB.layout());

  assert(ScaleGranularityM && M % ScaleGranularityM == 0 && "ScaleGranularityM must divide M");
  assert(ScaleGranularityN && N % ScaleGranularityN == 0 && "ScaleGranularityN must divide N");

  cute::Tensor blockscale_A = domain_offset(make_coord(m / ScaleGranularityM, _0{}), mainloop_params.ScaleA(_, _, l));
  cute::Tensor blockscale_B = domain_offset(make_coord(n / ScaleGranularityN, _0{}), mainloop_params.ScaleB(_, _, l));

  // Compute on this k-block
  for (int64_t k = 0; k < cute::size<1>(mainloop_params.A.layout()); ++k) {

    // Load Blockwise scaling factor from blockscale Tensors for B
    int64_t block_k = k / kBlockK;
    cute::Tensor scale_a = blockscale_A(_, block_k);
    cute::Tensor scale_b = blockscale_B(_, block_k);

    // Load A
    ElementAccumulator a_frag[kBlockM];
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      if (m + m_b < cute::size<0>(mainloop_params.A.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        a_frag[m_b] = static_cast<ElementAccumulator>(ElementA(mainloop_params.A(m + m_b, k, l)));
      } else {
        a_frag[m_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // Load B
    ElementAccumulator b_frag[kBlockN];
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (n + n_b < cute::size<0>(mainloop_params.B.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        b_frag[n_b] = static_cast<ElementAccumulator>(ElementB(mainloop_params.B(n + n_b, k, l)));
      } else {
        b_frag[n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // do compute
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      for (int n_b = 0; n_b < kBlockN; ++n_b) {
        acc_temp[m_b][n_b] = fma_op(a_frag[m_b], b_frag[n_b], acc_temp[m_b][n_b]);
      }
    }

    // Apply Groupwise-scaling at kBlockK boundary
    // (a) Apply group and block scaling factors on the partial accumulated results (acc_temp) at the kBlocK boundary 
    // (b) Zero-out partial temporary (acc_temp),
    // (c) Update permanent (accu)
    if ((k+1) % kBlockK == 0) {
      for (int m_b = 0; m_b < kBlockM; ++m_b) {
        auto scale_a_m_b = scale_a[m_b / ScaleGranularityM];
        for (int n_b = 0; n_b < kBlockN; ++n_b) {
          auto scale_b_n_b = scale_b[n_b / ScaleGranularityN];
          ElementAccumulator blockwise_scaled_accum = acc_temp[m_b][n_b] * scale_a_m_b * scale_b_n_b;
          acc[m_b][n_b] = blockwise_scaled_accum + acc[m_b][n_b];
          acc_temp[m_b][n_b] = ElementAccumulator(0); 
        }
      }
    }

  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Epilogue
template <class EpilogueParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_epilogue(
    EpilogueParams const& epilogue_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == 3, "M, K, B");
  static_assert(cute::rank(typename EpilogueParams::LayoutD{}) == 3, "N, K, B");

  using cute::raw_pointer_cast;

  using ElementCompute = typename EpilogueParams::ElementCompute;
  using ElementC = typename EpilogueParams::TensorC::value_type;
  using ElementD = typename EpilogueParams::TensorD::value_type;
  using ElementAux = typename EpilogueParams::TensorAux::value_type;
  using ElementBias = typename EpilogueParams::VectorBias::value_type;
  using ElementScalar = typename EpilogueParams::ElementScalar;
  using ElementScalingFactor = typename EpilogueParams::ElementScalingFactor;
  using ActivationFunctor = typename EpilogueParams::ActivationFunctor;
  using BiasBinaryOp = typename EpilogueParams::BiasBinaryOp;

  constexpr bool PerColBias = EpilogueParams::PerColumnBias;
  constexpr bool IsScalingAndAmaxOutputNeeded = 
      cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
      cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

  constexpr bool IsScalingAndAmaxAuxOutputNeeded =
      cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
      cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

  constexpr bool IsReLUAuxNeeded =
      (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ReLu<ElementCompute>> or
       cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>) and 
      cute::is_same_v<ElementAux, cutlass::uint1b_t>;
  constexpr bool IsClamp =
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>;

  constexpr bool IsBackpropFusion =
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::dGELU<ElementCompute>> or
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::dReLU<ElementCompute>>;

  // Input related converter
  NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
  NumericConverter<ElementCompute, ElementC> source_converter;
  NumericConverter<ElementCompute, ElementBias> bias_converter;
  [[maybe_unused]] NumericConverter<ElementCompute, ElementAux> aux_source_converter;

  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;
  NumericConverter<ElementCompute, ElementScalingFactor> scaling_factor_converter;

  // Abs max converter
  [[maybe_unused]] NumericConverter<ElementAccumulator, ElementCompute> abs_max_output_converter;

  // Output related converter
  NumericConverter<ElementD, ElementCompute> destination_converter;
  [[maybe_unused]] NumericConverter<ElementAux, ElementCompute> aux_destination_converter;
  NumericConverter<ElementBias, ElementCompute> dBias_converter;

  // Epilogue operations
  multiply_add<ElementCompute, ElementCompute, ElementCompute> epilogue_fma;
  multiplies<ElementCompute> mul;
  plus<ElementCompute> add;

  // Activation operation

  auto activation = [] (ElementCompute x, ElementCompute y = ElementCompute(0)) {
    if constexpr (std::is_same_v<ActivationFunctor, void>) {
      return x + y;
    } else {
      return ActivationFunctor()(x, y);
    }
  };

  // Bias binary operation
  BiasBinaryOp bias_op;

  // Do conversion
  ElementCompute converted_alpha = scale_converter(epilogue_params.alpha);
  ElementCompute converted_beta = scale_converter(epilogue_params.beta);
  ElementCompute converted_scale_a = scaling_factor_converter(epilogue_params.scale_a);
  ElementCompute converted_scale_b = scaling_factor_converter(epilogue_params.scale_b);
  ElementCompute converted_scale_c = scaling_factor_converter(epilogue_params.scale_c);
  ElementCompute converted_scale_d = scaling_factor_converter(epilogue_params.scale_d);
  ElementCompute converted_scale_aux = scaling_factor_converter(epilogue_params.scale_aux);

  // Init local var
  [[maybe_unused]] ElementCompute local_abs_max_output = ElementCompute(0);
  [[maybe_unused]] ElementCompute local_abs_max_aux_output = ElementCompute(0);

  converted_alpha = mul(converted_alpha, mul(converted_scale_a, converted_scale_b));
  converted_beta = mul(converted_beta, converted_scale_c);

  ElementCompute inter_accum[kBlockM][kBlockN];

  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    ElementCompute local_dBias = ElementCompute(0);

    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        // Convert every type to ElementCompute first, do compute, convert to output type, write it out
        ElementCompute converted_acc = accumulator_converter(acc[m_b][n_b]);
        // per-row alpha
        if (raw_pointer_cast(epilogue_params.Valpha.data())) {
          converted_alpha = scale_converter(epilogue_params.Valpha(m + m_b));
        }
        ElementCompute output = mul(converted_alpha, converted_acc);

        if (raw_pointer_cast(epilogue_params.Bias.data()) && not IsBackpropFusion) {
          ElementCompute converted_bias = bias_converter(epilogue_params.Bias(PerColBias ? n + n_b : m + m_b));
          output = bias_op(output, converted_bias);
        }

        if (raw_pointer_cast(epilogue_params.C.data())) {
          ElementCompute converted_src = source_converter(epilogue_params.C(m + m_b, n + n_b, l));
          // per-row beta
          if (epilogue_params.Vbeta.data()) {
            converted_beta = scale_converter(epilogue_params.Vbeta(m + m_b));
          }
          output = epilogue_fma(converted_beta, converted_src, output);
        }

        if constexpr (IsBackpropFusion) {
          ElementAux aux_input = ElementAux(0);
          if (raw_pointer_cast(epilogue_params.Aux.data())) {
            aux_input = epilogue_params.Aux(m + m_b, n + n_b, l);
          }

          output = activation(output, aux_source_converter(aux_input));
          local_dBias = add(local_dBias, output);
        }
        else {
          if (raw_pointer_cast(epilogue_params.Aux.data())) {
            auto aux_output = output;
            if constexpr (IsScalingAndAmaxAuxOutputNeeded) {
              maximum_absolute_value_reduction<ElementCompute, true> amax_op;
              local_abs_max_aux_output = amax_op(local_abs_max_aux_output, aux_output);
              aux_output = epilogue_fma(converted_scale_aux, aux_output, ElementCompute(0));
            }

            if constexpr (IsReLUAuxNeeded) {
              epilogue_params.Aux(m + m_b, n + n_b, l) = not (aux_output < 0) ? uint1b_t(1) : uint1b_t(0);
            } else {
              epilogue_params.Aux(m + m_b, n + n_b, l) = aux_destination_converter(aux_output);
            }
          }

          if constexpr (IsClamp) { // Treat Clamp as ReLU
            output = activation(output, {0, std::numeric_limits<ElementCompute>::max()});
          }
          else {
            output = activation(output);
          }
        }

        if constexpr (IsScalingAndAmaxOutputNeeded) {
          maximum_absolute_value_reduction<ElementCompute, true> amax_op;
          local_abs_max_output = amax_op(local_abs_max_output, output);
          output = epilogue_fma(converted_scale_d, output, ElementCompute(0));
        }

        inter_accum[m_b][n_b] = ElementCompute(output);
      }
    } // n_b

    if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n < cute::size<1>(epilogue_params.D.layout())) {
      if (raw_pointer_cast(epilogue_params.Bias.data()) && IsBackpropFusion) {
        ElementCompute converted_dBias = bias_converter(epilogue_params.Bias(m + m_b));
        local_dBias = add(local_dBias, converted_dBias);
        epilogue_params.Bias(m + m_b) = dBias_converter(local_dBias);
      }
    }
  } // m_b
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        epilogue_params.D(m + m_b, n + n_b, l) = destination_converter(inter_accum[m_b][n_b]);
      }
    }
  }

#if defined(_OPENMP)
  #pragma omp critical(Abs_Max_Data_Update)
#endif
  {
    if constexpr (IsScalingAndAmaxOutputNeeded) {
      if (epilogue_params.abs_max_D) {
        *epilogue_params.abs_max_D = maximum_with_nan_propogation<ElementAccumulator>{}(
          *epilogue_params.abs_max_D, abs_max_output_converter(local_abs_max_output));
      }
    }

    if constexpr (IsScalingAndAmaxAuxOutputNeeded) {
      if (epilogue_params.abs_max_Aux) {
        *epilogue_params.abs_max_Aux = maximum_with_nan_propogation<ElementAccumulator>{}(
            *epilogue_params.abs_max_Aux, abs_max_output_converter(local_abs_max_aux_output));
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM - General Matrix-Matrix contraction without conjugation options
template <
  class MainloopParams,
  class EpilogueParams
>
void Gemm3x(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{
  using namespace cute;

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == cute::rank(typename MainloopParams::LayoutB{}));
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == cute::rank(typename EpilogueParams::LayoutD{}));
  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == cute::rank(typename EpilogueParams::LayoutC{}));
  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == 3, "Only Rank3 Tensors (M, K, Batch_Count) "
                                                                     "with Batchmode are supported");
  // Lower the Matrix-Multiplication with Groupwise scaling (Gemm3x) to a Tensor Contraction (Gett).
  Gett(mainloop_params, epilogue_params);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A FP8 blockwise scaled GEMM example for the NVIDIA Blackwell SM100 architecture using CUTLASS.
*/



#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/util/reference/host/gett.hpp"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration
using ElementA            = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB            = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC            = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using LayoutC             = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

using ElementD           = ElementC;
using LayoutD            = LayoutC;
constexpr int AlignmentD = AlignmentC;

// MMA type
using ElementAccumulator = float;                                           // Element Accumulator will also be our scale factor type
using ElementCompute = float;


// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0 
using MmaTileShape_MNK = Shape<_128,_128,_128>;                          
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_1,_1,_1>;
 
using ScaleConfig = decltype(cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));

using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand


using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100 // Note: Groupwise and Blockwise only support 1 SM MMA at this moment
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                // Default to ClusterLaunchControl (CLC) based tile scheduler

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
// Strides just iterate over scalars and have no zeros
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;
// Layouts are tiled to the problem size and the strides have zeros
uint64_t seed;

cutlass::HostTensor<ElementA          , LayoutA> tensor_A;
cutlass::HostTensor<ElementAccumulator, cutlass::layout::PackedVectorLayout> tensor_SFA;
cutlass::HostTensor<ElementB          , LayoutB> tensor_B;
cutlass::HostTensor<ElementAccumulator, cutlass::layout::PackedVectorLayout> tensor_SFB;
cutlass::HostTensor<ElementC          , LayoutC> tensor_C;
cutlass::HostTensor<ElementD          , LayoutD> tensor_D;
cutlass::HostTensor<ElementD          , LayoutD> tensor_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;
  bool skip_verification = false;

  float alpha = 1.f, beta = 0.f;
  int iterations = 1000;
  int m = 1024, n = 512, k = 1024, l = 1;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("skip-verification")) {
      skip_verification = true;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "81_blackwell_gemm_blockwise\n\n"
      << "  Blackwell FP8 GEMM with Blockwise Scaling using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the l extent (batch) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --skip-verification         Skip verification.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "81_blackwell_gemm_blockwise" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;
    int bits_output = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 8) {
      scope_max = 2;
      scope_min = -2;
    } else if (bits_output == 16) {
      scope_max = 5;
      scope_min = -5;
    } else {
      scope_max = 8;
      scope_min = -8;
    }

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Helper to initialize a block of device data (scale_tensors)
template <typename Element, typename Layout>
bool initialize_scale_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;

    scope_min = -8;
    scope_max = 8;

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;

  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto blockscale_a_coord = cutlass::make_Coord(size(filter_zeros(layout_SFA)));
  auto blockscale_b_coord = cutlass::make_Coord(size(filter_zeros(layout_SFB)));

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);
  tensor_SFA.resize(blockscale_a_coord);
  tensor_SFB.resize(blockscale_b_coord);

  initialize_tensor(tensor_A.host_view(), cutlass::Distribution::Uniform, seed + 2022);
  initialize_tensor(tensor_B.host_view(), cutlass::Distribution::Uniform, seed + 2023);
  initialize_tensor(tensor_C.host_view(), cutlass::Distribution::Uniform, seed + 2024);

  initialize_scale_tensor(tensor_SFA.host_view(), cutlass::Distribution::Uniform, seed + 2025);
  initialize_scale_tensor(tensor_SFB.host_view(), cutlass::Distribution::Uniform, seed + 2026);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  tensor_SFA.sync_device();
  tensor_SFB.sync_device();

}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, 
     tensor_B.device_data(), stride_B,
     tensor_SFA.device_data(), layout_SFA,
     tensor_SFB.device_data(), layout_SFB},
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;

  return arguments;
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto C = cute::make_tensor(tensor_C.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto SFA = cute::make_tensor(tensor_SFA.host_data(), layout_SFA);
  auto SFB = cute::make_tensor(tensor_SFB.host_data(), layout_SFB);

  using unused_t = decltype(D);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(A), 
      decltype(SFA), 
      decltype(B),
      decltype(SFB)
    > mainloop_params{A, SFA, B, SFB};

  cutlass::reference::host::GettEpilogueParams<
      ElementAccumulator,
      ElementAccumulator,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D)
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  
  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  
  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

 
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  
  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  Result result;
  if (!options.skip_verification) {
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    result.passed = verify(options);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
      exit(-1);
    }
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least sm100a.
  
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100a)." << std::endl;
    return 0;
  } 
  

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Run
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

""",
    'key-files/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A FP8 groupwise scaled GEMM example for the NVIDIA Blackwell SM100 architecture using CUTLASS.
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/util/reference/host/gett.hpp"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration
using ElementA            = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB            = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC            = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using LayoutC             = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

using ElementD           = ElementC;
using LayoutD            = LayoutC;
constexpr int AlignmentD = AlignmentC;

// MMA type
using ElementAccumulator = float;                                           // Element Accumulator will also be our scale factor type
using ElementCompute = float;


// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0 
using MmaTileShape_MNK = Shape<_128,_128,_128>;                          
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_1,_1,_1>;
 
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

// Note when we have multiple scale factors per tile (in this case 128 scales in M per tile), we will restrict up to a 
// 16B alignment if possible (i.e., we have at least 16B of scales in M).
// In this case the smallest M that can be executed is 16. To avoid this for smaller M, you can swap A and B
// and transpose A, B, C, and scales since B^T A^T = C^T.
using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand


using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100 // Note: Groupwise and Blockwise only support 1 SM MMA at this moment
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                // Default to ClusterLaunchControl (CLC) based tile scheduler

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
// Strides just iterate over scalars and have no zeros
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;
// Layouts are tiled to the problem size and the strides have zeros
uint64_t seed;

cutlass::HostTensor<ElementA          , LayoutA> tensor_A;
cutlass::HostTensor<ElementAccumulator, cutlass::layout::PackedVectorLayout> tensor_SFA;
cutlass::HostTensor<ElementB          , LayoutB> tensor_B;
cutlass::HostTensor<ElementAccumulator, cutlass::layout::PackedVectorLayout> tensor_SFB;
cutlass::HostTensor<ElementC          , LayoutC> tensor_C;
cutlass::HostTensor<ElementD          , LayoutD> tensor_D;
cutlass::HostTensor<ElementD          , LayoutD> tensor_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;
  bool skip_verification = false;

  float alpha = 1.f, beta = 0.f;
  int iterations = 1000;
  int m = 1024, n = 512, k = 1024, l = 1;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("skip-verification")) {
      skip_verification = true;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "81_blackwell_gemm_groupwise\n\n"
      << "  Blackwell FP8 GEMM with Groupwise Scaling using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the l extent (batch) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --skip-verification         Skip verification.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "81_blackwell_gemm_groupwise" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;
    int bits_output = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 8) {
      scope_max = 2;
      scope_min = -2;
    } else if (bits_output == 16) {
      scope_max = 5;
      scope_min = -5;
    } else {
      scope_max = 8;
      scope_min = -8;
    }

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Helper to initialize a block of device data (scale_tensors)
template <typename Element, typename Layout>
bool initialize_scale_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;

    scope_min = -8;
    scope_max = 8;

    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;

  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto blockscale_a_coord = cutlass::make_Coord(size(filter_zeros(layout_SFA)));
  auto blockscale_b_coord = cutlass::make_Coord(size(filter_zeros(layout_SFB)));

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);
  tensor_SFA.resize(blockscale_a_coord);
  tensor_SFB.resize(blockscale_b_coord);

  initialize_tensor(tensor_A.host_view(), cutlass::Distribution::Uniform, seed + 2022);
  initialize_tensor(tensor_B.host_view(), cutlass::Distribution::Uniform, seed + 2023);
  initialize_tensor(tensor_C.host_view(), cutlass::Distribution::Uniform, seed + 2024);

  initialize_scale_tensor(tensor_SFA.host_view(), cutlass::Distribution::Uniform, seed + 2025);
  initialize_scale_tensor(tensor_SFB.host_view(), cutlass::Distribution::Uniform, seed + 2026);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  tensor_SFA.sync_device();
  tensor_SFB.sync_device();

}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, 
     tensor_B.device_data(), stride_B,
     tensor_SFA.device_data(), layout_SFA,
     tensor_SFB.device_data(), layout_SFB},
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;

  return arguments;
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto C = cute::make_tensor(tensor_C.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto SFA = cute::make_tensor(tensor_SFA.host_data(), layout_SFA);
  auto SFB = cute::make_tensor(tensor_SFB.host_data(), layout_SFB);

  using unused_t = decltype(D);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(A), 
      decltype(SFA), 
      decltype(B),
      decltype(SFB)
    > mainloop_params{A, SFA, B, SFB};

  cutlass::reference::host::GettEpilogueParams<
      ElementAccumulator,
      ElementAccumulator,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D)
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  
  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  
  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

 
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  
  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  Result result;
  if (!options.skip_verification) {
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    result.passed = verify(options);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
      exit(-1);
    }
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least sm100a.
  
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100a)." << std::endl;
    return 0;
  } 
  

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Run
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/detail/sm100_blockwise_scale_layout.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/



/*! \file
    \brief Block Wise Scale configs specific for SM100 Blockwise/Groupwise MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace cutlass::detail{

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
struct Sm100BlockwiseScaleConfig {

  using ShapeSFA = Shape<Shape<Int<SFVecSizeM>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;
  using ShapeSFB = Shape<Shape<Int<SFVecSizeN>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;

  using StrideSFA = conditional_t<majorSFA == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using StrideSFB = conditional_t<majorSFB == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutSFA = Layout<ShapeSFA, StrideSFA>;
  using LayoutSFB = Layout<ShapeSFB, StrideSFB>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSFA{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFA(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K] = cta_shape_mnk;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }


  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSFB{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFB(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFA(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(M, SFVecSizeM)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto mk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, cute::ceil_div(M, SFVecSizeM)),
                 make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFB(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;

      if constexpr (majorSFB == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(N, SFVecSizeN)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto nk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, cute::ceil_div(N, SFVecSizeN)),
                 make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(nk_layout), L), append(stride(nk_layout), size(filter_zeros(nk_layout))));
  }

};

template<class MmaTileShape_MNK>
constexpr auto sm100_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return Sm100BlockwiseScaleConfig<size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{})>{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
""",
    'key-files/include/cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  class TransformLoadPipelineStorage,
  class TransformPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockwise(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  class TransformLoadPipelineStorage,
  class TransformPipelineStorage,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockwise(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count. 
template<
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class ElementScalar,
  class ScaleShapeMNK,
  class TileShapeMNK,
  class MainloopPipelineStorage,
  class TransformLoadPipelineStorage,
  class TransformPipelineStorage,
  int carveout_bytes>
constexpr int
sm100_compute_stage_count_or_override_blockwise(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For F8/F6/F4 sub-bytes, ElementA/B will be passed in as uint8_t
  // For Planar Complex, ElementA/B will be passed in as cutlass::complex<ElementARaw>
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one of each of the pipelines
  constexpr auto pipeline_bytes = sizeof(MainloopPipelineStorage) + 
      sizeof(TransformLoadPipelineStorage) + sizeof(TransformPipelineStorage);

  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementScalar>;

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * size<0>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * size<1>(ScaleShapeMNK{}) * size<2>(ScaleShapeMNK{})) +
    static_cast<int>(pipeline_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA,
  class GmemLayoutATagPair,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTagPair,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATagPair,
    AlignmentA,
    ElementB,
    GmemLayoutBTagPair,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      not cute::is_tuple_v<ElementA>   && not cute::is_tuple_v<ElementB> &&
      not cute::is_complex_v<ElementA> && not cute::is_complex_v<ElementB> &&
      cute::is_tuple_v<GmemLayoutATagPair>   && cute::is_tuple_v<GmemLayoutBTagPair> &&
      // Dense Gemm
      cute::is_base_of_v<KernelScheduleSm100Blockwise, KernelScheduleType> &&
      // Alignment check
      detail::sm1xx_gemm_is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, KernelScheduleType>()>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::check_input_datatypes<ElementA, ElementB>(), "Incorrect input types");

  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutATagPair{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutATagPair{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutBTagPair{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutBTagPair{}))>;

  static_assert(cute::depth(GmemLayoutSFATag{}) == 2 and cute::depth(GmemLayoutSFBTag{}) == 2, 
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(GmemLayoutSFATag{}) == size<1, 0>(GmemLayoutSFBTag{}), 
      "SFA and SFB must have equivalent SF vector sizes along K");

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  // Data type used by MMA instruction
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm100_kernel_input_element_to_mma_input_element<ElementA>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm100_kernel_input_element_to_mma_input_element<ElementB>());

  static constexpr bool is_2sm = cute::is_base_of_v<KernelSchedule2Sm, KernelScheduleType> ||
                        (not cute::is_base_of_v<KernelSchedule1Sm, KernelScheduleType> &&
                          not cute::is_base_of_v<KernelSchedule2Sm, KernelScheduleType> &&
                          cute::is_static_v<ClusterShape_MNK> &&
                          cute::get<0>(ClusterShape_MNK{}) % 2 == 0 );

  static_assert(detail::sm100_gemm_check_for_f8f6f4_mix8bit_requirement<ElementAMma, ElementBMma,
                                                                      TileShape_MNK, ClusterShape_MNK,
                                                                      UmmaMajorA, UmmaMajorB, KernelScheduleType, is_2sm>(),
                "TileSize and MNK Major does not met with MMA Mix 8-bit TMA load requirement" );
  using TiledMma =  decltype(detail::sm100_make_trivial_tiled_mma<
      ElementAMma, ElementBMma, ElementAccumulator,
      decltype(cute::product_each(TileShape_MNK{})), ClusterShape_MNK,
      UmmaMajorA, UmmaMajorB, KernelScheduleType>());

  using ElementAMma_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using AtomThrID = typename TiledMma::AtomThrID;

  using AtomThrShapeMNK = cute::Shape<decltype(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaTileShape_MNK = decltype(cute::shape_div(TileShape_MNK{}, AtomThrShapeMNK{}));

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));

  static_assert(BlockTileA_K{} == BlockTileB_K{}, "Block tile Ks should be equal");

  using SmemShape_M = decltype(shape_div(shape<0>(TileShape_MNK{}), shape_div(shape<0>(TileShape_MNK{}), size<0>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(TileShape_MNK{}), shape_div(shape<1>(TileShape_MNK{}), size<1>(TileShape_MNK{}) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(TileShape_MNK{}));

  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
    ClusterShape_MNK{}, AtomThrID{}));
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, SmemShape_M, SmemShape_K>());
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, SmemShape_N, SmemShape_K>());
  static constexpr uint32_t TotalTmemRows = 128;
  static constexpr uint32_t Sm100TmemCapacityColumns = 512;
  static constexpr uint32_t TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;
  static constexpr uint32_t AccumulatorPipelineStageCount = (is_2sm || (!is_2sm && size(shape<0,0>(MmaShapeA_MK{}) > 64))) ? 
                                                              TotalTmem / (cute::size<0>(CtaTileShape_MNK{}) * cute::size<1>(CtaTileShape_MNK{}))
                                                            : (Sm100TmemCapacityColumns / cute::size<1>(CtaTileShape_MNK{})) * 2;                       // 1SM MMA_M = 64 case
  static_assert(AccumulatorPipelineStageCount > 0, "Accumulator pipeline stage count must be positive.  This error probably means that TileShape_MNK and/or TiledMma::ThrLayoutVMNK are wrong.");

  // Calculate scheduler pipeline stages. Having one more stage than the accumulator allows more latency hiding.
  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using InternalStrideA  = cute::remove_pointer_t<StrideA>;
  // Grouped GEMM (where Stride type is Stride*) does not use CLC based scheduler.
  // SchedulerPipelineStageCount could be set to zero for Grouped GEMM, but we shouldn't define CLC Pipeline's barrier arrays of size zero.
  static constexpr uint32_t SchedulerPipelineStageCount = cute::is_same_v<InternalStrideA, StrideA> ? (AccumulatorPipelineStageCount + 1) : 1;

  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      false
    >::KernelSmemCarveout;
  // Reduce SMEM capacity available for buffers considering barrier allocations.
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;
  using MainloopPipelineStorage = typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage;
  using TransformLoadPipelineStorage = typename cutlass::PipelineAsync<1>::SharedStorage;
  using TransformPipelineStorage = typename cutlass::PipelineUmmaAsync<1>::SharedStorage;

  static constexpr int ScaleGranularityM = size<0,0>(GmemLayoutSFATag{});
  static constexpr int ScaleGranularityN = size<0,0>(GmemLayoutSFBTag{});
  static constexpr int ScaleGranularityK = size<1,0>(GmemLayoutSFBTag{});

  static_assert(size<0>(CtaTileShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaTileShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaTileShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using BlockTileScale_M = Int<size<0>(TileShape_MNK{}) / ScaleGranularityM>;
  using BlockTileScale_N = Int<size<1>(TileShape_MNK{}) / ScaleGranularityN>;
  using BlockTileScale_K = Int<size<2>(TileShape_MNK{}) / ScaleGranularityK>;

  using ScaleTileShape = cute::Shape<BlockTileScale_M, BlockTileScale_N, BlockTileScale_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockwise<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, 
      ElementAccumulator, ScaleTileShape, SmemTileShape, MainloopPipelineStorage,
      TransformLoadPipelineStorage, TransformPipelineStorage>(StageCountType{});
  static_assert(PipelineStages > 0, "Smem usage is too high. Can't create any SMEM buffers for A, B, and scales.");

  using DispatchPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
      PipelineStages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      cute::tuple<cutlass::gemm::TagToStrideA_t<GmemLayoutATag>, cutlass::gemm::TagToStrideA_t<GmemLayoutSFATag>>,
      ElementB,
      cute::tuple<cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>, cutlass::gemm::TagToStrideB_t<GmemLayoutSFBTag>>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;
};


} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override(cute::Int<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity in gemm of blockwise/groupwise scale.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementBlockScale, class TileShapeMNK, int ScaleMsPerTile, int ScaleNsPerTile, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_with_blockwise_scale(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto scale_bits = cute::sizeof_bits_v<ElementBlockScale>;
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(scale_bits * ScaleMsPerTile) + // scale of tensor A
    cutlass::bits_to_bytes(scale_bits * ScaleNsPerTile);  // scale of tensor B

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int stages, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCount<stages> stage_count) {
  return stages;
}

template <class Element>
constexpr int get_bits_for_possibly_void_element() {
  if constexpr (cute::is_same_v<Element, void>) {
    return 0;
  }
  else {
    return sizeof_bits<Element>::value;
  }
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix), or overrides with manual count.
template<int capacity_bytes_, class ElementA, class ElementB, class ElementScale, class ElementZero, class TileShapeMNK, int carveout_bytes_, int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input(StageCountAutoCarveout<carveout_bytes_> stage_count) {

  // 32 bytes to account for barriers etc.
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int scale_zero_k_tile = 1;
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto s_bits = get_bits_for_possibly_void_element<ElementScale>();
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();

  constexpr auto scale_bytes = cutlass::bits_to_bytes(s_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  constexpr auto zero_bytes  = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  static_assert(scale_bytes % 128 == 0, "Scale bytes must be a multiple of 128");
  static_assert(zero_bytes  % 128 == 0, "Zero bytes must be a multiple of 128");

  // When scales are void, s_bits will be 0 so no smem will be allocated for scales.
  constexpr int stage_bytes_ =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    scale_bytes + zero_bytes;

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
    static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB>
constexpr bool
is_swapAB(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAkBmn = cutlass::gemm::detail::is_k_major_A<LayoutA>() &&
                                 cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool SwapAB = !IsInputSizeTwoBytes && IsLayoutAkBmn;
  return SwapAB;
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class KernelScheduleType>
constexpr bool
is_warpspecialized_transpose_B(){
  constexpr bool IsInputSizeTwoBytes = is_input_size_two_bytes<ElementA, ElementB>();
  constexpr bool IsLayoutAmnBmn = cutlass::gemm::detail::is_mn_major_A<LayoutA>() &&
                                  cutlass::gemm::detail::is_mn_major_B<LayoutB>();
  constexpr bool IsWarpSpecialized = cute::is_base_of_v<KernelTmaWarpSpecialized, KernelScheduleType>                ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, KernelScheduleType>        ||
                                     cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType>     ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecialized, KernelScheduleType>            ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedPingpong, KernelScheduleType>    ||
                                     cute::is_base_of_v<KernelCpAsyncWarpSpecializedCooperative, KernelScheduleType>;
  constexpr bool IsWarpSpecializedTransposeB = !IsInputSizeTwoBytes && IsLayoutAmnBmn && IsWarpSpecialized;
  return IsWarpSpecializedTransposeB;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_any_of_v<KernelScheduleType,
                         KernelTmaWarpSpecialized,
                         KernelTmaWarpSpecializedCooperative,
                         KernelTmaWarpSpecializedPingpong,
                         KernelPtrArrayTmaWarpSpecializedCooperative,
                         KernelPtrArrayTmaWarpSpecializedPingpong>) &&
       not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpong>);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperative,
                                                          KernelPtrArrayTmaWarpSpecializedCooperative>;
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
 
  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<Sm90ReducedSmemCapacityBytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  /* For FP8 use a separate mainloop compared to other datatypes */
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      cute::conditional_t<IsFP8Input,
          MainloopSm90ArrayTmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>
      >,
      cute::conditional_t<IsFP8Input,
          MainloopSm90TmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
          MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>
      >
  >;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_RS 
template <
  class ElementA_,
  class GmemLayoutATag_,
  int AlignmentA,
  class ElementB_,
  class GmemLayoutBTag_,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA_,
    GmemLayoutATag_,
    AlignmentA,
    ElementB_,
    GmemLayoutBTag_,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedPingpong> ||
       cute::is_same_v<KernelScheduleType,  KernelTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType,  KernelPtrArrayTmaWarpSpecializedPingpong>) && 
      (detail::is_use_rmem_A<ElementA_, GmemLayoutATag_, ElementB_, GmemLayoutBTag_>() ||
       // ConvertAndScale and ConvertAndScaleWithZero 
       cute::is_tuple<ElementA_>::value || cute::is_tuple<ElementB_>::value || 
       // DirectConvert
       sizeof_bits<ElementA_>::value != sizeof_bits<ElementB_>::value)>
> {

private:
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementA_>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementB_>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementA_>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementB_>;
  static constexpr bool NeitherIsTuple = !cute::is_tuple<ElementA_>::value && !cute::is_tuple<ElementB_>::value;
  // Determine if mixed input types.
  static constexpr bool IsMixedInput =
    cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementA_>> != cute::sizeof_bits_v<detail::deduce_mixed_width_dtype_t<0, ElementB_>>;
  static constexpr bool IsArrayOfPointersGemm = cute::is_any_of_v<KernelScheduleType,
                                                                  KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                  KernelPtrArrayTmaWarpSpecializedPingpong>;
  static_assert(IsMixedInput || !IsArrayOfPointersGemm, "Only mixed input grouped RS GEMM is supported.");

public:
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementA_>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementB_>;

  static_assert(!IsMixedInput || (cute::is_tuple<ElementA_>::value ^ cute::is_tuple<ElementB_>::value ||
               (NeitherIsTuple && (sizeof_bits<ElementA>::value != sizeof_bits<ElementB>::value))),
    "Either A OR B must be a tuple or the widths of A and B must be different.");

  static constexpr bool IsANarrow = sizeof_bits<ElementA>::value < sizeof_bits<ElementB>::value;

  template<class T>
  static auto get_stride(T const& t) {
    if constexpr (not cute::is_layout<cute::remove_pointer_t<T>>::value) {
      return t;
    }
    else {
      if constexpr (cute::is_pointer_v<T>) {
        return &cute::stride(*t);
      }
      else {
        return cute::stride(t);
      }
    }
  }

  using GmemLayoutATag = decltype(get_stride(GmemLayoutATag_{}));
  using GmemLayoutBTag = decltype(get_stride(GmemLayoutBTag_{}));

  using ElementPairA = cute::conditional_t<IsMixedInput && IsANarrow && NeitherIsTuple, cute::tuple<ElementA>, ElementA_>;
  using ElementPairB = cute::conditional_t<IsMixedInput && !IsANarrow && NeitherIsTuple, cute::tuple<ElementB>, ElementB_>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementPairA>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  // If A is scaled, then we don't need to swap. Otherwise, we must ensure B goes to rmem and we must swap the operands.
  static constexpr bool SwapAB = IsMixedInput ? !IsATransformed : detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();
  static_assert(!IsMixedInput || !IsWarpSpecializedTransposeB, "Mixed input GEMM does not support WS transpose B.");

  // When we relax the above assertion, we must handle setting the tile mma GmmaMajorB correctly.
  static constexpr cute::GMMA::Major TiledMmaGmmaMajorB = SwapAB ? GmmaMajorA : GmmaMajorB;

  // For fp32 types, map to tf32 MMA value type.
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // Handle mixed dtypes and MMA.
  using RealElementA = cute::conditional_t<SwapAB, ElementBMma, ElementAMma>;
  using RealElementB = cute::conditional_t<SwapAB, ElementAMma, ElementBMma>;
  using RealElementAMma = cute::conditional_t<IsMixedInput, RealElementB, RealElementA>;
  // Always the same for element B.
  using RealElementBMma = RealElementB;

  static_assert(!IsMixedInput || TiledMmaGmmaMajorB == GMMA::Major::K || sizeof_bits<RealElementB>::value == 16, 
      "Mixed input GEMM does not support MN major layout except for 16bit");

  using AtomLayoutMNK = cute::conditional_t<
      cute::is_any_of_v<KernelScheduleType, 
                        KernelTmaWarpSpecializedCooperative,
                        KernelPtrArrayTmaWarpSpecializedCooperative>,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      RealElementAMma, RealElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomA{}); 
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutAtomB{});
  static constexpr int    SmemAlignment  = static_cast<int>(cute::max(SmemAlignmentA, SmemAlignmentB));

  // Handle mixed dtype array GEMM's size of tensor map storage.
  static constexpr size_t TensorMapStorage = sizeof(cute::TmaDescriptor) * size_t(IsMixedInput) * 4;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = IsMixedInput ?
      ( IsArrayOfPointersGemm ? 
        detail::compute_stage_count_or_override_single_affine_transformed_input<Sm90ReducedSmemCapacityBytes,
          RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{}) : 
        detail::compute_stage_count_or_override_single_affine_transformed_input<detail::sm90_smem_capacity_bytes,
          RealElementA, RealElementB, ElementScale, ElementZero, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{})
      ) 
      : detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
          ElementAMma, ElementBMma, TileShape_MNK, StageCountType::bytes, SmemAlignment>(StageCountType{});
      
  using DispatchPolicy = cute::conditional_t<IsMixedInput,
      cute::conditional_t<IsArrayOfPointersGemm,
        MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>, 
        MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<PipelineStages, ClusterShape_MNK, KernelScheduleType>>, 
        MainloopSm90TmaGmmaRmemAWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;
  
  // We pack the scale data with the operand that will be optionally scaled and converted before MMA.
  using StrideA = cute::conditional_t<cute::is_layout<cute::remove_pointer_t<GmemLayoutATag_>>::value, GmemLayoutATag_, TagToStrideA_t<GmemLayoutATag>>;
  using StrideB = cute::conditional_t<cute::is_layout<cute::remove_pointer_t<GmemLayoutBTag_>>::value, GmemLayoutBTag_, TagToStrideB_t<GmemLayoutBTag>>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementPairA,
      StrideA,
      ElementPairB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;

  static_assert(SmemAlignment == static_cast<int>(cute::max(CollectiveOp::SmemAlignmentA, CollectiveOp::SmemAlignmentB)));
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_FP8_FAST_ACCUM_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_any_of_v<KernelScheduleType,
                        KernelTmaWarpSpecializedFP8FastAccum,
                        KernelTmaWarpSpecializedPingpongFP8FastAccum,
                        KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                        KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
                        KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Not meet TMA alignment requirement yet\n");
  static_assert(detail::is_input_fp8<ElementA, ElementB>(),
                "Only FP8 datatypes are compatible with these kernel schedules\n");
  // Dispatch TN fp8 kernels only to TMA warp specialized FP8 builder
  static_assert(!detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>(),
                 "Not supported for fp8 non-TN warp specialized kernels yet\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  static constexpr bool IsArrayOfPointersGemm = cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>;

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                                                          KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>;

  using AtomLayoutMNK = cute::conditional_t<IsCooperative, Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementA, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementB, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);
  static constexpr int Sm90ReducedSmemCapacityBytes = detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<Sm90ReducedSmemCapacityBytes,
      ElementA, ElementB, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
      MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelTma> &&
                     not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>()));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopSm90TmaGmma<PipelineStages, ClusterShape_MNK>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct [[deprecated("Use one of KernelCpAsyncWarpSpecialized schedules instead")]]
CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  // Map to warp-specialized kernels for better performance
  using CollectiveOp = typename CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelCpAsyncWarpSpecialized
  >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync_WS_SS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutBTag>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ? 2 : 1;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_CpAsync_WS_RS
template <
  class ElementA,
  class GmemLayoutATag,
  int   AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int   AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecialized> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative> ||
       cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedPingpong>) &&
      detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::cp_async_min_alignment_bytes>(),
                "Minimum alignment required for cp.async is 4B.");

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_rs_tag_to_major_A<GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_rs_tag_to_major_B<GmemLayoutBTag>();
  static constexpr bool SwapAB = detail::is_swapAB<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>();
  static constexpr bool IsWarpSpecializedTransposeB = detail::is_warpspecialized_transpose_B<
      ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag, KernelScheduleType>();

  using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelCpAsyncWarpSpecializedCooperative>,
      Layout<Shape<cute::Int<(size<0>(TileShape_MNK{}) < 128) ? 1 : 2>,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GMMA::Major::K, GMMA::Major::K>(), AtomLayoutMNK{}));

  static constexpr int NumLoadWarpGroups = 1;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NumThreadsPerWarpGroup * NumLoadWarpGroups, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::rs_smem_selector<GmmaMajorA, ElementAMma,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());
  using SmemLayoutAtomB = decltype(detail::rs_smem_selector<GmmaMajorB, ElementBMma,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{})), IsWarpSpecializedTransposeB>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

  using DispatchPolicy = MainloopSm90CpAsyncGmmaRmemAWarpSpecialized<
      PipelineStages, ClusterShape_MNK, KernelScheduleType>;

  using SmemCopyAtomA = cute::conditional_t<SwapAB, void, Copy_Atom<cute::AutoVectorizingCopy, ElementA>>;
  using SmemCopyAtomB = cute::conditional_t<SwapAB, Copy_Atom<cute::AutoVectorizingCopy, ElementB>, void>;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA auto kernel schedule
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

using ExtractedElementA = detail::deduce_mixed_width_dtype_t<0, ElementA>;
using ExtractedElementB = detail::deduce_mixed_width_dtype_t<0, ElementB>;

static constexpr bool IsTmaCompatible = detail::is_aligned<
    ExtractedElementA, AlignmentA, ExtractedElementB, AlignmentB, detail::tma_alignment_bytes>();

// Users opt into scales via the builder by passing a tuple of Elements for the input that will be scaled. We detect
// scale support if ONLY one of the inputs have tuples to describe them.
static constexpr bool OnlyOneIsTuple = cute::is_tuple<ElementA>::value ^ cute::is_tuple<ElementB>::value;
static constexpr bool IsDifferentWidth = sizeof_bits<ExtractedElementA>::value != sizeof_bits<ExtractedElementB>::value;
static constexpr bool IsMixedWidthInput = IsDifferentWidth || (IsDifferentWidth && OnlyOneIsTuple);

#if ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
  // Persistent schedules perform best for CUDA Toolkits with version >= 12.1
  // KernelTmaWarpSpecializedCooperative requires TileShape_M to be at least 128
  using KernelTmaWarpSpecializedSchedule = cute::conditional_t<size<0>(TileShape_MNK{}) == Int<64>{},
      KernelTmaWarpSpecializedPingpong, KernelTmaWarpSpecializedCooperative>;
#else
  using KernelTmaWarpSpecializedSchedule = KernelTmaWarpSpecialized;
#endif

  // Non-persistent schedule is a safer choice for CpAsync kernels due to register pressure
  using KernelCpAsyncWarpSpecializedSchedule = KernelCpAsyncWarpSpecialized;
  using KernelSchedule = cute::conditional_t<IsTmaCompatible, KernelTmaWarpSpecializedSchedule, KernelCpAsyncWarpSpecializedSchedule>;
  static_assert((cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedSchedule> && IsMixedWidthInput) || !IsMixedWidthInput, "Only TMA warp specialized kernels are supported for mixed width input.");
  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm90,
      arch::OpClassTensorOp,
      ElementA,
      GmemLayoutATag,
      AlignmentA,
      ElementB,
      GmemLayoutBTag,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS (BlockScaled Builders)
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<decltype(KernelScheduleType::ScaleGranularityM), decltype(KernelScheduleType::ScaleGranularityN)> and
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()
    >
> {

  static constexpr auto ScaleGranularityM_ = KernelScheduleType::ScaleGranularityM;
  static constexpr auto ScaleGranularityN_ = KernelScheduleType::ScaleGranularityN;
  static constexpr auto ScalePromotionInterval_ = KernelScheduleType::ScalePromotionInterval;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelScheduleType> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelScheduleType>);

  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert(IsFP8Input, "Warp Specialized gemm with FP8 BlockScaled Accumulator is only compatible with FP8 Blocked Scaled version right now.");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using ElementBlockScale = ElementAccumulator;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, KernelScheduleType> ||
                                        cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelScheduleType>;

  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);

  static constexpr int ScaleGranularityM = ScaleGranularityM_ == 0 ? size<0>(TileShape_MNK{}) : ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_ == 0 ? size<1>(TileShape_MNK{}) : ScaleGranularityN_;
  static constexpr int ScaleMsPerTile = size<0>(TileShape_MNK{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape_MNK{}) / ScaleGranularityN;
  static_assert((size<0>(TileShape_MNK{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape_MNK{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  static constexpr int PipelineStages = detail::compute_stage_count_with_blockwise_scale<detail::sm90_smem_capacity_bytes - KernelSmemCarveout,
      ElementAMma, ElementBMma, ElementBlockScale, TileShape_MNK, ScaleMsPerTile, ScaleNsPerTile>(StageCountType{});
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
    MainloopSm90ArrayTmaGmmaWarpSpecializedBlockScaling<PipelineStages, ClusterShape_MNK, KernelScheduleType, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>,
    MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/collective_builder.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/builders/sm90_gmma_builder.inl"
#include "cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl"
#if !defined(__CUDACC_RTC__) 
#include "cutlass/gemm/collective/builders/sm100_umma_builder.inl"              
#include "cutlass/gemm/collective/builders/sm100_9xBF16_umma_builder.inl"       
#include "cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl"  
#include "cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl"
#endif 

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/collective_mma.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/collective/collective_mma_decl.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/collective/sm70_mma_twostage.hpp"
#include "cutlass/gemm/collective/sm80_mma_multistage.hpp"
#include "cutlass/gemm/collective/sm90_mma_multistage_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_multistage_gmma_rs_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp" 
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"
#include "cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"

#if !defined(__CUDACC_RTC__)
#include "cutlass/gemm/collective/sm100_mma_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp"
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_emulated.hpp"
#include "cutlass/gemm/collective/sm100_mma_array_warpspecialized_emulated.hpp"

#include "cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp" 
#include "cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp" 
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp"
#endif // !defined(__CUDACC_RTC__) 
/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/fp8_accumulation.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cute/algorithm/clear.hpp"
#include "cute/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////
///////////////////////////////////FP8 Accumulation///////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// This class provides API to promote (add) or scale (multiply_add) the results 
/// from the tensor core accumulators to the main accumulators when the number 
/// of MMAs reaches the max number of MMA interval specified by user, after that
/// the tensor core accumulators are zeroed.
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

template <
    class EngineAccum,
    class LayoutAccum>
struct GmmaFP8Accumulation {  
  using TensorAccum = cute::Tensor<EngineAccum, LayoutAccum>;
  using ElementAccumulator = typename EngineAccum::value_type;

  static_assert(is_static<LayoutAccum>::value, "Accumulator Layout should be static");
  static_assert(is_rmem<TensorAccum>::value , "Accumulator tensor must be rmem resident.");

private:
  TensorAccum& accum_;
  TensorAccum accum_temp_;

  uint32_t accum_promotion_interval_;         // defines the max num of executed MMAs after which accum should be promoted.
  uint32_t mma_count_per_mainloop_iteration_; // num of MMAs per k_tile of mainloop
  uint32_t mma_count_;                        // current executed MMAs
  uint32_t reset_accum_flag_;                 // accum needs to be zeroed or not. 

  // promote or `add` the partial accumulators to main accumulator (FADD).
  CUTLASS_DEVICE
  void promote_core() {
    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i);
    }
  }

  // `multiply` scale the partial accumulators and `add` to main accumulator (FFMA).
  CUTLASS_DEVICE
  void scale_core(ElementAccumulator const &scale) {
    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scale;
    }
  }

  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_core(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    using TensorScale = cute::Tensor<EngineScale, LayoutScale>;

    static_assert(is_static<LayoutScale>::value, "Scale Layout should be static");
    static_assert(is_rmem<TensorScale>::value , "Scale tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScale{}.shape(), "Accumulator and scale must have same shape.");

    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scale(i);
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  CUTLASS_DEVICE
  void scale_core(const cute::Tensor<EngineScaleA, LayoutScaleA> &scaleA, const cute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    using TensorScaleA = cute::Tensor<EngineScaleA, LayoutScaleA>;
    using TensorScaleB = cute::Tensor<EngineScaleB, LayoutScaleB>;

    static_assert(is_static<LayoutScaleA>::value, "ScaleA Layout should be static");
    static_assert(is_static<LayoutScaleB>::value, "ScaleB Layout should be static");
    static_assert(is_rmem<TensorScaleA>::value, "ScaleA tensor must be rmem resident.");
    static_assert(is_rmem<TensorScaleB>::value, "ScaleB tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScaleA{}.shape(), "Accumulator and scaleA must have same shape.");
    static_assert(LayoutAccum{}.shape() == LayoutScaleB{}.shape(), "Accumulator and scaleB must have same shape.");

    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scaleA(i) * scaleB(i);
    }
  }

public:
  CUTLASS_DEVICE
  GmmaFP8Accumulation(
      TensorAccum &accum,
      uint32_t accum_promotion_interval,
      uint32_t mma_count_per_mainloop_iteration)
      : accum_(accum), 
        accum_promotion_interval_(accum_promotion_interval),
        mma_count_per_mainloop_iteration_(mma_count_per_mainloop_iteration),
        mma_count_(0), 
        reset_accum_flag_(0) 
  {
    accum_temp_ = cute::make_fragment_like(accum);
  }

  //
  // Methods (Common)
  //

  CUTLASS_DEVICE 
  TensorAccum& operator()() {
    return accum_temp_;
  }

  /// prepare the MMA accumulators when initialization or zeroing is required.
  CUTLASS_DEVICE
  bool prepare_if_needed() { 
    return reset_accum_flag_;
  }

  //
  // Methods (for FADD version)
  //

  /// promote (add) the results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void promote_if_needed() {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      promote_core();
      mma_count_ = 0;
    }
  }

  /// promote (add) the residue results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void promote_residue_if_needed() {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      promote_core();
    }
  }

  //
  // Methods (for FFMA version)
  //

  /// scale (multiply_add) the results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void scale_if_needed(ElementAccumulator const &scale) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      scale_core(scale);
      mma_count_ = 0;
    }
  }

  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_if_needed(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      scale_core(scale);
      mma_count_ = 0;
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  CUTLASS_DEVICE
  void scale_if_needed(const cute::Tensor<EngineScaleA, LayoutScaleA> &scaleA, const cute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      scale_core(scaleA, scaleB);
      mma_count_ = 0;
    }
  }
  
  /// scale (multiply_add) the results from the MMA accumulators to main accumulator without checking the counter.
  CUTLASS_DEVICE
  void scale(ElementAccumulator const &scale) {
    scale_core(scale);
  }

  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    scale_core(scale);
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  CUTLASS_DEVICE
  void scale(const cute::Tensor<EngineScaleA, LayoutScaleA> &scaleA, const cute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    scale_core(scaleA, scaleB);
  }

  /// scale (multiply_add) the residue results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void scale_residue_if_needed(ElementAccumulator const &scale) {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      scale_core(scale);
    }
  }

  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_residue_if_needed(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      scale_core(scale);
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  CUTLASS_DEVICE
  void scale_residue_if_needed(const cute::Tensor<EngineScaleA, LayoutScaleA> &scaleA, const cute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      scale_core(scaleA, scaleB);
    }
  }
};

} // namespace cutlass::gemm::collective
""",
    'key-files/include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/sm100_blockwise_scale_layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <
  int Stages,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,   // Static cluster shape or dynamic (int, int, _1)
  class TileShape_,     // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutSFA = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using ElementSFA = typename TiledMma::ValTypeC;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = cute::remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
  using LayoutSFB = cute::remove_cvref_t<decltype(get<1>(StridePairB_{}))>;
  using ElementSFB = typename TiledMma::ValTypeC;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static_assert(size<0>(TileShape{}) % ScaleGranularityM == 0 and ScaleGranularityM <= size<0>(TileShape{}), "Scale Granularity M must divide Tile Shape");

  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;
  static_assert(size<1>(TileShape{}) % ScaleGranularityN == 0 and ScaleGranularityN <= size<1>(TileShape{}), "Scale Granularity N must divide Tile Shape");

  static_assert(size<1, 0>(LayoutSFA{}) == size<1, 0>(LayoutSFB{}), "Vector size K must be equal for SFA and SFB");

  static constexpr int ScaleGranularityK = size<1, 0>(LayoutSFA{});
  static constexpr int ScaleKsPerTile = size<2>(TileShape{}) / ScaleGranularityK;
  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0 and ScaleGranularityK <= size<2>(TileShape{}), "Scale Granularity K must divide Tile Shape");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0, "Scale Granularity K must be divisible by MMA_K");

  static constexpr int K_BLOCK_MMAS_PER_SCALE_K = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});

  static constexpr int TILE_M = size<0>(TileShape{});
  static constexpr int TILE_N = size<1>(TileShape{});

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, 
      ScaleGranularityN, 
      ScaleGranularityK, 
      size<0,1>(LayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K,
      size<0,1>(LayoutSFB{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K>;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  static_assert(size<0>(AtomThrShapeMNK{}) == 1, "2SM MMA is not yet supported");

  static_assert(size<0>(CtaShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(CtaShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(CtaShape_MNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::Stages,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  using Load2TransformPipeline = cutlass::PipelineAsync<DispatchPolicy::Stages>;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using Mma2TransformPipeline = cutlass::PipelineUmmaAsync<
                                  AccumulatorPipelineStageCount,
                                  AtomThrShapeMNK>;
  using Mma2TransformPipelineState = typename Mma2TransformPipeline::PipelineState;

  // Two arrivals per CTA (1 arrival and 1 arrival through cp.async.mbarrier)
  static constexpr int NumLoad2TransformProducerThreadEvents = 2;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2 (M,K)");
  static_assert(((size<0,0>(MmaShapeA_MK{}) * size<1>(MmaShapeA_MK{})) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeA_MK{}) * size<2>(MmaShapeA_MK{})) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2 (N,K)");
  static_assert(((size<0,0>(MmaShapeB_NK{}) * size<1>(MmaShapeB_NK{})) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeB_NK{}) * size<2>(MmaShapeB_NK{})) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy - invalid TMA copy atom specified.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy -  invalid TMA copy atom specified.");

  using TmaInternalElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, cutlass::tfloat32_t, ElementAMma>;
  using TmaInternalElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, cutlass::tfloat32_t, ElementBMma>;

  using SmemAllocTypeA = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA, cute::UMMA::MXF8F6F4Format, void*>;
  using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB, cute::UMMA::MXF8F6F4Format, void*>;

  using SmemLayoutScaleA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutScaleB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  // Scaling gmem-to-smem copy atom 
  static constexpr int LeadingScalesPerTileSFA = size<0,1>(LayoutSFA{}.stride()) == 1 ? ScaleMsPerTile : ScaleKsPerTile;
  using ScaleCopyTypeA = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementAccumulator)) * LeadingScalesPerTileSFA, 16)>;
  using SmemScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ScaleCopyTypeA>, ElementAccumulator>;

  static constexpr int LeadingScalesPerTileSFB = size<0,1>(LayoutSFB{}.stride()) == 1 ? ScaleNsPerTile : ScaleKsPerTile;
  using ScaleCopyTypeB = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementAccumulator)) * LeadingScalesPerTileSFB, 16)>;
  using SmemScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ScaleCopyTypeB>, ElementAccumulator>;

  using TiledCopyScaleA = decltype(make_tiled_copy(SmemScalingCopyAtomA{}, Layout<Shape<_1>>{}, Layout<Shape<Int<LeadingScalesPerTileSFA>>>{}));
  using TiledCopyScaleB = decltype(make_tiled_copy(SmemScalingCopyAtomB{}, Layout<Shape<_1>>{}, Layout<Shape<Int<LeadingScalesPerTileSFB>>>{}));

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementAccumulator, cute::cosize_v<SmemLayoutScaleA>> smem_scale_A;
      cute::ArrayEngine<ElementAccumulator, cute::cosize_v<SmemLayoutScaleB>> smem_scale_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;

    using Load2TransformPipelineStorage = typename Load2TransformPipeline::SharedStorage;
    Load2TransformPipelineStorage transform2load_pipeline;

    using Mma2TransformPipelineStorage = typename Mma2TransformPipeline::SharedStorage;
    Mma2TransformPipelineStorage mma2transform_pipeline;
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
  using Mma2TransformPipelineStorage = typename SharedStorage::Mma2TransformPipelineStorage;
  using Load2TransformPipelineStorage = typename SharedStorage::Load2TransformPipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t TmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);

  template<class AccTensor>
  struct TmemStorage {
    AccTensor accumulators;
  };

  template<
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class GTensorPartitionedScaleA, class GTensorPartitionedScaleB,
    class IdentTensorPartitionedScaleA, class IdentTensorPartitionedScaleB,
    class STensorScaleA, class STensorScaleB
  >
  struct LoadParams {
    // for scheduler
    KTileCount k_tiles;
    // for input tensor values
    GTensorPartitionedA tAgA_mkl;
    GTensorPartitionedB tBgB_nkl;
    STensorA tAsA;
    STensorB tBsB;

    GTensorPartitionedScaleA tSFAgSFA_mkl;
    GTensorPartitionedScaleB tSFBgSFB_nkl;
    IdentTensorPartitionedScaleA tSFAIdentSFA_mkl;
    IdentTensorPartitionedScaleB tSFBIdentSFB_nkl;
    STensorScaleA tSFAsSFA;
    STensorScaleB tSFBsSFB;

    // the TMA multicast masks
    uint16_t mcast_mask_a;
    uint16_t mcast_mask_b;

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;

    CUTLASS_DEVICE
    LoadParams (
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_,
        STensorA tAsA_, STensorB tBsB_,
        GTensorPartitionedScaleA tSFAgSFA_mkl_, GTensorPartitionedScaleB tSFBgSFB_nkl_,
        IdentTensorPartitionedScaleA tSFAIdentSFA_mkl_, IdentTensorPartitionedScaleB tSFBIdentSFB_nkl_,
        STensorScaleA tSFAsSFA_, STensorScaleB tSFBsSFB_,
        uint16_t mcast_mask_a_, uint16_t mcast_mask_b_,
        LayoutSFA layout_SFA_, LayoutSFB layout_SFB_)
    : k_tiles(k_tiles_)
    , tAgA_mkl(tAgA_mkl_), tBgB_nkl(tBgB_nkl_)
    , tAsA(tAsA_), tBsB(tBsB_)
    , tSFAgSFA_mkl(tSFAgSFA_mkl_), tSFBgSFB_nkl(tSFBgSFB_nkl_)
    , tSFAIdentSFA_mkl(tSFAIdentSFA_mkl_), tSFBIdentSFB_nkl(tSFBIdentSFB_nkl_)
    , tSFAsSFA(tSFAsSFA_), tSFBsSFB(tSFBsSFB_)
    , mcast_mask_a(mcast_mask_a_), mcast_mask_b(mcast_mask_b_)
    , layout_SFA(layout_SFA_), layout_SFB(layout_SFB_) {}
  };

  template<class FragmentA, class FragmentB>
  struct MmaParams {
    TiledMma tiled_mma;
    FragmentA tCrA;
    FragmentB tCrB;

    CUTLASS_DEVICE
    MmaParams (
        TiledMma tiled_mma_,
        FragmentA tCrA_, FragmentB tCrB_)
    : tiled_mma(tiled_mma_)
    , tCrA(tCrA_), tCrB(tCrB_) {}
  };

  template<
    class STensorScaleA, class STensorScaleB
  >
  struct TransformParams {
    // for scheduler
 
    STensorScaleA sSFA;
    STensorScaleB sSFB;

    CUTLASS_DEVICE
    TransformParams (
        STensorScaleA sSFA_, STensorScaleB sSFB_)
    :  sSFA(sSFA_), sSFB(sSFB_) {}
  };


  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementAccumulator const* ptr_scale_A{nullptr};
    LayoutSFA layout_SFA{};
    ElementAccumulator const* ptr_scale_B{nullptr};
    LayoutSFB layout_SFB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_B = decltype(make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;

    ElementAccumulator const* ptr_scale_A;
    LayoutSFA layout_SFA;
    ElementAccumulator const* ptr_scale_B;
    LayoutSFB layout_SFB;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster)
    , runtime_data_type_a_(params.runtime_data_type_a)
    , runtime_data_type_b_(params.runtime_data_type_b) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x && 
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shape,
    Arguments const& args,
    [[maybe_unused]] void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    return {
      tma_load_a,
      tma_load_b,
      tma_load_a_fallback,
      tma_load_b_fallback,
      hw_info.cluster_shape_fallback,
      args.runtime_data_type_a,
      args.runtime_data_type_b,
      args.ptr_scale_A,
      args.layout_SFA,
      args.ptr_scale_B,
      args.layout_SFB
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});
    
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    bool implementable_sf = cutlass::detail::check_alignment<sizeof(ScaleCopyTypeA) / sizeof(ElementAccumulator)>(args.layout_SFA);
    implementable_sf = implementable_sf && cutlass::detail::check_alignment<sizeof(ScaleCopyTypeB) / sizeof(ElementAccumulator)>(args.layout_SFB);

    if (!implementable_sf) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for Scale Factors.\n");
    }

    return implementable && implementable_sf;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static
  auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));     // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  auto
  slice_accumulator(TmemStorage tmem_storage, int stage) {
    return cute::make_tuple(tmem_storage.accumulators(_,_,_,stage));
  }

  template<class EpilogueTile, bool IsOverlappingAccum = false>
  CUTLASS_DEVICE static
  auto
  init_tmem_tensors(EpilogueTile epi_tile) {
    TiledMma tiled_mma;
    auto acc_shape = partition_accumulator_shape();
    // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for mainloop and epilogue.
    Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
        tiled_mma, acc_shape, EpilogueTile{});
    TmemStorage<decltype(accumulators)> tmem_storage;
    tmem_storage.accumulators = accumulators;
    return tmem_storage;
  }

  template<class AccTensor>
  CUTLASS_DEVICE static
  void
  set_tmem_offsets(TmemStorage<AccTensor>& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
  }

  /// Set up the data needed by this collective for load.
  /// Return load params containing 
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// tSFAgSFA_mkl - partitioned gmem tensor for SFA
  /// tSFBgSFB_nkl - partitioned gmem tensor for SFB
  /// tSFAIdentSFA_mkl - partitioned identity tensor for SFA in gmem
  /// tSFBIdentSFB_nkl - partitioned identity tensor for SFB in gmem
  /// tSFAsSFA - partitioned smem tensor for SFA 
  /// tSFBsSFB - partitioned smem tensor for SFB
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  /// layout_SFA - layout of SFA in gmem
  /// layout_SFB - layout of SFB in gmem
  template <class ProblemShape_MNKL,
            class MainloopParams>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);                                       // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);                                       // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});      // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});      // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    // Scales

    Tensor mSFA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_A), mainloop_params.layout_SFA);    // (m,k,l)
    Tensor mSFB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_B), mainloop_params.layout_SFB);    // (n,k,l)

    Tensor SFA_mkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFA));

    Tensor SFB_nkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFB));

    // Tile the tensors and defer the slice
    Tensor gSFA_mkl = local_tile(mSFA_mkl, CtaShape_MNK{}, 
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, CtaShape_MNK{}, 
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    Tensor identSFA_mkl = local_tile(SFA_mkl_ident, CtaShape_MNK{}, 
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor identSFB_nkl = local_tile(SFB_nkl_ident, CtaShape_MNK{}, 
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    static_assert(rank(decltype(gSFA_mkl){}) == 5);
    static_assert(rank(decltype(gSFB_nkl){}) == 5);

    // 1 thread copies entire set of scalar
    TiledCopyScaleA scale_copy_a{};
    TiledCopyScaleB scale_copy_b{};

    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(_0{});
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(_0{});

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_scale_A.begin()), 
        SmemLayoutScaleA{});                                                                          // (CTA_M,CTA_K,P)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_scale_B.begin()), 
        SmemLayoutScaleB{});                                                                          // (CTA_M,CTA_K,P)

    Tensor tSFAgSFA_mkl = thr_scale_copy_a.partition_S(gSFA_mkl);                        // (CPY, BLK_M, BLK_K, m, k, l)
    Tensor tSFAIdentSFA_mkl = thr_scale_copy_a.partition_S(identSFA_mkl);                // (CPY, BLK_M, BLK_K, m, k, l)

    Tensor tSFAsSFA = thr_scale_copy_a.partition_D(sSFA);
    
    Tensor tSFBgSFB_nkl = thr_scale_copy_b.partition_S(gSFB_nkl);                        // (CPY, BLK_N, BLK_K, m, k, l)
    Tensor tSFBIdentSFB_nkl = thr_scale_copy_b.partition_S(identSFB_nkl);                // (CPY, BLK_N, BLK_K, m, k, l)
    Tensor tSFBsSFB = thr_scale_copy_b.partition_D(sSFB);

    static_assert(rank(decltype(tSFAgSFA_mkl){}) == 6);
    static_assert(rank(decltype(tSFBgSFB_nkl){}) == 6);

    LoadParams load_params {
      shape<3>(gA_mkl),                               // for scheduler
      tAgA_mkl, tBgB_nkl, tAsA, tBsB,                 // for input tensor values
      tSFAgSFA_mkl, tSFBgSFB_nkl,                     // for input scale tensor values
      tSFAIdentSFA_mkl, tSFBIdentSFB_nkl,             // for predicating scale tensor copies
      tSFAsSFA, tSFBsSFB,                             // for scale tensor values
      mcast_mask_a, mcast_mask_b,                     // multicast masks
      mainloop_params.layout_SFA,                     // for predicating scale tensor copies
      mainloop_params.layout_SFB                      // for predicating scale tensor copies
    };
    return load_params;
  }

  /// Set up the data needed by this collective for mma compute.
  template <class AccTensor>
  CUTLASS_DEVICE auto
  mma_init(
      [[maybe_unused]] TmemStorage<AccTensor> tmem_tensors,
      TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = TiledMma::make_fragment_A(sA);                                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = TiledMma::make_fragment_B(sB);                                               // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                          // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    TiledMma tiled_mma;

    if constexpr (IsRuntimeDataType) {
      // Update instruction descriptor according to runtime argument.
      // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
      tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
    }
    MmaParams<decltype(tCrA), decltype(tCrB)> mma_params {
      tiled_mma,
      tCrA, tCrB
    };
    return mma_params;
  }

  /// Set up the data needed by this collective for transform.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  transform_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_A.begin()), 
        SmemLayoutScaleA{});                                                        // (ScaleMsPerTile,ScakeKsPerTile,P)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_B.begin()), 
        SmemLayoutScaleB{});                                                        // (ScaleNsPerTile,ScaleKsPerTile,P)


    TransformParams transform_params {
      sSFA, sSFB                        // for input tensor values
    };
    return transform_params;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class LoadParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
      MainloopPipeline mainloop_pipeline,
      Load2TransformPipeline load2transform_pipeline,
      MainloopPipelineState mainloop_pipe_producer_state, 
      Load2TransformPipelineState load2transform_pipe_producer_state,
      LoadParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_k_tiles,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          tSFAgSFA_mkl, tSFBgSFB_nkl, 
          tSFAIdentSFA_mkl, tSFBIdentSFB_nkl,
          tSFAsSFA, tSFBsSFB,
          mcast_mask_a, mcast_mask_b,
          layout_SFA, layout_SFB] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    TiledCopyScaleA scale_copy_a{};
    TiledCopyScaleB scale_copy_b{};

    Tensor tSFAgSFA = tSFAgSFA_mkl(_, _, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor tSFBgSFB = tSFBgSFB_nkl(_, _, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor thr_tile_SFA_k = tSFAIdentSFA_mkl(_0{}, _, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    Tensor thr_tile_pSFA = make_tensor<bool>(shape(filter_zeros(thr_tile_SFA_k(_,_,_0{}), tSFAgSFA(_0{},_,_,_0{}).stride())));
    Tensor thr_tile_SFB_k = tSFBIdentSFB_nkl(_0{}, _, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor thr_tile_pSFB = make_tensor<bool>(shape(filter_zeros(thr_tile_SFB_k(_,_,_0{}), tSFBgSFB(_0{},_,_,_0{}).stride())));

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
      
      load2transform_pipeline.producer_acquire(load2transform_pipe_producer_state);

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      auto curr_mainloop_pipe_producer_state = mainloop_pipe_producer_state;
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFA); ++i) {
        Tensor thr_tile_SFA = filter_zeros(thr_tile_SFA_k(_,_,*k_tile_iter), tSFAgSFA(_0{},_,_,_0{}).stride()); 
        thr_tile_pSFA(i) = elem_less(thr_tile_SFA(i), shape(filter_zeros(layout_SFA)));
      }
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFB); ++i) {
        Tensor thr_tile_SFB = filter_zeros(thr_tile_SFB_k(_,_,*k_tile_iter), tSFBgSFB(_0{},_,_,_0{}).stride()); 
        thr_tile_pSFB(i) = elem_less(thr_tile_SFB(i), shape(filter_zeros(layout_SFB)));
      }

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
        copy_if(scale_copy_a, thr_tile_pSFA, filter_zeros(tSFAgSFA(_,_,_,*k_tile_iter)), filter_zeros(tSFAsSFA(_,_,_,load2transform_pipe_producer_state.index())));
        copy_if(scale_copy_b, thr_tile_pSFB, filter_zeros(tSFBgSFB(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,load2transform_pipe_producer_state.index())));
        load2transform_pipeline.producer_commit(load2transform_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive_noinc);        
      }

      __syncwarp();

      ++load2transform_pipe_producer_state;
      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, load2transform_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline mainloop_pipeline, 
      Load2TransformPipeline load2transform_pipeline, 
      MainloopPipelineState mainloop_pipe_producer_state,
      Load2TransformPipelineState load2transform_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
    load2transform_pipeline.producer_tail(load2transform_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class TmemStorage,
    class MmaParams,
    class CtaTileCoord
  >
  CUTLASS_DEVICE auto
  mma(
      cute::tuple<MainloopPipeline,
                  Mma2TransformPipeline> pipelines,
      cute::tuple<MainloopPipelineState,
                  Mma2TransformPipelineState> pipeline_states,
      TmemStorage tmem_storage,
      MmaParams const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count) {
    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    auto [mainloop_pipeline, 
          mma2transform_pipeline] = pipelines;

    auto [mainloop_pipe_consumer_state, 
          mma2transform_pipe_producer_state] = pipeline_states;

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on mainloop_pipe_consumer_state until its data are available
      // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);

      // Compute on k_tile
      int read_stage = mainloop_pipe_consumer_state.index();
      // Save current mainlop pipeline read state
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      // Advance mainloop_pipe
      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      // Peek at next iteration
      barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

      static_assert(size<2>(tCrA) / K_BLOCK_MMAS_PER_SCALE_K, "k blocks must be divisible by K_BLOCK_MMAS_PER_SCALE_K");

      CUTLASS_PRAGMA_UNROLL
      for (int scale_k_blocks = 0; scale_k_blocks < size<2>(tCrA) / K_BLOCK_MMAS_PER_SCALE_K; ++scale_k_blocks) {
        mma2transform_pipeline.producer_acquire(mma2transform_pipe_producer_state);

        auto acc = get<0>(slice_accumulator(tmem_storage, mma2transform_pipe_producer_state.index()));
        static_assert(is_tmem<remove_cvref_t<decltype(acc)>>::value, "Accumulator must be tmem resident.");
        static_assert(rank(remove_cvref_t<decltype(acc)>{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        // for each set of scale_k_blocks we zero the accumulator
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        int start_k_block = scale_k_blocks * size<2>(tCrA) / K_BLOCK_MMAS_PER_SCALE_K;
        // Unroll the K mode manually so we can set scale C to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block_offset = 0; k_block_offset < K_BLOCK_MMAS_PER_SCALE_K; ++k_block_offset) {
          int k_block = start_k_block + k_block_offset;
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma,
                     tCrA(_,_,k_block,read_stage),
                     tCrB(_,_,k_block,read_stage),
                     acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        mma2transform_pipeline.producer_commit(mma2transform_pipe_producer_state);
        ++mma2transform_pipe_producer_state;
      }
      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);

    }

    return make_tuple(mainloop_pipe_consumer_state, mma2transform_pipe_producer_state);
  }

  /// Transform
  template <
    class TransformParams,
    class TmemStorage,
    class CtaTileCoord,
    class CopyOpT2R,
    class EpilogueTile
  >
  CUTLASS_DEVICE auto 
  transform(
      cute::tuple<Mma2TransformPipeline, Load2TransformPipeline> pipelines,
      cute::tuple<Mma2TransformPipelineState, Load2TransformPipelineState> consumer_states,
      TmemStorage tmem_storage,
      TransformParams const& transform_inputs,
      CtaTileCoord cta_tile_coord,
      CopyOpT2R,
      EpilogueTile,
      int k_tile_count) {

    static_assert(size<0>(EpilogueTile{}) <= size<0>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");
    static_assert(size<1>(EpilogueTile{}) <= size<1>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");


    //
    // PIPELINED Transform
    //
    
    Tensor acc = get<0>(slice_accumulator(tmem_storage, _0{}));

    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});

    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Append N with a stride of 0 to SFA
    Tensor sSFA_ = transform_inputs.sSFA;
    Tensor sSFA = make_tensor(sSFA_.data(), make_layout(
      make_shape(get<0>(sSFA_.shape()), get<1>(CtaShape_MNK{}), get<1>(sSFA_.shape()), get<2>(sSFA_.shape())),
      make_stride(get<0>(sSFA_.stride()), _0{}, get<1>(sSFA_.stride()), get<2>(sSFA_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFA) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFA) == size<1>(tAcc));

    Tensor sSFA_epi = flat_divide(sSFA, EpilogueTile{});

    // Append M with a stride of 0 to SFB
    Tensor sSFB_ = transform_inputs.sSFB;
    Tensor sSFB = make_tensor(sSFB_.data(), make_layout(
      make_shape(get<0>(CtaShape_MNK{}), get<0>(sSFB_.shape()), get<1>(sSFB_.shape()), get<2>(sSFB_.shape())),
      make_stride(_0{}, get<0>(sSFB_.stride()), get<1>(sSFB_.stride()), get<2>(sSFB_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFB) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFB) == size<1>(tAcc));

    Tensor sSFB_epi = flat_divide(sSFB, EpilogueTile{});

    TiledCopy tiled_t2r_epi = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_,_,_0{},_0{}));

    int thread_idx = threadIdx.x % size(tiled_t2r_epi);

    ThrCopy thread_t2r_epi = tiled_t2r_epi.get_slice(thread_idx);   

    Tensor acc_ident_epi = make_identity_tensor(shape(tAcc_epi));
    
    Tensor tTR_rAcc_epi = thread_t2r_epi.partition_D(acc_ident_epi);                // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

    Tensor tTR_sSFA_epi = thread_t2r_epi.partition_D(sSFA_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    Tensor tTR_sSFB_epi = thread_t2r_epi.partition_D(sSFB_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    
    static_assert(rank(decltype(tTR_sSFA_epi){}) == 7);

    Tensor tTR_FullAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi));
    Tensor tTR_PartAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));

    Tensor tTR_rSFA_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,_0{})));
    Tensor tTR_rSFB_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,_0{})));

    Layout tTR_rSFA_layout = make_layout(tTR_sSFA_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFA_compact.stride());
    Layout tTR_rSFB_layout = make_layout(tTR_sSFB_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFB_compact.stride());

    // Zero our accumulator
    clear(tTR_FullAcc);

    auto [mma2transform_pipeline, load2transform_pipeline] = pipelines;
    auto [mma2transform_pipe_state, load2transform_pipe_state] = consumer_states;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      load2transform_pipeline.consumer_wait(load2transform_pipe_state);
      int read_idx = load2transform_pipe_state.index();

      copy(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,read_idx)), tTR_rSFA_compact);
      copy(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,read_idx)), tTR_rSFB_compact);

      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFA_layout) == size(tTR_rSFA_compact));
      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFB_layout) == size(tTR_rSFB_compact));
      
      Tensor tTR_rSFA = make_tensor(tTR_rSFA_compact.data(), tTR_rSFA_layout);
      Tensor tTR_rSFB = make_tensor(tTR_rSFB_compact.data(), tTR_rSFB_layout);
      
      load2transform_pipeline.consumer_release(load2transform_pipe_state);
      ++load2transform_pipe_state;

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < ScaleKsPerTile; ++k_block) {

        mma2transform_pipeline.consumer_wait(mma2transform_pipe_state);

        Tensor acc = get<0>(slice_accumulator(tmem_storage, mma2transform_pipe_state.index()));
        Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
        Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                   // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        Tensor tTR_tAcc = thread_t2r_epi.partition_S(tAcc_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

        CUTLASS_PRAGMA_UNROLL
        for (int epi_m = 0; epi_m < size<2>(tAcc_epi); ++epi_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int epi_n = 0; epi_n < size<3>(tAcc_epi); ++epi_n) {

            auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
            auto scale_b = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);

            Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
            // Compute tmem load predication if necessary
            copy(tiled_t2r_epi, tTR_tAcc(_,_,_,epi_m,epi_n), tTR_PartAcc);
            cutlass::arch::fence_view_async_tmem_load();
 
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(full_acc); ++i) {
              ElementAccumulator scale = scale_a(i) * scale_b(i);
              full_acc(i) += scale * tTR_PartAcc(i);
            }
          }
        }              
        cutlass::arch::fence_view_async_tmem_load();
        mma2transform_pipeline.consumer_release(mma2transform_pipe_state);
        // release acc
        ++mma2transform_pipe_state;
      } 

      --k_tile_count;
    }

    return cute::make_tuple(tTR_FullAcc, tiled_t2r_epi, cute::make_tuple(mma2transform_pipe_state, load2transform_pipe_state));
 }

protected:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};

  RuntimeDataTypeA runtime_data_type_a_{};
  RuntimeDataTypeB runtime_data_type_b_{};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule_,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
public:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };
  
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

private:
  template<class T> friend struct detail::MixedInputUtils;
  using CollectiveType = CollectiveMma<DispatchPolicy, TileShape_, 
                                       ElementAOptionalTuple, StrideA_, 
                                       ElementBOptionalTuple, StrideB_,
                                       TiledMma_, 
                                       GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_,
                                       TransformA_,
                                       GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,
                                       TransformB_>;
  using Utils = detail::MixedInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

public:
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^ cute::is_tuple<ElementBOptionalTuple>::value, 
    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale], [ElementZero]}. Inputs in [] are optional.");
  
  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  
  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert(( IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) || 
                (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value) && 
                 (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)), 
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(), 
    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");
  
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = cute::SM90_TMA_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA  = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB  = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
  using TmaElementScale = uint_bit_t<sizeof_bits_v<NonVoidElementScale> >; // in case we have array. translating to uint to satisfy tma descriptor's specialization

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  using SmemLayoutAtomScale = Layout<Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<1>>>;
  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomA{}, select<0,2>(TileShape{}), InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(SwappedSmemLayoutAtomB{}, select<1,2>(TileShape{}), InternalSwappedStrideB{}));
  
  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{}, 
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{}),
    cute::conditional_t< ::cutlass::gemm::detail::is_major<0,NonVoidStrideScale>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for every main loop iteration.
  // We must also handle updating the pipeline transaction bytes on the fly.
  static_assert(size<1>(SmemLayoutAtomScale{}) == 1, "size<1>(SmemLayoutAtomScale) must be 1.");

private:
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }  

public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool UseScaleLookupTable = KernelConversionMode == ConversionMode::ConvertAndScale &&
                                              cutlass::detail::is_Array_v<ElementScale>;
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{}); 
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct SharedStorage {
    static constexpr int scale_elements = Utils::elements_per_smem_scale();
    static constexpr int zero_elements = Utils::elements_per_smem_zero();
    struct TensorStorage {
      CUTE_ALIGNAS(SmemAlignmentA) cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      CUTE_ALIGNAS(SmemAlignmentB) cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
      cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    } tensors;

    struct TensorMapStorage {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
      cute::TmaDescriptor smem_tensormap_scale;
      cute::TmaDescriptor smem_tensormap_zero;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScale const** ptr_S = nullptr;
    NonVoidStrideScale const* dS{};
    int chunk_size = 0;
    ElementZero const** ptr_Z = nullptr;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using LayoutA = decltype(detail::get_gmem_layout(repeat_like(InternalSwappedStrideA{}, int32_t(0)), InternalSwappedStrideA{}));
    using LayoutB = decltype(detail::get_gmem_layout(repeat_like(InternalSwappedStrideB{}, int32_t(0)), InternalSwappedStrideB{}));

    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementA const*>(nullptr)), LayoutA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)), LayoutB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    using TMA_Scale = decltype(make_tma_copy<TmaElementScale>(
        GmemTiledCopyScale{},
        make_tensor(detail::get_logical_ptr(static_cast<NonVoidElementScale const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),
        SmemLayoutScale{}(_,_,cute::Int<0>{}),
        ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel

   using TMA_Zero = decltype(make_tma_copy(
        GmemTiledCopyScale{},
        make_tensor(detail::get_logical_ptr(static_cast<NonVoidElementZero const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),
        SmemLayoutScale{}(_,_,cute::Int<0>{}),
        ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel
    
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    TMA_Scale tma_load_scale;
    TMA_Zero tma_load_zero;
    void* tensormaps;
    SwappedElementA const** ptr_A;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    NonVoidElementScale const** ptr_S;
    NonVoidStrideScale const* dS;
    NonVoidElementZero const** ptr_Z;
    int64_t scale_k;
    int chunk_size;
    int reload_factor = (chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {

    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    SwappedElementA const* ptr_A_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_A);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_B);
    }
    else {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_B);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_A);
    }

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      }
      else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(
          transform_leaf(dA.shape(), [](auto x){ 
            if constexpr (not is_static_v<decltype(x)>) {
              return static_cast<decltype(x)>(1);
            } else {
              return x;
            }
          }),
          dA.stride());
      }
      dB = InternalSwappedStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      }
      else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch, detail::get_gmem_layout(make_shape(init_M,init_K,mock_L), dA));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, detail::get_gmem_layout(make_shape(init_N,init_K,mock_L), dB));

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    typename Params::TMA_Scale tma_load_scale{};
    typename Params::TMA_Zero tma_load_zero{};

    void* tensormaps = workspace;
    auto args_setup = [&](auto ptr_A, auto ptr_B, int64_t scale_k = 0, int chunk_size = 0, int reload_factor = 1) -> Params {
      return {
          tma_load_a,
          tma_load_b,
          TmaTransactionBytes,
          tma_load_scale,
          tma_load_zero,
          tensormaps,
          reinterpret_cast<SwappedElementA const**>(ptr_A),
          ptr_dA,
          reinterpret_cast<SwappedElementB const**>(ptr_B),
          ptr_dB,
          reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),
          args.dS,
          reinterpret_cast<NonVoidElementZero const**>(args.ptr_Z),
          scale_k,
          chunk_size,
          reload_factor,
          dA,
          dB
      };
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A)
                    : args_setup(args.ptr_A, args.ptr_B);
    }
    else if constexpr (ModeHasScales) {
      auto scale_k = 1;
      ElementScale const* ptr_S = reinterpret_cast<ElementScale const*>(args.ptr_S);
      StrideScale dS{};
      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S), make_layout(make_shape(init_M,scale_k,mock_L), dS));
      tma_load_scale = make_tma_copy<TmaElementScale>(
          GmemTiledCopyScale{},
          tensor_scale,
          SmemLayoutScale{}(_,_,cute::Int<0>{}),
          ScaleTileShape{},
          _1{}); // mcast along N mode for this M load, if any

      if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale) {
        return SwapAB ? args_setup(args.ptr_B, args.ptr_A, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))
                      : args_setup(args.ptr_A, args.ptr_B, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));
      }
      else if constexpr(KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        ElementZero const* ptr_Z = reinterpret_cast<ElementZero const*>(args.ptr_Z);
        Tensor tensor_zero = make_tensor(detail::get_logical_ptr(ptr_Z), make_layout(make_shape(init_M,scale_k,mock_L), dS));
        tma_load_zero = make_tma_copy(
            GmemTiledCopyScale{},
            tensor_zero,
            SmemLayoutScale{}(_,_,cute::Int<0>{}),
            ScaleTileShape{},
            _1{}); // mcast along N mode for this M load, if any
        return SwapAB ? args_setup(args.ptr_B, args.ptr_A, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}))
                      : args_setup(args.ptr_A, args.ptr_B, scale_k, args.chunk_size, (args.chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{}));
        
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
    } 
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);

    // Calculating workspace size
    auto calculate_workspace_size = [SizeOfCuTensorMap, sm_count](uint32_t num_input_tensors) {
        return num_input_tensors * SizeOfCuTensorMap * sm_count;
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
      return calculate_workspace_size(2);
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies, followed by scale tensormap copies
      return calculate_workspace_size(3);
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies, followed by scale and zeros tensormap copies
      return calculate_workspace_size(4);
    } 
    else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in get_workspace_size.");
    }
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }


  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        auto get_stride = [](auto stride) {
          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {
            return *stride;
          }
          else {
            return stride;
          }
        };
        auto dA = get_stride(args.dA);
        auto dB = get_stride(args.dB);
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(detail::get_gmem_layout(cute::make_shape(M,K,L), dA));
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(detail::get_gmem_layout(cute::make_shape(N,K,L), dB));
        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
          implementable = implementable && (args.ptr_S == nullptr);
          implementable = implementable && (args.ptr_Z == nullptr);
        }
        else if constexpr (ModeHasScales) {
          const int scale_mn = SwapAB ? N : M;
          const int scale_k = (K + args.chunk_size - 1) / args.chunk_size;
          constexpr int min_tma_aligned_elements_scale = tma_alignment_bits / cutlass::sizeof_bits<ElementScale>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});
          implementable = implementable && (args.chunk_size == K || ((args.chunk_size % size<2>(TileShape{})) == 0));
          implementable = implementable && args.chunk_size != 0;
          implementable = implementable && (args.ptr_S != nullptr);
          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
            implementable = implementable && (args.ptr_Z == nullptr);
          }
          else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            constexpr int min_tma_aligned_elements_zero = tma_alignment_bits / cutlass::sizeof_bits<ElementZero>::value;
            implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_zero>(cute::make_shape(scale_mn,scale_k,L), StrideScale{});
            implementable = implementable && (args.ptr_Z != nullptr);
          } 
          else {
            static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
          }
        }
        else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
        }
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
  static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
  static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(M,K,mock_L), mainloop_params.dA))); // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(shape(detail::get_gmem_layout(make_shape(N,K,mock_L), mainloop_params.dB))); // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA_mkl, gB_nkl);
    } 
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor mS_mkl = mainloop_params.tma_load_scale.get_tma_tensor(make_shape(M,scale_k,L));      // (m,scale_k,l)
      Tensor gS_mkl = local_tile(mS_mkl, ScaleTileShape{}, make_coord(_,_));       // (BLK_M,BLK_Scale_K,m,scale_k,l)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_mkl = mainloop_params.tma_load_zero.get_tma_tensor(make_shape(M,scale_k,L));      // (m,scale_k,l)
        Tensor gZ_mkl = local_tile(mZ_mkl, ScaleTileShape{}, make_coord(_,_));      // (BLK_M,BLK_Scale_K,m,scale_k,l)
        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl, gZ_mkl);
      }
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
      }
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
    }
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class... Ts,
    class... TMs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<Ts...> const& load_inputs,
      cute::tuple<TMs...> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      static_assert(sizeof... (Ts) == 2, "Direct convert needs two inputs");
      static_assert(sizeof... (TMs) == 2, "Direct convert needs two tensormaps");
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      static_assert(sizeof... (Ts) == 3, "Scaled convert needs three inputs");
      static_assert(sizeof... (TMs) == 3, "Scaled convert needs three tensormaps");
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(sizeof... (Ts) == 4, "Scaled and zero convert needs four inputs");
      static_assert(sizeof... (TMs) == 4, "Scaled and zero convert needs four tensormaps");
    } 
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in TMA load.");
    }

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
    Tensor sA  = as_position_independent_swizzle_tensor(sA_);                                       // (BLK_M,BLK_K,PIPE)
    Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                       // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    uint16_t mcast_mask_s = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    auto extra_input_partitions = Utils::partition_extra_tma_inputs(mainloop_params, load_inputs, shared_tensors, cluster_local_block_id, m_coord, l_coord);

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // Nothing extra to do.
      }
      else if constexpr (ModeHasScales) {
        auto tSgS = get<0>(extra_input_partitions);
        auto tSsS = get<1>(extra_input_partitions);

        // Temporary factor which will determine which k tile to reload from gmem. Needed so we don't modify tma transaction bytes
        // on the fly.
        // We must do a ceiling divide here to correctly handle with chunk_size == K. In that case, we don't require that K
        // is a multiple of the threadblock tile K
        const int scale_load_k = *k_tile_iter / mainloop_params.reload_factor; // This will always be 0 when chunk_size == K.
        if (cute::elect_one_sync()) {
          copy(mainloop_params.tma_load_scale.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_s), tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));
        }

        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } 
        else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tZgZ = get<2>(extra_input_partitions);
          auto tZsZ = get<3>(extra_input_partitions);
          if (cute::elect_one_sync()) {
            copy(mainloop_params.tma_load_zero.with(get<3>(input_tensormaps), *tma_barrier, mcast_mask_s), tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,write_stage));
          }
        }
        else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");
        } 
      } 
      else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all 
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was 
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SwappedSmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    
    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);                                      // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)


    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&]{
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      }
      else {
        // Make register tensor matching smem layout, converter will take care of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_,_,_,Int<0>{}));
      }
    }();
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);                                        // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                                  // (CPY,CPY_M,CPY_K)

    // Partition of thread -> shared and thread -> RF
    auto partitioned_extra_info = Utils::partition_extra_mma_info(mma_thread_slice, shared_tensors);
    auto copy_partitions_extra_info = Utils::retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE
    
    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);
    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    // first k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      // copy smem->rmem for A operand

      Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
        partitioned_extra_info, copy_partitions_extra_info, 0, read_stage);
      if (K_BLOCK_MAX > 1) {
        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
          partitioned_extra_info, copy_partitions_extra_info, 1, read_stage);
      }
      
      Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);
      
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();

        if (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);
        }
        if (k_block < K_BLOCK_MAX - 1) {
          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
      }     

      --k_tile_count;
      if (k_tile_count > 0) {
        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite the A registers for the first mma. 
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
        
        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());
        
        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
          partitioned_extra_info, copy_partitions_extra_info, 1, smem_pipe_read.index());
        
        warpgroup_wait<K_WAIT_MAX>(); 
        Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      warpgroup_fence_operand(accum);
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();

        warpgroup_wait<K_WAIT_MAX>(); // We have K_BLOCK_MAX - 1 GMMA instructions pending for this stage, so we can release prior barrier
        if (k_block == K_BLOCK_MAX - 1) {
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if (k_block == 0) {
          barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        }

        if (k_block == K_BLOCK_MAX - 1) { 
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());

          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, 1, smem_pipe_read.index());
          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, 0);
        } 
        else {
          if (k_block < K_BLOCK_MAX - 2) {
            Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
              partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);
          }
          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
      }
      warpgroup_fence_operand(accum);

    }

    warpgroup_fence_operand(accum);

    {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accum);
      
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();

        warpgroup_wait<K_WAIT_MAX>();
        if (k_block == K_BLOCK_MAX - 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, k_block + 2, read_stage);
        }
        if (k_block < K_BLOCK_MAX - 1) {
          Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
      }
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    cute::TmaDescriptor* tma_desc_scale = &gmem_tensormap[sm_idx + 2*sm_count];
    cute::TmaDescriptor* tma_desc_zero = &gmem_tensormap[sm_idx + 3*sm_count];

    // Bringing tensormaps from params to smem for modification later
    Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
    Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

    if (cute::elect_one_sync()) {
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      Tensor pS_tensormap = make_tensor(mainloop_params.tma_load_scale.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sS_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_scale), Int<1>{}, Int<1>{});
      if (cute::elect_one_sync()) {
        copy(recast<uint128_t>(pS_tensormap), recast<uint128_t>(sS_tensormap));
      }
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      Tensor pZ_tensormap = make_tensor(mainloop_params.tma_load_zero.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sZ_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_zero), Int<1>{}, Int<1>{});
      if (cute::elect_one_sync()) {
        copy(recast<uint128_t>(pZ_tensormap), recast<uint128_t>(sZ_tensormap));
      }
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_init.");
    }

    __syncwarp();

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(tma_desc_a, tma_desc_b);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale, tma_desc_zero);
    }
    else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_init.");
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_scale,
                                                    mainloop_params.ptr_S[next_batch]);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_zero,
                                                    mainloop_params.ptr_Z[next_batch]);
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_address.");
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_scale  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_scale = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_zero   = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_zero  = {0,0,0,0,0};

    SwappedElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, detail::get_gmem_layout(make_shape(M,K,Int<1>{}), mainloop_params.ptr_dA[next_group]));

    SwappedElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, detail::get_gmem_layout(make_shape(N,K,Int<1>{}), mainloop_params.ptr_dB[next_group]));

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, 
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, 
                                             prob_shape_B, prob_stride_B);

    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      NonVoidElementScale const* ptr_S = nullptr;
      auto scale_k = 1;
      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S), make_shape(M,scale_k,Int<1>{}), mainloop_params.dS[next_group]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_scale, tensor_scale, 
                                             prob_shape_scale, prob_stride_scale);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      ElementZero const* ptr_Z = nullptr;
      auto scale_k = 1;
      Tensor tensor_zero = make_tensor(detail::get_logical_ptr(ptr_Z), make_shape(M,scale_k,Int<1>{}), mainloop_params.dS[next_group]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_zero, tensor_zero, 
                                               prob_shape_zero, prob_stride_zero);
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_tensor_properties.");
    }

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;
    }
    for (uint64_t& stride : prob_stride_scale) {
      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;
    }
    for (uint64_t& stride : prob_stride_zero) {
      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;
    }


    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);

    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_scale,
                                                            prob_shape_scale,
                                                            prob_stride_scale);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_zero,
                                                            prob_shape_zero,
                                                            prob_stride_zero);
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_replace_global_tensor_properties.");
    } 
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TMs...> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TMs...> const& input_tensormaps) {
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      tma_descriptor_cp_fence_release(get<2>(input_tensormaps), shared_tensormaps.smem_tensormap_scale);
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      tma_descriptor_cp_fence_release(get<3>(input_tensormaps), shared_tensormaps.smem_tensormap_zero);
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_cp_fence_release.");
    }
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
    }
    else if constexpr (KernelConversionMode != ConversionMode::DirectConvert){
      static_assert(cutlass::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in tensormaps_fence_acquire.");
    }
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

  static constexpr int NumProducerThreadEvents = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  // Assumption: StrideA is congruent with Problem_MK
  using TMA_A = decltype(make_tma_copy(
      GmemTiledCopyA{},
      make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
      SmemLayoutA{}(_,_,cute::Int<0>{}),
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
      size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
  // Assumption: StrideB is congruent with Problem_NK
  using TMA_B = decltype(make_tma_copy(
      GmemTiledCopyB{},
      make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
      SmemLayoutB{}(_,_,cute::Int<0>{}),
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  struct Params {
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    InternalElementA const** ptr_A;
    StrideA dA;
    InternalElementB const** ptr_B;
    StrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    InternalElementA const* ptr_A_first_batch = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    InternalElementB const* ptr_B_first_batch = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,mock_L), stride_a));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,mock_L), stride_b));
    TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    void* tensormaps = workspace;

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes,
      tensormaps,
      reinterpret_cast<InternalElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<InternalElementB const**>(args.ptr_B),
      args.dB
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytes =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,mock_L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,mock_L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorMapA, class TensorMapB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all 
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was 
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and 
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    assert(k_tile_count >= 1);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum); // (V,M,K) x (V,N,K) => (V,M,N)
      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum); // (V,M,K) x (V,N,K) => (V,M,N)
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    InternalElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);

    InternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, 
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, 
                                             prob_shape_B, prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<InternalElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<InternalElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  template <class TensorMapA, class TensorMapB, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/fp8_accumulation.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/tensor.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedFP8<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedFP8<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // Assumption: StrideA is congruent with Problem_MK
  using TMA_A = decltype(make_tma_copy(
      GmemTiledCopyA{},
      make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
      SmemLayoutA{}(_,_,cute::Int<0>{}),
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
      size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
  // Assumption: StrideB is congruent with Problem_NK
  using TMA_B = decltype(make_tma_copy(
      GmemTiledCopyB{},
      make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
      SmemLayoutB{}(_,_,cute::Int<0>{}),
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t mma_promotion_interval = 4;
    void* tensormaps;
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {
        // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
        // These will be replaced with correct values before the initial tma load.
        auto init_shape = repeat_like(append<4>(typename ProblemShape::UnderlyingProblemShape{}, 1), int32_t(1));
        auto init_M = get<0>(init_shape);
        auto init_N = get<1>(init_shape);
        auto init_K = get<2>(init_shape);
        auto init_L = get<3>(init_shape);

    ElementA const* ptr_A_first_batch = reinterpret_cast<ElementA const*>(args.ptr_A);
    ElementB const* ptr_B_first_batch = reinterpret_cast<ElementB const*>(args.ptr_B);

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,init_L), stride_a));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,init_L), stride_b));
    TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    void* tensormaps = workspace;

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes,
      args.mma_promotion_interval,
      tensormaps,
      reinterpret_cast<ElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<ElementB const**>(args.ptr_B),
      args.dB
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytes =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,mock_L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,mock_L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorMapA, class TensorMapB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and 
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    GmmaFP8Accumulation accumulation(accum, mainloop_params.mma_promotion_interval, size<2>(tCrA));
    warpgroup_fence_operand(accumulation());
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      if (accumulation.prepare_if_needed()) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      accumulation.promote_if_needed();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      if (accumulation.prepare_if_needed()) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accumulation());

      accumulation.promote_if_needed();

      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    accumulation.promote_residue_if_needed();

    warpgroup_fence_operand(accumulation());
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    ElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);

    ElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, 
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, 
                                             prob_shape_B, prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<ElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<ElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  template <class TensorMapA, class TensorMapB, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  int ScaleGranularityM_,
  int ScaleGranularityN_,
  int ScalePromotionInterval_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedBlockScaling<Stages, ClusterShape, KernelSchedule, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedBlockScaling<Stages, ClusterShape, KernelSchedule, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementBlockScale = ElementAccumulator;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

  static constexpr int NumProducerThreadEvents = 2;

  static constexpr int ScaleGranularityM = ScaleGranularityM_ == 0 ? size<0>(TileShape{}) : ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_ == 0 ? size<1>(TileShape{}) : ScaleGranularityN_;
  static constexpr int ScalePromotionInterval = ScalePromotionInterval_;
  static_assert(ScalePromotionInterval % 4 == 0, "ScalePromotionInterval must be a multiple of 4.");

  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert((size<0>(TileShape{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // Block scaling gmem-to-smem copy atom 
  using BlockScaleCopyTypeA = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementBlockScale)) * ScaleMsPerTile, 16)>;
  using BlockScaleCopyTypeB = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementBlockScale)) * ScaleNsPerTile, 16)>;
  using SmemBlockScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<BlockScaleCopyTypeA>, ElementBlockScale>;
  using SmemBlockScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<BlockScaleCopyTypeB>, ElementBlockScale>;

  // Block scaling smem layout
  using SmemLayoutScaleA = Layout<Shape<Int<ScaleMsPerTile>, Int<DispatchPolicy::Stages>>>;
  using SmemLayoutScaleB = Layout<Shape<Int<ScaleNsPerTile>, Int<DispatchPolicy::Stages>>>;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleA>> smem_scale_A;
      cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleB>> smem_scale_B;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementBlockScale const** ptr_scale_A;
    ElementBlockScale const** ptr_scale_B;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    InternalElementA const** ptr_A;
    StrideA dA;
    InternalElementB const** ptr_B;
    StrideB dB;
    // Block scaling factors for A and B
    ElementBlockScale const** ptr_scale_A; 
    ElementBlockScale const** ptr_scale_B;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    InternalElementA const* ptr_A_first_batch = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    InternalElementB const* ptr_B_first_batch = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,mock_L), stride_a));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,mock_L), stride_b));
    auto tma_load_a = make_tma_copy(
         GmemTiledCopyA{},
         tensor_a,
         SmemLayoutA{}(_,_,cute::Int<0>{}),
         make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
         size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    auto tma_load_b = make_tma_copy(
         GmemTiledCopyB{},
         tensor_b,
         SmemLayoutB{}(_,_,cute::Int<0>{}),
         make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
         size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    void* tensormaps = workspace;

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes,
      tensormaps,
      reinterpret_cast<InternalElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<InternalElementB const**>(args.ptr_B),
      args.dB,
      args.ptr_scale_A,
      args.ptr_scale_B
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    bool implementable = true;
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
        // We expect full tiles in K
        implementable = implementable && K % size<2>(TileShape{}) == 0;
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = DispatchPolicy::PipelineAsyncMmaStages;
  static constexpr uint32_t TmaTransactionBytes =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value))+
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
    ProblemShape_MNKL const& problem_shape_MNKL,
    Params const& mainloop_params,
    ElementBlockScale const* ptr_scale_A = nullptr,
    ElementBlockScale const* ptr_scale_B = nullptr
  ) const {

    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,mock_L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,mock_L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)
    auto tK = get<3>(gA_mkl.shape());

    // Make the tiled views of scale tensors
    auto scaleA_shape = make_shape(M / ScaleGranularityM, tK, L); // (scale_m,k,l)
    auto scaleB_shape = make_shape(N / ScaleGranularityN, tK, L); // (scale_n,k,l)
    auto scaleA_layout = make_ordered_layout(scaleA_shape, Step<_0, _1, _2>{});
    auto scaleB_layout = make_ordered_layout(scaleB_shape, Step<_0, _1, _2>{});

    // Note that mScaleA_mkl and mScaleB_nkl are already blocked tiled in the `m` host and
    // gScaleA_mkl and gScaleB_nkl in `g` global memory are same as mScaleA_mkl and mScaleB_nkl.

    Tensor mScaleA_mkl = make_tensor(make_gmem_ptr(ptr_scale_A), scaleA_layout); // (scale_m,k,l)
    Tensor mScaleB_nkl = make_tensor(make_gmem_ptr(ptr_scale_B), scaleB_layout); // (scale_n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, mScaleA_mkl, mScaleB_nkl);

  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorMapA, class TensorMapB,
    class TensorScaleA, class TensorScaleB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();
    // Blockscaling: Tma loads for load_input and CpAsync for load_scale
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});                        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});                        // (BLK_N,BLK_K,PIPE)
      Tensor sScaleA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_A.data()), SmemLayoutScaleA{});  // (ScaleMsPerTile,k)
      Tensor sScaleB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_B.data()), SmemLayoutScaleB{});  // (ScaleNsPerTile,k)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Block scaling: load_scale has scaling tensors in global memory which are not tiled
      Tensor mScaleA_mkl = get<2>(load_inputs);
      Tensor mScaleB_nkl = get<3>(load_inputs);

      auto scales_m = get<0>(mScaleA_mkl.shape());
      auto scales_n = get<0>(mScaleB_nkl.shape());

      Tensor gScaleA = local_tile(mScaleA_mkl, make_tile(Int<ScaleMsPerTile>{}), make_coord(m_coord,_,l_coord));                   // (ScaleMsPerTile,k,1)
      Tensor gScaleB = local_tile(mScaleB_nkl, make_tile(Int<ScaleNsPerTile>{}), make_coord(n_coord,_,l_coord));                   // (ScaleNsPerTile,k,1)

      TiledCopy scale_copy_a = make_tiled_copy(SmemBlockScalingCopyAtomA{}, Layout<Shape<_1>>{}, Layout<Shape<Int<ScaleMsPerTile>>>{});
      TiledCopy scale_copy_b = make_tiled_copy(SmemBlockScalingCopyAtomB{}, Layout<Shape<_1>>{}, Layout<Shape<Int<ScaleNsPerTile>>>{});

      ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x);
      ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x);

      Tensor tAgA_ScaleA = thr_scale_copy_a.partition_S(gScaleA);
      Tensor tAsA_ScaleA = thr_scale_copy_a.partition_D(sScaleA);

      Tensor tBgB_ScaleB = thr_scale_copy_b.partition_S(gScaleB);
      Tensor tBsB_ScaleB = thr_scale_copy_b.partition_D(sScaleB);

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        // Copy scale tensors from global memory to shared memory
        copy(scale_copy_a, tAgA_ScaleA(_,_,*k_tile_iter), tAsA_ScaleA(_,_,write_stage));
        copy(scale_copy_b, tBgB_ScaleB(_,_,*k_tile_iter), tBsB_ScaleB(_,_,write_stage));
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);

        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all 
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was 
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }


  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor scaleFactor) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor);
    }
  }

  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor1,
    class ScaleFactor2
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor1 scaleFactor1, ScaleFactor2 scaleFactor2) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor1, scaleFactor2);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor1, scaleFactor2);
    }
  }


  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Block scaling
    Tensor sScaleAViewAsC = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_A.data()),
      Layout<
        Shape<Shape<Int<ScaleGranularityM>, Int<ScaleMsPerTile>>, cute::tuple_element_t<1, TileShape>, Int<DispatchPolicy::Stages>>,
        Stride<Stride<_0, _1>, _0, Int<ScaleMsPerTile>>
      >{}); // ((ScaleGranularityM,ScaleMsPerTile),n,k)
    Tensor sScaleBViewAsC = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_B.data()),
      Layout<
        Shape<cute::tuple_element_t<0, TileShape>, Shape<Int<ScaleGranularityN>, Int<ScaleNsPerTile>>, Int<DispatchPolicy::Stages>>,
        Stride<_0, Stride<_0, _1>, Int<ScaleNsPerTile>>
      >{}); // (m,(ScaleGranularityN,ScaleNsPerTile),k)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and 
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsScaleAViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleAViewAsC);    // (MMA,MMA_M,MMA_N,PIPE), `thread_mma` above is correct when partitioning A and B, but it is not correct when partitioning C.
    Tensor tCsScaleBViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleBViewAsC);    // (MMA,MMA_M,MMA_N,PIPE), `thread_mma` above is correct when partitioning A and B, but it is not correct when partitioning C.

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Per block scale values for operand A and B
    Tensor tCrScaleAViewAsC = make_tensor_like<ElementBlockScale>(tCsScaleAViewAsC(_, _, _, 0));    // (MMA,MMA_M,MMA_N)
    Tensor tCrScaleBViewAsC = make_tensor_like<ElementBlockScale>(tCsScaleBViewAsC(_, _, _, 0));    // (MMA,MMA_M,MMA_N)

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    assert(k_tile_count >= 1);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // fence_operand();
    GmmaFP8Accumulation accumulation(accum, ScalePromotionInterval, size<2>(tCrA));
    warpgroup_fence_operand(accumulation());

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      int read_stage = smem_pipe_read.index();
      // Load per block scale values from shared memory to registers
      copy(tCsScaleAViewAsC(_, _, _, read_stage), tCrScaleAViewAsC);
      copy(tCsScaleBViewAsC(_, _, _, read_stage), tCrScaleBViewAsC);
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrScaleAViewAsC.data()[0] = tCrScaleAViewAsC.data()[0] * tCrScaleBViewAsC.data()[0];
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrScaleBViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleAViewAsC); i++) {
          tCrScaleAViewAsC.data()[i] = tCrScaleAViewAsC.data()[i] * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrScaleAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBViewAsC); i++) {
          tCrScaleBViewAsC.data()[i] = tCrScaleBViewAsC.data()[i] * scale_a;
        }
      }
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      // Block scale the accumulators with reg tensor `tCrScaleAViewAsC` and `tCrScaleBViewAsC`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC, tCrScaleBViewAsC);
      }

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      // fence_operand();
      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsScaleAViewAsC(_, _, _, read_stage), tCrScaleAViewAsC);
      copy(tCsScaleBViewAsC(_, _, _, read_stage), tCrScaleBViewAsC);
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrScaleAViewAsC.data()[0] = tCrScaleAViewAsC.data()[0] * tCrScaleBViewAsC.data()[0];
      }
      if constexpr (ScaleMsPerTile > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrScaleBViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleAViewAsC); i++) {
          tCrScaleAViewAsC.data()[i] = tCrScaleAViewAsC.data()[i] * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile > 1) {
        ElementBlockScale scale_a = tCrScaleAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBViewAsC); i++) {
          tCrScaleBViewAsC.data()[i] = tCrScaleBViewAsC.data()[i] * scale_a;
        }
      }

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());

      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accumulation());

      // Block scale the accumulators with reg tensor `tCrScaleAViewAsC` and `tCrScaleBViewAsC`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC, tCrScaleBViewAsC);
      }

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }
    if constexpr (ScalePromotionInterval != 4) {
      // residues only exists when granularity is not the finnest
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        accumulation.scale_residue_if_needed(scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        accumulation.scale_residue_if_needed(tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrScaleAViewAsC, tCrScaleBViewAsC);
      }
    }

    warpgroup_fence_operand(accumulation());

  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    InternalElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);

    InternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, 
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, 
                                             prob_shape_B, prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<InternalElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<InternalElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  template <class TensorMapA, class TensorMapB, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      [[maybe_unused]] InputTensors const& input_tensors,
      Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {

    if constexpr (IsGroupedGemmKernel) {
      return load_init(
        problem_shape_mnkl,
        mainloop_params,
        mainloop_params.ptr_scale_A[next_batch],
        mainloop_params.ptr_scale_B[next_batch]
      );
    } else {
      auto [gA_mkl, gB_nkl, mScaleA_mkl, mScaleB_nkl] = input_tensors;

      auto scaleA_layout = mScaleA_mkl.layout();
      auto scaleB_layout = mScaleB_nkl.layout();

      mScaleA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_A[next_batch]), scaleA_layout); // (m,ScaleMsPerTile,k,l)
      mScaleB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_B[next_batch]), scaleB_layout); // (n,ScaleNsPerTile,k,l)
      return cute::make_tuple(gA_mkl, gB_nkl, mScaleA_mkl, mScaleB_nkl);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  int ScaleGranularityM_,
  int ScaleGranularityN_,
  int ScalePromotionInterval_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule, ScaleGranularityM_, ScaleGranularityN_, ScalePromotionInterval_>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementBlockScale = ElementAccumulator;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  // Two threads per CTA are producers (1 for operand tile and 1 for scales)
  static constexpr int NumProducerThreadEvents = 2; 

  static constexpr int ScaleGranularityM = ScaleGranularityM_ == 0 ? size<0>(TileShape{}) : ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_ == 0 ? size<1>(TileShape{}) : ScaleGranularityN_;
  static constexpr int ScalePromotionInterval = ScalePromotionInterval_;
  static_assert(ScalePromotionInterval % 4 == 0, "ScalePromotionInterval must be a multiple of 4.");
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert((size<0>(TileShape{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // Block scaling gmem-to-smem copy atom 
  using BlockScaleCopyTypeA = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementBlockScale)) * ScaleMsPerTile, 16)>;
  using BlockScaleCopyTypeB = cute::uint_byte_t<cute::min(static_cast<int>(sizeof(ElementBlockScale)) * ScaleNsPerTile, 16)>;
  using SmemBlockScalingCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<BlockScaleCopyTypeA>, ElementBlockScale>;
  using SmemBlockScalingCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<BlockScaleCopyTypeB>, ElementBlockScale>;

  // Block scaling smem layout
  using SmemLayoutScaleA = Layout<Shape<Int<ScaleMsPerTile>, Int<DispatchPolicy::Stages>>>;
  using SmemLayoutScaleB = Layout<Shape<Int<ScaleNsPerTile>, Int<DispatchPolicy::Stages>>>;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;  // mxk
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;  // nxk
      cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleA>> smem_scale_A; // ScaleMsPerTile x k
      cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleB>> smem_scale_B; // ScaleNsPerTile x k
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    uint32_t mma_promotion_interval = 4;
    ElementBlockScale const* ptr_scale_A; 
    ElementBlockScale const* ptr_scale_B;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,0),
        TileShape{},
        ClusterShape{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t mma_promotion_interval = 4;
    // Block scaling factors for A and B
    ElementBlockScale const* ptr_scale_A; 
    ElementBlockScale const* ptr_scale_B;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = reinterpret_cast<ElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<ElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    return {
      tma_load_a,
      tma_load_b,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk,
      args.mma_promotion_interval,
      args.ptr_scale_A,
      args.ptr_scale_B
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});
    /* MMA promotion interval should be a multiple of 4, since each mainloop iteration would issue 4 MMA instructions. */
    constexpr int pipe_k = size<2>(TileShape{}) / tile_size<2>(TiledMma{});
    implementable = implementable && (args.mma_promotion_interval % 4 == 0) && (args.mma_promotion_interval == ScalePromotionInterval);
    implementable = implementable && (pipe_k % 4 == 0) && (pipe_k <= args.mma_promotion_interval);

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params)
  {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    // Make the tiled views of scale tensors
    auto scaleA_shape = make_shape(shape<2>(gA_mkl), Int<ScaleMsPerTile>{}, shape<3>(gA_mkl), shape<4>(gA_mkl));  // (m,ScaleMsPerTile,k,l)
    auto scaleB_shape = make_shape(shape<2>(gB_nkl), Int<ScaleNsPerTile>{}, shape<3>(gB_nkl), shape<4>(gB_nkl));  // (n,ScaleNsPerTile,k,l)
    auto scale_dA = compact_order(scaleA_shape, Step<_2,_0,_1,_3>{});
    auto scale_dB = compact_order(scaleB_shape, Step<_2,_0,_1,_3>{});
    auto scaleA_layout = make_layout(scaleA_shape, scale_dA);
    auto scaleB_layout = make_layout(scaleB_shape, scale_dB);

    // Note that mScaleA_mkl and mScaleB_nkl are already blocked tiled in the `m` host and
    // gScaleA_mkl and gScaleB_nkl in `g` global memory are same as mScaleA_mkl and mScaleB_nkl.
    Tensor mScaleA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_A), scaleA_layout); // (m,ScaleMsPerTile,k,l)
    Tensor mScaleB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_scale_B), scaleB_layout); // (n,ScaleNsPerTile,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, mScaleA_mkl, mScaleB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorScaleA, class TensorScaleB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();
    // Blockscaling: Tma loads for load_input and CpAsync for load_scale
    if (lane_predicate) {

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      Tensor sScaleA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_A.data()), SmemLayoutScaleA{}); // (ScaleMsPerTile,k)
      Tensor sScaleB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_B.data()), SmemLayoutScaleB{}); // (ScaleNsPerTile,k)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)


      // Block scaling: load_scale has scaling tensors in global memory which are not tiled
      Tensor mScaleA_mkl = get<2>(load_inputs);
      Tensor mScaleB_nkl = get<3>(load_inputs);

      Tensor gScaleA = mScaleA_mkl(m_coord,_,_,l_coord);                                         // (1,ScaleMsPerTile,k,1)
      Tensor gScaleB = mScaleB_nkl(n_coord,_,_,l_coord);                                         // (1,ScaleNsPerTile,k,1)

      TiledCopy scale_copy_a = make_tiled_copy(SmemBlockScalingCopyAtomA{}, Layout<Shape<_1>>{}, Layout<Shape<Int<ScaleMsPerTile>>>{}); // (1,ScaleMsPerTile,1)
      TiledCopy scale_copy_b = make_tiled_copy(SmemBlockScalingCopyAtomB{}, Layout<Shape<_1>>{}, Layout<Shape<Int<ScaleNsPerTile>>>{}); // (1,ScaleNsPerTile,1)
      ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x);
      ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x);

      Tensor tAgA_ScaleA = thr_scale_copy_a.partition_S(gScaleA);
      Tensor tAsA_ScaleA = thr_scale_copy_a.partition_D(sScaleA);

      Tensor tBgB_ScaleB = thr_scale_copy_b.partition_S(gScaleB);
      Tensor tBsB_ScaleB = thr_scale_copy_b.partition_D(sScaleB);

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads for GEMM operands A/B and CpAsync for scale tensors
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //
        int write_stage = smem_pipe_write.index();
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        // Copy operands A and B from global memory to shared memory
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        // Copy scale tensors from global memory to shared memory
        copy(scale_copy_a, tAgA_ScaleA(_,_,*k_tile_iter), tAsA_ScaleA(_,_,write_stage));
        copy(scale_copy_b, tBgB_ScaleB(_,_,*k_tile_iter), tBsB_ScaleB(_,_,write_stage));
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);

        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor scaleFactor) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor);
    }
  }
  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor1,
    class ScaleFactor2
  >
  CUTLASS_DEVICE
  void scale_if_needed(GmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor1 scaleFactor1, ScaleFactor2 scaleFactor2) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor1, scaleFactor2);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor1, scaleFactor2);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {


    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Block scaling
    Tensor sScaleAViewAsC = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_A.data()),
      Layout<
        Shape<Shape<Int<ScaleGranularityM>, Int<ScaleMsPerTile>>, cute::tuple_element_t<1, TileShape>, Int<DispatchPolicy::Stages>>,
        Stride<Stride<_0, _1>, _0, Int<ScaleMsPerTile>>
      >{}); // ((ScaleGranularityM,ScaleMsPerTile),n,k)
    Tensor sScaleBViewAsC = make_tensor(cute::make_smem_ptr(shared_tensors.smem_scale_B.data()),
      Layout<
        Shape<cute::tuple_element_t<0, TileShape>, Shape<Int<ScaleGranularityN>, Int<ScaleNsPerTile>>, Int<DispatchPolicy::Stages>>,
        Stride<_0, Stride<_0, _1>, Int<ScaleNsPerTile>>
      >{}); // (m,(ScaleGranularityN,ScaleNsPerTile),k)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and 
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup, 
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{}, 
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsScaleAViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleAViewAsC);    // (MMA,MMA_M,MMA_N,PIPE), `thread_mma` above is correct when partitioning A and B, but it is not correct when partitioning C.
    Tensor tCsScaleBViewAsC = tiled_mma.get_slice(thread_idx).partition_C(sScaleBViewAsC);    // (MMA,MMA_M,MMA_N,PIPE), `thread_mma` above is correct when partitioning A and B, but it is not correct when partitioning C.

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Per block scale values for operand A and B
    Tensor tCrScaleAViewAsC = make_tensor_like<ElementBlockScale>(tCsScaleAViewAsC(_, _, _, 0));    // (MMA,MMA_M,MMA_N)
    Tensor tCrScaleBViewAsC = make_tensor_like<ElementBlockScale>(tCsScaleBViewAsC(_, _, _, 0));    // (MMA,MMA_M,MMA_N)

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    GmmaFP8Accumulation accumulation(accum, ScalePromotionInterval, size<2>(tCrA));
    warpgroup_fence_operand(accumulation());
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers
      copy(tCsScaleAViewAsC(_, _, _, read_stage), tCrScaleAViewAsC);
      copy(tCsScaleBViewAsC(_, _, _, read_stage), tCrScaleBViewAsC);
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrScaleAViewAsC.data()[0] = tCrScaleAViewAsC.data()[0] * tCrScaleBViewAsC.data()[0];
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrScaleBViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleAViewAsC); i++) {
          tCrScaleAViewAsC.data()[i] = tCrScaleAViewAsC.data()[i] * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrScaleAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBViewAsC); i++) {
          tCrScaleBViewAsC.data()[i] = tCrScaleBViewAsC.data()[i] * scale_a;
        }
      }

      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      // Block scale the accumulators with reg tensor `tCrScaleAViewAsC` and `tCrScaleBViewAsC`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC, tCrScaleBViewAsC);
      }

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsScaleAViewAsC(_, _, _, read_stage), tCrScaleAViewAsC);
      copy(tCsScaleBViewAsC(_, _, _, read_stage), tCrScaleBViewAsC);
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrScaleAViewAsC.data()[0] = tCrScaleAViewAsC.data()[0] * tCrScaleBViewAsC.data()[0];
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrScaleBViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleAViewAsC); i++) {
          tCrScaleAViewAsC.data()[i] = tCrScaleAViewAsC.data()[i] * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrScaleAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrScaleBViewAsC); i++) {
          tCrScaleBViewAsC.data()[i] = tCrScaleBViewAsC.data()[i] * scale_a;
        }
      }

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accumulation());

      // Block scale the accumulators with reg tensor `tCrScaleAViewAsC` and `tCrScaleBViewAsC`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrScaleAViewAsC, tCrScaleBViewAsC);
      }

      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    if constexpr (ScalePromotionInterval != 4) {
      // residues only exists when granularity is not the finnest
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrScaleAViewAsC.data()[0];
        accumulation.scale_residue_if_needed(scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        accumulation.scale_residue_if_needed(tCrScaleAViewAsC);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrScaleBViewAsC);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrScaleAViewAsC, tCrScaleBViewAsC);
      }
    }

    warpgroup_fence_operand(accumulation());
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/dispatch_policy.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp" // cute::false_type
#include "cute/arch/copy_sm100.hpp" 

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

template <class T, template <int...> class U>
struct is_kernel_tag_of : cute::false_type {};

template <template <int...> class U, int... Args>
struct is_kernel_tag_of<U<Args...>, U> : cute::true_type {};

template <class T, template <int...> class U>
constexpr bool is_kernel_tag_of_v = is_kernel_tag_of<T, U>::value;

template <class T, template <int,bool> class U>
struct is_asymmetric_dma_kernel_tag_of : cute::false_type {};

template <template <int, bool> class U, int I0, bool B0>
struct is_asymmetric_dma_kernel_tag_of<U<I0, B0>, U> : cute::true_type {};

template <class T, template <int, bool> class U>
constexpr bool is_asymmetric_dma_kernel_tag_of_v = \
                              is_asymmetric_dma_kernel_tag_of<T, U>::value;

}

//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

//////////////////////////////////////////////////////////////////////////////

namespace detail {

enum class KernelInputTransformType {
    FastF32,
    InterleavedComplexTF32
};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////

namespace kernel::detail {

// Has_SwapAB<T>::value will be true only if:
//   class T has member SwapAB and T::SwapAB is true
template <typename T, typename = void>
struct Has_SwapAB { static constexpr bool value = false; };

template <typename T>
struct Has_SwapAB <T, CUTE_STL_NAMESPACE::void_t<decltype(T::SwapAB)>>
{ static constexpr bool value = T::SwapAB; };

template <typename T>
static constexpr bool Has_SwapAB_v = Has_SwapAB<T>::value;

} // namespace kernel::detail

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { 
};
struct KernelTmaWarpSpecializedCooperative { 
};

struct KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpong { };

// FP8 related policies (including Blocked Scaled Accumulation)
template<
  // `ScaleGranularityM`/`ScaleGranularityN` specifies scaling granularity along M/N, while zero-value
  // `ScaleGranularityM`/`ScaleGranularityN` indicates that scaling granularity is
  // `size<0>(TileShape_MNK{})`/`size<1>(TileShape_MNK{})` along M/N.
  int ScaleGranularityM_ = 0,
  int ScaleGranularityN_ = 0,
  // `ScalePromotionInterval` specifies the interval to promote the accumulator for scaling
  // It is required to be a multiple of 4 and specified in terms of number of MMA instructions
  // in the reduction dimension. i.e for FP8 kernels, it is 
  // ScalePromotionInterval * MMA_K = ScalePromotionInterval * 32 = 128 elements in K by default
  int ScalePromotionInterval_ = 4

>
struct KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum: KernelTmaWarpSpecializedCooperative {
  constexpr static int ScaleGranularityM = ScaleGranularityM_;
  constexpr static int ScaleGranularityN = ScaleGranularityN_;
  constexpr static int ScalePromotionInterval = ScalePromotionInterval_;
};

template<
  // `ScaleGranularityM`/`ScaleGranularityN` specifies scaling granularity along M/N, while zero-value
  // `ScaleGranularityM`/`ScaleGranularityN` indicates that scaling granularity is
  // `size<0>(TileShape_MNK{})`/`size<1>(TileShape_MNK{})` along M/N.
  int ScaleGranularityM_,
  int ScaleGranularityN_,
  // `ScalePromotionInterval` specifies the interval to promote the accumulator for scaling
  // It is required to be a multiple of 4 and specified in terms of number of MMA instructions
  // in the reduction dimension. i.e for FP8 kernels, it is 
  // ScalePromotionInterval * MMA_K = ScalePromotionInterval * 32 = 128 elements in K by default
  int ScalePromotionInterval_ = 4
>
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum: KernelPtrArrayTmaWarpSpecializedCooperative {
  constexpr static int ScaleGranularityM = ScaleGranularityM_;
  constexpr static int ScaleGranularityN = ScaleGranularityN_;
  constexpr static int ScalePromotionInterval = ScalePromotionInterval_;
};

// Policies to opt into mixed type GEMMs
struct KernelTmaWarpSpecializedMixedInput : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongMixedInput : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeMixedInput: KernelTmaWarpSpecializedCooperative { };

//////////////////////////////////////////////////////////////////////////////

//
// Builder dispatch policies (not a part of the main CUTLASS layers, simply used to opt into
// specific collective builder dispatches)
//

// FP8 related policies (including Fast Accumulation)
struct KernelTmaWarpSpecializedFP8FastAccum : KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpongFP8FastAccum : KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperativeFP8FastAccum: KernelTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum : KernelPtrArrayTmaWarpSpecializedCooperative { };
struct KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum : KernelPtrArrayTmaWarpSpecializedPingpong { };

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopSm70TwoStageUnpredicated {
  constexpr static int Stages = 2;
  using ArchTag = arch::Sm70;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline through 1 stage in smem, 1 in rmem, with predicated gmem loads
struct MainloopSm70TwoStage {
  constexpr static int Stages = 2;
  using ArchTag = arch::Sm70;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, WITHOUT predicated gmem loads
template<int Stages_>
struct MainloopSm80CpAsyncUnpredicated {
  constexpr static int Stages = Stages_;
  using ArchTag = arch::Sm80;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// n-buffer in smem (cp.async), pipelined with registers, with predicated gmem loads
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm80CpAsync {
  constexpr static int Stages = Stages_;
  using ArchTag = cute::conditional_t<(size(ClusterShape_{}) > 1), arch::Sm90, arch::Sm80>;
  using Schedule = KernelMultistage;
  using ClusterShape = ClusterShape_;
};

// n-buffer in smem (cp.async), pipelined with Hopper GMMA, with predicated gmem loads, warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelCpAsyncWarpSpecialized
>
struct MainloopSm90CpAsyncGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (cp.async), pipelined with Hopper GMMA, with predicated gmem loads, warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelCpAsyncWarpSpecialized
>
struct MainloopSm90CpAsyncGmmaRmemAWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, static schedule between TMA and GMMA
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  int PipelineAsyncMmaStages_ = 1
>
struct MainloopSm90TmaGmma {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  constexpr static int PipelineAsyncMmaStages = PipelineAsyncMmaStages_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelTma;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// With GMMA's A data from registers.
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaRmemAWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};


template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<Schedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaWarpSpecializedFP8
  : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecialized> ||
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedPingpong> ||
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperative>,
    "KernelSchedule must be one of the warp specialized policies");
};


// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels with Block Scaling
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized,
  // `ScaleGranularityM`/`ScaleGranularityN` specifies scaling granularity along M/N, while zero-value
  // `ScaleGranularityM`/`ScaleGranularityN` indicates that scaling granularity is
  // `size<0>(TileShape_MNK{})`/`size<1>(TileShape_MNK{})` along M/N.
  int ScaleGranularityM = 0,
  int ScaleGranularityN = 0,
  // `ScalePromotionInterval` specifies the interval to promote the accumulator for scaling
  // It is required to be a multiple of 4 and specified in terms of number of MMA instructions
  // in the reduction dimension. i.e for FP8 kernels, it is 
  // ScalePromotionInterval * MMA_K = ScalePromotionInterval * 32 = 128 elements in K by default
  int ScalePromotionInterval = 4
>
struct MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8
  : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM, ScaleGranularityN, ScalePromotionInterval>>,
    "KernelSchedule must be one of the warp specialized policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  constexpr static int PipelineAsyncMmaStages = 1;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelSchedule> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelSchedule>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
// For FP8 kernels
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedFP8
  : MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, KernelSchedule> ||
    cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, KernelSchedule>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative or Pingpong policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper sparse GMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecializedSparse {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};

// For slow-accumulation sparse FP8 kernels
template<
  int Stages,
  class ClusterShape = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecializedSparseFP8 
  : MainloopSm90TmaGmmaWarpSpecializedSparse<Stages, ClusterShape, KernelSchedule> {
};

// Mixed precision version n-buffer in rmem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule for Ptr-Array and Grouped Gemm
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedCooperative> ||
    cute::is_same_v<Schedule, KernelPtrArrayTmaWarpSpecializedPingpong>,
    "KernelSchedule must be one of the Ptr-Array or Grouped Gemm TMA Warp Specialized Cooperative policies");
};

// n-buffer in smem (Hopper TMA), pipelined with Hopper GMMA and TMA, Warp specialized dynamic schedule
// For FP8 kernels with Block Scaling
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelPtrArrayTmaWarpSpecializedCooperative,
  // `ScaleGranularityM`/`ScaleGranularityN` specifies scaling granularity along M/N, while zero-value
  // `ScaleGranularityM`/`ScaleGranularityN` indicates that scaling granularity is
  // `size<0>(TileShape_MNK{})`/`size<1>(TileShape_MNK{})` along M/N.
  int ScaleGranularityM = 0,
  int ScaleGranularityN = 0,
  int ScalePromotionInterval = 4
>
struct MainloopSm90ArrayTmaGmmaWarpSpecializedBlockScaling
  : MainloopSm90ArrayTmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {
  static_assert(
    cute::is_same_v<KernelSchedule, KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM, ScaleGranularityN>>,
    "KernelSchedule must be one of the warp specialized policies");
};


template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Gemm with block scaling factors
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedBlockScaledSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedMmaTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};


// InputTransform GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelTmaWarpSpecializedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array Dense GEMM: SM100 tensor op policy that applies to both 1SM and 2SM MMA atoms
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array Block Scaled GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedBlockScaledSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};

// Ptr-Array InputTransform GEMM
template<
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_
>
struct KernelPtrArrayTmaWarpSpecializedInputTransformSm100 final {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  static constexpr int AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
};


//////////////////////////////////////////////////////////////////////////////

//
// Collective Builder Tag Property
//

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
//          SM100 Dispatch Policies
//
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Base Dispatch Policies
struct KernelSchedule1Sm {};
struct KernelSchedule2Sm {};
struct KernelScheduleSm100 {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100DenseGemm : KernelScheduleSm100 {};   // Base policy
// Dense GEMM: Specialize for 1SM vs 2SM
struct KernelTmaWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100DenseGemm {};  // Use for 1SM Dense GEMM Kernels for Collective Mainloop Builder
struct KernelTmaWarpSpecialized2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100DenseGemm {};  // Use for 2SM Dense GEMM Kernels for Collective Mainloop Builder
// Dense GEMM + (Ptr Array or Group GEMM)
struct KernelScheduleSm100PtrArrayDenseGemm : KernelScheduleSm100DenseGemm {};
// Ptr-Array Dense GEMM: Specialize for 1SM vs 2SM
struct KernelPtrArrayTmaWarpSpecialized1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayDenseGemm {};
struct KernelPtrArrayTmaWarpSpecialized2SmSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayDenseGemm {};


///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Blockwise GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100Blockwise  : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedBlockwise1SmSm100 final : KernelSchedule1Sm, KernelScheduleSm100Blockwise {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 Planar Complex GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100PlanarComplexGemm : KernelScheduleSm100{};
// Planar Complex GEMM: Specialize for 1SM vs 2SM
struct KernelTmaWarpSpecialized1SmPlanarComplexSm100 final : KernelSchedule1Sm, KernelScheduleSm100PlanarComplexGemm { };
struct KernelTmaWarpSpecialized2SmPlanarComplexSm100 final : KernelSchedule2Sm, KernelScheduleSm100PlanarComplexGemm { };
// Planar Complex GEMM + (Ptr Array or Group GEMM)
struct KernelScheduleSm100PtrArrayPlanarComplexGemm : KernelScheduleSm100PlanarComplexGemm {};
struct KernelPtrArrayTmaWarpSpecialized1SmPlanarComplexSm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayPlanarComplexGemm {};
struct KernelPtrArrayTmaWarpSpecialized2SmPlanarComplexSm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayPlanarComplexGemm {};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 FastF32 (9xBF16) GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleSm100FastFP32Gemm : KernelScheduleSm100 {};
struct KernelTmaWarpSpecializedFastFP32SmemSm100 : KernelScheduleSm100FastFP32Gemm { };
// Dispatch policies without smem load the A operand from tmem
struct KernelTmaWarpSpecialized1SmFastFP32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100FastFP32Gemm { };
struct KernelTmaWarpSpecialized2SmFastFP32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100FastFP32Gemm { };
// Dispatch policies with smem load the A operand from smem
struct KernelTmaWarpSpecialized1SmFastFP32SmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedFastFP32SmemSm100 { };
struct KernelTmaWarpSpecialized2SmFastFP32SmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedFastFP32SmemSm100 { };
// Ptr-Array Transform GEMM: Specialize for 1SM vs 2SM FastF32 GEMM
struct KernelScheduleSm100PtrArrayFastFP32Gemm : KernelScheduleSm100FastFP32Gemm {};
struct KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 : KernelScheduleSm100PtrArrayFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized1SmFastFP32Sm100 final : KernelSchedule1Sm, KernelScheduleSm100PtrArrayFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized2SmFastFP32Sm100 final : KernelSchedule2Sm, KernelScheduleSm100PtrArrayFastFP32Gemm { };
struct KernelPtrArrayTmaWarpSpecialized1SmFastFP32SmemSm100 final : KernelSchedule1Sm, KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmFastFP32SmemSm100 final : KernelSchedule2Sm, KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100 { };

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BlockScaled Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelScheduleBlockScaledGemmSm100 : KernelScheduleSm100 {};                  
struct KernelScheduleMxNvf4Sm100 : KernelScheduleBlockScaledGemmSm100 {};
struct KernelScheduleMxf8f6f4Sm100 : KernelScheduleBlockScaledGemmSm100 {};
// Block Scaled Dense GEMM: Specialize for instruction type, scale factor vector size, and 1SM vs. 2SM
struct KernelTmaWarpSpecialized1SmBlockScaledSm100 final : KernelSchedule1Sm, KernelScheduleBlockScaledGemmSm100 { };
struct KernelTmaWarpSpecialized2SmBlockScaledSm100 final : KernelSchedule2Sm, KernelScheduleBlockScaledGemmSm100 { };
struct KernelTmaWarpSpecialized1SmNvf4Sm100 final : KernelSchedule1Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized2SmNvf4Sm100 final : KernelSchedule2Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized1SmMxf4Sm100 final : KernelSchedule1Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized2SmMxf4Sm100 final : KernelSchedule2Sm, KernelScheduleMxNvf4Sm100 { };
struct KernelTmaWarpSpecialized1SmMxf8f6f4Sm100 final : KernelSchedule1Sm, KernelScheduleMxf8f6f4Sm100 { };
struct KernelTmaWarpSpecialized2SmMxf8f6f4Sm100 final : KernelSchedule2Sm, KernelScheduleMxf8f6f4Sm100 { };
///////////////////////////////////////////////////////////////////////////////////////////////////////
// SM100 BlockScaled Ptr Array Dense GEMM Dispatch Policies
///////////////////////////////////////////////////////////////////////////////////////////////////////
// BlockScaled Dense GEMM + (Ptr Array or Group GEMM)
struct KernelSchedulePtrArrayBlockScaledGemmSm100 : KernelScheduleBlockScaledGemmSm100 {};
struct KernelSchedulePtrArrayMxNvf4Sm100 : KernelSchedulePtrArrayBlockScaledGemmSm100 {};
struct KernelSchedulePtrArrayMxf8f6f4Sm100 : KernelSchedulePtrArrayBlockScaledGemmSm100 {};
// Ptr-Array Block Scaled Dense GEMM: Specialize for instruction type, scale factor vector size, and 1SM vs. 2SM
struct KernelPtrArrayTmaWarpSpecialized1SmBlockScaledSm100 final : KernelSchedule1Sm, KernelSchedulePtrArrayBlockScaledGemmSm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100 final : KernelSchedule2Sm, KernelSchedulePtrArrayBlockScaledGemmSm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100 final : KernelSchedule1Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100 final : KernelSchedule2Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100 final : KernelSchedule1Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100 final : KernelSchedule2Sm, KernelSchedulePtrArrayMxNvf4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100 final : KernelSchedule1Sm, KernelSchedulePtrArrayMxf8f6f4Sm100 { };
struct KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100 final : KernelSchedule2Sm, KernelSchedulePtrArrayMxf8f6f4Sm100 { };

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedMmaTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
  constexpr static bool IsOverlappingAccum = false;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100TmaUmmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};



// n-buffer in smem, pipelined with Blackwell Fast FP32 kernel with UMMA (HwScaled) and TMA,
// Warp specialized dynamic schedule
template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Load2TransformPipelineStageCount_,
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Transform2MmaPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of MMA Bands to be computed in a single FastF32 MMA operation.
  // For BF16 emulation, we have 3 compute matrices, with 9 MMAs forming 5 bands.
  //    We can eliminate bands 4 and/or 5 (up to last 3 MMA operations).
  //    Valid values are 3, 4, 5
  int NumBandsToCompute_,
  // Scaling factor for decomposed matrices (2^ScalingFactor)
  // 8 for BF16, 11 for TF32
  int ScalingFactor_,
  // Number of UMMA instructions emulated a single stage
  // Ex: Staged16 has 1 FastF32 MMA per stage
  // Should be smaller than K-mode of a single ClusterTile
  int AccPromotionInterval_,
  // ClusterShape for the kernel
  class ClusterShape_ = Shape<_1,_1,_1>,
  // The TMEM_LOAD atom to be used for loading local accumulator
  // from TMEM to registers
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_32dp32b32x
>
struct MainloopSm100TmaUmmaWarpSpecializedFastF32 {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
  constexpr static int AccPromotionInterval = AccPromotionInterval_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::FastF32;
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};



// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = false;
  using Schedule = KernelPtrArrayTmaWarpSpecializedSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};

// n-buffer in smem, pipelined with Blackwell UMMA and TMA, Warp specialized dynamic schedule
template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape_ = Shape<_1,_1,_1>
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  constexpr static bool IsOverlappingAccum = AccumulatorPipelineStageCount_ == 1;
  using Schedule = KernelPtrArrayTmaWarpSpecializedBlockScaledSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};



// n-buffer in smem, pipelined with Blackwell Fast FP32 kernel with UMMA (HwScaled) and TMA,
// Warp specialized dynamic schedule
template<
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Load2TransformPipelineStageCount_,
  // Number of Pipeline stages for
  // MainloopLoad <-> Conversion <-> MainLoad
  int Transform2MmaPipelineStageCount_,
  // TileScheduler pipeline depth
  int SchedulerPipelineStageCount_,
  // Accmulator pipeline depth
  int AccumulatorPipelineStageCount_,
  // Number of MMA Bands to be computed in a single FastF32 MMA operation.
  // For BF16 emulation, we have 3 compute matrices, with 9 MMAs forming 5 bands.
  //    We can eliminate bands 4 and/or 5 (up to last 3 MMA operations).
  //    Valid values are 3, 4, 5
  int NumBandsToCompute_,
  // Scaling factor for decomposed matrices (2^ScalingFactor)
  // 8 for BF16, 11 for TF32
  int ScalingFactor_,
  // Number of UMMA instructions emulated a single stage
  // Ex: Staged16 has 1 FastF32 MMA per stage
  // Should be smaller than K-mode of a single ClusterTile
  int AccPromotionInterval_,
  // ClusterShape for the kernel
  class ClusterShape_ = Shape<_1,_1,_1>,
  // The TMEM_LOAD atom to be used for loading local accumulator
  // from TMEM to registers
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_32dp32b32x
>
struct MainloopSm100ArrayTmaUmmaWarpSpecializedFastF32 {
  constexpr static int Load2TransformPipelineStageCount = Load2TransformPipelineStageCount_;
  constexpr static int Transform2MmaPipelineStageCount = Transform2MmaPipelineStageCount_;
  constexpr static int NumBandsToCompute = NumBandsToCompute_;
  constexpr static int ScalingFactor = ScalingFactor_;
  constexpr static int AccPromotionInterval = AccPromotionInterval_;
  constexpr static detail::KernelInputTransformType InputTransformType = detail::KernelInputTransformType::FastF32;
  using ClusterShape = ClusterShape_;
  using AccumulatorCopyAtom = AccumulatorCopyAtom_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelPtrArrayTmaWarpSpecializedInputTransformSm100<SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;

  // For backwards compatibility with GemmUniversalAdapter.
  constexpr static int Stages = Load2TransformPipelineStageCount;
};



//////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm
""",
    'key-files/include/cutlass/gemm/kernel/gemm_universal.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

// In cases where ProblemShape is not a tuple, this is used to check if the
// underlying problem shape type is aliased within or not.
// Used for dispatching GemmUniversal to 2.x API or 3.x API
template <class ProblemShape, class = void>
struct IsCutlass3ArrayKernel : cute::false_type { };

template <typename ProblemShape>
struct IsCutlass3ArrayKernel<ProblemShape, cute::void_t<typename ProblemShape::UnderlyingProblemShape>>
    : cute::true_type { };

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel

////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/kernel/sm70_gemm.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_warpspecialized_pingpong.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_warpspecialized_cooperative.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp"

#include "cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp"
#include "cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_input_transform.hpp"
#include "cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp"

////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/mainloop_fusion_helper_scale_factor.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedMmaTransformSm100>>> {
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool IsComplex = CollectiveEpilogue::NumAccumulatorMtxs == 2;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  static_assert(!IsOverlappingAccum, "Does not support overlapping accumulator");

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps       = NumEpilogueThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});

  // Fixup performed for split-/stream-K is done across warps in different CTAs
  // at epilogue subtile granularity. Thus, there must be one barrier per sub-tile per
  // epilogue warp.
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using Mma2TransformPipeline = typename CollectiveMainloop::Mma2TransformPipeline;
  using Mma2TransformPipelineState = typename Mma2TransformPipeline::PipelineState;

  using Load2TransformPipeline = typename CollectiveMainloop::Load2TransformPipeline;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  static constexpr uint32_t GenericRegisterRequirement = 104;
  static constexpr uint32_t AccumRegisterRequirement = 256;

  // Kernel level shared memory storage
  struct SharedStorage {
    // Barriers should be allocated in lower 8KB of SMEM for SM100
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using Load2TransformPipelineStorage = typename CollectiveMainloop::Load2TransformPipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using Mma2TransformPipelineStorage = typename CollectiveMainloop::Mma2TransformPipelineStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) Load2TransformPipelineStorage load2transform;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) Mma2TransformPipelineStorage mma2transform;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
      alignas(16) arch::ClusterBarrier epilogue_throttle;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

  // Host facing host arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{}; 
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueLoad = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_load  = false;
    uint32_t epilogue  = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
          "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace
      )
      ,args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // NOTE cluster_shape here is the major cluster shape, not fallback one
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_grid_shape(
        params.scheduler,
        problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
  }

  static constexpr
  dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_category == WarpCategory::Sched) && lane_predicate) {
      collective_mainloop.prefetch_tma_descriptors();
    }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
    }

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

    typename Load2TransformPipeline::Params load2transform_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      load2transform_pipeline_params.role = Load2TransformPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      load2transform_pipeline_params.role = Load2TransformPipeline::ThreadCategory::Consumer;
    }
    load2transform_pipeline_params.initializing_warp = 0;
    load2transform_pipeline_params.producer_arv_count = CollectiveMainloop::NumLoad2TransformProducerThreadEvents;
    load2transform_pipeline_params.consumer_arv_count = NumEpilogueThreads;

    Load2TransformPipeline load2transform_pipeline(shared_storage.pipelines.load2transform,
                                          load2transform_pipeline_params);

    // Epilogue Load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 4;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopLoadThreads;
    load_order_barrier_params.initializing_warp = 5;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                 (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads);
    if (is_epi_load_needed) {
      clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 1;
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);

    // Mainloop-Epilogue pipeline
    typename Mma2TransformPipeline::Params mma2transform_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      mma2transform_pipeline_params.role = Mma2TransformPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      mma2transform_pipeline_params.role = Mma2TransformPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    mma2transform_pipeline_params.producer_arv_count = 1;
    mma2transform_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    mma2transform_pipeline_params.initializing_warp = 2;
    Mma2TransformPipeline mma2transform_pipeline(shared_storage.pipelines.mma2transform,
                                                 mma2transform_pipeline_params,
                                                 cluster_shape);

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
    clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
    clc_throttle_pipeline_params.dst_blockid = 0;
    clc_throttle_pipeline_params.initializing_warp = 3;
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    
    if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
      tmem_deallocation_result_barrier.init(NumMMAThreads);
    }


    // Initialize smem barrier for prologue throttling. Epilogue warps are stalled until the prologue finishes.
    arch::ClusterBarrier& epilogue_throttle_barrier = shared_storage.pipelines.epilogue_throttle;
    if (WarpCategory::MMA == warp_category && lane_predicate) {
      epilogue_throttle_barrier.init(                          NumMMAThreads +
                                    (is_first_cta_in_cluster ? NumSchedThreads : 0) +
                                                               NumMainloopLoadThreads +
                                    (is_epi_load_needed      ? NumEpilogueLoadThreads : 0));
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    auto load_inputs = collective_mainloop.load_init(
        problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop);

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    Mma2TransformPipelineState mma2transform_pipe_consumer_state;
    Mma2TransformPipelineState mma2transform_pipe_producer_state = cutlass::make_producer_start_state<Mma2TransformPipeline>();

    Load2TransformPipelineState load2transform_pipe_consumer_state;
    Load2TransformPipelineState load2transform_pipe_producer_state = cutlass::make_producer_start_state<Load2TransformPipeline>();

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    mma2transform_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    //
    // TMEM "Allocation"
    //
    auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();
      bool requires_clc_query = true;

      auto pipelines = cute::make_tuple(mainloop_pipeline, load2transform_pipeline);
      auto states = cute::make_tuple(mainloop_pipe_producer_state, load2transform_pipe_producer_state);

      do {

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }

        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        auto [mainloop_producer_state_next, load2transform_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
          mainloop_pipeline,
          load2transform_pipeline,
          mainloop_pipe_producer_state,
          load2transform_pipe_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter, k_tile_prologue
        );
        mainloop_pipe_producer_state = mainloop_producer_state_next;
        load2transform_pipe_producer_state = load2transform_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }

        auto [mainloop_producer_state_next_, load2transform_producer_state_next_, unused_] = collective_mainloop.load(
          mainloop_pipeline,
          load2transform_pipeline,
          mainloop_pipe_producer_state,
          load2transform_pipe_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter_next, k_tile_count - k_tile_prologue
        );
        mainloop_pipe_producer_state = mainloop_producer_state_next_;
        load2transform_pipe_producer_state = load2transform_producer_state_next_;

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());

      collective_mainloop.load_tail(
        mainloop_pipeline, 
        load2transform_pipeline, 
        mainloop_pipe_producer_state, 
        load2transform_pipe_producer_state
      );
      
    }

    else if (is_participant.sched) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      if constexpr (IsSchedDynamicPersistent) {

        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info,
            clc_pipeline,
            clc_pipe_consumer_state
          );

          // Only perform a new CLC query if we consumed a new CLC query result in
          // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
          // not consume a new CLC query response is when processing stream-K units.
          // The current stream-K scheduler uses single WorkTileInfo to track multiple
          // (potentially-partial) tiles to be computed via stream-K. In this case,
          // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
          // rather than consuming a CLC query response.
          requires_clc_query = increment_pipe;
          if (increment_pipe) {
            ++clc_pipe_consumer_state;
          }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipe_producer_state);

      }
    }

    else if (is_participant.mma) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Tmem allocation sequence
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      auto mma_inputs = collective_mainloop.mma_init(
        tmem_storage,
        shared_storage.tensors.mainloop);

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      do {

        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        if (is_mma_leader_cta) {
          auto [mainloop_pipe_consumer_state_, mma2transform_pipe_producer_state_] = collective_mainloop.mma(
            cute::make_tuple(mainloop_pipeline, mma2transform_pipeline),
            cute::make_tuple(mainloop_pipe_consumer_state, mma2transform_pipe_producer_state),
            tmem_storage,
            mma_inputs,
            cta_coord_mnkl,
            k_tile_count
          );
          mainloop_pipe_consumer_state = mainloop_pipe_consumer_state_;
          mma2transform_pipe_producer_state = mma2transform_pipe_producer_state_;
        }

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      // Leader MMA waits for leader + peer epilogues to release stage
      if (is_mma_leader_cta) {
        mma2transform_pipeline.producer_tail(mma2transform_pipe_producer_state);
      }
      // Signal to peer MMA that entire tmem allocation can be deallocated
      if constexpr (has_mma_peer_cta) {
        // Leader does wait + arrive, follower does arrive + wait
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
      }
 
      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();
      
      bool do_load_order_wait = true;
      bool do_tail_load = false;
      int current_wave = 0;

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      do {
        bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

        // Get current work tile and fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        if (compute_epilogue) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);
          epi_load_pipe_producer_state = collective_epilogue.template load<IsOverlappingAccum>(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue,
            reverse_epi_n
          );

          do_tail_load = true;
        }
        current_wave++;

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_load) {
        collective_epilogue.load_tail(
          epi_load_pipeline, epi_load_pipe_producer_state,
          epi_store_pipeline, epi_store_pipe_producer_state);
      }
    }

    else if (is_participant.epilogue) {
      // Register reconfiguration
      arch::warpgroup_reg_alloc<AccumRegisterRequirement>();

      // Throttle the epilogue warps to improve prologue performance
      static constexpr int epilogue_throttle_phase_bit = 0;
      epilogue_throttle_barrier.wait(epilogue_throttle_phase_bit);
      
      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      auto transform_inputs = collective_mainloop.transform_init(
        problem_shape_MNKL, 
        shared_storage.tensors.mainloop
      );

      auto pipelines = cute::make_tuple(mma2transform_pipeline, load2transform_pipeline);
      auto states = cute::make_tuple(mma2transform_pipe_consumer_state, load2transform_pipe_consumer_state);
      bool do_tail_store = false;
      do {

        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        auto [accum, tiled_t2r, next_state] = collective_mainloop.transform(
          pipelines,
          states,
          tmem_storage,
          transform_inputs,
          cta_coord_mnkl,
          typename CollectiveEpilogue::CopyOpT2R{},
          typename CollectiveEpilogue::EpilogueTile{},
          k_tile_count
        );

        states = next_state;

        auto fixup_next_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accum,
          get<0>(pipelines),
          get<0>(next_state),
          typename CollectiveEpilogue::CopyOpT2R{}
        );

        get<0>(states) = fixup_next_state;

        //
        // Epilogue and write to gD
        //
        if (scheduler.compute_epilogue(work_tile_info)) {
            auto [load_state_next, store_state_next] = collective_epilogue.store(
              epi_load_pipeline,
              epi_load_pipe_consumer_state,
              epi_store_pipeline,
              epi_store_pipe_producer_state,
              problem_shape_MNKL,
              CtaShape_MNK{},
              cta_coord_mnkl,
              TileShape{},
              TiledMma{},
              accum,
              shared_storage.tensors.epilogue,
              tiled_t2r
            );
            epi_load_pipe_consumer_state = load_state_next;
            epi_store_pipe_producer_state = store_state_next;
            do_tail_store = true;
        }
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

      } while (work_tile_info.is_valid());

      // Only perform a tail store if one of the work units processed performed
      // an epilogue. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_store) {
        collective_epilogue.store_tail(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          CtaShape_MNK{});
      }
    } else {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
""",
    'key-files/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>>
>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>);

  static constexpr bool IsGdcEnabled = false;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using Schedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);
  static_assert(cute::is_void_v<TileScheduler_>,
    "Ptr-Array Cooperative and Grouped Gemm Cooperative kernel only supports the default scheduler.");

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  using TileScheduler = cute::conditional_t<IsGroupedGemmKernel,
    typename detail::TileSchedulerSelector<
      GroupScheduler, ArchTag,
      TileShape, ClusterShape,
      2, // Default unused parameter - SchedulerPipelineStageCount
      ProblemShape>::Scheduler,
    typename detail::TileSchedulerSelector<
    void, ArchTag, TileShape, ClusterShape>::Scheduler>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaThreads = size(TiledMma{});
  static constexpr uint32_t NumMmaWarpGroups = NumMmaThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;

      alignas(128) MainloopTensorMapStorage mainloop;
      alignas(128) EpilogueTensorMapStorage epilogue;
    } tensormaps;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveMainloop::initialize_workspace(args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape.get_host_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    return grid_shape;
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(size(TiledMma{}) == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");
    static_assert(NumMmaWarpGroups == 2, "Cooperative kernels currently only support NumMmaWarpGroups == 2");

    if constexpr (cutlass::epilogue::collective::detail::sm90_is_ptr_array_tma_dispatch_policy_v<typename CollectiveEpilogue::DispatchPolicy>) {
      static_assert(NumMmaWarpGroups == CollectiveEpilogue::NumEpilogueWarpGroups,
                    "Tiled MmA does not match expected warp groups performing the epilogue");
    }

    static_assert(cute::rank(InternalStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_idx = canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Note: Tma Descriptor Prefetch (from either const or param) is not applicable here

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMmaThreads;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    const auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    const auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    const auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    if (not work_tile_info.is_valid()) {
      // When problem shapes are only on device, the grid launched may be larger than the total number of blocks across groups
      return;
    }

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx;
        int32_t const mock_l_coord = 0;
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        // Fetch a copy of tensormaps for the CTA
        auto input_tensormaps = collective_mainloop.tensormaps_init(params.mainloop, shared_storage.tensormaps.mainloop, sm_count, sm_idx);

        // Update tensormap for the initial batch for the CTA
        if (work_tile_info.is_valid()) {
          collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            problem_shape_MNKL,
            curr_batch
          );
          // Ensure warp is converged before issuing tensormap fence release
          __syncwarp();
          // Entire warp must do this (i.e. it's aligned)
          collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
        }

        bool do_load_order_arrive = true;
        bool did_batch_change = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (did_batch_change) {
            load_inputs = collective_mainloop.tensors_perform_update(load_inputs, params.mainloop, problem_shape_MNKL, curr_batch);
            collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
          }

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            input_tensormaps,
            blk_coord,
            k_tile_iter, work_k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          // Wait for the last TMA stage to complete loading, before issuing tensormap updates
          mainloop_pipe_producer_state.advance(work_k_tile_count - 1);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
          auto next_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx
          did_batch_change = next_batch != curr_batch;
          if (work_tile_info.is_valid() && did_batch_change) {
            curr_batch = next_batch;
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
            }
            // Purpose of this pipeline state is to make sure TMA loads have finished before doing descriptor updates
            // Since this state is waiting for loads to finish, it must start in the inverted phase.
            typename CollectiveMainloop::PipelineState mainloop_pipe_tma_consumer_state =
              {mainloop_pipe_producer_state.index(), !mainloop_pipe_producer_state.phase(), mainloop_pipe_producer_state.count()};
            mainloop_pipeline.consumer_wait(mainloop_pipe_tma_consumer_state);
            collective_mainloop.tensormaps_perform_update(
              shared_storage.tensormaps.mainloop,
              params.mainloop,
              input_tensormaps,
              problem_shape_MNKL,
              curr_batch
            );
            // Ensure warp is converged before issuing tensor replace
            __syncwarp();
            // Entire warp must do this (i.e. it's aligned)
            collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
          }
          // Advance the producer state for the last remaining stage that was being waited for above
          mainloop_pipe_producer_state.advance(1);
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        auto epi_load_tensormap = get<0>(collective_epilogue.load_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx));

        bool did_batch_change = true;
        constexpr bool IsEpiLoad = true;

        if (work_tile_info.is_valid()) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_load_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            0
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
        }

        load_order_barrier.wait();

        while (work_tile_info.is_valid()) {
          int32_t curr_batch = work_tile_info.L_idx;

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);

          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            if (did_batch_change) {
              collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_load_tensormap);
            }

            bool wait = work_tile_info.is_valid() && curr_batch != next_work_tile_info.L_idx;

            epi_load_pipe_producer_state = collective_epilogue.load(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord,
              tiled_mma,
              lane_idx,
              shared_storage.tensors.epilogue,
              epi_load_tensormap,
              work_tile_info.reduction_subtile_idx(),
              wait
            );
          }

          work_tile_info = next_work_tile_info;
          did_batch_change = curr_batch != work_tile_info.L_idx;

          if (work_tile_info.is_valid() && did_batch_change) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // tensormap update
            {
              collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                shared_storage.tensormaps.epilogue,
                params.epilogue,
                epi_load_tensormap,
                problem_shape_MNKL,
                work_tile_info.L_idx,
                0
              );

              // Converge before issuing tensormap fence release since fence is aligned
              __syncwarp();
              collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
            }
          }

        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx = warp_group_role == WarpGroupRole::Consumer0 ? 0 : 1;

      int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
      int32_t const sm_count = params.hw_info.sm_count;
      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      // Get a copy of tensormaps
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx, consumer_warp_group_idx));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;

      if (work_tile_info.is_valid()) {

        if (warp_idx_in_warp_group == 0) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_store_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            consumer_warp_group_idx
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, 
                                                                     epi_store_tensormap, 
                                                                     consumer_warp_group_idx);
        }
      }

      while (work_tile_info.is_valid()) {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }

        int32_t curr_batch = work_tile_info.L_idx;

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        static_assert(cute::is_any_of_v<TileScheduler,
            detail::PersistentTileSchedulerSm90Group<ProblemShape>,
            detail::PersistentTileSchedulerSm90>);
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {

          collective_mainloop.mma(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            params.mainloop
          );

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            work_k_tile_count
          );

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // Perform reduction across splits, if needed
        TileScheduler::fixup(
          params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (did_batch_change) {
          collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
        }

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {

          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
          collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            accumulators,
            tiled_mma,
            mma_thread_idx,
            shared_storage.tensors.epilogue,
            epi_store_tensormap,
            work_tile_info.reduction_subtile_idx()
          );

          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = next_work_tile_info;

        did_batch_change = curr_batch != work_tile_info.L_idx;
        if (work_tile_info.is_valid() && did_batch_change) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          if (warp_idx_in_warp_group == 0) {
            collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_store_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              consumer_warp_group_idx
            );

            // Converge before issuing tensormap fence release since fence is aligned
            __syncwarp();
            collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, 
                                                                       epi_store_tensormap,
                                                                       consumer_warp_group_idx);
          }
        }

      } // Scheduler work fetch loop

      // Cooperative only needs TMA to complete at the very end of the kernel
      if (do_store_tail) {
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state
        );
      }
    } // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
""",
    'key-files/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, typename CollectiveMainloop_::DispatchPolicy::Schedule>>
>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedPingpong, typename CollectiveMainloop_::DispatchPolicy::Schedule>);

  static constexpr bool IsGdcEnabled = false;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using Schedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);
  static_assert(cute::is_void_v<TileScheduler_>,
    "Ptr-Array Pingpong and Grouped Gemm Pingpong kernel only supports the default scheduler.");

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  using TileScheduler = cute::conditional_t<IsGroupedGemmKernel,
    typename detail::TileSchedulerSelector<
      GroupScheduler, ArchTag,
      TileShape, ClusterShape,
      2, // Default unused parameter - SchedulerPipelineStageCount
      ProblemShape>::Scheduler,
    typename detail::TileSchedulerSelector<
    void, ArchTag, TileShape, ClusterShape>::Scheduler>;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumMmaWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup, NumMmaWarpGroups>;
  using MathWarpGroupOrderBarrierSharedStorage = cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
      MathWarpGroupOrderBarrier::SequenceDepth,
      MathWarpGroupOrderBarrier::SequenceLength>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
    } pipelines;

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;

      alignas(128) MainloopTensorMapStorage mainloop;
      alignas(128) EpilogueTensorMapStorage epilogue;
    } tensormaps;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles);
    }

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveMainloop::initialize_workspace(args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }
    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape.get_host_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    return grid_shape;
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(size(TiledMma{}) == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
    static_assert(NumMmaWarpGroups == 2, "Pingpong kernels currently only support NumMmaWarpGroups == 2");

    if constexpr (cutlass::epilogue::collective::detail::sm90_is_ptr_array_tma_dispatch_policy_v<typename CollectiveEpilogue::DispatchPolicy>) {
      static_assert(NumMmaWarpGroups == CollectiveEpilogue::NumEpilogueWarpGroups,
                    "Tiled MmA does not match expected warp groups performing the epilogue");
    }

    static_assert(cute::rank(InternalStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_idx = canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Note: Tma Descriptor Prefetch (from either const or param) is not applicable here

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = warp_group_idx - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    const auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    const auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    const auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    if (not work_tile_info.is_valid()) {
      // When problem shapes are only on device, the grid launched may be larger than the total number of blocks across groups
      return;
    }

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    if (warp_group_role == WarpGroupRole::Consumer1) {
      // Advance 2nd Math WG to the next work tile for the startup
      const auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = next_work_tile_info;
      if (!work_tile_info.is_valid()) {
        return;
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);

      problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
    }

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx;
        int32_t const mock_l_coord = 0;
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        // Fetch a copy of tensormaps for the CTA
        auto input_tensormaps = collective_mainloop.tensormaps_init(params.mainloop, shared_storage.tensormaps.mainloop, sm_count, sm_idx);

        // Update tensormap for the initial batch for the CTA
        if (work_tile_info.is_valid()) {
          collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            problem_shape_MNKL,
            curr_batch
          );
          // Ensure warp is converged before issuing tensormap fence release
          __syncwarp();
          // Entire warp must do this (i.e. it's aligned)
          collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
        }

        bool do_load_order_arrive = true;
        bool did_batch_change = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (did_batch_change) {
            load_inputs = collective_mainloop.tensors_perform_update(load_inputs, params.mainloop, problem_shape_MNKL, curr_batch);
            collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
          }

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            input_tensormaps,
            blk_coord,
            k_tile_iter, work_k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          // Wait for the last TMA stage to complete loading, before issuing tensormap updates
          mainloop_pipe_producer_state.advance(work_k_tile_count - 1);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
          auto next_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx
          did_batch_change = next_batch != curr_batch;
          if (work_tile_info.is_valid() && did_batch_change) {
            curr_batch = next_batch;
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
            }
            // Purpose of this pipeline state is to make sure TMA loads have finished before doing descriptor updates
            // Since this state is waiting for loads to finish, it must start in the inverted phase.
            typename CollectiveMainloop::PipelineState mainloop_pipe_tma_consumer_state =
              {mainloop_pipe_producer_state.index(), !mainloop_pipe_producer_state.phase(), mainloop_pipe_producer_state.count()};
            mainloop_pipeline.consumer_wait(mainloop_pipe_tma_consumer_state);
            collective_mainloop.tensormaps_perform_update(
              shared_storage.tensormaps.mainloop,
              params.mainloop,
              input_tensormaps,
              problem_shape_MNKL,
              curr_batch
            );
            // Ensure warp is converged before issuing tensor replace
            __syncwarp();
            // Entire warp must do this (i.e. it's aligned)
            collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
          }
          // Advance the producer state for the last remaining stage that was being waited for above
          mainloop_pipe_producer_state.advance(1);
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        auto epi_load_tensormap = get<0>(collective_epilogue.load_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx));

        bool did_batch_change = true;
        constexpr bool IsEpiLoad = true;

        if (work_tile_info.is_valid()) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_load_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            0
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
        }

        load_order_barrier.wait();

        while (work_tile_info.is_valid()) {
          int32_t curr_batch = work_tile_info.L_idx;

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);

          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            if (did_batch_change) {
              collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_load_tensormap);
            }

            bool wait = work_tile_info.is_valid() && curr_batch != next_work_tile_info.L_idx;

            epi_load_pipe_producer_state = collective_epilogue.load(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord,
              tiled_mma,
              lane_idx,
              shared_storage.tensors.epilogue,
              epi_load_tensormap,
              work_tile_info.reduction_subtile_idx(),
              wait
            );
          }

          work_tile_info = next_work_tile_info;
          did_batch_change = curr_batch != work_tile_info.L_idx;

          if (work_tile_info.is_valid() && did_batch_change) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // tensormap update
            {
              collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                shared_storage.tensormaps.epilogue,
                params.epilogue,
                epi_load_tensormap,
                problem_shape_MNKL,
                work_tile_info.L_idx,
                0
              );

              // Converge before issuing tensormap fence release since fence is aligned
              __syncwarp();
              collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
            }
          }

        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx = warp_group_role == WarpGroupRole::Consumer0 ? 0 : 1;

      int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
      int32_t const sm_count = params.hw_info.sm_count;
      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      // Get a copy of tensormaps
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx, consumer_warp_group_idx));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;

      if (work_tile_info.is_valid()) {

        if (warp_idx_in_warp_group == 0) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_store_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            consumer_warp_group_idx
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                     epi_store_tensormap,
                                                                     consumer_warp_group_idx);
        }
      }

      while (work_tile_info.is_valid()) {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }

        int32_t curr_batch = work_tile_info.L_idx;

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        static_assert(cute::is_any_of_v<TileScheduler,
            detail::PersistentTileSchedulerSm90Group<ProblemShape>,
            detail::PersistentTileSchedulerSm90>);
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {

          math_wg_order_barrier.wait();

          collective_mainloop.mma(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            params.mainloop
          );

          math_wg_order_barrier.arrive();

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            work_k_tile_count
          );

           math_wg_order_barrier.wait();

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // Perform reduction across splits, if needed
        TileScheduler::fixup(
          params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (did_batch_change) {
          collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
        }

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {

          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
          collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            accumulators,
            tiled_mma,
            mma_thread_idx,
            shared_storage.tensors.epilogue,
            epi_store_tensormap,
            work_tile_info.reduction_subtile_idx()
          );

          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = next_work_tile_info;

        // Skip a tile for pingpong
        if (work_tile_info.is_valid()) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          mainloop_pipe_consumer_state.advance(work_k_tile_count);

          // Go to next tile
          auto [next_next_work_tile_info, next_increment_pipe] = scheduler.fetch_next_work(work_tile_info);

          work_tile_info = next_next_work_tile_info;
          increment_pipe = next_increment_pipe;
        }

        did_batch_change = curr_batch != work_tile_info.L_idx;
        if (work_tile_info.is_valid() && did_batch_change) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          if (warp_idx_in_warp_group == 0) {
            collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_store_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              consumer_warp_group_idx
            );

            // Converge before issuing tensormap fence release since fence is aligned
            __syncwarp();
            collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                       epi_store_tensormap,
                                                                       consumer_warp_group_idx);
          }
        }

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer kernels
        // we need to wait for all TMA stores to complete before issuing consumer order barrier arrives
        // to ensure next math consumer doesn't overwrite smem of in-flight TMA stores of current consumer.
        auto [epi_load_pipe_consumer_state_next_, epi_store_pipe_producer_state_next_] =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state
        );

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for ping pong
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();

      } // Scheduler work fetch loop
    } // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
