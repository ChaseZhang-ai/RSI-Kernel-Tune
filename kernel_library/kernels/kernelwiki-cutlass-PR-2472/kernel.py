"""KernelWiki artifact PR-2472 (NVIDIA/cutlass)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/examples/77_blackwell_fmha/77_blackwell_fmha.cu': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Example implementation of fused multi-head attention for the NVIDIA Blackwell SM100
    architecture using CUTLASS 3.

    MQA/GQA
    -------

    The head dimension can be represented as a tuple, where the K/V strides in the
    first dimension is zero. This has the effect of MQA or GQA.
    * MHA is (head_size:head_stride).
    * MQA is (head_size:head_stride) in Q and (head_size:_0) in K and V.
    * GQA is (grouped_heads,heads_kv):(head_stride,grouped_heads*head_stride) in Q
      and (grouped_heads,heads_kv):(0,head_stride) in K and V

    Output Scale
    ------------

    The output scale gets passed to the collective mainloop, and is applied
    using FP32 compute pre-quantization

    Variable Sequence Length
    ------------------------

    For variable sequence length, pass in VariableLength objects
    (max_seqlen, cumulative_seqlen_ptr) in the problem shape for
    seqlen Q and KV.

    Support
    ---------

    Right now e4m3 with fp32 compute is using a 256x256 tiling and a head dimension
    of 128 is supported.


    Example usage:
      $ ./examples/77_blackell_fmha/77_blackell_fmha_fp8 \
            --b=2048 --h=2048 --d=2048 --q=2048 --k=2048
*/

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_fwd_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  std::vector<int> varlen_q;
  std::vector<int> varlen_k;
  int d = 128;
  int warmup_iterations = 1;
  int iterations = 3;
  int tensor_ring_buffers = 1;
  bool verify = false;
  bool verbose = false;

  bool causal = false;
  bool residual = false;
  bool varlen = false;
  bool persistent = false;
  int sm_count = 0;
  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_k = InitStyle::kRandom;
  InitStyle init_style_v = InitStyle::kRandom;

  static void get_init_style_argument(cutlass::CommandLine& cmd, const char* name, InitStyle& dst, InitStyle const& src) {
    std::string s;
    cmd.get_cmd_line_argument(name, s, s);
    if (s.empty()) {
      dst = src;
    }
    else {
      if (s == "r") {
        dst = InitStyle::kRandom;
      }
      else if (s == "1") {
        dst = InitStyle::kOne;
      }
      else if (s == "d") {
        dst = InitStyle::kLinearStride1;
      }
      else if (s == "s") {
        dst = InitStyle::kLinearStride128;
      }
      else if (s == "n") {
        dst = InitStyle::kNone;
      }
      else {
        std::cout << "Error: " << s << " is not a valid input type.\n";
        std::exit(-1);
      }
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    Options defaults;

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("d", d, defaults.d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

    cmd.get_cmd_line_argument("h_k", h_k, -1);
    if (h_k == -1) h_k = h;

    varlen = cmd.check_cmd_line_flag("varlen");

    cmd.get_cmd_line_argument("q", q, -1);
    cmd.get_cmd_line_argument("k", k, -1);
    cmd.get_cmd_line_argument("b", b, -1);

    std::string varlen_q_str;
    cmd.get_cmd_line_argument("varlen-q", varlen_q_str);
    std::string varlen_k_str;
    cmd.get_cmd_line_argument("varlen-k", varlen_k_str);

    if (varlen && ! varlen_q_str.empty()) {
      varlen_q.clear();
      while (! varlen_q_str.empty()) {
        size_t pos = varlen_q_str.find(':');
        varlen_q.push_back(std::stoi(varlen_q_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_q_str = varlen_q_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_q.size());
      }
      if (b != static_cast<int>(varlen_q.size())) {
        std::cout << "Error: Invalid --varlen-q length\n";
        std::exit(-1);
      }
      int new_q = 0;
      for (auto elem : varlen_q) {
        new_q += elem;
      }
      if (q != -1) {
        std::cout << "Error: Can't provide --q and --varlen-q\n";
        std::exit(-1);
      }
      q = new_q;
    }

    if (varlen && ! varlen_k_str.empty()) {
      varlen_k.clear();
      while (! varlen_k_str.empty()) {
        size_t pos = varlen_k_str.find(':');
        varlen_k.push_back(std::stoi(varlen_k_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_k_str = varlen_k_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_k.size());
      }
      if (b != static_cast<int>(varlen_k.size())) {
        std::cout << " Error: Invalid --varlen-k length\n";
        std::exit(-1);
      }
      int new_k = 0;
      for (auto elem : varlen_k) {
        new_k += elem;
      }
      if (k != -1) {
        std::cout << "Error: Can't provide --k and --varlen-k\n";
        std::exit(-1);
      }
      k = new_k;
    }

    if (q == -1) q = k;
    if (k == -1) k = q;
    if (q == -1 && k == -1) q = k = defaults.q;
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, defaults.warmup_iterations);
    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    cmd.get_cmd_line_argument("tensor_ring_buffers", tensor_ring_buffers, defaults.tensor_ring_buffers);

    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    persistent = cmd.check_cmd_line_flag("persistent");

    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
      if (varlen) {
        residual = true;
      }
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
    }
    else if (mask == "residual") {
      residual = true;
      causal = false;
    }
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);
    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_k, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_v, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-k", init_style_k, init_style_k);
    get_init_style_argument(cmd, "init-style-v", init_style_v, init_style_v);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_fmha\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention forward-passkernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --h_k=<int>                 Sets the H_K/V extent (for GQA/MQA)\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --varlen-q=<int>:<int...>   Sets the variable Q extent per batch (colon separated)\n"
      << "  --varlen-k=<int>:<int...>   Sets the variable K extent per batch (colon separated)\n"
      << "  --d=<int>                   Sets the D extent\n"
      << "  --tensor_ring_buffers=<int> Sets the number of tensor ring buffers\n"
      << "  --warmup_iterations=<int>   Sets the warmup iterations\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
      << "  --persistent                Enables persistent scheduler\n"
      << "  --varlen                    Enables variable sequence length\n"
      << "                              B*Q and B*K become the total sequence length\n"
      << "                              and are split B-ways, alternatingly +10% and -10%\n"
      << "                              with the last batch sized to make it fit\n"
      << "                              implies at least residual masking for correctness\n"
      << "  --sm-count                  Sets SM count rather than querying it\n"
      << "  --kernel-filter=<filter>    Sets regexp to match kernel against\n"
      << "\n";

    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(
    DeviceAllocation<Element>& block,
    uint64_t seed=2023, InitStyle init_style = InitStyle::kRandom) {

  switch (init_style) {
    case InitStyle::kOne: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 1, (Element) 1);
      break;
    }
    case InitStyle::kRandom: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) 0, (Element) 1);
      break;
    }
    case InitStyle::kLinearStride1: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (j % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kLinearStride128: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (i % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kNone: {
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleResult {
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  double tops_exp2_s = 0;
  double tbytes_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  bool kIsVarlen,
  class TileShape,
  class DispatchPolicy,
  class ActiveMask,
  class... KernelOptions
>
struct FwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#else
  using Element = cutlass::half_t;
#endif

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  // Q K D (B H)
  using ProblemShapeRegular = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType = std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;
  
  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D (H_G H_R B)
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;  // K D (H_G H_R B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;     // Q   (H_G H_R B)

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;
  using TileScheduler = std::conditional_t<kIsPersistent, cutlass::fmha::kernel::PersistentTileScheduler, cutlass::fmha::kernel::IndividualTileScheduler>;

  using Mainloop = 
    cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      ActiveMask
    >;
  using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType,
      Mainloop,
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE
      >,
      TileScheduler
    >>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  struct DeviceBuffer {
    DeviceAllocation<Element> block_Q;
    DeviceAllocation<Element> block_K;
    DeviceAllocation<Element> block_V;
    DeviceAllocation<ElementOut> block_O;
    DeviceAllocation<ElementAccumulatorPV> block_LSE;
    DeviceAllocation<ElementOut> block_ref_O;
    DeviceAllocation<ElementAccumulatorPV> block_ref_LSE;
    DeviceAllocation<int> device_cumulative_seqlen_q;
    DeviceAllocation<int> device_cumulative_seqlen_kv;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    size_t get_storage_size() const {
      return block_Q.get_storage_size() + block_K.get_storage_size() + block_V.get_storage_size()
          + block_O.get_storage_size() + block_LSE.get_storage_size() + block_ref_O.get_storage_size()
          + block_ref_LSE.get_storage_size() + device_cumulative_seqlen_q.get_storage_size()
          + device_cumulative_seqlen_kv.get_storage_size();
    }
  };

  std::vector<std::unique_ptr<DeviceBuffer>> buffers;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;

  //
  // Methods
  //
  bool verify(const ProblemShapeType& problem_shape, DeviceBuffer& buffer) {
    Tensor mQ = make_tensor(make_gmem_ptr(buffer.block_Q.get()),
      select<0,2,3>(problem_shape),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(buffer.block_K.get()),
      select<1,2,3>(problem_shape),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(buffer.block_V.get()),
      select<1,2,3>(problem_shape),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(buffer.block_ref_O.get()),
      select<0,2,3>(problem_shape),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(buffer.block_ref_LSE.get()),
      select<0,3>(problem_shape),
      stride_LSE);

    fmha_reference(problem_shape, mQ, mK, mV, mO, mLSE, ActiveMask{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(buffer.block_O, buffer.block_ref_O, max_diff, mean_diff);

    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(buffer.block_LSE, buffer.block_ref_LSE, max_diff, mean_diff);

    bool passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_LSE;
  }

  template<class ProblemShape>
  auto initialize_varlen(
      const Options& options, const ProblemShape& problem_size,
      const bool kVarlenSame = true) {

    int num_batches = get<3,1>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative 
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<0>(problem_size), get<0>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<1>(problem_size), get<1>(problem_size) / 2);
    std::cout << "N: " << num_batches << ", Q: " << get<0>(problem_size) << ", KV: " << get<1>(problem_size) << std::endl;

    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;

    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = (! options.varlen_q.empty()) ? options.varlen_q.at(i) : 
              kVarlenSame ? get<0>(problem_size) :
              generate_positive_int(dist_q, rng);
      int seqlen_kv = (! options.varlen_k.empty()) ? options.varlen_k.at(i) :
              kVarlenSame ? get<1>(problem_size) :
              generate_positive_int(dist_kv, rng);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
    }
    std::cout << "Q max: " << max_seqlen_q << " total: " << total_seqlen_q << " vs even " << num_batches * get<0>(problem_size) << std::endl;
    std::cout << "KV max: " << max_seqlen_kv << " total: " << total_seqlen_kv << " vs even " << num_batches * get<1>(problem_size) << std::endl;

    ProblemShape problem_size_for_init = problem_size;
    get<3,1>(problem_size_for_init) = 1;
    get<0>(problem_size_for_init) = total_seqlen_q;
    get<1>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = VariableLength{max_seqlen_q};
    get<1>(problem_size_for_launch) = VariableLength{max_seqlen_kv};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }


  /// Initialize operands to be used in the GEMM and reference GEMM

  ProblemShapeType initialize(const Options& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);
    auto problem_shape_in = cute::make_tuple(options.q, options.k, options.d, cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    
    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(options, problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    get<2>(problem_size) = cutlass::round_up(get<2>(problem_size), 8);  // alignment

    auto shape_QO = select<0,2,3>(problem_size);
    auto shape_KV = select<1,2,3>(problem_size);
    auto shape_LSE = select<0,3>(problem_size);

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B = size<3,1>(problem_size);

    stride_Q = make_stride(H*D , _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    stride_O = stride_Q;
    stride_K = make_stride(H_K*D , _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    stride_V = stride_K;
    stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    if (kIsVarlen) {
      get<2,1>(stride_Q) = 0;
      get<2,1>(stride_K) = 0;
      get<2,1>(stride_V) = 0;
      get<2,1>(stride_O) = 0;
      get<1,1>(stride_LSE) = 0;
    }

    auto buffer_init_fn = [&](auto& buffer) {
      buffer.block_Q.reset(size(shape_QO), kIsVarlen ? D*SQ*H : 0);
      buffer.block_K.reset(size(shape_KV), kIsVarlen ? D*SK*H_K : 0);
      buffer.block_V.reset(size(shape_KV), kIsVarlen ? D*SK*H_K : 0);
      buffer.block_O.reset(size(shape_QO), kIsVarlen ? D*SQ*H : 0);
      buffer.block_LSE.reset(size(shape_LSE));
      buffer.block_ref_O.reset(size(shape_QO), kIsVarlen ? D*SQ*H : 0);
      buffer.block_ref_LSE.reset(size(shape_LSE));

      initialize_block(buffer.block_Q, seed + 2023, options.init_style_q);
      initialize_block(buffer.block_K, seed + 2022, options.init_style_k);
      initialize_block(buffer.block_V, seed + 2021, options.init_style_v);

      if ( ! cumulative_seqlen_q.empty()) {
        buffer.device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
        buffer.device_cumulative_seqlen_q.copy_from_host(
          cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      }
      if ( ! cumulative_seqlen_kv.empty()) {
        buffer.device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
        buffer.device_cumulative_seqlen_kv.copy_from_host(
          cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
      }   
    };

    buffers.push_back(std::make_unique<DeviceBuffer>());
    buffer_init_fn(*buffers.back());

    int tensor_ring_buffers = options.tensor_ring_buffers;
    for (int i = 1; i < tensor_ring_buffers; i++) {
      buffers.push_back(std::make_unique<DeviceBuffer>());
      buffer_init_fn(*buffers.back());
    }

    if constexpr (kIsVarlen) {
      get<0>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_kv.get();
    }

    return problem_shape;
  }

  auto get_arguments(const ProblemShapeType& problem_shape, const cutlass::KernelHardwareInfo& hw_info, int buffer_index) {
    auto problem_shape_ = problem_shape;
    if constexpr (kIsVarlen) {
      get<0>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_kv.get();
    }
    typename Operation::Arguments arguments{
      problem_shape_,
      { buffers[buffer_index]->block_Q.get(), stride_Q,
        buffers[buffer_index]->block_K.get(), stride_K,
        buffers[buffer_index]->block_V.get(), stride_V },
      { buffers[buffer_index]->block_O.get(), stride_O,
        buffers[buffer_index]->block_LSE.get(), stride_LSE },
      hw_info
    };
    return arguments;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {

    ProblemShapeType problem_shape = initialize(options);

    int buffer_index = 0;
    typename Operation::Arguments arguments = get_arguments(problem_shape, hw_info, buffer_index);

    Operation op;

    ExampleResult example_result;

    example_result.smem_size = Operation::Kernel::SharedStorageSize;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    status = op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    // Run
    for (int i = 0; i < options.warmup_iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops;
    if (kIsVarlen) {
      flops = 0.0;
      for (int i = 0; i < size<3,1>(problem_shape); i++) {
        flops += (cumulative_seqlen_q[i+1] - cumulative_seqlen_q[i])
               * 1.0
               * (cumulative_seqlen_kv[i+1] - cumulative_seqlen_kv[i]);
      }
    }
    else {
      flops = 1.0;
      flops *= static_cast<double>(size<0>(problem_shape));
      flops *= static_cast<double>(size<1>(problem_shape));
      flops *= static_cast<double>(size<3,1>(problem_shape));
    }
    flops *= 4.0 * (std::is_same_v<ActiveMask, CausalMask<true>> || std::is_same_v<ActiveMask, CausalMask<false>> ? 0.5 : 1.0);
    flops *= static_cast<double>(size<2>(problem_shape));
    flops *= static_cast<double>(size<3,0>(problem_shape));
    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_shape, *buffers[0]);
      if (passed) example_result.verified = true;
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_result = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << (result.passed ? (result.verified ? " [OK]  " : " [--] ") : "[FAIL] ");
  if (! result.passed) {
    main_result = -1;
  }
  std::cout << std::setw(32) << std::left << description;
  std::cout.copyfmt(fmt);
  std::cout << " : " << result.tflops_tc_s << " TFLOPS/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms << "ms, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_fwd_128(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    if (options.varlen) {
      FwdRunner<true, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
    else 
    {
      FwdRunner<false, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
  };

  using HeadDim = _128;

  if (options.persistent) {
    // Persistent Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 persistent", Option<Tag::kIsPersistent, true_type>{});
  }
  else {
    // Individual Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 individual", Option<Tag::kIsPersistent, false_type>{});
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_fwd_64(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    if (options.varlen) {
      FwdRunner<true, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
    else 
    {
      FwdRunner<false, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
  };

  using HeadDim = _64;

  if (options.persistent) {
    // Persistent Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 persistent", Option<Tag::kIsPersistent, true_type>{});
  }
  else {
    // Individual Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 individual", Option<Tag::kIsPersistent, false_type>{});
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_fwd_32(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if (options.varlen) {
      FwdRunner<true, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
    else {
      FwdRunner<false, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    }
  };

  using HeadDim = _32;

#ifdef FP8
  if (options.persistent) {
    // Persistent Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 persistent", Option<Tag::kIsPersistent, true_type>{});
  }
  else {
    // Individual Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 individual", Option<Tag::kIsPersistent, false_type>{});
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_single(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major != 10) {
    std::cout
      << "This example requires a GPU of NVIDIA's Blackwell Architecture "
      << "(compute capability major 10) and CUDA 12.8 or greater.\n";
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

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  if (options.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  else {
    hw_info.sm_count = options.sm_count;
  }

  std::cout << "###### B " << options.b << " H " << options.h << " H_K " << options.h_k << " Q " << options.q << " K " << options.k << " D " << options.d << " ";
  std::cout << "Forward" << " " << (options.causal ? "Causal" : (options.residual ? "Residual" : "None")) << " ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  auto with_mask = [&](auto fn) {
    if (options.causal) {
      fn(CausalMask{});
    }
    else if (options.residual) {
      fn(ResidualMask{});
    }
    else {
      fn(NoMask{});
    }
  };

  with_mask([&](auto fusion) {
    if (options.d <= 32) {
      run_fwd_32(fusion, options, hw_info);
    }
    else if (options.d <= 64) {
      run_fwd_64(fusion, options, hw_info);
    }
    else if (options.d <= 128) {
      run_fwd_128(fusion, options, hw_info);
    }
    else {
      std::cout << "No kernel instantiated for d=" << options.d << std::endl;
    }
  });
#endif

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  std::vector<std::string> full_arguments(args, args + argc);

  bool recursed = false;
  for (size_t i = 1; i < full_arguments.size(); i++) {
    if (full_arguments[i].find(',') != std::string::npos) {
      auto arg = full_arguments[i];
      size_t eq_pos = arg.find('=');
      std::string prefix = eq_pos == std::string::npos ? "" : arg.substr(0, eq_pos+1);
      std::string rest = eq_pos == std::string::npos ? arg : arg.substr(eq_pos+1);
      for (;;) {
        size_t comma_pos = rest.find(',');
        std::string current = rest.substr(0, comma_pos);
        full_arguments[i] = prefix + current;
        std::vector<const char*> next_args;
        for (auto& elem : full_arguments) { next_args.push_back(elem.data()); }
        main(argc, next_args.data());
        if (comma_pos == std::string::npos) break;
        rest = rest.substr(comma_pos+1);
      }
      recursed = true;
      break;
    }
  }

  if (! recursed) {
    main_single(argc, args);
  }

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_fwd_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int h = 1;
  int h_k = 1;
  int q = 256;
  int k = 256;
  std::vector<int> varlen_q;
  std::vector<int> varlen_k;
  int dl = 128; // headdim latent
  int dr = 64;  // headdim rope
  int warmup_iterations = 1;
  int iterations = 3;
  int tensor_ring_buffers = 1;
  bool verify = false;
  bool verbose = false;

  bool causal = false;
  bool residual = false;
  bool varlen = false;
  bool persistent = false;
  int sm_count = 0;
  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_k = InitStyle::kRandom;
  InitStyle init_style_v = InitStyle::kRandom;

  static void get_init_style_argument(cutlass::CommandLine& cmd, const char* name, InitStyle& dst, InitStyle const& src) {
    std::string s;
    cmd.get_cmd_line_argument(name, s, s);
    if (s.empty()) {
      dst = src;
    }
    else {
      if (s == "r") {
        dst = InitStyle::kRandom;
      }
      else if (s == "1") {
        dst = InitStyle::kOne;
      }
      else if (s == "d") {
        dst = InitStyle::kLinearStride1;
      }
      else if (s == "s") {
        dst = InitStyle::kLinearStride128;
      }
      else if (s == "n") {
        dst = InitStyle::kNone;
      }
      else {
        std::cout << "Error: " << s << " is not a valid input type.\n";
        std::exit(-1);
      }
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    Options defaults;

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("dl", dl, defaults.dl);
    cmd.get_cmd_line_argument("dr", dr, defaults.dr);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / dl;

    cmd.get_cmd_line_argument("h_k", h_k, -1);
    if (h_k == -1) h_k = h;

    varlen = cmd.check_cmd_line_flag("varlen");

    cmd.get_cmd_line_argument("q", q, -1);
    cmd.get_cmd_line_argument("k", k, -1);
    cmd.get_cmd_line_argument("b", b, -1);

    std::string varlen_q_str;
    cmd.get_cmd_line_argument("varlen-q", varlen_q_str);
    std::string varlen_k_str;
    cmd.get_cmd_line_argument("varlen-k", varlen_k_str);

    if (varlen && ! varlen_q_str.empty()) {
      varlen_q.clear();
      while (! varlen_q_str.empty()) {
        size_t pos = varlen_q_str.find(':');
        varlen_q.push_back(std::stoi(varlen_q_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_q_str = varlen_q_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_q.size());
      }
      if (b != static_cast<int>(varlen_q.size())) {
        std::cout << "Error: Invalid --varlen-q length\n";
        std::exit(-1);
      }
      int new_q = 0;
      for (auto elem : varlen_q) {
        new_q += elem;
      }
      if (q != -1) {
        std::cout << "Error: Can't provide --q and --varlen-q\n";
        std::exit(-1);
      }
      q = new_q;
    }

    if (varlen && ! varlen_k_str.empty()) {
      varlen_k.clear();
      while (! varlen_k_str.empty()) {
        size_t pos = varlen_k_str.find(':');
        varlen_k.push_back(std::stoi(varlen_k_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_k_str = varlen_k_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_k.size());
      }
      if (b != static_cast<int>(varlen_k.size())) {
        std::cout << " Error: Invalid --varlen-k length\n";
        std::exit(-1);
      }
      int new_k = 0;
      for (auto elem : varlen_k) {
        new_k += elem;
      }
      if (k != -1) {
        std::cout << "Error: Can't provide --k and --varlen-k\n";
        std::exit(-1);
      }
      k = new_k;
    }

    if (q == -1) q = k;
    if (k == -1) k = q;
    if (q == -1 && k == -1) q = k = defaults.q;
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, defaults.warmup_iterations);
    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    cmd.get_cmd_line_argument("tensor_ring_buffers", tensor_ring_buffers, defaults.tensor_ring_buffers);

    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    persistent = cmd.check_cmd_line_flag("persistent");

    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
      if (varlen) {
        residual = true;
      }
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
    }
    else if (mask == "residual") {
      residual = true;
      causal = false;
    }
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);
    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_k, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_v, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-k", init_style_k, init_style_k);
    get_init_style_argument(cmd, "init-style-v", init_style_v, init_style_v);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_mla_fwd\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head latent attention forward-passkernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --h_k=<int>                 Sets the H_K/V extent (for GQA/MQA)\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --varlen-q=<int>:<int...>   Sets the variable Q extent per batch (colon separated)\n"
      << "  --varlen-k=<int>:<int...>   Sets the variable K extent per batch (colon separated)\n"
      << "  --dl=<int>                  Sets the D latent extent\n"
      << "  --dr=<int>                  Sets the D rope extent\n"
      << "  --tensor_ring_buffers=<int> Sets the number of tensor ring buffers\n"
      << "  --warmup_iterations=<int>   Sets the warmup iterations\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
      << "  --persistent                Enables persistent scheduler\n"
      << "  --varlen                    Enables variable sequence length\n"
      << "                              B*Q and B*K become the total sequence length\n"
      << "                              and are split B-ways, alternatingly +10% and -10%\n"
      << "                              with the last batch sized to make it fit\n"
      << "                              implies at least residual masking for correctness\n"
      << "  --sm-count                  Sets SM count rather than querying it\n"
      << "  --kernel-filter=<filter>    Sets regexp to match kernel against\n"
      << "\n";

    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(
    DeviceAllocation<Element>& block,
    uint64_t seed=2023, InitStyle init_style = InitStyle::kRandom) {

  switch (init_style) {
    case InitStyle::kOne: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 1, (Element) 1);
      break;
    }
    case InitStyle::kRandom: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) 0, (Element) 1);
      break;
    }
    case InitStyle::kLinearStride1: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (j % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kLinearStride128: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (i % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kNone: {
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleResult {
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  double tops_exp2_s = 0;
  double tbytes_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  bool kIsMaskTileSchedulerValid,
  bool kIsVarlen,
  class TileShape,
  class DispatchPolicy,
  class ActiveMask,
  class... KernelOptions
>
struct MlaFwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#else
  using Element = cutlass::half_t;
#endif

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  // Q K (D_latent D_rope) (H B)
  using ProblemShapeRegular = cute::tuple<int, int, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<VariableLength, VariableLength, cute::tuple<int, int>, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType = std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;
  
  using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D (H_G H_R B)
  using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;  // K D (H_G H_R B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;     // Q   (H_G H_R B)

  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, true_type, KernelOptions...>::value;
  using TileScheduler = std::conditional_t<kIsPersistent, 
                                          std::conditional_t<std::is_same_v<ActiveMask, CausalMask<false>> 
                                                                          || std::is_same_v<ActiveMask, CausalMask<true>>, 
                                                            cutlass::fmha::kernel::CausalPersistentTileScheduler,
                                                            cutlass::fmha::kernel::PersistentTileScheduler>,
                                          std::conditional_t<kIsMaskTileSchedulerValid, 
                                                            cutlass::fmha::kernel::CausalIndividualTileScheduler,
                                                            cutlass::fmha::kernel::IndividualTileScheduler>>;

  static constexpr bool IsOrderLoadEpilogue = kIsPersistent && (sizeof(Element) == sizeof(ElementOut));
  using OrderLoadEpilogue = std::conditional_t<IsOrderLoadEpilogue, true_type, false_type>;

  using Mainloop = 
    cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      ActiveMask, Shape<_2, _1, _1>, OrderLoadEpilogue
    >;
  using Operation = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShapeType,
      Mainloop,
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        ElementOut, ElementAccumulatorPV,
        typename Mainloop::TileShapePV,
        StrideO, StrideLSE, OrderLoadEpilogue
      >,
      TileScheduler,
      cutlass::fmha::kernel::Sm100MlaFwdCtxKernelWarpspecializedSchedule
    >>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  struct DeviceBuffer {
    DeviceAllocation<Element> block_Q;
    DeviceAllocation<Element> block_K;
    DeviceAllocation<Element> block_V;
    DeviceAllocation<ElementOut> block_O;
    DeviceAllocation<ElementAccumulatorPV> block_LSE;
    DeviceAllocation<ElementOut> block_ref_O;
    DeviceAllocation<ElementAccumulatorPV> block_ref_LSE;
    DeviceAllocation<int> device_cumulative_seqlen_q;
    DeviceAllocation<int> device_cumulative_seqlen_kv;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    size_t get_storage_size() const {
      return block_Q.get_storage_size() + block_K.get_storage_size() + block_V.get_storage_size()
          + block_O.get_storage_size() + block_LSE.get_storage_size() + block_ref_O.get_storage_size()
          + block_ref_LSE.get_storage_size() + device_cumulative_seqlen_q.get_storage_size()
          + device_cumulative_seqlen_kv.get_storage_size();
    }
  };

  std::vector<std::unique_ptr<DeviceBuffer>> buffers;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;

  //
  // Methods
  //
  bool verify(const ProblemShapeType& problem_shape, DeviceBuffer& buffer) {
    int D_latent_rope = size<2, 0>(problem_shape) + size<2, 1>(problem_shape);
    Tensor mQ = make_tensor(make_gmem_ptr(buffer.block_Q.get()),
      replace<1>(select<0,2,3>(problem_shape), D_latent_rope),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(buffer.block_K.get()),
      replace<1>(select<1,2,3>(problem_shape), D_latent_rope),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(buffer.block_V.get()),
      replace<1>(select<1,2,3>(problem_shape), get<2, 0>(problem_shape)),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(buffer.block_ref_O.get()),
      replace<1>(select<0,2,3>(problem_shape), get<2, 0>(problem_shape)),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(buffer.block_ref_LSE.get()),
      select<0,3>(problem_shape),
      stride_LSE);

    fmha_reference(problem_shape, mQ, mK, mV, mO, mLSE, ActiveMask{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(buffer.block_O, buffer.block_ref_O, max_diff, mean_diff);

    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(buffer.block_LSE, buffer.block_ref_LSE, max_diff, mean_diff);

    bool passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_LSE;
  }

  template<class ProblemShape>
  auto initialize_varlen(
      const Options& options, const ProblemShape& problem_size,
      const bool kVarlenSame = true) {

    int num_batches = get<3,1>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative 
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<0>(problem_size), get<0>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<1>(problem_size), get<1>(problem_size) / 2);
    std::cout << "N: " << num_batches << ", Q: " << get<0>(problem_size) << ", KV: " << get<1>(problem_size) << std::endl;

    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;

    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = (! options.varlen_q.empty()) ? options.varlen_q.at(i) : 
              kVarlenSame ? get<0>(problem_size) :
              generate_positive_int(dist_q, rng);
      int seqlen_kv = (! options.varlen_k.empty()) ? options.varlen_k.at(i) :
              kVarlenSame ? get<1>(problem_size) :
              generate_positive_int(dist_kv, rng);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
    }
    std::cout << "Q max: " << max_seqlen_q << " total: " << total_seqlen_q << " vs even " << num_batches * get<0>(problem_size) << std::endl;
    std::cout << "KV max: " << max_seqlen_kv << " total: " << total_seqlen_kv << " vs even " << num_batches * get<1>(problem_size) << std::endl;

    ProblemShape problem_size_for_init = problem_size;
    get<3,1>(problem_size_for_init) = 1;
    get<0>(problem_size_for_init) = total_seqlen_q;
    get<1>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = VariableLength{max_seqlen_q};
    get<1>(problem_size_for_launch) = VariableLength{max_seqlen_kv};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }


  /// Initialize operands to be used in the GEMM and reference GEMM

  ProblemShapeType initialize(const Options& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);
    auto problem_shape_in = cute::make_tuple(options.q, options.k, cute::make_tuple(options.dl, options.dr), cute::make_tuple(cute::make_tuple(h_r, options.h_k), options.b));
    
    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(options, problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    int D_latent_rope = size<2, 0>(problem_shape) + size<2, 1>(problem_shape);
    auto shape_Q = replace<1>(select<0,2,3>(problem_shape), D_latent_rope);
    auto shape_K = replace<1>(select<1,2,3>(problem_shape), D_latent_rope);

    auto shape_O = replace<1>(select<0,2,3>(problem_shape), get<2, 0>(problem_shape));
    auto shape_V = replace<1>(select<1,2,3>(problem_shape), get<2, 0>(problem_shape));

    auto shape_LSE = select<0,3>(problem_size);

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2, 0>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B = size<3,1>(problem_size);

    stride_Q = make_stride(H*D_latent_rope , _1{}, make_stride(make_stride(D_latent_rope, H_Q*D_latent_rope), H*D_latent_rope*SQ));
    stride_O = make_stride(H*D , _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    stride_K = make_stride(H_K*D_latent_rope , _1{}, make_stride(make_stride(_0{}, D_latent_rope), H_K*D_latent_rope*SK));
    stride_V = make_stride(H_K*D , _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    if (kIsVarlen) {
      get<2,1>(stride_Q) = 0;
      get<2,1>(stride_K) = 0;
      get<2,1>(stride_V) = 0;
      get<2,1>(stride_O) = 0;
      get<1,1>(stride_LSE) = 0;
    }

    auto buffer_init_fn = [&](auto& buffer) {
      buffer.block_Q.reset(size(shape_Q), kIsVarlen ? D_latent_rope*SQ*H : 0);
      buffer.block_K.reset(size(shape_K), kIsVarlen ? D_latent_rope*SK*H_K : 0);
      buffer.block_V.reset(size(shape_V), kIsVarlen ? D*SK*H_K : 0);
      buffer.block_O.reset(size(shape_O), kIsVarlen ? D*SQ*H : 0);
      buffer.block_LSE.reset(size(shape_LSE));
      buffer.block_ref_O.reset(size(shape_O), kIsVarlen ? D*SQ*H : 0);
      buffer.block_ref_LSE.reset(size(shape_LSE));

      initialize_block(buffer.block_Q, seed + 2023, options.init_style_q);
      initialize_block(buffer.block_K, seed + 2022, options.init_style_k);
      initialize_block(buffer.block_V, seed + 2021, options.init_style_v);

      if ( ! cumulative_seqlen_q.empty()) {
        buffer.device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
        buffer.device_cumulative_seqlen_q.copy_from_host(
          cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      }
      if ( ! cumulative_seqlen_kv.empty()) {
        buffer.device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
        buffer.device_cumulative_seqlen_kv.copy_from_host(
          cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
      }   
    };

    buffers.push_back(std::make_unique<DeviceBuffer>());
    buffer_init_fn(*buffers.back());

    int tensor_ring_buffers = options.tensor_ring_buffers;
    for (int i = 1; i < tensor_ring_buffers; i++) {
      buffers.push_back(std::make_unique<DeviceBuffer>());
      buffer_init_fn(*buffers.back());
    }

    if constexpr (kIsVarlen) {
      get<0>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape).cumulative_length = buffers[0]->device_cumulative_seqlen_kv.get();
    }

    return problem_shape;
  }

  auto get_arguments(const ProblemShapeType& problem_shape, const cutlass::KernelHardwareInfo& hw_info, int buffer_index) {
    auto problem_shape_ = problem_shape;
    if constexpr (kIsVarlen) {
      get<0>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_q.get();
      get<1>(problem_shape_).cumulative_length = buffers[buffer_index]->device_cumulative_seqlen_kv.get();
    }
    typename Operation::Arguments arguments{
      problem_shape_,
      { buffers[buffer_index]->block_Q.get(), stride_Q,
        buffers[buffer_index]->block_K.get(), stride_K,
        buffers[buffer_index]->block_V.get(), stride_V },
      { buffers[buffer_index]->block_O.get(), stride_O,
        buffers[buffer_index]->block_LSE.get(), stride_LSE },
      hw_info
    };
    return arguments;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {

    ProblemShapeType problem_shape = initialize(options);

    int buffer_index = 0;
    typename Operation::Arguments arguments = get_arguments(problem_shape, hw_info, buffer_index);

    Operation op;

    ExampleResult example_result;

    example_result.smem_size = Operation::Kernel::SharedStorageSize;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    status = op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    // Run
    for (int i = 0; i < options.warmup_iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }
      buffer_index = (buffer_index + 1) % buffers.size();
      arguments = get_arguments(problem_shape, hw_info, buffer_index);
      status = op.update(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to update the CUTLASS kernel's parameters. Last CUDA error is: "
                  << std::endl;
        return example_result;
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops;
    if (kIsVarlen) {
      flops = 0.0;
      for (int i = 0; i < size<3,1>(problem_shape); i++) {
        flops += (cumulative_seqlen_q[i+1] - cumulative_seqlen_q[i])
               * 1.0
               * (cumulative_seqlen_kv[i+1] - cumulative_seqlen_kv[i]);
      }
    }
    else {
      flops = 1.0;
      flops *= static_cast<double>(size<0>(problem_shape));
      flops *= static_cast<double>(size<1>(problem_shape));
      flops *= static_cast<double>(size<3,1>(problem_shape));
    }

    flops *= 2.0 * (std::is_same_v<ActiveMask, CausalMask<false>> ? 0.5 : 1.0);
    flops *= static_cast<double>(size<3,0>(problem_shape));

    double flops0 = flops * static_cast<double>(size<2, 0>(problem_shape) + size<2, 1>(problem_shape));
    double flops1 = flops * static_cast<double>(size<2, 0>(problem_shape));
    flops = flops0 + flops1;

    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_shape, *buffers[0]);
      if (passed) example_result.verified = true;
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_result = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << (result.passed ? (result.verified ? " [OK]  " : " [--] ") : "[FAIL] ");
  if (! result.passed) {
    main_result = -1;
  }
  std::cout << std::setw(32) << std::left << description;
  std::cout.copyfmt(fmt);
  std::cout << " : " << result.tflops_tc_s << " TFLOPS/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms << "ms, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_prefill_mla_fwd(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    if (options.varlen) {
      if(options.h % cutlass::fmha::kernel::CausalIndividualTileScheduler::TileH == 0 && (!std::is_same_v<Mask, NoMask>)) {
        MlaFwdRunner<true, true, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
        auto result = runner.run(options, hw_info);
        print_result(name, result, options.verbose);
      } else {
        MlaFwdRunner<false, true, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
        auto result = runner.run(options, hw_info);
        print_result(name, result, options.verbose);
      }
    }
    else 
    {
      if(options.h % cutlass::fmha::kernel::CausalIndividualTileScheduler::TileH == 0 && (!std::is_same_v<Mask, NoMask>)) {
        MlaFwdRunner<true, false, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
        auto result = runner.run(options, hw_info);
        print_result(name, result, options.verbose);
      } else {
        MlaFwdRunner<false, false, decltype(shape), void, Mask, decltype(kernel_options)...> runner;
        auto result = runner.run(options, hw_info);
        print_result(name, result, options.verbose);
      }
    }
  };

  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;

  if (options.persistent) {
    // Persistent Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 persistent", Option<Tag::kIsPersistent, true_type>{});
  }
  else {
    // Individual Tile Scheduler
    run(Shape<_256, _128, HeadDim>{}, "tma ws 256x128 acc fp32 individual", Option<Tag::kIsPersistent, false_type>{});
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_single(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major != 10) {
    std::cout
      << "This example requires a GPU of NVIDIA's Blackwell Architecture "
      << "(compute capability major 10) and CUDA 12.8 or greater.\n";
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

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  if (options.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  else {
    hw_info.sm_count = options.sm_count;
  }

  std::cout << "###### B " << options.b << " H " << options.h << " H_K " << options.h_k << " Q " << options.q << " K " << options.k << " D latent " << options.dl << " D rope " << options.dr << " ";
  std::cout << "MLA Forward" << " " << (options.causal ? "Causal" : (options.residual ? "Residual" : "None")) << " ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  auto with_mask = [&](auto fn) {
    if (options.causal) {
      fn(CausalMask<false>{});
    }
    else if (options.residual) {
      fn(ResidualMask{});
    }
    else {
      fn(NoMask{});
    }
  };

  with_mask([&](auto fusion) {
    if (options.dl == 128 && options.dr == 64) {
      run_prefill_mla_fwd(fusion, options, hw_info);
    }
    else {
      std::cout << "No kernel instantiated for dl=" << options.dl << " dr=" << options.dr << std::endl;
    }
  });
#endif

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  std::vector<std::string> full_arguments(args, args + argc);

  bool recursed = false;
  for (size_t i = 1; i < full_arguments.size(); i++) {
    if (full_arguments[i].find(',') != std::string::npos) {
      auto arg = full_arguments[i];
      size_t eq_pos = arg.find('=');
      std::string prefix = eq_pos == std::string::npos ? "" : arg.substr(0, eq_pos+1);
      std::string rest = eq_pos == std::string::npos ? arg : arg.substr(eq_pos+1);
      for (;;) {
        size_t comma_pos = rest.find(',');
        std::string current = rest.substr(0, comma_pos);
        full_arguments[i] = prefix + current;
        std::vector<const char*> next_args;
        for (auto& elem : full_arguments) { next_args.push_back(elem.data()); }
        main(argc, next_args.data());
        if (comma_pos == std::string::npos) break;
        rest = rest.substr(comma_pos+1);
      }
      recursed = true;
      break;
    }
  }

  if (! recursed) {
    main_single(argc, args);
  }

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/77_blackwell_fmha/collective/fmha_fusion.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

struct NoMask {
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    return ceil_div(get<1>(problem_size), get<1>(tile_shape));
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size) {

    return;
  }
};

struct ResidualMask : NoMask {

  using Base = NoMask;

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return 1;
    }
    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size) {

    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<1>(pos) >= get<1>(problem_size)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

struct ResidualMaskForBackward : NoMask {

  using Base = NoMask;

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return 1;
    }
    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0) {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size) {

    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (! elem_less(pos, select<0,1>(problem_size))) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

// There are two ways to do causal if N_Q != N_K
// (1) The Q is at the beginning of the matrix
// (2) The Q is at the end of the matrix
template<bool kIsQBegin = true>
struct CausalMask : NoMask {

  using Base = NoMask;

  static constexpr bool IsQBegin = kIsQBegin;

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    // See note below on different ways to think about causal attention
    // Again, we'd add the offset_q into the max_blocks_q calculation
    if constexpr (IsQBegin) {
      int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape), get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    } else {
      const int offset_q = get<1>(problem_size) - get<0>(problem_size);
      int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape) + offset_q, get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {
        
    if constexpr (IsQBegin) {
      int trip_count = get_trip_count(blk_coord, tile_shape, problem_size);
      return std::min(trip_count, int(ceil_div(size<0>(tile_shape), size<1>(tile_shape))));
    } else {
      const int offset_tile_q = get<1>(problem_size) % get<1>(tile_shape);
      return ceil_div(get<0>(tile_shape) +  offset_tile_q, get<1>(tile_shape));
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
      BlkCoord const& blk_coord,
      TileShape const& tile_shape,
      ProblemSize const& problem_size) {

    return get_trip_count(blk_coord, tile_shape, problem_size) - get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size) {

    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is the default setting.
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to set kIsQBegin=false

    if constexpr (IsQBegin) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++) {
        auto pos = index_qk(i);
        if ((get<0>(pos) < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size))) {
          acc_qk(i) = -INFINITY;
        }
      }
    } else {
      const auto offset_q = get<1>(problem_size) - get<0>(problem_size);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++) {
        auto pos = index_qk(i);
        if ((get<0>(pos) + offset_q < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size))) {
          acc_qk(i) = -INFINITY;
        }
      }
    }
  }
};

struct CausalForBackwardMask : CausalMask<true>, ResidualMaskForBackward {

  using Base = CausalMask<true>;

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      ProblemSize const& problem_size) {

    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is what we demonstrate here
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to add an offset like so:
    //      get<0>(pos) + offset_q < get<1>(pos)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      bool masked = (get<0>(pos) < get<1>(pos)) || !elem_less(pos, problem_size);
      if (masked) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

};

struct VariableLength {
  int max_length;
  int* cumulative_length = nullptr;
  int total_length = -1;

  CUTE_HOST_DEVICE operator int() const {
    return max_length;
  }
};

template<class T> struct is_variable_length_impl : std::false_type {};
template<> struct is_variable_length_impl<VariableLength> : std::true_type {};
template<class T> constexpr bool is_variable_length_v = is_variable_length_impl<remove_cvref_t<T>>::value;

template<class Shape, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Idx const& idx) {
  return transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
}

template<class Shape, class Coord, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(Shape const& shape, Coord const& coord, Idx const& idx) {
  auto new_shape = apply_variable_length(shape, idx);
  auto new_coord = transform_leaf(shape, coord, [&](auto const& s, auto const& c) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return cute::make_tuple(c, s.cumulative_length[idx]);
    }
    else {
      return c;
    }
  });
  return cute::make_tuple(new_shape, new_coord);
}

template<class Shape, class Coord>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length_offset(Shape const& shape, Coord const& coord) {
  auto idx = back(back(coord));
  auto result_shape = transform_leaf(shape, [&](auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else {
      return s;
    }
  });
  auto result_offset = transform_leaf(coord, shape, [&](auto const& c, auto const& s) {
    if constexpr (is_variable_length_v<decltype(s)>) {
      return s.cumulative_length[idx];
    }
    else {
      return _0{};
    }
  });
  return cute::make_tuple(result_shape, result_offset);
}

}  // namespace cutlass::fmha::collective

namespace cute {

template<>
struct is_integral<cutlass::fmha::collective::VariableLength> : true_type {};

CUTE_HOST_DEVICE
void print(cutlass::fmha::collective::VariableLength a) {
  printf("Varlen<%d, %p>", a.max_length, a.cumulative_length);
}

}
""",
    'key-files/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cute/layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cutlass::fmha::collective {

template<
  class Element,
  class ElementAcc,
  class TileShape,  // Q, D, _
  class StrideO,    // Q, D, B
  class StrideLSE_,   // Q, B
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100FmhaFwdEpilogueTmaWarpspecialized {
    
  using Pipeline = cutlass::PipelineAsync<2>;

//  using SmemLayoutO = decltypa(make_layout(append<3>(select<0,1>(TileShape_WG{}), _2{})));
  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, Element, tuple_element_t<0, TileShape>, tuple_element_t<1, TileShape>>());
//  using SmemLayoutAtomO = decltype(make_ordered_layout(select<0,1>(TileShape{}), Step<_1, _0>{}));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, replace<2>(TileShape{}, _2{}), Step<_2, _1, _3>{}));
  using SmemLayoutO_ = SmemLayoutO;
  using StrideLSE = StrideLSE_;
  using ElementOut = Element;

  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  struct TensorStorage {

    using SmemLayoutO = SmemLayoutO_;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;

  };

  struct Arguments {
    Element* ptr_O;
    StrideO dO;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  using TMA_O = decltype(make_tma_copy(
    SM90_TMA_STORE{},
    make_tensor((Element*) nullptr, repeat_like(StrideO{}, 0), StrideO{}),
    SmemLayoutO{}(_,_,_0{})
  ));


  struct Params {
    TMA_O tma_store_o;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  // FMHA and MLA have different input ProblemShapes; 
  // get problem_shape_O according to the input ProblemShape.
  template<class ProblemShape>
  CUTLASS_DEVICE static constexpr
  auto get_problem_shape_O (
    ProblemShape const& problem_shape) {
    if constexpr (rank_v<decltype(get<2>(ProblemShape{}))> == 2) {
      return replace<1>(select<0,2,3>(problem_shape), get<2, 0>(problem_shape));
    } else {
      return select<0,2,3>(problem_shape);
    }
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace = nullptr) {

    auto ptr_O = args.ptr_O;
    StrideO dO = args.dO;

    auto problem_shape_O = get_problem_shape_O(problem_shape);

    if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
      auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
          int max_length_q = get<0>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dO) = get<0>(dO);
          get<2,1>(problem_shape_O) = max_length_q * (1 + get<2,1>(problem_shape_O));
          // offset ptr by the amount we add back in later
          ptr_O -= max_length_q * get<0>(dO);
      }
    }

    auto tma_store_o = make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(ptr_O, problem_shape_O, dO),
      SmemLayoutO{}(_,_,_0{})
    );

    return {
      tma_store_o,
      args.ptr_LSE,
      args.dLSE
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  const Params& params;

  CUTLASS_DEVICE Sm100FmhaFwdEpilogueTmaWarpspecialized(const Params& params) : params(params) {}

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE auto
  store(
      BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& shared_storage,
      Pipeline& pipeline, typename Pipeline::PipelineState& pipeline_consumer_state) {

    BlkCoord blk_coord = blk_coord_in;
    uint32_t lane_predicate = cute::elect_one_sync();

    using X = Underscore;

    int o0_index = 2 * get<0>(blk_coord);
    int o1_index = 2 * get<0>(blk_coord) + 1;

    Tensor mO_qdl_p = params.tma_store_o.get_tma_tensor(get_problem_shape_O(problem_shape));
    // offset mode 0 by (max_length - real_length)
    // offset mode 3,1 by cumulative_length + real_length
    // the ptr is already offset by - max_length
    // so in total this achieves 
    int offs_0 = 0;
    int offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
      auto cumulative_length_q = get<0>(params_problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
        int max_length_q = get<0>(params_problem_shape).max_length;
        offs_0 = max_length_q - get<0>(problem_shape);
        offs_2_1 = cumulative_length_q[get<2,1>(blk_coord)] + get<0>(problem_shape);
        get<2,1>(blk_coord) = 0;
      }
    }

    Tensor mO_qdl = domain_offset(make_coord(offs_0, _0{}, make_coord(_0{}, offs_2_1)), mO_qdl_p);

    Tensor gO_qdl = local_tile(mO_qdl, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gO = gO_qdl(_, _, _, _0{}, get<2>(blk_coord));
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto block_tma = params.tma_store_o.get_slice(0);
    Tensor tOsO = block_tma.partition_S(sO);
    Tensor tOgO = block_tma.partition_D(gO);

    auto pipeline_release_state = pipeline_consumer_state;

    // O1 O2
    // one pipeline: O
    // wait from corr, issue tma store on smem
    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate) {
      copy(params.tma_store_o, tOsO(_,_,_,_0{}), tOgO(_,_,_,o0_index));
    }
    tma_store_arrive();

    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate) {
      copy(params.tma_store_o, tOsO(_,_,_,_1{}), tOgO(_,_,_,o1_index));
    }
    tma_store_arrive();

    tma_store_wait<1>();

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;

    tma_store_wait<0>();

    if constexpr (cute::is_same_v<OrderLoadEpilogue, cute::true_type>) {
      cutlass::arch::NamedBarrier::arrive((NumWarpsLoad + NumWarpsEpilogue) * NumThreadsPerWarp, 
                                          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;

  } 

};

}  // namespace cutlass::fmha::collective
""",
    'key-files/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class TileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Mask_,
  // shape here is QG K H
  // and referes to the two softmax warps
  // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
  // (1, 2, 1) means they sit side by side (best for small Q / large K)
  class ThreadShape = Shape<_2, _1, _1>,
  // Since shared memory is sufficient for FMHA, there is no need to reuse shared memory.
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100FmhaFwdMainloopTmaWarpspecialized {

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = TileShape_;
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  static constexpr int StageCountQ = 2;
  static constexpr int StageCountKV = sizeof(Element_) == 1 ? 4 : 3;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;
  
  using ClusterShape = Shape<_1, _1, _1>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  using TileShapeQK = decltype(shape_div(TileShape{}, ThreadShape{}));

  using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountKV>{}));

  // Reuse shared memory for V and O.
  static constexpr bool IsOrderLoadEpilogue = std::is_same_v<OrderLoadEpilogue, cute::true_type>;
  struct TensorStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
  };

  enum class TmemAllocation : uint32_t {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage from softmax to correction
    V1 = S1,
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
  };

  // indices for V0 / V1
  enum : int {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
  };

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineTmaUmmaAsync<
    StageCountQ,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    StageCountKV,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  using PipelineO = cutlass::PipelineUmmaAsync<2>;

  // from corr to epilogue
  using PipelineE = cutlass::PipelineAsync<2>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static const int TransactionBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);

  static const int TransactionBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static const int TransactionBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  static_assert(TransactionBytesLoadK == TransactionBytesLoadV, "K and V smem layouts must be of equal size");

  using Load = Sm100FmhaLoadTmaWarpspecialized<
    Element, StrideQ, StrideK, StrideV,
    CollectiveMmaQK, CollectiveMmaPV,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV,
    TensorStorage, PipelineQ, PipelineKV, Mask, TileShape
  >;

  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / (float) std::sqrt(get<2>(problem_shape));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
      Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state);
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});  // slice out staging

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q1
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_,_,_,q_index);


    // wait for K1
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q1 * K1 -> S1
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K1
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q2
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    // gemm Q2 * K1 -> S2
    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K1
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V1
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // this acquire returns the ownership of all of S0 to the mma warp
    // including the P0 part
    // acquire corr first to take it out of the critical
    // path since softmax takes longer
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P1 * V1 -> O1
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1) {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
    }

    // release Q1
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q2
    if constexpr (get<0>(ThreadShape{}) > 1) {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1) {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0 S11 B2, ...
    // Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 * K3  , S22 * K2 , ...
  }

  template<bool need_apply_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      BlkCoord const& blk_coord, CoordTensor const& cS,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
    tStS.data() = uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    tStS_v.data() = uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1);
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    auto tilePlikeFP32 = size<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
    Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));
    tStS_P.data() = warp_uniform(uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1));
    Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));

    // Each thread owns a single row
      using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x; // 4x32 threads with 128 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;  // 4x32 threads with 128 cols of 8b elem
    using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS);
    Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS);

    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
    auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);

    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
    auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
    tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
    Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

    // wait on tensor core pipe
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    // read all of S from tmem into reg mem
    Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));
    copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

    if constexpr (need_apply_mask) {
      Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
    }

    ElementQK old_row_max = row_max;
    {
      // compute rowmax
      float row_max_0 = row_max;
      float row_max_1 = row_max;
      float row_max_2 = row_max;
      float row_max_3 = row_max;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
        row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
        row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
      }
      row_max = ::fmax(row_max_0, row_max_1);
      row_max = ::fmax(row_max, row_max_2);
      row_max = ::fmax(row_max, row_max_3);
    }

    ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

    Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
    tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
    tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // notify correction wg that they are ready (might need addtl ordering between S0 and S1 WG's)

    ElementQK scale = params.scale_softmax_log2;
    ElementQK row_max_scale = row_max_safe * scale;

    float2 scale_fp32x2 = make_float2(scale, scale);
    float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

    Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

    constexpr int kConversionsPerStep = 2;

    Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

    NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

    const int kReleasePipeCount = 10;  // must be multiple of 2

    order_s.wait();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(i + 0),
        tTMEM_LOADrS(i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
      tTMEM_LOADrS(i + 0) = out.x;
      tTMEM_LOADrS(i + 1) = out.y;

      tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
      tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

      Array<ElementQK, kConversionsPerStep> in_conv;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kConversionsPerStep; j++) {
        in_conv[j] = tTMEM_LOADrS(i + j);
      }
      tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);


      if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
        order_s.arrive();
      }

      // this prevents register spills in fp16
      if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
        if (i == size(tTMEM_LOADrS) - 6) {
          copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
        }
      }
    }

    // tmem_store(reg_S8) -> op_P
    CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
    CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
    copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));

    cutlass::arch::fence_view_async_tmem_store();

    // notify tensor core warp that P is ready
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    ElementQK acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
    row_sum *= acc_scale;
    // row_sum = sum(reg_S)
    float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
    float2 local_row_sum_1 = make_float2(0, 0);
    float2 local_row_sum_2 = make_float2(0, 0);
    float2 local_row_sum_3 = make_float2(0, 0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
      // row_sum += tTMEM_LOADrS(i);
      float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

      in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
      cute::add(local_row_sum_1, local_row_sum_1, in);

      in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
      cute::add(local_row_sum_2, local_row_sum_2, in);

      in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
      cute::add(local_row_sum_3, local_row_sum_3, in);
    }

    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
    cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
    float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

    row_sum = local_row_sum;

    if (final_call) {
      // re-acquire the S part in the final step
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
      tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
    }
  }

  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    int mask_tile_count = Mask{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
        get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
        0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage,
          (mask_tile_count == 1) &&
              (Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    // Masked iterations
    mask_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
  }

  template<class Stage, class TensorO>
  CUTLASS_DEVICE auto
  correction_epilogue(
      float scale,
      Stage stage,
      TensorO const& sO_01) {

    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor sO = sO_01(_,_,stage);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32 / sizeof(ElementOut);

    using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32, SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOsO = mma.get_slice(0).partition_C(sO);

    Tensor tOtO_i = logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOsO_i = logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    if constexpr (decltype(stage == _0{})::value) {
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O0);
    }
    else {
      static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O1);
    }

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

    float2 scale_f32x2 = make_float2(scale, scale);

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
      Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

#ifndef ONLY_SOFTMAX
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j+1) = out.y;
      }
#endif

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
      Tensor tSMrO_i = recast<uint32_t>(tSMrO);

      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  CUTLASS_DEVICE auto
  correction_rescale(
      float scale,
      uint32_t tmem_O) {

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 16;

    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b16x;  // 4x32 threads with 64 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);

    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    tOtO_i.data() = tOtO_i.data().get() + tmem_O;

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store   = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO = thr_tmem_store.partition_S(tOcO_i);
    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);

    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO), Int<128 / kCorrectionTileSize>{}));

    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    // sequence: LLMSLMSLMSS

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    int count = get<2>(TileShape{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO_i); j += 2) {
        float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO_i(j) = out.x;
        tTMrO_i(j+1) = out.y;
      }

      copy_out(i);
    }
  }

  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      // read row_wise new global max
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

      // e^(scale * (old_max - new_max)
      float scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O0));

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

      scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O1));

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0
    // read row_sum and final row_max here
    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);
    // store to epi smem

    // loop:
    //    TMEM_LOAD
    //    FMUL2 scale = 1 / global_sum * out_quant_scale
    //    F2FP
    //    store to smem
    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    
    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // load from V1
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }


  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction_empty(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    float lse = -INFINITY;
    int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

#define DSHOW(x) print(#x ": "); print(x); print("\n")
    if (threadIdx.x % 128 == 0 && block0()) {
      DSHOW(sO);
    }
#if 1

    using ElementOut = typename CollectiveEpilogue::ElementOut;
    auto tiled_copy = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
        make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}), Step<_1, _0>{}),
        sO.layout());

    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tOgO = thr_copy.partition_D(sO);
    auto tOrO = make_tensor<ElementOut>(shape(tOgO(_,_,_,_0{})));
    clear(tOrO);
    
    copy(tiled_copy, tOrO, tOgO(_,_,_,_0{}));
#endif
    
    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    copy(tiled_copy, tOrO, tOgO(_,_,_,_1{}));
    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

};

}  // namespace cutlass::fmha::collective
""",
    'key-files/examples/77_blackwell_fmha/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_mla_load_tma_warpspecialized.hpp"
#include "common/pipeline_mla.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class ComposedTileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Mask_,
  // shape here is QG K H
  // and referes to the two softmax warps
  // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
  // (1, 2, 1) means they sit side by side (best for small Q / large K)
  class ThreadShape = Shape<_2, _1, _1>,
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100MlaFwdMainloopTmaWarpspecialized {

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using ComposedTileShape = ComposedTileShape_;
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  static constexpr int StageCountQ = 2;
  static constexpr int StageCountK = 1;
  static constexpr int StageCountV = 1;
  static constexpr int StageCountKV = StageCountK + StageCountV;
  // Support StageCountKV > 2 in the future. 
  static_assert(StageCountK == 1 && StageCountV == 1, "Only support StageCountK = StageCountV = 1!");
  static_assert(std::is_same_v<ThreadShape, Shape<_2, _1, _1>>, "Only support ThreadShape = Shape<_2, _1, _1>");

  using ClusterShape = Shape<_1, _1, _1>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  static constexpr auto  HeadDimLatent = size<2, 0>(ComposedTileShape{});
  static constexpr auto  HeadDimRope = size<2, 1>(ComposedTileShape{});
  static constexpr auto  HeadDimQK = HeadDimLatent + HeadDimRope;
  static constexpr auto  HeadDimPV = HeadDimLatent;

  using TileShapeQK = decltype(shape_div(replace<2>(ComposedTileShape{}, HeadDimQK), ThreadShape{}));
  using TileShapePV = decltype(select<0,2,1>(shape_div(replace<2>(ComposedTileShape{}, HeadDimPV), ThreadShape{})));
  using TileShape = decltype(replace<2>(ComposedTileShape{}, HeadDimLatent));

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountK>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountV>{}));

  using SmemStorageOneStageO = decltype(make_layout(replace<2>(TileShapePV{}, _1{})));
  
  // Since the shared memory is not sufficient if we use separate Q, K, V, and O shared memory, 
  // we reuse shared memory for V and O to address this problem, 
  // and a barrier has been added to coordinate access to shared memory.
  static constexpr bool IsOrderLoadEpilogue = std::is_same_v<OrderLoadEpilogue, cute::true_type>;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;
  
  struct TensorStorageQKVO {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k; 
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_o; // use as O0
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v; // use as V0 and O1
  };

  struct TensorStorageQKV {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k; 
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
  };

  using TensorStorage = std::conditional_t<IsOrderLoadEpilogue, TensorStorageQKVO, TensorStorageQKV>;

  enum class TmemAllocation : uint32_t {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage from softmax to correction
    V1 = S1,
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
  };

  // indices for V0 / V1
  enum : int {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
  };

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineTmaUmmaAsync<
    StageCountQ,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineTmaAsyncMla<
    StageCountKV,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  using PipelineO = cutlass::PipelineUmmaAsync<2>;

  // from corr to epilogue
  using PipelineE = cutlass::PipelineAsync<2>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static constexpr int TransactionBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static constexpr int TransactionBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static constexpr int TransactionBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  using Load = Sm100MlaFwdLoadTmaWarpspecialized<
    Element, StrideQ, StrideK, StrideV,
    CollectiveMmaQK, CollectiveMmaPV,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV,
    TensorStorage, PipelineQ, PipelineKV, Mask, TileShape, OrderLoadEpilogue
  >;

  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / (float) std::sqrt(get<2, 0>(problem_shape) + get<2, 1>(problem_shape));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
      Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state);
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});  // slice out staging

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q1
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_,_,_,q_index);


    // wait for K1
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q1 * K1 -> S1
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index / 2), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K1
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q2
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    // gemm Q2 * K1 -> S2
    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index / 2), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K1
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V1
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // this acquire returns the ownership of all of S0 to the mma warp
    // including the P0 part
    // acquire corr first to take it out of the critical
    // path since softmax takes longer
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P1 * V1 -> O1
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index / 2), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index / 2), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1) {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index / 2), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index / 2), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index / 2), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
    }

    // release Q1
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q2
    if constexpr (get<0>(ThreadShape{}) > 1) {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1) {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index / 2), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0 S11 B2, ...
    // Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 * K3  , S22 * K2 , ...
  }

  template<bool need_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      bool need_apply_mask,
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      BlkCoord const& blk_coord, CoordTensor const& cS,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
    tStS.data() = uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    tStS_v.data() = uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1);
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    auto tilePlikeFP32 = size<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
    Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));
    tStS_P.data() = warp_uniform(uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1));
    Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));

    // Each thread owns a single row
    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x; // 4x32 threads with 128 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;  // 4x32 threads with 128 cols of 8b elem
    using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS);
    Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS);

    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
    auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);

    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
    auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
    tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
    Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

    // wait on tensor core pipe
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    // read all of S from tmem into reg mem
    Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));
    copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

    if constexpr (need_mask) {
      if(need_apply_mask) {
        Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
      }
    }

    ElementQK old_row_max = row_max;
    {
      // compute rowmax
      float row_max_0 = row_max;
      float row_max_1 = row_max;
      float row_max_2 = row_max;
      float row_max_3 = row_max;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
        row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
        row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
      }
      row_max = ::fmax(row_max_0, row_max_1);
      row_max = ::fmax(row_max, row_max_2);
      row_max = ::fmax(row_max, row_max_3);
    }

    ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

    Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
    tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
    tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // notify correction wg that they are ready (might need addtl ordering between S0 and S1 WG's)

    ElementQK scale = params.scale_softmax_log2;
    ElementQK row_max_scale = row_max_safe * scale;

    float2 scale_fp32x2 = make_float2(scale, scale);
    float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

    Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

    constexpr int kConversionsPerStep = 2;

    Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

    NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

    constexpr int kReleasePipeCount = 10;  // must be multiple of 2

    order_s.wait();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(i + 0),
        tTMEM_LOADrS(i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
      tTMEM_LOADrS(i + 0) = out.x;
      tTMEM_LOADrS(i + 1) = out.y;

      tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
      tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

      Array<ElementQK, kConversionsPerStep> in_conv;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kConversionsPerStep; j++) {
        in_conv[j] = tTMEM_LOADrS(i + j);
      }
      tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);


      if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
        order_s.arrive();
      }

      // this prevents register spills in fp16
      if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
        if (i == size(tTMEM_LOADrS) - 6) {
          copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
        }
      }
    }

    // tmem_store(reg_S8) -> op_P
    CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
    CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
    copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));

    cutlass::arch::fence_view_async_tmem_store();

    // notify tensor core warp that P is ready
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    ElementQK acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
    row_sum *= acc_scale;
    // row_sum = sum(reg_S)
    float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
    float2 local_row_sum_1 = make_float2(0, 0);
    float2 local_row_sum_2 = make_float2(0, 0);
    float2 local_row_sum_3 = make_float2(0, 0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
      // row_sum += tTMEM_LOADrS(i);
      float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

      in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
      cute::add(local_row_sum_1, local_row_sum_1, in);

      in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
      cute::add(local_row_sum_2, local_row_sum_2, in);

      in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
      cute::add(local_row_sum_3, local_row_sum_3, in);
    }

    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
    cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
    float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

    row_sum = local_row_sum;

    if (final_call) {
      // re-acquire the S part in the final step
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
      tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
    }
  }

  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {
    const int mask_trip_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);
    const int total_trip_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    int trip_idx = total_trip_count;

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
        get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
        0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    
    constexpr bool NeedMask = !std::is_same_v<Mask, NoMask>;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; trip_idx > 0; trip_idx -= 1) {
      softmax_step<NeedMask /* need_mask */>(
          trip_idx <= mask_trip_count,
          row_max, row_sum, stage,
          trip_idx == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
  }

  template<class Stage, class TensorO>
  CUTLASS_DEVICE auto
  correction_epilogue(
      float scale,
      Stage stage,
      TensorO const& sO_01) {

    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor sO = sO_01(_,_,stage);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    constexpr int kCorrectionTileSize = 32 / sizeof(ElementOut);

    using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32, SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOsO = mma.get_slice(0).partition_C(sO);

    Tensor tOtO_i = logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOsO_i = logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    if constexpr (decltype(stage == _0{})::value) {
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O0);
    }
    else {
      static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O1);
    }

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

    float2 scale_f32x2 = make_float2(scale, scale);

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
      Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

#ifndef ONLY_SOFTMAX
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j+1) = out.y;
      }
#endif

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
      Tensor tSMrO_i = recast<uint32_t>(tSMrO);

      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  CUTLASS_DEVICE auto
  correction_rescale(
      float scale,
      uint32_t tmem_O) {

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    constexpr int kCorrectionTileSize = 16;

    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b16x;  // 4x32 threads with 64 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);

    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    tOtO_i.data() = tOtO_i.data().get() + tmem_O;

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store   = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO = thr_tmem_store.partition_S(tOcO_i);
    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);

    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO), Int<128 / kCorrectionTileSize>{}));

    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    // sequence: LLMSLMSLMSS

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    constexpr int count = get<2>(TileShape{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO_i); j += 2) {
        float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO_i(j) = out.x;
        tTMrO_i(j+1) = out.y;
      }

      copy_out(i);
    }
  }

  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      // read row_wise new global max
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

      // e^(scale * (old_max - new_max)
      float scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O0));

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

      scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O1));

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0
    // read row_sum and final row_max here
    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);
    // store to epi smem

    // loop:
    //    TMEM_LOAD
    //    FMUL2 scale = 1 / global_sum * out_quant_scale
    //    F2FP
    //    store to smem
    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // load from V1
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }


  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction_empty(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    float lse = -INFINITY;
    int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

#define DSHOW(x) print(#x ": "); print(x); print("\n")
    if (threadIdx.x % 128 == 0 && block0()) {
      DSHOW(sO);
    }
#if 1

    using ElementOut = typename CollectiveEpilogue::ElementOut;
    auto tiled_copy = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
        make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}), Step<_1, _0>{}),
        sO.layout());

    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tOgO = thr_copy.partition_D(sO);
    auto tOrO = make_tensor<ElementOut>(shape(tOgO(_,_,_,_0{})));
    clear(tOrO);
    
    copy(tiled_copy, tOrO, tOgO(_,_,_,_0{}));
#endif
    
    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    copy(tiled_copy, tOrO, tOgO(_,_,_,_1{}));
    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape)) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

};

}  // namespace cutlass::fmha::collective
""",
    'key-files/examples/77_blackwell_fmha/collective/sm100_fmha_mla_load_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element,
  class StrideQ,
  class StrideK,
  class StrideV,
  class CollectiveMmaQK,
  class CollectiveMmaPV,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class TensorStorage,
  class PipelineQ,
  class PipelineKV,
  class Mask,
  class TileShape,
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100MlaFwdLoadTmaWarpspecialized {

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  static constexpr int TransactionBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static constexpr int TransactionBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  struct Arguments {
    const Element* ptr_Q;
    StrideQ dQ;
    const Element* ptr_K;
    StrideK dK;
    const Element* ptr_V;
    StrideV dV;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    auto dQ = args.dQ;
    auto dK = args.dK;
    auto dV = args.dV;
    auto problem_shape_qk = replace<2>(problem_shape, get<2, 0>(problem_shape) + get<2, 1>(problem_shape));

    if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
      auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
          int max_length_q = get<0>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dQ) = get<0>(dQ);
          get<3,1>(problem_shape_qk) = std::max(get<3,1>(problem_shape_qk), max_length_q * (1 + get<3,1>(problem_shape)));
          // offset ptr by the amount we add back in later
          ptr_Q -= max_length_q * get<0>(dQ);
      }
    }

    if constexpr (is_variable_length_v<tuple_element_t<1, ProblemShape>>) {
      auto cumulative_length_kv = get<1>(problem_shape).cumulative_length;
      if (cumulative_length_kv != nullptr) {
          int max_length_kv = get<1>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dK) = get<0>(dK);
          get<2,1>(dV) = get<0>(dV);
          get<3,1>(problem_shape_qk) = std::max(get<3,1>(problem_shape_qk), max_length_kv * (1 + get<3,1>(problem_shape)));
          // offset ptr by the amount we add back in later
          ptr_K -= max_length_kv * get<0>(dK);
          ptr_V -= max_length_kv * get<0>(dV);
      }
    }

    auto problem_shape_pv = replace<1>(select<0,2,1,3>(problem_shape_qk), get<2, 0>(problem_shape));

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        problem_shape_qk,
        typename CollectiveMmaQK::Arguments {
            ptr_Q, dQ,
            ptr_K, dK,
        }, /*workspace=*/ nullptr);

    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
            ptr_K, dK,  // never used, dummy
            ptr_V, select<1,0,2>(dV),
        }, /*workspace=*/ nullptr);

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b
    };
  }


  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    BlkCoord blk_coord_q = blk_coord_in;
    BlkCoord blk_coord_kv = blk_coord_in;

    auto problem_shape_qk = replace<2>(problem_shape, get<2, 0>(problem_shape) + get<2, 1>(problem_shape));
    auto problem_shape_v = replace<2>(problem_shape, get<2, 0>(problem_shape));

    int mask_tile_count = Mask{}.get_trip_count(blk_coord_in, TileShape{}, problem_shape);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one

    // Q1, K1, Q2, V1, K2, V2, K3, V3, ...
    // two pipes: Q and KV
    // from Memory (prod) to TensorCore (cons)

    // compute gQ, sQ
    // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    Tensor mQ_qdl_p = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape_qk));

    int q_offs_0 = 0;
    int q_offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
      auto cumulative_length_q = get<0>(params_problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
        int max_length_q = get<0>(params_problem_shape).max_length;
        q_offs_0 = max_length_q - get<0>(problem_shape);
        q_offs_2_1 = cumulative_length_q[get<2,1>(blk_coord_q)] + get<0>(problem_shape);
        get<2,1>(blk_coord_q) = 0;
      }
    }

    Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{}, q_offs_2_1)), mQ_qdl_p);

    Tensor gQ_qdl = local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ_qdl);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto [tQgQ_qdl, tQsQ] = tma_partition(
      params.tma_load_q, _0{}, make_layout(_1{}), 
      group_modes<0,3>(sQ), group_modes<0,3>(tSgQ_qdl)
    );
    Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));

    // compute gK, sK
    Tensor mK_kdl_p = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape_qk));

    int kv_offs_0 = 0;
    int kv_offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<1, ParamsProblemShape>>) {
      auto cumulative_length = get<1>(params_problem_shape).cumulative_length;
      if (cumulative_length != nullptr) {
        int max_length = get<1>(params_problem_shape).max_length;
        kv_offs_0 = max_length - get<1>(problem_shape);
        kv_offs_2_1 = cumulative_length[get<2,1>(blk_coord_kv)] + get<1>(problem_shape);
        get<2,1>(blk_coord_kv) = 0;
      }
    }

    Tensor mK_kdl = domain_offset(make_coord(kv_offs_0, _0{}, make_coord(_0{}, kv_offs_2_1)), mK_kdl_p);

    Tensor gK_kdl = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgK_kdl = mma_qk.partition_B(gK_kdl);
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    auto [tKgK_kdl, tKsK] = tma_partition(
      params.tma_load_k, _0{}, make_layout(_1{}),
      group_modes<0,3>(sK), group_modes<0,3>(tSgK_kdl)
    );
    Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));

    // compute gV, sV
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor mV_dkl_p = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape_v));

    Tensor mV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, make_coord(_0{}, kv_offs_2_1)), mV_dkl_p);

    Tensor gV_dkl = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgV_dkl = mma_pv.partition_B(gV_dkl);
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    auto [tVgV_dkl, tVsV] = tma_partition(
      params.tma_load_v, _0{}, make_layout(_1{}),
      group_modes<0,3>(sV), group_modes<0,3>(tOgV_dkl)
    );
    auto tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));

    // blk_coord in decomposed in terms of TileShape, not TileShapeQK
    // As such, it needs to be transformed as
    // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
    //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q1
    int q0_index = 2 * get<0>(blk_coord_q);
    int q1_index = 2 * get<0>(blk_coord_q) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K1
    int k_index = 0;
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index() / 2));
    }
    ++pipeline_kv_producer_state;

    // Q2
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q1_index), tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    if constexpr (cute::is_same_v<OrderLoadEpilogue, cute::true_type>) {
      cutlass::arch::NamedBarrier::sync((NumWarpsLoad + NumWarpsEpilogue) * NumThreadsPerWarp, 
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    // V1
    pipeline_kv.producer_acquire_bytes(pipeline_kv_producer_state, TransactionBytesLoadV);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index() / 2));
    }
    ++pipeline_kv_producer_state;
    k_index += 1;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index() / 2));

        // prefetch vi
        cute::prefetch(params.tma_load_v, tVgV(_, k_index));
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire_bytes(pipeline_kv_producer_state, TransactionBytesLoadV);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index() / 2));

        // prefetch ki+1
        if(mask_tile_count > 1) {
          cute::prefetch(params.tma_load_k, tKgK(_, k_index + 1));
        }
      }
      ++pipeline_kv_producer_state;
      k_index += 1;
    }
  }
};

}  // namespace cutlass::fmha::collective
""",
    'key-files/examples/77_blackwell_fmha/common/pipeline_mla.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*!
  \file
  \brief Support the producer to acquire specific bytes of data.
*/

#pragma once

#include "cutlass/pipeline/sm100_pipeline.hpp"

namespace cutlass {

using namespace cute;

template <
  int Stages_,
  class ClusterShape = Shape<int,int,_1>,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaAsyncMla {

public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;

private:
  using Impl = PipelineTmaUmmaAsync<Stages_, ClusterShape, AtomThrShape_MNK_>;

public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;


  using McastDirection = McastDirection;

  // Helper function to initialize barriers
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      auto atom_thr_shape = AtomThrShape_MNK{};
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) +
                                     (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) - 1;

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction) {
    auto atom_thr_shape = AtomThrShape_MNK{};

    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      uint32_t const multicast_consumer_arrival_count = (mcast_direction == McastDirection::kRow) ?
        cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape) : // Mcast with row ctas
        cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape);  // Mcast with col ctas

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    // Calculate consumer mask
    if (params_.role == ThreadCategory::Consumer) {
      auto cluster_layout = make_layout(cluster_shape);
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRowCol>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, McastDirection mcast_direction) {
    // Calculate consumer mask
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    auto cluster_layout = make_layout(cluster_shape);
    if (mcast_direction == McastDirection::kRow) {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRow>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
    else {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kCol>(cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }


public:
  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaAsyncMla(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
        static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
        if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
          init_barriers(storage, params_, cluster_shape);
        }

        static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
        if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
          init_masks(cluster_shape);
        }
  }

  template<typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE
  PipelineTmaAsyncMla(SharedStorage& storage, Params params, ClusterShape cluster_shape, McastDirection mcast_direction, InitBarriers = {}, InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, cute::false_type{})
      , params_(params)
      , empty_barrier_ptr_(&storage.empty_barrier_[0])
      , full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape, mcast_direction);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape, mcast_direction);
    }
  }


  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_acquire_bytes(uint32_t stage, uint32_t bytes, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  void producer_acquire_bytes(PipelineState state, uint32_t bytes, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire_bytes(state.index(), bytes, state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index(), false);
  }

private:
  Impl impl_;
  Params params_;
  EmptyBarrier *empty_barrier_ptr_;
  FullBarrier *full_barrier_ptr_;
  uint16_t block_id_mask_ = 0;
  static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip) {
    detail::pipeline_check_is_consumer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&empty_barrier_ptr_[stage]);
    if constexpr (is_2sm_mma) { // Mma cluster shape is 2x1
      if (!skip) {
        cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, block_id_mask_);
      }
    }
    else {
      if (!skip) {
        if constexpr (cute::is_static_v<ClusterShape> and size(ClusterShape{}) == 1) {
          cutlass::arch::umma_arrive(smem_ptr);
        }
        else {
          cutlass::arch::umma_arrive_multicast(smem_ptr, block_id_mask_);
        }
      }
    }
  }
};

}
""",
    'key-files/examples/77_blackwell_fmha/kernel/fmha_causal_tile_scheduler.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/fast_math.h"

namespace cutlass::fmha::kernel {

////////////////////////////////////////////////////////////////////////////////

// Swizzle Q tile and H tile to improve L2 cache hit rate, 
// and launch the longest main loop first to keep most SMs busy.

struct CausalIndividualTileScheduler {
  
  static constexpr int TileQ = 16;
  static constexpr int TileH = 8;
  static constexpr int TileSize = TileQ * TileH;

  struct Params {
    dim3 grid;
    int tile_max_q;
    FastDivmod divmod_tile_col;
    FastDivmod divmod_tile_size;
    FastDivmod divmod_tile_head;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  CausalIndividualTileScheduler(Params const& params) : params(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape) {
    using namespace cute;

    dim3 grid(size<3,0>(problem_size), round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape)), size<3,1>(problem_size));
    // gridDim.x must multiple of TileH
    const int tile_col_count = grid.x / TileH;
    const int tile_max_q = grid.y / TileQ * TileQ;
    return Params{ grid , tile_max_q, tile_col_count, TileSize, TileH};
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    int tile_idx, tile_tail;
    params.divmod_tile_size(tile_idx, tile_tail, block_idx);

    int tile_row_idx, tile_col_idx;
    params.divmod_tile_col(tile_row_idx,tile_col_idx, tile_idx);

    int row_offset_in_tail, col_offset_in_tail;
    params.divmod_tile_head(row_offset_in_tail,col_offset_in_tail, tile_tail);

    const int row_idx = tile_row_idx * TileQ + row_offset_in_tail;
    const int col_idx = tile_col_idx * TileH + col_offset_in_tail;
    
    // last q tile launch first
    if(blockIdx.y >= params.tile_max_q) {
      return make_coord(int(gridDim.y - 1 - blockIdx.y), _0{}, make_coord(int(blockIdx.x), int(blockIdx.z)));
    } 

    return make_coord(int(gridDim.y) - 1 - row_idx, _0{}, make_coord(col_idx, int(blockIdx.z)));
  }

  CUTLASS_DEVICE
  CausalIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

// Launch order: H Q B
struct CausalPersistentTileScheduler {

  struct Params {
    int num_blocks;
    FastDivmod divmod_h;
    FastDivmod divmod_m_block;
    FastDivmod divmod_b;

    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  CausalPersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape) {
    using namespace cute;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    hw_info.sm_count = sm_count;

    int num_m_blocks = cutlass::round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape));
    int num_blocks = num_m_blocks * size<3,0>(problem_size) * size<3,1>(problem_size);

    return Params {
      num_blocks,
      { size<3,0>(problem_size) }, { num_m_blocks}, { size<3,1>(problem_size) },
      hw_info
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return block_idx < params.num_blocks;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int block_decode = block_idx;
    int m_block, bidb, bidh;
    params.divmod_h(block_decode, bidh, block_decode);
    params.divmod_m_block(block_decode, m_block, block_decode);
    params.divmod_b(block_decode, bidb, block_decode);
    return make_coord(m_block, _0{}, make_coord(bidh, bidb));
  }

  CUTLASS_DEVICE
  CausalPersistentTileScheduler& operator++() {
    block_idx += gridDim.x;
    return *this;
  }
};
////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025  - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/tensor.hpp"
#include "cute/arch/simd_sm100.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "collective/fmha_common.hpp"

#include <cmath>

namespace cutlass::fmha::kernel {

using namespace cutlass::fmha::collective;

using namespace cute;

template<
    class ProblemShape,
    class Element,
    class ElementAcc,
    class TileShape,
    class Mask
>
struct Sm100FmhaBwdKernelTmaWarpSpecialized {

  using TileShapeQ = decltype(get<0>(TileShape{}));
  static_assert(std::is_same_v<TileShapeQ, _128>, "tile shape K must be 128");
  using TileShapeK = decltype(get<1>(TileShape{}));
  static_assert(std::is_same_v<TileShapeK, _128>, "tile shape K must be 128");
  using TileShapeDQK = decltype(get<2>(TileShape{}));
  using TileShapeDVO = decltype(get<2>(TileShape{}));

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  struct TmemAllocation {
    static constexpr uint32_t kDK = 0;                     // TileShapeK x TileShapeDQK x acc
    static constexpr uint32_t kDV = kDK + TileShapeDQK{};  // TileShapeK x TileShapeDVO x acc
    static constexpr uint32_t kDQ = kDV + TileShapeDVO{};  // TileShapeQ x TileShapeDQK x acc
    static constexpr uint32_t kDP = kDQ;                   // TileShapeK x TileShapeQ   x inp
    static constexpr uint32_t kS = kDQ + max(TileShapeQ{}, TileShapeDQK{});
    static constexpr uint32_t kP = kS;
    static constexpr uint32_t kTotal = kS + TileShapeQ{};
  };

  static_assert(
      static_cast<int>(TmemAllocation::kTotal) <= TmemAllocator::Sm100TmemCapacityColumns,
      "using too much tmem"
  );

  enum class WarpRole {
    Empty = 0x0, Load = 0x1, Mma = 0x2, Compute = 0x3, Reduce = 0x4
  };

  static constexpr unsigned long long kWarpAssignment = 0x12'3333'3333'4444ull;
  static constexpr int kNumComputeWarps = 8;
  static constexpr int kNumReduceWarps = 4;
  CUTLASS_DEVICE WarpRole warp_idx_to_role(int warp_idx) {
    return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
  }

  struct RegisterAllocation {
    static constexpr int kWarpgroup0 = 160-8;
    static constexpr int kWarpgroup1 = 128;
    static constexpr int kWarpgroup2 = 96;
    static constexpr int kReduce = kWarpgroup0;
    static constexpr int kCompute = kWarpgroup1;
    static constexpr int kMma = kWarpgroup2;
    static constexpr int kEmpty = kWarpgroup2;
    static constexpr int kLoad = kWarpgroup2;

    static_assert(kWarpgroup0 + 2 * kWarpgroup1 + kWarpgroup2 <= 512);
  };

  using ArchTag = cutlass::arch::Sm100;

  using ClusterShape = Shape<_1, _1, _1>;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;

  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int kNumWarps = kNumComputeWarps + kNumReduceWarps + 4;
  static constexpr int MaxThreadsPerBlock = NumThreadsPerWarp * kNumWarps;

  static constexpr int Alignment = 128 / sizeof_bits_v<Element>;
  static constexpr int kStages = 2;

  using TensorStrideContiguousK = Stride<int, _1, Stride<int, int>>;
  using TensorStrideContiguousMN = Stride<_1, int, Stride<int, int>>;

  // compute S
  using CollectiveMmaKQ = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeQ, TileShapeDQK>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeKQ = typename CollectiveMmaKQ::TileShape;
  using TiledMmaKQ = typename CollectiveMmaKQ::TiledMma;

  // compute dP
  using CollectiveMmaVDO = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeQ, TileShapeDVO>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeVDO = typename CollectiveMmaVDO::TileShape;
  using TiledMmaVDO = typename CollectiveMmaVDO::TiledMma;

  // compute dV
  using CollectiveMmaPDO = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // needs to match ordering of S calculation
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeDVO, TileShapeQ>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapePDO = typename CollectiveMmaPDO::TileShape;
  using TiledMmaPDO = decltype(to_tiled_mma_sm100_ts(typename CollectiveMmaPDO::TiledMma{}));

  // compute dK
  using CollectiveMmaDSQ = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // somewhat arbitrary since we dump to smem, need to agree with the next one
      Element, TensorStrideContiguousK , Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeK, TileShapeDQK, TileShapeQ>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDSQ = typename CollectiveMmaDSQ::TileShape;
  using TiledMmaDSQ = typename CollectiveMmaDSQ::TiledMma;

  // compute dQ
  using CollectiveMmaDSK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // somewhat arbitrary since we dump to smem, need to agree with the previous one
      Element, TensorStrideContiguousMN, Alignment,
      Element, TensorStrideContiguousMN, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeDQK, TileShapeK>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDSK = typename CollectiveMmaDSK::TileShape;
  using TiledMmaDSK = typename CollectiveMmaDSK::TiledMma;

  // pipelines are named Pipeline<Producer><Consumer><Resource>
  static constexpr int kStagesComputeSmem = 1;
  using PipelineLoadMmaQ = PipelineTmaUmmaAsync<2, ClusterShape>;
  using PipelineLoadMmaDO = PipelineTmaUmmaAsync<1, ClusterShape>;
  using PipelineLoadComputeLSE = PipelineAsync<1>;
  using PipelineLoadComputeSumOdO = PipelineAsync<1>;
  using PipelineMmaComputeS = PipelineUmmaAsync<1>;
  using PipelineMmaComputeDP = PipelineUmmaAsync<1>;
  using PipelineMmaReduceDQ = PipelineUmmaAsync<1>;
  using PipelineComputeMmaP = PipelineUmmaConsumerAsync<1>;
  using PipelineComputeMmaDS = PipelineUmmaConsumerAsync<kStagesComputeSmem>;
  using PipelineMmaComputeDKDV = PipelineUmmaAsync<2>;
  static constexpr int kStagesReduceTmaStore = 2;
  using PipelineReduceTmaStore = PipelineTmaStore<kStagesReduceTmaStore>;

  struct PipelineStorage {
    alignas(16) typename PipelineLoadMmaQ::SharedStorage load_mma_q;
    alignas(16) typename PipelineLoadMmaDO::SharedStorage load_mma_do;
    alignas(16) typename PipelineLoadComputeLSE::SharedStorage load_compute_lse;
    alignas(16) typename PipelineLoadComputeSumOdO::SharedStorage load_compute_sum_odo;
    alignas(16) typename PipelineMmaComputeS::SharedStorage mma_compute_s;
    alignas(16) typename PipelineMmaComputeDP::SharedStorage mma_compute_dp;
    alignas(16) typename PipelineMmaReduceDQ::SharedStorage mma_reduce_dq;
    alignas(16) typename PipelineComputeMmaP::SharedStorage compute_mma_p;
    alignas(16) typename PipelineComputeMmaDS::SharedStorage compute_mma_ds;
    alignas(16) typename PipelineMmaComputeDKDV::SharedStorage mma_compute_dkdv;
  };

  template<class Layout, class Stages = _1>
  static CUTE_DEVICE constexpr auto restage(Layout const& layout, Stages stages = {}) {
    return composition(layout, make_tuple(_, _, _, make_layout(stages)));
  }

  using SmemLayoutK = decltype(restage(typename CollectiveMmaKQ::SmemLayoutA{}));
  using SmemLayoutV = decltype(restage(typename CollectiveMmaVDO::SmemLayoutA{}));
  using SmemLayoutQ = decltype(restage(typename CollectiveMmaKQ::SmemLayoutB{}, _2{}));
  using SmemLayoutDO = decltype(restage(typename CollectiveMmaVDO::SmemLayoutB{}, _1{}));
  using SmemLayoutDS = decltype(restage(typename CollectiveMmaDSK::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutLSE = Layout<Shape<TileShapeQ, _1>>;
  using SmemLayoutSumOdO = Layout<Shape<TileShapeQ, _1>>;

  using SmemLayoutQT = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutB{}, _2{}));
  using SmemLayoutKT = decltype(restage(typename CollectiveMmaDSK::SmemLayoutB{}));
  using SmemLayoutDST = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutDOT = decltype(restage(typename CollectiveMmaPDO::SmemLayoutB{}, _1{}));

  using TileShapeDQ = _32;
  using SmemAtomDQ = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::K, ElementAcc, TileShapeQ, TileShapeDQ
  >());
  using SmemShapeDQ = Shape<TileShapeQ, TileShapeDQ, Int<kStagesReduceTmaStore>>;
  using SmemLayoutDQ = decltype(tile_to_shape(SmemAtomDQ{}, SmemShapeDQ{}, Step<_2, _1, _3>{}));

  struct TensorStorage {
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKT>> smem_k_t;
    };
    alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQT>> smem_q_t;
    };
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDO>> smem_do;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDOT>> smem_do_t;
    };
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDS>> smem_ds;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutDST>> smem_ds_t;
    };
    alignas(1024) cute::array<ElementAcc, cute::cosize_v<SmemLayoutDQ>> smem_dq;
    alignas(16) cute::array<ElementAcc, cute::cosize_v<SmemLayoutLSE>> smem_lse;
    alignas(16) cute::array<ElementAcc, cute::cosize_v<SmemLayoutSumOdO>> smem_sum_odo;
  };

  static constexpr int kTransactionsBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static constexpr int kTransactionsBytesLoadDO = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutDO{})) * cute::sizeof_bits_v<Element>);

  static constexpr int kTransactionsBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static constexpr int kTransactionsBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  struct SharedStorage {
    TensorStorage tensors;
    PipelineStorage pipelines;
    uint32_t tmem_base_ptr;
  };

  // this is tight enough that it won't work with sizeof due to padding for alignment
  static constexpr int SharedStorageSize = offsetof(SharedStorage, tmem_base_ptr) + sizeof(uint32_t);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "using too much smem");

  using TensorStride = TensorStrideContiguousK;  // S D (H B)
  using RowTensorStride = Stride<_1, Stride<int, int>>;    // S (H B)

  struct MainloopArguments {
    const Element* ptr_q;
    TensorStride stride_q;
    const Element* ptr_k;
    TensorStride stride_k;
    const Element* ptr_v;
    TensorStride stride_v;
    const Element* ptr_do;
    TensorStride stride_do;

    const ElementAcc* ptr_lse;
    RowTensorStride stride_lse;

    const ElementAcc* ptr_sum_odo;
    RowTensorStride stride_sum_odo;

    ElementAcc* ptr_dq_acc;
    TensorStride stride_dq_acc;

    ElementAcc softmax_scale = 1.0f / sqrtf(TileShapeDQK{});
  };

  using TMA_K = typename CollectiveMmaKQ::Params::TMA_A;
  using TMA_V = typename CollectiveMmaVDO::Params::TMA_A;
  using TMA_Q = typename CollectiveMmaKQ::Params::TMA_B;
  using TMA_DO = typename CollectiveMmaVDO::Params::TMA_B;

  using TMA_DQ = decltype(make_tma_copy(SM90_TMA_REDUCE_ADD{},
      make_tensor((const ElementAcc*)nullptr, make_shape(1, 1, make_shape(1, 1)), TensorStride{}),
      SmemLayoutDQ{}(_, _, _0{})
  ));

  struct MainloopParams {
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_Q tma_load_q;
    TMA_DO tma_load_do;
    TMA_DQ tma_red_dq;
  };

  struct EpilogueArguments {
    Element* ptr_dk;
    TensorStride stride_dk;
    Element* ptr_dv;
    TensorStride stride_dv;
  };

  struct Arguments {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    MainloopParams mainloop_params;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
  };


  static bool can_implement(Arguments const& args) {
    auto [Q, K, D, HB] = args.problem_shape;
    auto [H, B] = HB;
    if (Q <= 0 || K <= 0 || D <= 0 || H <= 0 || B <= 0) {
      return false;
    }
    if (D % Alignment != 0) {
      return false;
    }
    return true;
  }


  static Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return Status::kSuccess;
  }


  static Params to_underlying_arguments(Arguments const& args, void*) {
    auto [Q_, K_, D, HB] = args.problem_shape;
    int Q = Q_;
    int K = K_;

    if constexpr (is_variable_length_v<decltype(Q_)>) {
      Q = Q_.total_length;
    }
    if constexpr (is_variable_length_v<decltype(K_)>) {
      K = K_.total_length;
    }

    auto params_kq = CollectiveMmaKQ::to_underlying_arguments(
      make_shape(K, Q, D, HB),
      typename CollectiveMmaKQ::Arguments {
        args.mainloop.ptr_k, args.mainloop.stride_k,
        args.mainloop.ptr_q, args.mainloop.stride_q,
      }, /*workspace=*/nullptr);

    auto params_vdo = CollectiveMmaVDO::to_underlying_arguments(
      make_shape(K, Q, D, HB),
      typename CollectiveMmaVDO::Arguments {
        args.mainloop.ptr_v, args.mainloop.stride_v,
        args.mainloop.ptr_do, args.mainloop.stride_do,
      }, /*workspace=*/nullptr);

    TMA_DQ tma_red_dq = make_tma_copy(
        SM90_TMA_REDUCE_ADD{},
        make_tensor(args.mainloop.ptr_dq_acc, make_shape(Q_, D, HB), args.mainloop.stride_dq_acc),
        SmemLayoutDQ{}(_, _, _0{})
    );

    return Params{
      args.problem_shape,
      args.mainloop,
      MainloopParams{
        params_kq.tma_load_a,
        params_vdo.tma_load_a,
        params_kq.tma_load_b,
        params_vdo.tma_load_b,
        tma_red_dq
      },
      args.epilogue,
      args.hw_info
    };
  }


  template<class T>
  static CUTLASS_DEVICE auto quantize(T const& input) {
    constexpr int AlignmentS = 4;
    auto output = make_tensor<Element>(shape(input));
    auto input_vec = recast<Array<ElementAcc, AlignmentS>>(input);
    auto output_vec = recast<Array<Element, AlignmentS>>(output);

    cutlass::NumericArrayConverter<Element, ElementAcc, AlignmentS> epilogue_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(input_vec); i++) {
      output_vec(i) = epilogue_op(input_vec(i));
    }

    return output;
  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void load(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineLoadMmaQ& pipeline_load_mma_q,
      typename PipelineLoadMmaQ::PipelineState& pipeline_load_mma_q_producer_state,
      PipelineLoadMmaDO& pipeline_load_mma_do,
      typename PipelineLoadMmaDO::PipelineState& pipeline_load_mma_do_producer_state,
      PipelineLoadComputeLSE& pipeline_load_compute_lse,
      typename PipelineLoadComputeLSE::PipelineState& pipeline_load_compute_lse_producer_state,
      PipelineLoadComputeSumOdO& pipeline_load_compute_sum_odo,
      typename PipelineLoadComputeSumOdO::PipelineState& pipeline_load_compute_sum_odo_producer_state) {

    auto [Q, K, D, HB] = problem_shape;

    using X = Underscore;

    uint16_t mcast_mask = 0;

    auto mK_in = mainloop_params.tma_load_k.get_tma_tensor(make_shape(K, D, HB));
    auto mV_in = mainloop_params.tma_load_v.get_tma_tensor(make_shape(K, D, HB));
    auto mQ_in = mainloop_params.tma_load_q.get_tma_tensor(make_shape(Q, D, HB));
    auto mDO_in = mainloop_params.tma_load_do.get_tma_tensor(make_shape(Q, D, HB));

    auto mK = domain_offset(select<1,2,3>(blk_offset), mK_in);
    auto mV = domain_offset(select<1,2,3>(blk_offset), mV_in);
    auto mQ = domain_offset(select<0,2,3>(blk_offset), mQ_in);
    auto mDO = domain_offset(select<0,2,3>(blk_offset), mDO_in);

    auto gK = local_tile(mK, TileShapeKQ{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gQ = local_tile(mQ, TileShapeKQ{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gV = local_tile(mV, TileShapeVDO{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gDO = local_tile(mDO, TileShapeVDO{}, make_coord(_,_,_), Step<X, _1, _1>{});

    ThrMMA cta_mma_kq = TiledMmaKQ{}.get_slice(_0{});
    ThrMMA cta_mma_vdo = TiledMmaVDO{}.get_slice(_0{});

    auto tSTgK = cta_mma_kq.partition_A(gK);
    auto tSTgQ = cta_mma_kq.partition_B(gQ);
    auto tDPTgV = cta_mma_vdo.partition_A(gV);
    auto tDPTgDO = cta_mma_vdo.partition_B(gDO);

    auto sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.begin()), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(shared_tensors.smem_v.begin()), SmemLayoutV{});
    auto sDO = make_tensor(make_smem_ptr(shared_tensors.smem_do.begin()), SmemLayoutDO{});

    auto [tKgK_mkl, tKsK] = tma_partition(
        mainloop_params.tma_load_k, _0{}, make_layout(_1{}),
        group_modes<0,3>(sK), group_modes<0,3>(tSTgK));
    auto [tQgQ_mkl, tQsQ] = tma_partition(
        mainloop_params.tma_load_q, _0{}, make_layout(_1{}),
        group_modes<0,3>(sQ), group_modes<0,3>(tSTgQ));
    auto [tVgV_mkl, tVsV] = tma_partition(
        mainloop_params.tma_load_v, _0{}, make_layout(_1{}),
        group_modes<0,3>(sV), group_modes<0,3>(tDPTgV));
    auto [tDOgDO_mkl, tDOsDO] = tma_partition(
        mainloop_params.tma_load_do, _0{}, make_layout(_1{}),
        group_modes<0,3>(sDO), group_modes<0,3>(tDPTgDO));

    // set up lse and sum_odo

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch] = blk_coord;

    pipeline_load_mma_q.producer_acquire(pipeline_load_mma_q_producer_state);
    auto tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);

    pipeline_load_mma_q.producer_expect_transaction(pipeline_load_mma_q_producer_state, kTransactionsBytesLoadK);

    // load K
    if (cute::elect_one_sync()) {
      cute::copy(
          mainloop_params.tma_load_k.with(*tma_barrier, mcast_mask),
          tKgK_mkl(_, blk_coord_k, _0{}, blk_coord_batch),
          tKsK(_, _0{})
      );
    }

    // load Q
    if (cute::elect_one_sync()) {
      cute::copy(
          mainloop_params.tma_load_q.with(*tma_barrier, mcast_mask),
          tQgQ_mkl(_, iter_index, _0{}, blk_coord_batch),
          tQsQ(_, pipeline_load_mma_q_producer_state.index())
      );
    }

    ++pipeline_load_mma_q_producer_state;

    pipeline_load_compute_lse.producer_acquire(pipeline_load_compute_lse_producer_state);

    // load LSE
    // 32 threads loading 128 values of 32b each
    // so 4*32b=128b

    int thread_idx = threadIdx.x % NumThreadsPerWarp;
    int smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * 4;
    int gmem_idx = TileShapeQ{} * iter_index + thread_idx * 4;
    auto mLSE = make_tensor(mainloop_args.ptr_lse, make_shape(Q, HB), mainloop_args.stride_lse);
    for (int i = 0; i < 4; i++) {
      cutlass::arch::cp_async_zfill<4>(
          shared_tensors.smem_lse.begin() + smem_idx + i,
          &mLSE(gmem_idx + i, blk_coord_batch),
          gmem_idx + i < Q
      );
    }

    pipeline_load_compute_lse.producer_commit(pipeline_load_compute_lse_producer_state, cutlass::arch::cpasync_barrier_arrive);
    ++pipeline_load_compute_lse_producer_state;


    pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
    tma_barrier = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);

    pipeline_load_mma_do.producer_expect_transaction(pipeline_load_mma_do_producer_state, kTransactionsBytesLoadV);

    // load V
    if (cute::elect_one_sync()) {
      cute::copy(
          mainloop_params.tma_load_v.with(*tma_barrier, mcast_mask),
          tVgV_mkl(_, blk_coord_k, _0{}, blk_coord_batch),
          tVsV(_, _0{})
      );
    }

    // load dO
    if (cute::elect_one_sync()) {
      cute::copy(
          mainloop_params.tma_load_do.with(*tma_barrier, mcast_mask),
          tDOgDO_mkl(_, iter_index, _0{}, blk_coord_batch),
          tDOsDO(_, pipeline_load_mma_do_producer_state.index())
      );
    }

    ++pipeline_load_mma_do_producer_state;

    pipeline_load_compute_sum_odo.producer_acquire(pipeline_load_compute_sum_odo_producer_state);

    // load sum_OdO
    smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * 4;
    gmem_idx = TileShapeQ{} * iter_index + thread_idx * 4;
    auto mSumOdO = make_tensor(mainloop_args.ptr_sum_odo, make_shape(Q, HB), mainloop_args.stride_sum_odo);
    for (int i = 0; i < 4; i++) {
      cutlass::arch::cp_async_zfill<4>(
          shared_tensors.smem_sum_odo.begin() + smem_idx + i,
          &mSumOdO(gmem_idx + i, blk_coord_batch),
          gmem_idx + i < Q
      );
    }

    pipeline_load_compute_sum_odo.producer_commit(pipeline_load_compute_sum_odo_producer_state, cutlass::arch::cpasync_barrier_arrive);
    ++pipeline_load_compute_sum_odo_producer_state;

    iter_count -= 1;
    iter_index += 1;

    while (iter_count > 0) {
      pipeline_load_mma_q.producer_acquire(pipeline_load_mma_q_producer_state);
      tma_barrier = pipeline_load_mma_q.producer_get_barrier(pipeline_load_mma_q_producer_state);

      // load Q
      if (cute::elect_one_sync()) {
        cute::copy(
            mainloop_params.tma_load_q.with(*tma_barrier, mcast_mask),
            tQgQ_mkl(_, iter_index, _0{}, blk_coord_batch),
            tQsQ(_, pipeline_load_mma_q_producer_state.index())
        );
      }

      ++pipeline_load_mma_q_producer_state;

      pipeline_load_compute_lse.producer_acquire(pipeline_load_compute_lse_producer_state);

      // load LSE
      smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * 4;
      gmem_idx = TileShapeQ{} * iter_index + thread_idx * 4;
      for (int i = 0; i < 4; i++) {
        cutlass::arch::cp_async_zfill<4>(
            shared_tensors.smem_lse.begin() + smem_idx + i,
            &mLSE(gmem_idx + i, blk_coord_batch),
            gmem_idx + i < Q
        );
      }

      pipeline_load_compute_lse.producer_commit(pipeline_load_compute_lse_producer_state, cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_load_compute_lse_producer_state;

      pipeline_load_mma_do.producer_acquire(pipeline_load_mma_do_producer_state);
      tma_barrier = pipeline_load_mma_do.producer_get_barrier(pipeline_load_mma_do_producer_state);

      // load dO
      if (cute::elect_one_sync()) {
        cute::copy(
            mainloop_params.tma_load_do.with(*tma_barrier, mcast_mask),
            tDOgDO_mkl(_, iter_index, _0{}, blk_coord_batch),
            tDOsDO(_, pipeline_load_mma_do_producer_state.index())
        );
      }

      ++pipeline_load_mma_do_producer_state;

      pipeline_load_compute_sum_odo.producer_acquire(pipeline_load_compute_sum_odo_producer_state);

      // load sum_OdO
      smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * 4;
      gmem_idx = TileShapeQ{} * iter_index + thread_idx * 4;
      for (int i = 0; i < 4; i++) {
        cutlass::arch::cp_async_zfill<4>(
            shared_tensors.smem_sum_odo.begin() + smem_idx + i,
            &mSumOdO(gmem_idx + i, blk_coord_batch),
            gmem_idx + i < Q
        );
      }

      pipeline_load_compute_sum_odo.producer_commit(pipeline_load_compute_sum_odo_producer_state, cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_load_compute_sum_odo_producer_state;

      iter_count -= 1;
      iter_index += 1;
    }
  }


  template<class BlkCoord, class ProblemShape_>
  CUTLASS_DEVICE void mma(
      BlkCoord const& blk_coord,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      TensorStorage& shared_tensors,
      PipelineLoadMmaQ& pipeline_load_mma_q,
      typename PipelineLoadMmaQ::PipelineState& pipeline_load_mma_q_consumer_state,
      PipelineLoadMmaDO& pipeline_load_mma_do,
      typename PipelineLoadMmaDO::PipelineState& pipeline_load_mma_do_consumer_state,
      PipelineMmaComputeS& pipeline_mma_compute_s,
      typename PipelineMmaComputeS::PipelineState& pipeline_mma_compute_s_producer_state,
      PipelineMmaComputeDP& pipeline_mma_compute_dp,
      typename PipelineMmaComputeDP::PipelineState& pipeline_mma_compute_dp_producer_state,
      PipelineMmaReduceDQ& pipeline_mma_reduce_dq,
      typename PipelineMmaReduceDQ::PipelineState& pipeline_mma_reduce_dq_producer_state,
      PipelineComputeMmaP& pipeline_compute_mma_p,
      typename PipelineComputeMmaP::PipelineState& pipeline_compute_mma_p_consumer_state,
      PipelineComputeMmaDS& pipeline_compute_mma_ds,
      typename PipelineComputeMmaDS::PipelineState& pipeline_compute_mma_ds_consumer_state,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_producer_state) {

    auto [Q, K, D, HB] = problem_shape;

    auto sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.begin()), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(shared_tensors.smem_v.begin()), SmemLayoutV{});
    auto sDO = make_tensor(make_smem_ptr(shared_tensors.smem_do.begin()), SmemLayoutDO{});

    auto sQT = make_tensor(make_smem_ptr(shared_tensors.smem_q_t.begin()), SmemLayoutQT{});
    auto sKT = make_tensor(make_smem_ptr(shared_tensors.smem_k_t.begin()), SmemLayoutKT{});
    auto sDS = make_tensor(make_smem_ptr(shared_tensors.smem_ds.begin()), SmemLayoutDS{});
    auto sDST = make_tensor(make_smem_ptr(shared_tensors.smem_ds_t.begin()), SmemLayoutDST{});
    auto sP = make_tensor(make_smem_ptr((Element*) nullptr), typename CollectiveMmaPDO::SmemLayoutA{});
    auto sDOT = make_tensor(make_smem_ptr(shared_tensors.smem_do_t.begin()), SmemLayoutDOT{});

    Tensor tSTrK = TiledMmaKQ::make_fragment_A(sK);
    Tensor tSTrQ = TiledMmaKQ::make_fragment_B(sQ);

    Tensor tDPTrV = TiledMmaVDO::make_fragment_A(sV);
    Tensor tDPTrDO = TiledMmaVDO::make_fragment_B(sDO);

    Tensor tDQrDS = TiledMmaDSK::make_fragment_A(sDS);
    Tensor tDQrKT = TiledMmaDSK::make_fragment_B(sKT);

    Tensor tDKrDST = TiledMmaDSQ::make_fragment_A(sDST);
    Tensor tDKrQT = TiledMmaDSQ::make_fragment_B(sQT);

    Tensor tDVrP = TiledMmaPDO::make_fragment_A(sP)(_, _, _, _0{});
    tDVrP.data() = TmemAllocation::kP;
    Tensor tDVrDOT = TiledMmaPDO::make_fragment_B(sDOT);

    TiledMmaKQ tiled_mma_kq;
    TiledMmaVDO tiled_mma_vdo;
    TiledMmaDSK tiled_mma_dsk;
    TiledMmaDSQ tiled_mma_dsq;
    TiledMmaPDO tiled_mma_pdo;

    tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::Zero;
    tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::Zero;

    Tensor tSTtST =  partition_fragment_C(tiled_mma_kq, select<0,1>(TileShapeKQ{}));
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT = partition_fragment_C(tiled_mma_vdo, select<0,1>(TileShapeVDO{}));
    tDPTtDPT.data() = TmemAllocation::kDP;

    Tensor tDQtDQ = partition_fragment_C(tiled_mma_dsk, select<0,1>(TileShapeDSK{}));
    tDQtDQ.data() = TmemAllocation::kDQ;

    Tensor tDKtDK = partition_fragment_C(tiled_mma_dsq, select<0,1>(TileShapeDSQ{}));
    tDKtDK.data() = TmemAllocation::kDK;

    Tensor tDVtDV = partition_fragment_C(tiled_mma_pdo, select<0,1>(TileShapePDO{}));
    tDVtDV.data() = TmemAllocation::kDV;

    auto pipeline_load_mma_q_release_state = pipeline_load_mma_q_consumer_state;

    pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);
    pipeline_mma_compute_s.producer_acquire(pipeline_mma_compute_s_producer_state);

    // S = Q*K
    tiled_mma_kq.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
      cute::gemm(tiled_mma_kq,
                 tSTrK(_,_,k_block,_0{}),
                 tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                 tSTtST);
      tiled_mma_kq.accumulate_ = UMMA::ScaleOut::One;
    }

    ++pipeline_load_mma_q_consumer_state;

    pipeline_mma_compute_s.producer_commit(pipeline_mma_compute_s_producer_state);
    ++pipeline_mma_compute_s_producer_state;

    pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);

    pipeline_mma_compute_dp.producer_acquire(pipeline_mma_compute_dp_producer_state);
    pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);

    // dP = dO*V
    tiled_mma_vdo.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
      cute::gemm(tiled_mma_vdo,
                 tDPTrV(_,_,k_block,_0{}),
                 tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                 tDPTtDPT);
      tiled_mma_vdo.accumulate_ = UMMA::ScaleOut::One;
    }

    pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
    ++pipeline_mma_compute_dp_producer_state;

    pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

    // dV = P*dO
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
      cute::gemm(tiled_mma_pdo,
                 tDVrP(_,_,k_block),
                 tDVrDOT(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                 tDVtDV);
      tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::One;
    }

    pipeline_compute_mma_p.consumer_release(pipeline_compute_mma_p_consumer_state);
    ++pipeline_compute_mma_p_consumer_state;

    pipeline_load_mma_do.consumer_release(pipeline_load_mma_do_consumer_state);
    ++pipeline_load_mma_do_consumer_state;

    iter_count -= 1;

    // in tmem, S & P overlap
    // and dP and dQ overlap
    // so we need to acquire dQ and dP at the same time
    while (iter_count > 0) {
      pipeline_load_mma_q.consumer_wait(pipeline_load_mma_q_consumer_state);
      pipeline_mma_compute_s.producer_acquire(pipeline_mma_compute_s_producer_state);

      // S = Q*K
      tiled_mma_kq.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
        cute::gemm(tiled_mma_kq,
                   tSTrK(_,_,k_block,_0{}),
                   tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tSTtST);
        tiled_mma_kq.accumulate_ = UMMA::ScaleOut::One;
      }

      ++pipeline_load_mma_q_consumer_state;

      pipeline_mma_compute_s.producer_commit(pipeline_mma_compute_s_producer_state);
      ++pipeline_mma_compute_s_producer_state;

      pipeline_compute_mma_ds.consumer_wait(pipeline_compute_mma_ds_consumer_state);

      // we need to acquire dP here, because tmem dQ == tmem dP
      pipeline_mma_compute_dp.producer_acquire(pipeline_mma_compute_dp_producer_state);

      // dQ = dS*K
      tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDQrDS); ++k_block) {
        cute::gemm(tiled_mma_dsk,
                   tDQrDS(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                   tDQrKT(_,_,k_block,_0{}),
                   tDQtDQ);
        tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_mma_reduce_dq.producer_commit(pipeline_mma_reduce_dq_producer_state);
      ++pipeline_mma_reduce_dq_producer_state;

      // dK = dS*Q
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDKrDST); ++k_block) {
        cute::gemm(tiled_mma_dsq,
                   tDKrDST(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                   tDKrQT(_,_,k_block,pipeline_load_mma_q_release_state.index()),
                   tDKtDK);
        tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_release_state);
      ++pipeline_load_mma_q_release_state;

      pipeline_compute_mma_ds.consumer_release(pipeline_compute_mma_ds_consumer_state);
      ++pipeline_compute_mma_ds_consumer_state;

      // we grab dq here, because in tmem dq == dp
      pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);

      pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);

      // dP = dO*V
      tiled_mma_vdo.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
        cute::gemm(tiled_mma_vdo,
                   tDPTrV(_,_,k_block,_0{}),
                   tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                   tDPTtDPT);
        tiled_mma_vdo.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
      ++pipeline_mma_compute_dp_producer_state;

      pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

      // dV = P*dO
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
        cute::gemm(tiled_mma_pdo,
                   tDVrP(_,_,k_block),
                   tDVrDOT(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                   tDVtDV);
        tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_compute_mma_p.consumer_release(pipeline_compute_mma_p_consumer_state);
      ++pipeline_compute_mma_p_consumer_state;

      pipeline_load_mma_do.consumer_release(pipeline_load_mma_do_consumer_state);
      ++pipeline_load_mma_do_consumer_state;

      iter_count -= 1;
    }

    // signal to the epilogue that dV is ready
    pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);
    pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
    ++pipeline_mma_compute_dkdv_producer_state;

    pipeline_mma_compute_dkdv.producer_acquire(pipeline_mma_compute_dkdv_producer_state);

    pipeline_compute_mma_ds.consumer_wait(pipeline_compute_mma_ds_consumer_state);

    // dK = dS*Q
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDKrDST); ++k_block) {
      cute::gemm(tiled_mma_dsq,
                 tDKrDST(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                 tDKrQT(_,_,k_block,pipeline_load_mma_q_release_state.index()),
                 tDKtDK);
      tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::One;
    }

    // signal to epilgue that dK is ready
    pipeline_mma_compute_dkdv.producer_commit(pipeline_mma_compute_dkdv_producer_state);
    ++pipeline_mma_compute_dkdv_producer_state;

    // we've already acquired mma_reduce_dq in the loop

    // dQ = dS*K
    tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDQrDS); ++k_block) {
      cute::gemm(tiled_mma_dsk,
                 tDQrDS(_,_,k_block,pipeline_compute_mma_ds_consumer_state.index()),
                 tDQrKT(_,_,k_block,_0{}),
                 tDQtDQ);
      tiled_mma_dsk.accumulate_ = UMMA::ScaleOut::One;
    }

    pipeline_mma_reduce_dq.producer_commit(pipeline_mma_reduce_dq_producer_state);
    ++pipeline_mma_reduce_dq_producer_state;

    pipeline_load_mma_q.consumer_release(pipeline_load_mma_q_release_state);
    ++pipeline_load_mma_q_release_state;

    pipeline_compute_mma_ds.consumer_release(pipeline_compute_mma_ds_consumer_state);
    ++pipeline_compute_mma_ds_consumer_state;
  }



  template<class TensorG, class TensorR, class TensorC, class TensorShape>
  CUTLASS_DEVICE void store(
      TensorG gmem,
      TensorR const& regs,
      TensorC const& coord,
      TensorShape const& tensor_shape) {

    Tensor preds = cute::lazy::transform(coord, [&](auto const& c) { return elem_less(c, tensor_shape); });

    auto copy_op = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        make_layout(make_shape(_1{}, Int<sizeof(uint128_t) / sizeof(Element)>{})),
        regs.layout()
    );
    auto thr_copy = copy_op.get_slice(_0{});

    Tensor quantized_regs = quantize(regs);
    Tensor tCr = thr_copy.partition_S(quantized_regs);
    Tensor tCg = thr_copy.partition_D(gmem);
    Tensor tPc = thr_copy.partition_D(preds);

    copy_if(copy_op, tPc, tCr, tCg);
  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void epilogue_clear(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args) {

    auto [Q, K, D, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch] = blk_coord;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,3>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,2,3>(blk_offset), mDV_in);
    auto gDV = local_tile(mDV, TileShapePDO{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDV = domain_offset(
        make_coord(blk_coord_k * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapePDO{}))
    );
    
    for (int i = threadIdx.x; i < size(gDK); i += blockDim.x) {
      if (elem_less(cDK(i), select<1,2>(problem_shape))) {
        gDK(i) = Element(0);
      }
    }
    for (int i = threadIdx.x; i < size(gDV); i += blockDim.x) {
      if (elem_less(cDV(i), select<1,2>(problem_shape))) {
        gDV(i) = Element(0);
      }
    }
  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void epilogue(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_consumer_state) {

    auto [Q, K, D, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch] = blk_coord;

    auto load_op = SM100_TMEM_LOAD_32dp32b16x{};

    auto tDKtDK = partition_fragment_C(TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
    tDKtDK.data() = TmemAllocation::kDK;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,3>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    constexpr int kNumWarpgroups = kNumComputeWarps / 4;
    int dp_idx = threadIdx.x % 128;
    int wg_idx = (threadIdx.x % (kNumComputeWarps * NumThreadsPerWarp)) / 128;

    auto split_wg = [&](auto const& t) {
      if constexpr (decltype(rank(t))::value == 3) {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, make_coord(wg_idx, _));
      }
      else {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), size<2>(t), make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, _, make_coord(wg_idx, _));
      }
    };

    auto tiled_t2r_dk = make_tmem_copy(load_op, tDKtDK);
    auto thread_t2r_dk = tiled_t2r_dk.get_slice(dp_idx);

    Tensor tTR_cDK   = split_wg(thread_t2r_dk.partition_D(cDK));
    Tensor tTR_gDK   = split_wg(thread_t2r_dk.partition_D(gDK));
    Tensor tTR_rDK = make_tensor<ElementAcc>(shape(tTR_cDK));
    Tensor tTR_tDK = split_wg(thread_t2r_dk.partition_S(tDKtDK));

    auto tDVtDV = partition_fragment_C(TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
    tDVtDV.data() = TmemAllocation::kDV;

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,2,3>(blk_offset), mDV_in);
    auto gDV = local_tile(mDV, TileShapePDO{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDV = domain_offset(
        make_coord(blk_coord_k * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapePDO{}))
    );

    auto tiled_t2r_dv = make_tmem_copy(load_op, tDVtDV);
    auto thread_t2r_dv = tiled_t2r_dv.get_slice(dp_idx);

    Tensor tTR_cDV   = split_wg(thread_t2r_dv.partition_D(cDV));
    Tensor tTR_gDV   = split_wg(thread_t2r_dv.partition_D(gDV));
    Tensor tTR_rDV = make_tensor<ElementAcc>(shape(tTR_cDV));
    Tensor tTR_tDV = split_wg(thread_t2r_dv.partition_S(tDVtDV));

    pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

    // load tDVtDV
    cute::copy(tiled_t2r_dv, tTR_tDV, tTR_rDV);

    // store tDVgDV
    store(tTR_gDV, tTR_rDV, tTR_cDV, select<1,2>(problem_shape));

    cutlass::arch::fence_view_async_tmem_load();
    pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
    ++pipeline_mma_compute_dkdv_consumer_state;

    pipeline_mma_compute_dkdv.consumer_wait(pipeline_mma_compute_dkdv_consumer_state);

    // load tDKtDK
    cute::copy(tiled_t2r_dk, tTR_tDK, tTR_rDK);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rDK); i++) {
      tTR_rDK(i) = mainloop_args.softmax_scale * tTR_rDK(i);
    }

    // store tDKgDK
    store(tTR_gDK, tTR_rDK, tTR_cDK, select<1,2>(problem_shape));

    cutlass::arch::fence_view_async_tmem_load();
    pipeline_mma_compute_dkdv.consumer_release(pipeline_mma_compute_dkdv_consumer_state);
    ++pipeline_mma_compute_dkdv_consumer_state;

  }


  template<class BlkCoord, class BlkOffset, class ProblemShape_>
  CUTLASS_DEVICE void compute(
      BlkCoord const& blk_coord,
      BlkOffset const& blk_offset,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      EpilogueArguments const& epilogue_args,
      TensorStorage& shared_tensors,
      PipelineLoadComputeLSE& pipeline_load_compute_lse,
      typename PipelineLoadComputeLSE::PipelineState& pipeline_load_compute_lse_consumer_state,
      PipelineLoadComputeSumOdO& pipeline_load_compute_sum_odo,
      typename PipelineLoadComputeSumOdO::PipelineState& pipeline_load_compute_sum_odo_consumer_state,
      PipelineMmaComputeS& pipeline_mma_compute_s,
      typename PipelineMmaComputeS::PipelineState& pipeline_mma_compute_s_consumer_state,
      PipelineMmaComputeDP& pipeline_mma_compute_dp,
      typename PipelineMmaComputeDP::PipelineState& pipeline_mma_compute_dp_consumer_state,
      PipelineComputeMmaP& pipeline_compute_mma_p,
      typename PipelineComputeMmaP::PipelineState& pipeline_compute_mma_p_producer_state,
      PipelineComputeMmaDS& pipeline_compute_mma_ds,
      typename PipelineComputeMmaDS::PipelineState& pipeline_compute_mma_ds_producer_state,
      PipelineMmaComputeDKDV& pipeline_mma_compute_dkdv,
      typename PipelineMmaComputeDKDV::PipelineState& pipeline_mma_compute_dkdv_consumer_state) {


    auto [Q, K, D, HB] = problem_shape;

    // in tmem, S & P overlap
    // and dP and dQ overlap

    // there are two compute wg's that cooperatively compute softmax
    // they are striped by this tmem atom, i.e. wg0 has 16 elems, then wg1 etc

    auto load_op = SM100_TMEM_LOAD_32dp32b16x{};
    auto store_op = []() {
      if constexpr (sizeof(Element) == 1) {
        return SM100_TMEM_STORE_32dp32b4x{};
      }
      else {
        return SM100_TMEM_STORE_32dp32b8x{};
      }
    }();

    Tensor tSTtST =  partition_fragment_C(TiledMmaKQ{}, select<0,1>(TileShapeKQ{}))(make_coord(_,_),_0{},_0{});
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT =  partition_fragment_C(TiledMmaVDO{}, select<0,1>(TileShapeVDO{}))(make_coord(_,_),_0{},_0{});
    tDPTtDPT.data() = TmemAllocation::kDP;

    Tensor cST = make_identity_tensor(take<0,2>(TileShapeKQ{}));
    Tensor cDPT = make_identity_tensor(take<0,2>(TileShapeVDO{}));

    constexpr int kNumWarpgroups = kNumComputeWarps / 4;
    int dp_idx = threadIdx.x % 128;
    int wg_idx = (threadIdx.x % (kNumComputeWarps * NumThreadsPerWarp)) / 128;
    auto tiled_t2r = make_tmem_copy(load_op, tSTtST);
    auto thread_t2r = tiled_t2r.get_slice(dp_idx);

    auto split_wg = [&](auto const& t) {
      if constexpr (decltype(size<1>(t))::value > 1) {
        if constexpr (decltype(rank(t))::value == 3) {
          auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t))));
          return p(_, make_coord(wg_idx, _), _);
        }
        else {
          auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t), size<3>(t))));
          return p(_, make_coord(wg_idx, _), _, _);
        }
      }
      else {
        if constexpr (decltype(rank(t))::value == 3) {
          auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
          return p(_, _, make_coord(wg_idx, _));
        }
        else {
          auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), size<2>(t), make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
          return p(_, _, _, make_coord(wg_idx, _));
        }

      }
    };


    Tensor tTR_cST_p = thread_t2r.partition_D(cST);
    Tensor tTR_cST   = split_wg(tTR_cST_p);
    Tensor tTR_rST = make_tensor<ElementAcc>(shape(tTR_cST));
    Tensor tTR_tST = split_wg(thread_t2r.partition_S(tSTtST));

    Tensor tTR_cDPT_p = thread_t2r.partition_D(cDPT);
    Tensor tTR_cDPT = split_wg(tTR_cDPT_p);
    Tensor tTR_rDPT = make_tensor<ElementAcc>(shape(tTR_cDPT));
    Tensor tTR_tDPT = split_wg(thread_t2r.partition_S(tDPTtDPT));

    Tensor sLSE = make_tensor(make_smem_ptr(shared_tensors.smem_lse.begin()), SmemLayoutLSE{});
    Tensor sSumOdO = make_tensor(make_smem_ptr(shared_tensors.smem_sum_odo.begin()), SmemLayoutSumOdO{});

    auto sP = make_tensor(make_smem_ptr((Element*) nullptr), typename CollectiveMmaPDO::SmemLayoutA{});

    auto tDVrP = TiledMmaPDO::make_fragment_A(sP)(_, _, _, _0{});
    auto tDVcST = TiledMmaPDO{}.get_slice(_0{}).partition_A(cST);
    tDVrP.data() = TmemAllocation::kP;

    auto tiled_r2t = make_tmem_copy(store_op, tDVrP);
    auto thread_r2t = tiled_r2t.get_slice(dp_idx);

    auto tRT_tP = split_wg(thread_r2t.partition_D(tDVrP));
    auto tRT_cST_p = thread_r2t.partition_S(tDVcST);
    auto tRT_cST = split_wg(tRT_cST_p);

    bool is_residual_k = get<1>(blk_coord) * TileShapeK{} + TileShapeK{} >= get<1>(problem_shape);
    int last_iter = iter_count - 1 + iter_index;

    CUTLASS_PRAGMA_NO_UNROLL
    while (iter_count > 0) {
      // wait for S and P
      pipeline_mma_compute_s.consumer_wait(pipeline_mma_compute_s_consumer_state);
      pipeline_compute_mma_p.producer_acquire(pipeline_compute_mma_p_producer_state);
      // wait for LSE
      pipeline_load_compute_lse.consumer_wait(pipeline_load_compute_lse_consumer_state);

      auto dispatch_bool = [](bool b, auto fn) {
        if (b) {
          fn(cute::true_type{});
        }
        else {
          fn(cute::false_type{});
        }
      };

      bool leading_causal_masking = false;
      if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask> 
                    || std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
        leading_causal_masking = warp_uniform(iter_index == get<1>(blk_coord));
      }
      bool trailing_residual_masking = false;
      if constexpr (std::is_base_of_v<cutlass::fmha::collective::ResidualMaskForBackward, Mask>) {
        trailing_residual_masking = warp_uniform((iter_index == last_iter) || is_residual_k);
      }

      dispatch_bool(leading_causal_masking || trailing_residual_masking, [&](auto is_masked_tile) {

        // compute P = softmax(S, LSE)
        cute::copy(tiled_t2r, tTR_tST, tTR_rST);

        if constexpr (decltype(is_masked_tile)::value) {
          Mask{}.apply_mask(tTR_rST, [&](int i) {
            auto c_transpose = tTR_cST(i);
            return make_coord(get<1>(c_transpose) + iter_index * TileShapeQ{}, get<0>(c_transpose) + get<1>(blk_coord) * TileShapeK{});
          }, problem_shape);
        }

        ElementAcc log2_e = static_cast<ElementAcc>(M_LOG2E);
        float2 softmax_scale_log2_e;
        softmax_scale_log2_e.x = mainloop_args.softmax_scale * log2_e;
        softmax_scale_log2_e.y = mainloop_args.softmax_scale * log2_e;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTR_rST); i += 2) {
          float2 acc;
          float2 lse;
          float2 out;
          acc.x = tTR_rST(i);
          acc.y = tTR_rST(i + 1);
          lse.x = sLSE(get<1>(tTR_cST(i)), pipeline_load_compute_lse_consumer_state.index());
          lse.y = sLSE(get<1>(tTR_cST(i+1)), pipeline_load_compute_lse_consumer_state.index());
          cute::fma(out, softmax_scale_log2_e, acc, lse);
          tTR_rST(i) = ::exp2f(out.x);
          tTR_rST(i+1) = ::exp2f(out.y);
        }

        auto tRT_rST = quantize(tTR_rST);
        auto tRT_rST_reshaped = make_tensor(tRT_rST.data(), shape(tRT_cST));

        cutlass::arch::fence_view_async_tmem_load();
        cutlass::arch::NamedBarrier(
          kNumComputeWarps * NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::TransformBarrier
        ).arrive_and_wait();

        cute::copy(tiled_r2t, tRT_rST_reshaped, tRT_tP);
      });

      // notify for P
      cutlass::arch::fence_view_async_tmem_store();
      pipeline_compute_mma_p.producer_commit(pipeline_compute_mma_p_producer_state);
      ++pipeline_compute_mma_p_producer_state;
      // release S
      pipeline_mma_compute_s.consumer_release(pipeline_mma_compute_s_consumer_state);
      ++pipeline_mma_compute_s_consumer_state;
      // release LSE
      pipeline_load_compute_lse.consumer_release(pipeline_load_compute_lse_consumer_state);
      ++pipeline_load_compute_lse_consumer_state;

      // wait for OdO
      pipeline_load_compute_sum_odo.consumer_wait(pipeline_load_compute_sum_odo_consumer_state);
      // wait for dP
      pipeline_mma_compute_dp.consumer_wait(pipeline_mma_compute_dp_consumer_state);

      // wait for dS
      // in principle, we could defer waiting for dS, and move in the freeing of dP
      // however, that would force us to keep dS in registers longer
      pipeline_compute_mma_ds.producer_acquire(pipeline_compute_mma_ds_producer_state);

      // compute dS = dsoftmax(P, dP, sum_OdO)
      cute::copy(tiled_t2r, tTR_tDPT, tTR_rDPT);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rDPT); i += 2) {
        float2 st;
        st.x = tTR_rST(i);
        st.y = tTR_rST(i+1);
        float2 dpt;
        dpt.x = tTR_rDPT(i);
        dpt.y = tTR_rDPT(i+1);
        float2 odo;
        odo.x = sSumOdO(get<1>(tTR_cDPT(i)), pipeline_load_compute_sum_odo_consumer_state.index());
        odo.y = sSumOdO(get<1>(tTR_cDPT(i+1)), pipeline_load_compute_sum_odo_consumer_state.index());
        float2 dif;
        // sum odo is negated during preprocess
        cute::add(dif, dpt, odo);
        float2 out;
        cute::mul(out, dif, st);
        tTR_rDPT(i) = out.x;
        tTR_rDPT(i+1) = out.y;
      }

      auto tTR_rDST = quantize(tTR_rDPT);

      // release dP
      cutlass::arch::fence_view_async_tmem_load();
      pipeline_mma_compute_dp.consumer_release(pipeline_mma_compute_dp_consumer_state);
      ++pipeline_mma_compute_dp_consumer_state;

      Tensor sDS = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_ds.begin()), SmemLayoutDS{})
          (_, _, _, pipeline_compute_mma_ds_producer_state.index());

      auto thread_layout = make_ordered_layout(
          make_shape(_128{}, _128{}),
          make_stride(_1{}, _0{})
      );

      auto sDS_pi = as_position_independent_swizzle_tensor(sDS);
      auto sDS_pi_slice_p = sDS_pi.compose(thread_layout)(dp_idx, _).compose(make_layout(shape(tTR_cDPT_p)));
      auto sDS_pi_slice = split_wg(sDS_pi_slice_p);

      copy_aligned(tTR_rDST, sDS_pi_slice);

      // notify for dS
      cutlass::arch::fence_view_async_shared();
      pipeline_compute_mma_ds.producer_commit(pipeline_compute_mma_ds_producer_state);
      ++pipeline_compute_mma_ds_producer_state;
      // release OdO
      pipeline_load_compute_sum_odo.consumer_release(pipeline_load_compute_sum_odo_consumer_state);
      ++pipeline_load_compute_sum_odo_consumer_state;

      iter_count -= 1;
      iter_index += 1;
    }

    epilogue(
        blk_coord, blk_offset, problem_shape, mainloop_args, epilogue_args,
        pipeline_mma_compute_dkdv, pipeline_mma_compute_dkdv_consumer_state
    );
  }

  template<class BlkCoord, class ProblemShape_>
  CUTLASS_DEVICE void reduce(
      BlkCoord const& blk_coord,
      ProblemShape_ const& problem_shape,
      int iter_index,
      int iter_count,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineMmaReduceDQ& pipeline_mma_reduce_dq,
      typename PipelineMmaReduceDQ::PipelineState& pipeline_mma_reduce_dq_consumer_state,
      PipelineReduceTmaStore& pipeline_reduce_tma_store,
      typename PipelineReduceTmaStore::PipelineState& pipeline_reduce_tma_store_producer_state) {

    using X = Underscore;

    auto [Q, K, D, HB] = problem_shape;

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_batch] = blk_coord;

    // must match TileShapeDQ
    auto load_op = SM100_TMEM_LOAD_32dp32b32x{};

    auto tDQtDQ = partition_fragment_C(TiledMmaDSK{}, select<0,1>(TileShapeDSK{}))(make_coord(_,_),_0{},_0{});
    tDQtDQ.data() = TmemAllocation::kDQ;

    Tensor mDQ = mainloop_params.tma_red_dq.get_tma_tensor(make_shape(Q, D, HB));
    auto gDQ = local_tile(mDQ, TileShapeKQ{}, make_coord(_,_,_), Step<X, _1, _1>{})
        (_, _, _, _0{}, blk_coord_batch);

    Tensor cDQ = make_identity_tensor(take<0,2>(TileShapeDSK{}));

    Tensor sDQ = make_tensor(make_smem_ptr(shared_tensors.smem_dq.begin()), SmemLayoutDQ{});

    int thread_idx = threadIdx.x % (kNumComputeWarps * NumThreadsPerWarp);
    auto tiled_t2r = make_tmem_copy(load_op, tDQtDQ);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);

    Tensor tTR_cDQ   = thread_t2r.partition_D(cDQ);
    Tensor tTR_gDQ   = thread_t2r.partition_D(gDQ);
    Tensor tTR_sDQ   = thread_t2r.partition_D(sDQ);
    Tensor tTR_tDQ = thread_t2r.partition_S(tDQtDQ);

    auto block_tma = mainloop_params.tma_red_dq.get_slice(_0{});

    Tensor tDQsDQ = block_tma.partition_S(sDQ);
    Tensor tDQcDQ = block_tma.partition_S(cDQ);
    Tensor tDQgDQ = block_tma.partition_D(gDQ);

    int lane_predicate = (threadIdx.x % (kNumReduceWarps * NumThreadsPerWarp)) == 0;

    while (iter_count > 0) {
      pipeline_mma_reduce_dq.consumer_wait(pipeline_mma_reduce_dq_consumer_state);

      Tensor tTR_rDQ = make_tensor<ElementAcc>(shape(tTR_cDQ));

      // load dQ from tmem to rmem
      cute::copy(tiled_t2r, tTR_tDQ, tTR_rDQ);

      cutlass::arch::fence_view_async_tmem_load();
      pipeline_mma_reduce_dq.consumer_release(pipeline_mma_reduce_dq_consumer_state);
      ++pipeline_mma_reduce_dq_consumer_state;

      // we don't have enough smem to dump it all to smem, so we do it in stages
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<2>(tTR_cDQ); i++) {
        if (lane_predicate) {
          pipeline_reduce_tma_store.producer_acquire(pipeline_reduce_tma_store_producer_state);
        }
        // wait in all threads for the acquire to complete
        cutlass::arch::NamedBarrier(
            kNumReduceWarps * NumThreadsPerWarp,
            cutlass::arch::ReservedNamedBarriers::TransposeBarrier
        ).arrive_and_wait();

        cute::copy(tTR_rDQ(_, _, i), tTR_sDQ(_, _, _0{}, pipeline_reduce_tma_store_producer_state.index()));

        // wait for the stores to all be visible to the TMA
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier(
            kNumReduceWarps * NumThreadsPerWarp,
            cutlass::arch::ReservedNamedBarriers::TransposeBarrier
        ).arrive_and_wait();
        if (lane_predicate) {
          // launch tma store
          copy(mainloop_params.tma_red_dq, tDQsDQ(_,_,_0{}, pipeline_reduce_tma_store_producer_state.index()), tDQgDQ(_,_,i,iter_index));
          pipeline_reduce_tma_store.producer_commit(pipeline_reduce_tma_store_producer_state);
        }

        ++pipeline_reduce_tma_store_producer_state;
      }

      iter_count -= 1;
      iter_index += 1;
    }
  }


  CUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    if (role == WarpRole::Load && lane_predicate) {
      prefetch_tma_descriptor(params.mainloop_params.tma_load_q.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_k.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_v.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_do.get_tma_descriptor());
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    int initializing_warp = 0;
    typename PipelineLoadMmaQ::Params pipeline_load_mma_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_mma_q_params.role = PipelineLoadMmaQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::Mma) {
      pipeline_load_mma_q_params.role = PipelineLoadMmaQ::ThreadCategory::Consumer;
    }
    pipeline_load_mma_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
    // Also loads K in the first iteration
    pipeline_load_mma_q_params.transaction_bytes = kTransactionsBytesLoadQ;
    pipeline_load_mma_q_params.initializing_warp = initializing_warp++;
    PipelineLoadMmaQ pipeline_load_mma_q(shared_storage.pipelines.load_mma_q, pipeline_load_mma_q_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineLoadMmaDO::Params pipeline_load_mma_do_params;
    if (role == WarpRole::Load) {
      pipeline_load_mma_do_params.role = PipelineLoadMmaDO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Mma) {
      pipeline_load_mma_do_params.role = PipelineLoadMmaDO::ThreadCategory::Consumer;
    }
    pipeline_load_mma_do_params.is_leader = lane_predicate && (role == WarpRole::Load);
    // Also loads V in the first iteration
    pipeline_load_mma_do_params.transaction_bytes = kTransactionsBytesLoadDO;
    pipeline_load_mma_do_params.initializing_warp = initializing_warp++;
    PipelineLoadMmaDO pipeline_load_mma_do(shared_storage.pipelines.load_mma_do, pipeline_load_mma_do_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineLoadComputeLSE::Params pipeline_load_compute_lse_params;
    if (role == WarpRole::Load) {
      pipeline_load_compute_lse_params.role = PipelineLoadComputeLSE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_load_compute_lse_params.role = PipelineLoadComputeLSE::ThreadCategory::Consumer;
    }
    pipeline_load_compute_lse_params.producer_arv_count = NumThreadsPerWarp;
    pipeline_load_compute_lse_params.consumer_arv_count = kNumComputeWarps * NumThreadsPerWarp;
    pipeline_load_compute_lse_params.initializing_warp = initializing_warp++;
    PipelineLoadComputeLSE pipeline_load_compute_lse(
      shared_storage.pipelines.load_compute_lse,
      pipeline_load_compute_lse_params,
      /*barrier init*/ cute::true_type{});

    typename PipelineLoadComputeSumOdO::Params pipeline_load_compute_sum_odo_params;
    if (role == WarpRole::Load) {
      pipeline_load_compute_sum_odo_params.role = PipelineLoadComputeSumOdO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_load_compute_sum_odo_params.role = PipelineLoadComputeSumOdO::ThreadCategory::Consumer;
    }
    pipeline_load_compute_sum_odo_params.producer_arv_count = NumThreadsPerWarp;
    pipeline_load_compute_sum_odo_params.consumer_arv_count = kNumComputeWarps * NumThreadsPerWarp;
    pipeline_load_compute_sum_odo_params.initializing_warp = initializing_warp++;
    PipelineLoadComputeSumOdO pipeline_load_compute_sum_odo(
      shared_storage.pipelines.load_compute_sum_odo,
      pipeline_load_compute_sum_odo_params,
      /*barrier init*/ cute::true_type{});

    typename PipelineMmaComputeS::Params pipeline_mma_compute_s_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_s_params.role = PipelineMmaComputeS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_s_params.role = PipelineMmaComputeS::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_s_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_s_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeS pipeline_mma_compute_s(
      shared_storage.pipelines.mma_compute_s,
      pipeline_mma_compute_s_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaComputeDP::Params pipeline_mma_compute_dp_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_dp_params.role = PipelineMmaComputeDP::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_dp_params.role = PipelineMmaComputeDP::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_dp_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_dp_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeDP pipeline_mma_compute_dp(
      shared_storage.pipelines.mma_compute_dp,
      pipeline_mma_compute_dp_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaReduceDQ::Params pipeline_mma_reduce_dq_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_reduce_dq_params.role = PipelineMmaReduceDQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::Reduce) {
      pipeline_mma_reduce_dq_params.role = PipelineMmaReduceDQ::ThreadCategory::Consumer;
    }
    pipeline_mma_reduce_dq_params.consumer_arv_count = kNumReduceWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_reduce_dq_params.initializing_warp = initializing_warp++;
    PipelineMmaReduceDQ pipeline_mma_reduce_dq(
      shared_storage.pipelines.mma_reduce_dq,
      pipeline_mma_reduce_dq_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineComputeMmaP::Params pipeline_compute_mma_p_params;
    if (role == WarpRole::Mma) {
      pipeline_compute_mma_p_params.role = PipelineComputeMmaP::ThreadCategory::Consumer;
    }
    if (role == WarpRole::Compute) {
      pipeline_compute_mma_p_params.role = PipelineComputeMmaP::ThreadCategory::Producer;
    }
    pipeline_compute_mma_p_params.producer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_compute_mma_p_params.consumer_arv_count = 1;
    pipeline_compute_mma_p_params.initializing_warp = initializing_warp++;
    PipelineComputeMmaP pipeline_compute_mma_p(
      shared_storage.pipelines.compute_mma_p,
      pipeline_compute_mma_p_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineComputeMmaDS::Params pipeline_compute_mma_ds_params;
    if (role == WarpRole::Mma) {
      pipeline_compute_mma_ds_params.role = PipelineComputeMmaDS::ThreadCategory::Consumer;
    }
    if (role == WarpRole::Compute) {
      pipeline_compute_mma_ds_params.role = PipelineComputeMmaDS::ThreadCategory::Producer;
    }
    pipeline_compute_mma_ds_params.producer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_compute_mma_ds_params.consumer_arv_count = 1;
    pipeline_compute_mma_ds_params.initializing_warp = initializing_warp++;
    PipelineComputeMmaDS pipeline_compute_mma_ds(
      shared_storage.pipelines.compute_mma_ds,
      pipeline_compute_mma_ds_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineMmaComputeDKDV::Params pipeline_mma_compute_dkdv_params;
    if (role == WarpRole::Mma) {
      pipeline_mma_compute_dkdv_params.role = PipelineMmaComputeDKDV::ThreadCategory::Producer;
    }
    if (role == WarpRole::Compute) {
      pipeline_mma_compute_dkdv_params.role = PipelineMmaComputeDKDV::ThreadCategory::Consumer;
    }
    pipeline_mma_compute_dkdv_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp;
    pipeline_mma_compute_dkdv_params.initializing_warp = initializing_warp++;
    PipelineMmaComputeDKDV pipeline_mma_compute_dkdv(
      shared_storage.pipelines.mma_compute_dkdv,
      pipeline_mma_compute_dkdv_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});
    PipelineReduceTmaStore pipeline_reduce_tma_store;

    TmemAllocator tmem_allocator;

    pipeline_init_arrive_relaxed(size(ClusterShape{}));

    pipeline_load_mma_q.init_masks(ClusterShape{});
    pipeline_load_mma_do.init_masks(ClusterShape{});
    pipeline_mma_compute_s.init_masks(ClusterShape{});
    pipeline_mma_compute_dp.init_masks(ClusterShape{});
    pipeline_mma_reduce_dq.init_masks(ClusterShape{});
    pipeline_compute_mma_p.init_masks(ClusterShape{});
    pipeline_compute_mma_ds.init_masks(ClusterShape{});
    pipeline_mma_compute_dkdv.init_masks(ClusterShape{});

    typename decltype(pipeline_load_mma_q)::PipelineState pipeline_load_mma_q_consumer_state;
    typename decltype(pipeline_load_mma_do)::PipelineState pipeline_load_mma_do_consumer_state;
    typename decltype(pipeline_load_compute_lse)::PipelineState pipeline_load_compute_lse_consumer_state;
    typename decltype(pipeline_load_compute_sum_odo)::PipelineState pipeline_load_compute_sum_odo_consumer_state;
    typename decltype(pipeline_mma_compute_s)::PipelineState pipeline_mma_compute_s_consumer_state;
    typename decltype(pipeline_mma_compute_dp)::PipelineState pipeline_mma_compute_dp_consumer_state;
    typename decltype(pipeline_mma_reduce_dq)::PipelineState pipeline_mma_reduce_dq_consumer_state;
    typename decltype(pipeline_compute_mma_p)::PipelineState pipeline_compute_mma_p_consumer_state;
    typename decltype(pipeline_compute_mma_ds)::PipelineState pipeline_compute_mma_ds_consumer_state;
    typename decltype(pipeline_mma_compute_dkdv)::PipelineState pipeline_mma_compute_dkdv_consumer_state;

    auto pipeline_load_mma_q_producer_state = make_producer_start_state<decltype(pipeline_load_mma_q)>();
    auto pipeline_load_mma_do_producer_state = make_producer_start_state<decltype(pipeline_load_mma_do)>();
    auto pipeline_load_compute_lse_producer_state = make_producer_start_state<decltype(pipeline_load_compute_lse)>();
    auto pipeline_load_compute_sum_odo_producer_state = make_producer_start_state<decltype(pipeline_load_compute_sum_odo)>();
    auto pipeline_mma_compute_s_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_s)>();
    auto pipeline_mma_compute_dp_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_dp)>();
    auto pipeline_mma_reduce_dq_producer_state = make_producer_start_state<decltype(pipeline_mma_reduce_dq)>();
    auto pipeline_compute_mma_p_producer_state = make_producer_start_state<decltype(pipeline_compute_mma_p)>();
    auto pipeline_compute_mma_ds_producer_state = make_producer_start_state<decltype(pipeline_compute_mma_ds)>();
    auto pipeline_mma_compute_dkdv_producer_state = make_producer_start_state<decltype(pipeline_mma_compute_dkdv)>();
    auto pipeline_reduce_tma_store_producer_state = make_producer_start_state<decltype(pipeline_reduce_tma_store)>();

    pipeline_init_wait(size(ClusterShape{}));

    auto blk_coord = make_coord(_0{}, blockIdx.x, _0{}, make_coord(blockIdx.y, blockIdx.z));
    auto [problem_shape, blk_offset] = apply_variable_length_offset(
        params.problem_shape,
        blk_coord
    );
    int iter_count = ceil_div(get<0>(problem_shape), TileShapeQ{});
    int iter_start = 0;
    if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask> || 
                  std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
      iter_start = (get<1>(blk_coord) * TileShapeK{}) / TileShapeQ{};
    }
    if (get<1>(blk_coord) * TileShapeK{} >= get<1>(problem_shape)) {
      return;
    }
    iter_count -= iter_start;

    if (iter_count <= 0) {
      epilogue_clear(
          blk_coord,
          blk_offset,
          problem_shape,
          params.mainloop,
          params.epilogue
      );
      return;
    }

    if (role == WarpRole::Load) {
      warpgroup_reg_set<RegisterAllocation::kLoad>();

      load(
          blk_coord,
          blk_offset,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.mainloop_params,
          shared_storage.tensors,
          pipeline_load_mma_q, pipeline_load_mma_q_producer_state,
          pipeline_load_mma_do, pipeline_load_mma_do_producer_state,
          pipeline_load_compute_lse, pipeline_load_compute_lse_producer_state,
          pipeline_load_compute_sum_odo, pipeline_load_compute_sum_odo_producer_state
      );

    }
    else if (role == WarpRole::Mma) {
      warpgroup_reg_set<RegisterAllocation::kMma>();

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();

      mma(
          blk_coord,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          shared_storage.tensors,
          pipeline_load_mma_q, pipeline_load_mma_q_consumer_state,
          pipeline_load_mma_do, pipeline_load_mma_do_consumer_state,
          pipeline_mma_compute_s, pipeline_mma_compute_s_producer_state,
          pipeline_mma_compute_dp, pipeline_mma_compute_dp_producer_state,
          pipeline_mma_reduce_dq, pipeline_mma_reduce_dq_producer_state,
          pipeline_compute_mma_p, pipeline_compute_mma_p_consumer_state,
          pipeline_compute_mma_ds, pipeline_compute_mma_ds_consumer_state,
          pipeline_mma_compute_dkdv, pipeline_mma_compute_dkdv_producer_state
      );

    }
    else if (role == WarpRole::Compute) {
      warpgroup_reg_set<RegisterAllocation::kCompute>();

      compute(
          blk_coord,
          blk_offset,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.epilogue,
          shared_storage.tensors,
          pipeline_load_compute_lse, pipeline_load_compute_lse_consumer_state,
          pipeline_load_compute_sum_odo, pipeline_load_compute_sum_odo_consumer_state,
          pipeline_mma_compute_s, pipeline_mma_compute_s_consumer_state,
          pipeline_mma_compute_dp, pipeline_mma_compute_dp_consumer_state,
          pipeline_compute_mma_p, pipeline_compute_mma_p_producer_state,
          pipeline_compute_mma_ds, pipeline_compute_mma_ds_producer_state,
          pipeline_mma_compute_dkdv, pipeline_mma_compute_dkdv_consumer_state
      );

      cutlass::arch::NamedBarrier(
          kNumComputeWarps * NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier
      ).arrive_and_wait();

      if (warp_idx % kNumComputeWarps == 0) {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == WarpRole::Reduce) {
      warpgroup_reg_set<RegisterAllocation::kReduce>();

      reduce(
          blk_coord,
          problem_shape,
          iter_start,
          iter_count,
          params.mainloop,
          params.mainloop_params,
          shared_storage.tensors,
          pipeline_mma_reduce_dq, pipeline_mma_reduce_dq_consumer_state,
          pipeline_reduce_tma_store, pipeline_reduce_tma_store_producer_state
      );

      pipeline_reduce_tma_store.producer_tail(pipeline_reduce_tma_store_producer_state);
    }
    else {
      warpgroup_reg_set<RegisterAllocation::kEmpty>();

      /* no-op */

    }
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static dim3 get_grid_shape(Params const& params) {
    auto [Q, K, D, HB] = params.problem_shape;
    auto [H, B] = HB;
    dim3 grid(ceil_div(K, TileShapeK{}), H, B);
    return grid;
  }
};

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_common.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;

struct Sm100FmhaCtxKernelWarpspecializedSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    int wg_idx = warp_idx / 4;                        // warp_idx
    if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
    if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
    if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
    if (warp_idx == 12) return WarpRole::MMA;         //       12
    if (warp_idx == 13) return WarpRole::Load;        //       13
    if (warp_idx == 14) return WarpRole::Epilogue;    //       14
    return WarpRole::Empty;                           //       15
  }

  static const int NumWarpsSoftmax = 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  static const int NumRegsSoftmax = 192;
  static const int NumRegsCorrection = 96 - (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsOther = 32 + (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsEmpty = 24;
  
  static const int NumWarps = 16;
  
};


struct Sm100MlaFwdCtxKernelWarpspecializedSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    int wg_idx = warp_idx / 4;                        // warp_idx
    if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
    if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
    if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
    if (warp_idx == 12) return WarpRole::MMA;         //       12
    if (warp_idx == 13) return WarpRole::Load;        //       13
    if (warp_idx == 14) return WarpRole::Epilogue;    //       14
    return WarpRole::Empty;                           //       15
  }

  static const int NumWarpsSoftmax = 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  static const int NumRegsSoftmax = 184;
  static const int NumRegsCorrection = 96 - (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsOther = 48 + (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsEmpty = 24;

  static const int NumWarps = 16;

};

template<
  class ProblemShapeIn,
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler,
  class KernelSchedule = Sm100FmhaCtxKernelWarpspecializedSchedule
>
struct Sm100FmhaFwdKernelTmaWarpspecialized {

  using TileShape = typename CollectiveMainloop::TileShape;
  using ProblemShape = ProblemShapeIn;

  using WarpRole = typename KernelSchedule::WarpRole;

  constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    return KernelSchedule::warp_idx_to_WarpRole(warp_idx);
  }

  static const int NumWarpsSoftmax = KernelSchedule::NumWarpsSoftmax;
  static const int NumWarpsCorrection = KernelSchedule::NumWarpsCorrection;
  static const int NumWarpsEpilogue = KernelSchedule::NumWarpsEpilogue;
  static const int NumWarpsLoad = KernelSchedule::NumWarpsLoad;
  
  static_assert(NumWarpsEpilogue == CollectiveEpilogue::NumWarpsEpilogue);
  static_assert(NumWarpsLoad == CollectiveEpilogue::NumWarpsLoad);

  static const int NumRegsSoftmax = KernelSchedule::NumRegsSoftmax;
  static const int NumRegsCorrection = KernelSchedule::NumRegsCorrection;
  static const int NumRegsOther = KernelSchedule::NumRegsOther;
  static const int NumRegsEmpty = 24;

  static const int NumWarps = KernelSchedule::NumWarps;

  static constexpr bool IsMla = std::is_same_v<KernelSchedule, Sm100MlaFwdCtxKernelWarpspecializedSchedule>;

  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using TmemAllocator = cute::TMEM::Allocator1Sm;

  struct SharedStorage {
    using UnionType = union {
      typename CollectiveMainloop::TensorStorage mainloop;
      typename CollectiveEpilogue::TensorStorage epilogue;
    };

    using  StructType = struct {
      typename CollectiveMainloop::TensorStorage mainloop;
      typename CollectiveEpilogue::TensorStorage epilogue;
    };

    static constexpr bool IsPersistent = std::is_same_v<TileScheduler, PersistentTileScheduler> || std::is_same_v<TileScheduler, CausalPersistentTileScheduler>;
    using MainloopEpilogueStorage = std::conditional_t<IsPersistent, 
                                                       std::conditional_t<IsMla, 
                                                                          std::conditional_t<CollectiveMainloop::IsOrderLoadEpilogue, UnionType, StructType>,
                                                                          StructType>,
                                                       UnionType>;

    MainloopEpilogueStorage mainloop_epilogue; 

    struct PipelineStorage {
      alignas(16) typename CollectiveMainloop::PipelineQ::SharedStorage load_q;
      alignas(16) typename CollectiveMainloop::PipelineKV::SharedStorage load_kv;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s0;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s1;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s0_corr;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s1_corr;
      alignas(16) typename CollectiveMainloop::PipelineO::SharedStorage mma_corr;
      alignas(16) typename CollectiveMainloop::PipelineE::SharedStorage corr_epi;
      alignas(16) typename CollectiveMainloop::OrderBarrierSoftmax::SharedStorage order_s01;
    } pipelines;

    uint32_t tmem_base_ptr;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  using ArchTag = cutlass::arch::Sm100;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, ClusterShape{}, TileShape{})
    };
  }

  CUTLASS_DEVICE auto apply_batch(const Params &params, ProblemShape const& problem_shape, int batch_idx) {
    return apply_variable_length(params.problem_shape, batch_idx);
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem) {

    TileScheduler tile_scheduler{params.tile_scheduler};

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_WarpRole(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    if (role == WarpRole::Load && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    if (role == WarpRole::Epilogue && lane_predicate) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    auto get_epilogue_storage = [&]() {
      if constexpr (IsMla && CollectiveMainloop::IsOrderLoadEpilogue) {
        return reinterpret_cast<typename CollectiveEpilogue::TensorStorage *>(shared_storage.mainloop_epilogue.mainloop.smem_o.data());
      } else {
        return &shared_storage.mainloop_epilogue.epilogue;
      }
    };
    typename CollectiveEpilogue::TensorStorage & epilogue_storage = *get_epilogue_storage();


    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
    pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadQ;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(
      shared_storage.pipelines.load_q,
      pipeline_load_q_params,
      ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{});
    
    typename CollectiveMainloop::PipelineKV::Params pipeline_load_kv_params;
    if (role == WarpRole::Load) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Consumer;
    }
    pipeline_load_kv_params.is_leader = lane_predicate && (role == WarpRole::Load);
    pipeline_load_kv_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadK;
    typename CollectiveMainloop::PipelineKV pipeline_load_kv(
      shared_storage.pipelines.load_kv,
      pipeline_load_kv_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s0_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax0) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s0_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s0(
      shared_storage.pipelines.mma_s0,
      pipeline_mma_s0_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s1_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax1) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s1_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s1(
      shared_storage.pipelines.mma_s1,
      pipeline_mma_s1_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
    if (role == WarpRole::Softmax0) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s0_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s0_corr(
      shared_storage.pipelines.s0_corr,
      pipeline_s0_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s1_corr_params;
    if (role == WarpRole::Softmax1) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s1_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s1_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s1_corr(
      shared_storage.pipelines.s1_corr,
      pipeline_s1_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineO pipeline_mma_corr(
      shared_storage.pipelines.mma_corr,
      pipeline_mma_corr_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineE::Params pipeline_corr_epi_params;
    if (role == WarpRole::Correction) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Epilogue) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Consumer;
    }
    pipeline_corr_epi_params.producer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.consumer_arv_count = NumWarpsEpilogue * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineE pipeline_corr_epi(
      shared_storage.pipelines.corr_epi,
      pipeline_corr_epi_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::OrderBarrierSoftmax::Params params_order_s01;
    params_order_s01.group_id = role == WarpRole::Softmax1 ? 1 : 0;
    params_order_s01.group_size = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::OrderBarrierSoftmax order_s01(
      shared_storage.pipelines.order_s01, params_order_s01);

    TmemAllocator tmem_allocator;

    __syncthreads();

    pipeline_load_q.init_masks(ClusterShape{});
    pipeline_load_kv.init_masks(ClusterShape{});
    pipeline_mma_s0.init_masks(ClusterShape{});
    pipeline_mma_s1.init_masks(ClusterShape{});
    pipeline_mma_corr.init_masks(ClusterShape{});

    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_consumer_state;
    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineKV>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_consumer_state;
    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineE>();

    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_consumer_state;
    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineO>();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue{params.epilogue};

    if (role == WarpRole::Softmax0 || role == WarpRole::Softmax1) {
      warpgroup_reg_set<NumRegsSoftmax>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        bool is_softmax_0 = role == WarpRole::Softmax0;

        mainloop.softmax(
           is_softmax_0 ? 0 : 1, blk_coord,
           params.mainloop, logical_problem_shape,
           is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
           is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
           is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
           is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
           order_s01
         );

       }
    }
    else if (role == WarpRole::Correction) {
      cutlass::arch::warpgroup_reg_dealloc<NumRegsCorrection>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          mainloop.correction_empty(
            blk_coord,
            params.mainloop, logical_problem_shape,
            params.problem_shape,
            epilogue_storage,
            pipeline_corr_epi, pipeline_corr_epi_producer_state,
            epilogue
          );
          continue;
        }

        mainloop.correction(
          blk_coord,
          params.mainloop, logical_problem_shape,
          params.problem_shape,
          epilogue_storage,
          pipeline_s0_corr, pipeline_s0_corr_consumer_state,
          pipeline_s1_corr, pipeline_s1_corr_consumer_state,
          pipeline_mma_corr, pipeline_mma_corr_consumer_state,
          pipeline_corr_epi, pipeline_corr_epi_producer_state,
          epilogue
        );

      }

      if constexpr (NumWarpsEpilogue == 0) {
        static_assert(NumWarpsCorrection == 1);

        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == WarpRole::MMA) {
      warpgroup_reg_set<NumRegsOther>();

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        mainloop.mma(
          blk_coord,
          params.mainloop, logical_problem_shape,
          shared_storage.mainloop_epilogue.mainloop,
          pipeline_load_q, pipeline_load_q_consumer_state,
          pipeline_load_kv, pipeline_load_kv_consumer_state,
          pipeline_mma_s0, pipeline_mma_s0_producer_state,
          pipeline_mma_s1, pipeline_mma_s1_producer_state,
          pipeline_mma_corr, pipeline_mma_corr_producer_state
        );

      }
    }
    else if (role == WarpRole::Load) {
      warpgroup_reg_set<NumRegsOther>();

      if constexpr (IsMla && CollectiveMainloop::IsOrderLoadEpilogue) {
        cutlass::arch::NamedBarrier::arrive((NumWarpsLoad + NumWarpsEpilogue) * NumThreadsPerWarp, 
                                      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        mainloop.load(
          blk_coord, logical_problem_shape,
          params.mainloop, params.problem_shape,
          shared_storage.mainloop_epilogue.mainloop,
          pipeline_load_q, pipeline_load_q_producer_state,
          pipeline_load_kv, pipeline_load_kv_producer_state
        );

      }
    }
    else if (role == WarpRole::Epilogue) {
      warpgroup_reg_set<NumRegsOther>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        epilogue.store(
          blk_coord, logical_problem_shape,
          params.epilogue, params.problem_shape,
          epilogue_storage,
          pipeline_corr_epi, pipeline_corr_epi_consumer_state
        );

      }

      static_assert(NumWarpsEpilogue <= 1);
      if constexpr (NumWarpsEpilogue == 1) {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == WarpRole::Empty) {
      warpgroup_reg_set<NumRegsEmpty>();

      /* no-op, donate regs and exit */
    }
  }

};

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/reference/fmha_fwd_reference.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/tensor.hpp"
#include "collective/fmha_fusion.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShapeIn,
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Mask
>
void __global__ fmha_reference_kernel(
    ProblemShapeIn problem_shape_in,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Mask mask) {

  using namespace cute;
  using namespace cutlass::fmha::collective;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;

  extern __shared__ char mS_mem[];
  ElementAccumulator* mS = reinterpret_cast<ElementAccumulator*>(mS_mem);

  ElementAccumulator softmax_scale = static_cast<ElementAccumulator>(1.0 / sqrt(1.0 * size<1>(mQ)));

  auto id = make_identity_tensor(make_shape(1, 1));
  for (int idx_L = blockIdx.y; idx_L < size<3>(problem_shape_in); idx_L += gridDim.y) {
    for (int idx_Q = blockIdx.x; idx_Q < size<0>(problem_shape_in); idx_Q += gridDim.x) {
      
      auto coord_L = idx2crd(idx_L, shape<3>(problem_shape_in));
      auto get_coord_in = [&]() {
        if constexpr (rank_v<decltype(get<2>(ProblemShapeIn{}))> == 2) {
          return cute::make_tuple(idx_Q, _0{}, cute::make_tuple(_0{}, _0{}), coord_L);
        } else {
          return cute::make_tuple(idx_Q, _0{}, _0{}, coord_L);
        }
      };
      auto coord_in = get_coord_in();
      auto [problem_shape, coord] = apply_variable_length(problem_shape_in, coord_in, get<3,1>(coord_in));

      int head_qk = 0;
      int head_v = 0;
      if constexpr (rank_v<decltype(get<2>(problem_shape))> == 2) {
        // MLA case: head_qk 192, head_v = 128
        head_qk = size<2, 0>(problem_shape) + size<2, 1>(problem_shape);
        head_v = size<2, 0>(problem_shape);
      } else {
        head_qk = size<2>(problem_shape);
        head_v = head_qk;
      }

      if (get<0,0>(coord) >= get<0>(problem_shape)) continue;

      int offset_Q = 0;
      if constexpr (rank<0>(decltype(coord){}) == 2) {
        offset_Q = get<0,1>(coord);
      }
  
      int offset_K = 0;
      if constexpr (rank<1>(decltype(coord){}) == 2) {
        offset_K = get<1,1>(coord);
      }

      if (get<1>(problem_shape) == 0) {
        for (int idx_D = threadIdx.x; idx_D < head_qk; idx_D += blockDim.x) {
          mO(idx_Q + offset_Q, idx_D, idx_L) = Element(0);
        }

        if (threadIdx.x == 0 && mLSE.data() != nullptr) {
          mLSE(idx_Q + offset_Q, idx_L) = -INFINITY;
        }
        continue;
      }
  
      for (int idx_K = threadIdx.x; idx_K < size<1>(problem_shape); idx_K += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_D = 0; idx_D < head_qk; idx_D++) {
          ElementAccumulator eQ = mQ(idx_Q + offset_Q, idx_D, idx_L);
          ElementAccumulator eK = mK(idx_K + offset_K, idx_D, idx_L);
          acc += eQ * eK;
        }
        auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
        frag(0) = acc;
        mask.apply_mask(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
        mS[idx_K] = frag(0);
      }

      __syncthreads();

      ElementAccumulator maxS = -std::numeric_limits<ElementAccumulator>::infinity();
      for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
        maxS = std::max<ElementAccumulator>(maxS, mS[idx_K]);
      }
      if (maxS == -std::numeric_limits<ElementAccumulator>::infinity()) maxS = 0;

      __syncthreads();

      for (int idx_K = threadIdx.x; idx_K < size<1>(problem_shape); idx_K += blockDim.x) {
        mS[idx_K] = expf(softmax_scale * (mS[idx_K] - maxS));
      }

      __syncthreads();

      ElementAccumulator sum = 0;
      for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
        sum += mS[idx_K];
      }

      ElementAccumulator scale = 1.0f / sum;


      for (int idx_D = threadIdx.x; idx_D < head_v; idx_D += blockDim.x) {
        ElementAccumulator acc = 0;
        for (int idx_K = 0; idx_K < size<1>(problem_shape); idx_K++) {
          ElementAccumulator eV = mV(idx_K + offset_K, idx_D, idx_L);
          ElementAccumulator eK = static_cast<Element>(mS[idx_K]);
          acc += eK * eV;
        }
        mO(idx_Q + offset_Q, idx_D, idx_L) = static_cast<typename TensorO::value_type>(acc * scale);
      }

      if (threadIdx.x == 0 && mLSE.data() != nullptr) {
        mLSE(idx_Q + offset_Q, idx_L) = log(sum) + softmax_scale * maxS;
      }

    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ProblemShapeIn,
  class TensorQ,
  class TensorK,
  class TensorV,
  class TensorO,
  class TensorLSE,
  class Mask
>
void fmha_reference(
    ProblemShapeIn problem_shape_in,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE,
    Mask mask) {

  using namespace cute;

  dim3 grid(size<0>(mO), size<2>(mO), 1);
  dim3 block(256);
  int shared_mem = size<0>(mK) * int(sizeof(typename TensorLSE::value_type));
  fmha_reference_kernel<<<grid, block, shared_mem>>>(problem_shape_in, mQ, mK, mV, mO, mLSE, mask);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
