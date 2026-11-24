"""KernelWiki artifact PR-2466 (NVIDIA/cutlass)

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
  bool causal_q_begin = true;
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
    std::string causal_type;
    cmd.get_cmd_line_argument<std::string>("causal-type", causal_type, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
      if (varlen) {
        residual = true;
      }
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
      if(causal_type == "qend") {
        causal_q_begin = false;
      } else {
        causal_q_begin = true;
      }
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
      << "  --causal-type=<qbegin|qend> Causal mask type\n"
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
    
    auto [Q, K, D, HB] = problem_shape;

    auto problem_shape_ref = cute::make_tuple(Q, K, D, D, HB);

    fmha_reference(problem_shape_ref, mQ, mK, mV, mO, mLSE, ActiveMask{});

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
      if(options.causal_q_begin) {
        fn(CausalMask{});
      } else {
        fn(CausalMask<false>{});
      }
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
    'key-files/examples/77_blackwell_fmha/77_blackwell_fmha_bwd.cu': r"""/***************************************************************************************************
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
    \brief Example implementation of fused multi-head attention for Blackwell using CUTLASS 3.

    This example showcases the use of CUTLASS to build backward fused
    multi-head attention (FMHA) collectives from existing CUTLASS collectives targeting
    the NVIDIA Blackwell architecture.

    Background and motivation
    -------------------------
    CUTLASS is a highly flexible library that provides open-source building blocks
    for tensor core programming for GEMM or GEMM-like problems. Fused multi-head
    attention (FMHA) is a foundational kernel for large language models (LLMs) since it
    makes long sequence lengths feasible from a memory-usage perspective. It also
    improves computational efficiency since it transforms an outer-product-like and
    a matrix-vector-like GEMM into a fused operation with much higher arithmetic
    intensity. For more details, see Dao et al, 2022; Dao, 2023.
    Implementing this kernel in CUTLASS enabled easy customization and high
    performance.

    Introduction
    ------------
    The example targets the NVIDIA Blackwell architecture, and takes advantage of
    5th gen tensor cores and the Tensor Memory Accelerator (TMA), just like
    GEMMs do. It provides a backward pass (often abbreviated
    bwd in the code).
    The code is structured into three layers: The runner (and the reference kernels)
    takes care of initialization, measurement, and testing; the device layer
    orchestrates kernel calls and partitions workspace; and the kernel layer (just
    like the CUTLASS kernel layer.

    Support
    ---------

    We support fp16 and fp8 data types with a head dimension of 128.

    Example usage:
    $ ./examples/77_blackwell_fmha/77_blackwell_fmha_bwd_fp16 \
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
#include "reference/fmha_bwd_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "collective/fmha_fusion.hpp"
#include "device/fmha_device_bwd.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kZero, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 16;
  int h = 16;
  int h_k = 1;
  int q = 1024;
  int k = 1024;
  std::vector<int> varlen_q;
  std::vector<int> varlen_k;
  int d = 128;
  int d_vo = 128;
  int iterations = 3;
  bool verify = false;
  bool verbose = false;

  bool causal = false;
  bool residual = false;
  bool varlen = false;
  int sm_count = 0;

  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_k = InitStyle::kRandom;
  InitStyle init_style_v = InitStyle::kRandom;
  InitStyle init_style_do = InitStyle::kRandom;
  bool skip_reference = false;

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
      else if (s == "0") {
        dst = InitStyle::kZero;
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
    cmd.get_cmd_line_argument("d_vo", d_vo, d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

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

    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    if (mask == "causal") {
      causal = true;
    }
    else if (mask == "residual") {
      residual = true;
    }
    else {
      causal = defaults.causal;
    }
    if (varlen) {
      residual = true;
    }

    skip_reference = cmd.check_cmd_line_flag("skip-reference");
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);

    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_k, defaults.init_style_k);
    get_init_style_argument(cmd, "init-style", init_style_v, defaults.init_style_v);
    get_init_style_argument(cmd, "init-style", init_style_do, defaults.init_style_do);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-k", init_style_k, init_style_k);
    get_init_style_argument(cmd, "init-style-v", init_style_v, init_style_v);
    get_init_style_argument(cmd, "init-style-do", init_style_v, init_style_do);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_fmha_bwd\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention kernels for the backward pass targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --varlen-q=<int>:<int...>   Sets the variable Q extent per batch (colon separated)\n"
      << "  --varlen-k=<int>:<int...>   Sets the variable K extent per batch (colon separated)\n"
      << "  --d=<int>                   Sets the D extent\n"
      << "  --d_vo=<int>                  Sets the D_VO extent\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
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
    case InitStyle::kZero: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 0, (Element) 0);
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
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  bool kIsVarlen,
  bool kIsMla,
  class TileShape,
  class DispatchPolicy,
  class ActiveMask,
  class... KernelOptions
>
struct BwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#else
  using Element = cutlass::half_t;
#endif
  using ElementAccumulator = float;

  // Q K D (H B)
  using ProblemShape = std::conditional_t<
      kIsVarlen,
      cute::tuple<VariableLength, VariableLength, int, int, cute::tuple<int, int>>,
      cute::tuple<int, int, int, int, cute::tuple<int, int>>
  >;
  
  using TensorStride = Stride<int, _1, Stride<int, int>>; // Seq D (H B)
  using StrideQ = TensorStride;
  using StrideK = TensorStride;
  using StrideV = TensorStride;
  using StrideO = TensorStride;
  using StrideLSE = Stride<_1, Stride<int, int>>; // Seq (H B)

  // Backwards specific
  using StrideDQ = TensorStride;
  using StrideDK = TensorStride;
  using StrideDV = TensorStride;
  using StrideDO = TensorStride;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;

  StrideDQ stride_dQ;
  StrideDK stride_dK;
  StrideDV stride_dV;
  StrideDO stride_dO;

  uint64_t seed = 0;

  DeviceAllocation<Element> block_Q;
  DeviceAllocation<Element> block_K;
  DeviceAllocation<Element> block_V;
  DeviceAllocation<Element> block_O;
  DeviceAllocation<ElementAccumulator> block_LSE;

  DeviceAllocation<int> block_cumulative_seqlen_q;
  DeviceAllocation<int> block_cumulative_seqlen_kv;

  DeviceAllocation<Element> block_dQ;
  DeviceAllocation<Element> block_dK;
  DeviceAllocation<Element> block_dV;
  DeviceAllocation<Element> block_dO;

  DeviceAllocation<Element> block_ref_dQ;
  DeviceAllocation<Element> block_ref_dK;
  DeviceAllocation<Element> block_ref_dV;

  //
  // Methods
  //
  bool verify(const ProblemShape& problem_shape) {
    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [H, B] = HB;

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      select<0,2,4>(problem_shape),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      select<1,2,4>(problem_shape),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      select<1,3,4>(problem_shape),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()),
      select<0,3,4>(problem_shape),
      stride_O);

    // keep going here! (this might be better in cursor)

    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()),
      select<0,4>(problem_shape),
      stride_LSE);

    Tensor mDQ = make_tensor(make_gmem_ptr(block_ref_dQ.get()),
      select<0,2,4>(problem_shape),
      stride_dQ);

    Tensor mDK = make_tensor(make_gmem_ptr(block_ref_dK.get()),
      select<1,2,4>(problem_shape),
      stride_dK);

    Tensor mDV = make_tensor(make_gmem_ptr(block_ref_dV.get()),
      select<1,3,4>(problem_shape),
      stride_dV);

    Tensor mDO = make_tensor(make_gmem_ptr(block_dO.get()),
      select<0,3,4>(problem_shape),
      stride_dO);

    fmha_bwd_reference(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDQ, mDK, mDV, ActiveMask{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-0 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(block_dQ, block_ref_dQ, max_diff, mean_diff);

    bool passed_dQ = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dQ) {
      std::cerr << "failed dQ: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dK, block_ref_dK, max_diff, mean_diff);

    bool passed_dK = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dK) {
      std::cerr << "failed dK: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dV, block_ref_dV, max_diff, mean_diff);

    bool passed_dV = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dV) {
      std::cerr << "failed dV: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_dQ && passed_dK && passed_dV;
  }

  auto initialize_problem_shape(Options const& options) {
    if constexpr (kIsVarlen) {
      int num_batches = options.b;

      // generate Q as --b times
      //    gaussian (--Q, --Q / 2) sampled positive
      //    track cumulative 
      std::mt19937 rng(0x202305151552ull);
      std::normal_distribution<double> dist_q(options.q, options.q / 2);
      std::normal_distribution<double> dist_kv(options.k, options.k / 2);

      auto generate_positive_int = [](auto& dist, auto& gen) {
        // "0" is a valid value we test here
        return std::max(0, static_cast<int>(dist(gen)));
      };

      std::vector<int> cumulative_seqlen_q = {0};
      std::vector<int> cumulative_seqlen_kv = {0};

      int total_seqlen_q = 0;
      int total_seqlen_kv = 0;
      int max_seqlen_q = 0;
      int max_seqlen_kv = 0;

      const bool kVarlenSame = false;
      for (int i = 0; i < num_batches; i++) {
        int seqlen_q = (! options.varlen_q.empty()) ? options.varlen_q.at(i) : 
                kVarlenSame ? options.q :
                generate_positive_int(dist_q, rng);
        int seqlen_kv = (! options.varlen_k.empty()) ? options.varlen_k.at(i) :
                kVarlenSame ? options.k :
                generate_positive_int(dist_kv, rng);

        total_seqlen_q += seqlen_q;
        total_seqlen_kv += seqlen_kv;

        max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
        max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

        cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
        cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
      }

      block_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      block_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      block_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      block_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());

      ProblemShape problem_shape{
          {max_seqlen_q, block_cumulative_seqlen_q.get(), total_seqlen_q},
          {max_seqlen_kv, block_cumulative_seqlen_kv.get(), total_seqlen_kv},
          options.d, options.d_vo, {options.h, options.b}
      };
      auto tensor_shape = make_shape(total_seqlen_q, total_seqlen_kv, options.d, options.d_vo, make_shape(options.h, 1));

      return cute::make_tuple(problem_shape, tensor_shape);
    }
    else {
      ProblemShape problem_shape{options.q, options.k, options.d, options.d_vo, {options.h, options.b}};
      return cute::make_tuple(problem_shape, problem_shape);
    }
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShape initialize(Options const& options) {
    auto [problem_shape, tensor_shape] = initialize_problem_shape(options);
    auto [Q, K, D, D_VO, HB] = tensor_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment

    // for varlen, Q == total_Q, K == total_K, B = 1
    // but in problem_shape, they've got to be max_Q/max_K, and B = B

    auto shape_Q = make_shape(Q, D, make_shape(H, B));
    auto shape_O = make_shape(Q, D_VO, make_shape(H, B));
    auto shape_K = make_shape(K, D, make_shape(H, B));
    auto shape_V = make_shape(K, D_VO, make_shape(H, B));
    auto shape_LSE = make_shape(Q, make_shape(H, B));

    stride_Q = make_stride(D, _1{}, make_stride(D*Q, B == 1 ? 0 : D*Q*H));
    stride_K = make_stride(D, _1{}, make_stride(D*K, B == 1 ? 0 : D*K*H));
    stride_V = make_stride(D_VO, _1{}, make_stride(D_VO*K, B == 1 ? 0 : D_VO*K*H));
    stride_O = make_stride(D_VO, _1{}, make_stride(D_VO*Q, B == 1 ? 0 : D_VO*Q*H));
    stride_LSE = make_stride(_1{}, make_stride(Q, B == 1 ? 0 : Q*H));

    stride_dQ = stride_Q;
    stride_dK = stride_K;
    stride_dV = stride_V;
    stride_dO = stride_O;

    auto lsize = [](auto shape) {
      return size(make_shape(1ull, shape));
    };

    block_Q.reset(lsize(shape_Q));
    block_K.reset(lsize(shape_K));
    block_V.reset(lsize(shape_V));
    block_O.reset(lsize(shape_O));
    block_LSE.reset(lsize(shape_LSE));

    block_dQ.reset(lsize(shape_Q));
    block_dK.reset(lsize(shape_K));
    block_dV.reset(lsize(shape_V));
    block_dO.reset(lsize(shape_O));

    block_ref_dQ.reset(lsize(shape_Q));
    block_ref_dK.reset(lsize(shape_K));
    block_ref_dV.reset(lsize(shape_V));

    initialize_block(block_Q, seed + 2023, options.init_style_q);
    initialize_block(block_K, seed + 2022, options.init_style_k);
    initialize_block(block_V, seed + 2021, options.init_style_v);
    initialize_block(block_dO, seed + 2020, options.init_style_do);

    initialize_block(block_dQ, seed + 2030, InitStyle::kOne);
    initialize_block(block_dK, seed + 2031, InitStyle::kOne);
    initialize_block(block_dV, seed + 2032, InitStyle::kOne);
    initialize_block(block_ref_dQ, seed + 2033);
    initialize_block(block_ref_dK, seed + 2034);
    initialize_block(block_ref_dV, seed + 2035);

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      select<0,2,4>(problem_shape),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      select<1,2,4>(problem_shape),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      select<1,3,4>(problem_shape),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()),
      select<0,3,4>(problem_shape),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()),
      select<0,4>(problem_shape),
      stride_LSE);

    if (! options.skip_reference) {
      fmha_reference(problem_shape, mQ, mK, mV, mO, mLSE, ActiveMask{});
    }

    return problem_shape;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    auto problem_shape = initialize(options);

    ElementAccumulator softmax_scale = 1.0f / sqrtf(options.d);

    ExampleResult example_result;

    using Operation = cutlass::fmha::device::Sm100FmhaBwd<ProblemShape, Element, ElementAccumulator, TileShape, kIsMla, ActiveMask>;

    typename Operation::Arguments arguments{
      problem_shape,
      block_Q.get(), stride_Q,
      block_K.get(), stride_K,
      block_V.get(), stride_V,
      block_O.get(), stride_O,
      block_LSE.get(), stride_LSE,
      block_dO.get(), stride_dO,
      block_dQ.get(), stride_dQ,
      block_dK.get(), stride_dK,
      block_dV.get(), stride_dV,
      softmax_scale,
      hw_info
    };

    Operation op;

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
    status = op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
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

    double flops = 2.0 * (std::is_same_v<ActiveMask, CausalForBackwardMask<false>> || std::is_same_v<ActiveMask, CausalForBackwardMask<true>> ? 0.5 : 1.0);
    flops *= static_cast<double>(get<0>(problem_shape));
    flops *= static_cast<double>(get<1>(problem_shape));
    flops *= (3 * static_cast<double>(get<2>(problem_shape)) + 2 * static_cast<double>(get<3>(problem_shape)));
    flops *= static_cast<double>(get<4,0>(problem_shape));
    flops *= static_cast<double>(get<4,1>(problem_shape));
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
      passed = verify(problem_shape);
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

struct KernelCoop {};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Fn>
auto dispatch_bool(bool value, Fn fn) {
  if (value) {
    return fn(std::true_type{});
  }
  else {
    return fn(std::false_type{});
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_bwd_64(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, false,decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _64;

  run(Shape<_128, _128, HeadDim, HeadDim>{}, KernelCoop{}, "tma");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_bwd_128(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, false, decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _128;

  run(Shape<_128, _128, HeadDim, HeadDim>{}, KernelCoop{}, "tma");
}

template<class Mask>
void run_bwd_mla_192(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, true, decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _192;

  run(Shape<_64, _128, HeadDim, _128>{}, KernelCoop{}, "tma");
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
      << "(compute capability 100a) and CUDA 12.8 or greater.\n";
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

  std::cout << "###### B " << options.b << " H " << options.h << " Q " << options.q << " K " << options.k << " D " << options.d << " D_VO " << options.d_vo << " ";
  std::cout << "Backward" << " " << (options.causal ? "Causal" : "Full") << " ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  auto with_causal = [&](auto fn) {
    if (options.causal) {
      fn(CausalForBackwardMask{});
    }
    else if (options.residual) {
      fn(ResidualMaskForBackward{});
    }
    else {
      fn(NoMask{});
    }
  };

  with_causal([&](auto fusion) {
    if (options.d <= 64 && options.d_vo == options.d) {
      run_bwd_64(fusion, options, hw_info);
    }
    else if (options.d <= 128 && options.d_vo == options.d) {
      run_bwd_128(fusion, options, hw_info);
    }
    else if (options.d == 192 && options.d_vo == 128) {
      run_bwd_mla_192(fusion, options, hw_info);
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
    'key-files/examples/77_blackwell_fmha/77_blackwell_mla.cu': r"""/***************************************************************************************************
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
/*! \file A MLA (Multi-Head Latent Attention) inference kernel sample for the
          NVIDIA Blackwell Architecture.
*/

#include <iostream>
#include <random>
#include <regex>
#include <cmath>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_mla_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kLinearStride128, kLinearStride1, kRandom, kRandomLarge, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int k = 256;
  int split_kv = -1; // number of split along k dim.
  bool is_var_split_kv = false;
  int max_split_kv = 16;
  int page = -1;
  float spread = 0.2f;
  int iterations = 3;
  bool verify = false;
  bool verbose = false;
  bool is_fused_reduction = false;

  int sm_count = 0;

  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_c = InitStyle::kRandom;

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
      else if (s == "l") {
        dst = InitStyle::kRandomLarge;
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

    cmd.get_cmd_line_argument("k", k, -1);
    if (k == -1) k = defaults.k;

    cmd.get_cmd_line_argument("b", b, -1);
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("split_kv", split_kv, defaults.split_kv);
    if (split_kv == 0) {
      split_kv = 1;
    }
    cmd.get_cmd_line_argument("page", page, defaults.page);
    cmd.get_cmd_line_argument("spread", spread, defaults.spread);
    is_var_split_kv = cmd.check_cmd_line_flag("var_split_kv");
    if (page == -1) {
      is_var_split_kv = false;
    }
    cmd.get_cmd_line_argument("max_split_kv", max_split_kv, defaults.max_split_kv);
    if (is_var_split_kv == true) {
      split_kv = max_split_kv;
    }
    is_fused_reduction = cmd.check_cmd_line_flag("fuse_reduction");
    if (split_kv == 1) {
      is_fused_reduction = false;
    }
    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);
    
    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_c, defaults.init_style_c);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-c", init_style_c, init_style_c);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_mla\n\n"
      << "  This example showcases the use of CUTLASS for fused multi-head latent\n"
      << "  attention kernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --page=<int>                Enables paging and sets the page size\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --spread=<float>            Relative spread away from K for paging\n"
      << "  --split_kv=<int>            Split KV factor\n"
      << "  --fused_reduction           Fuse the reduction operation\n"
      << "  --var_split_kv              Use varying split KV factor\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << " --sm-count                   Sets SM count rather than querying it\n"
      << " --kernel-filter=<filter>     Sets regexp to match kernel against\n"
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
        block.get(), block.size(), seed, (Element) -1, (Element) 1);
      break;
    }
    case InitStyle::kRandomLarge: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) -1, (Element) 100);
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
      for (size_t i = 0; i < block.size() / 64; i ++) {
        for (int j = 0; j < 64; j++) {
          data[j + 64*i] = static_cast<Element>((double) (i % 9));
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
  double tbytes_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<bool v>
struct IsPersistent {
  static const bool value = v;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class TileShape,
  class PersistenceOption = IsPersistent<true>
>
struct Runner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#elif FP16
  using Element = cutlass::half_t;
#else
  #error "Must either define FP8 or FP16"
#endif

  using ElementAcc = float;
  using ElementOut = cutlass::half_t;

  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;
  
  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler = std::conditional_t<
      PersistenceOption::value,
      Sm100MlaPersistentTileScheduler,
      Sm100MlaIndividualTileScheduler
  >;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
    TileShape, Element, ElementAcc, ElementOut, ElementAcc, TileScheduler
  >;
  using Operation = cutlass::fmha::device::MLA<Kernel>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q_latent;
  StrideK stride_C_latent;
  StrideQ stride_Q_rope;
  StrideK stride_K_rope;
  StrideO stride_O;
  StrideLSE stride_LSE;
  StrideLSE stride_PT;
  
  uint64_t seed = 0;

  int page_size = -1;
  int page_count = -1;

  // We allocate Q and C as first latent, then rope
  // This means that we offset the pointer by HeadDim_latent to get the rope
  // portion
  DeviceAllocation<Element> block_Q;
  DeviceAllocation<Element> block_C;
  DeviceAllocation<ElementOut> block_O;
  DeviceAllocation<int> block_seq;
  DeviceAllocation<int> block_PT;
  DeviceAllocation<int> block_split_kv;
  DeviceAllocation<int> block_accum_split_len; 
  DeviceAllocation<ElementAcc> block_LSE;
  DeviceAllocation<ElementOut> block_ref_O;
  DeviceAllocation<ElementAcc> block_ref_LSE;
   
  ElementAcc scale;

  //
  // Methods
  //

  bool verify(const ProblemShape& problem_shape) {
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    int page_K = K;
    int page_B = B;
    if (block_PT.get() != nullptr) {
      page_K = page_size;
      page_B = page_count;
    }

    Tensor mQ_latent = make_tensor(make_gmem_ptr(block_Q.get()),
      cute::make_tuple(H, D_latent, B),
      stride_Q_latent);

    Tensor mQ_rope = make_tensor(make_gmem_ptr(block_Q.get() + D_latent),
      cute::make_tuple(H, D_rope, B),
      stride_Q_rope);

    Tensor mC_latent = make_tensor(make_gmem_ptr(block_C.get()),
      cute::make_tuple(page_K, D_latent, page_B),
      stride_C_latent);

    Tensor mK_rope = make_tensor(make_gmem_ptr(block_C.get() + D_latent),
      cute::make_tuple(page_K, D_rope, page_B),
      stride_K_rope);

    Tensor mO = make_tensor(make_gmem_ptr(block_ref_O.get()),
      cute::make_tuple(H, D_latent, B),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(block_ref_LSE.get()),
      cute::make_tuple(H, B),
      stride_LSE);

    Tensor mSeq = make_tensor(make_gmem_ptr(static_cast<int*>(block_seq.get())), make_shape(B));
    Tensor mPT = make_tensor(make_gmem_ptr(static_cast<int*>(block_PT.get())), make_shape(ceil_div(K, page_size), B), stride_PT);

    fmha_mla_reference(problem_shape, mSeq, mPT, mQ_latent, mQ_rope, mC_latent, mK_rope, mO, mLSE, scale);

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
    reference_abs_diff(block_O, block_ref_O, max_diff, mean_diff);

    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    bool passed_LSE = true;
    reference_abs_diff(block_LSE, block_ref_LSE, max_diff, mean_diff);

    passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_LSE;
  }

  ProblemShape initialize(const Options& options) {
    auto problem_shape = cute::make_tuple(TileShapeH{}, options.k, TileShapeD{}, options.b);

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    // the scale is based on the non-absorbed sizes, change as appropriate
    // we can't determine this parameter from the info we have, it's an input
    int D_non_latent = 128;
    scale = static_cast<decltype(scale)>(1.0 / sqrt(1.0 * (D_non_latent + D_rope)));
    // Shape (H, D, B)
    stride_Q_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(H * (0 + D_latent + D_rope)));
    stride_Q_rope = stride_Q_latent;
    stride_O = cute::make_tuple(static_cast<int64_t>(0 + D_latent), _1{}, static_cast<int64_t>(0 + H * D_latent));
    stride_LSE = cute::make_tuple(_1{}, 0 + H);

    block_Q.reset(static_cast<size_t>(options.b) * H * (D_latent + D_rope));
    block_O.reset(static_cast<size_t>(options.b) * H * D_latent);
    block_LSE.reset(static_cast<size_t>(options.b) * H);
    block_ref_O.reset(static_cast<size_t>(options.b) * H * D_latent);
    block_ref_LSE.reset(static_cast<size_t>(options.b) * H);

    if (options.page == -1) {

      stride_C_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(options.k) * (D_latent + D_rope));
      stride_K_rope = stride_C_latent;

      block_C.reset(static_cast<size_t>(options.b) * options.k * (D_latent + D_rope));

    }
    else {
      
      float spread = options.spread;
      int max_K = static_cast<int>((1 + spread) * K);
      int min_K = static_cast<int>((1 - spread) * K);
      page_size = options.page;
      page_count = B * ceil_div(max_K, page_size);
      stride_PT = cute::make_stride(_1{}, page_count);

      std::vector<int> host_seq(B);
      std::vector<int> host_PT(page_count * B);

      for (int i = 0; i < B; i++) {
        int seq = min_K + rand() % (max_K - min_K + 1);
        host_seq[i] = seq;
        for (int j = 0; j < ceil_div(seq, page_size); j++) {
          host_PT[page_count * i + j] = i + j * B;
        }
      }

      block_seq.reset(host_seq.size());
      block_seq.copy_from_host(host_seq.data(), host_seq.size());
      block_PT.reset(host_PT.size());
      block_PT.copy_from_host(host_PT.data(), host_PT.size());

      get<1>(problem_shape) = max_K;

      stride_C_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, page_size * static_cast<int64_t>((D_latent + D_rope)));
      stride_K_rope = stride_C_latent;

      block_C.reset(page_count * page_size * static_cast<int64_t>((D_latent + D_rope)));

      if (options.is_var_split_kv == true) {
        std::vector<int> host_split_kv(B);
        for(int i = 0; i < B; ++i) {
          auto len = host_seq[i];
	  int split = ceil_div(options.max_split_kv, ceil_div(max_K, len));
	  host_split_kv[i] = split;
        }
	block_split_kv.reset(B);
        block_split_kv.copy_from_host(host_split_kv.data(), host_split_kv.size());
      } 
    }

    initialize_block(block_Q, seed + 2023, options.init_style_q);
    initialize_block(block_C, seed + 2022, options.init_style_c);

    return problem_shape;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {

    ProblemShape problem_shape = initialize(options);

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    typename Operation::Arguments arguments{
      problem_shape,
      { scale,
        block_Q.get(), stride_Q_latent,
        block_Q.get() + D_latent, stride_Q_rope,
        block_C.get(), stride_C_latent,
        block_C.get() + D_latent, stride_K_rope,
        block_seq.get(),
        block_PT.get(), stride_PT,
        page_count, page_size},
      { block_O.get(), 
        stride_O,
        block_LSE.get(),
        stride_LSE}, 
      hw_info,
      options.split_kv,
      options.is_var_split_kv ? block_split_kv.get() : nullptr,
      options.is_fused_reduction
    };
    if (options.split_kv < 0 && !options.is_var_split_kv) {
      Operation::set_split_kv(arguments);
    }

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
    status = op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
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

    double flops = 1.0;
    flops *= B;
    flops *= K;
    flops *= H;
    flops *= 2.0;
    flops *= (2.0 * D_latent + D_rope);

    double bytes_q = sizeof(Element);
    bytes_q *= B;
    bytes_q *= H;
    bytes_q *= (D_latent + D_rope);
    double bytes_c = sizeof(Element);
    bytes_c *= B;
    bytes_c *= options.k;  // K may be max_K here
    bytes_c *= (D_latent + D_rope);
    double bytes_o = sizeof(ElementOut);
    bytes_o *= B;
    bytes_o *= H;
    bytes_o *= D_latent;
    double bytes = bytes_q + bytes_c + bytes_o;

    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    double tbytes_s = bytes * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
    example_result.tbytes_s = tbytes_s;
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
      passed = verify(problem_shape);
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
  std::cout << " : " << result.tflops_tc_s << " TFLOPS/s " << result.tbytes_s << " TB/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms * 1e3 << " us, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_mla(Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    Runner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options, hw_info);
    print_result(name, result, options.verbose);
  };

  using NumHeads = _128;
  using HeadDimLatent = _512;
  using HeadDim = Shape<HeadDimLatent, _64>;

  std::cout << "###### B " << options.b << " MLA H " << 0 + NumHeads{} << " ";
  std::cout << "D_rope " << 0 + get<1>(HeadDim{}) << " D_latent " << 0 + get<0>(HeadDim{}) << " ";
  std::cout << "Q 1 K " << options.k << " Gen None ";
  std::cout << "Split " << options.split_kv << " Gen None ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  using Blocking = _128;
  std::string name = std::to_string((int) NumHeads{}) + "x" + std::to_string((int) Blocking{});
  std::string individual = " individual";
  std::string persistent = " persistent";
#if FP8
  name += " fp8";
  // Persistent Tile Scheduler
  run(Shape<NumHeads, Blocking, HeadDim>{}, (name + persistent).c_str(), IsPersistent<true>{});
  // Individual Tile Scheduler
  if (!options.is_fused_reduction || options.split_kv == 1) {
    run(Shape<NumHeads, Blocking, HeadDim>{}, (name + individual).c_str(), IsPersistent<false>{});
  }
#elif FP16
  name += " fp16";
  // Persistent Tile Scheduler
  run(Shape<NumHeads, Blocking, HeadDim>{}, (name + persistent).c_str(), IsPersistent<true>{});
  // Individual Tile Scheduler
  if (!options.is_fused_reduction || options.split_kv == 1) {
    run(Shape<NumHeads, Blocking, HeadDim>{}, (name + individual).c_str(), IsPersistent<false>{});
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

  run_mla(options, hw_info);
#endif

  return 0;
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
  bool causal_q_begin = true;
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
    std::string causal_type;
    cmd.get_cmd_line_argument<std::string>("causal-type", causal_type, "");
    if (mask == "no" || mask == "") {
      causal = residual = false;
      if (varlen) {
        residual = true;
      }
    }
    else if (mask == "causal") {
      residual = false;
      causal = true;
      if(causal_type == "qend") {
        causal_q_begin = false;
      } else {
        causal_q_begin = true;
      }
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
      << "  --causal-type=<qbegin|qend> Causal mask type\n"
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

    auto [Q, K, D, HB] = problem_shape;

    auto problem_shape_ref = cute::make_tuple(Q, K, D, D, HB);

    fmha_reference(problem_shape_ref, mQ, mK, mV, mO, mLSE, ActiveMask{});

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
      if(options.causal_q_begin) {
        fn(CausalMask{});
      } else {
        fn(CausalMask<false>{});
      }
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
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    if constexpr (IsQBegin) {
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape), get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    } else {
      const int offset_q = get<1>(problem_size) - get<0>(problem_size);
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
        
    int trip_count = get_trip_count(blk_coord, tile_shape, problem_size);
    if constexpr (IsQBegin) {
      return std::min(trip_count, int(ceil_div(size<0>(tile_shape), size<1>(tile_shape))));
    } else {
      const int offset_tile_q = get<1>(problem_size) % get<1>(tile_shape);
      return std::min(trip_count, int(ceil_div(get<0>(tile_shape) + offset_tile_q, get<1>(tile_shape))));
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

template<bool kIsQBegin = true>
struct CausalForBackwardMask : CausalMask<kIsQBegin>, ResidualMaskForBackward {

  using Base = CausalMask<kIsQBegin>;

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
    int offset_q = 0;
    if constexpr (!kIsQBegin) {
      offset_q = get<1>(problem_size) - get<0>(problem_size);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      bool masked = (get<0>(pos) + offset_q < get<1>(pos)) || !elem_less(pos, problem_size);
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
    'key-files/examples/77_blackwell_fmha/device/fmha_device_bwd.hpp': r"""/***************************************************************************************************
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

// common
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/tensor.hpp"

#include "../device/fmha.hpp"
#include "../kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp"
#include "../kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"
#include "../kernel/fmha_kernel_bwd_sum_OdO.hpp"
#include "../kernel/fmha_kernel_bwd_convert.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class Element,
    class ElementAccumulator,
    class TileShape,
    bool IsMla,
    class Mask
>
class Sm100FmhaBwd {
public:
  /// Argument structure: User API
  struct Arguments {
    // Q K D D_VO HB
    ProblemShape problem_shape;

    const Element* ptr_Q;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_Q;
    const Element* ptr_K;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_K;
    const Element* ptr_V;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_V;

    const Element* ptr_O;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_O;
    const ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> stride_LSE;

    const Element* ptr_dO;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dO;

    Element* ptr_dQ;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dQ;
    Element* ptr_dK;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dK;
    Element* ptr_dV;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> stride_dV;

    ElementAccumulator softmax_scale;

    cutlass::KernelHardwareInfo hw_info;
  };

  using OperationSumOdO = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdSumOdO<ProblemShape, Element, ElementAccumulator>
  >;
  using OperationConvert = cutlass::fmha::device::FMHA<
    cutlass::fmha::kernel::FmhaKernelBwdConvert<ProblemShape, Element, ElementAccumulator>
  >;

  using OperationMha= cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaBwdKernelTmaWarpSpecialized<
          ProblemShape, Element, ElementAccumulator, TileShape, Mask
      >
  >;

  using OperationMla = cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaBwdMlaKernelTmaWarpSpecialized<
          ProblemShape, Element, ElementAccumulator, TileShape, Mask
      >
  >;

  using Operation = std::conditional_t<IsMla, OperationMla, OperationMha>;

  using Kernel = typename Operation::Kernel;

  struct Params {
    OperationSumOdO op_sum_OdO;
    Operation op;
    OperationConvert op_convert;
    ElementAccumulator* dQ_acc;
    size_t dQ_acc_size;
  };

private:
  Params params_;

  static typename OperationSumOdO::Arguments to_sum_OdO_arguments(
        Arguments const& args,
        ElementAccumulator* sum_odo = nullptr,
        ElementAccumulator* scaled_lse = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    auto stride_sum_OdO = make_stride(_1{}, make_stride(Q, Q*H));
    auto stride_scaled_lse = make_stride(_1{}, make_stride(Q, Q*H));
    auto log2_e = log2f(expf(1.0f));
    return typename OperationSumOdO::Arguments {
      args.problem_shape,
      args.ptr_O, args.stride_O,
      args.ptr_dO, args.stride_dO,
      sum_odo, stride_sum_OdO,
      args.ptr_LSE, args.stride_LSE,
      scaled_lse, stride_scaled_lse,
      -1.0f, -log2_e
    };
  }

  static typename OperationConvert::Arguments to_convert_arguments(Arguments const& args, ElementAccumulator* src = nullptr) {
    using namespace cute;
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    auto stride_src_dQ = make_stride(D, _1{}, make_stride(D*Q, D*Q*H));
    return typename OperationConvert::Arguments {
      args.problem_shape,
      src, stride_src_dQ,
      nullptr, stride_src_dQ,
      nullptr, stride_src_dQ,
      args.ptr_dQ, args.stride_dQ,
      nullptr, args.stride_dK,
      nullptr, args.stride_dV,
      args.softmax_scale
    };
  }

  static typename Operation::Arguments to_bwd_arguments(
      Arguments const& args,
      ElementAccumulator* sum_OdO = nullptr, cute::tuple<cute::_1, cute::tuple<int, int>> const& stride_sum_OdO = {},
      ElementAccumulator* scaled_lse = nullptr, cute::tuple<cute::_1, cute::tuple<int, int>> const& stride_scaled_lse = {},
      ElementAccumulator* dQ_acc = nullptr, cute::tuple<int, cute::_1, cute::tuple<int, int>> const& stride_dQ = {}) {
           
    return typename Operation::Arguments{
      args.problem_shape,
      { args.ptr_Q,  args.stride_Q,
        args.ptr_K,  args.stride_K,
        args.ptr_V,  args.stride_V,
        args.ptr_dO, args.stride_dO,
        scaled_lse, stride_scaled_lse,
        sum_OdO, stride_sum_OdO,
        dQ_acc, stride_dQ,
        args.softmax_scale },
      { args.ptr_dK, args.stride_dK,
        args.ptr_dV, args.stride_dV },
      args.hw_info
    };
  }

public:

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    Status status = Status::kSuccess;

    status = OperationSumOdO::can_implement(to_sum_OdO_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    status = OperationConvert::can_implement(to_convert_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    status = Operation::can_implement(to_bwd_arguments(args));
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    size_t workspace_bytes = 0;
    // OdO vector
    workspace_bytes += B*H*Q * sizeof(ElementAccumulator);
    // scaled LSE vector
    workspace_bytes += B*H*Q * sizeof(ElementAccumulator);
    // FP32 versions of outputs that are churned (start off with Q only)
    workspace_bytes += B*H*Q*D * sizeof(ElementAccumulator);
    return workspace_bytes;
  }

  /// Initializes state from arguments.
  Status
  initialize_split(Arguments const& args, void* workspace_dQ, void* workspace_sum_OdO, void* workspace_scaled_lse, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize_split() - workspace_dQ="
      << workspace_dQ << ", workspace_sum_OdO=" << workspace_sum_OdO << "stream: " << (stream ? "non-null" : "null"));

    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_sum_OdO);
    ElementAccumulator* scaled_lse = reinterpret_cast<ElementAccumulator*>(workspace_scaled_lse);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_dQ);
    params_.dQ_acc = dQ_acc;
    params_.dQ_acc_size = B*H*Q*D * sizeof(ElementAccumulator);
    auto args_sum_OdO = to_sum_OdO_arguments(args, sum_OdO, scaled_lse);
    auto args_convert = to_convert_arguments(args, dQ_acc);
    params_.op_sum_OdO.initialize(args_sum_OdO, nullptr, stream);
    params_.op_convert.initialize(args_convert, nullptr, stream);
    auto args_bwd = to_bwd_arguments(
        args, sum_OdO, args_sum_OdO.stride_sum_OdO,
        scaled_lse, args_sum_OdO.stride_scaled_lse,
        dQ_acc, args_convert.stride_src_dQ
    );
    params_.op.initialize(args_bwd, nullptr, stream);

    return Status::kSuccess;
  }

  /// Initializes state from arguments.
  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    auto [Q_, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    D = cutlass::round_up(D, 8);  // Alignment
    int Q = cutlass::round_up(static_cast<int>(Q_), 8);  // Alignment
    char* workspace_chr = reinterpret_cast<char*>(workspace);
    ElementAccumulator* sum_OdO = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    workspace_chr += B*H*Q * sizeof(ElementAccumulator);
    ElementAccumulator* scaled_lse = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    workspace_chr += B*H*Q * sizeof(ElementAccumulator);
    ElementAccumulator* dQ_acc = reinterpret_cast<ElementAccumulator*>(workspace_chr);
    return initialize_split(args, dQ_acc, sum_OdO, scaled_lse, stream);
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling Kernel::to_underling_arguments()
  static Status
  run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("FmhaDeviceBwd::run()");

    Status result = Status::kSuccess;
    result = params.op_sum_OdO.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    auto cuda_result = cudaMemsetAsync(params.dQ_acc, 0, params.dQ_acc_size, stream);
    if (cuda_result != cudaSuccess) {
       return Status::kErrorInternal;
    }

    result = params.op.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    result = params.op_convert.run(stream);
    if (result != Status::kSuccess) {
      return result;
    }

    return Status::kSuccess;
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::device

////////////////////////////////////////////////////////////////////////////////
""",
    'key-files/examples/77_blackwell_fmha/device/sm100_mla.hpp': r"""/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief An universal device layer for cutlass 3.x-style kernels.
*/

#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

#include "kernel/sm100_fmha_mla_tma_warpspecialized.hpp"
#include "kernel/sm100_fmha_mla_reduction.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device {

using namespace cute;
using namespace cutlass::fmha::kernel;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<
    class Kernel_
>
class MLA {
public:

  using Kernel = Kernel_;

  using ReductionKernel = cutlass::fmha::kernel::Sm100FmhaMlaReductionKernel<
      typename Kernel::ElementOut,
      typename Kernel::ElementAcc,
      typename Kernel::ElementAcc,
      Kernel::TileShapeH::value,
      Kernel::TileShapeL::value,
      256 /*Max split*/
  >;

  /// Argument structure: User API
  using KernelArguments = typename Kernel::Arguments;
  using ReductionArguments = typename ReductionKernel::Arguments;

  using Arguments = KernelArguments;

  /// Argument structure: Kernel API
  using KernelParams = typename Kernel::Params;
  using ReductionParams = typename ReductionKernel::Params;
  struct Params {
    KernelParams fmha_params;
    ReductionParams reduction_params;
  };

private:

  /// Kernel API parameters object
  Params params_;

  bool is_initialized(bool set = false) {
    static bool initialized = false;
    if (set) initialized = true;
    return initialized;
  }

  static ReductionArguments to_reduction_args(Arguments const& args) {
    auto [H, K, D, B] = args.problem_shape;
    return ReductionArguments{
      nullptr, args.epilogue.ptr_o, nullptr, args.epilogue.ptr_lse,
      args.mainloop.softmax_scale, B, args.split_kv, K, args.mainloop.ptr_seq, 
      args.ptr_split_kv, Kernel::TileShapeS::value
    };
  }

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  static void set_split_kv (KernelArguments& args) {
    if (args.split_kv >= 1) return;
    auto [H, K, D, B] = args.problem_shape; 
    int sm_count = args.hw_info.sm_count;
    int max_splits = ceil_div(K, 128);
    int sms_per_batch = max(1, sm_count / B);
    int split_heur = min(max_splits, sms_per_batch);
    int waves = ceil_div(B * split_heur, sm_count);
    int k_waves = ceil_div(max_splits, split_heur);
    int split_wave_aware = ceil_div(max_splits, k_waves);
    if (args.is_fused_reduction && split_wave_aware > 1) {
      args.split_kv = std::min(split_wave_aware, static_cast<int>(sm_count/2));
    } else {
      args.split_kv = split_wave_aware;
    }
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (! Kernel::can_implement(args)) {
      return Status::kInvalid;
    }
    if (! ReductionKernel::can_implement(to_reduction_args(args))) {
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    workspace_bytes += ReductionKernel::get_workspace_size(to_reduction_args(args));
    return workspace_bytes;
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("MLA::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = Kernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<Kernel>,
        Kernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = Kernel::initialize_workspace(args, workspace, stream);
    if (status != Status::kSuccess) {
      return status;
    }
    status = ReductionKernel::initialize_workspace(to_reduction_args(args), workspace, stream);
    if (status != Status::kSuccess) {
      return status;
    }
    KernelParams kernel_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum   = kernel_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = kernel_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace);
    // Initialize the Params structure
    params_ = Params {kernel_params, reduction_params};

    if (is_initialized()) return Status::kSuccess;

    // account for dynamic smem capacity if needed
    // no dynamic smem is needed for reduction kernel
    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result = cudaFuncSetAttribute(
          device_kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    is_initialized(true);

    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("MLA()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    auto fmha_params = Kernel::to_underlying_arguments(args, workspace);

    ReductionArguments reduction_args = to_reduction_args(args);
    if (reduction_args.split_kv > 1) {
      reduction_args.ptr_oaccum   = fmha_params.epilogue.ptr_o_acc;
      reduction_args.ptr_lseaccum = fmha_params.epilogue.ptr_lse_acc;
    }
    ReductionParams reduction_params = ReductionKernel::to_underlying_arguments(reduction_args, workspace); 
    // Initialize the Params structure
    params_ = Params {fmha_params, reduction_params};

    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling Kernel::to_underling_arguments()
  static Status
  run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("MLA::run()");
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params.fmha_params);
    auto [H, K, D, B] = params.fmha_params.problem_shape;
    auto [D_latent, D_rope] = D;

    // configure smem size and carveout
    int smem_size = Kernel::SharedStorageSize;

    Status launch_result;
    if (params.fmha_params.is_fused_reduction && params.reduction_params.split_kv > 1) {
      auto result = cudaMemsetAsync(params.fmha_params.epilogue.ptr_o, 0, sizeof(typename Kernel::ElementOut) * H * D_latent * B, stream);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
        "  cudaMemsetAsync() returned error: "
        << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
      auto total_bytes = H * B * (sizeof(int) + sizeof(typename Kernel::ElementLSE)) + 2 * B * sizeof(int);
      uint8_t* ws = reinterpret_cast<uint8_t*>(params.fmha_params.epilogue.ptr_lse_exchange_buff);
      result = cudaMemsetAsync(ws, 0, total_bytes, stream);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
        "  cudaMemsetAsync() returned error: "
        << cudaGetErrorString(result));
        return Status::kErrorInternal;;
      }
    }
    // Use extended launch API only for mainloops that use it
    if constexpr(Kernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
                   cute::size<1>(typename Kernel::ClusterShape{}),
                   cute::size<2>(typename Kernel::ClusterShape{}));
      void const* kernel = (void const*) device_kernel<Kernel>;
      void* kernel_params[] = {&params.fmha_params};
      launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
    }
    else {
      launch_result = Status::kSuccess;
      device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params.fmha_params);
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess != result or Status::kSuccess != launch_result) {
      //return Status::kSuccess;
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
    if (!params.fmha_params.is_fused_reduction && params.reduction_params.split_kv > 1) {
      // launch reduction kernel
      dim3 const block = ReductionKernel::get_block_shape();
      dim3 const grid  = ReductionKernel::get_grid_shape(params.reduction_params);
      device_kernel<ReductionKernel><<<grid, block, 0, stream>>>(params.reduction_params);
      cudaError_t result = cudaGetLastError();
      if (cudaSuccess == result) {
        return Status::kSuccess;
      }
      else {
        CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
        return Status::kErrorInternal;
      }
    }
    else {
      return Status::kSuccess;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::device

////////////////////////////////////////////////////////////////////////////////
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
    auto [Q, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    if (Q <= 0 || K <= 0 || D <= 0 || D_VO <= 0 || H <= 0 || B <= 0) {
      return false;
    }
    if (D % Alignment != 0 || D_VO % Alignment != 0) {
      return false;
    }
    return true;
  }


  static Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return Status::kSuccess;
  }


  static Params to_underlying_arguments(Arguments const& args, void*) {
    auto [Q_, K_, D, D_VO, HB] = args.problem_shape;
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
      make_shape(K, Q, D_VO, HB),
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

    auto [Q, K, D, D_VO, HB] = problem_shape;

    using X = Underscore;

    uint16_t mcast_mask = 0;

    auto mK_in = mainloop_params.tma_load_k.get_tma_tensor(make_shape(K, D, HB));
    auto mV_in = mainloop_params.tma_load_v.get_tma_tensor(make_shape(K, D_VO, HB));
    auto mQ_in = mainloop_params.tma_load_q.get_tma_tensor(make_shape(Q, D, HB));
    auto mDO_in = mainloop_params.tma_load_do.get_tma_tensor(make_shape(Q, D_VO, HB));

    auto mK = domain_offset(select<1,2,4>(blk_offset), mK_in);
    auto mV = domain_offset(select<1,3,4>(blk_offset), mV_in);
    auto mQ = domain_offset(select<0,2,4>(blk_offset), mQ_in);
    auto mDO = domain_offset(select<0,3,4>(blk_offset), mDO_in);

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

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

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

    auto [Q, K, D, D_VO, HB] = problem_shape;

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

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
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
      if (elem_less(cDV(i), select<1,3>(problem_shape))) {
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

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto load_op = SM100_TMEM_LOAD_32dp32b16x{};

    auto tDKtDK = partition_fragment_C(TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
    tDKtDK.data() = TmemAllocation::kDK;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
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
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
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
    store(tTR_gDV, tTR_rDV, tTR_cDV, select<1,3>(problem_shape));

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


    auto [Q, K, D, D_VO, HB] = problem_shape;

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
      if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask>) {
        leading_causal_masking = warp_uniform(iter_index == get<1>(blk_coord));
      } else if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
        int offset = get<1>(problem_shape) - get<0>(problem_shape);
        int kv_left = get<1>(blk_coord) * TileShapeK{};
        int kv_right = kv_left + TileShapeK{} - 1;
        int q_left = iter_index * TileShapeQ{} + offset;
        int q_right = q_left + TileShapeQ{} - 1;

        leading_causal_masking = warp_uniform(!((q_left > kv_right) || (q_right < kv_left)));
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

    auto [Q, K, D, D_VO, HB] = problem_shape;

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

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

    auto blk_coord = make_coord(_0{}, blockIdx.x, _0{}, _0{}, make_coord(blockIdx.y, blockIdx.z));
    auto [problem_shape, blk_offset] = apply_variable_length_offset(
        params.problem_shape,
        blk_coord
    );
    int iter_count = ceil_div(get<0>(problem_shape), TileShapeQ{});
    int iter_start = 0;
    if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask>) {
      iter_start = (get<1>(blk_coord) * TileShapeK{}) / TileShapeQ{};
    } else if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
      int offset = get<1>(problem_shape) - get<0>(problem_shape);
      iter_start = max(0, (int(get<1>(blk_coord) * TileShapeK{}) - offset) / (int)TileShapeQ{});
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
    auto [Q, K, D, D_VO, HB] = params.problem_shape;
    auto [H, B] = HB;
    dim3 grid(ceil_div(K, TileShapeK{}), H, B);
    return grid;
  }
};

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp': r"""/***************************************************************************************************
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
struct Sm100FmhaBwdMlaKernelTmaWarpSpecialized {

  using TileShapeQ = decltype(get<0>(TileShape{}));
  using TileShapeK = decltype(get<1>(TileShape{}));
  using TileShapeDQK = decltype(get<2>(TileShape{}));
  using TileShapeDVO = decltype(get<3>(TileShape{}));

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  struct TmemAllocation {
    static constexpr uint32_t kDK = 0;                     // TileShapeK x TileShapeDQK x acc
    static constexpr uint32_t kDV = kDK + TileShapeDQK{};  // TileShapeK x TileShapeDVO x acc
    static constexpr uint32_t kDQ = kDV + TileShapeDVO{};  // TileShapeQ x TileShapeDQK x acc
    static constexpr uint32_t kDP = kDQ;                   // TileShapeK x TileShapeQ   x inp
    static constexpr uint32_t kS = kDQ + 65536 * 16;
    static constexpr uint32_t kP = kS;
    static constexpr uint32_t kTotal = kDQ + TileShapeDQK{};
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

  static constexpr int kLoadPerThread = TileShapeQ{} / NumThreadsPerWarp;
  static_assert(TileShapeQ{} % NumThreadsPerWarp == 0, "TileShapeQ must be divisible by NumThreadsPerWarp");
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
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeK, TileShapeDQK>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;

  // compute dP
  using CollectiveMmaDOV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStrideContiguousK, Alignment,
      Element, TensorStrideContiguousK, Alignment,
      ElementAcc,
      Shape<TileShapeQ, TileShapeK, TileShapeDVO>,
      ClusterShape, cutlass::gemm::collective::StageCount<kStages>,
      Schedule>::CollectiveOp;
  using TileShapeDOV = typename CollectiveMmaDOV::TileShape;
  using TiledMmaDOV = typename CollectiveMmaDOV::TiledMma;

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
  using TiledMmaPDO = typename CollectiveMmaPDO::TiledMma;

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

  using SmemLayoutK = decltype(restage(typename CollectiveMmaQK::SmemLayoutB{}));
  using SmemLayoutV = decltype(restage(typename CollectiveMmaDOV::SmemLayoutB{}));
  using SmemLayoutQ = decltype(restage(typename CollectiveMmaQK::SmemLayoutA{}, _2{}));
  using SmemLayoutDO = decltype(restage(typename CollectiveMmaDOV::SmemLayoutA{}, _1{}));
  using SmemLayoutDS = decltype(restage(typename CollectiveMmaDSK::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutLSE = Layout<Shape<TileShapeQ, _1>>;
  using SmemLayoutSumOdO = Layout<Shape<TileShapeQ, _1>>;

  using SmemLayoutQT = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutB{}, _2{}));
  using SmemLayoutKT = decltype(restage(typename CollectiveMmaDSK::SmemLayoutB{}));
  using SmemLayoutDST = decltype(restage(typename CollectiveMmaDSQ::SmemLayoutA{}, Int<kStagesComputeSmem>{}));
  using SmemLayoutDOT = decltype(restage(typename CollectiveMmaPDO::SmemLayoutB{}, _1{}));
  using SmemLayoutP = decltype(restage(typename CollectiveMmaPDO::SmemLayoutA{}, _1{}));
  using SmemLayoutPT = decltype(restage(typename CollectiveMmaDSK::SmemLayoutA{}, _1{}));

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
    union{
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutP>> smem_p;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutPT>> smem_p_t;
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

  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaDOV::Params::TMA_B;
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_DO = typename CollectiveMmaDOV::Params::TMA_A;

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
    auto [Q, K, D, D_VO, HB] = args.problem_shape;
    auto [H, B] = HB;
    if (Q <= 0 || K <= 0 || D <= 0 || H <= 0 || B <= 0 || D_VO <= 0) {
      return false;
    }
    if (D % Alignment != 0 || D_VO % Alignment != 0) {
      return false;
    }
    return true;
  }


  static Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return Status::kSuccess;
  }


  static Params to_underlying_arguments(Arguments const& args, void*) {
    auto [Q_, K_, D, D_VO, HB] = args.problem_shape;
    int Q = Q_;
    int K = K_;

    if constexpr (is_variable_length_v<decltype(Q_)>) {
      Q = Q_.total_length;
    }
    if constexpr (is_variable_length_v<decltype(K_)>) {
      K = K_.total_length;
    }

    auto params_kq = CollectiveMmaQK::to_underlying_arguments(
      make_shape(Q, K, D, HB),
      typename CollectiveMmaQK::Arguments {
        args.mainloop.ptr_q, args.mainloop.stride_q,
        args.mainloop.ptr_k, args.mainloop.stride_k,
      }, /*workspace=*/nullptr);

    auto params_vdo = CollectiveMmaDOV::to_underlying_arguments(
      make_shape(Q, K, D_VO, HB),
      typename CollectiveMmaDOV::Arguments {
        args.mainloop.ptr_do, args.mainloop.stride_do,
        args.mainloop.ptr_v, args.mainloop.stride_v,
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
        params_kq.tma_load_b,
        params_vdo.tma_load_b,
        params_kq.tma_load_a,
        params_vdo.tma_load_a,
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

    auto [Q, K, D, D_VO, HB] = problem_shape;

    using X = Underscore;

    uint16_t mcast_mask = 0;

    auto mK_in = mainloop_params.tma_load_k.get_tma_tensor(make_shape(K, D, HB));
    auto mV_in = mainloop_params.tma_load_v.get_tma_tensor(make_shape(K, D_VO, HB));
    auto mQ_in = mainloop_params.tma_load_q.get_tma_tensor(make_shape(Q, D, HB));
    auto mDO_in = mainloop_params.tma_load_do.get_tma_tensor(make_shape(Q, D_VO, HB));

    auto mK = domain_offset(select<1,2,4>(blk_offset), mK_in);
    auto mV = domain_offset(select<1,3,4>(blk_offset), mV_in);
    auto mQ = domain_offset(select<0,2,4>(blk_offset), mQ_in);
    auto mDO = domain_offset(select<0,3,4>(blk_offset), mDO_in);

    auto gK = local_tile(mK, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gQ = local_tile(mQ, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gV = local_tile(mV, TileShapeDOV{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gDO = local_tile(mDO, TileShapeDOV{}, make_coord(_,_,_), Step<_1, X, _1>{});

    ThrMMA cta_mma_kq = TiledMmaQK{}.get_slice(_0{});
    ThrMMA cta_mma_vdo = TiledMmaDOV{}.get_slice(_0{});

    auto tSTgK = cta_mma_kq.partition_B(gK);
    auto tSTgQ = cta_mma_kq.partition_A(gQ);
    auto tDPTgV = cta_mma_vdo.partition_B(gV);
    auto tDPTgDO = cta_mma_vdo.partition_A(gDO);

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

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

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
    // 32 threads loading kLoadPerThread * 32 values of 32b each

    int thread_idx = threadIdx.x % NumThreadsPerWarp;
    int smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * kLoadPerThread;
    int gmem_idx = TileShapeQ{} * iter_index + thread_idx * kLoadPerThread;
    auto mLSE = make_tensor(mainloop_args.ptr_lse, make_shape(Q, HB), mainloop_args.stride_lse);
    for (int i = 0; i < kLoadPerThread; i++) {
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
    smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * kLoadPerThread;
    gmem_idx = TileShapeQ{} * iter_index + thread_idx * kLoadPerThread;
    auto mSumOdO = make_tensor(mainloop_args.ptr_sum_odo, make_shape(Q, HB), mainloop_args.stride_sum_odo);
    for (int i = 0; i < kLoadPerThread; i++) {
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
      smem_idx = TileShapeQ{} * pipeline_load_compute_lse_producer_state.index() + thread_idx * kLoadPerThread;
      gmem_idx = TileShapeQ{} * iter_index + thread_idx * kLoadPerThread;
      for (int i = 0; i < kLoadPerThread; i++) {
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
      smem_idx = TileShapeQ{} * pipeline_load_compute_sum_odo_producer_state.index() + thread_idx * kLoadPerThread;
      gmem_idx = TileShapeQ{} * iter_index + thread_idx * kLoadPerThread;
      for (int i = 0; i < kLoadPerThread; i++) {
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

    auto [Q, K, D, D_VO, HB] = problem_shape;

    auto sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(shared_tensors.smem_k.begin()), SmemLayoutK{});
    auto sV = make_tensor(make_smem_ptr(shared_tensors.smem_v.begin()), SmemLayoutV{});
    auto sDO = make_tensor(make_smem_ptr(shared_tensors.smem_do.begin()), SmemLayoutDO{});

    auto sQT = make_tensor(make_smem_ptr(shared_tensors.smem_q_t.begin()), SmemLayoutQT{});
    auto sKT = make_tensor(make_smem_ptr(shared_tensors.smem_k_t.begin()), SmemLayoutKT{});
    auto sDS = make_tensor(make_smem_ptr(shared_tensors.smem_ds.begin()), SmemLayoutDS{});
    auto sDST = make_tensor(make_smem_ptr(shared_tensors.smem_ds_t.begin()), SmemLayoutDST{});
    auto sP = make_tensor(make_smem_ptr(shared_tensors.smem_p.begin()), SmemLayoutP{});
    auto sDOT = make_tensor(make_smem_ptr(shared_tensors.smem_do_t.begin()), SmemLayoutDOT{});

    Tensor tSTrK = TiledMmaQK::make_fragment_B(sK);
    Tensor tSTrQ = TiledMmaQK::make_fragment_A(sQ);

    Tensor tDPTrV = TiledMmaDOV::make_fragment_B(sV);
    Tensor tDPTrDO = TiledMmaDOV::make_fragment_A(sDO);

    Tensor tDQrDS = TiledMmaDSK::make_fragment_A(sDS);
    Tensor tDQrKT = TiledMmaDSK::make_fragment_B(sKT);

    Tensor tDKrDST = TiledMmaDSQ::make_fragment_A(sDST);
    Tensor tDKrQT = TiledMmaDSQ::make_fragment_B(sQT);

    Tensor tDVrP = TiledMmaPDO::make_fragment_A(sP);
    Tensor tDVrDOT = TiledMmaPDO::make_fragment_B(sDOT);

    TiledMmaQK tiled_mma_qk;
    TiledMmaDOV tiled_mma_dov;
    TiledMmaDSK tiled_mma_dsk;
    TiledMmaDSQ tiled_mma_dsq;
    TiledMmaPDO tiled_mma_pdo;

    tiled_mma_dsq.accumulate_ = UMMA::ScaleOut::Zero;
    tiled_mma_pdo.accumulate_ = UMMA::ScaleOut::Zero;

    Tensor tSTtST =  partition_fragment_C(tiled_mma_qk, select<0,1>(TileShapeQK{}));
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT = partition_fragment_C(tiled_mma_dov, select<0,1>(TileShapeDOV{}));
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
    tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
      cute::gemm(tiled_mma_qk,
                 tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                 tSTrK(_,_,k_block,_0{}),
                 tSTtST);
      tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
    }

    ++pipeline_load_mma_q_consumer_state;

    pipeline_mma_compute_s.producer_commit(pipeline_mma_compute_s_producer_state);
    ++pipeline_mma_compute_s_producer_state;

    pipeline_load_mma_do.consumer_wait(pipeline_load_mma_do_consumer_state);

    pipeline_mma_compute_dp.producer_acquire(pipeline_mma_compute_dp_producer_state);
    pipeline_mma_reduce_dq.producer_acquire(pipeline_mma_reduce_dq_producer_state);

    // dP = dO*V
    tiled_mma_dov.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
      cute::gemm(tiled_mma_dov,
                 tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                 tDPTrV(_,_,k_block,_0{}),
                 tDPTtDPT);
      tiled_mma_dov.accumulate_ = UMMA::ScaleOut::One;
    }

    pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
    ++pipeline_mma_compute_dp_producer_state;

    pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

    // dV = P*dO
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
      cute::gemm(tiled_mma_pdo,
                 tDVrP(_,_,k_block,_0{}),
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
      tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSTrQ); ++k_block) {
        cute::gemm(tiled_mma_qk,
                   tSTrQ(_,_,k_block,pipeline_load_mma_q_consumer_state.index()),
                   tSTrK(_,_,k_block,_0{}),
                   tSTtST);
        tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
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
      tiled_mma_dov.accumulate_ = UMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDPTrV); ++k_block) {
        cute::gemm(tiled_mma_dov,
                   tDPTrDO(_,_,k_block,pipeline_load_mma_do_consumer_state.index()),
                   tDPTrV(_,_,k_block,_0{}),
                   tDPTtDPT);
        tiled_mma_dov.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_mma_compute_dp.producer_commit(pipeline_mma_compute_dp_producer_state);
      ++pipeline_mma_compute_dp_producer_state;

      pipeline_compute_mma_p.consumer_wait(pipeline_compute_mma_p_consumer_state);

      // dV = P*dO
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tDVrP); ++k_block) {
        cute::gemm(tiled_mma_pdo,
                   tDVrP(_,_,k_block,_0{}),
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

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
    auto gDK = local_tile(mDK, TileShapeDSQ{}, make_coord(_,_,_), Step<_1, _1, X>{})
        (_, _, blk_coord_k, _0{}, blk_coord_batch);

    Tensor cDK = domain_offset(
        make_coord(get<1>(blk_coord) * TileShapeK{}, _0{}),
        make_identity_tensor(take<0,2>(TileShapeDSQ{}))
    );

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
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
      if (elem_less(cDV(i), select<1,3>(problem_shape))) {
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

    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    auto load_op = SM100_TMEM_LOAD_32dp32b16x{};

    auto tDKtDK = partition_fragment_C(TiledMmaDSQ{}, select<0,1>(TileShapeDSQ{}))(make_coord(_,_),_0{},_0{});
    tDKtDK.data() = TmemAllocation::kDK;

    auto mDK_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dk), make_shape(K, TileShapeDQK{}, HB), epilogue_args.stride_dk);
    auto mDK = domain_offset(select<1,2,4>(blk_offset), mDK_in);
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

    auto tDVtDV = partition_fragment_C(TiledMmaPDO{}, select<0,1>(TileShapePDO{}))(make_coord(_,_),_0{},_0{});
    tDVtDV.data() = TmemAllocation::kDV;

    auto mDV_in = make_tensor(make_gmem_ptr(epilogue_args.ptr_dv), make_shape(K, TileShapeDVO{}, HB), epilogue_args.stride_dv);
    auto mDV = domain_offset(select<1,3,4>(blk_offset), mDV_in);
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
    store(tTR_gDV, tTR_rDV, tTR_cDV, select<1,3>(problem_shape));

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


    auto [Q, K, D, D_VO, HB] = problem_shape;

    // in tmem, S & P overlap
    // and dP and dQ overlap

    // there are two compute wg's that cooperatively compute softmax
    // they are striped by this tmem atom, i.e. wg0 has 16 elems, then wg1 etc

    auto load_op = SM100_TMEM_LOAD_16dp32b32x{};

    Tensor tSTtST =  partition_fragment_C(TiledMmaQK{}, select<0,1>(TileShapeQK{}))(make_coord(_,_),_0{},_0{});
    tSTtST.data() = TmemAllocation::kS;

    Tensor tDPTtDPT =  partition_fragment_C(TiledMmaDOV{}, select<0,1>(TileShapeDOV{}))(make_coord(_,_),_0{},_0{});
    tDPTtDPT.data() = TmemAllocation::kDP;

    Tensor cST = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor cDPT = make_identity_tensor(take<0,2>(TileShapeDOV{}));
    Tensor cPT = make_identity_tensor(take<0,2>(TileShapeQK{}));

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
    Tensor tTR_cPT_p = thread_t2r.partition_D(cPT);
    Tensor tTR_cDPT = split_wg(tTR_cDPT_p);
    Tensor tTR_rDPT = make_tensor<ElementAcc>(shape(tTR_cDPT));
    Tensor tTR_tDPT = split_wg(thread_t2r.partition_S(tDPTtDPT));

    Tensor sLSE = make_tensor(make_smem_ptr(shared_tensors.smem_lse.begin()), SmemLayoutLSE{});
    Tensor sSumOdO = make_tensor(make_smem_ptr(shared_tensors.smem_sum_odo.begin()), SmemLayoutSumOdO{});

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
      if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask>) {
        leading_causal_masking = warp_uniform(iter_index == get<1>(blk_coord));
      } else if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
        int offset = get<1>(problem_shape) - get<0>(problem_shape);
        int kv_left = get<1>(blk_coord) * TileShapeK{};
        int kv_right = kv_left + TileShapeK{} - 1;
        int q_left = iter_index * TileShapeQ{} + offset;
        int q_right = q_left + TileShapeQ{} - 1;

        leading_causal_masking = warp_uniform(!((q_left > kv_right) || (q_right < kv_left)));
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
            return make_coord(get<0>(c_transpose) + iter_index * TileShapeQ{}, get<1>(c_transpose) + get<1>(blk_coord) * TileShapeK{});
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
          lse.x = sLSE(get<0>(tTR_cST(i)), pipeline_load_compute_lse_consumer_state.index());
          lse.y = sLSE(get<0>(tTR_cST(i+1)), pipeline_load_compute_lse_consumer_state.index());
          cute::fma(out, softmax_scale_log2_e, acc, lse);
          tTR_rST(i) = ::exp2f(out.x);
          tTR_rST(i+1) = ::exp2f(out.y);
        }

        auto tRT_rST = quantize(tTR_rST);
        
        Tensor sP = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_p.begin()), SmemLayoutP{})
          (_, _, _, pipeline_compute_mma_p_producer_state.index());

        cutlass::arch::fence_view_async_tmem_load();
        cutlass::arch::NamedBarrier(
          kNumComputeWarps * NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::TransformBarrier
        ).arrive_and_wait();

        auto sP_pi = as_position_independent_swizzle_tensor(sP);

        auto thread_layout = make_ordered_layout(
            make_shape(_64{}, _32{}, _2{}, _2{}),
            make_stride(_3{}, _0{}, _1{}, _2{})
            );
        auto sP_pi_slice_p = sP_pi.compose(thread_layout)(((dp_idx/32) * 16) + (dp_idx % 16) , _, (dp_idx % 32 / 16), _).compose(make_layout(shape(tTR_cPT_p)));
        auto sP_pi_slice = split_wg(sP_pi_slice_p);
        copy_aligned(tRT_rST, sP_pi_slice);
      });

      // notify for P
      cutlass::arch::fence_view_async_shared();
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
        odo.x = sSumOdO(get<0>(tTR_cDPT(i)), pipeline_load_compute_sum_odo_consumer_state.index());
        odo.y = sSumOdO(get<0>(tTR_cDPT(i+1)), pipeline_load_compute_sum_odo_consumer_state.index());
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

      Tensor sDS = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_ds_t.begin()), SmemLayoutDST{})
          (_, _, _, pipeline_compute_mma_ds_producer_state.index());

      auto thread_layout = make_ordered_layout(
          make_shape(_64{}, _32{}, _2{}, _2{}),
          make_stride(_3{}, _0{}, _1{}, _2{})
          );
      auto sDS_pi = as_position_independent_swizzle_tensor(sDS);
      auto sDS_pi_slice_p = sDS_pi.compose(thread_layout)(((dp_idx/32) * 16) + (dp_idx % 16) , _, (dp_idx % 32 / 16), _).compose(make_layout(shape      (tTR_cDPT_p)));
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

    auto [Q, K, D, D_VO, HB] = problem_shape;

    auto [blk_coord_q, blk_coord_k, blk_coord_d, blk_coord_dv, blk_coord_batch] = blk_coord;

    // must match TileShapeDQ
    auto load_op = SM100_TMEM_LOAD_16dp32b16x{};

    auto tDQtDQ = partition_fragment_C(TiledMmaDSK{}, select<0,1>(TileShapeDSK{}))(make_coord(_,_),_0{},_0{});
    tDQtDQ.data() = TmemAllocation::kDQ;

    Tensor mDQ = mainloop_params.tma_red_dq.get_tma_tensor(make_shape(Q, D, HB));
    auto gDQ = local_tile(mDQ, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{})
        (_, _, _, _0{}, blk_coord_batch);

    Tensor cDQ = make_identity_tensor(take<0,2>(TileShapeDSK{}));

    Tensor sDQ = make_tensor(make_smem_ptr(shared_tensors.smem_dq.begin()), SmemLayoutDQ{});

    int thread_idx = threadIdx.x % (kNumReduceWarps * NumThreadsPerWarp);
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

    auto blk_coord = make_coord(_0{}, blockIdx.x, _0{}, _0{}, make_coord(blockIdx.y, blockIdx.z));
    auto [problem_shape, blk_offset] = apply_variable_length_offset(
        params.problem_shape,
        blk_coord
    );
    int iter_count = ceil_div(get<0>(problem_shape), TileShapeQ{});
    int iter_start = 0;
    if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<true>, Mask>) {
      iter_start = (get<1>(blk_coord) * TileShapeK{}) / TileShapeQ{};
    } else if constexpr (std::is_base_of_v<cutlass::fmha::collective::CausalMask<false>, Mask>) {
      int offset = get<1>(problem_shape) - get<0>(problem_shape);
      iter_start = max(0, (int(get<1>(blk_coord) * TileShapeK{}) - offset) / (int)TileShapeQ{});
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
    auto [Q, K, D, D_VO, HB] = params.problem_shape;
    auto [H, B] = HB;
    dim3 grid(ceil_div(K, TileShapeK{}), H, B);
    return grid;
  }
};

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_fmha_mla_reduction.hpp': r"""/***************************************************************************************************
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
#include "cutlass/arch/arch.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
template<
    class ElementOut,
    class ElementAcc,
    class ElementScale,
    size_t kNumHeads,
    size_t kHeadDimLatent,
    int kMaxSplits
>
struct Sm100FmhaMlaReductionKernel {

  static const int SharedStorageSize = 0;
  static const int MaxThreadsPerBlock = 128;
  static const int MinBlocksPerMultiprocessor = 1;

  using ArchTag = cutlass::arch::Sm100;

  static_assert(kHeadDimLatent % MaxThreadsPerBlock == 0);
  struct Arguments {
    ElementAcc* ptr_oaccum = nullptr;
    ElementOut* ptr_o = nullptr;
    ElementAcc* ptr_lseaccum = nullptr;
    ElementAcc* ptr_lse = nullptr;
    ElementScale scale = 1.f;
    int num_batches = 0;
    int split_kv = -1;
    int dim_k = -1;
    int* ptr_seq = nullptr;
    int* ptr_split_kv = nullptr;
    int tile_shape_s = 128;
  };
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {args.ptr_oaccum, args.ptr_o, args.ptr_lseaccum, args.ptr_lse, 
	    args.scale, args.num_batches, args.split_kv, args.dim_k, args.ptr_seq, 
	    args.ptr_split_kv, args.tile_shape_s};    
  }

  static size_t get_workspace_size(Arguments const& /*args*/) {
    return 0;
  }

  static Status initialize_workspace(
      Arguments const& /*args*/, void* /*ws*/, cudaStream_t /*stream*/) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(kNumHeads, 1, params.num_batches);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  static bool can_implement(Arguments const& args) {
    if (args.num_batches <= 0) return false;
    if (args.split_kv <= 0) return false;
    return true;
  }

  CUTLASS_DEVICE void operator() (Params const& params, char* smem_raw) {
    if (params.split_kv <= 1) return;

    auto blk_coord = make_coord(blockIdx.x, _0{}, blockIdx.z);

    auto dim_k = params.ptr_seq == nullptr ?  params.dim_k : params.ptr_seq[get<2>(blk_coord)];
    auto local_split_kv = params.ptr_split_kv == nullptr ? params.split_kv : params.ptr_split_kv[get<2>(blk_coord)];
    auto k_tile_total = ceil_div(dim_k, params.tile_shape_s);
    auto k_tile_per_cta = ceil_div(k_tile_total, local_split_kv);
    local_split_kv = ceil_div(k_tile_total, k_tile_per_cta);

    if (local_split_kv == 1) return;

    __shared__ ElementAcc sLseScale[kMaxSplits];
    const size_t offset_lseaccum = get<0>(blk_coord) + kNumHeads * params.split_kv * get<2>(blk_coord);
    const size_t offset_lse = get<0>(blk_coord) + kNumHeads * get<2>(blk_coord);

    Tensor gLSEaccum = make_tensor(make_gmem_ptr(params.ptr_lseaccum + offset_lseaccum),
                                   make_shape(params.split_kv), Stride<Int<kNumHeads>>{});

    Tensor gLSE = make_tensor(make_gmem_ptr(params.ptr_lse + offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
      constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

      ElementAcc local_lse[kNLsePerThread];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        local_lse[i] = split < local_split_kv ? gLSEaccum(split) : -std::numeric_limits<ElementAcc>::infinity();
      }

      ElementAcc lse_max = -std::numeric_limits<ElementAcc>::infinity();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        lse_max = fmax(local_lse[i], lse_max);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        lse_max = fmax(__shfl_xor_sync(0xffffffff, lse_max, offset), lse_max);
      }

      lse_max = __shfl_sync(0xffffffff, lse_max, 0);

      ElementAcc sum_lse = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        sum_lse = sum_lse + expf(local_lse[i] - lse_max);
      }
    
      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        sum_lse = sum_lse + __shfl_xor_sync(0xffffffff, sum_lse, offset);
      }

      sum_lse = __shfl_sync(0xffffffff, sum_lse, 0);

      ElementAcc global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? std::numeric_limits<ElementAcc>::infinity() : logf(sum_lse) + lse_max;
      if (threadIdx.x == 0 and params.ptr_lse != nullptr) {
        gLSE(0) = global_lse;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        if (split < local_split_kv) {
          sLseScale[split] = expf(local_lse[i] - global_lse);
        }
      }
    }
    __syncthreads();

    constexpr int Elements = kHeadDimLatent / MaxThreadsPerBlock;
    const size_t offset_oaccum = kHeadDimLatent * params.split_kv * (get<0>(blk_coord) + kNumHeads * get<2>(blk_coord));
    Tensor gOaccum = make_tensor(make_gmem_ptr(params.ptr_oaccum + offset_oaccum),
                               Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});
    ElementAcc local_val[Elements] = {0};
    for (int split = 0; split < local_split_kv; ++split) {
      ElementAcc lse_scale = sLseScale[split];
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < Elements; ++i) {
        local_val[i] += lse_scale * gOaccum(threadIdx.x + MaxThreadsPerBlock * i);
      }
      gOaccum.data() = gOaccum.data() + kHeadDimLatent;
    }
    auto ptr_o_local = params.ptr_o + (get<0>(blk_coord) + get<2>(blk_coord) * kNumHeads) * kHeadDimLatent;
    Tensor gO = make_tensor(make_gmem_ptr(ptr_o_local), Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < Elements; ++i) {
      gO(threadIdx.x + MaxThreadsPerBlock * i) = static_cast<ElementOut>(local_val[i]);
    }
  }
};

}  // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_fmha_mla_tma_warpspecialized.hpp': r"""/***************************************************************************************************
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
#include "cute/arch/simd_sm100.hpp"
#include "cutlass/barrier.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "gather_tensor.hpp"  // from examples/common
#include "common/pow_2.hpp"
#include "sm100_mla_tile_scheduler.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

template<
    class TileShape,
    class Element_,
    class ElementAcc_,
    class ElementOut_,
    class ElementLSE_,
    class TileScheduler,
#ifdef CPASYNC
    bool kIsCpAsync = true
#else
    bool kIsCpAsync = false
#endif
>
struct Sm100FmhaMlaKernelTmaWarpspecialized {

  using Element = Element_;
  using ElementAcc = ElementAcc_;
  using ElementOut = ElementOut_;
  using ElementLSE = ElementLSE_;
  
  // only 2Sm mode is supported
  static const bool kIs2Sm = true;
  static const int MaxThreadsPerBlock = 256;
  static const int MinBlocksPerMultiprocessor = 1;
  static const int TotalSNum = 2;
  static const int TotalPNum = 2;
  using ArchTag = cutlass::arch::Sm100;

  using ClusterShape = cute::conditional_t<kIs2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;

  using TileShapeH = tuple_element_t<0, TileShape>;
  using TileShapeS = tuple_element_t<1, TileShape>;
  using TileShapeD = tuple_element_t<2, TileShape>;

  using TileShapeL = tuple_element_t<0, TileShapeD>;
  using TileShapeR = tuple_element_t<1, TileShapeD>;
  static_assert(TileShapeL{} % TileShapeR{} == 0, "Rope head dim must divide latent head dim");

  using ProblemShape  = Shape<TileShapeH, int, TileShapeD, int>;
  using TensorStride  = Stride<int64_t, _1, int64_t>;
  using TmemAllocator = cute::conditional_t<kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  static_assert(TileShapeH{} == 128);
  static const int kWarpsInN = kIs2Sm ? 2 : 1;

  static const int kNumComputeWarps = 4;
  static const int kNumLoadWarps = kIsCpAsync ? 2 : 1;

  enum class WarpRole {
    kMma = 0x1, kLoad = 0x2, kCompute = 0x3, kLoadPageTable = 0x4, kEmpty=0x0
  };

  static const long long unsigned int kWarpAssignment = kIsCpAsync ? 0x4221'3333ull : 0x0021'3333ull;

  static CUTLASS_DEVICE WarpRole warp_idx_to_role(int warp_idx) {
      return static_cast<WarpRole>((kWarpAssignment >> (4 * warp_idx)) & 0xF);
  }

  static const int Alignment = 128 / sizeof_bits_v<Element>;
  static const int AlignmentOut = 128 / sizeof_bits_v<ElementOut>;

  using TileShapeQK = Shape<TileShapeH, TileShapeS, decltype(TileShapeR{} / _1{})>;
  static const int StagesQK = 24 / sizeof(Element);  // free parameter
  static const int IterationsQKLatent = decltype(TileShapeL{} / get<2>(TileShapeQK{}))::value;
  static const int IterationsQKRope = decltype(TileShapeR{} / get<2>(TileShapeQK{}))::value;
  static const int IterationsQK = IterationsQKLatent + IterationsQKRope;

  using Schedule = cute::conditional_t<kIs2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStride, Alignment,
      Element, TensorStride, Alignment,
      ElementAcc,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<StagesQK>,
      Schedule>::CollectiveOp;
  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using CtaShapeQK = typename CollectiveMmaQK::CtaShape_MNK;

  // chosen for unified smem staging between K and V
  using TileShapePV = Shape<TileShapeH, _256, _32>;
  using TransposeTensorStride = decltype(select<1,0,2>(TensorStride{}));
  static const int StagesPV = StagesQK;  // not sure why, but must be at least two. check pipes
  static const int IterationsPV_K = decltype(TileShapeS{} / get<2>(TileShapePV{}))::value;
  static const int IterationsPV_N = decltype(TileShapeL{} / get<1>(TileShapePV{}))::value;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, TensorStride, Alignment,
      Element, TransposeTensorStride, Alignment,
      ElementAcc,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<StagesPV>,
      Schedule>::CollectiveOp;
  using CtaShapePV = typename CollectiveMmaPV::CtaShape_MNK;
  static_assert(std::is_same_v<TransposeTensorStride, typename CollectiveMmaPV::StrideB>);

  using TiledMmaPV = typename CollectiveMmaPV::TiledMma;

  using AtomThrShapeMNK = typename CollectiveMmaQK::AtomThrShapeMNK;
  static_assert(typename CollectiveMmaQK::AtomThrShapeMNK{} == typename CollectiveMmaPV::AtomThrShapeMNK{}, "schedule must match");

  static const int StagesPageTable = kIsCpAsync ? StagesPV : 1;

  // pipelines from load to mma, PipelineTmaUmmaAsync, stages tbd
  // use expect_tx for Q load
  using PipelineLoadQK = cute::conditional_t<kIsCpAsync, PipelineUmmaConsumerAsync<StagesQK, AtomThrShapeMNK>, PipelineTmaUmmaAsync<StagesQK, ClusterShape, AtomThrShapeMNK>>;
  using PipelineLoadPV = PipelineLoadQK;
  // pipeline from mma (Q@K) to softmax, PipelineUmmaAsync, 2 stages
  using PipelineS = PipelineUmmaAsync<TotalSNum, AtomThrShapeMNK>;
  // pipeline from softmax (P) to mma (bmm2), PipelineUmmaAsync, 2 stages
  using PipelineP = PipelineUmmaConsumerAsync<TotalPNum, AtomThrShapeMNK>;
  // pipeline from mma to softmax (for rescale), PipelineUmmaAsync, 1 stage
  using PipelineO = PipelineUmmaAsync<1, AtomThrShapeMNK>;

  using PipelinePT = PipelineAsync<StagesPageTable>;

  struct PipelineStorage {
    alignas(16) typename PipelineLoadQK::SharedStorage load_qk;
    alignas(16) typename PipelineS::SharedStorage mma_s;
    alignas(16) typename PipelineP::SharedStorage p_mma;
    alignas(16) typename PipelineO::SharedStorage mma_o;
    alignas(16) typename PipelinePT::SharedStorage load_page_table;
  };

  template<class Layout, class Stages = _1>
  static CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
      return composition(layout, make_tuple(_, _, _, make_layout(stages)));
  }

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<IterationsQK>{}));
  using SmemLayoutKC = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutVC = typename CollectiveMmaPV::SmemLayoutB;
  using SmemLayoutP = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutA{}, make_shape(Int<IterationsPV_K>{}, _2{})));
  using SmemLayoutOut = decltype(take<0,2>(typename CollectiveMmaQK::CtaShape_MNK{}));
  using TileShapeAcc = decltype(take<0,2>(typename CollectiveMmaPV::CtaShape_MNK{}));

  static const int kBytesLoadQ  = size(AtomThrShapeMNK{}) * cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static const int kBytesLoadKC = size(AtomThrShapeMNK{}) * cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutKC{})) * cute::sizeof_bits_v<Element>);
  static const int kBytesLoadVC = size(AtomThrShapeMNK{}) * cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutVC{})) * cute::sizeof_bits_v<Element>);

  // pre-condition for overlapped smem staging
  static_assert(kBytesLoadKC == kBytesLoadVC);
  static_assert(StagesQK == StagesPV);

  static const int kTransactionsBytesLoadQK = kBytesLoadKC;
  static const int kTransactionsBytesLoadExtraQ = kBytesLoadQ;
  static const int kTransactionsBytesLoadPV = kBytesLoadVC;

  static const int kNamedBarrierExchange = (int) cutlass::arch::ReservedNamedBarriers::TransformBarrier;
  // This Named Barrier is introduced to solve Q tile loading overwritten issue when enable persistent 
  // tile scheduler for FP8 MLA.
  static const int kNamedBarrierEpilogue = (int) cutlass::arch::ReservedNamedBarriers::EpilogueBarrier;
  // 
  static const int kNamedBarrierTmemDealloc = (int) cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier;

  enum class TmemAllocation : uint32_t {
    kSizeS = TileShapeS::value / kWarpsInN,
    // Overall
    kSizeO = TileShapeL::value / kWarpsInN,
    // Between accumulators we loop over
    kSizeAccO = decltype(get<1>(TileShapePV{}))::value / kWarpsInN,
    kNumS = TotalSNum,
    kNumP = TotalPNum,
    kNumO = 1,
    kS0 = 0,
    kS1 = kS0 + kSizeS,
    kO0 = kS1 + kSizeS,
    kTotal = kO0 + kSizeO
  };

  static_assert(static_cast<int>(TmemAllocation::kTotal) <= TmemAllocator::Sm100TmemCapacityColumns, "using too much tmem");

  struct TensorStorage {
    // to communicate max and row_sum
    cute::array<ElementAcc, kNumComputeWarps * cutlass::NumThreadsPerWarp> smem_exchange;
    cute::array<int, StagesPageTable * TileShapeS::value> smem_page_table;
    alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutKC>> smem_kc;
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutVC>> smem_vc;
    };
    union {
      alignas(2048) cute::array<Element, cute::cosize_v<SmemLayoutP>> smem_p;
      alignas(2048) cute::array<ElementOut, size(TileShapeAcc{})> smem_acc;
    };
  };

  struct SharedStorage {
    PipelineStorage pipelines;
    TensorStorage tensors;
    uint32_t tmem_base_ptr;
  };

  static const int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "using too much smem");

  struct MainloopArguments {
    ElementAcc softmax_scale;
    
    // all tensors strides are (num_heads or seqlen, head_dim, batch)
    // head_dim stride is always 1
    Element* ptr_q_latent;
    TensorStride stride_q_latent;
    Element* ptr_q_rope;
    TensorStride stride_q_rope;

    Element* ptr_c_latent;
    TensorStride stride_c_latent;
    Element* ptr_k_rope;
    TensorStride stride_k_rope;

    // for paged attention, we interpret what was previously [batch, seqlen]
    // as [page_count, page_size], and index according to page_table
    int* ptr_seq = nullptr;
    int* ptr_page_table = nullptr;
    // page table is [batch, seqlen or similar]
    Stride<_1, int> stride_page_table = {};
    int page_count = 0;
    int page_size = TileShapeS{};  // powers of two if kIsCpAsync, otherwise TileShapeS
  };
  
  struct EpilogueArguments {
    ElementOut* ptr_o = nullptr;
    TensorStride stride_o;
    ElementLSE* ptr_lse = nullptr;
    Stride<_1, int> stride_lse;
    ElementAcc output_scale = 1.0f;
  };

  struct Arguments {
    // (num_heads=128, seqlen, (d_latent=512, d_rope=64), batch_count)
    // for paged attention, seqlen is max seqlen
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
    KernelHardwareInfo hw_info;
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
    bool is_fused_reduction = false;
  };
  
  using TmaLoadQLatent = typename CollectiveMmaQK::Params::TMA_A;
  using TmaLoadQRope = typename CollectiveMmaQK::Params::TMA_A;
  using TmaLoadCLatent = typename CollectiveMmaQK::Params::TMA_B;
  using TmaLoadKRope = typename CollectiveMmaQK::Params::TMA_B;
  using TmaLoadCLatentTranspose = typename CollectiveMmaPV::Params::TMA_B;

  using GmemLayout = decltype(make_layout(Shape<int,int,int>{}, Stride<int64_t, _1, int64_t>{}));
  using SmemLayout = decltype(make_layout(TileShapeAcc{}, LayoutRight{}));

  using TmaReduceSum = decltype(make_tma_copy(SM90_TMA_REDUCE_ADD{}, 
		  make_tensor(recast_ptr<ElementOut>(nullptr), GmemLayout{}), SmemLayout{}));

  struct MainloopParams {
    TmaLoadQLatent tma_load_q_latent;
    TmaLoadQRope tma_load_q_rope;
    TmaLoadCLatent tma_load_c_latent;
    TmaLoadKRope tma_load_k_rope;
    TmaLoadCLatentTranspose tma_load_c_latent_transpose;
  };

  struct EpilogueParams {
    ElementOut* ptr_o = nullptr;
    ElementAcc* ptr_o_acc = nullptr;
    TensorStride stride_o;
    TensorStride stride_o_acc;
    ElementLSE* ptr_lse = nullptr;
    ElementLSE* ptr_lse_acc = nullptr;
    Stride<_1, int> stride_lse;
    Stride<_1, int> stride_lse_acc;
    ElementAcc output_scale = 1.0f;
    ElementLSE* ptr_lse_exchange_buff = nullptr;
    int* ptr_lse_max_exchange_buff = nullptr;
    int* ptr_lock = nullptr;    // semaphore
    TmaReduceSum tma_reduce_sum;		       
  };

  struct Params {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueParams epilogue;
    MainloopParams mainloop_params;
    typename TileScheduler::Params tile_scheduler;
    int split_kv = -1;
    int* ptr_split_kv = nullptr;
    bool is_fused_reduction = false;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    //workspace = nullptr;  // let's get an error if one of these needs workspace

    auto [H, K, D, B] = args.problem_shape;
    auto [L, R] = D;

    int paged_B = B;
    int paged_K = K;
    if (args.mainloop.ptr_page_table != nullptr) {
      paged_B = args.mainloop.page_count;
      paged_K = args.mainloop.page_size;
    }

    auto params_qk_latent = CollectiveMmaQK::to_underlying_arguments(
        make_shape(H, K, L, B),
        typename CollectiveMmaQK::Arguments {
          args.mainloop.ptr_q_latent, args.mainloop.stride_q_latent,
          args.mainloop.ptr_c_latent, args.mainloop.stride_c_latent,
        }, nullptr);

    auto params_qk_latent_paged = CollectiveMmaQK::to_underlying_arguments(
        make_shape(H, paged_K, L, paged_B),
        typename CollectiveMmaQK::Arguments {
          args.mainloop.ptr_q_latent, args.mainloop.stride_q_latent,
          args.mainloop.ptr_c_latent, args.mainloop.stride_c_latent,
        }, nullptr);

    auto params_qk_rope = CollectiveMmaQK::to_underlying_arguments(
        make_shape(H, K, R, B),
        typename CollectiveMmaQK::Arguments {
          args.mainloop.ptr_q_rope, args.mainloop.stride_q_rope,
          args.mainloop.ptr_k_rope, args.mainloop.stride_k_rope,
        }, nullptr);

    auto params_qk_rope_paged = CollectiveMmaQK::to_underlying_arguments(
        make_shape(H, paged_K, R, paged_B),
        typename CollectiveMmaQK::Arguments {
          args.mainloop.ptr_q_rope, args.mainloop.stride_q_rope,
          args.mainloop.ptr_k_rope, args.mainloop.stride_k_rope,
        }, nullptr);


    auto stride_c_latent_transpose = select<1,0,2>(args.mainloop.stride_c_latent);
    auto params_pv_latent = CollectiveMmaPV::to_underlying_arguments(
        make_shape(H, L, paged_K, paged_B),
        typename CollectiveMmaPV::Arguments {
          args.mainloop.ptr_q_latent, args.mainloop.stride_q_latent,  // dummy, never used
          args.mainloop.ptr_c_latent, stride_c_latent_transpose,
        }, nullptr);

    MainloopParams mainloop_params {
      params_qk_latent.tma_load_a,
      params_qk_rope.tma_load_a,
      params_qk_latent_paged.tma_load_b,
      params_qk_rope_paged.tma_load_b,
      params_pv_latent.tma_load_b
    };

    EpilogueParams epilogue_params;

    epilogue_params.ptr_o = args.epilogue.ptr_o;
    epilogue_params.stride_o = args.epilogue.stride_o;
    epilogue_params.ptr_lse = args.epilogue.ptr_lse;
    epilogue_params.stride_lse = args.epilogue.stride_lse;
    epilogue_params.output_scale = args.epilogue.output_scale;
    epilogue_params.tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, make_tensor(recast_ptr<ElementOut>(args.epilogue.ptr_o), make_layout(make_shape(H, L, B), args.epilogue.stride_o)), SmemLayout{});

    if (!args.is_fused_reduction && args.split_kv > 1) {
      ElementAcc* ptr_o_acc   = reinterpret_cast<ElementAcc*>(workspace);
      ElementLSE* ptr_lse_acc = reinterpret_cast<ElementLSE*>(ptr_o_acc + H * L * args.split_kv * B);
      epilogue_params.ptr_o_acc   = ptr_o_acc;
      epilogue_params.ptr_lse_acc = ptr_lse_acc;

      epilogue_params.stride_o_acc = make_tuple(static_cast<int64_t>(0 + L) * args.split_kv, _1{}, static_cast<int64_t>(0 + H * L) * args.split_kv);
      epilogue_params.stride_lse_acc = make_tuple(_1{}, (0 + H) * args.split_kv);
    } else if (args.is_fused_reduction && args.split_kv > 1) {
      ElementLSE* ptr_lse_exchange_buff = reinterpret_cast<ElementLSE*>(workspace);
      epilogue_params.ptr_lse_exchange_buff = ptr_lse_exchange_buff;
      int* ptr_lse_max_exchange_buff = reinterpret_cast<int*>(ptr_lse_exchange_buff + H * B);
      epilogue_params.ptr_lse_max_exchange_buff = ptr_lse_max_exchange_buff;
      int* ptr_lock = ptr_lse_max_exchange_buff + H * B;
      epilogue_params.ptr_lock = ptr_lock;
    }

    return {args.problem_shape, args.mainloop, epilogue_params, mainloop_params,
            TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, ClusterShape{}, args.split_kv), 
	    args.split_kv, args.ptr_split_kv, args.is_fused_reduction};
  }

  static size_t get_workspace_size(Arguments const& args) { 
    ProblemShape problem_shape = args.problem_shape;
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;
    auto split_kv = args.split_kv;
    size_t workspace_size {0};
    if (args.is_fused_reduction && args.split_kv > 1) {
      //  one exchange buffer for LSE max and another buffer for total LSE 
      //  two locks per batch, frist lock is for CTA0 / H=0..63 and the second is for CTA1 / H=64..127
      workspace_size = H * B * (sizeof(int) + sizeof(ElementLSE)) + 2 * B * sizeof(int);
    } else if (!args.is_fused_reduction && args.split_kv > 1) {
      workspace_size = (sizeof(ElementAcc) * D_latent + sizeof(ElementLSE)) * H * split_kv * B;
    }
    return workspace_size;
  }
  static Status initialize_workspace(
      Arguments const& args, void* ws, cudaStream_t stream) {
    auto workspace_size = get_workspace_size(args);
    if (args.is_fused_reduction && args.split_kv > 1) {
      auto result = cudaMemsetAsync(ws, 0, workspace_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
        "  cudaMemsetAsync() returned error: "
        << cudaGetErrorString(result));
        return Status::kErrorInternal;;
      }
    }
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static bool can_implement(Arguments const& args) {
    if (kIsCpAsync) {
      if ((args.mainloop.page_size & (args.mainloop.page_size - 1)) != 0) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): cpasync page size pow2\n";
        return false;
      }
      if (args.mainloop.page_size > TileShapeS{}) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): cpasync page size too big\n";
        return false;
      }
    }
    else {
      if (args.mainloop.ptr_page_table != nullptr && args.mainloop.page_size != TileShapeS{}) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): tma page size off\n";
        return false;
      }
    }
    if (get<0>(args.problem_shape) != 128) {
      std::cerr << __FILE__ << "(" << __LINE__ << "): heads off\n";
      return false;
    }
    if (get<1>(args.problem_shape) <= 0) {
      std::cerr << __FILE__ << "(" << __LINE__ << "): heads off\n";
      return false;
    }
    if (args.split_kv <= 0) {
      std::cerr << __FILE__ << "(" << __LINE__ << "): split-k off\n";
      return false;
    }
    if (args.is_fused_reduction && args.split_kv > 1) {
      if (2 * args.split_kv > args.hw_info.sm_count || 
      std::is_same_v<TileScheduler, Sm100MlaIndividualTileScheduler>) {
        return false;
      }
    }
    return true;
  }


  CUTLASS_DEVICE void operator()(Params const& params, char* smem_raw) {

    TileScheduler tile_scheduler(params.tile_scheduler);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_role(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    int cta_coord_v = cta_rank_in_cluster % size<0>(AtomThrShapeMNK{});
    bool is_mma_leader_cta = cta_coord_v == 0;

    if (role == WarpRole::kLoad && lane_predicate && ! kIsCpAsync) {
      prefetch_tma_descriptor(params.mainloop_params.tma_load_q_latent.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_c_latent.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_q_rope.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_k_rope.get_tma_descriptor());
      prefetch_tma_descriptor(params.mainloop_params.tma_load_c_latent_transpose.get_tma_descriptor());
    }
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_raw);

    typename PipelineLoadQK::Params pipeline_load_qk_params;
    if (role == WarpRole::kLoad) {
      pipeline_load_qk_params.role = PipelineLoadQK::ThreadCategory::Producer;
    }
    if (role == WarpRole::kMma) {
      pipeline_load_qk_params.role = PipelineLoadQK::ThreadCategory::Consumer;
    }
    if constexpr (kIsCpAsync) {
      // we can make our life easier by unconditionally loading blocks
      // since we know it'll always be legal
      pipeline_load_qk_params.producer_arv_count = kNumLoadWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    }
    else {
      pipeline_load_qk_params.is_leader = lane_predicate && (role == WarpRole::kLoad) && is_mma_leader_cta;
      pipeline_load_qk_params.transaction_bytes = kTransactionsBytesLoadQK;
    }
    pipeline_load_qk_params.initializing_warp = 0;
    PipelineLoadQK pipeline_load_qk(shared_storage.pipelines.load_qk, pipeline_load_qk_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineS::Params pipeline_mma_s_params;
    if (role == WarpRole::kMma) {
      pipeline_mma_s_params.role = PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_mma_s_params.role = PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_mma_s_params.initializing_warp = 1;
    PipelineS pipeline_mma_s(
      shared_storage.pipelines.mma_s,
      pipeline_mma_s_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineP::Params pipeline_p_mma_params;
    if (role == WarpRole::kMma) {
      pipeline_p_mma_params.role = PipelineP::ThreadCategory::Consumer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_p_mma_params.role = PipelineP::ThreadCategory::Producer;
    }
    pipeline_p_mma_params.producer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_p_mma_params.consumer_arv_count = 1;
    pipeline_p_mma_params.initializing_warp = 2;
    PipelineP pipeline_p_mma(
      shared_storage.pipelines.p_mma,
      pipeline_p_mma_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelineO::Params pipeline_mma_o_params;
    if (role == WarpRole::kMma) {
      pipeline_mma_o_params.role = PipelineO::ThreadCategory::Producer;
    }
    if (role == WarpRole::kCompute) {
      pipeline_mma_o_params.role = PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_o_params.consumer_arv_count = kNumComputeWarps * cutlass::NumThreadsPerWarp * size(AtomThrShapeMNK{});
    pipeline_mma_o_params.initializing_warp = 3;
    PipelineO pipeline_mma_o(
      shared_storage.pipelines.mma_o,
      pipeline_mma_o_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename PipelinePT::Params pipeline_pt_params;
    if (role == WarpRole::kLoad) {
      pipeline_pt_params.role = PipelinePT::ThreadCategory::Consumer;
    }
    if (role == WarpRole::kLoadPageTable) {
      pipeline_pt_params.role = PipelinePT::ThreadCategory::Producer;
    }
    pipeline_pt_params.consumer_arv_count = kNumLoadWarps * cutlass::NumThreadsPerWarp;
    pipeline_pt_params.producer_arv_count = cutlass::NumThreadsPerWarp;
    pipeline_pt_params.initializing_warp = 4;
    PipelinePT pipeline_page_table(
      shared_storage.pipelines.load_page_table,
      pipeline_pt_params);

    TmemAllocator tmem_allocator;

    pipeline_init_arrive_relaxed(size(ClusterShape{}));

    pipeline_load_qk.init_masks(ClusterShape{});  // do we need an update here for 2Sm?
    pipeline_mma_s.init_masks(ClusterShape{});
    pipeline_p_mma.init_masks(ClusterShape{});
    pipeline_mma_o.init_masks(ClusterShape{});

    typename PipelineLoadQK::PipelineState pipeline_load_qk_consumer_state;
    typename PipelineLoadQK::PipelineState pipeline_load_qk_producer_state = cutlass::make_producer_start_state<PipelineLoadQK>();

    typename PipelineS::PipelineState pipeline_mma_s_consumer_state;
    typename PipelineS::PipelineState pipeline_mma_s_producer_state = cutlass::make_producer_start_state<PipelineS>();

    typename PipelineP::PipelineState pipeline_p_mma_consumer_state;
    typename PipelineP::PipelineState pipeline_p_mma_producer_state = cutlass::make_producer_start_state<PipelineP>();

    typename PipelineO::PipelineState pipeline_mma_o_consumer_state;
    typename PipelineO::PipelineState pipeline_mma_o_producer_state = cutlass::make_producer_start_state<PipelineO>();

    typename PipelinePT::PipelineState pipeline_pt_consumer_state;
    typename PipelinePT::PipelineState pipeline_pt_producer_state = cutlass::make_producer_start_state<PipelinePT>();

    pipeline_init_wait(size(ClusterShape{}));

    if (role == WarpRole::kLoadPageTable) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();
        auto problem_shape = params.problem_shape;
	auto local_split_kv = params.split_kv;
        if (params.mainloop.ptr_seq != nullptr) {
          get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
	  if (params.ptr_split_kv != nullptr) {
            local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
          }
        }
	if (local_split_kv <= get<3>(blk_coord))
	  continue;
        load_page_table(
          blk_coord,
          problem_shape,
          params.mainloop,
          shared_storage.tensors,
          pipeline_page_table, pipeline_pt_producer_state,
	  local_split_kv
        );
      }
    }
    else if (role == WarpRole::kLoad) {
      if constexpr (kIsCpAsync) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
	  auto problem_shape = params.problem_shape;
	  auto local_split_kv = params.split_kv;
          if (params.mainloop.ptr_seq != nullptr) {
            get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
	    if (params.ptr_split_kv != nullptr) {
              local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
            }
          }
	  if (local_split_kv <= get<3>(blk_coord))
            continue;
          load_cpasync(
            blk_coord,
            problem_shape,
            params.mainloop,
            params.mainloop_params,
            shared_storage.tensors,
            pipeline_load_qk, pipeline_load_qk_producer_state,
	    local_split_kv,
            /* must be shared pipe */
            pipeline_page_table, pipeline_pt_consumer_state
          );
          cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, kNamedBarrierEpilogue).arrive_and_wait();
        }
      }
      else {
        if (params.mainloop.ptr_page_table != nullptr) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; tile_scheduler.is_valid(); ++tile_scheduler) {
            auto blk_coord = tile_scheduler.get_block_coord();
	    auto problem_shape = params.problem_shape;
	    auto local_split_kv = params.split_kv;
            if (params.mainloop.ptr_seq != nullptr) {
              get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
	      if (params.ptr_split_kv != nullptr) {
	        local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
	      }
            }
	    if (local_split_kv <= get<3>(blk_coord))
              continue;
            load_tma</* paged= */ true>(
              blk_coord,
              problem_shape,
              params.mainloop,
              params.mainloop_params,
              shared_storage.tensors,
              pipeline_load_qk, pipeline_load_qk_producer_state,
              pipeline_load_qk, pipeline_load_qk_producer_state,
	      local_split_kv
            );
            cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, kNamedBarrierEpilogue).arrive_and_wait();
          }
        }
        else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; tile_scheduler.is_valid(); ++tile_scheduler) {
            auto blk_coord = tile_scheduler.get_block_coord();
	    auto problem_shape = params.problem_shape;
	    auto local_split_kv = params.split_kv;
            if (params.mainloop.ptr_seq != nullptr) {
              get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
	      if (params.ptr_split_kv != nullptr) {
                local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
	      }
            }
	    if (local_split_kv <= get<3>(blk_coord))
              continue;
            load_tma<false>(
              blk_coord,
              problem_shape,
              params.mainloop,
              params.mainloop_params,
              shared_storage.tensors,
              pipeline_load_qk, pipeline_load_qk_producer_state,
              pipeline_load_qk, pipeline_load_qk_producer_state,
	      local_split_kv
            );
            cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, kNamedBarrierEpilogue).arrive_and_wait();
          }
        }
      }
    }
    else if (role == WarpRole::kMma) {
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
    
      if (is_mma_leader_cta) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          auto problem_shape = params.problem_shape;
	  auto local_split_kv = params.split_kv;
          if (params.mainloop.ptr_seq != nullptr) {
            get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
            if (params.ptr_split_kv != nullptr) {
                local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
            }
          }
	  if (local_split_kv <= get<3>(blk_coord))
            continue;
          mma(blk_coord,
            problem_shape,
            shared_storage.tensors,
            pipeline_load_qk, pipeline_load_qk_consumer_state,
            pipeline_load_qk, pipeline_load_qk_consumer_state,
            pipeline_mma_s, pipeline_mma_s_producer_state,
            pipeline_p_mma, pipeline_p_mma_consumer_state,
            pipeline_mma_o, pipeline_mma_o_producer_state,
	    local_split_kv
          );
        }
      }

      //cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp, kNamedBarrierTmemDealloc).arrive_and_wait();

      //uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
      //tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
    else if (role == WarpRole::kCompute) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();
        auto problem_shape = params.problem_shape;
	auto split_kv = params.split_kv;
	auto local_split_kv = split_kv;
        if (params.mainloop.ptr_seq != nullptr) {
          get<1>(problem_shape) = params.mainloop.ptr_seq[get<2>(blk_coord)];
	  if (params.ptr_split_kv != nullptr) {
            local_split_kv = params.ptr_split_kv[get<2>(blk_coord)];
          }
        }
	if (local_split_kv <= get<3>(blk_coord))
          continue;
        compute(
          blk_coord,
          problem_shape,
          params.mainloop,         // for softmax_scale
          params.epilogue,
          shared_storage.tensors,  // for smem_comm
          pipeline_mma_s, pipeline_mma_s_consumer_state,
          pipeline_p_mma, pipeline_p_mma_producer_state,
          pipeline_mma_o, pipeline_mma_o_consumer_state,
	  local_split_kv,
	  params.is_fused_reduction
        );
      }

      //cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp, kNamedBarrierTmemDealloc).arrive();
    }

    cute::cluster_sync();
    cutlass::arch::NamedBarrier((kNumComputeWarps + 1) * NumThreadsPerWarp, kNamedBarrierTmemDealloc).arrive();
    if (role == WarpRole::kMma) {
      uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
      tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
  }

  template<class BlkCoord>
  CUTLASS_DEVICE void load_page_table(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      TensorStorage& shared_tensors,
      PipelinePT& pipeline_page_table,
      typename PipelinePT::PipelineState& pipeline_pt_producer_state, int const& split_kv) {

    auto [H, K, D, B] = problem_shape;
    int batch_coord = get<2>(blk_coord);

    auto mPT_l = make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table), 
                            make_shape(mainloop_args.page_count, B), 
                            mainloop_args.stride_page_table);
    auto mPT = mPT_l(_, batch_coord);
    
    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta; // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    auto page_size = Pow2{mainloop_args.page_size};
    auto pages_per_tile = Pow2{TileShapeS{} / page_size};
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;

    for (; k_tile_count > 0; ++k_index, --k_tile_count) {
      pipeline_page_table.producer_acquire(pipeline_pt_producer_state);
    
      // assume a single warp

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < TileShapeS{}; i += cutlass::NumThreadsPerWarp) {
        int idx = i + thread_idx;
        bool guard = idx < pages_per_tile;
        int smem_idx = pipeline_pt_producer_state.index() * TileShapeS::value + idx;
        int pt_idx = pages_per_tile * k_index + idx;

        cutlass::arch::cp_async_zfill<sizeof(int), cutlass::arch::CacheOperation::Always>(
          &shared_tensors.smem_page_table[smem_idx], &mPT(pt_idx), guard
        );
      }
      
      pipeline_page_table.producer_commit(pipeline_pt_producer_state, cutlass::arch::cpasync_barrier_arrive);
      ++pipeline_pt_producer_state;
    }
  }


  struct Gather {
    int& page_table_stage;
    Pow2 pages_per_tile;
    const int * __restrict__ smem_page_table;

    CUTLASS_DEVICE int operator()(int idx) const {
      return smem_page_table[page_table_stage * TileShapeS::value + idx % pages_per_tile];
    }

    CUTLASS_DEVICE friend void print(Gather const&) {
      printf("<gather>");
    }

  };


  template<class BlkCoord>
  CUTLASS_DEVICE void load_cpasync(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineLoadQK& pipeline_load,
      typename PipelineLoadQK::PipelineState& pipeline_load_producer_state,
      int const& split_kv,
      PipelinePT& pipeline_page_table,
      typename PipelinePT::PipelineState& pipeline_pt_consumer_state) {

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    using X = Underscore;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta; // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    // partition all tensors
    auto mQL = make_tensor(make_gmem_ptr(mainloop_args.ptr_q_latent), make_shape(H, D_latent, B), mainloop_args.stride_q_latent);
    auto mQR = make_tensor(make_gmem_ptr(mainloop_args.ptr_q_rope), make_shape(H, D_rope, B), mainloop_args.stride_q_rope);

    int  paged_B = mainloop_args.page_count;
    auto paged_K = Pow2{mainloop_args.page_size};
    auto mPT_l = make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table), make_shape(paged_B, B), mainloop_args.stride_page_table);

    int batch_coord = get<2>(blk_coord);
    auto mPT = mPT_l(_, batch_coord);

    auto gQL = local_tile(mQL, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gQR = local_tile(mQR, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});

    ThrMMA cta_mma_qk = TiledMmaQK{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));
    ThrMMA cta_mma_pv = TiledMmaPV{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));

    auto tSgQL = cta_mma_qk.partition_A(gQL);
    auto tSgQR = cta_mma_qk.partition_A(gQR);

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});

    auto make_copy_for = [](auto sT) {
      auto rT_a = sT.layout()(_, _, _, _0{});
      auto rT = make_ordered_layout(shape(rT_a), stride(rT_a));
      auto threads = Int<kNumLoadWarps * cutlass::NumThreadsPerWarp>{};
      auto values = Int<sizeof(uint128_t) / sizeof(Element)>{};
      return make_cotiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
          make_ordered_layout(
              make_shape(threads, values),
              make_stride(_1{}, _0{})),
          rT);
    };

    // like cute::copy, but makes sure we do all page table lookups first
    auto copy_split = [](auto atom, auto src, auto dst) {
      auto src_v = group_modes<1, rank_v<decltype(src)>>(src);
      auto dst_v = group_modes<1, rank_v<decltype(dst)>>(dst);

      auto src_v_ptrs = make_tensor<Element*>(size<1>(src_v));
      for (int i = 0; i < size<1>(src_v); i++) {
        src_v_ptrs(i) = &src_v(_0{}, i);
      }


      for (int i = 0; i < size<1>(src_v); i++) {
        auto src_v_i = make_tensor(
            make_gmem_ptr(src_v_ptrs(i)),
            make_shape(shape<0>(src_v)),
            make_stride(make_stride(_1{}, _0{}))
        );
        atom.call(src_v_i, dst_v(_, i));
      }
    };

    auto tiled_copy_q = make_copy_for(sQ);
    auto tiled_copy_kc = make_copy_for(sKC);
    auto tiled_copy_vc = make_copy_for(sVC);

    auto thr_copy_q = tiled_copy_q.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));
    auto thr_copy_kc = tiled_copy_kc.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));
    auto thr_copy_vc = tiled_copy_vc.get_thread_slice(threadIdx.x % (kNumLoadWarps * cutlass::NumThreadsPerWarp));

    auto tQsQ = thr_copy_q.partition_D(sQ);
    auto tQgQL = thr_copy_q.partition_S(tSgQL);
    auto tQgQR = thr_copy_q.partition_S(tSgQR);

    auto tKCsKC = thr_copy_kc.partition_D(sKC);
    auto tVCsVC = thr_copy_vc.partition_D(sVC);

    auto pipeline_pt_release_state = pipeline_pt_consumer_state;

    int page_table_stage = -1;
    Pow2 pages_per_tile{TileShapeS{} / paged_K};
    const int * __restrict__ smem_page_table = shared_tensors.smem_page_table.begin();
    Gather gather{page_table_stage, pages_per_tile, smem_page_table};

    auto mCL = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_c_latent),
        ComposedLayout{
            make_layout(
                make_shape(make_shape(paged_K, paged_B), _1{}),
                make_stride(make_stride(get<0>(mainloop_args.stride_c_latent), example::CustomStride(gather, get<2>(mainloop_args.stride_c_latent))), get<1>(mainloop_args.stride_c_latent))),
            make_coord(_0{}, _0{}),
            make_identity_layout(make_shape(paged_K * paged_B, D_latent))});

    auto mKR = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_k_rope),
        ComposedLayout{
            make_layout(
                make_shape(make_shape(paged_K, paged_B), _1{}),
                make_stride(make_stride(get<0>(mainloop_args.stride_k_rope), example::CustomStride(gather, get<2>(mainloop_args.stride_k_rope))), get<1>(mainloop_args.stride_k_rope))),
            make_coord(_0{}, _0{}),
            make_identity_layout(make_shape(paged_K * paged_B, D_latent))});

    auto mCLT = make_tensor(
        make_gmem_ptr(mainloop_args.ptr_c_latent),
        ComposedLayout{
            make_layout(
                make_shape(_1{}, make_shape(paged_K, paged_B)),
                make_stride(get<1>(mainloop_args.stride_c_latent), make_stride(get<0>(mainloop_args.stride_c_latent), example::CustomStride(gather, get<2>(mainloop_args.stride_c_latent))))),
            make_coord(_0{}, _0{}),
            make_identity_layout(make_shape(D_latent, paged_K * paged_B))});

    auto gCL = local_tile(mCL, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gKR = local_tile(mKR, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gCLT = local_tile(mCLT, TileShapePV{}, make_coord(_,_,_), Step<X, _1, _1>{});

    auto tSgCL = cta_mma_qk.partition_B(gCL);
    auto tSgKR = cta_mma_qk.partition_B(gKR);
    auto tOgCLT = cta_mma_pv.partition_B(gCLT);

    auto tKCgCL = thr_copy_kc.partition_S(tSgCL);
    auto tKCgKR = thr_copy_kc.partition_S(tSgKR);
    auto tVCgCLT = thr_copy_vc.partition_S(tOgCLT);

    // latent is first in memory, so let's load it first always
    // startup: alternate Q and K, set tx count appropriately, for k_idx = 0
    auto& pipeline_acquire_state = pipeline_load_producer_state;
    auto pipeline_commit_state = pipeline_acquire_state;
    int pipeline_offset = 0;

    for (int i = 0; i < StagesPV; i++) {
      cutlass::arch::cp_async_fence();
    }

    auto load_stage = [&](auto fn) {
      pipeline_load.producer_acquire(pipeline_acquire_state);
      fn(pipeline_acquire_state.index());
      cutlass::arch::cp_async_fence();

      ++pipeline_acquire_state;
      ++pipeline_offset;

      if (pipeline_offset == StagesPV - 1) {
        cutlass::arch::cp_async_wait<StagesPV - 1>();
        pipeline_load.producer_commit(pipeline_commit_state);
        ++pipeline_commit_state;
        --pipeline_offset;
      }
    };

    pipeline_page_table.consumer_wait(pipeline_pt_consumer_state);
    page_table_stage = pipeline_pt_consumer_state.index();
    ++pipeline_pt_consumer_state;

    // each Q/K tile consists of rope and latent
    for (int i = 0; i < IterationsQKLatent; i++) {
      load_stage([&](int index) {
        cute::copy(tiled_copy_q, tQgQL(_, _, _, _, _0{}, i, batch_coord),  tQsQ(_, _, _, _, i));
        copy_split(tiled_copy_kc, tKCgCL(_, _, _, _, k_index, i),  tKCsKC(_, _, _, _, index));
      });
    }

    for (int i = 0; i < IterationsQKRope; i++) {
      load_stage([&](int index) {
        cute::copy(tiled_copy_q, tQgQR(_, _, _, _, _0{}, i, batch_coord),  tQsQ(_, _, _, _, IterationsQKLatent + i));
        copy_split(tiled_copy_kc, tKCgKR(_, _, _, _, k_index, i),  tKCsKC(_, _, _, _, index));
      });
    }

    k_index += 1;
    k_tile_count -= 1;

    // assume k_tile_count >= 1
    // perform K+Q load here
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      pipeline_page_table.consumer_wait(pipeline_pt_consumer_state);
      page_table_stage = pipeline_pt_consumer_state.index();
      ++pipeline_pt_consumer_state;

      for (int i = 0; i < IterationsQKLatent; i++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_kc, tKCgCL(_, _, _, _, k_index, i),  tKCsKC(_, _, _, _, index));
        });
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_kc, tKCgKR(_, _, _, _, k_index, i),  tKCsKC(_, _, _, _, index));
        });
      }

      page_table_stage = pipeline_pt_release_state.index();

      for (int i = 0; i < IterationsPV_K; i++) {
        for (int j = 0; j < IterationsPV_N; j++) {
          load_stage([&](int index) {
            copy_split(tiled_copy_vc, tVCgCLT(_, _, _, _, j, IterationsPV_K * (k_index - 1) + i),  tVCsVC(_, _, _, _, index));
          });
        }
      }

      pipeline_page_table.consumer_release(pipeline_pt_release_state);
      ++pipeline_pt_release_state;

      k_index += 1;
      k_tile_count -= 1;
    }

    page_table_stage = pipeline_pt_release_state.index();

    for (int i = 0; i < IterationsPV_K; i++) {
      for (int j = 0; j < IterationsPV_N; j++) {
        load_stage([&](int index) {
          copy_split(tiled_copy_vc, tVCgCLT(_, _, _, _, j, IterationsPV_K * (k_index - 1) + i),  tVCsVC(_, _, _, _, index));
        });
      }
    }

    pipeline_page_table.consumer_release(pipeline_pt_release_state);
    ++pipeline_pt_release_state;

    while (pipeline_offset > 0) {
      cutlass::arch::cp_async_fence();
    
      cutlass::arch::cp_async_wait<StagesPV - 1>();
      pipeline_load.producer_commit(pipeline_commit_state);
      ++pipeline_commit_state;
      --pipeline_offset;
    }

    cutlass::arch::cp_async_wait<0>();

  }


  template<bool kIsPaged = false, class BlkCoord>
  CUTLASS_DEVICE void load_tma(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors,
      PipelineLoadQK& pipeline_load_qk,
      typename PipelineLoadQK::PipelineState& pipeline_load_qk_producer_state,
      PipelineLoadPV& pipeline_load_pv,
      typename PipelineLoadPV::PipelineState& pipeline_load_pv_producer_state,
      int const& split_kv) {

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta; // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    using X = Underscore;
  
    // partition all tensors
    auto mQL = mainloop_params.tma_load_q_latent.get_tma_tensor(make_shape(H, D_latent, B));
    auto mQR = mainloop_params.tma_load_q_rope.get_tma_tensor(make_shape(H, D_rope, B));

    int paged_B = B;
    int paged_K = K;
    if constexpr (kIsPaged) {
      paged_B = mainloop_args.page_count;
      paged_K = mainloop_args.page_size;
    }
    auto mPT_l = make_tensor(make_gmem_ptr(mainloop_args.ptr_page_table), make_shape(paged_B, B), mainloop_args.stride_page_table);

    auto mCL = mainloop_params.tma_load_c_latent.get_tma_tensor(make_shape(paged_K, D_latent, paged_B));
    auto mKR = mainloop_params.tma_load_k_rope.get_tma_tensor(make_shape(paged_K, D_rope, paged_B));

    auto mCLT = mainloop_params.tma_load_c_latent_transpose.get_tma_tensor(make_shape(D_latent, paged_K, paged_B));

    auto gQL = local_tile(mQL, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});
    auto gQR = local_tile(mQR, TileShapeQK{}, make_coord(_,_,_), Step<_1, X, _1>{});

    auto gCL = local_tile(mCL, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gKR = local_tile(mKR, TileShapeQK{}, make_coord(_,_,_), Step<X, _1, _1>{});
    auto gCLT = local_tile(mCLT, TileShapePV{}, make_coord(_,_,_), Step<X, _1, _1>{});

    ThrMMA cta_mma_qk = TiledMmaQK{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));
    ThrMMA cta_mma_pv = TiledMmaPV{}.get_slice(get<0>(blk_coord) % size(AtomThrShapeMNK{}));

    auto tSgQL = cta_mma_qk.partition_A(gQL);
    auto tSgQR = cta_mma_qk.partition_A(gQR);

    auto tSgCL = cta_mma_qk.partition_B(gCL);
    auto tSgKR = cta_mma_qk.partition_B(gKR);

    auto tOgCLT = cta_mma_pv.partition_B(gCLT);

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});

    auto [tQLgQL_mkl, tQsQ] = tma_partition(
        mainloop_params.tma_load_q_latent, _0{}, make_layout(_1{}),
        group_modes<0,3>(sQ), group_modes<0,3>(tSgQL));

    auto [tQRgQR_mkl, tQsQ_ignore] = tma_partition(
        mainloop_params.tma_load_q_rope, _0{}, make_layout(_1{}),
        group_modes<0,3>(sQ), group_modes<0,3>(tSgQR));

    auto [tCLgCL_nkl, tKCsKC] = tma_partition(
        mainloop_params.tma_load_c_latent, _0{}, make_layout(_1{}),
        group_modes<0,3>(sKC), group_modes<0,3>(tSgCL));

    auto [tKRgKR_nkl, tKCsKC_ignore] = tma_partition(
        mainloop_params.tma_load_k_rope, _0{}, make_layout(_1{}),
        group_modes<0,3>(sKC), group_modes<0,3>(tSgKR));

    auto [tCLTgCLT_nkl, tVCsVC] = tma_partition(
        mainloop_params.tma_load_c_latent_transpose, _0{}, make_layout(_1{}),
        group_modes<0,3>(sVC), group_modes<0,3>(tOgCLT));

    uint16_t mcast_mask = 0;

    int batch_coord = get<2>(blk_coord);
    Tensor tQLgQL = tQLgQL_mkl(_, _, _, batch_coord);
    Tensor tQRgQR = tQRgQR_mkl(_, _, _, batch_coord);

    auto mPT = mPT_l(_, batch_coord);

    Tensor tCLgCL = tCLgCL_nkl(_, _, _, _);
    Tensor tKRgKR = tKRgKR_nkl(_, _, _, _);

    // careful: stage and k are swapped here!
    Tensor tCLTgCLT = tCLTgCLT_nkl(_, _, _, _);

    // latent is first in memory, so let's load it first always
    // startup: alternate Q and K, set tx count appropriately, for k_idx = 0

    // each Q/K tile consists of rope and latent
    for (int i = 0; i < IterationsQKLatent; i++) {
      pipeline_load_qk.producer_expect_transaction(pipeline_load_qk_producer_state, kTransactionsBytesLoadExtraQ);
      pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
      auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

      if (cute::elect_one_sync()) {
        // expect the extra bytes
        // load_qk ql
        cute::copy(mainloop_params.tma_load_q_latent.with(*tma_barrier, mcast_mask), tQLgQL(_, _0{}, i), tQsQ(_, i));
        // load_qk cl
        if constexpr (kIsPaged) {
          cute::copy(
              mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
              tCLgCL(_, _0{}, i, mPT(k_index)),
              tKCsKC(_, pipeline_load_qk_producer_state.index())
          );
        }
        else {
          cute::copy(
              mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
              tCLgCL(_, k_index, i, batch_coord),
              tKCsKC(_, pipeline_load_qk_producer_state.index()));
        }
      }
      ++pipeline_load_qk_producer_state;
    }

    for (int i = 0; i < IterationsQKRope; i++) {
      pipeline_load_qk.producer_expect_transaction(pipeline_load_qk_producer_state, kTransactionsBytesLoadExtraQ);
      pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
      auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

      if (cute::elect_one_sync()) {
        // expect the extra bytes
        // load_qk ql
        cute::copy(mainloop_params.tma_load_q_rope.with(*tma_barrier, mcast_mask), tQRgQR(_, _0{}, i), tQsQ(_, i + IterationsQKLatent));
        // load_qk cl
        if constexpr (kIsPaged) {
          cute::copy(
              mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
              tKRgKR(_, _0{}, i, mPT(k_index)),
              tKCsKC(_, pipeline_load_qk_producer_state.index())
          );
        }
        else {
          cute::copy(
              mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
              tKRgKR(_, k_index, i, batch_coord),
              tKCsKC(_, pipeline_load_qk_producer_state.index()));
        }
      }
      ++pipeline_load_qk_producer_state;
    }

    k_index += 1;
    k_tile_count -= 1;

    // assume k_tile_count >= 1
    // perform K+Q load here
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      // perform K load
      for (int i = 0; i < IterationsQKLatent; i++) {
        pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
        auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

        if (cute::elect_one_sync()) {
          // load_qk cl
          if constexpr (kIsPaged) {
            cute::copy(
                mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                tCLgCL(_, _0{}, i, mPT(k_index)),
                tKCsKC(_, pipeline_load_qk_producer_state.index())
            );
          }
          else {
            cute::copy(
                mainloop_params.tma_load_c_latent.with(*tma_barrier, mcast_mask),
                tCLgCL(_, k_index, i, batch_coord),
                tKCsKC(_, pipeline_load_qk_producer_state.index()));
          }
        }
        ++pipeline_load_qk_producer_state;
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        pipeline_load_qk.producer_acquire(pipeline_load_qk_producer_state);
        auto tma_barrier = pipeline_load_qk.producer_get_barrier(pipeline_load_qk_producer_state);

        if (cute::elect_one_sync()) {
          // load_qk cl
          if constexpr (kIsPaged) {
            cute::copy(
                mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                tKRgKR(_, _0{}, i, mPT(k_index)),
                tKCsKC(_, pipeline_load_qk_producer_state.index())
            );
          }
          else {
            cute::copy(
                mainloop_params.tma_load_k_rope.with(*tma_barrier, mcast_mask),
                tKRgKR(_, k_index, i, batch_coord),
                tKCsKC(_, pipeline_load_qk_producer_state.index()));
          }
        }
        ++pipeline_load_qk_producer_state;
      }

      // prefetch next K load to keep busy while we transpose-load from cache
      const int kPrefetchDistance = 1;
      for (int i = 0; i < IterationsQKLatent; i++) {
        if (cute::elect_one_sync()) {
          if constexpr (kIsPaged) {
            if (k_tile_count > kPrefetchDistance) {
              cute::prefetch(
                  mainloop_params.tma_load_c_latent,
                  tCLgCL(_, _0{}, i, mPT(k_index + kPrefetchDistance))
              );
            }
          }
          else {
            cute::prefetch(
                mainloop_params.tma_load_c_latent,
                tCLgCL(_, k_index + kPrefetchDistance, i, batch_coord)
            );
          }
        }
      }

      for (int i = 0; i < IterationsQKRope; i++) {
        if (cute::elect_one_sync()) {
          if constexpr (kIsPaged) {
            if (k_tile_count > kPrefetchDistance) {
              cute::prefetch(
                  mainloop_params.tma_load_k_rope,
                  tKRgKR(_, _0{}, i, mPT(k_index + kPrefetchDistance))
              );
            }
          }
          else {
            cute::prefetch(
                mainloop_params.tma_load_k_rope,
                tKRgKR(_, k_index + kPrefetchDistance, i, batch_coord)
            );
          }
        }
      }

      // perform V load (k_idx - 1)

      for (int i = 0; i < IterationsPV_K; i++) {
        for (int j = 0; j < IterationsPV_N; j++) {
          pipeline_load_pv.producer_acquire(pipeline_load_pv_producer_state);
          auto tma_barrier = pipeline_load_pv.producer_get_barrier(pipeline_load_pv_producer_state);

          if (cute::elect_one_sync()) {
            // load_pv cl
            // note the transpose in indices!
            // note we are off-by-one on k_index
            if constexpr (kIsPaged) {
              cute::copy(
                  mainloop_params.tma_load_c_latent_transpose.with(*tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                  tCLTgCLT(_, j, i, mPT(k_index - 1)),
                  tVCsVC(_, pipeline_load_pv_producer_state.index())
              );
            }
            else {
              cute::copy(
                  mainloop_params.tma_load_c_latent_transpose.with(*tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                  tCLTgCLT(_, j, IterationsPV_K * (k_index - 1) + i, batch_coord),
                  tVCsVC(_, pipeline_load_pv_producer_state.index())
              );
            }
          }
          ++pipeline_load_pv_producer_state;
        }
      }

      k_index += 1;
      k_tile_count -= 1;
    }

    for (int i = 0; i < IterationsPV_K; i++) {
      for (int j = 0; j < IterationsPV_N; j++) {
        pipeline_load_pv.producer_acquire(pipeline_load_pv_producer_state);
        auto tma_barrier = pipeline_load_pv.producer_get_barrier(pipeline_load_pv_producer_state);

        if (cute::elect_one_sync()) {
          // load_pv cl
          // note the transpose in indices
          // note we are off-by-one on k_index

          if constexpr (kIsPaged) {
            cute::copy(
                mainloop_params.tma_load_c_latent_transpose.with(*tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                tCLTgCLT(_, j, i, mPT(k_index - 1)),
                tVCsVC(_, pipeline_load_pv_producer_state.index())
            );
          }
          else {
            cute::copy(
                mainloop_params.tma_load_c_latent_transpose.with(*tma_barrier, mcast_mask, cute::TMA::CacheHintSm100::EVICT_FIRST),
                tCLTgCLT(_, j, IterationsPV_K * (k_index - 1) + i, batch_coord),
                tVCsVC(_, pipeline_load_pv_producer_state.index())
            );
          }
        }
        ++pipeline_load_pv_producer_state;
      }
    }
  }

  template<class BlkCoord>
  CUTLASS_DEVICE void mma(
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      TensorStorage& shared_tensors,
      PipelineLoadQK& pipeline_load_qk,
      typename PipelineLoadQK::PipelineState& pipeline_load_qk_consumer_state,
      PipelineLoadPV& pipeline_load_pv,
      typename PipelineLoadPV::PipelineState& pipeline_load_pv_consumer_state,
      PipelineS& pipeline_mma_s,
      typename PipelineS::PipelineState& pipeline_mma_s_producer_state,
      PipelineP& pipeline_p_mma,
      typename PipelineP::PipelineState& pipeline_p_mma_consumer_state,
      PipelineO& pipeline_mma_o,
      typename PipelineO::PipelineState& pipeline_mma_o_producer_state,
      int const& split_kv) {

    auto [H, K, D, B] = problem_shape;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(blk_coord) * k_tile_per_cta; // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {
      return;
    }

    // mma init
    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_q.begin()), SmemLayoutQ{});
    Tensor sKC = make_tensor(make_smem_ptr(shared_tensors.smem_kc.begin()), SmemLayoutKC{});
    Tensor sVC = make_tensor(make_smem_ptr(shared_tensors.smem_vc.begin()), SmemLayoutVC{});
    Tensor sP = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_p.begin()), SmemLayoutP{});

    Tensor tSrQ = TiledMmaQK::make_fragment_A(sQ);
    Tensor tSrKC = TiledMmaQK::make_fragment_B(sKC);
    Tensor tOrP = TiledMmaPV::make_fragment_A(sP);
    Tensor tOrVC = TiledMmaPV::make_fragment_B(sVC);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    Tensor tStS =  partition_fragment_C(tiled_mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO =  partition_fragment_C(tiled_mma_pv, select<0,1>(TileShapePV{}));

    tiled_mma_pv.accumulate_ = UMMA::ScaleOut::Zero;

    pipeline_mma_s.producer_acquire(pipeline_mma_s_producer_state);

    // Mma           S0 S1 O0 S2 O1 ... Sn On-1 On
    // S0 ownership  --    -----        --      --
    // S1 ownership     --       -----     ----
    // O ownership         --    --        ---- --

    tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
    for (int i = 0; i < IterationsQK; i++) {
      pipeline_load_qk.consumer_wait(pipeline_load_qk_consumer_state);
      int read_stage = pipeline_load_qk_consumer_state.index();

      tStS.data() = uint32_t(pipeline_mma_s_producer_state.index() == 0 ? TmemAllocation::kS0 : TmemAllocation::kS1);

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
        cute::gemm(tiled_mma_qk,
                   tSrQ(_,_,k_block,i),
                   tSrKC(_,_,k_block,read_stage),
                   tStS);
        tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
      }

      pipeline_load_qk.consumer_release(pipeline_load_qk_consumer_state);
      ++pipeline_load_qk_consumer_state;
    }

    pipeline_mma_s.producer_commit(pipeline_mma_s_producer_state);
    ++pipeline_mma_s_producer_state;

    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      pipeline_mma_s.producer_acquire(pipeline_mma_s_producer_state);
      tiled_mma_qk.accumulate_ = UMMA::ScaleOut::Zero;
      for (int i = 0; i < IterationsQK; i++) {
        pipeline_load_qk.consumer_wait(pipeline_load_qk_consumer_state);
        int read_stage = pipeline_load_qk_consumer_state.index();

        tStS.data() = uint32_t(pipeline_mma_s_producer_state.index() == 0 ? TmemAllocation::kS0 : TmemAllocation::kS1);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
          cute::gemm(tiled_mma_qk,
                     tSrQ(_,_,k_block,i),
                     tSrKC(_,_,k_block,read_stage),
                     tStS);
          tiled_mma_qk.accumulate_ = UMMA::ScaleOut::One;
        }

        pipeline_load_qk.consumer_release(pipeline_load_qk_consumer_state);
        ++pipeline_load_qk_consumer_state;
      }

      pipeline_mma_s.producer_commit(pipeline_mma_s_producer_state);
      ++pipeline_mma_s_producer_state;

      pipeline_mma_o.producer_acquire(pipeline_mma_o_producer_state);
      pipeline_p_mma.consumer_wait(pipeline_p_mma_consumer_state);

      for (int i = 0; i < IterationsPV_K; i++) {
        auto acc_flag = tiled_mma_pv.accumulate_;
        for (int j = 0; j < IterationsPV_N; j++) {
          pipeline_load_pv.consumer_wait(pipeline_load_pv_consumer_state);

          int read_stage = pipeline_load_pv_consumer_state.index();

          tOtO.data() = uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO);
          tiled_mma_pv.accumulate_ = acc_flag;

          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
            cute::gemm(tiled_mma_pv,
                       tOrP(_,_,k_block, make_coord(i, pipeline_p_mma_consumer_state.index())),
                       tOrVC(_,_,k_block,read_stage),
                       tOtO);
            tiled_mma_pv.accumulate_ = UMMA::ScaleOut::One;
          }

          pipeline_load_pv.consumer_release(pipeline_load_pv_consumer_state);
          ++pipeline_load_pv_consumer_state;
        }
      }

      pipeline_p_mma.consumer_release(pipeline_p_mma_consumer_state);
      ++pipeline_p_mma_consumer_state;
      pipeline_mma_o.producer_commit(pipeline_mma_o_producer_state);
      ++pipeline_mma_o_producer_state;

      --k_tile_count;
    }

    pipeline_mma_o.producer_acquire(pipeline_mma_o_producer_state);
    pipeline_p_mma.consumer_wait(pipeline_p_mma_consumer_state);

    for (int i = 0; i < IterationsPV_K; i++) {
      auto acc_flag = tiled_mma_pv.accumulate_;
      for (int j = 0; j < IterationsPV_N; j++) {
        pipeline_load_pv.consumer_wait(pipeline_load_pv_consumer_state);

        int read_stage = pipeline_load_pv_consumer_state.index();

        tOtO.data() = uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO);
        tiled_mma_pv.accumulate_ = acc_flag;

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tOrP); ++k_block) {
          cute::gemm(tiled_mma_pv,
                     tOrP(_,_,k_block, make_coord(i, pipeline_p_mma_consumer_state.index())),
                     tOrVC(_,_,k_block,read_stage),
                     tOtO);
          tiled_mma_pv.accumulate_ = UMMA::ScaleOut::One;
        }

        pipeline_load_pv.consumer_release(pipeline_load_pv_consumer_state);
        ++pipeline_load_pv_consumer_state;
      }
    }

    pipeline_p_mma.consumer_release(pipeline_p_mma_consumer_state);
    ++pipeline_p_mma_consumer_state;
    pipeline_mma_o.producer_commit(pipeline_mma_o_producer_state);
    ++pipeline_mma_o_producer_state;
  }


  template<class IsLastTile>
  CUTLASS_DEVICE void softmax(
      IsLastTile const& is_last_tile,
      ElementAcc& row_max,
      ElementAcc& row_sum,
      ElementAcc& correction_factor,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      TensorStorage& shared_tensors,
      int k_index,
      uint32_t tmem_s,
      int smem_p_index) {

    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};

    TiledMmaQK tiled_mma_qk;

    Tensor tStS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShapeQK{}));
    tStS.data() = tmem_s;

    CUTE_STATIC_ASSERT_V(shape<1>(tStS) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tStS) == _1{});
    Tensor tAcc = tStS(make_coord(_,_),_0{},_0{});

    Tensor cS = make_identity_tensor(take<0,2>(CtaShapeQK{}));

    auto tiled_t2r = make_tmem_copy(load_op, tAcc);
    auto thread_idx = threadIdx.x % size(tiled_t2r);

    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_cS   = thread_t2r.partition_D(cS);
    Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_cS));

    Tensor tTR_rS_frag = make_tensor<Element>(shape(tTR_rAcc));
    const int AlignmentS = 4;
    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);
    Tensor tTR_rAcc_vec = recast<Array<ElementAcc, AlignmentS>>(tTR_rAcc);
    Tensor tTR_rS_vec = recast<Array<Element, AlignmentS>>(tTR_rS_frag);

    // load s
    copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

    if (is_last_tile) {
      for (int i = 0; i < size(tTR_rAcc); i++) {
        if (get<1>(tTR_cS(i)) + TileShapeS{} * k_index >= get<1>(problem_shape)) {
          tTR_rAcc(i) = -std::numeric_limits<ElementAcc>::infinity();
        }
      }
    }

    // max
    ElementAcc row_max_new = row_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i += 1) {
      row_max_new = ::fmax(row_max_new, tTR_rAcc(i));
    }

    // for 2x2 dp, reduce here
    if constexpr (kWarpsInN > 1) {
      shared_tensors.smem_exchange[threadIdx.x] = row_max_new;
      cutlass::arch::NamedBarrier(kNumComputeWarps*NumThreadsPerWarp, kNamedBarrierExchange).sync();
      // (64, 2) shape
      int peer_index = (threadIdx.x + 64) % 128;
      row_max_new = cutlass::max(row_max_new, shared_tensors.smem_exchange[peer_index]);
    }

    // find correction factor
    ElementAcc softmax_scale_log2 = mainloop_args.softmax_scale * static_cast<ElementAcc>(M_LOG2E);
    correction_factor = ::exp2f(softmax_scale_log2 * (row_max - row_max_new));
    row_max = row_max_new;

    // softmax
    ElementAcc row_max_scale_log2 = row_max * softmax_scale_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i++) {
      tTR_rAcc(i) = ::exp2f(softmax_scale_log2 * tTR_rAcc(i) - row_max_scale_log2);
    }

    // quantize
    cutlass::NumericArrayConverter<Element, ElementAcc, AlignmentS> epilogue_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc_vec); i++) {
      tTR_rS_vec(i) = epilogue_op(tTR_rAcc_vec(i));
    }

    Tensor sP = make_tensor(make_smem_ptr((Element*) shared_tensors.smem_p.begin()), SmemLayoutP{})(_, _, _, make_coord(_, smem_p_index));

    Tensor tOcP = TiledMmaPV{}.get_slice(_0{}).partition_A(cS);

    // have a mapping for each thread to coord
    // find identical mapping to coords for the MMA
    auto l = make_ordered_layout(make_shape(make_shape(_64{}, _2{}), make_shape(_16{}, TileShapeS{} / _32{})), make_stride(make_stride(_0{}, _3{}), make_stride(_1{}, _2{})));
    auto sP_ = as_position_independent_swizzle_tensor(sP);
    copy_aligned(tTR_rS_frag, sP_.compose(l)(threadIdx.x, _));

    // sum
    row_sum *= correction_factor;

    static_assert(cute::is_same_v<ElementAcc, float>);
    auto tTR_rAcc_float2 = recast<float2>(tTR_rAcc);
    auto sums = make_tensor<float2>(_4{});
    static_assert(size(tTR_rAcc_float2) % size(sums) == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(sums); i++) {
      sums(i) = tTR_rAcc_float2(i);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = size(sums); i < size(tTR_rAcc_float2); i += size(sums)) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(sums); j++) {
        cute::add(sums(j), sums(j), tTR_rAcc_float2(i + j));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < size(sums); i *= 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(sums); j += 2*i) {
        cute::add(sums(j), sums(j), sums(j+i));
      }
    }
    row_sum += sums(0).x + sums(0).y;
  }


  CUTLASS_DEVICE void rescale(
      ElementAcc correction_factor,
      uint32_t tmem_o) {

    // for b2b gemm, do nothing
    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};
    auto store_op = TMEM::tmem_load_to_store(load_op);

    TiledMmaPV tiled_mma_pv;

    Tensor tOtO = partition_fragment_C(tiled_mma_pv, select<0,1>(TileShapePV{}));
    tOtO.data() = tmem_o;

    CUTE_STATIC_ASSERT_V(shape<1>(tOtO) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tOtO) == _1{});
    Tensor tAcc = tOtO(make_coord(_,_),_0{},_0{});

    auto cta_tiler_pv = take<0,2>(typename CollectiveMmaPV::CtaShape_MNK{});
    Tensor gO = make_tensor(make_gmem_ptr((ElementAcc*) nullptr), cta_tiler_pv, make_stride(0, 0));

    auto tiled_t2r = make_tmem_copy(load_op, tAcc);
    auto tiled_r2t = make_tmem_copy(store_op, tAcc);
    auto thread_idx = threadIdx.x % size(tiled_t2r);

    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    auto thread_r2t = tiled_r2t.get_slice(thread_idx);
    Tensor tTR_gO   = thread_t2r.partition_D(gO);
    Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

    // load o
    copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

    // multiply by correction factor
    float2 correction_factor_vec = make_float2(correction_factor, correction_factor);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); i += 2) {
      float2 in = make_float2(tTR_rAcc(i + 0), tTR_rAcc(i + 1));
      float2 out;
      cute::mul(out, in, correction_factor_vec);
      tTR_rAcc(i + 0) = out.x;
      tTR_rAcc(i + 1) = out.y;
    }

    // store o
    copy(tiled_r2t, tTR_rAcc, tTR_tAcc);
 }


  template<class BlkCoord>
  CUTLASS_DEVICE void epilogue(
      ElementAcc& row_max,
      ElementAcc& row_sum,
      BlkCoord const& cta_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueParams const& epilogue_args,
      TensorStorage& shared_tensors,
      uint32_t tmem_o,
      int const& split_kv) {

    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};

    TiledMmaPV tiled_mma_pv;
    
    Tensor tOtO = TiledMmaPV::make_fragment_C(partition_shape_C(TiledMmaPV{}, take<0, 2>(TileShapePV{})));
    tOtO.data() = tmem_o;

    CUTE_STATIC_ASSERT_V(shape<1>(tOtO) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tOtO) == _1{});
    Tensor tAcc = tOtO(make_coord(_,_),_0{},_0{});

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    if (split_kv > 1) { 
      using ElementOutAcc = ElementAcc;
      constexpr auto AlignmentOutAcc = 128 / cute::sizeof_bits_v<ElementOutAcc>;
      Tensor mO = make_tensor(make_gmem_ptr(epilogue_args.ptr_o_acc + get<3>(cta_coord) * D_latent), make_shape(H, D_latent, B), epilogue_args.stride_o_acc);
      auto cta_tiler_pv = take<0,2>(typename CollectiveMmaPV::CtaShape_MNK{});
      Tensor gO = local_tile(mO, cta_tiler_pv, take<0,3>(cta_coord));

      auto tiled_t2r = make_tmem_copy(load_op, tAcc);
      auto thread_idx = threadIdx.x % size(tiled_t2r);

      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_gO   = thread_t2r.partition_D(gO);
      Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

      Tensor tTR_rO_frag = make_tensor<ElementOutAcc>(shape(tTR_rAcc));
      Tensor tTR_rO_src = recast<Array<ElementOutAcc, AlignmentOutAcc>>(coalesce(tTR_rO_frag));
      Tensor tR2G_rO_dst = recast<Array<ElementOutAcc, AlignmentOutAcc>>(coalesce(tTR_gO));
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      cutlass::epilogue::thread::LinearCombination<ElementOutAcc, 1, ElementAcc, ElementAcc, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling> epilogue_op({epilogue_args.output_scale / row_sum});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rO_frag(i) = epilogue_op(tTR_rAcc(i));
      }

      copy(tTR_rO_src, tR2G_rO_dst);

      if (get<1>(cta_coord) == 0) {
	if (epilogue_args.ptr_lse != nullptr) {
          // compute LSE
          ElementAcc lse = cutlass::fast_log(row_sum) + mainloop_args.softmax_scale * row_max;

          // store LSE
          Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse_acc + H * get<3>(cta_coord)), make_shape(H, B), epilogue_args.stride_lse_acc);
          Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), take<0,3>(cta_coord), Step<_1, Underscore, _1>{});
          // for 2x2 dp, this must be conditional and the index is wrong
          if (! kIs2Sm || (threadIdx.x < 64))
          {
            gLSE(threadIdx.x) = lse;
          }
	}
      }
    }
    else {
      Tensor mO = make_tensor(make_gmem_ptr(epilogue_args.ptr_o), make_shape(H, D_latent, B), epilogue_args.stride_o);
      auto cta_tiler_pv = take<0,2>(typename CollectiveMmaPV::CtaShape_MNK{});
      Tensor gO = local_tile(mO, cta_tiler_pv, take<0,3>(cta_coord));

      auto tiled_t2r = make_tmem_copy(load_op, tAcc);
      auto thread_idx = threadIdx.x % size(tiled_t2r);

      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_gO   = thread_t2r.partition_D(gO);
      Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));

      Tensor tTR_rO_frag = make_tensor<ElementOut>(shape(tTR_rAcc));
      Tensor tTR_rO_src = recast<Array<ElementOut, AlignmentOut>>(coalesce(tTR_rO_frag));
      Tensor tR2G_rO_dst = recast<Array<ElementOut, AlignmentOut>>(coalesce(tTR_gO));
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      cutlass::epilogue::thread::LinearCombination<ElementOut, 1, ElementAcc, ElementAcc, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling> epilogue_op({epilogue_args.output_scale / row_sum});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rO_frag(i) = epilogue_op(tTR_rAcc(i));
      }

      copy(tTR_rO_src, tR2G_rO_dst);

      if (get<1>(cta_coord) == 0) {
        if (epilogue_args.ptr_lse != nullptr) {
          // compute LSE
          ElementAcc lse = cutlass::fast_log(row_sum) + mainloop_args.softmax_scale * row_max;

          // store LSE
          Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse), make_shape(H, B), epilogue_args.stride_lse);
          Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), take<0,3>(cta_coord), Step<_1, Underscore, _1>{});

          // for 2x2 dp, this must be conditional and the index is wrong
          if (! kIs2Sm || (threadIdx.x < 64))
          {
            gLSE(threadIdx.x) = lse;
          }
        }
      }
    }
  }

  template<class BlkCoord>
  CUTLASS_DEVICE ElementLSE epilogue_lse_reduction(
      ElementAcc& row_max,
      ElementAcc& row_sum,
      BlkCoord const& cta_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueParams const& epilogue_args,
      int const& local_split_kv) {

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;
    auto cta_tiler_pv = take<0,2>(typename CollectiveMmaPV::CtaShape_MNK{});

    constexpr int kNumThreads = kNumComputeWarps * NumThreadsPerWarp;
    using Sync = cutlass::detail::NamedBarrierSync<kNumThreads, kNamedBarrierExchange>;

    auto wait = [](int* lock, int count) {
      __threadfence();
      if (threadIdx.x == 0) {
        atomicAdd(lock, 1);
        while (atomicCAS(lock, count, count) != count) {};
      }
      __threadfence();
      Sync::sync();
    };

    const ElementLSE lse  = cutlass::fast_log(row_sum) + mainloop_args.softmax_scale * row_max;
    Tensor mLSE_max_buff = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse_max_exchange_buff), make_shape(H, B), epilogue_args.stride_lse);
    Tensor gLSE_max_buff = local_tile(mLSE_max_buff, append<3>(cta_tiler_pv, _1{}), take<0,3>(cta_coord), Step<_1, Underscore, _1>{});

    int* local_lock = epilogue_args.ptr_lock + get<0>(cta_coord) + 2 * get<2>(cta_coord);

    if (! kIs2Sm || (threadIdx.x < 64)) {
      atomicMax(&(gLSE_max_buff(threadIdx.x)), __float2int_rn(lse));
    }
    wait(local_lock, local_split_kv);

    auto global_lse_max = static_cast<ElementLSE>(gLSE_max_buff(kIs2Sm ? threadIdx.x % 64 : threadIdx.x));

    Tensor mLSE_buff = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse_exchange_buff), make_shape(H, B), epilogue_args.stride_lse);
    Tensor gLSE_buff = local_tile(mLSE_buff, append<3>(cta_tiler_pv, _1{}), take<0,3>(cta_coord), Step<_1, Underscore, _1>{});

    if (! kIs2Sm || (threadIdx.x < 64)) {
      atomicAdd(&(gLSE_buff(threadIdx.x)), expf(lse - global_lse_max));
    }
    wait(local_lock, 2*local_split_kv);

    const auto sum_lse = gLSE_buff(kIs2Sm ? threadIdx.x % 64 : threadIdx.x);
    const auto global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? std::numeric_limits<ElementLSE>::infinity() : 
	       cutlass::fast_log(sum_lse) + global_lse_max;
    const auto lse_scale = expf(lse - global_lse);

    if (epilogue_args.ptr_lse != nullptr) {
      Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_args.ptr_lse), make_shape(H, B), epilogue_args.stride_lse);
      Tensor gLSE = local_tile(mLSE, append<3>(cta_tiler_pv, _1{}), take<0,3>(cta_coord), Step<_1, Underscore, _1>{});

      // write out the global LSE
      if (! kIs2Sm || (threadIdx.x < 64)) {
        gLSE(threadIdx.x) = global_lse;
      }
    }
    return lse_scale;
  }


  template<class BlkCoord>
  CUTLASS_DEVICE void epilogue_reduction(
      ElementAcc& row_max,
      ElementAcc& row_sum,
      BlkCoord const& blk_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueParams const& epilogue_args,
      TensorStorage& shared_tensors,
      int const& local_split_kv,
      ElementLSE const& lse_scale) {

    constexpr int kNumThreads = kNumComputeWarps * NumThreadsPerWarp;
    using Sync = cutlass::detail::NamedBarrierSync<kNumThreads, kNamedBarrierExchange>;

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    auto load_op = cute::SM100_TMEM_LOAD_32dp32b32x{};

    TiledMmaPV tiled_mma_pv;
    Tensor tOtO = TiledMmaPV::make_fragment_C(partition_shape_C(TiledMmaPV{}, take<0, 2>(TileShapePV{})));

    CUTE_STATIC_ASSERT_V(shape<1>(tOtO) == _1{});
    CUTE_STATIC_ASSERT_V(shape<2>(tOtO) == _1{});

    using EpilogueLinearCombination = cutlass::epilogue::thread::LinearCombination<ElementOut, 1, ElementAcc, ElementAcc, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
    EpilogueLinearCombination epilogue_op({epilogue_args.output_scale / row_sum * lse_scale});

    CUTLASS_PRAGMA_UNROLL
    for(int k = 0; k < IterationsPV_N; ++k) {
      auto cta_coord = replace<1>(blk_coord, k);

      uint32_t tmem_o = uint32_t(TmemAllocation::kO0) + k * uint32_t(TmemAllocation::kSizeAccO);
      tOtO.data() = tmem_o;

      Tensor tAcc = tOtO(make_coord(_,_),_0{},_0{});

      Tensor mO = make_tensor(make_gmem_ptr(epilogue_args.ptr_o), make_shape(H, D_latent, B), epilogue_args.stride_o);
      Tensor gO = local_tile(mO, TileShapeAcc{}, take<0,3>(cta_coord));
      
      auto tiled_t2r = make_tmem_copy(load_op, tAcc);
      auto thread_idx = threadIdx.x % size(tiled_t2r);

      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_gO   = thread_t2r.partition_D(gO);
      Tensor tTR_rAcc = make_tensor<ElementAcc>(shape(tTR_gO));
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

      Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementOut*>(shared_tensors.smem_acc.begin())), SmemLayout{});
      Tensor tTR_sO = thread_t2r.partition_D(sO);

      Sync::sync();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_sO(i) = epilogue_op(tTR_rAcc(i));
      }
      tma_store_fence();
      Sync::sync();

      auto tma_reduce_sum_per_cta = epilogue_args.tma_reduce_sum.get_slice(_0{});
      auto gmem_tensor_coord = epilogue_args.tma_reduce_sum.get_tma_tensor(shape(mO));
      auto gmem_tensor_coord_per_cta = local_tile(gmem_tensor_coord, TileShapeAcc{}, take<0,3>(cta_coord));
      if (threadIdx.x % kNumThreads == 0) {
        copy(epilogue_args.tma_reduce_sum,
           tma_reduce_sum_per_cta.partition_S(sO),
           tma_reduce_sum_per_cta.partition_D(gmem_tensor_coord_per_cta));
	tma_store_arrive();
      }
      tma_store_wait<0>();
    }
  }

  template<class CtaCoord>
  CUTLASS_DEVICE void compute(
      CtaCoord const& cta_coord,
      ProblemShape const& problem_shape,
      MainloopArguments const& mainloop_args,
      EpilogueParams const& epilogue_args,
      TensorStorage& shared_tensors,
      PipelineS& pipeline_mma_s,
      typename PipelineS::PipelineState& pipeline_mma_s_consumer_state,
      PipelineP& pipeline_p_mma,
      typename PipelineP::PipelineState& pipeline_p_mma_producer_state,
      PipelineO& pipeline_mma_o,
      typename PipelineO::PipelineState& pipeline_mma_o_consumer_state,
      int const& split_kv,
      bool const& is_fused_reduction) {

    auto [H, K, D, B] = problem_shape;

    int k_tile_total = ceil_div(K, TileShapeS{});
    int k_tile_per_cta = ceil_div(k_tile_total, split_kv);
    int k_index = get<3>(cta_coord) * k_tile_per_cta; // lower limit
    int k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index);
    if (k_tile_count == 0) {

      // if we return early, we have to make sure we release the load warp
      cutlass::arch::NamedBarrier(
          (kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
          kNamedBarrierEpilogue
      ).arrive();

      return;
    }
    int k_index_final = k_tile_total - 1;

    ElementAcc row_max = -std::numeric_limits<ElementAcc>::infinity();
    ElementAcc row_sum = 0;
    ElementAcc correction_factor = 1;

    pipeline_p_mma.producer_acquire(pipeline_p_mma_producer_state);
    pipeline_mma_s.consumer_wait(pipeline_mma_s_consumer_state);

    auto dispatch_bool = [](bool b, auto fn) {
      if (b) {
        fn(cute::true_type{});
      }
      else {
        fn(cute::false_type{});
      }
    };

    // softmax s0 -> p0
    dispatch_bool(k_index == k_index_final, [&](auto is_last_tile) {
      softmax(
          is_last_tile,
          row_max, row_sum, correction_factor,
          problem_shape, mainloop_args, shared_tensors, k_index,
          uint32_t(pipeline_mma_s_consumer_state.index() == 0 ? TmemAllocation::kS0 : TmemAllocation::kS1),
          pipeline_p_mma_producer_state.index()
      );
    });
    
    k_index += 1;

    cutlass::arch::fence_view_async_tmem_load();
    cutlass::arch::fence_view_async_shared();
    pipeline_mma_s.consumer_release(pipeline_mma_s_consumer_state);
    ++pipeline_mma_s_consumer_state;
    pipeline_p_mma.producer_commit(pipeline_p_mma_producer_state);
    ++pipeline_p_mma_producer_state;

    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      pipeline_p_mma.producer_acquire(pipeline_p_mma_producer_state);
      pipeline_mma_s.consumer_wait(pipeline_mma_s_consumer_state);

      // softmax s1 -> p1
      dispatch_bool(k_index == k_index_final, [&](auto is_last_tile) {
        softmax(
            is_last_tile,
            row_max, row_sum, correction_factor,
            problem_shape, mainloop_args, shared_tensors, k_index,
            uint32_t(pipeline_mma_s_consumer_state.index() == 0 ? TmemAllocation::kS0 : TmemAllocation::kS1),
            pipeline_p_mma_producer_state.index()
        );
      });

      cutlass::arch::fence_view_async_tmem_load();
      cutlass::arch::fence_view_async_shared();
      pipeline_mma_s.consumer_release(pipeline_mma_s_consumer_state);
      ++pipeline_mma_s_consumer_state;
      pipeline_p_mma.producer_commit(pipeline_p_mma_producer_state);
      ++pipeline_p_mma_producer_state;

      pipeline_mma_o.consumer_wait(pipeline_mma_o_consumer_state);

      // rescale
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IterationsPV_N; j++) {
        rescale(correction_factor, uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO));
      }

      cutlass::arch::fence_view_async_tmem_store();
      pipeline_mma_o.consumer_release(pipeline_mma_o_consumer_state);
      ++pipeline_mma_o_consumer_state;

      --k_tile_count;
      k_index += 1;
    }

    pipeline_mma_o.consumer_wait(pipeline_mma_o_consumer_state);

    if constexpr (kWarpsInN > 1) {
      // reduce row_sum if needed (for 2x2 dp)
      shared_tensors.smem_exchange[threadIdx.x] = row_sum;
      cutlass::arch::NamedBarrier(kNumComputeWarps*NumThreadsPerWarp, kNamedBarrierExchange).sync();
      // (64, 2) shape
      int peer_index = (threadIdx.x + 64) % 128;
      row_sum += shared_tensors.smem_exchange[peer_index];
    }

    cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, kNamedBarrierEpilogue).arrive();

    const int actual_split_kv = ceil_div(k_tile_total, k_tile_per_cta);
    if (!is_fused_reduction || actual_split_kv == 1) {
      // epilogue
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IterationsPV_N; j++) {
        epilogue(
            row_max, row_sum,
            replace<1>(cta_coord, j), problem_shape,
            mainloop_args, epilogue_args, shared_tensors,
            uint32_t(TmemAllocation::kO0) + j * uint32_t(TmemAllocation::kSizeAccO),
	    actual_split_kv
    	);
      }
    } else {
      const ElementLSE lse_scale =
        epilogue_lse_reduction(
            row_max, row_sum,
	    cta_coord,
            problem_shape,
            mainloop_args, epilogue_args,
            actual_split_kv
        );
     
      epilogue_reduction(row_max, row_sum,
            cta_coord, 
	    problem_shape,
            mainloop_args, epilogue_args,
	    shared_tensors,
	    actual_split_kv,
            lse_scale
      );
    }
    cutlass::arch::fence_view_async_tmem_load();
    pipeline_mma_o.consumer_release(pipeline_mma_o_consumer_state);
    ++pipeline_mma_o_consumer_state;
  }

};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel
""",
    'key-files/examples/77_blackwell_fmha/kernel/sm100_mla_tile_scheduler.hpp': r"""/***************************************************************************************************
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
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

////////////////////////////////////////////////////////////////////////////////

struct Sm100MlaIndividualTileScheduler {

  struct Params {
    dim3 grid;
  };

  bool valid_ = true;

  CUTLASS_DEVICE
  Sm100MlaIndividualTileScheduler(Params const&) {}

  template<class ProblemShape, class ClusterShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, int const& split_kv) {
    using namespace cute;
    dim3 grid(get<0>(cluster_shape), get<3>(problem_shape) /* Batch */, split_kv /*Maximum Split KV*/);
    return Params{ grid };
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
    return make_coord(blockIdx.x, _0{}, blockIdx.y, blockIdx.z);
  }

  CUTLASS_DEVICE
  Sm100MlaIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct Sm100MlaPersistentTileScheduler {

  struct Params {
    int num_blocks;
    FastDivmod divmod_m_block;
    FastDivmod divmod_b;
    FastDivmod divmod_split_kv;
    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  Sm100MlaPersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

  template<class ProblemShape, class ClusterShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, int const& split_kv) {
    using namespace cute;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = hw_info.sm_count;
    if (sm_count <= 1 || sm_count % size<0>(cluster_shape) != 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    hw_info.sm_count = sm_count;

    int num_m_blocks = size<0>(cluster_shape);
    int num_blocks = num_m_blocks * get<3>(problem_shape)  /* Batch */;
    num_blocks *= split_kv; /* Maximum Split KV*/

    return Params {
      num_blocks,
      { num_m_blocks}, { get<3>(problem_shape) }, {split_kv},
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
    int m_block, bidb, n_split_kv;
    params.divmod_m_block(block_decode, m_block, block_decode);
    params.divmod_split_kv(block_decode, n_split_kv, block_decode);
    params.divmod_b(block_decode, bidb, block_decode);
    return make_coord(m_block, _0{}, bidb, n_split_kv);
  }

  CUTLASS_DEVICE
  Sm100MlaPersistentTileScheduler& operator++() {
    block_idx += gridDim.x;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel

""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
