"""KernelWiki artifact PR-2378 (NVIDIA/cutlass)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/include/cute/arch/mma_sm89.hpp': r"""/***************************************************************************************************
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

//

//
#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>

////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
#  define CUTE_ARCH_MMA_F32_SM89_SUPPORTED
#endif

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#  define CUTE_ARCH_MMA_F16_SM89_SUPPORTED
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#  if defined(CUTE_ARCH_MMA_F32_SM89_SUPPORTED)
#    define CUTE_ARCH_MMA_F32_SM89_ENABLED
#  endif

#  if defined(CUTE_ARCH_MMA_F16_SM89_SUPPORTED)
#    define CUTE_ARCH_MMA_F16_SM89_ENABLED
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cute {
// MMA 16x8x32 TN
struct SM89_16x8x32_F32E4M3E4M3F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F32E4M3E5M2F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E5M2F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F32E5M2E5M2F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E5M2E5M2F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F32E5M2E4M3F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E5M2E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F16E4M3E4M3F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F16E4M3E4M3F16_TN without CUTE_ARCH_MMA_F16_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F16E4M3E5M2F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F16E4M3E5M2F16_TN without CUTE_ARCH_MMA_F16_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F16E5M2E4M3F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F16E5M2E4M3F16_TN without CUTE_ARCH_MMA_F16_SM89_ENABLED");
#endif
  }
};

struct SM89_16x8x32_F16E5M2E5M2F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F16E5M2E5M2F16_TN without CUTE_ARCH_MMA_F16_SM89_ENABLED");
#endif
  }
};
} // namespace cute
""",
    'key-files/include/cute/atom/mma_traits_sm89.hpp': r"""/***************************************************************************************************
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

//

//
#pragma once

#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace cute 
{

namespace {

// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;

}

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  using CLayout = SM80_16x8_Row;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E4M3E5M2F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E5M2F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F32E5M2E4M3F32_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E4M3E4M3F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e4m3_t;
  using ValTypeB = cutlass::float_e4m3_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E4M3E5M2F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e4m3_t;
  using ValTypeB = cutlass::float_e5m2_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E5M2E5M2F16_TN>
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e5m2_t;
  using ValTypeB = cutlass::float_e5m2_t;
  using ValTypeC = cutlass::half_t;
};

template <>
struct MMA_Traits<SM89_16x8x32_F16E5M2E4M3F16_TN> 
     : MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using ValTypeD = cutlass::half_t;
  using ValTypeA = cutlass::float_e5m2_t;
  using ValTypeB = cutlass::float_e4m3_t;
  using ValTypeC = cutlass::half_t;
};


} // end namespace cute
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
