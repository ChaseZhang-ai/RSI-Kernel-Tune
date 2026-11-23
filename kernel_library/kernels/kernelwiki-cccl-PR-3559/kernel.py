"""KernelWiki artifact PR-3559 (NVIDIA/cccl)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/cub/cub/device/device_scan.cuh': r"""/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data
//! items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/std/__functional/invoke.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceScan provides device-wide, parallel operations for computing a
//! prefix scan across a sequence of data items residing within
//! device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Given a sequence of input elements and a binary reduction operator, a
//! `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`_ produces an output
//! sequence where each element is computed to be the reduction of the elements
//! occurring earlier in the input sequence. *Prefix sum* connotes a prefix scan
//! with the addition operator. The term *inclusive* indicates that the
//! *i*\ :sup:`th` output reduction incorporates the *i*\ :sup:`th` input.
//! The term *exclusive* indicates the *i*\ :sup:`th` input is not
//! incorporated into the *i*\ :sup:`th` output reduction. When the input and
//! output sequences are the same, the scan is performed in-place.
//!
//! As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our
//! *"decoupled look-back"* algorithm for performing global prefix scan with
//! only a single pass through the input data, as described in our 2016 technical
//! report [1]_. The central idea is to leverage a small, constant factor of
//! redundant work in order to overlap the latencies of global prefix
//! propagation with local computation. As such, our algorithm requires only
//! ``~2*n*`` data movement (``n`` inputs are read, ``n`` outputs are written), and
//! typically proceeds at "memcpy" speeds. Our algorithm supports inplace operations.
//!
//! .. [1] Duane Merrill and Michael Garland. `Single-pass Parallel Prefix Scan with Decoupled Look-back
//!    <https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back>`_,
//!    *NVIDIA Technical Report NVR-2016-002*, 2016.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceScan}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{prefix scan}
//!
//! @endrst
struct DeviceScan
{
  //! @name Exclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide exclusive prefix sum.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using InitT   = cub::detail::value_t<InputIteratorT>;

    // Initial value
    InitT init_value{};

    return DispatchScan<InputIteratorT, OutputIteratorT, ::cuda::std::plus<>, detail::InputValue<InitT>, OffsetT>::
      Dispatch(d_temp_storage,
               temp_storage_bytes,
               d_in,
               d_out,
               ::cuda::std::plus<>{},
               detail::InputValue<InitT>(init_value),
               num_items,
               stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum in-place.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_data``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading scan inputs and wrigin scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage, size_t& temp_storage_bytes, IteratorT d_data, NumItemsT num_items, cudaStream_t stream = 0)
  {
    return ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_data``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an
  //! ``int`` device vector:
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary ``scan_op`` functor. The ``init_value`` value is provided as a future value.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, detail::InputValue<InitValueT>, OffsetT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified binary ``scan_op`` functor.
  //! The ``init_value`` value is provided as a future value.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value` used Binary scan functor type
  //!   having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Pointer to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @}  end member group
  //! @name Inclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide inclusive prefix sum.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ::cuda::std::plus<>, NullType, OffsetT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, ::cuda::std::plus<>{}, NullType{}, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum in-place.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage, size_t& temp_storage_bytes, IteratorT d_data, NumItemsT num_items, cudaStream_t stream = 0)
  {
    return InclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_out <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, NullType, OffsetT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, NullType(), num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and ``*d_in``
  //! is assigned to ``*d_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive max-scan of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-inclusive-scan
  //!     :end-before: example-end device-inclusive-scan
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to the size in bytes of the `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the inclusive scan (`scan_op(init_value, d_in[0])`
  //!   is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within.
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanInit");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using AccumT  = ::cuda::std::__accumulator_t<ScanOpT, cub::detail::value_t<InputIteratorT>, InitValueT>;

    return DispatchScan<
      InputIteratorT,
      OutputIteratorT,
      ScanOpT,
      detail::InputValue<InitValueT>,
      OffsetT,
      AccumT,
      detail::scan::
        policy_hub<detail::value_t<InputIteratorT>, detail::value_t<OutputIteratorT>, AccumT, OffsetT, ScanOpT>,
      ForceInclusive::Yes>::Dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     scan_op,
                                     detail::InputValue<InitValueT>(init_value),
                                     num_items,
                                     stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_data <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return InclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum-by-key with key equality
  //! defined by ``equality_op``. The value of ``0`` is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_values_out <-- [0, 8, 0, 7, 12, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = std::uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSumByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using InitT   = cub::detail::value_t<ValuesInputIteratorT>;

    // Initial value
    InitT init_value{};

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ::cuda::std::plus<>,
      InitT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         ::cuda::std::plus<>{},
                         init_value,
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan-by-key using the
  //! specified binary ``scan_op`` functor. The key equality is defined by
  //! ``equality_op``.  The ``init_value`` value is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan-by-key of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // d_values_out <-- [2147483647, 8, 2147483647, 7, 5, 2147483647, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the `init_value` value used in Binary scan
  //!   functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage. When `nullptr`, the
  //!    required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary scan functor
  //!
  //!  @param[in] init_value
  //!    Initial value to seed the exclusive scan (and is assigned to the
  //!    beginning of each segment in `d_values_out`)
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and
  //!    `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = std::uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScanByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ScanOpT,
      InitValueT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         scan_op,
                         init_value,
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum-by-key with key equality defined by ``equality_op``.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 7, 12, 15, 0, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = std::uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSumByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ::cuda::std::plus<>,
      NullType,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         ::cuda::std::plus<>{},
                         NullType{},
                         num_items,
                         stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan-by-key using the
  //! specified binary ``scan_op`` functor. The key equality is defined by ``equality_op``.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!    #include <climits>       // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // d_out <-- [8, 6, 7, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary scan functor
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = std::uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanByKey");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchScanByKey<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ScanOpT,
      NullType,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_values_out,
                         equality_op,
                         scan_op,
                         NullType(),
                         num_items,
                         stream);
  }

  //! @}  end member group
};

CUB_NAMESPACE_END
""",
    'key-files/cub/cub/device/dispatch/dispatch_scan.cuh': r"""/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceScan provides device-wide, parallel operations for
 *       computing a prefix scan across a sequence of data items residing
 *       within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/util_namespace.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/kernels/scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

enum class ForceInclusive
{
  Yes,
  No
};

namespace detail::scan
{

template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitValueT,
          typename OffsetT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct DeviceScanKernelSource
{
  using ScanTileStateT = typename cub::ScanTileState<AccumT>;

  CUB_DEFINE_KERNEL_GETTER(ScanInitKernel, DeviceScanInitKernel<ScanTileStateT>)

  CUB_DEFINE_KERNEL_GETTER(
    ScanKernel,
    DeviceScanKernel<MaxPolicyT,
                     InputIteratorT,
                     OutputIteratorT,
                     ScanTileStateT,
                     ScanOpT,
                     InitValueT,
                     OffsetT,
                     AccumT,
                     EnforceInclusive == ForceInclusive::Yes>)

  CUB_RUNTIME_FUNCTION static constexpr std::size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

} // namespace detail::scan

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        DeviceScan
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading scan inputs @iterator
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type for writing scan outputs @iterator
 *
 * @tparam ScanOpT
 *   Binary scan functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   The init_value element type for ScanOpT (cub::NullType for inclusive scans)
 *
 * @tparam OffsetT
 *   Unsigned integer type for global offsets
 *
 * @tparam EnforceInclusive
 *   Enum flag to specify whether to enforce inclusive scan.
 *
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename ScanOpT,
  typename InitValueT,
  typename OffsetT,
  typename AccumT = ::cuda::std::__accumulator_t<ScanOpT,
                                                 cub::detail::value_t<InputIteratorT>,
                                                 ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>,
                                                                  cub::detail::value_t<InputIteratorT>,
                                                                  typename InitValueT::value_type>>,
  typename PolicyHub =
    detail::scan::policy_hub<detail::value_t<InputIteratorT>, detail::value_t<OutputIteratorT>, AccumT, OffsetT, ScanOpT>,
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename KernelSource           = detail::scan::DeviceScanKernelSource<
              typename PolicyHub::MaxPolicy,
              InputIteratorT,
              OutputIteratorT,
              ScanOpT,
              InitValueT,
              OffsetT,
              AccumT,
              EnforceInclusive>,
  typename KernelLauncherFactory = detail::TripleChevronFactory>
struct DispatchScan
{
  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  // The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  /// Device-accessible allocation of temporary storage.  When nullptr, the
  /// required allocation size is written to \p temp_storage_bytes and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  size_t& temp_storage_bytes;

  /// Iterator to the input sequence of data items
  InputIteratorT d_in;

  /// Iterator to the output sequence of data items
  OutputIteratorT d_out;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Total number of input items (i.e., the length of \p d_in)
  OffsetT num_items;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  /**
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
      , num_items(num_items)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename ActivePolicyT, typename InitKernelT, typename ScanKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  Invoke(InitKernelT init_kernel, ScanKernelT scan_kernel, ActivePolicyT policy = {})
  {
    using ScanTileStateT = typename KernelSource::ScanTileStateT;

    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    static_assert(policy.LoadModifier() != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");

    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = policy.Scan().BlockThreads() * policy.Scan().ItemsPerThread();
      int num_tiles = static_cast<int>(::cuda::ceil_div(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[1];
      error = CubDebug(ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break; // bytes needed for tile status descriptors
      }

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void* allocations[1] = {};

      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_state;
      error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log init_kernel configuration
      int init_grid_size = ::cuda::ceil_div(num_tiles, INIT_KERNEL_THREADS);

#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);
#endif // CUB_DEBUG_LOG

      // Invoke init_kernel to initialize tile descriptors
      launcher_factory(init_grid_size, INIT_KERNEL_THREADS, 0, stream).doit(init_kernel, tile_state, num_tiles);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get SM occupancy for scan_kernel
      int scan_sm_occupancy;
      error = CubDebug(MaxSmOccupancy(scan_sm_occupancy, // out
                                      scan_kernel,
                                      policy.Scan().BlockThreads()));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(launcher_factory.MaxGridDimX(max_dim_x));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
      {
// Log scan_kernel configuration
#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items "
                "per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                policy.Scan().BlockThreads(),
                (long long) stream,
                policy.Scan().ItemsPerThread(),
                scan_sm_occupancy);
#endif // CUB_DEBUG_LOG

        // Invoke scan_kernel
        launcher_factory(scan_grid_size, policy.Scan().BlockThreads(), 0, stream)
          .doit(scan_kernel, d_in, d_out, tile_state, start_tile, scan_op, init_value, num_items);

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    using ScanTileStateT = typename KernelSource::ScanTileStateT;
    auto wrapped_policy  = detail::scan::MakeScanPolicyWrapper(active_policy);
    // Ensure kernels are instantiated.
    return Invoke(kernel_source.ScanInitKernel(), kernel_source.ScanKernel(), wrapped_policy);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] num_items
   *   Total number of input items (i.e., the length of `d_in`)
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   *
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    OffsetT num_items,
    cudaStream_t stream,
    KernelSource kernel_source = {},
    MaxPolicyT max_policy      = {})
  {
    cudaError_t error;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchScan dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        scan_op,
        init_value,
        stream,
        ptx_version,
        kernel_source);

      // Dispatch to chained policy
      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
""",
    'key-files/cub/cub/device/dispatch/tuning/tuning_scan.cuh': r"""/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/functional>
#include <cuda/std/functional>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace scan
{
enum class keep_rejects
{
  no,
  yes
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
{
  no,
  yes
};
enum class op_type
{
  plus,
  unknown
};
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class value_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class AccumT>
constexpr primitive_accum is_primitive_accum()
{
  return is_primitive<AccumT>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ScanOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
}

template <typename Op>
struct is_plus
{
  static constexpr bool value = false;
};

template <typename T>
struct is_plus<::cuda::std::plus<T>>
{
  static constexpr bool value = true;
};

template <class ScanOpT>
constexpr op_type classify_op()
{
  return is_plus<ScanOpT>::value ? op_type::plus : op_type::unknown;
}

template <class ValueT>
constexpr value_size classify_value_size()
{
  return sizeof(ValueT) == 1 ? value_size::_1
       : sizeof(ValueT) == 2 ? value_size::_2
       : sizeof(ValueT) == 4 ? value_size::_4
       : sizeof(ValueT) == 8 ? value_size::_8
       : sizeof(ValueT) == 16
         ? value_size::_16
         : value_size::unknown;
}

template <class AccumT>
constexpr accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}

template <class OffsetT>
constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm80_tuning;

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 14;
  using delay_constructor                              = fixed_delay_constructor_t<368, 725>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2>
{
  static constexpr int threads                         = 352;
  static constexpr int items                           = 16;
  using delay_constructor                              = fixed_delay_constructor_t<488, 1040>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<268, 1180>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <class T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 22;
  using delay_constructor                              = fixed_delay_constructor_t<716, 785>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<float, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 8;
  using delay_constructor                              = fixed_delay_constructor_t<724, 1050>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 384;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<388, 1100>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{
  static constexpr int threads                         = 640;
  static constexpr int items                           = 24;
  using delay_constructor                              = no_delay_constructor_t<1200>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
};

template <>
struct sm80_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm80_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm90_tuning;

template <class AccumT, int Threads, int Items, int L2B, int L2W>
struct sm90_tuning_vals
{
  static constexpr int threads = Threads;
  static constexpr int items   = Items;
  using delay_constructor      = fixed_delay_constructor_t<L2B, L2W>;
  // same logic as default policy:
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm load_algorithm =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;
};

// clang-format off
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1> : sm90_tuning_vals<T, 192, 22, 168, 1140> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2> : sm90_tuning_vals<T, 512, 12, 376, 1125> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<T, 128, 24, 648, 1245> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<T, 224, 24, 632, 1290> {};

template <> struct sm90_tuning<float,  primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<float,  128, 24, 688, 1140> {};
template <> struct sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<double, 224, 24, 576, 1215> {};

#if _CCCL_HAS_INT128()
template <> struct sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16> : sm90_tuning_vals<__int128_t, 576, 21, 860, 630> {};
template <>
struct sm90_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif
// clang-format on

template <class ValueT,
          class AccumT,
          class OffsetT,
          op_type OpTypeT,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          offset_size OffsetSize               = classify_offset_size<OffsetT>(),
          value_size ValueSize                 = classify_value_size<ValueT>()>
struct sm100_tuning;

// sum
template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_1>
{
  // ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0 1.188818  1.005682  1.173041  1.305288
  static constexpr int items                           = 18;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<768, 820>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_1>
{
  // ipt_14.tpb_384.ns_228.dcid_7.l2w_775.trp_1.ld_1 1.107210  1.000000  1.100637  1.307692
  static constexpr int items                           = 14;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_constructor_t<228, 775>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_2>
{
  // ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0 1.128443  1.002841  1.119688  1.307692
  static constexpr int items                           = 13;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<1384, 720>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Regresses for large inputs. Find better tuning.
// template <class ValueT, class AccumT, class OffsetT>
// struct sm100_tuning<ValueT,
//                     AccumT,
//                     OffsetT,
//                     op_type::plus,
//                     primitive_value::yes,
//                     primitive_accum::yes,
//                     offset_size::_8,
//                     value_size::_2>
// {
//   // ipt_13.tpb_288.ns_1520.dcid_5.l2w_895.trp_1.ld_1 1.080934  0.983509  1.077724  1.305288
//   static constexpr int items                           = 13;
//   static constexpr int threads                         = 288;
//   using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1520, 895>;
//   static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
// };

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_4>
{
  // ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442  0.997167  1.139902  1.462651
  static constexpr int items                           = 22;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_jitter_constructor_t<1904, 830>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_4>
{
  // ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1 1.146142  0.994350  1.137459  1.455636
  static constexpr int items                           = 19;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_constructor_t<956, 550>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_4, value_size::_8>
{
  // ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0 1.089468  1.015581  1.085630  1.264583
  static constexpr int items                           = 23;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<772, 710>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
{
  // ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0 1.080133  1.000000  1.075577  1.248963
  static constexpr int items                           = 22;
  static constexpr int threads                         = 320;
  using delay_constructor                              = exponential_backoff_constructor_t<328, 965>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for i128, float and double.
// template <class OffsetT> struct sm100_tuning<float, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8,
// accum_size::_4>;
// Default explicitly so it doesn't pick up the sm100<I64, I64> tuning.
template <class AccumT, class OffsetT>
struct sm100_tuning<double, AccumT, OffsetT, op_type::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
    : sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{};

#if CUB_IS_INT128_ENABLED
// template <class OffsetT> struct sm100_tuning<__int128_t, OffsetT, op_type::plus, primitive_accum::no,
// offset_size::_8, accum_size::_16> : tuning<576, 21, 860, 630> {}; template <class OffsetT> struct
// sm100_tuning<__uint128_t, OffsetT, op_type::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
//     : sm100_tuning<__int128_t, OffsetT, op_type::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
// {};
#endif

template <typename PolicyT, typename = void, typename = void>
struct ScanPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION ScanPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ScanPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::ScanPolicyT::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION ScanPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr PolicyWrapper<typename StaticPolicyT::ScanPolicyT> Scan()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::ScanPolicyT());
  }

  CUB_RUNTIME_FUNCTION constexpr CacheLoadModifier LoadModifier()
  {
    return StaticPolicyT::ScanPolicyT::LOAD_MODIFIER;
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION ScanPolicyWrapper<PolicyT> MakeScanPolicyWrapper(PolicyT policy)
{
  return ScanPolicyWrapper<PolicyT>{policy};
}

template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_hub
{
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm scan_transposed_load =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm scan_transposed_store =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, BLOCK_SCAN_RAKING>;
  };
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    // Titan X: 32.47B items/s @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct DefaultPolicy
  {
    using ScanPolicyT =
      AgentScanPolicy<128, 15, AccumT, scan_transposed_load, LOAD_DEFAULT, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy600
      : DefaultPolicy
      , ChainedPolicy<600, Policy600, Policy520>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentScanPolicy<Tuning::threads,
                       Tuning::items,
                       AccumT,
                       Tuning::load_algorithm,
                       LOAD_DEFAULT,
                       Tuning::store_algorithm,
                       BLOCK_SCAN_WARP_SCANS,
                       cub::detail::MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                       typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy::ScanPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy600>
  {
    using ScanPolicyT = decltype(select_agent_policy<sm80_tuning<AccumT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ScanPolicyT = decltype(select_agent_policy<sm90_tuning<AccumT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists that matches a benchmark, otherwise pick Policy900
    template <typename Tuning,
              typename IVT,
              // In the tuning benchmarks the Initial-, Input- and OutputType are the same. Let's check that the
              // accumulator type's size matches what we used during the benchmark since that has an impact (The
              // tunings also check later that it's a primitive type, so arithmetic impact is also comparable to the
              // benchmark). Input- and OutputType only impact loading and storing data (all arithmetic is done in the
              // accumulator type), so let's check that they are the same size and dispatch the size in the tunings.
              ::cuda::std::enable_if_t<sizeof(AccumT) == sizeof(::cuda::std::__accumulator_t<ScanOpT, IVT, IVT>)
                                         && sizeof(IVT) == sizeof(OutputValueT),
                                       int> = 0>
    static auto select_agent_policy100(int)
      -> AgentScanPolicy<Tuning::threads,
                         Tuning::items,
                         AccumT,
                         Tuning::load_algorithm,
                         Tuning::load_modifier,
                         Tuning::store_algorithm,
                         BLOCK_SCAN_WARP_SCANS,
                         MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                         typename Tuning::delay_constructor>;
    template <typename Tuning, typename IVT>
    static auto select_agent_policy100(long) -> typename Policy900::ScanPolicyT;

    using ScanPolicyT =
      decltype(select_agent_policy100<sm100_tuning<InputValueT, AccumT, OffsetT, classify_op<ScanOpT>()>, InputValueT>(
        0));
  };

  using MaxPolicy = Policy1000;
};
} // namespace scan
} // namespace detail

template <typename AccumT, typename ScanOpT = ::cuda::std::plus<>>
using DeviceScanPolicy CCCL_DEPRECATED_BECAUSE("This class is considered an implementation detail and it will be "
                                               "removed.") = detail::scan::policy_hub<int, int, AccumT, int, ScanOpT>;

CUB_NAMESPACE_END
""",
    'key-files/thrust/testing/scan.cu': r"""#include <thrust/detail/config.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scan.h>

#include <cuda/functional>
#include <cuda/std/array>

#include <unittest/unittest.h>

template <class Vector>
void TestScanSimple()
{
  using T = typename Vector::value_type;
  typename Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input = {1, 3, -2, 4, -5};
  Vector input_copy(input);

  // inclusive scan
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(0));
  result = {0, 1, 4, 2, 6};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with init
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with init and op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), T(-1), thrust::multiplies<T>());
  result = {-1, -3, 6, 24, -120};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with init and op
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3), thrust::plus<T>());
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inplace inclusive scan
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace inclusive scan with init and op
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin(), T(3), thrust::plus<T>());
  result = {4, 7, 5, 9, 4};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with implicit init=0
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
  result = {0, 1, 4, 2, 6};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);
}
DECLARE_VECTOR_UNITTEST(TestScanSimple);

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestInclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestInclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::inclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestInclusiveScanDispatchImplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestExclusiveScanDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchExplicit);

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

void TestExclusiveScanDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::exclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestExclusiveScanDispatchImplicit);

void TestInclusiveScan32()
{
  using T  = int;
  size_t n = 32;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestInclusiveScan32);

void TestExclusiveScan32()
{
  using T  = int;
  size_t n = 32;
  T init   = 13;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestExclusiveScan32);

template <class IntVector, class FloatVector>
void TestScanMixedTypes()
{
  // make sure we get types for default args and operators correct
  IntVector int_input{1, 2, 3, 4};
  FloatVector float_input{1.5, 2.5, 3.5, 4.5};
  IntVector int_output(4);
  FloatVector float_output(4);

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQUAL(int_output[0], 1); // in: 1.5 accum: 1.5f out: 1
  ASSERT_EQUAL(int_output[1], 4); // in: 2.5 accum: 4.0f out: 4
  ASSERT_EQUAL(int_output[2], 7); // in: 3.5 accum: 7.5f out: 7
  ASSERT_EQUAL(int_output[3], 12); // in: 4.5 accum: 12.f out: 12

  // float -> float with plus<int> operator (float accumulator)
  thrust::inclusive_scan(float_input.begin(), float_input.end(), float_output.begin(), thrust::plus<int>());
  ASSERT_EQUAL(float_output[0], 1.5f); // in: 1.5 accum: 1.5f out: 1.5f
  ASSERT_EQUAL(float_output[1], 3.0f); // in: 2.5 accum: 3.0f out: 3.0f
  ASSERT_EQUAL(float_output[2], 6.0f); // in: 3.5 accum: 6.0f out: 6.0f
  ASSERT_EQUAL(float_output[3], 10.0f); // in: 4.5 accum: 10.f out: 10.f

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQUAL(int_output[0], 0); // out: 0.0f  in: 1.5 accum: 1.5f
  ASSERT_EQUAL(int_output[1], 1); // out: 1.5f  in: 2.5 accum: 4.0f
  ASSERT_EQUAL(int_output[2], 4); // out: 4.0f  in: 3.5 accum: 7.5f
  ASSERT_EQUAL(int_output[3], 7); // out: 7.5f  in: 4.5 accum: 12.f

  // float -> int should use plus<> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float) 5.5);
  ASSERT_EQUAL(int_output[0], 5); // out: 5.5f  in: 1.5 accum: 7.0f
  ASSERT_EQUAL(int_output[1], 7); // out: 7.0f  in: 2.5 accum: 9.5f
  ASSERT_EQUAL(int_output[2], 9); // out: 9.5f  in: 3.5 accum: 13.0f
  ASSERT_EQUAL(int_output[3], 13); // out: 13.f  in: 4.5 accum: 17.4f

  // int -> float should use using plus<> operator and int accumulator by default
  thrust::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
  ASSERT_EQUAL(float_output[0], 1.f); // in: 1 accum: 1  out: 1
  ASSERT_EQUAL(float_output[1], 3.f); // in: 2 accum: 3  out: 3
  ASSERT_EQUAL(float_output[2], 6.f); // in: 3 accum: 6  out: 6
  ASSERT_EQUAL(float_output[3], 10.f); // in: 4 accum: 10 out: 10

  // int -> float + float init_value should use using plus<> operator and
  // float accumulator by default
  thrust::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float) 5.5);
  ASSERT_EQUAL(float_output[0], 5.5f); // out: 5.5f  in: 1 accum: 6.5f
  ASSERT_EQUAL(float_output[1], 6.5f); // out: 6.0f  in: 2 accum: 8.5f
  ASSERT_EQUAL(float_output[2], 8.5f); // out: 8.0f  in: 3 accum: 11.5f
  ASSERT_EQUAL(float_output[3], 11.5f); // out: 11.f  in: 4 accum: 15.5f
}
void TestScanMixedTypesHost()
{
  TestScanMixedTypes<thrust::host_vector<int>, thrust::host_vector<float>>();
}
DECLARE_UNITTEST(TestScanMixedTypesHost);
void TestScanMixedTypesDevice()
{
  TestScanMixedTypes<thrust::device_vector<int>, thrust::device_vector<float>>();
}
DECLARE_UNITTEST(TestScanMixedTypesDevice);

template <typename T>
struct TestScanWithOperator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), cuda::maximum<T>{});
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), cuda::maximum<T>{});
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), T(13), cuda::maximum<T>{});
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), T(13), cuda::maximum<T>{});
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestScanWithOperator, SignedIntegralTypes> TestScanWithOperatorInstance;

template <typename T>
struct TestScanWithOperatorToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(n);

    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), cuda::maximum<T>{});

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), cuda::maximum<T>{});

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);

    h_result = thrust::exclusive_scan(
      h_input.begin(), h_input.end(), thrust::make_discard_iterator(), T(13), cuda::maximum<T>{});

    d_result = thrust::exclusive_scan(
      d_input.begin(), d_input.end(), thrust::make_discard_iterator(), T(13), cuda::maximum<T>{});

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanWithOperatorToDiscardIterator,
                 unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestScanWithOperatorToDiscardIteratorInstance;

template <typename T>
struct TestScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
    ASSERT_EQUAL(d_output, h_output);

    // in-place scans
    h_output = h_input;
    d_output = d_input;
    thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::inclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);

    h_output = h_input;
    d_output = d_input;
    thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
    thrust::exclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
    ASSERT_EQUAL(d_output, h_output);
  }
};
VariableUnitTest<TestScan, IntegralTypes> TestScanInstance;

template <typename T>
struct TestScanToDiscardIterator
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> h_result =
      thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(n);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);

    h_result = thrust::exclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), (T) 11);

    d_result = thrust::exclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), (T) 11);

    ASSERT_EQUAL_QUIET(reference, h_result);
    ASSERT_EQUAL_QUIET(reference, d_result);
  }
};
VariableUnitTest<TestScanToDiscardIterator, unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestScanToDiscardIteratorInstance;

void TestScanMixedTypes()
{
  const unsigned int n = 113;

  thrust::host_vector<unsigned int> h_input = unittest::random_integers<unsigned int>(n);
  for (size_t i = 0; i < n; i++)
  {
    h_input[i] %= 10;
  }
  thrust::device_vector<unsigned int> d_input = h_input;

  thrust::host_vector<float> h_float_output(n);
  thrust::device_vector<float> d_float_output(n);
  thrust::host_vector<int> h_int_output(n);
  thrust::device_vector<int> d_int_output(n);

  // mixed input/output types
  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.5);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.5);
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
  ASSERT_EQUAL(d_float_output, h_float_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
  ASSERT_EQUAL(d_int_output, h_int_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.5);
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.5);
  ASSERT_EQUAL(d_int_output, h_int_output);
}
DECLARE_UNITTEST(TestScanMixedTypes);

template <typename T, unsigned int N>
void _TestScanWithLargeTypes()
{
  size_t n = (1024 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_input(n);
  thrust::host_vector<FixedVector<T, N>> h_output(n);

  for (size_t i = 0; i < h_input.size(); i++)
  {
    h_input[i] = FixedVector<T, N>(static_cast<T>(i));
  }

  thrust::device_vector<FixedVector<T, N>> d_input = h_input;
  thrust::device_vector<FixedVector<T, N>> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  ASSERT_EQUAL_QUIET(h_output, d_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), FixedVector<T, N>(0));
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), FixedVector<T, N>(0));

  ASSERT_EQUAL_QUIET(h_output, d_output);
}

void TestScanWithLargeTypes()
{
  _TestScanWithLargeTypes<int, 1>();

#if !defined(__QNX__)
  _TestScanWithLargeTypes<int, 8>();
  _TestScanWithLargeTypes<int, 64>();
#else
  KNOWN_FAILURE;
#endif
}
DECLARE_UNITTEST(TestScanWithLargeTypes);

template <typename T>
struct plus_mod3
{
  T* table;

  plus_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

template <typename Vector>
void TestInclusiveScanWithIndirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(data.begin(), data.end(), data.begin(), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestInclusiveScanWithIndirection);

template <typename T>
struct const_ref_plus_mod3
{
  T* table;

  const_ref_plus_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE const T& operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

template <typename Vector>
void TestInclusiveScanWithConstAccumulator()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(
    data.begin(), data.end(), data.begin(), const_ref_plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQUAL(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestInclusiveScanWithConstAccumulator);

struct only_set_when_expected_it
{
  long long expected;
  bool* flag;

  _CCCL_HOST_DEVICE only_set_when_expected_it operator++() const
  {
    return *this;
  }
  _CCCL_HOST_DEVICE only_set_when_expected_it operator*() const
  {
    return *this;
  }
  template <typename Difference>
  _CCCL_HOST_DEVICE only_set_when_expected_it operator+(Difference) const
  {
    return *this;
  }
  template <typename Index>
  _CCCL_HOST_DEVICE only_set_when_expected_it operator[](Index) const
  {
    return *this;
  }

  _CCCL_DEVICE void operator=(long long value) const
  {
    if (value == expected)
    {
      *flag = true;
    }
  }
};

THRUST_NAMESPACE_BEGIN
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type = long long;
  using reference  = only_set_when_expected_it;
};
THRUST_NAMESPACE_END

namespace std
{
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type = long long;
  using reference  = only_set_when_expected_it;
};
} // namespace std

void TestInclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude), thrust::raw_pointer_cast(has_executed)};

  thrust::inclusive_scan(thrust::device, begin, end, out);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestInclusiveScanWithBigIndexes()
{
  TestInclusiveScanWithBigIndexesHelper(30);
  TestInclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestInclusiveScanWithBigIndexesHelper(32);
  TestInclusiveScanWithBigIndexesHelper(33);
#endif
}

DECLARE_UNITTEST(TestInclusiveScanWithBigIndexes);

void TestExclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::exclusive_scan(thrust::device, begin, end, out, 0ll);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQUAL(has_executed_h, true);
}

void TestExclusiveScanWithBigIndexes()
{
  TestExclusiveScanWithBigIndexesHelper(30);
  TestExclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestExclusiveScanWithBigIndexesHelper(32);
  TestExclusiveScanWithBigIndexesHelper(33);
#endif
}

DECLARE_UNITTEST(TestExclusiveScanWithBigIndexes);

struct Int
{
  int i{};
  _CCCL_HOST_DEVICE explicit Int(int num)
      : i(num)
  {}
  _CCCL_HOST_DEVICE Int()
      : i{}
  {}
  _CCCL_HOST_DEVICE Int operator+(Int const& o) const
  {
    return Int{this->i + o.i};
  }
};

void TestInclusiveScanWithUserDefinedType()
{
  thrust::device_vector<Int> vec(5, Int{1});

  thrust::inclusive_scan(thrust::device, vec.cbegin(), vec.cend(), vec.begin());

  ASSERT_EQUAL(static_cast<Int>(vec.back()).i, 5);
}
DECLARE_UNITTEST(TestInclusiveScanWithUserDefinedType);

// Represents a permutation as a tuple of integers, see also: https://en.wikipedia.org/wiki/Permutation
// We need a distinct type (instead of an alias) for operator<< to be found via ADL
struct permutation_t : ::cuda::std::array<int, 5>
{
  permutation_t() = default;

  constexpr _CCCL_HOST_DEVICE permutation_t(int a, int b, int c, int d, int e)
      : ::cuda::std::array<int, 5>{a, b, c, d, e}
  {}

  friend std::ostream& operator<<(std::ostream& os, const permutation_t& p)
  {
    os << '{';
    for (std::size_t i = 0; i < p.size(); i++)
    {
      if (i > 0)
      {
        os << ", ";
      }
      os << p[i];
    }
    return os << '}';
  }
};

// Composes two permutations. This operation is associative, but not commutative.
struct composition_op_t
{
  _CCCL_HOST_DEVICE permutation_t operator()(permutation_t lhs, permutation_t rhs) const
  {
    permutation_t result;
    for (std::size_t i = 0; i < lhs.size(); i++)
    {
      result[i] = rhs[lhs[i]];
    }
    return result;
  }
};

void TestInclusiveScanWithNonCommutativeOp()
{
  const thrust::device_vector<permutation_t> input = {
    {3, 2, 0, 1, 4},
    {2, 4, 0, 1, 3},
    {3, 2, 1, 4, 0},
    {4, 3, 1, 0, 2},
    {0, 3, 2, 4, 1},
    {3, 2, 1, 0, 4},
    {3, 4, 1, 2, 0},
    {4, 2, 1, 0, 3},
    {4, 0, 1, 3, 2},
    {0, 2, 3, 1, 4}};
  thrust::device_vector<permutation_t> output(10);
  constexpr auto identity = permutation_t{0, 1, 2, 3, 4};

  thrust::inclusive_scan(input.begin(), input.end(), output.begin(), composition_op_t{});
  ASSERT_EQUAL(
    output,
    (thrust::device_vector<permutation_t>{
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3},
      {4, 0, 3, 2, 1}}));

  thrust::exclusive_scan(input.begin(), input.end(), output.begin(), identity, composition_op_t{});
  ASSERT_EQUAL(
    output,
    (thrust::device_vector<permutation_t>{
      {0, 1, 2, 3, 4},
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3}}));
}
DECLARE_UNITTEST(TestInclusiveScanWithNonCommutativeOp);
""",
    'key-files/thrust/thrust/system/cuda/detail/async/inclusive_scan.h': r"""/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER

#  include <thrust/system/cuda/config.h>

#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/async/customization.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/system/cuda/future.h>

#  include <cuda/std/type_traits>

#  include <type_traits>

// TODO specialize for thrust::plus to use e.g. InclusiveSum instead of IncScan

_CCCL_SUPPRESS_DEPRECATED_PUSH
THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cuda
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIt, typename Size, typename OutputIt, typename BinaryOp>
unique_eager_event
async_inclusive_scan_n(execution_policy<DerivedPolicy>& policy, ForwardIt first, Size n, OutputIt out, BinaryOp op)
{
  using AccumT     = typename thrust::iterator_traits<ForwardIt>::value_type;
  using Dispatch32 = cub::DispatchScan<ForwardIt, OutputIt, BinaryOp, cub::NullType, std::uint32_t, AccumT>;
  using Dispatch64 = cub::DispatchScan<ForwardIt, OutputIt, BinaryOp, cub::NullType, std::uint64_t, AccumT>;

  auto const device_alloc = get_async_device_allocator(policy);
  unique_eager_event ev;

  // Determine temporary device storage requirements.
  cudaError_t status;
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      n,
      (nullptr, tmp_size, first, out, op, cub::NullType{}, n_fixed, nullptr));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Allocate temporary storage.
  auto content        = uninitialized_allocate_unique_n<std::uint8_t>(device_alloc, tmp_size);
  void* const tmp_ptr = raw_pointer_cast(content.get());

  // Set up stream with dependencies.
  cudaStream_t const user_raw_stream = thrust::cuda_cub::stream(policy);

  if (thrust::cuda_cub::default_stream() != user_raw_stream)
  {
    ev = make_dependent_event(
      std::tuple_cat(std::make_tuple(std::move(content), unique_stream(nonowning, user_raw_stream)),
                     extract_dependencies(std::move(thrust::detail::derived_cast(policy)))));
  }
  else
  {
    ev = make_dependent_event(std::tuple_cat(
      std::make_tuple(std::move(content)), extract_dependencies(std::move(thrust::detail::derived_cast(policy)))));
  }

  // Run scan.
  {
    THRUST_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      n,
      (tmp_ptr, tmp_size, first, out, op, cub::NullType{}, n_fixed, user_raw_stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
  }

  return ev;
}

template <typename DerivedPolicy,
          typename ForwardIt,
          typename Size,
          typename OutputIt,
          typename InitialValueType,
          typename BinaryOp>
unique_eager_event async_inclusive_scan_n(
  execution_policy<DerivedPolicy>& policy, ForwardIt first, Size n, OutputIt out, InitialValueType init, BinaryOp op)
{
  using InputValueT = cub::detail::InputValue<InitialValueType>;
  using AccumT      = typename ::cuda::std::
    __accumulator_t<BinaryOp, typename ::cuda::std::iterator_traits<ForwardIt>::value_type, InitialValueType>;

  using Dispatch32 = cub::DispatchScan<
    ForwardIt,
    OutputIt,
    BinaryOp,
    InputValueT,
    std::int32_t,
    AccumT,
    cub::detail::scan::
      policy_hub<cub::detail::value_t<ForwardIt>, cub::detail::value_t<OutputIt>, AccumT, std::int32_t, BinaryOp>,
    cub::ForceInclusive::Yes>;
  using Dispatch64 = cub::DispatchScan<
    ForwardIt,
    OutputIt,
    BinaryOp,
    InputValueT,
    std::int64_t,
    AccumT,
    cub::detail::scan::
      policy_hub<cub::detail::value_t<ForwardIt>, cub::detail::value_t<OutputIt>, AccumT, std::int64_t, BinaryOp>,
    cub::ForceInclusive::Yes>;

  InputValueT init_value(init);

  auto const device_alloc = get_async_device_allocator(policy);
  unique_eager_event ev;

  // Determine temporary device storage requirements.
  cudaError_t status;
  size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      n,
      (nullptr, tmp_size, first, out, op, init_value, n_fixed, nullptr));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Allocate temporary storage.
  auto content        = uninitialized_allocate_unique_n<std::uint8_t>(device_alloc, tmp_size);
  void* const tmp_ptr = raw_pointer_cast(content.get());

  // Set up stream with dependencies.
  cudaStream_t const user_raw_stream = thrust::cuda_cub::stream(policy);

  if (thrust::cuda_cub::default_stream() != user_raw_stream)
  {
    ev = make_dependent_event(
      std::tuple_cat(std::make_tuple(std::move(content), unique_stream(nonowning, user_raw_stream)),
                     extract_dependencies(std::move(thrust::detail::derived_cast(policy)))));
  }
  else
  {
    ev = make_dependent_event(std::tuple_cat(
      std::make_tuple(std::move(content)), extract_dependencies(std::move(thrust::detail::derived_cast(policy)))));
  }

  // Run scan.
  {
    THRUST_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      n,
      (tmp_ptr, tmp_size, first, out, op, init_value, n_fixed, user_raw_stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
  }

  return ev;
}

} // namespace detail
} // namespace cuda
} // namespace system

namespace cuda_cub
{

// ADL entry point.
template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename OutputIt, typename BinaryOp>
auto async_inclusive_scan(
  execution_policy<DerivedPolicy>& policy, ForwardIt first, Sentinel&& last, OutputIt&& out, BinaryOp&& op)
  THRUST_RETURNS(thrust::system::cuda::detail::async_inclusive_scan_n(
    policy, first, thrust::distance(first, THRUST_FWD(last)), THRUST_FWD(out), THRUST_FWD(op)))

  // ADL entry point.
  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename BinaryOp>
  auto async_inclusive_scan(
    execution_policy<DerivedPolicy>& policy,
    ForwardIt first,
    Sentinel&& last,
    OutputIt&& out,
    InitialValueType&& init,
    BinaryOp&& op)
    THRUST_RETURNS(thrust::system::cuda::detail::async_inclusive_scan_n(
      policy, first, thrust::distance(first, THRUST_FWD(last)), THRUST_FWD(out), THRUST_FWD(init), THRUST_FWD(op)))

} // namespace cuda_cub

_CCCL_SUPPRESS_DEPRECATED_POP
THRUST_NAMESPACE_END

#endif // _CCCL_CUDA_COMPILER
""",
    'key-files/thrust/thrust/system/cuda/detail/scan.h': r"""/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER

#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_scan.cuh>

#  include <thrust/detail/integer_math.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/type_traits.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>

#  include <cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt result, ScanOp scan_op)
{
  using AccumT     = typename thrust::iterator_traits<InputIt>::value_type;
  using Dispatch32 = cub::DispatchScan<InputIt, OutputIt, ScanOp, cub::NullType, std::uint32_t, AccumT>;
  using Dispatch64 = cub::DispatchScan<InputIt, OutputIt, ScanOp, cub::NullType, std::uint64_t, AccumT>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, cub::NullType{}, num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, cub::NullType{}, num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "inclusive_scan failed to synchronize");
  }

  return result + num_items;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename InitValueT, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  InitValueT init,
  ScanOp scan_op)
{
  using InputValueT  = cub::detail::InputValue<InitValueT>;
  using ValueT       = cub::detail::value_t<InputIt>;
  using OutputValueT = cub::detail::value_t<OutputIt>;
  using AccumT       = ::cuda::std::__accumulator_t<ScanOp, ValueT, InitValueT>;

  using Dispatch32 =
    cub::DispatchScan<InputIt,
                      OutputIt,
                      ScanOp,
                      InputValueT,
                      std::uint32_t,
                      AccumT,
                      cub::detail::scan::policy_hub<ValueT, OutputValueT, AccumT, std::int32_t, ScanOp>,
                      cub::ForceInclusive::Yes>;
  using Dispatch64 =
    cub::DispatchScan<InputIt,
                      OutputIt,
                      ScanOp,
                      InputValueT,
                      std::uint64_t,
                      AccumT,
                      cub::detail::scan::policy_hub<ValueT, OutputValueT, AccumT, std::int64_t, ScanOp>,
                      cub::ForceInclusive::Yes>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "inclusive_scan failed to synchronize");
  }

  return result + num_items;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename InitValueT, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  InitValueT init,
  ScanOp scan_op)
{
  using InputValueT = cub::detail::InputValue<InitValueT>;
  using Dispatch32  = cub::DispatchScan<InputIt, OutputIt, ScanOp, InputValueT, std::uint32_t, InitValueT>;
  using Dispatch64  = cub::DispatchScan<InputIt, OutputIt, ScanOp, InputValueT, std::uint64_t, InitValueT>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for exclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching exclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "exclusive_scan failed to synchronize");
  }

  return result + num_items;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::inclusive_scan_n_impl(policy, first, num_items, result, init, scan_op);),
    (result =
       thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, init, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt result, ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::inclusive_scan_n_impl(policy, first, num_items, result, scan_op);),
    (result = thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::inclusive_scan_n(policy, first, num_items, result, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::inclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt>
_CCCL_HOST_DEVICE OutputIt
inclusive_scan(thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  return thrust::cuda_cub::inclusive_scan(policy, first, last, result, thrust::plus<>{});
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::exclusive_scan_n_impl(policy, first, num_items, result, init, scan_op);),
    (result =
       thrust::exclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, init, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::exclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt, typename T>
_CCCL_HOST_DEVICE OutputIt exclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, T init)
{
  return thrust::cuda_cub::exclusive_scan(policy, first, last, result, init, thrust::plus<>{});
}

template <typename Derived, typename InputIt, typename OutputIt>
_CCCL_HOST_DEVICE OutputIt
exclusive_scan(thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  using init_type = typename thrust::iterator_traits<InputIt>::value_type;
  return cuda_cub::exclusive_scan(policy, first, last, result, init_type{});
};

} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/scan.h>

#endif // NVCC
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
