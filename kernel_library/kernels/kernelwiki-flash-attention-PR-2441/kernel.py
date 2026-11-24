"""KernelWiki artifact PR-2441 (Dao-AILab/flash-attention)

Reference bundle — a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission"""

_SOURCES = {
    'key-files/flash_attn/cute/cute_dsl_utils.py': r'''# Copyright (c) 2025, Tri Dao.

from typing import Tuple
from functools import lru_cache

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.runtime import from_dlpack

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI, tailored to FlexAttention aux tensors.
    This allows the user to specify alignment and leading dimension for aux tensors used in
    custom score_mod callables.
    """
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def get_aux_tensor_metadata(aux_tensors):
    return tuple(
        (
            getattr(t, "__assumed_align__", 0),
            getattr(t, "__leading_dim__", -1),
            hasattr(t, "__leading_dim__"),
        )
        for t in aux_tensors
    )


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())


# credit: monellz (https://github.com/NVIDIA/cutlass/issues/2658#issuecomment-3630564264)
def dump_kernel_attributes(compiled_kernel):
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo
    import torch

    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    assert cubin_data is not None, "cubin_data is None, need '--keep-cubin' option when compiling"
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(driver.cuLibraryEnumerateKernels(1, cuda_library))
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    # more metrics: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print("--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print("--- End Kernel Info ---")
''',
    'key-files/flash_attn/cute/flash_fwd_mla_sm100.py': r'''import math
import time
from functools import partial
from typing import Callable, Optional

import torch
import torch.utils.benchmark as benchmark

import cuda.bindings.driver as cuda

import cutlass
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
import cutlass.cute as cute
from cutlass import Float32, BFloat16, Int64, Int32, Uint32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from quack import copy_utils, layout_utils

from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.mask import AttentionMask
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.tile_scheduler import (
    ClcState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from flash_attn.cute.fa_logging import fa_log, fa_printf
from flash_attn.cute.utils import smid

from flash_attn.cute.topk_gather_kv import CpasyncGatherKVManager

from flash_attn.cute.testing import attention_ref

from flash_attn.cute.named_barrier import NamedBarrierFwdSm100_MLA2CTA

from flash_attn.cute.cute_dsl_utils import dump_kernel_attributes

class FlashAttentionMLAForwardSm100:
    def __init__(
        self,
        is_causal: bool = False,
        use_cpasync_load_KV: bool = False,
        topk_length: int = 2048,
        is_topk_gather: bool = True,
        pack_gqa: bool = False,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        is_varlen_q: bool = False,
        disable_bitmask: bool = False,
        use_clc_scheduler: bool = True,
    ):
        self.is_causal = is_causal
        self.is_local = False
        self.pack_gqa = pack_gqa
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv
        self.is_varlen_q = is_varlen_q
        self.use_tma_O = True
        self.use_cpasync_load_KV = use_cpasync_load_KV
        self.use_tma_KV = not use_cpasync_load_KV
        self.topk_length = topk_length
        self.is_topk_gather = is_topk_gather
        if is_topk_gather:
            assert pack_gqa
            assert qhead_per_kvhead == 128, "require MQA 128 for DSA path"
            assert use_cpasync_load_KV
        # user-provided option if topk indices guaranteed in bounds
        self.disable_bitmask = disable_bitmask
        
        # ==== tile scheduler ====
        self.is_persistent = False
        self.use_clc_scheduler = use_clc_scheduler and not is_varlen_q
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC

        if const_expr(is_varlen_q):
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(1, f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}")

        # ==== thread info ====
        self.num_softmax_threads = 128
        self.num_epilogue_threads = 128
        self.num_load_threads = 32
        self.num_mma_threads = 32
        self.num_empty_threads = 32 if use_cpasync_load_KV else 64
        self.num_relay_threads = 32 if use_cpasync_load_KV else 0
        self.num_cpasync_load_threads = 128 if use_cpasync_load_KV else 0
        self.num_threads = (
            self.num_softmax_threads
            + self.num_epilogue_threads
            + self.num_load_threads
            + self.num_mma_threads
            + self.num_empty_threads
            + self.num_relay_threads
            + self.num_cpasync_load_threads
        )
        self.num_warps = self.num_threads // 32
        assert self.num_warps == 12 or self.num_warps == 16
        self.softmax_warp_indices = (0, 1, 2, 3,)
        self.epilogue_warp_indices = (4, 5, 6, 7,)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.relay_warp_id = 11
        self.empty_warp_ids = tuple(
            w for w, active in [
                (self.relay_warp_id,         not use_cpasync_load_KV),
                (self.clc_scheduler_warp_id, not self.use_clc_scheduler),
            ] if active
        )
        self.cpasync_load_warp_indices = (12, 13, 14, 15,)

        # ==== register usage ====
        if self.num_warps == 16:
            self.num_regs_load = 80
            self.num_regs_mma = 80
            self.num_regs_softmax = 208
            self.num_regs_epilogue = 128
            self.num_regs_cpasync = 96 if self.use_cpasync_load_KV else 0
            self.num_regs_other = 48
        else:
            self.num_regs_load = 168 - 40
            self.num_regs_mma = 168 - 40
            self.num_regs_softmax = 168 + 80
            self.num_regs_epilogue = 168 - 40
            self.num_regs_cpasync = 0
            self.num_regs_other = 48

        self.num_regs_per_thread = 168 if self.num_warps == 12 else 128
        self.num_regs_total = 504 if self.num_warps == 12 else 512
        
        assert self.num_regs_mma + self.num_regs_softmax + self.num_regs_epilogue + self.num_regs_cpasync <= self.num_regs_total

        # ==== 2cta info ====
        self.use_2cta_instrs = True
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cta_group_size = 2
        self.cluster_shape_mn = (2, 1,)
        self.cluster_shape_mnk = (2, 1, 1,)

        # ==== problem shape info ====
        self.hdim = hdim
        self.hdimv = hdimv
        self.cta_tile_m = 64
        self.cluster_tile_m = self.cta_group_size * self.cta_tile_m
        self.tile_n = 128
        assert (
            pack_gqa is False
            or self.cluster_tile_m % qhead_per_kvhead == 0
            or qhead_per_kvhead % self.cluster_tile_m == 0
        )
        self.num_hdimv_splits = 2  # split hdimv in half for our Qv @ V^T and P @ V mmas.
        assert hdimv % 32 == 0
        assert self.topk_length % (self.tile_n * 2) == 0 or not self.is_topk_gather
        self.epi_tile = (self.cta_tile_m, self.hdimv//self.num_hdimv_splits)

        # ==== MMA info ====
        self.mma_tiler_QK = (self.cluster_tile_m, self.tile_n, self.hdim,)
        self.mma_tiler_QviVi = (self.cluster_tile_m, self.tile_n, self.hdimv//self.num_hdimv_splits,)
        self.mma_tiler_PVti = (self.cluster_tile_m, self.hdimv // self.num_hdimv_splits, self.tile_n,)
        self.major_mode_Q = tcgen05.OperandMajorMode.K
        self.major_mode_Qvi = tcgen05.OperandMajorMode.K
        self.major_mode_K = tcgen05.OperandMajorMode.K
        self.major_mode_Vi = tcgen05.OperandMajorMode.K
        self.major_mode_Vti = tcgen05.OperandMajorMode.MN
        self.major_mode_P = tcgen05.OperandMajorMode.K
        self.operand_source_Q = tcgen05.OperandSource.SMEM
        self.operand_source_Qvi = tcgen05.OperandSource.SMEM
        self.operand_source_P = tcgen05.OperandSource.SMEM

        # ==== pipeline info ====
        self.num_stages_Q = 1
        self.num_stages_K = 1
        self.num_stages_Qvi = 1
        self.num_stages_Vi = 2
        self.num_stages_S = 2
        self.num_stages_P = 1
        self.num_stages_Oi = 1
        self.num_stages_sm_stats = 2
        self.num_stages_bitmask = 4
        assert self.num_stages_S == 2, "mainloops expect 2 stages for S"

        # ==== dtype info ====
        self.dtype_acc = Float32

        # ==== TMEM info ====
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_cols_S = self.tile_n // self.cta_group_size
        self.tmem_cols_Oi = (self.hdimv // self.num_hdimv_splits) // self.cta_group_size
        self.tmem_offset_S = [self.tmem_cols_S*stage for stage in range(self.num_stages_S)]  # allocate 64 TMEM columns for each stage of S
        self.tmem_offset_O0 = self.tmem_cols_S * self.num_stages_S 
        self.tmem_offset_O1 = self.tmem_offset_O0 + self.tmem_cols_Oi
        self.tmem_offsets_O = [self.tmem_offset_O0, self.tmem_offset_O1]
        self.total_tmem = self.tmem_offset_O1 + self.tmem_cols_Oi
        assert self.total_tmem <= self.tmem_alloc_cols, f"Total TMEM columns allocated {self.total_tmem} exceeds capacity {self.tmem_alloc_cols}"


    def _get_shared_storage_cls(self):
        self.buffer_align_bytes = 1024
        
        def smem_struct_align(dtype, staged_layout):
            return cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(staged_layout)],
                self.buffer_align_bytes,
            ]

        def mbar_struct(num_stages):
            return cute.struct.MemRange[Int64, 2 * num_stages]

        (sQ_struct, sK_struct, sQv0_struct, sQv1_struct, sV0_struct, sV1_struct, sP_struct) = (
            smem_struct_align(dtype, layout) for dtype, layout in [
                (self.dtype_Q,  self.sQ_layout_staged),
                (self.dtype_K,  self.sK_layout_staged),
                (self.dtype_Qv, self.sQvi_layout_staged),
                (self.dtype_Qv, self.sQvi_layout_staged),
                (self.dtype_V,  self.sVi_layout_staged),
                (self.dtype_V,  self.sVi_layout_staged),
                (self.dtype_P,  self.sP_layout_staged),
            ]
        )
        sStats_struct = cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)]
        sScale_struct = cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)]
        sBitmask_struct = cute.struct.MemRange[Uint32, cute.cosize(self.sBitmask_layout)]

        (mbar_ptr_Q_struct, mbar_ptr_K_struct, mbar_ptr_Qv0_struct, mbar_ptr_Qv1_struct,
         mbar_ptr_V0_struct, mbar_ptr_V1_struct, mbar_ptr_S_struct, mbar_ptr_P_struct,
         mbar_ptr_O0_struct, mbar_ptr_O1_struct, mbar_sm_stats_struct,
         mbar_bitmask_struct) = (
            mbar_struct(n) for n in [
                self.num_stages_Q, self.num_stages_K,
                self.num_stages_Qvi, self.num_stages_Qvi,
                self.num_stages_Vi, self.num_stages_Vi,
                self.num_stages_S, self.num_stages_P,
                self.num_stages_Oi, self.num_stages_Oi,
                self.num_stages_sm_stats,
                self.num_stages_bitmask,
            ]
        )
        mbar_ptr_tmem_dealloc_struct = Int64
        tmem_holding_buf_struct = Int32

        self.sched_stages = 1
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_Qv0: mbar_ptr_Qv0_struct
            mbar_ptr_Qv1: mbar_ptr_Qv1_struct
            mbar_ptr_V0: mbar_ptr_V0_struct
            mbar_ptr_V1: mbar_ptr_V1_struct
            mbar_ptr_S: mbar_ptr_S_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_O0: mbar_ptr_O0_struct
            mbar_ptr_O1: mbar_ptr_O1_struct
            mbar_ptr_K_cpasync: mbar_ptr_K_struct
            mbar_ptr_V0_cpasync: mbar_ptr_V0_struct
            mbar_ptr_V1_cpasync: mbar_ptr_V1_struct
            mbar_ptr_sm_stats: mbar_sm_stats_struct
            mbar_ptr_bitmask: mbar_bitmask_struct
            mbar_ptr_tmem_dealloc: mbar_ptr_tmem_dealloc_struct
            tmem_holding_buf: tmem_holding_buf_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO_empty_mbar_ptr: cutlass.Int64

            sRowMax: sStats_struct
            sRowSum: sStats_struct
            sScale: sScale_struct
            sBitmask: sBitmask_struct
            sQv0: sQv0_struct
            sQv1: sQv1_struct
            sQ: sQ_struct
            sK: sK_struct
            sV0: sV0_struct
            sV1: sV1_struct
            sP: sP_struct

        # print("smem bytes = ", SharedStorage.size_in_bytes())

        return SharedStorage
 
    
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,   # (b, s_q, h, d)  or (total_q, h, d) if there is cu_seqlens_q
        mQv: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,   # (b_k, s_k, h_k, d)  or (total_k, h_k, d)  if there is cu_seqlens_k or (num_pages, page_size, h_k, d)  if there is page_table
        mV: cute.Tensor,   # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,   # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],  # (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None, # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None, # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,   # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,   # (b)
        mIndexTopk: Optional[cute.Tensor] = None,  # (b, s_q, topk) or (total_q, topk) if there is cu_seqlens_q
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):  
        # ==== asserts for unimplemented features ====
        assert mPageTable is None, "page table tbd for MLA"

        # ==== dtype info ====
        self.dtype_Q = mQ.element_type
        self.dtype_K = mK.element_type
        self.dtype_Qv = mQv.element_type
        self.dtype_V = mV.element_type
        self.dtype_P = mV.element_type
        self.dtype_O = mO.element_type

        # ==== Prepare Tensors ====
        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQ, mQv, mK, mV, mO = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            for mX in (mQ, mQv, mK, mV, mO)
        ]

        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h) or
        # (num_pages, page_size, h_k, d) -> (page_size, d, h_k, num_pages)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ, mQv, mO = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            for mX in (mQ, mQv, mO)
        ]
        mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            for mX in (mK, mV)
        ]
        # (s_k, dv, h_k, b)  -> (dv, s_k, h_k, b) or
        # (total_k, dv, h_k) -> (dv, total_k, h_k)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mVt = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )
        # (b, h, s_q) -> (s_q, h, b) or (h, total_q) -> (total_q, h)
        # (b, s_q, topk) -> (topk, s_q, b) or (total_q, topk) -> (topk, total_q)
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mIndexTopk = (
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=LSE_layout_transpose))
            if t is not None else None for t in (mLSE, mIndexTopk)
        )
        topk_length_dynamic = mIndexTopk.shape[0] if mIndexTopk is not None else None

        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        mO_og = mO
        if const_expr(self.pack_gqa):
            mQ, mQv, mO = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                for mX in (mQ, mQv, mO)
            ]
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        def split_hdimv(m, dim: int):
            """Re-tile mode `dim` of tensor `m` from hdimv into (hdimv//S, S),
            and return (slice0, slice1) where slice_i selects chunk i."""
            S = self.num_hdimv_splits
            chunk = self.hdimv // S
            split_shape  = (*m.shape[:dim],  (chunk, S), *m.shape[dim+1:])
            split_stride = (*m.layout.stride[:dim], (1, chunk), *m.layout.stride[dim+1:])
            split = cute.make_tensor(m.iterator, cute.make_layout(split_shape, stride=split_stride))
            ndim = len(split.shape)
            slices = [
                split[(*([None] * dim), (None, i), *([None] * (ndim - dim - 1)))]
                for i in range(S)
            ]
            return slices

        # (seqlen_q, hdimv//2, nheads, batch) or (total_q, hdimv//2, nheads)
        mQv0, mQv1 = split_hdimv(mQv, dim=1)
        mV0,  mV1  = split_hdimv(mV,  dim=1)
        # (hdimv//2, seqlen_k, nheads_k, batch) or (hdimv//2, total_k, nheads_k)
        mVt0, mVt1 = split_hdimv(mVt, dim=0)

        # ==== Prepare MMAs ====
        # (local_var, dtype_a, major_a, major_b, mma_tiler, operand_source_a)
        _mma_specs = [
            ("tiled_mma_QK",    self.dtype_Q,  self.major_mode_Q,   self.major_mode_K,   self.mma_tiler_QK,    self.operand_source_Q),
            ("tiled_mma_QviVi", self.dtype_Qv, self.major_mode_Qvi, self.major_mode_Vi,  self.mma_tiler_QviVi, self.operand_source_Qvi),
            ("tiled_mma_PVti",  self.dtype_P,  self.major_mode_P,   self.major_mode_Vti, self.mma_tiler_PVti,  self.operand_source_P),
        ]
        tiled_mma_QK, tiled_mma_QviVi, tiled_mma_PVti = (
            sm100_utils.make_trivial_tiled_mma(
                dtype_a, major_a, major_b, self.dtype_acc, self.cta_group, mma_tiler[:2], operand_source_a,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, operand_source_a in _mma_specs
        )

        # ==== Prepare SMEM layouts and TMAs ====
        # (attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages)
        _smem_layout_specs = [
            ("sQ_layout",   sm100_utils.make_smem_layout_a, tiled_mma_QK,    self.mma_tiler_QK,    self.dtype_Q,  self.num_stages_Q),
            ("sK_layout",   sm100_utils.make_smem_layout_b, tiled_mma_QK,    self.mma_tiler_QK,    self.dtype_K,  self.num_stages_K),
            ("sQvi_layout", sm100_utils.make_smem_layout_a, tiled_mma_QviVi, self.mma_tiler_QviVi, self.dtype_Qv, self.num_stages_Qvi),
            ("sVi_layout",  sm100_utils.make_smem_layout_b, tiled_mma_QviVi, self.mma_tiler_QviVi, self.dtype_V,  self.num_stages_Vi),
            ("sVti_layout", sm100_utils.make_smem_layout_b, tiled_mma_PVti,  self.mma_tiler_PVti,  self.dtype_V,  self.num_stages_Vi),
            ("sP_layout",   sm100_utils.make_smem_layout_a, tiled_mma_PVti,  self.mma_tiler_PVti,  self.dtype_P,  self.num_stages_P),
        ]
        for attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages in _smem_layout_specs:
            ab_kwarg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            staged = make_fn(
                tiled_mma=tiled_mma,
                mma_tiler_mnk=mma_tiler,
                num_stages=num_stages,
                **{ab_kwarg: dtype},
            )
            setattr(self, f"{attr}_staged", staged)
            setattr(self, attr, cute.select(staged, mode=[0, 1, 2]))

        self.sStats_layout = cute.make_layout((self.cta_tile_m, self.cta_group_size))
        self.sScale_layout = cute.make_layout((self.cta_tile_m, self.num_stages_sm_stats))
        self.sBitmask_layout = cute.make_layout((self.tile_n//32, self.num_stages_bitmask))

        for attr, dtype, layout in [
            ("tma_copy_bytes_Q",   self.dtype_Q,  self.sQ_layout),
            ("tma_copy_bytes_K",   self.dtype_K,  self.sK_layout),
            ("tma_copy_bytes_Qvi", self.dtype_Qv, self.sQvi_layout),
            ("tma_copy_bytes_Vi",  self.dtype_V,  self.sVi_layout),
        ]:
            setattr(self, attr, cute.size_in_bytes(dtype, layout) * self.cta_group_size)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QK.thr_id.shape,)
        )
        cta_shape = cta_layout_vmnk.shape

        def make_tma(make_fn, mX, smem_layout, mma_tiler, tiled_mma):
            return make_fn(tma_load_op, mX, smem_layout, mma_tiler, tiled_mma, cta_shape)

        A, B = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B

        # (atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma, kv_only)
        _tma_specs = [
            ("tma_atom_Q",   "tma_tensor_Q",   A, mQ,   self.sQ_layout,   self.mma_tiler_QK,    tiled_mma_QK,    False),
            ("tma_atom_Qv0", "tma_tensor_Qv0", A, mQv0, self.sQvi_layout, self.mma_tiler_QviVi, tiled_mma_QviVi, False),
            ("tma_atom_Qv1", "tma_tensor_Qv1", A, mQv1, self.sQvi_layout, self.mma_tiler_QviVi, tiled_mma_QviVi, False),
            ("tma_atom_K",   "tma_tensor_K",   B, mK,   self.sK_layout,   self.mma_tiler_QK,    tiled_mma_QK,    True),
            ("tma_atom_V0",  "tma_tensor_V0",  B, mV0,  self.sVi_layout,  self.mma_tiler_QviVi, tiled_mma_QviVi, True),
            ("tma_atom_V1",  "tma_tensor_V1",  B, mV1,  self.sVi_layout,  self.mma_tiler_QviVi, tiled_mma_QviVi, True),
            ("tma_atom_Vt0", "tma_tensor_Vt0", B, mVt0, self.sVti_layout, self.mma_tiler_PVti,  tiled_mma_PVti,  True),
            ("tma_atom_Vt1", "tma_tensor_Vt1", B, mVt1, self.sVti_layout, self.mma_tiler_PVti,  tiled_mma_PVti,  True),
        ]
        _tmas = {}
        for atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma, kv_only in _tma_specs:
            _tmas[atom_name], _tmas[tensor_name] = (
                make_tma(make_fn, m, smem_layout, mma_tiler, tiled_mma)
                if const_expr(not kv_only or self.use_tma_KV)
                else (None, None)
            )

        (tma_atom_Q,   tma_tensor_Q,
         tma_atom_Qv0, tma_tensor_Qv0,
         tma_atom_Qv1, tma_tensor_Qv1,
         tma_atom_K,   tma_tensor_K,
         tma_atom_V0,  tma_tensor_V0,
         tma_atom_V1,  tma_tensor_V1,
         tma_atom_Vt0, tma_tensor_Vt0,
         tma_atom_Vt1, tma_tensor_Vt1) = _tmas.values()

        # ==== Set up Oi smem -> gmem tma store ====

        self.overlap_sO_sV = True
        if const_expr(self.overlap_sO_sV):
            num_stages_sO = self.num_hdimv_splits * self.num_stages_Vi
        else:
            num_stages_sO = self.num_hdimv_splits
        sO_layout = sm100_utils.make_smem_layout_epi(
            self.dtype_O, self.o_layout, self.epi_tile, num_stages_sO
        )
        self.ragged_tma_O = (
            self.use_tma_O
            and self.is_varlen_q
            and self.pack_gqa
            and self.cta_tile_m % self.qhead_per_kvhead == 0
        )
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.ragged_tma_O)
            else cpasync.make_tiled_tma_atom
        )
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.ragged_tma_O) else mO
            if const_expr(self.ragged_tma_O):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                tma_store_op, mO_tma, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
        else:
            tma_atom_O = None
            tma_tensor_O = None

        # ==== Set up Oi rmem -> gmem copy ====
        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_O,
            num_bits_per_copy=universal_copy_bits,
        )
        thread_layout_O_r2g = cute.make_layout((64, 2), stride=(1, 64))
        value_layout_O_r2g = cute.make_layout(
            (1, self.hdimv//self.num_hdimv_splits//self.cta_group_size)
        )
        tiled_copy_O_r2g = cute.make_tiled_copy_tv(
            atom=atom_universal_copy,
            thr_layout=thread_layout_O_r2g,
            val_layout=value_layout_O_r2g,
        )

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()

        # ==== Tile scheduler ====

        TileScheduler = self.TileScheduler
        
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits=1, # todo: split_kv
            seqlen_k=cute.size(mK.shape[0]), # todo: page table
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.cta_tile_m, self.tile_n,),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype_K.width // 8,
            is_persistent=self.is_persistent,
            # lpt=self.is_causal or self.is_local,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        fa_printf(1, "grid = {}", grid_dim)
        
        # ==== Named Barrier ====
        self.cpasync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Cpasync),
            num_threads=self.num_cpasync_load_threads,
        )
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )
        # softmax -> correction
        self.sm_stats_barrier_full = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsFull),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )
        self.sm_stats_barrier_empty = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsEmpty),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ==== Launch kernel ====
        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qv0,
            tma_tensor_Qv1,
            tma_tensor_K if self.use_tma_KV else mK,
            tma_tensor_V0 if self.use_tma_KV else mV0,
            tma_tensor_V1 if self.use_tma_KV else mV1,
            tma_tensor_Vt0 if self.use_tma_KV else mVt0,
            tma_tensor_Vt1 if self.use_tma_KV else mVt1,
            tma_tensor_O if self.use_tma_O else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mIndexTopk,
            tma_atom_Q,
            tma_atom_Qv0,
            tma_atom_Qv1,
            tma_atom_K,
            tma_atom_V0,
            tma_atom_V1,
            tma_atom_Vt0,
            tma_atom_Vt1,
            tma_atom_O,
            tiled_copy_O_r2g,
            self.sQ_layout_staged,
            self.sK_layout_staged,
            self.sQvi_layout_staged,
            self.sVi_layout_staged,
            self.sVti_layout_staged,
            self.sP_layout_staged,
            self.sStats_layout,
            self.sScale_layout,
            self.sBitmask_layout,
            sO_layout,
            tiled_mma_QK,
            tiled_mma_QviVi,
            tiled_mma_PVti,
            softmax_scale,
            softmax_scale_log2,
            topk_length_dynamic,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1,),
            cluster=self.cluster_shape_mnk,
            smem = SharedStorage.size_in_bytes(),
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQv0: cute.Tensor,
        mQv1: cute.Tensor,
        mK: cute.Tensor,
        mV0: cute.Tensor,
        mV1: cute.Tensor,
        mVt0: cute.Tensor,
        mVt1: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mIndexTopk: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qv0: cute.CopyAtom,
        tma_atom_Qv1: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V0: Optional[cute.CopyAtom],
        tma_atom_V1: Optional[cute.CopyAtom],
        tma_atom_Vt0: Optional[cute.CopyAtom],
        tma_atom_Vt1: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_copy_O_r2g: cute.TiledCopy,
        sQ_layout_staged: cute.ComposedLayout,
        sK_layout_staged: cute.ComposedLayout,
        sQvi_layout_staged: cute.ComposedLayout,
        sVi_layout_staged: cute.ComposedLayout,
        sVti_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        sScale_layout: cute.Layout,
        sBitmask_layout: cute.Layout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_QviVi: cute.TiledMma,
        tiled_mma_PVti: cute.TiledMma,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        topk_length_dynamic: Optional[Int32],
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QK.thr_id.shape,)
        )
        
        cta_m_block, head_idx, batch_idx = cute.arch.block_idx()
        cluster_m_block = cta_m_block // self.cta_group_size
        mma_tile_coord_v = cta_m_block % cute.size(tiled_mma_QK.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # ==== Allocate SMEM ====
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ==== TMEM stuff ====
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.TmemPtr),
            num_threads=self.num_mma_threads + self.num_softmax_threads + self.num_epilogue_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.mbar_ptr_tmem_dealloc,
        )

        # ==== Prefetch TMA descriptors ====
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_Qv0)
            cpasync.prefetch_descriptor(tma_atom_Qv1)
            if const_expr(self.use_tma_KV):
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V0)
                cpasync.prefetch_descriptor(tma_atom_V1)
                cpasync.prefetch_descriptor(tma_atom_Vt0)
                cpasync.prefetch_descriptor(tma_atom_Vt1)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        # ==== Construct pipelines ====
        tma_warp            = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        mma_warp            = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads          = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads         = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        sm_threads_cluster  = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads * self.cta_group_size)
        epi_threads_cluster = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads * self.cta_group_size)

        TmaUmma   = pipeline.PipelineTmaUmma
        AsyncUmma = pipeline.PipelineAsyncUmma
        UmmaAsync = pipeline.PipelineUmmaAsync
        Async     = pipeline.PipelineAsync

        def make_pipeline(cls, mbar_ptr, num_stages, producer, consumer, tx_count=None):
            return cls.create(
                barrier_storage=mbar_ptr.data_ptr(),
                num_stages=num_stages,
                producer_group=producer,
                consumer_group=consumer,
                defer_sync=True,
                **({"cta_layout_vmnk": cta_layout_vmnk} if cls is not Async else {}),
                **({"tx_count": tx_count} if tx_count is not None else {}),
            )

        # Unconditional pipelines
        pipeline_Q        = make_pipeline(TmaUmma,   storage.mbar_ptr_Q,        self.num_stages_Q,        tma_warp,           mma_warp,           self.tma_copy_bytes_Q)
        pipeline_Qv0      = make_pipeline(TmaUmma,   storage.mbar_ptr_Qv0,      self.num_stages_Qvi,      tma_warp,           mma_warp,           self.tma_copy_bytes_Qvi)
        pipeline_Qv1      = make_pipeline(TmaUmma,   storage.mbar_ptr_Qv1,      self.num_stages_Qvi,      tma_warp,           mma_warp,           self.tma_copy_bytes_Qvi)
        pipeline_S        = make_pipeline(UmmaAsync, storage.mbar_ptr_S,        self.num_stages_S,        mma_warp,           sm_threads_cluster)
        pipeline_P        = make_pipeline(AsyncUmma, storage.mbar_ptr_P,        self.num_stages_P,        sm_threads_cluster, mma_warp)
        pipeline_O0       = make_pipeline(UmmaAsync, storage.mbar_ptr_O0,       self.num_stages_Oi,       mma_warp,           epi_threads_cluster)
        pipeline_O1       = make_pipeline(UmmaAsync, storage.mbar_ptr_O1,       self.num_stages_Oi,       mma_warp,           epi_threads_cluster)
        pipeline_sm_stats = make_pipeline(Async,     storage.mbar_ptr_sm_stats, self.num_stages_sm_stats, sm_threads,         epi_threads)

        # K/V pipelines: type and producer depend on use_tma_KV
        if const_expr(self.use_tma_KV):
            pipeline_K          = make_pipeline(TmaUmma,   storage.mbar_ptr_K,  self.num_stages_K,  tma_warp, mma_warp, self.tma_copy_bytes_K)
            pipeline_V0         = make_pipeline(TmaUmma,   storage.mbar_ptr_V0, self.num_stages_Vi, tma_warp, mma_warp, self.tma_copy_bytes_Vi)
            pipeline_V1         = make_pipeline(TmaUmma,   storage.mbar_ptr_V1, self.num_stages_Vi, tma_warp, mma_warp, self.tma_copy_bytes_Vi)
            pipeline_K_cpasync  = pipeline_V0_cpasync = pipeline_V1_cpasync = pipeline_bitmask = None
        else:
            cpasync_load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cpasync_load_threads)
            relay_warps_cluster  = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cta_group_size)
            relay_threads        = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_relay_threads)

            pipeline_K          = make_pipeline(AsyncUmma, storage.mbar_ptr_K,          self.num_stages_K,   relay_warps_cluster,  mma_warp)
            pipeline_V0         = make_pipeline(AsyncUmma, storage.mbar_ptr_V0,         self.num_stages_Vi,  relay_warps_cluster,  mma_warp)
            pipeline_V1         = make_pipeline(AsyncUmma, storage.mbar_ptr_V1,         self.num_stages_Vi,  relay_warps_cluster,  mma_warp)
            pipeline_K_cpasync  = make_pipeline(Async,     storage.mbar_ptr_K_cpasync,  self.num_stages_K,   cpasync_load_threads, relay_threads)
            pipeline_V0_cpasync = make_pipeline(Async,     storage.mbar_ptr_V0_cpasync, self.num_stages_Vi,  cpasync_load_threads, relay_threads)
            pipeline_V1_cpasync = make_pipeline(Async,     storage.mbar_ptr_V1_cpasync, self.num_stages_Vi,  cpasync_load_threads, relay_threads)
            pipeline_bitmask    = (
                make_pipeline(Async, storage.mbar_ptr_bitmask, self.num_stages_bitmask, cpasync_load_threads, sm_threads)
                if const_expr(self.is_topk_gather and not self.disable_bitmask) else None
            )
        
        sO_empty_mbar_ptr = None
        if const_expr(self.use_tma_O and self.overlap_sO_sV):
            sO_empty_mbar_ptr = storage.sO_empty_mbar_ptr
            if warp_idx == 0:
                cute.arch.mbarrier_init(sO_empty_mbar_ptr, 1)

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ==== Get SMEM tensors ====
        sQ, sK, sQv0, sQv1, sV0, sV1, sVt0, sVt1, sP = (
            store.get_tensor(layout.outer, swizzle=layout.inner)
            for store, layout in [
                (storage.sQ,   sQ_layout_staged),
                (storage.sK,   sK_layout_staged),
                (storage.sQv0, sQvi_layout_staged),
                (storage.sQv1, sQvi_layout_staged),
                (storage.sV0,  sVi_layout_staged),
                (storage.sV1,  sVi_layout_staged),
                (storage.sV0,  sVti_layout_staged),  # sVt0 reuses sV0 storage
                (storage.sV1,  sVti_layout_staged),  # sVt1 reuses sV1 storage
                (storage.sP,   sP_layout_staged),
            ]
        )
        sRowMax = storage.sRowMax.get_tensor(sStats_layout)
        sRowSum = storage.sRowSum.get_tensor(sStats_layout)
        sScale  = storage.sScale.get_tensor(sScale_layout)
        sBitmask = None
        if const_expr(self.is_topk_gather):
            sBitmask = storage.sBitmask.get_tensor(sBitmask_layout)

        if const_expr(self.overlap_sO_sV):
            sO_iterator = sV0.iterator
            assert cute.cosize(sO_layout) <= cute.cosize(sVi_layout_staged) * self.num_hdimv_splits
        else:
            sO_iterator = sQv0.iterator
            assert cute.cosize(sO_layout) <= cute.cosize(sQvi_layout_staged) * self.num_hdimv_splits
        sO = cute.make_tensor(cute.recast_ptr(sO_iterator, sO_layout.inner, self.dtype_O), sO_layout.outer)

        # ==== Get thread MMAs and accumulator fragments ====
        thr_mma_QK = tiled_mma_QK.get_slice(mma_tile_coord_v)
        thr_mma_QviVi = tiled_mma_QviVi.get_slice(mma_tile_coord_v)
        thr_mma_PVti = tiled_mma_PVti.get_slice(mma_tile_coord_v)

        acc_shape_QK = thr_mma_QK.partition_shape_C(self.mma_tiler_QK[:2])
        tStS = thr_mma_QK.make_fragment_C(cute.append(acc_shape_QK, self.num_stages_S))

        acc_shape_PVi = thr_mma_PVti.partition_shape_C(self.mma_tiler_PVti[:2])
        tO0tO0 = thr_mma_PVti.make_fragment_C(acc_shape_PVi)
        tO1tO1 = thr_mma_PVti.make_fragment_C(acc_shape_PVi)
        tO0tO0 = cute.make_tensor(tO0tO0.iterator + self.tmem_offset_O0, tO0tO0.layout)
        tO1tO1 = cute.make_tensor(tO1tO1.iterator + self.tmem_offset_O1, tO1tO1.layout)

        block_info = BlockInfo(
            self.cta_tile_m * self.cta_group_size,
            self.tile_n,
            is_causal=self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            tile_m=self.cta_tile_m,
            tile_n=self.tile_n,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.cta_tile_m * self.cta_group_size,
            self.tile_n,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.num_threads // cute.arch.WARP_SIZE
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta * self.cta_group_size
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        if const_expr(self.use_cpasync_load_KV):
            if warp_idx == self.relay_warp_id:
                if const_expr(self.num_regs_load < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.relay(
                    pipeline_K,
                    pipeline_V0,
                    pipeline_V1,
                    pipeline_K_cpasync,
                    pipeline_V0_cpasync,
                    pipeline_V1_cpasync,
                    sO_empty_mbar_ptr,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )

            if warp_idx in self.cpasync_load_warp_indices:
                if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
                self.load_cpasync(
                    mIndexTopk,
                    mK,
                    mV0,
                    mV1,
                    mVt0,
                    mVt1,
                    sK,
                    sV0,
                    sV1,
                    sVt0,
                    sVt1,
                    sBitmask,
                    pipeline_K,
                    pipeline_V0,
                    pipeline_V1,
                    pipeline_K_cpasync,
                    pipeline_V0_cpasync,
                    pipeline_V1_cpasync,
                    pipeline_bitmask,
                    sO_empty_mbar_ptr,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                )

        if warp_idx == self.load_warp_id:
            if const_expr(self.num_regs_load < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                mQ,
                mK,
                mQv0,
                mQv1,
                mV0,
                mV1,
                mVt0,
                mVt1,
                sQ,
                sK,
                sQv0,
                sQv1,
                sV0,
                sV1,
                sVt0,
                sVt1,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Qv0,
                tma_atom_Qv1,
                tma_atom_V0,
                tma_atom_V1,
                tma_atom_Vt0,
                tma_atom_Vt1,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv0,
                pipeline_Qv1,
                pipeline_V0,
                pipeline_V1,
                sO_empty_mbar_ptr,
                thr_mma_QK,
                thr_mma_QviVi,
                thr_mma_PVti,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.mma(
                sQ,
                sK,
                sQv0,
                sQv1,
                sV0,
                sV1,
                sVt0,
                sVt1,
                sP,
                tiled_mma_QK,
                tiled_mma_QviVi,
                tiled_mma_PVti,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv0,
                pipeline_Qv1,
                pipeline_V0,
                pipeline_V1,
                pipeline_S,
                pipeline_P,
                pipeline_O0,
                pipeline_O1,
                sO_empty_mbar_ptr,
                is_leader_cta,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.softmax_loop(
                softmax_scale,
                softmax_scale_log2,
                mLSE,
                sRowMax,
                sRowSum,
                sScale,
                sBitmask,
                sP,
                tStS,
                thr_mma_QK,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                sO_empty_mbar_ptr,
                AttentionMaskCls,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.correction_loop(
                softmax_scale_log2,
                mO,
                mLSE,
                tma_atom_O,
                sRowMax,
                sRowSum,
                sScale,
                sO,
                tO0tO0,
                tO1tO1,
                pipeline_O0,
                pipeline_O1,
                pipeline_sm_stats,
                sO_empty_mbar_ptr,
                tiled_copy_O_r2g,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE and cute.arch.block_idx() == (0, 0, 0):
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def relay(
        self,
        pipeline_K: pipeline.PipelineAsyncUmma,
        pipeline_V0: pipeline.PipelineAsyncUmma,
        pipeline_V1: pipeline.PipelineAsyncUmma,
        pipeline_K_cpasync: pipeline.PipelineAsync,
        pipeline_V0_cpasync: pipeline.PipelineAsync,
        pipeline_V1_cpasync: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== Make pipeline states ====
        # pipeline_{K,V0,V1} producer
        # pipeline_{K,V0,V1}_cpasync consumer
        producer_state_K = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_K
        )
        producer_state_V0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
        )
        producer_state_V1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
        )
        consumer_state_K = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_K
        )
        consumer_state_V0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Vi
        )
        consumer_state_V1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Vi
        )
        relay_K_fn = partial(self.relay_inner, pipeline_K_cpasync, pipeline_K)
        relay_V0_fn = partial(self.relay_inner, pipeline_V0_cpasync, pipeline_V0)
        relay_V1_fn = partial(self.relay_inner, pipeline_V1_cpasync, pipeline_V1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            # ==== Prologue ====
            # relay K, V0, V1
            consumer_state_K, producer_state_K = relay_K_fn(consumer_state_K, producer_state_K)
            consumer_state_V0, producer_state_V0 = relay_V0_fn(consumer_state_V0, producer_state_V0)
            consumer_state_V1, producer_state_V1 = relay_V1_fn(consumer_state_V1, producer_state_V1)
            
            # ==== Mainloop ====
            for _ in cutlass.range(num_n_blocks-1, unroll=2):
                # relay K, V0, V1, Vt0, Vt1
                consumer_state_K, producer_state_K = relay_K_fn(consumer_state_K, producer_state_K)
                for _ in cutlass.range_constexpr(2):
                    consumer_state_V0, producer_state_V0 = relay_V0_fn(consumer_state_V0, producer_state_V0)
                    consumer_state_V1, producer_state_V1 = relay_V1_fn(consumer_state_V1, producer_state_V1)
            
            # ==== Epilogue ===
            # relay Vt0, Vt1
            consumer_state_V0, producer_state_V0 = relay_V0_fn(consumer_state_V0, producer_state_V0)
            consumer_state_V1, producer_state_V1 = relay_V1_fn(consumer_state_V1, producer_state_V1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_K.producer_tail(producer_state_K)
        pipeline_V0.producer_tail(producer_state_V0)
        pipeline_V1.producer_tail(producer_state_V1)


    @cute.jit
    def relay_inner(
        self,
        pipeline_cpasync: pipeline.PipelineAsync,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        consumer_state: pipeline.PipelineState,
        producer_state: pipeline.PipelineState,
    ):
        pipeline_cpasync.consumer_wait(consumer_state)
        with cute.arch.elect_one():
            pipeline_mma.producer_commit(producer_state)
        consumer_state.advance()
        producer_state.advance()
        return consumer_state, producer_state

    
    @cute.jit
    def load_cpasync(
        self,
        mIndexTopk: cute.Tensor,
        mK: cute.Tensor,
        mV0: cute.Tensor,
        mV1: cute.Tensor,
        mVt0: cute.Tensor,
        mVt1: cute.Tensor,
        sK: cute.Tensor,
        sV0: cute.Tensor,
        sV1: cute.Tensor,
        sVt0: cute.Tensor,
        sVt1: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        pipeline_K: pipeline.PipelineAsyncUmma,
        pipeline_V0: pipeline.PipelineAsyncUmma,
        pipeline_V1: pipeline.PipelineAsyncUmma,
        pipeline_K_cpasync: pipeline.PipelineAsync,
        pipeline_V0_cpasync: pipeline.PipelineAsync,
        pipeline_V1_cpasync: pipeline.PipelineAsync,
        pipeline_bitmask: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== cpasync load warpgroup ====
        # Description: loads tiles of K, V, V0, V1 from gmem to smem using cpasync
        # produces: K, V, V0, V1, bitmask
        # consumes: -
        
        # TODO: use cpasync for non-topk paged attn
        assert sBitmask is not None, "cpasync load meant to be used with topk gather"
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads 
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (self.num_cpasync_load_threads//32)
        
        # ==== Make pipeline states ====
        # producer: acquire PipelineAsyncUmma <- mma
        # producer: commit  PipelineAsync     -> relay
        producer_state_K = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_K
        )
        producer_state_V0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
        )
        producer_state_V1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
        )
        if const_expr(not self.disable_bitmask):
            producer_state_bitmask = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_bitmask,
            )
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            # cluster_m_block == m_idx under MQA 128 assumption
            m_idx = cluster_m_block
            if const_expr(not seqlen.has_cu_seqlens_q):
                mIndexTopk_cur = mIndexTopk[None, m_idx, batch_idx]
            else:
                offset_q = seqlen.offset_q
                mIndexTopk_cur = mIndexTopk[None, m_idx + offset_q]
            
            if const_expr(self.is_causal):
                seqlen_k_limit = m_idx + 1 + seqlen.seqlen_k - seqlen.seqlen_q
            else:
                seqlen_k_limit = seqlen.seqlen_k
            cpasync_gather_kv_manager = CpasyncGatherKVManager.create(
                mIndexTopk_cur,
                sBitmask,
                cta_rank_in_cluster,
                tidx,
                warp_idx,
                self.topk_length,
                seqlen_k_limit,
                self.tile_n,
                self.hdim,
                self.hdimv,
                self.num_hdimv_splits,
                self.num_cpasync_load_threads,
                mK.element_type,
                self.cta_group_size,
                pipeline_bitmask,
                self.num_stages_bitmask,
                self.cpasync_barrier,
                self.disable_bitmask,
            )

            # (seqlen_k, hdim) or (seqlen_k, hdimv//2)
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV0_cur = seqlen.offset_batch_K(mV0, batch_idx, dim=3)[None, None, head_idx_kv]
            mV1_cur = seqlen.offset_batch_K(mV1, batch_idx, dim=3)[None, None, head_idx_kv]
            # (hdimv//2, seqlen_k)
            if const_expr(not seqlen.has_cu_seqlens_k):
                mVt0_cur = mVt0[None, None, head_idx_kv, batch_idx]
                mVt1_cur = mVt1[None, None, head_idx_kv, batch_idx]
            else:
                mVt0_cur = cute.domain_offset((0, seqlen.offset_k), mVt0[None, None, head_idx_kv])
                mVt1_cur = cute.domain_offset((0, seqlen.offset_k), mVt1[None, None, head_idx_kv])
            # (hdimv//4, seqlen_k)
            hdimv_split_per_cta = self.hdimv // self.num_hdimv_splits // self.cta_group_size
            mVt0_cur = cute.tiled_divide(mVt0_cur, (hdimv_split_per_cta, ))[None, cta_rank_in_cluster, None]
            mVt1_cur = cute.tiled_divide(mVt1_cur, (hdimv_split_per_cta, ))[None, cta_rank_in_cluster, None]

            load_K = partial(self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_K, pipeline_K_cpasync, sK, False, "K", mK_cur,
            )
            load_V0 = partial(self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_V0, pipeline_V0_cpasync, sV0, False, "V", mV0_cur,
            )
            load_V1 = partial(self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_V1, pipeline_V1_cpasync, sV1, False, "V", mV1_cur,
            )
            load_Vt0 = partial(self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_V0, pipeline_V0_cpasync, sVt0, True, "V", mVt0_cur,
            )
            load_Vt1 = partial(self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_V1, pipeline_V1_cpasync, sVt1, True, "V", mVt1_cur,
            )

            # gather KV path processes n_blocks in increasing order
            n_block = 0

            # ==== Prologue ====
            # K, V0, V1
            cpasync_gather_kv_manager.load_index_topk(n_block, transpose=False)
            producer_state_K = load_K(producer_state_K)
            producer_state_V0 = load_V0(producer_state_V0)
            producer_state_V1 = load_V1(producer_state_V1)
            if const_expr(not self.disable_bitmask):
                producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                    producer_state_bitmask
                )

            if const_expr(self.use_tma_O and self.overlap_sO_sV):
                cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                producer_phase_O ^= 1

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups-1, unroll=1):
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    n_block = n_block_group * self.num_stages_S + stage
                    # K, V0, V1
                    cpasync_gather_kv_manager.load_index_topk(n_block + 1, transpose=False)
                    producer_state_K = load_K(producer_state_K)
                    producer_state_V0 = load_V0(producer_state_V0)
                    producer_state_V1 = load_V1(producer_state_V1)
                    if const_expr(not self.disable_bitmask):
                        producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                            producer_state_bitmask
                        )
                    # Vt0, Vt1
                    cpasync_gather_kv_manager.load_index_topk(n_block, transpose=True)
                    producer_state_V0 = load_Vt0(producer_state_V0)
                    producer_state_V1 = load_Vt1(producer_state_V1)

            # ==== Epilogue ====
            for stage in cutlass.range_constexpr(self.num_stages_S):
                n_block = (num_n_block_groups-1) * self.num_stages_S + stage
                if const_expr(stage == 0):
                    # K, V0, V1
                    cpasync_gather_kv_manager.load_index_topk(n_block + 1, transpose=False)
                    producer_state_K = load_K(producer_state_K)
                    producer_state_V0 = load_V0(producer_state_V0)
                    producer_state_V1 = load_V1(producer_state_V1)
                    if const_expr(not self.disable_bitmask):
                        producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                            producer_state_bitmask
                        )
                        
                # Vt0, Vt1
                cpasync_gather_kv_manager.load_index_topk(n_block, transpose=True)
                producer_state_V0 = load_Vt0(producer_state_V0)
                producer_state_V1 = load_Vt1(producer_state_V1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        
        pipeline_K_cpasync.producer_tail(producer_state_K)
        pipeline_V0_cpasync.producer_tail(producer_state_V0)
        pipeline_V1_cpasync.producer_tail(producer_state_V1)
        if const_expr(not self.disable_bitmask):
            pipeline_bitmask.producer_tail(producer_state_bitmask)


    @cute.jit
    def cpasync_gather_load_KV(
        self,
        cpasync_gather_kv_manager: CpasyncGatherKVManager,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        pipeline_cpasync: pipeline.PipelineAsync,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        mX: cute.Tensor,
        producer_state: pipeline.PipelineState,
    ):
        stage, phase = producer_state.index, producer_state.phase
        pipeline_mma.producer_acquire(producer_state)
        cpasync_gather_kv_manager.load_X(
            mX, sX[None, None, None, stage], transpose, K_or_V
        )
        cute.arch.cp_async_commit_group()
        pipeline_cpasync.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state
            

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mQv0: cute.Tensor,
        mQv1: cute.Tensor,
        mV0: cute.Tensor,
        mV1: cute.Tensor,
        mVt0: cute.Tensor,
        mVt1: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sQv0: cute.Tensor,
        sQv1: cute.Tensor,
        sV0: cute.Tensor,
        sV1: cute.Tensor,
        sVt0: cute.Tensor,
        sVt1: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Qv0: cute.CopyAtom,
        tma_atom_Qv1: cute.CopyAtom,
        tma_atom_V0: cute.CopyAtom,
        tma_atom_V1: cute.CopyAtom,
        tma_atom_Vt0: cute.CopyAtom,
        tma_atom_Vt1: cute.CopyAtom,
        pipeline_Q: pipeline.PipelineAsync,
        pipeline_K: pipeline.PipelineAsync,
        pipeline_Qv0: pipeline.PipelineAsync,
        pipeline_Qv1: pipeline.PipelineAsync,
        pipeline_V0: pipeline.PipelineAsync,
        pipeline_V1: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        thr_mma_QK: cute.ThrMma,
        thr_mma_QviVi: cute.ThrMma,
        thr_mma_PVti: cute.ThrMma,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== Load warp ====
        # Description: loads tiles of Q, Qv, K, V, V0, V1 from gmem to smem using TMA
        # produces: Q, Qv, K, V, V0, V1
        # consumes: -

        mQvs = [mQv0, mQv1]
        mVs = [mV0, mV1]
        mVts = [mVt0, mVt1]

        sQvs = [sQv0, sQv1]
        sVs = [sV0, sV1]
        sVts = [sVt0, sVt1]

        tma_atom_Qvs = [tma_atom_Qv0, tma_atom_Qv1]
        tma_atom_Vs = [tma_atom_V0, tma_atom_V1]
        tma_atom_Vts = [tma_atom_Vt0, tma_atom_Vt1]

        pipeline_Qvs = [pipeline_Qv0, pipeline_Qv1]
        pipeline_Vs = [pipeline_V0, pipeline_V1]
        
        # ==== Make pipeline states ====
        producer_state_Q = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Q
        )
        producer_state_Qv0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Qvi
        )
        producer_state_Qv1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Qvi
        )
        if const_expr(self.use_tma_KV):
            producer_state_K = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_K
            )
            producer_state_V0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
            )
            producer_state_V1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_Vi
            )
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            # ==== Partition GMEM tensors ====
            # (seqlen_q, hdim or hdimv//2)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mQvs_cur = [
                seqlen.offset_batch_Q(mQvs[split], batch_idx, dim=3)[None, None, head_idx]
                for split in range(self.num_hdimv_splits)
            ]
            # (mma_tile_m, hdim or hdimv//2)
            gQ = cute.local_tile(
                mQ_cur,  
                (self.mma_tiler_QK[0], self.mma_tiler_QK[2]),
                (cluster_m_block, 0),
            )  
            gQvs = [
                cute.local_tile(
                    mQvs_cur[split],
                    (self.mma_tiler_QviVi[0], self.mma_tiler_QviVi[2]),
                    (cluster_m_block, 0),
                ) for split in range(self.num_hdimv_splits)
            ]
            tSgQ = thr_mma_QK.partition_A(gQ)
            tSgQvs = [
                thr_mma_QviVi.partition_A(gQvs[split])
                for split in range(self.num_hdimv_splits)
            ]
            tQsQ, tQgQ = cpasync.tma_partition(
                atom=tma_atom_Q,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sQ, 0, 3),
                gmem_tensor=cute.group_modes(tSgQ, 0, 3),
            )
            tQvsQvs, tQvgQvs = zip(*[
                cpasync.tma_partition(
                    atom=tma_atom,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sQv, 0, 3),
                    gmem_tensor=cute.group_modes(tSgQv, 0, 3),
                )
                for tma_atom, sQv, tSgQv in zip(tma_atom_Qvs, sQvs, tSgQvs)
            ])

            if const_expr(self.use_tma_KV):
                # (seqlen_k, hdim) or (seqlen_k, hdimv//2)
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mVs_cur = [
                    seqlen.offset_batch_K(mVs[split], batch_idx, dim=3)[None, None, head_idx_kv]
                    for split in range(self.num_hdimv_splits)
                ]
                # (hdimv//2, seqlen_k)
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mVts_cur = [
                        mVts[split][None, None, head_idx_kv, batch_idx]
                        for split in range(self.num_hdimv_splits)
                    ]
                else:
                    mVts_cur = [
                        cute.domain_offset((0, seqlen.offset_k), mVts[split][None, None, head_idx_kv])
                        for split in range(self.num_hdimv_splits)
                    ]
                # (tile_n, hdim or hdimv//2, num_n_blocks)
                gK = cute.local_tile(
                    mK_cur,
                    (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                    (None, 0),
                )  
                gVs = [
                    cute.local_tile(
                        mVs_cur[split],
                        (self.mma_tiler_QviVi[1], self.mma_tiler_QviVi[2]),
                        (None, 0),
                    ) for split in range(self.num_hdimv_splits)
                ]
                # (hdim or hdimv//2, tile_n, num_n_blocks)
                gVts = [
                    cute.local_tile(
                        mVts_cur[split],
                        (self.mma_tiler_PVti[1], self.mma_tiler_PVti[2]),
                        (0, None),
                    ) for split in range(self.num_hdimv_splits)
                ]
                tSgK = thr_mma_QK.partition_B(gK)
                tSgVs = [
                    thr_mma_QviVi.partition_B(gVs[split])
                    for split in range(self.num_hdimv_splits)
                ]
                tOgVts = [
                    thr_mma_PVti.partition_B(gVts[split])
                    for split in range(self.num_hdimv_splits)
                ]
                tKsK, tKgK = cpasync.tma_partition(
                    atom=tma_atom_K,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sK, 0, 3),
                    gmem_tensor=cute.group_modes(tSgK, 0, 3),
                )
                tVsVs, tVgVs = zip(*[
                    cpasync.tma_partition(
                        atom=tma_atom,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sV, 0, 3),
                        gmem_tensor=cute.group_modes(tSgV, 0, 3),
                    )
                    for tma_atom, sV, tSgV in zip(tma_atom_Vs, sVs, tSgVs)
                ])
                tVtsVts, tVtgVts = zip(*[
                    cpasync.tma_partition(
                        atom=tma_atom,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sVt, 0, 3),
                        gmem_tensor=cute.group_modes(tOgV, 0, 3),
                    )
                    for tma_atom, sVt, tOgV in zip(tma_atom_Vts, sVts, tOgVts)
                ])

            load_Q = partial(self.load_inner, tma_atom_Q, tQgQ, tQsQ, pipeline_Q)
            load_Qv = partial(self.load_inner, tma_atom_Qvs, tQvgQvs, tQvsQvs, pipeline_Qvs)
            if const_expr(self.use_tma_KV):
                load_K = partial(self.load_inner, tma_atom_K, tKgK, tKsK, pipeline_K)
                load_V = partial(self.load_inner, tma_atom_Vs, tVgVs, tVsVs, pipeline_Vs)
                load_Vt = partial(self.load_inner, tma_atom_Vts, tVtgVts, tVtsVts, pipeline_Vs)

            # ==== Load stationary operands ====

            # copy Q, Qvi gmem -> smem
            producer_state_Q = load_Q(producer_state_Q)
            producer_state_Qv0 = load_Qv(producer_state_Qv0, split=0)
            producer_state_Qv1 = load_Qv(producer_state_Qv1, split=1)

            if const_expr(self.use_tma_KV):
                # ==== Prologue ====
                n_block_first = n_block_max - 1
                # copy K gmem -> smem
                producer_state_K = load_K(producer_state_K, n_block=n_block_first)
                # copy Vi gmem -> smem
                producer_state_V0 = load_V(producer_state_V0, n_block=n_block_first, split=0)
                producer_state_V1 = load_V(producer_state_V1, n_block=n_block_first, split=1)

                if const_expr(self.use_tma_O and self.overlap_sO_sV):
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Main loop ====
                for n_block_group in cutlass.range(num_n_block_groups-1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        n_block = n_block_max - 1 - n_block_group * self.num_stages_S - stage
                        # copy K gmem -> smem
                        producer_state_K = load_K(producer_state_K, n_block=n_block-1)
                        # copy Vi gmem -> smem
                        producer_state_V0 = load_V(producer_state_V0, n_block=n_block-1, split=0)
                        producer_state_V1 = load_V(producer_state_V1, n_block=n_block-1, split=1)
                        # copy Vti gmem -> smem
                        producer_state_V0 = load_Vt(producer_state_V0, n_block=n_block, split=0)
                        producer_state_V1 = load_Vt(producer_state_V1, n_block=n_block, split=1)
                
                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range(num_final_n_blocks, unroll_full=True):
                    n_block = num_final_n_blocks - 1 - stage
                    if n_block > 0:
                        # copy K gmem -> smem
                        producer_state_K = load_K(producer_state_K, n_block=n_block-1)
                        # copy Vi gmem -> smem
                        producer_state_V0 = load_V(producer_state_V0, n_block=n_block-1, split=0)
                        producer_state_V1 = load_V(producer_state_V1, n_block=n_block-1, split=1)
                    # copy Vti gmem -> smem
                    producer_state_V0 = load_Vt(producer_state_V0, n_block=n_block, split=0)
                    producer_state_V1 = load_Vt(producer_state_V1, n_block=n_block, split=1)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_Q.producer_tail(producer_state_Q)
        pipeline_Qv0.producer_tail(producer_state_Qv0)
        pipeline_Qv1.producer_tail(producer_state_Qv1)
        if const_expr(self.use_tma_KV):
            pipeline_K.producer_tail(producer_state_K)
            pipeline_V0.producer_tail(producer_state_V0)
            pipeline_V1.producer_tail(producer_state_V1)

    @cute.jit
    def load_inner(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        n_block: Optional[Int32] = None,
        split: Optional[Int32] = None,
    ):
        stage = producer_state.index
        if const_expr(split is not None):
            tma_atom = tma_atom[split]
            tXgX = tXgX[split]
            tXsX = tXsX[split]
            load_pipeline = load_pipeline[split]
        if const_expr(n_block is not None):
            tXgX = tXgX[(None, n_block)]
        tXsX = tXsX[(None, stage)]

        load_pipeline.producer_acquire(producer_state)
        tma_bar_ptr = load_pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom, tXgX, tXsX, tma_bar_ptr=tma_bar_ptr)
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma(
        self,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sQv0: cute.Tensor,
        sQv1: cute.Tensor,
        sV0: cute.Tensor,
        sV1: cute.Tensor,
        sVt0: cute.Tensor,
        sVt1: cute.Tensor,
        sP: cute.Tensor,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_QviVi: cute.TiledMma,
        tiled_mma_PVti: cute.TiledMma,
        pipeline_Q: pipeline.PipelineAsync,
        pipeline_K: pipeline.PipelineAsync,
        pipeline_Qv0: pipeline.PipelineAsync,
        pipeline_Qv1: pipeline.PipelineAsync,
        pipeline_V0: pipeline.PipelineAsync,
        pipeline_V1: pipeline.PipelineAsync,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        is_leader_cta: Boolean,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== mma warp ====
        # Description: Computes Q @ K^T, Qv @ V^T, and P @ V
        # Produces: S, O
        # Consumes: Q, K, Qv, V, P

        pipelines_V = [pipeline_V0, pipeline_V1]
        pipelines_Qv = [pipeline_Qv0, pipeline_Qv1]
        pipelines_O = [pipeline_O0, pipeline_O1]

        sQvs = [sQv0, sQv1]
        sVs = [sV0, sV1]
        sVts = [sVt0, sVt1]

        # Set accumulate = True for Qv @ V^T since we are accumulating on the Q @ K^T result
        tiled_mma_QviVi.set(tcgen05.Field.ACCUMULATE, True)

        # Operands for S = Q @ K^T
        tSrQ = tiled_mma_QK.make_fragment_A(sQ)
        tSrK = tiled_mma_QK.make_fragment_B(sK)

        # Operands for S += Qv @ V^T
        tSrQvs = [
            tiled_mma_QviVi.make_fragment_A(sQvs[split])
            for split in range(self.num_hdimv_splits)
        ]
        tSrVs = [
            tiled_mma_QviVi.make_fragment_B(sVs[split])
            for split in range(self.num_hdimv_splits)
        ]

        # Operands for Oi = P @ Vi
        tOrP = tiled_mma_PVti.make_fragment_A(sP)
        tOrVts = [
            tiled_mma_PVti.make_fragment_B(sVts[split])
            for split in range(self.num_hdimv_splits)
        ]

        # GEMM functions
        gemm_QK = [
            partial(
                fa_sm100_utils.gemm_ptx_partial,
                tiled_mma_QK.op,
                self.tmem_offset_S[stage],
                tCrA=tSrQ[None, None, None, 0],
                sA=sQ[None, None, None, 0],
                zero_init=True,
                cta_group=self.cta_group_size,
            ) for stage in range(self.num_stages_S)
        ]
        gemms_QvV = [
            [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_QviVi.op,
                    self.tmem_offset_S[stage],
                    tCrA=tSrQvs[split][None, None, None, 0],
                    sA=sQvs[split][None, None, None, 0],
                    zero_init=False,
                    cta_group=self.cta_group_size,
                ) for stage in range(self.num_stages_S)
            ] for split in range(self.num_hdimv_splits)
        ]
        gemms_PVt = [
            partial(
                fa_sm100_utils.gemm_ptx_partial,
                tiled_mma_PVti.op,
                self.tmem_offsets_O[split],
                tOrP[None, None, None, 0],
                sA=sP[None, None, None, 0],
                cta_group=self.cta_group_size,
            ) for split in range(self.num_hdimv_splits)
        ]

        consumer_state_Q = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Q
        )
        consumer_state_K = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_K
        )
        consumer_state_Qv0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Qvi
        )
        consumer_state_Qv1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Qvi
        )
        consumer_state_V0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Vi
        )
        consumer_state_V1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Vi
        )
        producer_state_S = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_S
        )
        consumer_state_P = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_P
        )
        producer_state_O0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Oi
        )
        producer_state_O1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_Oi
        )

        mma_QK = partial(self.mma_inner, gemm_QK, pipeline_K, tSrK, sK)
        mma_QvV = partial(self.mma_inner, gemms_QvV, pipelines_V, tSrVs, sVs)
        mma_PVt = partial(self.mma_inner, gemms_PVt, pipelines_V, tOrVts, sVts)

        work_tile = tile_scheduler.initial_work_tile_info()
        O_should_accumulate = False
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                # n_block_max = self.topk_length // self.tile_n
                n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            if is_leader_cta:
                pipeline_Q.consumer_wait(consumer_state_Q)
                pipeline_Qv0.consumer_wait(consumer_state_Qv0)
                pipeline_Qv1.consumer_wait(consumer_state_Qv1)

                consumer_states_V = [consumer_state_V0, consumer_state_V1]
                producer_states_O = [producer_state_O0, producer_state_O1]

                # ==== Prologue ====
                pipeline_S.producer_acquire(producer_state_S)
                # S = Q @ K^T
                consumer_state_K = mma_QK(consumer_state_K, stage=0)
                # S += Qvi @ Vi^T
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_states_V[split] = mma_QvV(
                        consumer_states_V[split], stage=0, split=split
                    )
                pipeline_S.producer_commit(producer_state_S)
                producer_state_S.advance()

                # ==== Mainloop ====
                for _ in cutlass.range(num_n_block_groups-1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        next_stage = const_expr((stage + 1) % self.num_stages_S)
                        pipeline_S.producer_acquire(producer_state_S)
                        # S = Q @ K^T
                        consumer_state_K = mma_QK(consumer_state_K, stage=next_stage)
                        # S += Qvi @ Vi^T
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            consumer_states_V[split] = mma_QvV(
                                consumer_states_V[split], stage=next_stage, split=split
                            )
                        pipeline_S.producer_commit(producer_state_S)
                        producer_state_S.advance()
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_states_V[split] = mma_PVt(
                                consumer_states_V[split],
                                split=split,
                                zero_init=not O_should_accumulate
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                        O_should_accumulate = True
                
                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    n_block = num_final_n_blocks - 1 - stage
                    if const_expr(stage == 0):
                        if n_block > 0:
                            pipeline_S.producer_acquire(producer_state_S)
                            # S = Q @ K^T
                            consumer_state_K = mma_QK(consumer_state_K, stage=stage+1)
                            # S += Qvi @ Vi^T
                            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                                consumer_states_V[split] = mma_QvV(
                                    consumer_states_V[split], stage=stage+1, split=split
                                )
                            pipeline_S.producer_commit(producer_state_S)
                            producer_state_S.advance()
                    if n_block >= 0:
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_states_V[split] = mma_PVt(
                                consumer_states_V[split],
                                split=split,
                                zero_init=not O_should_accumulate
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                    O_should_accumulate = True

                consumer_state_V0, consumer_state_V1 = consumer_states_V
                producer_state_O0, producer_state_O1 = producer_states_O

                pipeline_Q.consumer_release(consumer_state_Q)

                # if we overlap sOi with sQvi for tma store, need to acquire signal
                if const_expr(self.use_tma_O and not self.overlap_sO_sV):
                    pipeline_O0.producer_tail(producer_state_O0.clone())
                    pipeline_O1.producer_tail(producer_state_O1.clone())
                
                pipeline_Qv0.consumer_release(consumer_state_Qv0)
                pipeline_Qv1.consumer_release(consumer_state_Qv1)
                consumer_state_Q.advance()
                consumer_state_Qv0.advance()
                consumer_state_Qv1.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            O_should_accumulate = False

        pipeline_S.producer_tail(producer_state_S)
        pipeline_O0.producer_tail(producer_state_O0)
        pipeline_O1.producer_tail(producer_state_O1)

    @cute.jit
    def mma_inner(
        self,
        gemm,
        load_pipeline,
        tCrB,
        sB,
        consumer_state: pipeline.PipelineState,
        stage: Optional[Int32] = None,
        split: Optional[Int32] = None,
        zero_init: Optional[bool] = None,
    ):
        if const_expr(split is not None):
            gemm = gemm[split]
            load_pipeline = load_pipeline[split]
            tCrB = tCrB[split]
            sB = sB[split]
        if const_expr(stage is not None):
            gemm = gemm[stage]

        smem_stage = consumer_state.index
        tCrB_cur = tCrB[None, None, None, smem_stage]
        sB_cur = sB[None, None, None, smem_stage]

        load_pipeline.consumer_wait(consumer_state)
        if const_expr(zero_init is not None):
            gemm(tCrB=tCrB_cur, sB=sB_cur, zero_init=zero_init)
        else:
            gemm(tCrB=tCrB_cur, sB=sB_cur)
        load_pipeline.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state
        

    @cute.jit
    def softmax_loop(
        self,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        mLSE: Optional[cute.Tensor],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        sP: cute.Tensor,
        tStS: cute.Tensor,
        thr_mma_QK: cute.ThrMma,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        AttentionMaskCls: Callable,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== softmax warpgroup ====
        # Description: computes softmax on S and writes the result to P
        # Produces: P, softmax stats
        # Consumes: S, bitmask (for topk sparsity)

        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (self.num_softmax_threads//32)

        tSAcc = tStS[(None, None), 0, 0, 0]
        tSAcc_staged = [tStS[(None, None), 0, 0, stage] for stage in range(self.num_stages_S)]
        
        cS = cute.make_identity_tensor(self.mma_tiler_QK[:2])  # (128, 128)
        tScS = thr_mma_QK.partition_C(cS)[(None, None), 0, 0]  # (64, 128)

        # S tmem -> rmem copy objects
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc)
        tmem_load_thr = tmem_load_tiled.get_slice(tidx)
        # S tmem -> rmem copy operands
        tStS_t2r = tmem_load_thr.partition_S(tSAcc)  # (((32, 32), 1), 1, 2)
        tStS_t2r_staged = [tmem_load_thr.partition_S(tSAcc_staged[stage]) for stage in range(self.num_stages_S)]
        tScS_t2r = tmem_load_thr.partition_D(tScS)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)

        # P rmem -> smem copy objects
        universal_copy_bits = 128
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_P,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        smem_store_thr = smem_store_tiled.get_slice(tidx)
        # P rmem -> smem copy operands
        sP_slice = sP[None, None, None, 0]
        sP_mn = cute.make_tensor(
            sP_slice.iterator,
            cute.make_layout(
                (
                    (sP_slice.shape[0][0], sP_slice.shape[1]),
                    (sP_slice.shape[0][1], sP_slice.shape[2]),
                ),
                stride=(
                    (sP_slice.stride[0][0], sP_slice.stride[1]),
                    (sP_slice.stride[0][1], sP_slice.stride[2]),
                ),
            ),
        )
        sP_smem_view = smem_store_thr.partition_D(sP_mn)

        consumer_state_S = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_S
        )
        producer_state_P = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_P
        )
        producer_state_sm_stats = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_sm_stats
        )
        consumer_state_bitmask = None
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            consumer_state_bitmask = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_bitmask
            )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask_sm100,
                m_block=cluster_m_block,
                thr_mma=thr_mma_QK,
                thr_tmem_load=tmem_load_thr,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                r2p=False, # TODO: fix r2p for 2cta
            )
            disable_mask = self.disable_bitmask and self.is_topk_gather

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.dtype_Q.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step_fn = partial(
                self.softmax_step,
                softmax,
                sRowMax,
                sScale,
                sBitmask,
                tStS_t2r_staged,
                tSrS_t2r,
                sP_smem_view,
                tmem_load_thr,
                smem_store_thr,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                tidx,
                warp_idx,
            )

            ### first iteration ###
            n_block = n_block_max - 1
            (consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask
             ) = softmax_step_fn(
                consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask,
                0, n_block,
                mask_fn=partial(mask_fn, mask_seqlen=True)
                if not const_expr(disable_mask) else None,
                is_first=True,
            )
            n_block -= 1

            ### Separate iterations with causal masking
            # note: For square mma tile, can mask at most 1 n_block_group
            if const_expr((self.is_causal or self.is_local) and not self.is_topk_gather):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, cluster_m_block, n_block_min
                )
                num_masked_n_blocks = n_block_max - 1 - n_block_min_causal_local_mask
                num_masked_n_block_groups = min(
                    num_n_block_groups - 1,
                    cute.ceil_div(num_masked_n_blocks, self.num_stages_S)
                )
                num_n_block_groups -= num_masked_n_block_groups
                for _ in cutlass.range(num_masked_n_block_groups, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        (consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask
                         ) = softmax_step_fn(
                            consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask,
                            1-stage, n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                        )
                        n_block -= 1

            ### Mainloop ###
            for n_block_group in cutlass.range(num_n_block_groups-1, unroll=1):
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    (consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask
                     ) = softmax_step_fn(
                        consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask,
                        1-stage, n_block,
                        mask_fn=partial(mask_fn, mask_seqlen=False)
                        if const_expr(self.is_topk_gather and not self.disable_bitmask) else None,
                    )
                    n_block -= 1

            ### last iteration if even ###
            # always mask to simplify logic
            if even_n_blocks:
                (consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask
                 ) = softmax_step_fn(
                    consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask,
                    1, n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False)
                    if not const_expr(disable_mask) else None,
                )
                n_block -= 1
            
            # write row max and sum to smem
            sRowSum[tidx % self.cta_tile_m, warp_idx//self.cta_group_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                if tidx < self.cta_tile_m:
                    sRowMax[tidx, 0] = softmax.row_max[0]
            self.sm_stats_barrier_full.arrive()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            self.sm_stats_barrier_empty.arrive_and_wait()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_sm_stats.producer_tail(producer_state_sm_stats)


    @cute.jit
    def softmax_step(
        self,
        softmax: SoftmaxSm100,
        sRowMax: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        tStS_t2r_staged: cute.Tensor,
        tSrS_t2r: cute.Tensor,
        sP_smem_view: cute.Tensor,
        tmem_load_thr: cute.CopyAtom,
        smem_store_thr: cute.CopyAtom,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        tidx: Int32,
        warp_idx: Int32,
        consumer_state_S: pipeline.PipelineState,
        producer_state_P: pipeline.PipelineState,
        producer_state_sm_stats: pipeline.PipelineState,
        consumer_state_bitmask: Optional[pipeline.PipelineState],
        stage: cutlass.Constexpr[Int32],
        n_block: Int32,
        mask_fn: Optional[Callable] = None,
        is_first: Boolean = False,
    ):
        tSrP = cute.make_rmem_tensor(tSrS_t2r.shape, self.dtype_P)
        rP_smem_view = smem_store_thr.retile(tSrP)

        pipeline_S.consumer_wait(consumer_state_S)
        cute.copy(tmem_load_thr, tStS_t2r_staged[stage], tSrS_t2r)
        cute.arch.fence_view_async_tmem_load()
        pipeline_S.consumer_release(consumer_state_S)

        rBitmask = None
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            assert pipeline_bitmask is not None
            assert consumer_state_bitmask is not None
            pipeline_bitmask.consumer_wait(consumer_state_bitmask)
            rBitmask = cute.make_rmem_tensor((self.tile_n//64, ), dtype=Uint32)
            bitmask_col_offset = self.tile_n//64 if warp_idx >= 2 else 0
            for i in cutlass.range_constexpr(cute.size(rBitmask)):
                rBitmask[i] = sBitmask[bitmask_col_offset + i, consumer_state_bitmask.index]

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block, rBitmask=rBitmask)

        # compute threadwise row_max
        row_max = softmax.compute_row_max_local(tSrS_t2r.load(), is_first)
        self.softmax_barrier.arrive_and_wait()

        # 2-thread reduce row_max through smem
        assert self.cta_tile_m * self.cta_group_size == 128
        sRowMax[tidx % self.cta_tile_m, warp_idx//self.cta_group_size] = row_max
        self.softmax_barrier.arrive_and_wait()
        # must release after barrier sync
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            pipeline_bitmask.consumer_release(consumer_state_bitmask)
        row_max0 = sRowMax[tidx % self.cta_tile_m, 0]
        row_max1 = sRowMax[tidx % self.cta_tile_m, 1]
        row_max = max(row_max0, row_max1)

        row_max, acc_scale = softmax.update_row_max_from_local(row_max, is_first)

        # note: acc_scales agree for paired threads
        pipeline_sm_stats.producer_acquire(producer_state_sm_stats)
        if warp_idx < self.cta_group_size:
            sScale[tidx % self.cta_tile_m, producer_state_sm_stats.index] = acc_scale
        pipeline_sm_stats.producer_commit(producer_state_sm_stats)
        
        # x -> scale_log2*x-rowmax
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        # x -> exp2(x)
        softmax.apply_exp2_convert(tSrS_t2r, tSrP)

        pipeline_P.producer_acquire(producer_state_P)
        cute.copy(smem_store_thr, rP_smem_view, sP_smem_view)
        cute.arch.fence_view_async_shared()
        pipeline_P.producer_commit(producer_state_P)

        consumer_state_S.advance()
        producer_state_P.advance()
        producer_state_sm_stats.advance()
        if const_expr(self.is_topk_gather and not self.disable_bitmask):
            consumer_state_bitmask.advance()

        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)   

        return consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask


    @cute.jit
    def correction_loop(
        self,
        softmax_scale_log2: Float32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_O: Optional[cute.CopyAtom],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sO: cute.Tensor,
        tO0tO0: cute.Tensor,
        tO1tO1: cute.Tensor,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        tiled_copy_O_r2g: cute.TiledCopy,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        ### ==== correction/epilogue warpgroup ====
        # Correction: copy scale smem -> rmem, copy O tmem -> rmem, rescale O, store O rmem -> tmem
        # Epilogue:   copy O tmem -> rmem, do final scaling of O, store O rmem -> gmem,
        #             optionally store LSE
        # Produces: -
        # Consumes: O, softmax stats
        
        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (self.num_epilogue_threads//32)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        leader_warp = warp_idx==0

        tO0tO0 = tO0tO0[(None, None), 0, 0]  # (64, (128, 2))
        tO1tO1 = tO1tO1[(None, None), 0, 0]  # (64, (128, 2))
        tOtOs = [tO0tO0, tO1tO1]

        # tuneable parameter
        corr_tile_size = math.gcd(32, self.tmem_cols_Oi)
        
        tmem_load_atom_O = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        tmem_store_atom_O = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        thr_tmem_load_O = tcgen05.make_tmem_copy(tmem_load_atom_O, tO0tO0).get_slice(tidx)
        thr_tmem_store_O = tcgen05.make_tmem_copy(tmem_store_atom_O, tO0tO0).get_slice(tidx)

        # ((32,1),1,4)
        tOtOs_t2r = [
            thr_tmem_load_O.partition_S(tOtOs[split])
            for split in range(self.num_hdimv_splits)
        ]
        tOtOs_r2t = [
            thr_tmem_store_O.partition_D(tOtOs[split])
            for split in range(self.num_hdimv_splits)
        ]

        cOi = cute.make_identity_tensor((self.cta_tile_m, self.hdimv // self.num_hdimv_splits))
        thr_tiled_copy_O_r2g = tiled_copy_O_r2g.get_slice(tidx)
        tOicOi = thr_tiled_copy_O_r2g.partition_S(cOi)

        tOicOi_t2r = thr_tmem_load_O.partition_D(tOicOi[(None, None), 0, 0])

        pipelines_O = [pipeline_O0, pipeline_O1]

        consumer_state_O0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Oi
        )
        consumer_state_O1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_Oi
        )
        consumer_state_sm_stats = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_sm_stats
        )

        do_correction_rescale = partial(
            self.correction_rescale,
            thr_tmem_load_O,
            thr_tmem_store_O,
            tOicOi_t2r,
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            cluster_m_block = cta_m_block // self.cta_group_size

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            consumer_states_O = [consumer_state_O0, consumer_state_O1]

            # acquire first signal and release immediately
            pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
            pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
            consumer_state_sm_stats.advance()

            for _ in cutlass.range(num_n_blocks - 1, unroll=1):
                pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
                scale = sScale[tidx % self.cta_tile_m, consumer_state_sm_stats.index]
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
                consumer_state_sm_stats.advance()
                
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_Oi = consumer_states_O[split]
                    pipelines_O[split].consumer_wait(consumer_state_Oi)
                    if should_rescale:
                        do_correction_rescale(
                            tOtOs_t2r[split],
                            tOtOs_r2t[split],
                            scale,
                        )
                    pipelines_O[split].consumer_release(consumer_state_Oi)
                    consumer_state_Oi.advance()
                    consumer_states_O[split] = consumer_state_Oi

            # (seqlen_q, hdimv)
            mO_cur = seqlen.offset_batch_Q(
                mO, batch_idx, dim=3, ragged=self.ragged_tma_O
            )[None, None, head_idx]
            # (cta_tile_m, hdimv//2, 2)
            gO = cute.local_tile(
                mO_cur,
                (self.cta_tile_m, self.hdimv // self.num_hdimv_splits),
                (cta_m_block, None),
            )
            tOgO = thr_tiled_copy_O_r2g.partition_D(gO)
            # ((32, 1), 1, 4)
            tOrOs_t2r = [
                cute.make_rmem_tensor(tOicOi_t2r.shape, self.dtype_acc)
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g_f32 = [
                thr_tiled_copy_O_r2g.retile(tOrOs_t2r[split])
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g = [
                cute.make_rmem_tensor_like(tOrOs_r2g_f32[split], self.dtype_O)
                for split in range(self.num_hdimv_splits)
            ]
            if const_expr(self.use_tma_O):
                tOsO = thr_tiled_copy_O_r2g.partition_D(sO)
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1),
                    sO, gO,
                )

            self.sm_stats_barrier_full.arrive_and_wait()

            row_sum0 = sRowSum[tidx % self.cta_tile_m, 0]
            row_sum1 = sRowSum[tidx % self.cta_tile_m, 1]
            row_sum = row_sum0 + row_sum1
            acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)

            self.sm_stats_barrier_empty.arrive()

            seqlen_q = (
                seqlen.seqlen_q
                if const_expr(not self.pack_gqa)
                else seqlen.seqlen_q * self.qhead_per_kvhead
            )

            # compute and store lse to gmem 
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    lse_offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    mLSE_cur = cute.domain_offset((lse_offset,), mLSE[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (cta_m_block,))
                if tidx < self.cta_tile_m:
                    row_max = sRowMax[tidx, 0]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if tidx < seqlen_q - cta_m_block * self.cta_tile_m:
                        gLSE[tidx] = lse

            row_idx = cta_m_block * self.cta_tile_m + tOicOi[0][0]

            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_Oi = consumer_states_O[split]
                pipelines_O[split].consumer_wait(consumer_state_Oi)
                # copy Oi tmem -> rmem
                cute.copy(
                    thr_tmem_load_O,
                    tOtOs_t2r[split],
                    tOrOs_t2r[split],
                )

                # scale and downcast Oi
                tOrOs_r2g[split].store((tOrOs_r2g_f32[split].load() * scale).to(self.dtype_O))
                
                if const_expr(not self.use_tma_O):
                    # copy Oi rmem -> gmem
                    if row_idx < seqlen_q:
                        cute.copy(
                            thr_tiled_copy_O_r2g,
                            tOrOs_r2g[split],
                            tOgO[None, None, None, split],
                        )
                else:
                    # copy Oi rmem -> smem
                    if const_expr(self.overlap_sO_sV):
                        # last slot for Vti is always 1, 3
                        sO_idx = 1 + 2 * split
                    else:
                        sO_idx = split
                    cute.copy(
                        thr_tiled_copy_O_r2g,
                        tOrOs_r2g[split],
                        tOsO[None, None, None, sO_idx],
                    )
                    cute.arch.fence_view_async_shared()
                    self.epi_barrier.arrive_and_wait()
                    # tma store Oi smem -> gmem
                    if leader_warp:
                        store_O(src_idx=sO_idx, dst_idx=split)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(1 - split, read=True)
                        if const_expr(split == 1 and self.overlap_sO_sV):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(sO_empty_mbar_ptr)

            consumer_state_O0, consumer_state_O1 = consumer_states_O

            cute.arch.fence_view_async_tmem_load()
            pipeline_O0.consumer_release(consumer_state_O0)
            pipeline_O1.consumer_release(consumer_state_O1)
            consumer_state_O0.advance()
            consumer_state_O1.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def correction_rescale(
        self,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tOcO_t2r: cute.Tensor,
        tOtO_t2r: cute.Tensor,
        tOtO_r2t: cute.Tensor,
        scale: Float32,
    ):
        tOrO_t2r_frg = cute.make_rmem_tensor_like(tOcO_t2r[None, None, 0], self.dtype_acc)

        for i in cutlass.range_constexpr(cute.size(tOtO_t2r, mode=[2])):
            tOtO_t2r_cur = tOtO_t2r[None, None, i]
            tOtO_r2t_cur = tOtO_r2t[None, None, i]

            cute.copy(thr_tmem_load, tOtO_t2r_cur, tOrO_t2r_frg)
            for j in cutlass.range(0, cute.size(tOrO_t2r_frg), 2, unroll_full=True):
                tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1]), (scale, scale)
                )
            cute.copy(thr_tmem_store, tOrO_t2r_frg, tOtO_r2t_cur)
        cute.arch.fence_view_async_tmem_store()


def test_mla_kernel(
    seqlen_q=2048,
    seqlen_k=2048,
    topk_length=2048,
    nheads=1,
    batch=1,
    iter=0,
    compile_cache=dict(),
    validate=True,
    seed=0,
    gather_kv=True,
    pack_gqa=False,
    is_causal=False,
    varlen_q=False,
    varlen_k=False,
    disable_bitmask=False,
):
    torch.manual_seed(seed)
    hdim = 64
    hdimv = 512
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv)

    nheads_kv = 1
    qhead_per_kvhead = nheads

    compile_key = (
        is_causal,
        gather_kv,
        topk_length if gather_kv else None,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
        varlen_q,
        varlen_k,
        disable_bitmask,
    )
    if compile_key not in compile_cache:
        total_q_dummy = batch * seqlen_q
        total_k_dummy = batch * seqlen_k

        if varlen_q:
            Q   = torch.randn(total_q_dummy, nheads,    hdim,  dtype=torch.bfloat16, device="cuda")
            Qv  = torch.randn(total_q_dummy, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
            O   = torch.empty(total_q_dummy, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
            lse = torch.empty(nheads, total_q_dummy,           dtype=torch.float32,  device="cuda")
            index_topk = (
                torch.rand(total_q_dummy, topk_length, device="cuda")
                .argsort(dim=-1).to(torch.int32)
            )
            cu_seqlens_q_dummy = torch.arange(
                0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device="cuda"
            )
        else:
            Q   = torch.randn(batch, seqlen_q, nheads,    hdim,  dtype=torch.bfloat16, device="cuda")
            Qv  = torch.randn(batch, seqlen_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
            O   = torch.empty(batch, seqlen_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
            lse = torch.empty(batch, nheads, seqlen_q,           dtype=torch.float32,  device="cuda")
            index_topk = (
                torch.rand(batch, seqlen_q, topk_length, device="cuda")
                .argsort(dim=-1).to(torch.int32)
            )

        if varlen_k:
            K = torch.randn(total_k_dummy, nheads_kv, hdim,  dtype=torch.bfloat16, device="cuda")
            V = torch.randn(total_k_dummy, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")
            cu_seqlens_k_dummy = torch.arange(
                0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device="cuda"
            )
        else:
            K = torch.randn(batch, seqlen_k, nheads_kv, hdim,  dtype=torch.bfloat16, device="cuda")
            V = torch.randn(batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")

        mQ  = from_dlpack(Q,  assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim  - 1)
        mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
        mK  = from_dlpack(K,  assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim  - 1)
        mV  = from_dlpack(V,  assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim  - 1)
        mO  = from_dlpack(O,  assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim  - 1)
        mLSE       = from_dlpack(lse,        assumed_align=4 ).mark_layout_dynamic(leading_dim=lse.ndim - 1)
        if gather_kv:
            mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(leading_dim=index_topk.ndim - 1)
        else:
            mIndexTopk = None

        compile_kwargs = dict(mIndexTopk=mIndexTopk)
        if varlen_q:
            compile_kwargs["mCuSeqlensQ"] = from_dlpack(cu_seqlens_q_dummy, assumed_align=4)
        if varlen_k:
            compile_kwargs["mCuSeqlensK"] = from_dlpack(cu_seqlens_k_dummy, assumed_align=4)

        kernel = cute.compile(
            FlashAttentionMLAForwardSm100(
                is_causal=is_causal,
                use_cpasync_load_KV=gather_kv,
                topk_length=topk_length if gather_kv else 2048,
                is_topk_gather=gather_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                nheads_kv=nheads_kv,
                is_varlen_q=varlen_q,
                disable_bitmask=disable_bitmask,
            ),
            mQ, mQv, mK, mV, mO, mLSE, softmax_scale,
            **compile_kwargs,
            options="--keep-ptx --keep-cubin --generate-line-info"
        )
        dump_kernel_attributes(kernel)
        compile_cache[compile_key] = kernel

    # ================================================================
    # ---- Generate variable seqlens for this run ----
    if varlen_q:
        torch.manual_seed(seed + 1000)
        # When causal without varlen_k, every per-batch seqlen_q must not exceed seqlen_k.
        max_seqlen_q = (seqlen_k if (is_causal and not varlen_k) else seqlen_q)
        seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch,), dtype=torch.int32)
        cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1:] = seqlens_q.cumsum(0).to(torch.int32).cuda()
        total_q = cu_seqlens_q[-1].item()
    else:
        seqlens_q = torch.full((batch,), seqlen_q, dtype=torch.int32)
        total_q = None   # unused

    if varlen_k:
        torch.manual_seed(seed + 2000)
        # Each batch item must have at least topk_length keys so topk gather is valid.
        min_seqlen_k = topk_length if gather_kv else 1
        seqlens_k = torch.randint(min_seqlen_k, seqlen_k + 1, (batch,), dtype=torch.int32)
        # When causal, every batch item needs seqlens_k[b] >= seqlens_q[b].
        if is_causal:
            seqlens_k = torch.maximum(seqlens_k, seqlens_q)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_k[1:] = seqlens_k.cumsum(0).to(torch.int32).cuda()
        total_k = cu_seqlens_k[-1].item()
    else:
        seqlens_k = torch.full((batch,), seqlen_k, dtype=torch.int32)
        total_k = None   # unused

    torch.manual_seed(seed)   # restore main seed before drawing actual tensors

    # ---- Allocate Q / Qv / O / lse ----
    if varlen_q:
        Q   = torch.randn(total_q, nheads,    hdim,  dtype=torch.bfloat16, device="cuda")
        Qv  = torch.randn(total_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
        O   = torch.empty(total_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(nheads, total_q,           dtype=torch.float32,  device="cuda")
    else:
        Q   = torch.randn(batch, seqlen_q, nheads,    hdim,  dtype=torch.bfloat16, device="cuda")
        Qv  = torch.randn(batch, seqlen_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
        O   = torch.empty(batch, seqlen_q, nheads,    hdimv, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(batch, nheads, seqlen_q,           dtype=torch.float32,  device="cuda")

    # ---- Allocate K / V ----
    if varlen_k:
        K = torch.randn(total_k, nheads_kv, hdim,  dtype=torch.bfloat16, device="cuda")
        V = torch.randn(total_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")
    else:
        K = torch.randn(batch, seqlen_k, nheads_kv, hdim,  dtype=torch.bfloat16, device="cuda")
        V = torch.randn(batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")

    # ---- Generate index_topk with per-batch valid ranges when varlen_k ----
    # index_topk shape: (total_q, topk_length) if varlen_q else (batch, seqlen_q, topk_length)
    if gather_kv:
        topk_parts = []
        for b in range(batch):
            sl_q_b = seqlens_q[b].item()
            sl_k_b = seqlens_k[b].item()
            # Draw topk_length unique indices from [0, sl_k_b) for each query in this batch item.
            topk_b = (
                torch.rand(sl_q_b, sl_k_b, device="cuda")
                .argsort(dim=-1)[..., :topk_length]
                .to(torch.int32)
            )  # (sl_q_b, topk_length), all < sl_k_b
            topk_parts.append(topk_b)

        if varlen_q:
            index_topk = torch.cat(topk_parts, dim=0)          # (total_q, topk_length)
        else:
            index_topk = torch.stack(topk_parts, dim=0)        # (batch, seqlen_q, topk_length)
    else:
        index_topk = None

    # ---- Reference computation (per-batch loop covers all four varlen combos) ----
    O_ref_list, O_pt_list, lse_ref_list, lse_pt_list = [], [], [], []
    for b in range(batch):
        qs = cu_seqlens_q[b].item()   if varlen_q else b * seqlen_q
        qe = cu_seqlens_q[b+1].item() if varlen_q else (b + 1) * seqlen_q
        ks = cu_seqlens_k[b].item()   if varlen_k else b * seqlen_k
        ke = cu_seqlens_k[b+1].item() if varlen_k else (b + 1) * seqlen_k

        Q_b  = Q [qs:qe].unsqueeze(0) if varlen_q else Q [b:b+1]   # (1, sl_q, nheads, hdim)
        Qv_b = Qv[qs:qe].unsqueeze(0) if varlen_q else Qv[b:b+1]   # (1, sl_q, nheads, hdimv)
        K_b  = K [ks:ke].unsqueeze(0) if varlen_k else K [b:b+1]   # (1, sl_k, nheads_kv, hdim)
        V_b  = V [ks:ke].unsqueeze(0) if varlen_k else V [b:b+1]   # (1, sl_k, nheads_kv, hdimv)
        if gather_kv:
            topk_b = index_topk[qs:qe].unsqueeze(0) if varlen_q else index_topk[b:b+1]
        else:
            topk_b = None

        O_b,    _, lse_b    = attention_ref(Q_b, K_b, V_b, qv=Qv_b, causal=is_causal,
                                             return_lse=True, gather_kv_indices=topk_b)
        O_pt_b, _, lse_pt_b = attention_ref(Q_b, K_b, V_b, qv=Qv_b, causal=is_causal,
                                             upcast=False, reorder_ops=True,
                                             return_lse=True, gather_kv_indices=topk_b)
        O_ref_list.append(O_b.squeeze(0))
        O_pt_list.append(O_pt_b.squeeze(0))
        lse_ref_list.append(lse_b.squeeze(0))
        lse_pt_list.append(lse_pt_b.squeeze(0))

    cat_dim_o   = 0  if (varlen_q) else 0   # always 0: leading token/batch dim
    cat_dim_lse = -1 if (varlen_q) else -1  # always last: token dim

    if varlen_q:
        O_ref   = torch.cat(O_ref_list,   dim=0)    # (total_q, nheads, hdimv)
        O_pt    = torch.cat(O_pt_list,    dim=0)
        lse_ref = torch.cat(lse_ref_list, dim=-1)   # (nheads, total_q)
        lse_pt  = torch.cat(lse_pt_list,  dim=-1)
    else:
        O_ref   = torch.stack(O_ref_list,   dim=0)  # (batch, seqlen_q, nheads, hdimv)
        O_pt    = torch.stack(O_pt_list,    dim=0)
        lse_ref = torch.stack(lse_ref_list, dim=0)  # (batch, nheads, seqlen_q)
        lse_pt  = torch.stack(lse_pt_list,  dim=0)

    rtol = 2
    atol = 2 * (O_ref + 0.3 - 0.3 - O_ref).abs().max().item()

    # ---- CuTe tensor wrappers ----
    mQ  = from_dlpack(Q,  assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim  - 1)
    mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
    mK  = from_dlpack(K,  assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim  - 1)
    mV  = from_dlpack(V,  assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim  - 1)
    mO  = from_dlpack(O,  assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim  - 1)
    mLSE       = from_dlpack(lse,        assumed_align=4 ).mark_layout_dynamic(leading_dim=lse.ndim - 1)
    if index_topk is not None:
        mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(leading_dim=index_topk.ndim - 1)
    else:
        mIndexTopk = None

    run_kwargs = dict(mIndexTopk=mIndexTopk)
    if varlen_q:
        run_kwargs["mCuSeqlensQ"] = from_dlpack(cu_seqlens_q, assumed_align=4)
    if varlen_k:
        run_kwargs["mCuSeqlensK"] = from_dlpack(cu_seqlens_k, assumed_align=4)

    # ---- Run kernel ----
    compile_cache[compile_key](
        mQ, mQv, mK, mV, mO, mLSE, softmax_scale,
        **run_kwargs,
    )

    print(f"Pytorch max O diff: {(O_pt - O_ref).abs().max().item()}")
    print(f"Pytorch mean O diff: {(O_pt - O_ref).abs().mean().item()}")
    print(f"Max abs diff O, O_ref: {(O - O_ref).abs().max().item()}")
    print(f"Mean abs diff O, O_ref: {(O - O_ref).abs().mean().item()}")

    # print(f"Pytorch LSE max diff: {(lse_pt - lse_ref).abs().max().item()}")
    # print(f"Pytorch LSE mean diff: {(lse_pt - lse_ref).abs().mean().item()}")
    # print(f"Max abs diff LSE: {(lse - lse_ref).abs().max().item()}")
    # print(f"Mean abs diff LSE: {(lse - lse_ref).abs().mean().item()}")

    if validate:
        assert (O - O_ref).abs().max().item() <= rtol * (O_pt - O_ref).abs().max().item() + atol
        varlen_tag = ""
        if varlen_q: varlen_tag += f", total_q:{total_q}"
        if varlen_k: varlen_tag += f", total_k:{total_k}"
        print(
            f"batch:{batch:3d}, nheads:{nheads:3d}, seqlen_q:{seqlen_q:5d}, seqlen_k:{seqlen_k:5d}"
            f"{varlen_tag}, iter:{iter:2d} PASSED"
        )
    else:
        print(mO)
        print(
            f"batch:{batch:3d}, nheads:{nheads:3d}, seqlen_q:{seqlen_q:5d}, seqlen_k:{seqlen_k:5d}"
            f", iter:{iter:2d} RUN (NOT TESTING CORRECTNESS)"
        )

    return None

def timeit(fn, *args, **kwargs):
    # Synchronize before timing
    torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        fn(*args, **kwargs)

    # Benchmark using PyTorch's Timer
    t = benchmark.Timer(
        stmt="fn(*args, **kwargs)", globals={"fn": fn, "args": args, "kwargs": kwargs}
    )

    # Time it multiple runs
    measurement = t.timeit(20)  # 20 repeats
    avg_time = measurement.mean  # Average time in seconds

    time.sleep(1)

    return avg_time


def benchmark_mla_kernel(
    batch=1, seqlen_q=2048, seqlen_k=2048, topk_length=2048, nheads=128, hdim=64, hdimv=512, compile_cache=dict(),
    gather_kv=True, is_causal=False, disable_bitmask=False,
):
    assert hdim == 64, "hdim must be 64"
    assert hdimv == 512, "hdimv must be 512"

    qhead_per_kvhead=nheads
    nheads_kv=1
    pack_gqa=True
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv)

    compile_key = (
        is_causal,
        gather_kv,
        topk_length if gather_kv else None,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
        disable_bitmask,
    )
    if compile_key not in compile_cache:
        Q = torch.randn(batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda")
        Qv = torch.randn(batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")
        O = torch.empty(batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
        index_topk = torch.rand(batch, seqlen_q, topk_length, device="cuda").argsort(dim=-1).to(torch.int32)

        mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
        mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
        mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
        mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
        mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
        if gather_kv:
            mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(leading_dim=index_topk.ndim - 1)
        else:
            mIndexTopk = None

        mLSE = None

        kernel = cute.compile(
            FlashAttentionMLAForwardSm100(
                is_causal=is_causal,
                use_cpasync_load_KV = gather_kv,
                topk_length = topk_length if gather_kv else 2048,
                is_topk_gather = gather_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                nheads_kv=nheads_kv,
                disable_bitmask=disable_bitmask,
            ),
            mQ, mQv, mK, mV, mO, mLSE, softmax_scale,
            mIndexTopk=mIndexTopk,
        )
        compile_cache[compile_key] = kernel
    
    Q = torch.randn(batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda")
    Qv = torch.randn(batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")
    O = torch.empty(batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")

    index_topk = torch.rand(batch, seqlen_q, topk_length, device="cuda").argsort(dim=-1).to(torch.int32)

    mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
    mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    if gather_kv:
        mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(leading_dim=index_topk.ndim - 1)
    else:
        mIndexTopk = None
    mLSE = None

    exec_time_in_s = timeit(
        compile_cache[compile_key],
        mQ, mQv, mK, mV, mO, mLSE, softmax_scale,
        mIndexTopk=mIndexTopk,
    )
    
    seqlen_k_eff = topk_length if gather_kv else seqlen_k

    FLOPs = 2 * batch * nheads * seqlen_q * seqlen_k_eff * (hdim + 2 * hdimv)
    if is_causal and not gather_kv:
        FLOPs /= 2

    TFLOPS = FLOPs / exec_time_in_s / 1e12

    q_bytes = 2 * batch * nheads * seqlen_q * hdim
    qv_bytes = 2 * batch * nheads * seqlen_q * hdimv
    k_bytes = 2 * batch * nheads_kv * seqlen_k_eff * hdim
    v_bytes = 2 * batch * nheads_kv * seqlen_k_eff * hdimv
    o_bytes = 2 * batch * nheads * seqlen_q * hdimv
    total_bytes = q_bytes + qv_bytes + k_bytes + v_bytes + o_bytes
    TBs = total_bytes / exec_time_in_s / 1e12

    print(
        f"batch: {batch}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, nheads: {nheads}, -> {exec_time_in_s * 1e3:.2f} ms, {TFLOPS:.2f} TFLOPS, {TBs:.2f} TBs"
    )


if __name__ == "__main__":
    run_test = True
    run_benchmark = True
    gather_kv = False
    is_causal = True
    pack_gqa = True
    topk_length = 2048
    varlen_q = False
    varlen_k = False
    disable_bitmask = True
    validate = True

    if run_test:
        if not gather_kv:
            seqlen_q_test_values = range(1, 4002, 400)
            seqlen_k_test_values = range(1, 4002, 400)
        else:
            seqlen_q_test_values = range(1, 1001, 200)
            seqlen_k_test_values = range(topk_length, 9001, 2000)
        seqlen_q_test_values = [1]
        seqlen_k_test_values = [4096]
        nheads_test_values = [128]
        batch_test_values = [4]
        test_configs = [
            (batch, nheads, seqlen_q, seqlen_k,)
            for batch in batch_test_values
            for nheads in nheads_test_values
            for seqlen_q in seqlen_q_test_values
            for seqlen_k in seqlen_k_test_values
        ]
        iters_per_config = 1
        compile_cache = dict()
        print("=" * 40)
        print("Testing MLA Kernel")
        print("=" * 40)
        for config in test_configs:
            batch, nheads, seqlen_q, seqlen_k = config
            # if is_causal and seqlen_k < seqlen_q:
            #     continue
            for iter in range(iters_per_config):
                test_mla_kernel(
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    topk_length=topk_length,
                    nheads=nheads,
                    batch=batch,
                    iter=iter,
                    compile_cache=compile_cache,
                    validate=validate,
                    seed=0,
                    gather_kv=gather_kv,
                    pack_gqa=pack_gqa,
                    is_causal=is_causal,
                    varlen_q=varlen_q,
                    varlen_k=varlen_k,
                    disable_bitmask=disable_bitmask,
                )
    if run_benchmark:
        if gather_kv:
            seqlen_q_benchmark_values = [1]
            seqlen_k_benchmark_values = [8192*2]
            nheads_benchmark_values = [128]
            batch_benchmark_values = [512]
        else:
            seqlen_q_benchmark_values = [1]
            seqlen_k_benchmark_values = [8192*2]
            nheads_benchmark_values = [128]
            batch_benchmark_values = [512]
        seqlen_q_benchmark_values = [4096]
        seqlen_k_benchmark_values = [4096]
        nheads_benchmark_values = [16]
        batch_benchmark_values = [8]
        benchmark_configs = [ (batch, nheads, seqlen_q, seqlen_k,)
                for batch in batch_benchmark_values
                for nheads in nheads_benchmark_values
                for seqlen_q in seqlen_q_benchmark_values
                for seqlen_k in seqlen_k_benchmark_values
                ]
        compile_cache=dict()
        print("="*40)
        print("Benchmarking MLA Kernel")
        print("="*40)
        for config in benchmark_configs:
            batch, nheads, seqlen_q, seqlen_k = config
            benchmark_mla_kernel(
                batch = batch,
                seqlen_q = seqlen_q,
                seqlen_k = seqlen_k,
                topk_length = topk_length,
                nheads = nheads,
                gather_kv=gather_kv,
                is_causal=is_causal,
                disable_bitmask=disable_bitmask,
                compile_cache=compile_cache
            )
''',
    'key-files/flash_attn/cute/interface.py': r'''# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    to_cute_tensor, to_cute_aux_tensor, get_aux_tensor_metadata, get_broadcast_dims,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)

def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = head_dim == 64  and head_dim_v == 512
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_deepseek_mla_absorbed_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64, n_block_size=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )



def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim) if qv is None else 1.0 / math.sqrt(head_dim + head_dim_v)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default()

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is None:
        if arch // 10 == 12:
            # SM120 tile sizes tuned for 99 KB SMEM capacity:
            # D<=64:  128x128 → 48 KB (good occupancy)
            # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
            if head_dim <= 64:
                fwd_cfg = FwdConfig(128, 128, True, True)
            else:
                fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q)
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    # TODO: fix GQA + SplitKV + non-varlen
    if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
        pack_gqa = False
    
    if pack_gqa and qv is not None and 128 % qhead_per_kvhead != 0:
        pack_gqa = False

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k 
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
        
    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    if mask_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[1] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim == 64 and head_dim_v == 512, "only support MLA weight absorbed shape with qv"
        assert not local, "local not yet supported with qv"
        assert page_table is None, "page table not yet supported with qv"

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None
        
        qv = maybe_contiguous(qv)

        gather_kv_length = 2048
        sparse_kv = gather_kv_indices is not None
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == q.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 256 == 0
            if min_seqlen_k is None or causal:
                disable_sparse_kv_bitmask = False
            else:
                # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
                seqlen_k_boundary = min_seqlen_k
                disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
    else:
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        requested_use_clc_scheduler,
        qv is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv) if qv is not None else None
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices) if gather_kv_indices is not None else None

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                )
            else:
                fa_fwd = FlashAttentionForwardSm100(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_persistent=not causal
                        and not local
                        and cu_seqlens_q is None
                        and seqused_q is None
                        and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, tile_n],
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=requested_use_clc_scheduler,
                )
        elif arch // 10 == 12:
            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
                sparse_tensors,
                cute_aux_tensors,
                current_stream,
                options="--enable-tvm-ffi",
            )

    # In "fake mode", we will take torch fake tensors as input and the expected behaviors are:
    # - Use those fake metadata to populate compilation cache
    # - Return "fake" output tensors, which could be needed in follow-up fake operations
    # Thus, we skip the actual kernel invocation here.
    if not is_fake_mode():
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q.detach(),
                qv.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        else:
            _flash_attn_fwd.compile_cache[compile_key](
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
                normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
                aux_tensors,
            )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    # divisibility in elements: assumed_align_bytes = divisibility * dtype.width // 8
    # For 16-byte align: fp16/bf16 → divisibility=8, float32 → divisibility=4
    div = 128 // dtype.width  # 8 for fp16/bf16
    # Shared sym_ints for dimensions that must match across tensors
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(
    dtype, head_dim, head_dim_v, m_block_size, has_cuseqlens_q, has_seqused_q, has_dlse,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size, is_varlen, seqused_q is not None, dlse is not None,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, cu_seqlens, seqused,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    sparse_q = None
    if block_sparse_tensors is not None and arch // 10 == 9:
        sparse_q = block_sparse_tensors.block_size[0] if block_sparse_tensors.block_size is not None else 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).
        m_block_size = 64
        n_block_size = 64
        if head_dim <= 64:
            num_stages_Q = 2
            num_stages_dO = 2
        else:
            num_stages_Q = 1
            num_stages_dO = 1
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        V_in_regs = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 128
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = utils._get_disable_2cta_default()
        disable_2cta = (
            requested_disable_2cta
            or score_mod is not None
            or score_mod_bwd is not None
            or mask_mod is not None
            or block_sparse_tensors is not None
        )
        cluster_size = 2 if head_dim >= 128 and not disable_2cta else 1
        use_2cta_instrs = cluster_size==2

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 12:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False
    
    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    elif score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "varlen + score_mod not supported in bwd yet"
        )
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    dKV_postprocess = qhead_per_kvhead > 1
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
    )
    # num_threads: SM90 derives from BwdConfig.num_wg, SM120 is set to 128 above,
    # SM100/SM110 uses default from function signature (384).
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            deterministic,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
        )
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                subtile_factor=subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                is_local=local,
                qhead_per_kvhead=qhead_per_kvhead,
                tile_m=m_block_size,
                tile_n=n_block_size,
                cluster_size=cluster_size,
                use_2cta_instrs=use_2cta_instrs,
                deterministic=deterministic,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                subtile_factor=subtile_factor,
            )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            aux_tensors,
            normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
        )

    if arch // 10 == 9:
        # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
        num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
        num_threads_post_dKV = cfg.num_wg * 128
    else:
        num_threads_post_dQ = 128
        num_threads_post_dKV = 128

    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    _bwd_postprocess_convert(
        dq_accum, dq, softmax_scale,
        cu_seqlens_q, seqused_q,
        arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
        AtomLayoutMdQ, dQ_swapAB,
        use_2cta_instrs=use_2cta_instrs, cluster_size=1,
    )

    if dKV_postprocess:
        # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
        _bwd_postprocess_convert(
            dk_accum, dk, softmax_scale,
            cu_seqlens_k, seqused_k,
            arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
            AtomLayoutNdKV, dKV_swapAB,
            cluster_size=cluster_size,
        )
        # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
        _bwd_postprocess_convert(
            dv_accum, dv, 1.0,
            cu_seqlens_k, seqused_k,
            arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
            AtomLayoutNdKV, dKV_swapAB,
            cluster_size=cluster_size,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[torch.Tensor] = None,
        full_block_idx: Optional[torch.Tensor] = None,
        mask_block_cnt: Optional[torch.Tensor] = None,
        mask_block_idx: Optional[torch.Tensor] = None,
        block_size: Optional[Tuple[int, int]] = None,
        return_lse: bool = False,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
                block_size=block_size,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
            dlse=dlse,
        )
        return dq, dk, dv, *((None,) * 30)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            dlse=dlse,
        )

        return dq, dk, dv, *((None,) * 30)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
    block_size: Optional[Tuple[int, int]] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
        block_size,
        return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
):
    """
    Explanation of some optional arguments:

    qv: we write the MLA weight absorbed formula as
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache.

    gather_kv_indices: a tensor of shape (batch, seqlen_q, gather_kv_length) or
        (total_q, gather_kv_length) if there is cu_seqlens_q.
        Currently, only used for topk sparsity with MLA absorption kernel.
    
    min_seqlen_k: for varlen, specifies the minimum kv sequence length for any batch.
        Used with gather_kv_indices to determine if we need oob masking.
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        aux_tensors,
        return_lse,
    )


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = None  # Not parametrized in compile_key

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        varlen_batch_idx: Optional mapping from virtual batch index to real batch index
            (int32 tensor of shape (batch_size,)). Used by persistent tile schedulers
            that reorder batch processing for load balancing.
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=device)
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        varlen_batch_idx=varlen_batch_idx,
    )
    return out, lse
''',
    'key-files/flash_attn/cute/mask.py': r'''# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable, TypeAlias
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr, Boolean

from quack import layout_utils
import flash_attn.cute.utils as utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions >= limit (inclusive lower bound).

    Positions limit..31 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
    """Transform SM90 MMA column coordinate to R2P element index.

    SM90 MMA accumulator column indices are non-contiguous: 0, 1, 8, 9, 16, 17, ...
    Element indices are contiguous: 0, 1, 2, 3, 4, 5, ...
    This converts a column-space threshold to element-space for r2p_bitmask_below/above.
    """
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    """Convert a row coordinate to an R2P element index in the warp-group interleaved layout.

    In the SM100 backward pass, 2 warp groups share TMEM. The TMEM load atom
    distributes rows in an interleaved pattern: elements 0..num_rep-1 map to
    rows 0..num_rep-1 (warp group 0), elements num_rep..2*num_rep-1 map to
    rows num_rep*num_wg..num_rep*num_wg+num_rep-1 (warp group 1), and so on.
    Row-coordinate thresholds (causal limits, window bounds, uih_len) must be
    converted to element indices before use with r2p_bitmask_above/below.

    Rows not owned by this thread (in the gap between warp groups) are clamped
    to the boundary element index, which is safe because R2P thresholds are
    monotonic.

    Example with num_rep=16, num_wg=2:
        row  0 -> elem  0,  row 15 -> elem 15,
        row 16 -> elem 16 (clamped), row 31 -> elem 16 (clamped),
        row 32 -> elem 16, row 33 -> elem 17, row 47 -> elem 31.
    """
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                r2p = const_expr(not self.swap_AB)
                if const_expr(not r2p):
                    # traverse column index.
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                    mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

        elif const_expr(
            not mask_causal and not mask_local and mask_mod is not None
        ):  # FlexAttention mask mod
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            wrap_aux_indices = const_expr(
                has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
            )

            for r in cutlass.range_constexpr(nrow):
                # Respect swap_AB: ROW/COL determine which coordinate component corresponds to Q/KV.
                local_row = tScS_mn[r, 0][ROW]
                global_row_idx = local_row + m_block * self.tile_m
                row_for_mod = global_row_idx
                head_idx_for_mod = head_idx
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
                row_for_seqlen = row_for_mod
                if const_expr(wrap_aux_indices):
                    _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][COL]
                    # Convert to absolute column index
                    global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
                    col_for_mod = global_col_idx
                    if const_expr(wrap_aux_indices):
                        _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

                    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                    head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                    q_idx_ssa = utils.scalar_to_ssa(row_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(col_for_mod, cutlass.Int32)
                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    if const_expr(mask_seqlen):
                        out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                    else:
                        acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf

        else:  # Causal or local
            if const_expr(not self.swap_AB):
                # If PackGQA, we split the work of compute divmod among threads in the same row
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(not self.swap_AB)  # R2P trick, see apply_mask_sm100
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            mask_r2p_lambda(
                                acc_S_mn[r, None],
                                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                                rank1=True,
                            )
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    r2p_local = const_expr(not self.swap_AB)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        if const_expr(not r2p_local):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                col_idx = t0ScS_mn[0, c][1]
                                if col_idx >= col_limit_right or col_idx < col_limit_left:
                                    acc_S_mn[r, c] = -Float32.inf
                        else:
                            col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

                            def mask_gen_fn(s: int) -> Uint32:
                                return r2p_bitmask_below(
                                    col_limit_right_r2p, s
                                ) & r2p_bitmask_above(col_limit_left_r2p, s)

                            mask_r2p_lambda(acc_S_mn[r, None], mask_gen_fn, rank1=True)
            else:  # swap_AB
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else (
                                col0 - causal_row_offset - self.window_size_right
                                if const_expr(self.window_size_right is not None)
                                else 0
                            )
                        )
                        row_limit_bot = (
                            col0 - causal_row_offset + self.window_size_left
                            if const_expr(self.window_size_left is not None)
                            else self.tile_m
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
        r2p: bool = True,
        rBitmask: Optional[cute.Tensor] = None,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n

        if const_expr(rBitmask is not None):
            ncol_packed = const_expr(cute.size(rBitmask.shape[0]))
            for i in cutlass.range_constexpr(ncol_packed):
                col_start = 32 * i  # mask is bit-packed into uint32
                curr_mask_val = rBitmask[i]
                for j in cutlass.range_constexpr(32):
                    curr_col = col_start + j
                    mask = (curr_mask_val >> j) & 1
                    acc_S[curr_col] = acc_S[curr_col] if Boolean(mask) else -Float32.inf

        elif const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
                        rank1=True,
                    )

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case w/ mask_mod
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)

            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
                col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
                global_row = row_coord + m_block * self.tile_m
                global_col = col_coord + n_block * self.tile_n

                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert head_divmod is not None
                    mask_row, head_offset = divmod(global_row, head_divmod)
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                else:
                    head_idx_for_mod = head_idx
                    mask_row = global_row

                mask_row_for_mod = mask_row
                if const_expr(has_fastdiv and aux_tensors is not None):
                    if check_q_boundary:
                        _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
                global_col_for_mod = global_col
                if const_expr(has_fastdiv and mask_seqlen and aux_tensors is not None):
                    _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

                head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
                kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
                mask_value = mask_mod(
                    batch_idx_ssa,
                    head_idx_ssa,
                    mask_row_ssa,
                    kv_idx_ssa,
                    self.seqlen_info,
                    aux_tensors,
                )
                cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                acc_S[i] = acc_S[i] if cond else -Float32.inf
                if const_expr(mask_seqlen):
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

        else:  # Causal or local
            causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset + 1
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(col_limit_right, s),
                        rank1=True,
                    )
            else:
                local_row_offset_right = (
                    causal_row_offset + 1 + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # Dual-bound R2P masking for SM100.
                    # Masks elements where: NOT (col_limit_left <= col < col_limit_right)

                    def mask_gen_fn(s: int) -> Uint32:
                        return r2p_bitmask_below(col_limit_right, s) & r2p_bitmask_above(
                            col_limit_left, s
                        )

                    mask_r2p_lambda(acc_S, mask_gen_fn, rank1=True)

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        """
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.

        Coordinate conventio:
        - ROW corresponds to Q (m_block)
        - COL corresponds to KV (n_block)

        is_full_block: If True, skip mask_mod (all elements valid). Only apply seqlen masking.
        check_m_boundary: If False, skip seqlen_q boundary check (optimization for non-boundary m_blocks).
                          When iterating m_blocks in forward order, only the last m_block may be partial.
        """
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        # assert t0ScS_t2r[0][COL] == 0, "col0 == 0" # tmp comment for 2-cta bwd
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case with mask_mod (backward)
            #
            # Coordinate convention: ROW → Q (m_block), COL → KV (n_block).
            # These already account for swap_AB.
            #
            # FULL blocks: mask_mod returns True for all elements, so skip it.
            #   Still need seqlen bounds check (elements may be OOB on last m_block).
            # PARTIAL blocks: apply mask_mod element-wise, then seqlen bounds.
            if is_full_block:
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        # Entire tile is OOB for K
                        for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                            acc_S[i] = -cutlass.Float32.inf
                    elif check_m_boundary:
                        # Last m_block: check Q and K boundaries
                        ncol = const_expr(cute.size(tScS_t2r.shape))
                        for i in cutlass.range_constexpr(ncol):
                            row_coord = tScS_t2r[i][ROW]
                            col_coord = tScS_t2r[i][COL]
                            global_q = row_coord + m_block * self.tile_m
                            global_kv = col_coord + n_block * self.tile_n
                            q_out_of_bounds = global_q >= self.seqlen_q
                            kv_out_of_bounds = global_kv >= self.seqlen_k
                            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                            acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]
            else:
                # Partial block
                has_fastdiv = const_expr(
                    fastdiv_mods is not None
                    and fastdiv_mods[0] is not None
                    and fastdiv_mods[1] is not None
                )
                wrap_aux_indices = const_expr(
                    has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
                )
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)

                ncol = const_expr(cute.size(tScS_t2r.shape))
                for i in cutlass.range_constexpr(ncol):
                    row_coord = tScS_t2r[i][ROW]
                    col_coord = tScS_t2r[i][COL]
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n

                    q_idx_for_mod = global_q
                    kv_idx_for_mod = global_kv
                    if const_expr(wrap_aux_indices):
                        _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
                        _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

                    q_idx_ssa = utils.scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

                    mask_value = mask_mod(
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_tensors,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

                    if const_expr(mask_seqlen):
                        # check_m_boundary=False skips q check for non-boundary m_blocks
                        q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
                        kv_out_of_bounds = global_kv >= self.seqlen_k
                        out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                        acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:  # Causal or local
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                # tidx = cute.arch.thread_idx()[0] % 256
                # if tidx < 32:
                #     cute.printf("tidx = {}, {} {}, {} {}", tidx, tScS_t2r[0][0], tScS_t2r[0][1], tScS_t2r[1][0], tScS_t2r[1][1])
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    # If col is beyond the column limit, we want to mask out the entire
                    # column, by setting row limit to be self.tile_m.
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])  # 16 or 32
                    num_wg = 2
                    row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_above(row_limit, s),
                        rank1=True,
                    )
            else:
                if const_expr(self.window_size_right is not None):
                    row_limit_top = causal_offset - self.window_size_right
                else:
                    row_limit_top = 0
                if const_expr(self.window_size_left is not None):
                    row_limit_bot = causal_offset + self.window_size_left
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        row_idx = t0ScS_t2r[i][ROW]
                        local_mask = row_idx < row_limit_top
                        if const_expr(self.window_size_left is not None):
                            local_mask |= row_idx > row_limit_bot
                        acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
                else:

                    def mask_gen_fn(s: int) -> Uint32:
                        num_rep = cute.size(tScS_t2r, mode=[0])
                        num_wg = 2

                        row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                        mask = r2p_bitmask_above(row_limit, s)

                        if const_expr(self.window_size_left is not None):
                            row_limit_bottom = row_to_r2p_idx(row_limit_bot + 1, num_rep, num_wg)
                            mask = mask & r2p_bitmask_below(row_limit_bottom, s)

                        return mask

                    mask_r2p_lambda(
                        acc_S,
                        mask_gen_fn,
                        rank1=True,
                    )
''',
    'key-files/flash_attn/cute/named_barrier.py': r"""# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()


class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()


class NamedBarrierBwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PdS = enum.auto()
    dQFullWG0 = enum.auto()
    dQFullWG1 = enum.auto()
    dQFullWG2 = enum.auto()
    dQEmptyWG0 = enum.auto()
    dQEmptyWG1 = enum.auto()
    dQEmptyWG2 = enum.auto()


class NamedBarrierBwdSm100(enum.IntEnum):
    EpilogueWG1 = enum.auto()
    EpilogueWG2 = enum.auto()
    Compute = enum.auto()
    dQaccReduce = enum.auto()
    TmemPtr = enum.auto()


class NamedBarrierFwdSm100_MLA2CTA(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    Cpasync = enum.auto()
    Softmax = enum.auto()
    SoftmaxStatsFull = enum.auto()
    SoftmaxStatsEmpty = enum.auto()
""",
    'key-files/flash_attn/cute/softmax.py': r'''# Copyright (c) 2025, Tri Dao.

import math
import operator
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Boolean

from quack import layout_utils
import flash_attn.cute.utils as utils
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.seqlen_info import SeqlenInfoQK


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            # Update row_max before changing row_max_cur to safe value for -inf
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                row_sum[r] += cute.math.exp2(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
                )

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
        )

    @cute.jit
    def compute_row_max_local(self, acc_S_row: cute.TensorSSA, is_first: Boolean) -> Float32:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        return row_max_new

    @cute.jit
    def update_row_max_from_local(
        self,
        row_max_new: Float32,
        is_first: Boolean,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        # self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=self.row_sum[0] * row_scale)
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)
        # tmp = self._compute_row_sum(acc_S_row_exp)
        # self.row_sum[0] = self.row_sum[0] * row_scale + tmp

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        row_max_scaled = row_max * self.scale_log2
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                if cutlass.const_expr(ex2_emu_freq == 0):
                    acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                else:
                    if cutlass.const_expr(
                        k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(
                            acc_S_row_frg[k + 1, j], fastmath=True
                        )
                    else:
                        # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        #     acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
        #         (acc_S_row[i], acc_S_row[i + 1]),
        #         (self.scale_log2, self.scale_log2),
        #         (minus_row_max_scaled, minus_row_max_scaled),
        #     )
        #     acc_S_row[i] = cute.math.exp2(acc_S_row[i], fastmath=True)
        #     acc_S_row[i + 1] = cute.math.exp2(acc_S_row[i + 1], fastmath=True)

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                #     cute.arch.fma_packed_f32x2(
                #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                #         (self.scale_log2, self.scale_log2),
                #         (minus_row_max_scaled, minus_row_max_scaled),
                #     )
                # )
                # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert q_idx to packed format for Pack-GQA."""
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
                        If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor (for bwd kernel where S is transposed)
    """
    # Index positions in the index_tensor tuple
    # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
    # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)

    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # SSA values for batch (constant across all elements)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))

    # Handle q_idx based on whether it's constant
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    # since a thread my process multiple query head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            # Extract head offset from packed q_idx for Pack-GQA
            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            # If we will do loads we mod, in order to not read OOB
            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        # Convert to SSA for score_mod call
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to((vec_size,))

        # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        post_mod_scores = score_mod(
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            seqlen_info=seqlen_info,
            aux_tensors=aux_args,
        )

        # Write back modified scores
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]


@cute.jit
def apply_score_mod_bwd_inner(
    grad_tensor,
    score_tensor,
    index_tensor,
    score_mod_bwd: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    seqlen_info,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    """Apply backward score modification (joint graph).

    Args:
        grad_tensor: in/out: dlogits rewritten in-place with d(scaled_scores)
        score_tensor: pre-mod scores (unscaled QK tile), scaled by softmax_scale internally
        index_tensor: Index positions (same as forward)
        score_mod_bwd: The backward score modification function (joint graph)
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply to score_tensor
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
        qhead_per_kvhead: Pack-GQA replication factor
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor
    """
    # Index positions in the index_tensor tuple
    # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
    # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)
    n_vals = cutlass.const_expr(cute.size(grad_tensor.shape))
    grad_vec = cute.make_fragment(vec_size, qk_acc_dtype)
    score_vec = cute.make_fragment(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))
    q_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_fragment(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_vec[j] = grad_tensor[i + j]
            # Scale score so joint graph sees same value as forward score_mod
            score_vec[j] = score_tensor[i + j] * softmax_scale

            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        grad_ssa = grad_vec.load()
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()

        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            q_idx_ssa = utils.scalar_to_ssa(constant_q_idx, cutlass.Int32).broadcast_to((vec_size,))

        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        grad_out_ssa = score_mod_bwd(
            grad_ssa,
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            seqlen_info=seqlen_info,
            aux_tensors=aux_args,
        )

        grad_vec.store(grad_out_ssa)
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_tensor[i + j] = grad_vec[j]
''',
    'key-files/flash_attn/cute/testing.py': r'''import math
from contextlib import nullcontext
from functools import wraps
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    in_fake_mode = active_fake_mode() is not None
    if not in_fake_mode:
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
    else:
        # torch.nonzero and .item() are not supported in FakeTensorMode
        batch_size, seqlen = attention_mask.shape
        indices = torch.arange(batch_size * seqlen, device=hidden_states.device)
        max_seqlen_in_batch = seqlen
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False, min_seqlen=None
):
    assert mode in ["full", "random", "third"]
    min_seqlen = min_seqlen if min_seqlen is not None else 0 if zero_lengths else 1
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(min_seqlen, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    else:
        lengths = torch.randint(
            max(min_seqlen, max_seqlen // 3),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )

    if zero_lengths:
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    qv=None,
    kvpacked=False,
    qkvpacked=False,
    query_unused_mask=None,
    key_unused_mask=None,
):
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    d_v = v.shape[-1]
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q, query_padding_mask, query_unused_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if qv is not None else None
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        seqused_q = None
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...") if qv is not None else None

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(
            k, key_padding_mask, key_unused_mask
        )
        v_unpad, *_ = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            qv_unpad.detach() if qv is not None else None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            qv.detach() if qv is not None else None,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(None, None),
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] is None:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        if window_size[1] is None:
            local_mask_left = col_idx > sk
        else:
            local_mask_left = col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk)
        return torch.logical_or(
            local_mask_left,
            torch.logical_and(
                col_idx < row_idx + sk - sq - window_size[0], col_idx >= sink_token_length
            ),
        )


def construct_chunk_mask(
    seqlen_q,
    seqlen_k,
    attention_chunk,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    col_limit_left_chunk = row_idx + sk - sq - (row_idx + sk - sq) % attention_chunk
    return torch.logical_or(
        col_idx < col_limit_left_chunk, col_idx >= col_limit_left_chunk + attention_chunk
    )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(None, None),
    attention_chunk=0,
    sink_token_length=0,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
    return_lse=False,
    gather_kv_indices=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        qv = qv.float() if qv is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
        qv = (qv.float() * q_descale).to(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None:
        scores = scores + torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    local_mask = None
    if window_size[0] is not None or window_size[1] is not None:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
        local_mask = (
            torch.logical_or(local_mask, chunk_mask) if local_mask is not None else chunk_mask
        )
    if gather_kv_indices is not None:
        batch = q.shape[0]
        topk_len = gather_kv_indices.shape[2]
        if topk_len < seqlen_k:
            topk_index_mask = torch.full(
                (batch, seqlen_q, seqlen_k), False, device="cuda"
            ).scatter_(-1, gather_kv_indices, True)
            scores.masked_fill_(rearrange(~topk_index_mask, "b t s -> b 1 t s"), float("-inf"))
    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    # After all masks are applied, before softmax:
    # scores shape: [b, h, t, s]
    lse = torch.logsumexp(scores, dim=-1)  # [b, h, t]
    if learnable_sink is None:
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
    else:
        scores_fp32 = scores.to(torch.float32)
        logits_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
        learnable_sink = rearrange(learnable_sink, "h -> h 1 1")
        logits_or_sinks_max = torch.maximum(learnable_sink, logits_max)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(
            learnable_sink - logits_or_sinks_max
        )
        # LSE with sink: log(Z) = log(normalizer) + max
        lse = (torch.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1)).to(dtype_og)
        attention = (unnormalized_scores / normalizer).to(v.dtype)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    if key_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    if local_mask is not None:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    if return_lse:
        return output.to(dtype_og), attention.to(dtype_og), lse.to(dtype_og)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def maybe_fake_tensor_mode(fake: bool = True):
    """
    One way to populate/pre-compile cache is to use torch fake tensor mode,
    which does not allocate actual GPU tensors but retains tensor shape/dtype
    metadata for cute.compile.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with FakeTensorMode() if fake else nullcontext():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
''',
    'key-files/flash_attn/cute/tile_scheduler.py': r'''# Copyright (c) 2025, Tri Dao.

from enum import IntEnum, auto
from typing import Optional, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.utils import ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerParams

from quack.cute_dsl_utils import ParamsBase

import flash_attn.cute.utils as utils
from flash_attn.cute.fast_math import clz


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `FlashAttentionForwardSm100` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - accept `clc: ClcState | None` in `create(...)` / `__init__`
    - map `clc.initial_work_tile_info()` and `clc.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement.

    Schedulers are responsible for:
    1. Coordinate mapping: linear tile index -> (m_block, head, batch, split)
    2. Work distribution: how to get the next tile (static grid-stride vs CLC dynamic)
    """

    def get_current_work(self) -> WorkTileInfo:
        """Get the current work tile coordinates."""
        ...

    def initial_work_tile_info(self) -> WorkTileInfo:
        """Get the initial work tile for this CTA."""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Consumer-side advance: move to next tile and return it.

        For static schedulers: grid-stride increment + get_current_work.
        For CLC schedulers: consumer wait + get_current_work + consumer release + state advance.
        """
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        """Producer-side prefetch of next work tile (no-op for static schedulers).

        For CLC schedulers: producer acquire + issue CLC query + producer state advance.
        Only called by the scheduler warp.
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """Producer-side cleanup after the last tile.

        No-op for static schedulers. For CLC schedulers: pipeline producer_tail.
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    use_cluster_idx: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        use_cluster_idx: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.use_cluster_idx,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        if const_expr(cute.size(params.cluster_shape_mn) == 1 or not params.use_cluster_idx):
            blk_coord = cute.arch.block_idx()
        else:
            blk_coord = cute.arch.cluster_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        if const_expr(params.use_cluster_idx):
            # Grid must have num_block * cluster_m physical blocks so that there are num_block clusters
            grid_x = params.num_block * params.cluster_shape_mn[0]
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        return (
            grid_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_cluster_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks_cluster: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            num_block_cluster = cute.ceil_div(args.num_block, cute.size(args.cluster_shape_mn))
            total_blocks_cluster = num_block_cluster * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisor(num_block_cluster),
                FastDivmodDivisor(args.num_head),
                total_blocks_cluster,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"StaticPersistentTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "StaticPersistentTileScheduler":
        if const_expr(cute.size(params.cluster_shape_m) == 1):
            tile_idx = cute.arch.block_idx()[0]
        else:
            tile_idx = cute.arch.cluster_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
        grid_x = cutlass.min(max_ctas, params.total_blocks_cluster * params.cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_cluster_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks_cluster
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.cluster_shape_m == 1):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            self._tile_idx += cute.arch.cluster_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        lpt: cutlass.Constexpr[bool] = True
        use_cluster_idx: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileLPTScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # Seems faster if swizzle is a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmodDivisor(args.num_splits),
                is_split_kv=args.is_split_kv,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                lpt=args.lpt,
                use_cluster_idx=args.use_cluster_idx,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        num_batch_splits = (
            params.num_batch * params.num_splits
            if const_expr(params.is_split_kv)
            else params.num_batch
        )
        return (
            cute.round_up(params.num_block, params.cluster_shape_m),
            params.num_head,
            num_batch_splits,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileLPTScheduler._clc_grid_shape(params),
            cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileLPTScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler(
                params, cute.arch.block_idx()[0], Int32(0), clc, loc=loc, ip=ip
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler._clc_grid_shape(params)
        return (params.total_blocks, params.num_splits, Int32(1))

    @cute.jit
    def clc_work_to_coords(self, work) -> WorkTileInfo:
        """Convert CLC response (block, head, batch_split) to WorkTileInfo.

        CLC returns raw grid coordinates — no L2 swizzle (hardware decides order).
        We only apply cluster division, optional LPT block reversal, and split_kv unpacking.
        """
        block_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            block_idx = block_idx // self.params.cluster_shape_m
        if const_expr(self.params.lpt):
            # Longest-processing-time-first: reverse block order
            if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
                num_block = self.params.num_block // self.params.cluster_shape_m
            else:
                num_block = self.params.num_block
            block_idx = num_block - 1 - block_idx
        split_idx = Int32(0)
        if const_expr(self.params.is_split_kv):
            batch_idx, split_idx = divmod(work.tile_idx[2], self.params.num_splits_divmod)
        else:
            batch_idx = work.tile_idx[2]
        if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block_idx = block_idx * self.params.cluster_shape_m + bidx_in_cluster[0]
        return WorkTileInfo(
            (Int32(block_idx), Int32(work.tile_idx[1]), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        # Static path: L2-swizzled coordinate mapping
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self.clc.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 50 * 1024 * 1024
            size_one_qdo_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_dqaccum_head = args.seqlen_k * (args.headdim) * 4
            # size_one_dqaccum_head = 0
            size_one_head = size_one_qdo_head + size_one_dqaccum_head
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 8
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_block=num_block,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileLPTBwdScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        if cutlass.const_expr(params.spt):
            block = params.num_block - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        num_splits: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        is_split_kv: cutlass.Constexpr[bool] = False
        head_swizzle: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            # if backward, this is qdo block size
            kv_block_size = (
                (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
            )
            # if backward, add dqaccum block size to calculate swizzle
            if args.head_swizzle:
                kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
            max_kvblock_in_l2 = size_l2 // kv_block_size
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            # TODO: Support varlen CLC with cluster_shape_m > 1 by refactoring the
            # flattened-tile decode so cluster unpacking semantics are explicit.
            assert scheduling_mode != SchedulingMode.CLC or args.cluster_shape_mn[0] == 1, (
                "Varlen CLC currently requires cluster_shape_mn[0] == 1"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                num_splits=args.num_splits,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                is_split_kv=args.is_split_kv,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self.clc = clc
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileVarlenScheduler.get_grid_shape(params),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileVarlenScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            block_idx = cute.arch.block_idx()
            split_idx = Int32(0)
            if const_expr(params.is_split_kv):
                split_idx = block_idx[1]
            return SingleTileVarlenScheduler(
                params,
                block_idx[0],
                split_idx,
                clc,
                loc=loc,
                ip=ip,
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q
            + params.num_batch * (params.cluster_shape_m * params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        # Round down to nearest multiple of cluster since odd excess is always padding.
        total_blocks_max = total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
        return (total_blocks_max * params.num_head, params.num_splits, Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < params.num_batch:
                seqlen = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(cute.ceil_div(seqlen, params.tile_shape_mn[0]), params.cluster_shape_m)
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _varlen_coord_map(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d", self._tile_idx, group_end_tile, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group)
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx // params.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, batch_idx = %d", self._tile_idx, group_end_tile, num_m_blocks, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            if cutlass.const_expr(params.lpt or params.head_swizzle):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    * params.cluster_shape_m
                    // params.qhead_per_kvhead_packgqa
                    // params.tile_shape_mn[1]
                )
                # nheads_in_l2 = min(max(self.max_kvblock_in_l2 // num_n_blocks, 1), self.num_head)
                # Seems faster to have this be a power of 2
                nheads_in_l2 = (
                    16
                    if num_n_blocks * 16 <= params.max_kvblock_in_l2
                    else (
                        8
                        if num_n_blocks * 8 <= params.max_kvblock_in_l2
                        else (
                            4
                            if num_n_blocks * 4 <= params.max_kvblock_in_l2
                            else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)
                        )
                    )
                )
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = (
                    nheads_in_l2
                    if nheads_in_l2 * (section_idx + 1) <= params.num_head
                    else params.num_head - section_idx * nheads_in_l2
                )
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                if cutlass.const_expr(params.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
            if cutlass.const_expr(params.cluster_shape_m > 1):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * params.cluster_shape_m + bidx_in_cluster[0]
        # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, head_idx=%d, block=%d, is_valid = %d", self._tile_idx, batch_idx, head_idx, block, is_valid)
        split_idx = self._split_idx if const_expr(params.is_split_kv) else Int32(0)
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.get_current_work()
            # Default to grid_dim (one past last valid flat index) so _varlen_coord_map
            # returns is_valid=False when CLC is exhausted. CLC tile_idx is garbage when
            # invalid, so we can't trust it. Local-then-assign avoids CuTe DSL structural
            # mismatch on self inside the runtime if.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._varlen_coord_map()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.initial_work_tile_info()
            # See get_current_work for why grid_dim and local-then-assign.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._varlen_coord_map()

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)
''',
    'key-files/flash_attn/cute/topk_gather_kv.py': r"""from typing import Type, Optional
from dataclasses import dataclass
import operator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass import Int32, Uint32, const_expr, Boolean

from flash_attn.cute import utils
from flash_attn.cute.utils import warp_reduce
from quack.cute_dsl_utils import ParamsBase

import math


@dataclass
class CpasyncGatherKVManager(ParamsBase):
    mIndexTopk: cute.Tensor
    sBitmask: cute.Tensor

    cta_rank_in_cluster: Int32
    thread_idx: Int32
    warp_idx: Int32

    topk_length: Int32
    seqlen_k_limit: Int32
    tile_n: Int32
    num_threads: cutlass.Constexpr[Int32]
    hdim: cutlass.Constexpr[Int32]
    hdim_v: cutlass.Constexpr[Int32]
    num_hdimv_splits: cutlass.Constexpr[Int32]
    cta_group_size: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    topk_indices_per_thread: Int32
    async_copy_elems: Int32

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy

    rTopk: cute.Tensor
    rTopkHalf: cute.Tensor
    # for bitmask
    rTopk_NonInterleaved: cute.Tensor

    pipeline_bitmask: Optional[pipeline.PipelineAsync]
    cpasync_barrier: pipeline.NamedBarrier

    disable_bitmask: cutlass.Constexpr[Boolean]

    @staticmethod
    def create(
        mIndexTopk: cute.Tensor,
        sBitmask: cute.Tensor,
        cta_rank_in_cluster: Int32,
        thread_idx: Int32,
        warp_idx: Int32,
        topk_length: Int32,
        seqlen_k_limit: Int32,
        tile_n: cutlass.Constexpr[Int32],
        hdim: cutlass.Constexpr[Int32],
        hdim_v: cutlass.Constexpr[Int32],
        num_hdimv_splits: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
        cta_group_size: cutlass.Constexpr[Int32],
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        num_stages_bitmask: cutlass.Constexpr[Int32],
        cpasync_barrier: pipeline.NamedBarrier,
        disable_bitmask: cutlass.Constexpr[Boolean],
    ):
        assert tile_n % num_threads == 0
        assert num_threads == 128
        assert hdim % 64 == 0
        assert (hdim_v // num_hdimv_splits // cta_group_size) % 64 == 0
        assert num_threads % cute.arch.WARP_SIZE == 0
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        # assumes hdim is never part of transposed operand
        gmem_k_block_size = math.gcd(
            hdim,
            hdim_v // num_hdimv_splits // cta_group_size,
            128 // dtype_bytes,
        )
        assert gmem_k_block_size % async_copy_elems == 0
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert cute.arch.WARP_SIZE % gmem_threads_per_row == 0
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)
        topk_indices_per_thread = tile_n // num_threads

        rTopk = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopkHalf = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopk_NonInterleaved = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)

        return CpasyncGatherKVManager(
            mIndexTopk,
            sBitmask,
            cta_rank_in_cluster,
            thread_idx,
            warp_idx,
            topk_length,
            seqlen_k_limit,
            tile_n,
            num_threads,
            hdim,
            hdim_v,
            num_hdimv_splits,
            cta_group_size,
            gmem_threads_per_row,
            topk_indices_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            rTopk,
            rTopkHalf,
            rTopk_NonInterleaved,
            pipeline_bitmask,
            cpasync_barrier,
            disable_bitmask,
        )

    @cute.jit
    def load_index_topk(
        self,
        n_block: Int32,
        transpose: bool,
    ):
        entries_per_thread = self.topk_indices_per_thread
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            row = (
                i * self.num_threads
                + (self.thread_idx % self.gmem_threads_per_row)
                * (self.num_threads // self.gmem_threads_per_row)
                + (self.thread_idx // self.gmem_threads_per_row)
            )
            # need this if not offset in load_X
            # if const_expr(not transpose):
            #     row += self.cta_rank_in_cluster * (self.tile_n//self.cta_group_size)
            #     row = row % self.tile_n
            row_idx = n_block * self.tile_n + row
            rTopk[i] = self.mIndexTopk[row_idx]

            if const_expr(not transpose and not self.disable_bitmask):
                row_non_interleaved = i * self.num_threads + self.thread_idx
                row_idx_non_interleaved = n_block * self.tile_n + row_non_interleaved
                self.rTopk_NonInterleaved[0] = self.mIndexTopk[row_idx_non_interleaved]

    @cute.jit
    def compute_bitmask(
        self,
        producer_state_bitmask,
    ):
        lane_idx = cute.arch.lane_idx()
        assert cute.size(self.rTopk_NonInterleaved) == 1
        bitmask = Uint32(0)

        # Step 1. Construct per-thread bitmask
        topk_idx = self.rTopk_NonInterleaved[0]
        is_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        if is_valid:
            bitmask = Uint32(1 << lane_idx)

        # Step 2. Warp shuffle bitwise OR = add since indices are exclusive.
        bitmask = warp_reduce(bitmask, operator.add)

        self.pipeline_bitmask.producer_acquire(producer_state_bitmask)
        # store to smem and sync threads
        if lane_idx == 0:
            self.sBitmask[self.warp_idx, producer_state_bitmask.index] = bitmask
        self.cpasync_barrier.arrive_and_wait()

        self.pipeline_bitmask.producer_commit(producer_state_bitmask)
        producer_state_bitmask.advance()
        return producer_state_bitmask

    @cute.jit
    def compute_X_ptr(
        self,
        mX: cute.Tensor,
        transpose: bool,
    ):
        entries_per_thread = self.topk_indices_per_thread
        tPrXPtr = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int64)
        tPrRowValid = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int32)
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            topk_idx = rTopk[i]
            if const_expr(not self.disable_bitmask):
                row_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
                tPrRowValid[i] = row_valid
            if const_expr(not transpose):
                tPrXPtr[i] = utils.elem_pointer(mX, (topk_idx, 0)).toint()
            else:
                tPrXPtr[i] = utils.elem_pointer(mX, (0, topk_idx)).toint()

        return tPrXPtr, tPrRowValid

    @cute.jit
    def load_X(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
    ):
        assert K_or_V in ("K", "V")
        cta_tile_n = self.tile_n if const_expr(transpose) else self.tile_n // self.cta_group_size
        head_dim = self.hdim if const_expr(K_or_V == "K") else self.hdim_v // self.num_hdimv_splits
        if const_expr(transpose):
            head_dim = head_dim // self.cta_group_size
        order = (1, 0) if const_expr(transpose) else (0, 1)

        sX_nd_layout = cute.make_ordered_layout((cta_tile_n, head_dim), order=order)
        sX_nd = cute.composition(sX, sX_nd_layout)

        cX = cute.make_identity_tensor((cta_tile_n, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)

        tPrXPtr, tPrRowValid = self.compute_X_ptr(mX, transpose)

        if const_expr(not transpose):
            offset = self.cta_rank_in_cluster * (self.gmem_threads_per_row // self.cta_group_size)
        else:
            offset = 0

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            if const_expr(not self.disable_bitmask):
                row_valid = utils.shuffle_sync(
                    tPrRowValid[m // self.gmem_threads_per_row],
                    (m + offset) % self.gmem_threads_per_row,
                    width=self.gmem_threads_per_row,
                )
                should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], Boolean)
                should_load.fill(Boolean(row_valid))
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                (m + offset) % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (self.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // self.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    self.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load if const_expr(not self.disable_bitmask) else None,
                )
""",
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (a KernelWiki artifact: upstream files captured from a pull request; reference material for `forge lib adapt`, not a submission): adapt it first (`forge lib adapt`)")
