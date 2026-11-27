"""Destination-passing FP16 GEMM for N=5120 and K=2048."""

import inspect

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as bw
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDescriptor
from triton.tools.tensor_descriptor import TensorDescriptor


N = 5120
K = 2048


@triton.jit
def _gemv(A, B, C, M: tl.constexpr):
    n = tl.program_id(0) * 4 + tl.arange(0, 4)
    k = tl.arange(0, 2048)
    b = tl.load(B + n[:, None] * 2048 + k[None, :]).to(tl.float32)
    for m in tl.static_range(M):
        a = tl.load(A + m * 2048 + k).to(tl.float32)
        acc = tl.sum(b * a[None, :], axis=1)
        tl.store(C + m * 5120 + n, acc)


@triton.jit
def _pointer_gemm(A, B, C, M, AM, AK, BN, BK, CM, CN,
                  TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    m = tl.program_id(1) * TM + tl.arange(0, TM)
    n = tl.program_id(0) * TN + tl.arange(0, TN)
    k = tl.arange(0, TK)
    a = A + m[:, None] * AM + k[None, :] * AK
    b = B + n[:, None] * BN + k[None, :] * BK
    acc = tl.zeros((TM, TN), tl.float32)
    for step in range(2048 // TK):
        av = tl.load(a + step * TK * AK, m[:, None] < M, 0)
        bv = tl.load(b + step * TK * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + m[:, None] * CM + n[None, :] * CN, acc, m[:, None] < M)


@triton.jit
def _tile_coords(tile, m_tiles, n_tiles: tl.constexpr, GROUP: tl.constexpr):
    band = tile // (GROUP * n_tiles) * GROUP
    height = tl.minimum(m_tiles - band, GROUP)
    within = tile % (GROUP * n_tiles)
    return band + within % height, within // height


# The small-M weight-first mapping follows the CompoundingKernel Explorer
# implementation by QianQu and collaborators.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/009_gemm_n5120_k2048
@triton.jit
def _accumulate(A, B, row, col, TM: tl.constexpr, TN: tl.constexpr,
                TK: tl.constexpr, SWAP: tl.constexpr):
    if SWAP:
        acc = tl.zeros((TN, TM), tl.float32)
    else:
        acc = tl.zeros((TM, TN), tl.float32)
    for k in range(2048 // TK):
        a = A.load([row, k * TK])
        b = B.load([col, k * TK])
        if SWAP:
            acc = tl.dot(b, tl.trans(a), acc)
        else:
            acc = tl.dot(a, tl.trans(b), acc)
    if SWAP:
        acc = tl.trans(acc)
    return acc


@triton.jit
def _raster_gemm(A, B, C, M, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                 SWAP: tl.constexpr):
    mi, ni = _tile_coords(tl.program_id(0), tl.cdiv(M, TM), 5120 // TN, 4)
    row = mi * TM
    col = ni * TN
    acc = _accumulate(A, B, row, col, TM, TN, TK, SWAP)
    m = row + tl.arange(0, TM)
    n = col + tl.arange(0, TN)
    tl.store(C + m[:, None] * 5120 + n[None, :], acc, m[:, None] < M)


def _descriptor(tensor, rows, columns):
    return TensorDescriptor(tensor, tensor.shape, tensor.stride(), [rows, columns])


def _launch_pointer(A, B, C, tm, tn, tk, warps=4, stages=3):
    m = A.shape[0]
    _pointer_gemm[(N // tn, triton.cdiv(m, tm))](
        A, B, C, m, *A.stride(), *B.stride(), *C.stride(), tm, tn, tk,
        num_warps=warps, num_stages=stages,
    )


def _raster_config(m):
    if m <= 16:
        return 16, 32, 256, 3, True, 2
    if m <= 32:
        return 32, 32, 256, 3, True, 4
    if m <= 64:
        return 64, 64, 128, 4, True, 4
    if m <= 128:
        return 128, 64, 128, 4, True, 4
    return 64, 64, 128, 3, True, 4


# Explicit TMA/MMA warp partitions adapted from RSI's FlashInfer 011 and
# Triton's Gluon warp-specialization tutorial. The persistent schedule uses
# row-major FP16 A[M,K], B[N,K], and a grouped M raster.
# https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py
_LIST_WARP_SPECIALIZE = gl.constexpr("functions_and_args" in inspect.signature(gl.warp_specialize).parameters)
_COL_STRIDE_LAYOUT = gl.constexpr("col_stride" in inspect.signature(bw.TensorMemoryLayout).parameters)


@gluon.jit
def _persistent_coords(tile, m_tiles, n_tiles: gl.constexpr, group):
    group_start = tile // (group * n_tiles) * group
    within_group = tile % (group * n_tiles)
    rows = gl.minimum(m_tiles - group_start, group)
    return group_start + within_group % rows, within_group // rows


@gluon.jit
def _load_tiles(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    STAGES: gl.constexpr = sa.shape[0]
    n_tiles: gl.constexpr = 5120 // BN
    # Load and MMA warps advance the staging ring across output tiles.
    sequence = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        m, n = _persistent_coords(tile, m_tiles, n_tiles, group)
        for step in range(2048 // 64):
            stage = sequence % STAGES
            wrap = (sequence // STAGES) & 1
            mbarrier.wait(freed.index(stage), wrap ^ 1)
            barrier = ready.index(stage)
            mbarrier.expect(barrier, (128 + BN) * 64 * 2)
            tma.async_copy_global_to_shared(A, [m * 128, step * 64], barrier, sa.index(stage))
            tma.async_copy_global_to_shared(B, [n * BN, step * 64], barrier, sb.index(stage))
            sequence += 1


@gluon.jit
def _accumulate_tiles(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    STAGES: gl.constexpr = sa.shape[0]
    n_tiles: gl.constexpr = 5120 // BN
    sequence = 0
    ordinal = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        slot = ordinal & 1
        phase = (ordinal // 2) & 1
        mbarrier.wait(released.index(slot), phase ^ 1)
        dst = acc.index(slot)
        for step in range(2048 // 64):
            stage = sequence % STAGES
            wrap = (sequence // STAGES) & 1
            mbarrier.wait(ready.index(stage), wrap)
            bw.tcgen05_mma(sa.index(stage), sb.index(stage).permute((1, 0)), dst, use_acc=step != 0)
            bw.tcgen05_commit(freed.index(stage))
            sequence += 1
        bw.tcgen05_commit(done.index(slot))
        ordinal += 1


@gluon.jit
def _store_tiles(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    PART: gl.constexpr = C.block_type.shape[1]
    n_tiles: gl.constexpr = 5120 // BN
    smem = gl.allocate_shared_memory(gl.float16, [128, PART], C.layout)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    ordinal = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        m, n = _persistent_coords(tile, m_tiles, n_tiles, group)
        slot = ordinal & 1
        mbarrier.wait(done.index(slot), (ordinal // 2) & 1)
        src = acc.index(slot)
        for sub in gl.static_range(BN // PART):
            fp16 = src.slice(sub * PART, PART).load(layout).to(gl.float16)
            tma.store_wait(pendings=0)
            smem.store(fp16)
            fence_async_shared()
            tma.async_copy_shared_to_global(C, [m * 128, n * BN + sub * PART], smem)
        mbarrier.arrive(released.index(slot), count=1)
        ordinal += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _init_barriers(COUNT: gl.constexpr):
    bars = gl.allocate_shared_memory(gl.int64, [COUNT, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(COUNT):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def _persistent_tcgen(A, B, C, GROUP: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    sa = gl.allocate_shared_memory(gl.float16, [STAGES, 128, 64], A.layout)
    sb = gl.allocate_shared_memory(gl.float16, [STAGES, BN, 64], B.layout)
    if _COL_STRIDE_LAYOUT:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BN], col_stride=1)
    else:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BN], unpacked=True)
    acc = bw.allocate_tensor_memory(gl.float32, [2, 128, BN], layout)
    # Two TMEM slots let the MMA warp advance while the epilogue drains.
    args = (A, B, C, sa, sb, _init_barriers(STAGES), _init_barriers(STAGES), acc, _init_barriers(2), _init_barriers(2))
    config = (args, gl.to_tensor(GROUP))
    if _LIST_WARP_SPECIALIZE:
        gl.warp_specialize([(_store_tiles, config), (_load_tiles, config), (_accumulate_tiles, config)], [1, 1], [24, 24])
    else:
        gl.warp_specialize(default_args=config, default_partition=_store_tiles, worker_args=config,
                           worker_partitions=[_load_tiles, _accumulate_tiles], worker_num_warps=[1, 1], worker_num_regs=[24, 24])


def _launch_tcgen(A, B, C, bn, stages, part):
    def descriptor(tensor, tile):
        layout = gl.NVMMASharedLayout.get_default_for(tile, gl.float16)
        return GluonDescriptor.from_tensor(tensor, tile, layout)

    a = descriptor(A, [128, 64])
    b = descriptor(B, [bn, 64])
    c = descriptor(C, [128, part])
    sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = min(sms, triton.cdiv(A.shape[0], 128) * (5120 // bn))
    return _persistent_tcgen[(grid,)](a, b, c, 4, bn, stages, num_warps=4)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (A, B, C)):
        _launch_pointer(A, B, C, 32, 128, 64)
    elif m <= 2:
        _gemv[(N // 4,)](A, B, C, m, num_warps=8 if m == 1 else 4)
    elif m <= 8:
        _launch_pointer(A, B, C, 16, 16, 128, warps=2, stages=4)
    elif m <= 192:
        tm, tn, tk, stages, swap, warps = _raster_config(m)
        _raster_gemm[(triton.cdiv(m, tm) * (N // tn),)](
            _descriptor(A, tm, tk), _descriptor(B, tn, tk), C,
            m, tm, tn, tk, swap, num_warps=warps, num_stages=stages,
        )
    elif m <= 384 or 768 < m <= 1024:
        _launch_tcgen(A, B, C, bn=128, stages=5, part=64)
    elif m <= 768:
        _launch_tcgen(A, B, C, bn=256, stages=4, part=128)
    else:
        _launch_tcgen(A, B, C, bn=256, stages=4, part=32)
