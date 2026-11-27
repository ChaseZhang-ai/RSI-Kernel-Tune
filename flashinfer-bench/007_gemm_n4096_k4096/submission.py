"""Destination-passing FP16 GEMM with FP32 accumulation on Blackwell."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


# Pipeline adapted from the repository submission for FlashInfer 010.
# The producer/MMA/epilogue partitioning follows the Triton Gluon tutorial.
# https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py
# Swapped operands for small M follow the CompoundingKernel Explorer archive
# (search-agent implementation by Claude, published by QianQu).
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/007_gemm_n4096_k4096
# Tile residency and staging draw on RSI-Kernel-Tune's H100 GEMM.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py


@gluon.jit
def _tile_origin(tile, m, TM: gl.constexpr, TN: gl.constexpr, GROUP: gl.constexpr):
    tiles_n: gl.constexpr = 4096 // TN
    if GROUP == 1:
        return tile // tiles_n * TM, tile % tiles_n * TN
    first_m = tile // (GROUP * tiles_n) * GROUP
    rows_in_group = gl.minimum(GROUP, gl.cdiv(m, TM) - first_m)
    local_tile = tile % (GROUP * tiles_n)
    return (first_m + local_tile % rows_in_group) * TM, local_tile // rows_in_group * TN


@gluon.jit
def _read_tiles(a, b, a_ring, b_ring, load_free, load_ready,
                TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                STAGES: gl.constexpr, WORKERS: gl.constexpr, GROUP: gl.constexpr):
    tiles_n: gl.constexpr = 4096 // TN
    total = gl.cdiv(a.shape[0], TM) * tiles_n
    step = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row, col = _tile_origin(tile, a.shape[0], TM, TN, GROUP)
        for k in range(4096 // TK):
            index = step % STAGES
            phase = step // STAGES & 1
            mbarrier.wait(load_free.index(index), phase ^ 1)
            ready = load_ready.index(index)
            mbarrier.expect(ready, (TM + TN) * TK * 2)
            tma.async_copy_global_to_shared(a, [row, k * TK], ready, a_ring.index(index))
            tma.async_copy_global_to_shared(b, [col, k * TK], ready, b_ring.index(index))
            step += 1


@gluon.jit
def _multiply_tiles(m, a_ring, b_ring, load_free, load_ready, acc, acc_free, acc_ready,
                    TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                    STAGES: gl.constexpr, WORKERS: gl.constexpr, SWAP: gl.constexpr,
                    ACC_STAGES: gl.constexpr):
    total = gl.cdiv(m, TM) * (4096 // TN)
    step = 0
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_free.index(acc_index), acc_phase ^ 1)
        out = acc.index(acc_index)
        for k in range(4096 // TK):
            index = step % STAGES
            phase = step // STAGES & 1
            mbarrier.wait(load_ready.index(index), phase)
            operand_a = a_ring.index(index)
            operand_b = b_ring.index(index)
            if SWAP:
                tcgen05_mma(operand_b, operand_a.permute((1, 0)), out, use_acc=k != 0)
            else:
                tcgen05_mma(operand_a, operand_b.permute((1, 0)), out, use_acc=k != 0)
            tcgen05_commit(load_free.index(index))
            step += 1
        tcgen05_commit(acc_ready.index(acc_index))
        wave += 1


# Triton 3.6 requires a register layout. Select the API at module load so
# the JIT only resolves symbols provided by the installed compiler.
if hasattr(blackwell, "get_tmem_reg_layout"):
    _get_tmem_reg_layout = blackwell.get_tmem_reg_layout

    @gluon.jit
    def _load_output(tile):
        layout: gl.constexpr = _get_tmem_reg_layout(tile.dtype, tile.shape, tile.layout, gl.num_warps())
        return tile.load(layout).to(gl.float16)

else:
    @gluon.jit
    def _load_output(tile):
        return tile.load().to(gl.float16)


# Direct stores retain the weight-first TMEM layout. Segmenting the large
# epilogue leaves room for four K stages and two accumulator slots.
@gluon.jit
def _write_tiles(c, out, m, acc, acc_free, acc_ready, TM: gl.constexpr, TN: gl.constexpr,
                 WORKERS: gl.constexpr, SWAP: gl.constexpr, ACC_STAGES: gl.constexpr,
                 GROUP: gl.constexpr, EPI: gl.constexpr):
    total = gl.cdiv(m, TM) * (4096 // TN)
    if EPI != 0:
        temp = gl.allocate_shared_memory(gl.float16, [TM, EPI], c.layout)
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row, col = _tile_origin(tile, m, TM, TN, GROUP)
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_ready.index(acc_index), acc_phase)
        slot = acc.index(acc_index)
        if EPI == 0:
            values = _load_output(slot)
            if SWAP:
                values = values.permute((1, 0))
            mbarrier.arrive(acc_free.index(acc_index), count=1)
            layout: gl.constexpr = values.type.layout
            rows = row + gl.arange(0, TM, layout=gl.SliceLayout(1, layout))
            cols = col + gl.arange(0, TN, layout=gl.SliceLayout(0, layout))
            gl.store(out + rows[:, None] * 4096 + cols[None, :], values, rows[:, None] < m)
        else:
            for segment in gl.static_range(TN // EPI):
                values = _load_output(slot.slice(segment * EPI, EPI))
                tma.store_wait(pendings=0)
                temp.store(values)
                if segment == TN // EPI - 1:
                    mbarrier.arrive(acc_free.index(acc_index), count=1)
                fence_async_shared()
                tma.async_copy_shared_to_global(c, [row, col + segment * EPI], temp)
        wave += 1
    if EPI != 0:
        tma.store_wait(pendings=0)


@gluon.jit
def _matmul(a, b, c, out, m, TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
            STAGES: gl.constexpr, WORKERS: gl.constexpr, SWAP: gl.constexpr,
            ACC_STAGES: gl.constexpr, GROUP: gl.constexpr, EPI: gl.constexpr):
    a_ring = gl.allocate_shared_memory(gl.float16, [STAGES, TM, TK], a.layout)
    b_ring = gl.allocate_shared_memory(gl.float16, [STAGES, TN, TK], b.layout)
    load_free = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    load_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    acc_free = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(STAGES):
        mbarrier.init(load_free.index(i), count=1)
        mbarrier.init(load_ready.index(i), count=1)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(acc_free.index(i), count=1)
        mbarrier.init(acc_ready.index(i), count=1)
    mm: gl.constexpr = TN if SWAP else TM
    nn: gl.constexpr = TM if SWAP else TN
    layout: gl.constexpr = TensorMemoryLayout([mm, nn], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [ACC_STAGES, mm, nn], layout)
    gl.warp_specialize([
        (_write_tiles, (c, out, m, acc, acc_free, acc_ready, TM, TN, WORKERS, SWAP, ACC_STAGES, GROUP, EPI)),
        (_read_tiles, (a, b, a_ring, b_ring, load_free, load_ready, TM, TN, TK, STAGES, WORKERS, GROUP)),
        (_multiply_tiles, (a.shape[0], a_ring, b_ring, load_free, load_ready,
                           acc, acc_free, acc_ready, TM, TN, TK, STAGES, WORKERS, SWAP, ACC_STAGES)),
    ], [1, 1], [24, 32])


def _desc(tensor, rows, cols):
    shape = [rows, cols]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float16)
    return TensorDescriptor.from_tensor(tensor, shape, layout)


def _launch(A, B, C, tm, tn, tk, stages, swap, acc_stages, group, epi):
    sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
    m = A.shape[0]
    workers = min(triton.cdiv(m, tm) * (4096 // tn), sm_count)
    descriptor = _desc(C, tm, epi) if epi else None
    _matmul[(workers,)](
        _desc(A, tm, tk), _desc(B, tn, tk), descriptor, C, m,
        tm, tn, tk, stages, workers, swap, acc_stages, group, epi, num_warps=4,
    )


@triton.jit
def _gemv(A, B, C):
    n = tl.program_id(0) * 4 + tl.arange(0, 4)
    k = tl.arange(0, 4096)
    a = tl.load(A + k).to(tl.float32)
    b = tl.load(B + n[:, None] * 4096 + k[None, :]).to(tl.float32)
    tl.store(C + n, tl.sum(a[None, :] * b, axis=1))


# Four interleaved K slices improve weight-read parallelism within a CTA.
# Reduce their FP32 partials before the sole FP16 output store.
@triton.jit
def _thin_gemm(A, B, C, M):
    TM: tl.constexpr = 16
    TN: tl.constexpr = 16
    TK: tl.constexpr = 128
    PARTS: tl.constexpr = 4
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    cols = tl.program_id(1) * TN + tl.arange(0, TN)
    part = tl.arange(0, PARTS)
    k = part[:, None] * TK + tl.arange(0, TK)[None, :]
    a = A + rows[None, :, None] * 4096 + k[:, None, :]
    b = B + cols[None, :, None] * 4096 + k[:, None, :]
    acc = tl.full((PARTS, TN, TM), 0, tl.float32)
    for step in range(4096 // (TK * PARTS)):
        av = tl.load(a + step * TK * PARTS, rows[None, :, None] < M, 0)
        bv = tl.load(b + step * TK * PARTS)
        acc = tl.dot(bv, tl.permute(av, (0, 2, 1)), acc)
    tl.store(C + rows[:, None] * 4096 + cols[None, :], tl.trans(tl.sum(acc, axis=0)),
             rows[:, None] < M)


@triton.jit
def _strided_gemm(A, B, C, M, AM, AK, BN, BK, CM, CN):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    a = A + rows[:, None] * AM + k[None, :] * AK
    b = B + cols[:, None] * BN + k[None, :] * BK
    acc = tl.full((32, 64), 0, tl.float32)
    for step in range(4096 // 64):
        av = tl.load(a + step * 64 * AK, rows[:, None] < M, 0)
        bv = tl.load(b + step * 64 * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + rows[:, None] * CM + cols[None, :] * CN, acc, rows[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16 for tensor in (A, B, C)):
        _strided_gemm[(triton.cdiv(m, 32), 4096 // 64)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(),
            num_warps=4, num_stages=3,
        )
        return
    if m == 1:
        _gemv[(4096 // 4,)](A, B, C, num_warps=4)
        return

    if m <= 16:
        _thin_gemm[(1, 4096 // 16)](A, B, C, m, num_warps=4, num_stages=3)
        return

    if m <= 128:
        tm = min(64, triton.next_power_of_2(m))
        _launch(A, B, C, tm, 64, 256, 3, True, acc_stages=1, group=1, epi=0)
    elif m <= 256:
        _launch(A, B, C, 64, 128, 128, 4, True, acc_stages=1, group=1, epi=0)
    else:
        _launch(A, B, C, 128, 256, 64, 4, False, acc_stages=2, group=4 if m > 1024 else 1, epi=32)
