"""Destination-passing FP16 GEMM using Blackwell tensor cores."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


# The explicit pipeline is adapted from RSI-Kernel-Tune’s neighboring 010 GEMM.
# It follows the producer/MMA/epilogue partitioning in the Triton Gluon tutorial.
# https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py
# Swapped operands for small M follow QianQu and the CompoundingKernel Explorer.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/008_gemm_n4096_k14336
# Tile residency and staging draw on RSI-Kernel-Tune's H100 GEMM.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py


@gluon.jit
def _read_tiles(a, b, a_ring, b_ring, load_free, load_ready,
                TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                STAGES: gl.constexpr, WORKERS):
    tiles_n: gl.constexpr = 4096 // TN
    total = gl.cdiv(a.shape[0], TM) * tiles_n
    step = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row = tile // tiles_n * TM
        col = tile % tiles_n * TN
        for k in range(14336 // TK):
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
                    STAGES: gl.constexpr, WORKERS, SWAP: gl.constexpr,
                    ACC_STAGES: gl.constexpr):
    total = gl.cdiv(m, TM) * (4096 // TN)
    step = 0
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_free.index(acc_index), acc_phase ^ 1)
        out = acc.index(acc_index)
        for k in range(14336 // TK):
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


@gluon.jit
def _write_tiles(c, acc, acc_free, acc_ready, TM: gl.constexpr, TN: gl.constexpr,
                 WORKERS, SWAP: gl.constexpr, ACC_STAGES: gl.constexpr):
    tiles_n: gl.constexpr = 4096 // TN
    total = gl.cdiv(c.shape[0], TM) * tiles_n
    temp = gl.allocate_shared_memory(gl.float16, [TM, TN], c.layout)
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row = tile // tiles_n * TM
        col = tile % tiles_n * TN
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_ready.index(acc_index), acc_phase)
        values = acc.index(acc_index).load().to(gl.float16)
        tma.store_wait(pendings=0)
        if SWAP:
            temp.store(values.permute((1, 0)))
        else:
            temp.store(values)
        mbarrier.arrive(acc_free.index(acc_index), count=1)
        fence_async_shared()
        tma.async_copy_shared_to_global(c, [row, col], temp)
        wave += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _matmul(a, b, c, TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
            STAGES: gl.constexpr, SWAP: gl.constexpr,
            ACC_STAGES: gl.constexpr):
    workers = gl.num_programs(0)
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
        (_write_tiles, (c, acc, acc_free, acc_ready, TM, TN, workers, SWAP, ACC_STAGES)),
        (_read_tiles, (a, b, a_ring, b_ring, load_free, load_ready, TM, TN, TK, STAGES, workers)),
        (_multiply_tiles, (a.shape[0], a_ring, b_ring, load_free, load_ready,
                           acc, acc_free, acc_ready, TM, TN, TK, STAGES, workers, SWAP, ACC_STAGES)),
    ], [1, 1], [24, 32])


def _desc(tensor, rows, cols):
    shape = [rows, cols]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float16)
    return TensorDescriptor.from_tensor(tensor, shape, layout)


def _launch(A, B, C, tm, tn, tk, stages, swap, acc_stages=2, waves=1):
    properties = torch.cuda.get_device_properties(A.device)
    tiles = triton.cdiv(A.shape[0], tm) * (4096 // tn)
    workers = min(tiles, properties.multi_processor_count * waves)
    _matmul[(workers,)](_desc(A, tm, tk), _desc(B, tn, tk), _desc(C, tm, tn),
                       tm, tn, tk, stages, swap, acc_stages, num_warps=4)


@triton.jit
def _gemv(A, B, C, TN: tl.constexpr, TK: tl.constexpr):
    n = tl.program_id(0) * TN + tl.arange(0, TN)
    k = tl.arange(0, TK)
    vector = tl.full((TN, TK), 0, tl.float32)
    for step in range(tl.cdiv(14336, TK)):
        kk = k + step * TK
        a = tl.load(A + kk, kk < 14336, 0).to(tl.float32)
        b = tl.load(B + n[:, None] * 14336 + kk[None, :], kk[None, :] < 14336, 0)
        vector = tl.fma(b.to(tl.float32), a[None, :], vector)
    tl.store(C + n, tl.sum(vector, axis=1))


@triton.jit
def _split_gemm(A, B, P, M, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                SPLITS: tl.constexpr):
    row = tl.program_id(0) * TM + tl.arange(0, TM)
    col = tl.program_id(1) * TN + tl.arange(0, TN)
    split = tl.program_id(2)
    k = split * (14336 // SPLITS) + tl.arange(0, TK)
    a_ptr = A + row[:, None] * 14336 + k[None, :]
    b_ptr = B + col[:, None] * 14336 + k[None, :]
    acc = tl.full((TM, TN), 0, tl.float32)
    for step in range(14336 // (TK * SPLITS)):
        a = tl.load(a_ptr + step * TK, row[:, None] < M, 0)
        b = tl.load(b_ptr + step * TK)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(P + (split * M + row[:, None]) * 4096 + col[None, :], acc, row[:, None] < M)


@triton.jit
def _sum_partials(P, C, M, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    partials = tl.load(P + tl.arange(0, SPLITS)[:, None] * (M * 4096) + offsets[None, :])
    tl.store(C + offsets, tl.sum(partials, axis=0))


def _launch_split(A, B, C, tm, tn, tk, stages, splits):
    m = A.shape[0]
    partials = torch.empty((splits, m, 4096), device=A.device, dtype=torch.float32)
    _split_gemm[(triton.cdiv(m, tm), 4096 // tn, splits)](
        A, B, partials, m, tm, tn, tk, splits, num_warps=4, num_stages=stages,
    )
    _sum_partials[(m * 16,)](partials, C, m, splits, 256)


@triton.jit
def _strided_gemm(A, B, C, M, AM, AK, BN, BK, CM, CN):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    a_ptr = A + row[:, None] * AM + k[None, :] * AK
    b_ptr = B + col[:, None] * BN + k[None, :] * BK
    acc = tl.full((32, 64), 0, tl.float32)
    for step in range(224):
        a = tl.load(a_ptr + step * 64 * AK, row[:, None] < M, 0)
        b = tl.load(b_ptr + step * 64 * BK)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(C + row[:, None] * CM + col[None, :] * CN, acc, row[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    if any(not x.is_contiguous() or x.data_ptr() % 16 for x in (A, B, C)):
        _strided_gemm[(triton.cdiv(m, 32), 64)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(), num_warps=4, num_stages=3,
        )
    elif m == 1:
        _gemv[(1024,)](A, B, C, 4, 2048, num_warps=8)
    elif m <= 16:
        _launch_split(A, B, C, 16, 32, 128, 4, 4)
    elif m <= 64:
        _launch(A, B, C, triton.next_power_of_2(m), 64, 256, 3, True, 1)
    elif m <= 128:
        _launch(A, B, C, 64, 64, 256, 3, False, 1)
    elif m <= 256:
        _launch(A, B, C, 64, 128, 128, 4, False, 1)
    else:
        _launch(A, B, C, 128, 256, 64, 3, False)
