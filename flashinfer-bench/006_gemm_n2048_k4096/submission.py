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


# The producer, MMA consumer, and store warps adapt this repository's FlashInfer
# 010 GEMM. The grouped schedule and sliced epilogue draw on FlashInfer 011.
# Tile residency and staged barriers follow the repository's H100 GEMM.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py
# The small-M split-K experiments draw on QianQu's CompoundingKernel Explorer.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/006_gemm_n2048_k4096


@gluon.jit
def _coordinates(tile, tiles_m, TILES_N: gl.constexpr, GROUP: gl.constexpr):
    if GROUP == 1:
        return tile // TILES_N, tile % TILES_N
    group_begin = tile // (GROUP * TILES_N) * GROUP
    group_size = gl.minimum(tiles_m - group_begin, GROUP)
    within = tile % (GROUP * TILES_N)
    return group_begin + within % group_size, within // group_size


@gluon.jit
def _read_tiles(a, b, a_ring, b_ring, load_free, load_ready,
                TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                STAGES: gl.constexpr, GROUP: gl.constexpr):
    tiles_n: gl.constexpr = 2048 // TN
    tiles_m = gl.cdiv(a.shape[0], TM)
    total = tiles_m * tiles_n
    step = 0
    for tile in range(gl.program_id(0), total, gl.num_programs(0)):
        row, col = _coordinates(tile, tiles_m, tiles_n, GROUP)
        row *= TM
        col *= TN
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
                    STAGES: gl.constexpr, GROUP: gl.constexpr, ACC_STAGES: gl.constexpr):
    tiles_n: gl.constexpr = 2048 // TN
    tiles_m = gl.cdiv(m, TM)
    total = tiles_m * tiles_n
    step = 0
    wave = 0
    for tile in range(gl.program_id(0), total, gl.num_programs(0)):
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
            tcgen05_mma(operand_a, operand_b.permute((1, 0)), out, use_acc=k != 0)
            tcgen05_commit(load_free.index(index))
            step += 1
        tcgen05_commit(acc_ready.index(acc_index))
        wave += 1


@gluon.jit
def _write_tiles(c, acc, acc_free, acc_ready, TM: gl.constexpr, TN: gl.constexpr,
                 GROUP: gl.constexpr, ACC_STAGES: gl.constexpr,
                 EPILOGUE: gl.constexpr):
    tiles_n: gl.constexpr = 2048 // TN
    tiles_m = gl.cdiv(c.shape[0], TM)
    total = tiles_m * tiles_n
    temp = gl.allocate_shared_memory(gl.float16, [TM, EPILOGUE], c.layout)
    wave = 0
    for tile in range(gl.program_id(0), total, gl.num_programs(0)):
        row, col = _coordinates(tile, tiles_m, tiles_n, GROUP)
        row *= TM
        col *= TN
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_ready.index(acc_index), acc_phase)
        output = acc.index(acc_index)
        if EPILOGUE == TN:
            values = output.load().to(gl.float16)
            tma.store_wait(pendings=0)
            temp.store(values)
            mbarrier.arrive(acc_free.index(acc_index), count=1)
            fence_async_shared()
            tma.async_copy_shared_to_global(c, [row, col], temp)
        else:
            for part in gl.static_range(TN // EPILOGUE):
                values = output.slice(part * EPILOGUE, EPILOGUE).load().to(gl.float16)
                tma.store_wait(pendings=0)
                temp.store(values)
                fence_async_shared()
                tma.async_copy_shared_to_global(c, [row, col + part * EPILOGUE], temp)
            mbarrier.arrive(acc_free.index(acc_index), count=1)
        wave += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _matmul(a, b, c, TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
            STAGES: gl.constexpr, GROUP: gl.constexpr, ACC_STAGES: gl.constexpr, EPILOGUE: gl.constexpr):
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
    layout: gl.constexpr = TensorMemoryLayout([TM, TN], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [ACC_STAGES, TM, TN], layout)
    gl.warp_specialize([
        (_write_tiles, (c, acc, acc_free, acc_ready, TM, TN, GROUP, ACC_STAGES, EPILOGUE)),
        (_read_tiles, (a, b, a_ring, b_ring, load_free, load_ready, TM, TN, TK, STAGES, GROUP)),
        (_multiply_tiles, (a.shape[0], a_ring, b_ring, load_free, load_ready,
                           acc, acc_free, acc_ready, TM, TN, TK, STAGES, GROUP, ACC_STAGES)),
    ], [1, 1], [24, 32])


def _desc(tensor, rows, cols):
    shape = [rows, cols]
    layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float16)
    return TensorDescriptor.from_tensor(tensor, shape, layout)


def _launch_tma(A, B, C, tm, tn, tk, stages, acc_stages=2, persistent=True, group=1, part=None):
    tiles = triton.cdiv(A.shape[0], tm) * (2048 // tn)
    workers = tiles
    if persistent:
        workers = min(tiles, torch.cuda.get_device_properties(A.device).multi_processor_count)
    part = tn if part is None else part
    _matmul[(workers,)](_desc(A, tm, tk), _desc(B, tn, tk), _desc(C, tm, part),
                       tm, tn, tk, stages, group, acc_stages, part, num_warps=4)


@triton.jit
def _gemv(A, B, C, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 4096)
    a = tl.load(A + k).to(tl.float32)
    b = tl.load(B + n[:, None] * 4096 + k[None, :]).to(tl.float32)
    tl.store(C + n, tl.sum(a[None, :] * b, axis=1))


@triton.jit
def _vector_gemm(A, B, C, M, BN: tl.constexpr, ROWS: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 4096)
    b = tl.load(B + n[:, None] * 4096 + k[None, :]).to(tl.float32)
    for row in tl.static_range(ROWS):
        a = tl.load(A + row * 4096 + k, row < M, 0).to(tl.float32)
        tl.store(C + row * 2048 + n, tl.sum(a[None, :] * b, axis=1), row < M)


@triton.jit
def _gemm(A, B, C, M, AM, AK, BN, BK, CM, CN,
          TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SPLIT: tl.constexpr):
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    cols = tl.program_id(1) * TN + tl.arange(0, TN)
    part = tl.program_id(2)
    kk = part * (4096 // SPLIT) + tl.arange(0, TK)
    a = A + rows[:, None] * AM + kk[None, :] * AK
    b = B + cols[:, None] * BN + kk[None, :] * BK
    acc = tl.full((TM, TN), 0, tl.float32)
    for step in range(4096 // (TK * SPLIT)):
        av = tl.load(a + step * TK * AK, rows[:, None] < M, 0)
        bv = tl.load(b + step * TK * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + part * M * CM + rows[:, None] * CM + cols[None, :] * CN,
             acc, rows[:, None] < M)


@triton.jit
def _merge(P, C, M, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLIT)
    v = tl.load(P + parts[:, None] * M * 2048 + index[None, :], index[None, :] < M * 2048, 0)
    tl.store(C + index, tl.sum(v, 0), index < M * 2048)


def _launch_gemm(A, B, C, tm, tn, tk, split, warps=4, stages=3):
    m = A.shape[0]
    dest = C if split == 1 else torch.empty((split, m, 2048), device=A.device, dtype=torch.float32)
    _gemm[(triton.cdiv(m, tm), 2048 // tn, split)](
        A, B, dest, m, *A.stride(), *B.stride(), *dest.stride()[-2:],
        tm, tn, tk, split, num_warps=warps, num_stages=stages,
    )
    if split > 1:
        _merge[(triton.cdiv(m * 2048, 512),)](dest, C, m, 512, split, num_warps=4)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16 for tensor in (A, B, C)):
        _launch_gemm(A, B, C, 32, 64, 64, 1)
    elif m == 1:
        _gemv[(2048 // 4,)](A, B, C, 4, num_warps=8)
    elif m == 2:
        _vector_gemm[(2048 // 2,)](A, B, C, m, 2, 2, num_warps=4)
    elif m <= 4:
        _vector_gemm[(2048 // 4,)](A, B, C, m, 4, 4, num_warps=8)
    elif m <= 16:
        _launch_gemm(A, B, C, 16, 32, 128, 4, 2, 4)
    elif m <= 192:
        _launch_tma(A, B, C, 64, 64, 128, 5, acc_stages=1, persistent=False)
    elif m <= 512:
        _launch_tma(A, B, C, 128, 64, 128, 4)
    elif m <= 1024:
        _launch_tma(A, B, C, 128, 128, 64, 6)
    else:
        _launch_tma(A, B, C, 128, 256, 64, 4, group=8, part=32)
