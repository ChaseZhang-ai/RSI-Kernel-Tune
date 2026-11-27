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

try:
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    get_tmem_reg_layout = None

_EXPLICIT_TMEM_LAYOUT = gl.constexpr(get_tmem_reg_layout is not None)


# The explicit pipeline is adapted from RSI-Kernel-Tune’s neighboring 010 GEMM.
# It follows the producer/MMA/epilogue partitioning in the Triton Gluon tutorial.
# https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py
# The large GEMM drains the tensor-memory epilogue in subtiles as in Triton's example.
# https://github.com/triton-lang/triton/blob/main/python/examples/gluon/03-matmul-multicta.py
# Swapped operands for small M follow QianQu and the CompoundingKernel Explorer.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/008_gemm_n4096_k14336
# Tile residency and staging draw on RSI-Kernel-Tune's H100 GEMM.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py


@gluon.jit
def _tile_origin(tile, m, TM: gl.constexpr, TN: gl.constexpr, GROUP: gl.constexpr):
    tiles_n: gl.constexpr = 4096 // TN
    if GROUP == 1:
        return tile // tiles_n * TM, tile % tiles_n * TN
    first = tile // (GROUP * tiles_n) * GROUP
    span = gl.minimum(GROUP, gl.cdiv(m, TM) - first)
    local = tile % (GROUP * tiles_n)
    return (first + local % span) * TM, local // span * TN


@gluon.jit
def _read_tiles(a, b, a_ring, b_ring, load_free, load_ready,
                TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                STAGES: gl.constexpr, WORKERS, GROUP: gl.constexpr,
                SPLITS: gl.constexpr):
    tiles_n: gl.constexpr = 4096 // TN
    total = gl.cdiv(a.shape[0], TM) * tiles_n
    step = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row, col = _tile_origin(tile, a.shape[0], TM, TN, GROUP)
        for k in range(14336 // (TK * SPLITS)):
            index = step % STAGES
            phase = step // STAGES & 1
            mbarrier.wait(load_free.index(index), phase ^ 1)
            ready = load_ready.index(index)
            # Each TMA copy fills its whole FP16 tile, including padded rows.
            mbarrier.expect(ready, (TM + TN) * TK * 2)
            k_offset = (gl.program_id(1) * (14336 // SPLITS)) + k * TK
            tma.async_copy_global_to_shared(a, [row, k_offset], ready, a_ring.index(index))
            tma.async_copy_global_to_shared(b, [col, k_offset], ready, b_ring.index(index))
            step += 1


@gluon.jit
def _multiply_tiles(m, a_ring, b_ring, load_free, load_ready, acc, acc_free, acc_ready,
                    TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                    STAGES: gl.constexpr, WORKERS, SWAP: gl.constexpr,
                    ACC_STAGES: gl.constexpr, SPLITS: gl.constexpr):
    total = gl.cdiv(m, TM) * (4096 // TN)
    step = 0
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_free.index(acc_index), acc_phase ^ 1)
        out = acc.index(acc_index)
        for k in range(14336 // (TK * SPLITS)):
            index = step % STAGES
            phase = step // STAGES & 1
            mbarrier.wait(load_ready.index(index), phase)
            operand_a = a_ring.index(index)
            operand_b = b_ring.index(index)
            if SWAP:
                lhs, rhs = operand_b, operand_a.permute((1, 0))
            else:
                lhs, rhs = operand_a, operand_b.permute((1, 0))
            tcgen05_mma(lhs, rhs, out, use_acc=k != 0, mbarriers=[load_free.index(index)])
            step += 1
        tcgen05_commit(acc_ready.index(acc_index))
        wave += 1


@gluon.jit
def _load_accumulator(slot):
    if _EXPLICIT_TMEM_LAYOUT:
        # Triton 3.6 requires the epilogue's tensor-memory load layout explicitly.
        layout: gl.constexpr = get_tmem_reg_layout(gl.float32, slot.shape, slot.layout, 4)
        return slot.load(layout)
    else:
        return slot.load()


@gluon.jit
def _write_tiles(c, m, acc, acc_free, acc_ready, TM: gl.constexpr, TN: gl.constexpr,
                 WORKERS, SWAP: gl.constexpr, ACC_STAGES: gl.constexpr,
                 GROUP: gl.constexpr, SPLITS: gl.constexpr, EN: gl.constexpr):
    total = gl.cdiv(m, TM) * (4096 // TN)
    temp = gl.allocate_shared_memory(c.dtype, [TM, EN], c.layout)
    wave = 0
    for tile in range(gl.program_id(0), total, WORKERS):
        row, col = _tile_origin(tile, m, TM, TN, GROUP)
        if SPLITS > 1:
            row += gl.program_id(1) * (gl.cdiv(m, TM) * TM)
        acc_index = wave % ACC_STAGES
        acc_phase = wave // ACC_STAGES & 1
        mbarrier.wait(acc_ready.index(acc_index), acc_phase)
        slot = acc.index(acc_index)
        if SWAP:
            values = _load_accumulator(slot).to(c.dtype)
            tma.store_wait(pendings=0)
            temp.store(values.permute((1, 0)))
            fence_async_shared()
            tma.async_copy_shared_to_global(c, [row, col], temp)
        else:
            for sub in gl.static_range(TN // EN):
                values = _load_accumulator(slot.slice(sub * EN, EN)).to(c.dtype)
                tma.store_wait(pendings=0)
                temp.store(values)
                fence_async_shared()
                tma.async_copy_shared_to_global(c, [row, col + sub * EN], temp)
        mbarrier.arrive(acc_free.index(acc_index))
        wave += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _matmul(a, b, c, TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
            STAGES: gl.constexpr, SWAP: gl.constexpr,
            ACC_STAGES: gl.constexpr, GROUP: gl.constexpr, SPLITS: gl.constexpr,
            EN: gl.constexpr):
    workers = gl.num_programs(0)
    a_ring = gl.allocate_shared_memory(gl.float16, [STAGES, TM, TK], a.layout)
    b_ring = gl.allocate_shared_memory(gl.float16, [STAGES, TN, TK], b.layout)
    # One CTA owns each ring, so every barrier occupies one local 64-bit slot.
    barrier_layout: gl.constexpr = mbarrier.MBarrierLayout()
    load_free = gl.allocate_shared_memory(gl.int64, [STAGES, 1], barrier_layout)
    load_ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], barrier_layout)
    acc_free = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 1], barrier_layout)
    acc_ready = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 1], barrier_layout)
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
        (_write_tiles, (c, a.shape[0], acc, acc_free, acc_ready, TM, TN, workers, SWAP,
                        ACC_STAGES, GROUP, SPLITS, EN)),
        (_read_tiles, (a, b, a_ring, b_ring, load_free, load_ready, TM, TN, TK,
                       STAGES, workers, GROUP, SPLITS)),
        (_multiply_tiles, (a.shape[0], a_ring, b_ring, load_free, load_ready, acc, acc_free,
                           acc_ready, TM, TN, TK, STAGES, workers, SWAP, ACC_STAGES,
                           SPLITS)),
    ], [1, 1], [24, 32])


def _desc(tensor, rows, cols, dtype=gl.float16):
    shape = [rows, cols]
    layout = gl.NVMMASharedLayout.get_default_for(shape, dtype)
    return TensorDescriptor.from_tensor(tensor, shape, layout)


def _launch(A, B, C, tm, tn, tk, stages, swap, acc_stages=1, group=1, splits=1, epilogue_n=None):
    m = A.shape[0]
    sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    tiles_m = triton.cdiv(m, tm)
    tiles = tiles_m * (4096 // tn)
    workers = min(tiles, sms // splits)
    en = epilogue_n or tn
    if splits == 1:
        desc_c = _desc(C, tm, en)
    else:
        # Pad each partial plane independently so that the TMA store can write
        # the last full tile without stepping into the next split.
        padded_m = tiles_m * tm
        partials = torch.empty((splits * padded_m, 4096), device=A.device, dtype=torch.float32)
        desc_c = _desc(partials, tm, en, gl.float32)
    _matmul[(workers, splits)](
        _desc(A, tm, tk), _desc(B, tn, tk), desc_c,
        tm, tn, tk, stages, swap, acc_stages, group, splits, en, num_warps=4,
    )
    if splits > 1:
        _sum_partials[(m * 16,)](partials, C, padded_m, splits, 256)


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
def _sum_partials(P, C, M, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    partials = tl.load(P + tl.arange(0, SPLITS)[:, None] * (M * 4096) + offsets[None, :])
    tl.store(C + offsets, tl.sum(partials, axis=0))


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
    """Write A @ B.T into the caller's FP16 destination on the current stream."""
    m = A.shape[0]
    if m == 0:
        return
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16 for tensor in (A, B, C)):
        _strided_gemm[(triton.cdiv(m, 32), 64)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(), num_warps=4, num_stages=3,
        )
    elif m == 1:
        _gemv[(2048,)](A, B, C, 2, 2048, num_warps=4)
    elif m <= 16:
        _launch(A, B, C, 16, 64, 256, 5, True, splits=2)
    elif m <= 32:
        _launch(A, B, C, 32, 64, 256, 4, True, splits=2)
    elif m <= 64:
        _launch(A, B, C, 32, 64, 256, 4, True, group=4)
    elif m <= 128:
        _launch(A, B, C, 128, 64, 128, 4, True, splits=2)
    elif m <= 256:
        _launch(A, B, C, 64, 128, 128, 4, False)
    else:
        _launch(A, B, C, 128, 256, 64, 4, False, acc_stages=2, group=4, epilogue_n=64)
