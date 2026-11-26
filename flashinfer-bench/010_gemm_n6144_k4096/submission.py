"""FP16 GEMM with FP32 accumulation and destination-passing output."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


N = 6144
_N: tl.constexpr = tl.constexpr(6144)
_K: tl.constexpr = tl.constexpr(4096)


@triton.jit
def _gemv(A, B, C):
    columns = tl.program_id(0) * 4 + tl.arange(0, 4)
    k = tl.arange(0, _K)
    a = tl.load(A + k).to(tl.float32)
    b = tl.load(B + columns[:, None] * _K + k[None, :]).to(tl.float32)
    tl.store(C + columns, tl.sum(a[None, :] * b, axis=1))


# Swapping the operands for small M follows the public CompoundingKernel
# Explorer implementation by QianQu and collaborators.
# https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/010_gemm_n6144_k4096
@triton.jit
def _swapped_gemm(A, B, C, M, TM: tl.constexpr, TK: tl.constexpr,
                  WARP_SPECIALIZE: tl.constexpr):
    row = tl.program_id(0) * TM
    col = tl.program_id(1) * 64
    acc = tl.full((64, TM), 0, tl.float32)
    for step in tl.range(_K // TK, warp_specialize=WARP_SPECIALIZE):
        a = A.load([row, step * TK])
        b = B.load([col, step * TK])
        acc = tl.dot(b, tl.trans(a), acc)
    rows = row + tl.arange(0, TM)
    cols = col + tl.arange(0, 64)
    tl.store(C + rows[:, None] * _N + cols[None, :], tl.trans(acc), rows[:, None] < M)


# The resident CTA schedule and asynchronous staging build on the ideas in
# RSI-Kernel-Tune's H100 GEMM, with K-contiguous FP16 operands for Blackwell.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py
@triton.jit
def _persistent_gemm(A, B, C, M, TM: tl.constexpr, TN: tl.constexpr,
                     TK: tl.constexpr, WORKERS: tl.constexpr,
                     SEGMENTED_EPILOGUE: tl.constexpr):
    tiles: tl.constexpr = _N // TN
    total_tiles = tl.cdiv(M, TM) * tiles
    for tile in tl.range(tl.program_id(0), total_tiles, WORKERS,
                         flatten=True, warp_specialize=True):
        row = tile // tiles * TM
        col = tile % tiles * TN
        acc = tl.full((TM, TN), 0, tl.float32)
        for step in range(_K // TK):
            a = A.load([row, step * TK])
            b = B.load([col, step * TK])
            acc = tl.dot(a, tl.trans(b), acc)
        if SEGMENTED_EPILOGUE:
            halves = tl.reshape(acc, (TM, 2, TN // 2))
            left, right = tl.split(tl.permute(halves, (0, 2, 1)))
            C.store([row, col], left.to(tl.float16))
            C.store([row, col + TN // 2], right.to(tl.float16))
        else:
            C.store([row, col], acc.to(tl.float16))


@triton.jit
def _strided_gemm(A, B, C, M, AM, AK, BN, BK, CM, CN):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    a = A + rows[:, None] * AM + k[None, :] * AK
    b = B + cols[:, None] * BN + k[None, :] * BK
    acc = tl.full((32, 64), 0, tl.float32)
    for step in range(_K // 64):
        av = tl.load(a + step * 64 * AK, rows[:, None] < M, 0)
        bv = tl.load(b + step * 64 * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + rows[:, None] * CM + cols[None, :] * CN, acc, rows[:, None] < M)


def _descriptor(tensor, rows, cols):
    return TensorDescriptor(tensor, tensor.shape, tensor.stride(), [rows, cols])


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    tensors = (A, B, C)
    if any(not tensor.is_contiguous() or tensor.data_ptr() % 16 for tensor in tensors):
        _strided_gemm[(triton.cdiv(m, 32), N // 64)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(),
            num_warps=4, num_stages=3,
        )
        return
    if m == 1:
        _gemv[(N // 4,)](A, B, C, num_warps=8)
        return
    if m <= 128:
        if m <= 16:
            tm, tk, stages, specialize = 16, 256, 3, True
        elif m <= 64:
            tm, tk, stages, specialize = 64, 256, 3, True
        elif m <= 72:
            tm, tk, stages, specialize = 32, 128, 4, False
        else:
            tm, tk, stages, specialize = 128, 128, 4, True
        _swapped_gemm[(triton.cdiv(m, tm), N // 64)](
            _descriptor(A, tm, tk), _descriptor(B, 64, tk), C, m, tm, tk,
            specialize, num_warps=4, num_stages=stages,
        )
        return
    if m <= 192:
        tm, tn, tk, stages, cluster, segmented = 64, 128, 128, 4, 1, False
    elif m <= 256:
        tm, tn, tk, stages, cluster, segmented = 128, 256, 128, 3, 2, False
    elif 2048 < m <= 2176 or m >= 4096:
        tm, tn, tk, stages, cluster, segmented = 128, 256, 64, 3, 1, True
    else:
        tm, tn, tk, stages, cluster, segmented = 128, 128, 128, 3, 1, False
    tiles = triton.cdiv(m, tm) * (N // tn)
    workers = torch.cuda.get_device_properties(A.device).multi_processor_count // cluster
    _persistent_gemm[(min(tiles, workers),)](
        _descriptor(A, tm, tk), _descriptor(B, tn, tk),
        _descriptor(C, tm, tn // 2 if segmented else tn), m,
        tm, tn, tk, workers, segmented, num_warps=4, num_stages=stages, num_ctas=cluster,
    )
