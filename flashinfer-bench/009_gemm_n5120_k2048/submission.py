"""Destination-passing FP16 GEMM for N=5120 and K=2048."""

import torch
import triton
import triton.language as tl
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


# The resident CTA schedule and staged, output-stationary accumulation draw
# on RSI-Kernel-Tune's H100 GEMM, with K-contiguous FP16 inputs for Blackwell.
# https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py
@triton.jit
def _persistent_gemm(A, B, C, M, WORKERS: tl.constexpr, TK: tl.constexpr,
                     GROUP: tl.constexpr):
    m_tiles = tl.cdiv(M, 128)
    for tile in tl.range(tl.program_id(0), m_tiles * (5120 // 256), WORKERS, flatten=True):
        mi, ni = _tile_coords(tile, m_tiles, 5120 // 256, GROUP)
        row = mi * 128
        col = ni * 256
        acc = _accumulate(A, B, row, col, 128, 256, TK, False)
        halves = tl.reshape(acc.to(tl.float16), (128, 2, 128))
        left, right = tl.split(tl.permute(halves, (0, 2, 1)))
        C.store([row, col], left)
        C.store([row, col + 128], right)


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
        return 64, 64, 256, 3, True, 4
    if m <= 128:
        return 128, 64, 128, 4, True, 4
    if m <= 192:
        return 64, 64, 128, 3, True, 4
    return 128, 128, 64, 3, False, 4


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
    elif 192 < m <= 320:
        _launch_pointer(A, B, C, 64, 128, 64)
    elif m <= 1024:
        tm, tn, tk, stages, swap, warps = _raster_config(m)
        _raster_gemm[(triton.cdiv(m, tm) * (N // tn),)](
            _descriptor(A, tm, tk), _descriptor(B, tn, tk), C,
            m, tm, tn, tk, swap, num_warps=warps, num_stages=stages,
        )
    else:
        workers = torch.cuda.get_device_properties(A.device).multi_processor_count
        tk, stages, group = 64, 4, 4
        grid = min(workers, triton.cdiv(m, 128) * (N // 256))
        _persistent_gemm[(grid,)](
            _descriptor(A, 128, tk), _descriptor(B, 256, tk), _descriptor(C, 128, 128),
            m, workers, tk, group, num_warps=4, num_stages=stages,
        )
