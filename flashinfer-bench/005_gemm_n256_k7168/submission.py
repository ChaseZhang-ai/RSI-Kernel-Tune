"""Direct-launch FP16 GEMM baseline for FlashInfer-Bench operator 005."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(A, B, C, AM: tl.constexpr, AK: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, CM: tl.constexpr, CN: tl.constexpr):
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, 8192)
    a = tl.load(A + m * AM + k * AK, k < 7168, 0).to(tl.float32)
    b = tl.load(B + n * BN + k * BK, k < 7168, 0).to(tl.float32)
    tl.store(C + m * CM + n * CN, tl.sum(a * b, 0))


@triton.jit
def _matmul(A, B, C, M, AM: tl.constexpr, AK: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr, CM: tl.constexpr, CN: tl.constexpr,
            TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SPLIT: tl.constexpr):
    m = tl.program_id(0) * TM + tl.arange(0, TM)
    n = tl.program_id(1) * TN + tl.arange(0, TN)
    split = tl.program_id(2)
    k = split * (7168 // SPLIT) + tl.arange(0, TK)
    a = A + m[:, None] * AM + k[None, :] * AK
    b = B + n[:, None] * BN + k[None, :] * BK
    acc = tl.full((TM, TN), 0, tl.float32)
    for step in range(7168 // SPLIT // TK):
        av = tl.load(a + step * TK * AK, m[:, None] < M, 0)
        bv = tl.load(b + step * TK * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    if SPLIT == 1:
        tl.store(C + m[:, None] * CM + n[None, :] * CN, acc, m[:, None] < M)
    else:
        tl.store(C + (split * M + m[:, None]) * 256 + n[None, :], acc, m[:, None] < M)


@triton.jit
def _merge(P, C, M, CM: tl.constexpr, CN: tl.constexpr, S: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    s = tl.arange(0, S)
    values = tl.load(P + s[:, None] * M * 256 + i[None, :], i[None, :] < M * 256, 0)
    tl.store(C + i // 256 * CM + i % 256 * CN, tl.sum(values, 0), i < M * 256)


def run(A, B, C):
    m = A.shape[0]
    if not m:
        return
    strides = (*A.stride(), *B.stride(), *C.stride())
    if m <= 4:
        _gemv[(m, 256)](A, B, C, *strides, num_warps=8)
        return
    if m <= 128:
        tm, tn, tk, split = max(16, triton.next_power_of_2(min(m, 64))), 32, 64, 16
        partial = torch.empty((split, m, 256), dtype=torch.float32, device=A.device)
        _matmul[(triton.cdiv(m, tm), 256 // tn, split)](
            A, B, partial, m, *strides, tm, tn, tk, split, num_warps=4, num_stages=3
        )
        _merge[(triton.cdiv(m * 256, 128),)](partial, C, m, *C.stride(), split, num_warps=4)
        return
    tm, tn, tk = 64, 64, 64
    _matmul[(triton.cdiv(m, tm), 256 // tn, 1)](
        A, B, C, m, *strides, tm, tn, tk, 1, num_warps=4, num_stages=3
    )
