"""Direct FP16 Tensor Core GEMM with a single destination-passing launch."""

import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, AM, AK, BN, BK, CM, CN,
          TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    cols = tl.program_id(1) * TN + tl.arange(0, TN)
    k = tl.arange(0, TK)
    a_ptr = A + rows[:, None] * AM + k[None, :] * AK
    b_ptr = B + cols[:, None] * BN + k[None, :] * BK
    acc = tl.full((TM, TN), 0, tl.float32)
    for step in range(4096 // TK):
        a = tl.load(a_ptr + step * TK * AK, rows[:, None] < M, 0)
        b = tl.load(b_ptr + step * TK * BK)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(C + rows[:, None] * CM + cols[None, :] * CN, acc, rows[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    if m == 0:
        return
    tm, tn = (16, 32) if m <= 16 else ((32, 64) if m <= 256 else (64, 128))
    _gemm[(triton.cdiv(m, tm), 4096 // tn)](
        A, B, C, m, *A.stride(), *B.stride(), *C.stride(), tm, tn, 64,
        num_warps=4, num_stages=3,
    )
