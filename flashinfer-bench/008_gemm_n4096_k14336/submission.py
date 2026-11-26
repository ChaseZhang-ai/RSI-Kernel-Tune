"""Direct-launch FP16 GEMM with FP32 accumulation."""

import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, AM, AK, BN, BK, CM, CN):
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
    if m:
        _gemm[(triton.cdiv(m, 32), 64)](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(),
            num_warps=4, num_stages=3,
        )
