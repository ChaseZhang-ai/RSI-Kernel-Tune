"""Direct-launch FP16 GEMM for the fixed N=5120, K=2048 contract."""

import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, AM, AK, BN, BK, CM, CN,
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


def run(A, B, C):
    m = A.shape[0]
    if m:
        _gemm[(5120 // 128, triton.cdiv(m, 32))](
            A, B, C, m, *A.stride(), *B.stride(), *C.stride(),
            32, 128, 64, num_warps=4, num_stages=3,
        )
