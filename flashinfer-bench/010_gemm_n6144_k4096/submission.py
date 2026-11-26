import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, AM, AK, BN, BK, CM, CN,
          TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    cols = tl.program_id(1) * TN + tl.arange(0, TN)
    k = tl.arange(0, TK)
    a = A + rows[:, None] * AM + k[None, :] * AK
    b = B + cols[:, None] * BN + k[None, :] * BK
    acc = tl.full((TM, TN), 0, tl.float32)
    for step in range(4096 // TK):
        av = tl.load(a + step * TK * AK, rows[:, None] < M, 0)
        bv = tl.load(b + step * TK * BK)
        acc = tl.dot(av, tl.trans(bv), acc)
    tl.store(C + rows[:, None] * CM + cols[None, :] * CN, acc, rows[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    tm = 16 if m <= 16 else 32 if m <= 64 else 64
    _gemm[(triton.cdiv(m, tm), 6144 // 64)](
        A, B, C, m, *A.stride(), *B.stride(), *C.stride(),
        tm, 64, 64, num_warps=4, num_stages=3,
    )
