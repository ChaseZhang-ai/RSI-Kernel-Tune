import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for step in range(4096 // BK):
        column = k + step * BK
        a = tl.load(A + m[:, None] * 4096 + column[None, :], m[:, None] < M, 0)
        b = tl.load(B + n[:, None] * 4096 + column[None, :])
        accumulator = tl.dot(a, tl.trans(b), accumulator)
    tl.store(C + m[:, None] * 28672 + n[None, :], accumulator.to(tl.float16), m[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    bm = 16 if m <= 16 else 32 if m <= 32 else 64
    _gemm[(28672 // 128, triton.cdiv(m, bm))](A, B, C, m, bm, 128, 64, num_warps=4, num_stages=3)
