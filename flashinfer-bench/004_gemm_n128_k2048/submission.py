import triton
import triton.language as tl


@triton.jit
def _direct(A, B, C, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = A + rows[:, None] * 2048 + k[None, :]
    b = B + cols[None, :] * 2048 + k[:, None]
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(2048 // BK):
        av = tl.load(a, rows[:, None] < M, 0)
        bv = tl.load(b)
        acc = tl.dot(av, bv, acc)
        a += BK
        b += BK
    tl.store(C + rows[:, None] * 128 + cols[None, :], acc, rows[:, None] < M)


def run(A, B, C):
    m = A.shape[0]
    bm, bn, bk = (16, 16, 64) if m < 2048 else (64, 64, 64)
    _direct[(triton.cdiv(m, bm), 128 // bn)](A, B, C, m, bm, bn, bk)
