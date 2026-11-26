"""Direct DPS baseline using the repository seed Triton implementation."""

import triton
import triton.language as tl


@triton.jit
def _tri_mm(A, B, C, M, N, K,
            sam, sak, sbn, sbk, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    kstride = BK * SK


    for k0 in range(0, K, kstride):
        abs_k = k0 + pid_k * BK + rk
        a = tl.load(A + (offs_m[:, None] * sam + abs_k[None, :] * sak), mask=m_mask, other=0.0)
        b = tl.load(B + (abs_k[:, None] * sbk + offs_n[None, :] * sbn), mask=n_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = C + (offs_m[:, None] * scm + offs_n[None, :] * scn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SK == 1:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)



def run(A, B, C):
    m, k = A.shape
    n = B.shape[0]
    if m <= 16:
        bm, bn, warps = 16, 32, 2
    elif m <= 64:
        bm, bn, warps = 32, 64, 4
    elif m <= 1024:
        bm, bn, warps = 64, 128, 8
    else:
        bm, bn, warps = 128, 256, 8
    _tri_mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), 1)](
        A, B, C, m, n, k, *A.stride(), *B.stride(), *C.stride(),
        BM=bm, BN=bn, BK=64, SK=1, num_warps=warps, num_stages=4,
    )
