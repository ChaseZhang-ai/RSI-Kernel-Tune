"""FP16 row-major GEMM with FP32 accumulation and a destination-passing store.

The weight-first small-M mapping follows the approach documented by
CompoundingKernel Explorer for FlashInfer 011. The grouped persistent schedule
follows RSI-Kernel-Tune's H100 GEMM and Triton's persistent matmul tutorial.
"""

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemv(A, B, C):
    cols = tl.program_id(0) * 4 + tl.arange(0, 4)
    ks = tl.arange(0, 4096)
    weights = tl.load(B + cols[:, None] * 4096 + ks[None, :]).to(tl.float32)
    activations = tl.load(A + ks).to(tl.float32)
    tl.store(C + cols, tl.sum(weights * activations[None, :], 1).to(tl.float16))


@triton.jit
def _weight_first(A, B, C, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN
    m = tl.program_id(1) * BM
    acc = tl.zeros((BN, BM), tl.float32)
    for step in tl.range(4096 // BK):
        k = step * BK
        weights = B.load([n, k])
        activations = A.load([m, k])
        acc = tl.dot(weights, activations.T, acc)
    rows = m + tl.arange(0, BM)
    cols = n + tl.arange(0, BN)
    tl.store(C + rows[:, None] * 28672 + cols[None, :], acc.T.to(tl.float16), rows[:, None] < M)


@triton.jit
def _tile(tile, rows, COLS: tl.constexpr, GROUP: tl.constexpr):
    group_start = tile // (COLS * GROUP) * GROUP
    group_size = tl.minimum(rows - group_start, GROUP)
    local = tile % (COLS * GROUP)
    return group_start + local % group_size, local // group_size


@triton.jit
def _persistent(A, B, C, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GRID: tl.constexpr, GROUP: tl.constexpr, SUBTILE: tl.constexpr,
                SPECIALIZE: tl.constexpr):
    rows = tl.cdiv(M, BM)
    cols: tl.constexpr = 28672 // BN
    start = tl.program_id(0)
    store_tile = start - GRID
    for tile in tl.range(start, rows * cols, GRID, flatten=True, warp_specialize=SPECIALIZE):
        row, col = _tile(tile, rows, cols, GROUP)
        acc = tl.zeros((BM, BN), tl.float32)
        for step in range(4096 // BK):
            k = step * BK
            activations = A.load([row * BM, k])
            weights = B.load([col * BN, k])
            acc = tl.dot(activations, weights.T, acc)

        # A separate loop-carried index keeps the epilogue independent of TMA loads.
        store_tile += GRID
        row, col = _tile(store_tile, rows, cols, GROUP)
        if SUBTILE:
            left, right = tl.split(acc.reshape(BM, 2, BN // 2).permute(0, 2, 1))
            C.store([row * BM, col * BN], left.to(tl.float16))
            C.store([row * BM, col * BN + BN // 2], right.to(tl.float16))
        else:
            C.store([row * BM, col * BN], acc.to(tl.float16))


def _launch_weight_first(A, B, C, m, bm, bn, bk, stages):
    a = TensorDescriptor(A, [m, 4096], [4096, 1], [bm, bk])
    b = TensorDescriptor(B, [28672, 4096], [4096, 1], [bn, bk])
    return _weight_first[(28672 // bn, triton.cdiv(m, bm))](
        a, b, C, m, bm, bn, bk, num_warps=4, num_stages=stages,
    )


def _launch_persistent(A, B, C, m, bm, bn, bk, stages, group, subtile, specialize):
    a = TensorDescriptor(A, [m, 4096], [4096, 1], [bm, bk])
    b = TensorDescriptor(B, [28672, 4096], [4096, 1], [bn, bk])
    c = TensorDescriptor(C, [m, 28672], [28672, 1], [bm, bn // 2 if subtile else bn])
    sm_count = triton.runtime.driver.active.utils.get_device_properties(A.device.index)["multiprocessor_count"]
    grid = min(sm_count, triton.cdiv(m, bm) * (28672 // bn))
    return _persistent[(grid,)](a, b, c, m, bm, bn, bk, grid, group, subtile, specialize,
                                num_warps=4, num_stages=stages)


def run(A, B, C):
    m = A.shape[0]
    if m == 1:
        _gemv[(28672 // 4,)](A, B, C, num_warps=4)
    elif m <= 64:
        bm = max(16, triton.next_power_of_2(m))
        bk, stages = (128, 3) if bm <= 32 else (64, 4)
        _launch_weight_first(A, B, C, m, bm, 64, bk, stages)
    elif m <= 128:
        _launch_persistent(A, B, C, m, 128, 256, 64, 3, 4, True, True)
    elif m <= 256:
        _launch_persistent(A, B, C, m, 128, 128, 64, 4, 8, False, True)
    else:
        _launch_persistent(A, B, C, m, 128, 256, 64, 3, 8, True, True)
