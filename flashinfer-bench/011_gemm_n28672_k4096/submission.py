"""FP16 GEMM using row-major inputs and an FP32 tensor-memory accumulator.

The weight-first small-M layout follows CompoundingKernel Explorer's
FlashInfer 011 notes. The persistent pipeline draws on the RSI H100 GEMM and
Triton's Gluon warp-specialization tutorial.
"""

import inspect

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell as bw
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDescriptor
from triton.tools.tensor_descriptor import TensorDescriptor


_LIST_WARP_SPECIALIZE = gl.constexpr("functions_and_args" in inspect.signature(gl.warp_specialize).parameters)
_COL_STRIDE_LAYOUT = gl.constexpr("col_stride" in inspect.signature(bw.TensorMemoryLayout).parameters)


@triton.jit
def _gemv(A, B, C):
    columns = tl.program_id(0) * 4 + tl.arange(0, 4)
    ks = tl.arange(0, 4096)
    weights = tl.load(B + columns[:, None] * 4096 + ks[None, :]).to(tl.float32)
    activations = tl.load(A + ks).to(tl.float32)
    tl.store(C + columns, tl.sum(weights * activations[None, :], 1).to(tl.float16))


@triton.jit
def _small_mma(A, B, C, M):
    n = tl.program_id(0) * 64 + tl.arange(0, 64)
    m = tl.arange(0, 16)
    k = tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for step in range(4096 // 128):
        columns = step * 128 + k
        a = tl.load(A + m[:, None] * 4096 + columns[None, :], m[:, None] < M, 0)
        b = tl.load(B + n[:, None] * 4096 + columns[None, :], cache_modifier=".cg")
        acc = tl.dot(a, b.T, acc)
    tl.store(C + m[:, None] * 28672 + n[None, :], acc.to(tl.float16), m[:, None] < M)


@triton.jit
def _weight_first(A, B, C, M, BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    n = tl.program_id(0) * BN
    m = tl.program_id(1) * BM
    acc = tl.zeros((BN, BM), tl.float32)
    for step in range(4096 // BK):
        k = step * BK
        acc = tl.dot(B.load([n, k]), A.load([m, k]).T, acc)
    rows = m + tl.arange(0, BM)
    cols = n + tl.arange(0, BN)
    tl.store(C + rows[:, None] * 28672 + cols[None, :], acc.T.to(tl.float16), rows[:, None] < M)


@gluon.jit
def _coords(tile, m_tiles, n_tiles: gl.constexpr, group):
    group_start = tile // (group * n_tiles) * group
    within_group = tile % (group * n_tiles)
    rows = gl.minimum(m_tiles - group_start, group)
    return group_start + within_group % rows, within_group // rows


@gluon.jit
def _load(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    STAGES: gl.constexpr = sa.shape[0]
    n_tiles: gl.constexpr = 28672 // BN
    sequence = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        m, n = _coords(tile, m_tiles, n_tiles, group)
        for step in range(64):
            stage = sequence % STAGES
            wrap = (sequence // STAGES) & 1
            mbarrier.wait(freed.index(stage), wrap ^ 1)
            barrier = ready.index(stage)
            mbarrier.expect(barrier, (128 + BN) * 64 * 2)
            tma.async_copy_global_to_shared(A, [m * 128, step * 64], barrier, sa.index(stage))
            tma.async_copy_global_to_shared(B, [n * BN, step * 64], barrier, sb.index(stage))
            sequence += 1


@gluon.jit
def _mma(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    STAGES: gl.constexpr = sa.shape[0]
    n_tiles: gl.constexpr = 28672 // BN
    sequence = 0
    ordinal = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        slot = ordinal & 1
        phase = (ordinal // 2) & 1
        mbarrier.wait(released.index(slot), phase ^ 1)
        dst = acc.index(slot)
        for step in range(64):
            stage = sequence % STAGES
            wrap = (sequence // STAGES) & 1
            mbarrier.wait(ready.index(stage), wrap)
            bw.tcgen05_mma(sa.index(stage), sb.index(stage).permute((1, 0)), dst, use_acc=step != 0)
            bw.tcgen05_commit(freed.index(stage))
            sequence += 1
        bw.tcgen05_commit(done.index(slot))
        ordinal += 1


@gluon.jit
def _store(args, group):
    A, B, C, sa, sb, ready, freed, acc, done, released = args
    m_tiles = gl.cdiv(C.shape[0], 128)
    BN: gl.constexpr = B.block_type.shape[0]
    PART: gl.constexpr = C.block_type.shape[1]
    n_tiles: gl.constexpr = 28672 // BN
    smem = gl.allocate_shared_memory(gl.float16, [128, PART], C.layout)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    ordinal = 0
    for tile in range(gl.program_id(0), m_tiles * n_tiles, gl.num_programs(0)):
        m, n = _coords(tile, m_tiles, n_tiles, group)
        slot = ordinal & 1
        mbarrier.wait(done.index(slot), (ordinal // 2) & 1)
        src = acc.index(slot)
        for sub in gl.static_range(BN // PART):
            fp16 = src.slice(sub * PART, PART).load(layout).to(gl.float16)
            tma.store_wait(pendings=0)
            smem.store(fp16)
            fence_async_shared()
            tma.async_copy_shared_to_global(C, [m * 128, n * BN + sub * PART], smem)
        mbarrier.arrive(released.index(slot), count=1)
        ordinal += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _barriers(COUNT: gl.constexpr):
    bars = gl.allocate_shared_memory(gl.int64, [COUNT, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(COUNT):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def _persistent(A, B, C, GROUP: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    sa = gl.allocate_shared_memory(gl.float16, [STAGES, 128, 64], A.layout)
    sb = gl.allocate_shared_memory(gl.float16, [STAGES, BN, 64], B.layout)
    if _COL_STRIDE_LAYOUT:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BN], col_stride=1)
    else:
        layout: gl.constexpr = bw.TensorMemoryLayout([128, BN], unpacked=True)
    acc = bw.allocate_tensor_memory(gl.float32, [2, 128, BN], layout)
    args = (A, B, C, sa, sb, _barriers(STAGES), _barriers(STAGES), acc, _barriers(2), _barriers(2))
    config = (args, gl.to_tensor(GROUP))
    if _LIST_WARP_SPECIALIZE:
        gl.warp_specialize([(_store, config), (_load, config), (_mma, config)], [1, 1], [24, 24])
    else:
        gl.warp_specialize(default_args=config, default_partition=_store, worker_args=config,
                           worker_partitions=[_load, _mma], worker_num_warps=[1, 1], worker_num_regs=[24, 24])


def _launch_persistent(A, B, C, bn=256, stages=4, part=64, group=8):
    def descriptor(tensor, tile):
        layout = gl.NVMMASharedLayout.get_default_for(tile, gl.float16)
        return GluonDescriptor.from_tensor(tensor, tile, layout)

    a = descriptor(A, [128, 64])
    b = descriptor(B, [bn, 64])
    c = descriptor(C, [128, part])
    sms = triton.runtime.driver.active.utils.get_device_properties(A.device.index)["multiprocessor_count"]
    grid = min(sms, triton.cdiv(A.shape[0], 128) * (28672 // bn))
    return _persistent[(grid,)](a, b, c, group, bn, stages, num_warps=4)


def run(A, B, C):
    m = A.shape[0]
    if m == 1:
        _gemv[(28672 // 4,)](A, B, C, num_warps=4)
    elif m <= 16:
        _small_mma[(28672 // 64,)](A, B, C, m, num_warps=2, num_stages=4)
    elif m <= 64:
        bm = max(16, triton.next_power_of_2(m))
        bk, stages = (128, 3) if bm <= 32 else (64, 4)
        a = TensorDescriptor.from_tensor(A, [bm, bk])
        b = TensorDescriptor.from_tensor(B, [64, bk])
        _weight_first[(28672 // 64,)](a, b, C, m, bm, 64, bk, num_warps=4, num_stages=stages)
    elif 128 < m <= 256:
        _launch_persistent(A, B, C, bn=128, stages=6)
    elif m <= 1024:
        _launch_persistent(A, B, C)
    elif m < 4096:
        _launch_persistent(A, B, C, part=32, group=24)
    else:
        _launch_persistent(A, B, C, part=32, group=16)
