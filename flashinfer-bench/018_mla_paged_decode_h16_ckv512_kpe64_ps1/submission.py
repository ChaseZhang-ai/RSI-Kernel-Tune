"""BF16 paged MLA decode with FP32 online softmax and a split-KV merge.

Tile reuse follows the RSI-Kernel-Tune H100 seed. Length-dependent scheduling
is inspired by CompoundingKernel Explorer's FlashInfer 018 CUDA submission
(Claude via Amazon Bedrock).
"""

import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, mbarrier, tcgen05_mma, tma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@triton.jit
def _small(Q, QP, C, KP, Indptr, Indices, O, L, scale, T: tl.constexpr, N: tl.constexpr):
    h = tl.program_id(0)
    begin = tl.load(Indptr)
    end = tl.load(Indptr + 1)
    t = tl.arange(0, N)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    live = (t >= begin) & (t < end) & (t < T)
    pages = tl.load(Indices + t, t < T, 0)
    ckv = tl.load(C + pages[:, None] * 512 + d[None, :], live[:, None], 0).to(tl.float32)
    kpe = tl.load(KP + pages[:, None] * 64 + r[None, :], live[:, None], 0).to(tl.float32)
    q = tl.load(Q + h * 512 + d).to(tl.float32)
    qp = tl.load(QP + h * 64 + r).to(tl.float32)
    logits = tl.sum(ckv * q[None, :], 1) + tl.sum(kpe * qp[None, :], 1)
    score = tl.where(live, logits * scale, float("-inf"))
    peak = tl.max(score, 0)
    weights = tl.exp2(score - tl.where(begin < end, peak, 0.0))
    denom = tl.sum(weights, 0)
    safe = tl.where(denom > 0, denom, 1.0)
    out = tl.sum(ckv * weights[:, None], 0) / safe
    tl.store(O + h * 512 + d, out)
    tl.store(L + h, peak + tl.log2(safe))


@triton.jit
def _merge(Indptr, Partial, Stats, O, L, CHUNK: tl.constexpr,
           M: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    part = tl.program_id(2)
    begin = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    first = begin // CHUNK + row
    count = tl.cdiv(end - begin, CHUNK)
    d = part * D + tl.arange(0, D)
    i = tl.arange(0, M)
    acc = tl.full((D,), 0, tl.float32)
    peak = float("-inf")
    denom = 0.0
    for offset in range(0, count, M):
        slot = first + offset + i
        live = offset + i < count
        maxima = tl.load(Stats + slot * 32 + h, live, float("-inf"))
        sums = tl.load(Stats + slot * 32 + h + 16, live, 0)
        new_peak = tl.maximum(peak, tl.max(maxima, 0))
        decay = tl.exp2(peak - new_peak)
        weights = tl.exp2(maxima - new_peak)
        values = tl.load(Partial + ((slot * 16 + h) * 512)[:, None] + d[None, :], live[:, None], 0)
        acc = acc * decay + tl.sum(values * weights[:, None], 0)
        denom = denom * decay + tl.sum(sums * weights, 0)
        peak = new_peak
    safe = tl.where(denom > 0, denom, 1.0)
    head = row * 16 + h
    tl.store(O + head * 512 + d, acc / safe)
    if part == 0:
        tl.store(L + head, peak + tl.log2(safe))


@triton.constexpr_function
def _tmem_layout(n):
    if 'col_stride' in TensorMemoryLayout.__dataclass_fields__:
        return TensorMemoryLayout([n, 16], col_stride=1)
    return TensorMemoryLayout([n, 16], unpacked=True)


@gluon.jit
def _tc_decode(Q, QP, C, KP, Indptr, Indices, Partial, Stats, O, L, scale,
               B: gl.constexpr, N: gl.constexpr, DIRECT: gl.constexpr,
               GATHER: gl.constexpr):
    slot = gl.program_id(0)
    if B == 1:
        row = 0
        first = gl.load(Indptr)
    else:
        # Each CSR row owns ceil(length / N) slots and at most one spare slot.
        # The row offset makes the slot bases strictly increasing, even for empty rows.
        rows = gl.arange(0, triton.next_power_of_2(B), layout=gl.BlockedLayout([1], [32], [4], [0]))
        starts = gl.load(Indptr + rows, rows < B, 0)
        row = gl.max(gl.where((rows < B) & (starts // N + rows <= slot), rows, 0), 0)
        first = gl.sum(gl.where(rows == row, starts, 0), 0)
    end = gl.load(Indptr + row + 1)
    begin = first if DIRECT else first + (slot - first // N - row) * N
    GLOBAL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    STORAGE: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    Q_LAYOUT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([16, 512], gl.bfloat16)
    QP_LAYOUT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([16, 64], gl.bfloat16)
    C_LAYOUT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, 512], gl.bfloat16)
    KP_LAYOUT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, 64], gl.bfloat16)
    P_LAYOUT: gl.constexpr = gl.NVMMASharedLayout.get_default_for([N, 16], gl.bfloat16)
    h = gl.arange(0, 16, layout=gl.SliceLayout(1, GLOBAL))
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, GLOBAL))
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, GLOBAL))
    head = row * 16 + h

    if begin < end:
        t = begin + gl.arange(0, N, layout=gl.SliceLayout(1, GLOBAL))
        IDX: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        q = gl.allocate_shared_memory(gl.bfloat16, [16, 512], Q_LAYOUT)
        qp = gl.allocate_shared_memory(gl.bfloat16, [16, 64], QP_LAYOUT)
        ckv = gl.allocate_shared_memory(gl.bfloat16, [N, 512], C_LAYOUT)
        kpe = gl.allocate_shared_memory(gl.bfloat16, [N, 64], KP_LAYOUT)
        async_copy.async_copy_global_to_shared(q, Q + head[:, None] * 512 + d[None, :])
        async_copy.async_copy_global_to_shared(qp, QP + head[:, None] * 64 + r[None, :])
        if GATHER:
            ti = begin + gl.arange(0, N, layout=IDX)
            page = gl.load(Indices + ti, ti < end, -1)
            OFFSETS: gl.constexpr = gl.SliceLayout(1, gl.BlockedLayout([4, 1], [1, 32], [4, 1], [0, 1]))
            gather_page = gl.convert_layout(page, OFFSETS)
            copy_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(copy_bar, count=1)
            mbarrier.expect(copy_bar, N * 1152)
            tma.async_gather(C, gather_page, 0, copy_bar, ckv)
            tma.async_gather(KP, gather_page, 0, copy_bar, kpe)
            async_copy.commit_group()
            async_copy.wait_group(0)
            mbarrier.wait(copy_bar, phase=0)
            mbarrier.invalidate(copy_bar)
        else:
            page = gl.load(Indices + t, t < end, 0)
            async_copy.async_copy_global_to_shared(ckv, C + page[:, None] * 512 + d[None, :],
                                                   (t < end)[:, None], cache_modifier=".cg")
            async_copy.async_copy_global_to_shared(kpe, KP + page[:, None] * 64 + r[None, :],
                                                   (t < end)[:, None], cache_modifier=".cg")
            async_copy.commit_group()
            async_copy.wait_group(0)

        SCORE: gl.constexpr = _tmem_layout(N)
        if N == 64:
            SCORE_REG: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [4, 1], [0, 1])
        else:
            SCORE_REG: gl.constexpr = gl.BlockedLayout([1, 16], [32, 1], [4, 1], [0, 1])
        s = allocate_tensor_memory(gl.float32, [N, 16], SCORE)
        score_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(score_bar, count=1)
        # Both contributions accumulate in the same FP32 score tile before scaling.
        tcgen05_mma(ckv, q.permute((1, 0)), s, use_acc=False)
        tcgen05_mma(kpe, qp.permute((1, 0)), s, mbarriers=[score_bar])
        mbarrier.wait(score_bar, phase=0)
        mbarrier.invalidate(score_bar)
        score = s.load(SCORE_REG) * scale

        SOFTMAX: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [4, 1], [0, 1])
        score = gl.convert_layout(score, SOFTMAX)
        live = gl.convert_layout(t < end, gl.SliceLayout(1, SOFTMAX))
        masked = gl.where(live[:, None], score, float("-inf"))
        maximum = gl.max(masked, 0)
        weight = gl.exp2(masked - maximum[None, :])
        denom = gl.sum(weight, 0)
        p = gl.allocate_shared_memory(gl.bfloat16, [N, 16], P_LAYOUT, weight.to(gl.bfloat16))
        fence_async_shared()

        OUT: gl.constexpr = _tmem_layout(128)
        OUT_REG: gl.constexpr = gl.BlockedLayout([1, 16], [32, 1], [4, 1], [0, 1])
        value_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(value_bar, count=1)
        z0 = allocate_tensor_memory(gl.float32, [128, 16], OUT)
        z1 = allocate_tensor_memory(gl.float32, [128, 16], OUT)
        z2 = allocate_tensor_memory(gl.float32, [128, 16], OUT)
        z3 = allocate_tensor_memory(gl.float32, [128, 16], OUT)
        # Reinterpret the same shared-memory CKV tile as the value operand.
        value = ckv.permute((1, 0))
        tcgen05_mma(value.slice(0, 128), p, z0, use_acc=False)
        tcgen05_mma(value.slice(128, 128), p, z1, use_acc=False)
        tcgen05_mma(value.slice(256, 128), p, z2, use_acc=False)
        tcgen05_mma(value.slice(384, 128), p, z3, use_acc=False, mbarriers=[value_bar])
        mbarrier.wait(value_bar, phase=0)
        mbarrier.invalidate(value_bar)

        heads = row * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, STORAGE))
        dims = gl.arange(0, 128, layout=gl.SliceLayout(0, STORAGE))
        m = gl.convert_layout(maximum, gl.SliceLayout(1, STORAGE))
        l = gl.convert_layout(denom, gl.SliceLayout(1, STORAGE))
        if DIRECT:
            out = O + heads[:, None] * 512 + dims[None, :]
            inv = 1.0 / l
            gl.store(L + heads, m + gl.log2(l))
        else:
            out = Partial + (slot * 16 + heads[:, None] % 16) * 512 + dims[None, :]
            inv = gl.full((16,), 1.0, gl.float32, gl.SliceLayout(1, STORAGE))
            gl.store(Stats + slot * 32 + heads % 16, m)
            gl.store(Stats + slot * 32 + heads % 16 + 16, l)
        gl.store(out, gl.convert_layout(z0.load(OUT_REG).permute((1, 0)), STORAGE) * inv[:, None])
        gl.store(out + 128, gl.convert_layout(z1.load(OUT_REG).permute((1, 0)), STORAGE) * inv[:, None])
        gl.store(out + 256, gl.convert_layout(z2.load(OUT_REG).permute((1, 0)), STORAGE) * inv[:, None])
        gl.store(out + 384, gl.convert_layout(z3.load(OUT_REG).permute((1, 0)), STORAGE) * inv[:, None])
    elif DIRECT:
        zeros = gl.full((16, 512), 0.0, gl.float32, GLOBAL)
        empty_lse = gl.full((16,), float("-inf"), gl.float32, gl.SliceLayout(1, GLOBAL))
        gl.store(O + head[:, None] * 512 + d[None, :], zeros)
        gl.store(L + head, empty_lse)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch = q_nope.shape[0]
    tokens = kv_indices.numel()
    scale = sm_scale * math.log2(math.e)
    if batch == 1 and tokens <= 16:
        _small[(16,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                      output, lse, scale, tokens, triton.next_power_of_2(max(tokens, 1)))
        return
    if batch == 1 and tokens <= 128:
        # Gather directly into the tensor-core tile and skip the global merge.
        layout = gl.NVMMASharedLayout(128, 16, rank=2)
        ckv_desc = TensorDescriptor(ckv_cache, [ckv_cache.shape[0], 512], [512, 1], [1, 512], layout)
        kpe_desc = TensorDescriptor(kpe_cache, [kpe_cache.shape[0], 64], [64, 1], [1, 64], layout)
        _tc_decode[(1,)](q_nope, q_pe, ckv_desc, kpe_desc, kv_indptr, kv_indices,
                         output, lse, output, lse, scale, batch, 128, True, True)
        return

    n = 64 if batch == 1 or tokens < 9000 else 128
    slots = batch + tokens // n
    partial_dtype = torch.float32 if batch == 1 else torch.bfloat16
    partial = torch.empty((slots, 16, 512), device=q_nope.device, dtype=partial_dtype)
    stats = torch.empty((slots, 32), device=q_nope.device, dtype=torch.float32)
    _tc_decode[(slots,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                         partial, stats, output, lse, scale, batch, n, False, False)
    width = 128 if batch == 1 else 512
    _merge[(batch, 16, 512 // width)](kv_indptr, partial, stats, output, lse, n, 8, width)
