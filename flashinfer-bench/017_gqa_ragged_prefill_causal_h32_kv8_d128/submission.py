# SPDX-License-Identifier: Apache-2.0
"""Fused destination-passing ragged causal GQA for FlashInfer-Bench 017.

The four-head packing and heavy-first tile order draw on CompoundingKernel
Explorer's FlashInfer 017 archive. The TMA and tcgen05 pipeline adapts the
repository's FlashInfer 016 solution, which follows the Triton Gluon tutorial
on warp specialization. The prefix and diagonal online softmax follows the
repository's dense H100 causal GQA reference.
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tcgen05_mma,
)
try:
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    get_tmem_reg_layout = None

from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor




@triton.jit
def _single(Q, K, V, O, L, scale, QS: tl.constexpr, KS: tl.constexpr,
            VS: tl.constexpr, OS: tl.constexpr, LS: tl.constexpr):
    head = tl.program_id(0)
    d = tl.arange(0, 128)
    q = tl.load(Q + head * QS[1] + d * QS[2]).to(tl.float32)
    k = tl.load(K + (head // 4) * KS[1] + d * KS[2]).to(tl.float32)
    v = tl.load(V + (head // 4) * VS[1] + d * VS[2])
    tl.store(O + head * OS[1] + d * OS[2], v)
    tl.store(L + head * LS[1], tl.sum(q * k, 0) * (scale * 1.4426950408889634))


@triton.jit
def _short(Q, K, V, QP, KP, O, L, scale,
           QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
           OS: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
           NQ: tl.constexpr, NK: tl.constexpr, BATCH: tl.constexpr,
           B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr,
           BOUND_K: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    TOKENS: tl.constexpr = M // GROUP
    if BATCH == 1:
        qs, qe, ks, ke = 0, NQ, 0, NK
        local = tile
    else:
        seq = tl.arange(0, B)
        starts = tl.load(QP + seq * PS[0], seq < BATCH, 0)
        ends = tl.load(QP + (seq + 1) * PS[0], seq < BATCH, 0)
        counts = tl.cdiv(ends - starts, TOKENS)
        cumulative = tl.cumsum(counts)
        owns = (cumulative - counts <= tile) & (tile < cumulative)
        b = tl.sum(tl.where(owns, seq, 0), 0)
        qs = tl.sum(tl.where(owns, starts, 0), 0)
        qe = tl.sum(tl.where(owns, ends, 0), 0)
        if qs == qe:
            return
        ks = tl.load(KP + b * PS[1])
        ke = tl.load(KP + (b + 1) * PS[1])
        local = tile - tl.sum(tl.where(owns, cumulative - counts, 0), 0)
    rows = tl.arange(0, M)
    qt = qs + local * TOKENS + rows // GROUP
    active = qt < qe
    qh = group * GROUP + rows % GROUP
    h = group // (4 // GROUP)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    kt = ks + n
    key_end = ke
    if BOUND_K:
        key_end = tl.minimum(ke, qs + (local + 1) * TOKENS - qe + ke)
    q = tl.load(Q + qt[:, None] * QS[0] + qh[:, None] * QS[1] + d[None, :] * QS[2],
                active[:, None], 0)
    k = tl.load(K + kt[:, None] * KS[0] + h * KS[1] + d[None, :] * KS[2],
                (kt < key_end)[:, None], 0)
    v = tl.load(V + kt[:, None] * VS[0] + h * VS[1] + d[None, :] * VS[2],
                (kt < key_end)[:, None], 0)
    end = qt - qe + ke + 1
    scores = tl.dot(q, tl.trans(k)) * (scale * 1.4426950408889634)
    scores = tl.where(kt[None, :] < end[:, None], scores, -float("inf"))
    maximum = tl.maximum(tl.max(scores, 1), -3.4028234663852886e38)
    weights = tl.exp2(scores - maximum[:, None])
    denom = tl.sum(weights, 1)
    acc = tl.dot(weights.to(tl.bfloat16), v)
    valid = denom > 0
    out = tl.where(valid[:, None], acc / tl.where(valid, denom, 1.0)[:, None],
                   tl.where(ke > ks, float("nan"), 0.0))
    tl.store(O + qt[:, None] * OS[0] + qh[:, None] * OS[1] + d[None, :] * OS[2],
             out, active[:, None])
    tl.store(L + qt * LS[0] + qh * LS[1], maximum + tl.log2(denom), active)


@triton.jit
def _step(K, V, q, ks, ke, h, n0, end, d, scale, m, l, acc,
          KS: tl.constexpr, VS: tl.constexpr, N: tl.constexpr, MASK: tl.constexpr):
    kt = ks + n0 + tl.arange(0, N)
    k_ptr = K + kt[:, None] * KS[0] + h * KS[1] + d[None, :] * KS[2]
    v_ptr = V + kt[:, None] * VS[0] + h * VS[1] + d[None, :] * VS[2]
    if MASK:
        k = tl.load(k_ptr, (kt < ke)[:, None], 0)
        v = tl.load(v_ptr, (kt < ke)[:, None], 0)
    else:
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
    scores = tl.dot(q, tl.trans(k)) * scale
    if MASK:
        scores = tl.where(kt[None, :] < end[:, None], scores, -float("inf"))
    m_new = tl.maximum(m, tl.max(scores, 1))
    alpha = tl.exp2(m - m_new)
    weights = tl.exp2(scores - m_new[:, None])
    l = l * alpha + tl.sum(weights, 1)
    acc = tl.dot(weights.to(tl.bfloat16), v, acc * alpha[:, None])
    return m_new, l, acc


@triton.jit
def _ragged(Q, K, V, QP, KP, O, L, scale,
            QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
            OS: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
            BATCH: tl.constexpr, B: tl.constexpr, M: tl.constexpr,
            FULL_N: tl.constexpr, DIAG_N: tl.constexpr, GROUP: tl.constexpr):
    seq = tl.arange(0, B)
    ends = tl.load(QP + (seq + 1) * PS[0], seq < BATCH, 0)
    starts = tl.load(QP + seq * PS[0], seq < BATCH, 0)
    tiles = tl.cdiv(ends - starts, M // GROUP)
    cumulative = tl.cumsum(tiles)
    groups: tl.constexpr = 32 // GROUP
    work = tl.program_id(0)
    tile = work // groups
    group = work % groups
    scale2 = scale * 1.4426950408889634
    if tile < tl.sum(tiles, 0):
        b = tl.sum(((cumulative <= tile) & (seq < BATCH)).to(tl.int32), 0)
        # Start with the larger causal windows within each request.
        local = tl.sum(tl.where(seq == b, cumulative, 0), 0) - 1 - tile
        qs = tl.sum(tl.where(seq == b, starts, 0), 0)
        qe = tl.sum(tl.where(seq == b, ends, 0), 0)
        ks = tl.load(KP + b * PS[1])
        ke = tl.load(KP + (b + 1) * PS[1])
        row0 = qs + local * (M // GROUP)
        r = tl.arange(0, M)
        d = tl.arange(0, 128)
        qt = row0 + r // GROUP
        qh = group * GROUP + r % GROUP
        h = group // (4 // GROUP)
        active = qt < qe
        q = tl.load(Q + qt[:, None] * QS[0] + qh[:, None] * QS[1] + d[None, :] * QS[2],
                    active[:, None], 0)
        end = qt - qe + ke + 1
        fully_visible = tl.maximum(row0 - qe + ke - ks + 1, 0)
        prefix = fully_visible // FULL_N * FULL_N
        hi = tl.maximum(tl.minimum(row0 + M // GROUP, qe) - qe + ke - ks, 0)
        m = tl.full((M,), -3.4028234663852886e38, tl.float32)
        l = tl.zeros((M,), tl.float32)
        acc = tl.zeros((M, 128), tl.float32)
        for n0 in tl.range(0, prefix, FULL_N):
            m, l, acc = _step(K, V, q, ks, ke, h, n0, end, d, scale2, m, l, acc,
                              KS, VS, FULL_N, False)
        for n0 in range(prefix, hi, DIAG_N):
            m, l, acc = _step(K, V, q, ks, ke, h, n0, end, d, scale2, m, l, acc,
                              KS, VS, DIAG_N, True)
        valid = l > 0
        out = tl.where(valid[:, None], acc / tl.where(valid, l, 1.0)[:, None],
                       tl.where(ke > ks, float("nan"), 0.0))
        tl.store(O + qt[:, None] * OS[0] + qh[:, None] * OS[1] + d[None, :] * OS[2],
                 out, active[:, None])
        tl.store(L + qt * LS[0] + qh * LS[1], m + tl.log2(l), active)


@triton.jit
def _simt_short(Q, K, V, O, L, scale, QS: tl.constexpr, KS: tl.constexpr,
                VS: tl.constexpr, OS: tl.constexpr, LS: tl.constexpr,
                NQ: tl.constexpr, NK: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    h = head // 4
    key = tl.arange(0, N)
    d = tl.arange(0, 128)
    valid = key <= row + NK - NQ
    q = tl.load(Q + row * QS[0] + head * QS[1] + d * QS[2]).to(tl.float32)
    k = tl.load(K + key[:, None] * KS[0] + h * KS[1] + d[None, :] * KS[2],
                valid[:, None], 0).to(tl.float32)
    v = tl.load(V + key[:, None] * VS[0] + h * VS[1] + d[None, :] * VS[2],
                valid[:, None], 0).to(tl.float32)
    score = tl.where(valid, tl.sum(k * q[None, :], 1) * (scale * 1.4426950408889634),
                     -float("inf"))
    maximum = tl.maximum(tl.max(score, 0), -3.4028234663852886e38)
    p = tl.exp2(score - maximum)
    denom = tl.sum(p, 0)
    out = tl.sum(v * p[:, None], 0) / denom
    tl.store(O + row * OS[0] + head * OS[1] + d * OS[2], out)
    tl.store(L + row * LS[0] + head * LS[1], maximum + tl.log2(denom))



# Long-path pipeline adapted from the repository FlashInfer 016 solution.
# The producer-ring and Tensor Core barriers follow the Triton Gluon tutorial
# on warp specialization (triton-lang/triton, MIT).
@gluon.jit
def _load_tiles(Q, K, V, qbuf, kbuf, vbuf, qready, ready, free, first, ks, group,
                blocks, BN: gl.constexpr, STAGES: gl.constexpr):
    mbarrier.expect(qready, Q.block_type.nbytes)
    tma.async_copy_global_to_shared(Q, [first, group, 0, 0], qready, qbuf)
    for i in range(blocks):
        slot = i % STAGES
        if i >= STAGES:
            mbarrier.wait(free.index(slot), (i // STAGES - 1) & 1)
        barrier = ready.index(slot)
        mbarrier.expect(barrier, K.block_type.nbytes + V.block_type.nbytes)
        tma.async_copy_global_to_shared(K, [ks + i * BN, group, 0], barrier, kbuf.index(slot))
        tma.async_copy_global_to_shared(V, [ks + i * BN, group, 0], barrier, vbuf.index(slot))


@gluon.jit
def _matrix_pipeline(qbuf, kbuf, vbuf, qready, ready, free, score_ready,
                     score_read, prob_ready, out_ready, scores, probs, accum, blocks,
                     MT: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr):
    mbarrier.wait(qready, 0)
    for i in range(blocks):
        slot = i % STAGES
        mbarrier.wait(ready.index(slot), (i // STAGES) & 1)
        if i > 0:
            mbarrier.wait(score_read, (i - 1) & 1)
        keys = kbuf.index(slot).reshape((BN, 128)).permute((1, 0))
        tcgen05_mma(qbuf.reshape((MT * 4, 128)), keys, scores, use_acc=False,
                    mbarriers=[score_ready])
        if i > 0:
            prev = i - 1
            mbarrier.wait(prob_ready, prev & 1)
            vals = vbuf.index(prev % STAGES).reshape((BN, 128))
            tcgen05_mma(probs, vals, accum, use_acc=prev > 0,
                        mbarriers=[out_ready, free.index(prev % STAGES)])
    if blocks > 0:
        last = blocks - 1
        mbarrier.wait(prob_ready, last & 1)
        vals = vbuf.index(last % STAGES).reshape((BN, 128))
        tcgen05_mma(probs, vals, accum, use_acc=last > 0, mbarriers=[out_ready])


@gluon.jit
def _softmax_pipeline(O, L, score_ready, score_read, prob_ready, out_ready,
                      scores, probs, accum, first, qe, group, key_len, blocks, scale,
                      MT: gl.constexpr, BN: gl.constexpr, MARGIN: gl.constexpr,
                      SCORE_LAYOUT: gl.constexpr, PROB_LAYOUT: gl.constexpr, OUT_LAYOUT: gl.constexpr):
    r = gl.arange(0, MT * 4, layout=gl.SliceLayout(1, SCORE_LAYOUT))
    c = gl.arange(0, BN, layout=gl.SliceLayout(0, SCORE_LAYOUT))
    limit = first + r // 4 + 1 + key_len - qe
    m = gl.full((MT * 4,), -float("inf"), gl.float32, layout=gl.SliceLayout(1, SCORE_LAYOUT))
    sums = gl.full((MT * 4,), 0.0, gl.float32, layout=gl.SliceLayout(1, SCORE_LAYOUT))
    for i in range(blocks):
        mbarrier.wait(score_ready, i & 1)
        score = scores.load(SCORE_LAYOUT)
        # Only the four softmax warps share this barrier. All TMEM score
        # reads must finish before the matrix warp reuses the buffer.
        gate = gl.inline_asm_elementwise("bar.sync 7, 128; mov.b32 $0, $1;", "=r,r", [i],
                                         dtype=gl.int32, is_pure=False, pack=1)
        mbarrier.arrive(score_read, pred=gate >= 0)
        score = gl.where(i * BN + c[None, :] < limit[:, None],
                         score * scale, -float("inf"))
        tile_max = gl.max(score, 1)
        new_m = gl.where(tile_max > m, tile_max + MARGIN, m)
        base = gl.where(new_m == -float("inf"), 0.0, new_m)
        p = gl.exp2(score - base[:, None])
        alpha = gl.exp2(m - base)
        sums = sums * alpha + gl.sum(p, 1)
        if i > 0:
            mbarrier.wait(out_ready, (i - 1) & 1)
            if gl.max((new_m != m).to(gl.int32), 0) != 0:
                a = accum.load(OUT_LAYOUT)
                correction = gl.convert_layout(alpha, gl.SliceLayout(1, OUT_LAYOUT))
                accum.store(a * correction[:, None])
        m = new_m
        probs.store(gl.convert_layout(p.to(gl.bfloat16), PROB_LAYOUT))
        gate = gl.inline_asm_elementwise("bar.sync 7, 128; mov.b32 $0, $1;", "=r,r", [i],
                                         dtype=gl.int32, is_pure=False, pack=1)
        mbarrier.arrive(prob_ready, pred=gate >= 0)

    den = gl.convert_layout(sums, gl.SliceLayout(1, OUT_LAYOUT))
    if blocks > 0:
        mbarrier.wait(out_ready, (blocks - 1) & 1)
        out = accum.load(OUT_LAYOUT) / den[:, None]
    else:
        zero_or_nan = gl.where(key_len == 0, 0.0, float("nan"))
        out = gl.full((MT * 4, 128), 0.0, gl.float32, OUT_LAYOUT) + zero_or_nan
    x = gl.arange(0, MT * 4, layout=gl.SliceLayout(1, OUT_LAYOUT))
    d = gl.arange(0, 128, layout=gl.SliceLayout(0, OUT_LAYOUT))
    row = first + x // 4
    head = group * 4 + x % 4
    gl.store(O + row[:, None] * 4096 + head[:, None] * 128 + d[None, :], out,
             row[:, None] < qe)
    logz = gl.convert_layout(m + gl.log2(sums), gl.SliceLayout(1, OUT_LAYOUT))
    gl.store(L + row * 32 + head, logz, row < qe)


@gluon.jit
def _gluon_long(Q, K, V, QO, KO, O, L, scale, BATCH: gl.constexpr,
                MT: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr,
                MARGIN: gl.constexpr, LEGACY_TMEM: gl.constexpr):
    tile = gl.program_id(0) // 8
    group = gl.program_id(0) % 8
    req = gl.arange(0, triton.next_power_of_2(BATCH), layout=gl.BlockedLayout([1], [32], [4], [0]))
    if LEGACY_TMEM:
        qb = gl.load(QO + req, req < BATCH, 0)
        qn = gl.load(QO + req + 1, req < BATCH, 0)
    else:
        # Preserve the distributed layout of the offset vector on Gluon builds
        # whose generic load returns an undistributed block type.
        qb, qn = gl.inline_asm_elementwise(
            """
            {
                .reg .pred valid;
                setp.ne.b32 valid, $4, 0;
                mov.b32 $0, 0;
                mov.b32 $1, 0;
                @valid ld.global.b32 $0, [$2];
                @valid ld.global.b32 $1, [$3];
            }
            """,
            "=r,=r,l,l,r", [QO + req, QO + req + 1, (req < BATCH).to(gl.int32)],
            dtype=(gl.int32, gl.int32), is_pure=True, pack=1,
        )
    count = (qn - qb + MT - 1) // MT
    end = gl.associative_scan(count, 0, _add_counts)
    owns = (tile < end) & (end - count <= tile)
    seq = gl.sum(gl.where(owns, req, 0), 0)
    qs = gl.sum(gl.where(owns, qb, 0), 0)
    qe = gl.sum(gl.where(owns, qn, 0), 0)
    # Start at the last tile of each request, where the causal window is largest.
    first = qs + gl.sum(gl.where(owns, end - 1 - tile, 0), 0) * MT
    if first < qe:
        ks = gl.load(KO + seq)
        ke = gl.load(KO + seq + 1)
        key_len = ke - ks
        high = gl.maximum(0, gl.minimum(first + MT, qe) + key_len - qe)
        blocks = (high + BN - 1) // BN
        qbuf = gl.allocate_shared_memory(gl.bfloat16, Q.block_type.shape, Q.layout)
        kbuf = gl.allocate_shared_memory(gl.bfloat16, [STAGES] + K.block_type.shape, K.layout)
        vbuf = gl.allocate_shared_memory(gl.bfloat16, [STAGES] + V.block_type.shape, V.layout)
        qready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        handoff = gl.allocate_shared_memory(gl.int64, [4, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(4):
            mbarrier.init(handoff.index(i), count=1)
        score_ready = handoff.index(0)
        score_read = handoff.index(1)
        prob_ready = handoff.index(2)
        out_ready = handoff.index(3)
        ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        free = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        mbarrier.init(qready, count=1)
        for slot in gl.static_range(STAGES):
            mbarrier.init(ready.index(slot), count=1)
            mbarrier.init(free.index(slot), count=1)
        score_t: gl.constexpr = TensorMemoryLayout([MT * 4, BN], col_stride=1)
        prob_t: gl.constexpr = TensorMemoryLayout([MT * 4, BN], col_stride=1)
        out_t: gl.constexpr = TensorMemoryLayout([MT * 4, 128], col_stride=1)
        scores = allocate_tensor_memory(gl.float32, [MT * 4, BN], score_t)
        probs = allocate_tensor_memory(gl.bfloat16, [MT * 4, BN], prob_t)
        accum = allocate_tensor_memory(gl.float32, [MT * 4, 128], out_t)
        if LEGACY_TMEM:
            score_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (MT * 4, BN), score_t, 4)
            prob_layout: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, (MT * 4, BN), prob_t, 4)
            out_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (MT * 4, 128), out_t, 4)
        else:
            score_layout: gl.constexpr = scores.get_reg_layout(4)
            prob_layout: gl.constexpr = probs.get_reg_layout(4)
            out_layout: gl.constexpr = accum.get_reg_layout(4)
        gl.warp_specialize([
            (_softmax_pipeline, (O, L, score_ready, score_read, prob_ready, out_ready,
                                  scores, probs, accum, first, qe, group, key_len,
                                  blocks, scale, MT, BN, MARGIN, score_layout, prob_layout, out_layout)),
            (_load_tiles, (Q, K, V, qbuf, kbuf, vbuf, qready, ready, free, first, ks, group,
                           blocks, BN, STAGES)),
            (_matrix_pipeline, (qbuf, kbuf, vbuf, qready, ready, free, score_ready, score_read,
                                prob_ready, out_ready, scores, probs, accum, blocks, MT, BN, STAGES)),
        ], [1, 1], [24, 24])
        mbarrier.invalidate(qready)
        for i in gl.static_range(4):
            mbarrier.invalidate(handoff.index(i))
        for slot in gl.static_range(STAGES):
            mbarrier.invalidate(ready.index(slot))
            mbarrier.invalidate(free.index(slot))


@gluon.jit
def _add_counts(a, b):
    return a + b


def _launch_gluon(q, k, v, qo, ko, out, lse, scale, batch, mt, bn, stages, margin):
    nq = q.shape[0]
    q_desc = GluonTensorDescriptor(q, [nq, 8, 4, 128], [4096, 512, 128, 1],
                                   [mt, 1, 4, 128], gl.NVMMASharedLayout(128, 16, rank=4))
    k_desc = GluonTensorDescriptor(k, list(k.shape), list(k.stride()), [bn, 1, 128],
                                   gl.NVMMASharedLayout(128, 16, rank=3))
    v_desc = GluonTensorDescriptor(v, list(v.shape), list(v.stride()), [bn, 1, 128],
                                   gl.NVMMASharedLayout(128, 16, rank=3))
    tiles = (nq + (mt - 1) * batch) // mt
    _gluon_long[(tiles * 8,)](q_desc, k_desc, v_desc, qo, ko, out, lse, scale,
                              batch, mt, bn, stages, margin, get_tmem_reg_layout is not None,
                              num_warps=4)



def run(q, k, v, qo_indptr, kv_indptr, sm_scale, output, lse):
    """Write BF16 attention output and FP32 base-2 log-sum-exp."""
    nq, nk = q.shape[0], k.shape[0]
    if nq == 0:
        return

    batch = qo_indptr.numel() - 1
    strides = (q.stride(), k.stride(), v.stride(), output.stride(), lse.stride())
    if nq == nk == batch == 1:
        _single[(32,)](q, k, v, output, lse, sm_scale, *strides, num_warps=1)
        return

    if batch == 1 and nq <= 35 and 0 < nk <= 8:
        _simt_short[(nq, 32)](
            q, k, v, output, lse, sm_scale, *strides, nq, nk,
            triton.next_power_of_2(nk), num_warps=1,
        )
        return

    if nq >= 4096 and nk > 0 and all(
        tensor.is_contiguous() for tensor in (q, k, v, qo_indptr, kv_indptr, output, lse)
    ):
        _launch_gluon(
            q, k, v, qo_indptr, kv_indptr, output, lse, sm_scale * 1.4426950408889634,
            batch, 32, 128, 2, 8.0,
        )
        return

    indptr_strides = (qo_indptr.stride(0), kv_indptr.stride(0))
    args = (q, k, v, qo_indptr, kv_indptr, output, lse, sm_scale, *strides, indptr_strides)
    b = triton.next_power_of_2(batch)
    group = 4
    if nq <= 128 and 0 < nk <= 128:
        m, n = 16, max(16, triton.next_power_of_2(nk))
        tokens = m // group
        upper = (nq + (tokens - 1) * batch) // tokens
        _short[(upper, 32 // group)](
            *args, nq, nk, batch, b, m, n, group, nq > 64 or batch > 1,
            num_warps=4, num_stages=1,
        )
        return

    m, full_n, diag_n = 64, 64, 32
    tokens = m // group
    upper = (nq + (tokens - 1) * batch) // tokens
    _ragged[(upper * (32 // group),)](
        *args, batch, b, m, full_n, diag_n, group, num_warps=4, num_stages=2,
    )
