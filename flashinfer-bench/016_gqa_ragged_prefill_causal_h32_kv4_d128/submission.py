# SPDX-License-Identifier: Apache-2.0
# Packed GQA and the one-key reduction follow Tencent Hyra submission 25891.
# Short/long dispatch is informed by CompoundingKernel Explorer's FlashInfer 016.
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
def _one_key(Q, K, V, O, L, scale, S: tl.constexpr):
    head = tl.program_id(0)
    group = head // 8
    d = tl.arange(0, 128)
    q = tl.load(Q + head * S[1] + d * S[2]).to(tl.float32)
    k = tl.load(K + group * S[4] + d * S[5]).to(tl.float32)
    v = tl.load(V + group * S[7] + d * S[8])
    tl.store(O + head * S[10] + d * S[11], v)
    tl.store(L + head * S[13], tl.sum(q * k, 0) * scale)


@triton.jit
def _tiny(Q, K, V, O, L, scale, S: tl.constexpr, NQ: tl.constexpr,
          NK: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    group = head // 8
    key = tl.arange(0, BN)
    d = tl.arange(0, 128)
    valid = (key < NK) & (key <= token + NK - NQ)
    q = tl.load(Q + token * S[0] + head * S[1] + d * S[2]).to(tl.float32)
    k = tl.load(K + key[:, None] * S[3] + group * S[4] + d[None, :] * S[5],
                valid[:, None], 0).to(tl.float32)
    v = tl.load(V + key[:, None] * S[6] + group * S[7] + d[None, :] * S[8],
                valid[:, None], 0).to(tl.float32)
    score = tl.where(valid, tl.sum(k * q[None, :], 1) * scale, -float("inf"))
    active = token + NK >= NQ
    m = tl.where(active, tl.max(score, 0), 0.0)
    p = tl.exp2(score - m)
    z = tl.sum(p, 0)
    out = tl.sum(v * p[:, None], 0) / z
    tl.store(O + token * S[9] + head * S[10] + d * S[11], out)
    tl.store(L + token * S[12] + head * S[13], m + tl.log2(z))


@triton.jit
def _single_sequence(Q, K, V, O, L, scale, S: tl.constexpr, NQ: tl.constexpr,
                     NK: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    head = tl.program_id(1)
    group = head // 8
    col = tl.arange(0, N)
    d = tl.arange(0, 128)
    q = tl.load(Q + row[:, None] * S[0] + head * S[1] + d[None, :] * S[2],
                row[:, None] < NQ, 0)
    k = tl.load(K + col[:, None] * S[3] + group * S[4] + d[None, :] * S[5],
                col[:, None] < NK, 0)
    v = tl.load(V + col[:, None] * S[6] + group * S[7] + d[None, :] * S[8],
                col[:, None] < NK, 0)
    x = tl.dot(q, tl.trans(k)) * scale
    x = tl.where((col[None, :] < NK) & (col[None, :] <= row[:, None] + NK - NQ),
                 x, -float("inf"))
    m = tl.where(row + NK >= NQ, tl.max(x, 1), 0.0)
    p = tl.exp2(x - m[:, None])
    z = tl.sum(p, 1)
    out = tl.dot(p.to(tl.bfloat16), v) / z[:, None]
    tl.store(O + row[:, None] * S[9] + head * S[10] + d[None, :] * S[11],
             out, row[:, None] < NQ)
    tl.store(L + row * S[12] + head * S[13], m + tl.log2(z), row < NQ)


@triton.jit
def _attention_step(Q, K, V, ks, col, group, d, limit, key_len, scale, m, z, a,
                    S: tl.constexpr, MASKED: tl.constexpr):
    kth = ks + col
    if MASKED:
        key_ok = col < key_len
        k = tl.load(K + kth[:, None] * S[3] + group * S[4] + d[None, :] * S[5],
                    key_ok[:, None], 0)
        v = tl.load(V + kth[:, None] * S[6] + group * S[7] + d[None, :] * S[8],
                    key_ok[:, None], 0)
    else:
        k = tl.load(K + kth[:, None] * S[3] + group * S[4] + d[None, :] * S[5])
        v = tl.load(V + kth[:, None] * S[6] + group * S[7] + d[None, :] * S[8])
    score = tl.dot(Q, tl.trans(k)) * scale
    if MASKED:
        visible = (col[None, :] < limit[:, None]) & (col[None, :] < key_len)
        score = tl.where(visible, score, -float("inf"))
    new_m = tl.maximum(m, tl.max(score, 1))
    base = tl.where(new_m == -float("inf"), 0.0, new_m)
    p = tl.exp2(score - base[:, None])
    correction = tl.exp2(m - base)
    z = z * correction + tl.sum(p, 1)
    a = tl.dot(p.to(tl.bfloat16), v, a * correction[:, None])
    return new_m, z, a


@triton.jit
def _packed(Q, K, V, QO, KO, O, L, scale, S: tl.constexpr, BATCH: tl.constexpr,
            MT: tl.constexpr, BN: tl.constexpr,
            REVERSE: tl.constexpr, SPLIT_MASK: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    if REVERSE:
        tile = tl.num_programs(0) - 1 - tile
    if BATCH == 1:
        seq = 0
        qs = tl.load(QO)
        qe = tl.load(QO + 1)
        first = qs + tile * MT
    else:
        request = tl.arange(0, triton.next_power_of_2(BATCH))
        qb = tl.load(QO + request, request < BATCH, 0)
        qn = tl.load(QO + request + 1, request < BATCH, 0)
        count = tl.cdiv(qn - qb, MT)
        end = tl.cumsum(count)
        owns = (tile < end) & (end - count <= tile)
        seq = tl.sum(tl.where(owns, request, 0), 0)
        qs = tl.sum(tl.where(owns, qb, 0), 0)
        qe = tl.sum(tl.where(owns, qn, 0), 0)
        first = qs + (tile - tl.sum(tl.where(owns, end - count, 0), 0)) * MT

    if first < qe:
        ks = tl.load(KO + seq)
        ke = tl.load(KO + seq + 1)
        key_len = ke - ks
        x = tl.arange(0, MT * 8)
        row = first + x // 8
        head = group * 8 + x % 8
        d = tl.arange(0, 128)
        col = tl.arange(0, BN)
        active_row = (row >= qs) & (row < qe)
        limit = row + 1 + key_len - qe
        high = tl.maximum(0, tl.minimum(first + MT, qe) + key_len - qe)
        q = tl.load(Q + row[:, None] * S[0] + head[:, None] * S[1] + d[None, :] * S[2],
                    active_row[:, None] & (limit[:, None] > 0), 0)
        m = tl.full((MT * 8,), -float("inf"), tl.float32)
        z = tl.zeros((MT * 8,), tl.float32)
        a = tl.zeros((MT * 8, 128), tl.float32)
        split = 0
        if SPLIT_MASK:
            split = tl.maximum(0, first + 1 + key_len - qe) // BN * BN
            for start in range(0, split, BN):
                m, z, a = _attention_step(
                    q, K, V, ks, start + col, group, d, limit, key_len, scale, m, z, a, S, False,
                )
        for start in range(split, high, BN):
            m, z, a = _attention_step(
                q, K, V, ks, start + col, group, d, limit, key_len, scale, m, z, a, S, True,
            )
        # torch.softmax returns NaN for an all-masked, nonempty KV slice.
        # The reference leaves the zero-initialized output for empty KV slices.
        out = tl.where(key_len == 0, 0.0, a / z[:, None])
        tl.store(O + row[:, None] * S[9] + head[:, None] * S[10] + d[None, :] * S[11],
                 out, active_row[:, None])
        tl.store(L + row * S[12] + head * S[13], m + tl.log2(z), active_row)


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
        tcgen05_mma(qbuf.reshape((MT * 8, 128)), keys, scores, use_acc=False,
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
    r = gl.arange(0, MT * 8, layout=gl.SliceLayout(1, SCORE_LAYOUT))
    c = gl.arange(0, BN, layout=gl.SliceLayout(0, SCORE_LAYOUT))
    limit = first + r // 8 + 1 + key_len - qe
    m = gl.full((MT * 8,), -float("inf"), gl.float32, layout=gl.SliceLayout(1, SCORE_LAYOUT))
    sums = gl.full((MT * 8,), 0.0, gl.float32, layout=gl.SliceLayout(1, SCORE_LAYOUT))
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
        out = gl.full((MT * 8, 128), 0.0, gl.float32, OUT_LAYOUT) + zero_or_nan
    x = gl.arange(0, MT * 8, layout=gl.SliceLayout(1, OUT_LAYOUT))
    d = gl.arange(0, 128, layout=gl.SliceLayout(0, OUT_LAYOUT))
    row = first + x // 8
    head = group * 8 + x % 8
    gl.store(O + row[:, None] * 4096 + head[:, None] * 128 + d[None, :], out,
             row[:, None] < qe)
    logz = gl.convert_layout(m + gl.log2(sums), gl.SliceLayout(1, OUT_LAYOUT))
    gl.store(L + row * 32 + head, logz, row < qe)


@gluon.jit
def _gluon_long(Q, K, V, QO, KO, O, L, scale, BATCH: gl.constexpr,
                MT: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr,
                BUCKET_REVERSE: gl.constexpr, MARGIN: gl.constexpr, LEGACY_TMEM: gl.constexpr):
    tile = gl.program_id(0) // 4
    group = gl.program_id(0) % 4
    if BUCKET_REVERSE:
        tile = gl.num_programs(0) // 4 - 1 - tile
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
        score_t: gl.constexpr = TensorMemoryLayout([MT * 8, BN], col_stride=1)
        prob_t: gl.constexpr = TensorMemoryLayout([MT * 8, BN], col_stride=1)
        out_t: gl.constexpr = TensorMemoryLayout([MT * 8, 128], col_stride=1)
        scores = allocate_tensor_memory(gl.float32, [MT * 8, BN], score_t)
        probs = allocate_tensor_memory(gl.bfloat16, [MT * 8, BN], prob_t)
        accum = allocate_tensor_memory(gl.float32, [MT * 8, 128], out_t)
        if LEGACY_TMEM:
            score_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (MT * 8, BN), score_t, 4)
            prob_layout: gl.constexpr = get_tmem_reg_layout(gl.bfloat16, (MT * 8, BN), prob_t, 4)
            out_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (MT * 8, 128), out_t, 4)
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


def _launch_gluon(q, k, v, qo, ko, out, lse, scale, batch, mt, bn, stages,
                  bucket_reverse, margin):
    nq = q.shape[0]
    q_desc = GluonTensorDescriptor(q, [nq, 4, 8, 128], [4096, 1024, 128, 1],
                                   [mt, 1, 8, 128], gl.NVMMASharedLayout(128, 16, rank=4))
    k_desc = GluonTensorDescriptor(k, list(k.shape), list(k.stride()), [bn, 1, 128],
                                   gl.NVMMASharedLayout(128, 16, rank=3))
    v_desc = GluonTensorDescriptor(v, list(v.shape), list(v.stride()), [bn, 1, 128],
                                   gl.NVMMASharedLayout(128, 16, rank=3))
    tiles = (nq + (mt - 1) * batch) // mt
    _gluon_long[(tiles * 4,)](q_desc, k_desc, v_desc, qo, ko, out, lse, scale,
                              batch, mt, bn, stages, bucket_reverse, margin, get_tmem_reg_layout is not None,
                              num_warps=4)


def run(q, k, v, qo_indptr, kv_indptr, sm_scale, output, lse):
    """Write bottom-right causal GQA and base-2 log-sum-exp to supplied tensors."""
    nq, nk, batch = q.shape[0], k.shape[0], qo_indptr.shape[0] - 1
    if nq == 0:
        return
    scale = sm_scale * 1.4426950408889634
    strides = (*q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride())
    if nq == nk == batch == 1:
        _one_key[(32,)](q, k, v, output, lse, scale, strides, num_warps=1)
    elif batch == 1 and nq <= 64 and 0 < nk <= 8:
        _tiny[(nq, 32)](
            q, k, v, output, lse, scale, strides, nq, nk,
            triton.next_power_of_2(nk), num_warps=1,
        )
    elif batch == 1 and nq <= 64 and 0 < nk <= 64:
        _single_sequence[(triton.cdiv(nq, 16), 32)](
            q, k, v, output, lse, scale, strides, nq, nk, 16,
            max(16, triton.next_power_of_2(nk)), num_warps=4, num_stages=1,
        )
    elif nq >= 512 and nk > 0 and all(x.is_contiguous() for x in (q, k, v, output, lse)):
        bn, reverse_grid = (128, False) if nq >= 4096 else (64, True)
        _launch_gluon(q, k, v, qo_indptr, kv_indptr, output, lse, scale, batch,
                      16, bn, 2, reverse_grid, 8.0)
    else:
        if nq < 512:
            mt, stages = 2, 1
        else:
            mt, stages = (16 if nq >= 4096 else 8), 3
        bn = 64
        tiles = (nq + (mt - 1) * batch) // mt
        _packed[(tiles, 4)](
            q, k, v, qo_indptr, kv_indptr, output, lse, scale, strides,
            batch, mt, bn, nq >= 4096, nq >= 512,
            num_warps=4, num_stages=stages,
        )
