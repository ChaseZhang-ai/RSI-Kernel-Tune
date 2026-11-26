# SPDX-License-Identifier: Apache-2.0
"""Fused destination-passing ragged causal GQA for FlashInfer-Bench 017."""

import triton
import triton.language as tl


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
           B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    token = tile * (M // GROUP)
    if BATCH == 1:
        qs, qe, ks, ke = 0, NQ, 0, NK
    else:
        seq = tl.arange(0, B)
        ends = tl.load(QP + (seq + 1) * PS[0], seq < BATCH, NQ)
        # A global tile can cross sequence boundaries. Use the first token's
        # sequence and launch one token per tile for ragged short inputs.
        b = tl.sum((ends <= token).to(tl.int32), 0)
        qs = tl.load(QP + b * PS[0])
        qe = tl.load(QP + (b + 1) * PS[0])
        ks = tl.load(KP + b * PS[1])
        ke = tl.load(KP + (b + 1) * PS[1])
    rows = tl.arange(0, M)
    if BATCH == 1:
        qt = token + rows // GROUP
        active = qt < qe
    else:
        qt = tl.full((M,), 0, tl.int32) + token
        active = rows < GROUP
    qh = group * GROUP + rows % GROUP
    h = group // (4 // GROUP)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    kt = ks + n
    q = tl.load(Q + qt[:, None] * QS[0] + qh[:, None] * QS[1] + d[None, :] * QS[2],
                active[:, None], 0)
    k = tl.load(K + kt[:, None] * KS[0] + h * KS[1] + d[None, :] * KS[2],
                (kt < ke)[:, None], 0)
    v = tl.load(V + kt[:, None] * VS[0] + h * VS[1] + d[None, :] * VS[2],
                (kt < ke)[:, None], 0)
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
            FULL_N: tl.constexpr, DIAG_N: tl.constexpr, GROUP: tl.constexpr,
            PERSIST: tl.constexpr, REVERSE: tl.constexpr):
    seq = tl.arange(0, B)
    ends = tl.load(QP + (seq + 1) * PS[0], seq < BATCH, 0)
    starts = tl.load(QP + seq * PS[0], seq < BATCH, 0)
    tiles = tl.cdiv(ends - starts, M // GROUP)
    cumulative = tl.cumsum(tiles)
    num_tiles = tl.sum(tiles, 0)
    groups: tl.constexpr = 32 // GROUP
    if PERSIST:
        first = tl.program_id(0)
        stop = num_tiles * groups
        stride = tl.num_programs(0)
    else:
        first = tl.program_id(0)
        stop = tl.minimum(first + 1, num_tiles * groups)
        stride = 1
    scale2 = scale * 1.4426950408889634
    for work in range(first, stop, stride):
        tile = work // groups
        group = work % groups
        b = tl.sum(((cumulative <= tile) & (seq < BATCH)).to(tl.int32), 0)
        local = tile - tl.sum(tl.where(seq == b, cumulative - tiles, 0), 0)
        if REVERSE:
            local = tl.sum(tl.where(seq == b, tiles, 0), 0) - 1 - local
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
        q = q.to(tl.bfloat16)
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


def run(q, k, v, qo_indptr, kv_indptr, sm_scale, output, lse):
    nq, nk = q.shape[0], k.shape[0]
    if nq == 0:
        return
    batch = qo_indptr.numel() - 1
    strides = (q.stride(), k.stride(), v.stride(), output.stride(), lse.stride())
    if nq == nk == batch == 1:
        _single[(32,)](q, k, v, output, lse, sm_scale, *strides, num_warps=1)
        return
    ps = (qo_indptr.stride(0), kv_indptr.stride(0))
    args = (q, k, v, qo_indptr, kv_indptr, output, lse, sm_scale, *strides, ps)
    b = triton.next_power_of_2(batch)
    if nq <= 96 and nk <= 128 and batch == 1:
        m, n, group = 16, max(16, triton.next_power_of_2(nk)), 4
        _short[(triton.cdiv(nq, m // group), 32 // group)](
            *args, nq, nk, batch, b, m, n, group, num_warps=4, num_stages=1,
        )
        return
    m, full_n, diag_n, group = 64, 64, 32, 4
    upper = (triton.cdiv(nq, m // group) + batch - 1) * (32 // group)
    _ragged[(upper,)](*args, batch, b, m, full_n, diag_n, group,
                      False, False, num_warps=4, num_stages=2)
