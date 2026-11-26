# SPDX-License-Identifier: Apache-2.0
# Packed GQA and the one-key reduction follow Tencent Hyra submission 25891.
# Short/long dispatch is informed by CompoundingKernel Explorer's FlashInfer 016.
import triton
import triton.language as tl


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
            MT: tl.constexpr, BN: tl.constexpr, SCHED: tl.constexpr,
            REVERSE: tl.constexpr, SPLIT_MASK: tl.constexpr, SEARCH_STEPS: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    if REVERSE:
        tile = tl.num_programs(0) - 1 - tile
    if BATCH == 1:
        seq = 0
        qs = tl.load(QO)
        qe = tl.load(QO + 1)
        first = qs + tile * MT
    elif SCHED == 0:
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
    else:
        # Each request starts in the bucket containing its first query. The
        # request index provides an extra bucket at every ragged boundary.
        lo = 0
        hi = BATCH
        for step in tl.static_range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            boundary = tl.load(QO + mid) // MT + mid
            lo, hi = tl.where(boundary <= tile, mid, lo), tl.where(boundary <= tile, hi, mid)
        seq = lo
        qs = tl.load(QO + seq)
        qe = tl.load(QO + seq + 1)
        first = (tile - seq) * MT

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


def run(q, k, v, qo_indptr, kv_indptr, sm_scale, output, lse):
    nq, nk, batch = q.shape[0], k.shape[0], qo_indptr.shape[0] - 1
    if nq == 0:
        return
    scale = sm_scale * 1.4426950408889634
    strides = (*q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride())
    if nq == nk == batch == 1:
        _one_key[(32,)](q, k, v, output, lse, scale, strides, num_warps=1)
    elif batch == 1 and 0 < nk <= 8:
        _tiny[(nq, 32)](
            q, k, v, output, lse, scale, strides, nq, nk,
            triton.next_power_of_2(nk), num_warps=1,
        )
    elif batch == 1 and 0 < nk <= 64:
        _single_sequence[(triton.cdiv(nq, 16), 32)](
            q, k, v, output, lse, scale, strides, nq, nk, 16,
            max(16, triton.next_power_of_2(nk)), num_warps=4, num_stages=1,
        )
    else:
        mt, bn, warps, stages = (
            (16, 64, 4, 3) if nq >= 4096 else (8, 64, 4, 3)
        )
        sched = 0
        tiles = (nq + (mt - 1) * batch) // mt
        _packed[(tiles, 4)](
            q, k, v, qo_indptr, kv_indptr, output, lse, scale, strides,
            batch, mt, bn, sched, nq >= 4096, nq >= 512, (batch - 1).bit_length(),
            num_warps=warps, num_stages=stages,
        )
