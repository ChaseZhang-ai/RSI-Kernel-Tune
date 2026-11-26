"""Direct-launch reference candidate for the paged 4:1 GQA evaluator."""
import math

import triton
import triton.language as tl


@triton.jit
def _initialize(O, L, Q_COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + i, 0, i < Q_COUNT * 4096)
    j = tl.program_id(0) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
    tl.store(L + j, float("-inf"), j < Q_COUNT * 32)


@triton.jit
def _active_tile(QO, KO, tile, REQUESTS: tl.constexpr, R: tl.constexpr, TQ: tl.constexpr):
    if REQUESTS == 1:
        qs = tl.load(QO)
        qe = tl.load(QO + 1)
        ks = tl.load(KO)
        ke = tl.load(KO + 1)
        active = tl.minimum(qe - qs, ke - ks)
        query = qe - active + tile * TQ
    else:
        r = tl.arange(0, R)
        qs_vec = tl.load(QO + r, r < REQUESTS, 0)
        qe_vec = tl.load(QO + r + 1, r < REQUESTS, 0)
        ks_vec = tl.load(KO + r, r < REQUESTS, 0)
        ke_vec = tl.load(KO + r + 1, r < REQUESTS, 0)
        active_vec = tl.minimum(qe_vec - qs_vec, ke_vec - ks_vec)
        tiles = tl.cdiv(active_vec, TQ)
        end = tl.cumsum(tiles)
        req = tl.sum((end <= tile).to(tl.int32), 0)
        start = tl.sum(tl.where(r == req, end - tiles, 0), 0)
        qe = tl.load(QO + req + 1, req < REQUESTS, 0)
        ks = tl.load(KO + req, req < REQUESTS, 0)
        ke = tl.load(KO + req + 1, req < REQUESTS, 0)
        active = tl.sum(tl.where(r == req, active_vec, 0), 0)
        query = qe - active + (tile - start) * TQ
    return query, qe, ks, ke


@triton.jit
def _attention(Q, K, V, Pages, O, L, query, qe, ks, ke, h, scale,
               M: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, M)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    row = query + m // 4
    head = h * 4 + m % 4
    cutoff = row + ke - qe + 1
    q = tl.load(Q + row[:, None] * 4096 + head[:, None] * 128 + d[None, :],
                row[:, None] < qe, 0)
    max_logit = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    accum = tl.full((M, 128), 0, tl.float32)
    limit = tl.minimum(ke, query + M // 4 + ke - qe)
    for key in range(ks, limit, N):
        index = key + n
        page = tl.load(Pages + index, index < limit, 0)
        kv = page[:, None] * 1024 + h * 128 + d[None, :]
        k = tl.load(K + kv, index[:, None] < limit, 0)
        score = tl.dot(q, tl.trans(k)) * scale
        score = tl.where((index[None, :] < cutoff[:, None]) & (index[None, :] < limit),
                         score, -float("inf"))
        next_max = tl.maximum(max_logit, tl.max(score, 1))
        p = tl.exp2(score - next_max[:, None])
        correction = tl.exp2(max_logit - next_max)
        denom = denom * correction + tl.sum(p, 1)
        v = tl.load(V + kv, index[:, None] < limit, 0)
        accum = tl.dot(p.to(tl.bfloat16), v, accum * correction[:, None])
        max_logit = next_max
    output = accum / denom[:, None]
    lse = max_logit + tl.log2(denom)
    tl.store(O + row[:, None] * 4096 + head[:, None] * 128 + d[None, :],
             output, row[:, None] < qe)
    tl.store(L + row * 32 + head, lse, row < qe)


@triton.jit
def _compute(Q, K, V, QO, KO, Pages, O, L, scale,
             REQUESTS: tl.constexpr, R: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    query, qe, ks, ke = _active_tile(QO, KO, tl.program_id(0), REQUESTS, R, M // 4)
    if (query < qe) & (ks < ke):
        _attention(Q, K, V, Pages, O, L, query, qe, ks, ke, tl.program_id(1), scale, M, N)


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, output, lse):
    total_q = q.shape[0]
    requests = qo_indptr.numel() - 1
    keys = kv_indices.numel()
    m, n = (64, 64) if keys > 512 else (16, 32)
    if total_q:
        _initialize[(triton.cdiv(total_q * 4096, 4096),)](output, lse, total_q, 4096)
    if total_q and keys:
        tiles = triton.cdiv(min(total_q, keys), m // 4) + requests - 1
        _compute[(tiles, 8)](
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse,
            float(sm_scale) * math.log2(math.e), requests, triton.next_power_of_2(requests),
            m, n, num_warps=4)
