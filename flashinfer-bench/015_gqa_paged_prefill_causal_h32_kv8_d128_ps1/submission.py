"""Destination-passing paged causal attention with four query heads per KV head.

The separate short-window and empty-prefix work follows the scheduling ideas in
QianQu's CompoundingKernel-Explorer, FlashInfer 015. The grouped tensor-core
tiles use online softmax as in RSI-Kernel-Tune's H100 causal GQA schedule.
This is an independent Triton implementation.

https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/015_gqa_paged_prefill_causal_h32_kv8_d128_ps1
https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/attention/attn_fwd_gqa16/h100/gen2/kernel.py
"""
import math

import torch
import triton
import triton.language as tl


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
def _online(Q, K, V, Pages, query, qe, first, last, ke, h, scale,
            M: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, M)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    row = query + m // 4
    head = h * 4 + m % 4
    cutoff = row + ke - qe + 1
    q = tl.load(Q + row[:, None] * 4096 + head[:, None] * 128 + d[None, :],
                row[:, None] < qe, 0)
    max_logit = tl.full((M,), -1e30, tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    accum = tl.full((M, 128), 0, tl.float32)
    limit = tl.minimum(last, query + M // 4 + ke - qe)
    for key in range(first, limit, N):
        index = key + n
        page = tl.load(Pages + index, index < limit, 0)
        kv = page[:, None].to(tl.int64) * 1024 + h * 128 + d[None, :]
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
    return accum, max_logit, denom


@triton.jit
def _store(O, L, query, qe, h, accum, peak, denom, M: tl.constexpr):
    m = tl.arange(0, M)
    d = tl.arange(0, 128)
    row = query + m // 4
    head = h * 4 + m % 4
    tl.store(O + row[:, None] * 4096 + head[:, None] * 128 + d[None, :],
             accum / denom[:, None], row[:, None] < qe)
    tl.store(L + row * 32 + head, peak + tl.log2(denom), row < qe)


@triton.jit
def _zeros(QO, KO, O, L, tile, q_count,
           REQUESTS: tl.constexpr, R: tl.constexpr, ZQ: tl.constexpr):
    row = tile * ZQ + tl.arange(0, ZQ)
    if REQUESTS == 1:
        end = tl.load(QO + 1)
        keys = tl.load(KO + 1) - tl.load(KO)
        empty = row < end - keys
    else:
        r = tl.arange(0, R)
        qs = tl.load(QO + r, r < REQUESTS, q_count)
        qe = tl.load(QO + r + 1, r < REQUESTS, 0)
        keys = tl.load(KO + r + 1, r < REQUESTS, 0) - tl.load(KO + r, r < REQUESTS, 0)
        start = tl.maximum(qs, qe - keys)
        # Empty-prefix boundaries are monotonic across requests.
        active_start = tl.max(tl.where(row[:, None] >= qs[None, :], start[None, :], 0), 1)
        empty = row < active_start
    empty = empty & (row < q_count)
    d = tl.arange(0, 4096)
    h = tl.arange(0, 32)
    tl.store(O + row[:, None] * 4096 + d[None, :], 0, empty[:, None], cache_modifier=".cs")
    tl.store(L + row[:, None] * 32 + h[None, :], -float("inf"), empty[:, None])


@triton.jit
def _short(Q, K, V, Pages, O, L, row, qe, ks, ke, head_group, scale,
           HEADS: tl.constexpr, SHORT: tl.constexpr):
    h = head_group * HEADS + tl.arange(0, HEADS)
    kh = head_group // (4 // HEADS)
    d = tl.arange(0, 128)
    n = tl.arange(0, SHORT)
    cutoff = row + ke - qe + 1
    idx = ks + n
    page = tl.load(Pages + idx, idx < cutoff, 0)
    addr = page[:, None].to(tl.int64) * 1024 + kh * 128 + d[None, :]
    q = tl.load(Q + row * 4096 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    k = tl.load(K + addr, idx[:, None] < cutoff, 0).to(tl.float32)
    score = tl.sum(q[:, None, :] * k[None, :, :], 2) * scale
    score = tl.where(idx[None, :] < cutoff, score, -float("inf"))
    peak = tl.max(score, 1)
    p = tl.exp2(score - peak[:, None])
    denom = tl.sum(p, 1)
    v = tl.load(V + addr, idx[:, None] < cutoff, 0).to(tl.float32)
    output = tl.sum(p[:, :, None] * v[None, :, :], 1) / denom[:, None]
    tl.store(O + row * 4096 + h[:, None] * 128 + d[None, :], output)
    tl.store(L + row * 32 + h, peak + tl.log2(denom))


@triton.jit
def _fused(Q, K, V, QO, KO, Pages, O, L, scale, active_ctas, q_count,
           REQUESTS: tl.constexpr, R: tl.constexpr, N: tl.constexpr, ZQ: tl.constexpr,
           SHORT: tl.constexpr):
    pid = tl.program_id(0)
    if pid < active_ctas:
        if SHORT:
            query, qe, ks, ke = _active_tile(QO, KO, pid // 32, REQUESTS, R, 1)
            if (query < qe) & (ks < ke):
                _short(Q, K, V, Pages, O, L, query, qe, ks, ke, pid % 32, scale, 1, SHORT)
        else:
            query, qe, ks, ke = _active_tile(QO, KO, pid // 8, REQUESTS, R, 4)
            if (query < qe) & (ks < ke):
                group = pid % 8
                acc, peak, denom = _online(Q, K, V, Pages, query, qe, ks, ke, ke, group, scale, 16, N)
                _store(O, L, query, qe, group, acc, peak, denom, 16)
    else:
        _zeros(QO, KO, O, L, pid - active_ctas, q_count, REQUESTS, R, ZQ)


@triton.jit
def _long_tasks(QO, KO, REQUESTS: tl.constexpr, R: tl.constexpr, SPLIT: tl.constexpr):
    r = tl.arange(0, R)
    qs = tl.load(QO + r, r < REQUESTS, 0)
    qe = tl.load(QO + r + 1, r < REQUESTS, 0)
    ks = tl.load(KO + r, r < REQUESTS, 0)
    ke = tl.load(KO + r + 1, r < REQUESTS, 0)
    active = tl.minimum(qe - qs, ke - ks)
    split = (active <= 16) & (ke - ks >= 512)
    parts = tl.where(split, tl.cdiv(ke - ks, SPLIT), 1)
    tasks = tl.cdiv(active, 4) * parts
    return r, active, parts, tasks


@triton.jit
def _long(Q, K, V, QO, KO, Pages, O, L, Partials, scale, active_ctas, q_count,
          REQUESTS: tl.constexpr, R: tl.constexpr, ZQ: tl.constexpr,
          SPLIT: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    if pid < active_ctas:
        tile = pid // 8
        h = pid % 8
        r, active_vec, part_vec, tasks = _long_tasks(QO, KO, REQUESTS, R, SPLIT)
        ends = tl.cumsum(tasks)
        req = tl.sum((ends <= tile).to(tl.int32), 0)
        if req < REQUESTS:
            relative = tile - tl.sum(tl.where(r == req, ends - tasks, 0), 0)
            parts = tl.sum(tl.where(r == req, part_vec, 0), 0)
            active = tl.sum(tl.where(r == req, active_vec, 0), 0)
            qe = tl.load(QO + req + 1)
            ks = tl.load(KO + req)
            ke = tl.load(KO + req + 1)
            local_q = relative // parts * 4
            query = qe - active + local_q
            if parts > 1:
                part = relative % parts
                first = ks + part * SPLIT
                last = tl.minimum(ke, first + SPLIT)
                acc, peak, denom = _online(Q, K, V, Pages, query, qe,
                                            first, last, ke, h, scale, 16, N)
                m = tl.arange(0, 16)
                d = tl.arange(0, 128)
                slot = ((part * REQUESTS + req) * 16 + local_q + m // 4) * 32 + h * 4 + m % 4
                addr = Partials + slot * 132
                valid = local_q + m // 4 < active
                tl.store(addr[:, None] + d[None, :], acc, valid[:, None])
                tl.store(addr + 128, peak, valid)
                tl.store(addr + 129, denom, valid)
            else:
                acc, peak, denom = _online(Q, K, V, Pages, query, qe,
                                            ks, ke, ke, h, scale, 16, 32)
                _store(O, L, query, qe, h, acc, peak, denom, 16)
    else:
        _zeros(QO, KO, O, L, pid - active_ctas, q_count, REQUESTS, R, ZQ)


@triton.jit
def _join(QO, KO, Partials, O, L, REQUESTS: tl.constexpr,
          SPLIT: tl.constexpr, PARTS: tl.constexpr):
    req = tl.program_id(0)
    qs = tl.load(QO + req)
    qe = tl.load(QO + req + 1)
    ks = tl.load(KO + req)
    ke = tl.load(KO + req + 1)
    keys = ke - ks
    active = tl.minimum(qe - qs, keys)
    if (active > 0) & (active <= 16) & (keys >= 512):
        head = tl.program_id(1)
        part = tl.arange(0, PARTS)
        d = tl.arange(0, 128)
        parts = tl.cdiv(keys, SPLIT)
        for local_q in range(active):
            slot = ((part * REQUESTS + req) * 16 + local_q) * 32 + head
            addr = Partials + slot * 132
            peak = tl.load(addr + 128, part < parts, -1e30)
            denom = tl.load(addr + 129, part < parts, 0)
            accum = tl.load(addr[:, None] + d[None, :], part[:, None] < parts, 0)
            maximum = tl.max(peak, 0)
            rescale = tl.exp2(peak - maximum)
            total = tl.sum(rescale * denom, 0)
            out = tl.sum(accum * rescale[:, None], 0) / total
            row = qe - active + local_q
            tl.store(O + row * 4096 + head * 128 + d, out)
            tl.store(L + row * 32 + head, maximum + tl.log2(total))


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, output, lse):
    total_q = q.shape[0]
    if not total_q:
        return
    requests = qo_indptr.numel() - 1
    keys = kv_indices.numel()
    request_block = triton.next_power_of_2(requests)
    scale = float(sm_scale) * math.log2(math.e)
    zq = 4 if total_q >= 512 else 1
    empty_ctas = triton.cdiv(total_q, zq)
    if keys >= 512:
        split, n = 128, 64
        max_parts = triton.cdiv(keys, split)
        partials = torch.empty((max_parts, requests, 16, 32, 132),
                               dtype=torch.float32, device=q.device)
        tiles = triton.cdiv(min(total_q, keys), 4) + requests - 1 + 4 * max_parts
        active_ctas = 8 * tiles
        _long[(active_ctas + empty_ctas,)](
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse, partials,
            scale, active_ctas, total_q, requests, request_block, zq, split, n, num_warps=4)
        _join[(requests, 32)](qo_indptr, kv_indptr, partials, output, lse,
                              requests, split, triton.next_power_of_2(max_parts), num_warps=4)
    else:
        short = triton.next_power_of_2(max(keys, 1)) if keys <= 4 else 0
        n = 64 if total_q <= 2 else 32
        if short:
            active_ctas = min(total_q, keys) * 32
        else:
            active_ctas = (triton.cdiv(min(total_q, keys), 4) + requests - 1) * 8
        _fused[(active_ctas + empty_ctas,)](
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse,
            scale, active_ctas, total_q, requests, request_block, n, zq, short, num_warps=4)
