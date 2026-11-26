"""Destination-passing BF16 paged causal GQA for 32 query heads and 4 KV heads.

The short/long dispatch follows the approach described in the public
CompoundingKernel implementation by Claude via Amazon Bedrock:
https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1

The kernels below are a Triton implementation.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _request_for_tile(QO, tile, B: tl.constexpr, T: tl.constexpr):
    # Pad the tile interval for each request by one slot. This keeps ownership
    # disjoint even when successive request boundaries share a tile.
    lo = 0
    hi = B
    for _ in tl.static_range(B.bit_length()):
        mid = (lo + hi + 1) // 2
        begins = tl.load(QO + mid) // T + mid
        take = (mid < B) & (begins <= tile)
        lo, hi = tl.where(take, mid, lo), tl.where(take, hi, mid - 1)
    return lo


@triton.jit
def _request_parallel(PTR, index, B: tl.constexpr, BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    batch = tl.arange(0, BLOCK)
    start = tl.load(PTR + batch, batch < B, 0x7fffffff)
    if SCALAR:
        return tl.max(tl.where(start <= index, batch, 0), 0)
    else:
        return tl.max(tl.where(start[None, :] <= index[:, None], batch[None, :], 0), 1)


@triton.jit
def _two_keys(Q, K, V, IDX, O, L, hidx, group, ks, prefix, scale):
    h = tl.arange(0, 8)
    d = tl.arange(0, 128)
    q_offset = (hidx + h[:, None]) * 128 + d[None, :]
    q = tl.load(Q + q_offset).to(tl.float32)
    page0 = tl.load(IDX + ks)
    offset0 = page0 * 512 + group * 128 + d
    k0 = tl.load(K + offset0).to(tl.float32)
    v0 = tl.load(V + offset0).to(tl.float32)
    score0 = tl.sum(q * k0[None, :], 1) * scale

    if prefix == 1:
        tl.store(O + q_offset, tl.broadcast_to(v0[None, :], (8, 128)))
        tl.store(L + hidx + h, score0)
    else:
        page1 = tl.load(IDX + ks + 1)
        offset1 = page1 * 512 + group * 128 + d
        k1 = tl.load(K + offset1).to(tl.float32)
        v1 = tl.load(V + offset1).to(tl.float32)
        score1 = tl.sum(q * k1[None, :], 1) * scale
        maximum = tl.maximum(score0, score1)
        p0 = tl.exp2(score0 - maximum)
        p1 = tl.exp2(score1 - maximum)
        denom = p0 + p1
        out = (p0[:, None] * v0[None, :] + p1[:, None] * v1[None, :]) / denom[:, None]
        tl.store(O + q_offset, out)
        tl.store(L + hidx + h, maximum + tl.log2(denom))


@triton.jit
def _sparse(Q, K, V, QO, KO, IDX, O, L, SCALE, TOTAL,
            B: tl.constexpr, NK: tl.constexpr, Z: tl.constexpr, N: tl.constexpr,
            B_PAD: tl.constexpr):
    tile = tl.program_id(0)
    active = tl.minimum(TOTAL, NK) if B == 1 else NK
    # Each key ordinal owns the query at the same distance from the request end.
    # A different group of CTAs writes the query positions before the first visible key.
    if tile < active * 4:
        ordinal = tile // 4 + NK - active
        group = tile % 4
        if B == 1:
            qs, qe, ks, ke = 0, TOTAL, 0, NK
        else:
            req = _request_parallel(KO, ordinal, B, B_PAD, True)
            qs = tl.load(QO + req)
            qe = tl.load(QO + req + 1)
            ks = tl.load(KO + req)
            ke = tl.load(KO + req + 1)
        qpos = qe + ordinal - ke
        if qpos >= qs:
            prefix = ordinal - ks + 1
            scale = SCALE * 1.4426950408889634
            hidx = qpos * 32 + group * 8
            if prefix <= 2:
                _two_keys(Q, K, V, IDX, O, L, hidx, group, ks, prefix, scale)
            else:
                h = tl.arange(0, 16)
                d = tl.arange(0, 128)
                q_offset = (hidx + h[:, None]) * 128 + d[None, :]
                q = tl.load(Q + q_offset, h[:, None] < 8, 0)
                norm = tl.full((16,), 0., tl.float32)
                max_score = tl.full((16,), -float("inf"), tl.float32)
                acc = tl.full((16, 128), 0., tl.float32)
                for start in range(0, prefix, N):
                    key = start + tl.arange(0, N)
                    page = tl.load(IDX + ks + key, key < prefix, 0)
                    offset = page[:, None] * 512 + group * 128 + d[None, :]
                    k = tl.load(K + offset, (key < prefix)[:, None], 0)
                    v = tl.load(V + offset, (key < prefix)[:, None], 0)
                    score = tl.dot(q, tl.trans(k)) * scale
                    score = tl.where(key[None, :] < prefix, score, -float("inf"))
                    m = tl.maximum(tl.max(score, 1), max_score)
                    p = tl.exp2(score - m[:, None])
                    blend = tl.exp2(max_score - m)
                    norm = norm * blend + tl.sum(p, 1)
                    acc = tl.dot(p.to(tl.bfloat16), v, acc * blend[:, None])
                    max_score = m
                tl.store(O + q_offset, acc / norm[:, None], h[:, None] < 8)
                tl.store(L + hidx + h, max_score + tl.log2(norm), h < 8)
    else:
        row = (tile - active * 4) * Z + tl.arange(0, Z)
        # Intervals use upper-bound search so empty requests own no query tokens.
        if B == 1:
            empty = row < TOTAL - NK
        else:
            q_req = _request_parallel(QO, row, B, B_PAD, False)
            q_end = tl.load(QO + q_req + 1)
            k_start = tl.load(KO + q_req)
            k_end = tl.load(KO + q_req + 1)
            empty = (row < TOTAL) & (row < q_end + k_start - k_end)
        d = tl.arange(0, 4096)
        h = tl.arange(0, 32)
        tl.store(O + row[:, None] * 4096 + d[None, :], 0, empty[:, None])
        tl.store(L + row[:, None] * 32 + h[None, :], -float("inf"), empty[:, None])


@triton.jit
def _gather(K, V, IDX, GK, GV, KEYS, N: tl.constexpr):
    key = tl.program_id(0) * N + tl.arange(0, N)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    p = tl.load(IDX + key, key < KEYS, 0)
    src = p[:, None] * 512 + h * 128 + d[None, :]
    dst = (h * KEYS + key[:, None]) * 128 + d[None, :]
    valid = (key < KEYS)[:, None]
    tl.store(GK + dst, tl.load(K + src, valid, 0), valid)
    tl.store(GV + dst, tl.load(V + src, valid, 0), valid)


@triton.jit
def _dense(Q, K, V, QO, KO, O, L, SCALE, B: tl.constexpr,
           M: tl.constexpr, N: tl.constexpr):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    T: tl.constexpr = M // 8
    req = _request_for_tile(QO, tile, B, T)
    qs = tl.load(QO + req)
    qe = tl.load(QO + req + 1)
    ks = tl.load(KO + req)
    ke = tl.load(KO + req + 1)
    first = qs + (tile - (qs // T + req)) * T
    r = tl.arange(0, M)
    qpos = first + r // 8
    head = group * 8 + r % 8
    hidx = qpos * 32 + head
    d = tl.arange(0, 128)
    prefix = qpos + 1 + ke - ks - qe
    stop = tl.where(first < qe, tl.minimum(ke - ks, first + T + ke - ks - qe), 0)
    acc = tl.full((M, 128), 0, tl.float32)
    normalizer = tl.full((M,), 0, tl.float32)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    q = Q.load([first, group, 0, 0]).reshape(M, 128)
    n = tl.arange(0, N)
    scale = SCALE * 1.4426950408889634
    for start in range(0, stop, N):
        key = start + n
        k = K.load([group, ks + start, 0]).reshape(N, 128)
        v = V.load([group, ks + start, 0]).reshape(N, 128)
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where((key[None, :] < prefix[:, None]) &
                          (key[None, :] < ke - ks), scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        correction = tl.exp2(maximum - safe_max)
        weights = tl.exp2(scores - safe_max[:, None])
        normalizer = normalizer * correction + tl.sum(weights, 1)
        acc = tl.dot(weights.to(tl.bfloat16), v, acc * correction[:, None])
        maximum = next_max
    out = acc / tl.where(normalizer > 0, normalizer, 1.)[:, None]
    lse = maximum + tl.log2(tl.where(normalizer > 0, normalizer, 1.))
    tl.store(O + hidx[:, None] * 128 + d[None, :], out, (qpos < qe)[:, None])
    tl.store(L + hidx, lse, qpos < qe)


@triton.jit
def _one_query(Q, K, V, IDX, O, L, SCALE,
               NK: tl.constexpr, KPAD: tl.constexpr):
    h = tl.program_id(0)
    d = tl.arange(0, 128)
    n = tl.arange(0, KPAD)
    q = tl.load(Q + h * 128 + d).to(tl.float32)
    page = tl.load(IDX + n, n < NK, 0)
    valid = n < NK
    off = (page[:, None] * 4 + h // 8) * 128 + d[None, :]
    k = tl.load(K + off, valid[:, None], 0).to(tl.float32)
    v = tl.load(V + off, valid[:, None], 0).to(tl.float32)
    score = tl.sum(q[None, :] * k, 1) * (SCALE * 1.4426950408889634)
    score = tl.where(valid, score, -float("inf"))
    maximum = tl.max(score, 0)
    p = tl.exp2(score - maximum)
    norm = tl.sum(p, 0)
    out = tl.sum(p[:, None] * v, 0) / norm
    tl.store(O + h * 128 + d, out)
    tl.store(L + h, maximum + tl.log2(norm))


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, output, lse):
    batch = qo_indptr.numel() - 1
    total = q.shape[0]
    keys = kv_indices.numel()
    if total == 0:
        return
    if total == 1 and batch == 1 and 2 < keys <= 64:
        _one_query[(32,)](
            q, k_cache, v_cache, kv_indices, output, lse, sm_scale,
            keys, triton.next_power_of_2(keys), num_warps=4,
        )
        return
    if keys < 256:
        z, warps = (4, 8) if total < 512 else (8, 4)
        grid = keys * 4 + triton.cdiv(total, z)
        if batch == 1:
            grid = min(total, keys) * 4 + triton.cdiv(max(0, total - keys), z)
        _sparse[(grid,)](
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse,
            sm_scale, total, batch, keys, z, 32,
            triton.next_power_of_2(batch), num_warps=warps,
        )
        return

    m, n = 128, 64
    # Pack page gathers once so the query tiles can reuse contiguous K/V through TMA.
    gk = torch.empty((4, keys, 128), device=k_cache.device, dtype=k_cache.dtype)
    gv = torch.empty_like(gk)
    q_desc = TensorDescriptor(q, [total, 4, 8, 128], [4096, 1024, 128, 1], [m // 8, 1, 8, 128])
    k_desc = TensorDescriptor.from_tensor(gk, [1, n, 128])
    v_desc = TensorDescriptor.from_tensor(gv, [1, n, 128])
    tiles = triton.cdiv(total, m // 8) + batch
    _gather[(triton.cdiv(keys, 16), 4)](k_cache, v_cache, kv_indices, gk, gv, keys, 16, num_warps=4)
    _dense[(tiles, 4)](
        q_desc, k_desc, v_desc, qo_indptr, kv_indptr, output, lse,
        sm_scale, batch, m, n, num_warps=4, num_stages=2,
    )
