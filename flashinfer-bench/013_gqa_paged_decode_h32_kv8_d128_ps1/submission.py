"""Destination-passing grouped-query attention over page-size-one KV caches.

The split softmax follows RSI-Kernel-Tune's AKO4X per-workload split-max
kernel. The CompoundingKernel archive demonstrates length-dependent CTA
dispatch and split-KV for this operator. The H100 RSI kernel motivates
loading the page index once per token tile and reusing it for K and V.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _small(
    Q, K, V, Indptr, Indices, Out, Lse, scale,
    PAGES: tl.constexpr, N: tl.constexpr,
):
    head = tl.program_id(0)
    start = tl.load(Indptr)
    length = tl.load(Indptr + 1) - start
    token = tl.arange(0, N)
    dim = tl.arange(0, 128)
    valid = token < length
    page = tl.load(Indices + start + token, valid, 0)
    address = page[:, None] * 1024 + (head // 4) * 128 + dim[None, :]
    query = tl.load(Q + head * 128 + dim).to(tl.float32)
    key = tl.load(K + address, valid[:, None], 0).to(tl.float32)
    score = tl.sum(key * query[None, :], 1) * scale
    score = tl.where(valid, score, -float("inf"))
    max_score = tl.max(score, 0)
    weights = tl.exp2(score - tl.where(length > 0, max_score, 0.0))
    denominator = tl.sum(weights, 0)
    value = tl.load(V + address, valid[:, None], 0).to(tl.float32)
    result = tl.sum(value * weights[:, None], 0) / tl.maximum(denominator, 1.0)
    tl.store(Out + head * 128 + dim, result)
    tl.store(Lse + head, max_score + tl.log2(denominator))


@triton.jit
def _decode(
    Q, K, V, Indptr, Indices, Out, Lse, Partial, PartialLse, scale,
    H: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr,
    MIN_CHUNK: tl.constexpr, SINGLE_TILE: tl.constexpr,
    ORDER: tl.constexpr, PV_FP16: tl.constexpr,
):
    if ORDER == 0:
        batch = tl.program_id(0)
        group = tl.program_id(1)
        split = tl.program_id(2)
    else:
        group = tl.program_id(0)
        split = tl.program_id(1)
        batch = tl.program_id(2)

    start = tl.load(Indptr + batch)
    end = tl.load(Indptr + batch + 1)
    length = end - start
    if SPLITS > 1:
        active_splits = tl.maximum(1, tl.minimum(SPLITS, tl.cdiv(length, MIN_CHUNK)))
        chunk = tl.cdiv(length, active_splits)
        start += split * chunk
        end = tl.minimum(start + chunk, end)

    row = tl.arange(0, 16)
    head = batch * 32 + group * H + row
    dim = tl.arange(0, 128)
    token = tl.arange(0, N)
    kv_head = group // (4 // H)
    query = tl.load(Q + head[:, None] * 128 + dim[None, :], row[:, None] < H, 0)
    max_score = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0.0, tl.float32)
    numerator = tl.full((16, 128), 0.0, tl.float32)

    for tile in range(start, start + N if SINGLE_TILE else end, N):
        valid = tile + token < end
        page = tl.load(Indices + tile + token, valid, 0)
        address = page[:, None] * 1024 + kv_head * 128 + dim[None, :]
        key = tl.load(K + address, valid[:, None], 0, cache_modifier=".cg")
        value = tl.load(V + address, valid[:, None], 0, cache_modifier=".cg")
        score = tl.dot(query, tl.trans(key)) * scale
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(max_score, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        rescale = tl.exp2(max_score - safe_max)
        weights = tl.exp2(score - safe_max[:, None])
        denominator = denominator * rescale + tl.sum(weights, 1)
        scaled = numerator * rescale[:, None]
        if PV_FP16:
            numerator = tl.dot(weights.to(tl.float16), value.to(tl.float16), scaled)
        else:
            numerator = tl.dot(weights.to(tl.bfloat16), value, scaled)
        max_score = next_max

    logsum = max_score + tl.log2(denominator)
    result = numerator / tl.maximum(denominator, 1.0)[:, None]
    if SPLITS == 1:
        tl.store(Out + head[:, None] * 128 + dim[None, :], result, row[:, None] < H)
        tl.store(Lse + head, logsum, row < H)
    else:
        slot = head * SPLITS + split
        tl.store(Partial + slot[:, None] * 128 + dim[None, :], result, row[:, None] < H)
        tl.store(PartialLse + slot, logsum, row < H)


@triton.jit
def _merge(Partial, PartialLse, Out, Lse, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    split = tl.arange(0, triton.next_power_of_2(SPLITS))
    dim = tl.arange(0, 128)
    slot = head * SPLITS + split
    logsum = tl.load(PartialLse + slot, split < SPLITS, -float("inf"))
    largest = tl.max(logsum, 0)
    weights = tl.exp2(logsum - tl.where(largest == -float("inf"), 0.0, largest))
    denominator = tl.sum(weights, 0)
    values = tl.load(Partial + slot[:, None] * 128 + dim[None, :], split[:, None] < SPLITS, 0)
    result = tl.sum(values * weights[:, None], 0) / tl.maximum(denominator, 1.0)
    tl.store(Out + head * 128 + dim, result)
    tl.store(Lse + head, largest + tl.log2(denominator))


def _config(batch, count):
    if batch == 1:
        if count <= 128:
            return 2, max(16, triton.next_power_of_2(count)), 1, 1, 4, 1, 1
        return 2, 64, 4, 64, 4, 2, 1
    if batch <= 16:
        if count <= 4500:
            return 2, 64, 1, 1, 4, 3, 0
        return 4, 64, 8, 128, 4, 3, 0
    return 4, 64, 8, 128, 4, 3, 1


def _launch(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
            config, *, fp16=True):
    batch = q.shape[0]
    heads, tile, splits, min_chunk, warps, stages, order = config
    if splits == 1:
        partial, partial_lse = output, lse
    else:
        partial = torch.empty((batch * 32 * splits, 128), device=q.device, dtype=torch.float32)
        partial_lse = torch.empty((batch * 32 * splits,), device=q.device, dtype=torch.float32)
    groups = 32 // heads
    grid = (batch, groups, splits) if order == 0 else (groups, splits, batch)
    one_tile = batch == 1 and splits == 1 and kv_indices.numel() <= tile
    _decode[grid](
        q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, partial, partial_lse,
        float(sm_scale) * 1.4426950408889634, heads, tile, splits, min_chunk, one_tile,
        order, fp16, num_warps=warps, num_stages=stages, enable_fp_fusion=True,
    )
    if splits > 1:
        _merge[(batch * 32,)](partial, partial_lse, output, lse, splits, num_warps=2)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch, count = q.shape[0], kv_indices.numel()
    if batch == 1 and count <= 16:
        _small[(32,)](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            float(sm_scale) * 1.4426950408889634, k_cache.shape[0],
            max(1, triton.next_power_of_2(count)), num_warps=4,
        )
    else:
        _launch(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
                _config(batch, count))
