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
    N: tl.constexpr,
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
    ORDER: tl.constexpr,
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
def _merge(Partial, PartialLse, Out, Lse, SPLITS: tl.constexpr, G: tl.constexpr):
    head = tl.program_id(0) * G + tl.arange(0, G)
    split = tl.arange(0, triton.next_power_of_2(SPLITS))
    dim = tl.arange(0, 128)
    slot = head[:, None] * SPLITS + split[None, :]
    logsum = tl.load(PartialLse + slot, split[None, :] < SPLITS, -float("inf"))
    largest = tl.max(logsum, 1)
    safe_max = tl.where(largest == -float("inf"), 0.0, largest)
    weights = tl.exp2(logsum - safe_max[:, None])
    denominator = tl.sum(weights, 1)
    values = tl.load(
        Partial + slot[:, :, None] * 128 + dim[None, None, :],
        split[None, :, None] < SPLITS, 0,
    )
    result = tl.sum(values * weights[:, :, None], 1)
    result /= tl.maximum(denominator, 1.0)[:, None]
    tl.store(Out + head[:, None] * 128 + dim[None, :], result)
    tl.store(Lse + head, largest + tl.log2(denominator))

def _config(batch, count):
    # Choose CTA density using only tensor shapes. The device reads the actual
    # row lengths for the split boundaries and makes every split cover its row.
    if batch == 1:
        if count <= 128:
            return 2, max(16, triton.next_power_of_2(count)), 1, 1, 4, 1, 1
        if count <= 512:
            return 2, 64, 8, 64, 4, 2, 1
        return 4, 64, 12, 32, 4, 2, 1
    return 4, 128, 1, 1, 4, 3, 0

def _launch(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
            config):
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
        order, num_warps=warps, num_stages=stages, enable_fp_fusion=True,
    )
    if splits > 1:
        _merge[(batch * 16,)](partial, partial_lse, output, lse, splits, 2, num_warps=2)


@triton.jit
def _chunked(
    Q, K, V, Indptr, Indices, Partial, PartialLse, scale,
    BATCH: tl.constexpr, CHUNK: tl.constexpr, N: tl.constexpr,
):
    slot = tl.program_id(0)
    kv_head = tl.program_id(1)
    # Sequence b owns [indptr[b] // CHUNK + b, indptr[b + 1] // CHUNK + b + 1).
    # The extra slot lets the same grid handle an empty row. Clamp each tile
    # to the live bounds in indptr.
    rows = tl.arange(0, triton.next_power_of_2(BATCH))
    offsets = tl.load(Indptr + rows, rows < BATCH, 0)
    row_slot = offsets // CHUNK + rows
    batch = tl.sum(((row_slot <= slot) & (rows < BATCH)).to(tl.int32), 0) - 1
    tile_begin = (slot - batch) * CHUNK
    begin = tl.maximum(tl.load(Indptr + batch), tile_begin)
    end = tl.minimum(tl.load(Indptr + batch + 1), tile_begin + CHUNK)

    row = tl.arange(0, 16)
    head = batch * 32 + kv_head * 4 + row
    dim = tl.arange(0, 128)
    token = tl.arange(0, N)
    query = tl.load(Q + head[:, None] * 128 + dim[None, :], row[:, None] < 4, 0)
    max_score = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.full((16,), 0.0, tl.float32)
    numer = tl.full((16, 128), 0.0, tl.float32)
    for start in range(begin, end, N):
        tile = start + token
        valid = tile < end
        page = tl.load(Indices + tile, valid, 0)
        address = page[:, None] * 1024 + kv_head * 128 + dim[None, :]
        key = tl.load(K + address, valid[:, None], 0, cache_modifier=".cg")
        value = tl.load(V + address, valid[:, None], 0, cache_modifier=".cg")
        score = tl.dot(query, tl.trans(key)) * scale
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(max_score, tl.max(score, 1))
        corr = tl.exp2(max_score - next_max)
        weights = tl.exp2(score - next_max[:, None])
        denom = denom * corr + tl.sum(weights, 1)
        numer = tl.dot(weights.to(tl.bfloat16), value, numer * corr[:, None])
        max_score = next_max

    result = numer / tl.maximum(denom, 1.0)[:, None]
    slot_head = slot * 32 + kv_head * 4 + row
    tl.store(Partial + slot_head[:, None] * 128 + dim[None, :], result, row[:, None] < 4)
    tl.store(PartialLse + slot_head, max_score + tl.log2(denom), row < 4)


@triton.jit
def _chunked_merge(
    Indptr, Partial, PartialLse, Out, Lse,
    CHUNK: tl.constexpr, G: tl.constexpr, STRIDE: tl.constexpr,
):
    pg = tl.program_id(0)
    batch = tl.program_id(1)
    begin = tl.load(Indptr + batch) // CHUNK + batch
    end = tl.load(Indptr + batch + 1) // CHUNK + batch + 1
    heads = pg * G + tl.arange(0, G)
    d = tl.arange(0, 128)
    relative_slot = tl.arange(0, STRIDE)
    m = tl.full((G,), -float("inf"), tl.float32)
    den = tl.full((G,), 0.0, tl.float32)
    acc = tl.full((G, 128), 0.0, tl.float32)
    for block in range(begin, end, STRIDE):
        read = block + relative_slot
        arg = read[None, :] * 32 + heads[:, None]
        lse = tl.load(PartialLse + arg, read[None, :] < end, -float("inf"))
        m_new = tl.maximum(m, tl.max(lse, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        weight = tl.exp2(lse - m_safe[:, None])
        factor = tl.exp2(m - m_safe)
        x = tl.load(Partial + arg[:, :, None] * 128 + d[None, None, :],
                    read[None, :, None] < end, 0.0)
        den = den * factor + tl.sum(weight, 1)
        acc = acc * factor[:, None] + tl.sum(x * weight[:, :, None], 1)
        m = m_new

    out = acc / tl.maximum(den, 1.0)[:, None]
    h = batch * 32 + heads
    tl.store(Out + h[:, None] * 128 + d[None, :], out)
    tl.store(Lse + h, m + tl.log2(den))


def _launch_chunked(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse, config):
    chunk, tile, warps, stages = config
    batch, count = q.shape[0], kv_indices.numel()
    slots = count // chunk + batch
    partial = torch.empty((slots * 32, 128), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty((slots * 32,), device=q.device, dtype=torch.float32)
    _chunked[(slots, 8)](
        q, k_cache, v_cache, kv_indptr, kv_indices, partial, partial_lse,
        float(sm_scale) * 1.4426950408889634, batch, chunk, tile,
        num_warps=warps, num_stages=stages,
    )
    _chunked_merge[(8, batch)](
        kv_indptr, partial, partial_lse, output, lse, chunk, 4, 8, num_warps=4,
    )


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch, count = q.shape[0], kv_indices.numel()
    if batch == 1 and count <= 4:
        _small[(32,)](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            float(sm_scale) * 1.4426950408889634,
            max(1, triton.next_power_of_2(count)), num_warps=1,
        )
    elif batch >= 32 or (batch > 1 and count > batch * 281):
        # Balance work across CTAs by splitting contiguous spans of the page
        # list, regardless of how the sequence lengths vary within a batch.
        if batch < 32 and count <= batch * 1024:
            config = (128, 32, 2, 2)
        else:
            config = (256, 64, 4, 3)
        _launch_chunked(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale,
                        output, lse, config)
    else:
        _launch(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
                _config(batch, count))
