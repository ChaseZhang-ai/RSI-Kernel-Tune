"""Paged grouped-query decode for BF16 caches and FP32 base-2 LSE.

Tile reuse follows RSI-Kernel-Tune's H100 decode seed. The transposed
K @ Q.T and V.T @ P layout and split-KV scheduling were inspired by
QianQu/CompoundingKernel-Explorer, FlashInfer 012 (Claude via Amazon Bedrock).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny(Q, K, V, Indptr, Indices, O, L, scale, N: tl.constexpr):
    head = tl.program_id(0)
    row = tl.program_id(1)
    start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    d = tl.arange(0, 128)
    t = start + tl.arange(0, N)
    q = tl.load(Q + (row * 32 + head) * 128 + d).to(tl.float32)
    page = tl.load(Indices + t, t < end, 0)
    offset = page[:, None] * 512 + (head // 8) * 128 + d[None, :]
    k = tl.load(K + offset, t[:, None] < end, 0).to(tl.float32)
    v = tl.load(V + offset, t[:, None] < end, 0).to(tl.float32)
    scores = tl.where(t < end, tl.sum(k * q[None, :], 1) * scale, float("-inf"))
    peak = tl.max(scores, 0)
    weights = tl.exp2(scores - tl.where(start < end, peak, 0))
    den = tl.sum(weights, 0)
    safe = tl.where(den > 0, den, 1)
    out = tl.sum(weights[:, None] * v, 0) / safe
    tl.store(O + (row * 32 + head) * 128 + d, out)
    tl.store(L + row * 32 + head, peak + tl.log2(safe))


@triton.jit
def _tiles(Q, K, V, Indptr, Indices, O, L, Scratch, scale,
           B: tl.constexpr, SEARCH: tl.constexpr, N: tl.constexpr,
           SPLIT: tl.constexpr, HD: tl.constexpr, TRANSPOSED: tl.constexpr):
    group = tl.program_id(1)
    vpart = tl.program_id(2)
    slot = tl.program_id(0)
    if SPLIT:
        # Row r owns slots floor(indptr[r] / N) + r ... one before the
        # next row's base. At most one slot per row can be padding.
        lo = 0
        hi = B
        for _ in tl.static_range(0, SEARCH):
            mid = (lo + hi) // 2
            row_base = tl.load(Indptr + mid) // N + mid
            lo, hi = tl.where(slot >= row_base, mid, lo), tl.where(slot >= row_base, hi, mid)
        row = lo
        first = tl.load(Indptr + row)
        start = first + (slot - first // N - row) * N
    else:
        row = slot
        start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)

    if not SPLIT or start < end:
        m = tl.arange(0, 16)
        d = tl.arange(0, 128)
        dv = vpart * HD + tl.arange(0, HD)
        t = start + tl.arange(0, N)
        head = row * 32 + group * 8 + m
        q = tl.load(Q + head[:, None] * 128 + d[None, :], m[:, None] < 8, 0)
        page = tl.load(Indices + t, t < end, 0)
        kv = page[:, None] * 512 + group * 128
        k = tl.load(K + kv + d[None, :], t[:, None] < end, 0)
        v = tl.load(V + kv + dv[None, :], t[:, None] < end, 0)
        if TRANSPOSED:
            scores = tl.trans(tl.dot(k, tl.trans(q))) * scale
        else:
            scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(t[None, :] < end, scores, float("-inf"))
        peak = tl.max(scores, 1)
        p = tl.exp2(scores - tl.where(start < end, peak, 0)[:, None])
        den = tl.sum(p, 1)
        if TRANSPOSED:
            acc = tl.trans(tl.dot(tl.trans(v), tl.trans(p.to(tl.bfloat16))))
        else:
            acc = tl.dot(p.to(tl.bfloat16), v)
        if SPLIT:
            base = (slot * 32 + group * 8 + m) * 130
            tl.store(Scratch + base[:, None] + dv[None, :], acc, m[:, None] < 8)
            if vpart == 0:
                tl.store(Scratch + base + 128, peak, m < 8)
                tl.store(Scratch + base + 129, den, m < 8)
        else:
            safe = tl.where(den > 0, den, 1)
            tl.store(O + head[:, None] * 128 + dv[None, :], acc / safe[:, None], m[:, None] < 8)
            if vpart == 0:
                tl.store(L + head, peak + tl.log2(safe), m < 8)


@triton.jit
def _merge(Indptr, Scratch, O, L, N: tl.constexpr, M: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    first_slot = start // N + row
    count = tl.cdiv(end - start, N)
    i = tl.arange(0, M)
    d = tl.arange(0, 128)
    maximum = float("-inf")
    total = 0.0
    acc = tl.full((128,), 0, tl.float32)
    for begin in range(0, count, M):
        chunk = begin + i
        base = ((first_slot + chunk) * 32 + head) * 130
        peak = tl.load(Scratch + base + 128, chunk < count, float("-inf"))
        den = tl.load(Scratch + base + 129, chunk < count, 0)
        next_maximum = tl.maximum(maximum, tl.max(peak, 0))
        weights = tl.exp2(peak - next_maximum)
        decay = tl.exp2(maximum - next_maximum)
        partial = tl.load(Scratch + base[:, None] + d[None, :], chunk[:, None] < count, 0)
        acc = acc * decay + tl.sum(partial * weights[:, None], 0)
        total = total * decay + tl.sum(den * weights, 0)
        maximum = next_maximum
    safe = tl.where(total > 0, total, 1)
    tl.store(O + (row * 32 + head) * 128 + d, acc / safe)
    tl.store(L + row * 32 + head, maximum + tl.log2(safe))


def _launch(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
            mode, tile, warps, width=128, transpose=True):
    batch = q.shape[0]
    tokens = kv_indices.numel()
    scale = sm_scale * 1.4426950408889634
    if mode == "tiny":
        _tiny[(32, batch)](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, scale, tile, num_warps=warps
        )
    elif mode == "direct":
        _tiles[(batch, 4, 128 // width)](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, lse, scale,
            batch, (batch - 1).bit_length(), tile, False, width, transpose, num_warps=warps,
        )
    else:
        slots = tokens // tile + batch
        scratch = torch.empty((slots, 32, 130), device=q.device, dtype=torch.float32)
        _tiles[(slots, 4, 128 // width)](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, scratch, scale,
            batch, (batch - 1).bit_length(), tile, True, width, transpose, num_warps=warps,
        )
        merge_tile = min(32, triton.next_power_of_2(max(1, (tokens + tile - 1) // tile)))
        _merge[(batch, 32)](kv_indptr, scratch, output, lse, tile, merge_tile, num_warps=4)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch = q.shape[0]
    tokens = kv_indices.numel()
    args = (q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse)
    if batch == 1 and tokens <= 32:
        _launch(*args, "tiny", triton.next_power_of_2(max(1, tokens)), 4)
    elif batch == 1 and tokens <= 128:
        _launch(*args, "direct", max(16, triton.next_power_of_2(tokens)), 4, 32)
    else:
        _launch(*args, "split", 128 if batch > 1 else 32, 4)
