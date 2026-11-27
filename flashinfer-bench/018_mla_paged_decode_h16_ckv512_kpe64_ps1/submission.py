"""Paged MLA decode using standard Triton BF16 matrix products.

The split-KV schedule and shared use of each compressed cache tile follow
RSI-Kernel-Tune's H100 seed and CompoundingKernel Explorer's FlashInfer 018.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _small(Q, QP, C, KP, Indptr, Indices, O, L, scale, T: tl.constexpr, N: tl.constexpr):
    h = tl.program_id(0)
    begin = tl.load(Indptr)
    end = tl.load(Indptr + 1)
    t = tl.arange(0, N)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    live = (t >= begin) & (t < end) & (t < T)
    pages = tl.load(Indices + t, t < T, 0)
    ckv = tl.load(C + pages[:, None] * 512 + d[None, :], live[:, None], 0).to(tl.float32)
    kpe = tl.load(KP + pages[:, None] * 64 + r[None, :], live[:, None], 0).to(tl.float32)
    q = tl.load(Q + h * 512 + d).to(tl.float32)
    qp = tl.load(QP + h * 64 + r).to(tl.float32)
    logits = tl.sum(ckv * q[None, :], 1) + tl.sum(kpe * qp[None, :], 1)
    score = tl.where(live, logits * scale, float("-inf"))
    peak = tl.max(score, 0)
    weights = tl.exp2(score - tl.where(begin < end, peak, 0.0))
    denom = tl.sum(weights, 0)
    safe = tl.where(denom > 0, denom, 1.0)
    out = tl.sum(ckv * weights[:, None], 0) / safe
    tl.store(O + h * 512 + d, out)
    tl.store(L + h, peak + tl.log2(safe))


@triton.jit
def _decode(Q, QP, C, KP, Indptr, Indices, Partial, Stats, O, L, scale,
            B: tl.constexpr, CHUNK: tl.constexpr, N: tl.constexpr):
    slot = tl.program_id(0)
    if CHUNK == 0:
        row = slot
        first = tl.load(Indptr + row)
    elif B == 1:
        row = 0
        first = tl.load(Indptr)
    else:
        # Adding the row index makes these slot bases strictly increasing,
        # including when adjacent CSR rows are empty.
        rows = tl.arange(0, triton.next_power_of_2(B))
        starts = tl.load(Indptr + rows, rows < B, 0)
        owns_slot = (rows < B) & (starts // CHUNK + rows <= slot)
        row = tl.max(tl.where(owns_slot, rows, 0), 0)
        first = tl.sum(tl.where(rows == row, starts, 0), 0)
    end = tl.load(Indptr + row + 1)
    begin = first
    if CHUNK:
        begin = first + (slot - first // CHUNK - row) * CHUNK
        end = tl.minimum(end, begin + CHUNK)

    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    n = tl.arange(0, N)
    head = row * 16 + h
    peak = tl.full((16,), float("-inf"), tl.float32)
    denom = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    if begin < end:
        q = tl.load(Q + head[:, None] * 512 + d[None, :])
        qp = tl.load(QP + head[:, None] * 64 + r[None, :])
        if CHUNK == N:
            peak, denom, acc = _tile(q, qp, C, KP, Indices, begin + n, end,
                                     d, r, scale, peak, denom, acc)
        else:
            for offset in range(begin, end, N):
                peak, denom, acc = _tile(q, qp, C, KP, Indices, offset + n, end,
                                         d, r, scale, peak, denom, acc)
        if CHUNK:
            tl.store(Partial + (slot * 16 + h[:, None]) * 512 + d[None, :], acc)
            tl.store(Stats + slot * 32 + h, peak)
            tl.store(Stats + slot * 32 + h + 16, denom)
    if CHUNK == 0:
        safe = tl.where(denom > 0, denom, 1.0)
        tl.store(O + head[:, None] * 512 + d[None, :], acc / safe[:, None])
        tl.store(L + head, peak + tl.log2(safe))


@triton.jit
def _tile(q, qp, C, KP, Indices, tokens, end, d, r, scale, peak, denom, acc):
    live = tokens < end
    page = tl.load(Indices + tokens, live, 0)
    ckv = tl.load(C + page[:, None] * 512 + d[None, :], live[:, None], 0)
    kpe = tl.load(KP + page[:, None] * 64 + r[None, :], live[:, None], 0)
    content = tl.dot(q, tl.trans(ckv))
    logits = tl.dot(qp, tl.trans(kpe), content)
    score = tl.where(live[None, :], logits * scale, float("-inf"))
    new_peak = tl.maximum(peak, tl.max(score, 1))
    weight = tl.exp2(score - new_peak[:, None])
    decay = tl.exp2(peak - new_peak)
    denom = denom * decay + tl.sum(weight, 1)
    acc = tl.dot(weight.to(tl.bfloat16), ckv, acc * decay[:, None])
    return new_peak, denom, acc


@triton.jit
def _merge(Indptr, Partial, Stats, O, L, CHUNK: tl.constexpr,
           M: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    part = tl.program_id(2)
    begin = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    first = begin // CHUNK + row
    count = tl.cdiv(end - begin, CHUNK)
    d = part * D + tl.arange(0, D)
    i = tl.arange(0, M)
    acc = tl.full((D,), 0, tl.float32)
    peak = float("-inf")
    denom = 0.0
    for offset in range(0, count, M):
        slot = first + offset + i
        live = offset + i < count
        maxima = tl.load(Stats + slot * 32 + h, live, float("-inf"))
        sums = tl.load(Stats + slot * 32 + h + 16, live, 0)
        new_peak = tl.maximum(peak, tl.max(maxima, 0))
        decay = tl.exp2(peak - new_peak)
        weights = tl.exp2(maxima - new_peak)
        values = tl.load(Partial + ((slot * 16 + h) * 512)[:, None] + d[None, :], live[:, None], 0)
        acc = acc * decay + tl.sum(values * weights[:, None], 0)
        denom = denom * decay + tl.sum(sums * weights, 0)
        peak = new_peak
    safe = tl.where(denom > 0, denom, 1.0)
    head = row * 16 + h
    tl.store(O + head * 512 + d, acc / safe)
    if part == 0:
        tl.store(L + head, peak + tl.log2(safe))


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch = q_nope.shape[0]
    tokens = kv_indices.numel()
    scale = sm_scale * math.log2(math.e)
    if batch == 1 and tokens <= 16:
        _small[(16,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                      output, lse, scale, tokens, triton.next_power_of_2(max(tokens, 1)))
        return

    if batch == 1 and tokens <= 128:
        _decode[(1,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                      output, lse, output, lse, scale, batch, 0, 32, num_warps=4)
        return

    tile = 32
    chunk = 64 if batch == 1 or tokens < 9000 else 128
    slots = batch + tokens // chunk
    partial_dtype = torch.float32 if batch == 1 else torch.bfloat16
    partial = torch.empty((slots, 16, 512), device=q_nope.device, dtype=partial_dtype)
    stats = torch.empty((slots, 32), device=q_nope.device, dtype=torch.float32)
    _decode[(slots,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                      partial, stats, output, lse, scale, batch, chunk, tile,
                      num_warps=4)
    width = 128 if batch == 1 else 512
    _merge[(batch, 16, 512 // width)](kv_indptr, partial, stats, output, lse, chunk, 8, width)
