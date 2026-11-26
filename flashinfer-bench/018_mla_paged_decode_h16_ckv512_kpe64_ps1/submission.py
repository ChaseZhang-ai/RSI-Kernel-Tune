"""BF16 paged MLA decode with FP32 online softmax and a split-KV merge.

Tile reuse follows the RSI-Kernel-Tune H100 seed. Length-dependent scheduling
is inspired by CompoundingKernel Explorer's FlashInfer 018 CUDA submission
(Claude via Amazon Bedrock).
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
    pages = tl.load(Indices + t, live, 0)
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
            B: tl.constexpr, CHUNK: tl.constexpr, N: tl.constexpr,
            TRANSPOSE: tl.constexpr, CORRECT_P: tl.constexpr):
    slot = tl.program_id(0)
    if CHUNK == 0 or B == 1:
        row = slot if CHUNK == 0 else 0
        first = tl.load(Indptr + row)
    else:
        # Each row has at most one padding slot. This arithmetic maps slots
        # directly onto the CSR ranges without a separate setup launch.
        rows = tl.arange(0, triton.next_power_of_2(B))
        starts = tl.load(Indptr + rows, rows < B, 0)
        row = tl.max(tl.where((rows < B) & (starts // CHUNK + rows <= slot), rows, 0), 0)
        first = tl.sum(tl.where(rows == row, starts, 0), 0)
    end = tl.load(Indptr + row + 1)
    if CHUNK:
        begin = first + (slot - first // CHUNK - row) * CHUNK
        end = tl.minimum(end, begin + CHUNK)
    else:
        begin = first

    h = tl.arange(0, 16)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    n = tl.arange(0, N)
    head = row * 16 + h
    peak = tl.full((16,), float("-inf"), tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    if TRANSPOSE:
        acc = tl.full((512, 16), 0, tl.float32)
    else:
        acc = tl.full((16, 512), 0, tl.float32)
    if begin < end:
        q = tl.load(Q + head[:, None] * 512 + d[None, :])
        qp = tl.load(QP + head[:, None] * 64 + r[None, :])
        for offset in range(begin, end, N):
            t = offset + n
            page = tl.load(Indices + t, t < end, 0)
            ckv = tl.load(C + page[:, None] * 512 + d[None, :], t[:, None] < end, 0)
            kpe = tl.load(KP + page[:, None] * 64 + r[None, :], t[:, None] < end, 0)
            if TRANSPOSE:
                logits = tl.dot(ckv, tl.trans(q)) + tl.dot(kpe, tl.trans(qp))
                score = tl.where(t[:, None] < end, logits * scale, float("-inf"))
                new_peak = tl.maximum(peak, tl.max(score, 0))
                weight = tl.exp2(score - new_peak[None, :])
                decay = tl.exp2(peak - new_peak)
                denom = denom * decay + tl.sum(weight, 0)
                p = weight.to(tl.bfloat16)
                acc = tl.dot(tl.trans(ckv), p, acc * decay[None, :])
                if CORRECT_P:
                    residual = (weight - p.to(tl.float32)).to(tl.bfloat16)
                    acc = tl.dot(tl.trans(ckv), residual, acc)
            else:
                logits = tl.dot(q, tl.trans(ckv)) + tl.dot(qp, tl.trans(kpe))
                score = tl.where(t[None, :] < end, logits * scale, float("-inf"))
                new_peak = tl.maximum(peak, tl.max(score, 1))
                weight = tl.exp2(score - new_peak[:, None])
                decay = tl.exp2(peak - new_peak)
                denom = denom * decay + tl.sum(weight, 1)
                p = weight.to(tl.bfloat16)
                acc = tl.dot(p, ckv, acc * decay[:, None])
                if CORRECT_P:
                    residual = (weight - p.to(tl.float32)).to(tl.bfloat16)
                    acc = tl.dot(residual, ckv, acc)
            peak = new_peak
        if CHUNK:
            if TRANSPOSE:
                values = tl.trans(acc)
            else:
                values = acc
            tl.store(Partial + (slot * 16 + h[:, None]) * 512 + d[None, :], values)
            tl.store(Stats + slot * 32 + h, peak)
            tl.store(Stats + slot * 32 + h + 16, denom)
    if CHUNK == 0:
        safe = tl.where(denom > 0, denom, 1.0)
        if TRANSPOSE:
            values = tl.trans(acc)
        else:
            values = acc
        tl.store(O + head[:, None] * 512 + d[None, :], values / safe[:, None])
        tl.store(L + head, peak + tl.log2(safe))


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

    tile = 32
    chunk = 64 if batch == 1 else 128
    slots = batch + tokens // chunk
    partial = torch.empty((slots, 16, 512), device=q_nope.device, dtype=torch.float32)
    stats = torch.empty((slots, 32), device=q_nope.device, dtype=torch.float32)
    _decode[(slots,)](q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                      partial, stats, output, lse, scale, batch, chunk, tile,
                      False, False, num_warps=8)
    _merge[(batch, 16, 4)](kv_indptr, partial, stats, output, lse, chunk, 8, 128)
