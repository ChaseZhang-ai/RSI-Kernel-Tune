"""Destination-passing causal paged MLA for FlashInfer-Bench 019.

The repository H100 seed informs CKV reuse and the online base-2 softmax.
The CompoundingKernel Explorer entry informs the short/long dispatch and
packing query tokens and heads into one MMA tile. This implementation is Triton.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sequence(QO, tile, BATCH: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr):
    if BATCH == 1:
        return 0, tile * BT, tl.load(QO + 1), tl.load(QO)
    r = tl.arange(0, BS)
    start = tl.load(QO + r, r < BATCH, 0)
    end = tl.load(QO + r + 1, r < BATCH, 0)
    count = tl.cdiv(end - start, BT)
    after = tl.cumsum(count, 0)
    owns = (tile < after) & (tile >= after - count)
    seq = tl.sum(tl.where(owns, r, 0), 0)
    first = tl.sum(tl.where(owns, after - count, 0), 0)
    q0 = tl.sum(tl.where(owns, start, 0), 0)
    q1 = tl.sum(tl.where(owns, end, 0), 0)
    return seq, q0 + (tile - first) * BT, q1, q0


@triton.jit
def _direct(Q, R, C, E, QO, KO, Page, O, L, scale, BATCH: tl.constexpr,
            BS: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    seq, _, end, _ = _sequence(QO, token, BATCH, 1, BS)
    k0 = tl.load(KO + seq)
    k1 = tl.load(KO + seq + 1)
    row = token * 16 + head
    n = tl.arange(0, BN)
    d = tl.arange(0, 512)
    e = tl.arange(0, 64)
    valid = n < k1 - k0 - (end - token) + 1
    page = tl.load(Page + k0 + n, valid, 0)
    q = tl.load(Q + row * 512 + d).to(tl.float32)
    r = tl.load(R + row * 64 + e).to(tl.float32)
    ck = tl.load(C + page[:, None] * 512 + d[None, :], valid[:, None], 0).to(tl.float32)
    kp = tl.load(E + page[:, None] * 64 + e[None, :], valid[:, None], 0).to(tl.float32)
    s = (tl.sum(ck * q[None, :], 1) + tl.sum(kp * r[None, :], 1)) * scale
    s = tl.where(valid, s, -float("inf"))
    m = tl.max(s, 0)
    p = tl.exp2(s - tl.where(m == -float("inf"), 0.0, m))
    z = tl.sum(p, 0)
    o = tl.sum(p[:, None] * ck, 0) / tl.where(z > 0, z, 1.0)
    o = tl.where((z == 0) & (k0 < k1), float("nan"), o)
    tl.store(O + row * 512 + d, o)
    tl.store(L + row, m + tl.log2(z))


@triton.jit
def _prefill(Q, R, C, E, QO, KO, Page, O, L, scale, BATCH: tl.constexpr,
             BS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr = 1):
    seq, first, end, _ = _sequence(QO, tl.program_id(0), BATCH, BM // 16, BS)
    if first < end:
        k0 = tl.load(KO + seq)
        k1 = tl.load(KO + seq + 1)
        kv_len = k1 - k0
        m = tl.arange(0, BM)
        n = tl.arange(0, BN)
        d = tl.arange(0, 512)
        e = tl.arange(0, 64)
        row = first * 16 + m
        token = first + m // 16
        q = tl.load(Q + row[:, None] * 512 + d[None, :], token[:, None] < end, 0)
        r = tl.load(R + row[:, None] * 64 + e[None, :], token[:, None] < end, 0)
        mx = tl.full((BM,), -float("inf"), tl.float32)
        z = tl.full((BM,), 0, tl.float32)
        acc = tl.full((BM, 512), 0, tl.float32)
        limit = tl.maximum(0, kv_len - end + tl.minimum(first + BM // 16, end))
        loops = tl.cdiv(limit, BN)
        if SPLIT > 1:
            part = tl.program_id(1)
            width = tl.cdiv(loops, SPLIT)
            begin = part * width
            finish = tl.minimum(begin + width, loops)
        else:
            begin, finish = 0, loops
        for t in range(begin, finish):
            pos = t * BN + n
            page = tl.load(Page + k0 + pos, pos < limit, 0)
            ck = tl.load(C + page[:, None] * 512 + d[None, :], pos[:, None] < limit, 0)
            kp = tl.load(E + page[:, None] * 64 + e[None, :], pos[:, None] < limit, 0)
            qk = tl.dot(q, tl.trans(ck))
            qr = tl.dot(r, tl.trans(kp))
            logits = (qk + qr) * scale
            logits = tl.where(pos[None, :] <= (kv_len - end + token[:, None]), logits, -float("inf"))
            new_mx = tl.maximum(mx, tl.max(logits, 1))
            pivot = tl.where(new_mx == -float("inf"), 0.0, new_mx)
            alpha = tl.exp2(mx - pivot)
            p = tl.exp2(logits - pivot[:, None])
            z = z * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), ck, acc * alpha[:, None])
            mx = new_mx
        acc = acc / tl.where(z > 0, z, 1.0)[:, None]
        if SPLIT > 1:
            row = row * SPLIT + part
        else:
            acc = tl.where(((z == 0) & (kv_len > 0))[:, None], float("nan"), acc)
        tl.store(O + row[:, None] * 512 + d[None, :], acc, token[:, None] < end)
        tl.store(L + row, mx + tl.log2(z), token < end)


@triton.jit
def _merge(P, Z, QO, KO, O, L, BATCH: tl.constexpr, BS: tl.constexpr,
           SPLIT: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    row = token * 16 + head
    part = tl.arange(0, SPLIT)
    dim = tl.arange(0, 512)
    index = row * SPLIT + part
    lse_part = tl.load(Z + index)
    mx = tl.max(lse_part, 0)
    p = tl.exp2(lse_part - tl.where(mx == -float("inf"), 0.0, mx))
    z = tl.sum(p, 0)
    value = tl.load(P + index[:, None] * 512 + dim[None, :])
    o = tl.sum(p[:, None] * value, 0) / tl.where(z > 0, z, 1.0)
    if z == 0:
        seq, _, _, _ = _sequence(QO, token, BATCH, 1, BS)
        if tl.load(KO + seq) < tl.load(KO + seq + 1):
            o = tl.full((512,), float("nan"), tl.float32)
    tl.store(O + row * 512 + dim, o)
    tl.store(L + row, mx + tl.log2(z))


def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices,
        sm_scale, output, lse):
    total = q_nope.size(0)
    if total == 0:
        return

    batch = qo_indptr.numel() - 1
    keys = kv_indices.numel()
    batch_tile = triton.next_power_of_2(batch)
    scale = sm_scale * 1.4426950408889634
    inputs = (q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices)

    if keys <= 16:
        _direct[(total, 16)](
            *inputs, output, lse, scale, batch, batch_tile,
            max(4, triton.next_power_of_2(keys)), num_warps=4)
    elif total <= 128 and keys <= 256:
        block_keys = min(128, triton.next_power_of_2(keys))
        _prefill[(total,)](
            *inputs, output, lse, scale, batch, batch_tile, 16, block_keys,
            num_warps=4, num_stages=1)
    elif total <= 128:
        # Split long prefixes to expose parallel work even for one query.
        partitions = 32
        partial = torch.empty((total * 16, partitions, 512), device=output.device, dtype=torch.float32)
        stats = torch.empty((total * 16, partitions), device=lse.device, dtype=torch.float32)
        _prefill[(total, partitions)](
            *inputs, partial, stats, scale, batch, batch_tile, 16, 32, partitions,
            num_warps=4, num_stages=1)
        _merge[(total, 16)](
            partial, stats, qo_indptr, kv_indptr, output, lse,
            batch, batch_tile, partitions, num_warps=4)
    else:
        # Two queries share each CKV tile across their heads. The same
        # Triton kernel handles the ragged bounds and causal frontier.
        block_rows, block_keys = 32, 32
        blocks = triton.cdiv(total, block_rows // 16) + batch - 1
        _prefill[(blocks,)](
            *inputs, output, lse, scale, batch, batch_tile, block_rows, block_keys,
            num_warps=4, num_stages=2)
