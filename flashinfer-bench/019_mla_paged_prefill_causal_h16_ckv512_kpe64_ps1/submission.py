"""Destination-passing causal paged MLA for FlashInfer-Bench 019.

The repository H100 seed informs CKV reuse and the online base-2 softmax.
The CompoundingKernel Explorer entry informs the short/long dispatch and
packing query tokens and heads into one MMA tile. This implementation is Triton.
"""
import torch
import triton
import triton.language as tl


from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier


@gluon.jit
def _tcgen_sequence(QO, tile, BATCH: gl.constexpr):
    if BATCH == 1:
        return 0, tile * 4, gl.load(QO + 1)
    # This tile numbering reserves one boundary slot per request. Each CTA
    # independently computes its owner from the live query offsets.
    lo = 0
    hi = BATCH
    for _ in gl.static_range(BATCH.bit_length()):
        mid = (lo + hi + 1) // 2
        q = gl.load(QO + mid)
        take = (mid < BATCH) & (q // 4 + mid <= tile)
        lo, hi = gl.where(take, mid, lo), gl.where(take, hi, mid - 1)
    start = gl.load(QO + lo)
    return lo, start + (tile - (start // 4 + lo)) * 4, gl.load(QO + lo + 1)


@gluon.jit
def _copy_rows(dst, X, rows, cols, mask, WIDTH: gl.constexpr, SHIFT: gl.constexpr):
    async_copy.async_copy_global_to_shared(
        dst, X + rows[:, None] * WIDTH + cols[None, :] + SHIFT, mask[:, None])


@gluon.jit
def _copy_keys(C, E, Page, keys, begin, limit, block, ck0, ck1, kp, dims, pos_dims):
    offset = block + keys
    page = gl.load(Page + begin + offset, offset < limit, 0)
    valid = offset < limit
    _copy_rows(ck0, C, page, dims, valid, 512, 0)
    _copy_rows(ck1, C, page, dims, valid, 512, 256)
    _copy_rows(kp, E, page, pos_dims, valid, 64, 0)
    async_copy.commit_group()


@gluon.jit
def _tcgen_store(O, state, rows, mask, OFFSET: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
    out = gl.convert_layout(state.to(gl.bfloat16), layout)
    rows = gl.convert_layout(rows, gl.SliceLayout(1, layout))
    mask = gl.convert_layout(mask, gl.SliceLayout(1, layout))
    col = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    gl.store(O + rows[:, None] * 512 + col[None, :] + OFFSET, out, mask[:, None])


@gluon.jit
def _tcgen(Q, R, C, E, QO, KO, Page, O, L, scale, BATCH: gl.constexpr,
           LAZY_RESCALE: gl.constexpr):
    BN: gl.constexpr = 64
    seq, first, end = _tcgen_sequence(QO, gl.program_id(0), BATCH)
    if first < end:
        k0 = gl.load(KO + seq)
        kv_len = gl.load(KO + seq + 1) - k0
        limit = gl.maximum(0, kv_len - end + gl.minimum(first + 4, end))
        COPY: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [4, 1], [1, 0])
        RC: gl.constexpr = gl.SliceLayout(1, COPY)
        CC: gl.constexpr = gl.SliceLayout(0, COPY)
        S: gl.constexpr = gl.BlockedLayout([1, BN // 2], [16, 2], [4, 1], [0, 1])
        SR: gl.constexpr = gl.SliceLayout(1, S)
        V: gl.constexpr = gl.BlockedLayout([1, 128], [16, 2], [4, 1], [0, 1])
        VR: gl.constexpr = gl.SliceLayout(1, V)
        query = first * 16 + gl.arange(0, 64, layout=RC)
        dims = gl.arange(0, 256, layout=CC)
        pos_dims = gl.arange(0, 64, layout=CC)
        valid_q = query // 16 < end
        qi = first * 16 + gl.arange(0, 64, layout=SR)
        vi = first * 16 + gl.arange(0, 64, layout=VR)
        max_score = gl.full((64,), -float("inf"), gl.float32, SR)
        norm = gl.full((64,), 0.0, gl.float32, SR)
        if limit > 0:
            QL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([64, 256], gl.bfloat16)
            RL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([64, 64], gl.bfloat16)
            CL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, 256], gl.bfloat16)
            EL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BN, 64], gl.bfloat16)
            PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([64, BN], gl.bfloat16)
            q0 = gl.allocate_shared_memory(gl.bfloat16, [64, 256], QL)
            q1 = gl.allocate_shared_memory(gl.bfloat16, [64, 256], QL)
            qr = gl.allocate_shared_memory(gl.bfloat16, [64, 64], RL)
            ck0 = gl.allocate_shared_memory(gl.bfloat16, [2, BN, 256], CL)
            ck1 = gl.allocate_shared_memory(gl.bfloat16, [2, BN, 256], CL)
            kp = gl.allocate_shared_memory(gl.bfloat16, [2, BN, 64], EL)
            p_smem = gl.allocate_shared_memory(gl.bfloat16, [64, BN], PL)
            bar = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
            mbarrier.init(bar.index(0), count=1)
            mbarrier.init(bar.index(1), count=1)
            score = allocate_tensor_memory(gl.float32, [64, BN], TensorMemoryLayout([64, BN], 1))
            rope = allocate_tensor_memory(gl.float32, [64, BN], TensorMemoryLayout([64, BN], 1))
            o0 = allocate_tensor_memory(gl.float32, [64, 256], TensorMemoryLayout([64, 256], 1))
            o1 = allocate_tensor_memory(gl.float32, [64, 256], TensorMemoryLayout([64, 256], 1))

            _copy_rows(q0, Q, query, dims, valid_q, 512, 0)
            _copy_rows(q1, Q, query, dims, valid_q, 512, 256)
            _copy_rows(qr, R, query, pos_dims, valid_q, 64, 0)
            kn = gl.arange(0, BN, layout=RC)
            _copy_keys(C, E, Page, kn, k0, limit, 0, ck0.index(0), ck1.index(0), kp.index(0), dims, pos_dims)
            sn = gl.arange(0, BN, layout=gl.SliceLayout(0, S))
            phase = 0
            slot = 0
            for block in range(0, limit, BN):
                async_copy.wait_group(0)
                c0, c1, e = ck0.index(slot), ck1.index(slot), kp.index(slot)
                slot ^= 1
                if block + BN < limit:
                    _copy_keys(C, E, Page, kn, k0, limit, block + BN,
                               ck0.index(slot), ck1.index(slot), kp.index(slot), dims, pos_dims)
                # The same CKV panels feed the score and value tensor cores.
                tcgen05_mma(q0, c0.permute((1, 0)), score, use_acc=False)
                tcgen05_mma(q1, c1.permute((1, 0)), score)
                tcgen05_mma(qr, e.permute((1, 0)), rope, use_acc=False)
                tcgen05_commit(bar.index(0))
                mbarrier.wait(bar.index(0), phase)
                logits = (score.load(S) + rope.load(S)) * scale
                logits = gl.where(block + sn[None, :] <= kv_len - end + qi[:, None] // 16,
                                  logits, -float("inf"))
                new_max = gl.maximum(max_score, gl.max(logits, 1))
                pivot = gl.maximum(new_max, -3.4e38)
                alpha = gl.where(new_max == -float("inf"), 1.0, gl.exp2(max_score - pivot))
                p = gl.exp2(logits - pivot[:, None])
                norm = norm * alpha + gl.sum(p, 1)
                max_score = new_max
                p_smem.store(p.to(gl.bfloat16))
                fence_async_shared()
                if block > 0:
                    if not LAZY_RESCALE or gl.sum((alpha != 1.0).to(gl.int32), 0) > 0:
                        a = gl.convert_layout(alpha, VR)
                        o0.store(o0.load(V) * a[:, None])
                        o1.store(o1.load(V) * a[:, None])
                tcgen05_mma(p_smem, c0, o0, use_acc=block > 0)
                tcgen05_mma(p_smem, c1, o1, use_acc=block > 0)
                tcgen05_commit(bar.index(1))
                mbarrier.wait(bar.index(1), phase)
                phase ^= 1
            mbarrier.invalidate(bar.index(0))
            mbarrier.invalidate(bar.index(1))
            nz = gl.convert_layout(norm, VR)
            norm_inv = 1.0 / gl.where(nz > 0, nz, 1.0)
            out0 = gl.where(nz[:, None] > 0, o0.load(V) * norm_inv[:, None], float("nan"))
            _tcgen_store(O, out0, vi, vi // 16 < end, 0)
            out1 = gl.where(nz[:, None] > 0, o1.load(V) * norm_inv[:, None], float("nan"))
            _tcgen_store(O, out1, vi, vi // 16 < end, 256)
        else:
            # The reference skips requests with no KV. With a nonempty KV
            # request it runs softmax on all -inf for fully masked rows.
            fill = gl.where(kv_len > 0, float("nan"), 0.0)
            out0 = gl.full((64, 256), 0.0, gl.float32, V) + fill
            _tcgen_store(O, out0, vi, vi // 16 < end, 0)
            _tcgen_store(O, out0, vi, vi // 16 < end, 256)
        gl.store(L + qi, max_score + gl.log2(norm), qi // 16 < end)


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
        _tcgen[(triton.cdiv(total, 4) + batch - 1,)](
            *inputs, output, lse, scale, batch, batch == 1 and total >= 8192,
            num_warps=4)
