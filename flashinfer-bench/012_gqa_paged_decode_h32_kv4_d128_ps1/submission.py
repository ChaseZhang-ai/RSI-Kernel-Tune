"""Paged grouped-query decode for BF16 caches and FP32 base-2 LSE.

Tile reuse follows RSI-Kernel-Tune's H100 decode seed. The transposed
K @ Q.T and V.T @ P layout and split-KV scheduling were inspired by
QianQu/CompoundingKernel-Explorer, FlashInfer 012 (Claude via Amazon Bedrock).
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout
from triton.experimental.gluon.language._core import builtin


@builtin
def _mma_v2(a, b, acc, _semantic=None):
    # This uses the same Triton dot lowering on versions before mma_v2
    # became a public Gluon function.
    dot = _semantic.dot(a, b, acc, input_precision=None,
                        max_num_imprecise_acc=None, out_dtype=acc.dtype)
    return gl.tensor(dot.handle, acc.type)


@triton.jit
def _short(Q, K, V, Indptr, Indices, O, L, scale,
           T, N: tl.constexpr):
    head = tl.program_id(0)
    begin = tl.load(Indptr)
    end = tl.load(Indptr + 1)
    d = tl.arange(0, 128)
    tokens = tl.arange(0, N)
    indices = tl.load(Indices + tokens, tokens < T, 0)
    q = tl.load(Q + head * 128 + d).to(tl.float32)
    page = indices
    live = tokens < T
    valid = (tokens >= begin) & (tokens < end)
    offset = page[:, None] * 512 + (head // 8) * 128 + d[None, :]
    k = tl.load(K + offset, live[:, None], 0).to(tl.float32)
    v = tl.load(V + offset, live[:, None], 0).to(tl.float32)
    k = tl.where(valid[:, None], k, 0)
    v = tl.where(valid[:, None], v, 0)
    score = tl.where(valid, tl.sum(k * q[None, :], 1) * scale, float("-inf"))
    peak = tl.max(score, 0)
    p = tl.exp2(score - tl.where(begin < end, peak, 0))
    den = tl.sum(p, 0)
    divisor = tl.where(den > 0, den, 1)
    tl.store(O + head * 128 + d, tl.sum(p[:, None] * v, 0) / divisor)
    tl.store(L + head, peak + tl.log2(divisor))


@gluon.jit
def _single_row(Q, K, V, Indptr, Indices, O, L, scale, T,
                N: gl.constexpr, D: gl.constexpr, NW: gl.constexpr):
    group = gl.program_id(1)
    dv_part = gl.program_id(2)
    start = gl.load(Indptr)
    end = gl.load(Indptr + 1)
    GL: gl.constexpr = BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    MM: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[NW, 1], instr_shape=[16, 8])
    A: gl.constexpr = DotOperandLayout(0, MM, 2)
    BB: gl.constexpr = DotOperandLayout(1, MM, 2)
    h = gl.arange(0, 8, layout=gl.constexpr(SliceLayout(1, GL)))
    d = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, GL)))
    dv = dv_part * D + gl.arange(0, D, layout=gl.constexpr(SliceLayout(0, GL)))
    token = gl.arange(0, N, layout=gl.constexpr(SliceLayout(1, GL)))
    head = group * 8 + h
    q = gl.load(Q + head[:, None] * 128 + d[None, :])
    page = gl.load(Indices + token, token < T, 0)
    live = (token >= start) & (token < end)
    kv = page[:, None] * 512 + group * 128
    k = gl.load(K + kv + d[None, :], (token < T)[:, None], 0)
    v = gl.load(V + kv + dv[None, :], (token < T)[:, None], 0)
    k = gl.where(live[:, None], k, 0)
    v = gl.where(live[:, None], v, 0)
    logit = _mma_v2(gl.convert_layout(k, A), gl.convert_layout(gl.permute(q, (1, 0)), BB),
                   gl.full((N, 8), 0, gl.float32, layout=MM)) * scale
    logit = gl.where(gl.convert_layout(live, gl.constexpr(SliceLayout(1, MM)))[:, None],
                     logit, float("-inf"))
    peak = gl.max(logit, 0)
    prob = gl.exp2(logit - gl.where(start < end, peak, 0)[None, :])
    den = gl.sum(prob, 0)
    acc = _mma_v2(gl.convert_layout(gl.permute(v, (1, 0)), A),
                 gl.convert_layout(prob.to(gl.bfloat16), BB),
                 gl.full((D, 8), 0, gl.float32, layout=MM))
    acc = gl.convert_layout(gl.permute(acc, (1, 0)), GL)
    peak = gl.convert_layout(peak, gl.constexpr(SliceLayout(1, GL)))
    den = gl.convert_layout(den, gl.constexpr(SliceLayout(1, GL)))
    safe = gl.where(den > 0, den, 1)
    gl.store(O + head[:, None] * 128 + dv[None, :], acc / safe[:, None])
    if dv_part == 0:
        gl.store(L + head, peak + gl.log2(safe))


@triton.jit
def _split(Q, K, V, Indptr, Indices, Scratch, scale,
           B: tl.constexpr, SEARCH: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
           VEC_SEARCH: tl.constexpr):
    group = tl.program_id(1)
    slot = tl.program_id(0)
    # A row owns slots floor(indptr[row] / N) + row onward.
    # At most one slot per row is padding, including for empty rows.
    if VEC_SEARCH:
        rows = tl.arange(0, triton.next_power_of_2(B))
        starts = tl.load(Indptr + rows, rows < B, 0)
        eligible = (rows < B) & (starts // N + rows <= slot)
        row = tl.max(tl.where(eligible, rows, 0), 0)
        first = tl.sum(tl.where(rows == row, starts, 0), 0)
    else:
        lo = 0
        hi = B
        for _ in tl.static_range(0, SEARCH):
            mid = (lo + hi) // 2
            base = tl.load(Indptr + mid) // N + mid
            lo, hi = tl.where(slot >= base, mid, lo), tl.where(slot >= base, hi, mid)
        row = lo
        first = tl.load(Indptr + row)
    start = first + (slot - first // N - row) * N
    end = tl.load(Indptr + row + 1)
    if start < end:
        h = tl.arange(0, 16)
        d = tl.arange(0, 128)
        t = start + tl.arange(0, N)
        head = row * 32 + group * 8 + h
        q = tl.load(Q + head[:, None] * 128 + d[None, :], h[:, None] < 8, 0)
        page = tl.load(Indices + t, t < end, 0)
        kv = page[:, None] * 512 + group * 128 + d[None, :]
        k = tl.load(K + kv, t[:, None] < end, 0)
        v = tl.load(V + kv, t[:, None] < end, 0)
        score = tl.dot(q, tl.trans(k)) * scale
        score = tl.where(t[None, :] < end, score, float("-inf"))
        peak = tl.max(score, 1)
        p = tl.exp2(score - peak[:, None])
        den = tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), v)
        record = (slot * 32 + group * 8 + h) * S
        tl.store(Scratch + record[:, None] + d[None, :], acc, h[:, None] < 8)
        tl.store(Scratch + record + 128, peak, h < 8)
        tl.store(Scratch + record + 129, den, h < 8)


@triton.jit
def _merge(Indptr, Scratch, O, L, N: tl.constexpr, M: tl.constexpr, S: tl.constexpr):
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
        base = ((first_slot + chunk) * 32 + head) * S
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



def _launch_split(q, k, v, indptr, indices, sm_scale, output, lse, *, tile,
                  warps, merge_tile=32, vector_search=True, stride=136):
    batch, tokens = q.shape[0], indices.numel()
    slots = tokens // tile + batch
    scratch = torch.empty((slots, 32, stride), dtype=torch.float32, device=q.device)
    _split[(slots, 4)](q, k, v, indptr, indices, scratch, sm_scale * 1.4426950408889634,
                       batch, (batch - 1).bit_length(), tile, stride, vector_search, num_warps=warps)
    merge_tile = min(merge_tile, triton.next_power_of_2(max(1, triton.cdiv(tokens, tile))))
    _merge[(batch, 32)](indptr, scratch, output, lse, tile, merge_tile, stride, num_warps=4)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse):
    batch, tokens = q.shape[0], kv_indices.numel()
    if batch == 0:
        return
    scale = sm_scale * 1.4426950408889634
    if batch == 1 and tokens <= 4:
        _short[(32,)](q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
                      scale, tokens, triton.next_power_of_2(max(1, tokens)), num_warps=1)
    elif batch == 1 and tokens <= 256:
        _single_row[(1, 4, 2)](q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
                               scale, tokens, max(16, triton.next_power_of_2(tokens)), 64, 4, num_warps=4)
    else:
        if batch == 1:
            tile, warps = 32, 4
        elif batch <= 16 and tokens < 256 * batch:
            tile, warps = 32, 2
        elif batch <= 16:
            tile, warps = 128, 8
        else:
            tile, warps = 128, 4
        _launch_split(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
                      tile=tile, warps=warps)
