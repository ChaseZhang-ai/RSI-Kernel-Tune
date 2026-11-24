import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_E = 128
_K = 8
_I = 1408
_H = 4096


_OVERLAP_MAX = 12000


torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _route_kernel(logits_ptr, bias_ptr, idx_ptr, w_ptr, counts_ptr, rank_ptr,
                  N, scale, E: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, E)
    s = tl.load(logits_ptr + t * E + offs).to(tl.float32)
    b = tl.load(bias_ptr + offs).to(tl.float32)
    work = tl.sigmoid(s) + b

    m0 = tl.argmax(work, axis=0); v0 = tl.max(work, axis=0); work = tl.where(offs == m0, -1e30, work)
    m1 = tl.argmax(work, axis=0); v1 = tl.max(work, axis=0); work = tl.where(offs == m1, -1e30, work)
    m2 = tl.argmax(work, axis=0); v2 = tl.max(work, axis=0); work = tl.where(offs == m2, -1e30, work)
    m3 = tl.argmax(work, axis=0); v3 = tl.max(work, axis=0); work = tl.where(offs == m3, -1e30, work)
    m4 = tl.argmax(work, axis=0); v4 = tl.max(work, axis=0); work = tl.where(offs == m4, -1e30, work)
    m5 = tl.argmax(work, axis=0); v5 = tl.max(work, axis=0); work = tl.where(offs == m5, -1e30, work)
    m6 = tl.argmax(work, axis=0); v6 = tl.max(work, axis=0); work = tl.where(offs == m6, -1e30, work)
    m7 = tl.argmax(work, axis=0); v7 = tl.max(work, axis=0)

    denom = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + 1e-20
    f = scale / denom


    r0 = tl.atomic_add(counts_ptr + m0, 1)
    r1 = tl.atomic_add(counts_ptr + m1, 1)
    r2 = tl.atomic_add(counts_ptr + m2, 1)
    r3 = tl.atomic_add(counts_ptr + m3, 1)
    r4 = tl.atomic_add(counts_ptr + m4, 1)
    r5 = tl.atomic_add(counts_ptr + m5, 1)
    r6 = tl.atomic_add(counts_ptr + m6, 1)
    r7 = tl.atomic_add(counts_ptr + m7, 1)

    base = t * K
    tl.store(idx_ptr + base + 0, m0.to(tl.int32)); tl.store(w_ptr + base + 0, (v0 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 0, r0)
    tl.store(idx_ptr + base + 1, m1.to(tl.int32)); tl.store(w_ptr + base + 1, (v1 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 1, r1)
    tl.store(idx_ptr + base + 2, m2.to(tl.int32)); tl.store(w_ptr + base + 2, (v2 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 2, r2)
    tl.store(idx_ptr + base + 3, m3.to(tl.int32)); tl.store(w_ptr + base + 3, (v3 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 3, r3)
    tl.store(idx_ptr + base + 4, m4.to(tl.int32)); tl.store(w_ptr + base + 4, (v4 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 4, r4)
    tl.store(idx_ptr + base + 5, m5.to(tl.int32)); tl.store(w_ptr + base + 5, (v5 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 5, r5)
    tl.store(idx_ptr + base + 6, m6.to(tl.int32)); tl.store(w_ptr + base + 6, (v6 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 6, r6)
    tl.store(idx_ptr + base + 7, m7.to(tl.int32)); tl.store(w_ptr + base + 7, (v7 * f).to(tl.bfloat16)); tl.store(rank_ptr + base + 7, r7)


def _route(logits, bias, scale):
    N = logits.shape[0]
    dev = logits.device
    idx = torch.empty((N, _K), dtype=torch.int32, device=dev)
    w = torch.empty((N, _K), dtype=torch.bfloat16, device=dev)
    counts = torch.zeros(_E, dtype=torch.int32, device=dev)
    rank = torch.empty((N, _K), dtype=torch.int32, device=dev)
    _route_kernel[(N,)](logits, bias, idx, w, counts, rank, N, float(scale),
                        E=_E, K=_K, num_warps=4)
    return idx, w, counts, rank


@triton.jit
def _router_gemm_kernel(a_ptr, b_ptr, c_ptr, N, H: tl.constexpr, E: tl.constexpr,
                        BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    re = tl.arange(0, E)
    rk = tl.arange(0, BK)
    mask_m = rm[:, None] < N
    a_row = rm[:, None].to(tl.int64) * H
    acc = tl.zeros((BM, E), dtype=tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + rk
        a = tl.load(a_ptr + a_row + kk[None, :], mask=mask_m, other=0.0)
        b = tl.load(b_ptr + re[None, :].to(tl.int64) * H + kk[:, None])
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None].to(tl.int64) * E + re[None, :], acc, mask=mask_m)


def _router_gemm(hidden, rw):
    N = hidden.shape[0]
    logits = torch.empty((N, _E), dtype=torch.float32, device=hidden.device)
    BM = 64
    _router_gemm_kernel[(triton.cdiv(N, BM),)](hidden, rw, logits, N,
                                               H=_H, E=_E, BM=BM, BK=128, num_warps=8)
    return logits


@triton.jit
def _scatter_gather_kernel(idx_ptr, base_ptr, rank_ptr, hidden_ptr, gathered_ptr,
                           inv_ptr, H: tl.constexpr, K: tl.constexpr, BLK: tl.constexpr):
    t = tl.program_id(0)
    b = t * K


    e0 = tl.load(idx_ptr + b + 0); p0 = tl.load(base_ptr + e0) + tl.load(rank_ptr + b + 0)
    e1 = tl.load(idx_ptr + b + 1); p1 = tl.load(base_ptr + e1) + tl.load(rank_ptr + b + 1)
    e2 = tl.load(idx_ptr + b + 2); p2 = tl.load(base_ptr + e2) + tl.load(rank_ptr + b + 2)
    e3 = tl.load(idx_ptr + b + 3); p3 = tl.load(base_ptr + e3) + tl.load(rank_ptr + b + 3)
    e4 = tl.load(idx_ptr + b + 4); p4 = tl.load(base_ptr + e4) + tl.load(rank_ptr + b + 4)
    e5 = tl.load(idx_ptr + b + 5); p5 = tl.load(base_ptr + e5) + tl.load(rank_ptr + b + 5)
    e6 = tl.load(idx_ptr + b + 6); p6 = tl.load(base_ptr + e6) + tl.load(rank_ptr + b + 6)
    e7 = tl.load(idx_ptr + b + 7); p7 = tl.load(base_ptr + e7) + tl.load(rank_ptr + b + 7)
    tl.store(inv_ptr + b + 0, p0.to(tl.int64)); tl.store(inv_ptr + b + 1, p1.to(tl.int64))
    tl.store(inv_ptr + b + 2, p2.to(tl.int64)); tl.store(inv_ptr + b + 3, p3.to(tl.int64))
    tl.store(inv_ptr + b + 4, p4.to(tl.int64)); tl.store(inv_ptr + b + 5, p5.to(tl.int64))
    tl.store(inv_ptr + b + 6, p6.to(tl.int64)); tl.store(inv_ptr + b + 7, p7.to(tl.int64))

    q0 = p0.to(tl.int64) * H; q1 = p1.to(tl.int64) * H; q2 = p2.to(tl.int64) * H; q3 = p3.to(tl.int64) * H
    q4 = p4.to(tl.int64) * H; q5 = p5.to(tl.int64) * H; q6 = p6.to(tl.int64) * H; q7 = p7.to(tl.int64) * H

    hbase = t.to(tl.int64) * H
    for hb in tl.static_range(0, H, BLK):
        cols = hb + tl.arange(0, BLK)
        v = tl.load(hidden_ptr + hbase + cols)
        tl.store(gathered_ptr + q0 + cols, v); tl.store(gathered_ptr + q1 + cols, v)
        tl.store(gathered_ptr + q2 + cols, v); tl.store(gathered_ptr + q3 + cols, v)
        tl.store(gathered_ptr + q4 + cols, v); tl.store(gathered_ptr + q5 + cols, v)
        tl.store(gathered_ptr + q6 + cols, v); tl.store(gathered_ptr + q7 + cols, v)


def _dispatch_gather(idx, counts, rank, N, hidden):
    dev = idx.device
    M = N * _K
    H = hidden.shape[1]
    offs = counts.cumsum(0).to(torch.int32)
    base = (offs - counts).to(torch.int32)
    inv = torch.empty(M, dtype=torch.int64, device=dev)
    gathered = torch.empty(M, H, dtype=hidden.dtype, device=dev)
    _scatter_gather_kernel[(N,)](idx, base, rank, hidden, gathered, inv,
                                 H=H, K=_K, BLK=1024, num_warps=4)
    return offs, gathered, inv


_FUSED_MAX = 1350
_BM = 128


@triton.jit
def _align_scatter_kernel(idx_ptr, padoff_ptr, base_ptr, rank_ptr,
                          src_ptr, cmap_ptr, inv_ptr, K: tl.constexpr):


    t = tl.program_id(0)
    b = t * K
    for k in tl.static_range(0, K):
        e = tl.load(idx_ptr + b + k)
        c = tl.load(rank_ptr + b + k)
        ppos = tl.load(padoff_ptr + e) + c
        cpos = tl.load(base_ptr + e) + c
        tl.store(src_ptr + ppos, t)
        tl.store(cmap_ptr + ppos, cpos)
        tl.store(inv_ptr + b + k, cpos.to(tl.int64))


@triton.autotune(
    configs=[
        triton.Config({'BN': 128, 'BK': 64},  num_warps=8, num_stages=2),
        triton.Config({'BN': 128, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BN': 256, 'BK': 64},  num_warps=8, num_stages=2),
        triton.Config({'BN': 128, 'BK': 64},  num_warps=8, num_stages=3),
        triton.Config({'BN': 128, 'BK': 64},  num_warps=8, num_stages=4),
        triton.Config({'BN': 64,  'BK': 128}, num_warps=8, num_stages=3),
        triton.Config({'BN': 128, 'BK': 128}, num_warps=4, num_stages=2),
    ],
    key=['I', 'E'],
)
@triton.jit
def _gate_up_fused_kernel(hidden_ptr, wg_ptr, wu_ptr, src_ptr, cmap_ptr, eob_ptr,
                          inter_ptr, H: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    blk = tl.program_id(0)
    e = tl.load(eob_ptr + blk)
    if e >= E:
        return
    ntile = tl.program_id(1)
    rows = blk * BM + tl.arange(0, BM)
    src = tl.load(src_ptr + rows)
    valid = src >= 0
    src_c = tl.where(valid, src, 0).to(tl.int64)
    crow = tl.load(cmap_ptr + rows, mask=valid, other=0)
    ncol = ntile * BN + tl.arange(0, BN)
    nmask = ncol < I
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    hrow = src_c[:, None] * H
    wbase = e.to(tl.int64) * I * H
    for k0 in tl.range(0, H, BK):
        kcol = k0 + tl.arange(0, BK)
        a = tl.load(hidden_ptr + hrow + kcol[None, :],
                    mask=valid[:, None], other=0.0)
        woff = wbase + ncol[None, :] * H + kcol[:, None]
        wg = tl.load(wg_ptr + woff, mask=nmask[None, :], other=0.0)
        wu = tl.load(wu_ptr + woff, mask=nmask[None, :], other=0.0)
        acc_g += tl.dot(a, wg)
        acc_u += tl.dot(a, wu)

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    o = (s * u).to(tl.bfloat16)
    irow = crow.to(tl.int64) * I
    tl.store(inter_ptr + irow[:, None] + ncol[None, :], o,
             mask=valid[:, None] & nmask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BN': 128, 'BK': 64},  num_warps=8, num_stages=2),
        triton.Config({'BN': 128, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BN': 256, 'BK': 64},  num_warps=8, num_stages=2),
        triton.Config({'BN': 128, 'BK': 128}, num_warps=8, num_stages=3),
        triton.Config({'BN': 256, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BN': 128, 'BK': 64},  num_warps=4, num_stages=3),

        triton.Config({'BN': 256, 'BK': 64},  num_warps=8, num_stages=3),
        triton.Config({'BN': 256, 'BK': 64},  num_warps=8, num_stages=4),
    ],
    key=['H', 'I'],
)
@triton.jit
def _down_fused_kernel(inter_ptr, wd_ptr, src_ptr, cmap_ptr, eob_ptr, out_ptr,
                       H: tl.constexpr, I: tl.constexpr, E: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    blk = tl.program_id(0)
    e = tl.load(eob_ptr + blk)
    if e >= E:
        return
    ntile = tl.program_id(1)
    rows = blk * BM + tl.arange(0, BM)
    src = tl.load(src_ptr + rows)
    valid = src >= 0
    crow = tl.load(cmap_ptr + rows, mask=valid, other=0).to(tl.int64)
    ncol = ntile * BN + tl.arange(0, BN)
    nmask = ncol < H
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    irow = crow[:, None] * I
    wbase = e.to(tl.int64) * H * I
    for k0 in tl.range(0, I, BK):
        kcol = k0 + tl.arange(0, BK)
        a = tl.load(inter_ptr + irow + kcol[None, :],
                    mask=valid[:, None], other=0.0)
        woff = wbase + ncol[None, :] * I + kcol[:, None]
        w = tl.load(wd_ptr + woff, mask=nmask[None, :], other=0.0)
        acc += tl.dot(a, w)
    o = acc.to(tl.bfloat16)
    orow = crow[:, None] * H
    tl.store(out_ptr + orow + ncol[None, :], o,
             mask=valid[:, None] & nmask[None, :])


@triton.jit
def _prep_index_kernel(counts_ptr, base_ptr, padoff_ptr, cumblk_ptr,
                       E: tl.constexpr, BM: tl.constexpr):
    e = tl.arange(0, E)
    c = tl.load(counts_ptr + e).to(tl.int32)
    base = tl.cumsum(c, axis=0) - c
    pc = ((c + (BM - 1)) // BM) * BM
    padoff = tl.cumsum(pc, axis=0) - pc
    blocks = pc // BM
    cumblk = tl.cumsum(blocks.to(tl.int64), axis=0)
    tl.store(base_ptr + e, base)
    tl.store(padoff_ptr + e, padoff)
    tl.store(cumblk_ptr + e, cumblk)


@triton.jit
def _eob_fill_kernel(cumblk_ptr, src_ptr, cmap_ptr, eob_ptr,
                     E: tl.constexpr, BM: tl.constexpr):
    blk = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, E)
    cb = tl.load(cumblk_ptr + e)
    eob = tl.sum((cb <= blk).to(tl.int32), axis=0)
    tl.store(eob_ptr + blk, eob)
    rows = blk * BM + tl.arange(0, BM)
    tl.store(src_ptr + rows, tl.full((BM,), -1, tl.int32))
    tl.store(cmap_ptr + rows, tl.zeros((BM,), tl.int32))


def _moe_forward_fused(
    hidden_states, router_weight,
    expert_gate_weights, expert_up_weights, expert_down_weights,
    shared_gate_weight, shared_up_weight, shared_down_weight,
    e_score_correction_bias, routed_scaling_factor,
):
    N, H = hidden_states.shape
    I = _I
    dev = hidden_states.device
    dt = hidden_states.dtype
    BM = _BM
    M = N * _K


    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        shared_out = _shared_expert(hidden_states, shared_gate_weight,
                                    shared_up_weight, shared_down_weight)


    router_logits = _router_gemm(hidden_states, router_weight)
    topk_idx, topk_w, counts, topk_rank = _route(router_logits,
                                                 e_score_correction_bias.float(),
                                                 float(routed_scaling_factor))


    base = torch.empty(_E, dtype=torch.int32, device=dev)
    pad_offs = torch.empty(_E, dtype=torch.int32, device=dev)
    cum_blocks = torch.empty(_E, dtype=torch.int64, device=dev)
    _prep_index_kernel[(1,)](counts, base, pad_offs, cum_blocks, E=_E, BM=BM)
    M_max = M + _E * BM
    NB_max = M_max // BM


    src = torch.empty(M_max, dtype=torch.int32, device=dev)
    cmap = torch.empty(M_max, dtype=torch.int32, device=dev)
    eob = torch.empty(NB_max, dtype=torch.int32, device=dev)
    _eob_fill_kernel[(NB_max,)](cum_blocks, src, cmap, eob, E=_E, BM=BM)
    inv = torch.empty(M, dtype=torch.int64, device=dev)
    _align_scatter_kernel[(N,)](topk_idx, pad_offs, base, topk_rank,
                                src, cmap, inv, K=_K, num_warps=4)


    inter = torch.empty(M, I, dtype=dt, device=dev)
    grid1 = lambda META: (NB_max, triton.cdiv(I, META['BN']))
    _gate_up_fused_kernel[grid1](hidden_states, expert_gate_weights, expert_up_weights,
                                 src, cmap, eob, inter,
                                 H=H, I=I, E=_E, BM=BM)


    expert_out = torch.empty(M, H, dtype=dt, device=dev)
    grid2 = lambda META: (NB_max, triton.cdiv(H, META['BN']))
    _down_fused_kernel[grid2](inter, expert_down_weights, src, cmap, eob, expert_out,
                              H=H, I=I, E=_E, BM=BM)


    main.wait_stream(side)
    output = torch.empty(N, H, dtype=dt, device=dev)
    BLOCK = 1024
    grid = (N, triton.cdiv(H, BLOCK))
    _undispatch_kernel[grid](expert_out, inv, topk_w.reshape(-1), shared_out, output,
                             H, K=_K, BLOCK=BLOCK)
    return output


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    o = s * u
    tl.store(o_ptr + offs, o.to(tl.bfloat16), mask=m)


def _silu_mul(gate, up):
    o = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](gate, up, o, n, BLOCK=BLOCK, num_warps=8)
    return o


@triton.jit
def _undispatch_kernel(eo_ptr, inv_ptr, w_ptr, shared_ptr, out_ptr, H,
                       K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cblk = tl.program_id(1)
    cols = cblk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = t * K
    for k in range(K):
        src = tl.load(inv_ptr + base + k)
        w = tl.load(w_ptr + base + k).to(tl.float32)
        v = tl.load(eo_ptr + src * H + cols, mask=mask, other=0.0).to(tl.float32)
        acc += w * v
    sh = tl.load(shared_ptr + t * H + cols, mask=mask, other=0.0).to(tl.float32)
    acc += sh
    tl.store(out_ptr + t * H + cols, acc.to(tl.bfloat16), mask=mask)


def _shared_expert(hs, w_gate, w_up, w_down):
    g = torch.mm(hs, w_gate.t())
    u = torch.mm(hs, w_up.t())
    inter = _silu_mul(g, u)
    return torch.mm(inter, w_down.t())


def _moe_forward(
    hidden_states,
    router_weight,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    e_score_correction_bias,
    routed_scaling_factor,
    overlap=True,
):
    N, H = hidden_states.shape
    if N <= _FUSED_MAX:
        return _moe_forward_fused(
            hidden_states, router_weight, expert_gate_weights, expert_up_weights,
            expert_down_weights, shared_gate_weight, shared_up_weight,
            shared_down_weight, e_score_correction_bias, routed_scaling_factor)
    dev = hidden_states.device
    dt = hidden_states.dtype


    main = torch.cuda.current_stream()
    side = None
    shared_out = None
    if overlap:
        side = torch.cuda.Stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared_out = _shared_expert(hidden_states, shared_gate_weight,
                                        shared_up_weight, shared_down_weight)


    router_logits = _router_gemm(hidden_states, router_weight)


    topk_idx, topk_w, counts, topk_rank = _route(router_logits,
                                                 e_score_correction_bias.float(),
                                                 float(routed_scaling_factor))


    offs, gathered, inv = _dispatch_gather(topk_idx, counts, topk_rank, N, hidden_states)


    wg = expert_gate_weights.transpose(1, 2)
    wu = expert_up_weights.transpose(1, 2)
    if N <= _OVERLAP_MAX:


        gu_side = torch.cuda.Stream()
        gu_side.wait_stream(main)
        with torch.cuda.stream(gu_side):
            up_out = torch._grouped_mm(gathered, wu, offs=offs)
        gate_out = torch._grouped_mm(gathered, wg, offs=offs)
        main.wait_stream(gu_side)
    else:
        gate_out = torch._grouped_mm(gathered, wg, offs=offs)
        up_out = torch._grouped_mm(gathered, wu, offs=offs)
    inter = _silu_mul(gate_out, up_out)


    wd = expert_down_weights.transpose(1, 2)
    expert_out = torch._grouped_mm(inter, wd, offs=offs)


    if not overlap:
        shared_out = _shared_expert(hidden_states, shared_gate_weight,
                                    shared_up_weight, shared_down_weight)
    else:
        main.wait_stream(side)


    output = torch.empty(N, H, dtype=dt, device=dev)
    BLOCK = 1024
    grid = (N, triton.cdiv(H, BLOCK))
    _undispatch_kernel[grid](expert_out, inv, topk_w.reshape(-1), shared_out, output,
                             H, K=_K, BLOCK=BLOCK)
    return output


_STATE = {}


def _ptr_key(args):
    return tuple(a.data_ptr() for a in args if torch.is_tensor(a))


def _capture_once(args, overlap):
    ws = torch.cuda.Stream()
    ws.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ws):
        for _ in range(3):
            _moe_forward(*args, overlap=overlap)
    torch.cuda.current_stream().wait_stream(ws)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _moe_forward(*args, overlap=overlap)
    return g, out


def _capture(args):
    try:
        return _capture_once(args, overlap=True)
    except Exception:
        torch.cuda.synchronize()
        return _capture_once(args, overlap=False)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    args = (hidden_states, router_weight, expert_gate_weights, expert_up_weights,
            expert_down_weights, shared_gate_weight, shared_up_weight,
            shared_down_weight, e_score_correction_bias, routed_scaling_factor)
    N = hidden_states.shape[0]
    key = _ptr_key(args)
    st = _STATE.get(N)

    if st is not None and st["mode"] == "graph" and st["key"] == key:
        st["graph"].replay()
        return st["out"]

    if st is not None and st["mode"] == "nograph" and st["key"] == key:
        return _moe_forward(*args)

    if st is None or st["key"] != key:
        _STATE[N] = {"mode": "pending", "key": key}
        return _moe_forward(*args)

    try:
        g, out = _capture(args)
        _STATE[N] = {"mode": "graph", "key": key, "graph": g, "out": out}
        g.replay()
        return out
    except Exception:
        torch.cuda.synchronize()
        _STATE[N] = {"mode": "nograph", "key": key}
        return _moe_forward(*args)
