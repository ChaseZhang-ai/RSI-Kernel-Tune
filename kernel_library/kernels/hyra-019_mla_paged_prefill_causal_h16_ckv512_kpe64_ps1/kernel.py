import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _mla_prefill_packed(
    qn_ptr, qp_ptr, ckv_ptr, kpe_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
    out_ptr, lse_ptr, scale2,
    stride_qn_r, stride_qp_r, stride_ckv_p, stride_kpe_p, stride_o_r,
    batch,
    H: tl.constexpr, D: tl.constexpr, DPE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, MAX_BATCH: tl.constexpr,
):
    pid = tl.program_id(0)


    acc_blk = 0
    my_b = -1
    my_local = 0
    for b in tl.static_range(MAX_BATCH):
        valid_b = b < batch
        qs = tl.load(qo_indptr_ptr + b, mask=valid_b, other=0)
        qe = tl.load(qo_indptr_ptr + b + 1, mask=valid_b, other=0)
        seqq = qe - qs
        nblk = (seqq + (BLOCK_T - 1)) // BLOCK_T
        hit = valid_b & (my_b < 0) & (pid >= acc_blk) & (pid < acc_blk + nblk)
        my_b = tl.where(hit, b, my_b)
        my_local = tl.where(hit, pid - acc_blk, my_local)
        acc_blk += tl.where(valid_b, nblk, 0)

    if my_b < 0:
        return

    q_start = tl.load(qo_indptr_ptr + my_b)
    q_end = tl.load(qo_indptr_ptr + my_b + 1)
    kv_beg = tl.load(kv_indptr_ptr + my_b)
    kv_end = tl.load(kv_indptr_ptr + my_b + 1)
    q_len = q_end - q_start
    kv_len = kv_end - kv_beg
    delta = kv_len - q_len

    t0 = my_local * BLOCK_T
    M: tl.constexpr = BLOCK_T * H
    r = tl.arange(0, M)
    tok = t0 + r // H
    m_valid = tok < q_len
    base_row = (q_start + t0) * H
    rows = base_row + r

    offs_d = tl.arange(0, D)
    offs_dpe = tl.arange(0, DPE)

    qn = tl.load(qn_ptr + rows[:, None] * stride_qn_r + offs_d[None, :],
                 mask=m_valid[:, None], other=0.0)
    qp = tl.load(qp_ptr + rows[:, None] * stride_qp_r + offs_dpe[None, :],
                 mask=m_valid[:, None], other=0.0)

    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, D], tl.float32)

    last_tok = tl.minimum(t0 + BLOCK_T - 1, q_len - 1)
    hi = tl.minimum(kv_len, last_tok + delta + 1)
    hi = tl.maximum(hi, 0)

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < kv_len
        page = tl.load(kv_indices_ptr + kv_beg + offs_n, mask=n_valid, other=0).to(tl.int64)
        kc = tl.load(ckv_ptr + page[:, None] * stride_ckv_p + offs_d[None, :],
                     mask=n_valid[:, None], other=0.0)
        kp = tl.load(kpe_ptr + page[:, None] * stride_kpe_p + offs_dpe[None, :],
                     mask=n_valid[:, None], other=0.0)

        qk = tl.dot(qn, tl.trans(kc))
        qk += tl.dot(qp, tl.trans(kp))
        qk = qk * scale2
        causal = offs_n[None, :] <= (tok[:, None] + delta)
        qk = tl.where(causal & n_valid[None, :], qk, -float("inf"))

        p = tl.exp2(qk)
        l_i += tl.sum(p, axis=1)
        acc += tl.dot(p.to(kc.dtype), kc)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(out_ptr + rows[:, None] * stride_o_r + offs_d[None, :],
             out.to(tl.bfloat16), mask=m_valid[:, None])
    lse_v = tl.where(l_i > 0.0, tl.log2(l_i), -float("inf"))
    tl.store(lse_ptr + rows, lse_v, mask=m_valid)


def _next_pow2(x):
    return 1 << max(0, (max(1, int(x)) - 1)).bit_length()


def _pick_cfg(total_q, batch):


    if total_q >= 512:
        return 2, 64, 4, 2
    return 1, 64, 4, 2


def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
                kv_indices, sm_scale):
    total_q, H, D = q_nope.shape
    DPE = q_pe.shape[-1]
    B = qo_indptr.shape[0] - 1
    device = q_nope.device

    out = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    if total_q == 0 or B <= 0:
        return out, lse

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = sm_scale.item()
    scale2 = float(sm_scale) * LOG2E

    qn = q_nope.reshape(total_q * H, D)
    qp = q_pe.reshape(total_q * H, DPE)
    out_flat = out.reshape(total_q * H, D)
    lse_flat = lse.reshape(total_q * H)

    ckv = ckv_cache.squeeze(1)
    kpe = kpe_cache.squeeze(1)

    if qo_indptr.dtype != torch.int32:
        qo_indptr = qo_indptr.to(torch.int32)
    if kv_indptr.dtype != torch.int32:
        kv_indptr = kv_indptr.to(torch.int32)
    if kv_indices.dtype != torch.int32:
        kv_indices = kv_indices.to(torch.int32)

    BLOCK_T, BLOCK_N, num_warps, num_stages = _pick_cfg(total_q, B)

    MAX_BATCH = _next_pow2(B)
    UB = triton.cdiv(total_q, BLOCK_T) + B
    grid = (UB,)

    _mla_prefill_packed[grid](
        qn, qp, ckv, kpe,
        qo_indptr, kv_indptr, kv_indices,
        out_flat, lse_flat, scale2,
        qn.stride(0), qp.stride(0), ckv.stride(0), kpe.stride(0), out_flat.stride(0),
        B,
        H=H, D=D, DPE=DPE,
        BLOCK_T=BLOCK_T, BLOCK_N=BLOCK_N, MAX_BATCH=MAX_BATCH,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


_WORKSPACE = None
_WRAPPER_CACHE = {}


def _get_workspace(device):
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    return _WORKSPACE


def _run_fa2(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
             kv_indices, sm_scale):
    import flashinfer
    total_q = q_nope.shape[0]
    num_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    page_size = ckv_cache.shape[1]
    batch_p1 = qo_indptr.shape[0]
    num_kv = kv_indices.shape[0]
    scale = float(sm_scale)

    key = (total_q, num_heads, head_dim_ckv, head_dim_kpe,
           page_size, batch_p1, num_kv, scale)

    wrapper = _WRAPPER_CACHE.get(key)
    if wrapper is None:
        device = q_nope.device
        ws = _get_workspace(device)
        qo_i = qo_indptr.to(torch.int32)
        kv_i = kv_indptr.to(torch.int32)
        kv_idx = kv_indices.to(torch.int32)
        kv_len_arr = (kv_i[1:] - kv_i[:-1]).to(torch.int32)

        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend="fa2")
        wrapper.plan(
            qo_i, kv_i, kv_idx, kv_len_arr,
            num_heads, head_dim_ckv, head_dim_kpe, page_size,
            True,
            scale,
            q_nope.dtype, ckv_cache.dtype,
        )
        _WRAPPER_CACHE[key] = wrapper

    out, lse = wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=True)
    return out, lse


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q_nope.shape[0]
    num_kv = kv_indices.shape[0]


    return _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr,
                       kv_indptr, kv_indices, sm_scale)
