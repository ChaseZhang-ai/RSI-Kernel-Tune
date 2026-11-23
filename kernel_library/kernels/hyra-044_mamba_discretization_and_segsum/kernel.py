import torch
import triton
import triton.language as tl


CS = 256
H = 16
DS = 256


FUSE_THRESH = 6


BJ = 64


BI_THRESH = 24


@triton.jit
def _stats_kernel(dt_ptr, alog_ptr, acs_ptr, decay_ptr, NC,
                  CS: tl.constexpr, H: tl.constexpr, MINOR: tl.constexpr):
    pid = tl.program_id(0)
    p = pid // H
    h = pid % H
    i = tl.arange(0, CS)
    dtv = tl.load(dt_ptr + (p * CS + i) * H + h).to(tl.float32)
    sp = tl.maximum(dtv, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dtv)))
    a_log = tl.load(alog_ptr + h).to(tl.float32)
    A = -tl.exp(a_log)
    Ad = A * sp
    acs = tl.cumsum(Ad, axis=0)
    total = tl.sum(Ad, axis=0)
    if MINOR:
        off = (p * CS + i) * H + h
    else:
        b = p // NC
        c = p % NC
        off = ((b * H + h) * NC + c) * CS + i
    tl.store(acs_ptr + off, acs)
    tl.store(decay_ptr + off, tl.exp(total - acs))


@triton.jit
def _write_kernel(gbc_ptr, acs_ptr, L_ptr, M_ptr, NC,
                  CS: tl.constexpr, H: tl.constexpr, BJ: tl.constexpr):
    njt = CS // BJ
    pid = tl.program_id(0)
    row = pid // njt
    jt = pid % njt
    p = row // CS
    i = row % CS
    j = jt * BJ + tl.arange(0, BJ)
    hh = tl.arange(0, H)

    out_off = row * (CS * H) + j[:, None] * H + hh[None, :]

    if jt * BJ > i:


        z = tl.zeros((BJ, H), dtype=tl.bfloat16)
        tl.store(L_ptr + out_off, z, cache_modifier=".cs")
        tl.store(M_ptr + out_off, z, cache_modifier=".cs")
    else:
        base_p = p * (CS * H)
        grow = tl.load(gbc_ptr + row * CS + j).to(tl.float32)
        acs_i = tl.load(acs_ptr + row * H + hh)
        acs_j = tl.load(acs_ptr + base_p + j[:, None] * H + hh[None, :])
        diff = acs_i[None, :] - acs_j

        diff = tl.minimum(diff, 0.0) * 1.4426950408889634
        maskCH = j[:, None] <= i
        Lv = tl.where(maskCH, tl.exp2(diff), 0.0)
        Gb = tl.broadcast_to(grow[:, None], (BJ, H))
        Mv = Gb * Lv

        tl.store(M_ptr + out_off, Mv.to(tl.bfloat16), cache_modifier=".cs")
        tl.store(L_ptr + out_off, Lv.to(tl.bfloat16), cache_modifier=".cs")


@triton.jit
def _write_kernel_bi(gbc_ptr, acs_ptr, L_ptr, M_ptr, NC,
                     CS: tl.constexpr, H: tl.constexpr,
                     BJ: tl.constexpr, BI: tl.constexpr):
    njt = CS // BJ
    nib = CS // BI
    pid = tl.program_id(0)
    jt = pid % njt
    t = pid // njt
    ib = t % nib
    p = t // nib
    i0 = ib * BI
    j = jt * BJ + tl.arange(0, BJ)
    hh = tl.arange(0, H)

    if jt * BJ > i0 + BI - 1:

        z = tl.zeros((BJ, H), dtype=tl.bfloat16)
        for r in tl.static_range(BI):
            row = p * CS + i0 + r
            out_off = row * (CS * H) + j[:, None] * H + hh[None, :]
            tl.store(L_ptr + out_off, z, cache_modifier=".cs")
            tl.store(M_ptr + out_off, z, cache_modifier=".cs")
    else:
        base_p = p * (CS * H)

        acs_j = tl.load(acs_ptr + base_p + j[:, None] * H + hh[None, :])
        for r in tl.static_range(BI):
            i = i0 + r
            row = p * CS + i
            grow = tl.load(gbc_ptr + row * CS + j).to(tl.float32)
            acs_i = tl.load(acs_ptr + row * H + hh)
            diff = acs_i[None, :] - acs_j
            diff = tl.minimum(diff, 0.0) * 1.4426950408889634
            maskCH = j[:, None] <= i
            Lv = tl.where(maskCH, tl.exp2(diff), 0.0)
            Gb = tl.broadcast_to(grow[:, None], (BJ, H))
            Mv = Gb * Lv
            out_off = row * (CS * H) + j[:, None] * H + hh[None, :]
            tl.store(M_ptr + out_off, Mv.to(tl.bfloat16), cache_modifier=".cs")
            tl.store(L_ptr + out_off, Lv.to(tl.bfloat16), cache_modifier=".cs")


@triton.jit
def _write_fused_kernel(C_ptr, B_ptr, acs_ptr, Gbc_ptr, L_ptr, M_ptr, NC,
                        CS: tl.constexpr, DS: tl.constexpr, H: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
    pid_bc = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    b = pid_bc // NC
    c = pid_bc % NC

    i = pid_i * BM + tl.arange(0, BM)
    j = pid_j * BN + tl.arange(0, BN)
    d = tl.arange(0, DS)
    hh = tl.arange(0, H)

    c_off = pid_bc * (CS * DS) + i[:, None] * DS + d[None, :]
    Ct = tl.load(C_ptr + c_off)
    b_off = pid_bc * (CS * DS) + j[None, :] * DS + d[:, None]
    Bt = tl.load(B_ptr + b_off)
    Gt = tl.dot(Ct, Bt)


    gbc_off = pid_bc * (CS * CS) + i[:, None] * CS + j[None, :]
    tl.store(Gbc_ptr + gbc_off, Gt.to(tl.bfloat16))


    off = (pid_bc * (CS * CS * H)
           + i[:, None, None] * (CS * H)
           + j[None, :, None] * H
           + hh[None, None, :])


    if pid_i * BM + (BM - 1) < pid_j * BN:
        z = tl.zeros((BM, BN, H), dtype=tl.bfloat16)
        tl.store(M_ptr + off, z)
        tl.store(L_ptr + off, z)
    else:

        base = (b * H * NC + c) * CS
        hstride = NC * CS
        acs_i = tl.load(acs_ptr + base + i[:, None] + hh[None, :] * hstride)
        acs_j = tl.load(acs_ptr + base + j[:, None] + hh[None, :] * hstride)

        seg = acs_i[:, None, :] - acs_j[None, :, :]
        seg = tl.minimum(seg, 0.0)
        mask = (i[:, None] >= j[None, :])
        Lf = tl.where(mask[:, :, None], tl.exp2(seg * 1.4426950408889634), 0.0)

        Go = tl.broadcast_to(Gt[:, :, None], (BM, BN, H))
        Mf = Go * Lf
        tl.store(M_ptr + off, Mf.to(tl.bfloat16))
        tl.store(L_ptr + off, Lf.to(tl.bfloat16))


@triton.jit
def _stats_gemm_kernel(dt_ptr, alog_ptr, acs_ptr, decay_ptr,
                       C_ptr, B_ptr, Gbc_ptr, N_STATS,
                       CS: tl.constexpr, H: tl.constexpr, DS: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    if pid < N_STATS:

        sp_ = pid // H
        h = pid % H
        si = tl.arange(0, CS)
        dtv = tl.load(dt_ptr + (sp_ * CS + si) * H + h).to(tl.float32)
        spv = tl.maximum(dtv, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dtv)))
        a_log = tl.load(alog_ptr + h).to(tl.float32)
        A = -tl.exp(a_log)
        Ad = A * spv
        acs = tl.cumsum(Ad, axis=0)
        total = tl.sum(Ad, axis=0)
        off = (sp_ * CS + si) * H + h
        tl.store(acs_ptr + off, acs)
        tl.store(decay_ptr + off, tl.exp(total - acs))
    else:

        g = pid - N_STATS
        nn = CS // BN
        nm = CS // BM
        gp = g // (nm * nn)
        r = g % (nm * nn)
        mi = r // nn
        nj = r % nn
        gi = mi * BM + tl.arange(0, BM)
        gj = nj * BN + tl.arange(0, BN)
        gd = tl.arange(0, DS)
        gbase = gp * (CS * DS)
        Ct = tl.load(C_ptr + gbase + gi[:, None] * DS + gd[None, :])
        Bt = tl.load(B_ptr + gbase + gj[None, :] * DS + gd[:, None])
        acc = tl.dot(Ct, Bt)
        g_off = gp * (CS * CS) + gi[:, None] * CS + gj[None, :]
        tl.store(Gbc_ptr + g_off, acc.to(tl.bfloat16))


SG_BM = 64
SG_BN = 64


@torch.no_grad()
def run(hidden_states, dt, A_log, B, C):
    batch, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    NC = num_chunks
    P = batch * num_chunks
    dev = dt.device

    dt = dt.contiguous()
    A_log = A_log.contiguous()


    L_phys = torch.empty((batch, NC, CS, CS, H), dtype=torch.bfloat16, device=dev)
    M = torch.empty((batch, NC, CS, CS, H), dtype=torch.bfloat16, device=dev)

    if P <= FUSE_THRESH:


        A_cumsum = torch.empty((batch, H, NC, CS), dtype=torch.float32, device=dev)
        decay_states = torch.empty((batch, H, NC, CS), dtype=torch.float32, device=dev)
        _stats_kernel[(P * H,)](dt, A_log, A_cumsum, decay_states, NC,
                                CS=CS, H=H, MINOR=False)
        Gbc = torch.empty((batch, NC, CS, CS), dtype=torch.bfloat16, device=dev)
        Cc = C.reshape(P, CS, DS).contiguous()
        Bc = B.reshape(P, CS, DS).contiguous()
        BM, BN = 16, 32
        grid = (P, CS // BM, CS // BN)
        _write_fused_kernel[grid](Cc, Bc, A_cumsum, Gbc, L_phys, M, NC,
                                  CS=CS, DS=DS, H=H, BM=BM, BN=BN, num_warps=4)
        G = Gbc.unsqueeze(-1).expand(batch, NC, CS, CS, H)
    else:


        A_cumsum_phys = torch.empty((batch, NC, CS, H), dtype=torch.float32, device=dev)
        decay_phys = torch.empty((batch, NC, CS, H), dtype=torch.float32, device=dev)
        Bc = B.reshape(P, CS, DS).contiguous()
        Cc = C.reshape(P, CS, DS).contiguous()
        njt = CS // BJ
        if P < BI_THRESH:


            Gbc = torch.empty((batch, NC, CS, CS), dtype=torch.bfloat16, device=dev)
            _stats_gemm_kernel[(P * H + P * (CS // SG_BM) * (CS // SG_BN),)](
                dt, A_log, A_cumsum_phys, decay_phys, Cc, Bc, Gbc, P * H,
                CS=CS, H=H, DS=DS, BM=SG_BM, BN=SG_BN, num_warps=4)
            _write_kernel[(P * CS * njt,)](Gbc.view(P, CS, CS), A_cumsum_phys,
                                           L_phys, M, NC, CS=CS, H=H, BJ=BJ, num_warps=4)
        else:


            _stats_kernel[(P * H,)](dt, A_log, A_cumsum_phys, decay_phys, NC,
                                    CS=CS, H=H, MINOR=True)
            Gbc = torch.bmm(Cc, Bc.transpose(1, 2))


            BI = 4

            _write_kernel_bi[(P * (CS // BI) * njt,)](Gbc, A_cumsum_phys, L_phys, M, NC,
                                                      CS=CS, H=H, BJ=BJ, BI=BI, num_warps=4)
        G = Gbc.view(batch, NC, CS, CS, 1).expand(batch, NC, CS, CS, H)
        A_cumsum = A_cumsum_phys.permute(0, 3, 1, 2)
        decay_states = decay_phys.permute(0, 3, 1, 2)

    L = L_phys.permute(0, 4, 1, 2, 3)

    return L, G, M, A_cumsum, decay_states
