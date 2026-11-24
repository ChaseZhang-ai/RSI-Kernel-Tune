import torch
import triton
import triton.language as tl


_CHUNK = 64


@triton.jit
def _gdr_fused(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale,
    T, NC,
    HK: tl.constexpr, HV: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    s_ob, s_ot, s_oh, s_od,
    EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // HV
    h = pid_bh % HV
    kh = h % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = pid_v * BV + tl.arange(0, BV)

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh
    v_base = v_ptr + b * s_vb + h * s_vh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_gb + h * s_gh
    o_base = o_ptr + b * s_ob + h * s_oh

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    S = tl.zeros([DK, BV], dtype=tl.float32)

    for ci in range(0, NC):
        toks = ci * C + rk
        row_mask = toks < T

        q = tl.load(q_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_base + toks[:, None] * s_vt + dv[None, :] * s_vd,
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)

        q = q * (tl.rsqrt(tl.sum(q * q, axis=1) + EPS) * scale)[:, None]
        k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
        qbf = q.to(tl.bfloat16)
        kbf = k.to(tl.bfloat16)

        gcum = tl.cumsum(g, axis=0)
        dm = tl.where(tril_incl, tl.exp(gcum[:, None] - gcum[None, :]), 0.0)

        kb = (k * beta[:, None]).to(tl.bfloat16)
        vb = (v * beta[:, None]).to(tl.bfloat16)

        kkt = tl.dot(kb, tl.trans(kbf))
        Bm = tl.where(strict_lower, -kkt * dm, 0.0)

        Tm = ident + Bm

        Tmb = Tm.to(tl.bfloat16)
        ge = tl.exp(g)
        kbg = (kb.to(tl.float32) * ge[:, None]).to(tl.bfloat16)

        ai = (tl.dot(qbf, tl.trans(kbf)) * dm).to(tl.bfloat16)
        qg = (q * ge[:, None]).to(tl.bfloat16)

        glast = tl.sum(tl.where(last_sel, g, 0.0))
        td = tl.exp(glast - g)
        Kd = (k * td[:, None]).to(tl.bfloat16)
        a = tl.exp(glast)

        Sb = S.to(tl.bfloat16)
        diff = vb.to(tl.float32) - tl.dot(kbg, Sb)
        vn = tl.dot(Tmb, diff.to(tl.bfloat16))
        vnb = vn.to(tl.bfloat16)
        o = tl.dot(qg, Sb) + tl.dot(ai, vnb)
        tl.store(o_base + toks[:, None] * s_ot + dv[None, :] * s_od,
                 o.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])

        S = a * S + tl.dot(tl.trans(Kd), vnb)


@triton.jit
def _gdr_fused_nc2(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale,
    T, NC,
    HK: tl.constexpr, HV: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    s_ob, s_ot, s_oh, s_od,
    EPS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // HV
    h = pid_bh % HV
    kh = h % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = pid_v * BV + tl.arange(0, BV)

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh
    v_base = v_ptr + b * s_vb + h * s_vh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_gb + h * s_gh
    o_base = o_ptr + b * s_ob + h * s_oh

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))


    toks = rk
    row_mask = toks < T
    q = tl.load(q_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_base + toks[:, None] * s_vt + dv[None, :] * s_vd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    g = tl.load(g_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)

    q = q * (tl.rsqrt(tl.sum(q * q, axis=1) + EPS) * scale)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
    qbf = q.to(tl.bfloat16)
    kbf = k.to(tl.bfloat16)

    gcum = tl.cumsum(g, axis=0)
    dm = tl.where(tril_incl, tl.exp(gcum[:, None] - gcum[None, :]), 0.0)

    kb = (k * beta[:, None]).to(tl.bfloat16)
    vb = (v * beta[:, None]).to(tl.bfloat16)

    kkt = tl.dot(kb, tl.trans(kbf))
    Bm = tl.where(strict_lower, -kkt * dm, 0.0)
    Tmb = (ident + Bm).to(tl.bfloat16)
    ai = (tl.dot(qbf, tl.trans(kbf)) * dm).to(tl.bfloat16)

    glast = tl.sum(tl.where(last_sel, g, 0.0))
    td = tl.exp(glast - g)
    Kd = (k * td[:, None]).to(tl.bfloat16)

    vnb = tl.dot(Tmb, vb).to(tl.bfloat16)
    o = tl.dot(ai, vnb)
    tl.store(o_base + toks[:, None] * s_ot + dv[None, :] * s_od,
             o.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])
    S = tl.dot(tl.trans(Kd), vnb)


    toks1 = C + rk
    row_mask1 = toks1 < T
    q1 = tl.load(q_base + toks1[:, None] * s_qt + dk[None, :] * s_qd,
                 mask=row_mask1[:, None], other=0.0).to(tl.float32)
    k1 = tl.load(k_base + toks1[:, None] * s_qt + dk[None, :] * s_qd,
                 mask=row_mask1[:, None], other=0.0).to(tl.float32)
    v1 = tl.load(v_base + toks1[:, None] * s_vt + dv[None, :] * s_vd,
                 mask=row_mask1[:, None], other=0.0).to(tl.float32)
    g1 = tl.load(g_base + toks1 * s_gt, mask=row_mask1, other=0.0).to(tl.float32)
    beta1 = tl.load(beta_base + toks1 * s_gt, mask=row_mask1, other=0.0).to(tl.float32)

    q1 = q1 * (tl.rsqrt(tl.sum(q1 * q1, axis=1) + EPS) * scale)[:, None]
    k1 = k1 * tl.rsqrt(tl.sum(k1 * k1, axis=1) + EPS)[:, None]
    qbf1 = q1.to(tl.bfloat16)
    kbf1 = k1.to(tl.bfloat16)

    gcum1 = tl.cumsum(g1, axis=0)
    dm1 = tl.where(tril_incl, tl.exp(gcum1[:, None] - gcum1[None, :]), 0.0)

    kb1 = (k1 * beta1[:, None]).to(tl.bfloat16)
    vb1 = (v1 * beta1[:, None]).to(tl.bfloat16)

    kkt1 = tl.dot(kb1, tl.trans(kbf1))
    Bm1 = tl.where(strict_lower, -kkt1 * dm1, 0.0)
    Tmb1 = (ident + Bm1).to(tl.bfloat16)
    ge1 = tl.exp(g1)
    kbg1 = (kb1.to(tl.float32) * ge1[:, None]).to(tl.bfloat16)
    ai1 = (tl.dot(qbf1, tl.trans(kbf1)) * dm1).to(tl.bfloat16)
    qg1 = (q1 * ge1[:, None]).to(tl.bfloat16)

    Sb = S.to(tl.bfloat16)
    diff1 = vb1.to(tl.float32) - tl.dot(kbg1, Sb)
    vnb1 = tl.dot(Tmb1, diff1.to(tl.bfloat16)).to(tl.bfloat16)
    o1 = tl.dot(qg1, Sb) + tl.dot(ai1, vnb1)
    tl.store(o_base + toks1[:, None] * s_ot + dv[None, :] * s_od,
             o1.to(o_ptr.dtype.element_ty), mask=row_mask1[:, None])


@triton.jit
def _gdr_prepare(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    W_ptr, Kcd_ptr, Ai_ptr, Qg_ptr, Kd_ptr, Aa_ptr,
    scale,
    T, NC,
    HK: tl.constexpr, HV: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    ci = pid % NC
    bh = pid // NC
    b = bh // HV
    h = bh % HV
    kh = h % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh
    v_base = v_ptr + b * s_vb + h * s_vh
    g_base = g_ptr + b * s_gb + h * s_gh
    beta_base = beta_ptr + b * s_gb + h * s_gh

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    toks = ci * C + rk
    row_mask = toks < T

    q = tl.load(q_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_base + toks[:, None] * s_vt + dv[None, :] * s_vd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    g = tl.load(g_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)

    q = q * (tl.rsqrt(tl.sum(q * q, axis=1) + EPS) * scale)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
    qbf = q.to(tl.bfloat16)
    kbf = k.to(tl.bfloat16)

    gcum = tl.cumsum(g, axis=0)
    dm = tl.where(tril_incl, tl.exp(gcum[:, None] - gcum[None, :]), 0.0)

    kb = (k * beta[:, None]).to(tl.bfloat16)
    vb = (v * beta[:, None]).to(tl.bfloat16)

    kkt = tl.dot(kb, tl.trans(kbf))
    Bm = tl.where(strict_lower, -kkt * dm, 0.0)

    Tm = ident + Bm

    Tmb = Tm.to(tl.bfloat16)
    ge = tl.exp(g)
    W = tl.dot(Tmb, vb)
    kbg = (kb.to(tl.float32) * ge[:, None]).to(tl.bfloat16)
    Kcd = tl.dot(Tmb, kbg)

    ai = tl.dot(qbf, tl.trans(kbf)) * dm
    qg = q * ge[:, None]

    glast = tl.sum(tl.where(last_sel, g, 0.0))
    td = tl.exp(glast - g)
    Kd = k * td[:, None]
    a = tl.exp(glast)


    is_last = ci == (NC - 1)
    ew = W_ptr.dtype.element_ty
    W_o = W_ptr + pid * C * DV
    tl.store(W_o + rk[:, None] * DV + dv[None, :], W.to(ew), mask=row_mask[:, None])
    Kcd_o = Kcd_ptr + pid * C * DK
    tl.store(Kcd_o + rk[:, None] * DK + dk[None, :], Kcd.to(ew), mask=row_mask[:, None])
    Ai_o = Ai_ptr + pid * C * C
    tl.store(Ai_o + rk[:, None] * C + rk[None, :], ai.to(ew), mask=row_mask[:, None])
    Qg_o = Qg_ptr + pid * C * DK
    tl.store(Qg_o + rk[:, None] * DK + dk[None, :], qg.to(ew), mask=row_mask[:, None])
    if not is_last:
        Kd_o = Kd_ptr + pid * C * DK
        tl.store(Kd_o + rk[:, None] * DK + dk[None, :], Kd.to(ew), mask=row_mask[:, None])
        tl.store(Aa_ptr + pid, a)


@triton.jit
def _gdr_prepare_gqa(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    W_ptr, Kcd_ptr, Ai_ptr, Qg_ptr, Kd_ptr, Aa_ptr,
    scale,
    T, NC,
    HK: tl.constexpr, HV: tl.constexpr, GH: tl.constexpr, NGRP: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    ci = pid % NC
    rest = pid // NC
    grp = rest % NGRP
    bkh = rest // NGRP
    b = bkh // HK
    kh = bkh % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    toks = ci * C + rk
    row_mask = toks < T

    q = tl.load(q_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_base + toks[:, None] * s_qt + dk[None, :] * s_qd,
                mask=row_mask[:, None], other=0.0).to(tl.float32)
    q = q * (tl.rsqrt(tl.sum(q * q, axis=1) + EPS) * scale)[:, None]
    k = k * tl.rsqrt(tl.sum(k * k, axis=1) + EPS)[:, None]
    qbf = q.to(tl.bfloat16)
    kbf = k.to(tl.bfloat16)
    kbfT = tl.trans(kbf)
    KK = tl.dot(kbf, kbfT)
    QK = tl.dot(qbf, kbfT)

    ew = W_ptr.dtype.element_ty


    is_c0 = ci == 0


    is_last = ci == (NC - 1)
    for g_idx in range(0, GH):
        h = kh + (grp * GH + g_idx) * HK
        v_base = v_ptr + b * s_vb + h * s_vh
        g_base = g_ptr + b * s_gb + h * s_gh
        beta_base = beta_ptr + b * s_gb + h * s_gh

        v = tl.load(v_base + toks[:, None] * s_vt + dv[None, :] * s_vd,
                    mask=row_mask[:, None], other=0.0).to(tl.float32)
        gg = tl.load(g_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_base + toks * s_gt, mask=row_mask, other=0.0).to(tl.float32)

        gcum = tl.cumsum(gg, axis=0)
        dm = tl.where(tril_incl, tl.exp(gcum[:, None] - gcum[None, :]), 0.0)

        kkt = beta[:, None] * KK
        Bm = tl.where(strict_lower, -kkt * dm, 0.0)
        Tm = ident + Bm
        Tmb = Tm.to(tl.bfloat16)

        ge = tl.exp(gg)
        vb = (v * beta[:, None]).to(tl.bfloat16)

        ai = QK * dm

        glast = tl.sum(tl.where(last_sel, gg, 0.0))
        td = tl.exp(glast - gg)
        Kd = k * td[:, None]

        ps = (b * HV + h) * NC + ci


        tl.store(Ai_ptr + ps * C * C + rk[:, None] * C + rk[None, :], ai.to(ew), mask=row_mask[:, None])
        if not is_last:
            tl.store(Kd_ptr + ps * C * DK + rk[:, None] * DK + dk[None, :], Kd.to(ew), mask=row_mask[:, None])

        if is_c0:
            W = tl.dot(Tmb, vb)
            tl.store(W_ptr + ps * C * DV + rk[:, None] * DV + dv[None, :], W.to(ew), mask=row_mask[:, None])
        else:
            kbg = (k * (beta * ge)[:, None]).to(tl.bfloat16)

            VK = tl.reshape(tl.join(vb, kbg), [C, DV * 2])
            WKcd = tl.reshape(tl.dot(Tmb, VK), [C, DV, 2])
            W, Kcd = tl.split(WKcd)
            qg = q * ge[:, None]
            tl.store(W_ptr + ps * C * DV + rk[:, None] * DV + dv[None, :], W.to(ew), mask=row_mask[:, None])
            tl.store(Kcd_ptr + ps * C * DK + rk[:, None] * DK + dk[None, :], Kcd.to(ew), mask=row_mask[:, None])
            tl.store(Qg_ptr + ps * C * DK + rk[:, None] * DK + dk[None, :], qg.to(ew), mask=row_mask[:, None])
            if not is_last:
                a = tl.exp(glast)
                tl.store(Aa_ptr + ps, a)


@triton.jit
def _gdr_scan(
    W_ptr, Kcd_ptr, Ai_ptr, Qg_ptr, Kd_ptr, Aa_ptr, o_ptr,
    T, NC,
    HV: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    s_ob, s_ot, s_oh, s_od,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bh // HV
    h = pid_bh % HV

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = pid_v * BV + tl.arange(0, BV)

    o_base = o_ptr + b * s_ob + h * s_oh


    idx = pid_bh * NC
    toks = rk
    row_mask = toks < T
    W = tl.load(W_ptr + idx * C * DV + rk[:, None] * DV + dv[None, :],
                mask=row_mask[:, None], other=0.0)
    ai = tl.load(Ai_ptr + idx * C * C + rk[:, None] * C + rk[None, :],
                 mask=row_mask[:, None], other=0.0)
    Kd = tl.load(Kd_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                 mask=row_mask[:, None], other=0.0)
    vnb = W
    o = tl.dot(ai, vnb)
    tl.store(o_base + toks[:, None] * s_ot + dv[None, :] * s_od,
             o.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])
    S = tl.dot(tl.trans(Kd), vnb)


    for ci in range(1, NC - 1):
        toks = ci * C + rk
        row_mask = toks < T
        idx = pid_bh * NC + ci

        W = tl.load(W_ptr + idx * C * DV + rk[:, None] * DV + dv[None, :],
                    mask=row_mask[:, None], other=0.0)
        Kcd = tl.load(Kcd_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                      mask=row_mask[:, None], other=0.0)
        ai = tl.load(Ai_ptr + idx * C * C + rk[:, None] * C + rk[None, :],
                     mask=row_mask[:, None], other=0.0)
        qg = tl.load(Qg_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                     mask=row_mask[:, None], other=0.0)
        Kd = tl.load(Kd_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                     mask=row_mask[:, None], other=0.0)
        a = tl.load(Aa_ptr + idx)

        Sb = S.to(tl.bfloat16)
        vn = W.to(tl.float32) - tl.dot(Kcd, Sb)
        vnb = vn.to(tl.bfloat16)
        o = tl.dot(qg, Sb) + tl.dot(ai, vnb)
        tl.store(o_base + toks[:, None] * s_ot + dv[None, :] * s_od,
                 o.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])

        S = a * S + tl.dot(tl.trans(Kd), vnb)


    idx = pid_bh * NC + (NC - 1)
    toks = (NC - 1) * C + rk
    row_mask = toks < T
    W = tl.load(W_ptr + idx * C * DV + rk[:, None] * DV + dv[None, :],
                mask=row_mask[:, None], other=0.0)
    Kcd = tl.load(Kcd_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                  mask=row_mask[:, None], other=0.0)
    ai = tl.load(Ai_ptr + idx * C * C + rk[:, None] * C + rk[None, :],
                 mask=row_mask[:, None], other=0.0)
    qg = tl.load(Qg_ptr + idx * C * DK + rk[:, None] * DK + dk[None, :],
                 mask=row_mask[:, None], other=0.0)
    Sb = S.to(tl.bfloat16)
    vn = W.to(tl.float32) - tl.dot(Kcd, Sb)
    vnb = vn.to(tl.bfloat16)
    o = tl.dot(qg, Sb) + tl.dot(ai, vnb)
    tl.store(o_base + toks[:, None] * s_ot + dv[None, :] * s_od,
             o.to(o_ptr.dtype.element_ty), mask=row_mask[:, None])


@triton.jit
def _gdr_fused_gqa_nc2(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale, T,
    HK: tl.constexpr, HV: tl.constexpr, GH: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    s_ob, s_ot, s_oh, s_od,
    EPS: tl.constexpr,
):
    pid_bk = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid_bk // HK
    kh = pid_bk % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = pid_v * BV + tl.arange(0, BV)

    toks0 = rk
    toks1 = C + rk
    m0 = toks0 < T
    m1 = toks1 < T

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh


    q0 = tl.load(q_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    k0 = tl.load(k_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    q1 = tl.load(q_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    k1 = tl.load(k_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    q0 = q0 * (tl.rsqrt(tl.sum(q0 * q0, axis=1) + EPS) * scale)[:, None]
    k0 = k0 * tl.rsqrt(tl.sum(k0 * k0, axis=1) + EPS)[:, None]
    q1 = q1 * (tl.rsqrt(tl.sum(q1 * q1, axis=1) + EPS) * scale)[:, None]
    k1 = k1 * tl.rsqrt(tl.sum(k1 * k1, axis=1) + EPS)[:, None]
    q0b = q0.to(tl.bfloat16); k0b = k0.to(tl.bfloat16)
    q1b = q1.to(tl.bfloat16); k1b = k1.to(tl.bfloat16)
    k0bT = tl.trans(k0b); k1bT = tl.trans(k1b)
    KK0 = tl.dot(k0b, k0bT); QK0 = tl.dot(q0b, k0bT)
    KK1 = tl.dot(k1b, k1bT); QK1 = tl.dot(q1b, k1bT)

    for g_idx in range(0, GH):
        h = kh + g_idx * HK
        v_base = v_ptr + b * s_vb + h * s_vh
        g_base = g_ptr + b * s_gb + h * s_gh
        beta_base = beta_ptr + b * s_gb + h * s_gh
        o_base = o_ptr + b * s_ob + h * s_oh


        v0 = tl.load(v_base + toks0[:, None] * s_vt + dv[None, :] * s_vd, mask=m0[:, None], other=0.0).to(tl.float32)
        g0 = tl.load(g_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)
        beta0 = tl.load(beta_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)

        gcum0 = tl.cumsum(g0, axis=0)
        dm0 = tl.where(tril_incl, tl.exp(gcum0[:, None] - gcum0[None, :]), 0.0)
        kkt0 = beta0[:, None] * KK0
        Bm0 = tl.where(strict_lower, -kkt0 * dm0, 0.0)
        Tm0 = ident + Bm0
        Tm0b = Tm0.to(tl.bfloat16)

        vb0 = (v0 * beta0[:, None]).to(tl.bfloat16)
        W0 = tl.dot(Tm0b, vb0)
        ai0 = (QK0 * dm0).to(tl.bfloat16)

        o0 = tl.dot(ai0, W0.to(tl.bfloat16))
        tl.store(o_base + toks0[:, None] * s_ot + dv[None, :] * s_od,
                 o0.to(o_ptr.dtype.element_ty), mask=m0[:, None])


        glast0 = tl.sum(tl.where(last_sel, g0, 0.0))
        td0 = tl.exp(glast0 - g0)
        Kd0 = (k0 * td0[:, None]).to(tl.bfloat16)
        S1 = tl.dot(tl.trans(Kd0), W0.to(tl.bfloat16))


        v1 = tl.load(v_base + toks1[:, None] * s_vt + dv[None, :] * s_vd, mask=m1[:, None], other=0.0).to(tl.float32)
        g1 = tl.load(g_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)
        beta1 = tl.load(beta_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)

        gcum1 = tl.cumsum(g1, axis=0)
        dm1 = tl.where(tril_incl, tl.exp(gcum1[:, None] - gcum1[None, :]), 0.0)
        kkt1 = beta1[:, None] * KK1
        Bm1 = tl.where(strict_lower, -kkt1 * dm1, 0.0)
        Tm1 = ident + Bm1
        Tm1b = Tm1.to(tl.bfloat16)

        ge1 = tl.exp(g1)
        vb1 = (v1 * beta1[:, None]).to(tl.bfloat16)
        kbg1 = (k1 * (beta1 * ge1)[:, None]).to(tl.bfloat16)
        W1 = tl.dot(Tm1b, vb1)
        Kcd1 = tl.dot(Tm1b, kbg1).to(tl.bfloat16)
        ai1 = (QK1 * dm1).to(tl.bfloat16)
        qg1 = (q1 * ge1[:, None]).to(tl.bfloat16)

        S1b = S1.to(tl.bfloat16)
        vn1 = W1 - tl.dot(Kcd1, S1b)
        vn1b = vn1.to(tl.bfloat16)
        o1 = tl.dot(qg1, S1b) + tl.dot(ai1, vn1b)
        tl.store(o_base + toks1[:, None] * s_ot + dv[None, :] * s_od,
                 o1.to(o_ptr.dtype.element_ty), mask=m1[:, None])


@triton.jit
def _gdr_fused_gqa_nc2_vloop(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale, T,
    HK: tl.constexpr, HV: tl.constexpr, GH: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BV: tl.constexpr, NVB: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    s_ob, s_ot, s_oh, s_od,
    EPS: tl.constexpr,
):
    pid_bk = tl.program_id(0)
    b = pid_bk // HK
    kh = pid_bk % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)

    toks0 = rk
    toks1 = C + rk
    m0 = toks0 < T
    m1 = toks1 < T

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh


    q0 = tl.load(q_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    k0 = tl.load(k_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    q1 = tl.load(q_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    k1 = tl.load(k_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    q0 = q0 * (tl.rsqrt(tl.sum(q0 * q0, axis=1) + EPS) * scale)[:, None]
    k0 = k0 * tl.rsqrt(tl.sum(k0 * k0, axis=1) + EPS)[:, None]
    q1 = q1 * (tl.rsqrt(tl.sum(q1 * q1, axis=1) + EPS) * scale)[:, None]
    k1 = k1 * tl.rsqrt(tl.sum(k1 * k1, axis=1) + EPS)[:, None]
    q0b = q0.to(tl.bfloat16); k0b = k0.to(tl.bfloat16)
    q1b = q1.to(tl.bfloat16); k1b = k1.to(tl.bfloat16)
    k0bT = tl.trans(k0b); k1bT = tl.trans(k1b)
    KK0 = tl.dot(k0b, k0bT); QK0 = tl.dot(q0b, k0bT)
    KK1 = tl.dot(k1b, k1bT); QK1 = tl.dot(q1b, k1bT)

    for g_idx in range(0, GH):
        h = kh + g_idx * HK
        v_base = v_ptr + b * s_vb + h * s_vh
        g_base = g_ptr + b * s_gb + h * s_gh
        beta_base = beta_ptr + b * s_gb + h * s_gh
        o_base = o_ptr + b * s_ob + h * s_oh


        g0 = tl.load(g_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)
        beta0 = tl.load(beta_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)
        g1 = tl.load(g_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)
        beta1 = tl.load(beta_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)


        gcum0 = tl.cumsum(g0, axis=0)
        dm0 = tl.where(tril_incl, tl.exp(gcum0[:, None] - gcum0[None, :]), 0.0)
        Bm0 = tl.where(strict_lower, -(beta0[:, None] * KK0) * dm0, 0.0)
        Tm0 = ident + Bm0
        Tm0b = Tm0.to(tl.bfloat16)
        ai0 = (QK0 * dm0).to(tl.bfloat16)
        glast0 = tl.sum(tl.where(last_sel, g0, 0.0))
        td0 = tl.exp(glast0 - g0)
        Kd0T = tl.trans((k0 * td0[:, None]).to(tl.bfloat16))


        gcum1 = tl.cumsum(g1, axis=0)
        dm1 = tl.where(tril_incl, tl.exp(gcum1[:, None] - gcum1[None, :]), 0.0)
        Bm1 = tl.where(strict_lower, -(beta1[:, None] * KK1) * dm1, 0.0)
        Tm1 = ident + Bm1
        Tm1b = Tm1.to(tl.bfloat16)
        ge1 = tl.exp(g1)
        kbg1 = (k1 * (beta1 * ge1)[:, None]).to(tl.bfloat16)
        Kcd1 = tl.dot(Tm1b, kbg1).to(tl.bfloat16)
        ai1 = (QK1 * dm1).to(tl.bfloat16)
        qg1 = (q1 * ge1[:, None]).to(tl.bfloat16)


        for vb_i in range(0, NVB):
            dv = vb_i * BV + tl.arange(0, BV)

            v0 = tl.load(v_base + toks0[:, None] * s_vt + dv[None, :] * s_vd, mask=m0[:, None], other=0.0).to(tl.float32)
            vb0 = (v0 * beta0[:, None]).to(tl.bfloat16)
            W0 = tl.dot(Tm0b, vb0)
            W0b = W0.to(tl.bfloat16)
            o0 = tl.dot(ai0, W0b)
            tl.store(o_base + toks0[:, None] * s_ot + dv[None, :] * s_od,
                     o0.to(o_ptr.dtype.element_ty), mask=m0[:, None])
            S1 = tl.dot(Kd0T, W0b)

            v1 = tl.load(v_base + toks1[:, None] * s_vt + dv[None, :] * s_vd, mask=m1[:, None], other=0.0).to(tl.float32)
            vb1 = (v1 * beta1[:, None]).to(tl.bfloat16)
            W1 = tl.dot(Tm1b, vb1)
            S1b = S1.to(tl.bfloat16)
            vn1 = W1 - tl.dot(Kcd1, S1b)
            vn1b = vn1.to(tl.bfloat16)
            o1 = tl.dot(qg1, S1b) + tl.dot(ai1, vn1b)
            tl.store(o_base + toks1[:, None] * s_ot + dv[None, :] * s_od,
                     o1.to(o_ptr.dtype.element_ty), mask=m1[:, None])


@triton.jit
def _gdr_fused_gqa_nc2_dedup(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    scale, T,
    HK: tl.constexpr, HV: tl.constexpr, GH: tl.constexpr,
    C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    s_qb, s_qh, s_qt, s_qd,
    s_vb, s_vh, s_vt, s_vd,
    s_gb, s_gh, s_gt,
    s_ob, s_ot, s_oh, s_od,
    EPS: tl.constexpr,
):
    pid_bk = tl.program_id(0)
    b = pid_bk // HK
    kh = pid_bk % HK

    rk = tl.arange(0, C)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)

    toks0 = rk
    toks1 = C + rk
    m0 = toks0 < T
    m1 = toks1 < T

    tril_incl = rk[:, None] >= rk[None, :]
    strict_lower = rk[:, None] > rk[None, :]
    ident = (rk[:, None] == rk[None, :]).to(tl.float32)
    last_sel = (rk == (C - 1))

    q_base = q_ptr + b * s_qb + kh * s_qh
    k_base = k_ptr + b * s_qb + kh * s_qh


    q0 = tl.load(q_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    k0 = tl.load(k_base + toks0[:, None] * s_qt + dk[None, :] * s_qd, mask=m0[:, None], other=0.0).to(tl.float32)
    q1 = tl.load(q_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    k1 = tl.load(k_base + toks1[:, None] * s_qt + dk[None, :] * s_qd, mask=m1[:, None], other=0.0).to(tl.float32)
    q0 = q0 * (tl.rsqrt(tl.sum(q0 * q0, axis=1) + EPS) * scale)[:, None]
    k0 = k0 * tl.rsqrt(tl.sum(k0 * k0, axis=1) + EPS)[:, None]
    q1 = q1 * (tl.rsqrt(tl.sum(q1 * q1, axis=1) + EPS) * scale)[:, None]
    k1 = k1 * tl.rsqrt(tl.sum(k1 * k1, axis=1) + EPS)[:, None]
    q0b = q0.to(tl.bfloat16); k0b = k0.to(tl.bfloat16)
    q1b = q1.to(tl.bfloat16); k1b = k1.to(tl.bfloat16)
    k0bT = tl.trans(k0b); k1bT = tl.trans(k1b)

    KK0 = tl.dot(k0b, k0bT).to(tl.bfloat16)
    QK0 = tl.dot(q0b, k0bT).to(tl.bfloat16)
    KK1 = tl.dot(k1b, k1bT).to(tl.bfloat16)
    QK1 = tl.dot(q1b, k1bT).to(tl.bfloat16)

    for g_idx in range(0, GH):
        h = kh + g_idx * HK
        v_base = v_ptr + b * s_vb + h * s_vh
        g_base = g_ptr + b * s_gb + h * s_gh
        beta_base = beta_ptr + b * s_gb + h * s_gh
        o_base = o_ptr + b * s_ob + h * s_oh


        v0 = tl.load(v_base + toks0[:, None] * s_vt + dv[None, :] * s_vd, mask=m0[:, None], other=0.0).to(tl.float32)
        g0 = tl.load(g_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)
        beta0 = tl.load(beta_base + toks0 * s_gt, mask=m0, other=0.0).to(tl.float32)

        gcum0 = tl.cumsum(g0, axis=0)
        dm0 = tl.where(tril_incl, tl.exp(gcum0[:, None] - gcum0[None, :]), 0.0)
        Bm0 = tl.where(strict_lower, -(beta0[:, None] * KK0.to(tl.float32)) * dm0, 0.0)
        Tm0b = (ident + Bm0).to(tl.bfloat16)

        vb0 = (v0 * beta0[:, None]).to(tl.bfloat16)
        W0 = tl.dot(Tm0b, vb0)
        W0b = W0.to(tl.bfloat16)
        ai0 = (QK0.to(tl.float32) * dm0).to(tl.bfloat16)
        o0 = tl.dot(ai0, W0b)
        tl.store(o_base + toks0[:, None] * s_ot + dv[None, :] * s_od,
                 o0.to(o_ptr.dtype.element_ty), mask=m0[:, None])

        glast0 = tl.sum(tl.where(last_sel, g0, 0.0))
        td0 = tl.exp(glast0 - g0)
        Kd0T = tl.trans((k0 * td0[:, None]).to(tl.bfloat16))
        S1 = tl.dot(Kd0T, W0b)


        v1 = tl.load(v_base + toks1[:, None] * s_vt + dv[None, :] * s_vd, mask=m1[:, None], other=0.0).to(tl.float32)
        g1 = tl.load(g_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)
        beta1 = tl.load(beta_base + toks1 * s_gt, mask=m1, other=0.0).to(tl.float32)

        gcum1 = tl.cumsum(g1, axis=0)
        dm1 = tl.where(tril_incl, tl.exp(gcum1[:, None] - gcum1[None, :]), 0.0)
        Bm1 = tl.where(strict_lower, -(beta1[:, None] * KK1.to(tl.float32)) * dm1, 0.0)
        Tm1b = (ident + Bm1).to(tl.bfloat16)

        ge1 = tl.exp(g1)
        vb1 = (v1 * beta1[:, None]).to(tl.bfloat16)
        kbg1 = (k1 * (beta1 * ge1)[:, None]).to(tl.bfloat16)
        W1 = tl.dot(Tm1b, vb1)
        Kcd1 = tl.dot(Tm1b, kbg1).to(tl.bfloat16)
        ai1 = (QK1.to(tl.float32) * dm1).to(tl.bfloat16)
        qg1 = (q1 * ge1[:, None]).to(tl.bfloat16)

        S1b = S1.to(tl.bfloat16)
        vn1 = W1 - tl.dot(Kcd1, S1b)
        vn1b = vn1.to(tl.bfloat16)
        o1 = tl.dot(qg1, S1b) + tl.dot(ai1, vn1b)
        tl.store(o_base + toks1[:, None] * s_ot + dv[None, :] * s_od,
                 o1.to(o_ptr.dtype.element_ty), mask=m1[:, None])


def _good(gr):


    return (128 <= gr <= 148) or (gr >= 224)


def _run_split(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC):


    if (B == 8 or B >= 16) and NC <= 4:
        BV = 128
    else:
        BV = 32 if B <= 2 else 64
    n_vblk = (DV + BV - 1) // BV


    long_sb = (B <= 2 and NC >= 24)


    nstg_scan = 3 if (long_sb or B <= 2) else (2 if B <= 4 else 1)
    N = B * HV * NC
    dev = query.device
    bf = query.dtype

    W = torch.empty((N, C, DV), dtype=bf, device=dev)
    Kcd = torch.empty((N, C, DK), dtype=bf, device=dev)
    Ai = torch.empty((N, C, C), dtype=bf, device=dev)
    Qg = torch.empty((N, C, DK), dtype=bf, device=dev)
    Kd = torch.empty((N, C, DK), dtype=bf, device=dev)
    Aa = torch.empty((N,), dtype=torch.float32, device=dev)
    out = torch.empty((B, T, HV, DV), dtype=bf, device=dev)


    GH_TOT = HV // HK
    g4 = B * HK * NC
    GH = 0
    if long_sb:
        GH = 0
    elif _good(g4):
        GH, NGRP = GH_TOT, 1
    elif (176 <= g4 < 224) and _good(B * HK * 2 * NC):
        GH, NGRP = 2, GH_TOT // 2
    if GH:
        _gdr_prepare_gqa[(B * HK * NGRP * NC,)](
            query, key, value, g, beta,
            W, Kcd, Ai, Qg, Kd, Aa,
            float(scale),
            T, NC,
            HK, HV, GH, NGRP,
            C, DK, DV,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            EPS=1e-6,
            num_warps=8,
            num_stages=1,
        )
    else:
        _gdr_prepare[(N,)](
            query, key, value, g, beta,
            W, Kcd, Ai, Qg, Kd, Aa,
            float(scale),
            T, NC,
            HK, HV,
            C, DK, DV,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            EPS=1e-6,
            num_warps=8,
            num_stages=1,
        )
    _gdr_scan[(B * HV, n_vblk)](
        W, Kcd, Ai, Qg, Kd, Aa, out,
        T, NC,
        HV,
        C, DK, DV, BV,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=8,
        num_stages=nstg_scan,
    )
    return out


def _run_fused(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC):


    BV = 16 if (B * HV) <= 32 else 64
    n_vblk = (DV + BV - 1) // BV
    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=query.device)
    grid = (B * HV, n_vblk)
    _gdr_fused[grid](
        query, key, value, g, beta, out,
        float(scale),
        T, NC,
        HK, HV,
        C, DK, DV, BV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-6,
        num_warps=8,
        num_stages=1,
    )
    return out


def _run_fused_nc2(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=64, nw=8):


    BV = bv
    n_vblk = (DV + BV - 1) // BV
    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=query.device)
    grid = (B * HV, n_vblk)
    _gdr_fused_nc2[grid](
        query, key, value, g, beta, out,
        float(scale),
        T, NC,
        HK, HV,
        C, DK, DV, BV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-6,
        num_warps=nw,
        num_stages=1,
    )
    return out


def _run_fused_gqa_nc2(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=64):


    GH = HV // HK
    BV = bv
    n_vblk = (DV + BV - 1) // BV
    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=query.device)
    grid = (B * HK, n_vblk)
    _gdr_fused_gqa_nc2[grid](
        query, key, value, g, beta, out,
        float(scale), T,
        HK, HV, GH,
        C, DK, DV, BV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-6,
        num_warps=8,
        num_stages=1,
    )
    return out


def _run_fused_gqa_nc2_vloop(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=64, nw=8):


    GH = HV // HK
    BV = bv
    NVB = (DV + BV - 1) // BV
    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=query.device)
    grid = (B * HK,)
    _gdr_fused_gqa_nc2_vloop[grid](
        query, key, value, g, beta, out,
        float(scale), T,
        HK, HV, GH,
        C, DK, DV, BV, NVB,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-6,
        num_warps=nw,
        num_stages=1,
    )
    return out


def _run_fused_gqa_nc2_dedup(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, nw=8):


    GH = HV // HK
    out = torch.empty((B, T, HV, DV), dtype=query.dtype, device=query.device)
    grid = (B * HK,)
    _gdr_fused_gqa_nc2_dedup[grid](
        query, key, value, g, beta, out,
        float(scale), T,
        HK, HV, GH,
        C, DK, DV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        EPS=1e-6,
        num_warps=nw,
        num_stages=1,
    )
    return out


@torch.no_grad()
def run(query, key, value, g, beta, scale):
    B, HK, T, DK = query.shape
    HV = value.shape[1]
    DV = value.shape[-1]
    C = _CHUNK
    NC = (T + C - 1) // C

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()


    if NC == 2 and B >= 32:


        return _run_fused_nc2(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=128, nw=4)
    if NC == 2 and B >= 16:


        return _run_fused_nc2(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=128, nw=4)
    if NC == 2:


        return _run_fused_nc2(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC, bv=128, nw=8)


    g4 = B * HK * NC
    if B <= 2:
        if NC < 6:
            return _run_fused(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC)
        return _run_split(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC)
    if B >= 32:
        return _run_fused(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC)
    use_split = _good(g4) or (176 <= g4 < 224 and _good(2 * g4))
    if use_split:
        return _run_split(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC)
    return _run_fused(query, key, value, g, beta, scale, B, HK, T, DK, HV, DV, C, NC)
