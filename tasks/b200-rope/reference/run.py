"""RoPE reference: PyTorch, rotate-half (GPT-NeoX) formulation, float32 angles."""

import torch


def run(q, k, pos):
    d = q.shape[-1]
    half = d // 2
    inv_freq = 10000.0 ** (-torch.arange(0, half, dtype=torch.float32,
                                         device=q.device) * 2.0 / d)
    ang = pos.float().unsqueeze(-1) * inv_freq.unsqueeze(0)   # [m, half]
    cos, sin = ang.cos(), ang.sin()

    def rope(x):
        xf = x.float()
        x1, x2 = xf[..., :half], xf[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat([y1, y2], dim=-1).to(x.dtype)

    return rope(q), rope(k)
