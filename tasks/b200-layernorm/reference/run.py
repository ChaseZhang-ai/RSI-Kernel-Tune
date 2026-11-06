"""LayerNorm reference: PyTorch, float32 statistics, biased variance."""

import torch


def run(x, w, b):
    xf = x.float()
    mu = xf.mean(dim=-1, keepdim=True)
    var = (xf - mu).pow(2).mean(dim=-1, keepdim=True)
    y = (xf - mu) * torch.rsqrt(var + 1e-5) * w.float() + b.float()
    return y.to(x.dtype)
