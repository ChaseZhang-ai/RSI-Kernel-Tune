"""Logsumexp reference: PyTorch, max-shifted, float32 arithmetic, float32 output."""

import torch


def run(x):
    xf = x.float()
    mx = xf.amax(dim=-1, keepdim=True)
    y = mx.squeeze(-1) + torch.log(torch.exp(xf - mx).sum(dim=-1))
    return y
