"""Softmax reference: PyTorch, max-subtracted, float32 arithmetic."""

import torch


def run(x):
    xf = x.float()
    e = torch.exp(xf - xf.amax(dim=-1, keepdim=True))
    y = e / e.sum(dim=-1, keepdim=True)
    return y.to(x.dtype)
