"""GELU reference: PyTorch, the exact erf form, float32 arithmetic."""

import math

import torch


def run(x):
    xf = x.float()
    y = xf * 0.5 * (1.0 + torch.erf(xf / math.sqrt(2.0)))
    return y.to(x.dtype)
