"""Fused add + GELU reference: PyTorch, fp32 end to end, the exact erf form."""

import math

import torch


def run(a, b):
    s = a + b
    return s * 0.5 * (1.0 + torch.erf(s / math.sqrt(2.0)))
