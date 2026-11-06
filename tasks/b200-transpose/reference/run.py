"""Matrix transpose reference: PyTorch, a materialized transpose."""

import torch


def run(x):
    return x.t().contiguous()
