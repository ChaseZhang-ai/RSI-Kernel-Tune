"""fp16 GEMM reference: PyTorch matmul on the operands as given (fp16 in, fp16 out)."""

import torch


def run(a, b):
    return a @ b
