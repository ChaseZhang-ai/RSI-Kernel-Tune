import torch


def run(attn_output, weight, residual):
    x = attn_output.to(torch.float32)
    projected = torch.matmul(x, weight.t())
    return (projected + residual).to(attn_output.dtype)
