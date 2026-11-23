import torch


def run(attn_output, weight):
    x = attn_output.to(torch.float32)
    projected = torch.matmul(x, weight.t())
    return projected.to(attn_output.dtype)
