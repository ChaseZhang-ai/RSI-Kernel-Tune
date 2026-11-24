import torch


def run(input, weight, eps=1e-5):
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    hidden_states = input * rstd
    return (hidden_states * weight).to(weight.dtype)
