"""Destination-passing BF16 RMSNorm for a 4096-element hidden dimension."""

import triton
import triton.language as tl


# Adapted from kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py.
@triton.jit
def _rmsnorm(hidden_states, weight, output):
    row = tl.program_id(0)
    columns = tl.arange(0, 4096)
    offsets = row * 4096 + columns
    x = tl.load(hidden_states + offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / 4096) + 1e-5)
    w = tl.load(weight + columns).to(tl.float32)
    tl.store(output + offsets, ((x * inv_rms) * w).to(tl.bfloat16))


def run(hidden_states, weight, output):
    _rmsnorm[(hidden_states.shape[0],)](hidden_states, weight, output, num_warps=4)
