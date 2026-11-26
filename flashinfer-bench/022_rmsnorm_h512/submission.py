"""Destination-passing BF16 RMSNorm for hidden size 512."""

import triton
import triton.language as tl


# Adapted from kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py.
@triton.jit
def _rmsnorm_row(hidden_states, weight, output):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    x = tl.load(hidden_states + row * 512 + col).to(tl.float32)
    w = tl.load(weight + col).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / 512) + 1.0e-6)
    tl.store(output + row * 512 + col, ((x * inv_rms) * w).to(tl.bfloat16))


def run(hidden_states, weight, output):
    _rmsnorm_row[(hidden_states.shape[0],)](hidden_states, weight, output, num_warps=4)
