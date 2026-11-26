"""Direct-launch adaptation of kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py."""

import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm(H, R, W, O):
    col = tl.arange(0, 4096)
    offset = tl.program_id(0) * 4096 + col
    x = tl.load(H + offset).to(tl.float32) + tl.load(R + offset).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 4096) + 1.0e-5)
    weight = tl.load(W + col).to(tl.float32)
    tl.store(O + offset, (x * inv_rms) * weight)


def run(hidden_states, residual, weight, output):
    _fused_add_rmsnorm[(hidden_states.shape[0],)](hidden_states, residual, weight, output, num_warps=4)
