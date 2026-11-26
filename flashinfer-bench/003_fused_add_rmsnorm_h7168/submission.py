"""Fused add adaptation of RSI-Kernel-Tune's Triton RMSNorm example."""

import triton
import triton.language as tl


@triton.jit
def _fused_row(hidden, residual, weight, output, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * H + col
    x = tl.load(hidden + offset, col < H, 0).to(tl.float32)
    r = tl.load(residual + offset, col < H, 0).to(tl.float32)
    value = x + r
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / H + 1e-6)
    w = tl.load(weight + col, col < H, 0).to(tl.float32)
    tl.store(output + offset, (value * inv_rms) * w, col < H)


def run(hidden_states, residual, weight, output):
    _fused_row[(hidden_states.shape[0],)](
        hidden_states, residual, weight, output, 7168, 8192, num_warps=8,
    )
