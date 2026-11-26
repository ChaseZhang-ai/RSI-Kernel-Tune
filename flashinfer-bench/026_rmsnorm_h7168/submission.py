# Adapted from RSI-Kernel-Tune's sol-example-triton-rmsnorm.
import triton
import triton.language as tl


@triton.jit
def _padded(X, W, Y):
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    mask = col < 7168
    x = tl.load(X + row * 7168 + col, mask, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 7168) + 1e-6)
    w = tl.load(W + col, mask, 0).to(tl.float32)
    tl.store(Y + row * 7168 + col, (x * scale) * w, mask)


def run(hidden_states, weight, output):
    _padded[(hidden_states.shape[0],)](hidden_states, weight, output, num_warps=4)
