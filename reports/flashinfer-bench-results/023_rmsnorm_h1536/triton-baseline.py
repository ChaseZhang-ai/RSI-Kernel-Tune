"""Padded one-CTA-per-row baseline from RSI-Kernel-Tune's Triton RMSNorm example."""
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_padded(x, w, y):
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    valid = col < 1536
    val = tl.load(x + row * 1536 + col, valid, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(val * val, 0) * (1.0 / 1536) + 1.0e-6)
    weight = tl.load(w + col, valid, 0).to(tl.float32)
    tl.store(y + row * 1536 + col, (val * scale) * weight, valid)


def run(hidden_states, weight, output):
    rows = hidden_states.shape[0]
    if rows:
        _rmsnorm_padded[(rows,)](hidden_states, weight, output, num_warps=4)
