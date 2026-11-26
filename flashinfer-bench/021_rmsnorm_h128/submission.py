"""Direct-launch BF16 RMSNorm for rows of 128 elements.

Adapted from RSI-Kernel-Tune's ako4x Triton tiered-rows and SOL example kernels.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["batch"])
def _rmsnorm(X, W, Y, batch, ROWS: tl.constexpr, FULL: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]
    if FULL:
        x = tl.load(X + offsets, eviction_policy="evict_first").to(tl.float32)
    else:
        x = tl.load(X + offsets, rows[:, None] < batch, 0,
                    eviction_policy="evict_first").to(tl.float32)
    w = tl.load(W + cols, eviction_policy="evict_last").to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) * (1.0 / 128) + 1e-6)
    result = (x * inv_rms[:, None] * w[None, :]).to(tl.bfloat16)
    if FULL:
        tl.store(Y + offsets, result)
    else:
        tl.store(Y + offsets, result, rows[:, None] < batch)


def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return
    if batch <= 256:
        rows, warps = 1, 1
    elif batch <= 1024:
        rows, warps = 4, 2
    elif batch <= 16384:
        rows, warps = 16, 4
    else:
        rows, warps = 32, 4
    _rmsnorm[(triton.cdiv(batch, rows),)](
        hidden_states, weight, output, batch, rows, batch % rows == 0,
        num_warps=warps, num_stages=2,
    )
