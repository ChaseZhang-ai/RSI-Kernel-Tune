"""Direct-launch BF16 RMSNorm for rows of 128 elements.

Based on RSI-Kernel-Tune's ako4x Triton tiered-rows kernel. The explicit
vector layout sweep was informed by the CompoundingKernel Explorer
FlashInfer/021_rmsnorm_h128 archive.
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit(do_not_specialize=["batch"])
def _rmsnorm(
    X, W, Y, batch, ROWS: gl.constexpr, WARPS: gl.constexpr,
    VECTOR: gl.constexpr, FULL: gl.constexpr,
):
    LANES: gl.constexpr = 128 // VECTOR
    LAYOUT: gl.constexpr = BlockedLayout(
        [1, VECTOR], [32 // LANES, LANES], [WARPS, 1], [1, 0]
    )
    rows = gl.program_id(0) * ROWS + gl.arange(
        0, ROWS, layout=gl.constexpr(SliceLayout(1, LAYOUT))
    )
    cols = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, LAYOUT)))
    offsets = rows[:, None] * 128 + cols[None, :]
    w = gl.load(W + cols)
    if FULL:
        x = gl.load(X + offsets)
    else:
        x = gl.load(X + offsets, rows[:, None] < batch, 0)

    # Keep the distributed type after global loads on fbtriton 3.7.1.
    x = gl.permute(x, (0, 1)).to(gl.float32)
    w = gl.permute(w, (0,)).to(gl.float32)
    inv_rms = gl.rsqrt(gl.sum(x * x, 1) * (1.0 / 128) + 1e-6)
    result = (x * inv_rms[:, None] * w[None, :]).to(gl.bfloat16)
    if FULL:
        gl.store(Y + offsets, result)
    else:
        gl.store(Y + offsets, result, rows[:, None] < batch)


def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch == 0:
        return

    if batch <= 512:
        rows, warps, vector = 4, 2, 16
    elif batch <= 4096:
        rows, warps, vector = 8, 4, 16
    elif batch <= 131072:
        rows, warps, vector = 32, 2, 8
    else:
        rows, warps, vector = 16, 4, 8

    _rmsnorm[(triton.cdiv(batch, rows),)](
        hidden_states, weight, output, batch, rows, warps, vector,
        batch % rows == 0, num_warps=warps,
    )
