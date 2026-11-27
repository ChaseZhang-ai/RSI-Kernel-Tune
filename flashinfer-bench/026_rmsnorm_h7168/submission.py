# SPDX-License-Identifier: Apache-2.0
# Based on RSI-Kernel-Tune's sol-example-triton-rmsnorm.
# CTA replication for small batches was inspired by QianQu's
# CompoundingKernel-Explorer, FlashInfer/026_rmsnorm_h7168.
import triton
import triton.language as tl


@triton.jit
def _small(X, W, Y, SPAN: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 8192)
    shifted = lane + tl.program_id(1) * SPAN
    col = tl.where(shifted < 7168, shifted, shifted - 7168)

    # Every CTA computes the full row statistic. Only its leading lanes
    # handle the owned slice, whose data is already in registers.
    x = tl.load(X + row * 7168 + col, lane < 7168, 0).to(tl.float32)
    owned = lane < SPAN
    w = tl.load(W + col, owned, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 7168) + 1e-6)
    tl.store(Y + row * 7168 + col, (x * scale) * w, owned)


@triton.jit
def _medium(X, W, Y):
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    valid = col < 7168
    x = tl.load(X + row * 7168 + col, valid, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 7168) + 1e-6)
    w = tl.load(W + col, valid, 0).to(tl.float32)
    tl.store(Y + row * 7168 + col, (x * scale) * w, valid)


@triton.jit
def _large(X, W, Y):
    row = tl.program_id(0)
    lane = tl.arange(0, 1024)
    base = X + row * 7168 + lane
    squares = tl.full((1024,), 0, tl.float32)
    values = ()
    for segment in tl.static_range(7):
        x = tl.load(base + segment * 1024, cache_modifier=".cg")
        values += (x,)
        xf = x.to(tl.float32)
        squares = tl.fma(xf, xf, squares)

    scale = tl.rsqrt(tl.sum(squares, 0) * (1.0 / 7168) + 1e-6)
    for segment in tl.static_range(7):
        col = lane + segment * 1024
        w = tl.load(W + col).to(tl.float32)
        y = (values[segment].to(tl.float32) * scale) * w
        tl.store(Y + row * 7168 + col, y, cache_modifier=".cs")


def run(hidden_states, weight, output):
    batch = hidden_states.shape[0]
    if batch <= 64:
        parts = 14 if batch == 1 else 7 if batch <= 7 else 4 if batch <= 32 else 2
        _small[(batch, parts)](hidden_states, weight, output, 7168 // parts, num_warps=16)
    elif batch < 1024:
        _medium[(batch,)](hidden_states, weight, output, num_warps=8)
    else:
        _large[(batch,)](hidden_states, weight, output, num_warps=4)
