# SPDX-License-Identifier: Apache-2.0
# Adapted from RSI-Kernel-Tune's sol-example-triton-rmsnorm.
# Replicated per-row reductions and multi-row tiles draw on QianQu's
# CompoundingKernel-Explorer, FlashInfer/001_fused_add_rmsnorm_h2048.

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout


@triton.jit
def _tiled_rmsnorm(H, R, W, Y, batch_size, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, 2048)
    off = row[:, None] * 2048 + col[None, :]
    active = row[:, None] < batch_size
    h = tl.load(H + off, active, other=0).to(tl.float32)
    r = tl.load(R + off, active, other=0).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    x = h + r
    inv_rms = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 2048) + 1.0e-6)
    y = (x * inv_rms[:, None]) * w[None, :]
    tl.store(Y + off, y.to(tl.bfloat16), active)


@gluon.jit
def _row_rmsnorm(H, R, W, Y, PARTS: gl.constexpr, WARPS: gl.constexpr, VEC: gl.constexpr,
           ROTATE: gl.constexpr, LOAD_CACHE: gl.constexpr):
    col = gl.arange(0, 2048, layout=BlockedLayout([VEC], [32], [WARPS], [0]))
    row = gl.program_id(0)
    if PARTS > 1:
        part = gl.program_id(1)
        if ROTATE:
            col = (col + part * (2048 // PARTS)) % 2048
    off = row * 2048 + col
    h = gl.load(H + off, cache_modifier=LOAD_CACHE).to(gl.float32)
    r = gl.load(R + off, cache_modifier=LOAD_CACHE).to(gl.float32)
    if PARTS == 1:
        w = gl.load(W + col).to(gl.float32)
    else:
        owns = col // (2048 // PARTS) == part
        w = gl.load(W + col, owns, other=0).to(gl.float32)
    x = h + r
    inv_rms = gl.rsqrt(gl.sum(x * x, 0) * (1.0 / 2048) + 1.0e-6)
    y = ((x * inv_rms) * w).to(gl.bfloat16)
    if PARTS == 1:
        gl.store(Y + off, y)
    else:
        gl.store(Y + off, y, owns)


def run(hidden_states, residual, weight, output):
    batch_size = hidden_states.shape[0]
    if batch_size == 1:
        # Each CTA reduces the complete row and writes its own 256 columns.
        _row_rmsnorm[(1, 8)](
            hidden_states, residual, weight, output, 8, 8, 8, True, "", num_warps=8
        )
    elif batch_size <= 79:
        _row_rmsnorm[(batch_size, 4)](
            hidden_states, residual, weight, output, 4, 8, 8, False, "", num_warps=8
        )
    else:
        _tiled_rmsnorm[(triton.cdiv(batch_size, 2),)](
            hidden_states, residual, weight, output, batch_size, 2, num_warps=4
        )
