# SPDX-License-Identifier: Apache-2.0
# Adapted from RSI-Kernel-Tune's sol-example-triton-rmsnorm.
# CTA replication and multi-row weight reuse draw on CompoundingKernel-Explorer,
# FlashInfer/001_fused_add_rmsnorm_h2048, published by QianQu.

import triton
import triton.language as tl


@triton.jit
def _replicated_row(H, R, W, Y, PARTS: tl.constexpr, ROTATE: tl.constexpr):
    block = tl.program_id(0)
    row = block // PARTS
    part = block % PARTS
    col = tl.arange(0, 2048)
    if ROTATE:
        col = (col + part * (2048 // PARTS)) % 2048
    offset = row * 2048 + col
    h = tl.load(H + offset).to(tl.float32)
    r = tl.load(R + offset).to(tl.float32)
    own = col // (2048 // PARTS) == part
    w = tl.load(W + col, mask=own, other=0).to(tl.float32)
    x = h + r
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048) + 1.0e-6)
    tl.store(Y + offset, ((x * inv_rms) * w).to(tl.bfloat16), mask=own)


@triton.jit
def _row_pair(H, R, W, Y, batch_size):
    row = 2 * tl.program_id(0) + tl.arange(0, 2)
    col = tl.arange(0, 2048)
    offset = row[:, None] * 2048 + col[None, :]
    active = row[:, None] < batch_size
    h = tl.load(H + offset, mask=active, other=0).to(tl.float32)
    r = tl.load(R + offset, mask=active, other=0).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    x = h + r
    inv_rms = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / 2048) + 1.0e-6)
    y = (x * inv_rms[:, None]) * w[None, :]
    tl.store(Y + offset, y.to(tl.bfloat16), mask=active)


def run(hidden_states, residual, weight, output):
    batch_size = hidden_states.shape[0]
    if batch_size == 1:
        _replicated_row[(16,)](
            hidden_states, residual, weight, output, 16, True, num_warps=16
        )
    elif batch_size <= 79:
        parts = 4 if batch_size <= 8 else 2
        _replicated_row[(batch_size * parts,)](
            hidden_states, residual, weight, output, parts, False, num_warps=8
        )
    else:
        _row_pair[(triton.cdiv(batch_size, 2),)](
            hidden_states, residual, weight, output, batch_size, num_warps=4
        )
