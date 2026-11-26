# SPDX-License-Identifier: Apache-2.0
"""Triton submission for SOL-ExecBench 084, SiLU activation backward."""

import triton
import triton.language as tl


# Block size, warps, load cache modifier, store cache modifier, load eviction policy.
_CONFIGS = {
    131: (32, 1, "", "", ""),
    2053: (256, 4, "", "", ""),
    3089: (256, 4, ".cg", "", ""),
    4096: (512, 4, "", "", ""),
    16384: (256, 4, "", "", ""),
    40960: (512, 4, ".cg", ".cs", ""),
    163840: (512, 4, ".cg", "", ""),
    262144: (512, 4, ".cg", ".cs", ""),
    655360: (512, 1, "", "", ""),
    786432: (512, 1, "", "", ""),
    2097152: (2048, 2, "", ".cs", "evict_first"),
    4194304: (2048, 2, "", ".cs", "evict_first"),
    5242880: (2048, 8, "", ".cs", "evict_first"),
    8388608: (512, 4, "", ".cs", "evict_first"),
    10485760: (8192, 8, "", ".cs", "evict_first"),
    16777216: (2048, 8, ".cg", "", ""),
}


@triton.jit
def _backward(
    grad_output, x, sigmoid_x, grad_input, n,
    G_STRIDE: tl.constexpr, X_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr, O_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr,
    LOAD: tl.constexpr, STORE: tl.constexpr, EVICT: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n if HAS_TAIL else None
    g = tl.load(grad_output + i * G_STRIDE, valid, cache_modifier=LOAD, eviction_policy=EVICT)
    v = tl.load(x + i * X_STRIDE, valid, cache_modifier=LOAD, eviction_policy=EVICT)
    s = tl.load(sigmoid_x + i * S_STRIDE, valid, cache_modifier=LOAD, eviction_policy=EVICT)
    derivative = s * tl.fma(v, 1.0 - s, 1.0)
    tl.store(grad_input + i * O_STRIDE, g * derivative, valid, cache_modifier=STORE)


def run(grad_output, x, sigmoid_x, grad_input):
    n = grad_output.numel()
    if n == 0:
        return
    block, warps, load, store, evict = _CONFIGS.get(n, (512, 4, "", "", ""))
    _backward[(triton.cdiv(n, block),)](
        grad_output, x, sigmoid_x, grad_input, n,
        grad_output.stride(0), x.stride(0), sigmoid_x.stride(0), grad_input.stride(0),
        block, n % block != 0, load, store, evict, num_warps=warps,
    )
