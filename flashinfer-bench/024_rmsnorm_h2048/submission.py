"""Direct-launch baseline adapted from sol-example-triton-rmsnorm."""
import triton
import triton.language as tl


@triton.jit
def _row_rmsnorm(X, W, Y):
    col = tl.arange(0, 2048)
    offsets = tl.program_id(0) * 2048 + col
    x = tl.load(X + offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 2048) + 1e-6)
    w = tl.load(W + col).to(tl.float32)
    tl.store(Y + offsets, ((x * inv_rms) * w).to(tl.bfloat16))


def run(hidden_states, weight, output):
    _row_rmsnorm[(hidden_states.shape[0],)](hidden_states, weight, output,
                                           num_warps=4, enable_fp_fusion=False)
