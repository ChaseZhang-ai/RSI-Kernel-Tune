# b200-layernorm

LayerNorm over the last axis (no batch/sequence normalization), as used in every
transformer block's pre-normalization.

## Signature

- inputs, in order:
  - `x`: `[m, n]`, bfloat16 -- the activations, one row per token
  - `w`: `[n]`, bfloat16 -- the learned scale
  - `b`: `[n]`, bfloat16 -- the learned bias
- output:
  - `y`: `[m, n]`, bfloat16

## Semantics

For every row `i`: `y[i, :] = (x[i, :] - mean_j(x[i, j])) / sqrt(var_j(x[i, j]) + 1e-5)
* w + b`, where `var` is the biased (population) variance over the same last axis as the
mean. The mean, the variance and the square root are taken in float32 (at least); the
result is rounded to bfloat16 once, at the end. This task's contract declares
compute_dtype=fp32: every arithmetic step below fp32 is a contract violation; a different
fp32 reduction order or a fused multiply-add is legitimate (the tolerance policy is the
gate's relative bound against this reference's own error at the declaration).

## Legal inputs

Any finite bfloat16 values. Real activations have occasional large outlier channels; rows
near constant must not produce NaN (the epsilon keeps the denominator positive). `w` is
positive in practice and `b` near zero, but the operator assumes neither.
