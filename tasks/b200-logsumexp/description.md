# b200-logsumexp

Logsumexp over the last axis -- the reduction behind every cross-entropy loss and the
log-domain softmax. The contract is the numerically safe (max-shifted) form.

## Signature

- inputs, in order:
  - `x`: `[m, n]`, bfloat16 -- the logits, one row per reduction
- output:
  - `y`: `[m]`, float32 -- one logsumexp per row

## Semantics

For every row `i`: `y[i] = log(sum_j exp(x[i, j]))`, evaluated as
`y[i] = max_k x[i, k] + log(sum_j exp(x[i, j] - max_k x[i, k]))`. The maximum, the
subtraction, the exponential and the sum are taken in float32 (at least); the output
stays float32 -- it is never rounded through bfloat16. This task's contract declares
compute_dtype=fp32: every arithmetic step below fp32 is a contract violation; a different
fp32 reduction order (a tree reduction, a warp-level scan) is legitimate (the tolerance
policy is the gate's relative bound against this reference's own error at the
declaration).

## Legal inputs

Any finite bfloat16 values; realistic logit magnitudes are in the tens, which is why the
max shift is part of the contract (a raw sum of exp would overflow). Rows of equal values
produce `x[i, 0] + log(n)`. A row of all `-inf` is NOT a legal input.
