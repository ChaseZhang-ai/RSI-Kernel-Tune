# b200-softmax

Softmax over the last axis -- the attention-score normalization. The contract is the
numerically safe form: the row maximum is subtracted before exponentiating (the result is
mathematically unchanged; skipping it is not).

## Signature

- inputs, in order:
  - `x`: `[m, n]`, bfloat16 -- the scores, one row per softmax
- output:
  - `y`: `[m, n]`, bfloat16 -- one probability distribution per row

## Semantics

For every row `i`: `y[i, j] = exp(x[i, j] - max_k x[i, k]) / sum_j exp(x[i, j] -
max_k x[i, k])`. The maximum, the subtraction, the exponential and the sum are taken in
float32 (at least); the result is rounded to bfloat16 once, at the end. Every row of `y`
sums to 1 within bfloat16 rounding. This task's contract declares compute_dtype=fp32:
every arithmetic step below fp32 is a contract violation; a different fp32 reduction
order or a fused multiply-add is legitimate (the tolerance policy is the gate's relative
bound against this reference's own error at the declaration).

## Legal inputs

Any finite bfloat16 values; realistic attention logits have magnitudes in the tens, which
is exactly why the max subtraction is part of the contract (exp of a raw logit near 88+
overflows even fp32). Rows of equal values produce a uniform distribution. A row of all
`-inf` (a fully masked-out attention row) is NOT a legal input.
