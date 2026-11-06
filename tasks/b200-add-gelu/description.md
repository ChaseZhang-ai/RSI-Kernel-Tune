# b200-add-gelu

Fused elementwise: a residual add followed by the exact (erf form) GELU, in one kernel
over flat fp32 tensors. The unfused PyTorch sequence is two launches and an intermediate
tensor; the contract is one kernel producing the same numbers.

## Signature

- inputs, in order:
  - `a`: `[n]`, float32 -- the branch output (e.g. the MLP's projection)
  - `b`: `[n]`, float32 -- the residual being added
- output:
  - `y`: `[n]`, float32

## Semantics

For every element: `y[i] = gelu_erf(a[i] + b[i])` where `gelu_erf(t) = t * 0.5 *
(1 + erf(t / sqrt(2)))`. The add is the IEEE float32 add; the erf evaluation and the
products are float32. This is the exact erf form, not the tanh approximation. This task's
contract declares compute_dtype=fp32: every arithmetic step below fp32 is a contract
violation; a fused multiply-add inside the erf path is legitimate (the tolerance policy
is the gate's relative bound against this reference's own error at the declaration).

## Legal inputs

Any finite float32 values with `a[i] + b[i]` finite for every element (inputs are drawn
so the sum cannot overflow). Both GELU tails are legal. The output must be finite
everywhere the inputs are.
