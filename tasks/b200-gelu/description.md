# b200-gelu

The exact (erf form) GELU activation, elementwise over a flat tensor.

## Signature

- inputs, in order:
  - `x`: `[n]`, bfloat16 -- the pre-activations, flattened
- output:
  - `y`: `[n]`, bfloat16

## Semantics

For every element: `y[i] = x[i] * 0.5 * (1 + erf(x[i] / sqrt(2)))`. This is the exact
erf form, not the tanh approximation: a tanh-form kernel is a different function and must
not be submitted. The product and the erf evaluation are taken in float32 (at least); the
result is rounded to bfloat16 once, at the end. This task's contract declares
compute_dtype=fp32: every arithmetic step below fp32 is a contract violation (the
tolerance policy is the gate's relative bound against this reference's own error at the
declaration).

## Legal inputs

Any finite bfloat16 values; both tails (large negative and large positive) are legal and
map to 0 and x respectively. The output must be finite everywhere the input is.
