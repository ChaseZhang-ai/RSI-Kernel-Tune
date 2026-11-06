# b200-transpose

A materialized 2D matrix transpose: `y = x^T` as a real tensor in memory (like PyTorch's
`.t().contiguous()`), not a view.

## Signature

- inputs, in order:
  - `x`: `[m, n]`, float32 -- row-major
- output:
  - `y`: `[n, m]`, float32, row-major

## Semantics

For every element: `y[j, i] = x[i, j]`. No arithmetic: every output element is exactly
one input element. The output must be a genuinely transposed, row-major materialization
(a strided view of `x` is not a valid implementation). This task's contract declares
compute_dtype=fp32 (no arithmetic happens; fp32 storage). The tolerance policy is exact:
the output is a permutation of the input's bits, so any mismatch anywhere is a failure,
independent of the gate's relative bound.

## Legal inputs

Any finite (or arbitrary) float32 values -- the values are irrelevant, only the addresses
matter. Both square and rectangular matrices are legal; `m` and `n` are independent.
