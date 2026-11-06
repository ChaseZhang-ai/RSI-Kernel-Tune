# b200-gemm-fp16

A square-ish fp16 matrix multiply `C = A @ B`, the tensor-core shape that dominates
transformer compute. Small for a B200 (4096^3 and below) -- the audit's compute-bound
regime point, deliberately well inside the machine's HBM.

## Signature

- inputs, in order:
  - `a`: `[m, k]`, float16, row-major
  - `b`: `[k, n]`, float16, row-major
- output:
  - `c`: `[m, n]`, float16, row-major

## Semantics

For every element: `c[i, j] = sum_l a[i, l] * b[l, j]`, with no transpose applied to
either input (PyTorch's plain matmul convention: both operands row-major, `b` consumed
as `[k, n]`). The declared compute_dtype is fp16: the products and their accumulation
run at fp16 tensor-core grade with fp32 accumulation (the standard fp16 tensor-core
contract), and the result is rounded to fp16 once, at the end. Arithmetic below the
declaration (bf16 products, fp16 storage with bf16 math) is a contract violation; a
different fp32 accumulation order (split-k, a different tile order) is legitimate. The
tolerance policy is the gate's relative bound against this reference's own error at the
declared fp16 arithmetic.

## Legal inputs

Any finite float16 values (inputs are drawn at moderate magnitudes so the products and
their sum stay far from fp16's overflow edge). Square-ish matrices: `m = n = k` at
2048, 4096, 8192 in this task's shapes; the operator itself allows independent `m`, `n`,
`k`.
