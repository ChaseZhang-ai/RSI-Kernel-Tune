# wp21mini

Elementwise add over float32 vectors — the smallest possible CUDA eval surface, for the
WP21 machine energization smoke only (link & dataflow evidence; never a performance claim).

## Signature

- inputs, in order:
  - `a`: `[n]`, float32
  - `b`: `[n]`, float32
- output:
  - `y`: `[n]`, float32

## Semantics

`y[i] = a[i] + b[i]` for every element.

## Legal inputs

Any finite float32 values; output must be exact (IEEE add).
