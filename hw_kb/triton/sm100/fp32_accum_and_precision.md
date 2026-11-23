# FP32 accumulation, `input_precision`, and rounding once

> **Certification source: [PENDING-CERT] written from the public Triton documentation and the
> CUDA programming guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/fp32_accum_and_precision.md (for human review; not in any agent read set).

## Purpose
Where the accuracy of a Triton kernel is actually decided: the `tl.dot` accumulator dtype, the
`input_precision` of an fp32 `tl.dot`, the order in which a reduction is done, and how many times
the value is rounded on its way to the output dtype.

## When needed (trigger)
- A kernel passes on one seed and fails the card's tolerance on another.
- A bf16/fp16 kernel's error grows with the reduction length.
- An fp32 `tl.dot` is slower than expected (it may be running in TF32 — a legitimate optimization,
  and a precision change that has to be justified against the tolerance).

## Minimal runnable example
```python
import triton
import triton.language as tl


@triton.jit
def _dot_bf16(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])          # bf16
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])          # bf16
    acc = tl.dot(a, b, out_dtype=tl.float32)                    # fp32 accumulation
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))   # round once


@triton.jit
def _dot_f32(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    # input_precision: "ieee" (exact fp32), "tf32" (10-bit mantissa inputs), "tf32x3" (3-pass,
    # near-fp32 with tf32 hardware). The choice is a precision/perf decision, not a default to
    # accept blindly.
    acc = tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)
```
- A tolerant operator can take `input_precision="tf32"` for a real speedup; it must then pass the
  tolerance **on several seeds**, not only the one the board uses.
- `out_dtype=tl.bfloat16` on the `tl.dot` itself halves the accumulator's footprint (TMEM on
  Blackwell) at a real accuracy cost — an option for an accumulator that does not fit, not a
  default.

## Gotchas
- **One rounding at the end**: keep the intermediate math in fp32 and `.to(out_dtype)` once. Casting
  down inside a loop re-quantizes every partial and the error is the sum of them.
- **A reduction's accuracy depends on the order.** A tree reduction (`tl.sum`) is better
  conditioned than an accumulating loop; splitting a sum across CTAs and adding the partials in a
  merge kernel keeps the same order as long as the merge is deterministic (see
  `reduce_two_pass_and_atomics.md`).
- **`eps` inside a reciprocal or a normalize must be in the accumulator's precision** (`1e-6` in
  fp32), or a small denominator loses the term entirely.
- **A `tl.dot` with fp32 inputs is not "the exact path" until you say so** — the compiler may use
  TF32. Reading the PTX mnemonics (see `dot_tile_and_warp_count.md`) is how to know which one ran.
- **A global precision flag is a process-wide mutation**: flipping a torch backend flag inside
  `run` changes every later timed call. Restore it before returning if a library forces it.

## Sources
- CUDA C++ Programming Guide — floating-point rounding and the TF32 tensor-core mode.
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Triton documentation — `tl.dot` (`out_dtype`, `input_precision`), `tl.sum`, `.to`:
  https://triton-lang.org/main/python-api/
- API symbols: `tl.dot`, `input_precision`, `out_dtype`, `tl.float32`, `tl.bfloat16`.
- Detailed provenance archive: docs/kb-provenance/triton/fp32_accum_and_precision.md (for human review; not in any agent read set).
