# Multi-stage reductions: the 2D-tile merge and the split-then-combine form

> **Certification source: [PENDING-CERT] written from the public Triton documentation and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/reduce_two_pass_and_atomics.md (for human review; not in any agent read set).

## Purpose
A reduction that does not fit one block is built from two kernels: per-program partials, then a
merge. The merge's form (a 2D tile loaded and reduced at once versus a scalar loop over the
partials) and the choice between an atomic and a second kernel decide the epilogue's cost.

## When needed (trigger)
- A softmax-like reduction (max, then `exp`, then a weighted sum) spread over several CTAs.
- The merge kernel's ledger is dominated by many small scalar loads.
- Non-determinism appears between runs of the same input: an `atomic_add` on floats has no fixed
  summation order.

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _merge(p_ptr, po_ptr, out_ptr, S: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per (row, column block): S split partials live along the outer axis."""
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D
    s = tl.arange(0, S)                                   # [S]
    # One 2D load of all partials per column, one reduction: not S scalar loads.
    po = tl.load(po_ptr + row * S * D + s[:, None] * D + d[None, :], mask=mask[None, :], other=0.0)
    acc = tl.sum(po.to(tl.float32), axis=0)                # [BLOCK_D]
    tl.store(out_ptr + row * D + d, acc, mask=mask)
```
- The merge reads `S × BLOCK_D` values per program in one go; the compiler streams the outer axis
  and keeps a per-thread partial. It needs `S × BLOCK_D × 4 bytes` of registers (roughly ≤ 8 KB is
  comfortable; beyond ~16 KB assume spilling and shrink `BLOCK_D` or keep the scalar loop).
- Atomic alternative, when one second kernel is more than the merge is worth:
  ```python
  tl.atomic_add(out_ptr + row * D + d, acc, mask=mask, sem="relaxed", scope="gpu")
  ```
  with the output zero-initialised first (`torch.zeros`), and the determinism cost understood.

## Gotchas
- **An fp32 `atomic_add` reduction is order-nondeterministic.** Two runs of the same input can
  differ in the last bits; a card measured against a reference with a tight tolerance can pass one
  run and fail the next. A separate merge kernel is deterministic — prefer it when the tolerance is
  tight.
- **`tl.sum` reduces one axis at a time.** Reducing two axes is two calls (or a `reshape` first);
  the reduce's axis order changes the register traffic.
- **Accumulate in fp32** (`.to(tl.float32)` before the reduce): a bf16/fp16 accumulator loses the
  low bits of every partial and the error grows with `S`.
- **The partial buffers are real memory**: `S × rows × D` fp32 for the partials plus the output.
  A shape whose `D` is wide can make the partial buffer larger than the input — the reason a
  two-kernel design sometimes loses to a wider single kernel.
- **The atomic path needs the output zeroed inside the timed call** (or carried by the same kernel):
  a stale value from the previous timed call is a silent wrong answer under the card's protocol.

## Sources
- Triton documentation — `tl.sum`, `tl.atomic_add` (`sem`, `scope`), reductions:
  https://triton-lang.org/main/python-api/
- CUDA C++ Programming Guide — atomics and floating-point non-associativity.
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- API symbols: `tl.sum`, `tl.atomic_add`, `tl.zeros`, `tl.arange`.
- Detailed provenance archive: docs/kb-provenance/triton/reduce_two_pass_and_atomics.md (for human review; not in any agent read set).
