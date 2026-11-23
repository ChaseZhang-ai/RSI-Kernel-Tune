# Deterministic concatenation: `tl.join` + `tl.permute` + `tl.reshape`

> **Certification source: [PENDING-CERT] written from the public Triton documentation and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/tl_join_permute_concat.md (for human review; not in any agent read set).

## Purpose
Build a larger register tile from two tiles **in a known element order**. `tl.cat` concatenates
without a guaranteed order; `tl.join` + `tl.permute` + `tl.reshape` is the ordered spelling.

## When needed (trigger)
- Two halves of a tile must be assembled positionally (a packed layout, an interleaved layout, a
  block of a block-matrix) and a later `tl.store` or `tl.dot` depends on which half is where.
- `tl.cat` "worked" at a small test shape and produced wrong numbers at the real shape: its order
  is unspecified, so a small shape can pass by luck.

## Minimal runnable example
```python
import triton
import triton.language as tl


@triton.jit
def _concat(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    a = tl.load(a_ptr + rm[:, None] * N + rn[None, :])     # [M, N]
    b = tl.load(b_ptr + rm[:, None] * N + rn[None, :])     # [M, N]

    joined = tl.join(a, b)                                 # [M, N, 2]
    hcat = tl.reshape(tl.permute(joined, (0, 2, 1)), (M, 2 * N))    # [M, 2N]: columns a | b
    vcat = tl.reshape(tl.permute(joined, (2, 0, 1)), (2 * M, N))   # [2M, N]: rows a ; b

    r2 = tl.arange(0, 2 * N)
    tl.store(out_ptr + rm[:, None] * (2 * N) + r2[None, :], hcat)
```
- `tl.join` stacks along a **new minor dimension**; the `permute` is what puts the two halves on
  the axis you are concatenating along, and the `reshape` is what collapses it back.

## Gotchas
- **`tl.cat` is order-unstable by definition** (its documented implementation supports only
  `can_reorder=True`). Use it only where the downstream math is order-invariant (a sum, a max).
- **A built tile costs registers**: `[64, 64]` bf16 is ~8 KB in flight per CTA. In a kernel already
  register-limited, verify the assembly does not push it over.
- **Zero padding is not free**: assembling a block-diagonal / triangular tile from per-block tiles
  wastes compute on padding; it wins only when the valid fraction is high enough to pay for it.
- The `reshape` needs the permuted shape to be contiguous **in the order the permute produced** —
  a mistyped permutation compiles and silently produces the other concatenation, so the example
  above (whose two lines differ only in the permutation) is the pair to test against.

## Sources
- Triton documentation — `tl.join`, `tl.permute`, `tl.reshape`, `tl.cat` (its `can_reorder`
  requirement): https://triton-lang.org/main/python-api/
- API symbols: `tl.join`, `tl.permute`, `tl.reshape`, `tl.cat`.
- Detailed provenance archive: docs/kb-provenance/triton/tl_join_permute_concat.md (for human review; not in any agent read set).
