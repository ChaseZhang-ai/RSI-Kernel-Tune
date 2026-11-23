# The masked vector load / store: what the mask and the hints cost

> **Certification source: [PENDING-CERT] transcribed from `hw_kb/triton/sm100/masked_vector_load_store.md`
> into this tree; the mechanism names no architecture, and no GPU probe in this repository has
> certified it. The coalescing floor quoted below was written for the target architecture's cache
> line and is a starting point here, not a reading.**
> Detailed provenance archive: docs/kb-provenance/triton/masked_vector_load_store.md (for human review; not in any agent read set).

## Purpose
`tl.load` / `tl.store` on a block of offsets lower to per-thread vector loads; whether they become
one wide (128-bit) access per thread or a scatter of narrow ones is decided by the **mask, the
alignment the compiler can prove, and the contiguity of the offsets** — not by the request size.

## When needed (trigger)
- A streaming / elementwise kernel sits far below the bandwidth roofline with no NCU stall reason
  that explains it.
- `tl.load` of a tile whose row stride is not a multiple of the vector width.
- A store whose per-row bytes fall below a cache line's fraction (a narrow output dtype written
  with a wide row stride).

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # The hints are promises the compiler uses to pick a wide access; a wrong promise is wrong code.
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    _copy[(triton.cdiv(n, 4096),)](x, out, n, BLOCK=4096)
    return out
```
- `tl.multiple_of(offs, BLOCK)` promises `offs` is a multiple of `BLOCK` (true when the base pointer
  is aligned and `BLOCK` divides the stride); `tl.max_contiguous(offs, BLOCK)` promises `BLOCK`
  contiguous elements. Together they are what lets the compiler emit the widest access.
- `other=` is the value masked-off lanes see; leaving it out gives undefined lanes, which is fine
  only when those lanes are never stored.

## Gotchas
- **A wrong hint is silent miscompilation, not an error.** Only promise alignment the caller
  guarantees; when the base pointer's alignment is not known, do not make the promise and accept a
  narrower access.
- **The mask costs a predicate per access.** When the shape is a multiple of `BLOCK` (the usual
  case for a table-keyed `BLOCK`), the unmasked form is both simpler and faster; keep the masked
  form for the tail (the last program) or for the shapes that need it.
- **Coalescing is per row, not per kernel.** A store of `H` rows of `D_CHUNK × sizeof(dtype)` bytes
  fragments into sub-cache-line writes when the per-row size drops below ~64 B, however large the
  total is and however many CTAs you launch. Prefer a per-row size of at least 64 B (128 B when the
  register budget allows).
- **`cache_modifier` / `eviction_policy` are mechanisms, not knobs to sprinkle**: `cache_modifier=".cg"`
  bypasses L1, `.cv` re-reads volatile, `eviction_policy="evict_first"` / `"evict_last"` steer the
  L2 replacement for streaming data. Use one, then measure; two at once cannot be attributed.
- A `tl.load` of a *transposed* tile (row-major pointer arithmetic with the fast axis on the
  column) is the classic shape where the access degrades: the fix is the layout, not the mask.

## Sources
- Triton documentation — `tl.load` / `tl.store` (mask/other/cache modifiers), `tl.multiple_of`,
  `tl.max_contiguous`: https://triton-lang.org/main/python-api/
- CUDA C++ Programming Guide — "Device Memory Accesses" (coalescing, 32/64/128-byte transactions).
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- API symbols: `tl.load`, `tl.store`, `tl.multiple_of`, `tl.max_contiguous`, `tl.arange`.
- Detailed provenance archive: docs/kb-provenance/triton/masked_vector_load_store.md (for human review; not in any agent read set).
