# Grid axes and CTA dispatch order (`tl.program_id(0)` varies fastest)

> **Certification source: [PENDING-CERT] written from the public Triton/CUDA documentation and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/grid_axis_dispatch_order.md (for human review; not in any agent read set).

## Purpose
`grid = (G0, G1, G2)` becomes the launch's `gridDim`, and the hardware dispatches CTAs in
`blockIdx.x`-fastest order — so **whatever the kernel binds to `tl.program_id(0)` is the axis whose
neighbours run concurrently** and therefore the axis that decides L2 reuse.

## When needed (trigger)
- A memory-bound kernel whose working set should be shared by concurrent CTAs (paged / tiled data
  where a run of CTAs reads the same pages).
- Any attempt at an "L2-friendly reorder" — which is a no-op unless the *binding* changes too.

## Minimal runnable example
```python
import triton
import triton.language as tl


@triton.jit
def _row_scaled(x_ptr, out_ptr, S, n_rows, n_cols, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    # pid(0) = the row: consecutive CTAs read consecutive rows of the same matrix, so the
    # pages a CTA's neighbours pull in are the pages this CTA wants next.
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offs = row * n_cols + col
    mask = (row < n_rows) & (col < n_cols)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * SCALE, mask=mask)


def run(x, out, scale):
    n_rows, n_cols = x.shape
    grid = (n_rows, triton.cdiv(n_cols, 512))
    _row_scaled[grid](x, out, n_cols, n_rows, n_cols, SCALE=scale, BLOCK=512)
```
The two launchers below are **not** equivalent — the second is a no-op reorder unless the kernel's
`program_id` binding changes with it:
```python
grid_a = (n_rows, n_splits)      # pid(0) = row
grid_b = (n_splits, n_rows)      # swapped tuple: pid(0) is still "row" inside the kernel
```
To make the split the fastest axis, change **which axis `pid(0)` names** in the kernel:
```python
split = tl.program_id(0)          # now the split varies fastest across concurrent CTAs
row = tl.program_id(1)
```

## Gotchas
- **Swapping the grid tuple without re-binding `program_id` changes nothing** (the launched
  dimensions move, the kernel's meaning of axis 0 does not). The reading will be within noise and
  easy to misread as "L2 reuse does not help here".
- **`tl.num_programs(axis)` and `tl.program_id(axis)` are the only ways to read the launch shape**
  inside the kernel; `grid` must be a tuple of ints (a `tl.constexpr`-time constant is not required).
- **The launch's grid is the product of the axes**, so a reorder that moves work between axes must
  keep that product — an overflowed axis silently drops work.
- A persistent kernel replaces the dispatch order entirely (see `persistent_loop_forms.md`): there,
  the *order the loop hands out tasks* is the locality decision, not `program_id`.

## Sources
- CUDA C++ Programming Guide — "Grid of Thread Blocks" / thread-block scheduling order.
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- API symbols: `tl.program_id`, `tl.num_programs`.
- Detailed provenance archive: docs/kb-provenance/triton/grid_axis_dispatch_order.md (for human review; not in any agent read set).
