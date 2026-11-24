# The one-file kernel contract (`@triton.jit` kernels + `run`)

> **Certification source: [PENDING-CERT] written from the public Triton documentation and the
> suite's published solution format; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/jit_kernel_shape.md (for human review; not in any agent read set).

## Purpose
A Triton solution is **one Python module** whose compute is `@triton.jit` kernels and whose entry
is a top-level `run(...)` that launches them; this entry is the shape of that module and what the
card checks about it.

## When needed (trigger)
- Writing the first version of a Triton kernel file.
- `SIGNATURE: run(...) — the definition wants run(...)`: the entry's parameters drifted from the
  definition's order (the card compares names and positions, not defaults).
- A kernel that works in a notebook but not in the job: the module itself must be importable by a
  plain `import` with no device, no `__main__` guard and no side effects beyond definitions.

## Minimal runnable example
```python
"""One module: kernels above, the entry below. Nothing runs at import time."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * SCALE, mask=mask)


def run(x, scale):
    """The entry: same positional parameters as the definition, in its order."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _scale_kernel[grid](x, out, n, SCALE=scale, BLOCK=1024)
    return out
```
- `constexpr` parameters are passed **by keyword** at the launch (`SCALE=`, `BLOCK=`), positional
  arguments in the kernel's declared order.
- The grid is a tuple; `triton.cdiv` is the ceiling division the launcher expects.
- A destination-passing-style card adds the output to `run`'s parameters (last) and the kernel
  writes it; a value-returning card returns the tensor(s). Follow the card's stub — it is generated
  from the definition and its signature is what the card checks.

## Gotchas
- **The first launch compiles the kernel.** That compile time is *not* in the timed window (the
  card warms up before timing), but it is inside the job's clock and inside the board's time limit:
  a file with many `constexpr` specializations compiles each one it launches.
- **Nothing at import time may touch a device.** The card imports the module to check its
  signature and to read the entry; a `torch.cuda.current_device()` or a `tl.dot` at module level
  breaks every reading, not just the first.
- **`run` is checked by position and count**, so an extra optional parameter (`def run(x, w, eps=1e-6)`)
  is still five-argument-shaped only if the definition has it: read the stub's signature.
- **A Python-level helper that is not `@triton.jit` cannot be called from inside a kernel** — the
  compiler only sees jitted functions (`@triton.jit` on a helper that returns a `tl` expression is
  the supported form, and it may not return from inside control flow in older releases).

## Sources
- Triton documentation — "Getting Started" / "Vector Addition" (the kernel + `[grid]` launch form):
  https://triton-lang.org/main/getting-started/tutorials/
- API symbols: `triton.jit`, `triton.cdiv`, `triton.language.program_id`, `tl.arange`, `tl.load`,
  `tl.store`.
- The suite's published solution format (one Python file, `entry_point: kernel.py::run`,
  `dependencies: ["triton >= 2.3"]`): SOL-ExecBench `docs/solution.md`, "Triton Solution".
- Detailed provenance archive: docs/kb-provenance/triton/jit_kernel_shape.md (for human review; not in any agent read set).
