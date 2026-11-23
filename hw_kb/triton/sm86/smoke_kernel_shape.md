# A kernel that runs on both the dev card and the target

> **Certification source: [PENDING-CERT] written from the public Triton documentation; no readings
> are claimed (this tree's card is a development target).**
> Detailed provenance archive: docs/kb-provenance/triton/smoke_kernel_shape.md (for human review; not in any agent read set).

## Purpose
The shape of a kernel file whose *first* run can happen on a small Ampere card and whose *tuned*
run happens on the target: no architecture named in the code, no capability check that changes the
math, and knobs that come from a table the caller can point at either machine.

## When needed (trigger)
- Bringing up a new task: the first version must run somewhere cheap.
- A kernel that compiles here but not on the target (or the reverse) — usually an arch-pinned
  construct or a capability branch that changes the algorithm.

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * SCALE, mask=mask)


def run(x, scale):
    out = torch.empty_like(x)
    n = x.numel()
    props = torch.cuda.get_device_properties(x.device)
    # Both cards run the SAME kernel; only the launch shape is derived, and only from
    # quantities that exist on every device.
    waves = max(1, min(8, triton.cdiv(n, 4096) // max(1, props.multi_processor_count)))
    block = 4096 if props.multi_processor_count >= 32 else 1024
    _scale_kernel[(triton.cdiv(n, block),)](x, out, n, SCALE=scale, BLOCK=block,
                                            num_warps=4, num_stages=2, num_ctas=1)
    return out
```
- The kernel body names no instruction family and no capability.
- Anything the launch needs from the device (`multi_processor_count`, capability) is queried at
  call time; nothing is written down.
- On the target, the same file is checked for compilation against CC 10.0 without running it:
  `python -m forgekernel.triton_check <file> --target 10.0`.

## Gotchas
- **A capability branch that changes the algorithm makes the smoke useless**: the cheap card then
  exercises a path the target never takes. Keep capability-dependent *launch* parameters, not
  capability-dependent math.
- **`num_ctas` (cluster size) is not available on Ampere**; passing it as 1 is the portable form,
  and a cluster larger than 1 belongs behind a target-only task.
- **The kernel's correctness contract must be identical on both cards** — same dtypes, same
  accumulation order; a difference there turns a smoke pass into a false positive.
- **A smoke run measures the dev card's absolute time**; record it as "launched and passed", not as
  a speed.

## Sources
- Triton documentation — `triton.cdiv`, launch parameters, `tl.program_id`:
  https://triton-lang.org/main/python-api/
- PyTorch documentation — `torch.cuda.get_device_properties` / `get_device_capability`:
  https://pytorch.org/docs/stable/cuda.html
- API symbols: `torch.cuda.get_device_properties`, `triton.cdiv`, `num_ctas`, `num_warps`.
- Detailed provenance archive: docs/kb-provenance/triton/smoke_kernel_shape.md (for human review; not in any agent read set).
