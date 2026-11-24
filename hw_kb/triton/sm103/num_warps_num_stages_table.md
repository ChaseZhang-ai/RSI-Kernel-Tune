# Shape-keyed knob table instead of `@triton.autotune`

> **Certification source: [PENDING-CERT] transcribed from `hw_kb/triton/sm100/num_warps_num_stages_table.md`
> into the development-GPU tree; no GPU probe in this repository has certified it. This architecture
> is the DEVELOPMENT target (a kernel is never compiled *for* it), so a table value read here is a
> starting point for the target's table, not the target's value.**
> Detailed provenance archive: docs/kb-provenance/triton/num_warps_num_stages_table.md (for human review; not in any agent read set).

## Purpose
The per-launch knobs (`num_warps`, `num_stages`, `BLOCK`, and any `constexpr` tile size) are chosen
**from the input shape by a plain Python table**, so the job compiles exactly the variants it runs
and no measurement happens inside the timed window.

## When needed (trigger)
- The card's protocol warms up a fixed, small number of calls and flushes L2 before every timed
  call. A tuner that times candidate configs *inside the job* therefore measures on the wrong side
  of the cache flush and spends the job's clock compiling configurations that never run.
- The same kernel must serve several shapes (a group card): a single `BLOCK` is either too small
  for the largest shape or wasteful on the smallest.
- `@triton.autotune` appears in the file (the loop's target-compile check warns about it).

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl

# (num_warps, num_stages, BLOCK) per shape regime, written down, not tuned in the job.
_KNOBS = {
    "small": (4, 3, 1024),
    "large": (8, 4, 4096),
}


@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * SCALE, mask=mask)


def run(x, scale):
    out = torch.empty_like(x)
    n = x.numel()
    warps, stages, block = _KNOBS["large" if n >= 1 << 20 else "small"]
    _scale_kernel[(triton.cdiv(n, block),)](x, out, n, SCALE=scale, BLOCK=block,
                                            num_warps=warps, num_stages=stages)
    return out
```
- The table is keyed on quantities the *entry* already knows (element counts, row width, the
  shape's own axes) — never on a timer.
- `num_warps` / `num_stages` are launch keywords; the `constexpr` tile is a keyword the kernel
  specializes on. One `constexpr` value = one compiled variant.

## Gotchas
- **`num_stages` is a smem/latency trade, not "more is better".** Each extra stage allocates another
  buffer for the pipelined loads; past the SM's shared memory the compiler reduces stages itself or
  refuses the launch.
- **`num_warps` changes more than occupancy.** It changes the *instruction* the tile picker chooses
  for a `tl.dot` (see `dot_tile_and_warp_count.md`), so a warp count that raises theoretical
  occupancy can lower throughput.
- **A `constexpr` value that depends on a runtime value is a recompile at that call.** Two shapes
  in one job that fall into two table rows compile two variants — that is fine when the table is
  small, expensive when it is a sweep.
- **The table must be a module-level constant**, not computed from a measurement: a table built at
  import from `torch.cuda.get_device_properties` keys the same file differently on the development
  GPU and on the target, which is exactly the difference the reading must not contain.

## Sources
- Triton documentation — `num_warps` / `num_stages` as launch parameters and `@triton.autotune`'s
  own semantics: https://triton-lang.org/main/python-api/ (see `triton.jit` / `triton.autotune`)
- API symbols: `num_warps`, `num_stages`, `tl.constexpr`.
- Detailed provenance archive: docs/kb-provenance/triton/num_warps_num_stages_table.md (for human review; not in any agent read set).
