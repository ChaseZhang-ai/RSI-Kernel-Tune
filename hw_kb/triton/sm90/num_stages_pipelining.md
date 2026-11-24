# Software pipelining and `num_stages` on Hopper

> **Certification source: [PENDING-CERT] written from the public Triton documentation and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/num_stages_pipelining.md (for human review; not in any agent read set).

## Purpose
`num_stages` is how many copies of the loop's loaded tiles the compiler keeps in flight: each extra
stage is another shared-memory buffer plus a deeper asynchronous copy pipeline. It is a shared-memory
budget decision expressed as a latency-hiding decision.

## When needed (trigger)
- A tiled mainloop on a Hopper part is latency-bound (the load waits are visible in an NCU stall
  breakdown) and shared memory is not the constraint yet.
- The launch is refused for shared memory, or the compiler silently reduces the stage count.
- A loop whose *first* iteration's prologue is a large share of the time (a short loop).

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_sum(x_ptr, out_ptr, n, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in tl.range(0, TILES):                       # the pipeliner works on this loop
        offs = row * TILES * BLOCK + t * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * BLOCK + tl.arange(0, BLOCK), acc)


def run(x, rows):
    out = torch.empty((rows, 1024), dtype=torch.float32, device=x.device)
    _tiled_sum[(rows,)](x, out, x.numel(), TILES=x.shape[1] // 1024, BLOCK=1024,
                        num_stages=4)                  # 4 buffers of 1024 fp32 = 16 KB of smem
    return out
```
- Shared memory per CTA is roughly `num_stages × BLOCK × sizeof(loaded dtype)` per pipelined load:
  the arithmetic above is the budget you are spending.
- `tl.range(..., num_stages=k)` sets the depth for one loop when the launch-level value is not what
  you want for it (a second, shorter loop in the same kernel usually wants its own).

## Gotchas
- **A stage count that exceeds shared memory does not error — the compiler lowers it.** The reading
  then shows the kernel you *did not* ask for, with no warning to explain it; verify the effective
  depth before attributing a change to it.
- **Short loops pay the prologue.** With `TILES` small, the `num_stages`-deep prologue plus epilogue
  is a large fraction of the loop; fewer stages or a non-persistent dispatch is then faster.
- **The pipeliner does not carry across an outer loop** in a persistent kernel (each task re-prologs)
  — see `hw_kb/triton/sm100/persistent_loop_forms.md` for what that costs.
- **A load that is not in the loop's own induction sequence does not get pipelined** (an address
  computed from a data-dependent value, a load inside a branch, an atomic). Only the compiler's
  recognized streaming pattern is prefetched.
- **Shared memory is shared with the MMA's operand stages** on Hopper: the stages the
  `tl.dot` consumes and the stages the loads fill are the same budget.

## Sources
- Triton documentation — `num_stages` (launch parameter), `tl.range(..., num_stages=...)`:
  https://triton-lang.org/main/python-api/
- PTX ISA — `cp.async` (the asynchronous copy the pipeline is built from):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- API symbols: `num_stages`, `tl.range`, `tl.load`, `tl.zeros`.
- Detailed provenance archive: docs/kb-provenance/triton/num_stages_pipelining.md (for human review; not in any agent read set).
