# Hopper `tl.dot`: warpgroup MMA and the warp count

> **Certification source: [PENDING-CERT] written from the public Triton documentation / Hopper
> material and a public MIT-licensed Triton guide; not certified by a GPU probe in this
> repository.**
> Detailed provenance archive: docs/kb-provenance/triton/mma_and_warp_count.md (for human review; not in any agent read set).

## Purpose
On Hopper a `tl.dot` lowers to a warpgroup MMA (wgmma) — one instruction issued by a group of four
warps — so the *warp count* is structural: a `num_warps` that is not a multiple of 4 either cannot
use the warpgroup path or pairs two warpgroups in ways the compiler decides for you.

## When needed (trigger)
- A matmul-shaped kernel: is the chosen `num_warps` giving each warp a tile it can feed?
- `num_warps=2` (or any non-multiple of 4) on a kernel whose body contains a large `tl.dot`.
- A tile shape whose `M` is smaller than the warpgroup's own granularity.

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _acc(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


def run(a, b):
    m, k = a.shape
    out = torch.empty((m, b.shape[1]), dtype=torch.bfloat16, device=a.device)
    # 4 warps = one warpgroup. 8 (two warpgroups) is the other candidate; both are multiples of
    # four, which is what the warpgroup path needs.
    _acc[(1,)](a, b, out, M=m, N=b.shape[1], K=k, num_warps=4, num_stages=2)
    return out
```
Read which instruction ran (no GPU needed):
```python
#   TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=./.triton-cache python3 kernel_job.py
#   grep -o 'wgmma[^ ]*' .triton-cache/*/*.ptx | sort -u        # Hopper's warpgroup MMA
```

## Gotchas
- **`num_warps` must be a multiple of 4 for the warpgroup path.** A count that is not splits the
  work in a way the compiler may lower to a per-warp `mma.sync` fallback — correct, and slower than
  the reading you thought you were taking.
- **The tile must be able to fill the warpgroup's `M`.** A `tl.dot` with a small `M` (as used by
  attention's QK) either pads or drops to another instruction family; padding costs compute, and
  the fallback changes the accumulation path.
- **The accumulator lives in registers** on Hopper — there is no TMEM. An accumulator that would
  not fit is a spill to local memory, which the `-Xptxas=-v`-style reports would show as
  `local_bytes`; watch for it when a tile grows.
- **The pipelined operand path is `cp.async` + a shared-memory stage** (see
  `num_stages_pipelining.md`); wgmma reads the operand from shared memory, so a stage count that is
  too small starves the MMA and a stage count that is too large costs shared memory (and can push
  occupancy down).

## Sources
- Triton documentation / Hopper (wgmma) material: https://triton-lang.org/main/python-api/
- PTX ISA — `wgmma.mma_async` (the warpgroup MMA family):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- API symbols: `tl.dot`, `out_dtype`, `num_warps`, `num_stages`.
- Detailed provenance archive: docs/kb-provenance/triton/mma_and_warp_count.md (for human review; not in any agent read set).
