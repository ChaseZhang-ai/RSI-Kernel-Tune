# TMA descriptors and `warp_specialize` on Blackwell (and when they lose)

> **Certification source: [PENDING-CERT] written from the public Triton 3.6+ documentation /
> release notes and a public MIT-licensed Triton guide; not certified by a GPU probe in this
> repository.** Detailed provenance archive: docs/kb-provenance/triton/blackwell_tma_and_warp_specialize.md (for human review; not in any agent read set).

## Purpose
`tl.make_tensor_descriptor` (TMA bulk copies to shared memory) and `tl.range(..., warp_specialize=True)`
(warp-specialized pipelining) are the two "advanced load" levers a Blackwell kernel is expected to
reach for; each has a cost model that is not visible in its surface API.

## When needed (trigger)
- A matmul-shaped kernel whose tiles are large (≳ 64 KB per CTA) and whose loads are the wall.
- A pure `acc += tl.dot(A, B)` accumulator loop that you want the compiler to warp-specialize.
- A kernel that fails with `RuntimeError: PassManager::run failed` in `make_ttgir` after adding
  `warp_specialize=True`.

## Minimal runnable example
```python
import torch
import triton
import triton.language as tl


@triton.jit
def _tma_copy(desc, out_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr):
    """Persistent grid: the descriptor is set up once per CTA, then reused across tiles."""
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    n_tiles = n_m * n_n
    for t in tl.range(pid, n_tiles, tl.num_programs(0)):
        m = (t // n_n) * BLOCK_M
        n = (t % n_n) * BLOCK_N
        blk = desc.load([m, n])                       # cp.async.bulk.tensor -> smem, no L1
        offs = (m + tl.arange(0, BLOCK_M))[:, None] * N + (n + tl.arange(0, BLOCK_N))[None, :]
        tl.store(out_ptr + offs, blk)
```
The host side creates the descriptor (`triton.tools.tensor_descriptor.TensorDescriptor.from_tensor`
in the release that exposes it, or `tl.make_tensor_descriptor` inside the kernel); check the
installed release's API before writing either form — see the smoke check below.

A compile-only check of a kernel that takes a descriptor needs the descriptor's own type spelling in
the signature (the JIT derives it at launch from the `TensorDescriptor` object, which a probe does
not have):
```bash
python -m rsi_kernel_tune.triton_check kernel.py --target 10.0 --kernel _tma_copy \
    --signature '{"desc": "tensordesc<fp32[64,64]>", "out_ptr": "*fp32"}' \
    --constexprs '{"M": 256, "N": 256, "BLOCK_M": 64, "BLOCK_N": 64}'
```

## Gotchas
- **`warp_specialize=True` is restricted to pure MMA-accumulator loops.** A loop body with
  non-MMA work between two `tl.dot`s in the same iteration (a softmax interlude, a reduce, a store)
  aborts in the `ttgir` pass with `PassManager::run failed` and no further diagnostic. Confirmed
  failing shapes: a load → reduce → store kernel (no MMA at all), and an attention-shaped
  load → dot(QK) → softmax → dot(PV) loop. Confirmed working: `acc += tl.dot(A, B)` with nothing
  between the dots.
- **A TMA descriptor built in a non-persistent grid costs more than it saves.** Per-CTA descriptor
  setup dominates at small tiles; amortize it over a grid-stride loop (the example above).
- **TMA pays for matmul-sized tiles, not for streaming.** Below roughly 16 KB per CTA a plain
  wide `tl.load` (many concurrent CTAs, several 128-bit loads each) beats one bulk-DMA-then-wait
  per CTA, even with `num_stages` of pipelining over the descriptor.
- **TMA bypasses L1**: data lands in shared memory. A kernel that relied on L1 reuse across
  neighbouring CTAs loses it — the L2 story (`grid_axis_dispatch_order.md`) becomes the only reuse.
- **Smoke before committing a round** to either lever: a compile-only run for `warp_specialize`
  (`PassManager::run failed` at compile time means this release does not accept the pattern) and
  one job at the largest shape for the descriptor form.

## Sources
- Triton release notes — Blackwell backend work (`tcgen05.cp`, TMEM encodings, warp-specialization
  compiler changes, Gluon multi-CTA): the `triton-lang/triton` release notes for the version in use.
- Triton documentation — `tl.make_tensor_descriptor`, `tl.range(..., warp_specialize=...)`,
  `triton.tools.tensor_descriptor` (naming depends on the release; verify on the image):
  https://triton-lang.org/main/python-api/
- PTX ISA — `cp.async.bulk.tensor` (the instruction the descriptor load lowers to):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- Detailed provenance archive: docs/kb-provenance/triton/blackwell_tma_and_warp_specialize.md (for human review; not in any agent read set).
