# `tl.dot`'s tile picker, `num_warps`, and where the accumulator lives

> **Certification source: [PENDING-CERT] written from the public Triton release notes and a
> public MIT-licensed Triton guide; not certified by a GPU probe in this repository.**
> Detailed provenance archive: docs/kb-provenance/triton/dot_tile_and_warp_count.md (for human review; not in any agent read set).

## Purpose
A `tl.dot` has no single lowering: the compiler picks an MMA form from `(M, N, K, operand dtypes,
warp count, target)` and places the fp32 accumulator either in registers or in Blackwell's tensor
memory (TMEM). Both choices are visible before anything is timed — in the PTX the compiler emits.

## When needed (trigger)
- Raising `num_warps` to fix a register-pressure / occupancy complaint made the kernel *slower*.
- The kernel refuses to compile with `OutOfResources: tensor memory`.
- `tl.dot` operands are fp8/fp16/bf16 and the tile is small in N: the chosen primitive's per-cycle
  throughput, not occupancy, is what the reading tracks.

## Minimal runnable example
```python
import triton
import triton.language as tl


@triton.jit
def _acc_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b)                     # fp32 accumulator; M >= 64 puts it in TMEM on CC 10.x
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc)
```
Read the lowering, do not guess it (no GPU needed to look at the artefacts):
```python
# Keep the compiled artefacts next to the kernel, then read the mnemonic:
#   TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR=./.triton-cache python3 kernel_job.py
#   grep -o 'mma[a-z0-9._]*' .triton-cache/*/*.ptx | sort -u      # e.g. mma.sync, tcgen05.mma, wgmma
#   grep -c 'tmem' .triton-cache/*/*.ttgir                        # >0: the accumulator is in TMEM
```
A compile-only probe (no launch) can read the same thing from Python:
```python
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
import triton
src = ASTSource(fn=_acc_kernel, signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "out_ptr": "*fp32",
                                           "M": "constexpr", "N": "constexpr", "K": "constexpr"},
                constexprs={"M": 64, "N": 64, "K": 128})
compiled = triton.compile(src, target=GPUTarget("cuda", 100, 32), options={"num_warps": 4})
print(len(compiled.asm["ptx"]), "tmem" in compiled.asm["ttgir"])
```
(`python -m forgekernel.triton_check <file> --target 10.0 --kernel <name> --signature '...'
--constexprs '...'` wraps exactly this for a kernel file.)

## Gotchas
- **The warp count is part of the picker's input.** At a small `N` tile, raising `num_warps` gives
  each warp a *smaller* sub-tile, and the smaller primitive's per-cycle throughput can be lower on
  the target than the one the lower warp count picked — theoretical occupancy rises while the
  measured duration rises with it.
- **M ≥ 64 with an fp32 accumulator moves the accumulator to TMEM on CC 10.x.** TMEM is a fixed
  per-SM budget in *columns*, and Triton's allocator gives each declared accumulator tensor its own
  cells for the kernel's lifetime (it does not multiplex them by liveness): a large `M × D` fp32
  accumulator can exceed the budget whatever the loop structure, and the compile fails with
  `OutOfResources: tensor memory`.
- **A single very large `K` in one `tl.dot` can misalign the operand descriptor** and fail at run
  time (a misaligned-address fault on the first launch) where the same math split into several
  chained `tl.dot(..., acc=...)` calls compiles and runs. When the fault's shape is "one big dot",
  split `K`.
- **Changing operand dtypes changes the picked primitive too.** fp8 vs fp16 vs tf32 are different
  instruction families with different K-granularity and accuracy (see `fp32_accum_and_precision.md`).
- The artefacts above are the compiler's, and the board's fence keeps the *kernel* from naming the
  cache — read them from a probe script, not from the kernel file.

## Sources
- Triton release notes for the Blackwell backend work (TMEM encodings, `tcgen05.cp`, distributed
  layouts for `tcgen05.ld/st`, fifth-generation MMA, Gluon multi-CTA): the `triton-lang/triton`
  release notes for the version in use.
- API symbols: `tl.dot`, `tl.dot(..., acc=)`, `tl.zeros(dtype=tl.float32)`, `num_warps`,
  `triton.compile`, `triton.backends.compiler.GPUTarget`, `triton.compiler.compiler.ASTSource`.
- Detailed provenance archive: docs/kb-provenance/triton/dot_tile_and_warp_count.md (for human review; not in any agent read set).
