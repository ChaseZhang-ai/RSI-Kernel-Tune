# Provenance — `dot_tile_and_warp_count` (hw_kb/triton/sm100/dot_tile_and_warp_count.md)

## Where the entry came from
- The compiler-side statements (a `tl.dot`'s lowering depends on `(M, N, K, dtypes, warp count,
  target)`; TMEM encodings and fifth-generation MMA work in the Blackwell backend) are the public
  Triton release notes for the version in use.
- The empirical claims were ported and reworded from the public MIT-licensed Triton guide
  (`templates/skills/triton/triton.md`, "num_warps can degrade MMA throughput at small-N fp8
  tiles", "Blackwell M=64 fp32-acc — tcgen05/tmem path + count-permanent allocator"):
  - at a small-N fp8 tile, raising `num_warps` from 2 to 4 picks a smaller per-warp sub-tile whose
    per-cycle throughput is lower, so theoretical occupancy rises while duration rises;
  - with `M >= 64` and an fp32 accumulator the accumulator moves to TMEM; TMEM is a fixed column
    budget per SM and the allocator gives each declared accumulator tensor permanent cells (no
    liveness multiplexing), so the budget can be exceeded regardless of loop structure;
  - a single very large-`K` `tl.dot` can misalign the operand descriptor and fault at launch, while
    the same math split into chained `tl.dot(..., acc=)` compiles and runs.
  The guide's numbers (registers, occupancy percentages, 512 columns, the 528 -> 520 cells table)
  were **left out** of the entry: they are readings on a different Triton build and not
  re-measurable here.
- The folding into one operator-agnostic entry (and the "read the PTX/ttgir instead of guessing"
  procedure, including the compile-only `ASTSource` + `GPUTarget` probe) is this repository's.

## Status
`[PENDING-CERT]`. To certify: a compile-only probe on the target recording which mnemonic each
`(tile, dtype, num_warps)` cell selects, plus one A/B duration per contested cell.
