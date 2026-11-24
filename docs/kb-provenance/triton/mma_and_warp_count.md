# Provenance — `mma_and_warp_count` and `num_stages_pipelining` (hw_kb/triton/sm90/)

## Where these entries came from
- Warpgroup MMA on Hopper and the "warp count is structural" reading: the PTX ISA's
  `wgmma.mma_async` family and the Triton documentation's `tl.dot` / `num_warps` surface. The
  statement that a warp count which is not a multiple of four cannot fill a warpgroup is a
  mechanism argument from the instruction's warp scope, not a measurement.
- `num_stages` as a shared-memory budget and the "short loops pay the prologue" caveat: the Triton
  documentation for the launch parameter, the `cp.async` material in the PTX ISA, and the public
  MIT-licensed Triton guide's persistent-kernel section (pipeliner not carrying across an outer
  loop), reworded; the guide's measured numbers were not carried over.
- The statement that Hopper keeps the accumulator in registers (no TMEM) follows from the
  architecture: TMEM is part of the fifth-generation MMA path the target uses.

## Status
`[PENDING-CERT]` for both. To certify: a compile-only probe recording the mnemonic family per
`(tile, num_warps)` cell, and one stage-count sweep with the card's clock.
