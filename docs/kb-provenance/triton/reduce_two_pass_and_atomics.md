# Provenance — `reduce_two_pass_and_atomics` (hw_kb/triton/sm100/reduce_two_pass_and_atomics.md)

## Where the entry came from
- The 2D-tile merge vs scalar `tl.static_range` comparison, the register-budget ceilings (≈ 8 KB
  comfortable, 8–16 KB measure it, > ~16 KB assume a spill) and the preference for `num_warps=1`
  with one output element per thread on tiny reduces were ported and reworded from the public
  MIT-licensed Triton guide (`templates/skills/triton/triton.md`, "Split-K reduce kernels"). The
  guide's measured figures were not carried over; the ceilings are stated as sizing aids.
- `tl.sum`'s single-axis semantics and `tl.atomic_add`'s `sem` / `scope` are the Triton
  documentation's API surface.
- The determinism and zeroing statements are from the CUDA C++ Programming Guide's floating-point
  non-associativity material, plus the card protocol's requirement that every timed call start from
  the same state.

## Status
`[PENDING-CERT]`. To certify: a rung-2 A/B of the merged form against the scalar loop on the target,
and a rung-1 probe that runs the atomic path twice on the same input to show the (non)determinism.
