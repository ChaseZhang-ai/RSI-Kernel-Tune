# Provenance — `fp32_accum_and_precision` (hw_kb/triton/sm100/fp32_accum_and_precision.md)

## Where the entry came from
- `out_dtype`, `input_precision` (`"ieee"` / `"tf32"` / `"tf32x3"`) and `tl.sum` are the Triton
  documentation's `tl.dot` surface; the guide's statement that the accumulator's placement depends
  on `out_dtype` (halving it with a bf16 accumulator) is cross-referenced from
  `dot_tile_and_warp_count.md`.
- The TF32 rounding mode and the non-associativity of floating-point reduction are the CUDA C++
  Programming Guide's material.
- The "round once at the end", "eps in the accumulator's precision" and "a global precision flag is
  a process-wide mutation" statements are written for this repository from the card protocol's
  tolerance rule (max relative error against the definition's reference, never below two units of
  the dtype's rounding) and the requirement that `run` leave no global state changed.

## Status
`[PENDING-CERT]`. To certify: a multi-seed tolerance sweep on the target for one kernel per
`input_precision` value, recorded here with the seeds and the observed max relative error.
