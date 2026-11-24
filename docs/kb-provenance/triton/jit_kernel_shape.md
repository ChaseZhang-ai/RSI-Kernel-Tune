# Provenance — `jit_kernel_shape` (hw_kb/triton/sm100/jit_kernel_shape.md)

## Where the entry came from
- The one-file / `entry_point: kernel.py::run` / `dependencies: ["triton >= 2.3"]` contract is the
  suite's own published solution format (SOL-ExecBench `docs/solution.md`, "Triton Solution").
- The kernel + `[grid]` launch form and `triton.cdiv` are the Triton documentation's "Getting
  Started" material.
- The "nothing device-touching at import time" and "first launch compiles" statements are written
  from the evaluation protocol the card implements (import for the signature check; warmup before
  the timed window), not from a measurement here.

## What the entry does NOT claim
No timing, no compilation cost figure, no per-release API guarantee beyond the surfaces named.

## Status
`[PENDING-CERT]`. To certify: a probe that imports a kernel file with no device available and then
launches once on the target, recording the first-launch compile time inside the job.
