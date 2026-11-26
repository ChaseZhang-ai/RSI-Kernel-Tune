# Provenance — `gpu_probe_ladder` (hw_kb/triton/sm100/)

## Where the entry came from
Written for this repository: it transcribes the probe discipline the other two knowledge-base trees
already follow (rung 0 structure / rung 1 contract / rung 2 price, and what each rung may certify)
into Triton terms, with the compile-only rung implemented by `src/rsi_kernel_tune/triton_check.py`
(`triton.compile` + an explicit `GPUTarget`, no launch).

The `PROBE PASS` / `PROBE FAIL` / `NO VERDICT` verdict-line contract is the loop's own probe
protocol (the shared student prompt), not third-party material.

## Status
`[PENDING-CERT]` — it is a methodology entry; its "claims" are procedural and are certified by the
first entry that walks the ladder and records the verdict.
