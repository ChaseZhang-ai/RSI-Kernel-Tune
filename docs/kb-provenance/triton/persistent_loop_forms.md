# Provenance — `persistent_loop_forms` (hw_kb/triton/sm100/persistent_loop_forms.md)

## Where the entry came from
Ported and reworded from the public MIT-licensed Triton guide
(`templates/skills/triton/triton.md`, "Persistent kernels + work-stealing — Triton 3.6 gotchas"):
- `return` inside a `while`/`for` body is rejected by the compiler; the sentinel-flag skeleton is
  the replacement;
- the pipeliner does not carry across outer-`while` iterations, so each task pays the
  `num_stages` prologue and a plain `program_id` dispatch can be faster at small task counts;
- a spin-wait at one CTA per SM stalls the whole SM, and oversubscribing the grid (or a `.cv`
  cache-modifier load) is the workaround.
The guide's measured figures (the 1.2-1.3x dispatch comparison, the -14x spin-wait subset) were
**not** carried into the entry — they are readings of another kernel on another build.

The `sem` / `scope` reminder is from the Triton documentation of `tl.atomic_add`.

## Status
`[PENDING-CERT]`. To certify: a probe that launches the sentinel-flag skeleton (rung 1) and an A/B
of `grid=(num_SMs,)` vs `grid=(N_TASKS,)` at the target's task count (rung 2).
