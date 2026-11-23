# hw_kb/triton/sm90 — Hopper

Entry point for a Triton run on Hopper. The same symptom-first discipline as the target tree; the
entries here are the ones whose mechanism is Hopper's (warpgroup MMA, `cp.async` pipelining) plus
the architecture-neutral copies a run here actually needs.

## Look up by symptom

| symptom | entry |
|---|---|
| an "L2-friendly" grid reorder changed nothing | [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) |
| a streaming kernel far below the bandwidth roofline with no stall reason to explain it | [`masked_vector_load_store`](masked_vector_load_store.md) |
| a `tl.dot` kernel: is the warp count feeding the warpgroup MMA? | [`mma_and_warp_count`](mma_and_warp_count.md) |
| a latency-bound mainloop; the launch refused for shared memory; a short loop's prologue dominates | [`num_stages_pipelining`](num_stages_pipelining.md) |

## Completion status

| entry | covers | status |
|---|---|---|
| [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) | `program_id(0)` varies fastest; reorders that are no-ops | [PENDING-CERT] |
| [`masked_vector_load_store`](masked_vector_load_store.md) | mask cost, alignment/contiguity hints, coalescing floor | [PENDING-CERT] |
| [`mma_and_warp_count`](mma_and_warp_count.md) | warpgroup MMA, warp counts that are multiples of four | [PENDING-CERT] |
| [`num_stages_pipelining`](num_stages_pipelining.md) | `num_stages` as a shared-memory budget, prologue cost | [PENDING-CERT] |

All `[PENDING-CERT]`: transcribed from the target tree or written from public documentation, not yet
earned by a probe of this repository (see `hw_kb/triton/README.md` and
`hw_kb/triton/sm100/gpu_probe_ladder.md`).

## To be filled (demand-driven)

- The target tree's remaining neutral entries, copied here when a run needs them (`tl.join` concat,
  the two-pass reduce, precision), each with its `Certification source` line naming this tree.
- Hopper-specific: the distributed-shared-memory (`cluster`) path a Triton kernel can reach.
- Hopper-specific: `wgmma`'s accumulator-in-registers budget and how the compiler splits a tile
  across two warpgroups.
