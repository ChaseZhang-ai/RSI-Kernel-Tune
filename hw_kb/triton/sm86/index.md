# hw_kb/triton/sm86 — Ampere (RTX 3060), a DEVELOPMENT / SMOKE tree

> **This is a development and smoke target, never a production one.** A run here answers "does the
> call path work: build, correctness, timing, the whole card" in minutes on a small card. Its
> numbers are not readings of the target (`hw_kb/triton/README.md`).

## Look up by symptom

| symptom | entry |
|---|---|
| "which `arch` do I put in task.toml for a local smoke run?" / "can this kernel even run here?" | [`dev_target_contract`](dev_target_contract.md) |
| a kernel that must run on this card and on the target | [`smoke_kernel_shape`](smoke_kernel_shape.md) |
| an "L2-friendly" grid reorder changed nothing | [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) |
| a streaming kernel far below the bandwidth roofline | [`masked_vector_load_store`](masked_vector_load_store.md) |

## Completion status

| entry | covers | status |
|---|---|---|
| [`dev_target_contract`](dev_target_contract.md) | what a smoke run may and may not conclude; 6 GB; no TMA/tcgen05/clusters | [PENDING-CERT] |
| [`smoke_kernel_shape`](smoke_kernel_shape.md) | one source both cards take; queried launch parameters | [PENDING-CERT] |
| [`grid_axis_dispatch_order`](grid_axis_dispatch_order.md) | `program_id(0)` varies fastest | [PENDING-CERT] |
| [`masked_vector_load_store`](masked_vector_load_store.md) | mask cost, hints, coalescing floor (re-measure here) | [PENDING-CERT] |

## To be filled (demand-driven)

- The Ampere MMA family's tile granularity as Triton picks it (this card's `tl.dot` goes to
  `mma.sync`, not `wgmma`/`tcgen05`) — worth one entry the first time a smoke run's numbers are
  being misread.
- What a 6 GB card can hold of a group card's workloads (the shapes that fit, and the rule for
  saying "smoked the smallest shape only").
