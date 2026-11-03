# Recipe: occupancy-scaling curve (grid-scaling)

Script: [`occupancy_scaling.py`](occupancy_scaling.py) — `PROBE_REQUEST: notes/occupancy_scaling.py` (one GPU job).

## Symptom it fits

- `waves_per_sm` low (<1: most SMs never saw work) or fractional (a tail wave);
- both SOL sides low (`compute_sol_pct` / `memory_sol_pct` well under 70) — the
  roofline says "neither wall", and the digest alone cannot split latency vs
  tail;
- a grid-shape decision (tile height, CTAs-per-SM, persistent vs one-shot) is on
  the table and the pricing round has no lever for it yet.

## What it measures

The SAME kernel launched at grid x {0.25, 0.5, 1.0} (loop body unchanged, only
fewer blocks issued; a persistent kernel scales its num-SM-equivalent knob), all
scales interleaved in one job, CUDA events, per-scale median, readings as ratios
to s=1.0.

## How to read it (pre-registered in the script's READ line)

| Curve shape | Reading | Lever direction it names |
|---|---|---|
| `flat` — t(s) ~ constant | **latency-bound**: SMs idle, the critical path lives inside one block/warp | dependency chains, sync points, occupancy floors — not the grid |
| `linear` — t(s) ~ s | **throughput-bound**: SMs are the saturated resource | more parallel work per pass / better SM utilization; shrinking the grid costs proportionally |
| `stepped` — between adjacent scales the smaller-scale time is ≥1.30× its linear share (shrinking the grid frees less than proportional) | **wave quantization**: a wave step sits there; the tail wave dominates | tile height vs SM count divisibility, persistent scheduling, partial-wave absorption |
| `mixed` | no clean shape | re-read with the SOL header + stall ledger before attributing anything |

## Adaptation (required before trusting a reading)

Fill `launch(scale)` with the tree's kernel at its grid x scale. The default
hook is a size-scaled torch streaming **self-check** — it proves the harness
measures, and nothing else. Persistent kernels: scale the resident-CTA / work-item
knob, not a dead grid dimension. Costs one verdict-probe round
(`[loop] max_verdict_probe_rounds` budget).

## Cite it

`via probe:<path>#1` — the single contract line `occupancy_curve_shape=<shape>`.
`max_diff=n/a` on purpose: the reading is a shape, not a number.
