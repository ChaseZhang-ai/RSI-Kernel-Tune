# Recipe: phase ablation (read-only / compute-only / full split)

Script: [`phase_ablation.py`](phase_ablation.py) — `PROBE_REQUEST: notes/phase_ablation.py` (one GPU job).

## Symptom it fits

- both faces of the kernel are plausible suspects (the digest shows tensor_pct
  mid-range, dram_pct mid-range, neither dominant);
- the stall ledger disperses (no single top item) — attribution needs a
  decomposition, not a bigger magnifier;
- "is it the supply path or the mma chain?" is about to be answered by argument
  instead of measurement.

## What it measures

Three variants of the tree's kernel interleaved in one job: **full**, **read
only** (loads the operands, discards them — no math), **compute only** (math
from register-resident data — no loads). CUDA events, medians; the READ line
reports shares of the full time.

## How to read it (pre-registered in the script's READ line)

| Split | Reading | Lever direction it names |
|---|---|---|
| `read ~ full`, compute small | **memory-face bound**: the loads are the kernel | tiling / supply: TMA shape, cp.async staging, L2 policy, sector efficiency |
| `compute ~ full`, read small | **compute-face bound**: the math is the kernel | issue / scheduling: unroll shape, accumulator split, pipe mix |
| `read + compute << full` | **the faces overlap today** — the sum names the floor the overlap hides | dependency shape / anti-phase — a timeline question (`PROBE_REQUEST: timeline`), not another ablation |
| both faces carry real time | mixed — split further (per operand / per segment) before attributing | — |

The analytic floor (when the run carries it) sits next to this naturally: the
read face against the bytes floor, the compute face against t_mma. This recipe
does not depend on it — it stands alone.

## Adaptation (required before trusting a reading)

Fill `variants()` with the tree's own full kernel plus its two degradations.
The defaults are built-in streaming / FMA torch **self-checks** — they prove the
harness measures and say nothing about the operator. Keep the degraded variants
launch-shaped like the original (same grid/smem skeleton) or the split measures
the degradation, not the kernel. Costs one verdict-probe round.

## Cite it

`via probe:<path>#1` — the contract line `phase_split=read=<pct>%/compute=<pct>%`.
