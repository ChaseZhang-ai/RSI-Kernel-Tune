# Probe recipe library (WP12) — pre-planted measurement instruments

Three operator-agnostic measurement recipes for the verdict-probe channel: the
questions a human expert would reach for before committing a diagnosis, pre-written
so the teacher spends one probe round instead of one authoring round + one probe
round. Each recipe = a script (this directory, self-contained — no forgekernel
import, torch only on the GPU path) + a README (symptoms it fits, how to read it,
what it costs).

| Recipe | Answers | Fits when you see |
|---|---|---|
| [`occupancy_scaling`](occupancy_scaling.md) | latency-bound vs throughput-bound vs wave quantization | `waves_per_sm` low or fractional; both SOL sides low; the SOL header alone cannot split the three classes |
| [`phase_ablation`](phase_ablation.md) | how much of the kernel is the load face vs the math face | both faces plausible; a stall ledger that disperses instead of pointing; "is the supply or the mma the kernel?" |
| [`effective_bw`](effective_bw.md) | the bandwidth this node actually reaches (vs NCU's theoretical SOL denominator) | `dram_pct` high and about to be called "the wall"; a memory-bound verdict that "85%" is supposed to carry |

## How to run one

Copy the script into the student workspace's `notes/` (adapt its CONFIG / hook to
the tree's kernel where the README says so), then send the ordinary probe marker
from a diagnosis turn:

```
PROBE_REQUEST: notes/occupancy_scaling.py
```

The driver executes it as `cmd:python3 notes/<name>.py` in the student tree (cwd =
repo root) inside ONE GPU job and feeds the log tail back. The scripts end with
`PROBE PASS max_diff=n/a (<contract>)` lines — the probe channel's "the instrument
ran and produced its reading" (a non-numerical contract per the probe brief),
citable in the diagnosis as `via probe:<path>#<n>`. They **produce readings
only**: no gate reads them, and a READ line names the curve shape / split — the
teacher judges, prices, and decides.

## What they cost

One verdict-probe round (one GPU job) each, from the diagnosis step's probe budget
(`[loop] max_verdict_probe_rounds`, default 4 per meta entry) — the budget the
brief already caps at a handful of experiments per diagnosis, which is exactly
enough for one or two of these plus a falsification probe. The built-in default
hooks (torch streaming self-checks) exist so a template runs unmodified; **a
curve/split measured on a default hook says nothing about the operator** — adapt
the hook first or say in `evidence` that the reading is instrument-only.

## Discipline (unchanged)

Same-job interleaving, ratio/median readings, pre-registered READ lines — the
verdict-probe rules apply verbatim; these scripts only pre-write the harness.
A recipe that needs the student's kernel edited to measure is NOT this channel
(add a lever to the pricing round instead). Teacher-side only: nothing here ever
reaches the student's read set.
