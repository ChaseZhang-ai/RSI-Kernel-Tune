# Recipe: effective-BW triad (the node's reachable bandwidth)

Script: [`effective_bw.py`](effective_bw.py) — `PROBE_REQUEST: notes/effective_bw.py` (one GPU job).

## Symptom it fits

- `dram_pct` high (≥85%) and a "bandwidth-bound, nothing to do here" verdict is
  about to be written — NCU's memory SOL is a percentage of the **theoretical**
  peak, and a node's reachable bandwidth is routinely 10–20% below it, so "85%"
  can sit far from the real wall;
- two candidates both claim "the wall" (e.g. dram vs tensor) and the digest
  cannot separate them;
- a new node/board is being calibrated (the reading is reusable across
  operators of the same run).

## What it measures

A pure streaming kernel in triad shape (read two fp32 arrays, write one — 12 B
of DRAM traffic per element, no reuse) at a working-set size matching the
operator's footprint, interleaved rounds in one job, median. Report:

```
achieved_GBps = 12 B x elements / median_time
```

## How to read it (pre-registered in the script's READ line)

- **Against the operator** (the reading that matters): take the kernel's
  achieved DRAM GB/s (digest `dram_bytes_read+write` / kernel time, or NCU's
  dram bytes over `gpu__time_duration`) and compute
  `effective_sol = kernel_GBps / triad_GBps x 100` — the SOL denominator becomes
  **measured-reachable** instead of theoretical:
  - `>= ~90%` → the kernel really is at the node's wall: bandwidth-bound, stop
    pulling that thread;
  - far below → the "85%" was the theoretical-headroom illusion; the deficit is
    locality / sectoring / replay effects, not bandwidth.
- **Calibration reading** (`--theoretical-gbps` given): the script reports what
  fraction of the theoretical peak this node delivers — every `dram_pct` seen
  this run can be re-read against that denominator.

## Adaptation

Set `WORKING_SET_MIB` (or `--mib`) to the operator's footprint — a triad far
smaller than the kernel's working set reads L2, not DRAM, and overstates the
node. Usually no kernel adaptation at all: the triad measures the node, never
the operator. Costs one verdict-probe round.

## Cite it

`via probe:<path>#1` — the contract line `node_bw=<GBps>GBps`.
