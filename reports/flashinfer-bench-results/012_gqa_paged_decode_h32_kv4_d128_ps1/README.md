# Paged GQA decode benchmark record

## GB300 direct baseline and split-KV v1

Paired at 2026-11-26 17:35 UTC. Both implementations passed all 48 official workloads against the reference and produced timings. The per-workload records include UUIDs, axes and median latency. The v1 submission is provisional and GB300 only. [Paired timings](gb300-pair-v1.jsonl)

| Batch size | Workloads | Direct baseline geomean (µs) | Split-KV v1 geomean (µs) |
|---:|---:|---:|---:|
| 1 | 16 | 6.663 | 7.657 |
| 16 | 16 | 46.108 | 24.806 |
| 64 | 16 | 107.673 | 50.444 |
| All | 48 | 32.101 | 21.240 |

- Physical GPU 0, NVIDIA GB300, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, SM/memory 1500/3996 MHz, 152 queried SMs, CUDA capability 10.3. The host-shared lock was held during correctness and timing and compute jobs were checked throughout.
- SOL-ExecBench 1.0.2 from NVIDIA evaluator commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, Triton 3.6.0. Paired trials used seed 200, 10 warmups and 50 timed iterations, cold L2, cloned and pointer-shifted inputs, preallocated outputs, and the evaluator's CUPTI first-to-last kernel timing.
- A fresh Triton cache was used for the pair. Summed first-call wall time including JIT, launches and synchronization was 1.245 s for the direct baseline and 3.010 s for v1. This bounds compilation time in this run.
- Submission SHA-256 `847014f6fa2c643ff9af32801813196a783b0dd9974f91dd4636894553e87cf8`. The exact Python file is embedded in the local DPS solution bundle. Full CLI evaluation is in progress.
- The initial baseline CLI run exceeded its 120 s evaluation timeout. An earlier split-KV build failed Triton's type checks when a scalar lookup variable was reused for a vector scratch offset. The paired run above uses the corrected source.
