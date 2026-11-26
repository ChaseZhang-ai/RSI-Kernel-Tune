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
- Submission SHA-256 `847014f6fa2c643ff9af32801813196a783b0dd9974f91dd4636894553e87cf8`. The exact Python file is embedded in the local DPS solution bundle. The exact bundle also passed 48/48 under the full official CLI. [CLI results](gb300-evaluator-v1.jsonl)
- The initial baseline CLI run exceeded its 120 s evaluation timeout. An earlier split-KV build failed Triton's type checks when a scalar lookup variable was reused for a vector scratch offset. The paired run above uses the corrected source.

The first private B200 upload attempt returned HTTP 429 before a submission ID was created.

## GB300 tuned Triton v2

Completed 2026-11-26 17:45:33 UTC on physical GPU 0 at the same queried clocks and device capability described above. Both sides of the pair passed 48/48. The exact v2 Python file also passed 48/48 through the SOL-ExecBench CLI. [Paired timings](gb300-pair-v2.jsonl) · [CLI results](gb300-evaluator-v2.jsonl) · [Tuning screen](tuning-screen.md)

| Batch size | Workloads | Split-KV v1 geomean (µs) | Tuned v2 geomean (µs) |
|---:|---:|---:|---:|
| 1 | 16 | 6.797 | 4.947 |
| 16 | 16 | 19.647 | 18.012 |
| 64 | 16 | 50.498 | 43.792 |
| All | 48 | 18.893 | 15.744 |

- The v2 CLI geomean was 15.779 µs. The 48 paired cold-cache first calls took 3.233 s in total including Triton JIT, execution and synchronization.
- The v2 submission SHA-256 is `0df5d83bb655f4475beb8c0fbc77b0c9a81bda774f61ac3fee438b0214d3529f`.
- The v2 synthetic suite passed 16 calls with shuffled and repeated page IDs, reversed uneven row lengths, empty rows, non-default scales, changed pointers and unreferenced NaN pages. Triton 3.5.0 required the CUDA 13.2 assembler on local SM 10.3 because the older bundled PTXAS lacks SM 10.3 support. The full evaluator above used Triton 3.6.0.
- The first two private B200 upload requests were rate limited and returned no submission ID. Local results are provisional GB300 only.
