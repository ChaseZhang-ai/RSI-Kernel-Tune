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

## GB300 hybrid v3

Completed 2026-11-26 17:51:08 UTC on physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`. SM/memory clocks were locked to 1500/3996 MHz. CUDA capability was 10.3 and the device reported 152 SMs. The paired comparison used the same lock and evaluator timing protocol as v1 and v2. Both sides passed 48/48. The exact standalone artifact passed the full CLI 48/48 and the 16-call synthetic suite. [Paired timings](gb300-pair-v3.jsonl) · [CLI results](gb300-evaluator-v3.jsonl)

| Batch size | Workloads | Tuned v2 geomean (µs) | Hybrid v3 geomean (µs) |
|---:|---:|---:|---:|
| 1 | 16 | 5.243 | 5.086 |
| 16 | 16 | 18.056 | 15.338 |
| 64 | 16 | 43.644 | 41.488 |
| All | 48 | 16.047 | 14.792 |

- The CLI geomean was 14.815 µs. The sum of cold first-call wall times, including the native build, Triton JIT and synchronization, was 4.638 s. Native compilation used the capability queried from PyTorch.
- The hybrid uses an asynchronous gather and a thread-block-cluster merge on the selected shapes. The Triton routes apply the same scaling, CSR traversal and destination-passing contract. Native partials live in shared memory, and global scratch on the remaining split route is bounded by the number of populated tiles plus one slot per row.
- Exact submission SHA-256 `b77763759a56fcab9cd1838dd46ced5d31a2c76c811264c3759d9e2281f4914d`. Provisional GB300 verification.

## GB300 hybrid v4

Completed 2026-11-26 17:58:28 UTC on physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with the same locks, SM count, capability and timing options as v3. Both sides of the pair passed 48/48. The exact v4 bundle passed 48/48 with the full CLI. [Paired timings](gb300-pair-v4.jsonl) · [CLI results](gb300-evaluator-v4.jsonl)

| Batch size | Workloads | Hybrid v3 geomean (µs) | Hybrid v4 geomean (µs) |
|---:|---:|---:|---:|
| 1 | 16 | 5.094 | 5.055 |
| 16 | 16 | 15.355 | 14.338 |
| 64 | 16 | 41.524 | 38.612 |
| All | 48 | 14.809 | 14.092 |

- The CLI geomean was 14.114 µs on evaluator 1.0.2. Cold first calls including native compilation, Triton JIT, execution and synchronization took 3.483 s in total.
- The exact file passed 20 synthetic calls with revised row boundaries, duplicate and shuffled indices, changed inputs and destinations, unreferenced NaN pages, non-default scale, and empty rows. The cases exercised all three native variants and both B=1 Triton paths.
- Cluster dispatch now covers all B=64 workloads. Its partials stay in distributed shared memory. The largest official K/V input pair occupies about 120.2 MiB.
- Submission SHA-256 `4670427cd5b87fd4047f36e2d644e11e94555f3eef2d47f16fc1be982f72be64`. Provisional GB300 verification.

The exact v4 bundle separately passed all 48 workloads with PyTorch 2.9.0+cu130 and Triton 3.5.0 on the same GB300. Its pinned-stack geomean was 14.153 µs at 2026-11-26T18:02:11.224648Z. [Pinned-stack v4 CLI results](gb300-evaluator-v4-pinned.jsonl)

## GB300 hybrid v5

The best local candidate passed 48/48 in the full pinned-stack SOL-ExecBench CLI at 2026-11-26T18:04:57.423390Z, with a 14.032 µs geomean. It was paired with v4 on physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, at locked 1500/3996 MHz. The GPU reported 152 SMs and CUDA capability 10.3. [Paired timings](gb300-pair-v5.jsonl) · [Pinned-stack CLI results](gb300-evaluator-v5.jsonl)

| Batch size | Workloads | Hybrid v4 geomean (µs) | Hybrid v5 geomean (µs) |
|---:|---:|---:|---:|
| 1 | 16 | 5.061 | 5.008 |
| 16 | 16 | 14.341 | 14.294 |
| 64 | 16 | 38.657 | 37.962 |
| All | 48 | 14.105 | 13.954 |

- Both sides passed all 48 in the paired run with 10 warmups and 50 timed iterations. The pair used PyTorch 2.10.0+cu130 and Triton 3.6.0. The exact CLI bundle used PyTorch 2.9.0+cu130 and Triton 3.5.0, with CUDA 13.2 PTXAS for GB300 support. Both used SOL-ExecBench 1.0.2. Cold first calls including the native build and Triton JIT took 3.401 s combined.
- Warp-coalesced index loads feed the existing asynchronous K/V staging. All 20 shuffled, unequal-length and empty-row synthetic calls passed.
- The final DPS bundle contains the exact source bytes and was evaluated against the official 48 workload UUIDs, with the trace dataset pinned to revision 1.0. The submitted Python SHA-256 is `b293696f31b3959a16556349f9b51a1e653877a4c526c15ab9bba7bfad0cf70e`.
