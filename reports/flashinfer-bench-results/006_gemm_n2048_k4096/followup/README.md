# Follow-up for 006_gemm_n2048_k4096

The revised destination-passing kernel passed **29/29 official workloads** with timing on GB300 at **23.913 µs** geometric mean. Two same-GPU paired trials against the exact archived comparator measured **23.776** and **23.804 µs**, versus **25.208** and **25.057 µs** for the archive. The aggregate paired speedup was **1.056×**. The exact file took **2.997 s** for cold JIT and synchronized first calls across the 29 workloads. [Official evaluator output](gb300-evaluator.jsonl) · [Paired trials](gb300-paired.json) · [Cold verification](gb300-cold.json)

## B200 evidence and dispatch decision

The standalone B200 v1.1 repeat `106252` timed 26/29 workloads. Its summary gives the total. The released collection child `106075` has a missing CUPTI record at M=34, UUID `f4513ae887ae4d42946dcf6630671c22`. It timed the other 28 workloads for the same archived source. The B200 evidence gives no verified standalone SOL. The public leader was **1.000000 SOL** at `2026-11-27T04:17:52.085232+00:00`. [Per-workload board evidence and archive SHA-256](record.json) · [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/215/B200)

The exact archived source reproduced a complete GB300 CLI pass at 26.881 µs. M=34 also timed in the screen and in both paired repeats. The released B200 data measured M=17 and M=25 at 14.528 and 14.480 µs, against baseline values of 10.880 and 10.816 µs, respectively. The direct narrow tile improved median latency on those shapes in the local screen. The earlier split-K route through M=16 also showed first-to-last latency variation between its two launches. At M=8, the maximum among the 50 timed calls in each paired trial was 40.032 and 41.728 µs for the archive and 9.056 and 9.120 µs for the revision. [Archived local evaluator output](gb300-archived-evaluator.jsonl) · [Screening results](gb300-screen.csv)

The new path for M=3–32 uses one tensor-core launch and reduces eight K partitions within each CTA in FP32. It writes FP16 once and removes the scratch allocation and merge launch from M=5–16. For M=17–32 it replaces the descriptor-based TMA launch with the faster narrow tile. The screen compared tile grouping, output store buffering, tile depth, and ordinary and resident schedules at M=34, near the crossover, and on the slower large B200 workloads. The best large variants varied by shape and stayed close to the archive. At M=34 the archive was fastest in the screen. The revision retains the archived TMA scheduling from M=33 onward. The screen logged 21 build or launch errors for exploratory configurations, chiefly shared-memory capacity failures. Two portability prototypes exposed tensor-memory layout API incompatibilities. The verified file uses a compile-time feature check and a separate branch for each load signature. [Screening results](gb300-screen.csv)

| M | Archive 1 (µs) | Revised 1 (µs) | Archive 2 (µs) | Revised 2 (µs) |
|---:|---:|---:|---:|---:|
| 4 | 10.368 | 8.704 | 10.400 | 8.752 |
| 8 | 12.672 | 8.736 | 11.744 | 8.704 |
| 16 | 10.256 | 8.768 | 10.096 | 8.832 |
| 17 | 13.536 | 12.416 | 13.568 | 12.320 |
| 25 | 13.520 | 12.608 | 13.472 | 12.608 |
| 32 | 13.536 | 12.544 | 13.521 | 12.576 |
| 34 | 13.552 | 13.520 | 13.536 | 13.600 |
| 93 | 13.328 | 13.328 | 13.344 | 13.392 |
| 289 | 16.864 | 16.864 | 16.864 | 16.832 |
| 11938 | 132.560 | 132.416 | 132.688 | 132.544 |
| 12251 | 152.512 | 152.352 | 152.400 | 152.384 |
| 16294 | 182.465 | 182.560 | 182.256 | 182.000 |

## Verification

SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` used 10 warmups and 50 measured calls, cold L2, cloned and pointer-shifted official inputs, preallocated outputs, and first-to-last CUPTI activities. The two paired trial orders were reversed. The cold check used changed input contents, changed input and output addresses on all 29 shapes, guarded row-strided views on eight shapes, and a non-default CUDA stream on five shapes. [NVIDIA evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd) · [Verification records](gb300-cold.json)

The grader-compatible run used fbtriton 3.7.1 on physical GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), and both paired trials used physical GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`). Both reported 152 SMs, CUDA capability 10.3, and sampled clocks of 1500 MHz SM and 3996 MHz memory under the per-device lock. Driver support for the memory lock is deferred. Peak PyTorch allocation was 1.255 GiB. [Run metadata](record.json)

The exact file also passed 29/29 workloads on stock Triton 3.6.0 at 24.386 µs geometric mean. That release requires an explicit register layout for a tensor-memory load. The launcher supplies the layout on Triton versions that expose the older layout helper. This compatibility result concerns the local compiler API. [Stock Triton evaluator output](gb300-stock-triton-evaluator.jsonl)

The candidate SHA-256 is `e03e6c0803fc672e813d93dd41dee040994b343e32bd974239c4bf5841a83b02`. The local evaluator bundle embeds those bytes under `submission.py::run` and uses the same staged Parquet row and official IDs as the previous reproduction instructions. B200 qualification is with the coordinator. [Original reproduction instructions](../README.md#verification-and-reproduction)
