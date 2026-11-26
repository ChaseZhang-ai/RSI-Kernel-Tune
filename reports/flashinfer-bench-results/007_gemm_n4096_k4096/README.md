# FlashInfer 007 FP16 GEMM

The archive candidate passed all 43 official workloads in the standalone evaluator at **19.30 µs** geometric mean. Two same-device trials reproduced **19.20 µs**, **2.31×** faster than the direct baseline. The exact submission took **3.17 s** for the sum of first synchronized calls across the 43 workloads from an isolated empty Triton cache. The candidate is provisional GB300 only and ready for the coordinator's B200 evaluation. [Official evaluator traces](final-gb300.jsonl) · [Device, source digest and measurement summary](final-gb300-summary.json) · [Cold compile measurements](final-cold-compile.json)

The standalone evaluator used physical GB300 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, with 152 queried SMs and CUDA capability 10.3. Active SM and memory clocks measured 1500/3996 MHz. The driver accepted a deferred memory-clock setting at the sole supported memory frequency. The Triton JIT targeted the queried GB300 architecture on fbtriton 3.7.1. The persistent schedule uses the active device's SM count.

The tuned dispatch uses a vector GEMV at M=1, a four-way intra-CTA FP32 K reduction through M=16, and weight-first TMA through M=128. It uses a 64×128 pipeline for M=129–256 and a 128×256 persistent pipeline for the large workloads. Each route performs one FP16 destination store per element. The archive file requires no global scratch.

## Large workloads

The selected three-stage 128×256×64 pipeline uses 213,236 bytes of shared memory per CTA on GB300. The 8192-row sweep measured 213.47 µs with groups of four M tiles against 216.93 µs for row-major raster order, and 226.72 µs when the worker grid targeted two resident waves. The SM tile-slot fill column is the tile count divided by 152 SMs and the number of waves. [Tile, stage and wave sweep](tile-sweep-gb300.jsonl)

| M | Evaluator latency (µs) | Tiles | Waves | SM tile-slot fill | FP16 PFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 972 | 39.745 | 128 | 1 | 84.2% | 0.821 |
| 2053 | 70.369 | 272 | 2 | 89.5% | 0.979 |
| 2379 | 70.242 | 304 | 2 | 100.0% | 1.136 |
| 8192 | 216.612 | 1024 | 7 | 96.2% | 1.269 |

## Verification history

The direct-launch Tensor Core baseline passed all 43 official workloads on GB300 with a 44.31 µs geometric mean. This is a provisional GB300 result. The coordinator owns B200 evaluation and collection submission. [Official evaluator traces](baseline-gb300.jsonl)

The unmodified Parquet row supplies 43 distinct workload UUIDs and generated FP16 operands. The evaluator applies 0.01 absolute and relative tolerances with a required matched ratio of 0.99. SOL-ExecBench 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd` used 10 warmups, 50 measured calls, cold L2, cloned inputs at shifting addresses, preallocated destinations, and CUPTI first-to-last activity spans. The baseline used PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0. [Evaluator source](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The first job was discarded after a foreign CUDA context was observed. The complete rerun held the lock on physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`. Active samples were 1500/3996 MHz. The driver rejected immediate memory-clock locking and accepted the deferred setting at its sole supported memory frequency of 3996 MHz. The driver stayed loaded. [GPU clock record](baseline-gpu.json)

The public B200 v1.1 leader remained at 1.000000 SOL at 2026-11-26T18:58:24Z. [Timestamped snapshot](public-b200-v1.1.json) · [NVIDIA leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/216/B200)

The initial staged TMA dispatch passed all 43 workloads at 21.04 µs geometric mean on physical GB300 1. The exact DPS bundle used fbtriton 3.7.1, which reports `3.7.0+fb.beta` at runtime. The submission uses an FP32 GEMV for M=1, a weight-first Tensor Core tile for thin matrices, and a persistent Tensor Core pipeline with separate TMA, MMA, and epilogue warps. [Staged evaluator traces](staged-gb300.jsonl) · [GPU clock record](staged-gpu.json) · [Archived CompoundingKernel](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/007_gemm_n4096_k4096/README.md)

The paired tuning screen exercised 13 workload UUIDs with 343 measurements or launch attempts. At M=4, 8 and 16, the intra-CTA four-way K split measured 12.38–12.51 µs, versus 14.08–14.18 µs for the staged dispatch. It reduces FP32 partials in one kernel. At M=128, a weight-first 64×64 tile measured 16.64 µs versus 19.41 µs. At M=256, a 64×128 tile measured 20.48 µs versus 24.61 µs. Several 2-CTA cases compiled, but their generated PTX had no TMA multicast and their timings were slower. The first explicit 4-CTA multicast adaptation hit API mismatches during compilation on the local grader compiler. The sweep logs keep shared-memory resource limits and build failures separate. [Tile sweep](tile-sweep-gb300.jsonl) · [Sweep device](tile-sweep-device.json) · [Multicast API probe](multicast-api-gb300.jsonl)

Two full paired trials put the tuned dispatch at 19.197 and 19.200 µs geometric mean, versus 44.271 and 44.287 µs for the direct baseline and 20.927 and 20.934 µs for staged TMA. Every arm passed correctness and CUPTI timing for all 43 workload IDs in both trials. The tuned candidate was 2.31× faster than the direct baseline in this same-device comparison. The paired run used physical GB300 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, with 152 queried SMs, CUDA capability 10.3, and measured clocks of 1500/3996 MHz. Peak reserved GPU memory across the paired and guard tests was 1.31 GiB. [Paired workload latencies](paired-gb300.csv) · [Device and timing metadata](paired-device.json)

Two exact integer-input checks for each workload used fresh input and output addresses, changed both operands, verified all output elements, and checked guards around the destination. A separate M=35 test verified the unaligned, strided fallback and its output guards. [Fresh-allocation checks](fresh-allocation-checks.json)
