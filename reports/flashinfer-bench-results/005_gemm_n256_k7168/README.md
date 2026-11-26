# 005_gemm_n256_k7168

The first optimized candidate passed all 17 official workloads, including CUPTI timing, at **8.515 µs** geometric mean on GB300. The kernel uses warp MMA with distributed FP32 cluster reductions for small M, a 64-row tile across M=53–63, and a staged TMA pipeline for larger matrices. Physical GPU 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, reported 152 SMs and CUDA capability 10.3. The SM was locked to 1500 MHz and memory was observed at its sole 3996 MHz bin throughout. The driver uses a deferred memory setting on this host. Source SHA-256 is `e3351e2ae3b71c430dfb43f912858d4afd8c80dc6673d3e6d5acf66057ac64fc`. [Hybrid candidate trace](cluster-tma-gb300.jsonl)

The first hybrid build exceeded static shared-memory limits in two native variants. Two large Triton configurations also exceeded the dynamic shared-memory limit. The passing trial moved the warp partials to dynamic shared memory and reduced the large-path stage count from four to three.

The direct-launch baseline passed all 17 official workloads, including CUPTI timing, at **14.447 µs** geometric mean on GB300. The evaluator replayed the staged Parquet workloads with their original UUIDs and default official tolerances. The trace records the per-workload latency and outcome. [Baseline trace](baseline-gb300.jsonl)

The locked job selected physical GB300 index 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3. The unprivileged clock-change command was denied. The fleet-preconfigured clocks read 1500/3996 MHz throughout the job, and 3996 MHz was the only supported memory bin and the deferred setting. The evaluator was SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` (v1.1 methodology), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. Cold compile time was not isolated in this initial CLI job.

The public B200 v1.1 board showed 0.915728 SOL at 2.214 µs for the leader, checked 2026-11-26 18:37 UTC. This baseline is provisional GB300 only. [NVIDIA operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/214/B200)
