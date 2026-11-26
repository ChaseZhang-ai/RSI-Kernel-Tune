# 005_gemm_n256_k7168

The direct-launch baseline passed all 17 official workloads, including CUPTI timing, at **14.447 µs** geometric mean on GB300. The evaluator replayed the staged Parquet workloads with their original UUIDs and default official tolerances. The trace records the per-workload latency and outcome. [Baseline trace](baseline-gb300.jsonl)

The locked job selected physical GB300 index 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3. The unprivileged clock-change command was denied. The fleet-preconfigured clocks read 1500/3996 MHz throughout the job, and 3996 MHz was the only supported memory bin and the deferred setting. The evaluator was SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` (v1.1 methodology), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. Cold compile time was not isolated in this initial CLI job.

The public B200 v1.1 board showed 0.915728 SOL at 2.214 µs for the leader, checked 2026-11-26 18:37 UTC. This baseline is provisional GB300 only. [NVIDIA operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/214/B200)
