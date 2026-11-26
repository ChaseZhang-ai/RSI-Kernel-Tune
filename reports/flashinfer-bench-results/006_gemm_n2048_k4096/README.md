# 006_gemm_n2048_k4096

The direct-launch Triton seed passed **29/29** official FlashInfer-Bench workloads on GB300 with **44.016 µs** geometric mean. The DPS adapter calls the repository seed's `_tri_mm` with its non-split configurations and writes the supplied FP16 destination. The repository prototype also contains a runtime selector and vendor-library routes, which are outside this measurement. [Per-workload evaluator traces](gb300-seed.jsonl)

The run used SOL-ExecBench 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0. All 29 staged Parquet workloads use generated inputs and unique IDs and M axes. The evaluator resolved the default tolerance and ran 10 warmups and 50 timed iterations per workload with cold L2, cloned inputs, preallocated outputs, and first-to-last CUPTI timing.

The baseline ran on physical GPU 3 (`GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, CUDA capability 10.3, 152 SMs). The GPU was held under the shared lock and sampled at 1500 MHz SM and 3996 MHz memory through the run. The driver accepted the SM lock and a deferred memory lock. An empty Triton cache was used for the CLI run. Standalone cold compile time is pending.

The public B200 v1.1 board listed leader submission `102864` at **1.000000 SOL** at the 2026-11-26 18:38 UTC check. The archive entry is provisional GB300 only. [NVIDIA operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/215/B200)
