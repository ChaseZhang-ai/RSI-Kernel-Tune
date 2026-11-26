# FlashInfer 021 RMSNorm H128

The direct-launch Triton candidate is provisional GB300 only. The exact DPS bundle passed correctness and CUPTI timing on **14/14** official workloads with a **4.778 µs** latency geomean. It ran on physical GB300 **3**, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, with 1500 MHz SM and 3996 MHz memory clocks. The evaluator was SOL-ExecBench 1.0.2 with PyTorch 2.10.0+cu130 and Triton 3.6.0. A paired cold compilation measurement is pending. [Evaluator traces](gb300-tiered-evaluator.jsonl) · [Run metadata](gb300-tiered-metadata.json)

The staged Parquet row supplied the original 14 workload UUIDs, all with random inputs and default tolerances. SOL-ExecBench used 10 warmups, 50 timed iterations, cold L2, shifted copies of inputs and preallocated outputs. The candidate adapts the repository's tiered-rows RMSNorm to the exact BF16 DPS signature. [Tiered Triton source](../../../kernel_library/kernels/ako4x-rmsnorm-h128-triton_tiered_rows/kernel.py) · [Submission](../../../flashinfer-bench/021_rmsnorm_h128/submission.py)

The public v1.1 leader was **0.799820 SOL** at **2026-11-26T18:37:20Z**. The coordinator owns B200 verification. [B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/230/B200)
