# 002_fused_add_rmsnorm_h4096

## Direct Triton baseline

The exact destination-passing archive entry passed all 14 official workloads with CUPTI timing on physical GB300 GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`). The GPU ran at 1500 MHz SM and 3996 MHz memory. SOL-ExecBench 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. Cold compilation and the SM count were not instrumented in this initial CLI trial. [CLI traces](gb300-baseline.jsonl)

The baseline adapts the repository Triton RMSNorm kernel to FP32 residual addition, epsilon 1e-5 and BF16 destination output. Baseline SHA-256 `b76a1b140ad9dbdbd164587ca03c179653f96c920d7d378141cee3cfc0ea32ad`. [Triton starting point](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py)

The public B200 v1.1 leader was rank 1 at **0.975342 SOL**, checked 2026-11-26 18:37:25 UTC. [NVIDIA kernel 211 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/211/B200)
