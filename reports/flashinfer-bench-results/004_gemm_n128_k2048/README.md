# 004_gemm_n128_k2048

The direct-launch Triton baseline passed 25/25 official workloads in SOL-ExecBench 1.0.2 (commit `a9fa0804c793d438e70850c33fe34426e66d53dd`). Its latency geomean was 16.394 µs. [Evaluator trace](gb300-baseline.jsonl)

Physical GPU 0 was an NVIDIA GB300 with UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 queried SMs, CUDA capability 10.3 and SM/memory clocks of 1500/3996 MHz. The evaluator used PyTorch 2.10.0+cu130 and Triton 3.6.0, 10 warmups and 50 timed iterations, with cold L2, cloned and pointer-shifted inputs, preallocated outputs and first-to-last CUPTI kernel timing. The baseline used a fresh Triton cache. Cold first-call times will be captured in the paired trial.

Verification is GB300 only. The public B200 v1.1 leader was 0.894803 SOL at 2026-11-26 18:37:10 UTC. [Public kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/213/B200)

## Initial paired trial

On physical GPU 3 (NVIDIA GB300, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, 152 SMs, CUDA capability 10.3, 1500/3996 MHz), the baseline and portable hybrid passed all 25 workloads. The hybrid uses direct GEMV through M=8, two-launch FP32 split-K through M=952 and a staged persistent TMA GEMM at large M. The paired geomeans were 16.543 µs for the baseline and 9.346 µs for the hybrid. Summed cold first-call wall times, including JIT and synchronization, were 1.220 s and 1.074 s respectively. Both sides used the evaluator's 10/50 cold-L2 CUPTI protocol. [Paired workload results](gb300-portable-pair.jsonl)
