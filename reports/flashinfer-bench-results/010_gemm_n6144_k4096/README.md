# 010_gemm_n6144_k4096

The direct-launch Triton baseline passed all 43 official FlashInfer-Bench workloads on GB300. Each workload has a positive CUPTI latency. [Per-workload results](baseline-gb300.jsonl)

The baseline used GB300 physical index 1 with UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, CUDA capability 10.3 and 152 queried SMs. The SM lock was 1500 MHz and active clock samples read 1500/3996 MHz. The driver rejected the immediate memory lock even though its command returned zero. The evaluator was SOL-ExecBench 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, with PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. The evaluator used 10 warmups and 50 measured calls per workload, cold L2, cloned inputs, preallocated destinations, and first-to-last CUPTI activity timing. A foreign context interrupted the first GPU attempt. The entire baseline was rerun on GPU 1.

The public v1.1 board reports a B200 leader SOL of 1.000000. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/219/B200)
