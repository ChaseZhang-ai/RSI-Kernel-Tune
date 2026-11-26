# FlashInfer 007 FP16 GEMM

The direct-launch Tensor Core baseline passed all 43 official workloads on GB300 with a 44.31 µs geometric mean. This is a provisional GB300 result. The coordinator owns B200 evaluation and collection submission. [Official evaluator traces](baseline-gb300.jsonl)

The unmodified Parquet row supplies 43 distinct workload UUIDs and generated FP16 operands. The evaluator applies 0.01 absolute and relative tolerances with a required matched ratio of 0.99. SOL-ExecBench 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd` used 10 warmups, 50 measured calls, cold L2, cloned inputs at shifting addresses, preallocated destinations, and CUPTI first-to-last activity spans. The baseline used PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0. [Evaluator source](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The first job was discarded after a foreign CUDA context was observed. The complete rerun held the lock on physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`. Active samples were 1500/3996 MHz. The driver rejected immediate memory-clock locking and accepted the deferred setting at its sole supported memory frequency of 3996 MHz. The driver stayed loaded. [GPU clock record](baseline-gpu.json)

The public B200 v1.1 leader is at 1.000000 SOL. [Timestamped snapshot](public-b200-v1.1.json) · [NVIDIA leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/216/B200)
