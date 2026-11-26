# FlashInfer 009 FP16 GEMM

The direct-launch Triton baseline passed **25/25** official workloads with a **58.049 µs** geometric mean latency on GB300. It uses a 32 × 128 output tile and accumulates FP16 products in FP32 before writing the supplied FP16 destination. The candidate is provisional GB300 only. [Evaluator traces](gb300-baseline-cli.jsonl)

The evaluator used physical GB300 **0**, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`. The SM lock and measured SM clock were both 1500 MHz. The card reported 3996 MHz for every active memory clock sample, its only supported memory frequency. This driver accepted the deferred memory lock. The live driver was left loaded. [Clock record](gb300-baseline-clocks.json)

The exact destination-passing Python bundle ran locally with SOL-ExecBench 1.0.2 from commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, fbtriton 3.7.1 (runtime version 3.7.0+fb.beta), CUPTI Python 13.5.0, seed 200, 10 warmups and 50 timed calls per workload. The Parquet row has 25 distinct UUIDs with random FP16 inputs. The evaluator applied its default 0.01 absolute and relative tolerances and 0.99 matched ratio. [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The public B200 v1.1 leader was at the 1.000000 SOL cap. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200)
