# FlashInfer 009 FP16 GEMM

The direct-launch Triton baseline passed **25/25** official workloads with a **58.049 µs** geometric mean latency on GB300. It uses a 32 × 128 output tile and accumulates FP16 products in FP32 before writing the supplied FP16 destination. The candidate is provisional GB300 only. [Evaluator traces](gb300-baseline-cli.jsonl)

The evaluator used physical GB300 **0**, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`. The SM lock and measured SM clock were both 1500 MHz. The card reported 3996 MHz for every active memory clock sample, its only supported memory frequency. This driver accepted the deferred memory lock. The live driver was left loaded. [Clock record](gb300-baseline-clocks.json)

The exact destination-passing Python bundle ran locally with SOL-ExecBench 1.0.2 from commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, fbtriton 3.7.1 (runtime version 3.7.0+fb.beta), CUPTI Python 13.5.0, seed 200, 10 warmups and 50 timed calls per workload. The Parquet row has 25 distinct UUIDs with random FP16 inputs. The evaluator applied its default 0.01 absolute and relative tolerances and 0.99 matched ratio. [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The public B200 v1.1 leader was at the 1.000000 SOL cap. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200)

## Tile and schedule screen

One 10-warmup, 50-iteration trial per configuration screened all 25 workload IDs on physical GB300 0. The direct baseline passed all 25 again. Every workload had a passing tuned configuration. The screen recorded 302 passes and 39 build failures. Shared-memory limits and automatic warp specialization lowering caused the build failures. [Screen results](gb300-tile-screen.csv) · [Device and trial metadata](gb300-tile-screen-metadata.json) · [Clock record](gb300-tile-screen-clocks.json)

At M=1 and M=2, four-column GEMV measured 7.324 and 8.812 µs. At M=4–8, the two-warp, 16-column MMA measured 9.670–9.741 µs. The 64-column descriptor tiles led the M=16–172 screen, and 128-column tiles led at M=289–952. The 256-column tile with segmented TMA stores and a 148-CTA persistent grid won at every M above 8,000. At M=12,251 it took 193.158 µs versus 225.976 µs for the best ordinary raster configuration in the screen.
