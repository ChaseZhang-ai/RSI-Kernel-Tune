# FlashInfer 009 FP16 GEMM

The direct-launch Triton baseline passed **25/25** official workloads with a **58.049 µs** geometric mean latency on GB300. It uses a 32 × 128 output tile and accumulates FP16 products in FP32 before writing the supplied FP16 destination. The candidate is provisional GB300 only. [Evaluator traces](gb300-baseline-cli.jsonl)

The evaluator used physical GB300 **0**, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`. The SM lock and measured SM clock were both 1500 MHz. The card reported 3996 MHz for every active memory clock sample, its only supported memory frequency. This driver accepted the deferred memory lock. The live driver was left loaded. [Clock record](gb300-baseline-clocks.json)

The exact destination-passing Python bundle ran locally with SOL-ExecBench 1.0.2 from commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, fbtriton 3.7.1 (runtime version 3.7.0+fb.beta), CUPTI Python 13.5.0, seed 200, 10 warmups and 50 timed calls per workload. The Parquet row has 25 distinct UUIDs with random FP16 inputs. The evaluator applied its default 0.01 absolute and relative tolerances and 0.99 matched ratio. [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The public B200 v1.1 leader was at the 1.000000 SOL cap. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200)

## Tile and schedule screen

One 10-warmup, 50-iteration trial per configuration screened all 25 workload IDs on physical GB300 0. The direct baseline passed all 25 again. Every workload had a passing tuned configuration. The screen recorded 302 passes and 39 build failures. Shared-memory limits and automatic warp specialization lowering caused the build failures. [Screen results](gb300-tile-screen.csv) · [Device and trial metadata](gb300-tile-screen-metadata.json) · [Clock record](gb300-tile-screen-clocks.json)

At M=1 and M=2, four-column GEMV measured 7.324 and 8.812 µs. At M=4–8, the two-warp, 16-column MMA measured 9.670–9.741 µs. The 64-column descriptor tiles led the M=16–172 screen, and 128-column tiles led at M=289–952. The 256-column tile with segmented TMA stores and a 148-CTA persistent grid won at every M above 8,000. At M=12,251 it took 193.158 µs versus 225.976 µs for the best ordinary raster configuration in the screen.

## First tuned candidate

Both variants passed 25/25 in three paired trials on physical GB300 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with CUDA capability 10.3 and 152 queried SMs. The direct baseline measured 57.881 µs geomean and the tuned candidate measured 24.440 µs, a 2.37× speedup. The timed calls used cloned inputs, rotated device addresses, preallocated outputs, cold L2, and CUPTI first-to-last kernel spans. Each trial used 10 warmups and 50 timed calls. The candidate's cold first calls, including JIT and synchronization, took 2.519 s summed over the official list. Two full-output guards per workload passed exactly with different input contents and device addresses and distinct patterns across all five column bands. [Paired results](gb300-pair-v1.jsonl) · [Metadata](gb300-pair-v1-metadata.json) · [Clock record](gb300-pair-v1-clocks.json)

The refinement measured 12 representative shapes twice on physical GB300 0. At M=16 a 32-column tile with two warps measured 9.260 µs against 10.890 µs for the original 64-column tile. At M=12,251 the 152-worker persistent tile with four grouped M tiles and two 128-column TMA stores measured 191.751 µs. The same tile with one grouped M tile took 193.682 µs. The four-group pointer-store candidate emitted 128-bit vector stores and took longer than the segmented TMA epilogue. The final dispatch uses the queried device SM count. [Refinement](gb300-refine.csv) · [Metadata](gb300-refine-metadata.json) · [Clock record](gb300-refine-clocks.json)
