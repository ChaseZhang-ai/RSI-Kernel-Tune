# 025_rmsnorm_h4096

The direct-launch Triton baseline passed correctness and CUPTI timing for all 14 official workloads on GB300 at 8.059 µs geometric mean. This is a provisional GB300 candidate. [Per-workload evaluator results](gb300-baseline.jsonl)

The baseline used physical GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`). The sampled SM and memory clocks were 1500 and 3996 MHz throughout. The driver denied new unprivileged clock lock requests. The device reported 3996 MHz for both its deferred memory setting and its sole supported memory clock.

SOL-ExecBench 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0 ran the exact DPS bundle from an empty Triton cache on 2026-11-26 at 18:38 UTC. Timing used 10 warmups, 50 measurements, cold L2, cloned inputs, preallocated outputs, and first-to-last CUPTI activity. Cold first-call timing was outside the baseline CLI trace and will be collected in the paired comparison.

The B200 v1.1 public leader was June at 0.904060 SOL at the 18:37 UTC check on 2026-11-26. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/234/B200)

The first paired screen ran on physical GB300 3 (`GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`), with 152 queried SMs, CUDA capability 10.3, and sampled clocks of 1500/3996 MHz. It covered representative B=1, 16, 79, 170, 8,804 and 14,509 workloads. The 16-warp eight-BF16 layout improved small and midrange latency. The B=1 replicated row reduction measured 3.200 µs versus a paired 4.432 µs baseline. The large-batch variants were close to the four-warp baseline at best. PTXAS rejected the reload variants that combined `.volatile` with `.ca`. Their results are marked as build failures. The first baseline call in the cold-cache screen took 0.704 s including JIT and initial launch. [Paired Triton screen](gb300-triton-screen.jsonl)
