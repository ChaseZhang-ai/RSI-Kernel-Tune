# FlashInfer 026 RMSNorm H=7168

The direct-launch Triton baseline passed correctness and CUPTI timing for all eight official workloads on GB300. The workspace started from `origin/main` at `dc3ccce2`.

## Benchmark conditions

SOL-ExecBench 1.0.2 at `a9fa0804` used the open CUPTI evaluator with PyTorch 2.10.0+cu130, Triton 3.6.0, CUDA 13.1 PTXAS and CUPTI Python 13.5.0. Runs held the fleet lock, pinned the worker to the physical GPU UUID, and checked for other compute jobs. The 1500 MHz SM lock and the sole supported 3996 MHz memory clock were verified before and after each run. The GB300s queried 152 SMs and capability 10.3. Native experiments compiled for the queried capability using NVRTC. The 148-SM proxy was used for the persistent CTA grid experiments.

All official inputs are generated from the staged Parquet workloads. The evaluator uses each workload's original UUID and its default tolerance of 0.01 absolute, 0.01 relative, and 99% required matches. The trials use 10 warmups, 50 timed iterations, cold L2, cloned inputs, preallocated destinations and CUPTI first-to-last kernel activity.

## Initial tuning

- The small-batch sweep tested CTA replication, disjoint output sections, and 2 to 16 warps. At batches 1 and 7 the fastest trial was 2.880 and 2.912 µs with packed CUDA, versus 3.040 and 2.976 µs with Triton. The comparison used the archived CompoundingKernel implementation's CTA replication and vector layout as ideas. [QianQu archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/026_rmsnorm_h7168)
- At batch 539 the retained 8192-padded Triton path at 8 warps ran in 6.208 µs. The best native trial ran in 5.984 µs. Both two-CTA Triton candidates failed during compilation in `TritonGPUPlanCTAPass`.
- For batches 11,949 and 14,521, the 7 × 1024 retained Triton path with 4 warps and `.cg` loads and `.cs` stores ran in 54.800 and 64.864 µs. With default cache modifiers, retaining activations took 54.944 and 65.408 µs. Reloading them took 59.872 and 70.256 µs. The fastest coalesced native reload path ran in 54.240 and 64.416 µs with 224 threads. Persistent CTA caps of 4, 8, and 12 times the 148-SM proxy were slower on both large cases.
- The raw results record each trial status and latency. [Triton sweep](sweep.csv) · [Native sweep](native-sweep.csv)

The public B200 leader on v1.1 was June at 0.937995 SOL when checked at 2026-11-26 18:37 UTC. [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/235/B200)
