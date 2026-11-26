# FlashInfer 026 RMSNorm H=7168

The direct-launch Triton baseline passed correctness and CUPTI timing for all eight official workloads on GB300. The workspace started from `origin/main` at `dc3ccce2`.

## Triton v1

The exact direct-launch DPS archive entry passed all eight workloads.
The paired geometric mean fell from 8.896 to 7.067 µs, a 20.6% reduction.
The paired results average three 50-sample trial medians. The evaluator column uses the median of the CLI's 50 timed samples.

| Batch | Official UUID | Baseline paired µs | Candidate paired µs | Exact CLI µs | Status |
|---:|---|---:|---:|---:|---|
| 1 | `f1133b8c-5ab3-4cf5-b753-84a32d9e3661` | 4.843 | 3.109 | 3.120 | PASSED |
| 7 | `98773f58-2d55-497d-a79d-df0bca20e369` | 4.117 | 2.976 | 2.976 | PASSED |
| 18 | `00b8ce81-3357-41ea-b7ec-731439bc51ad` | 4.139 | 3.019 | 3.040 | PASSED |
| 32 | `08ca44b8-2c3a-407d-87a7-85454ef4d748` | 4.160 | 3.072 | 3.104 | PASSED |
| 64 | `d1ebd937-923e-44b2-8e4e-3e9194aa8d09` | 4.235 | 3.296 | 3.280 | PASSED |
| 539 | `e8073f61-7ae6-44d0-9b13-29199e6c5595` | 6.507 | 6.165 | 6.208 | PASSED |
| 11,949 | `75e6d0e3-a957-47fe-84ff-d3e492f505ae` | 59.061 | 54.837 | 54.720 | PASSED |
| 14,521 | `5b601f70-1495-4c70-9870-e05dc771f1c1` | 70.229 | 65.088 | 64.576 | PASSED |

Cold compilation and first launches across the eight workloads totaled 0.623 s in an empty candidate cache.
The exact submission SHA-256 is `af64ac654c0a531ce5c063159c40061e61f335edfdc5e6747c17f648bd21b27d`.

[Exact evaluator output](v1-evaluator.jsonl) · [Paired trials](v1-paired.jsonl) · [Device and artifact metadata](v1-device.json)

## Reproduce the DPS evaluation

Generate the bundle from the collection entry and the staged Parquet. Run the evaluator under the fleet lock with the selected GB300 UUID set in `CUDA_VISIBLE_DEVICES` and the clocks set to 1500/3996 MHz.

```sh
python reports/flashinfer-bench-results/026_rmsnorm_h7168/prepare_bundle.py .artifacts/026-repro
sol-execbench .artifacts/026-repro --solution .artifacts/026-repro/solution.json --output .artifacts/026-repro/results.jsonl
```

## Benchmark conditions

SOL-ExecBench 1.0.2 at `a9fa0804` used the open CUPTI evaluator with PyTorch 2.10.0+cu130, Triton 3.6.0, CUDA 13.1 PTXAS and CUPTI Python 13.5.0. Runs held the fleet lock, pinned the worker to the physical GPU UUID, and checked for other compute jobs. The 1500 MHz SM lock and the sole supported 3996 MHz memory clock were verified before and after each run. The GB300s queried 152 SMs and capability 10.3. Native experiments compiled for the queried capability using NVRTC. The 148-SM proxy was used for the persistent CTA grid experiments.

All official inputs are generated from the staged Parquet workloads. The evaluator uses each workload's original UUID and its default tolerance of 0.01 absolute, 0.01 relative, and 99% required matches. The trials use 10 warmups, 50 timed iterations, cold L2, cloned inputs, preallocated destinations and CUPTI first-to-last kernel activity.

## Initial tuning

- The small-batch sweep tested CTA replication, disjoint output sections, and 2 to 16 warps. At batches 1 and 7 the fastest trial was 2.880 and 2.912 µs with packed CUDA, versus 3.040 and 2.976 µs with Triton. The comparison used the archived CompoundingKernel implementation's CTA replication and vector layout as ideas. [QianQu archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/026_rmsnorm_h7168)
- At batch 539 the retained 8192-padded Triton path at 8 warps ran in 6.208 µs. The best native trial ran in 5.984 µs. Both two-CTA Triton candidates failed during compilation in `TritonGPUPlanCTAPass`.
- For batches 11,949 and 14,521, the 7 × 1024 retained Triton path with 4 warps and `.cg` loads and `.cs` stores ran in 54.800 and 64.864 µs. With default cache modifiers, retaining activations took 54.944 and 65.408 µs. Reloading them took 59.872 and 70.256 µs. The fastest coalesced native reload path ran in 54.240 and 64.416 µs with 224 threads. Persistent CTA caps of 4, 8, and 12 times the 148-SM proxy were slower on both large cases.
- The raw results record each trial status and latency. [Triton sweep](sweep.csv) · [Native sweep](native-sweep.csv)

The public B200 leader on v1.1 was June at 0.937995 SOL when checked at 2026-11-26 18:48:08 UTC. [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/235/B200)

## Bounded follow-up sweep

The final sweep compared Triton v1 with 32-warp CTAs, a different address wrap for replicated reductions, reverse row order, and CTAs handling two or four rows. All 118 benchmark rows passed. Triton v1 was fastest on seven workloads. The best alternate at batch 11,949 gained 0.032 µs in one trial and was 0.560 µs slower at batch 14,521. The archive entry retains v1. [Follow-up sweep](round2.csv) · [Follow-up device](round2-device.json)
