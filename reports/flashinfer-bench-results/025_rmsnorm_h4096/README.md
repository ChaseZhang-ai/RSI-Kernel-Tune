# 025_rmsnorm_h4096

## GB300 candidate for the coordinator

The exact standalone DPS bundle passed **14/14 official workloads**, including CUPTI timing, at **7.190 µs** geometric mean in SOL-ExecBench. Official B200 verification is pending. [Submission source](../../../flashinfer-bench/025_rmsnorm_h4096/submission.py) · [Evaluator results with official UUIDs and tolerances](gb300-final-evaluator.jsonl)

- Standalone check on physical GB300 2, `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, with 152 queried SMs and CUDA capability 10.3. The driver denied unprivileged clock changes and sampled clocks stayed at 1500/3996 MHz. The memory clock was the sole supported setting.
- SOL-ExecBench 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`), PyTorch 2.9.0+cu130, Triton 3.5.0, CUDA 13.2 NVRTC and PTXAS, and CUPTI Python 13.5.0. CUDA compilation targets the active device's queried capability.
- 48 supplementary calls covered tiny inputs, mixed signed and zero weights, changed input contents and addresses, strided and unaligned buffers, and the 1/2 and 256/257 dispatch boundaries on a non-default CUDA stream. Input tensors were preserved. [Supplementary correctness record](gb300-final-synthetic.jsonl)
- The selected path for an aligned B=1 uses eight CTAs with disjoint output slices and a full FP32 reduction in each CTA. Aligned batches through 256 use one 256-thread CTA per row. Larger and strided or unaligned inputs use four-warp Triton. Each path uses epsilon 1e-5 and a final BF16 cast. The replication idea came from the archived CompoundingKernel implementation, and the packed FP32 operations build on the repository's fused-add implementation. [CompoundingKernel source](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/092577af37b8431d8dc16f6e90ec3d2ba1451aaa/kernels/FlashInfer/025_rmsnorm_h4096)
- Candidate SHA-256 `2686d9d2a4cf13114fd6f47d63c36952398dbeed2ecde12d55e12f742fdfbd00`. Staged Parquet SHA-256 `89d1feacf02b6f8c07ec8d67a9dbb2c531652251095a7f762f2bc296f6a1fe74`.
- On the public B200 v1.1 board, June led at 0.904060 SOL at 2026-11-26 18:49:29 UTC. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/234/B200)

| Batch size | Paired baseline (µs) | Paired candidate (µs) | Standalone candidate (µs) | Standalone outcome |
|---:|---:|---:|---:|---|
| 1 | 4.384 | 2.752 | 2.784 | Pass |
| 7 | 3.696 | 3.040 | 3.025 | Pass |
| 15 | 3.760 | 3.008 | 3.040 | Pass |
| 16 | 3.744 | 2.976 | 3.056 | Pass |
| 34 | 3.744 | 3.040 | 3.040 | Pass |
| 63 | 3.712 | 3.056 | 3.072 | Pass |
| 64 | 3.744 | 3.056 | 3.168 | Pass |
| 79 | 3.760 | 3.104 | 3.104 | Pass |
| 170 | 3.856 | 3.392 | 3.392 | Pass |
| 8,804 | 25.840 | 26.112 | 26.161 | Pass |
| 10,827 | 30.592 | 30.944 | 30.689 | Pass |
| 11,832 | 32.800 | 33.088 | 32.865 | Pass |
| 14,418 | 38.816 | 39.008 | 39.041 | Pass |
| 14,509 | 38.960 | 39.264 | 39.329 | Pass |

### Reproduction

From the repository root, materialize the staged definition, the original workload records, and the exact submission source. Run the evaluator on the locked, idle physical GPU selected by UUID. The 14 workloads use generated inputs.

```sh
python reports/flashinfer-bench-results/025_rmsnorm_h4096/materialize.py \
  .artifacts/sol-data/FlashInfer-Bench.parquet \
  flashinfer-bench/025_rmsnorm_h4096/submission.py .artifacts/rmsnorm-025/recheck
sol-execbench .artifacts/rmsnorm-025/recheck/problem \
  --solution .artifacts/rmsnorm-025/recheck/solution.json \
  --output .artifacts/rmsnorm-025/recheck/results.jsonl
```

## Baseline and tuning

The direct-launch Triton baseline passed correctness and CUPTI timing for all 14 official workloads on GB300 at 8.059 µs geometric mean. This is a provisional GB300 candidate. [Per-workload evaluator results](gb300-baseline.jsonl)

The baseline used physical GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`). The sampled SM and memory clocks were 1500 and 3996 MHz throughout. The driver denied new unprivileged clock lock requests. The device reported 3996 MHz for both its deferred memory setting and its sole supported memory clock.

SOL-ExecBench 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0 ran the exact DPS bundle from an empty Triton cache on 2026-11-26 at 18:38 UTC. Timing used 10 warmups, 50 measurements, cold L2, cloned inputs, preallocated outputs, and first-to-last CUPTI activity. Cold first-call timing was outside the baseline CLI trace and will be collected in the paired comparison.

The B200 v1.1 public leader was June at 0.904060 SOL at the 18:37 UTC check on 2026-11-26. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/234/B200)

The first paired screen ran on physical GB300 3 (`GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`), with 152 queried SMs, CUDA capability 10.3, and sampled clocks of 1500/3996 MHz. It covered representative B=1, 16, 79, 170, 8,804 and 14,509 workloads. The 16-warp eight-BF16 layout improved small and midrange latency. The B=1 replicated row reduction measured 3.200 µs versus a paired 4.432 µs baseline. The large-batch variants were close to the four-warp baseline at best. PTXAS rejected the reload variants that combined `.volatile` with `.ca`. Their results are marked as build failures. The first baseline call in the cold-cache screen took 0.704 s including JIT and initial launch. [Paired Triton screen](gb300-triton-screen.jsonl)

The packed CUDA screen used physical GB300 3 at the same clocks and tool versions. Eight-way output partitioning at B=1 measured 3.248 µs against 4.384 µs for the baseline, and the 256-thread CTA measured 3.072, 3.264 and 3.536 µs at B=16, 79 and 170. Retaining the full row and adding bulk L2 prefetch improved the native large-batch kernel to 25.760 and 39.712 µs at B=8,804 and 14,509. The paired Triton baseline ended at 25.744 and 39.264 µs. Two-row native tiles with weight reuse measured 32.512–34.288 µs and 49.008–52.608 µs, and reloading activations improved one two-row tile while leaving it behind the one-row Triton kernel. [Paired CUDA screen](gb300-native-screen.jsonl)

## Full paired comparison

Both sides passed 14/14 official workloads with correctness and all 50 CUPTI records per workload. The packed CUDA/Triton candidate measured **7.152 µs** geometric mean against **8.248 µs** for the paired direct Triton baseline, a **13.3% latency reduction**. B=1 improved from 4.384 to 2.752 µs. Physical GB300 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`) reported 152 SMs and CUDA capability 10.3. The benchmark sampled 1500/3996 MHz throughout and used the same evaluator and timing protocol as the baseline. [Per-workload paired results](gb300-final-pair.jsonl)

The first CUDA and Triton calls from the cold candidate build took 0.056 and 0.216 s respectively, including first launch and synchronization. The paired baseline's first call took 0.721 s.
