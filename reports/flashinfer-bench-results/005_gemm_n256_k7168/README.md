# 005_gemm_n256_k7168

The exact archive candidate passed **17/17** official FlashInfer workloads at **7.346 µs** geometric mean on GB300. The two paired runs measured 7.399 and 7.406 µs against the direct-launch baseline at 13.903 and 13.904 µs, for speedups of **1.879×** and **1.877×**. The candidate is provisional GB300 only and ready for the coordinator's B200 trial. [Official results](final-gb300.jsonl) · [Paired results](paired-gb300.jsonl) · [Machine-readable summary](gb300-summary.json)

The official job and both paired comparisons used physical GB300 index 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`. The synthetic suite used index 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`. Each queried 152 SMs and CUDA capability 10.3. The GPU lock covered each job and the process selected its GPU by UUID. The driver accepted the 1500 MHz SM lock. Memory stayed at the preconfigured deferred setting of 3996 MHz, the sole supported bin. The driver rejected an immediate memory-clock lock.

SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` used the v1.1 timing methodology with PyTorch 2.10.0+cu130, Triton 3.6.0, CUDA 13.2 PTXAS and CUPTI Python 13.5.0. Each comparison used 10 warmups and 50 timed iterations, cold L2, cloned and pointer-shifted inputs, preallocated destinations and CUPTI first-to-last timing. All final workloads recorded CUPTI activity. First synchronized calls across all 17 shapes took **0.944 s** from an empty Triton cache, including NVRTC module compilation and launch. The paired baseline took 1.756 s in the same process. Peak PyTorch allocation was 0.664 GiB.

All 53 synthetic checks passed on a non-default stream. They covered dispatch thresholds, misaligned and strided inputs and destinations, input preservation, changed input values and pointers, and near cancellation across K partitions. The maximum absolute error on the cancellation cases was 0 against an FP64 reference rounded to FP16. [Synthetic results](edge-checks-gb300.json)

## Dispatch and tuning

- M=1 uses a direct GEMV. The two-CTA K split measured 3.809–3.872 µs across tested widths and the selected direct route measured 3.744–3.777 µs. M=4 uses single-launch cluster split-K with an FP32 merge in distributed shared memory. It measured 3.872 µs in the sweep. The fastest direct GEMV at M=4 measured 4.512 µs.
- M=53–63 uses two 32-row MMA tiles with a seven-way K split. The one-tile 64-row design was tested. The 32-row variant ran in 5.409 and 5.536 µs at M=56 and 63 in the refinement. The best 64-row variant took 5.760 and 6.112 µs. FP32 partials are reduced within the cluster and each destination element is rounded once.
- M=901 uses a swapped-operand TMA pipeline with a 32×64 output tile and 128-deep K stages. Split-K reached 18.784 µs in the initial screen and was 28.225 µs or higher in the refinement. The selected single launch measured 23.105 and 23.184 µs across the screens.
- The largest shapes use full-width 128×256 tiles with four 64-deep TMA stages, a register epilogue and a persistent grid capped at the queried SM count. This measured 67.057 and 69.730 µs in the official run. Three stages and the full-tile TMA epilogue were slower. The wide tile lets each output tile read its A rows once.

The initial and refinement screens ran on physical GB300 indices 1 and 0 respectively, at 1500/3996 MHz. They logged 173 timed passes and 49 build failures across configurations and baseline observations. Numerical failures and CUPTI timing failures were zero. Compiler failures included Triton warp-specialization lowering and shared-memory limits. [Tuning results](tuning-gb300.csv)

The grader-version GB300 check with PyTorch 2.9.0+cu130 and Triton 3.5.0 stopped at LLVM's `Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` for SM 10.3. The same artifact compiled offline for the B200 SM 10.0 target in Triton 3.5.0 and NVRTC under the grader-version environment. B200 runtime verification is pending. [Offline build record](b200-offline-build.txt)

## Reproduction

The candidate SHA-256 is `74d68660b4b46e123f61c63ddb601e413b9284501ed9e1fde8236c172fdfa625`. The staged Parquet row supplied the definition and all 17 workload UUIDs. It uses generated tensors and the evaluator's default official tolerances. The materialization script embeds the exact Python file in a local destination-passing bundle. Run the evaluator under the fleet GPU lock with `CUDA_VISIBLE_DEVICES` set to the selected physical UUID. [Materialization script](prepare.py)

```bash
python reports/flashinfer-bench-results/005_gemm_n256_k7168/prepare.py \
    .artifacts/sol-data/FlashInfer-Bench.parquet \
    flashinfer-bench/005_gemm_n256_k7168/submission.py \
    .artifacts/005/problem .artifacts/005/solution.json
sol-execbench .artifacts/005/problem \
    --solution .artifacts/005/solution.json --output .artifacts/005/results.jsonl
```

The public v1.1 B200 leader was **0.915728 SOL** at 2.214 µs at the 2026-11-26 18:50:31 UTC check. [NVIDIA B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/214/B200)

## Earlier GB300 results

The first optimized candidate passed all 17 official workloads, including CUPTI timing, at **8.515 µs** geometric mean on GB300. The kernel uses warp MMA with distributed FP32 cluster reductions for small M, a 64-row tile across M=53–63, and a staged TMA pipeline for larger matrices. Physical GPU 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, reported 152 SMs and CUDA capability 10.3. The SM was locked to 1500 MHz and memory was observed at its sole 3996 MHz bin throughout. The driver uses a deferred memory setting on this host. Source SHA-256 is `e3351e2ae3b71c430dfb43f912858d4afd8c80dc6673d3e6d5acf66057ac64fc`. [Hybrid candidate trace](cluster-tma-gb300.jsonl)

The first hybrid build exceeded static shared-memory limits in two native variants. Two large Triton configurations also exceeded the dynamic shared-memory limit. The passing trial moved the warp partials to dynamic shared memory and reduced the large-path stage count from four to three.

The direct-launch baseline passed all 17 official workloads, including CUPTI timing, at **14.447 µs** geometric mean on GB300. The evaluator replayed the staged Parquet workloads with their original UUIDs and default official tolerances. The trace records the per-workload latency and outcome. [Baseline trace](baseline-gb300.jsonl)

The locked job selected physical GB300 index 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3. The unprivileged clock-change command was denied. The fleet-preconfigured clocks read 1500/3996 MHz throughout the job, and 3996 MHz was the only supported memory bin and the deferred setting. The evaluator was SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` (v1.1 methodology), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0.
