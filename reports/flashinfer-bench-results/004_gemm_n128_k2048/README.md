# 004_gemm_n128_k2048

The direct-launch Triton baseline passed 25/25 official workloads in SOL-ExecBench 1.0.2 (commit `a9fa0804c793d438e70850c33fe34426e66d53dd`). Its latency geomean was 16.394 µs. [Evaluator trace](gb300-baseline.jsonl)

Physical GPU 0 was an NVIDIA GB300 with UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 queried SMs, CUDA capability 10.3 and SM/memory clocks of 1500/3996 MHz. The evaluator used PyTorch 2.10.0+cu130 and Triton 3.6.0, 10 warmups and 50 timed iterations, with cold L2, cloned and pointer-shifted inputs, preallocated outputs and first-to-last CUPTI kernel timing. The baseline used a fresh Triton cache.

Verification is GB300 only. The public B200 v1.1 leader was 0.894803 SOL at 2026-11-26 18:37:10 UTC. [Public kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/213/B200)

## Initial paired trial

On physical GPU 3 (NVIDIA GB300, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, 152 SMs, CUDA capability 10.3, 1500/3996 MHz), the baseline and portable hybrid passed all 25 workloads. The hybrid uses direct GEMV through M=8, two-launch FP32 split-K through M=952 and a staged persistent TMA GEMM at large M. The paired geomeans were 16.543 µs for the baseline and 9.346 µs for the hybrid. Summed cold first-call wall times, including JIT and synchronization, were 1.220 s and 1.074 s respectively. Both sides used the evaluator's 10/50 cold-L2 CUPTI protocol. [Paired workload results](gb300-portable-pair.jsonl)

## Tuning screen

The locked physical GPU 3 ran 436 trials across 22 representative official shapes. Of these, 430 passed with CUPTI timing. Six trials hit the shared-memory limit during kernel preparation. There were no numerical failures or missing CUPTI activities. [Per-workload screen](gb300-screen.jsonl)

Native vector GEMV with one or two activation rows per block won through M=8. Warp-local FP32 K splits, merged in shared memory within one MMA launch, measured 3.520–4.096 µs at M=17, 34, 63, 64 and 93. The two-launch FP32 split-K candidates took 6.656–13.728 µs on those shapes, including the reduction. Clustered MMA was slower than the best single-CTA MMA. The public workload table lists a 0.5 µs SOL bound for all five of these shapes and for the tiny shapes. [Public workload table](https://research.nvidia.com/benchmarks/sol-execbench/api/kernels/213)

At M=289, 492 and 952, the best register-fed MMA tiles measured 5.408, 6.880 and 9.120 µs. At large M, the staged weight-first Tensor Core route with 128 rows, 64 K elements and six TMA buffers had the best geomean across the three screened shapes. The K starting tile is staggered across CTAs. The archive supplied the idea of jointly permuting A and B K fragments for vector loads. The repository H100 GEMM supplied the persistent staging pattern. [CompoundingKernel Explorer](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/004_gemm_n128_k2048) · [Repository H100 GEMM](https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py)

## Tuned candidate

All three implementations were measured on physical GPU 2, NVIDIA GB300, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, at 1500/3996 MHz. The queried CUDA capability was 10.3 and the device reported 152 SMs. The lock was held and foreign compute activity was checked throughout. Each implementation passed 25/25 with the evaluator's 10/50 cold-L2 CUPTI timing and pointer-shifted input clones and output buffers. [Paired results](gb300-final-pair.jsonl)

| M | Direct baseline (µs) | Initial hybrid (µs) | Tuned candidate (µs) |
|---|---:|---:|---:|
| 1–8 | 14.578 | 3.048 | 2.930 |
| 16–172 | 14.333 | 8.264 | 3.793 |
| 289–952 | 14.662 | 13.696 | 6.891 |
| 8,828–16,294 | 25.372 | 28.251 | 17.627 |
| All 25 | 16.550 | 9.282 | 5.537 |

The table shows latency geomeans. The tuned source took 1.891 s in summed first-call wall time with a fresh cache, including compilation, execution and synchronization. Its slowest first call took 1.494 s and included the CUDA 13.1 native build for the capability queried through PyTorch. The CLI and paired trial use the same Python bytes, SHA-256 `c520e227ac7516c2d637834a7c0ad0284dadbd35e1b3bc524dbeb25637a4c59a`.

The candidate performs a single GPU launch per call. Tiny shapes use vector GEMV and mid-M shapes use register-fed Tensor Core MMA with FP32 partials across warps. At M=492 and 952, the native 16×16 route reuses A across two N fragments. The large-M path rotates TMA reads and uses an FP32 Tensor Core accumulator with a separate drain warp group. Its persistent launch cap of 148 is a grid parameter. Both native and staged paths round to FP16 at the output and write the provided destination directly.

## Exact artifact verification

The exact DPS bundle passed **25/25 official workloads** with a latency geomean of **5.511 µs**. The trace preserves the workload IDs, effective tolerances, individual CUPTI latencies and status. It records zero build, numerical or timing failures. The staged Parquet row contains 25 unique IDs and 25 distinct M values. Both inputs are generated by the evaluator in every row. [Official evaluator trace](gb300-evaluator.jsonl)

Physical GPU 0 was an NVIDIA GB300 with UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 queried SMs, CUDA capability 10.3 and SM/memory clocks of 1500/3996 MHz. The exact file also passed 76 bit-exact synthetic calls over 38 shapes. Each shape tested distinct A, B and C pointers, asymmetric physical B rows, changed A and B contents, guarded output stores and cancellation between the K partitions. M=38,033 exercised persistent CTA and accumulator-buffer reuse. [Synthetic checks](gb300-edge.json)

The candidate adds no global partial buffer. At the largest official M, the combined A, B and C allocation is about 68 MiB. Native CUDA launches are built for the compute capability queried at runtime.

For local reproduction, materialize the named row from `.artifacts/sol-data/FlashInfer-Bench.parquet`. Decode the `axes`, `inputs` and `outputs` JSON fields in the definition, and write each original workload JSON object to its own line in `workload.jsonl`. Embed the exact Python file as `submission.py` in a solution bundle with entry point `submission.py::run`, language `triton` and DPS enabled. The verified bundle SHA-256 is `f2062d22bb1fc70ba374b2aecb24dadb3aec45e2400228b41ef05ecac1b698f1`.

```bash
sol-execbench <problem_dir> --solution solution.json --output results.jsonl
```

The public B200 v1.1 leader was **0.894803 SOL**, rank 1, at 2026-11-26 18:55:21 UTC. This candidate is provisional GB300 only and ready for the coordinator's B200 validation. [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/213/B200)

## B200 follow-up on 2026-11-27

The released collection child `106073` reports missing CUPTI activity at **M=16,294** (workload `c49c96bd-bee4-479b-8000-3e6bf297aef1`). The standalone repeat `106254` timed 24/25, and its available summary provides the aggregate count. Both results used archived SHA-256 `c520e227ac7516c2d637834a7c0ad0284dadbd35e1b3bc524dbeb25637a4c59a`, which matched the previous tracked candidate byte-for-byte. The public v1.1 leader was **0.894803 SOL** at 2026-11-27 04:25:15 UTC. [B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/213/B200)

The exact archive again passed 25/25 through the evaluator on physical GB300 GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`), 152 queried SMs, CUDA 10.3. Five separate paired timing trials at M=16,294 on physical GB300 GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`), 152 queried SMs, CUDA 10.3 all produced CUPTI timings between 19.808 and 19.936 µs. The launch uses 128 CTAs for 128 M tiles and fits within one wave. The final tile has 38 valid rows, with TMA zero-fill on the input tail and a masked output store. The missing activity did not reproduce on GB300. M=16,294 uses the archived six-stage TMA kernel and launch configuration in the revised source. [Archived CLI results](gb300-followup-archived.jsonl) · [Repeated timings and tuning screen](gb300-followup-screen.jsonl)

The screen covered all 25 official workload IDs, with 116 timed trials. The 32-byte vector access trials did not improve tiny M consistently, so the revision keeps the archived vector GEMV. At M=16–64 the revised native MMA shares each output pair's FP32 warp-partial reduction across four lanes. At M=172–492 it uses a 16×16 output tile with 16 warps, and at M=952 it uses a 16×32 tile. The 64×128×128 TMA tile at M=8,828 launches 138 CTAs into one wave, up from 69 CTAs. The other large-M variants offered little improvement, including at M=16,294. Every route writes the provided FP16 output in one launch after FP32 accumulation. [Per-workload screen](gb300-followup-screen.jsonl)

The exact pair ran on physical GB300 GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`), 152 queried SMs, CUDA 10.3. Both sources passed 25/25. The table shows latency geomeans. [Paired results](gb300-followup-pair.jsonl)

| M | Archived (µs) | Revised (µs) |
|---|---:|---:|
| 1–8 | 2.932 | 2.922 |
| 16–64 | 3.542 | 3.483 |
| 93–128 | 4.095 | 4.080 |
| 172–952 | 6.423 | 5.754 |
| 8,828 | 15.520 | 13.504 |
| 11,006–16,294 | 17.996 | 18.020 |
| All 25 | 5.531 | 5.374 |

The revised candidate reduced paired latency geomean by **2.8%**. The exact revised DPS bundle passed **25/25 official workloads** with a **5.368 µs geomean** on physical GB300 GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`), 152 queried SMs, CUDA 10.3. The same file passed 76 bit-exact checks of B layout, cancellation, M tails, CTA reuse, changed pointers and guarded output writes. Its summed cold first-call time was 2.225 s, including compilation, execution and synchronization. The slowest cold first call was 1.216 s. [Revised evaluator trace](gb300-followup-evaluator.jsonl) · [Synthetic checks](gb300-followup-edge.json)

Both CLI runs and the paired trials used SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, Triton 3.6.0, and 1500/3996 MHz SM/memory clocks. Each trial used 10 warmups and 50 timed iterations, with cold L2, cloned pointer-shifted inputs, preallocated outputs and the evaluator's first-to-last CUPTI timing. The revised source SHA-256 is `6e48da7cde42d44e01ad71c6d3ec82075a64f42bdbdfed7f6b08ab5f65ad62d1`. The exact local bundle SHA-256 is `adfe2dfd9d7a60d7cd2e1838128c56554dd47ffd06f1bced27ded84bbc3047a0`. Verification of the revision is provisional GB300 only.
