# 022_rmsnorm_h512

## B200 follow-up · Gluon layout compatibility

The archived source and the previous tracked source had SHA-256 `44e86693583ca1a0d2852277fe262ae5fbff8fd905157d608851a4a8d47fdb27`. The released B200 v1.1 collection child 106091 reported runtime errors on batches 1, 7, 18, 32, and 64, all on the single-row Gluon route. Its packed route timed all three remaining batches. The standalone repeat 106231 timed 3/8 and had no verified SOL. The standalone summary gives only the count, so its individual failure reasons and workload IDs are unknown. [Collection child](https://research.nvidia.com/benchmarks/sol-execbench/submission/106091)

Re-running the exact archive under fbtriton 3.7.1 reproduced that five-workload pattern with the same UUIDs. The evaluator reported the errors as `RUNTIME_ERROR` and the logs identify a JIT compilation failure at `x * x`. The Gluon load had a plain block type where the compiler required a distributed type. The two packed specializations compiled and timed, and the compiler check reproduced the Gluon failure for SM 10.0 and 10.3, including the unaligned fallback specialization. This establishes a route-specific source compatibility issue. [Route diagnostics](b200-followup.json)

The revised source inserts identity permutations after the two Gluon loads to preserve their distributed types. The repair is confined to the two load results. Each route still performs the FP32 RMS normalization in one direct launch. On local fbtriton the exact DPS source passed **8/8** with complete CUPTI timing and a **3.347 µs** geometric mean. The fbtriton cold compile for all four specializations took **0.598 s** for SM 10.3 and **0.608 s** for SM 10.0. The latter is a compiler check. [Exact CLI trace](gb300-fb-layout-fix.jsonl)

| Batch | Official UUID | Archive B200 collection (µs) | Archive paired (µs) | Fixed paired (µs) | Standard Triton hybrid paired (µs) | Fixed fbtriton CLI (µs) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `69d9dba5-9d8f-4f6f-901f-43b5ae58b151` | Runtime error | 2.656 | 2.656 | 3.296 | 2.592 |
| 7 | `e0fe7cd7-226c-45dd-af09-45463cc10d39` | Runtime error | 2.560 | 2.528 | 2.848 | 2.528 |
| 18 | `4b5050f5-d6ab-4b20-bb9c-409c073f4d37` | Runtime error | 2.560 | 2.560 | 2.848 | 2.528 |
| 32 | `0315f2ba-9254-4de4-854e-7aafe6460edc` | Runtime error | 2.560 | 2.560 | 2.864 | 2.528 |
| 64 | `eee1d057-c1a3-42c1-bb57-71223bc091ee` | Runtime error | 2.624 | 2.592 | 2.880 | 2.592 |
| 539 | `625f29dc-a8de-48ba-afdb-96727b2a7feb` | 3.104 | 2.976 | 2.944 | 2.976 | 2.960 |
| 11,949 | `6a4a7810-cc7b-4641-9152-91112a15e489` | 7.024 | 6.592 | 6.544 | 6.576 | 6.624 |
| 14,521 | `0581e118-09fa-4f5f-9910-d1b4a7573353` | 7.904 | 7.488 | 7.456 | 7.473 | 7.408 |

The paired GB300 run used stock Triton 3.6.0 so both the exact archive and the fix could run every shape. The archive and fix measured **3.383** and **3.364 µs** geometric means. The standard Triton hybrid used the same packed kernels for the three larger workloads and a two-warp vectorized row reduction for the five small workloads. It measured **3.661 µs** overall. The original direct Triton baseline measured **4.007 µs**. All four passed all eight workloads plus the near-zero, non-uniform weight, changed-pointer and unaligned-input checks. The layout repair retains the faster small-batch path and qualifies on compatibility. [Paired trace](gb300-layout-fix-pair.jsonl)

The archived fbtriton CLI used GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`). The fixed fbtriton CLI used GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`). The stock pair used GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`). Each was a UUID-pinned GB300 at **1500/3996 MHz**, capability 10.3, and **152 queried SMs**, held under its fleet lock. SOL-ExecBench 1.0.2 used 10 warmups and 50 cold-L2 first-to-last CUPTI timings with cloned inputs and preallocated destinations at shifted addresses. Each comparison released its lock before the next job.

Revised SHA-256 `42cf95e21eedaf29141412ba6caf7a7e98d02559da6feb8de7d12576e54baea8`. The public v1.1 B200 leader in the 2026-11-27 04:17:53 UTC snapshot was **0.818143 SOL**. The revision is provisional GB300 evidence and ready for the coordinator's B200 evaluation. [Archive entry](../../../flashinfer-bench/022_rmsnorm_h512/submission.py) · [Public board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/231/B200)

## Previous packed candidate · GB300 verification

The exact archive file passed **8/8 official workloads** with a **3.183 µs** geometric mean in the SOL-ExecBench CLI. The paired comparison on a second GB300 measured **1.205×** speedup over the direct Triton baseline (4.049 to 3.361 µs). [Submission](../../../flashinfer-bench/022_rmsnorm_h512/submission.py) · [Run metadata](record.json) · [CLI trace](gb300-final-cli.jsonl) · [Paired trace](gb300-final-pair.jsonl)

| Batch | Workload UUID | Paired baseline (µs) | Paired candidate (µs) | Exact CLI (µs) | CLI status |
|---:|---|---:|---:|---:|---|
| 1 | `69d9dba5-9d8f-4f6f-901f-43b5ae58b151` | 2.944 | 2.656 | 2.192 | PASSED |
| 7 | `e0fe7cd7-226c-45dd-af09-45463cc10d39` | 2.832 | 2.528 | 2.208 | PASSED |
| 18 | `4b5050f5-d6ab-4b20-bb9c-409c073f4d37` | 2.848 | 2.560 | 2.464 | PASSED |
| 32 | `0315f2ba-9254-4de4-854e-7aafe6460edc` | 2.816 | 2.592 | 2.464 | PASSED |
| 64 | `eee1d057-c1a3-42c1-bb57-71223bc091ee` | 2.880 | 2.560 | 2.576 | PASSED |
| 539 | `625f29dc-a8de-48ba-afdb-96727b2a7feb` | 3.024 | 2.944 | 2.896 | PASSED |
| 11,949 | `6a4a7810-cc7b-4641-9152-91112a15e489` | 10.368 | 6.448 | 6.528 | PASSED |
| 14,521 | `0581e118-09fa-4f5f-9910-d1b4a7573353` | 11.968 | 7.520 | 7.360 | PASSED |

The CLI used physical GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`). The paired trial used physical GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`). Each reported **152 SMs**, CUDA capability **10.3**, and **1500/3996 MHz** SM/memory clocks. The process held the fleet GPU lock, pinned visibility by UUID, and monitored compute activity throughout. SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0 were used for both runs.

All trials used direct launches, 10 warmups, and 50 cold-L2 timed iterations with copied inputs and preallocated destination buffers at shifting addresses. Timing covered the first-to-last kernel activity. The exact pair passed three additional cases per workload on a non-default stream. Those cases used signed non-uniform BF16 weights, zero and near-zero rows, changed input contents, distinct input and output pointers, and contiguous tensors with a two-byte offset to exercise the alignment fallback. There were no numerical, build, or CUPTI timing failures in the exact pair.

At small batches the 128-thread single-row CTA measured 2.528–2.656 µs in the pair. The four-row packed route changed batch 539 from 3.024 to 2.944 µs, a small gain at 1.10 MB of logical input/output traffic. Eight-row CTAs reuse weights in registers across two rows per warp. At 11,949 and 14,521 rows, logical traffic was 24.47 and 29.74 MB. The fastest paired latencies correspond to 3.80 and 3.95 TB/s of logical throughput versus 2.36 and 2.49 TB/s for the baseline. The size scaling and larger response to tiling point to launch and kernel latency at small batches and memory traffic at the largest batches. The largest call uses roughly 28.4 MiB across inputs and output, with zero external scratch allocation.

Triton cold compilation of all four dispatch configurations took **0.565 s** offline for SM 10.3 and **0.568 s** for SM 10.0 with fresh caches. The first GB300 candidate launch including JIT, module load, and synchronization took 0.704 s in the paired process. The SM 10.0 result is a compiler check. The public v1.1 B200 leader was **0.818143 SOL** at **2026-11-26 18:48:33 UTC**. B200 validation is pending coordinator submission. [B200 kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/231/B200)

SHA-256 `44e86693583ca1a0d2852277fe262ae5fbff8fd905157d608851a4a8d47fdb27`. The local DPS bundle `.artifacts/022/solution.json` embeds this exact source as `submission.py::run`. The problem directory contains the staged Parquet row's decoded definition and original workload JSON lines. All tensor inputs use the evaluator's random generator.

```bash
sol-execbench .artifacts/022/problem --solution .artifacts/022/solution.json --output .artifacts/022/results.jsonl
```

The packed memory layout draws on the archived CompoundingKernel experiment. The implementation uses a FP32 warp reduction and applies the weight after normalization. [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/022_rmsnorm_h512/README.md)

## GB300 direct-launch baseline

The one-row, 128-thread Triton CTA passed all eight official workloads in SOL-ExecBench 1.0.2 with PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0. Median workload latencies have a **3.795 µs geometric mean**. The [CLI trace](gb300-baseline.jsonl) records each official UUID, tolerance, status, and latency.

Physical GPU 0, `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, ran under the host lock at 1500 MHz SM and 3996 MHz memory. The evaluator performed 10 warmups and 50 cold-L2 CUPTI iterations per workload, with cloned inputs and destination buffers. Baseline SHA-256 `f592ab368e39a5ea40094688e0994d794804d0c6b77609f9efcba5a5cdb391da`. Triton offline cold compilation for SM 10.3 took 0.367 s. The paired baseline's first launch, including compilation and synchronization, took 0.686 s on GPU 3.

On 2026-11-26 at 18:38:08 UTC, June led the public v1.1 B200 kernel board with 0.818143 SOL (submission 104772) and the collection with 0.940633 SOL (104799). [B200 kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/231/B200) · [B200 collection](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)

| Batch | Workload UUID | Baseline (µs) | Status |
|---:|---|---:|---|
| 1 | `69d9dba5-9d8f-4f6f-901f-43b5ae58b151` | 2.496 | PASSED |
| 7 | `e0fe7cd7-226c-45dd-af09-45463cc10d39` | 2.464 | PASSED |
| 18 | `4b5050f5-d6ab-4b20-bb9c-409c073f4d37` | 2.656 | PASSED |
| 32 | `0315f2ba-9254-4de4-854e-7aafe6460edc` | 2.688 | PASSED |
| 64 | `eee1d057-c1a3-42c1-bb57-71223bc091ee` | 2.688 | PASSED |
| 539 | `625f29dc-a8de-48ba-afdb-96727b2a7feb` | 2.960 | PASSED |
| 11,949 | `6a4a7810-cc7b-4641-9152-91112a15e489` | 10.368 | PASSED |
| 14,521 | `0581e118-09fa-4f5f-9910-d1b4a7573353` | 11.872 | PASSED |


## GB300 layout sweep

Physical GPU 3 (`GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`) reported 152 SMs and CUDA capability 10.3. The locked SM and configured memory clocks were 1500/3996 MHz. The same process measured each candidate against the direct baseline with identical official workloads. [Layout timings and outcomes](gb300-layout-sweep.json)

The sweep measured 299 passing trials. PTXAS rejected 54 variants that combined `.cg` with `.evict_first`. The sweep recorded zero numerical and CUPTI timing failures. The exact archive entry remains the verified baseline during finalist verification.

A 128-thread CTA with eight contiguous BF16 values per thread was fastest among the small-batch layouts in the sweep. At batch 539 the four-row packed CTA improved 3.040 µs to 2.944 µs. The eight-row packed CTA reused the weight fragment across two rows per warp and measured 6.528–7.408 µs for the two largest workloads.


## GB300 finalist comparison

The direct baseline and both finalists passed all eight workloads with CUPTI timing on physical GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 SMs, capability 10.3, 1500/3996 MHz). Each trial used 10 warmups and 50 timed iterations. Both finalists and the baseline also passed two fresh-pointer tests per workload using changed non-uniform weights and rows near zero. [Finalist trial record](gb300-finalists.jsonl)

| Candidate | Median latency geometric mean (µs) | Passed |
|---|---:|---:|
| Direct Triton baseline | 4.008 | 8/8 |
| Gluon blocked | 3.470 | 8/8 |
| Gluon with packed 256-bit loads and stores | 3.359 | 8/8 |
