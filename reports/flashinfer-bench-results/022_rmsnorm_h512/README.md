# 022_rmsnorm_h512

## Selected candidate · Provisional GB300 only

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
