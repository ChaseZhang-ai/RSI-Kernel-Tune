# 002_fused_add_rmsnorm_h4096

## Provisional GB300 candidate

The exact archive file passed **14/14** official workloads, including CUPTI timing. Two paired rounds measured **1.343×** and **1.341×** geometric-mean speedup over the direct Triton baseline. The rounds ran in opposite baseline/candidate order. [Archive entry](../../../flashinfer-bench/002_fused_add_rmsnorm_h4096/submission.py) · [Evaluator traces](gb300-evaluator.jsonl) · [Paired timings](gb300-paired.jsonl)

- The exact CLI and paired trials ran on physical GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`) at 1500 MHz SM and 3996 MHz memory with 152 queried SMs and CUDA capability 10.3. NVRTC compiles for the capability reported by PyTorch.
- SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.9.0+cu130, Triton 3.5.0, CUDA bindings 13.1.1, and CUPTI Python 13.5.0. Each timing trial used 10 warmups, 50 timed iterations, cold L2, cloned inputs, preallocated outputs, and CUPTI first-to-last kernel time.
- Cold compilation, module load and first launch took **0.092 s** for the CUDA module and **0.693 s** for the Triton fallback. The selected SM 10.3 kernels used 40 or 64 registers per thread with zero spills. The embedded code also compiled for SM 10.0 offline.
- The non-default-stream checks covered all 14 batch sizes, low-magnitude inputs sensitive to epsilon, input preservation, changed contents at reused addresses, and changed input/output addresses. A separate trial passed with strided and misaligned tensors. The native path allocates no GPU scratch. The largest workload's four tensors occupy 340.1 MiB.
- Candidate SHA-256 `c6ed164c7847f4451b420d5ea346b31ca954dc5f3231b833f1d020126d45d837`. The local DPS solution embeds this exact file with entry point `submission.py::run`. [Machine-readable record](record.json)

Public B200 v1.1 rank 1 was **0.975342 SOL** at 2026-11-26 18:50:08 UTC. The provisional GB300 candidate is ready for coordinator B200 timing. [NVIDIA kernel 211 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/211/B200)

| Batch | Official workload ID | Eval status | Exact CLI (µs) | Paired baseline (µs) | Paired candidate (µs) |
|---:|---|---|---:|---:|---:|
| 1 | `146cc899-3cd6-4713-a85f-0cca97509a48` | PASSED | 2.896 | 4.880 | 2.920 |
| 7 | `9f5150a0-744c-4699-a970-f15b0ea14a38` | PASSED | 2.784 | 4.248 | 2.792 |
| 15 | `8edfb911-8a77-4c5e-9a09-817626ced351` | PASSED | 2.832 | 4.232 | 2.808 |
| 16 | `4eb3e01a-c4e4-4e93-8f4a-010308488552` | PASSED | 2.848 | 4.224 | 2.800 |
| 34 | `04e1c8da-5407-4d96-876c-a27259fe7ecf` | PASSED | 2.912 | 4.256 | 2.928 |
| 63 | `96c11c60-e281-4c6c-91e7-547a65a6bbca` | PASSED | 3.008 | 4.280 | 3.104 |
| 64 | `17defc5f-6708-4dbe-b03b-03cf314e8705` | PASSED | 3.104 | 4.360 | 3.120 |
| 79 | `7f65318b-59ac-478e-84ed-d19052b65a7e` | PASSED | 3.200 | 4.312 | 3.184 |
| 170 | `788c4d95-4e75-49c4-bf79-a00594161842` | PASSED | 3.632 | 4.536 | 3.640 |
| 8,804 | `b190423e-dac4-4725-a76b-420145b8d68d` | PASSED | 32.208 | 38.872 | 31.888 |
| 10,827 | `5cdec50e-3da2-423e-a519-d7e781f6247e` | PASSED | 38.368 | 46.024 | 38.928 |
| 11,832 | `06936f65-e175-42d8-a90d-8b59e38a346d` | PASSED | 42.528 | 49.592 | 42.072 |
| 14,418 | `9595903c-68c2-4bd2-bdbf-0bc1c6dcb8af` | PASSED | 50.912 | 58.800 | 50.856 |
| 14,509 | `3e460c8c-7cad-4071-959f-e689ca024206` | PASSED | 51.488 | 59.168 | 51.504 |

Paired columns average the two 50-iteration medians. The exact CLI column comes from its separate evaluator run.

## Layout sweep

The suite completed 569 workload/variant measurements on physical GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), with 1500/3996 MHz clocks and the same evaluator (PyTorch 2.10.0+cu130, Triton 3.6.0). All timed trials passed. The sweep covered 4–32-byte vectors and one, two, or four rows per CTA. Triton reported zero spills across the measured layouts. NVRTC reported zero spill load/store bytes and the CUDA driver reported zero local bytes for the native variants. [Sweep timings and register counts](gb300-sweep.csv)

For B=1, a 256-thread CTA with a 32-byte vector took 3.376 µs. Two CTAs with duplicated reductions took 3.072 µs and four took 2.976 µs. The measured dispatch uses four per row through B=16 and two per row through B=170, writing disjoint output slices. The large-batch sweep favored the 128-thread, 16-byte vector with one row per CTA across all five shapes, so one dispatch covers them.

| Batch | 128 threads, 1 row (µs) | 256 threads, 1 row (µs) | Fastest 2-row CUDA layout (µs) |
|---:|---:|---:|---:|
| 8,804 | 30.240 | 31.168 | 31.168 |
| 10,827 | 37.424 | 38.448 | 38.480 |
| 11,832 | 40.192 | 41.280 | 41.375 |
| 14,418 | 49.312 | 50.655 | 50.463 |
| 14,509 | 49.472 | 50.192 | 50.400 |

The direct Triton baseline adapts the repository RMSNorm starter. Packed FP32 and vector memory operations build on the RSI FlashInfer 001 path. The CompoundingKernel Explorer archive informed output partitioning, grouped CTAs, and cache tuning. [Triton starting point](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py) · [RSI FlashInfer 001](../../../flashinfer-bench/001_fused_add_rmsnorm_h2048/submission.py) · [CompoundingKernel Explorer](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/002_fused_add_rmsnorm_h4096)

## Reproduction

Materialize the named row from the staged `FlashInfer-Bench.parquet` into `definition.json` and `workload.jsonl`. Decode the definition's `axes`, `inputs` and `outputs` JSON, and write each decoded workload as one line with its original UUID. The workload inputs are all generated by the evaluator. Embed the archive file in `solution.json` as `submission.py`, with the entry point `submission.py::run`, language `triton` and `destination_passing_style` set to `true`. Under the shared GPU lock, pin the UUID and run `sol-execbench <problem_dir> --solution solution.json --output results.jsonl`.

## Initial direct Triton baseline

The exact destination-passing archive entry passed all 14 official workloads with CUPTI timing on physical GB300 GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`). The GPU ran at 1500 MHz SM and 3996 MHz memory. SOL-ExecBench 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`), PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. Cold compilation and the SM count were not instrumented in this initial CLI trial. [CLI traces](gb300-baseline.jsonl)

The baseline adapts the repository Triton RMSNorm kernel to FP32 residual addition, epsilon 1e-5 and BF16 destination output. Baseline SHA-256 `b76a1b140ad9dbdbd164587ca03c179653f96c920d7d378141cee3cfc0ea32ad`. [Triton starting point](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py)
