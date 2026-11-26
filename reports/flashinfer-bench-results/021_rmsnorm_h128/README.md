# FlashInfer 021 RMSNorm H128

The provisional GB300 candidate passed **14/14** official workloads for correctness and CUPTI timing. The exact DPS bundle's latency geomean was **4.504 µs**. In the paired trial, latency fell from **4.651 µs** with the tiered baseline to **4.502 µs** with the candidate, a **3.2% reduction**. [Submission](../../../flashinfer-bench/021_rmsnorm_h128/submission.py) · [Run record](gb300-record.json) · [Evaluator traces](gb300-results.jsonl) · [Paired traces](gb300-pair.jsonl)

Both the standalone evaluator and the paired trial ran on physical GB300 **1**, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with **152 queried SMs** and CUDA capability **10.3**. Both jobs used 1500 MHz SM and 3996 MHz memory clocks under the shared GPU lock. SOL-ExecBench **1.0.2**, PyTorch **2.10.0+cu130** and fbtriton **3.7.1** (Triton `3.7.0+fb.beta`) ran the exact candidate. The paired cold cache recorded **0.311 s** of JIT compilation across 4 candidate specializations. [Evaluator revision](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The staged Parquet row supplied all 14 original UUIDs. Every workload uses random inputs and the evaluator defaults of 0.01 absolute tolerance, 0.01 relative tolerance and a 0.99 required matched ratio. Each trial used 10 warmups and 50 cold-L2 CUPTI first-to-last kernel spans with cloned inputs at shifted addresses and preallocated destination tensors. Supplemental checks covered every workload and five extra row tails, including a zero row, input scales of 1 and 1e-5, signed and zero weights, changed contents at fixed addresses, separately allocated tensors at legal 2/6/10-byte offsets, and guard elements around the output. [Supplemental results](gb300-edge.json)

The public v1.1 B200 leader was **0.799820 SOL** at **2026-11-26T18:57:36Z**. B200 verification is pending with the coordinator. [B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/230/B200)

## Latency by workload

All rows passed. Effective bandwidth uses the logical BF16 bytes read and written, including one weight read, divided by the standalone evaluator median. The kernel allocates no workspace. The largest inputs and output total about 254 MiB.

| Batch size | Tiered paired µs | Candidate paired µs | Evaluator µs | Effective TB/s |
|---:|---:|---:|---:|---:|
| 4 | 2.848 | 2.560 | 2.528 | — |
| 24 | 2.752 | 2.608 | 2.592 | — |
| 32 | 2.736 | 2.496 | 2.528 | — |
| 136 | 2.768 | 2.528 | 2.512 | — |
| 192 | 2.688 | 2.592 | 2.624 | — |
| 256 | 2.720 | 2.592 | 2.592 | — |
| 316 | 2.784 | 2.625 | 2.560 | — |
| 1,088 | 2.896 | 2.752 | 2.752 | — |
| 2,048 | 2.880 | 2.816 | 2.848 | — |
| 2,528 | 2.961 | 2.880 | 2.912 | — |
| 49,532 | 6.752 | 6.944 | 6.912 | 3.67 |
| 65,016 | 8.384 | 9.312 | 9.345 | 3.56 |
| 396,256 | 32.081 | 31.936 | 32.433 | 6.26 |
| 520,128 | 40.848 | 41.024 | 40.817 | 6.52 |

## Dispatch and tuning

The candidate uses direct Triton Gluon launches. It selects four- or eight-row CTAs with eight lanes per row for smaller batches, 32-row CTAs for the medium streaming cases, and 16-row CTAs for the largest tensors. The streaming layouts assign 16 lanes per row. All paths preserve FP32 RMS and weight multiplication, followed by one BF16 cast.

The paired Triton 3.6.0 screen compared warp, half-warp and eight-lane row layouts, CTA sizes, columns per lane and streaming-store policies. The launch-attribute comparison used matched cold-L2 trials with and without programmatic dependent launch. Default memory policies and direct launches were selected after the sweep. The midrange B=65,016 case rose from 8.384 to 9.312 µs in the final pair. The bounded round prioritized exact-file verification after the compiler-generated vector layout sweep. [Tuning traces](gb300-tuning-screen.jsonl)

The initial fbtriton probe hit a distributed-type compilation error after global loads. Identity permutations in the final source recover the layouts on that compiler. [Build probe](gb300-fbtriton-build-probe.json)

The starting point was the repository's ako4x tiered-rows implementation and SOL example. The CompoundingKernel Explorer archive informed the explicit lane layout and load/store experiments. [Tiered Triton source](../../../kernel_library/kernels/ako4x-rmsnorm-h128-triton_tiered_rows/kernel.py) · [SOL example](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py) · [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/021_rmsnorm_h128/README.md) · [Initial baseline and evaluator setup](https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/tree/f928c5cb11eabe2e2183e928af2db35084a2b2e1/reports/flashinfer-bench-results/021_rmsnorm_h128)

## Local reproduction

Materialize the official Parquet row as `problem/definition.json` and `problem/workload.jsonl`, preserving the workload UUIDs and tolerance fields. Embed the exact `submission.py` in `solution.json` with entry point `submission.py::run`, Triton language and destination passing enabled. With the evaluator environment active and an idle GB300 claimed under its shared lock, run the CLI with that GPU's UUID as `CUDA_VISIBLE_DEVICES`.

```bash
sol-execbench problem --solution solution.json --output results.jsonl
```
