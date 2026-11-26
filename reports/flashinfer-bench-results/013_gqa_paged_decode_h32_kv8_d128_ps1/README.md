# 013_gqa_paged_decode_h32_kv8_d128_ps1

## GB300 paired trial v1

**48/48 passed** for both the direct-launch DPS adaptation of RSI-Kernel-Tune’s AKO4X seed and the provisional candidate. All 48 official workload IDs, including repeated axes, used the trace metadata from the NVIDIA FlashInfer-Bench Parquet row. The run checked BF16 output and FP32 base-2 LSE independently against a batched FP32 reference with the workload tolerances.

| Batch | Workloads | Seed geometric mean | Candidate geometric mean |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 6.348 µs | 6.015 µs |
| 16 | 16 | 31.506 µs | 19.293 µs |
| 64 | 16 | 69.392 µs | 68.904 µs |
| All | 48 | 24.032 µs | 19.997 µs |

The pair ran on physical GB300 index 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, compute capability 10.3, and 152 queried SMs with clocks locked at 1500/3996 MHz. Both sides used SOL-ExecBench 1.0.2’s cold-L2 CUPTI timing routine with 10 warmups, 50 timed calls, independent input copies at shifted addresses, preallocated DPS outputs, and elapsed time from the first through the last kernel. Torch was 2.10.0+cu130 and Triton was 3.6.0. All timed workloads produced their CUPTI records.

The fresh Triton cache trial took 2.596 seconds summed across candidate first calls, including synchronization and GPU work. The longest candidate first call took 0.508 seconds. The source SHA-256 was `ab649b1d201978b8977b191162ceacc5a71fcc6c85321dc7962abafec83d1b7f`.

The initial standalone evaluator processes hit the 120-second suite timeout. The exact candidate then passed the standalone SOL-ExecBench DPS evaluation with a 420-second suite timeout. It passed all ten correctness rounds and CUPTI timing on all 48 workloads on physical GB300 index 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, compute capability 10.3, with 152 queried SMs and 1500/3996 MHz clocks. [Official evaluator workload results](gb300-v1-official.jsonl) The private B200 submission attempt at 2026-11-26 17:38:04 UTC returned HTTP 429 before acceptance. Account capacity reported 0 active of 5 immediately beforehand. This leaves the current candidate provisional on GB300.

[Paired workload timings and errors](gb300-v1-paired.jsonl) · [AKO4X seed](../../../kernel_library/kernels/ako4x-gqa-paged-decode-h32-kv8-d128-ps1-triton_per_workload_split_max/kernel.py) · [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/013_gqa_paged_decode_h32_kv8_d128_ps1)

## Scheduling trials

The screening compared two and four query heads per CTA and swept each batch size independently. It timed the full dispatch including split merges. The sweep varied tile sizes, split counts, warp counts, stages, FP16 and BF16 probability products, KV cache policy, group order, and merge grouping. It also tested direct TMA K/V gathers and a batched warp-tile schedule. The batched warp-tile kernel exceeded resources in six configurations. The other recorded screening trials passed the numerical and timing checks.

- GPU 1, `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, ran the initial split sweep and TMA tests.
- GPU 0, `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, ran the batched warp-tile sweep and synthetic v1 checks.
- GPU 3, `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, ran the follow-up split and merge sweep.

All used 1500/3996 MHz, capability 10.3, 152 queried SMs, and SOL-ExecBench 1.0.2 cold-L2 CUPTI timing with 10 warmups and 50 timed calls. [Workload IDs, configurations and measurements](gb300-tuning.csv)

The synthetic v1 checks used randomized page permutations, empty rows, and highly imbalanced batches. The largest synthetic sequence had 51,000 tokens. Each case was repeated with different pointers, query and cache contents, page order, row order and scale. The checks passed independently for BF16 output and FP32 base-2 LSE. [Synthetic correctness results](gb300-v1-edge-cases.jsonl)

## GB300 paired and standalone trial v2

The selected direct launch uses four query heads per CTA for B=16, BF16 K/V tensor-core products, and two-head merge CTAs for split paths. B=1 chooses its own crossover and uses a SIMD kernel for the smallest rows.

The AKO4X seed, v1, and the new candidate each passed 48/48 paired checks and timings. The exact bundled DPS source passed 48/48 in the standalone evaluator, including all ten numerical rounds and CUPTI timing. All 18 synthetic invocations passed separately for output and base-2 LSE.

| Batch | AKO4X seed | v1 | New candidate |
| ---: | ---: | ---: | ---: |
| 1 | 6.771 µs | 6.082 µs | 5.724 µs |
| 16 | 32.241 µs | 19.268 µs | 14.097 µs |
| 64 | 69.190 µs | 68.688 µs | 65.319 µs |
| All | 24.719 µs | 20.042 µs | 17.403 µs |

Geometric means are from cold-L2 CUPTI first-to-last kernel timing with 10 warmups, 50 timed calls, cloned inputs at shifted addresses, and preallocated outputs. All runs in this trial used physical GB300 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, capability 10.3, 152 queried SMs, 1500/3996 MHz, SOL-ExecBench 1.0.2, Torch 2.10.0+cu130, and Triton 3.6.0.

The fresh Triton cache trial took 3.145 seconds summed across synchronized candidate first calls, with a maximum of 0.501 seconds per call. The maximum paired absolute errors were 0.0078125 for BF16 output and 1.90735e-06 for base-2 LSE. The submission SHA-256 is `f58b79e9768a6294cac27755b311e5d71fc546465447a43a9a8f2be8b8654091`.

[Paired timings](gb300-v2-paired.jsonl) · [Official evaluator timings](gb300-v2-official.jsonl) · [Synthetic checks](gb300-v2-edge-cases.jsonl)

## GB300 paired and standalone trial v3

The long-row dispatch assigns each CTA a bounded span of the flattened page list and derives its sequence from live CSR metadata. The merge reads the resulting per-row slot range and handles rows of any valid length, including empty rows. The B=1 and short B=16 routes retain their tuned direct launches.

The AKO4X seed, v1, and the new candidate each passed 48/48 paired checks and timings. The exact bundled DPS source passed 48/48 in the standalone evaluator, including all ten numerical rounds and CUPTI timing. All 18 synthetic invocations passed separately for output and base-2 LSE.

| Batch | AKO4X seed | v1 | New candidate |
| ---: | ---: | ---: | ---: |
| 1 | 6.697 µs | 6.014 µs | 5.658 µs |
| 16 | 36.834 µs | 19.245 µs | 13.970 µs |
| 64 | 68.993 µs | 68.529 µs | 61.947 µs |
| All | 25.722 µs | 19.943 µs | 16.981 µs |

Geometric means are from cold-L2 CUPTI first-to-last kernel timing with 10 warmups, 50 timed calls, cloned inputs at shifted addresses, and preallocated outputs. All runs in this trial used physical GB300 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, capability 10.3, 152 queried SMs, 1500/3996 MHz, SOL-ExecBench 1.0.2, Torch 2.10.0+cu130, and Triton 3.6.0.

The fresh Triton cache trial took 4.066 seconds summed across synchronized candidate first calls, with a maximum of 0.793 seconds per call. The maximum paired absolute errors were 0.0078125 for BF16 output and 1.90735e-06 for base-2 LSE. The submission SHA-256 is `a6affb9669025564fc861c0b7372fdcf3164165986e73cb523a93dd7b46cbfe3`.

[Paired timings](gb300-v3-paired.jsonl) · [Official evaluator timings](gb300-v3-official.jsonl) · [Synthetic checks](gb300-v3-edge-cases.jsonl)

The 256-token scheduling slots use at most 4.63 MiB of scratch across the official workloads. The largest official K and V tensors total 318 MiB. The distributed Gluon prototypes hit compiler lowering errors and were dropped. The materialized Parquet SHA-256 was `89d1feacf02b6f8c07ec8d67a9dbb2c531652251095a7f762f2bc296f6a1fe74`. The local SOL-ExecBench source checkout resolved to `a9fa0804c793d438e70850c33fe34426e66d53dd`.

## GB300 paired and standalone trial v4

The kernel uses 64-token slots for short B=16 rows and two pipeline stages for long rows. Programmatic dependent launch overlaps the split merge while preserving the device dependency before reading partials. This trial also remeasured v3 against the same inputs on the same GPU.

The AKO4X seed, v1, and the new candidate each passed 48/48 paired checks and timings. The exact bundled DPS source passed 48/48 in the standalone evaluator, including all ten numerical rounds and CUPTI timing. All 22 synthetic invocations passed separately for output and base-2 LSE.

| Batch | AKO4X seed | v1 | New candidate |
| ---: | ---: | ---: | ---: |
| 1 | 6.744 µs | 6.057 µs | 5.932 µs |
| 16 | 34.657 µs | 19.293 µs | 12.216 µs |
| 64 | 68.876 µs | 68.502 µs | 59.431 µs |
| All | 25.250 µs | 20.004 µs | 16.270 µs |

Geometric means are from cold-L2 CUPTI first-to-last kernel timing with 10 warmups, 50 timed calls, cloned inputs at shifted addresses, and preallocated outputs. All runs in this trial used physical GB300 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, capability 10.3, 152 queried SMs, 1500/3996 MHz, SOL-ExecBench 1.0.2, Torch 2.10.0+cu130, and Triton 3.6.0.

The fresh Triton cache trial took 3.657 seconds summed across synchronized candidate first calls, with a maximum of 0.765 seconds per call. The maximum paired absolute errors were 0.0078125 for BF16 output and 1.90735e-06 for base-2 LSE. The submission SHA-256 is `a87829109655b600d34e2d4741f2f495817caaa4affdefe561dff6f088956bf9`.

[Paired timings](gb300-v4-paired.jsonl) · [Official evaluator timings](gb300-v4-official.jsonl) · [Synthetic checks](gb300-v4-edge-cases.jsonl)

The same-GPU v3 rerun measured 16.971 µs overall, 13.822 µs at B=16 and 61.944 µs at B=64. The v4 candidate improved the overall geometric mean by 4.1% versus v3 and by 35.6% versus the seed.

The exact v4 DPS bundle also passed 48/48 on physical GB300 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, using Torch 2.9.0+cu130, Triton 3.5.0, and the CUDA 13.1 `ptxas` for the queried capability 10.3. GPU 3 has 152 queried SMs and was locked at 1500/3996 MHz. An earlier alternate-stack v3 trial hit a build error because the bundled assembler lacked `sm_103a` support. The build error is recorded separately from timing and numerical outcomes. [Alternate stack pass](gb300-v4-torch29-triton35.jsonl) · [Legacy assembler error](gb300-v3-legacy-assembler-build.jsonl)

Materialization preserved the official axes, reference and per-workload tolerances. The Parquet row’s empty optional `hf_id` field was omitted to satisfy the local evaluator’s definition schema.
