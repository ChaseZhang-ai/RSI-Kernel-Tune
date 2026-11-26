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

The initial standalone evaluator processes hit the 120-second suite timeout. The exact candidate is being re-evaluated with the standalone DPS bundle and a longer suite timeout. The private B200 submission attempt at 2026-11-26 17:38:04 UTC returned HTTP 429 before acceptance. Account capacity reported 0 active of 5 immediately beforehand. This leaves the current candidate provisional on GB300.

[Paired workload timings and errors](gb300-v1-paired.jsonl) · [AKO4X seed](../../../kernel_library/kernels/ako4x-gqa-paged-decode-h32-kv8-d128-ps1-triton_per_workload_split_max/kernel.py) · [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/013_gqa_paged_decode_h32_kv8_d128_ps1)
