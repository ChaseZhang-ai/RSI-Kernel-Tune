# Fused add RMSNorm h7168

## Direct Triton baseline · 2026-11-26 18:37 UTC

The exact DPS submission passed all eight official workloads with numerical checks and CUPTI timing on GB300. It fuses the FP32 input add, padded 8192-element reduction with epsilon 1e-6, and weighted BF16 store in one direct launch. The CLI used 10 warmups and 50 cold-L2 iterations with shifted input and output buffers. [CLI records](gb300-baseline-cli.jsonl) · [Triton starting point](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py)

- Physical GB300 GPU 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, SM/memory clocks 1500/3996 MHz, CUDA capability 10.3, 152 queried SMs. The GPU was pinned by UUID and monitored under the host lock.
- SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, Triton 3.6.0, CUPTI Python 13.5.0. A separate paired run with a fresh Triton specialization measured 0.3225 s for first JIT and launch.
- The official Parquet row supplied all eight workload UUIDs. Each workload uses evaluator-generated tensors and default tolerance of 0.01 absolute, 0.01 relative, and 0.99 required matched ratio.
- Provisional GB300 verification. The current B200 v1.1 leader was June at 0.863498 SOL, rank 1, as checked at 18:37 UTC on 2026-11-26. [Public board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/212/B200)
- Candidate SHA-256 `e780c2f5450edbd0256de061f5dc026c79493930d2c7996449aa285849dfe30d`. CLI geometric mean 9.707 µs.

| Batch | Workload UUID | CLI status | Latency (µs) |
|---:|---|---|---:|
| 1 | `d8b7a476-6877-429c-821d-35c779c56087` | PASSED | 4.4640 |
| 7 | `c4831914-583d-4dbf-a185-e246bf426098` | PASSED | 4.1280 |
| 18 | `37a25eac-6754-4c33-b5ca-37775bd5acc9` | PASSED | 4.1280 |
| 32 | `042aaf83-e452-482a-844f-0267918beaf2` | PASSED | 4.1440 |
| 64 | `f0fa41df-088a-4c01-935d-1e4813d0a955` | PASSED | 4.3360 |
| 539 | `38786469-9a12-4525-b284-865d5a37f856` | PASSED | 7.5680 |
| 11,949 | `ac1479c4-cfb0-4f86-a92d-47e7ad8e21cb` | PASSED | 79.8720 |
| 14,521 | `d9c27791-1f27-4feb-adb6-aa8a8c20556f` | PASSED | 95.3755 |

## Layout and cache-policy screen · 2026-11-26 18:37–18:43 UTC

Each screen paired its variants with the direct Triton control on physical GPU 3 at 1500/3996 MHz. The evaluator generated inputs with seed 200 and checked numerical tolerance and input preservation. Every measurement used cloned inputs, preallocated outputs, 10 warmups, 50 cold-L2 iterations and CUPTI first-to-last kernel timing. [Triton screen](gb300-triton-screen.jsonl) · [Native screen](gb300-native-screen.jsonl)

| Batch | Padded Triton (µs) | Exact 256-thread retain (µs) | Exact 256-thread reload (µs) | Native leader in screen (µs) |
|---:|---:|---:|---:|---|
| 1 | 4.512 | 6.752 | 6.784 | 3.200, 224 threads and seven output stripes |
| 7 | 4.192 | 6.304 | 6.272 | 3.040, 224 threads and seven output stripes |
| 18 | 4.144 | 6.048 | 5.984 | 3.152, 448 threads and two output stripes |
| 32 | 4.256 | 6.016 | 6.016 | 3.296, 448 threads and two or four output stripes |
| 64 | 4.352 | 6.400 | 6.400 | 3.520, 448 threads and two output stripes |
| 539 | 7.584 | 8.576 | — | 7.264, 224 threads and one whole row |
| 11,949 | 79.776 | 79.552 | — | 76.656, 224 threads and one whole row |
| 14,521 | 95.232 | 94.832 | — | 91.856, 224 threads and one whole row with a non-allocating store |

One native whole-row CTA used 32-byte loads and stores per thread and retained the FP32 sums. At B=539, 11,949, and 14,521, changes in the L1 load and store policies were within 0.3 µs of the ordinary store with L1 evict-last loads. At the two largest sizes, packing two rows per CTA or keeping 2, 4 or 8 resident CTAs per 148-SM grid-sizing proxy was slower. The physical count was 152 SMs.

All native trials passed. The Triton cache sweep had 24 build failures because PTXAS rejects the .cg load modifier paired with Triton's .evict_first policy. The measured large-workload cache and residency comparisons use the valid CUDA variants. Successful records contained all 50 CUPTI samples. [CompoundingKernel source, which inspired the replicated reduction and wide memory operations](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/003_fused_add_rmsnorm_h7168)
