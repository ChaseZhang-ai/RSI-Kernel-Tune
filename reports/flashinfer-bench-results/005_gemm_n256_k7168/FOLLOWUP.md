# 005_gemm_n256_k7168 follow-up

The exact revised candidate passed **17/17** official workloads with CUPTI timing on GB300 at **7.137 µs** geometric mean. In three full comparisons against the exact B200 batch artifact, the geometric mean of the per-workload medians improved from **7.316 to 7.215 µs (1.38% lower)**. The revision was faster in each trial. All 57 synthetic checks passed, including near cancellation and changed inputs and pointers at M=54. [Official trace](followup-gb300.jsonl) · [Paired trace](followup-paired-gb300.jsonl) · [Summary](followup-summary.json) · [Synthetic checks](followup-gb300-edges.json)

## B200 evidence

The supplied v1.1 records at 2026-11-27 04:17:51 UTC identify standalone submission 106249 with 15/17 timed workloads and collection child 106074 with 16/17 timed. The collection's missing activity was M=54, UUID `84f0b901-a934-4f54-9ae0-23fb641d5083`. The standalone summary gives an aggregate count. Both outcomes are incomplete and the per-kernel leader in the supplied snapshot is 0.915728 SOL. The exact starting comparator and previously tracked file shared SHA-256 `74d68660b4b46e123f61c63ddb601e413b9284501ed9e1fde8236c172fdfa625`. [B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/214/B200)

The released collection timed M=53, 55, 56, 57, 58 and 63 on the same dispatch with latencies from 5.791 to 5.888 µs. The archived comparator recorded M=54 activity twice in the screen and three times in the full local pairs. This supports retaining the existing GEMV/MMA crossover and treating the missing production activity as a timing failure with an unconfirmed cause.

## Revised kernel

The dense path retains the 32×16 tile and seven-way K split. Each owner holds its own FP32 contribution in registers, receives the other six contributions via distributed shared memory, and rounds to FP16 once. Skipping padded output groups and self-transfers reduces M=54 cluster sends by 28%. All seven official M=53–63 shapes improved by 2.9–3.5% in the paired medians. M=54 improved from **5.504 to 5.312 µs**. The archived direct GEMV at M=1 and clustered MMA at M=4 were faster than the screened alternatives. [Focused screen](followup-screen.csv)

The SM100 offline build used Triton 3.5.0, CUDA 13.1 PTXAS and NVRTC 12.9. The revised dense kernel uses 26,720 B of shared memory per CTA with no register spills in that build. The archive's largest allocation is 196,656 B per CTA, below the SM100 limit of 232,448 B. B200 runtime verification is pending. [Build record](followup-sm100-build.txt) · [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

## Local setup

The official run used physical GB300 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`), the paired runs used GB300 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), and edge checks used GB300 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`). Each queried 152 SMs and CUDA capability 10.3. SM clocks were locked at 1500 MHz and memory stayed at the host's deferred 3996 MHz setting. Each job held the shared GPU lock and selected its GPU by UUID. SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` used its v1.1 methodology with PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. Timing used 10 warmups, 50 iterations, cold L2, cloned inputs, preallocated destinations, and first-to-last CUPTI activity. Peak paired PyTorch allocation was 0.664 GiB.

The current SHA-256 is `7670abf8fd6c7ef726b8d2782c0b11c4e13f4862d1ca576a7f0396a65dcb5f51`. First synchronized calls from an empty Triton cache took 1.016 s, including the native build and Triton JIT. The local destination-passing solution embeds the exact source. The official trace retains all 17 workload UUIDs from the staged Parquet. [Materialization script](prepare.py)
