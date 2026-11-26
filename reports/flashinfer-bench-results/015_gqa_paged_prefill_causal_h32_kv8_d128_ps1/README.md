# FlashInfer 015 benchmark record

## Direct-launch baseline on GB300

The exact DPS bundle passed 38/38 official workloads, including CUPTI timing, on 2026-11-26 UTC. The geometric-mean latency was 12.659 µs. The results and source are provisional GB300 evidence.

- [Per-workload evaluator results](gb300-baseline.jsonl)
- Source SHA-256 `833e91b246c9f61a9af862f9b5b5fee6df4c999b0b2e0310cd2c56b552586928`
- Physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory
- SOL-ExecBench 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd` with v1.1 timing, Torch 2.10.0+cu130, Triton 3.6.0
- 10 warmups, 50 timed calls, cloned inputs, preallocated destinations, cold L2, and CUPTI first-to-last GPU activity
- The evaluator started with an empty Triton cache. The paired v1 record includes the baseline cold first-call measurements.

## Paired scheduling screens on GB300

Both screening trials passed all nine tested workloads per variant, including the requested small, large-empty-prefix and long-KV cases. Both used physical GPU 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 queried SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory. Every pair used the same pinned device with the baseline timing protocol.

- [Fused-prefix screen](gb300-fused-screen.jsonl). The one-launch path reduced the 34-query, one-key row from 6.896 to 3.968 µs and the 10,447-query, one-key row from 20.064 to 16.800 µs.
- [Dispatch and split-KV screen](gb300-dispatch-screen.jsonl). The tuned split-KV path reduced the 2,560-index row from 97.521 to 22.688 µs. The fused tensor-core path was faster than the wider SIMD path on the screened irregular medium batches. The `first_call_s` fields record cold-cache first-call wall times, including JIT when a specialization is first used.

## Candidate v1 on GB300

The exact DPS bundle passed all 38 official workloads on physical GPU 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`. The independent full paired run also passed all 38 workloads per variant on physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, and improved geometric-mean latency from 12.647 to 8.396 µs (1.51×). Both devices reported 152 SMs and CUDA capability 10.3 at 1500/3996 MHz SM/memory.

- [Exact DPS evaluator results](gb300-v1-evaluator.jsonl)
- [Paired timings, checks and first-call times](gb300-v1-paired.jsonl)
- Cold Triton compile-and-launch wall-time bound from an isolated cache was 10.309 s summed across the 38 first calls, with a 1.097 s maximum. The baseline bounds were 7.596 s total and 1.120 s maximum.
- The full paired run measured 3.888 µs for the 34-query, one-index row, 16.576 µs for the 10,447-query, one-index row, and 22.816 µs for the 2,560-index row.

## Candidate v2 screening on GB300

The [paired v2 screen](gb300-v2-screen.jsonl) used physical GPU 1 at the same clocks. All tested variants passed nine official workloads. A bounded grid reduced the 2,560-index row from 22.688 to 18.784 µs. TMA zero writes improved the screened 16,384-query row from 24.113 to 23.392 µs and the screened 2,002-query row from 8.209 to 7.376 µs. One-warp and grouped-head variants showed higher latency for most screened short windows, so v2 retains four warps and one query head per CTA on the smallest windows.

## Candidate v2 on GB300

The provisional submission is the exact v2 DPS source. SOL-ExecBench passed 38/38 official workloads with a geometric-mean latency of 8.294 µs on physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, 152 queried SMs, CUDA capability 10.3 and 1500/3996 MHz SM/memory. The 34-query one-index, 10,447-query one-index and 2,560-index workloads took 4.000, 16.448 and 18.832 µs.

- [Exact DPS evaluator timings](gb300-v2-evaluator.jsonl)
- Source SHA-256 `a0add67d61e67c3d1b69dcd541d3d1db5a367e37586bd707f15c9b26dccf3d6b`
- The largest official K/V pair occupies 2.733 GiB. The split-KV workspace is bounded at 87.66 MiB across the official workloads.
- The v3 screen on physical GPU 2 found comparable or higher latencies with preloaded single-request SIMT and single-tile split-KV alternatives. [Per-workload measurements](gb300-v3-screen.jsonl)
- The v4 screen on physical GPU 3 found that wider zero tiles regressed an irregular batch and that earlier dispatch to TMA mainly benefited the 398-query case. Both screens used the same locked clocks and v1.1 timing protocol. [Per-workload measurements](gb300-v4-screen.jsonl)

The local definition, 38 workload IDs and tolerances came from the named Parquet row. The dataset SHA-256 was `89d1feacf02b6f8c07ec8d67a9dbb2c531652251095a7f762f2bc296f6a1fe74`. The evaluator resolved the workload metadata from FlashInfer trace version 1.0. [Official Parquet](https://huggingface.co/datasets/nvidia/SOL-ExecBench/resolve/main/data/FlashInfer-Bench.parquet) · [FlashInfer trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/tree/1.0)

## Full paired comparison and cold compilation

On physical GB300 GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`), baseline and v2 each passed 38/38 official workloads. The geometric mean improved from 12.357 to 8.208 µs (1.51×). The 2,560-index workload improved from 97.457 to 19.024 µs in this matched trial. [Paired CUPTI timings and correctness checks](gb300-v2-paired.jsonl)

The Triton compiler reported 10.352 s across 33 cold compilations targeting CUDA architecture 10.3 on physical GB300 GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`). The slowest individual compile took 0.731 s, and the slowest cold first call took 1.254 s. Both GPUs had 152 queried SMs and 1500/3996 MHz SM/memory clocks. [Cold compiler timings](gb300-v2-cold-compile.jsonl)

The extra correctness cases passed with `q_len > kv_len`, `kv_len > q_len`, empty requests, masked tail splits, changed live metadata, new input and output pointers, a non-default negative scale and a page ID of 2,097,155. Empty rows were checked for exact zero output and negative-infinite LSE. The metadata and pointer changes also passed on a 541-query fixture that exercises the TMA zero writes on physical GPU 2. [Synthetic correctness results](gb300-v2-synthetic.jsonl)

## Private B200 v1.1 attempts

| Attempt (UTC on 2026-11-26) | Artifact | Outcome | Submission ID |
|---|---|---|---|
| 17:48:39 | v1 (`f0ddfe404fda64aecc1d11178dea50c9b1cc2a9957f5497d17f2c8a34ff3d3cc`) | HTTP 429 before evaluation | — |
| 17:53:46 | v2 | HTTP 429 before evaluation | — |
| 18:00:01 | Exact published v2 | HTTP 429 before evaluation | — |

Slot 09 reported zero active submissions against a limit of five at each attempt. Each upload returned without a retry interval. The v1.1 public leader was June at 0.996676 SOL at 2026-11-26 17:59 UTC. [Official B200 leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/224/B200)
