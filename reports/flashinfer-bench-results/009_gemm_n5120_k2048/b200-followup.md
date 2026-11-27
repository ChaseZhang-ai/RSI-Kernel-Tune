# FlashInfer 009 B200 follow-up

The revised DPS file passed **25/25** official workloads at **22.703 µs** geomean through the local SOL-ExecBench CLI. In three paired GB300 trials it reduced the geomean from **23.983 to 22.749 µs** (5.1%). Revised source SHA-256 is `5114cf13bae2cb134b23b96ad7f9c86b54c218033fedf608f41d3917a71ebdbe`. The coordinator will measure this exact file on B200. [Submission](../../../flashinfer-bench/009_gemm_n5120_k2048/submission.py) · [Exact CLI results](gb300-followup-cli.jsonl) · [Paired results](gb300-followup-paired.jsonl)

## Archived B200 evidence

The tracked starting file matched the fleet archive byte for byte. Standalone B200 v1.1 submission **106253** passed and timed all 25 workloads for **0.375971 SOL**. It is unranked on the public board. The signed gap to the **1.000000** leader is **+0.624029**, checked at 2026-11-27 04:25:50 UTC. That score describes the archived artifact. [Archived fleet evidence](followup-archived-b200.json) · [Public board check](followup-public-board.json) · [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200)

The released collection child **106078** timed 23 of 25 workloads for the same source. M=492 used a one-launch TMA raster and M=8,828 used a one-launch persistent TMA path. Both timed in the independent standalone repeat. The misses span different kernels and did not reproduce across the official runs. The supplied standalone record has no per-workload repeats or B200 counters, so it cannot resolve the capture failures or isolate clock differences. The revised dispatch follows measured latency gains. [Archived fleet evidence](followup-archived-b200.json)

- M=1 ran in 7.503 µs on B200 versus the 8.768 µs scoring baseline. M=4–8 ran at 1.18–1.20× scoring baseline latency. The local warp-split FP32 reduction lost to the archived MMA path at M=4 and M=8, and direct multirow GEMV was slower still. The candidate keeps the archived dispatch for M≤32. [Direct kernel screen](gb300-followup-screen.csv)
- M=34–172 ran at 1.38–1.52× the B200 scoring baseline. The local comparison supported a smaller K tile and four TMA stages for M=34–64. The other midrange tiles had no repeatable advantage in this screen. [Direct kernel screen](gb300-followup-screen.csv)
- M=289 and 952 used 24.143 and 41.636 µs on B200, or 1.85× and 1.95× their scoring baselines. The archive used pointer MMA at M=289 and ordinary 128-column rasterization at M=952. Both choices gave up latency to persistent tcgen05 with a five-stage operand ring and 64-column store segments in local tests. [Schedule screen](gb300-followup-explicit.csv)
- The five timed large M cases ran at 1.26–1.29× the B200 scoring baseline and 1.14–1.18× the same archive's prior GB300 latency. At M=16,294 the archive delivered about 1.19 PFLOP/s on B200 and 1.40 PFLOP/s on GB300. The prior tuning win was measured against a direct local Triton baseline. It does not establish an advantage over the board's scoring reference. [Archived fleet evidence](followup-archived-b200.json) · [Prior exact GB300 timings](gb300-v2-cli.jsonl)

## Schedule and output

The large path still uses K-contiguous FP16 A and B with grouped M rasterization. Dedicated load and MMA warps advance a shared-memory ring. A pair of FP32 tensor-memory slots lets MMA accumulate one tile while the epilogue drains another. The epilogue casts each destination element to FP16 once and stores TMA segments into C. At M=12,251, 32-column segments measured 171.573 µs versus 175.407 µs for 64 and 178.928 µs for 128. The archived path used 229,424 bytes of shared memory. The revised 32-column epilogue and four-stage ring use 205,044 bytes. Some two-CTA configurations failed compiler lowering on GB300. The selected kernels use one CTA and no global scratch. [Explicit screen](gb300-followup-explicit.csv) · [Compile-only targets](followup-persistent-targets.json)

The archived persistent schedule and all three revised persistent configurations compiled offline for both `sm_100a` and `sm_103a`. They emitted tcgen05 MMA and TMA stores on either target. This confirms the compiler accepts the selected B200 layouts and shared-memory sizes. Hardware timing and runtime behavior for this revision remain GB300 scope. [Compile-only record](followup-persistent-targets.json)

## Paired results

| M | Archived (µs) | Revised (µs) | Latency reduction |
|---:|---:|---:|---:|
| 34 | 11.996 | 11.842 | 1.3% |
| 63 | 12.053 | 11.809 | 2.0% |
| 64 | 11.965 | 11.772 | 1.6% |
| 289 | 21.662 | 15.479 | 28.5% |
| 492 | 23.422 | 20.659 | 11.8% |
| 952 | 37.427 | 30.705 | 18.0% |
| 8,828 | 142.576 | 129.578 | 9.1% |
| 11,006 | 174.742 | 157.624 | 9.8% |
| 12,251 | 190.320 | 172.629 | 9.3% |
| 12,853 | 200.818 | 181.454 | 9.6% |
| 14,915 | 229.946 | 208.370 | 9.4% |
| 16,294 | 244.391 | 221.642 | 9.3% |

Both arms passed every official workload in all three paired trials. The revised arm also passed 50 exact full-width checks with changed inputs and pointers. The GPU was physical GB300 **1**, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with CUDA capability **10.3**, **152 queried SMs**, and active **1500/3996 MHz** clocks. The SM clock was locked and the driver accepted the deferred memory setting. The CLI used physical GB300 **1**, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, at the same measured clocks. The initially empty paired JIT cache took **6.274 s** in summed first synchronized calls across both arms, including JIT. GPU timing used SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130, fbtriton 3.7.1, CUPTI Python 13.5.0, 10 warmups and 50 timed calls per trial, shifted cloned inputs, cold L2 and first-to-last CUPTI spans. [Paired metadata](gb300-followup-paired-metadata.json) · [Paired clocks](gb300-followup-paired-clocks.json) · [CLI clocks](gb300-followup-cli-clocks.json)
