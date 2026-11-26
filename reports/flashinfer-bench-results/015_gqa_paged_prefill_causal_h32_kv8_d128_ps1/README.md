# FlashInfer 015 benchmark record

## Direct-launch baseline on GB300

The exact DPS bundle passed 38/38 official workloads, including CUPTI timing, on 2026-11-26 UTC. The geometric-mean latency was 12.659 µs. The results and source are provisional GB300 evidence.

- [Per-workload evaluator results](gb300-baseline.jsonl)
- Source SHA-256 `833e91b246c9f61a9af862f9b5b5fee6df4c999b0b2e0310cd2c56b552586928`
- Physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory
- SOL-ExecBench commit `a9fa0804c793d438e70850c33fe34426e66d53dd` with v1.1 timing, Torch 2.10.0+cu130, Triton 3.6.0
- 10 warmups, 50 timed calls, cloned inputs, preallocated destinations, cold L2, and CUPTI first-to-last GPU activity
- The evaluator started with an empty Triton cache. The cold JIT measurement is pending the paired run.

## Paired scheduling screens on GB300

Both screening trials passed all nine tested workloads per variant, including the requested small, large-empty-prefix and long-KV cases. Both used physical GPU 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 queried SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory. Every pair used the same pinned device with the baseline timing protocol.

- [Fused-prefix screen](gb300-fused-screen.jsonl). The one-launch path reduced the 34-query, one-key row from 6.896 to 3.968 µs and the 10,447-query, one-key row from 20.064 to 16.800 µs.
- [Dispatch and split-KV screen](gb300-dispatch-screen.jsonl). The tuned split-KV path reduced the 2,560-index row from 97.521 to 22.688 µs. The fused tensor-core path was faster than the wider SIMD path on the screened irregular medium batches. The `first_call_s` fields record cold-cache first-call wall times, including JIT when a specialization is first used.
- The private B200 upload of candidate v1 (SHA-256 `f0ddfe404fda64aecc1d11178dea50c9b1cc2a9957f5497d17f2c8a34ff3d3cc`) returned HTTP 429 before issuing a submission ID. Slot 09 reported 0 active submissions against a limit of 5.
