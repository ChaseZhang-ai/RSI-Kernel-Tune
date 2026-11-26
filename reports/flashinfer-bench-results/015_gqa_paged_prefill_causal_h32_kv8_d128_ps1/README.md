# FlashInfer 015 benchmark record

## Direct-launch baseline on GB300

The exact DPS bundle passed 38/38 official workloads, including CUPTI timing, on 2026-11-26 UTC. The geometric-mean latency was 12.659 µs. The results and source are provisional GB300 evidence.

- [Per-workload evaluator results](gb300-baseline.jsonl)
- Source SHA-256 `833e91b246c9f61a9af862f9b5b5fee6df4c999b0b2e0310cd2c56b552586928`
- Physical GPU 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 queried SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory
- SOL-ExecBench commit `a9fa0804c793d438e70850c33fe34426e66d53dd` with v1.1 timing, Torch 2.10.0+cu130, Triton 3.6.0
- 10 warmups, 50 timed calls, cloned inputs, preallocated destinations, cold L2, and CUPTI first-to-last GPU activity
- The evaluator started with an empty Triton cache. The cold JIT measurement is pending the paired run.
