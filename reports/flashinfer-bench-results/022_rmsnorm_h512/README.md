# 022_rmsnorm_h512

## GB300 direct-launch baseline

The one-row, 128-thread Triton CTA passed all eight official workloads in SOL-ExecBench 1.0.2 with PyTorch 2.10.0+cu130, Triton 3.6.0, and CUPTI Python 13.5.0. Median workload latencies have a **3.795 µs geometric mean**. The [CLI trace](gb300-baseline.jsonl) records each official UUID, tolerance, status, and latency.

Physical GPU 0, `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, ran under the host lock at 1500 MHz SM and 3996 MHz memory. The evaluator performed 10 warmups and 50 cold-L2 CUPTI iterations per workload, with cloned inputs and destination buffers. Baseline SHA-256 `f592ab368e39a5ea40094688e0994d794804d0c6b77609f9efcba5a5cdb391da`. The layout sweep records queried SM count and the first-call cold compilation sample.

On 2026-11-26 at 18:38:08 UTC, June led the public v1.1 B200 kernel board with 0.818143 SOL (submission 104772) and the collection with 0.940633 SOL (104799). [B200 kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/231/B200) · [B200 collection](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)

| Batch | Workload UUID | Baseline (µs) | Status |
|---:|---|---:|---|
| 1 | `69d9dba5-9d8f-4f6f-901f-43b5ae58b151` | 2.496 | PASSED |
| 7 | `e0fe7cd7-226c-45dd-af09-45463cc10d39` | 2.464 | PASSED |
| 18 | `4b5050f5-d6ab-4b20-bb9c-409c073f4d37` | 2.656 | PASSED |
| 32 | `0315f2ba-9254-4de4-854e-7aafe6460edc` | 2.688 | PASSED |
| 64 | `eee1d057-c1a3-42c1-bb57-71223bc091ee` | 2.688 | PASSED |
| 539 | `625f29dc-a8de-48ba-afdb-96727b2a7feb` | 2.960 | PASSED |
| 11,949 | `6a4a7810-cc7b-4641-9152-91112a15e489` | 10.368 | PASSED |
| 14,521 | `0581e118-09fa-4f5f-9910-d1b4a7573353` | 11.872 | PASSED |
