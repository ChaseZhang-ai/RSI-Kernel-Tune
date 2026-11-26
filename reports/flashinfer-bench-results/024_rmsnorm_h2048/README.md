# 024_rmsnorm_h2048

## Baseline · 2026-11-26 18:37 UTC

The exact direct-launch Triton file passed all seven official workloads in the local SOL-ExecBench CLI. Physical GB300 GPU 2 was pinned by UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`. The SM clock was locked at 1500 MHz, and the sole supported memory clock was 3996 MHz with a matching active sample. The wrapper monitored for foreign compute processes throughout the run.

SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` ran 10 warmups and 50 cold-L2 CUPTI iterations per workload with cloned, shifted inputs and preallocated outputs. PyTorch 2.10.0+cu130, Triton 3.6.0 and CUPTI Python 13.5.0. The paired comparison will query SM count and measure fresh-cache JIT time.

Baseline SHA-256 `4912b590fa90f14e3582f4936db0c484c6bb4aba0a4d3c8972ff46996464a282`. GB300 only. B200 v1.1 public leader 0.815575 SOL, checked 2026-11-26 18:37 UTC. [Official evaluator traces](gb300-baseline.jsonl) · [B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/233/B200)

| Batch | Workload UUID | Status | Latency (µs) |
|---:|---|---|---:|
| 1 | `b8b3dda7-8959-4a11-bd6a-59863bc6fffc` | PASSED | 4.048 |
| 6 | `50bbd632-cf16-4021-885b-625552ab8262` | PASSED | 3.648 |
| 34 | `90ca7f35-1e84-43ae-ac4b-805bc012d842` | PASSED | 3.488 |
| 64 | `74280ca1-cb97-433d-b8a9-c2aec2ae560c` | PASSED | 3.488 |
| 79 | `7fd55dbb-7371-4b46-99d1-84d1e3f16f06` | PASSED | 3.472 |
| 12383 | `68e79061-ffd0-4733-958a-3415321da93b` | PASSED | 19.568 |
| 16254 | `932f75f9-e29a-4502-8794-68347b591fd5` | PASSED | 24.272 |

## Triton layout screen · 2026-11-26 18:39 UTC

The paired screen used physical GB300 GPU 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), 152 queried SMs, capability 10.3, and 1500/3996 MHz. All trials used cloned inputs, preallocated BF16 destinations and the same evaluator CUPTI timing function with 10 warmups and 50 cold-L2 iterations.

102/158 workload/configuration trials passed correctness and timing, including every small-batch layout. 56 streaming and persistent cases failed at build time because PTXAS rejects `.cg` combined with `.evict_first`. The retry uses compatible cache modifiers. No numerical or CUPTI failures occurred in this screen.

Partitioned output stores with replicated FP32 statistics reduced the five small workloads to 2.720–2.848 µs in this screen. The best single- and multirow variants reached 19.616 µs at batch 12,383 and 23.776 µs at batch 16,254. The sweep records actual register counts and scratch usage for each compiled Triton kernel. [Triton sweep results](gb300-triton-sweep.csv)
