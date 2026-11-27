# 024_rmsnorm_h2048 B200 follow-up

## 2026-11-27 04:27 UTC · Provisional GB300 candidate

The exact candidate passed **7/7** official workloads at **4.640 µs** geometric mean in the SOL-ExecBench CLI. Against the exact archived file, the two paired orders measured **1.0082×** and **0.9999×**, with **1.004×** geometric-mean speedup after averaging the two medians per workload. The three revised dispatches measured **1.014×** together, and each was faster or tied in both timing orders. Candidate SHA-256 `e8fd74560e21c931cbfc507825229f503c2b537fbbe8af372a602b777cd5eea1`. [Candidate](../../../flashinfer-bench/024_rmsnorm_h2048/submission.py) · [Official CLI results](gb300-followup-evaluator.jsonl) · [Paired records](gb300-followup-paired.jsonl)

### B200 evidence and diagnosis

The v1.1 collection child **106093** identifies workload `932f75f9-e29a-4502-8794-68347b591fd5` at batch 16,254 as **timing activity missing**. Standalone repeat **106229** independently timed **6/7** and has an aggregate result in the coordinator's evidence. Its missing workload UUID is unspecified. The archived standalone source was byte-identical to the tracked comparator, SHA-256 `ec39fe4165098aa669551b1f1231a05b1a82aed4d877c24ed0346dc0ffbf9d8f`. The public v1.1 leader was **0.815575 SOL** in the 2026-11-27 04:17 UTC snapshot. [B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/233/B200)

The archived file passed 7/7 in the original GB300 CLI run and again in this follow-up. Its batch 16,254 kernel issued one ordinary launch on the current stream. The in-kernel prefetch target stays inside the input allocation and the prefetch generates no separate kernel launch. Both old and new artifacts timed the largest workload in the fresh local CLI and both paired orders. The screen selected the archived streaming route for this workload on measured latency. The evidence for this revision is a local speed improvement. [Archived-file CLI repeat](gb300-followup-archived.jsonl)

### Paired results

Each paired column averages the two trial medians. The CLI latencies come from separate locked jobs and are listed alongside the paired comparison.

| Batch | Official workload UUID | New CLI status | Paired archived (µs) | Paired new (µs) | Archived CLI (µs) | New CLI (µs) |
|---:|---|---|---:|---:|---:|---:|
| 1 | `b8b3dda7-8959-4a11-bd6a-59863bc6fffc` | PASSED | 2.632 | 2.656 | 2.528 | 2.112 |
| 6 | `50bbd632-cf16-4021-885b-625552ab8262` | PASSED | 2.512 | 2.512 | 2.528 | 2.560 |
| 34 | `90ca7f35-1e84-43ae-ac4b-805bc012d842` | PASSED | 2.624 | 2.640 | 2.656 | 2.592 |
| 64 | `74280ca1-cb97-433d-b8a9-c2aec2ae560c` | PASSED | 2.736 | 2.704 | 2.784 | 2.704 |
| 79 | `7fd55dbb-7371-4b46-99d1-84d1e3f16f06` | PASSED | 2.776 | 2.736 | 2.736 | 2.720 |
| 12383 | `68e79061-ffd0-4733-958a-3415321da93b` | PASSED | 19.192 | 18.872 | 19.120 | 19.104 |
| 16254 | `932f75f9-e29a-4502-8794-68347b591fd5` | PASSED | 23.136 | 23.136 | 23.168 | 23.520 |

### Layout and coverage

The screen compared 128- and 256-thread output partitions, full-row CTAs, single-warp partials and groups of four partials per warp. The first exact revision passed 7/7 in the CLI and measured 4.884 µs versus 4.858 µs for the archive in the pair. Its batch 6 and 34 gains from the screen reversed in that pair, so the selected artifact uses the archived layouts for batches 1, 6 and 34. At batches 64 and 79 the selected 128-thread CTA loads corresponding vectors from both halves, writes its own half and uses an L2 prefetch CTA per row. The selected batch 12,383 route is a direct 128-thread CTA per row. The batch 16,254 conservative direct trials averaged 23.784 µs with native packed math and 25.832 µs with portable Triton. The archived streaming route averaged 23.248 µs in that screen. All 132 screening trials produced full timing records and passed correctness. [Screen results](gb300-followup-screen.csv) · [Initial exact revision pair](gb300-followup-initial-paired.jsonl)

The exact pair passed the official tolerances, BF16 output shape and dtype, input preservation, low-magnitude mixed signs, signed non-uniform weights, changed contents and pointers on a non-default stream, a two-byte-aligned view and a strided view. The paired reference comparison found 160 differing BF16 values out of 59,025,408, with maximum relative error 0.007812. The cold native compilation, module load and first launch took **0.082 s**. Peak measured PyTorch allocation in the paired check was 1.43 GiB.

Screen on GB300 GPU 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 SMs, CUDA capability 10.3. Exact pair on GB300 GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, 152 SMs, CUDA capability 10.3. The archived-file CLI ran on GB300 GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, 152 SMs, CUDA capability 10.3 and the candidate CLI ran on GB300 GPU 0, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 SMs, CUDA capability 10.3. Each job pinned its UUID, held the host GPU lock, monitored compute contention and used **1500/3996 MHz** SM/memory clocks. Paired timing used 10 warmups and 50 cold-L2 CUPTI iterations with cloned, shifted inputs and preallocated outputs. The evaluator was SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` with PyTorch 2.9.0+cu130, fbtriton 3.7.1 and CUPTI Python 13.5.0.
