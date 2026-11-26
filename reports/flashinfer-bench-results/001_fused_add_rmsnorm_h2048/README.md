# 001_fused_add_rmsnorm_h2048

## 2026-11-26 · GB300 setup blocked

The workspace process sandbox failed at startup with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. The failure reproduced with `/usr/bin/true`, and workspace file creation also failed. GPU lock acquisition, clock locking, the evaluator launch, and private B200 submission were never reached.

| Field | Outcome |
|---|---|
| Local device | Host inventory showed four NVIDIA GB300s at indices 0–3. No device was selected, so no device UUID, SM count, or benchmark clock sample was recorded. |
| Clocks | Inventory showed 1500/3996 MHz graphics/memory on GPU 0 and 120/3996 MHz on GPUs 1–3. These were observed clocks before any lock or test. |
| Verification scope | 0 of 7 workloads executed. Correctness, input preservation, cancellation, CUPTI timing and cold compile time were unmeasured. |
| Evaluator | Local evaluator was not started. The target B200 board uses evaluation stack v1.1. |
| B200 submission ID and score | — |
| Public B200 rank and candidate gap | — |

The v1.1 public B200 leader was 0.721444 SOL at the 2026-11-26 16:46 UTC check. [NVIDIA kernel 210 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/210/B200)

## Official workloads

The named dataset row specifies a hidden size of 2,048 and generated inputs for all seven workloads. [NVIDIA SOL-ExecBench dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench/viewer/FlashInfer-Bench/train)

| Batch size | Official workload UUID | Status | Latency |
|---:|---|---|---:|
| 6 | `906efdc6-aa3b-4b4f-a291-3a2be509d354` | Not run | — |
| 1 | `1526086b-0c06-41a1-b7b9-47b5cfc830d7` | Not run | — |
| 34 | `78c9b621-e151-4fb2-9162-f1b39afe573a` | Not run | — |
| 79 | `7fd454c3-9611-4376-a920-3d1c30639790` | Not run | — |
| 16254 | `724a6dfa-06ff-45b7-9972-d174a0e23737` | Not run | — |
| 12383 | `5ede385e-9b04-4b49-903f-375a5180d815` | Not run | — |
| 64 | `1602b8db-16e2-4d3f-b63d-68d4ee32e2cc` | Not run | — |
