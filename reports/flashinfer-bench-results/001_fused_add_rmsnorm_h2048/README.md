# 001_fused_add_rmsnorm_h2048

## 2026-11-26 17:23 UTC · B200 submission pending

Private v1.1 submission `105846` contains the candidate SHA-256 recorded below. The board lists 18:23:46 UTC as its scheduled result availability. [NVIDIA submission](https://research.nvidia.com/benchmarks/sol-execbench/submission/105846)

## 2026-11-26 17:22 UTC · GB300 candidate v1

The direct-launch Triton baseline and the exact collection candidate each passed 7/7 workloads in the local SOL-ExecBench CLI. The candidate uses replicated reductions for small batches and two rows per CTA for large batches. The paired run also passed input-preservation, near-cancellation, and changed-pointer checks. This candidate is provisional and verified on GB300 only.

- Physical GPU 0, `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 SMs, CUDA capability 10.3. CUDA was pinned by UUID. The GPU lock was held through each run, and the wrapper sampled for other compute jobs while its child ran.
- SM locked to 1500 MHz. Memory measured 3996 MHz and was the only supported memory clock. The device reported a 3996 MHz deferred setting.
- SOL-ExecBench 1.0.2 commit `a9fa080` with the v1.1 timing methodology, PyTorch 2.10.0+cu130, Triton 3.6.0, CUPTI Python 13.5.0. The evaluator and paired test used 10 warmups, 50 timed iterations, cold L2, cloned inputs, preallocated outputs and CUPTI first-to-last activity timing.
- Cold JIT plus first launch, in a fresh Triton cache, took about 0.518 s across the three candidate kernels. Baseline cold initialization, JIT and first launch took 0.690 s.
- Candidate SHA-256 `62947c0265d697eca4176f4faf2f70bfabd46ed39c8238454fa05599524b4689`. [Archive entry](../../../flashinfer-bench/001_fused_add_rmsnorm_h2048/submission.py)
- Basis for the candidate's tiling and split reductions. [Triton starting point](../../../kernel_library/kernels/sol-example-triton-rmsnorm/kernel.py) · [CompoundingKernel Explorer](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/001_fused_add_rmsnorm_h2048)

All latencies below are in microseconds. The CLI timed shifted pointers. The paired test used identical preallocated buffers for both implementations and also checked correctness with changed inputs and pointers. The input row in the official Parquet references generated tensors for all seven workloads.

| Batch | Workload UUID | Official GB300 status | CLI baseline | CLI candidate | Paired baseline | Paired candidate |
|---:|---|---|---:|---:|---:|---:|
| 6 | `906efdc6-aa3b-4b4f-a291-3a2be509d354` | PASSED | 2.752 | 2.752 | 2.688 | 2.656 |
| 1 | `1526086b-0c06-41a1-b7b9-47b5cfc830d7` | PASSED | 2.704 | 2.784 | 3.008 | 2.672 |
| 34 | `78c9b621-e151-4fb2-9162-f1b39afe573a` | PASSED | 2.784 | 2.816 | 3.008 | 2.816 |
| 79 | `7fd454c3-9611-4376-a920-3d1c30639790` | PASSED | 2.912 | 3.072 | 3.072 | 3.040 |
| 16254 | `724a6dfa-06ff-45b7-9972-d174a0e23737` | PASSED | 32.865 | 33.265 | 32.896 | 33.056 |
| 12383 | `5ede385e-9b04-4b49-903f-375a5180d815` | PASSED | 26.880 | 27.488 | 26.240 | 25.856 |
| 64 | `1602b8db-16e2-4d3f-b63d-68d4ee32e2cc` | PASSED | 2.880 | 3.008 | 3.040 | 3.008 |

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
