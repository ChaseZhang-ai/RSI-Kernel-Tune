# Paged MLA decode benchmark record

## GB300 initial direct and split-KV screen

Both candidates passed correctness and timing on all six screened official workloads. The split-KV variant reduced latency on the larger inputs, including 91.232 µs versus 829.584 µs for batch 64 with 75,145 pages. Both candidates shared the single-launch kernel on the eight-page workload. The full evaluator run is in progress. [Per-workload results](gb300-initial-screen.jsonl)

- Physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, locked to 1500/3996 MHz SM/memory. PyTorch reported 152 SMs and CUDA capability 10.3.
- NVIDIA SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130, Triton 3.6.0. The runner used the official definition, trace metadata and tolerances, and measured median CUPTI first-to-last activity with 10 warmups, 50 timed iterations, cold L2, cloned inputs and preallocated destinations.

## GB300 split-KV v1

The exact destination-passing bundle passed all **47 official workloads** in the SOL-ExecBench CLI at 2026-11-26 18:53 UTC. Every workload produced timing. The candidate uses a direct SIMT launch for the eight-page workload and an FP32 online softmax with an FP32 split-KV merge elsewhere. Each split reuses its CKV tile for the logits and weighted values. Grid and scratch size follow the active tile bound derived from the length of the page-index list. [Per-workload results](gb300-v1-evaluator.jsonl) · [Device and summary](gb300-v1-summary.json)

| Batch size | Workloads passed | CLI geomean (µs) |
|---:|---:|---:|
| 1 | 15/15 | 15.103 |
| 16 | 16/16 | 25.881 |
| 64 | 16/16 | 60.811 |
| All | 47/47 | 29.149 |

- The CLI ran on physical GPU 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`. Clocks were locked at 1500/3996 MHz and PyTorch queried 152 SMs, capability 10.3. The lock covered the evaluator. The first CLI attempt stopped before evaluation because the subprocess could not find `python`. The successful retry added the evaluator environment to `PATH`.
- Local evaluator 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130 and Triton 3.6.0. The official CLI used 10 warmups, 50 iterations, cold L2, cloned inputs and preallocated outputs. The cold paired screen took approximately 2.42 seconds for the first calls of the five v1 kernel specializations, including JIT, launch and synchronization. The eight-page kernel compiled on the baseline call because both variants use the same kernel.
- The bundle embeds submission SHA-256 `16d52f3ee123e6164d8a119a226688544a9847827227ee090451c2022c4efa6b`. A separate CPU-only compile check also emitted SM 10.0 cubins with Triton 3.5.0. The runtime verification was on GB300. The 75,145-index case requires about 20.5 MiB of split workspace.
- The v1.1 public B200 boards showed 0.994987 SOL for the operator leader and 0.940633 SOL for the collection leader at 2026-11-26 18:39 UTC. Candidate verification is provisional GB300 only. [Kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/227/B200) · [Collection board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)
