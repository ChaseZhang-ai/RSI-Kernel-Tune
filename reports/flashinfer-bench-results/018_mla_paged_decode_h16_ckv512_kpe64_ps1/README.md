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
- Local evaluator 1.0.2 at commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130 and Triton 3.6.0. The official CLI used 10 warmups, 50 iterations, cold L2, cloned inputs and preallocated outputs. The cold paired screen took approximately 2.42 seconds for the first calls of the six v1 kernel specializations, including JIT, launch and synchronization. The eight-page kernel compiled on the baseline call because both variants use the same kernel.
- The bundle embeds submission SHA-256 `16d52f3ee123e6164d8a119a226688544a9847827227ee090451c2022c4efa6b`. A separate CPU-only compile check also emitted SM 10.0 cubins with Triton 3.5.0. The runtime verification was on GB300. The 75,145-index case requires about 20.5 MiB of split workspace.
- The v1.1 public B200 boards showed 0.994987 SOL for the operator leader and 0.940633 SOL for the collection leader at 2026-11-26 18:39 UTC. Candidate verification is provisional GB300 only. [Kernel board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/227/B200) · [Collection board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200)

## GB300 Blackwell tuning screen

The largest v1 case read about 86.6 MB of cache data and took 90.866 µs. The tuning screen compared asynchronous staging and tensor-core execution, tile sizes, and scratch bandwidth. Every variant passed correctness and timing on the six screened official workloads, including batch 1 with 8, 108 and 2,708 references. [Paired timings](gb300-tuning-screen.jsonl)

| Batch | Index count | Split-KV v1 (µs) | Fastest screened variant (µs) |
|---:|---:|---:|---:|
| 1 | 8 | 6.528 | 6.160 |
| 1 | 108 | 14.176 | 9.841 |
| 1 | 2,708 | 20.497 | 17.504 |
| 16 | 457 | 22.992 | 12.337 |
| 64 | 30,745 | 46.113 | 33.233 |
| 64 | 75,145 | 90.770 | 70.530 |

- The screen ran on physical GPU 3 at 1500/3996 MHz SM/memory and completed at 18:55:11 UTC. It used the same software, workload tolerances, and paired protocol as the earlier comparison. The trace records cold first-call times including JIT and synchronization for each variant.
- A TMA gather and one tensor-core tile performed best at 108 references. The long single-row case favored asynchronous copy with FP32 split values, and the batched cases favored asynchronous copy with narrower BF16 partial storage and an FP32 merge. The all-SIMT one-launch 108-reference trial took 22.913 µs.
- A 28-call synthetic suite also passed on v1 after correcting a test fixture whose requested number of pages exceeded its backing allocation. The suite exercised randomized and repeated page IDs, empty rows, independent positional changes in place, a different scale, and changed destination and input pointers with reordered row lengths. [Synthetic results](gb300-v1-edge.json)

## GB300 hybrid v2 paired evaluation

The assembled hybrid and v1 both passed the eight-workload follow-up. On the same physical GB300, geomean latency fell from 22.575 to 16.656 µs, a 26.2% reduction. The 108-reference case took 10.064 µs and the 75,145-reference case took 69.1375 µs. [Paired timings](gb300-v2-pair.jsonl)

- Physical GPU 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, with 1500/3996 MHz clocks, 152 queried SMs and CUDA capability 10.3. The pair completed at 2026-11-26 18:56:37 UTC under the shared lock.
- Exact hybrid v2 passed all 28 synthetic calls, including the direct TMA route with permuted pages, independent positional changes, and replaced input and output pointers. The cold first calls for the paired sample summed to 2.050 seconds, including Triton JIT and run synchronization. [Synthetic results](gb300-v2-edge.json)

Hybrid v2 also passed all **47** official CLI workloads on physical GPU 1 at 1500/3996 MHz, finishing at 2026-11-26 18:59:21 UTC. Geomean was 21.250 µs across all workloads. Forty-six workloads were faster than in the v1 CLI run. The 1,908-reference workload measured 35.184 µs in v2 and 17.952 µs in v1. The paired results above control for physical GPU. [Per-workload results](gb300-v2-evaluator.jsonl) · [Device and summary](gb300-v2-summary.json)

An offline Triton 3.5.0 check caught a compiler error in v2's direct kernel when broadcasting scalar constants in the empty-row epilogue. The fix gives those constants explicit distributed layouts. The v1 source remains staged pending full evaluation of the compatibility fix.
