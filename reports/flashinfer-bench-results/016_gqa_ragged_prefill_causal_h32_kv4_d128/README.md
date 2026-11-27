# FlashInfer 016 ragged causal GQA

The collection candidate is provisional GB300 only. The exact DPS bundle passed all 15 official workload UUIDs with correctness and CUPTI timing on both Triton builds. Private B200 attempts returned HTTP 429 before a submission ID was issued. Slot 10 reported 0 of 5 active before each upload. NVIDIA's public B200 v1.1 leader was 0.965848 SOL at 2026-11-26T18:10:12Z. [Initial B200 attempt](b200-v2-attempt.json) · [Final B200 attempt](b200-final-attempt.json) · [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/225/B200)

## Local verification

- Primary GB300 physical GPU 0, `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, 152 SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory. SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130 and fbtriton 3.7.1 reporting Triton 3.7.0+fb.beta. The exact file's JIT compilation took 2.663 seconds across seven specializations. [Official results](gb300-final-fbtriton.json) · [Paired results](gb300-final-fbtriton-paired.json)
- Secondary GB300 physical GPU 1, `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, 152 SMs, CUDA capability 10.3, 1500 MHz SM and 3996 MHz memory. The exact file passed under stock Triton 3.6.0 with 2.315 seconds of JIT compilation. [Official results](gb300-final-triton36.json) · [Paired results](gb300-final-triton36-paired.json)
- The paired trials used 10 warmups and 50 timed iterations with cloned inputs, preallocated outputs, cold L2 and CUPTI first-to-last kernel timing. The GPU UUID mask and host-shared lock were held for each run. Each pair used the same physical GPU for both implementations.
- Supplemental checks compared against the official reference with unequal offsets, empty requests and KV slices, fully masked rows, changed tensor contents at the same pointers, and changed input and output pointers. [Reference checks](gb300-final-fbtriton.json)

## Paired GB300 latency

The all-workload geometric mean on physical GPU 0 was 7.853 µs, down from 8.272 µs for the direct Hyra-derived baseline, a 5.1% reduction. The table takes the median across workload IDs sharing each query count. Per-ID timing and status appear in the paired results.

| Total queries | Workload IDs | Baseline µs | Candidate µs |
|---:|---:|---:|---:|
| 1 | 3 | 2.560 | 2.560 |
| 6 | 3 | 3.408 | 2.880 |
| 34 | 5 | 4.064 | 4.240 |
| 172 | 1 | 9.937 | 9.521 |
| 969 | 1 | 25.376 | 21.553 |
| 12,251 | 1 | 465.444 | 410.547 |
| 16,294 | 1 | 663.364 | 580.242 |

The candidate launches directly and uses SIMT reductions for very short requests. For longer batches it reads both cumulative offset arrays on device and packs each group of eight query heads around one K/V head. A TMA producer and separate matrix warp overlap BF16 Tensor Core work with FP32 softmax and the output epilogue. The launch uses shape-derived grids. [Source file](../../../flashinfer-bench/016_gqa_ragged_prefill_causal_h32_kv4_d128/submission.py)

The direct baseline adapts Tencent Hyra submission 25891 under Apache-2.0. Short/long dispatch and the long pipeline draw on the CompoundingKernel Explorer notes, the repository's dense H100 GQA reference and the Triton Gluon warp-specialization tutorial. [Hyra seed](../../../kernel_library/kernels/hyra-016_gqa_ragged_prefill_causal_h32_kv4_d128/kernel.py) · [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/016_gqa_ragged_prefill_causal_h32_kv4_d128/README.md) · [Dense H100 reference](../../../kernels/attention/attn_fwd_gqa16/h100/gen2/kernel.py) · [Triton tutorial](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/gluon/08-warp-specialization.py)
