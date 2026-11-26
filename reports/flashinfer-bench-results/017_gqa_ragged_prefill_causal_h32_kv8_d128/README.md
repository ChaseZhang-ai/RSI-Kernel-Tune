# FlashInfer 017 ragged causal GQA

The provisional GB300 candidate passed **21/21 official workload IDs** in both the paired run and exact DPS evaluator run. The paired geometric mean was **8.691 µs**, down **9.6%** from 9.615 µs for the direct baseline. The standalone evaluator measured **8.576 µs** geometric mean. NVIDIA's public B200 v1.1 leader was **0.959860 SOL** at 2026-11-26T18:57:56Z. B200 verification is pending with the coordinator. [Paired results](gb300-paired.json) · [Exact evaluator results](gb300-final.json) · [Public B200 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/226/B200)

## Verification

- The exact Python artifact ran from a local DPS bundle with SOL-ExecBench 1.0.2, PyTorch 2.10.0+cu130 and fbtriton 3.7.1 reporting Triton 3.7.0+fb.beta. Cold JIT compilation took 5.653 seconds across 11 specializations. [Candidate](../../../flashinfer-bench/017_gqa_ragged_prefill_causal_h32_kv8_d128/submission.py) · [Baseline evaluator results](gb300-baseline.json)
- Exact evaluation used physical GPU 2 (GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0), 152 SMs, CUDA capability 10.3. The paired run used physical GPU 2 (GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0), 152 SMs, CUDA capability 10.3. Both held the host-shared lock, pinned the GPU by UUID, and ran at 1500 MHz SM and 3996 MHz memory. Each paired workload used the same physical GPU, 10 warmups, 50 timed calls, cold L2, cloned inputs, preallocated outputs, and CUPTI first-to-last kernel timing.
- Eleven supplemental cases on physical GPU 3 (GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f), 152 SMs, CUDA capability 10.3 passed 33 elementwise reference comparisons. They covered unequal lengths, empty slices, masked rows, skewed request lengths, varied scales, changes to contents at fixed pointers, and new input and output pointers. [Supplemental results](gb300-edges.json)
- The screen compared four-head groups with independent head pairs, a 148-CTA persistent grid with direct tiling, and separate diagonal and prefix key widths. The final dispatch uses a single-key reduction, a short SIMT or packed MMA tile, and a TMA and tcgen05 pipeline for long contiguous inputs. It reads live offsets on the GPU and requires no global scratch allocation. [Tuning screen](gb300-tuning.json)
- The grouped and heavy-first order draws on CompoundingKernel Explorer. The long pipeline adapts FlashInfer 016 and the Triton Gluon warp-specialization tutorial. The online softmax follows the repository's H100 GQA reference. [Archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/017_gqa_ragged_prefill_causal_h32_kv8_d128/README.md) · [FlashInfer 016](../../../flashinfer-bench/016_gqa_ragged_prefill_causal_h32_kv4_d128/submission.py) · [Triton tutorial](https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/gluon/08-warp-specialization.py) · [H100 reference](../../../kernels/attention/attn_fwd_gqa16/h100/gen2/kernel.py)

## Paired GB300 latency by query count

Medians across workload IDs with the same query count.

| Queries | Workload IDs | Baseline (µs) | Candidate (µs) |
|---:|---:|---:|---:|
| 1 | 3 | 2.880 | 2.896 |
| 7 | 3 | 3.872 | 3.201 |
| 18 | 2 | 4.072 | 4.080 |
| 35 | 5 | 4.400 | 4.400 |
| 71 | 2 | 5.144 | 4.992 |
| 81 | 1 | 10.593 | 6.785 |
| 92 | 1 | 9.568 | 6.720 |
| 982 | 1 | 28.640 | 27.409 |
| 12,571 | 1 | 574.189 | 434.746 |
| 12,845 | 1 | 523.164 | 428.233 |
| 13,557 | 1 | 528.108 | 440.955 |
