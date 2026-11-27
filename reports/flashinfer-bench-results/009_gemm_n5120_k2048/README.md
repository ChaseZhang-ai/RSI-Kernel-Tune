# FlashInfer 009 FP16 GEMM

The B200 follow-up records the verified archived score and the revised candidate's full GB300 results. [Follow-up report](b200-followup.md)

The archive candidate passed **25/25** official workloads through the SOL-ExecBench CLI at **23.971 µs geomean**. Three paired trials measured **23.976 µs** versus **24.433 µs** for the preceding tuned dispatch. This is a provisional GB300 result. The coordinator has the exact source for B200 evaluation. [Candidate](../../../flashinfer-bench/009_gemm_n5120_k2048/submission.py) · [Exact CLI traces](gb300-v2-cli.jsonl) · [Paired traces](gb300-pair-v2.jsonl)

The direct-launch Triton baseline passed **25/25** official workloads with a **58.049 µs** geometric mean latency on GB300. It uses a 32 × 128 output tile and accumulates FP16 products in FP32 before writing the supplied FP16 destination. The candidate is provisional GB300 only. [Evaluator traces](gb300-baseline-cli.jsonl)

The evaluator used physical GB300 **0**, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`. The SM lock and measured SM clock were both 1500 MHz. The card reported 3996 MHz for every active memory clock sample, its only supported memory frequency. This driver accepted the deferred memory lock. The live driver was left loaded. [Clock record](gb300-baseline-clocks.json)

The exact destination-passing Python bundle ran locally with SOL-ExecBench 1.0.2 from commit `a9fa0804c793d438e70850c33fe34426e66d53dd`, PyTorch 2.10.0+cu130, fbtriton 3.7.1 (runtime version 3.7.0+fb.beta), CUPTI Python 13.5.0, seed 200, 10 warmups and 50 timed calls per workload. The Parquet row has 25 distinct UUIDs with random FP16 inputs. The evaluator applied its default 0.01 absolute and relative tolerances and 0.99 matched ratio. [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

The public B200 v1.1 leader was at the 1.000000 SOL cap. [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200)

## Tile and schedule screen

One 10-warmup, 50-iteration trial per configuration screened all 25 workload IDs on physical GB300 0. The direct baseline passed all 25 again. Every workload had a passing tuned configuration. The screen recorded 302 passes and 39 build failures. Shared-memory limits and automatic warp specialization lowering caused the build failures. [Screen results](gb300-tile-screen.csv) · [Device and trial metadata](gb300-tile-screen-metadata.json) · [Clock record](gb300-tile-screen-clocks.json)

At M=1 and M=2, four-column GEMV measured 7.324 and 8.812 µs. At M=4–8, the two-warp, 16-column MMA measured 9.670–9.741 µs. The 64-column descriptor tiles led the M=16–172 screen, and 128-column tiles led at M=289–952. The 256-column tile with segmented TMA stores and a 148-CTA persistent grid won at every M above 8,000. At M=12,251 it took 193.158 µs versus 225.976 µs for the best ordinary raster configuration in the screen.

## First tuned candidate

Both variants passed 25/25 in three paired trials on physical GB300 1, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with CUDA capability 10.3 and 152 queried SMs. The direct baseline measured 57.881 µs geomean and the tuned candidate measured 24.440 µs, a 2.37× speedup. The timed calls used cloned inputs, rotated device addresses, preallocated outputs, cold L2, and CUPTI first-to-last kernel spans. Each trial used 10 warmups and 50 timed calls. The candidate's cold first calls, including JIT and synchronization, took 2.519 s summed over the official list. Two full-output guards per workload passed exactly with different input contents and device addresses and distinct patterns across all five column bands. [Paired results](gb300-pair-v1.jsonl) · [Metadata](gb300-pair-v1-metadata.json) · [Clock record](gb300-pair-v1-clocks.json)

The refinement measured 12 representative shapes twice on physical GB300 0. At M=16 a 32-column tile with two warps measured 9.260 µs against 10.890 µs for the original 64-column tile. At M=12,251 the 152-worker persistent tile with four grouped M tiles and two 128-column TMA stores measured 191.751 µs. The same tile with one grouped M tile took 193.682 µs. The four-group pointer-store candidate emitted 128-bit vector stores and took longer than the segmented TMA epilogue. The final dispatch uses the queried device SM count. [Refinement](gb300-refine.csv) · [Metadata](gb300-refine-metadata.json) · [Clock record](gb300-refine-clocks.json)

The exact v1 bundle separately passed 25/25 through the SOL-ExecBench CLI on physical GB300 0 and measured 24.388 µs geomean. [CLI traces](gb300-v1-cli.jsonl) · [Clock record](gb300-v1-cli-clocks.json)

## Refined candidate

The exact v1 and v2 artifacts each passed 25/25 in three paired trials on physical GB300 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, with 152 queried SMs, CUDA capability 10.3, and active clock samples at 1500/3996 MHz. V2 brought the geometric mean from **24.433 µs to 23.976 µs**, a 1.9% latency reduction. It changed the M=16–32 tiles to 32 columns and covered M=93 and M=128 with one 128-row tile per 64 columns. Two changed-pointer and changed-content full-width checks passed per workload. The paired job used one initially empty JIT cache for both arms, and the summed synchronized first calls took 4.611 s across both arms. [Paired results](gb300-pair-v2.jsonl) · [Metadata](gb300-pair-v2-metadata.json) · [Clock record](gb300-pair-v2-clocks.json)

## Exact bundle verification

The CLI used physical GB300 **3**, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, with **152 queried SMs**, CUDA capability **10.3**, and active 1500/3996 MHz clock samples. The SM lock succeeded. The driver accepted the deferred memory setting and 3996 MHz was the card's only reported supported memory frequency. The same device compiled 14 Triton specializations for `sm_103a`. With an empty Triton cache the exact standalone source took **3.862 s** summed over the 25 first calls, including kernel execution and synchronization. The largest call uses about 243 MiB for A, B and C and the implementation uses no global scratch. The three-trial paired job peaked at 2.27 GiB of CUDA allocations. [Clock record](gb300-v2-cli-clocks.json) · [Cold compilation](gb300-cold-v2.json)

The bundle embeds `submission.py::run` in destination-passing mode. The exact source SHA-256 is `124ae55fb8697dcfb2c167e33300ba54f02d2dde5f982308f8f253613652759d`. The CLI exercised the complete official workload list and returned a timing for every passing UUID. All three trials of each final paired arm likewise recorded the expected CUPTI activity.

The public B200 v1.1 operator leader was at **1.000000 SOL** at **2026-11-26 18:57:42 UTC**. The candidate has GB300 verification. B200 SOL, public rank and signed gap remain blank. [Public board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/218/B200) · [Public board check](public-b200-v1.1.json)

## Large-shape profile

At M=12,251, grouping four M tiles per band reduced L2 read sectors from 73.74 million to 59.87 million relative to the single-M-tile band. Both schedules read about 71.2 MB from DRAM. The four-tile schedule balances reuse of the same A rows across N tiles with B reuse across M tiles. The counters include both operands. In the separate paired timing, segmented TMA stores took 191.751 µs, while direct `st.global.v4.b32` vector stores took 198.495 µs. [Nsight Compute counters and PTX store forms](gb300-ncu-m12251.json) · [Profile clocks](gb300-profile-clocks.json)

The weight-first mapping follows the public CompoundingKernel Explorer archive by QianQu and collaborators. The resident CTA schedule and staged accumulation draw on this repository's H100 GEMM, adapted to FP16 and K-contiguous `B[N,K]`. [CompoundingKernel source](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/009_gemm_n5120_k2048) · [H100 GEMM](../../../kernels/gemm/gemm_fp16/h100/kernel.py)

## Reproduce

Materialize the staged Parquet row from the repository root using the evaluator environment.

```bash
op=009_gemm_n5120_k2048
report="reports/flashinfer-bench-results/$op"
problem=".artifacts/009/repro"
candidate="flashinfer-bench/$op/submission.py"
python "$report/materialize.py" "$candidate" "$problem"
```

Under the selected physical GPU's shared lock, set `CUDA_VISIBLE_DEVICES` to its UUID and `FLASHINFER_GPU_INDEX` to its physical index. Use a fresh `TRITON_CACHE_DIR` for cold measurements.

```bash
sol-execbench "$problem" --solution "$problem/solution.json" \
  --config "$problem/config.json" --output "$problem/results.jsonl"
python "$report/measure.py" "$problem" "$report/baseline.py" "$candidate" \
  "$problem/paired.jsonl" --guard
```
