# FlashInfer 020 GB300 verification

The provisional collection candidate passed **19/19** official workloads with a **409.05 µs** geometric mean of the per-workload median latencies. The exact source has SHA-256 `24a41b3ec2bfb6eed374419d6ccf08724e71ccf6c59d4cf600a29d7ca7edfea3`. [submission.py](../../../flashinfer-bench/020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/submission.py)

| GB300 run | 32-row direct baseline | Selected candidate | Device |
|---|---:|---:|---|
| Official evaluator | 434.34 µs | 409.05 µs | GPU 3 · `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f` |
| Paired comparison | 431.99 µs | 409.42 µs | GPU 3 · `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f` |
| Paired repeat with reversed initial order | 435.73 µs | 409.35 µs | GPU 2 · `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0` |

Paired latency decreased by **5.23%** and **6.05%**. The cold paired run measured **2.512 s** of cumulative Triton JIT compiler time and a **1.463 s** maximum first-call wall time for the selected artifact. Peak additional allocated memory for the candidate was **2.75 GiB** at the largest workload.

## Verification

- The official records contain workload IDs, axes, offsets, error statistics, and CUPTI latencies. SOL-ExecBench 1.0.2 evaluated the exact destination-passing bundle with PyTorch 2.10.0+cu130 and Triton 3.6.0. [Candidate results](gb300-official-sparse-native.json) · [Direct baseline results](gb300-official-fallback.json)
- Each paired comparison used 10 warmups and 50 timed iterations per variant per workload, cloned inputs, preallocated outputs, cold L2, and CUPTI first-to-last timing. The GPU was held under its host lock throughout each job and pinned by UUID. Both GB300 timing devices reported 152 SMs and CUDA capability 10.3, with 1500 MHz locked SM clocks and 3996 MHz memory. [Paired comparison](gb300-pair-2.json) · [Paired repeat](gb300-pair-3.json)
- Seven routing edge checks covered skew, exact group and expert cutoff ties, nonzero offsets, all-remote output, changed inputs and pointers, zero scaling, and both sparse and long dispatch. [Edge results](gb300-routing-edges.json)
- The rejected variants included a Triton build failure and numerical failures in FP8 residual conversion and wider GEMM tiles. [Initial failures](gb300-direct-initial.json) · [Corrected direct baseline](gb300-official-precise.json) · [Long-workload diagnostics](gb300-compact-diagnostic.json) · [First paired tile experiments](gb300-pair-1.json)

## Implementation

The sparse dispatch uses native FP8 Tensor Core instructions for both GEMMs. Longer batches use compact expert tiles with Triton's FP16 Tensor Core lowering. The official expert weights stay FP8 and the 128-element scales stay FP32 in both paths. The down projection consumes a three-term FP8 SwiGLU expansion and accumulates routing-weighted products in FP32 before writing BF16. The paired measurements favored the selected schedule over the wider fully native experiments. Offline SM100 compilation also succeeded for both GEMM paths and the supporting kernels.

The indirect GEMM and FP32 block-scale promotion follow the repository's ako4x fused routing reference. The FP8 residual expansion and grouped tile scheduling draw on the CompoundingKernel Explorer archive. [Repository reference](../../../kernel_library/kernels/ako4x-moe-fp8-block-scale-fused_routing_v2/kernel.py) · [CompoundingKernel source](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/README.md)

The evaluator run materialized the named row from the staged FlashInfer-Bench Parquet, preserving its JSON definition and all 19 workload IDs. The local trace root held the matching FlashInfer Trace 1.0 safetensors. The destination-passing bundle embedded the source as `submission.py::run`.

```bash
sol-execbench .artifacts/020/problem \
  --solution .artifacts/020/sparse-native/solution.json \
  --output .artifacts/020/sparse-native/results.jsonl
```

## B200 public board

The public v1.1 leader was **1.000000 SOL** at `2026-11-26T19:05:51Z`. This artifact is GB300-only, with B200 score, rank, and gap pending coordinator verification. [Public kernel 229 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/229/B200)
