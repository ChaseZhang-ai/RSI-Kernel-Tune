# FlashInfer 011 FP16 GEMM

The explicit TMA/MMA pipeline passed all 43 official workloads on GB300. It averaged **104.98 µs** in the paired evaluator run, compared with **117.21 µs** for the software-pipelined version. The initial direct-launch baseline averaged **187.44 µs** in its own paired run. The B200 jobs reported a compiler error and a missing CUPTI record, respectively. The current candidate is provisional. [Direct and tuned pair](gb300-tuned.json) · [Software and explicit pipeline pair](gb300-manual.json) · [Private job 105838](b200-105838.json) · [Private job 105844](b200-105844.json)

## Local verification

The final paired run used physical GB300 **0**, UUID `GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`, with **152 queried SMs** and CUDA capability **10.3**. Triton built for the active device and the persistent grid used its queried SM count. The SM clock was locked to 1500 MHz. The driver accepted a deferred memory-clock lock and reported 3996 MHz during measurement, its only supported memory clock. The shared driver stayed loaded.

The official Parquet row supplied 43 distinct workload UUIDs and random inputs for both operands, so no trace files were needed. The evaluator used its default tolerance of 0.01 absolute, 0.01 relative, and a 0.99 required matched ratio. All results were measured with 10 warmups and 50 cold-L2 CUPTI spans. The evaluator cloned inputs and shifted tensor addresses with preallocated destination outputs. The explicit pipeline took **2.44 s** for cold JIT compilation plus one direct call at each official shape. The exact collection file passed 43/43 again with a 105.45 µs mean using `sol-execbench . --solution solution.json --output results.jsonl`. A separate exact-value check ran each workload with two sets of changed inputs and addresses, checked every output column through column 28,672, and verified guards on both sides of `C`. [Evaluator and per-workload traces](gb300-manual.json) · [Exact bundle repeat](gb300-exact-repeat.json) · [Guard check](gb300-guard-check.json)

## Latency and resource tradeoffs

The 224 MiB cold weight read took **44.59 µs**, equivalent to **5.27 TB/s**. M=1 vector GEMV took 48.96 µs in the official evaluator. Weight-first MMA was faster for M=2–8 and stayed near 50 µs. M=32 used the same weight-first mapping with a 32-row activation tile. For larger M, grouped persistent scheduling reused weight tiles across M CTAs. Testing a weight-first CTA that covered up to all 256 M rows did improve weight reuse, but its transpose epilogue and launch waves erased the gain at the upper end of the range. [Cold read and small-M sweep](gb300-screen1.jsonl) · [Weight-first reuse sweep](gb300-screen4.jsonl)

| M | Software pipeline (µs) | Explicit pipeline (µs) | Explicit FP16 throughput (PFLOP/s) |
|---:|---:|---:|---:|
| 972 | 193.824 | 175.233 | 1.303 |
| 2053 | 420.401 | 348.657 | 1.383 |
| 2379 | 450.369 | 386.530 | 1.446 |
| 8192 | 1454.212 | 1220.531 | 1.576 |

The 128-wide N-tile kernel used 163,940 bytes of shared memory with four operand stages. The selected 256-wide N-tile variants used 213,092 or 229,476 bytes, depending on the epilogue segment width. Both widths provisioned 255 reported registers per thread with zero spills for the explicit schedule. The wider tile halves the number of A tile fetches for a fixed M band and leaves room for only one CTA per SM. At M=8192 it reduced median latency from 1955.88 µs for N=128 to 1216.76 µs for N=256 in the tile sweep. The timed call includes the epilogue and FP16 stores. The kernel uses no global scratch. The largest call has 736 MiB of FP16 inputs and output, and even 61 disjoint copies would consume under 44 GiB before allocator overhead. [Tile and staging sweep](gb300-screen3.jsonl)

## Sources

The weight-first mapping was informed by the CompoundingKernel archive. The persistent CTA schedule and operand staging draw on the repository's H100 GEMM, adapted to `[N,K]` FP16 weights. The explicit warp partitions and tensor-memory pipeline draw on Triton's Gluon tutorial. [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/011_gemm_n28672_k4096/README.md) · [H100 GEMM](../../../kernels/gemm/gemm_fp16/h100/kernel.py) · [Triton Gluon tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py) · [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)
