# FlashInfer 011 FP16 GEMM

The selected explicit TMA/MMA dispatch passed all 43 official workloads on GB300. Its paired mean was **101.95 µs**, compared with **104.72 µs** for the earlier explicit dispatch on the same GPU. The software-pipelined version averaged **117.21 µs** in the preceding pair. The initial direct-launch baseline averaged **187.44 µs** in its own paired run. The current candidate is provisional. [Selected dispatch pair](gb300-staged.json) · [Software and explicit pipeline pair](gb300-manual.json) · [Direct and tuned pair](gb300-tuned.json) · [Private job 105838](b200-105838.json) · [Private job 105844](b200-105844.json)

## Local verification

The final paired run used physical GB300 **3**, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`, with **152 queried SMs** and CUDA capability **10.3**. Triton built for the active device and the persistent grid used its queried SM count. The SM clock was locked to 1500 MHz. The driver accepted a deferred memory-clock lock and reported 3996 MHz during measurement, its only supported memory clock. The shared driver stayed loaded.

The official Parquet row supplied 43 distinct workload UUIDs and random inputs for both operands, so no trace files were needed. The evaluator used its default tolerance of 0.01 absolute, 0.01 relative, and a 0.99 required matched ratio. All results were measured with 10 warmups and 50 cold-L2 CUPTI spans. The evaluator cloned inputs and shifted tensor addresses with preallocated destination outputs. The selected dispatch took **2.82 s** for cold JIT compilation plus one direct call at each official shape. The earlier explicit dispatch also passed the standalone evaluation and verified every column with exact values, changed addresses and guard regions. [Selected dispatch traces](gb300-staged.json) · [Earlier exact bundle repeat](gb300-exact-repeat.json) · [Earlier guard check](gb300-guard-check.json)

## Latency and resource tradeoffs

The 224 MiB cold weight read took **44.59 µs**, equivalent to **5.27 TB/s**. M=1 vector GEMV took 48.96 µs in the official evaluator. Weight-first MMA was faster for M=2–8 and stayed near 50 µs. M=32 used the same weight-first mapping with a 32-row activation tile. For larger M, grouped persistent scheduling reused weight tiles across M CTAs. M=136–256 switched to N=128 with six stages, which cut the M=256 paired median from 73.25 to 66.54 µs. [K tile and staging sweep](gb300-screen6.jsonl) Testing a weight-first CTA that covered up to all 256 M rows did improve weight reuse, but its transpose epilogue and launch waves erased the gain at the upper end of the range. [Cold read and small-M sweep](gb300-screen1.jsonl) · [Weight-first reuse sweep](gb300-screen4.jsonl)

| M | Earlier explicit pipeline (µs) | Selected dispatch (µs) | Selected FP16 throughput (PFLOP/s) |
|---:|---:|---:|---:|
| 972 | 173.331 | 173.299 | 1.317 |
| 2053 | 345.895 | 344.087 | 1.401 |
| 2379 | 386.471 | 384.472 | 1.453 |
| 8192 | 1217.000 | 1201.082 | 1.602 |

The 128-wide N tile used 163,940 bytes of shared memory with four operand stages. Its selected six-stage configuration for M=136–256 used 213,124 bytes. The selected 256-wide N tiles used 204,900 or 213,092 bytes, depending on the epilogue segment width. Both widths provisioned 255 reported registers per thread with zero spills for the explicit schedule. The wider tile halves the number of A tile fetches for a fixed M band and leaves room for only one CTA per SM. At M=8192 it reduced median latency from 1955.88 µs for N=128 to 1216.76 µs for N=256 in the tile sweep. The timed call includes the epilogue and FP16 stores. The kernel uses no global scratch. The largest call has 736 MiB of FP16 inputs and output, and even 61 disjoint copies would consume under 44 GiB before allocator overhead. [Tile and staging sweep](gb300-screen3.jsonl)

The two-CTA Triton configurations failed during MLIR lowering on the local stack. A separate compatibility probe with Triton 3.5.0 on GB300 also stopped at LLVM instruction selection for `tcgen05.wait.ld`. The main local evaluator and all reported explicit-pipeline latency measurements used Triton 3.6.0. [Two-CTA sweep](gb300-screen5.jsonl)

## Sources

The weight-first mapping was informed by the CompoundingKernel archive. The persistent CTA schedule and operand staging draw on the repository's H100 GEMM, adapted to `[N,K]` FP16 weights. The explicit warp partitions and tensor-memory pipeline draw on Triton's Gluon tutorial. [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/011_gemm_n28672_k4096/README.md) · [H100 GEMM](../../../kernels/gemm/gemm_fp16/h100/kernel.py) · [Triton Gluon tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py) · [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)
