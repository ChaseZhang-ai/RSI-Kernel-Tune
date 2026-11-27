# FlashInfer 011 FP16 GEMM

The selected dispatch combines GEMV, pointer-based small-M MMA and persistent TMA/MMA. It passed all 43 official workloads on GB300 and averaged **101.35 µs**, compared with **101.65 µs** for the preceding dispatch on the same GPU. The direct-launch baseline averaged **187.44 µs** in its paired run. The current candidate is provisional. The two preceding staged B200 trials reported missing CUPTI activity at M=4 and M=7, then at M=16 on retry. [Selected dispatch pair](gb300-pointer.json) · [Direct and tuned pair](gb300-tuned.json) · [Staged B200 job 105895](b200-105895.json) · [Staged B200 retry 105899](b200-105899.json)

## Local verification

The final paired run used physical GB300 **1**, UUID `GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`, with **152 queried SMs** and CUDA capability **10.3**. Triton built for the active device and the persistent grid used its queried SM count. The SM clock was locked to 1500 MHz. The driver accepted a deferred memory-clock lock and reported 3996 MHz during measurement, its only supported memory clock. The shared driver stayed loaded.

The official Parquet row supplied 43 distinct workload UUIDs and random inputs for both operands, so no trace files were needed. The evaluator used its default tolerance of 0.01 absolute, 0.01 relative, and a 0.99 required matched ratio. All results were measured with 10 warmups and 50 cold-L2 CUPTI spans. The evaluator cloned inputs and shifted tensor addresses with preallocated destination outputs. The selected dispatch took **3.04 s** for cold JIT compilation plus one direct call at each official shape. The exact selected file passed 43/43 on the standalone evaluator again on physical GB300 0, averaging 101.83 µs. Its guard check verified all 28,672 columns with exact values on two changed inputs and addresses per workload. [Selected dispatch traces](gb300-pointer.json) · [Exact repeat and guard check](gb300-pointer-repeat.json)

## Latency and resource tradeoffs

The 224 MiB cold weight read took **44.59 µs**, equivalent to **5.27 TB/s**. M=1 vector GEMV took 49.33 µs in the selected official run. Weight-first TMA/MMA stayed near 50 µs for M=2–8 in the initial sweep. A later pointer-based MMA tile with two warps and four K stages covered M=2–16 at 47.66–48.14 µs in the paired evaluator. M=24–64 still uses weight-first MMA. [Pointer-based MMA sweep](gb300-screen7.jsonl) For larger M, grouped persistent scheduling reused weight tiles across M CTAs. M=136–256 switched to N=128 with six stages, which cut the M=256 paired median from 73.25 to 66.54 µs. [K tile and staging sweep](gb300-screen6.jsonl) Testing a weight-first CTA that covered up to all 256 M rows did improve weight reuse, but its transpose epilogue and launch waves erased the gain at the upper end of the range. [Cold read and small-M sweep](gb300-screen1.jsonl) · [Weight-first reuse sweep](gb300-screen4.jsonl)

| M | Staged dispatch (µs) | Selected dispatch (µs) | Selected FP16 throughput (PFLOP/s) |
|---:|---:|---:|---:|
| 972 | 172.894 | 173.358 | 1.317 |
| 2053 | 343.181 | 342.989 | 1.406 |
| 2379 | 376.397 | 377.437 | 1.480 |
| 8192 | 1199.670 | 1198.612 | 1.605 |

The 128-wide N tile used 163,940 bytes of shared memory with four operand stages. Its selected six-stage configuration for M=136–256 used 213,124 bytes. The selected 256-wide N tiles used 204,900 or 213,092 bytes, depending on the epilogue segment width. Both widths provisioned 255 reported registers per thread with zero spills for the explicit schedule. The wider tile halves the number of A tile fetches for a fixed M band and leaves room for only one CTA per SM. At M=8192 it reduced median latency from 1955.88 µs for N=128 to 1216.76 µs for N=256 in the tile sweep. The timed call includes the epilogue and FP16 stores. The kernel uses no global scratch. The largest call has 736 MiB of FP16 inputs and output, and even 61 disjoint copies would consume under 44 GiB before allocator overhead. [Tile and staging sweep](gb300-screen3.jsonl)

The two-CTA Triton configurations failed during MLIR lowering on the local stack. A separate compatibility probe with Triton 3.5.0 on GB300 also stopped at LLVM instruction selection for `tcgen05.wait.ld`. The main local evaluator and all reported explicit-pipeline latency measurements used Triton 3.6.0. [Two-CTA sweep](gb300-screen5.jsonl)

The automatic warp-specialized B200 trial passed 14/43 with 29 build failures. The software-pipelined trial passed 42/43 with one missing CUPTI record. [Private job 105838](b200-105838.json) · [Private job 105844](b200-105844.json)

## Sources

The weight-first mapping was informed by the CompoundingKernel archive. The persistent CTA schedule and operand staging draw on the repository's H100 GEMM, adapted to `[N,K]` FP16 weights. The explicit warp partitions and tensor-memory pipeline draw on Triton's Gluon tutorial. [CompoundingKernel archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/resolve/main/kernels/FlashInfer/011_gemm_n28672_k4096/README.md) · [H100 GEMM](../../../kernels/gemm/gemm_fp16/h100/kernel.py) · [Triton Gluon tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py) · [Official evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)
