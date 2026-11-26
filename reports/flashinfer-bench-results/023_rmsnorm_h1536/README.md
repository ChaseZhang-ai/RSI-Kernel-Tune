# 023_rmsnorm_h1536

The padded 2048-wide Triton direct-launch baseline passed all eight official workloads, including CUPTI timing, on GB300. The archive entry is provisional for the B200 collection. [Exact baseline source](triton-baseline.py) · [Evaluator traces](gb300-baseline.jsonl)

The SOL-ExecBench CLI 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`) evaluated the exact standalone Python file as a destination-passing solution with the official Parquet workload IDs and evaluator tolerances. It used 10 warmups and 50 timed iterations per workload with cold L2, cloned inputs, preallocated outputs, and CUPTI first-to-last activity. The run completed at 2026-11-26 18:39 UTC on physical GB300 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), at 1500 MHz locked SM and 3996 MHz memory. The device reports 152 SMs and capability 10.3. The evaluator used PyTorch 2.9.0+cu130 and Triton 3.5.0.

The initial attempt failed at build with the bundled PTX assembler (`Value 'sm_103a' is not defined for option 'gpu-name'`). The passing attempt used the installed CUDA 13.1 assembler through `TRITON_PTXAS_PATH`. Per-workload status, latency, and numerical error are in the evaluator traces. Cold first-launch timing is collected separately in the paired trials.

The public B200 v1.1 leader at 2026-11-26 18:37:21 UTC was rank 1 submission 104773 with 0.929768 SOL. [NVIDIA B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/232/B200)

## Local layout comparisons

Paired CUPTI screens retained the padded direct-launch baseline on each physical GPU. The Triton screen compared padded 2048-column and exact 3×512-column reductions with several warp counts. Its best exact layout timed at 14.752 and 17.104 µs for 11,949 and 14,521 rows. The packed CUDA screen tested 96- and 192-thread CTA layouts, 16- and 32-byte vector stores, and warp-per-row grids including persistent grids sized with a 148-SM proxy. The strongest native small-batch layouts took 2.656–2.784 µs. The native large layouts were slower than the exact Triton layout. [Per-trial data](gb300-screen.csv) · [Screen devices](gb300-screen-devices.json)

The archived CompoundingKernel implementation was adapted locally to write the supplied output. It passed all eight paired correctness and timing checks, with a 13.759 s first call covering its eager launcher compilation. It multiplies input by weight before applying the inverse RMS. The implementation with the requested FP32 multiplication order was developed independently using its work-distribution and prefetch ideas. [Attributed archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/023_rmsnorm_h1536)

The final screening round compared shared-memory reduction layouts, prefetch, programmatic dependent launch, CTA clusters, and persistent Gluon tiles. The native cluster retry reduced small-batch medians to 2.112–2.336 µs and the 539-row median to 3.008 µs. The retry configured the function's dynamic shared-memory limit before launch. The earlier cluster requests produced 34 `CUDA_ERROR_INVALID_VALUE` launch failures. The retry passed every attempted configuration. The direct two-row Triton tile remained faster than all tested persistent Gluon and CUDA tiles at 11,949 and 14,521 rows. [Screen timings and outcomes](gb300-screen.csv)
