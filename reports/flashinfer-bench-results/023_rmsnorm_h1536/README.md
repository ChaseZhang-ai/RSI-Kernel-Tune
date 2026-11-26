# 023_rmsnorm_h1536

The provisional GB300 candidate passed all eight official workloads in the paired SOL-ExecBench correctness and CUPTI timing harness. It was **1.280× faster** than the padded direct-launch Triton baseline by geometric mean on the same GPU. The exact-file SOL-ExecBench CLI run is queued. [Candidate](../../../flashinfer-bench/023_rmsnorm_h1536/submission.py) · [Paired traces](gb300-paired.jsonl)

## Selected candidate

The small-batch CUDA path uses coalesced 16-byte loads and stores, FP32 packed arithmetic, and exact-width reductions over 96- and 192-thread groups. The 539-row path handles two rows per CTA with separate barriers. The large-batch Triton path handles two rows per CTA and reduces three 512-column segments with no column padding. Offline Triton compilation emitted a 16-byte vector store for both rows in each of the three segments. The standalone file uses the supplied destination and the inputs' current values on every call.

The paired run finished at 2026-11-26 18:56 UTC on **physical GB300 2**, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`, with **152 queried SMs**, CUDA capability **10.3**, and **1500 MHz SM / 3996 MHz memory**. The SM clock was locked. The configured memory clock and sole supported memory clock were 3996 MHz. The process used the device UUID for CUDA visibility while holding the shared GPU lock.

The paired harness used the default workload tolerances from SOL-ExecBench 1.0.2 commit `a9fa0804c793d438e70850c33fe34426e66d53dd`. Each side used 10 warmups and 50 first-to-last CUPTI activity samples with cold L2, cloned inputs and pointer shifting, and preallocated outputs. The runtime used PyTorch 2.9.0+cu130, Triton 3.5.0, CUDA bindings 13.1.1, CUDA 13.1 PTX assembler, NVRTC 13.0, and CUPTI Python 13.5.0.

| Batch | Official workload ID | Baseline µs | Candidate µs | Candidate status |
| ---: | --- | ---: | ---: | --- |
| 1 | `88f9970d-0b80-4ccf-9b9a-7790fd916b48` | 4.0160 | 2.5920 | Passed |
| 7 | `a31ba22f-b640-4013-a369-daff2137fbe5` | 3.4080 | 2.4960 | Passed |
| 18 | `6661b4a5-3293-462c-9690-b1314d5adae6` | 3.4560 | 2.4960 | Passed |
| 32 | `40f74bf7-a557-494c-b93d-8d863e6319f2` | 3.3285 | 2.6560 | Passed |
| 64 | `c2c0808b-f368-434a-bbd7-09c79cec998b` | 3.3765 | 2.5600 | Passed |
| 539 | `487a514f-e33b-4247-a42a-4abe569fec45` | 3.5840 | 3.1360 | Passed |
| 11,949 | `a4b033fd-c564-4bef-b4bd-b6c78174270a` | 16.6720 | 14.8325 | Passed |
| 14,521 | `1a428793-51ec-4eba-9d7c-9b7032d059eb` | 19.6160 | 16.9770 | Passed |

The synthetic contract check used separate small and normal magnitudes for each segment, changed both input values in place between calls, and repeated the test with new unaligned input and output pointers. All eight workloads passed, with at most one BF16 ULP of error against the FP32 reference, and the inputs were preserved. A separate strided input, weight, and destination check passed. [Contract check](gb300-contract.jsonl)

Uncached NVRTC compilation and cubin extraction for the exact native source took 1.577 s with `CUDA_CACHE_DISABLE=1`. The large-batch Triton specialization took 0.245 s including first launch with an empty Triton cache. The unaligned fallback took 0.647 s including first launch. Static compilation of the exact native CUDA and large Triton paths also succeeded for `sm_100`.

The candidate SHA-256 is `4d294ae5cded66441711e9a517b2e9e897819d61af2933a264f691b725c55639`. The public v1.1 B200 leader was submission 104773 at 0.929768 SOL at **2026-11-26 18:56:37 UTC**. B200 evaluation is pending with the coordinator. [NVIDIA B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/232/B200)

## Padded Triton baseline

The padded 2048-wide Triton direct-launch baseline passed all eight official workloads, including CUPTI timing, on GB300. [Exact baseline source](triton-baseline.py) · [Evaluator traces](gb300-baseline.jsonl)

The SOL-ExecBench CLI 1.0.2 (`a9fa0804c793d438e70850c33fe34426e66d53dd`) evaluated the exact standalone Python file as a destination-passing solution with the official Parquet workload IDs and evaluator tolerances. It used 10 warmups and 50 timed iterations per workload with cold L2, cloned inputs, preallocated outputs, and CUPTI first-to-last activity. The run completed at 2026-11-26 18:39 UTC on physical GB300 1 (`GPU-87e04b70-e9c6-0c59-7810-a2a4a6af4f12`), at 1500 MHz locked SM and 3996 MHz memory. The device reports 152 SMs and capability 10.3. The evaluator used PyTorch 2.9.0+cu130 and Triton 3.5.0.

The initial attempt failed at build with the bundled PTX assembler (`Value 'sm_103a' is not defined for option 'gpu-name'`). The passing attempt used the installed CUDA 13.1 assembler through `TRITON_PTXAS_PATH`. Per-workload status, latency, and numerical error are in the evaluator traces. The padded baseline's fresh-cache first call took 0.661 s in the final paired run.

The public B200 v1.1 leader at 2026-11-26 18:37:21 UTC was rank 1 submission 104773 with 0.929768 SOL. [NVIDIA B200 operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/232/B200)

## Local layout comparisons

Paired CUPTI screens retained the padded direct-launch baseline on each physical GPU. The Triton screen compared padded 2048-column and exact 3×512-column reductions with several warp counts. Its best exact layout timed at 14.752 and 17.104 µs for 11,949 and 14,521 rows. The packed CUDA screen tested 96- and 192-thread CTA layouts, 16- and 32-byte vector stores, and warp-per-row grids including persistent grids sized with a 148-SM proxy. The strongest native small-batch layouts took 2.656–2.784 µs. The native large layouts were slower than the exact Triton layout. [Per-trial data](gb300-screen.csv) · [Screen devices](gb300-screen-devices.json)

The archived CompoundingKernel implementation was adapted locally to write the supplied output. It passed all eight paired correctness and timing checks, with a 13.759 s first call covering its eager launcher compilation. It multiplies input by weight before applying the inverse RMS. The implementation with the requested FP32 multiplication order was developed independently using its work-distribution and prefetch ideas. [Attributed archive](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/023_rmsnorm_h1536)

The final screening round compared shared-memory reduction layouts, prefetch, programmatic dependent launch, CTA clusters, and persistent Gluon tiles. The native cluster retry reduced small-batch medians to 2.112–2.336 µs and the 539-row median to 3.008 µs. The retry configured the function's dynamic shared-memory limit before launch. The earlier cluster requests produced 34 `CUDA_ERROR_INVALID_VALUE` launch failures. The retry passed every attempted configuration. The direct two-row Triton tile remained faster than all tested persistent Gluon and CUDA tiles at 11,949 and 14,521 rows. [Screen timings and outcomes](gb300-screen.csv)
