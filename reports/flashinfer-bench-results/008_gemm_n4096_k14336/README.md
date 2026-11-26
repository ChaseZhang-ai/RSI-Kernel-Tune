# 008_gemm_n4096_k14336

The direct-launch Triton baseline passed correctness and CUPTI timing on all 43 official workloads at **167.305 µs** geometric-mean latency on GB300. The source implements tiled FP16 GEMM against physical row-major B[N,K], accumulates in FP32 and stores the result to the provided FP16 C. A separate attempt on physical GPU 2 was discarded after the contention monitor detected another process. [Official workload traces](baseline-gb300.jsonl)

The accepted run used physical GB300 3, UUID `GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`. SMs were locked at 1500 MHz and the job sampled 1500/3996 MHz. Immediate memory clock locking was unsupported. The driver accepted the deferred 3996 MHz request, and 3996 MHz was the only supported memory clock reported. The evaluator was SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd`, with PyTorch 2.10.0+cu130, fbtriton 3.7.1 (`triton.__version__` reported 3.7.0+fb.beta), and CUPTI Python 13.5.0. [GPU record](baseline-gpu.json)

The exact baseline was embedded as the sole source of a DPS solution bundle and run through `sol-execbench <problem_dir> --solution solution.json --output results.jsonl`. The evaluator used the staged FlashInfer-Bench definition, all 43 UUIDs, default tolerances and random input specifications. This result is provisional GB300 evidence.

An initial warp-specialized TMA implementation with split-K for M=2–16 passed **43/43** official workloads at **53.073 µs** geometric-mean latency on physical GB300 2, UUID `GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`. Active clocks sampled at 1500/3996 MHz, under the same driver clock behavior as the baseline. [Official first TMA traces](first-tma-gb300.jsonl) · [GPU record](first-tma-gpu.json)
