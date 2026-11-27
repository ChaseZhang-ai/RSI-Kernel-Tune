# 008 GEMM follow-up

The revised archive entry passed **43/43** official workloads with CUPTI timing on both stock Triton **3.6.0** and **fbtriton 3.7.1** (reported as Triton 3.7.0+fb.beta). The evaluator measured **49.982 µs** and **50.758 µs** geometric means, respectively. The paired fbtriton repeat on physical GB300 0 measured **50.036 µs** for the released source and **50.058 µs** for the revised source across two trials. All 172 workload-arm measurements and 16 additional cancellation, pointer, stream and stride checks passed. Cold compile plus synchronized first calls across all workloads took **1.086 s** for the revised file. Peak PyTorch allocation was 1.212 GiB. [Run summary](record.json) · [Stock Triton evaluator trace](portable-stock36-gb300.jsonl) · [fbtriton evaluator trace](portable-fb-gb300.jsonl) · [Paired traces](paired-gb300.jsonl) · [Correctness checks](verification.json)

Stock Triton 3.6 rejected the archived TMA paths at `mbarrier.allocate_mbarrier`. Isolated compiles also identified its missing `nbytes_per_cta` helper and its required tensor-memory load layout. The revision explicitly allocates each CTA's barriers, computes each complete FP16 tile's byte count, uses the default unit barrier arrival, and supplies the explicit epilogue layout on the older API. All six fbtriton TMA variants had matching executable cubin text and shared-memory requirements under identical compile signatures. The runtime dispatch and its one- and two-launch sequences are unchanged. The Triton 3.5 probe exposed an older tensor-memory layout API and a packaged assembler that rejected the queried SM103 target. [Compiler evidence](portability.json) · [fbtriton cubin hashes](fb-cubin-text.json)

The revised source SHA-256 is `da009679c98dac2fcd039887936af33a697313021af00a5b6dc0ae009e3aed99`. Both evaluator bundles embed that exact file through `submission.py::run` in destination-passing mode. Comparison jobs used separate GPU-lock acquisitions, pinned the process by physical UUID, and timed each pair on the same physical GPU with 10 warmups and 50 cold-L2 CUPTI samples per trial. The revised source is ready for coordinator B200 validation on compatibility grounds. [Device and clock records](devices.json) · [Paired compile metadata](paired-metadata.json)

## Released B200 results and exact-case repeat

The staged B200 v1.1 summary identifies the archived SHA-256 `880169a983300e288e9a08676e9e6afbc56b36afbce2465e954ecb48e44c99dc`. It matched the tracked comparator byte for byte. Collection child **106077** timed **37/43** workloads. The independent standalone repeat **106243** timed **38/43**. The standalone summary supplies only the count, so the identities of its five missing records are unknown. The published collection rows mark six missing CUPTI activities and supply no build or numerical error for those rows. The public leader was June at **1.000000 SOL** in the 2026-11-27 04:17:52 UTC snapshot. [Released B200 evidence](baseline-b200.json) · [Public operator board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/217/B200)

The exact archived source passed **43/43** workloads through the local evaluator on physical GB300 1 at **50.101 µs** geometric mean. The focused repeat then checked 10 warmup outputs against the official tolerances on each of the six affected shapes and collected a diagnostic kernel sequence. All six passed all three independent cold-L2 timing trials, each with 10 warmups and 50 samples. The first focused run was discarded after a concurrent compute process was detected. The accepted run used physical GB300 3. Both devices sampled 1500/3996 MHz, queried 152 SMs and compiled for CUDA capability 10.3. [Exact local evaluator trace](released-gb300.jsonl) · [Focused timings and warmups](routes.jsonl) · [GPU records](devices.json)

| M | Released kernel sequence | Warmup checks | Timed samples | Released median (µs) | Best tested opposite split policy (µs) |
|---:|---|---:|---:|---:|---:|
| 32 | Matmul, FP32 merge | 10/10 | 150/150 | 32.080 | 35.792 single |
| 70 | Matmul, FP32 merge | 10/10 | 150/150 | 41.760 | 43.768 single |
| 176 | Matmul | 10/10 | 150/150 | 54.656 | 63.712 two-way split |
| 232 | Matmul | 10/10 | 150/150 | 54.960 | 65.096 two-way split |
| 256 | Matmul | 10/10 | 150/150 | 55.456 | 66.168 two-way split |
| 2379 | Persistent matmul | 10/10 | 150/150 | 185.808 | 277.561 two-way split |

The table uses the median of trial medians. Neighboring shapes with each of these dispatch configurations timed in the collection result. The records span both launch-sequence lengths. The focused local repeats did not reproduce any timing failures, leaving the origin of the B200 activity loss unresolved.

## Large-K schedule

At M=8, the released two-split route measured 30.848 µs. Removing the split at the same tile size took 32.640 µs. At M=70, removing the split at the same tile size took 54.424 µs, and a narrower direct tile took 43.768 µs. The split route increases the grid from 64 to 128 CTAs for these single-row-tile shapes. Each split writes an independent FP32 plane and the merge rounds the final result to FP16.

At M=2379, the released grid has 304 output tiles across 152 workers. It processes 224 K steps per tile with a four-stage K=64 input ring and two accumulator slots. Replacing this with one accumulator took 187.641 µs, and K=128 with two input stages took 242.153 µs. At M=8192 the released path measured 623.425 µs. The one-accumulator and two-split variants measured 630.746 and 917.443 µs, respectively. The screen retained the released schedule across all routes. [Paired schedule trials](routes.jsonl)
