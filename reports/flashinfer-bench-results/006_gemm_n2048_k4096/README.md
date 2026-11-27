# 006_gemm_n2048_k4096

The November 27 revision and B200 diagnostics are documented in the follow-up record. [Follow-up](followup/README.md)

## November 26 grouped TMA candidate

The November 26 archive candidate passed **29/29** official workloads with CUPTI timing at a **25.472 µs** geometric mean on GB300. Two paired trials gave **25.246** and **25.116 µs**, a **1.777×** geometric-mean speedup over the direct-launch seed. The exact archive file had **2.905 s** of synchronized first-call time across all 29 workloads from an empty Triton cache. That includes JIT compilation and one launch per workload. This candidate is provisional GB300 only. [Official traces](gb300-grouped-tma.jsonl) · [Paired results](gb300-paired.json) · [Cold compile and correctness](gb300-cold.json)

The candidate uses direct FP32 vector reductions through M=4 and tensor-core split-K with an FP32 merge through M=16. It uses an ordinary TMA/MMA launch through M=192, followed by persistent CTAs. Large tiles have separate TMA, MMA and store partitions, two accumulator slots, and a 32-column FP16 TMA epilogue. The input descriptors use the physical `B[N,K]` layout and multiply by its logical transpose. [H100 staging and scheduling](https://github.com/ChaseZhang-ai/RSI-Kernel-Tune/blob/main/kernels/gemm/gemm_fp16/h100/kernel.py) · [CompoundingKernel Explorer](https://huggingface.co/spaces/QianQu/CompoundingKernel-Explorer/tree/main/kernels/FlashInfer/006_gemm_n2048_k4096)

## Local measurements

| Candidate | Official passes | Official geomean (µs) | Same-GPU paired geomeans (µs) |
|---|---:|---:|---:|
| Direct Triton seed | 29/29 | 44.016 | 44.714, 44.775 |
| First explicit TMA dispatch | 29/29 | 28.544 | 25.676, 25.343 |
| Grouped TMA archive candidate | 29/29 | 25.472 | 25.246, 25.116 |

The seed measurement used the repository's `_tri_mm` with its non-split configurations and a DPS wrapper. The original prototype includes a runtime selector with vendor-library routes. The paired runs used the same compiler for all three candidates. [Seed traces](gb300-seed.jsonl) · [First TMA traces](gb300-explicit-tma.jsonl)

M=2 took 7.168 and 7.200 µs with one direct launch versus 9.424 and 9.392 µs with split-K. Ordinary TMA was near 13.6 µs from M=34 through M=172, and the 128×128 persistent tile was near 20.5 µs at M=952. The first candidate's two-launch route at M=17–32 had lower medians in the paired trials. The single-launch TMA route gave consistent 13.5–13.6 µs medians across these sizes, including M=25 where the first candidate's earlier CLI run measured 30.912 µs. The selected dispatch keeps that single-launch route. [Screening results](gb300-screen.csv) · [Paired results](gb300-paired.json)

The large-M screen compared equivalent 128×256×64 MMA pipelines with four TMA stages and group size eight. All values are median microseconds from the same physical GPU.

| M | Ordinary, 64-column epilogue | Persistent, 64-column epilogue | Persistent, 32-column epilogue |
|---:|---:|---:|---:|
| 11,938 | 155.508 | 134.500 | 133.876 |
| 14,915 | 204.838 | 180.260 | 178.789 |
| 16,294 | 212.197 | 183.509 | 181.333 |

Exploratory failures included Gluon compilation on stock Triton 3.6, a distributed-layout lowering error in the cluster split-K path, shared memory limits in wider tiles, and automatic warp-specialization partition lowering. The screen records distinguish build and launch outcomes from numerical and CUPTI timing outcomes. One full grouped-TMA CLI trial was discarded after the GPU monitor observed a foreign CUDA context. The official traces above are from the accepted rerun. [Screening results](gb300-screen.csv) · [Run metadata](record.json)

## Verification and reproduction

The official CLI run used GB300 physical GPU 2 (`GPU-71efe229-bc3a-a7fe-813c-db46c77eb9a0`), the paired runs used GPU 0 (`GPU-abbc4299-8500-54d7-2a37-3ccac762f30d`), and the cold compile check used GPU 3 (`GPU-6673a73f-3dc7-93b3-b06a-88ee44cbb25f`). Each device reported 152 SMs and CUDA capability 10.3. The Triton JIT emitted `sm_103a` targets on these devices. The 1500 MHz SM lock was applied. The driver accepted a deferred 3996 MHz memory lock and sampled 3996 MHz throughout each run, its sole supported memory clock. The shared flock covered correctness and timing, and each process saw just its GPU UUID. [Run metadata](record.json)

SOL-ExecBench 1.0.2 at `a9fa0804c793d438e70850c33fe34426e66d53dd` used PyTorch 2.10.0+cu130, fbtriton 3.7.1 (runtime `3.7.0+fb.beta`), and CUPTI Python 13.5.0. It used 10 warmups, 50 timed calls, cold L2, cloned and pointer-shifted inputs, preallocated destinations, and first-to-last CUPTI activity. The paired measurements include both launches on the split-K path. Peak PyTorch allocation across the cold and paired checks was 1.224 GiB. [NVIDIA evaluator](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)

All 29 original workload IDs and M axes were preserved. The staged row specifies random tensors for every input, so it needs no trace tensor files. The evaluator resolved its default tolerance from the workload records. A separate process checked all 29 shapes with edited input contents and relocated input/output buffers, and eight cases also checked an unaligned, row-strided fallback with a guarded destination. [Official traces](gb300-grouped-tma.jsonl) · [Pointer and layout checks](gb300-cold.json)

The archive file SHA-256 is `fa2b28aa4d048843606b1928ea511006a7e4bc471e88a9e47e81d12544bf5602`. Use the evaluator and grader-compatible Triton build above. Run the evaluator with a fresh `TRITON_CACHE_DIR` and the selected GPU UUID while holding the shared host lock.

```bash
python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

name = "006_gemm_n2048_k4096"
parquet = Path(".artifacts/sol-data/FlashInfer-Bench.parquet")
row, = pq.read_table(parquet, filters=[("name", "=", name)]).to_pylist()
workloads = json.loads(row.pop("workloads"))
for key in ("axes", "inputs", "outputs"):
    row[key] = json.loads(row[key])
if not row["hf_id"]:
    row.pop("hf_id")
assert len(workloads) == 29
problem = Path(".artifacts/gemm006/problem")
problem.mkdir(parents=True, exist_ok=True)
(problem / "definition.json").write_text(json.dumps(row))
(problem / "workload.jsonl").write_text(
    "".join(json.dumps(w) + "\n" for w in workloads)
)
source = Path("flashinfer-bench") / name / "submission.py"
bundle = {
    "name": "006-grouped-tma-dps",
    "definition": name,
    "author": "RSI-Kernel-Tune",
    "spec": {
        "languages": ["triton"],
        "target_hardware": ["LOCAL", "B200"],
        "entry_point": "submission.py::run",
        "destination_passing_style": True,
    },
    "sources": [{"path": "submission.py", "content": source.read_text()}],
}
(problem / "solution.json").write_text(json.dumps(bundle))
PY
sol-execbench .artifacts/gemm006/problem --solution .artifacts/gemm006/problem/solution.json --output .artifacts/gemm006/problem/results.jsonl
```

The public v1.1 B200 board showed leader submission `102864` at **1.000000 SOL** at `2026-11-26T18:57:03+00:00`. B200 verification is with the coordinator. [NVIDIA kernel 215 board](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/kernel/215/B200)
