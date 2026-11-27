import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


NAME = "009_gemm_n5120_k2048"
MS = {1, 2, 4, 5, 6, 8, 16, 17, 25, 32, 34, 63, 64, 93, 128, 172,
      289, 492, 952, 8828, 11006, 12251, 12853, 14915, 16294}

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("problem", type=Path)
parser.add_argument("--parquet", type=Path, default=Path(".artifacts/sol-data/FlashInfer-Bench.parquet"))
args = parser.parse_args()
args.problem.mkdir(parents=True, exist_ok=True)
row, = pq.read_table(args.parquet, filters=[("name", "=", NAME)]).to_pylist()
workloads = json.loads(row.pop("workloads"))
for field in ("axes", "inputs", "outputs"):
    row[field] = json.loads(row[field])
row["hf_id"] = row["hf_id"] or None
assert len(workloads) == len({w["uuid"] for w in workloads}) == 25
assert {w["axes"]["M"] for w in workloads} == MS
(args.problem / "definition.json").write_text(json.dumps(row, indent=2) + "\n")
(args.problem / "workload.jsonl").write_text("".join(json.dumps(w) + "\n" for w in workloads))
source = args.source.read_bytes()
bundle = {
    "name": f"{NAME}-{hashlib.sha256(source).hexdigest()[:12]}",
    "definition": NAME,
    "author": "RSI Kernel Tune",
    "spec": {
        "languages": ["triton"],
        "target_hardware": ["LOCAL", "B200"],
        "entry_point": "submission.py::run",
        "destination_passing_style": True,
    },
    "sources": [{"path": "submission.py", "content": source.decode()}],
}
(args.problem / "solution.json").write_text(json.dumps(bundle, indent=2) + "\n")
(args.problem / "config.json").write_text(json.dumps({
    "warmup_runs": 10,
    "iterations": 50,
    "benchmark_reference": True,
    "seed": 200,
}) + "\n")
print("Input types:", sorted({v["type"] for w in workloads for v in w["inputs"].values()}))
print("Tolerances:", sorted({json.dumps(w.get("tolerance"), sort_keys=True) for w in workloads}))
print("Submission SHA256:", hashlib.sha256(source).hexdigest())
