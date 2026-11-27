"""Materialize the official 005 problem and embed an exact Python DPS source."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


NAME = "005_gemm_n256_k7168"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("parquet", type=Path)
parser.add_argument("submission", type=Path)
parser.add_argument("problem", type=Path)
parser.add_argument("solution", type=Path)
args = parser.parse_args()

[row] = pq.read_table(args.parquet, filters=[("name", "=", NAME)]).to_pylist()
workloads = json.loads(row.pop("workloads"))
definition = {
    key: json.loads(value) if key in {"axes", "inputs", "outputs"} else value
    for key, value in row.items()
    if value not in (None, "")
}
assert len({workload["uuid"] for workload in workloads}) == len(workloads) == 17
assert all(inp["type"] == "random" for workload in workloads for inp in workload["inputs"].values())

args.problem.mkdir(exist_ok=True, parents=True)
(args.problem / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
(args.problem / "workload.jsonl").write_text("".join(json.dumps(workload) + "\n" for workload in workloads))
bundle = {
    "name": NAME,
    "definition": NAME,
    "author": "RSI-Kernel-Tune",
    "spec": {
        "languages": ["triton"],
        "target_hardware": ["LOCAL", "B200"],
        "entry_point": "submission.py::run",
        "destination_passing_style": True,
    },
    "sources": [{"path": "submission.py", "content": args.submission.read_text()}],
}
args.solution.write_text(json.dumps(bundle) + "\n")
