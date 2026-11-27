"""Materialize the official workload and the exact standalone DPS bundle."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    name = "025_rmsnorm_h4096"
    row, = pq.read_table(args.parquet, filters=[("name", "=", name)]).to_pylist()
    workloads = json.loads(row.pop("workloads"))
    row.pop("hf_id")
    for key in ("axes", "inputs", "outputs"):
        row[key] = json.loads(row[key])

    problem = args.output / "problem"
    problem.mkdir(parents=True, exist_ok=True)
    (problem / "definition.json").write_text(json.dumps(row))
    (problem / "workload.jsonl").write_text(
        "".join(json.dumps(workload) + "\n" for workload in workloads)
    )
    source = args.submission.read_bytes().decode("utf-8")
    bundle = {
        "name": "rsi-025-rmsnorm-h4096",
        "definition": name,
        "author": "RSI-Kernel-Tune",
        "spec": {
            "languages": ["triton"],
            "target_hardware": ["LOCAL", "B200"],
            "entry_point": "submission.py::run",
            "destination_passing_style": True,
        },
        "sources": [{"path": "submission.py", "content": source}],
    }
    (args.output / "solution.json").write_text(json.dumps(bundle))


if __name__ == "__main__":
    main()
