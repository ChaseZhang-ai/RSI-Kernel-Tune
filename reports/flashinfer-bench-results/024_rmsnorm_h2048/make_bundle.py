"""Materialize the official 024 workloads and embed the exact DPS entry point."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()

    name = "024_rmsnorm_h2048"
    [row] = pq.read_table(args.parquet, filters=[("name", "=", name)]).to_pylist()
    workloads = json.loads(row.pop("workloads"))
    if sorted(item["axes"]["batch_size"] for item in workloads) != [1, 6, 34, 64, 79, 12383, 16254]:
        raise ValueError("Unexpected official workload axes")
    if len({item["uuid"] for item in workloads}) != 7:
        raise ValueError("Unexpected official workload IDs")
    if any(spec["type"] != "random" for item in workloads for spec in item["inputs"].values()):
        raise ValueError("Stage the trace input files before running the evaluator")
    definition = {
        key: json.loads(value) if key in {"axes", "inputs", "outputs"} else value
        for key, value in row.items() if value not in (None, "")
    }

    problem = args.stage / name
    problem.mkdir(parents=True, exist_ok=True)
    (problem / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
    (problem / "workload.jsonl").write_text(
        "".join(json.dumps(workload) + "\n" for workload in workloads)
    )
    exact_source = args.submission.read_bytes()
    bundle = {
        "name": "rmsnorm024-packed",
        "definition": name,
        "author": "RSI-Kernel-Tune",
        "spec": {
            "languages": ["triton"],
            "entry_point": "submission.py::run",
            "target_hardware": ["B200", "LOCAL"],
            "destination_passing_style": True,
            "dependencies": ["triton"],
        },
        "sources": [{"path": "submission.py", "content": exact_source.decode("utf-8")}],
    }
    (args.stage / "solution.json").write_text(json.dumps(bundle) + "\n")
    print(f"submission.py SHA-256 {hashlib.sha256(exact_source).hexdigest()}")


if __name__ == "__main__":
    main()
