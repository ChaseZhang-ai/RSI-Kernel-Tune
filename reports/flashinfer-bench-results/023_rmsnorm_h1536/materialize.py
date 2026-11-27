"""Materialize the official Parquet row and bundle the exact DPS submission."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


NAME = "023_rmsnorm_h1536"
BATCH_SIZES = [1, 7, 18, 32, 64, 539, 11949, 14521]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("problem_dir", type=Path)
    args = parser.parse_args()

    [row] = pq.read_table(args.parquet, filters=[("name", "=", NAME)]).to_pylist()
    workloads = json.loads(row.pop("workloads"))
    definition = {key: value for key, value in row.items() if value not in (None, "")}
    for key in ("axes", "inputs", "outputs"):
        definition[key] = json.loads(definition[key])
    if (sorted(work["axes"]["batch_size"] for work in workloads) != BATCH_SIZES
            or len({work["uuid"] for work in workloads}) != len(BATCH_SIZES)):
        raise ValueError("The workload axes or IDs differ from the official eight-workload row")
    if any(spec["type"] != "random" for work in workloads for spec in work["inputs"].values()):
        raise ValueError("The staged row differs from the verified all-random workload row")

    solution = {
        "name": "rsi-023-h1536",
        "definition": NAME,
        "author": "RSI-Kernel-Tune",
        "spec": {
            "languages": ["triton"],
            "entry_point": "submission.py::run",
            "target_hardware": ["B200", "LOCAL"],
            "destination_passing_style": True,
            "dependencies": ["triton"],
        },
        "sources": [{"path": "submission.py", "content": args.submission.read_text()}],
    }

    dest = args.problem_dir
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "definition.json").write_text(json.dumps(definition, indent=2) + "\n")
    (dest / "workload.jsonl").write_text("".join(json.dumps(work) + "\n" for work in workloads))
    (dest / "solution.json").write_text(json.dumps(solution) + "\n")


if __name__ == "__main__":
    main()
