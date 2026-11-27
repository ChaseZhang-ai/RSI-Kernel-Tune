"""Measure the cold first-call envelope of one standalone DPS artifact."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from measure import load


solution, workload_file, output = map(Path, sys.argv[1:])
cache = Path(os.environ["TRITON_CACHE_DIR"])
assert not cache.exists() or not any(cache.iterdir()), "Expected an empty Triton cache"
workloads = [json.loads(line) for line in workload_file.read_text().splitlines()]
candidate = load(solution, "standalone_submission")
props = torch.cuda.get_device_properties(0)
b = torch.zeros((5120, 2048), dtype=torch.float16, device="cuda")
samples = []
for workload in workloads:
    m = workload["axes"]["M"]
    a = torch.zeros((m, 2048), dtype=torch.float16, device="cuda")
    c = torch.empty((m, 5120), dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    candidate.run(a, b, c)
    torch.cuda.synchronize()
    samples.append({"id": workload["uuid"], "M": m, "first_call_and_sync_s": time.perf_counter() - start})
target = Counter()
for ptx in cache.glob("*/*.ptx"):
    for line in ptx.read_text().splitlines():
        if line.startswith(".target "):
            target[line.removeprefix(".target ")] += 1
            break
result = {
    "device": props.name,
    "physical_index": int(os.environ["FLASHINFER_GPU_INDEX"]),
    "uuid": os.environ["CUDA_VISIBLE_DEVICES"],
    "capability": torch.cuda.get_device_capability(),
    "queried_sms": props.multi_processor_count,
    "ptx_targets": target,
    "source_sha256": hashlib.sha256(solution.read_bytes()).hexdigest(),
    "first_call_and_sync_s": sum(x["first_call_and_sync_s"] for x in samples),
    "workloads": samples,
    "finished_utc": datetime.now(timezone.utc).isoformat(),
}
output.write_text(json.dumps(result, indent=2) + "\n")
print("Cold first calls and synchronization", result["first_call_and_sync_s"], "seconds", flush=True)
