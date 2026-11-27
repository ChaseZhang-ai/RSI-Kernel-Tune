"""Pair candidates with the direct baseline using SOL-ExecBench's timing API."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from sol_execbench.core.bench.correctness import compute_error_stats, set_seed
from sol_execbench.core.bench.io import allocate_outputs, gen_inputs, load_safetensors
from sol_execbench.core.bench.timing import time_runnable
from sol_execbench.core.data import Definition, Workload


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fifths_guard(run, m):
    rows = torch.arange(m, device="cuda")
    cols = torch.arange(5120, device="cuda")
    k = torch.arange(2048, device="cuda")
    calls = []
    for epoch in (0, 1):
        # Integers are exactly representable here. The row selects one different
        # K position per call, and each fifth has its own disjoint value band.
        selected_k = (rows * 239 + 331 * epoch) % 2048
        a = torch.zeros((m, 2048), device="cuda", dtype=torch.float16)
        sign = 1 - 2 * ((rows + epoch) % 2)
        a[rows, selected_k] = sign.to(torch.float16)
        b = ((cols[:, None] // 1024) * 128
             + (cols[:, None] + 7 * k[None, :] + 5 * epoch) % 127).to(torch.float16)
        c = torch.full((m, 5120), float("nan"), device="cuda", dtype=torch.float16)
        expected = sign[:, None].to(torch.float16) * b[:, selected_k].T
        calls.append((a, b, c, expected))
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(calls[0][:3], calls[1][:3]))
    for a, b, c, expected in calls:
        run(a, b, c)
        if not torch.equal(c, expected):
            raise AssertionError("Fifths guard: incorrect value or unwritten destination")


def failure(exc):
    message = str(exc)
    if any(text in message.lower() for text in ("cupti", "kernel activit", "timing results")):
        return "TIMING_FAILURE"
    if (isinstance(exc, triton.CompilationError)
            or "out of resource" in message.lower() or "PassManager" in message):
        return "BUILD_FAILURE"
    if isinstance(exc, AssertionError):
        return "NUMERICAL_FAILURE"
    return "RUNTIME_ERROR"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("problem", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--screen", action="store_true")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--guard", action="store_true")
    p.add_argument("--m", nargs="+", type=int)
    args = p.parse_args()
    base = load(args.baseline, "gemm_baseline")
    candidate = load(args.candidate, "gemm_candidate")
    definition = Definition.model_validate_json((args.problem / "definition.json").read_text())
    workloads = [Workload.model_validate_json(line)
                 for line in (args.problem / "workload.jsonl").read_text().splitlines()]
    reference = {}
    exec(definition.reference, reference)
    set_seed(200)
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": props.name,
        "physical_index": int(os.environ["FLASHINFER_GPU_INDEX"]),
        "uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "capability": torch.cuda.get_device_capability(),
        "queried_sms": props.multi_processor_count,
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cupti_python": importlib.metadata.version("cupti-python"),
        "sol_execbench": importlib.metadata.version("sol-execbench"),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "seed": 200,
        "warmups": 10,
        "iterations": 50,
        "trials": args.trials,
    }
    cold_totals = {}
    with args.output.open("w") as out:
        for i, workload in enumerate(workloads):
            m = workload.axes["M"]
            if args.m and m not in args.m:
                continue
            safe = load_safetensors(definition, workload, [args.problem])
            inputs = gen_inputs(definition, workload, "cuda:0", safe)
            expected = reference["run"](*inputs)
            outputs = allocate_outputs(definition, definition.get_resolved_axes_values(workload.axes), "cuda:0")
            functions = [("baseline", base.run)]
            functions += list(candidate.variants(m)) if args.screen else [("candidate", candidate.run)]
            records = {label: {"workload_id": str(workload.uuid), "axes": workload.axes,
                               "implementation": label, "trials": []} for label, _ in functions}
            passed = []
            for label, fn in functions:
                record = records[label]
                try:
                    outputs[0].fill_(float("nan"))
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    compiled = fn(*inputs, *outputs)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    record["first_call_and_sync_s"] = elapsed
                    cold_totals[label] = cold_totals.get(label, 0) + elapsed
                    if hasattr(compiled, "metadata"):
                        record["kernel"] = compiled.metadata.name
                        record["registers"] = compiled.n_regs
                        record["shared_memory_bytes"] = compiled.metadata.shared
                        record["spills"] = compiled.n_spills
                    stats, bad = compute_error_stats(outputs[0], expected, workload.tolerance)
                    record["correctness"] = stats.model_dump()
                    record["status"] = "NUMERICAL_FAILURE" if bad else "PASSED"
                    if not bad:
                        if args.guard and label == "candidate":
                            fifths_guard(fn, m)
                            record["fifths_guard_calls"] = 2
                        passed.append((label, fn))
                except Exception as exc:
                    record.update(status=failure(exc), error=str(exc)[-600:])
            for trial in range(args.trials):
                ordered = passed if (i + trial) % 2 == 0 else reversed(passed)
                for label, fn in ordered:
                    record = records[label]
                    if record["status"] != "PASSED":
                        continue
                    try:
                        ns = time_runnable(fn, inputs, outputs, "cuda:0", warmup=10, rep=50,
                                           methodology="cupti", return_mode="all", seed=200)
                        assert len(ns) == 50
                        record["trials"].append({
                            "trial": trial + 1,
                            "mean_us": 1000 * statistics.mean(ns),
                            "median_us": 1000 * statistics.median(ns),
                        })
                    except Exception as exc:
                        record.update(status="TIMING_FAILURE", error=str(exc)[-600:])
            for record in records.values():
                if record["status"] == "PASSED":
                    record["latency_us"] = statistics.mean(t["mean_us"] for t in record["trials"])
                out.write(json.dumps(record) + "\n")
            out.flush()
            successful = sorted((r["latency_us"], label) for label, r in records.items()
                                if r["status"] == "PASSED")
            print(m, successful[:5], "failures",
                  [label for label, r in records.items() if r["status"] != "PASSED"], flush=True)
    metadata["cold_first_calls_and_sync_s"] = cold_totals
    metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
