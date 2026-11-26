#!/usr/bin/env python3
"""tensor: dense tensor throughput -- large square products in bf16, fp16 and fp8.

Standalone (``python3 tensor.py``).  A large square product is the cheapest way to reach a
GPU's tensor-core peak: the operands stay small relative to the arithmetic, so the reading
is the pipe's throughput.  Each precision is checked against a float32 computation on a
small size BEFORE it is timed; fp8 needs ``torch._scaled_mm`` and is skipped (with the
reason) where torch has no fp8 tensor path.

Prints one ``PROBE_JSON {..}`` line after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

MARKER = "PROBE_JSON "
PROBE = "tensor"


def measure(fn, *, seconds: float, repeats: int = 7, warmup: int = 3) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    single = max(float(start.elapsed_time(end)), 1e-3)
    calls = max(1, min(200, int((seconds * 1e3) / (repeats * single))))
    values = []
    for _ in range(repeats):
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)) / calls)
    return values


def entry(values: list[float], unit: str, method: str, **detail) -> dict:
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv = (statistics.pstdev(values) / mean) if len(values) > 1 and mean else None
    return {"value": median, "unit": unit, "method": method, "repeats": len(values),
            "cv": cv, **detail}


def skipped(reason: str) -> dict:
    return {"value": None, "unit": "", "method": "", "repeats": 0, "cv": None,
            "skipped": reason}


def _relative_error(out, ref) -> float:
    return float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _dtype_check(torch, dtype, *, size: int = 256, tolerance: float = 0.02) -> None:
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    error = _relative_error(a @ b, a.float() @ b.float())
    if not error < tolerance:
        raise RuntimeError(f"self-check failed: {dtype} product is off by {error:.3g} "
                           f"relative to float32")


def _scaled_mm(torch, a, b, out_dtype):
    """``torch._scaled_mm`` with unit scales, across the API shapes torch has had."""
    one = torch.ones((), device="cuda", dtype=torch.float32)
    try:
        return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=out_dtype)
    except TypeError:
        return torch._scaled_mm(a, b, one, one, None, out_dtype)


def _fp8_check(torch) -> None:
    size = 256
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn).t()
    out = _scaled_mm(torch, a, b, torch.bfloat16)
    error = _relative_error(out, a.float() @ b.float())
    if not error < 0.05:
        raise RuntimeError(f"self-check failed: the fp8 product is off by {error:.3g} "
                           f"relative to the dequantized operands")


def collect(*, seconds: float, size: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    if size < 64:
        raise SystemExit(f"{PROBE}: --size must be at least 64")
    flops = 2.0 * size ** 3
    entries: dict[str, dict] = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            _dtype_check(torch, dtype)
        except Exception as exc:  # noqa: BLE001 -- a failed self-check skips the entry
            entries[name] = skipped(f"{type(exc).__name__}: {exc}")
            continue
        try:
            a = torch.randn(size, size, device="cuda", dtype=dtype)
            b = torch.randn(size, size, device="cuda", dtype=dtype)
            times = measure(lambda: a @ b, seconds=seconds)
            entries[name] = entry([flops / (t / 1e3) / 1e12 for t in times], "TFLOPS",
                                  f"{size} x {size} x {size} dense product, cuBLAS "
                                  f"(2 x N^3 FLOPs per call)", size=size)
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entries[name] = skipped(f"{type(exc).__name__}: {exc}")
    try:
        _fp8_check(torch)
    except Exception as exc:  # noqa: BLE001 -- no fp8 path, or its check failed
        entries["fp8"] = skipped(f"torch._scaled_mm unavailable or wrong: "
                                 f"{type(exc).__name__}: {exc}")
    else:
        try:
            a = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn).t()
            times = measure(lambda: _scaled_mm(torch, a, b, torch.bfloat16),
                            seconds=seconds)
            entries["fp8"] = entry([flops / (t / 1e3) / 1e12 for t in times], "TFLOPS",
                                   f"{size}^3 dense fp8 product through torch._scaled_mm "
                                   f"with unit scales, bf16 output (2 x N^3 FLOPs per call)",
                                   size=size)
        except Exception as exc:  # noqa: BLE001
            entries["fp8"] = skipped(f"{type(exc).__name__}: {exc}")
    return {"probe": PROBE, "entries": entries,
            "detail": {"size": size, "device": torch.cuda.get_device_name(0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensor.py",
                                     description="Dense tensor-throughput probe")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget per precision (default 10)")
    parser.add_argument("--size", type=int, default=8192,
                        help="M = N = K of the timed product (default 8192)")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(seconds=max(args.seconds, 0.1), size=int(args.size))
    if not args.json:
        for name, item in result["entries"].items():
            state = (f"{item['value']:.1f} TFLOPS (cv {item['cv']:.3f})"
                     if item.get("value") is not None else f"skipped: {item['skipped']}")
            print(f"{PROBE}.{name:5s} {state}")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
