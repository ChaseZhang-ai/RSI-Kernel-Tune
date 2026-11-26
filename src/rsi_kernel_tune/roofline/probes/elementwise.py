#!/usr/bin/env python3
"""elementwise: one bandwidth-limited and one arithmetic-limited elementwise reading.

Standalone (``python3 elementwise.py``).

* ``bandwidth``: an out-of-place ternary elementwise operation over large float32 tensors
  through torch -- the classic share of the device bandwidth a simple kernel reaches.
* ``alu``: a chain of dependent multiply-adds per element in a Triton kernel, one element
  per thread, so the scalar arithmetic pipe's throughput is what binds (a torch eager
  reading of the same expression would be a memory reading: one kernel per operation).

Prints one ``PROBE_JSON {..}`` line after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

MARKER = "PROBE_JSON "
PROBE = "elementwise"
# 2 FLOPs per multiply-add; the chain is the unit of work the reading counts.
ITERATIONS = 1024
BLOCK = 1024


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
    calls = max(1, min(2000, int((seconds * 1e3) / (repeats * single))))
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


def _bandwidth(torch, *, seconds: float, mib: int) -> dict:
    elements = (int(mib) << 20) // 4
    a = torch.randn(elements, device="cuda")
    b = torch.randn(elements, device="cuda")
    out = torch.empty_like(a)
    # self-check first
    small_a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    small_b = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    if not bool(torch.allclose(small_a + small_b, torch.tensor([1.5, 2.5, 3.5],
                                                               device="cuda"))):
        raise RuntimeError("self-check failed: an elementwise sum is wrong")
    times = measure(lambda: torch.add(a, b, out=out), seconds=seconds)
    return entry([3 * elements * 4 / (t / 1e3) / 1e9 for t in times], "GB/s",
                 "out-of-place ternary elementwise operation over float32 tensors "
                 "(2 reads + 1 write per element)", elements=elements)


def _alu(torch, *, seconds: float, elements: int) -> dict:
    import triton
    import triton.language as tl

    @triton.jit
    def _fma_chain(x_ptr, out_ptr, n, a, b, ITER: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        for _ in range(ITER):
            x = x * a + b
        tl.store(out_ptr + off, x, mask=mask)

    def run(x, out, n, a, b):
        _fma_chain[(triton.cdiv(n, BLOCK),)](x, out, n, a, b, ITER=ITERATIONS, BLOCK=BLOCK)

    a, b = 1.0001, 0.5
    # self-check: the same recurrence on a small size, computed in torch
    small_n, small_iter = 4096, 8
    x0 = torch.rand(small_n, device="cuda") + 0.5
    out0 = torch.empty_like(x0)
    _fma_chain[(1,)](x0, out0, small_n, a, b, ITER=small_iter, BLOCK=small_n)
    expected = x0.clone()
    for _ in range(small_iter):
        expected = expected * a + b
    if not bool(torch.allclose(out0, expected, rtol=1e-5, atol=1e-5)):
        raise RuntimeError("self-check failed: the device chain differs from the same "
                           "recurrence on the host")
    n = int(elements)
    x = torch.rand(n, device="cuda") + 0.5
    out = torch.empty_like(x)
    times = measure(lambda: run(x, out, n, a, b), seconds=seconds)
    flops = 2.0 * n * ITERATIONS
    return entry([flops / (t / 1e3) / 1e9 for t in times], "Gop/s",
                 f"a chain of {ITERATIONS} dependent multiply-adds per element in a Triton "
                 f"kernel, one element per thread (2 x N x {ITERATIONS} FLOPs per call: the "
                 f"scalar arithmetic pipe's throughput)", elements=n,
                 iterations=ITERATIONS)


def collect(*, seconds: float, mib: int, elements: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    entries = {}
    try:
        entries["bandwidth"] = _bandwidth(torch, seconds=seconds, mib=mib)
    except Exception as exc:  # noqa: BLE001
        entries["bandwidth"] = skipped(f"{type(exc).__name__}: {exc}")
    try:
        import triton  # noqa: F401 -- the entry's own availability check
    except Exception as exc:  # noqa: BLE001
        entries["alu"] = skipped(f"triton is not importable: {type(exc).__name__}: {exc}")
    else:
        try:
            entries["alu"] = _alu(torch, seconds=seconds, elements=elements)
        except Exception as exc:  # noqa: BLE001
            entries["alu"] = skipped(f"{type(exc).__name__}: {exc}")
    return {"probe": PROBE, "entries": entries,
            "detail": {"mib": mib, "alu_elements": elements,
                       "device": torch.cuda.get_device_name(0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elementwise.py",
                                     description="Elementwise bandwidth and scalar-pipe "
                                                 "throughput probe")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget per entry (default 10)")
    parser.add_argument("--mib", type=int, default=256,
                        help="bytes per input tensor for the bandwidth entry, in MiB "
                             "(default 256)")
    parser.add_argument("--elements", type=int, default=1 << 22,
                        help="elements for the arithmetic entry (default 4194304)")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(seconds=max(args.seconds, 0.1), mib=max(args.mib, 1),
                     elements=max(args.elements, BLOCK))
    if not args.json:
        for name, item in result["entries"].items():
            state = (f"{item['value']:.1f} {item['unit']} (cv {item['cv']:.3f})"
                     if item.get("value") is not None else f"skipped: {item['skipped']}")
            print(f"{PROBE}.{name:10s} {state}")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
