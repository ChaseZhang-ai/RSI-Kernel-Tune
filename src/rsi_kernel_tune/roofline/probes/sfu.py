#!/usr/bin/env python3
"""sfu: special-function throughput -- exp, rsqrt and sigmoid chains.

Standalone (``python3 sfu.py``).  Transcendental operations run on the special-function
pipe (MUFU on NVIDIA parts), which is 4-8x narrower than the arithmetic pipe.  A torch
eager reading of one of these over a large tensor is a *memory* reading (read, compute,
write per call); to measure the pipe itself each element runs a dependent chain of the
operation and nothing else, with enough resident threads that the pipe's throughput, not
the chain's latency, is what binds.  Every chain is value-stable by construction (exp(0)
stays 1, rsqrt(2) tends to 1, sigmoid(0.5) tends to its fixed point), so no overflow or
denormal slows anything down.

Prints one ``PROBE_JSON {..}`` line after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

MARKER = "PROBE_JSON "
PROBE = "sfu"
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


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _exp_chain(x_ptr, out_ptr, n, ITER: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        for _ in range(ITER):
            x = tl.exp(x)
        tl.store(out_ptr + off, x, mask=mask)

    @triton.jit
    def _rsqrt_chain(x_ptr, out_ptr, n, ITER: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=1.0)
        for _ in range(ITER):
            x = tl.math.rsqrt(x)
        tl.store(out_ptr + off, x, mask=mask)

    @triton.jit
    def _sigmoid_chain(x_ptr, out_ptr, n, ITER: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        for _ in range(ITER):
            x = tl.sigmoid(x)
        tl.store(out_ptr + off, x, mask=mask)

    return {"exp": _exp_chain, "rsqrt": _rsqrt_chain, "sigmoid": _sigmoid_chain}


def _torch_chain(name: str, value, iterations: int):
    import torch

    out = value.clone()
    for _ in range(iterations):
        if name == "exp":
            out = torch.exp(out)
        elif name == "rsqrt":
            out = torch.rsqrt(out)
        else:
            out = torch.sigmoid(out)
    return out


def _start_values(name: str, n: int):
    import torch

    if name == "exp":
        return torch.zeros(n, device="cuda")             # exp(0) = 1, then 1 forever
    if name == "rsqrt":
        return torch.full((n,), 2.0, device="cuda")      # converges to 1
    return torch.full((n,), 0.5, device="cuda")          # sigmoid's fixed point is ~0.66


def _measure_op(name: str, kernel, *, seconds: float, elements: int) -> dict:
    import torch
    import triton

    # self-check: the same chain on a small size, computed in torch
    small_n, small_iter = 1024, 4
    x0 = _start_values(name, small_n)
    out0 = torch.empty_like(x0)
    kernel[(1,)](x0, out0, small_n, ITER=small_iter, BLOCK=small_n)
    expected = _torch_chain(name, x0, small_iter)
    if not bool(torch.allclose(out0, expected, rtol=1e-3, atol=1e-3)):
        raise RuntimeError(f"self-check failed: the device {name} chain differs from the "
                           f"same chain on the host")
    n = int(elements)
    x = _start_values(name, n)
    out = torch.empty_like(x)
    times = measure(lambda: kernel[(triton.cdiv(n, BLOCK),)](x, out, n, ITER=ITERATIONS,
                                                             BLOCK=BLOCK), seconds=seconds)
    return entry([n * ITERATIONS / (t / 1e3) / 1e9 for t in times], "Gop/s",
                 f"a chain of {ITERATIONS} dependent {name} operations per element in a "
                 f"Triton kernel, one element per thread (N x {ITERATIONS} operations per "
                 f"call: the special-function pipe's throughput)", elements=n,
                 iterations=ITERATIONS)


def collect(*, seconds: float, elements: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    entries: dict[str, dict] = {}
    try:
        kernels = _kernels()
    except Exception as exc:  # noqa: BLE001 -- no Triton: every entry is skipped
        reason = f"triton is not importable: {type(exc).__name__}: {exc}"
        for name in ("exp", "rsqrt", "sigmoid"):
            entries[name] = skipped(reason)
        return {"probe": PROBE, "entries": entries,
                "detail": {"elements": elements,
                           "device": torch.cuda.get_device_name(0)}}
    for name, kernel in kernels.items():
        try:
            entries[name] = _measure_op(name, kernel, seconds=seconds, elements=elements)
        except Exception as exc:  # noqa: BLE001
            entries[name] = skipped(f"{type(exc).__name__}: {exc}")
    return {"probe": PROBE, "entries": entries,
            "detail": {"elements": elements, "device": torch.cuda.get_device_name(0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfu.py",
                                     description="Special-function throughput probe")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget per operation (default 10)")
    parser.add_argument("--elements", type=int, default=1 << 22,
                        help="elements (threads) per launch (default 4194304)")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(seconds=max(args.seconds, 0.1), elements=max(args.elements, BLOCK))
    if not args.json:
        for name, item in result["entries"].items():
            state = (f"{item['value']:.1f} Gop/s (cv {item['cv']:.3f})"
                     if item.get("value") is not None else f"skipped: {item['skipped']}")
            print(f"{PROBE}.{name:8s} {state}")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
