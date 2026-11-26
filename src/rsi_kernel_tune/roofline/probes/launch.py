#!/usr/bin/env python3
"""launch: what one launch costs -- round-trip latency, minimal kernel time, back-to-back.

Standalone (``python3 launch.py``).  Three readings, all microseconds:

* ``roundtrip``: the host-side wall clock from just before a launch to just after the
  device reports the work done (launch + execute + synchronize).  This is the per-call
  floor of a program that does one launch per step.
* ``min_kernel``: the device-side duration of a minimal kernel, taken between CUDA events
  around the launch.  Nothing but the kernel's own time.
* ``back_to_back``: how many launches per second the host can sustain when the device is
  never waited for -- the launch *throughput*.

A Triton one-block kernel is used when Triton is importable; otherwise the smallest torch
operation available (a one-element add), which is still a real kernel launch.

Prints one ``PROBE_JSON {..}`` line after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

MARKER = "PROBE_JSON "
PROBE = "launch"
REPEATS = 50
BATCH = 100


def entry(values: list[float], unit: str, method: str, **detail) -> dict:
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv = (statistics.pstdev(values) / mean) if len(values) > 1 and mean else None
    return {"value": median, "unit": unit, "method": method, "repeats": len(values),
            "cv": cv, **detail}


def _launcher(torch):
    """(fn, description) for the smallest kernel launch this machine can run."""
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 -- fall through to torch
        a = torch.zeros(1, device="cuda")
        out = torch.zeros(1, device="cuda")
        return (lambda: torch.add(a, a, out=out),
                "a one-element torch add (a real kernel launch, no Triton here)")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    @triton.jit
    def _stamp(out_ptr):
        tl.store(out_ptr + tl.program_id(0), 1)

    def launch():
        _stamp[(1,)](out)

    return launch, "a single-block Triton kernel that stores one value"


def collect(*, repeats: int = REPEATS, batch: int = BATCH) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    launch, description = _launcher(torch)
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)

    roundtrip: list[float] = []
    min_kernel: list[float] = []
    for _ in range(max(repeats, 5)):
        t0 = time.perf_counter()
        launch()
        torch.cuda.synchronize()
        roundtrip.append((time.perf_counter() - t0) * 1e6)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        min_kernel.append(float(start.elapsed_time(end)) * 1e3)
    start.record()
    for _ in range(max(batch, 10)):
        launch()
    end.record()
    torch.cuda.synchronize()
    back_to_back = float(start.elapsed_time(end)) * 1e3 / max(batch, 10)
    entries = {
        "roundtrip": entry(roundtrip, "us",
                           f"{description}: host wall clock from before the launch to after "
                           f"the device finished (launch + execute + synchronize)"),
        "min_kernel": entry(min_kernel, "us",
                            f"{description}: device time between CUDA events around one "
                            f"launch"),
        "back_to_back": entry([back_to_back] * 3, "us",
                              f"{description} launched {max(batch, 10)} times without "
                              f"waiting: device time per launch (launch throughput)",
                              batch=max(batch, 10)),
    }
    return {"probe": PROBE, "entries": entries,
            "detail": {"device": torch.cuda.get_device_name(0), "launch": description}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="launch.py",
                                     description="Launch-latency probe")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"round-trip samples (default {REPEATS})")
    parser.add_argument("--batch", type=int, default=BATCH,
                        help=f"launches in the throughput measurement (default {BATCH})")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(repeats=max(args.repeats, 5), batch=max(args.batch, 10))
    if not args.json:
        for name, item in result["entries"].items():
            print(f"{PROBE}.{name:12s} {item['value']:.3f} us "
                  f"(cv {item['cv']:.3f})" if item["cv"] is not None
                  else f"{PROBE}.{name:12s} {item['value']:.3f} us")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
