#!/usr/bin/env python3
"""hbm: device-memory bandwidth -- a large copy (read + write) and a read-only reduction.

Standalone on purpose: ``python3 hbm.py`` runs on any machine with a CUDA torch, and the
calibration driver (``run_all.py``) runs this same file as a subprocess.  Method: a copy
reads every byte once and writes it once (2 x N bytes of traffic); a reduction reads N
bytes and writes almost nothing.  A minimal correctness check runs before any timing, so
a broken copy cannot be reported as a fast one.

Prints one ``PROBE_JSON {..}`` line (the driver parses it) after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

MARKER = "PROBE_JSON "
PROBE = "hbm"


def measure(fn, *, seconds: float, repeats: int = 7, warmup: int = 3) -> list[float]:
    """Per-call milliseconds, ``repeats`` times: each repeat times a batch of calls."""
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
    calls = max(1, min(500, int((seconds * 1e3) / (repeats * single))))
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
    """A probe entry from per-repeat derived values: median, CV, repeats."""
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv = (statistics.pstdev(values) / mean) if len(values) > 1 and mean else None
    return {"value": median, "unit": unit, "method": method, "repeats": len(values),
            "cv": cv, **detail}


def skipped(reason: str) -> dict:
    return {"value": None, "unit": "", "method": "", "repeats": 0, "cv": None,
            "skipped": reason}


def collect(*, seconds: float, mib: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    n_bytes = int(mib) << 20
    n_bytes -= n_bytes % 4
    entries: dict[str, dict] = {}
    # -- the copy: a self-check on 4096 bytes first ---------------------------------
    small = torch.arange(4096, dtype=torch.uint8, device="cuda")
    small_out = torch.empty_like(small)
    small_out.copy_(small)
    torch.cuda.synchronize()
    if not bool(torch.equal(small_out, small)):
        raise SystemExit(f"{PROBE}: self-check failed: a device copy did not reproduce "
                         f"its source")
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    src.fill_(3)
    torch.cuda.synchronize()
    try:
        times = measure(lambda: dst.copy_(src), seconds=seconds)
        entries["copy"] = entry([2 * n_bytes / (t / 1e3) / 1e9 for t in times], "GB/s",
                                "device-to-device copy: every byte read once and written "
                                "once (2 x N bytes of traffic)")
    except Exception as exc:  # noqa: BLE001 -- reported, the read part still runs
        entries["copy"] = skipped(f"{type(exc).__name__}: {exc}")
    del src, dst
    torch.cuda.empty_cache()
    # -- a read-only reduction ------------------------------------------------------
    elements = n_bytes // 4
    data = torch.empty(elements, dtype=torch.float32, device="cuda").normal_()
    ones = torch.ones(1 << 20, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    if abs(float(ones.sum()) - float(1 << 20)) > 0.5:
        raise SystemExit(f"{PROBE}: self-check failed: a sum of ones is not its count")
    try:
        times = measure(lambda: torch.sum(data), seconds=seconds)
        entries["read"] = entry([elements * 4 / (t / 1e3) / 1e9 for t in times], "GB/s",
                                "read-only reduction over N bytes (one pass, almost no "
                                "write traffic)")
    except Exception as exc:  # noqa: BLE001
        entries["read"] = skipped(f"{type(exc).__name__}: {exc}")
    return {"probe": PROBE, "entries": entries,
            "detail": {"copy_mib": mib, "read_mib": mib, "device":
                       torch.cuda.get_device_name(0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hbm.py",
                                     description="Device-memory bandwidth probe")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget per entry (default 10)")
    parser.add_argument("--mib", type=int, default=1024,
                        help="bytes moved per buffer, in MiB (default 1024)")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(seconds=max(args.seconds, 0.1), mib=max(args.mib, 1))
    if not args.json:
        for name, item in result["entries"].items():
            state = (f"{item['value']:.1f} {item['unit']} (cv {item['cv']:.3f})"
                     if item.get("value") is not None else f"skipped: {item['skipped']}")
            print(f"{PROBE}.{name:6s} {state}")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
