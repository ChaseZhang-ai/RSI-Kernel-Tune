#!/usr/bin/env python3
"""l2: cache bandwidth -- a copy whose working set fits in the last-level cache.

Standalone (``python3 l2.py``).  The same copy as the hbm probe, sized to stay resident in
L2 (a few MiB), so the reading is the cache's throughput rather than device memory's.
Two buffers, each ``--mib``; both must fit in L2 together to be a cache measurement --
the entry carries the size it used, so a reader can tell a cache hit from a spray.

Prints one ``PROBE_JSON {..}`` line after its human summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

MARKER = "PROBE_JSON "
PROBE = "l2"


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


def collect(*, seconds: float, mib: int) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"{PROBE}: no CUDA device")
    n_bytes = int(mib) << 20
    small = torch.arange(4096, dtype=torch.uint8, device="cuda")
    small_out = torch.empty_like(small)
    small_out.copy_(small)
    torch.cuda.synchronize()
    if not bool(torch.equal(small_out, small)):
        raise SystemExit(f"{PROBE}: self-check failed: a copy did not reproduce its source")
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    src.fill_(5)
    torch.cuda.synchronize()
    l2_bytes = 0
    try:
        l2_bytes = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0))
    except Exception:  # noqa: BLE001 -- a missing property is not a failure
        l2_bytes = 0
    try:
        times = measure(lambda: dst.copy_(src), seconds=seconds)
        item = entry([2 * n_bytes / (t / 1e3) / 1e9 for t in times], "GB/s",
                     "copy of a working set that fits in L2 (both buffers resident)",
                     working_set_mib=2 * mib, l2_mib=(l2_bytes >> 20) or None)
        if l2_bytes and 2 * n_bytes > l2_bytes:
            item["note"] = (f"2 x {mib} MiB exceeds the {l2_bytes >> 20} MiB L2: the "
                            f"reading is closer to device bandwidth than to the cache's")
    except Exception as exc:  # noqa: BLE001
        item = {"value": None, "unit": "", "method": "", "repeats": 0, "cv": None,
                "skipped": f"{type(exc).__name__}: {exc}"}
    return {"probe": PROBE, "entries": {"copy": item},
            "detail": {"mib": mib, "device": torch.cuda.get_device_name(0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="l2.py", description="Cache-bandwidth probe")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget (default 10)")
    parser.add_argument("--mib", type=int, default=8,
                        help="bytes per buffer, in MiB -- keep 2 x this inside L2 "
                             "(default 8)")
    parser.add_argument("--json", action="store_true", help="print the JSON line only")
    args = parser.parse_args(argv)
    result = collect(seconds=max(args.seconds, 0.1), mib=max(args.mib, 1))
    if not args.json:
        item = result["entries"]["copy"]
        state = (f"{item['value']:.1f} GB/s (cv {item['cv']:.3f})"
                 if item.get("value") is not None else f"skipped: {item['skipped']}")
        print(f"{PROBE}.copy {state}")
    print(MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
