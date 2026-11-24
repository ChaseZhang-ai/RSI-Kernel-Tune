"""``forge calibrate``: measure this campaign's GPU and archive the peaks.

    forge calibrate [--out calibration/<gpu>-<date>.json] [--seconds 10]
                    [--only hbm,l2,tensor,elementwise,sfu,launch]
                    [--remote | --local] [--timeout 900] [--json] [--root DIR]

Where it runs: the development machine, as a GPU job (supplement 2 of the task). With
``FK_DELIVERY=ssh`` in ``local.env`` the probes go to that host through ``DirectRunner``
-- the same delivery every run uses, with the host's per-GPU lock and the job timeout.
``--local`` measures this machine instead (a workstation with a usable GPU), which is for
development; the board is graded on B200 anyway, so a local reading is a reading of
another machine and the document says which one.

What it measures: device bandwidth (copy and read-only reduction), L2 bandwidth, dense
tensor throughput in bf16 / fp16 / fp8, the scalar pipe and a bandwidth-limited
elementwise path, special-function throughput (exp / rsqrt / sigmoid), and launch latency
(round trip, minimal kernel, back-to-back). Every probe self-checks its arithmetic before
it times anything; every entry reports repeats and the coefficient of variation.

The output JSON is the one ``forge roofline`` and ``skills/roofline/roofline_tools.py``
read. A rerun on the same day replaces the file (``--out`` picks another path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMAND = "calibrate"


def cmd_calibrate(args) -> int:
    from forgekernel.roofline import calibrate

    only = tuple(name.strip() for name in args.only.split(",") if name.strip())
    try:
        document = calibrate.calibrate(out=args.out, seconds=args.seconds, only=only,
                                       timeout=args.timeout, remote=args.remote,
                                       local=args.local, root=args.root)
    except calibrate.CalibrationError as e:
        sys.stderr.write(f"forge calibrate: {e}\n")
        return 2
    if args.json:
        print(json.dumps({k: v for k, v in document.items()}, indent=2))
        return 0
    for line in calibrate.summary_lines(document):
        print(line)
    print(f"\nwrote {document['path']}")
    print("use it with `forge roofline tasks/<op> --calibration "
          f"{document['path']} ...`")
    if (document.get("clocks") or {}).get("locked") is not True:
        print("the clocks were not pinned: pin them (the board locks SM 1500 / memory "
              "3996 MHz) before comparing a reading with the board's T_SOL / T_b")
    return 0


def _parser() -> argparse.ArgumentParser:
    from forgekernel.roofline import calibrate

    p = argparse.ArgumentParser(
        prog="forge calibrate",
        description="Measure a GPU's peaks (bandwidth, tensor throughput, pipes, launch "
                    "latency) and write them as one JSON document.")
    p.add_argument("--out", default=None,
                   help="where to write the document (default "
                        "calibration/<gpu>-<date>.json)")
    p.add_argument("--seconds", type=float, default=calibrate.DEFAULT_SECONDS,
                   help=f"timing budget per probe entry (default "
                        f"{calibrate.DEFAULT_SECONDS:g})")
    p.add_argument("--only", default=",".join(calibrate.PROBES),
                   help="comma-separated probes (default: all six)")
    p.add_argument("--remote", action="store_true",
                   help="run on the ssh host local.env names (the default when "
                        "FK_DELIVERY=ssh)")
    p.add_argument("--local", action="store_true",
                   help="run on this machine even when local.env says ssh")
    p.add_argument("--timeout", type=float, default=calibrate.DEFAULT_TIMEOUT_S,
                   help=f"seconds the whole job may take "
                        f"(default {calibrate.DEFAULT_TIMEOUT_S:g})")
    p.add_argument("--json", action="store_true", help="print the document as JSON")
    p.add_argument("--root", default=None, help="repository root (default: this checkout)")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    return cmd_calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
