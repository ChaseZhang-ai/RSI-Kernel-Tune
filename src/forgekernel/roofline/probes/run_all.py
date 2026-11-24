#!/usr/bin/env python3
"""The calibration driver: run every probe as a subprocess on the GPU machine.

This file is shipped VERBATIM into a GPU job tree (``calibrate.prepare_tree``) and also
runs on the development machine's own ``python3``; it imports nothing from ForgeKernel.
Each probe runs in its own process, so one that dies or hangs (a bad driver, an OOM)
costs that probe and not the calibration: the rest still run, and the failure is reported
inside the document as ``errors.<probe>`` instead of taking the job down.

One machine fact block (torch + ``nvidia-smi``) is collected once here; the probes report
only their own readings. The last line printed is ``CALIBRATION_JSON {..}``.

Usage:
    python3 run_all.py [--seconds 10] [--only hbm,l2,tensor,elementwise,sfu,launch]
                       [--timeout 900]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

MARKER = "CALIBRATION_JSON "
PROBE_MARKER = "PROBE_JSON "
PROBES = ("hbm", "l2", "tensor", "elementwise", "sfu", "launch")
DEFAULT_TIMEOUT_S = 900


def _int_or_none(text):
    digits = "".join(ch for ch in str(text or "") if ch.isdigit())
    return int(digits) if digits else None


def nvidia_smi() -> dict:
    """Name, driver and clock columns as CSV (the fields we can read on every driver)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,"
             "clocks.applications.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        return {"rc": proc.returncode, "out": proc.stdout.strip(),
                "err": proc.stderr[-2000:]}
    except FileNotFoundError:
        return {"rc": 127, "out": "", "err": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "", "err": "nvidia-smi timed out"}
    except Exception as exc:  # noqa: BLE001 -- a fact we could not read, not a crash
        return {"rc": 1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def parse_smi(text: str) -> dict:
    """The CSV line -> {name, driver, current_sm_mhz, current_mem_mhz, app_sm_mhz,
    max_sm_mhz}. An empty or unreadable line yields {}."""
    line = next((row for row in (text or "").splitlines() if row.strip()), "")
    if not line:
        return {}
    fields = [f.strip() for f in line.split(",")]
    names = ("name", "driver", "current_sm_mhz", "current_mem_mhz", "app_sm_mhz",
             "max_sm_mhz")
    out: dict = {}
    for name, value in zip(names, fields):
        if name in ("name", "driver"):
            if value and value not in ("N/A", "[N/A]"):
                out[name] = value
        else:
            number = _int_or_none(value)
            if number is not None:
                out[name] = number
    return out


def clocks_locked(clocks: dict) -> bool | None:
    """Is the SM clock pinned? Locked when an application clock is set, the current clock
    matches it, and it sits below the maximum; None when the numbers do not decide."""
    app = clocks.get("app_sm_mhz")
    current = clocks.get("sm_mhz")
    maximum = clocks.get("max_sm_mhz")
    if not app or not current:
        return False if (maximum and current and current >= maximum) else None
    if not maximum or abs(current - app) > 50:
        return None
    return app < maximum


def machine() -> dict:
    """This machine as the calibration document's header block."""
    info: dict = {"hostname": platform.node(), "python": platform.python_version(),
                  "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["arch"] = f"sm_{props.major}{props.minor}"
            info["sm_count"] = props.multi_processor_count
            info["memory_gib"] = round(props.total_memory / float(1 << 30), 1)
            l2 = int(getattr(props, "L2_cache_size", 0) or 0)
            info["l2_mib"] = (l2 >> 20) or None
        else:
            info["gpu_error"] = "torch sees no CUDA device"
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    smi = nvidia_smi()
    parsed = parse_smi(smi.get("out") or "")
    for key, value in parsed.items():
        if key in ("name", "driver"):
            info.setdefault(key, value)
    clocks = {"sm_mhz": parsed.get("current_sm_mhz"),
              "mem_mhz": parsed.get("current_mem_mhz"),
              "app_sm_mhz": parsed.get("app_sm_mhz"),
              "max_sm_mhz": parsed.get("max_sm_mhz")}
    clocks["locked"] = clocks_locked(clocks)
    info["clocks"] = clocks
    info["nvidia_smi"] = {k: smi.get(k) for k in ("rc", "err")}
    return info


def probe_command(name: str, *, seconds: float, here: Path) -> list[str]:
    return [sys.executable, str(here / f"{name}.py"), "--seconds", repr(float(seconds)),
            "--json"]


def parse_probe_payload(stdout: str) -> dict | None:
    """The probe's own JSON document (its last ``PROBE_JSON`` line), or None."""
    found = None
    for line in (stdout or "").splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                found = json.loads(line[len(PROBE_MARKER):])
            except json.JSONDecodeError:
                continue
    return found if isinstance(found, dict) else None


def run(*, seconds: float, only: tuple[str, ...], timeout: float,
        here: Path) -> dict:
    """Every probe, one subprocess each: {"machine", "probes", "errors", ...}."""
    document: dict = {"machine": machine(), "probes": {}, "errors": {},
                      "seconds": seconds, "only": list(only)}
    per_probe_timeout = max(float(timeout) / max(len(only), 1), 60.0)
    for name in only:
        script = here / f"{name}.py"
        if not script.is_file():
            document["errors"][name] = {"error": f"{script} is missing"}
            print(f"PROBE {name} MISSING {script}", flush=True)
            continue
        try:
            proc = subprocess.run(probe_command(name, seconds=seconds, here=here),
                                  capture_output=True, text=True, cwd=str(here),
                                  timeout=per_probe_timeout)
        except subprocess.TimeoutExpired:
            document["errors"][name] = {"error": f"timed out after "
                                                  f"{per_probe_timeout:.0f}s"}
            print(f"PROBE {name} TIMEOUT", flush=True)
            continue
        payload = parse_probe_payload(proc.stdout)
        if payload is None:
            tail = " / ".join((proc.stdout or proc.stderr or "").strip().splitlines()[-4:])
            document["errors"][name] = {"rc": proc.returncode, "tail": tail[:900]}
            print(f"PROBE {name} FAILED rc={proc.returncode} {tail[:200]}", flush=True)
            continue
        document["probes"][name] = payload
        values = [f"{key}={entry.get('value'):.4g}{entry.get('unit', '')}"
                  for key, entry in (payload.get("entries") or {}).items()
                  if isinstance(entry.get("value"), (int, float))]
        print(f"PROBE {name} ok " + " ".join(values), flush=True)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_all.py",
                                     description="Run every calibration probe on this "
                                                 "machine and print one JSON document")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="timing budget per probe entry (default 10)")
    parser.add_argument("--only", default=",".join(PROBES),
                        help="comma-separated probe names (default all)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"budget for the whole run (default {DEFAULT_TIMEOUT_S})")
    args = parser.parse_args(argv)
    only = tuple(name.strip() for name in args.only.split(",") if name.strip())
    unknown = [name for name in only if name not in PROBES]
    if unknown or not only:
        parser.error(f"unknown probe(s) {', '.join(unknown) or '(none given)'}; "
                     f"known: {', '.join(PROBES)}")
    document = run(seconds=max(args.seconds, 0.1), only=only, timeout=args.timeout,
                   here=Path(__file__).resolve().parent)
    print(MARKER + json.dumps(document), flush=True)
    return 0 if document["probes"] else 2


if __name__ == "__main__":
    sys.exit(main())
