"""``forge calibrate``: measure a machine's peaks and archive them as one JSON document.

The probes (``probes/``) are standalone scripts that only need torch (and Triton where a
reading needs it); this module is the control side: it builds a tiny GPU job tree holding
them, delivers it the way ForgeKernel delivers every GPU job, reads the JSON back and
validates it.

Why a job tree and not a bare subprocess: the probes need a GPU, and ForgeKernel already
has exactly one honest way to get one -- ``submit.direct`` -- which brings the per-GPU
lock (including ``FK_JOB_LOCK_CMD`` on a shared workstation), the job timeout, the log
and, for the development machine, the ssh delivery of ``local.env``. ``forge target
probe`` reaches its machine the same way (``targethw.probe``), so there is one delivery
mechanism and not two. Nothing here builds a shell command: the entry script is a file
with the probe names and the seconds budget written into it.

The document::

    {schema, gpu, arch, sm_count, memory_gib, driver, cuda, torch, clocks: {sm_mhz,
     mem_mhz, app_sm_mhz, max_sm_mhz, locked}, ts, delivery, seconds, only,
     probes: {<probe>: {probe, entries: {<entry>: {value, unit, method, repeats, cv,
                                                  skipped?}}, detail}}, errors: {}}

``clocks.locked`` is read, never guessed: an unpinned clock means the peaks are readings
at whatever the machine happened to run at, which the file says.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "forgekernel.calibration/1"
PAYLOAD_MARKER = "CALIBRATION_JSON "
PROBE_MARKER = "PROBE_JSON "
# The probe names, in the order a document lists them (the driver in probes/ is the
# authority for what exists; this list is what the command runs by default).
PROBES = ("hbm", "l2", "tensor", "elementwise", "sfu", "launch")
CALIBRATION_DIR = "calibration"
DEFAULT_SECONDS = 10.0
DEFAULT_TIMEOUT_S = 900
CALIBRATE_RUN = "_calibrate"
CALIBRATE_OP = "calibrate"
# A probe entry that takes far longer than the whole job's budget is not a reading.
MIN_PER_PROBE_TIMEOUT_S = 60.0


class CalibrationError(RuntimeError):
    """The calibration could not be delivered, produced no document, or the document is
    not readable (a malformed calibration must never be used as a peak set)."""


# -- the document ----------------------------------------------------------------------


def parse_payload(stdout: str) -> dict:
    """The driver's document: the last ``CALIBRATION_JSON`` line of a job's output."""
    found = None
    for line in (stdout or "").splitlines():
        if not line.startswith(PAYLOAD_MARKER):
            continue
        try:
            found = json.loads(line[len(PAYLOAD_MARKER):])
        except json.JSONDecodeError:
            continue
    if not isinstance(found, dict):
        tail = " / ".join((stdout or "").strip().splitlines()[-5:])
        raise CalibrationError("the calibration job printed no document: " + tail[:600])
    return found


def parse_probe_payload(stdout: str) -> dict | None:
    """One probe's own document (the last ``PROBE_JSON`` line), or None."""
    found = None
    for line in (stdout or "").splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                found = json.loads(line[len(PROBE_MARKER):])
            except json.JSONDecodeError:
                continue
    return found if isinstance(found, dict) else None


def validate_entry(probe: str, name: str, entry) -> dict:
    """One probe reading: a number with a unit, or an explicit ``skipped`` reason."""
    where = f"{probe}.{name}"
    if not isinstance(entry, dict):
        raise CalibrationError(f"{where}: not an entry object")
    skipped = entry.get("skipped")
    value = entry.get("value")
    if value is None:
        if not skipped:
            raise CalibrationError(f"{where}: no value and no reason (skipped)")
        return {"value": None, "unit": str(entry.get("unit") or ""), "skipped": str(skipped)}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{where}: value {value!r} is not a number")
    if not float(value) > 0:
        raise CalibrationError(f"{where}: value {value!r} is not positive")
    unit = str(entry.get("unit") or "")
    if not unit:
        raise CalibrationError(f"{where}: a reading without a unit")
    repeats = entry.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise CalibrationError(f"{where}: repeats must be a positive integer, got "
                               f"{repeats!r}")
    cv = entry.get("cv")
    if cv is not None and (isinstance(cv, bool) or not isinstance(cv, (int, float))
                           or cv < 0):
        raise CalibrationError(f"{where}: cv must be a non-negative number or null, "
                               f"got {cv!r}")
    out = {"value": float(value), "unit": unit, "repeats": repeats,
           "cv": (float(cv) if cv is not None else None),
           "method": str(entry.get("method") or "")}
    for key, item in entry.items():
        if key not in out and key not in ("skipped",):
            out[key] = item
    return out


def build(payload: dict, *, delivery: str, when: str | None = None,
          seconds: float | None = None, only=PROBES) -> dict:
    """The driver's document -> the calibration document (validated)."""
    machine = payload.get("machine") or {}
    raw_probes = payload.get("probes") or {}
    if not raw_probes:
        errors = payload.get("errors") or {}
        detail = "; ".join(f"{name}: {json.dumps(value)[:200]}"
                           for name, value in sorted(errors.items())) or "no probe output"
        raise CalibrationError(f"every probe failed ({detail})")
    probes: dict[str, dict] = {}
    for name, probe in raw_probes.items():
        entries = probe.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise CalibrationError(f"{name}: a probe document without entries")
        probes[name] = {
            "probe": str(probe.get("probe") or name),
            "entries": {entry: validate_entry(name, entry, value)
                        for entry, value in entries.items()},
            "detail": probe.get("detail") or {},
        }
    clocks = machine.get("clocks") or {}
    document = {
        "schema": SCHEMA,
        "gpu": str(machine.get("gpu") or ""),
        "arch": machine.get("arch"),
        "sm_count": machine.get("sm_count"),
        "memory_gib": machine.get("memory_gib"),
        "l2_mib": machine.get("l2_mib"),
        "driver": machine.get("driver"),
        "cuda": machine.get("cuda"),
        "torch": machine.get("torch"),
        "clocks": {key: clocks.get(key) for key in ("sm_mhz", "mem_mhz", "app_sm_mhz",
                                                   "max_sm_mhz", "locked")},
        "ts": when or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "delivery": delivery,
        "seconds": seconds if seconds is not None else payload.get("seconds"),
        "only": list(only),
        "hostname": machine.get("hostname"),
        "probes": probes,
        "errors": payload.get("errors") or {},
    }
    if not document["gpu"]:
        raise CalibrationError("the calibration job reported no GPU name: the document "
                               "would name no machine")
    return document


def load(path: str | Path) -> dict:
    """A calibration file, validated. ``CalibrationError`` on anything unusable -- a
    peak set read from a half-written file would look like a measurement."""
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        raise CalibrationError(f"{file}: {e.strerror or e}") from None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise CalibrationError(f"{file}: not JSON: {e}") from None
    if not isinstance(doc, dict):
        raise CalibrationError(f"{file}: not a calibration document")
    if doc.get("schema") != SCHEMA:
        raise CalibrationError(f"{file}: schema {doc.get('schema')!r}, expected {SCHEMA!r} "
                               f"-- rerun `forge calibrate`")
    if not doc.get("gpu"):
        raise CalibrationError(f"{file}: it names no GPU")
    if doc.get("clocks", {}).get("locked") is not True:
        # not fatal: the readings are still readings, they are just not at a pinned clock
        pass
    return doc


def default_out(document: dict, *, directory: str | Path | None = None) -> Path:
    """``calibration/<gpu>-<date>.json`` (a rerun on the same day replaces it)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(document.get("gpu") or "gpu").lower()).strip("-")
    date = str(document.get("ts") or "")[:10] or datetime.now(timezone.utc).date().isoformat()
    base = Path(directory) if directory is not None else Path(CALIBRATION_DIR)
    return base / f"{slug or 'gpu'}-{date}.json"


def summary_lines(document: dict) -> list[str]:
    """The human summary a caller prints (one line per reading, then the clocks)."""
    clocks = document.get("clocks") or {}
    lock = {True: "locked", False: "not locked", None: "lock unknown"}.get(
        clocks.get("locked"), "lock unknown")
    lines = [f"gpu       {document.get('gpu') or '?'}  {document.get('arch') or '?'}  "
             f"{document.get('sm_count') or '?'} SMs  "
             f"{document.get('memory_gib') or '?'} GiB",
             f"stack     cuda {document.get('cuda') or '?'}, torch "
             f"{document.get('torch') or '?'}, driver {document.get('driver') or '?'}",
             f"clocks    SM {clocks.get('sm_mhz') or '?'} MHz / memory "
             f"{clocks.get('mem_mhz') or '?'} MHz ({lock})"]
    for name in sorted(document.get("probes") or {}):
        for entry, reading in (document["probes"][name]["entries"] or {}).items():
            if reading.get("value") is None:
                lines.append(f"{name}.{entry:12s} skipped: {reading.get('skipped', '')}"
                             [:100])
            else:
                cv = reading.get("cv")
                lines.append(f"{name}.{entry:12s} {reading['value']:.4g} "
                             f"{reading['unit']}"
                             + (f"  cv {cv:.3f}" if isinstance(cv, float) else "")
                             + f"  x{reading.get('repeats')}")
    for name, error in sorted((document.get("errors") or {}).items()):
        lines.append(f"{name}: FAILED {json.dumps(error)[:160]}")
    if clocks.get("locked") is not True:
        lines.append("the clocks were not pinned: these readings are at whatever the "
                     "machine ran at, not at the board's lock")
    return lines


# -- delivery --------------------------------------------------------------------------


@dataclass(frozen=True)
class Delivered:
    rc: int
    stdout: str
    stderr: str


def entry_script(*, seconds: float, only: tuple[str, ...], timeout: float) -> str:
    """The job's ``gpu_entry.sh``: one task, the driver, with the budget written in."""
    return f"""#!/bin/bash
# The calibration's GPU job entry: one task, the probe driver, in this directory.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
case "${{1:-calibrate}}" in
  calibrate) exec python3 run_all.py --seconds {seconds!r} --only {','.join(only)} \\
      --timeout {timeout!r} ;;
  *) echo "calibrate: unknown task $1" >&2; exit 2 ;;
esac
"""


def prepare_tree(*, seconds: float = DEFAULT_SECONDS, only: tuple[str, ...] = PROBES,
                 timeout: float = DEFAULT_TIMEOUT_S) -> tuple[Path, Path]:
    """A tiny GPU job tree holding every probe and the driver, committed on ``main``.

    Copying the probe files (rather than importing them) is what makes the job
    self-contained: the host needs a Python with torch and nothing of ForgeKernel.
    """
    from forgekernel.taskcard import _git_init

    base = Path(tempfile.mkdtemp(prefix="fk_calibrate_"))
    op_dir = base / CALIBRATE_RUN / "operators" / CALIBRATE_OP
    op_dir.mkdir(parents=True, exist_ok=True)
    for name in list(PROBES) + ["run_all"]:
        source = Path(__file__).with_name("probes") / f"{name}.py"
        (op_dir / f"{name}.py").write_text(source.read_text(encoding="utf-8"),
                                           encoding="utf-8")
    script = op_dir / "gpu_entry.sh"
    script.write_text(entry_script(seconds=seconds, only=only, timeout=timeout),
                      encoding="utf-8")
    script.chmod(0o755)
    _git_init(op_dir, "Chore: calibration tree")
    return base, op_dir


class LocalRun:
    """Run the probes on this machine, under the per-GPU lock a GPU job takes."""

    def __init__(self, gpu: str | None = None):
        self.gpu = gpu

    def describe(self) -> str:
        return "this machine"

    def run(self, *, seconds: float, only: tuple[str, ...],
            timeout: float) -> Delivered:
        from forgekernel.submit.direct import gpu_lock

        gpu = self.gpu
        if gpu is None:
            try:
                from forgekernel.localenv import load_machine

                gpu = load_machine().gpu
            except Exception:  # noqa: BLE001 -- no local.env: the default device
                gpu = "0"
        _base, op_dir = prepare_tree(seconds=seconds, only=only, timeout=timeout)
        try:
            with gpu_lock(gpu):
                proc = subprocess.run(["bash", str(op_dir / "gpu_entry.sh"), "calibrate"],
                                      capture_output=True, text=True, cwd=str(op_dir),
                                      timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            return Delivered(124, "", f"the calibration exceeded {timeout + 300:.0f}s")
        except OSError as e:
            return Delivered(1, "", f"{type(e).__name__}: {e}")
        return Delivered(proc.returncode, proc.stdout, proc.stderr)


class RemoteRun:
    """Run the probes on the ssh host ``local.env`` names, through ``DirectRunner``.

    The same delivery every GPU job of a run uses: ``git archive`` the tree to
    ``REMOTE_WORKDIR``, the host's per-GPU lock, ``gpu_entry.sh`` under the job timeout.
    """

    def __init__(self, machine, gpu: str | None = None, runner=None):
        self.machine = machine
        self.gpu = gpu if gpu is not None else machine.gpu
        self._runner = runner                    # a factory(cfg) -> runner, for tests

    def describe(self) -> str:
        return f"ssh {self.machine.remote_host}"

    def config(self, op_dir: Path, timeout: float):
        from forgekernel.submit.config import SubmitConfig

        return SubmitConfig(
            branch="main",
            entry_script="gpu_entry.sh",
            env={"PYTHONUNBUFFERED": "1"},
            timeout=int(timeout),
            delivery="ssh",
            op_dir=str(op_dir),
            exec_host=self.machine.remote_host,
            exec_workdir=self.machine.remote_workdir,
            exec_env_file=self.machine.remote_env_file,
            exec_gpu=str(self.gpu),
        )

    def run(self, *, seconds: float, only: tuple[str, ...],
            timeout: float) -> Delivered:
        from forgekernel.submit.direct import DirectRunner
        from forgekernel.submit.jobs import STATUS_SUCCEEDED

        try:
            _base, op_dir = prepare_tree(seconds=seconds, only=only, timeout=timeout)
        except (OSError, RuntimeError) as e:
            return Delivered(1, "", f"could not build the calibration tree: {e}")
        cfg = self.config(op_dir, timeout)
        runner = (self._runner or DirectRunner)(cfg)
        try:
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
                status = runner.run_mode("main", "calibrate", log)
                log.flush()
                log.seek(0)
                text = log.read()
        except Exception as e:  # noqa: BLE001 -- report it as a failed calibration
            return Delivered(1, "", f"{type(e).__name__}: {e}")
        return Delivered(0 if status == STATUS_SUCCEEDED else 1, text, "")


def resolve_delivery(*, remote: bool = False, local: bool = False, root=None):
    """(the deliverer, its name). Explicit flags win; otherwise ``local.env`` decides --
    which is how this machine follows the development machine without being told twice."""
    from forgekernel.localenv import DELIVERY_SSH, load_machine

    if remote and local:
        raise CalibrationError("--remote and --local are mutually exclusive")
    machine = load_machine(root)
    if local:
        return LocalRun(machine.gpu), "this machine"
    if remote or machine.delivery == DELIVERY_SSH:
        if not machine.remote_host or not machine.remote_workdir:
            raise CalibrationError(
                "the ssh delivery needs FK_DELIVERY=ssh, REMOTE_HOST and REMOTE_WORKDIR "
                "in local.env (a calibration runs on the development machine as a GPU job "
                "does); use --local to measure this machine instead")
        return RemoteRun(machine), f"ssh {machine.remote_host}"
    return LocalRun(machine.gpu), "this machine"


def calibrate(*, out: str | Path | None = None, seconds: float = DEFAULT_SECONDS,
              only: tuple[str, ...] = PROBES, timeout: float = DEFAULT_TIMEOUT_S,
              remote: bool = False, local: bool = False, root=None,
              directory: str | Path | None = None) -> dict:
    """Measure, validate, archive. Returns the document (with ``path`` filled in)."""
    if not only:
        raise CalibrationError("no probe selected")
    unknown = [name for name in only if name not in PROBES]
    if unknown:
        raise CalibrationError(f"unknown probe(s) {', '.join(unknown)}; known: "
                               f"{', '.join(PROBES)}")
    deliverer, _name = resolve_delivery(remote=remote, local=local, root=root)
    delivered = deliverer.run(seconds=seconds, only=tuple(only), timeout=timeout)
    if delivered.rc != 0 and not (delivered.stdout or "").strip():
        raise CalibrationError(f"the calibration on {deliverer.describe()} failed "
                               f"(rc {delivered.rc}): {delivered.stderr[-600:]}")
    payload = parse_payload(delivered.stdout)
    document = build(payload, delivery=deliverer.describe(), seconds=seconds, only=only)
    path = Path(out) if out is not None else default_out(document, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    document["path"] = str(path)
    return document


def main(argv: list[str] | None = None) -> int:                       # pragma: no cover
    """Not the command's entry (that is ``forgekernel.ext.calibrate``); kept so this
    module can be run directly while developing."""
    sys.stderr.write("this module is a library; use `forge calibrate`\n")
    return 2


__all__ = [
    "CALIBRATION_DIR", "CalibrationError", "DEFAULT_SECONDS", "DEFAULT_TIMEOUT_S",
    "Delivered", "LocalRun", "PROBES", "RemoteRun", "SCHEMA", "build", "calibrate",
    "default_out", "entry_script", "load", "parse_payload", "parse_probe_payload",
    "prepare_tree", "resolve_delivery", "summary_lines", "validate_entry",
]
