"""Probing the development machine: what it is, what it can compile, what it reads.

``forge target probe`` collects the machine it runs on -- this one, or the ssh host
``local.env`` names -- and writes ``target/dev.json``:

* the GPU as the driver reports it (``nvidia-smi --query-gpu`` and ``-q``: name, compute
  capability, SM count, memory, driver, current / application / max clocks, lock state),
* the software stack (python, torch + its CUDA, Triton and *which distribution* it came
  from -- the board runs ``fbtriton`` -- nvidia-cutlass-dsl, CUPTI, nvcc),
* three smokes (``smoke_sources``): Triton, CuTe DSL, inline CUDA, each reporting
  "compiles / runs / reads X ms" or why not.

The collection program is ``probe_program`` -- a real module, not text built at run time:

* **local**: called in this process, under the per-GPU lock a GPU job would take;
* **remote**: the same file is put into a job tree and run through ForgeKernel's existing
  ssh delivery (``submit.direct.DirectRunner``: git archive to ``REMOTE_WORKDIR``, the GPU
  lock there, the job's timeout and session handling, the log). Nothing here builds a shell
  command or executes generated code.

Parsing (``parse_report``) is pure and lives here, so both paths -- and a canned payload in a
test -- go through exactly the same normalization, and a missing ``nvidia-smi`` or a torch
that will not import is a reported field, not an exception.

``target/dev.json`` is not committed: a committed probe describes one machine and is then
read as if it described every machine.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forgekernel import target_profile
from forgekernel.targethw import arch_of_cc
from forgekernel.targethw import dev_arch as resolved_dev_arch
from forgekernel.targethw import parity, probe_program, rules, smoke_sources

SCHEMA = "forgekernel.target-dev/1"
PAYLOAD_MARKER = probe_program.PAYLOAD_MARKER
SMOKE_MARKER = probe_program.SMOKE_MARKER
DEFAULT_TIMEOUT_S = 900

PROBE_RUN = "_targetprobe"            # the run name of the remote job tree
PROBE_OP = "target-probe"             # the tree's operator name (its directory on the host)


class ProbeError(RuntimeError):
    """The probe could not be delivered or produced no readable report."""


# -- delivery --------------------------------------------------------------------------

@dataclass(frozen=True)
class Delivered:
    rc: int
    stdout: str
    stderr: str


class LocalProbe:
    """Collect this machine's facts in this process, under its GPU lock."""

    def __init__(self, gpu: str | None = None):
        self.gpu = gpu

    def run(self, *, timeout: int = DEFAULT_TIMEOUT_S,
            include_smokes: bool = True) -> Delivered:
        from forgekernel.submit.direct import gpu_lock

        gpu = self.gpu
        if gpu is None:
            try:
                from forgekernel.localenv import load_machine

                gpu = load_machine().gpu
            except Exception:  # noqa: BLE001 -- no local.env: the default device
                gpu = "0"
        try:
            with gpu_lock(gpu):
                payload = probe_program.collect(include_smokes=include_smokes)
        except Exception as e:  # noqa: BLE001 -- a hard failure is a probe result too
            return Delivered(1, "", f"{type(e).__name__}: {e}")
        return Delivered(0, probe_program.payload_line(payload), "")

    def describe(self) -> str:
        return "this machine"


_ENTRY = """#!/bin/bash
# The target probe's GPU job entry: one task, the probe program, in this directory.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
case "${1:-probe}" in
  probe) exec python3 {options}probe_program.py ;;
  *) echo "target probe: unknown task $1" >&2; exit 2 ;;
esac
"""


def prepare_tree(*, include_smokes: bool = True) -> tuple[Path, Path]:
    """A tiny job tree holding the probe program, committed on ``main``.

    ``submit.direct`` ships a tree with ``git archive`` and runs its ``gpu_entry.sh``;
    building the tree is all the remote probe does before handing over. It lives under a
    fresh temporary directory, so a probe leaves nothing behind.

    Returns ``(base, op_dir)``: the directory that holds the log, and the git tree.
    """
    from forgekernel.taskcard import _git_init

    base = Path(tempfile.mkdtemp(prefix="fk_target_probe_"))
    op_dir = base / PROBE_RUN / "operators" / PROBE_OP
    op_dir.mkdir(parents=True, exist_ok=True)
    (op_dir / "probe_program.py").write_text(program_source(), encoding="utf-8")
    entry = op_dir / "gpu_entry.sh"
    entry.write_text(_ENTRY.format(options="" if include_smokes else "--no-smokes "),
                     encoding="utf-8")
    entry.chmod(0o755)
    _git_init(op_dir, "Chore: target probe tree")
    return base, op_dir


class RemoteProbe:
    """Run the probe on the ssh host through the same delivery a GPU job uses.

    ``DirectRunner`` is what ``forge run`` uses for every GPU job: it archives the tree to
    ``REMOTE_WORKDIR`` over ssh, takes the host's per-GPU lock, runs ``gpu_entry.sh`` under
    the job timeout and writes the log. The probe is one more task in that machinery, not a
    second way to reach the host.
    """

    def __init__(self, machine, gpu: str | None = None, runner=None):
        self.machine = machine
        self.gpu = gpu if gpu is not None else machine.gpu
        self._runner = runner          # a factory(cfg) -> runner, for tests

    def config(self, op_dir: Path, timeout: int):
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

    def run(self, *, timeout: int = DEFAULT_TIMEOUT_S,
            include_smokes: bool = True) -> Delivered:
        from forgekernel.submit.direct import DirectRunner
        from forgekernel.submit.jobs import STATUS_SUCCEEDED

        try:
            base, op_dir = prepare_tree(include_smokes=include_smokes)
        except (OSError, RuntimeError) as e:
            return Delivered(1, "", f"could not build the probe tree: {e}")
        cfg = self.config(op_dir, timeout)
        runner = (self._runner or DirectRunner)(cfg)
        log_path = base / "probe.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                status = runner.run_mode("main", "probe", log)
        except Exception as e:  # noqa: BLE001 -- report it as a failed probe
            return Delivered(1, "", f"{type(e).__name__}: {e}")
        text = log_path.read_text(encoding="utf-8", errors="replace") \
            if log_path.is_file() else ""
        return Delivered(0 if status == STATUS_SUCCEEDED else 1, text, "")

    def describe(self) -> str:
        return f"ssh {self.machine.remote_host}"


# -- parsing ---------------------------------------------------------------------------

def parse_payload(stdout: str) -> dict:
    """The probe program's JSON document (the last ``PROBE_JSON`` line it printed)."""
    payload = None
    for line in (stdout or "").splitlines():
        if not line.startswith(PAYLOAD_MARKER):
            continue
        try:
            payload = json.loads(line[len(PAYLOAD_MARKER):])
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        tail = " / ".join((stdout or "").strip().splitlines()[-5:])
        raise ProbeError("the probe printed no JSON document: " + tail[:600])
    return payload


def _int_or_none(text):
    digits = "".join(c for c in str(text or "") if c.isdigit())
    return int(digits) if digits else None


def parse_nvidia_smi_query(text: str) -> dict:
    """``--query-gpu=... --format=csv,noheader,nounits`` -> a dict.

    A machine without ``nvidia-smi`` (or with one that answers nothing) yields {}; the
    caller then falls back to what torch reports.
    """
    line = next((ln for ln in (text or "").splitlines() if ln.strip()), "")
    if not line:
        return {}
    fields = [f.strip() for f in line.split(",")]
    names = ("name", "compute_capability", "memory_total_mib", "driver", "current_sm_mhz",
             "current_mem_mhz", "app_sm_mhz", "app_mem_mhz", "max_sm_mhz", "max_mem_mhz")
    raw = dict(zip(names, fields))
    out: dict = {}
    if raw.get("name"):
        out["name"] = raw["name"]
    cc = str(raw.get("compute_capability") or "")
    if "." in cc:
        major, _, minor = cc.partition(".")
        try:
            out["compute_capability"] = [int(major), int(minor)]
        except ValueError:
            pass
    for key in ("memory_total_mib", "current_sm_mhz", "current_mem_mhz", "app_sm_mhz",
                "app_mem_mhz", "max_sm_mhz", "max_mem_mhz"):
        value = _int_or_none(raw.get(key))
        if value is not None:
            out[key] = value
    if raw.get("driver"):
        out["driver"] = raw["driver"]
    return out


def parse_nvidia_smi_q(text: str) -> dict:
    """``nvidia-smi -q`` -> {clocks, applications, max_clocks, throttle} (best effort).

    The output is human-readable and versioned; only the sections this probe reasons about
    are read, and a missing section is simply absent (not an error).
    """
    sections: dict[str, dict] = {}
    current = None
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if not line.startswith("    "):
            current = sections.setdefault(line.strip().rstrip(":"), {})
            continue
        if current is None or ":" not in line:
            continue
        name, _, value = line.strip().partition(":")
        current[name.strip()] = value.strip()
    clocks = sections.get("Clocks", {})
    apps = sections.get("Applications Clocks", {})
    maxima = sections.get("Max Clocks", {})
    reasons = sections.get("Clocks Event Reasons", {})
    out = {
        "clocks": {"sm_mhz": _int_or_none(clocks.get("SM")),
                   "mem_mhz": _int_or_none(clocks.get("Memory"))},
        "applications": {"sm_mhz": _int_or_none(apps.get("Graphics")),
                         "mem_mhz": _int_or_none(apps.get("Memory"))},
        "max_clocks": {"sm_mhz": _int_or_none(maxima.get("SM"))},
        "throttle": {k: v for k, v in reasons.items() if v},
    }
    return out if any(any(v.values()) for v in out.values()) else {}


def parse_nvcc_version(text: str) -> dict:
    """``nvcc --version`` -> {"toolkit": "13.3", "release": "V13.3.73"} ({} when absent)."""
    out: dict = {}
    for line in (text or "").splitlines():
        if "Cuda compilation tools" in line and "release" in line:
            after = line.split("release", 1)[1]
            pieces = [p.strip() for p in after.split(",")]
            if pieces and pieces[0]:
                out["toolkit"] = pieces[0]
            for piece in pieces[1:]:
                if piece.startswith("V"):
                    out["release"] = piece
            break
    return out


def clocks_locked(clocks: dict) -> bool | None:
    """Is the SM clock pinned?

    ``nvidia-smi -lgc`` pins an application clock below the maximum; a machine that never
    touched the clocks reports the application clock at the maximum (or 0). So: locked when
    an application clock is set, the current clock matches it, and it is below the maximum.
    None when the three numbers do not decide it (reported as unknown rather than guessed).
    """
    app = clocks.get("app_sm_mhz")
    current = clocks.get("current_sm_mhz")
    maximum = clocks.get("max_sm_mhz")
    if not app or not current:
        return False if (maximum and current and current >= maximum) else None
    if not maximum or abs(current - app) > 50:
        return None
    return app < maximum


def parse_report(payload: dict, *, mode: str = "local",
                 collected_at: str | None = None) -> dict:
    """The probe program's document -> ``target/dev.json``.

    Everything here is a pure function of the payload, so canned outputs (a missing
    ``nvidia-smi``, a torch that will not import, a smoke that fails) are testable without a
    GPU.
    """
    host = payload.get("host") or {}
    python_info = payload.get("python_info") or {}
    query = payload.get("nvidia_smi_query") or {}
    query_text = query.get("out") if query.get("rc") == 0 else ""
    smi_query = parse_nvidia_smi_query(query_text or "")
    smi_q = parse_nvidia_smi_q((payload.get("nvidia_smi_q") or {}).get("out") or "")
    torch_info = python_info.get("torch") or {}
    torch_device = python_info.get("torch_device") or {}
    nvcc = parse_nvcc_version((payload.get("nvcc") or {}).get("out") or "")

    gpu: dict = {}
    if torch_device:
        gpu.update({
            "name": torch_device.get("name"),
            "compute_capability": list(torch_device.get("compute_capability") or []) or None,
            "multi_processor_count": torch_device.get("multi_processor_count"),
            "memory_total_mib": torch_device.get("memory_total_mib"),
            "l2_cache_mib": torch_device.get("l2_cache_mib"),
        })
    for key in ("name", "compute_capability", "memory_total_mib", "driver"):
        if gpu.get(key) in (None, "", []) and smi_query.get(key) not in (None, ""):
            gpu[key] = smi_query[key]
    if gpu.get("memory_total_mib"):
        gpu["memory_gib"] = round(float(gpu["memory_total_mib"]) / 1024.0, 1)
    clocks = {
        "sm_mhz": smi_query.get("current_sm_mhz")
        or (smi_q.get("clocks") or {}).get("sm_mhz"),
        "mem_mhz": smi_query.get("current_mem_mhz")
        or (smi_q.get("clocks") or {}).get("mem_mhz"),
        "app_sm_mhz": smi_query.get("app_sm_mhz"),
        "app_mem_mhz": smi_query.get("app_mem_mhz"),
        "max_sm_mhz": smi_query.get("max_sm_mhz")
        or (smi_q.get("max_clocks") or {}).get("sm_mhz"),
    }
    if any(value is not None for value in clocks.values()):
        gpu["clocks"] = clocks
        gpu["clocks_locked"] = clocks_locked(clocks)
    if smi_q.get("throttle"):
        gpu["throttle"] = smi_q["throttle"]
    cc = gpu.get("compute_capability")
    if isinstance(cc, (list, tuple)) and len(cc) == 2:
        gpu["compute_capability"] = [int(cc[0]), int(cc[1])]
        gpu["arch"] = arch_of_cc(int(cc[0]), int(cc[1]))
        gpu["cuda_arch"] = resolved_dev_arch({}, fallback=gpu["arch"])
    if not gpu and torch_info.get("error"):
        gpu["error"] = torch_info["error"]

    triton_info = python_info.get("triton") or {}
    dsl_info = python_info.get("cutlass_dsl") or {}
    cupti_info = python_info.get("cupti") or {}
    distributions = list(triton_info.get("distributions") or [])
    software = {
        "python": host.get("python") or python_info.get("python"),
        "torch": torch_info.get("version"),
        "torch_cuda": torch_info.get("cuda"),
        "triton": triton_info.get("version"),
        "triton_distribution": ("fbtriton" if "fbtriton" in distributions
                                else ("triton" if distributions else None)),
        "triton_is_fbtriton": "fbtriton" in distributions,
        "cutlass_dsl": dsl_info.get("version"),
        "cupti": cupti_info.get("version"),
        "cupti_checked": True,
        "cuda_toolkit": nvcc.get("toolkit")
        or (python_info.get("cuda_toolkit") or {}).get("version"),
        "nvcc": nvcc.get("release") or nvcc.get("toolkit"),
        "cuda_home": (payload.get("env") or {}).get("CUDA_HOME")
        or (payload.get("env") or {}).get("CUDA_PATH"),
    }
    if torch_info.get("error"):
        software["torch_error"] = torch_info["error"]
    if triton_info.get("error"):
        software["triton_error"] = triton_info["error"]
    if dsl_info.get("error"):
        software["cutlass_dsl_error"] = dsl_info["error"]

    from forgekernel.targethw import find_cutlass, find_nvcc

    smokes = payload.get("smokes") or {}
    nvcc_path, cutlass = find_nvcc(), find_cutlass()
    return {
        "schema": SCHEMA,
        "collected_at": collected_at,
        "mode": mode,
        "host": {k: host.get(k) for k in ("hostname", "platform", "python", "executable")},
        "gpu": gpu,
        "software": software,
        "smokes": {name: _normalize_smoke(smokes.get(name))
                   for name in ("triton", "cutedsl", "cuda_inline")},
        "tools": {"nvcc": str(nvcc_path) if nvcc_path else None,
                  "cutlass": str(cutlass) if cutlass else None},
        "env": payload.get("env") or {},
        "probe": {"rc": payload.get("_rc"), "stderr_tail": payload.get("_stderr_tail", "")},
    }


def _normalize_smoke(value) -> dict:
    if not isinstance(value, dict):
        return {"compile": False, "run": False, "ms": None,
                "error": "the probe reported no result for this smoke", "detail": {}}
    out = {
        "compile": bool(value.get("compile")),
        "run": bool(value.get("run")),
        "ms": value.get("ms") if isinstance(value.get("ms"), (int, float)) else None,
        "error": str(value.get("error") or ""),
        "detail": value.get("detail") or {},
    }
    if value.get("stderr_tail"):
        out["stderr_tail"] = value["stderr_tail"]
    return out


# -- the command -----------------------------------------------------------------------

def probe(*, deliver=None, remote: bool = False, out: str | Path | None = None,
          root: str | Path | None = None, profile: dict | None = None,
          include_smokes: bool = True, timeout: int = DEFAULT_TIMEOUT_S,
          collected_at: str | None = None) -> dict:
    """Probe, compare against the target profile, write ``target/dev.json``, return it."""
    if deliver is None:
        if remote:
            from forgekernel.localenv import load_machine

            deliver = RemoteProbe(load_machine(root))
        else:
            deliver = LocalProbe()
    try:
        result = deliver.run(timeout=timeout, include_smokes=include_smokes)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"the probe on {deliver.describe()} exceeded {timeout}s") from None
    except OSError as e:
        raise ProbeError(f"the probe could not be delivered to {deliver.describe()}: "
                         f"{e.strerror or e}") from None
    payload = parse_payload(result.stdout)
    payload["_rc"] = result.rc
    payload["_stderr_tail"] = " / ".join((result.stderr or "").strip().splitlines()[-5:])[:1200]
    dev = parse_report(payload, mode="remote" if remote else "local", collected_at=collected_at)
    dev["delivery"] = deliver.describe()
    profile = profile if profile is not None else target_profile.load(root)
    differences = parity.compare(dev, profile)
    dev["differences"] = parity.lines(differences, levels=(parity.BLOCKER, parity.RISK))
    dev["parity"] = parity.summary(differences)
    path = Path(out) if out is not None else target_profile.dev_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dev, indent=2) + "\n", encoding="utf-8")
    dev["path"] = str(path)
    dev["_differences"] = [d.to_dict() for d in differences]
    return dev


def program_source() -> str:
    """The probe program exactly as it is shipped to a machine (for inspection)."""
    return Path(probe_program.__file__).read_text(encoding="utf-8")


def confidence_hints(dev: dict | None = None) -> list[str]:
    """The confidence labels a probe result forces on every local reading."""
    hints = ["dev-only"]
    if not dev:
        return hints
    gpu = dev.get("gpu") or {}
    if gpu.get("clocks_locked") is not True:
        hints.append("clocks-unlocked")
    return hints


__all__ = [
    "DEFAULT_TIMEOUT_S", "Delivered", "LocalProbe", "PROBE_OP", "PROBE_RUN", "ProbeError",
    "RemoteProbe", "clocks_locked", "confidence_hints", "parse_nvcc_version",
    "parse_nvidia_smi_q", "parse_nvidia_smi_query", "parse_payload", "parse_report",
    "prepare_tree", "probe", "program_source", "rules", "smoke_sources",
]
