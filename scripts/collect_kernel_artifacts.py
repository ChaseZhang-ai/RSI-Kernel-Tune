#!/usr/bin/env python3
"""Collect the mechanical prior artifacts of the published kernels (WP18; A9).

For every kernel under ``kernels/`` this script gathers, on whatever GPU machine
it runs, the two measured structural faces the prior library lacked:

- an NCU key-metric reading (``ncu.json``) — the SOL / DRAM / L2 / occupancy
  face a recon or a run would see, taken through the kernel's own runner;
- a SASS instruction census (``sass_census.json``) — the shipped instruction
  mix (MMA / TMA / barrier / register-role families), read off the compiled
  cubin, no GPU execution needed.

Artifacts land in ``priors/artifacts/<family>/<op>_<gpu>/`` beside the prior
skeletons they anchor, and every file names its regime (the operator's
published shape domain from ``kernels/README.md``). ``scripts/build_priors.py
--from-artifacts`` turns the artifacts into real prior entries.

Lanes and their honest limits (each kernel's ``manifest.json`` says which ran):

- **ncu lane** — needs a GPU whose architecture matches the kernel's target
  (an sm90 kernel does not run on sm120) AND a per-kernel runner script
  ``ncu_runner.py`` in the kernel directory (the harness that builds the
  operator's inputs and launches the kernel; shapes are the operator's
  published ones and live with the runner). Neither is fabricable here, so on
  a mismatched machine the lane records ``skipped`` with the reason instead of
  inventing a reading.
- **sass lane** — compile-only: a ``.cu`` kernel's cubin is built with nvcc at
  the target arch (no GPU needed at build time) and disassembled with
  nvdisasm. A DSL kernel compiles at runtime on the device, so its census
  needs the matching GPU too; on this machine it records ``skipped``.

Usage::

    python3 scripts/collect_kernel_artifacts.py            # dry run: the plan
    python3 scripts/collect_kernel_artifacts.py --write    # collect what can run

Red lines (the prior library's, enforced by the consumers' tests): artifacts
are machine readings only — no prose, no decisions, no implementation bodies;
the narrative a human wants lives in the prior files, written from these
numbers, never copied from someone's kernel source.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNELS = REPO / "kernels"
README = KERNELS / "README.md"
ARTIFACTS = REPO / "priors" / "artifacts"

# The kernel's target architecture, from its directory name (kernels/README:
# H100 = sm90, B200 = sm100). A directory that names neither is not arch-gated
# (None = any local GPU may run it).
_ARCH_BY_GPU_NAME = {"h100": "sm90", "b200": "sm100"}

# The NCU metric set: the same structural face the run-time digests carry
# (both SOL sides, DRAM, L2 hit rate, occupancy/waves, the launch stats).
_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__grid_size",
    "launch__block_size",
]

_METRIC_KEY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_sol_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_sol_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "launch__registers_per_thread": "regs_per_thread",
    "launch__shared_mem_per_block_static": "smem_per_block",
    "launch__grid_size": "grid_size",
    "launch__block_size": "block_size",
}

# SASS opcode families the census counts (a prefix match on the mnemonic).
_SASS_FAMILIES = [
    ("mma", ("HMMA", "QMMA", "IMMA", "UMMA", "UTCMMA", "OMMA")),
    ("tma", ("TMA",)),
    ("cp_async", ("CP.ASYNC", "CP.BULK", "CP.DECOMM", "CP.COMMIT")),
    ("mbarrier", ("MBAR", "MEXIT", "FENCE")),
    ("bar_sync", ("BAR.", "BAR.WARP")),
    ("reg_role", ("SETMAXNREG",)),
    ("st_matrix", ("STSM", "LDSM")),
    ("ld_global", ("LDG", "LD.",)),
    ("st_global", ("STG", "ST.")),
    ("named_sync", ("GRIDDEPCONTROL", "REDUX", "ELECT")),
]


def _rows() -> dict[str, dict[str, str]]:
    """kernels/README.md table rows keyed by kernel directory — the regime /
    provenance metadata every artifact file carries."""
    out: dict[str, dict[str, str]] = {}
    if not README.is_file():
        return out
    row = re.compile(
        r"^\|(?P<op>[^|]+)\|(?P<gpu>[^|]+)\|(?P<vs>[^|]+)\|(?P<speedup>[^|]+)\|"
        r"(?P<source>[^|]+)\|\s*\[?`(?P<dir>[^`]+)`", re.M)
    for m in row.finditer(README.read_text(encoding="utf-8")):
        d = m.group("dir").strip().rstrip("/")
        out[f"kernels/{d}"] = {k: m.group(k).strip() for k in
                               ("op", "gpu", "vs", "speedup", "source")}
    return out


def _kernels() -> list[str]:
    """Published kernel directories (leaves holding kernel.py)."""
    return sorted(str(f.parent.relative_to(REPO)) for f in KERNELS.rglob("kernel.py"))


def _artifact_dir(kernel_dir: str) -> Path:
    """kernels/<family>/<op>/<gpu>[/<gen>] -> priors/artifacts/<family>/<op>_<gpu>[_<gen>]/
    (the same naming the prior skeletons use, minus the extension)."""
    parts = Path(kernel_dir).parts[1:]
    return ARTIFACTS / parts[0] / "_".join(parts[1:])


def _kernel_arch(kernel_dir: str) -> str | None:
    for name, arch in _ARCH_BY_GPU_NAME.items():
        if f"/{name}/" in f"/{kernel_dir}/":
            return arch
    return None


def _local_arch() -> str | None:
    """This machine's compute capability as the repo spells architectures
    (sm<major><minor>), or None when no GPU answers."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability()
        return f"sm{cap[0]}{cap[1]}"
    except Exception:  # noqa: BLE001 — no torch / no driver: the gate just fails
        return None


def _parse_ncu_csv(csv_text: str) -> dict:
    """NCU CSV (one row per metric) -> the key-metric dict the prior fill
    reads. Values become floats when they are numbers; the dict stays honest
    about what NCU did not report (absent keys)."""
    out: dict = {}
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return out
    header = next((ln for ln in lines if '"Metric Name"' in ln), None)
    if header is None:
        return out
    cols = _split_csv(header)
    try:
        mi, ui, vi = cols.index("Metric Name"), cols.index("Metric Unit"), cols.index("Metric Value")
    except ValueError:
        return out
    for ln in lines[lines.index(header) + 1:]:
        cells = _split_csv(ln)
        if len(cells) <= max(mi, ui, vi):
            continue
        name = cells[mi].strip()
        key = _METRIC_KEY.get(name)
        if key is None:
            continue
        raw = cells[vi].strip().replace(",", "")
        unit = cells[ui].strip().lower()
        try:
            val = float(raw)
        except ValueError:
            continue
        if key in ("smem_per_block",) and unit in ("kbyte", "kb"):
            val *= 1024.0
        out[key] = val
    return out


def _split_csv(line: str) -> list[str]:
    """One CSV line -> cells (quoted commas tolerated; NCU quotes every cell)."""
    if ',' not in line:
        return [line]
    cells, cur, q = [], "", False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            if q and i + 1 < len(line) and line[i + 1] == '"':
                cur += '"'
                i += 1
            else:
                q = not q
        elif ch == ',' and not q:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur)
    return cells


# --- SASS census ----------------------------------------------------------------

_INSTR_RE = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+([@!P0-9_\s]*)\s*([A-Z][A-Z0-9.]*)")


def census_from_sass(sass_text: str) -> dict:
    """The instruction-family histogram of one nvdisasm dump (a prefix match on
    the mnemonic, one family per instruction; unlisted opcodes land in
    ``other``). Machine facts only — the reading that anchors the prior's
    role / pipeline / residency sections."""
    fam: dict[str, int] = {}
    total = 0
    for ln in sass_text.splitlines():
        m = _INSTR_RE.match(ln)
        if not m:
            continue
        op = m.group(2).upper()
        total += 1
        for name, prefixes in _SASS_FAMILIES:
            if any(op.startswith(p) for p in prefixes):
                fam[name] = fam.get(name, 0) + 1
                break
        else:
            fam["other"] = fam.get("other", 0) + 1
    return {"total": total, "families": dict(sorted(fam.items(), key=lambda kv: -kv[1]))}


def _sass_lane(kernel_dir: str, arch: str | None, dst: Path, write: bool) -> dict:
    """The compile-only census: nvcc -cubin at the target arch, then nvdisasm.
    Only the CUDA sources a kernel directory carries can take this lane (a DSL
    kernel compiles on-device at runtime)."""
    sources = sorted((REPO / kernel_dir).glob("*.cu"))
    if not sources:
        return {"status": "skipped", "reason": "no .cu source (DSL kernels compile on-device; "
                                               "this lane needs the matching GPU)"}
    if shutil.which("nvcc") is None or shutil.which("nvdisasm") is None:
        return {"status": "skipped", "reason": "nvcc/nvdisasm not on PATH"}
    target = arch or _local_arch() or "sm90"
    # the published CUDA kernel is a torch extension: without its include paths
    # nvcc stops at the first <torch/extension.h>. Compile-only census still —
    # no device, no execution, headers only.
    includes: list[str] = []
    try:
        import sysconfig
        from torch.utils import cpp_extension
        includes = [f"-I{p}" for p in cpp_extension.include_paths()]
        includes.append(f"-I{sysconfig.get_paths()['include']}")
    except Exception:  # noqa: BLE001 — no torch: the census lane records the miss
        pass
    workdir = dst.parent / f".{dst.name}.work"
    out: dict = {}
    try:
        if write:
            workdir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            cubin = workdir / (src.stem + ".cubin")
            # nvcc's flag spelling is sm_90a / sm_100a (underscore; the a-suffix
            # for the arch-feature variants the published kernels use)
            nvcc_arch = target.replace("sm", "sm_") + ("a" if target in ("sm90", "sm100") else "")
            r = subprocess.run(
                ["nvcc", f"-arch={nvcc_arch}", "-std=c++17", *includes,
                 "-cubin", str(src), "-o", str(cubin)],
                capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                return {"status": "skipped", "reason": f"nvcc refused {src.name}: "
                                                       f"{(r.stderr or r.stdout).strip()[:200]}"}
            d = subprocess.run(["nvdisasm", "-c", str(cubin)],
                               capture_output=True, text=True, timeout=300)
            if d.returncode != 0:
                return {"status": "skipped", "reason": f"nvdisasm refused {cubin.name}: "
                                                       f"{(d.stderr or d.stdout).strip()[:200]}"}
            out[src.name] = census_from_sass(d.stdout)
        return {"status": "ok", "target": target, "kernels": out}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"status": "skipped", "reason": f"{e.__class__.__name__}: {e}"}
    finally:
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def _ncu_lane(kernel_dir: str, arch: str | None, local: str | None, write: bool) -> dict:
    """The profiled reading: the kernel's own ``ncu_runner.py`` under NCU.
    Needs the matching GPU (an arch gate first — no invented readings)."""
    runner = REPO / kernel_dir / "ncu_runner.py"
    if not runner.is_file():
        return {"status": "skipped", "reason": ("no ncu_runner.py in the kernel directory "
                                                "(write one on the GPU box; shapes are the "
                                                "operator's published ones)")}
    if arch is not None and local is not None and arch != local:
        return {"status": "skipped", "reason": f"kernel targets {arch}, this GPU is {local}"}
    if local is None:
        return {"status": "skipped", "reason": "no usable GPU on this machine"}
    if shutil.which("ncu") is None:
        return {"status": "skipped", "reason": "ncu not on PATH"}
    if not write:
        return {"status": "planned", "runner": runner.name}
    r = subprocess.run(
        ["ncu", "--csv", "--metrics", ",".join(_NCU_METRICS),
         sys.executable, str(runner)],
        capture_output=True, text=True, timeout=1800, cwd=str(REPO / kernel_dir))
    parsed = _parse_ncu_csv(r.stdout)
    if not parsed:
        return {"status": "skipped", "reason": (f"ncu produced no key metrics "
                                                f"(rc={r.returncode}): {(r.stderr or '')[:200]}")}
    return {"status": "ok", "metrics": parsed}


def _plan(write: bool) -> int:
    rows = _rows()
    kernels = _kernels()
    local = _local_arch()
    print(f"{len(kernels)} published kernel(s); local GPU arch: {local or 'none'}")
    landed = 0
    for k in kernels:
        arch = _kernel_arch(k)
        meta = rows.get(k, {})
        dst = _artifact_dir(k)
        plan = (f"{k} [{arch or 'arch-agnostic'}] -> {dst.relative_to(REPO)}"
                + (f" ({meta.get('speedup', '?')} vs {meta.get('vs', '?')})" if meta else ""))
        if not write:
            print(f"would collect {plan}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        ncu = _ncu_lane(k, arch, local, write)
        sass = _sass_lane(k, arch, dst, write)
        manifest = {
            "kernel": k,
            "target_arch": arch,
            "local_arch": local,
            "regime": (meta.get("op", "") + " — published: " + meta.get("speedup", "?")
                       + " vs " + meta.get("vs", "?") + " (" + meta.get("source", "?") + ")")
                      if meta else "",
            "ncu": {kk: vv for kk, vv in ncu.items() if kk != "metrics"},
            "ncu_metrics": ncu.get("metrics", {}),
            "sass": {kk: vv for kk, vv in sass.items() if kk != "kernels"},
            "sass_census": sass.get("kernels", {}),
        }
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        if ncu.get("status") == "ok":
            (dst / "ncu.json").write_text(json.dumps(ncu["metrics"], indent=1) + "\n",
                                          encoding="utf-8")
        if sass.get("status") == "ok":
            (dst / "sass_census.json").write_text(json.dumps(sass["kernels"], indent=1) + "\n",
                                                  encoding="utf-8")
        landed += 1
        print(f"collected {plan} — ncu {ncu.get('status')}, sass {sass.get('status')}")
    if write:
        print(f"{landed} manifest(s) written under {ARTIFACTS.relative_to(REPO)}; "
              f"fill the priors with scripts/build_priors.py --from-artifacts")
    else:
        print("dry run — pass --write to collect")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="collect_kernel_artifacts.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--write", action="store_true",
                   help="collect what this machine can (default: dry run, list only)")
    p.add_argument("--repo", default=None, type=Path,
                   help="the repository root (default: this script's repo; tests point it "
                        "at a fixture checkout)")
    args = p.parse_args(argv)
    if args.repo is not None:
        globals()["REPO"] = args.repo.resolve()
        globals()["KERNELS"] = REPO / "kernels"
        globals()["README"] = KERNELS / "README.md"
        globals()["ARTIFACTS"] = REPO / "priors" / "artifacts"
    return _plan(args.write)


if __name__ == "__main__":
    sys.exit(main())
