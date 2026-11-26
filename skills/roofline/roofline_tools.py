#!/usr/bin/env python3
"""Roofline arithmetic and NCU-signal classification -- one implementation, two callers.

``forge roofline`` (``rsi_kernel_tune.roofline.analyze``) and this skill share the numbers:
the floor arithmetic below is the single implementation of

    t_mem    = bytes / (peak_bandwidth x efficiency)
    t_tensor = flops / (peak_throughput x efficiency)
    t_floor  = max(t_mem, t_tensor, t_launch)

and of the bottleneck class, the headroom grade and the distance ratios.  The skill
imports nothing from this project, so a GPU job can run it as
``python3 notes/roofline_tools.py distance ...``; ``forge roofline`` imports THIS file
(``from rsi_kernel_tune.skills import roofline_tools``, the same path the harness already
uses for ``sass_tools``), so the two cannot drift apart.

Three subcommands, all pure (no GPU, no network):

  classify  --metrics "name=value,..." | --csv FILE | --json FILE
            A profile's Speed-of-Light numbers -> the bottleneck class, with every
            number the verdict used.  Thresholds follow KernelAgent's
            ``_classify_bottleneck`` (both SOL percentages under 60 = neither resource
            saturated) refined by the stall evidence the ncu-report-skill's dimensions
            single out (DRAM under 10% with long-scoreboard stalls dominating = a
            latency-bound kernel, not a bandwidth-bound one).

  suggest   [--signal ID] [--list]
            The hardware-level playbook: NCU signal -> likely cause -> candidate
            action.  Ported from ncu-report-skill's diagnosis playbook; every row is
            (signal, cause, action) and none names an operator.

  distance  --calibration FILE --flops F --bytes-read B --bytes-written W [...]
            The same distance ``forge roofline`` computes for one workload: the floor,
            its bottleneck class and how far a reading is from it.

What this skill is NOT: a replacement for a profile or for the probes.  It maps numbers
you already measured to a first-order verdict; it never measures.  Nothing here may be
copied into ``hw_kb/`` -- entries there are certified by a probe, and a heuristic is
not a certification.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys

# -- units and defaults ---------------------------------------------------------------

GB = 1_000_000_000.0        # GB/s -> bytes/s (SI, as every vendor's peak bandwidth)
TFLOPS = 1_000_000_000_000.0
# Efficiency: the share of a peak a well-written kernel reaches.  AutoKernel's
# calibrate.py reports real kernels at "roughly 80-95% of the manual peak"; 0.85 is the
# middle of that band for a bandwidth-bound path.  A tensor path runs lower (the
# vendor peak assumes a perfect tile and no operand traffic): 0.80 of the dense peak is
# a large, well-tiled kernel's habitual ceiling.  Both are CLI-settable where they matter.
DEFAULT_EFF_HBM = 0.85
DEFAULT_EFF_TENSOR = 0.80

# Headroom grades: near the floor (a reading within 10% of it), in between, far (over 2x).
NEAR_FLOOR = 1.10
FAR = 2.0

# A pairwise difference under 10% of the larger term is not a bottleneck: both terms bind.
BALANCE_BAND = 0.10

CLASSES = ("memory", "compute", "latency", "balanced")

# -- the floor arithmetic -------------------------------------------------------------


def t_mem_ms(bytes_total: float, hbm_gb_s: float, eff: float = DEFAULT_EFF_HBM) -> float:
    """Milliseconds the bytes take at ``eff`` of the peak device bandwidth."""
    check_number("bytes_total", bytes_total, minimum=0.0)
    check_number("hbm_gb_s", hbm_gb_s, minimum=0.0, exclusive=True)
    check_number("eff", eff, minimum=0.0, maximum=1.0, exclusive=True)
    return bytes_total / (hbm_gb_s * GB * eff) * 1e3


def t_tensor_ms(flops: float, tflops: float, eff: float = DEFAULT_EFF_TENSOR) -> float:
    """Milliseconds the FLOPs take at ``eff`` of the peak tensor throughput."""
    check_number("flops", flops, minimum=0.0)
    check_number("tflops", tflops, minimum=0.0, exclusive=True)
    check_number("eff", eff, minimum=0.0, maximum=1.0, exclusive=True)
    return flops / (tflops * TFLOPS * eff) * 1e3


def floor_ms(t_mem: float, t_tensor: float, t_launch: float = 0.0) -> float:
    """The shortest a call can take: the slowest of its three floors.

    Raises ``ValueError`` when every term is zero: a cost model that counts no work and
    no launch cannot be turned into a floor (and silently answering 0 ms would make
    every ratio infinite).
    """
    for name, number in (("t_mem", t_mem), ("t_tensor", t_tensor), ("t_launch", t_launch)):
        check_number(name, number, minimum=0.0)
    floor = max(t_mem, t_tensor, t_launch)
    if floor <= 0.0:
        raise ValueError("every floor term is zero: the cost model counts no work and no "
                         "launch latency")
    return floor


def bottleneck(t_mem: float, t_tensor: float, t_launch: float = 0.0,
               band: float = BALANCE_BAND) -> str:
    """Which term sets the floor: ``memory`` / ``compute`` / ``latency`` / ``balanced``.

    ``latency`` when the launch floor is strictly the largest term (the kernel is too
    small to be limited by either throughput); ``balanced`` when the two throughput terms
    are within ``band`` of each other (the larger one); otherwise the larger of the two.
    """
    if t_launch > max(t_mem, t_tensor):
        return "latency"
    high, low = max(t_mem, t_tensor), min(t_mem, t_tensor)
    if high > 0.0 and (high - low) <= band * high:
        return "balanced"
    return "memory" if t_mem >= t_tensor else "compute"


def headroom(reading_ms: float | None, floor: float, *, near: float = NEAR_FLOOR,
             far: float = FAR) -> str | None:
    """``near-floor`` (within ``near`` x), ``mid``, ``far`` (over ``far`` x), or None
    without a reading."""
    if reading_ms is None:
        return None
    ratio = reading_ms / floor
    if ratio <= near:
        return "near-floor"
    if ratio > far:
        return "far"
    return "mid"


def distance(*, flops: float, bytes_read: float, bytes_written: float,
             hbm_gb_s: float | None, tflops: float | None,
             t_launch_ms: float = 0.0, eff_hbm: float = DEFAULT_EFF_HBM,
             eff_tensor: float = DEFAULT_EFF_TENSOR,
             current_ms: float | None = None) -> dict:
    """One workload's floor, class and distance -- the numbers both callers report.

    A peak the cost model needs but the caller does not have is a ``ValueError`` naming
    it (never a default: an unmeasured peak is not a measurement).
    """
    check_number("bytes_read", bytes_read, minimum=0.0)
    check_number("bytes_written", bytes_written, minimum=0.0)
    check_number("flops", flops, minimum=0.0)
    check_number("t_launch_ms", t_launch_ms, minimum=0.0)
    total_bytes = bytes_read + bytes_written
    if total_bytes > 0.0 and hbm_gb_s is None:
        raise ValueError("the cost model moves bytes but no device-bandwidth peak is "
                         "available: measure one with `forge calibrate` (hbm probe)")
    if flops > 0.0 and tflops is None:
        raise ValueError("the cost model counts FLOPs but no compute peak is available "
                         "for the precision it names")
    t_mem = t_mem_ms(total_bytes, hbm_gb_s, eff_hbm) if total_bytes > 0.0 else 0.0
    t_tensor = t_tensor_ms(flops, tflops, eff_tensor) if flops > 0.0 else 0.0
    floor = floor_ms(t_mem, t_tensor, t_launch_ms)
    out = {
        "bytes_total": total_bytes,
        "t_mem_ms": t_mem,
        "t_tensor_ms": t_tensor,
        "t_launch_ms": t_launch_ms,
        "t_floor_ms": floor,
        "bottleneck": bottleneck(t_mem, t_tensor, t_launch_ms),
        "current_ms": current_ms,
        "current_over_floor": (current_ms / floor) if current_ms is not None else None,
        "headroom": headroom(current_ms, floor),
    }
    return out


# -- calibration files ----------------------------------------------------------------
#
# A calibration document is what `forge calibrate` writes (the JSON schema lives in
# rsi_kernel_tune.roofline.calibrate, which builds it).  Only the peaks are read here, so a
# GPU job that has the file but not this project can still compute a distance from it.

# the tensor-precision name a cost model uses -> the probe entry the calibration holds
PEAK_ENTRY = {"bf16": "bf16", "fp16": "fp16", "fp8": "fp8", "fp4": "fp4"}


def read_calibration(path) -> dict:
    """A calibration JSON document (a ``str``/``Path``), validated enough to read peaks."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("probes"), dict):
        raise ValueError(f"{path}: not a calibration document (no probes block)")
    if not doc.get("gpu"):
        raise ValueError(f"{path}: the calibration names no GPU")
    return doc


def calibration_entry(doc: dict, probe: str, entry: str) -> dict | None:
    """One probe entry (``{"value": .., "unit": .., ...}``) or None when it is absent."""
    probes = doc.get("probes") or {}
    entries = (probes.get(probe) or {}).get("entries") or {}
    value = entries.get(entry)
    return value if isinstance(value, dict) else None


def calibration_peaks(doc: dict) -> dict:
    """The peaks a calibration document holds, in the units the analysis uses.

    ``hbm_gb_s`` and ``tensor`` are GB/s and TFLOPS (``sfu_gop_s`` is Gop/s); an entry
    that is missing or was skipped stays None -- an absent peak is never a default.
    """
    hbm = calibration_entry(doc, "hbm", "copy") or calibration_entry(doc, "hbm", "read")
    tensor = {}
    for peak, entry in PEAK_ENTRY.items():
        value = calibration_entry(doc, "tensor", entry)
        if value and isinstance(value.get("value"), (int, float)) and not value.get("skipped"):
            tensor[peak] = float(value["value"])
    sfu = calibration_entry(doc, "sfu", "exp")
    launch = calibration_entry(doc, "launch", "roundtrip")
    l2 = calibration_entry(doc, "l2", "copy")
    launched = calibration_entry(doc, "launch", "min_kernel")
    return {
        "gpu": doc.get("gpu"),
        "sm_count": doc.get("sm_count"),
        "ts": doc.get("ts"),
        "delivery": doc.get("delivery", ""),
        "hbm_gb_s": float(hbm["value"]) if hbm and not hbm.get("skipped") else None,
        "l2_gb_s": float(l2["value"]) if l2 and not l2.get("skipped") else None,
        "tensor": tensor,
        "sfu_gop_s": (float(sfu["value"]) if sfu and not sfu.get("skipped") else None),
        "launch_us": (float(launch["value"]) if launch and not launch.get("skipped") else None),
        "min_kernel_us": (float(launched["value"]) if launched and not launched.get("skipped")
                          else None),
        "sources": {key: f"measured ({doc.get('ts', 'unknown time')})"
                    for key in ("hbm_gb_s", "l2_gb_s", "sfu_gop_s", "launch_us")},
    }


def peak_per_second(peaks: dict, peak: str) -> float | None:
    """The ops/second a ``peak`` name runs at, from a ``calibration_peaks`` dict:
    a tensor precision in TFLOPS x 1e12, or the special-function rate in ops/s."""
    if peak == "sfu":
        rate = peaks.get("sfu_gop_s")
        return rate * 1e9 if rate else None
    tflops = (peaks.get("tensor") or {}).get(peak)
    return tflops * TFLOPS if tflops else None


# -- classify: NCU Speed-of-Light + stall evidence ------------------------------------

# The metric names this tool understands, new (ncu 2026.x on Blackwell) first and the
# older spellings after -- 08-metric-names.md in ncu-report-skill lists both, and a
# report exported from ncu 2024 must still classify.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "compute_sol": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_active",
    ),
    "memory_sol": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
        # the DRAM-only throughput: what a pure bandwidth-bound kernel reads
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    ),
    "tensor_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    ),
    "long_scoreboard": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct",
    ),
    "short_scoreboard": (
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__warps_issue_stalled_short_scoreboard_per_issue_active.pct",
    ),
    "barrier": (
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__warps_issue_stalled_barrier_per_issue_active.pct",
    ),
    "samples": ("smsp__pcsamp_sample_count",),
    "long_scoreboard_samples": ("smsp__pcsamp_warps_issue_stalled_long_scoreboard",),
    "dram_pct": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    ),
}

# Both SOL percentages under this = neither resource is saturated (KernelAgent's
# RooflineConfig.underutilized_threshold).
UNDERUTILIZED_PCT = 60.0
# Tensor-pipe activity above this counts as "tensor cores in use" (KernelAgent: 5%).
TENSOR_USE_PCT = 5.0
# "Not bandwidth-bound": DRAM under this, with long-scoreboard stalls dominating the
# samples, is Pattern E of the diagnosis playbook (memory-LATENCY bound).
DRAM_LOW_PCT = 10.0
LONG_SCOREBOARD_DOMINANT = 0.40
# A stall ratio (warps stalled per issue) above this is Pattern E's aggregate form.
LONG_SCOREBOARD_RATIO = 3.0

_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def check_number(name: str, value, *, minimum: float | None = None,
                  maximum: float | None = None, exclusive: bool = False) -> None:
    """``value`` must be a finite number within the bounds (bool is not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        if exclusive and not value > minimum:
            raise ValueError(f"{name} must be > {minimum}, got {value!r}")
        if not exclusive and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")


def _number(text) -> float | None:
    """A metric cell as a number: NCU writes ``"93.42"``, ``"93.42 %"``, ``"1,234"``."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUMBER_RE.match(str(text or "").replace(",", "").strip())
    return float(match.group(0)) if match else None


def parse_metrics_text(text: str) -> dict[str, float]:
    """``"name=value, name=value"`` / newline / ``name: value`` -> {name: value}.

    A cell that is not a number is reported as a ``ValueError`` naming it: a profile
    metric read as 0 would classify the kernel on a number it never had.
    """
    out: dict[str, float] = {}
    for chunk in re.split(r"[,;\n]+", text or ""):
        if not chunk.strip():
            continue
        name, sep, cell = chunk.partition("=")
        if not sep:
            name, sep, cell = chunk.partition(":")
        if not sep or not name.strip():
            raise ValueError(f"metric {chunk.strip()!r} is not name=value")
        value = _number(cell)
        if value is None:
            raise ValueError(f"metric {name.strip()!r} has a non-numeric cell {cell!r}")
        out[name.strip()] = value
    return out


def parse_ncu_csv(text: str, *, kernel: str | None = None) -> dict[str, float]:
    """An ``ncu --import ... --csv`` export -> {metric name: number}.

    NCU repeats a kernel's metric block per launch. Additive (``.sum``) metrics are
    summed over the launches of one kernel; the others take their last value, which is
    what a per-launch intensity means when the launches are the same kernel. With
    ``kernel`` only rows whose ``Kernel Name`` matches (substring) are read.
    """
    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames or []
    metric_col = next((f for f in fields if f and f.strip().lower() == "metric name"), None)
    value_col = next((f for f in fields if f and f.strip().lower() == "metric value"), None)
    kernel_col = next((f for f in fields if f and f.strip().lower() == "kernel name"), None)
    if not metric_col or not value_col:
        raise ValueError("not an ncu --csv export: no 'Metric Name' / 'Metric Value' column")
    sums: dict[str, float] = {}
    last: dict[str, float] = {}
    seen = False
    for row in reader:
        name = (row.get(metric_col) or "").strip()
        if not name:
            continue
        if kernel and kernel_col and kernel not in (row.get(kernel_col) or ""):
            continue
        value = _number(row.get(value_col))
        if value is None:
            continue
        seen = True
        if name.endswith(".sum"):
            sums[name] = sums.get(name, 0.0) + value
        last[name] = value
    if not seen:
        raise ValueError("the ncu CSV export holds no numeric metric row"
                         + (f" for kernel {kernel!r}" if kernel else ""))
    return {**last, **sums}


def read_metrics(*, text: str | None = None, csv_path: str | None = None,
                 json_path: str | None = None, kernel: str | None = None) -> dict[str, float]:
    """The metric dict from exactly one of the three sources."""
    given = [bool(text), bool(csv_path), bool(json_path)]
    if sum(given) != 1:
        raise ValueError("give exactly one of --metrics, --csv, --json")
    if text:
        return parse_metrics_text(text)
    if csv_path:
        with open(csv_path, encoding="utf-8") as fh:
            return parse_ncu_csv(fh.read(), kernel=kernel)
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path}: not a metric object")
    return {str(k): _number(v) for k, v in data.items() if _number(v) is not None}


def classify(metrics: dict[str, float], *, underutilized_pct: float = UNDERUTILIZED_PCT,
             tensor_use_pct: float = TENSOR_USE_PCT) -> dict:
    """Bottleneck class of one profile, with every number the verdict used.

    ``compute_sol`` / ``memory_sol``: the SM and the memory pipeline's throughput as a
    percentage of peak (ncu's Speed-of-Light headline pair).  Both under
    ``underutilized_pct`` = neither is saturated.  Within that, a kernel whose DRAM
    throughput is under 10% and whose stalls are dominated by long-scoreboard waits is
    *latency*-bound on memory, not bandwidth-bound -- the distinction the diagnosis
    playbook's Pattern E draws, and the one that decides between "move fewer bytes" and
    "keep more requests in flight".
    """
    resolved: dict[str, float] = {}
    absent: list[str] = []
    used: dict[str, str] = {}
    for key, aliases in METRIC_ALIASES.items():
        hit = next(((a, metrics[a]) for a in aliases if a in metrics), None)
        if hit is None:
            absent.append(key)
        else:
            used[key] = hit[0]
            resolved[key] = float(hit[1])
    # A missing SOL percentage is read as 0 and says so; the other metrics are optional
    # evidence, listed apart so a warning names a real gap only.
    missing = [key for key in ("compute_sol", "memory_sol") if key in absent]
    missing_optional = [key for key in absent if key not in ("compute_sol", "memory_sol")]
    if len(missing) == 2:
        raise ValueError("neither SOL metric is present: give "
                         + " or ".join(METRIC_ALIASES["compute_sol"][:1])
                         + " and " + METRIC_ALIASES["memory_sol"][0])
    compute = resolved.get("compute_sol", 0.0)
    memory = resolved.get("memory_sol", 0.0)
    dram = resolved.get("dram_pct", memory)
    efficiency = max(compute, memory)

    stall_share = None
    samples = resolved.get("samples")
    long_samples = resolved.get("long_scoreboard_samples")
    if samples and long_samples is not None and samples > 0:
        stall_share = long_samples / samples
    long_ratio = resolved.get("long_scoreboard")

    evidence = {
        "compute_sol_pct": compute,
        "memory_sol_pct": memory,
        "efficiency_pct": efficiency,
        "dram_pct": dram,
        "underutilized_pct": underutilized_pct,
        "tensor_cores": resolved.get("tensor_pct", 0.0) > tensor_use_pct,
        "gap_pct": abs(compute - memory),
        "missing": missing,
        "missing_optional": missing_optional,
        # which metric name supplied each number (an old spelling, or the DRAM-only
        # memory metric): the verdict is checkable against the report it came from
        "metric_names": used,
    }
    if stall_share is not None:
        evidence["long_scoreboard_share"] = stall_share
    if long_ratio is not None:
        evidence["long_scoreboard_ratio"] = long_ratio

    if compute < underutilized_pct and memory < underutilized_pct:
        latency = (dram < DRAM_LOW_PCT and (
            (stall_share is not None and stall_share >= LONG_SCOREBOARD_DOMINANT)
            or (long_ratio is not None and long_ratio >= LONG_SCOREBOARD_RATIO)))
        cls = "latency" if latency else "underutilized"
        evidence["reason"] = ("both throughputs under the threshold; "
                              + ("DRAM idle and long-scoreboard stalls dominate"
                                 if latency else "neither resource saturated"))
    elif high := max(compute, memory):
        high_low = high - min(compute, memory)
        if high_low <= BALANCE_BAND * high:
            cls = "balanced"
            evidence["reason"] = f"the two SOL percentages are within {BALANCE_BAND:.0%} of each other"
        elif memory > compute:
            cls = "memory"
            evidence["reason"] = "the memory pipeline is the closer to its peak"
        else:
            cls = "compute"
            evidence["reason"] = "the SM (arithmetic) pipeline is the closer to its peak"
    else:                                              # pragma: no cover - defensive
        cls = "underutilized"
        evidence["reason"] = "no throughput reported"
    return {"class": cls, "evidence": evidence}


# -- suggest: the hardware-level playbook ---------------------------------------------

# (id, signal as an NCU reading, likely cause, candidate action).  Every row is
# hardware-level and operator-agnostic: it says which resource or which instruction
# pattern to look at, never what the kernel computes.  Ported from ncu-report-skill's
# reference/06-diagnosis-playbook.md (MIT); see NOTICE.
SUGGESTIONS: tuple[dict, ...] = (
    {"id": "small_grid",
     "signal": "launch__grid_size below device__attribute_multiprocessor_count, or "
               "launch__waves_per_multiprocessor < 0.5",
     "cause": "fewer thread blocks than SMs: some SMs never receive work",
     "action": "add a parallel dimension (a split along a reduction axis or a second "
               "output axis), or use a persistent grid of one block per SM with a "
               "work counter"},
    {"id": "tail_effect",
     "signal": "launch__waves_per_multiprocessor slightly above 1 with a partial last "
               "wave, or per-SM active cycles spanning several times the mean",
     "cause": "per-block work varies with an input axis; a few blocks keep running "
              "after the rest of the machine is idle",
     "action": "chunk long work items across more blocks (split factor + a small "
               "combine step), or dispatch short and long items to different paths"},
    {"id": "low_occupancy",
     "signal": "sm__warps_active.avg.pct_of_peak_sustained_active far below "
               "sm__maximum_warps_per_active_cycle_pct",
     "cause": "the resident-warp budget is not reached: either a limit (registers, "
              "shared memory, block size) or stalls that empty the slots",
     "action": "read launch__occupancy_limit_registers / _shared_mem / _warps; relax "
               "the tightest limit, then re-check whether stalls (not the limit) are "
               "what keeps warps out"},
    {"id": "long_scoreboard",
     "signal": "smsp__pcsamp_warps_issue_stalled_long_scoreboard over 40% of "
               "smsp__pcsamp_sample_count, or the _per_issue_active ratio above 3",
     "cause": "warps wait on global-memory loads returning before they can issue; too "
              "few requests in flight per warp",
     "action": "issue independent loads before using any of them (unroll, vectorize), "
               "raise occupancy, or stage the data through shared memory / TMA-style "
               "bulk async copies so the wait overlaps arithmetic"},
    {"id": "short_scoreboard",
     "signal": "short-scoreboard stalls lead the per-instruction samples while "
               "long-scoreboard is small",
     "cause": "a dependency chain through shared or local memory, or a long arithmetic "
              "chain, has nothing to overlap",
     "action": "shorten the chain (more accumulators, tree-shaped reductions) or keep "
               "the state in registers across steps instead of round-tripping memory"},
    {"id": "barrier",
     "signal": "smsp__pcsamp_warps_issue_stalled_barrier over 20% of samples",
     "cause": "a block-wide barrier makes every warp wait for the slowest one",
     "action": "replace the block barrier with warp-level primitives where the scope "
               "allows it, consolidate phases, or split producer and consumer roles "
               "with mbarrier hand-offs"},
    {"id": "math_pipe_throttle",
     "signal": "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio "
               "high, with sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active "
               "near its peak",
     "cause": "the scalar arithmetic pipe is saturated -- the kernel is genuinely "
              "compute-bound on that pipe",
     "action": "move the arithmetic to the tensor-core path if the shape allows a "
               "tiled product, or trade FLOPs for fewer, wider operations"},
    {"id": "tensor_cores_idle",
     "signal": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed at 0% "
               "on a kernel whose FLOPs are mostly products and sums",
     "cause": "the arithmetic runs on the scalar pipes; the tensor path needs a tiled "
              "shape and the right instruction class",
     "action": "restructure around a tensor-core tile shape, or call a tuned library "
               "kernel for that stage instead of hand-rolling it"},
    {"id": "uncoalesced_loads",
     "signal": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum divided by "
               "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum well above 4/8/16 for "
               "4/8/16-byte loads",
     "cause": "lanes of a warp touch non-adjacent addresses, so the fabric fetches "
              "sectors only a few lanes use",
     "action": "change the thread-to-element mapping so consecutive lanes read "
               "consecutive addresses, or stage through shared memory and vectorize "
               "the per-thread width"},
    {"id": "sparse_stores",
     "signal": "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio below "
               "16 (32 is ideal)",
     "cause": "only part of each warp writes, or the writes are not contiguous",
     "action": "pack the results in registers or shared memory first, then have "
               "consecutive lanes write one contiguous run"},
    {"id": "bank_conflicts",
     "signal": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum large, with "
               "short-scoreboard stalls on the shared-memory lines",
     "cause": "lanes of a warp hit the same shared-memory banks, which serialize",
     "action": "pad the tile's leading dimension, swizzle the index by XOR, or change "
               "the access order so a warp spans distinct banks"},
    {"id": "register_spill",
     "signal": "smsp__sass_inst_executed_op_local_ld.sum or _local_st.sum above 0, or "
               "launch__registers_per_thread above 128",
     "cause": "the compiler ran out of registers and spilled to local memory, which is "
              "device-memory backed",
     "action": "cap the footprint with launch bounds, recompute instead of caching, or "
               "move per-thread arrays to shared memory"},
    {"id": "no_pipeline",
     "signal": "the PM timeline of sm__throughput and dram__throughput alternates "
               "(sawtooth) instead of overlapping",
     "cause": "a single buffered stage: load a tile, wait, compute it, load the next",
     "action": "double-buffer (at least two stages) and prefetch the next tile while "
               "the current one is being computed"},
    {"id": "fp64_used",
     "signal": "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active above 0 "
               "in code that should be single precision",
     "cause": "an unsuffixed floating-point literal or a double-typed temporary "
              "promotes the expression",
     "action": "suffix literals with f, use the fast single-precision intrinsics for "
               "transcendentals, and check the intermediate types"},
    {"id": "warp_divergence",
     "signal": "smsp__thread_inst_executed_per_inst_executed.ratio far below 32",
     "cause": "lanes take different branches, so the warp executes both sides",
     "action": "reorder or partition the data so a warp takes one path, or make the "
               "branchless form cheap enough to always execute"},
    {"id": "serial_chain",
     "signal": "wait and short-scoreboard stalls lead the samples, issue-active is low "
               "with one or two warps per scheduler doing the arithmetic, and the time "
               "scales with the chain's length",
     "cause": "each step needs the previous step's result and there is no independent "
              "work to fill the latency",
     "action": "split the chain's state across more warps or blocks so several chains "
               "interleave, or make each step larger to need fewer of them"},
)

SUGGESTION_IDS = tuple(row["id"] for row in SUGGESTIONS)


def suggestions(signal: str | None = None) -> list[dict]:
    """The playbook rows, all of them or the one ``signal`` names (unknown -> error)."""
    if signal is None:
        return [dict(row) for row in SUGGESTIONS]
    for row in SUGGESTIONS:
        if row["id"] == signal:
            return [dict(row)]
    raise KeyError(f"no signal {signal!r}; known: {', '.join(SUGGESTION_IDS)}")


# -- command line ----------------------------------------------------------------------


def _print_class(metrics: dict[str, float], kernel: str | None) -> int:
    try:
        result = classify(metrics)
    except ValueError as exc:
        print(f"CLASSIFIER-ERROR {exc}", file=sys.stderr)
        return 2
    evidence = result["evidence"]
    numbers = " ".join(
        f"{key}={_fmt(evidence[key])}"
        for key in ("compute_sol_pct", "memory_sol_pct", "efficiency_pct", "gap_pct",
                    "dram_pct", "long_scoreboard_share", "long_scoreboard_ratio")
        if key in evidence and evidence[key] is not None)
    print(f"CLASS {result['class']} {numbers}")
    print(f"EVIDENCE reason={_quote(evidence.get('reason', ''))} "
          f"tensor_cores={'yes' if evidence['tensor_cores'] else 'no'} "
          f"threshold_pct={_fmt(evidence['underutilized_pct'])}"
          + (f" kernel={_quote(kernel)}" if kernel else ""))
    if evidence["missing"]:
        print("WARNING missing " + ",".join(evidence["missing"])
              + " (read as 0: its metric is absent from the report)")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _quote(text: str) -> str:
    return '"' + str(text).replace('"', "'") + '"'


def _print_suggestions(rows: list[dict]) -> int:
    for row in rows:
        print(f"SUGGEST signal={row['id']} cause={_quote(row['cause'])} "
              f"action={_quote(row['action'])}")
        print(f"  when: {row['signal']}")
    return 0


def _print_distance(args) -> int:
    try:
        result = distance(flops=args.flops, bytes_read=args.bytes_read,
                          bytes_written=args.bytes_written, hbm_gb_s=args.hbm_gb_s,
                          tflops=args.tflops, t_launch_ms=args.t_launch_us / 1e3,
                          eff_hbm=args.eff_hbm, eff_tensor=args.eff_tensor,
                          current_ms=args.current_ms)
    except ValueError as exc:
        print(f"DISTANCE-ERROR {exc}", file=sys.stderr)
        return 2
    print("DISTANCE " + " ".join(f"{key}={_fmt(value)}" for key, value in result.items()))
    return 0


def _do_classify(args) -> int:
    try:
        metrics = read_metrics(text=args.metrics, csv_path=args.csv, json_path=args.json,
                               kernel=args.kernel)
    except (OSError, ValueError) as exc:
        print(f"CLASSIFIER-ERROR {exc}", file=sys.stderr)
        return 2
    return _print_class(metrics, args.kernel)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roofline_tools.py",
        description="Roofline arithmetic (the floor `forge roofline` uses) and NCU-signal "
                    "classification. First-order advice: it never measures.")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    c = sub.add_parser("classify", help="a profile's SOL numbers -> the bottleneck class")
    c.add_argument("--metrics", default=None,
                   help='metrics as "name=value, name=value" (ncu --csv names)')
    c.add_argument("--csv", default=None, help="an ncu --import ... --csv export")
    c.add_argument("--json", default=None, help="a JSON object of metric name -> number")
    c.add_argument("--kernel", default=None,
                   help="with --csv: only rows whose Kernel Name contains this text")
    c.set_defaults(func=_do_classify)

    s = sub.add_parser("suggest", help="NCU signal -> cause -> candidate action")
    s.add_argument("--signal", default=None, help="one row by id")
    s.add_argument("--list", action="store_true", help="list the ids only")
    def _suggest(a) -> int:
        if a.list:
            for row in SUGGESTIONS:
                print(row["id"])
            return 0
        try:
            return _print_suggestions(suggestions(a.signal))
        except KeyError as exc:
            print(f"SUGGEST-ERROR {exc.args[0]}", file=sys.stderr)
            return 2
    s.set_defaults(func=_suggest)

    d = sub.add_parser("distance", help="the floor and distance of one workload")
    d.add_argument("--calibration", required=True,
                   help="a file `forge calibrate` wrote (its peaks are used)")
    d.add_argument("--flops", type=float, required=True, help="FLOPs per call")
    d.add_argument("--bytes-read", type=float, default=0.0, help="bytes read per call")
    d.add_argument("--bytes-written", type=float, default=0.0, help="bytes written per call")
    d.add_argument("--peak", default="bf16",
                   choices=("bf16", "fp16", "fp8", "fp4", "fp32", "sfu"),
                   help="which calibration peak the FLOPs are counted against")
    d.add_argument("--hbm-gb-s", type=float, default=None,
                   help="override the device bandwidth (default: the calibration's)")
    d.add_argument("--tflops", type=float, default=None,
                   help="override the compute peak (default: the calibration's)")
    d.add_argument("--t-launch-us", type=float, default=0.0,
                   help="launch-latency floor in microseconds (default: the calibration's)")
    d.add_argument("--eff-hbm", type=float, default=DEFAULT_EFF_HBM)
    d.add_argument("--eff-tensor", type=float, default=DEFAULT_EFF_TENSOR)
    d.add_argument("--current-ms", type=float, default=None,
                   help="a measured time, for current/floor")
    d.set_defaults(func=_print_distance)
    return p


def _fill_from_calibration(args) -> None:
    """Defaults from ``--calibration`` (explicit flags still win)."""
    peaks = calibration_peaks(read_calibration(args.calibration))
    if args.hbm_gb_s is None:
        if peaks["hbm_gb_s"] is None:
            raise ValueError(f"{args.calibration}: the calibration holds no hbm value "
                             f"(was the hbm probe skipped?)")
        args.hbm_gb_s = peaks["hbm_gb_s"]
    if args.tflops is None:
        rate = peak_per_second(peaks, args.peak)
        if rate is None:
            raise ValueError(f"{args.calibration}: the calibration holds no {args.peak} peak")
        # the floor functions take TFLOPS; an ops/second rate is the same thing / 1e12
        args.tflops = rate / TFLOPS
    if not args.t_launch_us:
        args.t_launch_us = peaks["launch_us"] or 0.0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    if args.command == "distance":
        try:
            _fill_from_calibration(args)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            print(f"DISTANCE-ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
