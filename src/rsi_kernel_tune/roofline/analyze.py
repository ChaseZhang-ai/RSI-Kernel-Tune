"""The roofline arithmetic: how far a workload is from what the hardware allows.

One call's shortest possible time, from a cost model (counted FLOPs and bytes) and a set
of peaks:

    t_mem    = (bytes_read + bytes_written) / (bandwidth x efficiency)
    t_tensor = flops / (compute peak x efficiency)
    t_floor  = max(t_mem, t_tensor, t_launch)

``t_floor`` is a *model*, not a measurement: it says what a kernel of these counted FLOPs
and bytes cannot beat on this hardware, at the efficiencies the caller chose. The board's
T_SOL is a different estimate of the same quantity (SOLAR's), computed by the board; where
T_SOL sits below ``t_floor`` the board's floor is not reachable under this model, and the
rendered file says so instead of quietly printing a score that cannot be hit.

Two peak sets are used together, never mixed up:

* the **target** peaks -- what the graded GPU (B200) can do. They come from the target
  profile (``target/B200.json``, official figures) plus ``peaks.toml`` for the
  precisions the profile leaves out, and every value carries its source;
* the **development** peaks -- what this campaign's GPU actually does, measured by
  ``forge calibrate`` (a calibration document). A dev reading of a workload is optimistic
  where the two differ, so each workload is flagged (``fp4-bound``, ``sfu-bound``,
  ``sm-count-sensitive``).

The floor arithmetic itself is ``skills/roofline/roofline_tools.py``: the skill's
``distance`` subcommand and this module are one implementation, so `forge roofline` and
a GPU job's ``roofline_tools.py distance`` cannot disagree.

Every function here is pure except the three readers at the end (a card's
``problem.json``, a calibration document, a run's ``best_confirmation.json``).
"""

from __future__ import annotations

import json
import math
import statistics
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi_kernel_tune.leaderboard import sol_score, time_for_score

try:
    from rsi_kernel_tune.skills import roofline_tools as rt
except ImportError as e:                                  # pragma: no cover
    raise ImportError(
        "rsi_kernel_tune.skills.roofline_tools (skills/roofline/roofline_tools.py) is part of "
        "this repository and holds the floor arithmetic `forge roofline` and the roofline "
        "skill share") from e

# The tensor-precision names a cost model may name, in the order the peaks table shows
# them. "sfu" is not a tensor precision: it is the special-function pipe a scalar
# kernel's transcendental operations run on.
PEAK_NAMES = ("bf16", "fp16", "fp8", "fp4", "fp32", "sfu")

# key -> the unit the peaks table prints. The values themselves are kept in the unit the
# analysis works in (GB/s, TFLOPS, Gop/s, us, GiB) and converted only when printed.
PEAK_UNITS = {
    "sm_count": "SMs", "memory_gib": "GiB", "hbm_gb_s": "GB/s", "l2_gb_s": "GB/s",
    "tensor_bf16_tflops": "TFLOPS", "tensor_fp16_tflops": "TFLOPS",
    "tensor_fp8_tflops": "TFLOPS", "tensor_fp4_tflops": "TFLOPS",
    "tensor_fp32_tflops": "TFLOPS", "sfu_gop_s": "Gop/s",
    "launch_us": "us", "min_kernel_us": "us",
}
PEAKS_FILE = Path(__file__).with_name("peaks.toml")
DEFAULT_TARGET_PROFILE = Path("target") / "B200.json"
DEFAULT_SOL_GOAL = 0.90


class PeaksError(ValueError):
    """A peak set cannot be built: a source is missing, or a value is not a number."""


class AnalyzeError(ValueError):
    """The analysis cannot be computed: a workload is uncovered, or a peak it needs is
    unknown (never a default: an unmeasured peak is not a measurement)."""


# -- peaks -----------------------------------------------------------------------------


@dataclass
class Peaks:
    """One machine's peak set, with the provenance of every number.

    ``measured`` is what rows are authoritative: a calibration's values were measured on
    that machine, a profile's are the vendor's official figures. ``tensor`` holds TFLOPS
    per precision name (only the ones known); ``sfu_gop_s`` the special-function rate.
    """
    gpu: str
    measured: bool
    source_label: str
    sm_count: int | None = None
    memory_gib: float | None = None
    hbm_gb_s: float | None = None
    l2_gb_s: float | None = None
    tensor: dict[str, float] = field(default_factory=dict)
    sfu_gop_s: float | None = None
    launch_us: float | None = None
    min_kernel_us: float | None = None
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def tflops_for(self, peak: str) -> float | None:
        """The TFLOPS a cost model's ``peak`` name runs at: a tensor precision's dense
        peak, or the special-function rate converted to a FLOP/s-equivalent rate."""
        if peak == "sfu":
            return self.sfu_gop_s / 1e3 if self.sfu_gop_s else None
        return self.tensor.get(peak)

    def source_of(self, key: str) -> str:
        return self.sources.get(key, "")

    def rows(self) -> list[tuple[str, float | None, str, str]]:
        """(label, value, unit, source) for the peaks table, in a fixed order."""
        out = [("SM count", self.sm_count, PEAK_UNITS["sm_count"], self.source_of("sm_count")),
               ("device bandwidth", self.hbm_gb_s, PEAK_UNITS["hbm_gb_s"],
                self.source_of("hbm_gb_s")),
               ("L2 bandwidth", self.l2_gb_s, PEAK_UNITS["l2_gb_s"],
                self.source_of("l2_gb_s"))]
        for peak in ("bf16", "fp16", "fp8", "fp4", "fp32"):
            out.append((f"tensor {peak}", self.tensor.get(peak), "TFLOPS",
                        self.source_of(f"tensor_{peak}_tflops")))
        out.append(("special functions", self.sfu_gop_s, PEAK_UNITS["sfu_gop_s"],
                    self.source_of("sfu_gop_s")))
        out.append(("launch round trip", self.launch_us, PEAK_UNITS["launch_us"],
                    self.source_of("launch_us")))
        return out


def load_peaks(path: str | Path | None = None) -> dict:
    """``peaks.toml``: {GPU name: {"aliases": [...], "peaks": {key: {"value","source"}}}}.

    Every entry must be a number with a non-empty source: this file is the one place a
    vendor figure enters an analysis, and a figure without its provenance is not usable.
    """
    file = Path(path) if path is not None else PEAKS_FILE
    with open(file, "rb") as fh:
        data = tomllib.load(fh)
    gpus = data.get("gpus")
    if not isinstance(gpus, dict) or not gpus:
        raise PeaksError(f"{file}: no [gpus.<name>] tables")
    for name, entry in gpus.items():
        peaks = (entry or {}).get("peaks")
        if not isinstance(peaks, dict) or not peaks:
            raise PeaksError(f"{file}: [gpus.{name}] has no [peaks] table")
        for key, item in peaks.items():
            if not isinstance(item, dict):
                raise PeaksError(f"{file}: gpus.{name}.peaks.{key} must be "
                                 f"{{value = <number>, source = \"...\"}}")
            if isinstance(item.get("value"), bool) or not isinstance(item.get("value"),
                                                                      (int, float)):
                raise PeaksError(f"{file}: gpus.{name}.peaks.{key} value is not a number")
            if not str(item.get("source") or "").strip():
                raise PeaksError(f"{file}: gpus.{name}.peaks.{key} has no source")
    return gpus


def manual_entry(gpus: dict, gpu: str) -> tuple[str, dict]:
    """(the table's name for ``gpu``, its peaks) -- exact name first, then an alias;
    ({}, ``{gpu: unknown}``) is not an error: the profile's own figures still apply."""
    if gpu in gpus:
        return gpu, dict(gpus[gpu].get("peaks") or {})
    for name, entry in gpus.items():
        if gpu in [str(a) for a in (entry.get("aliases") or [])]:
            return name, dict(entry.get("peaks") or {})
    return "", {}


def peaks_from_profile(profile: dict, manual: dict | None = None,
                       *, path: str | Path | None = None) -> Peaks:
    """The target profile's official peaks, filled out by ``peaks.toml``.

    The profile (``target/B200.json``) wins wherever it defines a value; the manual file
    supplies the precisions the profile does not carry (and is where their sources live).
    """
    device = (profile or {}).get("device")
    if not isinstance(device, dict) or not device:
        raise PeaksError(f"{path or 'the target profile'}: no device block")
    name = str(device.get("name") or "target GPU")
    table, entries = manual_entry(manual if manual is not None else {}, name)
    sources: dict[str, str] = {}
    notes: list[str] = []

    def from_profile(key: str, value, field_name: str) -> None:
        if value is None:
            return
        peaks[key] = value
        where = ", ".join(str(s) for s in (device.get("source") or [])) or "target profile"
        sources[key] = f"{path or DEFAULT_TARGET_PROFILE} ({field_name}) -- {where}"

    peaks: dict[str, Any] = {}
    from_profile("sm_count", device.get("multi_processor_count"),
                 "device.multi_processor_count")
    from_profile("memory_gib", device.get("memory_gib"), "device.memory_gib")
    tb = device.get("memory_bandwidth_tb_s")
    if isinstance(tb, (int, float)) and not isinstance(tb, bool):
        from_profile("hbm_gb_s", float(tb) * 1e3, "device.memory_bandwidth_tb_s")
    for key, field_name in (("tensor_fp8_tflops", "device.dense_fp8_pflops"),
                            ("tensor_fp4_tflops", "device.dense_fp4_pflops")):
        value = device.get(field_name.rsplit(".", 1)[1])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            from_profile(key, float(value) * 1e3, field_name)
    sfu = device.get("sfu_tera_exp_s")
    if isinstance(sfu, (int, float)) and not isinstance(sfu, bool):
        from_profile("sfu_gop_s", float(sfu) * 1e3, "device.sfu_tera_exp_s")

    for key, item in entries.items():
        if key in peaks:
            continue                                  # the profile's own figure wins
        peaks[key] = item.get("value")
        sources[key] = f"roofline/peaks.toml [{table or name}] -- {item['source']}"
    if not table:
        notes.append(f"peaks.toml has no entry for {name!r}: only the profile's figures "
                     f"are available")

    tensor = {peak: float(peaks[f"tensor_{peak}_tflops"])
              for peak in ("bf16", "fp16", "fp8", "fp4", "fp32")
              if isinstance(peaks.get(f"tensor_{peak}_tflops"), (int, float))}
    return Peaks(
        gpu=name, measured=False,
        source_label=f"official figures ({path or DEFAULT_TARGET_PROFILE}"
                     + (f" + {table} in roofline/peaks.toml)" if table else ")"),
        sm_count=_as_int(peaks.get("sm_count")),
        memory_gib=_as_float(peaks.get("memory_gib")),
        hbm_gb_s=_as_float(peaks.get("hbm_gb_s")),
        l2_gb_s=_as_float(peaks.get("l2_gb_s")),
        tensor=tensor,
        sfu_gop_s=_as_float(peaks.get("sfu_gop_s")),
        launch_us=_as_float(peaks.get("launch_us")),
        min_kernel_us=_as_float(peaks.get("min_kernel_us")),
        sources={k: v for k, v in sources.items() if v},
        notes=notes,
    )


def peaks_from_calibration(doc: dict, *, path: str | Path | None = None) -> Peaks:
    """The peaks a ``forge calibrate`` document holds: measured on the machine it names.

    The extraction itself lives in the skill (``roofline_tools.calibration_peaks``), so a
    GPU job without this project reads the same file the same way.
    """
    try:
        raw = rt.calibration_peaks(doc)
    except (TypeError, ValueError) as e:
        raise PeaksError(f"{path or 'the calibration'}: {e}") from None
    if not raw.get("gpu"):
        raise PeaksError(f"{path or 'the calibration'}: it names no GPU")
    label = (f"measured {raw.get('ts') or 'at an unknown time'} on {raw['gpu']}"
             + (f" ({raw['delivery']})" if raw.get("delivery") else ""))
    missing = [key for key in ("hbm_gb_s", "launch_us") if raw.get(key) is None]
    if not raw.get("tensor"):
        missing.append("tensor")
    notes = ([f"the calibration holds no {', '.join(missing)} reading"]
             if missing else [])
    return Peaks(
        gpu=str(raw["gpu"]), measured=True, source_label=label,
        sm_count=_as_int(raw.get("sm_count")),
        hbm_gb_s=raw.get("hbm_gb_s"), l2_gb_s=raw.get("l2_gb_s"),
        tensor=dict(raw.get("tensor") or {}), sfu_gop_s=raw.get("sfu_gop_s"),
        launch_us=raw.get("launch_us"), min_kernel_us=raw.get("min_kernel_us"),
        sources={f"tensor_{peak}_tflops": label for peak in (raw.get("tensor") or {})}
        | {key: label for key in ("hbm_gb_s", "l2_gb_s", "sfu_gop_s", "launch_us")
           if raw.get(key) is not None},
        notes=notes,
    )


def _as_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _as_int(value) -> int | None:
    out = _as_float(value)
    return int(out) if out is not None else None


# -- the card's workloads ---------------------------------------------------------------


def card_workloads(problem: dict) -> list[dict]:
    """The workloads a card judges and times, in card order:

    ``{"index", "name", "axes", "t_b_ms", "t_sol_ms"}``. A one-workload card has a
    top-level ``workload``/``sol_board``; a group card lists them under ``group``. A
    missing ``sol_board`` (``--no-sol-board``, or the board had no row for it) leaves
    T_b / T_SOL None: the floors are still computed, the board comparison is not.
    """
    group = problem.get("group") or []
    rows = group if group else [{"name": str((problem.get("workload") or {}).get("uuid")
                                             or "workload"),
                                 "workload": problem.get("workload") or {},
                                 "sol_board": problem.get("sol_board"),
                                 "index": problem.get("workload_index")}]
    out = []
    for position, row in enumerate(rows):
        board = row.get("sol_board") or {}
        index = row.get("index")
        out.append({
            "index": index if isinstance(index, int) else position,
            "name": str(row.get("name") or f"wl{position}"),
            "axes": dict((row.get("workload") or {}).get("axes") or {}),
            "t_b_ms": _as_float(board.get("t_b_ms")),
            "t_sol_ms": _as_float(board.get("t_sol_ms")),
        })
    return out


# -- the analysis -----------------------------------------------------------------------


@dataclass
class WorkloadRoofline:
    """One workload's floor, class and distance, on the target and on the dev machine."""
    name: str
    index: int
    axes: dict
    t_b_ms: float | None
    t_sol_ms: float | None
    cost: Any                                  # costmodel.WorkloadCost
    t_mem_ms: float
    t_tensor_ms: float
    t_launch_ms: float
    t_floor_ms: float
    bottleneck: str
    dev_floor_ms: float | None = None
    dev_bottleneck: str | None = None
    current_ms: float | None = None
    current_over_floor: float | None = None
    current_over_dev_floor: float | None = None
    headroom: str | None = None
    score: float | None = None                 # the board's sol_score at current_ms
    goal_time_ms: float | None = None          # the time that scores the goal
    goal_reachable: bool | None = None
    t_sol_over_floor: float | None = None
    t_b_over_floor: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != "cost"}


@dataclass
class Analysis:
    """Everything the rendered file says, as data (the renderer only formats it)."""
    problem: str
    peaks: Peaks
    dev_peaks: Peaks | None
    workloads: list[WorkloadRoofline]
    eff_hbm: float
    eff_tensor: float
    sol_goal: float
    current_source: str = ""
    notes: list[str] = field(default_factory=list)

    def by_name(self) -> dict[str, WorkloadRoofline]:
        return {w.name: w for w in self.workloads}

    @property
    def unreachable(self) -> list[WorkloadRoofline]:
        """Workloads whose board T_SOL sits below the modelled hardware floor: the board
        thinks a time is possible that this model says is not."""
        return [w for w in self.workloads
                if w.t_sol_over_floor is not None and w.t_sol_over_floor < 1.0]

    @property
    def goal_unreachable(self) -> list[WorkloadRoofline]:
        return [w for w in self.workloads if w.goal_reachable is False]


def analyze(problem: dict, peaks: Peaks, cost_model, *, dev_peaks: Peaks | None = None,
            current: dict[str, float] | None = None, current_source: str = "",
            eff_hbm: float = rt.DEFAULT_EFF_HBM, eff_tensor: float = rt.DEFAULT_EFF_TENSOR,
            sol_goal: float = DEFAULT_SOL_GOAL) -> Analysis:
    """The roofline of every workload of ``problem``.

    ``peaks`` is the target machine (the graded GPU); ``dev_peaks`` the development
    machine when a calibration exists. ``current`` maps a workload name to the best
    measured time of the kernel under work (``--run``): the place a reading is placed
    against the floor.
    """
    rt.check_number("eff_hbm", eff_hbm, minimum=0.0, maximum=1.0, exclusive=True)
    rt.check_number("eff_tensor", eff_tensor, minimum=0.0, maximum=1.0, exclusive=True)
    rt.check_number("sol_goal", sol_goal, minimum=0.0, exclusive=True)
    rows = card_workloads(problem)
    if not rows:
        raise AnalyzeError("the card lists no workload")
    costs = {c.name: c for c in getattr(cost_model, "workloads", [])}
    notes: list[str] = []
    if not peaks.launch_us:
        notes.append("no launch latency is known for the target machine (it is the "
                     "board's, which this campaign cannot calibrate), so the launch floor "
                     "is not applied (t_launch = 0): a launch-bound workload's floor is "
                     "then a lower bound only")
    missing_costs = [r["name"] for r in rows if r["name"] not in costs]
    if missing_costs:
        raise AnalyzeError(f"the cost model does not cover {', '.join(missing_costs)} "
                           f"(the card's workloads are "
                           f"{', '.join(r['name'] for r in rows)})")
    extra = [name for name in costs if name not in {r["name"] for r in rows}]
    if extra:
        notes.append(f"the cost model carries entries the card does not time "
                     f"({', '.join(sorted(extra))}): ignored")

    out: list[WorkloadRoofline] = []
    for row in rows:
        cost = costs[row["name"]]
        reading = (current or {}).get(row["name"])
        floor = _floor_of(cost, peaks, eff_hbm=eff_hbm, eff_tensor=eff_tensor,
                          machine="target", workload=row["name"], current_ms=reading)
        dev = (_floor_of(cost, dev_peaks, eff_hbm=eff_hbm, eff_tensor=eff_tensor,
                         machine="development", workload=row["name"], current_ms=reading)
               if dev_peaks is not None else None)
        item = WorkloadRoofline(
            name=row["name"], index=row["index"], axes=row["axes"],
            t_b_ms=row["t_b_ms"], t_sol_ms=row["t_sol_ms"], cost=cost,
            t_mem_ms=floor["t_mem_ms"], t_tensor_ms=floor["t_tensor_ms"],
            t_launch_ms=floor["t_launch_ms"], t_floor_ms=floor["t_floor_ms"],
            bottleneck=floor["bottleneck"],
            dev_floor_ms=dev["t_floor_ms"] if dev else None,
            dev_bottleneck=dev["bottleneck"] if dev else None,
            current_ms=reading,
            current_over_floor=floor["current_over_floor"],
            current_over_dev_floor=(dev["current_over_floor"] if dev and reading is not None
                                    else None),
            headroom=floor["headroom"],
        )
        if item.t_sol_ms is not None:
            item.t_sol_over_floor = item.t_sol_ms / item.t_floor_ms
        if item.t_b_ms is not None:
            item.t_b_over_floor = item.t_b_ms / item.t_floor_ms
        if reading is not None and item.t_b_ms is not None and item.t_sol_ms is not None:
            item.score = sol_score(reading, item.t_b_ms, item.t_sol_ms)
        if item.t_b_ms is not None and item.t_sol_ms is not None:
            goal = time_for_score(sol_goal, item.t_b_ms, item.t_sol_ms)
            if goal is not None:
                item.goal_time_ms = goal
                item.goal_reachable = goal >= item.t_floor_ms
        item.flags = _flags(cost, item, peaks, dev_peaks)
        out.append(item)

    if not peaks.measured:
        notes.append("the target peaks are official figures, not measurements taken here: "
                     "the floor is exact only to the ratio a real kernel reaches of them "
                     f"(the efficiency factors: memory {eff_hbm:.2f}, tensor "
                     f"{eff_tensor:.2f})")
    for peak_note in peaks.notes:
        notes.append(f"target peaks: {peak_note}")
    if dev_peaks is None:
        notes.append("no calibration was given, so there is no development-machine peak "
                     "set: the `current` column (if any) is not compared with what this "
                     "machine can do. `forge calibrate` measures it.")
    else:
        for peak_note in dev_peaks.notes:
            notes.append(f"development peaks: {peak_note}")
        if dev_peaks.launch_us and not peaks.launch_us:
            notes.append("the development floor includes a measured launch latency while "
                         "the target floor does not: where a floor sits at the launch "
                         "scale, the two columns are not comparable — read t_mem / "
                         "t_tensor for those workloads")
    return Analysis(problem=str((problem.get("definition") or {}).get("name")
                                or problem.get("name") or "problem"),
                    peaks=peaks, dev_peaks=dev_peaks, workloads=out, eff_hbm=eff_hbm,
                    eff_tensor=eff_tensor, sol_goal=sol_goal,
                    current_source=current_source, notes=notes)


def _floor_of(cost, peaks: Peaks | None, *, eff_hbm: float, eff_tensor: float,
              machine: str, workload: str, current_ms: float | None = None) -> dict:
    """``rt.distance`` for one workload with a clear error when a peak is unknown."""
    if peaks is None:
        raise AnalyzeError(f"{workload}: no {machine} peak set")
    tflops = peaks.tflops_for(cost.peak)
    if cost.flops > 0 and tflops is None:
        hint = ("measure it with `forge calibrate` (tensor probe)"
                if machine == "development" else
                "add it to roofline/peaks.toml with its source")
        raise AnalyzeError(f"{workload}: the cost model counts {cost.flops:g} {cost.peak} "
                           f"FLOPs but the {machine} peak set has no {cost.peak} peak: "
                           f"{hint}")
    if cost.bytes_total > 0 and peaks.hbm_gb_s is None:
        raise AnalyzeError(f"{workload}: the cost model moves {cost.bytes_total:g} bytes but "
                           f"the {machine} peak set has no device-bandwidth peak")
    return rt.distance(flops=cost.flops, bytes_read=cost.bytes_read,
                       bytes_written=cost.bytes_written, hbm_gb_s=peaks.hbm_gb_s,
                       tflops=tflops,
                       t_launch_ms=(peaks.launch_us or 0.0) / 1e3,
                       eff_hbm=eff_hbm, eff_tensor=eff_tensor,
                       current_ms=current_ms)


def _flags(cost, item: WorkloadRoofline, peaks: Peaks, dev: Peaks | None) -> list[str]:
    """How a *development-machine* reading of this workload can mislead (the labels
    benchx carries on its confidence list). Derived, never guessed:

    ``fp4-bound`` / ``sfu-bound`` -- the floor is set by the compute term and that term
    runs on a pipe the two GPUs differ in (dense FP4 1.5x, special functions ~2x);
    ``sm-count-sensitive`` -- a compute- or launch-bound floor moves with the SM count,
    which differs between the two machines (bandwidth-bound ones do not: 8 TB/s each).
    """
    flags = []
    if item.bottleneck == "compute":
        if cost.peak == "fp4":
            flags.append("fp4-bound")
        if cost.peak == "sfu":
            flags.append("sfu-bound")
    if dev is not None and item.bottleneck in ("compute", "latency") \
            and peaks.sm_count and dev.sm_count and peaks.sm_count != dev.sm_count:
        flags.append("sm-count-sensitive")
    return flags


# -- readers (the only I/O in this module) ----------------------------------------------


def read_card(path: str | Path) -> tuple[dict, Path]:
    """A card's ``problem.json``: the path may be the card, the task or the file itself.

    Nothing here loads a whole task: the analysis needs the card's definition, its
    workloads and the board's T_b / T_SOL, and a card is readable whatever else a task
    directory holds.
    """
    given = Path(path).expanduser()
    candidates = [given] if given.name == "problem.json" else [
        given / "eval" / "leaderboard" / "problem.json",
        given / "leaderboard" / "card" / "eval" / "leaderboard" / "problem.json",
        given / "problem.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                problem = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise AnalyzeError(f"{candidate}: {e}") from None
            if not isinstance(problem, dict) or "workload" not in problem:
                raise AnalyzeError(f"{candidate}: not a card (no workload block)")
            return problem, candidate
    raise AnalyzeError(f"{path}: no card found (looked for eval/leaderboard/problem.json "
                       f"under {given}, and a leaderboard/card/ below it)")


def read_calibration(path: str | Path) -> dict:
    """A calibration document, read through the skill's validator."""
    try:
        return rt.read_calibration(path)
    except (OSError, ValueError) as e:
        raise AnalyzeError(f"{path}: {e}") from None


def read_run(run_dir: str | Path) -> tuple[dict[str, float], str, list[str]]:
    """The current best reading of a run directory -> (name -> ms, where it came from,
    caveats).

    The run's confirmation (``operators/<op>/best_confirmation.json``) is what the loop
    archives as the run's claim: per-workload medians when the card times several
    workloads, the overall median otherwise. A reading of one workload only is placed on
    the card's first (representative) workload and the callers says so -- never spread
    over the others as if each had been measured.
    """
    root = Path(run_dir).expanduser()
    if not root.is_dir():
        raise AnalyzeError(f"{run_dir}: no such run directory")
    task = ""
    record = root / "run.json"
    if record.is_file():
        try:
            task = str(json.loads(record.read_text(encoding="utf-8")).get("task") or "")
        except json.JSONDecodeError:
            task = ""
    candidates = sorted(root.glob("operators/*/best_confirmation.json"))
    if task:
        preferred = root / "operators" / task / "best_confirmation.json"
        candidates = [p for p in candidates if p == preferred] or candidates
    if not candidates:
        raise AnalyzeError(f"{run_dir}: no operators/*/best_confirmation.json — the run "
                           f"has no confirmed best yet (`forge status`)")
    path = candidates[0]
    try:
        confirm = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnalyzeError(f"{path}: {e}") from None
    per_workload: dict[str, list[float]] = {}
    for job in confirm.get("jobs") or []:
        for row in (job or {}).get("workloads") or []:
            value = None
            for key in ("device_median_ms", "bwd_median_ms"):
                if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool):
                    value = float(row[key])
                    break
            if value is not None and row.get("name"):
                per_workload.setdefault(str(row["name"]), []).append(value)
    caveats = []
    if per_workload:
        by_name = {name: statistics.median(values) for name, values in per_workload.items()}
        source = f"{path} (median of {len(confirm.get('jobs') or [])} job(s))"
        return by_name, source, caveats
    median = confirm.get("median_ms")
    if isinstance(median, (int, float)) and not isinstance(median, bool):
        caveats.append("the confirmation carries one overall median, not per-workload "
                       "readings: it is placed on the card's first workload")
        return {"__representative__": float(median)}, f"{path} (median_ms)", caveats
    raise AnalyzeError(f"{path}: no per-workload reading and no median_ms")


__all__ = [
    "Analysis", "AnalyzeError", "DEFAULT_SOL_GOAL", "PEAK_NAMES", "Peaks", "PeaksError",
    "WorkloadRoofline", "analyze", "card_workloads", "load_peaks", "manual_entry",
    "peaks_from_calibration", "peaks_from_profile", "read_calibration", "read_card",
    "read_run",
]
