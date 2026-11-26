"""The roofline context file: what a teacher/student agent reads about the hardware.

Markdown, English, deterministic (the same ``Analysis`` renders byte for byte the same
file), and honest about its own standing: it opens by saying the numbers are a model, it
names which peak set produced each figure and where the figures came from, and where the
board's T_SOL sits below the modelled floor it says so instead of printing a score that
cannot be reached.

The file is a run context (``rsi_kernel_tune.runcontext``): ``{ROOFLINE_PATH}`` in a prompt
points at it. Nothing here sets an environment variable or touches a task.
"""

from __future__ import annotations

import json

from rsi_kernel_tune.roofline.analyze import Analysis, WorkloadRoofline

# column headers in the per-workload table, in order; the renderer fills them row by row
COLUMNS = ("#", "workload", "axes", "T_SOL ms", "floor ms", "class", "T_b ms",
           "T_SOL/floor", "T_b/floor", "current ms", "current/floor", "headroom", "flags")


def _num(value, digits: int = 6) -> str:
    """A number as the tables print it: ``-`` when there is none, 6 significant digits
    (enough to hand-check a division, short enough for a table)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def _axes(axes: dict) -> str:
    """``b=4, s=128`` -- compact and stable (keys sorted: a dict's order is not data)."""
    if not axes:
        return "-"
    return ", ".join(f"{key}={_num(value)}" for key, value in sorted(axes.items()))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(cell) for cell in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"
    out = [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(row) for row in rows]
    return out


def render(analysis: Analysis) -> str:
    """The Markdown context file (the exact text the agents read)."""
    peaks, dev = analysis.peaks, analysis.dev_peaks
    lines: list[str] = []
    lines.append(f"# Roofline — {analysis.problem}")
    lines.append("")
    lines.append(
        "One call's shortest possible time, from counted FLOPs and bytes and the peaks of "
        "the machine that grades this kernel. **This is a first-order model, not a "
        "measurement.** A floor is a bound: a reading at it is at the hardware's limit, a "
        "reading far above it has room; a reading slightly below it means the cost model "
        "or the peaks are off somewhere, not that the hardware broke a law.")
    lines.append("")
    lines.append(f"- problem: `{analysis.problem}`")
    lines.append(f"- target machine (graded): **{peaks.gpu}** — {peaks.source_label}")
    if dev is not None:
        lines.append(f"- development machine: **{dev.gpu}** — {dev.source_label}")
    else:
        lines.append("- development machine: *not calibrated* (no calibration file was "
                     "given)")
    lines.append(f"- efficiency factors: memory **{analysis.eff_hbm:g}** of the device "
                 f"bandwidth, tensor **{analysis.eff_tensor:g}** of the compute peak "
                 f"(a well-written kernel's habitual share of a vendor peak)")
    if peaks.launch_us:
        lines.append(f"- launch-latency floor: **{_num(peaks.launch_us)} us** (measured on "
                     f"the target machine)")
    else:
        lines.append("- launch-latency floor: **not applied** (no launch measurement for "
                     "the target machine)")
    lines.append(f"- score goal: **{analysis.sol_goal:g}**")
    if analysis.current_source:
        lines.append(f"- current reading from: `{analysis.current_source}`")
    lines.append("")

    # -- peaks -------------------------------------------------------------------------
    lines.append("## Peaks")
    lines.append("")
    rows = []
    dev_rows = {label: (value, unit) for label, value, unit, _ in (dev.rows() if dev else [])}
    for label, value, unit, _source in peaks.rows():
        dev_value, _dev_unit = dev_rows.get(label, (None, unit))
        ratio = (dev_value / value) if (dev_value is not None and value) else None
        rows.append([label, f"{_num(value)} {unit}" if value is not None else "-",
                     f"{_num(dev_value)} {unit}" if dev_value is not None else "-",
                     _num(ratio, 3) if ratio is not None else "-"])
    lines += _table(["quantity", "target", "development", "development / target"], rows)
    lines.append("")
    sources = [(label, source) for label, value, _unit, source in peaks.rows()
               if value is not None and source]
    if sources:
        lines.append("Where the target figures come from:")
        lines.append("")
        for label, source in sources:
            lines.append(f"- **{label}**: {source}")
        lines.append("")

    # -- per workload ------------------------------------------------------------------
    lines.append("## Per workload")
    lines.append("")
    lines.append("`floor` = max(memory time, compute time, launch time) at the factors "
                 "above. `T_SOL/floor` below 1 means the board's own SOL estimate is "
                 "below the floor this model gives — not reachable here. `current` is "
                 "the best measured reading of the kernel under work; `current/floor` is "
                 "how many times the floor it still costs.")
    lines.append("")
    rows = []
    for w in analysis.workloads:
        rows.append([
            str(w.index), w.name, _axes(w.axes), _num(w.t_sol_ms), _num(w.t_floor_ms),
            w.bottleneck, _num(w.t_b_ms), _num(w.t_sol_over_floor),
            _num(w.t_b_over_floor), _num(w.current_ms), _num(w.current_over_floor),
            w.headroom or "-", ",".join(w.flags) or "-",
        ])
    lines += _table(list(COLUMNS), rows)
    lines.append("")
    terms = [f"{w.name}: memory {_num(w.t_mem_ms)} / compute {_num(w.t_tensor_ms)}"
             + (f" / launch {_num(w.t_launch_ms)}" if w.t_launch_ms else "")
             for w in analysis.workloads]
    lines.append("Floor terms (ms, before the max): " + "; ".join(terms) + ".")
    lines.append("")
    if dev is not None:
        dev_terms = [f"{w.name}: {_num(w.dev_floor_ms)} ({w.dev_bottleneck})"
                     for w in analysis.workloads]
        lines.append("Development-machine floor (measured peaks): " + "; ".join(dev_terms)
                     + ".")
        lines.append("")

    # -- summary -----------------------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    classes: dict[str, int] = {}
    for w in analysis.workloads:
        classes[w.bottleneck] = classes.get(w.bottleneck, 0) + 1
    lines.append("- bottleneck classes: " + ", ".join(
        f"{name} {classes[name]}" for name in ("memory", "compute", "latency", "balanced")
        if name in classes) + f" (over {len(analysis.workloads)} workload(s))")
    tsol = [w for w in analysis.workloads if w.t_sol_over_floor is not None]
    if tsol:
        ratios = sorted(w.t_sol_over_floor for w in tsol)
        lines.append(f"- T_SOL / floor: min {_num(ratios[0], 4)}, median "
                     f"{_num(ratios[len(ratios) // 2], 4)}, max {_num(ratios[-1], 4)} "
                     f"over {len(ratios)} workload(s) with a board floor")
    else:
        lines.append("- T_SOL / floor: no workload carries a board floor on this card")
    graded = [w for w in analysis.workloads if w.headroom]
    if graded:
        counts: dict[str, int] = {}
        for w in graded:
            counts[w.headroom] = counts.get(w.headroom, 0) + 1
        lines.append("- headroom of the current readings: " + ", ".join(
            f"{name} {counts[name]}" for name in ("near-floor", "mid", "far")
            if name in counts))
    else:
        lines.append("- headroom of the current readings: none (no reading given)")
    lines.append("")

    lines.append("### Board floors below the hardware floor")
    lines.append("")
    if analysis.unreachable:
        for w in analysis.unreachable:
            lines.append(f"- **{w.name}**: T_SOL {_num(w.t_sol_ms)} ms is "
                         f"{_num(w.t_sol_over_floor, 3)}x the {_num(w.t_floor_ms)} ms "
                         f"floor ({w.bottleneck}-bound). A score there is capped by what "
                         f"the hardware can do: treat T_SOL as an estimate that this "
                         f"kernel cannot reach, and do not chase the missing points.")
    else:
        lines.append("- none: every T_SOL is at or above the modelled floor.")
    lines.append("")

    lines.append(f"### The {analysis.sol_goal:g} goal")
    lines.append("")
    goal_rows = [[w.name, _num(w.goal_time_ms), _num(w.t_floor_ms),
                  {True: "yes", False: "no", None: "-"}[w.goal_reachable]]
                 for w in analysis.workloads if w.goal_time_ms is not None]
    if goal_rows:
        lines += _table(["workload", "time that scores it (ms)", "floor ms", "reachable"],
                        goal_rows)
        lines.append("")
        if analysis.goal_unreachable:
            lines.append("A `no` is not a judgement on the kernel: it says the board's own "
                         "T_SOL / T_b pair puts that score below what this hardware model "
                         "allows, so read a lower score there as the cap.")
            lines.append("")
    else:
        lines.append("- no workload carries both T_b and T_SOL, so the goal cannot be "
                     "placed on this card.")
        lines.append("")

    if graded:
        lines.append("### Where the headroom is")
        lines.append("")
        # Rank by how many times the floor each reading still costs; ties by name so the
        # list is stable. The absolute gap is what a campaign's time budget buys, so it
        # is printed too.
        ranked = sorted(graded, key=lambda w: (-(w.current_over_floor or 0.0), w.name))
        for w in ranked[:5]:
            gap = (w.current_ms - w.t_floor_ms) if w.current_ms is not None else None
            lines.append(f"- **{w.name}** ({w.bottleneck}-bound): "
                         f"{_num(w.current_over_floor, 4)}x the floor"
                         + (f", {_num(gap, 3)} ms above it" if gap is not None else "")
                         + f", grade `{w.headroom}`"
                         + (f", flags {', '.join(w.flags)}" if w.flags else ""))
        lines.append("")

    if any(w.flags for w in analysis.workloads):
        lines.append("### Reading confidence on the development machine")
        lines.append("")
        lines.append("A reading taken on the development machine can be optimistic where "
                     "the two GPUs differ. These labels say why, per workload:")
        lines.append("")
        lines.append("- `fp4-bound`: the floor is set by dense FP4 arithmetic, which the "
                     "development GPU runs faster than the graded one.")
        lines.append("- `sfu-bound`: the floor is set by special-function operations, "
                     "which the development GPU runs at about twice the graded GPU's rate.")
        lines.append("- `sm-count-sensitive`: a compute- or launch-bound floor moves with "
                     "the SM count, and the two machines have different counts "
                     f"(target {peaks.sm_count}, development "
                     f"{dev.sm_count if dev else 'unknown'}).")
        lines.append("")
        for w in analysis.workloads:
            if w.flags:
                lines.append(f"- {w.name}: {', '.join(w.flags)}")
        lines.append("")

    if analysis.notes:
        lines.append("## Caveats")
        lines.append("")
        for note in analysis.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## How to use this file")
    lines.append("")
    lines.append("- It is a run context: point `{ROOFLINE_PATH}` at it "
                 "(`rsi_kernel_tune.runcontext.env_for({\"roofline\": \"<this file>\"})`, or "
                 "the orchestration tool's `--context roofline=<this file>`).")
    lines.append("- Read the class first: a `memory`-bound workload gains from moving "
                 "fewer bytes (fusion, a smaller intermediate, a wider dtype is a cost) "
                 "before it gains from a better schedule; a `compute`-bound one from more "
                 "arithmetic per byte; a `latency`-bound one from less launch overhead or "
                 "more work per launch.")
    lines.append("- Use `current/floor` to decide whether to keep optimizing a workload: "
                 "near 1 there is little left in this model, and the next step is to "
                 "check the model against a profile rather than to chase the number.")
    lines.append("")
    return "\n".join(lines) + "\n"


def to_json(analysis: Analysis) -> str:
    """The analysis as JSON (``--json``): the same numbers, for a script."""
    payload = {
        "schema": "rsi_kernel_tune.roofline/1",
        "problem": analysis.problem,
        "eff_hbm": analysis.eff_hbm,
        "eff_tensor": analysis.eff_tensor,
        "sol_goal": analysis.sol_goal,
        "current_source": analysis.current_source,
        "target": {
            "gpu": analysis.peaks.gpu, "measured": analysis.peaks.measured,
            "source": analysis.peaks.source_label,
            "sm_count": analysis.peaks.sm_count,
            "hbm_gb_s": analysis.peaks.hbm_gb_s,
            "tensor": dict(analysis.peaks.tensor),
            "sfu_gop_s": analysis.peaks.sfu_gop_s,
            "launch_us": analysis.peaks.launch_us,
        },
        "development": None if analysis.dev_peaks is None else {
            "gpu": analysis.dev_peaks.gpu, "measured": analysis.dev_peaks.measured,
            "source": analysis.dev_peaks.source_label,
            "sm_count": analysis.dev_peaks.sm_count,
            "hbm_gb_s": analysis.dev_peaks.hbm_gb_s,
            "tensor": dict(analysis.dev_peaks.tensor),
            "sfu_gop_s": analysis.dev_peaks.sfu_gop_s,
            "launch_us": analysis.dev_peaks.launch_us,
        },
        "workloads": [
            {**w.to_dict(),
             "cost": {"flops": w.cost.flops, "bytes_read": w.cost.bytes_read,
                      "bytes_written": w.cost.bytes_written, "peak": w.cost.peak}}
            for w in analysis.workloads],
        "unreachable": [w.name for w in analysis.unreachable],
        "goal_unreachable": [w.name for w in analysis.goal_unreachable],
        "notes": list(analysis.notes),
    }
    return json.dumps(payload, indent=2) + "\n"


def workload_line(item: WorkloadRoofline) -> str:
    """One workload as a single log line (the commands print it before the file)."""
    return (f"{item.name}: floor {_num(item.t_floor_ms)} ms ({item.bottleneck}), "
            f"T_SOL {_num(item.t_sol_ms)} ms"
            + (f", current {_num(item.current_ms)} ms = "
               f"{_num(item.current_over_floor, 4)}x floor" if item.current_ms else "")
            + (f" [{','.join(item.flags)}]" if item.flags else ""))


__all__ = ["COLUMNS", "render", "to_json", "workload_line"]
