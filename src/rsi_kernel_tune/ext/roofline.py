"""``forge roofline``: how far each workload of a card is from the hardware's floor.

    forge roofline tasks/<op> [--calibration FILE] [--cost-model FILE | --estimate]
                              [--run <run directory>] [--out FILE]
                              [--eff-hbm 0.85] [--eff-tensor 0.80] [--sol-goal 0.90]
                              [--target FILE] [--peaks FILE] [--json]

What it combines:

* the card's ``problem.json`` -- every workload's axes and the board's T_b / T_SOL;
* two peak sets: the **target** machine's official figures (``target/B200.json`` plus
  ``roofline/peaks.toml``) and, when ``--calibration`` names one, the **development**
  machine's measured peaks;
* a cost model -- ``{flops, bytes_read, bytes_written}`` per workload, from
  ``--cost-model FILE`` or one teacher turn (``--estimate``);
* optionally a run directory, whose confirmed best reading is placed against the floor.

It writes a Markdown context file (``{ROOFLINE_PATH}``): per workload the floor, its
bottleneck class, the board comparison, how far the current reading is from it, and how
much a development-machine reading can be trusted (``fp4-bound``, ``sfu-bound``,
``sm-count-sensitive``). It sets no environment variable and changes no task: the last
line says how to hand the file to a run.

No ``--calibration`` is not an error: the target floors are still computed from the
official peaks and the file says they are not measurements. A ``--calibration`` path that
does not read, or a cost model that does not cover the card, is an error (exit 2).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

COMMAND = "roofline"


def _teacher_backend(args):
    """A ``RealBackend`` for the one cost-model turn: ``--agent-cmd`` or the task's
    ``[agent.teacher]`` (the same construction ``forge lib baselines --rewrite`` uses)."""
    from rsi_kernel_tune.loop.cli import RealBackend

    cmd = getattr(args, "agent_cmd", None)
    if not cmd:
        from rsi_kernel_tune.agents import agent_command
        from rsi_kernel_tune.task import load_task
        task = load_task(args.task)
        cmd = agent_command(task.teacher, "teacher")
    if not cmd:
        raise ValueError("--estimate needs --agent-cmd or a task with [agent]: the cost "
                         "model is a teacher turn and the harness must know which agent "
                         "to run")
    workspace = Path(tempfile.mkdtemp(prefix="fk_roofline_agent_"))
    return RealBackend(op="roofline", workspace=workspace, agent_cmd=cmd,
                       git_name="forge-roofline", git_email="forge-roofline@localhost")


def _target_peaks(args):
    """(the target peak set, the profile path it came from)."""
    from rsi_kernel_tune.paths import repo_root
    from rsi_kernel_tune.roofline import analyze

    root = Path(args.root) if args.root else repo_root()
    profile_path = Path(args.target) if args.target else root / "target" / "B200.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise analyze.PeaksError(f"{profile_path}: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise analyze.PeaksError(f"{profile_path}: {e}") from None
    manual = analyze.load_peaks(args.peaks)
    return analyze.peaks_from_profile(profile, manual, path=profile_path), profile_path


def _current(args, names: list[str]):
    """(name -> ms, where it came from, caveats) from ``--run`` (or nothing)."""
    from rsi_kernel_tune.roofline import analyze

    if not args.run:
        return {}, "", []
    readings, source, caveats = analyze.read_run(args.run)
    if "__representative__" in readings:
        value = readings.pop("__representative__")
        if names:
            readings = {names[0]: value}
    return readings, source, caveats


def cmd_roofline(args) -> int:
    from rsi_kernel_tune.roofline import analyze, costmodel, render

    try:
        problem, card_path = analyze.read_card(args.task)
    except analyze.AnalyzeError as e:
        sys.stderr.write(f"forge roofline: {e}\n")
        return 2
    names = [row["name"] for row in analyze.card_workloads(problem)]
    try:
        peaks, profile_path = _target_peaks(args)
    except analyze.PeaksError as e:
        sys.stderr.write(f"forge roofline: {e}\n")
        return 2
    dev_peaks = None
    if args.calibration:
        try:
            dev_peaks = analyze.peaks_from_calibration(
                analyze.read_calibration(args.calibration), path=args.calibration)
        except (analyze.PeaksError, analyze.AnalyzeError) as e:
            sys.stderr.write(f"forge roofline: {e}\n")
            return 2
    if args.cost_model and args.estimate:
        sys.stderr.write("forge roofline: --cost-model and --estimate are mutually "
                         "exclusive\n")
        return 2
    if not args.cost_model and not args.estimate:
        sys.stderr.write(
            "forge roofline: give a cost model: --cost-model FILE (JSON, "
            "{flops, bytes_read, bytes_written} per workload) or --estimate (one teacher "
            "turn)\n")
        return 2
    try:
        if args.cost_model:
            model = costmodel.load(args.cost_model, problem=str(
                (problem.get("definition") or {}).get("name") or ""))
        else:
            model = costmodel.estimate(problem, _teacher_backend(args),
                                       prompt_path=args.prompt,
                                       target_profile=str(profile_path))
    except (costmodel.CostModelError, ValueError, OSError) as e:
        sys.stderr.write(f"forge roofline: {e}\n")
        return 2
    coverage = costmodel.cover(model, names)
    if coverage:
        sys.stderr.write("forge roofline: the cost model does not fit the card: "
                         + "; ".join(coverage) + "\n")
        return 2
    try:
        current, current_source, caveats = _current(args, names)
    except analyze.AnalyzeError as e:
        sys.stderr.write(f"forge roofline: {e}\n")
        return 2
    try:
        result = analyze.analyze(problem, peaks, model, dev_peaks=dev_peaks,
                                 current=current, current_source=current_source,
                                 eff_hbm=args.eff_hbm, eff_tensor=args.eff_tensor,
                                 sol_goal=args.sol_goal)
    except analyze.AnalyzeError as e:
        sys.stderr.write(f"forge roofline: {e}\n")
        return 2
    result.notes.extend(caveats)
    if args.json:
        text = render.to_json(result)
    else:
        text = render.render(result)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        for line in result.workloads:
            print(render.workload_line(line))
        print(f"\nwrote {out}")
        hint = (f"hand it to a run with `rsi_kernel_tune.runcontext.env_for({{\"roofline\": "
                f"\"{out}\"}})` or the orchestration tool's `--context roofline={out}`")
        print(hint)
    else:
        # The markdown IS the output: the hint goes to stderr so a redirect keeps the file
        # clean.
        sys.stdout.write(text)
        sys.stdout.flush()
        hint = ("hand it to a run with `rsi_kernel_tune.runcontext.env_for({\"roofline\": "
                "\"<this file>\"})` or the orchestration tool's `--context roofline=<file>`")
        sys.stderr.write(hint + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    from rsi_kernel_tune.roofline import analyze as analyze_mod
    from rsi_kernel_tune.skills import roofline_tools as rt

    p = argparse.ArgumentParser(
        prog="forge roofline",
        description="Per workload of a card: the time the hardware allows, the bottleneck "
                    "class, and how far the current reading is from it (a first-order "
                    "model, written as the Markdown context file a prompt reads).")
    p.add_argument("task", help="a board task (tasks/<op>), a run's task snapshot, a card "
                                "directory or a problem.json")
    p.add_argument("--calibration", default=None,
                   help="a `forge calibrate` document: the development machine's measured "
                        "peaks (without it only the target's official peaks are used)")
    p.add_argument("--cost-model", default=None,
                   help="JSON with {flops, bytes_read, bytes_written} per workload")
    p.add_argument("--estimate", action="store_true",
                   help="ask one teacher turn for the cost model "
                        "(meta/cost_model_prompt.md)")
    p.add_argument("--prompt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--agent-cmd", default=None,
                   help="the teacher command --estimate runs (default: the task's "
                        "[agent.teacher])")
    p.add_argument("--run", default=None,
                   help="a run directory: its confirmed best reading is placed against "
                        "the floor")
    p.add_argument("--out", default=None,
                   help="write the Markdown here (default: stdout)")
    p.add_argument("--eff-hbm", type=float, default=rt.DEFAULT_EFF_HBM,
                   help=f"share of the device bandwidth a well-written kernel reaches "
                        f"(default {rt.DEFAULT_EFF_HBM})")
    p.add_argument("--eff-tensor", type=float, default=rt.DEFAULT_EFF_TENSOR,
                   help=f"share of the compute peak (default {rt.DEFAULT_EFF_TENSOR})")
    p.add_argument("--sol-goal", type=float, default=analyze_mod.DEFAULT_SOL_GOAL,
                   help=f"the board score to place against the floor "
                        f"(default {analyze_mod.DEFAULT_SOL_GOAL})")
    p.add_argument("--target", default=None,
                   help="the target profile JSON (default target/B200.json)")
    p.add_argument("--peaks", default=None,
                   help="the manual-peaks TOML (default roofline/peaks.toml)")
    p.add_argument("--root", default=None, help="repository root (default: this checkout)")
    p.add_argument("--json", action="store_true",
                   help="print the analysis as JSON instead of Markdown")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    return cmd_roofline(args)


if __name__ == "__main__":
    raise SystemExit(main())
