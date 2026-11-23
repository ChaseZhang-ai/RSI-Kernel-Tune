"""``forge bench`` -- bench a candidate kernel on a leaderboard card.

    forge bench <task dir | card dir> --candidate <kernel.py | solution.json>
                 [--workloads all | 0,3,5-9] [--repeat N] [--mode dev | target]
                 [--json OUT] [--keep]
    forge bench compare <a.json> <b.json> [...]
    forge bench best-of <a.json> <b.json> [...]

The reading comes from the card's own stopwatch and judge (the loop's GPU job path), so
it is the same reading an iteration of ``forge run`` would produce; see
``docs/benchx.md`` for what is and is not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgekernel.benchx import compare as compare_mod
from forgekernel.benchx.api import BenchError, BenchResult

COMMAND = "bench"

EXIT_OK = 0
EXIT_FAILED = 2          # usage error, or a bench whose verdict is not "correct"


def _say(msg: str = "") -> None:
    print(msg, flush=True)


# -- printing ---------------------------------------------------------------------------------


def _ms(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _print_target_check(result: BenchResult) -> None:
    check = result.integrity.get("target_compile") or {}
    external = "attached" if check.get("external") != "unavailable" else "unavailable"
    _say(f"target compile: {check.get('status')} "
         f"(engine {check.get('engine')}, features "
         f"{', '.join(check.get('features') or []) or 'none'}, external {external})")
    if check.get("error"):
        _say(f"  {check['error']}")


def print_result(result: BenchResult, *, task_name: str = "", suite: str = "",
                 definition: str = "", mode: str = "dev") -> None:
    head = f"{task_name or 'card'}"
    if suite:
        head += f" ({suite}{' ' + definition if definition else ''})"
    _say(f"bench {head}: mode {mode}, jobs {result.job or 'none'}, "
         f"{result.elapsed_s:.1f}s")
    if mode == "target":
        _print_target_check(result)
    else:
        cold = "n/a" if result.cold_start_s is None else f"{result.cold_start_s:.1f}s"
        _say(f"cold start: {cold} ({result.integrity.get('cold_start')})"
             + (f" -- {result.integrity.get('cold_start_note')}"
                if result.integrity.get("cold_start") not in (None, "ok") else ""))
        _say(f"correctness: {'PASSED' if result.correct else 'FAILED'}"
             + (f" -- {result.error}" if result.error else ""))
        _say()
        _say(f"{'workload':<28}{'t_ms':>12}{'T_b':>12}{'T_SOL':>12}{'S':>9}  fast")
        for w in result.workloads:
            fast = "yes" if (w.t_ms is not None and w.t_b is not None
                             and w.t_ms < w.t_b) else ("no" if w.t_ms is not None else "-")
            _say(f"{w.name:<28}{_ms(w.t_ms):>12}{_ms(w.t_b):>12}{_ms(w.t_sol):>12}"
                 f"{_ms(w.s):>9}  {fast}")
            if w.error:
                _say(f"    {w.error}")
        _say()
        ms = "n/a" if result.mean_s is None else f"{result.mean_s:.4f}"
        ratio = result.fast1_ratio()
        _say(f"mean S {ms}   fast1 {result.fast1}"
             + (f"/{len(result.timed())} ({ratio:.2f})" if ratio is not None else ""))
        _print_target_check(result)
        if result.integrity.get("workload_subset"):
            sub = result.integrity["workload_subset"]
            _say(f"workload subset: {sub.get('requested')} of {sub.get('card_workloads')} "
                 f"(the temp card was trimmed; a full-card reading needs --workloads all)")
        for key in ("clock", "clock_note", "reading_basis", "repeat"):
            if key in result.integrity:
                _say(f"{key}: {json.dumps(result.integrity[key], ensure_ascii=False)}")
        if result.confidence:
            _say(f"confidence: {', '.join(result.confidence)}")
    _say()


def print_compare(table: dict) -> None:
    labels = list(table["summary"])
    width = max([len("workload")] + [len(w["workload"]) for w in table["workloads"]])
    _say(f"{'workload':<{width}}  " + "  ".join(f"{label:>22}" for label in labels))
    for row in table["workloads"]:
        cells = []
        for label in labels:
            e = next((x for x in row["entries"] if x["candidate"] == label), None)
            if e is None or e["t_ms"] is None:
                cells.append(f"{'-':>22}")
                continue
            tag = "" if e["correct"] else "!"
            delta = f" ({e['delta_pct']:+.1f}%)" if e["delta_pct"] is not None else ""
            cells.append(f"{e['t_ms']:.4f}{delta}{tag:>1}".rjust(22))
        _say(f"{row['workload']:<{width}}  " + "  ".join(cells))
    _say()
    for label in labels:
        s = table["summary"][label]
        mean = "n/a" if s["mean_s"] is None else f"{s['mean_s']:.4f}"
        geo = "n/a" if s["geomean_ms"] is None else f"{s['geomean_ms']:.4f}"
        _say(f"{label}: correct {s['correct']}, mean S {mean}, "
             f"fast1 {s['fast1']}"
             + (f"/{s['n_timed']} ({s['fast1_ratio']:.2f})"
                if s["fast1_ratio"] is not None else "")
             + f", geomean ms {geo}")
    _say("\n(! = the candidate was not correct on that workload)")


def print_best_of(picked: dict) -> None:
    _say(f"{'workload':<28}{'winner':>24}{'t_ms':>12}{'S':>9}  within noise")
    for row in picked["routes"]:
        t = "n/a" if row["t_ms"] is None else f"{row['t_ms']:.4f}"
        s = "n/a" if row["s"] is None else f"{row['s']:.4f}"
        _say(f"{row['workload']:<28}{str(row['winner'] or '-'):>24}{t:>12}{s:>9}  "
             f"{', '.join(row['within_noise'])}")
    fb = picked["fallback"] or "-"
    fs = "n/a" if picked["fallback_mean_s"] is None else f"{picked['fallback_mean_s']:.4f}"
    _say(f"\nfallback: {fb} (mean S {fs})")


# -- subcommands ------------------------------------------------------------------------------


def _load_results(paths: list[str]) -> dict[str, BenchResult]:
    out: dict[str, BenchResult] = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise BenchError(f"{path} is not a bench result file")
        try:
            result = BenchResult.load(path)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise BenchError(f"{path} does not read as a BenchResult: {e}") from None
        label = path.name
        while label in out:                    # same basename twice: keep both
            label = f"{label}+"
        out[label] = result
    if len(out) < 2:
        raise BenchError("give at least two bench result files")
    return out


def _write_json(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _say(f"wrote {p}")


def cmd_bench(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge bench",
        description="Bench one candidate kernel on a leaderboard card, the card's way.")
    p.add_argument("task", help="the board task (tasks/<op>) or its card directory")
    p.add_argument("--candidate", required=True,
                   help="a .py file with a top-level run(), or an official solution JSON")
    p.add_argument("--workloads", default="all",
                   help="all (default), or indices/names such as 0,3,5-9 -- the temp "
                        "card's problem.json is trimmed to them (a subset reading, "
                        "recorded in integrity.workload_subset)")
    p.add_argument("--repeat", type=int, default=1,
                   help="eval jobs to run; each workload's reading is their median "
                        "(the cold start is the first probe job's)")
    p.add_argument("--mode", choices=("dev", "target"), default="dev",
                   help="dev: run GPU job(s) on the machine local.env names; "
                        "target: only the target compile check, nothing runs")
    p.add_argument("--json", help="write the BenchResult here")
    p.add_argument("--keep", action="store_true",
                   help="keep the temporary bench run (.artifacts/benchx_*)")
    args = p.parse_args(argv)
    from forgekernel.benchx.runner import CardBench, load_board   # the heavy half

    try:
        task, problem = load_board(args.task)
        bench = CardBench(repeat=args.repeat, keep=args.keep, log=_say)
        result = bench.bench_task(task, problem, args.candidate,
                                 workloads=args.workloads, mode=args.mode)
    except BenchError as e:
        sys.stderr.write(f"[forge bench] {e}\n")
        return EXIT_FAILED
    print_result(result, task_name=task.name, suite=str(problem.get("suite") or ""),
                 definition=str(problem.get("definition", {}).get("name") or ""),
                 mode=args.mode)
    if args.json:
        _write_json(args.json, result.to_dict())
    if args.mode == "target":
        check = result.integrity.get("target_compile") or {}
        return EXIT_OK if check.get("status") == "ok" else EXIT_FAILED
    return EXIT_OK if result.correct else EXIT_FAILED


def cmd_compare(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge bench compare",
        description="Per-workload comparison of two or more bench results (pure: no GPU).")
    p.add_argument("results", nargs="+", help="BenchResult JSON files (--json outputs)")
    p.add_argument("--json", help="write the comparison table here")
    args = p.parse_args(argv)
    try:
        results = _load_results(args.results)
    except BenchError as e:
        sys.stderr.write(f"[forge bench compare] {e}\n")
        return EXIT_FAILED
    table = compare_mod.compare(results)
    print_compare(table)
    if args.json:
        _write_json(args.json, table)
    return EXIT_OK


def cmd_best_of(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge bench best-of",
        description="Per-workload fastest candidate (ties within the noise floor go to "
                    "the earlier file: pass simpler implementations first).")
    p.add_argument("results", nargs="+", help="BenchResult JSON files (--json outputs)")
    p.add_argument("--json", help="write the routing table here (the shapes task reads it)")
    args = p.parse_args(argv)
    try:
        results = _load_results(args.results)
    except BenchError as e:
        sys.stderr.write(f"[forge bench best-of] {e}\n")
        return EXIT_FAILED
    picked = compare_mod.best_of(results)
    print_best_of(picked)
    if args.json:
        _write_json(args.json, picked)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return cmd_compare(argv[1:])
    if argv and argv[0] == "best-of":
        return cmd_best_of(argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(__doc__.strip() + "\n")
        return EXIT_OK
    return cmd_bench(argv)
