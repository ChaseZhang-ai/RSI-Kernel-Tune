"""``forge campaign``: the orchestration layer (task 10).

    forge campaign targets [--snapshot DIR] [--incumbent NAME] [--top N] [--json F] [--csv F]
    forge campaign plan    [--snapshot DIR] [--dataset DIR] [--pilot auto|default|N|name,…]
                           [--problems name,…] [--hours-per-problem H] [--parallel N] [--step S]
                           [--work-dir DIR] [--dry-run]
    forge campaign run     [--work-dir DIR] [--problems name,…] [--parallel N] [--hours H]
                           [--step S] [--route auto|cutedsl|cuda|triton] [--submit]
                           [--offline] [--allow-slow-cold-start] [--force STAGE,…]
    forge campaign status  [--work-dir DIR]
    forge campaign report  [--work-dir DIR] [--out DIR] [--gap-limit N]
    forge campaign export  <problem> [--work-dir DIR] [--out FILE]
    forge campaign pack    --track TRACK [--work-dir DIR] [--out DIR] [--cover DIR]
                           [--board-records DIR] [--allow-reference-fallback]
    forge campaign sync    --from-saved PATH [--username NAME] [--work-dir DIR]
    forge campaign tools   [--work-dir DIR]

`targets` and `report` are read-only. `run` drives the loop through the repository's own
commands (`forge init` / `new` / `run`, or `forge evolve` when it is installed) and every
other task's tool through `rsi_kernel_tune.campaign.tools`, which records what it found and
what it did not. Nothing here uploads to the board: `--submit` (off by default) is the
only path that would, and it needs the board task installed.

The ext hook (`src/rsi_kernel_tune/ext/`, step 0) discovers this module once it lands; until
then the same command runs as ``python -m rsi_kernel_tune.ext.campaign``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rsi_kernel_tune.campaign import pack as pack_mod, plan as plan_mod, report as report_mod
from rsi_kernel_tune.campaign import runner, solscore, sync
from rsi_kernel_tune.campaign import targets as targets_mod
from rsi_kernel_tune.campaign import tools as tools_mod

COMMAND = "campaign"
DEFAULT_WORK_NAME = "campaign"


def _work_dir(args) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser().resolve()
    return (Path(args.root) / ".artifacts" / DEFAULT_WORK_NAME).resolve()


def _snapshot(args):
    path = args.snapshot or targets_mod.default_snapshot(Path(args.root))
    if path is None:
        raise SystemExit("[campaign] no board snapshot found; pass --snapshot DIR")
    return targets_mod.load_snapshot(path)


def _plan(args, required: bool = True) -> dict | None:
    work = _work_dir(args)
    p = work / "plan.json"
    if not p.is_file():
        if required:
            raise SystemExit(f"[campaign] no plan at {p}; run `forge campaign plan` first")
        return None
    return plan_mod.load_plan(p)


def _rows(args, snap, work: Path, with_prior: bool = True):
    ledger = sync.Ledger(work / sync.LEDGER_FILE)
    ours = sync.ours_from_ledger(ledger.entries())
    starters = targets_mod.starter_situation(snap, args.root)
    return targets_mod.rows(snap, incumbent=args.incumbent, ours=ours, starters=starters,
                            with_prior=with_prior)


# --- targets -----------------------------------------------------------------------------------

def cmd_targets(args) -> int:
    snap = _snapshot(args)
    work = _work_dir(args)
    rows, who, why = _rows(args, snap, work, with_prior=not args.no_prior)
    ranked = targets_mod.ranked(rows, by="priority" if args.by == "priority" else "gap")
    print(targets_mod.format_table(ranked, limit=args.top))
    print()
    print(targets_mod.format_instrument(targets_mod.instrument(rows, snap, who)))
    print()
    print(f"starters: {sum(1 for r in rows if r.starter == targets_mod.STARTER_EXACT)} exact, "
          f"{sum(1 for r in rows if r.starter == targets_mod.STARTER_FAMILY)} family-only, "
          f"{sum(1 for r in rows if r.starter == targets_mod.STARTER_BASELINE)} baseline-only, "
          f"{sum(1 for r in rows if r.starter == targets_mod.STARTER_UNKNOWN)} unknown")
    for w in snap.warnings:
        print(f"[campaign] warning: {w}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(targets_mod.to_json(ranked, snap, who, why,
                                           targets_mod.instrument(rows, snap, who)),
                       indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[campaign] wrote {args.json}")
    if args.csv:
        Path(args.csv).write_text(targets_mod.to_csv(ranked), encoding="utf-8")
        print(f"[campaign] wrote {args.csv}")
    return 0


# --- plan --------------------------------------------------------------------------------------

def cmd_plan(args) -> int:
    snap = _snapshot(args)
    work = _work_dir(args)
    ledger = sync.Ledger(work / sync.LEDGER_FILE)
    ours = sync.ours_from_ledger(ledger.entries()) or None
    selection = args.problems or args.pilot
    tools = tools_mod.resolve(args.root, run_log=work / "tools.log")
    plan = plan_mod.build_plan(
        snap, selection=selection, dataset=args.dataset or os.environ.get("FK_DATASET", ""),
        work_dir=work, step=args.step, hours=args.hours_per_problem, parallel=args.parallel,
        count=args.count, workloads=args.workloads, incumbent=args.incumbent, ours=ours,
        with_prior=not args.no_prior, root=args.root)
    print(plan_mod.format_plan(plan))
    print()
    print("commands this plan would run:")
    for line in plan_mod.dry_run_commands(plan, evolve_available=tools.evolve is not None):
        print(f"  {line}")
    if args.dry_run:
        print("[campaign] --dry-run: nothing written")
        return 0
    path = plan_mod.write_plan(plan, work / "plan.json")
    print(f"[campaign] wrote {path}")
    for n in tools.notes:
        print(f"[campaign] tool: {n}")
    return 0


# --- run ---------------------------------------------------------------------------------------

def cmd_run(args) -> int:
    plan = _plan(args)
    work = _work_dir(args)
    tools = tools_mod.resolve(args.root, run_log=work / "tools.log")
    if tools.bench is None:
        print("[campaign] benchx is not installed: no starting point can be scored, so the "
              "task keeps its default target (readings will have no SOL score locally)")
    for n in tools.notes:
        print(f"[campaign] tool: {n}")
    states = runner.run_plan(plan, tools=tools, work_dir=work,
                             problem_names=args.problems, parallel=args.parallel,
                             hours=args.hours, step=args.step, route=args.route,
                             submit=args.submit, offline=args.offline,
                             allow_slow_cold_start=args.allow_slow_cold_start,
                             force=tuple(args.force or ()))
    p = runner.write_run_report(work, plan)
    print(f"[campaign] {p}")
    failed = [s for s in states if s.status != "done"]
    return 2 if failed else 0


def cmd_status(args) -> int:
    plan = _plan(args)
    work = _work_dir(args)
    print(runner.status(work, plan))
    return 0


# --- report / export ---------------------------------------------------------------------------

def cmd_report(args) -> int:
    snap = _snapshot(args)
    work = _work_dir(args)
    plan = _plan(args, required=False)
    rows, who, why = _rows(args, snap, work, with_prior=not args.no_prior)
    doc = report_mod.build(rows, snap, who, why, work_dir=work, plan=plan)
    out = Path(args.out) if args.out else work
    wrote = report_mod.write(doc, out, gap_limit=args.gap_limit)
    print(report_mod.markdown(doc, gap_limit=args.gap_limit))
    for kind, path in wrote.items():
        print(f"[campaign] wrote {kind}: {path}")
    return 0


def cmd_export(args) -> int:
    plan = _plan(args)
    work = _work_dir(args)
    entry = None
    for e in plan["problems"]:
        if args.problem in (e["name"], e["op"]):
            entry = e
            break
    if entry is None:
        raise SystemExit(f"[campaign] {args.problem!r} is not in the plan")
    state = runner.StateStore(work).load(entry["op"])
    if state is None or not (state.export or {}).get("path"):
        raise SystemExit(f"[campaign] {entry['name']} has no export yet; run it first")
    dest = Path(args.out) if args.out else work / "exports" / entry["op"] / "submission.py"
    src = Path(state.export["path"])
    if dest != src:
        runner.copy_export(work, entry["op"], dest)
    print(f"[campaign] {entry['name']} -> {dest} ({state.export.get('kind')}, "
          f"{state.export.get('candidate')})")
    return 0


# --- pack / sync / tools -----------------------------------------------------------------------

def cmd_pack(args) -> int:
    snap = _snapshot(args)
    work = _work_dir(args)
    tools = tools_mod.resolve(args.root, run_log=work / "tools.log")
    out = Path(args.out) if args.out else work / "out"
    res = pack_mod.pack(args.track, snap, work_dir=work, out_dir=out, dataset=args.dataset,
                        tools=tools, cover_dir=args.cover,
                        allow_reference_fallback=args.allow_reference_fallback,
                        ledger_path=work / sync.LEDGER_FILE)
    print(pack_mod.summary(res))
    return 0 if res.ok else 2


def cmd_sync(args) -> int:
    work = _work_dir(args)
    work.mkdir(parents=True, exist_ok=True)
    if args.from_saved:
        n, warns = sync.import_saved(args.from_saved, work / sync.LEDGER_FILE,
                                     username=args.username)
        print(f"[campaign] imported {n} board reading(s) from {args.from_saved}")
        for w in warns:
            print(f"[campaign] warning: {w}")
    elif args.board:
        tools = tools_mod.resolve(args.root, run_log=work / "tools.log")
        try:
            rc = tools.main_of("rsi_kernel_tune.ext.board", ["import", str(work)])
        except Exception as e:                                  # noqa: BLE001
            raise SystemExit(f"[campaign] forge board import is unavailable ({e}); use "
                             f"--from-saved with JSON you saved from the site")
        if rc != 0:
            return rc
    else:
        raise SystemExit("[campaign] pass --from-saved DIR (or --board when the board task "
                         "is installed and SOL_EXECBENCH_TOKEN is set)")
    ledger = sync.Ledger(work / sync.LEDGER_FILE)
    entries = ledger.entries()
    path = sync.write_ratios(work, entries)
    print(sync.summary(entries))
    print(f"[campaign] wrote {path}")
    return 0


def cmd_tools(args) -> int:
    work = _work_dir(args)
    tools = tools_mod.resolve(args.root, run_log=work / "tools.log")
    print("installed:")
    for k, v in tools.available().items():
        print(f"  {'ok ' if v else '-- '}{k}")
    print("notes:")
    for n in tools.notes:
        print(f"  {n}")
    profile = tools.target_profile()
    print(f"target profile: {profile if profile else 'absent'}")
    return 0


# --- argument parsing ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work-dir", default=None,
                   help="the campaign's working directory (default "
                        ".artifacts/campaign/)")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[3]),
                   help="the repository root (default: this checkout)")


def _add_snapshot(p: argparse.ArgumentParser) -> None:
    p.add_argument("--snapshot", default=None,
                   help="the saved board snapshot directory (default: the newest "
                        "../reports/board-snapshot-* next to the repository)")
    p.add_argument("--incumbent", default=None,
                   help="the team to beat (default: the highest weighted overall score)")
    p.add_argument("--no-prior", action="store_true",
                   help="ignore the public-solution feasibility prior")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge campaign", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("targets", help="the gap list: leader / best known / gap per problem")
    _add_common(t)
    _add_snapshot(t)
    t.add_argument("--top", type=int, default=25, help="rows to print (default 25)")
    t.add_argument("--by", choices=("gap", "priority"), default="gap",
                   help="order: the raw gap, or gap x feasibility (what --pilot auto reads)")
    t.add_argument("--json", default=None, help="write the full list as JSON here")
    t.add_argument("--csv", default=None, help="write the full list as CSV here")
    t.set_defaults(func=cmd_targets)

    pl = sub.add_parser("plan", help="choose the problems and write plan.json")
    _add_common(pl)
    _add_snapshot(pl)
    pl.add_argument("--dataset", default=None,
                    help="the SOL-ExecBench dataset tree (data/benchmark/<subset>/<problem>)")
    pl.add_argument("--pilot", default="default",
                    help="default | auto | N | name[,name…] (track:name when ambiguous)")
    pl.add_argument("--problems", default=None, help="the same selector, explicit")
    pl.add_argument("--count", type=int, default=plan_mod.DEFAULT_PILOT_COUNT,
                    help="how many problems --pilot auto takes (default 3)")
    pl.add_argument("--hours-per-problem", type=float, default=plan_mod.DEFAULT_HOURS)
    pl.add_argument("--parallel", type=int, default=plan_mod.DEFAULT_PARALLEL)
    pl.add_argument("--step", type=float, default=solscore.DEFAULT_STEP,
                    help="target = the best start's reading / this (default 1.10)")
    pl.add_argument("--workloads", default="all",
                    help="the card's workload selector; 'all' resolves to every workload "
                         "of the problem at run time")
    pl.add_argument("--dry-run", action="store_true", help="print, write nothing")
    pl.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="execute the plan (resumable)")
    _add_common(r)
    r.add_argument("--problems", nargs="*", default=None, help="only these problems")
    r.add_argument("--parallel", type=int, default=None,
                   help="problems in flight (default: the plan's)")
    r.add_argument("--hours", type=float, default=None, help="budget per problem")
    r.add_argument("--step", type=float, default=None, help="override the plan's step")
    r.add_argument("--route", choices=("auto", "cutedsl", "cuda", "triton"), default="auto",
                   help="implementation route for the task (auto: the first the harness "
                        "accepts)")
    r.add_argument("--submit", action="store_true",
                   help="submit each export to the board (needs the board task; default off)")
    r.add_argument("--offline", action="store_true",
                   help="never read the board's public API (T_b / T_SOL come from "
                        "board_records/ only)")
    r.add_argument("--allow-slow-cold-start", action="store_true",
                   help="export candidates whose cold start exceeds 100 s anyway")
    r.add_argument("--force", nargs="*", default=None,
                   help="re-run these stages (board_record task starts target optimize "
                        "versions export submit) even though the state says done")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="one line per problem of the plan")
    _add_common(s)
    s.set_defaults(func=cmd_status)

    rp = sub.add_parser("report", help="report.md / report.csv / report.json")
    _add_common(rp)
    _add_snapshot(rp)
    rp.add_argument("--out", default=None, help="where to write (default: the work dir)")
    rp.add_argument("--gap-limit", type=int, default=15)
    rp.set_defaults(func=cmd_report)

    ex = sub.add_parser("export", help="show / copy one problem's submission.py")
    _add_common(ex)
    ex.add_argument("problem")
    ex.add_argument("--out", default=None)
    ex.set_defaults(func=cmd_export)

    pk = sub.add_parser("pack", help="pack one track into an upload archive")
    _add_common(pk)
    _add_snapshot(pk)
    pk.add_argument("--track", required=True, choices=solscore.TRACKS)
    pk.add_argument("--dataset", required=True,
                    help="the dataset tree (needed for the reference fallback)")
    pk.add_argument("--out", default=None, help="where the zip and manifest go")
    pk.add_argument("--cover", default=None,
                    help="a tree `forge lib cover` wrote (name/submission.py per problem)")
    pk.add_argument("--allow-reference-fallback", action="store_true",
                    help="fill a missing problem with the definition's own reference "
                         "(otherwise a missing problem fails the pack)")
    pk.set_defaults(func=cmd_pack)

    sy = sub.add_parser("sync", help="import board results and refresh ratios.json")
    _add_common(sy)
    sy.add_argument("--from-saved", default=None,
                    help="a JSON file or directory saved from the board's site")
    sy.add_argument("--username", default=None,
                    help="with a per-kernel page: take only this team's row")
    sy.add_argument("--board", action="store_true",
                    help="use `forge board import` instead (needs the board task + a token)")
    sy.set_defaults(func=cmd_sync)

    tl = sub.add_parser("tools", help="what is installed and what each tool resolved to")
    _add_common(tl)
    tl.set_defaults(func=cmd_tools)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":                                     # python -m rsi_kernel_tune.ext.campaign
    raise SystemExit(main())
