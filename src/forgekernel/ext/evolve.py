"""``forge evolve`` -- multi-lineage self-evolution on one task.

    forge evolve tasks/<op> [--lineages N] [--generations G] [--step 1.05]
                           [--parallel P] [--langs cutedsl,cuda]
                           [--hypotheses <dir|file> …] [--lessons <file>]
                           [--elite K] [--refuge 1] [--migration-from 2]
                           [--s-floor 0.005] [--max-hours H] [--max-fails 2]
                           [--from-lib] [--sealed-ref] [--efforts high,max]
                           [--target-profile FILE] [--population DIR]
                           [--new-population] [--dry-run]
    forge evolve status <population dir>
    forge evolve show   <population dir>

A generation runs every active lineage once (``forge run``, at most ``--parallel``
at a time); after it, the elite are kept, stalled laggards adopt the global best,
a refuge keeps a different architecture, and the next generation starts.  ``--lineages 1``
reproduces ``scripts/self_evolve.py`` (same ``<op>-g<k>`` naming, same stop rule).

The population lives under ``.artifacts/evolve/<problem>-pop_<id>/`` and is not
repository content.  Rerunning the same command resumes it: finished generations
are not run again, and an interrupted lineage continues through ``forge run``'s
own resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgekernel import launch
from forgekernel.evolve import lineage as lineage_mod
from forgekernel.evolve import population as population_mod
from forgekernel.evolve.population import Config, Population
from forgekernel.task import TaskError, load_task

COMMAND = "evolve"


def _pop_dir() -> str:
    return ".artifacts/evolve/<problem>-pop_<id>"


def _print_help() -> None:
    print(__doc__.strip())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if argv[0] == "status":
        return _status(argv[1:])
    if argv[0] == "show":
        return _show(argv[1:])
    return _run(argv)


# -- run ---------------------------------------------------------------------------------


def _run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge evolve", description=__doc__.split("\n\n")[0])
    p.add_argument("task", help="the base task directory (tasks/<op>)")
    p.add_argument("--lineages", type=int, default=None,
                   help="how many lineages to keep alive (default: 2, or one per "
                        "--from-lib seed)")
    p.add_argument("--generations", type=int, default=None,
                   help="stop after this many generations (default: until every lineage "
                        "retires, stops improving, or --max-hours)")
    p.add_argument("--step", type=float, default=1.05,
                   help="each generation's target, x its reference's speed")
    p.add_argument("--parallel", type=int, default=2,
                   help="at most this many `forge run` subprocesses at a time")
    p.add_argument("--langs", default=None,
                   help="comma-separated languages, rotated over lineages "
                        "(default: the task's own language)")
    p.add_argument("--hypotheses", action="append", default=[], metavar="DIR_OR_FILE",
                   help="architecture hypotheses: a directory of lineage-<i>.md files or "
                        "one file for every lineage (runtime data; never a reference "
                        "implementation)")
    p.add_argument("--lessons", default=None, metavar="FILE",
                   help="lessons context file passed to every lineage (FK_LESSONS_PATH)")
    p.add_argument("--elite", type=int, default=None,
                   help="elites per generation (default: max(1, ceil(N/3)))")
    p.add_argument("--refuge", type=int, default=1,
                   help="non-elite lineages that never migrate (a different architecture "
                        "always survives)")
    p.add_argument("--migration-from", type=int, default=2, dest="migration_from",
                   help="first generation after which migration is allowed")
    p.add_argument("--s-floor", type=float, default=0.005, dest="s_floor",
                   help="board-task noise floor (SOL score points)")
    p.add_argument("--max-hours", type=float, default=None, dest="max_hours",
                   help="wall-clock budget for the whole population")
    p.add_argument("--max-fails", type=int, default=2, dest="max_fails",
                   help="retire a lineage after this many generations without a "
                        "confirmed best")
    p.add_argument("--from-lib", action="store_true", dest="from_lib",
                   help="start each lineage from `forge lib find <problem>` candidates "
                        "(task seedlib; soft dependency)")
    p.add_argument("--sealed-ref", "--blackbox", action="store_true", dest="blackbox",
                   help="board handoffs use --sealed-ref (black box) instead of the "
                        "readable --init-kernel default")
    p.add_argument("--efforts", default=None,
                   help="comma-separated efforts rotated over lineages (pilot A/B)")
    p.add_argument("--target-profile", default=None, dest="target_profile", metavar="FILE",
                   help="target profile JSON (default: FK_TARGET_PROFILE_PATH or "
                        "target/B200.json)")
    p.add_argument("--population", default=None, metavar="DIR",
                   help="population directory to continue (default: the latest one for "
                        "this problem, else a new one)")
    p.add_argument("--new-population", action="store_true", dest="new_population",
                   help="start a fresh population even if one exists for this problem")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the generation-1 task directories and commands; run nothing")
    args = p.parse_args(argv)

    try:
        base = load_task(args.task)
    except TaskError as e:
        sys.stderr.write(f"[evolve] {e}\n")
        return 2
    repo = Path(launch.repo_root())
    problem = lineage_mod.base_name(base.name)
    root, resuming = population_mod.resolve_root(
        repo, problem, explicit=args.population, new=args.new_population)

    if resuming:
        try:
            pop = Population.load(root, base_task=base, repo=repo)
        except (OSError, ValueError, TaskError) as e:
            sys.stderr.write(f"[evolve] cannot resume {root}: {e}\n")
            return 2
        if args.generations is not None:
            pop.config.generations = int(args.generations)
        print(f"[evolve] resuming population {root} with its stored configuration "
              f"({pop.config.lineages} lineages)", flush=True)
        if args.dry_run:
            return _dry_run(pop.state, pop.config, pop.base_task, pop.repo, pop.root)
        return pop.run()

    try:
        config, seeds = _config(args, base, problem, repo)
    except (ValueError, lineage_mod.LineageError) as e:
        sys.stderr.write(f"[evolve] {e}\n")
        return 2
    if args.dry_run:
        state = population_mod.initial_state(root, config, base, seeds)
        return _dry_run(state, config, base, repo, root)
    pop = Population.create(root, config, base_task=base, seeds=seeds, repo=repo)
    print(f"[evolve] population {root}: {config.lineages} lineage(s), "
          f"parallel {config.parallel}, {problem}", flush=True)
    return pop.run()


def _config(args, base, problem: str, repo: Path) -> tuple[Config, list[dict]]:
    from forgekernel.evolve import metric as metric_mod

    langs = ([x.strip() for x in args.langs.split(",") if x.strip()]
             if args.langs else [base.language])
    for lang in langs:
        lineage_mod.check_language(lang)
    efforts = ([x.strip() for x in args.efforts.split(",") if x.strip()]
               if args.efforts else [])
    for src in args.hypotheses:
        if not Path(src).expanduser().exists():
            print(f"[evolve] --hypotheses {src}: not found; ignored", flush=True)
    target = args.target_profile or population_mod.default_target_profile(repo)
    if target is None:
        print("[evolve] no target profile (target/B200.json / FK_TARGET_PROFILE_PATH); "
              "the prompts get no {TARGET_PROFILE_PATH}", flush=True)
    lineages = args.lineages
    seeds: list[dict] = []
    if args.from_lib:
        want = lineage_mod.base_name(base.name)
        seeds = lineage_mod.find_seeds(want, lineages or 8, log=print)
        if lineages is None:
            lineages = max(1, len(seeds)) if seeds else 2
        if lineages == 1 and seeds:
            print("[evolve] --lineages 1 is the old single-chain regression mode: "
                  "ignoring --from-lib seeds", flush=True)
            seeds = []
        seeds = seeds[:lineages]
    if lineages is None:
        lineages = 2
    if lineages < 1:
        raise ValueError("--lineages must be at least 1")
    if args.parallel < 1:
        raise ValueError("--parallel must be at least 1")
    config = Config(
        problem=problem, base_task=str(Path(args.task).resolve()),
        lineages=int(lineages), generations=args.generations, step=args.step,
        parallel=args.parallel, langs=langs, hypotheses=list(args.hypotheses),
        lessons=args.lessons, elite=args.elite, refuge=args.refuge,
        migration_from=args.migration_from,
        s_floor=args.s_floor if args.s_floor is not None else metric_mod.SOL_FLOOR_DEFAULT,
        max_hours=args.max_hours, max_fails=args.max_fails,
        from_lib=bool(args.from_lib), blackbox=bool(args.blackbox), efforts=efforts,
        target_profile=str(target) if target else None,
        population=args.population)
    return config, seeds


def _dry_run(state: dict, cfg: Config, base, repo: Path, root: Path) -> int:
    """Print generation 1's directories and commands; write nothing."""
    print(f"[evolve] dry-run: population {root} "
          f"({'existing' if root.is_dir() else 'would be created'})")
    print(f"  lineages={cfg.lineages} parallel={cfg.parallel} step={cfg.step:g} "
          f"refuge={cfg.refuge} migration-from={cfg.migration_from}")
    env = population_mod.context_env(
        {k: v for k, v in (("target", cfg.target_profile),
                           ("lessons", cfg.lessons)) if v})
    for lin in state["lineages"]:
        name = lineage_mod.task_name(base.name, lin["i"], 1, cfg.lineages)
        task = repo / "tasks" / name
        seed = (lin.get("seed") or {}).get("id")
        hint = f"seed={seed}" if seed else "base task's own reference"
        try:
            shown = task.relative_to(repo)
        except ValueError:
            shown = task
        print(f"  l{lin['i']} {lin['language']:8s} {shown}  <- {hint}")
        print(f"      python3 -m forgekernel new {task}")
        print(f"      python3 -m forgekernel run .artifacts/<run>")
        if lin.get("hypothesis"):
            print(f"      hypothesis: {lin['hypothesis']}")
        if env:
            print("      env: " + " ".join(f"{k}={v}" for k, v in env.items()))
    print("  later generations hand their elite to laggards after generation "
          f"{cfg.migration_from}")
    if not cfg.population:
        print("  (the first generation's runs decide everything after it)")
    return 0


# -- status / show -----------------------------------------------------------------------


def _load_state(root: str) -> dict:
    p = Path(root).expanduser() / population_mod.STATE_FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist: not a population directory")
    return json.loads(p.read_text(encoding="utf-8"))


def _status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge evolve status",
                                description="Where a population is, generation by "
                                            "generation.")
    p.add_argument("population", help="the population directory")
    args = p.parse_args(argv)
    try:
        state = _load_state(args.population)
    except (OSError, ValueError) as e:
        sys.stderr.write(f"[evolve] {e}\n")
        return 2
    print(_summary(state))
    return 0


def _summary(state: dict) -> str:
    cfg = state.get("config") or {}
    lines = [f"population {state.get('id')}: {state.get('problem')} — "
             f"{'stopped' if state.get('stopped') else 'active'}",
             f"  base task: {state.get('base_task')}",
             f"  lineages: {cfg.get('lineages')} (languages {cfg.get('langs')}), "
             f"parallel {cfg.get('parallel')}",
             f"  generations: {len(state.get('generations') or [])} done, "
             f"next {state.get('next_generation')}"]
    stop = state.get("stopped")
    if stop:
        lines.append(f"  stopped: {stop.get('reason')} after generation "
                     f"{stop.get('generation')}")
    for g in state.get("generations") or []:
        lines.append(f"  generation {g['gen']} ({g['state']}):")
        for rec in g["lineages"]:
            metric = rec.get("metric") or {}
            m = "-"
            if metric.get("value") is not None:
                kind = metric.get("kind")
                value = float(metric["value"])
                m = (f"sol {value:.4f}" if kind == "sol" else
                     f"{value:.4f}x" if kind == "speedup" else f"{value:.6g} ms")
            sha = ((rec.get("best") or {}).get("sha") or "-")[:12]
            ref = (rec.get("reference") or {}).get("kind", "-")
            extra = f" {m} {sha}" if rec.get("metric") else ""
            fail = f" — {rec['failure']}" if rec.get("failure") else ""
            lines.append(f"    l{rec['i']}: {rec['state']} ({ref}){extra}{fail}")
        for i, d in sorted((g.get("decisions") or {}).items()):
            if d.get("action") == "migrate":
                lines.append(f"    decision l{i}: migrate from l{d.get('donor')}")
    for lin in state.get("lineages") or []:
        if lin.get("retired"):
            lines.append(f"  l{lin.get('i')} retired: {lin.get('retire_reason')}")
    return "\n".join(lines)


def _show(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="forge evolve show",
                                description="Print a population's archive.md.")
    p.add_argument("population", help="the population directory")
    args = p.parse_args(argv)
    path = Path(args.population).expanduser() / "archive.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"[evolve] {e}\n")
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
