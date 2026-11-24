"""``forge shapes`` -- one kernel per shape, one entry point to dispatch between them.

    forge shapes plan    <task dir> [--max-clusters K] [--split-ratio R] [--min-size N]
                         [--init-args "..."] [--sol-goal S] [--out DIR] [--dry-run]
                         [--json FILE]
    forge shapes run     <plan.json | task dir> [--root DIR] [--python P] [--max-hours H]
                         [--cluster-hours H] [--sol-goal S] [--dry-run] [--json FILE]
    forge shapes compose <plan.json | task dir> [--root DIR] [--out PATH]
                         [--also FILE ...] [--overall FILE] [--allow-slower] [--no-reuse]
                         [--no-versions] [--no-lib] [--repeat N] [--keep] [--author A]
                         [--hardware H] [--json FILE]

The three steps, and what each leaves behind in ``tasks/<op>/shapes/``:

* ``plan`` -- the card's workloads grouped into size classes, one task per cluster:
  ``plan.json`` (and the cluster task directories listed in ``.git/info/exclude``).
  ``--dry-run`` prints the plan and writes nothing.
* ``run`` -- ``forge init`` / ``new`` / ``run`` / ``export`` per cluster, sequentially,
  resumable: ``runs.json``. ``--dry-run`` prints the commands and runs none of them.
* ``compose`` -- scores every candidate on the problem's **own card** through benchx,
  routes each workload to its fastest correct implementation (never slower than the eager
  reference unless ``--allow-slower``), and writes one ``submission.py`` plus
  ``matrix.json`` / ``shapes_compose.json`` / ``solution.json`` / ``report.md``.

Exit codes: 0 done, 1 the deliverable did not pass its own check, 2 a usage error,
3 `run` finished with a cluster failed or the budget exhausted (resumable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgekernel.paths import repo_root
from forgekernel.shapes import PRODUCT_DIR, ShapesError
from forgekernel.shapes import cluster as cluster_mod
from forgekernel.shapes import compose as compose_mod
from forgekernel.shapes import matrix as matrix_mod
from forgekernel.shapes import plan as plan_mod
from forgekernel.shapes import run as run_mod

COMMAND = "shapes"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNFINISHED = 3


def _say(message: str = "") -> None:
    print(message, flush=True)


def _fail(message: str) -> int:
    sys.stderr.write(f"[forge shapes] {message}\n")
    return EXIT_USAGE


# --- plan ---------------------------------------------------------------------------------

def cmd_plan(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge shapes plan",
        description="Cluster a board problem's workloads by size and lay out one task per "
                    "cluster (no GPU, no agent).")
    p.add_argument("task", help="the board task (tasks/<op>) or its card directory")
    p.add_argument("--max-clusters", type=int, default=cluster_mod.DEFAULT_MAX_CLUSTERS,
                   help="cluster ceiling; adjacent clusters merge until it holds (default 4)")
    p.add_argument("--split-ratio", type=float, default=cluster_mod.DEFAULT_SPLIT_RATIO,
                   help="split where a neighbour is this many times larger (default 2.0)")
    p.add_argument("--min-size", type=int, default=cluster_mod.DEFAULT_MIN_SIZE,
                   help="clusters smaller than this fold into their nearest neighbour "
                        "(default 2)")
    p.add_argument("--init-args", default=None,
                   help="the original problem's `forge init` arguments (after the op name), "
                        "e.g. \"--leaderboard sol-execbench --dataset data/... "
                        "--definition L1/069_rms_norm --sol-board board.json\"; each "
                        "cluster's --workload replaces the selector. Without it the "
                        "arguments are derived from the card's provenance")
    p.add_argument("--sol-goal", type=float, default=plan_mod.DEFAULT_SOL_GOAL,
                   help="the SOL score the per-cluster [target] ms aims at (default 0.90)")
    p.add_argument("--out", default=None, help="the product directory (default "
                                               "tasks/<op>/shapes)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    p.add_argument("--json", default=None, help="also write the plan document here")
    args = p.parse_args(argv)
    try:
        plan = plan_mod.build_plan(args.task, init_args=args.init_args,
                                   max_clusters=args.max_clusters,
                                   split_ratio=args.split_ratio, min_size=args.min_size,
                                   sol_goal=args.sol_goal)
    except plan_mod.NothingToCluster as e:
        _say(f"{e}")
        return EXIT_OK
    except ShapesError as e:
        return _fail(str(e))
    doc = plan.as_json()
    if args.out:
        doc["product_dir"] = str(Path(args.out).expanduser())
    _say(plan_mod.plan_format(doc))
    if args.dry_run:
        _say("\n(dry run: no plan written, no exclude entry added)")
        return EXIT_OK
    path = plan_mod.write_plan_doc(doc, Path(doc["product_dir"]))
    _say(f"\nplan: {path}")
    root = repo_root()
    excluded = plan_mod.ensure_excluded(root, plan_mod.excluded_rels(doc))
    if excluded["added"]:
        _say(f"exclude: added {', '.join(excluded['added'])} to {excluded['path']}")
    elif excluded["error"]:
        _say(f"exclude: not updated ({excluded['error']})")
        _say("  the cluster task directories are working files; keep them out of git "
             "yourself (they are never created by `shapes plan` anyway)")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        _say(f"wrote {out}")
    _say(f"\nnext: forge shapes run {doc['product_dir']}/plan.json")
    return EXIT_OK


# --- run ----------------------------------------------------------------------------------

def cmd_run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge shapes run",
        description="Init / new / run / export one ordinary task per cluster, sequentially "
                    "and resumably.")
    p.add_argument("plan", help="a plan.json, or the task directory holding one")
    p.add_argument("--root", default=None,
                   help="the ForgeKernel checkout the `forge` commands act on (default: "
                        "this checkout)")
    p.add_argument("--python", default=None,
                   help="the interpreter the `forge` child processes run (default: this one)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock cap on this invocation; unstarted clusters stay planned "
                        "and the exit code is 3 (run again to continue)")
    p.add_argument("--cluster-hours", type=float, default=None,
                   help="the loop budget written into each cluster task's [budget] hours")
    p.add_argument("--sol-goal", type=float, default=None,
                   help="the SOL score the cluster targets aim at (default: the plan's)")
    p.add_argument("--dry-run", action="store_true",
                   help="print every command, run none, write nothing")
    p.add_argument("--json", default=None, help="also write runs.json here")
    args = p.parse_args(argv)
    try:
        doc = _load_plan(args.plan)
    except (ShapesError, FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    root = Path(args.root).expanduser() if args.root else repo_root()
    rc, runs = run_mod.run_plan(doc, root=root, python=args.python,
                                max_hours=args.max_hours,
                                cluster_hours=args.cluster_hours,
                                sol_goal=args.sol_goal, dry_run=args.dry_run, log=_say)
    _say()
    _say(run_mod.format_runs(runs, doc))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(runs.as_json(), indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        _say(f"wrote {out}")
    if args.dry_run:
        return EXIT_OK
    if rc == EXIT_UNFINISHED:
        _say(f"\nruns: {Path(doc.get('product_dir') or '') / 'runs.json'} (resumable: run "
             f"`forge shapes run {args.plan}` again)")
    elif rc == EXIT_OK:
        _say(f"\nnext: forge shapes compose {args.plan}")
    return rc


# --- compose ------------------------------------------------------------------------------

def cmd_compose(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="forge shapes compose",
        description="Score every candidate on the problem's own card (through benchx) and "
                    "compose the per-workload winners into one submission.py.")
    p.add_argument("plan", help="a plan.json, or the task directory holding one")
    p.add_argument("--out", default=None,
                   help="where submission.py (and its solution.json) go: a directory or a "
                        ".py path (default tasks/<op>/shapes/submission.py)")
    p.add_argument("--also", action="append", default=[],
                   help="another candidate: a .py kernel or an official solution JSON "
                        "(repeatable)")
    p.add_argument("--overall", default=None,
                   help="one run's whole-card export (a .py or solution JSON); it competes "
                        "workload by workload and is the fallback when no candidate is "
                        "correct")
    p.add_argument("--allow-slower", action="store_true",
                   help="allow a workload to route to a candidate slower than the eager "
                        "reference (default: such a workload routes to the reference)")
    p.add_argument("--no-reuse", action="store_true",
                   help="re-bench every candidate instead of reusing matrix.json's readings")
    p.add_argument("--no-versions", action="store_true",
                   help="do not add the cluster runs' other correctness-passing versions")
    p.add_argument("--no-lib", action="store_true",
                   help="do not add the reference library's kernels for this problem, nor "
                        "`forge lib baselines`' torch candidates")
    p.add_argument("--repeat", type=int, default=1,
                   help="bench jobs per candidate; each workload's reading is their median")
    p.add_argument("--keep", action="store_true",
                   help="keep benchx's temporary run directories")
    p.add_argument("--author", default="forgekernel", help="the solution JSON's author")
    p.add_argument("--hardware", default="B200", help="the solution JSON's target hardware")
    p.add_argument("--json", default=None, help="also write the routing table here")
    args = p.parse_args(argv)
    try:
        doc = _load_plan(args.plan, allow_auto=True)
    except (ShapesError, FileNotFoundError, ValueError) as e:
        return _fail(str(e))
    opts = compose_mod.ComposeOptions(
        also=list(args.also), overall=args.overall, allow_slower=args.allow_slower,
        reuse=not args.no_reuse, with_versions=not args.no_versions,
        with_lib=not args.no_lib, repeat=args.repeat, keep=args.keep,
        out=Path(args.out) if args.out else None, author=args.author,
        hardware=args.hardware)
    res = compose_mod.compose_task(args.plan, opts=opts, plan_doc=doc, log=_say)
    if res.matrix is not None:
        _say()
        _say(matrix_mod.format_matrix(res.matrix))
    if res.routes is not None:
        _say()
        _say(matrix_mod.format_routes(res.routes))
    if res.submission is None:
        return _fail(res.error or "compose produced no submission")
    _say()
    _say(f"submission: {res.submission}")
    _say(f"solution:   {res.solution}")
    _say(f"matrix:     {res.matrix_json}")
    _say(f"routes:     {res.routes_json}")
    _say(f"report:     {res.report}")
    for key in ("submission", "solution"):
        check = res.validation if key == "submission" else res.solution_validation
        _say(f"check {key}: {'ok' if check.get('ok') else 'FAILED'}")
        for finding in check.get("findings") or []:
            _say(f"  finding: {finding}")
        for warning in check.get("warnings") or []:
            _say(f"  warning: {warning}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = (res.routes.as_json() if res.routes is not None else {})
        out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        _say(f"wrote {out}")
    return EXIT_OK if res.ok else EXIT_FAILED


# --- shared -------------------------------------------------------------------------------

def _load_plan(path: str, *, allow_auto: bool = False) -> dict:
    """The plan to act on: a plan.json, a task directory's plan, or (compose only) a plan
    built on the fly from the defaults -- which is how campaign calls it."""
    p = Path(path).expanduser()
    if p.is_dir():
        doc, used = plan_mod.plan_for(p)
        if used is None and not allow_auto:
            raise ShapesError(f"{p}: no plan.json under {p / PRODUCT_DIR} (run "
                              f"`forge shapes plan {path}` first)")
        return doc
    if not p.is_file():
        raise ShapesError(f"{p}: no such plan file")
    return plan_mod.load_plan(p)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(__doc__.strip() + "\n")
        return EXIT_OK
    sub, rest = argv[0], argv[1:]
    if sub == "plan":
        return cmd_plan(rest)
    if sub == "run":
        return cmd_run(rest)
    if sub == "compose":
        return cmd_compose(rest)
    sys.stderr.write(f"[forge shapes] unknown subcommand {sub!r} "
                     f"(plan | run | compose)\n")
    return EXIT_USAGE


if __name__ == "__main__":                                    # python -m forgekernel.ext.shapes
    raise SystemExit(main())
