"""VULCAN command line.

    python -m vulcan stats  [--dataset PATH]
    python -m vulcan plan   [--dataset PATH] [--hours 200] [--from forgekernel]
                            [--target best|golden_bridge|SCORE] [--json-out PATH]
    python -m vulcan demo   [--dataset PATH] [--hours 200] [--seed 7]
    python -m vulcan selftest

Everything here is CPU-only and standard-library-only, so it runs on a laptop for
validation and on the B300 for planning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import allocator, curriculum, dataset, objective
from .allocator import KernelState, WorkloadState
from .supervisor import Supervisor, reference_latency


def _default_dataset() -> str:
    here = Path(__file__).resolve()
    # .../vulcan/src/vulcan/cli.py -> .../vulcan/data/...
    return str(here.parents[2] / "data" / "sol_execbench_b200_flashinfer.json")


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table(rows: List[Sequence[str]], header: Sequence[str]) -> str:
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    def line(cells): return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    out = [line(header), line(["-" * w for w in widths])]
    out += [line(r) for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def cmd_stats(args: argparse.Namespace) -> int:
    ds = dataset.load(args.dataset)
    rows = dataset.headroom_table(ds, which=args.from_)
    print(f"dataset: {args.dataset}")
    print(f"kernels: {len(ds['kernels'])}   "
          f"workloads: {sum(e['n_workloads'] for e in ds['kernels'].values())}")
    for who in ("forgekernel", "golden_bridge", "best"):
        try:
            print(f"  mean per-kernel score [{who}] = {dataset.collection_score_of(ds, who):.6f}")
        except Exception:
            pass
    print()
    table_rows = []
    for r in rows:
        table_rows.append([
            r["kernel"][:46],
            str(ds["kernels"][r["kernel"]]["n_workloads"]),
            _fmt(r["score"]),
            _fmt(r["best_known"]),
            _fmt(r["delta_to_best"]),
            _fmt(r["headroom"]),
            str(r["fast1_missing"] if r["fast1_missing"] is not None else "-"),
        ])
    print(_table(table_rows,
                 ["kernel", "nw", "FK_S", "best_S", "gap", "headroom", "f1miss"]))
    print()
    print("Headroom is largest on the GEMM and small-norm kernels; because the "
          "collection\nscore is an unweighted mean, those dominate the leaderboard "
          "delta.")
    return 0


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #

def _target_scores(ds: dict, mode: str, from_: str) -> Dict[str, float]:
    if mode == "best":
        return dataset.published_scores(ds, "best")
    if mode == "golden_bridge":
        return dataset.published_scores(ds, "golden_bridge")
    try:
        s = float(mode)
    except ValueError:
        raise SystemExit(f"--target must be best|golden_bridge|<score>, got {mode!r}")
    return {n: s for n in dataset.kernel_names(ds)}


def cmd_plan(args: argparse.Namespace) -> int:
    ds = dataset.load(args.dataset)
    specs = dataset.build_specs(ds, current=args.from_)
    states = [s.state for s in specs]

    alloc = allocator.allocate(states, args.hours, args.quantum,
                               fast1_weight=args.fast1_weight, method=args.method)
    targets = _target_scores(ds, args.target, args.from_)

    print(f"plan: {args.hours:g} GPU-hours over {len(specs)} kernels "
          f"(from={args.from_}, method={alloc.method})")
    print(f"  projected collection score: {alloc.score_before:.6f} -> "
          f"{alloc.score_after:.6f}  (+{alloc.gain:.6f})")
    print(f"  projected Fast1: {alloc.fast1_before[0]}/{alloc.fast1_before[1]} -> "
          f"{alloc.fast1_after[0]}/{alloc.fast1_after[1]}")
    missing = sum(e.get("fast1", {}).get(f"{args.from_}_missing") or 0
                  for e in ds["kernels"].values())
    if missing:
        print(f"  NOTE: Fast1 projection is optimistic. Published {args.from_} misses "
              f"{missing} workloads, but the\n        board does not publish which ones, so the "
              f"reconstruction places every workload of a\n        kernel at one score. Use the "
              f"per-kernel fast1_missing counts in the dataset for Fast1\n        planning, not "
              f"this projection.")
    unweighted_best = sum(targets.values()) / len(targets)
    print(f"  if every kernel merely matched its target, collection score = "
          f"{unweighted_best:.6f}")
    print()

    rows = []
    plan = {"meta": {"dataset": args.dataset, "from": args.from_,
                     "hours": args.hours, "projected_score": alloc.score_after},
            "kernels": {}}
    for spec in sorted(specs, key=lambda s: alloc.hours.get(s.name, 0.0), reverse=True):
        name = spec.name
        cur = spec.current_score()
        tgt = targets.get(name, cur)
        entry = ds["kernels"][name]
        t_b_geo = math.exp(sum(math.log(w["t_b_ms"]) for w in entry["workloads"])
                           / len(entry["workloads"]))
        t_sol_geo = math.exp(sum(math.log(max(w["t_sol_ms"], 1e-12))
                                 for w in entry["workloads"]) / len(entry["workloads"]))
        target_ms = objective.latency_for_score(min(tgt, 1.0), t_b_geo, t_sol_geo)
        rows.append([
            name[:44],
            _fmt(cur),
            _fmt(tgt),
            f"{alloc.hours.get(name, 0.0):.1f}",
            f"{t_b_geo * 1000:.3f}",
            f"{t_sol_geo * 1000:.3f}",
            f"{target_ms * 1000:.3f}",
        ])
        plan["kernels"][name] = {
            "op_class": spec.op_class,
            "family": spec.family,
            "n_workloads": len(spec.state.workloads),
            "hours": round(alloc.hours.get(name, 0.0), 3),
            "score_current": round(cur, 6),
            "score_target": round(tgt, 6),
            "target_ms_geomean": target_ms,
            "workloads": [
                {"t_b_ms": w.t_b_ms, "t_sol_ms": w.t_sol_ms,
                 "target_ms": objective.latency_for_score(min(tgt, 1.0), w.t_b_ms, w.t_sol_ms)}
                for w in spec.state.workloads
            ],
        }
    print(_table(rows, ["kernel", "cur_S", "tgt_S", "hours", "Tb(us)", "Tsol(us)",
                        "target(us)"]))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=1)
        print(f"\nwrote {args.json_out}")
    return 0


# --------------------------------------------------------------------------- #
# demo (mock A/B)
# --------------------------------------------------------------------------- #

def _bounds_and_anchors(specs):
    anchors, bounds = {}, {}
    for s in specs:
        tb = math.exp(sum(math.log(w.t_b_ms) for w in s.state.workloads)
                      / len(s.state.workloads))
        ts = math.exp(sum(math.log(max(w.t_sol_ms, 1e-12)) for w in s.state.workloads)
                      / len(s.state.workloads))
        anchors[s.name] = reference_latency(s.state)
        bounds[s.name] = (tb, ts)
    return anchors, bounds


def cmd_demo(args: argparse.Namespace) -> int:
    from .backends.mock import MockBackend

    ds = dataset.load(args.dataset)
    base_specs = dataset.build_specs(ds, current=args.from_)

    def fresh_specs():
        return dataset.build_specs(ds, current=args.from_)

    strategies = ["uniform", "headroom", "vulcan_alloc", "vulcan_portfolio", "vulcan"]
    reports = {}
    print(f"mock A/B on {len(base_specs)} kernels, {args.hours:g} GPU-hours, "
          f"seed={args.seed}\n")
    for strat in strategies:
        specs = fresh_specs()
        anchors, bounds = _bounds_and_anchors(specs)
        backend = MockBackend(anchors, bounds, seed=args.seed)
        sup = Supervisor(backend, strategy=strat, quantum=args.quantum,
                         portfolio_budgets=(1.0, 3.0, 7.0), eta=2,
                         order_strategy="hybrid")
        rep = sup.run(specs, args.hours)
        reports[strat] = rep

    rows = []
    for strat in strategies:
        r = reports[strat]
        rows.append([
            strat,
            f"{r.score_before:.5f}",
            f"{r.score_after:.5f}",
            f"{r.gain:+.5f}",
            f"{r.fast1_after[0]}/{r.fast1_after[1]}",
            str(r.backend_calls),
        ])
    print(_table(rows, ["strategy", "S_before", "S_after", "gain", "Fast1", "calls"]))
    base = reports["uniform"].score_after
    for strat in strategies:
        if strat == "uniform":
            continue
        lift = reports[strat].score_after - base
        print(f"  {strat} vs uniform: {lift:+.6f} SOL")
    print("\n(Headroom/vulcan lift is a property of the orchestration under this "
          "synthetic\nground truth; it is falsifiable and reproducible via --seed.)")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({k: v.as_dict() for k, v in reports.items()}, fh, indent=1)
        print(f"wrote {args.json_out}")
    return 0


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #

def cmd_selftest(args: argparse.Namespace) -> int:
    from . import cluster, kb_graph, portfolio, precision_guard, stats

    # objective round-trips
    for s in (0.5, 0.63, 0.85, 0.99):
        t = objective.latency_for_score(s, 1.0, 0.1)
        assert abs(objective.sol_score(t, 1.0, 0.1) - s) < 1e-9, (s, t)

    # allocator: dp beats or ties greedy and uniform on a random instance
    states = [KernelState(f"k{i}", [WorkloadState(
        t_ms=1.0 + i, t_b_ms=3.0, t_sol_ms=0.2, rate=0.05 + 0.01 * i)]) for i in range(6)]
    dp = allocator.allocate(states, 30.0, 1.0, method="dp")
    gr = allocator.allocate(states, 30.0, 1.0, method="greedy")
    un = allocator.allocate_uniform(states, 30.0)
    assert dp.score_after + 1e-9 >= gr.score_after
    assert dp.score_after + 1e-9 >= un.score_after

    # clustering recovers two obvious groups
    assert len(cluster.cluster_values([1, 1.1, 1.2, 10, 10.2, 9.9], k=2)) == 2

    # portfolio identifies the truly best arm
    quality = {"a": 0.2, "b": 0.9, "c": 0.4}
    def ev(arm, hrs): return quality[arm] * (1 - math.exp(-hrs))
    res = portfolio.successive_halving(list(quality), ev, [1, 3, 9], eta=2)
    assert res.best.name == "b", res.best.name

    # precision guard flags accumulation downgrade
    from .precision_guard import PrecisionContract, default_contract, check_contract
    bad = PrecisionContract("bf16", "bf16", "bf16")
    assert any(f.code == "accum-precision"
               for f in check_contract(bad, default_contract("rmsnorm")))

    # KB predicate retrieval
    kg = kb_graph.KnowledgeGraph([
        kb_graph.KnowledgeEntry("e1", "sm100", "TMA bulk prefetch",
                                {"dram_pct": (60, 100)}, 0.12,
                                kb_graph.STATUS_CERTIFIED)])
    assert kg.retrieve({"dram_pct": 80}, "sm100")
    assert not kg.retrieve({"dram_pct": 10}, "sm100")

    # statistics: a clear paired gain is accepted, noise is not
    ref = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]
    cand = [0.80, 0.81, 0.79, 0.80, 0.82, 0.78]
    assert stats.accept_gain(ref, cand, min_margin=0.05).accepted
    noise = [1.0, 1.0, 1.0, 1.0]
    assert not stats.accept_gain(noise, noise, min_margin=0.01).accepted

    print("selftest: OK")
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vulcan", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ds_default = _default_dataset()

    ps = sub.add_parser("stats", help="describe the dataset and headroom")
    ps.add_argument("--dataset", default=ds_default)
    ps.add_argument("--from", dest="from_", default="forgekernel",
                    choices=["forgekernel", "golden_bridge", "best"])
    ps.set_defaults(func=cmd_stats)

    pp = sub.add_parser("plan", help="score-aware budget allocation + latency targets")
    pp.add_argument("--dataset", default=ds_default)
    pp.add_argument("--hours", type=float, default=200.0)
    pp.add_argument("--quantum", type=float, default=0.5)
    pp.add_argument("--from", dest="from_", default="forgekernel",
                    choices=["forgekernel", "golden_bridge", "best"])
    pp.add_argument("--target", default="best",
                    help="best | golden_bridge | a numeric per-kernel score")
    pp.add_argument("--method", default="auto", choices=["auto", "dp", "greedy"])
    pp.add_argument("--fast1-weight", type=float, default=0.0)
    pp.add_argument("--json-out", default="")
    pp.set_defaults(func=cmd_plan)

    pd = sub.add_parser("demo", help="mock A/B of allocation+portfolio strategies")
    pd.add_argument("--dataset", default=ds_default)
    pd.add_argument("--hours", type=float, default=200.0)
    pd.add_argument("--quantum", type=float, default=0.5)
    pd.add_argument("--seed", type=int, default=7)
    pd.add_argument("--from", dest="from_", default="forgekernel",
                    choices=["forgekernel", "golden_bridge", "best"])
    pd.add_argument("--json-out", default="")
    pd.set_defaults(func=cmd_demo)

    pt = sub.add_parser("selftest", help="run internal consistency checks")
    pt.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
