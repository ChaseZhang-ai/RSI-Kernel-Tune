"""ARGUS command-line interface (no GPU required for anything here)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import attribution as at
from . import fingerprint as fp
from . import objective as ob
from . import roofline as rl
from . import verifier as vf
from .cudaforge import CUDAFORGE_METRICS, CUDAFORGE_PIN, extract_code_block, extract_json
from .memory import MechanismMemory, MechanismRecord

ROOT = Path(__file__).resolve().parents[2]     # argus/


def _cmd_selftest(_args) -> int:
    from .objective import latency_for_score, workload_score
    s = workload_score(0.5, 0.1, 0.9)
    assert abs(latency_for_score(s, 0.1, 0.9) - 0.5) < 1e-9
    assert len(CUDAFORGE_METRICS) == 24
    from .search import MECHANISMS
    assert len(MECHANISMS) == 7
    print("selftest: objective round-trip, 24 CudaForge metrics, 7 mechanisms ... ok")
    return 0


def _cmd_hacklab(args) -> int:
    from experiments import exp_hack_detection as exp
    exp.main(seed=args.seed)
    return 0


def _cmd_ab(args) -> int:
    from experiments import exp_search_ab as exp
    exp.main(seeds=args.seeds, evals=args.evals)
    return 0


def _cmd_attrib(args) -> int:
    from experiments import exp_attribution as exp
    exp.main()
    return 0


def _cmd_demo(args) -> int:
    _cmd_selftest(args)
    print()
    _cmd_hacklab(args)
    print()
    _cmd_attrib(args)
    print()
    _cmd_ab(argparse.Namespace(seeds=args.seeds, evals=args.evals))
    return 0


def _cmd_dryrun(args) -> int:
    """Exercise the production adapter in dry-run (command/prompt construction, no GPU)."""
    from .backends.cudaforge import CudaForgeBackend
    be = CudaForgeBackend(repo_dir=Path(args.repo or "./vendor/CudaForge"),
                          task_path=Path(args.task or "./vendor/CudaForge/KernelBench/level1/1_example.py"),
                          label="dryrun_task", op="rmsnorm", dry_run=True)
    seed = be.seed(0)
    child = be.propose(seed, "pipeline", 0, "## Operating point\n- bottleneck: MEMORY")
    print("dry-run adapter ok (no GPU):")
    print("  seed   :", seed.splitlines()[0][:70])
    print("  propose:", child.splitlines()[0][:70])
    ok = extract_json('```json\n{"a":1}\n```') == {"a": 1}
    print("  prompt-contract parse:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def _cmd_board(args) -> int:
    path = Path(args.data or (ROOT / "data" / "board_snapshot.json"))
    if not path.exists():
        print(f"no snapshot at {path}; run scripts/fetch_board.py to refresh")
        return 1
    snap = json.loads(path.read_text())
    print(json.dumps(snap, indent=2)[:4000])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("argus", description="ARGUS: verifier-first kernel optimization")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(func=_cmd_selftest)

    h = sub.add_parser("hacklab", help="reward-hack detection: CudaForge gate vs ARGUS")
    h.add_argument("--seed", type=int, default=1)
    h.set_defaults(func=_cmd_hacklab)

    a = sub.add_parser("ab", help="verification + search ablation on the mock")
    a.add_argument("--seeds", type=int, default=6)
    a.add_argument("--evals", type=int, default=40)
    a.set_defaults(func=_cmd_ab)

    sub.add_parser("attrib", help="attribution + pre-screen consistency").set_defaults(func=_cmd_attrib)

    d = sub.add_parser("demo", help="selftest + hacklab + attrib + a quick ab")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--seeds", type=int, default=3)
    d.add_argument("--evals", type=int, default=30)
    d.set_defaults(func=_cmd_demo)

    dr = sub.add_parser("dryrun", help="production adapter dry-run (no GPU)")
    dr.add_argument("--repo", default=None)
    dr.add_argument("--task", default=None)
    dr.add_argument("--evals", type=int, default=6)
    dr.set_defaults(func=_cmd_dryrun)

    b = sub.add_parser("board", help="print the board snapshot if present")
    b.add_argument("--data", default=None)
    b.set_defaults(func=_cmd_board)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
