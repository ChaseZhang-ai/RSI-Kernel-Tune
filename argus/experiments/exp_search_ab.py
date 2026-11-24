"""Experiment 2: the two independent claims, on a rugged synthetic landscape.

Claim (i) -- verification.  Run *the same* greedy agent loop twice, once with CudaForge's
gate and once with ARGUS's.  The first ends on a hacked kernel and scores 0 on the board
(a correctness failure zeroes a kernel); the second ends on a verified kernel.

Claim (ii) -- search.  With verification on for both, compare the greedy chain against the
ARGUS population at equal budget.  The landscape contains prerequisite-ordered mechanisms
that *lock* if applied out of order, so the unordered chain permanently loses ceiling while
ARGUS's prerequisite-aware, regression-proof population reaches the floor.

Run: ``python -m argus.cli ab --seeds 6 --evals 40``.
"""

from __future__ import annotations

import statistics as st
from typing import Dict, List

from argus.backends.mock import MockBackend, MockConfig
from argus.cudaforge import RunConfig, run_argus_loop, run_cudaforge_loop
from argus.search import MECHANISMS, SearchConfig


def _mk(seed: int, strict: bool, temptation: bool) -> MockBackend:
    return MockBackend("ab", MockConfig(seed=seed, strict_verification=strict,
                                        temptation=temptation))


def claim_verification(seeds: int = 6, evals: int = 40) -> Dict[str, float]:
    gate_off, gate_on = [], []
    for s in range(seeds):
        rc = run_cudaforge_loop(_mk(s, False, True), RunConfig(rounds=evals))
        gate_off.append(_mk(s, True, True).audit(rc.best_code))      # board would zero a hack
        gate_on.append(run_cudaforge_loop(_mk(s, True, True), RunConfig(rounds=evals)).best_score)
    return {"effective_gate_off": st.mean(gate_off), "effective_gate_on": st.mean(gate_on)}


def claim_search(seeds: int = 6, evals: int = 40) -> Dict[str, object]:
    greedy, argus, per_g, per_a = [], [], [], []
    for s in range(seeds):
        g = run_cudaforge_loop(_mk(s, True, False), RunConfig(rounds=evals)).best_score
        a = run_argus_loop(_mk(s, True, False), RunConfig(rounds=evals),
                           SearchConfig(n_lineages=4, mechanisms=list(MECHANISMS),
                                        max_evals=evals, seed=s)).best_score
        greedy.append(g); argus.append(a); per_g.append(round(g, 4)); per_a.append(round(a, 4))
    return {
        "optimum": _mk(0, True, False).optimum_score(),
        "greedy_mean": st.mean(greedy), "argus_mean": st.mean(argus),
        "greedy_per_seed": per_g, "argus_per_seed": per_a,
    }


def main(seeds: int = 6, evals: int = 40) -> None:
    print("=== Claim (i): verification (same greedy loop, gate off vs on) ===")
    v = claim_verification(seeds, evals)
    print(f"effective board score, CudaForge gate OFF: {v['effective_gate_off']:.4f}")
    print(f"effective board score, ARGUS gate ON:      {v['effective_gate_on']:.4f}")
    print()
    print("=== Claim (ii): search (verification ON, temptation OFF) ===")
    s = claim_search(seeds, evals)
    print(f"landscape optimum: {s['optimum']:.4f}")
    print(f"greedy chain: {s['greedy_mean']:.4f}   per-seed {s['greedy_per_seed']}")
    print(f"ARGUS pop.:   {s['argus_mean']:.4f}   per-seed {s['argus_per_seed']}")


if __name__ == "__main__":
    main()
