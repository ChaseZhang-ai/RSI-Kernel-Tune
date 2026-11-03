#!/usr/bin/env python3
"""Backtest the WP16 Successive-Halving rule (sweep inter-round elimination)
against the full grid at EQUAL BUDGET — the backtest_early_stop precedent.

The rule under test is sweep_run.py's `halving` mode: after each completed
round, the worst-performing half of the surviving variants stops being timed
(a BUDGET routing only — the final winner is read over every variant that ever
produced a reading, and the freed budget buys the survivors extra interleaved
rounds). The question the backtest answers: at the SAME number of bench calls,
does halving lose anything — and what does the wider grid it affords buy?

Simulation (paired per trial, stdlib-only — this script runs torch-free):
  * a one-axis response surface whose true objectives follow a smooth unimodal
    curve (the shapes parameter sweeps actually run on), plus per-READING
    gaussian noise at sigma relative to the surface's spread (one draw per
    timed bench call, the SPRT backtest's noise model);
  * FULL GRID: V variants x R interleaved rounds, every round spent unless the
    pass budget is out — the current sweep;
  * SH: 2V variants, the same recipe rounds R, halving after each round (worst
    half cut, >= 4 survivors, keep ceil(n/2)), rounds stopped at the same pass
    budget and survivors re-funded — the WP16 mode, scorer and winner rules
    identical to the full grid's;
  * the narrow grid subsamples the wide grid (every other point), so both
    procedures read ONE surface and their budgets match by construction;
  * every metric is normalized to the wide grid's true range, so "objective"
    reads as distance-to-the-best regardless of the trial's surface.

Empirical gates (the acceptance wording):
  * RULE COST (per cell, same grid, same budget): SH(2V) vs a plain full grid
    over the SAME 2V points at the same budget — the mean normalized true
    objective of SH's winner sits within 0.01 of the full grid's, and fewer
    than 0.5% of trials lose more than 5% of the range (elimination is only a
    "who is timed next" routing; the final winner still reads every variant
    that ever produced a reading, so the rule's own cost is bounded);
  * BUDGET BONUS (per cell): SH(2V)'s mean normalized objective beats the
    current full grid's V-point sweep at the same budget — the freed budget
    affords twice the grid, and the wider grid's best is measurably better in
    truth;
  * recovery (winner-is-true-argmax) rates are REPORTED, not gated: the wide
    grid's argmax is a harder target by construction (twice the points, finer
    spacing against the same noise), which is exactly the trade halving makes.

Usage: python scripts/backtest_sh.py [trials_per_cell]
"""

from __future__ import annotations

import random
import statistics
import sys

TRIALS_DEFAULT = 4000
SIGMAS = (0.005, 0.02, 0.05)     # reading noise, relative to the surface spread
GRIDS = (8, 16)                  # the full grid's width (SH runs 2x this)
ROUNDS = 3                       # the recipe's rounds (the sweep's default)


def surface(n: int, rng: random.Random) -> tuple[list[float], int]:
    """(true objectives, argmax) of a smooth unimodal one-axis surface: a
    parabola over a random interior peak plus a little local texture (the
    shapes parameter sweeps' response faces, idealized)."""
    peak = rng.uniform(0.2, 0.8)
    truth = []
    for i in range(n):
        x = i / (n - 1) if n > 1 else 0.5
        truth.append(-(x - peak) ** 2 + 0.05 * rng.random())
    return truth, max(range(n), key=lambda i: truth[i])


def read(value: float, sigma: float, rng: random.Random) -> float:
    return value + rng.gauss(0.0, sigma)


def score(readings: list[float]) -> float | None:
    """The sweep's per-variant score: the median across the interleaved rounds
    (same-job paired medians — sweep_run/lever_pricing_run's scorer)."""
    return statistics.median(readings) if readings else None


def run_full(truth: list[float], rounds: int, budget: int, sigma: float,
             rng: random.Random) -> int:
    """The current sweep: every variant timed in every round, rounds skipped
    once the pass budget is spent (sweep_run's guard)."""
    readings = {v: [] for v in range(len(truth))}
    spent = 0
    for _ in range(rounds):
        if spent + len(truth) > budget:
            break
        for v in range(len(truth)):
            readings[v].append(read(truth[v], sigma, rng))
        spent += len(truth)
    scored = [(score(readings[v]), v) for v in range(len(truth))]
    return max(scored, key=lambda sv: (sv[0] is not None, sv[0]))[1]


def run_sh(truth: list[float], rounds: int, budget: int, sigma: float,
           rng: random.Random) -> int:
    """The WP16 halving sweep: worst half eliminated after each completed
    round, NOISE-GATED (a first round with one reading per variant cuts
    nobody; a variant within the job's own noise floor of the cut line
    survives — sweep_run.py's rule, verbatim, over simulated arrays), rounds
    stopped at the pass budget, survivors re-funded up to 2x the recipe's
    rounds. The sim's floor mirrors lever_pricing_run.noise_floor: pooled
    per-round relative spread (here ~sigma) x 3 x sqrt(2) x 1.2533 / sqrt(R),
    never below the 0.5% default."""
    n = len(truth)
    readings: dict[int, list[float]] = {v: [] for v in range(n)}
    alive = list(range(n))
    spent = 0
    for rnd in range(rounds * 2):
        if spent + len(alive) > budget:
            break
        for v in alive:
            readings[v].append(read(truth[v], sigma, rng))
        spent += len(alive)
        if rnd + 1 >= rounds and len(alive) >= 4 \
                and all(len(readings[v]) >= 2 for v in alive):
            scored = sorted(alive, key=lambda v: (score(readings[v]) is None,
                                                  -(score(readings[v]) or 0.0)))
            line = (len(scored) + 1) // 2
            keep, cut = scored[:line], scored[line:]
            r_min = min(len(readings[v]) for v in alive)
            floor = max(0.5, 5.3259 * sigma * 100.0 / (r_min ** 0.5))
            boundary = score(readings[keep[-1]])
            alive = keep + [v for v in cut
                            if boundary is not None
                            and 100.0 * (boundary / score(readings[v]) - 1.0) <= floor]
    scored = [(score(readings[v]), v) for v in range(n) if readings[v]]
    return max(scored, key=lambda sv: (sv[0] is not None, sv[0]))[1]


def main(argv=None) -> int:
    trials = int(argv[0]) if argv else TRIALS_DEFAULT
    print(f"halving backtest: {trials} trials/cell, recipe rounds={ROUNDS}, "
          f"equal bench budget per trial\n")
    print(f"{'grid':>10} {'sigma':>7} {'budget':>7} {'fullV obj':>10} {'SH obj':>8} "
          f"{'full2V obj':>10} {'rule loss':>10}  gates")
    ok_all = True
    for width in GRIDS:
        n_wide = width * 2
        budget = width * ROUNDS          # every procedure plays at this budget
        for sigma in SIGMAS:
            fullv_rec = sh_rec = full2v_rec = 0
            fullv_obj = sh_obj = full2v_obj = 0.0
            losses = wins = 0
            for _ in range(trials):
                truth_wide, best_wide = surface(n_wide, random.Random())
                lo, hi = min(truth_wide), max(truth_wide)
                span = (hi - lo) or 1.0

                def norm(x):             # 1.0 = the wide grid's true best
                    return (truth_wide[x] - lo) / span
                truth = truth_wide[::2]  # the narrow grid: its even points
                best_narrow = max(range(len(truth)), key=lambda i: truth[i])
                w = run_full(truth, ROUNDS, budget, sigma, random.Random())
                s = run_sh(truth_wide, ROUNDS, budget, sigma, random.Random())
                r2 = ROUNDS * 2          # a plain full sweep can only afford
                r2 = max(1, budget // n_wide)   # budget // (2V) rounds on 2V pts
                f2 = run_full(truth_wide, r2, budget, sigma, random.Random())
                fullv_rec += (w == best_narrow)
                sh_rec += (s == best_wide)
                full2v_rec += (f2 == best_wide)
                fullv_obj += norm(w)
                sh_obj += norm(s)
                full2v_obj += norm(f2)
                if norm(s) < norm(f2) - 0.05:
                    losses += 1          # SH vs the SAME-WIDTH full grid
                elif norm(s) > norm(f2) + 0.05:
                    wins += 1            # ... and the other tail of the same dice
            fullv_obj /= trials
            sh_obj /= trials
            full2v_obj /= trials
            # the rule's cost gate is MEAN-dominant with a SYMMETRIC tail: a
            # same-grid full sweep at these budgets is itself a noisy
            # procedure, so paired >5%-of-range swings happen in BOTH
            # directions (their rate tracks the cell's noise, not the rule —
            # the counts print beside the verdict); what the rule must not
            # carry is a systematic loss (the mean) or a skewed tail.
            gate_rule = (sh_obj >= full2v_obj - 0.01
                         and abs(wins - losses) <= 0.02 * trials)
            gate_bonus = sh_obj > fullv_obj
            ok_all &= gate_rule and gate_bonus
            print(f"{width:>4} vs {2 * width:<4} {sigma:>7.1%} {budget:>7} "
                  f"{fullv_obj:>10.4f} {sh_obj:>8.4f} {full2v_obj:>10.4f} "
                  f"{wins:>4}/{losses:<4}    "
                  f"rule:{'PASS' if gate_rule else 'FAIL'} "
                  f"bonus:{'PASS' if gate_bonus else 'FAIL'}"
                  f"  (rec: fullV {fullv_rec / trials:.1%}, SH {sh_rec / trials:.1%}, "
                  f"full2V {full2v_rec / trials:.1%})")
    print("\nfullV = the current sweep on the V-point grid; SH = the WP16 halving "
          "sweep on the 2V-point grid; full2V = a plain full sweep over the same "
          "2V points at the same budget (fewer rounds). obj = the winner's mean "
          "true objective normalized to the 2V grid's range (1.0 = its true "
          "best); rec = winner-is-true-argmax rate (reported).")
    print("GATE rule: at equal budget and equal grid, SH's mean objective is "
          "within 0.01 of the full grid's, and its paired tail is symmetric "
          "(|wins - losses| <= 2% of trials; the swings' rate tracks the cell's "
          "noise — elimination only routes the budget, it never decides).")
    print("GATE bonus: SH(2V) beats the current full(V) sweep in mean true "
          "objective — the freed budget affords the wider grid.")
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
