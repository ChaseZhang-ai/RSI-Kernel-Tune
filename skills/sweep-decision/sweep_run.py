"""Driver-seeded SWEEP run — time a teacher-authored variant sweep on the
current kernel, base and every variant interleaved, in one GPU job.

Submitted by the loop driver when the student ends a turn with
`SWEEP_REQUEST: notes/<decision_id>.json` at a milestone decision whose line
pre-writes a sweep recipe. The driver validates the submitted file against the
milestone's recipe (canonical-JSON hash; a mismatch is refused without a job —
the student cannot smuggle variants in), then submits
`cmd:python3 notes/sweep_run.py notes/<decision_id>.json`, cwd = repo root.
Not a student tool and not a lever-pricing replacement: the recipe is
teacher-authored in the milestone, the sweep only MEASURES it, and the winner
is landed by the student as an ordinary decision through the standard eval
gate. This file exists so every commit after seeding carries the frozen recipe
(git archive ships committed files only).

Spec file (JSON; the milestone's pre-written recipe, copied verbatim by the
student):

    {"kernel_file": "kernel.py", "rounds": 3,
     "criterion": "min same-job ratio above floor",
     "variants": [
       {"name": "tile64", "edits": [{"old": "<exact text>", "new": "<replacement>"}, ...]},
       ...]}

At most 8 variants (the driver refuses more; a `halving: true` recipe may
carry up to 32 — the driver expands it and its refusal ceiling moves with it).
Each variant is a list of exact-text replacements applied to the kernel file in
order — the same edit discipline as a pricing lever: every `old` must occur
exactly once at apply time, a no-op variant is refused, and a variant that
fails its edit refuses the whole sweep (SWEEP-FAIL reason=sweep-edit).
`criterion` is carried for the log; the mechanical rule is fixed: the winner is
the variant whose same-job paired ratio against base (vs_ref_device when the
bench carries a reference, inverted device ms otherwise) is best among those
whose delta clears the job's own noise floor — the same reading discipline as
the pricing table.

Successive-Halving (WP16, B-A4; spec key `"halving": true`): after each
completed round but the last, the worst-performing half of the surviving
variants is eliminated (correctness-dead variants were already kicked; `base`
is never eliminated) and the freed budget goes to the survivors — the recipe's
rounds still run, and leftover pass budget buys the survivors extra interleaved
rounds while two or more remain. The elimination is a BUDGET routing only: it
decides who is timed next round, never a verdict — the final table and winner
are read over every variant that ever produced a reading, exactly as without
halving (a variant eliminated at round one on one noisy draw can still win).
Each elimination prints a line the driver logs:

    SWEEP HALVING round=<r> kept=<a,b,...> eliminated=<x,y,...>

Machine-readable output (the driver parses these):

    SWEEP name=<n> ratio=<r|n/a> floor=<f%> delta=<+x.xx%|n/a> ms=<m|n/a> ok=<pass|fail>
    SWEEP-WINNER name=<n>
    SWEEP-OK variants=<n> rounds=<r> base_ms=<..>
    SWEEP-FAIL reason=<...>

The interleaved timing, node hygiene (process-group kills, NODE-DIRTY), the
noise floor (3σ of a difference of two R-round medians off the device-ms
spread) and the same-job ratio reading are lever_pricing_run's, imported from
the sibling seeded file — this script parameterizes that kernel over an
explicit variant list instead of the mechanical 2N+2 lever design.

Timeouts (the pricing tool's own precedents): one bench call is capped at
SWEEP_BENCH_TIMEOUT and the whole timed pass at SWEEP_BUDGET_S, each clamped
to the job clock's remaining time (TASK_TIMEOUT) so the closing lines always
land before gpu_entry's kill.
"""
import json
import os
import re
import sys
import time

try:                                    # vendored flat in notes/ (notes/ on sys.path)
    import lever_pricing_run as lpr
except ImportError:                     # the harness imports it as a skill
    from peregrine_kernel.skills import lever_pricing_run as lpr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KERNEL = "kernel.py"
MAX_VARIANTS = 8
MAX_VARIANTS_HALVING = 32     # a `halving` recipe's ceiling (the driver's
                              # _SWEEP_MAX_VARIANTS gate moves with this flag)
CRITERION = "min same-job ratio above floor"
# one bench call / the whole timed pass: the pricing tool's NCU_TIME_TIMEOUT /
# ABLATION_BUDGET_S precedent, both clamped to what the job clock still leaves
SWEEP_BENCH_TIMEOUT = 900
SWEEP_BUDGET_S = 3600


def sweep_name(variant: dict) -> str:
    """The name as it appears on the SWEEP line: one token (whitespace -> _)."""
    return re.sub(r"\s+", "_", str(variant.get("name", "")).strip()) or "unnamed"


def build_variants(base_src: str, variants: list, *, halving: bool = False) -> dict:
    """name -> source. Refuses (raises) on more than MAX_VARIANTS entries
    (MAX_VARIANTS_HALVING under a `halving` recipe), a duplicate / reserved
    name, a non-unique edit or a no-op variant — the sweep measures exactly
    the teacher's recipe, nothing derived."""
    ceiling = MAX_VARIANTS_HALVING if halving else MAX_VARIANTS
    if len(variants) > ceiling:
        raise ValueError(f"{len(variants)} variants (max {ceiling})")
    out = {"base": base_src}
    seen = set()
    for var in variants:
        name = sweep_name(var)
        if name in ("base",) or name.startswith("all") or name in seen:
            raise ValueError(f"variant name {name!r} is reserved or duplicated")
        seen.add(name)
        src = lpr.apply_lever(base_src, var)          # exact-once edits, raises
        if src == base_src:
            raise ValueError(f"variant {name!r} is byte-identical to base (no-op)")
        out[name] = src
    return out


def main(argv=None) -> int:
    t_job0 = time.monotonic()       # the job's kill clock, as near its start as this process sees
    spec_path = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) \
        else os.path.join("notes", "sweep.json")
    try:
        spec = json.load(open(os.path.join(ROOT, spec_path)))
    except (OSError, json.JSONDecodeError) as e:
        print(f"SWEEP-FAIL reason=spec-unreadable ({e})", flush=True)
        return 0
    variants = spec.get("variants") or []
    rounds = max(1, int(spec.get("rounds", 3)))
    halving = bool(spec.get("halving"))
    kernel = str(spec.get("kernel_file") or DEFAULT_KERNEL)
    criterion = str(spec.get("criterion") or CRITERION)
    if not variants:
        print("SWEEP-FAIL reason=no-variants", flush=True)
        return 0
    kpath = os.path.join(ROOT, kernel)
    try:
        base_src = open(kpath).read()
    except OSError as e:
        print(f"SWEEP-FAIL reason=kernel-unreadable ({kernel}: {e})", flush=True)
        return 0
    try:
        sources = build_variants(base_src, variants, halving=halving)
    except ValueError as e:
        print(f"SWEEP-FAIL reason=sweep-edit ({e})", flush=True)
        return 0
    order = lpr.paired_order([v for v in sources if v != "base"])
    # base, then the recipe's variants in order (the screening primitive's order)
    print(f"SWEEP kernel_file={kernel} criterion={criterion}"
          f"{' halving=on' if halving else ''} variants: {' '.join(order)}",
          flush=True)
    dev = {v: [] for v in order}
    ratio = {v: [] for v in order}
    ok = {v: True for v in order}
    dead = set()
    eliminated = set()
    # the pass's own budget, never more than the job's clock leaves (the closing
    # lines must land before gpu_entry's kill — the ablation pass's precedent)
    budget = float(SWEEP_BUDGET_S)
    left = lpr.job_seconds_left(t_job0)
    if left is not None:
        budget = lpr.clamp_pass_budget(budget, left, lpr.ROLE_PASS_MARGIN_S)
    t0 = time.monotonic()
    try:
        rnd = 0
        # WP16 (B-A4): a halving pass keeps its recipe's rounds, and the budget
        # the eliminated variants free up buys the survivors extra interleaved
        # rounds (capped at 2x the recipe's rounds — the budget guard bounds it
        # either way).
        max_rounds = rounds * 2 if halving else rounds
        while rnd < max_rounds:
            if time.monotonic() - t0 > budget:
                print(f"SWEEP ROUND {rnd} skipped: pass budget {budget:.0f}s spent", flush=True)
                break
            alive = [v for v in order[1:] if v not in dead and v not in eliminated]
            if halving and rnd >= rounds and len(alive) < 2:
                break
            for v in order:
                if v in dead or v in eliminated:
                    continue
                open(kpath, "w").write(sources[v])
                j = lpr.run_bench(v, keep_cubin=False, timeout=SWEEP_BENCH_TIMEOUT)
                if lpr.node_dirty():
                    # a timed-out bench left the GPU held: every later reading
                    # would be the orphan's share — stop; the driver re-runs
                    # the same recipe once on a fresh node (the pricing rule)
                    print(f"SWEEP-FAIL reason=node-dirty ({lpr.node_dirty()})", flush=True)
                    return 0
                if j is None or not j.get("correctness", {}).get("pass"):
                    ok[v] = False
                    dead.add(v)
                    print(f"SWEEP ROUND {rnd} {v} pass=False", flush=True)
                    if v == "base":
                        print("SWEEP-FAIL reason=base-failed-correctness", flush=True)
                        return 0
                    continue
                perf = j.get("performance") or {}
                dev[v].append(perf.get("device_median_ms"))
                ratio[v].append(lpr.ratio_reading(perf))
                print(f"SWEEP ROUND {rnd} {v} pass=True device={perf.get('device_median_ms')} "
                      f"vs_ref_device={lpr.ratio_reading(perf)}", flush=True)
            rnd += 1
            if halving:
                cut = halving_cut(order, ok, dev, ratio, dead, eliminated)
                if cut:
                    eliminated |= cut
                    kept = [v for v in order[1:] if v not in dead and v not in eliminated]
                    print(f"SWEEP HALVING round={rnd} kept={','.join(kept) or 'none'} "
                          f"eliminated={','.join(sorted(cut))}", flush=True)
        score, delta, use_ratio = _make_scorer(order, ok, dev, ratio)
        noise = lpr.noise_floor(order, ok, dev)
        for v in order:
            d = delta(v, "base") if v != "base" else None
            print(f"SWEEP name={v} ratio={_num(med(ratio[v]) if use_ratio else None)} "
                  f"floor={noise:.2f}% delta={_pct(d)} ms={_num(lpr.med(dev[v]))} "
                  f"ok={'pass' if ok[v] else 'fail'}", flush=True)
        winner = _winner(order, score, delta, noise)
        if winner is None:
            print(f"SWEEP-OK variants={len(order) - 1} rounds={len(dev['base'])} "
                  f"base_ms={lpr.med(dev['base'])} winner=none", flush=True)
            print("SWEEP-WINNER name=none (no variant clears the floor — keep the base)",
                  flush=True)
        else:
            print(f"SWEEP-OK variants={len(order) - 1} rounds={len(dev['base'])} "
                  f"base_ms={lpr.med(dev['base'])}", flush=True)
            print(f"SWEEP-WINNER name={winner}", flush=True)
    finally:
        open(kpath, "w").write(base_src)      # leave the tree as we found it
    return 0


def _make_scorer(order, ok, dev, ratio):
    """(score, delta, use_ratio): the same-job paired ratio when the bench
    carries a reference (higher = faster), else inverted device ms — the
    pricing tool's scorer, one place."""
    return lpr.make_scorer(order, ok, dev, ratio)


def med(xs):
    return lpr.med(xs)


def _num(x):
    return "n/a" if x is None else repr(round(x, 4)) if isinstance(x, float) else str(x)


def _pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def halving_cut(order, ok, dev, ratio, dead, eliminated) -> set:
    """WP16 (B-A4): the worst half of the still-timed variants, NOISE-GATED.
    `base` and the already-gone are never touched; fewer than four survivors
    cut nobody; a first round with a single reading per variant cuts nobody
    (one draw names no rank); and a variant sitting within the job's own noise
    floor of the cut line survives the cut — a within-noise gap is a dice
    roll, not a ranking (the MC prescreen defense's discipline, read through
    lpr.noise_floor, the final table's same rule). At a noisy node the cut
    therefore stands down and the pass degenerates toward the plain sweep —
    the safe direction. A budget ROUTING only: the cut decides who is timed
    next round, never a verdict (the final winner is read over every variant
    that ever produced a reading, exactly as without halving)."""
    alive = [v for v in order if v != "base" and v not in dead and v not in eliminated]
    if len(alive) < 4:
        return set()
    if any(len(dev[v]) < 2 for v in alive):
        return set()                      # one draw per variant names no rank
    score, _delta, _use = _make_scorer(order, ok, dev, ratio)
    floor = lpr.noise_floor(order, ok, dev)     # percent; the final table's rule
    ranked = sorted(alive, key=lambda v: (score(v) is None, -(score(v) or 0.0)))
    line = (len(ranked) + 1) // 2
    cut = [v for v in ranked[line:]]
    boundary = score(ranked[line - 1])
    if boundary is None:
        return set()
    cut = [v for v in cut if score(v) is None
           or 100.0 * (boundary / score(v) - 1.0) > floor]
    return set(cut)


def _winner(order, score, delta, noise) -> str | None:
    """The variant the criterion names: the best score among those whose delta
    vs base is measured and above the noise floor; None keeps the base."""
    best, best_score = None, None
    for v in order:
        if v == "base":
            continue
        d = delta(v, "base")
        if d is None or d <= noise:
            continue
        s = score(v)
        if s is not None and (best_score is None or s > best_score):
            best, best_score = v, s
    return best


if __name__ == "__main__":
    sys.exit(main())
