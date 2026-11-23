"""Comparing bench results -- pure functions, no GPU, no jobs.

Both functions take ``{label: BenchResult}`` in the caller's order, which is meaningful:
``best_of`` treats the first candidate as the simpler one, so callers pass candidates
"simplest implementation first".
"""

from __future__ import annotations

import math

from forgekernel.benchx.api import BenchResult

NOISE_FLOOR_PCT = 1.5
"""Readings within 1.5% of each other are a tie for ``best_of`` (an unchanged kernel
moves a few percent between jobs on the same node; a shorter difference is not a win)."""


def geometry(values: list[float]) -> float | None:
    """Geometric mean of the positive values, or None when there is none."""
    vals = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _workload_order(results: dict[str, BenchResult]) -> list[str]:
    seen: list[str] = []
    for result in results.values():
        for w in result.workloads:
            if w.name not in seen:
                seen.append(w.name)
    return seen


def compare(results: dict[str, BenchResult]) -> dict:
    """Per-workload table + summary.

    ``{"workloads": [{"workload", "entries": [{"candidate", "t_ms", "s", "correct",
    "delta_pct" (vs the best timed reading)}], "fastest", "best_t_ms"}],
    "summary": {candidate: {"correct", "mean_s", "fast1", "fast1_ratio",
    "geomean_ms", "n_timed"}}}``
    """
    labels = list(results)
    names = _workload_order(results)
    rows = []
    for name in names:
        entries = []
        timed: list[tuple[str, float]] = []
        for label in labels:
            w = next((x for x in results[label].workloads if x.name == name), None)
            if w is None:
                entries.append({"candidate": label, "t_ms": None, "s": None,
                                "correct": None, "delta_pct": None})
                continue
            entries.append({"candidate": label, "t_ms": w.t_ms, "s": w.s,
                            "correct": w.correct, "delta_pct": None})
            if w.t_ms is not None and w.correct:
                timed.append((label, w.t_ms))
        best_t = min((t for _, t in timed), default=None)
        fastest = next((label for label, t in timed if t == best_t), None)
        for e in entries:
            if best_t and e["t_ms"] is not None:
                e["delta_pct"] = round((e["t_ms"] / best_t - 1.0) * 100.0, 3)
        rows.append({"workload": name, "entries": entries, "fastest": fastest,
                     "best_t_ms": best_t})
    summary = {}
    for label in labels:
        r = results[label]
        timed = r.timed()
        summary[label] = {
            "correct": r.correct,
            "mean_s": r.mean_s,
            "fast1": r.fast1,
            "fast1_ratio": (round(r.fast1 / len(timed), 4) if timed else None),
            "geomean_ms": (round(geometry([w.t_ms for w in timed]), 6)
                           if timed else None),
            "n_timed": len(timed),
        }
    return {"workloads": rows, "summary": summary}


def best_of(results: dict[str, BenchResult]) -> dict:
    """Per-workload routing table + the fallback candidate.

    A workload's winner is the fastest *correct* reading; a candidate within
    ``NOISE_FLOOR_PCT`` of that reading is a tie and the earlier candidate (the simpler
    one, by the caller's order) wins. The fallback is the candidate with the highest mean
    SOL score -- what to run when a workload has no winner at all.

    ``{"routes": [{"workload", "winner" (a label) or None, "t_ms", "s",
    "within_noise": [labels]}], "fallback": label or None, "fallback_mean_s": float}``
    """
    labels = list(results)
    routes = []
    for name in _workload_order(results):
        rows = []
        for label in labels:
            w = next((x for x in results[label].workloads if x.name == name), None)
            if w is not None and w.correct and w.t_ms is not None:
                rows.append((label, w))
        if not rows:
            routes.append({"workload": name, "winner": None, "t_ms": None, "s": None,
                           "within_noise": []})
            continue
        best_t = min(w.t_ms for _, w in rows)
        floor = best_t * (1.0 + NOISE_FLOOR_PCT / 100.0)
        within = [(label, w) for label, w in rows if w.t_ms <= floor]
        winner, w = within[0]                   # rows keep the caller's order
        routes.append({"workload": name, "winner": winner, "t_ms": w.t_ms, "s": w.s,
                       "within_noise": [label for label, _ in within]})
    fallback, best_s = None, None
    for label in labels:
        r = results[label]
        if not r.correct or r.mean_s is None:
            continue
        if best_s is None or r.mean_s > best_s:
            fallback, best_s = label, r.mean_s
    return {"routes": routes, "fallback": fallback, "fallback_mean_s": best_s}
