"""SOL scores: one place, shared by campaign / evolve / shapes.

A SOL-ExecBench workload is scored ``S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))``
against the board's frozen ``T_b`` (the scoring baseline's latency) and ``T_SOL`` (the
SOLAR hardware floor) -- `forgekernel.leaderboard.sol_score` is that function and is
imported here, never re-derived. A problem's score is the arithmetic mean of its
workloads' S, and a track's score is the sum of its problems' scores over the track's
problem count (a problem with no submission scores 0) -- the arithmetic this module
adds on top is the per-workload/per-problem aggregation and the readings a card's
``problem.json`` carries.

Two shapes of card exist and both are handled:

* **one-workload card**: ``problem.json`` has a top-level ``sol_board`` and the eval
  JSON's ``performance.leaderboard.sol_board`` is that block with one ``sol_score``;
* **group card** (``--workload 0-15``, what a campaign uses): ``problem.json`` has
  ``group: [{index, name, workload, sol_board}, ...]`` and the eval JSON's block is
  ``{"sol_score_mean": x, "sol_score": {"wl0": ..., ...}}``.

A candidate's per-workload latencies come from a bench reading's
``workloads[].device_median_ms`` (the device window the board's own clock uses). The
card is the source of ``T_b`` / ``T_SOL``: a card built with ``--sol-board`` carries
them; a card built with ``--no-sol-board`` carries none and then a reading has **no**
score -- that is reported as ``s=None``, never as a zero and never as an estimate
scaled from means (the board computes S per workload, and averaging latencies first
would invent a different number).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from forgekernel.leaderboard import sol_score, time_for_score

# The problems of each track, by the board's kernel id. Verified against the 2026-11-24
# snapshot: the per-track mean of a user's per-problem scores reproduces that user's
# collection score to every printed digit (background §3.1).
TRACK_RANGES = {"l1": (1, 94), "l2": (95, 176), "quant": (177, 209),
                "flashinfer-bench": (210, 235)}
# The dataset directory each track is downloaded into (download_solexecbench.py).
TRACK_SUBSET = {"l1": "L1", "l2": "L2", "quant": "Quant",
                "flashinfer-bench": "FlashInfer-Bench"}
TRACKS = tuple(TRACK_RANGES)
TRACK_TITLE = {"l1": "L1 - Single Operations", "l2": "L2 - Fused Operations",
               "quant": "Quantization", "flashinfer-bench": "FlashInfer-Bench"}

PROBLEM_FILE = "eval/leaderboard/problem.json"
DEFAULT_STEP = 1.10


def track_of(kernel_id: int) -> str | None:
    for track, (lo, hi) in TRACK_RANGES.items():
        if lo <= int(kernel_id) <= hi:
            return track
    return None


def track_size(track: str) -> int:
    lo, hi = TRACK_RANGES[track]
    return hi - lo + 1


def geomean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v and float(v) > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def workload_s(t_k: float | None, t_b: float | None, t_sol: float | None) -> float | None:
    """One workload's SOL score, or None when the latency or the board's pair is missing."""
    if t_k is None or t_b is None or t_sol is None:
        return None
    if float(t_k) <= 0:
        return None
    return float(sol_score(float(t_k), float(t_b), float(t_sol)))


def reading_s(readings: dict, board: dict | None) -> dict:
    """The SOL block of one latency reading: ``{s_mean, fast1, s_by_workload, n}``.

    ``readings`` is ``{workload_name: t_ms}`` (a bench reading's device window).
    ``board`` is the card's board mapping (see `card_boards`): either the flat
    ``{t_b_ms, t_sol_ms}`` of a one-workload card (applied to every reading) or
    ``{name: {t_b_ms, t_sol_ms}}``. Missing board or missing reading -> ``s`` of that
    workload is None; ``s_mean`` is then the mean over the workloads that do have both,
    and None when none does (``scored``/``unscored`` say how many were counted)."""
    flat = _flat_board(board)
    per = _named_board(board)
    scores: dict[str, float | None] = {}
    fast1 = 0
    scored = 0
    for name, t_k in readings.items():
        b = per.get(name, flat) if per else flat
        t_b = (b or {}).get("t_b_ms")
        t_sol = (b or {}).get("t_sol_ms")
        s = workload_s(t_k, t_b, t_sol)
        scores[name] = s
        if s is not None:
            scored += 1
            if t_b is not None and float(t_k) < float(t_b):
                fast1 += 1
    vals = [s for s in scores.values() if s is not None]
    return {"s_mean": (sum(vals) / len(vals)) if vals else None,
            "fast1": fast1, "fast1_total": len(readings), "scored": scored,
            "unscored": len(readings) - scored, "s_by_workload": scores}


def _flat_board(board: dict | None) -> dict | None:
    if board and "t_b_ms" in board:
        return board
    return None


def _named_board(board: dict | None) -> dict:
    if not board or "t_b_ms" in board:
        return {}
    return {k: v for k, v in board.items() if isinstance(v, dict)}


def card_boards(card_dir: str | Path) -> dict | None:
    """The card's ``T_b`` / ``T_SOL`` per workload: flat for a one-workload card,
    ``{name: {...}}`` for a group card (the group entry's name is ``wl<index>``, the
    name the eval JSON reports). None when the card carries no ``sol_board``."""
    path = Path(card_dir) / PROBLEM_FILE
    problem = json.loads(path.read_text(encoding="utf-8"))
    group = problem.get("group") or []
    if group:
        out = {g["name"]: g["sol_board"] for g in group if g.get("sol_board")}
        return out or None
    return problem.get("sol_board")


def card_workloads(card_dir: str | Path) -> list[dict]:
    """The card's workloads in index order: ``[{name, axes, uuid, board}]``."""
    problem = json.loads((Path(card_dir) / PROBLEM_FILE).read_text(encoding="utf-8"))
    group = problem.get("group") or []
    if group:
        return [{"name": g["name"], "index": g.get("index"),
                 "axes": (g.get("workload") or {}).get("axes") or {},
                 "uuid": (g.get("workload") or {}).get("uuid"),
                 "board": g.get("sol_board")} for g in group]
    wl = problem.get("workload") or {}
    return [{"name": "wl0", "index": 0, "axes": wl.get("axes") or {},
             "uuid": wl.get("uuid"), "board": problem.get("sol_board")}]


def problem_mean_s(reading: dict, board: dict | None) -> float | None:
    """The problem's SOL score of one reading = mean over its workloads (None without a board)."""
    return reading_s(reading, board)["s_mean"]


def target_ms(readings: dict, *, step: float = DEFAULT_STEP) -> float | None:
    """The ``[target] ms`` a campaign task gets from a starting point's reading.

    The card's gate on a group card is the geometric mean of the per-workload readings
    (`card_template/eval/leaderboard.py`), so the target is that same aggregate divided
    by ``step`` -- one number the loop can compare its confirmation against."""
    g = geomean(readings.values())
    return (g / float(step)) if g else None


def score_time(s: float, t_b: float, t_sol: float) -> float | None:
    """The latency that scores ``s`` (the board's own inversion; kept as one import point)."""
    return time_for_score(s, t_b, t_sol)


def score_of_latency(t_k: float, t_b: float, t_sol: float) -> float:
    return float(sol_score(t_k, t_b, t_sol))


def mean_s_of_scores(scores: Iterable[float | None]) -> float | None:
    vals = [float(s) for s in scores if s is not None]
    return (sum(vals) / len(vals)) if vals else None


def track_mean(per_problem: dict[int, float | None], track: str) -> float:
    """A track's score: every problem of the track counts, an unscored one as 0."""
    lo, hi = TRACK_RANGES[track]
    total = sum(float(per_problem.get(k) or 0.0) for k in range(lo, hi + 1))
    return total / (hi - lo + 1)


def weighted_overall(per_track: dict[str, float]) -> float:
    """The leaderboard's overall score: the four track scores weighted by problem count."""
    n = sum(track_size(t) for t in TRACK_RANGES)
    return sum(per_track.get(t, 0.0) * track_size(t) for t in TRACK_RANGES) / n


# --- score_kernel: the entry `forge evolve` and `forge shapes` import --------------------

def score_kernel(reading: dict, board: dict | None, *, mean: bool = True):
    """``forgekernel.campaign.solscore.score_kernel`` -- the interface the evolve and
    shapes task files name.

    ``reading``: ``{workload_name: latency_ms}`` or ``{"workloads": [{name,
    device_median_ms}, ...]}``. ``board``: `card_boards`' mapping (or None). Returns the
    problem's mean SOL score (``mean=True``) or the full block from `reading_s`."""
    block = reading_s(_latencies(reading), board)
    return block["s_mean"] if mean else block


def _latencies(reading: dict) -> dict:
    if "workloads" in reading and isinstance(reading["workloads"], list):
        out = {}
        for w in reading["workloads"]:
            t = w.get("device_median_ms", w.get("t_ms"))
            if t is not None:
                out[str(w.get("name"))] = float(t)
        return out
    return {str(k): float(v) for k, v in reading.items() if v is not None}
