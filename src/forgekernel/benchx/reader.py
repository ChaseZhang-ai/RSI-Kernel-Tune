"""Reading a bench's GPU job logs into a BenchResult.

The logs are the card's own: ``gpu_entry.sh`` then ``eval/bench.py`` print a human
summary and, last, ``=== JSON ===`` + the eval JSON. Correctness, the error text and the
stopwatch keys are read the way the loop reads them (:func:`forgekernel.loop.parse.
parse_eval`); the board's T_b / T_SOL and per-workload SOL scores come from the same
JSON's ``performance.leaderboard.sol_board`` block, as the card wrote them -- this module
never re-derives a reading the card already reported.

What it does derive is only what a single job cannot carry: the median across
``--repeat`` jobs, and the spread that says how noisy the reading is.
"""

from __future__ import annotations

import json
import statistics

from forgekernel import leaderboard as lbmod
from forgekernel.benchx.api import (COLD_START_BUDGET_S, COLD_START_MARKER,
                                    BenchResult, WorkloadReading)
from forgekernel.loop.parse import parse_eval

# The card's eval JSON, as eval/leaderboard.py writes it: an object carrying
# ``correctness`` or ``performance`` (the same rule parse_eval applies).
_JSON_KEYS = ("correctness", "performance")


def card_workload_names(problem: dict) -> list[str]:
    """The card's workload names, in card order (``_view_name``'s rule)."""
    group = problem.get("group") or []
    if group:
        return [str(g.get("name")) for g in group]
    return [str(problem["workload"].get("uuid", "workload"))]


def _card_workload_axes(problem: dict) -> list[dict]:
    group = problem.get("group") or []
    if group:
        return [dict((g.get("workload") or {}).get("axes") or {}) for g in group]
    return [dict(problem["workload"].get("axes") or {})]


def _view_name(problem: dict) -> str:
    """The name the card's verdict rows use for a one-view card (``_view_name``)."""
    group = problem.get("group") or []
    return str(group[0].get("name")) if group else str(problem["workload"].get("uuid"))


def eval_json(log: str) -> dict | None:
    """The last eval JSON object in ``log`` (None: the job printed no verdict)."""
    dec = json.JSONDecoder()
    found = None
    i, n = 0, len(log)
    while i < n:
        if log[i] == "{":
            try:
                obj, j = dec.raw_decode(log, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and any(k in obj for k in _JSON_KEYS):
                found = obj
            i = j
        else:
            i += 1
    return found


def parse_cold_start(log: str | None) -> float | None:
    """``cold_start_s`` from the probe job's ``[benchx] COLD_START_S <s>`` line (the
    first one, when the probe prints several), or None when the log carries none."""
    if not log:
        return None
    for line in log.splitlines():
        line = line.strip()
        if not line.startswith(COLD_START_MARKER):
            continue
        tail = line[len(COLD_START_MARKER):].strip().split()
        try:
            return float(tail[0])
        except (IndexError, ValueError):
            continue
    return None


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


class _JobReading:
    """One eval job's reading, before the repeat merge."""

    def __init__(self, log: str, problem: dict, job: str = ""):
        self.job = job
        self.eval = parse_eval(log)
        self.obj = eval_json(log)
        self.kind = ""
        self.clock = ""
        self.basis = ""
        names = card_workload_names(problem)
        self.names = names
        perf = (self.obj or {}).get("performance") or {}
        corr = (self.obj or {}).get("correctness") or {}
        lb_block = perf.get("leaderboard") or {}
        self.clock = str(lb_block.get("clock") or "")
        self.basis = str(perf.get("device_window_basis") or "")
        # Per-workload rows -- a group card's ``performance.workloads``; a one-workload
        # card's numbers are at the top level and named by the card itself.
        rows: dict[str, dict] = {}
        if isinstance(perf.get("workloads"), list) and perf["workloads"]:
            for i, w in enumerate(perf["workloads"]):
                if not isinstance(w, dict):
                    continue
                name = str(w.get("name") or (names[i] if i < len(names) else i))
                wlb = (w.get("leaderboard") or {}).get("sol_board") or {}
                rows[name] = {
                    "t_ms": _num(w.get("device_median_ms")) or _num(w.get("bwd_median_ms")),
                    "t_b": _num(wlb.get("t_b_ms")),
                    "t_sol": _num(wlb.get("t_sol_ms")),
                    "s": _num(wlb.get("sol_score")),
                }
            mean_s = _num((lb_block.get("sol_board") or {}).get("sol_score_mean"))
        else:
            board = lb_block.get("sol_board") or {}
            name = names[0] if len(names) == 1 else _view_name(problem)
            rows[name] = {
                "t_ms": _num(perf.get("device_median_ms")) or _num(perf.get("bwd_median_ms")),
                "t_b": _num(board.get("t_b_ms")),
                "t_sol": _num(board.get("t_sol_ms")),
                "s": _num(board.get("sol_score")),
            }
            mean_s = _num(board.get("sol_score"))
        axes = dict(zip(names, _card_workload_axes(problem)))
        self.rows: list[dict] = []
        for name in names:
            row = {"t_ms": None, "t_b": None, "t_sol": None, "s": None}
            row.update(rows.get(name) or {})
            row["name"] = name
            row["axes"] = axes.get(name, {})
            row["correct"] = None
            row["error"] = ""
            self.rows.append(row)
        # Workload verdicts (the card writes them for a group card; a one-workload card
        # has only the top-level verdict).
        verdict_rows = {str(w.get("name")): w for w in corr.get("workloads") or ()
                        if isinstance(w, dict)}
        for row in self.rows:
            v = verdict_rows.get(row["name"])
            if v is not None:
                row["correct"] = str(v.get("status")) == "PASSED"
        self.correct = bool(self.eval.correctness)
        for row in self.rows:
            if row["correct"] is None:
                row["correct"] = self.correct
        self.error = self.eval.error
        if not self.correct:
            # The card's own words; a group card prefixes them with the failed workload.
            for row in self.rows:
                if row["correct"] is False:
                    row["error"] = self.error
                    break
        self.mean_s = mean_s if self.correct else None
        self.fast1 = sum(1 for r in self.rows
                         if r["t_ms"] is not None and r["t_b"] is not None
                         and r["t_ms"] < r["t_b"]) if self.correct else 0
        if self.obj is None:
            self.kind = "no_verdict"
        elif not self.correct:
            self.kind = "incorrect"

    def timed(self) -> list[dict]:
        return [r for r in self.rows if r["t_ms"] is not None]


def read_job(log: str, problem: dict, job: str = "") -> _JobReading:
    return _JobReading(log, problem, job=job)


def cold_start_state(cold: float | None, *, probe_ran: bool, error: str = "") -> dict:
    """The ``integrity`` entries for a cold-start reading: status and, when it is not a
    reading, why. Never "ok" for a missing number -- a bench with an unmeasurable cold
    start says so."""
    if cold is None:
        why = error.strip() or (
            "the cold-start probe job printed no " + COLD_START_MARKER + " line"
            if probe_ran else "the cold-start probe job did not run")
        return {"cold_start": "unmeasured", "cold_start_note": why[:600]}
    if cold > COLD_START_BUDGET_S:
        return {"cold_start": "over-budget",
                "cold_start_note": (
                    f"cold start {cold:.1f}s exceeds the {COLD_START_BUDGET_S:.0f}s budget "
                    f"(the official evaluator's compile timeout is 120s and every "
                    f"evaluation compiles from scratch)")}
    return {"cold_start": "ok"}


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def build_result(logs: list[str], *, problem: dict, jobs: list[str] | None = None,
                 cold_log: str | None = None, cold_error: str = "",
                 elapsed_s: float = 0.0, extra_integrity: dict | None = None,
                 confidence: list[str] | None = None,
                 workload_subset: list[str] | None = None,
                 card_workloads: int | None = None) -> BenchResult:
    """Merge one bench's job logs into a BenchResult.

    ``logs`` are the ``all`` jobs (one, or ``--repeat`` of them). With several, each
    workload's reading is the median and ``integrity["repeat"]`` carries its spread; the
    SOL score of a merged reading is recomputed with the board's own formula
    (``leaderboard.sol_score``) from the median -- a single job's score is the card's.
    """
    jobs = list(jobs or ())
    job_readings = [read_job(log, problem, job=(jobs[i] if i < len(jobs) else ""))
                    for i, log in enumerate(logs)]
    integrity: dict = dict(extra_integrity or {})
    integrity.setdefault("repeat", {"n": len(job_readings)})
    names = [n for n in card_workload_names(problem)]

    # Cold start: the probe's number, or an explicit reason why there is none.
    cold = parse_cold_start(cold_log)
    integrity.update(cold_start_state(cold, probe_ran=cold_log is not None,
                                      error=cold_error))

    correct = all(r.correct for r in job_readings)
    error = ""
    if not correct:
        # The first job that failed speaks; its words are the card's.
        bad = next((r for r in job_readings if not r.correct), None)
        error = (bad.error if bad is not None else "") or "the bench reported no verdict"
        if bad is not None and bad.kind == "no_verdict" and "clock lock" not in error:
            integrity.setdefault("failure_kind", "no_verdict")
        elif bad is not None and bad.kind == "incorrect":
            integrity.setdefault("failure_kind", "incorrect")

    readings: list[WorkloadReading] = []
    spreads: dict[str, dict] = {}
    for idx, name in enumerate(names):
        per_job = [(r.rows[idx] if idx < len(r.rows) else None) for r in job_readings]
        rows = [row for row in per_job if row is not None]
        ok_rows = [row for row in rows if row["correct"] and row["t_ms"] is not None]
        axes = rows[0]["axes"] if rows else {}
        row_error = next((row["error"] for row in rows if row["error"]), "")
        verdict = next((row["correct"] for row in rows if row["correct"] is not None),
                       None)
        t_b = next((row["t_b"] for row in rows if row["t_b"] is not None), None)
        t_sol = next((row["t_sol"] for row in rows if row["t_sol"] is not None), None)
        s_card = next((row["s"] for row in rows if row["s"] is not None), None)
        if not correct or not ok_rows:
            # No timing happened (the card stops at the first failure): say what the
            # card said per workload, and never invent a reading.
            readings.append(WorkloadReading(
                name=name, axes=axes, t_b=t_b, t_sol=t_sol,
                correct=verdict if verdict is not None else (False if not correct else None),
                error=row_error))
            continue
        t_ms = (ok_rows[0]["t_ms"] if len(ok_rows) == 1
                else _median([row["t_ms"] for row in ok_rows]))
        if len(ok_rows) == 1:
            s = s_card                       # the card's own score, not re-derived
        else:
            s = (lbmod.sol_score(t_ms, t_b, t_sol)
                 if t_b is not None and t_sol is not None else None)
        if len(ok_rows) > 1:
            vals = [row["t_ms"] for row in ok_rows]
            spread = (max(vals) - min(vals)) / t_ms if t_ms else None
            spreads[name] = {"n": len(ok_rows), "min_ms": min(vals), "max_ms": max(vals),
                             "spread_pct": (round(spread * 100.0, 3) if spread is not None
                                            else None)}
        readings.append(WorkloadReading(name=name, axes=axes, t_ms=t_ms, t_b=t_b,
                                        t_sol=t_sol, s=s, correct=True))
    if spreads:
        integrity["repeat"] = {"n": len(job_readings), "workloads": spreads,
                               "worst_spread_pct": max(
                                   (v["spread_pct"] or 0.0) for v in spreads.values())}

    timed = [w for w in readings if w.t_ms is not None]
    if len(job_readings) == 1 and job_readings[0].correct:
        # One job: the card's mean score, as it wrote it.
        mean_s = job_readings[0].mean_s
    else:
        scores = [w.s for w in timed]
        mean_s = (sum(scores) / len(scores)
                  if scores and all(s is not None for s in scores) else None)
    fast1 = sum(1 for w in timed
                if w.t_ms is not None and w.t_b is not None and w.t_ms < w.t_b) \
        if correct else 0

    # The reading's qualifiers: which clock it was taken under, which stopwatch, and
    # any lighting the caller asked for.
    clocks = [c for c in (r.clock for r in job_readings) if c]
    if clocks:
        integrity.setdefault("clock", clocks[0] if len(set(clocks)) == 1
                             else "mixed: " + "; ".join(dict.fromkeys(clocks)))
        if clocks[0].split(" ", 1)[0] != "locked" and problem.get("sol_board"):
            integrity.setdefault(
                "clock_note", "the reading was not taken under the board's clock lock; "
                              "T_b / T_SOL are readings at that lock and an unlocked "
                              "reading is faster (the card's sol_board says the same)")
    bases = [b for b in (r.basis for r in job_readings) if b]
    if bases:
        integrity.setdefault("reading_basis", bases[0] if len(set(bases)) == 1
                             else "mixed: " + "; ".join(dict.fromkeys(bases)))
    if workload_subset is not None:
        # ``problem`` is the (trimmed) card the job ran; ``card_workloads`` names the
        # full card it came from, so the subset is visible against its real size.
        integrity["workload_subset"] = {
            "requested": list(workload_subset),
            "card_workloads": (card_workloads if card_workloads is not None else len(names))}
    return BenchResult(
        correct=correct, workloads=readings, mean_s=mean_s, fast1=fast1,
        integrity=integrity, job=",".join(j for j in jobs if j), elapsed_s=elapsed_s,
        error=error, cold_start_s=cold,
        confidence=list(confidence or ["dev-only"]))
