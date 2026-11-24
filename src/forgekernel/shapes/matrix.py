"""The cross-evaluation matrix and the routing table: pure functions over readings.

Rows are candidates, columns are the card's workloads, cells are the card's own reading
(``t_ms``), its verdict (``correct``) and the SOL score it implies. Nothing here benches
anything: the readings come from :mod:`forgekernel.shapes.evalx` (hence from benchx), and
only the arithmetic and the choice are here, so both can be tested without a GPU.

The choice (task §5.3):

* **the fastest correct reading wins**, per workload;
* **a tie is not a win**: two readings within the noise floor (1.5%, the same number
  ``benchx.compare`` uses) are a tie, and the tie goes to the earlier candidate in the
  preference order -- cluster specialist, then that run's other correct versions, then the
  overall solution, then extra solutions, then torch baselines;
* **the eager reference is the floor, not a competitor**: it is measured on the same card
  and the routed choice is never allowed to be slower than it beyond the noise floor
  unless ``--allow-slower`` says so. When the floor binds, the workload routes to the
  reference and the route says why -- a submission that is never slower than eager is
  worth more than one that is faster on average;
* a workload with **no correct candidate** routes to the fallback candidate (the overall
  solution when one was scored, else the reference), and the generated dispatcher keeps a
  try/except around every call anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.shapes import SCHEMA
from forgekernel.shapes import cluster as cluster_mod
from forgekernel.shapes import evalx

NOISE_FLOOR_PCT = 1.5
"""Readings within this percentage of each other are a tie (benchx.compare's number)."""


@dataclass
class Cell:
    t_ms: float | None = None
    s: float | None = None
    correct: bool | None = None
    error: str = ""

    @property
    def timed(self) -> bool:
        return self.t_ms is not None and self.correct is True

    def as_json(self) -> dict:
        return {"t_ms": self.t_ms, "s": self.s, "correct": self.correct, "error": self.error}


@dataclass
class Row:
    """One candidate's whole reading."""
    label: str
    kind: str
    correct: bool = False
    mean_s: float | None = None
    fast1: int = 0
    cold_start_s: float | None = None
    error: str = ""
    integrity: dict = field(default_factory=dict)
    confidence: list[str] = field(default_factory=list)
    sha: str = ""
    cells: dict[str, Cell] = field(default_factory=dict)

    def cell(self, workload: str) -> Cell:
        return self.cells.get(workload) or Cell()

    def as_json(self) -> dict:
        return {"label": self.label, "kind": self.kind, "correct": self.correct,
                "mean_s": self.mean_s, "fast1": self.fast1,
                "cold_start_s": self.cold_start_s, "error": self.error,
                "confidence": list(self.confidence), "sha": self.sha,
                "integrity": dict(self.integrity),
                "cells": {k: v.as_json() for k, v in self.cells.items()}}


@dataclass
class Matrix:
    workloads: list[cluster_mod.Workload]
    rows: list[Row]
    card: str = ""
    card_sha: str = ""
    noise_floor_pct: float = NOISE_FLOOR_PCT
    notes: list[str] = field(default_factory=list)
    created: str = ""

    def by_label(self) -> dict[str, Row]:
        return {r.label: r for r in self.rows}

    def row(self, label: str) -> Row | None:
        return next((r for r in self.rows if r.label == label), None)

    def labels(self) -> list[str]:
        return [r.label for r in self.rows]

    def as_json(self) -> dict:
        return {"schema": SCHEMA, "created": self.created or _now(), "card": self.card,
                "card_sha": self.card_sha, "noise_floor_pct": self.noise_floor_pct,
                "workloads": [w.as_json() for w in self.workloads],
                "rows": [r.as_json() for r in self.rows], "notes": list(self.notes)}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_json(), indent=1, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Matrix":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = []
        for r in doc.get("rows") or []:
            rows.append(Row(label=str(r.get("label")), kind=str(r.get("kind") or "extra"),
                            correct=bool(r.get("correct")), mean_s=r.get("mean_s"),
                            fast1=int(r.get("fast1") or 0),
                            cold_start_s=r.get("cold_start_s"), error=str(r.get("error") or ""),
                            confidence=list(r.get("confidence") or ()), sha=str(r.get("sha") or ""),
                            integrity=dict(r.get("integrity") or {}),
                            cells={k: Cell(**_cell_kwargs(v))
                                   for k, v in (r.get("cells") or {}).items()}))
        return cls(workloads=[cluster_mod.Workload.from_json(w)
                              for w in doc.get("workloads") or ()],
                   rows=rows, card=str(doc.get("card") or ""),
                   card_sha=str(doc.get("card_sha") or ""),
                   noise_floor_pct=float(doc.get("noise_floor_pct") or NOISE_FLOOR_PCT),
                   notes=list(doc.get("notes") or ()), created=str(doc.get("created") or ""))


def _cell_kwargs(data: dict) -> dict:
    return {k: data.get(k) for k in ("t_ms", "s", "correct", "error") if k in data}


def from_readings(readings: list[tuple[evalx.Candidate, object]], *,
                  workloads: list[cluster_mod.Workload], card: str = "",
                  card_sha: str = "", notes: list[str] | None = None,
                  now: str | None = None) -> Matrix:
    """Build the matrix from ``[(candidate, BenchResult)]`` (a dict or a BenchResult).

    A reading's ``workloads`` list is matched to the card's workloads by name; a workload
    the reading does not mention is an absent cell (not a zero)."""
    rows: list[Row] = []
    for cand, result in readings:
        get = (lambda k, d=None: result.get(k, d)) if isinstance(result, dict) \
            else (lambda k, d=None: getattr(result, k, d))
        overall = bool(get("correct", False))
        cells: dict[str, Cell] = {}
        for w in get("workloads") or ():
            wget = (lambda k, d=None: w.get(k, d)) if isinstance(w, dict) \
                else (lambda k, d=None: getattr(w, k, d))
            name = str(wget("name") or "")
            t_ms = _num(wget("t_ms"))
            correct = wget("correct")
            if correct is None:
                # A reading without a per-workload verdict carries the card's own: the
                # card stops at the first failure, so a timed workload of a correct run
                # is correct (benchx's reader does the same).
                correct = overall and t_ms is not None
            cells[name] = Cell(t_ms=t_ms, s=_num(wget("s")), correct=correct,
                               error=str(wget("error") or ""))
        for w in workloads:
            cells.setdefault(w.name, Cell(correct=False, error="the reading does not "
                                                            "mention this workload"))
        rows.append(Row(label=cand.label, kind=cand.kind, correct=overall,
                        mean_s=_num(get("mean_s")), fast1=int(get("fast1") or 0),
                        cold_start_s=_num(get("cold_start_s")),
                        error=str(get("error") or ""), integrity=dict(get("integrity") or {}),
                        confidence=list(get("confidence") or ()), sha=cand.sha, cells=cells))
    return Matrix(workloads=list(workloads), rows=rows, card=str(card), card_sha=str(card_sha),
                  notes=list(notes or ()), created=now or _now())


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- routing ---------------------------------------------------------------------------

@dataclass
class Route:
    workload: str
    index: int
    winner: str | None = None
    winner_kind: str = ""
    t_ms: float | None = None
    s: float | None = None
    t_b: float | None = None
    t_sol: float | None = None
    eager_t_ms: float | None = None
    vs_eager: float | None = None          # routed t_ms / eager t_ms
    within_noise: list[str] = field(default_factory=list)
    slower_than_reference: bool = False
    reason: str = ""

    def as_json(self) -> dict:
        return {"workload": self.workload, "index": self.index, "winner": self.winner,
                "winner_kind": self.winner_kind, "t_ms": self.t_ms, "s": self.s,
                "t_b": self.t_b, "t_sol": self.t_sol, "eager_t_ms": self.eager_t_ms,
                "vs_eager": self.vs_eager, "within_noise": list(self.within_noise),
                "slower_than_reference": self.slower_than_reference, "reason": self.reason}


@dataclass
class Routes:
    routes: list[Route]
    fallback: str | None = None
    fallback_kind: str = ""
    eager: str = ""
    mean_s: float | None = None
    mean_s_eager: float | None = None
    guarantee: dict = field(default_factory=dict)
    allow_slower: bool = False
    noise_floor_pct: float = NOISE_FLOOR_PCT
    notes: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"schema": SCHEMA, "created": _now(), "fallback": self.fallback,
                "fallback_kind": self.fallback_kind, "eager": self.eager,
                "mean_s": self.mean_s, "mean_s_eager": self.mean_s_eager,
                "allow_slower": self.allow_slower,
                "noise_floor_pct": self.noise_floor_pct, "guarantee": dict(self.guarantee),
                "routes": [r.as_json() for r in self.routes], "notes": list(self.notes)}


def _mean_s(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def route(matrix: Matrix, *, eager: str = evalx.KIND_EAGER,
          fallback: str | None = None, allow_slower: bool = False,
          noise_floor_pct: float | None = None,
          exclude: "set[str] | frozenset[str] | tuple[str, ...]" = ()) -> Routes:
    """The routing table: one winner per workload, plus the floor guarantee's verdict.

    ``eager`` is the label of the reference candidate (the floor). ``fallback`` is the
    label a workload with no correct candidate routes to -- `compose` passes the overall
    solution; when it is None the reference is the fallback. ``exclude`` names candidates
    that cannot enter the submission at all (one whose source does not inline): they are
    not routed to, exactly as if their readings were incorrect.
    """
    noise = float(NOISE_FLOOR_PCT if noise_floor_pct is None else noise_floor_pct)
    skip = set(exclude)
    order = {r.label: i for i, r in enumerate(matrix.rows)}
    floor_label = eager if (matrix.row(eager) is not None and eager not in skip) else None
    fallback_label = fallback if (fallback and matrix.row(fallback) is not None
                                  and fallback not in skip) else None
    if fallback_label is None and floor_label is not None:
        fallback_label = floor_label
    notes: list[str] = []
    routes: list[Route] = []
    violations: list[str] = []
    for w in matrix.workloads:
        eager_cell = matrix.row(floor_label).cell(w.name) if floor_label else Cell()
        pool = [(r, r.cell(w.name)) for r in matrix.rows
                if r.label != floor_label and r.label not in skip
                and r.cell(w.name).timed]
        pool.sort(key=lambda rc: (evalx.KIND_ORDER.get(rc[0].kind, 9), order[rc[0].label]))
        winner: Row | None = None
        cell = Cell()
        within: list[str] = []
        reason = ""
        if pool:
            best_t = min(c.t_ms for _, c in pool)
            limit = best_t * (1.0 + noise / 100.0)
            within = [r.label for r, c in pool if c.t_ms <= limit]
            # `pool` is already in the preference order, so the first reading inside the
            # noise floor is the simplest implementation that is fast enough.
            winner, cell = next(rc for rc in pool if rc[1].t_ms <= limit)
            reason = (f"fastest correct reading ({cell.t_ms:.6g} ms"
                      + (f", within the {noise:g}% noise floor: {', '.join(within)}"
                         if len(within) > 1 else "") + ")")
        if winner is not None and eager_cell.timed and \
                float(cell.t_ms) > float(eager_cell.t_ms) * (1.0 + noise / 100.0):
            slowdown = (float(cell.t_ms) / float(eager_cell.t_ms) - 1.0) * 100.0
            if allow_slower:
                reason = (f"{reason}; {slowdown:+.1f}% slower than the eager reference "
                          f"but kept (--allow-slower)")
                slower = True
                violations.append(w.name)
            else:
                reason = (f"{winner.label} is {slowdown:+.1f}% slower than the eager "
                          f"reference ({eager_cell.t_ms:.6g} ms) -- routed to the reference "
                          f"to keep the no-slower-than-eager guarantee (--allow-slower "
                          f"keeps {winner.label})")
                winner, cell = matrix.row(floor_label), eager_cell
                slower = False
                within = []
        elif winner is not None:
            slower = False
        else:
            slower = False
        if winner is None:
            fb = matrix.row(fallback_label) if fallback_label else None
            if fb is not None and fb.cell(w.name).timed:
                winner, cell = fb, fb.cell(w.name)
                reason = (f"no correct candidate on this workload: the fallback "
                          f"{fb.label} ({cell.t_ms:.6g} ms)")
            elif eager_cell.timed:
                winner, cell = matrix.row(floor_label), eager_cell
                reason = ("no correct candidate on this workload and the fallback has no "
                          "reading here: routed to the eager reference")
            elif fb is not None:
                reason = (f"no correct candidate on this workload and the fallback "
                          f"{fb.label} has no reading here: the dispatcher's try/except "
                          f"falls back to the eager reference at run time")
            else:
                reason = ("no correct candidate on this workload: the dispatcher's "
                          "try/except falls back to the eager reference at run time")
        vs = (float(cell.t_ms) / float(eager_cell.t_ms)
              if cell.t_ms is not None and eager_cell.t_ms else None)
        routes.append(Route(
            workload=w.name, index=w.index, winner=winner.label if winner else None,
            winner_kind=winner.kind if winner else "", t_ms=cell.t_ms, s=cell.s,
            t_b=w.t_b, t_sol=w.t_sol, eager_t_ms=eager_cell.t_ms,
            vs_eager=(round(vs, 6) if vs is not None else None),
            within_noise=within if winner is not None else [],
            slower_than_reference=slower, reason=reason))
    if floor_label is None:
        notes.append("no eager reference in the matrix: the no-slower-than-eager "
                     "guarantee could not be checked")
    elif not matrix.row(floor_label).correct:
        notes.append("the eager reference was not correct on every workload: the "
                     "guarantee covers only the workloads where it has a reading")
    checked = [r.workload for r in routes if r.eager_t_ms is not None]
    eager_scores = [matrix.row(floor_label).cell(w.name).s for w in matrix.workloads] \
        if floor_label else []
    return Routes(routes=routes, fallback=fallback_label,
                  fallback_kind=(matrix.row(fallback_label).kind if fallback_label else ""),
                  eager=floor_label or "", mean_s=_mean_s([r.s for r in routes]),
                  mean_s_eager=_mean_s(eager_scores),
                  guarantee={"checked_on": checked,
                             "checked": bool(checked),
                             "violations": violations,
                             "noise_floor_pct": noise,
                             "allow_slower": bool(allow_slower)},
                  allow_slower=bool(allow_slower), noise_floor_pct=noise, notes=notes)


def shape_key(workload: cluster_mod.Workload) -> tuple:
    """The routing key documented for the dispatcher: ``axes`` as a tuple of pairs."""
    return tuple(sorted((str(k), str(v)) for k, v in (workload.axes or {}).items()))


def format_matrix(matrix: Matrix, *, limit_workloads: int = 12) -> str:
    """A text table: candidates down, workloads across, ``!`` for incorrect cells."""
    names = [w.name for w in matrix.workloads][:limit_workloads]
    width = max([len("candidate")] + [len(r.label) for r in matrix.rows] or [9])
    lines = [f"{'candidate':<{width}}  " + "  ".join(f"{n:>12}" for n in names)
             + "   mean S"]
    for r in matrix.rows:
        cells = []
        for n in names:
            c = r.cell(n)
            if c.t_ms is None:
                cells.append(f"{'-' if c.correct is not None else '?':>12}")
            else:
                cells.append(f"{c.t_ms:.4f}{'' if c.correct else '!':>1}".rjust(12))
        mean = "n/a" if r.mean_s is None else f"{r.mean_s:.4f}"
        lines.append(f"{r.label:<{width}}  " + "  ".join(cells) + f"{mean:>9}")
    if len(matrix.workloads) > limit_workloads:
        lines.append(f"({len(matrix.workloads) - limit_workloads} more workload(s) in "
                     f"matrix.json)")
    lines.append("(! = the card judged that cell incorrect; - = no reading)")
    return "\n".join(lines)


def format_routes(routes: Routes) -> str:
    lines = [f"{'workload':<12}{'winner':>22}{'t_ms':>11}{'vs eager':>10}{'S':>8}  why"]
    for r in routes.routes:
        t = "n/a" if r.t_ms is None else f"{r.t_ms:.4f}"
        vs = "n/a" if r.vs_eager is None else f"{r.vs_eager:.3f}x"
        s = "n/a" if r.s is None else f"{r.s:.4f}"
        lines.append(f"{r.workload:<12}{str(r.winner or '-'):>22}{t:>11}{vs:>10}{s:>8}  "
                     f"{r.reason}")
    mean = "n/a" if routes.mean_s is None else f"{routes.mean_s:.4f}"
    eager = "n/a" if routes.mean_s_eager is None else f"{routes.mean_s_eager:.4f}"
    lines.append(f"\nmean S: {mean} (eager reference over the same workloads: {eager})")
    lines.append(f"fallback: {routes.fallback or '-'}; guarantee checked on "
                 f"{len(routes.guarantee.get('checked_on') or [])} workload(s), "
                 f"{len(routes.guarantee.get('violations') or [])} over the floor"
                 + (" (allowed by --allow-slower)"
                    if routes.guarantee.get("allow_slower") else ""))
    for note in routes.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
