"""How one lineage's generation is measured, and how two generations compare.

A generation's ``forge run`` already carries the reading the loop gated on: the
run's ``best_confirmation.json`` (the confirmed sha, its same-job ratio and its
per-workload medians) and the eval JSON blocks in the job logs it names.  This
module turns that evidence into one :class:`Metric`; it never times anything
itself.

Kinds, in the order the campaign/benchx work allows:

``sol``
    A board task's mean SOL score.  Preferred sources, tried in order:
    ``rsi_kernel_tune.benchx.reader.read_run`` (task benchx), a one-argument
    ``rsi_kernel_tune.campaign.solscore.score_kernel`` (task campaign), then the
    card's own ``leaderboard.sol_board`` block read from the confirmation
    jobs' eval JSON (``sol_score_mean`` on a group card, ``sol_score`` on a
    one-workload card).  Higher is better.
``speedup``
    ``best_of()["speedup"]``: the confirmed kernel's same-job ratio against the
    reference its task declares.  Higher is better.  This is the ordinary
    task's metric and the fallback when a board card carries no SOL block.
``ms``
    The confirmed kernel's absolute median (``device`` window when the card
    declares one).  Lower is better.  Only used when no ratio exists at all.

The noise floor is the regression baseline of ``scripts/self_evolve.py``
(``max(1.5%, spread of the confirmation ratios)``) for ordinary tasks and
``--s-floor`` for board tasks, exactly as task evolve §5.4 asks.
"""

from __future__ import annotations

import importlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rsi_kernel_tune import launch
from rsi_kernel_tune.task import Task

# == rsi_kernel_tune.loop.parse.detect_plateau's default and scripts/self_evolve.py's
# NOISE_FLOOR_MIN: below this, two readings of one kernel are not told apart.
NOISE_FLOOR_MIN = 0.015
SOL_FLOOR_DEFAULT = 0.005

_KERNEL_FILES = {"cutedsl": "kernel.py", "cuda": "kernel.cu", "triton": "kernel.py"}


def kernel_file_name(task: Task) -> str:
    """The file the student edits (taskcard.KERNEL_FILE with a safe fallback)."""
    from rsi_kernel_tune import taskcard

    try:
        return taskcard.kernel_file(task)
    except KeyError:
        return _KERNEL_FILES.get(task.language, "kernel.py")


def board_kernel_file(task: Task) -> str:
    """A board card freezes its own kernel file name in problem.json."""
    lb = getattr(task, "leaderboard", None)
    if lb is not None:
        try:
            problem = json.loads((lb.card / "eval/leaderboard/problem.json")
                                 .read_text(encoding="utf-8"))
            if problem.get("kernel_file"):
                return str(problem["kernel_file"])
        except (OSError, ValueError):
            pass
    return kernel_file_name(task)


@dataclass(frozen=True)
class Metric:
    """One generation's reading.  ``value`` is the number the population ranks by;
    ``rank`` turns the direction into "higher is better" for a plain sort."""

    kind: str                      # "sol" | "speedup" | "ms"
    value: float
    floor: float
    ms: float | None = None
    source: str = ""               # how the number was obtained (path/API)
    confidence: tuple[str, ...] = ()
    workloads: tuple[dict, ...] = ()
    detail: dict = field(default_factory=dict)

    @property
    def higher_is_better(self) -> bool:
        return self.kind != "ms"

    @property
    def rank(self) -> float:
        return self.value if self.higher_is_better else -self.value

    @property
    def order(self) -> float:
        """The number lineages are ranked by -- higher is better.

        ``sol`` ranks by the score itself.  A ``speedup`` is relative to the
        lineage's OWN reference (its previous best), so it is not comparable
        across lineages; the absolute median is, so when it is known the ranking
        uses ``-ms``.  ``ms`` of course always does.
        """
        if self.kind == "sol":
            return self.value
        if self.ms is not None and self.kind == "speedup":
            return -self.ms
        return self.rank

    @property
    def board_pending(self) -> bool:
        """A development-machine reading: the board must confirm it."""
        return "dev-only" in self.confidence

    def as_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "floor": self.floor,
                "ms": self.ms, "source": self.source,
                "confidence": list(self.confidence),
                "workloads": [dict(w) for w in self.workloads],
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Metric | None":
        if not isinstance(d, dict) or d.get("value") is None:
            return None
        try:
            return cls(kind=str(d.get("kind") or "speedup"), value=float(d["value"]),
                       floor=float(d.get("floor", NOISE_FLOOR_MIN)),
                       ms=float(d["ms"]) if d.get("ms") is not None else None,
                       source=str(d.get("source") or ""),
                       confidence=tuple(d.get("confidence") or ()),
                       workloads=tuple(d.get("workloads") or ()),
                       detail=dict(d.get("detail") or {}))
        except (TypeError, ValueError):
            return None


def noise_floor(best: dict) -> float:
    """``scripts/self_evolve.py:noise_floor`` verbatim: the relative gain a generation
    must clear -- the spread of its same-job ratios, never below NOISE_FLOOR_MIN."""
    ratios = best.get("ratios") or []
    spread = (max(ratios) - min(ratios)) / min(ratios) if len(ratios) > 1 else 0.0
    return max(NOISE_FLOOR_MIN, spread)


def self_gain(cur: Metric | None, prev: Metric | None, *,
              reference_kind: str = "self", legacy: bool = False) -> float | None:
    """This generation's own gain over the lineage's previous best.

    * legacy (``--lineages 1``): ``speedup - 1`` -- every next-generation task's
      reference IS the previous best, which is what ``scripts/self_evolve.py``
      compares against 1 + noise floor.
    * ``speedup`` with a self reference: the same ``speedup - 1``.
    * a foreign/adopted reference or a kind mismatch: compare absolute medians
      (``prev.ms / cur.ms - 1``), the only quantity that is comparable across
      two different references.
    * ``sol``: score points (``cur.value - prev.value``).
    * ``ms``: ``prev.ms / cur.ms - 1``.
    None when the lineage has no previous reading (generation 1) or the
    quantities do not exist.
    """
    if cur is None:
        return None
    if legacy and cur.kind == "speedup":
        return cur.value - 1.0
    if cur.kind == "sol":
        return None if prev is None else cur.value - prev.value
    if cur.kind == "speedup" and reference_kind == "self":
        return cur.value - 1.0
    if prev is None:
        return cur.value - 1.0 if cur.kind == "speedup" else None
    if cur.ms and prev.ms:
        return prev.ms / cur.ms - 1.0
    if cur.kind == "speedup" and prev.kind == "speedup":
        return cur.value / prev.value - 1.0 if prev.value else None
    return None


def improvement(best: Metric, other: Metric) -> float:
    """How much better ``best`` is than ``other``, in the metric's own floor units
    (a relative gain for speedup/ms, score points for sol).

    For speedup readings the comparison goes through the absolute medians when both
    are known: two lineages' speedups are against two different references, and
    their ratio would compare the references, not the kernels."""
    if best.kind == "sol" and other.kind == "sol":
        return best.value - other.value
    if best.ms and other.ms:
        return other.ms / best.ms - 1.0
    if best.kind == "speedup" and other.kind == "speedup" and other.value:
        return best.value / other.value - 1.0
    return best.rank - other.rank


def stall_threshold(metric: Metric, floor: float | None = None) -> float:
    return float(metric.floor if floor is None else floor)


def is_stalled(gain: float | None, metric: Metric, floor: float | None = None) -> bool:
    """``gain`` at or below the noise floor (or unmeasurable) = stalled."""
    return gain is None or gain <= stall_threshold(metric, floor)


# -- reading one run ------------------------------------------------------------------


def eval_object(text: str) -> dict | None:
    """The last JSON object in a job log carrying 'correctness' or 'performance'.

    The same scan as ``rsi_kernel_tune.loop.parse._last_json_object`` (the loop's own
    reader); kept local so evolve does not import a private name.
    """
    dec = json.JSONDecoder()
    found = None
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, j = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and ("correctness" in obj or "performance" in obj):
                found = obj
            i = j
        else:
            i += 1
    return found


def _op_dir(run, task: Task) -> Path:
    return Path(run.operators_dir) / task.name


def _confirmation(run, task: Task) -> dict:
    try:
        return json.loads((_op_dir(run, task) / "best_confirmation.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _job_log(run, task: Task, job: object) -> str:
    p = _op_dir(run, task) / "jobs" / str(job) / "log.txt"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _sol_from_obj(obj: dict | None) -> tuple[float | None, dict, str]:
    """(mean SOL score, {workload name -> s}, clock warning) from one eval object."""
    perf = (obj or {}).get("performance") or {}
    lb = perf.get("leaderboard") if isinstance(perf, dict) else None
    board = (lb or {}).get("sol_board") if isinstance(lb, dict) else None
    if not isinstance(board, dict):
        return None, {}, ""
    per = board.get("sol_score")
    scores: dict = {}
    mean = None
    if isinstance(per, dict):
        for name, s in per.items():
            try:
                scores[str(name)] = float(s)
            except (TypeError, ValueError):
                continue
        if scores:
            mean = statistics.mean(scores.values())
        elif board.get("sol_score_mean") is not None:
            mean = float(board["sol_score_mean"])
    elif per is not None:
        try:
            mean = float(per)
            name = str((lb or {}).get("workload") or (lb or {}).get("definition") or "workload")
            scores[name] = mean
        except (TypeError, ValueError):
            mean = None
    elif board.get("sol_score_mean") is not None:
        try:
            mean = float(board["sol_score_mean"])
        except (TypeError, ValueError):
            mean = None
    warn = str(board.get("clock_warning") or "")
    return mean, scores, warn


def _workload_rows(obj: dict | None, scores: dict) -> list[dict]:
    perf = (obj or {}).get("performance") or {}
    rows = perf.get("workloads") if isinstance(perf, dict) else None
    out: list[dict] = []
    named: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("workload") or "")
            named.add(name)
            ms = row.get("device_median_ms", row.get("bwd_median_ms"))
            out.append({"name": name, "t_ms": float(ms) if ms is not None else None,
                        "s": scores.get(name)})
    # a group card's sol_score dict names every workload; the timing block may be
    # scoped to fewer, so keep the scores that have no timing row
    for name, score in scores.items():
        if name not in named:
            out.append({"name": name, "t_ms": None, "s": score})
    return out


def _benchx_metric(run, task: Task) -> Metric | None:
    """Soft dependency on task benchx: ``rsi_kernel_tune.benchx.reader.read_run(run_dir,
    task_dir) -> BenchResult``.  Absent (or refusing) = None, never an error."""
    try:
        reader = importlib.import_module("rsi_kernel_tune.benchx.reader")
    except Exception:  # noqa: BLE001 — a missing/partial package is "unavailable"
        return None
    fn = getattr(reader, "read_run", None)
    if not callable(fn):
        return None
    try:
        res = fn(run.dir, task.dir)
    except TypeError:
        try:
            res = fn(run.dir)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None
    mean = getattr(res, "mean_s", None)
    if mean is None:
        return None
    conf = tuple(getattr(res, "confidence", ()) or ())
    rows = tuple(dict(w) for w in (getattr(res, "workloads", ()) or ())
                 if isinstance(w, dict))
    ms = None
    for w in rows:
        if w.get("t_ms") is not None:
            ms = max(ms or 0.0, float(w["t_ms"]))
    return Metric(kind="sol", value=float(mean), floor=SOL_FLOOR_DEFAULT,
                  source="benchx.reader.read_run", confidence=conf or ("dev-only",),
                  workloads=rows, ms=ms)


def _campaign_sol(obj: dict) -> float | None:
    """Soft dependency on task campaign: ``solscore.score_kernel(eval_json) -> float``
    (or a dict with mean_s / sol_score_mean).  Called only with one positional
    argument; a signature mismatch is reported by the caller as unavailable."""
    try:
        mod = importlib.import_module("rsi_kernel_tune.campaign.solscore")
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(mod, "score_kernel", None)
    if not callable(fn):
        return None
    try:
        res = fn(obj)
    except Exception:  # noqa: BLE001 — the hook is optional; never break a metric
        return None
    if isinstance(res, dict):
        for key in ("mean_s", "sol_score_mean", "sol_score"):
            if res.get(key) is not None:
                try:
                    return float(res[key])
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return float(res)
    except (TypeError, ValueError):
        return None


def _native_sol(run, task: Task, best: dict | None) -> Metric | None:
    """The card's own SOL block from the confirmation jobs' eval JSON."""
    conf = _confirmation(run, task)
    jobs = [j.get("job") for j in conf.get("jobs") or [] if isinstance(j, dict)]
    per_job: list[float] = []
    rows: list[dict] = []
    warns: list[str] = []
    for job in jobs:
        obj = eval_object(_job_log(run, task, job))
        mean, scores, warn = _sol_from_obj(obj)
        if warn:
            warns.append(warn)
        if mean is None:
            continue
        per_job.append(mean)
        if not rows:
            rows = _workload_rows(obj, scores)
    if not per_job:
        return None
    value = statistics.median(per_job)
    confidence = ["dev-only"]
    if warns:
        confidence.append("sol-clock-warning")
    ms = None
    if best is not None and best.get("median_ms") is not None:
        try:
            ms = float(best["median_ms"])
        except (TypeError, ValueError):
            ms = None
    return Metric(kind="sol", value=value, floor=SOL_FLOOR_DEFAULT,
                  source="card eval JSON (leaderboard.sol_board)",
                  confidence=tuple(confidence), workloads=tuple(rows), ms=ms,
                  detail={"jobs": per_job, "clock_warnings": warns[:1]})


def metric_of(run, task: Task, *, best: dict | None = None,
              s_floor: float = SOL_FLOOR_DEFAULT) -> Metric | None:
    """The generation's metric, or None when the run has no confirmed best.

    ``best`` may be passed when the caller already called ``launch.best_of`` (the
    population does, for the kernel source); otherwise it is fetched here.
    """
    if best is None:
        best = launch.best_of(run, task)
    if best is None or not best.get("kernel"):
        return None
    if task.leaderboard is not None:
        m = _benchx_metric(run, task)
        if m is not None:
            return Metric(kind=m.kind, value=m.value, floor=float(s_floor), ms=m.ms,
                          source=m.source, confidence=m.confidence,
                          workloads=m.workloads, detail=m.detail)
        conf = _confirmation(run, task)
        jobs = [j for j in conf.get("jobs") or [] if isinstance(j, dict)]
        for job in jobs:
            obj = eval_object(_job_log(run, task, job.get("job")))
            score = _campaign_sol(obj) if obj is not None else None
            if score is not None:
                return Metric(kind="sol", value=score, floor=float(s_floor),
                              ms=best.get("median_ms"),
                              source="rsi_kernel_tune.campaign.solscore.score_kernel",
                              confidence=("dev-only",))
        m = _native_sol(run, task, best)
        if m is not None:
            return Metric(kind=m.kind, value=m.value, floor=float(s_floor), ms=m.ms,
                          source=m.source, confidence=m.confidence,
                          workloads=m.workloads, detail=m.detail)
    speedup = best.get("speedup")
    if speedup is not None:
        try:
            return Metric(kind="speedup", value=float(speedup),
                          floor=float(s_floor) if task.leaderboard is not None
                          else noise_floor(best),
                          ms=best.get("median_ms"),
                          source="best_of().speedup",
                          confidence=("dev-only",) if task.leaderboard is not None else ())
        except (TypeError, ValueError):
            pass
    if best.get("median_ms") is not None:
        try:
            return Metric(kind="ms", value=float(best["median_ms"]),
                          floor=NOISE_FLOOR_MIN, ms=float(best["median_ms"]),
                          source="best_of().median_ms",
                          confidence=("dev-only",) if task.leaderboard is not None else ())
        except (TypeError, ValueError):
            pass
    return None
