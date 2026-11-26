"""benchx: the one way to evaluate a candidate kernel on a board problem.

::

    bench = ...                                   # a Bench (this module)
    result = bench.bench(card_dir, candidate, workloads=[0, 3], mode="dev")
    result.correct, result.mean_s, result.fast1
    for w in result.workloads:
        w.name, w.t_ms, w.t_b, w.t_sol, w.s, w.correct

``card_dir`` is a frozen leaderboard card (``tasks/<op>/leaderboard/card/``: the board's own
judge, inputs and clock). ``candidate`` is a Python file with a top-level ``run`` (a
``submission.py``) or an official JSON solution. ``result`` carries one reading per workload,
each taken by the card's own clock and each verdict given by the card's own judge: a Bench
never re-times a kernel and never re-judges an output its own way.

**Do not write a second evaluator: wait for benchx.** The reference library's scoring,
campaign's choice of a starting point, evolve's selection and the per-workload dispatch of
shapes all come through ``Bench.bench``. Do not measure a candidate by calling the loop's
backend on your own, and do not compute a SOL score or an average again where the card has
already given one. Until the real ``Bench`` exists, tests stand ``FakeBench`` in for it.

This module holds the request/response shapes every consumer codes against (``Bench``,
``WorkloadReading``, ``BenchResult``, ``FakeBench``); the real implementation
(:class:`rsi_kernel_tune.benchx.runner.CardBench` -- a temporary run of the card with the
candidate as its kernel, read back through the loop's own parser) is re-exported from here
on first use, so this file stays a leaf.

``confidence`` is about how far a *development-machine* reading can be trusted as a
reading of the evaluation machine (B200): ``"dev-only"`` is always there; a target
profile module (the ``targethw`` task, ``rsi_kernel_tune.targethw.rules``) may add hints
such as ``"fp4-bound"`` or ``"sfu-bound"``. It never replaces the board's own reading.

``cold_start_s`` is the board-relevant startup cost: from process start to the first
successful ``run()`` return, every compile included (an inlined CUDA build, a Triton /
CuTe DSL JIT, ``torch.compile``). The board re-compiles on every evaluation, so a candidate
that takes longer than the official evaluator's compile budget is a zero however fast its
kernel is; ``COLD_START_BUDGET_S`` is where that budget sits.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# Every evaluation on the board compiles the submission again from scratch (the CUDA C++ of
# an inlined kernel through torch.utils.cpp_extension, Triton and CuTe DSL just in time,
# torch.compile), and the open-source evaluator's default compile timeout is 120 s. A
# candidate whose first call takes longer than this may be timed out there, whatever its
# steady-state speed: campaign's export refuses it (see cold_start_over_budget).
COLD_START_BUDGET_S = 100.0

COLD_START_MARKER = "[benchx] COLD_START_S"
"""The cold-start probe's output line (``rsi_kernel_tune.benchx.runner`` writes the probe)."""

MODE_DEV = "dev"          # run on the machine the harness delivers to
MODE_TARGET = "target"    # the target compile check only: nothing runs
MODES = (MODE_DEV, MODE_TARGET)

# What a reading's confidence tags say (BenchResult.confidence).
CONF_DEV_ONLY = "dev-only"               # always: measured on the development machine
CONF_FP4_BOUND = "fp4-bound"             # dense FP4 throughput: faster here than on the board
CONF_SFU_BOUND = "sfu-bound"             # special-function throughput: about twice the board's
CONF_SM_COUNT_SENSITIVE = "sm-count-sensitive"   # a grid sized for this machine's SM count


class BenchError(ValueError):
    """The bench cannot be set up or read (not a board task, a candidate that does not
    read, a result file that does not parse): a usage error, said in full instead of as
    half a reading. It lives here so that the command's light paths (``--help``,
    ``compare``, ``best-of``) can raise and catch it without importing the runner."""


def cold_start_over_budget(cold_start_s: float | None) -> bool:
    """Whether a cold-start reading is past ``COLD_START_BUDGET_S``.

    ``None`` (not measured) is not over budget: it is unknown. A caller that needs the
    reading (an export) says so itself instead of reading ``False`` as "fast"."""
    return cold_start_s is not None and cold_start_s > COLD_START_BUDGET_S


@dataclass
class WorkloadReading:
    """One workload of the card as the card read it (a reading is ``None`` where there is none,
    and ``error`` says why). ``t_b`` and ``t_sol`` are the board's own table for the workload,
    not something measured here; ``s`` is the score the card computed from the three."""
    name: str
    axes: dict = field(default_factory=dict)   # the workload's shape: axis name -> size
    t_ms: float | None = None       # the candidate's time
    t_b: float | None = None        # the board's baseline time
    t_sol: float | None = None      # the board's speed-of-light time
    s: float | None = None          # 0.5 at t_b, 1.0 at t_sol; a failed workload scores 0
    correct: bool | None = None     # the card's own verdict
    error: str = ""                 # the evaluator's own words when it failed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkloadReading":
        known = {k: data[k] for k in
                 ("name", "axes", "t_ms", "t_b", "t_sol", "s", "correct", "error")
                 if k in data}
        known.setdefault("name", "")
        return cls(**known)


@dataclass
class BenchResult:
    """What one evaluation of one candidate found.

    ``integrity`` says what else is known about the reading's standing, each under its own
    key: ``{"cold_start": "over-budget"}`` when ``cold_start_s`` is past the budget (and why
    it is ``None`` when it could not be measured), ``{"target_compile": ...}`` for the target
    compile check. An evaluation that did not complete has ``correct = False``, the reason in
    ``error`` and no readings pretending otherwise.
    """
    correct: bool
    workloads: list[WorkloadReading] = field(default_factory=list)
    mean_s: float | None = None      # the card's mean SOL score over the workloads run
    fast1: int = 0                   # workloads faster than the board's baseline
    integrity: dict = field(default_factory=dict)
    job: str = ""                    # the GPU job the reading came from
    elapsed_s: float = 0.0           # wall clock of the whole evaluation
    error: str = ""
    cold_start_s: float | None = None   # process start -> first run() returned, all compiling included
    confidence: list[str] = field(default_factory=list)   # the CONF_* tags that apply

    def by_name(self) -> dict[str, WorkloadReading]:
        """The readings keyed by workload name."""
        return {w.name: w for w in self.workloads}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["workloads"] = [w.to_dict() for w in self.workloads]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "BenchResult":
        d = dict(data)
        d["workloads"] = [WorkloadReading.from_dict(w) for w in d.get("workloads") or ()]
        allowed = {"correct", "workloads", "mean_s", "fast1", "integrity", "job",
                   "elapsed_s", "error", "cold_start_s", "confidence"}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "BenchResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def timed(self) -> list[WorkloadReading]:
        """The workloads that carry a reading (a correctness-failed card carries none)."""
        return [w for w in self.workloads if w.t_ms is not None]

    def fast1_ratio(self) -> float | None:
        """Share of the timed workloads that read faster than their T_b."""
        timed = self.timed()
        return (self.fast1 / len(timed)) if timed else None


@runtime_checkable
class Bench(Protocol):
    def bench(self, card_dir: Path, candidate: Path | dict, *, workloads=None,
              mode: str = MODE_DEV) -> BenchResult:
        """Evaluate ``candidate`` on the problem of the card at ``card_dir``.

        ``workloads``: ``None`` runs every workload of the card, else the ones named (each by
        its name or by its index in the card). ``mode``: ``"dev"`` runs them on the machine
        the harness delivers to; ``"target"`` only checks that the candidate compiles for the
        board's GPU -- nothing runs and the readings stay ``None``.
        """


@dataclass(frozen=True)
class BenchCall:
    """One call made to a ``FakeBench``, as it was made."""
    card_dir: Path
    candidate: Path | dict
    workloads: Sequence[str | int] | str | None = None
    mode: str = MODE_DEV


_UNSET = object()


class FakeBench:
    """A ``Bench`` that runs nothing: it answers each call from a script and records the call.

    ``script`` is one of:

    * a ``BenchResult`` -- the answer to every call (handed back as it is);
    * a sequence of them -- the answer to each call in turn, the last one repeating;
    * a callable ``f(call) -> BenchResult`` -- the answer computed from the ``BenchCall``;
    * nothing -- a passing result with no workload, for a caller that only needs the call.

    Keyword readings instead build that one result: ``FakeBench(correct=True,
    t_ms={"wl": 0.5}, s={"wl": 0.7})`` -- one workload per ``t_ms`` entry, ``mean_s`` from
    ``s`` unless given, and a fresh ``BenchResult`` on every call (the preset is not shared
    state).

    ``calls`` holds every call in order, as :class:`BenchCall`. Tests of everything that
    uses a Bench stand this in for the real one; nothing else in the repository evaluates a
    candidate.
    """

    def __init__(self, script=None, *, correct=_UNSET, t_ms=_UNSET, s=_UNSET,
                 mean_s=_UNSET, error=_UNSET, cold_start_s=_UNSET, integrity=_UNSET,
                 confidence=_UNSET):
        keywords = [v is not _UNSET for v in (correct, t_ms, s, mean_s, error, cold_start_s,
                                             integrity, confidence)]
        if script is not None and any(keywords):
            raise TypeError("FakeBench takes a script or keyword readings, not both")
        self.script = script
        self.calls: list[BenchCall] = []
        self._left: list[BenchResult] = list(script) if isinstance(script, (list, tuple)) else []
        self._preset = None
        if any(keywords):
            t_ms = {} if t_ms is _UNSET else dict(t_ms or {})
            s = {} if s is _UNSET else dict(s or {})
            self._preset = {
                "correct": True if correct is _UNSET else bool(correct),
                "t_ms": t_ms,
                "s": s,
                "mean_s": ((sum(s.values()) / len(s)) if s else None) if mean_s is _UNSET
                          else mean_s,
                "error": "" if error is _UNSET else error,
                "cold_start_s": None if cold_start_s is _UNSET else cold_start_s,
                "integrity": {} if integrity is _UNSET else dict(integrity or {}),
                "confidence": ["dev-only"] if confidence is _UNSET else list(confidence or []),
            }

    def _preset_result(self) -> BenchResult:
        p = self._preset
        return BenchResult(
            correct=p["correct"],
            workloads=[WorkloadReading(name=n, t_ms=v, s=p["s"].get(n))
                       for n, v in p["t_ms"].items()],
            mean_s=p["mean_s"], integrity=dict(p["integrity"]), error=p["error"],
            cold_start_s=p["cold_start_s"], confidence=list(p["confidence"]))

    def bench(self, card_dir: Path, candidate: Path | dict, *, workloads=None,
              mode: str = MODE_DEV) -> BenchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        call = BenchCall(Path(card_dir), candidate, workloads, mode)
        self.calls.append(call)
        if self._preset is not None:
            return self._preset_result()
        script = self.script
        if isinstance(script, BenchResult):
            return script
        if callable(script):
            return script(call)
        if self._left:
            return self._left.pop(0) if len(self._left) > 1 else self._left[0]
        return BenchResult(correct=True, workloads=[], mean_s=None)


def __getattr__(name: str):
    """``CardBench`` is the real implementation, imported on first use (no import cycle)."""
    if name == "CardBench":
        from rsi_kernel_tune.benchx.runner import CardBench

        return CardBench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
