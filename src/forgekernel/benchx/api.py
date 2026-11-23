"""benchx -- bench one candidate kernel on a leaderboard card, the card's own way.

One entry point for every consumer that needs a reading of a candidate (the kernel
library, the campaign, evolve, shapes): a candidate goes in, a :class:`BenchResult`
comes out, and the numbers are the ones the loop's own eval job produces -- the card's
stopwatch, the card's judge, the card's SOL block. No second scoring rule lives here.

The request/response shapes are the contract shared across tasks (``Bench``,
``WorkloadReading``, ``BenchResult``); ``FakeBench`` is the test stand-in, and the real
implementation is :class:`forgekernel.benchx.runner.CardBench` (re-exported from this
module, imported lazily so this file stays a leaf).

``confidence`` is about how far a *development-machine* reading can be trusted as a
reading of the evaluation machine (B200): ``"dev-only"`` is always there; a target
profile module (the ``targethw`` task, ``forgekernel.targethw.rules``) may add hints
such as ``"fp4-bound"`` or ``"sfu-bound"``. It never replaces the board's own reading.

``cold_start_s`` is the board-relevant startup cost: from process start to the first
successful ``run()`` return, every compile included (an inline-CUDA ``load_inline``, a
Triton / CuTe DSL JIT, ``torch.compile``). The board re-compiles on every evaluation,
so a candidate that takes longer than the official evaluator's compile budget is a
zero however fast its kernel is; ``COLD_START_BUDGET_S`` is where that budget sits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

COLD_START_BUDGET_S = 100.0
"""Seconds of cold start a candidate may cost before it is marked over budget.

The official open evaluator (SOL-ExecBench ``problem_packager`` / driver) defaults to a
120 s compile timeout and a 600 s evaluation budget, and every evaluation compiles from
scratch. 100 s leaves the compile a margin under that timeout while still being a
reading, not an estimate. ``--mode dev`` records ``{"cold_start": "over-budget"}`` in
``integrity`` when it is exceeded; the campaign's export refuses such a candidate.
"""

COLD_START_MARKER = "[benchx] COLD_START_S"
"""The cold-start probe's output line (``forgekernel.benchx.runner`` writes the probe)."""


class BenchError(ValueError):
    """The bench cannot be set up or read (not a board task, a candidate that does not
    read, a result file that does not parse): a usage error, said in full instead of as
    half a reading. It lives here so that the command's light paths (``--help``,
    ``compare``, ``best-of``) can raise and catch it without importing the runner."""


@dataclass
class WorkloadReading:
    """One workload's reading: the card's own stopwatch number and SOL score."""

    name: str
    axes: dict = field(default_factory=dict)
    t_ms: float | None = None
    t_b: float | None = None
    t_sol: float | None = None
    s: float | None = None
    correct: bool | None = None
    error: str = ""

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
    """One bench of one candidate on one card. All readings are the card's."""

    correct: bool
    workloads: list[WorkloadReading] = field(default_factory=list)
    # The mean of the per-workload SOL scores -- the track score's per-problem term
    # (mean, not geomean, is what the collection score sums). None when the card has
    # no sol_board or no workload was timed.
    mean_s: float | None = None
    # How many workloads read faster than their T_b (the board's "fast_1" numerator;
    # the ratio is fast1 / len(timed workloads) -- see fast1_ratio()).
    fast1: int = 0
    # Machine facts that qualify the reading: cold_start judgement, target compile
    # check, repeat spread, clock state, failure kind. Never a second verdict.
    integrity: dict = field(default_factory=dict)
    # The GPU job id(s), comma separated when --repeat ran several.
    job: str = ""
    elapsed_s: float = 0.0
    error: str = ""
    cold_start_s: float | None = None
    # Reading-confidence hints: always "dev-only"; the target profile module may add
    # "fp4-bound" / "sfu-bound" / "sm-count-sensitive"-style hints.
    confidence: list[str] = field(default_factory=list)

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
    """Bench one candidate against one card.

    ``card_dir`` is a task directory (``tasks/<op>``, the frozen card under
    ``leaderboard/card``) or a card directory itself; ``candidate`` is a Python file
    with a top-level ``run``, or an official solution JSON as a dict. ``mode="dev"``
    runs a GPU job; ``mode="target"`` only runs the target compile check.
    """

    def bench(self, card_dir: Path, candidate: "Path | dict", *,
              workloads=None, mode: str = "dev") -> BenchResult: ...


class FakeBench:
    """A :class:`Bench` with canned readings -- tests only; never touches a GPU."""

    def __init__(self, *, correct: bool = True, t_ms: dict | None = None,
                 s: dict | None = None, mean_s: float | None = None,
                 error: str = "", cold_start_s: float | None = None,
                 integrity: dict | None = None,
                 confidence: list[str] | None = None):
        t_ms = dict(t_ms or {})
        s = dict(s or {})
        self.result = BenchResult(
            correct=correct,
            workloads=[WorkloadReading(name=n, t_ms=v, s=s.get(n)) for n, v in t_ms.items()],
            mean_s=mean_s if mean_s is not None else (sum(s.values()) / len(s) if s else None),
            integrity=dict(integrity or {}),
            error=error, cold_start_s=cold_start_s,
            confidence=list(confidence or ["dev-only"]))
        self.calls: list[dict] = []

    def bench(self, card_dir: Path, candidate: "Path | dict", *,
              workloads=None, mode: str = "dev") -> BenchResult:
        self.calls.append({"card_dir": Path(card_dir), "candidate": candidate,
                           "workloads": workloads, "mode": mode})
        return BenchResult.from_dict(self.result.to_dict())


def __getattr__(name: str):
    """``CardBench`` is the real implementation, imported on first use (no import cycle)."""
    if name == "CardBench":
        from forgekernel.benchx.runner import CardBench

        return CardBench
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
