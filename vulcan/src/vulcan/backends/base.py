"""Backend protocol: the seam between VULCAN and a real kernel-generation system.

A backend runs *one kernel campaign*: given a task, a sealed reference (the current
champion, or the expert/SOL target) and an architecture hypothesis, spend up to
``hours`` GPU-hours and return a measured latency and correctness verdict.

* ``backends.mock``  -- a deterministic synthetic GPU used for CPU validation and the
  A/B experiment.  No torch, no CUDA.
* ``backends.forge`` -- drives the upstream ``forge`` CLI (OpenBMB/ForgeKernel),
  which is the production path on the B300.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence


@dataclass
class EvalResult:
    """One measured campaign outcome."""

    latency_ms: float
    correct: bool = True
    counters: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "measured"


class CampaignBackend(Protocol):
    """Implement this to run kernels for real."""

    def evaluate(
        self,
        task: str,
        reference_latency_ms: float,
        architecture: str,
        hours: float,
        variant: str = "",
    ) -> EvalResult:
        ...

    def export(self, task: str) -> Optional[str]:
        """Return the submission artifact (path or JSON text) for ``task``, if any."""
        ...
