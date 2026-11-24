"""Backend protocol.

Everything side-effecting (LLM calls, GPU evaluation, profiling) is behind this protocol so
ARGUS's reasoning stack -- search, verification, attribution -- can be validated on a CPU
with a deterministic mock, while production runs drive the real CudaForge CLI.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from ..roofline import GpuPeak
from ..search import EvalResult


@runtime_checkable
class CampaignBackend(Protocol):
    # ---- problem identity ----
    def task_name(self) -> str: ...
    def op_class(self) -> str: ...

    # ---- the agent seam ----
    def seed(self, lineage: int) -> str:
        """Produce a seed kernel for a lineage (a distinct architecture bet)."""

    def propose(self, parent_code: str, mechanism: str, lineage: int, context: str) -> str:
        """Produce a child kernel that (attempts to) apply ``mechanism`` to ``parent_code``.

        ``context`` is the diagnosed performance block the agent reads -- the causal,
        model-bounded block in ARGUS mode, or the raw counter dump in CudaForge mode.
        """

    # ---- the hardware seam ----
    def eval_kernel(self, code: str, fidelity: int) -> EvalResult:
        """Verify + time a candidate.  ``fidelity`` selects the timing protocol."""

    def profile(self, code: str) -> dict:
        """Return the counter dict for a compiled candidate."""

    # ---- machine model ----
    def peak(self) -> GpuPeak: ...
    def flops(self) -> float: ...


class BackendError(RuntimeError):
    pass
