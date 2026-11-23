"""Deterministic synthetic GPU for CPU validation and the A/B experiment.

Every architecture has a hidden latent quality for a task (how close to the SOL floor
it can get) and an efficiency (how fast hours close the gap).  Both are derived from a
hash of ``(seed, task, architecture)``, so results are reproducible across machines and
no GPU is required.

This is *not* a performance model of the B300; it is a falsifiable testbed for the
orchestration logic: it lets us check on a laptop that the allocator and portfolio
actually beat the naive strategy under a controlled ground truth.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .base import EvalResult


def _u01(*parts: object) -> float:
    """Stable float in [0, 1) from the parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


@dataclass
class _Latent:
    floor_frac: float    # fraction of the original gap that can never be closed
    rate: float          # fractional gap closed per hour
    valid: bool          # whether this architecture is even correct for the task


class MockBackend:
    """Implements ``CampaignBackend``.

    ``anchors`` maps task -> latency at hour 0 (the current reference).  ``bounds`` maps
    task -> (T_b, T_SOL) so the synthetic latency respects the SOL floor.
    """

    def __init__(
        self,
        anchors: Mapping[str, float],
        bounds: Mapping[str, Tuple[float, float]],
        seed: int = 0,
        noise: float = 0.01,
        p_invalid: float = 0.03,
    ):
        self.anchors = dict(anchors)
        self.bounds = dict(bounds)
        self.seed = seed
        self.noise = noise
        self.p_invalid = p_invalid
        self.calls = 0
        self.log: list = []

    def _latent(self, task: str, arch: str) -> _Latent:
        # architecture quality is shared across tasks (a good design is good generally)
        # but jittered per task so no architecture dominates everywhere.
        q = 0.15 + 0.7 * _u01(self.seed, "arch", arch)
        floor = max(0.02, min(0.6, (1.0 - q) * (0.35 + 0.4 * _u01(self.seed, task, "fl"))) + 0.02)
        rate = 0.02 + 0.10 * _u01(self.seed, task, arch, "rate") * (0.5 + q)
        valid = _u01(self.seed, task, arch, "valid") >= self.p_invalid
        return _Latent(floor_frac=floor, rate=rate, valid=valid)

    def evaluate(
        self,
        task: str,
        reference_latency_ms: float,
        architecture: str,
        hours: float,
        variant: str = "",
    ) -> EvalResult:
        self.calls += 1
        anchor = self.anchors.get(task, reference_latency_ms)
        t_b, t_sol = self.bounds.get(task, (anchor * 2.0, anchor * 0.1))
        lat = self._latent(task, architecture)
        gap0 = max(1e-12, anchor - t_sol)
        closed = math.exp(-lat.rate * max(0.0, hours))
        frac = max(lat.floor_frac, closed)
        t = t_sol + gap0 * frac
        # deterministic noise from the call index
        rng = random.Random(hash((self.seed, task, architecture, round(hours, 6), variant)) & 0xFFFFFFFF)
        t *= 1.0 + rng.uniform(-self.noise, self.noise)
        t = max(t, t_sol)
        counters = {
            "dram_pct": 20.0 + 60.0 * _u01(self.seed, task, "dram"),
            "tensor_pct": 10.0 + 70.0 * _u01(self.seed, task, architecture, "tensor"),
            "occupancy": 0.2 + 0.7 * _u01(self.seed, arch_key(architecture), "occ"),
        }
        self.log.append((task, architecture, round(hours, 3), round(t, 6)))
        if not lat.valid:
            return EvalResult(latency_ms=float("inf"), correct=False,
                              counters=counters,
                              info={"architecture": architecture, "reason": "invalid"})
        return EvalResult(latency_ms=t, correct=True, counters=counters,
                          info={"architecture": architecture,
                                "latent_floor": round(lat.floor_frac, 4)})

    def export(self, task: str) -> Optional[str]:
        return None


def arch_key(arch: str) -> str:
    return arch.split("/")[-1]
