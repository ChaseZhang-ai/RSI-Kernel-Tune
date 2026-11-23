"""Cross-device transfer: turn B300 measurements into predicted B200 scores.

The SOL-ExecBench board scores on **B200 at a locked clock**. Development happens on
**B300** (no B200 available). Absolute latencies do not transfer between the two, but a
small **calibration** fitted on paired measurements does — and the board already hands
us most of what a calibration needs.

Two ways to obtain paired samples, both using public data:

1. **Baseline anchoring (recommended, no B200 needed).** The board publishes the
   per-workload latency of its *Scoring Baseline* (``T_b``).  Run that same baseline
   solution on the B300 and pair each ``(t_b300, T_b200)``.  A handful of workloads
   gives a strong affine fit.
2. **Champion anchoring.** Once any kernel has been measured on both devices, add its
   ``(t_dev, t_ref)`` pair.

From the fitted map ``t_ref ≈ a·t_dev + b`` (or the through-origin ratio ``c`` when the
two devices scale proportionally), this module answers the two questions that matter:

* ``predicted_score(t_dev, workload)`` — what SOL score would this B300 latency earn on
  the board, with an interval.
* ``device_target(target_score, workload)`` — the **B300 latency to develop against** so
  that hitting it is expected to hit the board's score target.

Without paired samples the map is an *assumption* (default ratio 1.0 with wide
uncertainty); the API says so explicitly rather than silently pretending to know.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from . import objective
from .objective import Workload


# --------------------------------------------------------------------------- #
# Calibrations
# --------------------------------------------------------------------------- #

@dataclass
class AffineCalibration:
    """``t_ref = a * t_dev + b`` with predictive uncertainty."""

    a: float
    b: float
    n: int = 0
    residual_std: float = 0.0
    r2: float = 1.0
    assumed: bool = False

    def predict(self, t_dev: float) -> float:
        return self.a * t_dev + self.b

    def predict_interval(self, t_dev: float, k: float = 1.0) -> Tuple[float, float]:
        m = self.predict(t_dev)
        s = k * self.residual_std
        return m - s, m + s

    def device_target(self, t_ref: float) -> float:
        """Invert the map: the B300 latency that implies reference latency ``t_ref``."""
        if self.a <= 0:
            raise ValueError("non-invertible calibration (a <= 0)")
        return max(0.0, (t_ref - self.b) / self.a)

    @classmethod
    def fit(
        cls,
        samples: Sequence[Tuple[float, float]],
        weights: Optional[Sequence[float]] = None,
    ) -> "AffineCalibration":
        """Least-squares fit over ``(t_dev, t_ref)`` pairs."""
        pts = [(float(x), float(y)) for x, y in samples]
        if len(pts) < 2:
            raise ValueError("need at least 2 paired samples to fit an affine map")
        w = list(weights) if weights else [1.0] * len(pts)
        sw = sum(w)
        mx = sum(wi * x for wi, (x, _) in zip(w, pts)) / sw
        my = sum(wi * y for wi, (_, y) in zip(w, pts)) / sw
        sxx = sum(wi * (x - mx) ** 2 for wi, (x, _) in zip(w, pts))
        sxy = sum(wi * (x - mx) * (y - my) for wi, (x, y) in zip(w, pts))
        if sxx <= 0:
            raise ValueError("degenerate samples (all x equal)")
        a = sxy / sxx
        b = my - a * mx
        preds = [a * x + b for x, _ in pts]
        resid = [y - p for (_, y), p in zip(pts, preds)]
        sse = sum(wi * r * r for wi, r in zip(w, resid))
        sst = sum(wi * (y - my) ** 2 for wi, (_, y) in zip(w, pts))
        dof = max(1, len(pts) - 2)
        sigma = math.sqrt(sse / dof)
        r2 = 1.0 - sse / sst if sst > 0 else 1.0
        return cls(a=a, b=b, n=len(pts), residual_std=sigma, r2=r2)

    @classmethod
    def assumed(cls, ratio: float = 1.0, rel_uncertainty: float = 0.5) -> "AffineCalibration":
        """No paired data: assume a proportional map with a wide, explicit interval.

        Marked ``assumed=True`` so callers can refuse to treat a guess as a measurement.
        """
        if ratio <= 0:
            raise ValueError("ratio must be > 0")
        return cls(a=ratio, b=0.0, n=0, residual_std=rel_uncertainty,
                   r2=float("nan"), assumed=True)

    def interval_is_assumption(self) -> bool:
        return self.assumed


@dataclass
class RatioCalibration:
    """Through-origin ``t_ref = c * t_dev`` (devices scale proportionally)."""

    c: float
    n: int = 0
    residual_std: float = 0.0

    def predict(self, t_dev: float) -> float:
        return self.c * t_dev

    def device_target(self, t_ref: float) -> float:
        if self.c <= 0:
            raise ValueError("non-invertible ratio")
        return t_ref / self.c

    @classmethod
    def fit(cls, samples: Sequence[Tuple[float, float]]) -> "RatioCalibration":
        pts = [(float(x), float(y)) for x, y in samples]
        if not pts:
            raise ValueError("need at least 1 paired sample")
        num = sum(x * y for x, y in pts)
        den = sum(x * x for x, _ in pts)
        if den <= 0:
            raise ValueError("degenerate samples")
        c = num / den
        resid = [y - c * x for x, y in pts]
        sigma = math.sqrt(sum(r * r for r in resid) / len(pts))
        return cls(c=c, n=len(pts), residual_std=sigma)


# --------------------------------------------------------------------------- #
# Score transfer
# --------------------------------------------------------------------------- #

def predicted_score(
    t_dev: float,
    workload: Workload,
    calib,
    k: float = 1.0,
) -> Tuple[float, float, float]:
    """``(S_mean, S_low, S_high)`` for a device latency, via the calibration.

    ``S`` decreases with latency, so the low-latency end of the interval gives the
    high score and vice versa.
    """
    t_ref = calib.predict(t_dev)
    s = objective.sol_score(t_ref, workload.t_b_ms, workload.t_sol_ms)
    if getattr(calib, "residual_std", 0.0) > 0:
        lo_t = max(workload.t_sol_ms, t_ref - k * calib.residual_std)
        hi_t = t_ref + k * calib.residual_std
        s_hi = objective.sol_score(lo_t, workload.t_b_ms, workload.t_sol_ms)
        s_lo = objective.sol_score(hi_t, workload.t_b_ms, workload.t_sol_ms)
    else:
        s_lo = s_hi = s
    return s, s_lo, s_hi


def device_target(
    target_score: float,
    workload: Workload,
    calib,
) -> float:
    """The **B300 latency to develop against** so the board score target is expected."""
    t_ref = objective.latency_for_score(target_score, workload.t_b_ms, workload.t_sol_ms)
    return calib.device_target(t_ref)


def device_targets_for_kernel(
    workloads: Sequence[Workload],
    target_kernel_score: float,
    calib,
) -> List[float]:
    return [device_target(target_kernel_score, w, calib) for w in workloads]


def expected_collection(
    scores: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Mean of per-kernel ``(mean, low, high)`` into a collection interval."""
    if not scores:
        raise ValueError("no kernels")
    n = len(scores)
    mean = sum(s[0] for s in scores) / n
    lo = sum(s[1] for s in scores) / n
    hi = sum(s[2] for s in scores) / n
    return mean, lo, hi


def should_submit(
    predicted: Tuple[float, float, float],
    target: float,
    require_confidence: bool = True,
) -> bool:
    """Submit only if the target clears with (``low``) or without (``mean``) confidence."""
    if require_confidence:
        return predicted[1] > target
    return predicted[0] > target


# --------------------------------------------------------------------------- #
# Convenience: fit from published baselines
# --------------------------------------------------------------------------- #

def fit_from_baselines(
    board_ms: Sequence[float],
    device_ms: Sequence[float],
    through_origin: bool = True,
) -> object:
    """Pair published ``T_b`` (board) with the baseline's latency measured on-device.

    This is the routine the runbook recommends: run the board's scoring baseline on the
    B300, pair each workload's ``(device_ms, board_ms)``, and fit.  Through-origin
    (ratio) is the default because the two devices differ by a scale factor more than by
    an offset.
    """
    if len(board_ms) != len(device_ms):
        raise ValueError("board_ms and device_ms must have equal length")
    samples = list(zip(device_ms, board_ms))
    return RatioCalibration.fit(samples) if through_origin else AffineCalibration.fit(samples)
