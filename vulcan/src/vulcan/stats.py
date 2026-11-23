"""Statistical gates: is a measured gain real, or measurement noise?

The upstream harness already refuses to accept a gain below a *noise floor*, and
times the candidate and its reference in the same job (paired timing).  VULCAN
formalises both so the self-evolution chain never promotes a reference that only
looks faster:

* ``bootstrap_ci``          -- distribution-free CI for any statistic.
* ``paired_ratio_ci``       -- CI for ``T_ref / T_cand`` over paired samples.
* ``accept_gain``           -- the promotion decision: significant *and* above the
                               required margin, otherwise the change is reverted and
                               recorded as negative knowledge.
* ``noise_floor_relative``  -- empirical relative noise, used to auto-set the margin.

Only the standard library is used so this runs anywhere (no SciPy on the GPU box).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty sample")
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Basic percentile interval (no resampling); cheap and stable for large n."""
    s = sorted(samples)
    return _quantile(s, alpha / 2.0), _quantile(s, 1.0 - alpha / 2.0)


def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.median,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``samples``.

    Deterministic for a given ``seed`` so runs are reproducible.
    """
    if len(samples) < 2:
        v = float(samples[0]) if samples else float("nan")
        return v, v
    rng = random.Random(seed)
    n = len(samples)
    stats: List[float] = []
    for _ in range(n_resamples):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(draw))
    stats.sort()
    return _quantile(stats, alpha / 2.0), _quantile(stats, 1.0 - alpha / 2.0)


def paired_ratio_ci(
    ref_times: Sequence[float],
    cand_times: Sequence[float],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """CI for the paired ratio ``median(ref_i / cand_i)``.

    Pairing is what makes this powerful: both sides are measured in the same job
    under the same clocks, so the shared drift cancels.
    """
    if len(ref_times) != len(cand_times):
        raise ValueError("paired samples must have equal length")
    ratios = [r / c for r, c in zip(ref_times, cand_times)]
    return bootstrap_ci(ratios, statistics.median, alpha, n_resamples, seed)


def noise_floor_relative(repeats: Sequence[float]) -> float:
    """Relative half-width of the repeat distribution -- the empirical noise floor.

    Defined as ``0.5 * (p95 - p05) / median``.  A gain below this is not evidence.
    """
    if len(repeats) < 3:
        return float("inf")
    med = statistics.median(repeats)
    if med <= 0.0:
        return float("inf")
    lo, hi = percentile_ci(repeats, alpha=0.10)
    return 0.5 * (hi - lo) / med


@dataclass
class GainVerdict:
    """Outcome of a promotion decision."""

    accepted: bool
    ratio: float                 # point estimate of T_ref / T_cand (>1 == faster)
    ci_low: float
    ci_high: float
    margin: float                # required relative gain
    reason: str

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "ratio": round(self.ratio, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "margin": round(self.margin, 6),
            "reason": self.reason,
        }


def accept_gain(
    ref_times: Sequence[float],
    cand_times: Sequence[float],
    min_margin: float = 0.0,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> GainVerdict:
    """Decide whether the candidate beats the reference by more than noise.

    Accepted iff the lower end of the paired-ratio CI exceeds ``1 + min_margin``.
    The whole interval must clear the bar, so a lucky median cannot promote a
    reference that is not really faster.
    """
    ratios = [r / c for r, c in zip(ref_times, cand_times)]
    point = statistics.median(ratios)
    lo, hi = bootstrap_ci(ratios, statistics.median, alpha, n_resamples, seed)
    threshold = 1.0 + min_margin
    if lo > threshold:
        return GainVerdict(True, point, lo, hi, min_margin,
                           f"CI low {lo:.4f} > {threshold:.4f}")
    return GainVerdict(False, point, lo, hi, min_margin,
                       f"CI low {lo:.4f} <= {threshold:.4f} (noise or insufficient gain)")


def auto_margin(repeats: Sequence[float], floor: float = 0.005, ceil: float = 0.05) -> float:
    """Turn an observed noise floor into a promotion margin, clamped to sane limits."""
    nf = noise_floor_relative(repeats)
    if not math.isfinite(nf):
        return ceil
    return max(floor, min(ceil, nf))
