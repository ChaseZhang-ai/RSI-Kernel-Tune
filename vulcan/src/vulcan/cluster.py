"""Shape clustering: one kernel per *regime*, not per workload.

A definition ships many workloads (a GEMM definition has 25-43 shapes; the whole
FlashInfer-Bench collection has 660).  A per-workload kernel is wasteful and the
architectures across a cluster are often identical.  Grouping the shapes and tuning
one kernel per group amortises the search, which is exactly the "shape clustering"
item on the upstream roadmap -- here it is executable and testable.

1-D k-means is solved *exactly* by dynamic programming (Jenks), so clusters are
deterministic and need no initialisation; that matters on a GPU box where runs must be
reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class Cluster:
    indices: List[int]
    center: float
    weight: float = 1.0

    @property
    def size(self) -> int:
        return len(self.indices)


def _ssq_cost(sorted_vals: Sequence[float]) -> float:
    """Within-cluster sum of squared deviations for a contiguous slice."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mean = sum(sorted_vals) / n
    return sum((v - mean) ** 2 for v in sorted_vals)


def kmeans_1d_exact(values: Sequence[float], k: int) -> Tuple[List[int], List[float]]:
    """Exact 1-D k-means (Jenks) by DP.  Returns ``(assignment, centers)``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    pairs = sorted(enumerate(values), key=lambda t: t[1])
    idx_sorted = [i for i, _ in pairs]
    xs = [v for _, v in pairs]
    n = len(xs)
    if k >= n:
        assignment = [0] * n
        for rank in range(n):
            assignment[rank] = rank
        centers = list(xs)
        out = [0] * n
        for rank, orig in enumerate(idx_sorted):
            out[orig] = assignment[rank]
        return out, centers

    # prefix sums for O(1) slice cost
    ps = [0.0] * (n + 1)
    ps2 = [0.0] * (n + 1)
    for i, v in enumerate(xs):
        ps[i + 1] = ps[i] + v
        ps2[i + 1] = ps2[i] + v * v

    def cost(i: int, j: int) -> float:
        # slice xs[i:j]
        m = j - i
        if m <= 0:
            return 0.0
        s = ps[j] - ps[i]
        s2 = ps2[j] - ps2[i]
        return s2 - s * s / m

    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    split = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for c in range(1, k + 1):
        for j in range(1, n + 1):
            best = INF
            best_i = 0
            for i in range(c - 1, j):
                if dp[c - 1][i] == INF:
                    continue
                val = dp[c - 1][i] + cost(i, j)
                if val < best:
                    best = val
                    best_i = i
            dp[c][j] = best
            split[c][j] = best_i

    # backtrack
    bounds: List[Tuple[int, int]] = []
    j = n
    for c in range(k, 0, -1):
        i = split[c][j]
        bounds.append((i, j))
        j = i
    bounds.reverse()

    out = [0] * n
    centers: List[float] = []
    for c, (i, j) in enumerate(bounds):
        if j <= i:
            continue
        centers.append(sum(xs[i:j]) / (j - i))
        for rank in range(i, j):
            out[idx_sorted[rank]] = c
    return out, centers


def choose_k_exact(
    values: Sequence[float],
    max_k: int = 8,
    variance_explained: float = 0.90,
) -> int:
    """Smallest ``k`` whose clustering explains at least ``variance_explained``.

    Total variance is measured about the global mean; the unexplained part is the
    within-cluster sum of squares.
    """
    xs = list(values)
    n = len(xs)
    if n <= 1:
        return 1
    max_k = min(max_k, n)
    total = _ssq_cost(xs)
    if total <= 0.0:
        return 1
    for k in range(1, max_k + 1):
        assign, centers = kmeans_1d_exact(xs, k)
        within = 0.0
        for v, a in zip(xs, assign):
            within += (v - centers[a]) ** 2
        if 1.0 - within / total >= variance_explained:
            return k
    return max_k


def cluster_values(
    values: Sequence[float],
    k: Optional[int] = None,
    max_k: int = 8,
    variance_explained: float = 0.90,
    weights: Optional[Sequence[float]] = None,
) -> List[Cluster]:
    """Cluster 1-D values; ``weights`` are summed into each cluster's weight."""
    vals = [float(v) for v in values]
    if k is None:
        k = choose_k_exact(vals, max_k, variance_explained)
    assign, centers = kmeans_1d_exact(vals, k)
    buckets: Dict[int, List[int]] = {}
    for i, a in enumerate(assign):
        buckets.setdefault(a, []).append(i)
    clusters: List[Cluster] = []
    for a in sorted(buckets):
        idx = buckets[a]
        w = sum(weights[i] for i in idx) if weights else float(len(idx))
        clusters.append(Cluster(indices=idx, center=centers[a] if a < len(centers) else 0.0,
                                weight=w))
    return clusters


def cluster_workloads(
    items: Sequence,
    key_fn: Callable[[object], float],
    k: Optional[int] = None,
    max_k: int = 8,
    variance_explained: float = 0.90,
    weight_fn: Optional[Callable[[object], float]] = None,
) -> List[List[int]]:
    """Group item indices into shape clusters keyed by ``key_fn`` (e.g. total FLOPs)."""
    values = [float(key_fn(it)) for it in items]
    weights = [float(weight_fn(it)) for it in items] if weight_fn else None
    clusters = cluster_values(values, k, max_k, variance_explained, weights)
    return [c.indices for c in clusters]


def representative_index(items: Sequence, indices: Sequence[int],
                         key_fn: Callable[[object], float]) -> int:
    """The index whose key is closest to the cluster's median key (a real shape)."""
    ks = sorted((key_fn(items[i]), i) for i in indices)
    return ks[len(ks) // 2][1]
