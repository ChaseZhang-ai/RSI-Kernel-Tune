"""The card's workloads and the deterministic clustering that groups them.

The rule (task §5.1) is deliberately mechanical -- no random algorithm, no learned
clustering, and the same input always gives byte-identical output:

1. order the workloads by size: ``T_SOL`` when the board gives every workload one, else
   the element count of the definition's tensors at that workload's axes;
2. cut between neighbours whose size ratio exceeds ``split_ratio`` (default 2.0: two
   workloads closer than a factor of two are one class of shape);
3. fold a cluster smaller than ``min_size`` into the neighbour whose geometric mean is
   closest (in log space);
4. while more than ``max_clusters`` remain, merge the adjacent pair with the smallest
   ratio.

Steps 2-4 only ever look at *sizes*, and ties are broken by content (the element count,
then the workload's own name), never by the position in the list -- so shuffling the
card's workload order gives the same clusters, and two runs of ``plan`` write the same
bytes.

Why ``T_SOL`` and not ``T_b`` or the latency: ``T_SOL`` is the board's analytic floor for
that shape, i.e. the size of the *problem* rather than of one implementation; two
workloads with the same floor are the same kind of kernel problem. A card without the
board's table (``--no-sol-board``) falls back to the element count, and the cluster says
which metric it used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# The harness's one size order: the same element count `meta_workload_names` uses to pick
# a group card's smallest / largest workload. Imported rather than re-derived so a
# cluster boundary and a meta reading agree on what "bigger" means.
from forgekernel.leaderboard import _tensor_elements as tensor_elements

DEFAULT_MAX_CLUSTERS = 4
DEFAULT_SPLIT_RATIO = 2.0
DEFAULT_MIN_SIZE = 2

METRIC_T_SOL = "t_sol"
METRIC_ELEMENTS = "elements"


@dataclass(frozen=True)
class Workload:
    """One workload of the card as the clustering sees it."""
    index: int                  # the dataset index (`forge init --workload` takes these)
    name: str                   # the card's name for it (``wl3``, or the uuid)
    axes: dict = field(default_factory=dict)
    t_b: float | None = None    # the board's baseline latency (ms), when the card has one
    t_sol: float | None = None  # the board's speed-of-light latency (ms)
    elements: int = 0           # input+output elements at this workload's axes

    def as_json(self) -> dict:
        return {"index": self.index, "name": self.name, "axes": dict(self.axes),
                "t_b": self.t_b, "t_sol": self.t_sol, "elements": self.elements}

    @classmethod
    def from_json(cls, data: dict) -> "Workload":
        return cls(index=int(data["index"]), name=str(data.get("name") or ""),
                   axes=dict(data.get("axes") or {}), t_b=_num(data.get("t_b")),
                   t_sol=_num(data.get("t_sol")),
                   elements=int(data.get("elements") or 0))


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def card_workloads(problem: dict) -> list[Workload]:
    """The card's timed workloads, in card order (a group card), or the one workload.

    A group entry carries ``{index, name, workload, sol_board}``; a one-workload card
    carries ``workload`` + ``sol_board`` at the top level. ``index`` is the *dataset*
    index, which is what `forge init --workload` takes; for a one-workload card it is 0.
    """
    definition = problem.get("definition") or {}
    group = problem.get("group") or []
    out: list[Workload] = []
    if group:
        for position, entry in enumerate(group):
            workload = entry.get("workload") or {}
            board = entry.get("sol_board") or {}
            out.append(Workload(
                index=int(entry.get("index", position) if entry.get("index") is not None
                          else position),
                name=str(entry.get("name") or f"wl{position}"),
                axes=dict(workload.get("axes") or {}),
                t_b=_num(board.get("t_b_ms")), t_sol=_num(board.get("t_sol_ms")),
                elements=int(tensor_elements(definition, workload) or 0)))
        return out
    workload = problem.get("workload") or {}
    board = problem.get("sol_board") or {}
    out.append(Workload(index=0, name=str(workload.get("uuid") or "workload"),
                        axes=dict(workload.get("axes") or {}),
                        t_b=_num(board.get("t_b_ms")), t_sol=_num(board.get("t_sol_ms")),
                        elements=int(tensor_elements(definition, workload) or 0)))
    return out


def size_metric(workloads: list[Workload]) -> tuple[str, dict[int, float], list[str]]:
    """``(metric name, {workload index: value}, notes)``.

    ``t_sol`` only when *every* workload carries a positive one; a card with the table on
    some rows and not others is clustered by element count instead, with a note -- mixing
    two metrics in one ratio would compare unrelated numbers."""
    notes: list[str] = []
    tsol = [(w.index, w.t_sol) for w in workloads]
    if workloads and all(t is not None and float(t) > 0 for _, t in tsol):
        return METRIC_T_SOL, {i: float(t) for i, t in tsol}, notes
    if any(t is not None for _, t in tsol):
        notes.append("the card's T_b / T_SOL cover only some workloads: clustering by "
                     "tensor element count instead")
    values: dict[int, float] = {}
    zeros = [w.index for w in workloads if int(w.elements or 0) <= 0]
    for w in workloads:
        values[w.index] = float(max(1, int(w.elements or 0)))
    if zeros:
        notes.append(f"{len(zeros)} workload(s) have no resolvable element count "
                     f"(index {', '.join(str(i) for i in zeros)}): treated as size 1")
    return METRIC_ELEMENTS, values, notes


def _geo(values: list[float]) -> float:
    vals = [float(v) for v in values if v and float(v) > 0]
    if not vals:
        return 1.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@dataclass
class Cluster:
    """One size class: its workload indexes, ascending, and the metric it was built on."""
    name: str                        # ``c0``, ``c1`` ... in size order (plan adds the op)
    workloads: list[int]
    metric: str = METRIC_T_SOL
    values: list[float] = field(default_factory=list)
    reason: str = ""                 # one line: why this class exists (size span, gap)

    @property
    def geo(self) -> float:
        return _geo(self.values)

    @property
    def t_sol_range(self) -> list[float] | None:
        """``[min, max]`` of the cluster's T_SOL values, when it was clustered on them."""
        if self.metric != METRIC_T_SOL or not self.values:
            return None
        return [min(self.values), max(self.values)]

    @property
    def elements_range(self) -> list[int] | None:
        if self.metric != METRIC_ELEMENTS or not self.values:
            return None
        return [int(min(self.values)), int(max(self.values))]

    def as_json(self) -> dict:
        return {"name": self.name, "workloads": list(self.workloads),
                "metric": self.metric, "t_sol_range": self.t_sol_range,
                "elements_range": self.elements_range, "size": len(self.workloads),
                "reason": self.reason}


def _content_key(w: Workload, value: float):
    """The order inside a tie: the metric, then content, never the list position."""
    return (value, int(w.elements or 0), w.name, int(w.index))


def _group(sorted_workloads: list[Workload], values: dict[int, float],
           split_ratio: float) -> list[list[Workload]]:
    clusters: list[list[Workload]] = []
    for w in sorted_workloads:
        if clusters:
            prev = clusters[-1][-1]
            ratio = values[w.index] / values[prev.index] if values[prev.index] else float("inf")
            if ratio > split_ratio:
                clusters.append([w])
                continue
            clusters[-1].append(w)
        else:
            clusters.append([w])
    return clusters


def _merge_small(clusters: list[list[Workload]], values: dict[int, float],
                 min_size: int, notes: list[str]) -> list[list[Workload]]:
    """Fold every cluster below ``min_size`` into its nearest neighbour (log-space)."""
    while len(clusters) > 1:
        small = [(len(c), i) for i, c in enumerate(clusters) if len(c) < min_size]
        if not small:
            break
        _, pos = min(small)                     # the smallest, then the leftmost
        g = _geo([values[w.index] for w in clusters[pos]])
        options = []
        if pos > 0:
            options.append((abs(math.log(g) - math.log(_geo([values[w.index]
                                                             for w in clusters[pos - 1]]))),
                            pos - 1, "previous"))
        if pos + 1 < len(clusters):
            options.append((abs(math.log(g) - math.log(_geo([values[w.index]
                                                             for w in clusters[pos + 1]]))),
                            pos + 1, "next"))
        distance, target, which = min(options)   # a tie goes to the smaller neighbour
        notes.append(f"cluster of {len(clusters[pos])} workload(s) folded into the "
                     f"{which} one (min-size {min_size})")
        clusters[target] = sorted(clusters[target] + clusters[pos],
                                  key=lambda w: _content_key(w, values[w.index]))
        del clusters[pos]
    return clusters


def _merge_to_cap(clusters: list[list[Workload]], values: dict[int, float],
                  max_clusters: int, notes: list[str]) -> list[list[Workload]]:
    while len(clusters) > max(1, int(max_clusters)):
        best = None
        for i in range(len(clusters) - 1):
            left = _geo([values[w.index] for w in clusters[i]])
            right = _geo([values[w.index] for w in clusters[i + 1]])
            ratio = right / left if left else float("inf")
            if best is None or ratio < best[0]:
                best = (ratio, i)
        assert best is not None
        ratio, i = best
        notes.append(f"merged two adjacent clusters (size ratio {ratio:.3g}) to stay "
                     f"within max-clusters {max_clusters}")
        clusters[i] = sorted(clusters[i] + clusters[i + 1],
                             key=lambda w: _content_key(w, values[w.index]))
        del clusters[i + 1]
    return clusters


def cluster(workloads: list[Workload], *, max_clusters: int = DEFAULT_MAX_CLUSTERS,
            split_ratio: float = DEFAULT_SPLIT_RATIO,
            min_size: int = DEFAULT_MIN_SIZE,
            notes: list[str] | None = None) -> list[Cluster]:
    """The card's workloads grouped into size classes (see the module docstring).

    Deterministic and order-independent: the same set of ``Workload`` objects -- in any
    list order -- gives the same clusters, byte for byte after ``json.dumps``. ``notes``,
    when given, collects the global explanations (which metric was used, why clusters
    were folded or merged)."""
    sink: list[str] = notes if notes is not None else []
    if not workloads:
        return []
    if split_ratio <= 1:
        raise ValueError(f"split-ratio must be > 1, got {split_ratio!r}")
    metric, values, metric_notes = size_metric(workloads)
    ordered = sorted(workloads, key=lambda w: _content_key(w, values[w.index]))
    groups = _group(ordered, values, float(split_ratio))
    if len(groups) > 1:
        sink.append(f"split at a size ratio above {split_ratio:g} "
                    f"({len(groups)} clusters before merging)")
    sink += metric_notes
    groups = _merge_small(groups, values, int(min_size), sink)
    groups = _merge_to_cap(groups, values, max_clusters, sink)
    groups = sorted(groups, key=lambda c: _content_key(c[0], values[c[0].index]))
    geos = [_geo([values[w.index] for w in g]) for g in groups]
    out: list[Cluster] = []
    for position, members in enumerate(groups):
        members = sorted(members, key=lambda w: w.index)
        vals = [values[w.index] for w in members]
        if metric == METRIC_T_SOL:
            span = (f"T_SOL {min(vals):.4g}..{max(vals):.4g} ms"
                    if len(vals) > 1 else f"T_SOL {vals[0]:.4g} ms")
        else:
            span = (f"{int(min(vals))}..{int(max(vals))} elements"
                    if len(vals) > 1 else f"{int(vals[0])} elements")
        gap = ""
        if position + 1 < len(groups) and geos[position]:
            gap = f"; the next cluster starts {geos[position + 1] / geos[position]:.3g}x larger"
        out.append(Cluster(
            name=f"c{position}", workloads=[w.index for w in members], metric=metric,
            values=vals,
            reason=f"{len(members)} workload(s), {span}{gap}"
                   + ("" if len(groups) > 1 else " (one size class: nothing to split)")))
    return out


def by_index(workloads: list[Workload]) -> dict[int, Workload]:
    return {w.index: w for w in workloads}


def cluster_of(clusters: list[Cluster]) -> dict[int, str]:
    """``{workload index: cluster name}``."""
    return {i: c.name for c in clusters for i in c.workloads}


def check_partition(clusters: list[Cluster], workloads: list[Workload]) -> list[str]:
    """Problems with a partition: overlaps, missed or unknown indexes."""
    problems: list[str] = []
    seen: dict[int, str] = {}
    for c in clusters:
        for i in c.workloads:
            if i in seen:
                problems.append(f"workload {i} is in both {seen[i]} and {c.name}")
            seen[i] = c.name
    known = {w.index for w in workloads}
    missing = sorted(known - set(seen))
    unknown = sorted(set(seen) - known)
    if missing:
        problems.append(f"workload(s) {missing} belong to no cluster")
    if unknown:
        problems.append(f"cluster workload(s) {unknown} are not on the card")
    return problems


def format_clusters(clusters: list[Cluster], workloads: list[Workload],
                    op: str = "", notes: list[str] | None = None) -> str:
    """A human table of the clusters (for `forge shapes plan`)."""
    size = by_index(workloads)
    names = [f"{op}-{c.name}" if op else c.name for c in clusters]
    width = max([len("cluster")] + [len(n) for n in names])
    lines = [f"{'cluster':<{width}}{'workloads':<16}{'sizes':<30}metric"]
    for c, name in zip(clusters, names):
        wl = ", ".join(str(i) for i in c.workloads)
        if c.metric == METRIC_T_SOL:
            span = (f"T_SOL {c.values[0]:.4g}..{c.values[-1]:.4g} ms"
                    if len(c.values) > 1 else f"T_SOL {c.values[0]:.4g} ms")
        else:
            span = (f"{int(min(c.values))}..{int(max(c.values))} elements"
                    if len(c.values) > 1 else f"{int(c.values[0])} elements")
        lines.append(f"{name:<{width}}{wl:<16}{span:<30}{c.metric}")
        for w in c.workloads:
            axes = json.dumps((size[w].axes if w in size else {}), ensure_ascii=False)
            lines.append(f"    wl{w:<4} {size[w].name:<10} axes={axes}")
        if c.reason:
            lines.append(f"    {c.reason}")
    for note in dict.fromkeys(notes or []):
        lines.append(f"note: {note}")
    return "\n".join(lines)


def target_ms(clusters: list[Cluster], workloads: list[Workload], *,
              sol_goal: float = 0.90) -> dict[str, float]:
    """Each cluster's ``[target] ms``: the geometric mean of its workloads' latencies at
    ``sol_goal`` (``time_for_score`` inverted), the same aggregate the card's gate uses.

    The cluster's gate is the geometric mean of its workloads' readings (a group card's
    ``workloads_performance`` gate), so a target derived from the geometric mean of the
    per-workload target latencies is the latency at which the cluster *averages*
    ``sol_goal`` -- not its best workload's. A workload without a positive ``T_b > T_SOL``
    (no board table) contributes nothing; a cluster with no contribution gets no target.
    """
    from forgekernel.campaign import solscore

    size = by_index(workloads)
    out: dict[str, float] = {}
    for c in clusters:
        targets = []
        for i in c.workloads:
            w = size.get(i)
            if w is None or w.t_b is None or w.t_sol is None:
                continue
            t = solscore.score_time(float(sol_goal), float(w.t_b), float(w.t_sol))
            if t is not None and float(t) > 0:
                targets.append(float(t))
        g = solscore.geomean(targets)
        if g:
            out[c.name] = float(g)
    return out
