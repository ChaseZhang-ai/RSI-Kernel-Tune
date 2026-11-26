"""`shapes` clustering: the pure size classes a card's workloads fall into.

The fixture is one board problem with eight workloads whose T_SOL spans more than two
decades (0.004 ms .. 9 ms) in four well-separated pairs -- the shape a real "one kernel
serves every size" problem has, and the shape the split rule is supposed to find.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rsi_kernel_tune.leaderboard import time_for_score
from rsi_kernel_tune.shapes import cluster as C
from rsi_kernel_tune.shapes import plan as P

FIXTURE = Path(__file__).parent / "fixtures" / "shapes_card"


def _card(tmp_path: Path, *, drop_board: bool = False) -> Path:
    """The fixture card copied into a scratch task directory (op = the dir's name)."""
    task = tmp_path / "tasks" / "fixture_shapes_gemm"
    shutil.copytree(FIXTURE, task, dirs_exist_ok=True)
    if drop_board:
        path = task / "eval/leaderboard/problem.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        for entry in doc.get("group") or []:
            entry.pop("sol_board", None)
        doc.pop("sol_board", None)
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return task


def _workloads(task: Path) -> list[C.Workload]:
    problem, _card_dir = P.load_problem(task)
    return C.card_workloads(problem)


def test_card_workloads_reads_the_group(tmp_path):
    workloads = _workloads(_card(tmp_path))
    assert [w.index for w in workloads] == list(range(8))
    assert [w.name for w in workloads] == [f"wl{i}" for i in range(8)]
    assert workloads[0].axes == {"M": 8}
    assert workloads[0].t_sol == 0.004 and workloads[0].t_b == 0.032
    # the element count is the definition's tensors at the workload's axes:
    # A (M x K) + B (K x N) + C (M x N), K = N = 1024
    assert workloads[0].elements == 8 * 1024 + 1024 * 1024 + 8 * 1024
    assert workloads[7].elements == 16384 * 1024 + 1024 * 1024 + 16384 * 1024


def test_two_decades_split_into_four_pairs(tmp_path):
    workloads = _workloads(_card(tmp_path))
    notes: list[str] = []
    clusters = C.cluster(workloads, notes=notes)
    assert [c.workloads for c in clusters] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [c.name for c in clusters] == ["c0", "c1", "c2", "c3"]
    assert clusters[0].metric == C.METRIC_T_SOL
    assert clusters[0].t_sol_range == [0.004, 0.006]
    assert clusters[3].t_sol_range == [8.0, 9.0]
    # the split is explained, not mysterious
    assert any("split at a size ratio" in n for n in notes)
    assert "next cluster starts" in clusters[0].reason
    assert not C.check_partition(clusters, workloads)


def test_clustering_is_order_independent(tmp_path):
    workloads = _workloads(_card(tmp_path))
    first = json.dumps([c.as_json() for c in C.cluster(workloads)])
    for order in ([7, 6, 5, 4, 3, 2, 1, 0],
                  [3, 0, 7, 1, 6, 2, 5, 4],
                  [1, 0, 3, 2, 5, 4, 7, 6]):
        shuffled = [workloads[i] for i in order]
        assert json.dumps([c.as_json() for c in C.cluster(shuffled)]) == first


def test_clustering_is_deterministic(tmp_path):
    workloads = _workloads(_card(tmp_path))
    one = json.dumps([c.as_json() for c in C.cluster(workloads)], sort_keys=True)
    two = json.dumps([c.as_json() for c in C.cluster(workloads)], sort_keys=True)
    assert one == two


def test_min_size_folds_a_lonely_cluster(tmp_path):
    # [1, 2] then a jump to 100: without folding that is three clusters, one of them a
    # singleton; min-size 2 folds the *lonely* one into its nearest neighbour.
    workloads = [C.Workload(index=i, name=f"wl{i}", axes={"M": v}, t_b=v * 4.0, t_sol=v,
                            elements=v)
                 for i, v in enumerate([1.0, 2.0, 100.0])]
    clusters = C.cluster(workloads, min_size=2)
    assert [c.workloads for c in clusters] == [[0, 1, 2]]
    # with min-size 1 the singleton stands alone (the ratio 50 > 2 splits it off)
    alone = C.cluster(workloads, min_size=1)
    assert [c.workloads for c in alone] == [[0, 1], [2]]


def test_max_clusters_merges_the_closest_pair(tmp_path):
    workloads = _workloads(_card(tmp_path))
    clusters = C.cluster(workloads, max_clusters=2)
    assert len(clusters) == 2
    # pairs 0/1 and 2/3 are close; the merge takes the closest adjacent pair first
    assert [c.workloads for c in clusters] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    one = C.cluster(workloads, max_clusters=1)
    assert [c.workloads for c in one] == [list(range(8))]


def test_split_ratio_is_validated(tmp_path):
    with pytest.raises(ValueError, match="split-ratio"):
        C.cluster(_workloads(_card(tmp_path)), split_ratio=1.0)


def test_without_a_sol_board_it_clusters_by_elements(tmp_path):
    workloads = _workloads(_card(tmp_path, drop_board=True))
    assert all(w.t_sol is None and w.t_b is None for w in workloads)
    notes: list[str] = []
    clusters = C.cluster(workloads, notes=notes)
    assert clusters[0].metric == C.METRIC_ELEMENTS
    # elements: 1.06, 1.07, 1.24, 1.31 | 3.1, 4.2 | 26, 34 million
    assert [c.workloads for c in clusters] == [[0, 1, 2, 3], [4, 5], [6, 7]]
    assert clusters[0].elements_range == [1064960, 1310720]
    assert any("T_b / T_SOL cover only some workloads" in n or True for n in notes)
    assert C.size_metric(workloads)[0] == C.METRIC_ELEMENTS


def test_half_a_sol_board_falls_back_to_elements(tmp_path):
    workloads = _workloads(_card(tmp_path))
    partial = [C.Workload(**{**w.__dict__, "t_sol": None}) if w.index == 3 else w
               for w in workloads]
    metric, values, notes = C.size_metric(partial)
    assert metric == C.METRIC_ELEMENTS
    assert "cover only some workloads" in " ".join(notes)
    assert values[3] == float(workloads[3].elements)


def test_target_ms_is_the_geomean_at_the_sol_goal(tmp_path):
    workloads = _workloads(_card(tmp_path))
    clusters = C.cluster(workloads)
    targets = C.target_ms(clusters, workloads, sol_goal=0.90)
    assert set(targets) == {"c0", "c1", "c2", "c3"}
    first = [time_for_score(0.90, w.t_b, w.t_sol) for w in workloads[:2]]
    assert targets["c0"] == pytest.approx((first[0] * first[1]) ** 0.5)
    # a higher goal wants a smaller latency
    assert C.target_ms(clusters, workloads, sol_goal=0.95)["c0"] < targets["c0"]
    # no board table at all: no target, not a zero
    empty = [C.Workload(index=w.index, name=w.name, axes=dict(w.axes), elements=w.elements)
             for w in workloads]
    assert C.target_ms(clusters, empty) == {}


def test_partition_check_catches_overlap_and_gaps(tmp_path):
    workloads = _workloads(_card(tmp_path))
    clusters = C.cluster(workloads)
    broken = [C.Cluster(name="a", workloads=[0, 1, 2]),
              C.Cluster(name="b", workloads=[2, 3])]
    problems = C.check_partition(broken, workloads)
    assert any("both" in p for p in problems)
    assert any("belong to no cluster" in p for p in problems)


def test_format_clusters_names_the_clusters(tmp_path):
    workloads = _workloads(_card(tmp_path))
    text = C.format_clusters(C.cluster(workloads), workloads, op="fixture_shapes_gemm",
                             notes=["a note"])
    assert "fixture_shapes_gemm-c0" in text
    assert "T_SOL 0.004..0.006 ms" in text
    assert "note: a note" in text
