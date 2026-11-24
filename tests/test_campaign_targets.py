"""`campaign targets`: the gap list, the instrument panel, the feasibility prior.

The first two tests pin the numbers to the real board snapshot (2026-11-24): the
incumbent's per-track means must come out exactly as the board's collection pages print
them, and the best-known means as the "ceiling" row of the background's table. They skip
when the snapshot is not on this machine; the fixture-snapshot tests then cover the same
code paths with two real pages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import campaign_helpers as H
from forgekernel.campaign import solscore
from forgekernel.campaign import targets as T

REAL_SNAPSHOT = Path("/home/zcz/bestKernel/reports/board-snapshot-2026-11-24")
needs_real = pytest.mark.skipif(
    not (REAL_SNAPSHOT / "perkernel").is_dir(),
    reason="the full board snapshot is not on this machine")

LEADER_MEANS = {"l1": 0.818852, "l2": 0.820129, "quant": 0.905855,
                "flashinfer-bench": 0.805878}
BEST_MEANS = {"l1": 0.841240, "l2": 0.857025, "quant": 0.936968,
              "flashinfer-bench": 0.819011}
GAP_TOP10 = [177, 178, 30, 69, 51, 179, 181, 16, 180, 3]
PILOT = ["001_fp8_attention_output_projection", "030_attention_output_projection_with_residual",
         "069_rms_norm"]


@needs_real
def test_the_incumbent_and_its_per_track_means_match_the_board():
    snap = T.load_snapshot(REAL_SNAPSHOT)
    assert len(snap.records) == 235 and not snap.warnings
    rows, who, why = T.rows(snap)
    assert who == "Amir M. Mir | SF Tensor"
    assert round(why["overall"], 6) == 0.830080
    inst = T.instrument(rows, snap, who)
    for track, want in LEADER_MEANS.items():
        assert round(inst["tracks"][track]["leader"], 6) == want, track
    assert round(inst["overall"]["leader"], 6) == 0.830080


@needs_real
def test_best_known_means_match_the_ceiling_row():
    snap = T.load_snapshot(REAL_SNAPSHOT)
    rows, who, _ = T.rows(snap)
    inst = T.instrument(rows, snap, who)
    for track, want in BEST_MEANS.items():
        assert round(inst["tracks"][track]["best_known"], 6) == want, track
    assert round(inst["overall"]["best_known"], 6) == 0.857731


@needs_real
def test_gap_ranking_and_the_documented_pilot():
    snap = T.load_snapshot(REAL_SNAPSHOT)
    rows, _, _ = T.rows(snap)
    ranked = T.ranked(rows)
    assert [r.kernel_id for r in ranked[:10]] == GAP_TOP10
    top4 = [r.name for r in ranked[:4]]
    assert set(PILOT) <= set(top4), top4
    # the board's own names for the holders, verbatim
    assert ranked[0].best_known_by == "yue shui"
    assert ranked[0].best_known == 1.0


@needs_real
def test_incumbent_override_and_unknown_name():
    snap = T.load_snapshot(REAL_SNAPSHOT)
    who, why = T.pick_incumbent(snap, "Qian & Hubert @ CI")
    assert who == "Qian & Hubert @ CI" and why["by"] == "explicit"
    with pytest.raises(ValueError, match="not on this board"):
        T.pick_incumbent(snap, "nobody with this name")


def test_fixture_snapshot_reads_real_pages():
    snap = T.load_snapshot(H.SNAPSHOT)
    assert sorted(snap.records) == [30, H.L1_KERNEL_ID, H.QUANT_KERNEL_ID]
    assert snap.name_of(69) == H.L1_PROBLEM
    assert snap.name_of(177) == H.QUANT_PROBLEM
    assert sorted(snap.collections) == ["l1", "quant"]
    assert snap.cursor[(H.L1_PROBLEM, "L1")] == pytest.approx(0.683127)
    rows, who, why = T.rows(snap)
    assert {r.name for r in rows} == {H.L1_PROBLEM, H.QUANT_PROBLEM,
                                      "030_attention_output_projection_with_residual"}
    # Three pages: the incumbent is the team with the best *weighted overall* -- ERNIE
    # Evolve, because it has a score on all three while the per-problem leaders skip one.
    assert who == "ERNIE Evolve"
    assert why["by"] == "highest weighted overall score"
    row30 = next(r for r in rows if r.kernel_id == 30)
    assert row30.leader == pytest.approx(0.666922) and row30.gap == pytest.approx(0.0)
    row69 = next(r for r in rows if r.kernel_id == 69)
    assert row69.best_known_by == "yue shui"
    assert row69.gap == pytest.approx(0.997818 - 0.847656, abs=1e-6)


def test_ours_moves_the_gap():
    snap = T.load_snapshot(H.SNAPSHOT)
    rows, _, _ = T.rows(snap, ours={H.L1_PROBLEM: 0.5})
    row = next(r for r in rows if r.name == H.L1_PROBLEM)
    assert row.ours == 0.5
    assert row.gap == pytest.approx(row.best_known - 0.5)


def _write_snapshot(d: Path, problems: list[dict], *, cursor: str | None = None) -> Path:
    """A minimal snapshot: perkernel pages + one collection page + optional cursor CSV."""
    (d / "perkernel").mkdir(parents=True)
    for p in problems:
        (d / "perkernel" / f"{p['kernel_id']}.json").write_text(json.dumps({"data": p}))
    (d / "collection_1.json").write_text(json.dumps({"data": {
        "collection_name": "l1", "collection_title": "L1", "rankings": []}}))
    (d / "leaderboards.json").write_text(json.dumps({"data": {"kernel_leaderboards": [
        {"kernel_id": p["kernel_id"], "name": p["kernel_name"], "type": "kernel",
         "display_name": p["kernel_name"]} for p in problems]}}))
    if cursor is not None:
        (d / "cursor_problem_level_metrics.csv").write_text(
            "artifact_id,subset,num_workloads,selected_lang,selected_SOL,selected_speedup\n"
            + cursor)
    return d


def _problem(kid: int, name: str, lead: float, best: float, *, best_by="rival") -> dict:
    return {"kernel_id": kid, "kernel_name": name, "rankings": [
        {"username": "leader", "sol_score": lead, "latency_ms": 1.0},
        {"username": best_by, "sol_score": best, "latency_ms": 0.5}]}


def test_feasibility_prior_reorders_the_pilot(tmp_path):
    snap = _write_snapshot(tmp_path / "snap", [
        _problem(1, "a_big_gap", 0.50, 0.70),        # gap 0.20, no public score
        _problem(2, "b_low_public", 0.50, 0.65),     # gap 0.15, public 0.35 -> x0.7
        _problem(3, "c_high_public", 0.50, 0.60),    # gap 0.10, public 0.85 -> x1.2
    ], cursor=("a_big_gap,L1,16,cuda,,\n"
               "b_low_public,L1,16,cuda,0.350000,1.0\n"
               "c_high_public,L1,16,cuda,0.850000,2.0\n"))
    loaded = T.load_snapshot(snap)
    rows, who, _ = T.rows(loaded, incumbent="leader")
    assert who == "leader"
    by_gap = [r.name for r in T.ranked(rows)]
    by_priority = [r.name for r in T.ranked(rows, by="priority")]
    assert by_gap == ["a_big_gap", "b_low_public", "c_high_public"]
    # x1.2 lifts the 0.10 gap over the 0.15 one; x0.7 pushes the low-public one down
    assert by_priority == ["a_big_gap", "c_high_public", "b_low_public"]
    why = {r.name: r.priority_why for r in rows}
    assert "x1.2" in why["c_high_public"] and "0.850" in why["c_high_public"]
    assert "x0.7" in why["b_low_public"]
    assert "unknown" in why["a_big_gap"]


def test_without_the_prior_file_order_is_pure_gap(tmp_path):
    snap = _write_snapshot(tmp_path / "snap", [
        _problem(1, "a", 0.5, 0.7), _problem(2, "b", 0.5, 0.65)])
    loaded = T.load_snapshot(snap)
    assert any("feasibility prior is off" in w for w in loaded.warnings)
    rows, _, _ = T.rows(loaded)
    assert [r.name for r in T.ranked(rows)] == [r.name for r in T.ranked(rows, by="priority")]
    assert all(r.public_s is None for r in rows)


def test_starter_situation_reads_the_reference_library(tmp_path):
    snap = T.load_snapshot(H.SNAPSHOT)
    root = tmp_path / "repo"
    (root / "kernel_library").mkdir(parents=True)
    index = {"kernels": [
        {"id": "k1", "problem_keys": [H.L1_PROBLEM], "family": "rms_norm",
         "language": "triton", "license": "MIT", "path": "kernels/k1/kernel.py"},
        {"id": "k2", "problem_keys": ["rms_norm"], "family": "rms_norm",
         "language": "cuda-inline", "license": "Apache-2.0",
         "path": "kernels/k2/kernel.py"}]}
    (root / "kernel_library" / "index.json").write_text(json.dumps(index))
    got = T.starter_situation(snap, root)
    assert got[H.L1_PROBLEM][0] == T.STARTER_EXACT
    assert got[H.QUANT_PROBLEM][0] == T.STARTER_BASELINE
    # the family-only entry is ranked under a *different* problem, not this one
    assert T.family_of(H.L1_PROBLEM) == "rmsnorm"    # forgekernel.family's key after the merge
    # with no index at all every problem is "unknown"
    assert T.starter_situation(snap, tmp_path / "empty")[H.L1_PROBLEM][0] == T.STARTER_UNKNOWN


def test_csv_and_json_output():
    snap = T.load_snapshot(H.SNAPSHOT)
    rows, who, why = T.rows(snap, ours={H.L1_PROBLEM: 0.25})
    ranks = T.ranked(rows)
    csv_text = T.to_csv(ranks)
    assert csv_text.splitlines()[0].startswith("kernel_id,name,track,leader")
    assert H.L1_PROBLEM in csv_text and "0.250000" in csv_text
    doc = T.to_json(ranks, snap, who, why, T.instrument(rows, snap, who))
    assert doc["incumbent"] == who and len(doc["problems"]) == 3
    assert doc["problems"][0]["subset"] in ("L1", "Quant")
    assert T.format_table(ranks) and T.format_instrument(doc["instrument"])


def test_loading_a_broken_snapshot_says_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.load_snapshot(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no per-kernel pages"):
        T.load_snapshot(empty)


def test_track_ranges_cover_every_problem_once():
    seen = [k for t in solscore.TRACKS for k in range(*_range(t))]
    assert sorted(seen) == list(range(1, 236))
    assert sum(solscore.track_size(t) for t in solscore.TRACKS) == 235


def _range(track: str):
    lo, hi = solscore.TRACK_RANGES[track]
    return lo, hi + 1
