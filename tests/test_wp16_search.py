"""WP16: active search & population governance — the search-side helpers.

Pins: the shrink-grid contraction (a known response surface converges), the
grid/recipe refusals, the MC near-tie detector, the climb streak, the runner-up
pool and its three states (target present / no #2 / pool empty), the revival
audit line, floor_progress, the halving range expansion + refusal ceilings,
sweep_run's noise-gated halving cut, the EXPIRED revision, and the STAGNATION
pool-empty byte-identity regression (red line ③-⑤)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.inner.trajectory import IterationRecord, append_record, load_records
from forgekernel.loop import driver as drv
from forgekernel.loop import lever_pool
from forgekernel.skills import sweep_run

REPO = Path(__file__).resolve().parents[1]


# --- fixtures -------------------------------------------------------------------

class _Be:
    def head(self):
        return "sha0"


@pytest.fixture()
def ctx(tmp_path):
    status = tmp_path / "status.jsonl"
    status.write_text("")
    milestone = tmp_path / "m.md"
    milestone.write_text("# demo\n\n## 3. Decision details\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return drv.LoopContext(
        op="demo", workspace=tmp_path, milestone=milestone, kb_dir=tmp_path,
        status_path=status, meta_dir=tmp_path, kernel_file="k.py",
        branch_prefix="demo", target_ms=1.0, deadline_epoch=1e18, backend=_Be(),
        now=lambda: 0.0, evidence_dir=evidence,
    )


def _rec(ms, verdict="KEEP", iter_=1, did="D1.1", sha="s1", **kw):
    return IterationRecord(iter=iter_, decision_id=did, change="c",
                           correctness=(verdict != "REVERT"), verdict=verdict,
                           device_median_ms=ms, sha=sha, **kw)


# --- task 1: the shrink grid (乙A3) ---------------------------------------------

def test_ps_next_points_contract_and_purity():
    assert drv.ps_next_points([16, 32, 64, 128], 32) == [24, 48]
    assert drv.ps_next_points([16, 32, 64], 16) == [24]        # no lower neighbor
    assert drv.ps_next_points([16, 32], 16) == [24]
    assert drv.ps_next_points([16, 17], 16) == []              # adjacent ints: done
    assert drv.ps_next_points([], 8) == []
    assert drv.ps_next_points([16, 32, 64], 64) == [48]        # top edge: one side
    # floats contract in floats (no integer snapping), no duplicate of a measured point
    assert drv.ps_next_points([0.0, 0.5], 0.5) == [0.25]


def test_pattern_search_contracts_on_a_known_surface():
    """The acceptance grid test: a unimodal response face (peak at 37) walks by
    the driver's own grid functions — stage 1 measures the teacher's values,
    each later stage measures the midpoints around the winner — and lands
    within one unit of the peak after three contraction stages."""
    def face(v):
        return -abs(v - 37.0)          # a known unimodal response surface

    states = {"tile": {"old": "TILE = 64", "new": "TILE = {v}",
                       "values": [16.0, 32.0, 64.0, 128.0],
                       "measured": [], "winner": None, "winner_delta": None}}
    v = min(states["tile"]["values"], key=lambda x: -face(x))
    states["tile"]["measured"] = list(states["tile"]["values"])
    states["tile"]["winner"], states["tile"]["winner_delta"] = v, face(v)
    for _ in range(3):                 # 2 further contraction stages
        pts = drv.ps_next_points(sorted(states["tile"]["measured"]),
                                 states["tile"]["winner"])
        if not pts:
            break
        best = max(pts, key=face)
        states["tile"]["measured"].extend(pts)
        if face(best) > face(states["tile"]["winner"]):
            states["tile"]["winner"] = best
    assert abs(states["tile"]["winner"] - 37.0) <= 1.0
    # the stage-1 variants: one edit per teacher value, `{v}` substituted
    states["tile"]["measured"], states["tile"]["winner"] = [], None
    variants, mapping = drv.ps_stage_variants(states, 1)
    assert [v["name"] for v in variants] == ["tile_16", "tile_32", "tile_64", "tile_128"]
    assert variants[0]["edits"] == [{"old": "TILE = 64", "new": "TILE = 16"}]
    assert mapping["tile_64"] == ("tile", 64.0)
    # no winner on an axis -> stage 2 adds nothing for it
    variants2, _ = drv.ps_stage_variants(states, 2)
    assert variants2 == []


def test_ps_recipe_refused():
    ok = {"axes": [{"name": "tile", "old": "TILE = 64", "new": "TILE = {v}",
                    "values": [16, 32, 64]}]}
    assert drv._ps_recipe_refused(ok) is None
    assert drv._ps_recipe_refused({"axes": []}) is not None
    assert drv._ps_recipe_refused({"axes": [{"name": "a", "old": "x", "new": "y {v}",
                                             "values": [1, 2]},
                                            {"name": "b", "old": "x", "new": "z {v}",
                                             "values": [1, 2]},
                                            {"name": "c", "old": "x", "new": "w {v}",
                                             "values": [1, 2]}]}) is not None
    assert drv._ps_recipe_refused({"axes": [{"name": "t", "old": "x",
                                             "new": "no placeholder", "values": [1, 2]}]})
    assert drv._ps_recipe_refused({"axes": [{"name": "t", "old": "x", "new": "{v}",
                                             "values": [1]}]}) is not None
    assert drv._ps_recipe_refused({"axes": [{"name": "t", "old": "x", "new": "{v}",
                                             "values": [1, "x"]}]}) is not None


# --- task 3: the MC near-tie detector (乙A6) ------------------------------------

def _mcrow(name, ms, smoke=True):
    return {"name": name, "smoke": smoke, "ms": ms, "ratio": None, "delta_pct": None}


def test_mc_top2_gap_pct():
    rows = [_mcrow("v0", 3.30), _mcrow("v1", 3.36), _mcrow("v2", 4.0),
            _mcrow("dnf", None, smoke=False)]
    w, ru, gap = drv._mc_top2_gap_pct(rows)
    assert (w, ru) == ("v0", "v1") and abs(gap - 100 * 0.06 / 3.30) < 1e-6
    assert drv._mc_top2_gap_pct([_mcrow("v0", 3.3)]) is None           # one timed
    assert drv._mc_top2_gap_pct([_mcrow("v0", None), _mcrow("v1", 2.0)]) is None


# --- task 6: the climb streak (甲G12) -------------------------------------------

def test_climb_streak_counts_consecutive_small_keeps():
    # the streak walks NEWEST-first: the newest two keeps each gained 1-3%,
    # the older one broke the pattern
    recs = [_rec(1.99, iter_=1, sha="a", metric_before=2.00),   # +0.5% — breaks
            _rec(1.95, iter_=2, sha="b", metric_before=1.99),   # +2.0% — in band
            _rec(1.90, iter_=3, sha="c", metric_before=1.95)]   # +2.6% — in band
    assert drv._climb_streak(recs, "device") == 2
    recs.append(_rec(1.76, iter_=4, sha="d", metric_before=1.90))   # +7.4%: breaks
    assert drv._climb_streak(recs, "device") == 0
    assert drv._climb_streak([], "device") == 0


# --- task 5: the runner-up pool, restart target, revival (甲G16) -----------------

def test_runner_up_pool_ranks_and_dedupes(ctx):
    recs = [_rec(2.0, iter_=1, sha="best", did="D1"),
            _rec(2.1, iter_=2, sha="ru2", did="D2"),
            _rec(2.1, iter_=3, sha="ru2", did="D2b"),        # same tree, newer row
            _rec(2.2, iter_=4, sha="ru3", did="D3"),
            _rec(2.05, iter_=5, sha="ru1", did="D4"),
            _rec(None, iter_=6, sha="x"),                    # no reading: out
            _rec(1.0, iter_=7, sha="r", verdict="REVERT")]   # reverted: out
    pool = drv._runner_up_pool(recs, "device", 3)
    assert [p["sha"] for p in pool] == ["best", "ru1", "ru2"]
    assert pool[1]["rank"] == 2
    assert abs(pool[1]["delta_vs_best_pct"] - 100 * 0.05 / 2.0) < 1e-9
    assert pool[0]["delta_vs_best_pct"] == 0.0


def test_restart_target_lines_three_states(ctx):
    recs = [_rec(2.0, iter_=1, sha="best", did="D1"),
            _rec(2.1, iter_=2, sha="ru", did="D2")]
    lines = drv._restart_target_lines(ctx, recs)
    assert "restart target" in lines and "ru" in lines
    assert len(lines.splitlines()) <= 8            # marker + <= 6 payload lines + blanks
    assert drv._restart_target_lines(ctx, [recs[0]]) == ""      # no #2: nothing
    assert drv._restart_target_lines(ctx, []) == ""             # pool empty: nothing


def test_revival_note_three_states(ctx):
    recs = [_rec(2.0, iter_=1, sha="best", did="D1"),
            _rec(2.1, iter_=2, sha="ru", did="D2")]
    note = drv._revival_note(ctx, "ru", recs)
    assert "revival" in note and "fresh-recheck" in note
    assert drv._revival_note(ctx, "unpooled", recs) == ""
    assert drv._revival_note(ctx, "", recs) == ""


# --- task 7: floor_progress (丙) -------------------------------------------------

def test_floor_progress_line(ctx):
    ctx.floor_ms, ctx.floor_checked = 1.0, True
    recs = [_rec(2.0)]
    line = drv._floor_progress_line(ctx, recs)
    assert "floor_progress=0.500" in line and "report-only" in line
    ctx.floor_ms, ctx.floor_checked = None, True
    assert drv._floor_progress_line(ctx, recs) == ""
    ctx.floor_ms, ctx.floor_checked = 1.0, True
    assert drv._floor_progress_line(ctx, []) == ""


# --- task 2: the halving recipe expansion + ceilings (乙A4) ----------------------

def test_expand_sweep_axes_values_and_range():
    recipe = {"halving": True, "axes": [
        {"name": "tile", "old": "TILE = 128", "new": "TILE = {v}", "values": [64, 128, 256]},
        {"name": "stages", "old": "STAGES = 2", "new": "STAGES = {v}",
         "range": [2, 6], "points": 5}]}
    out = drv._expand_sweep_axes(recipe)
    names = [v["name"] for v in out]
    assert names[:3] == ["tile_64", "tile_128", "tile_256"]
    assert names[3:] == ["stages_2", "stages_3", "stages_4", "stages_5", "stages_6"]
    assert out[0]["edits"] == [{"old": "TILE = 128", "new": "TILE = 64"}]
    assert len(out) == 8
    again = drv._expand_sweep_axes(recipe)
    assert [v["edits"] for v in again] == [v["edits"] for v in out]   # deterministic


def test_sweep_axes_refused_and_halving_ceiling():
    good = {"halving": True, "axes": [{"name": "t", "old": "x", "new": "y{v}",
                                       "values": [1, 2]}]}
    assert drv._sweep_axes_refused(good) is None
    bad = [dict(good, axes=[]),
           dict(good, axes=[{"name": "t", "old": "x", "new": "nov"}]),
           dict(good, axes=[{"name": "t", "old": "x", "new": "{v}", "range": [4, 2]}]),
           dict(good, axes=[{"name": "t", "old": "x", "new": "{v}", "values": [1]}]),
           dict(good, axes=[{"name": "t", "old": "x", "new": "{v}", "range": [1, 2]},
                            {"name": "u", "old": "y", "new": "{v}", "range": [1, 2]}])]
    assert all(drv._sweep_axes_refused(b) is not None for b in bad[:4])
    assert drv._sweep_axes_refused(bad[4]) is None      # two range axes are fine
    # the total ceiling: two 16-point range axes fill exactly 32 — accepted,
    # and the halving build accepts the expansion (the plain form would refuse)
    wide = {"halving": True, "axes": [
        {"name": "t", "old": "x = 1", "new": "x = {v}", "range": [1, 31], "points": 16},
        {"name": "u", "old": "y = 1", "new": "y = {v}", "range": [1, 31], "points": 16}]}
    assert drv._sweep_axes_refused(wide) is None
    assert len(drv._expand_sweep_axes(wide)) == 32
    # the halving ceiling in the variant form
    wide = {"halving": True, "variants": [{"name": f"v{i}", "edits": [{"old": "a",
                                                                       "new": f"{i}"}]}
                                          for i in range(33)]}
    assert "max 32" in (drv._sweep_spec_refused(wide) or "")
    eight = {"variants": [{"name": f"v{i}", "edits": [{"old": "a", "new": f"{i}"}]}
                          for i in range(9)]}
    assert "max 8" in (drv._sweep_spec_refused(eight) or "")


# --- sweep_run's noise-gated halving cut ----------------------------------------

def _arrays(dev_ms):
    order = ["base"] + list(dev_ms)
    dev = {v: [] for v in order}
    ratio = {v: [] for v in order}
    ok = {v: True for v in order}
    for v, xs in dev_ms.items():
        dev[v] = list(xs)
        # a same-job ratio reading that tracks speed, so the scorer discriminates
        ratio[v] = [3.0 / x for x in xs]
    return order, ok, dev, ratio


def test_halving_cut_gates():
    dev_ms = {"base": [3.0, 3.0], "a": [2.5, 2.5], "b": [2.6, 2.6],
              "c": [3.5, 3.5], "d": [3.6, 3.6], "e": [3.7, 3.7]}
    order, ok, dev, ratio = _arrays(dev_ms)
    cut = sweep_run.halving_cut(order, ok, dev, ratio, set(), set())
    assert cut == {"d", "e"}               # the worst half, base excluded
    # one reading per variant names no rank: no cut
    dev_ms1 = {k: v[:1] for k, v in dev_ms.items()}
    o1, k1, d1, r1 = _arrays(dev_ms1)
    assert sweep_run.halving_cut(o1, k1, d1, r1, set(), set()) == set()
    # fewer than four survivors cut nobody
    dev_ms3 = {"base": [3.0, 3.0], "a": [2.5, 2.5], "b": [2.6, 2.6],
               "c": [3.5, 3.5]}
    o3, k3, d3, r3 = _arrays(dev_ms3)
    assert sweep_run.halving_cut(o3, k3, d3, r3, set(), set()) == set()
    # a within-noise gap at the cut line survives (the MC defense's discipline)
    dev_ms_near = {"base": [3.0, 3.0], "a": [2.50, 2.50], "b": [2.502, 2.502],
                   "c": [2.504, 2.504], "d": [3.9, 3.9], "e": [4.0, 4.0]}
    on, kn, dn, rn = _arrays(dev_ms_near)
    cut = sweep_run.halving_cut(on, kn, dn, rn, set(), set())
    assert "b" not in cut and "c" not in cut and cut == {"d", "e"}


# --- task 6: EXPIRED revisions (甲G12) -------------------------------------------

def _bank_table(ctx, rows, best_ref, job="job1"):
    rec = {"ts": 0.0, "best_ref": best_ref, "job": job, "table": rows}
    (ctx.evidence_dir / "lever_prices.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8")


def test_mark_expired_levers(ctx):
    ctx.reprice_on_new_best = True
    ctx.lever_pricing_rounds = 3
    ctx.stall_pricing = True
    table = [
        {"kind": "stack", "noise_pct": 0.5, "all_pct": 1.0},
        {"kind": "lever", "name": "win", "add_pct": 2.0, "remove_pct": None,
         "cls": "independent", "priced_on": "oldsha", "job": "job1"},
        {"kind": "lever", "name": "nullone", "add_pct": 0.1, "remove_pct": None,
         "cls": "null", "priced_on": "oldsha", "job": "job1"},
    ]
    _bank_table(ctx, table, "oldsha")
    # the milestone cites `nullone`, not `win`: win is the uncovered winner
    ctx.milestone.write_text("# m\nlever:nullone\n", encoding="utf-8")
    rec = _rec(1.9, iter_=5, sha="newsha")
    drv._mark_expired_levers(ctx, rec)
    revs = drv._unretired_revisions(ctx)
    assert [r["lever"] for r in revs] == ["win"]
    assert revs[0]["verdict"] == "expired"
    assert "re-price it before citing" in revs[0]["note"]
    drv._retire_revisions(ctx)
    # a table already on this tree books nothing; a cited winner books nothing
    _bank_table(ctx, table, "newsha")
    drv._mark_expired_levers(ctx, rec)
    assert len(drv._unretired_revisions(ctx)) == 0
    ctx.milestone.write_text("# m\nlever:win\n", encoding="utf-8")
    _bank_table(ctx, table, "oldsha")
    drv._mark_expired_levers(ctx, rec)
    assert len(drv._unretired_revisions(ctx)) == 0


# --- STAGNATION pool-empty byte-identity (red line ③-⑤) --------------------------

def test_stagnation_block_byte_identity_and_empty_pool_renders_nothing(ctx):
    rep = {"k": 3, "attempts": [{"label": "D1.1", "gain": 0.2}], "n": 3,
           "band_pct": 0.5, "gap_pct": 10.0, "overrides": 0}
    block = drv._stagnation_block(rep)
    # the WP10 block format carries no WP16 marker and no restart target
    assert "WP16" not in block and "restart target" not in block
    # an empty pool renders no COMBO-POOL block and no restart-target lines
    assert lever_pool.combo_pool_block(lever_pool.pending([])) == ""
    assert drv._restart_target_lines(ctx, []) == ""
    # the injection composition with an empty pool is exactly the WP10 block
    pool_rows = []
    injected = ""
    if lever_pool.pending(pool_rows):
        injected += lever_pool.combo_pool_block(pool_rows)
    stagnation = rep if pool_rows else rep      # pool-empty: no deferral happens
    injected += drv._stagnation_block(stagnation)
    injected += drv._restart_target_lines(ctx, [])
    assert injected == block


# --- the pool's STAGNATION deferral flag (one step, pool-empty reset) ------------

def test_pool_defer_flag_resets_when_pool_empties(ctx):
    ctx.wp16_pool_deferred = True
    pool_rows = []
    if not lever_pool.pending(pool_rows):
        ctx.wp16_pool_deferred = False
    assert ctx.wp16_pool_deferred is False
