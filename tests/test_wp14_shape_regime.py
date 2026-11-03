"""WP14: the shape-regime decision channel (two phases).

Phase 1 (甲G1+乙A5): the per-shape ledger — each workload of a group card gets
its own measured history (``per_shape_ratios`` / ``per_shape_best``, defined
ONLY in inner/trajectory.py per contract C8) — and the size-dispatch narrow
form: a decision line may declare shape-keyed lever value ranges
(``per_shape: <lever>[<workload>]=<value>``); a declared sweep is priced once
per workload and the feedback reads per shape; the landing is always the
student's runtime shape-dispatch code judged by the full eval gate.

Phase 2 (丙WP14): M0-level per-regime topology vocabulary — a decision (an M0
topology contract line included) may carry ``regime: <axis-predicate>``, a pure
function of the problem's shape axes that the driver machine-checks for
purity, mutual exclusion and coverage of every workload (a coverage failure
invalidates the freeze, bounced at authoring); per-regime NCU evidence reuses
the stall-pricing channel's sanctioned ``--workload`` scope, capped at
``regime_ncu_max`` (default 2) NEW jobs per meta step. A run whose milestone
carries no regime/per_shape lines renders and buys nothing: byte-identical to
the pre-WP14 loop.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    _MILESTONE, REC, _ctx,
)

from forgekernel.inner.trajectory import (  # noqa: E402
    IterationRecord, append_record, build_digest, load_records, per_shape_ledger,
    per_shape_best_readings,
)
from forgekernel.milestone import (  # noqa: E402
    parse_per_shape_declarations, parse_regime_expr, parse_regime_predicates,
    regime_coverage, regime_expr_matches, validate_milestone,
)
from forgekernel.loop.driver import (  # noqa: E402
    LoopContext, _arch_probe_stamp, _parse_arch_candidates, _run_stall_pricing,
    _sweep_per_shape_feedback, _wp14_declaration_violations,
    _wp14_regime_evidence_block, _wp14_regime_label_line,
    _wp14_shape_ledger_block, parse_sweep_results,
)


def _rec(iter_no, verdict, workloads=None, **kw):
    ms = 1.0
    if workloads:
        vals = [w.get("device_median_ms") for w in workloads
                if isinstance(w.get("device_median_ms"), (int, float))]
        ms = round(pow(1.0, 1) * __import__("math").exp(
            sum(__import__("math").log(v) for v in vals) / len(vals)), 6) if vals else None
    return IterationRecord(iter=iter_no, decision_id=f"D1.{iter_no}", change="c",
                           correctness=verdict != "REVERT", verdict=verdict,
                           device_median_ms=ms, bwd_median_ms=ms,
                           n_workloads=len(workloads or []),
                           workloads=workloads or [], **kw)


# --- the per-shape ledger (contract C8: trajectory.py is the only definition) ---

_WL_A = [{"name": "prefill", "device_median_ms": 1.0, "vs_ref_device": 0.98},
         {"name": "decode", "device_median_ms": 0.100, "vs_ref_device": 1.01}]
_WL_B = [{"name": "prefill", "device_median_ms": 0.900, "vs_ref_device": 0.97},
         {"name": "decode", "device_median_ms": 0.120, "vs_ref_device": 1.02}]


def test_ledger_keep_advances_only_beating_workloads():
    ratios, best = per_shape_ledger([], _WL_A, 1, improve=True)
    assert ratios == {"prefill": 0.98, "decode": 1.01}
    assert best == {"prefill": {"ms": 1.0, "iter": 1}, "decode": {"ms": 0.1, "iter": 1}}
    # a second KEEP that wins only prefill: decode's entry is inherited untouched
    ratios2, best2 = per_shape_ledger([_rec(1, "KEEP", _WL_A, per_shape_best=best)],
                                      _WL_B, 2, improve=True)
    assert ratios2["prefill"] == 0.97 and ratios2["decode"] == 1.02
    assert best2 == {"prefill": {"ms": 0.9, "iter": 2}, "decode": {"ms": 0.1, "iter": 1}}


def test_ledger_revert_never_advances_and_single_input_is_empty():
    _, best1 = per_shape_ledger([], _WL_A, 1, improve=True)
    # a failed row: its own ratios are recorded, the ledger is inherited as-is
    r2, best2 = per_shape_ledger([_rec(1, "KEEP", _WL_A, per_shape_best=best1)],
                                 _WL_B, 2, improve=False)
    assert r2 == {"prefill": 0.97, "decode": 1.02}
    assert best2 == best1
    # a single-input bench: {} on both — the top-level fields are that one shape
    assert per_shape_ledger([], [], 3, improve=True) == ({}, {})


def test_ledger_roundtrip_and_recompute_agree(tmp_path):
    _, best1 = per_shape_ledger([], _WL_A, 1, improve=True)
    path = tmp_path / "status.jsonl"
    append_record(path, _rec(1, "KEEP", _WL_A, per_shape_ratios={"prefill": 0.98},
                             per_shape_best=best1))
    [back] = load_records(path)
    assert back.per_shape_best == best1 and back.per_shape_ratios == {"prefill": 0.98}
    append_record(path, _rec(2, "KEEP", _WL_B))
    recs = load_records(path)
    # the read side recomputes from the KEEP rows' own workload blocks (pre-WP14
    # rows are accounted too) and matches the snapshot arithmetic where both exist
    assert per_shape_best_readings(recs) == {
        "prefill": {"ms": 0.9, "iter": 2}, "decode": {"ms": 0.1, "iter": 1}}
    d = build_digest(recs, {"value": 0.5})
    assert d["per_shape_best"] == per_shape_best_readings(recs)
    assert d["achieved"]["aggregate"] == "geomean over 2 workloads"


# --- regime predicates: purity grammar + coverage/mutual-exclusion --------------

def test_regime_expr_purity_whitelist():
    assert parse_regime_expr("m >= 1024") == [("m", ">=", 1024.0)]
    assert parse_regime_expr("m < 1024 and n == 4096") == [("m", "<", 1024.0),
                                                           ("n", "==", 4096.0)]
    for bad in ("data == zeros", "m >= ref", "m > x", "m", "m >= 1 or n < 2"):
        with pytest.raises(ValueError):
            parse_regime_expr(bad)
    # `best_ms < 1.0` is syntactically well-formed but names no shape axis: the
    # semantic purity check is the coverage layer's (it names the axis as no axis
    # of this card — see the coverage test below)


def test_regime_coverage_failure_is_reported():
    axes = {"prefill": {"m": 8192, "n": 4096}, "decode": {"m": 64, "n": 4096}}
    ok = [("D0.3", "m >= 1024"), ("D0.4", "m < 1024")]
    assert regime_coverage(ok, axes) == []
    # not covering: both predicates read m >= 1024, decode matches nothing
    miss = regime_coverage([("D0.3", "m >= 1024"), ("D0.4", "m >= 1024")], axes)
    assert any("coverage failed" in v and "decode" in v for v in miss)
    # not mutually exclusive: m >= 64 matches both workloads
    overlap = regime_coverage([("D0.3", "m >= 1024"), ("D0.4", "m >= 64")], axes)
    assert any("mutually exclusive" in v for v in overlap)
    # not pure: a predicate whose axis is no shape axis of the card fails there
    assert any("no shape axis" in v
               for v in regime_coverage([("D0.3", "best_ms < 1")], axes))
    # predicates with no axes to check against = an unverifiable freeze = violation
    assert regime_coverage([("D0.3", "m >= 1024")], {})
    # no predicates: nothing to check (the pre-WP14 shape)
    assert regime_coverage([], axes) == []
    # a workload without the axis cannot be matched by it
    assert not regime_expr_matches(parse_regime_expr("m >= 1"), {"n": 4096})


def test_regime_predicates_parse_per_decision_block():
    text = _MILESTONE.replace(
        "**D1.1 base**", "**D1.1 base**\nregime: m >= 1024", 1).replace(
        "**D2.3 next**", "**D2.3 next**\nregime: m < 1024", 1)
    preds = dict(parse_regime_predicates(text))
    assert preds.get("D1.1") == "m >= 1024" and preds.get("D2.3") == "m < 1024"
    assert parse_regime_predicates(_MILESTONE) == []   # the bare milestone has none


# --- per_shape declarations ------------------------------------------------------

def test_per_shape_declaration_parse():
    text = _MILESTONE.replace(
        "**D1.1 base**",
        "**D1.1 base**\nper_shape: raster[prefill]={row,col}, raster[decode]=row", 1)
    decls = parse_per_shape_declarations(text)
    assert decls["D1.1"] == [("raster[prefill]", "{row,col}"), ("raster[decode]", "row")]
    # a key naming neither a workload nor a shape predicate comes back parseable
    # but is a violation at the driver (the composition point with the card axes)
    assert parse_per_shape_declarations(_MILESTONE) == {}


def test_wp14_declaration_violations(tmp_path):
    ctx = _ctx(tmp_path, None)
    axes_file = ctx.workspace / "eval" / "leaderboard" / "problem.json"
    axes_file.parent.mkdir(parents=True)
    axes_file.write_text(json.dumps({
        "group": [{"name": "prefill", "workload": {"axes": {"m": 8192}}},
                  {"name": "decode", "workload": {"axes": {"m": 64}}}]}), encoding="utf-8")
    good = _MILESTONE.replace("**D1.1 base**", "**D1.1 base**\nregime: m >= 1024", 1).replace(
        "**D2.3 next**", "**D2.3 next**\nregime: m < 1024", 1)
    assert _wp14_declaration_violations(ctx, good) == []
    miss = _MILESTONE.replace("**D1.1 base**", "**D1.1 base**\nregime: m >= 1024", 1).replace(
        "**D2.3 next**", "**D2.3 next**\nregime: m >= 1024", 1)
    assert any("coverage failed" in v for v in _wp14_declaration_violations(ctx, miss))
    impure = _MILESTONE.replace("**D1.1 base**", "**D1.1 base**\nregime: best_ms < 1", 1)
    assert any("no shape axis" in v
               for v in _wp14_declaration_violations(ctx, impure))
    typo = _MILESTONE.replace("**D1.1 base**",
                              "**D1.1 base**\nper_shape: raster[preflil]=row", 1)
    assert any("names no workload" in v
               for v in _wp14_declaration_violations(ctx, typo))
    # patch mode: only lines the patch ADDED are held — the same grandfathering
    # the gate-line hygiene follows
    assert _wp14_declaration_violations(ctx, miss, baseline=_MILESTONE) != []
    assert _wp14_declaration_violations(ctx, _MILESTONE, baseline=_MILESTONE) == []
    pre_rule = _wp14_declaration_violations(ctx, miss, baseline=miss)
    assert pre_rule == []


def test_authoring_milestone_with_regime_lines_still_conformant():
    # the new vocabulary is §9-neutral: a milestone that carries it passes the
    # schema validator unchanged (the coverage check is the driver's, with axes)
    text = _MILESTONE.replace(
        "**D1.1 base**",
        "**D1.1 base**\nregime: m >= 1024\n- per_shape: raster[prefill]=row", 1)
    assert validate_milestone(text) == validate_milestone(_MILESTONE) == []


# --- per-shape ledger + regime rows rendering (zero-regime = byte-identical) -----

def test_shape_ledger_block_zero_and_multi(tmp_path):
    ctx = _ctx(tmp_path, None)
    assert _wp14_shape_ledger_block(ctx, []) == ""          # no rows at all
    single = [_rec(1, "KEEP")]                               # a single-input bench
    assert _wp14_shape_ledger_block(ctx, single) == ""
    recs = [_rec(1, "KEEP", _WL_A, per_shape_ratios={"prefill": 0.98}),
            _rec(2, "KEEP", _WL_B)]
    block = _wp14_shape_ledger_block(ctx, recs)
    assert "Per-shape ledger" in block and "`prefill` best 0.9000 ms (iter 2" in block
    assert "`decode` best 0.1000 ms (iter 1" in block
    assert "full eval gate" in block                          # the landing rule
    assert block.count("\n") <= 8                             # C10: <= 6 ledger rows


def test_regime_label_and_evidence_zero_regime_byte_identical(tmp_path):
    ctx = _ctx(tmp_path, None)
    # the bare milestone carries no regime lines: every WP14 block is '' — the
    # injection zone renders byte-identically to the pre-WP14 loop
    assert _wp14_regime_label_line(ctx) == ""
    assert _wp14_regime_evidence_block(ctx, None, []) == ""
    assert _wp14_regime_label_line(ctx, _MILESTONE) == ""
    assert _wp14_regime_evidence_block(ctx, type("P", (), {"best_sha": None,
                                                           "head_sha": None})(), []) == ""


def test_regime_map_and_evidence_selection(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, None)
    text = _MILESTONE.replace("### D1.1", "regime: m >= 1024\n\n### D1.1", 1).replace(
        "### D1.2", "regime: m < 1024\n\n### D1.2", 1)
    axes_file = ctx.workspace / "eval" / "leaderboard" / "problem.json"
    axes_file.parent.mkdir(parents=True)
    axes_file.write_text(json.dumps({
        "group": [{"name": "prefill", "workload": {"axes": {"m": 8192}}},
                  {"name": "decode", "workload": {"axes": {"m": 64}}},
                  {"name": "mid", "workload": {"axes": {"m": 2048}}}]}), encoding="utf-8")
    ctx.milestone.write_text(text, encoding="utf-8")
    rmap = {"prefill": "D1.1: m >= 1024", "decode": "D1.2: m < 1024",
            "mid": "D1.1: m >= 1024"}
    import forgekernel.loop.driver as drv
    monkeypatch.setattr(drv, "_wp14_regime_map", lambda c, t=None: dict(rmap))
    label = _wp14_regime_label_line(ctx)
    assert "Regime labels" in label and "`prefill` → regime `D1.1: m >= 1024`" in label
    # evidence selection: one regime representative per NEW (sha, workload) pair,
    # at most regime_ncu_max new jobs per step, memoized beyond it
    bought, priced = [], type("P", (), {"best_sha": "abc", "head_sha": "abc"})()

    def fake_stall(c, ref=None, bank_as=(), label="", workload=None):
        bought.append(workload)
        return ("LEDGER", None, "DOSSIER", None)

    monkeypatch.setattr(drv, "_run_stall_pricing", fake_stall)
    ctx.stall_pricing = True
    ctx.regime_ncu_max = 2
    out = _wp14_regime_evidence_block(ctx, priced, [])
    assert bought == ["decode", "mid"]          # one per DISTINCT regime, capped at 2
    assert out.count("\n") <= 7                 # C10: <= 6 lines
    assert "cap 2" in out
    # the capped-out regime is NOT memo'd: a later meta step buys it (the cap is
    # per step, the memo only carries what was already banked); after that call
    # every (tree, workload) pair is memo'd and nothing is re-bought
    bought.clear()
    out2 = _wp14_regime_evidence_block(ctx, priced, [])
    assert bought == ["prefill"] and out2.count("regime `") == 3
    bought.clear()
    out3 = _wp14_regime_evidence_block(ctx, priced, [])
    assert bought == [] and out3.count("regime `") == 3
    # cap 0 disables the channel entirely
    ctx2 = _ctx(tmp_path, None)
    ctx2.regime_ncu_max = 0
    monkeypatch.setattr(drv, "_wp14_regime_map", lambda c, t=None: dict(rmap))
    assert _wp14_regime_evidence_block(ctx2, priced, []) == ""


# --- per-shape sweep: selection + per-shape feedback table -----------------------

def test_sweep_per_shape_selection(tmp_path):
    ctx = _ctx(tmp_path, None)
    axes_file = ctx.workspace / "eval" / "leaderboard" / "problem.json"
    axes_file.parent.mkdir(parents=True)
    axes_file.write_text(json.dumps({
        "group": [{"name": "prefill", "workload": {"axes": {"m": 8192}}},
                  {"name": "decode", "workload": {"axes": {"m": 64}}}],
        "meta_workloads": ["prefill", "decode"]}), encoding="utf-8")
    import forgekernel.loop.driver as drv
    decl = _MILESTONE.replace(
        "**D1.1 base**",
        "**D1.1 base**\nper_shape: raster[prefill]={row,col}, raster[decode]=row", 1)
    ctx.milestone.write_text(decl, encoding="utf-8")
    assert drv._sweep_per_shape_workloads(ctx, "D1.1") == ["prefill", "decode"]
    assert drv._sweep_per_shape_workloads(ctx, "D1.2") == []   # no declaration
    bare = ctx.milestone.read_text(encoding="utf-8").replace(
        "per_shape: raster[prefill]={row,col}, raster[decode]=row\n", "")
    ctx.milestone.write_text(bare, encoding="utf-8")
    assert drv._sweep_per_shape_workloads(ctx, "D1.1") == []
    # predicate keys name no workload: the whole meta list is the sweep scope
    axis_decl = _MILESTONE.replace(
        "**D1.1 base**", "**D1.1 base**\nper_shape: raster[m >= 1024]=row", 1)
    ctx.milestone.write_text(axis_decl, encoding="utf-8")
    assert drv._sweep_per_shape_workloads(ctx, "D1.1") == ["prefill", "decode"]


def test_sweep_per_shape_feedback_table():
    rows_pre = parse_sweep_results(
        "SWEEP name=base ratio=1.0000 floor=0.50% delta=n/a ms=1.0000 ok=pass\n"
        "SWEEP name=tile64 ratio=1.0500 floor=0.50% delta=+5.00% ms=0.9524 ok=pass\n"
        "SWEEP-OK variants=1 rounds=2 base_ms=1.0\nSWEEP-WINNER name=tile64")
    rows_dec = parse_sweep_results(
        "SWEEP name=base ratio=1.0000 floor=0.50% delta=n/a ms=0.1000 ok=pass\n"
        "SWEEP name=tile64 ratio=0.9800 floor=0.50% delta=-2.00% ms=0.1020 ok=pass\n"
        "SWEEP-OK variants=1 rounds=2 base_ms=0.1\nSWEEP-WINNER name=none")
    per_wl = {"prefill": {"rows": rows_pre[0], "winner": "tile64", "fail": None},
              "decode": {"rows": rows_dec[0], "winner": None, "fail": None}}
    fb = _sweep_per_shape_feedback("notes/D2.1.json", "D2.1", ["prefill", "decode"], per_wl)
    assert "per-shape" in fb and "| variant |" in fb
    assert "**winner**" in fb                    # the per-column winner, within its job
    assert "cross-job ratios are not one measurement" in fb
    assert "full eval gate" in fb                # the landing rule, verbatim duty
    # no reading anywhere -> the refusal carries the per-workload reasons
    fb2 = _sweep_per_shape_feedback("notes/D2.1.json", "D2.1", ["prefill", "decode"],
                                    {"prefill": {"fail": "node-dirty"},
                                     "decode": {"fail": "spec-unreadable"}})
    assert "No sweep reading" in fb2 and "`prefill`: node-dirty" in fb2


def test_sweep_scope_cmd_scopes_the_bench():
    import forgekernel.loop.driver as drv
    cmd = drv._sweep_scope_cmd("__sweep_D2.1.json", "prefill")
    assert cmd.startswith("cmd:python3 -c")
    assert "l._SCOPE_WANT['name']='prefill'" in cmd       # the tool's own assignment
    assert "s.main(['__sweep_D2.1.json'])" in cmd
    # shell-safe: a validated token never carries a quote of its own
    assert drv._sweep_scope_cmd("x.json", "de-code_1.0").count("'") % 2 == 0


# --- ARCH_CANDIDATES per-regime rows ---------------------------------------------

def test_arch_candidates_per_regime_lines():
    text = ("ARCH_CANDIDATES: dual\n"
            "ARCH_CANDIDATES: dual regime=m >= 1024\n"
            "ARCH_CANDIDATES：single\n")
    assert _parse_arch_candidates(text) == [("dual", ""), ("dual", "m >= 1024"),
                                            ("single", "")]
    assert _parse_arch_candidates(_MILESTONE) == []


def test_arch_stamp_zero_regime_byte_identical_and_regime_note_line():
    readings = {"a.json": 1.2, "b.json": 1.4}
    base = _arch_probe_stamp(["a.json", "b.json"], readings, floor=0.02)
    assert _arch_probe_stamp(["a.json", "b.json"], readings, floor=0.02) == base
    reg = _arch_probe_stamp(["a.json", "b.json"], readings, floor=0.02,
                            regime_note="per-regime candidates: dual regime=m >= 1024")
    assert reg[:1] == base[:1] and reg[1] == \
        "     per-regime candidates: dual regime=m >= 1024" and reg[2:] == base[1:]
    skip = _arch_probe_stamp(["a.json"], {}, skipped="M0 probes skipped (budget)",
                             regime_note="per-regime candidates: dual regime=m < 1024")
    assert skip == ["<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)",
                    "     per-regime candidates: dual regime=m < 1024",
                    "     M0 probes skipped (budget) -->"]


# --- the landing path: a per_shape decision delivers through the standard gate ----

def test_dispatch_delivery_walks_the_standard_gate_path(tmp_path):
    """The size-dispatch landing fixture: a per_shape-declared decision is
    delivered as an ordinary record block (same fields, same gate) — the full
    gate's per-workload readings ARE the per-shape verdict, and the ledger
    advances per workload from that same row."""
    ctx = _ctx(tmp_path, None)
    wls = [{"name": "prefill", "device_median_ms": 0.90, "vs_ref_device": 0.97},
           {"name": "decode", "device_median_ms": 0.098, "vs_ref_device": 0.99}]
    ratios, best = per_shape_ledger([], wls, 1, improve=True,
                                    gate_window=ctx.gate_window)
    rec = _rec(1, "KEEP", wls, per_shape_ratios=ratios, per_shape_best=best)
    append_record(ctx.status_path, rec)
    [stored] = load_records(ctx.status_path)
    # one ordinary row carries the whole per-shape verdict for the gate...
    assert stored.verdict == "KEEP" and stored.workloads == wls
    assert stored.per_shape_ratios == {"prefill": 0.97, "decode": 0.99}
    assert stored.per_shape_best == {"prefill": {"ms": 0.9, "iter": 1},
                                     "decode": {"ms": 0.098, "iter": 1}}
    # ...and the ledger block renders it for the teacher with no side channel
    block = _wp14_shape_ledger_block(ctx, [stored])
    assert "0.9000 ms" in block and "0.0980 ms" in block


# --- MC-DISPATCH (乙A14): the per-shape structural prescreen ---------------------

def test_mc_dispatch_per_shape_readings_and_no_mechanical_merge(tmp_path, monkeypatch):
    """The prescreen times each structural variant PER WORKLOAD; a variant that
    wins everywhere is the ordinary winner; differing per-shape winners take the
    ordinary no-winner REVERT with the per-workload readings banked — the
    dispatch composition is the teacher's per_shape declaration, never a
    driver-written merge."""
    import forgekernel.loop.driver as drv

    class Rec:
        decision_id = "D2.4"
        change = "structure dispatch"

    ctx = _ctx(tmp_path, None)
    (ctx.workspace / "v_a.py").write_text("a", encoding="utf-8")
    (ctx.workspace / "v_b.py").write_text("b", encoding="utf-8")
    decl = _MILESTONE.replace(
        "**D2.3 next**",
        "**D2.3 next**\nper_shape: structure[prefill]=A, structure[decode]=B", 1)
    ctx.milestone.write_text(decl, encoding="utf-8")
    pjson = ctx.workspace / "eval" / "leaderboard" / "problem.json"
    pjson.parent.mkdir(parents=True)
    pjson.write_text(json.dumps({
        "group": [{"name": "prefill", "workload": {"axes": {"m": 8192}}},
                  {"name": "decode", "workload": {"axes": {"m": 64}}}],
        "meta_workloads": ["prefill", "decode"]}), encoding="utf-8")

    logs = {
        "prefill": ("MC name=v_a.py smoke=pass ms=0.9 ratio=1.06 delta=+6.00%\n"
                    "MC name=v_b.py smoke=pass ms=0.95 ratio=1.01 delta=+1.00%\n"
                    "MC-WINNER name=v_a.py"),
        "decode": ("MC name=v_a.py smoke=pass ms=0.1 ratio=1.00 delta=+0.00%\n"
                   "MC name=v_b.py smoke=pass ms=0.095 ratio=1.05 delta=+5.00%\n"
                   "MC-WINNER name=v_b.py"),
    }
    jobs = []

    def fake_mc_job(be, branch, cmd, what, rerun=True):
        jobs.append(what)
        return (logs[next(w for w in logs if f"[{w}]" in what)], "j1")

    monkeypatch.setattr(drv, "_mc_job", fake_mc_job)
    monkeypatch.setattr(drv, "_mc_seed_toolkit", lambda c, be: None)
    monkeypatch.setattr(drv, "_mc_write_variant_archive", lambda c, i, s: None)
    (ctx.workspace / "notes").mkdir(exist_ok=True)   # the seed would create it
    class _BE0:
        def commit(self, *a, **k):
            return "head"
        def push_branch(self, *a, **k):
            return None
        def reset(self, *a, **k):
            return None
    out = drv._mc_deliver_per_shape(
        ctx, _BE0(), 7, Rec(), "base", None, 2.0, ["v_a.py", "v_b.py"],
        ["prefill", "decode"])
    assert len(jobs) == 2                       # one prescreen job per workload
    assert out["winner"] is None and out["revert"] is not None
    assert out["revert"].verdict == "REVERT"
    assert "per-shape winners differ" in out["revert"].error
    tags = {r["workload"] for r in out["rows"]}   # the objective table reads per workload
    assert tags == {"prefill", "decode"}
    # the same winner on every workload is the ordinary winner path (full gate)
    logs_same = {k: v.replace("v_b.py", "v_a.py") if k == "decode" else v
                 for k, v in logs.items()}
    monkeypatch.setattr(drv, "_mc_job",
                        lambda be, branch, cmd, what, rerun=True:
                        (logs_same[next(w for w in logs_same if f"[{w}]" in what)], "j1"))
    monkeypatch.setattr(drv, "_mc_archive_variants", lambda c, be, i, v, w: None)
    (ctx.workspace / ctx.kernel_file).parent.mkdir(parents=True, exist_ok=True)
    out2 = drv._mc_deliver_per_shape(
        ctx, _BE0(), 8, Rec(), "base", None, 2.0, ["v_a.py", "v_b.py"],
        ["prefill", "decode"])
    assert out2["winner"] == "v_a.py" and out2["revert"] is None


def test_per_shape_predicate_key_matching_no_workload_is_a_violation(tmp_path):
    """An axis-predicate key must name a non-empty shape set: a key that matches
    no workload of the card is a typo'd dispatch, not a regime."""
    ctx = _ctx(tmp_path, None)
    axes_file = ctx.workspace / "eval" / "leaderboard" / "problem.json"
    axes_file.parent.mkdir(parents=True)
    axes_file.write_text(json.dumps({
        "group": [{"name": "prefill", "workload": {"axes": {"m": 8192}}},
                  {"name": "decode", "workload": {"axes": {"m": 64}}}],
        "meta_workloads": ["prefill", "decode"]}), encoding="utf-8")
    empty = _MILESTONE.replace(
        "**D1.1 base**", "**D1.1 base**\nper_shape: raster[m >= 999999]=row", 1)
    v = _wp14_declaration_violations(ctx, empty)
    assert any("matches no workload" in x for x in v)
    good = _MILESTONE.replace(
        "**D1.1 base**", "**D1.1 base**\nper_shape: raster[m >= 1024]=row", 1)
    assert _wp14_declaration_violations(ctx, good) == []
