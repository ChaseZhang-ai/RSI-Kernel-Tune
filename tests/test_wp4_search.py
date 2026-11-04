"""WP4 (search-mechanism upgrade): LANDABLE rendering, SWEEP_REQUEST decisions,
and slow-decision regression bookkeeping.

Every new path here is a READING or a PROMPT line — landings still go through
the ordinary record -> eval gate, and the student line stays teacher-free:
the driver renders, the teacher authors, the student delivers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peregrine_kernel.loop.driver import _lever_price_block, parse_lever_prices

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wp4"

PRICE_STACK = json.loads((FIXTURES / "lever_price_stack.json").read_text(encoding="utf-8"))

# The same table as lever_pricing_run output (the log the driver actually parses),
# with the all-pos combination above the job's own noise floor.
PRICE_LOG_WITH_STACK = (
    "PRICE variants: base +ring +lane +bad all all-ring all-lane all-bad all-pos\n"
    "LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
    "spin_add=880319 spin_all_minus=996231 cubin_same_as_base=False\n"
    "LEVER lane add=+0.20% remove=-4.90% add_ms=3.39 remove_ms=3.135 class=coupled "
    "spin_add=996231 spin_all_minus=880319 cubin_same_as_base=False\n"
    "LEVER bad add=-17.70% remove=+6.39% add_ms=4.10 remove_ms=3.05 class=harmful "
    "spin_add=1120000 spin_all_minus=900000 cubin_same_as_base=False\n"
    "VARIANT all-pos levers=ring,lane delta=+12.30% ms=3.04 spin=700000 "
    "q=auto: the stack without the levers that hurt alone (dropped bad)\n"
    "STACK all=-9.29% sum_singles=-8.94% coupling=-0.35% all_pos=+12.30% "
    "all_pos_levers=ring,lane dropped=bad\n"
    "PRICE noise_floor=0.50% metric=vs_ref_device\n"
    "PRICE-OK levers=3 rounds=3 base_ms=3.4052 all_ms=3.72 spin_base=1125779 spin_all=1300000 "
    "tensor_base=37.85 tensor_all=30.0\n")


# --- task 1: LANDABLE -----------------------------------------------------------


def test_landable_renders_when_all_pos_clears_the_floor():
    block = _lever_price_block(PRICE_STACK)
    assert block.count("LANDABLE:") == 1
    line = next(ln for ln in block.splitlines() if ln.startswith("LANDABLE:"))
    # the measured combination, its expected value, the job's own floor, the tree
    assert "levers=ring,lane" in line and "expected=+12.30%" in line
    assert "(floor=0.50%)" in line and "priced_on=5f3a2b1c4d5e" in line
    # the notice routes through the teacher: a milestone patch merges the levers,
    # the student delivers once — the driver never applies anything itself
    assert "milestone patch" in line and "eval gate" in line
    # and it sits at the top of the price block, before the lever table
    assert block.index("LANDABLE:") < block.index("| lever |")


def test_landable_silent_inside_the_noise_band():
    quiet = [dict(t) for t in PRICE_STACK]
    for t in quiet:
        if t["kind"] == "stack":
            t["all_pos_pct"] = 0.3                      # below the 0.5% floor
        if t["kind"] == "variant":
            t["delta_pct"] = 0.3
    assert "LANDABLE:" not in _lever_price_block(quiet)


def test_landable_silent_without_an_all_pos_reading():
    plain = [t for t in PRICE_STACK if t["kind"] != "variant"]
    plain = [dict(t) for t in plain]
    for t in plain:
        if t["kind"] == "stack":
            t["all_pos_pct"] = None
            t["all_pos_levers"] = []
            t["dropped"] = []
    assert "LANDABLE:" not in _lever_price_block(plain)


def test_landable_flows_from_the_parsed_job_log():
    """Parser -> price block: the log the pricing job printed renders LANDABLE
    with the tree the table was priced on."""
    table, stack, fail = parse_lever_prices(PRICE_LOG_WITH_STACK)
    assert fail is None and stack and stack.get("all_pos_pct") == 12.3
    for row in table:
        row.setdefault("priced_on", "5f3a2b1c4d5e6f708192a3b4c5d6e7f8091a2b3c")
    if stack.get("priced_on") is None:
        stack["priced_on"] = "5f3a2b1c4d5e6f708192a3b4c5d6e7f8091a2b3c"
    block = _lever_price_block(table + [dict(stack, kind="stack")])
    assert "LANDABLE: all-pos levers=ring,lane expected=+12.30% (floor=0.50%)" in block


def test_landable_keeps_the_student_line_teacher_free(tmp_path):
    """The landing channel is the teacher's milestone patch and the student's
    ordinary bundle delivery: a meta step that reads a LANDABLE table must not
    touch the kernel, and the diagnose prompt must carry the notice for the
    teacher to act on."""
    from test_loop_driver import (FakeBackend, _DIAG, _MILESTONE, _patching,
                                  _lever_design_with_subsets, _lever_kernel,
                                  _PRICE_LOG_SUBSETS, _STALL_LOG_OK, _lever_ctx)
    kernel = tmp_path / "flash_attn_dsl" / "flash_bwd.py"
    kernel.parent.mkdir(exist_ok=True)
    kernel.write_text("def fwd():\n    pass\n", encoding="utf-8")
    # what the design-turn fixture writes as the best tree's kernel (the tree the
    # table was priced on) — the baseline the driver must not add lever edits on top of
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _lever_kernel(scratch)
    expected = (scratch / "flash_attn_dsl" / "flash_bwd.py").read_text(encoding="utf-8")
    be = FakeBackend([_lever_design_with_subsets(tmp_path), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, _PRICE_LOG_SUBSETS])
    ctx = _lever_ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    from peregrine_kernel.loop.driver import run_meta_step
    run_meta_step(ctx)
    diag = be.prompts[1]
    # the all-pos combination of the subsets fixture (+12.30% over a 0.5% floor)
    assert "LANDABLE: all-pos levers=ring,lane expected=+12.30%" in diag
    # ...and the driver applied nothing to the kernel itself
    assert kernel.read_text(encoding="utf-8") == expected


# --- task 2: SWEEP_REQUEST ------------------------------------------------------

import json as _json  # noqa: E402  (module level is fine; keep the fixture block above readable)

SWEEP_RECIPE = _json.loads((FIXTURES / "sweep_recipe.json").read_text(encoding="utf-8"))
SWEEP_LOG = (FIXTURES / "sweep_job_output.txt").read_text(encoding="utf-8")
SWEEP_CMD_EXPECT = "cmd:python3 notes/sweep_run.py notes/__sweep_D1.1.json"
# (kept verbatim above: the expected submit line is exactly the brief's cmd shape)

from peregrine_kernel.loop.driver import (  # noqa: E402
    _canonical_recipe_hash,
    _milestone_sweep_recipe,
    _sweep_spec_refused,
    parse_sweep_results,
)


def _milestone_with_sweep(recipe: dict) -> str:
    """The §9-conformant milestone with a teacher-authored `sweep:` recipe on
    the D1.1 decision line."""
    from test_loop_driver import _MILESTONE
    anchor = "- Consumer phase: [PHASE: none]\n"
    return _MILESTONE.replace(
        anchor, anchor + "sweep: " + _json.dumps(recipe, ensure_ascii=False) + "\n", 1)


def _sweep_student(tmp_path, recipe, *, mangle=False, drop_field=False):
    """A student turn that copies the milestone recipe to notes/D1.1.json and
    ends with the request line; `mangle` smuggles one extra variant in."""
    def agent(_prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        submitted = _json.loads(_json.dumps(recipe))
        if mangle:
            submitted["variants"].append(
                {"name": "mine", "edits": [{"old": "a", "new": "b"}]})
        if drop_field:
            submitted.pop("criterion")
        (tmp_path / "notes" / "D1.1.json").write_text(
            _json.dumps(submitted, indent=3, ensure_ascii=False), encoding="utf-8")
        return "SWEEP_REQUEST: notes/D1.1.json"
    return agent


def test_sweep_recipe_hash_matches_across_formatting_and_breaks_on_content():
    a = _json.loads(_json.dumps(SWEEP_RECIPE, indent=3))
    b = _json.loads(_json.dumps({k: SWEEP_RECIPE[k] for k in reversed(list(SWEEP_RECIPE))}))
    assert _canonical_recipe_hash(a) == _canonical_recipe_hash(b)
    smuggled = _json.loads(_json.dumps(SWEEP_RECIPE))
    smuggled["variants"][0]["edits"][0]["new"] = "STAGE_TILE = 32"
    assert _canonical_recipe_hash(smuggled) != _canonical_recipe_hash(SWEEP_RECIPE)
    extra = _json.loads(_json.dumps(SWEEP_RECIPE))
    extra["variants"].append({"name": "mine", "edits": [{"old": "a", "new": "b"}]})
    assert _canonical_recipe_hash(extra) != _canonical_recipe_hash(SWEEP_RECIPE)


def test_sweep_recipe_is_read_from_the_milestone_not_the_student(tmp_path):
    from test_loop_driver import _ctx, FakeBackend
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_milestone_with_sweep(SWEEP_RECIPE), encoding="utf-8")
    got = _milestone_sweep_recipe(ctx, "D1.1")
    assert got == SWEEP_RECIPE
    assert _milestone_sweep_recipe(ctx, "D2.3") is None          # no sweep field there


def test_sweep_spec_refused_over_eight_variants_or_bad_shape():
    ok = {"variants": [{"name": "v", "edits": [{"old": "a", "new": "b"}]}],
          "criterion": "min same-job ratio above floor"}
    assert _sweep_spec_refused(ok) is None
    big = {"variants": [{"name": f"v{i}", "edits": [{"old": "a", "new": "b"}]}
                        for i in range(9)]}
    assert "max 8" in _sweep_spec_refused(big)
    dup = {"variants": [{"name": "v", "edits": [{"old": "a", "new": "b"}]},
                        {"name": "v", "edits": [{"old": "a", "new": "c"}]}]}
    assert "duplicated" in _sweep_spec_refused(dup)
    noedit = {"variants": [{"name": "v", "edits": []}]}
    assert "no edits" in _sweep_spec_refused(noedit)
    assert "unknown criterion" in _sweep_spec_refused({**ok, "criterion": "max vibes"})


def test_sweep_request_without_a_prewritten_recipe_is_refused_without_a_job(tmp_path):
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    be = FakeBackend(["SWEEP_REQUEST: notes/D2.3.json", REC("D2.3", 2.6)],
                     [EVAL := "=== JSON ===\n" + _json.dumps(
                         {"correctness": {"pass": True},
                          "performance": {"bwd_median_ms": 2.6, "device_median_ms": 2.6}}),
                      "(no csv)"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    # no sweep job was ever bought: the refusal is a prompt, not a GPU round
    assert not any("sweep_run.py" in str(c) for c in be.calls)
    # the student was told why, then delivered through the ordinary path
    assert "pre-writes no `sweep:` recipe" in be.prompts[1]
    assert rec is not None and rec.verdict == "KEEP"


def test_sweep_request_with_a_smuggled_variant_is_refused_without_a_job(tmp_path):
    from test_loop_driver import _ctx, FakeBackend, run_inner_iteration
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_milestone_with_sweep(SWEEP_RECIPE), encoding="utf-8")
    smuggle = _sweep_student(tmp_path, SWEEP_RECIPE, mangle=True)
    be = FakeBackend([smuggle, "EXHAUSTED: nothing more to deliver"], [])
    ctx.backend = be
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is None
    assert not any(c[0] == "gpu" for c in be.calls)          # not one job spent
    assert "does not hash-match the milestone" in be.prompts[1]
    assert (tmp_path / "notes" / "D1.1.json").is_file()      # the file was read, not erased


def test_sweep_request_runs_one_job_and_hands_the_winner_back(tmp_path):
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_milestone_with_sweep(SWEEP_RECIPE), encoding="utf-8")
    be = FakeBackend([_sweep_student(tmp_path, SWEEP_RECIPE), REC("D1.1", 2.6)],
                     [SWEEP_LOG,
                      '=== JSON ===\n' + _json.dumps(
                          {"correctness": {"pass": True},
                           "performance": {"bwd_median_ms": 2.6, "device_median_ms": 2.6}}),
                      "(no csv)"])
    ctx.backend = be
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    # exactly one sweep job, built from the milestone's recipe (kernel_file is
    # injected at SUBMIT time into the driver's own spec copy, never written
    # into the student's file)
    assert ("gpu", SWEEP_CMD_EXPECT) in be.calls
    order = [c for c in be.calls if c[0] == "gpu"]
    assert order[0] == ("gpu", SWEEP_CMD_EXPECT) and order[1][1] == "all"
    # the feedback carries the measured table and the winner instruction
    fb = be.prompts[1]
    assert "| `tile64` |" in fb and "+9.49%" in fb
    assert "**Winner: `tile64`**" in fb and "standard eval gate" in fb
    # the student's submitted file was neither rewritten nor removed
    kept = _json.loads((tmp_path / "notes" / "D1.1.json").read_text(encoding="utf-8"))
    assert kept == SWEEP_RECIPE
    # the delivery landed as an ordinary KEEP through the standard gate
    assert rec is not None and rec.verdict == "KEEP"


def test_sweep_winner_delivery_is_judged_by_the_standard_gate(tmp_path):
    """The lock the brief asks for: a sweep winner is landed as an ordinary
    decision — a correctness break is a hard REVERT, exactly as for any other
    delivery. The sweep channel adds no gate-free path."""
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_milestone_with_sweep(SWEEP_RECIPE), encoding="utf-8")
    be = FakeBackend([_sweep_student(tmp_path, SWEEP_RECIPE), REC("D1.1", 3.5)],
                     [SWEEP_LOG,
                      '=== JSON ===\n' + _json.dumps(
                          {"correctness": {"pass": False},
                           "performance": {"bwd_median_ms": 3.5, "device_median_ms": 3.5}}),
                      "(no csv)"])
    ctx.backend = be
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "REVERT" and rec.correctness is False
    assert "correctness FAIL" in rec.error


# --- task 3: slow-decision bookkeeping + REMOVE_CANDIDATES -----------------------

from peregrine_kernel.inner.trajectory import IterationRecord, load_records  # noqa: E402
from peregrine_kernel.loop import driver as _drv  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_slow_ledger():
    _drv._WP4_SLOW_LEDGER.clear()
    yield
    _drv._WP4_SLOW_LEDGER.clear()


def test_a_keep_slower_than_the_best_is_booked_and_rendered(tmp_path):
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    be = FakeBackend([REC("D2.3", 3.2)],
                     ['=== JSON ===\n' + _json.dumps(
                         {"correctness": {"pass": True},
                          "performance": {"bwd_median_ms": 3.2, "device_median_ms": 3.2}}),
                      "(no csv)"])
    be.last_cluster = "cl-a"
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 4, metric_before=2.9)
    # relaxed-revert stands: a perf regression is still a KEEP, never auto-reverted
    assert rec.verdict == "KEEP"
    assert rec.regression_vs_best_pct == pytest.approx(10.34, abs=0.01)
    # the in-memory per-decision account: (iter, delta_pct, workload)
    assert _drv.wp4_slow_ledger()["D2.3"] == [(4, rec.regression_vs_best_pct, "cl-a")]
    # the diagnose step renders it, with the levers priced for that decision
    levers = [{"kind": "lever", "name": "ring", "decisions": ["D2.3"]},
              {"kind": "variant", "name": "ring_lane", "decisions": ["D2.3"]}]
    block = _drv._wp4_remove_candidates_block(load_records(ctx.status_path), levers)
    assert "REMOVE_CANDIDATES: D2.3 Δ+10.34% @cl-a" in block
    assert "lever candidates: ring" in block
    assert "ring_lane" not in block          # a subset row is not a lever candidate
    assert "no automatic revert" in block    # the red line, said where it is read


def test_a_new_best_writes_no_regression_anywhere(tmp_path):
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    be = FakeBackend([REC("D2.3", 2.6)],
                     ['=== JSON ===\n' + _json.dumps(
                         {"correctness": {"pass": True},
                          "performance": {"bwd_median_ms": 2.6, "device_median_ms": 2.6}}),
                      "(no csv)"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 4, metric_before=2.9)
    assert rec.verdict == "KEEP" and rec.regression_vs_best_pct is None
    assert _drv.wp4_slow_ledger() == {}
    assert _drv._wp4_remove_candidates_block([rec], []) == ""


def test_a_regression_inside_the_band_is_not_booked(tmp_path):
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    be = FakeBackend([REC("D2.3", 2.93)],
                     ['=== JSON ===\n' + _json.dumps(
                         {"correctness": {"pass": True},
                          "performance": {"bwd_median_ms": 2.93, "device_median_ms": 2.93}}),
                      "(no csv)"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 4, metric_before=2.9)
    assert rec.verdict == "KEEP" and rec.regression_vs_best_pct is None     # 1.03% <= 1.5%
    assert _drv.wp4_slow_ledger() == {}


def test_the_legacy_threshold_is_the_only_threshold_when_pinned(monkeypatch):
    """The threshold reader pinned to its legacy value: at 1.5%, +2% books and
    +1% does not — the consumer honors whatever resolvable_diff returns."""
    monkeypatch.setattr(_drv, "resolvable_diff",
                        lambda records, gate_window: 0.015, raising=True)
    rec = IterationRecord(iter=7, decision_id="D1.1", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=3.06, metric_before=3.0)

    class _Ctx:
        gate_window = "device"
        status_path = "/nonexistent/status.jsonl"

    _drv._wp4_note_regression(_Ctx(), rec, 7, "cl-x")
    assert rec.regression_vs_best_pct == pytest.approx(2.0, abs=0.01)
    quiet = IterationRecord(iter=8, decision_id="D1.2", change="c", correctness=True,
                            verdict="KEEP", device_median_ms=3.03, metric_before=3.0)
    _drv._wp4_note_regression(_Ctx(), quiet, 8, "cl-x")
    assert quiet.regression_vs_best_pct is None


def test_resolvable_diff_is_imported_directly_from_wp1():
    """C1 integrated: the driver has no shim — its threshold reader IS WP1's
    parse.resolvable_diff, imported directly (not a module-local fallback)."""
    from peregrine_kernel.loop import parse as parse_mod
    assert _drv.resolvable_diff is parse_mod.resolvable_diff


def test_the_account_survives_a_driver_restart():
    """The diagnose step reads the records, not the in-memory ledger: a KEEP
    loaded back from an earlier process still renders."""
    rec = IterationRecord(iter=9, decision_id="D2.7", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=3.3, metric_before=3.0,
                          regression_vs_best_pct=10.0, cluster="cl-b")
    assert _drv._wp4_remove_candidates_block([rec], []) != ""
    assert "REMOVE_CANDIDATES: D2.7 Δ+10.00% @cl-b (no priced lever)" in \
        _drv._wp4_remove_candidates_block([rec], [])


def test_bookkeeping_never_touches_the_gate(tmp_path):
    """The red line, end to end: the booked KEEP keeps its verdict, its metric,
    and the tree — the record's only difference is the new optional field."""
    from test_loop_driver import REC, _ctx, FakeBackend, run_inner_iteration
    be = FakeBackend([REC("D2.3", 3.2)],
                     ['=== JSON ===\n' + _json.dumps(
                         {"correctness": {"pass": True},
                          "performance": {"bwd_median_ms": 3.2, "device_median_ms": 3.2}}),
                      "(no csv)"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 4, metric_before=2.9)
    stored = load_records(ctx.status_path)[0]
    assert stored.verdict == "KEEP" and stored.bwd_median_ms == 3.2
    assert stored.regression_vs_best_pct is not None
    assert ("reset", "sha") not in [c[:2] for c in be.calls if c[0] == "reset"]


def test_the_direct_import_holds_in_a_fresh_interpreter():
    """C1 integrated: in a fresh interpreter the driver's threshold reader is
    still parse's function — the same object, no import-time shim re-created
    against a nameless module."""
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    code = (
        "import peregrine_kernel.loop.parse as p\n"
        "from peregrine_kernel.loop import driver as d\n"
        "assert d.resolvable_diff is p.resolvable_diff\n"
        "print('direct-import-ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")})
    assert r.returncode == 0 and "direct-import-ok" in r.stdout, r.stderr
