"""The cross-run falsified-lever ledger: dead levers outlive their run.

A pricing table's negative readings (harmful / faded / coupled-negative /
null / noop) used to die with the run's evidence/ directory, and the next
run's teacher re-designed and re-priced the same dead ends. The ledger
appends them, at persist time, to a git-tracked file beside the per-language
KB roots; the diagnose prompt reads them back as PRIOR INFORMATION for the
teacher only — never a gate (the injection test pins the two states and the
arch filter, not any gate behavior, because no gate reads this file).
"""
import json

from rsi_kernel_tune.loop.driver import (
    _append_falsified_ledger,
    _falsified_ledger_block,
    _falsified_ledger_path,
    _persist_lever_table,
)
from test_loop_driver import FakeBackend, _ctx


def _wp3_ctx(tmp_path, be=None):
    """A _ctx whose kb_dir is the per-language root (hw_kb/<language>), the way
    the CLI builds it: the ledger then lands beside that root, at hw_kb/."""
    be = be or FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    kb_root = tmp_path / "hw_kb" / "cutedsl"
    (kb_root / "sm90").mkdir(parents=True, exist_ok=True)
    ctx.kb_dir = kb_root
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


def _row(name, cls, add=None, remove=None, decisions=()):
    return {"kind": "lever", "name": name, "cls": cls, "add_pct": add,
            "remove_pct": remove, "decisions": list(decisions)}


def _ledger(ctx):
    p = _falsified_ledger_path(ctx)
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            out.append(ln)          # the raw corrupt line, so tests can see it
    return out


# --- write side -----------------------------------------------------------------

def test_persist_appends_only_the_dead_levers(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    table = [_row("regsplit", "harmful", add=-11.63, decisions=["D0.4"]),
             _row("ring_lane", "null", add=0.2),
             _row("swizzle", "noop", add=None),
             _row("wide_ldg", "independent", add=25.59),
             {"kind": "stack", "all_pct": None, "noise_pct": 1.14}]
    _persist_lever_table(ctx, table, "bestsha", 885353, spec_levers=[])
    recs = _ledger(ctx)
    assert [r["lever"] for r in recs] == ["regsplit", "ring_lane", "swizzle"]
    assert [r["class"] for r in recs] == ["harmful", "null", "noop"]
    for r in recs:
        assert r["arch"] == "sm90" and r["op_family"] == "demo"
        assert r["run"] == tmp_path.name and r["date"] == "1970-01-01"
    assert recs[0]["delta_pct"] == -11.63 and "D0.4" in recs[0]["condition"]
    assert "noise" in recs[1]["condition"]
    # the pricing path itself is unchanged: the table is still banked
    banked = [json.loads(ln) for ln in
              (ctx.evidence_dir / "lever_prices.jsonl").read_text().splitlines()]
    assert len(banked) == 1 and banked[0]["job"] == 885353


def test_same_run_and_lever_is_not_double_booked(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    table = [_row("regsplit", "harmful", add=-11.63)]
    _persist_lever_table(ctx, table, "sha1", 1)
    _persist_lever_table(ctx, table, "sha2", 2)      # re-priced later in the run
    _persist_lever_table(ctx, [_row("other", "null", add=0.1)], "sha3", 3)
    recs = _ledger(ctx)
    assert [r["lever"] for r in recs] == ["regsplit", "other"]
    assert recs[0]["run"] == recs[1]["run"] == tmp_path.name


def test_coupled_row_that_costs_alone_is_coupled_negative(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    # a coupled row whose add leg reads negative: hurts alone, all-on needs it
    _persist_lever_table(ctx, [_row("pack_a", "coupled", add=-0.9, remove=-8.0)], "s", 1)
    # a coupled row at ≈ 0 alone is the normal companion case: not a dead lever
    _persist_lever_table(ctx, [_row("pack_b", "coupled", add=0.2, remove=-8.0)], "s", 1)
    recs = _ledger(ctx)
    assert [r["class"] for r in recs] == ["coupled-negative"]
    assert "package-only" in recs[0]["condition"]


def test_reprice_table_records_faded_and_reversed(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    # the teacher's table: two levers paid on the first tree, one confirmed later
    first = [_row("was_big", "independent", add=25.59, decisions=["D0.4"]),
             _row("flipped", "independent", add=9.0),
             _row("held", "independent", add=12.0),
             {"kind": "stack", "noise_pct": 1.14}]
    _persist_lever_table(ctx, first, "sha1", 100)
    assert _ledger(ctx) == []          # a green table books nothing
    # the re-price on the later best: one faded into the noise band, one flipped sign
    second = [_row("was_big", "null", add=0.4),
              _row("flipped", "harmful", add=-11.63),
              _row("held", "independent", add=24.1),
              {"kind": "stack", "noise_pct": 1.14}]
    _persist_lever_table(ctx, second, "sha2", 101, reprice=True)
    recs = _ledger(ctx)
    assert [r["lever"] for r in recs] == ["was_big", "flipped"]
    assert [r["class"] for r in recs] == ["faded", "harmful"]
    assert "add +25.59% -> add +0.40%" in recs[0]["condition"]
    assert "add +9.00% -> add -11.63%" in recs[1]["condition"]


def test_landed_lever_is_ledgered_by_its_remove_leg(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    _persist_lever_table(ctx, [_row("landed", "independent", add=9.0)], "sha1", 1)
    # a landed lever is re-priced by taking it out: add None, remove carries the worth
    row = {"kind": "lever", "name": "landed", "cls": "null", "add_pct": None,
           "remove_pct": 0.3, "decisions": []}
    _persist_lever_table(ctx, [row], "sha2", 2, reprice=True)
    recs = _ledger(ctx)
    assert recs[0]["class"] == "faded" and recs[0]["delta_pct"] == -0.3


# --- consume side ---------------------------------------------------------------

def test_ledger_block_omitted_when_missing_or_empty(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    assert _falsified_ledger_block(ctx) == ""          # no file
    _falsified_ledger_path(ctx).write_text("\n", encoding="utf-8")
    assert _falsified_ledger_block(ctx) == ""          # only blank lines


def test_ledger_block_filters_arch_and_prefers_same_op(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    p = _falsified_ledger_path(ctx)
    rows = [
        {"op_family": "demo", "arch": "sm90", "lever": "same_op", "class": "harmful",
         "delta_pct": -3.0, "condition": "costs"},
        {"op_family": "generic", "arch": "sm90", "lever": "generic_one", "class": "null",
         "delta_pct": 0.1, "condition": "flat"},
        {"op_family": "other_op", "arch": "sm90", "lever": "other_op_one", "class": "noop",
         "delta_pct": None, "condition": "cubin same"},
        {"op_family": "demo", "arch": "sm100", "lever": "wrong_arch", "class": "harmful",
         "delta_pct": -3.0, "condition": "costs"},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    block = _falsified_ledger_block(ctx)
    assert "## Falsified-lever ledger" in block
    assert "LEDGER same_op harmful Δ-3.00% costs" in block
    assert "LEDGER generic_one null Δ+0.10% flat" in block
    assert "LEDGER other_op_one noop Δn/a% cubin same" in block
    assert "wrong_arch" not in block                     # arch filter
    assert block.index("same_op") < block.index("generic_one") < block.index("other_op_one")
    assert "PRIOR INFORMATION" in block and "no gate" in block


def test_ledger_block_caps_at_twenty_lines(tmp_path):
    ctx = _wp3_ctx(tmp_path)
    rows = [{"op_family": "demo", "arch": "sm90", "lever": f"lev_{i:02d}",
             "class": "null", "delta_pct": 0.1, "condition": "flat"}
            for i in range(25)]
    _falsified_ledger_path(ctx).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    block = _falsified_ledger_block(ctx)
    assert block.count("\n- LEDGER ") == 20
    assert "lev_19" in block and "lev_20" not in block


def test_append_is_silent_on_an_unreadable_ledger(tmp_path):
    # a corrupt line must not take pricing down, and must not reset the dedup
    ctx = _wp3_ctx(tmp_path)
    p = _falsified_ledger_path(ctx)
    p.write_text("{not json}\n", encoding="utf-8")
    _append_falsified_ledger(ctx, [_row("a", "null", add=0.1)], best_ref="s", job=1)
    recs = [r for r in _ledger(ctx) if isinstance(r, dict)]
    assert len(recs) == 1 and recs[0]["lever"] == "a"


# --- end to end: the diagnose prompt --------------------------------------------

_DIAG = "cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision"


def _patch(tmp_path):
    from test_loop_driver import _patching
    return _patching(tmp_path, "\n### D6 added\nx\n",
                     "patch_class: additive\nchanged_entries: []\n")


def test_diagnose_prompt_carries_the_ledger_as_prior_only(tmp_path):
    from rsi_kernel_tune.loop.driver import run_meta_step
    be = FakeBackend([_DIAG, _patch(tmp_path)], [])
    ctx = _wp3_ctx(tmp_path, be)
    ctx.milestone.write_text(
        ctx.milestone.read_text(encoding="utf-8"), encoding="utf-8")
    _falsified_ledger_path(ctx).write_text(json.dumps({
        "op_family": "demo", "arch": "sm90", "lever": "regsplit", "class": "harmful",
        "delta_pct": -11.63, "condition": "negative add-one, all-on does not miss it",
        "run": "earlier", "date": "2026-01-01"}) + "\n", encoding="utf-8")
    run_meta_step(ctx)
    prompt = be.prompts[0]           # the first agent call is the diagnose step
    assert "LEDGER regsplit harmful Δ-11.63%" in prompt
    assert "PRIOR INFORMATION" in prompt


def test_diagnose_prompt_omits_the_section_without_a_ledger(tmp_path):
    from rsi_kernel_tune.loop.driver import run_meta_step
    be = FakeBackend([_DIAG, _patch(tmp_path)], [])
    ctx = _wp3_ctx(tmp_path, be)
    ctx.milestone.write_text(
        ctx.milestone.read_text(encoding="utf-8"), encoding="utf-8")
    run_meta_step(ctx)
    assert "Falsified-lever ledger" not in be.prompts[0]
