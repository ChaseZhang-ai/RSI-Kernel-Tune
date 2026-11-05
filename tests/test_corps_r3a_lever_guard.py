"""Regression test for the R3A audit wave: teacher levers.json robustness.

BUG-CH-B-01 (P1): the levers normalization loop in _run_lever_pricing was the
only one of the three sibling lists (levers / variants / ablations) without an
isinstance guard, so one non-object entry in the teacher's notes/levers.json
killed the whole run with an AttributeError at a seam with no net. A non-dict
top-level spec (a bare array) died the same way on spec.get().
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from test_loop_driver import (FakeBackend, _MILESTONE, _ctx, _patching)

from peregrine_kernel.loop import driver
from peregrine_kernel.meta_trajectory import load_meta_records

_MILESTONE_PATCH = "\n### D0.9 loosened\nx\n"
_PATCH_REPLY = "patch_class: structural\nchanged_entries: [D0.9]\n"
_DIAG = ("cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision\n"
         "reachability: floor_opt=1.20 ms vs target=1.40 ms -> OPEN")

_STALL_LOG = (
    "[sanity] kernel=main_kernel\n"
    "=== STALL LEDGER BEGIN ===\n"
    "[ledger] keying=source-line (81% of samples line-resolved)\n"
    "   900 45.00% (cum  45.0%)  kernel.py:101  HMMA.16816.F32 | long_scoreboard:70%\n"
    "=== STALL LEDGER END ===\n"
    "STALL-OK kernel=main_kernel samples=2000 rows=1 coverage=81.0\n")

_PRICE_LOG = (
    "PRICE variants: base +ring all all-ring\n"
    "PRICE ROUND 0 base pass=True device=3.40 vs_ref_device=0.605\n"
    "LEVER ring add=+8.56% remove=-8.10% add_ms=3.135 remove_ms=3.21 class=independent "
    "spin_add=880319 spin_all_minus=996231 cubin_same_as_base=False\n"
    "STACK all=+13.10% sum_singles=+8.56% coupling=0.00%\n"
    "PRICE noise_floor=0.50% metric=vs_ref_device\n"
    "PRICE-OK levers=1 rounds=3 base_ms=3.4052 all_ms=2.96 spin_base=1125779 spin_all=700000 "
    "tensor_base=37.85 tensor_all=44.0\n")


def _lever_kernel(tmp_path):
    """The best tree's kernel file, carrying the `old` the fixtures' lever edits."""
    kf = tmp_path / "flash_attn_dsl" / "flash_bwd.py"
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text("a\nc\ne\n", encoding="utf-8")
    return kf


def _pricing_ctx(tmp_path, spec_text):
    """A stall-priced plateau whose lever-design turn writes ``spec_text`` as the
    teacher's notes/levers.json (the rest of the entry scripts a clean run)."""
    def design(prompt):
        (tmp_path / "notes").mkdir(exist_ok=True)
        _lever_kernel(tmp_path)
        (tmp_path / "notes" / "levers.json").write_text(spec_text, encoding="utf-8")
        return "LEVERS: written"

    be = FakeBackend([design, _DIAG,
                      _patching(tmp_path, _MILESTONE_PATCH, _PATCH_REPLY)],
                     [_STALL_LOG, _PRICE_LOG])
    ctx = _ctx(tmp_path, be)
    ctx.stall_pricing = True
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    ctx.lever_pricing_rounds = 1
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


def test_a_non_object_lever_entry_is_dropped_not_fatal(tmp_path):
    """One bare-string entry among the levers must be skipped the way the
    variants/ablations siblings skip theirs — the run prices the rest."""
    ctx = _pricing_ctx(tmp_path, json.dumps({"levers": [
        "tile_64",
        {"name": "ring", "edits": [{"old": "a", "new": "b"}]}]}))
    driver.run_meta_step(ctx)   # no AttributeError escapes the pricing round
    spec = json.loads((tmp_path / "notes" / "levers.json").read_text(encoding="utf-8"))
    assert [lv["name"] for lv in spec["levers"]] == ["ring"]
    recs = load_meta_records(ctx.meta_status_path)
    names = [lv["name"] for lv in recs[-1].lever_prices if lv.get("kind") == "lever"]
    assert "ring" in names and "tile_64" not in names


def test_a_bare_array_levers_json_is_skipped_not_fatal(tmp_path):
    """levers.json holding a JSON array (not an object) must log and diagnose
    without a price table, not die on spec.get()."""
    ctx = _pricing_ctx(tmp_path, '["tile_64", "stage_2"]')
    driver.run_meta_step(ctx)   # no AttributeError escapes the pricing round
    diag = ctx.backend.prompts[1]
    assert "## Pricing table" not in diag
