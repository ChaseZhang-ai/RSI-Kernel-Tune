"""WP12: the deterministic KB-MENU synthesizer (loop/kbmenu.py), the driver's
injection glue, and the stale-KB-entry aggregation.

kbmenu is contract C7's single implementation: pure functions, no model calls,
same input -> same output. The snapshot test pins that; the robustness tests
pin the brief's "no empty-shell headings" rule; the pointer tests run the
matcher against the three REAL hw_kb trees and require every rendered quote
to be a verbatim substring of the index it points into (acceptance 3, over
every symptom row, not a sample); the stale-entry tests pin the via-KB
citation aggregation (task 2: two consecutive REVERT_PATCH reopens citing one
entry -> the warning).
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from forgekernel.loop import kbmenu
from forgekernel.loop.driver import (
    _wp12_kb_menu_block,
    _wp12_kb_menu_on,
    _wp12_readings,
    _wp12_resusage,
)
from test_loop_driver import FakeBackend, _ctx

# a KB tree without a symptom section (sm90 style) exercises the other parser
# branch; the two symptom-first trees exercise the primary one
_TREES = ("cuda/sm100", "cutedsl/sm100", "cutedsl/sm90")


def _tree_index(tree: str) -> str:
    return (REPO / "hw_kb" / tree / "index.md").read_text(encoding="utf-8")


def _rec(step=1, cite="", outcome="KEEP_PATCH", direction="dir", cls="R1"):
    return {"meta_step": step, "reopen_price": cite, "outcome": outcome,
            "patch_direction": direction, "cause_class": cls}


# --- determinism + snapshot ----------------------------------------------------

_READINGS = {"compute_sol_pct": 45.0, "memory_sol_pct": 50.0, "l2_hit_rate_pct": 35.0,
             "waves_per_sm": 0.7, "tensor_pct": 55.0, "dram_pct": 30.0,
             "spill_st_sectors": 64000, "spill_ld_sectors": 0, "bankconf_st": 1200000,
             "bankconf_ld": 0, "regs_per_thread": 168}
_RESUSAGE = {"regs": 168, "smem": 210000, "cstack": 8}
_LEDGER = [{"lever": "warp_split", "class": "harmful", "delta_pct": -11.63,
            "arch": "sm100", "op_family": "gemm", "condition": "negative add-one (-11.63%)"},
           {"lever": "stage3", "class": "faded", "delta_pct": 0.2,
            "arch": "sm100", "op_family": "other", "condition": "re-priced: add +9.00% -> add +0.20%: "
            "inside the noise band on this tree — the lever no longer pays"}]
_META = [_rec(1, "+3.1% via KB:wgedge", "REVERT_PATCH", "flip the wgmma edge"),
         _rec(2, "+2.2% via KB:wgedge", "REVERT_PATCH", "widen the tile"),
         _rec(3, "", "KEEP_PATCH", "two-pass readback")]

_GOLDEN_PREFIX = """## KB-MENU (deterministic retrieval aid — a pointer is not an answer, the menu is not a price)
KB-HINTS:
- spill/cstack>0 (spill_st=64000, spill_ld=0, cstack=8): hw_kb/cuda/sm100/lowtier_role_tmem_readout_spill_ladder.md — "**TMEM readback spills** in a `setmaxnreg.dec` role (`STL` / `LDL`, non-zero `spill stores`)"
"""


def test_snapshot_same_input_same_output():
    """Contract C7 determinism: the render is a pure lookup — identical inputs
    give byte-identical output, and the block opens with the pinned header +
    spill pointer (the snapshot half)."""
    idx = _tree_index("cuda/sm100")
    kwargs = dict(readings=dict(_READINGS), resusage=dict(_RESUSAGE),
                  ledger_rows=json.loads(json.dumps(_LEDGER)),
                  meta_records=[dict(m) for m in _META], index_text=idx,
                  kb_label="cuda/sm100", op="gemm")
    a = kbmenu.build_kb_menu(**kwargs)
    b = kbmenu.build_kb_menu(**kwargs)
    assert a == b
    assert a.startswith(_GOLDEN_PREFIX)
    # section order and the block's line discipline (contract C3: <=30 lines)
    lines = a.splitlines()
    assert len(lines) <= 30
    assert lines.index("KB-HINTS:") < lines.index("LEVER-MENU:") < lines.index("CALIBRATION:")


# --- robustness: no empty-shell headings ---------------------------------------

def test_all_empty_renders_nothing():
    assert kbmenu.build_kb_menu(readings={}, resusage={}, ledger_rows=[],
                                meta_records=[], index_text="", op="") == ""


def test_sections_shrink_independently():
    # ledger only: LEVER-MENU exclusions + the ledger-tail calibration line,
    # but no KB-HINTS shell
    menu = kbmenu.build_kb_menu(readings={}, ledger_rows=_LEDGER, meta_records=[],
                                index_text="", kb_label="cutedsl/sm100", op="gemm")
    assert "KB-HINTS:" not in menu
    assert "LEVER-MENU:" in menu and "LEDGER warp_split" in menu
    assert "- ledger tail (last 2): 1/2 rows faded/reversed" in menu
    # no reading fires: readings that miss every rule still allow the ledger
    quiet = {"compute_sol_pct": 99.0, "memory_sol_pct": 99.0, "l2_hit_rate_pct": 90.0,
             "waves_per_sm": 12.0, "dram_pct": 10.0}
    menu = kbmenu.build_kb_menu(readings=quiet, ledger_rows=_LEDGER, meta_records=[],
                                index_text="", kb_label="cutedsl/sm100", op="gemm")
    assert menu.splitlines()[1] == "LEVER-MENU:"
    # meta only: calibration alone
    menu = kbmenu.build_kb_menu(readings={}, meta_records=_META, index_text="")
    assert menu.startswith(kbmenu._HEADER) and menu.splitlines()[1] == "CALIBRATION:"


def test_corrupt_and_foreign_ledger_rows_ignored():
    rows = [{"lever": "ok", "class": "harmful", "delta_pct": -2.0, "arch": "sm90",
             "op_family": "gemm"},
            "not-a-dict", {"lever": None}, {"class": "harmful"},
            {"lever": "otherarch", "class": "harmful", "arch": "sm100", "op_family": "gemm"}]
    menu = kbmenu.build_kb_menu(readings={}, ledger_rows=rows, index_text="",
                                kb_label="cutedsl/sm90", op="gemm")
    assert "LEDGER ok" in menu and "otherarch" not in menu
    # same arch, other op family: still rendered (generic priors), after same-op ones
    menu = kbmenu.build_kb_menu(readings={}, ledger_rows=[{"lever": "ok2", "class": "harmful",
                                                           "arch": "sm90", "op_family": "conv"}],
                                index_text="", kb_label="cutedsl/sm90", op="gemm")
    assert "LEDGER ok2" in menu


# --- the rule table -------------------------------------------------------------

@pytest.mark.parametrize("readings,resusage,fired_names", [
    ({"spill_st_sectors": 1}, {}, {"spill"}),
    ({"spill_ld_sectors": 5}, {}, {"spill"}),
    ({}, {"cstack": 8}, {"spill"}),
    ({"regs_per_thread": 168, "spill_st_sectors": 3}, {}, {"spill", "regs-at-cap"}),
    ({"regs_per_thread": 60, "spill_st_sectors": 3}, {}, {"spill"}),   # low regs: no tier rule
    ({"regs_per_thread": 200}, {}, {"launch-bounds"}),                 # regs without spill: WP18 tier rule
    ({"bankconf_ld": 2}, {}, {"bankconf"}),
    ({"l2_hit_rate_pct": 35.0}, {}, {"l2-low"}),
    ({"l2_hit_rate_pct": 55.0}, {}, set()),
    ({"waves_per_sm": 0.4}, {}, {"waves-low", "wave-quantization"}),
    ({"waves_per_sm": 1.4}, {}, {"wave-quantization", "wave-tail"}),   # WP18: integer + tail
    ({"waves_per_sm": 2.0}, {}, {"wave-quantization"}),               # WP18: whole waves, no tail
    ({}, {"smem": 210000}, {"smem-cliff"}),
    ({}, {"smem": 100000}, set()),
    ({"compute_sol_pct": 40.0, "memory_sol_pct": 50.0}, {}, {"latency-bound"}),
    ({"compute_sol_pct": 40.0, "memory_sol_pct": 50.0, "tensor_pct": 50.0},
     {}, {"latency-bound", "unbound-attribution"}),
    ({"compute_sol_pct": 90.0, "memory_sol_pct": 50.0}, {}, set()),
    ({"compute_sol_pct": 75.0, "memory_sol_pct": 80.0},
     {}, {"warp-spec-pipeline", "tma-multicast"}),                     # WP18: both roles staged
    ({"dram_pct": 88.0}, {}, {"dram-high"}),
    ({"dram_pct": 60.0}, {}, set()),
    # --- WP18: the beyond-reference supply keys ---
    ({"memory_sol_pct": 85.0}, {}, {"tma-multicast"}),
    ({"l2_hit_rate_pct": 15.0}, {}, {"l2-low", "prefetch-lead"}),
    ({"regs_per_thread": 200}, {}, {"launch-bounds"}),                 # WP18: high regs, dry
    ({"regs_per_thread": 60}, {}, set()),                              # below the tier
    ({"dram_pct": 85.0, "l2_hit_rate_pct": 50.0},
     {}, {"dram-high", "l2-policy-window"}),
    ({"dram_pct": 85.0, "l2_hit_rate_pct": 30.0},
     {}, {"dram-high", "l2-low"}),          # streaming, no persisting competition
])
def test_rules_fire_on_the_documented_thresholds(readings, resusage, fired_names):
    fired = {r.name for r in kbmenu.RULES if r.when(readings, resusage)}
    assert fired == fired_names


def test_rule_table_is_extensible_and_labeled():
    assert len(kbmenu.RULES) >= 10  # the brief's initial 10-15
    for r in kbmenu.RULES:
        assert r.name and r.lever_hint


# --- pointers: verbatim quotes against the three REAL trees (acceptance 3) ------

def test_every_symptom_row_parses_with_a_real_quote():
    """Over EVERY parsed row of every tree (a superset of the brief's 10-per-tree
    sample): the entry resolves to a file that exists and the snippet is a
    verbatim substring of the raw row — the pointer never paraphrases."""
    for tree in _TREES:
        idx = _tree_index(tree)
        rows = kbmenu.parse_symptom_rows(idx)
        assert len(rows) >= 10, tree
        for row in rows:
            assert row.snippet, tree
            assert row.snippet in row.text, tree
            if row.entry:
                assert row.entry.endswith(".md"), tree
                assert (REPO / "hw_kb" / tree / row.entry).is_file(), tree


def test_rendered_pointers_quote_the_index_verbatim():
    """Strong readings fire every pointer rule; each rendered KB-HINTS quote
    must exist in that tree's index and point at a real entry file."""
    for tree in _TREES:
        idx = _tree_index(tree)
        menu = kbmenu.build_kb_menu(readings=_READINGS, resusage=_RESUSAGE,
                                    ledger_rows=[], meta_records=[],
                                    index_text=idx, kb_label=tree, op="gemm")
        assert "KB-HINTS:" in menu, tree
        for ln in menu.splitlines():
            if not ln.startswith("- ") or "hw_kb/" not in ln:
                continue
            left, sep, rest = ln[2:].partition(" — \"")
            assert sep, (tree, ln)
            quote = rest.removesuffix("\"")
            loc = left.partition(": ")[2]
            assert loc.startswith(f"hw_kb/{tree}/"), (tree, ln)
            entry = Path(loc).name
            assert (REPO / "hw_kb" / tree / entry).is_file(), (tree, ln)
            assert quote and quote in idx, (tree, ln)


def test_unknown_symptom_index_shrinks_the_hint_section():
    """An index with no matching row renders no hint for it — never a made-up
    pointer (the LEVER-MENU form still fires)."""
    idx = "| What you see | Look first at |\n|---|---|\n| something unrelated | [`x`](x.md): note |\n"
    menu = kbmenu.build_kb_menu(readings={"spill_st_sectors": 4}, index_text=idx,
                                kb_label="t/a", op="")
    assert "KB-HINTS:" not in menu
    assert "-> register-frame levers" in menu and "KB:" not in menu


# --- via-KB citation aggregation (task 2) ---------------------------------------

def test_kb_cite_name_normalisation():
    assert kbmenu.kb_cite_name("+3.1% via KB:wgma_edge") == "wgma_edge"
    assert kbmenu.kb_cite_name("+3.1% via KB:hw_kb/cuda/sm100/wgma.md") == "wgma"
    assert kbmenu.kb_cite_name("+3.1% via kb：wgma") == "wgma"     # fullwidth colon
    assert kbmenu.kb_cite_name("+3.1% via probe:notes/p.py#2") == ""
    assert kbmenu.kb_cite_name("") == ""


def test_stale_entry_two_consecutive_reverts():
    recs = [_rec(1, "+3% via KB:wgedge", "REVERT_PATCH"),
            _rec(2, "+2% via KB:wgedge", "REVERT_PATCH")]
    assert kbmenu.stale_kb_entries(recs) == ["wgedge"]


def test_not_stale_when_a_citing_reopen_pays_back():
    recs = [_rec(1, "+3% via KB:wgedge", "REVERT_PATCH"),
            _rec(2, "+2% via KB:wgedge", "KEEP_PATCH")]
    assert kbmenu.stale_kb_entries(recs) == []
    # and when only one reopen cites the entry so far (streak window unfilled)
    assert kbmenu.stale_kb_entries(recs[:1]) == []


def test_stale_reads_meta_step_order_and_ignores_uncited_records():
    # the REVERT_PATCH pair is the entry's LAST two citations even when other
    # outcomes (uncited records) sit between them in the file
    recs = [_rec(1, "+3% via KB:wgedge", "KEEP_PATCH"),
            _rec(2, "", "REVERT_PATCH", "uncited direction"),
            _rec(3, "+2% via KB:wgedge", "REVERT_PATCH"),
            _rec(4, "+1% via KB:wgedge", "REVERT_PATCH")]
    assert kbmenu.stale_kb_entries(recs) == ["wgedge"]
    # two entries, one stale one recovering
    recs.append(_rec(5, "+9% via KB:stage3", "KEEP_PATCH"))
    assert kbmenu.stale_kb_entries(recs) == ["wgedge"]


def test_stale_accepts_meta_record_objects():
    from forgekernel.meta_trajectory import MetaRecord
    recs = [MetaRecord(meta_step=1, cause_class="R1", deficient_locus="D1.1",
                       patch_class="structural", changed_entries=["D1.1"],
                       patch_direction="d", reopen_price="+3% via KB:wgedge",
                       outcome="REVERT_PATCH"),
            MetaRecord(meta_step=2, cause_class="R1", deficient_locus="D1.1",
                       patch_class="structural", changed_entries=["D1.1"],
                       patch_direction="d", reopen_price="+2% via KB:wgedge",
                       outcome="REVERT_PATCH")]
    assert kbmenu.stale_kb_entries(recs) == ["wgedge"]


def test_stale_warning_renders_into_calibration():
    menu = kbmenu.build_kb_menu(readings={}, meta_records=_META_with_stale(), index_text="")
    assert ("KB entry wgedge possibly stale (last 2 reopens citing it REVERTed) — "
            "schedule a harvest review") in menu


def _META_with_stale():
    return _META + [_rec(4, "+3.1% via KB:wgedge", "REVERT_PATCH", "retry the edge"),
                    _rec(5, "+2.2% via KB:wgedge", "REVERT_PATCH", "retry the edge again")]


# --- calibration aggregation ------------------------------------------------------

def test_reopen_payback_rate():
    menu = kbmenu.build_kb_menu(readings={}, meta_records=_META, index_text="")
    # _META carries one cited reopen pair that reverted, then a KEEP without a
    # price (a KEEP_PATCH without reopen_price is not a reopen)
    assert "- reopens (last 2): 0/2 paid back (KEEP_PATCH)" in menu


def test_reopen_window_is_the_last_five():
    recs = [_rec(i, f"+{i}% via KB:x{i}", "KEEP_PATCH" if i % 2 else "REVERT_PATCH")
            for i in range(1, 9)]                       # 8 reopens: only the last 5 render
    menu = kbmenu.build_kb_menu(readings={}, meta_records=recs, index_text="")
    assert "- reopens (last 5): 2/5 paid back" in menu  # 4..8 -> K,R,K on odd 5/7


def test_ledger_reversed_faded_rate():
    rows = [{"lever": f"l{i}", "class": "faded" if i % 2 else "harmful",
             "delta_pct": 0.1, "arch": "a", "op_family": "gemm",
             "condition": ("re-priced: the sign flipped" if i % 3 == 0 else "x")}
            for i in range(10)]
    menu = kbmenu.build_kb_menu(readings={}, ledger_rows=rows, index_text="", op="gemm")
    assert "- ledger tail (last 10): 7/10 rows faded/reversed" in menu  # 5 faded + 2 flipped


def test_revert_streak_directions_render_and_cap():
    recs = [_rec(1, "", "KEEP_PATCH", "kept direction"),
            _rec(2, "", "REVERT_PATCH", "shrink the tile M"),
            _rec(3, "", "REVERT_PATCH", "raise the stage count"),
            _rec(4, "", "REVERT_PATCH", "widen the N tile"),
            _rec(5, "", "REVERT_PATCH", "a fourth direction")]
    menu = kbmenu.build_kb_menu(readings={}, meta_records=recs, index_text="")
    line = next(ln for ln in menu.splitlines() if ln.startswith("- REVERT streak"))
    # capped at 3, newest first; the KEEP that ends the streak is not quoted
    assert line == ('- REVERT streak (3): "a fourth direction"; "widen the N tile"; '
                    '"raise the stage count" — dead ends; do not retry as-is')
    assert "kept direction" not in menu


# --- section caps (contract C3) ---------------------------------------------------

def test_section_caps():
    rows = [{"lever": f"l{i}", "class": "harmful", "delta_pct": -1.0, "arch": "a",
             "op_family": "gemm"} for i in range(9)]
    menu = kbmenu.build_kb_menu(readings=_READINGS, resusage=_RESUSAGE, ledger_rows=rows,
                                meta_records=[], index_text=_tree_index("cuda/sm100"),
                                kb_label="cuda/sm100", op="gemm")
    lines = menu.splitlines()
    assert len(lines) <= 30
    hints = lines[lines.index("KB-HINTS:"):lines.index("LEVER-MENU:") - 1]
    levers = lines[lines.index("LEVER-MENU:"):lines.index("CALIBRATION:") - 1]
    assert len(hints) - 1 <= 5
    assert len(levers) - 1 <= 10


# --- driver glue ------------------------------------------------------------------

def _glue_ctx(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    kb_root = tmp_path / "hw_kb" / "cutedsl"
    (kb_root / "sm100").mkdir(parents=True, exist_ok=True)
    ctx.kb_dir = kb_root
    ctx.kb_arch = "sm100"
    ctx.language = "cuda"
    return ctx


def test_glue_readings_take_the_newest_profiler(tmp_path):
    class R:
        iter, profiler = 0, {}
    old, new = R(), R()
    old.iter, old.profiler = 1, {"spill_st_sectors": 1}
    new.iter, new.profiler = 2, {"spill_st_sectors": 2, "waves_per_sm": 0.5}
    assert _wp12_readings([old, new])["spill_st_sectors"] == 2
    assert _wp12_readings([old]) == {"spill_st_sectors": 1}
    assert _wp12_readings([]) == {}


def test_glue_resusage_namespacing(tmp_path):
    readings = {"resusage_regs": 168, "resusage_smem": 210000, "resusage_cstack": 8,
                "resusage_kernel": "k", "spill_st_sectors": 1}
    ru = _wp12_resusage(readings)
    assert ru == {"regs": 168, "smem": 210000, "cstack": 8}


def test_menu_switch_default_and_card_override(tmp_path):
    ctx = _glue_ctx(tmp_path)
    assert _wp12_kb_menu_on(ctx) is True            # default, no loop.toml anywhere
    ctx.kb_menu = False                              # explicit value wins, never clobbered
    assert _wp12_kb_menu_on(ctx) is False
    ctx.kb_menu = None
    card = tmp_path / "loop.toml"
    card.write_text("[loop]\nkb_menu = false\n", encoding="utf-8")
    ctx.milestone = card                              # the card layer sits beside the milestone
    assert _wp12_kb_menu_on(ctx) is False
    card.write_text("[loop]\nkb_menu = true\n", encoding="utf-8")
    assert _wp12_kb_menu_on(ctx) is True
    card.write_text("[loop]\nkb_menu = \"yes\"\n", encoding="utf-8")   # bad type -> default
    assert _wp12_kb_menu_on(ctx) is True


def test_glue_block_empty_inputs_inject_nothing(tmp_path):
    ctx = _glue_ctx(tmp_path)
    assert _wp12_kb_menu_block(ctx, [], []) == ""


def test_glue_block_renders_with_real_tree(tmp_path):
    ctx = _glue_ctx(tmp_path)
    (ctx.kb_dir / "sm100" / "index.md").write_text(_tree_index("cuda/sm100"), encoding="utf-8")
    class R:
        pass
    rec = R()
    rec.profiler = dict(_READINGS)
    menu = _wp12_kb_menu_block(ctx, [rec], _META)
    assert menu.startswith("\n## KB-MENU") and menu.endswith("\n")
    assert "hw_kb/cuda/sm100/" in menu
    assert "KB entry wgedge possibly stale" in menu   # prior_records feed the calibration
    # recipe pointers resolve from the teacher's cwd (= the student workspace):
    # the glue rewrites them to the repo's absolute recipe directory
    assert str(REPO / "skills" / "probe-recipes") in menu
    assert "skills/probe-recipes/" not in menu.replace(str(REPO / "skills" / "probe-recipes"), "")


def test_recipe_dir_rewrite_is_deterministic_and_optional():
    readings = {"compute_sol_pct": 40.0, "memory_sol_pct": 50.0, "tensor_pct": 50.0,
                "waves_per_sm": 2.0, "dram_pct": 88.0}
    kw = dict(readings=readings, index_text="", op="")
    bare = kbmenu.build_kb_menu(**kw)
    assert "skills/probe-recipes/phase_ablation.py" in bare        # repo-relative default
    abs_ = kbmenu.build_kb_menu(**kw, recipe_dir="/repo/skills/probe-recipes")
    assert "/repo/skills/probe-recipes/phase_ablation.py" in abs_
    assert "skills/probe-recipes/" not in abs_.replace("/repo/skills/probe-recipes/", "")
    assert kbmenu.build_kb_menu(**kw, recipe_dir="/repo/skills/probe-recipes") == abs_


# --- prompt surface (task 2's harvest guidance ships with the aggregation) -------

def test_harvest_prompt_carries_the_three_wp12_sections():
    prompt = (REPO / "meta" / "kb_harvest_prompt.md").read_text(encoding="utf-8")
    assert "last_confirmed: job" in prompt
    assert '"class": "structural-dead-end"' in prompt
    assert "STALE-REVIEW" in prompt
    assert "falsified_ledger.jsonl" in prompt


def test_kbmenu_is_the_single_synthesizer():
    """Contract C7: kbmenu.py is the only injection synth — the driver imports
    it instead of re-implementing the render."""
    src = (REPO / "src" / "forgekernel" / "loop" / "driver.py").read_text(encoding="utf-8")
    assert "from forgekernel.loop import kbmenu" in src
    # and no second header string anywhere in the driver
    assert src.count("deterministic retrieval aid") <= 1
