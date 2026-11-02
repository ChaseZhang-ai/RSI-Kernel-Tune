"""The harvest-time probe / pricing recipe archive.

A KB entry cites the probe recipe that certified it, but the recipe lived in
the run workspace's notes/ — overwritten by the next certification — and
docs/kb-provenance/ ships no probes. The driver now copies the run's recipes
into evidence/probes/ at harvest time and the entry points at the archived
copy (or says explicitly "probe not archived").
"""
import json

from forgekernel.loop.driver import _archive_probe_recipes, run_kb_harvest
from forgekernel.loop.kb_lint import lint_kb_dir
from test_loop_driver import FakeBackend, REPO, _ctx

HARVEST_RECIPE_REPLY = ("wrote a recipe.\n"
                        "RECIPE_RECORDED: hw_kb/cutedsl/sm90/foo.md")


def _harvest_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 6
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


def _seed_notes(tmp_path):
    notes = tmp_path / "notes"
    (notes / "sub").mkdir(parents=True, exist_ok=True)
    (notes / "hard.py").write_text("# the winning probe\n", encoding="utf-8")
    (notes / "sub" / "abl.json").write_text('{"ablations": []}\n', encoding="utf-8")
    (notes / "levers.json").write_text('{"levers": []}\n', encoding="utf-8")
    # driver-seeded toolkits: re-vendored every run, nothing to preserve
    for name in ("lever_pricing_run.py", "orch_census_run.py", "stall_tools.py"):
        (notes / name).write_text("# vendored toolkit\n", encoding="utf-8")
    # non-recipe scratch: not a .py / .json recipe at all
    (notes / "scratch.md").write_text("prose\n", encoding="utf-8")


# --- the archive function -------------------------------------------------------

def test_archive_copies_recipes_and_skips_seeded_toolkits(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    _seed_notes(tmp_path)
    archived = _archive_probe_recipes(ctx)
    assert archived == ["notes/hard.py", "notes/levers.json", "notes/sub/abl.json"]
    dst = ctx.evidence_dir / "probes"
    assert (dst / "hard.py").read_text() == "# the winning probe\n"
    assert (dst / "sub" / "abl.json").is_file()
    assert (dst / "levers.json").is_file()
    assert not (dst / "lever_pricing_run.py").exists()      # seeded toolkit
    assert not (dst / "orch_census_run.py").exists()
    assert not (dst / "stall_tools.py").exists()
    assert not (dst / "scratch.md").exists()                # not a recipe suffix


def test_archive_is_empty_without_notes_or_evidence_channel(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    assert _archive_probe_recipes(ctx) == []                # no notes/ dir
    (tmp_path / "notes").mkdir()
    ctx.evidence_dir = None
    ctx.recon_path = None
    assert _archive_probe_recipes(ctx) == []                # no channel to bank in


def test_archive_overwrites_a_previous_copy_of_the_same_run(tmp_path):
    # a resumed run archives again: the copy is idempotent, not duplicated
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "p.py").write_text("v1\n", encoding="utf-8")
    assert _archive_probe_recipes(ctx) == ["notes/p.py"]
    (notes / "p.py").write_text("v2\n", encoding="utf-8")
    assert _archive_probe_recipes(ctx) == ["notes/p.py"]
    assert (ctx.evidence_dir / "probes" / "p.py").read_text() == "v2\n"
    assert len(list((ctx.evidence_dir / "probes").rglob("*"))) == 1


# --- the harvest hook -----------------------------------------------------------

def test_harvest_archives_and_hands_the_paths_to_the_teacher(tmp_path):
    be = FakeBackend([HARVEST_RECIPE_REPLY], [])
    ctx = _harvest_ctx(tmp_path, be)
    from forgekernel.loop.driver import ProbeBattle
    ctx.probe_battles["notes/hard.py"] = ProbeBattle(
        path="notes/hard.py", rounds=10, passed=True, pass_iter=17,
        verdict_line="PROBE PASS (max_diff=0.0)", log_tail="note: rung passed")
    _seed_notes(tmp_path)
    recipes = run_kb_harvest(ctx)
    assert recipes == ["hw_kb/cutedsl/sm90/foo.md"]
    assert (ctx.evidence_dir / "probes" / "hard.py").is_file()
    prompt = be.prompts[0]
    assert "Probe recipe archive" in prompt
    assert "evidence/probes/hard.py" in prompt


def test_harvest_archives_even_when_nothing_is_billed(tmp_path):
    # no kb-tripwire debt and no axis debt: no write-up turn — but the archive
    # still ran, because the pointer is written by later humans too
    be = FakeBackend([], [])
    ctx = _harvest_ctx(tmp_path, be)
    _seed_notes(tmp_path)
    assert run_kb_harvest(ctx) == []
    assert [c for c in be.calls if c[0] == "agent"] == []
    assert (ctx.evidence_dir / "probes" / "levers.json").is_file()


# --- the pointer's contract surfaces --------------------------------------------

def test_both_templates_require_an_explicit_probe_archive_line():
    for lang in ("cutedsl", "cuda"):
        template = (REPO / "hw_kb" / lang / "TEMPLATE.md").read_text(encoding="utf-8")
        assert "- Probe archive:" in template
        assert "probe not archived" in template
        assert "evidence/probes/<file>" in template


def test_harvest_prompts_demand_the_archive_line_and_render_clean():
    from forgekernel.loop.driver import _bindings, _meta_prompt, render_prompt
    for lang in ("cutedsl", "cuda"):
        ctx = _ctx(tmp_lang_path(lang), FakeBackend([], []))
        prompt_file = _meta_prompt(ctx, "kb_harvest_prompt.md")
        text = prompt_file.read_text(encoding="utf-8")
        assert "probe recipe archive" in text
        assert "Probe archive:" in text and "probe not archived" in text
        # render smoke: the template still renders with the runtime bindings
        rendered = render_prompt(prompt_file, _bindings(ctx), extra="")
        assert "{KB_DIR}" not in rendered


def tmp_lang_path(lang):
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp(prefix=f"kb_{lang}_"))
    (p / "sm90").mkdir()
    return p


def test_provenance_readme_documents_the_convention():
    text = (REPO / "docs" / "kb-provenance" / "README.md").read_text(encoding="utf-8")
    assert "probe recipe archive" in text.lower()
    assert "evidence/probes/" in text
    assert "probe not archived" in text
    # the historical-release note stays untouched
    assert "In this release the directory holds only this file." in text


def test_shipped_hw_kb_tree_stays_lint_clean():
    # TEMPLATE.md is lint-skipped by name, but the tree must stay green all the same
    assert lint_kb_dir(REPO / "hw_kb") == []
