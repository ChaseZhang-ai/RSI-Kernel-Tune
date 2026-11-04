"""R2-structural: the teacher-side ORACLE-external skeleton channel (WP6 task 2).

By definition R2 covered decisions "present in ORACLE", so the system could never
propose a skeleton the reference does not contain — its ceiling was the reference's
architecture. This opens exactly that on the teacher side, gated the way every
expensive action here is gated: an ORACLE-external skeleton is accepted only with
a measured price produced in this run (``reopen_price: <delta> via
probe:<path>#<n>``; an unpriced one is bounced once, then refused — the
UNPRICED_REOPEN pattern). A lever, a KB entry or an inference does not price it.
The landing path is the ordinary milestone patch; the student prompt is untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    _MILESTONE, FakeBackend, _ctx, _patching,
)

from peregrine_kernel.loop.driver import (  # noqa: E402
    _is_r2_structural, _verify_r2_structural_price, run_meta_step,
)

_R2S = ("cause_class: R2\ndeficient_locus: D2.3\n"
        "r2_structural: R2-structural\n"
        "patch_direction: extend the decision graph with a two-pass skeleton\n")
_PATCH = "patch_class: structural\nchanged_entries: [D2.3]\n"


def _tagged_ctx(tmp_path, be):
    ctx = _ctx(tmp_path, be)
    ctx.reopen_requires_price = True
    return ctx


def test_tag_parse_is_literal_and_lenient_about_punctuation():
    assert _is_r2_structural("r2_structural: R2-structural\n")
    assert _is_r2_structural("r2_structural：R2-structural\n")          # fullwidth colon
    assert _is_r2_structural("r2_structural: r2-STRUCTURAL\n")          # case
    assert not _is_r2_structural("r2_structural:\n")                    # left empty
    assert not _is_r2_structural("r2_structural: <write the literal tag...>")  # template verbatim
    assert not _is_r2_structural("cause_class: R2\npatch_direction: add D2.4\n")  # untagged


def test_priced_r2_structural_passes_and_lands_as_an_ordinary_patch(tmp_path):
    """The channel open for business: unpriced -> bounced once -> probe measured in
    this run -> the priced proposal lands as a normal structural milestone patch
    (outcome PENDING, standard transaction; the student prompt is untouched)."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "skel.py").write_text("# probe\n", encoding="utf-8")
    priced = _R2S + "reopen_price: +6.1% via probe:notes/skel.py#1\n"
    be = FakeBackend([_R2S, "PROBE_REQUEST: notes/skel.py", priced,
                      _patching(tmp_path, "\n### D2.3 two-pass skeleton\nx\n", _PATCH)],
                     ["PROBE PASS max_diff=1.061 — skeleton variant / base, same job\n"])
    ctx = _tagged_ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural" and res.changed_entries == ["D2.3"]
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert rec.outcome == "PENDING" and rec.cause_class == "R2"
    assert "probe:notes/skel.py#1" in rec.reopen_price
    # the student line is untouched: no student prompt was ever sent
    assert not any("student" in p.lower() and "decision_id:" in p for p in be.prompts)


def test_unpriced_r2_structural_is_bounced_once_then_refused(tmp_path):
    """A tagged proposal without a price: one bounce naming the probe route, then
    refusal — the landed patch is withdrawn, the step recorded UNPRICED_REOPEN."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    be = FakeBackend([_R2S, _R2S,
                      _patching(tmp_path, "\n### D2.3 unpriced skeleton\nx\n", _PATCH)], [])
    ctx = _tagged_ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    assert res.patch_class == "unpriced-reopen"
    assert "An ORACLE-external skeleton must be priced first" in be.prompts[1]
    assert ctx.milestone.read_text(encoding="utf-8") == before      # patch withdrawn
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert rec.outcome == "UNPRICED_REOPEN" and rec.patch_class == "unpriced-reopen"


def test_a_lever_or_kb_price_does_not_price_an_r2_structural(tmp_path):
    """Red line: the acceptance condition is a THIS-RUN measured probe — no
    relaxation to lever tables or KB inference (outside the ORACLE there is no
    reference reading to lean on)."""
    lever_cited = _R2S + "reopen_price: +8.6% via lever:ring\n"
    be = FakeBackend([lever_cited, lever_cited,
                      _patching(tmp_path, "\n### D2.3 x\n", _PATCH)], [])
    ctx = _tagged_ctx(tmp_path, be)
    assert run_meta_step(ctx).patch_class == "unpriced-reopen"
    assert "prices only a probe measured in this run" in be.prompts[1]
    kb_cited = _R2S + "reopen_price: +4% via KB:some_recipe\n"
    be2 = FakeBackend([kb_cited, kb_cited,
                       _patching(tmp_path, "\n### D2.3 x\n", _PATCH)], [])
    ctx2 = _tagged_ctx(tmp_path, be2)
    assert run_meta_step(ctx2).patch_class == "unpriced-reopen"
    assert "prices only a probe measured in this run" in be2.prompts[1]


def test_citing_a_probe_that_never_ran_is_no_price(tmp_path):
    """The probe path must carry a reading from this step: a hallucinated or
    unrun probe is bounced like any unpriced proposal."""
    cited = _R2S + "reopen_price: +6.1% via probe:notes/ghost.py#1\n"
    be = FakeBackend([cited, cited,
                      _patching(tmp_path, "\n### D2.3 x\n", _PATCH)], [])
    ctx = _tagged_ctx(tmp_path, be)
    assert run_meta_step(ctx).patch_class == "unpriced-reopen"
    assert "was not run this step" in be.prompts[1]


def test_evidence_file_gate_is_the_bounce_readings_are_the_provenance(tmp_path):
    """The disk requirement is enforced where it is checkable — the bounce gate,
    before the probe workspace reset removes the file. A proposal whose probe ran
    but whose file never landed is bounced with the file named; on insistence the
    in-run readings decide (the post-loop check does not re-demand a file the
    reset already archived away). With the file present there is no bounce."""
    cited = _R2S + "reopen_price: +6.1% via probe:notes/skel.py#1\n"
    log = ["PROBE PASS max_diff=1.061 — skeleton variant / base\n"]
    patch = _patching(tmp_path, "\n### D2.3 x\n", _PATCH)
    # the probe RAN first (a reading exists) but the file never landed on disk
    be = FakeBackend(["PROBE_REQUEST: notes/skel.py", cited, cited, patch], log)
    ctx = _tagged_ctx(tmp_path, be)
    assert run_meta_step(ctx).patch_class == "structural"   # insistence: readings decide
    # bounced with the file named (prompts: diagnose -> probe feedback -> bounce)
    assert "evidence file notes/skel.py" in be.prompts[2]
    # with the file present there is no bounce at all
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "skel.py").write_text("# probe\n", encoding="utf-8")
    be2 = FakeBackend(["PROBE_REQUEST: notes/skel.py", cited, patch], log)
    ctx2 = _tagged_ctx(tmp_path, be2)
    assert run_meta_step(ctx2).patch_class == "structural"
    assert len(be2.prompts) == 3                            # diagnose + probe feedback + patch; no bounce


def test_additive_landing_of_a_tagged_proposal_still_needs_the_price(tmp_path):
    """The strict price binds the PROPOSAL, not the transaction class: an
    ORACLE-external skeleton landing as an additive rung extension (no reopen) is
    refused without its in-run measured price all the same (red line: acceptance
    requires the price; the standard gate catching failure later is not a price)."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    additive = "patch_class: additive\nchanged_entries: [D2.4]\n"
    be = FakeBackend([_R2S, _R2S,
                      _patching(tmp_path, "\n### D2.4 new rung\nx\n", additive)], [])
    ctx = _tagged_ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    assert run_meta_step(ctx).patch_class == "unpriced-reopen"
    assert ctx.milestone.read_text(encoding="utf-8") == before
    assert load_meta_records(ctx.meta_status_path)[-1].outcome == "UNPRICED_REOPEN"


def test_tag_under_a_misfiled_cause_class_is_still_bounced_first(tmp_path):
    """The bounce binds to the `r2_structural:` tag itself, not to the cause
    class: a teacher that mis-files the proposal under a non-R1/non-R2 class
    (here G2, additive, so neither the R1 gate nor the structural three-source
    gate is in play) must still be told HOW to price it once — only the insisted
    second attempt hits the patch point's outright refusal."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    g2 = ("cause_class: G2\ndeficient_locus: D2.3\n"
          "r2_structural: R2-structural\n"
          "patch_direction: split and extend with a two-pass skeleton\n")
    additive = "patch_class: additive\nchanged_entries: [D2.4]\n"
    be = FakeBackend([g2, g2,
                      _patching(tmp_path, "\n### D2.4 new rung\nx\n", additive)], [])
    ctx = _tagged_ctx(tmp_path, be)
    before = ctx.milestone.read_text(encoding="utf-8")
    res = run_meta_step(ctx)
    # first diagnosis: bounced with the route named (prompts: diagnose -> bounce -> patch)
    assert "An ORACLE-external skeleton must be priced first" in be.prompts[1]
    # second, insisted: the patch point refuses — milestone restored, recorded
    assert res.patch_class == "unpriced-reopen"
    assert ctx.milestone.read_text(encoding="utf-8") == before
    rec = load_meta_records(ctx.meta_status_path)[-1]
    assert rec.outcome == "UNPRICED_REOPEN" and rec.cause_class == "G2"


def test_untagged_r2_with_a_structural_patch_keeps_the_old_rule(tmp_path):
    """Regression: an ordinary R2 (decision present in ORACLE) landing structural
    is governed by the existing three-source gate — a KB-cited price lands it
    exactly as before this channel existed; no tag, no strict rule."""
    from peregrine_kernel.meta_trajectory import load_meta_records
    (tmp_path / "sm90").mkdir(exist_ok=True)
    (tmp_path / "sm90" / "skel_recipe.md").write_text(
        "# kb\n- ✅ Verified: job 870533 `PROBE PASS`\n", encoding="utf-8")
    r2 = ("cause_class: R2\ndeficient_locus: D2.3\n"
          "patch_direction: add the missing decision\n"
          "reopen_price: +4% via KB:skel_recipe\n")
    be = FakeBackend([r2, _patching(tmp_path, "\n### D2.3 added\nx\n", _PATCH)], [])
    ctx = _tagged_ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "structural"
    assert load_meta_records(ctx.meta_status_path)[-1].outcome == "PENDING"


def test_verify_r2_structural_price_units(tmp_path):
    """The strict verifier in isolation: probe-with-reading is the one accepted
    source; lever / KB / ablation-name / no-source / no-cite all read as refusals
    with the route named; the disk check is a bounce-time extra."""
    ctx = _tagged_ctx(tmp_path, FakeBackend([], []))
    readings = [{"path": "notes/skel.py", "round": 1, "contract": 1, "verdict": "PASS",
                 "reading": "1.061", "note": ""}]
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "skel.py").write_text("# probe\n", encoding="utf-8")

    def why(c, **kw):
        return _verify_r2_structural_price(c, readings, ctx, ["notes/skel.py"], **kw)
    assert why("+6.1% via probe:notes/skel.py#1") is None
    assert "no reopen_price line" in why("")
    assert "names no probe:<path>#<n>" in why("+5% (derived from byte arithmetic)")
    assert "prices only a probe" in why("+8.6% via lever:ring")
    assert "prices only a probe" in why("+4% via KB:skel_recipe")
    assert "prices only a probe" in why("+2.2% via ablation:no_ring")
    assert "was not run this step" in why("+6.1% via probe:notes/other.py#1")
    # disk variants
    assert why("+6.1% via probe:notes/skel.py#1", check_disk=True) is None
    missing = _verify_r2_structural_price("+6.1% via probe:notes/ghost.py#1",
                                          [{"path": "notes/ghost.py", "round": 1,
                                            "contract": "1", "verdict": "PASS",
                                            "reading": "1.0", "note": ""}],
                                          ctx, ["notes/ghost.py"], check_disk=True)
    assert "not in this" in missing and "notes/ or evidence/" in missing
    # an evidence dir counts too
    ev = tmp_path / "evidence"
    ev.mkdir(exist_ok=True)
    (ev / "skel.py").write_text("# probe\n", encoding="utf-8")
    (tmp_path / "notes" / "skel.py").unlink()
    ctx.evidence_dir = ev
    assert _verify_r2_structural_price("+6.1% via probe:notes/skel.py#1", readings,
                                       ctx, ["notes/skel.py"], check_disk=True) is None
