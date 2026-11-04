"""WP18: knowledge closure & beyond-reference supply.

The KB's "use it -> measure it -> keep or kill it" loop, closed mechanically:

- kb_used — the student names the KB entries an iteration leaned on; the driver
  banks the per-entry KEEP/REVERT ledger (acceptance 1: quote -> verdict flow);
- the bandwidth-recipe bounce — a BLOCKED stop request with no measured
  bandwidth reading behind it is bounced once (acceptance 2), and NOT bounced
  when a reading exists or the reachability refusal is already failing it;
- the KB-MENU reads the second-newest reading too (acceptance 3: a prev-only
  fire renders its marker line);
- the scheduled candidate-queue batch certification (acceptance 4: N=4 in ONE
  job, one verdict per candidate, a failing candidate does not contaminate the
  others, batch=1 degrades to the per-candidate path, batch=0 disables);
- the appended kbmenu rules point at real rows on all three trees (acceptance 5,
  three per tree), the hit-telemetry row format is pinned (acceptance 6), the
  survey's CANDIDATE-DRAFT lines land deduped (dup not landed);
- the artifact fill turns skeletons into anchored priors (acceptance 7: the
  skeleton -> filled diff), the recon brief renders its KB binding (acceptance 8).
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from peregrine_kernel.inner.trajectory import IterationRecord
from peregrine_kernel.loop import driver, kbmenu
from peregrine_kernel.loop.driver import (
    _is_wp18_batch_request,
    _meta_prompt,
    _bindings,
    _wp18_bw_recipe_missing,
    _wp18_candidates_rows,
    _wp18_kb_cert_duty_block,
    _wp18_kb_usage_flow,
    _wp18_kb_usage_note,
    _wp18_kb_usage_path,
    _wp18_kb_usage_summary,
    _wp18_kbmenu_hits_summary,
    _wp18_land_candidate_drafts,
    _wp18_parse_kb_cert_log,
    _wp18_recent_readings,
    _wp18_stamp_kb_cert,
    _wp18_stamp_priced_null,
    _wp18_write_lever_shape_matrix,
    render_prompt,
    run_meta_step,
)
from test_loop_driver import FakeBackend, _ctx, _patching

sys.path.insert(0, str(REPO / "scripts"))

# a queue fixture: the header row + four open candidates + one landed one
_QUEUE = """# KB candidate queue (pending certification)

| Candidate entry | Tree(s) | Priority | Basis | Status |
|---|---|---|---|---|
| `wave_tail_governance` — price the tail-wave fix | `cutedsl/sm90` | High | waves cliff unmeasured | TODO |
| `tma_multicast_fanout` — cluster multicast operand load | `cutedsl/sm100` | High | fan-out unmeasured | TODO |
| `clc_persistent_claim` — CLC claims in a persistent grid | `cuda/sm100` | Medium | composition unmeasured | TODO |
| `l2_window_persistence` — persisting window carve-out | `cuda/sm100` | Medium | precondition unmeasured | TODO |
| `already_landed` — certified last run | `cutedsl/sm90` | Low | history | DONE |
"""


def _queue_ctx(tmp_path, backend, **kw):
    """A ctx whose kb_dir sits under tmp/hw_kb/<lang> (the production shape:
    the candidate queue lives at kb_dir/../CANDIDATES.md)."""
    ctx = _ctx(tmp_path, backend, **kw)
    ctx.kb_dir = tmp_path / "hw_kb" / "cutedsl"
    (ctx.kb_dir / "sm90").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hw_kb" / "CANDIDATES.md").write_text(_QUEUE, encoding="utf-8")
    return ctx


class _JobFakeBackend(FakeBackend):
    """A FakeBackend that reports a job id, the way a real runner does (the
    BATCH stamp cites it)."""
    last_job_id = "4242"


# --- task 1: kb_used ledger ------------------------------------------------------


def test_kb_used_quotes_flow_to_the_ledger(tmp_path):
    """The student quotes entries; the verdict flow banks per-entry KEEP/REVERT
    rows (acceptance 1). Names normalise (paths / .md stripped), dedup, order."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    turn = ("decision_id: D1\nchange: ring\nfiles_touched: k.py\n"
            "kb_used: mbarrier_ring_rolled_loop.md, hw_kb/cutedsl/sm90/tma_refill.md\n"
            "kb_used: mbarrier_ring_rolled_loop\n")   # a dup across two lines
    _wp18_kb_usage_note(ctx, 7, turn)
    assert ctx.kb_used_pending[7] == ["mbarrier_ring_rolled_loop", "tma_refill"]
    rows = [json.loads(ln) for ln in
            _wp18_kb_usage_path(ctx).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(r["verdict"] == "pending" for r in rows) and len(rows) == 2

    keep = IterationRecord(iter=7, decision_id="D1", change="ring", correctness=True,
                           verdict="KEEP", sha="sha7")
    _wp18_kb_usage_flow(ctx, keep)
    revert = IterationRecord(iter=8, decision_id="D2", change="tilt", correctness=False,
                             verdict="REVERT", sha="")
    _wp18_kb_usage_note(ctx, 8, "kb_used: mbarrier_ring_rolled_loop\n")
    _wp18_kb_usage_flow(ctx, revert)
    # the flow consumed the pending rows: a second call is a no-op
    _wp18_kb_usage_flow(ctx, keep)
    summary = _wp18_kb_usage_summary(ctx)
    assert summary == {"mbarrier_ring_rolled_loop": {"KEEP": 1, "REVERT": 1},
                       "tma_refill": {"KEEP": 1, "REVERT": 0}}


def test_kb_used_no_quote_no_row(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    _wp18_kb_usage_note(ctx, 3, "decision_id: D9\nchange: x\nfiles_touched: k.py\n")
    assert not ctx.kb_used_pending
    assert not _wp18_kb_usage_path(ctx).exists()
    assert _wp18_kb_usage_summary(ctx) == {}


# --- task 1: the bandwidth-recipe bounce -----------------------------------------


_INFEASIBLE_BLOCKED = ("reachability:      floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED\n"
                       "cause_class: INFEASIBLE\ndeficient_locus: none\n"
                       "evidence: the wall is bandwidth\n"
                       "patch_direction: stop\n")


def test_bw_bounce_fires_once_then_the_stop_lands(tmp_path):
    """A BLOCKED verdict with no measured bandwidth reading: bounced once, the
    resent diagnosis is taken as-is (acceptance 2)."""
    be = FakeBackend([_INFEASIBLE_BLOCKED, _INFEASIBLE_BLOCKED], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "infeasible-stop"
    assert len(be.prompts) == 2
    assert "effective_bw" in be.prompts[1]              # the bounce named the recipe
    assert sum("effective_bw" in p for p in be.prompts) == 1   # once, not a gate


def test_no_bw_bounce_when_a_bandwidth_reading_exists():
    """A same-family measured reading in the step's probe readings, or an
    effective_bw citation in the diagnosis itself, answers the check."""
    assert _wp18_bw_recipe_missing(_INFEASIBLE_BLOCKED, []) is True
    reading = [{"path": "notes/effective_bw.py", "contract": "node_bw", "note": "measured"}]
    assert _wp18_bw_recipe_missing(_INFEASIBLE_BLOCKED, reading) is False
    cited = _INFEASIBLE_BLOCKED.replace(
        "evidence: the wall is bandwidth", "evidence: effective_bw measured 1.9 TB/s node peak")
    assert _wp18_bw_recipe_missing(cited, []) is False


def test_no_bw_bounce_when_the_refusal_is_already_failing(tmp_path):
    """An OPEN request fails the refusal bounce; the WP18 check must not add a
    second, contradictory block on that path (the refusal's own conditions)."""
    open_line = _INFEASIBLE_BLOCKED.replace(
        "floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED",
        "floor_opt=1.20 ms vs target=1.9 ms -> OPEN")
    be = FakeBackend([open_line, open_line], [])
    ctx = _ctx(tmp_path, be)
    res = run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "infeasible-refused"
    assert sum("effective_bw" in p for p in be.prompts) == 0
    assert "INFEASIBLE stop request not accepted" in be.prompts[1]


# --- task 1: KB-MENU reads the recent two readings --------------------------------


def test_recent_readings_newest_first():
    from types import SimpleNamespace
    recs = [SimpleNamespace(profiler={}, i=0),
            SimpleNamespace(profiler={"spill_st_sectors": 5}, i=1),
            SimpleNamespace(profiler={"dram_pct": 90.0}, i=2),
            SimpleNamespace(profiler=None, i=3)]
    two = _wp18_recent_readings(recs, 2)
    assert two[0] == {"dram_pct": 90.0} and two[1] == {"spill_st_sectors": 5}
    assert _wp18_recent_readings(recs, 2)[:1] == [{"dram_pct": 90.0}]
    assert _wp18_recent_readings([], 2) == []


def test_menu_renders_a_prev_only_fire(tmp_path):
    """The newest reading went quiet but the previous one fired the spill rule:
    the menu keeps the phase visible with one [prev reading] line (acceptance 3);
    a rule firing on both carries no marker; None renders as before."""
    loud = {"spill_st_sectors": 64000}
    quiet = {"dram_pct": 90.0}          # fires dram-high only
    menu = kbmenu.build_kb_menu(readings=quiet, prev_readings=loud, index_text="",
                                kb_label="cuda/sm100", op="gemm")
    assert "spill/cstack>0" in menu and "[prev reading]" in menu
    assert "dram_pct=90%" in menu and "dram_pct=90% [prev reading]" not in menu
    # deterministic + no prev -> byte-identical to the pre-WP18 render
    assert kbmenu.build_kb_menu(readings=quiet, prev_readings=None, index_text="",
                                kb_label="cuda/sm100", op="gemm") == \
           kbmenu.build_kb_menu(readings=quiet, index_text="", kb_label="cuda/sm100", op="gemm")


# --- task 3: scheduled candidate-queue batch certification -------------------------


def test_duty_block_lists_the_open_head(tmp_path):
    be = FakeBackend([], [])
    ctx = _queue_ctx(tmp_path, be)
    block = _wp18_kb_cert_duty_block(ctx, meta_step=2)   # every=2 -> due on even steps
    assert "Scheduled candidate-queue certification" in block
    assert "`wave_tail_governance`" in block and "`already_landed`" not in block
    # one job, per-candidate verdicts, failures do not contaminate
    assert "ONE job" in block and "notes/kb_cert_batch.py" in block
    assert _is_wp18_batch_request("notes/kb_cert_batch.py")
    assert not _is_wp18_batch_request("notes/probe_layout.py")
    # odd step: not due
    assert _wp18_kb_cert_duty_block(ctx, meta_step=3) == ""
    # batch=1 degrades to the per-candidate path (acceptance 4, regression)
    ctx.kb_cert_batch_size = 1
    single = _wp18_kb_cert_duty_block(ctx, meta_step=2)
    assert "One candidate per job" in single and "ONE job" not in single
    # batch=0 disables the duty entirely
    ctx.kb_cert_batch_size = 0
    assert _wp18_kb_cert_duty_block(ctx, meta_step=2) == ""


def test_batch_certification_one_job_per_candidate_verdicts(tmp_path):
    """N=4 in ONE job; verdicts ride the meta record as kb_cert:<name> rows;
    the PASSed rows stamp BATCH:<job>, the failed one stays TODO (acceptance 4)."""
    diagnosis = "cause_class: R2\ndeficient_locus: M1\npatch_direction: x\n"
    patch = _patching(tmp_path, "\n### D6 added\nx\n",
                      "patch_class: additive\nchanged_entries: []\npatch: ...\n")
    log = ("\n".join([
        "PROBE PASS max_diff=0.0 # wave_tail_governance",
        "PROBE PASS max_diff=0.0 # tma_multicast_fanout",
        "PROBE FAIL max_diff=0.4 # clc_persistent_claim",
        "PROBE PASS max_diff=n/a # l2_window_persistence",
    ]) + "\n")
    be = _JobFakeBackend(["PROBE_REQUEST: notes/kb_cert_batch.py", diagnosis, patch], [log])
    ctx = _queue_ctx(tmp_path, be)
    res = run_meta_step(ctx)
    assert res.patch_class == "additive"
    # exactly ONE GPU job carried all four candidates
    gpu_calls = [c for c in be.calls if c[0] == "gpu"]
    assert gpu_calls == [("gpu", "cmd:python3 notes/kb_cert_batch.py")]
    # per-candidate verdicts ride the step's meta record
    rec = ctx.meta_status_path and [json.loads(ln) for ln in
                                    ctx.meta_status_path.read_text(encoding="utf-8").splitlines()][-1]
    certs = [r for r in rec["probe_readings"] if str(r.get("contract", "")).startswith("kb_cert:")]
    assert sorted(str(r["contract"]) for r in certs) == sorted(
        f"kb_cert:{n}" for n in ("wave_tail_governance", "tma_multicast_fanout",
                                 "clc_persistent_claim", "l2_window_persistence"))
    assert sum(r["verdict"] == "PASS" for r in certs) == 3
    # queue rows: exactly the three PASSed ones stamped, cell-only
    queue = (tmp_path / "hw_kb" / "CANDIDATES.md").read_text(encoding="utf-8")
    assert "BATCH:4242" in queue and queue.count("BATCH:4242") == 3
    assert "| `already_landed` — certified last run | `cutedsl/sm90` | Low | history | DONE |" in queue
    assert "| `clc_persistent_claim` — CLC claims in a persistent grid | `cuda/sm100` | Medium | composition unmeasured | TODO |" in queue
    assert "`wave_tail_governance` — price the tail-wave fix | `cutedsl/sm90` | High | waves cliff unmeasured | BATCH:4242 |" in queue


def test_parse_kb_cert_log_isolates_failures():
    log = ("PROBE PASS max_diff=0.0 # a_ok\n"
           "PROBE FAIL max_diff=9.9 # b_broken\n"
           "garbage line with no verdict\n"
           "PROBE PASS max_diff=n/a # c_contract\n")
    results = _wp18_parse_kb_cert_log(log)
    assert results["a_ok"]["verdict"] == "PASS"
    assert results["b_broken"]["verdict"] == "FAIL"     # isolated: still one verdict per candidate
    assert results["c_contract"]["verdict"] == "PASS"
    assert _wp18_parse_kb_cert_log("no verdicts at all") == {}


def test_batch_of_one_never_stamps_a_batch(tmp_path):
    """A single-candidate job (the per-candidate path) still records its verdict
    and stamps the row — BATCH:<job> is the notation whatever the size."""
    results = _wp18_parse_kb_cert_log("PROBE PASS max_diff=0.1 # wave_tail_governance\n")
    lines = _QUEUE.splitlines()
    stamped = _wp18_stamp_kb_cert(_queue_ctx(tmp_path, _JobFakeBackend([], [])),
                                  results, "77")
    assert stamped == 1
    out = (tmp_path / "hw_kb" / "CANDIDATES.md").read_text(encoding="utf-8")
    assert out.count("BATCH:77") == 1


# --- task 3/5: kbmenu appended rules -> real pointers ------------------------------


_WP18_RULES = ("wave-tail", "tma-multicast", "warp-spec-pipeline", "prefetch-lead",
               "launch-bounds", "l2-policy-window")


def test_wp18_rules_appended_only():
    names = [r.name for r in kbmenu.RULES]
    assert names[:10] == ["spill", "regs-at-cap", "bankconf", "l2-low", "waves-low",
                          "smem-cliff", "latency-bound", "unbound-attribution",
                          "wave-quantization", "dram-high"]   # the pre-WP18 table untouched
    assert [n for n in names if n in _WP18_RULES] == list(_WP18_RULES)   # appended, in order


@pytest.mark.parametrize("tree", ("cuda/sm100", "cutedsl/sm100", "cutedsl/sm90"))
def test_wp18_rules_point_at_real_rows_on_three_trees(tree):
    """Acceptance 5: on every tree, at least three of the appended rules resolve
    a pointer whose quote is a verbatim substring of that tree's index."""
    from peregrine_kernel.loop.kbmenu import _match_row, parse_symptom_rows
    idx = (REPO / "hw_kb" / tree / "index.md").read_text(encoding="utf-8")
    rows = parse_symptom_rows(idx)
    resolved = 0
    for rule in kbmenu.RULES:
        if rule.name not in _WP18_RULES or rule.keywords is None:
            continue
        hit = _match_row(rows, rule.keywords)
        if hit is not None and hit.entry:
            assert hit.entry.endswith(".md")
            assert (REPO / "hw_kb" / tree / hit.entry).is_file(), (tree, rule.name)
            assert hit.snippet in idx, (tree, rule.name)
            resolved += 1
    assert resolved >= 3, tree


# --- task 5: hit telemetry ---------------------------------------------------------


def test_hit_record_and_ledger_format(tmp_path):
    """Acceptance 6: one jsonl row per render — rule names, section line counts;
    the harvest aggregates it (renders + rule fire counts)."""
    readings = {"dram_pct": 90.0}
    menu = kbmenu.build_kb_menu(readings=readings, index_text="", kb_label="cuda/sm100",
                                op="gemm")
    rules = kbmenu.fired_rule_names(readings, {}, prev_readings=None)
    assert rules == ["dram-high"]
    row = kbmenu.hit_record(kb_label="cuda/sm100", op="gemm", rules=rules, n_hints=1,
                            n_lever_lines=1, n_calibration=0)
    assert row == {"kb_label": "cuda/sm100", "op": "gemm", "rules": ["dram-high"],
                   "hints": 1, "lever_lines": 1, "calibration_lines": 0}
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    live = [IterationRecord(iter=0, decision_id="D", change="x", correctness=True,
                            verdict="KEEP", profiler=dict(readings))]
    driver._wp18_kbmenu_hits_note(ctx, readings, menu, live)
    path = tmp_path / "evidence" / "kbmenu_hits.jsonl"
    saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert saved["rules"] == ["dram-high"] and saved["op"] == "demo" and "ts" in saved
    assert _wp18_kbmenu_hits_summary(ctx)["rule_fires"] == {"dram-high": 1}
    assert _wp18_kbmenu_hits_summary(ctx)["renders"] == 1
    # an empty menu writes nothing (the driver injects nothing -> nothing counted)
    driver._wp18_kbmenu_hits_note(ctx, {}, "", live)
    assert _wp18_kbmenu_hits_summary(ctx)["renders"] == 1


# --- task 3: CANDIDATE-DRAFT landing (dedup: dup not landed) ------------------------


def test_candidate_drafts_land_deduped(tmp_path):
    """Acceptance: every survey line lands as a DRAFT row EXCEPT one a tree
    already covers (pure string dedup, fail closed)."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.kb_dir = tmp_path / "hw_kb" / "cutedsl"
    tree = ctx.kb_dir / "sm90"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "index.md").write_text("| tail waves | [`wave_tail_governance`](wave_tail_governance.md): "
                                   "the tail-wave cliff |\n", encoding="utf-8")
    (tmp_path / "hw_kb" / "CANDIDATES.md").write_text(_QUEUE, encoding="utf-8")
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "prior_art.md").write_text(
        "### 1. some decision\n- Decision: x\n\n"
        "CANDIDATE-DRAFT: wave_tail_governance | cutedsl/sm90 | tail waves | already known\n"
        "CANDIDATE-DRAFT: tma_multicast_fanout | cutedsl/sm100 | operand fan-out | "
        "910->1450 GB/s in a public microbenchmark; worth a probe\n",
        encoding="utf-8")
    landed = _wp18_land_candidate_drafts(ctx)
    assert landed == ["tma_multicast_fanout"]
    queue = (tmp_path / "hw_kb" / "CANDIDATES.md").read_text(encoding="utf-8")
    assert queue.count("DRAFT") == 1 and "wave_tail_governance | cutedsl/sm90 | tail waves" not in queue
    assert "| `tma_multicast_fanout` — survey candidate" in queue


# --- task 1: priced_null + the lever x shape matrix ---------------------------------


def test_priced_null_stamps_the_milestone_lines(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text("### D6\nAction: try `lever:regsplit` [MEASURED, job 12]\n"
                             "### D7\nAction: keep `lever:other` [MEASURED, job 13]\n",
                             encoding="utf-8")
    table = [{"kind": "lever", "name": "regsplit", "cls": "null", "add_pct": 0.2,
              "remove_pct": -0.1, "decisions": ["D6"]},
             {"kind": "lever", "name": "other", "cls": "harmful", "add_pct": -3.0}]
    driver._wp18_stamp_priced_null(ctx, table)
    ms = ctx.milestone.read_text(encoding="utf-8")
    assert "PRICED_NULL: measured null (+0.20%)" in ms
    assert ms.index("PRICED_NULL") > ms.index("lever:regsplit")   # stamped after its citation
    assert "lever:other` [MEASURED, job 13]" in ms         # harmful rides REMOVE, not this stamp
    once = ms
    driver._wp18_stamp_priced_null(ctx, table)              # idempotent: one standing verdict
    assert ctx.milestone.read_text(encoding="utf-8") == once


def test_lever_shape_matrix_banks_per_workload(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    rows = [{"kind": "lever", "name": "regsplit", "cls": "independent", "add_pct": 1.0,
             "remove_pct": -1.0, "decisions": ["D6"],
             "workloads": [{"workload": "wl2", "add_pct": 2.2},
                           {"workload": "wl9", "add_pct": -0.3}]},
            {"kind": "stack", "all_pct": 1.2, "sum_singles_pct": 1.1, "coupling_pp": 0.1}]
    _wp18_write_lever_shape_matrix(ctx, rows, "refsha", "77")
    rec = json.loads((tmp_path / "evidence" / "lever_shape_matrix.json").read_text(
        encoding="utf-8").splitlines()[0])
    assert rec["best_ref"] == "refsha" and rec["job"] == "77"
    lever = next(r for r in rec["levers"] if r["lever"] == "regsplit")
    assert [w["workload"] for w in lever["workloads"]] == ["wl2", "wl9"]
    assert lever["decisions"] == ["D6"]
    assert len(rec["levers"]) == 1   # the stack row carries no name: not a matrix row


# --- task 3: the prompt segments -----------------------------------------------------


def test_harvest_prompts_carry_the_wp18_segments():
    for prompt_path in (REPO / "meta" / "kb_harvest_prompt.md",
                        REPO / "meta" / "cuda" / "kb_harvest_prompt.md"):
        text = prompt_path.read_text(encoding="utf-8")
        seg = text.split("## Regime field, survival weighting")[1].split("## Hard rules")[0]
        assert "`regime:` field" in seg and "survival: <n operators>" in seg
        assert "`DRAFT` rows" in seg and "`BATCH:<job>` rows" in seg
        # the four segments stay within the overlay budget (<= 20 lines)
        assert len([ln for ln in seg.splitlines() if ln.strip()]) <= 20, prompt_path


def test_survey_skill_carries_the_draft_mapping():
    text = (REPO / "skills" / "sota-survey" / "SKILL.md").read_text(encoding="utf-8")
    seg = text.split("## Map every uncovered mechanism")[1]
    assert "CANDIDATE-DRAFT: <mechanism-name> | <tree(s)" in seg
    assert "dedup" in seg and "certifies nothing" in seg


def test_student_prompts_carry_the_kb_used_field():
    for name in ("meta/student_prompt.md", "meta/cuda/student_prompt.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "kb_used:" in text and "objective citation" in text


def test_loop_toml_documents_the_wp18_keys():
    text = (REPO / "global_config" / "loop.toml").read_text(encoding="utf-8")
    block = text.split("# --- WP18:")[1]
    assert "kb_cert_batch_size" in block and "kb_cert_every_meta" in block


# --- task 5: recon renders its KB binding --------------------------------------------


def test_recon_prompt_renders_the_kb_binding(tmp_path):
    """The recon brief carries this arch's KB tree the same way distill does:
    the binding resolves, the discipline is on the line (acceptance 8)."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.evidence_dir = tmp_path / "evidence"
    text = render_prompt(_meta_prompt(ctx, "recon_prompt.md"), _bindings(ctx))
    assert "`KB_DIR`" in text and "distill step" in text
    assert "{KB_READ_DIR}" not in text          # the binding resolved
    assert str(ctx.kb_dir) in text


# --- task 4: build_priors from artifacts ---------------------------------------------


def _fixture_repo(tmp_path):
    """A one-kernel repo: kernels/<fam>/<op>/<gpu>/kernel.py + README row."""
    root = tmp_path / "repo"
    kdir = root / "kernels" / "gemm" / "demo_gem" / "h100"
    kdir.mkdir(parents=True)
    (kdir / "kernel.py").write_text("# demo kernel\n", encoding="utf-8")
    (root / "kernels" / "README.md").write_text(
        "| demo GEMM | H100 | cuBLAS | +1.2% | Figure 3 | [`gemm/demo_gem/h100`](gemm/demo_gem/h100/) |\n",
        encoding="utf-8")
    return root


def test_build_priors_from_artifacts_turns_skeletons_into_priors(tmp_path):
    """Acceptance 7: the skeleton -> filled diff; anchors are machine facts with
    their source named; full coverage flips the status, partial stays protected."""
    import build_priors as bp
    root = _fixture_repo(tmp_path)
    assert bp.main(["--repo", str(root), "--write"]) == 0
    prior = root / "priors" / "gemm" / "demo_gem_h100.md"
    text = prior.read_text(encoding="utf-8")
    assert "Status: skeleton" in text and text.count("TODO:") == 6
    before_filled = text.count("Status: filled")

    arts = root / "priors" / "artifacts" / "gemm" / "demo_gem_h100"
    arts.mkdir(parents=True)
    (arts / "ncu.json").write_text(json.dumps({
        "grid_size": 132, "block_size": 384, "occupancy_pct": 61.0,
        "smem_per_block": 232448.0}), encoding="utf-8")
    (arts / "sass_census.json").write_text(json.dumps({
        "kernel.cubin": {"total": 900, "families": {"mma": 210, "tma": 12, "mbarrier": 40,
                                                    "cp_async": 18, "reg_role": 4,
                                                    "st_matrix": 16, "st_global": 9,
                                                    "other": 591}}}), encoding="utf-8")
    (arts / "manifest.json").write_text(json.dumps({
        "kernel": "kernels/gemm/demo_gem/h100",
        "regime": "demo GEMM — published: +1.2% vs cuBLAS (Figure 3)"}), encoding="utf-8")

    # WP23 (review report 95, WP18 F-3): the fill pass rides the same write gate as
    # the skeleton pass — a dry run lists, --write lands
    text_before = prior.read_text(encoding="utf-8")
    assert bp.main(["--repo", str(root), "--from-artifacts"]) == 0
    assert prior.read_text(encoding="utf-8") == text_before   # dry run wrote nothing
    assert bp.main(["--repo", str(root), "--from-artifacts", "--write"]) == 0
    filled = prior.read_text(encoding="utf-8")
    assert filled != text_before
    assert filled.count("- Measured anchor (") >= 6       # every section anchored
    assert "Status: filled (auto from measured artifacts" in filled
    assert "the published launch is grid=132 CTAs" in filled
    assert "sass census" in filled and "STSM/LDSM" in filled
    assert "Shape domain" in filled and "published: +1.2%" in filled
    # a filled file is never touched again (write or not)
    assert bp.main(["--repo", str(root), "--from-artifacts", "--write"]) == 0
    assert prior.read_text(encoding="utf-8") == filled


def test_build_priors_partial_artifacts_stay_anchored_not_filled(tmp_path):
    import build_priors as bp
    root = _fixture_repo(tmp_path)
    assert bp.main(["--repo", str(root), "--write"]) == 0
    prior = root / "priors" / "gemm" / "demo_gem_h100.md"
    arts = root / "priors" / "artifacts" / "gemm" / "demo_gem_h100"
    arts.mkdir(parents=True)
    (arts / "sass_census.json").write_text(json.dumps({
        "k.cubin": {"total": 10, "families": {"mbarrier": 3}}}), encoding="utf-8")
    assert bp.main(["--repo", str(root), "--from-artifacts", "--write"]) == 0
    text = prior.read_text(encoding="utf-8")
    assert "Status: auto-anchored" in text
    assert "Measured anchor (sass census): 3" in text
    assert text.count("TODO:") == 5                        # the unmeasured sections keep theirs


def test_collector_census_and_ncu_parse():
    """The collector's two pure parsers: SASS families count off a real-shaped
    nvdisasm dump; the NCU CSV folds into the key-metric dict."""
    import collect_kernel_artifacts as ck
    sass = ("        /*0100*/                   HMMA.16816.F32.BF16 R0, R5, R9, R2;\n"
            "        /*0110*/                   TMA.LOAD.TILE R12, [R2];\n"
            "        /*0120*/                   CP.ASYNC.COL.RTO.32 R2, [R4];\n"
            "        /*0130*/                   MBAR.ARRIVE.DYNSYNC.ADD R2, 32;\n"
            "        /*0140*/                   SETMAXNREG INC 64;\n"
            "        /*0150*/                   STSM R6, [R2+0x10];\n"
            "        /*0160*/                   LDG.E.128 R6, [R2];\n"
            "        /*0170*/                   NOP;\n")
    census = ck.census_from_sass(sass)
    assert census["total"] == 8
    assert census["families"]["mma"] == 1 and census["families"]["tma"] == 1
    assert census["families"]["cp_async"] == 1 and census["families"]["mbarrier"] == 1
    assert census["families"]["reg_role"] == 1 and census["families"]["st_matrix"] == 1
    assert census["families"]["ld_global"] == 1 and census["families"]["other"] == 1

    csv = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context",'
           '"Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name",'
           '"Metric Unit","Metric Value"\n'
           '"0","1","p","h","k(int, int)","1","1","(128, 1, 1)","(132, 1, 1)","0","9.0",'
           '"Speed Of Light","sm__throughput.avg.pct_of_peak_sustained_elapsed","%",'
           '"62.31"\n'
           '"0","1","p","h","k(int, int)","1","1","(128, 1, 1)","(132, 1, 1)","0","9.0",'
           '"Launch Statistics","launch__registers_per_thread","register/thread","168"\n'
           '"0","1","p","h","k(int, int)","1","1","(128, 1, 1)","(132, 1, 1)","0","9.0",'
           '"Launch Statistics","launch__shared_mem_per_block_static","Kbyte","227"\n')
    metrics = ck._parse_ncu_csv(csv)
    assert metrics["compute_sol_pct"] == 62.31
    assert metrics["regs_per_thread"] == 168.0
    assert metrics["smem_per_block"] == 227 * 1024.0       # the unit folded honestly
    assert "l2_hit_rate_pct" not in metrics                # NCU did not report it
