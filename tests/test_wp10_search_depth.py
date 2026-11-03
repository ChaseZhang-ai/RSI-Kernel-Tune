"""WP10 (search-depth deepening): the STAGNATION architecture-escape trigger, the
compile-flags decision axis, and MC-DELIVERY (student structure fan-out).

Every path here keeps the WP10 red lines: the trigger only ADDS pressure to the
teacher's prompt (cause_class stays the teacher's, the override is recorded, the
driver never machine-changes a verdict); flags are pure prompt supply over the
existing whitelist; the MC winner is one delivery judged by the unmodified full
gate, and a run without the authorization behaves exactly as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.inner.trajectory import IterationRecord, append_record, load_records
from forgekernel.meta_trajectory import append_meta_record, load_meta_records
from forgekernel.loop.driver import (
    IterationRecord as _Rec,  # noqa: F401  (import shape check only)
    _mc_timings_rows,
    _milestone_multi_candidate,
    _parse_stagnation_override,
    _stagnation_block,
    _stagnation_report,
    _stagnation_attempts,
    parse_mc_results,
)

REPO = Path(__file__).resolve().parents[1]


# --- fixtures -------------------------------------------------------------------


def _row(iter_no, did, metric_before, ms, *, verdict="KEEP", correctness=True,
         regression=None, change="tune", reading_of=None):
    return IterationRecord(
        iter=iter_no, decision_id=did, change=change,
        correctness=correctness, verdict=verdict, metric_before=metric_before,
        device_median_ms=ms, regression_vs_best_pct=regression,
        reading_of=reading_of,
    )


def _stag_ctx(tmp_path, be, k=3, target_ms=1.9):
    from test_loop_driver import _ctx
    ctx = _ctx(tmp_path, be, target_ms=target_ms)
    ctx.stagnation_k = k
    return ctx


def _flat_rows():
    """Four consecutive parameter-level attempts, each inside the 1.5% band
    (a new best each time, but never beyond the noise), best 2.592 on a 1.9
    target: the parameter-level local optimum signature."""
    return [
        _row(1, "D1.1", 2.620, 2.610),   # +0.38%
        _row(2, "D2.1", 2.610, 2.615),   # -0.19% (kept, slower than best)
        _row(3, "D2.3", 2.610, 2.592),   # +0.69% (new best, inside band)
        _row(4, "D2.4", 2.592, 2.600),   # -0.31%
    ]


def _write_rows(tmp_path, rows):
    (tmp_path / "status.jsonl").write_text("")
    for r in rows:
        append_record(tmp_path / "status.jsonl", r)


# --- task 1: STAGNATION ---------------------------------------------------------


def test_stagnation_triggers_on_k_flat_attempts_with_a_real_gap(tmp_path):
    ctx = _stag_ctx(tmp_path, None)
    _write_rows(tmp_path, _flat_rows())
    rep = _stagnation_report(ctx, load_records(ctx.status_path), [])
    assert rep is not None and rep["k"] == 3
    assert rep["gap_pct"] == pytest.approx(100.0 * (2.592 - 1.9) / 1.9)
    assert rep["overrides"] == 0
    block = _stagnation_block(rep)
    # the injected block names the attempts and their gains, the band, the gap,
    # the structural constraint and the override release valve (contract C3:
    # one subsection, ≤ 10 lines)
    assert block.count("\n## Stagnation") == 1
    for needle in ("D2.3: +0.69%", "D2.4: -0.31%", "resolvable_diff",
                   "R2-structural", "stagnation_override:", "Past stagnation overrides: 0"):
        assert needle in block
    assert len(block.strip().splitlines()) <= 10


def test_stagnation_silent_when_an_attempt_clears_the_band(tmp_path):
    ctx = _stag_ctx(tmp_path, None)
    rows = _flat_rows()
    rows[3] = _row(4, "D2.4", 2.592, 2.500)     # +3.5%: the band was broken
    _write_rows(tmp_path, rows)
    assert _stagnation_report(ctx, load_records(ctx.status_path), []) is None


def test_stagnation_silent_without_a_real_gap(tmp_path):
    ctx = _stag_ctx(tmp_path, None)
    rows = _flat_rows()
    for r, ms in zip(rows, (1.92, 1.915, 1.912, 1.918)):
        r.metric_before, r.device_median_ms = None, ms
    # seed a metric_before chain by hand: flat, best 1.912, target 1.9 -> the gap
    # (0.6%) sits inside the band: nothing left to escape toward
    prev = 1.93
    for r in rows:
        r.metric_before = prev
        prev = min(prev, r.device_median_ms)
    _write_rows(tmp_path, rows)
    assert _stagnation_report(ctx, load_records(ctx.status_path), []) is None


def test_stagnation_needs_k_attempts_and_honors_the_off_switch(tmp_path):
    ctx = _stag_ctx(tmp_path, None)
    _write_rows(tmp_path, _flat_rows()[:2])
    assert _stagnation_report(ctx, load_records(ctx.status_path), []) is None  # 2 < K=3
    ctx.stagnation_k = 0                                                       # the off switch
    assert _stagnation_report(ctx, load_records(ctx.status_path), []) is None


def test_stagnation_counts_perf_reverts_and_additive_patch_outcomes(tmp_path):
    # an evaluated perf REVERT is a judged parameter-level attempt (gain 0); so is
    # a REVERT_PATCHed additive meta patch. A structural patch is not.
    ctx = _stag_ctx(tmp_path, None)
    rows = _flat_rows()[:2] + [
        _row(3, "D3.1", 2.610, None, verdict="REVERT", correctness=True),
    ]
    _write_rows(tmp_path, rows)
    from forgekernel.meta_trajectory import MetaRecord
    meta = [
        MetaRecord(meta_step=1, cause_class="G2", deficient_locus="D2.3",
                   patch_class="structural", changed_entries=[], patch_direction="split",
                   outcome="KEEP_PATCH", best_before=2.62, best_after=2.55),
        MetaRecord(meta_step=2, cause_class="R2", deficient_locus="D6",
                   patch_class="additive", changed_entries=[], patch_direction="add",
                   outcome="REVERT_PATCH", best_before=2.60),
    ]
    attempts = _stagnation_attempts(ctx, load_records(ctx.status_path), meta)
    gains = [a["gain"] for a in attempts]
    assert gains == pytest.approx([0.38, -0.19, 0.0, 0.0], rel=0.05, abs=1e-9)
    assert attempts[-1]["label"].startswith("step2 additive patch REVERT_PATCH")
    # the structural patch's outcome never enters the parameter-level sequence
    assert all("structural" not in a["label"] for a in attempts)


def test_stagnation_block_injected_into_the_diagnose_prompt_and_override_recorded(tmp_path):
    """Dry-run evidence: the K-flat fixture drives run_meta_step, the diagnose
    prompt carries the STAGNATION block (after the WP4 REMOVE_CANDIDATES
    section, per the C3 render order), and the teacher's `stagnation_override:`
    line is recorded into the meta trajectory for the next diagnosis to count."""
    from test_loop_driver import FakeBackend, _DIAG, _MILESTONE, _patching
    from forgekernel.loop.driver import run_meta_step

    ctx = _stag_ctx(tmp_path, None)
    rows = _flat_rows()
    rows[3] = _row(4, "D2.4", 2.592, 2.600, regression=0.31)   # a WP4 slow-decision row too
    _write_rows(tmp_path, rows)
    diag = _DIAG + "\nstagnation_override: the tile is one change from binding — measure it"
    be = FakeBackend([diag,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    ctx.backend = be
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    run_meta_step(ctx)
    prompt = be.prompts[0]
    assert "## Stagnation" in prompt and "stagnation_override:" in prompt
    # C3 render order: the injected block (its unique footer line) follows the
    # WP4 REMOVE_CANDIDATES section and adds no other section. ("## Stagnation"
    # also occurs in the rendered diagnose_prompt.md teaching text — the first
    # match is the template's, not the injection's.)
    assert "## Slow decisions" in prompt
    assert prompt.index("## Slow decisions") < prompt.index("Past stagnation overrides: 0.")
    recs = load_meta_records(ctx.meta_status_path)
    assert len(recs) == 1 and "[stagnation_override]" in recs[0].patch_direction
    # the next diagnosis counts it
    rep = _stagnation_report(ctx, load_records(ctx.status_path), load_meta_records(ctx.meta_status_path))
    assert rep is not None and rep["overrides"] == 1
    assert "Past stagnation overrides: 1" in _stagnation_block(rep)


def test_stagnation_silent_without_the_block_does_not_honor_a_stray_override(tmp_path):
    """No trigger -> no constraint -> an override line in a diagnosis of that
    step is not a stagnation override (run_meta_step never marks it)."""
    rows = _flat_rows()
    rows[3] = _row(4, "D2.4", 2.592, 2.500)     # band broken: no trigger
    _write_rows(tmp_path, rows)
    ctx = _stag_ctx(tmp_path, None)
    rep = _stagnation_report(ctx, load_records(ctx.status_path), [])
    assert rep is None


def test_stagnation_k_reads_as_loop_config_with_the_wp1_precedent():
    """The key's wiring shape: a LoopContext default (the WP1/WP6 precedent — the
    card loop.toml's [loop] is the override surface, the global layer takes
    [target]/[budget] only), default 3, plus its multi_candidate_max neighbour."""
    import dataclasses
    from forgekernel.loop import driver as drv
    names = {f.name: f.default for f in dataclasses.fields(drv.LoopContext)}
    assert names["stagnation_k"] == 3 and names["multi_candidate_max"] == 3


def test_parse_stagnation_override():
    assert _parse_stagnation_override("cause_class: R1\nstagnation_override: why\n") == "why"
    assert _parse_stagnation_override("stagnation_override：fullwidth colon") == "fullwidth colon"
    assert _parse_stagnation_override("stagnation_override:\n") is None
    assert _parse_stagnation_override("no line at all") is None


# --- task 2: the flags decision axis ---------------------------------------------


def _card_leaderboard_module():
    """The card's eval/leaderboard.py (the whitelist's single source), imported
    with a stand-in torch when there is none — the same loader shape the
    cuda-student tests use."""
    import importlib.util
    import sys
    import types
    path = REPO / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py"
    created = None
    if importlib.util.find_spec("torch") is None:
        created = types.ModuleType("torch")

        class _Any:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return _Any()

            def __call__(self, *a, **k):
                return _Any()

            def __iter__(self):
                return iter(())

            def __mro_entries__(self, bases):
                return (object,)

            def __hash__(self):
                return id(self)

        created.__getattr__ = lambda n: _Any()
        created.__version__ = "0.0+fake"
        sys.modules["torch"] = created
    try:
        spec = importlib.util.spec_from_file_location("_wp10_card_lb", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if created is not None:
            del sys.modules["torch"]


_KERNEL_TAIL = (
    "#include <torch/extension.h>\n"
    "__global__ void k(float* x, int n) {}\n"
    "void run(torch::Tensor t) {}\n"
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"run\", &run); }\n")


def test_flags_whitelist_accepts_the_wp10_lever_examples():
    mod = _card_leaderboard_module()
    # every flag family the diagnose prompt teaches, on one fk-cuda-cflags line
    line = ("// fk-cuda-cflags: --maxrregcount=168 --ftz=true --fmad=false "
            "--prec-div=false --prec-sqrt=false --extra-device-vectorization -lineinfo\n")
    flags = mod.cuda_kernel_flags(line + _KERNEL_TAIL)
    for f in ("--maxrregcount=168", "--ftz=true", "--fmad=false", "--prec-div=false",
              "--prec-sqrt=false", "--extra-device-vectorization", "-lineinfo"):
        assert f in flags
    # the sweep SKILL's gradient is legal on its own, and -O tuning too
    for extra in ("--maxrregcount=128", "--maxrregcount=224", "--maxrregcount=255",
                  "-O2", "--ftz=false", "--prec-div=true"):
        assert extra in mod.cuda_kernel_flags(f"// fk-cuda-cflags: {extra}\n" + _KERNEL_TAIL)


def test_flags_whitelist_refuses_what_the_prompt_does_not_teach():
    mod = _card_leaderboard_module()
    for bad in ("-I/secrets", "-lmylib", "-L/tmp", "-Xlinker ../ref_bin/x.so",
                "--use-fast-math=true"):      # not a whitelist spelling
        with pytest.raises(ValueError):
            mod.cuda_kernel_flags(f"// fk-cuda-cflags: {bad}\n" + _KERNEL_TAIL)


def test_use_fast_math_is_listed_with_the_precision_gate_caveat():
    """The prompt lists --use_fast_math but routes its numerics through the
    precision gate: the whitelist itself accepts the flag (the card's default
    already carries it), and the diagnose prompt's flags section says so."""
    mod = _card_leaderboard_module()
    assert "--use_fast_math" in mod.cuda_kernel_flags(
        "// fk-cuda-cflags: --use_fast_math\n" + _KERNEL_TAIL)
    text = (REPO / "meta" / "diagnose_prompt.md").read_text(encoding="utf-8")
    assert "precision gate" in text.split("Flags are a lever axis")[1].split("- **Stagnation")[0]


# --- task 3: MC-DELIVERY ----------------------------------------------------------


_MC_LOG = (
    "MC kernel_file=flash_attn_dsl/flash_bwd.py variants: "
    "flash_attn_dsl/flash_bwd_v0.py flash_attn_dsl/flash_bwd_v1.py\n"
    "MC name=flash_attn_dsl/flash_bwd_v0.py smoke=pass ms=n/a ratio=n/a delta=n/a\n"
    "MC name=flash_attn_dsl/flash_bwd_v1.py smoke=pass ms=n/a ratio=n/a delta=n/a\n"
    "MC ROUND 0 base pass=True device=2.60 vs_ref_device=0.605\n"
    "MC ROUND 0 flash_attn_dsl/flash_bwd_v0.py pass=True device=2.55 vs_ref_device=0.617\n"
    "MC ROUND 0 flash_attn_dsl/flash_bwd_v1.py pass=True device=2.30 vs_ref_device=0.684\n"
    "MC ROUND 1 base pass=True device=2.61 vs_ref_device=0.603\n"
    "MC ROUND 1 flash_attn_dsl/flash_bwd_v0.py pass=True device=2.56 vs_ref_device=0.615\n"
    "MC ROUND 1 flash_attn_dsl/flash_bwd_v1.py pass=True device=2.31 vs_ref_device=0.681\n"
    "MC name=flash_attn_dsl/flash_bwd_v1.py smoke=pass ms=2.305 ratio=0.6825 delta=+12.90%\n"
    "MC name=flash_attn_dsl/flash_bwd_v0.py smoke=pass ms=2.555 ratio=0.616 delta=+1.90%\n"
    "MC-OK variants=2 rounds=2 base_ms=2.605 winner=flash_attn_dsl/flash_bwd_v1.py "
    "floor=1.10%\n"
    "MC-WINNER name=flash_attn_dsl/flash_bwd_v1.py\n")

_MC_ALL_DNF_LOG = (
    "MC kernel_file=flash_attn_dsl/flash_bwd.py variants: "
    "flash_attn_dsl/flash_bwd_v0.py flash_attn_dsl/flash_bwd_v1.py\n"
    "MC name=flash_attn_dsl/flash_bwd_v0.py smoke=dnf ms=n/a ratio=n/a delta=n/a\n"
    "MC name=flash_attn_dsl/flash_bwd_v1.py smoke=dnf ms=n/a ratio=n/a delta=n/a\n"
    "MC-OK variants=0 rounds=0 winner=none\n"
    "MC-WINNER name=none (every variant failed the correctness smoke)\n")


def _milestone_multi(did_line_anchor="- Consumer phase: [PHASE: none]\n", n=2):
    from test_loop_driver import _MILESTONE
    return _MILESTONE.replace(did_line_anchor, did_line_anchor + f"multi_candidate: {n}\n", 1)


def _mc_student(tmp_path, variants):
    """A student turn that writes its own variant implementations and delivers
    the decision as one ordinary record block."""
    from test_loop_driver import REC

    def agent(_prompt):
        for rel, src in variants.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src, encoding="utf-8")
        return REC("D1.1", 2.6)
    return agent


def test_mc_parses_the_prescreen_log():
    rows, winner, fail = parse_mc_results(_MC_LOG)
    assert fail is None and winner == "flash_attn_dsl/flash_bwd_v1.py"
    assert [r["name"] for r in rows][:2] == [
        "flash_attn_dsl/flash_bwd_v0.py", "flash_attn_dsl/flash_bwd_v1.py"]
    assert rows[0]["smoke"] and rows[0]["ms"] is None          # the smoke row
    assert rows[2]["ms"] == pytest.approx(2.305)               # the timed row
    assert rows[2]["delta_pct"] == pytest.approx(12.90)
    rows2, winner2, fail2 = parse_mc_results(_MC_ALL_DNF_LOG)
    assert winner2 is None and fail2 is None
    assert len(rows2) == 2 and not any(r["smoke"] for r in rows2)
    rows3, winner3, fail3 = parse_mc_results("nothing here")
    assert winner3 is None and fail3 == "no MC line in the job log"
    banked = _mc_timings_rows(rows, winner)
    assert banked[2]["winner"] is True and banked[3]["winner"] is False
    assert "smoke_raw" not in json.dumps(banked)               # banked rows carry the schema, not raw tokens


def test_mc_authorization_is_read_from_the_milestone_and_clamped(tmp_path):
    from test_loop_driver import FakeBackend, _ctx
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_MILESTONE_MULTI := _milestone_multi(), encoding="utf-8")
    assert _milestone_multi_candidate(ctx, "D1.1") == 2
    assert _milestone_multi_candidate(ctx, "D2.3") == 0        # no field on that decision
    ctx.multi_candidate_max = 2
    assert min(_milestone_multi_candidate(ctx, "D1.1"), ctx.multi_candidate_max) == 2
    ctx.milestone.write_text(_milestone_multi(n=3), encoding="utf-8")
    ctx.multi_candidate_max = 2
    assert min(_milestone_multi_candidate(ctx, "D1.1"), ctx.multi_candidate_max) == 2  # clamped
    ctx.multi_candidate_max = 0                                 # the channel's off switch
    assert min(_milestone_multi_candidate(ctx, "D1.1"), ctx.multi_candidate_max) == 0
    del _MILESTONE_MULTI


def test_mc_variant_files_are_named_after_the_kernel_stem(tmp_path):
    from test_loop_driver import FakeBackend, _ctx
    ctx = _ctx(tmp_path, FakeBackend([], []))
    (tmp_path / "flash_attn_dsl").mkdir(exist_ok=True)
    for rel in ("flash_attn_dsl/flash_bwd_v0.py", "flash_attn_dsl/flash_bwd_v1.py"):
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "flash_attn_dsl/flash_bwd_v3.py").write_text("x = 3\n", encoding="utf-8")
    from forgekernel.loop.driver import _mc_variant_files
    assert _mc_variant_files(ctx, 2) == ["flash_attn_dsl/flash_bwd_v0.py",
                                         "flash_attn_dsl/flash_bwd_v1.py"]
    assert _mc_variant_files(ctx, 3)[2:] == []                 # v2 missing -> only two exist


def test_mc_delivery_full_chain_winner_applied_and_banked(tmp_path):
    """2 variants, one authorization: the prescreen job runs once, the winner's
    source becomes the kernel file, the variants are archived, and the record
    banks the objective variant table; the winner alone took the standard gate."""
    from test_loop_driver import EVAL, FakeBackend, NCU, REC, _ctx, run_inner_iteration
    from forgekernel.loop.driver import _MC_CMD
    v0, v1 = "def fwd_v0():\n    return 0\n", "def fwd_v1():\n    return 1\n"
    be = FakeBackend(
        [_mc_student(tmp_path, {"flash_attn_dsl/flash_bwd_v0.py": v0,
                                "flash_attn_dsl/flash_bwd_v1.py": v1})],
        [_MC_LOG, EVAL(2.31), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "KEEP" and rec.bwd_median_ms == 2.31
    # exactly one prescreen job, then the ordinary eval + ncu of the winner tree
    gpu = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu[0].startswith(_MC_CMD) and gpu[1] == "all" and gpu[2] == "ncu"
    # rec.sha is the anchor every later reset / confirmation / pricing re-read
    # follows: it must name the tree the eval job measured (the winner-applied
    # head), not the stale student delivery commit the iteration began with
    assert rec.sha == be.head()
    assert rec.sha != "sha1"    # sha1 = the student delivery commit (FakeBackend counter)
    # the winner IS the delivered kernel; the variants are out of the delivery tree
    assert (tmp_path / "flash_attn_dsl" / "flash_bwd.py").read_text(encoding="utf-8") == v1
    assert not (tmp_path / "flash_attn_dsl" / "flash_bwd_v0.py").exists()
    assert not (tmp_path / "flash_attn_dsl" / "flash_bwd_v1.py").exists()
    archived = tmp_path / "notes" / "variants" / "iter3"
    assert (archived / "flash_bwd_v0.py").read_text(encoding="utf-8") == v0
    assert (archived / "flash_bwd_v1.py").read_text(encoding="utf-8") == v1
    # the record banks the objective table: smoke verdicts, readings, the winner
    assert rec.variant_timings is not None and len(rec.variant_timings) == 4
    winners = [r for r in rec.variant_timings if r["winner"]]
    assert len(winners) == 1 and winners[0]["variant"] == "flash_attn_dsl/flash_bwd_v1.py"
    assert winners[0]["ms"] == pytest.approx(2.305)            # the winner's timed row
    assert any(r["ms"] is None for r in rec.variant_timings)   # smoke rows carry no timing
    on_disk = load_records(ctx.status_path)[0]
    assert on_disk.variant_timings == rec.variant_timings      # survives the jsonl round trip
    # the prescreen script and the vendored timing kernel were seeded into notes/
    assert (tmp_path / "notes" / "mc_prescreen.py").is_file()
    assert (tmp_path / "notes" / "lever_pricing_run.py").is_file()


def test_mc_delivery_dnf_variants_revert_without_an_eval(tmp_path):
    """Every variant fails the smoke: DNF, no winner, the ordinary REVERT flow —
    no eval job is spent, the workspace is restored, the table is banked."""
    from test_loop_driver import FakeBackend, _ctx, run_inner_iteration
    v0, v1 = "def fwd_v0():\n    return 0\n", "def fwd_v1():\n    return 1\n"
    be = FakeBackend(
        [_mc_student(tmp_path, {"flash_attn_dsl/flash_bwd_v0.py": v0,
                                "flash_attn_dsl/flash_bwd_v1.py": v1})],
        [_MC_ALL_DNF_LOG])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "REVERT" and rec.correctness is None
    assert "multi-candidate" in rec.error
    assert not any(c == ("gpu", "all") for c in be.calls)      # no eval job bought
    assert any(c[0] == "reset" for c in be.calls)              # the workspace was restored
    # the archive is written AFTER the reset and is NOT committed (a committed
    # archive would die with the discarded commit on a real git reset --hard)
    assert not any(c[0] == "commit" and c[1].startswith("mc: variants")
                   for c in be.calls)
    assert rec.variant_timings is not None and len(rec.variant_timings) == 2
    assert not any(r["winner"] for r in rec.variant_timings)
    # the variants were archived for the record even on the failed fan-out
    assert (tmp_path / "notes" / "variants" / "iter3" / "flash_bwd_v0.py").read_text(
        encoding="utf-8") == v0


def test_mc_dnf_archive_survives_a_git_semantics_reset(tmp_path):
    """The bug the no-op reset could not catch: with real reset --hard file
    semantics the DNF path's reset rewinds the tree and would delete any
    committed archive copy together with the variant sources. The fix reads the
    sources into memory, resets, and writes the archive back uncommitted — the
    files must be on disk after the iteration."""
    from test_loop_driver import FakeBackend, _ctx, run_inner_iteration

    class _HardResetBackend(FakeBackend):
        """reset with git reset --hard's file semantics: tracked files the
        target tree does not hold are removed from the disk workspace
        (untracked files survive). Commits snapshot the workspace's relative
        paths — the minimal model that makes the DNF archive's survival
        observable, which a no-op reset cannot catch."""

        def __init__(self, agent_outputs, gpu_logs, workspace):
            super().__init__(agent_outputs, gpu_logs)
            self.workspace = Path(workspace)
            self._snapshots = {self.head(): frozenset()}
            self._tracked_head = self.head()

        def _walk(self):
            out = set()
            for p in self.workspace.rglob("*"):
                if p.is_file() and ".git" not in p.parts:
                    out.add(str(p.relative_to(self.workspace)))
            return frozenset(out)

        def commit(self, message: str) -> str:
            sha = super().commit(message)
            self._snapshots[sha] = self._walk()
            self._tracked_head = sha
            return sha

        def reset(self, ref: str) -> None:
            super().reset(ref)
            gone = self._snapshots.get(self._tracked_head, frozenset()) - \
                self._snapshots.get(ref, frozenset())
            for rel in gone:
                p = self.workspace / rel
                if p.exists():
                    p.unlink()
            self._tracked_head = ref

    v0, v1 = "def fwd_v0():\n    return 0\n", "def fwd_v1():\n    return 1\n"
    be = _HardResetBackend(
        [_mc_student(tmp_path, {"flash_attn_dsl/flash_bwd_v0.py": v0,
                                "flash_attn_dsl/flash_bwd_v1.py": v1})],
        [_MC_ALL_DNF_LOG], tmp_path)
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "REVERT" and "multi-candidate" in rec.error
    # reset --hard took the tracked variant files off the tree, as a real backend does
    assert not (tmp_path / "flash_attn_dsl" / "flash_bwd_v0.py").exists()
    assert not (tmp_path / "flash_attn_dsl" / "flash_bwd_v1.py").exists()
    # ...and the archive survives it, on disk, in the workspace
    assert (tmp_path / "notes" / "variants" / "iter3" / "flash_bwd_v0.py").read_text(
        encoding="utf-8") == v0
    assert (tmp_path / "notes" / "variants" / "iter3" / "flash_bwd_v1.py").read_text(
        encoding="utf-8") == v1
    assert rec.variant_timings is not None and len(rec.variant_timings) == 2


def test_mc_prescreen_job_failure_keeps_banked_sha_at_the_push_head(tmp_path):
    """Authorized, but the prescreen job cannot run (BackendError): the toolkit
    and spec commits already moved the tree, the delivery proceeds as committed —
    and rec.sha must still name the HEAD the eval branch pushes (the anchor the
    next reset / confirmation / pricing re-read follows), not the stale
    student delivery commit."""
    from test_loop_driver import EVAL, NCU, FakeBackend, _ctx, run_inner_iteration

    class _PrescreenDownBackend(FakeBackend):
        """submit_gpu fails ONLY for the prescreen job (the BackendError path);
        every other job runs the ordinary queue."""

        def submit_gpu(self, branch: str, mode: str) -> str:
            if "mc_prescreen" in mode:
                from forgekernel.loop.driver import BackendError
                raise BackendError("node gone")
            return super().submit_gpu(branch, mode)

    v0, v1 = "def fwd_v0():\n    return 0\n", "def fwd_v1():\n    return 1\n"
    be = _PrescreenDownBackend(
        [_mc_student(tmp_path, {"flash_attn_dsl/flash_bwd_v0.py": v0,
                                "flash_attn_dsl/flash_bwd_v1.py": v1})],
        [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "KEEP"
    assert rec.variant_timings is None                     # no reading, no table
    assert rec.sha == be.head()                            # the tree the eval measured
    assert rec.sha != "sha1"                               # not the stale delivery commit
    # no winner was ever selected or applied
    assert not any(c[0] == "commit" and "winner" in c[1] for c in be.calls)
    assert (tmp_path / "flash_attn_dsl" / "flash_bwd_v0.py").exists()   # delivery as committed


def test_mc_delivery_without_authorization_is_byte_identical_to_the_old_path(tmp_path):
    """The regression gate: variant files present, NO multi_candidate field —
    no prescreen job, no winner selection, the ordinary delivery path, and no
    variant_timings on the record."""
    from test_loop_driver import EVAL, FakeBackend, NCU, _ctx, _MILESTONE, run_inner_iteration
    be = FakeBackend(
        [_mc_student(tmp_path, {"flash_attn_dsl/flash_bwd_v0.py": "a = 0\n",
                                "flash_attn_dsl/flash_bwd_v1.py": "a = 1\n"})],
        [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "KEEP"
    gpu = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu == ["all", "ncu"]                               # exactly the old job pair
    assert rec.variant_timings is None
    # the kernel file was never touched by a winner selection
    assert (tmp_path / "flash_attn_dsl" / "flash_bwd_v0.py").read_text(encoding="utf-8") == "a = 0\n"


def test_mc_delivery_authorized_but_no_variant_files_is_the_old_path(tmp_path):
    from test_loop_driver import EVAL, FakeBackend, NCU, _ctx, run_inner_iteration
    be = FakeBackend(["decision_id: D1.1\nchange: did it\nfiles_touched: "
                      "flash_attn_dsl/flash_bwd.py\n"], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    rec = run_inner_iteration(ctx, 3, metric_before=2.9)
    assert rec is not None and rec.verdict == "KEEP"
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "ncu"]
    assert rec.variant_timings is None


def test_mc_gate_rows_and_reopens_never_fan_out(tmp_path):
    """A `multi_candidate:` field is for a decision's implementations: a gate row
    (or a REOPEN row) at an authorized id is bookkeeping, never a fan-out."""
    from forgekernel.loop.driver import _mc_deliver
    from test_loop_driver import FakeBackend, REC, _ctx

    class _Rec2:
        decision_id = "M1-gate"
        change = "gate"

    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.milestone.write_text(_milestone_multi(), encoding="utf-8")
    assert _mc_deliver(ctx, ctx.backend, 1, _Rec2(), "b", None, 2.9) is None


def test_mc_prescreen_script_runs_end_to_end_against_a_stubbed_timing_kernel(tmp_path):
    """The driver-generated script, executed for real (a stub lever_pricing_run
    stands in for the GPU bench): smoke first, interleaved timing, the winner
    line, and the tree restored byte-for-byte."""
    import subprocess
    import sys
    from forgekernel.loop.driver import _MC_PRESCREEN_SRC
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "mc_prescreen.py").write_text(_MC_PRESCREEN_SRC, encoding="utf-8")
    (tmp_path / "notes" / "lever_pricing_run.py").write_text(
        '''
_DEVICE_MS = {"BASE KERNEL": 2.60, "V0 KERNEL": 2.55, "V1 KERNEL": 2.30}
ROLE_PASS_MARGIN_S = 60.0
_SCOPE_WANT = {"name": "representative"}


def node_dirty():
    return False


def job_seconds_left(t0):
    return None


def run_bench(variant, keep_cubin, timeout=2400):
    src = open("flash_attn_dsl/flash_bwd.py").read()
    ms = _DEVICE_MS.get(src.strip())
    return {"correctness": {"pass": ms is not None},
            "performance": {"device_median_ms": ms, "vs_ref_device": 1.6 / (ms or 1)}}


def ratio_reading(perf):
    return perf.get("vs_ref_device")


def med(xs):
    xs = [x for x in xs if x is not None]
    return sorted(xs)[len(xs) // 2] if xs else None


def make_scorer(order, ok, dev, ratio):
    use_ratio = bool(ratio.get("base")) and all(x is not None for x in ratio["base"])

    def score(v):
        if not ok.get(v):
            return None
        return med(ratio[v]) if use_ratio else (1.0 / med(dev[v]) if med(dev[v]) else None)

    def delta(a, b):
        sa, sb = score(a), score(b)
        return None if sa is None or not sb else 100.0 * (sa / sb - 1.0)

    return score, delta, use_ratio


def noise_floor(order, ok, dev):
    return 1.0


def clamp_pass_budget(cap_s, left_s, margin_s):
    # WP17: the prescreen reads the timing kernel's shared job-clock clamp
    if left_s is None:
        return float(cap_s)
    return max(0.0, min(float(cap_s), float(left_s) - float(margin_s)))


def paired_order(variants):
    # WP17: the prescreen reads the timing kernel's shared interleaved order
    return ["base", *variants]
''', encoding="utf-8")
    (tmp_path / "flash_attn_dsl").mkdir()
    base = "BASE KERNEL\n"
    (tmp_path / "flash_attn_dsl" / "flash_bwd.py").write_text(base, encoding="utf-8")
    (tmp_path / "flash_attn_dsl" / "flash_bwd_v0.py").write_text("V0 KERNEL\n", encoding="utf-8")
    (tmp_path / "flash_attn_dsl" / "flash_bwd_v1.py").write_text("V1 KERNEL\n", encoding="utf-8")
    (tmp_path / "notes" / "mc.json").write_text(json.dumps({
        "kernel_file": "flash_attn_dsl/flash_bwd.py",
        "variants": ["flash_attn_dsl/flash_bwd_v0.py", "flash_attn_dsl/flash_bwd_v1.py"],
        "rounds": 2, "workload": "representative"}), encoding="utf-8")
    # run the way the job runs it (cwd = repo root; notes/ is then sys.path[0],
    # which is how the vendored flat import finds its sibling timing kernel)
    r = subprocess.run([sys.executable, "notes/mc_prescreen.py", "notes/mc.json"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert out.count("smoke=pass ms=n/a") == 2          # the smoke phase: one row per variant
    assert "smoke=dnf" not in out
    assert "MC ROUND 0" in out and "MC ROUND 1" in out
    assert "MC-WINNER name=flash_attn_dsl/flash_bwd_v1.py" in out
    assert "MC-OK variants=2 rounds=2 base_ms=2.6 winner=flash_attn_dsl/flash_bwd_v1.py" in out
    # the tree is left as the prescreen found it
    assert (tmp_path / "flash_attn_dsl" / "flash_bwd.py").read_text(encoding="utf-8") == base


def test_stagnation_injection_is_minimal_block_or_nothing(tmp_path):
    """Dry-run minimality evidence: with the trigger off, the diagnose prompt is
    byte-identical to the pre-WP10 assembly; with it on, the prompt differs by
    EXACTLY the STAGNATION block — nothing else moved."""
    import tempfile
    from test_loop_driver import FakeBackend, _DIAG, _MILESTONE, _patching
    from forgekernel.loop.driver import _stagnation_block, _stagnation_report, run_meta_step

    prompts = []
    tmpdirs = []
    for k in (0, 3):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            tmpdirs.append(str(t))
            ctx = _stag_ctx(t, None, k=k)
            rows = _flat_rows()
            rows[3] = _row(4, "D2.4", 2.592, 2.600, regression=0.31)
            _write_rows(t, rows)
            be = FakeBackend([_DIAG,
                              _patching(t, "\n### D6 added\nx\n",
                                        "patch_class: additive\nchanged_entries: []\n")], [])
            ctx.backend = be
            ctx.milestone.write_text(_MILESTONE, encoding="utf-8")
            run_meta_step(ctx)
            prompts.append(be.prompts[0])
    off, on = prompts
    assert "Past stagnation overrides: 0." not in off          # the off case: nothing injected
    rep = _stagnation_report(_stag_ctx(Path(tempfile.mkdtemp()), None, k=3),
                             _flat_rows(), [])
    block = _stagnation_block(rep)
    # the two runs ran in different tmp dirs (the prompt carries the milestone
    # path): normalize before comparing, so only the block can remain
    off, on = off.replace(tmpdirs[0], "<T>"), on.replace(tmpdirs[1], "<T>")
    assert len(on) - len(off) == len(block)
    assert on.replace(block, "") == off                        # the diff is the block, exactly
