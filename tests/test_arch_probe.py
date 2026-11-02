"""ARCH_PROBE: the M0 pre-freeze architecture probe channel (WP6 task 1).

The freeze is the priced action: a divided key-topology reading (cluster size,
tile scheme, warp-role split, smem topology) must be measured BEFORE M0 freezes
it. The teacher writes top-2 skeleton recipes into the milestone
(``ARCH_PROBE_REQUEST: notes/arch_<n>.json``), the driver freezes their execution
(time-only, correctness never read — the verdict-probe rule), one GPU job per
candidate, and stamps the outcome into the milestone head: a clear winner is what
the freeze follows; anything inconclusive freezes the teacher's preferred
candidate with the readings attached. Probe material (recipes + the frozen
runner) is committed for the jobs and reset away afterwards — the student line
stays teacher-free. Budget = m0_probe_rounds (default 2 = one per candidate).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    EVAL, NCU, REC, _MILESTONE, _E2EFake, FakeBackend, _ctx,
)

from forgekernel.loop.driver import (  # noqa: E402
    BackendError, _arch_probe_ms, _arch_probe_stamp, _parse_arch_probe_requests,
    _run_arch_probes, resolvable_diff, run_two_layer,
)

# a §9-conformant v1 whose head carries the ARCH_PROBE machine lines
_MS_DUAL = _MILESTONE.replace(
    "# demo milestone\n",
    "# demo milestone\n"
    "ARCH_CANDIDATES: dual\n"
    "ARCH_PROBE_REQUEST: notes/arch_0.json\n"
    "ARCH_PROBE_REQUEST: notes/arch_1.json\n",
    1)


def _distill_ctx(tmp_path, be, ms_text=_MS_DUAL, **kw):
    """A run_two_layer context entering the distill flow: no milestone on disk,
    the (fake) distill agent writes ms_text, the oracle path present."""
    ms = tmp_path / "distilled.md"
    ctx = _ctx(tmp_path, be, **kw)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    be.milestone_path = ms
    be.patch_writes = [ms_text]
    return ctx, ms


def _authored(ctx, ms_text=_MS_DUAL):
    """Put the teacher's (fake) authoring output on disk, recipes included."""
    ctx.milestone.write_text(ms_text, encoding="utf-8")
    notes = ctx.workspace / "notes"
    notes.mkdir(exist_ok=True)
    for n in (0, 1):
        (notes / f"arch_{n}.json").write_text(
            '{"name": "c%d", "rationale": "r", "language": "python", "source": "x"}' % n,
            encoding="utf-8")


def _recipes(tmp_path):
    """The two recipe files, as the teacher's distill turn left them on disk."""
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    for n in (0, 1):
        (notes / f"arch_{n}.json").write_text(
            '{"name": "c%d", "rationale": "r", "language": "python", "source": "x"}' % n,
            encoding="utf-8")


def test_two_candidates_two_jobs_winner_locked(tmp_path):
    """The headline path: two candidates -> exactly two probe jobs -> the winner's
    reading is stamped into the milestone head as what the freeze follows."""
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.200000\n", "ARCH_PROBE arch_1 ms=0.150000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    _recipes(tmp_path)
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "target"
    # one job per candidate, before anything else ran on the GPU
    gpu_modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert gpu_modes[0] == "cmd:python3 notes/arch_probe_run.py notes/arch_0.json"
    assert gpu_modes[1] == "cmd:python3 notes/arch_probe_run.py notes/arch_1.json"
    # the freeze follows the winner; the request lines are consumed
    text = ms.read_text(encoding="utf-8")
    assert "ARCH_PROBE (driver, pre-freeze" in text
    assert "winner notes/arch_1.json" in text and "notes/arch_0.json" in text
    assert "ARCH_PROBE_REQUEST" not in text and "ARCH_CANDIDATES" not in text
    # the §9 skeleton the teacher authored is untouched underneath the stamp
    assert "## 3. Decision details" in text and "## Worked Example" in text
    # probe material travels only inside the probe commit, then the workspace is
    # reset to the pre-probe head (the student line stays teacher-free)
    assert ("commit", "meta-probe: ") in be.calls
    assert ("push", "fs-demo-archprobe") in be.calls
    assert ("reset", "sha0") in be.calls


def test_two_candidates_within_floor_freeze_the_preferred(tmp_path):
    """A gap inside the resolution floor is not a verdict: the teacher's preferred
    candidate freezes, with both readings attached (a recorded measurement, not a
    silent bet). A data-poor run's floor is the 1.5% legacy constant."""
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.200000\n", "ARCH_PROBE arch_1 ms=0.202000\n", EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    _recipes(tmp_path)
    run_two_layer(ctx, max_meta=1)
    text = ms.read_text(encoding="utf-8")
    assert "readings within the resolution floor" in text
    assert "frozen as authored: the teacher's preferred candidate (notes/arch_0.json)" in text
    assert "0.2 ms vs 0.202 ms" in text  # the readings ride the note


def test_a_missing_reading_is_not_evidence_about_the_other(tmp_path):
    """A candidate that never printed a verdict line (compile fail / crash) reads
    as 'no reading': inconclusive, the preferred candidate freezes."""
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 status=FAIL reason=CompileError: nvcc\n",
                  "ARCH_PROBE arch_1 ms=0.150000\n", EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    _recipes(tmp_path)
    run_two_layer(ctx, max_meta=1)
    text = ms.read_text(encoding="utf-8")
    assert "inconclusive" in text and "arch_0" in text
    assert "winner" not in text


def test_single_candidate_runs_zero_jobs(tmp_path):
    """No divergence: `ARCH_CANDIDATES: single`, no recipes -> no probe jobs, and
    the freeze proceeds exactly as before this channel existed."""
    ms_text = _MILESTONE.replace(
        "# demo milestone\n", "# demo milestone\nARCH_CANDIDATES: single\n", 1)
    be = _E2EFake(agent_outputs=["authored", REC("D1.1", 1.8)],
                  gpu_logs=[EVAL(1.8), NCU],
                  milestone_path=tmp_path / "distilled.md", patch_writes=[ms_text])
    ctx, ms = _distill_ctx(tmp_path, be, ms_text=ms_text)
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "target"
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "ncu"]


def test_one_request_skips_with_a_record(tmp_path):
    """One recipe alone compares against nothing: skip, record, zero jobs."""
    ms_text = _MILESTONE.replace(
        "# demo milestone\n",
        "# demo milestone\nARCH_PROBE_REQUEST: notes/arch_0.json\n", 1)
    be = _E2EFake(agent_outputs=["authored", REC("D1.1", 1.8)],
                  gpu_logs=[EVAL(1.8), NCU],
                  milestone_path=tmp_path / "distilled.md", patch_writes=[ms_text])
    ctx, ms = _distill_ctx(tmp_path, be, ms_text=ms_text)
    run_two_layer(ctx, max_meta=1)
    assert [c[1] for c in be.calls if c[0] == "gpu" and str(c[1]).startswith("cmd:")] == []
    text = ms.read_text(encoding="utf-8")
    assert "M0 probes skipped (only one candidate recipe)" in text
    assert "ARCH_PROBE_REQUEST" not in text


def test_budget_below_two_skips_and_records(tmp_path):
    """The budget is per candidate: with one round for two candidates the channel
    skips entirely and says so in the milestone head."""
    be = _E2EFake(["should not run"], [], milestone_path=tmp_path / "distilled.md", patch_writes=[])
    ctx, _ms = _distill_ctx(tmp_path, be)
    _authored(ctx)
    ctx.m0_probe_rounds = 1
    _run_arch_probes(ctx)
    assert [c for c in be.calls if c[0] == "gpu"] == []
    assert "M0 probes skipped (budget)" in ctx.milestone.read_text(encoding="utf-8")


def test_a_missing_recipe_file_skips_rather_than_probes(tmp_path):
    """A recipe the teacher named but never wrote is not probeable: skip, record."""
    be = _E2EFake(["should not run"], [], milestone_path=tmp_path / "distilled.md", patch_writes=[])
    ctx, _ms = _distill_ctx(tmp_path, be)
    ctx.milestone.write_text(_MS_DUAL, encoding="utf-8")   # recipes NOT on disk
    _run_arch_probes(ctx)
    assert [c for c in be.calls if c[0] == "gpu"] == []
    assert "M0 probes skipped (recipe file missing)" in ctx.milestone.read_text(encoding="utf-8")


def test_a_failed_job_freezes_as_authored_and_still_resets(tmp_path):
    """A BackendError on the first probe ends the channel (no further jobs), the
    freeze proceeds on the preferred candidate, and the probe workspace reset
    still happens — a failed job must not leave teacher material behind."""
    class _Boom(FakeBackend):
        def submit_gpu(self, branch, mode):
            self.calls.append(("gpu", mode))
            raise BackendError("node lost")

    be = _Boom(["should not run"], [])
    ctx, _ms = _distill_ctx(tmp_path, be)
    _authored(ctx)
    _run_arch_probes(ctx)
    assert [c[1] for c in be.calls if c[0] == "gpu"] == \
        ["cmd:python3 notes/arch_probe_run.py notes/arch_0.json"]
    assert ("reset", "sha0") in be.calls
    text = ctx.milestone.read_text(encoding="utf-8")
    assert "inconclusive" in text and "ARCH_PROBE_REQUEST" not in text


def test_malformed_v1_never_reaches_the_probes(tmp_path):
    """The probes serve the freeze decision; a v1 that fails its schema gate stops
    the run before any freeze — so no probe job is bought for it either."""
    ms = tmp_path / "m.md"
    ms.write_text("# skeleton only\n", encoding="utf-8")
    be = _E2EFake(agent_outputs=["authored"] * 4, gpu_logs=[],
                  milestone_path=ms,
                  patch_writes=[_MS_DUAL.replace("## 3. Decision details\n", "")])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    ctx.milestone = ms
    ctx.oracle_path = tmp_path / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "distill_malformed"
    assert [c for c in be.calls if c[0] == "gpu"] == []


def test_arch_probe_ms_reads_the_runner_line():
    # exact stem match, positive floats only — a FAIL line and another candidate's
    # line must not be misattributed
    log = ("ARCH_PROBE arch_0 status=FAIL reason=CompileError: nvcc\n"
           "ARCH_PROBE arch_1 ms=0.150000\n")
    assert _arch_probe_ms(log, "arch_1") == pytest.approx(0.15)
    assert _arch_probe_ms(log, "arch_0") is None
    assert _arch_probe_ms("no verdict line here", "arch_1") is None
    assert _arch_probe_ms("ARCH_PROBE arch_1 ms=0.0\n", "arch_1") is None


def test_parse_requests_dedupes_and_reads_fullwidth_colon():
    text = ("ARCH_CANDIDATES: dual\n"
            "ARCH_PROBE_REQUEST: notes/arch_0.json\n"
            "ARCH_PROBE_REQUEST：notes/arch_1.json\n"     # fullwidth colon
            "ARCH_PROBE_REQUEST: notes/arch_0.json\n")    # duplicate
    assert _parse_arch_probe_requests(text) == ["notes/arch_0.json", "notes/arch_1.json"]
    assert _parse_arch_probe_requests("ARCH_CANDIDATES: single\n") == []


def test_stamp_decision_rule_units():
    """The rule in isolation: winner above the floor, preferred within it, the
    legacy 1.5% floor when no run resolution exists (WP1's data-poor fallback)."""
    reqs = ["notes/arch_0.json", "notes/arch_1.json"]
    win = _arch_probe_stamp(reqs, {reqs[0]: 0.200, reqs[1]: 0.150}, floor=0.015)
    assert any("winner notes/arch_1.json" in ln for ln in win)
    assert any("the M0 freeze follows the WINNER" in ln for ln in win)
    near = _arch_probe_stamp(reqs, {reqs[0]: 0.200, reqs[1]: 0.202}, floor=0.015)
    assert any("within the resolution floor" in ln for ln in near)
    assert any("(notes/arch_0.json)" in ln for ln in near)
    # an explicit floor of 0 makes any measurable gap decisive (a run that
    # measured its resolution as ~0 trusts small gaps)
    zero = _arch_probe_stamp(reqs, {reqs[0]: 0.200, reqs[1]: 0.2005}, floor=0.0)
    assert any("winner notes/arch_0.json" in ln for ln in zero)
    missing = _arch_probe_stamp(reqs, {reqs[1]: 0.150}, floor=0.015)
    assert any("inconclusive" in ln for ln in missing)
    skipped = _arch_probe_stamp(reqs, {}, skipped="M0 probes skipped (budget)")
    assert skipped == ["<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)",
                       "     M0 probes skipped (budget) -->"]


def test_resolvable_diff_data_poor_fallback_is_the_legacy_constant():
    """C1: the driver consumes WP1's resolvable_diff (direct import, no shim);
    for a data-poor run WP1's own fallback keeps the 1.5% legacy constant — the
    reader must stay callable with the C1 signature."""
    assert resolvable_diff([], "e2e") == pytest.approx(0.015)


def test_probe_runner_is_seeded_for_the_job(tmp_path):
    """The runner is vendored from the skills tree into notes/ — the committed file
    the probe job ships — alongside the recipes, inside the probe commit."""
    be = _E2EFake(agent_outputs=["authored", "EXHAUSTED: nothing"],
                  gpu_logs=["ARCH_PROBE arch_0 ms=0.2\n", "ARCH_PROBE arch_1 ms=0.1\n"],
                  milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, _ms = _distill_ctx(tmp_path, be)
    _authored(ctx)
    _run_arch_probes(ctx)
    runner = tmp_path / "notes" / "arch_probe_run.py"
    assert runner.is_file() and "arch_probe_run" in runner.read_text()
    assert ("commit", "meta-probe: ") in be.calls
