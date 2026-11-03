"""WP9 C1/C4/C5: the wiring-level fixes.

- C1: a latency-shaped problem's generated loop.toml writes `timeline_probe = "auto"`
  (the driver's auto mode keys on the milestone's `bottleneck: latency`); every
  other card stays byte-identical.
- C4: `{CANDIDATE}` — bound in every prompt since the binding existed, assigned by
  nobody — gets pinned to the ARCH_PROBE winner (the machine-readable "chosen
  candidate"; the freeze follows it by the distill prompt's own rule).
- C5: the re-anchor memo rebuilds from reanchor.jsonl after a driver restart, so
  the same best no longer re-buys the same-card re-measure.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from test_arch_probe import _MS_DUAL, _authored, _distill_ctx  # noqa: E402
from test_loop_driver import EVAL, NCU, REC, _E2EFake, FakeBackend, _ctx  # noqa: E402

from forgekernel.loop.driver import (  # noqa: E402
    _bindings, _phantom_gap_reanchor, _reanchor_memo, _run_arch_probes,
    _timeline_on, run_two_layer,
)
from forgekernel.taskcard import timeline_probe_auto, write_loop_toml


# --- C1 -------------------------------------------------------------------------------

def _fake_task(**over):
    task = SimpleNamespace(
        description="fuse a PyTorch RMSNorm into one kernel", name="rmsnorm",
        shapes=[SimpleNamespace(name="prefill", axes={"m": 8192, "n": 4096})],
        loop_language="cutedsl", arch="sm100", target=SimpleNamespace(timing="device"),
        budget=SimpleNamespace(reference_hours=8.0), reference=None, mode="generate",
    )
    for k, v in over.items():
        if k == "shapes":
            task.shapes = [SimpleNamespace(name=n, axes={}) for n in v]
        else:
            setattr(task, k, v)
    return task


def test_latency_shaped_task_writes_timeline_probe_auto(tmp_path):
    assert timeline_probe_auto(_fake_task(name="gqa_paged_decode")) is True
    assert timeline_probe_auto(_fake_task(description="a low-latency decode kernel")) is True
    assert timeline_probe_auto(_fake_task()) is False, "rmsnorm is not latency-shaped"


def test_generated_loop_toml_carries_auto_only_for_latency_tasks(tmp_path):
    run = SimpleNamespace(task_dir=tmp_path, operators_dir=tmp_path)
    for d in ("a", "b", "c"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    off = write_loop_toml(run, _fake_task(), tmp_path / "a", deny=[], ncu=True)
    assert "timeline_probe" not in off.read_text(encoding="utf-8"), \
        "non-latency card stays byte-identical to the historical default"
    auto = write_loop_toml(run, _fake_task(name="mla_paged_decode"), tmp_path / "b",
                           deny=[], ncu=True)
    assert 'timeline_probe    = "auto"' in auto.read_text(encoding="utf-8")
    # NCU-less machine: the explicit "off" wins (the probe cannot run there)
    off2 = write_loop_toml(run, _fake_task(name="mla_paged_decode"), tmp_path / "c",
                           deny=[], ncu=False)
    assert 'timeline_probe    = "off"' in off2.read_text(encoding="utf-8")


def test_auto_mode_keys_on_the_milestones_bottleneck_line(tmp_path):
    """The value the card writes is the driver's `auto`: on exactly when the
    milestone declares bottleneck: latency (driver.py _timeline_on)."""
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text("# m\nbottleneck: latency\n", encoding="utf-8")
    ctx.timeline_probe = "auto"
    assert _timeline_on(ctx) is True
    ctx.milestone.write_text("# m\nbottleneck: bankconflict\n", encoding="utf-8")
    assert _timeline_on(ctx) is False


# --- C4 -------------------------------------------------------------------------------

def test_arch_probe_winner_pins_the_candidate(tmp_path):
    """The probe's decisive winner becomes ctx.candidate: `{CANDIDATE}` renders the
    pinned name in every later prompt instead of '(not pinned)'."""
    from test_arch_probe import _recipes
    be = _E2EFake(
        agent_outputs=["authored", REC("D1.1", 1.8)],
        gpu_logs=["ARCH_PROBE arch_0 ms=0.200000\n", "ARCH_PROBE arch_1 ms=0.150000\n",
                  EVAL(1.8), NCU],
        milestone_path=tmp_path / "distilled.md", patch_writes=[_MS_DUAL])
    ctx, ms = _distill_ctx(tmp_path, be)
    assert ctx.candidate is None
    _recipes(tmp_path)
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "target"
    assert ctx.candidate == "arch_1", "the faster skeleton (0.150 ms) is the pinned candidate"
    assert _bindings(ctx)["{CANDIDATE}"] == "arch_1"


def test_no_probe_leaves_the_candidate_unpinned(tmp_path):
    """A milestone with no ARCH_PROBE_REQUEST lines: no probes, no pin — the
    binding renders the historical fallback text."""
    from test_loop_driver import _MILESTONE
    ms_text = _MILESTONE
    be = _E2EFake(agent_outputs=["authored", REC("D1.1", 1.8)],
                  gpu_logs=[EVAL(1.8), NCU],
                  milestone_path=tmp_path / "distilled.md", patch_writes=[ms_text])
    ctx, ms = _distill_ctx(tmp_path, be, ms_text=ms_text)
    ctx.m0_probe_rounds = 2
    summary = run_two_layer(ctx, max_meta=1)
    assert summary["stop"] == "target"
    assert ctx.candidate is None
    assert "not pinned" in _bindings(ctx)["{CANDIDATE}"]


# --- C5 -------------------------------------------------------------------------------

def _confirmed_gap_record(best_ms, ratio, eq=1.2) -> dict:
    return {"ts": 0, "best_ms": best_ms, "best_sha": "sha0",
            "gap_vs_absolute_target": 0.05, "estimator": "paired-in-job",
            "equivalence_ratio": eq, "median_ratio": ratio, "cluster": "",
            "student_jobs": ["job1", "job2"], "verdict": "confirmed_gap",
            "paired_residual": round(eq / ratio - 1.0, 4)}


def test_reanchor_memo_rebuilds_from_the_audit_trail_after_a_restart(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    (tmp_path / "reanchor.jsonl").write_text(
        json.dumps(_confirmed_gap_record(1.85, 1.05)) + "\n", encoding="utf-8")
    # a fresh process holds no memo: the accessor rebuilds it once
    assert ctx.reanchor_memo is None and not ctx.reanchor_memo_loaded
    memo = _reanchor_memo(ctx)
    assert memo is not None and memo[0] == 1.85
    assert "+14.3%" in memo[1], "the note carries the recomputed residual (1.2/1.05 - 1)"
    assert "1.05" in memo[1] and "1.2" in memo[1]
    assert ctx.reanchor_memo == memo, "rebuilt into the ctx (no re-read)"
    assert not any(c[0] == "gpu" for c in be.calls), "no ref job was bought"


def test_rebuilt_memo_prevents_the_rebuy(tmp_path):
    """The point of C5: a same-best re-check after the restart reuses the verdict
    instead of re-buying the ref measurement."""
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    (tmp_path / "reanchor.jsonl").write_text(
        json.dumps(_confirmed_gap_record(1.99, 1.05)) + "\n", encoding="utf-8")
    ctx.reanchor_ref = lambda: (_ for _ in ()).throw(
        AssertionError("a rebuilt memo must answer before any ref job"))
    ctx.target_anchor_ms = 2.0
    ctx.equivalence_ratio = 1.2
    # best 1.99 vs target 1.9: a +4.7% residual, inside the 6% phantom-gap band
    assert _phantom_gap_reanchor(ctx, best=1.99) is None
    assert ctx.reanchor_note and "reanchored" in ctx.reanchor_note
    assert not any(c[0] == "gpu" for c in be.calls)


def test_a_session_memo_wins_over_the_file(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    (tmp_path / "reanchor.jsonl").write_text(
        json.dumps(_confirmed_gap_record(1.85, 1.05)) + "\n", encoding="utf-8")
    ctx.reanchor_memo = (2.0, "session note")
    assert _reanchor_memo(ctx) == (2.0, "session note")


def test_no_file_no_memo(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    assert _reanchor_memo(ctx) is None
    assert ctx.reanchor_memo_loaded, "the (failed) read does not repeat"
